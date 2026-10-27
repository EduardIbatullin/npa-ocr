"""Таблицы страницы через DeepSeek-OCR (vLLM на GPU) со структурой по линиям скана —
замена медленного пути TATR + PaddleOCR/RapidOCR на CPU (3–9 мин на табличную страницу
против ~7 с). Перенесено из спайка npa-vlm-ocr (scripts/spike.py, table_grid.py; README и
docs/integration-plan.md там же — разметка 117 таблиц 14 документов: 113 верно, 4 с мелкими
ошибками).

DeepSeek-OCR по всей странице отдаёт блоки «<|ref|>table<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>»
с HTML таблицы. Текст ячеек она читает дословно, но объединённые ячейки отдаёт пустыми клетками,
текст объединения ставит туда, где он виден (посередине группы), а иногда теряет или
добавляет клетку. Таблицы НПА разлинованы, поэтому сетку берём из самого скана:

1. линии — морфологией (длинные горизонтальные/вертикальные штрихи), ячейки — связные области
   между линиями; объединённая ячейка получается одной областью сама;
2. границы строк/столбцов — кластеры краёв ячеек, отсюда rowspan/colspan;
3. текст: Tesseract (одним вызовом, слова по координатам) даёт черновой текст каждой ячейки,
   по нему ячейки DeepSeek выравниваются на ячейки сетки (жадно по похожести и положению +
   добор); в таблицу идёт текст DeepSeek, Tesseract — только для сопоставления.

Если сетка не построилась или не вместила текст DeepSeek (таблица без линий в теле,
Постановление №1430) — берётся HTML DeepSeek как есть. Ячейки, где текст есть только от
Tesseract или поставлен лишь по положению, помечаются низкой уверенностью (low_lines) —
DOCX подсвечивает их жёлтым. Такие ячейки (и «чернила» без текста, и текст DeepSeek, не похожий
на Tesseract) дочитываются по кропу ячейки — vlm_reread.py; принятый текст пометку снимает.

Сетевой адрес vLLM — переменная окружения VLLM_URL (OpenAI-совместимый /v1), модель —
VLLM_MODEL (served-model-name в vLLM). В сервисе это адрес шлюза vlm_gate.py: он поднимает
остановленный vLLM по первому запросу (холодный старт ~2–3 мин) и останавливает его после
простоя — GPU на сервере общая. Отсюда большой VLLM_TIMEOUT.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from PIL import Image

from npa_ocr.ocr_table import TableCell

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "deepseek-ocr")
# холодный старт vLLM через шлюз (до 10 мин ожидания) + сама страница (~7–60 с)
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "900"))

# Промпт режима «документ с координатами блоков» (рецепт DeepSeek-OCR)
DEEPSEEK_PROMPT = "<|grounding|>Convert the document to markdown."
# Длинная таблица на всю страницу — ~4000 токенов; от зацикливания страхует только этот предел
# (NGramPerReqLogitsProcessor из рецепта не совместим с vLLM 0.32)
MAX_TOKENS = 6144
# Координаты блоков DeepSeek — в долях 0..999 от размера страницы
_DS_SCALE = 999.0
# Сетка годна, если в ней не нашли места не более этой доли ячеек DeepSeek
_MAX_LOST_SHARE = 0.1


# --- DeepSeek-OCR в vLLM ---


class VlmUnavailable(RuntimeError):
    """vLLM (или шлюз перед ним) не ответил: не поднят, не хватило памяти GPU, таймаут."""


def deepseek_chat(image: Image.Image) -> str:
    """Ответ DeepSeek-OCR на страницу. skip_special_tokens=False обязателен: <tr>/<td> у
    DeepSeek-OCR — спецтокены, без него структура таблицы теряется (так было в Ollama)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": DEEPSEEK_PROMPT},
        ]}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
    }
    req = urllib.request.Request(
        f"{VLLM_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=VLLM_TIMEOUT) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"].get("content") or ""
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise VlmUnavailable(f"vLLM ({VLLM_URL}) недоступен: {exc}") from exc


# Ollama вырезает служебные токены, и метка приходит как «table[[x1, y1, x2, y2]]» — поддерживаем оба вида
_DS_BLOCK_RE = re.compile(
    r"(?:<\|ref\|>(\w+)<\|/ref\|><\|det\|>|^(\w+))(\[\[[\d,\s\[\]]*?\]\])(?:<\|/det\|>)?", re.S | re.M)


@dataclass
class DsTable:
    bbox: list[int]  # пиксели страницы
    html: str


def parse_deepseek(raw: str, width: int, height: int) -> list[DsTable]:
    """Таблицы из ответа DeepSeek-OCR. HTML может стоять не сразу за меткой table: между ними
    бывает блок table_caption (Приказ №228, стр. 23) — берём текст до следующей метки table."""
    matches = list(_DS_BLOCK_RE.finditer(raw))

    def label(m: re.Match) -> str:
        return (m.group(1) or m.group(2) or "").lower()

    out = []
    for i, m in enumerate(matches):
        if label(m) != "table":
            continue
        nxt = next((n.start() for n in matches[i + 1:] if label(n) == "table"), len(raw))
        try:
            x1, y1, x2, y2 = json.loads(m.group(3))[0]
        except (json.JSONDecodeError, IndexError, TypeError, ValueError):
            continue
        html = re.search(r"<table.*?</table>", raw[m.end():nxt], flags=re.S | re.I)
        if not html:
            continue
        box = [round(x1 / _DS_SCALE * width), round(y1 / _DS_SCALE * height),
               round(x2 / _DS_SCALE * width), round(y2 / _DS_SCALE * height)]
        out.append(DsTable(box, html.group(0)))
    return out


# --- Тексты: нормализация и похожесть ---

_SCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉ᵢ", "01234567890123456789i")
_DIGITS = re.compile(r"\d+")
# латиница, похожая на кириллицу (после lower в norm): DeepSeek пишет «C4-C6» латиницей, Tesseract
# (rus) — кириллицей; для сопоставления это одно и то же
_HOMO = str.maketrans("abcehkmoptxy", "авсенкмортху")


def norm(s: str) -> str:
    """Убираем различия оформления, не смысла: пробелы (≤60 / ≤ 60), над/подстрочные цифры
    (мг/нм³ / мг/нм3), виды тире, ё. Остаются настоящие расхождения букв и чисел."""
    s = s.translate(_SCRIPTS).replace("Ӏ", "I").replace("ё", "е").replace("_", "")
    s = re.sub(r"[–—−]", "-", s)
    return re.sub(r"\s+", "", s).lower()


def sim(a: str, b: str) -> float:
    """Похожесть текстов ячеек; цифры должны в основном совпасть: «≤ 60» и «≤ 240» — разные ячейки.
    Сверка мягкая: Tesseract читает «≤ 0,35» как «50,35», «С6» как «Сб», «I» как «1»."""
    a, b = norm(a).translate(_HOMO), norm(b).translate(_HOMO)
    if not a or not b:
        return 0.0
    s = SequenceMatcher(None, a, b, autojunk=False).ratio()
    da, db = "".join(_DIGITS.findall(a)), "".join(_DIGITS.findall(b))
    if da and db and SequenceMatcher(None, da, db, autojunk=False).ratio() < 0.75:
        s *= 0.3
    return s


def _contained(part: str, whole: str) -> float:
    """Доля символов part, найденных в whole по порядку."""
    a, b = norm(part).translate(_HOMO), norm(whole).translate(_HOMO)
    if not a or not b:
        return 0.0
    return sum(x.size for x in SequenceMatcher(None, b, a, autojunk=False).get_matching_blocks()) / len(a)


def _letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def cell_text(html: str) -> str:
    """HTML ячейки -> текст для DOCX: <br> — перенос строки, цифры в <sup>/<sub> — надстрочные и
    подстрочные символы («мг/нм³», «80¹»), остальная разметка убирается."""
    h = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)

    def script(m: re.Match) -> str:
        inner = re.sub(r"<[^>]+>", "", m.group(2))
        table = _SUP if m.group(1).lower() == "sup" else _SUB
        return inner.translate(table) if re.fullmatch(r"[\d+-]+", inner) else inner

    h = re.sub(r"<(sup|sub)>(.*?)</\1>", script, h, flags=re.S | re.I)
    h = unescape(re.sub(r"<[^>]+>", " ", h))
    return "\n".join(" ".join(line.split()) for line in h.split("\n") if line.strip())


# --- 1–2. Сетка по линиям ---


@dataclass
class _Cell:
    x0: int
    y0: int
    x1: int
    y1: int
    r: int = 0
    c: int = 0
    rs: int = 1
    cs: int = 1
    tess: str = ""
    text: str = ""          # итоговый HTML ячейки (DeepSeek)
    src: str = ""           # ds | pos (по положению) | tess (только Tesseract) | ""
    text_y: int = -1        # где в ячейке начинается текст (по словам Tesseract)
    ink: bool = False       # в ячейке что-то напечатано (даже если Tesseract не прочёл)
    ds_ids: list[int] = field(default_factory=list)
    rot: int = 0            # поворот, при котором Tesseract прочёл вертикальный текст


def _line_masks(gray):
    import cv2

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape
    hk = max(40, w // 25)
    vk = max(40, h // 40)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    # закрыть разрывы линий в скане и чуть утолщить, чтобы ячейки не «протекали» друг в друга
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)))
    return horiz, vert


def _cluster(values: list[int], tol: int) -> list[int]:
    import numpy as np

    out: list[list[int]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(np.median(g)) for g in out]


def _detect_cells(img: Image.Image) -> tuple[list[_Cell], dict]:
    import cv2
    import numpy as np

    gray = np.array(img.convert("L"))
    horiz, vert = _line_masks(gray)
    lines = cv2.bitwise_or(horiz, vert)
    ys, xs = np.nonzero(lines)
    info: dict = {"ok": False}
    if len(xs) == 0 or not vert.any() or not horiz.any():
        info["why"] = "мало линий"
        return [], info
    # рамка по крайним точкам всех линий: в НПА у таблиц часто нет боковых рамок (1430), а на стыке
    # страниц — верхней/нижней
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cv2.rectangle(lines, (x_min, y_min), (x_max, y_max), 255, 3)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(lines), connectivity=4)
    cells = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x <= x_min or y <= y_min or x + w >= x_max + 1 or y + h >= y_max + 1:
            continue  # фон снаружи рамки
        if w < 25 or h < 25 or area < 0.5 * w * h:
            continue  # обрывки линий, полоски от отступа кропа (строка текста при 200 DPI ≈ 35 px)
        cells.append(_Cell(int(x), int(y), int(x + w), int(y + h)))
    if not cells:
        info["why"] = "ячеек нет"
        return [], info
    tol = 14
    xe = _cluster([c.x0 for c in cells] + [c.x1 for c in cells], tol)
    ye = _cluster([c.y0 for c in cells] + [c.y1 for c in cells], tol)
    # край ячейки и край соседней разделены толщиной линии -> одна граница = пара близких краёв
    xe = _cluster(xe, 2 * tol)
    ye = _cluster(ye, 2 * tol)

    def idx(edges: list[int], v: int) -> int:
        return int(np.argmin([abs(e - v) for e in edges]))

    taken: dict[tuple[int, int], int] = {}
    overlap = 0
    for k, c in enumerate(cells):
        c.c, c.r = idx(xe, c.x0), idx(ye, c.y0)
        c.cs = max(1, idx(xe, c.x1) - c.c)
        c.rs = max(1, idx(ye, c.y1) - c.r)
        for dr in range(c.rs):
            for dc in range(c.cs):
                overlap += (c.r + dr, c.c + dc) in taken
                taken[(c.r + dr, c.c + dc)] = k
    n_rows, n_cols = len(ye) - 1, len(xe) - 1
    holes = n_rows * n_cols - len(taken)
    info.update(ok=overlap == 0 and holes == 0, n_rows=n_rows, n_cols=n_cols, overlap=overlap, holes=holes, ye=ye)
    cells.sort(key=lambda c: (c.r, c.c))
    return cells, info


# --- 3. Текст ---


def _tesseract_into_cells(img: Image.Image, cells: list[_Cell]) -> None:
    import cv2
    import numpy as np
    import pytesseract

    from npa_ocr.ocr import _resolve_tesseract_cmd  # здесь, а не наверху: ocr импортирует этот модуль

    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    gray = np.array(img.convert("L"))
    horiz, vert = _line_masks(gray)
    clean = gray.copy()
    clean[cv2.bitwise_or(horiz, vert) > 0] = 255  # линии мешают Tesseract
    d = pytesseract.image_to_data(Image.fromarray(clean), lang="rus", config="--psm 11",
                                  output_type=pytesseract.Output.DICT)
    for c in cells:
        inner = clean[c.y0 + 6:c.y1 - 6, c.x0 + 6:c.x1 - 6]
        c.ink = inner.size > 0 and (inner < 128).mean() > 0.002
    words: dict[int, list[tuple[int, int, str]]] = {}
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < 0:
            continue
        cx, cy = d["left"][i] + d["width"][i] // 2, d["top"][i] + d["height"][i] // 2
        for k, c in enumerate(cells):
            if c.x0 <= cx < c.x1 and c.y0 <= cy < c.y1:
                words.setdefault(k, []).append((d["top"][i], d["left"][i], t))
                break
    for k, ws in words.items():
        # строки: слова с близким top, внутри строки — слева направо
        ws.sort()
        lines: list[list[tuple[int, int, str]]] = []
        for w in ws:
            if lines and abs(w[0] - lines[-1][0][0]) < 15:
                lines[-1].append(w)
            else:
                lines.append([w])
        cells[k].tess = " ".join(" ".join(x[2] for x in sorted(line, key=lambda z: z[1])) for line in lines)
        cells[k].text_y = ws[0][0]
    # вертикальный текст (шапки альбомных таблиц, 228): Tesseract его не видит — читаем ячейку повёрнутой
    pil = Image.fromarray(clean)
    for c in cells:
        if c.y1 - c.y0 > 1.5 * (c.x1 - c.x0) and c.y1 - c.y0 > 150 and _letters(c.tess) < 4:
            part = pil.crop((c.x0, c.y0, c.x1, c.y1))
            # направление (снизу вверх или сверху вниз) — по уверенности Tesseract, а не по числу букв:
            # перевёрнутый текст тоже «читается» буквами, но с низкой уверенностью
            best, best_score, best_a = "", 0.0, 0
            for a in (-90, 90):
                d = pytesseract.image_to_data(part.rotate(a, expand=True), lang="rus", config="--psm 6",
                                              output_type=pytesseract.Output.DICT)
                ws = [(w, float(cf)) for w, cf in zip(d["text"], d["conf"]) if w.strip() and float(cf) >= 0]
                score = sum(cf * _letters(w) for w, cf in ws)
                if score > best_score:
                    best, best_score, best_a = " ".join(w for w, _ in ws), score, a
            if _letters(best) >= 4:
                c.tess, c.rot = best, best_a
                c.text_y = c.y0


class _Rows(HTMLParser):
    """HTML-таблица -> строки ячеек (text, rowspan, colspan)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = []
            self._span = (_span(a.get("rowspan")), _span(a.get("colspan")))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(("".join(self._cell), *self._span))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _span(v: str | None) -> int:
    try:
        return max(1, int(v or 1))
    except ValueError:
        return 1


@dataclass
class _DsCell:
    text: str
    r: float  # центр по строкам в сетке DeepSeek
    c: int


def _ds_cells(html: str) -> list[_DsCell]:
    p = _Rows()
    p.feed(html)
    taken: set[tuple[int, int]] = set()
    out = []
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs in row:
            while (r, c) in taken:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            if text.strip():
                out.append(_DsCell(" ".join(text.split()), r + (rs - 1) / 2, c))
            c += cs
    return out


def _cell_html(html: str) -> list[str]:
    """Тексты ячеек DeepSeek с сохранением <sup>/<sub>/<br>."""
    return [m for m in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, re.S) if m.strip()]


def _numbering(cells: list[_Cell], ds: list[_DsCell], used_c: set[int], used_d: set[int]) -> None:
    """Строка нумерации столбцов («1 2 … 14»): одиночные цифры похожим текстом не сопоставить
    (227 стр. 10: «11» уехало, «9» пропало), а DeepSeek читает её верно. Ставим её по порядку в ту
    строку сетки, где столько же ячеек и Tesseract видит в них короткие тексты. По порядку, а не по
    номеру столбца: у DeepSeek столбцов бывает больше (228 стр. 25: выдуманные столбцы шапки)."""
    by_row: dict[int, list[int]] = {}
    for j, d in enumerate(ds):
        if d.r == int(d.r):
            by_row.setdefault(int(d.r), []).append(j)
    for js in by_row.values():
        js.sort(key=lambda j: ds[j].c)
        nums = [int(ds[j].text) for j in js if re.fullmatch(r"\d{1,2}", ds[j].text)]
        if len(nums) < 3 or len(nums) < len(js) - 1 or any(b != a + 1 for a, b in zip(nums, nums[1:])):
            continue
        best, best_ks = 0, []
        for gr in sorted({c.r for c in cells}):
            ks = sorted((k for k, c in enumerate(cells) if c.r == gr and c.rs == 1), key=lambda k: cells[k].c)
            if len(ks) != len(js) or any(k in used_c for k in ks):
                continue
            exact = sum(norm(cells[k].tess) == norm(ds[j].text) for k, j in zip(ks, js))
            short = all(len(norm(cells[k].tess)) <= 4 for k in ks)
            if short and exact >= 2 and exact > best:
                best, best_ks = exact, ks
        for k, j in zip(best_ks, js):
            cells[k].ds_ids.append(j)
            used_c.add(k)
            used_d.add(j)


_VALUE = re.compile(r"[≤<>≥]?\s*\d+(?:[.,]\d+)?(?:\s*\([^)]*\))?")


def _split_values(cells: list[_Cell], ds: list[_DsCell]) -> int:
    """DeepSeek слила столбец значений в одну ячейку («≤ 4,1139 ≤ 0,1440 … ≤ 0,0023», 377 стр. 7),
    а соседние ячейки того же столбца остались с текстом одного Tesseract. Раскладываем значения
    по порядку в окно из подряд идущих ячеек столбца, если цифры каждого совпадают с Tesseract."""
    n = 0
    for c in cells:
        if len(c.ds_ids) != 1:
            continue
        text = ds[c.ds_ids[0]].text
        vals = [m.group().strip() for m in _VALUE.finditer(text)]
        if len(vals) < 2 or len("".join(vals).replace(" ", "")) < 0.9 * len(text.replace(" ", "")):
            continue
        col = sorted((x for x in cells if x.c == c.c and x.cs == c.cs and (x is c or x.src == "tess")),
                     key=lambda x: x.r)
        i = col.index(c)
        best, best_w = 0.0, None
        for s in range(max(0, i - len(vals) + 1), i + 1):
            w = col[s:s + len(vals)]
            if len(w) < len(vals) or any(b.r != a.r + a.rs for a, b in zip(w, w[1:])):
                continue
            score = 0.0
            for x, v in zip(w, vals):
                dx, dv = "".join(_DIGITS.findall(x.tess)), "".join(_DIGITS.findall(v))
                # «≤ 9» Tesseract читает как «59» — первая «5» может быть знаком
                ok = max(SequenceMatcher(None, d, dv, autojunk=False).ratio()
                         for d in ([dx, dx[1:]] if dx[:1] == "5" else [dx])) >= 0.75 if dx else x is c
                if not ok:
                    break
                score += sim(x.tess, v) if x.tess else 0.5
            else:
                if score > best:
                    best, best_w = score, w
        if best_w is None:
            continue
        for x, v in zip(best_w, vals):
            x.text, x.src = escape(v), "ds"
        n += 1
    return n


def _align(cells: list[_Cell], ds: list[_DsCell], row_edges: list[int]) -> None:
    """Жадное сопоставление ячеек DeepSeek ячейкам сетки: похожесть текста (DeepSeek против чернового
    Tesseract) минус штраф за разницу положения — строка, где виден текст, в долях высоты таблицы и
    столбец. Положение различает одинаковые тексты («мг/нм³» в каждой строке); текст объединённой
    ячейки DeepSeek ставит туда, где он виден на скане, поэтому строку ячейки сетки берём по словам."""
    import numpy as np

    n_rows_g = max(len(row_edges) - 1, 1)
    n_rows_d = max((int(d.r) for d in ds), default=0) + 1
    n_cols_g = max((c.c + c.cs for c in cells), default=1)
    n_cols_d = max((d.c for d in ds), default=0) + 1

    def g_row(c: _Cell) -> float:
        if c.text_y < 0:
            return c.r + (c.rs - 1) / 2
        r = int(np.searchsorted(row_edges, c.text_y, side="right")) - 1
        return float(min(max(r, c.r), c.r + c.rs - 1))

    used_c: set[int] = set()
    used_d: set[int] = set()
    _numbering(cells, ds, used_c, used_d)
    pairs = []
    for k, c in enumerate(cells):
        if not c.tess or k in used_c:
            continue
        yg, xg = g_row(c) / n_rows_g, (c.c + (c.cs - 1) / 2) / n_cols_g
        for j, d in enumerate(ds):
            s = sim(c.tess, d.text)
            if s < 0.35:
                continue
            if len(norm(d.text)) <= 3 and abs(yg - d.r / n_rows_d) > 1.5 / n_rows_g:
                continue  # «5», «1», «А» есть в любой таблице — короткий текст только рядом со своим местом
            pairs.append((s - 0.6 * abs(yg - d.r / n_rows_d) - 0.3 * abs(xg - d.c / n_cols_d), k, j))
    for _, k, j in sorted(pairs, reverse=True):
        if k in used_c or j in used_d:
            continue
        cells[k].ds_ids.append(j)
        used_c.add(k)
        used_d.add(j)
    # добор 1: DeepSeek разбил одну ячейку на несколько — кусок добавляем к ячейке, только если
    # вместе с ним текст заметно ближе к тому, что видит Tesseract
    for j, d in enumerate(ds):
        if j in used_d:
            continue
        best, bk = 0.05, -1
        for k, c in enumerate(cells):
            if not c.tess:
                continue
            if not c.ds_ids:
                # пустая ячейка: кусок должен почти целиком читаться в её тексте (248 стр. 9 — одна
                # ячейка из трёх абзацев, DeepSeek разбила её на три строки)
                gain = sim(c.tess, d.text) if _contained(d.text, c.tess) >= 0.85 else 0.0
            else:
                cur = " ".join(ds[i].text for i in sorted(c.ds_ids, key=lambda i: (ds[i].r, ds[i].c)))
                both = " ".join(ds[i].text for i in sorted(c.ds_ids + [j], key=lambda i: (ds[i].r, ds[i].c)))
                gain = sim(c.tess, both) - sim(c.tess, cur)
            if gain > best:
                best, bk = gain, k
        if bk >= 0:
            cells[bk].ds_ids.append(j)
            used_d.add(j)
    # добор 2: ячейка, которую Tesseract не прочёл («≤ 180» -> «АЗ — со >», одиночные цифры в строке
    # нумерации столбцов) — по позиции. Соответствие строк и столбцов DeepSeek -> сетка берём у уже
    # сопоставленных ячеек; строку без опор — от ближайших опорных строк (сдвиг сохраняется).
    # Ячейка сетки должна быть пустой и с «чернилами»: на пустом месте скана текст DeepSeek выдуман.
    where: dict[tuple[int, int], int] = {}
    for k, c in enumerate(cells):
        for dr in range(c.rs):
            for dc in range(c.cs):
                where[(c.r + dr, c.c + dc)] = k
    placed = {j: k for k, c in enumerate(cells) for j in c.ds_ids}
    row_votes: dict[int, list[int]] = {}
    col_votes: dict[int, list[int]] = {}
    for j, k in placed.items():
        if len(norm(ds[j].text)) <= 3:
            continue  # короткие тексты — ненадёжные опоры
        if ds[j].r == int(ds[j].r):  # центр объединения DeepSeek в опоры не берём
            row_votes.setdefault(int(ds[j].r), []).append(int(g_row(cells[k])))
        col_votes.setdefault(ds[j].c, []).append(cells[k].c)

    def med(v: list[int]) -> int:
        return sorted(v)[len(v) // 2]

    row_map = {r: med(v) for r, v in row_votes.items()}
    col_map = {c: med(v) for c, v in col_votes.items()}
    shift = med([g - c for c, g in col_map.items()]) if col_map else 0
    anchors = sorted(row_map)

    def map_row(r: int) -> int | None:
        if r in row_map:
            return row_map[r]
        above = [a for a in anchors if a < r]
        below = [b for b in anchors if b > r]
        if above and below:
            a, b = above[-1], below[0]
            if row_map[b] - row_map[a] == b - a:  # между опорами строки идут один в один
                return row_map[a] + (r - a)
            return None
        if above:
            return row_map[above[-1]] + (r - above[-1])
        if below:
            return row_map[below[0]] - (below[0] - r)
        return None

    for j, d in enumerate(ds):
        if j in used_d or d.r != int(d.r):
            continue
        gr = map_row(int(d.r))
        if gr is None:
            continue
        k = where.get((gr, col_map.get(d.c, d.c + shift)))
        if k is not None and not cells[k].ds_ids and cells[k].ink:
            cells[k].ds_ids.append(j)
            cells[k].src = "pos"
            used_d.add(j)


@dataclass
class GridResult:
    ok: bool                      # сетка построена
    cells: list[TableCell]
    n_rows: int
    n_cols: int
    info: dict                    # ds_cells, ds_lost, ds_invented, tess_only, by_pos, split, why
    # сомнительные ячейки для дочитки (vlm_reread.py): r, c, box (в координатах кропа), rot, tess,
    # left (соседняя слева ячейка с текстом DeepSeek: box, text), kind (tess|empty|mismatch), ds
    doubt: list[dict] = field(default_factory=list)


def _plain(h: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", h).split())


def _doubt_kind(c: _Cell) -> str | None:
    """Сомнительная ячейка: текст только от Tesseract, «чернила» без текста или текст DeepSeek, не
    похожий на то, что видит Tesseract (228 стр. 25: вертикальную шапку DeepSeek прочла плохо — «для
    управления» вместо «для утилизации»)."""
    if c.src == "tess":
        return "tess"
    if c.ink and not c.text:
        return "empty"
    # вертикальный текст (Tesseract прочёл его с поворотом) DeepSeek читает заметно хуже: «для
    # обращения» вместо «для обработки» — там проверяем любое расхождение
    if c.src in ("ds", "pos") and _letters(c.tess) >= 4 and sim(c.tess, _plain(c.text)) < (0.95 if c.rot else 0.5):
        return "mismatch"
    return None


def _doubt(cells: list[_Cell]) -> list[dict]:
    def left_of(c: _Cell) -> dict | None:
        """Соседняя слева ячейка с текстом DeepSeek — контекст для дочитки."""
        k = next((x for x in cells if x.c + x.cs == c.c and x.r <= c.r < x.r + x.rs and x.src in ("ds", "pos")),
                 None)
        return {"box": [k.x0, k.y0, k.x1, k.y1], "text": _plain(k.text)} if k else None

    out = []
    for c in cells:
        kind = _doubt_kind(c)
        if kind:
            out.append({"r": c.r, "c": c.c, "box": [c.x0, c.y0, c.x1, c.y1], "rot": c.rot, "tess": c.tess,
                        "left": left_of(c), "kind": kind, "ds": _plain(c.text) if kind == "mismatch" else ""})
    return out


def rebuild(img: Image.Image, ds_html: str) -> GridResult:
    """Сетка по линиям кропа таблицы + текст DeepSeek в её ячейках."""
    cells, info = _detect_cells(img)
    ds = _ds_cells(ds_html)
    if not info["ok"]:
        info.update(ds_cells=len(ds), ds_lost=len(ds))
        return GridResult(False, [], 0, 0, info)
    _tesseract_into_cells(img, cells)
    rich = _cell_html(ds_html)
    if len(rich) != len(ds):
        rich = [escape(d.text) for d in ds]
    _align(cells, ds, info["ye"])
    for c in cells:
        if c.ds_ids:
            ids = sorted(c.ds_ids, key=lambda j: (ds[j].r, ds[j].c))
            c.text = "<br>".join(rich[j] for j in ids)
            c.src = c.src or "ds"
        elif c.tess:
            c.text, c.src = escape(c.tess), "tess"
    n_split = _split_values(cells, ds)
    doubt = _doubt(cells)
    lost = [ds[j].text for j in range(len(ds)) if not any(j in c.ds_ids for c in cells)]
    # повтор уже размещённого текста (DeepSeek дублирует «кг/т переработанного сырья» в выдуманных
    # строках) — не потеря; потеря — текст, которого в итоговой таблице нет вовсе
    placed_norm = {norm(ds[j].text) for c in cells for j in c.ds_ids}
    lost_unique = [t for t in lost if norm(t) not in placed_norm]
    # текст DeepSeek, которого Tesseract на скане не видит вовсе, — выдуман (228 стр. 25: «Экологическая
    # безопасность» в шапке); потеря — только то, что на скане есть, а в сетку не попало
    all_tess = " ".join(c.tess for c in cells)
    invented = [t for t in lost_unique if len(norm(t)) > 3 and _contained(t, all_tess) < 0.7]
    lost_unique = [t for t in lost_unique if t not in invented]
    info.update(ds_cells=len(ds), ds_lost=len(lost_unique), lost=lost_unique[:10], ds_invented=invented[:10],
                tess_only=sum(c.src == "tess" for c in cells), by_pos=sum(c.src == "pos" for c in cells),
                split=n_split)
    out = []
    for c in cells:
        text = cell_text(c.text)
        low = c.src in ("tess", "pos")
        out.append(TableCell(c.r, c.c, c.rs, c.cs, text,
                             (True,) * len(text.split("\n")) if low and text else ()))
    return GridResult(True, out, info["n_rows"], info["n_cols"], info, doubt)


# (кроп таблицы, сомнительные ячейки GridResult.doubt) -> принятый текст по (строка, столбец)
RereadFn = Callable[[Image.Image, list[dict]], dict[tuple[int, int], str]]


def apply_reread(cells: list[TableCell], texts: dict[tuple[int, int], str]) -> list[TableCell]:
    """Дочитанный текст — в ячейки; он проверен (vlm_reread.accept), поэтому без жёлтой пометки."""
    out = []
    for c in cells:
        t = texts.get((c.row, c.col))
        out.append(c if t is None else TableCell(c.row, c.col, c.row_span, c.col_span, cell_text(escape(t))))
    return out


def html_cells(html: str) -> tuple[list[TableCell], int, int]:
    """HTML DeepSeek как есть -> ячейки (запасной путь, когда сетка по линиям не годится)."""
    cells: list[TableCell] = []
    taken: set[tuple[int, int]] = set()
    for r, row in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)):
        c = 0
        for attrs, inner in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S | re.I):
            while (r, c) in taken:
                c += 1
            rs = _span((re.search(r'rowspan="?(\d+)', attrs) or [0, None])[1])
            cs = _span((re.search(r'colspan="?(\d+)', attrs) or [0, None])[1])
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            cells.append(TableCell(r, c, rs, cs, cell_text(inner)))
            c += cs
    n_rows = max((x.row + x.row_span for x in cells), default=0)
    n_cols = max((x.col + x.col_span for x in cells), default=0)
    return cells, n_rows, n_cols


# --- Таблицы страницы ---


def refine_bbox(page: Image.Image, box: list[int], pad: int = 12) -> list[int]:
    """Рамка таблицы по разлиновке страницы: DeepSeek иногда даёт рамку только вокруг шапки
    (1430 стр. 8), а HTML — всей таблицы. Берём связные куски линий, задевающие её рамку."""
    import cv2
    import numpy as np

    gray = np.array(page.convert("L"))
    horiz, vert = _line_masks(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_or(horiz, vert), connectivity=8)
    x0, y0, x1, y1 = box
    ux0, uy0, ux1, uy1 = x0, y0, x1, y1
    # рамка DeepSeek с запасом: кусок разлиновки берём, только если он в основном внутри неё —
    # иначе рамку «съедают» тёмные поля скана и соседние таблицы
    mx, my = page.width * 0.05, page.height * 0.05
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if max(w, h) < 100 or x <= 5 or y <= 5 or x + w >= page.width - 5 or y + h >= page.height - 5:
            continue
        ix = max(0, min(x + w, x1 + mx) - max(x, x0 - mx))
        iy = max(0, min(y + h, y1 + my) - max(y, y0 - my))
        touches = x < x1 and x + w > x0 and y < y1 and y + h > y0
        if touches and ix * iy >= 0.8 * w * h:
            ux0, uy0, ux1, uy1 = min(ux0, x), min(uy0, y), max(ux1, x + w), max(uy1, y + h)
    return [int(max(0, ux0 - pad)), int(max(0, uy0 - pad)), int(min(page.width, ux1 + pad)),
            int(min(page.height, uy1 + pad))]


def _overlap(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(small, 1)


@dataclass
class VlmTable:
    """Таблица страницы и как она получена (для отчёта о прогоне и дочитки ячеек)."""

    bbox: list[int]
    cells: list[TableCell]
    n_rows: int
    n_cols: int
    src: str                      # grid — сетка по линиям, deepseek — HTML DeepSeek как есть
    html: str                     # HTML DeepSeek (склеенный, если таблиц в одной разлиновке несколько)
    info: dict = field(default_factory=dict)


def page_tables(page: Image.Image, tables: list[DsTable], reread_fn: RereadFn | None = None) -> list[VlmTable]:
    """Таблицы DeepSeek страницы -> итоговые таблицы. Таблицы DeepSeek, попавшие в одну разлиновку,
    — одна таблица (1430 стр. 8: разрезана на две по строке-подзаголовку «II. При сбросе…»).
    reread_fn — дочитка сомнительных ячеек таблиц по сетке (vlm_reread.reread_cells)."""
    boxes = [refine_bbox(page, t.bbox) for t in tables]
    groups: list[list[int]] = []
    for k, b in enumerate(boxes):
        g = next((g for g in groups if any(_overlap(b, boxes[i]) > 0.5 for i in g)), None)
        if g is None:
            groups.append([k])
        else:
            g.append(k)
    out = []
    for g in groups:
        box = [min(boxes[i][0] for i in g), min(boxes[i][1] for i in g),
               max(boxes[i][2] for i in g), max(boxes[i][3] for i in g)]
        html = "<table>" + "".join(re.sub(r"</?table[^>]*>", "", tables[i].html) for i in g) + "</table>"
        crop = page.crop(box)
        res = rebuild(crop, html)
        # итог: сетка по линиям, если она вместила текст DeepSeek; иначе (таблица без линий в теле,
        # как 1430 стр. 8, или сетка не построилась) — HTML DeepSeek как есть
        if res.ok and res.info["ds_lost"] <= _MAX_LOST_SHARE * max(res.info["ds_cells"], 1):
            cells = res.cells
            if reread_fn is not None and res.doubt:
                texts = reread_fn(crop, res.doubt)
                cells = apply_reread(cells, texts)
                res.info.update(doubt=len(res.doubt), by_vlm=len(texts))
            out.append(VlmTable(box, cells, res.n_rows, res.n_cols, "grid", html, res.info))
        else:
            cells, n_rows, n_cols = html_cells(html)
            out.append(VlmTable(box, cells, n_rows, n_cols, "deepseek", html, res.info))
    # рамка таблицы по HTML DeepSeek — только разлинованная шапка, тело без линий ниже неё; по
    # рамке из текста страницы вырезается область таблицы, иначе строки тела попадут в документ
    # дважды — и в таблицу, и обычным текстом
    out.sort(key=lambda t: t.bbox[1])
    for i, t in enumerate(out):
        if t.src == "deepseek":
            limit = out[i + 1].bbox[1] if i + 1 < len(out) else page.height
            t.bbox = extend_to_rows(page, t.bbox, t.cells, limit)
    return out


def extend_to_rows(page: Image.Image, box: list[int], cells: list[TableCell], limit_y: int,
                   pad: int = 12) -> list[int]:
    """Продлевает рамку вниз по строкам текста страницы, которые есть в таблице: строка Tesseract
    должна читаться (по порядку символов) в тексте строки таблицы — той же, что предыдущая, или
    ниже (строки таблицы идут по порядку). Первая чужая строка — конец таблицы. Порядок важен:
    1430 стр. 8 — подзаголовок «II. При сбросе … категории Б» вне этой таблицы почти дословно
    совпадает с её строкой «I. При сбросе … категории А», но та уже пройдена."""
    import pytesseract

    from npa_ocr.ocr import _resolve_tesseract_cmd

    x0, y0, x1, y1 = box
    if limit_y - y1 < 20:
        return box
    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    d = pytesseract.image_to_data(page.crop((x0, y1, x1, limit_y)), lang="rus", config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], list[int]] = {}
    for i, w in enumerate(d["text"]):
        if w.strip() and float(d["conf"][i]) >= 0:
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
    rows: dict[int, list[TableCell]] = {}
    for c in cells:
        rows.setdefault(c.row, []).append(c)
    row_texts = [" ".join(c.text for c in sorted(rows[r], key=lambda c: c.col)) for r in sorted(rows)]
    bottom, current = y1, 0
    for ids in sorted(lines.values(), key=lambda ids: min(d["top"][i] for i in ids)):
        text = " ".join(d["text"][i] for i in sorted(ids, key=lambda i: d["left"][i]))
        if len(norm(text)) < 2:
            continue
        found = next((k for k in range(current, len(row_texts)) if _contained(text, row_texts[k]) >= 0.8), None)
        if found is None:
            break
        current = found
        bottom = y1 + max(d["top"][i] + d["height"][i] for i in ids)
    return [x0, y0, x1, min(page.height, bottom + pad) if bottom > y1 else y1]


def run_vlm_tables(
    image_path: Path,
    *,
    chat_fn: Callable[[Image.Image], str] = deepseek_chat,
    reread_fn: RereadFn | None = None,
) -> list[VlmTable]:
    """Таблицы страницы-скана: DeepSeek-OCR по всей странице + сетка по линиям (+ дочитка
    сомнительных ячеек, если задан reread_fn). chat_fn — инъекция модели (в тестах — сохранённые
    ответы DeepSeek)."""
    page = Image.open(image_path).convert("RGB")
    raw = chat_fn(page)
    return page_tables(page, parse_deepseek(raw, page.width, page.height), reread_fn)
