"""Таблицы страницы через DeepSeek-OCR (vLLM на GPU) со структурой по линиям скана.

Работает быстрее пути TATR + PaddleOCR/RapidOCR на CPU: секунды на табличную
страницу вместо минут.

DeepSeek-OCR отдаёт по странице блоки
``<|ref|>table<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`` с HTML таблицы. Текст
ячеек модель читает дословно, но объединённые ячейки отдаёт пустыми клетками,
ставит текст объединения туда, где он виден, и иногда теряет или добавляет
клетку. Таблицы актов разлинованы, поэтому сетка берётся из самого скана:

1. Линии выделяются морфологией (длинные горизонтальные и вертикальные
   штрихи), ячейки — связные области между ними; объединённая ячейка сразу
   получается одной областью.
2. Границы строк и столбцов — кластеры краёв ячеек; отсюда объединения.
3. Tesseract даёт черновой текст каждой ячейки; по нему ячейки DeepSeek
   сопоставляются ячейкам сетки (жадно по похожести и положению, затем
   добор). В таблицу идёт текст DeepSeek, Tesseract нужен только для
   сопоставления.

Если сетка не построилась или не вместила текст DeepSeek (например, у таблицы
нет линий в теле), берётся HTML DeepSeek как есть. Ячейки с текстом только от
Tesseract или поставленным по положению помечаются как неуверенные и
выделяются в DOCX жёлтым. Такие ячейки, а также пустые ячейки с напечатанным
текстом и ячейки, где DeepSeek и Tesseract расходятся, читаются повторно
(:mod:`npa_ocr.vlm_reread`); принятый текст пометку снимает.

Переменные окружения:
    VLLM_URL: Адрес OpenAI-совместимого API vLLM (``…/v1``), обычно шлюза
        :mod:`npa_ocr.vlm_gate`.
    VLLM_MODEL: Имя модели в vLLM (``served-model-name``).
    VLLM_TIMEOUT: Таймаут запроса, с. Велик, потому что шлюз может запускать
        остановленный vLLM несколько минут.
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
# холодный старт vLLM через шлюз (до 10 мин) и распознавание страницы (до минуты)
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "900"))

# Промпт режима «документ с координатами блоков» (рецепт DeepSeek-OCR)
DEEPSEEK_PROMPT = "<|grounding|>Convert the document to markdown."
# Таблица во всю страницу — около 4000 токенов. Предел страхует от зацикливания генерации:
# NGramPerReqLogitsProcessor из рецепта DeepSeek-OCR несовместим с используемой версией vLLM
MAX_TOKENS = 6144
# Координаты блоков DeepSeek — в долях 0..999 от размера страницы
_DS_SCALE = 999.0
# Сетка годна, если в ней не нашли места не более этой доли ячеек DeepSeek
_MAX_LOST_SHARE = 0.1


# --- DeepSeek-OCR в vLLM ---


class VlmUnavailable(RuntimeError):
    """vLLM или шлюз перед ним не ответил: не запущен, не хватило видеопамяти, таймаут."""


class VlmResponse(str):
    """Text-compatible response retaining generation status for coverage review."""

    def __new__(cls, text, *, finish_reason='', usage=None):
        result = super().__new__(cls, text)
        result.finish_reason = finish_reason
        result.usage = usage or {}
        return result


def deepseek_chat(image: Image.Image) -> str:
    """Распознаёт страницу DeepSeek-OCR в режиме блоков с координатами.

    ``skip_special_tokens=False`` обязателен: ``<tr>``/``<td>`` у DeepSeek-OCR —
    служебные токены, без них теряется структура таблиц.

    Args:
        image: Изображение страницы.

    Returns:
        Ответ модели: Markdown с блоками ``<|ref|>…<|det|>``.

    Raises:
        VlmUnavailable: vLLM недоступен или ответ не разобран.
    """
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
        choice = result["choices"][0]
        return VlmResponse(choice["message"].get("content") or "",
                           finish_reason=choice.get("finish_reason") or '',
                           usage=result.get('usage'))
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise VlmUnavailable(f"vLLM ({VLLM_URL}) недоступен: {exc}") from exc


# Без служебных токенов метка выглядит как «table[[x1, y1, x2, y2]]» — поддерживаются оба вида
_DS_BLOCK_RE = re.compile(
    r"(?:<\|ref\|>(\w+)<\|/ref\|><\|det\|>|^(\w+))(\[\[[\d,\s\[\]]*?\]\])(?:<\|/det\|>)?", re.S | re.M)


@dataclass
class DsTable:
    """Таблица из ответа DeepSeek-OCR: рамка в пикселях страницы и HTML."""

    bbox: list[int]  # пиксели страницы
    html: str
    ruled: bool = False


def parse_deepseek(raw: str, width: int, height: int) -> list[DsTable]:
    """Извлекает таблицы из ответа DeepSeek-OCR.

    HTML может стоять не сразу за меткой ``table``: между ними бывает блок
    ``table_caption``. Поэтому HTML ищется до следующей метки ``table``.

    Args:
        raw: Ответ DeepSeek-OCR.
        width: Ширина страницы, px.
        height: Высота страницы, px.

    Returns:
        Таблицы с рамками в пикселях.
    """
    matches = list(_DS_BLOCK_RE.finditer(raw))

    def label(m: re.Match) -> str:
        """Метка блока в нижнем регистре."""
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
# латиница, похожая на кириллицу (после lower в norm): DeepSeek пишет «C4-C6» латиницей,
# Tesseract — кириллицей; для сопоставления это одно и то же
_HOMO = str.maketrans("abcehkmoptxy", "авсенкмортху")


def norm(s: str) -> str:
    """Нормализует текст для сравнения.

    Убирает различия оформления: пробелы, надстрочные и подстрочные цифры,
    виды тире, «ё», регистр. Остаются только расхождения букв и чисел.
    """
    s = s.translate(_SCRIPTS).replace("Ӏ", "I").replace("ё", "е").replace("_", "")
    s = re.sub(r"[–—−]", "-", s)
    return re.sub(r"\s+", "", s).lower()


def sim(a: str, b: str) -> float:
    """Похожесть текстов ячеек от 0 до 1.

    Сравнение мягкое к типичным ошибкам Tesseract («С6» → «Сб», «I» → «1»),
    но тексты с разными числами («≤ 60» и «≤ 240») считаются непохожими.
    """
    a, b = norm(a).translate(_HOMO), norm(b).translate(_HOMO)
    if not a or not b:
        return 0.0
    s = SequenceMatcher(None, a, b, autojunk=False).ratio()
    da, db = "".join(_DIGITS.findall(a)), "".join(_DIGITS.findall(b))
    if da and db and SequenceMatcher(None, da, db, autojunk=False).ratio() < 0.75:
        s *= 0.3
    return s


def _contained(part: str, whole: str) -> float:
    """Доля знаков ``part``, найденных в ``whole`` по порядку."""
    a, b = norm(part).translate(_HOMO), norm(whole).translate(_HOMO)
    if not a or not b:
        return 0.0
    return sum(x.size for x in SequenceMatcher(None, b, a, autojunk=False).get_matching_blocks()) / len(a)


def _letters(s: str) -> int:
    """Число букв в строке."""
    return sum(ch.isalpha() for ch in s)


_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def cell_text(html: str) -> str:
    """Переводит HTML ячейки в текст для DOCX.

    ``<br>`` становится переносом строки, цифры в ``<sup>``/``<sub>`` —
    надстрочными и подстрочными знаками («мг/нм³»), остальная разметка
    удаляется.
    """
    from npa_ocr.response_validation import UNREADABLE, is_service_response

    if is_service_response(unescape(html)):
        return UNREADABLE
    h = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)

    def script(m: re.Match) -> str:
        """Содержимое <sup>/<sub>: цифры — надстрочными или подстрочными знаками."""
        inner = re.sub(r"<[^>]+>", "", m.group(2))
        table = _SUP if m.group(1).lower() == "sup" else _SUB
        return inner.translate(table) if re.fullmatch(r"[\d+-]+", inner) else inner

    h = re.sub(r"<(sup|sub)>(.*?)</\1>", script, h, flags=re.S | re.I)
    h = unescape(re.sub(r"<[^>]+>", " ", h))
    from npa_ocr.scientific_text import chemical_indices

    return chemical_indices("\n".join(" ".join(line.split()) for line in h.split("\n") if line.strip()))


# --- 1–2. Сетка по линиям ---


@dataclass
class _Cell:
    """Ячейка сетки по линиям: рамка, позиция и сопоставленный текст."""

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
    literal: str | None = None  # однозначный короткий знак по геометрии изображения
    expected: str = ""      # тип графы из её заголовка
    number_rotation: bool = False  # подтверждён несколькими изображениями цифр


def _line_masks(gray):
    """Маски горизонтальных и вертикальных линий разлиновки."""
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
    """Группирует близкие значения (шаг не больше ``tol``) и возвращает медианы групп."""
    import numpy as np

    out: list[list[int]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(np.median(g)) for g in out]


def _detect_cells(img: Image.Image) -> tuple[list[_Cell], dict]:
    """Строит сетку ячеек по линиям разлиновки.

    Returns:
        Ячейки по строкам и столбцам и сведения о сетке; ``info["ok"]`` —
        сетка без перекрытий и дыр.
    """
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
    # рамка по крайним точкам всех линий: у таблиц часто нет боковых линий, а на стыке
    # страниц — верхней или нижней
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
    # При разрыве внешней рамки фон может обойти несколько настоящих ячеек.
    # Такой компонент не является гигантской объединённой ячейкой.
    cells = [c for c in cells if not any(
        d is not c and c.x0 <= d.x0 and c.y0 <= d.y0 and c.x1 >= d.x1 and c.y1 >= d.y1
        for d in cells)]
    tol = 14
    xe = _cluster([c.x0 for c in cells] + [c.x1 for c in cells], tol)
    ye = _cluster([c.y0 for c in cells] + [c.y1 for c in cells], tol)
    # края соседних ячеек разделены толщиной линии: пара близких краёв — одна граница
    xe = _cluster(xe, 2 * tol)
    ye = _cluster(ye, 2 * tol)

    def idx(edges: list[int], v: int) -> int:
        """Номер ближайшей границы."""
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
    # Восстанавливаем пропуски сетки по сохранившимся разделителям. Это
    # помогает строкам-разделам и крайним ячейкам с разорванной рамкой.
    missing = {(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) not in taken}
    repaired = 0
    # Утолщённые линии нужны для связности, но их выступающие концы не
    # должны превращать широкую строку-раздел в несколько узких ячеек.
    v_core = cv2.erode(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)))
    h_core = cv2.erode(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    if not overlap and len(missing) <= 0.4 * max(1, n_rows * n_cols):
        while missing:
            seed = min(missing)
            group, stack = {seed}, [seed]
            missing.remove(seed)
            while stack:
                r, c = stack.pop()
                for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                    if (rr, cc) not in missing:
                        continue
                    if rr == r:
                        edge = xe[max(c, cc)]
                        strip = v_core[ye[r] + 3:ye[r + 1] - 3, max(0, edge - 4):edge + 5]
                        divided = strip.size > 0 and (strip > 0).any(axis=1).mean() > 0.4
                    else:
                        edge = ye[max(r, rr)]
                        strip = h_core[max(0, edge - 4):edge + 5, xe[c] + 3:xe[c + 1] - 3]
                        divided = strip.size > 0 and (strip > 0).any(axis=0).mean() > 0.4
                    if not divided:
                        group.add((rr, cc)); stack.append((rr, cc)); missing.remove((rr, cc))
            r0, r1 = min(r for r, _ in group), max(r for r, _ in group) + 1
            c0, c1 = min(c for _, c in group), max(c for _, c in group) + 1
            if len(group) != (r1 - r0) * (c1 - c0):
                continue
            cells.append(_Cell(xe[c0] + 2, ye[r0] + 2, xe[c1] - 2, ye[r1] - 2,
                               r=r0, c=c0, rs=r1 - r0, cs=c1 - c0))
            for position in group:
                taken[position] = len(cells) - 1
            repaired += 1
    holes = n_rows * n_cols - len(taken)
    info.update(ok=overlap == 0 and holes == 0, n_rows=n_rows, n_cols=n_cols, overlap=overlap,
                holes=holes, repaired=repaired, ye=ye, xe=xe)
    cells.sort(key=lambda c: (c.r, c.c))
    return cells, info


# --- 3. Текст ---


def _tesseract_into_cells(img: Image.Image, cells: list[_Cell]) -> None:
    """Заполняет черновой текст ячеек по Tesseract.

    Слова раскладываются по ячейкам по центрам. Вертикальный текст в высоких
    узких ячейках читается повёрнутым; направление выбирается по уверенности
    Tesseract. Отмечаются ячейки с напечатанным текстом (``ink``).
    """
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
        from npa_ocr.stroke_geometry import faint_dash
        if faint_dash(gray[c.y0+6:c.y1-6,c.x0+6:c.x1-6]):
            c.literal = "—"
            c.ink = True
        c.ink = inner.size > 0 and (inner < 128).mean() > 0.002
        c.ink = c.ink or c.literal is not None
        if inner.size:
            binary = (inner < 128).astype(np.uint8)
            count, _, components, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            visible = [components[k] for k in range(1, count) if components[k, cv2.CC_STAT_AREA] >= 5]
            # Одиночная цифра в высокой объединённой ячейке занимает меньше
            # порога доли чернил. Такой фрагмент нельзя объявлять пустым.
            c.ink = c.ink or any(s[cv2.CC_STAT_HEIGHT] >= 8 and s[cv2.CC_STAT_WIDTH] >= 2
                                and s[cv2.CC_STAT_AREA] >= 12 for s in visible)
            # Короткий горизонтальный штрих в пустой ячейке — прочерк,
            # а не цифра 1 или римская I от повторного чтения модели.
            if len(visible) == 1:
                _, _, width, height, area = visible[0]
                if width >= 8 and width >= 3 * height and height <= 8 and area >= width:
                    c.literal = "—"
                    c.ink = True
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
    # вертикальный текст (шапки альбомных таблиц) Tesseract не видит — ячейка читается повёрнутой
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
    for c in cells:
        if c.literal is not None and re.fullmatch(r"\s*1[.]?\s*", c.tess):
            # Поворот узкой цифры в шапке может выглядеть как штрих.
            # Явно прочитанное число не заменяется геометрической догадкой.
            c.literal = None
        if c.literal is not None:
            c.tess = c.literal


def _set_column_types(cells: list[_Cell]) -> None:
    """Определяет типы граф по распознанным заголовкам, не по значениям норм."""
    from npa_ocr.scientific_text import ROMAN_CODE, doubtful_roman_code, section_heading

    types: dict[int, str] = {}
    headers: set[int] = set()
    for c in cells:
        if c.cs != 1:
            continue
        text = (c.tess + " " + cell_text(c.text)).lower()
        kind = ""
        if "формул" in text:
            kind = "formula"
        elif "регистрацион" in text or re.search(r"\bcas\b", text):
            kind = "cas"
        elif "категория работ" in text or "разряд" in text:
            kind = "code"
        elif "наименование" in text:
            kind = "name"
        elif "п/п" in text or text.strip() == "№":
            kind = "index"
        elif ("класс опасности" in text or "количество" in text or "величина" in text
              or "освещенность, лк" in text or "освещённость, лк" in text
              or (c.r<=3 and any(re.match(r'\s*пд[ку]\b',v.lower())
                                for v in (c.tess,cell_text(c.text))))):
            kind = "number"
        if kind:
            types[c.c] = kind
            headers.add(id(c))
    # В продолжении таблицы заголовка может не быть: тип графы определяется
    # по большинству её коротких записей, но никогда по ожидаемому числу.
    for col in {c.c for c in cells if c.cs == 1} - types.keys():
        values = [cell_text(c.text) or c.tess for c in cells if c.c == col and c.cs == 1]
        values = [v.strip() for v in values if v.strip() and len(v) < 40]
        column = [c for c in cells if c.c == col and c.cs == 1]
        if (col == 0 and len(column) >= 3 and values
                and all(re.fullmatch(r"\d{1,4}[.]?", v) for v in values)
                and max(c.x1 - c.x0 for c in column) < .08 * max(c.x1 for c in cells)):
            types[col] = "index"
            continue
        if len(values) < 2:
            continue
        codes = sum(bool(ROMAN_CODE.fullmatch(v)) or doubtful_roman_code(v) for v in values)
        numbers = sum(bool(re.fullmatch(r"[\d.,*\s≤≥<>=–—−-]+", v)) for v in values)
        if codes >= max(3, .6 * len(values)):
            types[col] = "code"
        elif numbers >= max(2, .8 * len(values)):
            types[col] = "index" if col == 0 else "number"
    for c in cells:
        if id(c) not in headers and c.cs == 1:
            c.expected = types.get(c.c, "")
            if c.expected == "name" and "formula" in types.values() and "cas" in types.values():
                c.expected = "chemical_name"
        elif c.cs > 1:
            text = cell_text(c.text) or c.tess
            if section_heading(text) or re.match(r"^[IVXLC1l|ГТПШУХ!тiх.:\s]{1,15}[А-ЯЁ][а-яё]{2}",text):
                c.expected = "section"
            elif ("регистрационный номер" in text.lower() or "частотной коррекции" in text
                  or re.search(r"ГОСТ Р\s+[\d.]+/", text)):
                c.expected = "mixed"


class _Rows(HTMLParser):
    """Разбор HTML-таблицы в строки ячеек ``(текст, rowspan, colspan)``."""

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
    """Значение rowspan/colspan; неверное или пустое — 1."""
    try:
        return max(1, int(v or 1))
    except ValueError:
        return 1


@dataclass
class _DsCell:
    """Непустая ячейка DeepSeek: текст и позиция в сетке DeepSeek."""

    text: str
    r: float  # центр по строкам в сетке DeepSeek
    c: int


def _ds_cells(html: str) -> list[_DsCell]:
    """Непустые ячейки HTML DeepSeek с позициями с учётом объединений."""
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
    """HTML непустых ячеек DeepSeek с сохранением ``<sup>``/``<sub>``/``<br>``."""
    return [m for m in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, re.S) if m.strip()]


def _numbering(cells: list[_Cell], ds: list[_DsCell], used_c: set[int], used_d: set[int]) -> None:
    """Сопоставляет строку нумерации столбцов («1 2 … 14»).

    Одиночные цифры по похожести текста не сопоставить, а DeepSeek читает
    нумерацию верно. Она ставится по порядку в строку сетки с тем же числом
    ячеек и короткими текстами Tesseract. По порядку, а не по номеру столбца:
    у DeepSeek бывают лишние столбцы.
    """
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
    """Раскладывает по ячейкам столбец значений, слитый DeepSeek в одну ячейку.

    Значения («≤ 4,1139 ≤ 0,1440 …») раскладываются по порядку в подряд идущие
    ячейки столбца, оставшиеся только с текстом Tesseract, если цифры каждого
    значения совпадают с Tesseract.

    Returns:
        Число разложенных ячеек DeepSeek.
    """
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
    """Сопоставляет ячейки DeepSeek ячейкам сетки.

    Оценка пары — похожесть текста DeepSeek и чернового текста Tesseract за
    вычетом штрафа за разницу положения (строки и столбца в долях таблицы).
    Положение различает одинаковые тексты («мг/нм³» в каждой строке). Строка
    ячейки сетки берётся по месту её текста: DeepSeek ставит текст объединения
    туда, где он виден. Пары выбираются жадно, затем идут два добора: кусков
    ячеек, разбитых DeepSeek, и ячеек, не прочитанных Tesseract, по положению.

    Args:
        cells: Ячейки сетки; результат пишется в их ``ds_ids`` и ``src``.
        ds: Ячейки DeepSeek.
        row_edges: Границы строк сетки по y.
    """
    import numpy as np

    n_rows_g = max(len(row_edges) - 1, 1)
    n_rows_d = max((int(d.r) for d in ds), default=0) + 1
    n_cols_g = max((c.c + c.cs for c in cells), default=1)
    n_cols_d = max((d.c for d in ds), default=0) + 1

    def g_row(c: _Cell) -> float:
        """Строка сетки, где виден текст ячейки."""
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
                continue  # короткий текст («5», «А») есть в любой таблице — только рядом со своим местом
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
                # пустая ячейка: кусок должен почти целиком читаться в её тексте (ячейку из
                # нескольких абзацев DeepSeek разбивает на строки)
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
        """Медиана."""
        return sorted(v)[len(v) // 2]

    row_map = {r: med(v) for r, v in row_votes.items()}
    col_map = {c: med(v) for c, v in col_votes.items()}
    shift = med([g - c for c, g in col_map.items()]) if col_map else 0
    anchors = sorted(row_map)

    def map_row(r: int) -> int | None:
        """Строка сетки для строки DeepSeek по опорным строкам или None."""
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
    """Итог сборки таблицы по сетке."""

    ok: bool                      # сетка построена
    cells: list[TableCell]
    n_rows: int
    n_cols: int
    info: dict                    # ds_cells, ds_lost, ds_invented, tess_only, by_pos, split, why
    # сомнительные ячейки для повторного чтения (vlm_reread): r, c, box (в координатах кропа), rot,
    # tess, left (соседняя слева ячейка с текстом DeepSeek: box, text), kind (tess|empty|mismatch), ds
    doubt: list[dict] = field(default_factory=list)


def _plain(h: str) -> str:
    """Текст HTML без тегов, с одиночными пробелами."""
    return " ".join(re.sub(r"<[^>]+>", " ", h).split())


def _doubt_kind(c: _Cell) -> str | None:
    """Вид сомнения в ячейке.

    Returns:
        ``"tess"`` — текст только от Tesseract; ``"empty"`` — напечатано, но
        текста нет; ``"mismatch"`` — текст DeepSeek не похож на прочитанное
        Tesseract; None — ячейка не сомнительна.
    """
    from npa_ocr.response_validation import is_service_response

    if is_service_response(c.text):
        return "empty"
    if c.expected in ("formula", "chemical_name") and c.literal is None:
        return c.expected
    if c.expected == "section":
        return "section"
    if c.src == "local":
        return None
    if c.literal is not None:
        return None
    from npa_ocr.response_validation import matches_cell_type

    if c.text and c.expected and not matches_cell_type(cell_text(c.text), c.expected):
        return "schema"
    if c.src == "tess":
        return "tess"
    from npa_ocr.scientific_text import doubtful_roman_code

    if doubtful_roman_code(cell_text(c.text)):
        return "code"
    if c.ink and not c.text:
        return "empty"
    if c.src in ("ds", "pos") and c.tess:
        a, b = norm(c.tess), norm(cell_text(c.text))
        # Даже одна цифра важна: 20 и 2, прочерк и 1 не требуют четырёх букв.
        if re.findall(r"\d+", a) != re.findall(r"\d+", b):
            return "mismatch"
        if a != b and (re.fullmatch(r"[-–—−1Il|]+", a) or re.fullmatch(r"[-–—−1Il|]+", b)):
            return "mismatch"
    # вертикальный текст (Tesseract прочёл его с поворотом) DeepSeek читает заметно хуже —
    # там сомнительно любое расхождение
    if c.src in ("ds", "pos") and _letters(c.tess) >= 4 and sim(c.tess, _plain(c.text)) < (0.95 if c.rot else 0.5):
        return "mismatch"
    return None


def _doubt(cells: list[_Cell]) -> list[dict]:
    """Описания сомнительных ячеек для повторного чтения (см. ``GridResult.doubt``)."""

    def left_of(c: _Cell) -> dict | None:
        """Соседняя слева ячейка с текстом DeepSeek — контекст для повторного чтения."""
        k = next((x for x in cells if x.c + x.cs == c.c and x.r <= c.r < x.r + x.rs and x.src in ("ds", "pos")),
                 None)
        return {"box": [k.x0, k.y0, k.x1, k.y1], "text": _plain(k.text)} if k else None

    out = []
    for c in cells:
        kind = _doubt_kind(c)
        if kind:
            numbered=[(x.c,int(cell_text(x.text))) for x in cells if x.r==c.r and x.rs==1 and x.cs==1
                      and re.fullmatch(r'\d{1,2}',cell_text(x.text))]
            numbering=(c.r<=2 and len(numbered)>=3 and all(value-col==1 for col,value in numbered))
            out.append({"r": c.r, "c": c.c, "box": [c.x0, c.y0, c.x1, c.y1], "rot": c.rot, "tess": c.tess,
                        "left": left_of(c), "kind": kind, "ds": cell_text(c.text) if kind in ("mismatch", "formula", "chemical_name", "section") else "",
                        "expected": c.expected,"numbering":numbering,"number_rotation":c.number_rotation})
    return out


def rebuild(img: Image.Image, ds_html: str) -> GridResult:
    """Собирает таблицу: сетка по линиям и текст DeepSeek в её ячейках.

    Args:
        img: Вырезанная таблица.
        ds_html: HTML таблицы от DeepSeek.

    Returns:
        Итог сборки; при ``ok=False`` сетка не построилась.
    """
    cells, info = _detect_cells(img)
    ds = _ds_cells(ds_html)
    if not info["ok"]:
        info.update(ds_cells=len(ds), ds_lost=len(ds))
        return GridResult(False, [], 0, 0, info)
    _tesseract_into_cells(img, cells)
    from npa_ocr.cell_reading import split_index_cells

    info["split_index"] = split_index_cells(img, cells, info["ye"])
    _set_column_types(cells)
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
        if (c.literal is not None and c.expected not in ("number", "index", "code")
                and re.fullmatch(r"\s*1[.]?\s*", cell_text(c.text))):
            # Несогласие модели с коротким штрихом требует повторного
            # чтения: штрих может быть повёрнутой цифрой в узкой шапке.
            c.literal = None
        if c.literal is not None:
            c.text, c.src = c.literal, "visual"
    _set_column_types(cells)
    from npa_ocr.cell_reading import verify_cells, orient_numbering

    orient_numbering(img,cells)
    verify_cells(img, cells)
    n_split = _split_values(cells, ds)
    doubt = _doubt(cells)
    lost = [ds[j].text for j in range(len(ds)) if not any(j in c.ds_ids for c in cells)]
    # повтор уже размещённого текста (DeepSeek дублирует его в выдуманных строках) — не потеря;
    # потеря — текст, которого в итоговой таблице нет вовсе
    placed_norm = {norm(ds[j].text) for c in cells for j in c.ds_ids}
    lost_unique = [t for t in lost if norm(t) not in placed_norm]
    # текст DeepSeek, которого Tesseract на скане не видит вовсе, выдуман; потеря — только то,
    # что на скане есть, а в сетку не попало
    all_tess = " ".join(c.tess for c in cells)
    invented = [t for t in lost_unique if len(norm(t)) > 3 and _contained(t, all_tess) < 0.7]
    lost_unique = [t for t in lost_unique if t not in invented]
    info.update(ds_cells=len(ds), ds_lost=len(lost_unique), lost=lost_unique[:10], ds_invented=invented[:10],
                tess_only=sum(c.src == "tess" for c in cells), by_pos=sum(c.src == "pos" for c in cells),
                split=n_split)
    out = []
    for c in cells:
        from npa_ocr.response_validation import UNREADABLE

        text = cell_text(c.text) or (UNREADABLE if c.ink else "")
        if c.expected == "formula" and text:
            from npa_ocr.response_validation import matches_cell_type
            if not matches_cell_type(text, "formula"):
                text = UNREADABLE
        low = c.src in ("tess", "pos", "local") or _doubt_kind(c) is not None or text == UNREADABLE
        out.append(TableCell(c.r, c.c, c.rs, c.cs, text,
                             (True,) * len(text.split("\n")) if low and text else ()))
    return GridResult(True, out, info["n_rows"], info["n_cols"], info, doubt)


# (кроп таблицы, сомнительные ячейки GridResult.doubt) -> принятый текст по (строка, столбец)
RereadFn = Callable[[Image.Image, list[dict]], dict[tuple[int, int], str]]


def apply_reread(cells: list[TableCell], texts: dict[tuple[int, int], str]) -> list[TableCell]:
    """Подставляет в ячейки текст повторного чтения.

    Повторное чтение не доказывает правильность результата: исправленные
    ячейки остаются выделенными для сверки со сканом.
    """
    out = []
    for c in cells:
        t = texts.get((c.row, c.col))
        text = cell_text(escape(t)) if t is not None else None
        out.append(c if text is None else TableCell(c.row, c.col, c.row_span, c.col_span, text,
                                                    (True,) * len(text.split("\n"))))
    return out


def html_cells(html: str) -> tuple[list[TableCell], int, int]:
    """Переводит HTML DeepSeek в ячейки как есть.

    Запасной путь, когда сетка по линиям не годится.

    Returns:
        Ячейки, число строк и число столбцов.
    """
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
            text = cell_text(inner)
            from npa_ocr.response_validation import is_service_response

            cells.append(TableCell(r, c, rs, cs, text,
                                   (True,) * len(text.split("\n")) if is_service_response(text) else ()))
            c += cs
    n_rows = max((x.row + x.row_span for x in cells), default=0)
    n_cols = max((x.col + x.col_span for x in cells), default=0)
    return cells, n_rows, n_cols


# --- Таблицы страницы ---


def refine_bbox(page: Image.Image, box: list[int], pad: int = 12) -> list[int]:
    """Уточняет рамку таблицы по разлиновке страницы.

    DeepSeek иногда даёт рамку только вокруг шапки, а HTML — всей таблицы.
    Рамка расширяется до связных кусков линий, задевающих её и лежащих в
    основном внутри неё с запасом.

    Args:
        page: Изображение страницы.
        box: Рамка DeepSeek.
        pad: Запас вокруг итоговой рамки, px.
    """
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
    """Площадь пересечения рамок, отнесённая к площади меньшей."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(small, 1)


@dataclass
class VlmTable:
    """Таблица страницы и способ, которым она получена."""

    bbox: list[int]
    cells: list[TableCell]
    n_rows: int
    n_cols: int
    src: str                      # grid — сетка по линиям, deepseek — HTML DeepSeek как есть
    html: str                     # HTML DeepSeek (склеенный, если таблиц в одной разлиновке несколько)
    info: dict = field(default_factory=dict)


def page_tables(page: Image.Image, tables: list[DsTable], reread_fn: RereadFn | None = None) -> list[VlmTable]:
    """Собирает итоговые таблицы страницы из таблиц DeepSeek.

    Таблицы DeepSeek в одной разлиновке объединяются: DeepSeek разрезает
    таблицу по строке-подзаголовку.

    Args:
        page: Изображение страницы.
        tables: Таблицы из ответа DeepSeek.
        reread_fn: Повторное чтение сомнительных ячеек
            (:func:`npa_ocr.vlm_reread.reread_cells`) или None.

    Returns:
        Таблицы сверху вниз.
    """
    boxes = [t.bbox if t.ruled else refine_bbox(page, t.bbox) for t in tables]
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
        unruled_body = False
        if res.ok and res.info["ds_lost"] > _MAX_LOST_SHARE * max(res.info["ds_cells"], 1):
            # Бывает разлинована только шапка: тогда текст настоящего тела
            # ниже рамки должен подтверждаться на странице, а не одним HTML.
            fallback_cells, _, _ = html_cells(html)
            extended = extend_to_rows(page, box, fallback_cells, page.height)
            unruled_body = extended[3] > box[3] + 20
        # сетка по линиям, если она вместила текст DeepSeek; иначе (нет линий в теле таблицы
        # или сетка не построилась) — HTML DeepSeek как есть
        if res.ok and not unruled_body:
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
    """Продлевает рамку таблицы вниз по строкам текста страницы, входящим в таблицу.

    Строка Tesseract должна читаться в тексте строки таблицы — той же, что
    предыдущая, или ниже. Первая чужая строка заканчивает таблицу. Порядок
    важен: текст под таблицей может почти дословно повторять её уже пройденную
    строку.

    Args:
        page: Изображение страницы.
        box: Рамка таблицы.
        cells: Ячейки таблицы.
        limit_y: Нижний предел (начало следующей таблицы или низ страницы).
        pad: Запас под последней строкой, px.
    """
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
    """Распознаёт таблицы страницы-скана.

    Args:
        image_path: Изображение страницы.
        chat_fn: Запрос к DeepSeek-OCR; по умолчанию :func:`deepseek_chat`.
        reread_fn: Повторное чтение сомнительных ячеек или None.

    Returns:
        Таблицы страницы.

    Raises:
        VlmUnavailable: vLLM недоступен.
    """
    page = Image.open(image_path).convert("RGB")
    raw = chat_fn(page)
    tables = parse_deepseek(raw, page.width, page.height)
    # Отсутствие HTML не означает отсутствия таблицы: проверяем разлиновку
    # независимо от модели и отдельно читаем пропущенный регион.
    for box in ruled_regions(page):
        members = [t for t in tables if _region_coverage(t.bbox, box) >= 0.8]
        if members:
            # Разделы одной физической сетки объединяются до сопоставления
            # текста. Повторный ответ модели не добавляется поверх исходных
            # фрагментов: иначе появляются дубли и выдуманные строки.
            html = [t.html for t in sorted(members, key=lambda t: t.bbox[1])]
        else:
            recovered = chat_fn(page.crop(box))
            html = re.findall(r"<table\b.*?</table>", recovered, re.S | re.I)
        if not html:
            # Разлиновка подтверждает таблицу независимо от качества ответа.
            # Пустой HTML позволит восстановить её по Tesseract и явно
            # отметить те напечатанные ячейки, которые не удалось прочитать.
            html = ["<table></table>"]
        tables = [t for t in tables if t not in members]
        tables.append(DsTable(box, "<table>" + "".join(
            re.sub(r"</?table[^>]*>", "", h) for h in html) + "</table>", ruled=True))
    return page_tables(page, tables, reread_fn)


def _region_coverage(region: list[int], box: list[int]) -> float:
    """Доля региона, покрытая рамкой уже распознанной таблицы."""
    area = max(1, (region[2] - region[0]) * (region[3] - region[1]))
    return max(0, min(region[2], box[2]) - max(region[0], box[0])) * max(
        0, min(region[3], box[3]) - max(region[1], box[1])) / area


def ruled_regions(page: Image.Image) -> list[list[int]]:
    """Находит крупные связные сетки независимо от ответа модели.

    Рамки скана, одиночные подчёркивания и небольшие метки не считаются
    таблицами: нужны несколько длинных горизонтальных и вертикальных линий.
    """
    import cv2
    import numpy as np

    h, v = _line_masks(np.array(page.convert("L")))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_or(h, v), connectivity=8)
    result = []
    for i in range(1, count):
        x, y, w, height, _ = map(int, stats[i])
        if w < page.width * 0.35 or height < page.height * 0.08:
            continue
        if x < 5 or y < 5 or x + w >= page.width - 5 or y + height >= page.height - 5:
            continue
        rows = np.where((h[y:y + height, x:x + w] > 0).sum(axis=1) > w * 0.5)[0]
        cols = np.where((v[y:y + height, x:x + w] > 0).sum(axis=0) > height * 0.3)[0]
        if len(_cluster(rows.tolist(), 8)) < 3 or len(_cluster(cols.tolist(), 8)) < 3:
            continue
        result.append([max(0, x - 12), max(0, y - 12),
                       min(page.width, x + w + 12), min(page.height, y + height + 12)])
    return sorted(result, key=lambda b: b[1])
