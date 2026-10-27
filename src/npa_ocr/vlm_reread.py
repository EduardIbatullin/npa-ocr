"""Дочитка сомнительных ячеек таблицы DeepSeek-OCR по кропу ячейки. Перенесено из спайка
npa-vlm-ocr (scripts/reread.py): на 14 документах — 156 ячеек, ~30 с на все.

Сомнительные ячейки отмечает сетка по линиям (vlm_tables.rebuild, GridResult.doubt):
- kind="tess" — текст есть только от Tesseract («≤» он читает как «5», даты в повёрнутой ячейке —
  мусор);
- kind="empty" — в ячейке что-то напечатано, но текста нет;
- kind="mismatch" — текст DeepSeek не похож на то, что видит Tesseract (228 стр. 25: вертикальную
  шапку DeepSeek прочла плохо).

Попытки по порядку, следующая — только если ответ предыдущей отбракован (accept):
  1. кроп ячейки, «Free OCR.»; вертикальный текст (is_vertical) — повёрнутым на ±90°, сторону
     выбирает уверенность модели (средний logprob токенов);
  2. ячейка рядом с соседней слева: одна «мг/нм³» читается латиницей или формулой, а рядом с
     русским названием вещества — по-русски; известный текст соседа отрезается от ответа;
  3. кроп ячейки, запрос на markdown.
Для kind="mismatch" — только попытки 1–2, и текст DeepSeek заменяется лишь уверенным ответом
(logprob ≥ CONFIDENT), заметно отличным от него и более близким к Tesseract. Знаки и единицы не
угадываются — только чтение.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Callable

from PIL import Image, ImageOps

from npa_ocr.vlm_tables import VLLM_MODEL, VLLM_URL, VlmUnavailable, norm

PAD = 4        # отступ внутрь ячейки — линии рамки в кроп не берём
BORDER = 24    # белое поле вокруг кропа
MIN_H = 48     # мелкие ячейки увеличиваем, иначе одиночная цифра — пара десятков пикселей
CONFIDENT = -0.15  # средний logprob, с которого ответу верим без сверки с Tesseract
WORKERS = 8    # запросов к vLLM одновременно (vLLM сам собирает их в батч)
CELL_TIMEOUT = 120

# (кроп, промпт) -> (ответ, причина остановки, средний logprob токенов)
AskFn = Callable[[Image.Image, str], tuple[str, str, float]]


def deepseek_ask(img: Image.Image, prompt: str) -> tuple[str, str, float]:
    """Ответ DeepSeek-OCR на кроп, причина остановки и средний logprob токенов (уверенность)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": 256,
        "logprobs": True,
    }
    req = urllib.request.Request(f"{VLLM_URL}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=CELL_TIMEOUT) as resp:
            ch = json.loads(resp.read())["choices"][0]
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as exc:
        raise VlmUnavailable(f"vLLM ({VLLM_URL}) недоступен: {exc}") from exc
    lps = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or [])]
    return ch["message"].get("content") or "", ch.get("finish_reason") or "", sum(lps) / max(len(lps), 1)


# --- Ответ модели: очистка и проверка ---

_TO_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _sup(m: re.Match) -> str:
    return "²³"[int(m[1]) - 2]


def clean(text: str) -> str:
    text = re.sub(r"<sup>\s*([23])\s*</sup>", _sup, text)
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)          # обёртка таблицы/разметки, если модель её добавила
    # одиночное число модель отдаёт формулой: «\[\le 0,09\]», степень — «мг/нм\(^3\)»
    text = re.sub(r"\\text\{\s*No\s*\}|\\textnumero", "№", text)
    # индексы: «C_6-C_{10}», «H_2SO_4» -> «C₆-C₁₀», «H₂SO₄»
    text = re.sub(r"_\{?(\d+)\}?", lambda m: m[1].translate(_TO_SUB), text)
    text = re.sub(r"\\[\[\]()]", " ", text).replace("\\leq", "≤").replace("\\le", "≤").replace("$", "")
    text = re.sub(r"\s*\^\{?([23])\}?", _sup, text)
    text = " ".join(text.split()).strip("- ")
    return re.sub(r"\(\s+", "(", re.sub(r"\s+([),.;])", r"\1", text))  # пробелы от вырезанной разметки


_LAT = re.compile(r"[A-Za-z]")
_CYR = re.compile(r"[А-Яа-яЁё]")
# латинские буквы, неотличимые от кириллических: «C 01.01.2001 no 31.12.2025»
_TO_CYR = str.maketrans("ABCEHKMOPTXYaceopxyn", "АВСЕНКМОРТХУасеорхуп")


def accept(text: str, d: dict, lp: float = -1.0) -> str | None:
    """Ответ модели или None, если он негоден: формула вместо единиц («[HM+3].[Mg]», «\\text{MT}»),
    латиница вместо русского текста; в ячейке без текста Tesseract (сверить не с чем; в повёрнутой
    227 стр. 9 модель сочиняет русский текст) — число, «№», прочерк или текст, в котором модель
    уверена (средний logprob ≥ CONFIDENT; у бессмыслицы в неверной ориентации −0.36…−0.9)."""
    if not text or "\\" in text or "^" in text or re.search(r"\[[A-Za-z+0-9.-]+\]", text):
        return None
    if re.search(r"[Ͱ-Ͽ]", text):  # греческие буквы — модель читает кириллицу как формулу
        return None
    tess = d["tess"]
    if len(_CYR.findall(tess)) > len(_LAT.findall(tess)):
        # латинские слова без цифр («no», «C») — кириллица, прочитанная латиницей; «C6-C10» не трогаем
        text = " ".join(w if re.search(r"[\d₀-₉]", w) else w.translate(_TO_CYR) for w in text.split())
        if _LAT.search(re.sub(r"\S*[\d₀-₉]\S*", "", text)):
            return None
    if not tess and not re.fullmatch(r"[≤<>≥]?\s*[\d.,]+|№|[-—]", text) and lp < CONFIDENT:
        return None
    return text


def strip_left(text: str, left: str) -> str:
    """Отрезать от ответа начало, совпадающее с текстом соседа (модель читает обе ячейки подряд)."""
    words, target = text.split(), norm(left)
    best, k_best = 0.0, 0
    for k in range(1, len(words)):
        s = SequenceMatcher(None, norm(" ".join(words[:k])), target, autojunk=False).ratio()
        if s > best:
            best, k_best = s, k
    return " ".join(words[k_best:]) if best >= 0.6 else ""


def _sim(a: str, b: str) -> float:
    """Похожесть с учётом цифр: «предельные С1-С5» и «предельные C₆-C₁₀» — разные тексты."""
    a, b = norm(a), norm(b)
    s = SequenceMatcher(None, a, b, autojunk=False).ratio()
    da, db = (re.sub(r"\D", "", x.translate(_SUB)) for x in (a, b))
    if da != db and SequenceMatcher(None, da, db, autojunk=False).ratio() < 0.75:
        s *= 0.3
    return s


def _keep_ds(got: str | None, lp: float, d: dict) -> str | None:
    """Текст DeepSeek (kind=mismatch) заменяем только уверенным ответом, который от него заметно
    отличается и ближе к Tesseract; иначе расхождение — от Tesseract (он не читает индексы «C₆-C₁₀»)."""
    if d.get("kind") != "mismatch" or got is None:
        return got
    if lp < CONFIDENT or _sim(got, d["ds"]) >= 0.95 or _sim(got, d["tess"]) <= _sim(d["ds"], d["tess"]):
        return None
    return got


# --- Кропы ---


def _part(crop: Image.Image, box: list[int]) -> Image.Image:
    x0, y0, x1, y1 = box
    return crop.crop((x0 + PAD, y0 + PAD, x1 - PAD, y1 - PAD)).convert("L")


def cell_image(crop: Image.Image, d: dict, rot: int = 0) -> Image.Image:
    part = _part(crop, d["box"])
    if rot:
        part = part.rotate(rot, expand=True, fillcolor=255)
    if part.height < MIN_H:
        k = MIN_H / max(part.height, 1)
        part = part.resize((round(part.width * k), MIN_H), Image.LANCZOS)
    return ImageOps.expand(part, BORDER, fill=255).convert("RGB")


def pair_image(crop: Image.Image, d: dict) -> Image.Image:
    a, b = _part(crop, d["left"]["box"]), _part(crop, d["box"])
    gap = 48
    img = Image.new("L", (a.width + gap + b.width, max(a.height, b.height)), 255)
    img.paste(a, (0, (img.height - a.height) // 2))
    img.paste(b, (a.width + gap, (img.height - b.height) // 2))
    return ImageOps.expand(img, BORDER, fill=255).convert("RGB")


def is_vertical(part: Image.Image, k: int = 15) -> bool:
    """Строки текста идут сверху вниз. Ink растягиваем вдоль горизонтали и вдоль вертикали: строки
    сливаются в полосы, и у вертикального текста вытянутость вертикальных полос в разы больше
    (228 стр. 25: 5–16 против 0.7–2; у горизонтального — наоборот). Tesseract вертикальность
    замечает не всегда («Код вида отходов…» он прочёл как «* о я Е Яо»)."""
    import cv2
    import numpy as np

    a = (np.array(part.convert("L")) < 128).astype(np.uint8)

    def elong(kernel, horiz: bool) -> float:
        _, _, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(a, kernel))
        num = den = 0.0
        for _x, _y, w, h, area in st[1:]:
            if area >= 30:
                num += area * (w / h if horiz else h / w)
                den += area
        return num / den if den else 0.0

    return elong(np.ones((k, 1), np.uint8), False) > 1.5 * elong(np.ones((1, k), np.uint8), True)


# --- Попытки ---


def _try_cell(ask: AskFn, crop: Image.Image, d: dict) -> dict:
    # вертикальный текст — читаем повёрнутым, сторону (±90°) выбирает уверенность модели: в неверной
    # она пишет бессмыслицу с низкой уверенностью. Горизонтальный — как есть; если в ячейке без текста
    # Tesseract ответ негоден, пробуем и повороты (227 стр. 9: повёрнутая цифра нумерации «5»)
    def read(rot: int) -> dict:
        raw, reason, lp = ask(cell_image(crop, d, rot), "Free OCR.")
        text = clean(raw)
        got = _keep_ds(accept(text, d, lp) if reason != "length" else None, lp, d)
        return {"vlm": text, "lp": lp, "rot": rot, "used": got}

    if is_vertical(_part(crop, d["box"])):
        return max((read(90), read(-90)), key=lambda c: c["lp"])
    res = read(0)
    if res["used"] is None and not d["tess"]:
        res = max((res, read(90), read(-90)), key=lambda c: (c["used"] is not None, c["lp"]))
    return res


def _try_pair(ask: AskFn, crop: Image.Image, d: dict) -> dict:
    if not d.get("left") or d.get("rot"):
        return {"skip": True}
    raw, reason, lp = ask(pair_image(crop, d), "Free OCR.")
    text = strip_left(clean(raw), d["left"]["text"])
    return {"vlm": text, "lp": lp, "used": _keep_ds(accept(text, d, lp), lp, d) if reason != "length" else None}


def _try_cell_md(ask: AskFn, crop: Image.Image, d: dict) -> dict:
    if d.get("kind") == "mismatch":
        return {"skip": True}
    raw, reason, lp = ask(cell_image(crop, d), "Convert the document to markdown.")
    text = clean(raw)
    return {"vlm": text, "lp": lp, "used": accept(text, d, lp) if reason != "length" else None}


ATTEMPTS = (("cell", _try_cell), ("pair", _try_pair), ("cell_md", _try_cell_md))


def reread_cells(crop: Image.Image, doubt: list[dict], ask: AskFn = deepseek_ask) -> dict[tuple[int, int], str]:
    """Дочитка сомнительных ячеек таблицы. crop — кроп таблицы, в координатах которого заданы рамки
    ячеек doubt (vlm_tables.GridResult.doubt). Возвращает принятый текст по (строка, столбец).
    vlm_tables.VlmUnavailable пробрасывается."""
    results: list[dict] = [{"used": None} for _ in doubt]
    todo = list(range(len(doubt)))
    for _name, attempt in ATTEMPTS:
        if not todo:
            break
        with ThreadPoolExecutor(WORKERS) as ex:
            answers = list(ex.map(lambda i: attempt(ask, crop, doubt[i]), todo))
        for i, res in zip(todo, answers):
            if not res.get("skip"):
                results[i] = res
        todo = [i for i in todo if results[i]["used"] is None]
    return {(d["r"], d["c"]): res["used"] for d, res in zip(doubt, results) if res["used"] is not None}
