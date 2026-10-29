"""Повторное чтение сомнительных ячеек таблицы DeepSeek-OCR по фрагменту ячейки.

Сомнительные ячейки отмечает :func:`npa_ocr.vlm_tables.rebuild`
(``GridResult.doubt``):

- ``kind="tess"`` — текст есть только от Tesseract (например, «≤» он читает
  как «5»);
- ``kind="empty"`` — в ячейке что-то напечатано, но текста нет;
- ``kind="mismatch"`` — текст DeepSeek не похож на прочитанное Tesseract.

Попытки выполняются по порядку; следующая — только если ответ предыдущей
отбракован (:func:`accept`):

1. Фрагмент ячейки с запросом «Free OCR.». Вертикальный текст
   (:func:`is_vertical`) читается повёрнутым на ±90°; сторону выбирает
   уверенность модели (средний logprob токенов).
2. Ячейка вместе с соседней слева: единицы вроде «мг/нм³» отдельно читаются
   латиницей или формулой, а рядом с русским названием — по-русски. Известный
   текст соседа отрезается от ответа.
3. Фрагмент ячейки с запросом на markdown.

Для ``kind="mismatch"`` выполняются только попытки 1–2, и текст DeepSeek
заменяется лишь уверенным ответом (logprob ≥ ``CONFIDENT``), заметно отличным
от него и более близким к Tesseract. Знаки и единицы не угадываются.
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
    """Запрашивает DeepSeek-OCR по фрагменту изображения.

    Args:
        img: Фрагмент.
        prompt: Запрос.

    Returns:
        Текст ответа, причина остановки генерации и средний logprob токенов
        (мера уверенности).

    Raises:
        VlmUnavailable: vLLM недоступен или ответ не разобран.
    """
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
    """Надстрочная цифра «²» или «³» для найденной «2» или «3»."""
    return "²³"[int(m[1]) - 2]


def clean(text: str) -> str:
    """Очищает ответ модели от разметки и LaTeX.

    Убирает служебные токены и теги, переводит индексы в подстрочные цифры
    («H_2SO_4» → «H₂SO₄»), степени — в надстрочные, ``\\le`` — в «≤».
    """
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
    """Проверяет ответ модели для ячейки.

    Отбраковываются формулы вместо единиц («[HM+3].[Mg]», ``\\text{MT}``),
    греческие буквы и латиница на месте русского текста. Если у Tesseract в
    ячейке текста нет и сверить не с чем, принимаются только число, «№»,
    прочерк или ответ, в котором модель уверена (средний logprob ≥
    ``CONFIDENT``): в неверной ориентации модель пишет бессмыслицу с низкой
    уверенностью.

    Args:
        text: Очищенный ответ модели.
        d: Описание ячейки; ``d["tess"]`` — текст Tesseract.
        lp: Средний logprob ответа.

    Returns:
        Принятый текст (латинские двойники заменены кириллицей) или None.
    """
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
    """Отрезает от ответа начало, совпадающее с текстом соседней ячейки.

    Returns:
        Остаток ответа или пустая строка, если начало не совпало с соседом.
    """
    words, target = text.split(), norm(left)
    best, k_best = 0.0, 0
    for k in range(1, len(words)):
        s = SequenceMatcher(None, norm(" ".join(words[:k])), target, autojunk=False).ratio()
        if s > best:
            best, k_best = s, k
    return " ".join(words[k_best:]) if best >= 0.6 else ""


def _sim(a: str, b: str) -> float:
    """Похожесть текстов с учётом цифр.

    Тексты с разными цифрами («С1-С5» и «C₆-C₁₀») считаются непохожими, даже
    если совпадают буквы.
    """
    a, b = norm(a), norm(b)
    s = SequenceMatcher(None, a, b, autojunk=False).ratio()
    da, db = (re.sub(r"\D", "", x.translate(_SUB)) for x in (a, b))
    if da != db and SequenceMatcher(None, da, db, autojunk=False).ratio() < 0.75:
        s *= 0.3
    return s


def _keep_ds(got: str | None, lp: float, d: dict) -> str | None:
    """Решает, заменять ли текст DeepSeek ответом повторного чтения.

    Для ``kind="mismatch"`` ответ принимается, только если он уверенный,
    заметно отличается от текста DeepSeek и ближе к Tesseract; иначе
    расхождение, скорее всего, на стороне Tesseract (например, он не читает
    индексы «C₆-C₁₀»). Для остальных видов ответ возвращается без изменений.
    """
    if d.get("kind") != "mismatch" or got is None:
        return got
    if lp < CONFIDENT or _sim(got, d["ds"]) >= 0.95 or _sim(got, d["tess"]) <= _sim(d["ds"], d["tess"]):
        return None
    return got


# --- Кропы ---


def _part(crop: Image.Image, box: list[int]) -> Image.Image:
    """Фрагмент ячейки без линий рамки, в оттенках серого."""
    x0, y0, x1, y1 = box
    return crop.crop((x0 + PAD, y0 + PAD, x1 - PAD, y1 - PAD)).convert("L")


def cell_image(crop: Image.Image, d: dict, rot: int = 0) -> Image.Image:
    """Готовит изображение ячейки для модели.

    Args:
        crop: Фрагмент таблицы.
        d: Описание ячейки с рамкой ``d["box"]``.
        rot: Поворот, градусы.

    Returns:
        Повёрнутая и при необходимости увеличенная ячейка с белым полем.
    """
    part = _part(crop, d["box"])
    if rot:
        part = part.rotate(rot, expand=True, fillcolor=255)
    if part.height < MIN_H:
        k = MIN_H / max(part.height, 1)
        part = part.resize((round(part.width * k), MIN_H), Image.LANCZOS)
    return ImageOps.expand(part, BORDER, fill=255).convert("RGB")


def pair_image(crop: Image.Image, d: dict) -> Image.Image:
    """Изображение соседней слева ячейки и самой ячейки рядом, с белым полем."""
    a, b = _part(crop, d["left"]["box"]), _part(crop, d["box"])
    gap = 48
    img = Image.new("L", (a.width + gap + b.width, max(a.height, b.height)), 255)
    img.paste(a, (0, (img.height - a.height) // 2))
    img.paste(b, (a.width + gap, (img.height - b.height) // 2))
    return ImageOps.expand(img, BORDER, fill=255).convert("RGB")


def is_vertical(part: Image.Image, k: int = 15) -> bool:
    """Проверяет, что текст в ячейке написан вертикально.

    Изображение растягивается морфологическим расширением вдоль вертикали и
    вдоль горизонтали, чтобы строки слились в полосы. У вертикального текста
    вертикальные полосы вытянуты в разы сильнее горизонтальных. Tesseract
    вертикальный текст замечает не всегда.

    Args:
        part: Фрагмент ячейки.
        k: Длина ядра расширения, px.
    """
    import cv2
    import numpy as np

    a = (np.array(part.convert("L")) < 128).astype(np.uint8)

    def elong(kernel, horiz: bool) -> float:
        """Средняя вытянутость компонент после расширения, взвешенная по площади."""
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
    """Попытка 1: чтение фрагмента ячейки.

    Вертикальный текст читается повёрнутым, сторону (±90°) выбирает
    уверенность модели. Горизонтальный — как есть; если у Tesseract в ячейке
    текста нет и ответ отбракован, пробуются и повороты (повёрнутая цифра).
    """

    def read(rot: int) -> dict:
        """Читает ячейку, повёрнутую на ``rot`` градусов."""
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
    """Попытка 2: чтение ячейки вместе с соседней слева."""
    if not d.get("left") or d.get("rot"):
        return {"skip": True}
    raw, reason, lp = ask(pair_image(crop, d), "Free OCR.")
    text = strip_left(clean(raw), d["left"]["text"])
    return {"vlm": text, "lp": lp, "used": _keep_ds(accept(text, d, lp), lp, d) if reason != "length" else None}


def _try_cell_md(ask: AskFn, crop: Image.Image, d: dict) -> dict:
    """Попытка 3: фрагмент ячейки с запросом на markdown."""
    if d.get("kind") == "mismatch":
        return {"skip": True}
    raw, reason, lp = ask(cell_image(crop, d), "Convert the document to markdown.")
    text = clean(raw)
    return {"vlm": text, "lp": lp, "used": accept(text, d, lp) if reason != "length" else None}


ATTEMPTS = (("cell", _try_cell), ("pair", _try_pair), ("cell_md", _try_cell_md))


def reread_cells(crop: Image.Image, doubt: list[dict], ask: AskFn = deepseek_ask) -> dict[tuple[int, int], str]:
    """Повторно читает сомнительные ячейки таблицы.

    Args:
        crop: Фрагмент таблицы; рамки ячеек заданы в его координатах.
        doubt: Сомнительные ячейки (``GridResult.doubt``).
        ask: Запрос к модели; по умолчанию :func:`deepseek_ask`.

    Returns:
        Принятый текст по позиции ``(строка, столбец)``.

    Raises:
        VlmUnavailable: vLLM недоступен.
    """
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
