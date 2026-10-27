"""Штампы, печати, подписи на страницах-сканах: текст меток выкидывается, текст документа под ними
и рядом остаётся. Три прочтения страницы:

- qwen3-vl (Ollama) — рамки меток (`locate_marks`): штамп регистрации и штамп электронной подписи,
  круглая печать, рукописная подпись;
- Tesseract — слова с рамками и уверенностью (как и без меток, см. ocr.run_tesseract_text);
- DeepSeek-OCR — ответ на страницу, уже полученный для таблиц (vlm_tables): его текст почти не
  содержит меток и читает слова сквозь оттиск печати — опора для слов у меток.

Правила (`clean_lines`) — по спайку npa-vlm-ocr (marks/README.md: 56 страниц 14 документов; на
страницах без меток qwen3-vl не дал ни одной рамки; выкинут только текст меток):
- слово вне рамок не трогается; неуверенное слово в строке у метки (росчерк перечёркивает фамилию)
  берёт написание DeepSeek, если тот его подтверждает;
- подтверждение — похожее слово (или пара слов) DeepSeek из блока, лежащего там же, где слово
  Tesseract; блоки DeepSeek в рамке штампа — сам текст штампа, не опора (росчерк и оттиск DeepSeek
  не читает, печатное «Министр» в рамке подписи — настоящий текст);
- в штампе слово остаётся, только если продолжает печатную строку документа (та же высота и линия,
  уверенное, без разрыва) и в строке ещё не начался штамп («377 | регистрационный № 69836»);
- в рамке подписи или печати остаётся подтверждённое DeepSeek или уверенное печатное слово;
- задевающее рамку неподтверждённое: короткое, неуверенное или в строке без слов документа —
  выкидывается;
- строка у штампа, начатая как название органа (authority_name), дописывается им: край штампа
  Минюста ложится на «ЭКОЛОГИИ» в шапке Минприроды, и Tesseract его портит (как восстановление
  должности подписанта, ocr._restore_signatory_title_words);
- висящие в конце строки знаки рамки («|», «]») и строки без букв и цифр — выкидываются.

Замены и дописанное помечаются как неуверенные — в DOCX жёлтые, для проверки человеком."""

from __future__ import annotations

import base64
import difflib
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable

from PIL import Image

from npa_ocr.vlm_tables import _DS_BLOCK_RE

Box = tuple[float, float, float, float]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MARKS_MODEL = os.environ.get("NPA_OCR_MARKS_MODEL", "qwen3-vl:8b")
MARKS_TIMEOUT = 300

MARKS_PROMPT = """Find every stamp, seal and handwritten signature on this scanned document page.
Stamps are rectangular registration stamps (e.g. "ЗАРЕГИСТРИРОВАНО") and electronic signature stamps
(a frame "ДОКУМЕНТ ПОДПИСАН ЭЛЕКТРОННОЙ ПОДПИСЬЮ" with certificate details); seals are round ink seals;
signatures are handwritten scribbles. Do NOT include printed document text, the coat of arms or the name of the signer.
Handwritten dates and numbers written into blank fields (after "от" and "№") are document data, NOT signatures.
Answer only with JSON: [{"label": "stamp" | "seal" | "signature", "bbox_2d": [x1, y1, x2, y2]}]"""

LABELS = ("stamp", "seal", "signature")

DS_BLOCK_IN_STAMP = 0.6  # блок DeepSeek, лежащий в рамке штампа на эту долю площади, — текст штампа
MATCH = 0.7  # похожесть слова Tesseract на слово DeepSeek, чтобы считать его подтверждённым
CONFIDENT = 50  # уверенность Tesseract (0–100), с которой слово у метки может остаться без подтверждения
CONFIDENT_PRINTED = 85  # печатное слово в рамке подписи/печати (росчерк и оттиск читаются неуверенно)
LOW_NEAR_MARK = 80  # неуверенное слово в строке у метки берёт написание DeepSeek
HEADER_MATCH = 0.6  # похожесть испорченного хвоста шапки на название органа


class MarksUnavailable(RuntimeError):
    """Ollama недоступна или ответила ошибкой — страница остаётся без чистки меток."""


@dataclass(frozen=True)
class Mark:
    label: str  # stamp / seal / signature
    box: Box  # пиксели страницы


@dataclass(frozen=True)
class Word:
    text: str
    conf: float  # уверенность Tesseract, 0–100
    box: Box
    restored: bool = False  # текст заменён по DeepSeek или дописан по названию органа


# --- Поиск меток (qwen3-vl) ---


def qwen_locate(image: Image.Image) -> str:
    """Ответ qwen3-vl на страницу. Без рассуждений (think=false): с ними модель однажды думала
    450 с и вернула пустой ответ (спайк, 156 стр. 1)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    payload = {
        "model": MARKS_MODEL,
        "prompt": MARKS_PROMPT,
        "images": [base64.b64encode(buf.getvalue()).decode()],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=MARKS_TIMEOUT) as resp:
            return json.loads(resp.read())["response"]
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as exc:
        raise MarksUnavailable(f"Ollama ({OLLAMA_URL}, {MARKS_MODEL}) недоступна: {exc}") from exc


def parse_marks(raw: str, width: int, height: int) -> list[Mark]:
    """Ответ qwen3-vl → метки в пикселях страницы (Qwen3-VL даёт координаты в долях 0..1000)."""
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    marks = []
    for it in items:
        if not isinstance(it, dict) or it.get("label") not in LABELS:
            continue
        bbox = it.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        marks.append(Mark(it["label"], (x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000)))
    return marks


def locate_marks(image: Image.Image, ask: Callable[[Image.Image], str] = qwen_locate) -> list[Mark]:
    return parse_marks(ask(image), image.width, image.height)


def should_locate(page_number: int, total_pages: int, ds_raw: str | None, width: int, height: int,
                  low_clusters: int = 0) -> bool:
    """На какой странице звать qwen3-vl (~11 с на страницу): первая и последняя (штамп регистрации,
    подпись), страница со скоплением неуверенных слов Tesseract (mark_detection) и страница, где
    DeepSeek выделил картинку не на месте герба (печать, подпись, штамп электронной подписи)."""
    if page_number in (1, total_pages) or low_clusters:
        return True
    for label, box, _text in ds_blocks(ds_raw or "", width, height):
        if label == "image" and not _is_coat_of_arms(box, width, height):
            return True
    return False


def _is_coat_of_arms(box: Box, width: int, height: int) -> bool:
    cx = (box[0] + box[2]) / 2
    return box[3] < 0.2 * height and abs(cx - width / 2) < 0.1 * width


# --- Чистка строк ---


def norm(w: str) -> str:
    return re.sub(r"[^\wА-Яа-яЁё]", "", w).lower()


def overlap(a: Box, b: Box) -> float:
    """Доля площади a внутри b."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def ds_blocks(raw: str, width: int, height: int) -> list[tuple[str, Box, str]]:
    """Блоки ответа DeepSeek-OCR: (метка, рамка в пикселях, текст). Координаты DeepSeek — 0..999."""
    ms = list(_DS_BLOCK_RE.finditer(raw))
    out = []
    for i, m in enumerate(ms):
        label = m.group(1) or m.group(2)
        try:
            boxes = json.loads(m.group(3))
        except json.JSONDecodeError:
            continue
        text = raw[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(raw)].strip()
        for b in boxes:
            if len(b) == 4:
                out.append((label, (b[0] * width / 999, b[1] * height / 999, b[2] * width / 999, b[3] * height / 999), text))
    return out


def _expand(box: Box, dx: float, dy: float) -> Box:
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def _vocab_at(box: Box, support: list[tuple[Box, list[str]]]) -> list[str]:
    """Слова и пары слов DeepSeek из блоков там же, где слово Tesseract: «№» номера приказа
    подтверждается блоком с номером, «№» в штампе справа — нет; пары — «_А-А-Козяев» у Tesseract
    это «А.А. Козлов» у DeepSeek."""
    near = [words for b, words in support if overlap(_expand(box, 15, 10), b) > 0]
    return [t for words in near for t in words] + [f"{a} {b}" for words in near for a, b in zip(words, words[1:])]


def _best_match(word: str, vocab: list[str]) -> tuple[float, str]:
    n = norm(word)
    if not n:
        return 0.0, ""
    best = (0.0, "")
    for v in vocab:
        nv = norm(v)
        if not nv:
            continue
        r = difflib.SequenceMatcher(None, n, nv).ratio()
        if len(n) >= 4 and nv.startswith(n.rstrip(".")):  # «Правите.» — обрезано печатью
            r = max(r, 0.9)
        if r > best[0]:
            best = (r, v)
    return best


def _complete_header(kept: list[Word], original: list[Word], authority_name: str | None) -> list[Word]:
    """Строка шапки у штампа, начатая как название органа: «МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И Э»,
    «… И'ЭКОЛОГИИ», «… й эколог ИИ» → «МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И ЭКОЛОГИИ». Дописываются
    только слова названия, начало которых есть в исходной строке Tesseract за совпавшими словами,
    и не дальше, чем она доходит (следующие слова названия — на следующей строке шапки)."""
    if not authority_name or not kept:
        return kept
    auth = authority_name.split()
    n = 0
    while n < len(kept) and n < len(auth) and norm(kept[n].text) == norm(auth[n]):
        n += 1
    if n < 3 or n == len(auth):
        return kept
    # исходный хвост строки за совпавшими словами, слитно: «иэ», «йэкологии», «иэколэеистрированно»
    tail = "".join(norm(w.text) for w in original if w.box[0] >= kept[n - 1].box[2])
    best = 0
    for j in range(1, len(auth) - n + 1):
        prev = "".join(norm(w) for w in auth[n:n + j - 1])
        if prev and len(prev) >= len(tail):
            break  # хвост строки кончился — следующие слова названия на следующей строке шапки
        word = norm(auth[n + j - 1])
        cand = prev + word
        part = tail[:len(cand)]
        # кусок хвоста под новое слово начинается с его буквы: «ЭКОЛОГИИ и» — мусорная «и» после
        # названия не начало «РОССИЙСКОЙ»
        if (tail[len(prev):len(prev) + 1] == word[:1]
                and difflib.SequenceMatcher(None, part, cand[:len(part)]).ratio() >= HEADER_MATCH):
            best = j
    if not best:
        return kept
    added = auth[n:n + best]
    last = kept[n - 1]
    x = last.box[2]
    extra = [Word(w.upper() if last.text.isupper() else w, 100.0, (x, last.box[1], x, last.box[3]), restored=True)
             for w in added]
    return [*kept[:n], *extra]


def clean_lines(lines: list[list[Word]], marks: list[Mark], ds_raw: str | None, width: int, height: int,
                authority_name: str | None = None) -> list[list[Word]]:
    """Строки страницы без текста меток (см. docstring модуля). Порядок строк и слов сохраняется;
    строка, от которой ничего не осталось, выкидывается."""
    if not marks:
        return lines
    boxes = [m.box for m in marks]
    stamps = [m.box for m in marks if m.label == "stamp"]
    support = [
        (box, text.split())
        for label, box, text in ds_blocks(ds_raw or "", width, height)
        if label != "image" and not any(overlap(box, s) >= DS_BLOCK_IN_STAMP for s in stamps)
    ]
    return [cleaned for line in lines if (cleaned := _clean_line(line, boxes, stamps, support, authority_name))]


def _clean_line(line: list[Word], boxes: list[Box], stamps: list[Box], support: list[tuple[Box, list[str]]],
                authority_name: str | None) -> list[Word]:
    touch = [any(overlap(w.box, m) > 0 for m in boxes) for w in line]
    if not any(touch):
        return line
    in_mark = [any(overlap(w.box, m) > 0.5 for m in boxes) for w in line]
    in_stamp = [any(overlap(w.box, m) > 0.5 for m in stamps) for w in line]
    # геометрия печатной строки документа — по словам вне меток (без знаков: у них своя высота)
    outside = [w for w, t in zip(line, touch) if not t and len(norm(w.text)) >= 2]
    line_h = sorted(w.box[3] - w.box[1] for w in outside)[len(outside) // 2] if outside else None
    line_b = sorted(w.box[3] for w in outside)[len(outside) // 2] if outside else None
    kept: list[Word] = []
    dropped = False
    stamp_started = False  # в строке уже выкинут знак штампа (часто — черта его рамки «|»)

    def continues_line(w: Word) -> bool:
        """Продолжает печатную строку документа: та же высота и линия, уверенное, без разрыва после
        предыдущего оставленного слова (обрывок штампа на той же строке Tesseract стоит после
        разрыва или после выкинутого слова)."""
        if line_h is None or not kept or len(norm(w.text)) < 3 or w.conf < CONFIDENT:
            return False
        h = w.box[3] - w.box[1]
        return (abs(h - line_h) <= 0.3 * line_h and abs(w.box[3] - line_b) <= 0.4 * line_h
                and w.box[0] - kept[-1].box[2] <= 2.5 * line_h)

    def spelled(w: Word, v: str) -> Word:
        return w if v == w.text else replace(w, text=v, restored=True)

    for w, inside, stamp, t in zip(line, in_mark, in_stamp, touch):
        r, v = _best_match(w.text, _vocab_at(w.box, support))
        short = len(norm(w.text)) < 3
        # короткое («и», «№») подтверждается только уверенное: такие слова есть в любом блоке
        supported = r >= MATCH and (not short or w.conf >= CONFIDENT)
        if not t:
            kept.append(spelled(w, v) if w.conf < LOW_NEAR_MARK and supported else w)
        elif stamp:
            if not stamp_started and continues_line(w):
                kept.append(spelled(w, v) if supported else w)
            else:
                stamp_started = dropped = True
        elif supported:
            kept.append(spelled(w, v))
        elif inside and not short and w.conf >= CONFIDENT_PRINTED:
            kept.append(w)
        elif not inside and not short and w.conf >= CONFIDENT and outside:
            kept.append(w)
        elif inside and continues_line(w):
            kept.append(w)
        else:
            dropped = True
    if dropped:
        # висящие в конце строки знаки от рамки штампа («ФЕДЕРАЦИИ |», «России) ]»)
        while kept and not norm(kept[-1].text):
            kept.pop()
    if stamps and any(any(overlap(w.box, s) > 0 for s in stamps) for w in line):
        kept = _complete_header(kept, line, authority_name)
    return kept if any(norm(w.text) for w in kept) else []
