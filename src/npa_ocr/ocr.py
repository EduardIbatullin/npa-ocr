"""Точка входа OCR → DOCX для документов без текстового слоя (этап 3).

Гибридная схема движков — обоснование в docs/plan.md:
- PaddleOCR (PPStructureV3) — только для структуры таблиц (строки/столбцы/объединения):
  надёжна (после обхода известного бага с несколькими таблицами на странице — решается
  раздельным распознаванием каждой, см. ocr_table.py), но собственная confidence у
  PaddleOCR ненадёжна для детекции повреждений — на реальном примере явный мусор и
  обрезанный печатью текст получали confidence 0.9+, наравне с чистым текстом.
- RapidOCR (ONNX, eslav_PP-OCRv5) — текст внутри ячеек таблиц: та же модель
  PaddleOCR для восточнославянских языков, просто в ONNX-рантайме вместо нативной
  PaddlePaddle — на той же точности не падает SIGILL на CPU без AVX2 (см.
  _default_cell_text_fn).
- Tesseract — для всего текста вне таблиц: на чистом скане распознаёт почти идеально,
  и его confidence в среднем отделяет текст, повреждённый печатью (0–55 на одном
  реальном примере), от чистого (71–97) — но не всегда: на другом документе
  печать исказила «Президент» до «Ярезидент» с confidence 73 (выше чистого слова
  «В.Путин» с 62) — порог confidence не гарантирует 100% детекции повреждений,
  см. docs/plan.md, «Калибровка порога». Для формулы должности это
  компенсируется нечётким сравнением слов с ожидаемым текстом, не только маской
  (см. _restore_signatory_title_words), но для текста вне известной формулы
  такой страховки нет.

Каждый движок вызывается через инъекцию функции (predict_tables_fn/image_data_fn) —
в реальном использовании это настоящие тяжёлые модели, в юнит-тестах — фейки на
реальных фикстурах, без загрузки моделей на каждый прогон тестов.
"""

from __future__ import annotations

import difflib
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator

import pypdf
from PIL import Image

from npa_ocr import ai_escalation, mark_detection, table_structure
from npa_ocr.ocr_table import TableCell, normalize_header_text, reconstruct_table
from npa_ocr.table_structure import TableStructure

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

DEFAULT_DPI = 200
# Tesseract 0-100; на реальном примере со штампом чистый текст был 71-97, повреждённый
# печатью — 0-55 (см. docs/plan.md) — порог взят в середине этого разрыва.
DEFAULT_CONFIDENCE_THRESHOLD = 60
# Страница без текстового слоя обычно даёт < 20 символов "мусора" (пробелы/артефакты
# экстракции) даже когда реального текста нет — тот же порог, что уже использовался
# при проверке текстового слоя НПА/ИТС в этом проекте (см. docs/plan.md, «Охват»).
DEFAULT_MIN_PAGE_TEXT_CHARS = 20
UNRECOGNIZABLE_MARK = "[нрзб.]"


@dataclass
class TextBlock:
    """Одна строка обычного текста (не в таблице). low_words — флаги низкой уверенности
    по словам text.split(); такие слова остаются в тексте и выделяются в DOCX.
    x0/x1/height — геометрия строки (пиксели страницы) для абзацев и выравнивания."""

    y: float
    text: str
    x0: float = 0.0
    x1: float = 0.0
    height: float = 0.0
    low_words: tuple[bool, ...] = ()


@dataclass
class TableBlock:
    """Одна распознанная таблица страницы."""

    y: float
    bbox: Box
    cells: list[TableCell]
    n_rows: int
    n_cols: int
    dropped_columns: int = 0  # столбцы без текста, удалённые как артефакт структуры (TATR)


PageBlock = TextBlock | TableBlock


@dataclass
class PageResult:
    page_number: int
    ocr_applied: bool  # False — страница уже имела текстовый слой, OCR не запускался
    blocks: list[PageBlock] = field(default_factory=list)
    plain_text: str | None = None  # для ocr_applied=False — текст из текстового слоя как есть
    page_height: float = 0.0  # пиксели; нужна для отбора колонтитулов (0 — не отбирать)


@dataclass
class OcrDocumentResult:
    pages: list[PageResult]
    docx_path: Path
    timings: dict = field(default_factory=dict)
    timings_path: Path | None = None


# --- Проверка памяти ---

# Пик загрузки моделей PaddleOCR и распознавания таблицы на машине с 16 ГБ RAM
# приводил к полному зависанию системы (2026-10-03, журнал Windows: принудительная
# перезагрузка без BSOD). Перед каждой страницей-сканом проверяем свободную память
# и останавливаемся с понятной ошибкой вместо зависания. 0 — отключить проверку.
DEFAULT_MIN_FREE_MEMORY_MB = 2048


class InsufficientMemoryError(RuntimeError):
    pass


def ensure_free_memory(min_free_mb: int, where: str) -> None:
    if min_free_mb <= 0:
        return
    import psutil

    free_mb = psutil.virtual_memory().available / (1024 * 1024)
    if free_mb < min_free_mb:
        raise InsufficientMemoryError(
            f"Свободной памяти {free_mb:.0f} МБ, нужно не меньше {min_free_mb} МБ ({where}). "
            "Распознавание остановлено, чтобы не зависла система. Закройте другие программы "
            "или запустите с меньшим --min-free-mb (0 — отключить проверку)."
        )


# --- PaddleOCR: только таблицы ---


def _extract_table_blocks(table_json: dict) -> list[TableBlock]:
    """table_json — JSON-результат PPStructureV3 (как у save_to_json) с ключом
    table_res_list; каждый элемент — cell_box_list + table_ocr_pred
    (rec_boxes/rec_texts), см. ocr_table.py и tests/unit/fixtures/ocr_tables."""
    blocks: list[TableBlock] = []
    for t in table_json.get("table_res_list", []):
        cell_box_list = t.get("cell_box_list") or []
        if not cell_box_list:
            continue
        rec = t["table_ocr_pred"]
        texts = [normalize_header_text(x) for x in rec["rec_texts"]]
        cells = reconstruct_table(cell_box_list, rec["rec_boxes"], texts)
        n_rows = max(c.row + c.row_span for c in cells)
        n_cols = max(c.col + c.col_span for c in cells)
        x1 = min(b[0] for b in cell_box_list)
        y1 = min(b[1] for b in cell_box_list)
        x2 = max(b[2] for b in cell_box_list)
        y2 = max(b[3] for b in cell_box_list)
        blocks.append(
            TableBlock(y=y1, bbox=(x1, y1, x2, y2), cells=cells, n_rows=n_rows, n_cols=n_cols)
        )
    return blocks


def run_paddle_tables(image_path: Path, predict_fn: Callable[[Path], dict]) -> list[TableBlock]:
    """predict_fn возвращает JSON-совместимый dict (не сам объект результата
    PPStructureV3) — так вызывающий код и тесты не зависят от конкретного API
    тяжёлой модели, только от уже проверенной формы данных."""
    return _extract_table_blocks(predict_fn(image_path))



# --- Tesseract: весь текст вне таблиц ---


def _point_in_any_box(cx: float, cy: float, boxes: list[Box]) -> bool:
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in boxes)


# Канцелярская формула должности подписанта — восстанавливаем из уже известного
# authority_name (подтверждённые данные запроса этапа 1), не угадываем. Это
# структурное правило оформления документов (какое слово-должность идёт перед
# названием органа), не факт о конкретном документе — поэтому не нарушает принцип
# «не восполнять реквизиты не по тексту акта» (см. docs/plan.md, «ФИО
# подписанта» — тот принцип про факты, не про общий формат оформления). Покрывает
# только проверенные на реальных документах случаи этого проекта; для
# неизвестного типа органа — None, замены не будет.
def expected_signatory_title(authority_name: str) -> str | None:
    name = authority_name.strip()
    if name == "Президент Российской Федерации":
        return name
    if name.startswith("Правительство"):
        # Родительный падеж: «Правительство» (ср.р., -о) -> «Правительства».
        # Подтверждено реальным документом — именно это слово («Правительства»)
        # стояло под печатью, см. docs/plan.md, «Печать/подпись», п. 3.
        return "Председатель Правительства" + name[len("Правительство") :]
    if name.startswith(("Федеральное агентство", "Федеральная служба")):
        return "Руководитель"
    # "Министерство X" -> "Министр X" требует родительного падежа названия
    # ведомства ("Министр здравоохранения", не "Министр Министерство
    # здравоохранения") — склонение не выводится надёжно из именительного падежа
    # без риска угадывания текста, а реальных примеров для министерств в этом
    # проекте пока не было. Пока не поддерживаем, см. docs/plan.md.
    return None


_NEAR_MISS_SIMILARITY_THRESHOLD = 0.7


_MIN_PREFIX_MATCH_LENGTH = 4


def _is_near_miss(word: str, expected_word: str) -> bool:
    """Нечёткое сравнение (не только точное совпадение/маска) — нужно потому что
    confidence Tesseract не всегда ловит повреждение печатью: на реальном примере
    «Президент» под печатью было распознано как «Ярезидент» (confidence 73, выше
    порога 60) — не замаскировано как [нрзб.], но явно испорчено. Порог 0.7 по
    difflib.SequenceMatcher на полных словах ловит замену 1-2 символов в
    достаточно длинном слове («Ярезидент» vs «Президент» ≈ 0.89), но не ловит
    обрезание слова печатью (другой реальный случай, Постановление №1430):
    «Правительства»→«Прави», «Федерации»→«Федей» — полное сравнение даёт только
    0.56/0.57 (разная длина режет ratio), хотя для человека очевидно, что это
    начало того же слова. Поэтому отдельно сравниваем word с префиксом
    expected_word той же длины (при len(word) >= 4, чтобы не ловить короткие
    слова случайно) — «прави» vs «правительства»[:5]=«прави» → 1.0,
    «федей» vs «федерации»[:5]=«федер» → 0.8."""
    word_l, expected_l = word.lower(), expected_word.lower()
    full_ratio = difflib.SequenceMatcher(None, word_l, expected_l).ratio()
    if full_ratio >= _NEAR_MISS_SIMILARITY_THRESHOLD:
        return True
    if len(word_l) >= _MIN_PREFIX_MATCH_LENGTH:
        prefix_ratio = difflib.SequenceMatcher(None, word_l, expected_l[: len(word_l)]).ratio()
        if prefix_ratio >= _NEAR_MISS_SIMILARITY_THRESHOLD:
            return True
    return False


def _restore_signatory_title_words(words: list[str], authority_name: str | None) -> list[str]:
    """words — все слова страницы (вне таблиц) в порядке чтения, без разбивки по
    строкам. Разбивка по строкам Tesseract ненадёжна именно в области штампа
    (реальный документ: «Председатель Правительства Российской Федерации» было
    разорвано на две отдельные строки «Председатель [нрзб.]» / «Российской
    [нрзб.] Д.Медведев» — печать сдвинула геометрию слов) — поэтому поиск формулы
    должности ведётся по сквозной последовательности слов страницы, а не внутри
    одной уже сгруппированной строки.

    Окно длиной с формулу принимается, если КАЖДОЕ его слово — это либо
    [нрзб.], либо точное совпадение с ожидаемым словом, либо текстово похожее
    (_is_near_miss) — и при этом среди слов окна есть хотя бы одно ТОЧНОЕ
    совпадение (якорь: подтверждает, что это действительно формула должности, а
    не случайный фрагмент текста — «Российской Федерации» само по себе часто
    встречается и в обычном тексте документа, не только в подписи). Любое слово
    окна, не прошедшее ни точное совпадение, ни нечёткое сравнение, отменяет
    замену всего окна целиком (безопасный отказ, не угадываем). Всё остальное
    (включая ФИО сразу после формулы, даже на той же исходной строке) не
    трогается."""
    if authority_name is None:
        return words
    expected = expected_signatory_title(authority_name)
    if expected is None:
        return words
    expected_words = expected.split()
    n = len(expected_words)
    for start in range(len(words) - n + 1):
        window = words[start : start + n]
        has_exact_match = False
        accepted = True
        for word, expected_word in zip(window, expected_words):
            if word == UNRECOGNIZABLE_MARK:
                continue
            if word.lower() == expected_word.lower():
                has_exact_match = True
                continue
            if not _is_near_miss(word, expected_word):
                accepted = False
                break
        if not accepted or not has_exact_match:
            continue
        return [*words[:start], *expected_words, *words[start + n :]]
    return words


def run_tesseract_text(
    image_path: Path,
    table_bboxes: list[Box],
    *,
    image_data_fn: Callable[[Path], dict],
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    authority_name: str | None = None,
    detect_marks_fn: Callable[[Path, list[Box]], list[mark_detection.Mark]] | None = None,
    recover_covered_text_fn: Callable[[Path], str] | None = None,
) -> list[TextBlock]:
    """image_data_fn возвращает словарь формата pytesseract.image_to_data(...,
    output_type=Output.DICT) (ключи text/conf/left/top/width/height/block_num/
    par_num/line_num). Слова внутри table_bboxes пропускаются — та же область уже
    распознана отдельно через PaddleOCR (run_paddle_tables), иначе текст таблицы
    задвоился бы. Слова с confidence ниже порога заменяются на UNRECOGNIZABLE_MARK,
    а не угадываются — см. docs/plan.md, «Печать/подпись». authority_name —
    если задан, формула должности восстанавливается по всей странице сквозным
    поиском по словам (строки Tesseract у штампа ненадёжны), см.
    _restore_signatory_title_words.

    detect_marks_fn/recover_covered_text_fn — опциональная ИИ-эскалация текста
    под печатью/штампом (см. mark_detection.py, ai_escalation.py, docs/plan.md,
    «ИИ-эскалация»). По умолчанию (None) ничего не меняется — поведение то же,
    что и раньше (низкая уверенность просто выделяется в DOCX). Когда заданы:
    detect_marks_fn находит и классифицирует подозрительные области по уже
    вычисленным bbox низкоуверенных слов (без лишних вызовов VLM на обычных
    страницах без таких областей); для меток с covers_text=True
    низкоуверенные слова внутри области пробуют восстановиться через
    recover_covered_text_fn — результат всё равно остаётся с low=True
    (ИИ-восстановленный текст — не замена честной неуверенности слепым
    доверием, подлежит проверке человеком, как и раньше); слова внутри меток
    типа «электронная_подпись»/«штамп_регистрации» с covers_text=False
    (собственное содержимое метки — не текст документа, а служебная отметка)
    исключаются из текста целиком, см. примеры (Постановление №710,
    регистрационный штамп Минюста) в docs/plan.md."""
    data = image_data_fn(image_path)
    lines: dict[tuple[int, int, int], list[tuple[float, float, str, float, float, bool]]] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        cx, cy = x + w / 2, y + h / 2
        if _point_in_any_box(cx, cy, table_bboxes):
            continue
        conf = data["conf"][i]
        low = conf < confidence_threshold
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((y, x, word, w, h, low))

    line_items = []  # (y, key, [(x, word), ...]) — порядок строк на странице
    for key, words in lines.items():
        words.sort(key=lambda w: w[1])  # по x внутри строки
        y = min(w[0] for w in words)
        line_items.append((y, key, words))
    line_items.sort(key=lambda item: item[0])

    if detect_marks_fn is not None:
        line_items = _apply_mark_escalation(
            line_items,
            image_path,
            detect_marks_fn=detect_marks_fn,
            recover_covered_text_fn=recover_covered_text_fn,
        )

    flat_words = [w[2] for _, _, words in line_items for w in words]
    flat_low = [w[5] for _, _, words in line_items for w in words]
    restored_words = _restore_signatory_title_words(flat_words, authority_name)
    # слово, заменённое восстановленной формулой, уже не «сомнительное»
    flat_low = [
        low and new == old for low, old, new in zip(flat_low, flat_words, restored_words)
    ]

    blocks: list[TextBlock] = []
    pos = 0
    for y, _key, words in line_items:
        restored = restored_words[pos : pos + len(words)]
        low_flags = tuple(flat_low[pos : pos + len(words)])
        pos += len(words)
        blocks.append(
            TextBlock(
                y=y,
                text=" ".join(restored),
                x0=min(w[1] for w in words),
                x1=max(w[1] + w[3] for w in words),
                height=max(w[4] for w in words),
                low_words=low_flags,
            )
        )
    return blocks


_ESCALATION_CROP_PADDING = 25  # пикселей при 200 DPI — запас вокруг области метки при вырезании для ИИ-эскалации

_WordTuple = tuple[float, float, str, float, float, bool]  # y, x, word, w, h, low


def _word_center(word: _WordTuple) -> tuple[float, float]:
    y, x, _word, w, h, _low = word
    return x + w / 2, y + h / 2


def _apply_mark_escalation(
    line_items: list[tuple[float, tuple[int, int, int], list[_WordTuple]]],
    image_path: Path,
    *,
    detect_marks_fn: Callable[[Path, list[Box]], list[mark_detection.Mark]],
    recover_covered_text_fn: Callable[[Path], str] | None,
) -> list[tuple[float, tuple[int, int, int], list[_WordTuple]]]:
    """См. docstring run_tesseract_text. Детектор вызывается один раз на всю
    страницу (не на каждую строку) — экономит вызовы VLM, т.к. один штамп
    обычно задевает несколько соседних строк сразу.

    На реальном документе (Приказ №191) выяснилось эмпирически: если резать
    область метки на мелкие кропы по отдельным словам/строкам и делать
    отдельный вызов ИИ на каждый — модель видит крошечные обрезки без
    видимого контекста и выдаёт мусор (9 разных вызовов на один штамп
    регистрации дали 9 несвязанных обрывков). Поэтому восстановление теперь
    делает ОДИН вызов на ВСЮ область метки целиком (_recover_mark_region), а
    возвращённые слова просто по порядку раздаются на все исходные
    низкоуверенные слова внутри метки (по чтению — сверху вниз, слева
    направо), не по строкам — хватит слов или нет, заранее не известно, и
    это не страшно (см. _consume_escalated_words)."""
    low_word_boxes = [
        (x, y, x + w, y + h) for _, _, words in line_items for (y, x, _word, w, h, low) in words if low
    ]
    print(f"[ИИ-эскалация] низкоуверенных слов на странице: {len(low_word_boxes)}", file=sys.stderr)
    if not low_word_boxes:
        return line_items
    marks = detect_marks_fn(image_path, low_word_boxes)
    print(
        f"[ИИ-эскалация] найдено областей: {len(marks)} — "
        + "; ".join(f"{m.mark_type} (covers_text={m.covers_text})" for m in marks),
        file=sys.stderr,
    )
    if not marks:
        return line_items

    discard_bboxes = [
        m.bbox
        for m in marks
        if m.mark_type in ("электронная_подпись", "штамп_регистрации") and not m.covers_text
    ]
    escalate_marks = [m for m in marks if m.covers_text]

    mark_words: list[list[str] | None] = [None] * len(escalate_marks)
    if escalate_marks and recover_covered_text_fn is not None:
        page_image = Image.open(image_path)
        for i, mark in enumerate(escalate_marks):
            mark_words[i] = _recover_mark_region(page_image, mark.bbox, recover_covered_text_fn)
    mark_iters = [iter(words) if words else None for words in mark_words]

    result = []
    for _y, key, words in line_items:
        kept = words
        if discard_bboxes:
            kept = [w for w in kept if not _point_in_any_box(*_word_center(w), discard_bboxes)]
        if escalate_marks:
            kept = _consume_escalated_words(kept, escalate_marks, mark_iters)
        if kept:
            result.append((min(w[0] for w in kept), key, kept))
    return result


def _consume_escalated_words(
    words: list[_WordTuple],
    escalate_marks: list[mark_detection.Mark],
    mark_iters: list[Iterator[str] | None],
) -> list[_WordTuple]:
    """Низкоуверенные слова, попадающие в область метки с covers_text=True,
    заменяются следующим словом из уже восстановленного для этой метки
    текста (единый вызов на всю метку — см. _apply_mark_escalation), пока
    слова не кончатся; если их не хватило или эскалация для метки не
    удалась — исходное (сырое) слово остаётся как было, не теряем информацию
    молча. Результат всё равно помечается low=True, см. docstring
    run_tesseract_text."""
    result: list[_WordTuple] = []
    for word in words:
        if not word[5]:
            result.append(word)
            continue
        cx, cy = _word_center(word)
        mark_index = next(
            (i for i, m in enumerate(escalate_marks) if m.bbox[0] <= cx <= m.bbox[2] and m.bbox[1] <= cy <= m.bbox[3]),
            None,
        )
        iterator = mark_iters[mark_index] if mark_index is not None else None
        recovered_word = next(iterator, None) if iterator is not None else None
        if recovered_word is None:
            result.append(word)
            continue
        y, x, _text, w, h, _low = word
        result.append((y, x, recovered_word, w, h, True))
    return result


def _recover_mark_region(
    page_image: Image.Image,
    mark_bbox: Box,
    recover_fn: Callable[[Path], str],
) -> list[str] | None:
    """Один вызов ИИ-эскалации на всю область найденной метки (с запасом
    _ESCALATION_CROP_PADDING) — см. docstring _apply_mark_escalation, почему
    не по частям. None — эскалация не дала результата (сеть/таймаут/пустой
    ответ) — вызывающая сторона оставляет исходные слова как есть. Любая
    ошибка recover_fn ловится широко (Exception) намеренно — это
    необязательное улучшение, оно не должно уронить основной пайплайн
    распознавания."""
    x0 = mark_bbox[0] - _ESCALATION_CROP_PADDING
    y0 = mark_bbox[1] - _ESCALATION_CROP_PADDING
    x1 = mark_bbox[2] + _ESCALATION_CROP_PADDING
    y1 = mark_bbox[3] + _ESCALATION_CROP_PADDING
    box = (
        max(0, int(x0)),
        max(0, int(y0)),
        min(page_image.width, int(x1)),
        min(page_image.height, int(y1)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    crop = page_image.crop(box)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        crop.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        text = recover_fn(tmp_path)
    except Exception as exc:
        print(f"[ИИ-эскалация] восстановление не удалось: {exc}", file=sys.stderr)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)
    words = text.split()
    print(f"[ИИ-эскалация] восстановлено для области {mark_bbox}: {' '.join(words)!r}", file=sys.stderr)
    return words or None


_TABLE_CROP_PADDING = 20  # пикселей при 200 DPI — запас вокруг таблицы при вырезании


LOW_CONFIDENCE_THRESHOLD = 0.8  # уверенность строки распознавания (0–1): ниже — выделяется в DOCX


@lru_cache(maxsize=1)
def _paddle_runtime_usable() -> bool:
    """Можно ли вообще импортировать PaddlePaddle в этом окружении.

    CPU-колёса paddlepaddle собраны с AVX2; на процессорах без AVX2 (например,
    Intel Xeon E5-2696 v2 / Ivy Bridge) процесс падает с SIGILL ещё на
    `import paddle` — это не Python-исключение, try/except его не ловит.
    Код возврата подпроцесса -4 (= сигнал 4, SIGILL) как раз то, что видит
    пользователь в «Распознавание завершилось с кодом -4», когда TATR нашёл
    таблицу и пайплайн впервые тянет PPStructureV3 (структура таблиц).

    Текста ячеек (`_default_cell_text_fn`) это больше не касается — он теперь на
    RapidOCR/ONNX и падения не знает независимо от AVX2; проверка осталась только
    для PPStructureV3 (`_default_predict_tables_fn`, уточнение регионов таблиц).
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        if "avx2" not in cpuinfo.lower():
            return False
    except OSError:
        pass
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import paddle"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _default_cell_text_fn() -> Callable[[Path], tuple[str, tuple[bool, ...]]]:
    """Распознавание одного кропа ячейки: RapidOCR (ONNX Runtime), модель
    eslav_PP-OCRv5_rec_mobile — та же модель PaddleOCR для восточнославянских
    языков (рус./укр./белор.), что и раньше вызывалась напрямую через PaddlePaddle,
    просто в ONNX-рантайме. Переход сделан после того, как нативный
    PaddleOCR(lang="ru") на кропах ячеек стал падать SIGILL на сервере со старым CPU
    без AVX2 — enable_mkldnn=False не спасало, проблема не в MKL-DNN-бэкенде, а
    глубже в самой сборке PaddlePaddle; ONNX Runtime сам адаптируется к набору
    инструкций CPU. Точность на реальных документах проверена и совпадает с
    PaddleOCR (см. docs/plan.md)."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.ESLAV,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        }
    )

    def read(crop_path: Path) -> tuple[str, tuple[bool, ...]]:
        result = engine(str(crop_path))
        items: list[tuple[Box, str, float]] = []
        if result.boxes is not None:
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                rect = (
                    float(box[:, 0].min()),
                    float(box[:, 1].min()),
                    float(box[:, 0].max()),
                    float(box[:, 1].max()),
                )
                items.append((rect, text, float(score)))
        lines = table_structure.join_fragment_lines(items)
        text = "\n".join(line for line, _ in lines)
        return text, tuple(score < LOW_CONFIDENCE_THRESHOLD for _, score in lines)

    return read


def _tesseract_text_predict_fn() -> Callable[[Path], dict]:
    """Замена PPStructureV3 для текста таблиц, когда Paddle недоступен (нет AVX2).

    Возвращает JSON той же формы, что ждёт `_unique_text_blocks` (rec_boxes/rec_texts
    внутри table_res_list), но без cell_box_list — области таблиц тогда остаются
    за TATR (см. table_regions_for_page). Качество текста ячеек ниже, чем у Paddle,
    зато документ доходит до конца вместо SIGILL.
    """

    def predict(image_path: Path) -> dict:
        data = _default_image_data_fn(image_path)
        boxes: list[list[float]] = []
        texts: list[str] = []
        for text, conf, left, top, width, height in zip(
            data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
        ):
            word = (text or "").strip()
            if not word or int(conf) < 0:
                continue
            boxes.append([float(left), float(top), float(left + width), float(top + height)])
            texts.append(word)
        return {"table_res_list": [{"table_ocr_pred": {"rec_boxes": boxes, "rec_texts": texts}}]}

    return predict


TEXT_PREDICT_MAX_SIDE = 3000  # PPStructureV3 падает (сегфолт) на кропах больше ~3900 px по стороне
CELL_UPSCALE = 2
CELL_PADDING_PX = 12


def _prepare_cell_image(cell: Image.Image) -> Image.Image:
    """Белое поле и увеличение кропа ячейки: мелкий текст шапки распознаётся точнее,
    когда символы крупнее, а текст не прижат к краю."""
    from PIL import ImageOps

    padded = ImageOps.expand(cell, border=CELL_PADDING_PX, fill="white")
    return padded.resize((padded.width * CELL_UPSCALE, padded.height * CELL_UPSCALE), Image.LANCZOS)


def _recognize_cells(
    crop: Image.Image,
    structure: TableStructure,
    texts: list[tuple[Box, str]],
    cell_text_fn: Callable[[Path], tuple[str, tuple[bool, ...]]],
) -> tuple[list[tuple[Box, str]], dict[tuple[int, int], tuple[bool, ...]]]:
    """Каждая ячейка с текстом вырезается по границам своего текста и распознаётся
    отдельно; результат возвращается как текстовый блок в той же позиции. Ячейки с
    уверенностью ниже LOW_CONFIDENCE_THRESHOLD возвращаются отдельным набором."""
    refined: list[tuple[Box, str]] = []
    low: dict[tuple[int, int], tuple[bool, ...]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for (r, c), blocks in table_structure.cell_text_boxes(structure, texts).items():
            box = table_structure.text_crop_box(blocks)
            if box is None:
                continue
            cell_path = Path(tmp_dir) / f"cell_{r}_{c}.png"
            _prepare_cell_image(crop.crop(tuple(int(v) for v in box))).save(cell_path)
            text, low_lines = cell_text_fn(cell_path)
            refined.append((box, normalize_header_text(text)))
            if any(low_lines):
                low[(r, c)] = low_lines
    return refined, low


def _box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _unique_text_blocks(text_json: dict) -> list[tuple[Box, str]]:
    """PaddleOCR на одном кропе возвращает несколько записей таблиц; один и тот же
    текст приходит с почти совпадающими боксами. Оставляем один блок на каждое
    совпадение текста с перекрытием боксов IoU ≥ 0.5."""
    kept: list[tuple[Box, str]] = []
    for t in text_json.get("table_res_list", []):
        rec = t.get("table_ocr_pred", {})
        for box, text in zip(rec.get("rec_boxes", []), rec.get("rec_texts", [])):
            box = tuple(box)
            if any(k_text == text and _box_iou(k_box, box) >= 0.5 for k_box, k_text in kept):
                continue
            kept.append((box, normalize_header_text(text)))
    return kept


def _merge_overlapping_regions(boxes: list[Box]) -> list[Box]:
    """Пересекающиеся области (фрагменты одной таблицы) объединяются в одну."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if _box_iou(a, b) > 0 or (min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])):
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def paddle_table_regions(page: Image.Image, predict_fn: Callable[[Path], dict]) -> list[Box]:
    """Области таблиц страницы по PaddleOCR (PPStructureV3): TATR на страницах с несколькими
    таблицами отдает одну область на всю страницу, PaddleOCR — отдельные таблицы."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        page_path = Path(tmp_dir) / "page.png"
        page.save(page_path)
        data = predict_fn(page_path)
    boxes: list[Box] = []
    for t in data.get("table_res_list", []):
        cells = t.get("cell_box_list") or []
        if cells:
            boxes.append((
                min(b[0] for b in cells), min(b[1] for b in cells),
                max(b[2] for b in cells), max(b[3] for b in cells),
            ))
    return _merge_overlapping_regions(boxes)


def table_regions_for_page(page: Image.Image, predict_fn: Callable[[Path], dict]) -> list[Box]:
    """Области таблиц страницы: сначала дешёвый TATR (table_structure.detect_table_regions,
    уже загружен в памяти — см. load_tatr в ocr_document), PaddleOCR (PPStructureV3, ~11
    тяжёлых моделей PaddleX) вызывается только если TATR нашёл хотя бы одну область-кандидат.

    До этого PaddleOCR стоял первым (`paddle_table_regions(...) or detect_table_regions(...)`),
    поэтому даже страница без единой таблицы платила полную загрузку PPStructureV3 — на
    машине с 16 ГБ RAM (а тем более в контейнере с меньшим лимитом) это надёжно приводило
    к OOM ещё до начала распознавания текста. TATR на страницах с несколькими таблицами
    мёрджит их в одну область (см. paddle_table_regions) — поэтому при находке TATR всё
    равно предпочитается более точный результат PaddleOCR, если он непустой; свой результат
    TATR остаётся резервом, если PaddleOCR с ним не согласился (вернул пусто).

    Если runtime Paddle недоступен (SIGILL на CPU без AVX2, см. `_paddle_runtime_usable`),
    PaddleOCR не вызываем вообще — иначе подпроцесс распознавания умрёт с кодом -4.
    """
    tatr_found = table_structure.detect_table_regions(page)
    if not tatr_found:
        return []
    if not _paddle_runtime_usable():
        return tatr_found
    return paddle_table_regions(page, predict_fn) or tatr_found


class _LazyEngine:
    """Движок создаётся при первом вызове и выгружается по release(), а не сразу при
    старте документа — страницы с текстовым слоем (без таблиц-сканов) вообще не
    должны платить загрузку моделей. Было особенно важно, когда и регионы таблиц
    (PPStructureV3), и текст ячеек были на PaddleOCR — одновременно оба не
    помещались в память (контейнер 12 ГБ падал по OOM), поэтому грузились по
    очереди. Текст ячеек теперь на RapidOCR (лёгкий ONNX, не конфликтует по
    памяти с PPStructureV3), но ленивая загрузка оставлена и для него — дешевле и
    не требует разбирать, нужна ли она ещё."""

    def __init__(self, factory: Callable[[], Callable]) -> None:
        self._factory = factory
        self._engine: Callable | None = None

    def __call__(self, *args):
        if self._engine is None:
            self._engine = self._factory()
        return self._engine(*args)

    def release(self) -> None:
        self._engine = None
        gc.collect()


def run_tatr_tables(
    image_path: Path,
    text_predict_fn: Callable[[Path], dict],
    *,
    cell_text_fn: Callable[[Path], str] | None = None,
    region_fn: Callable[[Image.Image], list[Box]] | None = None,
    structure_fn: Callable[[Image.Image], TableStructure] | None = None,
    release_structure: Callable[[], None] | None = None,
    release_cells: Callable[[], None] | None = None,
) -> list[TableBlock]:
    """Таблицы страницы: регионы и структура (строки/столбцы/объединения) — TATR;
    текст ячеек — PaddleOCR по вырезанной таблице (text_predict_fn). Каждая таблица
    вырезается с отступом, структура и текст считаются по одному и тому же кропу.
    Столбцы и строки без текста удаляются как артефакты детекции.

    Порядок по этапам: сначала все регионы и текст кропов (text_predict_fn) и структура,
    затем release_structure() — выгрузка PPStructureV3, потом текст ячеек (cell_text_fn)
    и release_cells(). Так тяжёлые движки не находятся в памяти одновременно."""
    structure_fn = structure_fn or table_structure.detect_structure

    with Image.open(image_path) as opened:
        page = opened.convert("RGB")
    regions = region_fn(page) if region_fn is not None else table_structure.detect_table_regions(page)
    pending: list[tuple] = []
    for x1, y1, x2, y2 in regions:
        crop_box = (
            max(0, int(x1) - _TABLE_CROP_PADDING),
            max(0, int(y1) - _TABLE_CROP_PADDING),
            min(page.width, int(x2) + _TABLE_CROP_PADDING),
            min(page.height, int(y2) + _TABLE_CROP_PADDING),
        )
        crop = page.crop(crop_box)
        scale = min(1.0, TEXT_PREDICT_MAX_SIDE / max(crop.size))
        text_img = crop
        if scale < 1.0:
            text_img = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
        with tempfile.TemporaryDirectory() as tmp_dir:
            crop_path = Path(tmp_dir) / "table_crop.png"
            text_img.save(crop_path)
            text_json = text_predict_fn(crop_path)
        texts = [
            (tuple(v / scale for v in box), text) for box, text in _unique_text_blocks(text_json)
        ]

        structure = structure_fn(crop)
        structure, dropped_cols = table_structure.drop_empty_columns(structure, texts)
        structure = table_structure.expand_rows_to_text(structure, texts)
        structure = table_structure.drop_empty_rows(structure, texts)
        pending.append((crop_box, (x1, y1, x2, y2), crop, texts, structure, dropped_cols))

    if release_structure is not None:
        release_structure()

    blocks: list[TableBlock] = []
    for crop_box, (x1, y1, x2, y2), crop, texts, structure, dropped_cols in pending:
        low_cells: dict[tuple[int, int], tuple[bool, ...]] = {}
        if cell_text_fn is not None:
            texts, low_cells = _recognize_cells(crop, structure, texts, cell_text_fn)
        cells, n_rows, n_cols = table_structure.build_cells(structure, texts)
        if not cells or not table_structure.is_plausible_table(cells, n_rows, n_cols):
            continue
        for cell in cells:
            cell.low_lines = low_cells.get((cell.row, cell.col), ())
        content = [box for (_r, _c), blk in table_structure.cell_text_boxes(structure, texts).items() for box, _ in blk]
        if content:
            bx1 = min(b[0] for b in content) + crop_box[0]
            by1 = min(b[1] for b in content) + crop_box[1]
            bx2 = max(b[2] for b in content) + crop_box[0]
            by2 = max(b[3] for b in content) + crop_box[1]
        else:
            bx1, by1, bx2, by2 = x1, y1, x2, y2
        blocks.append(
            TableBlock(
                y=by1,
                bbox=(bx1, by1, bx2, by2),
                cells=cells,
                n_rows=n_rows,
                n_cols=n_cols,
                dropped_columns=dropped_cols,
            )
        )
    if release_cells is not None:
        release_cells()
    return blocks


def ocr_page(
    image_path: Path,
    *,
    predict_tables_fn: Callable[[Path], dict] | None = None,
    image_data_fn: Callable[[Path], dict],
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    authority_name: str | None = None,
    table_blocks_fn: Callable[[Path], list[TableBlock]] | None = None,
    detect_marks_fn: Callable[[Path, list[Box]], list[mark_detection.Mark]] | None = None,
    recover_covered_text_fn: Callable[[Path], str] | None = None,
) -> list[PageBlock]:
    """Таблицы и обычный текст страницы, слитые в порядке чтения (сверху вниз).
    Таблицы — через table_blocks_fn (TATR) или predict_tables_fn (прежний путь PaddleOCR).
    detect_marks_fn/recover_covered_text_fn — см. run_tesseract_text."""
    if table_blocks_fn is not None:
        tables = table_blocks_fn(image_path)
    else:
        tables = run_paddle_tables(image_path, predict_tables_fn)
    text_blocks = run_tesseract_text(
        image_path,
        [t.bbox for t in tables],
        image_data_fn=image_data_fn,
        confidence_threshold=confidence_threshold,
        authority_name=authority_name,
        detect_marks_fn=detect_marks_fn,
        recover_covered_text_fn=recover_covered_text_fn,
    )
    blocks: list[PageBlock] = [*tables, *text_blocks]
    blocks.sort(key=lambda b: b.y)
    return blocks


# --- Сборка DOCX ---


def _add_table(doc, block: TableBlock) -> None:
    """На реальном документе (Постановление №1430) встретилась таблица, где
    заявленные span'ы двух разных ячеек конфликтуют геометрически (одна уже
    объединена по вертикали, другая пытается объединить соседние ячейки по
    горизонтали через неё) — это даёт «Т-образную» область, которую python-docx
    принципиально не может представить одним прямоугольным объединением
    (`InvalidSpanError: requested span not rectangular`). Это artефакт
    несовершенной кластеризации геометрии реальной сложной таблицы, не баг с
    очевидным полным исправлением — вместо падения всего документа (потеря OCR
    всех страниц) такая ячейка остаётся неотъединённой: текст не теряется, но
    визуально ячейка не сливается с соседними, как в оригинале."""
    from docx.enum.text import WD_COLOR_INDEX
    from docx.exceptions import InvalidSpanError

    table = doc.add_table(rows=block.n_rows, cols=block.n_cols)
    table.style = "Table Grid"
    merged: set[tuple[int, int]] = set()
    for cell in block.cells:
        if (cell.row, cell.col) in merged:
            continue
        target = table.cell(cell.row, cell.col)
        if cell.row_span > 1 or cell.col_span > 1:
            bottom_right = table.cell(
                cell.row + cell.row_span - 1, cell.col + cell.col_span - 1
            )
            try:
                target = target.merge(bottom_right)
            except InvalidSpanError:
                target = table.cell(cell.row, cell.col)
            else:
                for r in range(cell.row, cell.row + cell.row_span):
                    for c in range(cell.col, cell.col + cell.col_span):
                        merged.add((r, c))
        lines = cell.text.split("\n")
        if any(cell.low_lines) and len(lines) == len(cell.low_lines):
            target.text = ""
            paragraph = target.paragraphs[0]
            for i, (line, low) in enumerate(zip(lines, cell.low_lines)):
                if i:
                    paragraph.add_run().add_break()
                run = paragraph.add_run(line)
                if low:
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            target.text = cell.text


_INDENT_MIN_PX = 40  # ~5 мм при 200 DPI — отступ первой строки абзаца
_SHORT_LINE_RATIO = 0.12  # строка, не доходящая до правого края на эту долю ширины, — конец абзаца
_CENTER_TOLERANCE_RATIO = 0.015
_CENTER_MAX_WIDTH_RATIO = 0.95
_PARAGRAPH_GAP_RATIO = 1.5  # зазор больше этой доли высоты строки — новый абзац


def _block_segments(b: TextBlock) -> list[tuple[str, bool]]:
    words = b.text.split(" ")
    flags = b.low_words if len(b.low_words) == len(words) else (False,) * len(words)
    segments: list[tuple[str, bool]] = []
    for i, (word, low) in enumerate(zip(words, flags)):
        if i:
            segments.append((" ", False))
        segments.append((word, low))
    return segments


def _paragraph_segments(group: list[TextBlock]) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    for i, b in enumerate(group):
        if i:
            segments.append((" ", False))
        segments.extend(_block_segments(b))
    return segments


def _text_paragraphs(run: list[TextBlock], dpi: int) -> list[tuple[str, str, float, list[tuple[str, bool]]]]:
    """Склеивает строки одного непрерывного текстового участка в абзацы по геометрии.
    Возвращает (текст, выравнивание 'left'|'center', отступ первой строки в пт, сегменты
    с флагами низкой уверенности). Жирный/курсив/сноски по Tesseract не определяются."""
    from statistics import median

    if not run:
        return []
    if not all(b.x1 > b.x0 for b in run):
        return [(b.text, "left", 0.0, _paragraph_segments([b])) for b in run]

    area_left = min(b.x0 for b in run)
    area_right = max(b.x1 for b in run)
    area_width = area_right - area_left
    area_center = (area_left + area_right) / 2
    line_height = median(b.height for b in run if b.height > 0) if any(b.height > 0 for b in run) else 0.0

    def is_centered(b: TextBlock) -> bool:
        width = b.x1 - b.x0
        return (
            abs((b.x0 + b.x1) / 2 - area_center) <= _CENTER_TOLERANCE_RATIO * area_width
            and width <= _CENTER_MAX_WIDTH_RATIO * area_width
        )

    def px_to_pt(px: float) -> float:
        return px * 72 / dpi

    paragraphs: list[list[TextBlock]] = []
    prev: TextBlock | None = None
    for b in run:
        centered = is_centered(b)
        if prev is None:
            new_paragraph = True
        elif centered or is_centered(prev):
            new_paragraph = True
        elif b.x0 - area_left > _INDENT_MIN_PX:
            new_paragraph = True
        elif prev.x1 < area_right - _SHORT_LINE_RATIO * area_width:
            new_paragraph = True
        elif line_height and b.y - (prev.y + prev.height) > _PARAGRAPH_GAP_RATIO * line_height:
            new_paragraph = True
        else:
            new_paragraph = False
        if new_paragraph:
            paragraphs.append([b])
        else:
            paragraphs[-1].append(b)
        prev = b

    result: list[tuple[str, str, float, list[tuple[str, bool]]]] = []
    for group in paragraphs:
        first = group[0]
        text = " ".join(b.text for b in group)
        segments = _paragraph_segments(group)
        if is_centered(first):
            result.append((text, "center", 0.0, segments))
        else:
            indent = first.x0 - area_left
            first_line = px_to_pt(indent) if indent > _INDENT_MIN_PX else 0.0
            result.append((text, "left", first_line, segments))
    return result


_HEADER_FOOTER_BAND_RATIO = 0.06
_PAGE_NUMBER_RE = re.compile(r"^[\s\-–—.]*\d{1,4}[\s\-–—.]*$")


def _normalize_for_repeat(text: str) -> str:
    return re.sub(r"\d", "#", text.strip().lower())


def remove_running_headers_and_footers(pages: list[PageResult]) -> list[PageResult]:
    """Убирает колонтитулы: строки в верхней/нижней полосе страницы (6% высоты), которые
    либо состоят только из номера страницы, либо повторяются (с учётом цифр) на
    половине и более OCR-страниц документа. Таблицы и строки вне полос не трогаются."""
    ocr_pages = [p for p in pages if p.ocr_applied and p.page_height > 0]
    if not ocr_pages:
        return pages

    def in_band(block: TextBlock, height: float) -> bool:
        top_limit = _HEADER_FOOTER_BAND_RATIO * height
        bottom_limit = (1 - _HEADER_FOOTER_BAND_RATIO) * height
        return block.y < top_limit or block.y + block.height > bottom_limit

    repeats: dict[str, int] = {}
    for page in ocr_pages:
        seen = {
            _normalize_for_repeat(b.text)
            for b in page.blocks
            if isinstance(b, TextBlock) and in_band(b, page.page_height)
        }
        for key in seen:
            repeats[key] = repeats.get(key, 0) + 1
    threshold = max(2, len(ocr_pages) // 2)

    result: list[PageResult] = []
    for page in pages:
        if not page.ocr_applied or page.page_height <= 0:
            result.append(page)
            continue
        kept = []
        for block in page.blocks:
            if isinstance(block, TextBlock) and in_band(block, page.page_height):
                key = _normalize_for_repeat(block.text)
                if _PAGE_NUMBER_RE.match(block.text.strip()) or repeats.get(key, 0) >= threshold:
                    continue
            kept.append(block)
        result.append(PageResult(
            page_number=page.page_number,
            ocr_applied=page.ocr_applied,
            blocks=kept,
            plain_text=page.plain_text,
            page_height=page.page_height,
        ))
    return result


def build_docx(pages: list[PageResult], out_path: Path, *, dpi: int = DEFAULT_DPI) -> None:
    """Страницы с готовым текстовым слоем (ocr_applied=False) — текст как есть, без
    изменений. OCR-страницы — блоки по порядку чтения: текстовые строки склеиваются
    в абзацы по геометрии (отступ первой строки, выравнивание по центру, короткая
    строка и зазор как конец абзаца), таблица — настоящей Word-таблицей с
    объединением ячеек (row_span/col_span), не текстовой имитацией."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    from docx.shared import Pt

    doc = Document()
    has_low_confidence = any(
        (isinstance(block, TableBlock) and any(any(cell.low_lines) for cell in block.cells))
        or (isinstance(block, TextBlock) and any(block.low_words))
        for page in pages
        if page.ocr_applied
        for block in page.blocks
    )
    if has_low_confidence:
        doc.add_paragraph(
            "Жёлтым выделено текст, распознанный с низкой уверенностью: проверьте его по скану."
        )
    for i, page in enumerate(pages):
        if i > 0:
            doc.add_page_break()
        if not page.ocr_applied:
            doc.add_paragraph(page.plain_text or "")
            continue
        run: list[TextBlock] = []

        def flush_run() -> None:
            for _text, align, first_line, segments in _text_paragraphs(run, dpi):
                para = doc.add_paragraph()
                for segment, low in segments:
                    word_run = para.add_run(segment)
                    if low:
                        word_run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                if align == "center":
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif first_line:
                    para.paragraph_format.first_line_indent = Pt(first_line)
            run.clear()

        for block in page.blocks:
            if isinstance(block, TextBlock):
                run.append(block)
            else:
                flush_run()
                _add_table(doc, block)
        flush_run()
    doc.save(str(out_path))


# --- Реальные движки по умолчанию (тяжёлые — только здесь, не в логике выше) ---


def _default_predict_tables_fn() -> Callable[[Path], dict]:
    """Создаёт PPStructureV3 один раз (инициализация моделей — секунды-минуты) и
    возвращает функцию для переиспользования на всех страницах документа.
    enable_mkldnn=False — обход падения движка на этой машине, см. docs/plan.md.
    Формулы/диаграммы/печати отключены — не нужны (формулы оставляем как есть внутри
    текста, печати не распознаём отдельно) и не стоит тратить время/память на них."""
    os.environ.setdefault("FLAGS_use_mkldnn", "false")
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        lang="ru",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        enable_mkldnn=False,
    )

    def predict(image_path: Path) -> dict:
        # use_table_orientation_classify=False — без этого PPStructureV3 лениво
        # строит ВТОРОЙ, полностью отдельный движок детекции+распознавания текста
        # (~4 тяжёлые модели, включая server-tier детектор) специально для проверки
        # поворота каждой отдельной таблицы, даже когда страница уже проверена на
        # поворот целиком (use_doc_orientation_classify=True выше). Проверено на
        # двух реальных фикстурах (простая таблица 21 ячейка, сложная с
        # объединёнными ячейками и двухуровневой шапкой 175 ячеек/219 текстов) —
        # table_res_list побайтово идентичен с этим параметром и без него, см.
        # docs/plan.md, «Реализация». Компромисс: теряется детекция
        # поворота ИНДИВИДУАЛЬНОЙ таблицы независимо от поворота страницы —
        # такого случая в документах этого проекта пока не встречалось.
        results = list(pipeline.predict(str(image_path), use_table_orientation_classify=False))
        # save_to_json — уже проверенный путь получения JSON-формы результата
        # (использовался во всех экспериментах этапа 3), не полагаемся на
        # недокументированный прямой доступ к внутреннему объекту.
        with tempfile.TemporaryDirectory() as tmp:
            results[0].save_to_json(tmp)
            json_path = next(Path(tmp).glob("*.json"))
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)

    return predict


# Установщик winget (tesseract-ocr.tesseract) кладёт бинарник сюда, но не всегда
# добавляет его в PATH текущего процесса — см. docs/plan.md, «Установка».
_WINDOWS_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _resolve_tesseract_cmd() -> str | None:
    import shutil

    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        return env_cmd
    if shutil.which("tesseract"):
        return None  # уже на PATH, pytesseract найдёт сам
    if Path(_WINDOWS_TESSERACT_DEFAULT).is_file():
        return _WINDOWS_TESSERACT_DEFAULT
    return None


def _default_image_data_fn(image_path: Path) -> dict:
    import pytesseract
    from PIL import Image

    tesseract_cmd = _resolve_tesseract_cmd()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    # Явно закрываем файл (не полагаемся на то, что PIL сделает это сам) — на
    # Windows незакрытый хендл не даёт удалить временный PNG после OCR, это
    # уже ловили на практике (PermissionError при уборке tempfile.TemporaryDirectory).
    with Image.open(image_path) as img:
        img.load()
        return pytesseract.image_to_data(img, lang="rus", output_type=pytesseract.Output.DICT)


def _default_detect_marks_fn(image_path: Path, low_word_boxes: list[Box]) -> list[mark_detection.Mark]:
    """Детектор по умолчанию: эвристика mark_detection.cluster_candidate_regions
    + классификация через настоящую Ollama (ai_escalation.classify_mark). См.
    docs/plan.md, «ИИ-эскалация»."""
    return mark_detection.detect_marks(image_path, low_word_boxes, classify_fn=ai_escalation.classify_mark)


# --- Постраничное сохранение и возобновление ---

# Сырые результаты страницы (блоки с координатами, таблицы, флаги уверенности) пишутся
# на диск сразу после распознавания страницы. Сборка DOCX читает их, поэтому после
# остановки (нехватка памяти, сбой, перезагрузка) уже распознанные страницы не
# пересчитываются. Колонтитулы и абзацы собираются только при сборке, по всем страницам.
PAGE_CACHE_FORMAT = 1


def _block_to_dict(block: PageBlock) -> dict:
    if isinstance(block, TableBlock):
        return {
            "type": "table",
            "y": block.y,
            "bbox": list(block.bbox),
            "n_rows": block.n_rows,
            "n_cols": block.n_cols,
            "dropped_columns": block.dropped_columns,
            "cells": [{**asdict(c), "low_lines": list(c.low_lines)} for c in block.cells],
        }
    return {
        "type": "text",
        "y": block.y,
        "text": block.text,
        "x0": block.x0,
        "x1": block.x1,
        "height": block.height,
        "low_words": list(block.low_words),
    }


def _block_from_dict(data: dict) -> PageBlock:
    if data["type"] == "table":
        return TableBlock(
            y=data["y"],
            bbox=tuple(data["bbox"]),
            cells=[TableCell(**{**c, "low_lines": tuple(c["low_lines"])}) for c in data["cells"]],
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            dropped_columns=data["dropped_columns"],
        )
    return TextBlock(
        y=data["y"],
        text=data["text"],
        x0=data["x0"],
        x1=data["x1"],
        height=data["height"],
        low_words=tuple(data["low_words"]),
    )


def _save_page(path: Path, page: PageResult, meta: dict) -> None:
    data = {
        "format": PAGE_CACHE_FORMAT,
        "meta": meta,
        "page_number": page.page_number,
        "ocr_applied": page.ocr_applied,
        "plain_text": page.plain_text,
        "page_height": page.page_height,
        "blocks": [_block_to_dict(b) for b in page.blocks],
    }
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(part, path)


def _load_page(path: Path, meta: dict) -> PageResult | None:
    """Сохранённая страница, если получена с теми же параметрами распознавания; иначе None."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("format") != PAGE_CACHE_FORMAT or data.get("meta") != meta:
        return None
    return PageResult(
        page_number=data["page_number"],
        ocr_applied=data["ocr_applied"],
        blocks=[_block_from_dict(b) for b in data["blocks"]],
        plain_text=data["plain_text"],
        page_height=data["page_height"],
    )


def _stderr_progress(done: int, total: int, page_number: int, status: str) -> None:
    percent = 100 * done / total if total else 100.0
    print(
        f"[{percent:5.1f}%] страница {page_number} из {total}: {status}",
        file=sys.stderr,
        flush=True,
    )


# --- Тайминги шагов ---


def _fmt_duration(seconds: float) -> str:
    minutes, sec = divmod(round(seconds), 60)
    return f"{minutes} мин {sec:02d} с" if minutes else f"{sec} с"


class StepTimer:
    """Длительность каждого шага. page=None — шаг документа, иначе номер страницы."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.records: list[dict] = []

    @contextmanager
    def step(self, page: int | None, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(page, name, time.perf_counter() - t0)

    def add(self, page: int | None, name: str, seconds: float) -> None:
        self.records.append({"page": page, "step": name, "seconds": round(seconds, 2)})
        where = "документ" if page is None else f"стр. {page}"
        print(f"  [{where}] {name}: {_fmt_duration(seconds)}", file=sys.stderr, flush=True)

    def report(self) -> dict:
        page_totals: dict[int, float] = {}
        for r in self.records:
            if r["page"] is not None:
                page_totals[r["page"]] = page_totals.get(r["page"], 0.0) + r["seconds"]
        return {
            "total_seconds": round(time.perf_counter() - self.started, 2),
            "document_steps": [r for r in self.records if r["page"] is None],
            "pages": {
                str(n): {
                    "total_seconds": round(page_totals[n], 2),
                    "steps": [r for r in self.records if r["page"] == n],
                }
                for n in sorted(page_totals)
            },
        }


def _print_timing_report(report: dict) -> None:
    lines = [f"ИТОГО по документу: {_fmt_duration(report['total_seconds'])}"]
    for step in report["document_steps"]:
        lines.append(f"  {step['step']}: {_fmt_duration(step['seconds'])}")
    for number, page in report["pages"].items():
        lines.append(f"Страница {number} — {_fmt_duration(page['total_seconds'])}")
        for step in page["steps"]:
            lines.append(f"  {step['step']}: {_fmt_duration(step['seconds'])}")
    print("\n".join(lines), file=sys.stderr, flush=True)


def _timed_fn(timer: StepTimer, page: int, name: str, fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        with timer.step(page, name):
            return fn(*args, **kwargs)

    return wrapped


def ocr_document(
    pdf_path: Path | str,
    out_docx_path: Path | str,
    *,
    dpi: int = DEFAULT_DPI,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    min_page_text_chars: int = DEFAULT_MIN_PAGE_TEXT_CHARS,
    predict_tables_fn: Callable[[Path], dict] | None = None,
    image_data_fn: Callable[[Path], dict] | None = None,
    authority_name: str | None = None,
    min_free_memory_mb: int = DEFAULT_MIN_FREE_MEMORY_MB,
    table_engine: str = "tatr",
    pages_dir: Path | str | None = None,
    resume: bool = True,
    progress_fn: Callable[[int, int, int, str], None] | None = None,
    use_ai_escalation: bool = False,
) -> OcrDocumentResult:
    """Точка входа этапа 3. Постранично: если у страницы уже есть текстовый слой
    (>= min_page_text_chars символов) — используется он как есть, OCR не
    запускается (см. docs/plan.md, «Охват» — справочники ИТС почти всегда
    уже с текстом). Иначе страница рендерится в изображение и распознаётся
    гибридной схемой (см. docstring модуля).

    use_ai_escalation — по умолчанию выключено: требует доступа к Ollama по
    сети (см. ai_escalation.OLLAMA_URL) и ощутимо замедляет страницы со
    штампами/печатями (VLM-инференс на CPU без GPU — десятки секунд на
    область). Включает поиск печатей/штампов/э-подписей и восстановление
    текста под ними, см. run_tesseract_text, docs/plan.md «ИИ-эскалация».

    predict_tables_fn/image_data_fn — инъекция движков; по умолчанию настоящие
    тяжёлые модели (см. _default_predict_tables_fn/_default_image_data_fn), в
    тестах подменяются фейками на реальных фикстурах. Движки по умолчанию
    создаются лениво — только при первой реально встреченной странице-скане, не
    при вызове функции: PPStructureV3 грузит ~19 моделей (минуты, сотни МБ —
    см. docs/plan.md, «Реализация»), и документ без единой страницы-скана
    (например, уже весь с текстовым слоем) не должен платить эту цену впустую.
    authority_name — орган, известный из реквизитов документа (не из OCR); при
    повреждении печатью строки с формулой должности подписанта она
    восстанавливается, см. expected_signatory_title/_restore_signatory_title.

    Постранично: результат каждой страницы сохраняется в pages_dir (по умолчанию
    <out>.pages/page_NNNN.json). При resume=True страницы, уже сохранённые с теми же
    параметрами, не распознаются заново — прогон можно продолжить после остановки.
    Сборка DOCX делается по всем страницам из этих файлов. progress_fn(done, total,
    номер страницы, статус) вызывается перед распознаванием и после каждой страницы;
    по умолчанию печатает процент в stderr. Длительность каждого шага пишется в
    <out>.timings.json и печатается в stderr (см. StepTimer).
    """
    import fitz

    timer = StepTimer()
    report = progress_fn or _stderr_progress

    with timer.step(None, "Открытие и подготовка"):
        pdf_path = Path(pdf_path)
        out_docx_path = Path(out_docx_path)
        if pages_dir is None:
            pages_dir = out_docx_path.with_name(out_docx_path.stem + ".pages")
        pages_dir = Path(pages_dir)
        meta = {
            "dpi": dpi,
            "confidence_threshold": confidence_threshold,
            "min_page_text_chars": min_page_text_chars,
            "authority_name": authority_name,
            "table_engine": table_engine,
            "pdf_size": pdf_path.stat().st_size,
        }
        reader = pypdf.PdfReader(str(pdf_path))
        fitz_doc = fitz.open(str(pdf_path))
        total = len(reader.pages)
        pages_dir.mkdir(parents=True, exist_ok=True)
        pages: list[PageResult] = []

    text_predict_fn: Callable | None = None
    cell_text_fn: Callable | None = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (pdf_page, fitz_page) in enumerate(zip(reader.pages, fitz_doc)):
            page_number = i + 1
            page_file = pages_dir / f"page_{page_number:04d}.json"
            if resume:
                with timer.step(page_number, "проверка сохранённого результата"):
                    cached = _load_page(page_file, meta)
                if cached is not None:
                    pages.append(cached)
                    report(page_number, total, page_number, "взято из сохранённого прогона")
                    continue
            with timer.step(page_number, "проверка текстового слоя"):
                text = (pdf_page.extract_text() or "").strip()
            if len(text) >= min_page_text_chars:
                page = PageResult(page_number=page_number, ocr_applied=False, plain_text=text)
                with timer.step(page_number, "сохранение страницы"):
                    _save_page(page_file, page, meta)
                pages.append(page)
                report(page_number, total, page_number, "текстовый слой")
                continue
            report(i, total, page_number, "распознаётся")
            with timer.step(page_number, "проверка свободной памяти"):
                ensure_free_memory(min_free_memory_mb, f"перед страницей {page_number}")
            use_tatr = table_engine == "tatr" and predict_tables_fn is None
            with timer.step(page_number, "загрузка моделей"):
                if use_tatr and text_predict_fn is None:
                    # TATR грузим ДО PaddleOCR: torch должен импортироваться раньше paddle на Windows.
                    # PaddleOCR-движок (уточнение регионов таблиц) создаётся лениво (см. _LazyEngine).
                    # На CPU без AVX2 Paddle падает с SIGILL — тогда регионы таблиц берёт Tesseract;
                    # текст ячеек (RapidOCR/ONNX) от этого не зависит — работает в любом случае.
                    table_structure.load_tatr()
                    cell_text_fn = _LazyEngine(_default_cell_text_fn)
                    if _paddle_runtime_usable():
                        text_predict_fn = _LazyEngine(_default_predict_tables_fn)
                    else:
                        print(
                            "PaddleOCR недоступен на этом CPU (нет AVX2 / SIGILL) — "
                            "регионы таблиц через Tesseract",
                            file=sys.stderr,
                        )
                        text_predict_fn = _LazyEngine(_tesseract_text_predict_fn)
                if predict_tables_fn is None and not use_tatr:
                    if not _paddle_runtime_usable():
                        raise RuntimeError(
                            "PaddleOCR недоступен на этом CPU (нет AVX2 / SIGILL). "
                            "Используйте table_engine='tatr' (значение по умолчанию)."
                        )
                    predict_tables_fn = _default_predict_tables_fn()
                if image_data_fn is None:
                    image_data_fn = _default_image_data_fn
            image_path = Path(tmp_dir) / f"page_{page_number}.png"
            with timer.step(page_number, "рендер страницы"):
                pix = fitz_page.get_pixmap(dpi=dpi)
                pix.save(str(image_path))
            timed_image_data = _timed_fn(timer, page_number, "текст: Tesseract", image_data_fn)
            if use_tatr:

                def tables_fn(path: Path, _page: int = page_number) -> list[TableBlock]:
                    regions_seconds = [0.0]

                    def timed_regions(img: Image.Image) -> list[Box]:
                        t0 = time.perf_counter()
                        found = table_regions_for_page(img, text_predict_fn)
                        regions_seconds[0] = time.perf_counter() - t0
                        return found

                    t0 = time.perf_counter()
                    result = run_tatr_tables(
                        path,
                        text_predict_fn,
                        cell_text_fn=cell_text_fn,
                        region_fn=timed_regions,
                        release_structure=text_predict_fn.release,
                        release_cells=cell_text_fn.release,
                    )
                    elapsed = time.perf_counter() - t0
                    timer.add(_page, "таблицы: поиск областей", regions_seconds[0])
                    timer.add(_page, "таблицы: структура и текст ячеек", elapsed - regions_seconds[0])
                    return result

                blocks = ocr_page(
                    image_path,
                    table_blocks_fn=tables_fn,
                    image_data_fn=timed_image_data,
                    confidence_threshold=confidence_threshold,
                    authority_name=authority_name,
                    detect_marks_fn=_default_detect_marks_fn if use_ai_escalation else None,
                    recover_covered_text_fn=ai_escalation.recover_covered_text if use_ai_escalation else None,
                )
            else:
                blocks = ocr_page(
                    image_path,
                    predict_tables_fn=_timed_fn(timer, page_number, "таблицы: PaddleOCR", predict_tables_fn),
                    image_data_fn=timed_image_data,
                    confidence_threshold=confidence_threshold,
                    authority_name=authority_name,
                    detect_marks_fn=_default_detect_marks_fn if use_ai_escalation else None,
                    recover_covered_text_fn=ai_escalation.recover_covered_text if use_ai_escalation else None,
                )
            page = PageResult(
                page_number=page_number,
                ocr_applied=True,
                blocks=blocks,
                page_height=float(pix.height),
            )
            with timer.step(page_number, "сохранение страницы"):
                _save_page(page_file, page, meta)
            pages.append(page)
            report(page_number, total, page_number, "готово")

    with timer.step(None, "Завершение распознавания"):
        fitz_doc.close()
    with timer.step(None, "Колонтитулы"):
        pages = remove_running_headers_and_footers(pages)
    with timer.step(None, "Сборка DOCX"):
        build_docx(pages, out_docx_path, dpi=dpi)

    timings = timer.report()
    timings_path = out_docx_path.with_name(out_docx_path.stem + ".timings.json")
    timings_path.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_timing_report(timings)
    return OcrDocumentResult(pages=pages, docx_path=out_docx_path, timings=timings, timings_path=timings_path)
