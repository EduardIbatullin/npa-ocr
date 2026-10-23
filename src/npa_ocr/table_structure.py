"""Структура таблиц через Table Transformer (TATR, MIT, веса Hugging Face).

Детекция строк/столбцов/объединённых ячеек — модель; текст ячеек — из PaddleOCR
(см. ocr.py). Чистая логика (сборка ячеек, проверка столбцов) — без тяжёлых
импортов, тестируется на синтетических координатах.

На Windows torch нужно импортировать ДО paddle в одном процессе (см. docs/plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from npa_ocr.ocr_table import TableCell

Box = tuple[float, float, float, float]

TATR_MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"
# Отдельная модель именно для поиска таблиц на ЦЕЛОЙ странице (detect_table_regions) —
# не структуры уже вырезанной таблицы (TATR_MODEL_NAME, тот используется только после
# того, как область уже найдена, см. detect_structure). Раньше detect_table_regions
# использовал TATR_MODEL_NAME с тем же TATR_TABLE_LABEL — но эта модель обучена на
# кропах таблиц, а не целых страницах, и на плотном выровненном по ширине тексте
# (типичная вёрстка НПА) систематически давала ложное срабатывание: одна область на
# всю страницу целиком. table-transformer-detection обучена на целых страницах
# документов (PubTables-1M, detection split) — это её прямое назначение.
TATR_DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
TATR_TABLE_LABEL = 0
# Метка detect_table_regions (модель TATR_DETECTION_MODEL_NAME) — своё
# пространство меток, отдельное от TATR_COLUMN_LABEL ниже (та — для
# detect_structure, модель TATR_MODEL_NAME); то, что оба равны 1, совпадение,
# не связь. См. docs/plan.md «Приказ №191 — TATR путает непривычный layout
# с повёрнутой таблицей» (2026-10-23): TATR_DETECTION_MODEL_NAME обучена на
# PubTables-1M (таблицы из научных статей, где «table rotated» — реально
# развёрнутая на 90° широкая таблица). На многостраничных таблицах НПА без
# подписи, залитых на весь лист без полей (обычное дело для страниц-
# продолжений), детектор на реальном документе систематически путает этот
# layout с «table rotated» — с уверенностью 0.93-0.999, т.е. таблицу он
# видит прекрасно, ошибается только в выборе из двух классов. Простого
# геометрического признака, отличающего верные и ошибочные случаи,
# эмпирически не нашлось (проверено: отступ от края листа, соотношение
# сторон региона — не предсказывают). Наш пайплайн не делает коррекцию
# поворота ни для одного из классов, так что оба трактуются одинаково —
# детектор не умеет надёжно их различать на этом типе документов, а для
# НЕ повёрнутых (подавляющее большинство реальных случаев) разница не нужна.
# Если область и правда повёрнута — текст в ней не читается ни TATR
# (структура), ни Tesseract/RapidOCR (текст), сетка выйдет почти пустой и
# отсеется тем же is_plausible_table, что и раньше — не хуже, чем сейчас.
TATR_TABLE_ROTATED_LABEL = 1
TATR_COLUMN_LABEL = 1
TATR_ROW_LABEL = 2
TATR_HEADER_LABEL = 3
TATR_SPANNING_LABEL = 5
TATR_REGION_THRESHOLD = 0.7
# Резервный порог — см. docs/plan.md, тот же раздел, что у
# TATR_TABLE_ROTATED_LABEL: на части страниц того же документа (не только
# из-за выбора класса) уверенность детектора по ОБОИМ классам нестабильна
# (0.5-0.65) на этом нетипичном для PubTables-1M layout. Применяется только
# если на обычном пороге не нашлось ни одной области (см.
# detect_table_regions) — не на каждой странице, лишний проход
# постобработки (не повторный forward pass модели) дешёвый. Ложные
# срабатывания на этом пороге отсеивает is_plausible_table после сборки
# сетки — тот же защитный механизм, что уже проверен на Приказе №565.
TATR_REGION_FALLBACK_THRESHOLD = 0.3
TATR_STRUCTURE_THRESHOLD = 0.5


@dataclass
class TableStructure:
    rows: list[Box]
    cols: list[Box]
    spans: list[Box]


def _center(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _overlaps_vertically(a: Box, b: Box) -> bool:
    return a[1] < b[3] and b[1] < a[3]


def merge_header_bands(rows: list[Box], headers: list[Box]) -> list[Box]:
    """Заголовок, не перекрывающийся ни с одной строкой по вертикали, — это отдельная
    строка шапки; если перекрывается — строка уже есть, дубля не добавляем."""
    extra = [h for h in headers if not any(_overlaps_vertically(h, r) for r in rows)]
    return sorted(rows + extra, key=lambda b: b[1])


def _row_for_box(rows: list[Box], box: Box) -> int | None:
    """Строка с наибольшим вертикальным перекрытием с текстовым блоком; при равенстве
    и отсутствии перекрытия — ближайшая по центру. Перекрывающиеся полосы TATR не
    дают неоднозначности: текст уходит в ту строку, где он в основном лежит."""
    best, best_overlap = None, 0.0
    for i, r in enumerate(rows):
        overlap = min(r[3], box[3]) - max(r[1], box[1])
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    if best is not None:
        return best
    if not rows:
        return None
    cy = _center(box)[1]
    return min(range(len(rows)), key=lambda i: abs(_center(rows[i])[1] - cy))


def expand_rows_to_text(structure: TableStructure, texts: list[tuple[Box, str]]) -> TableStructure:
    """Полоса строки короче текста, который в ней центрирован, обрезает текст при
    вырезке ячейки. Растягиваем полосу по вертикали до границ такого текста."""
    rows = [list(r) for r in structure.rows]
    for box, _ in texts:
        i = _row_for_box(structure.rows, box)
        if i is None:
            continue
        cy = _center(box)[1]
        r = structure.rows[i]
        if r[1] <= cy <= r[3]:
            rows[i][1] = min(rows[i][1], box[1])
            rows[i][3] = max(rows[i][3], box[3])
    return TableStructure(rows=[tuple(r) for r in rows], cols=structure.cols, spans=structure.spans)


def _index_of(boxes: list[Box], value: float, axis: int) -> int | None:
    for i, b in enumerate(boxes):
        if b[axis] <= value <= b[axis + 2]:
            return i
    return None


def assign_text(
    structure: TableStructure, texts: list[tuple[Box, str]]
) -> dict[tuple[int, int], list[tuple[float, float, str]]]:
    """Назначает текстовые строки ячейкам сетки. Текст внутри объединённой ячейки
    попадает в её верхнюю-левую позицию."""
    grid: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    spans = [_span_range(structure, sp) for sp in structure.spans]
    for box, text in texts:
        cx, _ = _center(box)
        r = _row_for_box(structure.rows, box)
        c = _index_of(structure.cols, cx, 0)
        if r is None or c is None:
            continue
        for (r0, c0, r1, c1) in spans:
            if r0 <= r <= r1 and c0 <= c <= c1:
                r, c = r0, c0
                break
        grid.setdefault((r, c), []).append((box[1], box[0], text))
    return grid


_SPAN_COVER_RATIO = 0.5


def _covered_fraction(band: Box, span: Box, axis: int) -> float:
    lo, hi = (band[1], band[3]) if axis == 1 else (band[0], band[2])
    s_lo, s_hi = (span[1], span[3]) if axis == 1 else (span[0], span[2])
    length = hi - lo
    if length <= 0:
        return 0.0
    return max(0.0, min(hi, s_hi) - max(lo, s_lo)) / length


def merge_overlapping_row_bands(rows: list[Box], min_overlap: float = 0.5) -> list[Box]:
    """Полосы строк шапки TATR сильно перекрываются (одна лежит внутри другой): это
    одна строка. Сливаем полосы, перекрывающие друг друга на min_overlap и более
    доли меньшей из них."""
    merged = sorted(rows, key=lambda b: b[1])
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                smaller = min(a[3] - a[1], b[3] - b[1])
                if smaller > 0 and overlap / smaller >= min_overlap:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def _span_range(structure: TableStructure, span: Box) -> tuple[int, int, int, int]:
    rows_in = [i for i, r in enumerate(structure.rows) if _covered_fraction(r, span, 1) >= _SPAN_COVER_RATIO]
    cols_in = [i for i, c in enumerate(structure.cols) if _covered_fraction(c, span, 0) >= _SPAN_COVER_RATIO]
    if not rows_in or not cols_in:
        return (-1, -1, -2, -2)
    return (min(rows_in), min(cols_in), max(rows_in), max(cols_in))


def drop_empty_columns(structure: TableStructure, texts: list[tuple[Box, str]]) -> tuple[TableStructure, int]:
    """Столбец без единого текстового блока — артефакт детекции (на стр. 22 и 49
    Приказа №901 TATR выделяет лишний пустой четвёртый столбец). Удаляем его,
    число удалённых возвращаем, чтобы это было видно вызывающему коду."""
    used = set()
    for box, _ in texts:
        cx = _center(box)[0]
        c = _index_of(structure.cols, cx, 0)
        if c is not None:
            used.add(c)
    kept = [c for i, c in enumerate(structure.cols) if i in used]
    return TableStructure(rows=structure.rows, cols=kept, spans=structure.spans), len(structure.cols) - len(kept)


def drop_empty_rows(structure: TableStructure, texts: list[tuple[Box, str]]) -> TableStructure:
    used = set()
    for box, _ in texts:
        r = _row_for_box(structure.rows, box)
        if r is not None:
            used.add(r)
    kept = [r for i, r in enumerate(structure.rows) if i in used]
    return TableStructure(rows=kept, cols=structure.cols, spans=structure.spans)


FRAGMENT_GAP_RATIO = 0.12  # зазор между фрагментами одного слова — доля высоты строки
# Доля площади короткого бокса, перекрытая другим боксом той же строки, начиная с
# которой он считается дублирующей детекцией (не отдельным фрагментом текста).
_DUPLICATE_CONTAINMENT_RATIO = 0.5
_DUPLICATE_MAX_TEXT_LEN = 2  # дублирующая детекция — всегда короткий огрызок (цифра/символ)


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _is_duplicate_fragment(box: Box, others: list[Box]) -> bool:
    """True, если box почти целиком лежит внутри какого-то из others — см.
    docstring join_fragment_lines, «надстрочная цифра детектируется дважды»."""
    area = _box_area(box)
    if area <= 0:
        return False
    for other in others:
        ix = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
        iy = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        if (ix * iy) / area >= _DUPLICATE_CONTAINMENT_RATIO:
            return True
    return False


def join_fragment_lines(items: list[tuple[Box, str, float]]) -> list[tuple[str, float]]:
    """Склеивает фрагменты распознавания одной ячейки в строки текста. Блоки одной строки
    сортируются по x; если зазор между соседними блоками неотрицательный и меньше
    FRAGMENT_GAP_RATIO высоты — это части одного слова и склеиваются без пробела,
    иначе — через пробел. Возвращает (текст строки, минимальная уверенность её блоков).

    Отрицательный зазор (боксы соседних блоков слегка перекрываются по x) НЕ
    считается признаком одного слова — проверено на реальном документе (Приказ
    №565): RapidOCR (eslav_PP-OCRv5) систематически даёт перекрытие в 10-12px
    между боксами двух РАЗНЫХ соседних слов («Взвешенные»/«вещества»,
    «Серы»/«диоксид» — проверено впрямую на кропах ячеек, см. docs/plan.md),
    из-за чего раньше эти пары слипались в одно без пробела
    («Взвешенныевещества»). Условие `gap >= 0` защищает только от склейки
    случая, когда гарантированно НЕТ пробела (боксы касаются/чуть
    перекрываются в самом деле), не от обычного слегка неточного детектора.

    Отдельная, более редкая проблема — надстрочная цифра («мг/нм³») детектируется
    ДВАЖДЫ: один раз как часть основного бокса («MΓ/HM3», надстрочная «³» уже
    внутри, просто не надстрочная), и ещё раз отдельным мелким боксом вокруг того
    же самого символа (проверено впрямую на двух реальных кропах той же ячейки
    в разных местах документа — короткий бокс «3» на ~86% и ~100% перекрыт
    основным боксом). Такие короткие (<= _DUPLICATE_MAX_TEXT_LEN символов),
    почти целиком вложенные в другой бокс фрагменты отбрасываются целиком — не
    добавляются ни с пробелом, ни без (_is_duplicate_fragment)."""
    if not items:
        return []
    boxes_only = [box for box, _text, _score in items]
    items = [
        (box, text, score)
        for box, text, score in items
        if not (
            len(text) <= _DUPLICATE_MAX_TEXT_LEN
            and _is_duplicate_fragment(box, [b for b in boxes_only if b != box])
        )
    ]
    if not items:
        return []
    lines: list[list[tuple[Box, str, float]]] = []
    for box, text, score in sorted(items, key=lambda it: _center(it[0])[1]):
        placed = False
        for line in lines:
            ref = line[0][0]
            overlap = min(ref[3], box[3]) - max(ref[1], box[1])
            smaller = min(ref[3] - ref[1], box[3] - box[1])
            if smaller > 0 and overlap / smaller >= 0.5:
                line.append((box, text, score))
                placed = True
                break
        if not placed:
            lines.append([(box, text, score)])
    result: list[tuple[str, float]] = []
    for line in lines:
        line.sort(key=lambda it: it[0][0])
        out = ""
        prev_box = None
        for box, text, _score in line:
            if prev_box is None:
                out = text
            else:
                height = min(prev_box[3] - prev_box[1], box[3] - box[1])
                gap = box[0] - prev_box[2]
                sep = "" if 0 <= gap <= FRAGMENT_GAP_RATIO * height else " "
                out += sep + text
            prev_box = box
        result.append((out, min(sc for _, _, sc in line)))
    return result


def join_fragments(items: list[tuple[Box, str]]) -> str:
    """Строки текста ячейки через перенос строки (перенос в ячейке — w:br в DOCX),
    см. join_fragment_lines."""
    return "\n".join(text for text, _ in join_fragment_lines([(b, t, 1.0) for b, t in items]))


def cell_text_boxes(
    structure: TableStructure, texts: list[tuple[Box, str]]
) -> dict[tuple[int, int], list[tuple[Box, str]]]:
    """Текстовые блоки, попавшие в каждую ячейку сетки (та же логика назначения, что
    и в assign_text, но с сохранением геометрии блоков)."""
    out: dict[tuple[int, int], list[tuple[Box, str]]] = {}
    for box, text in texts:
        cx = _center(box)[0]
        r = _row_for_box(structure.rows, box)
        c = _index_of(structure.cols, cx, 0)
        if r is None or c is None:
            continue
        out.setdefault((r, c), []).append((box, text))
    return out


def text_crop_box(blocks: list[tuple[Box, str]], pad: int = 4) -> Box | None:
    """Кроп ячейки по реальным границам её текста. Границы столбцов TATR не используются:
    граница столбца может проходить через слово шапки, и тогда слово обрезалось бы
    («выполнени», «цаемый»). Текст ячейки задаёт и её кроп."""
    if not blocks:
        return None
    x1 = min(b[0] for b, _ in blocks) - pad
    y1 = min(b[1] for b, _ in blocks) - pad
    x2 = max(b[2] for b, _ in blocks) + pad
    y2 = max(b[3] for b, _ in blocks) + pad
    return (max(0.0, x1), max(0.0, y1), max(x1 + 1, x2), max(y1 + 1, y2))


def build_cells(structure: TableStructure, texts: list[tuple[Box, str]]) -> tuple[list[TableCell], int, int]:
    """Строит TableCell (с row_span/col_span по объединённым ячейкам) из структуры и
    текстов. Возвращает (ячейки, число строк, число столбцов)."""
    grid_text = assign_text(structure, texts)
    n_rows, n_cols = len(structure.rows), len(structure.cols)
    spans = [_span_range(structure, sp) for sp in structure.spans]
    spans = [s for s in spans if s[0] >= 0]
    covered: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    for (r0, c0, r1, c1) in spans:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) != (r0, c0):
                    covered.add((r, c))
        words = sorted(grid_text.get((r0, c0), []))
        cells.append(TableCell(row=r0, col=c0, row_span=r1 - r0 + 1, col_span=c1 - c0 + 1,
                               text=" ".join(w[2] for w in words)))
    span_origins = {(s[0], s[1]) for s in spans}
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in covered or (r, c) in span_origins:
                continue
            words = sorted(grid_text.get((r, c), []))
            cells.append(TableCell(row=r, col=c, row_span=1, col_span=1,
                                   text=" ".join(w[2] for w in words)))
    cells.sort(key=lambda t: (t.row, t.col))
    return cells, n_rows, n_cols


# Приказ №565 (2026-10-22), реальный документ: TATR (сам детектор регионов,
# microsoft/table-transformer-detection) с уверенностью 0.862 — выше
# TATR_REGION_THRESHOLD=0.7 — принял нижнюю часть страницы (конец пункта 3 +
# строку подписи министра) за таблицу 5x12. Поднять порог детекции не
# получится надёжно: на другом реальном документе (Постановление №1430)
# настоящие таблицы встречались со скорами 0.679-0.883 — диапазон
# пересекается с этим ложным срабатыванием, чистого порога по самой
# уверенности детектора нет (проверено эмпирически, см. docs/plan.md).
# Отличие — в структуре итоговой сетки: у настоящей таблицы большинство
# столбцов заполнено в большинстве строк (данные повторяются по вертикали,
# измерено на 4 реальных таблицах — 0.812-0.955), а здесь каждый «столбец» —
# случайный межсловный пробел одной строки текста, заполнен лишь в 1-2
# строках из 5 (измерено на реальной сетке — 0.417). Порог 0.6 — с запасом
# между этими двумя группами реальных измерений.
_TABLE_MIN_AVG_COL_FILL_RATIO = 0.6


def is_plausible_table(cells: list[TableCell], n_rows: int, n_cols: int) -> bool:
    """После сборки сетки (build_cells) решает, похоже ли это на настоящую
    таблицу, или TATR разбил на псевдо-столбцы обычный абзац текста (см.
    комментарий у _TABLE_MIN_AVG_COL_FILL_RATIO). Сетки меньше 2x2 слишком
    малы для этой проверки — не отбраковываются ею (такие случаи — вне
    текущей проблемы, см. docs/plan.md)."""
    if n_rows < 2 or n_cols < 2:
        return True
    filled_rows_per_col = [0] * n_cols
    for cell in cells:
        if not cell.text.strip():
            continue
        for c in range(cell.col, min(n_cols, cell.col + cell.col_span)):
            filled_rows_per_col[c] += 1
    ratios = [min(1.0, f / n_rows) for f in filled_rows_per_col]
    return (sum(ratios) / len(ratios)) >= _TABLE_MIN_AVG_COL_FILL_RATIO


def _disable_mkldnn_workaround() -> None:
    """MKL-DNN (oneDNN) бэкенд torch на некоторых старых CPU без AVX2 (например,
    Intel Xeon E5-2696 v2, Ivy Bridge, 2013 — подтверждено на реальном сервере)
    генерирует код, падающий с SIGILL на первом же реальном forward pass модели
    (после успешной загрузки весов). На CPU с AVX2/AVX512 эффект на точность
    результата нулевой — отключение mkldnn не меняет, что возвращает модель,
    только какой бэкенд считает свёртки; падение скорости не измерялось точно,
    но лучше медленнее работающий движок, чем падающий. Тот же мотив, что уже
    применён к PaddleOCR в ocr.py (enable_mkldnn=False)."""
    import torch

    torch.backends.mkldnn.enabled = False


@lru_cache(maxsize=1)
def load_tatr():
    _disable_mkldnn_workaround()
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    model = TableTransformerForObjectDetection.from_pretrained(TATR_MODEL_NAME).eval()
    proc = AutoImageProcessor.from_pretrained(TATR_MODEL_NAME, size={"shortest_edge": 800, "longest_edge": 1000})
    return model, proc


@lru_cache(maxsize=1)
def load_tatr_detector():
    """Модель для detect_table_regions — см. комментарий у TATR_DETECTION_MODEL_NAME.
    Отдельный кэш от load_tatr(): структура и детекция на целой странице — разные
    модели, обе могут понадобиться на одном документе (детекция находит область,
    detect_structure затем разбирает её на строки/столбцы)."""
    _disable_mkldnn_workaround()
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    model = TableTransformerForObjectDetection.from_pretrained(TATR_DETECTION_MODEL_NAME).eval()
    proc = AutoImageProcessor.from_pretrained(TATR_DETECTION_MODEL_NAME)
    return model, proc


def _run_detector(loader, img):
    """Один forward pass модели (дорогая часть) — отдельно от постобработки
    (дешёвая, можно повторить с другим порогом без повторного инференса, см.
    detect_table_regions)."""
    import torch

    model, proc = loader()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return proc, out


def _postprocess(proc, out, img, labels: set[int], threshold: float) -> list[Box]:
    res = proc.post_process_object_detection(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
    return [tuple(b.tolist()) for b, lab in zip(res["boxes"], res["labels"]) if int(lab) in labels]


def _detect_with(loader, img, labels: set[int], threshold: float) -> list[Box]:
    proc, out = _run_detector(loader, img)
    return _postprocess(proc, out, img, labels, threshold)


def _detect(img, labels: set[int], threshold: float) -> list[Box]:
    return _detect_with(load_tatr, img, labels, threshold)


def detect_table_regions(img) -> list[Box]:
    """Области таблиц на целой странице. Обе метки TATR_DETECTION_MODEL_NAME
    («table» и «table rotated») принимаются как равнозначные — см. комментарий
    у TATR_TABLE_ROTATED_LABEL. Если на обычном пороге не нашлось ни одной
    области, один и тот же forward pass переиспользуется для повторной
    постобработки на TATR_REGION_FALLBACK_THRESHOLD (см. комментарий там же) —
    без повторного инференса модели."""
    proc, out = _run_detector(load_tatr_detector, img)
    labels = {TATR_TABLE_LABEL, TATR_TABLE_ROTATED_LABEL}
    found = _postprocess(proc, out, img, labels, TATR_REGION_THRESHOLD)
    if found:
        return found
    return _postprocess(proc, out, img, labels, TATR_REGION_FALLBACK_THRESHOLD)


def detect_structure(img) -> TableStructure:
    rows = _detect(img, {TATR_ROW_LABEL}, TATR_STRUCTURE_THRESHOLD)
    headers = _detect(img, {TATR_HEADER_LABEL}, TATR_STRUCTURE_THRESHOLD)
    cols = _detect(img, {TATR_COLUMN_LABEL}, TATR_STRUCTURE_THRESHOLD)
    spans = _detect(img, {TATR_SPANNING_LABEL}, TATR_STRUCTURE_THRESHOLD)
    return TableStructure(
        rows=merge_overlapping_row_bands(merge_header_bands(rows, headers)),
        cols=sorted(cols, key=lambda b: b[0]),
        spans=spans,
    )
