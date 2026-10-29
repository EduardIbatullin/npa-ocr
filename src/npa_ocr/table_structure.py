"""Структура таблиц через Table Transformer (TATR, лицензия MIT, веса Hugging Face).

Строки, столбцы и объединённые ячейки находит модель, текст ячеек — OCR (см.
:mod:`npa_ocr.ocr`). Логика сборки ячеек и проверок не требует тяжёлых
импортов и тестируется на синтетических координатах.

На Windows torch нужно импортировать раньше paddle в одном процессе.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from npa_ocr.ocr_table import TableCell

Box = tuple[float, float, float, float]

TATR_MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"
# Поиск таблиц на целой странице (detect_table_regions). Модель структуры
# (TATR_MODEL_NAME) обучена на фрагментах таблиц и на плотном тексте, выровненном
# по ширине, находит «таблицу» во всю страницу; модель детекции обучена на целых
# страницах (PubTables-1M, detection split).
TATR_DETECTION_MODEL_NAME = "microsoft/table-transformer-detection"
TATR_TABLE_LABEL = 0
# Метка «table rotated» модели детекции (своё пространство меток, совпадение с
# TATR_COLUMN_LABEL случайно). Таблицы-продолжения во весь лист без подписи
# детектор уверенно относит к «table rotated», хотя они не повёрнуты, и
# геометрического признака, отличающего такие случаи, нет. Поворот здесь не
# исправляется, поэтому обе метки считаются таблицей; действительно повёрнутая
# область даст почти пустую сетку и будет отсеяна is_plausible_table.
TATR_TABLE_ROTATED_LABEL = 1
TATR_COLUMN_LABEL = 1
TATR_ROW_LABEL = 2
TATR_HEADER_LABEL = 3
TATR_SPANNING_LABEL = 5
TATR_REGION_THRESHOLD = 0.7
# Запасной порог: на таблицах во весь лист уверенность детектора бывает 0.5–0.65.
# Применяется, только если на обычном пороге не нашлось ни одной области; это
# повторная постобработка без повторного прогона модели. Ложные области отсеивает
# is_plausible_table.
TATR_REGION_FALLBACK_THRESHOLD = 0.3
TATR_STRUCTURE_THRESHOLD = 0.5


@dataclass
class TableStructure:
    """Рамки строк, столбцов и объединённых ячеек таблицы."""

    rows: list[Box]
    cols: list[Box]
    spans: list[Box]


def _center(b: Box) -> tuple[float, float]:
    """Центр рамки."""
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _overlaps_vertically(a: Box, b: Box) -> bool:
    """Проверяет, пересекаются ли рамки по вертикали."""
    return a[1] < b[3] and b[1] < a[3]


def merge_header_bands(rows: list[Box], headers: list[Box]) -> list[Box]:
    """Добавляет к строкам полосы шапки, не пересекающиеся ни с одной строкой.

    Полоса шапки, пересекающая строку, дублирует её и не добавляется.

    Returns:
        Строки сверху вниз.
    """
    extra = [h for h in headers if not any(_overlaps_vertically(h, r) for r in rows)]
    return sorted(rows + extra, key=lambda b: b[1])


def _row_for_box(rows: list[Box], box: Box) -> int | None:
    """Номер строки, в которой в основном лежит текстовый блок.

    Выбирается строка с наибольшим перекрытием по вертикали, без перекрытия —
    ближайшая по центру. None, если строк нет.
    """
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
    """Растягивает полосы строк по вертикали до границ их текста.

    Полоса короче текста, центрированного в ней, обрезала бы его при вырезании
    ячейки.
    """
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
    """Номер рамки, содержащей координату по оси ``axis`` (0 — x, 1 — y), или None."""
    for i, b in enumerate(boxes):
        if b[axis] <= value <= b[axis + 2]:
            return i
    return None


def assign_text(
    structure: TableStructure, texts: list[tuple[Box, str]]
) -> dict[tuple[int, int], list[tuple[float, float, str]]]:
    """Раскладывает текстовые блоки по ячейкам сетки.

    Текст внутри объединённой ячейки относится к её левой верхней позиции.

    Returns:
        Для позиции ``(строка, столбец)`` — тройки ``(y, x, текст)``.
    """
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
    """Доля длины полосы по оси ``axis``, покрытая объединённой ячейкой."""
    lo, hi = (band[1], band[3]) if axis == 1 else (band[0], band[2])
    s_lo, s_hi = (span[1], span[3]) if axis == 1 else (span[0], span[2])
    length = hi - lo
    if length <= 0:
        return 0.0
    return max(0.0, min(hi, s_hi) - max(lo, s_lo)) / length


def merge_overlapping_row_bands(rows: list[Box], min_overlap: float = 0.5) -> list[Box]:
    """Сливает полосы строк, сильно перекрывающие друг друга.

    В шапке TATR часто даёт полосу внутри другой полосы — это одна строка.

    Args:
        rows: Полосы строк.
        min_overlap: Доля высоты меньшей полосы, начиная с которой полосы
            сливаются.

    Returns:
        Полосы сверху вниз.
    """
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
    """Диапазон ``(r0, c0, r1, c1)`` позиций объединённой ячейки; ``(-1, -1, -2, -2)``, если пуст."""
    rows_in = [i for i, r in enumerate(structure.rows) if _covered_fraction(r, span, 1) >= _SPAN_COVER_RATIO]
    cols_in = [i for i, c in enumerate(structure.cols) if _covered_fraction(c, span, 0) >= _SPAN_COVER_RATIO]
    if not rows_in or not cols_in:
        return (-1, -1, -2, -2)
    return (min(rows_in), min(cols_in), max(rows_in), max(cols_in))


# TATR иногда объединяет ячейки разных строк с разными значениями. Настоящая
# объединённая ячейка несёт одно значение, и её текст попадает в одну из
# охватываемых позиций. Если текст (без учёта объединения, см. cell_text_boxes)
# есть больше чем в одной позиции, объединение ложное.
_SPAN_MAX_INDEPENDENT_CELLS = 1


def filter_false_positive_spans(
    structure: TableStructure, texts: list[tuple[Box, str]]
) -> TableStructure:
    """Отбрасывает ложные объединения ячеек (см. ``_SPAN_MAX_INDEPENDENT_CELLS``).

    Args:
        structure: Структура таблицы.
        texts: Текстовые блоки ``(рамка, текст)``.

    Returns:
        Структура без ложных объединений.
    """
    grouped = cell_text_boxes(structure, texts)
    kept: list[Box] = []
    for span in structure.spans:
        r0, c0, r1, c1 = _span_range(structure, span)
        if r0 < 0:
            kept.append(span)
            continue
        independent = sum(
            1
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if grouped.get((r, c))
        )
        if independent <= _SPAN_MAX_INDEPENDENT_CELLS:
            kept.append(span)
    return TableStructure(rows=structure.rows, cols=structure.cols, spans=kept)


def drop_empty_columns(structure: TableStructure, texts: list[tuple[Box, str]]) -> tuple[TableStructure, int]:
    """Удаляет столбцы без текста — артефакты детекции.

    Returns:
        Структура без пустых столбцов и число удалённых столбцов.
    """
    used = set()
    for box, _ in texts:
        cx = _center(box)[0]
        c = _index_of(structure.cols, cx, 0)
        if c is not None:
            used.add(c)
    kept = [c for i, c in enumerate(structure.cols) if i in used]
    return TableStructure(rows=structure.rows, cols=kept, spans=structure.spans), len(structure.cols) - len(kept)


def drop_empty_rows(structure: TableStructure, texts: list[tuple[Box, str]]) -> TableStructure:
    """Удаляет строки без текста."""
    used = set()
    for box, _ in texts:
        r = _row_for_box(structure.rows, box)
        if r is not None:
            used.add(r)
    kept = [r for i, r in enumerate(structure.rows) if i in used]
    return TableStructure(rows=kept, cols=structure.cols, spans=structure.spans)


# TATR иногда даёт слишком высокую полосу строки, вмещающую текст двух строк
# таблицы. Признак — разрыв между строками текста внутри ячейки не меньше высоты
# строки текста (у переноса одного значения разрыв в 2–7 раз меньше высоты).
# Случайно широкий разрыв при переносе бывает в одном столбце, настоящая граница
# строк видна в нескольких столбцах на одной высоте, поэтому разрез принимается,
# только если его подтверждают минимум два столбца.
_ROW_SPLIT_GAP_RATIO = 1.0
_ROW_SPLIT_MIN_COLUMNS = 2
# Разрывы в разных столбцах на одной границе не совпадают по y в точности:
# кандидаты на разрез ближе этого расстояния (px) считаются одной границей.
_ROW_SPLIT_MERGE_TOLERANCE = 20.0


def _cluster_line_ranges(boxes: list[Box]) -> list[tuple[float, float]]:
    """Группирует рамки в строки текста по перекрытию по вертикали.

    Returns:
        Диапазоны ``(y1, y2)`` строк сверху вниз.
    """
    lines: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: _center(b)[1]):
        placed = False
        for line in lines:
            ref = line[0]
            overlap = min(ref[3], box[3]) - max(ref[1], box[1])
            smaller = min(ref[3] - ref[1], box[3] - box[1])
            if smaller > 0 and overlap / smaller >= 0.5:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    ranges = [(min(b[1] for b in line), max(b[3] for b in line)) for line in lines]
    return sorted(ranges, key=lambda r: r[0])


def split_oversized_rows(structure: TableStructure, texts: list[tuple[Box, str]]) -> TableStructure:
    """Разбивает полосы строк, вмещающие текст нескольких строк таблицы.

    Условия разреза описаны у ``_ROW_SPLIT_GAP_RATIO``. Вызывается до
    распознавания текста по ячейкам, чтобы дальнейшая сборка работала с
    исправленными границами строк.

    Args:
        structure: Структура таблицы.
        texts: Текстовые блоки ``(рамка, текст)``.

    Returns:
        Структура с разбитыми строками.
    """
    grouped = cell_text_boxes(structure, texts)
    new_rows: list[Box] = []
    for i, row in enumerate(structure.rows):
        candidates: list[tuple[int, float]] = []  # (столбец, середина разрыва)
        for c in range(len(structure.cols)):
            blocks = grouped.get((i, c))
            if not blocks:
                continue
            lines = _cluster_line_ranges([box for box, _text in blocks])
            for (y1a, y2a), (y1b, y2b) in zip(lines, lines[1:]):
                gap = y1b - y2a
                height = min(y2a - y1a, y2b - y1b)
                if height > 0 and gap >= _ROW_SPLIT_GAP_RATIO * height:
                    candidates.append((c, (y2a + y1b) / 2))
        groups: list[list[tuple[int, float]]] = []
        for cand in sorted(candidates, key=lambda cm: cm[1]):
            if groups and cand[1] - groups[-1][-1][1] <= _ROW_SPLIT_MERGE_TOLERANCE:
                groups[-1].append(cand)
            else:
                groups.append([cand])
        splits = [
            sum(m for _c, m in g) / len(g)
            for g in groups
            if len({c for c, _m in g}) >= _ROW_SPLIT_MIN_COLUMNS
        ]
        if not splits:
            new_rows.append(row)
            continue
        bounds = [row[1], *splits, row[3]]
        new_rows.extend((row[0], lo, row[2], hi) for lo, hi in zip(bounds, bounds[1:]))
    return TableStructure(rows=new_rows, cols=structure.cols, spans=structure.spans)


FRAGMENT_GAP_RATIO = 0.12  # зазор между фрагментами одного слова — доля высоты строки
# доля площади короткой рамки внутри другой рамки, начиная с которой она дубль
_DUPLICATE_CONTAINMENT_RATIO = 0.5
_DUPLICATE_MAX_TEXT_LEN = 2  # дублирующая детекция — всегда короткий огрызок (цифра/символ)


def _box_area(box: Box) -> float:
    """Площадь рамки."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _is_duplicate_fragment(box: Box, others: list[Box]) -> bool:
    """Проверяет, что рамка почти целиком лежит внутри одной из ``others``."""
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
    """Склеивает распознанные фрагменты ячейки в строки текста.

    Фрагменты одной строки идут слева направо. Если зазор между соседями
    неотрицательный и меньше ``FRAGMENT_GAP_RATIO`` высоты, это части одного
    слова и они склеиваются без пробела, иначе — через пробел. Перекрытие рамок
    (отрицательный зазор) признаком одного слова не считается: RapidOCR часто
    даёт перекрытие 10–12 px между разными словами.

    Короткие фрагменты (до ``_DUPLICATE_MAX_TEXT_LEN`` знаков), почти целиком
    лежащие внутри другой рамки, отбрасываются: так детектор дублирует
    надстрочную цифру («³» в «мг/нм³»), уже прочитанную в основном фрагменте.

    Args:
        items: Фрагменты ``(рамка, текст, уверенность)``.

    Returns:
        Строки ``(текст, наименьшая уверенность её фрагментов)`` сверху вниз.
    """
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
    """Склеивает фрагменты ячейки в текст; строки разделены ``\\n``."""
    return "\n".join(text for text, _ in join_fragment_lines([(b, t, 1.0) for b, t in items]))


def cell_text_boxes(
    structure: TableStructure, texts: list[tuple[Box, str]]
) -> dict[tuple[int, int], list[tuple[Box, str]]]:
    """Раскладывает текстовые блоки по ячейкам сетки, сохраняя их рамки.

    В отличие от :func:`assign_text` объединения ячеек не учитываются.
    """
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
    """Рамка для вырезания ячейки по границам её текста с запасом ``pad``.

    Границы столбцов TATR не используются: они могут проходить через слово
    шапки и обрезать его.

    Returns:
        Рамка или None, если блоков нет.
    """
    if not blocks:
        return None
    x1 = min(b[0] for b, _ in blocks) - pad
    y1 = min(b[1] for b, _ in blocks) - pad
    x2 = max(b[2] for b, _ in blocks) + pad
    y2 = max(b[3] for b, _ in blocks) + pad
    return (max(0.0, x1), max(0.0, y1), max(x1 + 1, x2), max(y1 + 1, y2))


def build_cells(structure: TableStructure, texts: list[tuple[Box, str]]) -> tuple[list[TableCell], int, int]:
    """Собирает ячейки таблицы из структуры и текста.

    Args:
        structure: Структура таблицы.
        texts: Текстовые блоки ``(рамка, текст)``.

    Returns:
        Ячейки с объединениями, число строк и число столбцов.
    """
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


# Детектор принимает за таблицу и обычный текст (например, строку подписи) с той же
# уверенностью, что и настоящие таблицы, поэтому порогом уверенности их не
# различить. Различает заполненность: у настоящей таблицы столбцы заполнены в
# большинстве строк (0.8–0.95), а «столбцы» абзаца — случайные пробелы между
# словами, заполненные в одной-двух строках (около 0.4).
_TABLE_MIN_AVG_COL_FILL_RATIO = 0.6


def is_plausible_table(cells: list[TableCell], n_rows: int, n_cols: int) -> bool:
    """Проверяет, что собранная сетка — таблица, а не абзац, разбитый на столбцы.

    Сетка 1×1 таблицей не считается: такие области детектор находит на
    обычном тексте. Сетки в одну строку или один столбец слишком малы для
    проверки заполненности и принимаются. Остальные проверяются по средней
    заполненности столбцов (см. ``_TABLE_MIN_AVG_COL_FILL_RATIO``).

    Args:
        cells: Ячейки сетки.
        n_rows: Число строк.
        n_cols: Число столбцов.
    """
    if n_rows <= 1 and n_cols <= 1:
        return False
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
    """Отключает бэкенд oneDNN (MKL-DNN) в torch.

    На CPU без AVX2 oneDNN генерирует код, падающий с SIGILL при первом выводе
    модели. На результат отключение не влияет, только на скорость.
    """
    import torch

    torch.backends.mkldnn.enabled = False


@lru_cache(maxsize=1)
def load_tatr():
    """Загружает модель структуры таблиц TATR и её обработчик (один раз за процесс)."""
    _disable_mkldnn_workaround()
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    model = TableTransformerForObjectDetection.from_pretrained(TATR_MODEL_NAME).eval()
    proc = AutoImageProcessor.from_pretrained(TATR_MODEL_NAME, size={"shortest_edge": 800, "longest_edge": 1000})
    return model, proc


@lru_cache(maxsize=1)
def load_tatr_detector():
    """Загружает модель поиска таблиц на странице и её обработчик (один раз за процесс)."""
    _disable_mkldnn_workaround()
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    model = TableTransformerForObjectDetection.from_pretrained(TATR_DETECTION_MODEL_NAME).eval()
    proc = AutoImageProcessor.from_pretrained(TATR_DETECTION_MODEL_NAME)
    return model, proc


def _run_detector(loader, img):
    """Прогоняет модель на изображении.

    Отделено от постобработки, чтобы повторять её с другим порогом без
    повторного прогона модели.

    Returns:
        Обработчик и выход модели.
    """
    import torch

    model, proc = loader()
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return proc, out


def _postprocess(proc, out, img, labels: set[int], threshold: float) -> list[Box]:
    """Рамки объектов с метками ``labels`` и уверенностью не ниже ``threshold``."""
    res = proc.post_process_object_detection(out, threshold=threshold, target_sizes=[img.size[::-1]])[0]
    return [tuple(b.tolist()) for b, lab in zip(res["boxes"], res["labels"]) if int(lab) in labels]


def _detect_with(loader, img, labels: set[int], threshold: float) -> list[Box]:
    """Находит объекты моделью из ``loader``."""
    proc, out = _run_detector(loader, img)
    return _postprocess(proc, out, img, labels, threshold)


def _detect(img, labels: set[int], threshold: float) -> list[Box]:
    """Находит объекты моделью структуры таблиц."""
    return _detect_with(load_tatr, img, labels, threshold)


def _box_iou(a: Box, b: Box) -> float:
    """Отношение площади пересечения рамок к площади объединения."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping_table_regions(boxes: list[Box]) -> list[Box]:
    """Сливает пересекающиеся области таблиц в общую охватывающую рамку.

    DETR-детектор TATR не подавляет дубли (NMS) и может выдать одну таблицу
    несколькими почти одинаковыми областями; без слияния таблица попала бы в
    документ несколько раз.
    """
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if _box_iou(a, b) > 0:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def detect_table_regions(img) -> list[Box]:
    """Находит области таблиц на странице.

    Метки «table» и «table rotated» равнозначны (см.
    ``TATR_TABLE_ROTATED_LABEL``). Если на обычном пороге областей нет, выход
    модели обрабатывается повторно с ``TATR_REGION_FALLBACK_THRESHOLD``.
    Пересекающиеся области сливаются.

    Args:
        img: Изображение страницы (PIL).

    Returns:
        Рамки областей таблиц.
    """
    proc, out = _run_detector(load_tatr_detector, img)
    labels = {TATR_TABLE_LABEL, TATR_TABLE_ROTATED_LABEL}
    found = _postprocess(proc, out, img, labels, TATR_REGION_THRESHOLD)
    if not found:
        found = _postprocess(proc, out, img, labels, TATR_REGION_FALLBACK_THRESHOLD)
    return _merge_overlapping_table_regions(found)


def detect_structure(img) -> TableStructure:
    """Находит строки, столбцы и объединённые ячейки на изображении таблицы.

    Args:
        img: Вырезанная таблица (PIL).

    Returns:
        Структура: строки сверху вниз (с полосами шапки, слитыми
        перекрывающимися), столбцы слева направо, объединённые ячейки.
    """
    rows = _detect(img, {TATR_ROW_LABEL}, TATR_STRUCTURE_THRESHOLD)
    headers = _detect(img, {TATR_HEADER_LABEL}, TATR_STRUCTURE_THRESHOLD)
    cols = _detect(img, {TATR_COLUMN_LABEL}, TATR_STRUCTURE_THRESHOLD)
    spans = _detect(img, {TATR_SPANNING_LABEL}, TATR_STRUCTURE_THRESHOLD)
    return TableStructure(
        rows=merge_overlapping_row_bands(merge_header_bands(rows, headers)),
        cols=sorted(cols, key=lambda b: b[0]),
        spans=spans,
    )
