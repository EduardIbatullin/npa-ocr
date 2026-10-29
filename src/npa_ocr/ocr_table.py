"""Сборка таблицы из геометрии ячеек PaddleOCR (PPStructureV3).

Таблица строится по рамкам ячеек (``cell_box_list``), а не по готовому
``pred_html``: у ``pred_html`` бывает неровная структура (разное число ``<td>``
в строках) даже при верной геометрии ячеек.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

_LINE_TOLERANCE = 5.0  # px — соседние ячейки редко дают идеально совпадающие границы


@dataclass
class TableCell:
    """Одна ячейка результата (уже с учётом объединений — см. row_span/col_span)."""

    row: int
    col: int
    row_span: int
    col_span: int
    text: str
    low_lines: tuple[bool, ...] = ()  # по строкам text: распознано с низкой уверенностью


def _cluster_lines(values: list[float], tolerance: float = _LINE_TOLERANCE) -> list[float]:
    """Группирует близкие координаты границ ячеек в общие линии сетки строк/столбцов."""
    if not values:
        return []
    ordered = sorted(values)
    lines = [ordered[0]]
    counts = [1]
    for v in ordered[1:]:
        if v - lines[-1] > tolerance:
            lines.append(v)
            counts.append(1)
        else:
            n = counts[-1]
            lines[-1] = (lines[-1] * n + v) / (n + 1)
            counts[-1] = n + 1
    return lines


def _line_index(lines: list[float], value: float, tolerance: float = _LINE_TOLERANCE) -> int:
    """Номер линии сетки, к которой относится координата; при промахе — ближайшей."""
    for i, line in enumerate(lines):
        if abs(line - value) <= tolerance:
            return i
    # Геометрия детектора не идеальна — берём ближайшую линию, а не падаем.
    return min(range(len(lines)), key=lambda i: abs(lines[i] - value))


def _assign_text_to_cells(
    cell_boxes: list[Box], rec_boxes: list[Box], rec_texts: list[str]
) -> list[list[str]]:
    """Раскладывает распознанные фрагменты текста по ячейкам.

    Фрагмент относится к ячейке, в которую попал его центр; при нескольких
    подходящих — к наименьшей.

    Returns:
        Для каждой ячейки — её фрагменты в порядке чтения.
    """
    assigned: list[list[tuple[float, float, str]]] = [[] for _ in cell_boxes]
    for box, text in zip(rec_boxes, rec_texts):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        best_idx: int | None = None
        best_area: float | None = None
        for i, (x1, y1, x2, y2) in enumerate(cell_boxes):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                # Детектор иногда даёт перекрывающиеся рамки — наименьшая
                # точнее соответствует настоящей ячейке.
                if best_area is None or area < best_area:
                    best_idx = i
                    best_area = area
        if best_idx is not None:
            assigned[best_idx].append((cy, cx, text))
    result = []
    for items in assigned:
        items.sort(key=lambda t: (round(t[0] / 10), t[1]))
        result.append([t[2] for t in items])
    return result


def reconstruct_table(
    cell_box_list: list[Box], rec_boxes: list[Box], rec_texts: list[str]
) -> list[TableCell]:
    """Строит сетку строк и столбцов по рамкам ячеек и раскладывает по ней текст.

    Ошибки самой детекции ячеек (лишние или неверные рамки) функция не
    исправляет — они переходят в результат.

    Args:
        cell_box_list: Рамки ячеек.
        rec_boxes: Рамки распознанных фрагментов текста.
        rec_texts: Тексты фрагментов.

    Returns:
        Ячейки с позициями и объединениями, по строкам и столбцам.
    """
    row_lines = _cluster_lines(
        [b[1] for b in cell_box_list] + [b[3] for b in cell_box_list]
    )
    col_lines = _cluster_lines(
        [b[0] for b in cell_box_list] + [b[2] for b in cell_box_list]
    )

    texts_per_cell = _assign_text_to_cells(cell_box_list, rec_boxes, rec_texts)

    cells: list[TableCell] = []
    for box, texts in zip(cell_box_list, texts_per_cell):
        x1, y1, x2, y2 = box
        row = _line_index(row_lines, y1)
        row_end = _line_index(row_lines, y2)
        col = _line_index(col_lines, x1)
        col_end = _line_index(col_lines, x2)
        cells.append(
            TableCell(
                row=row,
                col=col,
                row_span=max(1, row_end - row),
                col_span=max(1, col_end - col),
                text=" ".join(texts),
            )
        )
    cells.sort(key=lambda c: (c.row, c.col))
    return cells


# Наблюдаемые искажения OCR в коротких сокращениях шапок таблиц. Каждый образец —
# смесь латиницы и греческих букв, не встречающаяся в русских словах, поэтому
# замена подстрокой безопасна и внутри длинного текста ячейки.
_HEADER_ABBREVIATIONS_SUBSTRING: dict[str, str] = {
    "Cpoк": "Срок",
    "Сумmа": "Сумма",
    "N π/n$": "N п/п",
    "HДT": "НДТ",
    "HДТ": "НДТ",
    "наименование 3B": "наименование ЗВ",
    "MΓ/∂M3$": "мг/дм³",
    "MΓ/∂M3": "мг/дм³",
    "MΓ/M3$": "мг/дм³",
    "τ/Γ": "т/г",
    "τ/r": "т/г",
    "τ/": "т/г",
    "XПK": "ХПК",
    "XПК": "ХПК",
    "5r": "5г",
    "dO": "до",
    "mr/": "мг/",
    "дмз": "дм³",
    # «мг/нм³» и «кг/т»: греческая Γ вместо «г», латинские H/M/T вместо «н»/«м»/«т»,
    # надстрочная «³», прочитанная обычной цифрой.
    "MΓ/HM33": "мг/нм³",
    "MΓ/HM3": "мг/нм³",
    "MΓ/HM": "мг/нм³",
    "Mr/HM33": "мг/нм³",
    "Mr/HM3": "мг/нм³",
    "KΓ/T": "кг/т",
}

# Замены только для всего текста ячейки: ключи — обычные буквы, которые
# встречаются внутри слов («з» в «Взвешенные»), подстрокой их менять нельзя.
# Целиком такой текст бывает, например, в строке нумерации столбцов «1 2 з 4».
_HEADER_ABBREVIATIONS_EXACT: dict[str, str] = {
    "з": "3",
    # «г.» (год) с греческой Γ вместо «г»
    "Γ.": "г.",
}


_LATIN_TO_CYRILLIC_LOOKALIKES = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    # строчные: «Меtаn» → «Метан»
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
})


def _fix_mixed_script_word(word: str) -> str:
    """Заменяет в слове со смешанным алфавитом латинские двойники кириллицей.

    Латинская буква рядом с кириллицей в одном слове («Cpок», «Вeрсия») почти
    всегда ошибка OCR. Слова без кириллицы («HDT») не меняются.
    """
    if not re.search(r"[А-Яа-яЁё]", word):
        return word
    return word.translate(_LATIN_TO_CYRILLIC_LOOKALIKES)


def normalize_header_text(text: str) -> str:
    """Исправляет известные искажения OCR в коротких сокращениях шапок таблиц.

    Сначала проверяется совпадение всего текста ячейки
    (``_HEADER_ABBREVIATIONS_EXACT``), затем выполняются замены подстрок
    (``_HEADER_ABBREVIATIONS_SUBSTRING``) от длинных к коротким, чтобы короткий
    образец не испортил длинный («τ/Γ» раньше «τ/»). В конце исправляются слова
    со смешанным алфавитом.

    Args:
        text: Текст ячейки.

    Returns:
        Исправленный текст.
    """
    if text in _HEADER_ABBREVIATIONS_EXACT:
        return _HEADER_ABBREVIATIONS_EXACT[text]
    result = text
    for wrong, right in sorted(
        _HEADER_ABBREVIATIONS_SUBSTRING.items(), key=lambda kv: -len(kv[0])
    ):
        result = result.replace(wrong, right)
    return re.sub(r"\S+", lambda m: _fix_mixed_script_word(m.group(0)), result)


def grid_from_cells(cells: list[TableCell]) -> list[list[str | None]]:
    """Строит двумерную сетку текста таблицы.

    Текст объединённой ячейки стоит в её левой верхней позиции, остальные
    позиции объединения — None.
    """
    if not cells:
        return []
    n_rows = max(c.row + c.row_span for c in cells)
    n_cols = max(c.col + c.col_span for c in cells)
    grid: list[list[str | None]] = [[None] * n_cols for _ in range(n_rows)]
    for c in cells:
        grid[c.row][c.col] = c.text
    return grid
