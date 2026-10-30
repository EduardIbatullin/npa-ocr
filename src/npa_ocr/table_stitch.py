"""Сшивка таблицы, продолжающейся на следующей странице.

Пару «таблица и её продолжение» находит :func:`npa_ocr.ocr.stitch_tables`:
последняя таблица страницы доходит до низа, первая таблица следующей страницы
начинается сверху, число столбцов совпадает. Этот модуль склеивает ячейки:

1. Повтор шапки или строки нумерации столбцов («1 2 3 4») в начале продолжения
   убирается.
2. Строка, разрезанная границей страницы (вверху продолжения заполнены только
   хвосты текста, остальные ячейки пусты), дописывается в последнюю строку
   верхней части.
3. Объединённая ячейка, продолжающаяся на новой странице (вверху продолжения
   ячейка группы пуста), продлевается вниз через ``row_span``.
"""

from __future__ import annotations

import re
from dataclasses import replace

from npa_ocr.ocr_table import TableCell
from npa_ocr.vlm_tables import norm

# строки шапки верхней части, с которыми сравнивается начало продолжения
_HEADER_ROWS = 6
# хвост разрезанной строки начинается со строчной буквы, скобки или знака препинания
_TAIL_RE = re.compile(r"[a-zа-яё()\]»,;.]")


def _flat(cell: TableCell) -> str:
    """Текст ячейки одной строкой с одиночными пробелами."""
    return " ".join(cell.text.split())


def _row(cells: list[TableCell], r: int) -> list[TableCell]:
    """Ячейки, начинающиеся в строке ``r``, слева направо."""
    return sorted((x for x in cells if x.row == r), key=lambda x: x.col)


def _covering(cells: list[TableCell], r: int, c: int) -> TableCell | None:
    """Ячейка, покрывающая позицию (r, c) с учётом объединений."""
    return next((x for x in cells
                 if x.row <= r < x.row + x.row_span and x.col <= c < x.col + x.col_span), None)


def _is_numbering(row: list[TableCell]) -> bool:
    """Проверяет, что строка — нумерация столбцов «1 2 3 …»."""
    t = [_flat(x) for x in row]
    return len(t) >= 3 and all(re.fullmatch(r"\d{1,2}", s) for s in t) and int(t[0]) <= 2


def _line_flags(cell: TableCell) -> tuple[bool, ...]:
    """Признаки неуверенного распознавания по строкам текста ячейки."""
    n = len(cell.text.split("\n"))
    return cell.low_lines if len(cell.low_lines) == n else (False,) * n


def _append_tail(top: TableCell, tail: TableCell) -> TableCell:
    """Дописывает хвост строки к последней строке текста ячейки через пробел."""
    a, b = _line_flags(top), _line_flags(tail)
    flags = a[:-1] + (a[-1] or b[0],) + b[1:]
    return replace(top, text=f"{top.text} {tail.text}".strip(), low_lines=flags if any(flags) else ())


def align_columns(
    top: list[TableCell], top_edges: list[float], cont: list[TableCell], cont_edges: list[float],
) -> tuple[list[TableCell], list[TableCell], list[float]] | None:
    """Приводит близкие границы граф двух страниц к общей элементарной сетке.

    Дополнительная граница внутри объединённой ячейки допустима. Если
    большинство границ различается, таблицы не считаются продолжением.
    """
    if len(top_edges) < 2 or len(cont_edges) < 2:
        return None
    tolerance = 0.012
    shared = sum(any(abs(a - b) <= tolerance for b in cont_edges) for a in top_edges)
    # A continuation can use fewer divisions (a new section of the same
    # table). Requiring 70% of the larger grid split such continuations.
    subset = (min(len(top_edges), len(cont_edges)) >= 4
              and abs(top_edges[0] - cont_edges[0]) <= tolerance
              and abs(top_edges[-1] - cont_edges[-1]) <= tolerance
              and sum(any(abs(a - b) <= tolerance for a in top_edges) for b in cont_edges)
                  >= .9 * len(cont_edges))
    if shared < 0.7 * max(len(top_edges), len(cont_edges)) and not subset:
        return None
    if len(top_edges) == len(cont_edges) and all(abs(a-b) <= .02 for a,b in zip(top_edges,cont_edges)):
        # The same logical columns, with small scan/perspective drift.
        # A union would create an artificial narrow column inside the body.
        return list(top), list(cont), list(top_edges)
    groups: list[list[float]] = []
    for edge in sorted(top_edges + cont_edges):
        if groups and edge - groups[-1][0] <= tolerance:
            groups[-1].append(edge)
        else:
            groups.append([edge])
    edges = [sum(g) / len(g) for g in groups]

    def remap(cells: list[TableCell], old: list[float]) -> list[TableCell]:
        mapping = [min(range(len(edges)), key=lambda i: abs(edges[i] - x)) for x in old]
        return [replace(c, col=mapping[c.col], col_span=mapping[c.col + c.col_span] - mapping[c.col]) for c in cells]

    return remap(top, top_edges), remap(cont, cont_edges), edges


def _shift(cells: list[TableCell], dr: int) -> list[TableCell]:
    """Сдвигает ячейки на ``dr`` строк."""
    return [replace(x, row=x.row + dr) for x in cells]


def _drop_first_row(cells: list[TableCell]) -> list[TableCell]:
    """Убирает строку 0: её одиночные ячейки удаляются, объединения укорачиваются."""
    out = []
    for x in cells:
        if x.row == 0 and x.row_span == 1:
            continue
        out.append(replace(x, row_span=x.row_span - 1) if x.row == 0 else replace(x, row=x.row - 1))
    return out


def merge_continuation(
    top: list[TableCell], top_rows: int, cont: list[TableCell], cont_rows: int,
) -> tuple[list[TableCell], int, list[str]]:
    """Склеивает верхнюю часть таблицы с продолжением со следующей страницы.

    Args:
        top: Ячейки верхней части.
        top_rows: Число строк верхней части.
        cont: Ячейки продолжения.
        cont_rows: Число строк продолжения.

    Returns:
        Ячейки склеенной таблицы, число её строк и список выполненных
        действий для журнала.
    """
    a = list(top)
    b = list(cont)
    log: list[str] = []
    # 1. повтор шапки / нумерации столбцов
    header = [[norm(_flat(x)) for x in _row(a, r)] for r in range(min(top_rows, _HEADER_ROWS))]
    while cont_rows:
        first = _row(b, 0)
        if not first or any(x.row_span > 1 for x in first):
            break
        if not (_is_numbering(first) or [norm(_flat(x)) for x in first] in header):
            break
        b = _shift([x for x in b if x.row > 0], -1)
        cont_rows -= 1
        log.append("убран повтор шапки/нумерации")
    if not cont_rows:
        return a, top_rows, log
    # 2. строка, разрезанная страницей: часть клеток первой строки пуста, а непустые — хвосты
    # текста последней строки верхней части (объединения через эту строку вниз продлит шаг 3)
    first = [x for x in _row(b, 0) if x.row_span == 1]
    filled = [x for x in first if _flat(x)]
    def is_tail(cell: TableCell) -> bool:
        """Продолжение текста, в том числе процент после «более/менее»."""
        if _TAIL_RE.match(_flat(cell)):
            return True
        target = _covering(a, top_rows - 1, cell.col)
        return bool(target and re.search(r"(?:более|менее)\s*$", _flat(target), re.I)
                    and re.match(r"\d+(?:[.,]\d+)?\s*%", _flat(cell)))

    if filled and len(filled) < len(first) and all(is_tail(x) for x in filled):
        for x in filled:
            target = _covering(a, top_rows - 1, x.col)
            if target is not None:
                a[a.index(target)] = _append_tail(target, x)
        b = _drop_first_row(b)
        cont_rows -= 1
        log.append("дописана разрезанная строка: " + "; ".join(_flat(x)[:30] for x in filled))
    # 3. объединения, продолжающиеся на следующей странице
    removed: set[int] = set()
    for i, x in enumerate(b):
        if x.row != 0:
            continue
        target = _covering(a, top_rows - 1, x.col)
        if target is None or not _flat(target) or target.col != x.col or target.col_span != x.col_span:
            continue
        if _flat(x):
            index = _covering(b, 0, 0)
            if (x.col == 1 and x.row_span > 1 and index is not None and not _flat(index)
                    and _TAIL_RE.match(_flat(x)) and re.search(r"[а-яёA-Za-z]", _flat(target))
                    and len(_flat(target)) <= 50 and not re.search(r"[\d()]", _flat(target))):
                joined = _append_tail(target, x)
                a[a.index(target)] = replace(joined, row_span=target.row_span + x.row_span)
                removed.add(i)
                log.append("соединено разрезанное наименование")
            continue
        # Пустота на новой странице не доказывает продолжение числовой
        # нормы. Номер позиции в первой графе может продолжаться, но
        # норматив из предыдущей строки нельзя переносить в новую.
        if x.col > 0 and re.fullmatch(r"[\d\s.,*%–—−-]+", _flat(target)):
            continue
        # столбец продолжения не должен быть пуст целиком (пустая форма) — ниже в нём есть текст
        if not any(_flat(y) for y in b if y.col == x.col and y.row >= x.row_span) and x.row_span < cont_rows:
            continue
        a[a.index(target)] = replace(target, row_span=target.row_span + x.row_span)
        removed.add(i)
        log.append(f"продлено объединение «{_flat(target)[:30]}» на {x.row_span} строк(и)")
    b = [x for i, x in enumerate(b) if i not in removed]
    return a + _shift(b, top_rows), top_rows + cont_rows, log
