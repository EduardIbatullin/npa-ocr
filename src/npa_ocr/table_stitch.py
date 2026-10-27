"""Сшивка таблицы, продолжающейся на следующей странице. Перенесено из спайка npa-vlm-ocr
(scripts/stitch.py): на 14 документах спайка 117 таблиц страниц сшиваются в 37.

Пару «продолжение» находит ocr.stitch_tables (последняя таблица страницы доходит до низа, первая
таблица следующей начинается сверху, число столбцов одно). Здесь — склейка ячеек:

1. повтор шапки или строки нумерации столбцов («1 2 3 4») в начале продолжения убирается;
2. строка, разрезанная страницей (Приказ №191, стр. 13: сверху «монооксид)», остальные клетки
   пусты), дописывается в последнюю строку верхней части;
3. объединённая ячейка, продолжающаяся на новой странице (191, стр. 15: «Обезвоживание…» начато
   на стр. 14, сверху стр. 15 клетка группы пуста), продлевается вниз (row_span).
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
    return " ".join(cell.text.split())


def _row(cells: list[TableCell], r: int) -> list[TableCell]:
    return sorted((x for x in cells if x.row == r), key=lambda x: x.col)


def _covering(cells: list[TableCell], r: int, c: int) -> TableCell | None:
    return next((x for x in cells
                 if x.row <= r < x.row + x.row_span and x.col <= c < x.col + x.col_span), None)


def _is_numbering(row: list[TableCell]) -> bool:
    t = [_flat(x) for x in row]
    return len(t) >= 3 and all(re.fullmatch(r"\d{1,2}", s) for s in t) and int(t[0]) <= 2


def _line_flags(cell: TableCell) -> tuple[bool, ...]:
    n = len(cell.text.split("\n"))
    return cell.low_lines if len(cell.low_lines) == n else (False,) * n


def _append_tail(top: TableCell, tail: TableCell) -> TableCell:
    """Хвост строки дописывается к последней строке текста ячейки через пробел."""
    a, b = _line_flags(top), _line_flags(tail)
    flags = a[:-1] + (a[-1] or b[0],) + b[1:]
    return replace(top, text=f"{top.text} {tail.text}".strip(), low_lines=flags if any(flags) else ())


def _shift(cells: list[TableCell], dr: int) -> list[TableCell]:
    return [replace(x, row=x.row + dr) for x in cells]


def _drop_first_row(cells: list[TableCell]) -> list[TableCell]:
    """Убрать строку 0: её одиночные ячейки исчезают, объединения через неё укорачиваются."""
    out = []
    for x in cells:
        if x.row == 0 and x.row_span == 1:
            continue
        out.append(replace(x, row_span=x.row_span - 1) if x.row == 0 else replace(x, row=x.row - 1))
    return out


def merge_continuation(
    top: list[TableCell], top_rows: int, cont: list[TableCell], cont_rows: int,
) -> tuple[list[TableCell], int, list[str]]:
    """Склейка верхней части таблицы и её продолжения со следующей страницы.
    Возвращает ячейки, число строк и список сделанного (для журнала)."""
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
    if filled and len(filled) < len(first) and all(_TAIL_RE.match(_flat(x)) for x in filled):
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
        if x.row != 0 or _flat(x):
            continue
        target = _covering(a, top_rows - 1, x.col)
        if target is None or not _flat(target) or target.col != x.col or target.col_span != x.col_span:
            continue
        # столбец продолжения не должен быть пуст целиком (пустая форма) — ниже в нём есть текст
        if not any(_flat(y) for y in b if y.col == x.col and y.row >= x.row_span) and x.row_span < cont_rows:
            continue
        a[a.index(target)] = replace(target, row_span=target.row_span + x.row_span)
        removed.add(i)
        log.append(f"продлено объединение «{_flat(target)[:30]}» на {x.row_span} строк(и)")
    b = [x for i, x in enumerate(b) if i not in removed]
    return a + _shift(b, top_rows), top_rows + cont_rows, log
