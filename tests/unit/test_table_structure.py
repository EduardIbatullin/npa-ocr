"""Тесты чистой логики структуры таблиц (без моделей TATR)."""

from __future__ import annotations

from npa_ocr.ocr_table import TableCell
from npa_ocr.table_structure import (
    TableStructure,
    build_cells,
    drop_empty_columns,
    drop_empty_rows,
    filter_false_positive_spans,
    is_plausible_table,
    join_fragment_lines,
    merge_header_bands,
    split_oversized_rows,
)


def _grid_2x3():
    rows = [(0, 0, 300, 50), (0, 50, 300, 100)]
    cols = [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]
    return TableStructure(rows=rows, cols=cols, spans=[])


def test_build_cells_plain_grid_assigns_text_by_center():
    s = _grid_2x3()
    texts = [((10, 10, 90, 40), "a"), ((110, 60, 190, 90), "b")]
    cells, n_rows, n_cols = build_cells(s, texts)
    assert (n_rows, n_cols) == (2, 3)
    by_pos = {(c.row, c.col): c.text for c in cells}
    assert by_pos[(0, 0)] == "a"
    assert by_pos[(1, 1)] == "b"
    assert by_pos[(0, 2)] == ""


def test_build_cells_spanning_cell_covers_its_grid_and_takes_text():
    s = _grid_2x3()
    s = TableStructure(rows=s.rows, cols=s.cols, spans=[(0, 0, 200, 50)])  # верхний ряд, два столбца
    texts = [((10, 10, 180, 40), "шапка"), ((210, 60, 290, 90), "x")]
    cells, n_rows, n_cols = build_cells(s, texts)
    span = next(c for c in cells if c.row_span > 1 or c.col_span > 1)
    assert (span.row, span.col, span.row_span, span.col_span) == (0, 0, 1, 2)
    assert span.text == "шапка"
    assert all(not (c.row == 0 and c.col == 1) for c in cells)  # накрытая ячейка не дублируется


def test_drop_empty_columns_removes_artifact_column_and_reports_it():
    s = _grid_2x3()
    texts = [((10, 10, 90, 40), "a"), ((110, 60, 190, 90), "b")]  # третий столбец пуст
    fixed, dropped = drop_empty_columns(s, texts)
    assert dropped == 1
    assert len(fixed.cols) == 2


def test_drop_empty_rows_removes_rows_without_text():
    s = _grid_2x3()
    texts = [((10, 10, 90, 40), "a")]
    fixed = drop_empty_rows(s, texts)
    assert len(fixed.rows) == 1


def _grid_3x2():
    rows = [(0, 0, 200, 50), (0, 50, 200, 100), (0, 100, 200, 150)]
    cols = [(0, 0, 100, 150), (100, 0, 200, 150)]
    return TableStructure(rows=rows, cols=cols, spans=[])


def test_filter_false_positive_spans_keeps_genuine_span_with_one_value():
    """Приказ №191 (2026-10-24): настоящая объединённая ячейка несёт одно
    значение (текст есть только в одной из охватываемых позиций)."""
    s = _grid_3x2()
    span = (0, 0, 100, 150)  # весь столбец 0, три строки
    s = TableStructure(rows=s.rows, cols=s.cols, spans=[span])
    texts = [
        ((10, 10, 90, 40), "общее значение"),  # только в (row0, col0)
        ((110, 10, 190, 40), "a"),
        ((110, 60, 190, 90), "b"),
        ((110, 110, 190, 140), "c"),
    ]
    fixed = filter_false_positive_spans(s, texts)
    assert fixed.spans == [span]


def _grid_1row_2col(row_height: float = 300):
    rows = [(0, 0, 300, row_height)]
    cols = [(0, 0, 150, row_height), (150, 0, 300, row_height)]
    return TableStructure(rows=rows, cols=cols, spans=[])


def test_split_oversized_rows_splits_when_two_columns_agree_on_gap():
    """Приказ №191, пункт 3 (2026-10-24), страница 15: TATR рисует слишком
    высокую полосу строки без всякого объединения (span) — две независимые
    короткие строки с большим разрывом между ними (как два значения
    «мг/нм³» на реальной странице) должны разъехаться по разным строкам.
    Разрез подтверждён ДВУМЯ столбцами (вещество + единица измерения) — как
    на реальном подтверждённом случае, см. _ROW_SPLIT_MIN_COLUMNS."""
    s = _grid_1row_2col()
    texts = [
        ((10, 20, 140, 55), "вещество1"),
        ((10, 130, 140, 165), "вещество2"),
        ((160, 20, 290, 55), "мг/нм3"),  # разрыв 75px (>= высоты строки) — новая строка
        ((160, 130, 290, 165), "мг/нм3"),
    ]
    fixed = split_oversized_rows(s, texts)
    assert len(fixed.rows) == 2
    cells, n_rows, n_cols = build_cells(fixed, texts)
    by_pos = {(c.row, c.col): c.text for c in cells}
    assert by_pos[(0, 0)] == "вещество1"
    assert by_pos[(1, 0)] == "вещество2"
    assert by_pos[(0, 1)] == "мг/нм3"
    assert by_pos[(1, 1)] == "мг/нм3"


def test_split_oversized_rows_ignores_gap_confirmed_by_only_one_column():
    """Регрессия, пойманная на реальном документе: перенос одного длинного
    значения («Азота оксид (азот (II) оксид; азот монооксид)») на две
    визуальные строки случайно дал разрыв шире порога ТОЛЬКО в одном
    столбце (остальные столбцы для этой строки были пустыми) — такой разрыв
    не должен разбивать строку, иначе значение режется пополам."""
    s = _grid_1row_2col()
    texts = [
        ((160, 20, 290, 55), "Азота оксид (азот"),
        ((160, 130, 290, 165), "монооксид)"),  # большой разрыв, но столбец только один
    ]
    fixed = split_oversized_rows(s, texts)
    assert len(fixed.rows) == 1
    cells, n_rows, n_cols = build_cells(fixed, texts)
    by_pos = {(c.row, c.col): c.text for c in cells}
    assert by_pos[(0, 1)] == "Азота оксид (азот монооксид)"


def test_split_oversized_rows_keeps_single_wrapped_value_intact():
    """Обычный перенос длинного значения на несколько строк (малый разрыв
    между строками одного и того же значения) не должен разбиваться."""
    s = _grid_1row_2col()
    texts = [
        ((160, 20, 290, 55), "Пыль"),
        ((160, 60, 290, 95), "неорганическая"),  # разрыв 5px (<< высоты строки)
    ]
    fixed = split_oversized_rows(s, texts)
    assert len(fixed.rows) == 1
    cells, n_rows, n_cols = build_cells(fixed, texts)
    by_pos = {(c.row, c.col): c.text for c in cells}
    assert by_pos[(0, 1)] == "Пыль неорганическая"


def test_filter_false_positive_spans_drops_span_with_independent_values_per_row():
    """Три разные строки с разными значениями, ошибочно размеченные TATR как
    одна объединённая ячейка — отбрасываем объединение, текст должен
    остаться в своих отдельных строках (не склеиваться)."""
    s = _grid_3x2()
    span = (0, 0, 100, 150)  # весь столбец 0, три строки — ложное объединение
    s = TableStructure(rows=s.rows, cols=s.cols, spans=[span])
    texts = [
        ((10, 10, 90, 40), "вещество1"),
        ((10, 60, 90, 90), "вещество2"),
        ((10, 110, 90, 140), "вещество3"),
    ]
    fixed = filter_false_positive_spans(s, texts)
    assert fixed.spans == []
    cells, n_rows, n_cols = build_cells(fixed, texts)
    by_pos = {(c.row, c.col): c.text for c in cells}
    assert by_pos[(0, 0)] == "вещество1"
    assert by_pos[(1, 0)] == "вещество2"
    assert by_pos[(2, 0)] == "вещество3"


def test_merge_header_bands_adds_only_non_overlapping_headers():
    rows = [(0, 50, 300, 100)]
    headers = [(0, 0, 300, 45), (0, 60, 300, 90)]  # первый — отдельная шапка, второй — дубль строки
    merged = merge_header_bands(rows, headers)
    assert merged[0] == (0, 0, 300, 45)
    assert len(merged) == 2


def test_expand_rows_to_text_covers_clipped_text():
    from npa_ocr.table_structure import expand_rows_to_text

    s = TableStructure(rows=[(0.0, 100.0, 300.0, 140.0)], cols=[(0.0, 0.0, 300.0, 100.0)], spans=[])
    texts = [((10.0, 120.0, 200.0, 160.0), "Крупнейшие")]  # текст выходит за нижнюю границу полосы
    fixed = expand_rows_to_text(s, texts)
    assert fixed.rows[0][3] >= 160.0


def test_text_assigned_to_row_with_largest_overlap_when_bands_overlap():
    s = TableStructure(
        rows=[(0.0, 0.0, 300.0, 100.0), (0.0, 90.0, 300.0, 200.0)],  # полосы перекрываются
        cols=[(0.0, 0.0, 300.0, 200.0)],
        spans=[],
    )
    texts = [((10.0, 95.0, 100.0, 190.0), "вторая")]  # почти полностью во второй полосе
    cells, _, _ = build_cells(s, texts)
    assert next(c for c in cells if c.row == 1).text == "вторая"


def test_text_crop_box_contains_clipped_text_extent():
    from npa_ocr.table_structure import cell_text_boxes, text_crop_box

    s = TableStructure(rows=[(0.0, 100.0, 300.0, 140.0)], cols=[(0.0, 0.0, 300.0, 200.0)], spans=[])
    texts = [((10.0, 120.0, 200.0, 160.0), "Крупнейшие")]
    blocks = cell_text_boxes(s, texts)[(0, 0)]
    x1, y1, x2, y2 = text_crop_box(blocks)
    assert y1 <= 120.0 and y2 >= 160.0  # полный текст внутри кропа, а не обрезанный по полосе


def test_merge_overlapping_row_bands_collapses_header_bands_into_one():
    from npa_ocr.table_structure import merge_overlapping_row_bands

    # Полосы 233–336 и 286–335 — одна строка (вторая внутри первой); 289–416 перекрывает
    # её лишь на ~46% и остаётся отдельной строкой; 413–464 — следующая строка.
    merged = merge_overlapping_row_bands([(0, 233, 300, 336), (0, 286, 300, 335), (0, 289, 300, 416), (0, 413, 300, 464)])
    assert len(merged) == 3
    assert merged[0][1] == 233 and merged[0][3] == 336


def test_span_covers_rows_by_overlap_not_center():
    from npa_ocr.table_structure import TableStructure, build_cells

    s = TableStructure(
        rows=[(0.0, 0.0, 300.0, 100.0), (0.0, 100.0, 300.0, 200.0)],
        cols=[(0.0, 0.0, 300.0, 200.0)],
        spans=[(0.0, 40.0, 300.0, 180.0)],  # накрывает большую часть обеих строк
    )
    cells, _, _ = build_cells(s, [((10.0, 50.0, 100.0, 70.0), "объединено")])
    span = cells[0]
    assert (span.row, span.row_span) == (0, 2)


def test_text_crop_box_keeps_word_crossing_column_boundary():
    from npa_ocr.table_structure import text_crop_box

    # слово «выполнения» пересекает границу столбца x=150: кроп должен сохранить его целиком
    blocks = [((120.0, 10.0, 190.0, 40.0), "выполнения")]
    x1, y1, x2, y2 = text_crop_box(blocks)
    assert x2 >= 190.0


def test_join_fragments_glues_word_parts_and_spaces_words():
    from npa_ocr.table_structure import join_fragments

    items = [
        ((100.0, 10.0, 150.0, 40.0), "выполнени"),
        ((151.0, 10.0, 160.0, 40.0), "я"),  # тот же слов — почти вплотную
        ((200.0, 10.0, 260.0, 40.0), "по"),  # другое слово — заметный пробел
        ((90.0, 60.0, 200.0, 90.0), "годам"),  # другая строка текста
    ]
    assert join_fragments(items) == "выполнения по\nгодам"


def test_join_fragment_lines_does_not_glue_overlapping_adjacent_words():
    # Реальные боксы RapidOCR (eslav_PP-OCRv5) на кропе ячейки из реального
    # документа (Приказ Минприроды №565, таблица техпоказателей, ячейка
    # «Взвешенные вещества») — два РАЗНЫХ слова, но их боксы слегка
    # перекрываются по x (gap = 227-238 = -11), раньше это читалось как один
    # разорванный токен и склеивалось без пробела («Взвешенныевещества»),
    # см. docs/plan.md.
    items = [
        ((9.0, 30.0, 238.0, 79.0), "Взвешенные", 0.98892),
        ((227.0, 36.0, 396.0, 79.0), "вещества", 0.9904),
    ]
    assert join_fragment_lines(items) == [("Взвешенные вещества", 0.98892)]


def test_join_fragment_lines_does_not_glue_overlapping_words_second_real_example():
    # Та же проверка на другой реальной паре слов того же документа (ячейка
    # «Серы диоксид», gap = 174-186 = -12).
    items = [
        ((78.0, 0.0, 186.0, 49.0), "Серы", 0.95),
        ((174.0, 0.0, 331.0, 43.0), "диоксид", 0.95),
    ]
    assert join_fragment_lines(items) == [("Серы диоксид", 0.95)]


def test_join_fragment_lines_drops_duplicate_superscript_box():
    # Реальные боксы RapidOCR на двух РАЗНЫХ кропах одной и той же ячейки
    # («мг/нм³») в разных местах документа (Приказ №565) — надстрочная «³»
    # детектируется дважды: один раз внутри основного бокса («MΓ/HM3»), и ещё
    # раз отдельным мелким боксом вокруг того же символа (~86% перекрытия
    # площади с основным боксом). Второй бокс должен быть отброшен целиком —
    # ни с пробелом, ни без, см. docs/plan.md.
    items = [
        ((26.0, 18.0, 147.0, 77.0), "MΓ/HM3", 0.9),
        ((128.0, 22.0, 150.0, 48.0), "3", 0.9),
    ]
    assert join_fragment_lines(items) == [("MΓ/HM3", 0.9)]


def test_join_fragment_lines_keeps_short_fragment_that_is_not_contained():
    # Короткий фрагмент («3» и т.п.) сам по себе не повод его выбрасывать —
    # только когда он почти целиком вложен в другой бокс той же строки.
    # Здесь второй бокс лишь слегка соприкасается с первым (не перекрыт) —
    # обычное отдельное короткое слово/число, должно остаться.
    items = [
        ((0.0, 0.0, 50.0, 30.0), "в", 0.9),
        ((55.0, 0.0, 80.0, 30.0), "3", 0.9),
    ]
    assert join_fragment_lines(items) == [("в 3", 0.9)]


def _cells(rows: int, cols: int, text_by_pos: dict[tuple[int, int], str]) -> list[TableCell]:
    return [
        TableCell(row=r, col=c, row_span=1, col_span=1, text=text_by_pos.get((r, c), ""))
        for r in range(rows)
        for c in range(cols)
    ]


def test_is_plausible_table_rejects_real_false_positive_from_tatr_detection():
    # Реальная сетка с развёрнутого сервера (Приказ Минприроды №565, стр. 1,
    # bbox=[165,1733,1607,2150]) — TATR (table-transformer-detection) дал
    # score=0.862 на нижней части страницы (конец пункта 3 + строка подписи
    # министра), приняв её за таблицу 5x12. На самом деле «столбцы» —
    # случайные межсловные пробелы одной строки текста (см. docs/plan.md).
    text_by_pos = {
        (0, 0): "(зарегистрирован", (0, 2): "Министерством", (0, 4): "юстициИ",
        (0, 7): "Российской", (0, 11): "Федерации",
        (1, 0): "28 января 2021 Г.", (1, 1): "регистрационный", (1, 2): "$^$", (1, 3): "62264).",
        (2, 0): "3 Настоящий", (2, 1): "приказ", (2, 2): "вступает", (2, 3): "B", (2, 4): "силу",
        (2, 5): "C", (2, 6): "1", (2, 7): "apra", (2, 8): "2024", (2, 9): "г.", (2, 11): "действует",
        (3, 0): "B течение шести лет.",
        (4, 0): "Министр", (4, 11): "A.A. Козлов",
    }
    cells = _cells(5, 12, text_by_pos)
    assert is_plausible_table(cells, 5, 12) is False


def test_is_plausible_table_accepts_real_genuine_table():
    # Реальная сетка с того же сервера (Постановление №1430, стр. 11,
    # bbox=[193,871,1455,1363]) — настоящая таблица нормативов 6x6, TATR
    # нашёл её с score=0.996. Объединённые ячейки шапки учтены col_span=5/
    # row_span=2, как их фактически строит build_cells.
    cells = [
        TableCell(row=0, col=0, row_span=2, col_span=1, text="Категории\nВОДНЫХ\nобъектов"),
        TableCell(row=0, col=1, row_span=1, col_span=5, text="Технологические пс\nзагряз"),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="взвешенные\nвещества"),
        TableCell(row=1, col=2, row_span=1, col_span=1, text="нефтепродукты"),
        TableCell(row=1, col=3, row_span=1, col_span=1, text=""),
        TableCell(row=1, col=4, row_span=1, col_span=1, text="БПК5"),
        TableCell(row=1, col=5, row_span=1, col_span=1, text="фосфатов"),
        TableCell(row=2, col=0, row_span=1, col_span=1, text="A"),
        TableCell(row=2, col=1, row_span=1, col_span=1, text="5"),
        TableCell(row=2, col=2, row_span=1, col_span=1, text="0,3"),
        TableCell(row=2, col=3, row_span=1, col_span=1, text="30"),
        TableCell(row=2, col=4, row_span=1, col_span=1, text="5"),
        TableCell(row=2, col=5, row_span=1, col_span=1, text="0,5"),
        TableCell(row=3, col=0, row_span=1, col_span=1, text="B"),
        TableCell(row=3, col=1, row_span=1, col_span=1, text="15"),
        TableCell(row=3, col=2, row_span=1, col_span=1, text="1"),
        TableCell(row=3, col=3, row_span=1, col_span=1, text="50*"),
        TableCell(row=3, col=4, row_span=1, col_span=1, text="10"),
        TableCell(row=3, col=5, row_span=1, col_span=1, text="1"),
        TableCell(row=4, col=0, row_span=1, col_span=1, text="B"),
        TableCell(row=4, col=1, row_span=1, col_span=1, text="15"),
        TableCell(row=4, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=4, col=3, row_span=1, col_span=1, text="60"),
        TableCell(row=4, col=4, row_span=1, col_span=1, text="12"),
        TableCell(row=4, col=5, row_span=1, col_span=1, text="1"),
        TableCell(row=5, col=0, row_span=1, col_span=1, text="Γ"),
        TableCell(row=5, col=1, row_span=1, col_span=1, text="15"),
        TableCell(row=5, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=5, col=3, row_span=1, col_span=1, text="60"),
        TableCell(row=5, col=4, row_span=1, col_span=1, text="12"),
        TableCell(row=5, col=5, row_span=1, col_span=1, text="5"),
    ]
    assert is_plausible_table(cells, 6, 6) is True


def test_is_plausible_table_rejects_1x1_grid():
    # Реальные документы (Постановление №1430, стр. 2-4; Постановление №710,
    # стр. 5): TATR иногда принимает штамп нумерации страницы или случайный
    # фрагмент обычного абзаца за «таблицу» из одной ячейки — единственная
    # ячейка никогда не таблица, отбраковывается всегда (2026-10-24).
    cells = [TableCell(row=0, col=0, row_span=1, col_span=1, text="678924")]
    assert is_plausible_table(cells, 1, 1) is False


def test_is_plausible_table_does_not_reject_other_small_grids():
    # Сетки меньше 2x2, но не 1x1 (одна строка из многих столбцов или один
    # столбец из многих строк) — по-прежнему вне охвата проверки по
    # заполненности столбцов, см. docs/plan.md.
    row_cells = [TableCell(row=0, col=c, row_span=1, col_span=1, text=str(c)) for c in range(5)]
    assert is_plausible_table(row_cells, 1, 5) is True

    col_cells = [TableCell(row=r, col=0, row_span=1, col_span=1, text=str(r)) for r in range(5)]
    assert is_plausible_table(col_cells, 5, 1) is True
