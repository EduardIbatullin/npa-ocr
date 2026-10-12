"""Тесты чистой логики структуры таблиц (без моделей TATR)."""

from __future__ import annotations

from npa_ocr.table_structure import (
    TableStructure,
    build_cells,
    drop_empty_columns,
    drop_empty_rows,
    merge_header_bands,
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
