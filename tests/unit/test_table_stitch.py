"""Тесты сшивки таблиц между страницами (table_stitch.py, ocr.stitch_tables). Реальные случаи —
страницы прогона npa-ocr (table_engine="vlm") на документах спайка npa-vlm-ocr; на всех
14 документах спайка сшивка npa-ocr совпадает со сшивкой спайка (40 таблиц из 117)."""

from __future__ import annotations

import json
from pathlib import Path

from npa_ocr.ocr import PageResult, TableBlock, TextBlock, _load_page, build_docx, stitch_tables


def test_blank_continuation_does_not_inherit_numeric_norm():
    from npa_ocr.table_stitch import merge_continuation
    top = [TableCell(0, 0, 1, 1, "36", []), TableCell(0, 1, 1, 1, "15", [])]
    cont = [TableCell(0, 0, 1, 1, "", []), TableCell(0, 1, 1, 1, "", []),
            TableCell(1, 0, 1, 1, "37", []), TableCell(1, 1, 1, 1, "20", [])]
    cells, rows, _ = merge_continuation(top, 1, cont, 2)
    assert next(c for c in cells if c.text == "36").row_span == 2
    assert next(c for c in cells if c.text == "15").row_span == 1
    assert any(c.row == 1 and c.col == 1 and c.text == "" for c in cells)


def test_plain_note_continuation_returns_to_merged_cell_only_with_closing_quote():
    note = "Регистрационный номер CAS; класс опасности — пункт ГОСТ. Система стандартов"
    table = TableBlock(y=800, bbox=(0, 800, 800, 950), n_rows=1, n_cols=2,
                       cells=[TableCell(0, 0, 1, 2, note, [])], source_pages=[1])
    first = PageResult(1, True, [table], page_height=1000)
    tail = TextBlock(60, "безопасности труда. Вредные вещества.")
    quote = TextBlock(100, "».")
    next_text = TextBlock(200, "Следующий пункт")
    second = PageResult(2, True, [tail, quote, next_text], page_height=1000)
    output, log = stitch_tables([first, second])
    assert output[0].blocks[0].cells[0].text == note + "\n" + tail.text
    assert output[0].blocks[0].source_pages == [1, 2]
    assert output[1].blocks == [quote, next_text]
    assert first.blocks[0].cells[0].text == note
    assert second.blocks == [tail, quote, next_text]
    no_quote = PageResult(2, True, [tail, next_text], page_height=1000)
    output, _ = stitch_tables([first, no_quote])
    assert output[0].blocks[0].cells[0].text == note
from npa_ocr.ocr_table import TableCell
from npa_ocr.table_stitch import merge_continuation

FIXTURES = Path(__file__).parent / "fixtures" / "table_stitch"


def _page(doc: str, number: int) -> PageResult:
    path = FIXTURES / f"{doc}_page_{number:04d}.json"
    return _load_page(path, json.loads(path.read_text(encoding="utf-8"))["meta"])


def _tables(page: PageResult) -> list[TableBlock]:
    return [b for b in page.blocks if isinstance(b, TableBlock)]


def _cell(table: TableBlock, row: int, col: int) -> TableCell:
    return next(c for c in table.cells if (c.row, c.col) == (row, col))


def _c(row: int, col: int, text: str, row_span: int = 1, low_lines: tuple[bool, ...] = ()) -> TableCell:
    return TableCell(row=row, col=col, row_span=row_span, col_span=1, text=text, low_lines=low_lines)


# --- Реальные страницы ---


def test_split_row_tail_and_group_continue_191():
    """191 стр. 12→13: сверху стр. 13 «монооксид)» — хвост строки «Азота оксид (…; азот», остальные
    клетки строки пусты; группа «Производство титановых слитков» продолжается на стр. 13."""
    pages, log = stitch_tables([_page("191", 12), _page("191", 13)])
    assert _tables(pages[1]) == []
    (table,) = _tables(pages[0])
    assert table.n_rows == 9 + 8 - 1
    assert _cell(table, 8, 1).text == "Азота оксид (азот (II) оксид; азот монооксид)"
    group = _cell(table, 6, 0)
    assert group.text.startswith("Производство титановых слитков") and group.row_span == 5
    assert log == ["стр. 13: продолжение таблицы со стр. 12 — дописана разрезанная строка: монооксид); "
                   "продлено объединение «Производство титановых слитков» на 2 строк(и)"]


def test_split_row_tail_in_brackets_376():
    """376 стр. 3→4: хвост «(исключая метан)» начинается со скобки."""
    pages, _ = stitch_tables([_page("376", 3), _page("376", 4)])
    (table,) = _tables(pages[0])
    assert table.n_rows == 21 + 18 - 1
    assert _cell(table, 20, 1).text == "Углеводороды предельные С1-С-5 (исключая метан)"
    assert _cell(table, 17, 0).row_span == 7


def test_repeated_header_removed_156():
    """156 стр. 10→11: продолжение начинается с повтора шапки — она убирается."""
    pages, log = stitch_tables([_page("156", 10), _page("156", 11)])
    (table,) = _tables(pages[0])
    assert table.n_rows == 6 + 3 - 1
    assert [c.text for c in table.cells].count("№") == 1
    assert log == ["стр. 11: продолжение таблицы со стр. 10 — убран повтор шапки/нумерации"]


def _page_breaks(path: Path) -> int:
    import zipfile

    return zipfile.ZipFile(path).read("word/document.xml").decode("utf-8").count('w:type="page"')


def test_docx_has_no_page_break_before_continuation(tmp_path):
    """Таблица на две страницы — одна Word-таблица без разрыва страницы внутри (иначе цепочка
    на 18 страниц, как в 191, оставила бы за таблицей 17 пустых страниц)."""
    from docx import Document

    pages = [_page("191", 12), _page("191", 13)]
    build_docx(pages, tmp_path / "raw.docx", diagnostic=True)
    stitched, _ = stitch_tables(pages)
    assert [p.continues_table for p in stitched] == [False, True]
    build_docx(stitched, tmp_path / "stitched.docx", diagnostic=True)
    assert _page_breaks(tmp_path / "raw.docx") == 1
    assert _page_breaks(tmp_path / "stitched.docx") == 0
    assert [len(t.rows) for t in Document(str(tmp_path / "stitched.docx")).tables] == [16]


def test_input_pages_not_modified():
    p12, p13 = _page("191", 12), _page("191", 13)
    rows_before = _tables(p12)[0].n_rows
    stitch_tables([p12, p13])
    assert _tables(p12)[0].n_rows == rows_before and len(_tables(p13)) == 1


# --- Правила склейки ---


def test_numbering_row_removed():
    top = [_c(0, 0, "№"), _c(0, 1, "Вещество"), _c(0, 2, "Норма"),
           _c(1, 0, "1"), _c(1, 1, "2"), _c(1, 2, "3"),
           _c(2, 0, "1."), _c(2, 1, "Азот"), _c(2, 2, "10")]
    cont = [_c(0, 0, "1"), _c(0, 1, "2"), _c(0, 2, "3"),
            _c(1, 0, "2."), _c(1, 1, "Сера"), _c(1, 2, "5")]
    cells, n_rows, log = merge_continuation(top, 3, cont, 2)
    assert n_rows == 4
    assert sorted((c.row, c.text) for c in cells if c.col == 1)[-1] == (3, "Сера")
    assert log == ["убран повтор шапки/нумерации"]


def test_tail_merges_low_confidence_flags():
    """Хвост дописывается к последней строке текста ячейки; флаг низкой уверенности переносится."""
    top = [_c(0, 0, "1."), _c(0, 1, "Азота оксид\n(азот", low_lines=(False, False)), _c(0, 2, "10")]
    cont = [_c(0, 0, ""), _c(0, 1, "монооксид)\nи иные", low_lines=(True, False)), _c(0, 2, ""),
            _c(1, 0, "2."), _c(1, 1, "Сера"), _c(1, 2, "5")]
    cells, n_rows, _ = merge_continuation(top, 1, cont, 2)
    assert n_rows == 2
    merged = next(c for c in cells if (c.row, c.col) == (0, 1))
    assert merged.text == "Азота оксид\n(азот монооксид)\nи иные"
    assert merged.low_lines == (False, True, False)


def test_percentage_tail_does_not_duplicate_numeric_limits():
    top = [_c(0, 0, "5."), _c(0, 1, "Кабинет с ПК, работа на ПК более"), _c(0, 2, "500")]
    cont = [_c(0, 0, ""), _c(0, 1, "50% рабочей смены"), _c(0, 2, ""),
            _c(1, 0, "6."), _c(1, 1, "Склад"), _c(1, 2, "5")]
    cells, rows, _ = merge_continuation(top, 1, cont, 2)
    assert rows == 2
    assert next(c for c in cells if c.col == 1 and c.row == 0).text.endswith("более 50% рабочей смены")
    assert [c.text for c in cells].count("500") == 1


def test_continuation_with_new_internal_column_boundary_preserves_spans():
    from npa_ocr.table_stitch import align_columns

    top = [_c(0, 0, "2."), _c(0, 1, "Столы"), _c(0, 2, "340 мм")]
    cont = [_c(0, 0, ""), _c(0, 1, "дошкольные"), _c(0, 2, "0"), _c(0, 3, "400 мм")]
    aligned = align_columns(top, [0, .1, .6, 1], cont, [0, .101, .3, .601, 1])
    assert aligned is not None
    a, b, edges = aligned
    assert len(edges) == 5
    assert a[1].col == 1 and a[1].col_span == 2
    assert b[-1].col == 3 and b[-1].col_span == 1


def test_capitalized_first_row_is_new_row_not_tail():
    top = [_c(0, 0, "1."), _c(0, 1, "Азот"), _c(0, 2, "10")]
    cont = [_c(0, 0, ""), _c(0, 1, "Сера"), _c(0, 2, "")]
    cells, n_rows, log = merge_continuation(top, 1, cont, 1)
    assert n_rows == 2 and not any("разрезанная" in x for x in log)
    assert next(c for c in cells if (c.row, c.col) == (0, 1)).text == "Азот"
    assert next(c for c in cells if (c.row, c.col) == (1, 1)).text == "Сера"


def test_group_not_extended_into_empty_form_column():
    """Пустая клетка сверху продолжения не продлевает объединение, если весь столбец ниже пуст
    (незаполненная форма), — иначе объединение съело бы пустые строки."""
    top = [_c(0, 0, "Группа"), _c(0, 1, "Азот")]
    cont = [_c(0, 0, ""), _c(0, 1, "Сера"), _c(1, 0, ""), _c(1, 1, "Хлор")]
    cells, n_rows, log = merge_continuation(top, 1, cont, 2)
    assert n_rows == 3 and log == []
    assert next(c for c in cells if (c.row, c.col) == (0, 0)).row_span == 1


# --- Какие таблицы считаются продолжением ---


def _table(top: float, bottom: float, n_cols: int = 2, text: str = "Азот") -> TableBlock:
    return TableBlock(y=top, bbox=(100.0, top, 900.0, bottom),
                      cells=[_c(0, c, f"{text}{c}") for c in range(n_cols)], n_rows=1, n_cols=n_cols)


def _ocr_page(number: int, *blocks) -> PageResult:
    return PageResult(page_number=number, ocr_applied=True, blocks=list(blocks), page_height=1000.0)


def test_chain_over_three_pages():
    pages, log = stitch_tables([
        _ocr_page(1, TextBlock(y=50, text="Заголовок"), _table(500, 950)),
        _ocr_page(2, _table(60, 960, text="Сера")),
        _ocr_page(3, _table(60, 400, text="Хлор"), TextBlock(y=500, text="Подпись")),
    ])
    assert [len(_tables(p)) for p in pages] == [1, 0, 0]
    assert _tables(pages[0])[0].n_rows == 3
    assert [type(b).__name__ for b in pages[2].blocks] == ["TextBlock"]
    assert len(log) == 2


def test_not_continuation():
    cases = [
        # верхняя часть не доходит до низа страницы
        [_ocr_page(1, _table(500, 700)), _ocr_page(2, _table(60, 400))],
        # продолжение начинается не сверху
        [_ocr_page(1, _table(500, 950)), _ocr_page(2, _table(300, 600))],
        # разное число столбцов
        [_ocr_page(1, _table(500, 950)), _ocr_page(2, _table(60, 400, n_cols=3))],
        # между ними страница с текстовым слоем
        [_ocr_page(1, _table(500, 950)), PageResult(page_number=2, ocr_applied=False, plain_text="текст"),
         _ocr_page(3, _table(60, 400))],
        # на странице 2 нет таблицы — страница 3 не продолжает таблицу страницы 1
        [_ocr_page(1, _table(500, 950)), _ocr_page(2, TextBlock(y=100, text="текст")),
         _ocr_page(3, _table(60, 400))],
    ]
    for pages in cases:
        out, log = stitch_tables(pages)
        assert log == [] and sum(len(_tables(p)) for p in out if p.ocr_applied) == 2
