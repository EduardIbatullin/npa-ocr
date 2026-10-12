"""Тесты сборки таблицы по геометрии ячеек — на реальных фикстурах из PaddleOCR
(см. docs/plan.md)."""

from __future__ import annotations

import json
from pathlib import Path

from npa_ocr.ocr_table import grid_from_cells, normalize_header_text, reconstruct_table

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_tables"


def _load(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


def test_reconstruct_simple_table_exact_grid():
    """Таблица без объединений (3 колонки x 7 строк) — геометрия и текст у этого
    образца верны (один из двух реальных экспериментов, где распознавание прошло
    без ошибок), поэтому ожидаем точную сетку, без допусков."""
    data = _load("simple_table_0.json")
    cells = reconstruct_table(data["cell_box_list"], data["rec_boxes"], data["rec_texts"])
    grid = grid_from_cells(cells)

    assert len(grid) == 7
    assert all(len(row) == 3 for row in grid)
    assert grid[0] == ["Год", "Месяц", "Значения коэффициентов"]
    assert grid[1] == ["2026", "7", "0,99"]
    assert grid[5] == ["2026", "11", "1,08"]
    assert grid[6] == ["2026", "12", "1,11"]
    # Без объединённых ячеек каждая запись — row_span=col_span=1.
    assert all(c.row_span == 1 and c.col_span == 1 for c in cells)


def test_reconstruct_does_not_crash_on_flawed_cell_geometry():
    """Второй образец того же эксперимента — у него сама детекция ячеек на входе
    уже не совпадает со структурой документа (24 cell_box_list вместо верных 21
    для такой же таблицы 3x7) — это известный, отдельный от сборки дефект
    (см. docs/plan.md, «главный открытый технический вопрос»). Сборка по
    координатам не обязана это чинить — геометрия на входе уже неверна, чинить
    нечем. Тест фиксирует факт: функция не падает и не завершается тихим неверным
    результатом без возможности это заметить — лишние ячейки дают лишние строки
    в сетке, что видно по размеру результата, а не теряется молча."""
    data = _load("simple_table_1.json")
    cells = reconstruct_table(data["cell_box_list"], data["rec_boxes"], data["rec_texts"])
    grid = grid_from_cells(cells)

    assert len(cells) == len(data["cell_box_list"])
    # 24 ячейки геометрии на 3 колонки неизбежно дают больше 7 строк в сетке —
    # это сигнал вызывающей стороне, что с этой таблицей на входе что-то не так,
    # а не тихо проглоченная ошибка.
    assert len(grid) > 7


def test_reconstruct_handles_merged_cells_rowspan():
    """Синтетическая проверка объединения по вертикали — два ряда одной высоты
    (строки 0 и 1) и один ряд той же ширины, но вдвое выше (объединяет строки 0-1
    в соседнем столбце), как у колонок «Сумма»/«Источник финансирования» в реальной
    сложной таблице (см. docs/plan.md)."""
    cell_box_list = [
        (0, 0, 100, 50),
        (100, 0, 200, 50),
        (0, 50, 100, 100),
        (100, 50, 200, 100),
        (200, 0, 300, 100),  # объединяет оба ряда в третьем столбце
    ]
    rec_boxes = [
        (10, 10, 90, 40),
        (110, 10, 190, 40),
        (10, 60, 90, 90),
        (110, 60, 190, 90),
        (210, 40, 290, 60),
    ]
    rec_texts = ["a1", "b1", "a2", "b2", "merged"]

    cells = reconstruct_table(cell_box_list, rec_boxes, rec_texts)
    merged = next(c for c in cells if c.text == "merged")
    assert merged.row == 0
    assert merged.row_span == 2
    assert merged.col_span == 1


def test_normalize_header_text_real_observed_garbling():
    """Строки взяты дословно из tests/unit/fixtures/ocr_tables/complex_table_0.json —
    реально распознанный OCR-мусор, не придуманные примеры."""
    assert normalize_header_text("Cpoк") == "Срок"
    assert normalize_header_text("Сумmа") == "Сумма"
    assert normalize_header_text("N π/n$") == "N п/п"
    assert normalize_header_text("HДT") == "НДТ"
    assert normalize_header_text("HДT 6д") == "НДТ 6д"
    assert normalize_header_text("HДТ 5r") == "НДТ 5г"
    assert normalize_header_text("наименование 3B") == "наименование ЗВ"
    assert normalize_header_text("MΓ/∂M3$") == "мг/дм³"
    assert normalize_header_text("τ/Γ") == "т/г"
    assert normalize_header_text("τ/") == "т/г"
    assert normalize_header_text("XПK") == "ХПК"
    assert normalize_header_text("з") == "3"
    assert normalize_header_text("(снижение с мг/дмз,") == "(снижение с мг/дм³,"


def test_normalize_header_text_does_not_corrupt_real_words():
    """«з» — опасный ключ как подстрочная замена (входит во многие обычные слова);
    должен срабатывать только когда это весь текст ячейки целиком, не часть слова."""
    assert normalize_header_text("Взвешенные") == "Взвешенные"
    assert normalize_header_text("сооружения") == "сооружения"
    assert normalize_header_text("Применение спецтехники (земснаряд) для") == (
        "Применение спецтехники (земснаряд) для"
    )


def test_mixed_script_latin_lookalikes_become_cyrillic_inside_russian_words():
    from npa_ocr.ocr_table import normalize_header_text

    assert normalize_header_text("Cpок выполнения") == "Срок выполнения"
    assert normalize_header_text("Вeрсия") == "Версия"


def test_mixed_script_fix_leaves_pure_latin_and_pure_cyrillic_tokens_alone():
    from npa_ocr.ocr_table import normalize_header_text

    assert normalize_header_text("PDF") == "PDF"
    assert normalize_header_text("НДТ") == "НДТ"
    assert normalize_header_text("мероприятия") == "мероприятия"
