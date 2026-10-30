"""Сетка сложных таблиц на реальных сканах, без запросов к моделям."""

from pathlib import Path

import pytest
from PIL import Image

from npa_ocr.vlm_tables import _detect_cells, ruled_regions
from npa_ocr.scientific_text import chemical_indices, doubtful_roman_code
from npa_ocr.vlm_reread import accept

FIXTURES = Path(__file__).parent / "fixtures" / "sanpin_2025_2"


def test_tiny_character_in_tall_cell_is_not_treated_as_blank(monkeypatch):
    import pytesseract
    from PIL import ImageDraw
    from npa_ocr.vlm_tables import _Cell, _tesseract_into_cells

    image = Image.new("RGB", (400, 1000), "white")
    ImageDraw.Draw(image).rectangle((50, 30, 52, 42), fill="black")
    cells = [_Cell(10, 10, 190, 990), _Cell(210, 10, 390, 990)]
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: dict(text=[], conf=[]))
    _tesseract_into_cells(image, cells)
    assert cells[0].ink
    assert not cells[1].ink


@pytest.mark.parametrize("page,columns", [(12, 9), (13, 9), (14, 9), (17, 7), (18, 5)])
def test_ruled_table_has_no_overlaps_or_missing_positions(page, columns):
    image = Image.open(FIXTURES / f"page_{page:04}.png").convert("RGB")
    regions = ruled_regions(image)
    assert regions
    cells, info = _detect_cells(image.crop(regions[-1]))
    assert info["ok"], info
    assert info["n_cols"] == columns
    assert info["overlap"] == info["holes"] == 0
    assert len(cells) > 20
    if page in (12, 13, 14):
        assert any(c.cs >= 8 for c in cells)


@pytest.mark.parametrize("source,expected", [
    ("NH4ClO4", "NH₄ClO₄"), ("C11H26NO2 PS", "C₁₁H₂₆NO₂PS"),
    ("NH ₄ ClO ₄", "NH₄ClO₄"), ("C ₂ H ₆ N ₂ O", "C₂H₆N₂O"),
    ("S(CH2CH2Cl)2", "S(CH₂CH₂Cl)₂"), ("NC2H6N2O", "NC₂H₆N₂O"),
    ("CAS 7440-32-2 ГОСТ Р 59701.1-2022", "CAS 7440-32-2 ГОСТ Р 59701.1-2022"),
])
def test_chemical_notation_preserves_original_composition(source, expected):
    assert chemical_indices(source) == expected


@pytest.mark.parametrize("code", ["IIа (175 – 232)", "VIIIб", "IVг", "III"])
def test_roman_code_is_not_transliterated(code):
    assert accept(code, {"tess": "Шв"}, lp=0) == code


@pytest.mark.parametrize("code", ["1а (175 – 232)", "УШв", "Шб", "1II"])
def test_damaged_category_is_not_guessed(code):
    assert doubtful_roman_code(code)
    assert accept(code, {"tess": code, "kind": "code"}, lp=0) is None
