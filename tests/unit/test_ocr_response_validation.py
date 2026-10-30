"""Регрессии на служебных ответах и числах из проверенного постановления."""

import pytest

from npa_ocr.ocr_table import TableCell
from npa_ocr.vlm_reread import accept, clean
from npa_ocr.vlm_tables import _Cell, _doubt_kind, apply_reread, cell_text, html_cells
from npa_ocr.response_validation import UNREADABLE


@pytest.mark.parametrize('value',['≤ 50','≥ 1','< 0,1','> 2','<= 50'])
def test_numeric_schema_accepts_comparison_operators(value):
    from npa_ocr.response_validation import matches_cell_type
    assert matches_cell_type(value,'number')
    assert accept(value,{'tess':value,'expected':'number'},lp=0)==value


@pytest.mark.parametrize("text", [
    '{ "Line": { "line_type": ["-"], "line_endpoint": ["I: (8, 0)"] } }',
    "I'm sorry, but the image provided is a graphic design and not a data chart.",
    "It does not contain any data points or numerical information that can be extracted.",
])
def test_service_response_is_rejected_in_every_cell_path(text):
    assert clean(text) == ""
    assert accept(text, {"tess": "1"}, lp=0) is None
    assert cell_text(text) == UNREADABLE
    cells, _, _ = html_cells(f'<table><tr><td>{text}</td></tr></table>')
    assert cells[0].text == UNREADABLE
    assert any(cells[0].low_lines)
    updated = apply_reread([TableCell(0, 0, 1, 1, "1")], {(0, 0): text})[0]
    assert updated.text == UNREADABLE and any(updated.low_lines)


@pytest.mark.parametrize("text", ["-", "—", "–", "-0,5", "10⁻³"])
def test_clean_preserves_dashes_and_signed_values(text):
    assert clean(text) == text


@pytest.mark.parametrize("tess,ds", [("20", "2"), ("5", "10"), ("—", "1"), ("1600–1750", "600–1750")])
def test_short_numeric_disagreements_are_doubtful(tess, ds):
    assert _doubt_kind(_Cell(0, 0, 50, 50, tess=tess, text=ds, src="ds", ink=True)) == "mismatch"


def test_reread_keeps_visible_uncertainty_and_numeric_cell_rejects_neighbour():
    updated = apply_reread([TableCell(0, 0, 1, 1, "2", (True,))], {(0, 0): "20"})[0]
    assert updated.text == "20" and any(updated.low_lines)
    assert accept("питатель)", {"tess": "—"}, lp=0) is None
    assert accept("Chemical Abstracts Service Registry Number", {"tess": ""}, lp=0)
