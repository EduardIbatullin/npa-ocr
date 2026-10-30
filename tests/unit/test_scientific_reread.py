"""Химические ячейки: состав подтверждается изображением, не справочником."""
import pytest
from pathlib import Path
from PIL import Image

from npa_ocr.vlm_reread import _verify_scientific, reread_cells
from npa_ocr.scientific_text import chemical_indices


def test_smiles_response_is_rejected_and_markdown_formula_is_read_twice():
    answers = iter([('<smiles>Cl.N.[Cl]</smiles>', 'stop', -.1),
                    (r'NH\\(_4\\)ClO\\(_4\\)', 'stop', -.1),
                    (r'NH\\(_4\\)ClO\\(_4\\)', 'stop', -.1)])
    d = descriptor(old='4.')
    assert _verify_scientific(lambda *a: next(answers), Image.new('RGB',(100,40)), d) == 'NH₄ClO₄'


@pytest.mark.parametrize("file,expected", [
    ("formula-full-3-0-1.png", "C₂H₈N₂"),
    ("formula-full-4-0-5.png", "C₃H₈N₄O₄"),
    ("formula-full-7-0-2.png", "C₂H₇N"),
    ("formula-full-7-1-1.png", "NC₂H₆N₂O"),
    ("formula-full-7-2-3.png", "C₂H₆N₂O"),
])
def test_glyph_geometry_on_real_formula_crops(file, expected):
    import pytesseract
    from npa_ocr.ocr import _resolve_tesseract_cmd
    from npa_ocr.formula_geometry import read_formula
    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        pytest.skip("Tesseract not installed")
    image = Image.open(Path(__file__).parent / "fixtures/chemical_formulas" / file)
    assert read_formula(image, 4)[0] == expected


def descriptor(kind="formula", old="CrHsN₂"):
    return dict(r=0, c=0, box=[0, 0, 100, 40], expected=kind, ds=old, tess=old)


@pytest.mark.parametrize("old,new,expected", [
    ("CrHsN₂", "C2H8N2", "C₂H₈N₂"),
    ("C₂H₆N₂₀", "C2H6N2O", "C₂H₆N₂O"),
    ("NC₂H₆N₂O", "NC2H6N2O", "NC₂H₆N₂O"),
])
def test_formula_composition_can_change_only_after_two_matching_crop_reads(old, new, expected):
    calls = []
    def ask(image, prompt):
        calls.append(image.size)
        return new, "stop", -.1
    assert _verify_scientific(ask, Image.new("RGB", (100, 40)), descriptor(old=old)) == expected
    assert len(calls) == 2 and calls[0] != calls[1]


def test_disagreeing_scientific_reads_do_not_fall_back_to_single_read():
    answers = iter([("C2H8N2", "stop", -.1), ("C2H6N2", "stop", -.1)])
    got = reread_cells(Image.new("RGB", (100, 40)), [descriptor()], lambda *a: next(answers))
    assert got == {}


def test_numeric_zero_is_not_replaced_by_oxygen_without_an_image_reading():
    assert chemical_indices("C₂H₆N₂₀") == "C₂H₆N₂₀"


def test_chemical_name_recovers_latin_locants_from_agreeing_crops():
    text = "N-метил-N-нитрозометанамин +"
    d = descriptor("chemical_name", "М-метил-М№-нитрозометанамин +")
    assert _verify_scientific(lambda *a: (text, "stop", -.1), Image.new("RGB", (100, 40)), d) == text


def test_chemical_name_cannot_change_locant_numbers():
    d = descriptor("chemical_name", "2,4,6-тринитротолуол")
    assert _verify_scientific(lambda *a: ("2,4,8-тринитротолуол", "stop", -.1), Image.new("RGB", (100, 40)), d) is None


def test_docx_uses_a4_and_one_centimetre_margins(tmp_path):
    from docx import Document
    from npa_ocr.ocr import build_docx
    out = tmp_path / "out.docx"
    build_docx([], out)
    section = Document(out).sections[0]
    assert section.page_width.cm == pytest.approx(21, abs=.01)
    assert section.page_height.cm == pytest.approx(29.7, abs=.01)
    for margin in [section.top_margin, section.bottom_margin, section.left_margin, section.right_margin]:
        assert margin.cm == pytest.approx(1, abs=.01)
