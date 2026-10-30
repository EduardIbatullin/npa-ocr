"""Числа, римские категории и реквизиты сверяются по изображению, без подсказок норм."""

from PIL import Image

from npa_ocr.cell_reading import _code, _numeric, _mixed_labels, _consensus_words
from npa_ocr.scientific_text import section_heading
from npa_ocr.vlm_marks import Word
from npa_ocr.vlm_text import reread_lines
from npa_ocr.requisites import restore_header_number


def test_roman_prefix_and_cyrillic_suffix_are_read_separately():
    assert _code("Ila (175 - 232)", "Па (175 - 232)", "1а (175 – 232)") == "IIа (175 – 232)"
    assert _code("116 (233 - 290)", "1б (233 - 290)", "1б (233 – 290)") == "IIб (233 – 290)"
    assert _code("IITs", "Шв", "IIв") == "IIIв"
    assert _code("IVB", "ГУв", "ТУв") == "IVв"
    assert _code("16 (140 - 174)", "16 (140 - 174)", "1б (140 – 174)") == "Iб (140 – 174)"
    assert _code("VI", "VI", "VII") is None
    assert _code("XIII", "XIII", "XI", support="XIII") == "XIII"
    assert _code("VI", "VI", "VII", support="VII") is None
    assert _code("XII", "ХШ", "XIII") == "XIII"
    assert _code("unreadable", "неразборчиво", "неразборчиво") is None


def test_short_values_keep_digits_and_remove_separate_border_trace():
    assert _numeric("20 |", "number") == "20"
    assert _numeric("20 .", "number") == "20"
    assert _numeric("—", "number") == "—"
    assert _numeric("1 10.", "index") is None
    assert _numeric("питатель)", "number") is None
    assert _numeric("9.", "index") == _numeric("9", "index") == "9"


def test_index_split_uses_sparse_reading_when_block_reading_misses_digit(monkeypatch):
    import pytesseract
    from npa_ocr.cell_reading import split_index_cells
    from npa_ocr.vlm_tables import _Cell

    cell = _Cell(0, 0, 80, 200, r=0, rs=3)
    cells = [cell]
    def data(*args, **kwargs):
        return dict(text=["12.", "1B."], top=[30, 250]) if kwargs["config"] == "--psm 6" else dict(text=["12", "13"], top=[30, 250])
    monkeypatch.setattr(pytesseract, "image_to_data", data)
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: "12.\n13.")
    assert split_index_cells(Image.new("RGB", (200, 400), "white"), cells, [0, 100, 150, 200]) == 1
    assert [(c.r, c.rs, c.tess) for c in cells] == [(0, 1, "12."), (1, 2, "13.")]


def test_local_formula_reading_preserves_existing_composition(monkeypatch):
    import pytesseract
    from npa_ocr.cell_reading import verify_cells
    from npa_ocr.vlm_tables import _Cell

    cell = _Cell(0, 0, 100, 30, text="C₂H₆N₂O", expected="formula")
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: "C2HeN20")
    verify_cells(Image.new("RGB", (200, 200), "white"), [cell])
    assert cell.text == "C₂H₆N₂O"


def test_section_numbers_are_normalised_without_guessing_from_order():
    assert section_heading("УП. Участок сгущения") == "VII. Участок сгущения"
    assert section_heading("Х!. Электроцех") == "XI. Электроцех"
    assert section_heading("Ш.Участок дробления") == "III. Участок дробления"
    assert section_heading("3. Пункт постановления") is None


def test_spelling_consensus_cannot_change_numeric_values():
    assert _consensus_words("тетратен", "тетразен", "тетразен") == "тетразен"
    assert _consensus_words("гидромоторной установки", "гидромониторной установки", "гидромониторной установки") == "гидромониторной установки"
    assert _consensus_words("Норма 20", "Норма 2", "Норма 2") is None
    assert _consensus_words("тетратен", "тетразен", "другое") is None


def test_latin_labels_are_matched_by_their_definition():
    previous = "М - коэффициент частотной коррекции для локальной вибрации; ГОСТ Р 597/01.1-2022"
    local = "Wh — коэффициент частотной коррекции для локальной вибрации; ГОСТ Р 59701.1-2022"
    result = _mixed_labels(previous, local)
    assert result.startswith("Wh —")
    assert "59701.1-2022" in result


def test_handwritten_date_uses_only_corresponding_words_not_city():
    lines = [[Word("ТУ.", 20, (105, 250, 160, 275)), Word("032025", 30, (165, 250, 300, 275)),
              Word("Москва", 95, (450, 250, 540, 275))]]
    raw = "text[[100, 245, 310, 280]]\n17.03.2025"
    out, _ = reread_lines(lines, raw, 1000, 1000)
    assert [w.text for w in out[0]] == ["17.03.2025", "Москва"]
    assert out[0][0].restored and not out[0][1].restored


def test_header_number_requires_two_readings_to_agree(monkeypatch):
    import pytesseract

    raw = "title[[300, 150, 700, 200]]\nПОСТАНОВЛЕНИЕ\ntext[[100, 245, 310, 280]]\n17.03.2025"
    lines = [[Word("Москва", 95, (450, 250, 540, 275))]]
    image = Image.new("RGB", (1000, 1000), "white")
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: "№ 2")
    out = restore_header_number(lines, raw, image, lambda *a: ("№ 2", "stop", 0))
    assert any([w.text for w in line] == ["№", "2"] for line in out)
    assert restore_header_number(lines, raw, image, lambda *a: ("№ 3", "stop", 0)) == lines


def test_title_recovery_requires_matching_numbers(monkeypatch):
    import pytesseract
    from npa_ocr.text_recovery import recover_titles

    title = "Таблица 3.15 Полное название таблицы и продолжение заголовка"
    raw = "title[[100, 100, 900, 200]]\n" + title
    lines = [[Word("Таблица", 90, (110, 110, 190, 130))]]
    def data(text):
        words = text.split()
        return dict(text=words, left=[0]*len(words), top=[0]*len(words),
                    width=[10]*len(words), height=[10]*len(words), conf=[90]*len(words),
                    block_num=[1]*len(words), par_num=[1]*len(words), line_num=[1]*len(words))
    image = Image.new("RGB", (1000, 1000), "white")
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: data(title))
    result = recover_titles(lines, raw, image)
    assert " ".join(w.text for line in result for w in line) == title
    assert all(w.restored for line in result for w in line)
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: data(title.replace("3.15", "3.16")))
    assert recover_titles(lines, raw, image) == lines
