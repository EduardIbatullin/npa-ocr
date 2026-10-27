"""Тесты дочитки сомнительных ячеек (vlm_reread.py). Правила приёма ответа — на строках; дочитка
на страницах — на ответах DeepSeek-OCR, записанных на живом vLLM (fixtures/vlm_tables/<key>.reread.json:
sha1 кропа + промпт -> ответ, причина остановки, средний logprob), без GPU."""

from __future__ import annotations

from PIL import Image, ImageDraw

from npa_ocr.vlm_reread import accept, clean, is_vertical, reread_cells, strip_left


def test_clean_formula_and_markup():
    assert clean(r"\[\le 0,09\]") == "≤ 0,09"
    assert clean(r"мг/нм\(^3\)") == "мг/нм³"
    assert clean("мг/нм<sup>3</sup>") == "мг/нм³"
    assert clean("C_6-C_{10}") == "C₆-C₁₀"
    assert clean("<table><tr><td>≤ 50</td></tr></table>") == "≤ 50"
    assert clean(r"\text{No}") == "№"


def test_accept_rejects_formula_latin_and_greek():
    d = {"tess": "мг/нм3"}
    assert accept("[HM+3].[Mg]", d) is None
    assert accept("mg/nm³", d) is None
    assert accept("μг/нм³", d) is None
    assert accept("мг/нм³", d) == "мг/нм³"


def test_accept_turns_latin_lookalikes_into_cyrillic():
    """«C 01.01.2001 no 31.12.2025» — кириллица, прочитанная латиницей; «C6-C10» не трогаем."""
    d = {"tess": "С 01.01.2001 по 31.12.2025"}
    assert accept("C 01.01.2001 no 31.12.2025", d) == "С 01.01.2001 по 31.12.2025"
    assert accept("предельные C6-C10", {"tess": "предельные Сб-С10"}) == "предельные C6-C10"


def test_accept_without_tesseract_text_needs_number_or_confidence():
    """В ячейке без текста Tesseract сверить не с чем: число/«№»/прочерк — да, остальное — только
    уверенный ответ (227 стр. 9: в повёрнутой ячейке модель сочиняет русский текст)."""
    d = {"tess": ""}
    assert accept("≤ 50", d, lp=-0.9) == "≤ 50"
    assert accept("№", d, lp=-0.9) == "№"
    assert accept("Пыль неорганическая", d, lp=-0.5) is None
    assert accept("Пыль неорганическая", d, lp=-0.05) == "Пыль неорганическая"


def test_strip_left_removes_neighbour_text():
    assert strip_left("Серы диоксид мг/нм³", "Серы диоксид") == "мг/нм³"
    assert strip_left("мг/нм³", "Серы диоксид") == ""


def _lines_image(vertical: bool) -> Image.Image:
    img = Image.new("L", (400, 400), 255)
    draw = ImageDraw.Draw(img)
    for line in range(5):
        for ch in range(20):
            x, y = 20 + ch * 18, 40 + line * 60
            if vertical:
                x, y = y, x
            draw.rectangle((x, y, x + 10, y + 14) if not vertical else (x, y, x + 14, y + 10), fill=0)
    return img


def test_is_vertical():
    assert is_vertical(_lines_image(vertical=True))
    assert not is_vertical(_lines_image(vertical=False))


def test_attempts_go_in_order_until_accepted():
    """Негодный ответ на кроп ячейки — следующая попытка (пара с соседом слева); ячейка, принятая
    первой попыткой, дальше не спрашивается."""
    crop = Image.new("RGB", (600, 200), "white")
    # ширина кропа ячейки — ширина рамки − 2·PAD + 2·BORDER: 240 у первой, 190 у второй, 480 у пары
    doubt = [
        {"r": 0, "c": 1, "box": [200, 0, 400, 100], "rot": 0, "tess": "мг/нм3", "kind": "tess", "ds": "",
         "left": {"box": [0, 0, 200, 100], "text": "Серы диоксид"}},
        {"r": 1, "c": 1, "box": [200, 100, 350, 200], "rot": 0, "tess": "5 50", "kind": "tess", "ds": "",
         "left": None},
    ]
    answers = {240: "[HM+3].[Mg]", 190: "≤ 50", 480: "Серы диоксид мг/нм³"}
    calls: list[tuple[int, str]] = []

    def ask(img: Image.Image, prompt: str):
        calls.append((img.width, prompt))
        return answers[img.width], "stop", -0.05

    got = reread_cells(crop, doubt, ask)
    assert got == {(0, 1): "мг/нм³", (1, 1): "≤ 50"}
    assert sorted(calls) == [(190, "Free OCR."), (240, "Free OCR."), (480, "Free OCR.")]
