"""Тесты таблиц через DeepSeek-OCR + сетку по линиям (vlm_tables.py) — на сохранённых ответах
DeepSeek-OCR и страницах спайка npa-vlm-ocr, без GPU. Ожидаемые таблицы — итог спайка, проверенный
разметкой (docs/review_marks_*.json там же). Ячейки, которые в спайке дочитаны VLM по кропу
(этап 5 плана встраивания), сравниваются только с дочиткой — на ответах модели, записанных на
живом vLLM (<key>.reread.json).

Сетка использует настоящий Tesseract (как и в работе), поэтому тесты на страницах пропускаются,
если его нет."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from npa_ocr.vlm_reread import reread_cells
from npa_ocr.vlm_tables import cell_text, html_cells, parse_deepseek, run_vlm_tables

FIXTURES = Path(__file__).parent / "fixtures" / "vlm_tables"
PAGES = sorted(p.stem for p in FIXTURES.glob("*.png"))


def _has_tesseract() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    from npa_ocr.ocr import _resolve_tesseract_cmd

    return bool(_resolve_tesseract_cmd() or shutil.which("tesseract"))


needs_tesseract = pytest.mark.skipif(not _has_tesseract(), reason="нет Tesseract или OpenCV")


def _run(key: str, reread_fn=None):
    raw = (FIXTURES / f"{key}.txt").read_text(encoding="utf-8")
    return run_vlm_tables(FIXTURES / f"{key}.png", chat_fn=lambda _img: raw, reread_fn=reread_fn)


def _replay_ask(key: str):
    """Ответы DeepSeek-OCR на кропы ячеек, записанные на живом vLLM: sha1 пикселей кропа + промпт
    (не байтов PNG — они зависят от версии кодировщика). Какие ячейки сомнительны, зависит и от
    версии Tesseract (Windows-сборка читает «H₂SO₄» в 191 стр. 14 как «Н›$О1», и ячейка становится
    сомнительной, а 5.5 в образе — нет): на незаписанный кроп — негодный ответ, ячейка остаётся как есть."""
    answers = json.loads((FIXTURES / f"{key}.reread.json").read_text(encoding="utf-8"))

    def ask(img, prompt: str):
        digest = hashlib.sha1(f"{img.mode}{img.size}".encode() + img.tobytes()).hexdigest()
        text, reason, lp = answers.get(digest + "|" + prompt, ("", "stop", -9.0))
        return text, reason, lp

    return ask


REREAD_PAGES = sorted(p.name.removesuffix(".reread.json") for p in FIXTURES.glob("*.reread.json"))


def _grid(table) -> dict[tuple[int, int], str]:
    return {(c.row, c.col): c.text for c in table.cells}


# --- Разбор ответа DeepSeek ---


def test_parse_deepseek_skips_caption_between_label_and_table():
    """228 стр. 23: метка table, за ней table_caption, и только потом HTML — таблица одна, рамка
    в пикселях страницы, подпись в неё не попадает."""
    raw = (FIXTURES / "228_p023.txt").read_text(encoding="utf-8")
    tables = parse_deepseek(raw, 2336, 1656)
    assert len(tables) == 1
    x0, y0, x1, y1 = tables[0].bbox
    assert 0 <= x0 < x1 <= 2336 and 0 <= y0 < y1 <= 1656
    assert tables[0].html.startswith("<table>") and tables[0].html.endswith("</table>")
    assert "Сведения о местах" not in tables[0].html


def test_parse_deepseek_accepts_label_without_special_tokens():
    """Ollama вырезает служебные токены — метка приходит как «table[[…]]»; текстовые блоки — не таблицы."""
    raw = "text[[0, 0, 999, 100]]\nПРИЛОЖЕНИЕ\n\ntable[[0, 500, 999, 999]]\n<table><tr><td>А</td></tr></table>"
    tables = parse_deepseek(raw, 1000, 2000)
    assert [t.bbox for t in tables] == [[0, 1001, 1000, 2000]]


def test_parse_deepseek_ignores_table_label_without_html():
    raw = "<|ref|>table<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|>\nтекст без таблицы"
    assert parse_deepseek(raw, 100, 100) == []


def test_cell_text_converts_scripts_breaks_and_entities():
    assert cell_text("мг/нм<sup>3</sup>") == "мг/нм³"
    assert cell_text("80<sup>1</sup>") == "80¹"
    assert cell_text("C<sub>6</sub>-C<sub>10</sub>") == "C₆-C₁₀"
    assert cell_text("Азота оксид<br/>Азота диоксид") == "Азота оксид\nАзота диоксид"
    assert cell_text("&lt;*&gt;  в   сухих") == "<*> в сухих"
    # не цифры в <sup> — как есть, без надстрочных символов
    assert cell_text("x<sup>a</sup>") == "xa"


def test_html_cells_places_spans():
    cells, n_rows, n_cols = html_cells(
        '<table><tr><td rowspan="2">А</td><td colspan="2">Б</td></tr><tr><td>В</td><td>Г</td></tr></table>')
    assert (n_rows, n_cols) == (2, 3)
    assert [(c.row, c.col, c.row_span, c.col_span, c.text) for c in cells] == [
        (0, 0, 2, 1, "А"), (0, 1, 1, 2, "Б"), (1, 1, 1, 1, "В"), (1, 2, 1, 1, "Г")]


# --- Таблицы страниц: итог спайка ---


@needs_tesseract
@pytest.mark.parametrize("key", PAGES)
def test_page_matches_spike(key: str):
    expected = json.loads((FIXTURES / f"{key}.json").read_text(encoding="utf-8"))
    got = _run(key)
    assert len(got) == len(expected)
    for t, e in zip(got, expected):
        assert t.src == e["src"]
        assert (t.n_rows, t.n_cols) == (e["n_rows"], e["n_cols"])
        assert sorted((c.row, c.col, c.row_span, c.col_span) for c in t.cells) == sorted(
            (r, c, rs, cs) for r, c, rs, cs, _, _ in e["cells"])
        texts = _grid(t)
        for r, c, _, _, text, by_vlm in e["cells"]:
            if not by_vlm:
                assert texts[(r, c)] == text, (r, c)


@needs_tesseract
@pytest.mark.parametrize("key", REREAD_PAGES)
def test_page_with_reread_matches_spike(key: str):
    """С дочиткой сомнительных ячеек (этап 5) итог совпадает со спайком во всех ячейках, включая
    дочитанные; дочитанные — без жёлтой пометки."""
    ask = _replay_ask(key)
    expected = json.loads((FIXTURES / f"{key}.json").read_text(encoding="utf-8"))
    got = _run(key, reread_fn=lambda crop, doubt: reread_cells(crop, doubt, ask))
    assert len(got) == len(expected)
    for t, e in zip(got, expected):
        assert (t.n_rows, t.n_cols) == (e["n_rows"], e["n_cols"])
        cells = {(c.row, c.col): c for c in t.cells}
        for r, c, _, _, text, by_vlm in e["cells"]:
            assert cells[(r, c)].text == text, (r, c)
            if by_vlm:
                assert not any(cells[(r, c)].low_lines), (r, c)


@needs_tesseract
def test_merged_cell_over_seven_rows():
    """191 стр. 15: объединение на 7 строк, текст в нём не по центру — DeepSeek ставит его в
    середину группы, а сетка по линиям — в одну ячейку на все строки."""
    (t,) = _run("191_p015")
    assert t.src == "grid"
    assert any(c.row_span == 7 and c.text for c in t.cells)


@needs_tesseract
def test_table_without_body_lines_falls_back_to_deepseek_html():
    """1430 стр. 8: линии только в шапке — сетка не вмещает текст DeepSeek, берём её HTML."""
    tables = _run("1430_p008")
    assert [t.src for t in tables] == ["deepseek", "deepseek"]
    texts = list(_grid(tables[0]).values())
    assert "Большие - сверхкрупные очистные сооружения" in texts
    assert all(not any(c.low_lines) for t in tables for c in t.cells)


@needs_tesseract
def test_low_confidence_only_for_tesseract_or_position_cells():
    """Ячейки с текстом только от Tesseract или поставленные лишь по положению — жёлтые в DOCX
    (low_lines на каждую строку текста); текст DeepSeek на своём месте — без пометки."""
    (t,) = _run("376_p006")
    low = [c for c in t.cells if any(c.low_lines)]
    assert low
    assert all(len(c.low_lines) == len(c.text.split("\n")) and all(c.low_lines) for c in low)
    assert len(low) < len(t.cells) / 2
