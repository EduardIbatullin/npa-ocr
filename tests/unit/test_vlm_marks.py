"""Тесты чистки меток (vlm_marks.py) — на страницах спайка npa-vlm-ocr (marks/README.md): слова
Tesseract (5.5 в образе npa-ocr), ответ DeepSeek-OCR и ответ qwen3-vl записаны на nv-02
(fixtures/vlm_marks: <key>.tess.json, <key>.ds.txt, <key>.qwen.txt), без моделей и без Tesseract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa_ocr.vlm_marks import Mark, Word, clean_lines, parse_marks, should_locate

FIXTURES = Path(__file__).parent / "fixtures" / "vlm_marks"
MINPRIRODY = "Министерство природных ресурсов и экологии Российской Федерации"
GOVERNMENT = "Правительство Российской Федерации"
AUTHORITY = {"1430": GOVERNMENT, "428r": GOVERNMENT, "710": GOVERNMENT}
MARK_PAGES = sorted(p.name.removesuffix(".tess.json") for p in FIXTURES.glob("*_p001.tess.json"))
ORDERS = [k for k in MARK_PAGES if k.split("_")[0] not in AUTHORITY]
# текст меток на этих страницах — по разметке спайка (set/out/*.review.png)
MARK_TEXT = ("ЗАРЕГИСТ", "ЗАРЕГНСТ", "РЕГИСТРАЦИОН", "ЮСТИЦИИ", "ПОДПИСАН", "ДЕЙСТВИТЕЛЕН")


def _page(key: str) -> tuple[list[list[Word]], list[Mark], str, int, int]:
    d = json.loads((FIXTURES / f"{key}.tess.json").read_text(encoding="utf-8"))
    width, height = d["width"][0], d["height"][0]  # первая запись image_to_data — вся страница
    lines: dict[tuple[int, int, int], list[Word]] = {}
    for i, t in enumerate(d["text"]):
        if t.strip():
            box = (d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(
                Word(t, d["conf"][i], box))
    marks = parse_marks((FIXTURES / f"{key}.qwen.txt").read_text(encoding="utf-8"), width, height)
    return list(lines.values()), marks, (FIXTURES / f"{key}.ds.txt").read_text(encoding="utf-8"), width, height


def _clean(key: str) -> tuple[list[str], list[str]]:
    lines, marks, ds, width, height = _page(key)
    out = clean_lines(lines, marks, ds, width, height, AUTHORITY.get(key.split("_")[0], MINPRIRODY))
    return [" ".join(w.text for w in line) for line in lines], [" ".join(w.text for w in line) for line in out]


# --- Ответ qwen3-vl и когда его звать ---


def test_parse_marks_scales_to_page_pixels_and_skips_unknown():
    raw = ('```json\n[{"label": "stamp", "bbox_2d": [500, 100, 1000, 200]},'
           ' {"label": "logo", "bbox_2d": [0, 0, 10, 10]}, {"label": "seal", "bbox_2d": [1, 2]}]\n```')
    assert parse_marks(raw, 2000, 3000) == [Mark("stamp", (1000.0, 300.0, 2000.0, 600.0))]
    assert parse_marks("не JSON", 100, 100) == []


def test_should_locate_first_last_clusters_and_images_off_coat_of_arms():
    coat = "<|ref|>image<|/ref|><|det|>[[430, 20, 560, 120]]<|/det|>\n"
    seal = "<|ref|>image<|/ref|><|det|>[[340, 607, 515, 733]]<|/det|>\n"
    assert should_locate(1, 10, None, 1650, 2340)
    assert should_locate(10, 10, None, 1650, 2340)
    assert should_locate(5, 10, None, 1650, 2340, low_clusters=1)
    assert not should_locate(5, 10, coat, 1650, 2340)  # герб наверху по центру — не метка
    assert should_locate(5, 10, coat + seal, 1650, 2340)


# --- Страницы спайка ---


def test_marks_found_on_every_first_page():
    for key in MARK_PAGES:
        assert _page(key)[1], key


@pytest.mark.parametrize("key", MARK_PAGES)
def test_mark_text_is_dropped_and_untouched_lines_kept(key: str):
    """Строки у меток — без текста меток (в теле приказа «зарегистрирован Министерством юстиции …
    регистрационный №» — настоящий текст, он вдали от меток); строки вдали от меток — как были."""
    _, after = _clean(key)
    lines, marks, *_ = _page(key)
    untouched = [" ".join(w.text for w in line) for line in lines
                 if not any(w.box[0] < m.box[2] and m.box[0] < w.box[2] and w.box[1] < m.box[3] and m.box[1] < w.box[3]
                            for w in line for m in marks)]
    assert all(u in after for u in untouched)
    near = [a for a in after if a not in untouched]
    assert near
    assert not [a for a in near if any(t in a.upper() for t in MARK_TEXT)]


@pytest.mark.parametrize("key", ORDERS)
def test_ministry_header_under_stamp_edge_is_completed(key: str):
    """Край штампа Минюста ложится на «ЭКОЛОГИИ» в шапке: «И Э», «й эколог ИИ», «И'ЭКОЛОГИИ»."""
    _, after = _clean(key)
    assert "МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И ЭКОЛОГИИ" in after


@pytest.mark.parametrize("key, expected", [
    ("215_p001", ["14.04.2026 ‚ 215", "Министр А.А. Козлов"]),  # росчерк перечёркивает фамилию
    ("235_p001", ["Министр А.А. Козлов"]),  # «Министр» в рамке подписи, DeepSeek его не выдал
    ("377_p001", ["27.05.2022 № 377"]),  # «377 | регистрационный № 69836» — после черты рамки штамп
    ("565_p001", ["3. Настоящий приказ вступает в силу с 1 марта 2024 г. и действует"]),  # в рамке подписи
    ("428r_p001", ["Российской Федерации Д.Медведев"]),  # «Федерай» под печатью — по DeepSeek
    ("1430_p001", ["Председатель Правительства"]),  # «Прави» под печатью
    ("710_p001", ["Председатель Правительства", "Российской Федерации М.Мишустин"]),  # штамп ЭП
])
def test_document_text_near_marks(key: str, expected: list[str]):
    _, after = _clean(key)
    for line in expected:
        assert line in after, line


def test_page_without_marks_is_unchanged():
    lines, marks, ds, width, height = _page("228_p002")
    assert marks == []
    assert clean_lines(lines, marks, ds, width, height, MINPRIRODY) == lines


def test_replaced_and_completed_words_are_marked_restored():
    lines, marks, ds, width, height = _page("227_p001")
    out = clean_lines(lines, marks, ds, width, height, MINPRIRODY)
    header = next(line for line in out if line[0].text == "МИНИСТЕРСТВО")
    assert [w.text for w in header if w.restored] == ["И", "ЭКОЛОГИИ"]
    assert not any(w.restored for w in header[:3])


def test_header_completion_needs_authority_name():
    lines, marks, ds, width, height = _page("227_p001")
    out = clean_lines(lines, marks, ds, width, height, None)
    assert "МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ" in [" ".join(w.text for w in line) for line in out]
