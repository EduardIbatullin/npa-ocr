"""Тесты детектора печатей/штампов/э-подписей (mark_detection.py, см.
docs/plan.md «ИИ-эскалация»). classify_fn — фейк, без реальных вызовов VLM."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from npa_ocr.mark_detection import Mark, cluster_candidate_regions, detect_marks


def test_cluster_candidate_regions_groups_nearby_words_into_one_region():
    # Имитация слов, задетых одним штампом — плотно расположены друг к другу.
    boxes = [
        (100.0, 100.0, 150.0, 120.0),
        (155.0, 100.0, 200.0, 120.0),
        (100.0, 125.0, 180.0, 145.0),
    ]
    regions = cluster_candidate_regions(boxes, gap=10.0, min_words=2)
    assert len(regions) == 1
    x1, y1, x2, y2 = regions[0]
    assert (x1, y1, x2, y2) == (100.0, 100.0, 200.0, 145.0)


def test_cluster_candidate_regions_keeps_far_apart_words_separate():
    boxes = [
        (0.0, 0.0, 20.0, 20.0),
        (10.0, 0.0, 30.0, 20.0),  # рядом с первым — один кластер
        (1000.0, 1000.0, 1020.0, 1020.0),  # далеко — отдельная случайная ошибка OCR
        (1030.0, 1000.0, 1050.0, 1020.0),
    ]
    regions = cluster_candidate_regions(boxes, gap=10.0, min_words=2)
    assert len(regions) == 2


def test_cluster_candidate_regions_drops_clusters_below_min_words():
    # Одно случайно испорченное слово посреди страницы — не кандидат, слишком
    # частый шум обычного OCR, не повод тратить вызов VLM (см. docs/plan.md).
    boxes = [(500.0, 500.0, 540.0, 520.0)]
    assert cluster_candidate_regions(boxes, min_words=2) == []


def test_cluster_candidate_regions_empty_input():
    assert cluster_candidate_regions([]) == []


def test_detect_marks_returns_empty_when_no_candidates():
    calls = []
    marks = detect_marks(
        Path("unused.png"),
        [],
        classify_fn=lambda p: calls.append(p) or ("печать", True),
    )
    assert marks == []
    assert calls == []  # VLM не вызывается, когда эвристика не нашла кандидатов


def test_detect_marks_crops_and_classifies_candidate(tmp_path: Path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 400), "white").save(image_path)

    seen_crops = []

    def fake_classify(crop_path: Path) -> tuple[str, bool]:
        with Image.open(crop_path) as img:
            seen_crops.append(img.size)
        return "штамп_регистрации", True

    boxes = [
        (100.0, 100.0, 150.0, 120.0),
        (155.0, 100.0, 200.0, 120.0),
        (100.0, 125.0, 180.0, 145.0),
    ]
    marks = detect_marks(image_path, boxes, classify_fn=fake_classify, crop_padding=10.0)
    assert len(marks) == 1
    assert marks[0] == Mark(bbox=(100.0, 100.0, 200.0, 145.0), mark_type="штамп_регистрации", covers_text=True)
    assert len(seen_crops) == 1
    # кроп обрезан по границам страницы и с запасом crop_padding, не просто
    # bbox региона один в один
    w, h = seen_crops[0]
    assert w == 120  # (200+10) - (100-10)
    assert h == 65  # (145+10) - (100-10)
