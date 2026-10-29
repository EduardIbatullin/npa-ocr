"""Тесты detect_table_regions (выбор меток/порога) — модель TATR подменяется
фейком, но сама функция использует torch.no_grad() безусловно, поэтому эти
тесты (в отличие от test_table_structure.py) требуют torch и пропускаются
там, где его нет (см. docs/history/plan.md «Приказ №191», 2026-10-23)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from npa_ocr import table_structure


class _FakeModel:
    def __init__(self, calls: dict) -> None:
        self._calls = calls

    def __call__(self, **inputs):
        self._calls["forward"] += 1
        return "raw_out"


class _FakeProc:
    """candidates — список (score, label, box), как если бы это была модель
    detect_table_regions с порогом 0 (все кандидаты сразу, без фильтра)."""

    def __init__(self, candidates: list[tuple[float, int, tuple[float, float, float, float]]]) -> None:
        self._candidates = candidates

    def __call__(self, images, return_tensors):
        return {}

    def post_process_object_detection(self, out, threshold, target_sizes):
        kept = [(s, l, b) for s, l, b in self._candidates if s >= threshold]
        return [
            {
                "scores": torch.tensor([s for s, _, _ in kept]),
                "labels": torch.tensor([l for _, l, _ in kept]),
                "boxes": torch.tensor([list(b) for _, _, b in kept]) if kept else torch.empty((0, 4)),
            }
        ]


class _FakeImg:
    size = (100, 200)


def _patch(monkeypatch, candidates):
    calls = {"forward": 0}
    monkeypatch.setattr(
        table_structure, "load_tatr_detector", lambda: (_FakeModel(calls), _FakeProc(candidates))
    )
    return calls


def test_detect_table_regions_accepts_rotated_label(monkeypatch):
    """label=1 («table rotated») принимается наравне с label=0 — см. комментарий
    у TATR_TABLE_ROTATED_LABEL: на реальном документе (Приказ №191) детектор
    систематически путает обычные продолжения многостраничных таблиц с
    повёрнутыми, с высокой уверенностью."""
    calls = _patch(
        monkeypatch,
        [(0.9, table_structure.TATR_TABLE_ROTATED_LABEL, (1.0, 2.0, 3.0, 4.0))],
    )
    regions = table_structure.detect_table_regions(_FakeImg())
    assert regions == [(1.0, 2.0, 3.0, 4.0)]
    assert calls["forward"] == 1


def test_detect_table_regions_falls_back_to_lower_threshold_without_second_inference(monkeypatch):
    """Ничего не нашлось на обычном пороге (0.7) — повторная постобработка на
    TATR_REGION_FALLBACK_THRESHOLD (0.3) того же forward pass, не повторный
    инференс модели."""
    calls = _patch(monkeypatch, [(0.5, table_structure.TATR_TABLE_LABEL, (1.0, 2.0, 3.0, 4.0))])
    regions = table_structure.detect_table_regions(_FakeImg())
    assert regions == [(1.0, 2.0, 3.0, 4.0)]
    assert calls["forward"] == 1


def test_detect_table_regions_prefers_normal_threshold_result(monkeypatch):
    """Если на обычном пороге уже что-то нашлось, резервный порог не
    используется (ниже уверенность у постороннего кандидата не должна
    попадать в результат)."""
    calls = _patch(
        monkeypatch,
        [
            (0.8, table_structure.TATR_TABLE_LABEL, (1.0, 2.0, 3.0, 4.0)),
            (0.4, table_structure.TATR_TABLE_LABEL, (9.0, 9.0, 9.0, 9.0)),
        ],
    )
    regions = table_structure.detect_table_regions(_FakeImg())
    assert regions == [(1.0, 2.0, 3.0, 4.0)]
    assert calls["forward"] == 1


def test_detect_table_regions_empty_when_nothing_above_fallback_threshold(monkeypatch):
    calls = _patch(monkeypatch, [(0.1, table_structure.TATR_TABLE_LABEL, (1.0, 2.0, 3.0, 4.0))])
    regions = table_structure.detect_table_regions(_FakeImg())
    assert regions == []
    assert calls["forward"] == 1
