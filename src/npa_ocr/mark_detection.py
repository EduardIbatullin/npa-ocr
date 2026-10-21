"""Детектор печатей/штампов/электронных подписей на странице.

Два шага (см. docs/plan.md, раздел «ИИ-эскалация»), оба обязательны — ни
эвристика одна, ни VLM на каждой странице:
1. Дёшевая эвристика без ИИ: печать/штамп/э-подпись почти всегда даёт плотное
   скопление слов с низкой уверенностью Tesseract в одной области страницы
   (в отличие от редких одиночных ошибок OCR, разбросанных по всему тексту) —
   кластеризация bbox таких слов находит кандидатов практически бесплатно,
   на страницах без штампов кандидатов просто не находится и VLM не
   вызывается вовсе.
2. Каждый кандидат подтверждается и классифицируется через VLM
   (qwen3-vl, см. ai_escalation.classify_mark) — печать? штамп? э-подпись?
   перекрывает ли текст документа, который нужно восстановить?

Решение, что делать с каждым типом — в ocr.py (вызывающая сторона), не
здесь: этот модуль только находит и классифицирует области, не решает их
судьбу (recover/discard/оставить как есть)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

# Печать/штамп физически занимают пару сантиметров — на 200 DPI это около
# 150-200px в поперечнике. Слова дальше этого расстояния друг от друга — уже
# не один и тот же штамп, а не связанные одиночные ошибки OCR в разных
# местах страницы.
_CLUSTER_GAP = 80.0
# Кластер из одного случайно испорченного слова — слишком частый шум обычного
# OCR (нечёткая печать, слипшиеся буквы), не повод тратить вызов VLM.
_MIN_CLUSTER_WORDS = 3


def _boxes_overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _expand(box: Box, gap: float) -> Box:
    return (box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap)


def cluster_candidate_regions(
    low_confidence_word_boxes: list[Box],
    *,
    gap: float = _CLUSTER_GAP,
    min_words: int = _MIN_CLUSTER_WORDS,
) -> list[Box]:
    """Группирует bbox низкоуверенных слов страницы в кандидатные области —
    связные компоненты графа «пересекаются после раздутия на gap» (union-find).
    Возвращает bbox каждого кластера размером >= min_words слов, без запаса
    (запас добавляется отдельно при вырезании кропа для VLM, см. ocr.py)."""
    n = len(low_confidence_word_boxes)
    if n == 0:
        return []
    expanded = [_expand(b, gap) for b in low_confidence_word_boxes]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(expanded[i], expanded[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    regions = []
    for members in groups.values():
        if len(members) < min_words:
            continue
        xs0 = [low_confidence_word_boxes[i][0] for i in members]
        ys0 = [low_confidence_word_boxes[i][1] for i in members]
        xs1 = [low_confidence_word_boxes[i][2] for i in members]
        ys1 = [low_confidence_word_boxes[i][3] for i in members]
        regions.append((min(xs0), min(ys0), max(xs1), max(ys1)))
    return regions


@dataclass
class Mark:
    bbox: Box
    mark_type: str  # см. ai_escalation.classify_mark — "печать"/"штамп"/"электронная_подпись"/"неизвестно"
    covers_text: bool  # перекрывает ли содержательный текст документа (не свой собственный)


def detect_marks(
    image_path: Path,
    low_confidence_word_boxes: list[Box],
    *,
    classify_fn: Callable[[Path], tuple[str, bool]],
    crop_padding: float = 20.0,
) -> list[Mark]:
    """classify_fn(crop_path) -> (mark_type, covers_text) — в реальном
    использовании это ai_escalation.classify_mark, в тестах — фейк на
    реальных данных (не загружает модель на каждый тест, см. ocr.py для
    аналогичного паттерна с другими движками)."""
    regions = cluster_candidate_regions(low_confidence_word_boxes)
    if not regions:
        return []
    page_image = Image.open(image_path)
    marks = []
    for region in regions:
        box = (
            max(0, int(region[0] - crop_padding)),
            max(0, int(region[1] - crop_padding)),
            min(page_image.width, int(region[2] + crop_padding)),
            min(page_image.height, int(region[3] + crop_padding)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        crop = page_image.crop(box)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            crop.save(tmp.name)
            tmp_path = Path(tmp.name)
        try:
            mark_type, covers_text = classify_fn(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        marks.append(Mark(bbox=region, mark_type=mark_type, covers_text=covers_text))
    return marks
