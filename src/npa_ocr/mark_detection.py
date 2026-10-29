"""Поиск печатей, штампов и электронных подписей на странице.

Поиск идёт в два шага:

1. Эвристика без модели. Метка почти всегда даёт плотное скопление слов с
   низкой уверенностью Tesseract в одной области, а обычные ошибки OCR
   разбросаны по тексту. Кластеризация рамок таких слов находит кандидатов
   почти бесплатно; на страницах без меток кандидатов нет и VLM не вызывается.
2. Каждый кандидат классифицирует VLM (:func:`npa_ocr.ai_escalation.classify_mark`):
   тип метки и перекрывает ли она текст документа.

Что делать с меткой каждого типа, решает вызывающий код (:mod:`npa_ocr.ocr`);
модуль только находит и классифицирует области.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

# Печать или штамп занимают пару сантиметров — около 150–200 px при 200 DPI.
# Слова дальше друг от друга относятся к разным местам страницы.
_CLUSTER_GAP = 80.0
# Одно-два испорченных слова — обычный шум OCR, не повод вызывать VLM.
_MIN_CLUSTER_WORDS = 3


def _boxes_overlap(a: Box, b: Box) -> bool:
    """Проверяет, пересекаются ли рамки."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _expand(box: Box, gap: float) -> Box:
    """Расширяет рамку на ``gap`` во все стороны."""
    return (box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap)


def cluster_candidate_regions(
    low_confidence_word_boxes: list[Box],
    *,
    gap: float = _CLUSTER_GAP,
    min_words: int = _MIN_CLUSTER_WORDS,
) -> list[Box]:
    """Группирует рамки неуверенных слов в области-кандидаты на метки.

    Кластер — связная компонента графа, где рамки соединены, если
    пересекаются после расширения на ``gap`` (система непересекающихся
    множеств).

    Args:
        low_confidence_word_boxes: Рамки слов с низкой уверенностью.
        gap: Расширение рамок, px.
        min_words: Наименьшее число слов в кластере.

    Returns:
        Рамки кластеров без запаса; запас добавляется при вырезании фрагмента.
    """
    n = len(low_confidence_word_boxes)
    if n == 0:
        return []
    expanded = [_expand(b, gap) for b in low_confidence_word_boxes]
    parent = list(range(n))

    def find(i: int) -> int:
        """Корень множества элемента со сжатием пути."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        """Объединяет множества элементов."""
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
    """Найденная метка: рамка, тип и перекрывает ли она текст документа."""

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
    """Находит и классифицирует метки на странице.

    Ошибка классификации (сеть, таймаут Ollama) не пробрасывается: эскалация —
    необязательное улучшение, и её сбой не должен ронять распознавание всего
    документа. Такой кандидат пропускается, как без эскалации.

    Args:
        image_path: Изображение страницы.
        low_confidence_word_boxes: Рамки слов с низкой уверенностью.
        classify_fn: Классификатор фрагмента ``(crop_path) -> (mark_type,
            covers_text)``, обычно :func:`npa_ocr.ai_escalation.classify_mark`.
        crop_padding: Запас вокруг кластера при вырезании фрагмента, px.

    Returns:
        Классифицированные метки.
    """
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
        except Exception as exc:
            print(f"[ИИ-эскалация] классификация не удалась: {exc}", file=sys.stderr)
            continue
        finally:
            tmp_path.unlink(missing_ok=True)
        marks.append(Mark(bbox=region, mark_type=mark_type, covers_text=covers_text))
    return marks
