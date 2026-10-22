"""Качает и кэширует в образ все модели, которые использует npa-ocr — TATR
(структура и отдельно детекция на целой странице, см. table_structure.py),
полный стек PaddleX внутри PPStructureV3 (ocr.py, _default_predict_tables_fn)
и RapidOCR для текста в ячейках таблиц (ocr.py, _default_cell_text_fn).

Запускается один раз на этапе `docker build` (см. Dockerfile), не в рантайме:
без этого шага первый реальный запрос к сервису (npa_ocr.server) тратил бы
на скачивание моделей от нескольких минут до часа (наблюдалось на практике —
нестабильная сеть к Hugging Face/ModelScope, см. docs/plan.md) вместо
мгновенного старта.

На CPU без AVX2 (см. `_paddle_runtime_usable`) голый `import paddle` падает
SIGILL — не только в рантайме, но и здесь, на этапе сборки, если собирать
образ прямо на таком хосте (а не переносить уже собранный с другой машины).
Поэтому PPStructureV3 преднагружается, только если Paddle вообще можно
импортировать на хосте сборки; иначе шаг пропускается — так же, как рантайм
сам пропускает PPStructureV3 и уходит на Tesseract-фоллбэк для регионов
таблиц (см. `table_regions_for_page`, `ocr_document`). Текст ячеек
(RapidOCR/ONNX) от этого не зависит и грузится всегда.

Для RapidOCR мало просто СОЗДАТЬ движок (`RapidOCR(params=...)`) — проверено
на реальном деплое (2026-10-22): конструктор грузит только детектор и
классификатор (общие для всех языков, идут в комплекте), а языковая модель
распознавания (eslav_PP-OCRv5_rec_mobile, specific to Rec.lang_type=ESLAV)
скачивается ЛЕНИВО — только при первом настоящем вызове движка на
изображении. Без явного вызова она оставалась не в образе, а значит
качалась с ModelScope при первом же реальном документе в проде (первый
же прогон после пересборки упал без трассировки — с высокой вероятностью
именно из-за этого: нестабильная сеть к ModelScope в момент лениво
начавшейся загрузки посреди инференса). Поэтому движок явно вызывается на
тестовом изображении ниже (`_preload_rapidocr_rec_model`), не только
создаётся.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from npa_ocr import table_structure
from npa_ocr.ocr import _default_cell_text_fn, _default_predict_tables_fn, _paddle_runtime_usable


def _preload_rapidocr_rec_model() -> None:
    read = _default_cell_text_fn()
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_path = Path(tmp_dir) / "dummy.png"
        Image.new("RGB", (64, 32), "white").save(dummy_path)
        read(dummy_path)


print("Загрузка TATR (структура)...")
table_structure.load_tatr()

print("Загрузка TATR (детекция на целой странице)...")
table_structure.load_tatr_detector()

if _paddle_runtime_usable():
    print("Загрузка PPStructureV3 (весь стек PaddleX)...")
    _default_predict_tables_fn()
else:
    print(
        "PaddlePaddle недоступен на этом CPU сборки (нет AVX2 / SIGILL) — "
        "PPStructureV3 не преднагружается, в рантайме регионы таблиц возьмёт Tesseract."
    )

print("Загрузка RapidOCR (eslav_PP-OCRv5, текст ячеек таблиц)...")
_preload_rapidocr_rec_model()

print("Все модели закэшированы.")
