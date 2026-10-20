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
"""

from __future__ import annotations

from npa_ocr import table_structure
from npa_ocr.ocr import _default_cell_text_fn, _default_predict_tables_fn, _paddle_runtime_usable

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
_default_cell_text_fn()

print("Все модели закэшированы.")
