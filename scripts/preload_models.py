"""Качает и кэширует в образ все модели, которые использует npa-ocr — TATR
(структура и отдельно детекция на целой странице, см. table_structure.py) и
полный стек PaddleX внутри PPStructureV3 (ocr.py, _default_predict_tables_fn).

Запускается один раз на этапе `docker build` (см. Dockerfile), не в рантайме:
без этого шага первый реальный запрос к сервису (npa_ocr.server) тратил бы
на скачивание моделей от нескольких минут до часа (наблюдалось на практике —
нестабильная сеть к Hugging Face/ModelScope, см. docs/plan.md) вместо
мгновенного старта.
"""

from __future__ import annotations

from npa_ocr import table_structure
from npa_ocr.ocr import _default_predict_tables_fn

print("Загрузка TATR (структура)...")
table_structure.load_tatr()

print("Загрузка TATR (детекция на целой странице)...")
table_structure.load_tatr_detector()

print("Загрузка PPStructureV3 (весь стек PaddleX)...")
_default_predict_tables_fn()

print("Все модели закэшированы.")
