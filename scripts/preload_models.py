"""Загружает в образ все модели npa-ocr при сборке.

Модели: TATR (структура таблиц и поиск таблиц на странице), PaddleX внутри
PPStructureV3 и RapidOCR для текста ячеек. Скрипт выполняется при
``docker build``, чтобы первый запрос к сервису не ждал загрузки моделей
(при нестабильной сети к Hugging Face и ModelScope — до часа).

На CPU без AVX2 ``import paddle`` падает с SIGILL, поэтому PPStructureV3
загружается, только если PaddlePaddle импортируется на хосте сборки (так же
его пропускает и распознавание). RapidOCR загружается всегда.

Языковую модель распознавания RapidOCR (eslav_PP-OCRv5_rec_mobile) конструктор
не загружает: она скачивается при первом распознавании, а детектор вызывает
распознаватель, только если нашёл текст. Поэтому движок вызывается на
изображении с нарисованным текстом (встроенным шрифтом Pillow — в образе нет
системных шрифтов).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from npa_ocr import table_structure
from npa_ocr.ocr import _default_cell_text_fn, _default_predict_tables_fn, _paddle_runtime_usable


def _preload_rapidocr_rec_model() -> None:
    """Вызывает RapidOCR на изображении с текстом, чтобы загрузилась модель распознавания."""
    read = _default_cell_text_fn()
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_path = Path(tmp_dir) / "dummy.png"
        img = Image.new("RGB", (200, 60), "white")
        ImageDraw.Draw(img).text((5, 5), "TEST", fill="black", font=ImageFont.load_default(size=40))
        img.save(dummy_path)
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
