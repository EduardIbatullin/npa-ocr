"""Проверка ответа OCR-модели до удаления служебной разметки."""

from __future__ import annotations

import re

UNREADABLE = "[не распознано]"
_SERVICE = re.compile(
    r'"(?:line_type|line_endpoint)"\s*:|'
    r"\b(?:I(?:'m| am) sorry|I (?:cannot|can't|am unable to)|"
    r"the (?:provided )?image (?:provided )?is (?:a graphic|not a)|"
    r"does not contain any data points)\b|"
    r"(?:не могу|невозможно) (?:распознать|извлечь|прочитать)",
    re.IGNORECASE,
)


def is_service_response(text: str) -> bool:
    """Возвращает True для описания линий, отказа модели или маркера утраты.

    Проверяется исходный ответ, пока очистка не скрыла признаки служебного
    формата. Обычные английские названия и химические формулы допускаются.
    """
    return UNREADABLE in text or bool(_SERVICE.search(text))


def matches_cell_type(text: str, expected: str) -> bool:
    """Проверяет тип графы, не подставляя ожидаемое значение из справочника."""
    from npa_ocr.scientific_text import ROMAN_CODE, is_chemical_formula

    text = text.strip()
    if re.fullmatch(r"[-–—−]", text):
        return True
    if expected == "formula":
        return is_chemical_formula(text)
    if expected == "code":
        return bool(ROMAN_CODE.fullmatch(text) or re.fullmatch(r"[АA]-\d+", text))
    if expected == "number":
        if re.fullmatch(r"\d+(?:[.,]\d+)?(?:\.{2,}|…)",text):
            return False
        return bool(re.match(r"^(?:(?:не более|не менее|до)\s+)?(?:[≤≥<>]=?\s*)?[+−–—-]?\d", text, re.I))
    if expected == "index":
        return bool(re.fullmatch(r"\d{1,4}[.)]?", text))
    if expected == "cas":
        return bool(re.fullmatch(r"\d{2,7}-\d{2}-\d", text))
    return True
