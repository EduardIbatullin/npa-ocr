"""Тесты разбора ответа Ollama (ai_escalation.py) — сетевая часть (_call_ollama)
не тестируется юнит-тестами (см. docs/plan.md — проверено вручную на реальном
сервере, не на CI)."""

from __future__ import annotations

from npa_ocr.ai_escalation import _parse_classify_response


def test_parse_classify_response_well_formed():
    response = "TYPE: штамп_регистрации\nCOVERS_TEXT: да"
    assert _parse_classify_response(response) == ("штамп_регистрации", True)


def test_parse_classify_response_no_mark():
    response = "TYPE: не_метка\nCOVERS_TEXT: нет"
    assert _parse_classify_response(response) == ("не_метка", False)


def test_parse_classify_response_tolerates_extra_whitespace_and_case():
    response = "  type:  Электронная_Подпись  \n  covers_text:  Да, перекрывает текст\n"
    assert _parse_classify_response(response) == ("электронная_подпись", True)


def test_parse_classify_response_unknown_type_falls_back_to_not_a_mark():
    # Модель не всегда держит формат — неизвестное значение не должно
    # маскироваться под случайный существующий тип.
    response = "TYPE: что-то странное\nCOVERS_TEXT: да"
    mark_type, covers_text = _parse_classify_response(response)
    assert mark_type == "не_метка"
    assert covers_text is True


def test_parse_classify_response_missing_lines_defaults_safe():
    # Пустой/мусорный ответ — безопасный дефолт (не метка, не перекрывает),
    # не угадываем на основе мусора.
    assert _parse_classify_response("бла бла бла") == ("не_метка", False)
