"""Восстановление текста, перекрытого печатью, штампом или подписью, через VLM.

Используется локальная модель qwen3-vl:8b в Ollama. Эскалация дополняет
обычный OCR, а не заменяет его: она вызывается только для слов и строк, где
Tesseract или RapidOCR дали низкую уверенность (см.
``DEFAULT_CONFIDENCE_THRESHOLD`` в :mod:`npa_ocr.ocr`).

Промпт запрещает модели додумывать: она выводит только видимый текст, а
неразборчивое слово заменяет на ``[нрзб.]``. Без такого запрета модель
добавляет слова, которых нет на изображении.

Адрес Ollama задаёт переменная окружения ``OLLAMA_URL``, по умолчанию
``http://localhost:11434``. Из контейнера, когда Ollama работает в другой сети
Docker, обычно указывают ``http://host.docker.internal:11434``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen3-vl:8b"

# Совпадает с ocr.UNRECOGNIZABLE_MARK; не импортируется, чтобы не было
# циклического импорта (ocr вызывает этот модуль).
_UNRECOGNIZABLE_MARK = "[нрзб.]"

_PROMPT = f"""На изображении — фрагмент официального документа. Часть текста перекрыта печатью, штампом или подписью.

Задача: прочитать и вывести ТОЛЬКО тот текст документа, который перекрыт (заслонён печатью/штампом/подписью). Саму печать/штамп описывать не нужно, её собственное содержимое (номер, название органа, дату на самой печати) выводить не нужно — только скрытый ею текст документа.

Строгие правила:
- Выводи только то, что физически видно на изображении, в том числе частично видимые под краем печати буквы и цифры.
- Ничего не добавляй и не придумывай — ни слов, ни окончаний, ни знаков пунктуации, которых не видно на картинке.
- Если слово не удаётся разобрать даже частично — выведи {_UNRECOGNIZABLE_MARK} вместо него вместо того, чтобы угадывать.
- Не используй общие знания о документе (кто подписант, какая должность и т.п.) — только то, что видно на изображении.

Ответ — только восстановленный текст, без пояснений и комментариев."""

# Классификация кандидата (скопления неуверенных слов, см. mark_detection):
# что это и перекрывает ли оно текст документа. Ответ — строки «КЛЮЧ: значение»,
# а не JSON: VLM нередко оборачивает JSON в пояснения, построчный формат
# разбирается надёжнее (см. _parse_classify_response).
_CLASSIFY_PROMPT = """На изображении — кандидат на печать, штамп или подпись в официальном документе (область найдена автоматически по скоплению плохо распознанного текста, на картинке может оказаться и что-то другое — часть обычного текста, артефакт скана).

Ответь СТРОГО в две строки, без пояснений:
TYPE: один из печать / штамп_регистрации / электронная_подпись / не_метка
COVERS_TEXT: да / нет (перекрывает ли эта метка текст ДОКУМЕНТА — не свой собственный номер/дату/QR-код на самой метке, а другой, содержательный текст под/рядом с ней)

Если на кропе вообще нет печати/штампа/подписи (случайный фрагмент обычного текста) — TYPE: не_метка, COVERS_TEXT: нет."""


class EscalationError(RuntimeError):
    """Ollama недоступна или ответила ошибкой; запасной путь выбирает вызывающий код."""


def _call_ollama(prompt: str, crop_path: Path, *, timeout: float) -> str:
    """Отправляет промпт и изображение в Ollama и возвращает текст ответа.

    Raises:
        EscalationError: Ollama недоступна или ответ без поля ``response``.
    """
    image_b64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise EscalationError(f"Ollama ({OLLAMA_URL}) недоступна: {exc}") from exc
    if "response" not in result:
        raise EscalationError(f"Ollama вернула неожиданный ответ: {result}")
    return result["response"].strip()


def recover_covered_text(crop_path: Path, *, timeout: float = 120.0) -> str:
    """Восстанавливает текст документа, перекрытый меткой на фрагменте.

    Args:
        crop_path: PNG фрагмента страницы с меткой.
        timeout: Таймаут запроса к Ollama, с.

    Returns:
        Восстановленный текст; неразборчивые слова заменены на ``[нрзб.]``.

    Raises:
        EscalationError: Ollama недоступна или ответила ошибкой. Вызывающий
            код может, например, оставить ``[нрзб.]``, как при обычной низкой
            уверенности.
    """
    return _call_ollama(_PROMPT, crop_path, timeout=timeout)


_VALID_MARK_TYPES = {"печать", "штамп_регистрации", "электронная_подпись", "не_метка"}


def _parse_classify_response(response: str) -> tuple[str, bool]:
    """Разбирает ответ на ``_CLASSIFY_PROMPT``.

    Строки ищутся по префиксу без учёта регистра, пробелов и порядка: VLM не
    всегда держит формат буквально. Неизвестный тип считается ``не_метка``.

    Returns:
        Тип метки и признак того, что она перекрывает текст документа.
    """
    mark_type = "не_метка"
    covers_text = False
    for line in response.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("type:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in _VALID_MARK_TYPES:
                mark_type = candidate
        elif lower.startswith("covers_text:"):
            value = line.split(":", 1)[1].strip().lower()
            covers_text = value.startswith("да")
    return mark_type, covers_text


def classify_mark(crop_path: Path, *, timeout: float = 120.0) -> tuple[str, bool]:
    """Классифицирует область-кандидат на метку.

    Args:
        crop_path: PNG области (см. :func:`npa_ocr.mark_detection.cluster_candidate_regions`).
        timeout: Таймаут запроса к Ollama, с.

    Returns:
        Пара ``(mark_type, covers_text)``: тип — ``"печать"``,
        ``"штамп_регистрации"``, ``"электронная_подпись"`` или ``"не_метка"``;
        признак — перекрывает ли метка текст документа.

    Raises:
        EscalationError: Ollama недоступна или ответила ошибкой.
    """
    response = _call_ollama(_CLASSIFY_PROMPT, crop_path, timeout=timeout)
    return _parse_classify_response(response)
