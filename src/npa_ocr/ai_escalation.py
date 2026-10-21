"""ИИ-эскалация: восстановление текста, перекрытого печатью/штампом/подписью,
через локальную VLM на сервере (Ollama, модель qwen3-vl:8b на nv-02). Это
последний рубеж — вызывается только после того, как обычный OCR
(Tesseract/RapidOCR) уже дал низкую уверенность на конкретном слове/строке
(см. LOW_CONFIDENCE_THRESHOLD/DEFAULT_CONFIDENCE_THRESHOLD в ocr.py), не
вместо него.

Промпт проверен эмпирически на реальных документах (3 разных скана с
печатью/подписью, перекрывающей текст подписанта) — без явного запрета на
«додумывание» модель один раз добавила слово, которого не было на картинке;
с текущими правилами (только то, что физически видно, неуверенное —
[нрзб.], ничего не добавлять) — 3/3 без искажений. См. docs/plan.md.

Сетевой адрес берётся из переменной окружения OLLAMA_URL (в docker-compose
npa-downloader указывает на host.docker.internal:11434 — Ollama крутится в
своём отдельном docker-compose проекте на хосте nv-02, в другой docker-сети);
без переменной — localhost:11434 (прямой запуск без контейнера).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "qwen3-vl:8b"

# Маркер должен совпадать с ocr.UNRECOGNIZABLE_MARK — не импортируется отсюда
# напрямую, чтобы не создавать цикл (ocr.py вызывает эту функцию, а не
# наоборот).
_UNRECOGNIZABLE_MARK = "[нрзб.]"

_PROMPT = f"""На изображении — фрагмент официального документа. Часть текста перекрыта печатью, штампом или подписью.

Задача: прочитать и вывести ТОЛЬКО тот текст документа, который перекрыт (заслонён печатью/штампом/подписью). Саму печать/штамп описывать не нужно, её собственное содержимое (номер, название органа, дату на самой печати) выводить не нужно — только скрытый ею текст документа.

Строгие правила:
- Выводи только то, что физически видно на изображении, в том числе частично видимые под краем печати буквы и цифры.
- Ничего не добавляй и не придумывай — ни слов, ни окончаний, ни знаков пунктуации, которых не видно на картинке.
- Если слово не удаётся разобрать даже частично — выведи {_UNRECOGNIZABLE_MARK} вместо него вместо того, чтобы угадывать.
- Не используй общие знания о документе (кто подписант, какая должность и т.п.) — только то, что видно на изображении.

Ответ — только восстановленный текст, без пояснений и комментариев."""

# Классификация кандидата (найденного дёшевой эвристикой — плотный кластер
# низкоуверенных слов, см. mark_detection.py) — что это и перекрывает ли оно
# содержательный текст документа. Ответ разбирается построчно (см.
# _parse_classify_response) — не JSON, формат проще и устойчивее для VLM,
# которая иногда оборачивает JSON в пояснения вопреки просьбе не делать этого.
_CLASSIFY_PROMPT = """На изображении — кандидат на печать, штамп или подпись в официальном документе (область найдена автоматически по скоплению плохо распознанного текста, на картинке может оказаться и что-то другое — часть обычного текста, артефакт скана).

Ответь СТРОГО в две строки, без пояснений:
TYPE: один из печать / штамп_регистрации / электронная_подпись / не_метка
COVERS_TEXT: да / нет (перекрывает ли эта метка текст ДОКУМЕНТА — не свой собственный номер/дату/QR-код на самой метке, а другой, содержательный текст под/рядом с ней)

Если на кропе вообще нет печати/штампа/подписи (случайный фрагмент обычного текста) — TYPE: не_метка, COVERS_TEXT: нет."""


class EscalationError(RuntimeError):
    """Ollama недоступна или ответила ошибкой — вызывающая сторона решает фоллбэк."""


def _call_ollama(prompt: str, crop_path: Path, *, timeout: float) -> str:
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
    """Отправляет кроп изображения в Ollama (qwen3-vl) и возвращает восстановленный текст.

    Бросает EscalationError при сетевых проблемах или ошибке модели —
    вызывающая сторона сама решает фоллбэк (например, оставить
    UNRECOGNIZABLE_MARK, как при обычном низком confidence)."""
    return _call_ollama(_PROMPT, crop_path, timeout=timeout)


_VALID_MARK_TYPES = {"печать", "штамп_регистрации", "электронная_подпись", "не_метка"}


def _parse_classify_response(response: str) -> tuple[str, bool]:
    """Построчный разбор ответа classify_mark — не JSON (см. _CLASSIFY_PROMPT).
    Нестрогое соответствие формату (лишние пробелы/регистр/порядок строк)
    разбирается построчным поиском префикса, не fail-fast по всей строке —
    VLM не всегда держит формат побуквенно."""
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


def classify_mark(crop_path: Path, *, timeout: float = 60.0) -> tuple[str, bool]:
    """Классифицирует кандидатную область (см. mark_detection.cluster_candidate_regions)
    — возвращает (mark_type, covers_text). mark_type — одно из "печать",
    "штамп_регистрации", "электронная_подпись", "не_метка".

    Бросает EscalationError при сетевых проблемах — вызывающая сторона
    (mark_detection.detect_marks) не ловит её сама, решение о фоллбэке
    принимает ocr.py (конкретно: при ошибке классификации кандидат лучше
    пропустить, чем гадать, см. docs/plan.md)."""
    response = _call_ollama(_CLASSIFY_PROMPT, crop_path, timeout=timeout)
    return _parse_classify_response(response)
