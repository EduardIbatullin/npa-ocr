"""Фоновые задания распознавания — запускает CLI `npa-ocr` как подпроцесс
(себя же, другим входом).

Используется HTTP-сервисом (npa_ocr.server) ради изоляции: падение, OOM или
зависание тяжёлых моделей (PaddleOCR, torch) в подпроцессе не роняет сам сервис.

Одновременно выполняется не больше одного задания — параллельные прогоны
моделей требовательны к памяти (см. README, «Установка»).

Прогресс берётся из журнала stderr дочернего процесса (строки
«[ x%] страница N из M») — тот же «Внешний контракт», что описан в README.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROGRESS_RE = re.compile(r"\[\s*([\d.]+)%\]\s*страница\s+(\d+)\s+из\s+(\d+):\s*(.+)")
STEP_RE = re.compile(r"^\s*\[стр\.\s*\d+\]\s*(.+?):\s*([\d.]+)\s*(с|мин)")


class JobBusyError(RuntimeError):
    """Другое задание распознавания уже выполняется (разрешено только одно одновременно)."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Уже выполняется распознавание (задание {job_id}). Дождитесь окончания.")
        self.job_id = job_id


def read_progress(progress_path: Path) -> dict:
    """Читает журнал прогресса дочернего процесса и разбирает последние строки.

    Returns:
        dict с ключами percent/page/total/page_status (из последней строки вида
        «[ x%] страница N из M: статус»; все None, если журнала ещё нет или строк
        такого вида в нём не нашлось) и steps — до 6 последних завершённых шагов
        (строки вида «[стр. N] шаг: время»).
    """
    result = {"percent": None, "page": None, "total": None, "page_status": None, "steps": []}
    if not progress_path.is_file():
        return result
    try:
        text = progress_path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return result
    for line in text.splitlines():
        m = PROGRESS_RE.search(line)
        if m:
            result.update(
                percent=float(m.group(1)),
                page=int(m.group(2)),
                total=int(m.group(3)),
                page_status=m.group(4).strip(),
            )
            continue
        s = STEP_RE.match(line)
        if s:
            result["steps"].append(f"{s.group(1)}: {s.group(2)} {s.group(3)}")
    result["steps"] = result["steps"][-6:]
    return result


@dataclass
class Job:
    """Состояние одного фонового задания распознавания — то, что видит HTTP-клиент."""

    job_id: str
    pdf_path: Path
    docx_path: Path
    progress_path: Path
    log_path: Path
    process: subprocess.Popen | None = None
    status: str = "running"  # running | done | failed | cancelled
    return_code: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    authority_name: str | None = None
    retries: int = 0

    def to_dict(self) -> dict:
        """Снимок состояния для JSON-ответа (GET /jobs/{job_id})."""
        progress = read_progress(self.progress_path)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "docx_ready": self.status == "done" and self.docx_path.is_file(),
            "percent": progress["percent"],
            "page": progress["page"],
            "total_pages": progress["total"],
            "page_status": progress["page_status"],
            "steps": progress["steps"],
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at),
            "return_code": self.return_code,
            "retries": self.retries,
            "error": self.error,
        }


def _default_command(pdf_path: Path, docx_path: Path, authority_name: str | None) -> list[str]:
    """Команда CLI npa-ocr (self-exec, см. докстринг модуля).

    --ai-escalation включена здесь всегда: защита от замены настоящего текста
    плохим ответом VLM (счётчик слов на метку, см. docs/plan.md, «штампы
    просачиваются в текст») уже проверена на реальном документе и безопасна
    по деградации — эскалация ничего не портит хуже исходного низкоуверенного
    текста, только иногда не убирает мусор штампа (отдельный, пока открытый
    дефект). Требует сеть до Ollama на хосте (см. OLLAMA_URL в
    docker-compose.yml) — без неё classify_mark/recover_covered_text просто
    логируют ошибку и остаются без эскалации (mark_detection.detect_marks
    глушит исключения классификации), страница не падает."""
    cmd = ["npa-ocr", str(pdf_path), "--out", str(docx_path), "--ai-escalation"]
    if authority_name:
        cmd += ["--authority-name", authority_name]
    return cmd


class JobManager:
    """Запускает и отслеживает задания распознавания как подпроцессы CLI npa-ocr.

    Одновременно выполняется не больше одного задания (см. докстринг модуля) —
    попытка запустить новое, пока предыдущее ещё running, бросает JobBusyError.

    Args:
        jobs_root: корень для файлов заданий (вход/выход/журналы); один
            подкаталог не заводится — файлы одного задания различаются по
            stem входного PDF (см. start), который должен быть уникален —
            это обеспечивает вызывающий код (см. npa_ocr.server).
        command_builder: как собрать команду запуска — по умолчанию настоящий
            CLI (_default_command), в тестах подменяется на фейковую команду.

    Падение подпроцесса по сигналу (return_code < 0 — так subprocess.Popen
    сообщает о завершении сигналом, например -4 для SIGILL) автоматически
    повторяется до MAX_CRASH_RETRIES раз: это флаки-краш, не ошибка в данных
    (см. docs/plan.md, «Флаки-SIGILL» — гонка потоков BLAS/OpenMP при первом
    TATR-инференсе на CPU без AVX2/FMA; сам подпроцесс гибнет мгновенно без
    трейсбека, сервер и прочие задания не страдают). Повтор — тот же CLI-вызов
    с тем же --out: CLI резюмирует с pages_dir (см. ocr_document, resume=True
    по умолчанию) и пропускает уже сохранённые страницы — _save_page пишет их
    атомарно (.part + os.replace), так что крах посреди записи страницы не
    оставляет полуготового файла, который резюм принял бы за готовый.
    """

    MAX_CRASH_RETRIES = 2

    def __init__(
        self,
        jobs_root: Path,
        *,
        command_builder: Callable[[Path, Path, str | None], list[str]] = _default_command,
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self._command_builder = command_builder
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def start(self, pdf_path: Path, *, authority_name: str | None = None) -> Job:
        """Запускает новое задание распознавания для уже сохранённого pdf_path.

        Args:
            pdf_path: путь к уже сохранённому на диске PDF (см. npa_ocr.server —
                там входящий файл сначала сохраняется под уникальным именем).
            authority_name: передаётся в CLI как --authority-name (если задан).

        Returns:
            Созданный Job (уже запущен, status="running") — прогресс
            отслеживается фоновым потоком (_watch), не этим вызовом.

        Raises:
            JobBusyError: уже выполняется другое задание.
        """
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise JobBusyError(self._current.job_id)
            self.jobs_root.mkdir(parents=True, exist_ok=True)
            stem = pdf_path.stem
            docx_path = self.jobs_root / f"{stem}.docx"
            progress_path = self.jobs_root / f"{stem}.progress.log"
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                pdf_path=pdf_path,
                docx_path=docx_path,
                progress_path=progress_path,
                log_path=self.jobs_root / f"{stem}.log",
                authority_name=authority_name,
            )
            job.process = self._spawn(job)
            self._jobs[job.job_id] = job
            self._current = job
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _spawn(self, job: Job) -> subprocess.Popen:
        """Запускает CLI npa-ocr для job (исходный запуск или повтор после краха —
        см. _watch). stdout (log_path) — заново с нуля: контракт CLI печатает
        туда финальный JSON только при успехе, прошлый краш там и так пуст.
        stderr (progress_path) — дозаписью: _watch перед повтором уже добавил в
        него отметку о краше (см. _log_crash_retry), её не нужно терять."""
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "FLAGS_use_mkldnn": "false",
        }
        cmd = self._command_builder(job.pdf_path, job.docx_path, job.authority_name)
        with open(job.log_path, "wb") as stdout_file, open(job.progress_path, "ab") as stderr_file:
            return subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, env=env)

    def cancel(self, job: Job) -> None:
        """Останавливает задание, если оно ещё выполняется; иначе ничего не делает."""
        with self._lock:
            if job.status != "running" or job.process is None:
                return
            job.status = "cancelled"
            job.process.kill()

    def _watch(self, job: Job) -> None:
        """Ждёт завершения процесса задания и выставляет терминальный статус.

        Работает в отдельном потоке (см. start). Не трогает статус, если задание
        уже отменено (cancel уже выставил "cancelled") — не перезатирает его на
        "failed" из-за ненулевого кода возврата после kill().

        Падение по сигналу (code < 0) перезапускает тот же CLI-вызов до
        MAX_CRASH_RETRIES раз вместо немедленного "failed" — см. докстринг
        класса.
        """
        while True:
            code = job.process.wait()
            with self._lock:
                job.return_code = code
                if job.status == "cancelled":
                    job.finished_at = time.time()
                    return
                if code < 0 and job.retries < self.MAX_CRASH_RETRIES:
                    job.retries += 1
                    self._log_crash_retry(job, code)
                    job.process = self._spawn(job)
                    continue
                job.finished_at = time.time()
                if code == 0 and job.docx_path.is_file():
                    job.status = "done"
                else:
                    job.status = "failed"
                    job.error = self._failure_reason(job)
                return

    @staticmethod
    def _log_crash_retry(job: Job, code: int) -> None:
        """Дописывает в журнал прогресса отметку о краше подпроцесса сигналом и
        о повторе — чтобы след краха не терялся молча (см. _spawn)."""
        marker = f"\n--- подпроцесс упал (сигнал {-code}), повтор {job.retries}/{JobManager.MAX_CRASH_RETRIES} ---\n"
        try:
            with open(job.progress_path, "a", encoding="utf-8") as f:
                f.write(marker)
        except OSError:
            pass

    def _failure_reason(self, job: Job) -> str:
        """Человекочитаемая причина неудачи задания.

        CLI npa-ocr при нехватке памяти печатает JSON со статусом в stdout — если
        получилось его распарсить, возвращает это сообщение; иначе — обобщённый
        текст с кодом возврата (и числом попыток, если были повторы после краха)
        и ссылкой на журнал прогресса.
        """
        try:
            out = json.loads(job.log_path.read_text(encoding="utf-8", errors="replace"))
            if out.get("status") == "insufficient_memory":
                return out.get("message", "Недостаточно памяти")
        except (OSError, ValueError):
            pass
        attempts_note = f" (после {job.retries + 1} попыток)" if job.retries else ""
        return f"Распознавание завершилось с кодом {job.return_code}{attempts_note}. Журнал: {job.progress_path.name}"
