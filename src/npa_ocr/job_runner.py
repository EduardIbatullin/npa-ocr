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
            "error": self.error,
        }


def _default_command(pdf_path: Path, docx_path: Path, authority_name: str | None) -> list[str]:
    """Команда CLI npa-ocr (self-exec, см. докстринг модуля)."""
    cmd = ["npa-ocr", str(pdf_path), "--out", str(docx_path)]
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
    """

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
            )
            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "FLAGS_use_mkldnn": "false",
            }
            cmd = self._command_builder(pdf_path, docx_path, authority_name)
            with open(job.log_path, "wb") as stdout_file, open(progress_path, "wb") as stderr_file:
                job.process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, env=env)
            self._jobs[job.job_id] = job
            self._current = job
        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

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
        """
        code = job.process.wait()
        with self._lock:
            job.return_code = code
            job.finished_at = time.time()
            if job.status == "cancelled":
                return
            if code == 0 and job.docx_path.is_file():
                job.status = "done"
            else:
                job.status = "failed"
                job.error = self._failure_reason(job)

    @staticmethod
    def _failure_reason(job: Job) -> str:
        """Человекочитаемая причина неудачи задания.

        CLI npa-ocr при нехватке памяти печатает JSON со статусом в stdout — если
        получилось его распарсить, возвращает это сообщение; иначе — обобщённый
        текст с кодом возврата и ссылкой на журнал прогресса.
        """
        try:
            out = json.loads(job.log_path.read_text(encoding="utf-8", errors="replace"))
            if out.get("status") == "insufficient_memory":
                return out.get("message", "Недостаточно памяти")
        except (OSError, ValueError):
            pass
        return f"Распознавание завершилось с кодом {job.return_code}. Журнал: {job.progress_path.name}"
