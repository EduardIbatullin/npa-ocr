"""Фоновые задания распознавания: CLI ``npa-ocr`` в подпроцессе.

Используется HTTP API (:mod:`npa_ocr.server`). Подпроцесс изолирует сервер:
падение, нехватка памяти или зависание моделей (PaddleOCR, torch) не роняют его.

Одновременно выполняется одно задание: модели требовательны к памяти (см.
README, «Установка»).

Ход распознавания берётся из stderr подпроцесса — строк вида
«[ x%] страница N из M» (см. README, «Внешний контракт CLI»).
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
    """Уже выполняется другое задание; одновременно разрешено одно.

    Attributes:
        job_id: Идентификатор выполняющегося задания.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Уже выполняется распознавание (задание {job_id}). Дождитесь окончания.")
        self.job_id = job_id


def read_progress(progress_path: Path) -> dict:
    """Разбирает журнал хода распознавания подпроцесса.

    Args:
        progress_path: Журнал (stderr подпроцесса).

    Returns:
        Словарь с ключами ``percent``, ``page``, ``total``, ``page_status`` из
        последней строки «[ x%] страница N из M: статус» (None, если таких строк
        нет) и ``steps`` — до шести последних завершённых шагов
        («[стр. N] шаг: время»).
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
    """Состояние фонового задания распознавания."""

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
        """Возвращает состояние задания для ответа ``GET /jobs/{job_id}``."""
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
    """Собирает команду CLI ``npa-ocr`` для задания.

    ИИ-эскалация включена всегда: ответ VLM не заменяет настоящий текст хуже
    исходного неуверенного, а без связи с Ollama (``OLLAMA_URL``) эскалация
    пропускается без ошибки. Движок таблиц задаёт переменная окружения
    ``NPA_OCR_TABLE_ENGINE`` (``tatr`` по умолчанию или ``vlm``).

    Args:
        pdf_path: Входной PDF.
        docx_path: Выходной DOCX.
        authority_name: Орган, издавший акт, или None.

    Returns:
        Аргументы команды.
    """
    cmd = ["npa-ocr", str(pdf_path), "--out", str(docx_path), "--ai-escalation"]
    # движок таблиц — настройка сервиса, а не запроса: vlm требует доступного vLLM (VLLM_URL)
    cmd += ["--table-engine", os.environ.get("NPA_OCR_TABLE_ENGINE", "tatr")]
    if authority_name:
        cmd += ["--authority-name", authority_name]
    return cmd


class JobManager:
    """Запускает задания распознавания подпроцессами CLI и следит за ними.

    Одновременно выполняется одно задание; запуск второго вызывает
    :class:`JobBusyError`.

    Завершение подпроцесса по сигналу (отрицательный код возврата, например
    -4 для SIGILL) повторяется до ``MAX_CRASH_RETRIES`` раз: такие падения
    случайны (гонка потоков BLAS/OpenMP при первом выводе модели на CPU без
    AVX2/FMA), а не вызваны данными. Повтор продолжает распознавание с уже
    сохранённых страниц: страницы пишутся атомарно, поэтому недописанная
    страница не примется за готовую.

    Args:
        jobs_root: Каталог файлов заданий. Файлы задания различаются по stem
            входного PDF, уникальность которого обеспечивает вызывающий код.
        command_builder: Сборка команды запуска; по умолчанию — CLI ``npa-ocr``.
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
        """Возвращает задание по идентификатору или None."""
        return self._jobs.get(job_id)

    def start(self, pdf_path: Path, *, authority_name: str | None = None) -> Job:
        """Запускает распознавание сохранённого PDF.

        Args:
            pdf_path: PDF на диске с уникальным stem.
            authority_name: Орган, издавший акт; передаётся в CLI как
                ``--authority-name``.

        Returns:
            Запущенное задание. За его завершением следит фоновый поток.

        Raises:
            JobBusyError: Уже выполняется другое задание.
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
        """Запускает подпроцесс CLI для задания (первый запуск или повтор).

        stdout пишется заново: итоговый JSON CLI печатает только при успехе.
        stderr дописывается, чтобы сохранить отметки о предыдущих падениях.
        """
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
        """Ждёт завершения подпроцесса и выставляет итоговый статус задания.

        Выполняется в отдельном потоке. Статус отменённого задания не меняется.
        Падение по сигналу перезапускает подпроцесс (см. :class:`JobManager`).
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
        """Дописывает в журнал хода отметку о падении подпроцесса и повторе."""
        marker = f"\n--- подпроцесс упал (сигнал {-code}), повтор {job.retries}/{JobManager.MAX_CRASH_RETRIES} ---\n"
        try:
            with open(job.progress_path, "a", encoding="utf-8") as f:
                f.write(marker)
        except OSError:
            pass

    def _failure_reason(self, job: Job) -> str:
        """Возвращает понятную причину неудачи задания.

        При нехватке памяти CLI печатает в stdout JSON с сообщением — оно и
        возвращается. Иначе возвращается код возврата, число попыток и имя
        журнала хода.
        """
        try:
            out = json.loads(job.log_path.read_text(encoding="utf-8", errors="replace"))
            if out.get("status") == "insufficient_memory":
                return out.get("message", "Недостаточно памяти")
        except (OSError, ValueError):
            pass
        attempts_note = f" (после {job.retries + 1} попыток)" if job.retries else ""
        return f"Распознавание завершилось с кодом {job.return_code}{attempts_note}. Журнал: {job.progress_path.name}"
