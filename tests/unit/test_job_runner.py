"""Фоновые задания распознавания (подпроцесс CLI, см. npa_ocr.job_runner)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from npa_ocr.job_runner import JobBusyError, JobManager, read_progress

FAKE_OCR = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "print('[  50.0%] страница 1 из 2: готово', file=sys.stderr, flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
    "Path(sys.argv[1]).write_bytes(b'docx')\n"
)

FAKE_OCR_FAILS = (
    "import sys\n"
    "print('{\"status\": \"insufficient_memory\", \"message\": \"мало памяти\"}')\n"
    "sys.exit(1)\n"
)

# Симулирует флаки-SIGILL (см. docs/history/plan.md) — убивает себя сигналом вместо
# обычного exit, что даёт subprocess.Popen отрицательный return_code. Счётчик
# попыток живёт во внешнем файле (counter), а не в памяти процесса — каждый
# повтор JobManager — это новый процесс.
FAKE_OCR_CRASHES_THEN_SUCCEEDS = (
    "import sys, os, signal\n"
    "from pathlib import Path\n"
    "counter = Path(sys.argv[2])\n"
    "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
    "counter.write_text(str(n))\n"
    "if n == 1:\n"
    "    os.kill(os.getpid(), signal.SIGILL)\n"
    "Path(sys.argv[1]).write_bytes(b'docx')\n"
)

FAKE_OCR_ALWAYS_CRASHES = "import os, signal\nos.kill(os.getpid(), signal.SIGILL)\n"


def _manager(tmp_path: Path, sleep_seconds: float, *, script: str = FAKE_OCR) -> JobManager:
    return JobManager(
        tmp_path,
        command_builder=lambda pdf, docx, authority: [sys.executable, "-c", script, str(docx), str(sleep_seconds)],
    )


def _wait(job, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)


def test_read_progress_takes_last_percent_and_steps(tmp_path: Path):
    log = tmp_path / "p.log"
    log.write_text(
        "[  0.0%] страница 1 из 21: распознаётся\n"
        "  [стр. 1] текст: Tesseract: 1 с\n"
        "[ 23.8%] страница 6 из 21: распознаётся\n"
        "  [стр. 6] рендер страницы: 0 с\n",
        encoding="utf-8",
    )
    result = read_progress(log)
    assert result["percent"] == 23.8
    assert result["page"] == 6
    assert result["total"] == 21
    assert result["steps"] == ["текст: Tesseract: 1 с", "рендер страницы: 0 с"]


def test_read_progress_handles_missing_file(tmp_path: Path):
    assert read_progress(tmp_path / "none.log")["percent"] is None


def test_manager_runs_one_job_and_produces_docx(tmp_path: Path):
    manager = _manager(tmp_path, sleep_seconds=0.3)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    job = manager.start(pdf)
    assert job.status == "running"
    _wait(job)
    assert job.status == "done"
    assert job.docx_path.read_bytes() == b"docx"
    assert job.to_dict()["docx_ready"] is True


def test_manager_refuses_second_job_while_one_runs(tmp_path: Path):
    manager = _manager(tmp_path, sleep_seconds=3)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    first = manager.start(pdf)
    with pytest.raises(JobBusyError):
        manager.start(pdf)
    manager.cancel(first)
    _wait(first)
    assert first.status == "cancelled"


def test_manager_reports_insufficient_memory_failure(tmp_path: Path):
    manager = _manager(tmp_path, sleep_seconds=0, script=FAKE_OCR_FAILS)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    job = manager.start(pdf)
    _wait(job)
    assert job.status == "failed"
    assert job.error == "мало памяти"


@pytest.mark.skipif(sys.platform == "win32", reason="симуляция краха сигналом требует POSIX os.kill/signal")
def test_manager_retries_after_crash_signal_and_succeeds(tmp_path: Path):
    counter = tmp_path / "attempts.txt"
    manager = JobManager(
        tmp_path,
        command_builder=lambda pdf, docx, authority: [
            sys.executable,
            "-c",
            FAKE_OCR_CRASHES_THEN_SUCCEEDS,
            str(docx),
            str(counter),
        ],
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    job = manager.start(pdf)
    _wait(job)
    assert job.status == "done"
    assert job.retries == 1
    assert int(counter.read_text()) == 2
    assert "подпроцесс упал" in job.progress_path.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="симуляция краха сигналом требует POSIX os.kill/signal")
def test_manager_gives_up_after_max_crash_retries(tmp_path: Path):
    manager = JobManager(
        tmp_path,
        command_builder=lambda pdf, docx, authority: [sys.executable, "-c", FAKE_OCR_ALWAYS_CRASHES],
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    job = manager.start(pdf)
    _wait(job)
    assert job.status == "failed"
    assert job.retries == JobManager.MAX_CRASH_RETRIES
    assert f"после {JobManager.MAX_CRASH_RETRIES + 1} попыток" in job.error


@pytest.mark.skipif(sys.platform == "win32", reason="cancel() во время повтора — таймингоёмкий тест, см. docstring")
def test_manager_cancel_stops_retry_loop(tmp_path: Path):
    manager = JobManager(
        tmp_path,
        command_builder=lambda pdf, docx, authority: [sys.executable, "-c", FAKE_OCR_ALWAYS_CRASHES],
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    job = manager.start(pdf)
    time.sleep(0.2)
    manager.cancel(job)
    _wait(job)
    assert job.status == "cancelled"


def test_manager_allows_next_job_after_previous_finishes(tmp_path: Path):
    manager = _manager(tmp_path, sleep_seconds=0.1)
    first_pdf = tmp_path / "one.pdf"
    first_pdf.write_bytes(b"%PDF")
    first = manager.start(first_pdf)
    _wait(first)
    assert first.status == "done"

    second_pdf = tmp_path / "two.pdf"
    second_pdf.write_bytes(b"%PDF")
    second = manager.start(second_pdf)
    _wait(second)
    assert second.status == "done"
    assert second.job_id != first.job_id
