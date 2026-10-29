"""HTTP API асинхронного распознавания (см. README, «HTTP API»).

Сервер принимает PDF файлом и отдаёт DOCX файлом; о потребителях и реквизитах
документов он не знает. Тот же образ Docker по умолчанию запускает CLI, сервер
запускается отдельной командой (см. README, «Docker»).

Задания выполняет :class:`~npa_ocr.job_runner.JobManager`: CLI запускается
подпроцессом, поэтому падение или нехватка памяти в задании не роняют сервер.
Одновременно выполняется одно задание; пока оно идёт, ``POST /jobs`` отвечает 409.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from npa_ocr.job_runner import Job, JobBusyError, JobManager

JOBS_DIR = Path(os.environ.get("NPA_OCR_JOBS_DIR", "jobs"))
UPLOADS_DIR = JOBS_DIR / "uploads"

app = FastAPI(title="npa-ocr", version="0.1.0")
_manager = JobManager(JOBS_DIR)


def _job_or_404(job_id: str) -> Job:
    """Задание по идентификатору; HTTPException 404, если его нет."""
    job = _manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return job


@app.get("/health")
def health() -> dict:
    """Проверка готовности сервера."""
    return {"ok": True}


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    authority_name: str | None = Form(default=None),
) -> dict:
    """Загружает PDF и запускает распознавание.

    Ответ приходит сразу, не дожидаясь конца распознавания; ход задания
    отдаёт ``GET /jobs/{job_id}``.

    Args:
        file: PDF документа.
        authority_name: Наименование органа, издавшего акт; по нему
            восстанавливается шапка документа.

    Returns:
        ``{"job_id": ...}``.

    Raises:
        HTTPException: 409, если уже выполняется другое задание.
    """
    data = await file.read()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Имя случайное, а не из запроса: JobManager строит пути файлов задания из
    # stem этого пути, и он должен быть уникален между заданиями.
    upload_path = UPLOADS_DIR / f"{uuid.uuid4().hex}.pdf"
    upload_path.write_bytes(data)
    try:
        job = _manager.start(upload_path, authority_name=authority_name)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job.job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Состояние задания: статус, ход распознавания, ошибка."""
    return _job_or_404(job_id).to_dict()


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
    """DOCX готового задания; 404, пока распознавание не завершено."""
    job = _job_or_404(job_id)
    if job.status != "done" or not job.docx_path.is_file():
        raise HTTPException(status_code=404, detail="Распознавание ещё не завершено")
    return FileResponse(
        job.docx_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=job.docx_path.name,
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Отменяет задание и возвращает его состояние."""
    job = _job_or_404(job_id)
    _manager.cancel(job)
    return job.to_dict()
