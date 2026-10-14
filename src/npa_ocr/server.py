"""HTTP API поверх CLI npa-ocr — асинхронные задания распознавания для внешних
потребителей (см. README, «HTTP API»).

Не знает ни о каком конкретном потребителе — принимает PDF файлом, отдаёт
DOCX файлом; реквизиты/разметка документа не его забота. Один и тот же образ Docker служит и CLI (ENTRYPOINT ["npa-ocr"]
по умолчанию), и этим сервером — см. README, «Docker», раздел про запуск
сервера отдельной командой.

Задания — JobManager (job_runner.py) запускает CLI как подпроцесс (self-exec):
падение/OOM в подпроцессе не роняет сам сервер, см. докстринг job_runner.py.
Одновременно выполняется не больше одного задания — POST /jobs отвечает 409,
пока предыдущее ещё running.
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
    job = _manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    return job


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    authority_name: str | None = Form(default=None),
) -> dict:
    """Загружает PDF и запускает распознавание. Возвращает job_id сразу, не
    дожидаясь завершения — прогресс см. GET /jobs/{job_id}.

    Raises:
        HTTPException 409: уже выполняется другое задание (см. JobManager —
            одновременно разрешено только одно, тяжёлые модели в памяти).
    """
    data = await file.read()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Имя — случайное, не оригинальное из запроса: job_runner.JobManager строит
    # пути файлов задания из stem этого пути, он должен быть уникален между
    # заданиями (оригинальные имена могли бы совпасть у разных запросов).
    upload_path = UPLOADS_DIR / f"{uuid.uuid4().hex}.pdf"
    upload_path.write_bytes(data)
    try:
        job = _manager.start(upload_path, authority_name=authority_name)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job.job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job_or_404(job_id).to_dict()


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
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
    job = _job_or_404(job_id)
    _manager.cancel(job)
    return job.to_dict()
