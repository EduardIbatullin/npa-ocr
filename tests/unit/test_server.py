"""HTTP API (npa_ocr.server) — на фейковом JobManager, без реальных тяжёлых
движков (см. test_job_runner.py для самого JobManager)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from npa_ocr import server as server_module
from npa_ocr.job_runner import JobManager

FAKE_OCR = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "print('[  50.0%] страница 1 из 1: готово', file=sys.stderr, flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
    "Path(sys.argv[1]).write_bytes(b'fake docx content')\n"
)


def _client(tmp_path: Path, monkeypatch, *, sleep_seconds: float = 0.1) -> TestClient:
    manager = JobManager(
        tmp_path / "jobs",
        command_builder=lambda pdf, docx, authority: [
            sys.executable, "-c", FAKE_OCR, str(docx), str(sleep_seconds)
        ],
    )
    monkeypatch.setattr(server_module, "_manager", manager)
    monkeypatch.setattr(server_module, "UPLOADS_DIR", tmp_path / "jobs" / "uploads")
    return TestClient(server_module.app)


def _wait_done(client: TestClient, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/jobs/{job_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.05)
    raise TimeoutError("задание не завершилось вовремя")


def test_health():
    client = TestClient(server_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_create_job_and_poll_until_done(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/jobs", files={"file": ("scan.pdf", b"%PDF-fake", "application/pdf")})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "running"

    final = _wait_done(client, job_id)
    assert final["status"] == "done"
    assert final["docx_ready"] is True


def test_download_result_after_done(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = client.post(
        "/jobs", files={"file": ("scan.pdf", b"%PDF-fake", "application/pdf")}
    ).json()["job_id"]
    _wait_done(client, job_id)

    resp = client.get(f"/jobs/{job_id}/result")
    assert resp.status_code == 200
    assert resp.content == b"fake docx content"
    assert resp.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_result_404_before_job_is_done(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, sleep_seconds=5)
    job_id = client.post(
        "/jobs", files={"file": ("scan.pdf", b"%PDF-fake", "application/pdf")}
    ).json()["job_id"]
    resp = client.get(f"/jobs/{job_id}/result")
    assert resp.status_code == 404


def test_unknown_job_id_returns_404(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/result").status_code == 404
    assert client.post("/jobs/does-not-exist/cancel").status_code == 404


def test_second_job_while_busy_returns_409(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, sleep_seconds=3)
    first = client.post("/jobs", files={"file": ("a.pdf", b"%PDF-a", "application/pdf")})
    assert first.status_code == 202
    second = client.post("/jobs", files={"file": ("b.pdf", b"%PDF-b", "application/pdf")})
    assert second.status_code == 409

    client.post(f"/jobs/{first.json()['job_id']}/cancel")


def test_cancel_stops_running_job(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, monkeypatch, sleep_seconds=5)
    job_id = client.post(
        "/jobs", files={"file": ("scan.pdf", b"%PDF-fake", "application/pdf")}
    ).json()["job_id"]

    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    final = _wait_done(client, job_id)
    assert final["status"] == "cancelled"


_WRITE_SCRIPT = "import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_bytes(b'x')\n"


def test_authority_name_passed_through_to_command(tmp_path: Path, monkeypatch):
    captured = {}

    def command_builder(pdf, docx, authority):
        captured["authority"] = authority
        return [sys.executable, "-c", _WRITE_SCRIPT, str(docx)]

    manager = JobManager(tmp_path / "jobs", command_builder=command_builder)
    monkeypatch.setattr(server_module, "_manager", manager)
    monkeypatch.setattr(server_module, "UPLOADS_DIR", tmp_path / "jobs" / "uploads")
    client = TestClient(server_module.app)

    job_id = client.post(
        "/jobs",
        files={"file": ("scan.pdf", b"%PDF-fake", "application/pdf")},
        data={"authority_name": "Правительство Российской Федерации"},
    ).json()["job_id"]
    _wait_done(client, job_id)
    assert captured["authority"] == "Правительство Российской Федерации"
