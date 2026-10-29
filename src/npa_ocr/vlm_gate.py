"""Шлюз перед vLLM: запускает его по первому запросу и останавливает после простоя.

vLLM с DeepSeek-OCR занимает около 10 ГБ видеопамяти всё время, пока запущен.
Если GPU общая, npa-ocr обращается к vLLM через этот шлюз
(``VLLM_URL=http://<шлюз>/v1``):

- Если контейнер vLLM остановлен, шлюз запускает его через Docker API, ждёт
  ``/health`` (холодный старт — 2–3 мин) и затем передаёт запрос. Одновременные
  запросы ждут один и тот же старт.
- Если vLLM не запустился (не хватило видеопамяти, контейнер упал, истёк
  ``VLM_START_TIMEOUT``), шлюз отвечает 503, и npa-ocr распознаёт таблицы
  запасным путём (TATR).
- После ``VLM_IDLE_MIN`` минут без запросов (по умолчанию 10) контейнер
  останавливается.

Контейнер останавливается, а не удаляется, чтобы vLLM при следующем старте
использовал свой кэш компиляции. Модуль использует только стандартную
библиотеку и запускается из образа npa-ocr::

    python -m npa_ocr.vlm_gate      # нужен том /var/run/docker.sock

Переменные окружения:
    VLM_CONTAINER: Имя контейнера vLLM.
    VLM_UPSTREAM: Адрес vLLM внутри сети Docker.
    VLM_IDLE_MIN: Простой до остановки, мин.
    VLM_START_TIMEOUT: Наибольшее время старта, с.
    VLM_GATE_PORT: Порт шлюза.
    DOCKER_SOCKET: Путь к сокету Docker.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTAINER = os.environ.get("VLM_CONTAINER", "npa-vlm")
UPSTREAM = os.environ.get("VLM_UPSTREAM", "http://vlm:8000").rstrip("/")
IDLE_SECONDS = float(os.environ.get("VLM_IDLE_MIN", "10")) * 60
START_TIMEOUT = float(os.environ.get("VLM_START_TIMEOUT", "600"))
PORT = int(os.environ.get("VLM_GATE_PORT", "8000"))
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
# распознавание страницы — до минуты; запас на очередь из нескольких страниц
REQUEST_TIMEOUT = 900.0


def _log(msg: str) -> None:
    """Пишет сообщение со временем в stderr."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


class _UnixConnection(http.client.HTTPConnection):
    """HTTP-соединение через Unix-сокет."""

    def __init__(self, path: str, timeout: float = 30.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class Docker:
    """Нужные шлюзу вызовы Docker Engine API."""

    def __init__(self, socket_path: str = DOCKER_SOCKET) -> None:
        self.socket_path = socket_path

    def _call(self, method: str, path: str, timeout: float = 30.0) -> tuple[int, bytes]:
        """Выполняет запрос к Docker API и возвращает код и тело ответа."""
        conn = _UnixConnection(self.socket_path, timeout=timeout)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def state(self, name: str) -> dict:
        """Возвращает раздел ``State`` контейнера.

        Raises:
            RuntimeError: Контейнер не найден.
        """
        status, body = self._call("GET", f"/containers/{name}/json")
        if status != 200:
            raise RuntimeError(f"контейнер {name} не найден (Docker API {status})")
        return json.loads(body)["State"]

    def start(self, name: str) -> None:
        """Запускает контейнер; уже запущенный не считается ошибкой.

        Raises:
            RuntimeError: Docker не смог запустить контейнер.
        """
        status, body = self._call("POST", f"/containers/{name}/start")
        if status not in (204, 304):  # 304 — уже запущен
            raise RuntimeError(f"не удалось запустить {name}: {status} {body[:200]!r}")

    def stop(self, name: str) -> None:
        """Останавливает контейнер с таймаутом 30 с."""
        self._call("POST", f"/containers/{name}/stop?t=30", timeout=60.0)


class Gate:
    """Запуск vLLM по требованию и остановка после простоя.

    Args:
        docker: Клиент Docker API.
        container: Имя контейнера vLLM.
        upstream: Адрес vLLM.
        idle_seconds: Простой до остановки, с.
        start_timeout: Наибольшее время старта, с.
        health_fn: Проверка готовности vLLM; по умолчанию запрос ``/health``.
        clock: Источник монотонного времени.
        sleep: Функция ожидания.
    """

    def __init__(self, docker: Docker, container: str = CONTAINER, upstream: str = UPSTREAM,
                 idle_seconds: float = IDLE_SECONDS, start_timeout: float = START_TIMEOUT,
                 health_fn=None, clock=time.monotonic, sleep=time.sleep) -> None:
        self.docker = docker
        self.container = container
        self.upstream = upstream
        self.idle_seconds = idle_seconds
        self.start_timeout = start_timeout
        self.health_fn = health_fn or self._health
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()        # счётчики
        self.start_lock = threading.Lock()  # один старт на всех ждущих
        self.in_flight = 0
        self.last_activity = clock()

    def _health(self) -> bool:
        """Проверяет, что vLLM отвечает на ``/health``."""
        try:
            with urllib.request.urlopen(f"{self.upstream}/health", timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def ensure_up(self) -> None:
        """Запускает vLLM, если нужно, и ждёт готовности.

        Одновременные вызовы ждут один и тот же запуск.

        Raises:
            RuntimeError: Контейнер остановился при старте или не ответил за
                ``start_timeout``.
        """
        if self.health_fn():
            return
        with self.start_lock:
            if self.health_fn():
                return
            st = self.docker.state(self.container)
            if not st.get("Running"):
                _log(f"запуск {self.container}")
                self.docker.start(self.container)
            t0 = self.clock()
            while self.clock() - t0 < self.start_timeout:
                if self.health_fn():
                    _log(f"{self.container} готов за {self.clock() - t0:.0f} с")
                    return
                st = self.docker.state(self.container)
                if not st.get("Running"):
                    raise RuntimeError(f"{self.container} остановился при старте (код {st.get('ExitCode')}): "
                                       "не хватило памяти GPU? см. docker logs")
                self.sleep(3)
            raise RuntimeError(f"{self.container} не ответил за {self.start_timeout:.0f} с")

    def begin(self) -> None:
        """Отмечает начало запроса."""
        with self.lock:
            self.in_flight += 1
            self.last_activity = self.clock()

    def end(self) -> None:
        """Отмечает конец запроса."""
        with self.lock:
            self.in_flight -= 1
            self.last_activity = self.clock()

    def idle_check(self) -> bool:
        """Останавливает vLLM, если он простаивает.

        Returns:
            True, если контейнер остановлен этим вызовом.
        """
        with self.lock:
            idle = self.in_flight == 0 and self.clock() - self.last_activity >= self.idle_seconds
        if not idle:
            return False
        with self.start_lock:  # не останавливать посреди старта
            try:
                if not self.docker.state(self.container).get("Running"):
                    return False
            except RuntimeError:
                return False
            with self.lock:
                if self.in_flight:
                    return False
            _log(f"{self.idle_seconds / 60:.0f} мин без запросов — остановка {self.container}")
            self.docker.stop(self.container)
            return True


def _handler(gate: Gate):
    """Создаёт класс обработчика HTTP, передающего запросы в vLLM через ``gate``."""

    class Handler(BaseHTTPRequestHandler):
        """Передаёт запросы GET и POST в vLLM; ``/gate/health`` отвечает сам шлюз."""

        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            """Отправляет ответ с телом."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, method: str) -> None:
            """Передаёт запрос в vLLM, при необходимости запустив его."""
            if self.path == "/gate/health":  # сам шлюз; vLLM при этом не будится
                self._send(200, json.dumps({"ok": True}).encode())
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
            gate.begin()
            try:
                try:
                    gate.ensure_up()
                except RuntimeError as exc:
                    _log(f"vLLM недоступен: {exc}")
                    self._send(503, json.dumps({"error": str(exc)}, ensure_ascii=False).encode())
                    return
                req = urllib.request.Request(gate.upstream + self.path, data=body, method=method,
                                             headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
                try:
                    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                        self._send(resp.status, resp.read(), resp.headers.get("Content-Type", "application/json"))
                except urllib.error.HTTPError as exc:
                    self._send(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))
                except (urllib.error.URLError, OSError) as exc:
                    self._send(502, json.dumps({"error": f"vLLM: {exc}"}, ensure_ascii=False).encode())
            finally:
                gate.end()

        def do_GET(self) -> None:  # noqa: N802
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._proxy("POST")

        def log_message(self, fmt: str, *args) -> None:
            _log(f"{self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    """Запускает шлюз и фоновую проверку простоя."""
    gate = Gate(Docker())
    _log(f"шлюз vLLM: {gate.container} ({gate.upstream}), остановка после {gate.idle_seconds / 60:.0f} мин простоя")

    def watch() -> None:
        """Раз в 30 с проверяет простой vLLM."""
        while True:
            time.sleep(30)
            try:
                gate.idle_check()
            except Exception as exc:  # noqa: BLE001 — сторож не должен умирать
                _log(f"проверка простоя: {exc}")

    threading.Thread(target=watch, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), _handler(gate)).serve_forever()


if __name__ == "__main__":
    main()
