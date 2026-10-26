"""Шлюз vLLM (vlm_gate.py): подъём по запросу и остановка после простоя — на фейковом Docker
и фейковых часах, без контейнеров и сети."""

from __future__ import annotations

import pytest

from npa_ocr.vlm_gate import Gate


class FakeDocker:
    def __init__(self, running: bool = False, healthy_after: int = 2, dies: bool = False) -> None:
        self.running = running
        self.calls: list[str] = []
        self.healthy_after = healthy_after  # сколько проверок /health после старта до готовности
        self.checks = 0
        self.dies = dies

    def state(self, name: str) -> dict:
        return {"Running": self.running, "ExitCode": 1 if self.dies else 0}

    def start(self, name: str) -> None:
        self.calls.append("start")
        self.running = not self.dies
        self.checks = 0

    def stop(self, name: str) -> None:
        self.calls.append("stop")
        self.running = False

    def health(self) -> bool:
        if not self.running:
            return False
        self.checks += 1
        return self.checks > self.healthy_after


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _gate(docker: FakeDocker, clock: Clock, **kw) -> Gate:
    return Gate(docker, container="vlm", upstream="http://vlm:8000", idle_seconds=600, start_timeout=60,
                health_fn=docker.health, clock=clock, sleep=clock.sleep, **kw)


def test_starts_stopped_container_and_waits_for_health():
    docker, clock = FakeDocker(running=False), Clock()
    gate = _gate(docker, clock)
    gate.ensure_up()
    assert docker.calls == ["start"]
    assert docker.running


def test_does_not_start_running_container():
    docker, clock = FakeDocker(running=True, healthy_after=0), Clock()
    _gate(docker, clock).ensure_up()
    assert docker.calls == []


def test_container_dying_on_start_is_reported():
    """vLLM не хватило памяти GPU — контейнер вышел; npa-ocr получит 503 и возьмёт TATR."""
    docker, clock = FakeDocker(running=False, dies=True), Clock()
    with pytest.raises(RuntimeError, match="остановился при старте"):
        _gate(docker, clock).ensure_up()


def test_start_timeout():
    docker, clock = FakeDocker(running=False, healthy_after=10**6), Clock()
    with pytest.raises(RuntimeError, match="не ответил"):
        _gate(docker, clock).ensure_up()


def test_stops_only_after_idle_period_without_requests():
    docker, clock = FakeDocker(running=True), Clock()
    gate = _gate(docker, clock)
    gate.begin()
    gate.end()
    clock.t += 599
    assert not gate.idle_check()  # 10 минут ещё не прошло
    clock.t += 2
    assert gate.idle_check()
    assert docker.calls == ["stop"]


def test_never_stops_while_request_in_flight():
    docker, clock = FakeDocker(running=True), Clock()
    gate = _gate(docker, clock)
    gate.begin()
    clock.t += 3600  # долгая генерация
    assert not gate.idle_check()
    assert docker.calls == []


def test_idle_check_on_stopped_container_does_nothing():
    docker, clock = FakeDocker(running=False), Clock()
    gate = _gate(docker, clock)
    clock.t += 3600
    assert not gate.idle_check()
    assert docker.calls == []
