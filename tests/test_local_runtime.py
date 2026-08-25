from __future__ import annotations

from types import SimpleNamespace

import pytest

import codex_shim.local_runtime as runtime


@pytest.mark.asyncio
async def test_unknown_slug_is_noop():
    assert await runtime.ensure_local_runtime("not-a-local-model") is None


@pytest.mark.asyncio
async def test_active_runtime_fast_path_does_not_touch_systemd(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: {
            "id": "ornith",
            "n_params": 35505251456,
            "n_ctx": 131072,
        },
    )

    result = await runtime.ensure_local_runtime("ornith")

    assert result == "ornith"


@pytest.mark.asyncio
async def test_none_to_9b_stops_both_then_starts_9b(monkeypatch):
    calls = []

    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: None,
    )

    async def fake_systemctl(*args: str):
        calls.append(args)
        return 0, ""

    async def fake_wait_ready(expected_params: int):
        assert expected_params == 8953803264
        return {
            "id": "slot1-ornith-1.5-9b",
            "n_params": 8953803264,
            "n_ctx": 32768,
        }

    monkeypatch.setattr(runtime, "_systemctl", fake_systemctl)
    monkeypatch.setattr(runtime, "_wait_ready", fake_wait_ready)

    result = await runtime.ensure_local_runtime(
        "ornith-1-5-9b"
    )

    assert result == "slot1-ornith-1.5-9b"

    assert calls == [
        ("stop", "ornith-llama.service"),
        ("stop", "ornith-9b.service"),
        ("start", "ornith-9b.service"),
    ]


@pytest.mark.asyncio
async def test_9b_to_35b_stops_both_then_starts_35b(monkeypatch):
    calls = []

    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: {
            "id": "slot1-ornith-1.5-9b",
            "n_params": 8953803264,
            "n_ctx": 32768,
        },
    )

    async def fake_systemctl(*args: str):
        calls.append(args)
        return 0, ""

    async def fake_wait_ready(expected_params: int):
        assert expected_params == 35505251456
        return {
            "id": "ornith",
            "n_params": 35505251456,
            "n_ctx": 131072,
        }

    monkeypatch.setattr(runtime, "_systemctl", fake_systemctl)
    monkeypatch.setattr(runtime, "_wait_ready", fake_wait_ready)

    result = await runtime.ensure_local_runtime("ornith")

    assert result == "ornith"

    assert calls == [
        ("stop", "ornith-llama.service"),
        ("stop", "ornith-9b.service"),
        ("start", "ornith-llama.service"),
    ]


@pytest.mark.asyncio
async def test_start_failure_raises_local_runtime_error(monkeypatch):
    calls = []

    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: None,
    )

    async def fake_systemctl(*args: str):
        calls.append(args)

        if args == ("start", "ornith-9b.service"):
            return 1, "simulated start failure"

        return 0, ""

    async def forbidden_wait_ready(_expected_params: int):
        raise AssertionError(
            "_wait_ready must not run after start failure"
        )

    monkeypatch.setattr(runtime, "_systemctl", fake_systemctl)
    monkeypatch.setattr(
        runtime,
        "_wait_ready",
        forbidden_wait_ready,
    )

    with pytest.raises(
        runtime.LocalRuntimeError,
        match="Failed starting ornith-9b.service",
    ):
        await runtime.ensure_local_runtime(
            "ornith-1-5-9b"
        )

    assert calls == [
        ("stop", "ornith-llama.service"),
        ("stop", "ornith-9b.service"),
        ("start", "ornith-9b.service"),
    ]


@pytest.mark.asyncio
async def test_wait_ready_accepts_matching_runtime(monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: {
            "id": "ornith",
            "n_params": 35505251456,
            "n_ctx": 131072,
        },
    )

    info = await runtime._wait_ready(
        35505251456,
        timeout=0.1,
    )

    assert info == {
        "id": "ornith",
        "n_params": 35505251456,
        "n_ctx": 131072,
    }


@pytest.mark.asyncio
async def test_wait_ready_rejects_identity_mismatch(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.values = iter(
                (
                    0.0,
                    0.1,
                    1.1,
                )
            )

        def monotonic(self):
            return next(self.values)

    async def fake_to_thread(func, *args):
        return func(*args)

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(
        runtime,
        "_runtime_info",
        lambda: {
            "id": "wrong-model",
            "n_params": 123,
            "n_ctx": 4096,
        },
    )

    monkeypatch.setattr(
        runtime,
        "time",
        FakeClock(),
    )

    monkeypatch.setattr(
        runtime,
        "asyncio",
        SimpleNamespace(
            to_thread=fake_to_thread,
            sleep=fake_sleep,
        ),
    )

    with pytest.raises(
        runtime.LocalRuntimeError,
        match="Timed out waiting for requested llama-server",
    ):
        await runtime._wait_ready(
            35505251456,
            timeout=1.0,
        )
