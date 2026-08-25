from __future__ import annotations

from contextlib import asynccontextmanager

import pytest


@pytest.fixture(autouse=True)
def _disable_cursor_passthrough_by_default(monkeypatch, request):
    if "cursor_present" in request.fixturenames:
        return

    def _off(**_kwargs):
        return False

    for target in (
        "codex_shim.cursor_passthrough.cursor_passthrough_available",
        "codex_shim.server.cursor_passthrough_available",
        "codex_shim.catalog.cursor_passthrough_available",
        "codex_shim.cli.cursor_passthrough_available",
    ):
        monkeypatch.setattr(target, _off, raising=False)

@pytest.fixture(autouse=True)
def _isolate_local_runtime_from_real_systemd(monkeypatch):
    @asynccontextmanager
    async def _noop_runtime_request(_slug: str):
        yield None

    async def _forbid_real_systemctl(*_args: str):
        raise AssertionError(
            "pytest attempted to call real local-runtime systemctl"
        )

    # server.py imports ensure_local_runtime directly, so patch the symbol
    # actually awaited by ShimServer request handlers.
    monkeypatch.setattr(
        "codex_shim.server.local_runtime_request",
        _noop_runtime_request,
    )

    # Defense in depth: if a test reaches local_runtime.py directly without
    # explicitly mocking its systemctl boundary, fail instead of touching
    # the users real model services.
    monkeypatch.setattr(
        "codex_shim.local_runtime._systemctl",
        _forbid_real_systemctl,
    )
