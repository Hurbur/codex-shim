from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from urllib.request import urlopen


PROFILES = {
    "ornith": {
        "service": "ornith-llama.service",
        "alias": "ornith",
        "n_params": 35505251456,
    },
    "ornith-1-5-9b": {
        "service": "ornith-9b.service",
        "alias": "slot1-ornith-1.5-9b",
        "n_params": 8953803264,
    },
}

SERVICES = tuple(
    profile["service"]
    for profile in PROFILES.values()
)

_switch_lock = asyncio.Lock()
_request_lock = asyncio.Lock()


class LocalRuntimeError(RuntimeError):
    pass


def _runtime_info() -> dict | None:
    try:
        with urlopen(
            "http://127.0.0.1:8080/v1/models",
            timeout=2,
        ) as response:
            payload = json.load(response)
    except Exception:
        return None

    rows = payload.get("data") or []

    if not rows:
        return None

    row = rows[0]
    meta = row.get("meta") or {}

    return {
        "id": str(row.get("id") or ""),
        "n_params": int(meta.get("n_params") or 0),
        "n_ctx": int(meta.get("n_ctx") or 0),
    }


async def _systemctl(*args: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "systemctl",
        "--user",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    stdout, _ = await process.communicate()

    return (
        process.returncode,
        stdout.decode(errors="replace").strip(),
    )


async def _wait_ready(
    expected_params: int,
    timeout: float = 180.0,
) -> dict:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        info = await asyncio.to_thread(_runtime_info)

        if info and info["n_params"] == expected_params:
            return info

        await asyncio.sleep(0.5)

    raise LocalRuntimeError(
        "Timed out waiting for requested llama-server "
        f"n_params={expected_params}"
    )


@asynccontextmanager
async def local_runtime_request(slug: str):
    if slug not in PROFILES:
        yield None
        return

    async with _request_lock:
        runtime_alias = await ensure_local_runtime(slug)
        yield runtime_alias


async def ensure_local_runtime(slug: str) -> str | None:
    profile = PROFILES.get(slug)

    if profile is None:
        return None

    async with _switch_lock:
        current = await asyncio.to_thread(_runtime_info)

        if current and current["n_params"] == profile["n_params"]:
            return profile["alias"]

        started = time.monotonic()

        if current:
            old_desc = (
                f"{current["id"]} "
                f"params={current["n_params"]} "
                f"ctx={current["n_ctx"]}"
            )
        else:
            old_desc = "none"

        print(
            f"[local-runtime] switch requested "
            f"from={old_desc} to={slug}",
            flush=True,
        )

        for service in SERVICES:
            code, output = await _systemctl(
                "stop",
                service,
            )

            if code != 0:
                raise LocalRuntimeError(
                    f"Failed stopping {service}: {output}"
                )

        code, output = await _systemctl(
            "start",
            profile["service"],
        )

        if code != 0:
            raise LocalRuntimeError(
                f"Failed starting "
                f"{profile["service"]}: {output}"
            )

        info = await _wait_ready(
            profile["n_params"]
        )

        elapsed = time.monotonic() - started

        print(
            f"[local-runtime] ready "
            f"slug={slug} "
            f"model={info["id"]} "
            f"params={info["n_params"]} "
            f"ctx={info["n_ctx"]} "
            f"swap_seconds={elapsed:.2f}",
            flush=True,
        )

        return profile["alias"]
