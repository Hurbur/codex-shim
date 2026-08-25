# Local Runtime Switching Bootstrap Findings

Date: 2026-08-25

Status: validated bootstrap checkpoint

## Scope

This work is part of the local-model bootstrap research environment.

It is not the final ValKhana runtime architecture.

The purpose of this checkpoint was to determine whether Work/Codex can use multiple local llama.cpp models through one shim while only one model occupies GPU memory at a time.

## Tested Environment

- CachyOS
- NVIDIA RTX 5080 Laptop GPU with 16 GB VRAM
- 64 GB system RAM
- llama.cpp server on `127.0.0.1:8080`
- Codex shim on `127.0.0.1:8765`
- fish shell
- systemd user services

## Local Models

### Ornith 1.0 35B

Runtime slug: `ornith`

Model:

`/mnt/linux-data/AI/models/ornith/Ornith-1.0-35B-MTP-APEX-I-Compact.gguf`

Observed runtime:

- 35,505,251,456 parameters
- 131072 context
- Q4_K Medium
- approximately 15.5 GiB VRAM

Service:

`ornith-llama.service`

### Ornith 1.5 9B Q8

Work/Codex slug:

`ornith-1-5-9b`

Runtime alias:

`slot1-ornith-1.5-9b`

Model:

`/mnt/linux-data/AI/models/valkhana-fleet/slot1/ornith-1.5-9b/Ornith-1.5-9B-Q8_0.gguf`

Observed runtime:

- 8,953,803,264 parameters
- 32768 context
- Q8_0
- approximately 9.1 GiB VRAM

Service:

`ornith-9b.service`

## Runtime Switching Design

The shim now contains `codex_shim/local_runtime.py`.

For known local model slugs, `ensure_local_runtime`:

1. Checks the llama.cpp model currently exposed on port 8080.
2. Uses parameter count to identify the currently loaded model.
3. Returns immediately when the requested model is already resident.
4. Serializes model switching with one process-wide asyncio lock.
5. Stops both known local model services before a transition.
6. Starts the service for the requested model.
7. Waits until the expected model identity appears on port 8080.
8. Returns the llama.cpp runtime alias.
9. Allows the original request to continue after the model becomes ready.

The original inference request is held during the transition. The user does not need to resend it.

## Proven Runtime Matrix

| Transition | Result |
| --- | --- |
| 35B to 9B | PASS |
| 9B to 35B | PASS |
| 35B to 35B | PASS, fast path with no restart |
| zero models to 9B | PASS |
| zero models to 35B | PASS |
| real Work/Codex zero models to 35B | PASS |
| real Work/Codex 35B to 9B | PASS |

## Observed Switching Measurements

### Direct shim cold start to 9B

- Model-load and verification time: 2.53 seconds
- End-to-end request time: 3.17 seconds
- Final response completed from the same request
- Observed VRAM: approximately 9084 MiB

### Direct shim cold start to 35B

- Model-load and verification time: 3.06 seconds
- End-to-end request time: 6.05 seconds
- Final response completed from the same request
- Observed VRAM: approximately 15566 MiB

### Real Work/Codex zero models to 35B

- Observed model transition time: 3.77 seconds
- Same Work/Codex request continued after the model became ready

### Real Work/Codex 35B to 9B

- Observed model transition time: 3.04 seconds
- Final state contained only the 9B model

Other observed swaps during testing were generally in the approximate 2.5 to 4.5 second range.

These are bootstrap measurements, not formal performance guarantees.

## Work/Codex Selection Behavior

Changing the visible model selection in Work/Codex does not itself trigger a runtime transition.

The selected model slug is sent on the next actual inference request.

Therefore the observed behavior is:

`select model -> no load -> send request -> requested model loads`

This allows model selection without immediately consuming VRAM.

## Zero-Model Startup Policy

The persistent startup policy was changed to:

- `codex-shim.service`: enabled
- `ornith-llama.service`: disabled
- `ornith-9b.service`: static
- no llama.cpp backend required for the shim to start

The old shim backend dependency and health precheck were removed.

The shim can therefore remain active while port 8080 is empty.

Expected idle flow:

`login -> shim active -> zero local models -> zero local-model VRAM -> first inference request -> requested model starts`

A real logout or reboot verification remains a future validation step.

## Ornith Message Ordering Compatibility Finding

A real accumulated Work/Codex conversation failed on Ornith 1.0 with:

`Jinja Exception: System message must be at the beginning.`

Inspection of the actual forwarded request showed late developer instructions normalized into system messages after user and assistant history.

The forwarded pattern included:

`system -> user -> assistant -> system -> user -> assistant -> system -> user`

The existing provider-specific system-message hoist was enabled for Ornith as well as Qwen.

After the fix, the real Work/Codex request was forwarded as:

`system -> user -> assistant -> user -> assistant -> user -> assistant -> user`

Only one system position remained at index zero.

The real Work/Codex conversation then completed without the Jinja ordering error.

DeepSeek behavior was deliberately left unchanged.

## Reasoning Output Finding

A small 64-token output cap was insufficient for one 35B exact-output experiment because the model consumed the available output budget on reasoning before producing the requested final text.

This was not a runtime switching failure.

Later validation used larger output budgets.

## Pytest Isolation Failure Discovered

After runtime switching was connected to `ShimServer`, the existing test suite could indirectly call the real runtime switcher.

A full test run passed but unexpectedly started the real 35B systemd service.

This was a test-isolation failure.

The fix added an autouse pytest fixture that:

- replaces the server-level runtime switch with a no-op during ordinary tests
- replaces the real `_systemctl` boundary with a function that raises if a test reaches it unexpectedly

Dedicated mocked runtime tests were then added in:

`tests/test_local_runtime.py`

These tests exercise the switching behavior without touching real systemd services.

## Current Test Result

Dedicated runtime tests:

`7 passed`

Complete shim test suite:

`221 passed`

After the complete suite:

- 35B remained inactive
- 9B remained inactive
- port 8080 remained empty
- no real model-start events occurred

## Dedicated Runtime Coverage

Current mocked tests cover:

- unknown model slug no-op
- already-active fast path
- zero models to 9B
- 9B to 35B
- service start failure
- matching runtime readiness
- runtime identity mismatch timeout

## Files Added or Modified

Repository changes:

- `codex_shim/server.py`
- `codex_shim/translate.py`
- `codex_shim/local_runtime.py`
- `tests/conftest.py`
- `tests/test_translate.py`
- `tests/test_local_runtime.py`

Systemd configuration was also changed outside the repository.

## Important Bootstrap Lessons

1. On-demand model loading is viable on this hardware.
2. A local model does not need to occupy VRAM merely to keep the shim available.
3. A model transition can occur while holding the original inference request.
4. Work/Codex model selection is request-driven rather than selection-driven.
5. Provider chat-template constraints must remain provider-specific.
6. Integration tests must never be allowed to reach real machine-control boundaries by default.
7. Runtime identity must be verified after a service starts rather than assuming that a successful systemd start means the correct model is serving.
8. Real application-path tests revealed issues that direct curl tests alone did not reveal.

## Current Bootstrap Limitations

The current implementation is intentionally small and experimental.

Known limitations include:

- the switch lock protects one shim process only
- model identity currently relies primarily on parameter count
- every transition stops all known local model services before starting the target
- readiness timeout is fixed at 180 seconds
- no idle-time automatic unload policy is implemented
- persistent zero-model behavior has not yet been validated across a fresh logout or reboot
- recovery behavior for simultaneous requests during failed transitions needs separate stress testing
- Work/Codex retry behavior can produce repeated requests after upstream failures
- this implementation should not yet be treated as final ValKhana runtime architecture

## Freeze Decision

The switching behavior should remain unchanged at this checkpoint.

The next bootstrap work should use this implementation as evidence rather than immediately adding more switcher features.

Future validation can include:

- fresh login or reboot zero-model verification
- concurrent request behavior
- controlled failed-load recovery
- additional model slots
- stronger runtime identity metadata
- eventual translation of the findings into ValKhana design requirements

## Cross-model in-flight concurrency

### Destructive-switch failure discovered

A controlled overlap test exposed a real failure in the original switcher.

Test sequence:

- Ornith 1.5 9B was actively serving a long inference.
- A request for Ornith 1.0 35B arrived while the 9B request was still in flight.
- The original `_switch_lock` serialized the runtime transition itself, but did not protect the lifetime of the active inference.
- The switcher immediately requested that systemd stop `ornith-9b.service`.

Observed result:

- systemd sent the configured SIGINT.
- llama.cpp entered cleanup while an inference was still active.
- shutdown did not finish within `TimeoutStopSec=30`.
- systemd sent SIGKILL after 30 seconds.
- the active 9B request failed with HTTP 500.
- the shim traceback ended in `aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected`.
- `ornith-9b.service` ended in `Result=timeout`, `ExecMainStatus=9`, and failed state.
- the requested 35B transition eventually succeeded, but reported `swap_seconds=33.70`.

This proves that a switch lock alone does not provide inference ownership. A model must not be evicted while an active request is still using it.

### Exclusive local-runtime request lease

The bootstrap switcher was updated with an exclusive local-runtime request lease.

For local model routes:

1. acquire the local request lease;
2. ensure or switch to the requested runtime;
3. keep the lease for the entire upstream inference;
4. for streaming traffic, keep the lease until the stream helper returns and the upstream response is released;
5. release the lease only after the request is finished.

A request for another local runtime therefore waits instead of stopping the model underneath an active inference.

Non-local routes bypass this local-runtime lease.

This is intentionally conservative. The tested llama.cpp services currently use `-np 1`, so serializing local inference provides a safe bootstrap baseline. It is not a claim that this is the final ValKhana concurrency architecture.

### Mocked regression coverage

A dedicated concurrency regression test verifies that:

- request A acquires the local runtime lease;
- request B begins while A still owns the lease;
- request B cannot invoke `ensure_local_runtime()` while A is active;
- after A releases the lease, B proceeds.

Dedicated local-runtime suite after the change:

- 8 passed.

Full shim suite after the change:

- 222 passed.

The pytest isolation fixture continues to prevent the test suite from starting or stopping real local model services.

### Real non-streaming overlap validation

Real runtime test:

- Request A: Ornith 1.5 9B, non-streaming, long generation.
- Request B: Ornith 1.0 35B, submitted approximately 3 seconds after A began.

Observed:

- A remained active while B waited.
- B did not stop 9B while A owned the lease.
- A completed with HTTP 200 in approximately 15.09 seconds.
- only after A completed did the switch from 9B to 35B begin.
- 9B stopped cleanly.
- 35B became ready with `swap_seconds=3.81`.
- B completed with HTTP 200 in approximately 19.32 seconds.
- no SIGKILL, shutdown timeout, service failure, or upstream disconnect occurred.

### Real streaming overlap validation

Real streaming test:

- Request A: Ornith 1.5 9B with `stream=true`.
- Request B: Ornith 1.0 35B with `stream=true`, submitted approximately 3 seconds after A began.

Observed:

- A remained active while B waited.
- A completed with HTTP 200 in approximately 23.64 seconds.
- A emitted `response.completed`.
- the 9B llama.cpp slot released before the switch started.
- 9B then stopped cleanly.
- 35B became ready with `swap_seconds=3.28`.
- B completed with HTTP 200 in approximately 26.20 seconds.
- B emitted `response.completed` and returned the expected final text.
- no `ServerDisconnectedError`, SIGKILL, stop timeout, or failed service state occurred.

### Bootstrap conclusion

The exclusive request lease fixes the destructive cross-model in-flight eviction demonstrated by the original switcher.

Current proven behavior is:

- cold model loading works;
- same-model requests avoid unnecessary swaps;
- model-to-model switching works;
- zero-model idle startup works;
- active local inference is protected from cross-model eviction;
- waiting cross-model requests resume after the current inference completes;
- the protection holds for both non-streaming and streaming traffic.

Remaining limitations include:

- the lease is process-local;
- all current local model requests are serialized rather than supporting multiple concurrent inference slots;
- queue fairness and cancellation behavior have not yet been characterized;
- failed-transition recovery under queued demand has not yet been characterized;
- a fresh logout/reboot zero-model startup validation remains outstanding;
- this bootstrap mechanism is evidence for later ValKhana design, not the final ValKhana runtime architecture.
