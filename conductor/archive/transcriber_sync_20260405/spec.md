# Track Specification: Implement robust synchronization API for the Transcriber

## Overview
Currently, the `listener.py` in Python busy-polls the `Transcriber` to check if transcription has finished (`while self.transcriber.is_transcribing(): time.sleep(.1)`). This is inefficient, adds latency, and is not a robust design for a library. This track will implement proper synchronization primitives in the Rust `Transcriber` to allow Python callers to block until transcription is done or receive an event-driven callback.

## Functional Requirements
- **Blocking Wait API**: Implement a `wait_until_done(timeout: Option<f64>)` method in the Rust `Transcriber` exposed via PyO3.
    - If no timeout is provided, it should block indefinitely until transcription completes.
    - If a timeout is provided and reached, it should raise a Python `TimeoutError`.
    - If the transcription worker encounters an internal error, it should be propagated as a Python exception.
- **Callback Registration API**: Implement a `register_on_complete(callback: PyObject)` method to allow users to register a Python function that will be called when transcription finishes.
    - The callback should receive the final transcript as an argument.
    - It must handle the Python GIL correctly within the Rust worker thread.
- **Internal Synchronization**: Replace atomic polling flags with condition variables (`Condvar`) and mutexes for thread signaling.
- **Listener Update**: Refactor `jarvis_voice/listener.py` to use the new `wait_until_done()` method instead of busy-polling with `time.sleep()`.

## Non-Functional Requirements
- **Efficiency**: No busy-waiting; the CPU should not be active while waiting for transcription.
- **Responsiveness**: Minimal latency between transcription completion and signaling.
- **Safety**: Ensure proper thread safety and avoid deadlocks or GIL-related crashes.

## Acceptance Criteria
- [ ] `wait_until_done()` successfully blocks until transcription finishes.
- [ ] `wait_until_done(timeout=...)` raises `TimeoutError` if transcription exceeds the timeout.
- [ ] `register_on_complete(cb)` correctly triggers the callback after transcription.
- [ ] `listener.py` no longer contains a `while ...: time.sleep()` polling loop.
- [ ] All new APIs are covered by unit or integration tests (Rust and Python).

## Out of Scope
- Implementing full `asyncio` support in the Rust layer (this can be a separate track).
- Major refactoring of the transcription logic itself.
