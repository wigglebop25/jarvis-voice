# Implementation Plan: Implement robust synchronization API for the Transcriber

## Phase 1: Rust Internal Synchronization [checkpoint: 3d21317]
- [x] Task: Add synchronization primitives to `Transcriber` and `TranscriptionWorker`. [x] 3d21317
    - [x] **Red Phase**: Write unit tests in Rust for internal signaling (mocking transcription).
    - [x] **Green Phase**: Update structs in `src/transcriber.rs` to include `Arc<(Mutex<bool>, Condvar)>`.
- [x] Task: Implement notification logic in the transcription worker. [x] 3d21317
    - [x] **Red Phase**: Write tests to verify that the `Condvar` is notified when transcription stops.
    - [x] **Green Phase**: Update `stop_capture()` in `src/transcriber.rs` to notify the `Condvar`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Rust Internal Synchronization' (Protocol in workflow.md)

## Phase 2: Rust API Implementation [checkpoint: be501f1]
- [x] Task: Implement the `wait_until_done` blocking API. [x] be501f1
    - [x] **Red Phase**: Write Python tests that call `wait_until_done` and expect it to block/timeout.
    - [x] **Green Phase**: Implement `wait_until_done` in `src/transcriber.rs` using `cvar.wait_timeout`. Raise `TimeoutError` on timeout.
- [x] Task: Implement the `register_on_complete` callback API. [x] be501f1
    - [x] **Red Phase**: Write Python tests that register a callback and expect it to be called with the transcript.
    - [x] **Green Phase**: Implement `register_on_complete` in `src/transcriber.rs`, handling the GIL and Python callback invocation.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Rust API Implementation' (Protocol in workflow.md)

## Phase 3: Python Integration and Refactoring [checkpoint: 17ed312]
- [x] Task: Refactor `jarvis_voice/listener.py` to use the new API. [x] 17ed312
    - [x] **Red Phase**: Update existing listener tests to fail when they expect the old polling behavior (or mock the new API).
    - [x] **Green Phase**: Replace the `while ... time.sleep(0.1)` loop in `listener.py` with `self.transcriber.wait_until_done()`.
- [x] Task: Final verification of the end-to-end flow. [x] 17ed312
    - [x] **Red Phase**: Write an integration test that simulates a full voice-to-transcript cycle without polling.
    - [x] **Green Phase**: Ensure the integration test passes with the refactored listener.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Python Integration and Refactoring' (Protocol in workflow.md)
