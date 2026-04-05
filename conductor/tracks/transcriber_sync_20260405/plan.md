# Implementation Plan: Implement robust synchronization API for the Transcriber

## Phase 1: Rust Internal Synchronization
- [x] Task: Add synchronization primitives to `Transcriber` and `TranscriptionWorker`.
    - [x] **Red Phase**: Write unit tests in Rust for internal signaling (mocking transcription).
    - [x] **Green Phase**: Update structs in `src/transcriber.rs` to include `Arc<(Mutex<bool>, Condvar)>`.
- [x] Task: Implement notification logic in the transcription worker.
    - [x] **Red Phase**: Write tests to verify that the `Condvar` is notified when transcription stops.
    - [x] **Green Phase**: Update `stop_capture()` in `src/transcriber.rs` to notify the `Condvar`.
- [~] Task: Conductor - User Manual Verification 'Phase 1: Rust Internal Synchronization' (Protocol in workflow.md)

## Phase 2: Rust API Implementation
- [ ] Task: Implement the `wait_until_done` blocking API.
    - [ ] **Red Phase**: Write Python tests that call `wait_until_done` and expect it to block/timeout.
    - [ ] **Green Phase**: Implement `wait_until_done` in `src/transcriber.rs` using `cvar.wait_timeout`. Raise `TimeoutError` on timeout.
- [ ] Task: Implement the `register_on_complete` callback API.
    - [ ] **Red Phase**: Write Python tests that register a callback and expect it to be called with the transcript.
    - [ ] **Green Phase**: Implement `register_on_complete` in `src/transcriber.rs`, handling the GIL and Python callback invocation.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Rust API Implementation' (Protocol in workflow.md)

## Phase 3: Python Integration and Refactoring
- [ ] Task: Refactor `jarvis_voice/listener.py` to use the new API.
    - [ ] **Red Phase**: Update existing listener tests to fail when they expect the old polling behavior (or mock the new API).
    - [ ] **Green Phase**: Replace the `while ... time.sleep(0.1)` loop in `listener.py` with `self.transcriber.wait_until_done()`.
- [ ] Task: Final verification of the end-to-end flow.
    - [ ] **Red Phase**: Write an integration test that simulates a full voice-to-transcript cycle without polling.
    - [ ] **Green Phase**: Ensure the integration test passes with the refactored listener.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Python Integration and Refactoring' (Protocol in workflow.md)
