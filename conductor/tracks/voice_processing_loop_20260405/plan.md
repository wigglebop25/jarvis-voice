# Implementation Plan: Implement and test the core voice processing loop with VAD

## Phase 1: Foundation and Audio Input
- [ ] Task: Set up the core Rust structure for the voice loop.
    - [ ] Define the `VoiceLoop` struct and its state machine.
    - [ ] Implement audio capture using `cpal`.
    - [ ] Add unit tests for audio capture and buffer management.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Foundation and Audio Input' (Protocol in workflow.md)

## Phase 2: Wake Word Detection Integration
- [ ] Task: Integrate `pvporcupine` into the voice loop.
    - [ ] Write tests for Porcupine integration (mocking audio).
    - [ ] Implement wake word detection in the `IDLE` state.
    - [ ] Verify wake word detection with real/recorded audio.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Wake Word Detection Integration' (Protocol in workflow.md)

## Phase 3: VAD and Transcription Integration
- [ ] Task: Integrate VAD and `transcribe-rs`.
    - [ ] Write tests for VAD logic and transcription integration.
    - [ ] Implement speech boundary detection using VAD.
    - [ ] Implement real-time transcription in the `LISTENING` state.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: VAD and Transcription Integration' (Protocol in workflow.md)

## Phase 4: Python Bindings and Final Integration
- [ ] Task: Expose the voice loop to Python.
    - [ ] Implement PyO3 bindings for `VoiceLoop`.
    - [ ] Add Python-level tests using `pytest`.
    - [ ] Create an example Python script demonstrating the full loop.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Python Bindings and Final Integration' (Protocol in workflow.md)
