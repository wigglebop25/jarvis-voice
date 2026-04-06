# Implementation Plan: Refactor Rust Logic for Maintainability

## Phase 1: Preparation & Error Handling Standardisation [checkpoint: d02f29e]
- [x] Task: Add `anyhow` dependency to `Cargo.toml`. d20c464
- [x] Task: Define a central `Error` type (if needed) or start using `anyhow::Result` in `src/lib.rs`. d20c464
- [x] Task: Refactor `src/config.rs` to use `anyhow::Result` for error handling. f86ca8d
- [x] Task: Conductor - User Manual Verification 'Phase 1: Preparation & Error Handling Standardisation' (Protocol in workflow.md) d02f29e

## Phase 2: Core & Utils Extraction [checkpoint: eed406b]
- [x] Task: Create `src/core/mod.rs` and `src/core/config.rs`. 3a48284
- [x] Task: Move configuration logic from `src/config.rs` to `src/core/config.rs`. 4982cc3
- [x] Task: Create `src/utils/mod.rs` and move any shared utility functions. c69b146
- [x] Task: Update `src/lib.rs` to export the new `core` and `utils` modules. 8d6715f
- [x] Task: Conductor - User Manual Verification 'Phase 2: Core & Utils Extraction' (Protocol in workflow.md) eed406b

## Phase 3: Audio Processing Extraction [checkpoint: 815fee6]
- [x] Task: Create `src/audio/mod.rs`, `src/audio/input.rs`, and `src/audio/resampler.rs`. bdb3657
- [x] Task: Move `cpal` related logic to `src/audio/input.rs`. 1d5129c
- [x] Task: Move `rubato` related logic from `src/resampler.rs` to `src/audio/resampler.rs`. b690aee
- [x] Task: Refactor `src/audio` modules to use `anyhow` for error handling. 8c4eb29
- [x] Task: Conductor - User Manual Verification 'Phase 3: Audio Processing Extraction' (Protocol in workflow.md) 815fee6

## Phase 4: Transcription Logic Isolation
- [ ] Task: Create `src/transcription/mod.rs` and `src/transcription/engine.rs`.
- [ ] Task: Move logic from `src/transcriber.rs` and `src/model.rs` into the `src/transcription` module.
- [ ] Task: Standardize transcription error handling with `anyhow`.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Transcription Logic Isolation' (Protocol in workflow.md)

## Phase 5: PyO3 Bindings Separation
- [ ] Task: Create `src/python/mod.rs` or `src/ffi/mod.rs` for PyO3 bindings.
- [ ] Task: Move `#[pyfunction]` and `#[pymodule]` definitions from `src/lib.rs` to the new module.
- [ ] Task: Ensure the Rust-Python interface correctly handles `anyhow` errors (converting to `PyResult`).
- [ ] Task: Conductor - User Manual Verification 'Phase 5: PyO3 Bindings Separation' (Protocol in workflow.md)

## Phase 6: Test Reorganization & Final Cleanup
- [ ] Task: Create a `tests/` directory (if not already present for Rust integration tests) or a `src/tests/` module.
- [ ] Task: Move unit tests from `src/*.rs` files into a dedicated `tests` structure.
- [ ] Task: Move `src/test_sync.rs` and any other test-only files to the new test structure.
- [ ] Task: Perform final code cleanup, removing unused imports and ensuring consistent formatting.
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Test Reorganization & Final Cleanup' (Protocol in workflow.md)
