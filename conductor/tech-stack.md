# Tech Stack - JARVIS Voice

## Core Languages and Runtimes
- **Rust (2024 Edition):** Used for performance-critical audio processing, wake word detection, and transcription logic.
- **Python (>=3.8):** Provides a high-level API and integration layer for the rest of the JARVIS system.

## Integration and Build
- **PyO3:** Enables seamless Rust-to-Python bindings, allowing Rust code to be called from Python.
- **Maturin:** The build backend for the project, simplifying the development and distribution of the Rust-Python hybrid.

## Audio and Voice Processing
- **Wake Word Detection:** `pvporcupine` for high-performance detection of the "Jarvis" wake word.
- **Transcription:** `transcribe-rs` with Parakeet integration for real-time speech-to-text processing.
- **Audio I/O (Rust):** `cpal` for low-level audio input and output across different platforms.
- **Audio I/O (Python):** `sounddevice` for microphone capture during wake word detection. Ships pre-built wheels with bundled PortAudio.
- **Resampling:** `rubato` for high-quality audio resampling.

## Error Handling and Communication
- **Error Handling:** `anyhow` for standardized and ergonomic error propagation in Rust.
- **Communication:** `crossbeam-channel` for efficient message passing between threads.

## Concurrency and Performance
- **Tokio:** An asynchronous runtime for Rust, used for handling multiple tasks and asynchronous operations concurrently.
- **Rust Primitives:** `Condvar` and `Mutex` for efficient thread synchronization and event signaling across the Rust-Python boundary.

## Data and Validation
- **Pydantic:** Used in Python for data validation and settings management.
