# Specification: Implement and test the core voice processing loop with VAD

## Objective
Implement a robust, high-performance voice processing loop that integrates wake word detection, Voice Activity Detection (VAD), and real-time transcription. The loop should handle audio input, detect the "Jarvis" wake word, identify when the user is speaking, and transcribe the speech into text.

## Scope
- Integrate `pvporcupine` for wake word detection.
- Integrate a VAD mechanism (e.g., Silero VAD or similar) to detect speech boundaries.
- Integrate `transcribe-rs` with Parakeet for real-time transcription.
- Implement a state machine to manage transitions between "Listening for Wake Word", "Recording Speech", and "Transcribing".
- Ensure low latency and efficient resource usage.

## Requirements
- **Wake Word Detection:** Detect "Jarvis" with high accuracy and low latency.
- **Voice Activity Detection:** Accurately identify the start and end of user speech.
- **Real-time Transcription:** Convert speech to text as it is being recorded.
- **Concurrency:** Use `tokio` for non-blocking audio processing and transcription.
- **Error Handling:** Gracefully handle audio device errors and network issues (for remote transcription).

## Technical Design
- **Audio Stream:** Use `cpal` to capture audio from the default input device.
- **Buffer Management:** Efficiently manage audio buffers to pass to Porcupine, VAD, and Parakeet.
- **State Machine:**
    - `IDLE`: Listening for wake word.
    - `LISTENING`: Wake word detected, recording audio for VAD/transcription.
    - `PROCESSING`: Finalizing transcription after VAD detects end-of-speech.
- **Rust-Python Integration:** Expose the loop and its events (wake word detected, transcription result) to Python via PyO3.
