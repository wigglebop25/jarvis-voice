# 🎙️ JARVIS Voice

> **A high-performance voice processing engine combining wake-word detection and local speech-to-text transcription.**

---

## About

**JARVIS Voice** is the core auditory interface for the JARVIS Project. It provides a seamless, hands-free voice interaction system by integrating lightweight wake-word detection with a robust, locally-run speech-to-text backend. 

Built with **Python** for ease of use and **Rust** for performance, JARVIS Voice ensures low latency and high accuracy, allowing developers to build responsive voice-activated applications without relying on cloud APIs for transcription.

### Key Features
* **Lightning-Fast Local Transcription:** Powered by Parakeet and Rust for rapid speech-to-text without network latency.
* **Reliable Wake-Word Detection:** Uses Picovoice Porcupine to efficiently listen for custom trigger words (e.g., "Jarvis").
* **Cross-Platform Audio Capture:** Leverages `sounddevice` for robust and compatible microphone access.
* **Simple Python API:** A high-level, developer-friendly interface wrapping the complex Rust backend.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
* **Python** 3.10 or higher
* **Rust** & **Cargo** (for building the backend)
* **uv** (for fast Python package management)
* **maturin** (for building the Rust/Python bindings)
* A [Picovoice Console](https://console.picovoice.ai/) account to get a `PORCUPINE_KEY`.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/jarvis-voice.git
   cd jarvis-voice
   ```

2. **Set up the environment:**
   Use `uv` to sync dependencies and create a virtual environment:
   ```bash
   uv sync
   uv venv
   ```

3. **Build the Rust extensions:**
   Compile the Rust backend and install the package into your environment:
   ```bash
   maturin develop
   ```

4. **Configure your API Keys:**
   Create a `.env` file in the root directory (you can copy `.env.example`) and add your Porcupine access key:
   ```env
   PORCUPINE_KEY=your_picovoice_access_key_here
   ```

## Usage

Using JARVIS Voice is straightforward. The `Listener` class abstracts away the complexity of audio streaming and thread management.

Here is a simple example to get you started:

```python
import os
from jarvis_voice import Listener

# Ensure your PORCUPINE_KEY is set in your environment variables.
# Alternatively, this is automatically loaded from your .env file if python-dotenv is used.
os.environ["PORCUPINE_KEY"] = "your_picovoice_access_key_here"

# Initialize the listener with your chosen wake word
listener = Listener(wake_words=["jarvis"])

print("Starting JARVIS Voice...")

# Start the blocking loop to listen for the wake word
# Once triggered, it will transcribe your speech and print the result.
try:
    listener.listen()
except KeyboardInterrupt:
    print("\nShutting down gracefully.")
finally:
    listener.stop()
```

## Roadmap

* [ ] Implement voice activity detection (VAD) for better silence trimming.
* [ ] Support streaming transcription for real-time text output.
* [ ] Package pre-compiled wheels for easier installation on major platforms.

## Contributing

We welcome contributions! To help improve JARVIS Voice, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure your code passes all tests by running `pytest` in the `tests/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Maintainer: **JARVIS Team**  
For support or inquiries, please open an issue on the GitHub repository or reach out via email at `dev@example.com`.
