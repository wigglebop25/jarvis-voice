import os
import struct
import threading
from typing import Callable, Optional, Union

import pvporcupine
import sounddevice as sd

from . import jarvis_transcriber


class Listener:
    """
    A high-level interface for voice interaction, combining wake-word detection
    (via Porcupine) and speech-to-text transcription (via Parakeet/Rust).

    Supports optional callbacks for integration with GUI applications and
    backend services that cannot rely on stdout.

    Usage (blocking):
        ```python
        from jarvis_voice import Listener

        listener = Listener(wake_words=["jarvis"])
        listener.listen()  # blocks until stop() is called or KeyboardInterrupt
        ```

    Usage (non-blocking with callbacks):
        ```python
        from jarvis_voice import Listener

        def handle_transcript(text: str) -> None:
            print(f"Got: {text}")

        listener = Listener(
            wake_words=["jarvis"],
            on_transcript=handle_transcript,
        )
        listener.listen_async()   # returns immediately
        # ... later ...
        listener.stop()
        ```
    """

    def __init__(
        self,
        wake_words: Union[str, list[str]],
        access_key: Optional[str] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_wake_word: Optional[Callable[[str], None]] = None,
    ):
        """
        Initializes the Listener with specified wake words and Porcupine access key.

        Args:
            wake_words: A string or list of strings representing the wake words
                to listen for (e.g., ``"jarvis"``).
            access_key: Your Porcupine access key. If not provided, it looks
                for the ``PORCUPINE_KEY`` environment variable.
            on_transcript: Optional callback invoked with the transcript string
                each time a transcription completes. When ``None``, transcripts
                are printed to stdout.
            on_wake_word: Optional callback invoked with the detected wake word
                string each time a wake word is recognised.
        """
        if access_key:
            self.access_key = access_key
        else:
            self.access_key = os.getenv('PORCUPINE_KEY')

        if not self.access_key:
            raise ValueError("Error: PORCUPINE_KEY not found in environment.")

        self.wake_words = [wake_words] if isinstance(wake_words, str) else wake_words
        self.transcriber = jarvis_transcriber.default()
        self.handle = None
        self.audio_stream = None

        # Callbacks
        self._on_transcript = on_transcript
        self._on_wake_word = on_wake_word

        # Graceful shutdown flag
        self._is_running = False
        self._listen_thread: Optional[threading.Thread] = None

        self._setup_resources()

    def _setup_resources(self):
        if self.access_key is None:
            raise ValueError("Error: PORCUPINE_KEY not found in environment.")

        self.handle = pvporcupine.create(access_key=self.access_key, keywords=self.wake_words)
        self.audio_stream = sd.RawInputStream(
            samplerate=self.handle.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.handle.frame_length,
        )
        self.audio_stream.start()

    def _get_next_audio_frame(self):
        try:
            pcm, overflowed = self.audio_stream.read(self.handle.frame_length)
            return struct.unpack_from("h" * self.handle.frame_length, pcm)
        except Exception as e:
            print(f"Audio read error: {e}")
            return None

    def listen(self):
        """
        Starts a blocking loop that continually listens for the wake word.

        Once detected, it automatically starts the transcription process,
        waits for the user to finish speaking (detected via silence),
        and delivers the resulting transcript via the ``on_transcript``
        callback (or prints it to stdout when no callback is registered).

        The loop runs until :meth:`stop` is called from another thread
        or a ``KeyboardInterrupt`` is received.
        """
        self._is_running = True
        print("--- JARVIS Voice System Active ---")
        try:
            while self._is_running:
                audio_frame = self._get_next_audio_frame()
                if audio_frame is None:
                    continue

                keyword_index = self.handle.process(audio_frame)

                if keyword_index >= 0:
                    detected_word = self.wake_words[keyword_index]
                    print(f"\n[WAKE WORD DETECTED: {detected_word}]")

                    if self._on_wake_word is not None:
                        self._on_wake_word(detected_word)

                    self.force_start_transcribe()
                    self.transcriber.wait_until_done()
                    transcript = self.get_transcript()

                    if self._on_transcript is not None:
                        self._on_transcript(transcript)
                    else:
                        print(f'Transcript: "{transcript}"')
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._is_running = False
            self._release_resources()

    def listen_async(self) -> threading.Thread:
        """
        Starts the :meth:`listen` loop in a background daemon thread and
        returns immediately.

        Returns:
            The ``threading.Thread`` running the listener loop.

        Raises:
            RuntimeError: If the listener is already running.
        """
        if self._is_running:
            raise RuntimeError("Listener is already running.")

        self._listen_thread = threading.Thread(target=self.listen, daemon=True)
        self._listen_thread.start()
        return self._listen_thread

    def force_start_transcribe(self):
        """Manually triggers the transcription process."""
        self.transcriber.start_transcription()

    def restart(self):
        """Stops and re-initialises the listener resources."""
        self.stop()
        self._setup_resources()

    def stop(self):
        """
        Signals the listen loop to exit gracefully and releases resources.

        Safe to call from any thread.
        """
        self._is_running = False

        # If running in a background thread, wait for it to finish
        if self._listen_thread is not None and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=5.0)
            self._listen_thread = None

        self._release_resources()

    def _release_resources(self):
        """Releases audio stream and Porcupine handle if they are active."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self.handle:
            self.handle.delete()
            self.handle = None

    def __del__(self):
        self.stop()

    def get_transcript(self) -> str:
        """Returns the most recent transcription result as a string."""
        return self.transcriber.get_latest_transcript()

    @property
    def is_running(self) -> bool:
        """Returns ``True`` if the listen loop is currently active."""
        return self._is_running

    def is_listening(self) -> bool:
        """Returns ``True`` if the underlying transcription engine is currently capturing audio."""
        return self.transcriber.is_transcribing()