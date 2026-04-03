import os
import struct
import time
from typing import Union

import pyaudio
import pvporcupine
import jarvis_transcriber

class Listener:
    def __init__(self, wake_words: Union[str, list[str]], access_key: str = None):
        if access_key:
            self.access_key = access_key
        else:
            self.access_key = os.getenv('PORCUPINE_KEY')
            
        if not self.access_key:
            raise ValueError("Error: PORCUPINE_KEY not found in environment.")

        self.wake_words = [wake_words] if isinstance(wake_words, str) else wake_words
        self.transcriber = jarvis_transcriber.default()
        self.handle = None
        self.pa = None
        self.audio_stream = None
        self._setup_resources()

    def _setup_resources(self):
        self.handle = pvporcupine.create(access_key=self.access_key, keywords=self.wake_words)
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.handle.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.handle.frame_length
        )

    def _get_next_audio_frame(self):
        try:
            pcm = self.audio_stream.read(self.handle.frame_length, exception_on_overflow=False)
            return struct.unpack_from("h" * self.handle.frame_length, pcm)
        except Exception as e:
            print(f"Audio read error: {e}")
            return None

    def listen(self):
        print("--- JARVIS Voice System Active ---")
        try:
            while True:
                audio_frame = self._get_next_audio_frame()
                if audio_frame is None:
                    continue
                    
                keyword_index = self.handle.process(audio_frame)
                
                if keyword_index >= 0:
                    print(f"\n[WAKE WORD DETECTED: {self.wake_words[keyword_index]}]")
                    
                    self.transcriber.start_transcription()
                    
                    while self.transcriber.is_transcribing():
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def restart(self):
        self.stop()
        self._setup_resources()

    def stop(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        if self.pa:
            self.pa.terminate()
            self.pa = None
        if self.handle:
            self.handle.delete()
            self.handle = None
    
    def __del__(self):
        self.stop()
        
    def get_transcript(self) -> str:
        return self.transcriber.get_latest_transcript()

    def is_listening(self) -> bool:
        return self.transcriber.is_transcribing()