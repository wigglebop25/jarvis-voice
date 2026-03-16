import os
import time
import pvporcupine
import pyaudio
import struct
from dotenv import load_dotenv
import jarvis_transcriber

load_dotenv()

def get_next_audio_frame(stream, frame_length):
    try:
        pcm = stream.read(frame_length, exception_on_overflow=False)
        return struct.unpack_from("h" * frame_length, pcm)
    except Exception as e:
        print(f"Audio read error: {e}")
        return None

def main():
    access_key = os.getenv('PORCUPINE_KEY')
    if not access_key:
        print("Error: PORCUPINE_KEY not found in environment.")
        return

    # Initialize Transcriber (Rust backend)
    # This automatically downloads the model if it doesn't exist
    print("Initializing Transcriber...")
    transcriber = jarvis_transcriber.default()
    
    # Initialize Porcupine for wake word detection
    handle = pvporcupine.create(access_key=access_key, keywords=['jarvis'])
    
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=handle.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=handle.frame_length)

    print("--- JARVIS Voice System Active ---")
    print("Say 'Jarvis' to start transcribing.")
    
    try:
        while True:
            # Wake Word Detection Phase
            audio_frame = get_next_audio_frame(audio_stream, handle.frame_length)
            if audio_frame is None:
                continue
                
            keyword_index = handle.process(audio_frame)
            
            if keyword_index == 0:
                print("\n[WAKE WORD DETECTED] Starting transcription...")
                
                # Start Rust transcription
                # The Rust backend handles the mic capture and silence detection internally
                transcriber.start_transcription()
                
                print("Listening for speech (silence will trigger stop)...")
                
                # Wait while transcribing (silence detection happens in Rust)
                while transcriber.is_transcribing():
                    time.sleep(0.1)
                
                # Get and print the result
                transcript = transcriber.get_latest_transcript()
                print(f"Transcript: \"{transcript}\"")
                print("\nReturning to wake word detection...")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        handle.delete()

if __name__ == '__main__':
    main()
