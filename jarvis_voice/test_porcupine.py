from os import getenv

import struct
import sounddevice as sd
import pvporcupine
from dotenv import load_dotenv

try:
    from . import jarvis_transcriber
except ImportError:
    import jarvis_voice.jarvis_transcriber as jarvis_transcriber

load_dotenv()

def get_next_audio_frame(stream, frame_length):
    pcm, overflowed = stream.read(frame_length)
    return struct.unpack_from("h" * frame_length, pcm)

if __name__ == '__main__':
    access_key = getenv('PORCUPINE_KEY')

    print(jarvis_transcriber.___version())

    handle = pvporcupine.create(access_key=access_key, keywords=['jarvis'])
    
    audio_stream = sd.RawInputStream(
        samplerate=handle.sample_rate,
        channels=1,
        dtype='int16',
        blocksize=handle.frame_length,
    )
    audio_stream.start()

    print("Listening...")
    try:
        while True:
            audio_frame = get_next_audio_frame(audio_stream, handle.frame_length)
            keyword_index = handle.process(audio_frame)
            if keyword_index == 0:
                print('Jarvis detected')
                break
    finally:
        audio_stream.stop()
        audio_stream.close()
        handle.delete()
