import pytest
import jarvis_transcriber
import time
import threading

def test_wait_until_done_exists():
    transcriber = jarvis_transcriber.default()
    # This should fail if wait_until_done is not implemented
    assert hasattr(transcriber, 'wait_until_done')

def test_wait_until_done_timeout():
    transcriber = jarvis_transcriber.default()
    # We can't easily trigger transcription without a mic here,
    # but we can test that calling it doesn't crash if nothing is happening.
    # Since we reset 'completed' to False in start_capture (which we haven't called),
    # it might block forever or return immediately if we initialize it to false.
    
    # In our implementation, we want it to raise TimeoutError on timeout.
    with pytest.raises(TimeoutError):
        transcriber.wait_until_done(timeout=0.1)

def test_register_callback_exists():
    transcriber = jarvis_transcriber.default()
    assert hasattr(transcriber, 'register_on_complete')

def test_wait_until_done_functional():
    transcriber = jarvis_transcriber.default()
    
    # Start transcription (mocking mic if necessary, but here it just starts the worker)
    transcriber.start_transcription()
    assert transcriber.is_transcribing()
    
    # We need to stop it manually to trigger the condvar
    # Since start_transcription runs in background, we can just stop it
    time.sleep(0.5) # Give it a moment to start
    transcriber.stop_transcription()
    
    # wait_until_done should now return true
    result = transcriber.wait_until_done(timeout=5.0)
    assert result is True
    assert not transcriber.is_transcribing()

def test_register_on_complete_functional():
    transcriber = jarvis_transcriber.default()
    
    event = threading.Event()
    received_transcript = None
    
    def on_complete(transcript):
        nonlocal received_transcript
        received_transcript = transcript
        event.set()
        
    transcriber.register_on_complete(on_complete)
    
    transcriber.start_transcription()
    time.sleep(0.5)
    transcriber.stop_transcription()
    
    # Wait for it to finish
    transcriber.wait_until_done(timeout=5.0)
    
    # The event should be set
    assert event.wait(timeout=1.0)
    assert isinstance(received_transcript, str)
