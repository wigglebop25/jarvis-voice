import pytest
from unittest.mock import MagicMock, patch
from jarvis_voice.listener import Listener
import time

def test_listener_uses_wait_until_done():
    # Mock resources to avoid real audio/porcupine init
    with patch('pvporcupine.create') as mock_porcupine, \
         patch('sounddevice.RawInputStream') as mock_raw_stream, \
         patch('jarvis_voice.jarvis_transcriber.default') as mock_transcriber_default:

        mock_handle = MagicMock()
        mock_handle.sample_rate = 16000
        mock_handle.frame_length = 512
        mock_porcupine.return_value = mock_handle
        
        mock_transcriber = MagicMock()
        mock_transcriber_default.return_value = mock_transcriber
        # Make is_transcribing return True once then False to simulate old polling loop finishing
        mock_transcriber.is_transcribing.side_effect = [True, False]
        # wait_until_done should not be called in old implementation
        mock_transcriber.wait_until_done.return_value = True
        
        # Setup Listener
        listener = Listener(wake_words=['jarvis'], access_key='fake_key')
        
        # Mock _get_next_audio_frame to return one frame then raise KeyboardInterrupt
        listener._get_next_audio_frame = MagicMock(side_effect=[(0,) * 512, KeyboardInterrupt()])
        
        # Mock handle.process to return 0 (wake word detected)
        mock_handle.process.return_value = 0
        
        # We want to verify that it DOES NOT call is_transcribing in a loop
        # and DOES call wait_until_done
        
        try:
            listener.listen()
        except KeyboardInterrupt:
            pass
            
        # Verify start_transcription was called
        mock_transcriber.start_transcription.assert_called_once()
        
        # Verify wait_until_done was called
        # If this fails in the Red Phase, it means it's still using the old polling method
        # (or the test needs adjustment)
        assert mock_transcriber.wait_until_done.called
        
        # Verify is_transcribing was NOT called (which was used for polling)
        assert not mock_transcriber.is_transcribing.called
