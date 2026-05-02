import pytest
import threading
import time
from unittest.mock import MagicMock, patch, call
from jarvis_voice.listener import Listener


# ---------------------------------------------------------------------------
# Helpers — shared mock setup
# ---------------------------------------------------------------------------

def _create_listener(**kwargs):
    """Create a Listener with all external deps mocked out."""
    with patch('pvporcupine.create') as mock_porcupine, \
         patch('sounddevice.RawInputStream') as mock_raw_stream, \
         patch('jarvis_voice.jarvis_transcriber.default') as mock_default:

        mock_handle = MagicMock()
        mock_handle.sample_rate = 16000
        mock_handle.frame_length = 512
        mock_porcupine.return_value = mock_handle

        mock_transcriber = MagicMock()
        mock_transcriber.wait_until_done.return_value = True
        mock_transcriber.get_latest_transcript.return_value = "hello world"
        mock_default.return_value = mock_transcriber

        listener = Listener(wake_words=['jarvis'], access_key='fake_key', **kwargs)

    return listener, mock_handle, mock_transcriber


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListenBlockingLoop:
    """Tests for the blocking ``listen()`` method."""

    def test_uses_wait_until_done_not_polling(self):
        """wait_until_done is called; is_transcribing polling is NOT used."""
        listener, mock_handle, mock_transcriber = _create_listener()

        # One real frame → wake word detected, then stop the loop
        listener._get_next_audio_frame = MagicMock(
            side_effect=[(0,) * 512, KeyboardInterrupt()],
        )
        mock_handle.process.return_value = 0

        try:
            listener.listen()
        except KeyboardInterrupt:
            pass

        mock_transcriber.start_transcription.assert_called_once()
        assert mock_transcriber.wait_until_done.called
        assert not mock_transcriber.is_transcribing.called

    def test_is_running_flag_controls_loop(self):
        """Setting ``_is_running = False`` breaks the loop without KeyboardInterrupt."""
        listener, mock_handle, mock_transcriber = _create_listener()

        call_count = 0

        def _fake_frame():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                listener._is_running = False
            return (0,) * 512

        listener._get_next_audio_frame = _fake_frame
        mock_handle.process.return_value = -1  # no wake word

        listener.listen()

        assert call_count >= 3
        assert listener._is_running is False

    def test_stop_sets_is_running_false(self):
        """Calling ``stop()`` from another context sets the flag."""
        listener, _, _ = _create_listener()
        listener._is_running = True
        listener.stop()
        assert listener._is_running is False


class TestCallbacks:
    """Tests for on_transcript and on_wake_word callbacks."""

    def test_on_transcript_callback_invoked(self):
        on_transcript = MagicMock()
        listener, mock_handle, mock_transcriber = _create_listener(
            on_transcript=on_transcript,
        )

        listener._get_next_audio_frame = MagicMock(
            side_effect=[(0,) * 512, KeyboardInterrupt()],
        )
        mock_handle.process.return_value = 0

        try:
            listener.listen()
        except KeyboardInterrupt:
            pass

        on_transcript.assert_called_once_with("hello world")

    def test_on_wake_word_callback_invoked(self):
        on_wake = MagicMock()
        listener, mock_handle, mock_transcriber = _create_listener(
            on_wake_word=on_wake,
        )

        listener._get_next_audio_frame = MagicMock(
            side_effect=[(0,) * 512, KeyboardInterrupt()],
        )
        mock_handle.process.return_value = 0

        try:
            listener.listen()
        except KeyboardInterrupt:
            pass

        on_wake.assert_called_once_with("jarvis")

    def test_no_callback_prints_to_stdout(self, capsys):
        """When no callback is registered, transcript is printed."""
        listener, mock_handle, mock_transcriber = _create_listener()

        listener._get_next_audio_frame = MagicMock(
            side_effect=[(0,) * 512, KeyboardInterrupt()],
        )
        mock_handle.process.return_value = 0

        try:
            listener.listen()
        except KeyboardInterrupt:
            pass

        captured = capsys.readouterr()
        assert 'hello world' in captured.out


class TestListenAsync:
    """Tests for the non-blocking ``listen_async()`` method."""

    def test_returns_thread_and_runs(self):
        listener, mock_handle, _ = _create_listener()
        mock_handle.process.return_value = -1  # no wake word

        # Let it spin for a moment then stop
        call_count = 0

        def _fake_frame():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                listener._is_running = False
            return (0,) * 512

        listener._get_next_audio_frame = _fake_frame
        thread = listener.listen_async()

        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_double_start_raises(self):
        listener, _, _ = _create_listener()
        listener._is_running = True

        with pytest.raises(RuntimeError, match="already running"):
            listener.listen_async()


class TestIsRunningProperty:
    """Tests for the ``is_running`` property."""

    def test_initially_false(self):
        listener, _, _ = _create_listener()
        assert listener.is_running is False

    def test_true_during_listen(self):
        listener, mock_handle, _ = _create_listener()
        mock_handle.process.return_value = -1
        observed = []

        call_count = 0

        def _fake_frame():
            nonlocal call_count
            call_count += 1
            observed.append(listener.is_running)
            if call_count >= 2:
                listener._is_running = False
            return (0,) * 512

        listener._get_next_audio_frame = _fake_frame
        listener.listen()

        # The first observation should have been True
        assert observed[0] is True

