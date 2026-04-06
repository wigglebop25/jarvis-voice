use crossbeam_channel::unbounded;
use jarvis_transcriber::core::config::Config;
use jarvis_transcriber::transcription::engine::TranscriptionWorker;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};

#[test]
fn test_worker_init_invalid_path() {
    let (_command_tx, command_rx) = unbounded();
    let is_transcribing = Arc::new(AtomicBool::new(false));
    let latest_transcript = Arc::new(Mutex::new(String::new()));
    let completion_notifier = Arc::new((Mutex::new(false), Condvar::new()));
    let on_complete_callback = Arc::new(Mutex::new(None));
    let config = Config::default();

    let res = TranscriptionWorker::new(
        command_rx,
        is_transcribing,
        latest_transcript,
        completion_notifier,
        on_complete_callback,
        config,
        "invalid_uri".to_string(),
        "invalid_path".to_string(),
    );

    assert!(res.is_err());
}
