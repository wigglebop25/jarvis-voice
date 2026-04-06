use crossbeam_channel::unbounded;
use jarvis_transcriber::audio::resampler::AudioResampler;

#[test]
fn flushes_stream_to_expected_length() {
    let (tx, rx) = unbounded();
    let mut resampler = AudioResampler::new(48_000, 1, 480, 16_000, tx).unwrap();

    resampler.process_f32(&vec![0.0; 1000]).unwrap();
    resampler.flush().unwrap();

    let mut total = Vec::new();
    while let Ok(chunk) = rx.try_recv() {
        total.extend(chunk);
    }

    assert_eq!(total.len(), 334);
    assert!(total.iter().all(|sample| sample.is_finite()));
}
