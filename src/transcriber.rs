use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use transcribe_rs::TranscriptionEngine;
use transcribe_rs::engines::parakeet::ParakeetEngine;

use crate::config::Config;
use crate::audio::resampler::AudioResampler;
use crate::audio::input::{AudioInput, RawAudio};

const DEFAULT_MODEL_URI: &str = "https://blob.handy.computer/parakeet-v3-int8.tar.gz";
const DEFAULT_MODEL_PATH: &str = "parakeet-tdt-0.6b-v3-int8";
const TARGET_SAMPLE_RATE: usize = 16000;

enum Command {
    Start,
    Stop,
    Shutdown,
}

#[pyclass]
pub struct Transcriber {
    command_tx: Sender<Command>,
    is_transcribing: Arc<AtomicBool>,
    latest_transcript: Arc<Mutex<String>>,
    completion_notifier: Arc<(Mutex<bool>, Condvar)>,
    on_complete_callback: Arc<Mutex<Option<Py<PyAny>>>>,
}

#[pymethods]
impl Transcriber {
    #[new]
    #[pyo3(signature = (config=None, model_uri=None, model_path=None))]
    fn py_new(
        config: Option<Py<Config>>,
        model_uri: Option<String>,
        model_path: Option<String>,
    ) -> PyResult<Self> {
        let uri = model_uri.unwrap_or_else(|| DEFAULT_MODEL_URI.to_string());
        let path = model_path.unwrap_or_else(|| DEFAULT_MODEL_PATH.to_string());

        let rust_config = Python::attach(|py| {
            if let Some(c) = config {
                let b = c.borrow(py);
                Config {
                    silence_duration: b.silence_duration,
                    silence_threshold_rms: b.silence_threshold_rms,
                }
            } else {
                Config::default()
            }
        });

        let (command_tx, command_rx) = unbounded();
        let is_transcribing = Arc::new(AtomicBool::new(false));
        let latest_transcript = Arc::new(Mutex::new(String::new()));
        let completion_notifier = Arc::new((Mutex::new(false), Condvar::new()));
        let on_complete_callback = Arc::new(Mutex::new(None));

        let is_transcribing_clone = is_transcribing.clone();
        let latest_transcript_clone = latest_transcript.clone();
        let completion_notifier_clone = completion_notifier.clone();
        let on_complete_callback_clone = on_complete_callback.clone();

        thread::spawn(move || {
            worker_thread(
                command_rx,
                is_transcribing_clone,
                latest_transcript_clone,
                completion_notifier_clone,
                on_complete_callback_clone,
                rust_config,
                uri,
                path,
            );
        });

        Ok(Self {
            command_tx,
            is_transcribing,
            latest_transcript,
            completion_notifier,
            on_complete_callback,
        })
    }

    fn start_transcription(&self) -> PyResult<()> {
        self.is_transcribing.store(true, Ordering::SeqCst);
        if let Ok(mut guard) = self.latest_transcript.lock() {
            guard.clear();
        }
        self.command_tx.send(Command::Start).map_err(|e| {
            self.is_transcribing.store(false, Ordering::SeqCst);
            PyRuntimeError::new_err(format!("Failed to send start command: {}", e))
        })
    }

    fn stop_transcription(&self) -> PyResult<()> {
        self.command_tx
            .send(Command::Stop)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to send stop command: {}", e)))
    }

    fn is_transcribing(&self) -> bool {
        self.is_transcribing.load(Ordering::SeqCst)
    }

    fn get_latest_transcript(&self) -> String {
        self.latest_transcript.lock().unwrap().clone()
    }

    fn register_on_complete(&self, callback: Py<PyAny>) {
        let mut guard = self.on_complete_callback.lock().unwrap();
        *guard = Some(callback);
    }

    #[pyo3(signature = (timeout=None))]
    fn wait_until_done(&self, timeout: Option<f64>) -> PyResult<bool> {
        let (lock, cvar) = &*self.completion_notifier;
        let mut completed = lock.lock().unwrap();

        if *completed {
            return Ok(true);
        }

        if let Some(timeout_s) = timeout {
            let duration = Duration::from_secs_f64(timeout_s);
            let start = Instant::now();
            while !*completed {
                let elapsed = start.elapsed();
                if elapsed >= duration {
                    return Err(PyTimeoutError::new_err("Transcription timed out"));
                }
                let remaining = duration - elapsed;
                let (guard, res) = cvar
                    .wait_timeout(completed, remaining)
                    .map_err(|e| PyRuntimeError::new_err(format!("Condvar wait error: {}", e)))?;
                completed = guard;
                if res.timed_out() && !*completed {
                    return Err(PyTimeoutError::new_err("Transcription timed out"));
                }
            }
        } else {
            while !*completed {
                completed = cvar
                    .wait(completed)
                    .map_err(|e| PyRuntimeError::new_err(format!("Condvar wait error: {}", e)))?;
            }
        }

        Ok(true)
    }
}

impl Drop for Transcriber {
    fn drop(&mut self) {
        let _ = self.command_tx.send(Command::Shutdown);
    }
}

#[pyfunction]
pub fn default() -> PyResult<Transcriber> {
    Transcriber::py_new(None, None, None)
}

struct TranscriptionWorker {
    engine: ParakeetEngine,
    command_rx: Receiver<Command>,
    is_transcribing: Arc<AtomicBool>,
    latest_transcript: Arc<Mutex<String>>,
    completion_notifier: Arc<(Mutex<bool>, Condvar)>,
    on_complete_callback: Arc<Mutex<Option<Py<PyAny>>>>,
    config: Config,

    audio_input: AudioInput,

    raw_audio_tx: Sender<RawAudio>,
    raw_audio_rx: Receiver<RawAudio>,
    resampled_tx: Sender<Vec<f32>>,
    resampled_rx: Receiver<Vec<f32>>,

    resampler: Option<AudioResampler>,
    accumulated_audio: Vec<f32>,
    silence_frames: usize,
    frames_required_for_silence: usize,
    is_active: bool,
}

impl TranscriptionWorker {
    fn new(
        command_rx: Receiver<Command>,
        is_transcribing: Arc<AtomicBool>,
        latest_transcript: Arc<Mutex<String>>,
        completion_notifier: Arc<(Mutex<bool>, Condvar)>,
        on_complete_callback: Arc<Mutex<Option<Py<PyAny>>>>,
        config: Config,
        model_uri: String,
        model_path: String,
    ) -> Result<Self> {
        let rt = Runtime::new().map_err(|e| anyhow!("Tokio runtime error: {}", e))?;

        let engine = rt
            .block_on(crate::model::load_model(&model_uri, &model_path))
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        let audio_input = AudioInput::new()?;

        let (raw_audio_tx, raw_audio_rx) = unbounded();
        let (resampled_tx, resampled_rx) = unbounded();

        let frames_required_for_silence =
            (config.silence_duration * TARGET_SAMPLE_RATE as f32) as usize;

        Ok(Self {
            engine,
            command_rx,
            is_transcribing,
            latest_transcript,
            completion_notifier,
            on_complete_callback,
            config,
            audio_input,
            raw_audio_tx,
            raw_audio_rx,
            resampled_tx,
            resampled_rx,
            resampler: None,
            accumulated_audio: Vec::new(),
            silence_frames: 0,
            frames_required_for_silence,
            is_active: false,
        })
    }

    fn run(&mut self) {
        loop {
            if self.is_active {
                crossbeam_channel::select! {
                    recv(self.command_rx) -> cmd => {
                        match cmd {
                            Ok(Command::Stop) => self.stop_capture(),
                            Ok(Command::Shutdown) | Err(_) => break,
                            Ok(Command::Start) => {} // ignore
                        }
                    }
                    recv(self.raw_audio_rx) -> raw_chunk => {
                        if let Ok(chunk) = raw_chunk {
                            self.process_raw_audio(chunk);
                        }
                    }
                    recv(self.resampled_rx) -> resampled_chunk => {
                        if let Ok(chunk) = resampled_chunk {
                            self.process_resampled_chunk(chunk);
                        }
                    }
                }
            } else {
                match self.command_rx.recv() {
                    Ok(Command::Start) => self.start_capture(),
                    Ok(Command::Stop) => {}
                    Ok(Command::Shutdown) | Err(_) => break,
                }
            }
        }
    }

    fn start_capture(&mut self) {
        self.is_active = true;
        self.is_transcribing.store(true, Ordering::SeqCst);
        self.accumulated_audio.clear();
        self.silence_frames = 0;

        // Reset completion notifier flag
        {
            let (lock, _cvar) = &*self.completion_notifier;
            let mut completed = lock.lock().unwrap();
            *completed = false;
        }

        if let Ok(mut guard) = self.latest_transcript.lock() {
            guard.clear();
        }

        self.resampler = Some(
            AudioResampler::new(
                self.audio_input.stream_config.sample_rate as usize,
                self.audio_input.stream_config.channels as usize,
                480, // chunk size
                TARGET_SAMPLE_RATE,
                self.resampled_tx.clone(),
            )
            .unwrap(),
        );

        if let Err(e) = self.audio_input.start_stream(self.raw_audio_tx.clone()) {
            eprintln!("Failed to start audio input stream: {}", e);
            self.is_active = false;
            self.is_transcribing.store(false, Ordering::SeqCst);
        }
    }

    fn stop_capture(&mut self) {
        self.is_active = false;
        self.audio_input.stop_stream();

        if let Some(r) = self.resampler.as_mut() {
            let _ = r.flush();
        }
        while let Ok(flushed_chunk) = self.resampled_rx.try_recv() {
            self.accumulated_audio.extend_from_slice(&flushed_chunk);
        }

        self.transcribe_accumulated();
        self.is_transcribing.store(false, Ordering::SeqCst);

        // Invoke callback
        let transcript = self.latest_transcript.lock().unwrap().clone();
        let callback_guard = self.on_complete_callback.lock().unwrap();
        if let Some(callback) = &*callback_guard {
            let callback = callback;
            Python::attach(|py| {
                if let Err(e) = callback.call1(py, (transcript,)) {
                    eprintln!("Error invoking Python callback: {}", e);
                }
            });
        }

        // Notify waiters
        {
            let (lock, cvar) = &*self.completion_notifier;
            let mut completed = lock.lock().unwrap();
            *completed = true;
            cvar.notify_all();
        }
    }

    fn process_raw_audio(&mut self, chunk: RawAudio) {
        if let Some(r) = self.resampler.as_mut() {
            match chunk {
                RawAudio::F32(data) => {
                    let _ = r.process_f32(&data);
                }
                RawAudio::I16(data) => {
                    let _ = r.process_i16(&data);
                }
            }
        }
    }

    fn process_resampled_chunk(&mut self, chunk: Vec<f32>) {
        self.accumulated_audio.extend_from_slice(&chunk);

        let mut sum_sq = 0.0;
        for &sample in &chunk {
            sum_sq += sample * sample;
        }
        let rms = if chunk.is_empty() {
            0.0
        } else {
            (sum_sq / chunk.len() as f32).sqrt()
        };

        if rms < self.config.silence_threshold_rms {
            self.silence_frames += chunk.len();
        } else {
            self.silence_frames = 0;
        }

        if self.silence_frames >= self.frames_required_for_silence {
            self.stop_capture();
        }
    }

    fn transcribe_accumulated(&mut self) {
        if !self.accumulated_audio.is_empty() {
            if let Ok(result) = self
                .engine
                .transcribe_samples(self.accumulated_audio.clone(), None) {
                if let Ok(mut guard) = self.latest_transcript.lock() {
                    *guard = result.text;
                }
            }
            self.accumulated_audio.clear();
        }
    }
}

fn worker_thread(
    command_rx: Receiver<Command>,
    is_transcribing: Arc<AtomicBool>,
    latest_transcript: Arc<Mutex<String>>,
    completion_notifier: Arc<(Mutex<bool>, Condvar)>,
    on_complete_callback: Arc<Mutex<Option<Py<PyAny>>>>,
    config: Config,
    model_uri: String,
    model_path: String,
) {
    match TranscriptionWorker::new(
        command_rx,
        is_transcribing,
        latest_transcript,
        completion_notifier,
        on_complete_callback,
        config,
        model_uri,
        model_path,
    ) {
        Ok(mut worker) => {
            worker.run();
        }
        Err(e) => {
            eprintln!("Failed to initialize TranscriptionWorker: {}", e);
        }
    }
}
