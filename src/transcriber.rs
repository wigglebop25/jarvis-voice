use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, unbounded};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::runtime::Runtime;
use transcribe_rs::TranscriptionEngine;
use transcribe_rs::engines::parakeet::ParakeetEngine;

use crate::config::Config;
use crate::resampler::AudioResampler;

const DEFAULT_MODEL_URI: &str = "https://blob.handy.computer/parakeet-v3-int8.tar.gz";
const DEFAULT_MODEL_PATH: &str = "parakeet-tdt-0.6b-v3-int8";
const TARGET_SAMPLE_RATE: usize = 16000;

enum Command {
    Start,
    Stop,
    Shutdown,
}

enum RawAudio {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

#[pyclass]
pub struct Transcriber {
    command_tx: Sender<Command>,
    is_transcribing: Arc<AtomicBool>,
    latest_transcript: Arc<Mutex<String>>,
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

        let is_transcribing_clone = is_transcribing.clone();
        let latest_transcript_clone = latest_transcript.clone();

        thread::spawn(move || {
            worker_thread(
                command_rx,
                is_transcribing_clone,
                latest_transcript_clone,
                rust_config,
                uri,
                path,
            );
        });

        Ok(Self {
            command_tx,
            is_transcribing,
            latest_transcript,
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
    config: Config,

    device: cpal::Device,
    stream_config: cpal::StreamConfig,
    sample_format: cpal::SampleFormat,

    raw_audio_tx: Sender<RawAudio>,
    raw_audio_rx: Receiver<RawAudio>,
    resampled_tx: Sender<Vec<f32>>,
    resampled_rx: Receiver<Vec<f32>>,

    stream: Option<cpal::Stream>,
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
        config: Config,
        model_uri: String,
        model_path: String,
    ) -> Result<Self> {
        let rt = Runtime::new().map_err(|e| anyhow!("Tokio runtime error: {}", e))?;

        let engine = rt
            .block_on(crate::model::load_model(&model_uri, &model_path))
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device found"))?;

        let config_cpal = device
            .default_input_config()
            .map_err(|e| anyhow!("Failed to get default input config: {}", e))?;

        let stream_config = config_cpal.config();
        let sample_format = config_cpal.sample_format();

        let (raw_audio_tx, raw_audio_rx) = unbounded();
        let (resampled_tx, resampled_rx) = unbounded();

        let frames_required_for_silence =
            (config.silence_duration * TARGET_SAMPLE_RATE as f32) as usize;

        Ok(Self {
            engine,
            command_rx,
            is_transcribing,
            latest_transcript,
            config,
            device,
            stream_config,
            sample_format,
            raw_audio_tx,
            raw_audio_rx,
            resampled_tx,
            resampled_rx,
            stream: None,
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

        if let Ok(mut guard) = self.latest_transcript.lock() {
            guard.clear();
        }

        self.resampler = Some(
            AudioResampler::new(
                self.stream_config.sample_rate as usize,
                self.stream_config.channels as usize,
                480, // chunk size
                TARGET_SAMPLE_RATE,
                self.resampled_tx.clone(),
            )
            .unwrap(),
        );

        let tx = self.raw_audio_tx.clone();
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let new_stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &self.stream_config,
                move |data: &[f32], _: &_| {
                    let _ = tx.send(RawAudio::F32(data.to_vec()));
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => self.device.build_input_stream(
                &self.stream_config,
                move |data: &[i16], _: &_| {
                    let _ = tx.send(RawAudio::I16(data.to_vec()));
                },
                err_fn,
                None,
            ),
            _ => {
                eprintln!("Unsupported sample format");
                self.is_active = false;
                self.is_transcribing.store(false, Ordering::SeqCst);
                return;
            }
        };

        match new_stream {
            Ok(s) => {
                if let Err(e) = s.play() {
                    eprintln!("Failed to play stream: {}", e);
                    self.is_active = false;
                    self.is_transcribing.store(false, Ordering::SeqCst);
                } else {
                    self.stream = Some(s);
                }
            }
            Err(e) => {
                eprintln!("Failed to build input stream: {}", e);
                self.is_active = false;
                self.is_transcribing.store(false, Ordering::SeqCst);
            }
        }
    }

    fn stop_capture(&mut self) {
        self.is_active = false;
        self.stream = None;

        if let Some(r) = self.resampler.as_mut() {
            let _ = r.flush();
        }
        while let Ok(flushed_chunk) = self.resampled_rx.try_recv() {
            self.accumulated_audio.extend_from_slice(&flushed_chunk);
        }

        self.transcribe_accumulated();
        self.is_transcribing.store(false, Ordering::SeqCst);
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
                .transcribe_samples(self.accumulated_audio.clone(), None)
                && let Ok(mut guard) = self.latest_transcript.lock()
            {
                *guard = result.text;
            }
            self.accumulated_audio.clear();
        }
    }
}

fn worker_thread(
    command_rx: Receiver<Command>,
    is_transcribing: Arc<AtomicBool>,
    latest_transcript: Arc<Mutex<String>>,
    config: Config,
    model_uri: String,
    model_path: String,
) {
    match TranscriptionWorker::new(
        command_rx,
        is_transcribing,
        latest_transcript,
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
