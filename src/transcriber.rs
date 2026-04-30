use crossbeam_channel::{Sender, unbounded};
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::transcription::engine::{Command, worker_thread};

const DEFAULT_MODEL_URI: &str = "https://blob.handy.computer/parakeet-v3-int8.tar.gz";
const DEFAULT_MODEL_PATH: &str = "parakeet-tdt-0.6b-v3-int8";

#[pyclass(skip_from_py_object)]
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
                crate::core::config::Config {
                    silence_duration: b.silence_duration,
                    silence_threshold_rms: b.silence_threshold_rms,
                }
            } else {
                crate::core::config::Config::default()
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
    fn wait_until_done(&self, py: Python, timeout: Option<f64>) -> PyResult<bool> {
        py.detach(|| {
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
                    let (guard, res) = cvar.wait_timeout(completed, remaining).map_err(|e| {
                        PyRuntimeError::new_err(format!("Condvar wait error: {:?}", e))
                    })?;
                    completed = guard;
                    if res.timed_out() && !*completed {
                        return Err(PyTimeoutError::new_err("Transcription timed out"));
                    }
                }
            } else {
                while !*completed {
                    completed = cvar.wait(completed).map_err(|e| {
                        PyRuntimeError::new_err(format!("Condvar wait error: {:?}", e))
                    })?;
                }
            }
            Ok(true)
        })
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
