use pyo3::prelude::*;

pub mod audio;
mod config;
pub mod core;
pub mod python;
pub mod transcriber;
pub mod transcription;
pub mod utils;

#[pymodule]
fn jarvis_transcriber(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    python::jarvis_transcriber(py, m)
}
