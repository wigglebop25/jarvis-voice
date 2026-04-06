use pyo3::prelude::*;

pub mod audio;
mod config;
pub mod core;
pub mod model;
pub mod transcriber;
pub mod utils;

#[cfg(test)]
mod test_sync;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
fn ___version() -> &'static str {
    VERSION
}

#[pymodule]
fn jarvis_transcriber(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(___version))?;
    m.add_wrapped(wrap_pyfunction!(transcriber::default))?;

    m.add_class::<config::Config>()?;
    m.add_class::<transcriber::Transcriber>()?;
    Ok(())
}
