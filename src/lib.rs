use pyo3::prelude::*;

mod config;
mod core;
mod model;
mod resampler;
mod transcriber;
mod utils;

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
