use anyhow::{Result, bail};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub trait AnyhowError<T> {
    fn to_py(self) -> PyResult<T>;
}

impl<T> AnyhowError<T> for Result<T> {
    fn to_py(self) -> PyResult<T> {
        self.map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }
}

pub fn interleaved_i16_to_mono(samples: &[i16], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("channels must be greater than zero");
    }
    if !samples.len().is_multiple_of(channels) {
        bail!(
            "expected input sample count to be divisible by {channels} channel(s), got {}",
            samples.len()
        );
    }

    let frames = samples.len() / channels;
    if channels == 1 {
        return Ok(samples
            .iter()
            .map(|sample| normalize_i16(*sample))
            .collect());
    }

    let mut mono = Vec::with_capacity(frames);
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().map(|sample| normalize_i16(*sample)).sum();
        mono.push(sum / channels as f32);
    }

    Ok(mono)
}

pub fn interleaved_f32_to_mono(samples: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("channels must be greater than zero");
    }
    if !samples.len().is_multiple_of(channels) {
        bail!(
            "expected input sample count to be divisible by {channels} channel(s), got {}",
            samples.len()
        );
    }

    let frames = samples.len() / channels;
    if channels == 1 {
        return Ok(samples.to_vec());
    }

    let mut mono = Vec::with_capacity(frames);
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }

    Ok(mono)
}

pub fn normalize_i16(sample: i16) -> f32 {
    if sample == i16::MIN {
        -1.0
    } else {
        sample as f32 / i16::MAX as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmixes_interleaved_i16_to_mono() {
        let mono = interleaved_i16_to_mono(&[i16::MAX, 0, 0, i16::MAX], 2).unwrap();
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }
}
