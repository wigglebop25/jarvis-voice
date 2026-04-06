use anyhow::{Result, bail};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
pub struct Config {
    #[pyo3(get, set)]
    pub silence_duration: f32,
    #[pyo3(get, set)]
    pub silence_threshold_rms: f32,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.silence_duration < 0.0 {
            bail!("silence_duration must be non-negative");
        }
        if self.silence_threshold_rms < 0.0 {
            bail!("silence_threshold_rms must be non-negative");
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            silence_duration: 1.0,
            silence_threshold_rms: 0.005,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = Config {
            silence_duration: -1.0,
            silence_threshold_rms: 0.005,
        };
        assert!(config.validate().is_err());

        let config = Config {
            silence_duration: 1.0,
            silence_threshold_rms: -0.1,
        };
        assert!(config.validate().is_err());

        let config = Config::default();
        assert!(config.validate().is_ok());
    }
}
