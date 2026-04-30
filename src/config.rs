use crate::core::config::Config as CoreConfig;
use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub struct Config {
    #[pyo3(get, set)]
    pub silence_duration: f32,
    #[pyo3(get, set)]
    pub silence_threshold_rms: f32,
}

impl From<CoreConfig> for Config {
    fn from(c: CoreConfig) -> Self {
        Self {
            silence_duration: c.silence_duration,
            silence_threshold_rms: c.silence_threshold_rms,
        }
    }
}

impl From<Config> for CoreConfig {
    fn from(c: Config) -> Self {
        Self {
            silence_duration: c.silence_duration,
            silence_threshold_rms: c.silence_threshold_rms,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        CoreConfig::default().into()
    }
}

#[pymethods]
impl Config {
    #[new]
    #[pyo3(signature = (silence_duration=None, silence_threshold_rms=None))]
    fn py_new(silence_duration: Option<f32>, silence_threshold_rms: Option<f32>) -> Self {
        let mut config = CoreConfig::default();
        if let Some(d) = silence_duration {
            config.silence_duration = d;
        }
        if let Some(t) = silence_threshold_rms {
            config.silence_threshold_rms = t;
        }
        config.into()
    }

    fn validate(&self) -> PyResult<()> {
        use crate::utils::AnyhowError;
        let core: CoreConfig = (*self).into();
        core.validate().to_py()
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
