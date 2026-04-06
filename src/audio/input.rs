use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;

pub enum RawAudio {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

pub struct AudioInput {
    device: cpal::Device,
    pub stream_config: cpal::StreamConfig,
    pub sample_format: cpal::SampleFormat,
    stream: Option<cpal::Stream>,
}

impl AudioInput {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No input device found"))?;

        let config_cpal = device
            .default_input_config()
            .map_err(|e| anyhow!("Failed to get default input config: {}", e))?;

        let stream_config = config_cpal.config();
        let sample_format = config_cpal.sample_format();

        Ok(Self {
            device,
            stream_config,
            sample_format,
            stream: None,
        })
    }

    pub fn start_stream(&mut self, tx: Sender<RawAudio>) -> Result<()> {
        let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

        let stream = match self.sample_format {
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
            _ => return Err(anyhow!("Unsupported sample format")),
        }?;

        stream.play().map_err(|e| anyhow!("Failed to play stream: {}", e))?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop_stream(&mut self) {
        self.stream = None;
    }
}
