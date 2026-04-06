use anyhow::{Result, anyhow, bail};
use audioadapter_buffers::direct::InterleavedSlice;
use crossbeam_channel::Sender;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, calculate_cutoff,
};

use crate::utils::{interleaved_f32_to_mono, interleaved_i16_to_mono};

pub struct AudioResampler {
    input_channels: usize,
    chunk_size: usize,
    resampler: Async<f32>,
    pending_input: Vec<f32>,
    chunk_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    delay_frames_remaining: usize,
    total_input_frames: usize,
    total_output_frames: usize,
    channel_tx: Sender<Vec<f32>>,
}

impl AudioResampler {
    pub fn new(
        input_sample_rate: usize,
        input_channels: usize,
        chunk_size: usize,
        target_sample_rate: usize,
        channel_tx: Sender<Vec<f32>>,
    ) -> Result<Self> {
        if input_sample_rate == 0 {
            bail!("input_sample_rate must be greater than zero");
        }
        if input_channels == 0 {
            bail!("input_channels must be greater than zero");
        }
        if chunk_size == 0 {
            bail!("chunk_size must be greater than zero");
        }
        if target_sample_rate == 0 {
            bail!("target_sample_rate must be greater than zero");
        }

        let window = WindowFunction::BlackmanHarris2;
        let sinc_len = 256;
        let params = SincInterpolationParameters {
            sinc_len,
            f_cutoff: calculate_cutoff(sinc_len, window),
            interpolation: SincInterpolationType::Quadratic,
            oversampling_factor: 256,
            window,
        };

        let resample_ratio = target_sample_rate as f64 / input_sample_rate as f64;
        let resampler = Async::<f32>::new_sinc(
            resample_ratio,
            1.0,
            &params,
            chunk_size,
            1,
            FixedAsync::Input,
        )
        .map_err(|err| anyhow!("failed to construct rubato resampler: {err}"))?;

        let output_buffer = vec![0.0; resampler.output_frames_max()];
        let delay_frames_remaining = resampler.output_delay();

        Ok(Self {
            input_channels,
            chunk_size,
            resampler,
            pending_input: Vec::with_capacity(chunk_size * 2),
            chunk_buffer: vec![0.0; chunk_size],
            output_buffer,
            delay_frames_remaining,
            total_input_frames: 0,
            total_output_frames: 0,
            channel_tx,
        })
    }

    pub fn process_i16(&mut self, samples: &[i16]) -> Result<()> {
        let mono = interleaved_i16_to_mono(samples, self.input_channels)?;
        self.push_mono_frames(&mono)
    }

    pub fn process_f32(&mut self, samples: &[f32]) -> Result<()> {
        let mono = interleaved_f32_to_mono(samples, self.input_channels)?;
        self.push_mono_frames(&mono)
    }

    pub fn flush(&mut self) -> Result<()> {
        let expected_total_output =
            (self.total_input_frames as f64 * self.resampler.resample_ratio()).ceil() as usize;
        let expected_flush_output = expected_total_output.saturating_sub(self.total_output_frames);
        let mut emitted = Vec::new();

        if !self.pending_input.is_empty() {
            let remaining = self.pending_input.len();
            self.chunk_buffer.fill(0.0);
            self.chunk_buffer[..remaining].copy_from_slice(&self.pending_input);
            let produced = self.process_chunk(Some(remaining))?;
            emitted.extend(self.take_useful_output(produced));
            self.pending_input.clear();
        }

        while self.total_output_frames < expected_total_output {
            self.chunk_buffer.fill(0.0);
            let produced = self.process_chunk(Some(0))?;
            if produced == 0 {
                bail!(
                    "resampler flush stalled before producing the expected output (produced {}, expected {})",
                    self.total_output_frames,
                    expected_total_output
                );
            }
            emitted.extend(self.take_useful_output(produced));
        }

        if emitted.len() > expected_flush_output {
            emitted.truncate(expected_flush_output);
        }

        if !emitted.is_empty() {
            self.channel_tx
                .send(emitted)
                .map_err(|e| anyhow!("failed to send flushed audio: {e}"))?;
        }

        self.reset_stream();
        Ok(())
    }

    pub fn reset_stream(&mut self) {
        self.resampler.reset();
        self.pending_input.clear();
        self.chunk_buffer.fill(0.0);
        self.delay_frames_remaining = self.resampler.output_delay();
        self.total_input_frames = 0;
        self.total_output_frames = 0;
    }

    fn push_mono_frames(&mut self, mono_frames: &[f32]) -> Result<()> {
        self.pending_input.extend_from_slice(mono_frames);
        self.total_input_frames += mono_frames.len();

        let mut processed = 0usize;

        while self.pending_input.len().saturating_sub(processed) >= self.chunk_size {
            let end = processed + self.chunk_size;
            self.chunk_buffer
                .copy_from_slice(&self.pending_input[processed..end]);
            let produced = self.process_chunk(None)?;
            let emitted = self.take_useful_output(produced);
            if !emitted.is_empty() {
                self.channel_tx
                    .send(emitted)
                    .map_err(|e| anyhow!("failed to send resampled audio: {e}"))?;
            }
            processed = end;
        }

        if processed > 0 {
            self.pending_input.drain(..processed);
        }

        Ok(())
    }

    fn process_chunk(&mut self, partial_len: Option<usize>) -> Result<usize> {
        let input_adapter =
            InterleavedSlice::new(&self.chunk_buffer, 1, self.chunk_buffer.len())
                .map_err(|err| anyhow!("failed to wrap input buffer for rubato: {err}"))?;
        let output_frames = self.output_buffer.len();
        let mut output_adapter =
            InterleavedSlice::new_mut(&mut self.output_buffer, 1, output_frames)
                .map_err(|err| anyhow!("failed to wrap output buffer for rubato: {err}"))?;

        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len,
            active_channels_mask: None,
        };

        let (_, produced) = self
            .resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .map_err(|err| anyhow!("rubato failed while processing audio chunk: {err}"))?;

        Ok(produced)
    }

    fn take_useful_output(&mut self, produced_frames: usize) -> Vec<f32> {
        let trim = self.delay_frames_remaining.min(produced_frames);
        self.delay_frames_remaining -= trim;

        let useful = self.output_buffer[trim..produced_frames].to_vec();
        self.total_output_frames += useful.len();
        useful
    }
}

#[cfg(test)]
mod tests {
    use super::AudioResampler;
    use crossbeam_channel::unbounded;

    #[test]
    fn flushes_stream_to_expected_length() {
        let (tx, rx) = unbounded();
        let mut resampler = AudioResampler::new(48_000, 1, 480, 16_000, tx).unwrap();

        resampler.process_f32(&vec![0.0; 1000]).unwrap();
        resampler.flush().unwrap();

        let mut total = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            total.extend(chunk);
        }

        assert_eq!(total.len(), 334);
        assert!(total.iter().all(|sample| sample.is_finite()));
    }
}
