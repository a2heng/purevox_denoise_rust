//! One model, one call per 10 ms hop — the entry point most users need.
//!
//! [`Engine`] hides which architecture a preset uses and keeps the streaming
//! cache for you. Feed it a complex spectrum (`2 × 481` f32, real part first)
//! and it returns the denoised complex spectrum of the same shape. The FFT and
//! the overlap-add belong to the caller, exactly like the ONNX contract this
//! crate mirrors.

use std::path::{Path, PathBuf};

use crate::model::Denoiser;
use crate::model05::Denoiser05;
use crate::stream::StreamCache;
use crate::weights::WeightError;

/// Architecture-agnostic streaming engine.
pub enum Engine {
    /// 11d family: FIR gate, `N` dual-path GRUs, optional bottleneck memory.
    D { model: Denoiser, cache: StreamCache },
    /// 05 family: time/frequency attention, dual-path BiGRU.
    A05 { model: Denoiser05, cache: crate::model05::Cache05 },
}

impl Engine {
    /// Load a model directory that contains `meta.json` and `weights.safetensors`.
    ///
    /// The architecture is detected from `meta.json`; both families use the same
    /// directory layout, so swapping a model is just swapping a directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, WeightError> {
        let dir: PathBuf = dir.as_ref().to_path_buf();
        let meta_txt = std::fs::read_to_string(dir.join("meta.json"))
            .map_err(WeightError::Io)?;
        if meta_txt.contains("202605") {
            let model = Denoiser05::load(&dir)?;
            let cache = crate::model05::Cache05::zeros(&model.meta);
            Ok(Engine::A05 { model, cache })
        } else {
            let model = Denoiser::load(&dir)?;
            let cache = StreamCache::zeros(&model.meta);
            Ok(Engine::D { model, cache })
        }
    }

    /// Load a bundled preset by name (e.g. `purevox_denoise_202605a`) from
    /// [`crate::models_dir`].
    pub fn preset(name: &str) -> Result<Self, WeightError> {
        Self::load(crate::models_dir().join(name))
    }

    /// Number of f32 values in one hop of spectrum (`2 × bins`, real part first).
    pub fn spec_len(&self) -> usize {
        match self {
            Engine::D { model, .. } => 2 * model.meta.erb.n_freq,
            Engine::A05 { model, .. } => 2 * model.meta.erb.n_freq,
        }
    }

    /// Model name recorded in `meta.json`.
    pub fn name(&self) -> &str {
        match self {
            Engine::D { model, .. } => &model.meta.model,
            Engine::A05 { model, .. } => &model.meta.model,
        }
    }

    /// Sample rate the model was trained for (48 kHz for every preset).
    pub fn sample_rate(&self) -> u32 {
        match self {
            Engine::D { model, .. } => model.meta.stft.fs,
            Engine::A05 { model, .. } => model.meta.stft.fs,
        }
    }

    /// Number of f32 values in the streaming cache.
    pub fn cache_len(&self) -> usize {
        match self {
            Engine::D { cache, .. } => cache.total(),
            Engine::A05 { cache, .. } => cache.total(),
        }
    }

    /// Snapshot of the streaming cache (same order as the ONNX `cache_out`).
    pub fn cache_flat(&self) -> Vec<f32> {
        match self {
            Engine::D { cache, .. } => cache.to_flat(),
            Engine::A05 { cache, .. } => cache.to_flat(),
        }
    }

    /// Restore a cache snapshot (session hand-over, or parity tests).
    pub fn set_cache_flat(&mut self, flat: &[f32]) {
        match self {
            Engine::D { cache, .. } => cache.from_flat(flat),
            Engine::A05 { cache, .. } => cache.from_flat(flat),
        }
    }

    /// One hop: complex spectrum in, denoised complex spectrum out.
    pub fn process_hop(&mut self, spec: &[f32]) -> Vec<f32> {
        match self {
            Engine::D { model, cache } => model.stream_hop(spec, cache, None),
            Engine::A05 { model, cache } => model.stream_hop(spec, cache),
        }
    }

    /// Same as [`Engine::process_hop`] but with an explicit mask-depth control
    /// `(floor_db, depth_gain)`. Only the 11d family honours it; the 05 family
    /// ignores the extra argument.
    pub fn process_hop_with_ctrl(&mut self, spec: &[f32], ctrl: (f32, f32)) -> Vec<f32> {
        match self {
            Engine::D { model, cache } => model.stream_hop(spec, cache, Some(ctrl)),
            Engine::A05 { model, cache } => model.stream_hop(spec, cache),
        }
    }

    /// Streaming file-to-file processing (keeps the cache across hops).
    ///
    /// The output is aligned with the input; the frame analysis/synthesis is the
    /// same `torch.stft(center=True)` / `torch.istft` pair used by the reference.
    pub fn process_wav(&mut self, wav: &[f32]) -> Vec<f32> {
        let (stft, spec) = match self {
            Engine::D { model, .. } => (model.stft.clone(), model.stft.stft(wav)),
            Engine::A05 { model, .. } => (model.stft.clone(), model.stft.stft(wav)),
        };
        let bins = spec.bins;
        let mut out = crate::stft::Spectrogram {
            frames: spec.frames,
            bins,
            data: vec![0.0f32; spec.data.len()],
        };
        for t in 0..spec.frames {
            let frame = &spec.data[t * 2 * bins..(t + 1) * 2 * bins];
            let y = self.process_hop(frame);
            out.data[t * 2 * bins..(t + 1) * 2 * bins].copy_from_slice(&y);
        }
        stft.istft(&out, wav.len())
    }

    /// Whole-file **offline** path (no streaming cache, one batched pass).
    /// Numerically equivalent to [`Engine::process_wav`] up to ~1e-7.
    pub fn process_wav_offline(&self, wav: &[f32]) -> Vec<f32> {
        match self {
            Engine::D { model, .. } => model.forward(wav),
            Engine::A05 { model, .. } => model.forward(wav),
        }
    }
}
