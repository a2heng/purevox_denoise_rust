//! `purevox_denoise_rust` — pure-Rust (f32) inference for the PureVox speech
//! denoiser. No ONNX, no ONNX Runtime, no C/C++.
//!
//! The crate implements the whole chain: STFT → log-magnitude → ERB banding →
//! encoder → dual-path GRU bottleneck → decoder → band-domain mask → ERB
//! expansion → complex-spectrum multiply → iSTFT/OLA.
//!
//! # Quick start
//!
//! ```no_run
//! use purevox_denoise_rust::{Engine, models_dir};
//!
//! // One call per 10 ms hop: complex spectrum in, complex spectrum out.
//! let mut engine = Engine::load(models_dir().join("purevox_denoise_202605a")).unwrap();
//! let spec = vec![0.0f32; engine.spec_len()];      // 2 × 481 bins
//! let out = engine.process_hop(&spec);
//! ```
//!
//! For file-to-file use see the `purevox` binary; for a self-check against the
//! bundled golden vectors see the `purevox_verify` binary.
//!
//! # Models
//!
//! Four presets ship in `models/` (see [`PRESETS`]). They come in two
//! architecture families:
//!
//! * **05 family** ([`Denoiser05`]) — grouped 1×1 → BN → SiLU → channel shuffle →
//!   causal depthwise conv → BN → SiLU → time/frequency attention
//!   (`ta_gru` + frequency BiGRU). Mask is `sigmoid` in the band domain.
//! * **11d family** ([`Denoiser`]) — 1×1 → BN → h-swish → causal depthwise conv →
//!   BN → FIR gate → residual, then `N` dual-path GRUs and an optional
//!   four-branch bottleneck memory. Mask is `sigmoid` in the bin domain.
//!
//! Both families are streaming: all state lives in a flat `cache` vector whose
//! layout is documented in `models/<name>.json` and can be inspected or
//! restored through [`Engine::cache_flat`] / [`Engine::set_cache_flat`].

pub mod block;
pub mod dpgrnn;
pub mod engine;
pub mod erb;
pub mod gru;
pub mod memory;
pub mod meta;
pub mod model;
pub mod model05;
pub mod net;
pub mod ops;
pub mod simd;
pub mod stft;
pub mod stream;
pub mod wavio;
pub mod weights;

pub use block::{CTfa, GateBlock};
pub use dpgrnn::DpGrnn;
pub use engine::Engine;
pub use erb::Erb;
pub use gru::GruNet;
pub use memory::BottleneckMemory;
pub use meta::Meta;
pub use model::Denoiser;
pub use model05::{Cache05, Denoiser05};
pub use net::{Decoder, Encoder};
pub use stft::{Spectrogram, Stft};
pub use stream::{HopTrace, StreamCache};
pub use weights::{WeightError, Weights};

use std::path::PathBuf;

/// Model presets bundled in `models/`.
///
/// Each name maps to `<name>.safetensors` (weights), `<name>.json` (architecture
/// and cache layout) and `<name>.golden.safetensors` (parity fixtures).
pub const PRESETS: [&str; 4] = [
    "purevox_denoise_202605a",
    "purevox_denoise_202605b",
    "purevox_denoise_202611d",
    "purevox_denoise_202611f",
];

/// Directory that ships the presets (resolved at compile time from
/// `CARGO_MANIFEST_DIR`). Copy `models/` next to your binary, or pass your own
/// path to [`Engine::load`].
pub fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}
