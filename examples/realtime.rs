//! Real-time usage: analyse, run one hop at a time through [`Engine`], synthesise.
//!
//! ```text
//! cargo run --release --example realtime -- models/purevox_denoise_202605a input.wav
//! ```
//!
//! Prints the per-hop latency distribution so you can check the 10 ms budget.

use purevox_denoise_rust::wavio::read_wav;
use purevox_denoise_rust::{Engine, Stft};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .unwrap_or_else(|| "models/purevox_denoise_202605a".to_string());
    let wav_path = args.next().expect("usage: realtime <model-dir> <wav>");

    let (wav, sr) = read_wav(&wav_path);
    let mut engine = Engine::load(&model).expect("load model");
    println!(
        "model {}  cache {}  spec {}  {} samples @ {sr} Hz",
        engine.name(),
        engine.cache_len(),
        engine.spec_len(),
        wav.len()
    );

    // Analysis: identical to torch.stft(center=True) with a periodic Hann window.
    let stft = Stft::new(960, 480);
    let spec = stft.stft(&wav);
    let bins = spec.bins;
    let mut out = purevox_denoise_rust::Spectrogram {
        frames: spec.frames,
        bins,
        data: vec![0.0f32; spec.data.len()],
    };

    let mut times = Vec::with_capacity(spec.frames);
    for t in 0..spec.frames {
        let frame = &spec.data[t * 2 * bins..(t + 1) * 2 * bins];
        let t0 = std::time::Instant::now();
        let y = engine.process_hop(frame);
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        out.data[t * 2 * bins..(t + 1) * 2 * bins].copy_from_slice(&y);
    }
    let y = stft.istft(&out, wav.len());

    let audio_ms = wav.len() as f64 / sr as f64 * 1000.0;
    println!(
        "audio {audio_ms:.0} ms   hops {}   RTF {:.3}",
        times.len(),
        times.iter().sum::<f64>() / audio_ms
    );
    let mut v = times;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "hop p50 {:.3} / p90 {:.3} / p99 {:.3} / max {:.3} ms  (budget 10 ms)",
        v[v.len() / 2],
        v[v.len() * 9 / 10],
        v[v.len() * 99 / 100],
        v[v.len() - 1]
    );
    println!("output {} samples, first 4: {:.5?}", y.len(), &y[..4]);
}
