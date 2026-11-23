//! `purevox` — file-to-file denoiser.
//!
//! ```text
//! purevox --list
//! purevox --model models/purevox_denoise_202605a --in noisy.wav --out clean.wav
//! purevox --model models/purevox_denoise_202605a --in noisy.wav --out clean.wav --offline
//! ```
//!
//! `--model` accepts either a preset name (resolved under `models/`) or a path to
//! a model directory. Input must be mono; multi-channel input is downmixed.

use std::path::PathBuf;

use purevox_denoise_rust::wavio::{read_wav, write_wav};
use purevox_denoise_rust::{Engine, PRESETS};

fn resolve_model(arg: &str) -> PathBuf {
    let p = PathBuf::from(arg);
    if p.is_dir() {
        p
    } else {
        purevox_denoise_rust::models_dir().join(arg)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = String::from(PRESETS[0]);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut offline = false;
    let mut ctrl: Option<(f32, f32)> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => {
                println!("presets (in models/):");
                for p in PRESETS {
                    let d = purevox_denoise_rust::models_dir().join(p);
                    let ok = if d.is_dir() { "ok" } else { "missing" };
                    println!("  {p:<28} {ok}");
                }
                return;
            }
            "--model" => { model = args[i + 1].clone(); i += 2; }
            "--in" => { input = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--out" => { output = Some(PathBuf::from(&args[i + 1])); i += 2; }
            "--offline" => { offline = true; i += 1; }
            "--floor-db" => { ctrl = Some((args[i + 1].parse().expect("--floor-db expects a number"), 1.0)); i += 2; }
            "-h" | "--help" => {
                println!("usage: purevox [--list] [--model <preset|dir>] --in <wav> --out <wav> [--offline] [--floor-db <dB>]");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let input = input.unwrap_or_else(|| {
        eprintln!("--in is required (see --help)");
        std::process::exit(2);
    });
    let output = output.unwrap_or_else(|| {
        eprintln!("--out is required (see --help)");
        std::process::exit(2);
    });

    let dir = resolve_model(&model);
    let mut engine = match Engine::load(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot load {}: {e}", dir.display());
            std::process::exit(1);
        }
    };
    let (wav, sr) = read_wav(&input);
    if sr != engine.sample_rate() {
        eprintln!("warning: input is {sr} Hz, model expects {} Hz", engine.sample_rate());
    }

    let t0 = std::time::Instant::now();
    let out = if offline {
        engine.process_wav_offline(&wav)
    } else if let Some(c) = ctrl {
        process_with_ctrl(&mut engine, &wav, c)
    } else {
        engine.process_wav(&wav)
    };
    let wall = t0.elapsed();
    write_wav(&output, &out, sr);

    let audio_ms = wav.len() as f64 / sr as f64 * 1000.0;
    println!("model {}  in {} samples @ {sr} Hz ({audio_ms:.0} ms)",
             engine.name(), wav.len());
    println!("wall {:.1} ms  RTF {:.3}", wall.as_secs_f64() * 1000.0,
             wall.as_secs_f64() * 1000.0 / audio_ms);
    println!("-> {}", output.display());
}

/// Same as `process_wav` but with an explicit mask-depth control.
fn process_with_ctrl(engine: &mut Engine, wav: &[f32], ctrl: (f32, f32)) -> Vec<f32> {
    let stft = purevox_denoise_rust::Stft::new(960, 480);
    let spec = stft.stft(wav);
    let bins = spec.bins;
    let mut out = purevox_denoise_rust::Spectrogram {
        frames: spec.frames,
        bins,
        data: vec![0.0f32; spec.data.len()],
    };
    for t in 0..spec.frames {
        let frame = &spec.data[t * 2 * bins..(t + 1) * 2 * bins];
        let y = engine.process_hop_with_ctrl(frame, ctrl);
        out.data[t * 2 * bins..(t + 1) * 2 * bins].copy_from_slice(&y);
    }
    stft.istft(&out, wav.len())
}
