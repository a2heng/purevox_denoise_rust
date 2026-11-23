//! `purevox_verify` — parity self-check against the golden vectors shipped in
//! `models/<name>.golden.safetensors`.
//!
//! ```text
//! purevox_verify                  # all presets found in models/
//! purevox_verify --model <dir|preset>
//! ```
//!
//! Checks, per model:
//! * `feat`      — ERB front-end (needs the reference log-magnitude features)
//! * `out_wav`   — whole offline chain (wav → wav)
//! * `hop{i}`    — each of the first 8 streaming hops
//! * `hop cache` — the flat cache after each hop
//!
//! Exit code is non-zero if any check falls below its threshold.

use std::path::PathBuf;

use purevox_denoise_rust::{Engine, PRESETS};
use safetensors::SafeTensors;

fn get(st: &SafeTensors<'_>, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let t = st.tensor(name).ok()?;
    let data = t.data().chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((t.shape().to_vec(), data))
}

fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &t) in reference.iter().zip(test) {
        sig += (r as f64) * (r as f64);
        let d = (r - t) as f64;
        err += d * d;
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

struct Report {
    worst: f64,
    fails: Vec<String>,
}

impl Report {
    fn check(&mut self, tag: &str, got: &[f32], want: &[f32], min: f64) {
        if got.len() != want.len() {
            let msg = format!("{tag}: length {} vs {}", got.len(), want.len());
            println!("{tag:<14} length mismatch {} vs {}", got.len(), want.len());
            self.fails.push(msg);
            return;
        }
        let s = snr(want, got);
        println!("{tag:<14} {s:7.1} dB");
        if s < min {
            self.fails.push(format!("{tag}: {s:.1} dB < {min}"));
        }
        self.worst = self.worst.min(s);
    }
}

fn verify(dir: &PathBuf) -> Report {
    let mut rep = Report { worst: f64::INFINITY, fails: Vec::new() };
    let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    println!("== {name} ==");

    let mut engine = match Engine::load(dir) {
        Ok(e) => e,
        Err(e) => {
            rep.fails.push(format!("load: {e}"));
            return rep;
        }
    };
    let golden_path = dir.join("golden.safetensors");
    let bytes = match std::fs::read(&golden_path) {
        Ok(b) => b,
        Err(_) => {
            println!("(no golden file at {}, skipping)", golden_path.display());
            return rep;
        }
    };
    let g = SafeTensors::deserialize(&bytes).unwrap();
    let (_, wav) = get(&g, "wav").expect("golden: wav");
    let _ = get(&g, "spec");

    // Offline chain (wav → wav).
    if let Some((_, want)) = get(&g, "out_wav") {
        let out = engine.process_wav_offline(&wav);
        rep.check("out_wav", &out, &want, 90.0);
    }

    // Streaming hops + cache.
    let (hs, hop_spec) = get(&g, "hop_spec").expect("golden: hop_spec");
    let (_, hop_out) = get(&g, "hop_out").expect("golden: hop_out");
    let (hcs, hop_cache) = get(&g, "hop_cache_out").expect("golden: hop_cache_out");
    let bins = hs[2];
    let cache_len = hcs[1];
    if engine.cache_len() != cache_len {
        rep.fails.push(format!("cache length {} vs {cache_len}", engine.cache_len()));
    }
    let mut worst_cache = f64::INFINITY;
    for h in 0..hs[0] {
        let frame = &hop_spec[h * 2 * bins..(h + 1) * 2 * bins];
        let out = engine.process_hop(frame);
        rep.check(&format!("hop{h}"), &out, &hop_out[h * 2 * bins..(h + 1) * 2 * bins], 90.0);
        let flat = engine.cache_flat();
        let want = &hop_cache[h * cache_len..(h + 1) * cache_len];
        let cs = snr(want, &flat);
        worst_cache = worst_cache.min(cs);
        rep.worst = rep.worst.min(cs);
    }
    println!("hop cache      {worst_cache:7.1} dB");
    if worst_cache < 90.0 {
        rep.fails.push(format!("hop cache: {worst_cache:.1} dB < 90"));
    }
    rep
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let explicit = args.windows(2).find(|w| w[0] == "--model").map(|w| w[1].clone());
    let mut targets: Vec<PathBuf> = Vec::new();
    match explicit {
        Some(m) => {
            let p = PathBuf::from(&m);
            targets.push(if p.is_dir() { p } else { purevox_denoise_rust::models_dir().join(&m) });
        }
        None => {
            for p in PRESETS {
                let d = purevox_denoise_rust::models_dir().join(p);
                if d.is_dir() {
                    targets.push(d);
                }
            }
        }
    }
    let mut ok = true;
    for dir in &targets {
        let rep = verify(dir);
        println!("worst {:.1} dB  {}", rep.worst, if rep.fails.is_empty() { "OK\n" } else { "FAILED\n" });
        for f in &rep.fails {
            println!("  - {f}");
        }
        ok &= rep.fails.is_empty();
    }
    if !ok {
        std::process::exit(1);
    }
}
