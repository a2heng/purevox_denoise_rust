//! End-to-end parity against the golden vectors shipped in `models/`.
//!
//! Each preset directory carries `golden.safetensors` with one second of input
//! audio, the reference `out_wav`, and the first 8 streaming hops. The test
//! asserts:
//!
//! * whole-file offline output  ≥ 90 dB SNR
//! * every streaming hop        ≥ 90 dB SNR
//! * the flat cache after a hop ≥ 90 dB SNR
//!
//! Presets whose directory is missing are skipped, so the repository keeps
//! working while only a subset of models is present.

use purevox_denoise_rust::{Engine, PRESETS};
use safetensors::SafeTensors;

fn get<'a>(st: &'a SafeTensors<'a>, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let t = st.tensor(name).ok()?;
    let data = t
        .data()
        .chunks_exact(4)
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

fn check(tag: &str, got: &[f32], want: &[f32], min: f64) {
    assert_eq!(got.len(), want.len(), "{tag}: length mismatch");
    let s = snr(want, got);
    println!("{tag:<16} {s:7.1} dB");
    assert!(s > min, "{tag}: {s:.1} dB < {min}");
}

#[test]
fn presets_match_golden() {
    let mut checked = 0;
    for name in PRESETS {
        let dir = purevox_denoise_rust::models_dir().join(name);
        let golden = dir.join("golden.safetensors");
        if !golden.exists() {
            println!("skip {name} (no golden)");
            continue;
        }
        let bytes = std::fs::read(&golden).unwrap();
        let g = SafeTensors::deserialize(&bytes).unwrap();
        let (_, wav) = get(&g, "wav").expect("golden: wav");
        let (hs, hop_spec) = get(&g, "hop_spec").expect("golden: hop_spec");
        let (_, hop_out) = get(&g, "hop_out").expect("golden: hop_out");
        let (hcs, hop_cache) = get(&g, "hop_cache_out").expect("golden: hop_cache_out");

        let mut engine = Engine::load(&dir).expect("load preset");
        assert_eq!(engine.cache_len(), hcs[1], "{name}: cache width");

        if let Some((_, want)) = get(&g, "out_wav") {
            let out = engine.process_wav_offline(&wav);
            check(&format!("{name}/offline"), &out, &want, 90.0);
        }

        let bins = hs[2];
        for h in 0..hs[0] {
            let frame = &hop_spec[h * 2 * bins..(h + 1) * 2 * bins];
            let out = engine.process_hop(frame);
            check(&format!("{name}/hop{h}"), &out, &hop_out[h * 2 * bins..(h + 1) * 2 * bins], 90.0);
            let flat = engine.cache_flat();
            check(&format!("{name}/cache{h}"), &flat, &hop_cache[h * hcs[1]..(h + 1) * hcs[1]], 90.0);
        }

        // Streaming and offline paths must agree on the same audio.
        let mut engine = Engine::load(&dir).unwrap();
        let stream = engine.process_wav(&wav);
        let offline = engine.process_wav_offline(&wav);
        let s = snr(&offline, &stream);
        println!("{name:<28} stream vs offline {s:7.1} dB");
        assert!(s > 120.0, "{name}: stream vs offline {s:.1} dB");

        checked += 1;
    }
    assert!(checked > 0, "no presets found under models/");
}
