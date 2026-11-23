
use std::fs;
use std::path::Path;

use crate::dpgrnn::DpGrnn;
use crate::erb::Erb;
use crate::memory::BottleneckMemory;
use crate::meta::Meta;
use crate::net::{Decoder, Encoder};
use crate::stft::{Spectrogram, Stft};
use crate::weights::{WeightError, Weights};

pub struct Denoiser {
    pub meta: Meta,
    pub stft: Stft,
    pub erb: Erb,
    pub encoder: Encoder,
    pub dps: Vec<DpGrnn>,
    pub mem: BottleneckMemory,
    pub decoder: Decoder,
}

impl Denoiser {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, WeightError> {
        let dir = dir.as_ref();
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?)
            .expect("meta.json 解析失败");
        let w = Weights::load(dir.join("weights.safetensors"))?;
        let erb = Erb::from_weights(&w, meta.erb.low, meta.erb.high, meta.erb.n_freq)?;
        let encoder = Encoder::from_weights(&w, &meta)?;
        let c = meta.arch.enc_ch[meta.arch.enc_ch.len() - 1];
        let f = meta.arch.enc_out_w[meta.arch.enc_out_w.len() - 1];
        let list_prefix = w.map.contains_key("dpgrnns.0.intra_dw.weight");
        let mut dps = Vec::with_capacity(meta.arch.n_dp);
        for k in 0..meta.arch.n_dp {
            let prefix = if list_prefix { format!("dpgrnns.{k}") } else { format!("dpgrnn{k}") };
            dps.push(DpGrnn::from_weights(&w, &prefix, c, f, meta.arch.dp_hid)?);
        }
        let mem = BottleneckMemory::from_weights(&w, "mem", c, meta.arch.mem_hid, meta.arch.mem_layers)?;
        let decoder = Decoder::from_weights(&w, &meta)?;
        let stft = Stft::new(meta.stft.n_fft, meta.stft.hop);
        Ok(Self { meta, stft, erb, encoder, dps, mem, decoder })
    }

    pub fn features(&self, spec: &Spectrogram) -> Vec<f32> {
        let logmag = Erb::logmag(spec);
        self.erb.bm(&logmag, spec.frames)
    }

    pub fn mask(&self, spec: &Spectrogram) -> MaskResult {
        let feat = self.features(spec);
        let t = spec.frames;
        let (top, skips) = self.encoder.forward(&feat, t);
        let c = self.meta.arch.enc_ch[self.meta.arch.enc_ch.len() - 1];
        let f = self.meta.arch.enc_out_w[self.meta.arch.enc_out_w.len() - 1];
        let _ = c;
        let mut x = top.clone();
        for dp in &self.dps {
            x = dp.forward(&x, t, f);
        }
        let mut pre = vec![0.0f32; x.len()];
        for i in 0..pre.len() { pre[i] = x[i] + top[i]; }
        let mem = self.mem.forward(&pre, t, f);
        let logit = self.decoder.forward(&mem, t, &skips);
        let u = self.erb.bs(&logit, t);
        let mask_bin = mask_act(&u, self.meta.mask.floor_db, self.meta.mask.depth_gain);
        let mask_band = mask_act(&logit, self.meta.mask.floor_db, self.meta.mask.depth_gain);
        MaskResult { feat, logit, mask_band, mask_bin }
    }

    pub fn forward(&self, wav: &[f32]) -> Vec<f32> {
        let spec = self.stft.stft(wav);
        let m = self.mask(&spec);
        let mut masked = Spectrogram { frames: spec.frames, bins: spec.bins,
                                       data: vec![0.0f32; spec.data.len()] };
        for t in 0..spec.frames {
            for b in 0..spec.bins {
                let g = m.mask_bin[t * spec.bins + b];
                masked.data[(t * 2) * spec.bins + b] = spec.at(t, 0, b) * g;
                masked.data[(t * 2 + 1) * spec.bins + b] = spec.at(t, 1, b) * g;
            }
        }
        self.stft.istft(&masked, wav.len())
    }
}

pub struct MaskResult {
    pub feat: Vec<f32>,
    pub logit: Vec<f32>,
    pub mask_band: Vec<f32>,
    pub mask_bin: Vec<f32>,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

pub fn mask_act(u: &[f32], floor_db: Option<f32>, depth_gain: f32) -> Vec<f32> {
    if floor_db.is_none() && depth_gain == 1.0 {
        return u.iter().map(|&v| sigmoid(v)).collect();
    }
    u.iter().map(|&v| {
        let att = depth_gain * 20.0 * sigmoid(v).max(1e-30).log10();
        let att = match floor_db { Some(a) => att.max(a), None => att };
        10f32.powf(att / 20.0)
    }).collect()
}

impl Denoiser {
    pub fn stream_wav(&self, wav: &[f32], ctrl: Option<(f32, f32)>) -> (Vec<f32>, Vec<std::time::Duration>) {
        let spec = self.stft.stft(wav);
        let bins = spec.bins;
        let mut cache = crate::stream::StreamCache::zeros(&self.meta);
        let mut out_spec = Spectrogram { frames: spec.frames, bins, data: vec![0.0f32; spec.data.len()] };
        let mut times = Vec::with_capacity(spec.frames);
        for t in 0..spec.frames {
            let frame = &spec.data[t * 2 * bins..(t + 1) * 2 * bins];
            let t0 = std::time::Instant::now();
            let y = self.stream_hop(frame, &mut cache, ctrl);
            times.push(t0.elapsed());
            out_spec.data[t * 2 * bins..(t + 1) * 2 * bins].copy_from_slice(&y);
        }
        (self.stft.istft(&out_spec, wav.len()), times)
    }
}
