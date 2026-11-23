
use std::fs;
use std::path::Path;

use crate::erb::Erb;
use crate::gru::GruNet;
use crate::meta::{LayerCfg, Meta, PlanEntry};
use crate::ops::conv::{Conv, Conv2d, ConvTranspose2d};
use crate::ops::linear::Linear;
use crate::block::Bn2d;
use crate::ops::norm::LayerNorm;
use crate::stft::{Spectrogram, Stft};
use crate::weights::{WeightError, Weights};

fn shuffle_channels(x: &[f32], ch: usize, t: usize, w: usize) -> Vec<f32> {
    let half = ch / 2;
    let mut out = vec![0.0f32; x.len()];
    for c in 0..ch {
        let src = (c % 2) * half + c / 2;
        out[c * t * w..(c + 1) * t * w].copy_from_slice(&x[src * t * w..(src + 1) * t * w]);
    }
    out
}

pub struct Fa {
    pub nfreq: usize,
    pub r: usize,
    pub h_groups: usize,
    pub padded: usize,
    pub gru: GruNet,   // r → r，双向
    pub fc: Linear,    // 2r → r
}

impl Fa {
    pub fn from_weights(w: &Weights, prefix: &str, nfreq: usize, r: usize) -> Result<Self, WeightError> {
        let h_groups = nfreq.div_ceil(r);
        Ok(Self { nfreq, r, h_groups, padded: h_groups * r,
                  gru: GruNet::from_weights(w, &format!("{prefix}.gru"), 1, true)?,
                  fc: Linear::from_weights(w, &format!("{prefix}.fc"))? })
    }

    pub fn forward(&self, x: &[f32], c: usize, t: usize, f: usize) -> Vec<f32> {
        assert_eq!(f, self.nfreq);
        let r = self.r;
        let hg = self.h_groups;
        let mut g = vec![0.0f32; t * self.padded];
        for ti in 0..t {
            for fi in 0..f {
                let mut acc = 0.0f32;
                for ci in 0..c { let v = x[(ci * t + ti) * f + fi]; acc += v * v; }
                g[ti * self.padded + fi] = acc / c as f32;
            }
        }
        let seq = t * hg;
        let h0 = vec![0.0f32; 2 * t * r];
        let (y, _) = self.gru.forward(&g, t, hg, &h0);
        let y = self.fc.forward(&y, seq);
        let mut out = vec![0.0f32; t * f];
        for ti in 0..t {
            out[ti * f..(ti + 1) * f].copy_from_slice(&y[ti * self.padded..ti * self.padded + f]);
        }
        crate::simd::sigmoid_slice(&mut out);
        out
    }
}

pub struct Ctfa05 {
    pub ta_gru: GruNet,   // C → 2C，单向
    pub ta_fc: Linear,    // 2C → C
    pub fa: Fa,
}

impl Ctfa05 {
    pub fn from_weights(w: &Weights, prefix: &str, width: usize) -> Result<Self, WeightError> {
        Ok(Self {
            ta_gru: GruNet::from_weights(w, &format!("{prefix}.ta_gru"), 1, false)?,
            ta_fc: Linear::from_weights(w, &format!("{prefix}.ta_fc"))?,
            fa: Fa::from_weights(w, &format!("{prefix}.fa"), width, 4)?,
        })
    }

    pub fn channels(&self) -> usize { self.ta_gru.input }

    pub fn forward(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        self.forward_trace(x, t, f).out
    }

    pub fn forward_trace(&self, x: &[f32], t: usize, f: usize) -> Tfa05Trace {
        let c = self.channels();
        // zt = mean over F of x² → (T,C)
        let mut zt = vec![0.0f32; t * c];
        for ti in 0..t {
            for ci in 0..c {
                let row = &x[(ci * t + ti) * f..(ci * t + ti) * f + f];
                let mut acc = 0.0f32;
                for &v in row { acc += v * v; }
                zt[ti * c + ci] = acc / f as f32;
            }
        }
        let h0 = vec![0.0f32; 2 * c];
        let (at, _) = self.ta_gru.forward(&zt, 1, t, &h0);
        let at = self.ta_fc.forward(&at, t);
        let af = self.fa.forward(x, c, t, f);
        let mut at_sig = at.clone();
        crate::simd::sigmoid_slice(&mut at_sig);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for ti in 0..t {
                let g = at_sig[ti * c + ci];
                for fi in 0..f {
                    out[(ci * t + ti) * f + fi] = g * x[(ci * t + ti) * f + fi] * af[ti * f + fi];
                }
            }
        }
        Tfa05Trace { zt, at, at_sig, af, out }
    }

    pub fn stream(&self, x: &[f32], f: usize, state: &mut [f32]) -> Vec<f32> {
        let c = self.channels();
        assert_eq!(x.len(), c * f);
        let mut zt = vec![0.0f32; c];
        for ci in 0..c {
            let mut acc = 0.0f32;
            for fi in 0..f { let v = x[ci * f + fi]; acc += v * v; }
            zt[ci] = acc / f as f32;
        }
        let at = self.ta_gru.step(&zt, state);
        let mut at = self.ta_fc.forward(&at, 1);
        crate::simd::sigmoid_slice(&mut at);          // 与离线同一内核 ⇒ 逐位一致
        let af = self.fa.forward(x, c, 1, f);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            let g = at[ci];
            for fi in 0..f { out[ci * f + fi] = g * x[ci * f + fi] * af[fi]; }
        }
        out
    }
}

#[derive(Debug)]
pub struct XdwsTrace {
    pub pconv: Vec<f32>,   // 1×1 输出
    pub bn1: Vec<f32>,
    pub silu: Vec<f32>,
    pub shuf: Vec<f32>,    // 组卷积后的通道洗牌结果
    pub pad: Vec<f32>,     // 时间左补后的张量
    pub conv: Vec<f32>,    // dw/deconv 输出
    pub bn2: Vec<f32>,
    pub act: Vec<f32>,     // SiLU（末块是 Identity）
    pub tfa: Option<Tfa05Trace>,
    pub out: Vec<f32>,
}

#[derive(Debug)]
pub struct Tfa05Trace {
    pub zt: Vec<f32>,      // (T,C)
    pub at: Vec<f32>,      // (T,C) ta_fc 输出（未 sigmoid）
    pub at_sig: Vec<f32>,  // (T,C)
    pub af: Vec<f32>,      // (T,F)
    pub out: Vec<f32>,     // (C,T,F)
}

pub struct XdwsBlock {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kt: usize,
    pub in_width: usize,
    pub out_width: usize,
    pub groups: usize,
    pub pconv: Conv2d,
    pub pbn: Bn2d,
    pub conv: Conv,
    pub bn: Bn2d,
    pub act_out: bool,     // 末块为 false（Identity）
    pub tfa: Option<Ctfa05>,
}

impl XdwsBlock {
    pub fn from_weights(w: &Weights, prefix: &str, c: &LayerCfg, bn_eps: f32,
                        use_deconv: bool) -> Result<Self, WeightError> {
        let groups = c.groups.unwrap_or(1);
        let pconv = Conv2d::from_weights(w, &format!("{prefix}.pconv.0"), 0, 0, 1, 1, groups)?;
        let conv = if use_deconv {
            let mut k = ConvTranspose2d::from_weights(w, &format!("{prefix}.conv"), c.pad_t,
                                                      c.pad_w, 1, c.stride, c.out_ch)?;
            k.ph = c.pad_t;
            Conv::Transposed(k)
        } else {
            Conv::Direct(Conv2d::from_weights(w, &format!("{prefix}.conv"), 0, c.pad_w,
                                              1, c.stride, c.out_ch)?)
        };
        let tfa = if c.tfa {
            Some(Ctfa05::from_weights(w, &format!("{prefix}.tfa"), c.out_width)?)
        } else {
            None
        };
        Ok(Self {
            in_ch: pconv.in_ch, out_ch: pconv.out_ch, kt: c.kt,
            in_width: c.in_width, out_width: c.out_width, groups,
            pbn: Bn2d::from_weights(w, &format!("{prefix}.pconv.1"), bn_eps)?,
            bn: Bn2d::from_weights(w, &format!("{prefix}.bn"), bn_eps)?,
            pconv, conv, act_out: c.tfa, tfa,
        })
    }

    pub fn forward_trace(&self, x: &[f32], t: usize) -> XdwsTrace {
        assert_eq!(x.len(), self.in_ch * t * self.in_width);
        let pconv = self.pconv.forward(x, 1, t, self.in_width).0;
        let bn1 = self.pbn.forward(&pconv, self.out_ch, t, self.in_width);
        let mut silu_out = bn1.clone();
        crate::simd::silu_slice(&mut silu_out);
        let shuf = if self.groups == 2 {
            shuffle_channels(&silu_out, self.out_ch, t, self.in_width)
        } else {
            silu_out.clone()
        };
        let pad = self.kt - 1;
        let tp = t + pad;
        let w = self.in_width;
        let mut dp = vec![0.0f32; self.out_ch * tp * w];
        if pad > 0 {
            for c in 0..self.out_ch {
                dp[c * tp * w + pad * w..(c + 1) * tp * w]
                    .copy_from_slice(&shuf[c * t * w..(c + 1) * t * w]);
            }
        } else {
            dp.copy_from_slice(&shuf);
        }
        let (conv, _, ow) = self.conv.forward(&dp, 1, tp, w);
        let bn2 = self.bn.forward(&conv, self.out_ch, t, ow);
        let act: Vec<f32> = if self.act_out {
            let mut a = bn2.clone();
            crate::simd::silu_slice(&mut a);
            a
        } else { bn2.clone() };
        let (out, tfa) = match &self.tfa {
            Some(tfa) => { let tr = tfa.forward_trace(&act, t, ow); (tr.out.clone(), Some(tr)) }
            None => (act.clone(), None),
        };
        XdwsTrace { pconv, bn1, silu: silu_out, shuf, pad: dp, conv, bn2, act, tfa, out }
    }

    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        self.forward_trace(x, t).out
    }

    pub fn stream(&self, x: &[f32], conv_cache: Option<&[f32]>, tfa_state: Option<&mut [f32]>)
        -> (Vec<f32>, Option<Vec<f32>>)
    {
        let h = {
            let h = self.pconv.forward(x, 1, 1, self.in_width).0;
            let h = self.pbn.forward(&h, self.out_ch, 1, self.in_width);
            let mut h = h;
            crate::simd::silu_slice(&mut h);
            if self.groups == 2 { shuffle_channels(&h, self.out_ch, 1, self.in_width) } else { h }
        };
        let (y, tp, new_cache) = if self.kt == 1 {
            (h.clone(), 1usize, None)
        } else {
            let frames = self.kt - 1;
            let cache = conv_cache.expect("kt>1 必须给 cache");
            assert_eq!(cache.len(), self.out_ch * frames * self.in_width);
            let mut p = vec![0.0f32; self.out_ch * self.kt * self.in_width];
            for c in 0..self.out_ch {
                p[c * self.kt * self.in_width..c * self.kt * self.in_width + frames * self.in_width]
                    .copy_from_slice(&cache[c * frames * self.in_width..(c + 1) * frames * self.in_width]);
                p[c * self.kt * self.in_width + frames * self.in_width..(c + 1) * self.kt * self.in_width]
                    .copy_from_slice(&h[c * self.in_width..(c + 1) * self.in_width]);
            }
            let mut nc = vec![0.0f32; self.out_ch * frames * self.in_width];
            for c in 0..self.out_ch {
                nc[c * frames * self.in_width..(c + 1) * frames * self.in_width]
                    .copy_from_slice(&p[c * self.kt * self.in_width + self.in_width..(c + 1) * self.kt * self.in_width]);
            }
            (p, self.kt, Some(nc))
        };
        let (mut y, _, ow) = self.conv.forward(&y, 1, tp, self.in_width);
        assert_eq!(ow, self.out_width, "流式频率宽度不符");
        y = self.bn.forward(&y, self.out_ch, 1, ow);
        if self.act_out { crate::simd::silu_slice(&mut y); }
        match &self.tfa {
            Some(tfa) => (tfa.stream(&y, ow, tfa_state.expect("tfa state")), new_cache),
            None => (y, new_cache),
        }
    }
}

pub struct DpGrnn05 {
    pub input: usize,
    pub width: usize,
    pub hidden: usize,
    pub intra_rnn: GruNet,   // C → hidden/2，双向
    pub intra_fc: Linear,    // hidden → C
    pub intra_ln: LayerNorm, // (width, C)，eps 1e-8
    pub inter_rnn: GruNet,   // C → hidden，单向
    pub inter_fc: Linear,
    pub inter_ln: LayerNorm,
}

impl DpGrnn05 {
    pub fn from_weights(w: &Weights, prefix: &str, input: usize, width: usize) -> Result<Self, WeightError> {
        let hidden = w.get(&format!("{prefix}.intra_fc.weight"))?.shape[0];
        Ok(Self {
            input, width, hidden,
            intra_rnn: GruNet::from_weights(w, &format!("{prefix}.intra_rnn"), 1, true)?,
            intra_fc: Linear::from_weights(w, &format!("{prefix}.intra_fc"))?,
            intra_ln: LayerNorm::from_weights(w, &format!("{prefix}.intra_ln"), 1e-8)?,
            inter_rnn: GruNet::from_weights(w, &format!("{prefix}.inter_rnn"), 1, false)?,
            inter_fc: Linear::from_weights(w, &format!("{prefix}.inter_fc"))?,
            inter_ln: LayerNorm::from_weights(w, &format!("{prefix}.inter_ln"), 1e-8)?,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.input;
        let h = self.hidden;
        let mut seq = vec![0.0f32; t * f * c]; // (T,F,C)
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f { seq[(ti * f + fi) * c + ci] = x[(ci * t + ti) * f + fi]; }
            }
        }
        let h0 = vec![0.0f32; 2 * t * (h / 2)];
        let (iy, _) = self.intra_rnn.forward(&seq, t, f, &h0);
        let iy = self.intra_fc.forward(&iy, t * f); // (T,F,C)
        let iy = self.intra_ln.forward(&iy);        // 逐 (T) 归一化 (F,C)
        let mut intra = vec![0.0f32; x.len()];
        for ti in 0..t {
            for fi in 0..f {
                for ci in 0..c {
                    let k = (ti * f + fi) * c + ci;
                    intra[(ci * t + ti) * f + fi] = x[(ci * t + ti) * f + fi] + iy[k];
                }
            }
        }
        let mut seq2 = vec![0.0f32; f * t * c];
        for fi in 0..f {
            for ti in 0..t {
                for ci in 0..c { seq2[(fi * t + ti) * c + ci] = intra[(ci * t + ti) * f + fi]; }
            }
        }
        let h0 = vec![0.0f32; f * h];
        let (ry, _) = self.inter_rnn.forward(&seq2, f, t, &h0);
        let ry = self.inter_fc.forward(&ry, f * t);
        let mut tl = vec![0.0f32; t * f * c];
        for fi in 0..f {
            for ti in 0..t {
                for ci in 0..c { tl[(ti * f + fi) * c + ci] = ry[(fi * t + ti) * c + ci]; }
            }
        }
        let ln = self.inter_ln.forward(&tl);
        let mut out = vec![0.0f32; x.len()];
        for ti in 0..t {
            for fi in 0..f {
                for ci in 0..c {
                    let k = (ti * f + fi) * c + ci;
                    out[(ci * t + ti) * f + fi] = intra[(ci * t + ti) * f + fi] + ln[k];
                }
            }
        }
        out
    }

    pub fn stream(&self, x: &[f32], f: usize, state: &mut [f32]) -> Vec<f32> {
        let c = self.input;
        let h = self.hidden;
        assert_eq!(state.len(), f * h);
        let mut seq = vec![0.0f32; f * c];
        for ci in 0..c {
            for fi in 0..f { seq[fi * c + ci] = x[ci * f + fi]; }
        }
        let h0 = vec![0.0f32; 2 * (h / 2)];
        let (iy, _) = self.intra_rnn.forward(&seq, 1, f, &h0);
        let iy = self.intra_fc.forward(&iy, f);
        let iy = self.intra_ln.forward(&iy);
        let mut intra = vec![0.0f32; x.len()];
        for ci in 0..c {
            for fi in 0..f { intra[ci * f + fi] = x[ci * f + fi] + iy[fi * c + ci]; }
        }
        let mut seq2 = vec![0.0f32; f * c];
        for fi in 0..f {
            for ci in 0..c { seq2[fi * c + ci] = intra[ci * f + fi]; }
        }
        let ry = self.inter_rnn.step(&seq2, state);
        let ry = self.inter_fc.forward(&ry, f);
        let ln = self.inter_ln.forward(&ry);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for fi in 0..f { out[ci * f + fi] = intra[ci * f + fi] + ln[fi * c + ci]; }
        }
        out
    }
}

pub struct Encoder05 { pub layers: Vec<XdwsBlock> }
pub struct Decoder05 { pub layers: Vec<XdwsBlock> }

impl Encoder05 {
    pub fn from_weights(w: &Weights, meta: &Meta) -> Result<Self, WeightError> {
        let layers = meta.enc.iter().enumerate()
            .map(|(i, c)| XdwsBlock::from_weights(w, &format!("encoder.layers.{i}"), c, meta.arch.bn_eps, false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }
    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut h = x.to_vec();
        let mut skips = Vec::with_capacity(self.layers.len());
        for blk in &self.layers {
            h = blk.forward(&h, t);
            skips.push(h.clone());
        }
        (h, skips)
    }
}

impl Decoder05 {
    pub fn from_weights(w: &Weights, meta: &Meta) -> Result<Self, WeightError> {
        let layers = meta.dec.iter().enumerate()
            .map(|(i, c)| XdwsBlock::from_weights(w, &format!("decoder.layers.{i}"), c, meta.arch.bn_eps, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &[f32], t: usize, skips: &[Vec<f32>]) -> Vec<f32> {
        self.forward_stages(x, t, skips).0
    }

    pub fn forward_stages(&self, x: &[f32], t: usize, skips: &[Vec<f32>])
        -> (Vec<f32>, Vec<Vec<f32>>)
    {
        let n = self.layers.len();
        let mut h = x.to_vec();
        let mut w = self.layers[0].in_width;
        let mut stages = Vec::with_capacity(n);
        for (i, blk) in self.layers.iter().enumerate() {
            let skip = &skips[n - 1 - i];
            let target = skip_width(blk, skip, t);
            let fitted = fit_width(&h, self.channels_in(i), t, w, target);
            let mut sum = fitted;
            for (a, b) in sum.iter_mut().zip(skip) { *a += b; }
            h = blk.forward(&sum, t);
            w = blk.out_width;
            stages.push(h.clone());
        }
        (h, stages)
    }

    fn channels_in(&self, i: usize) -> usize { self.layers[i].in_ch }
}

fn skip_width(blk: &XdwsBlock, skip: &[f32], t: usize) -> usize {
    let ch = blk.in_ch;
    assert!(skip.len() % (t * ch) == 0, "skip 形状不符");
    skip.len() / (t * ch)
}

fn fit_width(x: &[f32], ch: usize, t: usize, w: usize, target: usize) -> Vec<f32> {
    if w == target { return x.to_vec(); }
    let mut out = vec![0.0f32; ch * t * target];
    let keep = w.min(target);
    for c in 0..ch {
        for ti in 0..t {
            out[(c * t + ti) * target..(c * t + ti) * target + keep]
                .copy_from_slice(&x[(c * t + ti) * w..(c * t + ti) * w + keep]);
        }
    }
    out
}

pub struct Denoiser05 {
    pub meta: Meta,
    pub stft: Stft,
    pub erb: Erb,
    pub encoder: Encoder05,
    pub dp0: DpGrnn05,
    pub dp1: DpGrnn05,
    pub decoder: Decoder05,
}

pub struct Cache05 {
    pub enc: Vec<Option<Vec<f32>>>,
    pub dec: Vec<Option<Vec<f32>>>,
    pub tfa: Vec<Vec<f32>>,
    pub inter: Vec<Vec<f32>>,
}

impl Cache05 {
    pub fn zeros(meta: &Meta) -> Self {
        let mk = |plan: &Vec<PlanEntry>| -> Vec<Option<Vec<f32>>> {
            plan.iter().map(|e| e.shape.as_ref().map(|s| vec![0.0f32; s.iter().product()])).collect()
        };
        Self {
            enc: mk(&meta.cache.enc),
            dec: mk(&meta.cache.dec),
            tfa: meta.cache.tfa.iter().map(|e| vec![0.0f32; e.shape.as_ref().unwrap().iter().product()]).collect(),
            inter: meta.cache.inter.iter().map(|e| vec![0.0f32; e.shape.as_ref().unwrap().iter().product()]).collect(),
        }
    }

    pub fn total(&self) -> usize {
        let sz = |v: &Vec<Option<Vec<f32>>>| v.iter().flatten().map(|x| x.len()).sum::<usize>();
        sz(&self.enc) + sz(&self.dec)
            + self.tfa.iter().map(|v| v.len()).sum::<usize>()
            + self.inter.iter().map(|v| v.len()).sum::<usize>()
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total());
        for v in self.enc.iter().chain(self.dec.iter()).flatten() { out.extend_from_slice(v); }
        for v in &self.tfa { out.extend_from_slice(v); }
        for v in &self.inter { out.extend_from_slice(v); }
        out
    }

    pub fn from_flat(&mut self, flat: &[f32]) {
        assert_eq!(flat.len(), self.total());
        let mut o = 0;
        for v in self.enc.iter_mut().chain(self.dec.iter_mut()).flatten() {
            let n = v.len(); v.copy_from_slice(&flat[o..o + n]); o += n;
        }
        for v in self.tfa.iter_mut() { let n = v.len(); v.copy_from_slice(&flat[o..o + n]); o += n; }
        for v in self.inter.iter_mut() { let n = v.len(); v.copy_from_slice(&flat[o..o + n]); o += n; }
    }
}

impl Denoiser05 {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, WeightError> {
        let dir = dir.as_ref();
        let meta: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?)
            .expect("meta.json 解析失败");
        let w = Weights::load(dir.join("weights.safetensors"))?;
        let erb = Erb::from_weights_named(&w, "erb.w_avg", "erb.w_up",
                                          meta.erb.low, meta.erb.high, meta.erb.n_freq)?;
        let encoder = Encoder05::from_weights(&w, &meta)?;
        let c = meta.arch.enc_ch[meta.arch.enc_ch.len() - 1];
        let f = meta.arch.enc_out_w[meta.arch.enc_out_w.len() - 1];
        let dp0 = DpGrnn05::from_weights(&w, "dpgrnn0", c, f)?;
        let dp1 = DpGrnn05::from_weights(&w, "dpgrnn1", c, f)?;
        let decoder = Decoder05::from_weights(&w, &meta)?;
        let stft = Stft::new(meta.stft.n_fft, meta.stft.hop);
        Ok(Self { meta, stft, erb, encoder, dp0, dp1, decoder })
    }

    pub fn features(&self, spec: &Spectrogram) -> Vec<f32> {
        let logmag = Erb::logmag(spec);
        self.erb.bm(&logmag, spec.frames)
    }

    pub fn mask(&self, spec: &Spectrogram) -> (Vec<f32>, Vec<f32>) {
        let feat = self.features(spec);
        let t = spec.frames;
        let (top, skips) = self.encoder.forward(&feat, t);
        let f = self.meta.arch.enc_out_w[self.meta.arch.enc_out_w.len() - 1];
        let y0 = self.dp0.forward(&top, t, f);
        let y1 = self.dp1.forward(&y0, t, f);
        let logit = self.decoder.forward(&y1, t, &skips);
        let mut mask_band = logit;
        crate::simd::sigmoid_slice(&mut mask_band);
        let mask_bin = self.erb.bs(&mask_band, t);
        (mask_bin, mask_band)
    }

    pub fn forward(&self, wav: &[f32]) -> Vec<f32> {
        let spec = self.stft.stft(wav);
        let (mask, _) = self.mask(&spec);
        self.apply_mask(&spec, &mask, wav.len())
    }

    fn apply_mask(&self, spec: &Spectrogram, mask: &[f32], len: usize) -> Vec<f32> {
        let bins = spec.bins;
        let mut out = Spectrogram { frames: spec.frames, bins, data: vec![0.0f32; spec.data.len()] };
        for t in 0..spec.frames {
            for b in 0..bins {
                let g = mask[t * bins + b];
                out.data[(t * 2) * bins + b] = spec.at(t, 0, b) * g;
                out.data[(t * 2 + 1) * bins + b] = spec.at(t, 1, b) * g;
            }
        }
        self.stft.istft(&out, len)
    }

    pub fn stream_hop(&self, frame: &[f32], cache: &mut Cache05) -> Vec<f32> {
        let bins = self.meta.erb.n_freq;
        assert_eq!(frame.len(), 2 * bins);
        let spec = Spectrogram { frames: 1, bins, data: frame.to_vec() };
        let logmag = Erb::logmag(&spec);
        let mut x = self.erb.bm(&logmag, 1);

        let mut skips: Vec<Vec<f32>> = Vec::with_capacity(self.encoder.layers.len());
        let mut ti = 0usize;
        for (i, blk) in self.encoder.layers.iter().enumerate() {
            let state = if blk.tfa.is_some() { Some(cache.tfa[ti].as_mut_slice()) } else { None };
            let (y, nc) = blk.stream(&x, cache.enc[i].as_deref(), state);
            cache.enc[i] = nc;
            if blk.tfa.is_some() { ti += 1; }
            skips.push(y.clone());
            x = y;
        }
        let f = self.meta.arch.enc_out_w[self.meta.arch.enc_out_w.len() - 1];
        let y0 = self.dp0.stream(&x, f, &mut cache.inter[0]);
        let y1 = self.dp1.stream(&y0, f, &mut cache.inter[1]);

        let mut h = y1;
        let n = self.decoder.layers.len();
        let mut w = self.decoder.layers[0].in_width;
        ti = 0;
        for (i, blk) in self.decoder.layers.iter().enumerate() {
            let skip = &skips[n - 1 - i];
            let target = skip.len() / (blk.in_ch);
            let fitted = fit_width(&h, blk.in_ch, 1, w, target);
            let mut sum = fitted;
            for (a, b) in sum.iter_mut().zip(skip) { *a += b; }
            let state = if blk.tfa.is_some() { Some(cache.tfa[skips_tfa_base(&self.meta) + ti].as_mut_slice()) } else { None };
            let (y, nc) = blk.stream(&sum, cache.dec[i].as_deref(), state);
            cache.dec[i] = nc;
            if blk.tfa.is_some() { ti += 1; }
            h = y;
            w = blk.out_width;
        }
        let mut mask_band = h;
        crate::simd::sigmoid_slice(&mut mask_band);
        let mask = self.erb.bs(&mask_band, 1);
        let mut out = vec![0.0f32; 2 * bins];
        for ch in 0..2 {
            for b in 0..bins { out[ch * bins + b] = frame[ch * bins + b] * mask[b]; }
        }
        out
    }
}

impl Denoiser05 {
    pub fn stream_wav(&self, wav: &[f32]) -> (Vec<f32>, Vec<std::time::Duration>) {
        let spec = self.stft.stft(wav);
        let bins = spec.bins;
        let mut cache = Cache05::zeros(&self.meta);
        let mut out = Spectrogram { frames: spec.frames, bins, data: vec![0.0f32; spec.data.len()] };
        let mut times = Vec::with_capacity(spec.frames);
        for t in 0..spec.frames {
            let frame = &spec.data[t * 2 * bins..(t + 1) * 2 * bins];
            let t0 = std::time::Instant::now();
            let y = self.stream_hop(frame, &mut cache);
            times.push(t0.elapsed());
            out.data[t * 2 * bins..(t + 1) * 2 * bins].copy_from_slice(&y);
        }
        (self.stft.istft(&out, wav.len()), times)
    }
}

fn skips_tfa_base(meta: &Meta) -> usize {
    meta.enc.iter().filter(|c| c.tfa).count()
}
