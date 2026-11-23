
use crate::ops::{act, conv::{Conv, Conv2d, ConvTranspose2d}, norm::batch_norm};
use crate::weights::{WeightError, Weights};

#[derive(Debug, Clone)]
pub struct Bn2d {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub eps: f32,
}

impl Bn2d {
    pub fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self, WeightError> {
        Ok(Self {
            gamma: w.get(&format!("{prefix}.weight"))?.data.clone(),
            beta: w.get(&format!("{prefix}.bias"))?.data.clone(),
            mean: w.get(&format!("{prefix}.running_mean"))?.data.clone(),
            var: w.get(&format!("{prefix}.running_var"))?.data.clone(),
            eps,
        })
    }

    pub fn forward(&self, x: &[f32], ch: usize, t: usize, w: usize) -> Vec<f32> {
        batch_norm(x, 1, ch, t, w, &self.gamma, &self.beta, &self.mean, &self.var, self.eps)
    }
}

pub struct CTfa {
    pub kt: usize,
    pub fgate: Conv2d, // (C,1,1,kf) pad_w=kf/2
    pub gate: Conv2d,  // (C,1,kt,1) pad 0（调用方左补 kt−1 帧）
    pub frames: usize,
}

impl CTfa {
    pub fn from_weights(w: &Weights, prefix: &str, tfa_kt: usize) -> Result<Self, WeightError> {
        let fwt = w.get(&format!("{prefix}.fgate.weight"))?;
        let kf = fwt.shape[3];
        let ch = fwt.shape[0];
        let fgate = Conv2d::from_weights(w, &format!("{prefix}.fgate"), 0, kf / 2, 1, 1, ch)?;
        let gate = Conv2d::from_weights(w, &format!("{prefix}.gate"), 0, 0, 1, 1, ch)?;
        Ok(Self { kt: tfa_kt, fgate, gate, frames: tfa_kt - 1 })
    }

    pub fn channels(&self) -> usize { self.fgate.out_ch }

    pub fn forward_offline(&self, x: &[f32], t: usize, w: usize) -> TfaTrace {
        let ch = self.channels();
        assert_eq!(x.len(), ch * t * w);
        let (fg, _, _) = self.fgate.forward(x, 1, t, w);
        let mut fgate_sig = fg;
        act::sigmoid_slice(&mut fgate_sig);
        let mut g1 = vec![0.0f32; x.len()];
        for i in 0..x.len() { g1[i] = x[i] * fgate_sig[i]; }
        // zt = mean(g1²) over width → (ch, t, 1)
        let mut zt = vec![0.0f32; ch * t];
        for c in 0..ch {
            for ti in 0..t {
                let row = &g1[(c * t + ti) * w..(c * t + ti + 1) * w];
                let mut acc = 0.0f32;
                for &v in row { acc += v * v; }
                zt[c * t + ti] = acc / w as f32;
            }
        }
        let tp = t + self.frames;
        let mut zp = vec![0.0f32; ch * tp];
        for c in 0..ch {
            zp[c * tp + self.frames..(c + 1) * tp].copy_from_slice(&zt[c * t..(c + 1) * t]);
        }
        let (g, _, _) = self.gate.forward(&zp, 1, tp, 1);
        let mut gate_sig = g;
        act::sigmoid_slice(&mut gate_sig);
        let mut out = vec![0.0f32; x.len()];
        for c in 0..ch {
            for ti in 0..t {
                let gv = gate_sig[c * t + ti];
                for wi in 0..w {
                    let i = (c * t + ti) * w + wi;
                    out[i] = g1[i] * gv;
                }
            }
        }
        TfaTrace { fgate_sig, g1, zt, gate_sig, out }
    }
}

#[derive(Debug)]
pub struct TfaTrace {
    pub fgate_sig: Vec<f32>,
    pub g1: Vec<f32>,
    pub zt: Vec<f32>,
    pub gate_sig: Vec<f32>,
    pub out: Vec<f32>,
}

pub struct GateBlock {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kt: usize,
    pub in_width: usize,
    pub out_width: usize,
    pub pconv: Conv2d,
    pub bn1: Bn2d,
    pub dconv: Conv,
    pub bn2: Bn2d,
    pub tfa: Option<CTfa>,
}

impl GateBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(w: &Weights, prefix: &str, in_width: usize, stride: usize,
                        pad_w: usize, use_deconv: bool, has_tfa: bool, tfa_kt: usize,
                        bn_eps: f32) -> Result<Self, WeightError> {
        let pconv = Conv2d::from_weights(w, &format!("{prefix}.pconv"), 0, 0, 1, 1, 1)?;
        let dconv = if use_deconv {
            let c = ConvTranspose2d::from_weights(w, &format!("{prefix}.dconv"), 0, pad_w, 1,
                                                  stride, pconv.out_ch)?;
            let kt = c.kh;
            let mut c = c;
            c.ph = kt - 1;
            Conv::Transposed(c)
        } else {
            Conv::Direct(Conv2d::from_weights(w, &format!("{prefix}.dconv"),
                                              0, pad_w, 1, stride, pconv.out_ch)?)
        };
        let tfa = if has_tfa {
            Some(CTfa::from_weights(w, &format!("{prefix}.tfa"), tfa_kt)?)
        } else {
            None
        };
        let (dkh, dkw) = dconv.kernel();
        let out_width = if use_deconv {
            (in_width - 1) * stride + dkw - 2 * pad_w
        } else {
            (in_width + 2 * pad_w).saturating_sub(dkw) / stride + 1
        };
        Ok(Self {
            in_ch: pconv.in_ch,
            out_ch: pconv.out_ch,
            kt: dkh,
            in_width,
            out_width,
            bn1: Bn2d::from_weights(w, &format!("{prefix}.bn1"), bn_eps)?,
            bn2: Bn2d::from_weights(w, &format!("{prefix}.bn2"), bn_eps)?,
            pconv, dconv, tfa,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.in_ch * t * self.in_width);
        let pconv = self.pconv.forward(x, 1, t, self.in_width).0;
        let mut h = self.bn1.forward(&pconv, self.out_ch, t, self.in_width);
        act::hswish_slice(&mut h);
        let pad = self.kt - 1;
        let tw = self.in_width;
        let tp = t + pad;
        let mut dp = vec![0.0f32; self.out_ch * tp * tw];
        if pad > 0 {
            for c in 0..self.out_ch {
                dp[c * tp * tw + pad * tw..(c + 1) * tp * tw]
                    .copy_from_slice(&h[c * t * tw..(c + 1) * t * tw]);
            }
        } else {
            dp.copy_from_slice(&h);
        }
        let (y, _, ow) = self.dconv.forward(&dp, 1, tp, tw);
        let mut out = self.bn2.forward(&y, self.out_ch, t, ow);
        if let Some(tfa) = &self.tfa {
            out = tfa.forward_offline(&out, t, ow).out;
        }
        if self.in_ch == self.out_ch && self.in_width == self.out_width {
            for i in 0..out.len() { out[i] += x[i]; }
        }
        out
    }

    pub fn forward_trace(&self, x: &[f32], t: usize) -> BlockTrace {
        assert_eq!(x.len(), self.in_ch * t * self.in_width);
        let pconv = self.pconv.forward(x, 1, t, self.in_width).0;
        let bn1 = self.bn1.forward(&pconv, self.out_ch, t, self.in_width);
        let mut hswish = bn1.clone();
        act::hswish_slice(&mut hswish);
        let pad = self.kt - 1;
        let tw = self.in_width;
        let tp = t + pad;
        let mut dpad = vec![0.0f32; self.out_ch * tp * tw];
        if pad > 0 {
            for c in 0..self.out_ch {
                dpad[c * tp * tw + pad * tw..(c + 1) * tp * tw]
                    .copy_from_slice(&hswish[c * t * tw..(c + 1) * t * tw]);
            }
        } else {
            dpad.copy_from_slice(&hswish);
        }
        let (dconv, _, ow) = self.dconv.forward(&dpad, 1, tp, tw);
        let bn2 = self.bn2.forward(&dconv, self.out_ch, t, ow);
        let (tfa_out, tfa_trace) = match &self.tfa {
            Some(tfa) => {
                let tr = tfa.forward_offline(&bn2, t, ow);
                (Some(tr.out.clone()), Some(tr))
            }
            None => (None, None),
        };
        let mut out = tfa_out.clone().unwrap_or_else(|| bn2.clone());
        if self.in_ch == self.out_ch && self.in_width == self.out_width {
            for i in 0..out.len() { out[i] += x[i]; }
        }
        BlockTrace { pconv, bn1, hswish, dpad, dconv, bn2, tfa: tfa_trace, out, out_w: ow }
    }
}

#[derive(Debug)]
pub struct BlockTrace {
    pub pconv: Vec<f32>,
    pub bn1: Vec<f32>,
    pub hswish: Vec<f32>,
    pub dpad: Vec<f32>,
    pub dconv: Vec<f32>,
    pub bn2: Vec<f32>,
    pub tfa: Option<TfaTrace>,
    pub out: Vec<f32>,
    pub out_w: usize,
}

impl GateBlock {
    pub fn debug_trace_pconv(&self, x: &[f32], t: usize) -> Vec<f32> {
        let p = self.pconv.forward(x, 1, t, self.in_width).0;
        let b = self.bn1.forward(&p, self.out_ch, t, self.in_width);
        let mut h: Vec<f32> = b;
        act::hswish_slice(&mut h);
        h
    }
    pub fn debug_trace_dconv(&self, x: &[f32], t: usize) -> Vec<f32> {
        let h = self.debug_trace_pconv(x, t);
        let pad = self.kt - 1;
        let tp = t + pad;
        let w = self.in_width;
        let mut dp = vec![0.0f32; self.out_ch * tp * w];
        if pad > 0 {
            for c in 0..self.out_ch {
                dp[c * tp * w + pad * w..(c + 1) * tp * w]
                    .copy_from_slice(&h[c * t * w..(c + 1) * t * w]);
            }
        } else { dp.copy_from_slice(&h); }
        let (y, _, ow) = self.dconv.forward(&dp, 1, tp, w);
        self.bn2.forward(&y, self.out_ch, t, ow)
    }
    pub fn debug_trace_tfa(&self, x: &[f32], t: usize) -> Vec<f32> {
        let b2 = self.debug_trace_dconv(x, t);
        match &self.tfa {
            Some(tfa) => tfa.forward_offline(&b2, t, self.out_width).out,
            None => b2,
        }
    }
}
