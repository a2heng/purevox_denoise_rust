
use crate::ops::linear::{transpose_w};
use crate::weights::{WeightError, Weights};

#[derive(Debug, Clone)]
pub struct GruLayer {
    pub input: usize,
    pub hidden: usize,
    pub reverse: bool,
    pub w_ih: Vec<f32>,   // (3H, input)
    pub w_hh: Vec<f32>,   // (3H, H)
    pub w_ih_t: Vec<f32>, // (input, 3H)
    pub w_hh_t: Vec<f32>, // (H, 3H)
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
}

impl GruLayer {
    pub fn from_weights(w: &Weights, prefix: &str, suffix: &str, input: usize, hidden: usize,
                        reverse: bool) -> Result<Self, WeightError> {
        let p = |n: &str| format!("{prefix}.{n}{suffix}");
        let w_ih = w.get(&p("weight_ih"))?.data.clone();
        let w_hh = w.get(&p("weight_hh"))?.data.clone();
        Ok(Self {
            input, hidden, reverse,
            w_ih_t: transpose_w(&w_ih, 3 * hidden, input),
            w_hh_t: transpose_w(&w_hh, 3 * hidden, hidden),
            w_ih, w_hh,
            b_ih: w.get(&p("bias_ih"))?.data.clone(),
            b_hh: w.get(&p("bias_hh"))?.data.clone(),
        })
    }

    pub fn forward(&self, x: &[f32], batch: usize, seq: usize, h0: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (i, h) = (self.input, self.hidden);
        assert_eq!(x.len(), batch * seq * i);
        assert_eq!(h0.len(), batch * h);
        let mut state = h0.to_vec();
        let mut y = vec![0.0f32; batch * seq * h];
        let mut gx = vec![0.0f32; batch * seq * 3 * h];
        for row in 0..batch * seq {
            crate::simd::matvec_t(&x[row * i..(row + 1) * i], &self.w_ih_t, i, 3 * h,
                                  Some(&self.b_ih), &mut gx[row * 3 * h..(row + 1) * 3 * h]);
        }
        let mut gh = vec![0.0f32; 3 * h];
        let mut pre = vec![0.0f32; 3 * h];
        let mut next = vec![0.0f32; batch * h];
        for step in 0..seq {
            let t = if self.reverse { seq - 1 - step } else { step };
            for b in 0..batch {
                let hb = &state[b * h..(b + 1) * h];
                crate::simd::matvec_t(hb, &self.w_hh_t, h, 3 * h, Some(&self.b_hh), &mut gh);
                let gxi = &gx[(b * seq + t) * 3 * h..(b * seq + t + 1) * 3 * h];
                for r in 0..2 * h { pre[r] = gxi[r] + gh[r]; }
                crate::simd::sigmoid_slice(&mut pre[..2 * h]);
                for k in 0..h { pre[2 * h + k] = gxi[2 * h + k] + pre[k] * gh[2 * h + k]; }
                crate::simd::tanh_slice(&mut pre[2 * h..]);
                let out = &mut next[b * h..(b + 1) * h];
                for k in 0..h {
                    let zz = pre[h + k];
                    out[k] = (1.0 - zz) * pre[2 * h + k] + zz * hb[k];
                }
            }
            std::mem::swap(&mut state, &mut next);
            for b in 0..batch {
                y[(b * seq + t) * h..(b * seq + t + 1) * h]
                    .copy_from_slice(&state[b * h..(b + 1) * h]);
            }
        }
        (y, state)
    }

    pub fn step(&self, x: &[f32], h: &mut [f32]) -> Vec<f32> {
        let (i, hd) = (self.input, self.hidden);
        let batch = h.len() / hd;
        assert_eq!(x.len(), batch * i);
        let mut y = vec![0.0f32; batch * hd];
        let mut gh = vec![0.0f32; 3 * hd];
        let mut pre = vec![0.0f32; 3 * hd];
        for b in 0..batch {
            let xb = &x[b * i..(b + 1) * i];
            let hb = &h[b * hd..(b + 1) * hd];
            crate::simd::matvec_t(hb, &self.w_hh_t, hd, 3 * hd, Some(&self.b_hh), &mut gh);
            crate::simd::matvec_t(xb, &self.w_ih_t, i, 3 * hd, Some(&self.b_ih), &mut pre);
            for r in 0..2 * hd { pre[r] += gh[r]; }
            crate::simd::sigmoid_slice(&mut pre[..2 * hd]);
            for k in 0..hd {
                pre[2 * hd + k] += pre[k] * gh[2 * hd + k];
            }
            crate::simd::tanh_slice(&mut pre[2 * hd..]);
            for k in 0..hd {
                let zz = pre[hd + k];
                y[b * hd + k] = (1.0 - zz) * pre[2 * hd + k] + zz * hb[k];
            }
        }
        h.copy_from_slice(&y);
        y
    }
}

#[derive(Debug, Clone)]
pub struct GruNet {
    pub input: usize,
    pub hidden: usize,
    pub bidirectional: bool,
    pub layers: Vec<GruLayer>,
}

impl GruNet {
    pub fn from_weights(w: &Weights, prefix: &str, num_layers: usize, bidirectional: bool)
        -> Result<Self, WeightError>
    {
        let dirs = if bidirectional { 2 } else { 1 };
        let mut layers = Vec::with_capacity(num_layers * dirs);
        let mut input = w.get(&format!("{prefix}.weight_ih_l0"))?.shape[1];
        for l in 0..num_layers {
            let hidden = w.get(&format!("{prefix}.weight_hh_l{l}"))?.shape[1];
            for d in 0..dirs {
                let suffix = if d == 1 { format!("_l{l}_reverse") } else { format!("_l{l}") };
                layers.push(GruLayer::from_weights(w, prefix, &suffix, input, hidden, d == 1)?);
            }
            input = hidden * dirs;
        }
        let hidden = w.get(&format!("{prefix}.weight_hh_l0"))?.shape[1];
        Ok(Self { input: w.get(&format!("{prefix}.weight_ih_l0"))?.shape[1],
                  hidden, bidirectional, layers })
    }

    pub fn out_dim(&self) -> usize {
        self.hidden * if self.bidirectional { 2 } else { 1 }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len() / if self.bidirectional { 2 } else { 1 }
    }

    pub fn forward(&self, x: &[f32], batch: usize, seq: usize, h0: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let dirs = if self.bidirectional { 2 } else { 1 };
        let h = self.hidden;
        assert_eq!(h0.len(), self.layers.len() * batch * h);
        let mut cur = x.to_vec();
        let mut hn = vec![0.0f32; self.layers.len() * batch * h];
        let mut li = 0;
        for _ in 0..self.num_layers() {
            let (yf, hf) = self.layers[li].forward(&cur, batch, seq,
                                                   &h0[li * batch * h..(li + 1) * batch * h]);
            hn[li * batch * h..(li + 1) * batch * h].copy_from_slice(&hf);
            li += 1;
            if dirs == 2 {
                let (yb, hb) = self.layers[li].forward(&cur, batch, seq,
                                                       &h0[li * batch * h..(li + 1) * batch * h]);
                hn[li * batch * h..(li + 1) * batch * h].copy_from_slice(&hb);
                li += 1;
                let mut merged = vec![0.0f32; batch * seq * 2 * h];
                for b in 0..batch {
                    for t in 0..seq {
                        let dst = (b * seq + t) * 2 * h;
                        let src = (b * seq + t) * h;
                        merged[dst..dst + h].copy_from_slice(&yf[src..src + h]);
                        merged[dst + h..dst + 2 * h].copy_from_slice(&yb[src..src + h]);
                    }
                }
                cur = merged;
            } else {
                cur = yf;
            }
        }
        (cur, hn)
    }

    pub fn step(&self, x: &[f32], h: &mut [f32]) -> Vec<f32> {
        assert!(!self.bidirectional, "双向 GRU 不支持流式 step");
        let batch = h.len() / (self.layers.len() * self.hidden);
        let mut cur = x.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            let hd = self.hidden;
            let sl = &mut h[li * batch * hd..(li + 1) * batch * hd];
            cur = layer.step(&cur, sl);
        }
        cur
    }
}
