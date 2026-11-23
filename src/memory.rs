
use crate::gru::GruNet;
use crate::ops::linear::Linear;
use crate::ops::norm::LayerNorm;
use crate::weights::{WeightError, Weights};

pub struct BottleneckMemory {
    pub channels: usize,
    pub fhid: usize,
    pub bhid: usize,
    pub bands: usize,
    pub slots: usize,
    pub dim: usize,
    pub ln: LayerNorm,
    pub fgru: GruNet,       // C → fhid，双向，沿频
    pub fproj: Linear,      // 2·fhid → C
    pub sgru: GruNet,       // C → bhid，沿时，每个子带一条
    pub sproj: Linear,      // bhid → C
    pub pq: Linear,         // C → dim
    pub slots_w: Vec<f32>,  // (slots, dim)
    pub pv: Linear,         // dim → C
    pub tgru: GruNet,       // C → hid，2 层，沿时
    pub tproj: Linear,      // hid → C
}

impl BottleneckMemory {
    pub fn from_weights(w: &Weights, prefix: &str, channels: usize, _hid: usize, layers: usize)
        -> Result<Self, WeightError>
    {
        let fhid = w.get(&format!("{prefix}.fproj.weight"))?.shape[1] / 2;
        let bhid = w.get(&format!("{prefix}.sproj.weight"))?.shape[1];
        let dim = w.get(&format!("{prefix}.pq.weight"))?.shape[0];
        let slots = w.get(&format!("{prefix}.slots"))?.data.clone();
        Ok(Self {
            channels, fhid, bhid,
            bands: BANDS,
            slots: slots.len() / dim,
            dim,
            ln: LayerNorm::from_weights(w, &format!("{prefix}.ln"), 1e-5)?,
            fgru: GruNet::from_weights(w, &format!("{prefix}.fgru"), 1, true)?,
            fproj: Linear::from_weights(w, &format!("{prefix}.fproj"))?,
            sgru: GruNet::from_weights(w, &format!("{prefix}.sgru"), 1, false)?,
            sproj: Linear::from_weights(w, &format!("{prefix}.sproj"))?,
            pq: Linear::from_weights(w, &format!("{prefix}.pq"))?,
            slots_w: slots,
            pv: Linear::from_weights(w, &format!("{prefix}.pv"))?,
            tgru: GruNet::from_weights(w, &format!("{prefix}.tgru"), layers, false)?,
            tproj: Linear::from_weights(w, &format!("{prefix}.tproj"))?,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let mut x = x.to_vec();
        x = self.freq(&x, t, f);
        x = self.proto(&x, t, f);
        x = self.subband(&x, t, f);
        x = self.global(&x, t, f);
        x
    }

    fn to_tfc(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.channels;
        let mut y = vec![0.0f32; t * f * c];
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    y[(ti * f + fi) * c + ci] = x[(ci * t + ti) * f + fi];
                }
            }
        }
        y
    }

    fn freq(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.channels;
        let flat = self.to_tfc(x, t, f); // (T,F,C)
        let ln = self.ln.forward(&flat);
        let h0 = vec![0.0f32; 2 * t * self.fhid];
        let (y, _) = self.fgru.forward(&ln, t, f, &h0); // (T,F,2·fhid)
        let proj = self.fproj.forward(&y, t * f);       // (T,F,C)
        let mut out = x.to_vec();
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    out[(ci * t + ti) * f + fi] += proj[(ti * f + fi) * c + ci];
                }
            }
        }
        out
    }

    fn proto(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.channels;
        let d = self.dim;
        // q = pq(mean over F) → (T, dim)
        let mut pooled = vec![0.0f32; t * c];
        for ti in 0..t {
            for ci in 0..c {
                let mut acc = 0.0f32;
                for fi in 0..f { acc += x[(ci * t + ti) * f + fi]; }
                pooled[ti * c + ci] = acc / f as f32;
            }
        }
        let q = self.pq.forward(&pooled, t); // (T,d)
        let scale = 1.0 / (d as f32).sqrt();
        let mut attn = vec![0.0f32; t * self.slots];
        for ti in 0..t {
            let mut logits = vec![0.0f32; self.slots];
            for si in 0..self.slots {
                let mut a = 0.0f32;
                for k in 0..d { a += q[ti * d + k] * self.slots_w[si * d + k]; }
                logits[si] = a * scale;
            }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in logits.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            for (si, v) in logits.iter().enumerate() { attn[ti * self.slots + si] = v / sum; }
        }
        // y = pv(attn @ slots) → (T,C)
        let mut rd = vec![0.0f32; t * d];
        for ti in 0..t {
            for k in 0..d {
                let mut a = 0.0f32;
                for si in 0..self.slots { a += attn[ti * self.slots + si] * self.slots_w[si * d + k]; }
                rd[ti * d + k] = a;
            }
        }
        let y = self.pv.forward(&rd, t); // (T,C)
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for ti in 0..t {
                let g = 1.0 + y[ti * c + ci].tanh();
                for fi in 0..f { out[(ci * t + ti) * f + fi] = x[(ci * t + ti) * f + fi] * g; }
            }
        }
        out
    }

    fn subband(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.channels;
        let bands = self.bands;
        assert_eq!(f % bands, 0);
        let fb = f / bands;
        let mut g = vec![0.0f32; bands * t * c];
        for ci in 0..c {
            for ti in 0..t {
                for bi in 0..bands {
                    let mut acc = 0.0f32;
                    for k in 0..fb { acc += x[(ci * t + ti) * f + bi * fb + k]; }
                    g[(bi * t + ti) * c + ci] = acc / fb as f32;
                }
            }
        }
        let h0 = vec![0.0f32; bands * self.bhid];
        let (hh, _) = self.sgru.forward(&g, bands, t, &h0); // (bands,T,bhid)
        let proj = self.sproj.forward(&hh, bands * t);     // (bands,T,C)
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for ti in 0..t {
                for bi in 0..bands {
                    let gate = proj[(bi * t + ti) * c + ci].tanh();
                    for k in 0..fb {
                        let i = (ci * t + ti) * f + bi * fb + k;
                        out[i] = x[i] * (1.0 + gate);
                    }
                }
            }
        }
        out
    }

    fn global(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.channels;
        let mut g = vec![0.0f32; t * c];
        for ti in 0..t {
            for ci in 0..c {
                let mut acc = 0.0f32;
                for fi in 0..f { acc += x[(ci * t + ti) * f + fi]; }
                g[ti * c + ci] = acc / f as f32;
            }
        }
        let h0 = vec![0.0f32; self.tgru.layers.len() * self.tgru.hidden];
        let (hh, _) = self.tgru.forward(&g, 1, t, &h0); // (1,T,hid)
        let proj = self.tproj.forward(&hh, t);          // (T,C)
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for ti in 0..t {
                let gate = proj[ti * c + ci].tanh();
                for fi in 0..f { out[(ci * t + ti) * f + fi] = x[(ci * t + ti) * f + fi] * (1.0 + gate); }
            }
        }
        out
    }
}

pub const BANDS: usize = 6;
