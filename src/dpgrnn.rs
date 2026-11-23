
use crate::gru::GruNet;
use crate::ops::conv::Conv2d;
use crate::ops::linear::Linear;
use crate::ops::norm::DyT;
use crate::weights::{WeightError, Weights};

pub struct DpGrnn {
    pub input: usize,      // C
    pub width: usize,      // F
    pub hidden: usize,
    pub intra_dw: Conv2d,  // (C,1,1,kf) groups=C pad (0,kf/2)
    pub intra_pw: Conv2d,  // 1×1
    pub intra_norm: DyT,   // (1, F, C)
    pub inter_rnn: GruNet, // C → hidden，沿时间，单层单向
    pub inter_fc: Linear,  // hidden → C
    pub inter_norm: DyT,   // (1, F, C)
}

impl DpGrnn {
    pub fn from_weights(w: &Weights, prefix: &str, input: usize, width: usize, hidden: usize)
        -> Result<Self, WeightError>
    {
        let kf = w.get(&format!("{prefix}.intra_dw.weight"))?.shape[3];
        Ok(Self {
            input, width, hidden,
            intra_dw: Conv2d::from_weights(w, &format!("{prefix}.intra_dw"), 0, kf / 2, 1, 1, input)?,
            intra_pw: Conv2d::from_weights(w, &format!("{prefix}.intra_pw"), 0, 0, 1, 1, 1)?,
            intra_norm: DyT::from_weights(w, &format!("{prefix}.intra_norm"))?,
            inter_rnn: GruNet::from_weights(w, &format!("{prefix}.inter_rnn"), 1, false)?,
            inter_fc: Linear::from_weights(w, &format!("{prefix}.inter_fc"))?,
            inter_norm: DyT::from_weights(w, &format!("{prefix}.inter_norm"))?,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize, f: usize) -> Vec<f32> {
        let c = self.input;
        assert_eq!(x.len(), c * t * f);
        let (dw, _, _) = self.intra_dw.forward(x, 1, t, f);
        let (pw, _, _) = self.intra_pw.forward(&dw, 1, t, f);
        let mut perm = vec![0.0f32; c * t * f]; // (T,F,C)
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    perm[(ti * f + fi) * c + ci] = pw[(ci * t + ti) * f + fi];
                }
            }
        }
        let n = self.intra_norm.forward(&perm);
        let mut intra = vec![0.0f32; x.len()]; // (C,T,F)
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    intra[(ci * t + ti) * f + fi] = n[(ti * f + fi) * c + ci];
                }
            }
        }
        let mut intra_out = vec![0.0f32; x.len()];
        for i in 0..x.len() { intra_out[i] = x[i] + intra[i]; }

        let mut seq = vec![0.0f32; f * t * c];
        for fi in 0..f {
            for ti in 0..t {
                for ci in 0..c {
                    seq[(fi * t + ti) * c + ci] = intra_out[(ci * t + ti) * f + fi];
                }
            }
        }
        let h0 = vec![0.0f32; f * self.hidden];
        let (ry, _) = self.inter_rnn.forward(&seq, f, t, &h0);
        let lin = self.inter_fc.forward(&ry, f * t);
        // (F,T,C) → (T,F,C)
        let mut tl = vec![0.0f32; t * f * c];
        for fi in 0..f {
            for ti in 0..t {
                for ci in 0..c {
                    tl[(ti * f + fi) * c + ci] = lin[(fi * t + ti) * c + ci];
                }
            }
        }
        let nl = self.inter_norm.forward(&tl);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for ti in 0..t {
                for fi in 0..f {
                    out[(ci * t + ti) * f + fi] =
                        intra_out[(ci * t + ti) * f + fi] + nl[(ti * f + fi) * c + ci];
                }
            }
        }
        out
    }
}
