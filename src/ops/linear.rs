
use crate::weights::{WeightError, Weights};

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

pub fn linear(x: &[f32], rows: usize, input: usize, output: usize,
              w: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), rows * input);
    assert_eq!(w.len(), output * input);
    let wt = transpose_w(w, output, input);
    linear_t(x, rows, input, output, &wt, b)
}

pub fn linear_t(x: &[f32], rows: usize, input: usize, output: usize,
                wt: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), rows * input);
    assert_eq!(wt.len(), output * input);
    let mut y = vec![0.0f32; rows * output];
    for r in 0..rows {
        crate::simd::matvec_t(&x[r * input..(r + 1) * input], wt, input, output, b,
                              &mut y[r * output..(r + 1) * output]);
    }
    y
}

pub fn transpose_w(w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut wt = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for k in 0..in_dim {
            wt[k * out_dim + o] = w[o * in_dim + k];
        }
    }
    wt
}

#[derive(Debug, Clone)]
pub struct Linear {
    pub w: Vec<f32>,      // 原始 (out, in)
    pub wt: Vec<f32>,     // 转置 (in, out)：matvec_t 用
    pub b: Option<Vec<f32>>,
    pub input: usize,
    pub output: usize,
}

impl Linear {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self, WeightError> {
        let wt = w.get(&format!("{prefix}.weight"))?;
        let (output, input) = (wt.shape[0], wt.shape[1]);
        let wdata = wt.data.clone();
        Ok(Self {
            wt: transpose_w(&wdata, output, input),
            w: wdata,
            b: w.map.get(&format!("{prefix}.bias")).map(|t| t.data.clone()),
            input,
            output,
        })
    }

    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        linear_t(x, rows, self.input, self.output, &self.wt, self.b.as_deref())
    }
}
