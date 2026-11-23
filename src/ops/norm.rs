/// BatchNorm2d inference (NCHW).
pub fn batch_norm(
    x: &[f32], n: usize, c: usize, h: usize, w: usize,
    gamma: &[f32], beta: &[f32], mean: &[f32], var: &[f32], eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for ni in 0..n {
        for ci in 0..c {
            let inv = 1.0 / (var[ci] + eps).sqrt();
            for i in 0..h * w {
                let idx = ((ni * c + ci) * h) * w + i;
                y[idx] = (x[idx] - mean[ci]) * inv * gamma[ci] + beta[ci];
            }
        }
    }
    y
}

/// DyT: `y = gamma * tanh(alpha * x) + beta` (gamma/beta same length as x).
pub fn dyt(x: &[f32], gamma: &[f32], beta: &[f32], alpha: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len());
    assert_eq!(x.len(), beta.len());
    x.iter().zip(gamma).zip(beta)
        .map(|((&xv, &g), &b)| g * (alpha * xv).tanh() + b)
        .collect()
}

#[derive(Debug, Clone)]
pub struct DyT {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: f32,
}

impl DyT {
    pub fn from_weights(w: &crate::weights::Weights, prefix: &str)
        -> Result<Self, crate::weights::WeightError>
    {
        let alpha = w.get(&format!("{prefix}.alpha"))?.data[0];
        Ok(Self {
            gamma: w.get(&format!("{prefix}.gamma"))?.data.clone(),
            beta: w.get(&format!("{prefix}.beta"))?.data.clone(),
            alpha,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let l = self.gamma.len();
        assert!(l > 0 && x.len() % l == 0, "DyT broadcast {} vs {}", x.len(), l);
        let mut y: Vec<f32> = x.iter().map(|&v| self.alpha * v).collect();
        crate::simd::tanh_slice(&mut y);              // AVX2 tanh
        for (i, yv) in y.iter_mut().enumerate() {
            let k = i % l;
            *yv = self.gamma[k] * *yv + self.beta[k];
        }
        y
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn from_weights(w: &crate::weights::Weights, prefix: &str, eps: f32)
        -> Result<Self, crate::weights::WeightError>
    {
        Ok(Self {
            gamma: w.get(&format!("{prefix}.weight"))?.data.clone(),
            beta: w.get(&format!("{prefix}.bias"))?.data.clone(),
            eps,
        })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.gamma.len();
        assert_eq!(x.len() % dim, 0);
        let rows = x.len() / dim;
        let mut y = vec![0.0f32; x.len()];
        for r in 0..rows {
            let src = &x[r * dim..(r + 1) * dim];
            let mean = src.iter().sum::<f32>() / dim as f32;
            let var = src.iter().map(|v| { let d = v - mean; d * d }).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            for i in 0..dim {
                y[r * dim + i] = (src[i] - mean) * inv * self.gamma[i] + self.beta[i];
            }
        }
        y
    }
}
