//! ERB front-end: `bm` = low 160 bins passthrough + high 321 bins × w_avg (80 bands).
//! `w_avg` is row-sum normalized (feature side); `w_up = wᵀ` is the mask expander.

use crate::weights::{WeightError, Weights};

pub struct Erb {
    pub low: usize,
    pub high: usize,
    pub n_freq: usize,
    w_avg: Vec<f32>, // (high, n_freq-low)
    w_up: Vec<f32>,  // (n_freq-low, high)
}

impl Erb {
    pub fn from_weights(w: &Weights, low: usize, high: usize, n_freq: usize) -> Result<Self, WeightError> {
        Self::from_weights_named(w, "band.w_avg", "band.w_up", low, high, n_freq)
    }

    pub fn from_weights_named(w: &Weights, avg: &str, up: &str, low: usize, high: usize,
                              n_freq: usize) -> Result<Self, WeightError> {
        let a = w.get(avg)?;
        let u = w.get(up)?;
        assert_eq!(a.shape, vec![high, n_freq - low], "w_avg shape");
        assert_eq!(u.shape, vec![n_freq - low, high], "w_up shape");
        Ok(Self { low, high, n_freq, w_avg: a.data.clone(), w_up: u.data.clone() })
    }

    /// `x`: `[frame][481]` log10 magnitudes → `[frame][low+high]`.
    pub fn bm(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let out_dim = self.low + self.high;
        let high_in = self.n_freq - self.low;
        let mut out = vec![0.0f32; frames * out_dim];
        for t in 0..frames {
            let src = &x[t * self.n_freq..(t + 1) * self.n_freq];
            let dst = &mut out[t * out_dim..(t + 1) * out_dim];
            dst[..self.low].copy_from_slice(&src[..self.low]);
            for b in 0..self.high {
                let row = &self.w_avg[b * high_in..(b + 1) * high_in];
                let mut acc = 0.0f32;
                for j in 0..high_in {
                    acc += row[j] * src[self.low + j];
                }
                dst[self.low + b] = acc;
            }
        }
        out
    }

    /// `x`: `[frame][low+high]` mask → `[frame][481]` (transpose expander).
    pub fn bs(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let out_dim = self.low + self.high;
        let high_in = self.n_freq - self.low;
        let mut out = vec![0.0f32; frames * self.n_freq];
        for t in 0..frames {
            let src = &x[t * out_dim..(t + 1) * out_dim];
            let dst = &mut out[t * self.n_freq..(t + 1) * self.n_freq];
            dst[..self.low].copy_from_slice(&src[..self.low]);
            for j in 0..high_in {
                let row = &self.w_up[j * self.high..(j + 1) * self.high];
                let mut acc = 0.0f32;
                for b in 0..self.high {
                    acc += row[b] * src[self.low + b];
                }
                dst[self.low + j] = acc;
            }
        }
        out
    }

    pub fn logmag(spec: &crate::stft::Spectrogram) -> Vec<f32> {
        let mut out = vec![0.0f32; spec.frames * spec.bins];
        for t in 0..spec.frames {
            for b in 0..spec.bins {
                let re = spec.at(t, 0, b);
                let im = spec.at(t, 1, b);
                out[t * spec.bins + b] = (re * re + im * im).sqrt().max(1e-12).log10();
            }
        }
        out
    }
}
