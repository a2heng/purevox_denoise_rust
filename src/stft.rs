//! STFT matching `torch.stft(..., n_fft=960, hop=480, window=periodic Hann,
//! center=True, pad_mode='constant')` and `Fourier.stft` layout `(T,2,481)`.

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex32;

/// Layout: `[frame][channel(0=re,1=im)][bin]`.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    pub frames: usize,
    pub bins: usize,
    pub data: Vec<f32>,
}

impl Spectrogram {
    pub fn at(&self, t: usize, ch: usize, bin: usize) -> f32 {
        self.data[(t * 2 + ch) * self.bins + bin]
    }
}

#[derive(Clone)]
pub struct Stft {
    n_fft: usize,
    hop: usize,
    win: Vec<f32>,
}

impl Stft {
    /// Periodic Hann, same as `torch.hann_window(n_fft)` (periodic=True).
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let win = (0..n_fft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
            .collect();
        Self { n_fft, hop, win }
    }

    pub fn n_fft(&self) -> usize { self.n_fft }
    pub fn hop(&self) -> usize { self.hop }

    pub fn stft(&self, wav: &[f32]) -> Spectrogram {
        let bins = self.n_fft / 2 + 1;
        let pad = self.n_fft / 2;
        let padded_len = wav.len() + 2 * pad;
        let frames = if padded_len < self.n_fft {
            0
        } else {
            1 + (padded_len - self.n_fft) / self.hop
        };
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(self.n_fft);
        let mut buf = vec![0.0f32; self.n_fft];
        let mut out = vec![Complex32::new(0.0, 0.0); bins];
        let mut spec = Spectrogram { frames, bins, data: vec![0.0; frames * 2 * bins] };
        for t in 0..frames {
            let start = t * self.hop;
            for i in 0..self.n_fft {
                let idx = start + i;
                let x = if idx >= pad && idx < pad + wav.len() {
                    wav[idx - pad]
                } else {
                    0.0
                };
                buf[i] = x * self.win[i];
            }
            r2c.process(&mut buf, &mut out).expect("realfft length mismatch");
            for b in 0..bins {
                spec.data[(t * 2) * bins + b] = out[b].re;
                spec.data[(t * 2 + 1) * bins + b] = out[b].im;
            }
        }
        spec
    }
}

impl Stft {
    pub fn istft(&self, spec: &Spectrogram, length: usize) -> Vec<f32> {
        let n = self.n_fft;
        let hop = self.hop;
        let pad = n / 2;
        let total = if spec.frames == 0 { 0 } else { (spec.frames - 1) * hop + n };
        let mut acc = vec![0.0f32; total];
        let mut wsum = vec![0.0f32; total];
        let mut planner = RealFftPlanner::<f32>::new();
        let c2r = planner.plan_fft_inverse(n);
        let mut buf = vec![Complex32::new(0.0, 0.0); spec.bins];
        let mut out = vec![0.0f32; n];
        for t in 0..spec.frames {
            for b in 0..spec.bins {
                buf[b] = Complex32::new(spec.at(t, 0, b), spec.at(t, 1, b));
            }
            buf[0].im = 0.0;
            buf[spec.bins - 1].im = 0.0;
            c2r.process(&mut buf, &mut out).expect("realfft length mismatch");
            let inv = 1.0 / n as f32;
            for v in out.iter_mut() { *v *= inv; }
            let start = t * hop;
            for i in 0..n {
                acc[start + i] += out[i] * self.win[i];
                wsum[start + i] += self.win[i] * self.win[i];
            }
        }
        let mut y = vec![0.0f32; total];
        for i in 0..total {
            y[i] = if wsum[i] > 1e-11 { acc[i] / wsum[i] } else { 0.0 };
        }
        let start = pad.min(total);
        let end = (start + length).min(total);
        y[start..end].to_vec()
    }

    pub fn window(&self) -> &[f32] { &self.win }
}
