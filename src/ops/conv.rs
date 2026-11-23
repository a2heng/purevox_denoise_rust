
use crate::weights::{WeightError, Weights};

#[derive(Debug, Clone)]
pub struct Conv2d {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub in_ch: usize,
    pub out_ch: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub ph: usize,
    pub pw: usize,
    pub groups: usize,
}

impl Conv2d {
    pub fn from_weights(w: &Weights, prefix: &str, ph: usize, pw: usize,
                        sh: usize, sw: usize, groups: usize) -> Result<Self, WeightError> {
        let wt = w.get(&format!("{prefix}.weight"))?;
        let (out_ch, cg, kh, kw) = (wt.shape[0], wt.shape[1], wt.shape[2], wt.shape[3]);
        let b = w.map.get(&format!("{prefix}.bias")).map(|t| t.data.clone());
        Ok(Self {
            w: wt.data.clone(), b,
            in_ch: cg * groups, out_ch, kh, kw, sh, sw, ph, pw, groups,
        })
    }

    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        ((h + 2 * self.ph).saturating_sub(self.kh) / self.sh + 1,
         (w + 2 * self.pw).saturating_sub(self.kw) / self.sw + 1)
    }

    pub fn forward(&self, x: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (oh, ow) = self.out_hw(h, w);
        let y = conv2d(x, n, self.in_ch, h, w, &self.w, self.out_ch, self.kh, self.kw,
                       self.b.as_deref(), self.sh, self.sw, self.ph, self.pw, self.groups);
        (y, oh, ow)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &[f32], n: usize, c: usize, h: usize, w: usize,
    wt: &[f32], o: usize, kh: usize, kw: usize, bias: Option<&[f32]>,
    sh: usize, sw: usize, ph: usize, pw: usize, groups: usize,
) -> Vec<f32> {
    assert_eq!(c % groups, 0);
    assert_eq!(o % groups, 0);
    assert_eq!(x.len(), n * c * h * w);
    let cg = c / groups;
    let og = o / groups;
    assert_eq!(wt.len(), o * cg * kh * kw);
    let oh = (h + 2 * ph).saturating_sub(kh) / sh + 1;
    let ow = (w + 2 * pw).saturating_sub(kw) / sw + 1;

    if kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 && groups == 1 {
        let plane = h * w;
        let mut y = vec![0.0f32; n * o * plane];
        const P: usize = 128;
        let mut acc = vec![0.0f32; P];
        for ni in 0..n {
            let xb = &x[ni * c * plane..(ni + 1) * c * plane];
            let yb = &mut y[ni * o * plane..(ni + 1) * o * plane];
            let mut p0 = 0;
            while p0 < plane {
                let p1 = (p0 + P).min(plane);
                let len = p1 - p0;
                for oc in 0..o {
                    let a = &mut acc[..len];
                    if let Some(b) = bias { a.fill(b[oc]); } else { a.fill(0.0); }
                    let wrow = &wt[oc * c..(oc + 1) * c];
                    for ic in 0..c {
                        let wv = wrow[ic];
                        let xp = &xb[ic * plane + p0..ic * plane + p1];
                        crate::simd::axpy(a, wv, xp);
                    }
                    yb[oc * plane + p0..oc * plane + p1].copy_from_slice(a);
                }
                p0 = p1;
            }
        }
        return y;
    }

    if groups == c && c == o && cg == 1 {
        let hp = h + 2 * ph;
        let wp = w + 2 * pw;
        let mut pp = vec![0.0f32; hp * wp];
        let mut y = vec![0.0f32; n * o * oh * ow];
        for ni in 0..n {
            for oc in 0..o {
                let xp = &x[(ni * c + oc) * h * w..(ni * c + oc + 1) * h * w];
                for iy in 0..h {
                    pp[(iy + ph) * wp + pw..(iy + ph) * wp + pw + w]
                        .copy_from_slice(&xp[iy * w..(iy + 1) * w]);
                }
                let yb = &mut y[(ni * o + oc) * oh * ow..(ni * o + oc + 1) * oh * ow];
                let wbase = &wt[oc * kh * kw..(oc + 1) * kh * kw];
                let bv = bias.map(|b| b[oc]).unwrap_or(0.0);
                if sh == 1 && sw == 2 {
                    let half = wp.div_ceil(2);
                    let mut ev = vec![0.0f32; half];
                    let mut od = vec![0.0f32; half];
                    for oy in 0..oh {
                        let out_row = &mut yb[oy * ow..(oy + 1) * ow];
                        out_row.fill(bv);
                        for ky in 0..kh {
                            let base = (oy + ky) * wp;
                            for j in 0..half {
                                let e = 2 * j;
                                ev[j] = pp[base + e];
                                od[j] = if e + 1 < wp { pp[base + e + 1] } else { 0.0 };
                            }
                            let wr = &wbase[ky * kw..(ky + 1) * kw];
                            for kx in 0..kw {
                                let off = kx / 2;
                                let src = if kx % 2 == 0 { &ev } else { &od };
                                if off + ow <= src.len() {
                                    crate::simd::axpy(out_row, wr[kx], &src[off..off + ow]);
                                }
                            }
                        }
                    }
                } else if sh == 1 && sw == 1 {
                    for oy in 0..oh {
                        let out_row = &mut yb[oy * ow..(oy + 1) * ow];
                        out_row.fill(bv);
                        for ky in 0..kh {
                            let base = (oy + ky) * wp;
                            let wr = &wbase[ky * kw..(ky + 1) * kw];
                            for kx in 0..kw {
                                crate::simd::axpy(out_row, wr[kx], &pp[base + kx..base + kx + ow]);
                            }
                        }
                    }
                } else {
                    for oy in 0..oh {
                        let row0 = oy * sh;
                        for ox in 0..ow {
                            let ix0 = ox * sw;
                            let mut acc = bv;
                            for ky in 0..kh {
                                let base = (row0 + ky) * wp + ix0;
                                let wr = &wbase[ky * kw..(ky + 1) * kw];
                                for kx in 0..kw {
                                    acc += pp[base + kx] * wr[kx];
                                }
                            }
                            yb[oy * ow + ox] = acc;
                        }
                    }
                }
            }
        }
        return y;
    }

    let mut y = vec![0.0f32; n * o * oh * ow];
    let mut patch = vec![0.0f32; cg * kh * kw];
    for ni in 0..n {
        for g in 0..groups {
            let ic0 = g * cg;
            let oc0 = g * og;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut pi = 0;
                    for ic in 0..cg {
                        let xp = &x[(ni * c + ic0 + ic) * h * w..(ni * c + ic0 + ic + 1) * h * w];
                        for ky in 0..kh {
                            let iy = oy * sh + ky;
                            if iy < ph { patch[pi..pi + kw].fill(0.0); pi += kw; continue; }
                            let iy = iy - ph;
                            if iy >= h { patch[pi..pi + kw].fill(0.0); pi += kw; continue; }
                            let row = &xp[iy * w..iy * w + w];
                            for kx in 0..kw {
                                let ix = ox * sw + kx;
                                patch[pi] = if ix < pw || ix - pw >= w { 0.0 } else { row[ix - pw] };
                                pi += 1;
                            }
                        }
                    }
                    for j in 0..og {
                        let oc = oc0 + j;
                        let wrow = &wt[oc * cg * kh * kw..(oc + 1) * cg * kh * kw];
                        let mut acc = bias.map(|b| b[oc]).unwrap_or(0.0);
                        acc += crate::simd::dot(wrow, &patch);
                        y[((ni * o + oc) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
    }
    y
}

#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub in_ch: usize,
    pub out_ch: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub ph: usize,
    pub pw: usize,
    pub oph: usize,
    pub opw: usize,
    pub groups: usize,
}

impl ConvTranspose2d {
    pub fn from_weights(w: &Weights, prefix: &str, ph: usize, pw: usize,
                        sh: usize, sw: usize, groups: usize) -> Result<Self, WeightError> {
        let wt = w.get(&format!("{prefix}.weight"))?;
        let (in_ch, cg_out, kh, kw) = (wt.shape[0], wt.shape[1], wt.shape[2], wt.shape[3]);
        let b = w.map.get(&format!("{prefix}.bias")).map(|t| t.data.clone());
        Ok(Self { w: wt.data.clone(), b, in_ch, out_ch: cg_out * groups,
                  kh, kw, sh, sw, ph, pw, oph: 0, opw: 0, groups })
    }

    pub fn forward(&self, x: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let oh = (h - 1) * self.sh + self.kh - 2 * self.ph + self.oph;
        let ow = (w - 1) * self.sw + self.kw - 2 * self.pw + self.opw;
        let mut y = vec![0.0f32; n * self.out_ch * oh * ow];
        let cg_out = self.out_ch / self.groups;
        let cg_in = self.in_ch / self.groups;
        for ni in 0..n {
            for ic in 0..self.in_ch {
                let g = ic / cg_in;
                let xp = &x[(ni * self.in_ch + ic) * h * w..(ni * self.in_ch + ic + 1) * h * w];
                for iy in 0..h {
                    for ix in 0..w {
                        let xv = xp[iy * w + ix];
                        if xv == 0.0 { continue; }
                        for ky in 0..self.kh {
                            let oy = (iy * self.sh + ky) as isize - self.ph as isize;
                            if oy < 0 || oy >= oh as isize { continue; }
                            for kx in 0..self.kw {
                                let ox = (ix * self.sw + kx) as isize - self.pw as isize;
                                if ox < 0 || ox >= ow as isize { continue; }
                                for ocg in 0..cg_out {
                                    let oc = g * cg_out + ocg;
                                    let wv = self.w[((ic * cg_out + ocg) * self.kh + ky) * self.kw + kx];
                                    y[((ni * self.out_ch + oc) * oh + oy as usize) * ow + ox as usize]
                                        += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(b) = &self.b {
            for ni in 0..n {
                for oc in 0..self.out_ch {
                    let base = (ni * self.out_ch + oc) * oh * ow;
                    for v in &mut y[base..base + oh * ow] { *v += b[oc]; }
                }
            }
        }
        (y, oh, ow)
    }
}

#[derive(Debug, Clone)]
pub enum Conv {
    Direct(Conv2d),
    Transposed(ConvTranspose2d),
}

impl Conv {
    pub fn out_ch(&self) -> usize {
        match self { Conv::Direct(c) => c.out_ch, Conv::Transposed(c) => c.out_ch }
    }
    pub fn in_ch(&self) -> usize {
        match self { Conv::Direct(c) => c.in_ch, Conv::Transposed(c) => c.in_ch }
    }
    pub fn kernel(&self) -> (usize, usize) {
        match self { Conv::Direct(c) => (c.kh, c.kw), Conv::Transposed(c) => (c.kh, c.kw) }
    }
    pub fn groups(&self) -> usize {
        match self { Conv::Direct(c) => c.groups, Conv::Transposed(c) => c.groups }
    }
    pub fn forward(&self, x: &[f32], n: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        match self {
            Conv::Direct(c) => c.forward(x, n, h, w),
            Conv::Transposed(c) => c.forward(x, n, h, w),
        }
    }
}
