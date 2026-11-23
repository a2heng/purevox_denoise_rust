
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn axpy(dst: &mut [f32], a: f32, src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += a * s;
    }
}

pub fn matvec_t(x: &[f32], wt: &[f32], in_dim: usize, out_dim: usize,
                bias: Option<&[f32]>, y: &mut [f32]) {
    for r in 0..out_dim {
        let mut acc = bias.map(|b| b[r]).unwrap_or(0.0);
        for k in 0..in_dim {
            acc += x[k] * wt[k * out_dim + r];
        }
        y[r] = acc;
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = sigmoid(*v);
    }
}

pub fn tanh_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.tanh();
    }
}

pub fn hswish_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * (*v + 3.0).clamp(0.0, 6.0) / 6.0;
    }
}

pub fn silu_slice(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * sigmoid(*v);
    }
}
