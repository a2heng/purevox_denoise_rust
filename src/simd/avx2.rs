
use std::arch::x86_64::*;

#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support (see `simd::level`).
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
            i += 16;
        }
        if i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            s = (*pa.add(i)).mul_add(*pb.add(i), s);   // 与向量体同一种舍入
            i += 1;
        }
        s
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support (see `simd::level`).
pub unsafe fn axpy(dst: &mut [f32], a: f32, src: &[f32]) {
    unsafe {
        let n = dst.len().min(src.len());
        let (pd, ps) = (dst.as_mut_ptr(), src.as_ptr());
        let va = _mm256_set1_ps(a);
        let mut i = 0usize;
        while i + 8 <= n {
            let d = _mm256_loadu_ps(pd.add(i));
            let s = _mm256_loadu_ps(ps.add(i));
            _mm256_storeu_ps(pd.add(i), _mm256_fmadd_ps(va, s, d));
            i += 8;
        }
        while i < n {
            *pd.add(i) = a.mul_add(*ps.add(i), *pd.add(i));
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support (see `simd::level`).
/// `x.len() >= in_dim`, `wt.len() >= in_dim * out_dim`, `y.len() >= out_dim`.
pub unsafe fn matvec_t(x: &[f32], wt: &[f32], in_dim: usize, out_dim: usize,
                       bias: Option<&[f32]>, y: &mut [f32]) {
    unsafe {
        let px = x.as_ptr();
        let pw = wt.as_ptr();
        let py = y.as_mut_ptr();
        let mut r = 0usize;
        while r + 16 <= out_dim {
            let mut a0 = if let Some(b) = bias { _mm256_loadu_ps(b.as_ptr().add(r)) } else { _mm256_setzero_ps() };
            let mut a1 = if let Some(b) = bias { _mm256_loadu_ps(b.as_ptr().add(r + 8)) } else { _mm256_setzero_ps() };
            for k in 0..in_dim {
                let xk = _mm256_set1_ps(*px.add(k));
                let base = pw.add(k * out_dim + r);
                a0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(base), a0);
                a1 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(base.add(8)), a1);
            }
            _mm256_storeu_ps(py.add(r), a0);
            _mm256_storeu_ps(py.add(r + 8), a1);
            r += 16;
        }
        while r + 8 <= out_dim {
            let mut a0 = if let Some(b) = bias { _mm256_loadu_ps(b.as_ptr().add(r)) } else { _mm256_setzero_ps() };
            for k in 0..in_dim {
                let xk = _mm256_set1_ps(*px.add(k));
                a0 = _mm256_fmadd_ps(xk, _mm256_loadu_ps(pw.add(k * out_dim + r)), a0);
            }
            _mm256_storeu_ps(py.add(r), a0);
            r += 8;
        }
        for rr in r..out_dim {
            let mut acc = bias.map(|b| *b.as_ptr().add(rr)).unwrap_or(0.0);
            for k in 0..in_dim {
                acc = (*px.add(k)).mul_add(*pw.add(k * out_dim + rr), acc);
            }
            *py.add(rr) = acc;
        }
    }
}


#[inline]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// Requires `avx2` + `fma`.
unsafe fn exp256(x: __m256) -> __m256 {
    {
        let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.336_55)), _mm256_set1_ps(88.376_26));
        let n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.442_695_1)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let mut r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
        r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
        // e^r ≈ 1 + r + r²/2 + … + r⁶/720
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_199_9e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_5e-2));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0));
        let e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(0x7f));
        _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(e, 23)))
    }
}

pub fn exp_tail(x: f32) -> f32 {
    let x = x.clamp(-87.336_55, 88.376_26);
    let n = (x * 1.442_695_1).round_ties_even();
    let mut r = (-n).mul_add(0.693_359_4, x);
    r = n.mul_add(2.121_944_4e-4, r);
    let mut y = 1.987_569_1e-4f32;
    y = y.mul_add(r, 1.398_199_9e-3);
    y = y.mul_add(r, 8.333_452e-3);
    y = y.mul_add(r, 4.166_579_5e-2);
    y = y.mul_add(r, 1.666_666_5e-1);
    y = y.mul_add(r, 0.5);
    y = y.mul_add(r, 1.0);
    y = y.mul_add(r, 1.0);
    y * f32::from_bits((((n as i32) + 0x7f) as u32) << 23)
}

#[inline]
pub fn sigmoid_tail(x: f32) -> f32 { 1.0 / (1.0 + exp_tail(-x)) }

#[inline]
pub fn tanh_tail(x: f32) -> f32 {
    let x = x.clamp(-15.0, 15.0);
    let e = exp_tail(2.0 * x);
    (e - 1.0) / (e + 1.0)
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support.
pub unsafe fn sigmoid_slice(x: &mut [f32]) {
    unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), v));
            _mm256_storeu_ps(p.add(i), _mm256_div_ps(one, _mm256_add_ps(one, e)));
            i += 8;
        }
        while i < n {
            *p.add(i) = sigmoid_tail(*p.add(i));
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support.
pub unsafe fn tanh_slice(x: &mut [f32]) {
    unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);
        let lim = _mm256_set1_ps(15.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let v = _mm256_min_ps(_mm256_max_ps(v, _mm256_sub_ps(zero, lim)), lim);
            let e = exp256(_mm256_mul_ps(two, v));
            _mm256_storeu_ps(p.add(i), _mm256_div_ps(_mm256_sub_ps(e, one), _mm256_add_ps(e, one)));
            i += 8;
        }
        while i < n {
            *p.add(i) = tanh_tail(*p.add(i));
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support.
pub unsafe fn silu_slice(x: &mut [f32]) {
    unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let e = exp256(_mm256_sub_ps(zero, v));
            let s = _mm256_div_ps(one, _mm256_add_ps(one, e));
            _mm256_storeu_ps(p.add(i), _mm256_mul_ps(v, s));
            i += 8;
        }
        while i < n {
            let v = *p.add(i);
            *p.add(i) = v * sigmoid_tail(v);
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The caller must have verified `avx2` and `fma` support.
pub unsafe fn hswish_slice(x: &mut [f32]) {
    unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let three = _mm256_set1_ps(3.0);
        let zero = _mm256_setzero_ps();
        let six = _mm256_set1_ps(6.0);
        let inv6 = _mm256_set1_ps(1.0 / 6.0);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(p.add(i));
            let t = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(v, three), zero), six);
            _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_mul_ps(v, t), inv6));
            i += 8;
        }
        while i < n {
            let v = *p.add(i);
            *p.add(i) = v * (v + 3.0).clamp(0.0, 6.0) / 6.0;
            i += 1;
        }
    }
}
