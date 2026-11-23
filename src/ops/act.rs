
#[inline]
pub fn hswish(x: f32) -> f32 {
    x * (x + 3.0).clamp(0.0, 6.0) / 6.0
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn hswish_slice(x: &mut [f32]) { crate::simd::hswish_slice(x); }

pub fn sigmoid_slice(x: &mut [f32]) { crate::simd::sigmoid_slice(x); }

pub fn silu_slice(x: &mut [f32]) { crate::simd::silu_slice(x); }
