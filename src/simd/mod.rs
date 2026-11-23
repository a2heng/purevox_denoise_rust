
use std::sync::OnceLock;

pub mod reference;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Scalar,
    Avx2,
}

fn env_off(name: &str) -> bool {
    std::env::var(name).map(|v| v == "0").unwrap_or(false)
}

pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        if env_off("PUREVOX_SIMD") {
            return Level::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Level::Avx2;
            }
        }
        Level::Scalar
    })
}

fn dot_level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| if env_off("PUREVOX_SIMD_DOT") { Level::Scalar } else { level() })
}

fn axpy_level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| if env_off("PUREVOX_SIMD_AXPY") { Level::Scalar } else { level() })
}

fn act_level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| if env_off("PUREVOX_SIMD_ACT") { Level::Scalar } else { level() })
}


pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if dot_level() == Level::Avx2 {
            return unsafe { avx2::dot(a, b) };
        }
    }
    reference::dot(a, b)
}

pub fn axpy(dst: &mut [f32], a: f32, src: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if axpy_level() == Level::Avx2 {
            unsafe { avx2::axpy(dst, a, src) };
            return;
        }
    }
    reference::axpy(dst, a, src);
}

pub fn matvec_t(x: &[f32], wt: &[f32], in_dim: usize, out_dim: usize,
                bias: Option<&[f32]>, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if dot_level() == Level::Avx2 {
            unsafe { avx2::matvec_t(x, wt, in_dim, out_dim, bias, y) };
            return;
        }
    }
    reference::matvec_t(x, wt, in_dim, out_dim, bias, y);
}

pub fn sigmoid_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if act_level() == Level::Avx2 {
            unsafe { avx2::sigmoid_slice(x) };
            return;
        }
    }
    reference::sigmoid_slice(x);
}

pub fn tanh_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if act_level() == Level::Avx2 {
            unsafe { avx2::tanh_slice(x) };
            return;
        }
    }
    reference::tanh_slice(x);
}

pub fn silu_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if act_level() == Level::Avx2 {
            unsafe { avx2::silu_slice(x) };
            return;
        }
    }
    reference::silu_slice(x);
}

pub fn hswish_slice(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if act_level() == Level::Avx2 {
            unsafe { avx2::hswish_slice(x) };
            return;
        }
    }
    reference::hswish_slice(x);
}

pub use reference::{axpy as axpy_scalar, dot as dot_scalar,
                    hswish_slice as hswish_slice_scalar,
                    sigmoid_slice as sigmoid_slice_scalar,
                    silu_slice as silu_slice_scalar,
                    tanh_slice as tanh_slice_scalar};
