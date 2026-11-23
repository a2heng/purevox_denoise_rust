use purevox_denoise_rust::simd::{self, Level};

fn rng(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s as f32 / u32::MAX as f32 - 0.5) * 4.0
    }).collect()
}

#[test]
fn simd_matches_scalar() {
    println!("simd level = {:?}", simd::level());
    let a = rng(1027, 7);
    let b = rng(1027, 11);
    let d_s = simd::dot_scalar(&a, &b);
    let d_v = simd::dot(&a, &b);
    let rel = (d_s - d_v).abs() / d_s.abs().max(1.0);
    println!("dot {d_s} vs {d_v}  rel {rel:.3e}");
    assert!(rel < 1e-5);

    let mut y_s = rng(1027, 3);
    let mut y_v = y_s.clone();
    simd::axpy_scalar(&mut y_s, 0.37, &a);
    simd::axpy(&mut y_v, 0.37, &a);
    let md = y_s.iter().zip(&y_v).map(|(x, z)| (x - z).abs()).fold(0.0f32, f32::max);
    println!("axpy maxdiff {md:.3e}");
    assert!(md < 1e-6);

    let x = rng(1003, 5);
    let mut s_s = x.clone();
    simd::sigmoid_slice_scalar(&mut s_s);
    let mut s_v = x.clone();
    simd::sigmoid_slice(&mut s_v);
    let md = s_s.iter().zip(&s_v).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    println!("sigmoid maxdiff {md:.3e}");
    assert!(md < 1e-6);

    let mut t_s = x.clone();
    simd::tanh_slice_scalar(&mut t_s);
    let mut t_v = x.clone();
    simd::tanh_slice(&mut t_v);
    let md = t_s.iter().zip(&t_v).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    println!("tanh maxdiff {md:.3e}");
    assert!(md < 1e-6);

    let mut h_s = x.clone();
    simd::hswish_slice_scalar(&mut h_s);
    let mut h_v = x.clone();
    simd::hswish_slice(&mut h_v);
    let md = h_s.iter().zip(&h_v).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    println!("hswish maxdiff {md:.3e}");
    assert!(md < 1e-6);

    if simd::level() == Level::Avx2 {
        let mut e = vec![-200.0f32, -88.0, -1.0, 0.0, 1.0, 88.0, 200.0, f32::MIN / 2.0];
        simd::tanh_slice(&mut e);
        for v in &e { assert!(v.is_finite(), "tanh 产生非有限值: {v}"); }
        let mut e = vec![-200.0f32, -88.0, 0.0, 88.0, 200.0];
        simd::sigmoid_slice(&mut e);
        for v in &e { assert!(v.is_finite(), "sigmoid 产生非有限值: {v}"); }
    }
}
