use purevox_denoise_rust::ops::conv::conv2d;


#[test]
fn conv1x1_batch_invariance() {
    let (c, o, w) = (3usize, 4usize, 240usize);
    let t = 201usize;
    let x: Vec<f32> = (0..c * t * w).map(|i| ((i * 37 % 1000) as f32 / 500.0) - 1.0).collect();
    let wt: Vec<f32> = (0..o * c * w).map(|i| ((i * 53 % 997) as f32 / 900.0) - 0.5).collect();
    let wt = &wt[..o * c * 1 * 1];
    let b: Vec<f32> = (0..o).map(|i| i as f32 * 0.1).collect();

    let y_full = conv2d(&x, 1, c, t, w, wt, o, 1, 1, Some(&b), 1, 1, 0, 0, 1);
    let mut frame0 = vec![0.0f32; c * w];
    for ch in 0..c {
        frame0[ch * w..(ch + 1) * w].copy_from_slice(&x[ch * t * w..ch * t * w + w]);
    }
    let y_one = conv2d(&frame0, 1, c, 1, w, wt, o, 1, 1, Some(&b), 1, 1, 0, 0, 1);
    let mut md = 0.0f32;
    for oc in 0..o {
        for k in 0..w {
            md = md.max((y_full[oc * t * w + k] - y_one[oc * w + k]).abs());
        }
    }
    println!("conv1x1 batch invariance maxdiff {md:.3e}");
    assert_eq!(md, 0.0);

    let w2 = 95usize;
    let x: Vec<f32> = (0..c * t * w2).map(|i| ((i * 41 % 1000) as f32 / 500.0) - 1.0).collect();
    let wt2: Vec<f32> = (0..o * c).map(|i| ((i * 59 % 997) as f32 / 900.0) - 0.5).collect();
    let y_full = conv2d(&x, 1, c, t, w2, &wt2, o, 1, 1, Some(&b), 1, 1, 0, 0, 1);
    let mut frame0 = vec![0.0f32; c * w2];
    for ch in 0..c { frame0[ch * w2..(ch + 1) * w2].copy_from_slice(&x[ch * t * w2..ch * t * w2 + w2]); }
    let y_one = conv2d(&frame0, 1, c, 1, w2, &wt2, o, 1, 1, Some(&b), 1, 1, 0, 0, 1);
    let mut md = 0.0f32;
    for oc in 0..o { for k in 0..w2 { md = md.max((y_full[oc * t * w2 + k] - y_one[oc * w2 + k]).abs()); } }
    println!("conv1x1 (w=95) batch invariance maxdiff {md:.3e}");
    assert_eq!(md, 0.0);
}
