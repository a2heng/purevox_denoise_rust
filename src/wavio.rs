
use std::path::Path;

pub fn read_wav(path: impl AsRef<Path>) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).expect("打开 wav 失败");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
    };
    let mono = if ch == 1 { samples } else {
        samples.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect()
    };
    (mono, spec.sample_rate)
}

pub fn write_wav(path: impl AsRef<Path>, wav: &[f32], sr: u32) {
    let spec = hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 32,
                               sample_format: hound::SampleFormat::Float };
    let mut w = hound::WavWriter::create(path, spec).expect("创建 wav 失败");
    for &s in wav { w.write_sample(s).unwrap(); }
    w.finalize().unwrap();
}
