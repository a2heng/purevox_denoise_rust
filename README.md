# purevox_denoise_rust

Pure-Rust, f32 inference for **PureVox Denoise** — the speech-denoising member of the
**PureVox** model family. No ONNX, no ONNX Runtime, no C/C++. Just `cargo`.

> **About the name.** *PureVox* is the model **family** (denoising, and later other
> speech-enhancement tasks such as AEC). `purevox_denoise_rust` is the **denoise**
> member; sibling crates will follow the same `<family>_<task>_rust` convention.

* **Four presets**, trained for 48 kHz / 10 ms hops.
* **Streaming**: one call per hop, all state in a flat `cache` vector (same layout
  as the ONNX export it mirrors).
* **Whole-file**: `process_wav` (streaming cache, sample-aligned) or
  `process_wav_offline` (one batched pass; agrees with streaming to ~1e-7).
* **Portable**: hand-written AVX2/FMA kernels with a plain scalar reference and a
  runtime dispatch — every operator also has an intrinsics-free implementation.

## Quick start

```toml
[dependencies]
purevox_denoise_rust = { git = "https://github.com/a2heng/purevox_denoise_rust" }
```

```rust
use purevox_denoise_rust::{Engine, models_dir};

// One hop = 10 ms of audio = 480 samples = 481-bin complex spectrum.
let mut engine = Engine::load(models_dir().join("purevox_denoise_202605a"))?;
let mut spec = vec![0.0f32; engine.spec_len()];   // [re(0..481), im(0..481)]
let out = engine.process_hop(&spec);              // denoised, same shape
# Ok::<(), purevox_denoise_rust::WeightError>(())
```

The FFT and the overlap-add stay with the caller, exactly like the spectral
in/out contract. `Stft` / `Spectrogram` are provided if you want the same framing
as the reference (`n_fft 960`, `hop 480`, periodic Hann, `center = true`).

### Command line

```bash
cargo run --release --bin purevox -- --list
cargo run --release --bin purevox -- --model models/purevox_denoise_202605a \
        --in noisy.wav --out clean.wav
cargo run --release --bin purevox -- --model models/purevox_denoise_202605a \
        --in noisy.wav --out clean.wav --offline
```

### Self-check

```bash
cargo test --release          # SIMD equivalence + parity against models/*/golden.safetensors
cargo run --release --bin purevox_verify      # per-preset SNR table
```

## Presets

| preset | frontend | params | cache (f32) | notes |
|---|---|---:|---:|---|
| `purevox_denoise_202605a` | 60 + 35 ERB bands | 400,549 | 10,208 | grouped-conv encoder, time/frequency attention |
| `purevox_denoise_202605b` | 100 + 41 ERB bands | 387,925 | 14,024 | same family, wider frontend |
| `purevox_denoise_202611d` | 160 + 80 ERB bands | 326,172 | 25,588 | h-swish blocks, FIR gate, 2 dual-path GRUs, bottleneck memory |
| `purevox_denoise_202611f` | 160 + 80 ERB bands | 364,710 | 32,308 | as `202611d` with wider channels and 3 dual-path GRUs |

All four are 48 kHz, mono, real-valued magnitude mask (noisy phase is preserved).
Weights and code are both **BSD-3-Clause**.


## Parity and performance

Measured against the PyTorch reference on the same weights (`purevox_verify`):

| preset | whole file (offline) | worst streaming hop | cache |
|---|---:|---:|---:|
| `purevox_denoise_202605a` | 116.7 dB | 109.3 dB | 121.8 dB |
| `purevox_denoise_202605b` | 114.0 dB | 110.5 dB | 119.7 dB |
| `purevox_denoise_202611d` | 114.2 dB | 109.5 dB | 123.4 dB |
| `purevox_denoise_202611f` | 115.6 dB | 112.6 dB | 123.5 dB |

Single thread, one core pinned (AMD Ryzen 5 7500F), release build:

| preset | p50 | p99 | RTF (10 ms budget) |
|---|---:|---:|---:|
| `202605a` | 0.25 ms | 0.37 ms | 0.025 |
| `202605b` | 0.35 ms | 0.36 ms | 0.035 |
| `202611d` | 0.43 ms | 0.63 ms | 0.043 |
| `202611f` | 0.68 ms | 0.69 ms | 0.068 |

Without SIMD (scalar reference path, `PUREVOX_SIMD=0`): 1.4–1.8 ms/hop — still well
inside the 10 ms budget.

## Layout

```
src/
  lib.rs            crate docs, presets, re-exports
  engine.rs         Engine: presets, one call per hop, whole-file helpers
  meta.rs weights.rs wavio.rs
  stft.rs erb.rs    STFT/iSTFT + ERB banding
  simd/             mod (dispatch) · reference (scalar) · avx2 (x86_64)
  ops/              conv (1×1 / depthwise / im2col), linear, norm, act
  gru.rs            multi-layer / bidirectional GRU with streaming step
  block.rs net.rs dpgrnn.rs memory.rs model.rs stream.rs   ← 11d family
  model05.rs                                               ← 05 family
  bin/              purevox (wav→wav), purevox_verify
models/<preset>/    weights.safetensors + meta.json + golden.safetensors
docs/porting.md     how the two architectures map onto this code, and the traps
```

## License

BSD-3-Clause. See [LICENSE](LICENSE). The bundled model weights are released under the
same license, stated again in [`models/LICENSE`](models/LICENSE) (© 2026 a2heng).

---

# purevox_denoise_rust（中文）

**PureVox Denoise** 的纯 Rust f32 推理实现 —— **PureVox 是模型系列名**（降噪，以及以后的
AEC 等语音增强任务），`purevox_denoise_rust` 是这个系列里的**降噪**成员；后续任务会按
`<系列>_<任务>_rust` 的命名单独出库。

不依赖 ONNX / ONNX Runtime / C++，只依赖 `cargo`。

## 特点

- **四个预设**，48 kHz、10 ms 一 hop。
- **流式**：每 hop 一次调用，全部状态放在一个扁平 `cache` 里（顺序与镜像的 ONNX 契约一致）。
- **整段**：`process_wav`（走流式缓存，输出与输入逐样本对齐）或 `process_wav_offline`
  （整段一次算完；与流式差 ~1e-7）。
- **可移植**：手写 AVX2/FMA 内核 + 纯标量参考实现 + 运行时派发；每个算子在
  没有 intrinsics 的平台上都有等价实现。

## 用法

```rust
use purevox_denoise_rust::{Engine, models_dir};

let mut engine = Engine::load(models_dir().join("purevox_denoise_202605a"))?;
let spec = vec![0.0f32; engine.spec_len()];   // [实部 481, 虚部 481]
let out = engine.process_hop(&spec);
# Ok::<(), purevox_denoise_rust::WeightError>(())
```

FFT 与 OLA 在调用方（与谱进谱出的契约一致）；仓库里也提供 `Stft` / `Spectrogram`，
分帧口径与参考实现完全一致（`n_fft 960`、`hop 480`、周期 Hann、`center = true`）。

```bash
cargo run --release --bin purevox -- --list
cargo run --release --bin purevox -- --model models/purevox_denoise_202605a \
        --in noisy.wav --out clean.wav
cargo test --release                          # SIMD 等价 + 与 golden 的对齐
cargo run --release --bin purevox_verify      # 逐预设 SNR 表
```

## 预设

| 预设 | 前端 | 参数 | cache | 结构 |
|---|---|---:|---:|---|
| `purevox_denoise_202605a` | 60 + 35 带 | 400,549 | 10,208 | 组卷积编码器 + 时/频注意力 |
| `purevox_denoise_202605b` | 100 + 41 带 | 387,925 | 14,024 | 同族，前端更宽 |
| `purevox_denoise_202611d` | 160 + 80 带 | 326,172 | 25,588 | h-swish 块 + FIR 门 + 2 层双路径 GRU + 瓶颈记忆 |
| `purevox_denoise_202611f` | 160 + 80 带 | 364,710 | 32,308 | 同 d，更宽通道 + 3 层双路径 GRU |

四个预设都是 48 kHz、实数幅度掩码（保留含噪相位）。**代码与权重均为 BSD-3-Clause**。


## 对齐与性能

与 PyTorch 参考实现同权重对比（`purevox_verify`）：整段离线 114~117 dB、最差流式 hop
109~111 dB、cache 120~123 dB。单线程绑核（Ryzen 5 7500F）每 hop：0.25~0.62 ms
（10 ms 预算，余量 16~40×）；标量路径 1.4~1.8 ms/hop。

## 许可

BSD-3-Clause，见 [LICENSE](LICENSE)；随仓库分发的四份模型权重采用同一许可，
并在 [`models/LICENSE`](models/LICENSE) 里单独声明（© 2026 a2heng）。
