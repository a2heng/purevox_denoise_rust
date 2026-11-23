# Porting notes

How the two architectures map onto this crate, and the traps that cost real time.
Read this before changing anything numerically.

## 1. Two families, one `Engine`

| | 05 family (`model05.rs`) | 11d family (`block.rs` …) |
|---|---|---|
| presets | `purevox_denoise_202605a/b` | `purevox_denoise_202611d/f` |
| block | 1×1 grouped conv → BN → SiLU → channel shuffle → causal depthwise conv → BN → SiLU → time/frequency attention | 1×1 → BN → h-swish → causal depthwise conv → BN → FIR gate → residual |
| attention | `ta_gru` (time) + frequency BiGRU over pooled energy | FIR: frequency gate (1×kf) + causal time-energy gate (kt×1) |
| bottleneck | 2× dual-path GRU (frequency BiGRU intra) | `N` dual-path GRUs (depthwise-conv intra) + four-branch memory |
| normalisation | LayerNorm (eps 1e-8) | DyT (`γ·tanh(αx)+β`), LayerNorm only inside memory |
| mask | `erb_expand(σ(logit))` — sigmoid **in the band domain** | `σ(erb_expand(logit))` — sigmoid **in the bin domain** |

The sigmoid position is the single easiest thing to get wrong: swapping it changes
the mask by ~30 dB while everything still "runs".

## 2. Streaming contract

`cache` is a flat `Vec<f32>`; `meta.json` records the offset and shape of every
segment:

* convolution history: one segment per block with `kt > 1`, shape
  `(out_channels, kt−1, in_width)`;
* attention state: `(C, kt−1, 1)` energy history (11d) or `(2C)` GRU hidden state (05);
* dual-path GRU: `(1, width, hidden)` per layer;
* memory: global `(layers, 1, hid)` and sub-band `(1, bands, bhid)`.

Three rules:

1. **The cache replaces zero-padding.** Offline code prepends `kt−1` zero frames;
   streaming concatenates the cache instead. Never do both.
2. **Transposed convolutions** consume `kt` frames (cache + current) and produce
   one; do not pad to `2·kt−1` and slice.
3. **Layout order is part of the public API** — `cache_flat()` must stay
   byte-compatible with the ONNX export, including unused slots (the last decoder
   block of the 05 family has no attention but still occupies a slot).

## 3. Numerical rules

* **Residual shape check.** Compare `(channels, width)`, not `len()`: a block that
  maps `16ch × 120 bins → 32ch × 60 bins` has exactly the same element count and
  must **not** add the residual.
* **Element-wise stability.** A value must not depend on which code path computed
  it: if the vector body uses FMA, the scalar tail must use `mul_add` too,
  otherwise the same sample differs by ~3e-8 between a batched pass and a
  single-hop pass.
* **Accumulation order** may differ between SIMD and scalar builds, but not
  between the streaming and offline paths of one build.
* **`exp` must be clamped** to `[−87.3, 88.4]` before building `2^n` from its
  exponent bits; `tanh` inputs are clamped to `±15`.

## 4. Implementation order

Each step is verifiable against the shipped golden vectors:

1. STFT / iSTFT (≥130 dB), ERB banding (≥100 dB),
2. one block (per-intermediate maxdiff ≤ 1e-5),
3. encoder + decoder,
4. bottleneck (dual-path GRUs, optional memory),
5. whole chain offline (≥110 dB),
6. streaming hops + cache (≥100 dB / ≥110 dB),
7. causality: modify the second half of the input, the first half must be unchanged.

`purevox_verify` runs steps 5–6 for every preset in `models/`.

## 5. Performance notes

* Pre-transpose `Linear` and GRU weights once at load time and use
  `simd::matvec_t` (one FMA covers 8 outputs). This alone was ~25% end-to-end.
* 1×1 convolutions: block over positions (128) so the input plane stays in L1.
* Depthwise convolutions: pad the plane once, then per-tap `axpy`; for a stride of
  2 split the row into even/odd phases so the taps stay contiguous.
* Do **not** build the full `*Trace` structures on the inference path — they clone
  every intermediate. `forward` and `forward_trace` are separate for that reason.
* `-C target-cpu=native` was slower than baseline on the machine we measured
  (the auto-vectorised depthwise loops regress); the hand-written kernels are the
  reason this crate is fast.
