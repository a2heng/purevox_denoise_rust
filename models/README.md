# Bundled models

Four presets, each a directory: `meta.json` (architecture + cache layout),
`weights.safetensors`, and `golden.safetensors` (one second of audio used by
`cargo test` and `purevox_verify`).

| preset | frontend | params | cache (f32) |
|---|---|---:|---:|
| `purevox_denoise_202605a` | 60 + 35 ERB bands | 400,549 | 10,208 |
| `purevox_denoise_202605b` | 100 + 41 ERB bands | 387,925 | 14,024 |
| `purevox_denoise_202611d` | 160 + 80 ERB bands | 326,172 | 25,588 |
| `purevox_denoise_202611f` | 160 + 80 ERB bands | 364,710 | 32,308 |

All four are 48 kHz, mono, streaming (10 ms hop), real-valued magnitude mask.

**License:** BSD-3-Clause, same as the code — see `LICENSE` in this directory
(© 2026 a2heng).
