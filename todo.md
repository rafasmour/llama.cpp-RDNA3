# RDNA3 (RX 7900 XT, gfx1100) tuning TODO

Status 2026-11-11: all four code fixes are open as PRs on the fork
(rafasmour/llama.cpp-RDNA3, tracking issue #15). Branch `rdna3-integration`
has them all merged. Numbers below were measured on this card (ROCm 10.1,
2300 MHz / 231 W cap), not estimated.

Scope: **global** = helps any model (sometimes per quant type);
**family** = one model architecture; **model** = one preset/config.

## 0. Measure first - done

- [x] Build master with tests in Docker (`rdna3-build:local` = ROCm 10.1 `-full`
      plus cmake/ninja; build tree `build-hip/`, about 25 min cold with 4 jobs).
- [x] `test-backend-ops perf` for FLASH_ATTN_EXT, MUL_MAT, MUL_MAT_ID (cases in PR #5).
      Note: MUL_MAT weights below ~80 MB stay in the Infinity Cache, so those
      numbers show the compute bound. Use real models for the memory bound.
- [x] `rocprofv3 --kernel-trace`: Ornith 9B plain decode = 899 dispatches/token,
      `mul_mat_vec_q` 8.4 of 10.5 ms kernel time, launch gaps ~1.6 ms.
      Not done: trace of a DFlash step and of Qwen3.6-35B-A3B decode.
- [x] `llama-bench -d 0,32768,65536` f16 vs q8_0 KV (Ornith, Gemma 4 31B, Qwen3.8 27B).

## 1. Free / config

- [ ] **Rebuild production from the fork**: set `LLAMA_REPO` to the fork and
      `LLAMA_REF=rdna3-integration` (or master after the PRs are merged).
- [x] **`--backend-sampling`** tried: +3% plain decode with a greedy chain, but
      `top-p` needs ARGSORT over the vocab on the device (not supported), so the
      production sampler chain falls back to the CPU. Not useful yet.
- [x] **DFlash n-max sweep** (`ornith-1.5-9b-q4-dflash`, 1.2k-token prompt, greedy):
      n-max 2 = 110.1 t/s, 3 = 115.0, 4 = 109.0, 6 = 110.9. Try `spec-draft-n-max = 3`
      on the real workload before changing the preset.

## 2. Code fixes

- [x] **FA: read q8_0/q4_0 KV directly** (PR #6, issue #2). Tile kernel dequantizes
      while loading; used for head size 256/512 with up to 8 tokens.
      Gemma 4 31B q8_0: 30.5 / 21.5 / 17.2 -> 32.0 / 27.9 / 25.7 t/s at depth 0 / 32k / 64k.
      Ornith q8_0: 79.0 / 63.3 / 51.8 -> 80.4 / 75.0 / 69.7. Qwen3.8 27B: 24.6 -> 28.6 at 32k.
      The cheap VEC routing was dropped: the tile path is faster for all of 1-8 tokens.
- [x] **Verify path for 2-8 tokens** (PR #7, issue #3; PR #11, issue #9).
      The suspected per-column unpack was not the cause in the unfused kernel (the
      compiler hoists it), only in the fused one. Multi-column fusion with that fixed
      plus one input quantization per layer: 5-token batch 267 -> 289 t/s,
      DFlash preset 101.6 -> 109.0 t/s, Qwen3.8 27B MTP 63.6 -> 67.7 t/s.
- [x] **Fuse small kernels in Qwen3.5-family layers** (PR #8, #11, #12).
      899 -> 787 dispatches per token on Ornith 9B (slot order once per graph is on
      top of that for speculative presets: 69 per step). Real launch cost is ~1.8 us
      of gap per dispatch, less than the 2.8-3.9 us assumed, so the 10-15% estimate
      was too high: these three give about 3%.
- [x] **Retune RDNA3 MMVQ table for the 7900 XT** (PR #14, issue #13).
      4 warps for Q4_K/Q5_K/Q6_K and the Q4_0 family, 1 warp for short rows.
      Ornith Q4_K_M 84.95 -> 87.88 t/s, Qwen3.8 27B Q4_0 31.67 -> 32.76, Mellum Q5_K_M MoE +5.6%.

Not worth it (measured): HIP graph replay / per-shape graph cache (no effect, graphs
on vs off is 83.0 vs 82.7 t/s), two rows per block or more warps for 2-8 columns,
lower MMQ threshold (MMQ only wins from 8 columns), MMA direct KV read.
Still not worth it: prefill/MMQ tuning, wave64 builds, a rewrite.

## Next ideas

- ARGSORT over the vocab on the device, so `--backend-sampling` works with top-p.
- GDN layers: state gather/concat/copy and gate math (~100 more dispatches/token).
- Qwen3.6-35B-A3B decode: take the kernel trace; this run did not profile the MoE path.
- FA with f16 KV at 3+ tokens and head size 512 is 2.6x slower than at 1-2 tokens.

## Research handoff

- **Utilization after the fixes:** Ornith 9B Q4_K_M decode 87.9 t/s (was 83.0).
  Before: 54% (Ornith 9B Q4_K_M), 60% (Qwen3.8 27B Q4_0), 45% (Gemma 4 31B IQ3_XXS),
  ~23% (Qwen3.6-35B-A3B MoE) of 800 GB/s. From the trace `mul_mat_vec_q` alone ran
  at ~665 GB/s on the 9B.
- **Batch cost on the 9B (after):** 1 token 11.4 ms, 2: 13.2, 3: 14.1, 4: 15.5,
  5: 17.3, 8: 23.1 ms. MMQ (9-16 tokens) is flat at ~21.3 ms.
- **FA kernel choice on RDNA3 now:** quantized KV with <= 8 tokens and head size
  256/512 -> TILE with direct read; f16 KV unchanged (TILE for 1-4 tokens at GQA 4,
  MMA above); D=512 is always TILE.
  Head layouts: Ornith 16/4 D256; Qwen3.8 27B 24/4 D256; Qwen3.6 16/2 D256;
  Gemma 4 31B SWA 32/16 D256, global (every 6th of 60) 32/4 D512. Gemma's V cache is
  a separate tensor, not a view of K.
- **Real tensor types:** Ornith Q4_K_M has Q4_K gate/up and Q6_K in ffn_down/qkv/output;
  Qwen3.8 27B Q4_0 has Q4_1 ffn_down; Qwen3.6 has IQ3_S gate/up experts, IQ4_XS down
  experts, Q8_0 everything else (256 experts, 8 used, n_embd 2048, expert FFN 512).
- **HIP launch overhead:** ROCm/TheRock#9241 reports 2.8-3.9 us per kernel in graph
  replay; the trace here shows ~1.8 us average gap and no gain from graphs.
- **hipEngine (shisa-ai, AGPL):** at most +10-30% over llama.cpp HIP decode; MTP at
  parity (69.5 vs 69.4, Strix Halo).
- **Wave64 experiment (llama.cpp#20934):** +0-16% tg; uses an AMD-unsupported flag.
