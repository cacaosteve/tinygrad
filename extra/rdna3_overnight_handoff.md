# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: refresh after next commit (was **`0cd25f5e4`**).

## Headline

**Flash DIRECT ACC_SMALL** ~**684µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed

- Safe SLOAD clustering (store/CF barriers) + tests; HW A/B cluster off/on both correct
- Promote A/B fails on SDPA maxdiff > tol
- **Pack phase0+1 root cause:** minimized **S=T=32**; packs 1–8 = soft-copy base A; **pack 9 = second scratch base** → corrupt (~0.98). `AMD_PACK_SLOAD_BASES=1` (default) blocks that; `MAX=99`+bases=1 matches pack-off (err~1e-4) but ~+10µs — keep pack **off**
- IR dump helper: `extra/rdna3_pack_sload_ir_dump.py`

## Defaults / keep off

- **FMA_MIX**, **PACK_SLOAD_B128**, eviction experiments

## Next leftovers

1. Why packing a second scratch base corrupts (VGPR/prefer_phys?) — optional; pack still not a win
2. Stabilize promote A/B multi-tile exactness (S≥256 flaky vs SKIP=2)
3. Decode partial 34→28
4. eye/GEMM TC_LDS_AB
