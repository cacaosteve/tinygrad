# Overnight RDNA3 — tip pending push

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**680–685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. Keep **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## This session

- Addr remat: CAST fix + leaf-only; deep remat → MMU. Off outside TC_LDS.
- SLOAD×4 pack (`AMD_PACK_SLOAD_B128`): deep AFTER remap; ≤8 packs OK but **perf-neutral**;
  pack 9+ fails (phase0+phase1 soft-copy). Default **off**. Not fixed by SLOAD addr CSE keying.
- SLOAD scratch addr CSE now keys by `(base, load id)` — safer across soft-copy phases.
- Decode partial still ~34 vs HIP ~28.6; FMA_MIX/SCORE_BATCH/WAVES/DELAY no win.

## Next leftovers

1. Why pack phase0+1 corrupts (not just TMP CSE)
2. Decode partial HIP gap (fma_mix unsafe; s_delay_alu neutral)
3. eye/GEMM TC_LDS_AB
