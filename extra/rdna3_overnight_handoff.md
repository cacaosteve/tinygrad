# Overnight RDNA3 — tip pending push

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**681µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. Keep **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## This session

- Addr remat CAST/leaf; deep → MMU. Off outside TC_LDS.
- SLOAD×4 pack opt-in; ≤8 OK / neutral; 9+ corrupts (phase0+1). Default off.
- Scratch SLOAD addr CSE keyed by load identity.
- **SCRATCH_LOAD_B64** fusion for SLOAD×2 (default on); serial 13/13; flash ~681µs.

## Next leftovers

1. Pack phase0+1 corruption root cause
2. Decode partial 34→28 (HIP)
3. eye/GEMM TC_LDS_AB
