# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`d411cd4ed`**.

## Headline

**Flash DIRECT ACC_SMALL** ~**681µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed this session

- Addr remat CAST/leaf; deep → MMU (off outside TC_LDS)
- SLOAD×4 pack opt-in (≤8 OK/neutral; 9+ corrupts) — default off
- Scratch SLOAD addr CSE by load identity
- **SCRATCH_LOAD_B64** (default on); SSTORE b64 opt-in off

## Next leftovers

1. Pack phase0+1 corruption (not TMP CSE / not EXTRACT prefer)
2. Decode partial 34→28
3. eye/GEMM TC_LDS_AB
