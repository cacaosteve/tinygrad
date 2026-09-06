# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`74f43a366`**.

## Headline

**Flash DIRECT ACC_SMALL** ~**681µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed this session

- Addr remat CAST/leaf; deep → MMU (off outside TC_LDS)
- SLOAD×4 pack opt-in (≤8 OK/neutral; 9+ corrupts) — default off
- Scratch SLOAD addr CSE by load identity
- **SCRATCH_LOAD_B64** (default on); SSTORE b64 opt-in off
- **Safe SLOAD clustering**: stop at same-base SSTORE / SPILL/FILL / CF; regression tests
- Promote A/B helper fails the run on SDPA maxdiff > tol (not just SKIP=2 match)

## Defaults / keep off

- **FMA_MIX**: keep off (correct subset ~12% slower; full fold → `inf`)
- **PACK_SLOAD_B128 / eviction experiments**: off for correctness baseline
- Packing resume only with minimized failing IR (8 packs OK, 9+ corrupt)

## Next leftovers

1. HW A/B: flash with `AMD_CLUSTER_SLOAD=0` then fixed-on (packing/evict off)
2. Pack phase0+1 corruption (minimized case + before/after IR) — only after cluster trusted
3. Decode partial 34→28
4. eye/GEMM TC_LDS_AB
