# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`e5e1445ac`**.

## Headline

**Flash DIRECT ACC_SMALL** ~**684–688µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed

- Addr remat CAST/leaf; deep → MMU (off outside TC_LDS)
- SLOAD×4 pack opt-in (≤8 OK/neutral; 9+ corrupts) — default off
- Scratch SLOAD addr CSE by load identity
- **SCRATCH_LOAD_B64** (default on); SSTORE b64 opt-in off
- **Safe SLOAD clustering** (`74f43a366`): stop at same-base SSTORE / SPILL/FILL / CF + regression tests
- Promote A/B fails on SDPA maxdiff > `--tol` (default 2e-4), not only SKIP=2 match

## HW A/B (7900, packing+evict off)

| `AMD_CLUSTER_SLOAD` | flash ACC_SMALL err | median µs |
|---------------------|---------------------|-----------|
| 0 | 1.03e-4 | ~688 |
| 1 (default, fixed) | 1.03e-4 | ~684 |

Clustering is **not** the flash pack9 root cause; keep default on after the store/CF fix.
FMA_MIX stays off. Promote vs SKIP=2 is exact on some lengths (e.g. S=128) but still noisy on larger multi-tile S — gate now fails the run when SDPA/SKIP diffs exceed tol.

## Defaults / keep off

- **FMA_MIX**: off (correct subset ~12% slower; full fold → `inf`)
- **PACK_SLOAD_B128 / eviction**: off until minimized pack failure IR exists

## Next leftovers

1. Pack phase0+1 corruption: minimize failing case + before/after IR (8 OK, 9+ corrupt)
2. Stabilize promote A/B multi-tile exactness (S≥256 flaky vs SKIP=2)
3. Decode partial 34→28
4. eye/GEMM TC_LDS_AB
