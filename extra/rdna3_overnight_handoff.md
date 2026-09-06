# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: refresh after next commit.

## Headline

**Flash DIRECT ACC_SMALL** ~**685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed

- Safe SLOAD **clustering** (store/CF) + tests
- Safe SLOAD **batching**: refuse SSTORE/STORE/SPILL as USE (hoist-past-write); composed `_schedule_scratch_load_passes` + tests
- Promote A/B fails on SDPA maxdiff > tol
- Pack: second-base trigger identified; `PACK_SLOAD_BASES=1`; pack stays **off** (loses ~10µs)

## Promote / larger-S (not “noise”)

Fresh-process A/B is **not** stable: S=128 can be exact one run and fail the next (prom≠skip2 and/or skip2≠SDPA beyond tol). Treat as a real multi-tile / path divergence bug to isolate — not FP tolerance.

## Defaults / keep off

- **FMA_MIX**, **PACK_SLOAD_B128**, second-base packing, eviction experiments

## Next leftovers

1. Isolate larger-S / cross-run promote vs SKIP=2 vs SDPA failures (bit-exact dumps)
2. Decode partial 34→28 + e2e
3. eye/GEMM TC_LDS_AB
