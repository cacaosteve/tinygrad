# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`adf2d1174`**.

## Headline

**Flash DIRECT ACC_SMALL** ~**685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

Decode: partial ~**34** (HIP ~28.6); e2e ~50–75 noisy; combine ~9.4 (beats HIP ~11.5).

## Landed

- Safe SLOAD **clustering** (store/CF)
- Safe SLOAD **batching**: refuse SSTORE/STORE/SPILL as USE; composed schedule tests
- Promote A/B fails on SDPA maxdiff > tol
- Pack second-base trigger; pack/FMA_MIX stay **off**

## Promote / larger-S (classified — not tolerance)

Fresh-process bit-exact dumps at **S=128** (4 runs/config):

| path | cross-run exact? | notes |
|------|------------------|-------|
| SDPA | yes | reference stable |
| SKIP=2 + work-copy (defaults) | yes | shipping path |
| SKIP=2 + `BATCH=0` (cluster on) | **no** | maxΔ~1.5e-3 |
| promote (`SKIP=""`, work=0) | **no** | fails with batch/cluster on **or** off (maxΔ~few e-3) |

So promote vs SKIP=2 / SDPA failures are **path nondeterminism / incorrect promote codegen**, not FP rounding. Shipping SKIP=2 defaults stay bit-stable in this matrix. Do not call these “noise.”

## Defaults / keep off

- **FMA_MIX**, **PACK_SLOAD_B128**, second-base packing, eviction

## Next leftovers

1. Root-cause promote-path cross-run instability (independent of batch/cluster)
2. Decode partial 34→28 + e2e (after promote path trustworthy or explicitly deferred)
3. eye/GEMM TC_LDS_AB

Branch still has pre-existing Ruff/mypy debt (not introduced by these commits).
