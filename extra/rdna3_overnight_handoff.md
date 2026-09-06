# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: refresh after next commit.

## Headline

**Flash DIRECT ACC_SMALL** ~**685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## Correctness status (freeze/replay)

`extra/rdna3_flash_kernel_freeze_replay.py` on S=T=128:

| config | unique ELF | verdict |
|--------|------------|---------|
| prom | 1 | **launch nondeterminism** (frozen ELF, varying outs) |
| skip2_nobatch | 1 | **launch nondeterminism** |
| skip2 (defaults) | 1 | **launch nondeterminism** on frozen replay |

Compile is **deterministic** (same sha256). Host q/kv are **not** mutated. Same binary + synchronized launches still diverge — investigate missing waits / uninit scratch/VGPR / overlap. First divergent samples often at query tile ~2 (token≈80), head ~27–30, dim 64.

Stability probe (`extra/rdna3_flash_promote_stability.py`) now **exits non-zero** and **keeps artifacts**.

## Defaults / keep off

- **FMA_MIX**, **PACK_SLOAD_B128**, second-base packing, eviction

## Next leftovers

1. Root-cause **launch** nondeterminism on frozen flash ELF (waits / uninit / overlap)
2. Decode partial 34→28 + e2e (after flash deterministic)
3. eye/GEMM TC_LDS_AB
