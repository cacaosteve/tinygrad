# Overnight RDNA3 — tip `eb13b6277`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**673µs** (was ~702 after cleanup; overnight ~669–710).
HIP ~278; SDPA ~360 (still faster default). Serial correctness **13/13**.

## Structural win this stretch

Tile-local VGPR work copy for REDUCE-carried `acc` (slot 2 → promotable slot 19 for the
tile body). Keeps `alpha*acc` overlapping V loads; cuts SLOAD/SSTORE 128→96 and machine
scratch ~137/102 → ~97/74.

**Attribution correction:** soft buffers (16/17) already promote; the old “128 soft-copy”
traffic was mostly unpromotable slot-2 `acc`. Fusing alpha into pv-add cut traffic but
regressed to ~820µs (lost V-load overlap) — reverted.

## Scorecard (7900)

| Workload | AMD | notes |
|----------|-----|-------|
| Flash DIRECT | ~673 | err~1e-4, vgpr 214, priv 212, SPILL 21 |
| SDPA | ~360 | |
| HIP flash | ~278 | |
| Serial | 13/13 | |

## Do not retry

- Park from `AMD_FLASH_DIRECT` alone / ≤64 multi-pack auto-detect
- `AMD_REG_PROMOTE_SKIP_SLOTS=` (promote slot 2) → wrong numerics
- Fuse `acc=alpha*acc+beta*pv` as default → ~820µs regression
- `AMD_FLASH_K_UNROLL` / PV_ACC_DIRECT / ACC_SEP=0

## Next leftovers

1. Cut remaining slot-2 cross-tile scratch / 21 allocator spills (toward HIP ~278)
2. Safer K unroll without MMU fault
3. IQ4 vgpr 118→95
4. Decode partial 34→29
