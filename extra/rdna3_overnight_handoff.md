# Overnight RDNA3 (to ~5AM PT) — tip `13daf34d9`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline win this stretch

**Flash DIRECT ACC_SMALL is correct and default-on for DIRECT:** ~**669µs** (was ~1030µs scratch ACC).
Still ~2.5× HIP flash (~266µs). SDPA default remains ~307µs (faster than DIRECT).

How: python-unroll QK+PV WMMA columns + `AMD_WMMA_REDEF_ACC` (incl. PACK redef) +
renderer auto-park ≤64 REG with ≥2 packs. **Never** set `AMD_WMMA_ACC_SMALL=1` globally (breaks quant).

See `extra/rdna3_flash_acc_small.md`.

## Scorecard (7900)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT | ~669 | ~266 |
| SDPA (no DIRECT) | ~307 | (HIP uses flash) |
| Decode e2e | ~49.5 | ~54 |
| Decode partial | ~34 | ~29 |
| Q5_K t32 | ~61 | ~61 |
| Q6_K t32 | ~65 | ~64 |
| IQ4_XS t32 | ~65 | ~60 (vgpr 118 vs 95) |

## Tried / leave off

- K_UNROLL=1 MMU fault; =2 nan; =4 wrong — leave 0
- QK-only ACC unroll ~833µs — worse
- PV ACC direct (skip soft) ~810µs — worse; soft copy stays
- REMAT_ADDR=1 — no spill change on flash
- Decode score_batch/delay/gap — flat vs tip

## Also landed
- IQ4 `dequant_halves`: share LUT pairs across halves (vgpr 102 @t16; t32 still 118/~65 vs HIP ~60)

## Also landed
- `AMD_PACK_F16_GENERAL=1`: IQ4 vgpr 118→110; Q6 sometimes ahead of HIP; **breaks FLASH_DIRECT** (nan). Leave opt-in.
- Scalar LSTORE fix for multi-slot phys (enables experimenting with general PACK pool).

## Next leftovers

1. Flash 669→266: cut 21 spills / 128 SLOAD-SSTORE; more correct static WMMA without MMU fault
2. IQ4 vgpr 118→95 (~5µs)
3. Decode partial 34→29
4. SDPA 307→~267 only if safe VOPD (bank pairing)


## Session end

Gaming PC unreachable ~4:48 PT (network down / powered off). Loop stopped.
