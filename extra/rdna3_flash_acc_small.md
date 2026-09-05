# Flash ACC_SMALL (DIRECT ISA)

**Status (7900, tip):** `AMD_FLASH_DIRECT=1` defaults `AMD_FLASH_ACC_SMALL=1`
(python-unroll QK+PV WMMA + renderer auto-park ≤64 REG with ≥2 packs).

| Path | median µs | err | notes |
|------|-----------|-----|-------|
| DIRECT + ACC_SMALL | ~670 | ~1e-4 | 6 WMMA, ~189 VGPR, ~21 spills, priv 212 |
| DIRECT + ACC_SMALL=0 | ~1030 | ~1e-4 | scratch ACC (prior tip) |
| HIP flash | ~268 | ~1e-4 | 24 WMMA, 0 scratch, ~206 VGPR |
| SDPA (no DIRECT) | ~306 | ~1e-6 | still default / faster than DIRECT |

## How it works

1. Kernel unrolls `(tm,tn)` / `(tm,td)` so each column has its own WMMA pack tag.
2. `AMD_WMMA_REDEF_ACC` allows PACK/WMMA tag reuse across unrolled tiles.
3. Renderer auto-parks ≤64 buffers with ≥2 reload packs (no process-wide env needed).
4. **Do not** set `AMD_WMMA_ACC_SMALL=1` globally — parks ≤64 quant tiles and breaks Q4/Q6.

## Remaining gap vs HIP

- HIP fully unrolls more WMMA (24 vs 6) and keeps 0 scratch.
- Direct still spills (~21) under ACC VGPR pressure (`WMMA_ACC_VGPR` from v121).
- QK-only unroll regresses (~833µs); keep full QK+PV unroll.

## Toggles

- `AMD_FLASH_ACC_SMALL=0` — scratch ACC path (~1030µs)
- `AMD_WMMA_ACC_SMALL=1` — force ≤64 park (unsafe for quant)
- `AMD_WMMA_REDEF_ACC=0` — LinearScan assert on unrolled packs
