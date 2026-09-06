# Flash ACC_SMALL (DIRECT ISA)

**Status (7900, tip `6c0bb21c1`):** `AMD_FLASH_DIRECT=1` defaults ACC_SMALL on
(python-unroll QK+PV WMMA; `flash_attention` realizes under `AMD_FLASH_ACC_SMALL=1` briefly;
tile-local VGPR work copy + float4 copies + normalize fused into store).

| Path | median µs | err | notes |
|------|-----------|-----|-------|
| DIRECT + ACC_SMALL | ~675 | ~1e-4 | 214 VGPR, priv 212, 21 SPILL, 64 SLOAD/SSTORE |
| DIRECT (pre work-copy) | ~702 | ~1e-4 | 189 VGPR, 128 SLOAD/SSTORE |
| DIRECT + ACC_SMALL=0 | ~1315 | ~1e-4 | 149 VGPR, priv 384, 16 SPILL, 192 SLOAD/SSTORE |
| HIP flash | ~278 | ~1e-4 | 0 scratch path |
| SDPA (no DIRECT) | ~360 | ~1e-6 | still default / faster than DIRECT |

## Scratch attribution (ACC_SMALL on, prefill 32)

| Bucket | Count | Meaning |
|--------|------:|---------|
| Slot-2 `acc` SLOAD/SSTORE | 64 / 64 | Tile load+writeback only (normalize fused into global store) |
| Soft/stats REG (slots 16/17/3/…) | 0 scratch | Already VGPR-promoted (`REG_STORE` elided) |
| Allocator `SPILL`/`FILL` | 21 / 22 | LinearScan under ACC VGPR pressure (ACC from pool idx 121 → v126) |
| Machine `scratch_store`/`load` | ~65 / 62 | After float4 work-copy + fused normalize |
| `private_segment_size` | 212 B | |

**Corrected takeaway:** the old “128 soft-copy” count was mostly **slot-2 acc**, not S_soft/pv_soft
(those already promote). Tile-local slot-19 work copy cuts mid-tile scratch while keeping
`alpha*acc` before V load (latency hiding). Fusing `acc=alpha*acc+beta*pv` cut traffic but
**regressed to ~820µs** — lost overlap with V loads; do not revive.

**Loop-carried promote (2026-09-05):** `REG_STORE` is now a tagged two-address redef +
env-keyed `to_program_cache`. Minimal REDUCE-carried promote is correct; **flash still
nans if slot 2 is promoted** (`SKIP_SLOTS=`). Leave slot 2 skipped. See overnight handoff.

## How it works

1. Kernel unrolls `(tm,tn)` / `(tm,td)` so each column has its own WMMA pack tag.
2. `AMD_WMMA_REDEF_ACC` allows PACK/WMMA tag reuse across unrolled tiles.
3. Renderer parks ≤64 ACC buffers only while `AMD_FLASH_ACC_SMALL` is set.
   `flash_attention` realizes under that env when ACC_SMALL is on (default for DIRECT).
   Do **not** key parking off `AMD_FLASH_DIRECT` alone — that broke peer matmul/eye.
4. Tile-local `acc` work copy (slot 19) for ACC_SMALL + unroll correction path.
5. **Do not** set `AMD_WMMA_ACC_SMALL=1` globally — parks ≤64 quant tiles and breaks Q4/Q6.

## Remaining gap vs HIP

- HIP fully unrolls more WMMA (24 vs 6) and keeps 0 scratch.
- Direct still spills (~21) under ACC VGPR pressure; slot-2 still scratch-backed across tiles.
- Promoting slot 2 (`AMD_REG_PROMOTE_SKIP_SLOTS=`) → **nan** — leave skipped.
- QK-only unroll regresses (~833µs); keep full QK+PV unroll.
- `AMD_FLASH_K_UNROLL=1` full K chain **MMU-faults**; `=2` nan; `=4` err~1.17 — leave 0.

## Toggles

- `AMD_FLASH_ACC_SMALL=0` — scratch ACC path (~1315µs)
- `AMD_WMMA_ACC_SMALL=1` — force ≤64 park (unsafe for quant)
- `AMD_WMMA_REDEF_ACC=0` — LinearScan assert on unrolled packs
