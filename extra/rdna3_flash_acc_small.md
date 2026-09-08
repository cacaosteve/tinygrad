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

## Remaining gap vs HIP (re-measured 2026-09-08)

- Fair fresh-process: DIRECT **~730 µs** vs HIP **~596 µs** (~1.2×), not the older ~2×.
- HIP fully unrolls K → **24 WMMA**, **0** private, **~126** `s_delay_alu`, **~28** `v_mov`.
- DIRECT ACC_SMALL: **6 WMMA**, priv **212**, **0** `s_delay_alu`, **~385** `v_mov`.
- Only 6 ACC packs parked; 21 allocator SPILLs hit low VGPRs (`v23..v52`); slot-2 still scratch.
- Promoting slot 2 (`AMD_REG_PROMOTE_SKIP_SLOTS=`) is **correct now** (was nan) but **slower** — leave skipped.
- `AMD_FLASH_K_UNROLL=2` is **correct** but SPILL 87 / priv 492 → **much slower**.
- Full K chain / QK-only expand **hangs** even with `AMD_INSTR_WAIT=1` — not soft waitcnt alone.
- PV-only chained expand (`K_UNROLL=-1 HIP_SCOPE=pv`) is **correct** but slower (SPILL 65).
- `AMD_WMMA_ACC_BASE=201` moves ACC to `v206..` but does **not** remove the 21 SPILLs.
- Codegen UNROLL of K axes fails (WMMA shrink); n_tile UNROLL is incorrect (carried softmax).

## Toggles

- `AMD_FLASH_ACC_SMALL=0` — scratch ACC path (~1315µs)
- `AMD_WMMA_ACC_SMALL=1` — force ≤64 park (unsafe for quant)
- `AMD_WMMA_REDEF_ACC=0` — LinearScan assert on unrolled packs
