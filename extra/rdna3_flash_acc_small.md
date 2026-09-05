# Flash ACC_SMALL (DIRECT ISA)

**Status (7900, tip `b85508fcf`):** `AMD_FLASH_DIRECT=1` defaults ACC_SMALL on
(python-unroll QK+PV WMMA; `flash_attention` realizes under `AMD_FLASH_ACC_SMALL=1` briefly).

| Path | median µs | err | notes |
|------|-----------|-----|-------|
| DIRECT + ACC_SMALL | ~700–710 | ~1e-4 | 189 VGPR, priv 212, 21 SPILL, 128 SLOAD/SSTORE |
| DIRECT + ACC_SMALL=0 | ~1315 | ~1e-4 | 149 VGPR, priv 384, 16 SPILL, 192 SLOAD/SSTORE |
| HIP flash | ~278 | ~1e-4 | 0 scratch path |
| SDPA (no DIRECT) | ~360 | ~1e-6 | still default / faster than DIRECT |

## Scratch attribution (ACC_SMALL on, prefill 32)

| Bucket | Count | Meaning |
|--------|------:|---------|
| Allocator `SPILL`/`FILL` | 21 / 22 | LinearScan spills under ACC VGPR pressure (ACC from v121) |
| REG soft-copy `SLOAD`/`SSTORE` | 128 / 128 | S/PV ACC↔private REG copies (kept; PV ACC-direct ~810µs worse) |
| Machine `scratch_store`/`load` | 137 / 102 | Private segment is scratch-backed; ≈ soft-copy + spills |
| `private_segment_size` | 212 B | Down from 384 B with ACC_SMALL=0 |

**Takeaway:** ACC_SMALL wins by cutting soft-copy + private size (~192→128 SLOAD, priv 384→212),
not by killing allocator spills (those rise 16→21 because ACC eats VGPRs). Next structural
lever is fewer soft-copies or less ACC pressure — not more env knobs.

## How it works

1. Kernel unrolls `(tm,tn)` / `(tm,td)` so each column has its own WMMA pack tag.
2. `AMD_WMMA_REDEF_ACC` allows PACK/WMMA tag reuse across unrolled tiles.
3. Renderer parks ≤64 ACC buffers only while `AMD_FLASH_ACC_SMALL` is set.
   `flash_attention` realizes under that env when ACC_SMALL is on (default for DIRECT).
   Do **not** key parking off `AMD_FLASH_DIRECT` alone — that broke peer matmul/eye.
4. **Do not** set `AMD_WMMA_ACC_SMALL=1` globally — parks ≤64 quant tiles and breaks Q4/Q6.

## Remaining gap vs HIP

- HIP fully unrolls more WMMA (24 vs 6) and keeps 0 scratch.
- Direct still spills (~21) under ACC VGPR pressure (`WMMA_ACC_VGPR` from v121).
- QK-only unroll regresses (~833µs); keep full QK+PV unroll.
- `AMD_FLASH_K_UNROLL=1` full K chain **MMU-faults**; `=2` nan; `=4` err~1.17 — leave 0.
  Scalarize FILL→v_pack fix remains (harmless for tip).

## Toggles

- `AMD_FLASH_ACC_SMALL=0` — scratch ACC path (~1315µs)
- `AMD_WMMA_ACC_SMALL=1` — force ≤64 park (unsafe for quant)
- `AMD_WMMA_REDEF_ACC=0` — LinearScan assert on unrolled packs
