# Overnight RDNA3 — tip pending push

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**681–685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. Keep **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## This session

- REG_STORE after spill: MOV+phys+scratch (multi-tile promote **correct**, ~1512µs — do not ship)
- `AMD_SCRATCH_DEST_ADDR` default **1**
- Sort/cluster const SLOAD → more `SCRATCH_LOAD_B128` (2→6); flash **~688→~685µs**
- Promote slow = 79 SPILL vs 21 (VGPR starvation), not correctness
- `AMD_FLASH_WAVES=4` partial faster but e2e **nan**
- **Addr remat:** `_pure_addr` missed `Ops.CAST` (SPECIAL→MOV→CAST→SHL), so remat never fired.
  Leaf remat (`AMD_REMAT_ADDR=1`) cuts ~1 spill, perf-neutral, correct.
  **Deep** remat (`AMD_REMAT_ADDR_DEEP=1`, nested ADD trees) → **MMU** on flash — do not ship.
  Default remat addr still off outside `TC_LDS_AB`.

## Next leftovers

1. Fix deep addr remat MMU (or prefer consecutive VGPRs for more scratch b128 stores)
2. Decode partial 34→28 (HIP 28.7); e2e ~55–58 vs HIP ~53
3. eye/GEMM TC_LDS_AB
