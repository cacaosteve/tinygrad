# Overnight RDNA3 — tip pending push

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. Keep **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## This session

- REG_STORE after spill: MOV+phys+scratch (multi-tile promote **correct**, ~1512µs — do not ship)
- `AMD_SCRATCH_DEST_ADDR` default **1**
- Sort/cluster const SLOAD → more `SCRATCH_LOAD_B128` (2→6); flash **~688→~685µs**
- Promote slow = 79 SPILL vs 21 (VGPR starvation), not correctness
- `AMD_FLASH_WAVES=4` partial faster but e2e **nan**

## Next leftovers

1. More b128 (still 30× B32 scratch loads); cut 21 allocator spills
2. Decode partial 34→28 (HIP 28.7); e2e ~55–58 vs HIP ~53
3. eye/GEMM TC_LDS_AB
