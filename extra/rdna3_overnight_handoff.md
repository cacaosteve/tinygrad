# Overnight RDNA3 — tip pending push

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**680–685µs** (err~1e-4) on 7900/`gfx1100`.
HIP ~278; SDPA ~360. Keep **`SKIP_SLOTS=2`** + work-copy. **`SPILL_ON_EVICT=0`**.

## This session

- REG_STORE after spill: MOV+phys+scratch (multi-tile promote **correct**, ~1512µs — do not ship)
- `AMD_SCRATCH_DEST_ADDR` default **1**; SLOAD cluster → more b128 (2→6); ~688→~685µs
- **Addr remat:** `_pure_addr` missed `Ops.CAST`. Leaf remat correct but tiny; **deep** remat → MMU. Off outside TC_LDS.
- **Pack SLOAD×4** (`AMD_PACK_SLOAD_B128`): needs deep AFTER remap; ≤8 packs correct but **perf-neutral**; >8 corrupts. Default **off**.
- FMA_MIX on decode partial: wrong + slower (reconfirmed). SCORE_BATCH 8/16/32 neutral.
- Decode: partial **~34** vs HIP **~28.6**; e2e ~50 vs HIP sum ~47. Combine still wins.

## Next leftovers

1. Why pack>8 corrupts / emit-time b128 with temp consecutive dests
2. Decode partial gap (HIP has fma_mix + s_delay_alu; mix unsafe today)
3. eye/GEMM TC_LDS_AB
