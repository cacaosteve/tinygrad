# Overnight RDNA3 — tip `6c0bb21c1`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**675µs** (err~1e-4). HIP ~278; SDPA ~360.
Decode e2e ~49 vs HIP ~46; partial ~34 vs HIP ~29. Serial **13/13**.

## This session (loop-carried REG promote)

ChatGPT/KGEN direction: freeze ~672µs baseline; fix REDUCE-carried promote
correctness before more scheduling.

### Landed

1. **`REG_STORE` is a tagged two-address redef** (`rdna3.py`): linear-scan now
   spill-writes after promote updates (was use-only → stale FILL). Matches KGEN
   Mem2Reg “carry variants through the region” at the machine level.
2. **In-memory `to_program_cache` keys on compiler env** (`to_program_mem_key`):
   toggling `AMD_REG_PROMOTE_SKIP_SLOTS` no longer silently returns a stale program.
   (Disk cache already did; memory did not — masked A/B tests.)
3. **Minimal repro** `extra/rdna3_reg_promote_loop_repro.py`: REDUCE-carried
   scalar/vector promote vs CPU — **all OK** after REG_STORE fix.
4. Unit: `test_reg_store_redef_is_two_address`.

### Still broken / do not ship

- **Flash + promote slot 2** (`AMD_REG_PROMOTE_SKIP_SLOTS=`) → **nan**, ~1.0–2.8ms.
  Minimal REDUCE-carried promote is fine; flash+WMMA ACC_SMALL is not.
- Tried excluding WMMA ACC VGPR pool from promoted regs — broke SKIP=2 soft/pv_soft.
  Reverted. Suspected clash remains unproven as the sole cause.
- Keep default **`AMD_REG_PROMOTE_SKIP_SLOTS=2`**.

### Attribution note

Cutting SLOAD/SSTORE 96→64 without a time win does **not** prove scratch dominates;
21 spills / copies / unroll still candidates. Fewer static WMMA ops ≠ less work in loops.

## Scorecard (7900)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT | ~675 | ~278 |
| SDPA | ~360 | |
| Decode e2e | ~49 | ~46 |
| Decode partial | ~34 | ~29 |

## Do not retry

- Park from `AMD_FLASH_DIRECT` alone / ≤64 multi-pack auto-detect
- Promote slot 2 / move acc to slot 19 as REDUCE-carried → nan/wrong + spill thrash
- Blind `PROMOTE_VGPR=VGPR[:121]` for all promote (breaks soft under SKIP=2)
- Fuse `acc=alpha*acc+beta*pv` as default → ~820µs
- `AMD_FLASH_K_UNROLL` / PV_ACC_DIRECT / ACC_SEP=0

## Next leftovers

1. **Flash promote diagnosis (continued):** first bad n_tile through UOps → phys assign
   → spill/FILL with SKIP='' (clear `to_program_cache` when A/B-ing). Compare soft
   (promotes OK) vs slot-2 lifetimes vs WMMA ACC occupancy.
2. Cut 21 allocator spills / remaining 64 slot-2 tile copies (toward HIP ~278)
3. Decode partial 34→29
4. Safer K unroll without MMU fault
