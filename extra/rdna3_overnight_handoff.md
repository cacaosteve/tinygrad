# Overnight RDNA3 — tip `6f5a8e053`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**675µs** (err~1e-4). HIP ~278; SDPA ~360.
Keep default **`AMD_REG_PROMOTE_SKIP_SLOTS=2`** + slot-19 work-copy.

## Promote diagnosis (closed for now)

| Config | 1 tile (pn=32) | 2+ tiles | Spills | Notes |
|--------|----------------|----------|--------|-------|
| SKIP=2 + work-copy | OK | OK | ~31 | **shipping baseline** |
| SKIP='' + work-copy | nan | nan | — | clash; work-copy now disabled when 2∉SKIP |
| SKIP='' no work-copy | **match** | **wrong ~20–70** | 0 → 165 | loop-carried under spill |

Root causes found:

1. **Work-copy + promote slot 2 together nan** — fixed: `use_acc_work` only when
   `2 in SKIP_SLOTS`, passed into `@functools.cache` (env alone was sticky-wrong).
2. **Slot-2 promote VGPRs must stay below WMMA ACC** — `_promote_reg_cons` for slot 2.
3. **Multi-tile promote still wrong once LinearScan spills** (0 spill → OK; 165 spill → bad).
   Likely spill-on-evict gap: victim popped from `live` without storing phys. Soft/m_i
   survive (small); 32-wide acc spills. REG_STORE spill-after-def helps only after first spill mark.

Do **not** ship `SKIP_SLOTS=` (also parks slot 2 in peer GEMMs).

### Landed this loop

- `REG_STORE` tagged two-address redef + spill value
- `to_program_mem_key` env in memory program cache
- Slot-2 ACC VGPR exclusion; work-copy gated on SKIP + cache key
- Repros: `extra/rdna3_reg_promote_loop_repro.py`, `extra/rdna3_flash_promote_ab.py`

## Scorecard (7900)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT | ~675 | ~278 |
| SDPA | ~360 | |
| Decode e2e | ~49 | ~46 |
| Decode partial | ~34 | ~29 |

## Do not retry

- Promote slot 2 with work-copy on
- Blind `PROMOTE_VGPR=VGPR[:121]` for soft/slot 19
- Process-wide `SKIP_SLOTS=` as default
- Fuse `acc=alpha*acc+beta*pv` as default
- `AMD_FLASH_K_UNROLL` / PV_ACC_DIRECT / ACC_SEP=0

## Next leftovers

1. **Spill-on-evict** landed gated `AMD_SPILL_ON_EVICT=0` (default off) — `=1` MMU-faults on
   multi-tile promote; needs a safer emit path before enabling. Dedicated VGPR[89:121] pool alone → nan on pn>=64
2. Cut 21 allocator spills / slot-2 tile copies on baseline (~675→HIP)
3. Decode partial 34→29
4. eye/GEMM serial fails exist since before this tip (TC_LDS_AB) — separate track
