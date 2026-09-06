# Overnight RDNA3 — tip `803cb06e0`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**689µs** (err~1e-4) on 7900/`gfx1100` after REG_STORE fix.
HIP ~278; SDPA ~360. Keep default **`AMD_REG_PROMOTE_SKIP_SLOTS=2`** + slot-19 work-copy.
**`AMD_SPILL_ON_EVICT=0`**.

## HW validation (803cb06e0, fresh processes)

| Config | Multi-tile (S=32..256) | Prefill median | Notes |
|--------|------------------------|----------------|-------|
| SKIP=2 + work-copy | vs SDPA maxdiff ~1–2e-4 | **~689µs** | shipping baseline |
| SKIP='' + work=0 (slot-2 promote) | **exact match vs SKIP=2** | **~1512µs** | correct, ~2.2× slower — do not ship |
| Promote loop repro | all OK | — | scalar/wide |

REG_STORE after spill (MOV+SPILL) fixed the stale-phys bug; multi-tile promote is
**correct but not a win**. Leave SKIP=2 default. Peer serial (SKIP=2): 13/13 OK.

## Spill / REG_STORE fixes landed

- Spilled `REG_STORE` → MOV into phys + SPILL to scratch
- Eviction spill records at defining insn (`spill_at`), still default-off
- VGPR scratch page via TMP_VDATA (preserve TMP_VADDR CSE)
- A/B helper now full maxdiff: `extra/rdna3_flash_promote_ab.py`

## Scorecard (7900)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT | ~689 | ~278 |
| SDPA | ~360 | |
| Decode e2e | ~58 | ~54 |
| Decode partial | ~34 | ~29 |

## Do not retry / do not ship

- Process-wide `SKIP_SLOTS=` as default (correct now but ~1512µs; also peer GEMM risk)
- `AMD_SPILL_ON_EVICT=1` until MMU-clean on flash
- Promote slot 2 with work-copy on
- Fuse `acc=alpha*acc+beta*pv` as default
- `AMD_FLASH_K_UNROLL` / PV_ACC_DIRECT / ACC_SEP=0

## Next leftovers

1. Cut baseline spills / slot-2 tile copies (~689→HIP) — promote path correctness unlocked
   diagnosis but not the perf lever yet
2. Decode partial 34→29
3. Optional: why promote is 2.2× slower (spill storm?) if revisiting SKIP=''
4. eye/GEMM TC_LDS_AB — separate track
5. Re-check ~675→689 after TMP_VDATA scratch page (noise vs real)
