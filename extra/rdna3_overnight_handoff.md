# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`677550428`**.

## Headline

**QK LDS reuse race fixed and validated.** Flash DIRECT still opt-in
(`AMD_FLASH_DIRECT=1`); do not make it the prefill default yet. Packing /
FMA_MIX / K-unroll / perf stay off.

## Fix (`478c23b49`)

1. **Dedicated P LDS** (`slot=5`) — P no longer aliases `QP_lds` / Q.
2. **`UOp.barrier(qk_done)` before V_store** — prior `V_lds.after(qk_done)` was
   per-wave only; early waves could overwrite K (KV slot 1) during peer QK.

## Validation (gaming PC)

| Test | Result |
|------|--------|
| Uninstrumented fixed ×1000 | **exact=True maxdiff=0 ref_ok** |
| Uninstrumented recreate ×500 | **exact=True maxdiff=0 ref_ok** |
| Instrumented `qk_wmma` ×300 | **instrumented_stable**; vs HIP ~1e-7 |
| Instrumented `pv_wmma` / `pv` ×100 | no out divergence |

Pre-fix: fail by replay ~1–10; wave_n=1 QK wrong; fail wn0≠wn1; shared Q/K LDS exact.

## Next

1. Keep DIRECT opt-in; real prefill soak before flipping defaults.
2. Perf leftovers only after more confidence.
3. Optional: rename QP_lds → Q_lds now that P has its own slot.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed --replays 1000
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --phase-only --phases qk_wmma --replays 100
```
