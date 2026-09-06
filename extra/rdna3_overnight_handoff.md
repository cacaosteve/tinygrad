# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **pending push** (rename Q_lds + hip_near).

## Headline

**QK LDS reuse race fixed and validated.** Flash DIRECT still opt-in
(`AMD_FLASH_DIRECT=1`); do not make it the prefill default yet. Packing /
FMA_MIX / K-unroll / perf stay off.

## Fix (`478c23b49`)

1. **Dedicated P LDS** (`slot=5`) — P no longer aliases Q's LDS.
2. **`UOp.barrier(qk_done)` before V_store** — prior `V_lds.after(qk_done)` was
   per-wave only; early waves could overwrite K (KV slot 1) during peer QK.

## Validation (gaming PC, tip `cbd9f1da2`)

| Test | Result |
|------|--------|
| Uninstrumented fixed ×2000 | **exact=True maxdiff=0 ref_ok** |
| Instrumented `qk_wmma` ×200 | **instrumented_stable**; means match HIP |
| Instrumented `p_lds,pv_wmma,pv,acc` ×100 | **instrumented_stable** |

Pre-fix: fail by replay ~1–10; wave_n=1 QK wrong; fail wn0≠wn1; shared Q/K LDS exact.

## Cleanup

- Renamed `QP_lds` → `Q_lds` (P has its own slot).
- Phase stability reports `hip_out_near` / `hip_maxdiff` (atol 1e-5) alongside bit-exact.

## Next

1. Serial prefill correctness soak (`extra/rdna3_serial_correctness.py`).
2. Keep DIRECT opt-in; more soak before flipping defaults.
3. Perf leftovers only after more confidence.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed --replays 1000
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
```
