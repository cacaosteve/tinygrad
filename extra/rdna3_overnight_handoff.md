# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`49bbc502f`**.

## Headline

**QK LDS reuse race fixed and validated.** Flash DIRECT still opt-in
(`AMD_FLASH_DIRECT=1`); do not make it the prefill default yet. Packing /
FMA_MIX / K-unroll / perf stay off.

## Fix (`478c23b49`)

1. **Dedicated P LDS** (`slot=5`) — P no longer aliases Q's LDS.
2. **`UOp.barrier(qk_done)` before V_store** — prior `V_lds.after(qk_done)` was
   per-wave only; early waves could overwrite K (KV slot 1) during peer QK.

## Validation (gaming PC)

| Test | Tip | Result |
|------|-----|--------|
| fixed ×3000 | `d16a29e77` | **exact=True maxdiff=0 ref_ok** |
| recreate ×1000 | `d16a29e77` | **exact=True maxdiff=0 ref_ok** |
| `qk_wmma` ×100 | `d16a29e77` | **instrumented_stable**; hip_near=True maxdiff≈1.8e-7 |

## Cleanup

- Renamed `QP_lds` → `Q_lds` (`ec88ab5b0`).
- Phase stability reports `hip_out_near` / `hip_maxdiff` (atol 1e-5).
- `rdna3_serial_correctness.py` selects `AMD:AMD` vs `AMD:HIP` from `AMD_FLASH_DIRECT`.

## Next

1. Run serial prefill/decode correctness under DIRECT + HIP.
2. Keep DIRECT opt-in; more soak before flipping defaults.
3. Perf leftovers only after more confidence.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed --replays 1000
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
```
