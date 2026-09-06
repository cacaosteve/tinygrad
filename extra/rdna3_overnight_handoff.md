# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`0951ad79d`**.

## Headline

**QK LDS reuse race fixed and validated** — including real prefill/decode shapes
via `rdna3_serial_correctness`. Flash DIRECT still **opt-in**
(`AMD_FLASH_DIRECT=1`); do not make it the prefill default yet. Packing /
FMA_MIX / K-unroll / perf stay off. (HIP prefill ~3× faster today.)

## Fix (`478c23b49`)

1. **Dedicated P LDS** (`slot=5`) — P no longer aliases Q's LDS.
2. **`UOp.barrier(qk_done)` before V_store** — prior `V_lds.after(qk_done)` was
   per-wave only; early waves could overwrite K (KV slot 1) during peer QK.

## Validation (gaming PC)

| Test | Result |
|------|--------|
| fixed ×5000 | **exact=True maxdiff=0 ref_ok** |
| recreate ×1000 | **exact=True maxdiff=0 ref_ok** |
| all phases ×50 | **instrumented_stable**; hip_near maxdiff≈1.8e-7 |
| serial DIRECT 13/13 | prefill_gqa_32/tiles/mha + decode OK (~1e-4 vs ref) |
| serial HIP 13/13 | same flash errs; prefill median ~302µs vs DIRECT ~938µs |

## Cleanup

- Renamed `QP_lds` → `Q_lds` (`ec88ab5b0`).
- Phase stability: `hip_out_near` / `hip_maxdiff` (atol 1e-5).
- `rdna3_serial_correctness.py` selects `AMD:AMD` vs `AMD:HIP` from `AMD_FLASH_DIRECT`.

## Next

1. Multi-shape state-diag soaks (`--S/--T` variants).
2. Keep DIRECT opt-in; more soak before flipping defaults.
3. Perf leftovers only after more confidence.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed --replays 1000
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
```
