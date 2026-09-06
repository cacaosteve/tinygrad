# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`4b80a18e8`**.

## Headline

**QK LDS reuse race fixed and validated.** Continuous multi-shape + serial soak:
**50+ clean rounds** on gaming PC (~1h+). Flash DIRECT still **opt-in**
(`AMD_FLASH_DIRECT=1`); do not flip the prefill default. Packing / FMA_MIX /
K-unroll / perf stay off. (HIP prefill ~3× faster today.)

## Fix (`478c23b49`)

1. **Dedicated P LDS** (`slot=5`) — P no longer aliases Q's LDS.
2. **`UOp.barrier(qk_done)` before V_store** — prior `V_lds.after(qk_done)` was
   per-wave only; early waves could overwrite K (KV slot 1) during peer QK.

## Validation (gaming PC)

| Test | Result |
|------|--------|
| fixed ×5000 | **exact=True maxdiff=0 ref_ok** |
| continuous soak | multi-shape fixed ×500 (128/128,256/64,512/32,2048/32) + serial DIRECT + recreate ×300 — **50+ rounds clean** |
| all phases ×50 | **instrumented_stable**; hip_near ≈1.8e-7 |
| serial DIRECT+HIP | **13/13** each |

## Cleanup

- Renamed `QP_lds` → `Q_lds`.
- Phase stability: `hip_out_near` / `hip_maxdiff`.
- Serial harness selects `AMD:AMD` vs `AMD:HIP`.
- State diag `--shapes S:T,...` (`b9cfd9a35`).

## Next

1. Keep soak until gaming PC disconnects; then record final round count.
2. Keep DIRECT opt-in; more soak before flipping defaults.
3. Perf leftovers only after more confidence.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed \
  --shapes 128:128,256:64,512:32,2048:32 --replays 500
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
```
