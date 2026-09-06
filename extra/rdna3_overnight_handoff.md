# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`29144c446`**.

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
| Uninstrumented recreate ×200 | **exact=True maxdiff=0 ref_ok** |
| Instrumented `qk_wmma` ×300 | stable (HIP mean); no out divergence |
| Instrumented `pv_wmma` / `pv` ×100 | no out divergence (harness “inconclusive”) |

Pre-fix: fail by replay ~1–10; wave_n=1 QK wrong; fail wn0≠wn1; shared Q/K LDS exact.

## Next

1. Phase harness: “no fail in N replays” → stability pass (not FAIL).
2. Keep DIRECT opt-in; more soak / real prefill before flipping defaults.
3. Perf leftovers only after more soak confidence.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed --replays 1000
```
