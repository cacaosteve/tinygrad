# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`478c23b49`**.

## Headline

**QK wave race fixed.** Prefill Flash DIRECT still experimental (not default) until
longer soak + PV path confidence. Packing / FMA_MIX / K-unroll / perf stay off.

## Fix

1. **Dedicated P LDS** (`slot=5`) — stop P from aliasing `QP_lds` / Q.
2. **`UOp.barrier(qk_done)` before V_store** — `V_lds.after(qk_done)` was per-wave
   only; early waves could overwrite K (KV slot 1) while peers were still in QK.

## Evidence

| Test | Result |
|------|--------|
| Pre-fix `qk_wmma` | fail ~replay 1–10; wn1-only; fail wn0≠wn1 |
| Same-ELF `q_lds`/`k_lds` | EXACT while `qk_wmma` diverged |
| Post-fix `qk_wmma` ×300 | stable mean=HIP; no out divergence |
| Uninstrumented recreate×200 | **exact=True maxdiff=0 ref_ok** |
| Uninstrumented fixed×200 | **exact=True maxdiff=0 ref_ok** |

`v28` K-loop spill was a red herring (`TC_LDS_AB` removed it; bug remained).

## Next

1. Longer soak (1k+ fixed launches) if time.
2. Phase-dump `p_lds`/`pv_wmma`/`pv` once — expect match now that QK is stable.
3. Keep DIRECT opt-in; do not enable by default yet.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes recreate,fixed --replays 200
```
