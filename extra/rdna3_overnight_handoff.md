# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **pending** — QP_lds Q/P reuse barrier.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`.

## Root cause (candidate fix)

`QP_lds` holds **Q during QK**, then **P reuses the same buffer**. There was **no barrier**
between QK reads and P writes. Faster waves (often wave_n=0) finish soft and overwrite
Q while slower waves are still in the K-loop → wave_n=1-only QK nondeterminism.

**Fix:** `S_reg = S_reg.after(UOp.barrier(S_reg))` immediately before `P_store`.

## Evidence

| Probe | Result |
|-------|--------|
| `q_lds` / `k_lds` alone | EXACT on fail vs pred |
| Same ELF `q_lds,k_lds,qk_wmma` | LDS **EXACT**, `qk_wmma` **diverges** |
| fail wn0 vs wn1 | **DIFF** (same launch; must match) |
| HIP wn0 vs wn1 | always EXACT |
| K-loop `v28@128` spill | **not causal** — `TC_LDS_AB=1` removes spill, bug remains |
| INSTR_WAIT / IN_ORDER / K_UNROLL=2 | still diverge |

## Next

1. GPU: confirm `qk_wmma` + out stable after Q/P barrier (100+ replays).
2. If stable: leave PV/perf blocked until a few clean overnight runs.
3. Do not ship DIRECT as default yet.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases qk_wmma --replays 100
```
