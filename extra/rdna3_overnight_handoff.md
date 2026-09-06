# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **(pending)**.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / K-unroll / eviction / scratch
zeroing / perf tuning stay **off**.

## Pre-WMMA localization (strongest so far)

Dumps: `p_reg` (per wave, before LDS), `p_lds`, `v_lds`, `pv_wmma`.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases p_reg,p_lds --replays 100
```

### GPU fail/pred (frozen instrumented ELF)

| Stage | fail vs pred |
|-------|----------------|
| `p_reg` **wave_n=0** | EXACT |
| `p_reg` **wave_n=1** | **DIVERGES** |
| `p_lds` wave_n=0 | EXACT |
| `p_lds` wave_n=1 | diverges (matches p_reg) |
| `v_lds` | EXACT |
| pred: wave_n0 vs wave_n1 | EXACT (both waves agree within a launch) |

**PROBE: `p_reg_wrong`** — softmax **P in REG for wave_n=1** is already nondeterministic
across launches; LDS write is not the first fault. wave_n=0 is bit-stable.

Earliest example: `p_reg` @ n_tile=2, coords `[wave_n=1, qrow=0, kcol=8]`.
Same class as prior dcol=64 / wave_n=1 PV failures (second N-wave).

Prior `soft_m`/`soft_l` dumps only wrote **wave_n=0**, so they could not see this.

Artifacts: `extra/rdna3_state_diag_pre_wmma`, `extra/rdna3_state_diag_preg_waves`.

## Next

1. Trace **wave_n=1** softmax / S_reg path: spills of slot 6/16, LDS QK aliasing with
   wave_n=0, warp-local vs shared scratch.
2. Dump `soft_m`/`soft_l` **per wave_n** (or confirm S_reg before softmax per wave).
3. Do not chase PV copy / slot-2 acc until wave_n=1 `p_reg` is stable.
4. Keep DIRECT blocked.

## Local gates

- Ruff clean on touched files
- `pytest -k 'cluster_sload or batch_sload or in_order_emit'` → 8 + 2 subtests
