# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`134a2ca35`**.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / eviction stay **off**.

## Phase recorder fixes (trustworthy dumps)

Four diagnostic defects in `1420dca2c` are fixed (not explanations of the original bug):

1. **Unique writers** — QK/soft use gated index axes (`kcol.valid(wave_n.eq(0))`,
   `qrow.valid(wave_n.eq(0)&lane_n.eq(0))`); PV/acc include `wave_n` in `dcol`.
2. **Tile dims** — `flash_tile_dims(D) → (8,2,4)` for D=128; harness derives shapes
   from `BLOCK_*` / that helper (no hardcoded `(4,4,4)`).
3. **Inf-aware compare** — matching ±inf equal; NaN mismatches → `kind=nan`
   (no `a-b` NaN false divergence on `[1,-inf]`).
4. **Fail/pred on instrumented ELF** — `--phase-only --phases qk` replays the *same*
   instrumented binary until **out** fails; saves fail+pred dumps. Old uninstrumented
   fail indices are not transferred.

Harness: `extra/rdna3_flash_state_diag.py`

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases qk --replays 100
```

## GPU evidence (gfx1100)

Instrumented DIRECT still **reproduces** out nondeterminism (fail at replay 1).

| phase | focus tile all n_tiles fail_vs_pred |
|-------|-------------------------------------|
| qk | EXACT (all n_tiles; also vs HIP) |
| soft_m / soft_l | EXACT |
| **pv** | EXACT n_tile 0..2; **diverges at n_tile=3** |
| acc | diverges (downstream of pv) |

Earliest dump divergence: **`pv` @ n_tile=3**, logical `(qrow=8, dcol=64)` on the
failing query tile. Artifacts: `extra/rdna3_state_diag_phase_qk`,
`extra/rdna3_state_diag_phase_later` on the gaming PC checkout.

If instrumentation ever suppresses out failure within the replay budget →
**inconclusive**, not a pass.

## Next

1. Trace earliest PV divergence through **def → spill → reload** (last KV tile).
2. Do not ship scratch zeroing; uninstrumented fixed/scratch still nondeterministic.
3. Decode only after DIRECT is bit-exact vs HIP.

## Local gates

- Inf/tile unit checks on harness helpers
- `pytest -k 'cluster_sload or batch_sload or in_order_emit'` → 8 passed + 2 subtests
- Ruff clean on touched files; mypy baseline noise unchanged
