# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **pending push** (q_lds/k_lds dumps + fail wn0/wn1 compare).

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / K-unroll / eviction / scratch
zeroing / perf tuning stay **off**.

**Do not touch** PV copy, `v_lds`, or slot-2 acc until wave_n=1 QK/`S_reg` is stable.

## Wave_n=1 QK is the first diverge

One probe per frozen ELF (`--phases` single name):

| Probe ELF | Earliest fail vs pred | wave_n=0 | wave_n=1 |
|-----------|----------------------|----------|----------|
| **`qk_wmma`** (after `qk_done`, slot 6) | **DIVERGES** @ n_tile=2 `[wn=1,qrow=16,kcol=8]` | EXACT | **wrong** |
| `s_soft` / `p_reg` | same class (downstream) | EXACT | wrong |

**Within one failing launch:** at n_tile=2, **fail wn0 ≠ wn1** while pred wn0 == wn1.
Same `wave_m` waves must compute identical S (QK does not index `wave_n`; shared Q/K LDS).
Same-launch disagreement ⇒ **wave-local** nondeterminism (VGPR/scratch/spill), not bad tile data.

Coord map: `wave_m=1, wave_n=1, lane=8, ri=0, rj=0` (ACC pack `v126`).

## IR / emit notes (DIRECT `AMD:AMD`)

- ACC cin zeroed `v126`–`v141` ← `v5` (const 0) after qk barrier; **no SPILL of ACC packs**.
- Emit has `s_waitcnt_lgkmcnt` before each QK WMMA (LDS→WMMA waits present).
- **K-loop address base `v28`:** spilled once to scratch+128; each K iter `ds_load` into
  `v[27:28]` clobbers it; `scratch_load` restores at iter end. Hypothesis: wave-local
  spill/reload of address bases (not ACC). New tracer lists `qk_window_spill_fill`.

## New dumps

- `q_lds` / `k_lds` — shared LDS after `qk_load_barrier` (expect EXACT). Rule out tile data.
- Harness now prints **fail** wn0 vs wn1 (not only pred).

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases q_lds --replays 100
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases k_lds --replays 100
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases qk_wmma --replays 100
```

## Next

1. Confirm `q_lds`/`k_lds` EXACT on fail vs pred.
2. If LDS exact: attack K-loop `v28` spill (pin / avoid LLOAD into address VGPR / extra vscnt).
3. Keep PV / slot-2 / perf blocked.

## Local gates

- `map_s_coord(1,16,8)` → lane=8, ri=0, rj=0
- Ruff clean; `pytest test/amd/test_amd_renderer.py -k 'cluster_sload or batch_sload or in_order_emit'` → 8 + 2 subtests
