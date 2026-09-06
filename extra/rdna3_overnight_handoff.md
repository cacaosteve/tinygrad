# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`aea9a2041`**.

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
| `s_soft` (after ACC→soft +scale/mask) | diverges (same class; separate ELF) | EXACT | wrong |
| `p_reg` (after exp) | diverges (same class; separate ELF) | EXACT | wrong |

**Authoritative:** `qk_wmma` alone already fails → **Q/K WMMA or ACC extract/init**, not
soft copy / softmax / P LDS. (QK math does not index `wave_n`; both waves share Q/K LDS,
so wave_n=1-only nondeterminism points at **wave-local ACC/VGPR/spill state**.)

Coord map for that hit: `wave_m=1, wave_n=1, lane=8, ri=0, rj=0` (ACC pack `v126` lane).

Legacy `qk` dump still writes only `wave_n=0` — does **not** prove wave 1.

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases qk_wmma --replays 100
```

Artifacts: `extra/rdna3_state_diag_qk_wmma` (also `*_s_soft`, `*_p_reg2`).

## Next

1. Trace **slot 6** QK WMMA C for wave_n=1: ACC cin init, EXTRACT, spills of S packs
   (`v126`/`v134`), physical VGPR overlap with wave_n=0.
2. Optionally dump Q/K LDS once (shared) to rule out tile data — expect exact.
3. Keep PV / slot-2 / perf work blocked.

## Local gates

- `map_s_coord(1,0,8)` → lane=8, ri=0, rj=0
- Ruff clean; `pytest -k 'cluster_sload or batch_sload or in_order_emit'` → 8 + 2 subtests
