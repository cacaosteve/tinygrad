# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`0bd065a1f`**.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / K-unroll / eviction / scratch
zeroing / perf tuning stay **off**.

## PV split probe (pre-copy vs post-copy)

`pv` dump was after ACC→`pv_soft` (slot 17). Added **`pv_wmma`** dump of slot 10
*before* that copy, plus compile-time slot trace (WMMA / MOV / SPILL / FILL with
phys + scratch offsets).

```
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --phase-only --phases pv_wmma,pv --replays 100
```

### GPU (gfx1100) — frozen instrumented ELF fail/pred

Instrumented out still fails (replay 1). Probe verdict:

| Probe | Result |
|-------|--------|
| `pv_wmma` before copy | **already wrong** |
| post-copy `pv` | same coords + same values as `pv_wmma` |
| Conclusion | **not** REG_STORE / soft spill primary; look at P/V LDS, WMMA inputs, ACC init, WMMA scheduling |

Earliest this run: **`pv_wmma` @ n_tile=1**, `(qrow=16, dcol=64)` →
`wave_m=1, wave_n=1, lane=0, ri=0, rj=0` (still the **wave_n=1 / dcol=64** boundary;
prior `pv`-only run had n_tile=3 / qrow=8 — tile index varies with launch, class does not).

Slot trace for that fragment: ACC pack `v142` lane `v142` → soft `v24`;
`v24` SPILL/FILL scratch_off **156** (secondary once pre-copy is wrong).

Artifacts on gaming PC: `extra/rdna3_state_diag_pv_split/`.

## Next

1. Dig into **pre-WMMA** path for PV: P/V LDS contents, WMMA A/B packing, ACC cin init,
   scheduling around wave_n=1.
2. Do **not** chase slot-2 acc until both PV stages match.
3. Keep DIRECT blocked; no merge / no “fixed” claim.

## Local gates

- `map_pv_coord(8,64)` → wave_m=0,wave_n=1,lane=0,ri=4,rj=0
- `pytest -k 'cluster_sload or batch_sload or in_order_emit'` → 8 + 2 subtests
- Ruff clean on touched files
