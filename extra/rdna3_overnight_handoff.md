# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`7f3eabbb6`**.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / eviction stay **off**.

## State diag (100 replays, S=T=128)

HIP fixed: **exact**. DIRECT still **launch_nondeterministic** in every arrangement.

| mode | helper kernels between launches | scratch fill | exact | first_fail |
|------|----------------------------------|--------------|-------|------------|
| recreate | new q/kv/out each time | — | no | i=3 |
| fixed | **none** (no out assign) | — | no | i=19 |
| fixed | none | untouched | no | i=2 |
| fixed | none | **zero** | no | i=19 |
| fixed | none | **nonzero 0xA5** | no | i=3 |

First DIRECT out still ~ulp of HIP. **Scratch init does not change correctness** —
do **not** ship blanket scratch zeroing. Fixed same-buffer launches still diverge,
so this is not explained by host buffer recreate alone.

Fail/pred pairs saved under `extra/rdna3_state_diag*`. Phase dumps exist
(`--phase`); next is dump **fail vs pred** and find the first wrong phase value.

Harness: `extra/rdna3_flash_state_diag.py`

## Next

1. Phase-dump first failing replay + predecessor; locate first incorrect phase
2. Trace that value through def → spill → reload
3. Decode only after DIRECT is bit-exact vs HIP
