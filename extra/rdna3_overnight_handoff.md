# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`4aad8e706`** (+ pending no-helper-kernel fix).

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / eviction stay **off**.

## State diag results (100 replays, S=T=128, tip `4aad8e706`)

HIP fixed: **exact**. DIRECT under all modes still **launch_nondeterministic**.

| mode | exact | first_fail_i | notes |
|------|-------|--------------|-------|
| recreate | no | 3 | prior harness behavior |
| fixed (+ out assign refill) | no | 4 | still diverged |
| fixed + scratch untouched | no | 3 | |
| fixed + scratch zero | no | 4 | zeroing does **not** stabilize |
| fixed + scratch nonzero | no | 4 | pattern fill does **not** stabilize |

First DIRECT out still ~ulp of HIP (`ref_ok` with ~1.8e-7). Scratch init changing
correctness was **not** observed — do **not** ship blanket scratch zeroing.

**Caveat:** fixed mode still used `Tensor.assign` NaN refill between launches (a helper
kernel). Follow-up removes that so fixed = pure same-buffer Flash launches only.

### Phase dumps (single good launch, HIP vs DIRECT)

Focused tile from recreate fail (head=26, qt=3, n_tile=0): QK/soft_m/PV exact;
soft_l/acc ~ulp. Full-buffer dumps show larger diffs on other tiles/waves — next:
dump on **fail vs pred** launches, then trace first wrong value.

Harness: `extra/rdna3_flash_state_diag.py` (`--phase` for dumps).

## Next

1. Re-run fixed/scratch **without** out-refill helper kernels
2. Phase-dump the first failing replay and its predecessor; find first wrong phase
3. Trace that value through def → spill → reload
4. Decode only after DIRECT is bit-exact vs HIP
