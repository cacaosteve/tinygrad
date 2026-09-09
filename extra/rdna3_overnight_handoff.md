# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`20d2fdfd2`** (code tip **`a3d953ae8`**).

## Headline

Flash DIRECT **opt-in**. Serial PERF **~763 µs**; fair often **~790** (noisy after MMU probes):

| | DIRECT (now) | start of night | HIP (prior) |
|--|--:|--:|--:|
| prefill | **~760–790 µs** | ~700–720 / SPILL14 | **best ~328** |
| WMMA | **12** | 6 | **24** |
| private / VGPR | **128 / 206** | 404 / 166 | **0 / 206** |
| SPILL | **0** | **14** | **0** |
| decode | **beats HIP** (~136–145) | ~141 | ~156 |

Gap to HIP still ~**2.3×**. Do **not** flip DIRECT default.

## Landed tonight

1. Sticky **`ALLOW_UPCAST16=0`** (TinyJit-safe) → SPILL 0, HIP-like VGPR/priv
2. Default **`AMD_FLASH_K_UNROLL=2`** → WMMA 6→12 (~100µs)
3. Sticky **`AMD_PACK_SLOAD_B128=1`** → ~5–10µs (bases=1/max=8)

## Dead ends (do not revive)

- K_UNROLL=-1 (HIP QK) / 1 / 8 / 8×qk: **MMU** even with SPILL_DRAIN
- K_UNROLL=-1 ×pv: correct but slower (~887)
- PACK_SLOAD_BASES=2: **wrong** (as documented)
- PACK_SLOAD_MAX 4|12|16: wash vs 8
- ACC_SEP=0, INSTR_WAIT, PV_ACC_DIRECT, SOFT_FUSE=0, deeper remat: no win / worse
- SHALLOW remat: now correct under SPILL0 but slower

## Scratch note

SPILL 0 but **private 128** remains: SOURCE has SCRATCH_SIZE/ADDR (no FILL/SPILL). Likely REG-scratch plumbing / ACC work — not free spills.

## Next

1. Soak on tip; **avoid** K_UNROLL=-1/1/8 probes (MMU tax).
2. Structural path to WMMA 24 without HIP-style full-K MMU.
3. Cut remaining private 128 or SLOAD pressure (148 SLOADs in ISA).
4. Fork-only.
