# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: recover soft-fuse (post q_lane revert); prior docs tip `2f5d4ae28`.

## Headline

Flash DIRECT **opt-in**. Fair fresh-process (7900, recovered 2026-09-09):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~685–720 µs** | **~330 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode (WAVES=8) | **~141 µs** | **~156 µs** |

Decode: DIRECT **beats** HIP. Prefill ~**2.1×** behind.

## Landed

- Decode ml_lds + **default `AMD_FLASH_WAVES=8`**
- Addr remat depth-1; const MOV remat (SPILL 16→14)
- Factor K_UNROLL scope; half×16 LSTORE→B128; deep LDS peel

## Spill anatomy (prefill 32×2048)

**7 / 14** SPILL: causal `q_idx` `ADD(ADD(SHL,MOV),C2016..)`. Rest: LDS-stride MUL / FILL / nested addr.

## Dead ends this loop

- SHALLOW remat: SPILL↓ but **FAIL** + slower — leave off
- `NO_STICKY_*` ADD: hang or no win
- Emit toggles (PACK_SLOAD_B128, SCRATCH_STORE_B64, FMA_MIX, LOAD_EXEC, …): no win vs ~720
- Soft_fuse off / ACC_SEP=0 / SOFT_SCALE=0: much slower
- **`AMD_FLASH_QIDX_REG=1`**: correct but **~977 vs ~723** — leave off
- **Shared q_lane `(k-rm)<=q` soft-fuse**: **prefill_gqa FAIL** + slower + SPILL 30 — reverted

- LDS_PAD≠4: pad0 wrong prefill; pad8 wrong; pad16/32 MMU — keep 4
- Slot-2 promote / ACC_WORK=0: slower

## Next

1. Keep soak on tip; recover after MMU with fresh process.
2. Prefill: WMMA 6→24 needs ACC/spill plan (not mask algebra / QIDX REG).
3. Cut MUL/FILL spills; do not revive SHALLOW / q_lane / QIDX_REG without new plan.
4. Fork-only; no DIRECT default flip.
