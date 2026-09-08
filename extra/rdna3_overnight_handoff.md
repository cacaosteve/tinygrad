# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`954c8b6f7`**.

## Headline

Flash DIRECT **opt-in**. Prefill gap ~**2.4×** vs HIP (fresh processes):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~660–670 µs** | **~280 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode | ~57 µs | ~48 µs |

Soak: continuous on tip (serial+fixed multi-shape).

## Landed

- Decode ml_lds `(WAVES,2,G)` — 16-wave + 32k
- Addr remat depth-1 on flash realize
- Const MOV remat — SPILL 16→14, priv 192→184
- Factor K_UNROLL scope via `AMD_FLASH_K_HIP_SCOPE`
- half×16 LSTORE → two B128 (isel parity with LLOAD)

## Dead ends this loop

- `MUL` in pure-addr remat: wrong @depth1, MMU @depth2
- Power-of-2 LDS stride (132→256): compiles with half×16 store, but
  **SPILL 14→45** — more pressure, not a win
- Full-K / `K_UNROLL=-1` QK: hang; PV-only correct but slower

## Next

1. Keep soak running.
2. Cut remaining 14 nested-ADD addr spills without depth-2 remat.
3. Slot-2 96 SLOAD/SSTORE / 24 WMMA only with spill plan.
4. Do not flip DIRECT default.
