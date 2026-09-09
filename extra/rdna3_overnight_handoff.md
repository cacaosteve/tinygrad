# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`a3d953ae8`**.

## Headline

Flash DIRECT **opt-in**. Fair/serial (7900 XTX):

| | DIRECT (now) | earlier tonight | HIP |
|--|--:|--:|--:|
| prefill | **~760–790 µs** (serial PERF **~763**) | ~700–720 / SPILL14 | **best ~328** |
| WMMA | **12** | 6 | **24** |
| private / VGPR | **128 / 206** | 404 / 166 | **0 / 206** |
| SPILL | **0** | **14** | **0** |
| decode | **~136–145 µs** | ~141 | **~156** |

Decode beats HIP. Prefill still ~**2.3×**. No DIRECT default flip.

## Landed (stable)

- Sticky **`ALLOW_UPCAST16=0`** (TinyJit-safe): SPILL→0, HIP-like VGPR/priv
- Default **`AMD_FLASH_K_UNROLL=2`**: WMMA 6→12; was unsafe under product-16
- Sticky **`AMD_PACK_SLOAD_B128=1`**: ~5–10µs; serial OK; opt out `AMD_FLASH_PACK_SLOAD_B128=0`
- Decode ml_lds + WAVES=8; remat depth-1; soft-fuse / ACC_SEP / ACC_SMALL

## Dead ends (do not revive)

- K_UNROLL=-1 / 1 / 8: **MMU**
- K_UNROLL=4: slower than 2
- ACC_SEP=0: ~2.5ms
- INSTR_WAIT=1: much slower
- PV_ACC_DIRECT=1: slower + spill
- REG_PROMOTE_SKIP empty: tiny best, spill 16 — not default
- Realize-only UPCAST16 pop: TinyJit misses it

## Next

1. Soak; recover after MMU.
2. WMMA 12→24 without MMU (not full/HIP K_UNROLL).
3. Occupancy/emit with SPILL0 headroom.
4. Fork-only.
