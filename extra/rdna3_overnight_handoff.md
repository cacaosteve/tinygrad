# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`79e15511c`**.

## Headline

Flash DIRECT **opt-in**. Fair (7900 XTX; recovering GPU soft):

| | DIRECT (now) | prior tip | HIP |
|--|--:|--:|--:|
| prefill median | **~767–800 µs** (serial PERF **~767**) | ~700–720 / SPILL14 | **best ~328** |
| WMMA | **12** | 6 | **24** |
| private | **~128 B** | ~184–404 | **0** |
| SPILL | **0** | **14** | **0** |
| VGPR | **~206** | ~166 | **~206** |
| decode (WAVES=8) | **~136–145 µs** | ~141 | **~156** |

Decode still beats HIP. Prefill ~**2.3×** HIP best. Do **not** flip DIRECT default.

## Landed (stable)

- Decode ml_lds + **default `AMD_FLASH_WAVES=8`**
- Addr remat depth-1; const MOV remat
- Soft-fuse / ACC_SEP / ACC_SMALL defaults
- **Sticky `ALLOW_UPCAST16=0`** for flash DIRECT (`_flash_direct_compile_env`): SPILL→0, priv→128, VGPR→206. TinyJit-safe sticky. Override `AMD_FLASH_ALLOW_UPCAST16=1`.
- **Default `AMD_FLASH_K_UNROLL=2`** (with ACC_SMALL): WMMA 6→12, ~797 vs ~911 @unroll0; serial 13/13. Was unsafe under product-16 spills — OK after UPCAST16=0.

## Dead ends (do not revive)

- Prior remat/q_idx/ALU dead ends (see history)
- AMD_WMMA_ACC_SHARED: MMU
- Sequential soft-fuse rows: FAIL
- Realize-only UPCAST16 pop: no effect (TinyJit)
- **K_UNROLL=8**: MMU fault
- K_UNROLL=4: slower than 2 (same WMMA 12)
- Scoped HIP_SCOPE qk|pv: worse than all@2
- PACK_F16_GENERAL / WAVES 4|16 / WMMA_DELAY / SOFT_SCALE=0: no win vs sticky baseline

## Next

1. Soak on tip; recover after MMU/HW.
2. Prefill: WMMA 12→24 without MMU (not full K=8).
3. Occupancy / emit / waitcnt with SPILL0+206 VGPR headroom.
4. Fork-only; no DIRECT default flip.
