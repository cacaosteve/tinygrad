# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`892bdf1b3`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~760–790 µs** | **~324 µs** |
| decode median | **~130–147 µs** (scoped envs) | **~107 µs** |
| WMMA / SPILL / priv / VGPR | **12 / 0 / 128 / 206** | (HIP) |

Prefill ~**2.4×** HIP. Decode ~**1.3×** HIP. No DIRECT default flip.

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. Default **`AMD_FLASH_K_UNROLL=2`** → WMMA 12  
3. **Decode-scoped envs**: `ALLOW_UPCAST16=1` + PACK off (~134–147 vs ~153 when inheriting prefill stickies)

## Dead ends

- K_UNROLL=-1/1/8 MMU; WAVES=4 wrong; WAVES=32 hang  
- Slot-2 promote: spill/priv regress  
- Shared prefill stickies on decode: ~20µs decode tax  

## Next

1. Soak; avoid MMU K_UNROLL probes.  
2. Prefill WMMA 12→24; decode → HIP ~107.  
3. Fork-only.
