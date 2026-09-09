# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`ac577e6a4`** (code **`897c8e806`**).

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~730–750 µs** (serial ~736) | **~324 µs** |
| decode median | **~121–134 µs** (SCORE_BATCH=8) | **~108 µs** |
| WMMA / SPILL / priv / VGPR | **24 / 0 / 128 / 206** | (HIP) |

Prefill ~**2.3×** HIP. Decode ~**1.12–1.25×** HIP. No DIRECT default flip.

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. **Decode-scoped envs** + **`AMD_FLASH_SCORE_BATCH=8`** (`warp_reduce_many`)  
3. **`AMD_FLASH_K_UNROLL=1` + auto `MIDSTORE=4`**: ACC checkpoint every 4 WMMA → **WMMA 24**, ~790→~735µs, serial 13/13

## Dead ends / washes

- CLUSTER_SLOAD_SCAN=128, COMBINE_UNROLL=64, K_TILE_OUTER, SWIZZLE_DELAY/VALU_GAP e2e  
- DECODE_PACK_SLOAD: wash when quiet; REG_PROMOTE_MAX does not cut decode priv 64  
- MIDSTORE=2 slower; MIDSTORE=8 **MMU**; hip without midstore **MMU**  
- WMMA_DELAY probe flaked MMU (recover+serial OK) — leave off  
- Scoped K_UNROLL slower; slot-2 promote SPILL16/priv256; ACC_SHARED off  
- SCORE_BATCH 1/2 slower than 8 on aggregate

## Decode asm gap (remaining)

DIRECT: waits ~95 after SCORE_BATCH; priv 64, 0 delay_alu · HIP: 146 waits historically, 120 delay_alu, priv 0.

## Next

1. Close remaining decode vs HIP (~108); cut decode priv 64.  
2. Prefill still ~2.3× HIP — cut priv 128 / schedule.  
3. Fork-only; soak on tip.
