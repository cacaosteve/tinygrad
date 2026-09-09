# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`efdc1ed8e`** (code **`f6d38d301`**).

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~757–790 µs** | **~324 µs** |
| decode median | **~121–122 µs** (was ~134) | **~108 µs** |
| WMMA / SPILL / priv / VGPR | **12 / 0 / 128 / 206** | (HIP) |

Prefill ~**2.4×** HIP. Decode ~**1.12×** HIP after SCORE_BATCH. No DIRECT default flip.

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. Default **`AMD_FLASH_K_UNROLL=2`** → WMMA 12  
3. **Decode-scoped envs**: UPCAST16=1 + PACK off  
4. **`AMD_FLASH_SCORE_BATCH=8`** restored (`warp_reduce_many`) — lost in #18010; **~134→121 µs**, serial 13/13

## Dead ends / washes

- CLUSTER_SLOAD_SCAN=128, COMBINE_UNROLL=64, K_TILE_OUTER, SWIZZLE_DELAY/VALU_GAP e2e  
- Scoped K_UNROLL slower; hip_qk **MMU**; slot-2 promote SPILL16/priv256  
- ACC_SHARED: leave off

## Decode asm gap (remaining ~13 µs)

DIRECT: 207 waits, 0 delay_alu, priv 64 · HIP: 146 waits, 120 delay_alu, priv 0  
SWIZZLE_DELAY adds delay_alu but no e2e win under park.

## Next

1. Close remaining decode vs HIP (~108); cut decode priv 64.  
2. Prefill WMMA 12→24 without MMU; cut priv 128.  
3. Fork-only; soak on tip.
