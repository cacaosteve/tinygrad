# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`897c8e806`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~750 µs** (was ~790 @ K_UNROLL=2) | **~324 µs** |
| decode median | **~121–134 µs** (SCORE_BATCH=8; noisy) | **~108 µs** |
| WMMA / SPILL / priv / VGPR | **24 / 0 / 128 / 206** | (HIP) |

Prefill ~**2.3×** HIP. Decode ~**1.12–1.25×** HIP. No DIRECT default flip.

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. **Decode-scoped envs** + **`AMD_FLASH_SCORE_BATCH=8`** (`warp_reduce_many`)  
3. **`AMD_FLASH_K_UNROLL=-1` + `AMD_FLASH_K_MIDSTORE=4`**: mid-store every 4 WMMA breaks MMU on hip chains → **WMMA 24**, ~790→~750µs, serial 13/13

## Dead ends / washes

- CLUSTER_SLOAD_SCAN=128, COMBINE_UNROLL=64, K_TILE_OUTER, SWIZZLE_DELAY/VALU_GAP e2e  
- Scoped K_UNROLL slower; hip_qk **without midstore** → **MMU**; slot-2 promote SPILL16/priv256  
- ACC_SHARED: leave off; SCORE_BATCH 1/2 slower than 8 on aggregate

## Decode asm gap (remaining)

DIRECT: fewer waits after SCORE_BATCH; still priv 64, 0 delay_alu vs HIP 120 delay_alu / priv 0.

## Next

1. Close remaining decode vs HIP (~108); cut decode priv 64.  
2. Prefill still ~2.3× HIP — cut priv 128 / further schedule.  
3. Fork-only; soak on tip.
