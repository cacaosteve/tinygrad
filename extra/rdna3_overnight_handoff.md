# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`ac577e6a4`** (code **`897c8e806`**).

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~715–738 µs** (midstore K1) | **~293–324 µs** |
| decode median | **~120–125 µs** quiet (SCORE_BATCH=8) | **~108 µs** |
| flash VGPR / priv / SPILL | **206 / 128 / 0** | **206 / 0 / —** |

Same VGPR as HIP; DIRECT still pays **priv 128** (slot-2 `acc`). Prefill ~**2.4×**. Decode ~**1.12×**.

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. **`AMD_FLASH_SCORE_BATCH=8`** restore — decode ~134→~121  
3. Prefill **`K_UNROLL=1` + `MIDSTORE=4`** — serial ~731–738 vs ~768 @factor2

## Dead ends

- MIDSTORE=8 / hip_qk without mid → **MMU**  
- Slot-2 promote / ACC_SLOT=19 → SPILL~115–132 / priv~300–368 / ~1.5ms  
- ACC_SHARED re-probe: no MMU now, but **priv still 128** (wash)  
- SCORE_BATCH=2/1 slower; scratch B64/SCLAUSE/PACK_MAX/CLUSTER/FMA wash  

## Next

1. Prefill: close ~2.4× vs HIP beyond priv (asm: waits/SLOAD/WMMA shape).  
2. Decode priv 64 + ~12µs.  
3. Fork-only; soak on tip.
