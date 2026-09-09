# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`c9054ea37`** (code tip **`a3d953ae8`**).

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~770–790 µs** (serial PERF **~760**) | **~324 µs** (best ~314) |
| decode median | **~133–153 µs** | **~107 µs** |
| WMMA / SPILL / priv / VGPR | **12 / 0 / 128 / 206** | (HIP path) |

Prefill gap **~2.4×**. Decode: HIP ahead (~1.3×); older “DIRECT beats HIP decode” was vs a slower HIP baseline.
Do **not** flip DIRECT default.

## Landed tonight

1. Sticky **`ALLOW_UPCAST16=0`** → SPILL 0, HIP-like VGPR/priv  
2. Default **`AMD_FLASH_K_UNROLL=2`** → WMMA 6→12  
3. Sticky **`AMD_PACK_SLOAD_B128=1`** → ~5–10µs  

## Dead ends (do not revive)

- K_UNROLL=-1/1/8 (and 8×qk): **MMU**; -1×pv slower  
- WAVES=4 decode: faster but **decode_gqa FAIL** (err ~676)  
- WAVES=32 decode: hang/timeout  
- PACK_SLOAD_BASES=2: wrong  
- Slot-2 promote: ~8µs but SPILL16/priv256  
- SCRATCH_STORE_B64, WMMA_DELAY, REG_PROMOTE=0, etc.: wash/worse  

## Scratch note

SPILL 0 but priv **128** remains (SCRATCH_SIZE/ADDR; slot-2 acc path).

## Next

1. Soak; avoid MMU probes (-1/1/8 K_UNROLL, WAVES=32).  
2. Prefill: WMMA 12→24 without MMU; cut SLOAD/MOV pressure (262 MOV).  
3. Decode: close gap to HIP ~107 without WAVES=4.  
4. Fork-only.
