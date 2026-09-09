# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`281bb2e2a`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~715–742 µs** (+LDS_2ADDR when paired) | **~293–324 µs** |
| decode median | **~120–125 µs** (SCORE_BATCH=8) | **~108 µs** |
| flash VGPR / priv | **206 / 128** | **206 / 0** |

## Landed tonight

1. Prefill sticky **UPCAST16=0** + **PACK_SLOAD** → SPILL 0  
2. **SCORE_BATCH=8** — decode ~134→~121  
3. Prefill **K_UNROLL=1 + MIDSTORE=4** — ~731–738 vs ~768 @factor2  
4. **`AMD_LDS_2ADDR=1`** (default) — `ds_load_2addr_b64` when dest VGPRs consecutive; −40 LDS/LSHL when it fires; serial OK

## Asm gap still (prefill)

DIRECT: 556 MOV, 164 scratch_store_b32, 128→~88 ds_load  
HIP: 32 MOV, 0 scratch, 64× `ds_load_2addr_b64`, 119 delay_alu, VOPD

## Dead ends

- Slot-2 promote / ACC_SLOT=19 / ACC_SHARED: spill or wash (priv stays 128)  
- MIDSTORE=8 MMU; SCORE_BATCH=2 slower  

## Next

1. Force consecutive LLOAD VGPRs so 2addr always pairs (→ HIP’s 64).  
2. Pack scratch_store_b32 (164) → b128; cut MOV/addr tax.  
3. Decode priv 64. Fork-only; soak.
