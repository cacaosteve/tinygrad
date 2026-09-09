# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`9b60cb7f1`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~711–720 µs** (64× `ds_load_2addr_b64`) | **~293–324 µs** |
| decode median | **~120–125 µs** (SCORE_BATCH=8) | **~108 µs** |
| flash VGPR / priv | **206 / 128** | **206 / 0** |

## Landed tonight

1. Prefill sticky **UPCAST16=0** + **PACK_SLOAD** → SPILL 0  
2. **SCORE_BATCH=8** — decode ~134→~121  
3. Prefill **K_UNROLL=1 + MIDSTORE=4** — ~731–738 vs ~768 @factor2  
4. **`AMD_LDS_2ADDR=1`** + **`AMD_LDS_2ADDR_FOLD=1`** + LLOAD pair schedule — fold min LDS imm into addr so u8 qword offs fit; **40→64 2addr / 48→0 b64**; ~7µs vs fold=0; serial 13/13

## Asm gap still (prefill)

DIRECT: 556 MOV, 164 scratch_store_b32, **64× 2addr** (matches HIP count)  
HIP: 32 MOV, 0 scratch, 64× 2addr, 119 delay_alu, VOPD

## Dead ends

- Slot-2 promote / ACC_SLOT=19 / ACC_SHARED: spill or wash (priv stays 128)  
- MIDSTORE=8 MMU; SCORE_BATCH=2 slower  
- Decode **SWIZZLE_DELAY / VALU_GAP**: wash (~134µs); delay_alu 0→128 but no win  
- LDS_2ADDR miss was **qrange (>2KB offs)**, not consecutive VGPRs (only 1 not_cons)  
- **DECODE_PACK_SLOAD**: wash / regress (decode noisy; prefill ~848 when set)  
- **ACC_WORK=0**: slower (~840 vs ~790), more waits; keep default  
- **SCRATCH_STORE_B64**: ~164→156 st32, timing wash; b128 fusion still 0

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)

## Next

1. Why 164× scratch_store_b32 don’t pair for b128 (VGPR adjacency / schedule).  
2. Decode priv 64 / remaining ~12µs vs HIP.  
3. Prefill still ~2.3× HIP beyond LDS parity. Fork-only; soak.
