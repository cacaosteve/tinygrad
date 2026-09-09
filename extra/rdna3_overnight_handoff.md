# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`657680b41`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~711–720 µs** (64× `ds_load_2addr_b64`) | **~293–324 µs** |
| decode median | **~118–121 µs** (decode promote slot2) | **~108–110 µs** |
| flash VGPR / priv | **206 / 128** | **206 / 0** |

## Landed tonight

1. Prefill sticky **UPCAST16=0** + **PACK_SLOAD** → SPILL 0  
2. **SCORE_BATCH=8** — decode ~134→~121  
3. Prefill **K_UNROLL=1 + MIDSTORE=4** — ~731–738 vs ~768 @factor2  
4. **`AMD_LDS_2ADDR=1`** + **`AMD_LDS_2ADDR_FOLD=1`** + LLOAD pair schedule
5. **Decode-scoped `SKIP_SLOTS=`** (promote slot 2) — ~134→~119µs; prefill keeps SKIP=2 — fold min LDS imm into addr so u8 qword offs fit; **40→64 2addr / 48→0 b64**; ~7µs vs fold=0; serial 13/13

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
- **MIDSTORE=2/3**: slower; **MIDSTORE=2 fails** serial (11/13); keep default 4
- **AMD_D16_HI=1**: wash/regress decode; serial HW wait timeouts — leave off
- **SCRATCH_STORE_B64**: ~164→156 st32, timing wash; b128 fusion still 0
- **scratch_store b128**: 22× contiguous-16B groups exist but data is often the **same SSA MOV** (regs all vN) — not 4 distinct VGPRs; only ~3 fusible
- **AMD_WHERE_ALIAS=1**: decode wash (~112–121→~121); prefill wash; **serial FAIL 8/13** (flash + eye) — leave off
- **AMD_FMA_MIX / MAX_CAST=128 / ALL**: fires ~126 `v_fma_mix` on partial but **MOV stays ~264**; timing wash/regress; **MAX_CAST=128 serial FAIL decode_gqa** — leave off (HIP mix comes with different scheduling)
- Decode leftover env wash: `SWIZZLE_NO_PARK`, `BATCH_SWIZZLE_MOV=0`, `VOPD_FMAC_SCAN=32`, `LOAD_GAP_FILL=0`, `SINK_VALU_SWIZZLE`, `SWIZZLE_VALU_GAP`, `SWIZZLE_DELAY`; SCORE_BATCH 4/16/32 ≈ default 8

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)

## Next

1. Decode ~119 vs HIP ~109 (~10µs); leftover swizzle/MOV. Prefill still ~2.3× HIP (priv 128).
2. Prefill ~711 serial / ~2.3× HIP: 556 MOV class tax, priv 128; VOPD=0 on prefill; defaults beat ACC_SEP=0 / ACC_SMALL=0 / K_UNROLL=2.
3. Fork-only; soak.
2. Prefill still ~2.3× HIP (556 MOV / priv 128); scratch b128 mostly same-SSA zeros.
3. Fork-only; soak.  
