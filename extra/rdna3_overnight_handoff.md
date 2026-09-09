# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`ae467f6e0`** (docs); code tip still **`657680b41`** decode promote slot2.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~708–716 µs** serial (64× `ds_load_2addr_b64`) | **~293–324 µs** fair historically; noisy runs vary |
| decode median | **~118–121 µs** (decode promote slot2) | **~108–111 µs** |
| flash VGPR / priv | **206 / 128** | **206 / 0** |

## Landed tonight

1. Prefill sticky **UPCAST16=0** + **PACK_SLOAD** → SPILL 0  
2. **SCORE_BATCH=8** — decode ~134→~121  
3. Prefill **K_UNROLL=1 + MIDSTORE=4** — ~731–738 vs ~768 @factor2  
4. **`AMD_LDS_2ADDR=1`** + **`AMD_LDS_2ADDR_FOLD=1`** + LLOAD pair schedule
5. **Decode-scoped `SKIP_SLOTS=`** (promote slot 2) — ~134→~119µs; prefill keeps SKIP=2 — fold min LDS imm into addr so u8 qword offs fit; **40→64 2addr / 48→0 b64**; ~7µs vs fold=0; serial 13/13

## Asm gap (prefill)

DIRECT: **556 MOV**, **164 scratch_store_b32**, 0 delay_alu, ~1 VOPD, priv **128**, ninst ~2226  
HIP: **32 MOV**, **0 scratch**, **119 delay_alu**, ~103 VOPD, priv **0**, ninst ~1483  
(LDS 2addr both 64; WMMA both 24)

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
- **AMD_WHERE_ALIAS=1**: decode wash; prefill wash; **serial FAIL 8/13** — leave off
- **AMD_FMA_MIX / MAX_CAST=128 / ALL**: ~126 `v_fma_mix` on partial but MOV ~264; wash/regress; **MAX_CAST=128 serial FAIL decode_gqa**
- Decode leftover env wash: `SWIZZLE_NO_PARK`, `BATCH_SWIZZLE_MOV=0`, `VOPD_FMAC_SCAN=32`, `LOAD_GAP_FILL=0`, `SINK_VALU_SWIZZLE`, `SWIZZLE_VALU_GAP`, `SWIZZLE_DELAY`; SCORE_BATCH 4/16/32 ≈ 8
- Prefill env wash/regress: `SOFT_SCALE=0`, `SOFT_FUSE=0`, `VEC_COPY`, `PV_ACC_DIRECT`, `K_UNROLL=-1`, `MIDSTORE=5`, `REG_PROMOTE_MAX=128`; WAVES 2/4/16 worse/wash
- **Emit-time VOPD MOV pair** (~21 dual / −16 MOV): serial OK, **fair wash** (~714µs); most adjacent MOVs are **broadcast** (same src→many dst) — not VOPD-bankable. Unmerged.

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)

## Next

1. Prefill **priv 128 / 556 MOV** vs HIP 0 scratch / 32 MOV — need structural (not more env washes). Ideas: cut broadcast-MOV tax at source; shrink REDUCE-carried scratch traffic without promoting slot 2 into the 1.5ms cliff.
2. Decode leftover ~10µs: HIP wait→ADD / delay_alu / fma_mix scheduling (park path still wins e2e vs NO_PARK).
3. Fork-only; soak on tip.
