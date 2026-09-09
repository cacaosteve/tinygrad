# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`6f2a0ba6c`**.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~708–716 µs** serial | **~293–324 µs** fair historically |
| decode e2e | **~120–121 µs** | **~109–112 µs** |
| decode partial (isolated) | **~41 µs** | **~30 µs** |
| decode combine (isolated) | **~9 µs** (ahead of HIP) | **~11 µs** |
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
- **Prefill promote slot2** (`AMD_FLASH_PREFILL_SKIP_SLOTS=`): vgpr stays 206 but **priv 128→368**, **SPILL 132 / FILL 66** (base had 0 spills) → ~1.5–1.7ms. Soft/ACC_SEP/PROMOTE_MAX cuts do not fix. Slot2 scratch is intentional; promote displaces other live values into worse spill.
- **SCRATCH_STORE_B64**: fair wash (~714µs); only +4 b64 / −8 st32 — leave off
- **SCRATCH_DEST_ADDR=0**: fair **regress** (~725–730 vs ~714); keep default 1 (s_clause path)
- Prefill `lshl` tax (~196): mostly **LDS** addr (`ds_load_2addr` within +3), not DEST_ADDR scratch; HIP has same 64× 2addr
- Swizzle gap fill (`VALU_GAP` / `LOAD_GAP` / skip-scan): **no bubble change** (ops_before still 3.5, pct_global 0) — nothing independent after SW×N,MOV×N in flash_decode; e2e wash. Unmerged.
- **AMD_SWIZZLE_NO_PARK_LT** (selective): wash/slight regress; full NO_PARK still ~131µs e2e
- **AMD_SWIZZLE_REUSE_PARK**: serial OK but **regress** (partial ~47 vs ~41; e2e ~125 vs ~121); vgpr unchanged at 121
- **AMD_LGKM_DELAY** (s_delay_alu before every lgkm flush): e2e wash
- **AMD_FLASH_ACC_LDS**: park slot2 in per-thread LDS — **slower** (~1250), priv **200**, **serial FAIL** prefill — leave off
- Decode schedule toggles: `AMD_SCHEDULE_ALU=0` / `AMD_SCHEDULE_VMEM=0` **regress** (~133–140µs) — keep defaults

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)

## Next

1. Prefill: priv **128 = TM×TD×4** (slot 2 only). Promote spills elsewhere — need VGPR freed before promote, or fewer scratch round-trips without promote. LDS-park for slot2 untried (likely same ACC_WORK roundtrip).
2. Decode ~10µs: HIP wait→ADD / delay_alu / fma_mix; park path still wins vs NO_PARK.
3. Fork-only; soak on tip.
