# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`d3c3744d3`** (docs); code tip **`7e1af310e`** SCORE_BATCH=4.

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~647–676 µs** serial | **~293–324 µs** fair historically |
| decode e2e | **~119–123 µs** (SCORE_BATCH=4) | **~110–111 µs** |
| decode partial (isolated) | **~42 µs** | **~30 µs** |
| decode combine (isolated) | **~9 µs** (ahead of HIP) | **~11 µs** |
| flash VGPR / priv | **206 / 128** prefill; decode partial **121 / 0** | **206 / 0** |

## Landed tonight

1. Prefill sticky **UPCAST16=0** + **PACK_SLOAD** → SPILL 0  
2. **SCORE_BATCH=8** — decode ~134→~121  
3. Prefill **K_UNROLL=1 + MIDSTORE=4** — ~731–738 vs ~768 @factor2  
4. **`AMD_LDS_2ADDR=1`** + **`AMD_LDS_2ADDR_FOLD=1`** + LLOAD pair schedule
5. **Decode-scoped `SKIP_SLOTS=`** (promote slot 2) — ~134→~119µs; prefill keeps SKIP=2 — fold min LDS imm into addr so u8 qword offs fit; **40→64 2addr / 48→0 b64**; ~7µs vs fold=0; serial 13/13
6. **SCORE_BATCH default 8→4** — after slot-2 promote, batch=8 spills (`priv 64`, scratch st/ld, ~139µs); batch=4 stays `priv 0` (~123–125µs); serial 13/13

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
- **Skip park on permlanex16** (`AMD_SWIZZLE_PARK_PERMLANE=0`): MOV 262→230 but **e2e regress** (~126–128 vs ~120–121); isolated partial ~45 vs park baseline. Keep parking offset-16.
- **SCORE_BATCH=8** after slot-2 promote: **priv 64** / scratch — leave at default **4** (6≈4; 12/16 same spill class as 8)
- **AMD_FLASH_WAVES=4**: isolated partial ~39 vs ~41 but **serial FAIL decode_gqa (nan)**; priv 16 / vgpr 194 — keep default 8
- Prefill scratch env: `AMD_SCRATCH_LOAD_B64=1` → **MMU fault** (recoverable); leave off. Odd SCORE_BATCH=3/5 ≈ 4 (wash)
- `AMD_FMA_MIX=1` alone: fair wash; serial can **MMU fault** — leave off (MAX_CAST=128 already FAIL)
- `AMD_COMBINE_UNROLL=64`: fair wash vs default 32; serial OK — keep 32
- `AMD_FLASH_DECODE_LATE_V` (defer V load): SCORE_BATCH=8 still **priv 64** (park temps, not V prefetch) — unmerged
- Prefill `SOFT_SCALE=0` / `ACC_SEP=0`: **serial FAIL**; `SOFT_FUSE=0` slower (~731 vs ~648)
- `AMD_FMA_MIX_EXP=1` + `MAX_CAST=128`: **MMU fault** on serial — leave off
- Env recheck after SCORE_BATCH=4: `ACC_UNROLL=0` **FAIL**; `PEER_SWIZZLE=0` / `PEER_NOPARK` / `SINK_VALU` / `VOPD_FMAC_SCAN` wash; `PV_ACC_DIRECT` decode noise / prefill slower; `REG_PROMOTE=0` decode ~237µs; `BATCH_SWIZZLE_MOV=0` regress; K_UNROLL=1 still best
- Decode partial: **0 VOPD** (HIP ~62) — FMAC dest banks not even/odd; scan=64 no help. LDS_2ADDR no-op on decode (48× `ds_load_b32`). Vec swizzle park ≡ scalar (wash).
- Decode-scoped `FMA_MIX` sticky: **decode_gqa err=inf** — leave off (process-wide FMA_MIX still MMUs prefill)
- WHERE-peel for `AMD_FMA_MIX_EXP` (decode beta): fires **112× fma_mix** but output **wrong** (mean 0 vs ~0.38) / MMU on serial — unmerged. Keep-cast vs skip-cast both fail correctness.
- `AMD_FLASH_UNROLL` 0/1/2/4/5/7: prefill serial wash (~640–662); `AMD_IN_ORDER_EMIT` decode wash
- Selective `AMD_SWIZZLE_NO_PARK_OFFSETS` (1 / 1,2 / 16 / 8,4,2,1): e2e wash vs park-all
- Prefill `SKIP_SLOTS=10,17` / `3,4,10,17` (promote slot2 + skip soft/pv): slower (790–1430 vs ~653); empty SKIP **FAIL** err~0.75

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)
- Default **SCORE_BATCH=4** with decode slot-2 promote (batch=8 spills)

## Next

1. Prefill: priv **128 = TM×TD×4** (slot 2). Promote spills; need fewer scratch round-trips or freed VGPRs before promote.
2. Decode ~10µs vs HIP (~120 vs ~110): HIP partial has vgpr **83** / 0 MOV / 234 `fma_mix` / 162 cndmask / 62 VOPD / 150 delay_alu vs DIRECT vgpr **121** / 262 MOV / 0 mix / 0 VOPD. FMA_MIX paths MMU — need a correct fold, not env wash. VOPD needs bank-aware alloc.
3. Fork-only; soak on tip.
