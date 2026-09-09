# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`adc25cfe5`** (docs); code tip **`7e1af310e`** SCORE_BATCH=4.

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
- Remat/cluster recheck: `REMAT_ADDR=0` / `DEEP=0` / `CLUSTER_SLOAD=0` decode wash; `REMAT_ADDR_SHALLOW=1` **FAIL** prefill_gqa_32
- `AMD_FLASH_SCORE_HEAD_BATCH` 1/2: serial OK, e2e/partial wash vs all-heads (unmerged)
- `AMD_GATED_VMEM=0`: decode e2e wash. Fair tip: partial **~42** vs HIP **~29.5**; combine **~9.4** vs **~11**
- `AMD_LOAD_EXEC=1`: no change on flash_decode_partial asm (WHERE≠LOAD pattern); e2e wash
- `AMD_HOIST_KERNARG=0` / `PACK_F16_GENERAL` / VOPD disable: decode e2e wash
- Prefill scratch/LDS dest-addr env re-touch (`SCLAUSE`/`LDS_DEST_ADDR`): serial hung/no PERF after baseline — leave defaults; GPU recovered
- Swizzle bubbles @ tip: DIRECT wait→ADD **9** ops (park MOVs) / 0% VALU before wait; HIP wait→ADD **1** / **35.9%** VALU before wait. `SINK_VALU`+MAX 0..16 wash

- `AMD_FLASH_DECODE_PACK_SLOAD=1` after SCORE_BATCH=4: e2e noisy wash (~113–125); serial OK — leave off

- Env batch after SCORE_BATCH=4 (decode e2e wash vs ~123 noisy): GROUPED_REDUCE_UNROLL 1/4, FUSE_KERNARG, MERGE_U32, VCC_CSE=0, B_LSHL_ADD=0, SPILL_DRAIN_LGKM, CLUSTER_SSTORE=0, BATCH_SLOAD_USE=0, PACKED_WMMA_ACC, K_HIP_SCOPE qk/pv, SINK_VMEM_SWIZZLE=0 (fair ≈ base)
- `AMD_REMAT_NO_STICKY_ADD=1`: decode-only can look fast but **serial MMU** — leave off (matches prior hung/MMU note)
- `AMD_FLASH_DECODE_ALLOW_UPCAST16=0`: wash/slight regress

- `AMD_FLASH_K_MIDSTORE=6/7`: serial OK but **regress** (~978 / ~1377 vs ~660 @4) — keep 4
- Scratch sclause toggles (`STORE_SCLAUSE=0` / `SCLAUSE=0` / `SCLAUSE_MAX`): **MMU** after first fault — leave defaults (DEST_ADDR=1)
- `AMD_VOPD_PAIR_ALLOC` sticky even→odd LinearScan affinity: **0 dual_fmac** still; e2e **regress** (~126 vs ~122); serial **FAIL nan + MMU** — unmerged (needs true sibling affinity + mul-src banks, not global sticky)

- `AMD_FLASH_DECODE_PROMOTE_SLOT2=0` @ SCORE_BATCH=4: serial OK but e2e **regress** (~138–146 vs ~120–126) — keep promote
- `AMD_REG_PROMOTE_MAX` retune after tip: MMU once GPU poisoned — leave default (prior 128 wash)

## Confirmed keep

- Default **K_UNROLL=1 + MIDSTORE=4** (WMMA24) beats factor2 (~750 vs ~790; serial 13/13)
- Default **SCORE_BATCH=4** with decode slot-2 promote (batch=8 spills)


- Gaming PC SSH briefly unreachable mid-loop (2026-09-09 ~04:52 PDT); tip docs pushed; resume HW probes when back.
- Gaming PC `69.57.221.45` unreachable since ~04:52 PDT (ping/SSH network unreachable); small leftovers exhausted; next HW probe ready: FMA_MIX_EXP accept WHERE(EXP2,0) without peel.

## Next

1. **HW**: restore SSH to gaming PC, then probe FMA_MIX_EXP + WHERE(EXP2,0) accept (no peel) — local WIP in `rdna3.py` uncommitted.
2. Decode partial ~42→~30: park MOV tax; true sibling VOPD affinity (sticky pair-alloc dead); correct fma_mix.
3. Prefill: priv **128** slot2; MIDSTORE stays 4.
4. Fork-only; tip **`adc25cfe5`**; code tip **`7e1af310e`**.

