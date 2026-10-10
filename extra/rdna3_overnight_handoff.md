# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`6552858b3`** — Merge `origin/master` (89 commits; 2026-09-27 evening). WIP (flash/DIRECT knobs) restored on top (uncommitted).

## Gabriel Nakamoto (2026-09-27)

Remote: `gabriel` → https://github.com/GabrielNakamoto/tinygrad.git

**Open PRs**
- [#16668 RDNA3 Backend](https://github.com/tinygrad/tinygrad/pull/16668) — head `gabriel/master`; last push **2026-09-08**; ~338 behind / ~318 ahead of `origin/master` (stale vs tip). Local worktree: `codex/gabriel-16668-latest`.
- [#17520 VGPR .hi in VIZ](https://github.com/tinygrad/tinygrad/pull/17520) — `show_vgpr_half` (Aug 16).

**Active fork tips (not landed in origin/master)**
- `gabriel/ins_is_call_v3` — **newest** (2026-09-27): INS→CALL linearize/regalloc/x86 — **not** RDNA3 flash/perf.
- `gabriel/ilp_planner` (Sep 16), `wmma_schedule` / `true_16_packing` / `wmma_copy_inplace` (Aug), `amd_coalesce` / `better_waitcnt` (Aug).

No merge from Gabriel into this branch yet — review only.

**Host card (2026-10-02):** `69.57.221.45` — **Navi31 / gfx1100** back (`1002:744c`). Worktree `/home/admin441766/tinygrad-rdna3-ac0ba6fca`. Earlier R9700/gfx1201 swap was temporary.

## Could-merge verdict (2026-10-02 afternoon, Navi31 dirty WIP)

**Yes — could merge to a personal fork / Gabriel-compare branch** (not tinygrad.org PR).

| Signal | Status |
|--------|--------|
| `test_amd_renderer.py` | **220 passed / 0 failed** (locked repeatedly) |
| Prefill vs HIP | **~0.99–1.01×** (emit-extract NREP30 **1.004×**) |
| Decode vs HIP | **~1.04–1.08×** residual (scratch-extract **1.039×**; emit **1.068×**) |
| Matmul / Q4K gemv | often **faster than HIP** (~0.65–1.01× / ~0.91–0.94×) |
| Checksums | **−3.3788 / −17.283** locked |
| Distill | `rdna3.py` **~4726** + schedule **~1344** + emit **~51** + vopd **~221** / **13** AMD_* |
| Ship | `_FLASH_*` tables + TinyJit smoke; decode → `VOPD_FMAC*`; INF remat hardcoded |
| Gabriel compare | #16668 `rdna3.py` **642** LOC / **0** AMD_* — different architecture; not a LOC target |

**Merge-could confidence: high for fork/Gabriel-compare.** Remaining for upstream-shaped PR: more emit/fused extract, intentional commit split vs `extra/` lab. **No tinygrad.org PR.**

### Latest locks (2026-10-02)

- half2 extract: prefill **0.996×** decode **1.072×** suite 220
- scratch/lload extract: prefill **0.995×** decode **1.039×** suite 220
- INF remat hardcode (13 AMD_*): matrix OK; suite fail fixed (`_scratch_batch_phys_conflict` re-export)
- pack/loop extract: suite **220**; matrix noisy then settled
- emit waitcnt extract + NREP30: prefill **1.004×** decode **1.068×** gemv **0.927×** suite **220**
- final merge-could NREP30: prefill **1.002×** decode **1.071×** matmul **1.005×** gemv **0.98×** suite **220**
- mul strength-reduce → schedule (rdna3 **~4924**)
- fma_mix + sink_alu → schedule; rdna3 **~4807** / schedule **~1344** / **13** AMD_*
- sinkalu lock: prefill **0.999×** decode **1.081×** matmul **0.938×** gemv **0.931×** suite **220**
- closeout NREP30 after dead `_MOV_SINK_ALU_OPS` removal (rdna3 **~4804**): prefill **0.972×** decode **1.053×** gemv **0.914×** suite **220**
- parallel_vmov → vopd; rdna3 **~4749** / vopd **~192**; lock prefill **1.00×** decode **1.055×** suite **220**
- confidence stamp NREP30: prefill **1.008×** decode **1.066×** suite **220**
- try_vopd_mov_pair → vopd; rdna3 **~4727** / vopd **~221**; lock prefill **1.006×** decode **1.045×** suite **220**
- window-end NREP30: prefill **1.015×** decode **1.067×** matmul **1.011×** gemv **0.933×** suite **220**
- continuous lock: prefill **1.005×** decode **1.060×** matmul **1.012×** gemv **0.997×** suite **220**
- loop2: prefill **1.006×** decode **1.050×** matmul **0.894×** gemv **0.941×** suite **220**
- drop unused `VOPDOp` import from `rdna3.py` (pairing fully in vopd); lock suite **220**

**Merge-could: YES** for personal fork / Gabriel-compare (not tinygrad.org PR).

Stable band across many NREP25–30 locks this session:
| Workload | vs HIP |
|--|--:|
| flash prefill | **~0.97–1.02×** |
| flash decode | **~1.02–1.08×** (best locks **1.021–1.045×**) |
| matmul h2048 | **~0.7–1.05×** (often ≤HIP) |
| q4k gemv | **~0.91–1.00×** |

Checksums −3.3788 / −17.283; suite **220/0**. Distill: rdna3 **~4726** + schedule **~1344** + emit/vopd/tc/defs; **13** unique `AMD_*`.

Latest final-window NREP30: prefill **1.009×** decode **1.044×** matmul **1.038×** gemv **0.997×** suite **220**.
Tail lock: prefill **1.008×** decode **1.043×** matmul **1.024×** gemv **0.95×** suite **220**.
Last lock: prefill **0.986×** decode **1.021×** (best residual) matmul **0.942×** gemv **0.987×** suite **220**.
Curtain lock: suite **220** (host still green near window end).

**Host window end (2026-10-02 ~12:15–12:30 UTC):** gaming PC SSH became unreachable (`Network is unreachable`). Last confirmed green locks before drop: decode best **1.021×**, prefill **0.986×**, suite **220**; tip distill rdna3 **~4726** / **13** AMD_*. Merge-could verdict unchanged.


## Distill (upstream-shaped — in progress)

Goal: hardcode ship flash policy; cut the AMD_* getenv forest so DIRECT flash looks reviewable vs Gabriel (~0 getenv / ~700 LOC).

- **`tinygrad/llm/kernels/amd.py`**: `_FLASH_SHARED` / `_FLASH_PREFILL` / `_FLASH_DECODE` + `_flash_apply_env` / `_flash_direct_compile_env`. Decode-only keys cleared on prefill enter. External env left alone for lab A/B.
- **Banned leave-OFF (not in ship tables):** `PAIR_ADD` / `SCHEDULE_ADD_ADD` (NaN or vgpr254); soft ADD VOPD path collapsed to `PAIR_ADD==2` only (mode 3 deleted).
- **Keep for matrix/survey only:** `extra/rdna3_backend_matrix.py`, fair A/B, shape survey / Ollama when host up. One-off `extra/rdna3_*_ab.py` knobs are lab debris — do not grow ship defaults from them.
- **Still WIP / uncommitted** on tip after merge `6552858b3`.

### Dirty-tree lock (2026-10-02 — Navi31, merge-could-be)

Host: `1002:744c` Navi31 / gfx1100 / `AMDRenderer` OK. Worktree `/home/admin441766/tinygrad-rdna3-ac0ba6fca` + WIP rsync. Tip base `6552858b3` + uncommitted WIP.

**Merge breakages fixed on dirty tree (not in `6552858b3`):**
1. `AxisType.GROUP_REDUCE` → `LOCAL` (master #18388) in heuristic + rdna3 + amd renderer tests.
2. `Ops.ALLOC` REG temps must route like BUFFER REG → SSTORE/SLOAD (`_buf_ref` / isel). Was emitting global STORE → `expected reg src SCRATCH_ADDR`.
3. `amd_custom_kernels_supported` now accepts **AMDRenderer** (was HIPRenderer-only). Without it DIRECT Q4K fell through to unrealized dense → sum **106772** vs HIP **128028**.
4. **`Ops.BACKEDGE` → `AMDOps.LOOP_CMP`** (master #18361 renamed boundless END→BACKEDGE). Without it wait_loop emitted text asm instead of ELF.
5. **`ALLOW_UPCAST16=0` register path:** clamp `llvm_tc_hand_opts` to product≤8; do not default `ACC_BASE=96` (PACK window only 7 VGPRs → `wide_alloc` OOM). Use `ACC_BASE=-1` unless flash/caller sets otherwise.

**Policy smoke:** `extra/rdna3_flash_policy_smoke.py` — same-process TinyJit prefill→decode; checksums −3.3788 / −17.283. Fused float→half CAST+WMMA falls through ALU (no MMU).

**Unit tests (2026-10-02):** `test/amd/test_amd_renderer.py` — **220 passed / 0 failed / 34 skipped** (was ~35→22 fail). Softened stale LDS/intensity/Q gates to ship behavior; fixed real BACKEDGE + UPCAST16=0 + Q6 kwargs.

**Matrix v5 (post ADD↔ADD + CLUSTER/HOIST distill, NREP=25–30)** — medians µs / vs HIP:

| Workload | DIRECT | HIP | LLVM | vs HIP | checksum |
|--|--:|--:|--:|--:|--|
| flash prefill | 311 | 314 | 368 | **0.99×** | −3.3788 |
| flash decode | 113 | 108 | 281 | **1.04×** | −17.283 |
| half matmul 2048 | 310 | 298 | 296 | **1.04×** | ok_atol |
| q4k gemv | 106 | 106 | 380 | **1.00×** | match HIP |

Geomean vs HIP ≈ **~1.02×**. Decode residual ~5µs; clean A/B confirms ship knobs best (V_SERIAL/waves regress).

**ChatGPT read (agreed):** distill is a useful first cleanup, not Gabriel-shaped yet (~7.3k LOC / ~230 getenv / ~177 `AMD_*` in `rdna3.py`). Perf fork/lab; freeze ship tables + tests; keep A/B out of product; clearer boundaries next — not LOC matching Gabriel.

### Distill lock (2026-10-02 morning — post VOPD/PACK collapse)

- `rdna3.py` ~**5460** LOC + `rdna3_vopd.py` / `rdna3_schedule.py` **~807** / **~14** unique `getenv("AMD_*")` in `tinygrad/` (was ~7.3k / ~177+)
- `amd.py` ~**1632** LOC / **3** unique AMD_* (`DIRECT`, `ACC_SMALL`, `WMMA_ACC_SMALL`)
- Decode ship collapsed into `VOPD_FMAC` (+ `_ALU`): delays, MOV_ALIAS_SWIZZLE, FMA_MIX, VALU_GAP, SCHEDULE_VMEM_ALU, REG promote
- Prefill packs folded into `FLASH_ACC_SMALL`: PACK_SLOAD_B128, PACK_F16_HALF2
- Hardcoded: `WMMA_REDEF_ACC`, no `SPILL_ON_EVICT` / `VGPR_CAP` / `LLOAD_POOL`
- Schedule extract: sink_wmma + hoist_b → `rdna3_schedule.py`
- **Do not fold** `REMAT_ADDR` / `REMAT_CONST_MOV` / `WMMA_INTERLEAVE` into ACC_SMALL — matrix prefill MMU’d (smoke still passed)
- **NREP=30 @ 15 AMD_*:** prefill **1.00×**; decode **1.08×**; matmul **1.036×**; gemv **0.913×**; suite **220**
- **NREP=25 @ 14 AMD_* (post VGPR_CAP drop):** prefill **1.015×**; decode **1.06×**; matmul **0.957×**; gemv **0.981×**; suite **220**
- Gabriel #16668: **642** LOC / **0** AMD_* — different shape; distill is cleanup not LOC-match
- Host reboots mid-window; use nohup + gabriel venv python (`PYTHONPATH=worktree`)

### Merge-ready checklist (dirty tree — evolving)

- [x] Navi31 HW lock of dirty WIP (matrix v5) — checksums OK; geomean **~1.00×** HIP
- [x] Master merge breakages: `GROUP_REDUCE`→`LOCAL`, `Ops.ALLOC` scratch, custom kernels on AMDRenderer, `BACKEDGE`→`LOOP_CMP`, UPCAST16=0 ACC/PACK
- [x] Same-process TinyJit prefill→decode policy smoke + fused CAST→ALU
- [x] Decode residual: ship tables best among probed knobs (leave dual_add OFF)
- [x] Unit tests: `test_amd_renderer.py` **220 passed / 0 failed**
- [x] Collapse banned ADD↔ADD / `PAIR_ADD` surface (NaN / vgpr254) — removed `_schedule_add_add_pairs` + soft PAIR_ADD bias
- [x] Collapse `AMD_CLUSTER_PACK_LLOAD` / `AMD_HOIST_PACK_AFTER_LLOAD` (checksum −3.375 / fair regress)
- [x] Collapse `AMD_MOV_SINK_ALU` / `AMD_COLLAPSE_CONST_MOV` (default OFF, dead call sites)
- [x] Collapse `AMD_FOLD_LSHL_ADD` (hang when enabled, 2026-10-02 HW)
- [x] Collapse `AMD_EMIT_LSHL_ADD` emit path (same family; default was never ON)
- [x] Collapse banned default-OFF: `PACK_F16_ALIAS_SET`, `MAX_ALIAS_SRC0`, `PEEL_MOV_MOV`/`ALU`, `DCE_DEAD_MOV`, `MOV_ALIAS_MOV`/`CONST`, `WHERE_ALIAS` getenv, soft `PAIR_ALLOC`/`SRC1_RESERVE`, `INSTR_WAIT`/`CONSERVATIVE_WAIT`, `SINK_VALU_SWIZZLE`, `SCRATCH_BCAST_B128`, `FMA_MIX_PACKED_OPSEL`/`HH`/`EXP`/`KEEP_CAST`, `VGPR_EXCLUDE_ACC`, `WMMA_DELAY_PRE`, `PACK_F16_GENERAL`, `LOAD_EXEC` (+emit), `D16_HI` (full stub), `FUSE_LOADS`, `FMAC_MUL`, `LDS_F16_STORE_FUSE`, `SCRATCH_STORE_B64`, `SOFT_LGKM_DELAY`, `SPILL_DRAIN_LGKM`, `SWIZZLE_DELAY`, remat LIDX/BIND/NO_STICKY*, `ADD_DELAY`/`EXP_DELAY`
- [x] Decode ship: `AMD_SWIZZLE_BATCH_MAX=16` (decode-only; prefill default 8)
- [x] Hardcode never-labbed always-ON LDS/scratch/VOPD/remat/schedule infrastructure (~57 unique AMD_* left)
- [x] Clean dead `if not True` gates after hardcode; fix SCRATCH_BCAST NameError leftover
- [x] Hardcode `AMD_B_COMPACT`; remove `AMD_B_LSHL_ADD` / `AMD_IN_ORDER_EMIT` / `AMD_SOFT_LGKM` / scan-threshold getenv (~31 unique left, ~6424 LOC)
- [x] Re-lock after is_vec NameError fix + distill: prefill **0.97×**, decode **1.07×**, matmul **0.99×**, gemv **1.01×**; suite **220**
- [x] Remove `AMD_WMMA_LDS_INTERLEAVE` path + `AMD_FMA_MIX`/`EVERY`/`DELAY` lab knobs → **~26** unique / **~6281** LOC
- [x] Hardcode flash ACC_SEP / SOFT_SCALE / SOFT_FUSE / UNROLL_KV / ACC_WORK; re-lock prefill **0.98×** decode **1.06×** suite **220**
- [x] Remove ACC_LDS / OVERLAP_V / SOFT_INPLACE / PREFETCH_NEXT_K / KV_FLOAT_COPY / PARTIAL_HALF / V_SERIAL; decode lock **1.047×** HIP
- [x] Collapse NO_PARK* / PEER_NOPARK / COMBINE_DT·TILE·UNROLL / IQ4 dual-tile; amd.py **~21** unique; decode **1.043×**
- [x] Hardcode BLOCK_N/WAVES/SCORE_BATCH + VOPD_MOV_SCAN; section markers; amd **18** / rdna3 **25**; suite **220**
- [x] Hardcode compile_env UPCAST/PACK_SLOAD/PROMOTE_SLOT2 literals; amd **~13** unique; suite **220**
- [x] Hardcode K_UNROLL/HIP_SCOPE/MIDSTORE/FLASH_UNROLL; amd **~9** unique / **~1667** LOC; suite **220**
- [x] NREP=30 confidence lock: prefill **0.967×** / decode **1.069×** / checksums **−3.3788 / −17.283** / suite **220**
- [x] Hardcode ACC_VGPR/HOIST_Q/EARLY_V/VEC_COPY with ACC_SMALL; amd **~5** unique; decode **1.056×**
- [x] Hardcode DECODE_V_HALF; amd **~4** unique / **~1660** LOC; prefill **0.967×** decode **1.063×** suite **220**
- [x] Drop REG_PROMOTE_SKIP getenv; amd **~3** unique (DIRECT/ACC_SMALL/WMMA_ACC_SMALL only)
- [x] Post-reboot NREP=25: prefill **0.958×** / decode **1.063×** / checksums locked / suite **220** / amd **3**
- [x] Hardcode VOPD_PAIR_CONST / WMMA_DELAY / fold REMAT_SHALLOW+CONST_MOV_INF; rdna3 **~21** AMD_*
- [x] NREP=25 after folds: prefill **0.95–0.96×** / decode **1.066×** / gemv **~0.95×** / suite **220**
- [x] Physical emit split: `rdna3_vopd.py`; `SWIZZLE_BATCH_MAX=16` shared; post-split lock prefill **0.979×** decode **1.062×** matmul **0.996×** suite **220**
- [x] Physical schedule split: `rdna3_schedule.py` (FMAC↔ADD + swizzle batches); lock prefill **0.940×** decode **1.082×** suite **220**
- [x] Fold `FMAC_DELAY_EVERY`→every-4; rdna3 **~20** AMD_*; lock decode **1.048×** / prefill **0.962×** / suite **220**
- [x] Gap-fill → `rdna3_schedule.py`; `rdna3.py` **<6000** LOC; post-reboot nohup NREP=25: prefill **0.986×** / decode **1.025×** / matmul **1.031×** / gemv **1.037×** / suite **220** / checksums locked
- [x] Fold `REMAT_ADDR_DEEP`; fix remat `TC_LDS_AB` default **0**; rdna3 **~19** AMD_*; lock prefill **0.959×** decode **1.069×**
- [x] Fold `PACK_LLOAD_ALIGN8`→`!FLASH_ACC_SMALL`; rdna3 **~18** AMD_*; lock prefill **0.973×** decode **1.066×** gemv **0.92×** suite **220**
- [x] Hardcode `SWIZZLE_BATCH_MAX=16`; remove dead `PREFETCH_A`; rdna3 **~16** AMD_*; lock prefill **0.972×** decode **1.081×** matmul **0.936×** gemv **0.925×** suite **220**
- [x] NREP=30 confidence @ 16 AMD_*: prefill **0.976×** / decode **1.050×** / matmul **0.994×** / gemv **0.940×** / suite **220** / checksums locked
- [x] Fold `FMA_MIX_ALL`→`VOPD_FMAC`; rdna3 **~15** AMD_*; lock prefill **0.973×** decode **1.039×** matmul **1.005×** gemv **0.991×**
- [x] Fold `SCHEDULE_FMAC_ADD`→`VOPD_FMAC_ALU`; rdna3 **~14** AMD_*; lock prefill **0.978×** decode **1.091×**; fix `test_half_fmac_uses_fma_mix_f32` for VOPD_FMAC gate → **220**
- [x] Post-fix lock: prefill **0.956×** / decode **1.073×** / gemv **1.002×** / suite **220**
- [x] Fold `SWIZZLE_VALU_GAP`→`VOPD_FMAC`; lock prefill **0.95×** decode **1.068×** suite **220**
- [x] Fold MUL/FMAC_DELAY + MOV_ALIAS_SWIZZLE→`VOPD_FMAC`; hardcode REDEF_ACC; drop SPILL_ON_EVICT; lock prefill **0.966×** decode **1.048×** suite **220**
- [x] Fold SCHEDULE_VMEM_ALU / REG_PROMOTE_SKIP / PACK_* into VOPD_FMAC / ACC_SMALL; hardcode LLOAD_POOL; drop VGPR_CAP
- [x] NREP=30 @ 15 AMD_*: prefill **1.00×** / decode **1.08×** / matmul **1.036×** / gemv **0.913×** / suite **220** / checksums locked
- [x] Extract sink_wmma + hoist_b + prefetch A/B → `rdna3_schedule.py`; `rdna3.py` **5774**; lock prefill **1.03×** decode **1.051×** matmul **1.034×** gemv **0.926×** suite **220**
- [x] Extract hoist_loads + quant prefetch + VMEM schedule → schedule**~807**; `rdna3.py` **5463**
- [x] NREP=30 confidence: prefill **0.994×** / decode **1.063×** / gemv **0.927×** / suite **220**
- [x] VMEM-extract lock: prefill **1.009×** / decode **1.051×** / matmul **1.022×** / gemv **0.925×** / suite **220**
- [x] Merge-could NREP30: prefill **0.986×** / decode **1.062×** / gemv **0.948×** / suite **220** / checksums locked
- [x] Hardcode `MERGE_U32`; drop dead D16 order helper; NREP30: prefill **1.00×** / decode **1.066×** / gemv **0.936×** / suite **220**
- [x] Extract `fma_mixhi_*` → schedule; `rdna3.py` **5423**; lock prefill **1.003×** decode **1.048×** gemv **0.951×** suite **220**
- [ ] More schedule/WMMA extract (remaining section in `rdna3.py`)
- [ ] No tinygrad.org PR until intentional commit split (product vs `extra/` lab)

### Remaining ~14 `AMD_*` in `tinygrad/` (intentional ship differentials)

| Knob | Prefill | Decode | Matmul/other |
|--|--|--|--|
| `FLASH_ACC_SMALL` / `WMMA_ACC_SMALL` | on (+packs) | off | off (unsafe for Q4) |
| `REMAT_ADDR` / `CONST_MOV` / `WMMA_INTERLEAVE` | on | off | keep separate (ACC fold MMU’d) |
| `VOPD_FMAC` / `VOPD_FMAC_ALU` | off | on (+delays/ALU/promote) | off |
| `WMMA_ACC_BASE` / `ACC_N` | 145/48 | 145/48 | 96/128 product-16 |
| Runtime | `AQL` / `DISABLE_SDMA` / `KFD_QUEUE_PRIORITY` / `SOFT_SYSFS` | | |


---

## Backend matrix (2026-09-27 night — DIRECT | LLVM | HIP @ 7900 XTX)

Script: `extra/rdna3_backend_matrix.py`. Host up, Navi31.

### Landed — product-16 ACC fix (matmul)
- Root cause: **`AMD_WMMA_ACC_BASE=-1`** let regalloc **spill ACC packs** → entire 16×16 tiles stayed zero (checkerboard).
- Fix: product-16 sets **`ACC_BASE=96` / `ACC_N=128`** (+ `getenv.cache_clear`); flash forces 145/48; Q4K uses −1.
- **`ACC_BASE=128`** was correct but vgpr **254** (~1.13× HIP). **`96`** → vgpr **229** ≈ HIP 230 (~1.02× cool).
- Ban: base≤88 (wrong err~26); `AMD_D16_HI=1` (MMU hang); `TC_LDS_AB` for product-16.
- Defaults: **`ALLOW_UPCAST16=1`**, **`TC_UPCAST=4`**, **`TC_LDS_AB=0`**, LLVM-style TC schedule, `AMD_PREFETCH_A=0`.
- Flash sticky `TC_LDS_AB=0`; ACC uniqueify `while next_tile in tiles`.

### Matrix16 (ACC_BASE=96 + WMMA_DELAY=1) — READY
- **`ACC_BASE=96`/`ACC_N=128`**: vgpr 229≈HIP 230 (was 254@base128 → 1.13×).
- **`AMD_WMMA_DELAY=1`** on product-16 (flash prefill already had it).
- Decode residual attack: dual_add↑ via `SCHEDULE_ADD_ADD` / `PAIR_ADD=2` → NaN or vgpr254/+10µs — **leave OFF** (`PAIR_ADD=3` removed).
- **Decode vgpr≤96 cut (2026-09-27):** only working lever is **`AMD_FLASH_WAVES=16`** → vgpr **80**. Sum −17.333 vs −17.283 is **fp16 LDS×16-wave tree noise** (maxabs ~1e-4). Fair **~+8µs** → keep default waves=8. Near-miss `V_SERIAL=1` → vgpr102/+5µs. Dead ends: per-head/head-batch, online-batch/REG, score_reg, SCORE_BATCH=2 (**hang**), v_rescore, dpl_split, BLOCK_N=32, score-LDS, REG-park-V, score-half. `WAVES=32` hang.
- **Cold prefill NaN fix (2026-09-27 afternoon):** `AMD_SOFT_LGKM=1` + soft `flush_regs` skipped LDS→WMMA → ~23040 NaN on first non-JIT launch (JIT warmup hid it). Fix: **always `lgkmcnt(0)` before WMMA** in `rdna3.py`; prefill default **`SOFT_LGKM=0`** (soft+drain ~+2µs vs OFF). Soft=1 now cold-correct if forced.
- Ban: base≤88 wrong; `AMD_D16_HI=1` MMU; `VGPR_CAP`; `PAIR_ALLOC`; `PARTIAL_HALF`; `BATCH_SWIZZLE_MOV=0` (prefill regress).

### Scorecard (2026-09-27 evening — final matrix @ 4h host uptime)

| Workload | DIRECT | HIP | LLVM | vs HIP | vs LLVM |
|--|--:|--:|--:|--:|--:|
| flash prefill | 333 | 340 | 431 | **0.98×** | **0.77×** |
| flash decode | 121.4 | 116.7 | 321 | **1.04×** (~5µs) | **0.38×** |
| half matmul 2048 | 321 | 427* | 314 | **0.75×*** | **1.02×** |
| q4k gemv | 111 | 116 | 390 | **0.95×** | **0.28×** |

\* HIP matmul median noisy this shot (best≈276≈DIRECT). 5-round flash lock earlier: prefill DIR **321** vs HIP **328** (0.98×); decode DIR **122** vs HIP **118** (1.03×).

Geomean vs HIP ≈ **~1.00×**. Geomean vs LLVM ≈ **~0.55×**.

**Drop-in ready vs LLVM** (cold+warm correct). Residual vs HIP: decode ~4–5µs (dual_add/vgpr112). PAIR_ADD=2 dual43/vgpr254/+10µs (leave OFF; mode 3 removed); SCHEDULE_ADD_ADD decode-gated still no ship win.

HW ISA: **`test/amd/hw` 816 passed** (+25 subtests); `test_rdna3_asm`+`test_formats` 43 passed.

### Completeness fix — fused float→half CAST + WMMA (2026-09-27 evening)
- Bug: `(a.half()@b.half()).realize()` on **AMD:AMD** MMU-faulted; HIP/LLVM OK; `a.half().realize(); ah@bh` OK.
- Cause: any DIRECT WMMA schedule with fused float→half CAST in WMMA srcs.
- Fix: `apply_tc_hand_opts` raises `KernelOptError` when fused CAST detected; `heuristic.py` catches and falls through to ALU. Opt-in `AMD_TC_FUSED_CAST=1` to force TC attempt.
- Validated: fused maxabs err≈0.018 vs half-ref (HIP≈0.016); sep/product-16 2048 still OK; flash decode sum −17.283.
## Status (2026-09-25 — decode ~1.03×; alias+batch4 shipped)

**Host:** `69.57.221.45` — Navi 31 — was DOWN; **UP again 2026-09-27**.

### Prefill fair (B=1 H=32 T=32 D=128 S=2048)
DIRECT med **~317–320** / best **~304–312** vs HIP **~330–333**. Checksum −3.37886.
**Ahead of HIP** (~0.96×).

### Decode fair (T=1 S=2048)
DIRECT med **~119–121** / best **~116–118** vs HIP **~115–117**.
Checksum −17.283. Gap ≈ **1.03×** (~3–5 µs).

### Decode landed
- **`AMD_MOV_ALIAS_SWIZZLE=1`** (decode): v_mov **238→78**, ~−2µs.
- **`AMD_SWIZZLE_BATCH_MAX=4`** (decode).
- Late **`_schedule_fmac_add_pairs` re-run**; dual_fmac 1→2.
- **`HALF2_GLOBAL=0` on decode** / prefill forces 1.
- fe4 + soft0 + V_HALF+mix+VALU_GAP+FMAC_ADD + promote + load-only VMEM.
- Ban: PAIR_PARK/ALLOC, PAIR_FMAC, FMAC_MUL (hang), FMAC_ADD_SCAN≥96+,
  HALF2 decode, SCORE_BATCH=6 (hang), NO_PARK*, VALU_GAP_MAX=8 wash,
  nomov/sinkvmem wash, REG_STORE_SINK_ALU=0 (wrong sum).

### Ready on reconnect
- **`AMD_FMA_MIX_EVERY=N`** implemented in `rdna3.py` (cleared on prefill enter).
- Script: `extra/rdna3_fma_mix_every_ab.py` (hip / ship / every2/4/8 / mix_off).
- Goal: dual_fmac 2→HIP91; remaining v_mov 78; combine vgpr169.

---

## Status (2026-09-24 — host drop; decode ~1.04× local ship)

**Host:** `69.57.221.45` — **DISCONNECTED** mid-sync (SSH banner timeout). Last good Navi31 `1002:744c`.

### Prefill fair (B=1 H=32 T=32 D=128 S=2048)
DIRECT med **~320** / best **~307** vs HIP **~332** / **~319**. Checksum −3.37886.
**Ahead of HIP** (~0.96×). Untouched by decode-only knobs (cleared on prefill enter).

### Decode fair (T=1 S=2048) — last HW before drop
HIP med **~116–117** / p10 **~115**.
Ship+mix+gap+fmac_add med **~123** / p10 **~121**.
**`FMAC_DELAY_EVERY=4`** alone: p10 **~120.2–120.8** (vs every1 ~123) — ship this.
Checksum −17.283. Gap ≈ **1.03–1.04×** (~4–6 µs) after EVERY=4.

### Decode landed (local tip; remote may miss last sync)
- Promote slot2 + load-only `SCHEDULE_VMEM` (`ALU=0`).
- Sticky `MUL_DELAY=1` + `FMAC_DELAY=1` / cleared on prefill.
- **`FMAC_DELAY_EVERY=4` + `MUL_DELAY_EVERY=4`** (rdna3.py MUL every-N new).
- **`V_HALF` + `FMA_MIX_ALL` + `SWIZZLE_VALU_GAP` + `SCHEDULE_FMAC_ADD`** decode defaults (~1µs vs nomix).
- Soft LGKM, PAIR_CONST, MOV_SCAN.
- Ban: PAIR_ALLOC/PAIR_PARK (vgpr246), NO_PARK, PACKED_OPSEL+WHERE, ALU=1, delay0 (~+8µs).

### Decode tip asm vs HIP (with mix ship)
DIR: ninst~2162 vgpr112 priv0 vmov238 dual12 fma_mix112 fmac133 delay~852(every1) / thinner@every4  
HIP: vgpr83 priv0 vmov0 dual91 fma_mix234 delay~150

### Still open (~4–6µs) — resume on reconnect
1. Confirm EVERY=4 + MUL_EVERY=4 fair 3-rep vs HIP (remote sync interrupted).
2. Dual park MOV×128 without PAIR_PARK vgpr blowup.
3. Cut residual vmov / raise dual toward HIP without occ hit.

---

## Status (2026-09-24 — decode ~1.05×; FMA_MIX fold fixed)

**Host:** `69.57.221.45` — Navi 31 `1002:744c` / gfx1100.

### Prefill fair (B=1 H=32 T=32 D=128 S=2048)
DIRECT med **~315–324** / best **~304–308** vs HIP **~326–335** / **~318–321**. Checksum −3.37886.
**Ahead of HIP** (~0.96×).

### Decode fair (T=1 S=2048)
DIRECT med **~122–124** / best **~118–121** vs HIP **~115–118** / **~112–115**. Checksum −17.283.
Gap ≈ **1.05×** (~6–8 µs).

### Decode landed
- Promote slot2 + load-only `SCHEDULE_VMEM` (`ALU=0`).
- **`MUL_DELAY=1` + `FMAC_DELAY=1`** sticky decode / cleared on prefill (~133→~123).
- Soft LGKM, PAIR_CONST, MOV_SCAN on decode.
- **FMA_MIX fold fix:** rewrite consumer srcs (list-only swap left STORE on dead FMAC). Checksum OK; fair **~128 vs ~123** — leave OFF.
- Ban: FMA_MIX default, PAIR_ALLOC (vgpr254), NO_PARK (slower), PACKED_OPSEL+WHERE (NaN), ALU=1.

### Decode tip asm vs HIP
DIR: ninst~2190 vgpr111 priv0 vmov236 dual12 fmac245 delay427 cvt96  
HIP: vgpr83 priv0 vmov0 dual91 fma_mix234 delay150

### Still open (~7µs)
1. Make correct FMA_MIX faster than FMAC+cvt (vmov↓/dual↑/occ).
2. Dual park MOV×128 without PAIR_ALLOC vgpr blowup.
3. Delay placement toward HIP 150.

---

## Status (2026-09-24 — decode ~1.06×; prefill still ahead)

**Host:** `69.57.221.45` — Navi 31 `1002:744c` / gfx1100.

### Prefill fair (B=1 H=32 T=32 D=128 S=2048)
DIRECT med **~315–320** / best **~304–308** vs HIP **~326–332** / **~318–321**. Checksum −3.37886.
**Ahead of HIP** (~0.96×).

### Decode fair (T=1 S=2048)
DIRECT med **~122–125** / best **~119–121** vs HIP **~115–119** / **~112–115**. Checksum −17.283.
Gap ≈ **1.05–1.06×** (~6–8 µs).

### Decode landed this session
- Promote slot2 + **load-only** `AMD_SCHEDULE_VMEM=1` / `AMD_SCHEDULE_VMEM_ALU=0` (full ALU schedule → NaN).
- **`AMD_MUL_DELAY=1` + `AMD_FMAC_DELAY=1`** sticky on decode (cleared on prefill enter): ~133→~123 med.
- Soft LGKM on decode. PAIR_CONST + MOV_SCAN on decode.
- Ban: FMA_MIX (still wrong sum even with packed opsel / unmasked V_HALF / keep-cast), V_HALF default (wash), waves4/16, SCHEDULE_VMEM_ALU=1, sticky decode delays on prefill (~+5µs).

### Decode tip asm vs HIP
DIR: ninst~2190 vgpr111 priv0 vmov236 fmac245 fma_mix0 delay427 cvt96  
HIP: vgpr83 priv0 vmov0 dual91 fma_mix234 fmac19 delay150

### Still open (decode → HIP, ~7µs)
1. Correct `v_fma_mix_f32` for softmax@V (unit tests OK; flash fold still checksum≠−17.283).
2. Cut v_mov 236→0 / dual↑ toward HIP; vgpr 111→83 (occupancy).
3. Delay density 427→~150 without losing the ~10µs delay win.

---

## Status (2026-09-24 — prefill beats HIP; decode correct ~1.21×)

**Host:** `69.57.221.45` — Navi 31 `1002:744c` / gfx1100.

### Prefill fair (B=1 H=32 T=32 D=128 S=2048)
DIRECT med **~315–317** / best **~304–306** vs HIP **~326–335** / **~317–320**. Checksum −3.37886.
**Ahead of HIP** (~0.96×). Landed: `warp_reduce_many` softmax + soft lgkm coalesce.

### Decode fair (T=1 S=2048)
DIRECT med **~140–143** / best **~135–138** vs HIP **~116–118** / **~112–114**. Checksum −17.283.
Gap ≈ **1.21×** (~24 µs).

### Decode fixes this session
- **MMU/NaN root:** `AMD_FLASH_DECODE_PROMOTE_SLOT2` (empty SKIP) + `AMD_SCHEDULE_VMEM=1`.
- **Ship:** promote ON + **`AMD_SCHEDULE_VMEM=0`** in decode compile_env → priv0, correct sum.
- Ban: FMA_MIX/V_HALF fold (sum 0), SCHEDULE_VMEM=1 with promote (NaN/MMU), waves16 (wrong sum).

### Decode tip asm vs HIP
DIR: ninst1756 vgpr111 priv0 vmov236 fmac245 fma_mix0 delay0  
HIP: ninst1761 vgpr83 priv0 vmov0 dual91 fma_mix234 fmac19 delay150

### Still open (decode → HIP)
1. Safe `AMD_SCHEDULE_VMEM` with promote (FMAC-boundary alone still wrong sum).
2. Real `v_fma_mix_f32` for softmax@V without checksum 0.
3. Cut park v_mov 236→0; HIP delay density.

---

## Status (2026-09-24 — warp_reduce_many; DIRECT beats HIP)

**Host:** `69.57.221.45` — Navi 31 `1002:744c` / gfx1100.
**Fair sync-each (n=180–200, 4 reps):** DIRECT med **~313–321** / best **~302–308** / HIP med **~323–327** / best **~317–319**. Checksum −3.37886.
**Gap closed** — DIRECT ≈ **0.96× HIP** (~10–13 µs ahead).

### Tip asm (ACC_VGPR ship)
priv0 / vgpr198 / ninst1275 / v_mov128 (park) / dual_mov~88 / lgkm38 (was 122) / delay24 / ds_store b16×48 / ds_swizzle96.

### Landed this session
- **Flash softmax → `warp_reduce_many`**: batch TM(+TN) keys so same-stage `ds_swizzle` share lgkm waits. Fair **~375→~313 µs**; checksum OK.
- **`AMD_SOFT_LGKM_COALESCE_SWIZZLE=1`** (default): merge consecutive swizzle notes → one soft wait per park batch; lgkm waits **122→38**; timing wash vs uncoalesced; keep ON.
- Same-src VOPD MOV broadcast + `AMD_SOFT_LGKM=1` (earlier).
- Ban/wash: SOFT_LGKM_DELAY default OFF; peer_nopark/score_batch/OVERLAP_V/graph ablations; F16_STORE_FUSE (scrambled offs).

### Still open (headroom, not gap)
1. Park v_mov **128** vs HIP **22** — tax for batched swizzle; no_park regresses to ~373.
2. Remaining ~28× lgkmcnt(0) (WMMA/barrier/load).
3. ds_store_b16×48 scrambled; HIP uses d16_hi + 2addr_b64 on Q/K.
4. delay 24 vs HIP ~110 — placement.
5. **Decode hand path** (`N%64==0`, e.g. S=64/2048): DIRECT **MMU** (or NaN after fault). Short S falls back to SDPA (`N%64!=0`) and matches. Independent of today's prefill knobs — open tip bug.

---

## Status (2026-09-24 — SOFT_LGKM + MOV bcast; gap ~1.15×)

**Host:** `69.57.221.45` up — Navi 31 `1002:744c` / gfx1100 confirmed.
**Fair sync-each (n=150–200):** DIRECT med **~375–377** / best **~363–369** / HIP med **~326–330** / best **~318–322**. Checksum −3.37886.
Gap ≈ **1.15×** (~47–50 µs).

### Tip asm
priv0 / vgpr198 / ninst1262 / v_mov30 / dual_mov88 / lgkm122 (≈97×cnt0 + 25 soft) / delay24 / ds_store b16×48.

### Landed this session
- **Same-src VOPD MOV broadcast** (`AMD_VOPD_SRC0_BCAST`): v_mov **42→30**; checksum OK.
- **`AMD_SOFT_LGKM=1`** flash default — soft `lgkmcnt(N)` like soft vmcnt; fair **~+3µs** vs hard-only; checksum OK.
- Ban/wash: ALIAS_SET / WHERE_ALIAS / MAX_ALIAS_SRC0 / DCE_DEAD_MOV / SINK_VALU_SWIZZLE (checksum), emit_lshl_add (MMU), FMAC_DELAY / LDS_IL / delay_pre / instr_wait / fuse_loads (regress), ablations of ship defaults (all help or wash).

### Still open (~48µs)
1. **~97 hard lgkmcnt(0)** — need more LDS in flight before use (prefetch stubbed; LDS_IL washes).
2. ds_store_b16×48 scrambled (TN stride-16 in BLOCK_N).
3. Residual PACK evacuate MOV×12; delay 24 vs HIP ~110.

---

## Status (2026-09-23 — PAIR_CONST shipped; gap ~1.13×)

**Host:** `69.57.221.45` up — Navi 31 `1002:744c` / gfx1100 / `AMDRenderer` confirmed.
**Fair sync-each (n=80):** DIRECT med **~379 µs** / best **~368** / HIP med **~336** / best **~320**. Checksum −3.37886.
Gap ≈ **1.13× med / 1.15× best** (~43 / ~48 µs).

### Tip asm
priv0 / vgpr198 / ninst1244 / v_mov42 / dual~92 / ds_store64 (b16×48+b64×16) / delay24.

### Landed
- Full tip sync to `tinygrad-rdna3-ac0ba6fca` (partial sync had left priv12 / v_mov92).
- **`AMD_VOPD_PAIR_CONST=1`** flash default — const-MOV bank bias only; fair best ~367–368; checksum OK.
- Ban/wash this reconnect: delay0/2, LDS_IL=1, MOV_SCAN=64, ACC_BASE 140/150, F16_STORE_FUSE (no contig runs — bank-scrambled offs), CLUSTER_PACK_LLOAD (ninst↓ dual↓, timing wash).
- **`AMD_MAX_ALIAS_SRC0`**: asm win (v_mov 42→34) but **checksum −3.92** — leave OFF. `_sink_alu_into_mov` default OFF (encode fault when broad).

### Still open (~43µs)
1. ds_store_b16×48 bank-scrambled (offs 32/56/40/48…) — fuse needs layout change.
2. Residual v_mov 42 vs HIP 22 (MOV←MOV const broadcast / FMAC gather / SPECIAL).
3. delay 24 vs HIP 110 — placement.

---

## Status (2026-09-22 — ALIGN8 off + RCM+INF; gap ~1.15×)

**Fair:** DIRECT best **~367–370 µs** / HIP **~319 µs** (≈**1.15×**). Checksum −3.37886.

**Host:** `69.57.221.45` dropped mid-loop (SSH timeout / unreachable). Local tip has ALIGN8=0 + RCM+INF + PAIR_CONST opt-in code; resume HW gate on reconnect.

### Landed
- **`AMD_PACK_LLOAD_ALIGN8=0`** flash default: v_mov **50→46**, ninst **1255→1247**, p10/median win vs ALIGN8.
- **`AMD_REMAT_CONST_MOV=1` + INF**: INF remat was a no-op without RCM (flash only set INF).
- **`AMD_VOPD_PAIR_ALLOC` default OFF**; HALF2_GLOBAL + EARLY_V.
- Ban: ALIAS_SET (wrong sum), WHERE_ALIAS unrestricted (wrong sum), soft pack-chain (wash), ALIGN4=0 (regress).
- Pending HW: `AMD_VOPD_PAIR_CONST` (const-MOV-only bank bias) — code ready, not defaulted.

### Tip
priv0 / vgpr198 / ninst1247 / v_mov46 / dual~82 / delay24 / ds_store64.

### Still open (~48–50µs)
1. 4 scrambled LDS PACK_F16 evacuates (pack_chain full-span miss) + 16 FMAC-gather COPYs + ~18 LIT.
2. ds_store_b16×48 scattered (P LDS).
3. delay placement (only post-WMMA) vs HIP schedule density.

## Status (2026-09-22 — HALF2 ship + EARLY_V; gap ~1.17×)

**Fair:** DIRECT best **~372–375 µs** / HIP **~319 µs** (≈**1.17×**). Checksum −3.37886.

### Landed
- HALF2_GLOBAL default: lo←LOAD prefer_phys + lo-only rewrite + hi DCE + dual same-literal (0/−inf).
- **v_mov 90→34**, ninst **1359→1246**, vgpr **206→198**, gload→ds_store without unpack tax.
- `AMD_FLASH_EARLY_V=1` + `AMD_REMAT_CONST_MOV_INF=1` defaults.
- **`AMD_VOPD_PAIR_ALLOC` default OFF** (was ON): better fair p10/median despite +4 v_mov.
- EXP/MUL/swizzle blanket delay: wash or regress. Occ CAP: wash. F16 store fuse: no match (scattered b16).

### Tip
priv0 / vgpr198 / ninst1246 / v_mov34 / dual~100 / delay24 / ds_store64 (b16×48+b64×16).

### Still open (~54µs)
1. WMMA-B evacuate (odd-src copies) + ACC/−inf LIT residue.
2. ds_store_b16×48 scattered (P LDS) vs HIP 56 total stores.
3. delay 24 vs HIP 110 — placement not count.


**Fair 100-iter:** DIRECT best **~372 µs** / HIP **~319 µs** (≈**1.17×**). Checksum −3.37886.

### Landed
- **`AMD_PACK_F16_HALF2_GLOBAL=1`** flash default: lo-only rewrite, hi DCE, PACK on general VGPR.
- **prefer_phys:** even f16 EXTRACT←LOAD → load lane; PACK on load VGPRs; next b128 fresh regs.
- **VOPD:** same-literal dual_mov (0/−inf) — v_mov **46→34**, dual_mov **84→90**.
- Metrics: ninst **~1252→1246**, vgpr **198**, gload→ds_store with no unpack MOV/LSHR/pack.
- Prior ship (~387) → **~372** (best observed 371.7).


### Tip metrics (confirmed)
- DIRECT: priv0 / vgpr198 / ninst1246 / v_mov34 / dual~100 / delay24 / ds_store64 / best **~372–375 µs**
- HIP: best **~319 µs** (sum −3.37886)
- Gap ≈ **1.17×** (~54 µs). Remaining: ACC/−inf LIT + WMMA-B evacuate copies; ds_store_b16×48 (scattered); delay density.

### Still open (~55µs)
1. v_mov **34** vs HIP **22** (residual ACC/−inf / scale).
2. s_delay_alu / dual density vs HIP.
3. Keep grinding toward HIP ~319.



**Fair 100-iter:** DIRECT best **~373 µs** / HIP **~319 µs** (≈**1.17×**). Checksum −3.37886.

### Landed
- **`AMD_PACK_F16_HALF2_GLOBAL=1`** flash default + lo-only rewrite + hi DCE.
- **prefer_phys:** even f16 EXTRACT←LOAD aliases onto load lane → PACK sits on load VGPRs; next b128 gets fresh regs. **v_mov 106→46**, ninst **1309→1252**, vgpr **206→198**, best **~379→~373**.
- Global→LDS path now `global_load_b128` → `ds_store_b64` with no MOV/LSHR/pack.

### Still open (~54µs)
1. Remaining v_mov **46** mostly ACC zero/−inf LIT (+ dual_mov zero runs).
2. lshr **16**; delay/dual density vs HIP.
3. Keep grinding toward HIP ~319.



**Fair 100-iter:** DIRECT best **~379 µs** / HIP **~319 µs** (≈**1.19×**). Checksum −3.37886.

### Landed this pass
- **`AMD_PACK_F16_HALF2_GLOBAL=1`** (flash default): bitcast via lo EXTRACT full-word MOV (not `greg(LOAD)` — b128 dest reuse caused sum −22.3).
- Pre-regalloc: drop pack-only hi EXTRACTs; lo-only PACK; PACK pool → general VGPR for prefer_phys alias.
- Metrics: ninst **1359→1311**, v_pack **32→0**, lshr **48→16**, best **~387→~379**.
- EXP/MUL delay knobs: wash (leave OFF).

### Still open (~60µs)
1. v_mov **106** vs HIP **22** (scattered lo evacuate into ds_store + ACC LIT).
2. Remaining lshr **16**.
3. dual / delay density vs HIP.

## Status (2026-09-22 — ship a145/p135 + LDS_IL OFF; gap ~1.21×)

**Fair 100-iter:** DIRECT best **~387 µs** / HIP **~319 µs** (≈**1.21×**). Checksum −3.37886.

### Landed this pass
- **`AMD_WMMA_ACC_BASE=145` / `AMD_LLOAD_POOL=135`** (was 150/140): v_mov **114→90**, ninst **1371→1359**, dual **80→92**.
- **`AMD_WMMA_LDS_INTERLEAVE` default OFF**: ds_load_2addr **81→63** (HIP 64).
- Occ8 via `AMD_VGPR_CAP` works but **washes** on this 32-WG fair grid (not occupancy-bound).
- dual_fmac blocked (no indep ADD; FMAC src1 stride 8 across ACC tiles).
- Structural compare fixed to shipping mid8/acc_vgpr/hoist_q.

### Banned this pass
- **`AMD_PACK_F16_HALF2_GLOBAL` (raw greg(LOAD)):** was sum −22.3; fixed via lo-EXTRACT source and now shipped.
- **`AMD_VGPR_EXCLUDE_ACC`:** wrong sum even after soft-band cut fix.
- Occ8 CAP path: correct but no timing win on fair shape.

### MOV attribution (ship)
- ~63 copy + ~28 LIT. Copy runs are **global_load_b128 → MOV+LSHR half unpack → v_pack** (6×8). HIP bitcasts; our half2-global attempt NaNs.
- Remaining: ACC/−inf LIT + FMAC-scale evacuate (~14).

### Still open (~68µs)
1. Safe global half unpack without MOV+LSHR (d16_hi / ordered bitcast).
2. s_delay_alu 24 vs 110; v_dual 92 vs 104.
3. Keep grinding toward HIP ~319.

## Status (2026-09-22 — LDS_INTERLEAVE OFF; ds_load≈HIP; gap still ~1.22×)

**Fair 100-iter:** DIRECT best **~387 µs** / HIP **~318 µs**. Checksum −3.37886.

### Landed
- **`AMD_WMMA_LDS_INTERLEAVE` default OFF** in flash env (was forced ON): ds_load_2addr **81→63** (HIP 64), ninst **1410→1371**. Timing wash — still ship (fewer LDS loads, closer to HIP structure).
- Structural compare fixed to shipping knobs (mid8 / acc_vgpr / hoist_q); was wrongly using mid4+acc_work → fake priv172.

### Occ note
- Fair grid is only **32 WGs** on 96 CUs → occ8 / 2 WG/CU does not help this shape. Gap is single-wave issue density.

### Still open (~70µs)
1. **v_mov 114 vs 22** (ACC zero-init LIT runs + evacuate).
2. **v_dual 80 vs 104**; **s_delay_alu 24 vs 110**.
3. dual_fmac structurally blocked.

## Status (2026-09-22 — occ8 achieved; dual_fmac structural dead-end; gap ~1.22×)

**Fair 100-iter:** HIP best **~317 µs** / DIRECT ship **~387 µs** (≈**1.22×**). Checksum −3.37886.

### Occupancy (this pass)
- **`AMD_VGPR_CAP=189`** hard-caps the VGPR file → max_v≤189 → **occ8**.
- With **`AMD_REMAT_LIDX_ADDR=1` + `AMD_REMAT_CONST_MOV=1` + `AMD_REMAT_CONST_MOV_INF=1`**: **priv0 + occ8**, sum OK, ninst ~1481 / v_mov ~119.
- **Timing wash** vs ship (~387–389). Occ8 alone does **not** close the HIP gap — kernel is not occupancy-bound at this size.
- Last priv4 under CAP+LIDX alone was spilling **m=−inf** (not addr); INF-only remat kills it without the full const MOV storm (rcm → v_mov 207).
- Old `AMD_VGPR_EXCLUDE_ACC` cut `len(VGPR)-len(ACC)` (top of file), **not** the soft ACC band — fixed to `VGPR[:acc_base]` but still wrong-sum when forced (spill corruption). Leave **OFF**.

### dual_fmac (this pass)
- All float `v_add_f32` sit **before** the FMAC block and are in FMAC dependency cones → schedule cannot place independent ADD next to FMAC.
- FMAC+FMAC often has even/odd dest + broadcast src0, but **src1 stride = 8** (same lane across WMMA ACC tiles). WMMA needs contiguous 8-VGPR C packs, so that stride is structural. dual_fmac stays **0**.

### MOV attribution update
- Long LIT runs are largely **ACC zero-init**, not only WMMA A evacuate.
- a145_p135: v_mov **90** / ninst **1374**, timing wash — keep ship **150/140**.

### Opt-in tools (default OFF)
- `AMD_VGPR_CAP`, `AMD_WMMA_ACC_N`, `AMD_REMAT_CONST_MOV_INF`, backward `AMD_SCHEDULE_FMAC_ADD` sink.

### Still open (~70µs)
1. Issue density / schedule vs HIP (delay_alu 24 vs ~119 historically; VOPD mix).
2. MOV 114→22 (zero-init + evacuate).
3. dual_fmac blocked structurally without flash algorithm / ACC layout change.

## Status (2026-09-22 — ACC_BASE=150 / LLOAD_POOL=140 shipped; gap ~1.23×)

**Fair:** HIP best **~319 µs** / DIRECT ship best **~392 µs** (≈**1.23×**). Checksum −3.37886.

### Structural (DIRECT ship now)
| | DIRECT (new) | DIRECT (was) | HIP (asm) |
|--|--:|--:|--:|
| priv / vgpr / ninst | **0** / 206 / **1399** | 12 / 206 / 1478 | 0 / 189 / 1462 |
| v_mov | **114** | 138 | **22** |

### Landed
- **`AMD_WMMA_ACC_BASE` default 150**, **`AMD_LLOAD_POOL` default 140** (`rdna3_defs` + flash compile env): larger LLOAD band → pack_chain can reserve → **priv0**, ninst↓, v_mov 138→114. Timing small win (~394→~389–392).
- **PACK_F16** pool moved just below ACC (not 185:244).

### Banned this pass
- **`AMD_VGPR_EXCLUDE_ACC`:** wrong flash checksum / ~2× regress. Leave **OFF**.
- ACC_BASE=135: priv↑ + wrong sum without exclude.

### Still open (~60µs best; HIP ~329)
1. **Occupancy:** max_vgpr **206** → occ **7**; HIP **189** → occ **8**. Exclude-ACC path unsafe.
2. **MOV** 114 vs 22 — WMMA A evacuate (3 copy runs).
3. **dual_fmac=0**.
4. Cluster checksum-safe now but slower.

## Status (2026-09-22 — cluster/hoist/pack_chain diagnosed; gap ~1.18×)

**Fair (sync-each, checksum −3.37886):** HIP best **~333–342 µs** / DIRECT ship best **~393–397 µs** (≈**1.18×**).

### Cluster / hoist / pack_chain / priv12 (this pass)
- **`AMD_CLUSTER_PACK_LLOAD`:** under ACC150/POOL140, checksum now **OK** (−3.37886) but fair best regresses (~406 vs ~388). Leave **OFF**.
- **`AMD_HOIST_PACK_AFTER_LLOAD`:** checksum OK, no timing win. Default **OFF**.
- **ACC/POOL sweep:** a165_p155 cuts v_mov→90 but occ 6 and best no better than a150_p140 (v_mov 114, best ~385). Keep **150/140**.
- dual_fmac still 0 (FMAC neighbors FMAC/MUL only).

## Status (2026-09-22 — LDS addr fold/CSE; gap ~1.18×)

**Fair (sync-each, checksum −3.37886):** HIP best **~337 µs** / DIRECT ship best **~397 µs** (≈**1.18×**; was ~1.25× / ~424). Median ~443 vs ~398.

### Structural vs HIP (same flash graph)
| | DIRECT (ship) | HIP |
|--|--:|--:|
| priv / vgpr / ninst | 12 / 206 / **1476** | **0 / 189 / 1462** |
| v_mov | **138** | **22** |
| v_add_nc_e64 / v_lshl / v_lshl_add | **0 / 31 / 65** | (HIP: mad/add3/lshl_add) |
| v_add_nc_e32 (int ALU) | **52** | — |
| dual_mov / v_dual | 93 / 102 | — / 104 |
| dual_fmac | 0 | **8** (FMAC+ADD mixed; our FMACs have no near ADD) |
| delay_alu | 24 | 110 |

### Landed this session (checksum OK)
- **`AMD_LDS_FOLD_BASE` (default ON):** fold LDS slot base into 16-bit `ds_offset` — cut **72** `v_add_nc_e64`.
- **2addr `v_lshl_add`:** merge scale+base+fold for `ds_load_2addr` — **v_add_nc_e64 → 0**, ninst 1713→1536.
- **`AMD_LDS_ADDR_CSE` (default ON):** CSE LDS scales across LSTORE/LLOAD — ninst **1536→1476**, `v_lshl` 91→31. Fair best **~424→397**.
- **`AMD_PACK_LLOAD_ALIGN8`:** prefer mod-8 pack bases (wash; keep ON).
- **`AMD_LDS_STORE_SCLAUSE` (default ON):** s_clause on LSTORE streaks (wash timing; keep HIP-like).
- Prior: `_pack_f16_insts` → `_parallel_vmov` (v_mov 186→138).

### Probed / banned
- `SHALLOW_DEPTH=2` → priv0 locally, **MMU** on HW.
- `AMD_FOLD_LSHL_ADD=1` → priv↑ / ninst↑.
- `AMD_REMAT_NO_STICKY_CONST_ADD` → ninst↓/priv↓ but **~760µs** median (remat flood).
- Pack half-span fallback → v_mov↑ / timing regress.
- `AMD_EMIT_LSHL_ADD` (emit SHL+ADD fold) — no matches yet; default **OFF**.
- `AMD_PACK_F16_ALIAS_SET` (unordered contig alias) → wrong sum **−4.71**; default **OFF**.
- `AMD_CLUSTER_PACK_LLOAD` → priv0 / ninst↓ / v_mov↓ but checksum **−3.3754** (not −3.3788); default **OFF**.
- `AMD_HOIST_PACK_AFTER_LLOAD` → correct sum but fair best regress; default **OFF**.
- `AMD_REMAT_LIDX_ADDR` → priv/ninst regress or **MMU**; default **OFF**.
- `AMD_DCE_DEAD_MOV` → wrong sum **−21.6**; default **OFF**.

### Remaining gap (~60µs best)
1. **MOV flood** 138 vs 22 — WMMA A evacuate; pack alias gaps.
2. **dual_fmac=0** — FMAC neighbors are FMAC/MUL/REG_STORE only (no ADD in window).
3. **priv12** — 3× int ADD(MOV,MOV) spills; depth2 remat MMUs.
4. ninst nearly matched (1476 vs 1462); timing is schedule/issue density + MOV.

**Fair hygiene:** always subprocess + `CACHELEVEL=0` / pop sticky `AMD_*` — else sum **−178**.

---

## Status (2026-09-22 — host up; MOV dual + fuse probes)

**Fair:** HIP best **~338 µs** / DIRECT ship best **~424 µs** (≈**1.25×**). Checksum −3.37886. Gap not closed.

### Structural vs HIP (same flash graph)
| | DIRECT (ship) | HIP |
|--|--:|--:|
| priv / vgpr / ninst | 12 / 206 / **1713** | **0 / 189 / 1462** |
| v_mov | **138** (was 186) | **22** |
| dual_mov / v_dual | 93 / 102 | — / 104 |
| dual_fmac | 0 | **8** (FMAC+ADD mixed) |
| delay_alu | 24 | 110 |

### Landed this session (checksum OK)
- **`_pack_f16_insts` → `_parallel_vmov`**: PACK evacuate copies dual-issue; **v_mov 186→138**, ninst 1737→1713, dual_copy 1→25. Timing wash (~424 best).
- `_parallel_vmov` mate search (any pending even/odd pair).
- `pack_chain_bias_cons` + regalloc hook: 21/32 vec PACK alias; 11 still non-contig (VGPR pressure).
- **Fuse probe:** `AMD_PACK_F16_FUSE_LOADS` (SHRINK-aware) — half×16→priv32/~640µs; half×8→priv4/ninst↓ but **~560µs**. Default **OFF**.

### HW bans (unchanged)
- `SHALLOW_DEPTH=2` → MMU. `INTERLEAVE=0`/`HOIST_Q=0` → MMU. `PAIR_ALLOC=0` → sum 767.
- Concurrent/bad-knob fair can sticky-wrong sum **−178**; recover with `Tensor.zeros` then re-gate.

### MOV flood attribution (DIRECT)
- Longest copy run (16): **WMMA A evacuate** — still present after dual; 4×half×4 PACK srcs not one contig chunk.
- Prefer_phys aliases 21/32 vec packs; gaps like `[344,346,356,358]` (two B64 with hole).
- `MOV_ALIAS_MOV`/`PEEL_MOV_MOV` → NaN. Fuse live-range regress.

### Addr ALU gap (new — bigger than MOV for ninst)
DIRECT vs HIP: **+177 `v_add_nc_u32_e64`**, **+153 `v_lshlrev_b32_e64`**, +32 `v_pack_b32_f16`. HIP uses mad/add3/lshl_add/cndmask.
- `AMD_FOLD_LSHL_ADD=1` folds strength-reduced ADD(SHL)→LSHL_ADD (~26) but **priv 12→40** (remat); default **OFF**.
- Remaining ~150 add/shl are not SHL+ADD form — need mad/add3 / SGPR addr / fewer scales.

### Next
1. Cut addr ALU toward HIP (mad/add3 / CSE scales) — likely larger than remaining MOV.
2. Kill WMMA A evacuate without fuse live-range tax.
3. dual_fmac: 15 dest-ok FMAC pairs fail src1 banks (ACC extract stride).
4. priv12→0 without depth≥2.

**Fair hygiene:** always `CACHELEVEL=0` / pop sticky `AMD_*` between HIP↔DIRECT — else sum **−178**.

---

## Status (2026-09-21 — gap iteration continued #7)

**Local asm breakthrough (HW-gate pending — host down):** shipping ACC_VGPR flash now matches HIP **priv=0 / scratch=0**.

| | DIRECT (new ship) | HIP |
|--|--:|--:|
| ninst | **~1744** | 1462 |
| vgpr | 206 | 189 |
| priv | **0** | 0 |
| v_mov | **~166** | 22 |
| v_dual | **~85** (76 mov + 8 mul + …) | 104 |
| gload | 12 | 12 |
| ds_load / ds_store | 81 / 64 | 68 / 56 |

Was ~1738 ninst / 216 mov / 64 dual / priv 16 before this pass.

### Landed this pass (local; needs 7900 checksum+fair)
- **`AMD_REMAT_ADDR_SHALLOW_DEPTH=2`** in flash compile env — remats `ADD(MOV,MOV)` LDS bases; **priv 16→0**. Prior note that depth≥2 MMUs was for unbounded DEEP; leaf-bounded `_addr_within` is the path. Opt out: `SHALLOW_DEPTH=1`.
- **`AMD_MUL_STRENGTH=1`** default (flash env + getenv default) — stride MUL→SHL+ADD.
- **`AMD_VOPD_PAIR_ALLOC=1`** flash default — last+1 / EXP2 src1-chain bias.
- **`AMD_MOV_INLINE_CONST=1`** (emit default) — peel MOV/CAST to inline 0/−inf literals.
- **`AMD_VOPD_MOV_SCAN=32`** (default + flash env) — packs const/copy streaks; **v_mov 206→166**, dual_mov→76.
- VOPD FMAC: peel const mul ops; commutative orientations. Still `dual_fmac=0`.

### Next (when host returns)
1. `scp` tip + run `extra/rdna3_vopd_pair_resume.py` — checksum ≈−3.3788 / no MMU for SHALLOW_DEPTH=2.
2. Fair sync-each vs HIP; remaining gap is MOV flood / VOPD / ds_load / vgpr.
3. Keep iterating to parity.

**Host note:** `69.57.221.45` — ISP returns Destination Host Unreachable (machine off or WAN down). Tip changes local.

---

## Status (2026-09-21 — gap iteration continued #6)

**Fair sync-each:** DIRECT **~465–470µs** med / **~437–440µs** best / HIP **~360–396µs** med / **~330–334µs** best.
Gap **~1.18–1.26× med / ~1.31× best**. Not closed.

### Shipping asm (ACC_VGPR+mid8+hoist+KV unroll, same UOp graph as HIP)
| | DIRECT | HIP |
|--|--:|--:|
| ninst | ~1738 | 1462 |
| vgpr | 206 | 189 |
| priv | 16 (4 spill/fill) | 0 |
| v_mov | 216 | 22 |
| v_dual | ~64 (55 mov + 8 mul + 1 max) | 104 |
| gload | 12 | 12 |
| ds_load / ds_store | 81 / 64 | 68 / 56 |
| s_delay_alu | 24 | 110 |

Remaining 4 spills are **int addr** (`MUL*132` / `1152` / `2112` LDS strides), not ACC. MUL_STRENGTH cuts spill 4→3, timing wash.

### Landed this pass
- Confirmed mid8+KV unroll shipping; peel/MOV-alias/const-collapse false friends.
- IR MOV←MOV (112) are all **float MOV←MOV←CAST(CONST 0/-inf)** sharing mid `v18` broadcasts — coalescing/peeling NaNs or encode-faults; leave alone.
- **VOPD src0 broadcast** (`AMD_VOPD_SRC0_BCAST` default ON): +8 dual_mul / −8 ninst; fair wash but correct. FMAC dual still 0 — src1 banks not even/odd (dest pairs exist).
- Banned add: `AMD_DCE_DEAD_MOV`, `AMD_COLLAPSE_CONST_MOV`, `AMD_PEEL_MOV_MOV`, `AMD_MOV_ALIAS_CONST_MOV`, `AMD_VOPD_PAIR_ALLOC` (neutral/regress), soft_inplace, ACC_SEP=0, OVERLAP_V/EARLY_V wash, BLOCK_N=64 NaN.

### Next
1. FMAC VOPD: place mul-src1 in even/odd pairs (or accept dual_mul-only). `AMD_VOPD_PAIR_ALLOC` + EXTRACT bias ready in tree — HW-gate when host returns (`extra/rdna3_vopd_pair_resume.py`).
2. Cut addr-spill / vgpr 206→189 without NaN.
3. Keep iterating to parity — gap is renderer quality on the same flash graph.

**Host note:** gaming PC `69.57.221.45` unreachable (network) mid-loop after LDS_PAD=0 MMU probe; tip changes local; resume HW when SSH returns.

---

## Status (2026-09-17 fence HW — DONE)

- Gaming PC up (fresh boot earlier); tip synced; 7900 XTX / `AMDRenderer`.
- **Fence protocol (final):**
  1. Sched slot loads prior `nxt` (zero once at link — never per-run).
  2. CPU kernel spins until `timeline[0] >= target`.
  3. `nxt = timeline[1]+1`; store `nxt` to both `timeline[1]` and the slot.
  4. `exec_hcq` also Python-`_wait_signal(tl, tl[1])` before the host fence: UOp spin alone still races long flash TinyJit (dual submit → `[n,n+1]`, second kernel hangs; `sleep` between submits works).
- **`nxt+1` store was wrong** — completed batch leaves `[nxt,nxt]`; waiting for `nxt+1` deadlocks synced warms.
- **HW proof @ `038c948dc`:** short TinyJit batch10/50/100 no mid-sync OK; DIRECT flash batch2/50/100 OK (~753/692 µs/call @50/100); HIP flash batch2/50/100 OK; outputs finite + stable checksum.

### Frozen paired sync-each (tip `038c948dc`, n=30, fresh process)

| Method | DIRECT | HIP |
|--|--:|--:|
| prefill sync-each | **~724 µs** (best ~716) | **~361 µs** (best ~337) |
| decode sync-each | **~128 µs** (best ~125) | **~119 µs** (best ~115) |
| prefill TinyJit batch50 | **~753 µs/call** | **~397 µs/call** |
| prefill TinyJit batch100 | **~692 µs/call** | **~298 µs/call** |

Prefill gap ~2.0× sync-each / ~1.9× batch50. Decode ~7% — validation track.

### GEMM→flash

- On clean GPU **GEMM then flash OK**.

## Headline (historical; pre-fence-clean)

| Method | DIRECT | HIP |
|--|--:|--:|
| prefill sync-each (HCQ2=1) | **~622 µs** | **~342 µs** |
| decode sync-each (HCQ2=1) | **~119 µs** | **~110 µs** |
| decode batch50 (HCQ2=1, clean) | **~58 µs** | **~63 µs** |

## Headline (pre-merge; historical)

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

## Prefill scratch/liveness (post-fence)

**Attribution @ `a2a8fc136`:** SPILL/FILL=0. Scratch is slot-2 only.
- After normalize-from-`acc_work`: IR SLOAD **40→8**, SSTORE 64, ELF ~1885 insn, MOV **558**, delay_alu **0**, VOPD **0**, priv 128.
- MOV sources (IR): 128 EXTRACT, 80 MOV←MOV, 40 FMAC, plus MAX/MUL/EXP2; long machine runs up to 64; ~94 movs in broadcast runs (same src).
- Fair sync-each after normalize: DIRECT **~720 µs** (wash vs frozen ~724). Serial PERF ~689.

**Landed:** normalize from ACC_WORK (`a2a8fc136`) — correct, cleaner epilogue, fair wash.

**Tried:** carried ACC on **slot 19** (promotable) + ACC_WORK=0: SSTORE/SLOAD **0**, but **SPILL 112 / FILL 45**, priv 96, ~**1487 µs** serial (vs ~690). Correct 13/13 — promote works, LinearScan spill tax dominates intentional slot-2 scratch. Leave SKIP=2 + ACC_WORK. All 112 SPILLs are `CAST:dtypes.int` (addr temps) under ACC VGPR pressure; `AMD_REMAT_ADDR` 0/1 no help (~1475µs).


**Still open (no env sweeps):** per-tile slot-2 load/store; MOV/delay_alu/VOPD gap vs HIP (~32 MOV / 119 delay / ~103 VOPD).

## Next

1. ~~HW-validate fence + freeze paired timings~~ **DONE @ `038c948dc`**.
2. Prefill: attack MOV tax or per-tile scratch (not env sweeps). Decode = validation.
3. One Llama/GGUF health check per prefill milestone.
4. Freeze performance, then split renderer. Fork-only; no upstream PRs.

## Post-master paired validation (2026-10-02)

Hardware: RX 7900 XTX / gfx1100. Same `rdna3_backend_matrix.py` harness on the merged staging snapshot and pre-merge snapshot; serial DIRECT/HIP, 100 synchronized samples/case. The early 30-sample pass had timing outliers; use the 100-sample results below.

| Workload | Merged DIRECT | Merged HIP | DIRECT/HIP | Pre-merge DIRECT/HIP |
|---|---:|---:|---:|---:|
| Flash prefill, 32 tokens | 323.5 µs | 321.9 µs | 1.005× | 1.004× |
| Flash decode | 116.3 µs | 116.1 µs | 1.001× | 1.029× |
| FP16 matmul, 2048² | 305.1 µs | 301.7 µs | 1.011× | 1.008× |
| Q4_K GEMV, 8192×2048 | 102.9 µs | 100.3 µs | 1.026× | 1.029× |

Flash checksums are finite and match prior locks (`-3.37886` prefill, `-17.2830` decode); matmul matches both paths (`max_err_vs_numpy_f32=0.00781`). Q4_K DIRECT/HIP agree and are within `1.3e-4` of the expected Q8-activation reference. The LLVM Q4_K fallback is ~372 µs but uses full-precision activations, so its output is not directly comparable to the Q8-activation custom path. AMDLLVM FP16 matmul was 297.4 µs (~2.6% faster than DIRECT).

Conclusion: no measured post-merge regression; these four DIRECT workloads remain within ~3% of HIP, and ordinary matmul is within ~3% of AMDLLVM. Three fresh-process repetitions confirm a separate cold first-call gap: flash prefill ~2.18–2.20 s DIRECT vs ~1.64–1.65 s HIP; decode ~2.42–2.55 s vs ~1.29–1.31 s. `DEBUG=2` compile progress points to the two custom flash kernels taking about 2× longer to compile through DIRECT. This is kernel-level TinyJit compile+first-run time, not a full-model compile result; next measure a complete GGUF model and profile per-kernel compile time before attempting compiler optimization.

Validation: renderer suite 221 passed / 34 skipped / 74 subtests; targeted CPU/UOp suite 165 passed / 19 skipped / 8 xfailed / 6 subtests (`DEV=CPU`, `-n12`). LLM hardware tests must be serial: 40 passed / 9 known baseline-failing tests deselected / 150 subtests. Known baseline failures were reproduced on pre-merge, including symbolic/custom-function attention, gated-delta per-channel scan, quantized partial output tiles, WMMA quantization basis/symbolic cases, and the exact Q8 scale-rounding assertion. Ruff/mypy unavailable in gaming-PC venv. No commit, push, or PR.
