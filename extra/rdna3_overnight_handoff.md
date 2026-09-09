# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: *(update after commit)*.

## Headline

Flash DIRECT **opt-in**. Fair fresh-process (7900 XTX; GPU recovering → latencies soft):

| | DIRECT (now) | DIRECT (prior) | HIP |
|--|--:|--:|--:|
| prefill median | **~875–910 µs** (healthy was **~670**) | ~700–720 | **best ~328** |
| WMMA | **6** | 6 | **24** |
| private | **~128 B** | ~184–404 | **0** |
| SPILL | **0** | **14** | **0** |
| VGPR | **~206** | ~166 | **~206** |
| decode (WAVES=8) | **~136 µs** | ~141 | **~156** |

Decode: DIRECT **beats** HIP. Prefill still ~**2×** on best-case.
**SPILL 0** via flash sticky `ALLOW_UPCAST16=0` (product ≤8). Do **not** flip DIRECT default.

Soak: continuous on tip; serial 13/13 when healthy. After MMU/HW fault → gemm smoke + fresh process.

## Landed (stable)

- Decode ml_lds + **default `AMD_FLASH_WAVES=8`**
- Addr remat depth-1; const MOV remat
- Factor K_UNROLL scope; half×16 LSTORE→B128; deep LDS peel
- Soft-fuse / ACC_SEP / ACC_SMALL defaults (do not flip)
- **Flash sticky `ALLOW_UPCAST16=0`** (`_flash_direct_compile_env`): SPILL 14→0, priv ~404→128, VGPR ~166→206 (HIP-like). TinyJit re-reads env after realize — must stay sticky. Override: `AMD_FLASH_ALLOW_UPCAST16=1` or explicit `ALLOW_UPCAST16`. Serial eye/GEMM OK after sticky set.

## Spill anatomy (prefill 32×2048)

**Resolved under default flash path** (was 14; 7 causal `q_idx`). Product-16 (`ALLOW_UPCAST16=1`) was the pressure source under soft-partitioned pools.

## Dead ends this session (do not revive without new plan)

- SHALLOW remat: SPILL↓ but **FAIL** + slower
- QIDX_REG park: correct but **~977 vs ~720**
- Shared q_lane `(k-rm)<=q` rewrite: **FAIL**
- Algebra-identical q_lane hoist: SPILL↑ + slower
- MUL_LEA two-bit strength-reduce: **FAIL**
- CONST_OUTER remat: **FAIL** (gqa/mha)
- LDS_PAD≠4: wrong or MMU
- Slot-2 promote / ACC_WORK=0: slower
- K_UNROLL 2/4 / scoped qk|pv / HIP_SCOPE=-1: slower, wrong, or MMU
- Emit toggles (PACK_SLOAD_B128, FMA_MIX, INSTR_WAIT, WMMA_DELAY, …): no win vs ~720
- Soft-fuse `q_uni+lane_m` split: correct but SPILL 14→21 / ~976µs — reverted
- AMD_WMMA_ACC_BASE sweep: spill stays 14; 201 slower
- AMD_WMMA_ACC_SHARED (full VGPR ACC pool): **MMU fault** — leave off
- Sequential soft-fuse per-row after(): **prefill_mha FAIL** + slower — reverted
- Realize-only `ALLOW_UPCAST16=0` (pop after realize): **no effect** — TinyJit captures post-restore

## Next

1. Keep soak; fresh process after any MMU/HW fault.
2. Prefill: **WMMA 6→24** only with a real plan (K unroll without regressing SPILL 0).
3. With SPILL 0 / VGPR headroom ~206 like HIP: revisit occupancy / waves / emit for the remaining ~2× prefill gap.
4. Optional: `AMD_PACK_F16_GENERAL=1` with UPCAST16=0 was wash (~same as sticky alone).
5. Fork-only; no DIRECT default flip.
