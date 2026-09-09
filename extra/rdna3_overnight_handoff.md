# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`1d7f0907d`** (`AMD_WMMA_DELAY` opt-in, default off).

## Headline

Flash DIRECT **opt-in**. Fair fresh-process (7900 XTX):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~700–720 µs** | **best ~328** (median noisy) |
| WMMA | **6** | **24** |
| private | **~184 B** | **0** |
| SPILL | **14** | **0** |
| decode (WAVES=8) | **~141 µs** | **~156 µs** |

Decode: DIRECT **beats** HIP. Prefill still ~**2×** on best-case.

Soak: continuous on tip; serial 13/13 when healthy. After MMU experiments, recover with fresh process + gemm smoke.

## Landed (stable)

- Decode ml_lds + **default `AMD_FLASH_WAVES=8`**
- Addr remat depth-1; const MOV remat (SPILL 16→14)
- Factor K_UNROLL scope; half×16 LSTORE→B128; deep LDS peel
- Soft-fuse / ACC_SEP / ACC_SMALL defaults (do not flip)

## Spill anatomy (prefill 32×2048)

**7 / 14** SPILL: causal `q_idx` `ADD(ADD(SHL,MOV),C…)`. Rest: LDS-stride MUL / FILL / nested addr.

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

## Next

1. Keep soak; fresh process after any MMU.
2. Prefill: **WMMA 6→24** only with a real spill/ACC plan (K unroll without SPILL explosion).
3. Causal/MUL spills: remat variants exhausted for now — need allocator/parking idea.
4. Fork-only; no DIRECT default flip.
