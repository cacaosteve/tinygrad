# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`37300f9bf`** (docs; code tip **`cfa1a2407`** deep LDS peel).

## Headline

Flash DIRECT **opt-in**. Fair fresh-process gap (7900, 2026-09-08):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~700 µs** | **~415 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode | ~57 µs | ~48 µs |

Soak: continuous on tip (serial+fixed multi-shape). Restart after MMU/dirty GPU with a fresh process + `nohup /tmp/rdna3_soak/run_continuous.sh`.

## Landed

- Decode ml_lds `(WAVES,2,G)` — 16-wave + 32k
- Addr remat depth-1 on flash realize
- Const MOV remat — SPILL 16→14, priv 192→184
- Factor K_UNROLL scope via `AMD_FLASH_K_HIP_SCOPE`
- half×16 LSTORE → two B128 (isel parity with LLOAD)
- Deep peel for LDS LLOAD/LSTORE (`_peel_add_imm(..., deep=True)`) — correct, ~neutral latency

## Spill anatomy (prefill 32×2048)

Of **14** SPILLs, **7** are `ADD(ADD(SHL,MOV), CAST(C2016..2028))` — causal `q_idx` with folded `q_base+rm`. Pre-RA each feeds **2× CMPLT** against the same two `k` trees; post-RA they often remain as SPILL-only (no FILL) after VCC/remat traffic.

Remaining 7: MUL/FILL/nested addr (not CMP-only).

## Dead ends this loop

- `MUL` in pure-addr remat: wrong @depth1, MMU @depth2
- Sparse MUL→SHL+ADD expand: correct, slower (~913→~932)
- Power-of-2 LDS stride (132→256): SPILL 14→45
- Full-K / `K_UNROLL=-1` QK: hang; PV-only correct but slower
- `K_UNROLL=1` full chain: **MMU fault** on tip (do not enable)
- `_addr_leaf` SHL+ sticky remat: SPILL 14→5 but ~725→~950 µs
- `_addr_leaf` SHL + `keep_remat=False` on const-offset ADD: SPILL 14→5, ~neutral/slightly slower
- `CMPLT(ADD(x,c),y)` fold (general): SPILL 14→25 (more pressure)
- `K_UNROLL=2` (pv/all): correct, slower (~959–1012 vs ~700); SPILL 56
- `K_UNROLL=4` pv: tied with baseline; `all`/`qk` nan
- Slot-2 promote (`SKIP_SLOTS=`): correct, still slightly slower
- SSTORE byte_off clustering: sorts offs but values are same VGPR (redef/init) — no b128 win
- `s_delay_alu` after WMMA/swizzle: ~neutral (not default)
- MOV-chain fold: inner MOVs are fan-out (32–96 uses), not single-use chains
- `AMD_WMMA_ACC_BASE` default 201: same SPILL 14 / k2 SPILL 79 as 121 — leave 121
- Soft toggles (`SOFT_SCALE=0`, `ACC_WORK=0`, `VEC_COPY=1`, `PV_ACC_DIRECT=1`, `ACC_SEP=0`): all slower than defaults
- Shared `q_row` CSE for causal: correct, **slower** (~696→~949)
- Causal `(k-q_base)<=q_local` / `k+(-q_base)`: breaks mask (unsigned wrap / wrong WHERE) — huge err
- Remat deepen for CMP-only addr (`AMD_REMAT_CMP_DEEP`): SPILL 14→7, ~neutral latency, but **prefill_gqa_32 err ~2.6e-2** (threshold 5e-3). Likely silent remat-eviction with stale MOV binder srcs
- Same + `AMD_SPILL_ON_EVICT=1`: SPILL→42 and **MMU fault** — do not combine
- Insert-new-ADD fold in `prepare_pre_regalloc`: CompileError (new INS never tagged)

## Next

1. Keep soak running; recover GPU with a fresh serial after any MMU.
2. Cut CMP q_idx spills **without** remat-from-stale-srcs: e.g. leaf-only remat (no MOV binders), or signed compare path that shares one `k` adjust without uint wrap, or teach `wide_alloc` not to silently drop non-leaf remats.
3. Cut remaining non-cmp spills (MUL/FILL) or shrink slot-2 scratch without promoting slot 2.
4. Path to more WMMA only with spill plan (`K_UNROLL=1` MMU — needs new approach).
5. Do not flip DIRECT default.
