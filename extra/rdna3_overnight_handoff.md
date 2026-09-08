# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`55e312414`**.

## Headline

Flash DIRECT **opt-in**. Fair fresh-process gap (7900, 2026-09-08):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~700 µs** | **~415 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode | ~57 µs | ~48 µs |

Soak: continuous on tip (serial+fixed multi-shape).

## Landed

- Decode ml_lds `(WAVES,2,G)` — 16-wave + 32k
- Addr remat depth-1 on flash realize
- Const MOV remat — SPILL 16→14, priv 192→184
- Factor K_UNROLL scope via `AMD_FLASH_K_HIP_SCOPE`
- half×16 LSTORE → two B128 (isel parity with LLOAD)
- Deep peel for LDS LLOAD/LSTORE (`_peel_add_imm(..., deep=True)`) — correct, ~neutral latency

## Dead ends this loop

- `MUL` in pure-addr remat: wrong @depth1, MMU @depth2
- Sparse MUL→SHL+ADD expand: correct, slower (~913→~932)
- Power-of-2 LDS stride (132→256): SPILL 14→45
- Full-K / `K_UNROLL=-1` QK: hang; PV-only correct but slower
- `K_UNROLL=1` full chain: **MMU fault** on tip (do not enable)
- `_addr_leaf` SHL+ sticky remat: SPILL 14→5 but ~725→~950 µs
- `_addr_leaf` SHL + `keep_remat=False` on const-offset ADD: SPILL 14→5, ~neutral/slightly slower
- `CMPLT(ADD(x,c),y)` fold: SPILL 14→25 (more pressure)
- `K_UNROLL=4` pv: tied with baseline; `all`/`qk` nan
- Slot-2 promote (`SKIP_SLOTS=`): correct, still slightly slower
- SSTORE byte_off clustering: sorts offs but values are same VGPR (redef/init) — no b128 win
- `s_delay_alu` after WMMA/swizzle: ~neutral (not default)
- MOV-chain fold: inner MOVs are fan-out (32–96 uses), not single-use chains
- `AMD_WMMA_ACC_BASE` default 201: same SPILL 14 / k2 SPILL 79 as 121 — leave 121

## Next

1. Keep soak running.
2. Cut remaining non-cmp spills (MUL/FILL addr) or shrink slot-2 scratch without promoting slot 2.
3. Path to more WMMA only with spill plan (`K_UNROLL=1` now MMU — needs new approach).
4. Do not flip DIRECT default.
