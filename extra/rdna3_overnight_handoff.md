# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`5ecac324c`** / remat SHALLOW **`40e16ec0a`**; decode waves **`e1fea66f8`**.

## Headline

Flash DIRECT **opt-in**. Fair fresh-process (serial-style 10×inner, 7900, 2026-09-08):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~666 µs** | **~361 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode (1×call TinyJit) | **~141 µs** @WAVES=8 | **~156 µs** |

Decode: DIRECT **beats** HIP on 32h/2k with default WAVES=8. Prefill still ~1.85×.

Soak: continuous on tip (serial+fixed multi-shape). Restart after MMU/dirty GPU with a fresh process + `nohup /tmp/rdna3_soak/run_continuous.sh`.

## Landed

- Decode ml_lds `(WAVES,2,G)` — 16-wave safe; **default `AMD_FLASH_WAVES=8`** (~141 vs ~152@16; WAVES=4 fails decode_gqa). Serial+fixed OK.
- Addr remat depth-1 on flash realize
- Const MOV remat — SPILL 16→14, priv 192→184
- Factor K_UNROLL scope via `AMD_FLASH_K_HIP_SCOPE`
- half×16 LSTORE → two B128 (isel parity with LLOAD)
- Deep peel for LDS LLOAD/LSTORE (`_peel_add_imm(..., deep=True)`) — correct, ~neutral latency

## Spill anatomy (prefill 32×2048)

Of **14** SPILLs, **7** are `ADD(ADD(SHL,MOV), CAST(C2016..2028))` — causal `q_idx`. Remaining: LDS-stride MUL (`*132`/`*1056`/`*2112`), FILL/nested addr.

## Dead ends this loop

- `MUL` in pure-addr remat: wrong @depth1, MMU @depth2
- Sparse MUL→SHL+ADD expand: correct, slower
- Power-of-2 LDS stride (132→256): SPILL 14→45
- Full-K / `K_UNROLL=-1` QK: hang; PV-only correct but slower
- `K_UNROLL=1` full chain: **MMU fault**
- `_addr_leaf` SHL+ sticky remat: SPILL 14→5 but ~725→~950 µs
- `CMPLT(ADD(x,c),y)` fold (general): SPILL 14→25
- `K_UNROLL=2` (pv/all): correct, slower; SPILL 56
- `K_UNROLL=4` pv: tied with baseline; `all`/`qk` nan
- Slot-2 promote: correct, slightly slower
- Soft toggles / shared `q_row` CSE / unsigned causal adj: dead
- Remat deepen CMP-only (`AMD_REMAT_CMP_DEEP`): SPILL 14→7, wrong numerics
- `AMD_WHERE_ALIAS=1` global: breaks flash + eye (finite-but-wrong); do not enable
- Causal `AMD_FLASH_CAUSAL=2` adj `(k-qb)<=q_local`: GQA FAIL; `=1` signed cast OK but no win
- Decode WAVES=4: FAIL decode_gqa
- `AMD_REMAT_NO_STICKY_ADD=1`: hang/MMU risk — leave off

## In flight

- `AMD_REMAT_ADDR_SHALLOW=1`: treats SHL/ADD-of-leaves as leaf-equivalent under DEEP=1 → **SPILL 14→5**, TinyJit looked ~651 vs ~663, but one serial run showed ~910 (dirty GPU?). Needs clean re-validate then enable on flash realize. Gaming PC unreachable mid-test after NO_STICKY hang.

## Next

1. Reconnect GPU; recover after MMU; clean rebench SHALLOW; if stable win, set on flash realize.
2. Cut remaining 5 spills (MUL/FILL) without promoting slot 2.
3. More WMMA only with spill plan (`K_UNROLL=1` MMU).
4. Do not flip DIRECT default.
