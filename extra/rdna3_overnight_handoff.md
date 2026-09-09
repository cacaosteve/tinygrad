# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`fcd65ce71`** (docs); code: decode waves **`e1fea66f8`**, NO_STICKY_CONST opt-in **`bdbc7cbbd`**, SHALLOW **`40e16ec0a`** (do **not** enable).

## Headline

Flash DIRECT **opt-in**. Fair fresh-process (7900, recovered 2026-09-09):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~685–720 µs** | **~330 µs** |
| WMMA | **6** | **24** |
| private | **184 B** | **0** |
| SPILL | **14** | **0** |
| decode (WAVES=8) | **~141 µs** | **~156 µs** |

Decode: DIRECT **beats** HIP. Prefill ~**2.1×** behind (HIP remeasured ~330).

Soak: continuous on tip after host reboot. GPU OK: Navi 31, `/dev/kfd`, serial 13/13.

## Landed

- Decode ml_lds + **default `AMD_FLASH_WAVES=8`**
- Addr remat depth-1; const MOV remat (SPILL 16→14)
- Factor K_UNROLL scope; half×16 LSTORE→B128; deep LDS peel

## Spill anatomy (prefill 32×2048)

**7 / 14** SPILL: causal `q_idx` `ADD(ADD(SHL,MOV),C2016..)`. Rest: LDS-stride MUL / FILL / nested addr.

## Dead ends this loop

- SHALLOW remat (`AMD_REMAT_ADDR_SHALLOW=1`): SPILL 14→5 but **prefill_gqa_32 FAIL** (err~1.2) and **slower** (~940 vs ~685) — leave off
- `NO_STICKY_CONST_ADD` alone: no spill/latency win; combined with SHALLOW still wrong
- `NO_STICKY_ADD`: hang/MMU — leave off
- Soft toggles / ACC_WORK=0 / VEC_COPY / K_UNROLL=2 pv / FLASH_UNROLL 1–7: all ≥ baseline
- Prior: K_UNROLL=1 MMU; cmp-deep remat; causal unsigned rewrite; q_row CSE; etc.

## Next

1. Keep soak; after any MMU recover with fresh process.
2. Prefill: WMMA 6→24 needs new ACC/spill plan (not more unroll toggles).
3. Cut causal/MUL spills without broken remat (emit-time cmp imm, or leaf-only remat fix).
4. Do not flip DIRECT default; do not enable SHALLOW.
