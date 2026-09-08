# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`97c38feb4`** (docs) / **`b63240d51`** (ml_lds G==SEC fix; decode 16-wave + 32k).

## Headline

**Decode DIRECT fixed at 16 waves + long context.** Prefill still opt-in
(`AMD_FLASH_DIRECT=1`). Fair hand-kernel compare (fresh process each backend,
2026-09-08 remeasure):

| | DIRECT | HIP (same hand kernel) |
|--|--:|--:|
| prefill median | **~730 µs** | **~596 µs** |
| machine WMMA | **6** | **24** |
| private_segment | **212 B** | **0** |
| `s_delay_alu` | **0** | **126** |
| `v_mov` | **~385** | **~28** |
| decode median | ~56 µs | ~51 µs |

Gap is ~1.2× on this card/clock (earlier ~2× numbers were hotter HIP / colder DIRECT).
Do **not** flip DIRECT default until prefill ≤ SDPA and soak is solid.

## Decode ml_lds fix (landed)

**Bug:** `ml_lds (WAVES, G, 2)` aliases when `G==SEC` (16-wave GQA). Stats `L`
undercounts on AMDRenderer; partials look fine → combine scales wrong.
**Fix:** layout `ml_lds` as `(WAVES, 2, G)`. Works through **32k** kv.

## Prefill gap — status 2026-09-08 (continued)

HIP’s 24 WMMA = full K-unroll of the same tile math:
`QK: TN×(TM/ACC)×(D/K) = 2×1×8 = 16` + `PV: TD×1×(BN/K) = 4×1×2 = 8`.

DIRECT ACC_SMALL keeps K ranged → **6** static WMMA. Only **6** ACC packs are
parked (`v126..v166` or raised band); pool still reserves `VGPR[121:]` (~128).

### Tried this loop

| Lever | Result |
|--|--|
| `AMD_FLASH_K_UNROLL=2` | **Correct** (err 0). WMMA 12, SPILL 87, priv 492, **slower** |
| `K_UNROLL=4` | wrong (~0.8 err) |
| `K_UNROLL=1` / full chain both | MMU / hang — even with `AMD_INSTR_WAIT=1` |
| HIP store-each-K (`K_UNROLL=-1` old) | Compiles 24 WMMA; **hangs**. INSTR_WAIT still hangs → **not** soft waitcnt-only |
| Chained PV-only (`K_UNROLL=-1 HIP_SCOPE=pv`) | **Correct** (err 0), WMMA 10, SPILL 65, **slower** than base |
| Chained QK-only (`HIP_SCOPE=qk`) | SPILL ~121 → **hangs** (small + full) |
| `AMD_SPILL_DRAIN_LGKM=1/2` | Opt-in drain before SPILL/FILL; does **not** make full-K safe |
| Soft levers (`SOFT_SCALE=0`, `FUSE=0`, `VEC_COPY`, `ACC_WORK=0`, promote slot 2) | All **slower** than default soft-fuse |
| `AMD_REMAT_ADDR=1` | no spill cut; slower |
| `AMD_WMMA_ACC_BASE=201` | ACC moves to `v206..`; still same SPILL count; no QK-hang fix |
| Flash addr remat depth-1 + MOV-const | **Landed for DIRECT flash realize**: SPILL **21→16**, priv **212→192**. Correct (serial 13/13). |

**Takeaway:** more WMMA without a spill/occupancy plan loses. QK full-K hang survives
full instr waits → suspect address/scratch pressure or schedule, not missing lgkm alone.
PV-only expand is a correct stepping stone but not a win until SPILL drops.

Base still has **16 allocator SPILLs** (mostly nested addr ADD/MUL) + slot-2 scratch
(SLOAD/SSTORE 96). Soft/stats REG already promote. Remat depth≥2 still MMUs.

### Bench hygiene

Never measure HIP after DIRECT in the **same** Python process — `Device["AMD"]`
stays on the first renderer. Use separate processes. Warm TinyJit ≥3× before
`program_metrics` (need `captured`).

### Opt-in experiment envs (leave off for soak)

- `AMD_FLASH_K_UNROLL=-1` + `AMD_FLASH_K_HIP_SCOPE=pv|qk|all` — chained K expand
- `AMD_SPILL_DRAIN_LGKM=1` (lgkm+vs) / `=2` (vm+lgkm+vs) before SPILL; lgkm before FILL
- `AMD_WMMA_ACC_BASE=201` — shrink ACC band to 48 VGPRs (import-time; set before process)
- Flash realize now sets `AMD_REMAT_ADDR=1` + `AMD_REMAT_ADDR_DEEP=1` (depth-1; override to 0 to opt out)

### Next leftovers

1. Continuous soak on tip (default env).
2. Prefill: make QK full-K (or 24 WMMA) **correct** — need spill/address fix beyond
   waitcnt; or cut remaining 16 SPILLs / `v_mov` / slot-2 scratch toward HIP’s 0 private.
3. Optional: remaining ~5 µs decode (gloads 20 vs 12; FMA_MIX still unsafe).
4. Do not flip DIRECT default yet.

```bash
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed \
  --shapes 128:128,256:64,512:32,2048:32 --replays 500
```
