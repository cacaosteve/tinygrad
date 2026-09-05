# Overnight RDNA3 — tip `b85508fcf`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**700–710µs** (correct, err~1e-4). HIP ~278; SDPA ~360 (still faster default).

Parking is scoped: `flash_attention` realizes under `AMD_FLASH_ACC_SMALL=1` then clears it.
Do **not** park from `AMD_FLASH_DIRECT` alone (broke peer GEMM/eye) or auto-detect ≤64 multi-packs.

See `extra/rdna3_flash_acc_small.md` for scratch attribution.

## Scorecard (7900, this session)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT ACC_SMALL | ~705 | ~278 |
| Flash DIRECT ACC_SMALL=0 | ~1315 | |
| SDPA (no DIRECT) | ~360 | (HIP uses flash) |
| Serial correctness | 13/13 OK | flash/GQA/eye/post-flash GEMM |

## Cleanup landed

1. Removed ≤64 multi-pack auto-park (eye PACK SPILL under REDEF).
2. Scoped ACC_SMALL env to flash realize only (`b85508fcf`).
3. Local `test_amd_renderer` + `test_llm_amd`: 210 passed.
4. `extra/rdna3_serial_correctness.py` on 7900.

## Scratch attribution (ACC_SMALL)

- 21 allocator SPILL (VGPR pressure from ACC)
- 128/128 SLOAD/SSTORE soft-copy (dominant private traffic)
- priv 212 B; machine scratch ~137 st / 102 ld
- ACC_SMALL=0: more soft-copy (192) + priv 384 → slower despite fewer spills

## Next leftovers

1. Structural: cut soft-copy or ACC pressure (not new knobs) — toward HIP ~278
2. IQ4 vgpr 118→95
3. Decode partial 34→29
4. Stronger quant correctness than random ggml bytes (use packed-from-float)
