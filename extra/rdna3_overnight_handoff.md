# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: see latest push (ml_lds G==SEC fix).

## Headline

**Decode DIRECT fixed at 16 waves + long context.** Prefill still opt-in
(`AMD_FLASH_DIRECT=1`), ~2.4× behind HIP (~665 vs ~278 µs). Packing /
FMA_MIX / K-unroll stay off for prefill.

## Decode ml_lds fix (2026-09-08)

**Bug:** `ml_lds (WAVES, G, 2)` aliases when `G==SEC` (16-wave GQA). Stats `L`
undercounts on AMDRenderer; partials look fine → combine scales wrong.
HIP unaffected. Misdiagnosed earlier as #18010 REG / waves=8-only.

**Fix:** layout `ml_lds` as `(WAVES, 2, G)`. DIRECT uses upstream #18010
online-softmax path at 16 waves; works through **32k** kv.

| | DIRECT | HIP |
|--|--:|--:|
| decode median | ~56 µs | ~51 µs |
| decode err | ~7e-5 | ~7e-5 |
| prefill median | ~665 µs | ~278 µs |

Soak: 43+ clean rounds on prior tip; restart after this fix.

## Prefill gap (next)

DIRECT: priv 212, 21 spills, 385 `v_mov`. HIP: priv 0, `s_delay_alu`.
Existing toggles (FMA_MIX, K_UNROLL, VEC_COPY) do not close the gap.

## Next

1. Continuous soak on tip.
2. Prefill scratch / spill reduction (main remaining gap).
3. Optional: close remaining ~5 µs decode (gloads 20 vs 12, FMA_MIX unsafe).
4. Do not flip DIRECT default yet.

```bash
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed \
  --shapes 128:128,256:64,512:32,2048:32 --replays 500
```
