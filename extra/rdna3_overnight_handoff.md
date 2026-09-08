# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`7c7cdcae1`** (DIRECT decode: legacy partial @ 8 waves).

## Headline

**QK LDS reuse race fixed** (prefill). Flash DIRECT still **opt-in**
(`AMD_FLASH_DIRECT=1`). Prefill packing / FMA_MIX / K-unroll stay off.
HIP prefill still ~2.4–3× faster (~278 µs vs DIRECT ~675–910 µs).

## Decode DIRECT (2026-09-07)

After merging `#18010` (online-softmax past 16k), **DIRECT decode_gqa failed**
(err~1.18). HIP OK. Root cause:

1. `#18010` REG `sum_reg` undercounts on AMDRenderer (partials cancel via
   normalize; stats L wrong → combine scales).
2. **waves=16** breaks DIRECT LDS exchange even on pre-#18010 partial;
   **waves=8** is correct.

**Fix:** DIRECT uses pre-#18010 partial + default 8 waves; SDPA if
`max_kv_len > chunks*64`. HIP keeps `#18010` / 16-wave path.

Serial on 7900: **13/13 OK** after fix.

## Prefill validation (earlier)

61 clean multi-shape soak rounds then SSH disconnect. Re-soak on tip after
decode fix.

## Next

1. Continuous soak on tip (prefill + serial).
2. Re-bench prefill/decode vs HIP.
3. Perf leftovers only after soak confidence.
4. Do not flip DIRECT default yet.

```bash
AMD_FLASH_DIRECT=1 PYTHONPATH=.:extra python extra/rdna3_serial_correctness.py
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed \
  --shapes 128:128,256:64,512:32,2048:32 --replays 500
```
