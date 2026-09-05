# flash_decode_partial: HIP gap follow-up

**Status (7900 XTX):** decode e2e ~**49 µs** AMD:AMD vs ~**49.5 µs** HIP (parity / slight AMD lead).
`flash_decode_partial` ~**34 µs** direct (score_batch default **16**) vs ~28 µs HIP.
`flash_decode_combine` already faster than HIP (~9.4 vs ~11.5 µs).

## Landed: spill-safe same-stage swizzle batching

`warp_reduce_many` parks each stage’s swizzles in unique REG temps (per-element stores),
and `_schedule_swizzle_mov_batches` rewrites `SW,MOV,…` → `SW×N,MOV×N` **before regalloc**.
Score loop reduces **two keys at once** (batch of 8 = 2×G) when `SEC` allows.

HW: partial **45 → 35.3 µs**, checksum/first8 match tip, 0 scratch.
Toggle: `AMD_BATCH_SWIZZLE_MOV=0`.

**Do not** emit-reorder SW/MOV after regalloc (VGPR alias → wrong numerics).
**Do not** elide park MOVs by rewriting AFTER/store edges (broke decode checksum).

## Does HIP do DPP / different reduce?

**Same kernel source.** Both paths use `warp_reduce` / `warp_reduce_many` in
`tinygrad/llm/kernels/amd.py` (`ds_swizzle` CUSTOM). DPP `row_shl` was tried on direct ISA —
codegen looked better but **HW regressed ~+3.5 µs**; reverted.

## Why is direct ISA still slower on partial?

HIP still wins some on scheduling (`s_delay_alu`, longer streaks up to ~17, more VALU in the
lgkm gap). Direct now batches G=4 per stage; HIP often batches more aggressively across work.

## Failed approaches (do not repeat blindly)

- Global swizzle-breadth ready-list (FMAC hoist / uncapped) → spills or VMEM overlap loss
- Emit-time SWIZZLE/MOV reorder after regalloc → **wrong numerics** (VGPR alias)
- Soft lgkm alone without park → streak always 1
- VALU/ADD gap sink → GPU hang; `AMD_SWIZZLE_DELAY=1` → neutral on park path
- `AMD_SWIZZLE_NO_PARK=1` (direct SW→ADD, HIP-like wait→add): correct with stage-safe
  batching, but **partial ~46 µs** (worse than park ~34). Leave off.

```bash
DEV=AMD:AMD  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py -o /tmp/direct.json
DEV=AMD:HIP  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py -o /tmp/hip.json
PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py --compare /tmp/direct.json /tmp/hip.json
```

## TODO (when resuming)

1. Close remaining ~9 µs isolated partial gap vs HIP (`s_delay_alu` placement, longer batches
   without spills, more VALU/VMEM in swizzle latency gaps).
2. Re-check e2e vs HIP after any further partial work (~1.4 µs left).

## Bench commands (7900)

```bash
PY=/home/admin441766/tinygrad-gabriel-16668-latest/.venv/bin/python
cd /home/admin441766/tinygrad-rdna3-ac0ba6fca && git pull  # codex/rdna3-perf-coverage

DEV=AMD:AMD  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:HIP  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:AMD  PYTHONPATH=.:extra $PY extra/bench_amd_attention_kernels.py
```
