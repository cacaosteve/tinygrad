# flash_decode_partial: HIP gap follow-up

**Status @ e39b40c22 (7900 XTX):** decode e2e ~54.4 µs AMD:AMD vs ~52.5 µs HIP (~2 µs gap).
`flash_decode_partial` ~45 µs direct vs ~28 µs HIP isolated — main remaining bottleneck.
`flash_decode_combine` already faster than HIP (~9 vs ~11.5 µs).

## Does HIP do DPP / different reduce?

**Same kernel source.** Both paths use `warp_reduce()` in `tinygrad/llm/kernels/amd.py`, which emits
`__builtin_amdgcn_ds_swizzle` CUSTOM ops (butterfly offsets 16,8,4,2,1). HIP compiles those via
LLVM/comgr; direct ISA lowers them in `tinygrad/renderer/isa/rdna3.py`.

We tried replacing row-local swizzle stages with **V_ADD DPP row_shl** on direct ISA only — codegen
had far fewer lgkm waits, but **HW regressed ~+3.5 µs on partial**. So HIP is **not** winning
because it uses DPP for these reduces (likely still ds_swizzle in LLVM output).

## Why is direct ISA slower on partial?

Machine code (partial kernel, ~1558 insts each):

| | AMD:AMD direct | AMD:HIP |
|--|----------------|---------|
| `DS_SWIZZLE_B32` | ~160 | ~160 (same algorithm) |
| Global loads | ~20 | ~20 |
| **`S_WAITCNT_LGKMCNT`** | **~202** | **~150** |

Same swizzles, **~52 extra lgkm hard-waits** on direct. Our emitter calls `flush_regs` before
most ALU that consumes a swizzle dest → `lgkmcnt(0)` every butterfly stage. LLVM tends to
**schedule independent VMEM/ALU in the swizzle latency gap** and may use **`s_delay_alu`** /
tighter wait counts — we have not fully diff'd HIP asm yet.

Soft lgkm scoreboard (defer waits until dest read) was tried twice: correct with DS_SWIZZLE in
scoreboard, but almost all waits stayed hard `lgkmcnt(0)` — no e2e win.

Emit-time VMEM before swizzle (`fb031f1a2`): neutral on HW.

Emit-time VMEM **after** swizzle (`e39b40c22`): LLVM-style lgkm gap placement; partial still
~45 µs (neutral), e2e slightly improved (~54.4 µs). ALU in gap (`08f7d875a`) reverted — e2e regression.

## Disasm findings (7900 XTX, `2d3957def`)

Tool: `extra/diff_flash_decode_partial_asm.py` — run on HW with `DEV=AMD:AMD` and `DEV=AMD:HIP`.

| Metric | Direct ISA | HIP |
|--------|------------|-----|
| `ds_swizzle_b32` | 160 | 160 |
| Total waitcnt | 202 | 150 |
| **`s_delay_alu`** | **0** | **108** |
| Global loads | 20 | 20 |
| Swizzles with **0 ops before lgkm wait** | **158 / 160** | **59 / 160** |
| Swizzles with VALU before wait | 1.2% | **27.5%** |
| Median insts swizzle → wait | 1 | 2 |
| **Swizzle streak before wait** | **always 1** | **1–17** (often batched) |

**Conclusion:** HIP does **not** use a different reduce algorithm (same 160 swizzles). It wins by
**software-pipelining independent butterflies**: many `ds_swizzle` then one `s_waitcnt`, plus
`s_delay_alu`. Direct ISA emits **`swizzle → lgkm wait → add`** per stage (streak always 1).

**Tried (`9b4a96af5`, reverted):** `warp_reduce_many` + swizzle-breadth schedule + emit-time batching.
Locally got some long streaks, but **VGPR spills** → HW partial **45 → 52 µs**. Need a
spill-safe way to batch same-stage swizzles (smaller batches / better regalloc) without
holding all lane-dots live.

**Also tried:** emit-time VALU/ADD gap sink (GPU hang); pre-regalloc MUL reorder (regalloc assert);
`AMD_SWIZZLE_DELAY=1` (safe, neutral).

```bash
DEV=AMD:AMD  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py -o /tmp/direct.json
DEV=AMD:HIP  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py -o /tmp/hip.json
PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py --compare /tmp/direct.json /tmp/hip.json
```

## TODO (when resuming)

1. ~~**Side-by-side disasm**~~ — done; see above.
2. **Spill-safe butterfly batching:** software-pipeline same-stage `ds_swizzle` in groups of ~4–8
   without keeping all SEC×G lane-dots live; validate no SCRATCH ops; HW A/B partial.
3. Kernel option: change score loop structure so LLVM-style batching falls out of linearization
   with low register pressure.

## Bench commands (7900)

```bash
PY=/home/admin441766/tinygrad-gabriel-16668-latest/.venv/bin/python
cd /home/admin441766/tinygrad-rdna3-ac0ba6fca && git pull  # codex/rdna3-perf-coverage

DEV=AMD:AMD  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:HIP  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:AMD  PYTHONPATH=.:extra $PY extra/bench_amd_attention_kernels.py
```
