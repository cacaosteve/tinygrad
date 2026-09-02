# flash_decode_partial: HIP gap follow-up

**Status @ 9737080a8 (7900 XTX):** decode e2e ~54.7 µs AMD:AMD vs ~52.7 µs HIP (~2 µs gap).
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

Emit-time VMEM before swizzle (`AMD_SINK_VMEM_SWIZZLE`, fb031f1a2): neutral on HW.

## TODO (when resuming)

1. **Side-by-side disasm:** `flash_decode_partial` @ `DEV=AMD:AMD` vs `DEV=AMD:HIP` on 7900 —
   diff wait placement around score `warp_reduce` loops (not DPP-first; focus lgkmcnt + inst order).
2. **Mimic LLVM schedule in `insts_from_linear`:** defer lgkm flush on swizzle consumers when
   intervening ops don't read swizzle dest; validate WMMA/LLOAD tests.
3. **Kernel option (larger):** fewer warp reduces in `_amd_flash_attention_decode_partial` (head/key
   restructuring) if scheduling alone can't close ~17 µs isolated gap.
4. **Do not repeat without new evidence:** DPP row_shl reduces (HW regression); pre-regalloc VMEM
   sink (regalloc break); soft lgkm without WMMA-safe hard flush.

## Bench commands (7900)

```bash
PY=/home/admin441766/tinygrad-gabriel-16668-latest/.venv/bin/python
cd /home/admin441766/tinygrad-rdna3-ac0ba6fca && git pull  # codex/rdna3-perf-coverage

DEV=AMD:AMD  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:HIP  PYTHONPATH=. $PY extra/bench_amd_attention.py --case decode
DEV=AMD:AMD  PYTHONPATH=.:extra $PY extra/bench_amd_attention_kernels.py
```
