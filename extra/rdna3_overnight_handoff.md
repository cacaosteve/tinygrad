# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **(pending push)**.

## Headline — correctness first

**Do not treat ~685µs Flash DIRECT ACC_SMALL as validated-correct.** Frozen-ELF
replay still showed **launch nondeterminism** under shipping SKIP=2. Normal
prefill still defaults to **SDPA** unless `AMD_FLASH_DIRECT=1`. Packing /
FMA_MIX / eviction stay **off**. Flash DIRECT remains blocked until freeze
replay is bit-exact vs a reference.

## Phys-guard stage fix (this tip)

`_scratch_batch_phys_conflict` must **not** run pre-regalloc: virtual regs share
placeholder index `0`, so distinct SSA values look overlapping and suppress
valid batching (batch-on/off identical ELF was a scheduling regression, not
“batching never matters”). Pre-regalloc uses `_batch_scratch_load_uses(...,
check_phys=False)`; phys checks remain for any post-alloc caller.

## Emit diagnostics (new)

- `AMD_IN_ORDER_EMIT=1` — disable optional post-alloc motion/fusions together
  (VOPD FMAC/ADD/MOV, WMMA hoist/sink, emit store/load clauses, fused loops,
  `AMD_SINK_VMEM_SWIZZLE`, WHERE-load exec fuse). Required lowering (d16 order,
  waits) stays.
- `AMD_INSTR_WAIT=1` — hard `waitcnt(0)` after each tracked mem burst (does not
  rely on soft pending bookkeeping). `AMD_CONSERVATIVE_WAIT` still exists and
  remains insufficient alone on the prior path.

Harness: `extra/rdna3_flash_backend_diag.py` — same `_amd_flash_attention` graph
through **HIP** (`AMD:HIP`) vs DIRECT emit configs; default **100 replays** +
reference compare (stable-but-wrong fails).

```bash
PYTHONPATH=.:extra python extra/rdna3_flash_backend_diag.py \
  --configs hip,direct,inorder,inorder_instr --replays 100 --S 128
```

## Prior leftovers

1. If in-order(+instr) stabilizes → re-enable transform groups to isolate culprit
2. If neither stabilizes → instrument intermediate kernel phases for first wrong value
3. Decode only after DIRECT freeze replay is bit-exact
4. eye/GEMM TC_LDS_AB
