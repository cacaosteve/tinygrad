# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`25a1bbf31`**.

## Headline — correctness first

**Do not treat ~685µs Flash DIRECT ACC_SMALL as validated-correct.** Normal
prefill still defaults to **SDPA** unless `AMD_FLASH_DIRECT=1`. Packing /
FMA_MIX / eviction stay **off**. Flash DIRECT remains blocked.

## GPU diag (100 replays, S=T=128, SKIP=2 work=1) — `25a1bbf31`

Same `_amd_flash_attention` graph; `DEV=AMD:HIP` vs `DEV=AMD:AMD`.

| config | renderer | 100× frozen exact | vs HIP (1st out) | verdict |
|--------|----------|-------------------|------------------|---------|
| hip | HIPRenderer | **yes** | (ref) | **ok** |
| direct | AMDRenderer | **no** (~8e-3) | ~1.8e-7 | launch_nondeterminism |
| inorder | AMDRenderer | **no** (~4e-3) | ~1.8e-7 | launch_nondeterminism |
| inorder_instr | AMDRenderer | **no** (~2e-2) | ~1.8e-7 | launch_nondeterminism |

**Read:** shared kernel/race is unlikely — HIP same-graph is bit-stable. Optional
post-alloc motion + instr-level waits do **not** stabilize DIRECT. First DIRECT
out is nearly HIP-exact (~ulp); later launches diverge → backend launch defect.

Harness: `extra/rdna3_flash_backend_diag.py` (DEV ContextVar selects renderer).

## Phys-guard stage fix (landed)

Pre-regalloc must use `_batch_scratch_load_uses(..., check_phys=False)` —
virtual regs share placeholder index `0`. Tests cover virt batch + phys refuse.

## Emit knobs (still useful for bisect)

- `AMD_IN_ORDER_EMIT=1` — disables optional post-alloc motion/fusions (VOPD,
  WMMA hoist/sink, clauses, fused loops, SINK_VMEM, WHERE-load fuse). ELF
  differs from shipping (confirmed). Does not fix nondeterminism alone.
- `AMD_INSTR_WAIT=1` — hard waitcnt after each tracked mem burst. Also does not
  stabilize. Missing soft-wait bookkeeping remains possible for *untracked* ops,
  but draining tracked domains is insufficient.

## Next leftovers

1. **Instrument intermediate kernel phases** for first incorrect value (before
   per-slot scratch/ACC hunts) — in-order + instr did not stabilize
2. Pre-regalloc optional passes as a separate bisect group (emit-side already
   cleared)
3. Decode only after DIRECT freeze replay is bit-exact vs HIP
4. eye/GEMM TC_LDS_AB
