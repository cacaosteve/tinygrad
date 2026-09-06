# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: refresh after next commit.

## Headline — correctness first

**Do not treat ~685µs Flash DIRECT ACC_SMALL as validated-correct.** Frozen-ELF
replay still shows **launch nondeterminism** under shipping SKIP=2 (fails under
12–16 replays). Normal DIRECT prefill still defaults to **SDPA** unless
`AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / eviction stay **off**.

## Post-regalloc hazard (landed)

Emit-time `_batch_scratch_load_uses` was UOp-only after phys regalloc — VGPR reuse
makes SSA-independent pairs unsafe. **Removed post-alloc batching**, emit-time
swizzle batch / gap-fill / LLOAD hoist (moved LLOAD hoist to `after_pre_regalloc`).
Added phys-overlap guard + regression test.

`AMD_CONSERVATIVE_WAIT=1` (full vm/lgkm/vs drain on `flush_regs` / after `note_vm`)
**does not** stabilize freeze replay — not a trivial soft-wait miss on that path.

## Freeze/replay (after post-alloc batch removal)

| config | unique ELF | 12+ replay |
|--------|------------|------------|
| skip2 (defaults) | 1 | still **diverges** (earlier 8-replay pass was luck) |
| skip2_nobatch | 1 (same ELF as skip2 now) | diverges |
| prom | 1 | diverges |

Device q/kv readbacks match; outs are not all-sentinel. First-mismatch coords are
**flattened output order**, not first workgroup.

## Next leftovers

1. Remaining launch nondeterminism: uninit VGPR/scratch, other emit reorders
   (WMMA hoist/sink, d16, store cluster, `AMD_SINK_VMEM_SWIZZLE`), or algo race
2. Decode only after `AMD_FLASH_DIRECT=1` freeze replay is bit-exact
3. eye/GEMM TC_LDS_AB
