# PR title

```
amd: RDNA3 (gfx11) assembly backend via ISARenderer
```

# PR description

RDNA3 (`gfx1100`) asm backend on `ISARenderer`, `DEV=AMD:AMD`. LLVM/HIP stays default at `DEV=AMD`.

`tinygrad/renderer/isa/amd.py` is the renderer class; isel+encode live in `tinygrad/runtime/autogen/amd/rdna3/amd_lib.py` next to XML `ins.py`. f32 float4 memory via b128 loads/stores + coalescer guards for padded/gated indices. RDNA3 WMMA (`v_wmma_f32_16x16x16_f16`) for half→float tensor cores, with a two-address accumulator and PACK_F16 early-clobber handled in regalloc. Shared codegen: wide VGPR regalloc, regalloc PC sync, late i64 decomp after transcendental lowering. ISA post-TC upcast tile is capped (fixed VGPR/scratch budget).

Peak half GEMM (~60–71k GFLOPS @4096) lives as an **explicit** experiment in `extra/gemm/rdna3_asm_wmma_gemm.py` (same pattern as CDNA hand GEMM) — not wired into `Tensor.matmul`.

Mock infra (not the backend itself): half log2/sqrt/exp2 in `test/mockgpu/amd/pcode.py` evaluate via f32 so gfx950 emu avoids a clang-18 SIGSEGV on dense half soft-float.

```bash
DEV=AMD:AMD python3 -m unittest test.amd.test_amd_renderer -q
DEV=MOCKKFD+AMD:AMD python3 -m unittest test.amd.test_amd_renderer -q
# hand WMMA bench (optional):
DEV=AMD:AMD python extra/gemm/rdna3_asm_wmma_gemm.py
```

Validated on gfx1100 (7900 XTX): `test_amd_renderer` **112**, `test_tiny` 19, `test.backend.test_ops` 417 passed, `test.opt.test_tensor_cores` 6 passed.

f32 GEMM (float4/FMA): ASM ~LLVM at N=2048 (2691 vs 2674 GFLOPS).

Out of scope: bf16/half float4, pre-gfx11/CDNA.
