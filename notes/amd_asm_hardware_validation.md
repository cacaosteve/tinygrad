# AMD ASM backend — hardware validation (real gfx1100 / RDNA3)

**Hardware:** AMD RX 7900 class, `arch=gfx1100`, Linux bare metal
**Backends compared:** `DEV=AMD:AMD` (ISA asm via `ISARenderer`) vs `DEV=AMD` (AMD LLVM/HIP) on the same box
**Mock dev (macOS/CI):** `DEV=MOCKKFD+AMD:AMD`

## Correctness (green on real hardware)

| Suite | Result |
|-------|--------|
| `examples/beautiful_mnist.py` | Train/eval OK, 98.24% test accuracy |
| `test.amd.test_amd_renderer` | **112** OK, 21 skipped |
| `test.test_tiny` | 19 OK, 2 skipped |
| `test.backend.test_ops` | **417 passed**, 6 skipped, 157 subtests before float4; latest full post-float4 sweep pending |
| `test.backend.test_ops` targeted post-float4 regressions | 8 passed, 415 deselected, 3 subtests passed |
| `test.test_tiny` + `test_uops` + `test_linearizer` + `test_jit` + `test_nn` + `test_optim` + `test_randomness` + `test_multitensor` | 235 OK, 168 skipped |
| `test.backend.test_dtype` + `test_dtype_alu` + `test_tensor` | 335 OK, 51 skipped, 1 expected failure |
| `test_uops` + `test_tensor_variable` + `test_symbolic_ops` + `test_symbolic_jit` (earlier run) | 124 OK, 8 skipped |

### Known skips / limitations
- `test_ops` extreme-input `sin`/`cos`/`tan` skipped on AMD asm (same precedent as NAK)
- int64/bfloat edge cases: mostly covered on mock; full `test_arange` MNIST path may still timeout on mock
- `t.realize()` wall-time microbenchmarks without `run_linear(..., wait=True)` report enqueue-only times (~0.02 ms) — not valid bandwidth numbers

## Speed — GEMM matmul (GFLOPS, higher is better)

### Hand WMMA experiment (explicit, not default matmul)

`extra/gemm/rdna3_asm_wmma_gemm.py` is a CDNA-style hand kernel for peak half GEMM on gfx1100. Call it explicitly; it is **not** hooked into `Tensor.matmul`.

```python
from extra.gemm.rdna3_asm_wmma_gemm import can_use_rdna3_wmma_gemm, rdna3_wmma_gemm
if can_use_rdna3_wmma_gemm(a, b): c = rdna3_wmma_gemm(a, b)
```

| N | Hand WMMA (ASM) | Notes |
|---:|---:|---|
| 4096 | ~60–71k GFLOPS | A/B LDS blocked; codegen-only `Tensor.matmul` uses capped TC tile |

Default `Tensor.matmul` on `DEV=AMD:AMD` uses codegen WMMA with the ISA TC upcast cap (no `extra` import).

### Current slice-3 float4 memory path

After enabling float4 memory coalescing, a padded conv case exposed a real hardware MMU fault: a `WHERE(valid, idx, Invalid)` input load could be coalesced into a wide `GLOBAL_LOAD_B128`, which reads all four addresses before any per-lane valid select can protect padded/OOB lanes. The fix is to leave valid-gated loads scalar while still allowing wide loads for ungated inputs such as GEMM.

Hardware validation on gfx1100:

```text
DEV=AMD:AMD python -m unittest test.backend.test_ops.TestOps.test_asymmetric_padding_conv2d -q
Ran 1 test in 6.560s
OK
```

Post-fix f32 GEMM parity check (codegen, not hand WMMA):

| N | ASM | LLVM |
|---:|---:|---:|
| 512 | 258 GFLOPS (1.04 ms/iter) | 261 GFLOPS (1.03 ms/iter) |
| 1024 | 1804 GFLOPS (1.19 ms/iter) | 1397 GFLOPS (1.54 ms/iter) |
| 2048 | 2691 GFLOPS (6.38 ms/iter) | 2674 GFLOPS (6.42 ms/iter) |

Takeaway: the asymmetric-padding MMU fault is fixed, and f32 GEMM performance is preserved after the guarded-coalescing fix.

Mock targeted check on macOS:

```text
DEV=MOCKKFD+AMD:AMD python -m pytest test/backend/test_ops.py -q -k "max_unpool2d or simple_conv2d_nhwc or asymmetric_padding_conv2d or matmul"
8 passed, 415 deselected, 3 subtests passed in 146.16s
```

### Slice-4a matmul schedule guard

The generic optimizer already selects the desired non-WMMA f32 GEMM shape for AMD asm:

```text
applied_opts=(UPCAST(1,4), UPCAST(0,4), UNROLL(0,4), LOCAL(0,8), LOCAL(1,16))
local_size=(128,1,1)
```

Compile profile for the 64x64 matmul kernel:

| Signal | Count |
|--------|------:|
| `GLOBAL_LOAD_B128` | 8 |
| `GLOBAL_STORE_B128` | 4 |
| `V_FMA_F32` / `AMDOps.MULACC` | 64 |
| scratch ops | 0 |

This means there is no separate matmul heuristic to add yet: the low-risk Slice-4a work is to keep this default schedule from regressing. Large half GEMM uses the hand WMMA path above.

### Pre-float4 baseline

| N | LLVM | ASM | ASM/LLVM |
|---:|---:|---:|---:|
| 512 | 2619 | 522 | 20% |
| 1024 | 4812 | 777 | 16% |
| 2048 | 3061 | 1266 | 41% |
| 4096 | 5802 | 1895 | 33% |

## Speed — elementwise `((a+b)*2)` (GB/s)

| N | LLVM | ASM | ASM/LLVM |
|---:|---:|---:|---:|
| 1M | 300 | 241 | 80% |
| 10M | 327 | 329 | 101% |
| 50M | 527 | 752 | 143% |

## Speed — reduction sum

### Early kernel-bandwidth probe (pre-heuristic, DEBUG stats)

These were per-kernel GB/s from early hardware profiling — they showed a large gap before schedule work:

| N | LLVM GB/s | ASM GB/s | ASM/LLVM |
|---:|---:|---:|---:|
| 1M | ~143 | ~43 | ~30% |
| 10M | ~293 | ~27 | ~9% |
| 50M | ~331 | ~27 | ~8% |

Diagnosis: LDS tree reduction lowering exists; the gap was **schedule geometry**, not missing backend code.

### Forced-opt wall time — N=1M (full linear, `forced_red50.py`)

**Important:** `default=[]` in the probe disables `hand_coded_optimizations` (no-opt baseline). Production default uses GROUPTOP/GROUP heuristics.

| Opt | ASM (ms) | LLVM (ms) |
|-----|---:|---:|
| `default=[]` (no heuristic) | 85 | 137* |
| `GROUPTOP(0, 32)` | 24 | 18 |
| `GROUPTOP(0, 32), UPCAST(0, 2)` | 29 | 23 |
| `GROUPTOP(0, 32), LOCAL(0, 2)` | 19 | 11 |
| **`GROUP(0, 16)`** | **15** | **45** |

\*LLVM first-case cold start can inflate the no-opt row; compare optimized rows.

**Takeaway:** ASM wins with `GROUP(16)`; LLVM wins with `GROUPTOP(32)+LOCAL(2)`. Different backends prefer different group geometry.

### Forced-opt wall time — N=10M (full linear)

| Opt | ASM (ms) | LLVM (ms) |
|-----|---:|---:|
| `default=[]` | 71 | 174 |
| `GROUPTOP(0, 32)` | 24 | 49 |
| `GROUPTOP(0, 32), LOCAL(0, 2)` | 19 | 50 |
| **`GROUP(0, 16)`** | **13** | **38** |

### Forced-opt wall time — N=50M (ASM only, from earlier probe)

| Opt | ASM (ms) |
|-----|---:|
| `default=[]` | ~82 |
| `GROUPTOP(0, 32)` | ~24 |
| `GROUP(0, 16)` | ~15 |

## Reduction heuristic (landed)

Production schedule now selects **`GROUP(0, 16)`** for simple partial scalar ADD reductions via:

- `Renderer.preferred_reduce_group: int | None = None` (generic hook)
- `AMDRenderer.preferred_reduce_group = 16` (only asm opts in)
- Heuristic gate: single ADD reduce axis, `reduceop.src[0]` is `INDEX`, `resolve(prod(output_shape) > 1, False)` (partial output only)

**Hardware verification (`applied_opts` on kernel 0):**

```text
N=1_000_000   → GROUP(0, 16)
N=10_000_000  → GROUP(0, 16)
N=50_000_000  → GROUP(0, 16)
```

**Pending:** synchronized wall-time confirmation with `run_linear(..., wait=True)` at 1M/10M/50M after off-peak credits return.

## Reciprocal / division fix

`V_RCP_F32` + one Newton-Raphson step fixes trunc division (`test_div_rounding_mode`); NaN-safe fallback via `v_cmp_eq` + `v_cndmask` preserves 0/inf edges (sigmoid/gelu/log extremes).

## Other shared-infra fixes (justified independently)

| Change | Why |
|--------|-----|
| `Register.__hash__` by `(name, index)` | Default dataclass hash walked VGPR constraint pools — regalloc hot loop minutes on mock |
| `local_prod_max` + `apply_opt` check | Prevents illegal workgroup sizes (>1024 threads) on gfx11 |
| `unwrap_multi` reads `_device_num` from `ProgramInfo.vars` | Multitensor sharded compiled programs |
| ISA renderer preserves nonzero estimates before `Ops.INS` | `test_global_counters_jit` |
| Late dtype decomposition re-run after transcendental expand | `ulong CDIV/CMOD` for software sin path |
| Regalloc stack epilogue anchoring | X86/AMD spill programs |

## Conclusions

- **Correctness:** Strong on real gfx1100 for main test suites.
- **Elementwise:** Competitive; sometimes faster than LLVM at large N.
- **Reduction:** Schedule was the bottleneck; `GROUP(16)` heuristic addresses the common `sum()` path. Forced-opt shows asm can beat LLVM on the same opt family.
- **f32 GEMM:** float4 memory path is competitive with LLVM on the simple GEMM bench.
- **Float4 memory:** enabled for f32 global/LDS where safe; valid-gated padded/OOB loads intentionally stay scalar.
- **half GEMM / WMMA:** codegen WMMA with ISA TC tile cap. Peak numbers use explicit `extra/gemm/rdna3_asm_wmma_gemm.py` (~60–71k GFLOPS @4096), not auto-dispatch.

## Reproduce on hardware

```bash
cd ~/tinygrad && source venv/bin/activate

# Pre-PR gate (preferred)
DEV=AMD:AMD python ~/github/tiny-tests/amd_gate.py --no-bench

# Core correctness
DEV=AMD:AMD python3 -m unittest test.amd.test_amd_renderer test.test_tiny -q
DEV=AMD:AMD python3 -m pytest test/backend/test_ops.py -q

# Hand WMMA check (explicit kernel, not Tensor.matmul)
DEV=AMD:AMD python ~/github/tiny-tests/amd_hand_wmma.py --check --sizes 4096

# Targeted padded-load regression for float4 memory coalescing
DEV=AMD:AMD python3 -m unittest test.backend.test_ops.TestOps.test_asymmetric_padding_conv2d -q

# Verify reduction heuristic
DEV=AMD:AMD python3 - <<'PY'
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.device import Device
ren = Device['AMD'].renderer
for N in (1_000_000, 10_000_000):
  prg = to_program(Tensor.empty(N, device="AMD").sum().schedule_linear().src[0].src[0], ren)
  print(N, prg.src[0].arg.applied_opts)
PY
```
