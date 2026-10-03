#!/usr/bin/env python3
"""Paired DIRECT | LLVM | HIP scorecard on one machine.

Flash: DIRECT/HIP use hand kernels (CUSTOM); LLVM cannot render those — column uses
SDPA (documented). Matmul/GEMV compare same Tensor graphs.

Usage:
  AMD_SOFT_SYSFS=1 PYTHONPATH=. python extra/rdna3_backend_matrix.py
"""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKENDS = [
  ("direct", "AMD:AMD", {"AMD_FLASH_DIRECT": "1"}),
  ("llvm", "AMD:LLVM", {}),
  ("hip", "AMD", {}),
]

FLASH_CHILD = r'''
import json, os, statistics, time, math
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import DEV, getenv
from tinygrad.codegen import to_program_cache

getenv.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
tokens = int(os.environ["TOKENS"])
backend = os.environ["BACKEND"]

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

q = Tensor(patterned((1, 32, tokens, 128)), device=Device.DEFAULT).realize()
cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
valid = 2048

if backend == "llvm":
  # Hand flash uses Ops.CUSTOM (ds_swizzle etc.) — LLVM IR cannot render them.
  from tinygrad.llm.kernels.amd import flash_attention as _fa  # noqa: F401
  def run(qq):
    k = cache[0, :, :, :valid].float()
    v = cache[1, :, :, :valid].float()
    mask = None if tokens == 1 else Tensor.full((tokens, valid), -math.inf, dtype=dtypes.float32, device=qq.device).triu(valid - tokens + 1)
    return qq.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).realize()
  path = "sdpa"
else:
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  amd._amd_flash_attention.cache_clear()
  if hasattr(amd, "_amd_flash_attention_decode_partial"):
    amd._amd_flash_attention_decode_partial.cache_clear()
  def run(qq):
    return flash_attention(qq, cache, valid).realize()
  path = "hand_flash"

t_cold0 = time.perf_counter()
runner = TinyJit(run)
out = runner(q); Device[Device.DEFAULT].synchronize()
cold_ms = (time.perf_counter() - t_cold0) * 1e3
for _ in range(7):
  out = runner(q); Device[Device.DEFAULT].synchronize()
arr = out.numpy()
xs = []
for _ in range(int(os.environ.get("NREP", "80"))):
  t0 = time.perf_counter_ns()
  runner(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
xs = sorted(xs)
print(json.dumps({
  "workload": os.environ["WORKLOAD"], "backend": backend, "path": path,
  "dev": os.environ["DEV"], "renderer": type(Device[Device.DEFAULT].renderer).__name__,
  "sum": float(arr.astype("float64").sum()) if np.isfinite(arr).all() else None,
  "finite": bool(np.isfinite(arr).all()),
  "median_us": statistics.median(xs), "best_us": min(xs),
  "p10_us": xs[max(0, len(xs)//10 - 1)], "cold_ms": cold_ms,
}))
'''

MATMUL_CHILD = r'''
import json, os, statistics, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import DEV, getenv
from tinygrad.codegen import to_program_cache

getenv.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
N = int(os.environ.get("N", "2048"))
rng = np.random.default_rng(0)
a_np = (rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16)
b_np = (rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16)
ref = (a_np.astype(np.float32) @ b_np.astype(np.float32))
a = Tensor(a_np).realize(); b = Tensor(b_np).realize()

t_cold0 = time.perf_counter()
runner = TinyJit(lambda: (a @ b).realize())
out = runner(); Device[Device.DEFAULT].synchronize()
cold_ms = (time.perf_counter() - t_cold0) * 1e3
for _ in range(5):
  out = runner(); Device[Device.DEFAULT].synchronize()
got = out.numpy().astype(np.float32)
xs = []
for _ in range(int(os.environ.get("NREP", "40"))):
  t0 = time.perf_counter_ns()
  runner(); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
xs = sorted(xs)
max_err = float(np.max(np.abs(got - ref)))
print(json.dumps({
  "workload": "matmul_h_2048", "backend": os.environ["BACKEND"],
  "dev": os.environ["DEV"], "renderer": type(Device[Device.DEFAULT].renderer).__name__,
  "sum": float(got.astype(np.float64).sum()), "finite": bool(np.isfinite(got).all()),
  "max_err_vs_numpy_f32": max_err,
  "ok_atol": bool(max_err < 1.0),  # half matmul loose gate
  "median_us": statistics.median(xs), "best_us": min(xs),
  "p10_us": xs[max(0, len(xs)//10 - 1)], "cold_ms": cold_ms,
}))
'''

GEMV_CHILD = r'''
import json, os, statistics, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, nn
from tinygrad.helpers import DEV, getenv
from tinygrad.codegen import to_program_cache
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear

getenv.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
rows, cols, tokens = 8192, 2048, 1
QK, block_bytes, ggml_type = 256, 144, 12
blocks = rows * cols // QK
idx = np.arange(blocks * block_bytes, dtype=np.uint32)
qdata = ((idx * 37 + 13 + (idx >> 8) * 17) & 0xff).astype(np.uint8)
qblocks = qdata.reshape(blocks, block_bytes)
qblocks[:, :4] = np.array([0.03125, 0.015625], dtype=np.float16).view(np.uint8)
x_np = (((np.arange(tokens * cols, dtype=np.int32) % 31) - 15) / 16.0).astype(np.float32).reshape(tokens, cols)
raw = Tensor(np.pad(qdata, (4, 0)), device=Device.DEFAULT).contiguous().realize()[4:]
decoded = ggml_data_to_tensor(raw, rows * cols, ggml_type).reshape(rows, cols)
linear = Linear(cols, rows, bias=False)
nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
x = Tensor(x_np, device=Device.DEFAULT).realize()

t_cold0 = time.perf_counter()
runner = TinyJit(lambda xx: linear(xx).realize())
out = runner(x); Device[Device.DEFAULT].synchronize()
cold_ms = (time.perf_counter() - t_cold0) * 1e3
for _ in range(5):
  out = runner(x); Device[Device.DEFAULT].synchronize()
arr = out.numpy()
xs = []
for _ in range(int(os.environ.get("NREP", "60"))):
  t0 = time.perf_counter_ns()
  runner(x); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
xs = sorted(xs)
print(json.dumps({
  "workload": "q4k_gemv_8192x2048", "backend": os.environ["BACKEND"],
  "dev": os.environ["DEV"], "renderer": type(Device[Device.DEFAULT].renderer).__name__,
  "sum": float(arr.astype("float64").sum()) if np.isfinite(arr).all() else None,
  "finite": bool(np.isfinite(arr).all()),
  "median_us": statistics.median(xs), "best_us": min(xs),
  "p10_us": xs[max(0, len(xs)//10 - 1)], "cold_ms": cold_ms,
}))
'''


def run_child(label_backend: str, dev: str, extra: dict, child: str, env_extra: dict) -> dict:
  env = {k: v for k, v in os.environ.items() if not k.startswith("AMD_") or k == "AMD_SOFT_SYSFS"}
  # Drop experiment knobs so product defaults (TC_LDS_AB for AMDRenderer) apply.
  for k in ("TC_LDS_AB", "TC_UPCAST", "TC_LOCAL", "TC_UPCAST_TILES", "ALLOW_UPCAST16", "TC_LDS_UNROLL",
            "AMD_WMMA_ACC_BASE", "AMD_WMMA_ACC_N", "AMD_PREFETCH_A"):
    env.pop(k, None)
  env.update({
    "AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0",
    "DEV": dev, "BACKEND": label_backend,
  })
  env.update(extra)
  env.update(env_extra)
  p = subprocess.run([PY, "-c", child], env=env, capture_output=True, text=True, timeout=600, cwd=ROOT)
  if p.returncode != 0 or not p.stdout.strip():
    return {
      "workload": env_extra.get("WORKLOAD", "?"), "backend": label_backend, "dev": dev,
      "error": (p.stderr or p.stdout or "")[-500:].replace("\n", " "),
    }
  line = p.stdout.strip().splitlines()[-1]
  try:
    return json.loads(line)
  except json.JSONDecodeError:
    return {"backend": label_backend, "dev": dev, "error": line[-400:]}


def main() -> None:
  rows = []
  for case, tokens in (("flash_prefill", 32), ("flash_decode", 1)):
    for be, dev, extra in BACKENDS:
      print(f"… {case} {be}", flush=True)
      rows.append(run_child(be, dev, extra, FLASH_CHILD, {
        "WORKLOAD": case, "TOKENS": str(tokens), "NREP": "80",
      }))
  for be, dev, extra in BACKENDS:
    print(f"… matmul {be}", flush=True)
    rows.append(run_child(be, dev, extra, MATMUL_CHILD, {"N": "2048", "NREP": "40"}))
  for be, dev, extra in BACKENDS:
    print(f"… gemv {be}", flush=True)
    # Q4K wants full VGPR (product-16 ACC_BASE=96 taxes GEMV if sticky).
    rows.append(run_child(be, dev, extra, GEMV_CHILD, {"NREP": "60", "AMD_WMMA_ACC_BASE": "-1"}))

  print("=== MATRIX ===", flush=True)
  for r in rows:
    print(json.dumps(r), flush=True)

  by = {}
  for r in rows:
    if "error" in r: continue
    by.setdefault(r["workload"], {})[r["backend"]] = r
  print("=== REL vs llvm (median) ===", flush=True)
  for wl, m in by.items():
    base_be = "llvm" if "llvm" in m else ("hip" if "hip" in m else None)
    if not base_be: continue
    base = m[base_be]["median_us"]
    for be in ("direct", "llvm", "hip"):
      if be not in m: continue
      med = m[be]["median_us"]
      print(json.dumps({
        "workload": wl, "backend": be, "vs": base_be,
        "median_us": med, f"vs_{base_be}": round(med / base, 3),
        "sum": m[be].get("sum"), "cold_ms": round(m[be].get("cold_ms", 0), 1),
        "path": m[be].get("path"), "renderer": m[be].get("renderer"),
        "max_err": m[be].get("max_err_vs_numpy_f32"),
      }), flush=True)
  print("=== REL vs hip (median) ===", flush=True)
  for wl, m in by.items():
    if "hip" not in m: continue
    base = m["hip"]["median_us"]
    for be in ("direct", "llvm", "hip"):
      if be not in m: continue
      med = m[be]["median_us"]
      print(json.dumps({
        "workload": wl, "backend": be,
        "median_us": med, "vs_hip": round(med / base, 3),
      }), flush=True)


if __name__ == "__main__":
  main()
