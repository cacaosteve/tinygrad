#!/usr/bin/env python3
"""Serial correctness + baseline timing for RDNA3 direct ISA (7900)."""
from __future__ import annotations

import os
import statistics
import time

import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Context, getenv
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, flash_attention


def patterned(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def ref_attn(q: np.ndarray, cache: np.ndarray) -> np.ndarray:
  _, heads, tokens, dim = q.shape
  kv_heads, physical_n = cache.shape[2:4]
  group = heads // kv_heads
  qf = q.astype(np.float16).astype(np.float32)
  kf, vf = cache[0].astype(np.float32), cache[1].astype(np.float32)
  out = np.empty_like(q, dtype=np.float32)
  for h in range(heads):
    scores = qf[0, h] @ kf[0, h // group].T / np.sqrt(dim)
    for t in range(tokens):
      scores[t, physical_n - tokens + t + 1 :] = -np.inf
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out[0, h] = probs @ vf[0, h // group]
  return out


def status(ok: bool) -> str:
  return "OK" if ok else "FAIL"


def main() -> None:
  print("renderer", Device[Device.DEFAULT].renderer.__class__.__name__,
        "arch", getattr(Device[Device.DEFAULT], "arch", "?"))
  print("FLASH_DIRECT", getenv("AMD_FLASH_DIRECT"),
        "ACC_SMALL_env", os.environ.get("AMD_FLASH_ACC_SMALL"),
        "custom", amd_custom_kernels_supported(Device.DEFAULT))
  assert "AMDRenderer" in Device[Device.DEFAULT].renderer.__class__.__name__

  results: list[tuple[str, bool, float | None]] = []

  # Peer GEMM must stay healthy even with AMD_FLASH_DIRECT=1 in the process env.
  a = Tensor.randn(64, 64, dtype=dtypes.half, device=Device.DEFAULT).realize()
  b = Tensor.randn(64, 64, dtype=dtypes.half, device=Device.DEFAULT).realize()
  c = (a @ b).realize().numpy().astype(np.float32)
  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  err = float(np.max(np.abs(c - ref)))
  ok = err < 5e-2 and bool(np.isfinite(c).all())
  results.append(("pre_flash_matmul64", ok, err))
  print(f"GEMM pre_flash_matmul64: err={err:.3e} {status(ok)}")

  cfgs = [
    ("prefill_gqa_32", 32, 8, 128, 2048, 32),
    ("prefill_gqa_tiles", 32, 8, 128, 512, 64),
    ("prefill_mha", 16, 16, 64, 256, 32),
    ("decode_gqa", 32, 8, 128, 2048, 1),
  ]
  for name, heads, kv_heads, dim, physical_n, tokens in cfgs:
    batch = 1
    q_np = patterned((batch, heads, tokens, dim))
    cache_np = patterned((2, batch, kv_heads, physical_n, dim), np.float16)
    ref = ref_attn(q_np, cache_np)
    q = Tensor(q_np, device=Device.DEFAULT).realize()
    cache = Tensor(cache_np, device=Device.DEFAULT).realize()
    out = flash_attention(q, cache, physical_n).realize().numpy().astype(np.float32)
    err = float(np.max(np.abs(out - ref)))
    nz = float(np.mean(np.abs(out) > 1e-6))
    ok = err < 5e-3 and nz > 0.5 and bool(np.isfinite(out).all())
    results.append((name, ok, err))
    print(f"FLASH {name}: err={err:.3e} nonzero_frac={nz:.3f} {status(ok)}")

  # Flash ACC_SMALL realize must not leave peer kernels broken.
  a = Tensor.randn(64, 64, dtype=dtypes.half, device=Device.DEFAULT).realize()
  b = Tensor.randn(64, 64, dtype=dtypes.half, device=Device.DEFAULT).realize()
  c = (a @ b).realize().numpy().astype(np.float32)
  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  err = float(np.max(np.abs(c - ref)))
  ok = err < 5e-2 and bool(np.isfinite(c).all())
  results.append(("post_flash_matmul64", ok, err))
  print(f"GEMM post_flash_matmul64: err={err:.3e} {status(ok)}")

  os.environ["TC_LDS_AB"] = "1"
  getenv.cache_clear()
  with Context(BEAM=0):
    eye = Tensor.eye(256, dtype=dtypes.half).to(Device.DEFAULT).realize()
    b_np = patterned((256, 256), np.float16)
    b = Tensor(b_np, device=Device.DEFAULT).realize()
    c = (eye @ b).realize().numpy().astype(np.float32)
    err = float(np.max(np.abs(c - b_np.astype(np.float32))))
    ok = err < 1e-2 and bool(np.isfinite(c).all())
    results.append(("eye@B_TC_LDS_AB", ok, err))
    print(f"EYE eye@B: err={err:.3e} {status(ok)}")

    a_np = patterned((256, 256), np.float16)
    a = Tensor(a_np, device=Device.DEFAULT).realize()
    c2 = (a @ eye).realize().numpy().astype(np.float32)
    err2 = float(np.max(np.abs(c2 - a_np.astype(np.float32))))
    ok2 = err2 < 1e-2 and bool(np.isfinite(c2).all())
    results.append(("A@eye_TC_LDS_AB", ok2, err2))
    print(f"EYE A@eye: err={err2:.3e} {status(ok2)}")
  os.environ.pop("TC_LDS_AB", None)
  getenv.cache_clear()

  rng = np.random.default_rng(0)
  w = (rng.standard_normal((256, 256)) * 0.1).astype(np.float32)
  x = rng.standard_normal((256,)).astype(np.float32)
  y = (Tensor(w, device=Device.DEFAULT).half() @ Tensor(x, device=Device.DEFAULT).half()).realize().numpy().astype(np.float32)
  ref = w.astype(np.float16).astype(np.float32) @ x.astype(np.float16).astype(np.float32)
  err = float(np.max(np.abs(y - ref)))
  ok = err < 2e-2 and bool(np.isfinite(y).all())
  results.append(("fp16_gemv_256", ok, err))
  print(f"GEMV fp16_256: err={err:.3e} {status(ok)}")

  import subprocess, sys, textwrap
  x_np = patterned((2048,))
  for ftype, bpb, name, atol in [
    (12, 144, "Q4_K", 1e-3),
    (13, 176, "Q5_K", 1e-3),
    (14, 210, "Q6_K", 1e-3),
    (21, 136, "IQ4_XS", 1e-3),
  ]:
    rows, cols = 1024, 2048
    nblocks = rows * cols // 256
    qdata_np = (np.arange(nblocks * bpb, dtype=np.uint8) * 37 + 13).astype(np.uint8)
    y_amd = (ggml_data_to_tensor(Tensor(qdata_np, device=Device.DEFAULT), rows * cols, ftype).reshape(rows, cols)
             @ Tensor(x_np, device=Device.DEFAULT)).realize().numpy().astype(np.float32)
    finite_amd = bool(np.isfinite(y_amd).all())
    print(f"QUANT {name} AMD: finite={finite_amd}")
    np.save("/tmp/rdna3_qdata.npy", qdata_np)
    np.save("/tmp/rdna3_x.npy", x_np)
    child_py = textwrap.dedent(f"""
      import numpy as np
      from tinygrad import Tensor, Device
      from tinygrad.llm.gguf import ggml_data_to_tensor
      qdata_np = np.load("/tmp/rdna3_qdata.npy")
      x_np = np.load("/tmp/rdna3_x.npy")
      y = (ggml_data_to_tensor(Tensor(qdata_np, device=Device.DEFAULT), {rows}*{cols}, {ftype}).reshape({rows}, {cols})
           @ Tensor(x_np, device=Device.DEFAULT)).realize().numpy().astype(np.float32)
      np.save("/tmp/rdna3_hip_y.npy", y)
      print("finite", bool(np.isfinite(y).all()))
    """)
    env = {**os.environ, "DEV": "AMD", "PYTHONPATH": ".:extra",
           "AMD_FLASH_DIRECT": "0", "AMD_FLASH_ACC_SMALL": "0"}
    out = subprocess.check_output([sys.executable, "-c", child_py], cwd=os.getcwd(), env=env)
    print("  HIP:", out.decode().strip())
    y_hip = np.load("/tmp/rdna3_hip_y.npy")
    finite_hip = bool(np.isfinite(y_hip).all())
    if finite_amd and finite_hip:
      err = float(np.max(np.abs(y_amd - y_hip)))
      ok = err < atol
      results.append((f"{name}_vs_hip", ok, err))
      print(f"QUANT {name} vs HIP: err={err:.3e} {status(ok)}")
    else:
      ok = finite_amd == finite_hip
      results.append((f"{name}_vs_hip", ok, None))
      print(f"QUANT {name} vs HIP: finite_match={ok} amd={finite_amd} hip={finite_hip} {status(ok)}")

  batch, heads, kv_heads, dim, physical_n, tokens = 1, 32, 8, 128, 2048, 32
  q = Tensor(patterned((batch, heads, tokens, dim)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, batch, kv_heads, physical_n, dim), np.float16), device=Device.DEFAULT).realize()
  runner = TinyJit(lambda qu: flash_attention(qu, cache, physical_n).realize())
  for _ in range(3):
    runner(q)
    Device[Device.DEFAULT].synchronize()
  samples = []
  for _ in range(5):
    t0 = time.perf_counter_ns()
    for _ in range(10):
      runner(q)
    Device[Device.DEFAULT].synchronize()
    samples.append((time.perf_counter_ns() - t0) / 10 / 1e3)
  print(f"PERF prefill DIRECT median_us={statistics.median(samples):.1f} best={min(samples):.1f}")

  failed = [r for r in results if not r[1]]
  print("SUMMARY", status(not failed), f"{len(results) - len(failed)}/{len(results)}")
  for r in failed:
    print(" FAIL", r)
  raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
  main()
