#!/usr/bin/env python3
"""Same-process TinyJit smoke: prefill then decode under ship flash DIRECT policy.

Validates sticky compile env (prefill clears decode-only keys) and checksums.
Usage:
  AMD_SOFT_SYSFS=1 PYTHONPATH=. python extra/rdna3_flash_policy_smoke.py
"""
from __future__ import annotations
import json, math, os, statistics, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import DEV, getenv
from tinygrad.codegen import to_program_cache
from tinygrad.llm.kernels.amd import flash_attention, _FLASH_PREFILL, _FLASH_DECODE, _FLASH_DECODE_ONLY

DEV.value = os.environ.get("DEV", "AMD:AMD")
os.environ.setdefault("AMD_FLASH_DIRECT", "1")
getenv.cache_clear()
to_program_cache.clear()

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

def median_us(fn, n=40) -> float:
  xs = []
  for _ in range(n):
    t0 = time.perf_counter_ns()
    fn(); Device[Device.DEFAULT].synchronize()
    xs.append((time.perf_counter_ns() - t0) / 1e3)
  return float(statistics.median(sorted(xs)))

cache_np = patterned((2, 1, 8, 2048, 128), np.float16)
cache = Tensor(cache_np, device=Device.DEFAULT).realize()
valid = 2048

# Prefill
q_p = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
prefill = TinyJit(lambda qq: flash_attention(qq, cache, valid).realize())
out_p = prefill(q_p); Device[Device.DEFAULT].synchronize()
for _ in range(5): out_p = prefill(q_p); Device[Device.DEFAULT].synchronize()
arr_p = out_p.numpy()
sum_p = float(arr_p.astype("float64").sum()) if np.isfinite(arr_p).all() else None
med_p = median_us(lambda: prefill(q_p))

# After prefill: decode-only keys must not stick into a cold decode build.
# Decode
q_d_np = patterned((1, 32, 1, 128))
q_d = Tensor(q_d_np, device=Device.DEFAULT).realize()
decode = TinyJit(lambda qq: flash_attention(qq, cache, valid).realize())
out_d = decode(q_d); Device[Device.DEFAULT].synchronize()
for _ in range(5): out_d = decode(q_d); Device[Device.DEFAULT].synchronize()
arr_d = out_d.numpy()
sum_d = float(arr_d.astype("float64").sum()) if np.isfinite(arr_d).all() else None
med_d = median_us(lambda: decode(q_d))
ref_d = np.empty_like(arr_d)
for h in range(q_d.shape[1]):
  kvh = h // (q_d.shape[1] // cache.shape[2])
  scores = cache_np[0, 0, kvh].astype(np.float32) @ q_d_np[0, h, 0] * (1 / math.sqrt(q_d.shape[-1]))
  probs = np.exp(scores - scores.max())
  probs /= probs.sum()
  ref_d[0, h, 0] = probs @ cache_np[1, 0, kvh].astype(np.float32)

ok_p = sum_p is not None and abs(sum_p - (-3.3788577654734127)) < 1e-4
ok_d = sum_d is not None and np.allclose(arr_d, ref_d, atol=2e-4, rtol=1e-3)

print(json.dumps({
  "dev": Device.DEFAULT,
  "dev_env": os.environ.get("DEV"),
  "renderer": type(Device[Device.DEFAULT].renderer).__name__,
  "prefill": {"sum": sum_p, "median_us": med_p, "ok": ok_p,
              "table_keys": len(_FLASH_PREFILL), "decode_only": list(_FLASH_DECODE_ONLY[:4])},
  "decode": {"sum": sum_d, "median_us": med_d, "ok": ok_d, "table_keys": len(_FLASH_DECODE)},
  "pass": ok_p and ok_d,
}, indent=2))
raise SystemExit(0 if ok_p and ok_d else 1)
