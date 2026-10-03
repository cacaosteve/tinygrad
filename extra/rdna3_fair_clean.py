#!/usr/bin/env python3
"""Clean fair sync-each: hip vs ship (+ optional EXTRA json)."""
from __future__ import annotations
import json, os, subprocess, sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r'''
import json, os, statistics, time
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import DEV, getenv
from tinygrad.llm.kernels.amd import flash_attention
import tinygrad.llm.kernels.amd as amd
from tinygrad.codegen import to_program_cache
import numpy as np
for k in list(os.environ):
  if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
    pass
getenv.cache_clear(); amd._amd_flash_attention.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)
q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
out = None
for _ in range(8):
  out = runner(q); Device[Device.DEFAULT].synchronize()
arr = out.numpy(); xs = []
for _ in range(50):
  t0 = time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
print(json.dumps({"label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
                  "finite": bool(np.isfinite(arr).all()), "median": statistics.median(xs),
                  "best": min(xs), "p10": sorted(xs)[5]}))
'''

def run(label, cfg):
  env = {k: v for k, v in os.environ.items() if not k.startswith("AMD_") or k == "AMD_SOFT_SYSFS"}
  env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0", "LABEL": label})
  env.update(cfg)
  for k, v in json.loads(os.environ.get("EXTRA_JSON", "{}")).items():
    if label == "ship" or label.startswith("ab_"):
      env[k] = str(v)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300, cwd=ROOT)
  line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + label + " " + (p.stderr or "")[-300:].replace("\n", " ")
  print("T", line, flush=True)

def main():
  run("hip", {"DEV": "AMD"})
  run("ship", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  # A/Bs: turn off each shipped knob
  for label, extra in [
    ("ab_scan8", {"AMD_VOPD_MOV_SCAN": "8"}),
    ("ab_nopair", {"AMD_VOPD_PAIR_ALLOC": "0"}),
    ("ab_noinline", {"AMD_MOV_INLINE_CONST": "0"}),
    ("ab_nomul", {"AMD_MUL_STRENGTH": "0"}),
    ("ab_delay0", {"AMD_WMMA_DELAY": "0"}),
  ]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AMD_") or k == "AMD_SOFT_SYSFS"}
    env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0",
                "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
    env.update(extra)
    p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300, cwd=ROOT)
    line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + label + " " + (p.stderr or "")[-300:].replace("\n", " ")
    print("T", line, flush=True)

if __name__ == "__main__":
  main()
