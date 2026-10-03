#!/usr/bin/env python3
"""Quick fair A/B for flash DIRECT knobs on 7900."""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time

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
getenv.cache_clear(); amd._amd_flash_attention.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)
q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
out = None
for _ in range(5):
  out = runner(q); Device[Device.DEFAULT].synchronize()
arr = out.numpy(); xs = []
for _ in range(30):
  t0 = time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
print(json.dumps({"label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
                  "finite": bool(np.isfinite(arr).all()), "median": statistics.median(xs), "best": min(xs)}))
'''

def fair(label: str, extra: dict):
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "CCACHE": "0", "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
  line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else f"FAIL {label} {(p.stderr or '')[-250:]}".replace("\n", " ")
  print(line, flush=True)

def main():
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1"})
  subprocess.run([PY, "-c", "from tinygrad import Tensor, Device; Tensor.zeros(1).realize(); print('warm')"],
                 env=env, cwd=ROOT, check=False)
  for label, extra in [
    ("ship", {}),
    ("hip", {"DEV": "AMD"}),  # overwrite DEV; drop FLASH
    ("delay2", {"AMD_WMMA_DELAY": "2"}),
    ("fmac_sched", {"AMD_SCHEDULE_FMAC_ADD": "1"}),
    ("delay_alu8", {"AMD_FMAC_DELAY": "1", "AMD_FMAC_DELAY_N": "8"}),
  ]:
    if label == "hip":
      env = os.environ.copy()
      env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
                  "CCACHE": "0", "LABEL": "hip", "DEV": "AMD"})
      for k in list(env):
        if k.startswith("AMD_") and k != "AMD_SOFT_SYSFS":
          env.pop(k, None)
      p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
      print(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL hip", flush=True)
      continue
    fair(label, extra)

if __name__ == "__main__":
  main()
