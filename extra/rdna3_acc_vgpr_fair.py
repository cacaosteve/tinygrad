#!/usr/bin/env python3
"""Fresh-process fair A/B: shipping ACC_WORK vs ACC_VGPR vs HIP."""
from __future__ import annotations
import json, os, statistics, subprocess, sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [
  ("hip", {"DEV": "AMD"}),
  ("ship", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"}),
  ("vgpr", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "AMD_FLASH_ACC_VGPR": "1"}),
  ("vgpr_il", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "AMD_FLASH_ACC_VGPR": "1",
               "AMD_WMMA_INTERLEAVE": "1", "AMD_WMMA_LDS_INTERLEAVE": "1"}),
]

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
arr = out.numpy()
xs = []
for _ in range(40):
  t0 = time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
print(json.dumps({
  "label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
  "finite": bool(np.isfinite(arr).all()), "median": statistics.median(xs),
  "best": min(xs), "p90": sorted(xs)[int(0.9 * len(xs))],
  "ren": Device[Device.DEFAULT].renderer.__class__.__name__,
}))
'''

def main():
  rows = []
  for label, cfg in CONFIGS:
    env = os.environ.copy()
    env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0",
                "CACHELEVEL": "0", "CCACHE": "0", "LABEL": label})
    for k in ("AMD_FLASH_DIRECT", "AMD_FLASH_ACC_VGPR", "AMD_WMMA_INTERLEAVE", "AMD_WMMA_LDS_INTERLEAVE"):
      env.pop(k, None)
    env.update(cfg)
    p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
    if p.returncode != 0:
      print("FAIL", label, p.stderr[-600:], file=sys.stderr)
      sys.exit(1)
    row = json.loads(p.stdout.strip().splitlines()[-1])
    rows.append(row)
    print(row, flush=True)
  print(json.dumps({"rows": rows}, indent=2))

if __name__ == "__main__":
  main()
