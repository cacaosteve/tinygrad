#!/usr/bin/env python3
"""Elementwise ship vs AMD_CLUSTER_PACK_LLOAD=1 on patterned fair flash."""
from __future__ import annotations
import os, subprocess, sys
import numpy as np

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import json, os
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
for _ in range(3):
  out = runner(q); Device[Device.DEFAULT].synchronize()
arr = out.numpy()
np.save(os.environ["OUT_NPY"], arr)
print(json.dumps({"label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
                  "finite": bool(np.isfinite(arr).all())}))
'''

def run(label: str, extra: dict) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "CCACHE": "0", "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1",
              "OUT_NPY": f"/tmp/flash_{label}.npy"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
  print(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else f"FAIL {label} {(p.stderr or '')[-400:]}", flush=True)
  assert p.returncode == 0

def main():
  subprocess.run([PY, "-c", "from tinygrad import Tensor; Tensor.zeros(1).realize(); print('warm')"],
                 env={**os.environ, "PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "DEV": "AMD:AMD"},
                 cwd=ROOT, check=False)
  run("ship", {})
  run("cluster", {"AMD_CLUSTER_PACK_LLOAD": "1"})
  a = np.load("/tmp/flash_ship.npy"); b = np.load("/tmp/flash_cluster.npy")
  d = np.abs(a.astype(np.float64) - b.astype(np.float64))
  print("maxdiff", float(d.max()), "meandiff", float(d.mean()),
        "frac_gt_1e-3", float((d > 1e-3).mean()), "frac_gt_1e-2", float((d > 1e-2).mean()))
  idx = int(d.ravel().argmax())
  print("argmax", np.unravel_index(idx, d.shape), "ship", float(a.ravel()[idx]), "cluster", float(b.ravel()[idx]))
  # per-head sum delta
  if a.ndim >= 2:
    hs = a.astype(np.float64).reshape(a.shape[0], a.shape[1], -1).sum(-1)
    hc = b.astype(np.float64).reshape(b.shape[0], b.shape[1], -1).sum(-1)
    print("per_head_sum_maxdiff", float(np.abs(hs - hc).max()))

if __name__ == "__main__":
  main()
