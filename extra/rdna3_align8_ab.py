#!/usr/bin/env python3
"""Confirm ALIGN8 off + real HIP fair baseline."""
from __future__ import annotations
import json, os, pathlib, subprocess, statistics, time

ROOT = "/home/admin441766/tinygrad"
PY = f"{ROOT}/venv/bin/python"
CHILD = pathlib.Path(f"{ROOT}/extra/rdna3_fair_ab_knobs.py").read_text().split('CHILD = r"""')[1].split('"""')[0]

def run_direct(label: str, extra: dict) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
  print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-400:]}", flush=True)

HIP = r"""
import json,os,statistics,time
from tinygrad import Tensor,Device,TinyJit
from tinygrad.helpers import DEV,getenv
from tinygrad.llm.kernels.amd import flash_attention
import tinygrad.llm.kernels.amd as amd
from tinygrad.codegen import to_program_cache
import numpy as np
getenv.cache_clear(); amd._amd_flash_attention.cache_clear(); to_program_cache.clear()
DEV.value="AMD"
def patterned(shape,dtype=np.float32):
  c=int(np.prod(shape)); return (((np.arange(c,dtype=np.int32)%127)-63)/64.0).astype(dtype).reshape(shape)
q=Tensor(patterned((1,32,32,128)),device=Device.DEFAULT).realize()
cache=Tensor(patterned((2,1,8,2048,128),np.float16),device=Device.DEFAULT).realize()
runner=TinyJit(lambda qq: flash_attention(qq,cache,2048).realize())
for _ in range(5):
  out=runner(q); Device[Device.DEFAULT].synchronize()
arr=out.numpy(); xs=[]
for _ in range(80):
  t0=time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize(); xs.append((time.perf_counter_ns()-t0)/1e3)
xs=sorted(xs)
print(json.dumps({"label":os.environ["LABEL"],"ren":Device[Device.DEFAULT].renderer.__class__.__name__,
  "sum":float(arr.astype("float64").sum()),"best":xs[0],"p10":xs[8],"median":statistics.median(xs)}))
"""

def run_hip(label: str) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "LABEL": label, "DEV": "AMD"})
  for k in list(env):
    if k.startswith("AMD_") and k != "AMD_SOFT_SYSFS":
      env.pop(k, None)
  p = subprocess.run([PY, "-c", HIP], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
  print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-500:]}", flush=True)

if __name__ == "__main__":
  for i in range(3):
    run_direct(f"ship{i}", {})
  for i in range(3):
    run_direct(f"a8off{i}", {"AMD_PACK_LLOAD_ALIGN8": "0"})
  for i in range(2):
    run_direct(f"a48off{i}", {"AMD_PACK_LLOAD_ALIGN8": "0", "AMD_PACK_LLOAD_ALIGN4": "0"})
  for i in range(2):
    run_hip(f"hip{i}")
