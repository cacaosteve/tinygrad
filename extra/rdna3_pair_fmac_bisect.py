#!/usr/bin/env python3
import json, os, subprocess, sys
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
DEV.value = "AMD:AMD"
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
print(json.dumps({"label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
                  "finite": bool(np.isfinite(arr).all()),
                  "pair": os.environ.get("AMD_VOPD_PAIR_ALLOC"), "fmac": os.environ.get("AMD_VOPD_FMAC")}))
'''
for label, extra in [
  ("ship", {}),
  ("pair0", {"AMD_VOPD_PAIR_ALLOC": "0"}),
  ("pair0_nofmac", {"AMD_VOPD_PAIR_ALLOC": "0", "AMD_VOPD_FMAC": "0"}),
  ("ship_nofmac", {"AMD_VOPD_FMAC": "0"}),
  ("pair0_b", {"AMD_VOPD_PAIR_ALLOC": "0"}),
]:
  env = {k: v for k, v in os.environ.items() if not k.startswith("AMD_") or k == "AMD_SOFT_SYSFS"}
  env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0",
              "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
  print(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + (p.stderr or "")[-200:], flush=True)
