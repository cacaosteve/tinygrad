#!/usr/bin/env python3
"""Resume gap iteration when gaming PC is back: priv0 ship gate + fair vs HIP."""
from __future__ import annotations
import json, os, subprocess, sys
from collections import Counter

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPILE = r'''
import json, os
from collections import Counter
os.environ.update({"DEV":"AMD:AMD","AMD_FLASH_DIRECT":"1","PARALLEL":"0","CACHELEVEL":"0","CCACHE":"0","AMD_SOFT_SYSFS":"1"})
for k,v in json.loads(os.environ.get("EXTRA","{}")).items():
  if v is None: os.environ.pop(k, None)
  else: os.environ[k] = str(v)
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
DEV.value = "AMD:AMD"
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
ren = AMDRenderer(Target("AMD", arch="gfx1100"))
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
  prg = to_program(_amd_flash_attention(
    UOp.placeholder((32, 32, 128), dtypes.float32, 0),
    UOp.placeholder((32, 32, 128), dtypes.float16, 1),
    UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
    valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
    use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)
binary = prg.src[-1].arg
text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
insts, off = [], 0
while off < len(text):
  inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
  if getattr(inst, "op_name", "").lower() == "s_endpgm": break
mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
vgpr, _, _, priv = scan_elf_regs(insts, scratch_inst_size)
print(json.dumps({
  "label": os.environ["LABEL"], "priv": priv, "vgpr": vgpr, "ninst": len(insts),
  "v_mov": mc.get("v_mov_b32_e32", 0),
  "v_dual": sum(v for k, v in mc.items() if "dual" in k),
  "dual_fmac": sum(v for k, v in mc.items() if "dual_fmac" in k),
  "dual_mul": sum(v for k, v in mc.items() if "dual_mul" in k),
  "scratch": sum(v for k, v in mc.items() if "scratch_" in k),
}))
'''

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
for _ in range(40):
  t0 = time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
print(json.dumps({"label": os.environ["LABEL"], "sum": float(arr.astype("float64").sum()),
                  "finite": bool(np.isfinite(arr).all()), "median": statistics.median(xs), "best": min(xs)}))
'''

def main():
  # Compile A/B: shipping priv0 vs shallow_depth=1 opt-out
  for label, extra in [
    ("ship", {}),
    ("shallow1", {"AMD_REMAT_ADDR_SHALLOW_DEPTH": "1"}),
    ("nopair", {"AMD_VOPD_PAIR_ALLOC": "0"}),
  ]:
    env = os.environ.copy()
    env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "LABEL": label, "EXTRA": json.dumps(extra)})
    p = subprocess.run([PY, "-c", COMPILE], env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
    print("C", p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + (p.stderr or "")[-300:].replace("\n", " "), flush=True)

  # Fair sync-each: HIP + ship + shallow1 (checksum gate ≈ −3.3788)
  for label, cfg in [
    ("hip", {"DEV": "AMD"}),
    ("ship", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"}),
    ("shallow1", {"DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "AMD_REMAT_ADDR_SHALLOW_DEPTH": "1"}),
  ]:
    env = os.environ.copy()
    env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0", "LABEL": label})
    for k in ("AMD_FLASH_DIRECT", "AMD_VOPD_PAIR_ALLOC", "AMD_REMAT_ADDR_SHALLOW_DEPTH", "AMD_MUL_STRENGTH"):
      env.pop(k, None)
    env.update(cfg)
    p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=240, cwd=ROOT)
    print("T", p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + label + " " + (p.stderr or "")[-250:].replace("\n", " "), flush=True)

if __name__ == "__main__":
  main()
