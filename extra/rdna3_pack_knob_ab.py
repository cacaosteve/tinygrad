#!/usr/bin/env python3
"""Quick compile+fair A/B for MOV/pack knobs vs HIP."""
from __future__ import annotations
import json, os, subprocess, sys

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
  "dual_mov": sum(v for k, v in mc.items() if "dual_mov" in k),
  "scratch": sum(v for k, v in mc.items() if "scratch_" in k),
}))
'''

FAIR = r'''
import json, os, statistics, time
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import DEV, getenv
from tinygrad.llm.kernels.amd import flash_attention
import tinygrad.llm.kernels.amd as amd
from tinygrad.codegen import to_program_cache
import numpy as np
for k,v in json.loads(os.environ.get("EXTRA","{}")).items():
  if v is None: os.environ.pop(k, None)
  else: os.environ[k] = str(v)
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

VARIANTS = [
  ("ship", {}),
  ("alias_lload0", {"AMD_PACK_F16_ALIAS_LLOAD": "0"}),
  ("pack_general", {"AMD_PACK_F16_GENERAL": "1"}),
  ("contig0", {"AMD_PACK_F16_ALIAS_CONTIG": "0"}),
  ("acc201", {"AMD_WMMA_ACC_BASE": "201"}),
  ("interleave0", {"AMD_WMMA_INTERLEAVE": "0"}),
  ("lds_il0", {"AMD_WMMA_LDS_INTERLEAVE": "0"}),
]

def run(label, extra, kind):
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "LABEL": label, "EXTRA": json.dumps(extra)})
  code = COMPILE if kind == "C" else FAIR
  if kind == "T":
    env["DEV"] = extra.get("DEV", "AMD:AMD")
    if "DEV" not in extra and label != "hip":
      extra = {**extra, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"}
      env["EXTRA"] = json.dumps(extra)
      env["DEV"] = "AMD:AMD"
  p = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, timeout=180, cwd=ROOT)
  line = (p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip()
          else "FAIL " + (p.stderr or p.stdout or "")[-400:].replace("\n", " "))
  print(kind, line, flush=True)

def main():
  for label, extra in VARIANTS:
    run(label, extra, "C")
  run("hip", {"DEV": "AMD"}, "T")
  for label, extra in VARIANTS[:5]:  # fair subset
    run(label, {**extra, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"}, "T")

if __name__ == "__main__":
  main()
