#!/usr/bin/env python3
"""Quick fair A/B for flash DIRECT knobs."""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time
from collections import Counter

ROOT = "/home/admin441766/tinygrad"
PY = f"{ROOT}/venv/bin/python"
CHILD = r"""
import json,os,statistics,time
from collections import Counter
from tinygrad import Tensor,Device,TinyJit,dtypes
from tinygrad.helpers import DEV,getenv,Context,Target
from tinygrad.llm.kernels.amd import flash_attention,_amd_flash_attention,_flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.codegen import to_program,to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.uop.ops import UOp
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs,scratch_inst_size
import numpy as np
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
DEV.value="AMD:AMD"
ren=AMDRenderer(Target("AMD", arch="gfx1100"))
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  prg=to_program(_amd_flash_attention(
    UOp.placeholder((32,32,128),dtypes.float32,0),
    UOp.placeholder((32,32,128),dtypes.float16,1),
    UOp.placeholder((2,1,8,2048,128),dtypes.float16,2),
    valid_kv_len=2048,acc_small=True,k_unroll=1,k_midstore=8,use_acc_work=False,acc_vgpr=True,hoist_q=True),ren)
text=next(s.content for s in elf_loader(prg.src[-1].arg)[1] if s.name==".text")
insts,off=[],0
while off<len(text):
  inst=decode_inst(text[off:],"rdna3"); insts.append(inst); off+=inst.size()
  if getattr(inst,"op_name","").lower()=="s_endpgm": break
mc=Counter(getattr(i,"op_name","").lower() for i in insts)
vgpr,_,_,priv=scan_elf_regs(insts,scratch_inst_size)
def patterned(shape,dtype=np.float32):
  count=int(np.prod(shape)); return (((np.arange(count,dtype=np.int32)%127)-63)/64.0).astype(dtype).reshape(shape)
q=Tensor(patterned((1,32,32,128)),device=Device.DEFAULT).realize()
cache=Tensor(patterned((2,1,8,2048,128),np.float16),device=Device.DEFAULT).realize()
runner=TinyJit(lambda qq: flash_attention(qq,cache,2048).realize())
for _ in range(5):
  out=runner(q); Device[Device.DEFAULT].synchronize()
arr=out.numpy(); xs=[]
for _ in range(80):
  t0=time.perf_counter_ns(); runner(q); Device[Device.DEFAULT].synchronize(); xs.append((time.perf_counter_ns()-t0)/1e3)
xs=sorted(xs)
print(json.dumps({"label":os.environ["LABEL"],"priv":priv,"vgpr":vgpr,"ninst":len(insts),"v_mov":mc.get("v_mov_b32_e32",0),
  "delay":mc.get("s_delay_alu",0),"ds_store":sum(v for k,v in mc.items() if k.startswith("ds_store")),
  "sum":float(arr.astype("float64").sum()),"best":xs[0],"p10":xs[8],"median":statistics.median(xs)}))
"""

def run(label: str, extra: dict[str, str]) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
  print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-400:]}", flush=True)

if __name__ == "__main__":
  run("ship", {})
  run("swz", {"AMD_SWIZZLE_DELAY": "1"})
  run("no_wmma_dly", {"AMD_WMMA_DELAY": "0"})
  run("pre_dly", {"AMD_WMMA_DELAY_PRE": "1"})
  run("acc140", {"AMD_WMMA_ACC_BASE": "140", "AMD_LLOAD_POOL": "130"})
