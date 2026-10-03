#!/usr/bin/env python3
"""DIRECT vs HIP structural + fair A/B for pack-chain soft / delay."""
from __future__ import annotations
import json, os, subprocess, sys
from collections import Counter

ROOT = "/home/admin441766/tinygrad"
PY = f"{ROOT}/venv/bin/python"

STRUCT = r"""
import json,os,collections
from tinygrad import dtypes
from tinygrad.helpers import Context,getenv,Target,DEV
from tinygrad.uop.ops import UOp
from tinygrad.codegen import to_program,to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.llm.kernels.amd import _amd_flash_attention,_flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs,scratch_inst_size
DEV.value=os.environ["DEV"]
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
ren=AMDRenderer(Target("AMD",arch="gfx1100"))
direct=os.environ.get("AMD_FLASH_DIRECT")=="1"
cm = _flash_direct_compile_env(flash_acc_small=True) if direct else Context(BEAM=0)
with cm, Context(BEAM=0):
  prg=to_program(_amd_flash_attention(
    UOp.placeholder((32,32,128),dtypes.float32,0),
    UOp.placeholder((32,32,128),dtypes.float16,1),
    UOp.placeholder((2,1,8,2048,128),dtypes.float16,2),
    valid_kv_len=2048,acc_small=True,k_unroll=1,k_midstore=8,use_acc_work=False,acc_vgpr=True,hoist_q=True),ren)
text=next(s.content for s in elf_loader(prg.src[-1].arg)[1] if s.name==".text")
insts=[]; off=0
while off<len(text):
  inst=decode_inst(text[off:],"rdna3"); insts.append(inst); off+=inst.size()
  if getattr(inst,"op_name","").lower()=="s_endpgm": break
names=[getattr(i,"op_name","").lower() for i in insts]
mc=collections.Counter(names)
vgpr,_,_,priv=scan_elf_regs(insts,scratch_inst_size)
delay_prev=collections.Counter(); delay_next=collections.Counter()
for i,n in enumerate(names):
  if n=="s_delay_alu":
    delay_prev[names[i-1] if i else "START"]+=1
    delay_next[names[i+1] if i+1<len(names) else "END"]+=1
print(json.dumps({"label":os.environ["LABEL"],"priv":priv,"vgpr":vgpr,"ninst":len(insts),
  "v_mov":mc.get("v_mov_b32_e32",0),"dual_mov":sum(v for k,v in mc.items() if "dual_mov" in k),
  "dual_all":sum(v for k,v in mc.items() if "dual" in k),"delay":mc.get("s_delay_alu",0),
  "wmma":mc.get("v_wmma_f32_16x16x16_f16",0),"fmac":mc.get("v_fmac_f32_e32",0)+mc.get("v_fmac_f32_e64",0),
  "ds_store":sum(v for k,v in mc.items() if k.startswith("ds_store")),
  "ds_load":sum(v for k,v in mc.items() if k.startswith("ds_load")),
  "gload":sum(v for k,v in mc.items() if "global_load" in k),
  "delay_prev":dict(delay_prev.most_common(6)),"delay_next":dict(delay_next.most_common(6))}))
"""

def run_struct(label: str, dev: str, direct: bool, extra: dict|None=None) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "CCACHE": "0", "LABEL": label, "DEV": dev})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS",):
      env.pop(k, None)
  if direct: env["AMD_FLASH_DIRECT"] = "1"
  if extra: env.update({k: str(v) for k, v in extra.items()})
  p = subprocess.run([PY, "-c", STRUCT], env=env, capture_output=True, text=True, cwd=ROOT, timeout=180)
  print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-400:]}", flush=True)

if __name__ == "__main__":
  run_struct("direct", "AMD:AMD", True)
  run_struct("hip", "AMD", False)
  # Fair knobs via fair_ab_knobs CHILD
  child = open(f"{ROOT}/extra/rdna3_fair_ab_knobs.py").read().split('CHILD = r"""')[1].split('"""')[0]
  def fair(label, extra):
    env = os.environ.copy()
    env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
                "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
    for k in list(env):
      if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
        env.pop(k, None)
    env.update(extra)
    p = subprocess.run([PY, "-c", child], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
    print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-350:]}", flush=True)
  for label, extra in [
    ("ship", {}),
    ("soft_chain", {"AMD_PACK_LLOAD_SOFT": "1"}),
    ("no_align8", {"AMD_PACK_LLOAD_ALIGN8": "0"}),
    ("pool145", {"AMD_LLOAD_POOL": "145"}),
    ("acc138", {"AMD_WMMA_ACC_BASE": "138", "AMD_LLOAD_POOL": "130"}),
  ]:
    fair(label, extra)
