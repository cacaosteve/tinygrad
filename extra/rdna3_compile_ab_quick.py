#!/usr/bin/env python3
"""Quick compile-only A/B for flash DIRECT asm stats."""
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
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
  prg = to_program(_amd_flash_attention(
    UOp.placeholder((32, 32, 128), dtypes.float32, 0),
    UOp.placeholder((32, 32, 128), dtypes.float16, 1),
    UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
    valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
    use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)
text = next(s.content for s in elf_loader(prg.src[-1].arg)[1] if s.name == ".text")
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
  "scratch": sum(v for k, v in mc.items() if "scratch_" in k),
  "ds_load": sum(v for k, v in mc.items() if "ds_load" in k),
  "ds_store": sum(v for k, v in mc.items() if "ds_store" in k),
  "global_load": sum(v for k, v in mc.items() if "global_load" in k),
  "wmma": sum(v for k, v in mc.items() if "wmma" in k),
  "delay": sum(v for k, v in mc.items() if "delay" in k),
}))
'''

CFGS = [
  ("ship", {}),
  ("remat_addr0", {"AMD_REMAT_ADDR": "0"}),
  ("const_remat1", {"AMD_REMAT_CONST_MOV": "1"}),
  ("reserve1", {"AMD_VOPD_SRC1_RESERVE": "1"}),
  ("mul_str0", {"AMD_MUL_STRENGTH": "0"}),
]

def main():
  for label, extra in CFGS:
    env = os.environ.copy()
    env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "LABEL": label, "EXTRA": json.dumps(extra)})
    p = subprocess.run([PY, "-c", COMPILE], env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
    print(p.stdout.strip().splitlines()[-1] if p.returncode == 0 else "FAIL " + label + " " + (p.stderr or "")[-300:].replace("\n", " "), flush=True)

if __name__ == "__main__":
  main()
