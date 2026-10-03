#!/usr/bin/env python3
"""Compare ship vs cluster ELF: waitcnt + checksum-sensitive schedule."""
from __future__ import annotations
import os, subprocess, sys
from collections import Counter

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import json, os
from collections import Counter
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

getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
DEV.value = "AMD:AMD"
ren = AMDRenderer(Target("AMD", arch="gfx1100"))
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
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
wait = [str(i) for i in insts if "wait" in getattr(i, "op_name", "").lower()]
print(json.dumps({
  "label": os.environ["LABEL"],
  "priv": priv, "vgpr": vgpr, "ninst": len(insts),
  "v_mov": mc.get("v_mov_b32_e32", 0),
  "s_waitcnt": mc.get("s_waitcnt", 0),
  "s_clause": sum(v for k,v in mc.items() if "clause" in k),
  "ds_load": sum(v for k,v in mc.items() if "ds_load" in k),
  "wait_sample": wait[:8],
  "wait_n": len(wait),
}))
'''

def run(label: str, extra: dict) -> dict:
  import json
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "CCACHE": "0", "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, cwd=ROOT, timeout=120)
  line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else None
  if not line:
    print("FAIL", label, (p.stderr or "")[-400:])
    return {}
  print(line)
  return json.loads(line)

if __name__ == "__main__":
  a = run("ship", {})
  b = run("cluster", {"AMD_CLUSTER_PACK_LLOAD": "1"})
  if a and b:
    for k in ("priv", "vgpr", "ninst", "v_mov", "s_waitcnt", "s_clause", "ds_load", "wait_n"):
      print(f"delta_{k}", b.get(k), "-", a.get(k), "=", (b.get(k) or 0) - (a.get(k) or 0))
