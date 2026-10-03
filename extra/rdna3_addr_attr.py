#!/usr/bin/env python3
"""Attribute DIRECT flash addr ALU (add_nc / lshl) after LDS fold-base."""
from __future__ import annotations
import json, os, sys
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
names = [getattr(i, "op_name", "").lower() for i in insts]
mc = Counter(names)
vgpr, _, _, priv = scan_elf_regs(insts, scratch_inst_size)

kinds = Counter()
for i, n in enumerate(names):
  if n != "v_add_nc_u32_e64": continue
  prev = names[i-1] if i else ""
  nxt = names[i+1] if i+1 < len(names) else ""
  if "lshl" in prev: kinds["after_lshl"] += 1
  elif nxt.startswith("ds_"): kinds["before_ds"] += 1
  elif "scratch_" in nxt or "global_" in nxt: kinds["before_mem"] += 1
  elif "add_nc" in prev or "add_nc" in nxt: kinds["add_chain"] += 1
  else: kinds[f"{prev[:28]} -> {nxt[:28]}"] += 1

# pair lshl then add (scale+base pattern still split)
pairs = 0
for i in range(len(names)-1):
  if "lshlrev" in names[i] and names[i+1] == "v_add_nc_u32_e64": pairs += 1

print(json.dumps({
  "priv": priv, "vgpr": vgpr, "ninst": len(insts),
  "v_add_nc": mc.get("v_add_nc_u32_e64", 0),
  "v_lshl": mc.get("v_lshlrev_b32_e64", 0),
  "v_lshl_add": mc.get("v_lshl_add_u32", 0),
  "v_mad_u32": sum(v for k,v in mc.items() if "mad_u32" in k),
  "v_add3": sum(v for k,v in mc.items() if "add3" in k),
  "lshl_then_add_pairs": pairs,
  "add_kinds": dict(kinds.most_common(25)),
}, indent=2))
