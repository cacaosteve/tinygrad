#!/usr/bin/env python3
"""Compile-path metrics for AMD_WMMA_ACC_BASE × promote/remat (fresh process)."""
from __future__ import annotations
import os, sys
from collections import Counter
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
from tinygrad.renderer.isa.rdna3 import AMDOps, _iop
from tinygrad.renderer.isa import rdna3_defs
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd

def main():
  skip = os.environ.get("AMD_FLASH_PREFILL_SKIP_SLOTS", "2")
  # Match flash_attention: ACC_WORK defaults on whenever ACC_SMALL (tile-local work copy).
  use_aw = os.environ.get("AMD_FLASH_ACC_WORK", "1") != "0"
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    # Re-apply SKIP inside env like flash_attention does
    if "AMD_FLASH_PREFILL_SKIP_SLOTS" in os.environ:
      os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = os.environ["AMD_FLASH_PREFILL_SKIP_SLOTS"]
    elif getenv("AMD_FLASH_ACC_LDS", 0):
      os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
    else:
      os.environ.setdefault("AMD_REG_PROMOTE_SKIP_SLOTS", "2")
    getenv.cache_clear()
    prg = to_program(_amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=use_aw,
      acc_lds=bool(getenv("AMD_FLASH_ACC_LDS", 0))), ren)
  lin = next(s for s in prg.src if s.op is Ops.LINEAR)
  spill = sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.SPILL)
  fill = sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.FILL)
  binary = prg.src[-1].arg
  text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3")
    insts.append(inst)
    off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, _, _, priv = scan_elf_regs(insts, scratch_inst_size)
  print({
    "ACC_BASE_env": os.environ.get("AMD_WMMA_ACC_BASE"),
    "ACC0": rdna3_defs.WMMA_ACC_VGPR[0].name,
    "spill": spill, "fill": fill, "priv": priv, "vgpr": vgpr,
    "mov": mc.get("v_mov_b32_e32", 0), "ninst": len(insts),
    "skip": skip, "use_aw": use_aw,
  })

if __name__ == "__main__":
  main()
