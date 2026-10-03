#!/usr/bin/env python3
"""Compile + optional HW A/B for AMD_MOV_ALIAS_EXTRACT."""
from __future__ import annotations
import json, os, sys
from collections import Counter

def compile_metrics(alias: int) -> dict:
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_MOV_ALIAS_EXTRACT"] = str(alias)
  os.environ["AMD_WMMA_INTERLEAVE"] = "0"
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer import Target
  from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
  from tinygrad.renderer.isa import greg
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.renderer.isa.rdna3 import _reg_idxs
  getenv.cache_clear(); to_program_cache.clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    prg = to_program(
      _amd_flash_attention(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=True),
      ren,
    )
  def iop(u):
    return u.arg[0] if u.op is Ops.INS and isinstance(u.arg, tuple) else None
  lin = list(prg.src[1].src)
  ext_mov_diff = 0
  for u in lin:
    if iop(u) is not AMDOps.MOV or not u.src: continue
    s = u.src[0]
    if not (s.op is Ops.INS and iop(s) is AMDOps.EXTRACT): continue
    ru, re = greg(u), greg(s)
    if ru is None or re is None or ru.index != re.index: ext_mov_diff += 1
  mov_emit = 0
  for u in lin:
    if iop(u) is AMDOps.MOV and u.src and u.src[0].op is not Ops.SPECIAL:
      sregs, dregs = _reg_idxs(u.src[0]), _reg_idxs(u)
      if sregs and sregs != dregs: mov_emit += 1
  text = next(s.content for s in elf_loader(prg.src[3].arg)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  spills = sum(1 for u in lin if iop(u) is AMDOps.SPILL)
  return {
    "alias": alias,
    "ninst": len(insts),
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "scratch_st": sum(v for k, v in mc.items() if k.startswith("scratch_store")),
    "ir_spill": spills,
    "ext_mov_diff_reg": ext_mov_diff,
    "mov_emit": mov_emit,
  }

def main():
  rows = [compile_metrics(0), compile_metrics(1)]
  print(json.dumps({"rows": rows}, indent=2))
  off, on = rows
  print("DELTA", {k: on[k] - off[k] for k in off if isinstance(off[k], (int, float))})
  bad = []
  if on["ir_spill"] > off["ir_spill"]: bad.append("spill")
  if on["scratch_st"] > off["scratch_st"]: bad.append("scratch")
  if on["v_mov"] >= off["v_mov"]: bad.append("no_vmov_win")
  print("GATE", "FAIL" if bad else "PASS", bad)
  return 1 if bad else 0

if __name__ == "__main__":
  sys.exit(main())
