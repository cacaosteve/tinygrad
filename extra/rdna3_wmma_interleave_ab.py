#!/usr/bin/env python3
"""Compile-path A/B for AMD_WMMA_INTERLEAVE on DIRECT flash prefill."""
from __future__ import annotations

import json, os, sys
from collections import Counter

def metrics(interleave: int) -> dict:
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_WMMA_INTERLEAVE"] = str(interleave)
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer import Target
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  getenv.cache_clear()
  to_program_cache.clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    prg = to_program(
      _amd_flash_attention(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=True),
      ren,
    )
  text = next(s.content for s in elf_loader(prg.src[3].arg)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3")
    insts.append(inst)
    off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  names = [getattr(i, "op_name", "").lower() for i in insts]
  mc = Counter(names)
  w = [i for i, n in enumerate(names) if "wmma" in n]
  gaps = []
  for a, b in zip(w, w[1:]):
    win = names[a + 1 : b]
    gaps.append((
      b - a - 1,
      sum(1 for n in win if "load" in n),
      sum(1 for n in win if "v_mov" in n),
    ))
  lin = prg.src[1]
  spills = sum(1 for u in lin.src if u.op.name == "INS" and "SPILL" in str(u.arg[0]))
  fills = sum(1 for u in lin.src if u.op.name == "INS" and "FILL" in str(u.arg[0]))
  # peak approx: max VGPR index mentioned in LINEAR regs if available
  return {
    "interleave": interleave,
    "ninst": len(insts),
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "wmma": sum(v for k, v in mc.items() if "wmma" in k),
    "scratch_st": sum(v for k, v in mc.items() if k.startswith("scratch_store")),
    "scratch_ld": sum(v for k, v in mc.items() if k.startswith("scratch_load")),
    "dual": sum(v for k, v in mc.items() if "dual" in k),
    "delay_alu": sum(v for k, v in mc.items() if "delay_alu" in k),
    "wmma_b2b": sum(1 for g, _, _ in gaps if g == 0),
    "wmma_gaps": len(gaps),
    "gaps_with_load": sum(1 for _, l, _ in gaps if l),
    "loads_before_first_wmma": sum(1 for n in names[: w[0]] if "load" in n) if w else None,
    "ir_spill": spills,
    "ir_fill": fills,
    "linear_n": len(lin.src),
  }

def main():
  rows = [metrics(0), metrics(1)]
  print(json.dumps({"rows": rows}, indent=2))
  off, on = rows
  print("DELTA", json.dumps({
    k: on[k] - off[k] for k in off if isinstance(off[k], (int, float)) and on[k] is not None
  }, indent=2))
  # Gate: reject if spills rise or scratch rises materially
  bad = []
  if on["scratch_st"] > off["scratch_st"]: bad.append("scratch_st")
  if on["ir_spill"] > off["ir_spill"]: bad.append("ir_spill")
  if on["wmma_b2b"] >= off["wmma_b2b"] and on["gaps_with_load"] <= off["gaps_with_load"]:
    bad.append("no_interleave_effect")
  print("GATE", "FAIL" if bad else "PASS", bad)
  return 1 if bad else 0

if __name__ == "__main__":
  sys.exit(main())
