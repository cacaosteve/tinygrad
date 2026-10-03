#!/usr/bin/env python3
"""Attribute remaining SPILL/FILL under shipping ACC_VGPR flash prefill."""
from __future__ import annotations
import json, os
from collections import Counter

os.environ.update({
  "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1",
  "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0", "AMD_SOFT_SYSFS": "1",
})

from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.renderer.isa import greg
from tinygrad.renderer.isa.rdna3_defs import Register
import tinygrad.renderer.isa.rdna3 as R

def iop(u):
  return u.arg[0] if u.op is Ops.INS and isinstance(u.arg, tuple) else None

def describe(x, depth=0):
  if x is None: return "None"
  if x.op is Ops.CONST: return f"CONST:{x.arg}"
  if x.op is Ops.SPECIAL: return f"SPECIAL:{x.arg}"
  if x.op is Ops.CAST:
    return f"CAST({x.dtype})<-{describe(x.src[0], depth+1) if x.src else '?'}"
  op = iop(x)
  if op is None: return f"{x.op}:{x.dtype}"
  if depth > 4: return f"{op.name}..."
  regs = []
  try:
    g = greg(x)
    if isinstance(g, Register): regs = [f"v{g.index}"]
    elif isinstance(g, (list, tuple)):
      regs = [f"v{r.index}" if isinstance(r, Register) else str(r) for r in list(g)[:4]]
  except Exception:
    pass
  kids = ",".join(describe(s, depth+1) for s in x.src[:3]) if x.src else ""
  return f"{op.name}[{'|'.join(regs)}]({kids})"

def main():
  DEV.value = "AMD:AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  captured: dict = {}
  _orig = R.insts_from_linear
  def wrap(lin_u):
    captured["ops"] = list(lin_u.src)
    return _orig(lin_u)
  R.insts_from_linear = wrap
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    to_program(_amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
      use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)
  ops = captured["ops"]
  uses: dict[int, list] = {}
  for u in ops:
    for s in u.src: uses.setdefault(id(s), []).append(u)
  spills = [(i, u) for i, u in enumerate(ops) if iop(u) is AMDOps.SPILL]
  fills = [(i, u) for i, u in enumerate(ops) if iop(u) is AMDOps.FILL]
  rows = []
  for i, u in spills:
    # SPILL src=(disp_offset, value[, slots])
    disp = u.src[0] if u.src else None
    val = u.src[1] if u.src and len(u.src) > 1 else None
    cons = []
    for c in uses.get(id(u), [])[:6]:
      cons.append(iop(c).name if iop(c) else c.op.name)
    win = []
    for j in range(max(0, i-3), min(len(ops), i+4)):
      oj = ops[j]
      win.append(f"{j}:{'SPILL' if j==i else (iop(oj).name if iop(oj) else oj.op.name)}")
    rows.append({
      "i": i, "dtype": str(u.dtype),
      "disp": describe(disp), "val": describe(val),
      "val_dtype": str(val.dtype) if val is not None else None,
      "consumers": cons, "arg": str(u.arg)[:160], "window": win,
    })
  fill_rows = []
  for i, u in fills:
    # FILL src=(disp, dtype_witness) or similar — dump all srcs
    fill_rows.append({
      "i": i, "dtype": str(u.dtype),
      "srcs": [describe(s) for s in u.src],
      "arg": str(u.arg)[:160],
    })
  out = {
    "spill": len(spills), "fill": len(fills),
    "spills": rows, "fills": fill_rows,
    "env": {k: os.environ.get(k) for k in (
      "AMD_REMAT_ADDR", "AMD_REMAT_ADDR_SHALLOW", "AMD_REMAT_ADDR_DEEP",
      "AMD_REG_PROMOTE_SKIP_SLOTS", "AMD_FLASH_ACC_VGPR", "AMD_REMAT_CONST_MOV")},
  }
  print(json.dumps(out, indent=2))

if __name__ == "__main__":
  main()
