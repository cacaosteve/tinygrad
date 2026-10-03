#!/usr/bin/env python3
"""Capture vdef trees when ADD first enters self.spills."""
import os
os.environ.update({
  "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "PARALLEL": "0",
  "CACHELEVEL": "0", "AMD_SOFT_SYSFS": "1",
})
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
import tinygrad.renderer.isa.rdna3 as r
import tinygrad.codegen.late.regalloc as ra

_orig_fill = None
seen = []

def patch():
  # Patch LinearScan after import — hook via wrapping AMDRenderer methods consulted in fill
  pass

# Simpler: wrap rematerialize and when spill is about to happen via stack_size growth, dump
# Actually monkeypatch wide_restore and watch spills dict — wrap fill by patching after class create

from tinygrad.codegen.late.regalloc import LinearScanRegallocContext

_orig_init = LinearScanRegallocContext.__init__ if hasattr(LinearScanRegallocContext, "__init__") else None

# Patch ren.spill_size path — intercept by wrapping Renderer after_pre and then
# wrapping the LinearScan.allocate if exists

# Easiest: after to_program, we already know. Instead hook AMDRenderer.spill and also
# print vdef from a patched fill by replacing ra module function... 

# Patch LinearScanRegallocContext by wrapping the method that creates spills.
# Looking at code, fill is nested inside allocate(). Hook ren.spill_size + check spills.

_orig_spill = AMDRenderer.spill
def spill(self, disp, x, slots=None):
  def peel(u, d=0):
    if d > 5: return "..."
    if u.op is Ops.CONST: return f"C({u.arg})"
    if u.op is Ops.SPECIAL: return f"S({u.arg})"
    if u.op is Ops.CAST: return f"CAST({peel(u.src[0], d+1)})"
    if u.op is Ops.INS:
      iop = r._iop(u)
      if iop is AMDOps.MOV:
        if not u.src: return "bind"
        return f"MOV({peel(u.src[0], d+1)})"
      return f"{str(iop).split('.')[-1]}(" + ",".join(peel(s, d+1) for s in u.src) + ")"
    return str(u.op)
  seen.append({"emit_spill": peel(x), "can": self.rematerialize(x), "shallow": self._addr_shallow(x), "pure": self._pure_addr(x)})
  return _orig_spill(self, disp, x, slots)

# Also wrap rematerialize to log when ADD gets False despite being the spill candidate's vdef
AMDRenderer.spill = spill

DEV.value = "AMD:AMD"
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()

# Instrument LinearScan by patching wide_restore's spill path — patch fill via
# replacing ren.rematerialize to record stack when False for ADD
_orig_r = AMDRenderer.rematerialize
def rematerialize(self, x):
  ret = _orig_r(self, x)
  return ret
AMDRenderer.rematerialize = rematerialize

with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  to_program(_amd_flash_attention(
    UOp.placeholder((32, 32, 128), dtypes.float32, 0),
    UOp.placeholder((32, 32, 128), dtypes.float16, 1),
    UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
    valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
    use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)

for s in seen:
  print(s)
