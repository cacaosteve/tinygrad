#!/usr/bin/env python3
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

_orig = AMDRenderer.spill

def spill(self, disp, x, slots=None):
  def peel(u):
    if u is None: return "None"
    if u.op is Ops.CONST: return f"CONST({u.arg})"
    if u.op is Ops.SPECIAL: return f"SPECIAL({u.arg})"
    if u.op is Ops.CAST: return f"CAST({peel(u.src[0])})"
    if u.op is Ops.INS:
      iop = r._iop(u)
      if iop is AMDOps.MOV:
        if not u.src: return "MOV(empty)"
        return f"MOV({peel(u.src[0])})"
      return f"{iop}(nsrc={len(u.src)},dt={u.dtype})"
    return f"{u.op}/dt={getattr(u,'dtype',None)}/arg={getattr(u,'arg',None)}"
  can = self.rematerialize(x)
  pure = self._pure_addr(x)
  shallow = self._addr_shallow(x)
  leaves = all(self._addr_leaf(s) for s in x.src) if x.op is Ops.INS else False
  keep = self.keep_remat(x)
  print({"can_remat": can, "pure": pure, "shallow": shallow, "leaves": leaves, "keep": keep,
         "tree": f"ADD({peel(x.src[0])},{peel(x.src[1])})"})
  return _orig(self, disp, x, slots)

AMDRenderer.spill = spill
DEV.value = "AMD:AMD"
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  print("DEEP", getenv("AMD_REMAT_ADDR_DEEP"), "SHALLOW", getenv("AMD_REMAT_ADDR_SHALLOW"),
        "ADDR", getenv("AMD_REMAT_ADDR"))
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  to_program(_amd_flash_attention(
    UOp.placeholder((32, 32, 128), dtypes.float32, 0),
    UOp.placeholder((32, 32, 128), dtypes.float16, 1),
    UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
    valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
    use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)
