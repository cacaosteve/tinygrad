#!/usr/bin/env python3
"""Count how many FMAC↔ADD pairs fail each dual_fmac gate."""
import os
os.environ.update({"DEV":"AMD:AMD","AMD_FLASH_DIRECT":"1","AMD_SOFT_SYSFS":"1",
                   "PARALLEL":"0","CACHELEVEL":"0","AMD_SCHEDULE_FMAC_ADD":"1"})
from collections import Counter
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.renderer.isa.rdna3 import _iop

DEV.value = "AMD:AMD"
stats = Counter()

# Hook after_pre_regalloc to count FMAC/ADD adjacency after schedule
_apr = AMDRenderer.after_pre_regalloc
def apr(self, lst):
  out = _apr(self, lst)
  fmacs = [i for i,u in enumerate(out) if u.op is Ops.INS and _iop(u) is AMDOps.FMAC]
  adds = [i for i,u in enumerate(out) if u.op is Ops.INS and _iop(u) is AMDOps.ADD and u.dtype is dtypes.float32]
  stats["n_fmac"] = len(fmacs)
  stats["n_add_f32"] = len(adds)
  # For each FMAC, find nearest independent ADD within 32
  near = indep = dep_mid = 0
  for fi in fmacs:
    u = out[fi]
    for j in range(fi+1, min(fi+33, len(out))):
      cand = out[j]
      if not (cand.op is Ops.INS and _iop(cand) is AMDOps.ADD and cand.dtype is dtypes.float32):
        continue
      near += 1
      if u in cand.toposort() or cand in u.toposort():
        continue
      if any(out[m] in cand.toposort() for m in range(fi+1, j)):
        dep_mid += 1
        continue
      indep += 1
      break
  stats["fmac_near_add"] = near
  stats["fmac_indep_add"] = indep
  stats["fmac_add_dep_mid"] = dep_mid
  return out
AMDRenderer.after_pre_regalloc = apr

getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
ren = AMDRenderer(Target("AMD", arch="gfx1100"))
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  to_program(_amd_flash_attention(
    UOp.placeholder((32,32,128),dtypes.float32,0),
    UOp.placeholder((32,32,128),dtypes.float16,1),
    UOp.placeholder((2,1,8,2048,128),dtypes.float16,2),
    valid_kv_len=2048,acc_small=True,k_unroll=1,k_midstore=8,use_acc_work=False,acc_vgpr=True,hoist_q=True),ren)
print(dict(stats))
