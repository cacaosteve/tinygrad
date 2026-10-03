#!/usr/bin/env python3
"""Log remat/keep at ADD def time and at spill time."""
import os
os.environ.update({
  "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "PARALLEL": "0",
  "CACHELEVEL": "0", "AMD_SOFT_SYSFS": "1", "AMD_REMAT_BIND_ADD": "1",
})
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
import tinygrad.renderer.isa.rdna3 as r

_orig_remat = AMDRenderer.rematerialize
_orig_keep = AMDRenderer.keep_remat
_orig_spill = AMDRenderer.spill
logs = []

def rematerialize(self, x):
  ret = _orig_remat(self, x)
  if x.op is Ops.INS and r._iop(x) is AMDOps.ADD and x.dtype in (dtypes.int32, dtypes.uint32, dtypes.uint):
    binds = all(self._is_bind_mov(s) for s in x.src) if x.src else False
    leaves = all(self._addr_leaf(s) for s in x.src) if x.src else False
    logs.append(("remat", ret, "binds", binds, "leaves", leaves, "pure", self._pure_addr(x),
                 "nsrc", len(x.src), "src0", r._iop(x.src[0]) if x.src and x.src[0].op is Ops.INS else x.src[0].op if x.src else None))
  return ret

def keep_remat(self, x):
  ret = _orig_keep(self, x)
  if x.op is Ops.INS and r._iop(x) is AMDOps.ADD and x.dtype in (dtypes.int32, dtypes.uint32, dtypes.uint):
    binds = all(self._is_bind_mov(s) for s in x.src) if x.src else False
    logs.append(("keep", ret, "binds", binds))
  return ret

def spill(self, disp, x, slots=None):
  logs.append(("spill", r._iop(x) if x.op is Ops.INS else x.op,
               "remat", self.rematerialize(x), "keep", self.keep_remat(x),
               "binds", all(self._is_bind_mov(s) for s in x.src) if x.op is Ops.INS and x.src else None))
  return _orig_spill(self, disp, x, slots)

AMDRenderer.rematerialize = rematerialize
AMDRenderer.keep_remat = keep_remat
AMDRenderer.spill = spill

DEV.value = "AMD:AMD"
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  to_program(_amd_flash_attention(
    UOp.placeholder((32, 32, 128), dtypes.float32, 0),
    UOp.placeholder((32, 32, 128), dtypes.float16, 1),
    UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
    valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
    use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)

# summarize
from collections import Counter
c = Counter(t[0] for t in logs)
print("counts", dict(c))
print("spill events:")
for t in logs:
  if t[0] == "spill": print(t)
print("sample remat False:")
n = 0
for t in logs:
  if t[0] == "remat" and t[1] is False:
    print(t); n += 1
    if n >= 6: break
print("sample remat True:")
n = 0
for t in logs:
  if t[0] == "remat" and t[1] is True:
    print(t); n += 1
    if n >= 6: break
