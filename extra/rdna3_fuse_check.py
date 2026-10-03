#!/usr/bin/env python3
import os
from collections import Counter
os.environ.update({"DEV":"AMD:AMD","AMD_FLASH_DIRECT":"1","AMD_SOFT_SYSFS":"1","PARALLEL":"0","CACHELEVEL":"0","CCACHE":"0"})
from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
import tinygrad.renderer.isa.rdna3 as R
DEV.value="AMD:AMD"
_old=R._wmma_ab_vec_loads
stats={"calls":0,"lens":[]}
def wrap(elems):
  stats["calls"]+=1
  got=_old(elems)
  if got is not None: stats["lens"].append(len(got))
  return got
R._wmma_ab_vec_loads=wrap
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
ren=AMDRenderer(Target("AMD",arch="gfx1100"))
with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
  to_program(_amd_flash_attention(
    UOp.placeholder((32,32,128),dtypes.float32,0),
    UOp.placeholder((32,32,128),dtypes.float16,1),
    UOp.placeholder((2,1,8,2048,128),dtypes.float16,2),
    valid_kv_len=2048,acc_small=True,k_unroll=1,k_midstore=8,use_acc_work=False,acc_vgpr=True,hoist_q=True),ren)
print("fuse_lens", stats["calls"], Counter(stats["lens"]))
