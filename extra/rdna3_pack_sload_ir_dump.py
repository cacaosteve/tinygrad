#!/usr/bin/env python3
"""Dump soft-copy / SLOAD pack IR for flash S=32 T=32, MAX=8 vs 9."""
from __future__ import annotations

import os
from collections import Counter

from tinygrad import dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Context, getenv
from tinygrad.llm.kernels.amd import _amd_flash_attention
from tinygrad.renderer import Target
from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, _const_int, _elem_count, _iop
from tinygrad.uop.ops import Ops, UOp
from test.amd.test_amd_renderer import _amd_inst_names, _prg_lin


def sink_for() -> UOp:
  # Match minimized HW corrupt case: physical_n=32, tokens=32.
  out = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 32, 128), dtypes.float16, 2)
  return _amd_flash_attention(out, q, kv, valid_kv_len=32, q_start=None, acc_small=True, k_unroll=0)


def summarize(max_packs: int) -> None:
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = "2"
  os.environ["AMD_FLASH_ACC_WORK"] = "1"
  os.environ["AMD_SPILL_ON_EVICT"] = "0"
  os.environ["AMD_CLUSTER_SLOAD"] = "1"
  os.environ["AMD_PACK_SLOAD_B128"] = "1"
  os.environ["AMD_PACK_SLOAD_MAX"] = str(max_packs)
  getenv.cache_clear()
  to_program_cache.clear()
  _amd_flash_attention.cache_clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with Context(BEAM=0):
    prg = to_program(sink_for(), ren)
  lin_ops = [u for u in _prg_lin(prg).src if u.op is Ops.INS]
  names = _amd_inst_names(prg)

  sloads = [u for u in lin_ops if _iop(u) is AMDOps.SLOAD]
  wide = [u for u in sloads if _elem_count(u) == 4]
  scalar = [u for u in sloads if _elem_count(u) == 1]
  extracts = sum(1 for u in lin_ops if _iop(u) is AMDOps.EXTRACT)
  spills = sum(1 for u in lin_ops if _iop(u) is AMDOps.SPILL)
  fills = sum(1 for u in lin_ops if _iop(u) is AMDOps.FILL)
  sstores = sum(1 for u in lin_ops if _iop(u) is AMDOps.SSTORE)

  print(f"\n=== PACK_MAX={max_packs} ===")
  print(f"SLOAD wide×4={len(wide)} scalar={len(scalar)} EXTRACT={extracts} SPILL={spills} FILL={fills} SSTORE={sstores}")
  scratch = [n for n in names if "scratch" in n.lower() or "buffer_load" in n.lower() or "buffer_store" in n.lower()]
  print("scratch/buffer top:", Counter(scratch).most_common(12))
  print("wide SLOAD const idxs:", [_const_int(u.src[1]) for u in wide])

  wi = 0
  for i, u in enumerate(lin_ops):
    if not (u.op is Ops.INS and _iop(u) is AMDOps.SLOAD and _elem_count(u) == 4):
      continue
    wi += 1
    print(f"  pack#{wi} @{i} idx={_const_int(u.src[1])} tag={u.tag[0] if u.tag else None}")
    for j in range(i + 1, min(i + 8, len(lin_ops))):
      v = lin_ops[j]
      if v.op is Ops.INS and _iop(v) is AMDOps.EXTRACT and v.src and v.src[0] is u:
        lane = _const_int(v.src[1]) if len(v.src) > 1 else "?"
        print(f"    EXTRACT lane={lane} tag={v.tag[0] if v.tag else None}")
    if wi >= 12:
      break


if __name__ == "__main__":
  for m in (8, 9):
    summarize(m)
