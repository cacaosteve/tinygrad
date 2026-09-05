#!/usr/bin/env python3
"""Classify flash SPILL/FILL sources (slot-2 scratch is separate)."""
from __future__ import annotations

import os
from collections import Counter

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Context, getenv
from tinygrad.llm.kernels.amd import _amd_flash_attention
from tinygrad.renderer import Target
from tinygrad.renderer.isa import greg
from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, _iop
from tinygrad.uop.ops import Ops, UOp
from test.amd.test_amd_renderer import _prg_lin


def main() -> None:
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  getenv.cache_clear()
  out = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  sink = _amd_flash_attention(out, q, kv, valid_kv_len=2048, q_start=None, acc_small=True, k_unroll=0)
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with Context(BEAM=0):
    prg = to_program(sink, ren)
  lin = _prg_lin(prg)
  spill_tags: Counter[str] = Counter()
  for u in lin.src:
    if u.op is not Ops.INS or _iop(u) is not AMDOps.SPILL:
      continue
    src = u.src[0] if u.src else None
    r = greg(src) if src is not None else None
    tag = getattr(src, "tag", None)
    spill_tags[f"{r}|{tag}"] += 1
    print("SPILL", r, tag)
  print("SPILL summary", spill_tags)
  print("SPILL", sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.SPILL),
        "FILL", sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.FILL),
        "SLOAD", sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.SLOAD),
        "SSTORE", sum(1 for u in lin.src if u.op is Ops.INS and _iop(u) is AMDOps.SSTORE))
  os.environ.pop("AMD_FLASH_ACC_SMALL", None)
  getenv.cache_clear()


if __name__ == "__main__":
  main()
