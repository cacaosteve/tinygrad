#!/usr/bin/env python3
"""Attribute SPILL/FILL in DIRECT flash prefill asm."""
from __future__ import annotations

import os
import re

import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import DEV, getenv
from tinygrad.llm.kernels.amd import flash_attention


def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def main() -> None:
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  DEV.value = "AMD:AMD"
  getenv.cache_clear()
  _ = Device["AMD"].renderer

  q = Tensor(patterned((1, 32, 32, 128)), device="AMD").realize()
  cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device="AMD").realize()
  runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
  for _ in range(3):
    runner(q)
    Device["AMD"].synchronize()
  assert runner.captured is not None
  for j in runner.captured.jit_cache:
    prg = j.prg
    p = getattr(prg, "p", None) or getattr(prg, "prg", None)
    if p is None:
      continue
    src = getattr(p, "src", "") or ""
    name = getattr(p, "function_name", None) or getattr(p, "name", "?")
    n_spill = src.count("SPILL")
    n_wmma = len(re.findall(r"\bWMMA\b", src))
    if n_spill == 0 and n_wmma < 3:
      continue
    print(f"PROGRAM {name} spills={n_spill} fills={src.count('FILL')} wmma={n_wmma} "
          f"priv={getattr(p, 'private_segment_size', None)} vgpr={getattr(p, 'max_vgpr', None)}")
    # Show SPILL context: previous DEFINE/ops mentioning spilled value
    lines = src.splitlines()
    for i, ln in enumerate(lines):
      if "SPILL" not in ln and "FILL" not in ln:
        continue
      lo, hi = max(0, i - 3), min(len(lines), i + 2)
      print(f"--- @{i} ---")
      for k in range(lo, hi):
        mark = ">>" if k == i else "  "
        print(f"{mark}{lines[k][:180]}")


if __name__ == "__main__":
  main()
