#!/usr/bin/env python3
"""Remote A/B helper for promote ACC exclusion experiments."""
from __future__ import annotations

import os
import sys

import numpy as np

from tinygrad import Device, Tensor
from tinygrad.codegen import to_program_cache
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import flash_attention


def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def run(skip: str) -> tuple[bool, int, float]:
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  getenv.cache_clear()
  to_program_cache.clear()
  q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
  out = flash_attention(q, cache, 2048).realize().numpy().astype(np.float32)
  ok = bool(np.isfinite(out).all())
  return ok, int(np.isnan(out).sum()), float(out.reshape(-1)[0])


if __name__ == "__main__":
  skip = sys.argv[1] if len(sys.argv) > 1 else "2"
  ok, nan, first = run(skip)
  print(f"skip={skip!r} ok={ok} nan={nan} first={first}")
  raise SystemExit(0 if ok else 1)
