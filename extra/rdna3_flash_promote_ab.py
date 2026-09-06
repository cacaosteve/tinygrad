#!/usr/bin/env python3
"""Fresh-process-friendly A/B for flash slot-2 promote vs SKIP=2 / SDPA.

Compares **every element** (maxdiff), not just finite/first. Example:

  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_ab.py
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_ab.py --S 256
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from tinygrad import Device, Tensor
from tinygrad.codegen import to_program_cache
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import flash_attention
import tinygrad.llm.kernels.amd as amd


def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def run(*, skip: str, work: int, S: int, T: int, direct: int) -> np.ndarray:
  os.environ["AMD_FLASH_DIRECT"] = str(direct)
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  os.environ["AMD_FLASH_ACC_WORK"] = str(work)
  os.environ["AMD_SPILL_ON_EVICT"] = "0"
  getenv.cache_clear()
  to_program_cache.clear()
  amd._amd_flash_attention.cache_clear()
  q = Tensor(patterned((1, 32, T, 128)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, 1, 8, S, 128), np.float16), device=Device.DEFAULT).realize()
  out = flash_attention(q, cache, S).realize()
  Device[Device.DEFAULT].synchronize()
  return out.numpy().astype(np.float32)


def maxdiff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, bool]:
  if not np.isfinite(a).all() or not np.isfinite(b).all():
    return float("nan"), float("nan"), False
  d = np.abs(a - b)
  return float(d.max()), float(d.mean()), True


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--S", type=int, default=256, help="KV length (tiles = ceil(S/32))")
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--tol", type=float, default=2e-4, help="maxdiff vs SDPA (fail the run if exceeded)")
  args = p.parse_args()
  T = min(args.T, args.S)
  print("device", Device.DEFAULT, Device[Device.DEFAULT].renderer.__class__.__name__)
  skip2 = run(skip="2", work=1, S=args.S, T=T, direct=1)
  prom = run(skip="", work=0, S=args.S, T=T, direct=1)
  sdpa = run(skip="2", work=1, S=args.S, T=T, direct=0)
  results: dict[str, tuple[float, float, bool]] = {}
  for name, a, b in (
    ("prom_vs_skip2", prom, skip2),
    ("skip2_vs_sdpa", skip2, sdpa),
    ("prom_vs_sdpa", prom, sdpa),
  ):
    mx, mn, ok = maxdiff(a, b)
    results[name] = (mx, mn, ok)
    print(f"{name}: ok={ok} maxdiff={mx:.6g} mean={mn:.6g}")
  prom_ok = np.isfinite(prom).all() and results["prom_vs_skip2"][2] and results["prom_vs_skip2"][0] == 0.0
  sdpa_ok = (
    results["skip2_vs_sdpa"][2] and results["skip2_vs_sdpa"][0] <= args.tol and
    results["prom_vs_sdpa"][2] and results["prom_vs_sdpa"][0] <= args.tol
  )
  if not prom_ok:
    print("FAIL: promote must match SKIP=2 exactly (and stay finite)")
  if not sdpa_ok:
    print(f"FAIL: vs SDPA maxdiff must be <= {args.tol:g}")
  raise SystemExit(0 if prom_ok and sdpa_ok else 1)


if __name__ == "__main__":
  main()
