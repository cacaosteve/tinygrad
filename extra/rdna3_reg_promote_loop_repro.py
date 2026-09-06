#!/usr/bin/env python3
"""Minimal REDUCE-carried REG promote reproducer (AMDRenderer).

One GLOBAL thread owns a `wide`-lane REG accumulator updated across T REDUCE
iterations. Compares SKIP_SLOTS='2' (scratch) vs '' (promote) to a CPU ref.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp


def run(*, skip: str, t: int, wide: int) -> tuple[float, np.ndarray, np.ndarray]:
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  getenv.cache_clear()
  to_program_cache.clear()

  x_np = (((np.arange(t * wide) % 17) - 8) / 8.0).astype(np.float32).reshape(t, wide)
  ref = x_np.sum(axis=0)

  def kernel(out: UOp, x: UOp) -> UOp:
    g = UOp.range(1, 0, AxisType.GLOBAL)
    acc = UOp.placeholder((wide,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
    acc = acc.after(acc.store(acc.const_like(0.0)))
    it = UOp.range(t, 100, AxisType.REDUCE)
    stores = []
    for i in range(wide):
      val = x[it, i].load().cast(dtypes.float)
      stores.append(acc[i].store(acc.after(it)[i] + val))
    acc = acc.after(UOp.group(*stores).end(it))
    return out[0:wide].store(acc).end(g).sink(arg=KernelInfo(opts_to_apply=()))

  x = Tensor(x_np, device=Device.DEFAULT).realize()
  out = Tensor.empty(wide, dtype=dtypes.float32, device=Device.DEFAULT)
  y = Tensor.custom_kernel(out, x, fxn=kernel)[0].realize()
  got = y.numpy().astype(np.float32)
  if not np.isfinite(got).all():
    return float("nan"), got, ref
  return float(np.nanmax(np.abs(got - ref))), got, ref


def main() -> None:
  print("device", Device.DEFAULT, Device[Device.DEFAULT].renderer.__class__.__name__)
  cases = [
    ("w1 skip", "2", 4, 1),
    ("w1 promote", "", 4, 1),
    ("w8 skip", "2", 4, 8),
    ("w8 promote", "", 4, 8),
    ("w32 skip", "2", 4, 32),
    ("w32 promote", "", 4, 32),
    ("w32x8t skip", "2", 8, 32),
    ("w32x8t promote", "", 8, 32),
  ]
  failed = 0
  for name, skip, t, wide in cases:
    try:
      err, got, ref = run(skip=skip, t=t, wide=wide)
      ok = (err == err) and err < 1e-4
      failed += not ok
      print(f"{'OK' if ok else 'FAIL'} {name}: err={err:.3e}")
      if not ok:
        print("  got", got)
        print("  ref", ref)
    except Exception as e:
      failed += 1
      print(f"FAIL {name}: EXC {type(e).__name__}: {e}")
  raise SystemExit(failed)


if __name__ == "__main__":
  main()
