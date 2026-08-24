#!/usr/bin/env python3
"""Cold first-call and steady-state probes for representative non-quantized kernels."""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import numpy as np

from bench_q6k_gemv import program_metrics
from tinygrad import Device, Tensor, TinyJit
from tinygrad.dtype import dtypes


def patterned(shape: tuple[int, ...], scale: float = 64.0, dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / scale).astype(dtype).reshape(shape)


def build_case(name: str) -> tuple[Callable, tuple[Tensor, ...], dict[str, object]]:
  if name == "elementwise":
    shape = (1 << 24,)
    x = Tensor(patterned(shape), device=Device.DEFAULT).realize()
    y = Tensor(patterned(shape, 96.0), device=Device.DEFAULT).realize()
    return lambda a, b: (a * 1.25 + b).relu().realize(), (x, y), {"shape": shape, "bytes": x.nbytes() * 3}
  if name == "reduce":
    shape = (1 << 24,)
    x = Tensor(patterned(shape), device=Device.DEFAULT).realize()
    return lambda a: a.sum().realize(), (x,), {"shape": shape, "bytes": x.nbytes()}
  if name == "gemv":
    rows, cols = 8192, 2048
    weights = Tensor(patterned((rows, cols)), device=Device.DEFAULT).realize()
    x = Tensor(patterned((cols,), 32.0), device=Device.DEFAULT).realize()
    return lambda inp: (weights @ inp).realize(), (x,), {"shape": (rows, cols), "bytes": weights.nbytes() + x.nbytes()}
  if name == "gemm":
    size = 1024
    a = Tensor(patterned((size, size), dtype=np.float16), device=Device.DEFAULT).realize()
    b = Tensor(patterned((size, size), 96.0, np.float16), device=Device.DEFAULT).realize()
    return lambda lhs, rhs: (lhs @ rhs).realize(), (a, b), {
      "shape": (size, size, size), "bytes": a.nbytes() + b.nbytes(), "dtype": str(dtypes.float16)}
  raise ValueError(name)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=("elementwise", "reduce", "gemv", "gemm"), required=True)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--batch", type=int, default=20)
  parser.add_argument("--rounds", type=int, default=5)
  args = parser.parse_args()

  fn, inputs, details = build_case(args.case)
  runner = TinyJit(fn)
  call_ms, result = [], None
  for _ in range(max(args.warmup, 2)):
    start = time.perf_counter_ns()
    result = runner(*inputs)
    Device[Device.DEFAULT].synchronize()
    call_ms.append((time.perf_counter_ns() - start) / 1e6)

  samples_us = []
  for _ in range(args.rounds):
    start = time.perf_counter_ns()
    for _ in range(args.batch): result = runner(*inputs)
    Device[Device.DEFAULT].synchronize()
    samples_us.append((time.perf_counter_ns() - start) / args.batch / 1e3)

  assert result is not None
  out = result.numpy().astype(np.float32)
  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "case": args.case,
    **details,
    "first_call_ms": call_ms[0],
    "capture_call_ms": call_ms[1],
    "warmup_call_ms": call_ms,
    "median_us": statistics.median(samples_us),
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "checksum": float(out.astype(np.float64).sum()),
    "first8": out.reshape(-1)[:8].tolist(),
    "programs": program_metrics(runner),
  }, indent=2))


if __name__ == "__main__": main()
