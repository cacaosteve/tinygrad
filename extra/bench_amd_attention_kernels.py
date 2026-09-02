#!/usr/bin/env python3
"""Per-kernel timing for AMD flash-attention decode (direct ISA vs HIP)."""

import argparse
import json
import statistics

import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.engine.jit import _prepare_jit_inputs
from tinygrad.engine.realize import time_call
from tinygrad.llm.kernels.amd import flash_attention
from tinygrad.uop.ops import Ops


def patterned(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=20)
  args = parser.parse_args()

  batch, heads, dim, physical_n, tokens = 1, 32, 128, 2048, 1
  q = Tensor(patterned((batch, heads, tokens, dim)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, batch, 8, physical_n, dim), np.float16), device=Device.DEFAULT).realize()

  def attention(query: Tensor) -> Tensor:
    return flash_attention(query, cache, physical_n).realize()

  runner = TinyJit(attention)
  for _ in range(3):
    runner(q)
    Device[Device.DEFAULT].synchronize()

  _, var_vals, _, _ = _prepare_jit_inputs((q,), {})
  linear = runner.captured.linear
  kernels: list[dict[str, object]] = []
  seen: set[str] = set()
  for call in linear.src:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.PROGRAM: continue
    name = call.src[0].arg.function_name
    if name in seen: continue
    seen.add(name)
    tms: list[float] = []
    timer = time_call(call, var_vals)
    for _ in range(args.samples):
      try:
        tms.append(next(timer) * 1e3)
      except StopIteration:
        break
    kernels.append({
      "name": name,
      "median_us": statistics.median(tms) if tms else None,
      "best_us": min(tms) if tms else None,
      "samples_us": tms,
      "global_size": call.src[0].arg.global_size,
      "local_size": call.src[0].arg.local_size,
    })

  total = sum(k["median_us"] or 0 for k in kernels)
  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "kernels": kernels,
    "sum_median_us": total,
  }, indent=2))


if __name__ == "__main__":
  main()
