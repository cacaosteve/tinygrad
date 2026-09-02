#!/usr/bin/env python3
"""Per-kernel timing for AMD flash-attention decode (direct ISA vs HIP)."""

import argparse
import json
import statistics

import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.engine.jit import _copy_input, _prepare_jit_inputs
from tinygrad.engine.realize import get_graph_runtime, get_runtime
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

  input_buf_uops, var_vals, _, _ = _prepare_jit_inputs((q,), {})
  concrete = tuple(_copy_input(u) if u in runner.captured._written_uops else u for u in input_buf_uops)
  graph_call = runner.captured.linear.src[0]
  if graph_call.src[0].op is not Ops.CUSTOM_FUNCTION or graph_call.src[0].arg != "graph":
    raise SystemExit("expected graphed flash decode jit")
  gr = get_graph_runtime(graph_call.src[0], concrete)

  kernels: list[dict[str, object]] = []
  for j, (_dev_idx, prg, bufs, device_vars) in enumerate(gr.calls):
    if prg.op is not Ops.PROGRAM: continue
    vv = {**var_vals, **device_vars}
    rt = get_runtime(bufs[0].device, prg)
    gs, ls = prg.arg.launch_dims(vv)
    vals = prg.arg.vals(vv)
    tms: list[float] = []
    for _ in range(args.samples):
      et = rt(*[b.get_buf(b.device) for b in bufs], global_size=gs, local_size=ls, vals=vals, wait=True)
      tms.append((et or 0) * 1e6)  # HCQ returns seconds → µs
    kernels.append({
      "name": prg.arg.function_name,
      "median_us": statistics.median(tms),
      "best_us": min(tms),
      "samples_us": tms,
      "global_size": gs,
      "local_size": ls,
    })

  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "kernels": kernels,
    "sum_median_us": sum(k["median_us"] for k in kernels),
  }, indent=2))


if __name__ == "__main__":
  main()
