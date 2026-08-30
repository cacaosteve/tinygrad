#!/usr/bin/env python3
"""Deterministic short-token quantized WMMA benchmark for AMD backends.

Examples:
  DEV=AMD:AMD CCACHE=0 python extra/bench_quant_wmma.py --qtype Q6_K --tokens 32
  DEV=AMD:HIP CCACHE=0 python extra/bench_quant_wmma.py --qtype IQ4_XS --tokens 64
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench_q6k_gemv import program_metrics
from bench_quant_gemv import QUANTS, make_inputs
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear


def reference_rows(qdata:np.ndarray, x:np.ndarray, rows:int, cols:int, ggml_type:int, count:int=16) -> np.ndarray:
  count = min(rows, count)
  row_bytes = qdata.size // rows
  raw = Tensor(qdata[:count*row_bytes].copy(), device="CPU").realize()
  weights = ggml_data_to_tensor(raw, count*cols, ggml_type).reshape(count, cols)
  return (Tensor(x.copy(), device="CPU") @ weights.T).numpy().astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--qtype", choices=("Q4_K", "Q5_K", "Q6_K", "IQ4_XS"), default="Q6_K")
  parser.add_argument("--tokens", type=int, choices=(16, 32, 64), default=32)
  parser.add_argument("--rows", type=int, default=8192)
  parser.add_argument("--cols", type=int, default=2048)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--warmup-seconds", type=float, default=0.5)
  parser.add_argument("--batch", type=int, default=20)
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--dump-machine", action="store_true")
  args = parser.parse_args()
  quant = QUANTS[args.qtype]
  qdata, _ = make_inputs(args.rows, args.cols, quant)
  token = np.arange(args.tokens, dtype=np.int32)[:, None]
  feature = np.arange(args.cols, dtype=np.int32)[None, :]
  x_np = (((feature * 7 + token * 13) % 31 - 15) / 16.0).astype(np.float32)
  ref = reference_rows(qdata, x_np, args.rows, args.cols, quant.ggml_type)

  # Model GGUF tensors are views into a larger file buffer. Keep the same SHRINK view here so
  # Linear.set_quantized can identify and retain the packed backing storage used in production.
  raw = Tensor(np.pad(qdata, (0, 4)), device=Device.DEFAULT).realize()
  decoded = ggml_data_to_tensor(raw, args.rows*args.cols, quant.ggml_type).reshape(args.rows, args.cols)
  layer = Linear(args.cols, args.rows, bias=False)
  layer.weight = decoded
  x = Tensor(x_np, device=Device.DEFAULT).realize()

  def linear(inp:Tensor) -> Tensor: return layer(inp).realize()

  runner = TinyJit(linear)
  call_ms, result = [], None
  for _ in range(max(args.warmup, 2)):
    start = time.perf_counter_ns()
    result = runner(x)
    Device[Device.DEFAULT].synchronize()
    call_ms.append((time.perf_counter_ns() - start) / 1e6)

  warmup_end = time.perf_counter() + args.warmup_seconds
  while time.perf_counter() < warmup_end:
    for _ in range(args.batch): result = runner(x)
    Device[Device.DEFAULT].synchronize()

  samples_us = []
  for _ in range(args.rounds):
    start = time.perf_counter_ns()
    for _ in range(args.batch): result = runner(x)
    Device[Device.DEFAULT].synchronize()
    samples_us.append((time.perf_counter_ns() - start) / args.batch / 1e3)

  assert result is not None
  out = result.numpy().astype(np.float32)
  error = np.abs(out[:, :ref.shape[1]] - ref)
  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "qtype": args.qtype,
    "shape": [args.tokens, args.cols, args.rows],
    "first_call_ms": call_ms[0],
    "capture_call_ms": call_ms[1],
    "warmup_call_ms": call_ms,
    "median_us": statistics.median(samples_us),
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "max_abs_error_first16": float(error.max()),
    "checksum_first16": float(out[:, :16].astype(np.float64).sum()),
    "first8": out[0, :8].tolist(),
    "programs": program_metrics(runner, dump_machine=args.dump_machine),
  }, indent=2))


if __name__ == "__main__": main()
