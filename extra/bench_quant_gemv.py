#!/usr/bin/env python3
"""Deterministic quantized GEMV correctness, compile-latency, and throughput probe.

Examples:
  DEV=AMD:AMD CCACHE=0 python extra/bench_quant_gemv.py --qtype Q6_K --rows 8192 --cols 2048
  DEV=AMD MV_FORCE_GROUP=32 MV_FORCE_UNROLL_INNER=-1 CCACHE=0 python extra/bench_quant_gemv.py --qtype Q6_K
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass

import numpy as np

from bench_q6k_gemv import program_metrics
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.gguf import ggml_data_to_tensor


@dataclass(frozen=True)
class Quant:
  ggml_type: int
  block_elements: int
  block_bytes: int
  scale_offsets: tuple[int, ...]
  scale_values: tuple[float, ...]


QUANTS = {
  "Q4_K": Quant(12, 256, 144, (0, 2), (0.03125, 0.015625)),
  "Q5_K": Quant(13, 256, 176, (0, 2), (0.03125, 0.015625)),
  "Q6_K": Quant(14, 256, 210, (208,), (0.03125,)),
  "Q8_0": Quant(8, 32, 34, (0,), (0.03125,)),
  "IQ4_XS": Quant(23, 256, 136, (0,), (0.03125,)),
}


def make_inputs(rows: int, cols: int, quant: Quant) -> tuple[np.ndarray, np.ndarray]:
  assert (rows * cols) % quant.block_elements == 0
  nblocks = rows * cols // quant.block_elements
  # uint8 arithmetic intentionally wraps, avoiding a large uint32 temporary for model-sized probes.
  qdata = (np.arange(nblocks * quant.block_bytes, dtype=np.uint8) * np.uint8(37) + np.uint8(13))
  blocks = qdata.reshape(nblocks, quant.block_bytes)
  for offset, value in zip(quant.scale_offsets, quant.scale_values):
    blocks[:, offset:offset+2] = np.array([value], dtype=np.float16).view(np.uint8)
  x = (((np.arange(cols, dtype=np.int32) % 31) - 15) / 16.0).astype(np.float32)
  return qdata, x


def reference_rows(qdata: np.ndarray, x: np.ndarray, rows: int, cols: int, quant: Quant, count: int = 16) -> np.ndarray:
  count = min(rows, count)
  row_bytes = cols // quant.block_elements * quant.block_bytes
  raw = Tensor(qdata[:count * row_bytes].copy(), device="CPU").realize()
  weights = ggml_data_to_tensor(raw, count * cols, quant.ggml_type).reshape(count, cols)
  return (weights @ Tensor(x.copy(), device="CPU")).numpy().astype(np.float32)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--qtype", choices=QUANTS, default="Q6_K")
  parser.add_argument("--rows", type=int, default=8192)
  parser.add_argument("--cols", type=int, default=2048)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--warmup-seconds", type=float, default=0.0)
  parser.add_argument("--batch", type=int, default=50)
  parser.add_argument("--rounds", type=int, default=5)
  args = parser.parse_args()
  quant = QUANTS[args.qtype]
  qdata, x_np = make_inputs(args.rows, args.cols, quant)
  ref = reference_rows(qdata, x_np, args.rows, args.cols, quant)

  qdata_dev = Tensor(qdata, device=Device.DEFAULT).realize()
  x = Tensor(x_np, device=Device.DEFAULT).realize()
  weights = ggml_data_to_tensor(qdata_dev, args.rows * args.cols, quant.ggml_type).reshape(args.rows, args.cols)

  def gemv(inp: Tensor) -> Tensor: return (weights @ inp).realize()

  runner = TinyJit(gemv)
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
  median_us = statistics.median(samples_us)
  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "qtype": args.qtype,
    "shape": [args.rows, args.cols],
    "qbytes": int(qdata.nbytes),
    "first_call_ms": call_ms[0],
    "capture_call_ms": call_ms[1],
    "warmup_call_ms": call_ms,
    "warmup_seconds": args.warmup_seconds,
    "median_us": median_us,
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "effective_qbytes_per_s": qdata.nbytes / (median_us * 1e-6),
    "max_abs_error_first16": float(np.max(np.abs(out[:len(ref)] - ref))),
    "checksum": float(out.astype(np.float64).sum()),
    "first8": out[:8].tolist(),
    "programs": program_metrics(runner),
  }, indent=2))


if __name__ == "__main__": main()
