#!/usr/bin/env python3
"""Benchmark the custom RDNA3 quantized-linear paths used by tinygrad LLMs.

Examples:
  DEV=AMD:AMD python extra/bench_amd_qlinear.py --format q4_k --rows 8192 --cols 2048 --tokens 1
  DEV=AMD:AMD python extra/bench_amd_qlinear.py --format iq4_xs --rows 8192 --cols 2048 --tokens 16
"""

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from bench_q6k_gemv import program_metrics
from tinygrad import Device, Tensor, TinyJit, nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
from tinygrad.uop.ops import Ops


QK = 256
FORMATS = {"q4_k": (12, 144), "q5_k": (13, 176), "q6_k": (14, 210), "iq4_xs": (23, 136)}


def make_inputs(rows: int, cols: int, tokens: int, ggml_format: str) -> tuple[np.ndarray, np.ndarray]:
  _, block_bytes = FORMATS[ggml_format]
  blocks = rows * cols // QK
  idx = np.arange(blocks * block_bytes, dtype=np.uint32)
  qdata = ((idx * 37 + 13 + (idx >> 8) * 17) & 0xff).astype(np.uint8)
  qblocks = qdata.reshape(blocks, block_bytes)
  if ggml_format in ("q4_k", "q5_k"):
    qblocks[:, :4] = np.array([0.03125, 0.015625], dtype=np.float16).view(np.uint8)
  elif ggml_format == "q6_k": qblocks[:, 208:210] = np.array([0.03125], dtype=np.float16).view(np.uint8)
  else: qblocks[:, :2] = np.array([0.03125], dtype=np.float16).view(np.uint8)
  x = (((np.arange(tokens * cols, dtype=np.int32) % 31) - 15) / 16.0).astype(np.float32).reshape(tokens, cols)
  return qdata, x


def disassemble(runner:Any) -> dict[str, list[str]]:
  assert runner.captured is not None
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  ret:dict[str, list[str]] = {}
  for call in runner.captured.linear.src:
    for program in call.src[0].toposort():
      if program.op is not Ops.PROGRAM: continue
      binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
      if not isinstance(binary, bytes) or not binary.startswith(b"\x7fELF"): continue
      text = next(section.content for section in elf_loader(binary)[1] if section.name == ".text")
      insts, offset = [], 0
      while offset < len(text):
        inst = decode_inst(text[offset:], "rdna3")
        insts.append(str(inst))
        offset += inst.size()
        if getattr(inst, "op_name", "").lower() == "s_endpgm": break
      ret[program.arg.function_name] = insts
  return ret


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rows", type=int, default=8192)
  parser.add_argument("--cols", type=int, default=2048)
  parser.add_argument("--tokens", type=int, default=1)
  parser.add_argument("--format", choices=tuple(FORMATS), default="q6_k")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--warmup-seconds", type=float, default=1.0)
  parser.add_argument("--batch", type=int, default=100)
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--disassemble", action="store_true")
  args = parser.parse_args()
  assert args.rows > 0 and args.cols > 0 and args.tokens > 0 and (args.rows * args.cols) % QK == 0

  ggml_type, _ = FORMATS[args.format]
  qdata, x_np = make_inputs(args.rows, args.cols, args.tokens, args.format)
  # Keep the same non-zero-offset packed view used by GGUF loading so Linear can
  # recover the original bytes instead of falling back to generic dequantization.
  raw = Tensor(np.pad(qdata, (4, 0)), device=Device.DEFAULT).contiguous().realize()[4:]
  ref_rows = min(args.rows, 16)
  ref_weights = ggml_data_to_tensor(raw, args.rows * args.cols, ggml_type).reshape(args.rows, args.cols)[:ref_rows].numpy()
  custom = amd_custom_kernels_supported(Device.DEFAULT)
  if custom and args.tokens % 16 == 0:
    x_ref, ref_weights = x_np.astype(np.float16).astype(np.float32), ref_weights.astype(np.float16).astype(np.float32)
  elif custom:
    scale = np.maximum(np.abs(x_np).reshape(args.tokens, args.cols//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    x_ref = (np.clip(np.rint(x_np.reshape(args.tokens, args.cols//32, 32) / scale), -127, 127) * scale).reshape(x_np.shape)
  else: x_ref = x_np
  reference = x_ref @ ref_weights.T
  decoded = ggml_data_to_tensor(raw, args.rows * args.cols, ggml_type).reshape(args.rows, args.cols)
  linear = Linear(args.cols, args.rows, bias=False)
  nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
  x = Tensor(x_np, device=Device.DEFAULT).realize()

  def run(inp: Tensor) -> Tensor: return linear(inp).realize()

  runner, result, call_ms = TinyJit(run), None, []
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
  summary = {
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "format": args.format,
    "ggml_type": ggml_type,
    "shape": [args.tokens, args.rows, args.cols],
    "qbytes": int(qdata.nbytes),
    "first_call_ms": call_ms[0],
    "capture_call_ms": call_ms[1],
    "warmup_call_ms": call_ms,
    "warmup_seconds": args.warmup_seconds,
    "median_us": statistics.median(samples_us),
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "effective_qbytes_per_s": qdata.nbytes / (statistics.median(samples_us) * 1e-6),
    "finite": bool(np.isfinite(out).all()),
    "max_abs_error_first16": float(np.max(np.abs(out[:, :ref_rows] - reference))),
    "checksum": float(out.astype(np.float64).sum()),
    "first8": out.reshape(-1)[:8].tolist(),
    "programs": program_metrics(runner),
  }
  if args.disassemble: summary["disassembly"] = disassemble(runner)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
