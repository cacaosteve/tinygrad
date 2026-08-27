#!/usr/bin/env python3
"""Deterministic Q6_K GEMV used to compare RDNA3 compiler backends.

The shape (8192, 2048) and Q6_K block layout match TestGGUFGEMV.  Inputs are
generated from integer formulas so the equivalent Mojo program uses the same
bytes without exchanging a model file.

AMD:AMD selects the complex-matvec wave-per-row schedule automatically.  Use
MV_FORCE_GROUP=32 MV_FORCE_UNROLL_INNER=-1 for the same launch geometry on HIP
or LLVM when making backend-to-backend comparisons.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from typing import Any

import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import Ops


ROWS, COLS = 8192, 2048
QK, BLOCK_BYTES = 256, 210
BLOCKS_PER_ROW = COLS // QK
N_BLOCKS = ROWS * BLOCKS_PER_ROW
WARMUP, WARMUP_SECONDS, BATCH, ROUNDS = 5, 1.0, 100, 7


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
  idx = np.arange(N_BLOCKS * BLOCK_BYTES, dtype=np.uint32)
  qdata = ((idx * 37 + 13 + (idx >> 8) * 17) & 0xFF).astype(np.uint8)
  blocks = qdata.reshape(N_BLOCKS, BLOCK_BYTES)
  # Exact fp16 0.03125, little-endian. This keeps every synthetic block finite.
  blocks[:, 208] = 0
  blocks[:, 209] = 40
  x = (((np.arange(COLS, dtype=np.int32) % 31) - 15) / 16.0).astype(np.float32)
  return qdata, x


def reference_rows(qdata: np.ndarray, x: np.ndarray, count: int = 16) -> np.ndarray:
  blocks = qdata.reshape(N_BLOCKS, BLOCK_BYTES)
  j = np.arange(QK, dtype=np.int32)
  half, within = j // 128, j % 128
  lo_idx, lo_shift = half * 64 + within % 64, (within // 64) * 4
  hi_idx, hi_shift = 128 + half * 32 + within % 32, (within // 32) * 2
  out = np.empty(count, dtype=np.float32)
  for row in range(count):
    rb = blocks[row * BLOCKS_PER_ROW:(row + 1) * BLOCKS_PER_ROW]
    lo = (rb[:, lo_idx] >> lo_shift) & 0xF
    hi = ((rb[:, hi_idx] >> hi_shift) & 0x3) << 4
    quant = (lo | hi).astype(np.int16) - 32
    scales = rb[:, 192 + j // 16].view(np.int8).astype(np.float32)
    d = rb[:, 208:210].copy().view(np.float16).astype(np.float32)
    weights = (d * quant * scales).reshape(COLS).astype(np.float32)
    out[row] = weights @ x
  return out


def program_metrics(runner: Any, dump_machine: bool = False) -> list[dict[str, object]]:
  assert runner.captured is not None
  ret, seen = [], set()
  for call in runner.captured.linear.src:
    for program in call.src[0].toposort():
      if program.op is not Ops.PROGRAM or program in seen: continue
      seen.add(program)
      source = next((s.arg for s in program.src if s.op is Ops.SOURCE), "")
      binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
      metrics: dict[str, object] = {
        "name": program.arg.function_name,
        "source_bytes": len(source.encode()) if isinstance(source, str) else 0,
        "binary_bytes": len(binary) if isinstance(binary, bytes) else 0,
        "global_size": program.arg.global_size,
        "local_size": program.arg.local_size,
      }
      if isinstance(source, str) and source.lstrip().startswith("AMDOps."):
        asm_ops = Counter(line.strip().removeprefix("AMDOps.") for line in source.splitlines() if line.strip())
        metrics.update({
          "asm_instruction_count": sum(asm_ops.values()),
          "asm_fills": asm_ops["FILL"],
          "asm_spills": asm_ops["SPILL"],
          "asm_waits": asm_ops["WAIT"],
          "asm_top_ops": dict(asm_ops.most_common(16)),
        })
      if Device.DEFAULT.split(":")[0] == "AMD" and isinstance(binary, bytes) and binary.startswith(b"\x7fELF"):
        from tinygrad.renderer.amd import decode_inst
        from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
        from tinygrad.runtime.support.elf import elf_loader

        text = next(section.content for section in elf_loader(binary)[1] if section.name == ".text")
        insts, offset = [], 0
        while offset < len(text):
          inst = decode_inst(text[offset:], "rdna3")
          insts.append(inst)
          offset += inst.size()
          if getattr(inst, "op_name", "").lower() == "s_endpgm": break
        machine_ops = Counter(getattr(inst, "op_name", type(inst).__name__).lower() for inst in insts)
        max_vgpr, max_sgpr, _, private_segment_size = scan_elf_regs(insts, scratch_inst_size)
        metrics.update({
          "machine_instruction_count": len(insts),
          "machine_waits": sum(count for op, count in machine_ops.items() if "waitcnt" in op),
          "machine_vmcnt_values": dict(Counter(str(getattr(inst, "simm16", "")) for inst in insts
                                                if getattr(inst, "op_name", "").lower() == "s_waitcnt_vmcnt")),
          "machine_global_loads": sum(count for op, count in machine_ops.items() if op.startswith("global_load")),
          "machine_top_ops": dict(machine_ops.most_common(24)),
          "max_vgpr": max_vgpr,
          "max_sgpr": max_sgpr,
          "private_segment_size": private_segment_size,
        })
        if dump_machine: metrics["machine_instructions"] = [repr(inst) for inst in insts]
      ret.append(metrics)
  return ret


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dump-machine", action="store_true")
  args = parser.parse_args()
  qdata, x_np = make_inputs()
  ref = reference_rows(qdata, x_np)
  qdata_dev = Tensor(qdata, device=Device.DEFAULT).realize()
  x = Tensor(x_np, device=Device.DEFAULT).realize()
  weights = ggml_data_to_tensor(qdata_dev, ROWS * COLS, 14).reshape(ROWS, COLS)

  def gemv(inp: Tensor) -> Tensor:
    return (weights @ inp).realize()

  runner = TinyJit(gemv)
  result = None
  for _ in range(WARMUP):
    result = runner(x)
  Device[Device.DEFAULT].synchronize()

  # A handful of launches primes the JIT but does not bring a desktop RDNA3 card out of
  # its low clock state. Keep the kernel busy for a fixed interval before taking samples.
  warmup_end = time.perf_counter() + WARMUP_SECONDS
  while time.perf_counter() < warmup_end:
    for _ in range(BATCH): result = runner(x)
    Device[Device.DEFAULT].synchronize()

  samples_us = []
  for _ in range(ROUNDS):
    start = time.perf_counter_ns()
    for _ in range(BATCH):
      result = runner(x)
    Device[Device.DEFAULT].synchronize()
    samples_us.append((time.perf_counter_ns() - start) / BATCH / 1e3)

  assert result is not None
  out = result.numpy().astype(np.float32)
  metrics = {
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "shape": [ROWS, COLS],
    "q6k_bytes": int(qdata.nbytes),
    "warmup_seconds": WARMUP_SECONDS,
    "median_us": statistics.median(samples_us),
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "max_abs_error_first16": float(np.max(np.abs(out[:16] - ref))),
    "checksum": float(out.astype(np.float64).sum()),
    "first8": out[:8].tolist(),
    "programs": program_metrics(runner, dump_machine=args.dump_machine),
  }
  print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()
