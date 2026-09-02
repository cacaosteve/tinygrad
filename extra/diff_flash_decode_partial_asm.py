#!/usr/bin/env python3
"""Compare flash_decode_partial machine code: DEV=AMD:AMD vs DEV=AMD:HIP.

Usage:
  DEV=AMD:AMD  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py
  DEV=AMD:HIP  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py
  PYTHONPATH=.:extra python extra/diff_flash_decode_partial_asm.py --compare direct.json hip.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from typing import Any

import numpy as np

from bench_q6k_gemv import program_metrics
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels.amd import flash_attention
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops


def patterned(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def partial_binary(runner: TinyJit) -> bytes:
  assert runner.captured is not None
  for call in runner.captured.linear.src:
    for program in call.src[0].toposort():
      if program.op is not Ops.PROGRAM or program.arg.function_name != "flash_decode_partial": continue
      binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
      if not isinstance(binary, bytes) or not binary.startswith(b"\x7fELF"):
        raise SystemExit("flash_decode_partial ELF not found")
      return binary
  raise SystemExit("flash_decode_partial program not found")


def decode_elf(binary: bytes) -> list:
  text = next(section.content for section in elf_loader(binary)[1] if section.name == ".text")
  insts, offset = [], 0
  while offset < len(text):
    inst = decode_inst(text[offset:], "rdna3")
    insts.append(inst)
    offset += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  return insts


def opname(inst) -> str:
  return getattr(inst, "op_name", "").lower()


def is_lgkm_wait(name: str) -> bool:
  return name in ("s_waitcnt_lgkmcnt", "s_waitcnt")


def is_add(name: str) -> bool:
  return name.startswith("v_add_f32")


def analyze_swizzle_bubbles(insts: list) -> dict[str, Any]:
  patterns: list[dict[str, Any]] = []
  i = 0
  while i < len(insts):
    if opname(insts[i]) != "ds_swizzle_b32":
      i += 1
      continue
    ops_before_wait: list[str] = []
    delay_before = 0
    j = i + 1
    wait_i: int|None = None
    while j < len(insts) and j < i + 24:
      n = opname(insts[j])
      if is_lgkm_wait(n):
        wait_i = j
        break
      if n == "s_delay_alu": delay_before += 1
      ops_before_wait.append(n)
      j += 1
    add_i: int|None = None
    ops_after_wait: list[str] = []
    delay_after = 0
    if wait_i is not None:
      k = wait_i + 1
      while k < len(insts) and k < wait_i + 12:
        n = opname(insts[k])
        if is_add(n):
          add_i = k
          break
        if n == "s_delay_alu": delay_after += 1
        ops_after_wait.append(n)
        k += 1
    patterns.append({
      "ops_before_wait": len(ops_before_wait),
      "delay_alu_before_wait": delay_before,
      "has_global_load_before_wait": any(x.startswith("global_load") for x in ops_before_wait),
      "has_valu_before_wait": any(x.startswith("v_") for x in ops_before_wait),
      "ops_after_wait_before_add": len(ops_after_wait),
      "delay_alu_after_wait": delay_after,
      "swizzle_to_wait": (wait_i - i) if wait_i is not None else None,
      "wait_to_add": (add_i - wait_i) if wait_i is not None and add_i is not None else None,
    })
    i += 1

  def med(key: str) -> float|None:
    vals = [p[key] for p in patterns if p.get(key) is not None]
    return statistics.median(vals) if vals else None

  before_hist = Counter(p["ops_before_wait"] for p in patterns)
  return {
    "swizzle_count": len(patterns),
    "median_ops_before_lgkm_wait": med("ops_before_wait"),
    "median_delay_alu_before_wait": med("delay_alu_before_wait"),
    "median_ops_after_wait_before_add": med("ops_after_wait_before_add"),
    "median_delay_alu_after_wait": med("delay_alu_after_wait"),
    "median_swizzle_to_wait": med("swizzle_to_wait"),
    "median_wait_to_add": med("wait_to_add"),
    "pct_global_load_before_wait": round(100 * sum(p["has_global_load_before_wait"] for p in patterns) / max(1, len(patterns)), 1),
    "pct_valu_before_wait": round(100 * sum(p["has_valu_before_wait"] for p in patterns) / max(1, len(patterns)), 1),
    "ops_before_wait_hist": dict(sorted(before_hist.items())),
    "machine_op_counts": dict(Counter(opname(i) for i in insts).most_common(20)),
    "machine_waits": sum(1 for i in insts if "waitcnt" in opname(i)),
    "machine_delay_alu": sum(1 for i in insts if opname(i) == "s_delay_alu"),
    "machine_swizzle": sum(1 for i in insts if opname(i) == "ds_swizzle_b32"),
    "machine_global_loads": sum(1 for i in insts if opname(i).startswith("global_load")),
  }


def capture_report() -> dict[str, Any]:
  batch, heads, kv_heads, dim, physical_n, tokens = 1, 32, 8, 128, 2048, 1
  q = Tensor(patterned((batch, heads, tokens, dim)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, batch, kv_heads, physical_n, dim), np.float16), device=Device.DEFAULT).realize()

  def attention(query: Tensor) -> Tensor:
    return flash_attention(query, cache, physical_n).realize()

  runner = TinyJit(attention)
  for _ in range(2):
    runner(q)
    Device[Device.DEFAULT].synchronize()

  metrics = next(m for m in program_metrics(runner) if m["name"] == "flash_decode_partial")
  bubble = analyze_swizzle_bubbles(decode_elf(partial_binary(runner)))
  return {
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "program_metrics": {k: v for k, v in metrics.items() if k != "machine_instructions"},
    "swizzle_bubbles": bubble,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--compare", nargs=2, metavar=("DIRECT_JSON", "HIP_JSON"), help="print side-by-side diff")
  parser.add_argument("-o", "--output", help="write JSON report")
  args = parser.parse_args()

  if args.compare:
    direct, hip = (json.load(open(p)) for p in args.compare)
    keys = ["machine_swizzle", "machine_waits", "machine_delay_alu", "machine_global_loads",
            "median_ops_before_lgkm_wait", "median_delay_alu_before_wait",
            "median_ops_after_wait_before_add", "pct_global_load_before_wait"]
    print(f"{'metric':40} {'direct':>12} {'hip':>12} {'delta':>10}")
    for k in keys:
      d = direct["swizzle_bubbles"].get(k, direct["program_metrics"].get(k))
      h = hip["swizzle_bubbles"].get(k, hip["program_metrics"].get(k))
      delta = "" if not isinstance(d, (int, float)) or not isinstance(h, (int, float)) else f"{h - d:+}"
      print(f"{k:40} {str(d):>12} {str(h):>12} {delta:>10}")
    return

  report = capture_report()
  if args.output: json.dump(report, open(args.output, "w"), indent=2)
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
