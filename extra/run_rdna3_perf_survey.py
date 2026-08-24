#!/usr/bin/env python3
"""Run isolated RDNA3 direct/HIP/LLVM performance probes and emit JSONL summaries."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKENDS = {
  "direct": {"DEV": "AMD:AMD"},
  "hip": {"DEV": "AMD"},
  "llvm": {"DEV": "AMD:LLVM"},
}
QUANT_OVERRIDES = {"MV_FORCE_GROUP": "32", "MV_FORCE_UNROLL_INNER": "-1"}
QUANT_TYPES = [(qtype, 8192, 2048) for qtype in ("Q4_K", "Q5_K", "Q6_K", "Q8_0")]
Q6_SHAPES = [("Q6_K", rows, cols) for rows, cols in (
  (4096, 4096), (11008, 4096), (4096, 11008), (32000, 4096))]
ORDINARY_CASES = ("elementwise", "reduce", "gemv", "gemm")


def run_probe(backend: str, suite: str, label: str, cmd: list[str]) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": str(ROOT), "CCACHE": "0", **BACKENDS[backend], **(QUANT_OVERRIDES if suite != "ordinary" else {})})
  for key in ("MV_FORCE_GROUP", "MV_FORCE_UNROLL_INNER"):
    if suite == "ordinary": env.pop(key, None)
  try:
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=180)
    data = json.loads(proc.stdout)
    programs: list[dict[str, Any]] = data.get("programs", [])
    program: dict[str, Any] = max(programs, key=lambda p: int(p.get("binary_bytes", 0)), default={})
    summary = {
      "backend": backend, "suite": suite, "label": label, "renderer": data["renderer"],
      "shape": data["shape"], "first_call_ms": data["first_call_ms"], "capture_call_ms": data["capture_call_ms"],
      "median_us": data["median_us"], "best_us": data["best_us"], "checksum": data["checksum"], "first8": data["first8"],
      "max_abs_error_first16": data.get("max_abs_error_first16"), "global_size": program.get("global_size"),
      "local_size": program.get("local_size"), "binary_bytes": program.get("binary_bytes"), "max_vgpr": program.get("max_vgpr"),
      "max_sgpr": program.get("max_sgpr"), "private_segment_size": program.get("private_segment_size"),
    }
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
    stderr = getattr(exc, "stderr", "") or ""
    stdout = getattr(exc, "stdout", "") or ""
    summary = {"backend": backend, "suite": suite, "label": label, "error": str(exc),
               "output_tail": (stderr + stdout)[-2000:]}
  print(json.dumps(summary), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--suite", choices=("quant-types", "q6-shapes", "ordinary", "all"), default="all")
  parser.add_argument("--backend", action="append", choices=tuple(BACKENDS), dest="backends")
  args = parser.parse_args()
  backends = args.backends or list(BACKENDS)

  if args.suite in ("quant-types", "all"):
    for qtype, rows, cols in QUANT_TYPES:
      for backend in backends:
        run_probe(backend, "quant-types", qtype, ["extra/bench_quant_gemv.py", "--qtype", qtype, "--rows", str(rows),
          "--cols", str(cols), "--warmup", "50", "--batch", "50", "--rounds", "5"])
  if args.suite in ("q6-shapes", "all"):
    for qtype, rows, cols in Q6_SHAPES:
      for backend in backends:
        run_probe(backend, "q6-shapes", f"{qtype}:{rows}x{cols}", ["extra/bench_quant_gemv.py", "--qtype", qtype,
          "--rows", str(rows), "--cols", str(cols), "--warmup", "50", "--batch", "50", "--rounds", "5"])
  if args.suite in ("ordinary", "all"):
    for case in ORDINARY_CASES:
      for backend in backends:
        run_probe(backend, "ordinary", case, ["extra/bench_backend_kernels.py", "--case", case,
          "--warmup", "50", "--batch", "20", "--rounds", "5"])


if __name__ == "__main__": main()
