#!/usr/bin/env python3
"""Cross-run bit-exact stability probe for flash promote vs SKIP=2 vs SDPA.

Distinguishes nondeterministic direct-ISA paths from SDPA-stable reference.
Example:

  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_stability.py
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_stability.py --S 256 --runs 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


CHILD = r"""
import os, sys, numpy as np
from tinygrad import Device, Tensor
from tinygrad.codegen import to_program_cache
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd import flash_attention
import tinygrad.llm.kernels.amd as amd

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

S = int(sys.argv[1]); T = min(int(sys.argv[2]), S)
skip, work, direct = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
outp = sys.argv[6]
os.environ["AMD_FLASH_DIRECT"] = str(direct)
os.environ["AMD_FLASH_ACC_SMALL"] = "1"
os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
os.environ["AMD_FLASH_ACC_WORK"] = str(work)
os.environ["AMD_SPILL_ON_EVICT"] = "0"
os.environ["AMD_PACK_SLOAD_B128"] = "0"
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
q = Tensor(patterned((1, 32, T, 128)), device=Device.DEFAULT).realize()
cache = Tensor(patterned((2, 1, 8, S, 128), np.float16), device=Device.DEFAULT).realize()
out = flash_attention(q, cache, S).realize().numpy().astype(np.float32)
np.save(outp, out)
print("finite", bool(np.isfinite(out).all()), "mean", float(out.mean()))
"""


def run_once(py: str, S: int, T: int, skip: str, work: int, direct: int, outp: Path) -> np.ndarray:
  env = {**os.environ, "PYTHONPATH": ".:extra", "CCACHE": "0", "CACHELEVEL": "0", "PARALLEL": "0"}
  subprocess.check_call(
    [py, "-c", CHILD, str(S), str(T), skip, str(work), str(direct), str(outp)], env=env)
  return np.load(outp)


def cross_run(label: str, arrs: list[np.ndarray]) -> None:
  base = arrs[0]
  diffs = [float(np.max(np.abs(a - base))) for a in arrs[1:]]
  exact = all(np.array_equal(base, a) for a in arrs[1:])
  print(f"{label}: exact={exact} max_cross={max(diffs) if diffs else 0.0:.6g} "
        f"diffs={','.join(f'{d:.4g}' for d in diffs)}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--runs", type=int, default=4)
  args = p.parse_args()
  py = sys.executable
  T = min(args.T, args.S)
  with tempfile.TemporaryDirectory(prefix="promote_stab_") as td:
    td_path = Path(td)
    configs = [
      ("skip2", "2", 1, 1),
      ("prom", "", 0, 1),
      ("sdpa", "2", 1, 0),
    ]
    for name, skip, work, direct in configs:
      arrs = []
      for i in range(args.runs):
        outp = td_path / f"{name}_{i}.npy"
        arrs.append(run_once(py, args.S, T, skip, work, direct, outp))
      cross_run(name, arrs)
    # Pairwise within last run folder already summarized; also print prom vs skip2 on run0/run1
    s0 = np.load(td_path / "skip2_0.npy")
    p0 = np.load(td_path / "prom_0.npy")
    d0 = np.load(td_path / "sdpa_0.npy")
    print(f"run0 skip2_vs_prom max={float(np.max(np.abs(s0-p0))):.6g} exact={np.array_equal(s0,p0)}")
    print(f"run0 skip2_vs_sdpa max={float(np.max(np.abs(s0-d0))):.6g}")
    print(f"run0 prom_vs_sdpa  max={float(np.max(np.abs(p0-d0))):.6g}")


if __name__ == "__main__":
  main()
