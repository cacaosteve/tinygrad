#!/usr/bin/env python3
"""Cross-run bit-exact stability probe for flash promote vs SKIP=2 vs SDPA.

Exits non-zero on instability or nonfinite outputs and keeps failing artifacts.
Records per-run ELF sha256 via to_program (not TinyJit — flash realizes internally).

Example:
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_stability.py
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_promote_stability.py --S 256 --runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


CHILD = r"""
import hashlib, json, os, sys
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.engine.realize import get_runtime, runtime_cache
from tinygrad.helpers import Context, getenv
from tinygrad.llm.kernels.amd import _amd_flash_attention
import tinygrad.llm.kernels.amd as amd
from tinygrad.uop.ops import Ops, UOp

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

S = int(sys.argv[1]); T = min(int(sys.argv[2]), S)
skip, work, direct = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
outp, metap, elfp = sys.argv[6], sys.argv[7], sys.argv[8]
batch = sys.argv[9] if len(sys.argv) > 9 else "1"
cluster = sys.argv[10] if len(sys.argv) > 10 else "1"

os.environ["AMD_FLASH_DIRECT"] = str(direct)
os.environ["AMD_FLASH_ACC_SMALL"] = "1"
os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
os.environ["AMD_FLASH_ACC_WORK"] = str(work)
os.environ["AMD_SPILL_ON_EVICT"] = "0"
os.environ["AMD_PACK_SLOAD_B128"] = "0"
os.environ["AMD_BATCH_SLOAD_USE"] = batch
os.environ["AMD_CLUSTER_SLOAD"] = cluster
getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear(); runtime_cache.clear()

use_acc_work = bool(work) and (2 in {int(s) for s in skip.split(",") if s.strip()})
elf_sha, fn = None, None
out = None

if int(direct) == 1:
  ph_out = UOp.placeholder((32, T, 128), dtypes.float32, 0)
  ph_q = UOp.placeholder((32, T, 128), dtypes.float16, 1)
  ph_kv = UOp.placeholder((2, 1, 8, S, 128), dtypes.float16, 2)
  sink = _amd_flash_attention(ph_out, ph_q, ph_kv, valid_kv_len=S, q_start=None,
                              acc_small=True, k_unroll=0, use_acc_work=use_acc_work)
  with Context(BEAM=0):
    os.environ["AMD_FLASH_ACC_SMALL"] = "1"; getenv.cache_clear()
    prg = to_program(sink, Device[Device.DEFAULT].renderer)
  binary = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  elf_sha = hashlib.sha256(binary).hexdigest()
  fn = prg.arg.function_name
  open(elfp, "wb").write(binary)
  q = Tensor(patterned((32, T, 128), np.float16), device=Device.DEFAULT, dtype=dtypes.float16).realize()
  kv = Tensor(patterned((2, 1, 8, S, 128), np.float16), device=Device.DEFAULT, dtype=dtypes.float16).realize()
  o = Tensor.empty(32, T, 128, dtype=dtypes.float32, device=Device.DEFAULT).realize()
  rt = get_runtime(Device.DEFAULT, prg, cache=True)
  gs, ls = prg.arg.launch_dims({}); vals = prg.arg.vals({})
  bufs = [o.uop.buffer.ensure_allocated().get_buf(Device.DEFAULT),
          q.uop.buffer.ensure_allocated().get_buf(Device.DEFAULT),
          kv.uop.buffer.ensure_allocated().get_buf(Device.DEFAULT)]
  rt(*[bufs[i] for i in prg.arg.globals], global_size=gs, local_size=ls, vals=vals, wait=True)
  Device[Device.DEFAULT].synchronize()
  out = o.numpy().astype(np.float32)
else:
  from tinygrad.llm.kernels.amd import flash_attention
  q = Tensor(patterned((1, 32, T, 128)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, 1, 8, S, 128), np.float16), device=Device.DEFAULT).realize()
  out = flash_attention(q, cache, S).realize().numpy().astype(np.float32)

np.save(outp, out)
meta = {"finite": bool(np.isfinite(out).all()), "mean": float(out.mean()),
        "elf_sha256": elf_sha, "function_name": fn}
json.dump(meta, open(metap, "w"))
print(json.dumps(meta))
"""


def run_once(py: str, S: int, T: int, skip: str, work: int, direct: int,
             outp: Path, metap: Path, elfp: Path, batch: str, cluster: str) -> tuple[np.ndarray, dict]:
  env = {**os.environ, "PYTHONPATH": ".:extra", "CCACHE": "0", "CACHELEVEL": "0", "PARALLEL": "0"}
  subprocess.check_call(
    [py, "-c", CHILD, str(S), str(T), skip, str(work), str(direct),
     str(outp), str(metap), str(elfp), batch, cluster], env=env)
  return np.load(outp), json.loads(metap.read_text())


def cross_run(label: str, arrs: list[np.ndarray], metas: list[dict]) -> dict:
  base = arrs[0]
  diffs = [float(np.max(np.abs(a - base))) for a in arrs[1:]]
  exact = all(np.array_equal(base, a) for a in arrs[1:])
  finite = all(m.get("finite", False) for m in metas) and all(np.isfinite(a).all() for a in arrs)
  shas = [m.get("elf_sha256") for m in metas]
  uniq = sorted({s for s in shas if s})
  row = {
    "label": label,
    "exact": exact,
    "finite": finite,
    "max_cross": max(diffs) if diffs else 0.0,
    "diffs": diffs,
    "unique_elf_shas": uniq,
    "elf_sha_count": len(uniq),
  }
  print(f"{label}: exact={exact} finite={finite} max_cross={row['max_cross']:.6g} "
        f"diffs={','.join(f'{d:.4g}' for d in diffs)} unique_elfs={len(uniq)}")
  return row


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--runs", type=int, default=4)
  p.add_argument("--batch", type=str, default="1")
  p.add_argument("--cluster", type=str, default="1")
  p.add_argument("--artifacts", type=str, default="extra/rdna3_promote_stab_artifacts")
  p.add_argument("--config", choices=("all", "skip2", "prom", "sdpa", "skip2_nobatch"), default="all")
  args = p.parse_args()
  py = sys.executable
  T = min(args.T, args.S)
  art = Path(args.artifacts)
  art.mkdir(parents=True, exist_ok=True)

  configs = {
    "skip2": ("2", 1, 1, args.batch),
    "prom": ("", 0, 1, args.batch),
    "sdpa": ("2", 1, 0, args.batch),
    "skip2_nobatch": ("2", 1, 1, "0"),
  }
  selected = list(configs) if args.config == "all" else [args.config]

  failed = False
  summary = []
  for name in selected:
    skip, work, direct, batch = configs[name]
    arrs, metas = [], []
    for i in range(args.runs):
      outp = art / f"{name}_{i}.npy"
      metap = art / f"{name}_{i}.json"
      elfp = art / f"{name}_{i}.elf"
      a, m = run_once(py, args.S, T, skip, work, direct, outp, metap, elfp, batch, args.cluster)
      arrs.append(a)
      metas.append(m)
    row = cross_run(name, arrs, metas)
    summary.append(row)
    # Direct paths: require bit-exact + single ELF when ELF recorded. SDPA: bit-exact only.
    bad = (not row["finite"]) or (not row["exact"])
    if direct == 1 and row["elf_sha_count"] > 1:
      bad = True
    if bad:
      failed = True
      (art / f"{name}_FAIL.txt").write_text(json.dumps(row, indent=2))

  (art / "summary.json").write_text(json.dumps(summary, indent=2))
  if failed:
    print(f"FAIL: instability or nonfinite — artifacts kept in {art}")
    raise SystemExit(1)
  print(f"OK: all selected configs bit-exact and finite — artifacts in {art}")
  raise SystemExit(0)


if __name__ == "__main__":
  main()
