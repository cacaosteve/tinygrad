#!/usr/bin/env python3
"""Compile + optional HW gate for prefill ACC promote (kill intentional slot-2 scratch).

Configs measure: SPILL/FILL, priv, vgpr, scratch ops, ninst, v_mov.
HW modes need gaming PC.

Example:
  PYTHONPATH=. python extra/rdna3_acc_promote_ab.py --mode compile
  PYTHONPATH=. python extra/rdna3_acc_promote_ab.py --mode sync --configs base,promote_remat
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from collections import Counter
from pathlib import Path

CONFIGS = {
  # shipping
  "base": dict(AMD_FLASH_PREFILL_SKIP_SLOTS="2", AMD_FLASH_ACC_WORK="1",
               AMD_REMAT_CONST_MOV="0", AMD_REMAT_ADDR="0", AMD_SPILL_ON_EVICT="0"),
  # promote ACC to VGPR (no work-copy)
  "promote": dict(AMD_FLASH_PREFILL_SKIP_SLOTS="", AMD_FLASH_ACC_WORK="0",
                  AMD_REMAT_CONST_MOV="0", AMD_REMAT_ADDR="0", AMD_SPILL_ON_EVICT="0"),
  "promote_remat": dict(AMD_FLASH_PREFILL_SKIP_SLOTS="", AMD_FLASH_ACC_WORK="0",
                        AMD_REMAT_CONST_MOV="1", AMD_REMAT_ADDR="1", AMD_REMAT_ADDR_DEEP="1",
                        AMD_SPILL_ON_EVICT="0"),
  "promote_remat_evict": dict(AMD_FLASH_PREFILL_SKIP_SLOTS="", AMD_FLASH_ACC_WORK="0",
                             AMD_REMAT_CONST_MOV="1", AMD_REMAT_ADDR="1", AMD_REMAT_ADDR_DEEP="1",
                             AMD_SPILL_ON_EVICT="1"),
  # keep scratch ACC but remat consts (control)
  "base_remat": dict(AMD_FLASH_PREFILL_SKIP_SLOTS="2", AMD_FLASH_ACC_WORK="1",
                     AMD_REMAT_CONST_MOV="1", AMD_REMAT_ADDR="0", AMD_SPILL_ON_EVICT="0"),
}

def _env(**cfg):
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  os.environ.setdefault("AMD_SOFT_SYSFS", "1")
  os.environ.setdefault("AMD_MOV_ALIAS_EXTRACT", "1")
  os.environ.setdefault("AMD_MOV_ALIAS_ALU", "1")
  os.environ.setdefault("AMD_VOPD_MOV_STREAM", "1")
  os.environ.setdefault("AMD_WMMA_INTERLEAVE", "0")
  os.environ.setdefault("AMD_PACK_F16_ALIAS_LLOAD", "0")
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  for k, v in cfg.items():
    if v is None or v == "":
      # empty SKIP means promote: set to empty string explicitly
      if k == "AMD_FLASH_PREFILL_SKIP_SLOTS":
        os.environ[k] = ""
        os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
      else:
        os.environ.pop(k, None)
    else:
      os.environ[k] = str(v)
      if k == "AMD_FLASH_PREFILL_SKIP_SLOTS":
        os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = str(v)

def compile_metrics(name: str, cfg: dict) -> dict:
  _env(**cfg)
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv, Target, DEV
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.renderer.isa.rdna3 import AMDOps, _iop
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  import tinygrad.llm.kernels.amd as amd

  DEV.value = "AMD:AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  # re-apply skip after cache clear (compile_env also sets it)
  skip = cfg.get("AMD_FLASH_PREFILL_SKIP_SLOTS", "2")
  os.environ["AMD_FLASH_PREFILL_SKIP_SLOTS"] = skip
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  for k in ("AMD_REMAT_CONST_MOV", "AMD_REMAT_ADDR", "AMD_SPILL_ON_EVICT", "AMD_FLASH_ACC_WORK"):
    if k in cfg: os.environ[k] = str(cfg[k])

  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  use_acc = int(cfg.get("AMD_FLASH_ACC_WORK", "1")) and ("2" in {s.strip() for s in skip.split(",")} or skip.strip() == "2")
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    # force skip inside env (prefill compile_env overwrites SKIP)
    os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
    os.environ["AMD_FLASH_PREFILL_SKIP_SLOTS"] = skip
    sink = _amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=bool(use_acc),
    )
    prg = to_program(sink, ren)

  ops = [u for u in prg.src[1].src if u.op is Ops.INS]
  spill = fill = scratch_size = 0
  for u in ops:
    try: op = _iop(u)
    except Exception: continue
    n = op.name
    if n == "SPILL": spill += 1
    elif n == "FILL": fill += 1
    elif n == "SCRATCH_SIZE": scratch_size += 1

  binary = prg.src[-1].arg
  text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, sgpr, _, priv = scan_elf_regs(insts, scratch_inst_size)
  return {
    "name": name, "cfg": cfg, "use_acc_work": use_acc,
    "ninst": len(insts), "max_vgpr": vgpr, "private_segment_size": priv,
    "IR_SPILL": spill, "IR_FILL": fill, "IR_SCRATCH_SIZE": scratch_size,
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "scratch_store": sum(v for k, v in mc.items() if k.startswith("scratch_store")),
    "scratch_load": sum(v for k, v in mc.items() if k.startswith("scratch_load")),
    "v_wmma": sum(v for k, v in mc.items() if "wmma" in k),
    "local_size": list(prg.arg.local_size),
  }

def sync_each(name: str, cfg: dict, *, warmup=5, samples=25) -> dict:
  _env(**cfg)
  from tinygrad import Tensor, Device, TinyJit, dtypes
  from tinygrad.helpers import DEV, getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  import numpy as np

  DEV.value = "AMD:AMD"
  getenv.cache_clear(); amd._amd_flash_attention.cache_clear()
  skip = cfg.get("AMD_FLASH_PREFILL_SKIP_SLOTS", "2")
  os.environ["AMD_FLASH_PREFILL_SKIP_SLOTS"] = skip
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  for k in ("AMD_REMAT_CONST_MOV", "AMD_REMAT_ADDR", "AMD_SPILL_ON_EVICT", "AMD_FLASH_ACC_WORK"):
    if k in cfg: os.environ[k] = str(cfg[k])

  Tensor.manual_seed(0)
  q = Tensor.randn(1, 32, 32, 128, dtype=dtypes.float16).realize()
  cache = Tensor.randn(2, 1, 8, 2048, 128, dtype=dtypes.float16).realize()
  fn = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
  for _ in range(warmup):
    out = fn(q); Device[Device.DEFAULT].synchronize()
  times = []
  for _ in range(samples):
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    out = fn(q)
    Device[Device.DEFAULT].synchronize()
    times.append((time.perf_counter() - t0) * 1e6)
  arr = out.numpy()
  return {
    "name": name,
    "median_us": statistics.median(times),
    "best_us": min(times),
    "stdev_us": statistics.pstdev(times) if len(times) > 1 else 0.0,
    "checksum": float(arr.astype("float64").sum()),
    "finite": bool(__import__("numpy").isfinite(arr).all()),
  }

def hip_sync(*, warmup=5, samples=25) -> dict:
  os.environ["DEV"] = "AMD"
  os.environ.pop("AMD_FLASH_DIRECT", None)
  from tinygrad import Tensor, Device, TinyJit, dtypes
  from tinygrad.helpers import DEV, getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  DEV.value = "AMD"
  getenv.cache_clear(); amd._amd_flash_attention.cache_clear()
  Tensor.manual_seed(0)
  q = Tensor.randn(1, 32, 32, 128, dtype=dtypes.float16).realize()
  cache = Tensor.randn(2, 1, 8, 2048, 128, dtype=dtypes.float16).realize()
  fn = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
  for _ in range(warmup):
    out = fn(q); Device[Device.DEFAULT].synchronize()
  times = []
  for _ in range(samples):
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    out = fn(q)
    Device[Device.DEFAULT].synchronize()
    times.append((time.perf_counter() - t0) * 1e6)
  arr = out.numpy()
  return {
    "name": "HIP",
    "median_us": statistics.median(times),
    "best_us": min(times),
    "stdev_us": statistics.pstdev(times) if len(times) > 1 else 0.0,
    "checksum": float(arr.astype("float64").sum()),
    "finite": bool(__import__("numpy").isfinite(arr).all()),
  }

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("compile", "sync", "all"), default="compile")
  ap.add_argument("--configs", default="base,promote,promote_remat,promote_remat_evict,base_remat")
  ap.add_argument("--out", type=Path, default=Path("extra/rdna3_acc_promote_ab.json"))
  args = ap.parse_args()
  names = [n.strip() for n in args.configs.split(",") if n.strip()]
  report: dict = {"configs": names}
  if args.mode in ("compile", "all"):
    rows = []
    for n in names:
      try:
        m = compile_metrics(n, CONFIGS[n])
        rows.append(m)
        print(json.dumps(m), flush=True)
      except Exception as e:
        rows.append({"name": n, "error": str(e)})
        print(n, "ERROR", e, flush=True)
    report["compile"] = rows
  if args.mode in ("sync", "all"):
    syncs = []
    for n in names:
      try:
        s = sync_each(n, CONFIGS[n])
        syncs.append(s)
        print("SYNC", json.dumps(s), flush=True)
      except Exception as e:
        syncs.append({"name": n, "error": str(e)})
        print("SYNC", n, "ERROR", e, flush=True)
    try:
      h = hip_sync()
      syncs.append(h)
      print("SYNC", json.dumps(h), flush=True)
    except Exception as e:
      syncs.append({"name": "HIP", "error": str(e)})
    report["sync"] = syncs
  args.out.write_text(json.dumps(report, indent=2))
  print("wrote", args.out, file=sys.stderr)

if __name__ == "__main__":
  main()
