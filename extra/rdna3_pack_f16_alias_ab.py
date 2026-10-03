#!/usr/bin/env python3
"""HW gate for AMD_PACK_F16_ALIAS_LLOAD — one-class PACK_F16←LLOAD alias.

Runs only when gaming PC is up. Fresh-process friendly via --mode.

Modes:
  compile   compile metrics only (OFF vs ON)
  sync      DIRECT OFF / DIRECT ON / HIP sync-each (fixed seed, same tensors)
  correct   output compare DIRECT OFF/ON vs HIP (short+long contexts)
  fence     TinyJit batch 50/100 fence exercise (DIRECT OFF then ON)
  all       sync + correct + fence + compile

Acceptance (manual): correct, no spill/VGPR regress, timing win beyond noise.
Default stays OFF until accepted.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from collections import Counter
from pathlib import Path

SHAPE_PREFILL = dict(B=1, H=32, tokens=32, D=128, S=2048, kv_heads=8)
SHAPE_SHORT = dict(B=1, H=32, tokens=1, D=128, S=128, kv_heads=8)
SHAPE_LONG = dict(B=1, H=32, tokens=32, D=128, S=2048, kv_heads=8)

def _env_base(**extra):
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  os.environ.setdefault("AMD_SOFT_SYSFS", "1")
  os.environ.setdefault("AMD_MOV_ALIAS_EXTRACT", "1")
  os.environ.setdefault("AMD_MOV_ALIAS_ALU", "1")
  os.environ.setdefault("AMD_VOPD_MOV_STREAM", "1")
  os.environ.setdefault("AMD_WMMA_INTERLEAVE", "0")
  for k, v in extra.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = str(v)

def compile_metrics(alias: int) -> dict:
  _env_base(DEV="AMD:AMD", AMD_FLASH_DIRECT="1", AMD_PACK_F16_ALIAS_LLOAD=alias)
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv, Target
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  getenv.cache_clear(); to_program_cache.clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    prg = to_program(
      _amd_flash_attention(
        UOp.placeholder((32, 32, 128), dtypes.float32, 0),
        UOp.placeholder((32, 32, 128), dtypes.float16, 1),
        UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
        valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=True,
      ),
      ren,
    )
  binary = prg.src[-1].arg
  text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, sgpr, _, priv = scan_elf_regs(insts, scratch_inst_size)
  import hashlib
  return {
    "alias": alias,
    "ninst": len(insts),
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "v_dual": sum(v for k, v in mc.items() if "dual" in k),
    "scratch_store": sum(v for k, v in mc.items() if k.startswith("scratch_store")),
    "scratch_load": sum(v for k, v in mc.items() if k.startswith("scratch_load")),
    "max_vgpr": vgpr,
    "max_sgpr": sgpr,
    "private_segment_size": priv,
    "elf_sha256": hashlib.sha256(binary).hexdigest()[:16],
    "elf_bytes": len(binary),
  }

def _make_tensors(shape: dict, seed: int = 0):
  from tinygrad import Tensor, dtypes
  Tensor.manual_seed(seed)
  q = Tensor.randn(shape["B"], shape["H"], shape["tokens"], shape["D"], dtype=dtypes.float16).realize()
  cache = Tensor.randn(2, shape["B"], shape["kv_heads"], shape["S"], shape["D"], dtype=dtypes.float16).realize()
  return q, cache

def sync_each(*, direct: bool, alias: int | None, q, cache, S: int, warmup=5, samples=30) -> dict:
  from tinygrad import Device, TinyJit
  from tinygrad.helpers import getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  if direct:
    _env_base(DEV="AMD:AMD", AMD_FLASH_DIRECT="1", AMD_PACK_F16_ALIAS_LLOAD=alias)
  else:
    _env_base(DEV="AMD", AMD_FLASH_DIRECT="0", AMD_PACK_F16_ALIAS_LLOAD=None)
    os.environ.pop("AMD_PACK_F16_ALIAS_LLOAD", None)
  getenv.cache_clear()
  amd._amd_flash_attention.cache_clear()
  runner = TinyJit(lambda qq: flash_attention(qq, cache, S).realize())
  for _ in range(warmup):
    out = runner(q); Device[Device.DEFAULT].synchronize()
  arr = out.numpy()
  xs = []
  for _ in range(samples):
    t0 = time.perf_counter_ns()
    runner(q); Device[Device.DEFAULT].synchronize()
    xs.append((time.perf_counter_ns() - t0) / 1e3)
  return {
    "direct": direct,
    "alias": alias,
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "checksum": float(arr.astype("float64").sum()),
    "finite": bool(__import__("numpy").isfinite(arr).all()),
    "median_us": statistics.median(xs),
    "best_us": min(xs),
    "mean_us": statistics.mean(xs),
    "stdev_us": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    "n": samples,
  }

def correct_vs_hip(shape: dict, seed: int = 0, atol: float = 2e-2, rtol: float = 2e-2) -> dict:
  import numpy as np
  from tinygrad import Device, TinyJit
  from tinygrad.helpers import getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  q, cache = _make_tensors(shape, seed)
  S = shape["S"]
  outs = {}
  for tag, direct, alias in (("hip", False, None), ("off", True, 0), ("on", True, 1)):
    if direct:
      _env_base(DEV="AMD:AMD", AMD_FLASH_DIRECT="1", AMD_PACK_F16_ALIAS_LLOAD=alias)
    else:
      _env_base(DEV="AMD", AMD_FLASH_DIRECT="0")
      os.environ.pop("AMD_PACK_F16_ALIAS_LLOAD", None)
    getenv.cache_clear(); amd._amd_flash_attention.cache_clear()
    runner = TinyJit(lambda qq: flash_attention(qq, cache, S).realize())
    for _ in range(2):
      out = runner(q); Device[Device.DEFAULT].synchronize()
    outs[tag] = out.numpy()
  ref = outs["hip"]
  def cmp(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()): return {"ok": False, "reason": "nan"}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-6)
    return {
      "ok": bool(np.all(diff <= atol + rtol * denom)),
      "max_abs": float(diff.max()),
      "mean_abs": float(diff.mean()),
      "checksum_a": float(a.astype(np.float64).sum()),
      "checksum_b": float(b.astype(np.float64).sum()),
    }
  return {
    "shape": shape,
    "off_vs_hip": cmp(outs["off"], ref),
    "on_vs_hip": cmp(outs["on"], ref),
    "on_vs_off": cmp(outs["on"], outs["off"]),
  }

def fence_batch(*, alias: int, batch: int = 50) -> dict:
  from tinygrad import Device, TinyJit
  from tinygrad.helpers import getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  _env_base(DEV="AMD:AMD", AMD_FLASH_DIRECT="1", AMD_PACK_F16_ALIAS_LLOAD=alias)
  getenv.cache_clear(); amd._amd_flash_attention.cache_clear()
  q, cache = _make_tensors(SHAPE_PREFILL, seed=0)
  S = SHAPE_PREFILL["S"]
  runner = TinyJit(lambda qq: flash_attention(qq, cache, S).realize())
  for _ in range(3):
    out = runner(q); Device[Device.DEFAULT].synchronize()
  t0 = time.perf_counter_ns()
  for _ in range(batch):
    out = runner(q)
  Device[Device.DEFAULT].synchronize()
  elapsed = (time.perf_counter_ns() - t0) / 1e3
  arr = out.numpy()
  return {
    "alias": alias,
    "batch": batch,
    "us_per_call": elapsed / batch,
    "total_us": elapsed,
    "checksum": float(arr.astype("float64").sum()),
    "finite": bool(__import__("numpy").isfinite(arr).all()),
  }

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--mode", choices=("compile", "sync", "correct", "fence", "all"), default="all")
  p.add_argument("--out", type=Path, default=None)
  args = p.parse_args()
  report: dict = {"tip": "AMD_PACK_F16_ALIAS_LLOAD HW gate", "modes": {}}

  if args.mode in ("compile", "all"):
    report["modes"]["compile"] = [compile_metrics(0), compile_metrics(1)]
    c0, c1 = report["modes"]["compile"]
    report["compile_delta"] = {k: c1[k] - c0[k] for k in c0 if isinstance(c0[k], (int, float))}

  if args.mode in ("sync", "all"):
    q, cache = _make_tensors(SHAPE_PREFILL, seed=0)
    S = SHAPE_PREFILL["S"]
    # Alternating in one session: OFF → ON → HIP
    rows = [
      sync_each(direct=True, alias=0, q=q, cache=cache, S=S),
      sync_each(direct=True, alias=1, q=q, cache=cache, S=S),
      sync_each(direct=False, alias=None, q=q, cache=cache, S=S),
    ]
    report["modes"]["sync"] = rows
    report["sync_delta_us_on_minus_off"] = rows[1]["median_us"] - rows[0]["median_us"]
    report["sync_checksum_match_on_off"] = abs(rows[1]["checksum"] - rows[0]["checksum"]) < 1e-2

  if args.mode in ("correct", "all"):
    report["modes"]["correct"] = {
      "short": correct_vs_hip(SHAPE_SHORT, seed=0),
      "long": correct_vs_hip(SHAPE_LONG, seed=0),
    }

  if args.mode in ("fence", "all"):
    report["modes"]["fence"] = {
      "off_50": fence_batch(alias=0, batch=50),
      "on_50": fence_batch(alias=1, batch=50),
      "off_100": fence_batch(alias=0, batch=100),
      "on_100": fence_batch(alias=1, batch=100),
    }

  # Gate summary
  bad = []
  if "compile" in report["modes"]:
    c0, c1 = report["modes"]["compile"]
    if c1["v_mov"] >= c0["v_mov"]: bad.append("no_vmov_win")
    if c1["scratch_store"] > c0["scratch_store"]: bad.append("scratch_up")
    if c1["max_vgpr"] > c0["max_vgpr"] + 4: bad.append("vgpr_up")
    if c1["private_segment_size"] > c0["private_segment_size"]: bad.append("priv_up")
  if "sync" in report["modes"]:
    if not report["sync_checksum_match_on_off"]: bad.append("checksum_off_on")
    if not all(r["finite"] for r in report["modes"]["sync"]): bad.append("nan_sync")
  if "correct" in report["modes"]:
    for name, block in report["modes"]["correct"].items():
      if not block["on_vs_off"]["ok"]: bad.append(f"correct_{name}_on_vs_off")
      if not block["on_vs_hip"]["ok"]: bad.append(f"correct_{name}_on_vs_hip")
  if "fence" in report["modes"]:
    for k, v in report["modes"]["fence"].items():
      if not v["finite"]: bad.append(f"fence_{k}_nan")

  report["gate"] = "FAIL" if bad else "PASS"
  report["gate_bad"] = bad
  text = json.dumps(report, indent=2)
  print(text)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  return 1 if bad else 0

if __name__ == "__main__":
  sys.exit(main())
