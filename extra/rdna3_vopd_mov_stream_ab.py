#!/usr/bin/env python3
"""Compile + HW sync-each A/B for AMD_VOPD_MOV_STREAM."""
from __future__ import annotations
import json, os, statistics, sys, time
from collections import Counter

def compile_metrics(stream: int) -> dict:
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_MOV_ALIAS_EXTRACT"] = "1"
  os.environ["AMD_MOV_ALIAS_ALU"] = "1"
  os.environ["AMD_VOPD_MOV_STREAM"] = str(stream)
  os.environ["AMD_WMMA_INTERLEAVE"] = "0"
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer import Target
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  getenv.cache_clear(); to_program_cache.clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    prg = to_program(
      _amd_flash_attention(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=True),
      ren,
    )
  text = next(s.content for s in elf_loader(prg.src[3].arg)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  return {
    "stream": stream,
    "ninst": len(insts),
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "v_dual": sum(v for k, v in mc.items() if "dual" in k),
    "v_dual_mov": mc.get("v_dual_mov_b32", 0),
  }

def hw_sync_each(stream: int, *, q, cache, warmup: int = 5, samples: int = 30) -> dict:
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_SOFT_SYSFS"] = "1"
  os.environ["AMD_MOV_ALIAS_EXTRACT"] = "1"
  os.environ["AMD_MOV_ALIAS_ALU"] = "1"
  os.environ["AMD_VOPD_MOV_STREAM"] = str(stream)
  os.environ["AMD_WMMA_INTERLEAVE"] = "0"
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  from tinygrad import Device, TinyJit
  from tinygrad.helpers import getenv
  from tinygrad.llm.kernels.amd import flash_attention
  import tinygrad.llm.kernels.amd as amd
  getenv.cache_clear()
  amd._amd_flash_attention.cache_clear()
  runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
  for _ in range(warmup):
    out = runner(q); Device[Device.DEFAULT].synchronize()
  arr = out.numpy()
  xs = []
  for _ in range(samples):
    t0 = time.perf_counter_ns()
    runner(q); Device[Device.DEFAULT].synchronize()
    xs.append((time.perf_counter_ns() - t0) / 1e3)
  return {
    "stream": stream,
    "checksum": float(arr.astype("float64").sum()),
    "finite": bool(__import__("numpy").isfinite(arr).all()),
    "median_us": statistics.median(xs),
    "best_us": min(xs),
    "mean_us": statistics.mean(xs),
    "n": samples,
  }

def main():
  hw = "--hw" in sys.argv
  rows = [compile_metrics(0), compile_metrics(1)]
  report: dict = {"compile": rows, "delta": {k: rows[1][k] - rows[0][k] for k in rows[0] if isinstance(rows[0][k], (int, float))}}
  if hw:
    from tinygrad import Tensor, dtypes
    Tensor.manual_seed(0)
    q = Tensor.randn(1, 32, 32, 128, dtype=dtypes.float16).realize()
    cache = Tensor.randn(2, 1, 8, 2048, 128, dtype=dtypes.float16).realize()
    off = hw_sync_each(0, q=q, cache=cache)
    on = hw_sync_each(1, q=q, cache=cache)
    report["hw"] = [off, on]
    report["hw_delta_us"] = on["median_us"] - off["median_us"]
    report["checksum_match"] = abs(on["checksum"] - off["checksum"]) < 1e-2
  print(json.dumps(report, indent=2))
  bad = []
  if rows[1]["v_dual"] <= rows[0]["v_dual"]: bad.append("no_dual_win")
  if rows[1]["v_mov"] >= rows[0]["v_mov"]: bad.append("no_vmov_win")
  if hw:
    if not report["hw"][0]["finite"] or not report["hw"][1]["finite"]: bad.append("nan")
    if not report["checksum_match"]: bad.append("checksum")
  print("GATE", "FAIL" if bad else "PASS", bad)
  return 1 if bad else 0

if __name__ == "__main__":
  sys.exit(main())
