#!/usr/bin/env python3
"""Half matmul 2048 structural compare DIRECT | LLVM | HIP via TinyJit ELF capture."""
from __future__ import annotations
import json, os, sys
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import DEV, getenv
from tinygrad.codegen import to_program_cache
from tinygrad.uop.ops import Ops

# reuse analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdna3_structural_compare import analyze_elf  # noqa: E402

N = 2048

def capture(dev: str, extra: dict | None = None) -> dict:
  for k in ("TC_LDS_AB", "TC_UPCAST", "TC_LOCAL", "ALLOW_UPCAST16", "TC_UPCAST_TILES", "TC_LDS_UNROLL"):
    os.environ.pop(k, None)
  if extra:
    for k, v in extra.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = str(v)
  os.environ["DEV"] = dev
  DEV.value = dev
  getenv.cache_clear(); to_program_cache.clear()
  rng = np.random.default_rng(0)
  a = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16)).realize()
  b = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16)).realize()
  runner = TinyJit(lambda: (a @ b).realize())
  out = runner(); Device[Device.DEFAULT].synchronize()
  for _ in range(2):
    out = runner(); Device[Device.DEFAULT].synchronize()
  got = out.numpy().astype(np.float32)
  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  err = float(np.max(np.abs(got - ref)))
  elfs = []
  seen = set()
  for call in runner.captured.linear.src:
    for program in call.src[0].toposort():
      if program.op is not Ops.PROGRAM or program in seen: continue
      seen.add(program)
      binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
      if isinstance(binary, bytes) and binary.startswith(b"\x7fELF") and len(binary) > 500:
        elfs.append((program.arg, binary))
  entry = {
    "dev": Device.DEFAULT,
    "ren": type(Device[Device.DEFAULT].renderer).__name__,
    "extra": extra or {},
    "max_err": err,
    "ok": err < 1.0,
    "n_elfs": len(elfs),
  }
  if elfs:
    arg, binary = max(elfs, key=lambda x: len(x[1]))
    entry.update(analyze_elf(binary, local_size=getattr(arg, "local_size", None),
                             global_size=getattr(arg, "global_size", None),
                             shared_bytes=getattr(arg, "shared", None) or getattr(arg, "shared_size", None)))
  return entry

def main():
  rows = []
  for label, dev, extra in [
    ("direct_ship", "AMD:AMD", {}),
    ("direct_loc3", "AMD:AMD", {"TC_LOCAL": "3"}),
    ("llvm", "AMD:LLVM", {}),
    ("hip", "AMD", {}),
  ]:
    try:
      r = capture(dev, extra)
      r["label"] = label
      print(json.dumps({k: r[k] for k in r if k != "machine_top"}), flush=True)
      rows.append(r)
    except Exception as e:
      print(json.dumps({"label": label, "error": repr(e)}), flush=True)
  if len(rows) >= 2:
    keys = ["v_wmma", "ninst", "max_vgpr", "private_segment_size", "ds_load", "ds_store",
            "global_load", "global_store", "v_mov", "v_dual", "s_barrier", "s_waitcnt",
            "local_size", "global_size", "shared_bytes", "max_waves_per_cu_vgpr"]
    base = next((r for r in rows if r.get("label") == "hip"), rows[-1])
    print("=== vs HIP ===", flush=True)
    for r in rows:
      if r is base: continue
      d = {"label": r.get("label")}
      for k in keys:
        d[k] = {"this": r.get(k), "hip": base.get(k)}
      print(json.dumps(d), flush=True)

if __name__ == "__main__":
  main()
