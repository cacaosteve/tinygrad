#!/usr/bin/env python3
"""DIRECT vs HIP structural compare on the shipping flash prefill shape.

Same graph (`_amd_flash_attention`, B=1 H=32 tokens=32 D=128 S=2048 kv=8).
Reports wave layout, WMMA count, LDS/global traffic, VGPR/scratch, occupancy bound.

Modes:
  compile   compile-path metrics (DIRECT always; HIP needs comgr/device)
  from-elf  analyze existing ELFs (e.g. extra/rdna3_baseline_freeze/{direct,hip}.elf)
  capture   TinyJit capture on live GPU (DIRECT + HIP devices)
  all       compile + capture when GPU present

Example:
  PYTHONPATH=. python extra/rdna3_structural_compare.py --mode compile
  PYTHONPATH=. python extra/rdna3_structural_compare.py --mode from-elf \\
      --direct-elf extra/rdna3_baseline_freeze/direct.elf \\
      --hip-elf extra/rdna3_baseline_freeze/hip.elf
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from pathlib import Path

SHAPE = dict(B=1, H=32, tokens=32, D=128, S=2048, kv_heads=8)

def _env_base(**extra):
  os.environ.setdefault("PARALLEL", "0")
  os.environ.setdefault("CACHELEVEL", "0")
  os.environ.setdefault("CCACHE", "0")
  os.environ.setdefault("AMD_SOFT_SYSFS", "1")
  # Do not override flash compile defaults (INTERLEAVE/PACK_ALIAS etc.) — stale OFF
  # values here used to force priv172 / MMU and poison the DIRECT vs HIP diff.
  for k, v in extra.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = str(v)

def _sumk(mc: Counter, *substrs: str) -> int:
  return sum(v for k, v in mc.items() if any(s in k for s in substrs))

def analyze_elf(binary: bytes, *, local_size=None, global_size=None, shared_bytes=None) -> dict:
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size

  text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3")
    insts.append(inst)
    off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, sgpr, _, priv = scan_elf_regs(insts, scratch_inst_size)
  wave_vgpr = ((vgpr + 7) // 8) * 8 if vgpr else 8
  # gfx1100: 1536 VGPR / CU. Waves/workgroup = product(local)/32.
  waves_per_wg = 1
  if local_size:
    from math import prod
    waves_per_wg = max(1, prod(local_size) // 32)
  max_wg_vgpr = (1536 // wave_vgpr) // waves_per_wg if wave_vgpr and waves_per_wg else None
  return {
    "ninst": len(insts),
    "max_vgpr": vgpr,
    "max_sgpr": sgpr,
    "private_segment_size": priv,
    "wave_vgpr_alloc": wave_vgpr,
    "waves_per_workgroup": waves_per_wg,
    "max_workgroups_per_cu_vgpr": max_wg_vgpr,
    "max_waves_per_cu_vgpr": 1536 // wave_vgpr if wave_vgpr else None,
    "v_wmma": _sumk(mc, "wmma"),
    "v_mov": mc.get("v_mov_b32_e32", 0),
    "v_dual": _sumk(mc, "dual"),
    "scratch_store": _sumk(mc, "scratch_store"),
    "scratch_load": _sumk(mc, "scratch_load"),
    "ds_load": _sumk(mc, "ds_load"),
    "ds_store": _sumk(mc, "ds_store"),
    "ds_swizzle": mc.get("ds_swizzle_b32", 0),
    "global_load": _sumk(mc, "global_load", "buffer_load", "flat_load"),
    "global_store": _sumk(mc, "global_store", "buffer_store", "flat_store"),
    "s_barrier": mc.get("s_barrier", 0),
    "s_waitcnt": _sumk(mc, "waitcnt"),
    "s_delay_alu": mc.get("s_delay_alu", 0),
    "local_size": list(local_size) if local_size else None,
    "global_size": list(global_size) if global_size else None,
    "shared_bytes": shared_bytes,
    "elf_bytes": len(binary),
    "machine_top": mc.most_common(15),
  }

def graph_layout() -> dict:
  from tinygrad.llm.kernels.amd import (
    BLOCK_M, BLOCK_N, WARP_SIZE, WAVES_M, WAVES_N, THREADS_PER_BLOCK, flash_tile_dims, LDS_PAD,
  )
  TM, TN, TD = flash_tile_dims(SHAPE["D"])
  return {
    "shape": SHAPE,
    "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "WARP_SIZE": WARP_SIZE,
    "WAVES_M": WAVES_M, "WAVES_N": WAVES_N, "THREADS_PER_BLOCK": THREADS_PER_BLOCK,
    "tile_TM_TN_TD": (TM, TN, TD),
    "LDS_PAD": LDS_PAD,
    "note": "Same `_amd_flash_attention` builder for DIRECT and HIP when AMD_FLASH_DIRECT selects path.",
  }

def compile_direct() -> dict:
  _env_base(DEV="AMD:AMD", AMD_FLASH_DIRECT="1")
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv, Target, DEV
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  import tinygrad.llm.kernels.amd as amd

  DEV.value = "AMD:AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    sink = _amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8, use_acc_work=False,
      acc_vgpr=True, hoist_q=True,
    )
    prg = to_program(sink, ren)
  binary = prg.src[-1].arg
  shared = getattr(prg.arg, "shared", None) or getattr(prg.arg, "shared_size", None)
  m = analyze_elf(binary, local_size=prg.arg.local_size, global_size=prg.arg.global_size, shared_bytes=shared)
  m["backend"] = "DIRECT"
  m["renderer"] = "AMDRenderer"
  m["program"] = {"local_size": list(prg.arg.local_size), "global_size": list(prg.arg.global_size),
                  "target": str(prg.arg.target)}
  return m

def compile_hip() -> dict:
  _env_base(DEV="AMD", AMD_FLASH_DIRECT=None)
  from tinygrad import dtypes, Device
  from tinygrad.helpers import Context, getenv, DEV
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.llm.kernels.amd import _amd_flash_attention
  import tinygrad.llm.kernels.amd as amd

  DEV.value = "AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  try:
    ren = Device["AMD"].renderer
  except Exception as e:
    return {"backend": "HIP", "error": f"no AMD device renderer: {e}"}
  with Context(BEAM=0):
    sink = _amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8, use_acc_work=False,
      acc_vgpr=True, hoist_q=True,
    )
    try:
      prg = to_program(sink, ren)
    except Exception as e:
      return {"backend": "HIP", "error": f"to_program failed: {e}"}
  binary = next((bytes(s.arg) for s in prg.src if s.op is Ops.BINARY and isinstance(s.arg, (bytes, bytearray))
                 and bytes(s.arg)[:4] == b"\x7fELF"), None)
  if binary is None:
    src = next((s.arg for s in prg.src if s.op is Ops.SOURCE), "")
    return {"backend": "HIP", "renderer": ren.__class__.__name__, "note": "no ELF",
            "local_size": list(getattr(prg.arg, "local_size", ()) or ()),
            "global_size": list(getattr(prg.arg, "global_size", ()) or ()),
            "source_len": len(src) if isinstance(src, str) else 0}
  m = analyze_elf(binary, local_size=prg.arg.local_size, global_size=prg.arg.global_size)
  m["backend"] = "HIP"
  m["renderer"] = ren.__class__.__name__
  m["program"] = {"local_size": list(prg.arg.local_size), "global_size": list(prg.arg.global_size),
                  "target": str(getattr(prg.arg, "target", ""))}
  return m

def capture_pair() -> dict:
  """Live GPU TinyJit capture for both backends."""
  import numpy as np
  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.kernels.amd import flash_attention
  from tinygrad.helpers import DEV, getenv

  def patterned(shape, dtype=np.float32):
    c = int(np.prod(shape))
    return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

  out = {}
  for direct in (True, False):
    tag = "DIRECT" if direct else "HIP"
    target = "AMD:AMD" if direct else "AMD"
    _env_base(DEV=target, AMD_FLASH_DIRECT=("1" if direct else None))
    DEV.value = target
    getenv.cache_clear()
    q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
    cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
    runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
    for _ in range(2):
      o = runner(q)
      Device[Device.DEFAULT].synchronize()
    arr = o.numpy()
    elfs = []
    seen = set()
    for call in runner.captured.linear.src:
      for program in call.src[0].toposort():
        if program.op is not Ops.PROGRAM or program in seen: continue
        seen.add(program)
        binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
        if isinstance(binary, bytes) and binary.startswith(b"\x7fELF") and len(binary) > 2000:
          elfs.append((program.arg, binary))
    entry = {"device": Device.DEFAULT, "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
             "checksum": float(arr.astype(np.float64).sum()), "finite": bool(np.isfinite(arr).all()),
             "n_elfs": len(elfs)}
    if elfs:
      arg, binary = max(elfs, key=lambda x: len(x[1]))
      entry.update(analyze_elf(binary, local_size=getattr(arg, "local_size", None),
                               global_size=getattr(arg, "global_size", None)))
    else:
      entry["note"] = "no AMD ELF in TinyJit capture"
    out[tag] = entry
  return out

def diff_table(a: dict, b: dict) -> list[dict]:
  keys = [
    "local_size", "global_size", "waves_per_workgroup", "v_wmma", "ninst",
    "max_vgpr", "private_segment_size", "wave_vgpr_alloc", "max_waves_per_cu_vgpr",
    "max_workgroups_per_cu_vgpr", "v_mov", "v_dual", "scratch_store", "scratch_load",
    "ds_load", "ds_store", "ds_swizzle", "global_load", "global_store",
    "s_barrier", "s_waitcnt", "s_delay_alu", "shared_bytes",
  ]
  rows = []
  for k in keys:
    va, vb = a.get(k), b.get(k)
    if va is None and vb is None: continue
    rows.append({"metric": k, "DIRECT": va, "HIP": vb, "delta": None if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) else va - vb})
  return rows

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("compile", "from-elf", "capture", "all"), default="compile")
  ap.add_argument("--direct-elf", type=Path, default=None)
  ap.add_argument("--hip-elf", type=Path, default=None)
  ap.add_argument("--out", type=Path, default=Path("extra/rdna3_structural_compare.json"))
  args = ap.parse_args()

  report: dict = {"graph": graph_layout(), "mode": args.mode}

  if args.mode in ("compile", "all"):
    report["direct_compile"] = compile_direct()
    report["hip_compile"] = compile_hip()
    if "error" not in report["hip_compile"] and "note" not in report.get("hip_compile", {}):
      report["diff_compile"] = diff_table(report["direct_compile"], report["hip_compile"])

  if args.mode == "from-elf" or (args.direct_elf and args.hip_elf):
    d_elf = args.direct_elf or Path("extra/rdna3_baseline_freeze/direct.elf")
    h_elf = args.hip_elf or Path("extra/rdna3_baseline_freeze/hip.elf")
    # Prefer fresh compile local_size for DIRECT; frozen ELFs lack ProgramInfo.
    ls = (32, 2, 2)
    gs = (32, 1, 1)
    report["direct_elf"] = analyze_elf(d_elf.read_bytes(), local_size=ls, global_size=gs)
    report["direct_elf"]["backend"] = "DIRECT"
    report["direct_elf"]["source"] = str(d_elf)
    report["hip_elf"] = analyze_elf(h_elf.read_bytes(), local_size=ls, global_size=gs)
    report["hip_elf"]["backend"] = "HIP"
    report["hip_elf"]["source"] = str(h_elf)
    report["diff_elf"] = diff_table(report["direct_elf"], report["hip_elf"])

  if args.mode in ("capture", "all"):
    try:
      report["capture"] = capture_pair()
      if "DIRECT" in report["capture"] and "HIP" in report["capture"]:
        report["diff_capture"] = diff_table(report["capture"]["DIRECT"], report["capture"]["HIP"])
    except Exception as e:
      report["capture"] = {"error": str(e)}

  # Verdict helpers
  d = report.get("direct_compile") or report.get("direct_elf") or {}
  h = report.get("hip_compile") or report.get("hip_elf") or {}
  if d and h and "v_wmma" in d and "v_wmma" in h:
    report["verdict"] = {
      "same_wmma_count": d.get("v_wmma") == h.get("v_wmma"),
      "same_wave_layout": d.get("local_size") == h.get("local_size"),
      "same_barriers": d.get("s_barrier") == h.get("s_barrier"),
      "lds_load_delta": (d.get("ds_load") or 0) - (h.get("ds_load") or 0),
      "lds_store_delta": (d.get("ds_store") or 0) - (h.get("ds_store") or 0),
      "vgpr_delta": (d.get("max_vgpr") or 0) - (h.get("max_vgpr") or 0),
      "scratch_delta": (d.get("private_segment_size") or 0) - (h.get("private_segment_size") or 0),
      "occ_waves_delta": (d.get("max_waves_per_cu_vgpr") or 0) - (h.get("max_waves_per_cu_vgpr") or 0),
    }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, default=str))
  print(json.dumps({k: report[k] for k in ("graph", "verdict", "diff_compile", "diff_elf", "diff_capture") if k in report}, indent=2, default=str))
  print(f"wrote {args.out}", file=sys.stderr)

if __name__ == "__main__":
  main()
