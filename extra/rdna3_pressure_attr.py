#!/usr/bin/env python3
"""Pressure attribution helpers for the post-fence flash baseline diagnosis."""
from __future__ import annotations

import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

from tinygrad import dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Context, DEV, getenv
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
from tinygrad.renderer import Target
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, _iop, greg
from tinygrad.uop.ops import Ops, UOp


def describe(u: UOp | None, depth: int = 0) -> str:
  if u is None or depth > 6: return "?"
  if u.op is Ops.CONST: return f"C:{u.arg}"
  if u.op is Ops.CAST:
    return f"CAST({u.dtype})<-{describe(u.src[0] if u.src else None, depth + 1)}"
  if u.op is Ops.INS:
    try: name = _iop(u).name
    except Exception: name = "INS"
    if name == "SCRATCH_ADDR" and u.src:
      return f"SCRATCH_ADDR({describe(u.src[0], depth + 1)})"
    if name == "MOV" and u.src:
      return f"MOV({describe(u.src[0], depth + 1)})"
    if name == "CAST" and u.src:
      return f"INS_CAST({u.dtype})<-{describe(u.src[0], depth + 1)}"
    return name
  return u.op.name


def compile_flash(*, use_acc_work: bool, acc_slot: int = 2):
  to_program_cache.clear(); getenv.cache_clear()
  if acc_slot != 2:
    amd_path = Path(__file__).resolve().parents[1] / "tinygrad" / "llm" / "kernels" / "amd.py"
    src = amd_path.read_text()
    patched = src.replace(
      "  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
      f"  acc, m_i, l_i = _reg((TM, TD), {acc_slot}, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
      1,
    )
    if use_acc_work is False:
      patched = patched.replace(
        'use_acc_work = bool(use_acc_small and (2 in _skip_reg) and getenv("AMD_FLASH_ACC_WORK", 1))',
        "use_acc_work = False  # diagnosis",
        1,
      )
    if patched == src:
      raise RuntimeError("failed to patch amd.py")
    tmp = Path("/tmp/_amd_pressure_attr.py")
    tmp.write_text(patched)
    import importlib.util
    spec = importlib.util.spec_from_file_location("amd_pressure_attr", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["amd_pressure_attr"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    builder = mod._amd_flash_attention
  else:
    builder = _amd_flash_attention

  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    sink = builder(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1,
                   k_midstore=4, use_acc_work=use_acc_work)
    prg = to_program(sink, ren)
  ops = [u for u in prg.src[1].src if u.op is Ops.INS]
  return ren, prg, ops


def live_maps(ops):
  first_def: dict[int, int] = {}
  last_use: dict[int, int] = {}
  def_uop: dict[int, UOp] = {}
  for i, u in enumerate(ops):
    if isinstance(u.tag, tuple):
      for t in u.tag:
        if isinstance(t, Register):
          first_def.setdefault(id(t), i)
          def_uop[id(t)] = u
    for s in u.src:
      try: r = greg(s)
      except Exception: r = None
      if isinstance(r, Register):
        last_use[id(r)] = i
  return first_def, last_use, def_uop


def analyze_slot19(out: Path):
  ren, prg, ops = compile_flash(use_acc_work=False, acc_slot=19)
  first_def, last_use, def_uop = live_maps(ops)
  wmma = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
  spill = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.SPILL]
  fill = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.FILL]
  fw, lw = wmma[0], wmma[-1]

  spill_info = []
  for i in spill:
    val = ops[i].src[0] if ops[i].src else None
    try: vr = greg(val) if val is not None else None
    except Exception: vr = None
    rid = id(vr) if isinstance(vr, Register) else None
    fd, lu = first_def.get(rid), last_use.get(rid)
    du = def_uop.get(rid)
    spill_info.append({
      "spill_i": i,
      "desc": describe(val),
      "def_i": fd,
      "last_use": lu,
      "lifetime": (None if fd is None or lu is None else lu - fd),
      "def_op": (_iop(du).name if du is not None and du.op is Ops.INS else (du.op.name if du else None)),
      "def_desc": describe(du) if du else None,
      "cross_wmma": bool(fd is not None and lu is not None and fd < fw and lu > lw),
      "wmma_before": sum(1 for w in wmma if w < i),
      "rematerialize": bool(du is not None and ren.rematerialize(du)),
    })

  # peak live at WMMA
  peak_i, peak_n, peak_ids = None, -1, []
  for wi in wmma:
    live = [rid for rid, fd in first_def.items() if fd <= wi and last_use.get(rid, fd) >= wi]
    if len(live) > peak_n:
      peak_n, peak_i, peak_ids = len(live), wi, live

  peak_cls = Counter()
  peak_addr = []
  for rid in peak_ids:
    du = def_uop[rid]
    if du.op is Ops.INS:
      n = _iop(du).name
      peak_cls[n] += 1
      d = describe(du)
      if n in ("CAST", "MOV", "ADD", "SCRATCH_ADDR", "SHL", "SHR") or "CAST" in d or d.startswith("C:"):
        peak_addr.append({
          "desc": d, "def_i": first_def[rid], "last_use": last_use.get(rid),
          "cross_wmma": first_def[rid] < fw and (last_use.get(rid) or 0) > lw,
          "rematerialize": ren.rematerialize(du),
        })
    else:
      peak_cls[du.op.name] += 1

  report = {
    "acc_slot": 19,
    "use_acc_work": False,
    "n_ops": len(ops),
    "n_spill": len(spill),
    "n_fill": len(fill),
    "wmma_window": [fw, lw],
    "spills_before": sum(1 for s in spill_info if s["spill_i"] < fw),
    "spills_during": sum(1 for s in spill_info if fw <= s["spill_i"] <= lw),
    "spills_after": sum(1 for s in spill_info if s["spill_i"] > lw),
    "spill_desc_top": Counter(s["desc"] for s in spill_info).most_common(20),
    "def_op_top": Counter(s["def_op"] for s in spill_info).most_common(15),
    "cross_wmma_spills": sum(1 for s in spill_info if s["cross_wmma"]),
    "spill_rematerialize_true": sum(1 for s in spill_info if s["rematerialize"]),
    "lifetime_hist": Counter(s["lifetime"] for s in spill_info).most_common(15),
    "peak_live_at_wmma_idx": peak_i,
    "peak_live_count": peak_n,
    "peak_class": peak_cls.most_common(20),
    "peak_addr_like": peak_addr[:40],
    "peak_addr_cross_wmma": sum(1 for a in peak_addr if a["cross_wmma"]),
    "sites_sample": spill_info[:12] + spill_info[-8:],
  }
  (out / "slot19_spill_attr.json").write_text(json.dumps(report, indent=2, default=str))
  return report


def analyze_baseline_peak(out: Path):
  ren, prg, ops = compile_flash(use_acc_work=True, acc_slot=2)
  first_def, last_use, def_uop = live_maps(ops)
  wmma = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
  fw, lw = wmma[0], wmma[-1]
  peak_i, peak_n, peak_ids = None, -1, []
  for wi in wmma:
    live = [rid for rid, fd in first_def.items() if fd <= wi and last_use.get(rid, fd) >= wi]
    if len(live) > peak_n:
      peak_n, peak_i, peak_ids = len(live), wi, live

  peak_cls = Counter()
  cross_addr = []
  for rid in peak_ids:
    du = def_uop[rid]
    if du.op is Ops.INS:
      n = _iop(du).name
      peak_cls[n] += 1
      d = describe(du)
      if first_def[rid] < fw and (last_use.get(rid) or 0) > lw:
        cross_addr.append({
          "op": n, "desc": d, "def_i": first_def[rid], "last_use": last_use.get(rid),
          "rematerialize": ren.rematerialize(du),
        })
    else:
      peak_cls[du.op.name] += 1

  report = {
    "n_ops": len(ops),
    "wmma_window": [fw, lw],
    "peak_live_at_wmma_idx": peak_i,
    "peak_live_count": peak_n,
    "peak_class": peak_cls.most_common(25),
    "live_across_full_wmma": len(cross_addr),
    "live_across_full_wmma_sample": cross_addr[:40],
    "live_across_by_op": Counter(a["op"] for a in cross_addr).most_common(20),
    "live_across_remat_true": sum(1 for a in cross_addr if a["rematerialize"]),
  }
  (out / "peak_live.json").write_text(json.dumps(report, indent=2, default=str))
  return report


def elf_metrics(prg) -> dict:
  from collections import Counter
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
  from tinygrad.runtime.support.elf import elf_loader

  binary = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  assert isinstance(binary, bytes) and binary.startswith(b"\x7fELF")
  text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3")
    insts.append(inst)
    off += inst.size()
    name = getattr(inst, "op_name", None)
    if name is None:
      name = getattr(getattr(inst, "op", None), "name", None) or getattr(inst, "opx", type(inst).__name__)
      if not isinstance(name, str): name = str(name)
    if str(name).lower() == "s_endpgm": break

  def opname(i) -> str:
    n = getattr(i, "op_name", None)
    if isinstance(n, str): return n.lower()
    try: return i.op_name.lower()
    except Exception:
      op = getattr(i, "op", None) or getattr(i, "opx", None)
      return str(getattr(op, "name", type(i).__name__)).lower()

  mc = Counter(opname(i) for i in insts)
  vgpr, sgpr, _, priv = scan_elf_regs(insts, scratch_inst_size)
  return {
    "elf_bytes": len(binary),
    "ninst": len(insts),
    "max_vgpr": vgpr,
    "max_sgpr": sgpr,
    "private_segment_size": priv,
    "v_mov_b32": mc.get("v_mov_b32_e32", 0) + mc.get("v_mov_b32", 0),
    "scratch_store_b32": mc.get("scratch_store_b32", 0),
    "scratch_store_b128": mc.get("scratch_store_b128", 0),
    "scratch_load_b32": mc.get("scratch_load_b32", 0),
    "scratch_load_b128": mc.get("scratch_load_b128", 0),
    "v_wmma": mc.get("v_wmma_f32_16x16x16_f16", 0),
    "s_delay_alu": mc.get("s_delay_alu", 0),
    "v_dual": sum(v for k, v in mc.items() if "dual" in k),
    "machine_top": mc.most_common(40),
    "binary": binary,
    "asm_lines": [opname(i) for i in insts],
  }


def capture_elfs(out: Path):
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  DEV.value = "AMD:AMD"
  getenv.cache_clear()
  _, prg_d, _ = compile_flash(use_acc_work=True, acc_slot=2)
  md = elf_metrics(prg_d)
  (out / "direct.elf").write_bytes(md.pop("binary"))
  (out / "direct.asm.txt").write_text("\n".join(f"{i:5} {n}" for i, n in enumerate(md.pop("asm_lines"))))
  md.update({"direct": True, "renderer": "AMDRenderer", "arch": "gfx1100"})
  (out / "direct.metrics.json").write_text(json.dumps(md, indent=2))

  # HIP: compile same sink through Device HIP renderer if available
  os.environ["DEV"] = "AMD:HIP"
  os.environ.pop("AMD_FLASH_DIRECT", None)
  DEV.value = "AMD:HIP"
  getenv.cache_clear()
  to_program_cache.clear()
  from tinygrad import Device
  from tinygrad.llm.kernels.amd import _amd_flash_attention as builder
  ren = Device["AMD"].renderer
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  hip_metrics = {"direct": False, "renderer": ren.__class__.__name__, "arch": getattr(Device["AMD"], "arch", None)}
  try:
    with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
      # HIP flash may not use _amd_flash_attention the same way — try anyway
      sink = builder(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=4, use_acc_work=True)
      prg_h = to_program(sink, ren)
    mh = elf_metrics(prg_h)
    (out / "hip.elf").write_bytes(mh.pop("binary"))
    (out / "hip.asm.txt").write_text("\n".join(f"{i:5} {n}" for i, n in enumerate(mh.pop("asm_lines"))))
    hip_metrics.update(mh)
  except Exception as e:
    # Fallback: run TinyJit once and pull AMD program from Device compiler cache / runtime
    hip_metrics["note"] = f"to_program HIP failed: {e!r}; trying runtime capture"
    try:
      import numpy as np
      from tinygrad import Tensor, TinyJit
      from tinygrad.llm.kernels.amd import flash_attention
      def patterned(shape, dtype=np.float32):
        c = int(np.prod(shape))
        return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)
      qq = Tensor(patterned((1, 32, 32, 128)), device="AMD").realize()
      cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device="AMD").realize()
      runner = TinyJit(lambda x: flash_attention(x, cache, 2048).realize())
      for _ in range(3):
        runner(qq); Device["AMD"].synchronize()
      # dig into jit_cache programs
      found = None
      for j in runner.captured.jit_cache:
        prg = j.prg
        lib = getattr(prg, "lib", None) or getattr(getattr(prg, "p", None), "lib", None)
        if isinstance(lib, bytes) and lib.startswith(b"\x7fELF") and len(lib) > 2000:
          if found is None or len(lib) > len(found): found = lib
      if found:
        # synthesize minimal metrics via decode
        class _P: pass
        p = _P(); p.src = (UOp(Ops.BINARY, arg=found),)
        # fake prg.src structure
        class _Prg:
          def __init__(self, b): self.src = (UOp(Ops.SINK), UOp(Ops.LINEAR), UOp(Ops.BINARY, arg=b))
        mh = elf_metrics(_Prg(found))
        (out / "hip.elf").write_bytes(mh.pop("binary"))
        (out / "hip.asm.txt").write_text("\n".join(f"{i:5} {n}" for i, n in enumerate(mh.pop("asm_lines"))))
        hip_metrics.update(mh)
      else:
        hip_metrics["note"] = "no HIP ELF found"
    except Exception as e2:
      hip_metrics["error"] = repr(e2)
  (out / "hip.metrics.json").write_text(json.dumps(hip_metrics, indent=2, default=str))
  return {"direct": md, "hip": hip_metrics}


if __name__ == "__main__":
  out = Path(sys.argv[1] if len(sys.argv) > 1 else "extra/rdna3_baseline_freeze")
  out.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("DEV", "AMD:AMD")
  os.environ.setdefault("AMD_FLASH_DIRECT", "1")
  DEV.value = os.environ["DEV"]
  print("=== ELFs ===", flush=True)
  capture_elfs(out)
  print("=== slot19 ===", flush=True)
  analyze_slot19(out)
  print("=== peak ===", flush=True)
  analyze_baseline_peak(out)
  print("done", flush=True)
