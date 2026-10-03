#!/usr/bin/env python3
"""Freeze DIRECT/HIP flash ELFs + pressure attribution for the post-fence baseline.

Writes under --out (default extra/rdna3_baseline_freeze/):
  direct.{elf,asm.txt,metrics.json}
  hip.{elf,asm.txt,metrics.json}          (HIP path; may be HIP-compiled blob)
  mov_attr.json                            (DIRECT MOV source patterns)
  slot19_spill_attr.json                   (promote-ACC experiment spill sites)
  peak_live.json                           (approx peak VGPR live at WMMA region)
  timings.json                             (fresh paired sync-each)
  SUMMARY.md

SQTT (optional): set VIZ=-2 / SQTT=1; copies profile pkl if present.
"""
from __future__ import annotations

import argparse, json, os, pickle, shutil, statistics, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

def _setup_env(direct: bool):
  from tinygrad.helpers import DEV, getenv
  target = "AMD:AMD" if direct else "AMD"
  os.environ["DEV"] = target
  if direct: os.environ["AMD_FLASH_DIRECT"] = "1"
  else: os.environ.pop("AMD_FLASH_DIRECT", None)
  DEV.value = target
  getenv.cache_clear()

def _flash_tensors():
  from tinygrad import Tensor, Device
  q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
  return q, cache

def _capture_program(direct: bool, out: Path, tag: str) -> dict:
  from tinygrad import Device, TinyJit
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.kernels.amd import flash_attention
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
  from tinygrad.runtime.support.elf import elf_loader

  _setup_env(direct)
  q, cache = _flash_tensors()
  runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
  for _ in range(3):
    out_t = runner(q)
    Device[Device.DEFAULT].synchronize()
  arr = out_t.numpy()

  # Find AMD ELF in captured linear (skip CPU hcq_submit)
  seen, elfs = set(), []
  for call in runner.captured.linear.src:
    for program in call.src[0].toposort():
      if program.op is not Ops.PROGRAM or program in seen: continue
      seen.add(program)
      binary = next((s.arg for s in program.src if s.op is Ops.BINARY), b"")
      source = next((s.arg for s in program.src if s.op is Ops.SOURCE), "")
      tgt = str(getattr(program.arg, "target", ""))
      if isinstance(binary, bytes) and binary.startswith(b"\x7fELF") and ("AMD" in tgt or b"amdgcn" in binary[:512] or len(binary) > 2000):
        elfs.append((tgt, source, binary, program.arg))

  metrics = {
    "direct": direct,
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": getattr(Device[Device.DEFAULT], "arch", None),
    "checksum": float(arr.astype(np.float64).sum()),
    "finite": bool(np.isfinite(arr).all()),
    "n_elfs": len(elfs),
  }
  if not elfs:
    # HIP may not expose ELF the same way — still record source if present
    metrics["note"] = "no AMD ELF found in TinyJit capture"
    (out / f"{tag}.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

  # Prefer largest ELF (flash kernel)
  tgt, source, binary, arg = max(elfs, key=lambda x: len(x[2]))
  (out / f"{tag}.elf").write_bytes(binary)
  if isinstance(source, str) and source:
    (out / f"{tag}.src.txt").write_text(source[:500_000])

  try:
    text = next(s.content for s in elf_loader(binary)[1] if s.name == ".text")
  except Exception as e:
    metrics["elf_decode_error"] = str(e)
    (out / f"{tag}.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3")
    insts.append(inst)
    off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, sgpr, _, priv = scan_elf_regs(insts, scratch_inst_size)
  metrics.update({
    "target": tgt,
    "elf_bytes": len(binary),
    "ninst": len(insts),
    "max_vgpr": vgpr,
    "max_sgpr": sgpr,
    "private_segment_size": priv,
    "v_mov_b32": mc.get("v_mov_b32_e32", 0),
    "scratch_store_b32": mc.get("scratch_store_b32", 0),
    "scratch_store_b128": mc.get("scratch_store_b128", 0),
    "scratch_load_b32": mc.get("scratch_load_b32", 0),
    "scratch_load_b128": mc.get("scratch_load_b128", 0),
    "v_wmma": mc.get("v_wmma_f32_16x16x16_f16", 0),
    "s_delay_alu": mc.get("s_delay_alu", 0),
    "v_dual": sum(v for k, v in mc.items() if "dual" in k),
    "ds_load_2addr_b64": mc.get("ds_load_2addr_b64", 0),
    "machine_top": mc.most_common(40),
  })
  # disasm dump (op names only — compact)
  (out / f"{tag}.asm.txt").write_text("\n".join(f"{i:5} {getattr(inst,'op_name','?')}" for i, inst in enumerate(insts)))
  (out / f"{tag}.metrics.json").write_text(json.dumps(metrics, indent=2))
  return metrics

def _compile_ir(acc_slot: int = 2, use_acc_work: bool = True):
  """Compile flash prefill IR with optional ACC slot override (diagnosis only)."""
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.helpers import Context, getenv
  from tinygrad.renderer import Target
  from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, _iop
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  import tinygrad.llm.kernels.amd as amd

  to_program_cache.clear(); getenv.cache_clear()
  # monkeypatch acc slot for slot-19 experiment without editing tree permanently
  orig = amd._amd_flash_attention
  def wrapped(*a, **k):
    # rebuild with patched slot by temporarily editing module constant path
    return orig(*a, **k)
  # For slot override: patch the line via wrapping builder internals is hard;
  # instead pass through a local copy of compile after string-level patch on source AST.
  # Simpler: use env-free edit of placeholder slot by substituting in the sink — skip;
  # call with use_acc_work and rely on external file patch when acc_slot!=2.
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    sink = _amd_flash_attention(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1,
                                k_midstore=4, use_acc_work=use_acc_work)
    prg = to_program(sink, ren)
  ops = [u for u in prg.src[1].src if u.op is Ops.INS]
  return ren, prg, ops, AMDOps, _iop, Ops

def mov_attribution(out: Path):
  from tinygrad.renderer.isa.rdna3 import AMDOps, _iop
  from tinygrad.uop.ops import Ops
  _setup_env(True)
  _, prg, ops, AMDOps, _iop, Ops = _compile_ir(use_acc_work=True)

  def kind(s):
    if s is None: return "none"
    if s.op is Ops.CONST: return f"const:{type(s.arg).__name__}"
    if s.op is Ops.INS:
      try: return "ins:" + _iop(s).name
      except Exception: return "ins"
    return s.op.name

  movs = [(i, u) for i, u in enumerate(ops) if _iop(u) is AMDOps.MOV]
  src_k = Counter(tuple(kind(s) for s in u.src[:2]) for _, u in movs)
  prev = Counter(_iop(ops[i - 1]).name if i else "START" for i, _ in movs)
  nxt = Counter(_iop(ops[i + 1]).name if i + 1 < len(ops) else "END" for i, _ in movs)
  # EXTRACT→MOV runs
  extract_mov_runs = []
  i = 0
  while i < len(ops):
    if _iop(ops[i]) is AMDOps.EXTRACT:
      j = i
      while j < len(ops) and _iop(ops[j]) in (AMDOps.EXTRACT, AMDOps.MOV):
        j += 1
      n_ext = sum(1 for k in range(i, j) if _iop(ops[k]) is AMDOps.EXTRACT)
      n_mov = sum(1 for k in range(i, j) if _iop(ops[k]) is AMDOps.MOV)
      if n_mov and n_ext: extract_mov_runs.append({"start": i, "end": j, "extract": n_ext, "mov": n_mov})
      i = j
    else:
      i += 1
  wmma_i = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
  mov_vs_wmma = Counter(sum(1 for w in wmma_i if w < i) for i, _ in movs)

  report = {
    "ir_mov": len(movs),
    "src_kinds": src_k.most_common(20),
    "prev_op": prev.most_common(12),
    "next_op": nxt.most_common(12),
    "extract_mov_runs": len(extract_mov_runs),
    "extract_mov_run_sizes": Counter((r["extract"], r["mov"]) for r in extract_mov_runs).most_common(12),
    "mov_after_n_wmma": mov_vs_wmma.most_common(20),
    "wmma_count": len(wmma_i),
    "wmma_idxs": wmma_i,
  }
  (out / "mov_attr.json").write_text(json.dumps(report, indent=2))
  return report

def slot19_spill_attribution(out: Path):
  """Patch-compile with ACC on slot 19, attribute SPILL sites + lifetimes."""
  import tempfile, textwrap, subprocess, sys
  from pathlib import Path as P
  # Write a small inline runner that patches amd.py in-memory by importing after exec rewrite
  # Safer: subprocess with sed on a temp copy of the repo file.
  amd_path = P(__file__).resolve().parents[1] / "tinygrad" / "llm" / "kernels" / "amd.py"
  src = amd_path.read_text()
  patched = src.replace(
    "  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
    "  acc, m_i, l_i = _reg((TM, TD), 19, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
    1).replace(
    'use_acc_work = bool(use_acc_small and (2 in _skip_reg) and getenv("AMD_FLASH_ACC_WORK", 1))',
    'use_acc_work = False  # diagnosis: no work-copy',
    1)
  if patched == src:
    raise RuntimeError("failed to patch amd.py for slot-19 diagnosis")

  tmp = out / "_amd_slot19.py"
  tmp.write_text(patched)
  # Load patched module under a unique name
  import importlib.util
  os.environ["DEV"] = "AMD:AMD"
  os.environ["AMD_FLASH_DIRECT"] = "1"
  from tinygrad.helpers import getenv
  getenv.cache_clear()

  spec = importlib.util.spec_from_file_location("amd_slot19_diag", tmp)
  mod = importlib.util.module_from_spec(spec)
  # Ensure package-relative imports work: execute with tinygrad.llm.kernels on path
  sys.modules["amd_slot19_diag"] = mod
  # The file imports from tinygrad.* — fine. But it may expect to be package member.
  spec.loader.exec_module(mod)

  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.helpers import Context
  from tinygrad.renderer import Target
  from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, _iop, greg
  from tinygrad.llm.kernels.amd import _flash_direct_compile_env

  to_program_cache.clear(); getenv.cache_clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  o = UOp.placeholder((32, 32, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, 32, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2)
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    sink = mod._amd_flash_attention(o, q, kv, valid_kv_len=2048, acc_small=True, k_unroll=1,
                                    k_midstore=4, use_acc_work=False)
    prg = to_program(sink, ren)
  ops = [u for u in prg.src[1].src if u.op is Ops.INS]
  wmma_i = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
  spill_i = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.SPILL]
  fill_i = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.FILL]

  def walk(u, depth=0):
    if u is None or depth > 8: return "?"
    if u.op is Ops.CONST: return f"CONST:{u.dtype}:{u.arg!r}"[:60]
    if u.op is Ops.INS:
      try: name = _iop(u).name
      except Exception: name = "INS"
      return f"{name}:{u.dtype}"
    return f"{u.op.name}:{u.dtype}"

  sites = []
  for i in spill_i:
    u = ops[i]
    s = u.src[0] if u.src else None
    # defining CAST chain
    chain = []
    x = s
    for _ in range(6):
      if x is None: break
      chain.append(walk(x))
      if x.op is Ops.CONST or not x.src: break
      x = x.src[0]
    n_wmma_before = sum(1 for w in wmma_i if w < i)
    # crude lifetime: next use of spilled value after spill
    next_use = None
    for j in range(i + 1, min(len(ops), i + 400)):
      if any(id(s) == id(t) or (t.op is Ops.INS and s in t.src) for t in [ops[j]] for _ in [0]):
        # weaker: FILL nearby
        pass
      if _iop(ops[j]) is AMDOps.FILL:
        next_use = j
        break
    sites.append({
      "idx": i,
      "wmma_before": n_wmma_before,
      "src_chain": chain,
      "next_fill_idx": next_use,
      "gap_to_fill": (None if next_use is None else next_use - i),
    })

  # Peak-live approximation: scan Linear-style — count distinct virt regs in open intervals
  # Using SPILL/FILL as pressure markers around WMMA window
  first_wmma = wmma_i[0] if wmma_i else 0
  last_wmma = wmma_i[-1] if wmma_i else 0
  spills_in_wmma = [s for s in sites if first_wmma <= s["idx"] <= last_wmma]
  spills_before = [s for s in sites if s["idx"] < first_wmma]
  spills_after = [s for s in sites if s["idx"] > last_wmma]

  cnt = Counter(_iop(u).name for u in ops)
  report = {
    "acc_slot": 19,
    "use_acc_work": False,
    "ir_counts": {k: cnt[k] for k in ("SSTORE", "SLOAD", "SPILL", "FILL", "MOV", "WMMA", "CAST")},
    "n_spill": len(spill_i),
    "n_fill": len(fill_i),
    "spill_vs_wmma": Counter(s["wmma_before"] for s in sites).most_common(30),
    "spills_before_first_wmma": len(spills_before),
    "spills_during_wmma_window": len(spills_in_wmma),
    "spills_after_last_wmma": len(spills_after),
    "src_chain_top": Counter(tuple(s["src_chain"]) for s in sites).most_common(15),
    "gap_to_fill_hist": Counter(s["gap_to_fill"] for s in sites).most_common(15),
    "wmma_idxs": wmma_i,
    "peak_hint": (
      "WMMA window" if len(spills_in_wmma) >= len(spills_after) else
      "post-WMMA epilogue" if len(spills_after) > len(spills_before) else
      "pre-WMMA setup"
    ),
    "sites_sample": sites[:20] + (sites[-10:] if len(sites) > 30 else []),
  }
  (out / "slot19_spill_attr.json").write_text(json.dumps(report, indent=2, default=str))
  tmp.unlink(missing_ok=True)
  return report

def peak_live_baseline(out: Path):
  """Approx peak live VGPRs from IR around WMMA for baseline (slot-2) kernel."""
  from tinygrad.renderer.isa.rdna3 import AMDOps, _iop, greg
  from tinygrad.uop.ops import Ops
  from tinygrad.renderer.isa import Register
  _setup_env(True)
  _, prg, ops, AMDOps, _iop, Ops = _compile_ir(use_acc_work=True)

  # Track virt regs: last def / last use
  defs: dict[int, int] = {}
  uses: dict[int, list[int]] = defaultdict(list)
  for i, u in enumerate(ops):
    if u.op is not Ops.INS: continue
    if isinstance(u.tag, tuple):
      for t in u.tag:
        if isinstance(t, Register):
          defs.setdefault(id(t), i)  # first def
          defs[id(t)] = i  # update last def idx store — use first
    for s in u.src:
      try:
        r = greg(s)
      except Exception:
        r = None
      if isinstance(r, Register):
        uses[id(r)].append(i)
    if isinstance(u.tag, tuple):
      for t in u.tag:
        if isinstance(t, Register) and id(t) not in defs:
          defs[id(t)] = i

  # Rebuild with first-def
  first_def: dict[int, int] = {}
  last_use: dict[int, int] = {}
  for i, u in enumerate(ops):
    if isinstance(getattr(u, "tag", None), tuple):
      for t in u.tag:
        if isinstance(t, Register):
          first_def.setdefault(id(t), i)
    for s in u.src:
      try: r = greg(s)
      except Exception: r = None
      if isinstance(r, Register):
        last_use[id(r)] = i
        first_def.setdefault(id(r), i)

  wmma_i = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
  # Live count at each WMMA index
  live_at = {}
  for wi in wmma_i:
    live = sum(1 for rid, fd in first_def.items() if fd <= wi and last_use.get(rid, fd) >= wi)
    live_at[wi] = live
  peak_idx = max(live_at, key=live_at.get) if live_at else None
  report = {
    "n_virt_regs_tracked": len(first_def),
    "live_at_each_wmma": live_at,
    "peak_live_at_wmma_idx": peak_idx,
    "peak_live_count": live_at.get(peak_idx) if peak_idx is not None else None,
    "wmma_idxs": wmma_i,
    "note": "Approximate SSA virt-reg overlap; phys VGPR peak is after LinearScan.",
  }
  (out / "peak_live.json").write_text(json.dumps(report, indent=2))
  return report

def timings(out: Path) -> dict:
  from tinygrad import Device, TinyJit
  from tinygrad.llm.kernels.amd import flash_attention

  def sync_each(direct: bool, case: str, n: int = 30) -> dict:
    _setup_env(direct)
    tokens = 1 if case == "decode" else 32
    q = __import__("tinygrad").Tensor(patterned((1, 32, tokens, 128)), device=Device.DEFAULT).realize()
    cache = __import__("tinygrad").Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
    runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
    for _ in range(5):
      runner(q); Device[Device.DEFAULT].synchronize()
    xs = []
    for _ in range(n):
      t0 = time.perf_counter_ns()
      runner(q); Device[Device.DEFAULT].synchronize()
      xs.append((time.perf_counter_ns() - t0) / 1e3)
    return {
      "direct": direct, "case": case,
      "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
      "median_us": statistics.median(xs), "best_us": min(xs), "mean_us": statistics.mean(xs), "n": n,
    }

  rows = [
    sync_each(True, "prefill"),
    sync_each(False, "prefill"),
    sync_each(True, "decode"),
    sync_each(False, "decode"),
  ]
  report = {"tip_note": "post-fence baseline freeze", "rows": rows}
  (out / "timings.json").write_text(json.dumps(report, indent=2))
  return report

def try_sqtt(out: Path) -> dict:
  """Capture one DIRECT flash launch with SQTT if VIZ/SQTT enabled by caller."""
  from tinygrad.helpers import temp, getenv
  info = {"attempted": True, "sqtt_env": getenv("SQTT", 0), "viz": getenv("VIZ", 0)}
  try:
    _setup_env(True)
    from tinygrad import Device, TinyJit
    from tinygrad.llm.kernels.amd import flash_attention
    q, cache = _flash_tensors()
    runner = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
    for _ in range(2):
      runner(q); Device[Device.DEFAULT].synchronize()
    runner(q); Device[Device.DEFAULT].synchronize()
    pkl = Path(temp("profile.pkl", append_user=True))
    info["profile_pkl"] = str(pkl)
    if pkl.exists():
      dest = out / "sqtt_profile.pkl"
      shutil.copy2(pkl, dest)
      with open(pkl, "rb") as f:
        events = pickle.load(f)
      kinds = Counter(type(e).__name__ for e in events)
      info["event_kinds"] = dict(kinds)
      info["copied_to"] = str(dest)
      # keep only profile-related
      slim = [e for e in events if type(e).__name__ in {"ProfilePMCEvent", "ProfileSQTTEvent", "ProfileProgramEvent"}]
      with open(out / "sqtt_profile_slim.pkl", "wb") as f:
        pickle.dump(slim, f)
      info["slim_events"] = len(slim)
    else:
      info["error"] = "profile.pkl not found — run with VIZ=-2 or SQTT=1"
  except Exception as e:
    info["error"] = repr(e)
  (out / "sqtt_info.json").write_text(json.dumps(info, indent=2))
  return info

def write_summary(out: Path, parts: dict):
  d = parts.get("direct_metrics", {})
  h = parts.get("hip_metrics", {})
  t = parts.get("timings", {})
  mov = parts.get("mov", {})
  s19 = parts.get("slot19", {})
  peak = parts.get("peak", {})
  lines = [
    "# RDNA3 flash baseline freeze (post-HCQ2 fence)",
    "",
    "## Timings (fresh sync-each)",
  ]
  for row in t.get("rows", []):
    lines.append(f"- {'DIRECT' if row['direct'] else 'HIP'} {row['case']}: median **{row['median_us']:.1f} µs** "
                 f"(best {row['best_us']:.1f}, {row['renderer']})")
  lines += [
    "",
    "## DIRECT ELF",
    f"- vgpr {d.get('max_vgpr')} / sgpr {d.get('max_sgpr')} / priv {d.get('private_segment_size')}",
    f"- ninst {d.get('ninst')} / v_mov {d.get('v_mov_b32')} / wmma {d.get('v_wmma')} / delay_alu {d.get('s_delay_alu')} / dual {d.get('v_dual')}",
    f"- scratch st_b32 {d.get('scratch_store_b32')} st_b128 {d.get('scratch_store_b128')} "
    f"ld_b32 {d.get('scratch_load_b32')} ld_b128 {d.get('scratch_load_b128')}",
    "",
    "## HIP ELF",
    f"- vgpr {h.get('max_vgpr')} / priv {h.get('private_segment_size')} / ninst {h.get('ninst')} / "
    f"v_mov {h.get('v_mov_b32')} / delay_alu {h.get('s_delay_alu')} / dual {h.get('v_dual')}",
    f"- note: {h.get('note', h.get('elf_decode_error', 'ok'))}",
    "",
    "## MOV attribution (DIRECT IR)",
    f"- IR MOV {mov.get('ir_mov')}; top src {mov.get('src_kinds', [])[:6]}",
    f"- EXTRACT→MOV runs {mov.get('extract_mov_runs')}; sizes {mov.get('extract_mov_run_sizes', [])[:5]}",
    f"- MOV vs WMMA-before count {mov.get('mov_after_n_wmma', [])[:8]}",
    "",
    "## Slot-19 promote spill attribution",
    f"- IR {s19.get('ir_counts')}",
    f"- spills before/during/after WMMA: {s19.get('spills_before_first_wmma')}/"
    f"{s19.get('spills_during_wmma_window')}/{s19.get('spills_after_last_wmma')}",
    f"- peak_hint: **{s19.get('peak_hint')}**",
    f"- src_chain_top: {s19.get('src_chain_top', [])[:5]}",
    "",
    "## Baseline peak-live (approx)",
    f"- peak live virt regs at WMMA idx {peak.get('peak_live_at_wmma_idx')}: {peak.get('peak_live_count')}",
    "",
    "## Candidate (diagnosis → next change)",
    parts.get("candidate", "_pending_"),
  ]
  (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", default="extra/rdna3_baseline_freeze")
  p.add_argument("--skip-sqtt", action="store_true")
  p.add_argument("--skip-timings", action="store_true")
  args = p.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  parts = {}
  print("=== DIRECT ELF ===", flush=True)
  parts["direct_metrics"] = _capture_program(True, out, "direct")
  print("=== HIP ELF ===", flush=True)
  parts["hip_metrics"] = _capture_program(False, out, "hip")
  print("=== MOV attr ===", flush=True)
  parts["mov"] = mov_attribution(out)
  print("=== slot19 spill attr ===", flush=True)
  parts["slot19"] = slot19_spill_attribution(out)
  print("=== peak live ===", flush=True)
  parts["peak"] = peak_live_baseline(out)
  if not args.skip_timings:
    print("=== timings ===", flush=True)
    parts["timings"] = timings(out)
  if not args.skip_sqtt:
    print("=== sqtt ===", flush=True)
    parts["sqtt"] = try_sqtt(out)

  # Candidate from evidence (filled after we inspect outputs — placeholder refined in driver)
  s19 = parts["slot19"]
  mov = parts["mov"]
  parts["candidate"] = (
    f"Shorten/delay **int address-CAST lifetimes that cluster {s19.get('peak_hint')}** "
    f"(slot-19: {s19.get('spills_during_wmma_window')} spills in WMMA window, "
    f"{s19.get('spills_after_last_wmma')} after). Keep slot-2 scratch. "
    f"Do not chase SLOAD count (already 8) or global remat/VOPD. "
    f"MOV tax is dominated by {mov.get('src_kinds', [['?']])[0]} — treat as secondary unless "
    f"liveness work also collapses EXTRACT→MOV runs."
  )
  write_summary(out, parts)
  (out / "parts.json").write_text(json.dumps({k: v for k, v in parts.items() if k != "candidate"}, indent=2, default=str))
  print("WROTE", out / "SUMMARY.md", flush=True)

if __name__ == "__main__":
  main()
