#!/usr/bin/env python3
"""Flash DIRECT state diagnostics: fixed buffers, scratch history, phase dumps.

Stops scheduling experiments. Compares:
  1. recreate  — new q/kv/out each launch (prior harness)
  2. fixed     — same runtime + q/kv/out buffers + launch args, 100 launches,
                 no intervening helper kernels
  3. scratch   — untouched / zero / nonzero fill of device scratch immediately
                 before each Flash launch (isolated sync process intent)

Also saves the first failing replay and its predecessor (not only the first eight).

Phase dumps: instrumented ELF is replayed until *output* fails; fail+pred dumps saved.
Do not transfer an uninstrumented fail index onto a newly instrumented binary.
If instrumentation suppresses the out failure, treat as inconclusive (not a pass).

Example:
  PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --phase-only --phases qk --replays 100
  PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --phase-only --phases pv_wmma,pv --replays 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.engine.realize import get_runtime, runtime_cache
from tinygrad.helpers import Context, getenv, DEV
from tinygrad.llm.kernels.amd import (
  _amd_flash_attention, BLOCK_M, BLOCK_N, WAVES_M, WAVES_N,
  LANES_PER_WAVE_M, LANES_PER_WAVE_N, WMMA_M, flash_tile_dims,
)
import tinygrad.llm.kernels.amd as amd
from tinygrad.uop.ops import Ops, UOp


def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def sha256(b: bytes) -> str:
  return hashlib.sha256(b).hexdigest()


def clear_caches():
  getenv.cache_clear()
  to_program_cache.clear()
  amd._amd_flash_attention.cache_clear()
  runtime_cache.clear()


def apply_env(*, skip="2", work=1, batch=1, cluster=1, direct=True):
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_SPILL_ON_EVICT"] = "0"
  os.environ["AMD_PACK_SLOAD_B128"] = "0"
  os.environ.pop("AMD_FMA_MIX", None)
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  os.environ["AMD_FLASH_ACC_WORK"] = str(work)
  os.environ["AMD_BATCH_SLOAD_USE"] = str(batch)
  os.environ["AMD_CLUSTER_SLOAD"] = str(cluster)
  for k in ("AMD_IN_ORDER_EMIT", "AMD_INSTR_WAIT", "AMD_CONSERVATIVE_WAIT"):
    os.environ.pop(k, None)
  if direct:
    os.environ["AMD_FLASH_DIRECT"] = "1"
  else:
    os.environ.pop("AMD_FLASH_DIRECT", None)
  clear_caches()


def set_dev(target: str):
  os.environ["DEV"] = target
  DEV.value = target
  clear_caches()
  return target


def use_acc_work(skip: str, work: int) -> bool:
  return bool(work and "2" in {s.strip() for s in skip.split(",")})


def phase_shapes(BH: int, M: int, S: int, D: int = 128,
                 phases: tuple[str, ...] = ("qk",)) -> dict[str, tuple]:
  """Logical dump shapes — must match `_amd_flash_attention` phase_dumps docstring."""
  nm = M // BLOCK_M
  ntiles = (S + BLOCK_N - 1) // BLOCK_N
  TM, TN, TD = flash_tile_dims(D)  # noqa: F841 — document dims; shapes use BLOCK_* / D
  assert (TM, TN, TD) == (BLOCK_M // (WAVES_M * LANES_PER_WAVE_M),
                          BLOCK_N // LANES_PER_WAVE_N,
                          D // (WAVES_N * LANES_PER_WAVE_N))
  all_shapes = {
    "qk": (BH, nm, ntiles, BLOCK_M, BLOCK_N),
    "soft_m": (BH, nm, ntiles, BLOCK_M),
    "soft_l": (BH, nm, ntiles, BLOCK_M),
    "pv_wmma": (BH, nm, ntiles, BLOCK_M, D),
    "pv": (BH, nm, ntiles, BLOCK_M, D),
    "acc": (BH, nm, BLOCK_M, D),
  }
  return {k: all_shapes[k] for k in phases}


def compile_program(S: int, T: int, acc_work: bool, *, phases: tuple[str, ...] = ()):
  BH = 32
  out = UOp.placeholder((BH, T, 128), dtypes.float32, 0)
  q = UOp.placeholder((BH, T, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, S, 128), dtypes.float16, 2)
  dumps: list[tuple[str, UOp]] = []
  dump_bufs: dict[str, UOp] = {}
  if phases:
    slot = 3
    for name, shape in phase_shapes(BH, T, S, phases=phases).items():
      buf = UOp.placeholder(shape, dtypes.float32, slot)
      dump_bufs[name] = buf
      dumps.append((name, buf))
      slot += 1
  sink = _amd_flash_attention(out, q, kv, valid_kv_len=S, q_start=None,
                              acc_small=True, k_unroll=0, use_acc_work=acc_work,
                              phase_dumps=tuple(dumps))
  with Context(BEAM=0):
    prev = os.environ.get("AMD_FLASH_ACC_SMALL")
    os.environ["AMD_FLASH_ACC_SMALL"] = "1"
    clear_caches()
    try:
      prg = to_program(sink, Device["AMD"].renderer)
    finally:
      if prev is None: os.environ.pop("AMD_FLASH_ACC_SMALL", None)
      else: os.environ["AMD_FLASH_ACC_SMALL"] = prev
      clear_caches()
  return prg, dump_bufs


def program_elf(prg) -> bytes | None:
  try:
    binary = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  except StopIteration:
    return None
  return binary if isinstance(binary, bytes) and binary.startswith(b"\x7fELF") else None


def arrays_close(a: np.ndarray, b: np.ndarray) -> bool:
  """Bit-exact finites; matching ±inf equal; matching NaN masks equal with equal non-NaNs."""
  a64, b64 = a.astype(np.float64), b.astype(np.float64)
  a_nan, b_nan = np.isnan(a64), np.isnan(b64)
  if not np.array_equal(a_nan, b_nan):
    return False
  nn = ~a_nan
  aa, bb = a64[nn], b64[nn]
  both_pos = np.isposinf(aa) & np.isposinf(bb)
  both_neg = np.isneginf(aa) & np.isneginf(bb)
  both_fin = np.isfinite(aa) & np.isfinite(bb)
  return bool(np.all(both_pos | both_neg | (both_fin & (aa == bb))))


def first_divergent(a: np.ndarray, b: np.ndarray) -> dict | None:
  """First mismatch. Matching ±inf equal; NaN mask/value issues → kind=nan (not via a-b)."""
  a64, b64 = a.astype(np.float64), b.astype(np.float64)
  a_nan, b_nan = np.isnan(a64), np.isnan(b64)
  both_pos = np.isposinf(a64) & np.isposinf(b64)
  both_neg = np.isneginf(a64) & np.isneginf(b64)
  both_fin = np.isfinite(a64) & np.isfinite(b64)
  both_nan = a_nan & b_nan
  equal = both_pos | both_neg | both_nan | (both_fin & (a64 == b64))
  if np.all(equal):
    return None
  idx = int(np.argmax(~equal))
  coords = [int(c) for c in np.unravel_index(idx, a.shape)]
  av, bv = float(a.reshape(-1)[idx]), float(b.reshape(-1)[idx])
  kind = "nan" if (a_nan.reshape(-1)[idx] or b_nan.reshape(-1)[idx]) else "value"
  md = abs(av - bv) if np.isfinite(av) and np.isfinite(bv) else float("nan")
  return {
    "flat_index": idx, "coords": coords, "maxdiff": md, "kind": kind,
    "a": av, "b": bv,
    "head": coords[0] if coords else None,
    "token": coords[1] if len(coords) > 1 else None,
    "query_tile": (coords[1] // BLOCK_M) if len(coords) > 1 else None,
  }


def fill_scratch(mode: str) -> dict:
  """Fill AMD device scratch. Diagnostic only — not a shipping fix."""
  dev = Device["AMD"]
  if not hasattr(dev, "scratch") or dev.scratch is None:
    return {"mode": mode, "skipped": "no scratch"}
  sc = dev.scratch
  size = int(sc.size)
  if mode == "untouched":
    return {"mode": mode, "size": size}
  if mode == "zero":
    payload = bytes(size)
  elif mode == "nonzero":
    payload = bytes([0xA5]) * size
  else:
    raise ValueError(mode)
  # Chunked copyin — full scratch is ~25MB
  mv = memoryview(bytearray(payload))
  chunk = 4 << 20
  for off in range(0, size, chunk):
    n = min(chunk, size - off)
    dest = sc.offset(off, n) if hasattr(sc, "offset") else sc
    src = mv[off:off + n]
    if hasattr(sc, "offset"):
      dev.allocator._copyin(dest, src)
    else:
      # whole-buffer only
      if off == 0: dev.allocator._copyin(sc, mv)
      break
  Device["AMD"].synchronize()
  return {"mode": mode, "size": size, "filled_bytes": size if hasattr(sc, "offset") else min(size, chunk)}


def make_fixed_state(prg, q_np, kv_np, T: int, S: int | None = None, *, sentinel=float("nan"),
                     dump_names: list[str] | None = None):
  device = "AMD"
  S = S if S is not None else int(kv_np.shape[3])
  q = Tensor(q_np.copy(), device=device, dtype=dtypes.float16).realize()
  kv = Tensor(kv_np.copy(), device=device, dtype=dtypes.float16).realize()
  out = Tensor.full((32, T, 128), sentinel, dtype=dtypes.float32, device=device).realize()
  dumps = {}
  if dump_names:
    shapes = phase_shapes(32, T, S, phases=tuple(dump_names))
    for name in dump_names:
      dumps[name] = Tensor.zeros(shapes[name], dtype=dtypes.float32, device=device).realize()
  rt = get_runtime(device, prg, cache=True)
  gs, ls = prg.arg.launch_dims({})
  vals = prg.arg.vals({})
  host_bufs = [out, q, kv] + [dumps[n] for n in (dump_names or [])]
  raw = [t.uop.buffer.ensure_allocated().get_buf(device) for t in host_bufs]
  ordered = [raw[i] for i in prg.arg.globals]
  return {
    "rt": rt, "ordered": ordered, "gs": gs, "ls": ls, "vals": vals,
    "out": out, "q": q, "kv": kv, "dumps": dumps, "dump_names": dump_names or [],
  }


def launch_fixed(state) -> np.ndarray:
  state["rt"](*state["ordered"], global_size=state["gs"], local_size=state["ls"],
              vals=state["vals"], wait=True)
  Device["AMD"].synchronize()
  return np.array(state["out"].numpy(), dtype=np.float32, copy=True)


def refill_out_sentinel(state, sentinel: float = float("nan")):
  state["out"].assign(Tensor.full(state["out"].shape, sentinel, dtype=dtypes.float32, device="AMD")).realize()


def launch_recreate(prg, q_np, kv_np, T: int) -> np.ndarray:
  device = "AMD"
  q = Tensor(q_np.copy(), device=device, dtype=dtypes.float16).realize()
  kv = Tensor(kv_np.copy(), device=device, dtype=dtypes.float16).realize()
  out = Tensor.full((32, T, 128), float("nan"), dtype=dtypes.float32, device=device).realize()
  rt = get_runtime(device, prg, cache=True)
  gs, ls = prg.arg.launch_dims({})
  vals = prg.arg.vals({})
  bufs = [out.uop.buffer.ensure_allocated().get_buf(device),
          q.uop.buffer.ensure_allocated().get_buf(device),
          kv.uop.buffer.ensure_allocated().get_buf(device)]
  ordered = [bufs[i] for i in prg.arg.globals]
  rt(*ordered, global_size=gs, local_size=ls, vals=vals, wait=True)
  Device["AMD"].synchronize()
  return np.array(out.numpy(), dtype=np.float32, copy=True)


def save_fail_pair(art: Path, outs: list[np.ndarray], ref0: np.ndarray | None = None) -> dict:
  """Persist first failing replay and its predecessor."""
  meta = {"first_fail_i": None, "pred_i": None}
  for i in range(1, len(outs)):
    if not arrays_close(outs[0], outs[i]):
      meta["first_fail_i"] = i
      meta["pred_i"] = i - 1
      np.save(art / f"fail_{i}.npy", outs[i])
      np.save(art / f"pred_{i-1}.npy", outs[i - 1])
      np.save(art / "replay_0.npy", outs[0])
      meta["div_vs_0"] = first_divergent(outs[0], outs[i])
      if i >= 1:
        meta["div_vs_pred"] = first_divergent(outs[i - 1], outs[i])
      if ref0 is not None:
        meta["fail_vs_ref"] = first_divergent(outs[i], ref0)
        meta["pred_vs_ref"] = first_divergent(outs[i - 1], ref0)
        meta["replay0_vs_ref"] = first_divergent(outs[0], ref0)
      print(f"  first_fail i={i} div_vs_0={meta['div_vs_0']}")
      if meta.get("div_vs_pred"):
        print(f"  fail_vs_pred={meta['div_vs_pred']}")
      break
  else:
    np.save(art / "replay_0.npy", outs[0])
    for i, o in enumerate(outs[:8]):
      np.save(art / f"replay_{i}.npy", o)
  (art / "fail_meta.json").write_text(json.dumps(meta, indent=2, default=str))
  return meta


def summarize_outs(outs: list[np.ndarray], ref: np.ndarray | None, ref_tol: float) -> dict:
  finite = all(np.isfinite(o).all() for o in outs)  # outputs should be finite; -inf only in QK dumps
  exact = all(arrays_close(outs[0], o) for o in outs[1:])
  maxdiff = 0.0
  for o in outs[1:]:
    both = np.isfinite(outs[0]) & np.isfinite(o)
    if np.any(both):
      maxdiff = max(maxdiff, float(np.max(np.abs(outs[0][both].astype(np.float64) - o[both].astype(np.float64)))))
  div = None if exact else first_divergent(outs[0], next(o for o in outs[1:] if not arrays_close(outs[0], o)))
  ref_ok, ref_max = True, 0.0
  if ref is not None:
    d = first_divergent(outs[0], ref)
    ref_max = 0.0 if d is None else float(d.get("maxdiff") or 0.0)
    if d is None:
      ref_ok = True
    elif np.isfinite(ref_max):
      ref_ok = bool(ref_max <= max(ref_tol, 1e-5))
    else:
      ref_ok = False
  verdict = (
    "launch_nondeterminism" if finite and not exact else
    "stable_wrong" if exact and finite and not ref_ok else
    "nonfinite" if not finite else
    "ok"
  )
  return {
    "ok": bool(finite and exact and ref_ok), "verdict": verdict,
    "exact": exact, "maxdiff": maxdiff, "finite": finite,
    "first_divergent": div, "ref_ok": ref_ok, "ref_maxdiff": ref_max,
  }


def run_mode(mode: str, prg, q_np, kv_np, T: int, replays: int, art: Path,
             ref: np.ndarray | None, ref_tol: float, scratch_mode: str | None = None) -> dict:
  cfg_art = art / mode
  if scratch_mode:
    cfg_art = art / f"{mode}_{scratch_mode}"
  if cfg_art.exists(): shutil.rmtree(cfg_art)
  cfg_art.mkdir(parents=True)
  print(f"\n======== mode={mode} scratch={scratch_mode} replays={replays} ========")

  outs: list[np.ndarray] = []
  scratch_info = None
  if mode == "recreate":
    for i in range(replays):
      if scratch_mode:
        scratch_info = fill_scratch(scratch_mode)
      outs.append(launch_recreate(prg, q_np, kv_np, T))
      if i < 4 or i + 1 == replays or (i + 1) % 25 == 0:
        print(f"  replay[{i}] mean={float(outs[-1].mean()):.8g}")
  elif mode == "fixed":
    # Same runtime + q/kv/out buffers + launch args. No intervening helper kernels:
    # do NOT Tensor.assign the out buffer (that would enqueue a fill kernel).
    state = make_fixed_state(prg, q_np, kv_np, T)
    _ = state["rt"]  # ensure scratch sized for this program
    for i in range(replays):
      if scratch_mode:
        scratch_info = fill_scratch(scratch_mode)
      outs.append(launch_fixed(state))
      if i < 4 or i + 1 == replays or (i + 1) % 25 == 0:
        print(f"  replay[{i}] mean={float(outs[-1].mean()):.8g}")
  else:
    raise ValueError(mode)

  row = summarize_outs(outs, ref, ref_tol)
  fail_meta = save_fail_pair(cfg_art, outs, ref)
  row.update({"mode": mode, "scratch": scratch_mode, "scratch_info": scratch_info,
              "replays": replays, "fail_meta": fail_meta})
  print(f"  verdict={row['verdict']} exact={row['exact']} maxdiff={row['maxdiff']:.6g} "
        f"ref_ok={row['ref_ok']} ref_maxdiff={row['ref_maxdiff']:.6g}")
  (cfg_art / "summary.json").write_text(json.dumps(row, indent=2, default=str))
  return row


def map_pv_coord(qrow: int, dcol: int, D: int = 128) -> dict:
  """Map logical (qrow,dcol) → wave/lane/fragment indices for PV dumps."""
  TM, TN, TD = flash_tile_dims(D)  # noqa: F841
  wave_m = qrow // WMMA_M
  rem_m = qrow % WMMA_M
  lane_m = rem_m % LANES_PER_WAVE_M
  ri = rem_m // LANES_PER_WAVE_M
  wave_n = dcol // (TD * LANES_PER_WAVE_N)
  rem_n = dcol % (TD * LANES_PER_WAVE_N)
  rj = rem_n // LANES_PER_WAVE_N
  lane_n = rem_n % LANES_PER_WAVE_N
  lane = lane_m * LANES_PER_WAVE_N + lane_n
  return {
    "qrow": qrow, "dcol": dcol,
    "wave_m": wave_m, "wave_n": wave_n, "lane": lane,
    "lane_m": lane_m, "lane_n": lane_n, "ri": ri, "rj": rj,
    "td_i": rj,  # with ACC_SMALL one WMMA pack per TD column
  }


def trace_pv_slots(prg, *, focus: dict | None = None) -> dict:
  """Trace slot-10 (pv_acc/WMMA C) and slot-17 (pv_soft) through lin IR.

  Reports WMMA defs, ACC→VGPR MOV copy (REG_STORE lowered), and SPILL/FILL of
  those physical VGPRs with scratch byte offsets. Compile-time only — same for
  every launch of this ELF.
  """
  from tinygrad.renderer.isa import greg
  from tinygrad.renderer.isa.rdna3 import AMDOps, _iop, _unwrap_const, _reg_slots

  lin = prg.src[1]
  TM, TN, TD = flash_tile_dims(128)
  events: list[dict] = []
  wmma_accs: list[str] = []
  soft_dsts: list[str] = []

  wmma_ops = [(i, u) for i, u in enumerate(lin.src) if u.op is Ops.INS and _iop(u) is AMDOps.WMMA]
  # ACC_SMALL: QK uses TN packs, PV uses TD packs afterward.
  pv_wmmas = wmma_ops[-TD:] if len(wmma_ops) >= TN + TD else wmma_ops
  qk_wmmas = wmma_ops[: len(wmma_ops) - len(pv_wmmas)]

  for i, u in pv_wmmas:
    acc = str(greg(u))
    wmma_accs.append(acc)
    events.append({
      "i": i, "op": "WMMA", "slot_hint": 10,
      "dst": acc, "srcs": [str(greg(s)) for s in u.src],
      "note": "pv_acc / parked ACC C",
    })

  # Each PV ACC pack occupies name v{base}..v{base+7} after EXTRACT.
  acc_bases = []
  for a in wmma_accs:
    try: acc_bases.append(int(a[1:]))
    except ValueError: pass
  acc_lane_names = {f"v{b + k}" for b in acc_bases for k in range(8)}
  soft_src_of: dict[str, str] = {}  # soft phys → ACC lane name
  if pv_wmmas:
    start = pv_wmmas[-1][0] + 1
    for j in range(start, len(lin.src)):
      u = lin.src[j]
      if u.op is not Ops.INS: continue
      op = _iop(u)
      if op is AMDOps.WMMA: break
      if op.name != "MOV" or not u.src: continue
      sreg, dreg = greg(u.src[0]), greg(u)
      if sreg is None or dreg is None: continue
      src_n, dst_n = str(sreg), str(dreg)
      if src_n == dst_n: continue
      if src_n in acc_lane_names and dst_n not in acc_lane_names:
        if dst_n not in soft_dsts:
          soft_dsts.append(dst_n)
          soft_src_of[dst_n] = src_n
          events.append({
            "i": j, "op": "MOV", "slot_hint": 17,
            "dst": dst_n, "srcs": [src_n],
            "note": "ACC→pv_soft copy (REG_STORE lowered)",
            "dst_phys_index": dreg.index, "src_phys_index": sreg.index,
          })
        if len(soft_dsts) >= TM * TD:
          break

  soft_set = set(soft_dsts)
  watch = set(wmma_accs) | soft_set
  for i, u in enumerate(lin.src):
    if u.op is not Ops.INS: continue
    op = _iop(u)
    if op is AMDOps.SPILL:
      src_reg = greg(u.src[1]) if len(u.src) > 1 else None
      if src_reg is None or str(src_reg) not in watch: continue
      disp = _unwrap_const(u.src[0])
      events.append({
        "i": i, "op": "SPILL",
        "slot_hint": 17 if str(src_reg) in soft_set else 10,
        "phys": str(src_reg),
        "scratch_off": int(disp.val) if disp is not None else None,
        "slots": int(_reg_slots(u.src[1])) if len(u.src) > 1 else None,
      })
    elif op is AMDOps.FILL:
      dst = greg(u)
      if dst is None or str(dst) not in watch: continue
      disp = _unwrap_const(u.src[0])
      events.append({
        "i": i, "op": "FILL",
        "slot_hint": 17 if str(dst) in soft_set else 10,
        "phys": str(dst),
        "scratch_off": int(disp.val) if disp is not None else None,
        "slots": int(_reg_slots(u)),
      })
    elif op is AMDOps.REG_STORE:
      dst = greg(u.src[0]) if u.src else None
      events.append({
        "i": i, "op": "REG_STORE", "slot_hint": 17,
        "phys": str(dst) if dst is not None else None,
        "srcs": [str(greg(s)) for s in u.src],
      })

  focus_map = None
  if focus and focus.get("qrow") is not None and focus.get("dcol") is not None:
    focus_map = map_pv_coord(int(focus["qrow"]), int(focus["dcol"]))
    td_i = focus_map["td_i"]
    if td_i < len(wmma_accs):
      focus_map["wmma_acc_phys"] = wmma_accs[td_i]
      if td_i < len(acc_bases):
        acc_lane = f"v{acc_bases[td_i] + focus_map['ri']}"
        focus_map["wmma_acc_lane"] = acc_lane
        soft_phys = next((d for d, s in soft_src_of.items() if s == acc_lane), None)
        focus_map["soft_phys"] = soft_phys
        if soft_phys:
          focus_map["soft_spill_fill"] = [
            e for e in events if e.get("op") in ("SPILL", "FILL") and e.get("phys") == soft_phys
          ]
    if "soft_spill_fill" not in (focus_map or {}):
      focus_map["soft_phys_all"] = soft_dsts
      focus_map["soft_spill_fill"] = [e for e in events if e.get("op") in ("SPILL", "FILL") and e.get("phys") in soft_set]

  return {
    "qk_wmma_count": len(qk_wmmas),
    "pv_wmma_count": len(pv_wmmas),
    "pv_wmma_acc_phys": wmma_accs,
    "pv_soft_phys": soft_dsts,
    "focus": focus_map,
    "events": events,
    "spill_fill_watched": [e for e in events if e.get("op") in ("SPILL", "FILL")],
  }


def read_dumps(state) -> dict[str, np.ndarray]:
  return {name: np.array(t.numpy(), dtype=np.float32, copy=True) for name, t in state["dumps"].items()}


def zero_dumps_host(state):
  """Zero dump buffers via allocator copyin — no compute helper kernel."""
  for t in state["dumps"].values():
    buf = t.uop.buffer.ensure_allocated()
    raw = buf.get_buf("AMD")
    nbytes = int(np.prod(t.shape)) * 4
    Device["AMD"].allocator._copyin(raw, memoryview(bytearray(nbytes)))
  Device["AMD"].synchronize()


def run_phase_fail_capture(S: int, T: int, acc_work: bool, art: Path, q_np, kv_np,
                           *, phases: tuple[str, ...] = ("qk",), replays: int = 100,
                           focus_head: int | None = None, focus_bm: int | None = None) -> dict:
  """Replay *instrumented* DIRECT ELF until out diverges; save fail+pred phase dumps.

  Also runs one HIP instrumented launch as a phase reference (not a fail index transfer).
  If instrumentation suppresses out failure within `replays`, report inconclusive.
  """
  print(f"\n======== phase fail-capture phases={phases} replays={replays} ========")
  phase_art = art / "phase"
  if phase_art.exists(): shutil.rmtree(phase_art)
  phase_art.mkdir(parents=True)
  TM, TN, TD = flash_tile_dims(128)
  print(f"  flash_tile_dims TM,TN,TD={TM},{TN},{TD} (expect 8,2,4)")

  # --- HIP instrumented reference (single launch; HIP out is stable) ---
  set_dev("AMD:HIP")
  apply_env(direct=False)
  set_dev("AMD:HIP")
  hip_prg, _ = compile_program(S, T, acc_work, phases=phases)
  assert Device["AMD"].renderer.__class__.__name__ == "HIPRenderer"
  hip_state = make_fixed_state(hip_prg, q_np, kv_np, T, S, dump_names=list(phases))
  zero_dumps_host(hip_state)
  hip_out = launch_fixed(hip_state)
  hip_dumps = read_dumps(hip_state)
  np.save(phase_art / "hip_out.npy", hip_out)
  for k, v in hip_dumps.items():
    np.save(phase_art / f"hip_{k}.npy", v)
  print(f"  HIP instrumented out_mean={float(hip_out.mean()):.8g}")

  # --- DIRECT instrumented: same ELF until output fails ---
  set_dev("AMD:AMD")
  apply_env(direct=True)
  set_dev("AMD:AMD")
  dir_prg, _ = compile_program(S, T, acc_work, phases=phases)
  assert Device["AMD"].renderer.__class__.__name__ == "AMDRenderer"
  elf = program_elf(dir_prg)
  if elf:
    (phase_art / "direct_instrumented.elf").write_bytes(elf)
    print(f"  DIRECT instrumented elf={sha256(elf)[:16]}…")

  state = make_fixed_state(dir_prg, q_np, kv_np, T, S, dump_names=list(phases))
  outs: list[np.ndarray] = []
  dump_hist: list[dict[str, np.ndarray]] = []
  fail_i = None
  for i in range(replays):
    zero_dumps_host(state)
    o = launch_fixed(state)
    d = read_dumps(state)
    outs.append(o)
    dump_hist.append(d)
    if i < 4 or (i + 1) % 25 == 0 or i + 1 == replays:
      print(f"  direct_instr replay[{i}] mean={float(o.mean()):.8g}")
    if i > 0 and not arrays_close(outs[0], o):
      fail_i = i
      print(f"  instrumented output FAILED at replay {i} (still reproduces)")
      break
  else:
    print("  INCONCLUSIVE: instrumentation suppressed output failure within replay budget")
    row = {
      "ok": False, "verdict": "instrumentation_suppressed_or_unlucky",
      "replays": replays, "phases": list(phases),
      "out_exact": True, "note": "no out divergence observed on instrumented ELF",
    }
    (phase_art / "summary.json").write_text(json.dumps(row, indent=2, default=str))
    return row

  pred_i = fail_i - 1
  np.save(phase_art / "replay_0_out.npy", outs[0])
  np.save(phase_art / f"pred_{pred_i}_out.npy", outs[pred_i])
  np.save(phase_art / f"fail_{fail_i}_out.npy", outs[fail_i])
  for k in phases:
    np.save(phase_art / f"pred_{pred_i}_{k}.npy", dump_hist[pred_i][k])
    np.save(phase_art / f"fail_{fail_i}_{k}.npy", dump_hist[fail_i][k])
    np.save(phase_art / f"replay0_{k}.npy", dump_hist[0][k])

  out_div = first_divergent(outs[pred_i], outs[fail_i])
  print(f"  out fail_vs_pred={out_div}")

  # Focus tile: from out divergence, or CLI override
  if focus_head is None or focus_bm is None:
    focus_head = int(out_div["head"]) if out_div and out_div.get("head") is not None else 0
    focus_bm = int(out_div["query_tile"]) if out_div and out_div.get("query_tile") is not None else 0
  print(f"  focus head={focus_head} query_tile={focus_bm}; inspecting ALL n_tiles")

  cmp: dict = {
    "fail_i": fail_i, "pred_i": pred_i, "out_div": out_div,
    "focus": {"head": focus_head, "query_tile": focus_bm},
    "phases": {}, "earliest": None,
  }
  earliest = None
  for name in phases:
    pred, fail = dump_hist[pred_i][name], dump_hist[fail_i][name]
    hip = hip_dumps[name]
    # Full fail vs pred
    full_div = first_divergent(pred, fail)
    # Per n_tile for this query tile (acc has no n_tile dim)
    per_ntile = []
    if name == "acc":
      pt, ft = pred[focus_head, focus_bm], fail[focus_head, focus_bm]
      div = first_divergent(pt, ft)
      per_ntile.append({"n_tile": None, "div": div, "vs_hip_pred": first_divergent(pt, hip[focus_head, focus_bm])})
      if div is not None and earliest is None:
        earliest = {"phase": name, "n_tile": None, "div": div}
    else:
      ntiles = pred.shape[2]
      for nt in range(ntiles):
        pt, ft = pred[focus_head, focus_bm, nt], fail[focus_head, focus_bm, nt]
        div = first_divergent(pt, ft)
        per_ntile.append({"n_tile": nt, "div": div,
                          "vs_hip_pred": first_divergent(pt, hip[focus_head, focus_bm, nt]),
                          "vs_hip_fail": first_divergent(ft, hip[focus_head, focus_bm, nt])})
        if div is not None and earliest is None:
          earliest = {"phase": name, "n_tile": nt, "div": div}
        print(f"  phase[{name}] head={focus_head} bm={focus_bm} n_tile={nt} "
              f"fail_vs_pred={'EXACT' if div is None else div}")
    cmp["phases"][name] = {"full_fail_vs_pred": full_div, "per_ntile": per_ntile}
    if full_div:
      print(f"  phase[{name}] FULL fail_vs_pred={full_div}")

  cmp["earliest"] = earliest
  if earliest:
    print(f"  EARLIEST divergence: phase={earliest['phase']} n_tile={earliest['n_tile']} {earliest['div']}")
    div = earliest.get("div") or {}
    coords = div.get("coords") or []
    # PV dumps: tile slice is (BLOCK_M, D) → coords [qrow, dcol]
    if earliest["phase"] in ("pv_wmma", "pv", "acc") and len(coords) >= 2:
      qrow, dcol = int(coords[0]), int(coords[1])
      cmap = map_pv_coord(qrow, dcol)
      cmp["earliest_coord_map"] = cmap
      print(f"  coord map: {cmap}")
  else:
    print("  WARNING: out diverged but selected phase dumps match — try more phases")

  # Probe table: pre-copy vs post-copy PV
  probe = None
  if "pv_wmma" in cmp["phases"] or "pv" in cmp["phases"]:
    def _phase_has_div(name: str) -> bool:
      p = cmp["phases"].get(name)
      if not p: return False
      return p.get("full_fail_vs_pred") is not None or any(x.get("div") for x in p.get("per_ntile", []))
    wmma_bad, pv_bad, acc_bad = _phase_has_div("pv_wmma"), _phase_has_div("pv"), _phase_has_div("acc")
    if wmma_bad:
      probe = "pv_wmma_before_copy_wrong"
      note = "P/V LDS, WMMA inputs, ACC init, or WMMA scheduling"
    elif pv_bad:
      probe = "pv_wmma_ok_post_copy_wrong"
      note = "REG_STORE/MOV copy, phys overlap, or spill/reload of slot 17"
    elif acc_bad:
      probe = "both_pv_ok_acc_diverges"
      note = "loop-carried slot-2 accumulator path"
    else:
      probe = "pv_stages_match"
      note = "selected PV dumps match fail vs pred"
    cmp["probe"] = {"verdict": probe, "note": note,
                    "pv_wmma_div": wmma_bad, "pv_div": pv_bad, "acc_div": acc_bad}
    print(f"  PROBE: {probe} — {note}")

  # Compile-time slot 10/17 trace (same ELF for all launches)
  slot_trace = None
  if any(p in phases for p in ("pv_wmma", "pv")):
    focus_coord = (cmp.get("earliest_coord_map") or {})
    try:
      slot_trace = trace_pv_slots(dir_prg, focus=focus_coord or None)
      (phase_art / "pv_slot_trace.json").write_text(json.dumps(slot_trace, indent=2, default=str))
      print(f"  slot trace: pv_wmma_acc={slot_trace['pv_wmma_acc_phys']} "
            f"soft={slot_trace['pv_soft_phys'][:8]}{'…' if len(slot_trace['pv_soft_phys'])>8 else ''} "
            f"spill/fill_watched={len(slot_trace['spill_fill_watched'])}")
      if slot_trace.get("focus"):
        print(f"  slot focus: {slot_trace['focus']}")
    except Exception as e:
      slot_trace = {"error": str(e)}
      print(f"  slot trace FAILED: {e}")

  row = {
    "ok": False, "verdict": "launch_nondeterminism_instrumented",
    "replays_until_fail": fail_i + 1, "phases": list(phases),
    "compare": cmp, "tile_dims": {"TM": TM, "TN": TN, "TD": TD},
    "probe": cmp.get("probe"), "pv_slot_trace": slot_trace,
  }
  (phase_art / "compare.json").write_text(json.dumps(cmp, indent=2, default=str))
  (phase_art / "summary.json").write_text(json.dumps(row, indent=2, default=str))
  return row


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--modes", default="recreate,fixed,scratch",
                 help="comma: recreate,fixed,scratch (ignored with --phase-only)")
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--replays", type=int, default=100)
  p.add_argument("--skip", default="2")
  p.add_argument("--work", type=int, default=1)
  p.add_argument("--ref-tol", type=float, default=0.0)
  p.add_argument("--phase", action="store_true",
                 help="after modes: instrumented fail/pred phase capture")
  p.add_argument("--phase-only", action="store_true",
                 help="skip recreate/fixed/scratch; only instrumented fail capture")
  p.add_argument("--phases", default="qk",
                 help="comma dumps: qk,soft_m,soft_l,pv_wmma,pv,acc "
                      "(pv_wmma=pre ACC→soft copy; pv=post-copy)")
  p.add_argument("--focus-head", type=int, default=None)
  p.add_argument("--focus-bm", type=int, default=None)
  p.add_argument("--artifacts", default="extra/rdna3_state_diag")
  args = p.parse_args()
  T = min(args.T, args.S)
  modes = [m.strip() for m in args.modes.split(",") if m.strip()]
  phases = tuple(x.strip() for x in args.phases.split(",") if x.strip())
  art = Path(args.artifacts)
  if art.exists(): shutil.rmtree(art)
  art.mkdir(parents=True)

  acc_work = use_acc_work(args.skip, args.work)
  q_np = patterned((32, T, 128), np.float16)
  kv_np = patterned((2, 1, 8, args.S, 128), np.float16)
  np.save(art / "q.npy", q_np)
  np.save(art / "cache.npy", kv_np)

  # Self-check: matching -inf must not look divergent
  _a = np.array([1.0, float("-inf")], np.float32)
  assert first_divergent(_a, _a.copy()) is None, "inf-aware compare broken"
  assert arrays_close(_a, _a.copy())

  summary: list = []
  if not args.phase_only:
    print("======== HIP reference ========")
    set_dev("AMD:HIP")
    apply_env(skip=args.skip, work=args.work, direct=False)
    set_dev("AMD:HIP")
    hip_prg, _ = compile_program(args.S, T, acc_work, phases=())
    assert Device["AMD"].renderer.__class__.__name__ == "HIPRenderer"
    hip_outs = []
    hip_state = make_fixed_state(hip_prg, q_np, kv_np, T)
    for i in range(args.replays):
      hip_outs.append(launch_fixed(hip_state))
    hip_row = summarize_outs(hip_outs, None, args.ref_tol)
    hip_art = art / "hip_fixed"
    hip_art.mkdir(parents=True)
    save_fail_pair(hip_art, hip_outs, None)
    (hip_art / "summary.json").write_text(json.dumps(hip_row, indent=2, default=str))
    ref = hip_outs[0]
    print(f"  HIP verdict={hip_row['verdict']} exact={hip_row['exact']}")
    summary.append({"config": "hip_fixed", **hip_row})

    set_dev("AMD:AMD")
    apply_env(skip=args.skip, work=args.work, direct=True)
    set_dev("AMD:AMD")
    prg, _ = compile_program(args.S, T, acc_work, phases=())
    assert Device["AMD"].renderer.__class__.__name__ == "AMDRenderer"
    elf = program_elf(prg)
    if elf:
      (art / "direct.elf").write_bytes(elf)
      print(f"DIRECT elf={sha256(elf)[:16]}…")

    for mode in modes:
      if mode == "scratch":
        for sm in ("untouched", "zero", "nonzero"):
          set_dev("AMD:AMD")
          apply_env(skip=args.skip, work=args.work, direct=True)
          set_dev("AMD:AMD")
          summary.append(run_mode("fixed", prg, q_np, kv_np, T, args.replays, art, ref, args.ref_tol, scratch_mode=sm))
      else:
        set_dev("AMD:AMD")
        apply_env(skip=args.skip, work=args.work, direct=True)
        set_dev("AMD:AMD")
        summary.append(run_mode(mode, prg, q_np, kv_np, T, args.replays, art, ref, args.ref_tol))

  if args.phase or args.phase_only:
    try:
      phase_row = run_phase_fail_capture(
        args.S, T, acc_work, art, q_np, kv_np, phases=phases, replays=args.replays,
        focus_head=args.focus_head, focus_bm=args.focus_bm)
      summary.append({"config": "phase", **phase_row})
    except Exception as e:
      import traceback
      traceback.print_exc()
      summary.append({"config": "phase", "ok": False, "error": str(e)})

  print("\n======== SUMMARY ========")
  print(json.dumps(summary, indent=2, default=str))
  (art / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
  # Phase capture expects failure (ok=False with launch_nondeterminism_instrumented is success-of-diag)
  phase_ok = any(
    r.get("config") == "phase" and r.get("verdict") == "launch_nondeterminism_instrumented"
    for r in summary)
  other_fail = any(not r.get("ok", False) for r in summary if r.get("config") not in ("phase",))
  if args.phase_only:
    if phase_ok:
      print(f"OK phase diag captured instrumented failure under {art}")
      raise SystemExit(0)
    print(f"FAIL phase diag under {art}")
    raise SystemExit(1)
  if other_fail and not (args.phase and phase_ok):
    print(f"FAIL artifacts under {art}")
    raise SystemExit(1)
  print(f"done artifacts under {art}")
  raise SystemExit(0 if not other_fail else 1)


if __name__ == "__main__":
  main()
