#!/usr/bin/env python3
"""Flash DIRECT state diagnostics: fixed buffers, scratch history, phase dumps.

Stops scheduling experiments. Compares:
  1. recreate  — new q/kv/out each launch (prior harness)
  2. fixed     — same runtime + q/kv/out buffers + launch args, 100 launches,
                 no intervening helper kernels
  3. scratch   — untouched / zero / nonzero fill of device scratch immediately
                 before each Flash launch (isolated sync process intent)

Also saves the first failing replay and its predecessor (not only the first eight).

Phase dumps (optional): compile with phase_dumps and compare QK / soft_m0 / soft_l0 /
PV / pre-norm acc for one affected tile against HIP.

Example:
  PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --replays 100
  PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py --modes fixed,recreate,scratch --phase
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.engine.realize import get_runtime, runtime_cache
from tinygrad.helpers import Context, getenv, DEV
from tinygrad.llm.kernels.amd import _amd_flash_attention, BLOCK_M, WAVES_M
import tinygrad.llm.kernels.amd as amd
from tinygrad.uop.ops import Ops, UOp


TM, TN, TD = 4, 4, 4  # D=128 flash ACC_SMALL tile locals


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


def phase_shapes(BH: int, M: int, S: int) -> dict[str, tuple]:
  nm = M // BLOCK_M
  ntiles = (S + 31) // 32  # BLOCK_N=32
  return {
    "qk": (BH, nm, ntiles, WAVES_M, TM, TN),
    "soft_m": (BH, nm, ntiles, WAVES_M, TM),
    "soft_l": (BH, nm, ntiles, WAVES_M, TM),
    "pv": (BH, nm, ntiles, WAVES_M, TM, TD),
    "acc": (BH, nm, WAVES_M, TM, TD),
  }


def compile_program(S: int, T: int, acc_work: bool, *, with_phases: bool = False):
  BH = 32
  out = UOp.placeholder((BH, T, 128), dtypes.float32, 0)
  q = UOp.placeholder((BH, T, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, S, 128), dtypes.float16, 2)
  dumps: list[tuple[str, UOp]] = []
  dump_bufs: dict[str, UOp] = {}
  if with_phases:
    slot = 3
    for name, shape in phase_shapes(BH, T, S).items():
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


def first_divergent(a: np.ndarray, b: np.ndarray) -> dict | None:
  d = np.abs(a.astype(np.float64) - b.astype(np.float64))
  if not np.any(d): return None
  idx = int(np.argmax(d > 0))
  coords = [int(c) for c in np.unravel_index(idx, a.shape)]
  return {
    "flat_index": idx, "coords": coords, "maxdiff": float(d.max()),
    "a": float(a.reshape(-1)[idx]), "b": float(b.reshape(-1)[idx]),
    "head": coords[0] if len(coords) >= 1 else None,
    "token": coords[1] if len(coords) >= 2 else None,
    "query_tile": (coords[1] // BLOCK_M) if len(coords) >= 2 else None,
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
    for name in dump_names:
      shape = phase_shapes(32, T, S)[name]
      dumps[name] = Tensor.zeros(shape, dtype=dtypes.float32, device=device).realize()
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
    if not np.array_equal(outs[0], outs[i]):
      meta["first_fail_i"] = i
      meta["pred_i"] = i - 1
      np.save(art / f"fail_{i}.npy", outs[i])
      np.save(art / f"pred_{i-1}.npy", outs[i - 1])
      np.save(art / "replay_0.npy", outs[0])
      meta["div_vs_0"] = first_divergent(outs[0], outs[i])
      if i >= 2:
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
  finite = all(np.isfinite(o).all() for o in outs)
  exact = all(np.array_equal(outs[0], o) for o in outs[1:])
  maxdiff = max((float(np.max(np.abs(outs[0] - o))) for o in outs[1:]), default=0.0)
  div = None if exact else first_divergent(outs[0], next(o for o in outs[1:] if not np.array_equal(outs[0], o)))
  ref_ok, ref_max = True, 0.0
  if ref is not None:
    ref_max = float(np.max(np.abs(outs[0].astype(np.float64) - ref.astype(np.float64))))
    # Allow ulp noise when judging first-out vs HIP; bit-exact still preferred in meta.
    ref_ok = bool(ref_max <= max(ref_tol, 1e-5)) and bool(np.isfinite(outs[0]).all())
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


def run_phase_compare(S: int, T: int, acc_work: bool, art: Path, q_np, kv_np,
                      fail_meta: dict | None) -> dict:
  """Compile with phase dumps; compare HIP vs DIRECT at first failing tile (or tile 0)."""
  print("\n======== phase dump HIP vs DIRECT ========")
  phase_art = art / "phase"
  if phase_art.exists(): shutil.rmtree(phase_art)
  phase_art.mkdir(parents=True)

  dump_names = ["qk", "soft_m", "soft_l", "pv", "acc"]
  results = {}
  for label, target in (("hip", "AMD:HIP"), ("direct", "AMD:AMD")):
    set_dev(target)
    apply_env(direct=(label == "direct"))
    set_dev(target)
    prg, _ = compile_program(S, T, acc_work, with_phases=True)
    ren = Device["AMD"].renderer.__class__.__name__
    expect = "HIPRenderer" if label == "hip" else "AMDRenderer"
    if ren != expect:
      results[label] = {"error": f"expected {expect}, got {ren}"}
      print(f"  {label}: FAIL renderer {ren}")
      continue
    state = make_fixed_state(prg, q_np, kv_np, T, S, dump_names=dump_names)
    # Zero dump buffers then launch once
    for name, t in state["dumps"].items():
      t.assign(Tensor.zeros(t.shape, dtype=dtypes.float32, device="AMD")).realize()
    refill_out_sentinel(state)
    out = launch_fixed(state)
    dumps_np = {name: np.array(t.numpy(), dtype=np.float32, copy=True) for name, t in state["dumps"].items()}
    for name, arr in dumps_np.items():
      np.save(phase_art / f"{label}_{name}.npy", arr)
    np.save(phase_art / f"{label}_out.npy", out)
    results[label] = {"renderer": ren, "out_mean": float(out.mean()),
                      "dumps": {k: {"mean": float(v.mean()), "finite": bool(np.isfinite(v).all())}
                                for k, v in dumps_np.items()}}
    print(f"  {label}: out_mean={results[label]['out_mean']:.8g}")

  if "hip" in results and "direct" in results and "error" not in results["hip"] and "error" not in results["direct"]:
    tile = None
    if fail_meta and fail_meta.get("div_vs_0"):
      d = fail_meta["div_vs_0"]
      tile = {"head": d.get("head"), "query_tile": d.get("query_tile"), "token": d.get("token"), "n_tile": 0}
    if tile is None:
      tile = {"head": 0, "query_tile": 0, "token": 0, "n_tile": 0}
    print(f"  focusing tile head={tile['head']} query_tile={tile['query_tile']} n_tile={tile['n_tile']}")
    cmp = {"tile": tile, "phases": {}}
    for name in dump_names:
      ha = np.load(phase_art / f"hip_{name}.npy")
      da = np.load(phase_art / f"direct_{name}.npy")
      h, bm, nt = int(tile["head"]), int(tile["query_tile"]), int(tile["n_tile"])
      try:
        if name == "acc":
          ht, dt = ha[h, bm], da[h, bm]
        else:
          ht, dt = ha[h, bm, nt], da[h, bm, nt]
      except Exception:
        ht, dt = ha, da
      div = first_divergent(ht, dt)
      md = float(np.max(np.abs(ht.astype(np.float64) - dt.astype(np.float64))))
      cmp["phases"][name] = {"maxdiff": md, "exact": div is None, "first_divergent": div}
      print(f"  phase[{name}] tile maxdiff={md:.6g} exact={div is None} div={div}")
    for name in dump_names:
      ha = np.load(phase_art / f"hip_{name}.npy")
      da = np.load(phase_art / f"direct_{name}.npy")
      full = first_divergent(ha, da)
      cmp["phases"][name]["full_first_divergent"] = full
      if full:
        print(f"  phase[{name}] FULL first_divergent={full}")
    results["compare"] = cmp
    (phase_art / "compare.json").write_text(json.dumps(cmp, indent=2, default=str))
  (phase_art / "summary.json").write_text(json.dumps(results, indent=2, default=str))
  return results


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--modes", default="recreate,fixed,scratch",
                 help="comma: recreate,fixed,scratch")
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--replays", type=int, default=100)
  p.add_argument("--skip", default="2")
  p.add_argument("--work", type=int, default=1)
  p.add_argument("--ref-tol", type=float, default=0.0)
  p.add_argument("--phase", action="store_true", help="also run HIP vs DIRECT phase dumps")
  p.add_argument("--artifacts", default="extra/rdna3_state_diag")
  args = p.parse_args()
  T = min(args.T, args.S)
  modes = [m.strip() for m in args.modes.split(",") if m.strip()]
  art = Path(args.artifacts)
  if art.exists(): shutil.rmtree(art)
  art.mkdir(parents=True)

  acc_work = use_acc_work(args.skip, args.work)
  q_np = patterned((32, T, 128), np.float16)
  kv_np = patterned((2, 1, 8, args.S, 128), np.float16)
  np.save(art / "q.npy", q_np)
  np.save(art / "cache.npy", kv_np)

  # HIP reference (fixed, 8 replays enough if previously stable; still do 100 if requested)
  print("======== HIP reference ========")
  set_dev("AMD:HIP")
  apply_env(skip=args.skip, work=args.work, direct=False)
  set_dev("AMD:HIP")
  hip_prg, _ = compile_program(args.S, T, acc_work, with_phases=False)
  assert Device["AMD"].renderer.__class__.__name__ == "HIPRenderer"
  hip_outs = []
  hip_state = make_fixed_state(hip_prg, q_np, kv_np, T)
  for i in range(args.replays):
    # No out-refill helper kernel between HIP launches either.
    hip_outs.append(launch_fixed(hip_state))
  hip_row = summarize_outs(hip_outs, None, args.ref_tol)
  hip_art = art / "hip_fixed"
  hip_art.mkdir(parents=True)
  save_fail_pair(hip_art, hip_outs, None)
  (hip_art / "summary.json").write_text(json.dumps(hip_row, indent=2, default=str))
  ref = hip_outs[0]
  print(f"  HIP verdict={hip_row['verdict']} exact={hip_row['exact']}")
  if not hip_row["exact"]:
    print("WARN: HIP fixed replay not exact — reference itself is unstable")

  # DIRECT program
  set_dev("AMD:AMD")
  apply_env(skip=args.skip, work=args.work, direct=True)
  set_dev("AMD:AMD")
  prg, _ = compile_program(args.S, T, acc_work, with_phases=False)
  assert Device["AMD"].renderer.__class__.__name__ == "AMDRenderer"
  elf = program_elf(prg)
  if elf:
    (art / "direct.elf").write_bytes(elf)
    print(f"DIRECT elf={sha256(elf)[:16]}…")

  summary = [{"config": "hip_fixed", **hip_row}]
  fail_meta = None
  for mode in modes:
    if mode == "scratch":
      for sm in ("untouched", "zero", "nonzero"):
        # Fresh process-equivalent: re-apply env, same prg, fixed buffers + scratch fill
        set_dev("AMD:AMD")
        apply_env(skip=args.skip, work=args.work, direct=True)
        set_dev("AMD:AMD")
        row = run_mode("fixed", prg, q_np, kv_np, T, args.replays, art, ref, args.ref_tol, scratch_mode=sm)
        summary.append(row)
        if fail_meta is None and row.get("fail_meta", {}).get("first_fail_i") is not None:
          fail_meta = row["fail_meta"]
    else:
      set_dev("AMD:AMD")
      apply_env(skip=args.skip, work=args.work, direct=True)
      set_dev("AMD:AMD")
      row = run_mode(mode, prg, q_np, kv_np, T, args.replays, art, ref, args.ref_tol)
      summary.append(row)
      if fail_meta is None and row.get("fail_meta", {}).get("first_fail_i") is not None:
        fail_meta = row["fail_meta"]

  phase_results = None
  if args.phase:
    try:
      phase_results = run_phase_compare(args.S, T, acc_work, art, q_np, kv_np, fail_meta)
      summary.append({"config": "phase", "ok": True, "phase": phase_results.get("compare")})
    except Exception as e:
      import traceback
      traceback.print_exc()
      summary.append({"config": "phase", "ok": False, "error": str(e)})

  print("\n======== SUMMARY ========")
  print(json.dumps(summary, indent=2, default=str))
  (art / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
  failed = any(not r.get("ok", False) for r in summary if r.get("config") != "phase")
  # Phase compare is informational; don't require HIP==DIRECT bit-exact on dumps for exit
  if failed:
    print(f"FAIL artifacts under {art}")
    raise SystemExit(1)
  print(f"OK artifacts under {art}")
  raise SystemExit(0)


if __name__ == "__main__":
  main()
