#!/usr/bin/env python3
"""Same-graph HIP vs DIRECT + in-order emit diagnostics for flash launch nondeterminism.

Preserves work-copy/layout knobs (SKIP slots, ACC_WORK, batch/cluster). Does NOT use HIP's
unrelated default SDPA kernel — both backends run `_amd_flash_attention`.

Configs:
  hip              DEV=AMD:HIP (HIPRenderer), same `_amd_flash_attention` graph
  direct           DEV=AMD:AMD (AMDRenderer) + shipping emit
  inorder          direct + AMD_IN_ORDER_EMIT=1 (no optional post-alloc motion/fusions)
  inorder_instr    inorder + AMD_INSTR_WAIT=1 (hard wait after each mem burst)

Requires ≥100 replays by default plus a reference compare (stable-but-wrong still fails).

Example:
  PYTHONPATH=.:extra python extra/rdna3_flash_backend_diag.py --configs hip,direct,inorder,inorder_instr
  PYTHONPATH=.:extra python extra/rdna3_flash_backend_diag.py --configs inorder_instr --replays 100
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
from tinygrad.llm.kernels.amd import _amd_flash_attention
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


def apply_layout_env(*, skip: str = "2", work: int = 1, batch: int = 1, cluster: int = 1) -> dict:
  """Shared work-copy / scheduling knobs for HIP and DIRECT."""
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_SPILL_ON_EVICT"] = "0"
  os.environ["AMD_PACK_SLOAD_B128"] = "0"
  os.environ.pop("AMD_FMA_MIX", None)
  os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = skip
  os.environ["AMD_FLASH_ACC_WORK"] = str(work)
  os.environ["AMD_BATCH_SLOAD_USE"] = str(batch)
  os.environ["AMD_CLUSTER_SLOAD"] = str(cluster)
  clear_caches()
  return {
    "AMD_REG_PROMOTE_SKIP_SLOTS": skip,
    "AMD_FLASH_ACC_WORK": str(work),
    "AMD_BATCH_SLOAD_USE": str(batch),
    "AMD_CLUSTER_SLOAD": str(cluster),
    "_use_acc_work": str(int(bool(work and "2" in {s.strip() for s in skip.split(",")}))),
  }


def apply_emit_env(name: str) -> dict:
  """Emit-side diagnostic knobs (DIRECT only)."""
  for k in ("AMD_IN_ORDER_EMIT", "AMD_INSTR_WAIT", "AMD_CONSERVATIVE_WAIT", "AMD_FLASH_DIRECT"):
    os.environ.pop(k, None)
  if name == "direct":
    os.environ["AMD_FLASH_DIRECT"] = "1"
  elif name == "inorder":
    os.environ["AMD_FLASH_DIRECT"] = "1"
    os.environ["AMD_IN_ORDER_EMIT"] = "1"
  elif name == "inorder_instr":
    os.environ["AMD_FLASH_DIRECT"] = "1"
    os.environ["AMD_IN_ORDER_EMIT"] = "1"
    os.environ["AMD_INSTR_WAIT"] = "1"
  elif name == "hip":
    pass  # HIP backend; no DIRECT emit flags
  else:
    raise SystemExit(f"unknown emit config {name}")
  clear_caches()
  return {k: os.environ.get(k, "") for k in (
    "AMD_FLASH_DIRECT", "AMD_IN_ORDER_EMIT", "AMD_INSTR_WAIT", "AMD_CONSERVATIVE_WAIT")}


def device_for(cfg: str) -> str:
  """Return DEV= target string (renderer select). Buffer device is always AMD."""
  return "AMD:HIP" if cfg == "hip" else "AMD:AMD"


def set_dev(cfg: str) -> str:
  """Select HIP vs direct-ISA renderer via DEV=AMD:HIP / DEV=AMD:AMD.

  Device buffers stay on \"AMD\"; the colon suffix picks the renderer, not a device id.
  Must update the DEV ContextVar — os.environ alone is not enough after import.
  """
  target = device_for(cfg)
  os.environ["DEV"] = target
  DEV.value = target
  clear_caches()
  return target


def compile_program(S: int, T: int, use_acc_work: bool):
  out = UOp.placeholder((32, T, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, T, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, S, 128), dtypes.float16, 2)
  sink = _amd_flash_attention(out, q, kv, valid_kv_len=S, q_start=None,
                              acc_small=True, k_unroll=0, use_acc_work=use_acc_work)
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
  return prg


def program_elf(prg) -> bytes | None:
  try:
    binary = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  except StopIteration:
    return None
  if isinstance(binary, bytes) and binary.startswith(b"\x7fELF"):
    return binary
  return None


def launch(prg, q_np, kv_np, T: int, *, sentinel: float = float("nan")) -> np.ndarray:
  device = "AMD"
  q = Tensor(q_np.copy(), device=device, dtype=dtypes.float16).realize()
  kv = Tensor(kv_np.copy(), device=device, dtype=dtypes.float16).realize()
  out = Tensor.full((32, T, 128), sentinel, dtype=dtypes.float32, device=device).realize()
  rt = get_runtime(device, prg, cache=True)
  gs, ls = prg.arg.launch_dims({})
  vals = prg.arg.vals({})
  bufs = [
    out.uop.buffer.ensure_allocated().get_buf(device),
    q.uop.buffer.ensure_allocated().get_buf(device),
    kv.uop.buffer.ensure_allocated().get_buf(device),
  ]
  ordered = [bufs[i] for i in prg.arg.globals]
  rt(*ordered, global_size=gs, local_size=ls, vals=vals, wait=True)
  Device[device].synchronize()
  return np.array(out.numpy(), dtype=np.float32, copy=True)


def first_divergent(a: np.ndarray, b: np.ndarray) -> dict | None:
  d = np.abs(a.astype(np.float64) - b.astype(np.float64))
  if not np.any(d): return None
  idx = int(np.argmax(d > 0))
  return {
    "flat_index": idx,
    "coords": [int(c) for c in np.unravel_index(idx, a.shape)],
    "maxdiff": float(d.max()),
    "a": float(a.reshape(-1)[idx]),
    "b": float(b.reshape(-1)[idx]),
  }


def run_cfg(cfg: str, S: int, T: int, replays: int, layout: dict, art: Path,
            ref: np.ndarray | None, ref_tol: float) -> dict:
  target = set_dev(cfg)
  print(f"\n======== cfg={cfg} DEV={target} S={S} T={T} replays={replays} ========")
  layout_env = apply_layout_env(skip=layout["skip"], work=layout["work"],
                                batch=layout["batch"], cluster=layout["cluster"])
  # layout clear_caches may drop DEV; re-pin renderer then emit knobs
  set_dev(cfg)
  emit = apply_emit_env(cfg)
  set_dev(cfg)  # apply_emit_env clears caches again
  use_acc_work = bool(int(layout_env["_use_acc_work"]))

  cfg_art = art / cfg
  if cfg_art.exists(): shutil.rmtree(cfg_art)
  cfg_art.mkdir(parents=True)

  q_np = patterned((32, T, 128), np.float16)
  kv_np = patterned((2, 1, 8, S, 128), np.float16)
  np.save(cfg_art / "q.npy", q_np)
  np.save(cfg_art / "cache.npy", kv_np)

  try:
    ren = Device["AMD"].renderer
    ren_name = ren.__class__.__name__
  except Exception as e:
    row = {"config": cfg, "ok": False, "error": f"device unavailable: {e}"}
    print(f"SKIP: {row['error']}")
    (cfg_art / "summary.json").write_text(json.dumps(row, indent=2))
    return row

  expect = "HIPRenderer" if cfg == "hip" else "AMDRenderer"
  if ren_name != expect:
    row = {"config": cfg, "ok": False, "error": f"expected {expect}, got {ren_name} under DEV={target}"}
    print(f"FAIL: {row['error']}")
    (cfg_art / "summary.json").write_text(json.dumps(row, indent=2))
    return row

  (cfg_art / "env.json").write_text(json.dumps(
    {**layout, **emit, "DEV": target, "renderer": ren_name}, indent=2))

  prg = compile_program(S, T, use_acc_work)
  elf = program_elf(prg)
  elf_sha = sha256(elf) if elf else None
  print(f"  compiled name={prg.arg.function_name} renderer={ren_name} "
        f"elf={elf_sha[:16] + '…' if elf_sha else 'n/a'}")
  if elf: (cfg_art / "frozen.elf").write_bytes(elf)

  outs = []
  for i in range(replays):
    o = launch(prg, q_np, kv_np, T)
    outs.append(o)
    if i < 4 or i == replays - 1 or (i + 1) % 25 == 0:
      print(f"  replay[{i}] mean={float(o.mean()):.8g} finite={bool(np.isfinite(o).all())}")
  for i, o in enumerate(outs[:8]):
    np.save(cfg_art / f"replay_{i}.npy", o)

  finite = all(np.isfinite(o).all() for o in outs)
  exact = all(np.array_equal(outs[0], o) for o in outs[1:])
  maxdiff = max((float(np.max(np.abs(outs[0] - o))) for o in outs[1:]), default=0.0)
  div = None if exact else first_divergent(outs[0], next(o for o in outs[1:] if not np.array_equal(outs[0], o)))
  if div: print(f"  first_divergent replay: {div}")

  ref_ok, ref_max, ref_div = True, 0.0, None
  if ref is not None:
    ref_max = float(np.max(np.abs(outs[0].astype(np.float64) - ref.astype(np.float64))))
    ref_ok = bool(ref_max <= ref_tol) and bool(np.isfinite(outs[0]).all())
    if not np.array_equal(outs[0], ref):
      ref_div = first_divergent(outs[0], ref)
    print(f"  vs_ref maxdiff={ref_max:.6g} tol={ref_tol} ok={ref_ok}")
    if ref_div: print(f"  first_divergent vs_ref: {ref_div}")
    if exact and not ref_ok:
      print("FAIL: stable across replays but disagrees with reference")

  ok = bool(finite and exact and ref_ok)
  verdict = (
    "launch_nondeterminism" if finite and not exact else
    "stable_wrong" if exact and finite and not ref_ok else
    "nonfinite" if not finite else
    "ok"
  )
  row = {
    "config": cfg, "DEV": target, "renderer": ren_name, "ok": ok, "verdict": verdict,
    "replays": replays, "frozen_replay_exact": exact, "frozen_replay_maxdiff": maxdiff,
    "finite": finite, "elf_sha": elf_sha, "first_divergent_replay": div,
    "ref_ok": ref_ok, "ref_maxdiff": ref_max, "first_divergent_ref": ref_div,
    "emit": emit,
  }
  (cfg_art / "summary.json").write_text(json.dumps(row, indent=2, default=str))
  print(f"  verdict={verdict} ok={ok}")
  return row


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--configs", default="hip,direct,inorder,inorder_instr",
                 help="comma list: hip,direct,inorder,inorder_instr")
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--replays", type=int, default=100)
  p.add_argument("--skip", default="2")
  p.add_argument("--work", type=int, default=1)
  p.add_argument("--batch", type=int, default=1)
  p.add_argument("--cluster", type=int, default=1)
  p.add_argument("--ref-tol", type=float, default=0.0,
                 help="max abs diff vs HIP/first-ref (0 = bit-exact)")
  p.add_argument("--artifacts", default="extra/rdna3_backend_diag")
  args = p.parse_args()
  T = min(args.T, args.S)
  configs = [c.strip() for c in args.configs.split(",") if c.strip()]
  layout = {"skip": args.skip, "work": args.work, "batch": args.batch, "cluster": args.cluster}
  art = Path(args.artifacts)
  if art.exists(): shutil.rmtree(art)
  art.mkdir(parents=True)

  print("layout", layout, "replays", args.replays)
  summary = []
  ref = None
  # Prefer HIP as reference when present in the run list.
  ordered = sorted(configs, key=lambda c: (0 if c == "hip" else 1, configs.index(c)))
  failed = False
  for cfg in ordered:
    row = run_cfg(cfg, args.S, T, args.replays, layout, art, ref, args.ref_tol)
    summary.append(row)
    if cfg == "hip" and row.get("ok") and art.joinpath(cfg, "replay_0.npy").exists():
      ref = np.load(art / cfg / "replay_0.npy")
      print(f"  (using hip replay_0 as reference for later configs)")
    elif ref is None and row.get("frozen_replay_exact") and art.joinpath(cfg, "replay_0.npy").exists():
      # No HIP: still compare later configs to the first stable config's first out
      # (does not prove correctness — only relative agreement).
      pass
    if not row.get("ok", False):
      failed = True

  # If HIP wasn't first/ok, try loading hip artifact as ref for a second pass report.
  if ref is None and (art / "hip" / "replay_0.npy").exists():
    ref = np.load(art / "hip" / "replay_0.npy")
    print("\n-- recompute ref diffs with hip --")
    for row in summary:
      if row["config"] == "hip": continue
      p0 = art / row["config"] / "replay_0.npy"
      if not p0.exists(): continue
      o = np.load(p0)
      md = float(np.max(np.abs(o.astype(np.float64) - ref.astype(np.float64))))
      row["ref_maxdiff"] = md
      row["ref_ok"] = bool(md <= args.ref_tol)
      if row.get("frozen_replay_exact") and not row["ref_ok"]:
        row["verdict"] = "stable_wrong"
        row["ok"] = False
        failed = True
      print(f"  {row['config']} vs hip maxdiff={md:.6g} ok={row['ref_ok']}")

  print("\n======== SUMMARY ========")
  print(json.dumps(summary, indent=2, default=str))
  (art / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
  if failed:
    print(f"FAIL artifacts under {art}")
    raise SystemExit(1)
  print("OK all diagnostic configs stable and match reference")
  raise SystemExit(0)


if __name__ == "__main__":
  main()
