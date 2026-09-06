#!/usr/bin/env python3
"""Freeze/replay flash kernel: same ELF vs fresh recompiles.

Distinguishes:
  - compile nondeterminism (ELF bytes differ across fresh to_program calls)
  - launch nondeterminism (one frozen PROGRAM/ELF, repeated launches, fresh out bufs)

Configs (packing/FMA/eviction off):
  prom          SKIP="" work=0 batch=1 cluster=1
  skip2_nobatch SKIP=2  work=1 batch=0 cluster=1
  skip2         SKIP=2  work=1 batch=1 cluster=1  (shipping control)

Example:
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_kernel_freeze_replay.py
  DEV=AMD:AMD PYTHONPATH=.:extra python extra/rdna3_flash_kernel_freeze_replay.py --config prom --S 128
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
from tinygrad.helpers import Context, getenv
from tinygrad.llm.kernels.amd import _amd_flash_attention
import tinygrad.llm.kernels.amd as amd
from tinygrad.uop.ops import Ops, UOp


def patterned(shape, dtype=np.float32):
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def apply_config(name: str) -> dict[str, str]:
  os.environ["AMD_FLASH_DIRECT"] = "1"
  os.environ["AMD_FLASH_ACC_SMALL"] = "1"
  os.environ["AMD_SPILL_ON_EVICT"] = "0"
  os.environ["AMD_PACK_SLOAD_B128"] = "0"
  os.environ.pop("AMD_FMA_MIX", None)
  if name == "prom":
    os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = ""
    os.environ["AMD_FLASH_ACC_WORK"] = "0"
    os.environ["AMD_BATCH_SLOAD_USE"] = "1"
    os.environ["AMD_CLUSTER_SLOAD"] = "1"
    use_acc_work = False
  elif name == "skip2_nobatch":
    os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = "2"
    os.environ["AMD_FLASH_ACC_WORK"] = "1"
    os.environ["AMD_BATCH_SLOAD_USE"] = "0"
    os.environ["AMD_CLUSTER_SLOAD"] = "1"
    use_acc_work = True
  elif name == "skip2":
    os.environ["AMD_REG_PROMOTE_SKIP_SLOTS"] = "2"
    os.environ["AMD_FLASH_ACC_WORK"] = "1"
    os.environ["AMD_BATCH_SLOAD_USE"] = "1"
    os.environ["AMD_CLUSTER_SLOAD"] = "1"
    use_acc_work = True
  else:
    raise SystemExit(f"unknown config {name}")
  getenv.cache_clear()
  to_program_cache.clear()
  amd._amd_flash_attention.cache_clear()
  runtime_cache.clear()
  env = {k: os.environ.get(k, "") for k in (
    "AMD_FLASH_DIRECT", "AMD_FLASH_ACC_SMALL", "AMD_REG_PROMOTE_SKIP_SLOTS",
    "AMD_FLASH_ACC_WORK", "AMD_BATCH_SLOAD_USE", "AMD_CLUSTER_SLOAD",
    "AMD_SPILL_ON_EVICT", "AMD_PACK_SLOAD_B128")}
  env["_use_acc_work"] = str(int(use_acc_work))
  return env


def sha256(b: bytes) -> str:
  return hashlib.sha256(b).hexdigest()


def compile_program(S: int, T: int, use_acc_work: bool):
  out = UOp.placeholder((32, T, 128), dtypes.float32, 0)
  q = UOp.placeholder((32, T, 128), dtypes.float16, 1)
  kv = UOp.placeholder((2, 1, 8, S, 128), dtypes.float16, 2)
  sink = _amd_flash_attention(out, q, kv, valid_kv_len=S, q_start=None,
                              acc_small=True, k_unroll=0, use_acc_work=use_acc_work)
  with Context(BEAM=0):
    # Match flash_attention's ACC_SMALL env park for the renderer.
    prev = os.environ.get("AMD_FLASH_ACC_SMALL")
    os.environ["AMD_FLASH_ACC_SMALL"] = "1"
    getenv.cache_clear()
    try:
      prg = to_program(sink, Device[Device.DEFAULT].renderer)
    finally:
      if prev is None: os.environ.pop("AMD_FLASH_ACC_SMALL", None)
      else: os.environ["AMD_FLASH_ACC_SMALL"] = prev
      getenv.cache_clear()
  return prg


def program_elf(prg) -> bytes:
  binary = next(s.arg for s in prg.src if s.op is Ops.BINARY)
  assert isinstance(binary, bytes) and binary.startswith(b"\x7fELF"), "missing ELF"
  return binary


def program_meta(prg) -> dict:
  info = prg.arg
  return {
    "function_name": info.function_name,
    "global_size": list(info.global_size) if info.global_size is not None else None,
    "local_size": list(info.local_size) if info.local_size is not None else None,
    "globals": list(info.globals),
    "estimates": str(getattr(info, "estimates", None)),
  }


def make_inputs(S: int, T: int):
  q_np = patterned((32, T, 128), np.float16)
  kv_np = patterned((2, 1, 8, S, 128), np.float16)
  q = Tensor(q_np, device=Device.DEFAULT, dtype=dtypes.float16).realize()
  kv = Tensor(kv_np, device=Device.DEFAULT, dtype=dtypes.float16).realize()
  return q, kv, q_np, kv_np


def launch(prg, q: Tensor, kv: Tensor, T: int) -> np.ndarray:
  """Launch frozen program with a brand-new output buffer; return host f32 [32,T,128]."""
  device = Device.DEFAULT
  out = Tensor.empty(32, T, 128, dtype=dtypes.float32, device=device).realize()
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


def first_divergent(a: np.ndarray, b: np.ndarray, block_m: int = 32) -> dict | None:
  d = np.abs(a.astype(np.float64) - b.astype(np.float64))
  if not np.any(d): return None
  idx = int(np.argmax(d > 0))
  coords = [int(c) for c in np.unravel_index(idx, a.shape)]
  # out layout from placeholder: (BH=32 heads, T, D) — head=coords[0], token=coords[1]
  head, token = coords[0], coords[1]
  return {
    "flat_index": idx,
    "coords_HTD": coords,
    "head": head,
    "token": token,
    "query_tile": token // block_m,
    "maxdiff": float(d.max()),
    "a": float(a.reshape(-1)[idx]),
    "b": float(b.reshape(-1)[idx]),
  }


def elf_inst_summary(elf: bytes) -> dict:
  """Lightweight asm fingerprint for differing binaries."""
  try:
    from tinygrad.renderer.amd import decode_inst
    from tinygrad.runtime.support.elf import elf_loader
    text = next(s.content for s in elf_loader(elf)[1] if s.name == ".text")
    names: list[str] = []
    off = 0
    while off < len(text):
      inst = decode_inst(text[off:], "rdna3")
      names.append(getattr(inst, "op_name", "?"))
      off += inst.size()
      if names[-1].lower() == "s_endpgm": break
    from collections import Counter
    return {"n_inst": len(names), "top": Counter(names).most_common(12),
            "text_sha256": sha256(text)}
  except Exception as e:
    return {"error": str(e)}


def save_fail(art: Path, **files) -> None:
  art.mkdir(parents=True, exist_ok=True)
  for name, val in files.items():
    path = art / name
    if isinstance(val, (bytes, bytearray)):
      path.write_bytes(val)
    elif isinstance(val, np.ndarray):
      np.save(path, val)
    else:
      path.write_text(val if isinstance(val, str) else json.dumps(val, indent=2, default=str))


def run_config(cfg: str, S: int, T: int, replays: int, recompiles: int, art_root: Path) -> dict:
  print(f"\n======== config={cfg} S={S} T={T} ========")
  env = apply_config(cfg)
  use_acc_work = bool(int(env["_use_acc_work"]))
  cfg_art = art_root / f"{cfg}_S{S}"
  if cfg_art.exists(): shutil.rmtree(cfg_art)
  cfg_art.mkdir(parents=True)
  save_fail(cfg_art, **{"env.json": env})

  _, _, q_np, kv_np = make_inputs(S, T)
  np.save(cfg_art / "q.npy", q_np)
  np.save(cfg_art / "cache.npy", kv_np)

  def launch_fresh_inputs(prg) -> np.ndarray:
    """New device copies of q/kv each launch (rules out host-visible input clobber)."""
    qq = Tensor(q_np.copy(), device=Device.DEFAULT, dtype=dtypes.float16).realize()
    kk = Tensor(kv_np.copy(), device=Device.DEFAULT, dtype=dtypes.float16).realize()
    return launch(prg, qq, kk, T)

  # --- fresh recompiles ---
  print("-- recompiles --")
  recomps: list[tuple[str, dict, bytes, object, np.ndarray]] = []
  for i in range(recompiles):
    to_program_cache.clear()
    amd._amd_flash_attention.cache_clear()
    runtime_cache.clear()
    getenv.cache_clear()
    prg = compile_program(S, T, use_acc_work)
    elf = program_elf(prg)
    meta = program_meta(prg)
    out = launch_fresh_inputs(prg)
    h = sha256(elf)
    print(f"  recompile[{i}] name={meta['function_name']} elf={h[:16]}… "
          f"mean={float(out.mean()):.8g} finite={bool(np.isfinite(out).all())}")
    recomps.append((h, meta, elf, prg, out))
    save_fail(cfg_art, **{
      f"recompile_{i}.elf": elf,
      f"recompile_{i}_meta.json": meta,
      f"recompile_{i}_out.npy": out,
      f"recompile_{i}_sha.txt": h + "\n",
      f"recompile_{i}_asm.json": elf_inst_summary(elf),
    })

  elf_shas = [r[0] for r in recomps]
  elf_unique = sorted(set(elf_shas))
  outs_r = [r[4] for r in recomps]
  out_finite = all(np.isfinite(o).all() for o in outs_r)
  out_exact = all(np.array_equal(outs_r[0], o) for o in outs_r[1:])
  out_max = max((float(np.max(np.abs(outs_r[0] - o))) for o in outs_r[1:]), default=0.0)
  print(f"  unique_elfs={len(elf_unique)} shas={[s[:12] for s in elf_unique]}")
  print(f"  recompile_out exact={out_exact} finite={out_finite} maxdiff={out_max:.6g}")
  div_c = None
  if not out_exact:
    other = next(o for o in outs_r[1:] if not np.array_equal(outs_r[0], o))
    div_c = first_divergent(outs_r[0], other)
    print(f"  first_divergent recompile: {div_c}")
  if len(elf_unique) > 1:
    a = next(r for r in recomps if r[0] == elf_unique[0])
    b = next(r for r in recomps if r[0] == elf_unique[1])
    print(f"  elf asm fingerprint A: {elf_inst_summary(a[2])}")
    print(f"  elf asm fingerprint B: {elf_inst_summary(b[2])}")

  # --- freeze first compile, replay ---
  print("-- freeze/replay --")
  apply_config(cfg)
  prg0 = compile_program(S, T, use_acc_work)
  elf0 = program_elf(prg0)
  meta0 = program_meta(prg0)
  print(f"  frozen name={meta0['function_name']} elf={sha256(elf0)[:16]}…")
  save_fail(cfg_art, **{
    "frozen.elf": elf0,
    "frozen_meta.json": meta0,
    "frozen_asm.json": elf_inst_summary(elf0),
  })

  replay_outs = [launch_fresh_inputs(prg0) for _ in range(replays)]
  q_reuse = Tensor(q_np.copy(), device=Device.DEFAULT, dtype=dtypes.float16).realize()
  kv_reuse = Tensor(kv_np.copy(), device=Device.DEFAULT, dtype=dtypes.float16).realize()
  reuse_outs = [launch(prg0, q_reuse, kv_reuse, T) for _ in range(min(replays, 6))]
  rep_finite = all(np.isfinite(o).all() for o in replay_outs)
  rep_exact = all(np.array_equal(replay_outs[0], o) for o in replay_outs[1:])
  rep_max = max((float(np.max(np.abs(replay_outs[0] - o))) for o in replay_outs[1:]), default=0.0)
  reuse_exact = all(np.array_equal(reuse_outs[0], o) for o in reuse_outs[1:])
  reuse_max = max((float(np.max(np.abs(reuse_outs[0] - o))) for o in reuse_outs[1:]), default=0.0)
  print(f"  frozen_replay_reload_inputs exact={rep_exact} finite={rep_finite} maxdiff={rep_max:.6g}")
  print(f"  frozen_replay_reuse_inputs  exact={reuse_exact} maxdiff={reuse_max:.6g}")
  for i, o in enumerate(replay_outs):
    np.save(cfg_art / f"replay_{i}.npy", o)
  for i, o in enumerate(reuse_outs):
    np.save(cfg_art / f"reuse_{i}.npy", o)
  div_r = None
  if not rep_exact:
    other = next(o for o in replay_outs[1:] if not np.array_equal(replay_outs[0], o))
    div_r = first_divergent(replay_outs[0], other)
    print(f"  first_divergent replay: {div_r}")
  elif not reuse_exact:
    other = next(o for o in reuse_outs[1:] if not np.array_equal(reuse_outs[0], o))
    div_r = first_divergent(reuse_outs[0], other)
    print(f"  first_divergent reuse: {div_r}")

  row = {
    "config": cfg,
    "unique_elfs": len(elf_unique),
    "elf_shas": elf_unique,
    "compile_deterministic": len(elf_unique) == 1,
    "recompile_out_exact": out_exact,
    "recompile_out_maxdiff": out_max,
    "recompile_finite": out_finite,
    "frozen_replay_exact": rep_exact,
    "frozen_replay_maxdiff": rep_max,
    "frozen_replay_finite": rep_finite,
    "frozen_reuse_exact": reuse_exact,
    "frozen_reuse_maxdiff": reuse_max,
    "frozen_elf_sha": sha256(elf0),
    "function_name": meta0["function_name"],
    "first_divergent_recompile": div_c,
    "first_divergent_replay": div_r,
    "verdict": (
      "launch_nondeterminism" if len(elf_unique) == 1 and (not rep_exact or not reuse_exact) else
      "compile_nondeterminism" if len(elf_unique) > 1 else
      "stable" if out_exact and rep_exact and reuse_exact else
      "output_mismatch"
    ),
  }
  save_fail(cfg_art, **{"summary.json": row})

  ok = (len(elf_unique) == 1 and out_exact and out_finite and rep_exact and reuse_exact and rep_finite)
  if not out_finite or not rep_finite:
    print("FAIL: nonfinite outputs")
  if len(elf_unique) > 1:
    print("FAIL: fresh compilations produced different ELF binaries (compile nondeterminism)")
  if len(elf_unique) == 1 and (not rep_exact or not reuse_exact):
    print("FAIL: identical frozen ELF, varying outputs across launches (launch nondeterminism)")
  if ok:
    print("OK: stable compile + stable frozen replay")
  row["ok"] = ok
  return row


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--config", choices=("prom", "skip2_nobatch", "skip2", "all"), default="all")
  p.add_argument("--S", type=int, default=128)
  p.add_argument("--T", type=int, default=128)
  p.add_argument("--replays", type=int, default=8)
  p.add_argument("--recompiles", type=int, default=4)
  p.add_argument("--artifacts", type=str, default="extra/rdna3_freeze_artifacts")
  args = p.parse_args()
  T = min(args.T, args.S)
  configs = ["prom", "skip2_nobatch", "skip2"] if args.config == "all" else [args.config]
  art_root = Path(args.artifacts)
  print("device", Device.DEFAULT, Device[Device.DEFAULT].renderer.__class__.__name__,
        getattr(Device[Device.DEFAULT], "arch", None))

  summary = []
  failed = False
  for cfg in configs:
    try:
      row = run_config(cfg, args.S, T, args.replays, args.recompiles, art_root)
    except Exception as e:
      import traceback
      traceback.print_exc()
      row = {"config": cfg, "ok": False, "error": str(e)}
      failed = True
    summary.append(row)
    if not row.get("ok", False):
      failed = True

  print("\n======== SUMMARY ========")
  print(json.dumps(summary, indent=2))
  art_root.mkdir(parents=True, exist_ok=True)
  (art_root / "summary.json").write_text(json.dumps(summary, indent=2))
  if failed:
    print(f"FAIL artifacts under {art_root}")
    raise SystemExit(1)
  print("OK all configs stable")
  raise SystemExit(0)


if __name__ == "__main__":
  main()
