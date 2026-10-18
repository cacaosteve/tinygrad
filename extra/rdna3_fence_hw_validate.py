#!/usr/bin/env python3
"""HCQ2 fence HW gate: prefill TinyJit multi-submit + sync-each fair timings.

Prints JSON lines. Success requires completed batches and NumPy-correct outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import DEV, getenv
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, flash_attention


def patterned(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def setup(case: str, *, direct: bool):
  target = "AMD:AMD" if direct else "AMD:HIP"
  os.environ["DEV"] = target
  if direct: os.environ["AMD_FLASH_DIRECT"] = "1"
  else: os.environ.pop("AMD_FLASH_DIRECT", None)
  DEV.value = target
  getenv.cache_clear()
  # Force reopen so DEV / renderer stick across modes in one process sweeps —
  # this script runs one mode per process.
  batch, heads, kv_heads, dim, physical_n = 1, 32, 8, 128, 2048
  tokens = 1 if case == "decode" else 32
  q = Tensor(patterned((batch, heads, tokens, dim)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, batch, kv_heads, physical_n, dim), np.float16), device=Device.DEFAULT).realize()
  ren = Device[Device.DEFAULT].renderer.__class__.__name__
  arch = getattr(Device[Device.DEFAULT], "arch", "?")
  custom = amd_custom_kernels_supported(Device.DEFAULT)

  def attention(query: Tensor) -> Tensor:
    return flash_attention(query, cache, physical_n).realize()

  keys, values = (np.repeat(patterned((2, batch, kv_heads, physical_n, dim), np.float16)[i].astype(np.float32),
                            heads // kv_heads, axis=1) for i in range(2))
  scores = patterned((batch, heads, tokens, dim)) @ keys.swapaxes(-1, -2) / np.sqrt(dim)
  if tokens > 1:
    scores = np.where(np.arange(physical_n)[None, :] <= physical_n-tokens+np.arange(tokens)[:, None], scores, -np.inf)
  probs = np.exp(scores-scores.max(axis=-1, keepdims=True))
  expected = probs/probs.sum(axis=-1, keepdims=True) @ values
  runner = TinyJit(attention)
  return {
    "case": case, "direct": direct, "device": Device.DEFAULT, "renderer": ren, "arch": arch,
    "custom": custom, "tokens": tokens, "physical_n": physical_n, "q": q, "runner": runner, "expected": expected,
  }


def check_output(ctx: dict, out: Tensor) -> float:
  actual = out.numpy()
  np.testing.assert_allclose(actual, ctx["expected"], atol=2e-3, rtol=2e-3)
  return float(np.max(np.abs(actual-ctx["expected"])))


def fence_probe(ctx: dict, *, batch: int) -> dict:
  q, runner = ctx["q"], ctx["runner"]
  for _ in range(3):
    out = runner(q)
    Device[Device.DEFAULT].synchronize()
  check_output(ctx, out)
  t0 = time.perf_counter()
  for _ in range(batch):
    out = runner(q)
  Device[Device.DEFAULT].synchronize()
  elapsed_ms = (time.perf_counter() - t0) * 1e3
  return {"ok": True, "batch": batch, "elapsed_ms": elapsed_ms, "per_call_us": elapsed_ms * 1e3 / batch,
          "max_abs_err": check_output(ctx, out)}


def sync_each(ctx: dict, *, warmup: int = 5, samples: int = 30) -> dict:
  q, runner = ctx["q"], ctx["runner"]
  for _ in range(warmup):
    out = runner(q)
    Device[Device.DEFAULT].synchronize()
  xs = []
  for _ in range(samples):
    t0 = time.perf_counter_ns()
    out = runner(q)
    Device[Device.DEFAULT].synchronize()
    xs.append((time.perf_counter_ns() - t0) / 1e3)
  return {
    "median_us": statistics.median(xs),
    "best_us": min(xs),
    "mean_us": statistics.mean(xs),
    "samples_us": xs,
    "max_abs_err": check_output(ctx, out),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--mode", choices=("fence", "sync-each", "all"), default="all")
  p.add_argument("--case", choices=("prefill", "decode", "both"), default="both")
  p.add_argument("--direct", type=int, choices=(0, 1), default=1)
  args = p.parse_args()
  cases = ["prefill", "decode"] if args.case == "both" else [args.case]
  direct = bool(args.direct)

  for case in cases:
    ctx = setup(case, direct=direct)
    base = {k: ctx[k] for k in ("case", "direct", "device", "renderer", "arch", "custom", "tokens", "physical_n")}
    print(json.dumps({"event": "setup", **base}), flush=True)
    if args.mode in ("fence", "all") and case == "prefill":
      for b in (10, 50, 100):
        r = fence_probe(ctx, batch=b)
        print(json.dumps({"event": "fence", **base, **r}), flush=True)
    if args.mode in ("sync-each", "all"):
      r = sync_each(ctx)
      print(json.dumps({"event": "sync_each", **base,
                        "median_us": r["median_us"], "best_us": r["best_us"], "mean_us": r["mean_us"],
                        "n": len(r["samples_us"]), "max_abs_err": r["max_abs_err"]}), flush=True)


if __name__ == "__main__":
  main()
