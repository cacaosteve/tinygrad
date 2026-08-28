#!/usr/bin/env python3
"""Cold first-call and steady-state probes for the RDNA3 LLM attention paths."""

import argparse
import json
import statistics
import time

import numpy as np

from bench_q6k_gemv import program_metrics
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported, flash_attention


def patterned(shape:tuple[int, ...], dtype=np.float32) -> np.ndarray:
  count = int(np.prod(shape))
  return (((np.arange(count, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)


def reference_attention(q:np.ndarray, cache:np.ndarray) -> np.ndarray:
  _, heads, tokens, dim = q.shape
  kv_heads, physical_n = cache.shape[2:4]
  group = heads // kv_heads
  qf, kf, vf = q.astype(np.float16).astype(np.float32), cache[0].astype(np.float32), cache[1].astype(np.float32)
  out = np.empty_like(q, dtype=np.float32)
  for h in range(heads):
    scores = qf[0, h] @ kf[0, h//group].T / np.sqrt(dim)
    for t in range(tokens): scores[t, physical_n-tokens+t+1:] = -np.inf
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out[0, h] = probs @ vf[0, h//group]
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--case", choices=("decode", "prefill"), required=True)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--warmup-seconds", type=float, default=1.0)
  parser.add_argument("--batch", type=int, default=50)
  parser.add_argument("--rounds", type=int, default=5)
  args = parser.parse_args()

  batch, heads, kv_heads, dim, physical_n = 1, 32, 8, 128, 2048
  tokens = 1 if args.case == "decode" else 32
  q_np, cache_np = patterned((batch, heads, tokens, dim)), patterned((2, batch, kv_heads, physical_n, dim), np.float16)
  reference = reference_attention(q_np, cache_np)
  q = Tensor(q_np, device=Device.DEFAULT).realize()
  cache = Tensor(cache_np, device=Device.DEFAULT).realize()

  def attention(query:Tensor) -> Tensor:
    if amd_custom_kernels_supported(query.device): return flash_attention(query, cache, physical_n).realize()
    k, v = cache[0], cache[1]
    mask = Tensor.full((1, 1, tokens, physical_n), float("-inf"), dtype=query.dtype, device=query.device, buffer=False).triu(physical_n-tokens+1)
    return query.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).realize()

  runner, result, call_ms = TinyJit(attention), None, []
  for _ in range(max(args.warmup, 2)):
    start = time.perf_counter_ns()
    result = runner(q)
    Device[Device.DEFAULT].synchronize()
    call_ms.append((time.perf_counter_ns() - start) / 1e6)

  warmup_end = time.perf_counter() + args.warmup_seconds
  while time.perf_counter() < warmup_end:
    for _ in range(args.batch): result = runner(q)
    Device[Device.DEFAULT].synchronize()

  samples_us = []
  for _ in range(args.rounds):
    start = time.perf_counter_ns()
    for _ in range(args.batch): result = runner(q)
    Device[Device.DEFAULT].synchronize()
    samples_us.append((time.perf_counter_ns() - start) / args.batch / 1e3)

  assert result is not None
  out = result.numpy().astype(np.float32)
  print(json.dumps({
    "device": Device.DEFAULT,
    "renderer": Device[Device.DEFAULT].renderer.__class__.__name__,
    "arch": Device[Device.DEFAULT].arch,
    "case": args.case,
    "shape": [batch, heads, tokens, physical_n, dim],
    "first_call_ms": call_ms[0],
    "capture_call_ms": call_ms[1],
    "warmup_call_ms": call_ms,
    "median_us": statistics.median(samples_us),
    "best_us": min(samples_us),
    "samples_us": samples_us,
    "max_abs_error": float(np.max(np.abs(out-reference))),
    "checksum": float(out.astype(np.float64).sum()),
    "first8": out.reshape(-1)[:8].tolist(),
    "programs": program_metrics(runner),
  }, indent=2))


if __name__ == "__main__": main()
