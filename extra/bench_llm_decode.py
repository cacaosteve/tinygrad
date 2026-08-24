#!/usr/bin/env python3
import argparse, json, time

from bench_q6k_gemv import program_metrics
from tinygrad import Device, GlobalCounters, Tensor
from tinygrad.helpers import fetch, profile_marker
from tinygrad.llm.model import Transformer


def main():
  parser = argparse.ArgumentParser(description="Benchmark GGUF decode steps without tokenizer-dependent output formatting")
  parser.add_argument("model", help="local GGUF path or URL understood by tinygrad.helpers.fetch")
  parser.add_argument("--count", type=int, default=1, help="number of generated tokens")
  parser.add_argument("--max-context", type=int, default=128)
  parser.add_argument("--bos", type=int, help="override the GGUF tokenizer BOS token")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--program-metrics", help="print captured program metrics whose name contains this string")
  args = parser.parse_args()

  Tensor.manual_seed(args.seed)
  model, kv = Transformer.from_gguf(fetch(args.model), args.max_context)
  bos = args.bos if args.bos is not None else kv.get("tokenizer.ggml.bos_token_id", 0)
  print(f"device={Device.DEFAULT} model={kv.get('general.name', args.model)!r} bos={bos} max_context={args.max_context}")
  gen = model.generate([bos])
  for step in range(args.count):
    profile_marker(f"decode @ {step}")
    GlobalCounters.reset()
    st = time.perf_counter()
    token = next(gen)
    elapsed = time.perf_counter() - st
    print(f"step={step} token={token} elapsed_ms={elapsed*1e3:.3f} tok_s={1/elapsed:.3f} kernels={GlobalCounters.kernel_count}")
  profile_marker(f"decode @ {args.count}")
  if args.program_metrics:
    for metrics in program_metrics(model.rollout_jit):
      if args.program_metrics in str(metrics["name"]): print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__": main()
