import argparse, json, time
from tinygrad.helpers import profile_marker
from tinygrad.llm.model import Transformer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="path to gguf model")
  parser.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  parser.add_argument("--prompt-tokens", type=int, default=1024, help="number of prompt tokens (default: %(default)s)")
  parser.add_argument("--decode-tokens", type=int, default=16, help="number of tokens to decode (default: %(default)s)")
  parser.add_argument("--chunk-size", type=int, default=32, help="chunk size for prefill (default: %(default)s)")
  parser.add_argument("--disable-custom-quant", action="store_true", help="use the generic quantized linear lowering")
  parser.add_argument("--program-metrics", help="print captured prefill program metrics whose name contains this string")
  parser.add_argument("--dump-machine", action="store_true", help="include decoded instructions with --program-metrics")
  args = parser.parse_args()

  if args.disable_custom_quant:
    from tinygrad.llm.kernels.amd import Linear
    Linear.use_custom_quant = False

  st = time.perf_counter()
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)

  profile_marker("warmup start")
  st = time.perf_counter()
  model.warmup()
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)
  profile_marker("warmup end")

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen = model.generate(prompt, chunk_size=args.chunk_size)
  profile_marker("prefill start")
  st = time.perf_counter()
  # first token is time-to-first-token; counted as part of prefill
  output = [next(gen)]
  pt = time.perf_counter()
  print(f"prefill {args.prompt_tokens/(pt-st):.3f} tok/s", flush=True)
  profile_marker("prefill end")

  profile_marker("decode start")
  for _ in range(args.decode_tokens): output.append(next(gen))
  et = time.perf_counter()
  print(f"decode {args.decode_tokens/(et-pt):.3f} tok/s output {output}", flush=True)
  profile_marker("decode end")

  if args.program_metrics:
    from bench_q6k_gemv import program_metrics
    for metrics in program_metrics(model.prefill_jit, dump_machine=args.dump_machine):
      if args.program_metrics in str(metrics["name"]): print(json.dumps(metrics, sort_keys=True, default=str))
