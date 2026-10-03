#!/usr/bin/env python3
"""Decode fair+asm A/B for AMD_FMA_MIX_EVERY (leave some ACC as FMAC for dual_fmac+ADD)."""
from __future__ import annotations
import json, os, subprocess, sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r'''
import json, os, statistics, time
from collections import Counter
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import DEV, getenv, Context, Target
from tinygrad.llm.kernels.amd import flash_attention, _amd_flash_attention_decode_partial, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.uop.ops import UOp
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size

getenv.cache_clear(); amd._amd_flash_attention.cache_clear()
amd._amd_flash_attention_decode_partial.cache_clear(); to_program_cache.clear()
DEV.value = os.environ["DEV"]
for k, v in json.loads(os.environ.get("EXTRA", "{}")).items():
  if v is None: os.environ.pop(k, None)
  else: os.environ[k] = str(v)

ren = AMDRenderer(Target("AMD", arch="gfx1100"))
B, H, Hk, N, D, waves, chunk = 1, 32, 8, 2048, 128, 8, 64
PARTIALS = min(48, N // 64)
with _flash_direct_compile_env(flash_acc_small=False), Context(BEAM=0):
  prg = to_program(_amd_flash_attention_decode_partial(
    UOp.placeholder((B, H, PARTIALS, D), dtypes.float32, 0),
    UOp.placeholder((B, H, PARTIALS, 2), dtypes.float32, 1),
    UOp.placeholder((B, H, 1, D), dtypes.float16, 2),
    UOp.placeholder((2, B, Hk, N, D), dtypes.float16, 3),
    N, N, chunk, waves=waves), ren)
text = next(s.content for s in elf_loader(prg.src[-1].arg)[1] if s.name == ".text")
insts, off = [], 0
while off < len(text):
  inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
  if getattr(inst, "op_name", "").lower() == "s_endpgm": break
mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
vgpr, _, _, priv = scan_elf_regs(insts, scratch_inst_size)

def sk(*ss): return sum(v for k, v in mc.items() if any(s in k for s in ss))
def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape)); return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

q = Tensor(patterned((1, 32, 1, 128)), device=Device.DEFAULT).realize()
cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
r = TinyJit(lambda qq: flash_attention(qq, cache, 2048).realize())
o = None
for _ in range(8):
  o = r(q); Device[Device.DEFAULT].synchronize()
a = o.numpy(); xs = []
nrep = int(os.environ.get("NREP", "160"))
for _ in range(nrep):
  t0 = time.perf_counter_ns(); r(q); Device[Device.DEFAULT].synchronize()
  xs.append((time.perf_counter_ns() - t0) / 1e3)
xs = sorted(xs)
print(json.dumps({
  "label": os.environ["LABEL"], "priv": priv, "vgpr": vgpr, "ninst": len(insts),
  "v_mov": mc.get("v_mov_b32_e32", 0),
  "fmac": sk("fmac"), "fma_mix": sk("fma_mix"), "dual_fmac": sk("dual_fmac"),
  "dual": sk("dual"), "cvt": mc.get("v_cvt_f32_f16_e32", 0),
  "delay": mc.get("s_delay_alu", 0),
  "sum": float(a.astype("float64").sum()) if np.isfinite(a).all() else None,
  "finite": bool(np.isfinite(a).all()),
  "median": statistics.median(xs), "best": min(xs), "p10": xs[max(0, nrep // 10 - 1)],
}))
'''

CFGS = [
  ("hip", {}, True),
  ("ship", {}, False),
  ("mix_every2", {"AMD_FMA_MIX_EVERY": "2"}, False),
  ("mix_every4", {"AMD_FMA_MIX_EVERY": "4"}, False),
  ("mix_every8", {"AMD_FMA_MIX_EVERY": "8"}, False),
  ("mix_off", {"AMD_FMA_MIX_ALL": "0", "AMD_FMA_MIX": "0"}, False),
]


def run(label: str, extra: dict, hip: bool) -> None:
  env = os.environ.copy()
  for k in list(env):
    if k.startswith("AMD_") and k != "AMD_SOFT_SYSFS":
      env.pop(k, None)
  env.update({"AMD_SOFT_SYSFS": "1", "PYTHONPATH": ROOT, "PARALLEL": "0", "CACHELEVEL": "0",
              "CCACHE": "0", "LABEL": label, "EXTRA": json.dumps(extra),
              "DEV": "AMD" if hip else "AMD:AMD", "NREP": os.environ.get("NREP", "160")})
  if not hip:
    env["AMD_FLASH_DIRECT"] = "1"
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, timeout=500, cwd=ROOT)
  line = (p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip()
          else "FAIL " + (p.stderr or "")[-350:].replace("\n", " "))
  print("T", line, flush=True)


def main() -> None:
  for label, extra, hip in CFGS:
    run(label, extra, hip)


if __name__ == "__main__":
  main()
