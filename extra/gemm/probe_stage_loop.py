"""MOCK gate: LOOP-ended STAGE enables peer LLOAD>LSTORE (UPCAST-as-STAGE hangs).

  PYTHONPATH=. python extra/gemm/probe_stage_loop.py
"""
from __future__ import annotations
import signal
from functools import reduce
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.uop import Ops as UOps
import tinygrad.codegen as cg

# Match codegen: only LOOP axis ids >= _WMMA_LDS_LOOP_BASE are ended on STAGE.
assert cg._WMMA_LDS_LOOP_BASE == 200

ren = AMDRenderer(Target("AMD", arch="gfx1100"))

def try_compile(name: str, sink: UOp, timeout: int = 10) -> dict[str, int] | None:
  to_program_cache.clear()
  print(f"--- {name}", flush=True)
  def handler(s, f): raise TimeoutError("hang")
  signal.signal(signal.SIGALRM, handler)
  signal.alarm(timeout)
  try:
    prg = to_program(sink.simplify(), ren)
    signal.alarm(0)
    lin = [u.arg for u in prg.src[1].src if u.op is UOps.INS]
    cnt = {k: lin.count(getattr(AMDOps, k)) for k in ["LOAD", "LLOAD", "LSTORE", "BARRIER"]}
    print("OK", cnt, flush=True)
    return cnt
  except TimeoutError:
    print("HANG", flush=True)
    return None
  except Exception as e:
    signal.alarm(0)
    print("FAIL", type(e).__name__, str(e)[:200], flush=True)
    return None

tid = UOp.special(32, "lidx0")
loop = UOp.range(4, cg._WMMA_LDS_LOOP_BASE + 10, AxisType.LOOP)
buf_g = UOp.placeholder((128,), dtypes.half, 1, AddrSpace.GLOBAL)
out = UOp.placeholder((32,), dtypes.float, 0, AddrSpace.GLOBAL)
peer = (tid + 1) % 32
st = buf_g.reshape(32, 4)[tid, loop].bufferize(tid, loop, arg=BufferizeOpts(None, AddrSpace.LOCAL))
vals = [st.index(peer, i).cast(dtypes.float) for i in range(4)]
sink = out[tid].store(reduce(lambda a, b: a + b, vals)).sink(arg=KernelInfo(name="peer4", opts_to_apply=()))
cnt = try_compile("LOOP-ended STAGE peer4", sink)
if cnt and cnt["LLOAD"] > cnt["LSTORE"] and cnt["LOAD"] <= cnt["LSTORE"]:
  print(f"GATE OK: LLOAD {cnt['LLOAD']} > LSTORE {cnt['LSTORE']} (peer reuse)")
else:
  print(f"GATE PENDING: {cnt}")
  raise SystemExit(1)
