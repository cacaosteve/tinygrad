"""WMMA/load interleave: schedule later groups' loads after prior WMMA (Gabriel-style).

Gated by AMD_WMMA_INTERLEAVE=1. Groups WMMAs by shared A-operand non-REG loads, then
makes each subsequent group's loads AFTER the previous group's WMMA (+ stores).

AFTER is applied to the load address base (BUFFER/PARAM/INDEX root), not wrapping INDEX
itself — wrapping INDEX breaks RDNA3 isel shrink/d16. Flash loads are often LOAD(SHRINK).
"""
from __future__ import annotations

from tinygrad.dtype import AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import PatternMatcher, UPat, UOp

class WMMASchedulePolicy:
  def __init__(self, sink:UOp):
    self.schedule_past: dict[UOp, tuple[UOp, ...]] = {}

    wmma_deps: dict[UOp, list[set[UOp]]] = {}
    wmma_consumers: dict[UOp, list[UOp]] = {}
    for u in sink.toposort():
      if u.op is Ops.WMMA:
        for s in u.src:
          deps = {l for l in s.toposort() if l.op is Ops.LOAD and l.src[0].addrspace is not AddrSpace.REG}
          wmma_deps.setdefault(u, []).append(deps)
      if u.op is Ops.STORE and len(u.src) > 1 and u.src[1].op is Ops.INDEX:
        w = u.src[1].src[0]
        if w.op is Ops.WMMA: wmma_consumers.setdefault(w, []).append(u)
    if len(wmma_deps) < 4: return

    sched_groups: dict[frozenset[UOp], list[UOp]] = {}
    for w in wmma_deps:
      sched_groups.setdefault(frozenset(wmma_deps[w][0]), []).append(w)

    carry: list[UOp] = []
    scheduled: set[UOp] = set()
    for i, g in enumerate(sched_groups.values()):
      deps: set[UOp] = set()
      for w in g:
        for ls in wmma_deps[w]: deps.update(ls)
      if i > 0:
        for l in deps:
          if l not in scheduled: self.schedule_past[l] = tuple(carry)
      scheduled.update(deps)
      carry = list(g)
      for w in g: carry.extend(wmma_consumers.get(w, []))

def _with_base_after(addr:UOp, past:tuple[UOp, ...]) -> UOp:
  """Attach AFTER to the storage base under INDEX/SHRINK/BITCAST views."""
  if addr.op is Ops.INDEX:
    return addr.replace(src=(addr.src[0].after(*past),) + addr.src[1:])
  if addr.op in {Ops.SHRINK, Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.PAD, Ops.BITCAST, Ops.CAST}:
    return addr.replace(src=(_with_base_after(addr.src[0], past),) + addr.src[1:])
  if addr.op is Ops.AFTER:
    # already ordered; append priors
    return addr.after(*past)
  return addr.after(*past)

def _delay_load(ctx:WMMASchedulePolicy, x:UOp) -> UOp|None:
  if x not in ctx.schedule_past: return None
  return x.replace(src=(_with_base_after(x.src[0], ctx.schedule_past[x]),) + x.src[1:])

pm_schedule_interleave_wmma = PatternMatcher([
  (UPat(Ops.LOAD, name="x"), _delay_load),
])
