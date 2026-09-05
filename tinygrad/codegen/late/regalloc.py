import itertools
from bisect import bisect_left
from tinygrad.device import CompileError
from tinygrad.helpers import dedup, getenv
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.renderer.isa import ISARenderer, Register, greg
from tinygrad.dtype import dtypes, AddrSpace

PSEUDO_OPS = {Ops.CONST, Ops.CAST, Ops.BITCAST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}

class LinearScanRegallocContext:
  # returns the uop that defines the virtual register
  def vdef(self, v:Register) -> UOp: return self.uops[self.live_range[v][0]]
  def __init__(self, uops:list[UOp], ren:ISARenderer):
    self.uops = uops
    self.ren = ren
    self.wide = ren.wide_regalloc
    self.idx = itertools.count()
    self.regalloc_i = 0
    self.reg_promotable: set[UOp] = set()
    if ren.pre_regalloc_matcher is not None:
      from tinygrad.renderer.isa import PreRegAllocContext
      from tinygrad.renderer.isa.rdna3 import _reg_promotable_buffers
      self.reg_promotable = _reg_promotable_buffers(PreRegAllocContext(uops))
    # the label associated with each loop NOTE: this is only used post regalloc and should be removed
    self.loop_label: dict[UOp, str] = {}

    # compute live ranges
    self.live_range: dict[Register, list[int]] = {}
    lr = self.live_range
    ranges: list[Register] = []
    for i,u in enumerate(reversed(uops)):
      if u.op in PSEUDO_OPS: continue
      defs = u.tag if isinstance(u.tag, tuple) else ()
      for v in defs + tuple(greg(s) for s in dedup(u.src)):
        if isinstance(v, Register): lr.setdefault(v, []).insert(0, len(uops) - 1 - i)
      for v in defs:
        if v in lr and (n:=max((lr[rng][-1] for rng in ranges if lr[rng][0] <= lr[v][-1] < lr[rng][-1]), default=None)): lr[v].append(n)
      if u.op is Ops.RANGE: ranges.append(greg(u))

    # allocate registers
    self.stack_size: int = 0
    self.locals: dict[UOp, UOp] = {}
    self.spills: dict[Register, UOp] = {} # mapping from virtual to stack slot
    self.remat: set[Register] = set()
    self.reals: dict[int, dict[Register, Register]] = {} # mapping from virtual to real at each program point
    self.insert_before: dict[int, list[tuple[Register, Register]]] = {} # fills to be inserted at each program point
    if self.wide:
      real_idxs = [i for i,u in enumerate(uops) if u.op not in PSEUDO_OPS and u.op is not Ops.SINK]
      self.first_real_idx, self.last_real_idx = (real_idxs[0], real_idxs[-1]) if real_idxs else (-1, -1)
    live: dict[Register, Register] = {} # mapping from virtual to real that's currently assigned to it
    live_ins: list[dict[Register, Register]] = [] # mapping from virtual to real at loop entry

    slot_counts: dict[Register, int] = {}
    def slots(v:Register) -> int:
      if (ret:=slot_counts.get(v)) is None: slot_counts[v] = ret = ren.register_slots(self.vdef(v), v)
      return ret

    pinned: set[int] = set()  # live source phys regs; defs must not steal (except two-address)

    def alloc(cons:tuple[Register, ...], i:int, v:Register|None=None, *, pin:bool=True) -> Register:
      if self.wide:
        assert v is not None
        return wide_alloc(cons, i, slots(v), v.cons, live, lr, len(uops), slots, pinned if pin else frozenset())
      live_inv = {rv:k for k,rv in live.items()}
      reg,vreg = max(((r,live_inv.get(r)) for r in cons),
                    key=lambda rv: next((j-i for j in ([] if rv[1] is None else lr[rv[1]]) if j >= i), len(uops)))
      return live.pop(vreg) if vreg is not None else reg

    def fill(v:Register, i:int, cons:tuple[Register, ...]|None=None, *, pin:bool=True) -> Register:
      vd = self.vdef(v)
      if ren.rematerialize(vd):
        self.remat.add(v)
        for s in vd.src:
          if s.op is Ops.CONST: continue
          if isinstance(sv:=greg(s), Register):
            if sv not in live: live[sv] = fill(sv, i, pin=pin)
            self.reals.setdefault(i, {})[sv] = live[sv]
            if pin: pinned.update(range(live[sv].index, live[sv].index + slots(sv)))
      elif v not in self.spills:
        sz = ren.spill_size(vd, v)
        if sz <= 0: sz = 4  # void/empty cons under ACC promote must not ZeroDivisionError
        offset = self.stack_size + (sz - self.stack_size % sz) % sz
        self.spills[v] = UOp.cconst(offset, dtypes.int32)
        self.stack_size = offset + sz
      r = alloc(cons if cons is not None else v.cons, i, v, pin=pin)
      self.insert_before.setdefault(i, []).append((v, r))
      return r

    for i,u in enumerate(uops):
      if u.op in PSEUDO_OPS: continue
      loop_end = ren.loop_end(u)
      pinned = set()
      for s in u.src:
        if loop_end is not None: continue
        if not isinstance(v:=greg(s), Register): continue
        # Remat usually rebuilds at every use; keep_remat ops reuse the phys reg.
        if v in self.remat and not ren.keep_remat(self.vdef(v)): live.pop(v, None)
        if v not in live: live[v] = fill(v, i)
        self.reals.setdefault(i, {})[v] = live[v]
        pinned.update(range(live[v].index, live[v].index + slots(v)))

      if isinstance(u.tag, tuple):
        for j,v in enumerate(u.tag):
          # Two-address WMMA/FMAC may redefine the same ACC pack tag across unrolled
          # tiles (flash ACC_SMALL). Live range starts at the first def; later defs
          # reuse the already-allocated phys reg instead of asserting.
          if not isinstance(v, Register): raise AssertionError(f"expected Register tag {v}")
          if lr[v][0] != i:
            if ren.is_two_address(u) and j == 0 and v in live and getenv("AMD_WMMA_REDEF_ACC", 1):
              self.reals.setdefault(i, {})[v] = live[v]
              continue
            assert lr[v][0] == i
          cons = v.cons
          if ren.is_two_address(u) and j == 0:
            uses = tuple(live.get(greg(s)) for s in u.src)
            if self.wide and uses[0] is not None and uses[0] in cons:
              live[v] = uses[0]
              self.reals.setdefault(i, {})[v] = uses[0]
              continue
            cons = ((uses[0],) if uses[0] is not None and uses[0] in cons else ()) + tuple(r for r in cons if r not in uses)
          elif j == 0 and (pref:=ren.prefer_phys(u, [live.get(greg(s)) for s in u.src])) is not None and pref in cons:
            # Alias onto a src sub-register (e.g. EXTRACT → WMMA pack+lane) — skip pinned check.
            live[v] = pref
            self.reals.setdefault(i, {})[v] = pref
            continue
          if pinned:
            filtered = tuple(r for r in cons if not (set(range(r.index, r.index + slots(v))) & pinned))
            if filtered: cons = filtered
            elif len(cons) > 1:
              raise CompileError(f"no unpinned regs for {v}")
            # len==1: dest constrained to one phys (may alias a pinned src) — allow
          live[v] = alloc(cons, i+1 if u.op is not Ops.RANGE else i, v)
          self.reals.setdefault(i, {})[v] = live[v]

      for rv in [rv for rv in live if rv in self.remat and not ren.keep_remat(self.vdef(rv))]: live.pop(rv, None)

      if u.op is Ops.BUFFER:
        if u.addrspace is AddrSpace.REG and u in self.reg_promotable: continue
        self.locals[u] = UOp.cconst(self.stack_size, dtypes.int32)
        self.stack_size += u.max_numel() * u.dtype.itemsize

      if u.op is Ops.RANGE:
        used_in_loop = [v for v in live.keys() | self.spills.keys() if any(i <= l < lr[greg(u)][-1] for l in lr[v])]
        sorted_uses = sorted(used_in_loop, key=lambda k: (next(l-i for l in lr[k] if l >= i), lr[k][0], k.name, k.index))
        live_in: dict[Register, Register] = {}
        for v in sorted_uses:
          if set(v.cons).issubset(live_in.values()): continue
          if v not in live: live[v] = fill(v, i)
          live_in[v] = live[v]
        live_ins.append(live_in)

      if loop_end is not None:
        # loop-carried restores need exact phys regs
        for v,r in live_ins.pop().items():
          if v not in live or live[v] != r: live[v] = fill(v, i, (r,), pin=False)

def wide_alloc(cons, i, nslots, allowed, live, lr, uops_len, slots_fn, pinned=frozenset()):
  # Candidates can describe an aligned multi-slot physical class (for example an
  # even 64-bit SGPR pair). Include covered sub-registers when validating width,
  # while still using only the listed candidates as legal starting positions.
  allowed_idxs = {idx for r in allowed for idx in range(r.index, r.index + max(1, r.size // 4))}
  occupied: dict[int, list[Register]] = {}
  for vr, r in live.items():
    for idx in range(r.index, r.index + slots_fn(vr)): occupied.setdefault(idx, []).append(vr)
  def blockers(reg):
    return tuple(dict.fromkeys(vr for idx in range(reg.index, reg.index+nslots) for vr in occupied.get(idx, ())))
  def next_use(vr):
    uses = lr[vr]
    pos = bisect_left(uses, i)
    return uses[pos] - i if pos < len(uses) else uops_len
  def get_candidates(check_pinned:bool):
    candidates = []
    for r in cons:
      indices = range(r.index, r.index+nslots)
      if not all(x in allowed_idxs for x in indices) or (check_pinned and any(x in pinned for x in indices)): continue
      blocked = blockers(r)
      score = min((next_use(vr) for vr in blocked), default=uops_len)
      # uops_len is the largest possible score. Returning its first occurrence is exactly
      # max(candidates, key=score), including Python's first-wins tie behavior, without
      # scanning the remaining physical windows or recomputing every candidate's score.
      if score == uops_len: return (r, blocked), []
      candidates.append((r, blocked, score))
    return None, candidates
  free, candidates = get_candidates(True)
  if free is not None:
    reg, vregs = free
    for vr in vregs: live.pop(vr, None)
    return reg
  if not candidates:  # no unpinned window (wide nslots / fill) — steal via blockers
    free, candidates = get_candidates(False)
    if free is not None:
      reg, vregs = free
      for vr in vregs: live.pop(vr, None)
      return reg
  if not candidates: raise CompileError(f"wide_alloc: no free regs ({nslots} slots)")
  reg,vregs,_ = max(candidates, key=lambda rv: rv[2])
  for vr in vregs: live.pop(vr, None)
  return reg

def wide_restore(ctx, v, r, i):
  if v in ctx.remat:
    vd, src_regs = ctx.vdef(v), []  # type: ignore[var-annotated]
    for su in vd.src:
      if su.op is Ops.CONST or not isinstance(sv:=greg(su), Register): src_regs.append(None)
      else: src_regs.append(ctx.reals[i][sv])
    return ctx.ren.remat(vd, r, src_regs)
  return ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r)

def wide_regalloc_rewrite(ctx, x:UOp):
  i = ctx.regalloc_i
  if x.op in {Ops.CONST, Ops.NOOP, Ops.AFTER, Ops.BARRIER, Ops.GROUP, Ops.STACK}: return None
  if x.op in (Ops.LOAD, Ops.STORE) and not ctx.insert_before.get(i):
    spilled = any(i in ctx.reals and ((vr:=greg(ctx.uops[i].src[j])) in ctx.spills or vr in ctx.remat)
                  for j in range(len(x.src)))
    if not spilled and i not in (ctx.first_real_idx, ctx.last_real_idx): return None
  nsrc = []
  loop_end = ctx.ren.loop_end(ctx.uops[i])
  for j,su in enumerate(x.src):
    if loop_end is not None:
      nsrc.append(su)
      continue
    vr = greg(ctx.uops[i].src[j]) if i in ctx.reals else None
    if isinstance(vr, Register) and vr in ctx.remat:
      # Actual remat runs in `before` via insert_before; source is just a bind to that phys.
      # Preserve the use dtype: no-op BITCASTs can share a vreg with a differently typed definition.
      nsrc.append(ctx.ren.bind(su.dtype, ctx.reals[i][vr]))
    elif isinstance(vr, Register) and vr in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[vr], su, ctx.reals[i][vr]))
    else: nsrc.append(su)
  ndefs = tuple(ctx.reals[i][vr] for vr in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.BUFFER:
    if x in ctx.reg_promotable and x not in ctx.locals: nx = ctx.ren.isel_matcher.rewrite(x.replace(tag=ndefs))
    else: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)
  before = [wide_restore(ctx, vr, r, i) for vr,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[vr], nx) for vr in x.tag if vr in ctx.spills] if isinstance(x.tag, tuple) else []
  if ctx.stack_size > 0:
    sp, offset = ctx.ren.stack_pointer(), UOp.cconst(ctx.stack_size, ctx.ren.stack_pointer().dtype)
    if i == ctx.first_real_idx: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, src=(sp, offset), tag=sp.tag))] + before
    elif i == ctx.last_real_idx: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, src=(sp, offset), tag=sp.tag))]
  return nx, before + [nx] + after

def regalloc_rewrite(ctx:LinearScanRegallocContext, x:UOp):
  if ctx.wide:
    return wide_regalloc_rewrite(ctx, x)
  if x.op in (Ops.LOAD, Ops.STORE, Ops.SHRINK): return None
  i = next(ctx.idx)
  if x.op in PSEUDO_OPS: return None

  nsrc = []
  for j,s in enumerate(x.src):
    if i in ctx.reals and (v:=greg(ctx.uops[i].src[j])) in ctx.spills: nsrc.append(ctx.ren.fill(ctx.spills[v], ctx.vdef(v), ctx.reals[i][v]))
    else: nsrc.append(s)
  ndefs = tuple(ctx.reals[i][v] for v in x.tag) if isinstance(x.tag, tuple) else x.tag
  if x.op is Ops.BUFFER:
    if x in ctx.reg_promotable and x not in ctx.locals: nx = ctx.ren.isel_matcher.rewrite(x.replace(tag=ndefs))
    else: nx = ctx.ren.isel_matcher.rewrite(ctx.ren.stack_pointer().index(ctx.locals[x], tag=ndefs))
  else: nx = x.replace(src=tuple(nsrc), tag=ndefs)

  before = [ctx.ren.fill(ctx.spills[v], ctx.vdef(v), r) for v,r in ctx.insert_before.get(i, [])]
  after = [ctx.ren.spill(ctx.spills[v], nx) for v in x.tag if v in ctx.spills] if isinstance(x.tag, tuple) else []

  if ctx.stack_size > 0:
    sp = ctx.ren.stack_pointer()
    offset = UOp.cconst(ctx.stack_size, sp.dtype)
    if i == 0: before = [ctx.ren.isel_matcher.rewrite(UOp(Ops.SUB, src=(sp, offset), tag=sp.tag))] + before
    elif i == len(ctx.uops) - 2: before += [ctx.ren.isel_matcher.rewrite(UOp(Ops.ADD, src=(sp, offset), tag=sp.tag))]

  return nx, before + [nx] + after

pm_regalloc_rewrite = PatternMatcher([
  (UPat({Ops.INS, Ops.RANGE, Ops.END, Ops.BUFFER, Ops.PARAM, Ops.SPECIAL, Ops.SHRINK, Ops.LOAD, Ops.STORE} | PSEUDO_OPS, name="x"),
        regalloc_rewrite),
])
