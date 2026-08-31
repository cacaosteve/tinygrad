from __future__ import annotations

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import getenv, prod
from tinygrad.schedule.rangeify import BufferizeOpts
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat
from tinygrad.renderer.isa.rdna3_defs import allow_upcast16

def _unwrap_const(x:UOp) -> UOp|None:
  while x.op in (Ops.CAST, Ops.BITCAST, Ops.NOOP) and len(x.src) == 1: x = x.src[0]
  return x if x.op is Ops.CONST else None

def _const_value(x:UOp):
  return c.val if (c:=_unwrap_const(x)) is not None else None

def _tconst(value, dtype:DType, tag=None) -> UOp:
  return UOp.cconst(value, dtype).rtag(tag)

# ***** TC_LDS_AB staging (codegen hooks via AMDRenderer.pm_stage_wmma_ab) *****
_WMMA_LDS_AXES, _WMMA_LDS_LOOP_BASE, _WMMA_TC = (AxisType.LOCAL, AxisType.WARP), 200, 16

def _range_size(r:UOp) -> int:
  return int(n) if (n:=_const_value(r.src[0])) is not None else int(r.vmax) + 1

def _linearize_ranges(axes:list[UOp]) -> UOp:
  out = axes[0]
  for a in axes[1:]: out = out * _range_size(a) + a
  return out

def _tid_axes(coop:list[UOp]) -> list[UOp]|None:
  # LOCALs by range id, WARP last in the product. gpudims maps WARP→lidx0 so
  # tid == hardware linear id when local_size is (32, …).
  locals_ = sorted([u for u in coop if u.arg[1] is AxisType.LOCAL], key=lambda u: u.arg[0])
  warps = [u for u in coop if u.arg[1] is AxisType.WARP]
  if len(locals_) < 2 or len(warps) != 1: return None
  if _range_size(warps[0]) != 32: return None
  return locals_ + warps

def _index_row_stride(idx:UOp) -> int|None:
  e = idx.src[1] if idx.op is Ops.INDEX else idx
  if e.op is not Ops.ADD: return None
  for side in e.src:
    if side.op is Ops.MUL:
      for t in side.src:
        if (n:=_const_value(t)) is not None and int(n) > 1: return int(n)
  return None

def _delinearize_ranges(linear:UOp, axes:list[UOp]) -> dict[UOp, UOp]:
  """Map a flat index onto axes (last axis fastest)."""
  subs, rem = {}, linear
  for a in reversed(axes):
    sz = _range_size(a)
    subs[a] = rem % sz
    rem = rem // sz
  return subs

def _bounce_a_shared(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                     as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Shared A via tid bufferize: LDS[tid,k]=A[g*block+tid,k], read [major,k].
  # STACK(8)×chunk → GLOBAL B128 (scalar ept is U16 and loses to frag-wide on INS).
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  buf, stride = ab.src[0], _index_row_stride(ab)
  if stride is None: return None
  reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
  grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
  if len(reds) != 1 or len(grids) != 1: return None
  k_tile, g_wg = reds[0], grids[0]
  op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
  if op_local is None: return None
  tile_prod = prod(_range_size(t) for t in tile) if tile else 1
  block = _range_size(op_local) * tile_prod * _WMMA_TC
  fsz = prod(_range_size(f) for f in frag)
  if block != threads or fsz != _WMMA_TC or fsz % 8: return None
  vec = 8
  chunk = UOp.range(fsz // vec, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
  elems = [buf.index((g_wg * block + tid) * stride + (k_tile * _WMMA_TC + chunk * vec + j)) for j in range(vec)]
  staged = UOp.stack(*elems).bufferize(*tid_axes, chunk, arg=BufferizeOpts(None, AddrSpace.LOCAL))
  # Flat 1D like B: major*16+k peels to base+imm for K-contig B128 LLOADs.
  flat = staged.reshape(threads * fsz)
  major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
  k_r = _linearize_ranges(as_up_frag)
  read = flat.index(major * fsz + k_r)
  return read.contract(*as_up) if as_up else read

def _bounce_frag_wide(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                      as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Identity fill along frag (unit-stride operand, typically A). WEAK(fsz/8)×STACK(8).
  # Drop N-LOCAL from bufferize when A only uses one LOCAL (M): threads that differ only in
  # N-local write the same cells (A independent of ln) → ~2× smaller A LDS.
  fsz, vec = prod(_range_size(f) for f in frag), 8
  if not frag or fsz % vec: return None
  tile_w = [r.replace(arg=(r.arg[0] + _WMMA_LDS_LOOP_BASE + i * 50 + n, AxisType.WEAK)) for n, r in enumerate(tile)]
  chunk = UOp.range(fsz // vec, _WMMA_LDS_LOOP_BASE + i * 50 + 40, AxisType.WEAK)
  elems = []
  for j in range(vec):
    sub = dict(zip(tile, tile_w))
    sub.update(_delinearize_ranges(chunk * vec + j, frag))
    elems.append(ab.substitute(sub))
  ab_locals = [u for u in coop if u.arg[1] is AxisType.LOCAL and any(x is u for x in ab.toposort())]
  write_coop = [u for u in coop if u.arg[1] is not AxisType.LOCAL or u in ab_locals]
  staged = UOp.stack(*elems).bufferize(*write_coop, *tile_w, chunk, arg=BufferizeOpts(None, AddrSpace.LOCAL))
  flat = staged.reshape(*[_range_size(x) for x in write_coop + tile_w], fsz)
  frag_lin = _linearize_ranges(as_up_frag) if len(as_up_frag) > 1 else as_up_frag[0]
  indexed = flat.index(*write_coop, *as_up_tile, frag_lin)
  return indexed.contract(*as_up) if as_up else indexed

def _bounce_tid_wide(ab:UOp, i:int, coop:list[UOp], frag:list[UOp], tile:list[UOp],
                     as_up_tile:list[UOp], as_up_frag:list[UOp], as_up:list[UOp]) -> UOp|None:
  # Tid-partitioned fill for strided B: STACK(8) GLOBAL B128 + scatter DS_STORE to (n,k) LDS.
  # Flat LDS + shared base+imm offsets → one addr VGPR (isel _peel_add_imm); K-contig reads B128.
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  if threads < 32 or threads % 32: return None
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  buf, stride = ab.src[0], _index_row_stride(ab)
  if stride is None: return None
  reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
  # WEAK = former LOOP (#17283). GLOBAL covers workgroup tiles after TC.
  grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
  if len(reds) != 1 or len(grids) != 1: return None
  k_tile, g_wg = reds[0], grids[0]
  op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
  if op_local is None: return None
  tile_prod = prod(_range_size(t) for t in tile) if tile else 1
  block = _range_size(op_local) * tile_prod * _WMMA_TC
  vec = 8
  ept_n = (_WMMA_TC * block) // threads
  if ept_n < vec or ept_n % vec or (_WMMA_TC * block) != threads * ept_n: return None
  t_per_k = block // ept_n
  if t_per_k < 1 or block != t_per_k * ept_n or threads != _WMMA_TC * t_per_k: return None
  k, n_base = tid // t_per_k, (tid % t_per_k) * ept_n
  # Flat (n,k) row-major: addr = n*16+k. One base + j*16 peels to ds_store offset.
  local = UOp.placeholder((block * _WMMA_TC,), ab.dtype, slot=100 + i, addrspace=AddrSpace.LOCAL)
  elems, stores = [], []
  # ept_n==vec (default 2×2): no chunk range — keeps addr math short-lived (avoids spills).
  if ept_n == vec:
    base = n_base * _WMMA_TC + k
    for j in range(vec):
      elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n_base + j)))
      stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
    flat = local.after(UOp.group(*stores).end(*tid_axes))
  else:
    chunk = UOp.range(ept_n // vec, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
    base = (n_base + chunk * vec) * _WMMA_TC + k
    for j in range(vec):
      n = n_base + chunk * vec + j
      elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n)))
      stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
    flat = local.after(UOp.group(*stores).end(*tid_axes, chunk))
  major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
  k_r = _linearize_ranges(as_up_frag)
  read = flat.index(major * _WMMA_TC + k_r)
  return read.contract(*as_up) if as_up else read

def stage_wmma_ab_bounce(wmma:UOp, coop:list[UOp]) -> UOp|None:
  # Hybrid bounce: shared A (tid-fill, major-read) when block==threads; else frag-wide A; tid-wide B.
  # expand_wmma slices STACK(16*tile,). Tile product ≤8 (LLOAD/PACK VGPR pools).
  news: list[UOp] = []
  changed = False
  for i, ab in enumerate(wmma.src[:2]):
    if any(u.op is Ops.STAGE for u in ab.toposort()):
      news.append(ab)
      continue
    # Both A and B must be GLOBAL INDEX to stage. Computed operands (eye/WHERE) stay
    # unstaged; mixing with a staged peer breaks expand_broadcast (eye@B IndexError).
    if ab.op is not Ops.INDEX or ab.addrspace != AddrSpace.GLOBAL: return None
    if not any(u in coop for u in ab.toposort() if u.op is Ops.RANGE): return None
    frag_rns = {rn for rn, _ in wmma.arg[4][i]}
    frag = list(dict.fromkeys(u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[0] in frag_rns))
    tile = list(dict.fromkeys(u for u in ab.toposort()
      if u.op is Ops.RANGE and u.arg[1] in (AxisType.UPCAST, AxisType.UNROLL) and u.arg[0] not in frag_rns))
    as_up_tile = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in tile]
    as_up_frag = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in frag]
    as_up = as_up_tile + as_up_frag
    if i == 0 and (ret := _bounce_a_shared(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    elif i == 1 and (ret := _bounce_tid_wide(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    elif (ret := _bounce_frag_wide(ab, i, coop, frag, tile, as_up_tile, as_up_frag, as_up)) is not None:
      news.append(ret)
    else:
      read_axes = tile + frag
      write_axes = [r.replace(arg=(r.arg[0] + _WMMA_LDS_LOOP_BASE + i * 50 + (25 if n < len(tile) else 0), AxisType.WEAK))
                    for n, r in enumerate(read_axes)]
      sval = ab.substitute(dict(zip(read_axes, write_axes))) if write_axes else ab
      staged = sval.bufferize(*coop, *write_axes, arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*coop, *as_up)
      news.append(staged.contract(*as_up) if as_up else staged)
    changed = True
  if not changed: return None
  _in0, _in1, out0 = wmma.arg[4]
  return wmma.replace(src=(news[0], news[1], wmma.src[2]), arg=(*wmma.arg[:4], ((), (), out0)))

def stage_wmma_ab_tid(wmma:UOp, coop:list[UOp]) -> UOp|None:
  if (tid_axes := _tid_axes(coop)) is None: return None
  warp = tid_axes[-1]
  threads = prod(_range_size(a) for a in tid_axes)
  if threads < 32 or threads % 32: return None
  tid, lane16 = _linearize_ranges(tid_axes), warp % 16
  news: list[UOp] = []
  changed = False
  for i, ab in enumerate(wmma.src[:2]):
    if any(u.op is Ops.STAGE for u in ab.toposort()):
      news.append(ab)
      continue
    if ab.op is not Ops.INDEX or ab.addrspace != AddrSpace.GLOBAL: return None
    if not any(u in coop for u in ab.toposort() if u.op is Ops.RANGE): return None
    buf, stride = ab.src[0], _index_row_stride(ab)
    if stride is None: return None
    frag_rns = {rn for rn, _ in wmma.arg[4][i]}
    frag = list(dict.fromkeys(u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[0] in frag_rns))
    tile = list(dict.fromkeys(u for u in ab.toposort()
      if u.op is Ops.RANGE and u.arg[1] in (AxisType.UPCAST, AxisType.UNROLL) and u.arg[0] not in frag_rns))
    if not frag or prod(_range_size(f) for f in frag) != _WMMA_TC: return None
    reds = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] is AxisType.REDUCE]
    grids = [u for u in ab.toposort() if u.op is Ops.RANGE and u.arg[1] in (AxisType.WEAK, AxisType.GLOBAL)]
    if len(reds) != 1 or len(grids) != 1: return None
    k_tile, g_wg = reds[0], grids[0]
    op_local = next((u for u in tid_axes[:-1] if any(x is u for x in ab.toposort())), None)
    if op_local is None: return None
    tile_prod = prod(_range_size(t) for t in tile) if tile else 1
    block = _range_size(op_local) * tile_prod * _WMMA_TC
    ept_n = (block * _WMMA_TC) // threads if i == 0 else (_WMMA_TC * block) // threads
    if ept_n < 1 or (block * _WMMA_TC if i == 0 else _WMMA_TC * block) != threads * ept_n: return None
    ept = UOp.range(ept_n, _WMMA_LDS_LOOP_BASE + i * 50, AxisType.WEAK)
    as_up_tile = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in tile]
    as_up_frag = [r.replace(arg=(r.arg[0], AxisType.UPCAST)) if r.arg[1] is not AxisType.UPCAST else r for r in frag]
    as_up = as_up_tile + as_up_frag
    major = _linearize_ranges(as_up_tile + [op_local]) * _WMMA_TC + lane16 if as_up_tile else op_local * _WMMA_TC + lane16
    k_r = _linearize_ranges(as_up_frag)
    if i == 0:
      if block != threads: return None
      gval = buf.index((g_wg * block + tid) * stride + (k_tile * _WMMA_TC + ept))
      staged = gval.bufferize(*tid_axes, ept, arg=BufferizeOpts(None, AddrSpace.LOCAL))
      read = staged.reshape(threads, ept_n).index(major, k_r)
    else:
      # B transpose: STACK(8) GLOBAL B128 + flat LDS base+imm scatter (same as bounce).
      t_per_k = block // ept_n
      if ept_n % 8 or t_per_k < 1 or block != t_per_k * ept_n or threads != _WMMA_TC * t_per_k: return None
      vec = 8
      k, n_base = tid // t_per_k, (tid % t_per_k) * ept_n
      local = UOp.placeholder((block * _WMMA_TC,), ab.dtype, slot=100 + i, addrspace=AddrSpace.LOCAL)
      elems, stores = [], []
      if ept_n == vec:
        base = n_base * _WMMA_TC + k
        for j in range(vec):
          elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n_base + j)))
          stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
        read = local.after(UOp.group(*stores).end(*tid_axes)).index(major * _WMMA_TC + k_r)
      else:
        chunk = UOp.range(ept_n // vec, _WMMA_LDS_LOOP_BASE + i * 50 + 1, AxisType.WEAK)
        base = (n_base + chunk * vec) * _WMMA_TC + k
        for j in range(vec):
          n = n_base + chunk * vec + j
          elems.append(buf.index((k_tile * _WMMA_TC + k) * stride + (g_wg * block + n)))
          stores.append(local.index(base + j * _WMMA_TC).store(elems[j]))
        read = local.after(UOp.group(*stores).end(*tid_axes, chunk)).index(major * _WMMA_TC + k_r)
    news.append(read.contract(*as_up) if as_up else read)
    changed = True
  if not changed: return None
  _in0, _in1, out0 = wmma.arg[4]
  return wmma.replace(src=(news[0], news[1], wmma.src[2]), arg=(*wmma.arg[:4], ((), (), out0)))

def stage_wmma_ab_to_local(wmma:UOp) -> UOp|None:
  if wmma.op is not Ops.WMMA: return None
  coop = list(dict.fromkeys(u for u in wmma.toposort() if u.op is Ops.RANGE and u.arg[1] in _WMMA_LDS_AXES))
  if not any(u.arg[1] == AxisType.LOCAL for u in coop): return None
  if getenv("TC_LDS_TID", 0):
    if (ret := stage_wmma_ab_tid(wmma, coop)) is not None: return ret
  return stage_wmma_ab_bounce(wmma, coop)

pm_stage_wmma_ab = PatternMatcher([(UPat(Ops.WMMA, name="wmma"), stage_wmma_ab_to_local)])

_WMMA_AB_WIDTH = 16
# Serialize A-tile batches so earlier LDS A packs die before later ones load (VGPR pressure).
# Product-8 is OK with batch 2 + disjoint LLOAD/PACK pools. Product-16 under LDS still spills /
# mis-lives without stronger live-range constraints than AFTER provides.

def expand_wmma_lds_tiles(u, a, b, c, done_arg, unroll_axis, ctx):
  # Shared AMDRenderer / AMDLLVMRenderer hook for TC_LDS_AB WMMA expansion: pre-contracted
  # STACK(16*tile,) is sliced per tile here. Staging only runs when the renderer installs
  # pm_stage_wmma_ab and TC_LDS_AB is set; import of this module wires the codegen hook.
  # Serialize A-tile batches with AFTER so earlier LDS A packs die before later ones load.
  if a.op is not Ops.STACK or len(a.src) <= _WMMA_AB_WIDTH or len(a.src) % _WMMA_AB_WIDTH != 0: return None
  ta, tb = len(a.src) // _WMMA_AB_WIDTH, (len(b.src) // _WMMA_AB_WIDTH) if b.op is Ops.STACK else 1
  # Soft-fail past expand budget (8 default; 16 with ALLOW_UPCAST16 / register default).
  max_prod = 16 if allow_upcast16() else 8
  if ta * tb > max_prod: return None
  a_batch = getenv("TC_LDS_A_BATCH", 2)
  c_stk = c if c.op is Ops.STACK else UOp.stack(*[c.index(_tconst(i, dtypes.weakint)) for i in range(c.max_numel())])
  wmmas: list[UOp] = []
  prev_batch: UOp|None = None
  for i0 in range(0, ta, a_batch):
    batch: list[UOp] = []
    for i in range(i0, min(i0 + a_batch, ta)):
      aa_elems = a.src[i*_WMMA_AB_WIDTH:(i+1)*_WMMA_AB_WIDTH]
      if prev_batch is not None:
        aa_elems = tuple(UOp(Ops.AFTER, src=(e, prev_batch)) for e in aa_elems)
      aa = UOp.stack(*aa_elems)
      for j in range(tb):
        bb = UOp.stack(*b.src[j*_WMMA_AB_WIDTH:(j+1)*_WMMA_AB_WIDTH]) if b.op is Ops.STACK else b
        batch.append(u.replace(src=(aa, bb, c_stk), arg=done_arg))
    wmmas.extend(batch)
    prev_batch = UOp.stack(*batch)
  return unroll_axis(ctx, UOp.stack(*wmmas).reshape(ta, tb, c_stk.max_numel()), u.arg[4][2])
