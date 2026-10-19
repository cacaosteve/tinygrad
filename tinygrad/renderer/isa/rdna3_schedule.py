"""RDNA3 pre-regalloc schedule helpers (FMAC↔ADD adjacency, swizzle batches)."""
from __future__ import annotations

import itertools

from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.isa import Register, greg
from tinygrad.renderer.isa.rdna3_defs import (AMDOps, VGPR, iop as _iop, reg_idxs as _reg_idxs, elem_count as _elem_count,
  pack_f16_is_vec_load as _pack_f16_is_vec_load, pack_f16_half2_load as _pack_f16_half2_load,
  pack_f16_is_half2_lo_only as _pack_f16_is_half2_lo_only, reg_slots as _reg_slots, const_int as _const_int,
  lds_byte_off as _lds_byte_off, load_count_src as _load_count_src, tconst as _tconst)
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp

def schedule_fmac_add_pairs(ops:list[UOp]) -> list[UOp]:
  """Pull an independent ADD/SUB next to each FMAC for mixed VOPD.

  HIP flash dual_fmac is FMAC+ADD with independent dests; our emit scan only pairs when
  the ADD does not depend on intervening ops. Without this hoist, ADDs sit too far from
  FMACs (behind MAX/MUL chains) and dual_fmac stays 0.

  Flash fair often places all float ADDs *before* the FMAC block (softmax x-m then later
  acc FMAC). Search backward as well and sink an independent ADD to immediately after
  the FMAC so emit-time VOPD can see FMAC,ADD.

  AMD_VOPD_FMAC_MUL removed (hang).
  """
  # Decode ship: with VOPD_FMAC_ALU (AMD_SCHEDULE_FMAC_ADD folded).
  if not getenv("AMD_VOPD_FMAC_ALU", 0): return ops
  max_scan = 64
  alu_ok = (AMDOps.ADD, AMDOps.SUB)
  out = list(ops)
  topo_cache: dict[UOp, dict[UOp, None]] = {}
  def topo(u:UOp) -> dict[UOp, None]:
    if u not in topo_cache: topo_cache[u] = u.toposort()
    return topo_cache[u]
  i = 0
  while i < len(out):
    u = out[i]
    if u.op is Ops.INS and _iop(u) is AMDOps.FMAC and u.dtype is dtypes.float32:
      placed = False
      for j in range(i + 1, min(i + 1 + max_scan, len(out))):
        cand = out[j]
        if not (cand.op is Ops.INS and _iop(cand) in alu_ok and
                cand.dtype is dtypes.float32):
          continue
        if u in topo(cand) or cand in topo(u): continue
        if any(out[m] in topo(cand) for m in range(i + 1, j)): continue
        if any(cand in topo(out[m]) for m in range(i + 1, j)): continue
        out.pop(j)
        out.insert(i + 1, cand)
        placed = True
        break
      if not placed:
        for j in range(i - 1, max(-1, i - 1 - max_scan), -1):
          cand = out[j]
          if not (cand.op is Ops.INS and _iop(cand) in alu_ok and
                  cand.dtype is dtypes.float32):
            continue
          if u in topo(cand) or cand in topo(u): continue
          if any(cand in topo(out[m]) for m in range(j + 1, i + 1)): continue
          if any(out[m] in topo(cand) for m in range(j + 1, i)): continue
          out.pop(j)
          fi = i - 1
          out.insert(fi + 1, cand)
          placed = True
          break
    i += 1
  return out

def schedule_swizzle_mov_batches(ops:list[UOp]) -> list[UOp]:
  """Rewrite (SWIZZLE|PERMLANEX16),(MOV|ADD|MAX)×… into op×N[, VALU×G],use×N before regalloc.

  Park path: SW,MOV pairs (REG temps). No-park path removed (`AMD_SWIZZLE_NO_PARK`).

  Emit-time reordering of the same pattern extends live ranges past what
  regalloc assumed (aliased VGPRs → wrong results). Scheduling here keeps liveness honest.

  AMD_SWIZZLE_VALU_GAP: pull independent FMAC/MUL/ADD from immediately after the use
  block into the swizzle→use gap so lgkm wait can overlap VALU (HIP flash_decode).
  """
  batchable = (AMDOps.SWIZZLE, AMDOps.PERMLANEX16)
  use_ops = (AMDOps.MOV, AMDOps.ADD, AMDOps.MAX)
  valu_gap = (AMDOps.FMAC, AMDOps.MUL, AMDOps.ADD, AMDOps.MAX, AMDOps.MULACC)
  pull_valu = bool(getenv("AMD_VOPD_FMAC", 0))  # SWIZZLE_VALU_GAP folded (decode ship)
  max_gap = 4
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if u.op is Ops.INS and _iop(u) in batchable and i + 1 < len(ops) and \
       ops[i + 1].op is Ops.INS and _iop(ops[i + 1]) in use_ops and u in ops[i + 1].src:
      kind, use_kind = _iop(u), _iop(ops[i + 1])
      sws, uses = [u], [ops[i + 1]]
      j = i + 2
      while j + 1 < len(ops) and len(sws) < 16 and \
            ops[j].op is Ops.INS and _iop(ops[j]) is kind and \
            ops[j + 1].op is Ops.INS and _iop(ops[j + 1]) is use_kind and \
            ops[j] in ops[j + 1].src:
        if any(prev in ops[j].toposort() for prev in uses): break
        sws.append(ops[j])
        uses.append(ops[j + 1])
        j += 2
      if len(sws) >= 2:
        gap: list[UOp] = []
        if pull_valu:
          blocked = set(sws) | set(uses)
          k = j
          while k < len(ops) and len(gap) < max_gap:
            cand = ops[k]
            if cand.op is Ops.INS and _iop(cand) in valu_gap and \
               not any(s in blocked for s in cand.src) and \
               not any(cand in m.toposort() for m in uses):
              gap.append(cand)
              blocked.add(cand)
              k += 1
              continue
            break
        out.extend(sws)
        out.extend(gap)
        out.extend(uses)
        i = j + len(gap)
        continue
    out.append(u)
    i += 1
  return out

def gap_fill_after_loads(ops:list[UOp]) -> list[UOp]:
  """Insert independent ALU between scratch/global LOAD and its first use (overlap wait).

  Flash tip still does scratch_load; waitcnt_vmcnt(0) with nothing in between (~109×).
  """
  load_ops = (AMDOps.SLOAD, AMDOps.LOAD)
  alu = (AMDOps.FMAC, AMDOps.MUL, AMDOps.ADD, AMDOps.MAX, AMDOps.MOV, AMDOps.MULACC,
         AMDOps.EXP2, AMDOps.WHERE, AMDOps.CAST)
  max_scan = 16
  max_gap = 8
  taken: set[int] = set()
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    if i in taken:
      i += 1
      continue
    u = ops[i]
    if u.op is Ops.INS and _iop(u) in load_ops:
      use_j: int|None = None
      for j in range(i + 1, min(i + 1 + max_scan, len(ops))):
        if j in taken: continue
        if u in ops[j].src:
          use_j = j
          break
      if use_j is not None and use_j == i + 1:
        gap: list[int] = []
        blocked = {u, ops[use_j]}
        k = use_j + 1
        while k < len(ops) and len(gap) < max_gap:
          if k in taken:
            k += 1
            continue
          cand = ops[k]
          if cand.op is Ops.INS and _iop(cand) in alu and \
             not any(s in blocked for s in cand.src) and \
             u not in cand.src and ops[use_j] not in cand.src:
            gap.append(k)
            blocked.add(cand)
            k += 1
            continue
          break
        if gap:
          out.append(u)
          for gi in gap:
            out.append(ops[gi])
            taken.add(gi)
          out.append(ops[use_j])
          taken.add(use_j)
          i += 1
          continue
    out.append(u)
    i += 1
  return out

# Addr / pack ops that can issue while a prior WMMA's inputs stay live.
_SINKABLE_PAST_WMMA = frozenset({
  AMDOps.LOAD, AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV,
  # ADD excluded: sinking past loop S_ADD reorders the trip counter vs the last WMMA.
  AMDOps.SUB, AMDOps.MUL, AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16, AMDOps.SHL, AMDOps.SHR,
  AMDOps.AND, AMDOps.OR, AMDOps.XOR,
  AMDOps.BFE,
  AMDOps.CVT_UBYTE_F32,
  AMDOps.LSHL_OR, AMDOps.LSHL_ADD,
})

def sink_wmma_past_loads(ops:list[UOp]) -> list[UOp]:
  # Sink WMMA (+ its ACC EXTRACTs) past independent loads — not past peer WMMAs.
  # AMD_WMMA_LDS_INTERLEAVE removed (ship OFF; wash vs HIP structure).
  sinkable = _SINKABLE_PAST_WMMA
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and _iop(u) is AMDOps.WMMA):
      i += 1
      continue
    wmma_dst = _reg_idxs(u)
    end = i + 1
    glued = set(wmma_dst)
    while end < len(out):
      v = out[end]
      if not (v.op is Ops.INS and _iop(v) is AMDOps.EXTRACT): break
      if not (_reg_idxs(v.src[0]) & glued): break
      glued |= _reg_idxs(v)
      end += 1
    block_src = set().union(*(_reg_idxs(s) for item in out[i:end] for s in item.src))
    block = end - i
    j = end
    while j < len(out):
      v = out[j]
      if v.op is not Ops.INS or _iop(v) not in sinkable: break
      if _iop(v) is AMDOps.LOAD and v.dtype is dtypes.half and _elem_count(v) == 1: break
      if _iop(v) is AMDOps.PACK_F16 and not _pack_f16_is_vec_load(v): break
      v_dst = _reg_idxs(v)
      v_src = set().union(*(_reg_idxs(s) for s in v.src))
      if v_src & glued or v_dst & (block_src | glued): break
      j += 1
    if j > end:
      chunk = out[i:end]
      del out[i:end]
      insert_at = j - block
      out[insert_at:insert_at] = chunk
      i = insert_at
      continue
    i += 1
  return out

def is_addr_alu(u:UOp) -> bool:
  return (u.op is Ops.INS and _iop(u) in (AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.SHL, AMDOps.SHR,
                                        AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.MOV) and
          u.dtype in dtypes.ints)

def hoist_b_between_a_and_pack(ops:list[UOp]) -> list[UOp]:
  """Issue wide B (B128) while scalar A U16 loads are still in flight."""
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and _iop(u) is AMDOps.LOAD and u.dtype is dtypes.half and _elem_count(u) >= 8):
      i += 1
      continue
    start = i
    while start > 0 and is_addr_alu(out[start - 1]): start -= 1
    end = i + 1
    if end < len(out) and out[end].op is Ops.INS and _iop(out[end]) is AMDOps.PACK_F16 and _pack_f16_is_vec_load(out[end]):
      end += 1
    j = start - 1
    while j >= 0 and out[j].op is Ops.INS and _iop(out[j]) is AMDOps.EXTRACT: j -= 1
    if j < 0 or not (out[j].op is Ops.INS and _iop(out[j]) is AMDOps.WMMA):
      i += 1
      continue
    wmma_i = j
    j -= 1
    if j < 0 or not (out[j].op is Ops.INS and _iop(out[j]) is AMDOps.PACK_F16 and not _pack_f16_is_vec_load(out[j])):
      i += 1
      continue
    pack_a_i = j
    k = pack_a_i - 1
    while k >= 0 and out[k].op is Ops.INS and _iop(out[k]) is AMDOps.LOAD and \
          out[k].dtype is dtypes.half and _elem_count(out[k]) == 1:
      k -= 1
    if k + 1 >= pack_a_i:
      i += 1
      continue
    mid, chunk = out[pack_a_i:start], out[start:end]
    mid_set, chunk_set = set(mid), set(chunk)
    if any(s in mid_set for cu in chunk for s in cu.src):
      i += 1
      continue
    if any(s in chunk_set for mu in mid for s in mu.src):
      i += 1
      continue
    if any(s in chunk_set for s in out[wmma_i].src):
      i += 1
      continue
    del out[start:end]
    out[pack_a_i:pack_a_i] = chunk
    i = pack_a_i + len(chunk)
  return out

def prefetch_next_bu16_before_pack(ops:list[UOp]) -> list[UOp]:
  """Issue next strided B U16 tile while current B U16 loads are still in flight.

  Pre-regalloc: B0_u16* → PACK_B0 → WMMA → EXTRACT* → B1_addr* → B1_u16* → PACK_B1
  becomes:      B0_u16* → B1_addr* → B1_u16* → PACK_B0 → WMMA → EXTRACT* → PACK_B1
  Regalloc assigns B0/B1 distinct VGPRs; soft wait on PACK_B0 leaves B1 in flight through WMMA0.
  """
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and _iop(u) is AMDOps.PACK_F16 and not _pack_f16_is_vec_load(u)):
      i += 1
      continue
    if i + 1 >= len(out) or not (out[i + 1].op is Ops.INS and _iop(out[i + 1]) is AMDOps.WMMA):
      i += 1
      continue
    if out[i + 1].src[1] is not u and out[i + 1].src[2] is not u:
      i += 1
      continue
    # Insert before optional PACK_A (vec) that sits between B0 U16 and PACK_B0.
    insert_at = i - 1 if (i > 0 and out[i - 1].op is Ops.INS and _iop(out[i - 1]) is AMDOps.PACK_F16 and
                          _pack_f16_is_vec_load(out[i - 1])) else i
    # Require a preceding scalar half load streak (current B tile).
    k = insert_at - 1
    while k >= 0 and out[k].op is Ops.INS and _iop(out[k]) is AMDOps.LOAD and \
          out[k].dtype is dtypes.half and _elem_count(out[k]) == 1:
      k -= 1
    if k + 1 >= insert_at:
      i += 1
      continue
    wmma_i = i + 1
    j = wmma_i + 1
    while j < len(out) and out[j].op is Ops.INS and _iop(out[j]) is AMDOps.EXTRACT: j += 1
    start = j
    while j < len(out) and is_addr_alu(out[j]): j += 1
    load0 = j
    while j < len(out) and out[j].op is Ops.INS and _iop(out[j]) is AMDOps.LOAD and \
          out[j].dtype is dtypes.half and _elem_count(out[j]) == 1:
      j += 1
    if j == load0:
      i += 1
      continue
    end = j
    # Also pull the next contiguous A wide load into the same VMEM window (leave its PACK_A
    # with PACK_B1 so WMMA0 only consumes B0). Restores U16→B128 overlap at A-row transitions.
    j2 = end
    while j2 < len(out) and is_addr_alu(out[j2]): j2 += 1
    if j2 < len(out) and out[j2].op is Ops.INS and _iop(out[j2]) is AMDOps.LOAD and \
       out[j2].dtype is dtypes.half and _elem_count(out[j2]) >= 8:
      end = j2 + 1
    mid, chunk = out[insert_at:start], out[start:end]
    mid_set, chunk_set = set(mid), set(chunk)
    if any(s in mid_set for cu in chunk for s in cu.src):
      i += 1
      continue
    if any(s in chunk_set for mu in mid for s in mu.src):
      i += 1
      continue
    del out[start:end]
    out[insert_at:insert_at] = chunk
    i = insert_at + len(chunk) + len(mid)
  return out

def hoist_loads_before_wmma(ops:list[UOp]) -> list[UOp]:
  # Bubble LOAD/PACK_F16 (+ int addr) above preceding WMMAs when independent.
  # Must not clobber a WMMA's A/B/ACC — UPCAST≥4 reuses PACK VGPRs across tiles; hoisting
  # the next tile's load above a prior WMMA left only the last A in those regs (wrong results).
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  i = 0
  while i < len(out):
    u = out[i]
    if not (u.op is Ops.INS and _iop(u) in (AMDOps.LOAD, AMDOps.LLOAD, AMDOps.PACK_F16)):
      i += 1
      continue
    # Only hoist wide B (half×8+) / vec-load packs / (opt-in) LDS loads above WMMA — not scalar A.
    # Hoisting scalar A above prior WMMA collapsed the schedule to all-A-then-B (no A/B overlap).
    if _iop(u) is AMDOps.LOAD and u.dtype is dtypes.half and _elem_count(u) == 1:
      i += 1
      continue
    if _iop(u) is AMDOps.PACK_F16 and not _pack_f16_is_vec_load(u):
      i += 1
      continue
    if _iop(u) is AMDOps.LLOAD:
      i += 1
      continue
    # Grow a hoistable prefix of addr ALU ending at this LOAD/PACK (and following PACK).
    start = i
    while start > 0:
      p = out[start - 1]
      if p.op is Ops.INS and _iop(p) in (AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.SHL, AMDOps.SHR,
                                       AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.MOV) and \
         p.dtype in dtypes.ints:
        start -= 1
        continue
      break
    end = i + 1
    if end < len(out) and out[end].op is Ops.INS and _iop(out[end]) is AMDOps.PACK_F16:
      end += 1
    chunk_src = set().union(*(set().union(*(_reg_idxs(s) for s in out[k].src)) for k in range(start, end)))
    chunk_dst = set().union(*(_reg_idxs(out[k]) for k in range(start, end)))
    dest = start
    wmma_passed = 0
    while dest > 0:
      p = out[dest - 1]
      if p.op is Ops.INS and _iop(p) is AMDOps.EXTRACT:
        # Walk through ACC EXTRACTs glued after WMMA, but never past EXTRACTs of a still-live
        # vector LOAD whose VGPRs the hoisted chunk would clobber (half 8×8: B addr ADDs into
        # A’s B128 dests → sq_intr hang / wrong mul).
        ext_src = set().union(*(_reg_idxs(s) for s in p.src))
        if chunk_dst & (ext_src | _reg_idxs(p)): break
        dest -= 1
        continue
      if p.op is Ops.INS and _iop(p) is AMDOps.WMMA:
        wmma_src = set().union(*(_reg_idxs(s) for s in p.src))
        if _reg_idxs(p) & chunk_src: break
        if chunk_dst & (wmma_src | _reg_idxs(p)): break
        wmma_passed += 1
        dest -= 1
        continue
      break
    if dest < start:
      chunk = out[start:end]
      del out[start:end]
      out[dest:dest] = chunk
      i = dest + len(chunk)
      continue
    i += 1
  return out

DEQUANT_MIX_OPS = frozenset({AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16, AMDOps.CVT_UBYTE_F32})


def prefetch_a_after_packed_quant(ops:list[UOp]) -> list[UOp]:
  """Issue activation B128s immediately after a packed quant load while it is in flight.

  IQ4 performs LDS table lookups between its packed weight read and activation reads.  Moving
  independent activations ahead of those lookups lets the soft VMEM scoreboard wait only for
  the quant data and leaves the activation burst outstanding until WMMA consumes it.
  """
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  for i,u in enumerate(out):
    if u.op is not Ops.INS or _iop(u) is not AMDOps.LOAD or u.dtype is not dtypes.uint32 or _reg_slots(u) != 4: continue
    wide:list[UOp] = []
    saw_lload, j = False, i + 1
    while j < len(out):
      v = out[j]
      if v.op is Ops.INS and _iop(v) is AMDOps.WMMA: break
      if v.op is Ops.INS and _iop(v) in (AMDOps.BARRIER, AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ,
                                        AMDOps.STORE, AMDOps.LSTORE, AMDOps.SSTORE):
        wide = []
        break
      if v.op is Ops.INS and _iop(v) is AMDOps.LLOAD: saw_lload = True
      if v.op is Ops.INS and _iop(v) is AMDOps.LOAD and v.dtype is dtypes.half and _elem_count(v) >= 8: wide.append(v)
      j += 1
    if not saw_lload or not wide: continue
    mid = out[i + 1:j]
    if any(s in mid for load in wide for s in load.src): continue
    out[i + 1:j] = wide + [x for x in mid if x not in wide]
    break
  return out

DEQUANT_MIX_OPS = frozenset({AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16, AMDOps.CVT_UBYTE_F32})

def prefetch_a_before_dequant_mix(ops:list[UOp]) -> list[UOp]:
  """Issue activation B128s (+ addr cone) right after packed weights.

  Without LDS lookups, linearize leaves A after unpack+mix. HIP issues A mid-unpack so soft
  vmcnt on the first weight use leaves A outstanding through shifts/CVT/FMA_MIX. Q6 A addressing
  does not touch weight data — hoist the addr cone with the loads to immediately after the last
  packed weight read (before weight EXTRACTs trigger a hard wait).

  Q6-only: requires CVT_UBYTE dequant and no LDS LUT traffic (IQ4 keeps its own prefetchers).
  """
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops): return ops
  out = list(ops)
  wmma0 = next(i for i,u in enumerate(out) if u.op is Ops.INS and _iop(u) is AMDOps.WMMA)
  if not any(u.op is Ops.INS and _iop(u) is AMDOps.CVT_UBYTE_F32 for u in out[:wmma0]): return ops
  if any(u.op is Ops.INS and _iop(u) is AMDOps.LLOAD for u in out[:wmma0]): return ops
  weights = [i for i,u in enumerate(out) if i < wmma0 and u.op is Ops.INS and _iop(u) is AMDOps.LOAD and
             u.dtype is dtypes.uint32 and _reg_slots(u) == 4]
  if not weights: return ops
  insert_at = weights[-1] + 1
  end = next((i for i,u in enumerate(out) if i > wmma0 and u.op is Ops.INS and
              _iop(u) in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ, AMDOps.BARRIER,
                          AMDOps.STORE, AMDOps.LSTORE, AMDOps.SSTORE)), len(out))
  late = [i for i,u in enumerate(out) if insert_at <= i < end and u.op is Ops.INS and
          _iop(u) is AMDOps.LOAD and u.dtype is dtypes.half and _elem_count(u) >= 8]
  if not late: return ops
  pos = {u:i for i,u in enumerate(out)}
  # Hoist each A load with ancestors that currently sit after the weight read (addr cone).
  move_idx: set[int] = set()
  for li in late:
    stack = [out[li]]
    seen: set[UOp] = set()
    while stack:
      u = stack.pop()
      if u in seen: continue
      seen.add(u)
      i = pos.get(u)
      if i is None or i < insert_at: continue
      move_idx.add(i)
      for src in u.src:
        if src in pos and pos[src] >= insert_at: stack.append(src)
  # Never drag weight EXTRACTs / dequant / packs into the A burst.
  weight_set = {out[i] for i in weights}
  for i in move_idx:
    u = out[i]
    if u.op is not Ops.INS: continue
    op = _iop(u)
    if op in DEQUANT_MIX_OPS or op in (AMDOps.WMMA, AMDOps.PACK_F16): return ops
    if op is AMDOps.EXTRACT and u.src and u.src[0] in weight_set: return ops
  chunks = [out[i] for i in sorted(move_idx)]
  stay = [out[i] for i in range(insert_at, end) if i not in move_idx]
  if any(s in set(stay) for cu in chunks for s in cu.src): return ops
  new = out[:insert_at] + chunks + stay + out[end:]
  return ops if new == out else new

def prefetch_late_iq4_a_before_mix(ops:list[UOp]) -> list[UOp]:
  """Issue the second 32-token A fragment after IQ4 LUT reads but before dequant FMAs.

  Hoisting both token fragments directly after the packed-weight read keeps too many
  VGPRs live across unpacking. This shorter overlap lets the later A loads run during
  mixlo/mixhi and the first WMMA pair instead.
  """
  wmmas = [i for i,u in enumerate(ops) if u.op is Ops.INS and _iop(u) is AMDOps.WMMA]
  if len(wmmas) != 4: return ops
  target = next((i for i,u in enumerate(ops) if u.op is Ops.INS and _iop(u) is AMDOps.PACKED_F16_MUL_TO_F16), -1)
  if target < 0 or target >= wmmas[0]: return ops
  late = [i for i,u in enumerate(ops) if wmmas[0] < i < wmmas[-1] and u.op is Ops.INS and
          _iop(u) is AMDOps.LOAD and u.dtype is dtypes.half and _elem_count(u) >= 8]
  pos = {u:i for i,u in enumerate(ops)}
  if len(late) != 2 or any(any(pos[s] >= target for s in ops[i].src if s in pos) for i in late): return ops
  loads = [ops[i] for i in late]
  late_set = set(late)
  out = [u for i,u in enumerate(ops) if i not in late_set]
  target = next(i for i,u in enumerate(out) if u.op is Ops.INS and _iop(u) is AMDOps.PACKED_F16_MUL_TO_F16)
  return out[:target] + loads + out[target:]

VMEM_SCHEDULABLE = {AMDOps.MOV, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.COLLECT, AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.MULACC, AMDOps.FMAC,
                     AMDOps.CAST, AMDOps.RECIPROCAL, AMDOps.EXP2, AMDOps.LOG2, AMDOps.SQRT, AMDOps.TRUNC, AMDOps.SIN,
                     AMDOps.MAX, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.BFE, AMDOps.CVT_UBYTE_F32,
                     AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16,
                     AMDOps.LSHL_OR, AMDOps.LSHL_ADD,
                     AMDOps.LOAD, AMDOps.PACK_F16}

def hoist_gated_fmac_loads(ops:list[UOp]) -> list[UOp]:
  """Issue adjacent independently-gated scalar loads before consuming any of them.

  VCC makes comparisons and WHERE inseparable scheduling pairs.  Handle the narrow
  CMP/WHERE-address/LOAD/CMP/WHERE-value/consumer form as six-UOp groups, preserving each
  pair while moving only the independent address/load halves ahead of the consumers.

  Consumers include FMAC (flash) and AND/BFE/LSHL_OR/CAST (quant dequant). Gate CMPs are
  excluded from the independence set — they are often shared across adjacent gated reads and
  previously forced every streak to length 1 (vmcnt(0) per uchar load).
  """
  cmps = (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ, AMDOps.CMP_GE)
  bad_consumer = {AMDOps.LOAD, AMDOps.STORE, AMDOps.LLOAD, AMDOps.LSTORE, AMDOps.SLOAD, AMDOps.SSTORE,
                  AMDOps.WHERE, AMDOps.IF_MASK, AMDOps.END_MASK, AMDOps.BARRIER, AMDOps.BRANCH,
                  AMDOps.CBRANCH_SCC1, AMDOps.LABEL, AMDOps.LOOP_CMP}
  def group_at(i:int) -> tuple[UOp, ...]|None:
    if i + 5 >= len(ops): return None
    cmpa, addr, load, cmpv, val, consumer = ops[i:i+6]
    if not (cmpa.op is addr.op is load.op is cmpv.op is val.op is consumer.op is Ops.INS): return None
    if _iop(cmpa) not in cmps or _iop(addr) is not AMDOps.WHERE or _iop(load) is not AMDOps.LOAD or \
       _iop(cmpv) not in cmps or _iop(val) is not AMDOps.WHERE or _iop(consumer) in bad_consumer: return None
    if not (addr.src and addr.src[0] is cmpa and len(load.src) >= 2 and load.src[1] is addr and
            val.src and val.src[0] is cmpv and len(val.src) >= 2 and val.src[1] is load and
            val in consumer.src): return None
    if not vmem_schedulable_load(load) or _reg_slots(load) > 4: return None
    return cmpa, addr, load, cmpv, val, consumer

  out:list[UOp] = []
  i = 0
  while i < len(ops):
    groups:list[tuple[UOp, ...]] = []
    late_data:set[UOp] = set()
    while (group:=group_at(i + len(groups) * 6)) is not None:
      if any(src in late_data for u in group[:3] for src in u.src): break
      groups.append(group)
      late_data.update((group[2], group[4], group[5]))  # load, value, consumer — not cmpv
    if len(groups) >= 2:
      out.extend(u for group in groups for u in group[:3])
      out.extend(u for group in groups for u in group[3:])
      i += len(groups) * 6
    else:
      out.append(ops[i])
      i += 1
  return out

def vmem_schedulable_load(u:UOp) -> bool:
  slots = _reg_slots(u)
  # half×4 B64 and packed u32×4 B128 (_amd_load SHRINK×4): independent addr+dest like scalar;
  # must not bail whole-kernel schedule (Q6 decode / flash).
  return slots == 1 or (u.dtype is dtypes.half and slots == 2) or \
    (u.dtype in (dtypes.uint8, dtypes.float32, dtypes.uint32, dtypes.int32) and slots <= 4)

def schedule_scalar_vmem(ops:list[UOp], alu_breadth:bool|None=None) -> list[UOp]:
  """Hoist independent global reads inside conservative straight-line segments.

  Run before register allocation so independent reads receive distinct live registers.
  Explicit SSA dependencies preserve value order; REG_STORE and all other implicit
  architectural state or memory side effects are hard boundaries.
  """
  # Custom quantized WMMA kernels use independent packed u32x4 reads, but also contain
  # wide activation reads that must remain hard scheduling boundaries. Schedule only
  # the compatible straight-line subsegments.
  schedule_wmma_segments = \
    any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA for u in ops) and \
    any(u.op is Ops.INS and _iop(u) is AMDOps.LOAD and u.dtype is dtypes.uint32 and _reg_slots(u) == 4 for u in ops)
  # Q6/Q4 decode (linear_q6): many scalar packed-weight reads + dp4a/BFE chains between LLOADs.
  schedule_decode_segments = not schedule_wmma_segments and \
    any(u.op is Ops.INS and _iop(u) is AMDOps.DOT4 for u in ops) and \
    sum(1 for u in ops if u.op is Ops.INS and _iop(u) is AMDOps.LOAD and u.dtype in (dtypes.uint, dtypes.uint32)) >= 8
  looped_non_fmac_reg_state = any(u.op is Ops.RANGE for u in ops) and \
    any(u.op is Ops.AFTER and u.addrspace is AddrSpace.REG for u in ops) and \
    not any(u.op is Ops.INS and _iop(u) is AMDOps.FMAC for u in ops)
  if not schedule_wmma_segments and not schedule_decode_segments and \
     any(u.op is Ops.INS and (_iop(u) is AMDOps.WMMA or (_iop(u) is AMDOps.LOAD and not vmem_schedulable_load(u))) for u in ops): return ops
  def schedulable(u:UOp) -> bool:
    if u.op is Ops.BITCAST or (u.op is Ops.NOOP and u.dtype is not dtypes.void): return True  # value alias, never a void ordering gate
    if u.op is Ops.AFTER and u.addrspace is AddrSpace.REG: return True  # register accumulator binding, not a memory/state boundary
    if u.op is not Ops.INS or _iop(u) not in VMEM_SCHEDULABLE: return False
    # Loop-carried REG values are implicit mutable state. Keep non-FMAC loop ALU ops out of the
    # VMEM scheduler window until it models their physical-register hazards with async loads.
    # SCHEDULE_VMEM_ALU folded into VOPD_FMAC invert (decode ship ALU=0; else 1).
    alu_mode = 0 if getenv("AMD_VOPD_FMAC", 0) or looped_non_fmac_reg_state else 1
    if alu_mode == 0 and _iop(u) not in (
        AMDOps.LOAD, AMDOps.MOV, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.COLLECT, AMDOps.CAST, AMDOps.PACK_F16):
      return False
    # Wide f32 and packed-byte loads have independent addresses and dedicated destination
    # VGPRs, so they can participate. Other wide/d16 loads retain emitter temp constraints.
    return _iop(u) is not AMDOps.LOAD or vmem_schedulable_load(u) or \
      (schedule_wmma_segments and u.dtype is dtypes.uint32 and _reg_slots(u) == 4) or \
      (schedule_decode_segments and u.dtype in (dtypes.uint, dtypes.uint32) and _reg_slots(u) <= 4)

  def schedule(segment:list[UOp]) -> list[UOp]:
    loads = [i for i,u in enumerate(segment) if _iop(u) is AMDOps.LOAD]
    if len(loads) < 2: return segment
    deps:list[set[int]] = [set() for _ in segment]
    users:list[list[int]] = [[] for _ in segment]
    order = {u:i for i,u in enumerate(segment)}
    for i,u in enumerate(segment):
      # Preserve explicit UOp dependencies even when the value has no allocated register.
      deps[i].update(order[s] for s in u.src if s in order and order[s] < i)
    for i,ds in enumerate(deps):
      for dep in ds: users[dep].append(i)

    # Address/dependency chains of loads are favored after already-ready loads.  Hard
    # segment boundaries keep implicit mutable state and memory side effects fixed.
    load_ancestors:set[int] = set()
    stack = [dep for i in loads for dep in deps[i]]
    while stack:
      i = stack.pop()
      if i in load_ancestors: continue
      load_ancestors.add(i)
      stack.extend(deps[i])

    # Independent dequantization chains are commonly linearized one at a time.  Once
    # their VMEM reads are issued, breadth-first ALU order keeps several BFE/cvt/mul
    # chains in flight instead of immediately consuming each dependent result.  This
    # approximates LLVM's latency-hiding list schedule without crossing hard segment
    # boundaries or inventing dependencies. AMD_SCHEDULE_ALU=0 is a diagnostic opt-out.
    use_alu_breadth = True if alu_breadth is None else alu_breadth
    alu_depth = [0] * len(segment)
    if use_alu_breadth:
      for i,u in enumerate(segment):
        if _iop(u) is not AMDOps.LOAD and deps[i]: alu_depth[i] = max(alu_depth[d] + 1 for d in deps[i])

    indegree = [len(ds) for ds in deps]
    ready = [i for i,n in enumerate(indegree) if n == 0]
    scheduled:list[UOp] = []
    while ready:
      i = min(ready, key=lambda j: (0 if _iop(segment[j]) is AMDOps.LOAD else 1 if j in load_ancestors else 2,
                                    alu_depth[j] if use_alu_breadth else 0, j))
      ready.remove(i)
      scheduled.append(segment[i])
      for user in users[i]:
        indegree[user] -= 1
        if indegree[user] == 0: ready.append(user)
    return scheduled if len(scheduled) == len(segment) else segment

  out:list[UOp] = []
  segment:list[UOp] = []
  mask_depth = 0
  for u in ops:
    if u.op is Ops.INS and _iop(u) is AMDOps.IF_MASK:
      out.extend(schedule(segment))
      segment = []
      out.append(u)
      mask_depth += 1
    elif u.op is Ops.INS and _iop(u) is AMDOps.END_MASK:
      out.extend(segment)
      segment = []
      out.append(u)
      mask_depth = max(0, mask_depth - 1)
    elif mask_depth or not schedulable(u):
      out.extend(schedule(segment))
      segment = []
      out.append(u)
    else: segment.append(u)
  out.extend(schedule(segment))
  return out

def fma_mixhi_lo_map(uops:list[UOp]) -> dict[UOp, UOp]:
  """Pair single-use half FMAs consumed by the same WMMA fragment pack.

  The low result remains live through PACK_F16, so mixhi can safely update its high
  half in place without changing instruction order or extending any input lifetime.
  """
  uses: dict[UOp, list[UOp]] = {}
  for u in uops:
    for src in u.src: uses.setdefault(src, []).append(u)
  ret: dict[UOp, UOp] = {}
  fma_ops = (AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16)
  for pack in uops:
    if pack.op is not Ops.INS or _iop(pack) is not AMDOps.PACK_F16 or _pack_f16_is_vec_load(pack): continue
    for lo, hi in zip(pack.src[::2], pack.src[1::2]):
      if lo.op is not Ops.INS or hi.op is not Ops.INS or _iop(lo) not in fma_ops or _iop(hi) is not _iop(lo): continue
      if uses.get(lo) == [pack] and uses.get(hi) == [pack]: ret[hi] = lo
  return ret

def schedule_fma_mixhi_pairs(uops:list[UOp]) -> list[UOp]:
  """Separate paired mixlo/mixhi instructions enough to cover the destination dependency."""
  pairs = fma_mixhi_lo_map(uops)
  lows, highs = set(pairs.values()), set(pairs)
  paired = lows | highs
  out, i = [], 0
  while i < len(uops):
    if uops[i] not in paired:
      out.append(uops[i])
      i += 1
      continue
    j = i
    while j < len(uops) and uops[j] in paired: j += 1
    run = uops[i:j]
    if any(src in paired for u in run for src in u.src):
      out.extend(run)
    else:
      for k in range(0, len(run), 8):
        chunk = run[k:k+8]
        out.extend([u for u in chunk if u in lows])
        out.extend([u for u in chunk if u in highs])
    i = j
  return out

def half2_global_hi_skip(uops:list[UOp]) -> set[UOp]:
  """Odd-lane EXTRACTs that only feed HALF2_GLOBAL packs — LSHR is dead (pack uses lo's full word)."""
  if not getenv("AMD_FLASH_ACC_SMALL", 0): return set()
  uses: dict[UOp, list[UOp]] = {}
  for u in uops:
    for src in u.src: uses.setdefault(src, []).append(u)
  skip: set[UOp] = set()
  for pack in uops:
    if pack.op is not Ops.INS or _iop(pack) is not AMDOps.PACK_F16 or _pack_f16_is_vec_load(pack): continue
    for lo, hi in zip(pack.src[::2], pack.src[1::2]):
      if _pack_f16_half2_load(lo, hi) is None: continue
      if uses.get(hi) == [pack]: skip.add(hi)
  return skip

def rewrite_half2_global_packs(ops:list[UOp]) -> list[UOp]:
  """Drop dead hi EXTRACTs and shrink PACK_F16 to lo-only srcs (full half2 words).

  Must run pre-regalloc so hi VGPRs are not allocated (they otherwise break lo-contig
  alias). Emit treats lo-only packs as bitcast MOVs from each even EXTRACT.
  """
  if not getenv("AMD_FLASH_ACC_SMALL", 0): return ops
  uses: dict[UOp, list[UOp]] = {}
  for u in ops:
    for src in u.src: uses.setdefault(src, []).append(u)
  skip_hi: set[UOp] = set()
  repl: dict[UOp, UOp] = {}
  for pack in ops:
    if pack.op is not Ops.INS or _iop(pack) is not AMDOps.PACK_F16 or _pack_f16_is_vec_load(pack): continue
    if len(pack.src) < 2 or len(pack.src) % 2: continue
    pairs = list(zip(pack.src[::2], pack.src[1::2]))
    if not all(_pack_f16_half2_load(lo, hi) is not None for lo, hi in pairs): continue
    if not all(uses.get(hi) == [pack] for _, hi in pairs): continue
    # Only rewrite when every hi is pack-only (safe to delete).
    if not all(_iop(lo.src[0]) is AMDOps.LOAD for lo, _ in pairs): continue
    repl[pack] = pack.replace(src=tuple(lo for lo, _ in pairs))
    for _, hi in pairs: skip_hi.add(hi)
  if not repl: return ops
  out: list[UOp] = []
  for u in ops:
    if u in skip_hi: continue
    if u in repl:
      out.append(repl[u])
      continue
    if any(s in repl for s in u.src):
      out.append(u.replace(src=tuple(repl.get(s, s) for s in u.src)))
    else:
      out.append(u)
  return out

def schedule_half2_lo_before_pack(ops:list[UOp]) -> list[UOp]:
  """Place lo-only HALF2 pack EXTRACTs immediately before their PACK (allocation order → contig)."""
  if not getenv("AMD_FLASH_ACC_SMALL", 0): return ops
  pos = {u: i for i, u in enumerate(ops)}
  move: dict[UOp, UOp] = {}  # extract → pack
  for pack in ops:
    if pack.op is not Ops.INS or _iop(pack) is not AMDOps.PACK_F16: continue
    if not _pack_f16_is_half2_lo_only(pack): continue
    for s in pack.src:
      if s in pos and pos[s] < pos[pack]: move[s] = pack
  if not move: return ops
  skip = set(move)
  out: list[UOp] = []
  pending: dict[UOp, list[UOp]] = {}
  for s, pack in move.items():
    pending.setdefault(pack, []).append(s)
  for u in ops:
    if u in skip: continue
    if u in pending:
      # Stable order matching pack.src so bias lane0..n-1 matches program order.
      want = [s for s in u.src if s in move]
      out.extend(want)
    out.append(u)
  return out

def hoist_kernargs(ops:list[UOp]) -> list[UOp]:
  """Issue all KERNARG SMEM reads early so one lgkmcnt covers them (HIP s_load_b256 pattern)."""
  idxs = [i for i,u in enumerate(ops) if u.op is Ops.INS and _iop(u) is AMDOps.KERNARG]
  if len(idxs) < 2: return ops
  kernarg_set = {ops[i] for i in idxs}
  kernargs = sorted(kernarg_set, key=lambda u: _const_int(u.src[0]) or 0)
  insert_at = idxs[0]
  rest = [u for i,u in enumerate(ops) if u not in kernarg_set]
  prefix_len = sum(1 for i in range(insert_at) if ops[i] not in kernarg_set)
  return rest[:prefix_len] + kernargs + rest[prefix_len:]

def hoist_lloads_before_extracts(ops:list[UOp]) -> list[UOp]:
  """Issue all ds_loads then EXTRACTs so one lgkmcnt covers the batch."""
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if u.op is Ops.INS and _iop(u) is AMDOps.LLOAD:
      j, lloads, extracts = i, [], []
      while j < len(ops):
        v = ops[j]
        if v.op is Ops.INS and _iop(v) is AMDOps.LLOAD:
          lloads.append(v)
          j += 1
        elif v.op is Ops.INS and _iop(v) is AMDOps.EXTRACT:
          extracts.append(v)
          j += 1
        else: break
      out.extend(lloads)
      out.extend(extracts)
      i = j
    else:
      out.append(u)
      i += 1
  return out

def _lload_addr_depends_on(consumer:UOp, producer:UOp) -> bool:
  for addr in consumer.src[:2]:
    if addr is producer or producer in addr.toposort(): return True
  return False

def schedule_lload_2addr_pairs(ops:list[UOp]) -> list[UOp]:
  """Within each LLOAD streak, order fusible 8-byte pairs adjacent for ds_load_2addr_b64."""
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if not (u.op is Ops.INS and _iop(u) is AMDOps.LLOAD):
      out.append(u)
      i += 1
      continue
    j = i
    while j < len(ops) and ops[j].op is Ops.INS and _iop(ops[j]) is AMDOps.LLOAD:
      j += 1
    streak = ops[i:j]
    if len(streak) >= 2:
      used = [False] * len(streak)
      ordered: list[UOp] = []
      for a in range(len(streak)):
        if used[a]: continue
        sa = streak[a]
        used[a] = True
        ordered.append(sa)
        if _elem_count(sa) * sa.dtype.itemsize != 8: continue
        off = _lds_byte_off(sa)
        for b in range(a + 1, len(streak)):
          if used[b]: continue
          sb = streak[b]
          if not (sb.src[0] is sa.src[0] and sb.src[1] is sa.src[1] and
                  _elem_count(sb) * sb.dtype.itemsize == 8 and
                  _lds_byte_off(sb) == off + 8):
            continue
          if any(_lload_addr_depends_on(sb, streak[k]) for k in range(a + 1, b)):
            continue
          used[b] = True
          ordered.append(sb)
          break
      out.extend(ordered)
    else:
      out.extend(streak)
    i = j
  return out

# Control-flow / sync ops that must not be crossed when clustering scratch traffic.
_SCRATCH_CLUSTER_CF = (AMDOps.LABEL, AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ,
                       AMDOps.IF_MASK, AMDOps.END_MASK, AMDOps.BARRIER, AMDOps.LOOP_CMP)
_SCRATCH_BATCH_UNSAFE_USE = (AMDOps.SSTORE, AMDOps.STORE, AMDOps.SPILL)

def cluster_const_scratch_stores(ops:list[UOp]) -> list[UOp]:
  """Sort independent const-index SSTORE to the same base so b128 fusion can match."""
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if u.op is Ops.INS and _iop(u) is AMDOps.SSTORE and _const_int(u.src[1]) is not None:
      base = u.src[0]
      group = [u]
      j = i + 1
      while j < len(ops) and len(group) < 32:
        v = ops[j]
        if v.op is Ops.INS and _iop(v) is AMDOps.SSTORE and v.src[0] is base and \
           _const_int(v.src[1]) is not None:
          group.append(v)
          j += 1
          continue
        break
      if len(group) >= 4:
        def _store_key(x:UOp) -> tuple[int, int]:
          return (int(_const_int(x.src[1])), _lds_byte_off(x))  # type: ignore[arg-type]
        group.sort(key=_store_key)
        out.extend(group)
        i = j
        continue
    out.append(u)
    i += 1
  return out

def _scratch_cluster_blocks_load(u:UOp, base:UOp) -> bool:
  if u.op is not Ops.INS: return True
  iop = _iop(u)
  if iop in _SCRATCH_CLUSTER_CF: return True
  if iop in (AMDOps.SPILL, AMDOps.FILL): return True
  if iop is AMDOps.SSTORE and u.src and u.src[0] is base: return True
  return False

def cluster_const_scratch_loads(ops:list[UOp]) -> list[UOp]:
  """Pull independent const-index SLOADs to the same base together (sorted) for b128 fusion."""
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if u.op is Ops.INS and _iop(u) is AMDOps.SLOAD and _const_int(u.src[1]) is not None and _lds_byte_off(u) == 0:
      base = u.src[0]
      group = [u]
      taken = {i}
      j = i + 1
      max_scan = i + 1 + 64
      while j < len(ops) and j < max_scan and len(group) < 32:
        v = ops[j]
        if _scratch_cluster_blocks_load(v, base):
          break
        if v.op is Ops.INS and _iop(v) is AMDOps.SLOAD and v.src[0] is base and \
           _const_int(v.src[1]) is not None and _lds_byte_off(v) == 0 and \
           not any(g in v.src or v in g.src for g in group):
          group.append(v)
          taken.add(j)
        j += 1
      if len(group) >= 4:
        group.sort(key=lambda x: int(_const_int(x.src[1])))  # type: ignore[arg-type]
        out.extend(group)
        for k in range(i + 1, j):
          if k not in taken: out.append(ops[k])
        i = j
        continue
    out.append(u)
    i += 1
  return out

def _scratch_batch_use_safe(u:UOp) -> bool:
  if u.op is not Ops.INS: return False
  iop = _iop(u)
  if iop in _SCRATCH_BATCH_UNSAFE_USE or iop in _SCRATCH_CLUSTER_CF: return False
  return True

def _scratch_batch_phys_conflict(loads:list[UOp], uses:list[UOp], new_load:UOp, new_use:UOp) -> bool:
  cand_loads, cand_uses = loads + [new_load], uses + [new_use]
  for i, ld in enumerate(cand_loads):
    live = _reg_idxs(ld)
    if not live: continue
    for later in cand_loads[i + 1:]:
      if _reg_idxs(later) & live: return True
      if _reg_idxs(later) & set().union(*(_reg_idxs(s) for s in cand_uses[i].src)): return True
  return False

def batch_scratch_load_uses(ops:list[UOp], *, check_phys:bool=False) -> list[UOp]:
  """Rewrite SLOAD,USE,SLOAD,USE → SLOAD×N,USE×N so waitcnt can cover a load burst."""
  out: list[UOp] = []
  i = 0
  while i < len(ops):
    u = ops[i]
    if (u.op is Ops.INS and _iop(u) is AMDOps.SLOAD and i + 1 < len(ops) and u in ops[i + 1].src and
        _scratch_batch_use_safe(ops[i + 1])):
      loads, uses = [u], [ops[i + 1]]
      j = i + 2
      while j + 1 < len(ops) and len(loads) < 32 and \
            ops[j].op is Ops.INS and _iop(ops[j]) is AMDOps.SLOAD and \
            ops[j] in ops[j + 1].src and _scratch_batch_use_safe(ops[j + 1]) and \
            not any(prev in ops[j].src or prev in ops[j + 1].src for prev in loads + uses) and \
            not (check_phys and _scratch_batch_phys_conflict(loads, uses, ops[j], ops[j + 1])):
        loads.append(ops[j])
        uses.append(ops[j + 1])
        j += 2
      if len(loads) >= 2:
        pairs = sorted(zip(loads, uses), key=lambda lu: (
          id(lu[0].src[0]), int(c) if (c:=_const_int(lu[0].src[1])) is not None else 1 << 30))
        loads = [p[0] for p in pairs]
        uses = [p[1] for p in pairs]
        out.extend(loads)
        out.extend(uses)
        i = j
        continue
    out.append(u)
    i += 1
  return out

def schedule_scratch_load_passes(ops:list[UOp]) -> list[UOp]:
  """Composed scratch scheduling: store cluster → load/use batch → load cluster."""
  ops = cluster_const_scratch_stores(ops)
  ops = batch_scratch_load_uses(ops, check_phys=False)
  ops = cluster_const_scratch_loads(ops)
  return ops

def pack_aligned_const_scratch_sloads(ops:list[UOp]) -> list[UOp]:
  """SLOAD×4 at const idx i..i+3 (i%4==0) → one SLOAD×4 + EXTRACT×4."""
  if not getenv("AMD_FLASH_ACC_SMALL", 0): return ops  # PACK_SLOAD_B128 folded into ACC_SMALL
  max_packs = 8
  max_bases = True
  n = itertools.count()
  remap: dict[UOp, UOp] = {}
  out: list[UOp] = []
  i = 0
  packs = 0
  packed_bases: list[UOp] = []
  while i < len(ops):
    u = ops[i]
    base_ok = (u.src and (u.src[0] in packed_bases or len(packed_bases) < max_bases))
    if (packs < max_packs and base_ok and u.op is Ops.INS and _iop(u) is AMDOps.SLOAD and _elem_count(u) == 1 and
        u.dtype in (dtypes.float32, dtypes.int32, dtypes.uint32) and
        _const_int(u.src[1]) is not None and _lds_byte_off(u) == 0 and i + 3 < len(ops)):
      g = ops[i:i + 4]
      idxs = [_const_int(v.src[1]) if (v.op is Ops.INS and _iop(v) is AMDOps.SLOAD) else None for v in g]
      if (all(v.op is Ops.INS and _iop(v) is AMDOps.SLOAD and _elem_count(v) == 1 and v.dtype is u.dtype and
              v.src[0] is u.src[0] and _lds_byte_off(v) == 0 for v in g) and
          all(t is not None for t in idxs) and int(idxs[0]) % 4 == 0 and  # type: ignore[arg-type]
          [int(t) for t in idxs] == [int(idxs[0]) + k for k in range(4)]):  # type: ignore[arg-type]
        if u.src[0] not in packed_bases: packed_bases.append(u.src[0])
        wide = UOp(Ops.INS, src=(u.src[0], g[0].src[1], _load_count_src(4)),
                   arg=(AMDOps.SLOAD, u.dtype), tag=(Register(f"sload4_{next(n)}", 0, _cons=VGPR),))
        out.append(wide)
        for lane, old in enumerate(g):
          ext = UOp(Ops.INS, src=(wide, _tconst(lane, dtypes.int32).rtag()),
                    arg=(AMDOps.EXTRACT, u.dtype), tag=(Register(f"sload4e{lane}_{next(n)}", 0, _cons=VGPR),))
          remap[old] = ext
          out.append(ext)
        packs += 1
        i += 4
        continue
    if u in remap:
      i += 1
      continue
    out.append(u)
    i += 1
  if not remap: return out
  cache: dict[UOp, UOp] = {}
  def deep(u:UOp) -> UOp:
    if u in remap: return remap[u]
    if u in cache: return cache[u]
    if not u.src:
      cache[u] = u
      return u
    nsrc = tuple(deep(s) for s in u.src)
    nu = u.replace(src=nsrc) if nsrc != u.src else u
    cache[u] = nu
    return nu
  return [deep(u) for u in out]

def protect_loop_invariant_fmac(lst:list[UOp]) -> list[UOp]:
  """VOP2 FMAC overwrites src0. Loop-invariant addend → MULACC reorder."""
  active:list[tuple[UOp, set[UOp]]] = []
  out:list[UOp] = []
  remap:dict[UOp, UOp] = {}
  for old in lst:
    u = old.replace(src=tuple(remap.get(s, s) for s in old.src))
    if u.op is Ops.RANGE:
      if active: active[-1][1].add(u)
      active.append((u, set()))
    elif u.op is Ops.INS and _iop(u) is AMDOps.FMAC and active and u.src[0] not in active[-1][1]:
      u = u.replace(src=(*u.src[1:], u.src[0]), arg=(AMDOps.MULACC, u.dtype))
    out.append(u)
    if u is not old: remap[old] = u
    loop_end = u.src[1] if u.op in (Ops.END, Ops.BACKEDGE) else u.src[3] if u.op is Ops.INS and _iop(u) is AMDOps.LOOP_CMP else None
    if loop_end is not None:
      assert active and active[-1][0] is loop_end
      active.pop()
    elif active: active[-1][1].add(u)
  return out

def schedule_loop_cmps(lst:list[UOp]) -> list[UOp]:
  """Evaluate boundless-loop condition before a REG_STORE mutates an operand."""
  out:list[UOp] = []
  cmp_ops = {Ops.CMPLT:AMDOps.CMPLT, Ops.CMPNE:AMDOps.CMPNE, Ops.CMPEQ:AMDOps.CMPEQ}
  for u in lst:
    if u.op is not Ops.INS or _iop(u) is not AMDOps.LOOP_CMP:
      out.append(u)
      continue
    cmp = UOp(Ops.INS, src=u.src[:2], arg=(cmp_ops[u.tag], dtypes.bool))
    cond_regs = {r for s in u.src[:2] if isinstance((r:=greg(s)), Register)}
    dep_reg = greg(u.src[2])
    insert = len(out)
    if isinstance(dep_reg, Register) and dep_reg in cond_regs:
      for i in range(len(out)-1, -1, -1):
        if out[i].op is not Ops.INS or _iop(out[i]) is not AMDOps.REG_STORE or greg(out[i].src[0]) != dep_reg: continue
        # The update feeding REG_STORE can be allocated onto dep_reg (e.g. v_add v5, 1, v5).
        # Comparing before only the final MOV is too late: the update already clobbered the old
        # loop value. Move the compare before that producer when its operands are available.
        update = out[i].src[1]
        def index_of(target:UOp) -> int|None: return next((j for j, op in enumerate(out) if op is target), None)
        update_idx = index_of(update)
        cond_defs = [j for s in u.src[:2] if (j:=index_of(s)) is not None]
        update_idx = i if update_idx is None else update_idx
        insert = update_idx if max(cond_defs, default=-1) < update_idx else i
        break
    out.insert(insert, cmp)
    out.append(u.replace(src=(cmp, u.src[2], u.src[3])))
  return out

_mul_strength_n = itertools.count()
def strength_reduce_int_muls(ops:list[UOp]) -> list[UOp]:
  """MUL(x, C) with few set bits → ADD of SHLs so flash LDS strides remat as pure-addr."""
  max_bits = 2  # AMD_MUL_STRENGTH_BITS hardcoded
  rewrite: dict[UOp, UOp] = {}
  out: list[UOp] = []
  for u in ops:
    srcs = tuple(rewrite.get(s, s) for s in u.src)
    u2 = u.replace(src=srcs) if srcs != u.src else u
    if u2.op is Ops.INS and _iop(u2) is AMDOps.MUL and u2.dtype in (dtypes.int32, dtypes.uint32, dtypes.int, dtypes.uint) and len(u2.src) == 2:
      a, b = u2.src
      c, x = _const_int(b), a
      if c is None: c, x = _const_int(a), b
      mul_cons = u2.tag[0].cons if isinstance(u2.tag, tuple) and u2.tag else None
      if c is not None and c > 1 and (c & (c - 1)) != 0 and c.bit_count() <= max_bits and mul_cons is VGPR:
        bits = [i for i in range(c.bit_length()) if c & (1 << i)]
        cons = mul_cons
        terms: list[UOp] = []
        for bit in bits:
          if bit == 0: terms.append(x)
          else:
            sh = UOp(Ops.INS, src=(x, _tconst(bit, dtypes.int32)), arg=(AMDOps.SHL, u2.dtype),
                     tag=(Register(f"mulsh{next(_mul_strength_n)}", 0, _cons=cons),))
            out.append(sh)
            terms.append(sh)
        acc = terms[0]
        for ti, t in enumerate(terms[1:]):
          tag = u2.tag if ti == len(terms) - 2 else (Register(f"mulad{next(_mul_strength_n)}", 0, _cons=cons),)
          if not isinstance(tag, tuple): tag = (tag,)
          acc = UOp(Ops.INS, src=(acc, t), arg=(AMDOps.ADD, u2.dtype), tag=tag)
          out.append(acc)
        rewrite[u] = acc
        rewrite[u2] = acc
        continue
    if u2 is not u: rewrite[u] = u2
    out.append(u2)
  return out

def _is_f16_to_f32_cast(u:UOp) -> bool:
  return u.op is Ops.INS and _iop(u) is AMDOps.CAST and u.dtype is dtypes.float32 and \
         bool(u.src) and u.src[0].dtype is dtypes.float16

def fma_mix_f32_folds(uops:list[UOp]) -> tuple[dict[UOp, tuple[UOp, UOp]], set[UOp]]:
  """Fold FMAC(acc, CAST(f16), f32) into v_fma_mix_f32. Opt-in via AMD_VOPD_FMAC."""
  if not getenv("AMD_VOPD_FMAC", 0): return {}, set()
  uses: dict[UOp, list[UOp]] = {}
  for u in uops:
    for src in u.src: uses.setdefault(src, []).append(u)
  folds: dict[UOp, tuple[UOp, UOp]] = {}
  skip: set[UOp] = set()
  for cast, consumers in uses.items():
    if not _is_f16_to_f32_cast(cast): continue
    if not consumers or any(c.op is not Ops.INS or _iop(c) is not AMDOps.FMAC or c.dtype is not dtypes.float32
                            for c in consumers): continue
    hbase = cast.src[0]
    if hbase.op is Ops.INS and _iop(hbase) is AMDOps.LLOAD:
      continue
    ok = True
    folded_here: list[UOp] = []
    for u in consumers:
      half_i = next((i for i in (1, 2) if u.src[i] is cast), None)
      if half_i is None:
        ok = False
        break
      f32_i = 2 if half_i == 1 else 1
      f32 = u.src[f32_i]
      if _is_f16_to_f32_cast(f32):
        ok = False
        break
      folds[u] = (hbase, f32)
      folded_here.append(u)
    if not ok:
      for u in folded_here: folds.pop(u, None)
    elif folded_here and len(folded_here) == len(consumers):
      skip.add(cast)
  return folds, skip

def lower_fma_mix_f32(lst:list[UOp]) -> list[UOp]:
  """Rewrite FMAC(acc, CAST(f16), f32) into FMA_MIX_F32 before regalloc."""
  folds, skip_casts = fma_mix_f32_folds(lst)
  if not folds: return lst
  replace: dict[UOp, UOp] = {
    u: UOp(Ops.INS, src=(u.src[0], hbase, f32), arg=(AMDOps.FMA_MIX_F32, dtypes.float32), tag=u.tag)
    for u, (hbase, f32) in folds.items()}
  out: list[UOp] = []
  for u in lst:
    if u in skip_casts: continue
    if u in replace:
      out.append(replace[u])
      continue
    new_src = tuple(replace.get(s, s) for s in u.src)
    if new_src != u.src:
      nu = u.replace(src=new_src)
      replace[u] = nu
      out.append(nu)
    else:
      out.append(u)
  return out

_REG_STORE_SINK_ALU_OPS = frozenset({AMDOps.MAX, AMDOps.MUL, AMDOps.EXP2, AMDOps.ADD, AMDOps.WHERE, AMDOps.EXTRACT})

def sink_alu_into_reg_store(ops: list[UOp]) -> list[UOp]:
  """Retarget single-use ALU that only feeds REG_STORE onto the promoted REG vreg."""
  uses: dict[UOp, int] = {}
  for u in ops:
    for s in u.src: uses[s] = uses.get(s, 0) + 1
  replace_alu: dict[UOp, UOp] = {}
  drop: set[UOp] = set()
  for u in ops:
    if u.op is not Ops.INS or _iop(u) is not AMDOps.REG_STORE or len(u.src) < 2: continue
    acc, val = u.src[0], u.src[1]
    while (val.op is Ops.INS and _iop(val) is AMDOps.MOV and uses.get(val, 0) == 1 and val.src and
           val.src[0].op is Ops.INS and _iop(val.src[0]) is not AMDOps.MOV):
      val = val.src[0]
    if val.op is not Ops.INS or _iop(val) not in _REG_STORE_SINK_ALU_OPS: continue
    if uses.get(val, 0) != 1: continue
    dest = greg(acc)
    if not isinstance(dest, Register) or not dest.name.startswith("reg"): continue
    if greg(val) is dest:
      drop.add(u)
      continue
    replace_alu[val] = val.replace(tag=(dest,))
    drop.add(u)
    cur = u.src[1]
    while cur is not val and cur.op is Ops.INS and _iop(cur) is AMDOps.MOV:
      drop.add(cur)
      cur = cur.src[0]
  if not replace_alu and not drop: return ops
  out: list[UOp] = []
  for u in ops:
    if u in drop: continue
    if u in replace_alu:
      out.append(replace_alu[u])
      continue
    if any(s in replace_alu for s in u.src):
      out.append(u.replace(src=tuple(replace_alu.get(s, s) for s in u.src)))
    else:
      out.append(u)
  return out
