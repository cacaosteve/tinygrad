"""RDNA3 VOPD dual-issue pairing (post-regalloc optional emit)."""
from __future__ import annotations
from typing import Callable

from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.amd.dsl import Reg
from tinygrad.renderer.isa.rdna3_defs import (AMDOps, iop as _iop, dst_operand as _dst, src_operand as _src, vgpr_num as _vgpr_num,
  mov_imm_or_src as _mov_imm_or_src, reg_slots as _reg_slots, unwrap_const as _unwrap_const, reg_idxs as _reg_idxs)
from tinygrad.runtime.autogen.amd.rdna3 import ins as r3
from tinygrad.runtime.autogen.amd.rdna3.enum import VOPDOp
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp

def _fmac_mul_src(x:UOp):
  """FMAC multiplier: peel MOV/CAST to a float literal when possible."""
  cur = x
  for _ in range(6):
    if (c:=_unwrap_const(cur)) is not None:
      try: return float(c.val)
      except Exception: return _src(x)
    if cur.op is Ops.CAST and cur.src:
      cur = cur.src[0]
      continue
    if cur.op is Ops.INS and _iop(cur) is AMDOps.MOV and cur.src and cur.src[0].op is not Ops.SPECIAL:
      cur = cur.src[0]
      continue
    break
  return _src(x)

def _fmac_mul_operand_choices(u:UOp) -> list[tuple[object, Reg]]:
  """VOPD-eligible multiplier orientations. Multiplication is commutative."""
  if u.dtype is not dtypes.float32: return []
  a, b = _fmac_mul_src(u.src[1]), _fmac_mul_src(u.src[2])
  out: list[tuple[object, Reg]] = []
  seen: set[tuple[object, int]] = set()
  for src0, src1 in ((a, b), (b, a)):
    if not (isinstance(src1, Reg) and src1.offset >= 256): continue
    num1 = _vgpr_num(src1)
    if num1 is None: continue
    key: tuple[object, int] = (src0 if not isinstance(src0, Reg) else _vgpr_num(src0), num1)
    if key in seen: continue
    seen.add(key)
    out.append((src0, src1))
  return out

def _vopd_src0_ok(a0, a1) -> bool:
  """VOPD src0: even/odd VGPR pair, or the same VGPR/literal broadcast to both lanes."""
  an0 = _vgpr_num(a0) if isinstance(a0, Reg) else None
  an1 = _vgpr_num(a1) if isinstance(a1, Reg) else None
  if an0 is not None and an1 is not None:
    if an0 == an1: return True  # flash scale/alpha broadcast
    return an1 == an0 + 1 and an0 % 2 == 0
  return an0 is None and an1 is None and a0 == a1

def _vopd_src1_ok(b0:Reg, b1:Reg) -> bool:
  bn0, bn1 = _vgpr_num(b0), _vgpr_num(b1)
  return bn0 is not None and bn1 == bn0 + 1 and bn0 % 2 == 0

def depends(u:UOp, dep:UOp) -> bool:
  return dep is u or dep in u.backward_slice

def independent(u0:UOp, u1:UOp) -> bool:
  return not depends(u0, u1) and not depends(u1, u0)

def scan_safe(candidate:UOp, between:list[UOp]) -> bool:
  """A scanned instruction can move only if its physical operands stay untouched."""
  dst, src = _reg_idxs(candidate), set().union(*(_reg_idxs(s) for s in candidate.src))
  for mid in between:
    if mid.op is Ops.INS and _iop(mid) in (AMDOps.LABEL, AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ,
                                          AMDOps.IF_MASK, AMDOps.END_MASK, AMDOps.BARRIER): return False
    reads = set().union(*(_reg_idxs(s) for s in mid.src)) if mid.src else set()
    if dst & (_reg_idxs(mid) | reads) or src & _reg_idxs(mid): return False
  return True

def find_pair(ops:list[UOp], i:int, emitted:set[int]) -> tuple[int, list]|None:
  """Find the first eligible partner, preserving each instruction class's scan policy."""
  u = ops[i]
  if u.op is not Ops.INS: return None
  policies:list[tuple[tuple[Callable[[UOp, UOp], list|None], ...], str, int]] = []
  if _iop(u) is AMDOps.FMAC and getenv("AMD_VOPD_FMAC", 0):
    policies.append(((try_vopd_fmac_pair, try_vopd_fmac_alu_pair), "forward", 8))
  if _iop(u) in (AMDOps.ADD, AMDOps.SUB) and getenv("AMD_VOPD_FMAC_ALU", 0):
    policies.append(((try_vopd_fmac_alu_pair,), "forward", 8))
  if _iop(u) is AMDOps.ADD: policies.append(((try_vopd_add_pair,), "both", 8))
  if _iop(u) is AMDOps.MUL: policies.append(((try_vopd_mul_pair,), "direct", 8))
  if _iop(u) is AMDOps.MAX: policies.append(((try_vopd_max_pair,), "direct", 8))
  if _iop(u) is AMDOps.MOV: policies.append(((try_vopd_mov_pair,), "mov", 32))
  for pairers, guard, window in policies:
    for k in range(i+1, min(i+1+window, len(ops))):
      if k in emitted: continue
      cand, between = ops[k], ops[i+1:k]
      pair = next((p for fn in pairers if (p:=fn(u, cand)) is not None), None)
      if pair is None: continue
      if not scan_safe(cand, between if guard == "mov" else ops[i:k]): continue
      if guard in ("forward", "mov") and any(depends(cand, mid) for mid in between): continue
      if guard == "both" and any(depends(cand, mid) or depends(mid, cand) for mid in between): continue
      if guard == "direct" and any(u in mid.src or cand in mid.src for mid in between): continue
      return k, pair
  return None

def _add_operands(u:UOp) -> tuple[object, object]|None:
  if u.dtype is not dtypes.float32 or len(u.src) < 2: return None
  return _src(u.src[0]), _src(u.src[1])

def try_vopd_fmac_pair(u0:UOp, u1:UOp) -> list|None:
  """Pack two independent float32 FMACs into VOPD when dest/src banks allow."""
  # Default OFF: VOPD FMAC pairing NaNs generic f32 matmul (2026-09-27 HW). Flash decode opts in.
  if not getenv("AMD_VOPD_FMAC", 0): return None
  if u0.op is not Ops.INS or u1.op is not Ops.INS: return None
  if _iop(u0) is not AMDOps.FMAC or _iop(u1) is not AMDOps.FMAC: return None
  if not independent(u0, u1): return None
  ch0, ch1 = _fmac_mul_operand_choices(u0), _fmac_mul_operand_choices(u1)
  if not ch0 or not ch1: return None
  d0, d1 = _dst(u0), _dst(u1)
  n0, n1 = _vgpr_num(d0), _vgpr_num(d1)
  if n0 is not None and n1 is not None and n1 % 2 == 0 and n0 == n1 + 1:
    u0, u1, ch0, ch1, d0, d1, n0, n1 = u1, u0, ch1, ch0, d1, d0, n1, n0
  if n0 is None or n1 != n0 + 1 or n0 % 2 != 0: return None
  for (a0, b0), (a1, b1) in ((x, y) for x in ch0 for y in ch1):
    if not _vopd_src1_ok(b0, b1): continue
    if not _vopd_src0_ok(a0, a1): continue
    return [r3.v_dual_fmac_f32(opy=VOPDOp.V_DUAL_FMAC_F32, vdstx=d0, vdsty=d1,
                               srcx0=a0, srcy0=a1, vsrcx1=b0, vsrcy1=b1)]
  return None

def try_vopd_fmac_alu_pair(u0:UOp, u1:UOp) -> list|None:
  """Pack FMAC with an independent ADD/SUB into mixed VOPD (HIP flash pattern).

  AMD_VOPD_FMAC_MUL removed (hang).
  """
  if not getenv("AMD_VOPD_FMAC_ALU", 0): return None
  if u0.op is not Ops.INS or u1.op is not Ops.INS: return None
  ops = {_iop(u0), _iop(u1)}
  if AMDOps.FMAC not in ops: return None
  alu_ops_ok = (AMDOps.ADD, AMDOps.SUB)
  alu_op = next((o for o in ops if o in alu_ops_ok), None)
  if alu_op is None or len(ops) != 2: return None
  if not independent(u0, u1): return None
  fmac_u, alu_u = (u0, u1) if _iop(u0) is AMDOps.FMAC else (u1, u0)
  if fmac_u.dtype is not dtypes.float32 or alu_u.dtype is not dtypes.float32: return None
  ch = _fmac_mul_operand_choices(fmac_u)
  alu_ops = _add_operands(alu_u)
  if not ch or alu_ops is None: return None
  df, da = _dst(fmac_u), _dst(alu_u)
  if _vgpr_num(df) is None: return None
  a_alu, b_alu = alu_ops
  if not (isinstance(b_alu, Reg) and b_alu.offset >= 256): return None
  if alu_op is AMDOps.ADD: opy = VOPDOp.V_DUAL_ADD_F32
  elif alu_op is AMDOps.SUB: opy = VOPDOp.V_DUAL_SUB_F32
  else: opy = VOPDOp.V_DUAL_MUL_F32
  for a_f, b_f in ch:
    if not (isinstance(b_f, Reg) and b_f.offset >= 256): continue
    return [r3.v_dual_fmac_f32(opy=opy, vdstx=df, vdsty=da,
                               srcx0=a_f, srcy0=a_alu, vsrcx1=b_f, vsrcy1=b_alu)]
  return None

def try_vopd_add_pair(u0:UOp, u1:UOp) -> list|None:
  """Pack two independent float32 ADDs into VOPD when dest/src banks allow."""
  if u0.op is not Ops.INS or u1.op is not Ops.INS: return None
  if _iop(u0) is not AMDOps.ADD or _iop(u1) is not AMDOps.ADD: return None
  if u0.dtype is not dtypes.float32 or u1.dtype is not dtypes.float32: return None
  if not independent(u0, u1): return None
  ops0, ops1 = _add_operands(u0), _add_operands(u1)
  if ops0 is None or ops1 is None: return None
  d0, d1 = _dst(u0), _dst(u1)
  n0, n1 = _vgpr_num(d0), _vgpr_num(d1)
  if n0 is not None and n1 is not None and n1 % 2 == 0 and n0 == n1 + 1:
    u0, u1, ops0, ops1, d0, d1, n0, n1 = u1, u0, ops1, ops0, d1, d0, n1, n0
  if n0 is None or n1 != n0 + 1 or n0 % 2 != 0: return None
  (a0, b0), (a1, b1) = ops0, ops1
  bn0, bn1 = (_vgpr_num(b0) if isinstance(b0, Reg) else None), (_vgpr_num(b1) if isinstance(b1, Reg) else None)
  if bn0 is None or bn1 != bn0 + 1 or bn0 % 2 != 0: return None
  if not _vopd_src0_ok(a0, a1): return None
  return [r3.v_dual_add_f32(opy=VOPDOp.V_DUAL_ADD_F32, vdstx=d0, vdsty=d1,
                            srcx0=a0, srcy0=a1, vsrcx1=b0, vsrcy1=b1)]

def _try_vopd_bin_pair(u0:UOp, u1:UOp, op:AMDOps, emit, opy:VOPDOp) -> list|None:
  """Pack two independent float32 binary ALU ops into VOPD when dest/src banks allow."""
  if u0.op is not Ops.INS or u1.op is not Ops.INS: return None
  if _iop(u0) is not op or _iop(u1) is not op: return None
  if u0.dtype is not dtypes.float32 or u1.dtype is not dtypes.float32: return None
  if u1 in u0.src or u0 in u1.src: return None
  ops0, ops1 = _add_operands(u0), _add_operands(u1)
  if ops0 is None or ops1 is None: return None
  d0, d1 = _dst(u0), _dst(u1)
  n0, n1 = _vgpr_num(d0), _vgpr_num(d1)
  if n0 is not None and n1 is not None and n1 % 2 == 0 and n0 == n1 + 1:
    u0, u1, ops0, ops1, d0, d1, n0, n1 = u1, u0, ops1, ops0, d1, d0, n1, n0
  if n0 is None or n1 != n0 + 1 or n0 % 2 != 0: return None
  (a0, b0), (a1, b1) = ops0, ops1
  bn0, bn1 = (_vgpr_num(b0) if isinstance(b0, Reg) else None), (_vgpr_num(b1) if isinstance(b1, Reg) else None)
  if bn0 is None or bn1 != bn0 + 1 or bn0 % 2 != 0: return None
  if not _vopd_src0_ok(a0, a1): return None
  return [emit(opy=opy, vdstx=d0, vdsty=d1, srcx0=a0, srcy0=a1, vsrcx1=b0, vsrcy1=b1)]

def try_vopd_mul_pair(u0:UOp, u1:UOp) -> list|None:
  return _try_vopd_bin_pair(u0, u1, AMDOps.MUL, r3.v_dual_mul_f32, VOPDOp.V_DUAL_MUL_F32)

def try_vopd_max_pair(u0:UOp, u1:UOp) -> list|None:
  return _try_vopd_bin_pair(u0, u1, AMDOps.MAX, r3.v_dual_max_f32, VOPDOp.V_DUAL_MAX_F32)

def parallel_vmov(moves:list[tuple[Reg, Reg|int|float]]) -> list:
  """Emit MOVs, pairing even/odd dest banks into dual_mov when possible."""
  from tinygrad.renderer.isa.rdna3_defs import TMP_VDATA
  pending = [(dst, src) for dst,src in moves if not isinstance(src, Reg) or dst != src]
  ret: list = []
  while pending:
    if len(pending) >= 2:
      paired = False
      for i0 in range(len(pending)):
        d0, s0 = pending[i0]
        n0 = d0.offset - 256 if d0.sz == 1 and 256 <= d0.offset < 512 else None
        if n0 is None: continue
        for i1 in range(i0 + 1, len(pending)):
          d1, s1 = pending[i1]
          n1 = d1.offset - 256 if d1.sz == 1 and 256 <= d1.offset < 512 else None
          if n1 is None: continue
          if n0 % 2 == 1 and n1 == n0 - 1:
            dx, dy, sx, sy = d1, d0, s1, s0
          elif n1 == n0 + 1 and n0 % 2 == 0:
            dx, dy, sx, sy = d0, d1, s0, s1
          else:
            continue
          if not isinstance(sx, Reg) and not isinstance(sy, Reg) and sx == sy:
            ret.append(r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=dx, vdsty=dy, srcx0=sx, srcy0=sy))
            for i in sorted((i0, i1), reverse=True): pending.pop(i)
            paired = True
            break
          if isinstance(sx, Reg) and isinstance(sy, Reg) and sx.sz == 1 and sy.sz == 1:
            snx = sx.offset - 256 if 256 <= sx.offset < 512 else None
            sny = sy.offset - 256 if 256 <= sy.offset < 512 else None
            if snx is not None and sny == snx:
              ret.append(r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=dx, vdsty=dy, srcx0=sx, srcy0=sy))
              for i in sorted((i0, i1), reverse=True): pending.pop(i)
              paired = True
              break
            if snx is not None and sny == snx + 1 and snx % 2 == 0:
              ret.append(r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=dx, vdsty=dy, srcx0=sx, srcy0=sy))
              for i in sorted((i0, i1), reverse=True): pending.pop(i)
              paired = True
              break
        if paired: break
      if paired: continue
    src_regs = {src for _,src in pending if isinstance(src, Reg)}
    for i,(dst,src) in enumerate(pending):
      if not isinstance(src, Reg) or dst not in src_regs:
        ret.append(r3.v_mov_b32_e32(dst, src))
        pending.pop(i)
        break
    else:
      src = pending[0][1]
      if not isinstance(src, Reg): raise RuntimeError("parallel copy cycle without a register source")
      ret.append(r3.v_mov_b32_e32(TMP_VDATA, src))
      pending = [(dst, TMP_VDATA if isinstance(s, Reg) and s == src else s) for dst,s in pending]
  return ret

def try_vopd_mov_pair(u0:UOp, u1:UOp) -> list|None:
  """Pack two independent scalar MOVs into VOPD (stream copies / const inits)."""
  if u0.op is not Ops.INS or u1.op is not Ops.INS: return None
  if _iop(u0) is not AMDOps.MOV or _iop(u1) is not AMDOps.MOV: return None
  if not u0.src or not u1.src: return None
  if u0.src[0].op is Ops.SPECIAL or u1.src[0].op is Ops.SPECIAL: return None
  if _reg_slots(u0) != 1 or _reg_slots(u1) != 1: return None
  if not independent(u0, u1): return None
  d0, d1 = _dst(u0), _dst(u1)
  n0, n1 = _vgpr_num(d0), _vgpr_num(d1)
  if n0 is not None and n1 is not None and n1 % 2 == 0 and n0 == n1 + 1:
    u0, u1, d0, d1, n0, n1 = u1, u0, d1, d0, n1, n0
  if n0 is None or n1 != n0 + 1 or n0 % 2 != 0: return None
  s0, s1 = _mov_imm_or_src(u0.src[0]), _mov_imm_or_src(u1.src[0])
  if isinstance(s0, Reg) and s0 == d0: return None
  if isinstance(s1, Reg) and s1 == d1: return None
  if not isinstance(s0, Reg) and not isinstance(s1, Reg) and s0 == s1:
    return [r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=d0, vdsty=d1, srcx0=s0, srcy0=s1)]
  if isinstance(s0, Reg) and isinstance(s1, Reg) and s0.sz == 1 and s1.sz == 1:
    sn0, sn1 = _vgpr_num(s0), _vgpr_num(s1)
    if sn0 is not None and sn1 == sn0:
      return [r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=d0, vdsty=d1, srcx0=s0, srcy0=s1)]
    if sn0 is not None and sn1 == sn0 + 1 and sn0 % 2 == 0:
      return [r3.v_dual_mov_b32(opy=VOPDOp.V_DUAL_MOV_B32, vdstx=d0, vdsty=d1, srcx0=s0, srcy0=s1)]
  return None
