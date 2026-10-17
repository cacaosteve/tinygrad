from __future__ import annotations
import math

from tinygrad.device import CompileError
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.helpers import Target, getenv, prod, is_image_shape
from tinygrad.renderer.isa import ISARenderer, IselContext, LinearContext, PreRegAllocContext, Register, greg
from tinygrad.renderer import Renderer
from tinygrad.renderer.amd.dsl import Reg, v, NULL, EXEC, VCC, VCC_LO, Inst
from tinygrad.runtime.autogen.amd.rdna3 import ins as r3
from tinygrad.codegen.decomp.op import fast_idiv
from tinygrad.renderer import tc
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.divandmod import affine_int_bounds
from tinygrad.uop.ops import AxisType, PatternMatcher, UOp, UPat
from tinygrad.renderer.isa.rdna3_defs import (AMDOps, KERNARG_REG, WGID, LID, SGPR, SGPR32, VGPR,
  TMP_VDATA, TMP_VADDR, TMP_BRANCH, TMP_SDATA0, TMP_SDATA1, RegisterPools, register_pools,
  allow_upcast16, unwrap_const as _unwrap_const, const_value as _const_value, tconst as _tconst,
  iop as _iop, elem_count as _elem_count, reg_slots as _reg_slots, reg_idxs as _reg_idxs, const_int as _const_int,
  mem_byte_off as _mem_byte_off, lds_byte_off as _lds_byte_off, load_count_src as _load_count_src,
  pack_f16_is_vec_load as _pack_f16_is_vec_load, pack_f16_is_half2_lo_only as _pack_f16_is_half2_lo_only,
  pack_f16_half2_load as _pack_f16_half2_load, reg_to_amd as _reg_to_amd, reg_lane as _reg_lane,
  src_operand as _src, dst_operand as _dst, mov_imm_or_src as _mov_imm_or_src)
from tinygrad.renderer.isa.rdna3_tc import pm_expand_wmma_lds, pm_stage_wmma_ab

def _register_pools(ctx:IselContext|PreRegAllocContext) -> RegisterPools:
  if (pools:=ctx.scratch.get("register_pools")) is None:
    default_base = 121 if allow_upcast16() or getenv("TC_LDS_AB", 0) else -1
    base = getenv("AMD_WMMA_ACC_BASE", default_base)
    ctx.scratch["register_pools"] = pools = register_pools(base, getenv("AMD_WMMA_ACC_N", 48 if base >= 140 else 128))
  return pools

# RDNA3: kernarg in s[0:1], local ids packed in v0. Even SGPR bases for 64-bit kernarg loads.
# WGID follows USER_SGPR_COUNT: s2 when count=2 (1D locals); s15 when gfx1100 pads to 15 (2D locals).
# COALESCE_U8 / UNIFORM_INT / VOPD_PAIR_CONST / SWIZZLE_BATCH_MAX hardcoded ON/16.
_FMAC_DELAY_N = [0]
# AMD_INSTR_WAIT / AMD_CONSERVATIVE_WAIT / AMD_IN_ORDER_EMIT removed (always optional emit).

_F32_UNARY = {AMDOps.RECIPROCAL: r3.v_rcp_f32_e32, AMDOps.EXP2: r3.v_exp_f32_e32, AMDOps.LOG2: r3.v_log_f32_e32,
              AMDOps.SQRT: r3.v_sqrt_f32_e32, AMDOps.TRUNC: r3.v_trunc_f32_e32}
_ISEL_UNARY = {Ops.RECIPROCAL: AMDOps.RECIPROCAL, Ops.EXP2: AMDOps.EXP2, Ops.LOG2: AMDOps.LOG2, Ops.SQRT: AMDOps.SQRT,
               Ops.TRUNC: AMDOps.TRUNC, Ops.SIN: AMDOps.SIN}

def _mem_itemsize(dt:DType) -> int: return dt.itemsize
def _parallel_vmov(moves:list[tuple[Reg, Reg|int|float]]) -> list:
  return _vopd.parallel_vmov(moves)

# --- emit: VOPD pairing (post-regalloc optional dual-issue) ---
from tinygrad.renderer.isa import rdna3_vopd as _vopd
_try_vopd_fmac_pair = _vopd.try_vopd_fmac_pair
_try_vopd_fmac_alu_pair = _vopd.try_vopd_fmac_alu_pair
_try_vopd_add_pair = _vopd.try_vopd_add_pair
_try_vopd_mul_pair = _vopd.try_vopd_mul_pair
_try_vopd_max_pair = _vopd.try_vopd_max_pair

from tinygrad.renderer.isa import rdna3_schedule as _sched

def _try_vopd_mov_pair(u0:UOp, u1:UOp) -> list|None:
  return _vopd.try_vopd_mov_pair(u0, u1)

def _full_src(x:UOp) -> Reg:
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg src {x}")
  return _reg_to_amd(greg(x), _reg_slots(x))

def _reg_chunk(reg:Register, off:int, slots:int) -> Reg:
  return _reg_to_amd(Register(reg.name, reg.index+off, _cons=reg.cons), slots)

def _global_load_insts(u:UOp, addr:Reg, byte_off:int=0) -> list:
  slots, sc = _reg_slots(u), u.dtype
  saddr = _src(u.src[0])
  off_kw = {"offset": byte_off} if 0 < byte_off <= 0xfff else {}
  # Volatile loads mark their count operand in _load_ins: by emit, the PARAM has
  # already become a KERNARG. The count is internal LOAD metadata, not an input value.
  cache_kw = {"glc": 1, "dlc": 1} if len(u.src) > 2 and u.src[2].tag == "volatile" else {}
  if slots == 1:
    if (load:=_global_load(u.dtype, _elem_count(u))) is None: raise CompileError(f"no global load {u.dtype}")
    return [load(_dst(u), addr, saddr=saddr, **off_kw, **cache_kw)]
  # multi-VGPR by byte width: float×2/×4, half×4/×8, or byte×8/×16 → B64/B128.
  if sc not in (dtypes.uint8, dtypes.float16, dtypes.float32, dtypes.int32, dtypes.uint32):
    raise CompileError(f"no vec global load {u.dtype}")
  if not isinstance(greg(u), Register): raise CompileError(f"expected reg dst {u}")
  if slots == 2: return [r3.global_load_b64(_dst(u), addr, saddr=saddr, **cache_kw)]
  if slots == 3: return [r3.global_load_b96(_dst(u), addr, saddr=saddr, **cache_kw)]
  if slots == 4: return [r3.global_load_b128(_dst(u), addr, saddr=saddr, **cache_kw)]
  if slots == 8:
    # offset+16 keeps addr live (no TMP bump). Emit clusters consecutive B128s into one
    # s_clause (LLVM-style burst); nested per-tile s_clause(1) would split that.
    return [r3.global_load_b128(_reg_chunk(greg(u), 0, 4), addr, saddr=saddr, **cache_kw),
            r3.global_load_b128(_reg_chunk(greg(u), 4, 4), addr, saddr=saddr, offset=16, **cache_kw)]
  raise CompileError(f"no global load {u.dtype}")

def _b_page_idx(base:UOp, page:int, itemsize:int) -> UOp:
  # Always ADD (incl. page 0) so the page VGPR is distinct from raw `base` (safe in-place <<1).
  # UOp interning already shares identical page expressions, including their element size.
  return base + (page * (0x1000 // itemsize))

def _apply_byte_off(addr:Reg, byte_off:int, idx:UOp|None=None, itemsize:int=1) -> tuple[list, Reg, int]:
  """GLOBAL offset is 0..4095; peel larger byte_off into addr via v_lshl_add / v_add."""
  if byte_off <= 0: return [], addr, 0
  if byte_off <= 0xfff: return [], addr, byte_off
  if idx is not None and itemsize in (2, 4):
    # (elem_base << shift) + byte_off — one op, avoids ADD-then-LSHL + addr VGPRs.
    return [r3.v_lshl_add_u32(TMP_VADDR, _src(idx), itemsize.bit_length() - 1, byte_off)], TMP_VADDR, 0
  return [r3.v_add_nc_u32_e64(TMP_VADDR, byte_off, addr)], TMP_VADDR, 0

class _StoreAddrCache:
  """CSE C-store bases into 4096-byte pages (one scale/add per page).

  Also CSE LDS scales when slot bases are folded into 16-bit ds_offset
  (AMD_LDS_ADDR_CSE): one v_lshlrev per (base,idx,itemsize) streak.
  """
  __slots__ = ("key", "page", "_bcast_src", "lds_key")
  def __init__(self): self.clear()
  def clear(self): self.key, self.page, self._bcast_src, self.lds_key = None, None, None, None
  def addr(self, idx:UOp, itemsize:int, byte_off:int, base_key:object=None) -> tuple[list, Reg, int]:
    src = _src(idx)
    # Key by the logical index, not its allocated register. Regalloc can reuse one
    # VGPR for different indices between stores; treating that as the same base
    # reuses a stale TMP_VADDR and writes the later value to the wrong element.
    # Const indices (flash REG init peel → fresh _tconst(0) each time) key by value.
    # base_key separates scratch REG buffers that share a const-0 index.
    ikey = ("c", _const_int(idx)) if _const_int(idx) is not None else ("u", id(idx))
    key = (ikey, itemsize, base_key)
    page, rem = divmod(max(byte_off, 0), 0x1000)
    if self.key == key and self.page == page: return [], TMP_VADDR, rem
    if itemsize == 1:
      pre:list = [r3.v_mov_b32_e32(TMP_VADDR, src)] if page == 0 else \
                 [r3.v_add_nc_u32_e64(TMP_VADDR, page * 0x1000, src)]
    elif itemsize in (2, 4):
      shift = itemsize.bit_length() - 1
      pre = [r3.v_lshl_add_u32(TMP_VADDR, src, shift, page * 0x1000)] if page else \
            [r3.v_lshlrev_b32_e64(TMP_VADDR, shift, src)]
    else:
      raise CompileError(f"bad addr scale {itemsize}")
    self.key, self.page, self.lds_key = key, page, None
    return pre, TMP_VADDR, rem
  def lds_addr(self, idx:UOp, itemsize:int, total_off:int, base_key:object=None) -> tuple[list, Reg, int]|None:
    """CSE LDS scale; total_off is the full ds_offset (base folded in). None if overflow."""
    if total_off > 0xffff or itemsize not in (1, 2, 4): return None
    ikey = ("c", _const_int(idx)) if _const_int(idx) is not None else ("u", id(idx))
    key = ("lds", ikey, itemsize, base_key)
    if self.lds_key == key: return [], TMP_VADDR, total_off
    pre, addr = _scaled_addr(TMP_VADDR, idx, itemsize)
    self.lds_key, self.key = key, None  # TMP_VADDR shared with scratch CSE
    return pre, addr, total_off

def _global_store_insts(u:UOp, addr:Reg, byte_off:int=0) -> list:
  val = u.src[2]
  slots, sc = _reg_slots(val), val.dtype
  saddr = _src(u.src[0])
  # byte_off already clamped to 0..4095 by _apply_byte_off; larger went through v_lshl_add/v_add.
  off_kw = {"offset": byte_off} if 0 < byte_off <= 0xfff else {}
  if slots == 1:
    if (store:=_global_store(val.dtype, _elem_count(val))) is None: raise CompileError(f"no global store {val.dtype}")
    dpre, data = _vgpr_data(TMP_VDATA, val)
    return dpre + [store(addr=addr, data=data, saddr=saddr, **off_kw)]
  if sc not in (dtypes.float16, dtypes.float32): raise CompileError(f"no vec global store {val.dtype}")
  if not isinstance(greg(val), Register): raise CompileError(f"expected reg src {val}")
  if slots == 2: return [r3.global_store_b64(addr=addr, data=_full_src(val), saddr=saddr, **off_kw)]
  if slots == 3: return [r3.global_store_b96(addr=addr, data=_full_src(val), saddr=saddr, **off_kw)]
  if slots == 4: return [r3.global_store_b128(addr=addr, data=_full_src(val), saddr=saddr, **off_kw)]
  if slots == 8:
    lo, hi = _reg_chunk(greg(val), 0, 4), _reg_chunk(greg(val), 4, 4)
    if off_kw:
      o = off_kw["offset"]
      if o + 16 <= 0xfff:
        return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr, offset=o),
                r3.global_store_b128(addr=addr, data=hi, saddr=saddr, offset=o+16)]
      # Second b128 bumps TMP_VADDR — caller must drop store-addr CSE.
      return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr, offset=o),
              r3.v_add_nc_u32_e64(TMP_VADDR, 16, addr),
              r3.global_store_b128(addr=TMP_VADDR, data=hi, saddr=saddr, offset=o)]
    return [r3.global_store_b128(addr=addr, data=lo, saddr=saddr),
            r3.global_store_b128(addr=addr, data=hi, saddr=saddr, offset=16)]
  raise CompileError(f"no global store {val.dtype}")

def _vopd_scan_safe(candidate:UOp, between:list[UOp]) -> bool:
  """A scanned instruction can move only if its physical operands stay untouched."""
  dst, src = _reg_idxs(candidate), set().union(*(_reg_idxs(s) for s in candidate.src))
  for mid in between:
    if mid.op is Ops.INS and _iop(mid) in (AMDOps.LABEL, AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ,
                                          AMDOps.IF_MASK, AMDOps.END_MASK, AMDOps.BARRIER): return False
    reads = set().union(*(_reg_idxs(s) for s in mid.src)) if mid.src else set()
    if dst & (_reg_idxs(mid) | reads) or src & _reg_idxs(mid): return False
  return True

def _wait_for_domain(domain:str, cnt:int=0):
  if domain == "vm": return r3.s_waitcnt_vmcnt(sdst=NULL, simm16=cnt)
  if domain == "lgkm": return r3.s_waitcnt_lgkmcnt(sdst=NULL, simm16=cnt)
  if domain == "vs": return r3.s_waitcnt_vscnt(sdst=NULL, simm16=cnt)
  raise CompileError(f"unknown wait domain {domain}")

def _wait_domain_for_load(u:UOp) -> str|None:
  if u.op is not Ops.INS: return None
  if _iop(u) in (AMDOps.LOAD, AMDOps.SLOAD, AMDOps.FILL): return "vm"
  if _iop(u) in (AMDOps.KERNARG, AMDOps.LLOAD, AMDOps.SWIZZLE): return "lgkm"
  return None

def _wait_domain_for_store(u:UOp) -> str|None:
  # RDNA3: vector store completion is vscnt. Track global + scratch stores for scoreboard
  # flush (hand-kernel style: burst stores, one wait before use). LDS stores use lgkm.
  if u.op is not Ops.INS: return None
  if _iop(u) in (AMDOps.STORE, AMDOps.SSTORE): return "vs"
  if _iop(u) is AMDOps.LSTORE: return "lgkm"
  return None

def _store_src_regs(u:UOp) -> set[int]:
  # Sentinel: any outstanding global/LDS/scratch store. Do not scoreboard TMP_VADDR — addr is sampled at issue.
  if _iop(u) in (AMDOps.STORE, AMDOps.LSTORE, AMDOps.SSTORE): return {-1}
  return set()

def _needs_vm_flush(u:UOp) -> bool:
  # Packs/extracts have their own emitted-instruction dependency check below. General
  # ALU must enter flush_regs: independent address ALU still overlaps VMEM, while an
  # integer consumer of a dest-as-address LOAD must wait before reading that VGPR.
  if u.op is not Ops.INS: return False
  if _iop(u) in (AMDOps.WMMA, AMDOps.STORE, AMDOps.ATOMIC_ADD, AMDOps.SSTORE, AMDOps.SPILL): return True
  if _iop(u) in (AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV): return False
  if _iop(u) in (AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.ADD, AMDOps.SUB, AMDOps.MUL,
               AMDOps.CMP_GE, AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ):
    return True
  return True

_SCALAR_LOAD = {
  dtypes.bool: (r3.global_load_u8, r3.scratch_load_u8, r3.ds_load_u8),
  dtypes.uint8: (r3.global_load_u8, r3.scratch_load_u8, r3.ds_load_u8),
  dtypes.int8: (r3.global_load_i8, r3.scratch_load_i8, r3.ds_load_i8),
  dtypes.uint16: (r3.global_load_u16, r3.scratch_load_u16, r3.ds_load_u16),
  dtypes.int16: (r3.global_load_i16, r3.scratch_load_i16, r3.ds_load_i16),
  dtypes.float16: (r3.global_load_u16, r3.scratch_load_u16, r3.ds_load_u16),
  dtypes.uint32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
  dtypes.int32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
  dtypes.float32: (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32),
}
_SCALAR_STORE = {
  dtypes.bool: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.uint8: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.int8: (r3.global_store_b8, r3.scratch_store_b8, r3.ds_store_b8),
  dtypes.uint16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.int16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.float16: (r3.global_store_b16, r3.scratch_store_b16, r3.ds_store_b16),
  dtypes.uint32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
  dtypes.int32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
  dtypes.float32: (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32),
}
_WIDE_LOAD = {8: (r3.global_load_b64, r3.ds_load_b64), 12: (r3.global_load_b96, r3.ds_load_b96),
              16: (r3.global_load_b128, r3.ds_load_b128)}
_WIDE_STORE = {8: (r3.global_store_b64, r3.ds_store_b64), 12: (r3.global_store_b96, r3.ds_store_b96),
               16: (r3.global_store_b128, r3.ds_store_b128)}

def _mem_load(kind:int, dt:DType, n:int=1):
  if n > 1:
    nbytes = dt.itemsize * n
    if dt is dtypes.uint8 and kind == 0:
      # uchar×2 → u16 (scale halfwords); ×4 → b32; ×8/16 → b64/b128.
      if nbytes == 2: return r3.global_load_u16
      if nbytes == 4: return r3.global_load_b32
      return _WIDE_LOAD.get(nbytes, (None, None))[0]
    # half×2 → B32; half×4/×8 → B64/B128 for global+LDS (PACK_F16 clobber fixed; gated stays scalar).
    # scratch stays half2 — no wide scratch half path yet.
    if dt is dtypes.float16:
      if nbytes == 4: return (r3.global_load_b32, r3.scratch_load_b32, r3.ds_load_b32)[kind]
      if kind != 1 and nbytes in _WIDE_LOAD: return _WIDE_LOAD[nbytes][0 if kind == 0 else 1]
      return None
    if dt in (dtypes.int32, dtypes.uint32):
      wide_ops = _WIDE_LOAD.get(nbytes)
      return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
    if dt is not dtypes.float32: return None
    wide_ops = _WIDE_LOAD.get(nbytes)
    return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
  scalar_ops = _SCALAR_LOAD.get(dt)
  return None if scalar_ops is None else scalar_ops[kind]

def _mem_store(kind:int, dt:DType, n:int=1):
  if n > 1:
    nbytes = dt.itemsize * n
    if dt is dtypes.float16:
      if nbytes == 4: return (r3.global_store_b32, r3.scratch_store_b32, r3.ds_store_b32)[kind]
      if kind != 1 and nbytes in _WIDE_STORE: return _WIDE_STORE[nbytes][0 if kind == 0 else 1]
      return None
    # Match wide loads: int/uint32×2/×3/×4 → b64/b96/b128 (IQ4 LUT LDS fill; HIP ds_store_b128).
    if dt in (dtypes.int32, dtypes.uint32):
      if kind == 1: return None  # no wide scratch int path yet
      wide_ops = _WIDE_STORE.get(nbytes)
      return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
    if dt is not dtypes.float32: return None
    wide_ops = _WIDE_STORE.get(nbytes)
    return None if wide_ops is None else wide_ops[0 if kind == 0 else 1]
  scalar_ops = _SCALAR_STORE.get(dt)
  return None if scalar_ops is None else scalar_ops[kind]

def _global_load(dt:DType, n:int=1): return _mem_load(0, dt, n)
def _global_store(dt:DType, n:int=1): return _mem_store(0, dt, n)
def _scratch_load(dt:DType, n:int=1): return _mem_load(1, dt, n)
def _scratch_store(dt:DType, n:int=1): return _mem_store(1, dt, n)
def _local_load(dt:DType, n:int=1): return _mem_load(2, dt, n)
def _local_store(dt:DType, n:int=1): return _mem_store(2, dt, n)

def _scaled_addr(dst:Reg, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  src = _src(idx)
  if itemsize == 1:
    return ([], src) if isinstance(src, Reg) and src.offset >= 256 else ([r3.v_mov_b32_e32(dst, src)], dst)
  if itemsize not in (2, 4): raise CompileError(f"bad addr scale {itemsize}")
  return [r3.v_lshlrev_b32_e64(dst, itemsize.bit_length()-1, src)], dst

def _masked_addr(pre:list, addr:Reg, masked:bool) -> tuple[list, Reg]:
  if not masked or addr.offset < 256: return pre, addr
  return pre + [r3.v_cndmask_b32_e32(TMP_VADDR, 0, addr)], TMP_VADDR

def _vgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [], src
  return [r3.v_mov_b32_e32(tmp, src)], tmp

def _sgpr_data(tmp:Reg, data:UOp) -> tuple[list, Reg|int|float]:
  src = _src(data)
  if isinstance(src, Reg) and src.offset >= 256: return [r3.v_readfirstlane_b32_e32(tmp, src)], tmp
  return [], src

def _buf_ref(x:UOp, lds:bool) -> bool:
  if x.op is Ops.AFTER: return _buf_ref(x.src[0], lds)
  # Ops.ALLOC is the post-#18316 REG/LOCAL temp; treat like BUFFER for scratch/LDS routing.
  if lds: return (x.op in (Ops.BUFFER, Ops.ALLOC) and x.addrspace is AddrSpace.LOCAL) or \
                 (x.op is Ops.INS and _iop(x) is AMDOps.LDS_BASE)
  return (x.op in (Ops.BUFFER, Ops.ALLOC) and x.addrspace is AddrSpace.REG) or \
         (x.op is Ops.INS and _iop(x) is AMDOps.SCRATCH_ADDR)
def _is_lds_ref(x:UOp) -> bool: return _buf_ref(x, True)
def _is_scratch_ref(x:UOp) -> bool: return _buf_ref(x, False)

def _lds_itemsize(x:UOp) -> int:
  return x.dtype.itemsize

def _lds_size_bytes(x:UOp) -> int:
  return x.max_numel() * x.dtype.itemsize

def _align(x:int, a:int) -> int: return x + (-x % a)

def _lds_offsets(ctx:IselContext) -> dict[int, int]:
  if (offsets:=ctx.scratch.get("lds_offsets")) is None:
    # Slot identity is physical allocation identity: logical buffers may share a slot across disjoint lifetimes.
    layouts: dict[int, tuple[int, int]] = {}
    for b in [u for u in ctx.uses if u.op in (Ops.BUFFER, Ops.ALLOC) and u.addrspace is AddrSpace.LOCAL]:
      align, size = layouts.get(b.arg.slot, (1, 0))
      layouts[b.arg.slot] = (max(align, _lds_itemsize(b)), max(size, _lds_size_bytes(b)))
    offsets, off = {}, 0
    for slot,(align,size) in sorted(layouts.items()):
      off = _align(off, align)
      offsets[slot] = off
      off += size
    ctx.scratch["lds_offsets"] = offsets
  return offsets

def _lds_base(ctx:IselContext, x:UOp) -> UOp|None:
  if x.addrspace is not AddrSpace.LOCAL: return None
  return UOp(Ops.INS,
             src=(_tconst(_lds_size_bytes(x), dtypes.int32).rtag(), _tconst(_lds_offsets(ctx)[x.arg.slot], dtypes.int32).rtag()),
             arg=(AMDOps.LDS_BASE, dtypes.uint32))

def _lds_base_offset(x:UOp) -> int:
  if x.op is Ops.AFTER: return _lds_base_offset(x.src[0])
  if x.op is Ops.INS and _iop(x) is AMDOps.LDS_BASE: return _const_int(x.src[1]) or 0
  return 0

def _scratch_base_offset(x:UOp) -> int:
  if x.op is Ops.AFTER: return _scratch_base_offset(x.src[0])
  if x.op is Ops.INS and _iop(x) is AMDOps.SCRATCH_ADDR: return _const_int(x.src[0]) or 0
  return 0

def _local_addr(base:UOp, idx:UOp, itemsize:int, addr_dst:Reg|None=None) -> tuple[list, Reg, int]:
  """Build LDS VGPR addr. Returns (pre, addr, base_byte_off_to_fold).

  When base_byte_off_to_fold != 0, the caller should add it into ds_offset if the sum
  fits in 16 bits — avoids a v_add_nc per access (HIP folds slot bases into offset).
  Gate: AMD_LDS_FOLD_BASE (default ON).
  """
  dst = addr_dst if addr_dst is not None else TMP_VADDR
  base_off = _lds_base_offset(base)
  if base_off:
    pre, addr = _scaled_addr(dst, idx, itemsize)
    if addr_dst is not None and addr != dst:
      pre, addr = pre + [r3.v_mov_b32_e32(dst, addr)], dst
    return pre, addr, base_off
  # No foldable slot base: scale idx into bytes.
  pre, addr = _scaled_addr(dst, idx, itemsize)
  if addr_dst is not None and addr != dst:
    pre, addr = pre + [r3.v_mov_b32_e32(dst, addr)], dst
  return pre, addr, 0

def _lds_emit_addr(base:UOp, idx:UOp, itemsize:int, elem_off:int=0,
                   addr_dst:Reg|None=None) -> tuple[list, Reg, int]:
  """LDS addr for emit: fold slot base into ds_offset when ≤0xffff."""
  pre, addr, base_off = _local_addr(base, idx, itemsize, addr_dst)
  total = base_off + elem_off
  if base_off and total > 0xffff:
    dst = addr_dst if addr_dst is not None else TMP_VADDR
    # Overflow ds_offset — bake base into VGPR via lshl_add when we only scaled.
    if itemsize in (2, 4) and len(pre) <= 1:
      return [r3.v_lshl_add_u32(dst, _src(idx), itemsize.bit_length() - 1, base_off)], dst, elem_off
    return pre + [r3.v_add_nc_u32_e64(dst, base_off, addr)], dst, elem_off
  return pre, addr, total

def _scratch_addr(base:UOp, idx:UOp, itemsize:int) -> tuple[list, Reg]:
  pre, addr = _scaled_addr(TMP_VADDR, idx, itemsize)
  if (off:=_scratch_base_offset(base)) == 0: return pre, addr
  return pre + [r3.v_add_nc_u32_e64(TMP_VADDR, off, addr)], TMP_VADDR

def _reg_buffer_base(x:UOp) -> UOp|None:
  if x.op is Ops.AFTER: return _reg_buffer_base(x.src[0])
  # Placeholders are RESHAPE(BUFFER); peel so promote/skip see the REG BUFFER.
  if x.op in (Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.PAD) and x.src:
    return _reg_buffer_base(x.src[0])
  return x if x.op in (Ops.BUFFER, Ops.ALLOC) and x.addrspace is AddrSpace.REG else None

def _reg_mem_key(base:UOp, idx:UOp, byte_off:int=0, itemsize:int=4) -> tuple[UOp, int]|None:
  # Include peeled SCRATCH imm offsets so const-fold to idx=0 + byte_off stays distinct.
  if (buf:=_reg_buffer_base(base)) is None: return None
  if (off:=_const_int(idx)) is None or off < 0: return None
  if itemsize <= 0 or byte_off % itemsize: return None
  return buf, off + byte_off // itemsize

def _is_zero_val(val:UOp) -> bool:
  if (c:=_unwrap_const(val)) is not None: return c.val == 0
  if val.op is Ops.INS and _iop(val) is AMDOps.MOV and val.src and (c:=_unwrap_const(val.src[0])) is not None:
    return c.val == 0
  return False

def _is_identity_sload(val:UOp, key:tuple[UOp, int]) -> bool:
  if val.op is not Ops.INS or _iop(val) is not AMDOps.SLOAD: return False
  return _reg_mem_key(val.src[0], val.src[1], _lds_byte_off(val), val.dtype.itemsize) == key

def _is_identity_load(val:UOp, addr:UOp) -> bool:
  if val.op is not Ops.LOAD or not val.src: return False
  load_addr = val.src[0]
  if load_addr.op not in (Ops.INDEX, Ops.SHRINK) or addr.op not in (Ops.INDEX, Ops.SHRINK): return False
  return _reg_mem_key(load_addr.src[0], load_addr.src[1]) == _reg_mem_key(addr.src[0], addr.src[1])

def _compute_amd_skip(uops:list[UOp]) -> set[UOp]:
  buffer_offset_stores: dict[tuple[UOp, int], list[tuple[UOp, bool, bool]]] = {}
  # A dynamic REG access can alias every constant offset in its buffer. In particular,
  # hand WMMA kernels zero each accumulator slot with constant stores, then access the
  # tile through loop-varying indices. Those zero stores are live across outer loops.
  dynamic_buffers = {buf for u in uops if u.op is Ops.INS and _iop(u) in (AMDOps.SLOAD, AMDOps.SSTORE) and len(u.src) >= 2 and
                     (buf:=_reg_buffer_base(u.src[0])) is not None and _const_int(u.src[1]) is None}
  for u in uops:
    if u.op is Ops.INS and _iop(u) is AMDOps.SSTORE and len(u.src) >= 3:
      if (key:=_reg_mem_key(u.src[0], u.src[1], _lds_byte_off(u), u.src[2].dtype.itemsize)) is None: continue
      val = u.src[2]
      is_identity, is_zero = _is_identity_sload(val, key), _is_zero_val(val)
    elif u.op is Ops.STORE and len(u.src) >= 2:
      addr = u.src[0]
      if addr.op not in (Ops.INDEX, Ops.SHRINK): continue
      if (key:=_reg_mem_key(addr.src[0], addr.src[1])) is None: continue
      val = u.src[1]
      is_identity, is_zero = _is_identity_load(val, addr), _is_zero_val(val)
    else: continue
    buffer_offset_stores.setdefault(key, []).append((u, is_identity, is_zero))
  dead_offsets = {key for key, stores in buffer_offset_stores.items()
                  if key[0] not in dynamic_buffers and all(is_identity or is_zero for _, is_identity, is_zero in stores)}
  skip: set[UOp] = set()
  identity_loads: set[UOp] = set()
  for key, stores in buffer_offset_stores.items():
    for store, is_identity, is_zero in stores:
      if is_identity:
        skip.add(store)
        val = store.src[2] if store.op is Ops.INS else store.src[1]
        if val.op is Ops.INS and _iop(val) is AMDOps.SLOAD: identity_loads.add(val)
        elif val.op is Ops.LOAD: identity_loads.add(val)
      elif key in dead_offsets and is_zero: skip.add(store)
  for u in uops:
    for src in u.src:
      if src in identity_loads and u not in skip: identity_loads.discard(src)
  skip |= identity_loads
  # SWHERE rematerializes a wave-uniform comparison with SCC. If that is the
  # comparison's only purpose, suppress its earlier VCC form (and the SGPR→VGPR
  # materialization it would otherwise emit).
  users: dict[UOp, list[UOp]] = {}
  for u in uops:
    for src in u.src: users.setdefault(src, []).append(u)
  for u in uops:
    if u.op is Ops.INS and _iop(u) in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) and \
       (us:=users.get(u)) and all(x.op is Ops.INS and _iop(x) is AMDOps.SWHERE and x.src[0] is u for x in us):
      skip.add(u)
  return skip

def _is_f16_to_f32_cast(u:UOp) -> bool:
  return _sched._is_f16_to_f32_cast(u)

def _fma_mix_f32_folds(uops:list[UOp]) -> tuple[dict[UOp, tuple[UOp, UOp]], set[UOp]]:
  return _sched.fma_mix_f32_folds(uops)

def _lower_fma_mix_f32(lst:list[UOp]) -> list[UOp]:
  return _sched.lower_fma_mix_f32(lst)

def _amd_skip(ctx:PreRegAllocContext) -> set[UOp]:
  if "skip" not in ctx.scratch and ctx.uops: ctx.scratch["skip"] = _compute_amd_skip(ctx.uops)
  return ctx.scratch.get("skip") or set()

def _strength_reduce_int_muls(ops:list[UOp]) -> list[UOp]:
  return _sched.strength_reduce_int_muls(ops)

def _u32_high_bits_clear(src:UOp, bits:int) -> bool:
  """True when src is already in [0, 2^bits) so a narrowing AND is a no-op (Q5 scale f16 extract)."""
  mask = (1 << bits) - 1
  if src.op is not Ops.INS: return False
  if _iop(src) is AMDOps.AND:
    for a, b in ((src.src[0], src.src[1]), (src.src[1], src.src[0])):
      if (m := _const_int(b)) is not None and m == mask: return True
  if _iop(src) is AMDOps.SHR:
    if (sh := _const_int(src.src[1])) is not None and sh >= 32 - bits: return True
  return False

def _is_wmma_acc_reload_pack(cin:UOp, ctx:PreRegAllocContext|None=None) -> bool:
  if cin.op is not Ops.INS or _iop(cin) is not AMDOps.PACK or len(cin.src) != 8: return False
  if all(s.op is Ops.INS and _iop(s) is AMDOps.SLOAD for s in cin.src): return True
  # LDS product-16: cin is PACK of zero MOVs (register path uses SLOAD reload).
  if all(s.op is Ops.INS and _iop(s) is AMDOps.MOV and s.src and s.src[0].dtype is dtypes.float and
         _const_value(s.src[0]) == 0.0 for s in cin.src):
    return True
  # SLOAD may already have been rewritten to EXTRACT from the zero-init packs
  if ctx is not None and all(s.op is Ops.INS and _iop(s) is AMDOps.EXTRACT for s in cin.src):
    tiles = ctx.scratch.get("wmma_acc_tiles") or {}
    tile_inits = set(tiles.values())
    return bool(tile_inits) and all(s.src[0] in tile_inits for s in cin.src)
  return False

def _flash_acc_small_enabled() -> bool:
  """≤64 ACC parking only via explicit env — never from FLASH_DIRECT alone.

  FLASH_DIRECT in the process env must not park every ≤64 matmul (breaks eye/GEMM).
  flash_attention sets AMD_FLASH_ACC_SMALL around its realize when ACC_SMALL is on.
  AMD_WMMA_ACC_SMALL forces parking for any ≤64 tile (unsafe for Q4/Q6; avoid globally).
  """
  return bool(getenv("AMD_WMMA_ACC_SMALL", 0) or getenv("AMD_FLASH_ACC_SMALL", 0))

def _wmma_acc_buffers(ctx:PreRegAllocContext) -> set[UOp]:
  """REG buffers whose scalar traffic can stay resident in WMMA accumulator fragments."""
  if (cached:=ctx.scratch.get("wmma_acc_buffers")) is not None: return cached
  bufs: set[UOp] = set()
  packed_quant = True and any(
    u.op is Ops.INS and _iop(u) in (AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16) for u in (ctx.uops or []))
  allow_small = _flash_acc_small_enabled()
  for u in ctx.uops or []:
    if u.op is not Ops.INS or _iop(u) is not AMDOps.WMMA: continue
    pack = u.src[0]
    if not _is_wmma_acc_reload_pack(pack): continue
    for slot in pack.src:
      if slot.op is Ops.INS and _iop(slot) is AMDOps.SLOAD:
        if (base:=_reg_buffer_base(slot.src[0])) is None: continue
        if base.max_numel() <= 128 and (base.max_numel() > 64 or packed_quant or allow_small):
          bufs.add(base)
  # LDS zero-cin path: packs are MOV zeros, so discover oversized REG via SLOAD/SSTORE traffic.
  if not bufs and any(u.op is Ops.INS and _iop(u) is AMDOps.WMMA and _is_wmma_acc_reload_pack(u.src[0])
                      for u in (ctx.uops or []) if u.src):
    for u in ctx.uops or []:
      if u.op is not Ops.INS or _iop(u) not in (AMDOps.SLOAD, AMDOps.SSTORE): continue
      if (base:=_reg_buffer_base(u.src[0])) is None: continue
      if base.max_numel() <= 128 and (base.max_numel() > 64 or packed_quant or allow_small):
        bufs.add(base)
  ctx.scratch["wmma_acc_buffers"] = bufs
  return bufs

def _wmma_acc_small_mode(ctx:PreRegAllocContext|None=None) -> bool:
  """Flash-style ≤64 ACC parking (explicit env only; ctx unused, kept for call sites)."""
  return _flash_acc_small_enabled()

def _wmma_slot_tile_lane(idx:int) -> tuple[int, int]:
  # 4×4 UPCAST packs floats as tile=(idx//32)*4+(idx%4), lane=(idx%32)//4
  return (idx // 32) * 4 + (idx % 4), (idx % 32) // 4

def _wmma_linear_tile_lane(idx:int, numel:int) -> tuple[int, int]|None:
  """Flash-style S_reg/pv_acc: reshape(TM=8, N).permute → ACC tile=col, lane=row."""
  if numel < 8 or numel % 8: return None
  n_tiles = numel // 8
  return idx % n_tiles, idx // n_tiles

def _wmma_acc_lane(ctx:PreRegAllocContext, buf:UOp, idx:int) -> tuple[UOp, int]|None:
  """Map a const REG index on a WMMA ACC buffer to (zero-init PACK, lane)."""
  idx_map = ctx.scratch.get("wmma_acc_idx_map") or {}
  if (got:=idx_map.get((buf, idx), idx_map.get(idx))) is not None: return got
  if _wmma_acc_small_mode(ctx) and (lin:=_wmma_linear_tile_lane(idx, buf.max_numel())) is not None:
    tile_local, lane = lin
    if (init:=(ctx.scratch.get("wmma_acc_buf_tiles") or {}).get(buf, {}).get(tile_local)) is not None:
      return init, lane
  tile, lane = _wmma_slot_tile_lane(idx)
  if (init:=(ctx.scratch.get("wmma_acc_tiles") or {}).get(tile)) is None: return None
  return init, lane

def _wmma_acc_extract(ctx:PreRegAllocContext, init:UOp, lane:int) -> UOp:
  n = ctx.scratch.get("wmma_ext_n", 0)
  ctx.scratch["wmma_ext_n"] = n + 1
  return UOp(Ops.INS, src=(init, _tconst(lane, dtypes.int32).rtag()), arg=(AMDOps.EXTRACT, dtypes.float32),
             tag=(Register(f"wmma_ext{n}", 0, _cons=VGPR),))

def _wmma_acc_zero_inits(uops:list[UOp]) -> tuple[list[UOp], dict[int, UOp], dict[int|tuple[UOp, int], tuple[UOp, int]], dict[UOp, dict[int, UOp]]]:
  """Zero-init WMMA ACC packs before the K-loop.

  Returns (inits, tile->init, reg_idx-or-(buffer,reg_idx)->(init,lane), buf->tile_local->init).
  tile->init uses the 4×4 interleaved formula. Consecutive product-16 SLOAD packs collide
  on first-idx tile keys (4 keys for 16 packs), so epilogue SLOADs use reg_idx->init.
  Flash ACC_SMALL parks multiple ≤64 buffers; linear tile lookups must use buf_tiles, not the
  global expand-order counter (S_reg tile0 would collide with pv_acc tile0).
  """
  ctx = PreRegAllocContext(uops)
  bufs = _wmma_acc_buffers(ctx)
  if not bufs: return [], {}, {}, {}
  seen: set[tuple] = set()
  inits: list[UOp] = []
  tiles: dict[int, UOp] = {}
  buf_tiles: dict[UOp, dict[int, UOp]] = {}
  idx_map: dict[int|tuple[UOp, int], tuple[UOp, int]] = {}
  next_tile = 0
  buf_next_tile: dict[UOp, int] = {}
  for u in uops:
    if u.op is not Ops.INS or _iop(u) is not AMDOps.WMMA: continue
    pack = u.src[0]
    if not _is_wmma_acc_reload_pack(pack): continue
    if not isinstance(pack.tag, tuple) or not pack.tag: continue
    if pack.tag in seen: continue
    # SLOAD-cin: tile from REG indices. Zero-MOV / dynamic-index cin: enumerate in expand order.
    sload_idxs: list[int|None] = []
    pack_base: UOp|None = None
    tile_local: int|None = None
    if all(s.op is Ops.INS and _iop(s) is AMDOps.SLOAD for s in pack.src):
      if not any((b:=_reg_buffer_base(s.src[0])) is not None and b in bufs for s in pack.src): continue
      pack_base = next(b for s in pack.src if (b:=_reg_buffer_base(s.src[0])) is not None and b in bufs)
      sload_idxs = [_const_int(s.src[1]) for s in pack.src]
      if any(i is None for i in sload_idxs):
        # Flash REG indices are loop-varying; still park ACC by expand order (flash ACC_SMALL).
        if not _wmma_acc_small_mode(ctx): continue
        sload_idxs = []
        tile_local = buf_next_tile.get(pack_base, 0)
        buf_next_tile[pack_base] = tile_local + 1
        tile = next_tile
        next_tile += 1
      else:
        tile, _ = _wmma_slot_tile_lane(sload_idxs[0])  # type: ignore[arg-type]
        # Product-16: first-idx formula only yields 4 keys for 16 packs. Colliding
        # tiles[tile]=init left later packs sharing an ACC zero (wrong N≥48). Uniqueify
        # with a free key — bare `tile = next_tile` when next_tile==0 overwrote tiles[0].
        if tile in tiles:
          while next_tile in tiles: next_tile += 1
          tile = next_tile
        next_tile = max(next_tile, tile + 1)
    else:
      while next_tile in tiles: next_tile += 1
      tile = next_tile
      next_tile += 1
    seen.add(pack.tag)
    init = UOp(Ops.INS, src=tuple(_tconst(0.0, dtypes.float32) for _ in range(8)), arg=(AMDOps.PACK, dtypes.float), tag=pack.tag)
    inits.append(init)
    tiles[tile] = init
    if pack_base is not None and tile_local is not None:
      buf_tiles.setdefault(pack_base, {})[tile_local] = init
    for lane, (sload, idx) in enumerate(zip(pack.src, sload_idxs)):
      if idx is not None and (base:=_reg_buffer_base(sload.src[0])) is not None:
        idx_map[idx if len(bufs) == 1 else (base, idx)] = (init, lane)
    # Small flash tiles: map linear REG idx → ACC lane for post-WMMA EXTRACT copies.
    if _wmma_acc_small_mode(ctx) and pack_base is not None and tile_local is not None:
      numel = pack_base.max_numel()
      n_tiles = numel // 8 if numel >= 8 and numel % 8 == 0 else 0
      if n_tiles and tile_local < n_tiles:
        for lane in range(8):
          idx = lane * n_tiles + tile_local
          if 0 <= idx < numel:
            idx_map[idx if len(bufs) == 1 else (pack_base, idx)] = (init, lane)
  return inits, tiles, idx_map, buf_tiles

def _reg_promotable_buffers(ctx:PreRegAllocContext) -> set[UOp]:
  if (promotable:=ctx.scratch.get("reg_promotable")) is not None: return promotable
  # AMD_REG_PROMOTE hardcoded ON.
  bases, bad, seen_store = set(), set(), set()
  wmma_bufs = _wmma_acc_buffers(ctx)
  for u in ctx.uops or []:
    if u.op is not Ops.INS or _iop(u) not in (AMDOps.SLOAD, AMDOps.SSTORE): continue
    if (base:=_reg_buffer_base(u.src[0])) is None: continue
    bases.add(base)
    # Flash output acc / row stats are REDUCE-carried; promoting them drops loop values.
    # REG_PROMOTE_SKIP_SLOTS folded: flash (ACC_SMALL prefill / VOPD_FMAC decode) promotes
    # slot 2; matmul/other keeps skip {2}.
    _skip = set() if (getenv("AMD_FLASH_ACC_SMALL", 0) or getenv("AMD_VOPD_FMAC", 0)) else {2}
    if (slot:=getattr(base.arg, "slot", None)) is not None and slot in _skip:
      bad.add(base)
      continue
    if base in wmma_bufs: continue  # handled by WMMA ACC aliasing
    idx = _const_int(u.src[1])
    dt = u.dtype if _iop(u) is AMDOps.SLOAD else u.src[2].dtype
    n = _elem_count(u) if _iop(u) is AMDOps.SLOAD else _elem_count(u.src[2])
    # WMMA→REG fragment stores need ACC-style K-carry; promoting those slots desyncs on
    # direct ISA (flash tile-unroll+soft). Only match the stored value itself — do not
    # walk AFTER history or every later soft store of S_reg gets poisoned.
    if _iop(u) is AMDOps.SSTORE:
      v = u.src[2]
      while v.op is Ops.AFTER: v = v.src[0]
      if v.op is Ops.INS and _iop(v) is AMDOps.EXTRACT: v = v.src[0]
      if (v.op is Ops.WMMA) or (v.op is Ops.INS and _iop(v) is AMDOps.WMMA):
        bad.add(base)
        continue
    bo = _lds_byte_off(u)
    if idx is not None:
      # Const index + peeled imm: fold into the element slot. Dynamic idx + peel is fine
      # (still non-promotable via idx is None below) — do not poison the whole buffer.
      if dt.itemsize and bo % dt.itemsize == 0:
        idx = idx + bo // dt.itemsize
      elif bo:
        bad.add(base)
        continue
    # Scalar or small const-indexed vectors (flash float4 soft copies). n!=1 used to
    # poison the whole REG buffer and force scratch for ACC_SEP soft/pv.
    if idx is None or idx < 0 or base.max_numel() > 64 or dt.itemsize > 4 or n not in (1, 2, 4):
      bad.add(base)
      continue
    if idx + n > base.max_numel():
      bad.add(base)
      continue
    for ei in range(n):
      key = (base, idx + ei)
      if _iop(u) is AMDOps.SLOAD and key not in seen_store: bad.add(base)
      if _iop(u) is AMDOps.SSTORE: seen_store.add(key)
  ctx.scratch["reg_promotable"] = promotable = bases - bad
  ctx.scratch["reg_values"] = {}
  ctx.scratch["reg_n"] = 0
  return promotable

def _reg_promote_slot(ctx:PreRegAllocContext, base:UOp, idx:UOp, byte_off:int=0, itemsize:int=4) -> tuple[UOp, int]|None:
  buf = _reg_buffer_base(base)
  if buf is None or buf not in _reg_promotable_buffers(ctx): return None
  if (slot:=_const_int(idx)) is None or itemsize <= 0 or byte_off % itemsize: return None
  return buf, slot + byte_off // itemsize

def _new_promoted_reg(ctx:PreRegAllocContext, val:UOp, cons:tuple[Register, ...]|None=None) -> UOp:
  n = ctx.scratch["reg_n"]
  ctx.scratch["reg_n"] = n + 1
  if cons is None: cons = VGPR
  return UOp(Ops.INS, src=(val,), arg=(AMDOps.MOV, val.dtype), tag=(Register(f"reg{n}", 0, _cons=cons),))

def _promote_reg_cons(ctx:PreRegAllocContext, buf:UOp) -> tuple[Register, ...]:
  """VGPR pool for a promoted REG buffer.

  Flash REDUCE-carried output acc (slot 2) must not share parked WMMA ACC (v126+):
  under pressure those lanes land in ACC and WMMA clobbers them (SKIP='' → nan).
  Do **not** constrain slot 19 (tile work-copy) — that broke the SKIP=2 baseline.
  """
  if getattr(buf.arg, "slot", None) != 2: return VGPR
  uops = ctx.uops or []
  if not any(u.op is Ops.WMMA or (u.op is Ops.INS and _iop(u) is AMDOps.WMMA) for u in uops):
    return VGPR
  packed_quant = True and any(
    u.op is Ops.INS and _iop(u) in (AMDOps.FMA_TO_F16, AMDOps.PACKED_F16_MUL_TO_F16) for u in uops)
  pools = _register_pools(ctx)
  acc_pool = pools.quant_acc if packed_quant else pools.acc
  if acc_pool and acc_pool is not VGPR:
    try: cut = next(i for i,r in enumerate(VGPR) if r.index == acc_pool[0].index)
    except StopIteration: cut = len(VGPR) - len(acc_pool)
  else:
    cut = len(VGPR) - len(acc_pool)
  allowed = VGPR[:cut]
  # AMD_FLASH_ACC_HIGH_VGPR removed (never shipped; LinearScan mid-band ok for flash).
  return allowed

def _index_bounds(root:UOp) -> tuple[int, int]|None:
  """Mathematical bounds before a 32-bit index is zero-extended into a GLOBAL address."""
  memo:dict[UOp, tuple[int, int]|None] = {}
  def bounds(u:UOp) -> tuple[int, int]|None:
    if u in memo: return memo[u]
    ret:tuple[int, int]|None = None
    if (c:=_const_int(u)) is not None: ret = c, c
    elif u.op is Ops.INS:
      op = _iop(u)
      if op in (AMDOps.MOV, AMDOps.CAST) and u.src: ret = bounds(u.src[0])
      elif op in (AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR):
        a, b = bounds(u.src[0]), bounds(u.src[1])
        if a is not None and b is not None:
          if op is AMDOps.ADD: ret = a[0]+b[0], a[1]+b[1]
          elif op is AMDOps.SUB: ret = a[0]-b[1], a[1]-b[0]
          elif op is AMDOps.MUL:
            products = [x*y for x in a for y in b]
            ret = min(products), max(products)
          elif op in (AMDOps.SHL, AMDOps.SHR) and 0 <= b[0] <= b[1] < 32 and a[0] >= 0:
            ret = (a[0]<<b[0], a[1]<<b[1]) if op is AMDOps.SHL else (a[0]>>b[1], a[1]>>b[0])
          elif op is AMDOps.AND:
            if b[0] == b[1] and b[0] >= 0: ret = 0, b[0]
            elif a[0] == a[1] and a[0] >= 0: ret = 0, a[0]
          elif op in (AMDOps.OR, AMDOps.XOR) and a[0] >= 0 and b[0] >= 0:
            ret = 0, (1 << max(a[1], b[1]).bit_length())-1
    elif u.op is Ops.CAST: ret = bounds(u.src[0])
    elif u.dtype in dtypes.ints+(dtypes.weakint,) and isinstance(u.vmin, int) and isinstance(u.vmax, int): ret = u.vmin, u.vmax
    if ret is not None and u.op in (Ops.CAST, Ops.INS) and u.dtype in dtypes.ints and \
       not (u.dtype.min <= ret[0] <= ret[1] <= u.dtype.max):
      ret = None
    memo[u] = ret
    return ret
  return bounds(root)

def _peel_add_imm(idx:UOp, itemsize:int, max_byte:int=0xffff, deep:bool=False) -> tuple[UOp, int]:
  """Peel ADD+imm from an index into a byte offset. Keeps one address base live.
  deep=True folds nested ADD+const chains (WMMA C stores: ADD(ADD(base,1024),16))."""
  total, cur = 0, idx
  while True:
    is_add = cur.op is Ops.ADD or (cur.op is Ops.INS and _iop(cur) is AMDOps.ADD)
    if not is_add or len(cur.src) != 2: break
    peeled = False
    for base_i, imm_i in ((0, 1), (1, 0)):
      if (c := _const_int(cur.src[imm_i])) is None: continue
      byte = total + c * itemsize
      if 0 <= byte <= max_byte:
        total, cur, peeled = byte, cur.src[base_i], True
        break
    if not peeled or not deep: break
  # GLOBAL's immediate is added after zero-extending the VGPR byte address. A negative
  # peeled base (e.g. flip's 47-i) would become base+4GiB, not the original valid index.
  if total and ((span:=_index_bounds(cur)) is None or span[0] < 0 or span[1]*itemsize+total > 0xffffffff): return idx, 0
  return (cur, total) if total else (idx, 0)

def _ds_off(byte_off:int) -> dict:
  return {"offset0": byte_off & 0xff, "offset1": (byte_off >> 8) & 0xff}

def _is_b_compact_load(u:UOp) -> bool:
  return len(u.src) > 3 and _const_value(u.src[3]) is not None and u.src[3].tag == "b_compact"

def _is_byte_addr_load(u:UOp) -> bool:
  return len(u.src) > 3 and _const_value(u.src[3]) is not None and u.src[3].tag == "byte_addr"

def _load_ins(x:UOp, a:UOp, alt:UOp|None=None, gate:UOp|None=None) -> UOp:
  n = x.max_numel()
  param = a.src[0].buf_uop
  volatile = param.op is Ops.PARAM and param.arg.volatile
  def global_load(src:tuple[UOp, ...]) -> UOp:
    if volatile: src = (src[0], src[1], src[2].rtag("volatile"), *src[3:])
    return x.ins(AMDOps.LOAD, dtype=x.dtype, src=src)
  if alt is not None and gate is not None:
    # Coalescing uses SHRINK addresses, which bypass the scalar pre-isel gated-load
    # rewrite. Selecting the result alone must not issue an out-of-bounds load.
    a = a.replace(src=(a.src[0], gate.where(a.src[1], a.src[1].const_like(0)), *a.src[2:]))
    raw = UOp(Ops.LOAD, src=(a,))
    if n == 1: return gate.where(raw, alt)
    return UOp(Ops.STACK, src=tuple(
      gate.where(raw.index(_tconst(i, dtypes.weakint)), alt.index(_tconst(i, dtypes.weakint)) if alt.max_numel() > 1 else alt)
      for i in range(n)))
  count = _load_count_src(n)
  if _is_lds_ref(a.src[0]):
    if _local_load(x.dtype, n) is None and not (x.dtype is dtypes.half and n == 16):
      raise CompileError(f"no lds load {x.dtype} x{n}")
    # deep=True: flash LDS uses ADD(ADD(base,tid),imm) — fold const chain into ds offset0/1.
    idx, off = _peel_add_imm(a.src[1], _mem_itemsize(x.dtype), max_byte=0xffff, deep=True)
    src = (a.src[0], idx, count) if off == 0 else (a.src[0], idx, count, _tconst(off, dtypes.int32).rtag())
    return x.ins(AMDOps.LLOAD, dtype=x.dtype, src=src)
  if _is_scratch_ref(a.src[0]):
    if _scratch_load(x.dtype, n) is None: raise CompileError(f"no scratch load {x.dtype} x{n}")
    # Match SSTORE: peel const byte offsets into SCRATCH's 12-bit imm (flash REG traffic).
    item = _mem_itemsize(x.dtype)
    # Peel ADD+imm only. Do not const-fold idx→0+off: that collapses REG promote/skip
    # slots unless every consumer accounts for byte_off (flash wants shared dynamic bases).
    idx, off = _peel_add_imm(a.src[1], item, max_byte=0xfff)
    src = (a.src[0], idx, count) if off == 0 else (a.src[0], idx, count, _tconst(off, dtypes.int32).rtag())
    return x.ins(AMDOps.SLOAD, dtype=x.dtype, src=src)
  if _global_load(x.dtype, n) is None and not (x.dtype is dtypes.half and n == 16):
    raise CompileError(f"no global load {x.dtype} x{n}")
  # Compact B: peel to per-k page idx + rem≤4095 GLOBAL offset (LLVM @N≥2048).
  if n == 1 and x.dtype is dtypes.half:
    itemsize = _mem_itemsize(x.dtype)
    base, total = _peel_add_imm(a.src[1], itemsize, max_byte=0x7fffffff, deep=True)
    if total > 0:
      page, rem = divmod(total, 0x1000)
      idx = _b_page_idx(base, page, itemsize)
      # Always attach rem (incl. 0) tagged so emit can in-place <<1 safely.
      src = (a.src[0], idx, count, _tconst(rem, dtypes.int32).rtag("b_compact"))
      return global_load(src)
  # Scalar global gathers: keep one shared byte-address base and use GLOBAL's 12-bit offset.
  # LLVM performs this fold for quantized GEMV; without it every adjacent load repeats ADD+LSHL.
  if n == 1 and _mem_itemsize(x.dtype) in (1, 4):
    itemsize = _mem_itemsize(x.dtype)
    idx, off = _peel_add_imm(a.src[1], itemsize, max_byte=0xfff, deep=True)
    # Keep 64-bit indices intact. Scaling a peeled long index introduces long CASTs after
    # instruction selection, which ISA lowering cannot encode (notably in spill-heavy kernels).
    if off > 0 and idx.dtype.itemsize <= 4:
      if itemsize == 4: idx = idx << _tconst(2, idx.dtype)
      return global_load((a.src[0], idx, count, _tconst(off, dtypes.int32).rtag("byte_addr")))
  return global_load((a.src[0], a.src[1], count))

def _store_ins(x:UOp, a:UOp, val:UOp) -> UOp:
  # Bottom-up isel can match STORE before INDEX→EXTRACT on vec/WMMA values. Defer (return None)
  # so the INDEX child is rewritten first; only raise when the value is already final.
  def try_store(check, op, peel:bool=False, max_byte:int=0xffff, deep:bool=False, allow_half16:bool=False):
    n = _elem_count(val)
    if check(val.dtype, n) is None and not (allow_half16 and val.dtype is dtypes.half and n == 16):
      if val.op is Ops.INDEX and _elem_count(val.src[0]) > 1: return None
      raise CompileError(f"no store {val.dtype}")
    if peel:
      # Peel base+imm so one addr VGPR is shared (TC_LDS C-stores: 64 ADD(base,imm) → few bases).
      item = _mem_itemsize(val.dtype)
      idx, off = _peel_add_imm(a.src[1], item, max_byte=max_byte, deep=deep)
      # Pure const index (flash REG init): fold to offset0 + imm so stores share a zero base.
      if off == 0 and (c := _const_int(idx)) is not None and c > 0 and c * item <= max_byte:
        off, idx = c * item, _tconst(0, idx.dtype)
      src = (a.src[0], idx, val) if off == 0 else (a.src[0], idx, val, _tconst(off, dtypes.int32).rtag())
      return x.ins(op, src=src)
    return x.ins(op, src=(a.src[0], a.src[1], val))
  if _is_lds_ref(a.src[0]): return try_store(_local_store, AMDOps.LSTORE, peel=True, deep=True, allow_half16=True)
  # Peel scratch REG stores to shared base + imm offset (flash init: 200× lshl+add → few bases).
  if _is_scratch_ref(a.src[0]): return try_store(_scratch_store, AMDOps.SSTORE, peel=True, max_byte=0xfff)
  # Soft-peel any ADD+imm (incl. nested). Emit uses GLOBAL offset when ≤4095 else v_lshl_add.
  # (Hard-peel-only-≤4095 left ~120 addr VGPRs + LSHL/store for WMMA C.)
  return try_store(_global_store, AMDOps.STORE, peel=True, max_byte=0x7fffffff, deep=True)

def _extract_vec_lane(ctx:IselContext, x:UOp) -> UOp|None:
  if len(x.src) != 2 or (lane:=_const_int(x.src[1])) is None: return None
  # INDEX into memory is an address. Only ALU values use INDEX as vector lane access.
  if x.src[0].addrspace not in (None, AddrSpace.ALU): return None
  if x.src[0].op is Ops.WMMA:
    return UOp(Ops.INS, src=(x.src[0], _tconst(lane, dtypes.int32).rtag()), arg=(AMDOps.EXTRACT, dtypes.float32))
  n = _elem_count(x.src[0])
  if n == 1 and lane == 0 and x.src[0].addrspace in (None, AddrSpace.ALU): return x.src[0]
  if n == 1: return None
  sc = x.src[0].dtype
  if sc not in (dtypes.uint8, dtypes.float32, dtypes.float16, dtypes.int32, dtypes.uint32):
    raise CompileError(f"no extract from {x.src[0].dtype}")
  if not 0 <= lane < n: raise CompileError(f"lane {lane} oob for {x.src[0].dtype} x{n}")
  if sc is dtypes.uint8 and x.src[0].op is Ops.INS and \
     _iop(x.src[0]) is AMDOps.LOAD and all(_is_scalar_source(s) for s in x.src[0].src):
    return _uniform_byte_extract(ctx, UOp(Ops.INS, src=(x.src[0], _tconst(lane, dtypes.int32).rtag()), arg=(AMDOps.EXTRACT, sc)))
  return UOp(Ops.INS, src=(x.src[0], _tconst(lane, dtypes.int32).rtag()), arg=(AMDOps.EXTRACT, sc))

def _pack_vec(x:UOp) -> UOp|None:
  if x.max_numel() == 1 and len(x.src) == 1: return x.src[0]
  if x.max_numel() == 1: return None
  if len(x.src) != x.max_numel(): raise CompileError(f"pack size {len(x.src)} != {x.max_numel()}")
  if x.dtype is dtypes.float32:
    return UOp(Ops.INS, src=x.src, arg=(AMDOps.PACK, dtypes.float32), tag=x.tag)
  if x.dtype is dtypes.float16:
    if len(x.src) % 2: raise CompileError(f"half pack needs even len, got {len(x.src)}")
    return UOp(Ops.INS, src=x.src, arg=(AMDOps.PACK_F16, dtypes.half), tag=x.tag)
  raise CompileError(f"no pack {x.dtype}")

def _wmma_ab_from_lds(wmma:UOp) -> bool:
  """True if WMMA A/B is staged from LDS (TC_LDS_AB), not unrelated LLOAD elsewhere in the kernel."""
  def from_lds(x:UOp, depth:int=0) -> bool:
    if depth > 6 or x.op is not Ops.INS: return False
    if _iop(x) is AMDOps.LLOAD: return True
    if _iop(x) in (AMDOps.PACK_F16, AMDOps.EXTRACT, AMDOps.MOV):
      return any(from_lds(s, depth + 1) for s in x.src)
    return False
  return len(wmma.src) >= 3 and (from_lds(wmma.src[1]) or from_lds(wmma.src[2]))

def _wmma_ab_vec_loads(elems:tuple[UOp, ...]) -> tuple[UOp, ...]|None:
  # STACK of INDEX(half×n LOAD, 0..n-1)... → PACK srcs are the Ops.LOAD nodes (isel tags them once).
  # Shape tracking can canonicalize INDEX(vec_load, lane) into RESHAPE(SHRINK(vec_load, lane, 1));
  # recognize both forms so a WMMA fragment remains two B128s instead of sixteen U16 loads.
  def vec_lane(e:UOp) -> tuple[UOp, int]|None:
    while e.op in (Ops.RESHAPE, Ops.NOOP) and e.src and e.max_numel() == 1: e = e.src[0]
    if e.op is Ops.INDEX and len(e.src) == 2 and (lane:=_const_int(e.src[1])) is not None and e.src[0].op is Ops.LOAD:
      return e.src[0], lane
    if e.op is Ops.SHRINK and len(e.src) > 2 and _const_value(e.src[2]) == 1 and \
       (lane:=_const_int(e.src[1])) is not None and e.src[0].op is Ops.LOAD:
      return e.src[0], lane
    return None
  if len(elems) != 16: return None
  # WMMA expansion may scalarize a formerly coalesced LOAD before ISA selection. Rebuild one
  # half×16 load when all fragment elements are unconditional adjacent reads from the same buffer.
  scalar_addrs = [e.src[0] for e in elems if e.op is Ops.LOAD and e.dtype is dtypes.half and len(e.src) == 1 and
                  e.src[0].op is Ops.INDEX and len(e.src[0].src) == 2]
  if len(scalar_addrs) == 16:
    peeled = [_peel_add_imm(a.src[1], 1, max_byte=0x7fffffff, deep=True) for a in scalar_addrs]
    base, first = peeled[0]
    if all(a.src[0] is scalar_addrs[0].src[0] and b is base and off == first+i
           for i,(a,(b,off)) in enumerate(zip(scalar_addrs, peeled))):
      start = base + _tconst(first, base.dtype) if first else base
      ptr = UOp(Ops.SHRINK, src=(scalar_addrs[0].src[0], start, _tconst(16, dtypes.int32)))
      return (UOp(Ops.LOAD, src=(ptr,)),)
  loads: list[UOp] = []
  i = 0
  while i < 16:
    if (got:=vec_lane(elems[i])) is None: return None
    base, lane0 = got
    if lane0 != 0: return None
    n = base.max_numel()
    if n < 2 or i + n > 16: return None
    for j in range(n):
      if (got:=vec_lane(elems[i+j])) is None or got != (base, j): return None
    loads.append(base)
    i += n
  if not loads: return None
  # AMD_PACK_F16_FUSE_LOADS removed (fair regress).
  return tuple(loads)

def _wmma_stack_operand(src:UOp, idx:int) -> UOp:
  # Coalesced half×16 frag may arrive as Ops.LOAD (STACK folded) or STACK of INDEX.
  if idx < 2 and src.dtype is dtypes.half and src.max_numel() == 16 and src.op is Ops.LOAD:
    return UOp(Ops.INS, src=(src,), arg=(AMDOps.PACK_F16, dtypes.half))
  if idx < 2 and src.op is Ops.INS and _iop(src) is AMDOps.LOAD and _elem_count(src) == 16:
    return UOp(Ops.INS, src=(src,), arg=(AMDOps.PACK_F16, dtypes.half))
  if src.op is not Ops.STACK: raise CompileError(f"wmma src must be stack, got {src.op}")
  n, sc = len(src.src), src.dtype
  if idx < 2 and n == 16 and sc is dtypes.half:
    if (loads := _wmma_ab_vec_loads(src.src)) is not None:
      return UOp(Ops.INS, src=loads, arg=(AMDOps.PACK_F16, dtypes.half))
    return UOp(Ops.INS, src=src.src, arg=(AMDOps.PACK_F16, dtypes.half))
  if idx == 2 and n == 8 and sc is dtypes.float:
    return UOp(Ops.INS, src=src.src, arg=(AMDOps.PACK, dtypes.float))
  raise CompileError(f"bad wmma stack idx={idx} len={n} dtype={src.dtype}")

def _isel_wmma(ctx:IselContext, x:UOp) -> UOp:
  a, b = (_wmma_stack_operand(s, i) for i, s in enumerate(x.src[:2]))
  # accumulator is the init STACK for the first WMMA, or a chained prior WMMA result when UNROLL
  # fuses K iterations. is_two_address coalesces dst with src[0] (=acc), so the chain shares one reg.
  cin = x.src[2]
  if cin.op is Ops.WMMA or (cin.op is Ops.INS and _iop(cin) is AMDOps.WMMA):
    c = cin
  else:
    # fresh zero accumulator. WMMA is two-address (D is written over C), so each independent output
    # tile (from UPCAST) needs its OWN accumulator register. the zero-init STACK is identical across
    # tiles and dedups to one UOp -> one reg, which the two-address coalesce can only satisfy for one
    # tile. pre-assign a unique vreg so each tile's accumulator stays distinct.
    quantized = True and any(
      u.op in (Ops.CUSTOM, Ops.CUSTOMI) and _custom_name(u) in (AMD_FMA_TO_F16, AMD_PACKED_F16_MUL_TO_F16) for u in ctx.uses)
    pools = _register_pools(ctx)
    c = _wmma_stack_operand(cin, 2).replace(tag=(ctx.vreg(pools.quant_acc if quantized else pools.acc),))
  return UOp(Ops.INS, src=(c, a, b), arg=(AMDOps.WMMA, dtypes.float if x.dtype is dtypes.float else x.dtype), tag=x.tag)

def _wmma_inst(u:UOp):
  dt_in, dt_out = u.src[1].dtype, u.dtype
  if dt_in is dtypes.half and dt_out is dtypes.float: return r3.v_wmma_f32_16x16x16_f16
  raise CompileError(f"no wmma {dt_in} -> {dt_out}")

def _pack_f16_identity_load(u:UOp) -> UOp|None:
  # PACK_F16(EXTRACT(L,0)..EXTRACT(L,2n-1)) with L = half×n LLOAD → reuse L's VGPRs.
  # Global LOAD stays off identity: LOAD dests are reused across tiles before PACK emit.
  if len(u.src) < 2 or len(u.src) % 2: return None
  base: UOp|None = None
  for i in range(len(u.src) // 2):
    got = _pack_f16_half2_load(u.src[2*i], u.src[2*i+1])
    if got is None: return None
    b, slot = got
    if slot != i: return None
    if _iop(b) is not AMDOps.LLOAD: return None
    if base is None: base = b
    elif base is not b: return None
  if base is None or _reg_slots(base) < len(u.src) // 2: return None
  return base

def _pack_f16_insts(u:UOp, fma_hi_lo:dict[UOp, UOp]|None=None, fma_pair_dst:dict[UOp, Reg]|None=None) -> list:
  # Vec-load form: PACK_F16(LOAD/LLOAD[, ...]) — bitcast half2 words into WMMA src VGPRs.
  # Emit via _parallel_vmov so consecutive even/odd evacuate copies become dual_mov (HIP-like).
  if _pack_f16_is_vec_load(u) or (len(u.src) == 1 and u.src[0].op is Ops.INS and
      _iop(u.src[0]) in (AMDOps.LOAD, AMDOps.LLOAD) and _reg_slots(u.src[0]) == _reg_slots(u)):
    moves: list[tuple[Reg, Reg|int|float]] = []
    off = 0
    for src in u.src:
      for j in range(_reg_slots(src)):
        src_lane, d = _reg_lane(greg(src), j), _reg_lane(greg(u), off + j)
        if src_lane != d: moves.append((d, src_lane))
      off += _reg_slots(src)
    return _parallel_vmov(moves)
  # Lo-only HALF2 rewrite: each src is an even EXTRACT holding a full half2 word.
  if _pack_f16_is_half2_lo_only(u):
    moves = []
    for i, src_item in enumerate(u.src):
      src_slot, dst_slot = _dst(src_item), _reg_lane(greg(u), i)
      if src_slot != dst_slot: moves.append((dst_slot, src_slot))
    return _parallel_vmov(moves)
  # Regalloc remat may replace a vec-load PACK src with a scalar MOV bind (same phys).
  if len(u.src) == 1:
    src = u.src[0]
    if isinstance(greg(src), Register) and isinstance(greg(u), Register) and greg(src).index == greg(u).index:
      return []
    return _parallel_vmov([(_dst(u), _src(src))])
  if len(u.src) < 2 or len(u.src) % 2: raise CompileError(f"pack_f16 needs even src, got {len(u.src)}")
  ret: list = []
  moves = []
  for i in range(len(u.src) // 2):
    lo, hi = u.src[2*i], u.src[2*i+1]
    # Flush pending MOVs before any v_pack so dual packing stays contiguous.
    def _flush() -> None:
      nonlocal moves
      if moves: ret.extend(_parallel_vmov(moves))
      moves = []
    # A paired mixhi has already written the high half into lo's VGPR. Move that
    # packed word into the WMMA fragment instead of executing a separate v_pack.
    if (fma_hi_lo or {}).get(hi) is lo:
      src_slot, dst_slot = (fma_pair_dst or {}).get(lo, _dst(lo)), _reg_lane(greg(u), i)
      if src_slot != dst_slot: moves.append((dst_slot, src_slot))
      continue
    if (got := _pack_f16_half2_load(lo, hi)) is not None:
      base, slot = got
      base_slot = _reg_lane(greg(base), slot)
      # Even-lane EXTRACT MOVs the full half2 word into lo's dest. Prefer that over the
      # LOAD VGPR: flash reuses global_load_b128 dests for the next tile before PACK
      # emits, so greg(base) can already hold a different vector (HALF2_GLOBAL sum −22.3).
      pack_src_slot: Reg|None
      if isinstance(greg(lo), Register) and _dst(lo) != base_slot:
        pack_src_slot = _dst(lo)
      elif _iop(base) is AMDOps.LLOAD:
        pack_src_slot = base_slot
      else:
        pack_src_slot = None  # global LOAD phys may be dead — fall through to v_pack
      if pack_src_slot is not None:
        dst_slot = _reg_lane(greg(u), i)
        if pack_src_slot != dst_slot: moves.append((dst_slot, pack_src_slot))
        continue
    # Always v_pack from lo/hi. Do not shortcut through the half2 load VGPR: EXTRACT(hi)
    # may LSHR that load in place, so a later mov from the load reg sees [hi,0] (2026-07-20).
    # Spill FILL of a multi-slot value can surface as v[n:n+k]; v_pack needs scalar lanes.
    _flush()
    pre0, a = _vgpr_data(TMP_VDATA, lo)
    pre1, b = _vgpr_data(TMP_VADDR, hi)
    if isinstance(a, Reg) and a.sz > 1: a = a[0]
    if isinstance(b, Reg) and b.sz > 1: b = b[0]
    ret += pre0 + pre1 + [r3.v_pack_b32_f16(_reg_lane(greg(u), i), a, b)]
  if moves: ret.extend(_parallel_vmov(moves))
  return ret

AMD_ATOMIC_ADD = "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);"
AMD_DOT4 = "__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)"
AMD_BYTE_PERM = "__builtin_amdgcn_perm({}, {}, {})"
AMD_NONTEMPORAL_LOAD = "__builtin_nontemporal_load({0})"
AMD_PACKED_U8X16_LOAD = "__builtin_nontemporal_load((const unsigned_int4*){0})"
AMD_FMA_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16({}, {}, {})"
AMD_PACKED_F16_MUL_TO_F16 = "__builtin_amdgcn_fma_mixlo_f16_packed({}, {}, {})"
AMD_MBCNT_LO = "__builtin_amdgcn_mbcnt_lo(-1, 0)"
AMD_SWIZZLE_PREFIX = "__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {0}), "
# Identity cross-16 gather: lane i reads lane i^16. Select nibbles 0..15 across src1|src2.
AMD_PERMLANEX16 = ("__builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, {0}), "
                   "__builtin_bit_cast(int, {0}), 0x76543210, 0xfedcba98, true, false))")

def _warp_group_reduce(x:UOp) -> UOp|None:
  """Use a wave32 butterfly for an isolated f32 ADD group reduction."""
  grouped = [r for r in x.src[1:] if r.op is Ops.RANGE and r.axis_type is AxisType.LOCAL]
  if x.dtype is not dtypes.float32 or not 1 <= x.max_numel() <= 4 or x.arg[0] is not Ops.ADD or len(grouped) != 1 or \
     _const_value(grouped[0].src[0]) != 32: return None
  group = grouped[0]
  # The group must map directly to hardware lane id. Mixed local/group layouts retain
  # the generic LDS reduction, whose addressing and barriers cover those cases.
  if any(r is not group and r.op is Ops.RANGE and r.axis_type in (AxisType.WARP, AxisType.LOCAL)
         for r in x.src[0].toposort()): return None
  remaining = tuple(r for r in x.src[1:] if r is not group)
  val = x.replace(src=(x.src[0],)+remaining) if remaining or x.arg[1] else x.src[0]
  def butterfly(lane_val:UOp) -> UOp:
    # CUSTOM inputs are not part of the generic add-loads matcher. Keep each reduced
    # register value behind an elementwise node so an accumulator AFTER becomes a LOAD.
    lane_val = lane_val + lane_val.const_like(0)
    for offset in (16, 8, 4, 2, 1):
      other = UOp(Ops.CUSTOM, src=(lane_val,), arg=(f"{AMD_SWIZZLE_PREFIX}{0x1f | offset<<10}))", dtypes.float))
      lane_val = lane_val + other
    return lane_val
  return butterfly(val) if x.max_numel() == 1 else UOp.stack(*(butterfly(val.index(i)) for i in range(x.max_numel())))

pm_warp_group_reduce = PatternMatcher([
  (UPat(Ops.REDUCE, name="x"), _warp_group_reduce),
])

def _custom_name(x:UOp) -> object: return x.arg[0] if isinstance(x.arg, tuple) else x.arg

def _amd_custom_intrinsic(x:UOp) -> UOp|None:
  arg = _custom_name(x)
  if arg == AMD_DOT4: return x.ins(AMDOps.DOT4)
  if arg == AMD_BYTE_PERM: return x.ins(AMDOps.BYTE_PERM)
  if arg == AMD_FMA_TO_F16: return x.ins(AMDOps.FMA_TO_F16)
  if arg == AMD_PACKED_F16_MUL_TO_F16: return x.ins(AMDOps.PACKED_F16_MUL_TO_F16)
  if arg == AMD_PACKED_U8X16_LOAD and len(x.src) == 1 and x.src[0].op is Ops.SHRINK:
    ptr = x.src[0]
    src:tuple[UOp, ...] = (ptr.src[0], ptr.src[1], _load_count_src(4))
    if ptr.src[0].dtype is dtypes.uint8: src += (_tconst(0, dtypes.int32).rtag("byte_addr"),)
    return x.ins(AMDOps.LOAD, src=src)
  if arg == AMD_MBCNT_LO and len(x.src) == 1: return x.src[0]
  if isinstance(arg, str) and arg.startswith(AMD_SWIZZLE_PREFIX) and arg.endswith("))"):
    try: offset = int(arg[len(AMD_SWIZZLE_PREFIX):-2])
    except ValueError: return None
    if not 0 <= offset <= 0xffff: raise CompileError(f"bad ds_swizzle offset {offset}")
    return x.ins(AMDOps.SWIZZLE, src=(x.src[0], _tconst(offset, dtypes.uint32).rtag()))
  if arg == AMD_PERMLANEX16: return x.ins(AMDOps.PERMLANEX16)
  return None

def _nontemporal_load(x:UOp) -> UOp|None:
  if _custom_name(x) != AMD_NONTEMPORAL_LOAD or len(x.src) != 1 or x.src[0].op is not Ops.INDEX: return None
  return x.src[0].load(dtype=x.dtype)

def _bitfield_extract(x:UOp, value:UOp, shift:UOp, mask:UOp) -> UOp|None:
  m = int(mask.val)
  if m <= 0 or (m & (m + 1)) != 0 or (width:=m.bit_length()) >= 32: return None
  return x.ins(AMDOps.BFE, src=(value, shift, _tconst(width, dtypes.uint32).rtag()))

def _cvt_ubyte_f32(x:UOp, value:UOp, mask:UOp, shift:UOp|None=None) -> UOp|None:
  """Fold u32 byte extraction followed by float conversion into V_CVT_F32_UBYTEN."""
  if _const_value(mask) != 0xff: return None
  bit = 0 if shift is None else _const_value(shift)
  if bit not in (0, 8, 16, 24): return None
  return x.ins(AMDOps.CVT_UBYTE_F32,
               src=(value, _tconst(int(bit) // 8, dtypes.uint32).rtag()))

def _lshl_or(x:UOp, value:UOp, shift:UOp, other:UOp) -> UOp:
  return x.ins(AMDOps.LSHL_OR, src=(value, shift, other))

def _lshl_add(x:UOp, value:UOp, shift:UOp, other:UOp) -> UOp:
  return x.ins(AMDOps.LSHL_ADD, src=(value, shift, other))

def _atomic_add_ins(x:UOp) -> UOp|None:
  if _custom_name(x) != AMD_ATOMIC_ADD: return None
  if len(x.src) != 2 or x.src[0].op is not Ops.INDEX: raise CompileError(f"bad atomic {x}")
  a, val = x.src
  if val.dtype is not dtypes.float32: raise CompileError(f"f32 atomic only, got {val.dtype}")
  if _is_lds_ref(a.src[0]) or _is_scratch_ref(a.src[0]): raise CompileError("global atomic only")
  return x.ins(AMDOps.ATOMIC_ADD, src=(a.src[0], a.src[1], val))

def _need_yi(ctx:IselContext) -> bool:
  # lidx1/lidx2 → ENABLE_VGPR_WORKITEM_ID≥1 → gfx1100 USER_SGPR pad to 15 (elf.py).
  return any(u.op is Ops.SPECIAL and u.arg.startswith("lidx") and u.arg[-1] in "12" for u in ctx.func_args)

def _wgid_reg(ctx:IselContext, dim:int) -> Register:
  # HSA: workgroup IDs are placed immediately after user SGPRs.
  base = 15 if _need_yi(ctx) else 2
  return Register(f"s{base + dim}", base + dim, size=4)

def _special_reg(name:str, ctx:IselContext|None=None) -> Register:
  if len(name) != 5 or name[:4] not in ("lidx", "gidx") or name[-1] not in "012":
    raise CompileError(f"bad special {name}")
  if name[0] == "l": return LID[int(name[-1])]
  if ctx is None: return WGID[int(name[-1])]
  return _wgid_reg(ctx, int(name[-1]))

def _emit_lidx(dst, dim:int) -> list:
  # gfx11+: packed work-item IDs in v0 (X:0..9, Y:10..19, Z:20..29). Unpacked v1/v2 are unused.
  v0 = _reg_to_amd(LID[0])
  return [r3.v_bfe_u32(dst, v0, dim * 10, 10)]

def _kernarg_offset(ctx:IselContext, x:UOp) -> int:
  params = [u for u in ctx.func_args if u.op is Ops.PARAM]
  bufs = [u for u in params if u.arg.addrspace is not AddrSpace.ALU]
  vals = [u for u in params if u.arg.addrspace is AddrSpace.ALU]
  if x.arg.addrspace is AddrSpace.ALU:
    return len(bufs) * 8 + next(i for i,u in enumerate(vals) if u.arg == x.arg) * 4
  return next(i for i,u in enumerate(bufs) if u.arg == x.arg) * 8

def _is_scalar_source(x:UOp) -> bool:
  if _unwrap_const(x) is not None: return True
  return isinstance((reg:=greg(x)), Register) and all(c.index < 256 for c in reg.cons)

def _uniform_byte_extract(ctx:IselContext, x:UOp) -> UOp|None:
  if _iop(x) is not AMDOps.EXTRACT or not x.src or x.src[0].op is not Ops.INS or _iop(x.src[0]) is not AMDOps.LOAD or \
     x.src[0].dtype is not dtypes.uint8 or not all(_is_scalar_source(s) for s in x.src[0].src): return None
  if (lane:=_const_int(x.src[1])) is None: return None
  collects = ctx.scratch.setdefault("uniform_byte_collects", {})
  word_lane = lane // 4
  word = collects.setdefault((x.src[0], word_lane), UOp(Ops.INS,
    src=(x.src[0], _tconst(word_lane, dtypes.int32).rtag()), arg=(AMDOps.COLLECT, dtypes.uint32)))
  return UOp(Ops.INS,
    src=(word, _tconst((lane % 4) * 8, dtypes.uint32).rtag(), _tconst(8, dtypes.uint32).rtag()), arg=(AMDOps.BFE, x.dtype))

def _wants_uniform_sgpr(x:UOp) -> bool:
  if _iop(x) is AMDOps.COLLECT: return True
  if _iop(x) is AMDOps.SWHERE: return True
  if _iop(x) is AMDOps.EXTRACT and x.src and x.src[0].op is Ops.INS and _iop(x.src[0]) is AMDOps.LOAD and \
     x.src[0].dtype is dtypes.uint8 and all(_is_scalar_source(s) for s in x.src[0].src):
    return True
  if _iop(x) is AMDOps.CAST and (not x.src or x.src[0].dtype not in dtypes.ints): return False
  return x.dtype in dtypes.ints and _iop(x) in (AMDOps.MOV, AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.MULHI, AMDOps.MAX, AMDOps.CAST,
    AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.BFE, AMDOps.LSHL_OR, AMDOps.LSHL_ADD) and \
    all(_is_scalar_source(s) for s in x.src)

def _alloc_vregs(ctx:IselContext, x:UOp, sgpr_pool:tuple[Register, ...], vgpr_pool:tuple[Register, ...]) -> UOp|None:
  scalar_pool = SGPR32 if sgpr_pool == SGPR else sgpr_pool
  if x.op in (Ops.BUFFER, Ops.ALLOC):
    return x.replace(src=tuple(s.rtag() for s in x.src), tag=None) if x.addrspace is AddrSpace.REG else None
  iop = _iop(x)
  if iop is AMDOps.LOOP_CMP: return None
  if (uniform_extract:=_uniform_byte_extract(ctx, x)) is not None: return uniform_extract
  if isinstance(x.tag, tuple):
    if _wants_uniform_sgpr(x) and not all(c.index < 256 for c in x.tag[0].cons):
      return x.replace(tag=(ctx.vreg(scalar_pool),))
    return None
  if iop in (AMDOps.DEFINE, AMDOps.SCRATCH_SIZE, AMDOps.SCRATCH_ADDR, AMDOps.LDS_BASE,
               AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) or x.dtype is dtypes.void:
    return x.replace(tag=None) if iop in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) and x.tag is not None else None
  if iop is AMDOps.KERNARG: return x.replace(tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.PARAM:
    if x.arg.addrspace is AddrSpace.ALU: return x.replace(src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
    return x.replace(src=tuple(s.rtag() for s in x.src), tag=(ctx.vreg(sgpr_pool),))
  if x.op is Ops.SPECIAL:
    # gidx → WGID SGPR (s2 or s15). lidx → normal VGPR; MOV emit unpacks from packed v0.
    if x.arg.startswith("gidx"): return x.replace(tag=(ctx.vreg(_special_reg(x.arg, ctx)),))
    return None
  if iop is AMDOps.PACK_F16:
    # Vec-load PACK shares a pool with its mem src so prefer_phys / coalesce can alias.
    if _pack_f16_is_vec_load(x):
      if all(s.op is Ops.INS and _iop(s) is AMDOps.LLOAD for s in x.src):
        pools = _register_pools(ctx)
        return x.replace(tag=(ctx.vreg(pools.load_upcast if allow_upcast16() else pools.load),))
      return x.replace(tag=(ctx.vreg(vgpr_pool),))
    if (base := _pack_f16_identity_load(x)) is not None and isinstance(base.tag, tuple):
      return x.replace(tag=base.tag)
    # The disjoint high band protects live LDS/WMMA fragments. Generic kernels have
    # no such fixed-register overlap, and forcing a late output pack to v64+ can cut
    # occupancy even when only a few low VGPRs remain live.
    has_wmma = ctx.scratch.setdefault("has_wmma", any(u.op is Ops.WMMA or
      (u.op is Ops.INS and _iop(u) is AMDOps.WMMA) for u in ctx.uses))
    # AMD_PACK_F16_GENERAL removed (flash needs scalar LSTORE fix). HALF2_GLOBAL still widens pool.
    if getenv("AMD_FLASH_ACC_SMALL", 0): pack_pool = vgpr_pool
    else:
      pools = _register_pools(ctx)
      pack_pool = (pools.pack_upcast if allow_upcast16() else pools.pack) if has_wmma else vgpr_pool
    return x.replace(tag=(ctx.vreg(pack_pool),))
  if iop is AMDOps.LLOAD:
    pools = _register_pools(ctx)
    return x.replace(tag=(ctx.vreg(pools.load_upcast if allow_upcast16() else pools.load),))
  if _wants_uniform_sgpr(x): return x.replace(tag=(ctx.vreg(scalar_pool),))
  return x.replace(tag=(ctx.vreg(vgpr_pool),))

def _gated_load(addr:UOp, alt:UOp, gate:UOp, x:UOp) -> UOp|None:
  if addr.op is not Ops.INDEX or len(addr.src) != 2: return None
  safe_addr = addr.replace(src=(addr.src[0], gate.where(addr.src[1], addr.src[1].const_like(0))))
  return gate.where(safe_addr.load(), alt.cast(x.dtype) if alt.dtype != x.dtype else alt)

def _pow2_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.val <= 0 or c.val & (c.val - 1) or (x.dtype not in dtypes.uints and x.vmin < 0): return None
  return x & _tconst(c.val - 1, x.dtype)

class _AMDFastDivRenderer(Renderer):
  def __init__(self): super().__init__(Target("NULL", ""))
  def supported_dtypes(self) -> set[DType]: return {dtypes.int32, dtypes.uint32}

def _const_cdiv(x:UOp, c:UOp) -> UOp|None:
  return fast_idiv(_AMDFastDivRenderer(), x, c.val) if c.val > 0 and x.vmin >= 0 else None

def _const_cmod(x:UOp, c:UOp) -> UOp|None:
  if c.val <= 0 or x.vmin < 0: return None
  if (q:=_const_cdiv(x, c)) is None: return None
  return x - q * _tconst(c.val, x.dtype)

def _bool_not(x:UOp) -> UOp:
  return x.where(_tconst(False, dtypes.bool), _tconst(True, dtypes.bool))

def _u32_divmod(n:UOp, d:UOp, bits:int|None=None) -> tuple[UOp, UOp]:
  zero, one = _tconst(0, dtypes.uint32), _tconst(1, dtypes.uint32)
  q, r = zero, zero
  # Restoring division only needs the numerator's reachable bits. Symbolic LLM indices are
  # often bounded by max_context; building all 32 rounds makes their kernels enormous.
  # Algebraic bounds may exceed the dtype after index normalization, but the machine
  # numerator is still 32 bits. Shifts by 32+ would wrap on RDNA3 instead of reading zero.
  for i in range(min(32, max(1, int(n.vmax).bit_length() if bits is None else bits)) - 1, -1, -1):
    r = (r << _tconst(1, dtypes.uint32)) | ((n >> _tconst(i, dtypes.uint32)) & one)
    ge = _bool_not(r < d)
    q = q | ge.where(one << _tconst(i, dtypes.uint32), zero)
    r = ge.where(r - d, r)
  return q, r

def _u32_fast_divmod(n:UOp, d:UOp) -> tuple[UOp, UOp]:
  # Reciprocal-based unsigned division, following AMDGPU's 32-bit lowering.
  # Two correction rounds make the estimate exact while avoiding one restoring
  # division round per reachable numerator bit for symbolic launch dimensions.
  def mulhi(a:UOp, b:UOp) -> UOp:
    return UOp(Ops.INS, src=(a, b), arg=(AMDOps.MULHI, dtypes.uint32))
  zero, one = _tconst(0, dtypes.uint32), _tconst(1, dtypes.uint32)
  z = (d.cast(dtypes.float32).reciprocal() * _tconst(2**32 - 256, dtypes.float32)).cast(dtypes.uint32)
  z = z + mulhi(z, zero - d * z)
  q = mulhi(n, z)
  r = n - q * d
  for _ in range(2):
    lt = r < d
    q, r = lt.where(q, q + one), lt.where(r, r - d)
  return q, r

def _var_divmod(x:UOp, d:UOp, op:UOp) -> UOp|None:
  if x.dtype != d.dtype or x.dtype not in (dtypes.int32, dtypes.uint32): return None
  if x.dtype is dtypes.uint32:
    q, r = _u32_fast_divmod(x, d) if d.vmin > 0 else _u32_divmod(x, d)
    return q if op.op is Ops.CDIV else r
  if x.vmin >= 0 and d.vmin > 0:
    q, r = _u32_fast_divmod(x.cast(dtypes.uint32), d.cast(dtypes.uint32))
    return (q if op.op is Ops.CDIV else r).cast(dtypes.int32)
  if x.vmin >= 0 and d.vmax < 0:
    signed_zero = _tconst(0, dtypes.int32)
    gap = affine_int_bounds((signed_zero - d) - x)
    if gap is not None and gap[0] > 0: return signed_zero if op.op is Ops.CDIV else x
    zero = _tconst(0, dtypes.uint32)
    q, r = _u32_fast_divmod(x.cast(dtypes.uint32), zero - d.cast(dtypes.uint32))
    return ((zero - q) if op.op is Ops.CDIV else r).cast(dtypes.int32)
  zero = _tconst(0, dtypes.int32)
  xneg, dneg = x < zero, d < zero
  ax, ad = xneg.where(zero - x, x).cast(dtypes.uint32), dneg.where(zero - d, d).cast(dtypes.uint32)
  # Taking the unsigned magnitude also handles INT_MIN. A proven nonzero divisor
  # can share the reciprocal lowering; keep restoring division when zero is possible.
  q, r = _u32_fast_divmod(ax, ad) if d.vmin > 0 or d.vmax < 0 else \
    _u32_divmod(ax, ad, max(abs(int(x.vmin)), abs(int(x.vmax))).bit_length())
  q, r = q.cast(dtypes.int32), r.cast(dtypes.int32)
  return xneg.where(zero - r, r) if op.op is Ops.CMOD else (xneg ^ dneg).where(zero - q, q)

_UNIFORM_GRAPH_OPS = (Ops.CAST, Ops.BITCAST, Ops.NOOP, Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.CDIV, Ops.CMOD,
                      Ops.RECIPROCAL, Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR, Ops.CMPLT, Ops.CMPNE,
                      Ops.CMPEQ, Ops.WHERE)
_UNIFORM_INS_OPS = (AMDOps.MOV, AMDOps.COLLECT, AMDOps.ADD, AMDOps.SUB, AMDOps.MUL, AMDOps.MULHI, AMDOps.MAX,
                    AMDOps.CAST, AMDOps.RECIPROCAL, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR,
                    AMDOps.XOR, AMDOps.SWHERE)

def _is_uniform_expr(x:UOp) -> bool:
  """True for values that are identical in every lane of a wave."""
  if _unwrap_const(x) is not None: return True
  if x.op is Ops.PARAM: return x.arg.addrspace is AddrSpace.ALU
  if x.op is Ops.SPECIAL: return x.arg.startswith("gidx")
  if x.op is Ops.INS: return _iop(x) in _UNIFORM_INS_OPS and all(_is_uniform_expr(s) for s in x.src)
  return x.op in _UNIFORM_GRAPH_OPS and all(_is_uniform_expr(s) for s in x.src)

def _scalarize_uniform_int_cast(y:UOp, x:UOp) -> UOp|None:
  # Float reciprocal/conversion instructions are vector-only on RDNA3. If their input is
  # wave-uniform, collect the converted result once so the integer correction stays SALU.
  if not _is_uniform_expr(y): return None
  converted = UOp(Ops.INS, src=(y,), arg=(AMDOps.CAST, x.dtype))
  return UOp(Ops.INS, src=(converted, _tconst(0, dtypes.int32).rtag()), arg=(AMDOps.COLLECT, x.dtype))

def _scalarize_uniform_where(m:UOp, a:UOp, b:UOp, x:UOp) -> UOp|None:
  # Normalize a boolean comparison of another comparison before SWHERE preserves the
  # outer operands. Comparison results live in flags, so they cannot themselves be an
  # SGPR operand to the scalar compare emitted for SWHERE.
  if m.op in (Ops.CMPNE, Ops.CMPEQ):
    for inner, c in ((m.src[0], m.src[1]), (m.src[1], m.src[0])):
      if inner.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and _const_value(c) in (False, True, 0, 1):
        m = _cmp_bool_const(m, inner, c)
        break
  if m.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and any(s.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) for s in m.src):
    m = m.replace(src=tuple(_bool_flag(s) if s.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) else s for s in m.src))
  if x.dtype not in dtypes.ints+(dtypes.bool,) or \
     not all(_is_uniform_expr(v) for v in (m, a, b)): return None
  # Comparisons carry their result in an implicit flag, so also keep their operands as
  # direct dependencies. Otherwise regalloc may reuse an operand before SWHERE rematerializes it.
  return UOp(Ops.INS, src=(m, a, b, *m.src) if m.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) else (m, a, b),
             arg=(AMDOps.SWHERE, x.dtype))

def _narrow_var_divmod(x:UOp, d:UOp, op:UOp) -> UOp|None:
  if x.dtype != d.dtype or x.dtype not in dtypes.ints or x.dtype.itemsize >= 4: return None
  wide = dtypes.int32 if x.dtype in dtypes.sints else dtypes.uint32
  return UOp(op.op, src=(x.cast(wide), d.cast(wide))).cast(x.dtype)

def _cmp_bool_const(x:UOp, m:UOp, c:UOp) -> UOp:
  keep = (x.op is Ops.CMPNE and not bool(c.val)) or (x.op is Ops.CMPEQ and bool(c.val))
  return m if keep else m.where(_tconst(False, dtypes.bool), _tconst(True, dtypes.bool))

def _bool_flag(x:UOp) -> UOp:
  return x.where(_tconst(True, dtypes.bool), _tconst(False, dtypes.bool))

def _materialize_flags(x:UOp, idx:tuple[int, ...]|None=None) -> UOp|None:
  src, changed = list(x.src), False
  for i in idx or range(len(src)):
    if src[i].op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ):
      src[i] = _bool_flag(src[i])
      changed = True
  return x.replace(src=tuple(src)) if changed else None

def _materialize_store_compare_flag(x:UOp) -> UOp|None:
  return _materialize_flags(x, (1,)) if len(x.src) >= 2 and x.src[1].op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) else None

def _cast_store_value(x:UOp, a:UOp, val:UOp) -> UOp|None:
  # C-style renderers implicitly convert through the destination pointer type. ISA stores
  # must make that conversion explicit or they also select the wrong width/address scale.
  return x.replace(src=(a, val.cast(a.dtype), *x.src[2:])) if val.dtype != a.dtype else None

def _materialize_bool_where(m:UOp, a:UOp, b:UOp) -> UOp|None:
  if m.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ): return None
  return UOp(Ops.WHERE, src=(UOp(Ops.CMPNE, src=(m, _tconst(False, dtypes.bool))), a, b))

def _where_ins(m:UOp, a:UOp, b:UOp, x:UOp) -> UOp:
  # Gated vector loads introduce WHEREs after pre-isel. A materialized boolean
  # register does not own VCC; compare it explicitly before selecting a value.
  if m.op not in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) and \
     (m.op is not Ops.INS or _iop(m) not in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ)):
    m = UOp(Ops.INS, src=(m, _tconst(False, dtypes.bool)), arg=(AMDOps.CMPNE, dtypes.bool))
  return x.ins(AMDOps.WHERE, src=(m, a, b))

def _merge_zero_cmp_and(a:UOp, b:UOp) -> UOp|None:
  def nonzero_src(x:UOp) -> UOp|None:
    if x.op is Ops.WHERE and len(x.src) == 3 and x.src[1].op is Ops.CONST and x.src[1].val is True and \
       x.src[2].op is Ops.CONST and x.src[2].val is False: x = x.src[0]
    # A merged equality can be materialized and negated while its parent boolean tree is still being rewritten:
    # CMPNE(WHERE(CMPNE(v, 0), true, false), true) is v == 0.
    if x.op is Ops.CMPNE:
      inner = x.src[0] if _const_value(x.src[1]) is True else x.src[1] if _const_value(x.src[0]) is True else None
      if inner is not None and inner.op is Ops.WHERE and _const_value(inner.src[1]) is True and _const_value(inner.src[2]) is False:
        inner = inner.src[0]
      if inner is not None and inner.op is Ops.CMPNE:
        if _const_value(inner.src[0]) == 0: return inner.src[1]
        if _const_value(inner.src[1]) == 0: return inner.src[0]
    if x.op is not Ops.CMPEQ: return None
    if _const_value(x.src[0]) == 0: return x.src[1]
    if _const_value(x.src[1]) == 0: return x.src[0]
    return None
  def zero_srcs(x:UOp) -> list[UOp]|None:
    if (v:=nonzero_src(x)) is not None: return [v]
    if x.op is not Ops.AND: return None
    left, right = zero_srcs(x.src[0]), zero_srcs(x.src[1])
    return None if left is None or right is None else left + right
  vals_a, vals_b = zero_srcs(a), zero_srcs(b)
  if vals_a is None or vals_b is None: return None
  vals = vals_a + vals_b
  if any(v.dtype not in dtypes.ints or v.dtype.itemsize != 4 for v in vals): return None
  merged = vals[0].cast(dtypes.uint32)
  for val in vals[1:]: merged = merged | val.cast(dtypes.uint32)
  return merged.eq(merged.const_like(0))

def _is_foldable(ctx:IselContext, x:UOp, s:UOp) -> bool: return len(ctx.uses[s]) == x.src.count(s) == 1

def _fused_mulacc(ctx:IselContext, a:UOp, b:UOp, c:UOp) -> UOp|None:
  if not _is_foldable(ctx, c, a): return None
  # Single-use addends can be overwritten in place with compact VOP2 FMAC. Keep
  # VOP3 FMA when the addend remains live elsewhere.
  # FMAC is a two-address op: the accumulator must already be in its destination. A
  # literal addend has no VGPR to tie, so keep it as a three-source FMA instead.
  return a.ins(AMDOps.FMAC, src=(b, *a.src)) if _const_value(b) is None and _is_foldable(ctx, c, b) else a.ins(AMDOps.MULACC, src=(*a.src, b))

def _protect_loop_invariant_fmac(lst:list[UOp]) -> list[UOp]:
  return _sched.protect_loop_invariant_fmac(lst)

def _promote_f16_unary(x:UOp, d:UOp) -> UOp:
  return UOp(x.op, src=(d.cast(dtypes.float32),)).cast(dtypes.float16)

def _int_cast(y:UOp, x:UOp) -> UOp|None:
  if x.dtype.itemsize == y.dtype.itemsize: return x.replace(op=Ops.BITCAST)
  return x.ins(AMDOps.CAST, src=(y,))

def _fuse_signed_byte_load_cast(x:UOp, y:UOp) -> UOp|None:
  if len(y.src) != 1 or (ld := y.src[0]).op is not Ops.LOAD or ld.dtype is not dtypes.uint8: return None
  return ld.bitcast(dtypes.int8).cast(x.dtype)

def _fuse_negated_add(a:UOp, b:UOp, x:UOp) -> UOp:
  return x.ins(AMDOps.SUB, src=(a, b))

pre_isel_matcher = PatternMatcher([
  ((UPat.var("a", dtypes.ints) + UPat.var("b", dtypes.ints) * -1).named("x"), _fuse_negated_add),
  (UPat(Ops.CAST, dtype=dtypes.float32,
   src=(UPat(Ops.AND, dtype=dtypes.uint32,
     src=(UPat(Ops.SHR, src=(UPat.var("value"), UPat.var("shift"))), UPat.cvar().cast(name="mask"))),),
   name="x"), _cvt_ubyte_f32),
  (UPat(Ops.CAST, dtype=dtypes.float32,
   src=(UPat(Ops.AND, dtype=dtypes.uint32, src=(UPat.var("value"), UPat.cvar().cast(name="mask"))),),
   name="x"), _cvt_ubyte_f32),
  (UPat(Ops.AND, dtype=(dtypes.uint8, dtypes.uint32),
   src=(UPat(Ops.SHR, src=(UPat.var("value"), UPat.var("shift"))), UPat.cvar().cast(name="mask")),
   name="x"), _bitfield_extract),
  (UPat(Ops.OR, dtype=(dtypes.uint8, dtypes.uint32), src=(UPat(Ops.SHL, src=(UPat.var("value"), UPat.var("shift"))), UPat.var("other")),
   name="x"), _lshl_or),
  (UPat(Ops.OR, dtype=(dtypes.uint8, dtypes.uint32), src=(UPat.var("other"), UPat(Ops.SHL, src=(UPat.var("value"), UPat.var("shift")))),
   name="x"), _lshl_or),
  (UPat(Ops.ADD, dtype=dtypes.uint32, src=(UPat(Ops.SHL, src=(UPat.var("value"), UPat.var("shift"))), UPat.var("other")),
   name="x"), _lshl_add),
  (UPat(Ops.ADD, dtype=dtypes.uint32, src=(UPat.var("other"), UPat(Ops.SHL, src=(UPat.var("value"), UPat.var("shift")))),
   name="x"), _lshl_add),
  (UPat(Ops.CUSTOMI, name="x"), _nontemporal_load),
  (UPat(Ops.INDEX, name="addr").load(UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool), name="x"), _gated_load),
  (UPat((Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN), dtype=dtypes.float16, src=(UPat.var("d"),), name="x"),
   _promote_f16_unary),
  (UPat(Ops.CDIV, src=(UPat.var("x", dtypes.ints), UPat.cvar().cast(name="c"))), _const_cdiv),
  (UPat(Ops.CMOD, src=(UPat.var("x", dtypes.ints), UPat.cvar().cast(name="c"))), _pow2_cmod),
  (UPat(Ops.CMOD, src=(UPat.var("x", dtypes.ints), UPat.cvar().cast(name="c"))), _const_cmod),
  (UPat((Ops.CDIV, Ops.CMOD), src=(UPat.var("x", dtypes.ints), UPat.var("d", dtypes.ints)), name="op"), _narrow_var_divmod),
  (UPat((Ops.CDIV, Ops.CMOD), src=(UPat.var("x", (dtypes.int32, dtypes.uint32)), UPat.var("d", (dtypes.int32, dtypes.uint32))), name="op"),
   _var_divmod),
  (UPat.var("y", dtypes.float32).cast(dtypes.ints, name="x"), _scalarize_uniform_int_cast),
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")).named("x"), _scalarize_uniform_where),
  (UPat((Ops.CMPNE, Ops.CMPEQ),
   src=(UPat((Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ), name="m"), UPat.cvar("c").cast(dtypes.bool, name="c")), name="x"),
   _cmp_bool_const),
  (UPat(Ops.AND, dtype=dtypes.bool, src=(UPat.var("a"), UPat.var("b"))), _merge_zero_cmp_and),
  (UPat((Ops.AND, Ops.OR, Ops.XOR, Ops.CMPNE, Ops.CMPEQ), dtype=dtypes.bool, name="x"), _materialize_flags),
  (UPat(Ops.STORE, name="x"), _materialize_store_compare_flag),
  (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("val")), allow_any_len=True, name="x"), _cast_store_value),
  (UPat(Ops.WHERE, name="x"), lambda x: _materialize_flags(x, (1, 2))),
  (UPat.var("m", dtypes.bool).cast(dtypes.ints+(dtypes.float16, dtypes.float32), name="x"),
   lambda m,x: m.where(_tconst(1, x.dtype), _tconst(0, x.dtype))),
  (UPat.var("m", dtypes.bool).where(UPat.var("a"), UPat.var("b")), _materialize_bool_where),
  (UPat(Ops.CAST, dtype=dtypes.float32,
        src=(UPat((Ops.NOOP, Ops.BITCAST), dtype=dtypes.int8, name="y"),), name="x"), _fuse_signed_byte_load_cast),
  (UPat.var("y", dtypes.ints).cast(dtypes.ints, name="x"), _int_cast),
  (UPat.var("y", dtypes.ints).cast(dtypes.float16), lambda y: y.cast(dtypes.float32).cast(dtypes.float16)),
  (UPat.var("y", dtypes.float16).cast(dtypes.ints, name="x"), lambda y,x: y.cast(dtypes.float32).cast(x.dtype)),
])

def make_isel_matcher(sgpr_pool:tuple[Register, ...]=SGPR, vgpr_pool:tuple[Register, ...]=VGPR) -> PatternMatcher:
  return PatternMatcher([
    # Regalloc creates canonical CAST(CONST(weak)) stack offsets after pre-isel has run.
    (UPat((Ops.ADD, Ops.SUB), src=(UPat(Ops.INS, arg=(AMDOps.SCRATCH_BASE, dtypes.uint32)), UPat.cvar("size").cast())),
     lambda size: UOp(Ops.INS, src=(size,), arg=(AMDOps.SCRATCH_SIZE, dtypes.void))),
    (UPat((Ops.ADD, Ops.SUB), src=(UPat(Ops.INS, arg=(AMDOps.SCRATCH_BASE, dtypes.uint32)), UPat.cvar()), name="x"),
     lambda x: UOp(Ops.INS, src=(x.src[1],), arg=(AMDOps.SCRATCH_SIZE, dtypes.void))),
    (UPat(Ops.INDEX, src=(UPat(Ops.INS, arg=(AMDOps.SCRATCH_BASE, dtypes.uint32)), UPat.cvar("off").cast())),
     lambda off: UOp(Ops.INS, src=(off,), arg=(AMDOps.SCRATCH_ADDR, dtypes.uint32))),
    (UPat(Ops.INDEX, src=(UPat(Ops.INS, arg=(AMDOps.SCRATCH_BASE, dtypes.uint32)), UPat.cvar("off")), name="x"),
     lambda off,x: UOp(Ops.INS, src=(off,), arg=(AMDOps.SCRATCH_ADDR, dtypes.uint32))),
    (UPat(Ops.RANGE, src=(UPat.cvar().cast(name="c"),), allow_any_len=True, name="x"), lambda c,x:
     x.replace(src=(_tconst(c.val, dtypes.uint32).rtag(),) + x.src[1:])),
    (UPat(Ops.RANGE, name="x"), lambda ctx,x,sgpr_pool=sgpr_pool:
     x.replace(tag=(ctx.vreg(sgpr_pool),))
     if not isinstance(x.tag, tuple) else None),
    (UPat(Ops.PARAM, name="x"), lambda ctx,x:
     UOp(Ops.INS, src=(_tconst(_kernarg_offset(ctx, x), dtypes.int32).rtag(),),
         arg=(AMDOps.KERNARG, dtypes.uint64 if x.arg.addrspace is not AddrSpace.ALU else x.dtype))
     if not isinstance(x.tag, tuple) else None),
    (UPat((Ops.BUFFER, Ops.ALLOC), name="x"), lambda ctx,x: _lds_base(ctx, x)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x,vgpr_pool=vgpr_pool:
     None if x.tag is not None else
     UOp(Ops.INS, src=(x.rtag(),), arg=(AMDOps.MOV, dtypes.uint32),
         tag=(ctx.vreg(vgpr_pool if x.arg.startswith("lidx") else _special_reg(x.arg, ctx)),))),
    # A boundless RANGE is a loop label; BACKEDGE carries the condition (was Ops.END).
    # Preserve the comparison operands and the RANGE edge until after regalloc so
    # the compare can be emitted immediately before the VCC conditional branch.
    (UPat(Ops.BACKEDGE, src=(UPat(), UPat(), UPat(GroupOp.Comparison, name="cond")), name="x"),
     lambda x,cond: cond.ins(AMDOps.LOOP_CMP, tag=cond.op, src=cond.src + x.src[:2])),
    # Legacy END-with-cond shape (pre-BACKEDGE); keep until fixtures migrate.
    (UPat(Ops.END, src=(UPat(), UPat(), UPat(GroupOp.Comparison, name="cond")), name="x"),
     lambda x,cond: cond.ins(AMDOps.LOOP_CMP, tag=cond.op, src=cond.src + x.src[:2])),
    (UPat(Ops.INDEX, name="x"), _extract_vec_lane),
    (UPat(Ops.STACK, name="x"), _pack_vec),
    # Int/bool CONST stay as CONST (_src inlines / _vgpr_data temps at use). Avoids
    # long-lived VGPR MOVs that dominate UPCAST4 spill. Float still needs a MOV VGPR.
    (UPat.cvar().cast(dtypes.float32, name="x"), lambda x:
     x.ins(AMDOps.MOV, src=(x.rtag(),)) if not x.tag else None),
    ((UPat(Ops.MUL, (dtypes.float16, dtypes.float32), name="a") + UPat.var("b")).named("c"), _fused_mulacc),
    ((UPat(dtype=dtypes.ints+(dtypes.bool, dtypes.float16, dtypes.float32)) + UPat()).named("x"), lambda x: x.ins(AMDOps.ADD)),
    (UPat(Ops.SUB, dtype=dtypes.ints+(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.SUB)),
    ((UPat(dtype=dtypes.ints+(dtypes.float16, dtypes.float32)) * UPat()).named("x"), lambda x: x.ins(AMDOps.MUL)),
    (UPat(Ops.MULACC, dtype=(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.MULACC)),
    *((UPat.var("y", fr).cast(to, name="x"), lambda y,x,op=op: x.ins(op, src=(y,)))
      for fr,to,op in ((dtypes.float16, dtypes.float32, AMDOps.CAST), (dtypes.float32, dtypes.float16, AMDOps.CAST),
                       (dtypes.ints, dtypes.float32, AMDOps.CAST), (dtypes.float32, dtypes.ints, AMDOps.CAST))),
    *((UPat(op, dtype=dtypes.float32, name="x"), lambda x,op=op,amd=amd: x.ins(amd)) for op,amd in _ISEL_UNARY.items()),
    (UPat(Ops.MAX, dtype=dtypes.ints+(dtypes.float16, dtypes.float32), name="x"), lambda x: x.ins(AMDOps.MAX)),
    ((UPat(dtype=dtypes.ints) << UPat()).named("x"), lambda x: x.ins(AMDOps.SHL)),
    ((UPat(dtype=dtypes.ints) >> UPat()).named("x"), lambda x: x.ins(AMDOps.SHR)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) & UPat()).named("x"), lambda x: x.ins(AMDOps.AND)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) | UPat()).named("x"), lambda x: x.ins(AMDOps.OR)),
    ((UPat(dtype=dtypes.ints+(dtypes.bool,)) ^ UPat()).named("x"), lambda x: x.ins(AMDOps.XOR)),
    (UPat(Ops.CMPLT, name="x"), lambda x: x.ins(AMDOps.CMPLT, dtype=dtypes.bool)),
    (UPat(Ops.CMPNE, name="x"), lambda x: x.ins(AMDOps.CMPNE, dtype=dtypes.bool)),
    (UPat(Ops.CMPEQ, name="x"), lambda x: x.ins(AMDOps.CMPEQ, dtype=dtypes.bool)),
    (UPat.var("m").where(UPat.var("a"), UPat.var("b")).named("x"), _where_ins),
    (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("alt"), UPat.var("gate", dtype=dtypes.bool)), name="x"), _load_ins),
    (UPat(Ops.LOAD, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"),), name="x"), _load_ins),
    (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("val")), name="x"), _store_ins),
    (UPat((Ops.CUSTOM, Ops.CUSTOMI), name="x"), _amd_custom_intrinsic),
    (UPat(Ops.CUSTOM, name="x"), _atomic_add_ins),
    (UPat(Ops.BARRIER, name="x"), lambda x: x.ins(AMDOps.BARRIER)),
    (UPat(Ops.WMMA, name="x"), lambda ctx,x: _isel_wmma(ctx, x)),
    (UPat((Ops.INS, Ops.BUFFER, Ops.ALLOC), name="x"),
     lambda ctx,x,sgpr_pool=sgpr_pool,vgpr_pool=vgpr_pool: _alloc_vregs(ctx, x, sgpr_pool, vgpr_pool)),
  ])

isel_matcher = make_isel_matcher()

def _loop_label(x:UOp) -> str: return "_".join(str(i) for i in x.axis_id)

def _lower_range(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = _loop_label(x)
  label = UOp(Ops.INS, arg=(AMDOps.LABEL, dtypes.void), tag=f".LOOP_{loop_label}")
  if x.dtype is dtypes.void: return label, [label]
  acc = x.ins(AMDOps.MOV, dtype=dtypes.uint32, src=(_tconst(0, dtypes.uint32).rtag(),))
  cmp = UOp(Ops.INS, src=(acc, x.src[0]), arg=(AMDOps.CMP_GE, dtypes.void))
  jump_out = UOp(Ops.INS, src=(cmp,), arg=(AMDOps.CBRANCH_SCC1, dtypes.void), tag=f".LOOP_OUT_{loop_label}")
  ctx.loop_label[acc] = loop_label
  return acc, [acc, label, cmp, jump_out]

def _lower_end(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  loop_label = ctx.loop_label[x.src[1]]
  jmp = UOp(Ops.INS, arg=(AMDOps.BRANCH, dtypes.void), tag=f".LOOP_{loop_label}")
  return jmp, [
    x.src[1].ins(AMDOps.ADD, dtype=dtypes.uint32, src=(x.src[1], _tconst(1, dtypes.uint32).rtag())),
    jmp,
    UOp(Ops.INS, arg=(AMDOps.LABEL, dtypes.void), tag=f".LOOP_OUT_{loop_label}")]

def _lower_loop_cmp(x:UOp) -> tuple[UOp, list[UOp]]:
  if len(x.src) == 3 and x.src[0].op is Ops.INS and _iop(x.src[0]) in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ):
    cmp, loop = x.src[0], x.src[2]
  else:
    cmp_op = {Ops.CMPLT:AMDOps.CMPLT, Ops.CMPNE:AMDOps.CMPNE, Ops.CMPEQ:AMDOps.CMPEQ}[x.tag]
    cmp, loop = UOp(Ops.INS, src=x.src[:2], arg=(cmp_op, dtypes.bool)), x.src[3]
  branch = UOp(Ops.INS, src=(cmp,), arg=(AMDOps.CBRANCH_VCCNZ, dtypes.void), tag=loop.tag)
  return branch, [branch] if len(x.src) == 3 else [cmp, branch]

def _lower_reg_store(x:UOp) -> tuple[UOp, list[UOp]]:
  acc, val = x.src
  # Prefer the (post-regalloc) dest tag when present; src[0] may be a bind/fill alias.
  dest = x.tag[0] if isinstance(x.tag, tuple) and x.tag else greg(acc)
  if not isinstance(dest, Register) or dest.index < 256:
    raise CompileError(f"bad reg store dest {dest} acc {acc}")
  # Always redefine the phys: allocator treats REG_STORE as a two-address def.
  st = UOp(Ops.INS, src=(val,), arg=(AMDOps.MOV, val.dtype), tag=(dest,))
  if acc.op is Ops.INS and _iop(acc) is AMDOps.FILL:
    # Spilled acc: keep scratch in sync with the updated phys (MOV alone left scratch stale
    # the other way; SPILL-only left the phys stale — both must match).
    sp = UOp(Ops.INS, src=(acc.src[0], st), arg=(AMDOps.SPILL, dtypes.void))
    return st, [st, sp]
  return st, [st]

post_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INS, arg=(AMDOps.LOOP_CMP, dtypes.bool), name="x"), _lower_loop_cmp),
  (UPat(Ops.RANGE, name="x"), lambda ctx,x: _lower_range(ctx, x)),
  (UPat(Ops.END, name="x"), lambda ctx,x: _lower_end(ctx, x)),
  (UPat(Ops.INS, arg=(AMDOps.REG_STORE, dtypes.void), name="x"), lambda x: _lower_reg_store(x)),
  # STRUCTURAL: INDEX/SHRINK/CAST/CONST etc. must not become LINEAR statements. Regalloc returns
  # None for SHRINK/LOAD/STORE → line_rewrite's `or (nu,[nu])` would otherwise re-emit them and
  # break do_assemble's all-INS match (flash_decode_partial ELF regression after master merge).
  (UPat((Ops.CONST, Ops.CAST, Ops.BITCAST, Ops.NOOP, Ops.AFTER, Ops.SPECIAL, Ops.SINK,
         Ops.SHRINK, Ops.INDEX, Ops.LOAD, Ops.STORE), name="x"),
   lambda x: (x, [])),
])

def _vcc_rematerialize(ctx, x:UOp):
  _flags = (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ)
  flag_def = x if x.op is Ops.INS and _iop(x) in _flags else \
             x.src[0] if _iop(x) in (AMDOps.WHERE, AMDOps.IF_MASK) and x.src[0].op is Ops.INS and _iop(x.src[0]) in _flags else None
  if flag_def is None: return None
  # VCC is implicit; rematerialize compares before WHERE/IF_MASK consumers
  if flag_def is not x:
    # A directly adjacent compare still owns VCC; emitting it twice only extends the dependency chain.
    if ctx.uops is not None and ctx.regalloc_i > 0 and ctx.uops[ctx.regalloc_i-1] is flag_def: return x, [x]
    return x, [flag_def, x]
  if ctx.lock is not None and ctx.lock is not flag_def: ctx.clobbered.add(ctx.lock)
  ctx.lock = flag_def
  if flag_def not in ctx.clobbered: return None
  ctx.clobbered.remove(flag_def)
  return x, [flag_def, x]

def _lower_late_index(x:UOp) -> tuple[UOp, list[UOp]]: return x, []
def _store_addr(a:UOp) -> UOp:
  return a if a.op in (Ops.INDEX, Ops.SHRINK) else UOp(Ops.INDEX, src=(a, _tconst(0, dtypes.int32).rtag()))

def _lower_late_store(ctx, x:UOp, a:UOp, val:UOp, gate:UOp|None=None) -> tuple[UOp, list[UOp]]:
  if x in _amd_skip(ctx): return x, []
  st = _store_ins(x, _store_addr(a), val)
  if gate is None: return st, [st]
  n = ctx.scratch.get("exec_mask_n", 0)
  ctx.scratch["exec_mask_n"] = n + 1
  mif = UOp(Ops.INS, src=(gate,), arg=(AMDOps.IF_MASK, dtypes.uint64),
            tag=(Register(f"exec_mask{n}", 0, _cons=SGPR),))
  remat = _vcc_rematerialize(ctx, mif)
  pre = remat[1] if remat is not None else [mif]
  mend = UOp(Ops.INS, src=(mif,), arg=(AMDOps.END_MASK, dtypes.void))
  return mend, [*pre, st, mend]

def _promote_wmma_acc_pack(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  """Redirect oversized WMMA acc reload PACK to the pre-loop zero-init with the same tag."""
  if not _is_wmma_acc_reload_pack(x, ctx): return None
  if not isinstance(x.tag, tuple) or not x.tag: return None
  inits = ctx.scratch.get("wmma_acc_inits") or {}
  if (init:=inits.get(x.tag)) is None: return None
  left = ctx.scratch.get("wmma_packs_left")
  if left is None: ctx.scratch["wmma_packs_left"] = left = len(inits)
  ctx.scratch["wmma_packs_left"] = left - 1
  if left - 1 == 0: ctx.scratch["wmma_past_acc"] = True
  return init, []  # cin aliases pre-loop zero pack; physical ACC updated in-place by WMMA

def _promote_reg_buffer(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x.addrspace is not AddrSpace.REG or x not in _reg_promotable_buffers(ctx): return None
  return x, []

def _promote_reg_access(ctx:PreRegAllocContext, x:UOp) -> tuple[UOp, list[UOp]]|None:
  if x in _amd_skip(ctx): return x, []
  if _iop(x) is AMDOps.SSTORE:
    if (buf:=_reg_buffer_base(x.src[0])) is not None and buf in _wmma_acc_buffers(ctx):
      # K-loop WMMA results use dynamic REG indices: ACC is updated in-place by two-address
      # WMMA; leave the SSTORE (dead scratch) alone so promote does not invent a VGPR.
      # Flash ACC_SMALL scale/mask use const indices and must write back into ACC lanes —
      # otherwise SLOAD EXTRACTs stale ACC while SSTORE only updates scratch (err=inf).
      idx = _const_int(x.src[1])
      bo = _lds_byte_off(x)
      if idx is not None and x.src[2].dtype.itemsize and bo % x.src[2].dtype.itemsize == 0:
        idx = idx + bo // x.src[2].dtype.itemsize
      elif bo:
        idx = None
      if idx is None: return x, []
      if not ctx.scratch.get("wmma_past_acc") and not _wmma_acc_small_mode(ctx): return x, []
      if (got:=_wmma_acc_lane(ctx, buf, idx)) is None: return None
      init, lane = got
      ext = _wmma_acc_extract(ctx, init, lane)
      # Tag REG_STORE as a redef of the ACC lane so spills write the new value (KGEN-style
      # loop-carried promote). Untagged REG_STORE was only a use → stale spill restores.
      st = UOp(Ops.INS, src=(ext, x.src[2]), arg=(AMDOps.REG_STORE, dtypes.void), tag=(greg(ext),))
      return ext, [ext, st]
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1], _lds_byte_off(x), x.src[2].dtype.itemsize)) is None: return None
    val = x.src[2]
    reg_values = ctx.scratch["reg_values"]
    if slot not in reg_values:
      reg_values[slot] = acc = _new_promoted_reg(ctx, val, _promote_reg_cons(ctx, slot[0]))
      return acc, [acc]
    acc = reg_values[slot]
    # Same vreg as the first promote MOV; tag makes this a defining redef for linear-scan
    # live ranges and spill-after-write (REDUCE-carried flash acc / m_i / soft).
    st = UOp(Ops.INS, src=(acc, val), arg=(AMDOps.REG_STORE, dtypes.void), tag=(greg(acc),))
    return acc, [st]
  if _iop(x) is AMDOps.SLOAD:
    if (buf:=_reg_buffer_base(x.src[0])) is not None and buf in _wmma_acc_buffers(ctx):
      # Loop-body SLOADs only feed WMMA PACK (redirected); post-loop reads need EXTRACT.
      # Flash ACC_SMALL: allow mid-kernel const-index EXTRACT for flash ACC_SEP copies.
      idx = _const_int(x.src[1])
      bo = _lds_byte_off(x)
      if idx is not None and x.dtype.itemsize and bo % x.dtype.itemsize == 0:
        idx = idx + bo // x.dtype.itemsize
      elif bo:
        idx = None
      if idx is None:
        if not ctx.scratch.get("wmma_past_acc"): return x, []
        return None
      if not ctx.scratch.get("wmma_past_acc") and not _wmma_acc_small_mode(ctx): return x, []
      if (got:=_wmma_acc_lane(ctx, buf, idx)) is None: return None
      init, lane = got
      ext = _wmma_acc_extract(ctx, init, lane)
      return ext, [ext]
    if (slot:=_reg_promote_slot(ctx, x.src[0], x.src[1], _lds_byte_off(x), x.dtype.itemsize)) is None: return None
    loaded = ctx.scratch["reg_values"].get(slot)
    if loaded is None: return None
    return loaded, []
  return None

def _lower_late_if(ctx, x:UOp) -> tuple[UOp, list[UOp]]:
  n = ctx.scratch.get("exec_mask_n", 0)
  ctx.scratch["exec_mask_n"] = n + 1
  mif = UOp(Ops.INS, src=(x.src[0],), arg=(AMDOps.IF_MASK, dtypes.uint64),
            tag=(Register(f"exec_mask{n}", 0, _cons=SGPR),))
  remat = _vcc_rematerialize(ctx, mif)
  return remat if remat is not None else (mif, [mif])

def _lower_late_endif(x:UOp) -> tuple[UOp, list[UOp]]:
  # Keep END_MASK after the guarded store. Source-less INS nodes are hoisted before regalloc.
  mend = UOp(Ops.INS, src=x.src, arg=(AMDOps.END_MASK, dtypes.void))
  return mend, [mend]

def _schedule_loop_cmps(lst:list[UOp]) -> list[UOp]:
  return _sched.schedule_loop_cmps(lst)

pre_regalloc_matcher = PatternMatcher([
  (UPat(Ops.INDEX, name="x"), _lower_late_index),
  (UPat(Ops.SHRINK, name="x"), lambda x: (x, [])),  # address mop; do not emit as a statement
  (UPat(Ops.STORE, src=(UPat.var("a"), UPat.var("val"), UPat.var("gate", dtype=dtypes.bool)), name="x"), _lower_late_store),
  (UPat(Ops.STORE, src=(UPat((Ops.INDEX, Ops.SHRINK), name="a"), UPat.var("val")), name="x"), _lower_late_store),
  (UPat(Ops.STORE, src=(UPat.var("a"), UPat.var("val")), name="x"), _lower_late_store),
  (UPat(Ops.INS, arg=(AMDOps.PACK, dtypes.float), name="x"), _promote_wmma_acc_pack),
  (UPat((Ops.BUFFER, Ops.ALLOC), name="x"), _promote_reg_buffer),
  (UPat(Ops.INS, name="x"), _promote_reg_access),
  (UPat(Ops.IF, name="x"), _lower_late_if),
  (UPat(Ops.ENDIF, name="x"), _lower_late_endif),
  (UPat(Ops.INS, name="x"), _vcc_rematerialize),
])


_ALU2: dict[AMDOps, tuple] = {
  AMDOps.ADD: (r3.v_add_f16_e32, r3.v_add_f32_e32, r3.s_add_u32, r3.v_add_nc_u32_e32),
  AMDOps.SUB: (r3.v_sub_f16_e32, r3.v_sub_f32_e32, r3.s_sub_u32, r3.v_sub_nc_u32_e64),
  AMDOps.MUL: (r3.v_mul_f16_e32, r3.v_mul_f32_e32, r3.s_mul_i32, r3.v_mul_lo_u32),
}
def _high_vgpr(*operands) -> bool: return any(isinstance(x, Reg) and x.offset >= 384 for x in operands)

def _alu2(u:UOp):
  f16, f32, sgpr, vint = _ALU2[_iop(u)]
  d = _dst(u)
  sc = u.dtype
  if sc in (dtypes.float16, dtypes.float32):
    inst, a, b = (f16 if sc is dtypes.float16 else f32), _src(u.src[0]), _src(u.src[1])
    if sc is dtypes.float16 and _high_vgpr(d, a, b):
      inst = {AMDOps.ADD:r3.v_add_f16_e64, AMDOps.SUB:r3.v_sub_f16_e64, AMDOps.MUL:r3.v_mul_f16_e64}[_iop(u)]
    if isinstance(b, Reg) and b.offset >= 256: out = [inst(d, a, b)]
    elif _iop(u) in (AMDOps.ADD, AMDOps.MUL) and isinstance(a, Reg) and a.offset >= 256: out = [inst(d, b, a)]
    else:
      pre, b = _vgpr_data(TMP_VDATA, u.src[1])
      out = pre + [inst(d, a, b)]
    # HIP inserts s_delay_alu after many MULs before dependent ADD/EXP/FMAC.
    # MUL_DELAY folded into VOPD_FMAC (decode ship).
    if sc is dtypes.float32 and _iop(u) is AMDOps.MUL and getenv("AMD_VOPD_FMAC", 0):
      out.append(r3.s_delay_alu(1))
    return out
  if greg(u).index < 256: return [sgpr(d, _src(u.src[0]), _src(u.src[1]))]
  if _iop(u) is AMDOps.ADD:
    a, b = _src(u.src[0]), _src(u.src[1])
    if isinstance(b, Reg) and b.offset >= 256: return [vint(d, a, b)]
    if isinstance(a, Reg) and a.offset >= 256: return [vint(d, b, a)]
    pre, b = _vgpr_data(TMP_VDATA, u.src[1])
    return pre + [vint(d, a, b)]
  # VOP: one src must be VGPR; materialize src0 if it's an imm/SGPR.
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  return pre + [vint(d, a, _src(u.src[1]))]
def _max(u:UOp):
  d = _dst(u)
  sc = u.dtype
  if greg(u).index < 256:
    scalar_max = r3.s_max_i32 if sc in dtypes.sints else r3.s_max_u32
    return [scalar_max(d, _src(u.src[0]), _src(u.src[1]))]
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  b = _src(u.src[1])
  if sc is dtypes.float16: return pre + [(r3.v_max_f16_e64 if _high_vgpr(d, a, b) else r3.v_max_f16_e32)(d, a, b)]
  if sc is dtypes.float32: return pre + [r3.v_max_f32_e32(d, a, b)]
  if sc in dtypes.sints: return pre + [r3.v_max_i32_e64(d, a, b)]
  return pre + [r3.v_max_u32_e64(d, a, b)]
def _cmp_lt(u:UOp):
  pre, a = _vgpr_data(TMP_VDATA, u.src[0])
  sc = u.src[0].dtype
  if sc is dtypes.float16 and _high_vgpr(a, _src(u.src[1])): return pre + [r3.v_cmp_gt_f16_e64(VCC_LO, _src(u.src[1]), a)]
  if sc is dtypes.float16: cmp = r3.v_cmp_gt_f16_e32
  elif sc is dtypes.float32: cmp = r3.v_cmp_gt_f32_e32
  elif sc in dtypes.sints: cmp = r3.v_cmp_gt_i32_e32
  else: cmp = r3.v_cmp_gt_u32_e32
  return pre + [cmp(_src(u.src[1]), a)]
def _cmp_ne(u:UOp):
  sc = u.src[0].dtype
  cmp = r3.v_cmp_neq_f16_e32 if sc is dtypes.float16 else r3.v_cmp_neq_f32_e32 if sc is dtypes.float32 else r3.v_cmp_ne_u32_e32
  a, b = _src(u.src[0]), _src(u.src[1])
  if sc is dtypes.float16 and _high_vgpr(a, b): return [r3.v_cmp_neq_f16_e64(VCC_LO, a, b)]
  if isinstance(b, Reg) and b.offset >= 256: return [cmp(a, b)]
  if isinstance(a, Reg) and a.offset >= 256: return [cmp(b, a)]
  pre, b = _vgpr_data(TMP_VDATA, u.src[1])
  return pre + [cmp(a, b)]
def _cmp_eq(u:UOp):
  sc = u.src[0].dtype
  cmp = r3.v_cmp_eq_f16_e32 if sc is dtypes.float16 else r3.v_cmp_eq_f32_e32 if sc is dtypes.float32 else r3.v_cmp_eq_u32_e32
  a, b = _src(u.src[0]), _src(u.src[1])
  if sc is dtypes.float16 and _high_vgpr(a, b): return [r3.v_cmp_eq_f16_e64(VCC_LO, a, b)]
  if isinstance(b, Reg) and b.offset >= 256: return [cmp(a, b)]
  if isinstance(a, Reg) and a.offset >= 256: return [cmp(b, a)]
  pre, b = _vgpr_data(TMP_VDATA, u.src[1])
  return pre + [cmp(a, b)]

def _scalar_condition(m:UOp, operands:tuple[UOp, ...]=()) -> list:
  if m.op is Ops.INS and _iop(m) in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ) and all(_is_scalar_source(s) for s in m.src):
    cmp_src = operands if len(operands) == 2 else m.src
    a, b, sc = _src(cmp_src[0]), _src(cmp_src[1]), cmp_src[0].dtype
    if _iop(m) is AMDOps.CMPLT: cmp = r3.s_cmp_lt_i32 if sc in dtypes.sints else r3.s_cmp_lt_u32
    elif _iop(m) is AMDOps.CMPNE: cmp = r3.s_cmp_lg_i32 if sc in dtypes.sints else r3.s_cmp_lg_u32
    else: cmp = r3.s_cmp_eq_i32 if sc in dtypes.sints else r3.s_cmp_eq_u32
    return [cmp(a, b)]
  pre, gate = _sgpr_data(TMP_SDATA0, m)
  return pre + [r3.s_cmp_lg_u32(gate, 0)]

def _commutative_vop2(u:UOp, inst) -> list:
  a, b = _src(u.src[0]), _src(u.src[1])
  if isinstance(b, Reg) and b.offset >= 256: return [inst(_dst(u), a, b)]
  if isinstance(a, Reg) and a.offset >= 256: return [inst(_dst(u), b, a)]
  pre, b = _vgpr_data(TMP_VDATA, u.src[1])
  return pre + [inst(_dst(u), a, b)]

_MASKED_MEM = (AMDOps.LOAD, AMDOps.STORE, AMDOps.LLOAD, AMDOps.LSTORE, AMDOps.SLOAD, AMDOps.SSTORE)

# --- emit: per-UOp instruction lowering ---
def insts_for_uop(u:UOp, skip:set[UOp]|None=None, masked:bool=False, store_addr_cache:_StoreAddrCache|None=None,
                  d16_hi_lo:dict[UOp, UOp]|None=None, byte_scaled:set[int]|None=None,
                  fma_hi_lo:dict[UOp, UOp]|None=None, fma_pair_dst:dict[UOp, Reg]|None=None):
  if u.op is not Ops.INS or (skip and u in skip): return []
  if isinstance(_iop(u), Inst): return [_iop(u)]
  match _iop(u):
    case (AMDOps.LABEL | AMDOps.BRANCH | AMDOps.CBRANCH_SCC1 | AMDOps.CBRANCH_VCCNZ | AMDOps.DEFINE | AMDOps.SCRATCH_BASE |
          AMDOps.SCRATCH_SIZE | AMDOps.SCRATCH_ADDR | AMDOps.LDS_BASE):
      return []
    case AMDOps.KERNARG:
      if (off:=_const_int(u.src[0])) is None: raise CompileError("non-constant kernarg offset")
      load = r3.s_load_b64(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off) if u.dtype.itemsize == 8 else \
             r3.s_load_b32(sdata=_dst(u), sbase=KERNARG_REG, soffset=NULL, offset=off)
      return [load]
    case AMDOps.MOV:
      if not u.src: return []
      if u.src[0].op is Ops.SPECIAL:
        name = u.src[0].arg
        if name.startswith("lidx"): return _emit_lidx(_dst(u), int(name[-1]))
        return []  # gidx coalesced onto WGID SGPR
      if (sregs:=_reg_idxs(u.src[0])) and sregs == _reg_idxs(u): return []
      if _reg_slots(u) > 1:
        if not isinstance(greg(u.src[0]), Register): raise CompileError(f"expected vec reg src {u}")
        return _parallel_vmov([(_reg_lane(greg(u), i), _reg_lane(greg(u.src[0]), i)) for i in range(_reg_slots(u))])
      # Peel MOV/CAST chains to inline float/int literals (flash zero/-inf broadcasts).
      # Avoids VGPR←VGPR copies of shared const mid-regs; enables more imm0 VOPD.
      if greg(u).index < 256: return [r3.s_mov_b32(_dst(u), _src(u.src[0]))]
      src = _mov_imm_or_src(u.src[0]) if True else _src(u.src[0])
      return [r3.v_mov_b32_e32(_dst(u), src)]
    case AMDOps.PACK:
      if u.dtype is not dtypes.float32: raise CompileError(f"f32 pack only, got {u.dtype}")
      return _parallel_vmov([(_reg_lane(greg(u), i), _src(s)) for i,s in enumerate(u.src)])
    case AMDOps.PACK_F16:
      return _pack_f16_insts(u, fma_hi_lo, fma_pair_dst)
    case AMDOps.WMMA:
      acc, src0, src1 = u.src[0], u.src[1], u.src[2]
      vdst = _reg_to_amd(greg(acc), 8)
      out = []
      # HIP flash: s_delay_alu after dual_mov/wait, before WMMA. Prefill wash with blanket
      # post-WMMA delay. AMD_WMMA_DELAY_PRE removed (no ship win).
      out.append(_wmma_inst(u)(vdst=vdst, src0=_reg_to_amd(greg(src0), 8), src1=_reg_to_amd(greg(src1), 8), src2=vdst))
      # HIP also inserts s_delay_alu after some WMMA; product-16 + flash prefill ship delay=1.
      # AMD_WMMA_DELAY getenv removed (always 1; AMD_WMMA_DELAY_PRE removed earlier).
      out.append(r3.s_delay_alu(1))
      return out
    case AMDOps.SWIZZLE:
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      if (offset:=_const_int(u.src[1])) is None: raise CompileError("non-constant swizzle offset")
      return pre + [r3.ds_swizzle_b32(vdst=_dst(u), addr=val, offset0=offset & 0xff, offset1=offset >> 8)]
    case AMDOps.PERMLANEX16:
      # VALU cross-16 (no lgkm). Lane-select SGPRs are installed by the emit streak hoist.
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_permlanex16_b32(_dst(u), val, TMP_SDATA0, TMP_SDATA1, opsel=1)]
    case AMDOps.DOT4:
      a, b = _src(u.src[0]), _src(u.src[1])
      pre0, a = ([], a) if not isinstance(a, Reg) else _vgpr_data(TMP_VDATA, u.src[0])
      pre1, b = ([], b) if not isinstance(b, Reg) else _vgpr_data(TMP_VADDR, u.src[1])
      if not isinstance(a, Reg) and not isinstance(b, Reg) and a != b: pre1, b = _vgpr_data(TMP_VADDR, u.src[1])
      return pre0 + pre1 + [r3.v_dot4_i32_iu8(_dst(u), a, b, _src(u.src[2]), neg=0b011)]
    case AMDOps.BYTE_PERM:
      pre0, a = _vgpr_data(TMP_VDATA, u.src[0])
      pre1, b = _vgpr_data(TMP_VADDR, u.src[1])
      return pre0 + pre1 + [r3.v_perm_b32(_dst(u), a, b, _src(u.src[2]))]
    case AMDOps.BFE:
      if greg(u).index < 256:
        shift, width = _const_int(u.src[1]), _const_int(u.src[2])
        if shift is None or width is None: raise CompileError("scalar bfe needs constant shift and width")
        scalar_bfe = r3.s_bfe_i32 if u.dtype in dtypes.sints else r3.s_bfe_u32
        return [scalar_bfe(_dst(u), _src(u.src[0]), shift | (width << 16))]
      pre, value = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_bfe_u32(_dst(u), value, _src(u.src[1]), _src(u.src[2]))]
    case AMDOps.CVT_UBYTE_F32:
      if (byte:=_const_int(u.src[1])) is None or not 0 <= byte < 4: raise CompileError("bad ubyte lane")
      return [(r3.v_cvt_f32_ubyte0_e32, r3.v_cvt_f32_ubyte1_e32,
               r3.v_cvt_f32_ubyte2_e32, r3.v_cvt_f32_ubyte3_e32)[byte](_dst(u), _src(u.src[0]))]
    case AMDOps.FMA_TO_F16:
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      if (lo := (fma_hi_lo or {}).get(u)) is not None:
        return pre + [r3.v_fma_mixhi_f16((fma_pair_dst or {}).get(u, _dst(lo)), a, _src(u.src[1]), _src(u.src[2]),
                                         opsel=0, opsel_hi=0, opsel_hi2=0)]
      return pre + [r3.v_fma_mixlo_f16((fma_pair_dst or {}).get(u, _dst(u)), a, _src(u.src[1]), _src(u.src[2]),
                                       opsel=0, opsel_hi=0, opsel_hi2=0)]
    case AMDOps.PACKED_F16_MUL_TO_F16:
      if (high:=_const_int(u.src[2])) not in (0, 1): raise CompileError("bad packed f16 lane")
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      if (lo := (fma_hi_lo or {}).get(u)) is not None:
        return pre + [r3.v_fma_mixhi_f16((fma_pair_dst or {}).get(u, _dst(lo)), a, _src(u.src[1]), 0.0,
                                         opsel=high, opsel_hi=1, opsel_hi2=0)]
      return pre + [r3.v_fma_mixlo_f16((fma_pair_dst or {}).get(u, _dst(u)), a, _src(u.src[1]), 0.0,
                                       opsel=high, opsel_hi=1, opsel_hi2=0)]
    case AMDOps.LSHL_OR:
      if greg(u).index < 256:
        return [r3.s_lshl_b32(_dst(u), _src(u.src[0]), _src(u.src[1])),
                r3.s_or_b32(_dst(u), _dst(u), _src(u.src[2]))]
      pre, value = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_lshl_or_b32(_dst(u), value, _src(u.src[1]), _src(u.src[2]))]
    case AMDOps.LSHL_ADD:
      if greg(u).index < 256:
        if (shift:=_const_int(u.src[1])) in (1, 2, 3, 4):
          return [(r3.s_lshl1_add_u32, r3.s_lshl2_add_u32, r3.s_lshl3_add_u32, r3.s_lshl4_add_u32)[shift-1](
            _dst(u), _src(u.src[0]), _src(u.src[2]))]
        return [r3.s_lshl_b32(_dst(u), _src(u.src[0]), _src(u.src[1])),
                r3.s_add_u32(_dst(u), _dst(u), _src(u.src[2]))]
      pre, value = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_lshl_add_u32(_dst(u), value, _src(u.src[1]), _src(u.src[2]))]
    case AMDOps.COLLECT:
      if (lane:=_const_int(u.src[1])) is None: raise CompileError("non-constant collect lane")
      if not isinstance(greg(u.src[0]), Register): raise CompileError(f"expected vec reg src {u}")
      return [r3.v_readfirstlane_b32_e32(_dst(u), _reg_lane(greg(u.src[0]), lane)), r3.s_delay_alu(1)]
    case AMDOps.EXTRACT:
      if (lane:=_const_int(u.src[1])) is None: raise CompileError("non-constant extract lane")
      src = u.src[0]
      if not isinstance(greg(src), Register): raise CompileError(f"expected vec reg src {u}")
      sc = src.dtype
      if sc in (dtypes.float32, dtypes.int32, dtypes.uint32):
        lane_src = _reg_lane(greg(src), lane)
        return [] if isinstance(greg(u), Register) and greg(u).index == greg(src).index+lane else [r3.v_mov_b32_e32(_dst(u), lane_src)]
      if sc is dtypes.uint8:
        lane_src, shift = _reg_lane(greg(src), lane // 4), (lane % 4) * 8
        if greg(u).index < 256:
          return [r3.v_readfirstlane_b32_e32(TMP_SDATA0, lane_src),
                  r3.s_bfe_u32(_dst(u), TMP_SDATA0, shift | (8 << 16))]
        return [r3.v_bfe_u32(_dst(u), lane_src, shift, 8)]
      if sc is dtypes.float16:
        # two f16s per VGPR; high lane needs a 16-bit shift
        slot, hi = divmod(int(lane), 2)
        lane_src = _reg_lane(greg(src), slot)
        if not hi:
          return [] if isinstance(greg(u), Register) and greg(u).index == greg(src).index+slot else [r3.v_mov_b32_e32(_dst(u), lane_src)]
        # Never LSHR in place: lo EXTRACT / PACK_F16 may still need the full half2 in src.
        if isinstance(greg(u), Register) and greg(u).index == greg(src).index+slot:
          return [r3.v_lshrrev_b32_e64(TMP_VDATA, 16, lane_src), r3.v_mov_b32_e32(_dst(u), TMP_VDATA)]
        return [r3.v_lshrrev_b32_e64(_dst(u), 16, lane_src)]
      raise CompileError(f"unsupported extract dtype {src.dtype}")
    case AMDOps.ADD | AMDOps.SUB | AMDOps.MUL:
      return _alu2(u)
    case AMDOps.MULHI:
      a, b = _src(u.src[0]), _src(u.src[1])
      if greg(u).index < 256: return [r3.s_mul_hi_u32(_dst(u), a, b)]
      pre, b = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_mul_hi_u32(_dst(u), a, b)]
    case AMDOps.MULACC:
      if u.dtype is dtypes.float16: return [r3.v_fma_f16(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
      if u.dtype is dtypes.float32: return [r3.v_fma_f32(_dst(u), _src(u.src[0]), _src(u.src[1]), _src(u.src[2]))]
      raise CompileError(f"f16/f32 mulacc only, got {u.dtype}")
    case AMDOps.FMAC:
      a, b = _src(u.src[1]), _src(u.src[2])
      inst = (r3.v_fmac_f16_e64 if _high_vgpr(_dst(u), a, b) else r3.v_fmac_f16_e32) if u.dtype is dtypes.float16 else \
             r3.v_fmac_f32_e32 if u.dtype is dtypes.float32 else None
      if inst is None: raise CompileError(f"f16/f32 fmac only, got {u.dtype}")
      if isinstance(b, Reg) and b.offset >= 256: out = [inst(_dst(u), a, b)]
      elif isinstance(a, Reg) and a.offset >= 256: out = [inst(_dst(u), b, a)]
      else:
        pre, b = _vgpr_data(TMP_VDATA, u.src[2])
        out = pre + [inst(_dst(u), a, b)]
      # HIP inserts s_delay_alu after many FMACs in the softmax/scale chains.
      # Every 4th FMAC (HIP ~150 delays vs blanket ~245). FMAC_DELAY folded into VOPD_FMAC.
      if u.dtype is dtypes.float32 and getenv("AMD_VOPD_FMAC", 0):
        _FMAC_DELAY_N[0] += 1
        if _FMAC_DELAY_N[0] % 4 == 0:
          out.append(r3.s_delay_alu(1))
      return out
    case AMDOps.FMA_MIX_F32:
      # srcs: (acc, half, f32) — half is mix src0 (opsel_hi=1). VOP3 accepts VGPR/SGPR.
      # AMD_FMA_MIX_PACKED_OPSEL removed (NaN with WHERE).
      acc_reg = _dst(u)
      half_u, f32_u = u.src[1], u.src[2]
      out = [r3.v_fma_mix_f32(acc_reg, _src(half_u), _src(f32_u), acc_reg, opsel=0, opsel_hi=1, opsel_hi2=0)]
      # Same as FMAC: s_delay_alu under VOPD_FMAC (decode ship; FMAC_DELAY folded).
      if getenv("AMD_VOPD_FMAC", 0):
        out.append(r3.s_delay_alu(1))
      return out
    case AMDOps.CAST:
      cast_src = _src(u.src[0])
      if greg(u).index < 256:
        if u.dtype not in dtypes.ints or u.src[0].dtype not in dtypes.ints:
          raise CompileError(f"no scalar cast {u.src[0].dtype} -> {u.dtype}")
        narrow = u.src[0].dtype if u.src[0].dtype.itemsize <= u.dtype.itemsize else u.dtype
        if narrow in dtypes.uints: return [r3.s_and_b32(_dst(u), cast_src, (1 << (narrow.itemsize * 8)) - 1)]
        return [r3.s_bfe_i32(_dst(u), cast_src, narrow.itemsize * 8 << 16)]
      pre = [] if isinstance(cast_src, Reg) else [r3.v_mov_b32_e32(TMP_VDATA, cast_src)]
      if pre: cast_src = TMP_VDATA
      if u.dtype in dtypes.ints and u.src[0].dtype in dtypes.ints:
        if u.dtype.itemsize > 4 or u.src[0].dtype.itemsize > 4: raise CompileError(f"no cast {u.src[0].dtype} -> {u.dtype}")
        narrow = u.src[0].dtype if u.src[0].dtype.itemsize <= u.dtype.itemsize else u.dtype
        if narrow in dtypes.uints:
          bits = narrow.itemsize * 8
          # (x & 0xffff).cast(u16) / (x >> 16).cast(u16): mask already applied — skip AND (HIP).
          if _u32_high_bits_clear(u.src[0], bits):
            if isinstance(greg(u), Register) and isinstance(greg(u.src[0]), Register) and \
               greg(u).index == greg(u.src[0]).index: return pre
            return pre + [r3.v_mov_b32_e32(_dst(u), cast_src)]
          return pre + [r3.v_and_b32_e32(_dst(u), (1 << bits) - 1, cast_src)]
        return pre + [r3.v_bfe_i32(_dst(u), cast_src, 0, narrow.itemsize * 8)]
      if u.dtype is dtypes.float32 and u.src[0].dtype is dtypes.float16:
        # E32 true16 operands use bit 7 as the half selector, not the VGPR index MSB.
        cast = r3.v_cvt_f32_f16_e64 if _high_vgpr(cast_src) else r3.v_cvt_f32_f16_e32
        return pre + [cast(_dst(u), cast_src)]
      if u.src[0].dtype is dtypes.float32 and u.dtype is dtypes.float16:
        cast = r3.v_cvt_f16_f32_e64 if _high_vgpr(_dst(u)) else r3.v_cvt_f16_f32_e32
        return pre + [cast(_dst(u), cast_src)]
      if u.dtype is dtypes.float32 and u.src[0].dtype in dtypes.ints:
        if u.src[0].dtype in dtypes.sints and u.src[0].dtype.itemsize < 4:
          if u.src[0].op is Ops.INS and _iop(u.src[0]) is AMDOps.LOAD:
            return pre + [r3.v_cvt_f32_i32_e32(_dst(u), cast_src)]
          # Narrow signed values normally arrive sign-extended from i8/i16 loads, but a
          # BITCAST from u8/u16 is a register no-op and leaves the high bits clear.
          # Sign-extend it in one instruction before using the i32 conversion instruction.
          return pre + [r3.v_bfe_i32(_dst(u), cast_src, 0, u.src[0].dtype.itemsize * 8),
                        r3.v_cvt_f32_i32_e32(_dst(u), _dst(u))]
        cast_op = r3.v_cvt_f32_i32_e32 if u.src[0].dtype in dtypes.sints else r3.v_cvt_f32_u32_e32
        return pre + [cast_op(_dst(u), cast_src)]
      if u.src[0].dtype is dtypes.float32 and u.dtype in dtypes.ints:
        cast_op = r3.v_cvt_i32_f32_e32 if u.dtype in dtypes.sints else r3.v_cvt_u32_f32_e32
        return pre + [cast_op(_dst(u), cast_src)]
      raise CompileError(f"no cast {u.src[0].dtype} -> {u.dtype}")
    case AMDOps.RECIPROCAL:
      if u.dtype is not dtypes.float32: raise CompileError(f"f32 rcp only, got {u.dtype}")
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      dst = _dst(u)
      return pre + [r3.v_rcp_f32_e32(TMP_VADDR, val), r3.v_mul_f32_e32(dst, val, TMP_VADDR),
                    r3.v_sub_f32_e32(dst, 1.0, dst), r3.v_fma_f32(dst, TMP_VADDR, dst, TMP_VADDR),
                    r3.v_cmp_eq_f32_e32(dst, dst), r3.v_cndmask_b32_e32(dst, TMP_VADDR, dst)]
    case AMDOps.EXP2 | AMDOps.LOG2 | AMDOps.SQRT | AMDOps.TRUNC:
      if u.dtype is not dtypes.float32: raise CompileError(f"f32 {_iop(u).name} only, got {u.dtype}")
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [_F32_UNARY[_iop(u)](_dst(u), val)]
    case AMDOps.SIN:
      if u.dtype is not dtypes.float32: raise CompileError(f"f32 sin only, got {u.dtype}")
      val = _src(u.src[0])
      pre = [] if isinstance(val, Reg) and val == TMP_VADDR else [r3.v_mov_b32_e32(TMP_VADDR, val)]
      return pre + [r3.v_mul_f32_e32(TMP_VDATA, 0.15915494309189535, TMP_VADDR),
                    r3.v_add_f32_e32(TMP_VDATA, 0.5, TMP_VDATA),
                    r3.v_fract_f32_e32(_dst(u), TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VDATA, TMP_VDATA, _dst(u)),
                    r3.v_mul_f32_e32(_dst(u), 6.28125, TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VADDR, TMP_VADDR, _dst(u)),
                    r3.v_mul_f32_e32(_dst(u), 0.0019353071795864769, TMP_VDATA),
                    r3.v_sub_f32_e32(TMP_VADDR, TMP_VADDR, _dst(u)),
                    r3.v_mul_f32_e32(TMP_VDATA, 0.15915494309189535, TMP_VADDR),
                    r3.v_sin_f32_e32(_dst(u), TMP_VDATA)]
    case AMDOps.MAX:
      return _max(u)
    case AMDOps.SHL:
      if greg(u).index < 256: return [r3.s_lshl_b32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_lshlrev_b32_e32(_dst(u), _src(u.src[1]), a)]
    case AMDOps.SHR:
      if greg(u).index < 256:
        scalar_shift = r3.s_ashr_i32 if u.dtype in dtypes.sints else r3.s_lshr_b32
        return [scalar_shift(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      pre, a = _vgpr_data(TMP_VDATA, u.src[0])
      if u.dtype in dtypes.sints: return pre + [r3.v_ashrrev_i32_e32(_dst(u), _src(u.src[1]), a)]
      return pre + [r3.v_lshrrev_b32_e32(_dst(u), _src(u.src[1]), a)]
    case AMDOps.AND:
      if greg(u).index < 256: return [r3.s_and_b32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      return _commutative_vop2(u, r3.v_and_b32_e32)
    case AMDOps.OR:
      if greg(u).index < 256: return [r3.s_or_b32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      return _commutative_vop2(u, r3.v_or_b32_e32)
    case AMDOps.XOR:
      if greg(u).index < 256: return [r3.s_xor_b32(_dst(u), _src(u.src[0]), _src(u.src[1]))]
      return _commutative_vop2(u, r3.v_xor_b32_e32)
    case AMDOps.CMPLT:
      return _cmp_lt(u)
    case AMDOps.CMPNE:
      return _cmp_ne(u)
    case AMDOps.CMPEQ:
      return _cmp_eq(u)
    case AMDOps.WHERE:
      pre, true_val = _vgpr_data(TMP_VDATA, u.src[1])
      return pre + [r3.v_cndmask_b32_e32(_dst(u), _src(u.src[2]), true_val)]
    case AMDOps.SWHERE:
      return _scalar_condition(u.src[0], u.src[3:5]) + \
        [r3.s_delay_alu(9), r3.s_cselect_b32(_dst(u), _src(u.src[1]), _src(u.src[2]))]
    case AMDOps.LOAD:
      if _is_scratch_ref(u.src[0]):
        return insts_for_uop(u.replace(arg=(AMDOps.SLOAD, u.dtype)), skip, masked, store_addr_cache,
                             d16_hi_lo, byte_scaled, fma_hi_lo, fma_pair_dst)
      # AMD_D16_HI emit path removed.
      itemsize, byte_off = (1 if _is_byte_addr_load(u) else _mem_itemsize(u.dtype)), _mem_byte_off(u)
      compact_idx = _src(u.src[1])
      # Compact B: in-place <<1 on page-idx UOp once (track by UOp id, not phys VGPR —
      # regalloc may reuse the VGPR for a later unscaled page idx).
      if byte_scaled is not None and _is_b_compact_load(u) and _reg_slots(u) == 1 and not masked and itemsize == 2:
        idx_uop, idx = u.src[1], compact_idx
        if isinstance(idx, Reg) and idx.offset >= 256:
          key = id(idx_uop)
          scale = [] if key in byte_scaled else [r3.v_lshlrev_b32_e64(idx, 1, idx)]
          byte_scaled.add(key)
          return scale + _global_load_insts(u, idx, byte_off)
      # Peeled B gather: (base<<1)+byte_off into dest — one VALU, still dest-as-addr for s_clause.
      if byte_off > 0 and _reg_slots(u) == 1 and not masked and itemsize == 2:
        dst = _dst(u)
        return [r3.v_lshl_add_u32(dst, _src(u.src[1]), 1, byte_off)] + _global_load_insts(u, dst)
      pre, addr = _scaled_addr(_dst(u) if _reg_slots(u) == 1 else TMP_VADDR, u.src[1], itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      return pre + _global_load_insts(u, addr, byte_off)
    case AMDOps.STORE:
      # Defensive: REG/ALLOC scratch that missed SSTORE isel (e.g. Ops.ALLOC before ALLOC-aware _buf_ref).
      if _is_scratch_ref(u.src[0]):
        return insts_for_uop(u.replace(arg=(AMDOps.SSTORE, dtypes.void)), skip, masked, store_addr_cache,
                             d16_hi_lo, byte_scaled, fma_hi_lo, fma_pair_dst)
      itemsize, byte_off = _mem_itemsize(u.src[2].dtype), _mem_byte_off(u)
      if store_addr_cache is not None and not masked:
        pre, addr, byte_off = store_addr_cache.addr(u.src[1], itemsize, byte_off)
      elif byte_off > 0xfff:
        pre, addr, byte_off = _apply_byte_off(TMP_VADDR, byte_off, u.src[1], itemsize)
      else:
        pre, addr = _scaled_addr(TMP_VADDR, u.src[1], itemsize)
        pre2, addr, byte_off = _apply_byte_off(addr, byte_off)
        pre = pre + pre2
      pre, addr = _masked_addr(pre, addr, masked)
      return pre + _global_store_insts(u, addr, byte_off)
    case AMDOps.ATOMIC_ADD:
      if u.src[2].dtype is not dtypes.float32: raise CompileError(f"f32 atomic only, got {u.src[2].dtype}")
      pre, addr = _scaled_addr(TMP_VADDR, u.src[1], u.src[2].dtype.itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
      return pre + dpre + [r3.global_atomic_add_f32(addr=addr, data=data, saddr=_src(u.src[0]), vdst=TMP_VDATA),
                           r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0)]
    case AMDOps.LLOAD:
      slots = _reg_slots(u)
      # Dest-as-addr: scale into the load VGPR so LDS gathers aren't serialized on TMP_VADDR.
      addr_dst = _dst(u) if isinstance(greg(u), Register) and slots == 1 else None
      itemsize, elem_off = _mem_itemsize(u.dtype), _lds_byte_off(u)
      cse = None
      if addr_dst is None and store_addr_cache is not None and not masked:
        total = _lds_base_offset(u.src[0]) + elem_off
        cse = store_addr_cache.lds_addr(u.src[1], itemsize, total, id(u.src[0]))
      if cse is not None: pre, addr, off = cse
      else: pre, addr, off = _lds_emit_addr(u.src[0], u.src[1], itemsize, elem_off, addr_dst)
      pre, addr = _masked_addr(pre, addr, masked)
      if slots == 8 and u.dtype is dtypes.half:
        return pre + [r3.ds_load_b128(vdst=_reg_chunk(greg(u), 0, 4), addr=addr, **_ds_off(off)),
                      r3.ds_load_b128(vdst=_reg_chunk(greg(u), 4, 4), addr=addr, **_ds_off(off + 16))]
      if (local_load:=_local_load(u.dtype, _elem_count(u))) is None: raise CompileError(f"no lds load {u.dtype}")
      return pre + [local_load(vdst=_dst(u), addr=addr, **_ds_off(off))]
    case AMDOps.LSTORE:
      nstore = _elem_count(u.src[2])
      # half×16 → two B128 (mirror LLOAD); _local_store has no single 32B op.
      if u.src[2].dtype is dtypes.half and nstore == 16:
        itemsize, elem_off = 2, _lds_byte_off(u)
        cse = None
        if store_addr_cache is not None and not masked:
          cse = store_addr_cache.lds_addr(u.src[1], itemsize, _lds_base_offset(u.src[0]) + elem_off, id(u.src[0]))
        if cse is not None: pre, addr, off = cse
        else: pre, addr, off = _lds_emit_addr(u.src[0], u.src[1], itemsize, elem_off)
        pre, addr = _masked_addr(pre, addr, masked)
        r = greg(u.src[2])
        return pre + [r3.ds_store_b128(addr=addr, data0=_reg_chunk(r, 0, 4), **_ds_off(off)),
                      r3.ds_store_b128(addr=addr, data0=_reg_chunk(r, 4, 4), **_ds_off(off + 16))]
      if (local_store:=_local_store(u.src[2].dtype, nstore)) is None: raise CompileError(f"no lds store {u.src[2].dtype}")
      itemsize, elem_off = _mem_itemsize(u.src[2].dtype), _lds_byte_off(u)
      cse = None
      if store_addr_cache is not None and not masked:
        cse = store_addr_cache.lds_addr(u.src[1], itemsize, _lds_base_offset(u.src[0]) + elem_off, id(u.src[0]))
      if cse is not None: pre, addr, off = cse
      else: pre, addr, off = _lds_emit_addr(u.src[0], u.src[1], itemsize, elem_off)
      pre, addr = _masked_addr(pre, addr, masked)
      if nstore == 1:
        # Scalar LDS store: PACK_F16 in the general pool can surface as a multi-slot
        # phys; ds_store_* wants a single VGPR lane.
        dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
        if isinstance(data, Reg) and data.sz > 1: data = data[0]
      else:
        dpre, data = [], _dst(u.src[2])
      # No per-store lgkmcnt — scoreboard flushes once before BARRIER / LLOAD use (hand-style).
      return pre + dpre + [local_store(addr=addr, data0=data, **_ds_off(off))]
    case AMDOps.SLOAD:
      slots = _reg_slots(u)
      itemsize, byte_off = u.dtype.itemsize, _lds_byte_off(u)
      soff = _scratch_base_offset(u.src[0])
      # Dest-as-addr scalar: each load scales into its own VGPR so SLOAD streaks can
      # s_clause (TMP_VADDR CSE blocks multi-index clauses). Default on with
      # AMD_SCRATCH_SCLAUSE — HW flash ~2µs faster; AMD_SCRATCH_DEST_ADDR=0 opts out.
      if not masked and slots == 1 and isinstance(greg(u), Register) and \
         itemsize in (1, 2, 4) and byte_off <= 0xfff:
        dst = _dst(u)
        pre, addr = _scaled_addr(dst, u.src[1], itemsize)
        if addr != dst:
          pre, addr = pre + [r3.v_mov_b32_e32(dst, addr)], dst
        if soff:
          pre, addr = pre + [r3.v_add_nc_u32_e64(dst, soff, addr)], dst
        if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"no scratch load {u.dtype}")
        return pre + [scratch_load(addr=addr, vdst=dst, offset=byte_off, sve=1)]
      if store_addr_cache is not None and not masked:
        # Scalar const-index soft copies: key by SLOAD id — phases reuse (base, idx=0)
        # after intervening TMP_VADDR clobbers; value-only CSE then loads the wrong slot.
        # Vector float4 ACC chunks: peel const element index into imm offset and share
        # base key so one TMP_VADDR serves offs 0/16/32… (no lshl between loads).
        idx_u, bo = u.src[1], byte_off
        if slots > 1 and (ci := _const_int(idx_u)) is not None:
          bo = bo + ci * itemsize
          idx_u = _tconst(0, dtypes.int32).rtag()
          bkey: object = id(u.src[0])
        else:
          bkey = (id(u.src[0]), id(u))
        pre, addr, byte_off = store_addr_cache.addr(idx_u, itemsize, bo, base_key=bkey)
        if pre and soff: pre = pre + [r3.v_add_nc_u32_e64(addr, soff, addr)]
      else:
        pre, addr = _scratch_addr(u.src[0], u.src[1], itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      if slots > 1:
        if u.dtype is not dtypes.float32: raise CompileError(f"no vec scratch load {u.dtype}")
        if slots == 4 and byte_off % 16 == 0 and byte_off + 12 <= 0xfff:
          return pre + [r3.scratch_load_b128(addr=addr, vdst=_reg_chunk(greg(u), 0, 4), offset=byte_off, sve=1)]
        if slots == 2 and byte_off % 8 == 0 and byte_off + 4 <= 0xfff:
          return pre + [r3.scratch_load_b64(addr=addr, vdst=_reg_chunk(greg(u), 0, 2), offset=byte_off, sve=1)]
        return pre + [r3.scratch_load_b32(addr=addr, vdst=_reg_lane(greg(u), i), offset=byte_off + i*4, sve=1) for i in range(slots)]
      if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"no scratch load {u.dtype}")
      return pre + [scratch_load(addr=addr, vdst=_dst(u), offset=byte_off, sve=1)]
    case AMDOps.SSTORE:
      slots = _reg_slots(u.src[2])
      itemsize, byte_off = u.src[2].dtype.itemsize, _lds_byte_off(u)
      soff = _scratch_base_offset(u.src[0])
      if store_addr_cache is not None and not masked:
        idx_u, bo = u.src[1], byte_off
        if slots > 1 and (ci := _const_int(idx_u)) is not None:
          bo = bo + ci * itemsize
          idx_u = _tconst(0, dtypes.int32).rtag()
        pre, addr, byte_off = store_addr_cache.addr(idx_u, itemsize, bo, base_key=id(u.src[0]))
        # Bake scratch segment base into the CSE'd TMP_VADDR on first use of this buffer+index.
        if pre and soff: pre = pre + [r3.v_add_nc_u32_e64(addr, soff, addr)]
      else:
        pre, addr = _scratch_addr(u.src[0], u.src[1], itemsize)
      pre, addr = _masked_addr(pre, addr, masked)
      if slots > 1:
        if u.src[2].dtype is not dtypes.float32: raise CompileError(f"no vec scratch store {u.src[2].dtype}")
        if slots == 4 and byte_off % 16 == 0 and byte_off + 12 <= 0xfff and isinstance(greg(u.src[2]), Register):
          return pre + [r3.scratch_store_b128(addr=addr, data=_reg_chunk(greg(u.src[2]), 0, 4), offset=byte_off, sve=1)]
        return pre + [r3.scratch_store_b32(addr=addr, data=_reg_lane(greg(u.src[2]), i), offset=byte_off + i*4, sve=1)
                      for i in range(slots)]
      if (scratch_store:=_scratch_store(u.src[2].dtype)) is None:
        raise CompileError(f"no scratch store {u.src[2].dtype}")
      dpre, data = _vgpr_data(TMP_VDATA, u.src[2])
      return pre + dpre + [scratch_store(addr=addr, data=data, offset=byte_off, sve=1)]
    case AMDOps.BARRIER:
      return [r3.s_barrier()]
    case AMDOps.FILL:
      slots = _reg_slots(u)
      if (disp_uop:=_unwrap_const(u.src[0])) is None: raise CompileError("non-constant scratch fill offset")
      disp = int(disp_uop.val)
      if disp < 0 or disp + (slots-1)*4 > 0xffffffff: raise CompileError(f"scratch fill oob: offset={disp}, slots={slots}")
      if greg(u).index < 256:
        # Scratch is a vector-memory path. Restore one lane into the reserved VGPR,
        # wait for it, then collect that identical per-lane value back into the SGPR.
        sgpr_loads:list[Inst] = []
        sgpr_fill_page:int|None = None
        for i in range(slots):
          scratch_addr = disp + i*4
          if (new_page:=scratch_addr & ~0xfff) != sgpr_fill_page:
            sgpr_fill_page = new_page
            sgpr_loads.append(r3.v_mov_b32_e32(TMP_VADDR, new_page))
          sgpr_loads += [r3.scratch_load_b32(addr=TMP_VADDR, vdst=TMP_VDATA, offset=scratch_addr & 0xfff, sve=1),
                         r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0),
                         r3.v_readfirstlane_b32_e32(_reg_lane(greg(u), i), TMP_VDATA)]
        return sgpr_loads
      if slots > 1:
        # raw VGPR lanes (f32 packs or half2 LDS loads). Page in TMP_VDATA so TMP_VADDR
        # store-addr CSE survives spill-on-evict between global/LDS ops.
        loads: list[Inst] = []
        fill_page: int|None = None
        for i in range(slots):
          scratch_addr = disp + i*4
          # SCRATCH's 13-bit immediate is signed, so page before its sign bit (0x1000).
          if (new_page:=scratch_addr & ~0xfff) != fill_page:
            fill_page = new_page
            loads.append(r3.v_mov_b32_e32(TMP_VDATA, new_page))
          loads.append(r3.scratch_load_b32(addr=TMP_VDATA, vdst=_reg_lane(greg(u), i), offset=scratch_addr & 0xfff, sve=1))
        return loads
      if (scratch_load:=_scratch_load(u.dtype)) is None: raise CompileError(f"no scratch fill {u.dtype}")
      return [r3.v_mov_b32_e32(TMP_VDATA, disp & ~0xfff),
              scratch_load(addr=TMP_VDATA, vdst=_dst(u), offset=disp & 0xfff, sve=1)]
    case AMDOps.SPILL:
      # Prefer explicit slot count (spill-on-evict) over _reg_slots(bind) which is often 1.
      if len(u.src) > 2 and (n:=_unwrap_const(u.src[2])) is not None: slots = int(n)
      else: slots = _reg_slots(u.src[1])
      if (disp_uop:=_unwrap_const(u.src[0])) is None: raise CompileError("non-constant scratch spill offset")
      disp = int(disp_uop.val)
      if disp < 0 or disp + (slots-1)*4 > 0xffffffff: raise CompileError(f"scratch spill oob: offset={disp}, slots={slots}")
      if greg(u.src[1]).index < 256:
        sgpr_stores:list[Inst] = []
        sgpr_spill_page:int|None = None
        for i in range(slots):
          scratch_addr = disp + i*4
          if (new_page:=scratch_addr & ~0xfff) != sgpr_spill_page:
            sgpr_spill_page = new_page
            sgpr_stores.append(r3.v_mov_b32_e32(TMP_VADDR, new_page))
          sgpr_stores += [r3.v_mov_b32_e32(TMP_VDATA, _reg_lane(greg(u.src[1]), i)),
                          r3.scratch_store_b32(addr=TMP_VADDR, data=TMP_VDATA, offset=scratch_addr & 0xfff, sve=1)]
        return sgpr_stores + [r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
      if slots > 1:
        # raw VGPR lanes — page in TMP_VDATA to preserve TMP_VADDR for mem CSE.
        stores: list[Inst] = []
        spill_page: int|None = None
        for i in range(slots):
          scratch_addr = disp + i*4
          # SCRATCH's 13-bit immediate is signed, so page before its sign bit (0x1000).
          if (new_page:=scratch_addr & ~0xfff) != spill_page:
            spill_page = new_page
            stores.append(r3.v_mov_b32_e32(TMP_VDATA, new_page))
          stores.append(r3.scratch_store_b32(addr=TMP_VDATA, data=_reg_lane(greg(u.src[1]), i), offset=scratch_addr & 0xfff, sve=1))
        return stores + [r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
      if (scratch_store:=_scratch_store(u.src[1].dtype)) is None:
        raise CompileError(f"no scratch spill {u.src[1].dtype}")
      # VGPR spill: scratch page in TMP_VDATA (not TMP_VADDR) so spill-on-evict before
      # LLOAD/STORE does not invalidate a live store-addr CSE in TMP_VADDR.
      return [r3.v_mov_b32_e32(TMP_VDATA, disp & ~0xfff),
              scratch_store(addr=TMP_VDATA, data=_src(u.src[1]), offset=disp & 0xfff, sve=1),
              r3.s_waitcnt_vscnt(sdst=NULL, simm16=0)]
    case AMDOps.CMP_GE:
      pre0, a = _sgpr_data(TMP_SDATA0, u.src[0])
      pre1, b = _sgpr_data(TMP_SDATA1, u.src[1])
      return pre0 + pre1 + [r3.s_cmp_ge_u32(a, b)]
    case AMDOps.IF_MASK:
      if u.src[0].op is Ops.INS and _iop(u.src[0]) in (AMDOps.CMPLT, AMDOps.CMPNE, AMDOps.CMPEQ):
        return [r3.s_and_saveexec_b64(_dst(u), VCC)]
      pre, gate = _vgpr_data(TMP_VDATA, u.src[0])
      return pre + [r3.v_cmp_ne_u32_e32(0, gate), r3.s_and_saveexec_b64(_dst(u), VCC)]
    case AMDOps.END_MASK:
      return [r3.s_mov_b64(EXEC, _src(u.src[0]))]
  raise CompileError(f"cannot encode {_iop(u)}")


def _hoist_kernargs(ops:list[UOp]) -> list[UOp]:
  return _sched.hoist_kernargs(ops)

def _hoist_lloads_before_extracts(ops:list[UOp]) -> list[UOp]:
  return _sched.hoist_lloads_before_extracts(ops)

def _schedule_lload_2addr_pairs(ops:list[UOp]) -> list[UOp]:
  return _sched.schedule_lload_2addr_pairs(ops)


# --- schedule: WMMA/load motion (pre-emit, post-regalloc order) ---
# Physical: sink_wmma / hoist_b live in rdna3_schedule.py
def _sink_wmma_past_loads(ops:list[UOp]) -> list[UOp]:
  return _sched.sink_wmma_past_loads(ops)

def _is_addr_alu(u:UOp) -> bool:
  return _sched.is_addr_alu(u)

def _hoist_b_between_a_and_pack(ops:list[UOp]) -> list[UOp]:
  return _sched.hoist_b_between_a_and_pack(ops)


def _prefetch_a_after_packed_quant(ops:list[UOp]) -> list[UOp]:
  return _sched.prefetch_a_after_packed_quant(ops)


def _prefetch_a_before_dequant_mix(ops:list[UOp]) -> list[UOp]:
  return _sched.prefetch_a_before_dequant_mix(ops)


def _prefetch_late_iq4_a_before_mix(ops:list[UOp]) -> list[UOp]:
  return _sched.prefetch_late_iq4_a_before_mix(ops)


def _prefetch_next_bu16_before_pack(ops:list[UOp]) -> list[UOp]:
  return _sched.prefetch_next_bu16_before_pack(ops)


def _hoist_loads_before_wmma(ops:list[UOp]) -> list[UOp]:
  return _sched.hoist_loads_before_wmma(ops)


_VMEM_SCHEDULABLE = _sched.VMEM_SCHEDULABLE

def _hoist_gated_fmac_loads(ops:list[UOp]) -> list[UOp]:
  return _sched.hoist_gated_fmac_loads(ops)

def _vmem_schedulable_load(u:UOp) -> bool:
  return _sched.vmem_schedulable_load(u)

def _schedule_scalar_vmem(ops:list[UOp], d16_hi_lo:dict[UOp, UOp], alu_breadth:bool|None=None) -> list[UOp]:
  return _sched.schedule_scalar_vmem(ops, d16_hi_lo, alu_breadth)

def _schedule_fmac_add_pairs(ops:list[UOp]) -> list[UOp]:
  return _sched.schedule_fmac_add_pairs(ops)

def _schedule_swizzle_mov_batches(ops:list[UOp]) -> list[UOp]:
  return _sched.schedule_swizzle_mov_batches(ops)

def _gap_fill_after_loads(ops:list[UOp]) -> list[UOp]:
  return _sched.gap_fill_after_loads(ops)

def _cluster_const_scratch_stores(ops:list[UOp]) -> list[UOp]:
  return _sched.cluster_const_scratch_stores(ops)

def _cluster_const_scratch_loads(ops:list[UOp]) -> list[UOp]:
  return _sched.cluster_const_scratch_loads(ops)

def _scratch_batch_phys_conflict(loads:list[UOp], uses:list[UOp], new_load:UOp, new_use:UOp) -> bool:
  return _sched._scratch_batch_phys_conflict(loads, uses, new_load, new_use)

def _schedule_scratch_load_passes(ops:list[UOp]) -> list[UOp]:
  return _sched.schedule_scratch_load_passes(ops)

def _batch_scratch_load_uses(ops:list[UOp], *, check_phys:bool=False) -> list[UOp]:
  return _sched.batch_scratch_load_uses(ops, check_phys=check_phys)

def _pack_aligned_const_scratch_sloads(ops:list[UOp]) -> list[UOp]:
  return _sched.pack_aligned_const_scratch_sloads(ops)

# --- emit: waitcnt / clause / fused mem (post-regalloc) ---
from tinygrad.renderer.isa import rdna3_emit as _emit

def _vm_load_count(insts:list) -> int: return _emit.vm_load_count(insts)
def _lgkm_store_count(insts:list) -> int: return _emit.lgkm_store_count(insts)
def _split_lgkm_scale_and_stores(emitted:list) -> tuple[list, list]: return _emit.split_lgkm_scale_and_stores(emitted)
def _lgkm_load_count(insts:list) -> int: return _emit.lgkm_load_count(insts)
def _split_lgkm_scale_and_loads(emitted:list) -> tuple[list, list]: return _emit.split_lgkm_scale_and_loads(emitted)
def _split_scale_and_loads(emitted:list) -> tuple[list, list]: return _emit.split_scale_and_loads(emitted)
def _vm_store_count(insts:list) -> int: return _emit.vm_store_count(insts)
def _split_scale_and_stores(emitted:list) -> tuple[list, list]: return _emit.split_scale_and_stores(emitted)
def _tmp_vaddr_clause_safe(scales:list, loads:list) -> bool: return _emit.tmp_vaddr_clause_safe(scales, loads)

def _clauseable_scalar_vmem_gload(u:UOp, skip:set[UOp], mask_depth:int) -> bool:
  # Scalar global LOAD with dest-as-addr (compact B). Streak → hoist scales + s_clause.
  if u in skip or mask_depth or u.op is not Ops.INS or _iop(u) is not AMDOps.LOAD: return False
  if _is_lds_ref(u.src[0]) or _is_scratch_ref(u.src[0]): return False
  slots = _reg_slots(u)
  if slots != 1:
    return u.dtype in (dtypes.float32, dtypes.float) and slots <= 4
  return u.dtype in (dtypes.half, dtypes.uint8, dtypes.int8, dtypes.uint, dtypes.int, dtypes.uint32, dtypes.int32, dtypes.float32, dtypes.float)

def _quant_b128_clause_info(u:UOp, skip:set[UOp], mask_depth:int) -> tuple[UOp, UOp, int]|None:
  """Packed quant weight LOAD (uint32×4 / B128): (saddr, base_idx, byte_off) or None."""
  if u in skip or mask_depth or u.op is not Ops.INS or _iop(u) is not AMDOps.LOAD: return None
  if _is_lds_ref(u.src[0]) or _is_scratch_ref(u.src[0]): return None
  if u.dtype not in (dtypes.uint32, dtypes.uint) or _reg_slots(u) != 4: return None
  if not isinstance(greg(u), Register): return None
  base, byte_off = _peel_add_imm(u.src[1], 4, max_byte=0xfff, deep=True)
  return u.src[0], base, byte_off

def _clauseable_half_gload(u:UOp, skip:set[UOp], mask_depth:int) -> bool:
  return _clauseable_scalar_vmem_gload(u, skip, mask_depth) and u.dtype is dtypes.half

def _fuse_kernarg_smem_loads(scheduled:list[UOp], oi:int, skip:set[UOp], mask_depth:int) -> tuple[int, list, list[UOp]]|None:
  """Fuse contiguous ulong KERNARGs into s_load_b128/b256 when SGPR dests are contiguous."""
  if mask_depth or oi >= len(scheduled): return None
  u = scheduled[oi]
  if u in skip or u.op is not Ops.INS or _iop(u) is not AMDOps.KERNARG or u.dtype.itemsize != 8: return None
  if not isinstance(greg(u), Register) or (base_off:=_const_int(u.src[0])) is None: return None
  group = [u]
  j = oi + 1
  while j < len(scheduled) and len(group) < 4:
    v = scheduled[j]
    if v in skip or v.op is not Ops.INS or _iop(v) is not AMDOps.KERNARG or v.dtype.itemsize != 8: break
    prev, prev_off = group[-1], _const_int(group[-1].src[0])
    off = _const_int(v.src[0])
    if prev_off is None or off is None or off != prev_off + 8: break
    if not isinstance(greg(prev), Register) or not isinstance(greg(v), Register): break
    if greg(v).index != greg(prev).index + 2: break
    group.append(v)
    j += 1
  if len(group) < 2: return None
  base = greg(group[0]).index
  # SMEM wide loads require destination SGPR alignment to the transfer width.
  if len(group) >= 4 and base % 8 == 0: n = 4
  elif base % 4 == 0: n = 2
  else: return None
  group = group[:n]
  sdata = _reg_to_amd(greg(group[0]), n * 2)
  load = r3.s_load_b256(sdata=sdata, sbase=KERNARG_REG, soffset=NULL, offset=base_off) if n == 4 else \
         r3.s_load_b128(sdata=sdata, sbase=KERNARG_REG, soffset=NULL, offset=base_off)
  return n, [load], group

def _clauseable_wide_half_gload(u:UOp, skip:set[UOp], mask_depth:int) -> bool:
  # Contiguous half×8+ global LOAD (A B128 pairs). Streak → one s_clause over all B128s.
  if u in skip or mask_depth or u.op is not Ops.INS or _iop(u) is not AMDOps.LOAD: return False
  return u.dtype is dtypes.half and _elem_count(u) >= 8 and not _is_lds_ref(u.src[0]) and \
         not _is_scratch_ref(u.src[0])


def _fma_mixhi_lo_map(uops:list[UOp]) -> dict[UOp, UOp]:
  return _sched.fma_mixhi_lo_map(uops)


def _half2_global_hi_skip(uops:list[UOp]) -> set[UOp]:
  return _sched.half2_global_hi_skip(uops)


def _rewrite_half2_global_packs(ops:list[UOp]) -> list[UOp]:
  return _sched.rewrite_half2_global_packs(ops)


def _schedule_half2_lo_before_pack(ops:list[UOp]) -> list[UOp]:
  return _sched.schedule_half2_lo_before_pack(ops)


def _fma_pair_pack_dsts(uops:list[UOp], fma_hi_lo:dict[UOp, UOp]) -> dict[UOp, Reg]:
  """Use the final PACK_F16 lane directly when it is idle throughout a paired FMA's live interval."""
  pos = {u:i for i,u in enumerate(uops)}
  last_use: dict[UOp, int] = {}
  for i,u in enumerate(uops):
    for src in u.src: last_use[src] = i
  ret: dict[UOp, Reg] = {}
  for pack in uops:
    if pack.op is not Ops.INS or _iop(pack) is not AMDOps.PACK_F16 or not isinstance(greg(pack), Register): continue
    for lane,(lo,hi) in enumerate(zip(pack.src[::2], pack.src[1::2])):
      if fma_hi_lo.get(hi) is not lo or lo not in pos or hi not in pos or not (pos[lo] < pos[hi] < pos[pack]): continue
      target_idx = greg(pack).index + lane
      # Inputs of either FMA, intervening defs, and any value live across [lo, pack]
      # (defined earlier, last use ≥ lo). Omitting live-ins let Q4/Q5/Q6 mixhi clobber
      # sibling dequant temps → NaN WMMA fragments (IQ4 packed-mul often missed the hazard).
      blocked = set().union(*(_reg_idxs(s) for s in lo.src+hi.src))
      for x in uops[pos[lo]+1:pos[pack]]:
        if x is not hi: blocked |= _reg_idxs(x)
      for x in uops[:pos[lo]]:
        if last_use.get(x, -1) >= pos[lo]: blocked |= _reg_idxs(x)
      if target_idx in blocked: continue
      dst = _reg_lane(greg(pack), lane)
      ret[lo] = ret[hi] = dst
  return ret

def _schedule_fma_mixhi_pairs(uops:list[UOp]) -> list[UOp]:
  return _sched.schedule_fma_mixhi_pairs(uops)


def _fused_lds_pack_store(uops:list[UOp], i:int) -> tuple[int, list, set[int]]|None:
  """Fold four aliasing EXTRACT+LSTORE pairs into one DS_STORE_B128 (f32 PACK or u32×4 LOAD)."""
  if i + 7 >= len(uops): return None
  extracts, stores = uops[i:i+8:2], uops[i+1:i+8:2]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.EXTRACT for x in extracts): return None
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.LSTORE and x.src[2] is extracts[n] for n,x in enumerate(stores)): return None
  pack, lane0 = extracts[0].src[0], _const_int(extracts[0].src[1])
  off0 = _lds_byte_off(stores[0])
  dt = stores[0].src[2].dtype
  if dt not in (dtypes.float32, dtypes.uint32, dtypes.int32): return None
  if lane0 is None or lane0 % 4 or off0 % 16 or not isinstance(greg(pack), Register) or _reg_slots(pack) < lane0 + 4:
    return None
  # f32 epilogue PACK, or quant LUT fill from packed global LOAD (u32×4).
  if _iop(pack) is AMDOps.PACK:
    if pack.dtype is not dtypes.float32: return None
  elif _iop(pack) is AMDOps.LOAD:
    if pack.dtype not in (dtypes.uint32, dtypes.int32, dtypes.float32) or _elem_count(pack) < lane0 + 4: return None
  else: return None
  if not all(x.src[0] is pack and _const_int(x.src[1]) == lane0+n and isinstance(greg(x), Register) and
             greg(x).index == greg(pack).index+lane0+n for n,x in enumerate(extracts)): return None
  base, idx = stores[0].src[:2]
  if not all(x.src[0] is base and x.src[1] is idx and _lds_byte_off(x) == off0+4*n and
             x.src[2].dtype is dt for n,x in enumerate(stores)): return None
  pre, addr, off = _lds_emit_addr(base, idx, dt.itemsize, off0)
  data = _reg_chunk(greg(pack), lane0, 4)
  deps = _reg_idxs(pack) | _reg_idxs(idx)
  return 8, pre + [r3.ds_store_b128(addr=addr, data0=data, **_ds_off(off))], deps

def _fused_lds_contig_store(uops:list[UOp], i:int) -> tuple[int, list, set[int]]|None:
  """Fold contiguous scalar LSTOREs into a wide DS_STORE.

  f32/u32×4 → B128 (IQ4 LUT / epilogue). Half fuse (LDS_F16_STORE_FUSE) removed — scrambled offs.
  """
  if i >= len(uops): return None
  # AMD_LDS_F16_STORE_FUSE removed (scrambled offs).
  if i + 3 >= len(uops): return None
  stores = uops[i:i+4]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.LSTORE for x in stores): return None
  dt = stores[0].src[2].dtype
  if dt not in (dtypes.float32, dtypes.uint32, dtypes.int32): return None
  if not all(x.src[2].dtype is dt and _elem_count(x.src[2]) == 1 and isinstance(greg(x.src[2]), Register)
             for x in stores): return None
  r0 = greg(stores[0].src[2])
  if not all(greg(x.src[2]).index == r0.index + n for n, x in enumerate(stores)): return None
  base, idx, off0 = stores[0].src[0], stores[0].src[1], _lds_byte_off(stores[0])
  if off0 % 16: return None
  if not all(x.src[0] is base and x.src[1] is idx and _lds_byte_off(x) == off0 + 4 * n for n, x in enumerate(stores)):
    return None
  pre, addr, off = _lds_emit_addr(base, idx, dt.itemsize, off0)
  deps = set().union(*(_reg_idxs(x.src[2]) for x in stores)) | _reg_idxs(idx)
  return 4, pre + [r3.ds_store_b128(addr=addr, data0=_reg_chunk(r0, 0, 4), **_ds_off(off))], deps

def _fused_scratch_contig_store(uops:list[UOp], i:int, store_addr_cache:_StoreAddrCache|None=None) -> tuple[int, list, set[int]]|None:
  """Fold four contiguous scalar SSTORE into SCRATCH_STORE_B128.

  Accepts shared idx + byte_off 0/4/8/12, or consecutive const element indices.
  Same-VGPR broadcast pack removed (clobber).
  """
  if i + 3 >= len(uops): return None
  stores = uops[i:i+4]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.SSTORE for x in stores): return None
  dt = stores[0].src[2].dtype
  if dt not in (dtypes.float32, dtypes.uint32, dtypes.int32): return None
  if not all(x.src[2].dtype is dt and _elem_count(x.src[2]) == 1 and isinstance(greg(x.src[2]), Register)
             for x in stores): return None
  r0 = greg(stores[0].src[2])
  reg_idxs = [greg(x.src[2]).index for x in stores]
  contig = all(reg_idxs[n] == r0.index + n for n in range(4))
  # AMD_SCRATCH_BCAST_B128 removed (same-VGPR broadcast clobber breaks flash).
  if not contig: return None
  base = stores[0].src[0]
  if not all(x.src[0] is base for x in stores): return None
  offs = [_lds_byte_off(x) for x in stores]
  idxs = [_const_int(x.src[1]) for x in stores]
  byte_off = 0
  idx = stores[0].src[1]
  if all(x.src[1] is idx for x in stores) and offs[0] % 16 == 0 and \
     all(offs[n] == offs[0] + 4 * n for n in range(4)) and offs[0] + 12 <= 0xfff:
    byte_off = offs[0]
  elif all(i is not None for i in idxs) and all(o == 0 for o in offs):
    i0 = idxs[0]
    assert i0 is not None
    if i0 % 4 == 0 and all(idxs[n] == i0 + n for n in range(4)) and (i0 + 3) * dt.itemsize <= 0xfff:
      byte_off = i0 * dt.itemsize
      idx = _tconst(0, dtypes.int32).rtag()
    else:
      return None
  else:
    return None
  soff = _scratch_base_offset(base)
  if store_addr_cache is not None:
    pre, addr, byte_off = store_addr_cache.addr(idx, dt.itemsize, byte_off, base_key=id(base))
    if pre and soff: pre = pre + [r3.v_add_nc_u32_e64(addr, soff, addr)]
  else:
    pre, addr = _scaled_addr(TMP_VADDR, idx, dt.itemsize)
    if soff: pre = pre + [r3.v_add_nc_u32_e64(TMP_VADDR, soff, addr)]
    addr = TMP_VADDR
  deps = set().union(*(_reg_idxs(x.src[2]) for x in stores)) | set().union(*(_reg_idxs(x.src[1]) for x in stores))
  if store_addr_cache is not None: store_addr_cache._bcast_src = None  # type: ignore[attr-defined]
  data = _reg_chunk(r0, 0, 4)
  return 4, pre + [r3.scratch_store_b128(addr=addr, data=data, offset=byte_off, sve=1)], deps

def _fused_scratch_contig_load(uops:list[UOp], i:int, store_addr_cache:_StoreAddrCache|None=None) -> tuple[int, list, set[int]]|None:
  """Fold four contiguous scalar SLOAD (offs +0..+12, consecutive dest VGPRs) into SCRATCH_LOAD_B128.

  Accepts either shared idx + byte_off 0/4/8/12, or consecutive const element indices
  (flash soft-copy SLOAD(base, i)..SLOAD(base, i+3)).
  """
  if i + 3 >= len(uops): return None
  loads = uops[i:i+4]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.SLOAD for x in loads): return None
  dt = loads[0].dtype
  if dt not in (dtypes.float32, dtypes.uint32, dtypes.int32): return None
  if not all(x.dtype is dt and _elem_count(x) == 1 and isinstance(greg(x), Register) for x in loads): return None
  r0 = greg(loads[0])
  if not all(greg(x).index == r0.index + n for n, x in enumerate(loads)): return None
  base = loads[0].src[0]
  if not all(x.src[0] is base for x in loads): return None
  offs = [_lds_byte_off(x) for x in loads]
  idxs = [_const_int(x.src[1]) for x in loads]
  byte_off = 0
  idx = loads[0].src[1]
  if all(x.src[1] is idx for x in loads) and offs[0] % 16 == 0 and \
     all(offs[n] == offs[0] + 4 * n for n in range(4)) and offs[0] + 12 <= 0xfff:
    byte_off = offs[0]
  elif all(i is not None for i in idxs) and all(o == 0 for o in offs):
    i0 = idxs[0]
    assert i0 is not None
    if i0 % 4 == 0 and all(idxs[n] == i0 + n for n in range(4)) and (i0 + 3) * dt.itemsize <= 0xfff:
      # Const element indices → encode as byte offset from a zero index.
      byte_off = i0 * dt.itemsize
      idx = _tconst(0, dtypes.int32).rtag()
    else:
      return None
  else:
    return None
  soff = _scratch_base_offset(base)
  if store_addr_cache is not None:
    # Key by scratch base only — per-load id breaks CSE across ACC float4 chunks
    # (lshl between every scratch_load_b128). Const idx+byte_off share one TMP_VADDR page.
    pre, addr, byte_off = store_addr_cache.addr(idx, dt.itemsize, byte_off, base_key=id(base))
    if pre and soff: pre = pre + [r3.v_add_nc_u32_e64(addr, soff, addr)]
  else:
    pre, addr = _scaled_addr(TMP_VADDR, idx, dt.itemsize)
    if soff: pre = pre + [r3.v_add_nc_u32_e64(TMP_VADDR, soff, addr)]
    addr = TMP_VADDR
  deps = set().union(*(_reg_idxs(x.src[1]) for x in loads))
  return 4, pre + [r3.scratch_load_b128(addr=addr, vdst=_reg_chunk(r0, 0, 4), offset=byte_off, sve=1)], deps

def _fused_scratch_contig_load_b64(uops:list[UOp], i:int, store_addr_cache:_StoreAddrCache|None=None) -> tuple[int, list, set[int]]|None:
  """Fold two contiguous scalar SLOAD into SCRATCH_LOAD_B64 (8-byte aligned)."""
  if i + 1 >= len(uops): return None
  loads = uops[i:i+2]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.SLOAD for x in loads): return None
  dt = loads[0].dtype
  if dt not in (dtypes.float32, dtypes.uint32, dtypes.int32): return None
  if not all(x.dtype is dt and _elem_count(x) == 1 and isinstance(greg(x), Register) for x in loads): return None
  r0 = greg(loads[0])
  if greg(loads[1]).index != r0.index + 1: return None
  base = loads[0].src[0]
  if loads[1].src[0] is not base: return None
  offs = [_lds_byte_off(x) for x in loads]
  idxs = [_const_int(x.src[1]) for x in loads]
  byte_off = 0
  idx = loads[0].src[1]
  if loads[1].src[1] is idx and offs[0] % 8 == 0 and offs[1] == offs[0] + 4 and offs[0] + 4 <= 0xfff:
    byte_off = offs[0]
  elif (all(t is not None for t in idxs) and all(o == 0 for o in offs) and
        int(idxs[0]) % 2 == 0 and int(idxs[1]) == int(idxs[0]) + 1 and  # type: ignore[arg-type]
        (int(idxs[0]) + 1) * dt.itemsize <= 0xfff):  # type: ignore[arg-type]
    byte_off = int(idxs[0]) * dt.itemsize  # type: ignore[arg-type]
    idx = _tconst(0, dtypes.int32).rtag()
  else:
    return None
  soff = _scratch_base_offset(base)
  if store_addr_cache is not None:
    pre, addr, byte_off = store_addr_cache.addr(idx, dt.itemsize, byte_off, base_key=id(base))
    if pre and soff: pre = pre + [r3.v_add_nc_u32_e64(addr, soff, addr)]
  else:
    pre, addr = _scaled_addr(TMP_VADDR, idx, dt.itemsize)
    if soff: pre = pre + [r3.v_add_nc_u32_e64(TMP_VADDR, soff, addr)]
    addr = TMP_VADDR
  deps = set().union(*(_reg_idxs(x.src[1]) for x in loads))
  return 2, pre + [r3.scratch_load_b64(addr=addr, vdst=_reg_chunk(r0, 0, 2), offset=byte_off, sve=1)], deps

def _fused_lds_pack_load(uops:list[UOp], i:int) -> tuple[int, list, set[int]]|None:
  """Fold four contiguous f32 LLOAD (same base, offs +0..+12) into one DS_LOAD_B128."""
  if i + 3 >= len(uops): return None
  loads = uops[i:i+4]
  if not all(x.op is Ops.INS and _iop(x) is AMDOps.LLOAD and x.dtype is dtypes.float32 and _elem_count(x) == 1
             for x in loads): return None
  if not all(isinstance(greg(x), Register) for x in loads): return None
  base, idx, off0 = loads[0].src[0], loads[0].src[1], _lds_byte_off(loads[0])
  if off0 % 16: return None
  if not all(x.src[0] is base and x.src[1] is idx and _lds_byte_off(x) == off0 + 4 * n for n, x in enumerate(loads)):
    return None
  if not all(greg(x).index == greg(loads[0]).index + n for n, x in enumerate(loads)): return None
  pre, addr, off = _lds_emit_addr(base, idx, dtypes.float32.itemsize, off0)
  deps = _reg_idxs(idx)
  return 4, pre + [r3.ds_load_b128(vdst=_reg_chunk(greg(loads[0]), 0, 4), addr=addr, **_ds_off(off))], deps

def _fused_lds_2addr_b64(uops:list[UOp], i:int) -> tuple[int, list, set[int]]|None:
  """Fold two 8-byte LLOADs into DS_LOAD_2ADDR_B64 (HIP flash LDS pattern).

  offset0/1 are u8 *qword* indices (not the ordinary ds 16-bit byte split). Requires
  consecutive dest VGPRs (4 lanes) and the same base+idx. Default on (AMD_LDS_2ADDR=0
  opts out). Flash LDS imm offs are often >2KB — fold min(off0,off1) into the address
  VGPR so both qword indices fit in 0..255 (AMD_LDS_2ADDR_FOLD=0 disables).
  """
  if i + 1 >= len(uops): return None
  a, b = uops[i], uops[i + 1]
  if not (a.op is Ops.INS and b.op is Ops.INS and _iop(a) is AMDOps.LLOAD and _iop(b) is AMDOps.LLOAD): return None
  if not (isinstance(greg(a), Register) and isinstance(greg(b), Register)): return None
  if _elem_count(a) * a.dtype.itemsize != 8 or _elem_count(b) * b.dtype.itemsize != 8: return None
  if a.src[0] is not b.src[0] or a.src[1] is not b.src[1]: return None
  ra, rb = greg(a), greg(b)
  if ra.index + 2 != rb.index or _reg_slots(a) != 2 or _reg_slots(b) != 2: return None
  off0, off1 = _lds_byte_off(a), _lds_byte_off(b)
  if off0 % 8 or off1 % 8: return None
  fold = min(off0, off1) if True else 0
  q0, q1 = (off0 - fold) // 8, (off1 - fold) // 8
  if not (0 <= q0 <= 0xff and 0 <= q1 <= 0xff): return None
  # 2addr offsets are qword indices (not byte ds_off) — bake slot base + fold into one
  # v_lshl_add (scale+imm) instead of lshl + add + add.
  itemsize = a.dtype.itemsize
  if itemsize in (2, 4):
    imm = _lds_base_offset(a.src[0]) + fold
    shift, dst, src = itemsize.bit_length() - 1, TMP_VADDR, _src(a.src[1])
    if imm: pre, addr = [r3.v_lshl_add_u32(dst, src, shift, imm)], dst
    else: pre, addr = [r3.v_lshlrev_b32_e64(dst, shift, src)], dst
  else:
    pre, addr, base_off = _local_addr(a.src[0], a.src[1], itemsize)
    if (imm := base_off + fold):
      pre, addr = pre + [r3.v_add_nc_u32_e64(TMP_VADDR, imm, addr)], TMP_VADDR
  deps = _reg_idxs(a.src[1])
  return 2, pre + [r3.ds_load_2addr_b64(vdst=_reg_chunk(ra, 0, 4), addr=addr, offset0=q0, offset1=q1)], deps

def _fused_mixed_dot4_loop(uops:list[UOp], i:int) -> tuple[int, list]|None:
  """Wide-load an exact four-element f16*f32 dot while retaining its sequential FMA order."""
  if i + 14 >= len(uops): return None
  init, ctr, loop, cmp, exit_branch = uops[i:i+5]
  if not (all(u.op is Ops.INS for u in (init, ctr, loop, cmp, exit_branch)) and
          _iop(init) is AMDOps.MOV and init.dtype is dtypes.float32 and init.src and _is_zero_val(init.src[0]) and
          _iop(ctr) is AMDOps.MOV and _const_int(ctr) == 0 and _iop(loop) is AMDOps.LABEL and
          _iop(cmp) is AMDOps.CMP_GE and cmp.src[0] is ctr and _const_int(cmp.src[1]) == 4 and
          _iop(exit_branch) is AMDOps.CBRANCH_SCC1 and exit_branch.src == (cmp,)): return None
  cast_i = next((j for j in range(i+5, min(i+18, len(uops)))
                 if uops[j].op is Ops.INS and _iop(uops[j]) is AMDOps.CAST and uops[j].dtype is dtypes.float32), None)
  if cast_i is None or cast_i + 5 >= len(uops): return None
  cast, fmac, copy, inc, back, out = uops[cast_i:cast_i+6]
  if not (all(u.op is Ops.INS for u in (cast, fmac, copy, inc, back, out)) and
          _iop(fmac) is AMDOps.FMAC and fmac.dtype is dtypes.float32 and fmac.src[0] is init and
          _iop(copy) is AMDOps.MOV and copy.src == (fmac,) and greg(copy) == greg(init) and
          _iop(inc) is AMDOps.ADD and inc.src[0] is ctr and _const_int(inc.src[1]) == 1 and greg(inc) == greg(ctr) and
          _iop(back) is AMDOps.BRANCH and back.tag == loop.tag and _iop(out) is AMDOps.LABEL and exit_branch.tag == out.tag): return None
  body = uops[i+5:cast_i]
  if any(u.op is not Ops.INS or _iop(u) not in (AMDOps.ADD, AMDOps.LOAD) for u in body): return None
  loads = [u for u in body if _iop(u) is AMDOps.LOAD]
  if len(loads) != 2: return None
  hload = next((u for u in loads if u.dtype is dtypes.float16), None)
  fload = next((u for u in loads if u.dtype is dtypes.float32), None)
  if hload is None or fload is None or cast.src != (hload,) or set(fmac.src[1:]) != {cast, fload}: return None
  if any(_elem_count(u) != 1 or _mem_byte_off(u) != 0 for u in loads): return None
  addr_deps = set(hload.src[1].toposort()) | set(fload.src[1].toposort())
  if any(_iop(u) is AMDOps.ADD and (u not in addr_deps or u.dtype not in (dtypes.int32, dtypes.uint32)) for u in body): return None
  def ctr_coeff(x:UOp) -> int|None:
    if x is ctr: return 1
    if ctr not in x.toposort(): return 0
    if x.op is not Ops.INS or _iop(x) is not AMDOps.ADD: return None
    a, b = ctr_coeff(x.src[0]), ctr_coeff(x.src[1])
    return None if a is None or b is None else a+b
  if ctr_coeff(hload.src[1]) != 1 or ctr_coeff(fload.src[1]) != 1: return None
  used_vgprs = {r-256 for u in uops for r in _reg_idxs(u) if 256 <= r < 512}
  free = next((r for r in range(5, 249) if not used_vgprs.intersection(range(r, r+6))), None)
  if free is None: return None
  emitted = list(insts_for_uop(init)) + list(insts_for_uop(ctr))
  for u in body:
    if u is hload:
      pre, addr = _scaled_addr(TMP_VADDR, u.src[1], 2)
      emitted += pre + [r3.global_load_b64(v[free:free+1], addr, saddr=_src(u.src[0]))]
    elif u is fload:
      pre, addr = _scaled_addr(TMP_VADDR, u.src[1], 4)
      emitted += pre + [r3.global_load_b128(v[free+2:free+5], addr, saddr=_src(u.src[0]))]
    else: emitted += insts_for_uop(u)
  emitted.append(r3.s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  acc, half_first = _dst(init), fmac.src[1] is cast
  for lane in range(4):
    hreg, freg, hi = v[free+lane//2], v[free+2+lane], lane & 1
    emitted.append(r3.v_fma_mix_f32(acc, hreg if half_first else freg, freg if half_first else hreg, acc,
                                    opsel=hi * (1 if half_first else 2), opsel_hi=1 if half_first else 2, opsel_hi2=0))
  return cast_i + 6 - i, emitted

def _fused_lds_reduce_loop(uops:list[UOp], i:int) -> tuple[int, list]|None:
  """Vector-load an exact 16-value f32 LDS reduction while preserving its left-to-right sum order."""
  if i == 0 or i + 10 >= len(uops) or uops[i-1].op is not Ops.INS or _iop(uops[i-1]) is not AMDOps.BARRIER: return None
  barrier = uops[i-1]
  init, ctr, loop, cmp, exit_branch, load, add, copy, inc, back, out = uops[i:i+11]
  if not (all(u.op is Ops.INS for u in (init, ctr, loop, cmp, exit_branch, load, add, copy, inc, back, out)) and
          _iop(init) is AMDOps.MOV and init.dtype is dtypes.float32 and init.src and _is_zero_val(init.src[0]) and
          _iop(ctr) is AMDOps.MOV and _const_int(ctr) == 0 and
          _iop(loop) is AMDOps.LABEL and _iop(cmp) is AMDOps.CMP_GE and cmp.src[0] is ctr and _const_int(cmp.src[1]) == 16 and
          _iop(exit_branch) is AMDOps.CBRANCH_SCC1 and exit_branch.src == (cmp,) and exit_branch.tag == out.tag and
          _iop(load) is AMDOps.LLOAD and load.dtype is dtypes.float32 and _elem_count(load) == 1 and load.src[1] is ctr and
          load.src[0].op is Ops.AFTER and barrier in load.src[0].src and
          _iop(add) is AMDOps.ADD and add.dtype is dtypes.float32 and add.src == (init, load) and
          _iop(copy) is AMDOps.MOV and copy.src == (add,) and greg(copy) == greg(init) and
          _iop(inc) is AMDOps.ADD and inc.src[0] is ctr and _const_int(inc.src[1]) == 1 and greg(inc) == greg(ctr) and
          _iop(back) is AMDOps.BRANCH and back.tag == loop.tag and _iop(out) is AMDOps.LABEL): return None
  used_vgprs = {r-256 for u in uops for r in _reg_idxs(u) if 256 <= r < 512}
  free = next((r for r in range(5, 253) if not used_vgprs.intersection(range(r, r+4))), None)
  if free is None: return None
  acc, base_off = _dst(init), _lds_base_offset(load.src[0]) + _lds_byte_off(load)
  emitted = list(insts_for_uop(init)) + [r3.v_mov_b32_e32(TMP_VADDR, 0)]
  for block in range(4):
    emitted += [r3.ds_load_b128(vdst=v[free:free+3], addr=TMP_VADDR, **_ds_off(base_off + block*16)),
                r3.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)]
    emitted += [r3.v_add_f32_e32(acc, acc, v[free+lane]) for lane in range(4)]
  return 11, emitted

def insts_from_linear(lin:UOp):
  _FMAC_DELAY_N[0] = 0
  ops = list(lin.src)
  skip = _compute_amd_skip(ops)  # fused d16 hi LOADs still emit (d16_hi into lo)
  skip |= _half2_global_hi_skip(ops)
  d16_hi_lo: dict[UOp, UOp] = {}  # AMD_D16_HI removed — map always empty
  fma_hi_lo = _fma_mixhi_lo_map(ops)
  fma_pair_dst: dict[UOp, Reg] = {}
  mask_depth = 0
  # vm: (dest_regs, n_vmem_ops) in issue order — soft vmcnt must count ops, not UOps
  # (PACK_F16 can emit 16 loads; treating that as 1 desyncs vmcnt → MMU faults).
  pending_vm: list[tuple[set[int], int]] = []
  # AMD_SOFT_LGKM removed (never shipped; always hard lgkmcnt(0)).
  pending: dict[str, set[int]] = {"lgkm": set(), "vs": set()}
  items, targets = [], {}
  store_addr_cache = _StoreAddrCache()
  # Compact B: page-idx UOps already shifted to bytes (in-place <<1 once per idx UOp).
  byte_scaled: set[int]|None = set()  # always-on compact B page-idx <<1
  def emit(inst):
    items.append(inst)
  def wait_vm(allow:int=0):
    total = sum(n for _, n in pending_vm)
    if not total: return
    allow = max(0, min(allow, total))
    emit(_wait_for_domain("vm", allow))
    done = total - allow
    while done > 0 and pending_vm:
      regs, n = pending_vm[0]
      if n <= done:
        pending_vm.pop(0)
        done -= n
      else:
        pending_vm[0] = (regs, n - done)
        done = 0
  def wait_lgkm(allow:int=0):
    if pending["lgkm"]:
      emit(_wait_for_domain("lgkm"))
      pending["lgkm"].clear()
  def _lgkm_any() -> bool:
    return bool(pending["lgkm"])
  def _lgkm_has_sentinel() -> bool:
    return -1 in pending["lgkm"]
  def flush(*domains:str):
    for domain in domains:
      if domain == "vm": wait_vm(0)
      elif domain == "lgkm": wait_lgkm(0)
      elif pending[domain]:
        emit(_wait_for_domain(domain))
        pending[domain].clear()
  def flush_regs(regs:set[int]):
    need_i = next((i for i in range(len(pending_vm) - 1, -1, -1) if pending_vm[i][0] & regs), -1)
    if need_i >= 0: wait_vm(sum(n for _, n in pending_vm[need_i + 1:]))
    if pending["lgkm"] & regs:
      flush("lgkm")
  def note_vm(regs:set[int], insts:list):
    if (n:=_vm_load_count(insts)) and regs: pending_vm.append((regs, n))
  def note_lgkm(regs:set[int], insts:list|None=None):
    pending["lgkm"] |= regs
  def note_vs(regs:set[int]|int=-1):
    if isinstance(regs, int): pending["vs"].add(regs)
    else: pending["vs"] |= regs
  def _pending_src(regs:set[int]) -> bool:
    if any(pr & regs for pr, _ in pending_vm): return True
    return bool(pending["lgkm"] & regs)
  def _emit_uop(u, masked=False, with_store_cache=False):
    # Physical operands are not value identities: regalloc may redefine the same
    # VGPR between two predicates. Keep every explicitly rematerialized comparison.
    return list(insts_for_uop(u, skip, masked, store_addr_cache if with_store_cache else None,
                              d16_hi_lo, byte_scaled, fma_hi_lo, fma_pair_dst))
  # Required: d16 lo-before-hi (no-op — AMD_D16_HI removed). Optional: WMMA hoist/sink + emit-time store cluster.
  scheduled = _hoist_loads_before_wmma(_sink_wmma_past_loads(ops))
  # Post-regalloc UOp-only reordering is unsound: regalloc reuses VGPRs, so SSA-independent
  # ops can still conflict in phys regs (load A→use; load B both in v5 → batch clobbers A).
  # Swizzle/SLOAD batch, LLOAD hoist, and load gap-fill run in after_pre_regalloc instead.
  scheduled = _cluster_const_scratch_stores(scheduled)
  fma_pair_dst = _fma_pair_pack_dsts(scheduled, fma_hi_lo)
  # AMD_DCE_DEAD_MOV removed (wrong sum −21.6). Identity MOV already no-ops in insts_for_uop.
  early_emitted: set[int] = set()
  perm_selects_ready = False
  oi = 0
  while oi < len(scheduled):
    if oi in early_emitted:
      oi += 1
      continue
    u = scheduled[oi]
    # AMD_EMIT_LSHL_ADD removed with UOp FOLD (hang risk / never shipped).
    # Anything that may clobber TMP_SDATA0/1 invalidates cached permlanex selects.
    if u.op is Ops.INS and _iop(u) not in (AMDOps.PERMLANEX16, AMDOps.EXTRACT, AMDOps.MOV):
      perm_selects_ready = False
    if mask_depth == 0 and (fused_dot:=_fused_mixed_dot4_loop(scheduled, oi)) is not None:
      count, emitted = fused_dot
      flush("vm", "lgkm")
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      oi += count
      continue
    # Pair independent FMACs into VOPD (HIP SDPA uses dual-issue heavily).
    # Scan a short window — bank-compatible partners are often not adjacent.
    # Also try FMAC+ADD/SUB (HIP's common mixed form; dest/src1 banks are independent).
    if mask_depth == 0 and getenv("AMD_VOPD_FMAC", 0) and u.op is Ops.INS and _iop(u) is AMDOps.FMAC:
      partner_i: int|None = None
      vopd: list|None = None
      max_scan = 8
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        pair = _try_vopd_fmac_pair(u, cand) or _try_vopd_fmac_alu_pair(u, cand)
        if pair is None: continue
        # Emit cand early with u only if cand does not depend on intervening ops
        # (those would still be pending). Mid depending on u is fine — u writes first.
        if not _vopd_scan_safe(cand, scheduled[oi:k]) or any(_vopd.depends(cand, scheduled[m]) for m in range(oi + 1, k)):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    if mask_depth == 0 and getenv("AMD_VOPD_FMAC_ALU", 0) and u.op is Ops.INS and \
       _iop(u) in (AMDOps.ADD, AMDOps.SUB):
      partner_i = None
      vopd = None
      max_scan = 8
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        if _iop(cand) is not AMDOps.FMAC: continue
        if (pair:=_try_vopd_fmac_alu_pair(u, cand)) is None: continue
        if not _vopd_scan_safe(cand, scheduled[oi:k]) or any(_vopd.depends(cand, scheduled[m]) for m in range(oi + 1, k)):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.ADD:
      partner_i = None
      vopd = None
      max_scan = 8
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        if (pair:=_try_vopd_add_pair(u, cand)) is None: continue
        # Refuse if anything between produces a cand src or consumes cand (early emit).
        if not _vopd_scan_safe(cand, scheduled[oi:k]) or any(_vopd.depends(cand, scheduled[m]) or _vopd.depends(scheduled[m], cand)
               for m in range(oi + 1, k)):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    # Pair independent MULs / MAXs into VOPD (flash soft/scale has many parallel lanes).
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.MUL:
      partner_i = None
      vopd = None
      max_scan = 8
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        if (pair:=_try_vopd_mul_pair(u, cand)) is None: continue
        if not _vopd_scan_safe(cand, scheduled[oi:k]) or any(u in scheduled[m].src or cand in scheduled[m].src for m in range(oi + 1, k)):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.MAX:
      partner_i = None
      vopd = None
      max_scan = 8
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        if (pair:=_try_vopd_max_pair(u, cand)) is None: continue
        if not _vopd_scan_safe(cand, scheduled[oi:k]) or any(u in scheduled[m].src or cand in scheduled[m].src for m in range(oi + 1, k)):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    # Pair independent scalar MOVs into VOPD (const zero runs / copy streaks).
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.MOV:
      partner_i = None
      vopd = None
      max_scan = 32  # AMD_VOPD_MOV_SCAN hardcoded
      for k in range(oi + 1, min(oi + 1 + max_scan, len(scheduled))):
        if k in early_emitted: continue
        cand = scheduled[k]
        if (pair:=_try_vopd_mov_pair(u, cand)) is None: continue
        # Dual emits cand at u. Preserve its value across every intervening read/write/control-flow.
        between = scheduled[oi + 1:k]
        if any(_vopd.depends(cand, mid) for mid in between) or not _vopd_scan_safe(cand, between):
          continue
        partner_i, vopd = k, pair
        break
      if vopd is not None and partner_i is not None:
        src = set().union(*(_reg_idxs(s) for s in u.src)) | set().union(*(_reg_idxs(s) for s in scheduled[partner_i].src))
        if src and _pending_src(src): flush_regs(src)
        for inst in vopd: emit(inst)
        early_emitted.add(partner_i)
        oi += 1
        continue
    if mask_depth == 0 and (fused_reduce:=_fused_lds_reduce_loop(scheduled, oi)) is not None:
      count, emitted = fused_reduce
      flush("lgkm")
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      oi += count
      continue
    if mask_depth == 0 and (fused_ka:=_fuse_kernarg_smem_loads(scheduled, oi, skip, mask_depth)) is not None:
      count, emitted, kas = fused_ka
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      for ka in kas: note_lgkm(_reg_idxs(ka))
      oi += count
      continue
    if u.op is Ops.INS and _iop(u) is AMDOps.LABEL:
      flush("vm", "lgkm", "vs")
      store_addr_cache.clear()
      targets[u.tag] = len(items)
      oi += 1
      continue
    if u.op is Ops.INS and _iop(u) in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ):
      flush("vm", "lgkm", "vs")
      store_addr_cache.clear()
      inst = r3.s_branch(0) if _iop(u) is AMDOps.BRANCH else \
             r3.s_cbranch_scc1(0) if _iop(u) is AMDOps.CBRANCH_SCC1 else r3.s_cbranch_vccnz(0)
      items.append((inst, u.tag))
      oi += 1
      continue
    if _needs_vm_flush(u):
      # Soft wait on WMMA A/B/ACC srcs only — full vm drain killed prefetched next-B U16 overlap.
      # Soft lgkm tracks VGPR dests; LDS→WMMA (and some Q/K LDS paths) are not in that
      # scoreboard — soft flush_regs alone NaN'd cold flash prefill (~23040 lanes). Always
      # lgkmcnt(0) before WMMA; then soft-vm on A/B/ACC srcs.
      if u.op is Ops.INS and _iop(u) is AMDOps.WMMA:
        if _lgkm_any(): flush("lgkm")
        flush_regs(set().union(*(_reg_idxs(s) for s in u.src)))
      else:
        regs = set().union(*(_reg_idxs(s) for s in u.src), _reg_idxs(u))
        flush_regs(regs)
    # AMD_SPILL_DRAIN_LGKM removed (wash).
    if u.op is Ops.INS and _iop(u) is AMDOps.IF_MASK:
      store_addr_cache.clear()
      emitted = _emit_uop(u)
      for inst in emitted: emit(inst)
      mask_depth += 1
      if (domain:=_wait_domain_for_load(u)) is not None:
        if domain == "vm": note_vm(_reg_idxs(u), emitted)
        elif domain == "lgkm": note_lgkm(_reg_idxs(u), emitted)
        else: pending[domain] |= _reg_idxs(u)
      oi += 1
      continue
    # LDS stores must complete before s_barrier / next LLOAD (hand: waitcnt then barrier).
    if u.op is Ops.INS and _iop(u) is AMDOps.BARRIER:
      flush("lgkm")
    elif u.op is Ops.INS and _iop(u) is AMDOps.LLOAD and _lgkm_has_sentinel():
      flush("lgkm")
    # Scratch stores must complete before SLOAD / FILL (soft vscnt — was per-store wait storm).
    elif u.op is Ops.INS and _iop(u) in (AMDOps.SLOAD, AMDOps.FILL) and -1 in pending["vs"]:
      flush("vs")
    masked = mask_depth > 0 and u.op is Ops.INS and _iop(u) in _MASKED_MEM
    if u in skip:
      if u.op is Ops.INS and _iop(u) is AMDOps.END_MASK: mask_depth -= 1
      oi += 1
      continue
    # REG-stack park of same-stage swizzles is scheduled pre-regalloc as SWIZZLE×N,MOV×N
    # (_schedule_swizzle_mov_batches). Soft lgkm then shares one wait on the first MOV.
    # Do not emit-reorder here — that extends live ranges past regalloc and corrupts results.
    # Overlap independent VMEM/VALU with in-flight ds_swizzle (emit after swizzle, before add's lgkm wait).
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.SWIZZLE and \
       oi + 1 < len(scheduled) and (add:=scheduled[oi + 1]).op is Ops.INS and _iop(add) in (AMDOps.ADD, AMDOps.MAX) and u in add.src:
      emitted = _emit_uop(u)
      for inst in emitted: emit(inst)
      note_lgkm(_reg_idxs(u), emitted)
      # AMD_SINK_VALU_SWIZZLE removed (checksum). VMEM-only sink kept (ship default ON).
      k, moved = oi + 2, 0
      max_moved = 6
      # This sink is post-regalloc: don't move a load's assigned destination over any register
      # still live at the intervening ADD/MAX (the allocator saw the load after that consumer).
      add_regs = _reg_idxs(add) | set().union(*(_reg_idxs(s) for s in add.src))
      while k < len(scheduled) and moved < max_moved:
        if k in early_emitted or scheduled[k] in skip:
          k += 1
          continue
        cand = scheduled[k]
        if cand.op is Ops.INS and _iop(cand) is AMDOps.LOAD and _vmem_schedulable_load(cand) and \
           u not in cand.toposort() and add not in cand.toposort() and \
           not (_reg_idxs(cand) & add_regs) and \
           not any(cand in scheduled[m].toposort() for m in range(oi + 2, k)):
          src_regs = set().union(*(_reg_idxs(s) for s in cand.src))
          if src_regs and _pending_src(src_regs): flush_regs(src_regs)
          gap_emitted = _emit_uop(cand)
          for inst in gap_emitted: emit(inst)
          if _iop(cand) is AMDOps.LOAD: note_vm(_reg_idxs(cand), gap_emitted)
          early_emitted.add(k)
          moved += 1
          k += 1
          continue
        break
      oi += 1
      continue
    # Quantized WMMA epilogues can scalarize a contiguous packed accumulator store into
    # aliasing EXTRACT+LSTORE pairs. Rejoin only exact four-f32 groups after regalloc.
    if mask_depth == 0 and (fused_store:=_fused_lds_pack_store(scheduled, oi)) is not None:
      count, emitted, deps = fused_store
      if deps and _pending_src(deps): flush_regs(deps)
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      note_lgkm({-1})
      oi += count
      continue
    # Contiguous LSTORE×4 (IQ4 LUT: EXTRACT×4 then stores) → ds_store_b128.
    if mask_depth == 0 and (fused_cstore:=_fused_lds_contig_store(scheduled, oi)) is not None:
      count, emitted, deps = fused_cstore
      if deps and _pending_src(deps): flush_regs(deps)
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      note_lgkm({-1})
      oi += count
      continue
    # Contiguous SSTORE×4 (flash ACC spill / soft copy) → scratch_store_b128.
    if mask_depth == 0 and (fused_sstore:=_fused_scratch_contig_store(scheduled, oi, store_addr_cache)) is not None:
      count, emitted, deps = fused_sstore
      if deps and _pending_src(deps): flush_regs(deps)
      for inst in emitted: emit(inst)
      note_vs(-1)
      oi += count
      continue
    # Burst SSTORE streak: hoist addr ALU, then s_clause + scratch_store* (soft/ACC spills).
    # Skip if any 4-window in the streak can b128-fuse — greedy clausing would swallow that window.
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.SSTORE:
      j = oi + 1
      while j < len(scheduled) and scheduled[j] not in skip and scheduled[j].op is Ops.INS and \
            _iop(scheduled[j]) is AMDOps.SSTORE and j - oi < 32:
        j += 1
      if j - oi >= 2 and not any(_fused_scratch_contig_store(scheduled, start, None) is not None
                                  for start in range(oi, j - 3)):
        cache_snap = (store_addr_cache.key, store_addr_cache.page, store_addr_cache.lds_key)
        parts = [_emit_uop(scheduled[k], with_store_cache=True) for k in range(oi, j)]
        if all(_vm_store_count(p) >= 1 for p in parts):
          scales, stores = [], []
          for p in parts:
            sc, st = _split_scale_and_stores(p)
            scales.extend(sc)
            stores.extend(st)
          if len(stores) >= 2 and _tmp_vaddr_clause_safe(scales, stores):
            deps = set().union(*(_reg_idxs(scheduled[k].src[2]) for k in range(oi, j))) | \
                   set().union(*(_reg_idxs(scheduled[k].src[1]) for k in range(oi, j)))
            if deps and _pending_src(deps): flush_regs(deps)
            for inst in scales: emit(inst)
            emit(r3.s_clause(simm16=len(stores) - 1))
            for inst in stores: emit(inst)
            note_vs(-1)
            oi = j
            continue
        store_addr_cache.key, store_addr_cache.page, store_addr_cache.lds_key = cache_snap
    # Contiguous SLOAD×4 → scratch_load_b128 (soft/ACC epilogue reads).
    if mask_depth == 0 and (fused_sload:=_fused_scratch_contig_load(scheduled, oi, store_addr_cache)) is not None:
      count, emitted, deps = fused_sload
      if deps and _pending_src(deps): flush_regs(deps)
      for inst in emitted: emit(inst)
      note_vm(set().union(*(_reg_idxs(scheduled[k]) for k in range(oi, oi + count))), emitted)
      oi += count
      continue
    # Contiguous SLOAD×2 → scratch_load_b64 when b128 cannot match.
    if mask_depth == 0 and (fused_s64:=_fused_scratch_contig_load_b64(scheduled, oi, store_addr_cache)) is not None:
      count, emitted, deps = fused_s64
      if deps and _pending_src(deps): flush_regs(deps)
      for inst in emitted: emit(inst)
      note_vm(set().union(*(_reg_idxs(scheduled[k]) for k in range(oi, oi + count))), emitted)
      oi += count
      continue
    # Burst SLOAD streak: hoist addr ALU, then s_clause + scratch_load* (HIP/LLVM VMEM clause).
    # Only when scales are TMP-safe (shared CSE'd TMP_VADDR or ≤1 TMP scale).
    # Skip if any 4-window can b128-fuse (same greed trap as stores).
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.SLOAD:
      j = oi + 1
      while j < len(scheduled) and scheduled[j] not in skip and scheduled[j].op is Ops.INS and \
            _iop(scheduled[j]) is AMDOps.SLOAD and j - oi < 32:
        j += 1
      if j - oi >= 2 and not any(_fused_scratch_contig_load(scheduled, start, None) is not None
                                  for start in range(oi, j - 3)) and \
         not any(_fused_scratch_contig_load_b64(scheduled, start, None) is not None
                 for start in range(oi, j - 1)):
        scratch_cache_snap = (store_addr_cache.key, store_addr_cache.page, store_addr_cache._bcast_src, store_addr_cache.lds_key)
        parts = [_emit_uop(scheduled[k], with_store_cache=True) for k in range(oi, j)]
        if all(_vm_load_count(p) >= 1 for p in parts):
          scales, loads = [], []
          for p in parts:
            sc, ld = _split_scale_and_loads(p)
            scales.extend(sc)
            loads.extend(ld)
          if len(loads) >= 2 and _tmp_vaddr_clause_safe(scales, loads):
            if (deps:=set().union(*(_reg_idxs(scheduled[k].src[1]) for k in range(oi, j)))) and _pending_src(deps):
              flush_regs(deps)
            for inst in scales: emit(inst)
            emit(r3.s_clause(simm16=len(loads) - 1))
            for inst in loads: emit(inst)
            note_vm(set().union(*(_reg_idxs(scheduled[k]) for k in range(oi, j))), loads)
            oi = j
            continue
        store_addr_cache.key, store_addr_cache.page, store_addr_cache._bcast_src, store_addr_cache.lds_key = scratch_cache_snap
    if mask_depth == 0 and (fused_load:=_fused_lds_pack_load(scheduled, oi)) is not None:
      count, emitted, deps = fused_load
      if deps and _pending_src(deps): flush_regs(deps)
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      for k in range(oi, oi + count): note_lgkm(_reg_idxs(scheduled[k]))
      oi += count
      continue
    if mask_depth == 0 and (fused_2addr:=_fused_lds_2addr_b64(scheduled, oi)) is not None:
      count, emitted, deps = fused_2addr
      if deps and _pending_src(deps): flush_regs(deps)
      store_addr_cache.clear()
      for inst in emitted: emit(inst)
      for k in range(oi, oi + count): note_lgkm(_reg_idxs(scheduled[k]))
      oi += count
      continue
    # PERMLANEX16: install lane-select SGPRs once across EXTRACT-separated peers.
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.PERMLANEX16:
      store_addr_cache.clear()
      if not perm_selects_ready:
        emit(r3.s_mov_b32(TMP_SDATA0, 0x76543210))
        emit(r3.s_mov_b32(TMP_SDATA1, 0xfedcba98))
        perm_selects_ready = True
      pre, val = _vgpr_data(TMP_VDATA, u.src[0])
      for inst in pre: emit(inst)
      emit(r3.v_permlanex16_b32(_dst(u), val, TMP_SDATA0, TMP_SDATA1, opsel=1))
      oi += 1
      continue
    # Cluster consecutive LLOAD streaks (post _hoist_lloads_before_extracts): s_clause + ds_load burst.
    # Must not hoist multiple TMP_VADDR scales before loads (same bug as VMEM clause).
    # With AMD_LDS_DEST_ADDR, scales target distinct load VGPRs so the TMP check allows the burst.
    if mask_depth == 0 and u.op is Ops.INS and _iop(u) is AMDOps.LLOAD:
      j = oi + 1
      while j < len(scheduled) and scheduled[j] not in skip and scheduled[j].op is Ops.INS and \
            _iop(scheduled[j]) is AMDOps.LLOAD: j += 1
      if j - oi >= 2:
        parts = [_emit_uop(scheduled[k]) for k in range(oi, j)]
        if all(_lgkm_load_count(p) >= 1 for p in parts):
          store_addr_cache.clear()
          scales, loads = [], []
          for p in parts:
            sc, ld = _split_lgkm_scale_and_loads(p)
            scales.extend(sc)
            loads.extend(ld)
          if len(loads) >= 2 and _tmp_vaddr_clause_safe(scales, loads):
            for inst in scales: emit(inst)
            emit(r3.s_clause(simm16=len(loads) - 1))
            for inst in loads: emit(inst)
            for k in range(oi, j): note_lgkm(_reg_idxs(scheduled[k]))
            oi = j
            continue
    # Cluster consecutive LSTORE streaks: CSE scales (AMD_LDS_ADDR_CSE) + s_clause + ds_store*.
    if mask_depth == 0 and \
       u.op is Ops.INS and _iop(u) is AMDOps.LSTORE:
      j = oi + 1
      while j < len(scheduled) and scheduled[j] not in skip and scheduled[j].op is Ops.INS and \
            _iop(scheduled[j]) is AMDOps.LSTORE and j - oi < 32:
        j += 1
      if j - oi >= 2:
        cache_snap = (store_addr_cache.key, store_addr_cache.page, store_addr_cache.lds_key)
        parts = [_emit_uop(scheduled[k], with_store_cache=True) for k in range(oi, j)]
        if all(_lgkm_store_count(p) >= 1 for p in parts):
          scales, stores = [], []
          for p in parts:
            sc, st = _split_lgkm_scale_and_stores(p)
            scales.extend(sc)
            stores.extend(st)
          if len(stores) >= 2 and _tmp_vaddr_clause_safe(scales, stores):
            deps = set().union(*(_reg_idxs(scheduled[k].src[2]) for k in range(oi, j))) | \
                   set().union(*(_reg_idxs(scheduled[k].src[1]) for k in range(oi, j)))
            if deps and _pending_src(deps): flush_regs(deps)
            for inst in scales: emit(inst)
            emit(r3.s_clause(simm16=len(stores) - 1))
            for inst in stores: emit(inst)
            note_lgkm(set().union(*(_reg_idxs(scheduled[k].src[2]) for k in range(oi, j))))
            oi = j
            continue
        store_addr_cache.key, store_addr_cache.page, store_addr_cache.lds_key = cache_snap
    # Cluster contiguous A B128 (half×8+): one s_clause over the burst (LLVM B128×8).
    # Per-tile s_clause stripped from _global_load_insts. Skip addr ALU between wide A tiles.
    if _clauseable_wide_half_gload(u, skip, mask_depth):
      j = oi + 1
      while j < len(scheduled):
        if _is_addr_alu(scheduled[j]):
          j += 1
          continue
        if _clauseable_wide_half_gload(scheduled[j], skip, mask_depth):
          j += 1
          continue
        break
      while j > oi + 1 and _is_addr_alu(scheduled[j - 1]): j -= 1
      idxs = [k for k in range(oi, j) if _clauseable_wide_half_gload(scheduled[k], skip, mask_depth)]
      parts = [_emit_uop(scheduled[k]) for k in range(oi, j)]
      if idxs and sum(_vm_load_count(p) for p in parts) >= 2:
        store_addr_cache.clear()
        scales, loads = [], []
        for p in parts:
          sc, ld = _split_scale_and_loads(p)
          scales.extend(sc)
          loads.extend(ld)
        if _tmp_vaddr_clause_safe(scales, loads):
          for inst in scales: emit(inst)
          emit(r3.s_clause(simm16=len(loads) - 1))
          for inst in loads: emit(inst)
          for k in idxs: note_vm(_reg_idxs(scheduled[k]), parts[k - oi])
          oi = j
          continue
    # Cluster packed quant B128 weight loads (uint32×4): one <<2 of the shared base, then
    # s_clause + global_load_b128 with imm offsets (HIP/LLVM pattern for Q5/Q4 tiles).
    if (info0:=_quant_b128_clause_info(u, skip, mask_depth)) is not None:
      saddr0, base0, off0 = info0
      group = [(u, off0)]
      j = oi + 1
      while len(group) < 8:
        # Index ADDs (base+4/base+8) sit between packed B128 loads; skip — offsets cover them.
        k = j
        while k < len(scheduled) and scheduled[k].op is Ops.INS and _iop(scheduled[k]) is AMDOps.ADD:
          k += 1
        if k >= len(scheduled): break
        info = _quant_b128_clause_info(scheduled[k], skip, mask_depth)
        if info is None: break
        saddr, base, off = info
        if saddr is not saddr0 or base is not base0: break
        if off <= group[-1][1] or off > 0xfff: break
        group.append((scheduled[k], off))
        j = k + 1
      if len(group) >= 2:
        store_addr_cache.clear()
        pre, addr = _scaled_addr(TMP_VADDR, base0, 4)
        for inst in pre: emit(inst)
        emit(r3.s_clause(simm16=len(group) - 1))
        for su, off in group:
          kw = {"offset": off} if off else {}
          gld = r3.global_load_b128(_dst(su), addr, saddr=_src(saddr0), **kw)
          emit(gld)
          note_vm(_reg_idxs(su), [gld])
        oi = j
        continue
    # Cluster scalar half loads: dest-as-addr scales, then s_clause + tight VMEM (LLVM-style B).
    # Quant decode uses the same pattern for packed u32 weight reads (linear_q6).
    # Always on. With AMD_D16_HI lo+…hi+ batch: extend the clause through following d16_his
    # (u16+d16_hi in one s_clause); do not hard-flush on lo mid-clause.
    if u not in d16_hi_lo and _clauseable_scalar_vmem_gload(u, skip, mask_depth):
      j = oi + 1
      while j < len(scheduled) and scheduled[j] not in d16_hi_lo and \
            _clauseable_scalar_vmem_gload(scheduled[j], skip, mask_depth): j += 1
      j_hi = j
      if d16_hi_lo and j > oi:
        while j_hi < len(scheduled) and scheduled[j_hi] in d16_hi_lo: j_hi += 1
      end = j_hi if j_hi - j >= 2 else j
      if end - oi >= 2:
        parts = [_emit_uop(scheduled[k]) for k in range(oi, end)]
        hi_ok = all(k < j or (not any(getattr(i, "vdst", None) == TMP_VDATA for i in parts[k - oi]))
                    for k in range(j, end))
        if hi_ok and all(_vm_load_count(p) == 1 for p in parts):
          store_addr_cache.clear()
          raw_scales, loads = [], []
          for p in parts:
            sc, ld = _split_scale_and_loads(p)
            raw_scales.extend(sc)
            loads.extend(ld)
          # Float×4 UPCAST B128s scale distinct byte offsets into TMP_VADDR; hoisting+dedup
          # would clobber addr (both loads read slot0). Compact B half: one <<1 is OK.
          if _tmp_vaddr_clause_safe(raw_scales, loads):
            scales, seen_scale = [], set()
            for s in raw_scales:
              # Compact B: one in-place <<1 per page idx; drop duplicate scales in the hoist.
              dst = getattr(s, "vdst", None)
              key = getattr(dst, "offset", None)
              if key is not None:
                if key in seen_scale: continue
                seen_scale.add(key)
              scales.append(s)
            for inst in scales: emit(inst)
            emit(r3.s_clause(simm16=len(loads) - 1))
            for inst in loads: emit(inst)
            for k, p in enumerate(parts):
              su = scheduled[oi + k]
              note_vm(_reg_idxs(d16_hi_lo[su]) if su in d16_hi_lo else _reg_idxs(su), p)
            oi = end
            continue
    if u in d16_hi_lo: flush_regs(_reg_idxs(d16_hi_lo[u]))
    is_mem_addr_cse = u.op is Ops.INS and _iop(u) in (AMDOps.STORE, AMDOps.SSTORE, AMDOps.SLOAD, AMDOps.LSTORE, AMDOps.LLOAD)
    # Scratch SPILL/FILL always clobber TMP_VADDR (page base). Clear before emit so a
    # following STORE/SLOAD CSE cannot reuse a stale TMP_VADDR across spill-on-evict.
    if u.op is Ops.INS and _iop(u) in (AMDOps.SPILL, AMDOps.FILL): store_addr_cache.clear()
    emitted = _emit_uop(u, masked, with_store_cache=is_mem_addr_cse)
    # VALU copy of an outstanding VMEM/LDS dest must wait first (PACK/MOV across pools).
    if emitted and u.op is Ops.INS and _iop(u) in (AMDOps.PACK_F16, AMDOps.PACK, AMDOps.EXTRACT, AMDOps.MOV):
      src = set().union(*(_reg_idxs(s) for s in u.src))
      for s in u.src:
        if s in d16_hi_lo: src |= _reg_idxs(d16_hi_lo[s])
      if src and _pending_src(src): flush_regs(src)
    # EXTRACT between C-stores often emits nothing (pack+lane alias); only clobber CSE on real emits.
    # Keep page CSE across CAST (uses TMP_VDATA, not TMP_VADDR) — cast-before-store otherwise
    # re-scales the C base for every half store (~100 extra V_LSHL_ADD).
    # half×16 STORE may V_ADD into TMP_VADDR for the second b128 — drop page CSE.
    # Scratch SSTORE/SLOAD also ADD segment base once; keep CSE across those.
    if _iop(u) is AMDOps.STORE and any(getattr(i, "op_name", "") == "V_ADD_NC_U32_E64" for i in emitted):
      store_addr_cache.clear()
    elif u.op is Ops.INS and _iop(u) in (AMDOps.SPILL, AMDOps.FILL):
      store_addr_cache.clear()
    elif not is_mem_addr_cse and any(getattr(i, "vdst", None) == TMP_VADDR for i in emitted):
      store_addr_cache.clear()
    vm_after_wait: list = []
    saw_vm_wait0 = False
    for inst in emitted:
      emit(inst)
      if getattr(inst, "op_name", "") == "S_WAITCNT_VMCNT" and getattr(inst, "simm16", None) == 0:
        pending_vm.clear()
        vm_after_wait = []
        saw_vm_wait0 = True
      elif getattr(inst, "op_name", "") == "S_WAITCNT_LGKMCNT" and getattr(inst, "simm16", None) == 0:
        pending["lgkm"].clear()
      else:
        vm_after_wait.append(inst)
    if u.op is Ops.INS and _iop(u) is AMDOps.END_MASK: mask_depth -= 1
    if (domain:=_wait_domain_for_load(u)) is not None:
      regs = _reg_idxs(d16_hi_lo[u]) if u in d16_hi_lo else _reg_idxs(u)
      if domain == "vm": note_vm(regs, vm_after_wait if saw_vm_wait0 else emitted)
      elif domain == "lgkm": note_lgkm(regs, emitted)
      else: pending[domain] |= regs
      # HIP flash_decode emits s_delay_alu after ds_swizzle; opt-in overlap with lgkm.
    if (domain:=_wait_domain_for_store(u)) is not None:
      regs = _store_src_regs(u)
      if domain == "vs": note_vs(regs)
      elif domain == "lgkm": note_lgkm(regs, emitted)
      else: pending[domain] |= regs
    oi += 1
  # Drain outstanding global stores before s_endpgm (appended by the renderer).
  flush("vs")
  # Relax out-of-range SOPP branches into a fixed-width getpc/add/setpc sequence. Keep
  # long forms fixed at 24 bytes (28 with the inverted conditional) so label layout is stable.
  long_branches: set[int] = set()
  while True:
    sizes = [(28 if getattr(item[0], "op_name", "").startswith("S_CBRANCH_") else 24) if i in long_branches else
             (4 if isinstance(item, tuple) else len(item.to_bytes())) for i,item in enumerate(items)]
    positions = [0]
    for sz in sizes: positions.append(positions[-1] + sz)
    new_long = set(long_branches)
    for i,item in enumerate(items):
      if not isinstance(item, tuple): continue
      _, target = item
      if target not in targets: raise CompileError(f"missing branch {target}")
      delta = (positions[targets[target]] - (positions[i] + 4)) // 4
      if not -0x8000 <= delta <= 0x7fff: new_long.add(i)
    if new_long == long_branches: break
    long_branches = new_long
  insts = []
  for i,item in enumerate(items):
    if not isinstance(item, tuple):
      insts.append(item)
      continue
    inst, target = item
    target_byte = positions[targets[target]]
    if i not in long_branches:
      inst.simm16 = ((target_byte - (positions[i] + 4)) // 4) & 0xffff
      insts.append(inst)
      continue
    conditional = getattr(inst, "op_name", "").startswith("S_CBRANCH_")
    getpc_byte = positions[i] + (4 if conditional else 0)
    delta = (target_byte - (getpc_byte + 4)) & 0xffffffffffffffff
    long = [r3.s_getpc_b64(TMP_BRANCH),
            r3.s_add_u32(TMP_BRANCH[0], TMP_BRANCH[0], delta & 0xffffffff),
            r3.s_addc_u32(TMP_BRANCH[1], TMP_BRANCH[1], delta >> 32)]
    while sum(len(x.to_bytes()) for x in long) < 20: long.append(r3.s_nop(0))
    long.append(r3.s_setpc_b64(ssrc0=TMP_BRANCH))
    if sum(len(x.to_bytes()) for x in long) != 24: raise CompileError("invalid long branch size")
    if conditional:
      inverse = r3.s_cbranch_scc0 if inst.op_name == "S_CBRANCH_SCC1" else r3.s_cbranch_vccz
      insts.append(inverse(sum(len(x.to_bytes()) for x in long) // 4))
    insts.extend(long)
  return insts


def _wmma_has_fused_float_to_half_cast(tk) -> bool:
  """True if a WMMA operand still contains float→half CAST (not a realized half buffer)."""
  for u in tk.ast.toposort():
    if u.op is not Ops.WMMA: continue
    for src in u.src[:2]:
      for x in src.toposort():
        if x.op is Ops.CAST and x.dtype is dtypes.half and x.src and x.src[0].dtype != dtypes.half:
          return True
  return False

# --- TC / WMMA hand schedule opts (expand tiles, prefetch A) ---
def apply_tc_hand_opts(tk, rngs):
  from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
  # Fused (a.half()@b.half()) MMU-faults on DIRECT under any WMMA schedule (2026-09-27 HW);
  # separate half.realize() then ah@bh is fine. Refuse TC so ALU path runs (HIP/LLVM OK).
  if _wmma_has_fused_float_to_half_cast(tk):
    raise KernelOptError("RDNA3 DIRECT: fused float→half CAST into WMMA MMU-faults")
  if not getenv("TC_LDS_AB", 0): return apply_register_tc_hand_opts(tk, rngs)
  lds_ab = getenv("TC_LDS_AB", 0)
  up_cap = getenv("TC_UPCAST", 4)
  loc_cap = getenv("TC_LOCAL", 2 if lds_ab else 4)
  up16 = allow_upcast16()
  max_tiles = min(getenv("TC_UPCAST_TILES", 16 if up16 else 8), 8 if not up16 else 10**9)
  if lds_ab:
    # ALLOW_LDS_PRODUCT8=1 (default): cap expand budget at product-8. =0: tighter (≤4).
    if getenv("ALLOW_LDS_PRODUCT8", 1) == 0:
      up_cap, max_tiles = min(up_cap, 2), min(max_tiles, 4)
    else:
      max_tiles = min(max_tiles, 8)
  local_dims, loc_szs = ([0, 1], [8, 4, 2]) if lds_ab else ([0], [4, 2])
  def do_local():
    if lds_ab:
      # Asymmetric LOCAL (≤loc_cap × ≤2). Symmetric 4×4 TDRs on display GPUs.
      for tc_dim, cap in ((0, loc_cap), (1, min(2, loc_cap) if loc_cap else 0)):
        if not cap: continue
        if (szs := [sz for sz in loc_szs if sz <= cap and rngs[tc_dim].src[0].divides(sz) is not None]):
          try: rngs[tc_dim] = tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[tc_dim]), (szs[0], AxisType.LOCAL)))[0]
          except KernelOptError: pass
      return
    for tc_dim in local_dims:
      if (szs := [sz for sz in loc_szs if sz <= loc_cap and rngs[tc_dim].src[0].divides(sz) is not None]):
        try: rngs[tc_dim] = tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[tc_dim]), (szs[0], AxisType.LOCAL)))[0]
        except KernelOptError: pass
  def do_upcast() -> int:
    tiles = 1
    for i, tc_dim in enumerate([1, 0]):
      other = rngs[0 if tc_dim == 1 else 1]
      # Avoid single-axis UPCAST=4 (product-4 as 4×1): wrong on register path. Leave a factor ≥2
      # for the other dim when it can still upcast.
      szs = []
      for sz in [5, 4, 3, 2]:
        if sz > up_cap or tiles * sz > max_tiles or rngs[tc_dim].src[0].divides(sz) is None: continue
        rem = max_tiles // (tiles * sz)
        if i == 0 and rem == 1 and other.src[0].divides(2) is not None and sz > 2: continue
        szs.append(sz)
      if szs:
        rngs[tc_dim] = tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[tc_dim]), (szs[0], AxisType.UPCAST)))[0]
        tiles *= szs[0]
    return tiles
  if lds_ab:
    do_local()
    tiles = do_upcast()
    # UNROLL multiplies WMMA STACK tiles; past max_tiles expand soft-fails then unroll_axis
    # IndexErrors. Only apply when the upcast×unroll product still fits the LDS expand budget.
    if (ku := getenv("TC_LDS_UNROLL", 0)) and tk.upcastable_dims(reduce=True) and tiles * ku <= max_tiles:
      try: tk.apply_opt(Opt(OptOps.SPLIT, tk.upcastable_dims(reduce=True)[0], (ku, AxisType.UPCAST)))
      except KernelOptError: pass
  else:
    do_upcast()
    do_local()
  # TC_LDS_GROUP omitted: GROUP≥2 on WMMA half GEMM needs >64KB LDS (always KernelOptError).

def apply_register_tc_hand_opts(tk, rngs):
  if getenv("TC_LDS_AB", 0): return apply_tc_hand_opts(tk, rngs)
  from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
  max_prod = 16 if allow_upcast16() else 8
  tiles = 1
  for i, tc_dim in enumerate([1, 0]):
    other = rngs[0 if tc_dim == 1 else 1]
    szs = []
    for sz in [5, 4, 3, 2]:
      if tiles * sz > max_prod or rngs[tc_dim].src[0].divides(sz) is None: continue
      rem = max_prod // (tiles * sz)
      if i == 0 and rem == 1 and other.src[0].divides(2) is not None and sz > 2: continue
      szs.append(sz)
    if szs:
      rngs[tc_dim] = tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[tc_dim]), (szs[0], AxisType.UPCAST)))[0]
      tiles *= szs[0]
  if (szs := [sz for sz in [4, 2] if rngs[0].src[0].divides(sz) is not None]):
    try: tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[0]), (szs[0], AxisType.LOCAL)))
    except KernelOptError: pass

def merge_adjacent_half_loads(sink:UOp) -> UOp:
  """Re-widen devectorized scalar half global loads into SHRINK×4 for B64 VMEM (flash _vec_load)."""
  import itertools
  from collections import defaultdict
  from tinygrad.dtype import Invalid
  memory: defaultdict[tuple, dict[int, list[UOp]]] = defaultdict(dict)
  for u in sink.toposort():
    if u.op is not Ops.LOAD or u.dtype is not dtypes.half or u.max_numel() != 1: continue
    if len(u.src) != 1 or u.src[0].op is not Ops.INDEX: continue
    buf, idx_u = u.src[0].src[0], u.src[0]
    if buf.addrspace is not AddrSpace.GLOBAL: continue
    idx, valid = idx_u.get_idx(), idx_u.get_valid()
    root_src: UOp | str
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST and isinstance((c:=idx.src[1].val), int):
      root_src, arg = idx.src[0], c
    elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST and isinstance((c:=idx.src[0].val), int):
      root_src, arg = idx.src[1], c
    elif idx.op is Ops.CONST and idx.val is not Invalid and isinstance(idx.val, int):
      root_src, arg = "CONST", idx.val
    else: continue
    memory[(buf, root_src, valid, u.arg)].setdefault(arg, []).append(u)
  replacements: dict[UOp, UOp] = {}
  for (buf, base, valid, ld_arg), offsets in memory.items():
    for full_grp in ([x for _, x in g] for _, g in itertools.groupby(enumerate(sorted(offsets)), lambda x: x[1]-x[0])):
      i = 0
      while i < len(full_grp):
        if i + 4 <= len(full_grp) and full_grp[i:i+4] == list(range(full_grp[i], full_grp[i]+4)):
          grp = full_grp[i:i+4]
          offset = (base + grp[0]) if isinstance(base, UOp) else UOp.const(grp[0], dtypes.int32)
          if valid is not None: offset = offset.valid(valid)
          ld = UOp(Ops.SHRINK, src=(buf.flatten(), offset, UOp.const(4, dtypes.int32))).load(arg=ld_arg)
          for j, off in enumerate(grp):
            for oo in offsets[off]: replacements[oo] = ld.index(j)
          i += 4
        else: i += 1
  return sink.substitute(replacements, name="merge adjacent half loads") if replacements else sink

def merge_adjacent_uint32_loads(sink:UOp) -> UOp:
  """Re-widen scalar uint32 global loads into SHRINK×4 for B128 (IQ4/Q6 weight headers)."""
  # AMD_MERGE_U32 hardcoded ON (default was 1; never labbed OFF)
  import itertools
  from collections import defaultdict
  from tinygrad.dtype import Invalid
  memory: defaultdict[tuple, dict[int, list[UOp]]] = defaultdict(dict)
  for u in sink.toposort():
    if u.op is not Ops.LOAD or u.dtype is not dtypes.uint32 or u.max_numel() != 1: continue
    if len(u.src) != 1 or u.src[0].op is not Ops.INDEX: continue
    buf, idx_u = u.src[0].src[0], u.src[0]
    if buf.addrspace is not AddrSpace.GLOBAL: continue
    idx, valid = idx_u.get_idx(), idx_u.get_valid()
    root_src: UOp | str
    if idx.op is Ops.ADD and idx.src[1].op is Ops.CONST and isinstance((c:=idx.src[1].val), int):
      root_src, arg = idx.src[0], c
    elif idx.op is Ops.ADD and idx.src[0].op is Ops.CONST and isinstance((c:=idx.src[0].val), int):
      root_src, arg = idx.src[1], c
    elif idx.op is Ops.CONST and idx.val is not Invalid and isinstance(idx.val, int):
      root_src, arg = "CONST", idx.val
    else: continue
    memory[(buf, root_src, valid, u.arg)].setdefault(arg, []).append(u)
  replacements: dict[UOp, UOp] = {}
  for (buf, base, valid, ld_arg), offsets in memory.items():
    for full_grp in ([x for _, x in g] for _, g in itertools.groupby(enumerate(sorted(offsets)), lambda x: x[1]-x[0])):
      i = 0
      while i < len(full_grp):
        if i + 4 <= len(full_grp) and full_grp[i:i+4] == list(range(full_grp[i], full_grp[i]+4)):
          grp = full_grp[i:i+4]
          offset = (base + grp[0]) if isinstance(base, UOp) else UOp.const(grp[0], dtypes.int32)
          if valid is not None: offset = offset.valid(valid)
          ld = UOp(Ops.SHRINK, src=(buf.flatten(), offset, UOp.const(4, dtypes.int32))).load(arg=ld_arg)
          for j, off in enumerate(grp):
            for oo in offsets[off]: replacements[oo] = ld.index(j)
          i += 4
        else: i += 1
  return sink.substitute(replacements, name="merge adjacent uint32 loads") if replacements else sink

# Re-export INDEX mops (tests / callers); definition lives in codegen.late.index_mops.
from tinygrad.codegen.late.index_mops import pm_index_mops, _index_through_reshape, _index_through_permute  # noqa: F401,E402

def isa_float4_coalesce(uops, ctx):
  from tinygrad.renderer.isa import ISARenderer
  from collections import defaultdict
  f4 = ctx.float4_dtypes if isinstance(ctx, ISARenderer) else None
  if f4 is None: return None, True, False, None
  if any(u.dtype in (dtypes.bfloat16, *dtypes.fp8s) for u in uops): return f4, True, True, None
  uses = defaultdict(list)
  for u in uops:
    for su in u.src: uses[su].append(u)
  for u in uops:
    if u.op not in {Ops.LOAD, Ops.STORE} or is_image_shape(u.src[0].src[0]._shape) or u.src[0].op is not Ops.INDEX: continue
    buf = u.src[0].src[0]
    if buf.addrspace is AddrSpace.REG: continue  # weakfloat ACC stores must not disable global half B128
    value_dtype = u.src[1].dtype if u.op is Ops.STORE else u.dtype
    if buf.dtype not in f4 or value_dtype not in (*f4, dtypes.weakfloat): return f4, False, False, uses
  return f4, True, False, uses

def isa_float4_mem_ok(f4, float4_safe, buf, value_dtype, u, uses, valid) -> bool:
  if f4 is None: return True
  if buf.dtype is dtypes.uint8:
    return u.op is Ops.LOAD and value_dtype is dtypes.uint8 and buf.addrspace is not AddrSpace.LOCAL and \
      bool(valid.op is Ops.CONST and valid.val is True)
  # float4_safe goes False when the kernel also has non-f4 mem (e.g. int/u8). Still coalesce
  # native float/half ops — otherwise LLM E_32 / mixed graphs stay on scalar b32 forever.
  if buf.dtype != value_dtype or buf.dtype not in f4: return False
  if not float4_safe and value_dtype not in (*f4, dtypes.weakfloat): return False
  # BITCAST may reinterpret packed lanes; keep those scalar. CAST (e.g. half→float after
  # SDPA KV loads) is fine: wide VMEM + EXTRACT + cvt matches HIP's B64 half traffic.
  if u.op is Ops.LOAD and any(v.op is Ops.BITCAST for v in uses[u]): return False
  if u.op is Ops.STORE and u.src[1].op is Ops.BITCAST: return False
  # Shared non-trivial valids still coalesce together (keyed by `valid` in memory_coalescing).
  # LLM E_32 RoPE/KV glue is gated; HIP wraps B128 in saveexec — ungated-only left AMD on b32×N.
  return True

_REG_STORE_SINK_ALU_OPS = _sched._REG_STORE_SINK_ALU_OPS
# FMAC excluded: sink+two-address uses[0] coalesce poisons flash ACC (NaN).
# EXTRACT: WMMA/PACK lane → REG slot (64 single-use REG_STOREs on flash).
# AMD_MOV_SINK_ALU removed (prefer_phys covers MAX).

def _sink_alu_into_reg_store(ops: list[UOp]) -> list[UOp]:
  return _sched.sink_alu_into_reg_store(ops)

# --- renderer entry: isel, remat, regalloc hooks, asm ---
class AMDLinearContext(LinearContext):
  ren: AMDRenderer
  def __init__(self, ren:AMDRenderer):
    super().__init__(ren)
    self.last_const_register: int|None = None

  def register_buffers(self, uops:list[UOp]) -> set[UOp]: return _reg_promotable_buffers(PreRegAllocContext(uops))

  def register_order(self, u:UOp, cons:tuple[Register, ...]) -> tuple[Register, ...]:
    # Prefer adjacent even/odd destinations for independent constant MOVs (VOPD).
    if not self.ren._vopd_pair_op(u) or not cons: return cons
    want = 0
    if self.last_const_register is not None:
      next_idx = self.last_const_register + 1
      exact = tuple(r for r in cons if r.index == next_idx)
      if exact: return exact + tuple(r for r in cons if r.index != next_idx)
      want = 1 - self.last_const_register % 2
    return tuple(r for r in cons if r.index % 2 == want) + tuple(r for r in cons if r.index % 2 != want)

  def allocated(self, u:UOp, reg:Register) -> None:
    if self.ren._vopd_pair_op(u): self.last_const_register = reg.index

class AMDRenderer(ISARenderer):
  device = "AMD"
  has_local = True
  has_shared = True
  supports_float4 = True
  float4_dtypes = (dtypes.float32, dtypes.half)
  wide_regalloc = True
  linear_ctx_type = AMDLinearContext
  preferred_reduce_group = 16
  preferred_complex_matvec_group = 32
  global_max = (0x8fffffff, 0x8fffffff, 0x8fffffff)
  # 2D locals (WARP=lidx0 → e.g. (32,4,1)); needs gfx1100 USER_SGPR=15 (elf.py).
  local_max = (1024, 1024, 64)
  local_prod_max = 1024
  shared_max = 65536
  pre_isel_matcher = pre_isel_matcher
  isel_matcher = isel_matcher
  pre_regalloc_matcher = pre_regalloc_matcher
  post_regalloc_matcher = post_regalloc_matcher
  pm_group_reduce = pm_warp_group_reduce
  pm_stage_wmma_ab = pm_stage_wmma_ab
  extra_expander = pm_expand_wmma_lds
  _code_ops = (Ops.ADD, Ops.SUB, Ops.MUL, Ops.RECIPROCAL, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC, Ops.SIN, Ops.MAX,
               Ops.SHL, Ops.SHR, Ops.AND, Ops.OR, Ops.XOR, Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ)
  code_for_op = {op: (lambda: None) for op in _code_ops}

  def __init__(self, target:Target):
    if not target.arch.startswith("gfx11"): raise RuntimeError(f"AMDRenderer is RDNA3/gfx11 only, got {target.arch}")
    super().__init__(target)
    self.tensor_cores = [x for x in tc.get_amd(target.arch) if (x.dtype_in, x.dtype_out) == (dtypes.half, dtypes.float)]
    # AMD_VGPR_EXCLUDE_ACC removed (fair checksum wrong; never protected mid-band ACC).

  def get_reduce_unroll(self, size:int, ast:UOp) -> int|None:
    # Broad unrolling of complex quantized reductions explodes native ISA. It is still valuable
    # for small-output quantized projections, where the 32-way loop is on the decoder's critical path.
    ast_uops = ast.toposort()
    output_size = next((u.max_numel() for u in ast_uops if u.op is Ops.PARAM and getattr(u.arg, "slot", None) == 0), None)
    quant_buffers = sum(u.op is Ops.PARAM and u.dtype is dtypes.uchar for u in ast_uops)
    small_quant_projection = size == 32 and isinstance(output_size, int) and 0 < output_size <= 512 and \
                             quant_buffers > 0
    # A four-way split avoids a runtime loop explosion for plain GEMV. Fused projections with
    # multiple packed inputs benefit enough to justify fully unrolling this innermost dimension.
    if small_quant_projection: return 0 if quant_buffers >= 2 else 4
    if size <= 3 or len(ast_uops) <= 32: return 0
    return None

  def get_large_reduce_unroll(self, size:int, ast:UOp) -> int|None:
    # The generic four-way split cannot expose adjacent loads for odd trip counts such as
    # Llama's 128256-token vocabulary (501 inner iterations). RDNA3 has native B96 loads.
    is_max = any(u.op is Ops.REDUCE and u.arg[0] is Ops.MAX for u in ast.toposort())
    return 3 if is_max and size % 4 != 0 and size % 3 == 0 else None

  def get_grouped_reduce_unroll(self, k) -> int|None:
    # GROUPTOP returns before the generic reduce-unroll heuristic. LLVM subsequently vectorizes simple inner
    # reduction loops, but direct ISA needs those adjacent iterations exposed while coalescing is still possible.
    unroll, ast_uops = 8, k.ast.toposort()  # AMD_GROUPED_REDUCE_UNROLL hardcoded
    if not unroll or k.reduceop is None or k.reduceop.arg[0] not in (Ops.ADD, Ops.MAX) or \
       (k.reduceop.arg[0] is Ops.ADD and len(ast_uops) > 32) or \
       any(u.op is Ops.PARAM and u.dtype is dtypes.uint8 for u in ast_uops) or not k.upcastable_dims(reduce=True): return None
    size = k.full_shape[k.upcastable_dims(reduce=True)[-1]]
    if not isinstance(size, int): return None
    if size >= unroll and size % unroll == 0: return unroll
    return self.get_large_reduce_unroll(size, k.ast)

  def get_complex_matvec_rows(self, k) -> int:
    # Large packed-quant projections amortize the activation reads across two rows.
    # Below this threshold the extra code/register pressure only increases cold latency.
    # output elems = non-reduce axes (Scheduler.output_shape was removed in #18090)
    output_size = prod((1 if k.rngs[i] in k.reduce_ranges else s) for i,s in enumerate(k.full_shape))
    if not isinstance(output_size, int) or output_size < 16384: return 1
    if not any(u.op is Ops.PARAM and u.dtype is dtypes.uchar for u in k.ast.toposort()): return 1
    return 2

  def apply_quant_matvec_opts(self, k) -> bool:
    """Expose memory-level parallelism that LLVM's loop optimizer otherwise supplies for Q8_0/IQ4_XS GEMV."""
    from tinygrad.codegen.opt import Opt, OptOps
    output_size = prod((1 if k.rngs[i] in k.reduce_ranges else s) for i,s in enumerate(k.full_shape))
    if k.reduceop is None or k.reduceop.arg[0] is not Ops.ADD or not isinstance(output_size, int) or output_size < 1024:
      return False
    if not any(u.op is Ops.PARAM and u.dtype is dtypes.uchar for u in k.ast.toposort()): return False
    reduce_sizes = tuple(k.full_shape[a] for a in k.axes_of(reduce=True))

    # Q8_0: use 16 lanes on the outer block loop and expose eight packed values per iteration.
    if len(reduce_sizes) == 2 and reduce_sizes[-1] == 32 and isinstance(reduce_sizes[0], int) and reduce_sizes[0] % 16 == 0:
      k.apply_opt(Opt(OptOps.SPLIT, k.axes_of(reduce=True)[0], (16, AxisType.LOCAL)))
      k.apply_opt(Opt(OptOps.SPLIT, k.upcastable_dims(reduce=True)[2], (8, AxisType.UPCAST)))
      return True

    # IQ4_XS: the dequantized reduction is [blocks, 4, 2, 2, 16]. Use a 2-D 32-128 lane group
    # and expose the adjacent two-way reduction. More unrolling has no runtime benefit here and
    # materially increases cold compile time. Cap the first group dimension so larger/unusual
    # reductions retain the conservative generic schedule.
    if len(reduce_sizes) == 5 and reduce_sizes[1:] == (4, 2, 2, 16) and isinstance(reduce_sizes[0], int) and \
       2 <= reduce_sizes[0] <= 32 and output_size % 4 == 0:
      for _ in range(2):
        localizable = [a for a in k.axes_of(reduce=True) if k.rngs[a].axis_type in (AxisType.GLOBAL, AxisType.WEAK)]
        if not localizable: return False
        k.apply_opt(Opt(OptOps.SPLIT, localizable[0], (0, AxisType.LOCAL)))
      k.apply_opt(Opt(OptOps.SPLIT, k.upcastable_dims(reduce=True)[3], (0, AxisType.UPCAST)))
      return True
    return False

  def apply_tc_hand_opts(self, tk, rngs):
    apply_tc_hand_opts(tk, rngs)

  def coalesce_gate(self, uops): return isa_float4_coalesce(uops, self)
  def coalesce_mem_ok(self, f4, float4_safe, buf, value_dtype, u, uses, valid):
    return isa_float4_mem_ok(f4, float4_safe, buf, value_dtype, u, uses, valid)
  def coalesce_vec_lengths(self, buf, f4):
    # 16 halves = one WMMA A frag (2×B128); PACK identity-aliases the load.
    if buf.dtype == dtypes.half: return [16, 8, 4, 2]
    if buf.dtype == dtypes.uint8: return [16, 8, 4, 2]
    if buf.dtype == dtypes.float: return [4, 3, 2]
    return None

  def coalesce_vec_alignment(self, buf):
    # gfx11 wide global memory operations accept any dword-aligned address; they do not require
    # natural vector alignment (for example, a B128 byte load may begin at byte offset four).
    return max(4 // buf.dtype.itemsize, 1) if buf.addrspace is AddrSpace.GLOBAL else None

  def merge_memory_loads(self, sink:UOp) -> UOp: return merge_adjacent_uint32_loads(merge_adjacent_half_loads(sink))

  def prepare_pre_regalloc(self, lst:list[UOp]) -> tuple[list[UOp], dict]:
    lst = _hoist_kernargs(_protect_loop_invariant_fmac(lst))
    # Unroll lowers WMMA cin to zero PACKs + ADD into phi. Those PACKs must run each K
    # iteration: if they stay pre-loop, two-address WMMA keeps ACC across iters and the
    # phi ADDs double-count (test_tensor_cores_unroll_phi).
    # Do not sink packs for WMMAs whose A/B come from LDS — those zeros are one-shot ACC
    # inits for two-address accumulate across K (TC_LDS_AB). Unrelated LLOAD elsewhere
    # must not suppress the UNROLL sink.
    if (loop_i := next((i for i,u in enumerate(lst) if u.op is Ops.RANGE), None)) is not None:
      zero_acc = {u.src[0] for u in lst if u.op is Ops.INS and _iop(u) is AMDOps.WMMA and
                  _is_wmma_acc_reload_pack(u.src[0]) and not _wmma_ab_from_lds(u)}
      move_i = [i for i,u in enumerate(lst) if i < loop_i and u in zero_acc]
      if move_i:
        packs = [lst[i] for i in move_i]
        lst = [u for i,u in enumerate(lst) if i not in set(move_i)]
        loop_i = next(i for i,u in enumerate(lst) if u.op is Ops.RANGE)
        ins = loop_i + 1
        if ins < len(lst) and lst[ins].op is Ops.AFTER: ins += 1
        lst = lst[:ins] + packs + lst[ins:]
    inits, tiles, idx_map, buf_tiles = _wmma_acc_zero_inits(lst)
    if not inits: return lst, {}
    loop_i = next((i for i,u in enumerate(lst) if u.op is Ops.RANGE), 0)
    return lst[:loop_i] + inits + lst[loop_i:], {
      "wmma_acc_inits": {u.tag: u for u in inits},
      "wmma_acc_tiles": tiles,
      "wmma_acc_buf_tiles": buf_tiles,
      "wmma_acc_idx_map": idx_map,
    }

  def is_two_address(self, x:UOp) -> bool:
    if x.op is not Ops.INS: return False
    # REG_STORE redefines the promoted REG vreg in place (loop-carried / RMW).
    if _iop(x) in (AMDOps.WMMA, AMDOps.FMAC, AMDOps.FMA_MIX_F32, AMDOps.REG_STORE): return True
    # PACK_F16(half×16 LOAD) — coalesce onto the load (hand FA/FB).
    return _iop(x) is AMDOps.PACK_F16 and _pack_f16_is_vec_load(x) and len(x.src) == 1
  def redefines_register(self, x:UOp) -> bool: return self.is_two_address(x) or _iop(x) is AMDOps.PACK
  def is_rematerialized_constant(self, x:UOp) -> bool: return self._const_mov_leaf(x) is not None
  def is_promoted_reg_redef(self, x:UOp) -> bool:
    """ALU sinked onto a promoted REG slot: allow mid-live redef without uses[0] coalesce.

    is_two_address would wrongly pin dest to src0 (MAX/MUL are not RMW). Default ON with
    AMD_REG_STORE_SINK_ALU (see _sink_alu_into_reg_store).
    """
    if x.op is not Ops.INS: return False
    if _iop(x) not in _REG_STORE_SINK_ALU_OPS: return False
    return bool(isinstance(x.tag, tuple) and x.tag and isinstance(x.tag[0], Register) and
                x.tag[0].name.startswith("reg"))
  def _vopd_pair_op(self, x:UOp) -> bool:
    """Ops whose dests we even/odd-bias for VOPD.

    Always on: bias float MOV of const/±inf for dual_mov (was AMD_VOPD_PAIR_CONST=1).
    PAIR_ALLOC / PAIR_FMAC / PAIR_PARK / PAIR_ADD removed (regress/NaN).
    """
    if x.op is not Ops.INS or x.dtype is not dtypes.float32: return False
    if _iop(x) is not AMDOps.MOV or not x.src: return False
    if _unwrap_const(x.src[0]) is not None: return True
    return self._const_mov_leaf(x) is not None

  def spill(self, disp:UOp, x:UOp, slots:int|None=None) -> UOp:
    # REG_STORE is void; spill the written value, not the void node.
    if x.op is Ops.INS and _iop(x) is AMDOps.REG_STORE: x = x.src[1]
    # Optional slots: spill-on-evict binds a phys with a scalar dtype (1-slot _reg_slots);
    # multi-VGPR victims need an explicit width matching register_slots / FILL.
    if slots is not None and slots > 1:
      return UOp(Ops.INS, src=(disp, x, _tconst(slots, dtypes.int32).rtag()), arg=(AMDOps.SPILL, dtypes.void))
    return UOp(Ops.INS, src=(disp, x), arg=(AMDOps.SPILL, dtypes.void))
  def loop_end(self, x:UOp) -> UOp|None:
    if x.op is Ops.INS and _iop(x) is AMDOps.LOOP_CMP: return x.src[2] if len(x.src) == 3 else x.src[3]
    return super().loop_end(x)
  def prefer_phys(self, x:UOp, src_phys:list) -> Register|None:
    # AMD_MAX_ALIAS_SRC0 removed (asm-positive but fair checksum −3.92).
    # PACK_F16 onto a contiguous src fragment — kills PACK←src evacuate MOVs into WMMA A/B
    # (DIRECT copies v86–93→v104–111; HIP keeps in place). Alias always on (LLOAD + contig).
    if x.op is Ops.INS and _iop(x) is AMDOps.PACK_F16 and src_phys and all(p is not None for p in src_phys) and \
       isinstance(x.tag, tuple):
      base = src_phys[0]
      off, ok = 0, True
      for src, p in zip(x.src, src_phys):
        n = _reg_slots(src)
        if p.index != base.index + off:
          ok = False
          break
        off += n
      if ok and off == _reg_slots(x) and base in x.tag[0].cons: return base
      # HALF2_GLOBAL: even-lane EXTRACT dests hold full half2 words — alias PACK onto those.
      if not _pack_f16_is_vec_load(x) and getenv("AMD_FLASH_ACC_SMALL", 0) and len(x.src) >= 2 and len(x.src) % 2 == 0:
        pairs = list(zip(x.src[::2], x.src[1::2]))
        if all(_pack_f16_half2_load(lo, hi) is not None for lo, hi in pairs):
          lo_phys = [src_phys[i] for i in range(0, len(src_phys), 2)]
          base = lo_phys[0]
          if all(p.index == base.index + i for i, p in enumerate(lo_phys)) and base in x.tag[0].cons:
            return base
    # WHERE/cndmask: alias onto true VGPR only when false arm is a literal (mask→0/-inf).
    # Unrestricted WHERE_ALIAS removed (quant / serial checksum breaks).
    if x.op is Ops.INS and _iop(x) is AMDOps.WHERE and x.dtype is dtypes.float32 and len(x.src) > 2 and _unwrap_const(x.src[2]) is not None:
      if len(src_phys) > 1 and src_phys[1] is not None and isinstance(x.tag, tuple):
        if src_phys[1] in x.tag[0].cons: return src_phys[1]
    # Redundant u32→u16 CAST after &0xffff / >>16: alias onto src (no mov).
    if x.op is Ops.INS and _iop(x) is AMDOps.CAST and x.dtype in dtypes.uints and x.src and \
       x.src[0].dtype in dtypes.uints and x.dtype.itemsize < x.src[0].dtype.itemsize and \
       _u32_high_bits_clear(x.src[0], x.dtype.itemsize * 8):
      if src_phys and src_phys[0] is not None and isinstance(x.tag, tuple) and src_phys[0] in x.tag[0].cons:
        return src_phys[0]
    # EXTRACT from a multi-VGPR value → alias onto its source lane. Besides WMMA
    # float stores, packed quantized byte loads use uint32 lanes exactly once.
    # Also IQ4 LUT fill: u32×4 nontemporal LOAD → LSTORE (needs alias for ds_store_b128 fold).
    if x.op is Ops.INS and _iop(x) is AMDOps.MOV and x.src and x.src[0].op is Ops.INS and \
       x.dtype is x.src[0].dtype and isinstance(x.tag, tuple) and src_phys and src_phys[0] is not None and \
       src_phys[0] in x.tag[0].cons:
      src_op = _iop(x.src[0])
      # MOV(EXTRACT): coalesce onto EXTRACT phys (flash ACC scale). Default ON.
      if src_op is AMDOps.EXTRACT:
        return src_phys[0]
      # MOV(MOV(EXTRACT)): only when the inner MOV is already an EXTRACT copy — not general MOV←MOV.
      if src_op is AMDOps.MOV and x.src[0].src and \
         x.src[0].src[0].op is Ops.INS and _iop(x.src[0].src[0]) is AMDOps.EXTRACT:
        return src_phys[0]
      # AMD_MOV_ALIAS_MOV / AMD_MOV_ALIAS_CONST_MOV removed (NaN / serial corpus).
      # MOV(ALU): same-dtype copy after MAX/MUL/EXP2/WHERE/FMAC/ADD — default ON (checksum OK).
      if src_op in (AMDOps.MAX, AMDOps.MUL, AMDOps.EXP2, AMDOps.WHERE, AMDOps.FMAC, AMDOps.ADD) and \
         True:
        return src_phys[0]
      # MOV(SWIZZLE|PERMLANE): park elim. Decode ship ON via VOPD_FMAC; full NO_PARK hung.
      if src_op in (AMDOps.SWIZZLE, AMDOps.PERMLANEX16) and getenv("AMD_VOPD_FMAC", 0):
        return src_phys[0]
    if x.op is not Ops.INS or _iop(x) is not AMDOps.EXTRACT: return None
    # HALF2_GLOBAL: even-lane f16 EXTRACT from global LOAD is a full half2 word — alias onto
    # the load lane so PACK can sit on the load VGPRs (kills MOV+evacuate; keeps load live
    # until PACK so the next b128 cannot reuse early).
    if x.dtype is dtypes.float16 and getenv("AMD_FLASH_ACC_SMALL", 0) and \
       x.src and x.src[0].op is Ops.INS and _iop(x.src[0]) is AMDOps.LOAD and \
       src_phys and src_phys[0] is not None and isinstance(x.tag, tuple):
      if (lane := _const_int(x.src[1])) is not None and lane % 2 == 0:
        want = src_phys[0].index + int(lane) // 2
        hit = next((r for r in x.tag[0].cons if r.index == want), None)
        if hit is not None: return hit
    packed_bytes = x.dtype is dtypes.uint32 and x.src and x.src[0].op is Ops.INS and _iop(x.src[0]) is AMDOps.LOAD and \
      _reg_slots(x.src[0]) == 4 and _is_byte_addr_load(x.src[0])
    packed_u32 = x.dtype in (dtypes.uint32, dtypes.int32) and x.src and x.src[0].op is Ops.INS and \
      _iop(x.src[0]) is AMDOps.LOAD and _reg_slots(x.src[0]) >= 4
    if x.dtype is not dtypes.float32 and not packed_bytes and not packed_u32: return None
    if not src_phys or src_phys[0] is None or not isinstance(x.tag, tuple): return None
    if (lane := _const_int(x.src[1])) is None: return None
    want = src_phys[0].index + int(lane)
    return next((r for r in x.tag[0].cons if r.index == want), None)
  def after_pre_regalloc(self, lst:list[UOp]) -> list[UOp]:
    """Pre-regalloc schedule tweaks: A/B VMEM overlap, then cast-before-store.

    1. Prefetch next wide A (B128) before PACK_A (within-K; default all N).
    2. Hoist next wide B between A U16 and A pack so B gets distinct VGPRs from live A dests.
    3. Prefetch next strided B U16 before current B pack so both tiles are in flight.
    4. Schedule each f32→f16 CAST immediately before its STORE — product-16 epilogue otherwise
       keeps 128 half temps live; regalloc spills them into live WMMA ACC (v126+) and clobbers
       unread lanes (half rows 62–63).
    5. AMD_D16_HI: keep each fused lo LOAD before its hi — post-regalloc-only reorder lets lo
       dest-as-addr reuse hi's still-live index VGPR (MMU on gfx1100).
    6. Hoist independent scalar VMEM reads in kernels without WMMA or wide global loads.
    7. Put a boundless-loop compare before any REG_STORE that mutates its old-value operand.
    """
    lst = _lower_fma_mix_f32(lst)
    lst = _schedule_fma_mixhi_pairs(lst)
    lst = _prefetch_a_after_packed_quant(lst) if True else lst
    lst = _prefetch_late_iq4_a_before_mix(lst) if True else lst
    lst = _prefetch_next_bu16_before_pack(_hoist_b_between_a_and_pack(lst))
    uses: dict[UOp, list[UOp]] = {}
    for u in lst:
      for src in u.src: uses.setdefault(src, []).append(u)
    store_cast: dict[UOp, UOp] = {}  # store -> cast
    for u in lst:
      if u.op is not Ops.INS or _iop(u) is not AMDOps.CAST: continue
      if u.dtype is not dtypes.float16 or not u.src or u.src[0].dtype is not dtypes.float32: continue
      us = uses.get(u, [])
      if len(us) == 1 and us[0].op is Ops.INS and _iop(us[0]) is AMDOps.STORE and len(us[0].src) > 2 and us[0].src[2] is u:
        store_cast[us[0]] = u
    if store_cast:
      skip = set(store_cast.values())
      out: list[UOp] = []
      for u in lst:
        if u in skip: continue
        if u in store_cast: out.append(store_cast[u])
        out.append(u)
      lst = out
    d16_hi_lo: dict[UOp, UOp] = {}  # AMD_D16_HI removed
    # _order_d16_lo_before_hi no-op under empty map
    lst = _hoist_gated_fmac_loads(lst)
    lst = _schedule_scalar_vmem(lst, d16_hi_lo)
    # After scalar VMEM (it otherwise reopens the weight→A gap). Default on; AMD_PREFETCH_Q6_A=0 opts out.
    lst = _prefetch_a_before_dequant_mix(lst) if True else lst
    # LLOAD hoist before phys assignment (emit-time hoist hits the same VGPR-reuse hazard as SLOAD batch).
    lst = _hoist_lloads_before_extracts(lst)
    # AMD_HOIST_PACK_AFTER_LLOAD / AMD_CLUSTER_PACK_LLOAD removed (checksum−3.375 / fair regress).
    lst = _schedule_lload_2addr_pairs(lst)
    lst = _schedule_swizzle_mov_batches(lst)
    lst = _schedule_fmac_add_pairs(lst)
    lst = _gap_fill_after_loads(lst)
    lst = _schedule_scratch_load_passes(lst)
    lst = _pack_aligned_const_scratch_sloads(lst)
    lst = _schedule_loop_cmps(lst)
    # Int MUL×(D+pad) → SHL+ADD before remat sees pure-addr (flash ACC_VGPR spills).
    lst = _strength_reduce_int_muls(lst)
    # AMD_FOLD_LSHL_ADD / AMD_PEEL_MOV_* removed (hang / NaN).
    # AMD_COLLAPSE_CONST_MOV / AMD_MOV_SINK_ALU removed (default OFF; prefer_phys covers MAX).
    lst = _sink_alu_into_reg_store(lst)
    # AMD_CLUSTER_PACK_LLOAD removed (fair checksum −3.375).
    # HALF2_GLOBAL: drop dead hi EXTRACTs + shrink packs to lo-only before phys assign.
    lst = _rewrite_half2_global_packs(lst)
    lst = _schedule_half2_lo_before_pack(lst)
    # Re-run FMAC↔ADD adjacency after gap-fill / peel / half2 (those undo early placements).
    # Keep default SCAN=64 — large SCAN (256+) wrong-sum / hang on decode (2026-09-25).
    # ADD↔ADD hoist removed (AMD_SCHEDULE_ADD_ADD): NaN root; dual_add not worth it.
    lst = _schedule_fmac_add_pairs(lst)
    return lst
  def _pure_addr(self, x:UOp) -> bool:
    if x.op in (Ops.CONST, Ops.SPECIAL): return True
    # SPECIAL(uint)→MOV→CAST(int)→SHL/ADD. Without CAST, AMD_REMAT_ADDR never sees pure addr.
    # Deep remat of nested ADD trees still MMUs on flash (gfx1100); leaf-only via AMD_REMAT_ADDR
    # (default off outside TC_LDS). AMD_REMAT_ADDR_DEEP=1 is unsafe until remat src binding is fixed.
    # Do NOT treat MUL as pure-addr: depth-1 remat of stride MULs wrongs flash numerics;
    # depth-2 remat hangs (MMU) on gfx1100.
    if x.op is Ops.CAST and x.dtype in dtypes.ints and x.src:
      return self._pure_addr(x.src[0])
    if x.op is not Ops.INS or x.dtype not in (dtypes.int32, dtypes.uint32, dtypes.uint): return False
    if _iop(x) is AMDOps.MOV and x.src: return self._pure_addr(x.src[0])
    if _iop(x) in (AMDOps.ADD, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.LSHL_ADD):
      return all(self._pure_addr(s) for s in x.src)
    return False
  def _addr_leaf(self, x:UOp) -> bool:
    if x.op in (Ops.CONST, Ops.SPECIAL): return True
    if x.op is Ops.CAST and x.dtype in dtypes.ints and x.src: return self._addr_leaf(x.src[0])
    # Include dtypes.uint — lidx SPECIAL often MOVs to uint before SHL/SHR (flash ACC_VGPR spills).
    if x.op is Ops.INS and x.dtype in (dtypes.int32, dtypes.uint32, dtypes.uint) and _iop(x) is AMDOps.MOV and x.src:
      return self._addr_leaf(x.src[0])
    return False
  def _addr_within(self, x:UOp, max_d:int) -> bool:
    # Pure-addr tree of depth ≤ max_d from leaves (CONST/SPECIAL/MOV). Depth-1 == old _addr_shallow.
    # Flash ACC_VGPR leftover spills are depth-2/3 ADD(SHL/AND) trees — DEEP=2 MMUs, this does not.
    if max_d < 0: return False
    if self._addr_leaf(x): return True
    if x.op is Ops.CAST and x.dtype in dtypes.ints and x.src: return self._addr_within(x.src[0], max_d)
    if x.op is Ops.INS and x.dtype in (dtypes.int32, dtypes.uint32, dtypes.uint) and _iop(x) is AMDOps.MOV and x.src:
      return self._addr_within(x.src[0], max_d)
    if x.op is Ops.INS and x.dtype in (dtypes.int32, dtypes.uint32, dtypes.uint) and \
       _iop(x) in (AMDOps.ADD, AMDOps.SHL, AMDOps.SHR, AMDOps.AND, AMDOps.OR, AMDOps.XOR, AMDOps.LSHL_ADD):
      return all(self._addr_within(s, max_d - 1) for s in x.src)
    return False
  def _addr_shallow(self, x:UOp) -> bool:
    # One ALU of leaves (e.g. SHL(MOV,C)). Depth≥2 MMUs on flash (same class as DEEP=2).
    return self._addr_within(x, 1)
  def _const_mov_leaf(self, x:UOp) -> UOp|None:
    """CONST or CAST(CONST), or MOV chain ending in those — free to remat (no live interval)."""
    # Bare CAST(CONST)/CONST: ACC-promote leftover spills (offsets 0/4/8) are Ops.CAST, not MOV.
    if x.op is Ops.CONST: return x
    if x.op is Ops.CAST and x.src and x.src[0].op is Ops.CONST: return x
    if x.op is not Ops.INS or _iop(x) is not AMDOps.MOV or not x.src: return None
    s0 = x.src[0]
    # Peel redundant MOV(MOV(...)) so flash zero/-inf inits remat (gated remat missed these).
    while s0.op is Ops.INS and _iop(s0) is AMDOps.MOV and s0.src: s0 = s0.src[0]
    if s0.op is Ops.CONST: return s0
    if s0.op is Ops.CAST and s0.src and s0.src[0].op is Ops.CONST: return s0
    return None
  def rematerialize(self, x:UOp) -> bool:
    # Const remat before the INS-only gate: bare CAST(CONST) must remat under ACC promote.
    # INF-only always on (was AMD_REMAT_CONST_MOV + CONST_MOV_INF). Non-INF remat
    # fair-A/B regresses prefill ~10–15µs — leave banned.
    if (leaf:=self._const_mov_leaf(x)) is not None:
      v = leaf.arg if leaf.op is Ops.CONST else (leaf.src[0].arg if leaf.op is Ops.CAST else None)
      if isinstance(v, float) and math.isinf(v): return True
    if x.op is not Ops.INS: return False
    # Under TC_LDS_AB: remat LDS half EXTRACTs (and LLOAD bases if ALLOW_UPCAST16).
    # Address remat defaults ON under LDS — EXTRACT-only leaves addr spills / wrong mock.
    # AMD_REMAT_ADDR=0 opts out. TC_LDS_AB default 0 (match apply_tc / flash ship).
    if getenv("TC_LDS_AB", 0):
      if (_iop(x) is AMDOps.EXTRACT and x.dtype is dtypes.half and x.src and
          x.src[0].op is Ops.INS and _iop(x.src[0]) is AMDOps.LLOAD):
        return True
      if allow_upcast16() and _iop(x) is AMDOps.LLOAD and x.dtype is dtypes.half:
        return True
    if not getenv("AMD_REMAT_ADDR", 1 if getenv("TC_LDS_AB", 0) else 0): return False
    # AMD_REMAT_LIDX_ADDR / AMD_REMAT_BIND_ADD removed (default OFF; never shipped).
    # Int leaf MOV under address remat (bool gates / index temps).
    if _iop(x) is AMDOps.MOV and x.src:
      s0 = x.src[0]
      if x.dtype in (dtypes.int32, dtypes.uint32, dtypes.bool, dtypes.uint) and self._addr_leaf(s0):
        return True
      return False
    if x.dtype not in (dtypes.int32, dtypes.uint32, dtypes.bool, dtypes.uint): return False
    if not self._pure_addr(x): return False
    # Depth: always one nested pure-addr when remat is on (was AMD_REMAT_ADDR_DEEP=1).
    # AMD_REMAT_ADDR_SHALLOW folded into depth>=1. depth>=2 historically MMU on flash — leave off.
    depth, shallow = 1, True
    def ok(s:UOp, d:int) -> bool:
      if shallow and self._addr_shallow(s): return True
      if self._addr_leaf(s): return True
      if d <= 0 or s.op is not Ops.INS or not self._pure_addr(s) or _iop(s) is AMDOps.MOV: return False
      return all(ok(t, d - 1) for t in s.src)
    return all(ok(s, depth) for s in x.src)
  def keep_remat(self, x:UOp) -> bool:
    # Const MOVs sticky while a phys is free (avoids MOV storms). INF remat is always-on.
    if self._const_mov_leaf(x) is not None: return True
    # Pure-addr remats under TC_LDS: without sticky, SHR/AND remat ~60× and SHL/ADD flood the loop.
    # AMD_REMAT_NO_STICKY_* / BIND / LIDX removed (hung/MMU/ninst flood).
    return x.op is Ops.INS and _iop(x) in (AMDOps.SHR, AMDOps.AND, AMDOps.SHL, AMDOps.ADD)
  def remat(self, x:UOp, reg:Register, src_regs:list[Register|None]) -> UOp:
    # Rebuild const MOV from the peeled leaf so MOV(MOV(0.0)) does not keep a live src VGPR.
    if (leaf := self._const_mov_leaf(x)) is not None:
      return UOp(Ops.INS, src=(leaf,), arg=(AMDOps.MOV, x.dtype), tag=(reg,))
    nsrc = [s if r is None else UOp(Ops.INS, arg=(AMDOps.MOV, s.dtype), tag=(r,)) for s, r in zip(x.src, src_regs)]
    return x.replace(src=tuple(nsrc), tag=(reg,))
  def bind(self, dtype, reg:Register) -> UOp: return UOp(Ops.INS, arg=(AMDOps.MOV, dtype), tag=(reg,))
  def stack_pointer(self) -> UOp: return UOp(Ops.INS, arg=(AMDOps.SCRATCH_BASE, dtypes.uint32))
  def register_slots(self, x:UOp, vreg:Register|None=None) -> int:
    if vreg is None: return 1
    if all(c.index < 256 for c in vreg.cons): return max(1, (x.dtype.itemsize + 3) // 4)
    return _reg_slots(x)
  def spill_size(self, x:UOp, vreg:Register) -> int:
    # Scalar scratch transport uses B32 even for bool/int8/int16 SGPR values.
    return max(4, x.dtype.itemsize) if all(c.index < 256 for c in vreg.cons) else super().spill_size(x, vreg)
  def copy(self, x:UOp, reg): return UOp(Ops.INS, src=(x,), arg=(AMDOps.MOV, x.dtype), tag=(reg,))
  def fill(self, disp:UOp, x:UOp, reg) -> UOp:
    return UOp(Ops.INS, src=(disp, _tconst(_reg_slots(x), dtypes.int32).rtag()), arg=(AMDOps.FILL, x.dtype), tag=(reg,))

  def asm_str(self, uops:list[UOp], function_name:str) -> str:
    ret = [f".{function_name}:"]
    for u in uops:
      if u.op is not Ops.INS: continue
      if _iop(u) is AMDOps.LABEL: ret.append(f"{u.tag}:")
      elif _iop(u) in (AMDOps.BRANCH, AMDOps.CBRANCH_SCC1, AMDOps.CBRANCH_VCCNZ): ret.append(f"  {_iop(u).name.lower()} {u.tag}")
      else: ret.append(f"  {_iop(u).name.lower()} " + ", ".join(str(greg(s) or s.arg) for s in u.src))
    return "\n".join(ret)

  def render(self, uops:list[UOp]) -> str:
    return self.asm_str(uops, "kernel")
  def _insts_for_uop(self, u:UOp): return insts_for_uop(u)
  def _insts_from_linear(self, lin:UOp): return insts_from_linear(lin)

  def asm(self, prg:UOp, lin:UOp) -> bytes:
    from tinygrad.renderer.amd.elf import assemble_linear
    insts = self._insts_from_linear(lin)
    insts.append(r3.s_endpgm())
    nlin = lin.replace(src=tuple(UOp(Ops.INS, arg=(i, dtypes.void)) for i in insts))
    return assemble_linear(prg, nlin, self.target.arch)

  def supported_dtypes(self):
    return {dtypes.bool, dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.float16, dtypes.float32}

# Gabriel/x86-style aliases
RDNA3Renderer = AMDRenderer
RDNA3Ops = AMDOps
