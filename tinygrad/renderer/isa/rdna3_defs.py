from dataclasses import dataclass
import struct
from tinygrad.device import CompileError
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.isa import Register, greg
from tinygrad.renderer.amd.dsl import Reg, s, v
from tinygrad.uop import FastEnum, Ops, auto
from tinygrad.uop.ops import UOp

# RDNA3 ABI and physical register constraints shared by selection, scheduling,
# tensor-core staging, and emission. Keeping these in one leaf module lets the
# larger backend split without scheduler/emitter import cycles.
KERNARG_REG = s[0:1]
WGID = tuple(Register(f"s{i}", i, size=4) for i in range(2, 5))
LID = tuple(Register(f"v{i}", 256+i, size=4) for i in range(3))

# USER_SGPR=15 places WGID_X/Y/Z in s15:s17. SGPRs are allocated as even
# 64-bit pairs, so reserve both s14:s15 and s16:s17 from the general pool.
# Scratch (also excluded): s20/s21 permlane+SALU temps; s22:23 long-branch PC.
# Keep scratch in the low range — parking them at s102+ forces wavefront SGPR
# granule up to ~106 and tanks occupancy (IQ4 WMMA was 1.19× behind HIP).
SGPR = tuple(Register(f"s{i}", i, size=8) for i in range(6, 104, 2) if i not in (14, 16, 20, 22))
SGPR32 = tuple(Register(f"s{i}", i, size=4) for i in range(6, 102) if i not in (14, 15, 16, 17, 20, 21, 22, 23))
VGPR = tuple(Register(f"v{i}", 256+i, size=4) for i in range(5, 254))
# AMD_VGPR_CAP lab path removed (never shipped; occupancy experiments via fork only).

# Accumulator tuples use a reserved VGPR band to avoid spilling large WMMA values.
@dataclass(frozen=True)
class RegisterPools:
  acc: tuple[Register, ...] = VGPR
  quant_acc: tuple[Register, ...] = VGPR
  load: tuple[Register, ...] = VGPR[:118]
  pack: tuple[Register, ...] = VGPR[185:244]
  pack_upcast: tuple[Register, ...] = VGPR[59:121]
  load_upcast: tuple[Register, ...] = VGPR[:59]

def register_pools(acc_base:int=121, acc_n:int|None=None) -> RegisterPools:
  if acc_base < 0: return RegisterPools()
  count = max(1, acc_n if acc_n is not None else (48 if acc_base >= 140 else 128))
  base = max(89, min(acc_base, len(VGPR) - count))
  end = base + min(count, len(VGPR) - base)
  pack_base = max(121, base - 60) if base >= 201 else max(89, base - 50)
  return RegisterPools(acc=VGPR[base:end], quant_acc=VGPR[min(89, base):end], load=VGPR[:min(135, base)],
                       pack=VGPR[pack_base:base], pack_upcast=VGPR[59:min(121, base)])

# v3/v4: per-instruction VGPR scratch; s20/s21: SALU/permlane; s22:23: long branch.
TMP_VDATA, TMP_VADDR = v[3], v[4]
TMP_SDATA0, TMP_SDATA1 = s[20], s[21]
TMP_BRANCH = s[22:23]

def allow_upcast16() -> bool:
  # Default ON: product-16 register path is correct with AMD_WMMA_ACC_BASE=121 (2026-09-27 HW).
  # Was wrong at ACC_BASE=-1 (ACC packs spilled → zero 16×16 tiles). Flash prefill forces 0.
  return bool(getenv("ALLOW_UPCAST16", 1))

def unwrap_const(x:UOp) -> UOp|None:
  while x.op in (Ops.CAST, Ops.BITCAST, Ops.NOOP) and len(x.src) == 1: x = x.src[0]
  return x if x.op is Ops.CONST else None

def const_value(x:UOp):
  return c.val if (c:=unwrap_const(x)) is not None else None

def tconst(value, dtype:DType, tag=None) -> UOp:
  """Typed bare constant for renderer-internal graphs, after the program spec boundary."""
  return UOp.cconst(value, dtype).rtag(tag)

class AMDOps(FastEnum):
  LABEL = auto()
  DEFINE = auto()
  SCRATCH_BASE = auto()
  SCRATCH_SIZE = auto()
  SCRATCH_ADDR = auto()
  KERNARG = auto()
  MOV = auto()
  PACK = auto()
  EXTRACT = auto()
  COLLECT = auto()
  ADD = auto()
  SUB = auto()
  MUL = auto()
  MULHI = auto()
  MULACC = auto()
  FMAC = auto()
  CAST = auto()
  RECIPROCAL = auto()
  EXP2 = auto()
  LOG2 = auto()
  SQRT = auto()
  TRUNC = auto()
  SIN = auto()
  MAX = auto()
  SHL = auto()
  SHR = auto()
  AND = auto()
  OR = auto()
  XOR = auto()
  CMPLT = auto()
  CMPNE = auto()
  CMPEQ = auto()
  WHERE = auto()
  SWHERE = auto()
  LOAD = auto()
  STORE = auto()
  ATOMIC_ADD = auto()
  LDS_BASE = auto()
  LLOAD = auto()
  LSTORE = auto()
  SLOAD = auto()
  SSTORE = auto()
  REG_STORE = auto()
  BARRIER = auto()
  FILL = auto()
  SPILL = auto()
  CMP_GE = auto()
  LOOP_CMP = auto()
  BRANCH = auto()
  CBRANCH_SCC1 = auto()
  CBRANCH_VCCNZ = auto()
  IF_MASK = auto()
  END_MASK = auto()
  PACK_F16 = auto()
  WMMA = auto()
  SWIZZLE = auto()
  PERMLANEX16 = auto()
  DOT4 = auto()
  BYTE_PERM = auto()
  BFE = auto()
  CVT_UBYTE_F32 = auto()
  FMA_TO_F16 = auto()
  PACKED_F16_MUL_TO_F16 = auto()
  LSHL_OR = auto()
  LSHL_ADD = auto()
  FMA_MIX_F32 = auto()  # srcs: (acc_f32, half, f32)

def iop(u:UOp):
  """Instruction selector carried by the dtype-less UOp INS argument."""
  return u.arg[0] if u.op is Ops.INS else None

def elem_count(u:UOp) -> int:
  """Logical vector element count. INS shape is scalar; width lives in the opcode/srcs."""
  if u.op is Ops.AFTER: return elem_count(u.src[0])
  if u.op is Ops.SHRINK and len(u.src) > 2 and (n:=const_value(u.src[2])) is not None: return int(n)
  if u.op is Ops.INS:
    if iop(u) is AMDOps.WMMA: return 8
    if iop(u) is AMDOps.PACK: return len(u.src)
    if iop(u) is AMDOps.PACK_F16:
      if pack_f16_is_vec_load(u): return sum(elem_count(s) for s in u.src)
      if pack_f16_is_half2_lo_only(u): return len(u.src) * 2
      return len(u.src)
    if iop(u) is AMDOps.EXTRACT: return 1
    if iop(u) in (AMDOps.LOAD, AMDOps.LLOAD, AMDOps.SLOAD):
      return int(n) if len(u.src) > 2 and (n:=const_value(u.src[2])) is not None else 1
    if iop(u) is AMDOps.MOV and u.src and u.src[0].op is not Ops.SPECIAL: return elem_count(u.src[0])
    return 1
  try: return u.max_numel()
  except (ValueError, RuntimeError): return 1

def reg_slots(u:UOp) -> int:
  """VGPR/SGPR slots occupied by u. PACK_F16 packs two halves per slot."""
  if u.op is Ops.AFTER: return reg_slots(u.src[0])
  if u.op is Ops.INS:
    if iop(u) is AMDOps.WMMA: return 8
    if iop(u) is AMDOps.PACK: return len(u.src)
    if iop(u) is AMDOps.PACK_F16:
      if pack_f16_is_vec_load(u): return sum(reg_slots(s) for s in u.src)
      if pack_f16_is_half2_lo_only(u): return len(u.src)
      return max(1, len(u.src) // 2)
    if iop(u) is AMDOps.EXTRACT: return 1
    if iop(u) is AMDOps.FILL:
      return int(n) if len(u.src) > 1 and (n:=const_value(u.src[1])) is not None else 1
    if iop(u) in (AMDOps.LOAD, AMDOps.LLOAD, AMDOps.SLOAD):
      return max(1, (u.dtype.itemsize * elem_count(u) + 3) // 4)
    if iop(u) in (AMDOps.STORE, AMDOps.LSTORE, AMDOps.SSTORE): return reg_slots(u.src[2])
    if iop(u) is AMDOps.SPILL: return reg_slots(u.src[1])
    if iop(u) is AMDOps.MOV and u.src and u.src[0].op is not Ops.SPECIAL: return reg_slots(u.src[0])
    return max(1, (u.dtype.itemsize + 3) // 4)
  return max(1, (u.dtype.itemsize * elem_count(u) + 3) // 4)

def reg_idxs(x:UOp) -> set[int]:
  if x.op is Ops.AFTER: return set().union(*(reg_idxs(s) for s in x.src))
  if not isinstance(greg(x), Register): return set()
  return set(range(greg(x).index, greg(x).index + reg_slots(x)))

def const_int(x:UOp) -> int|None:
  if (c:=unwrap_const(x)) is not None:
    try: return int(c.val)
    except (OverflowError, ValueError): return None
  if x.op is Ops.INS and iop(x) is AMDOps.MOV and x.src and (c:=unwrap_const(x.src[0])) is not None:
    try: return int(c.val)
    except (OverflowError, ValueError): return None
  return None

def mem_byte_off(u:UOp, src_i:int=3) -> int:
  return const_int(u.src[src_i]) or 0 if len(u.src) > src_i else 0

def lds_byte_off(u:UOp) -> int: return mem_byte_off(u, 3)
def load_count_src(n:int) -> UOp: return tconst(n, dtypes.int32).rtag()

def pack_f16_is_vec_load(u:UOp) -> bool:
  """PACK_F16(half×n LOAD/LLOAD, ...) — not scalar-LOAD B packs."""
  def is_vec_mem(s:UOp) -> bool:
    if s.op is Ops.LOAD and s.max_numel() >= 2: return True
    return s.op is Ops.INS and iop(s) in (AMDOps.LOAD, AMDOps.LLOAD) and elem_count(s) >= 2
  return bool(u.src) and all(is_vec_mem(s) for s in u.src)

def pack_f16_is_half2_lo_only(u:UOp) -> bool:
  """Each source is an even-lane EXTRACT carrying a full half2 word."""
  if not u.src or pack_f16_is_vec_load(u): return False
  for src in u.src:
    if not (src.op is Ops.INS and iop(src) is AMDOps.EXTRACT and src.dtype is dtypes.float16): return False
    if (lane := const_int(src.src[1])) is None or lane % 2: return False
  return True

def pack_f16_half2_load(lo:UOp, hi:UOp) -> tuple[UOp, int]|None:
  # LDS pairs rebuild one half2 word. Global pairs may only alias while flash keeps the load live.
  if not (lo.op is Ops.INS and iop(lo) is AMDOps.EXTRACT and hi.op is Ops.INS and iop(hi) is AMDOps.EXTRACT): return None
  if lo.src[0] is not hi.src[0]: return None
  base = lo.src[0]
  if base.op is not Ops.INS or iop(base) not in (AMDOps.LLOAD, AMDOps.LOAD): return None
  if iop(base) is AMDOps.LOAD and not getenv("AMD_FLASH_ACC_SMALL", 0): return None
  if not isinstance(greg(base), Register): return None
  lo_lane, hi_lane = const_int(lo.src[1]), const_int(hi.src[1])
  if lo_lane is None or hi_lane != lo_lane + 1 or lo_lane % 2: return None
  return base, lo_lane // 2

def reg_to_amd(reg:Register, sz:int=1) -> Reg:
  if reg.index >= 256:
    idx = reg.index - 256
    return v[idx] if sz == 1 else v[idx:idx+sz-1]
  return s[reg.index] if sz == 1 else s[reg.index:reg.index+sz-1]

def reg_lane(reg:Register, lane:int) -> Reg:
  return reg_to_amd(Register(f"{reg.name}_{lane}", reg.index + lane))

def src_operand(x:UOp):
  if x.op is Ops.AFTER: return src_operand(x.src[0])
  if (c:=unwrap_const(x)) is not None:
    if x.dtype is dtypes.float32: return float(c.val)
    if x.dtype is dtypes.float16: return struct.unpack("H", struct.pack("e", float(c.val)))[0]
    return int(c.val)
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg src {x}")
  if elem_count(x) > 1: return reg_lane(greg(x), 0)
  return reg_to_amd(greg(x), reg_slots(x))

def dst_operand(x:UOp) -> Reg:
  if not isinstance(greg(x), Register): raise CompileError(f"expected reg dst {x}")
  return reg_to_amd(greg(x), reg_slots(x))

def vgpr_num(reg:Reg) -> int|None:
  return reg.offset - 256 if isinstance(reg, Reg) and reg.sz == 1 and 256 <= reg.offset < 512 else None

def mov_imm_or_src(x:UOp):
  """Peel MOV/CAST to a scalar literal when possible; otherwise use the physical source."""
  cur = x
  for _ in range(6):
    if (c:=unwrap_const(cur)) is not None:
      try:
        if x.dtype in dtypes.floats or cur.dtype in dtypes.floats: return float(c.val)
        return int(c.val)
      except Exception: return src_operand(x)
    if cur.op is Ops.CAST and cur.src:
      cur = cur.src[0]
      continue
    if cur.op is Ops.INS and iop(cur) is AMDOps.MOV and cur.src and cur.src[0].op is not Ops.SPECIAL:
      cur = cur.src[0]
      continue
    break
  return src_operand(x)
