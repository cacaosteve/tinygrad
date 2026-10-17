from dataclasses import dataclass
from tinygrad.dtype import DType
from tinygrad.helpers import getenv
from tinygrad.renderer.isa import Register
from tinygrad.renderer.amd.dsl import s, v
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
