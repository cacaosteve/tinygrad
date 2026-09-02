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
SGPR = tuple(Register(f"s{i}", i, size=8) for i in range(6, 104, 2) if i not in (14, 16, 102))
SGPR32 = tuple(Register(f"s{i}", i, size=4) for i in range(6, 102) if i not in (14, 15, 16, 17))
VGPR = tuple(Register(f"v{i}", 256+i, size=4) for i in range(5, 254))

WMMA_ACC_VGPR = VGPR[121:]
WMMA_ACC_QUANT_VGPR = VGPR[89:]
LLOAD_VGPR = VGPR[:118]
PACK_F16_VGPR = VGPR[185:244]
PACK_F16_VGPR_UP16 = VGPR[59:121]
LLOAD_VGPR_UP16 = VGPR[:59]

# v3/v4: per-instruction VGPR scratch; s102:103: long branch; s104:105: SALU compare scratch.
TMP_VDATA, TMP_VADDR = v[3], v[4]
TMP_BRANCH = s[102:103]
TMP_SDATA0, TMP_SDATA1 = s[104], s[105]

def allow_upcast16() -> bool:
  # Off under TC_LDS_AB — product 16 still spills there.
  return bool(getenv("ALLOW_UPCAST16", 0 if getenv("TC_LDS_AB", 0) else 1))

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
