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

# Soft partition: ACC packs live in a high VGPR band so LinearScan does not spill them.
# Default product ACC_BASE=121: gives Q6_K WMMA a wider quant-ACC pool (8 vs 15 spills,
# ~15% faster at 4096x4096x32) without regressing the 2048² half-matmul check. Flash prefill
# compile_env sets 145 + LLOAD_POOL=135; call
# refresh_vgpr_pools() at render entry so sticky env applies after import.
WMMA_ACC_VGPR: tuple[Register, ...] = VGPR
WMMA_ACC_QUANT_VGPR: tuple[Register, ...] = VGPR
LLOAD_VGPR: tuple[Register, ...] = VGPR[:118]
PACK_F16_VGPR: tuple[Register, ...] = VGPR[185:244]
PACK_F16_VGPR_UP16: tuple[Register, ...] = VGPR[59:121]
LLOAD_VGPR_UP16: tuple[Register, ...] = VGPR[:59]

def refresh_vgpr_pools() -> None:
  """Re-bind soft-partition pools from current env (flash sets ACC_BASE after import)."""
  global WMMA_ACC_VGPR, WMMA_ACC_QUANT_VGPR, LLOAD_VGPR, PACK_F16_VGPR, PACK_F16_VGPR_UP16, LLOAD_VGPR_UP16
  # Default 121 for product-16. Q4K/flash override
  # (-1 / 145+N=48). Setting only via os.environ after getenv misses cache — callers must
  # getenv.cache_clear().
  acc_base_env = int(getenv("AMD_WMMA_ACC_BASE", 121))
  if acc_base_env < 0:
    WMMA_ACC_VGPR = VGPR
    WMMA_ACC_QUANT_VGPR = VGPR
    LLOAD_VGPR = VGPR[:118]
    PACK_F16_VGPR = VGPR[185:244]
    PACK_F16_VGPR_UP16 = VGPR[59:121]
    LLOAD_VGPR_UP16 = VGPR[:59]
    return
  # Product-16 needs ACC_N=128; flash ACC_BASE≥140 uses N=48 (set explicitly too).
  acc_n_req = max(1, int(getenv("AMD_WMMA_ACC_N", 48 if acc_base_env >= 140 else 128)))
  acc_base = max(89, min(acc_base_env, len(VGPR) - acc_n_req))
  acc_n = min(acc_n_req, len(VGPR) - acc_base)
  WMMA_ACC_VGPR = VGPR[acc_base:acc_base + acc_n]
  WMMA_ACC_QUANT_VGPR = VGPR[min(89, acc_base):acc_base + acc_n]
  LLOAD_VGPR = VGPR[:min(135, acc_base)]  # AMD_LLOAD_POOL hardcoded 135 (flash ship)
  if acc_base >= 201:
    PACK_F16_VGPR = VGPR[max(121, acc_base - 60):acc_base]
  else:
    PACK_F16_VGPR = VGPR[max(89, acc_base - 50):acc_base]
  PACK_F16_VGPR_UP16 = VGPR[59:min(121, acc_base)]
  LLOAD_VGPR_UP16 = VGPR[:59]

refresh_vgpr_pools()

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
