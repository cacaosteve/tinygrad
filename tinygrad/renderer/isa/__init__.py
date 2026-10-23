from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from tinygrad.dtype import DType
from tinygrad.renderer import Renderer
from tinygrad.device import TinyELF
from tinygrad.uop.ops import PatternMatcher, UOp, Ops
from typing import Any

def consumer_map_from_toposort(lst:list[UOp]):
  ret: dict[UOp, dict[UOp, None]] = {}
  for u in lst:
    ret[u] = {}
    for s in u.src:
      if s in ret: ret[s][u] = None
  return ret

@dataclass(frozen=True)
class Register:
  name: str
  index: int
  _cons: tuple[Register, ...] = field(default_factory=tuple)
  size: int = 8
  _cons_hash: int = field(init=False, repr=False, compare=False)
  def __post_init__(self): object.__setattr__(self, "_cons_hash", hash(tuple(r.index for r in self._cons)))
  @property
  def cons(self): return self._cons or (self,)
  def __hash__(self): return hash((self.name, self.index, self.size, self._cons_hash))
  def __repr__(self): return self.name

class IselContext:
  def __init__(self, sink:UOp):
    self.uses = consumer_map_from_toposort(list(sink.toposort()))
    self.reg_n = itertools.count()
    self.scratch: dict = {}
    def arg_key(u:UOp):
      if u.op is Ops.SPECIAL: return (2, u.arg)
      return (0, u.arg.slot) if u.arg.addrspace is not None else (1, u.expr)
    self.func_args = sorted([u for u in self.uses if u.op in {Ops.PARAM, Ops.SPECIAL}], key=arg_key)
    params = sorted([u for u in sink.toposort() if u.op is Ops.PARAM], key=lambda u: u.arg.slot)
    self.arg_offsets = {u: o for u, (o, _) in zip(params, TinyELF.iter_sig(tuple((None, u.addrspace, u.dtype, ()) for u in params)))}

  def vreg(self, cons:tuple[Register, ...]|Register, size:int=0):
    cons = cons if isinstance(cons, tuple) else (cons,)
    return Register(f"v{next(self.reg_n)}", 0, cons, size or cons[0].size)

# the registers an INS defines follow its opcode and dtype in its arg
def regs(u:UOp) -> tuple[Register, ...]: return u.arg[2:] if u.op is Ops.INS else ()
def with_regs(u:UOp, r:tuple[Register, ...]) -> UOp: return u.replace(arg=u.arg[:2]+r)

def rdef(u:UOp):
  if u.op in {Ops.AFTER, Ops.BITCAST}: return rdef(u.src[0])
  return r[0] if (r:=regs(u)) else None

greg = rdef

@dataclass
class PreRegAllocContext:
  uops: list[UOp]|None = None
  regalloc_i: int = 0
  lock: UOp|None = None
  clobbered: set[UOp] = field(default_factory=set)
  scratch: dict = field(default_factory=dict)

class LinearContext:
  def __init__(self, ren:ISARenderer):
    self.ren, self.stack_size = ren, 0
  def insert_linearized(self, uops:list[UOp]) -> list[UOp]: return uops
  def assign_spill_slot(self, r:Register, u:UOp) -> Any: raise NotImplementedError("arch specific")
  def register_buffers(self, uops:list[UOp]) -> set[UOp]: return set()
  def register_order(self, u:UOp, cons:tuple[Register, ...]) -> tuple[Register, ...]: return cons
  def allocated(self, u:UOp, reg:Register) -> None: pass

class ISARenderer(Renderer):
  float4_dtypes: tuple[DType, ...]|None = None
  preferred_reduce_group: int|None = None
  preferred_complex_matvec_group: int|None = None
  local_prod_max: int|None = None
  wide_regalloc: bool = False
  pre_isel_matcher: PatternMatcher
  isel_matcher: PatternMatcher
  pre_regalloc_matcher: PatternMatcher
  post_regalloc_matcher: PatternMatcher
  linear_ctx_type: type = LinearContext

  def prepare_pre_regalloc(self, lst:list[UOp]) -> tuple[list[UOp], dict]: return lst, {}
  def is_two_address(self, x:UOp) -> bool: return False
  def redefines_register(self, x:UOp) -> bool: return self.is_two_address(x)
  def is_promoted_reg_redef(self, x:UOp) -> bool: return False
  def prefer_phys(self, x:UOp, src_phys:list) -> Register|None: return None
  def prefer_phys_needs_liveness(self, x:UOp) -> bool: return False
  def loop_end(self, x:UOp) -> UOp|None: return x.src[1] if x.op is Ops.END else None
  def rematerialize(self, x:UOp) -> bool: return False
  def is_rematerialized_constant(self, x:UOp) -> bool: return False
  def remat(self, x:UOp, reg:Register, src_regs:list[Register|None]) -> UOp: raise NotImplementedError("arch specific")
  def bind(self, dtype:DType, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  # If True, a rematerialized value stays in its phys reg across uses (no re-remat each time).
  def keep_remat(self, x:UOp) -> bool: return False
  def register_slots(self, x:UOp, v:Register|None=None) -> int: return 1
  def spill_size(self, x:UOp, v:Register) -> int:
    return 16 if v.cons[0].size == 16 else (8 if x.op is Ops.BUFFER else x.dtype.itemsize)
  def stack_pointer(self) -> UOp: raise NotImplementedError("arch specific")
  def copy(self, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def spill(self, spill_slot:Any, x:UOp) -> UOp: raise NotImplementedError("arch specific")
  def fill(self, spill_slot:Any, x:UOp, reg:Register) -> UOp: raise NotImplementedError("arch specific")
  def asm_str(self, uops:list[UOp], function_name:str) -> str: raise NotImplementedError("arch specific")
