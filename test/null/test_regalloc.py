import unittest

from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.isa import ISARenderer, LinearContext, Register
from tinygrad.renderer.isa.x86 import X86Ops, X86Renderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp


class TestRegalloc(unittest.TestCase):
  def test_allocation_policy_state_is_per_program(self):
    class PolicyContext(LinearContext):
      def __init__(self, renderer):
        super().__init__(renderer)
        self.allocations = []
      def register_order(self, u, cons):
        offset = len(self.allocations) % len(cons)
        return cons[offset:] + cons[:offset]
      def allocated(self, u, reg): self.allocations.append(reg)

    renderer = ISARenderer(Target("CPU", arch="x86_64"))
    renderer.wide_regalloc = True
    physical = tuple(Register(f"p{i}", i, size=4) for i in range(2))
    virtual = tuple(Register(f"v{i}", 0, physical, size=4) for i in range(2))
    uops = [UOp(Ops.INS, arg=("value", dtypes.float32), tag=(v,)) for v in virtual]
    for _ in range(2):
      context = PolicyContext(renderer)
      allocated = LinearScanRegallocContext(context, uops, renderer)
      self.assertEqual(context.allocations, list(physical))
      self.assertEqual([allocated.reals[i][v] for i,v in enumerate(virtual)], list(physical))

  def test_stack_dealloc_emits_before_sink(self):
    renderer = X86Renderer(Target("CPU", arch="x86_64"))
    out = UOp.placeholder((1,), dtypes.float32, 0)
    local = UOp.placeholder((1,), dtypes.float32, 1, addrspace=AddrSpace.LOCAL)
    idx = UOp.const(0, dtypes.int32)
    store_local = local.index(idx).store(UOp.const(3.0, dtypes.float32))
    store_out = out.index(idx).store(local.index(idx).load())
    sink = UOp.sink(store_local, store_out, arg=KernelInfo(name="x86_regalloc_stack_dealloc"))
    to_program_cache.clear()
    lin = to_program(sink, renderer).src[1].src

    sub = [i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg[0] is X86Ops.SUBi]
    add = [i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg[0] is X86Ops.ADDi]
    ret = next(i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg[0] is X86Ops.RET)
    self.assertEqual(len(sub), 1)
    self.assertEqual(len(add), 1)
    self.assertLess(sub[0], add[0])
    self.assertLess(add[0], ret)
    self.assertEqual(ret, len(lin)-1)


if __name__ == "__main__":
  unittest.main()
