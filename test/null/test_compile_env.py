import os, subprocess, sys, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.engine.realize import _compile_kernel
from tinygrad.codegen import to_program_mem_key
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer import Renderer
from tinygrad.helpers import Target
import tinygrad.engine.worker as worker
from tinygrad.helpers import getenv
from tinygrad.uop.ops import KernelInfo, UOp

class TestCompileEnv(unittest.TestCase):
  def test_rdna3_schedule_import_is_independent(self):
    script = "\n".join((
      "import sys",
      "from tinygrad.renderer.isa import rdna3_schedule as schedule",
      "from tinygrad.renderer.isa import rdna3_vopd as vopd",
      "from tinygrad.renderer.isa.rdna3_defs import AMDOps",
      "from tinygrad import dtypes",
      "from tinygrad.uop.ops import Ops, UOp",
      "assert 'tinygrad.renderer.isa.rdna3' not in sys.modules",
      "value = UOp.const(3, dtypes.int32)",
      "pack = UOp(Ops.INS, src=(value, value), arg=(AMDOps.PACK, dtypes.int32))",
      "assert schedule._reg_slots(pack) == 2",
      "assert schedule.strength_reduce_int_muls([pack]) == [pack]",
      "assert vopd.parallel_vmov([]) == []",
    ))
    subprocess.run([sys.executable, "-c", script], check=True)

  def test_rdna3_defaults_do_not_write_environment(self):
    from tinygrad.renderer.isa.rdna3 import AMDRenderer, _register_pools, apply_tc_hand_opts
    from tinygrad.renderer.isa import PreRegAllocContext
    from tinygrad.renderer.isa.rdna3_defs import VGPR
    try:
      with patch.dict(os.environ):
        for key in ("ALLOW_UPCAST16", "TC_UPCAST", "TC_LDS_AB", "AMD_WMMA_ACC_BASE", "AMD_WMMA_ACC_N"): os.environ.pop(key, None)
        getenv.cache_clear()
        before = dict(os.environ)
        AMDRenderer(Target("AMD", "AMD", "gfx1100"))
        with patch("tinygrad.renderer.isa.rdna3._wmma_has_fused_float_to_half_cast", return_value=False), \
             patch("tinygrad.renderer.isa.rdna3.apply_register_tc_hand_opts"):
          apply_tc_hand_opts(None, [])
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(_register_pools(PreRegAllocContext()).acc, VGPR[121:249])
        os.environ["ALLOW_UPCAST16"] = "0"
        getenv.cache_clear()
        self.assertEqual(_register_pools(PreRegAllocContext()).acc, VGPR)
    finally: getenv.cache_clear()

  def test_rdna3_register_pools_are_per_compile(self):
    from tinygrad.renderer.isa import PreRegAllocContext
    from tinygrad.renderer.isa.rdna3 import _register_pools
    from tinygrad.renderer.isa.rdna3_defs import VGPR
    first, second = PreRegAllocContext(), PreRegAllocContext()
    try:
      with patch.dict(os.environ, {"AMD_WMMA_ACC_BASE":"-1", "AMD_WMMA_ACC_N":"128"}):
        getenv.cache_clear()
        original = _register_pools(first)
      with patch.dict(os.environ, {"AMD_WMMA_ACC_BASE":"145", "AMD_WMMA_ACC_N":"48"}):
        getenv.cache_clear()
        changed = _register_pools(second)
        self.assertIs(_register_pools(first), original)
      self.assertEqual(original.acc, VGPR)
      self.assertEqual(changed.acc, VGPR[145:193])
    finally: getenv.cache_clear()

  def test_rdna3_import_preserves_llvm_renderer(self):
    script = "\n".join((
      "from tinygrad.renderer.llvmir import AMDLLVMRenderer",
      "before = dict(vars(AMDLLVMRenderer))",
      "import tinygrad.renderer.isa.rdna3",
      "assert dict(vars(AMDLLVMRenderer)) == before, 'RDNA3 import changed AMDLLVM policy'",
    ))
    subprocess.run([sys.executable, "-c", script], check=True)

  def test_worker_compile_env_is_scoped(self):
    key = "AMD_TEST_WORKER_ENV"
    previous = os.environ.get(key)
    try:
      os.environ[key] = "1"
      getenv.cache_clear()
      self.assertEqual(getenv(key, 0), 1)
      with patch("tinygrad.engine.realize.to_program", side_effect=lambda *_: getenv(key, 0)):
        self.assertEqual(_compile_kernel((0, (None, None), {}, {key:"2"})), (0, 2))
        self.assertEqual(_compile_kernel((0, (None, None), {}, {key:None})), (0, 0))
      self.assertEqual(os.environ.get(key), "1")
      self.assertEqual(getenv(key, 0), 1)
    finally:
      if previous is None: os.environ.pop(key, None)
      else: os.environ[key] = previous
      getenv.cache_clear()

  def test_program_memory_key_tracks_compile_environment(self):
    key = "AMD_TEST_PROGRAM_CACHE_ENV"
    with patch("tinygrad.codegen.to_program_key", return_value=("same-ast",)), \
         patch("tinygrad.codegen._program_cache_env_names", return_value=(key,)):
      with patch.dict(os.environ, {key:"first"}): first = to_program_mem_key(None, None)
      with patch.dict(os.environ, {key:"second"}): second = to_program_mem_key(None, None)
    self.assertNotEqual(first, second)

  def test_kernel_schedule_hints_survive_optimization(self):
    ast = UOp.sink(UOp.const(0), arg=KernelInfo(opts_to_apply=(), schedule_hints=("wmma_interleave",)))
    optimized = Scheduler(ast, Renderer(Target())).get_optimized_ast("hint_test")
    self.assertEqual(optimized.arg.schedule_hints, ("wmma_interleave",))

  def test_worker_pool_restarts_for_new_compile_environment(self):
    class FakePool:
      terminated = False
      def terminate(self): self.terminated = True

    old_pool, new_pool = FakePool(), FakePool()
    context = Mock()
    context.return_value.Pool.return_value = new_pool
    with patch.object(worker, "worker_pool", old_pool), patch.object(worker, "worker_pool_env", (("A", "old"),)), \
         patch.object(worker.multiprocessing, "current_process", return_value=SimpleNamespace(daemon=False)), \
         patch.object(worker, "PARALLEL", SimpleNamespace(value=1)), patch.object(worker, "_WorkerContext", context), \
         patch.object(worker.atexit, "register"):
      self.assertIs(worker.get_worker_pool({"A":"new"}), new_pool)
      self.assertTrue(old_pool.terminated)

if __name__ == "__main__": unittest.main()
