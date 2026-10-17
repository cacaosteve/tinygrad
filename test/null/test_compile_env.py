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
