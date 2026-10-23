import multiprocessing, multiprocessing.util, atexit, signal, sys, os, threading, contextlib, subprocess, importlib
from multiprocessing.context import SpawnContext, SpawnProcess
from multiprocessing.pool import Pool
from tinygrad.helpers import Context, getenv, PARALLEL

# generic pool of worker processes for parallel compilation, shared by kernel lowering and BEAM search

# workers should not open devices and should ignore ctrl c and should not launch VIZ
def _init_worker():
  Context(ALLOW_DEVICE_USAGE=0, VIZ=0, TRACK_MATCH_STATS=0).__enter__()
  signal.signal(signal.SIGINT, signal.SIG_IGN)

# spawn reimports the user's __main__ before _init_worker, replaying top-level code and potentially creating pools recursively.
# A spec named "__main__" makes multiprocessing.spawn skip that import.
_spawn_lock = threading.Lock()
@contextlib.contextmanager
def _without_main():
  main = sys.modules["__main__"]
  with _spawn_lock:
    saved = main.__spec__
    try:
      main.__spec__ = importlib.machinery.ModuleSpec("__main__", None)
      yield
    finally: main.__spec__ = saved

# spawn execs with the live environ, other threads changing os.environ (pytest) can break it: https://github.com/python/cpython/issues/157613
def _spawnv_passfds(path, args, passfds):
  (p:=subprocess.Popen(args, executable=path, pass_fds=passfds, env=getattr(os.environ, "_data").copy())).returncode = 0 # multiprocessing reaps it
  return p.pid
multiprocessing.util.spawnv_passfds = _spawnv_passfds

class _WorkerProcess(SpawnProcess):
  @staticmethod
  def _Popen(process_obj):
    with _without_main(): return SpawnProcess._Popen(process_obj)

class _WorkerContext(SpawnContext): Process = _WorkerProcess

worker_pool: Pool|None = None
worker_pool_env: tuple[tuple[str, str|None], ...]|None = None
def get_worker_pool(compile_env:dict[str, str|None]|None=None) -> Pool|None:
  global worker_pool, worker_pool_env
  if multiprocessing.current_process().daemon or PARALLEL == 0: return None
  env_key = tuple(sorted((compile_env or {}).items())) if compile_env is not None else None
  if worker_pool is not None and worker_pool_env != env_key: terminate_worker_pool()
  if worker_pool is None:
    worker_pool = _WorkerContext().Pool(PARALLEL.value, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))
    worker_pool_env = env_key
    @atexit.register
    def close_pool(pool=worker_pool): pool.close()
  return worker_pool

def terminate_worker_pool():
  global worker_pool, worker_pool_env
  if worker_pool is not None: worker_pool.terminate()
  worker_pool, worker_pool_env = None, None
