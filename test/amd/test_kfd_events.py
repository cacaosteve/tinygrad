import contextlib, ctypes, struct, subprocess, sys, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_linear
from tinygrad.uop import Ops
from tinygrad.runtime.autogen import kfd
from tinygrad.runtime.autogen.am import pm4_nv
from tinygrad.runtime.ops_amd import AMDAllocator, AMDComputeQueue, AMDSDMAQueue, KFDIface
from tinygrad.runtime.support.hcq2 import timeline

class TestKFDEvents(unittest.TestCase):
  def test_gfx11_flushes_host_writes_before_fetching_indirect_buffer(self):
    base = UOp.placeholder((96,), dtypes.uint8, device="AMD")
    cmdbuf = base[16:80]
    for major in (9, 10, 11, 12):
      with self.subTest(major=major):
        q = object.__new__(AMDComputeQueue)
        q.dev, q.devs, q.queue, q.target = SimpleNamespace(host="CPU", compute_queue=object()), ("AMD",), "COMPUTE:0", (major, 0, 0)
        q.pm4, q.prof_bump, q.blob, q.patches = pm4_nv, lambda buf:buf, bytearray(b"body"), []
        q.nbio = SimpleNamespace(version=(7, 11), regBIF_BX_PF1_GPU_HDP_FLUSH_REQ=SimpleNamespace(addr=(100,)),
                                 regBIF_BX_PF1_GPU_HDP_FLUSH_DONE=SimpleNamespace(addr=(104,)))
        with patch("tinygrad.runtime.ops_amd._queue_args", return_value=[]), \
             patch("tinygrad.runtime.ops_amd.patch", return_value=cmdbuf) as patcher, patch("tinygrad.runtime.ops_amd.amd_push"):
          q.submit(cmdbuf)
        buf, rows, blob = patcher.call_args.args
        words = struct.unpack(f"{len(blob)//4}I", blob)
        self.assertEqual(words[-4:], (pm4_nv.PACKET3(pm4_nv.PACKET3_INDIRECT_BUFFER, 2), 0, 0, 16 | pm4_nv.INDIRECT_BUFFER_VALID))
        self.assertEqual(buf.shape, (len(blob),))
        self.assertEqual(rows, [(len(blob)-12, base.getaddr(q.devs) + 16)])
        if major == 11:
          self.assertEqual(len(blob), 76)
          self.assertEqual(words[:7], (pm4_nv.PACKET3(pm4_nv.PACKET3_WAIT_REG_MEM, 5),
                                      pm4_nv.WAIT_REG_MEM_OPERATION(1) | pm4_nv.WAIT_REG_MEM_FUNCTION(5), 100, 104, 0xffffffff, 0xffffffff, 4))
          self.assertEqual(words[7], pm4_nv.PACKET3(pm4_nv.PACKET3_ACQUIRE_MEM, 6))
        else: self.assertEqual(len(blob), 16)

  def test_event_page_allocation_does_not_create_queues(self):
    iface = object.__new__(KFDIface)
    class Dev:
      device = "AMD"
      @property
      def has_copy_queue(self): raise AssertionError("event allocation recursively initialized the copy queue")
    dev = Dev()
    dev.iface, iface.dev, iface.alloc = iface, dev, Mock()
    page = UOp.placeholder((4096,), dtypes.uint64, device="AMD")
    events = [SimpleNamespace(), *[SimpleNamespace(event_slot_index=s, event_id=i) for s,i in ((7, 11), (8, 12), (9, 13))]]
    def allocate(device, size, options, preallocate):
      AMDAllocator(dev)._alloc(size, options)
      return Mock(meta=SimpleNamespace(handle=3))
    with patch("tinygrad.runtime.ops_amd.KFDIface", type("Events", (), {"kfd":None})), \
         patch("tinygrad.runtime.ops_amd.Buffer", side_effect=allocate), \
         patch.object(kfd, "AMDKFD_IOC_CREATE_EVENT", side_effect=events), \
         patch("tinygrad.runtime.ops_amd.UOp.from_buffer", return_value=page):
      self.assertEqual(iface.queue_event, (page[7:8], 11))
    iface.alloc.assert_called_once_with(0x8000, host=False, uncached=True, cpu_access=True)

  def test_event_id_and_mailbox_slot_are_distinct(self):
    iface = object.__new__(KFDIface)
    iface.dev = SimpleNamespace(device="AMD")
    page = UOp.placeholder((4096,), dtypes.uint64, device="AMD")
    events = [SimpleNamespace(event_slot_index=slot, event_id=eid) for slot,eid in ((7, 11), (8, 12), (9, 13))]
    with patch.object(KFDIface, "event_page", Mock(), create=True), \
         patch.object(kfd, "AMDKFD_IOC_CREATE_EVENT", side_effect=events) as create, \
         patch("tinygrad.runtime.ops_amd.UOp.from_buffer", return_value=page):
      self.assertEqual(iface.queue_event, (page[7:8], 11))
      self.assertEqual(iface.queue_event[0].getaddr("AMD").op, Ops.GETADDR)
      self.assertEqual([e.event_id for e in iface.queue_event_arr], [11, 12, 13])
      self.assertIs(iface.queue_event, iface.queue_event)
      self.assertEqual(create.call_count, 3)

  def test_timeline_wakes_registered_event(self):
    mailbox = UOp.placeholder((1,), dtypes.uint64, device="AMD")
    iface = object.__new__(KFDIface)
    iface.__dict__["queue_event"] = (mailbox, 11)
    value = UOp.const(19, dtypes.uint64)
    for queue_type in (AMDComputeQueue, AMDSDMAQueue):
      for interface in (iface, object()):
        for signal in (timeline("AMD"), UOp.placeholder((2,), dtypes.uint64, device="AMD").rtag("batch_slots"), timeline("CPU")):
          with self.subTest(queue=queue_type.__name__, kfd=interface is iface, tag=signal.tag, device=signal.device):
            q = object.__new__(queue_type)
            q.dev, q.devs, q.target = SimpleNamespace(iface=interface), ("AMD",), (11, 0, 0)
            q.pm4 = SimpleNamespace(data_sel__mec_release_mem__send_32_bit_low=1,
                                   int_sel__mec_release_mem__send_interrupt_after_write_confirm=2)
            q.sdma = SimpleNamespace(SDMA_OP_FENCE=5, SDMA_PKT_FENCE_HEADER_MTYPE=lambda x:x<<16, SDMA_OP_TRAP=6,
                                     SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT=lambda x:x)
            q.pred_exec, q.release_mem, q.q = lambda **kw:contextlib.nullcontext(), Mock(), Mock()
            q.signal(signal, value)
            wake = interface is iface and signal.tag == "timeline" and signal.device == "AMD"
            emit = q.release_mem if queue_type is AMDComputeQueue else q.q
            self.assertEqual(emit.call_count, 2 if wake else 1)
            if wake:
              args, kwargs = emit.call_args
              self.assertEqual(args[:2] if queue_type is AMDComputeQueue else args[1:3], (mailbox.getaddr(q.devs), 11))
              self.assertEqual(kwargs["ctxid"] if queue_type is AMDComputeQueue else args[-1], 11)

  @unittest.skipUnless(Device.DEFAULT.startswith("AMD"), "requires physical AMD KFD runtime")
  def test_hardware_compute_first_initialization(self):
    dev = Device[Device.DEFAULT]
    if not isinstance(dev.iface, KFDIface) or type(dev.iface).__name__.startswith("MOCK"):
      self.skipTest("requires physical KFD events")
    code = "from tinygrad import Tensor; assert Tensor.full((128,), 3, device='AMD').contiguous().realize().tolist() == [3]*128"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

  @unittest.skipUnless(Device.DEFAULT.startswith("AMD"), "requires physical AMD KFD runtime")
  def test_hardware_compute_and_copy_wake_kfd_wait(self):
    dev = Device[Device.DEFAULT]
    if not isinstance(dev.iface, KFDIface) or type(dev.iface).__name__.startswith("MOCK"):
      self.skipTest("requires physical KFD events")
    def wait(timeout):
      return kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(dev.iface.queue_event_arr), num_events=3,
                                      wait_for_all=0, timeout=timeout).wait_result
    x = Tensor([1, 2, 3, 4], device=dev.device).realize()
    run = TinyJit(lambda a:(a*3+1).realize())
    for _ in range(3):
      out = run(x)
      dev.synchronize()
    src = Buffer(dev.device, 64, initial_value=bytes(range(64)))
    dst = Buffer("CPU", 64, preallocate=True)
    copy = UOp(Ops.LINEAR, src=(UOp.from_buffer(dst, dtypes.uint8).store_call(UOp.from_buffer(src, dtypes.uint8)),))
    run_linear(copy, wait=True)
    for kind in ("compute", "copy"):
      for repeat in range(3):
        with self.subTest(kind=kind, repeat=repeat):
          # Consume stale auto-reset events before submitting the operation being tested.
          self.assertIn(wait(0), (kfd.KFD_IOC_WAIT_RESULT_COMPLETE, kfd.KFD_IOC_WAIT_RESULT_TIMEOUT))
          self.assertEqual(wait(0), kfd.KFD_IOC_WAIT_RESULT_TIMEOUT)
          if kind == "compute": out = run(x)
          else: run_linear(copy)
          self.assertEqual(wait(2000), kfd.KFD_IOC_WAIT_RESULT_COMPLETE)
          dev.synchronize()
    self.assertEqual(out.tolist(), [4, 7, 10, 13])
    self.assertEqual(bytes(dst.as_memoryview()), bytes(range(64)))

if __name__ == "__main__": unittest.main()
