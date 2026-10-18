import unittest
from unittest.mock import Mock
from tinygrad.device import Allocator, Buffer, BufferStorage, Device
from tinygrad.helpers import Context

class TestBuffer(unittest.TestCase):
  def test_initial_values_survive_staging_reuse(self):
    for size in (4, 64, 4096):
      with self.subTest(size=size):
        values = [bytes([i]) * size for i in range(16)]
        bufs = [Buffer(Device.DEFAULT, size, initial_value=value) for value in values]
        for buf, value in zip(bufs, values): self.assertEqual(bytes(buf.as_memoryview()), value)

  def test_host_view(self):
    b = Buffer("CPU", 16)
    v = b.view(4, 4)
    host = v.host
    host.view(fmt='H')[0] = 0x1234
    self.assertEqual(b.host.view(fmt='H')[2], 0x1234)
    self.assertEqual(v._buf, b._buf + 4)
    self.assertIs(v.host, host)
    self.assertIs(v.meta, b.meta)

  def test_mapping(self):
    b = Buffer("CPU", 8, initial_value=b"abcdefgh")
    self.assertEqual(b.get_buf("PYTHON"), b._buf)
    v = b.view(4, 2)
    mapped = v.get_storage("PYTHON")
    self.assertEqual(mapped.buf, b._buf + 2)
    self.assertEqual(bytes(mapped.host.mv), b"cdef")
    self.assertIs(mapped.host, v.host)
    self.assertIsNone(mapped.meta)
    self.assertIs(v.get_storage("PYTHON"), mapped)

  def test_view_reallocation(self):
    b = Buffer("CPU", 8)
    v = b.view(4, 2)
    old = v.get_storage("PYTHON")
    b.deallocate()
    b.allocate()
    self.assertFalse(v.is_allocated())
    v.host[:] = b"test"
    self.assertIsNot(v.get_storage("PYTHON"), old)
    self.assertEqual(bytes(v.get_storage("PYTHON").host.mv), b"test")

  def test_cache_owned_storage_only(self):
    for opaque in (None, memoryview(bytearray(8))):
      with self.subTest(imported=opaque is not None), Context(LRU=1):
        b = Buffer("PYTHON", 8, opaque=opaque)
        buf = b._buf
        b.deallocate()
        self.assertEqual(b._buf is buf, opaque is None)

class TestAllocatorCache(unittest.TestCase):
  def test_reuse_waits_for_mapped_devices(self):
    allocator = Allocator(Mock())
    peers = [Mock(), Mock()]
    storage = BufferStorage(object(), maps={peer:BufferStorage(object()) for peer in peers})
    with Context(LRU=1):
      allocator.free(storage, 4)
      for peer in peers: peer.synchronize.assert_not_called()
      self.assertIs(allocator.alloc(4), storage)
      for peer in peers:
        peer.synchronize.assert_called_once_with()
        peer.allocator._unmap.assert_not_called()

if __name__ == "__main__": unittest.main()
