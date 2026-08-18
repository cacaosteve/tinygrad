import ctypes, unittest
from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, InstOp
from tinygrad.renderer.amd.dsl import NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.mockgpu.amd.sqtt import _mem_op, _op_name

def _run_kernel(instructions: list, lx=1, ly=1, lz=1, gx=1, gy=1, gz=1, args_ptr=0) -> bytes:
  from test.mockgpu.amd.emu import run_asm, sqtt_traces
  code = b''.join(inst.to_bytes() for inst in instructions)
  buf = (ctypes.c_char * len(code))(*code)
  lib = ctypes.addressof(buf)
  sqtt_traces.clear()
  with Context(PROFILE=1):
    run_asm(lib, len(code), gx, gy, gz, lx, ly, lz, args_ptr)
  assert len(sqtt_traces) == 1, f"expected 1 trace, got {len(sqtt_traces)}"
  return sqtt_traces.pop()

class TestSQTTEncoder(unittest.TestCase):
  def test_memory_instruction_classes(self):
    cases = [
      (global_load_b32(vdst=v[2], addr=v[0], saddr=s[2:3]), InstOp.SGMEM_RD_1),
      (global_load_b32(vdst=v[2], addr=v[0:1], saddr=NULL), InstOp.SGMEM_RD_2),
      (global_store_b32(addr=v[0], data=v[2], saddr=s[2:3]), InstOp.SGMEM_WR_2),
      (global_store_b64(addr=v[0], data=v[2:3], saddr=s[2:3]), InstOp.SGMEM_WR_3),
      (global_store_b32(addr=v[0:1], data=v[2], saddr=NULL), InstOp.SGMEM_WR_3),
      (global_store_b64(addr=v[0:1], data=v[2:3], saddr=NULL), InstOp.SGMEM_WR_4),
      (global_store_b96(addr=v[0:1], data=v[2:4], saddr=NULL), InstOp.SGMEM_WR_5),
      (global_store_b128(addr=v[0:1], data=v[2:5], saddr=NULL), InstOp.SGMEM_WR_6),
      (flat_load_b32(vdst=v[2], addr=v[0:1]), InstOp.FLAT_RD_2),
      (flat_store_b32(addr=v[0:1], data=v[2]), InstOp.FLAT_WR_3),
      (flat_store_b64(addr=v[0:1], data=v[2:3]), InstOp.FLAT_WR_4),
      (flat_store_b96(addr=v[0:1], data=v[2:4]), InstOp.FLAT_WR_5),
      (flat_store_b128(addr=v[0:1], data=v[2:5]), InstOp.FLAT_WR_6),
      (scratch_store_b128(addr=v[0], data=v[2:5], saddr=NULL), InstOp.FLAT_WR_6),
    ]
    for inst, expected in cases:
      with self.subTest(inst=inst): self.assertEqual(_mem_op(inst, _op_name(inst)), expected)

  def test_simple_salu(self):
    blob = _run_kernel([s_mov_b32(s[0], 42), s_endpgm()])
    packets = list(decode(blob))
    inst_pkts = [p for p in packets if isinstance(p, INST)]
    self.assertEqual(len(inst_pkts), 1)
    self.assertEqual(inst_pkts[0].op, InstOp.SALU)

  def test_valu_emits_valuinst(self):
    blob = _run_kernel([v_mov_b32_e32(v[0], 0), v_add_f32_e32(v[1], v[0], v[0]), s_endpgm()])
    packets = list(decode(blob))
    valu_pkts = [p for p in packets if isinstance(p, VALUINST)]
    self.assertEqual(len(valu_pkts), 2)
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)

  def test_waitcnt_emits_immediate(self):
    blob = _run_kernel([s_nop(simm16=0), s_waitcnt(simm16=0), s_endpgm()])
    imm_pkts = [p for p in decode(blob) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(imm_pkts), 2)

  def test_endpgm_skipped(self):
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)
    self.assertEqual(len([p for p in packets if isinstance(p, IMMEDIATE)]), 0)

  def test_wave_lifecycle(self):
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), sum(1 for p in packets if isinstance(p, WAVEEND)))

  def test_layout_header(self):
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertIsInstance(packets[0], LAYOUT_HEADER)
    self.assertEqual(packets[0].layout, 3)

  def test_blob_32byte_aligned(self):
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_endpgm()])
    self.assertEqual(len(blob) % 32, 0)

  def test_multiple_waves(self):
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()], lx=64)
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), 2)
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVEEND)), 2)
    self.assertEqual([p.wave for p in packets if isinstance(p, INST)], [0])
    starts = {(p.simd, p.wave): p._time for p in packets if isinstance(p, WAVESTART)}
    self.assertTrue(all(starts[(p.simd, p.wave)] < p._time for p in packets if isinstance(p, WAVEEND)))

  def test_branch_taken_and_not_taken(self):
    blob = _run_kernel([s_mov_b32(s[0], 2), s_sub_u32(s[0], s[0], 1), s_cmp_lg_u32(s[0], 0), s_cbranch_scc1(simm16=-3), s_endpgm()])
    inst_pkts = [p for p in decode(blob) if isinstance(p, INST)]
    ops = [p.op for p in inst_pkts]
    self.assertIn(InstOp.JUMP, ops)
    self.assertIn(InstOp.JUMP_NO, ops)

  def test_timestamps_monotonic(self):
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_mov_b32(s[2], 2), s_endpgm()])
    times = [p._time for p in decode(blob)]
    self.assertEqual(times, sorted(times))

  def test_no_trace_without_profile(self):
    from test.mockgpu.amd.emu import run_asm, sqtt_traces
    code = s_endpgm().to_bytes()
    buf = (ctypes.c_char * len(code))(*code)
    sqtt_traces.clear()
    with Context(PROFILE=0):
      run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, 1, 1, 1, 0)
    self.assertEqual(len(sqtt_traces), 0)

if __name__ == "__main__":
  unittest.main()
