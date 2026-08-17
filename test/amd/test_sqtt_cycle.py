#!/usr/bin/env python3
import ctypes, os, re, unittest
from dataclasses import dataclass
from typing import Iterable

from tinygrad.helpers import Context, Target, getenv
from tinygrad.renderer.amd.sqtt import (ALUEXEC, IMMEDIATE, IMMEDIATE_MASK, INST, INST_RDNA4, TS_DELTA_OR_MARK, VALUINST, VMEMEXEC, WAVEEND,
  WAVEEND_RDNA4, WAVERDY, WAVESTART, WAVESTART_RDNA4, InstOp, decode)
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.mockgpu.amd.emu import run_asm, sqtt_traces

_RSRC2_WITH_LDS = 0x19c | (1 << 15)

@dataclass(frozen=True)
class TraceCase:
  name: str
  instructions: list
  local_size: int = 1

@dataclass(frozen=True)
class NormalizedSQTTEvent:
  time: int
  kind: str
  wave: int|None
  pc: int|None
  op: str|None

  def without_time(self) -> tuple[str, int|None, int|None, str|None]:
    return (self.kind, self.wave, self.pc, self.op)

def _run_mock_blob(case: TraceCase) -> bytes:
  code = b''.join(inst.to_bytes() for inst in case.instructions)
  buf = (ctypes.c_char * len(code))(*code)
  sqtt_traces.clear()
  with Context(PROFILE=1):
    run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, case.local_size, 1, 1, 0, rsrc2=_RSRC2_WITH_LDS)
  assert len(sqtt_traces) == 1
  return sqtt_traces.pop()

def _run_mock_trace(case: TraceCase) -> list[NormalizedSQTTEvent]:
  return normalize_mock_trace(_run_mock_blob(case), case.instructions)

def _pc_map(instructions: list) -> dict[int, object]:
  pc, ret = 0, {}
  for inst in instructions:
    ret[pc] = inst
    pc += inst.size()
  return ret

def _trace_silent_inst(inst) -> bool:
  return isinstance(inst, SOPP) and inst.op == SOPPOp.S_DELAY_ALU

def _next_trace_pc(pc: int, pc_map: dict[int, object]) -> int:
  while pc in pc_map and _trace_silent_inst(pc_map[pc]): pc += pc_map[pc].size()
  return pc

def _advance_pc(pc: int, inst, pkt: INST|VALUINST|IMMEDIATE) -> int:
  if isinstance(pkt, INST) and pkt.op == InstOp.JUMP:
    x = getattr(inst, "simm16") & 0xffff
    return pc + inst.size() + (x - 0x10000 if x & 0x8000 else x) * 4
  return pc + inst.size()

def _rebase(events: Iterable[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  ret = list(events)
  base = min((e.time for e in ret), default=0)
  return [NormalizedSQTTEvent(e.time - base, e.kind, e.wave, e.pc, e.op) for e in ret]

def normalize_controlled_hardware_trace(blob: bytes, instructions: list) -> list[NormalizedSQTTEvent]:
  pc_map, wave_pc, events = _pc_map(instructions), {}, []
  for pkt in decode(blob):
    if isinstance(pkt, (WAVESTART, WAVESTART_RDNA4)):
      wave_pc[pkt.wave] = 0
      events.append(NormalizedSQTTEvent(pkt._time, "WAVESTART", pkt.wave, None, None))
    elif isinstance(pkt, (WAVEEND, WAVEEND_RDNA4)):
      events.append(NormalizedSQTTEvent(pkt._time, "WAVEEND", pkt.wave, wave_pc.pop(pkt.wave, None), None))
    elif isinstance(pkt, WAVERDY):
      for wave in range(16):
        if pkt.mask & (1 << wave): events.append(NormalizedSQTTEvent(pkt._time, "WAVERDY", wave, None, None))
    elif isinstance(pkt, (INST, INST_RDNA4)) and not pkt.op.name.startswith("OTHER_") and pkt.wave in wave_pc:
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, "INST", pkt.wave, pc, pkt.op.name))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt)
    elif isinstance(pkt, (VALUINST, IMMEDIATE)) and pkt.wave in wave_pc:
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, type(pkt).__name__.removesuffix("_MASK"), pkt.wave, pc, None))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt)
    elif isinstance(pkt, IMMEDIATE_MASK):
      for wave in range(16):
        if (pkt.mask & (1 << wave)) and wave in wave_pc:
          pc = _next_trace_pc(wave_pc[wave], pc_map)
          events.append(NormalizedSQTTEvent(pkt._time, "IMMEDIATE", wave, pc, None))
          wave_pc[wave] = _advance_pc(pc, pc_map[pc], pkt)
    elif isinstance(pkt, ALUEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "ALUEXEC", None, None, pkt.src.name))
    elif isinstance(pkt, VMEMEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "VMEMEXEC", None, None, pkt.src.name))
  return _rebase(events)

def normalize_mock_trace(blob: bytes, instructions: list) -> list[NormalizedSQTTEvent]:
  pc_map, wave_pc, events = _pc_map(instructions), {}, []
  for pkt in decode(blob):
    if isinstance(pkt, WAVESTART):
      wave_pc[pkt.wave] = 0
      events.append(NormalizedSQTTEvent(pkt._time, "WAVESTART", pkt.wave, None, None))
    elif isinstance(pkt, WAVEEND):
      events.append(NormalizedSQTTEvent(pkt._time, "WAVEEND", pkt.wave, wave_pc.pop(pkt.wave), None))
    elif isinstance(pkt, WAVERDY):
      for wave in range(16):
        if pkt.mask & (1 << wave): events.append(NormalizedSQTTEvent(pkt._time, "WAVERDY", wave, None, None))
    elif isinstance(pkt, INST) and not pkt.op.name.startswith("OTHER_"):
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, "INST", pkt.wave, pc, pkt.op.name))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt)
    elif isinstance(pkt, (VALUINST, IMMEDIATE)):
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, type(pkt).__name__, pkt.wave, pc, None))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt)
    elif isinstance(pkt, ALUEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "ALUEXEC", None, None, pkt.src.name))
    elif isinstance(pkt, VMEMEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "VMEMEXEC", None, None, pkt.src.name))
  return _rebase(events)

def _asm_source(case: TraceCase) -> str:
  code = b''.join(inst.to_bytes() for inst in case.instructions)
  byte_str = ', '.join(f'0x{x:02x}' for x in code)
  name = "sqtt_" + re.sub(r"[^0-9A-Za-z_]", "_", case.name)
  return f""".text
.globl {name}
.p2align 8
.type {name},@function
{name}:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
  .amdhsa_kernarg_size 0
  .amdhsa_group_segment_fixed_size 65536
  .amdhsa_private_segment_fixed_size 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 16
    .vgpr_count: 16
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

def _run_hardware_trace(case: TraceCase, require_instructions: bool=True) -> list[NormalizedSQTTEvent]:
  from tinygrad import Device
  from tinygrad.device import Compiled, ProfileDeviceEvent, ProfileProgramEvent, TinyELF
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  dev = Device["AMD"]
  dev.synchronize()
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, ProfileDeviceEvent)]

  compiler = HIPCompiler(dev.arch)  # type: ignore[attr-defined]
  name = "sqtt_" + re.sub(r"[^0-9A-Za-z_]", "_", case.name)
  lib = compiler.compile(_asm_source(case))
  prg = dev.runtime(TinyELF(lib, name, Target("AMD", arch=dev.arch), ()))

  best: list[NormalizedSQTTEvent] = []
  for _ in range(max(1, getenv("SQTT_CYCLE_HW_RETRIES", 4))):
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileDeviceEvent, ProfileProgramEvent))]
    prg(global_size=(1, 1, 1), local_size=(case.local_size, 1, 1), wait=True)
    dev.synchronize()
    programs = {e.tag: e for e in Compiled.profile_events if isinstance(e, ProfileProgramEvent) and e.tag is not None}
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent" and e.itrace and e.kern in programs]
    traces = [normalize_controlled_hardware_trace(e.blob, case.instructions) for e in sqtt_events]
    traces = [t for t in traces if any(e.kind == "WAVESTART" for e in t)]
    if not traces: continue
    ret = max(traces, key=len)
    if len(ret) > len(best): best = ret
    if not require_instructions or any(e.kind in {"INST", "VALUINST", "IMMEDIATE"} for e in ret): return ret
  if not best:
    raise unittest.SkipTest("hardware SQTT produced no wave lifecycle trace; run single-process with DEV=AMD VIZ=2 and SQTT_LIMIT_SE=1")
  raise unittest.SkipTest("hardware SQTT captured wave lifecycle but no instruction packets on this driver/interface")

def _first_mismatch(got: list[NormalizedSQTTEvent], expected: list[NormalizedSQTTEvent], *, check_time: bool, case: TraceCase) -> str|None:
  pc_map = _pc_map(case.instructions)
  for i, (g, e) in enumerate(zip(got, expected)):
    if (g != e if check_time else g.without_time() != e.without_time()):
      inst = pc_map.get(e.pc if e.pc is not None else g.pc)
      return f"event {i}\n  hardware: {g}\n  expected: {e}\n  decoded: {inst}"
  if len(got) != len(expected): return f"event count mismatch: hardware={len(got)} expected={len(expected)}\n  hardware={got}\n  expected={expected}"
  return None

def _instruction_events(events: Iterable[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  return [e for e in events if e.kind in {"INST", "VALUINST", "IMMEDIATE"}]

CASES: dict[str, tuple[TraceCase, list[NormalizedSQTTEvent]]] = {
  "salu_chain": (TraceCase("salu_chain", [s_mov_b32(s[0], 1), s_add_u32(s[1], s[0], 2), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "salu_saveexec": (TraceCase("salu_saveexec", [
    s_mov_b32(s[2], -1), s_and_saveexec_b32(s[0], s[2]), s_mov_b32(s[1], 1), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU_WR_EXEC"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(3, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(5, "WAVEEND", 0, 12, None),
  ]),
  "branch_unconditional": (TraceCase("branch_unconditional", [
    s_branch(simm16=1), s_mov_b32(s[0], 2), s_mov_b32(s[1], 1), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "JUMP"),
    NormalizedSQTTEvent(11, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(11, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(13, "WAVEEND", 0, 12, None),
  ]),
  "branch_loop": (TraceCase("branch_loop", [
    s_mov_b32(s[0], 2), s_sub_u32(s[0], s[0], 1), s_cmp_lg_u32(s[0], 0), s_cbranch_scc1(simm16=-3), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(3, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(12, "INST", 0, 12, "JUMP"),
    NormalizedSQTTEvent(22, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(22, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(23, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(23, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(33, "INST", 0, 12, "JUMP_NO"),
    NormalizedSQTTEvent(35, "WAVEEND", 0, 16, None),
  ]),
  "branch_vccz_taken": (TraceCase("branch_vccz_taken", [
    s_mov_b32(VCC_LO, 0), s_cbranch_vccz(simm16=1), s_mov_b32(s[1], 9), s_mov_b32(s[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(8, "INST", 0, 4, "JUMP"),
    NormalizedSQTTEvent(18, "INST", 0, 12, "SALU"), NormalizedSQTTEvent(18, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(20, "WAVEEND", 0, 16, None),
  ]),
  "branch_vccnz_not_taken": (TraceCase("branch_vccnz_not_taken", [
    s_mov_b32(VCC_LO, 0), s_cbranch_vccnz(simm16=1), s_mov_b32(s[1], 9), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(8, "INST", 0, 4, "JUMP_NO"),
    NormalizedSQTTEvent(11, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(11, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(13, "WAVEEND", 0, 12, None),
  ]),
  "branch_execz_taken": (TraceCase("branch_execz_taken", [
    s_mov_b32(EXEC_LO, 0), s_cbranch_execz(simm16=1), s_mov_b32(s[1], 9), s_mov_b32(s[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU_WR_EXEC"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(8, "INST", 0, 4, "JUMP"),
    NormalizedSQTTEvent(18, "INST", 0, 12, "SALU"), NormalizedSQTTEvent(18, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(20, "WAVEEND", 0, 16, None),
  ]),
  "branch_execnz_not_taken": (TraceCase("branch_execnz_not_taken", [
    s_mov_b32(EXEC_LO, 0), s_cbranch_execnz(simm16=1), s_mov_b32(s[1], 9), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU_WR_EXEC"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(8, "INST", 0, 4, "JUMP_NO"),
    NormalizedSQTTEvent(11, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(11, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(13, "WAVEEND", 0, 12, None),
  ]),
  "branch_vccz_after_vcmp_taken": (TraceCase("branch_vccz_after_vcmp_taken", [
    v_cmp_eq_u32_e32(1, v[0]), s_cbranch_vccz(simm16=1), s_mov_b32(s[1], 9), s_mov_b32(s[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(19, "INST", 0, 4, "JUMP"),
    NormalizedSQTTEvent(29, "INST", 0, 12, "SALU"), NormalizedSQTTEvent(29, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(31, "WAVEEND", 0, 16, None),
  ]),
  "branch_execz_after_vcmpx_taken": (TraceCase("branch_execz_after_vcmpx_taken", [
    v_cmpx_eq_u32_e32(1, v[0]), s_cbranch_execz(simm16=1), s_mov_b32(s[1], 9), s_mov_b32(s[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALU1_WR_EXEC"),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(19, "INST", 0, 4, "JUMP"),
    NormalizedSQTTEvent(29, "INST", 0, 12, "SALU"), NormalizedSQTTEvent(29, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(31, "WAVEEND", 0, 16, None),
  ]),
  "valu_independent": (TraceCase("valu_independent", [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 1), v_mov_b32_e32(v[2], 2), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(2, "VALUINST", 0, 4, None),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(3, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(4, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(5, "WAVEEND", 0, 12, None),
  ]),
  "valu_simple_dependency": (TraceCase("valu_simple_dependency", [
    v_add_f32_e32(v[1], v[0], v[0]), v_add_f32_e32(v[2], v[1], v[1]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(2, "VALUINST", 0, 4, None),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "valu_sopp_delay": (TraceCase("valu_sopp_delay", [v_mov_b32_e32(v[0], 0), s_nop(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(4, "IMMEDIATE", 0, 4, None),
    NormalizedSQTTEvent(6, "WAVEEND", 0, 8, None),
  ]),
  "valu_dependency": (TraceCase("valu_dependency", [v_lshlrev_b64(v[2:3], 2, v[0:1]), v_add_f32_e32(v[4], v[2], v[2]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_2"),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(3, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(4, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(5, "WAVEEND", 0, 12, None),
  ]),
  "valu_issue_limit": (TraceCase("valu_issue_limit", [v_lshlrev_b64(v[2:3], 2, v[0:1]), v_mov_b32_e32(v[6], 0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_2"),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(3, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(4, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(5, "WAVEEND", 0, 12, None),
  ]),
  "valu_long_latency": (TraceCase("valu_long_latency", [v_rcp_f32_e32(v[1], v[0]), v_mov_b32_e32(v[2], 0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUT_4"),
    NormalizedSQTTEvent(2, "VALUINST", 0, 4, None),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(6, "WAVEEND", 0, 8, None),
  ]),
  "valu_transcendent_dependency": (TraceCase("valu_transcendent_dependency", [
    v_rcp_f32_e32(v[1], v[0]), v_add_f32_e32(v[2], v[1], v[1]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUT_4"),
    NormalizedSQTTEvent(2, "VALUINST", 0, 4, None),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(6, "WAVEEND", 0, 8, None),
  ]),
  "valu_exec_write": (TraceCase("valu_exec_write", [v_cmpx_eq_u32_e32(v[0], v[0]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALU1_WR_EXEC"),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(3, "WAVEEND", 0, 4, None),
  ]),
  "valu_mad64": (TraceCase("valu_mad64", [v_mad_u64_u32(v[4:5], s[0], v[0], v[1], v[2:3]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_4"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(6, "WAVEEND", 0, 8, None),
  ]),
  "valu_mad64_issue_limit": (TraceCase("valu_mad64_issue_limit", [
    v_mad_u64_u32(v[4:5], s[0], v[0], v[1], v[2:3]), v_mov_b32_e32(v[6], 0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_4"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(5, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(6, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(7, "WAVEEND", 0, 12, None),
  ]),
  "valu_f64": (TraceCase("valu_f64", [v_add_f64(v[2:3], v[0:1], v[0:1]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_16"),
    NormalizedSQTTEvent(17, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(34, "WAVEEND", 0, 8, None),
  ]),
  "valu_f64_issue_limit": (TraceCase("valu_f64_issue_limit", [v_add_f64(v[2:3], v[0:1], v[0:1]), v_mov_b32_e32(v[6], 0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VALUB_16"),
    NormalizedSQTTEvent(17, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(33, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(34, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(35, "WAVEEND", 0, 12, None),
  ]),
  "wait_immediate": (TraceCase("wait_immediate", [s_nop(simm16=0), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "IMMEDIATE", 0, 0, None),
    NormalizedSQTTEvent(2, "IMMEDIATE", 0, 4, None),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "sopp_delay_alu_skip": (TraceCase("sopp_delay_alu_skip", [
    s_mov_b32(s[0], 1), s_delay_alu(simm16=0x3210), s_mov_b32(s[1], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 12, None),
  ]),
  "lds_roundtrip": (TraceCase("lds_roundtrip", [
    v_lshlrev_b32_e32(v[1], 2, v[0]), ds_store_b32(addr=v[1], data0=v[0]), ds_load_b32(vdst=v[2], addr=v[1]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(19, "INST", 0, 4, "LDS_WR_2"),
    NormalizedSQTTEvent(20, "INST", 0, 12, "LDS_RD"),
    NormalizedSQTTEvent(21, "VMEMEXEC", None, None, "LDS"), NormalizedSQTTEvent(21, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(55, "WAVEEND", 0, 20, None),
  ]),
  "lds_load_dependency": (TraceCase("lds_load_dependency", [ds_load_b32(vdst=v[2], addr=v[0]), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(8, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(33, "WAVEEND", 0, 12, None),
  ]),
  "lds_sopp_delay": (TraceCase("lds_sopp_delay", [
    ds_load_b32(vdst=v[2], addr=v[0]), s_nop(simm16=0), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(4, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(8, "VALUINST", 0, 12, None),
    NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(33, "WAVEEND", 0, 16, None),
  ]),
  "lds_waitcnt": (TraceCase("lds_waitcnt", [ds_load_b32(vdst=v[2], addr=v[0]), s_waitcnt(simm16=0), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(32, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(33, "VALUINST", 0, 12, None),
    NormalizedSQTTEvent(34, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(35, "WAVEEND", 0, 16, None),
  ]),
  "lds_wait_idle": (TraceCase("lds_wait_idle", [
    ds_load_b32(vdst=v[2], addr=v[0]), s_wait_idle(), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(33, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(34, "VALUINST", 0, 12, None),
    NormalizedSQTTEvent(35, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(36, "WAVEEND", 0, 16, None),
  ]),
  "lds_store_waitcnt": (TraceCase("lds_store_waitcnt", [ds_store_b32(addr=v[0], data0=v[0]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_2"),
    NormalizedSQTTEvent(3, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(34, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(36, "WAVEEND", 0, 12, None),
  ]),
  "lds_store_addtid_waitcnt": (TraceCase("lds_store_addtid_waitcnt", [ds_store_addtid_b32(), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_1"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(32, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(34, "WAVEEND", 0, 12, None),
  ]),
  "lds_load_addtid_waitcnt": (TraceCase("lds_load_addtid_waitcnt", [ds_load_addtid_b32(vdst=v[1]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(32, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(34, "WAVEEND", 0, 12, None),
  ]),
  "lds_permute_waitcnt": (TraceCase("lds_permute_waitcnt", [
    ds_permute_b32(vdst=v[2], addr=v[0], data0=v[1]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_2"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(36, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(38, "WAVEEND", 0, 12, None),
  ]),
  "lds_bpermute_dependency": (TraceCase("lds_bpermute_dependency", [
    ds_bpermute_b32(vdst=v[2], addr=v[0], data0=v[1]), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_2"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(9, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(37, "WAVEEND", 0, 12, None),
  ]),
  "lds_store_b64": (TraceCase("lds_store_b64", [ds_store_b64(addr=v[0], data0=v[1:2]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_3"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(36, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(38, "WAVEEND", 0, 12, None),
  ]),
  "lds_store_b96": (TraceCase("lds_store_b96", [ds_store_b96(addr=v[0], data0=v[1:3]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_4"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(39, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(41, "WAVEEND", 0, 12, None),
  ]),
  "lds_store_b128": (TraceCase("lds_store_b128", [ds_store_b128(addr=v[0], data0=v[1:4]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_5"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(40, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(42, "WAVEEND", 0, 12, None),
  ]),
  "single_wave_barrier": (TraceCase("single_wave_barrier", [s_barrier(), s_mov_b32(s[0], 1), s_endpgm()], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "IMMEDIATE", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "multi_wave_barrier": (TraceCase("multi_wave_barrier", [s_barrier(), s_mov_b32(s[0], 1), s_endpgm()], local_size=64), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(0, "WAVESTART", 1, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "BARRIER"), NormalizedSQTTEvent(1, "INST", 1, 0, "BARRIER"),
    NormalizedSQTTEvent(2, "WAVERDY", 0, None, None), NormalizedSQTTEvent(2, "WAVERDY", 1, None, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 1, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None), NormalizedSQTTEvent(4, "WAVEEND", 1, 8, None),
  ]),
}

class TestSQTTCycleModel(unittest.TestCase):
  def test_mock_trace_cases(self):
    for name, (case, expected) in CASES.items():
      with self.subTest(name=name):
        got = _run_mock_trace(case)
        self.assertListEqual(got, expected)

  def test_large_delta_packet(self):
    blob = _run_mock_blob(CASES["lds_roundtrip"][0])
    self.assertTrue(any(isinstance(pkt, TS_DELTA_OR_MARK) for pkt in decode(blob)))

@unittest.skipUnless(getenv("SQTT_CYCLE_HW", 0), "set SQTT_CYCLE_HW=1 to run hardware SQTT cycle comparisons")
class TestSQTTHardwareCycle(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # SQTT registers are device-global, so these tests must not race another SQTT capture.
    if os.environ.get("PYTEST_XDIST_WORKER") is not None: raise unittest.SkipTest("hardware SQTT cycle tests must run without xdist")
    from tinygrad import Device
    if Device.DEFAULT != "AMD": raise unittest.SkipTest("hardware SQTT cycle tests require DEV=AMD")
    if not Device["AMD"].sqtt_enabled: raise unittest.SkipTest("hardware SQTT cycle tests require VIZ=2 or SQTT=1")

  def test_hardware_trace_order(self):
    for name, (case, expected) in CASES.items():
      with self.subTest(name=name):
        got = _instruction_events(_run_hardware_trace(case))
        want = _instruction_events(expected)
        if case.local_size > 32: want = [e for e in want if e.wave == 0]
        if (msg:=_first_mismatch(got, want, check_time=False, case=case)) is not None: self.fail(msg)

  def test_hardware_trace_lifecycle(self):
    case = CASES["salu_chain"][0]
    got = _run_hardware_trace(case, require_instructions=False)
    self.assertIn("WAVESTART", [e.kind for e in got])
    self.assertIn("WAVEEND", [e.kind for e in got])

  def test_hardware_trace_cycles(self):
    if not getenv("SQTT_CYCLE_STRICT", 0): self.skipTest("set SQTT_CYCLE_STRICT=1 to require exact cycle timestamps")
    for name, (case, expected) in CASES.items():
      if case.local_size > 32: continue
      with self.subTest(name=name):
        got = _rebase(_instruction_events(_run_hardware_trace(case)))
        want = _rebase(_instruction_events(expected))
        if (msg:=_first_mismatch(got, want, check_time=True, case=case)) is not None: self.fail(msg)

if __name__ == "__main__":
  unittest.main()
