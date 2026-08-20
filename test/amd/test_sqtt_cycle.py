#!/usr/bin/env python3
import ctypes, os, random, re, unittest
from dataclasses import dataclass
from typing import Iterable

from tinygrad.helpers import Context, Target, getenv
from tinygrad.renderer.amd.sqtt import (ALUEXEC, IMMEDIATE, IMMEDIATE_MASK, INST, INST_RDNA4, TS_DELTA_OR_MARK, VALUINST, VMEMEXEC, WAVEEND,
  WAVEEND_RDNA4, WAVERDY, WAVESTART, WAVESTART_RDNA4, AluSrc, InstOp, decode)
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.runtime.autogen.amd.rdna3.enum import VOPDOp
from test.mockgpu.amd.emu import run_asm, sqtt_traces
from test.mockgpu.amd.sqtt import RDNA3SQTTTraceBuilder

_RSRC2_WITH_LDS = 0x19c | (1 << 15)

@dataclass(frozen=True)
class TraceCase:
  name: str
  instructions: list
  local_size: int = 1
  vgpr_count: int = 16
  normalize_exec_start: bool = False

def _arithmetic_corpus_case(seed: int) -> TraceCase:
  rng, instructions = random.Random(seed), []
  for _ in range(10):
    op = rng.randrange(7)
    if op == 0: instructions.append(s_mov_b32(s[rng.randrange(4)], rng.randrange(1, 32)))
    elif op == 1: instructions.append(s_add_u32(s[rng.randrange(4)], s[rng.randrange(4)], rng.randrange(1, 4)))
    elif op == 2: instructions.append(s_mul_i32(s[rng.randrange(4)], s[rng.randrange(4)], s[rng.randrange(4)]))
    elif op == 3: instructions.append(v_mov_b32_e32(v[rng.randrange(8)], rng.randrange(1, 32)))
    elif op == 4: instructions.append(v_add_f32_e32(v[rng.randrange(8)], v[rng.randrange(8)], v[rng.randrange(8)]))
    elif op == 5: instructions.append(v_lshlrev_b64(v[8:9], rng.randrange(1, 8), v[0:1]))
    else: instructions.append(v_rcp_f32_e32(v[rng.randrange(8)], v[rng.randrange(8)]))
  return TraceCase(f"arithmetic_corpus_{seed}", instructions + [s_endpgm()], local_size=32)

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

def _advance_pc(pc: int, inst, pkt: INST|VALUINST|IMMEDIATE, returns: list[int]) -> int:
  if isinstance(pkt, INST) and pkt.op == InstOp.JUMP:
    x = getattr(inst, "simm16") & 0xffff
    return pc + inst.size() + (x - 0x10000 if x & 0x8000 else x) * 4
  if isinstance(pkt, INST) and pkt.op == InstOp.CALL:
    if getattr(getattr(inst, "op", None), "name", "") == "S_CALL_B64":
      returns.append(pc + inst.size())
      x = getattr(inst, "simm16") & 0xffff
      return pc + inst.size() + (x - 0x10000 if x & 0x8000 else x) * 4
    if returns: return returns.pop()
  return pc + inst.size()

def _rebase(events: Iterable[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  ret = list(events)
  base = min((e.time for e in ret), default=0)
  return [NormalizedSQTTEvent(e.time - base, e.kind, e.wave, e.pc, e.op) for e in ret]

def _selected_trace_slot(pkt) -> bool:
  return pkt.simd == getenv("SQTT_SIMD_SEL", 0) and pkt.sa == getenv("SQTT_SA_SEL", 0) and pkt.wgp == getenv("SQTT_WGP_SEL", 0)

def normalize_controlled_hardware_trace(blob: bytes, instructions: list) -> list[NormalizedSQTTEvent]:
  pc_map, wave_pc, wave_returns, events = _pc_map(instructions), {}, {}, []
  for pkt in decode(blob):
    if isinstance(pkt, (WAVESTART, WAVESTART_RDNA4)):
      if not _selected_trace_slot(pkt): continue
      wave_pc[pkt.wave] = 0
      wave_returns[pkt.wave] = []
      events.append(NormalizedSQTTEvent(pkt._time, "WAVESTART", pkt.wave, None, None))
    elif isinstance(pkt, (WAVEEND, WAVEEND_RDNA4)):
      if not _selected_trace_slot(pkt): continue
      events.append(NormalizedSQTTEvent(pkt._time, "WAVEEND", pkt.wave, wave_pc.pop(pkt.wave, None), None))
      wave_returns.pop(pkt.wave, None)
    elif isinstance(pkt, WAVERDY):
      for wave in range(16):
        if pkt.mask & (1 << wave): events.append(NormalizedSQTTEvent(pkt._time, "WAVERDY", wave, None, None))
    elif isinstance(pkt, (INST, INST_RDNA4)) and pkt.wave in wave_pc:
      try: op = pkt.op
      except ValueError: continue
      if op.name.startswith("OTHER_"): continue
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, "INST", pkt.wave, pc, op.name))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt, wave_returns[pkt.wave])
    elif isinstance(pkt, (VALUINST, IMMEDIATE)) and pkt.wave in wave_pc:
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, type(pkt).__name__.removesuffix("_MASK"), pkt.wave, pc, None))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt, wave_returns[pkt.wave])
    elif isinstance(pkt, IMMEDIATE_MASK):
      for wave in range(16):
        if (pkt.mask & (1 << wave)) and wave in wave_pc:
          pc = _next_trace_pc(wave_pc[wave], pc_map)
          events.append(NormalizedSQTTEvent(pkt._time, "IMMEDIATE", wave, pc, None))
          wave_pc[wave] = _advance_pc(pc, pc_map[pc], pkt, wave_returns[wave])
    elif isinstance(pkt, ALUEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "ALUEXEC", None, None, pkt.src.name))
    elif isinstance(pkt, VMEMEXEC):
      events.append(NormalizedSQTTEvent(pkt._time, "VMEMEXEC", None, None, pkt.src.name))
  return _rebase(events)

def normalize_mock_trace(blob: bytes, instructions: list) -> list[NormalizedSQTTEvent]:
  pc_map, wave_pc, wave_returns, events = _pc_map(instructions), {}, {}, []
  for pkt in decode(blob):
    if isinstance(pkt, WAVESTART):
      if not _selected_trace_slot(pkt): continue
      wave_pc[pkt.wave] = 0
      wave_returns[pkt.wave] = []
      events.append(NormalizedSQTTEvent(pkt._time, "WAVESTART", pkt.wave, None, None))
    elif isinstance(pkt, WAVEEND):
      if not _selected_trace_slot(pkt): continue
      events.append(NormalizedSQTTEvent(pkt._time, "WAVEEND", pkt.wave, wave_pc.pop(pkt.wave), None))
      wave_returns.pop(pkt.wave)
    elif isinstance(pkt, WAVERDY):
      for wave in range(16):
        if pkt.mask & (1 << wave): events.append(NormalizedSQTTEvent(pkt._time, "WAVERDY", wave, None, None))
    elif isinstance(pkt, INST) and not pkt.op.name.startswith("OTHER_"):
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, "INST", pkt.wave, pc, pkt.op.name))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt, wave_returns[pkt.wave])
    elif isinstance(pkt, (VALUINST, IMMEDIATE)):
      pc = _next_trace_pc(wave_pc[pkt.wave], pc_map)
      events.append(NormalizedSQTTEvent(pkt._time, type(pkt).__name__, pkt.wave, pc, None))
      wave_pc[pkt.wave] = _advance_pc(pc, pc_map[pc], pkt, wave_returns[pkt.wave])
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
  .amdhsa_next_free_vgpr {case.vgpr_count}
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
  .amdhsa_workgroup_processor_mode 0
  .amdhsa_system_vgpr_workitem_id 0
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
    .vgpr_count: {case.vgpr_count}
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

def _run_hardware_trace(case: TraceCase, require_instructions: bool=True, min_instructions: int=1) -> list[NormalizedSQTTEvent]:
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

  for _ in range(getenv("SQTT_CYCLE_HW_WARMUP", 5)):
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileDeviceEvent, ProfileProgramEvent))]
    prg(global_size=(1, 1, 1), local_size=(case.local_size, 1, 1), wait=True)
    dev.synchronize()

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
    if not require_instructions or sum(e.kind in {"INST", "VALUINST", "IMMEDIATE"} for e in ret) >= min_instructions: return ret
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

def _timed_events(events: Iterable[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  ret = [e for e in events if e.kind in {"INST", "VALUINST", "IMMEDIATE", "ALUEXEC", "VMEMEXEC"}]
  return _rebase(ret)

def _normalize_exec_start(events: list[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  base = next((e.time for e in events if e.kind in {"ALUEXEC", "VMEMEXEC"}), 0)
  return [NormalizedSQTTEvent(e.time - base if e.kind in {"ALUEXEC", "VMEMEXEC"} else e.time, e.kind, e.wave, e.pc, e.op) for e in events]

def _structural_events(events: Iterable[NormalizedSQTTEvent]) -> list[NormalizedSQTTEvent]:
  return [e for e in events if e.kind not in {"ALUEXEC", "VMEMEXEC", "WAVEEND"}]

CASES: dict[str, tuple[TraceCase, list[NormalizedSQTTEvent]]] = {
  "salu_chain": (TraceCase("salu_chain", [s_mov_b32(s[0], 1), s_add_u32(s[1], s[0], 2), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"), NormalizedSQTTEvent(1, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "salu_mul_i32": (TraceCase("salu_mul_i32", [
    s_mul_i32(s[0], s[1], s[2]), s_delay_alu(11), s_add_u32(s[3], s[0], 1), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(4, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(6, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(7, "WAVEEND", 0, 12, None),
  ]),
  "salu_mul_hi": (TraceCase("salu_mul_hi", [
    s_mul_hi_u32(s[0], s[1], s[2]), s_delay_alu(11), s_add_u32(s[3], s[0], 1), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(4, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(6, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(7, "WAVEEND", 0, 12, None),
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
  "call_return": (TraceCase("call_return", [
    s_call_b64(s[0:1], 1), s_endpgm(), s_mov_b32(s[2], 1), s_setpc_b64(s[0:1]),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "CALL"),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(29, "INST", 0, 8, "SALU"),
    NormalizedSQTTEvent(30, "INST", 0, 12, "CALL"), NormalizedSQTTEvent(31, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(58, "WAVEEND", 0, 4, None),
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
  "valu_nop4": (TraceCase("valu_nop4", [v_mov_b32_e32(v[0], 0), s_nop(4), v_mov_b32_e32(v[1], 1), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(8, "IMMEDIATE", 0, 4, None),
    NormalizedSQTTEvent(9, "VALUINST", 0, 8, None), NormalizedSQTTEvent(15, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(16, "WAVEEND", 0, 12, None),
  ]),
  "valu_nop63": (TraceCase("valu_nop63", [v_mov_b32_e32(v[0], 0), s_nop(63), v_mov_b32_e32(v[1], 1), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(67, "IMMEDIATE", 0, 4, None),
    NormalizedSQTTEvent(68, "VALUINST", 0, 8, None), NormalizedSQTTEvent(74, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(75, "WAVEEND", 0, 12, None),
  ]),
  "salu_nop63": (TraceCase("salu_nop63", [s_nop(63), s_mov_b32(s[0], 1), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "IMMEDIATE", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(4, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(5, "WAVEEND", 0, 8, None),
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
  "vopd_dependency": (TraceCase("vopd_dependency", [
    VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32, vdstx=v[0], srcx0=v[10], vsrcx1=v[10],
         vdsty=v[1], srcy0=v[11], vsrcy1=v[11]),
    VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32, vdstx=v[2], srcx0=v[0], vsrcx1=v[0],
         vdsty=v[3], srcy0=v[1], vsrcy1=v[1]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None), NormalizedSQTTEvent(2, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(15, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(16, "WAVEEND", 0, 16, None),
  ]),
  "wmma_same_block": (TraceCase("wmma_same_block", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[0:7], src1=v[0:7], src2=v[0:7]), s_endpgm(),
  ], local_size=32, vgpr_count=8), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(42, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(43, "WAVEEND", 0, 8, None),
  ]),
  "wmma_two_blocks": (TraceCase("wmma_two_blocks", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[8:15], src1=v[0:7], src2=v[0:7]), s_endpgm(),
  ], local_size=32, vgpr_count=16), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(43, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(44, "WAVEEND", 0, 8, None),
  ]),
  "wmma_independent": (TraceCase("wmma_independent", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]),
    v_wmma_f32_16x16x16_f16(vdst=v[32:39], src0=v[48:55], src1=v[56:63], src2=v[32:39]), s_endpgm(),
  ], local_size=32, vgpr_count=64), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None), NormalizedSQTTEvent(2, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(44, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(78, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(79, "WAVEEND", 0, 16, None),
  ]),
  "wmma_dependency": (TraceCase("wmma_dependency", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]),
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[0:7], src1=v[24:31], src2=v[0:7]), s_endpgm(),
  ], local_size=32, vgpr_count=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None), NormalizedSQTTEvent(2, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(44, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(78, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(79, "WAVEEND", 0, 16, None),
  ]),
  "vinterp_independent": (TraceCase("vinterp_independent", [
    v_interp_p10_f32(vdst=v[0], src0=v[1], src1=v[2], src2=v[3]),
    v_interp_p10_f32(vdst=v[4], src0=v[5], src1=v[6], src2=v[7]), s_endpgm(),
  ], local_size=32, normalize_exec_start=True), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VINTERP"), NormalizedSQTTEvent(2, "INST", 0, 8, "VINTERP"),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(11, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(12, "WAVEEND", 0, 16, None),
  ]),
  "vinterp_dependency": (TraceCase("vinterp_dependency", [
    v_interp_p10_f32(vdst=v[0], src0=v[1], src1=v[2], src2=v[3]),
    v_interp_p10_f32(vdst=v[4], src0=v[0], src1=v[5], src2=v[6]), s_endpgm(),
  ], local_size=32, normalize_exec_start=True), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "VINTERP"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "VINTERP"), NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(16, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(17, "WAVEEND", 0, 16, None),
  ]),
  "vinterp_to_valu": (TraceCase("vinterp_to_valu", [
    v_interp_p10_f32(vdst=v[0], src0=v[1], src1=v[2], src2=v[3]), v_add_f32_e32(v[4], v[0], v[0]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "VINTERP"),
    NormalizedSQTTEvent(2, "VALUINST", 0, 8, None), NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(14, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(15, "WAVEEND", 0, 12, None),
  ]),
  "valu_to_vinterp": (TraceCase("valu_to_vinterp", [
    v_add_f32_e32(v[0], v[1], v[1]), v_interp_p10_f32(vdst=v[4], src0=v[0], src1=v[5], src2=v[6]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "VINTERP"), NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(15, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(16, "WAVEEND", 0, 12, None),
  ]),
  "const_to_vinterp": (TraceCase("const_to_vinterp", [
    v_mov_b32_e32(v[0], 0), v_interp_p10_f32(vdst=v[4], src0=v[0], src1=v[5], src2=v[6]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "VINTERP"), NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(13, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(14, "WAVEEND", 0, 12, None),
  ]),
  "vinterp_wmma": (TraceCase("vinterp_wmma", [
    v_interp_p10_f32(vdst=v[8], src0=v[9], src1=v[10], src2=v[11]),
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]),
    v_wmma_f32_16x16x16_f16(vdst=v[32:39], src0=v[48:55], src1=v[56:63], src2=v[32:39]), s_endpgm(),
  ], local_size=32, vgpr_count=64), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "VINTERP"), NormalizedSQTTEvent(2, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(3, "VALUINST", 0, 16, None), NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(45, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(79, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(80, "WAVEEND", 0, 24, None),
  ]),
  "wmma_vinterp_wmma": (TraceCase("wmma_vinterp_wmma", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]),
    v_interp_p10_f32(vdst=v[8], src0=v[9], src1=v[10], src2=v[11]),
    v_wmma_f32_16x16x16_f16(vdst=v[32:39], src0=v[48:55], src1=v[56:63], src2=v[32:39]), s_endpgm(),
  ], local_size=32, vgpr_count=64), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None), NormalizedSQTTEvent(2, "INST", 0, 8, "VINTERP"),
    NormalizedSQTTEvent(3, "VALUINST", 0, 16, None), NormalizedSQTTEvent(44, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(45, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(80, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(81, "WAVEEND", 0, 24, None),
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
  "delay_valu_dep1": (TraceCase("delay_valu_dep1", [
    v_mov_b32_e32(v[0], 0), s_delay_alu(1), v_mov_b32_e32(v[1], 1), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(6, "VALUINST", 0, 8, None), NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(12, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(13, "WAVEEND", 0, 12, None),
  ]),
  "delay_valu_dep4": (TraceCase("delay_valu_dep4", [
    v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 1), v_mov_b32_e32(v[2], 2), v_mov_b32_e32(v[3], 3),
    s_delay_alu(4), v_mov_b32_e32(v[4], 4), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "VALUINST", 0, 0, None), NormalizedSQTTEvent(2, "VALUINST", 0, 4, None),
    NormalizedSQTTEvent(3, "VALUINST", 0, 8, None), NormalizedSQTTEvent(4, "VALUINST", 0, 12, None),
    NormalizedSQTTEvent(6, "VALUINST", 0, 20, None), NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(8, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(12, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(13, "WAVEEND", 0, 24, None),
  ]),
  "delay_trans_dep1": (TraceCase("delay_trans_dep1", [
    v_rcp_f32_e32(v[1], v[0]), s_delay_alu(5), v_mov_b32_e32(v[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "VALUT_4"),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(11, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(17, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(18, "WAVEEND", 0, 12, None),
  ]),
  "delay_fma_accum": (TraceCase("delay_fma_accum", [
    v_fmac_f32_e32(v[0], v[1], v[2]), s_delay_alu(8), v_fmac_f32_e32(v[0], v[3], v[4]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "VALUINST", 0, 8, None), NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(15, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(16, "WAVEEND", 0, 12, None),
  ]),
  "delay_salu_cycle1": (TraceCase("delay_salu_cycle1", [
    s_mov_b32(s[0], 1), s_delay_alu(9), s_add_u32(s[1], s[0], 2), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(3, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(6, "WAVEEND", 0, 12, None),
  ]),
  "delay_salu_cycle2": (TraceCase("delay_salu_cycle2", [
    s_mov_b32(s[0], 1), s_delay_alu(10), s_add_u32(s[1], s[0], 2), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(6, "WAVEEND", 0, 12, None),
  ]),
  "delay_salu_cycle3": (TraceCase("delay_salu_cycle3", [
    s_mov_b32(s[0], 1), s_delay_alu(11), s_add_u32(s[1], s[0], 2), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(2, "INST", 0, 8, "SALU"), NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(6, "WAVEEND", 0, 12, None),
  ]),
  "delay_second_next": (TraceCase("delay_second_next", [
    v_mov_b32_e32(v[0], 0), s_delay_alu(0x90), v_mov_b32_e32(v[1], 1), v_mov_b32_e32(v[2], 2), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "VALUINST", 0, 8, None), NormalizedSQTTEvent(7, "VALUINST", 0, 12, None),
    NormalizedSQTTEvent(7, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(8, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(13, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(14, "WAVEEND", 0, 16, None),
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
  "lds_load_b64_waitcnt": (TraceCase("lds_load_b64_waitcnt", [ds_load_b64(vdst=v[2:3], addr=v[0]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(34, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(36, "WAVEEND", 0, 12, None),
  ]),
  "lds_load_b96_waitcnt": (TraceCase("lds_load_b96_waitcnt", [ds_load_b96(vdst=v[2:4], addr=v[0]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(37, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(39, "WAVEEND", 0, 12, None),
  ]),
  "lds_load_b128_waitcnt": (TraceCase("lds_load_b128_waitcnt", [ds_load_b128(vdst=v[2:5], addr=v[0]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(38, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(40, "WAVEEND", 0, 12, None),
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
  "lds_swizzle_waitcnt": (TraceCase("lds_swizzle_waitcnt", [
    ds_swizzle_b32(vdst=v[2], addr=v[0], offset0=0, offset1=0xe0), s_waitcnt(simm16=0), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(2, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(32, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(34, "WAVEEND", 0, 12, None),
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
  "lds_store_2addr_b32_waitcnt": (TraceCase("lds_store_2addr_b32_waitcnt", [
    ds_store_2addr_b32(addr=v[0], data0=v[1], data1=v[2]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_3"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(36, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(38, "WAVEEND", 0, 12, None),
  ]),
  "lds_store_2addr_b64_waitcnt": (TraceCase("lds_store_2addr_b64_waitcnt", [
    ds_store_2addr_b64(addr=v[0], data0=v[1:2], data1=v[3:4]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_5"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(40, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(42, "WAVEEND", 0, 12, None),
  ]),
  "lds_load_2addr_b32_dependency": (TraceCase("lds_load_2addr_b32_dependency", [
    ds_load_2addr_b32(vdst=v[4:5], addr=v[0]), v_add_f32_e32(v[6], v[4], v[4]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(8, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(9, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(34, "WAVEEND", 0, 12, None),
  ]),
  "lds_load_2addr_b64_waitcnt": (TraceCase("lds_load_2addr_b64_waitcnt", [
    ds_load_2addr_b64(vdst=v[4:7], addr=v[0]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_RD"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(36, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(38, "WAVEEND", 0, 12, None),
  ]),
  "lds_add_u32_waitcnt": (TraceCase("lds_add_u32_waitcnt", [ds_add_u32(addr=v[0], data0=v[1]), s_waitcnt(simm16=0), s_endpgm()]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_2"),
    NormalizedSQTTEvent(3, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(34, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(36, "WAVEEND", 0, 12, None),
  ]),
  "lds_add_rtn_u32_dependency": (TraceCase("lds_add_rtn_u32_dependency", [
    ds_add_rtn_u32(vdst=v[2], addr=v[0], data0=v[1]), v_add_f32_e32(v[3], v[2], v[2]), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_2"),
    NormalizedSQTTEvent(3, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(9, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(35, "WAVEEND", 0, 12, None),
  ]),
  "lds_add_rtn_u64_waitcnt": (TraceCase("lds_add_rtn_u64_waitcnt", [
    ds_add_rtn_u64(vdst=v[4:5], addr=v[0], data0=v[1:2]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_3"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(37, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(39, "WAVEEND", 0, 12, None),
  ]),
  "lds_cmpstore_b64_waitcnt": (TraceCase("lds_cmpstore_b64_waitcnt", [
    ds_cmpstore_b64(addr=v[0], data0=v[1:2], data1=v[3:4]), s_waitcnt(simm16=0), s_endpgm(),
  ]), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "INST", 0, 0, "LDS_WR_5"),
    NormalizedSQTTEvent(4, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(40, "IMMEDIATE", 0, 8, None),
    NormalizedSQTTEvent(42, "WAVEEND", 0, 12, None),
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
  "mixed_alu": (TraceCase("mixed_alu", [
    s_mov_b32(s[0], 3), v_mov_b32_e32(v[0], 0), v_add_f32_e32(v[1], v[0], v[0]), s_nop(4), v_rcp_f32_e32(v[2], v[1]),
    s_delay_alu(5), v_add_f32_e32(v[3], v[2], v[2]), s_branch(1), s_mov_b32(s[1], 9), s_mov_b32(s[2], 2), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "SALU"),
    NormalizedSQTTEvent(2, "VALUINST", 0, 4, None), NormalizedSQTTEvent(3, "VALUINST", 0, 8, None),
    NormalizedSQTTEvent(3, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(8, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(10, "IMMEDIATE", 0, 12, None), NormalizedSQTTEvent(11, "INST", 0, 16, "VALUT_4"),
    NormalizedSQTTEvent(14, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(19, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(21, "VALUINST", 0, 24, None), NormalizedSQTTEvent(22, "INST", 0, 28, "JUMP"),
    NormalizedSQTTEvent(29, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(32, "INST", 0, 36, "SALU"),
    NormalizedSQTTEvent(34, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(35, "WAVEEND", 0, 40, None),
  ]),
  "mixed_lds": (TraceCase("mixed_lds", [
    v_lshlrev_b32_e32(v[1], 2, v[0]), ds_store_b32(addr=v[1], data0=v[0]), ds_load_b32(vdst=v[2], addr=v[1]), s_waitcnt(0),
    v_add_f32_e32(v[3], v[2], v[2]), v_rcp_f32_e32(v[4], v[3]), s_endpgm(),
  ], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(10, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(19, "INST", 0, 4, "LDS_WR_2"),
    NormalizedSQTTEvent(20, "INST", 0, 12, "LDS_RD"), NormalizedSQTTEvent(22, "VMEMEXEC", None, None, "LDS"),
    NormalizedSQTTEvent(24, "VMEMEXEC", None, None, "LDS"), NormalizedSQTTEvent(54, "IMMEDIATE", 0, 20, None),
    NormalizedSQTTEvent(55, "VALUINST", 0, 24, None), NormalizedSQTTEvent(56, "INST", 0, 28, "VALUT_4"),
    NormalizedSQTTEvent(63, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(68, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(69, "WAVEEND", 0, 32, None),
  ]),
  "mixed_matrix": (TraceCase("mixed_matrix", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]),
    v_interp_p10_f32(vdst=v[8], src0=v[9], src1=v[10], src2=v[11]), v_add_f32_e32(v[12], v[8], v[8]),
    v_wmma_f32_16x16x16_f16(vdst=v[32:39], src0=v[48:55], src1=v[56:63], src2=v[32:39]), s_endpgm(),
  ], local_size=32, vgpr_count=64), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "VALUINST", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 8, "VINTERP"), NormalizedSQTTEvent(3, "VALUINST", 0, 16, None),
    NormalizedSQTTEvent(4, "VALUINST", 0, 20, None), NormalizedSQTTEvent(44, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(45, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(51, "ALUEXEC", None, None, "VALU"),
    NormalizedSQTTEvent(84, "ALUEXEC", None, None, "VALU"), NormalizedSQTTEvent(85, "WAVEEND", 0, 28, None),
  ]),
  "single_wave_barrier": (TraceCase("single_wave_barrier", [s_barrier(), s_mov_b32(s[0], 1), s_endpgm()], local_size=32), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None),
    NormalizedSQTTEvent(1, "IMMEDIATE", 0, 0, None),
    NormalizedSQTTEvent(2, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(2, "ALUEXEC", None, None, "SALU"),
    NormalizedSQTTEvent(4, "WAVEEND", 0, 8, None),
  ]),
  "multi_wave_barrier": (TraceCase("multi_wave_barrier", [s_barrier(), s_mov_b32(s[0], 1), s_endpgm()], local_size=96), [
    NormalizedSQTTEvent(0, "WAVESTART", 0, None, None), NormalizedSQTTEvent(1, "INST", 0, 0, "BARRIER"),
    NormalizedSQTTEvent(2, "WAVERDY", 0, None, None), NormalizedSQTTEvent(2, "WAVESTART", 1, None, None),
    NormalizedSQTTEvent(2, "INST", 1, 0, "BARRIER"), NormalizedSQTTEvent(3, "WAVERDY", 1, None, None),
    NormalizedSQTTEvent(3, "INST", 0, 4, "SALU"), NormalizedSQTTEvent(4, "INST", 1, 4, "SALU"),
    NormalizedSQTTEvent(5, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(6, "WAVEEND", 0, 8, None),
    NormalizedSQTTEvent(6, "ALUEXEC", None, None, "SALU"), NormalizedSQTTEvent(7, "WAVEEND", 1, 8, None),
  ]),
}

COMPOSITION_CASES: dict[str, tuple[TraceCase, list[int]]] = {
  "compose_salu_overlap": (TraceCase("compose_salu_overlap", [s_add_u32(s[0], s[0], 1), s_endpgm()], local_size=32), [0, 4]),
  "compose_salu_warm_overlap": (TraceCase("compose_salu_warm_overlap", [
    s_mov_b32(s[1], 1), s_add_u32(s[0], s[0], 1), s_endpgm()], local_size=32), [0, 1, 2, 5]),
  "compose_salu_gap_overlap": (TraceCase("compose_salu_gap_overlap", [
    s_mov_b32(s[0], 1), v_lshlrev_b64(v[8:9], 2, v[0:1]), s_add_u32(s[1], s[1], 1), s_endpgm()], local_size=32),
    [0, 1, 2, 6, 10, 11]),
  "compose_salu_anti_dependency": (TraceCase("compose_salu_anti_dependency", [
    s_add_u32(s[0], s[1], 2), s_mov_b32(s[1], 1), s_endpgm()], local_size=32), [0, 1, 4, 5]),
  "compose_salu_mul_anti_dependency": (TraceCase("compose_salu_mul_anti_dependency", [
    s_mul_i32(s[0], s[2], s[3]), s_mov_b32(s[3], 1), s_mov_b32(s[2], 1), s_endpgm()], local_size=32), [0, 1, 2, 5, 6, 7]),
  "compose_salu_mul_startup": (TraceCase("compose_salu_mul_startup", [
    s_mul_i32(s[1], s[3], s[0]), s_endpgm()], local_size=32), [0, 5]),
  "compose_salu_mul_initialized": (TraceCase("compose_salu_mul_initialized", [
    s_mov_b32(s[3], 1), s_mov_b32(s[0], 1), s_mul_i32(s[1], s[3], s[0]), s_endpgm()], local_size=32), [0, 1, 2, 2, 3, 6]),
  "compose_salu_mul_throughput": (TraceCase("compose_salu_mul_throughput", [
    s_mul_i32(s[i], s[4+i*2], s[5+i*2]) for i in range(4)] + [s_endpgm()], local_size=32), [0, 1, 2, 3, 3, 5, 7, 9]),
  "compose_interleaved_read_warmup": (TraceCase("compose_interleaved_read_warmup", [
    v_mov_b32_e32(v[3], 2), s_mov_b32(s[1], 1), v_add_f32_e32(v[0], v[0], v[0]), s_endpgm()], local_size=32), [0, 1, 2, 3, 6, 10]),
  "compose_valu_bank_conflict": (TraceCase("compose_valu_bank_conflict", [
    v_add_f32_e32(v[0], v[3], v[7]), s_endpgm()], local_size=32), [0, 10]),
  "compose_valu_read_order": (TraceCase("compose_valu_read_order", [
    v_mov_b32_e32(v[1], 1), s_mov_b32(s[4], 1), v_add_f32_e32(v[2], v[4], v[5]), s_endpgm()], local_size=32), [0, 1, 2, 3, 6, 10]),
  "compose_valub_read_order": (TraceCase("compose_valub_read_order", [
    v_mov_b32_e32(v[2], 1), s_mov_b32(s[4], 1)] + [v_lshlrev_b64(v[8:9], 2, v[0:1]) for _ in range(3)] + [s_endpgm()],
    local_size=32), [0, 1, 2, 3, 4, 6, 6, 11, 13, 15]),
  "compose_salu_overlap_wait": (TraceCase("compose_salu_overlap_wait", [
    s_add_u32(s[0], s[0], 1), s_waitcnt(simm16=0), s_endpgm()], local_size=32), [0, 3, 4]),
  "compose_salu_no_overlap_wait": (TraceCase("compose_salu_no_overlap_wait", [
    s_add_u32(s[0], s[1], 1), s_waitcnt(simm16=0), s_endpgm()], local_size=32), [0, 2, 3]),
  "compose_salu_mul_overlap_wait": (TraceCase("compose_salu_mul_overlap_wait", [
    s_mul_i32(s[0], s[0], s[1]), s_waitcnt(simm16=0), s_endpgm()], local_size=32), [0, 3, 5]),
  "compose_valub_salu": (TraceCase("compose_valub_salu", [
    v_lshlrev_b64(v[2:3], 2, v[0:1]), v_lshlrev_b64(v[2:3], 2, v[0:1]), s_mov_b32(s[0], 1), s_endpgm(),
  ], local_size=32), [0, 2, 7, 9, 10, 12]),
  "compose_salu_queue": (TraceCase("compose_salu_queue", [s_mov_b32(s[0], 1)] + [
    s_add_u32(s[0], s[0], 1) for _ in range(11)] + [s_endpgm()], local_size=32),
    [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24]),
  "compose_lds_read_queue": (TraceCase("compose_lds_read_queue", [
    ds_load_b32(vdst=v[2+i], addr=v[0]) for i in range(4)] + [s_endpgm()], local_size=32), [0, 1, 2, 3, 3, 4, 5, 6]),
  "compose_lds_write_queue": (TraceCase("compose_lds_write_queue", [
    ds_store_b32(addr=v[0], data0=v[1]) for _ in range(4)] + [s_endpgm()], local_size=32), [0, 1, 2, 3, 3, 5, 7, 9]),
  "compose_salu_valu": (TraceCase("compose_salu_valu", [
    s_mov_b32(s[0], 1), v_mov_b32_e32(v[0], s[0]), v_add_f32_e32(v[1], v[0], v[0]), s_add_u32(s[1], s[0], 2),
    v_rcp_f32_e32(v[2], v[1]), s_endpgm(),
  ], local_size=32), [0, 1, 2, 2, 7, 7, 8, 11, 13, 18]),
  "compose_alu_interleave": (TraceCase("compose_alu_interleave", [
    v_mov_b32_e32(v[0], 0), s_mov_b32(s[0], 1), v_add_f32_e32(v[1], v[0], v[0]), s_add_u32(s[1], s[0], 2),
    v_add_f32_e32(v[2], v[1], v[1]), s_endpgm(),
  ], local_size=32), [0, 1, 2, 3, 3, 4, 5, 6, 12, 17]),
  "compose_trans_chain": (TraceCase("compose_trans_chain", [
    v_rcp_f32_e32(v[0], v[1]), v_add_f32_e32(v[2], v[0], v[0]), v_rcp_f32_e32(v[3], v[2]),
    v_add_f32_e32(v[4], v[3], v[3]), s_endpgm(),
  ], local_size=32), [0, 1, 4, 5, 9, 19, 24, 34]),
  "compose_trans_to_valub": (TraceCase("compose_trans_to_valub", [
    v_rcp_f32_e32(v[0], v[1]), v_lshlrev_b64(v[8:9], 4, v[0:1]), s_endpgm(),
  ], local_size=32), [0, 4, 9, 20]),
  "compose_lds_no_wait": (TraceCase("compose_lds_no_wait", [
    v_lshlrev_b32_e32(v[1], 2, v[0]), ds_store_b32(addr=v[1], data0=v[0]), ds_load_b32(vdst=v[2], addr=v[1]),
    v_add_f32_e32(v[3], v[2], v[2]), v_rcp_f32_e32(v[4], v[3]), s_endpgm(),
  ], local_size=32), [0, 9, 18, 19, 21, 23, 27, 28, 35, 40]),
  "compose_wmma_salu": (TraceCase("compose_wmma_salu", [
    v_wmma_f32_16x16x16_f16(vdst=v[0:7], src0=v[16:23], src1=v[24:31], src2=v[0:7]), s_mov_b32(s[0], 1),
    v_add_f32_e32(v[8], v[9], v[9]), s_add_u32(s[1], s[0], 2),
    v_wmma_f32_16x16x16_f16(vdst=v[32:39], src0=v[48:55], src1=v[56:63], src2=v[32:39]), s_endpgm(),
  ], local_size=32, vgpr_count=64), [0, 1, 2, 3, 3, 4, 5, 43, 44, 79]),
  "compose_interp_lds": (TraceCase("compose_interp_lds", [
    v_interp_p10_f32(vdst=v[8], src0=v[9], src1=v[10], src2=v[11]), ds_store_b32(addr=v[0], data0=v[8]),
    ds_load_b32(vdst=v[12], addr=v[0]), v_add_f32_e32(v[13], v[12], v[12]), s_endpgm(),
  ], local_size=32), [0, 8, 17, 18, 20, 22, 26, 34]),
  "compose_fmac_trans": (TraceCase("compose_fmac_trans", [
    v_fmac_f32_e32(v[0], v[1], v[2]), s_delay_alu(8), v_fmac_f32_e32(v[0], v[3], v[4]),
    v_rcp_f32_e32(v[5], v[0]), v_add_f32_e32(v[6], v[5], v[5]), s_endpgm(),
  ], local_size=32), [0, 1, 2, 3, 9, 14, 19, 29]),
  "compose_long_valu": (TraceCase("compose_long_valu", [
    v_lshlrev_b64(v[2:3], 2, v[0:1]), v_add_f32_e32(v[4], v[2], v[2]), v_rcp_f32_e32(v[5], v[4]),
    v_mov_b32_e32(v[6], 1), s_endpgm(),
  ], local_size=32), [0, 2, 3, 4, 10, 15, 20, 21]),
  "compose_valub_backpressure": (TraceCase("compose_valub_backpressure", [
    v_lshlrev_b64(v[2:3], 2, v[0:1]), v_add_f32_e32(v[4], v[2], v[2]), v_rcp_f32_e32(v[5], v[4]), v_mov_b32_e32(v[6], 1),
  ] * 8 + [s_endpgm()], local_size=32),
    [0, 2, 3, 4, 7, 9, 10, 10, 11, 14, 15, 16, 17, 18, 20, 21, 24, 25, 26, 27, 28, 29, 34, 35, 39, 39, 41, 42, 43, 43,
     48, 49, 53, 53, 55, 56, 57, 57, 62, 63, 67, 67, 69, 70, 71, 71, 76, 77, 81, 81, 83, 84, 85, 85, 90, 91, 95, 99, 104, 105,
     109, 113, 118, 119]),
  "compose_valub_backpressure_no_move": (TraceCase("compose_valub_backpressure_no_move", [
    v_lshlrev_b64(v[2:3], 2, v[0:1]), v_add_f32_e32(v[4], v[2], v[2]), v_rcp_f32_e32(v[5], v[4]),
  ] * 8 + [s_endpgm()], local_size=32),
    [0, 2, 3, 7, 9, 10, 10, 14, 15, 16, 17, 20, 21, 23, 24, 25, 29, 29, 31, 32, 34, 39, 43, 43,
     45, 46, 48, 53, 57, 57, 59, 60, 62, 67, 71, 71, 73, 74, 76, 81, 85, 90, 95, 99, 104, 109, 113, 118]),
  "compose_valu_queue": (TraceCase("compose_valu_queue", [v_mov_b32_e32(v[0], 1)] + [
    v_add_f32_e32(v[i], v[i-1], v[i-1]) for i in range(1, 16)] + [s_endpgm()], local_size=32, vgpr_count=20),
    [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13, 16, 17, 21, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82]),
  "compose_valu_interleaved": (TraceCase("compose_valu_interleaved", [v_mov_b32_e32(v[i], i+1) for i in range(3)] + [
    v_add_f32_e32(v[d*3+c], v[(d-1)*3+c], v[(d-1)*3+c]) for d in range(1, 7) for c in range(3)] + [s_endpgm()],
    local_size=32, vgpr_count=24),
    [0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15, 16, 17, 17, 18, 18, 19, 21, 22, 22,
     23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39]),
  "compose_trans_queue": (TraceCase("compose_trans_queue", [
    v_rcp_f32_e32(v[1], v[0]), v_add_f32_e32(v[2], v[1], v[1]),
    v_rcp_f32_e32(v[3], v[2]), v_add_f32_e32(v[4], v[3], v[3]),
    v_rcp_f32_e32(v[5], v[4]), v_add_f32_e32(v[6], v[5], v[5]),
    v_rcp_f32_e32(v[7], v[6]), v_add_f32_e32(v[8], v[7], v[7]),
    v_rcp_f32_e32(v[9], v[8]), v_add_f32_e32(v[10], v[9], v[9]),
    v_rcp_f32_e32(v[11], v[10]), v_add_f32_e32(v[12], v[11], v[11]),
    v_rcp_f32_e32(v[13], v[12]), v_add_f32_e32(v[14], v[13], v[13]),
    v_rcp_f32_e32(v[15], v[14]), v_add_f32_e32(v[16], v[15], v[15]), s_endpgm(),
  ], local_size=32, vgpr_count=20),
    [0, 1, 4, 5, 8, 9, 9, 12, 13, 16, 17, 19, 20, 21, 24, 24, 25, 34, 39, 39, 40, 49, 54, 64, 69, 79, 84, 94, 99, 109, 114, 124]),
  "compose_repeat_alu": (TraceCase("compose_repeat_alu", [
    v_mov_b32_e32(v[0], 0), s_mov_b32(s[0], 1), v_add_f32_e32(v[1], v[0], v[0]), s_add_u32(s[1], s[0], 2),
    v_add_f32_e32(v[2], v[1], v[1]),
  ] * 4 + [s_endpgm()], local_size=32),
    [0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 18, 19, 20,
     23, 28, 29, 34, 39, 40, 45, 50]),
  "compose_valu_read_warmup": (TraceCase("compose_valu_read_warmup", [v_mov_b32_e32(v[0], 1)] + [
    v_mov_b32_e32(v[2+i], i) for i in range(6)] + [v_add_f32_e32(v[1], v[0], v[0]), s_endpgm()], local_size=32),
    [0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 15]),
  "compose_trans_full_queue": (TraceCase("compose_trans_full_queue", [v_mov_b32_e32(v[0], 1)] + [
    v_add_f32_e32(v[i], v[i-1], v[i-1]) for i in range(1, 14)] + [
    v_rcp_f32_e32(v[14], v[13]), v_add_f32_e32(v[15], v[14], v[14]), s_endpgm()], local_size=32, vgpr_count=20),
    [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13, 16, 17, 21, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 87]),
}

_ARITHMETIC_CORPUS_TIMES: dict[int, list[int]] = {
  0: [0, 1, 2, 3, 3, 4, 5, 6, 6, 9, 9, 11, 12, 19, 20, 21, 31, 33, 34, 44],
  1: [0, 1, 2, 2, 4, 5, 7, 8, 9, 12, 12, 13, 14, 17, 17, 18, 18, 19, 20, 24],
  2: [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 9, 9, 10, 11, 14, 15, 20, 21, 22, 23],
  3: [0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 11, 12, 12, 13, 15, 19, 20],
  4: [0, 1, 2, 3, 4, 4, 5, 5, 7, 8, 9, 9, 10, 12, 12, 13, 14, 16, 27],
  5: [0, 1, 5, 9, 10, 10, 11, 12, 12, 15, 15, 17, 18, 19, 22, 29, 35, 36, 37, 41],
  6: [0, 4, 5, 6, 7, 8, 9, 9, 12, 13, 13, 14, 14, 16, 17, 20, 23, 24, 27, 31],
  7: [0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 16, 17],
  8: [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10, 16, 17, 18, 19],
  9: [0, 1, 2, 3, 4, 5, 6, 6, 11, 12, 12, 13, 13, 15, 16, 17, 19, 21, 22, 26],
}
for _seed, _times in _ARITHMETIC_CORPUS_TIMES.items():
  _case = _arithmetic_corpus_case(_seed)
  COMPOSITION_CASES[_case.name] = (_case, _times)

MULTIWAVE_CASES = {
  "multi_wave_salu": TraceCase("multi_wave_salu", [s_mov_b32(s[0], 1)] + [s_add_u32(s[0], s[0], 1) for _ in range(6)] + [s_endpgm()],
                               local_size=160),
  "multi_wave_valu": TraceCase("multi_wave_valu", [v_mov_b32_e32(v[0], 1)] + [
    v_add_f32_e32(v[0], v[0], v[0]) for _ in range(6)] + [s_endpgm()], local_size=96),
  "multi_wave_lds": TraceCase("multi_wave_lds", [ds_load_b32(vdst=v[2+i], addr=v[0]) for i in range(4)] + [s_endpgm()], local_size=96),
}

MULTIWAVE_EXEC_TIMES = {
  "multi_wave_valu": [6, 12, 17, 22, 27, 32, 37, 38, 43, 48, 53, 58, 63, 68],
  "multi_wave_lds": list(range(3, 15)),
}

EXEC_TIMES: dict[str, list[tuple[str, int, str]]] = {
  "salu_chain": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 4, "SALU")],
  "salu_mul_i32": [("ALUEXEC", 3, "SALU"), ("ALUEXEC", 5, "SALU")],
  "salu_mul_hi": [("ALUEXEC", 3, "SALU"), ("ALUEXEC", 5, "SALU")],
  "call_return": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 30, "SALU")],
  "salu_saveexec": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 4, "SALU"), ("ALUEXEC", 5, "SALU")],
  "valu_independent": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 7, "VALU"), ("ALUEXEC", 8, "VALU")],
  "valu_simple_dependency": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 14, "VALU")],
  "valu_nop4": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 14, "VALU")],
  "valu_nop63": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 73, "VALU")],
  "salu_nop63": [("ALUEXEC", 3, "SALU")],
  "valu_dependency": [("ALUEXEC", 10, "VALU"), ("ALUEXEC", 15, "VALU")],
  "valu_issue_limit": [("ALUEXEC", 10, "VALU"), ("ALUEXEC", 11, "VALU")],
  "valu_long_latency": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 10, "VALU")],
  "valu_transcendent_dependency": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 19, "VALU")],
  "valu_mad64": [("ALUEXEC", 12, "VALU")],
  "valu_f64": [("ALUEXEC", 38, "VALU")],
  "vopd_dependency": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 14, "VALU")],
  "vinterp_independent": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 10, "VALU")],
  "vinterp_dependency": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 15, "VALU")],
  "vinterp_to_valu": [("ALUEXEC", 8, "VALU"), ("ALUEXEC", 13, "VALU")],
  "valu_to_vinterp": [("ALUEXEC", 8, "VALU"), ("ALUEXEC", 14, "VALU")],
  "const_to_vinterp": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 12, "VALU")],
  "wmma_same_block": [("ALUEXEC", 41, "VALU")],
  "wmma_two_blocks": [("ALUEXEC", 42, "VALU")],
  "wmma_independent": [("ALUEXEC", 43, "VALU"), ("ALUEXEC", 77, "VALU")],
  "wmma_dependency": [("ALUEXEC", 43, "VALU"), ("ALUEXEC", 77, "VALU")],
  "vinterp_wmma": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 44, "VALU"), ("ALUEXEC", 78, "VALU")],
  "wmma_vinterp_wmma": [("ALUEXEC", 43, "VALU"), ("ALUEXEC", 44, "VALU"), ("ALUEXEC", 79, "VALU")],
  "delay_valu_dep1": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 11, "VALU")],
  "delay_valu_dep4": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 7, "VALU"), ("ALUEXEC", 8, "VALU"),
                      ("ALUEXEC", 9, "VALU"), ("ALUEXEC", 11, "VALU")],
  "delay_trans_dep1": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 16, "VALU")],
  "delay_fma_accum": [("ALUEXEC", 9, "VALU"), ("ALUEXEC", 14, "VALU")],
  "delay_salu_cycle1": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 4, "SALU")],
  "delay_salu_cycle2": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 4, "SALU")],
  "delay_salu_cycle3": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 4, "SALU")],
  "delay_second_next": [("ALUEXEC", 6, "VALU"), ("ALUEXEC", 7, "VALU"), ("ALUEXEC", 12, "VALU")],
  "lds_roundtrip": [("ALUEXEC", 9, "VALU"), ("VMEMEXEC", 21, "LDS"), ("VMEMEXEC", 23, "LDS")],
  "lds_waitcnt": [("VMEMEXEC", 3, "LDS"), ("ALUEXEC", 41, "VALU")],
  "lds_bpermute_dependency": [("VMEMEXEC", 3, "LDS"), ("ALUEXEC", 17, "VALU")],
  "mixed_alu": [("ALUEXEC", 2, "SALU"), ("ALUEXEC", 7, "VALU"), ("ALUEXEC", 13, "VALU"), ("ALUEXEC", 18, "VALU"),
                ("ALUEXEC", 28, "VALU"), ("ALUEXEC", 33, "SALU")],
  "mixed_lds": [("ALUEXEC", 9, "VALU"), ("VMEMEXEC", 21, "LDS"), ("VMEMEXEC", 23, "LDS"),
                ("ALUEXEC", 62, "VALU"), ("ALUEXEC", 67, "VALU")],
  "mixed_matrix": [("ALUEXEC", 43, "VALU"), ("ALUEXEC", 44, "VALU"), ("ALUEXEC", 50, "VALU"), ("ALUEXEC", 83, "VALU")],
  "single_wave_barrier": [("ALUEXEC", 3, "SALU")],
}

class TestSQTTCycleModel(unittest.TestCase):
  def test_mock_trace_cases(self):
    for name, (case, expected) in CASES.items():
      with self.subTest(name=name):
        got = _run_mock_trace(case)
        self.assertListEqual(_structural_events(got), _structural_events(expected))

  def test_execution_timing(self):
    for name, expected in EXEC_TIMES.items():
      with self.subTest(name=name):
        got = [(e.kind, e.time, e.op or "") for e in _timed_events(_run_mock_trace(CASES[name][0])) if e.kind in {"ALUEXEC", "VMEMEXEC"}]
        self.assertListEqual(got, expected)

  def test_composition_timing(self):
    for name, (case, expected) in COMPOSITION_CASES.items():
      with self.subTest(name=name): self.assertListEqual([e.time for e in _timed_events(_run_mock_trace(case))], expected)

  def test_large_delta_packet(self):
    blob = _run_mock_blob(CASES["lds_roundtrip"][0])
    self.assertTrue(any(isinstance(pkt, TS_DELTA_OR_MARK) for pkt in decode(blob)))

  def test_finalize_idempotent(self):
    builder = RDNA3SQTTTraceBuilder()
    instructions = [s_mov_b32(s[0], 1), v_mov_b32_e32(v[0], s[0])] + [v_mov_b32_e32(v[i], i) for i in range(1, 7)] + \
                   [s_add_u32(s[1], s[2], 3)]
    for inst in instructions: builder.emit(0, inst, None)
    builder.finish(0)
    first = builder.finalize()
    self.assertTrue(any(isinstance(pkt, ALUEXEC) and pkt.src == AluSrc.VALU_SALU for pkt in decode(first)))
    self.assertEqual(first, builder.finalize())

  def test_selected_trace_slot(self):
    selected = {"simd": getenv("SQTT_SIMD_SEL", 0), "sa": getenv("SQTT_SA_SEL", 0), "wgp": getenv("SQTT_WGP_SEL", 0)}
    def packet(**fields):
      raw = WAVESTART.encoding.default
      for name, value in {**selected, "wave": 0, "id7": 0, **fields}.items(): raw = getattr(WAVESTART, name).set(raw, value)
      return WAVESTART.from_raw(raw)
    self.assertTrue(_selected_trace_slot(packet()))
    self.assertFalse(_selected_trace_slot(packet(simd=(selected["simd"] + 1) % 4)))
    self.assertFalse(_selected_trace_slot(packet(sa=1 - selected["sa"])))
    self.assertFalse(_selected_trace_slot(packet(wgp=(selected["wgp"] + 1) % 8)))

  def test_multi_wave_selected_simd(self):
    for name, expected in {"multi_wave_salu": [0] * 7 + [1] * 7 + [2] * 7,
                           "multi_wave_valu": [0] * 7 + [1] * 7,
                           "multi_wave_lds": [0] * 4 + [1] * 4}.items():
      with self.subTest(name=name):
        got = _instruction_events(_run_mock_trace(MULTIWAVE_CASES[name]))
        self.assertEqual([event.wave for event in got], expected)

  def test_multi_wave_execution(self):
    for name, expected in MULTIWAVE_EXEC_TIMES.items():
      with self.subTest(name=name):
        trace = _run_mock_trace(MULTIWAVE_CASES[name])
        base = _instruction_events(trace)[0].time
        exec_events = [event for event in trace if event.kind in {"ALUEXEC", "VMEMEXEC"}]
        self.assertEqual([event.time - base for event in exec_events], expected)
        if name == "multi_wave_lds": self.assertEqual([event.op for event in exec_events], ["LDS"] * 4 + ["LDS_ALT"] * 4 + ["LDS"] * 4)

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
    cases = [(name, case, expected) for name, (case, expected) in CASES.items()] + \
            [(name, case, _run_mock_trace(case)) for name, (case, _) in COMPOSITION_CASES.items()] + \
            [(name, case, _run_mock_trace(case)) for name, case in MULTIWAVE_CASES.items()]
    for name, case, expected in cases:
      with self.subTest(name=name):
        want = _instruction_events(expected)
        got = _instruction_events(_run_hardware_trace(case, min_instructions=len(want)))
        if (msg:=_first_mismatch(got, want, check_time=False, case=case)) is not None: self.fail(msg)

  def test_hardware_trace_lifecycle(self):
    case = CASES["salu_chain"][0]
    got = _run_hardware_trace(case, require_instructions=False)
    self.assertIn("WAVESTART", [e.kind for e in got])
    self.assertIn("WAVEEND", [e.kind for e in got])

  def test_hardware_trace_cycles(self):
    if not getenv("SQTT_CYCLE_STRICT", 0): self.skipTest("set SQTT_CYCLE_STRICT=1 to require exact cycle timestamps")
    cases = [(name, case) for name, (case, _) in CASES.items()] + [(name, case) for name, (case, _) in COMPOSITION_CASES.items()]
    for name, case in cases:
      if case.local_size > 32: continue
      with self.subTest(name=name):
        want = _timed_events(_run_mock_trace(case))
        got = _timed_events(_run_hardware_trace(case, min_instructions=len(_instruction_events(want))))
        if case.normalize_exec_start: got, want = _normalize_exec_start(got), _normalize_exec_start(want)
        if (msg:=_first_mismatch(got, want, check_time=True, case=case)) is not None: self.fail(msg)

  def test_hardware_multi_wave_execution(self):
    if not getenv("SQTT_CYCLE_STRICT", 0): self.skipTest("set SQTT_CYCLE_STRICT=1 to require exact cycle timestamps")
    for name, expected in MULTIWAVE_EXEC_TIMES.items():
      with self.subTest(name=name):
        case = MULTIWAVE_CASES[name]
        trace = _run_hardware_trace(case, min_instructions=len(_instruction_events(_run_mock_trace(case))))
        base = _instruction_events(trace)[0].time
        exec_events = [event for event in trace if event.kind in {"ALUEXEC", "VMEMEXEC"}]
        if name == "multi_wave_lds":
          self.assertEqual(len(exec_events), len(expected))
          self.assertEqual(sorted(event.op for event in exec_events), ["LDS"] * 8 + ["LDS_ALT"] * 4)
        else: self.assertEqual([event.time - base for event in exec_events], expected)

if __name__ == "__main__":
  unittest.main()
