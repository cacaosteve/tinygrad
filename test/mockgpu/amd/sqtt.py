"""SQTT packet generation for the AMD mock GPU emulator.

The mock emulator already executes decoded instructions. This helper emits the
SQTT packets for that execution with a small deterministic timing model, so
tests can compare instruction order and simple non-DRAM stalls.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any

from tinygrad.renderer.amd.sqtt import (ALUEXEC, IMMEDIATE, INST, LAYOUT_HEADER, PACKET_TYPES_RDNA3, TS_DELTA_OR_MARK, VALUINST,
  VMEMEXEC, WAVEEND, WAVERDY, WAVESTART, AluSrc, InstOp, MemSrc, PacketType, _build_decode_tables)
from tinygrad.runtime.autogen.amd.cdna import ins as irc
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3
from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4

# Global trace storage: populated by run_asm as raw SQTT blobs, consumed by amdgpu.py.
sqtt_traces: list[bytes] = []

_NIB_COUNTS: dict[type[PacketType], int] = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _encode_raw(pkt_cls: type[PacketType], **kwargs) -> tuple[int, int]:
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  return raw, _NIB_COUNTS[pkt_cls]

def _emit_nibbles(nibbles: list[int], pkt_cls: type[PacketType], **kwargs) -> None:
  raw, nc = _encode_raw(pkt_cls, **kwargs)
  for i in range(nc): nibbles.append((raw >> (i * 4)) & 0xF)

def _nibbles_to_bytes(nibbles: list[int]) -> bytes:
  result = bytearray()
  for i in range(0, len(nibbles), 2): result.append(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4))
  return bytes(result)

def _max_delta(pkt_cls: type[PacketType]) -> int:
  delta_field = getattr(pkt_cls, "delta", None)
  return 0 if delta_field is None else (1 << (delta_field.hi - delta_field.lo + 1)) - 1

_SOPP = (ir3.SOPP, ir4.SOPP, irc.SOPP)
_SALU = (ir3.SOP1, ir3.SOP2, ir3.SOPC, ir3.SOPK, ir4.SOP1, ir4.SOP2, ir4.SOPC, ir4.SOPK, irc.SOP1, irc.SOP2, irc.SOPC, irc.SOPK)
_SMEM = (ir3.SMEM, ir4.SMEM, irc.SMEM)
_VALU = (ir3.VOP1, ir3.VOP2, ir3.VOP3, ir3.VOP3P, ir3.VOPC, ir3.VOPD, ir3.VOP3SD, ir3.VOP3_SDST, ir3.VOP1_SDST,
         ir4.VOP1, ir4.VOP2, ir4.VOP3, ir4.VOP3P, ir4.VOPC, ir4.VOPD, ir4.VOP3SD, ir4.VOP3_SDST, ir4.VOP1_SDST,
         irc.VOP1, irc.VOP2, irc.VOP3, irc.VOP3P, irc.VOP3PX2, irc.VOPC, irc.VOP3SD, irc.VOP3_SDST)
_DS = (ir3.DS, ir4.DS, irc.DS)
_GLOBAL = (ir3.GLOBAL, ir4.VGLOBAL, irc.GLOBAL)
_FLAT = (ir3.FLAT, ir4.VFLAT, irc.FLAT)
_SCRATCH = (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH)

_SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value,
              SOPPOp3.S_DELAY_ALU.value}
_SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                   SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value,
                   SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
            SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
            SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
  _SOPP_IMMEDIATE.add(_op.value)
_SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
if hasattr(SOPPOp4, "S_BARRIER_WAIT"): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
if hasattr(SOPPOp4, "S_BARRIER_LEAVE"): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
_SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}

_VALUT_4_RE = re.compile(r"V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_")
_VALUB_2_RE = re.compile(r"V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64")
_VALUB_4_RE = re.compile(r"V_MAD_(U|I)64")
_VALUB_16_RE = re.compile(r"V_\w+_F64")

def _op_name(inst) -> str:
  if hasattr(inst, "opx"): return f"{inst.opx.name}_{inst.opy.name}"
  return inst.op.name if hasattr(inst, "op") and hasattr(inst.op, "name") else ""

def _valu_op(op_name: str) -> InstOp|None:
  if "CMPX" in op_name: return InstOp.VALU1_WR_EXEC
  if _VALUB_2_RE.search(op_name): return InstOp.VALUB_2
  if _VALUB_4_RE.search(op_name): return InstOp.VALUB_4
  if _VALUB_16_RE.search(op_name): return InstOp.VALUB_16
  if _VALUT_4_RE.search(op_name): return InstOp.VALUT_4
  return None

def _mem_op(t: type, op_name: str) -> InstOp:
  is_store = "STORE" in op_name
  if issubclass(t, _DS): return InstOp.LDS_WR_2 if is_store else InstOp.LDS_RD
  if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1
  if issubclass(t, _FLAT): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
  if issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
  return InstOp.SALU

def _op_duration(op: InstOp|None) -> int:
  if op is None: return 1
  return int(m.group(1)) if (m:=re.search(r"_(\d+)$", op.name)) else 1

def _field_offset(x: Any) -> int|None:
  if x is None: return None
  if hasattr(x, "offset"): return int(x.offset)
  return int(x) if isinstance(x, int) else None

def _src_vgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("src0", "src1", "src2"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None and off >= 256: ret.add(off - (384 if off >= 384 else 256))
  for name in ("vsrc0", "vsrc1", "vsrc2", "data0", "data1", "addr"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None:
      ret.add(off - (384 if off >= 384 else 256) if off >= 256 else off)
  return ret

def _dst_vgprs(inst, op_name: str) -> set[int]:
  ret: set[int] = set()
  if "STORE" not in op_name and (off:=_field_offset(getattr(inst, "vdst", None))) is not None: ret.add(off - 256 if off >= 256 else off)
  if "LOAD" in op_name and (off:=_field_offset(getattr(inst, "vdata", None))) is not None: ret.add(off)
  return ret

def _src_sgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("ssrc0", "ssrc1", "sbase", "soffset", "saddr"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None and off < 128: ret.add(off)
  return ret

def _dst_sgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("sdst", "sdata"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None and off < 128: ret.add(off)
  return ret

@dataclass(frozen=True)
class _TraceInfo:
  pkt_cls: type[PacketType]|None
  kwargs: dict[str, Any] = field(default_factory=dict)
  pipe: str|None = None
  exec_cls: type[PacketType]|None = None
  exec_kwargs: dict[str, Any] = field(default_factory=dict)
  duration: int = 1
  issue_latency: int = 1
  reads_scc_delay: int|None = None
  writes_scc_latency: int|None = None
  barrier: bool = False

@dataclass
class _WaveState:
  issue: int = 2
  salu_ready: int = 2
  valu_ready: int = 2
  lds_ready: int = 2
  vmem_ready: int = 2
  scc_ready: int = 2
  last_time: int = 1
  vgpr_ready: dict[int, int] = field(default_factory=dict)
  vgpr_lds_ready: dict[int, int] = field(default_factory=dict)
  sgpr_ready: dict[int, int] = field(default_factory=dict)

@dataclass(frozen=True)
class _PacketEvent:
  time: int
  seq: int
  pkt_cls: type[PacketType]
  kwargs: dict[str, Any]

class RDNA3SQTTTraceBuilder:
  def __init__(self, workgroup_waves: int=1):
    self.events: list[_PacketEvent] = []
    self.waves: dict[int, _WaveState] = {}
    self.started: set[int] = set()
    self.workgroup_waves = workgroup_waves
    self.seq = 0
    self._add(0, LAYOUT_HEADER, layout=3, sel_a=6)

  def _add(self, time: int, pkt_cls: type[PacketType], **kwargs) -> None:
    self.events.append(_PacketEvent(time, self.seq, pkt_cls, kwargs))
    self.seq += 1

  def _wave(self, wave_id: int) -> _WaveState:
    if wave_id not in self.waves: self.waves[wave_id] = _WaveState()
    if wave_id not in self.started:
      self._add(1, WAVESTART, simd=0, wgp=0, wave=wave_id & 0x1F, id7=wave_id)
      self.started.add(wave_id)
    return self.waves[wave_id]

  def _classify(self, inst, branch_taken: bool|None) -> _TraceInfo:
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, "op") else 0, _op_name(inst)
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return _TraceInfo(None)
      if inst_op in _SOPP_IMMEDIATE: return _TraceInfo(IMMEDIATE)
      if inst_op in _SOPP_BARRIER:
        if self.workgroup_waves <= 1: return _TraceInfo(IMMEDIATE)
        return _TraceInfo(INST, {"op": InstOp.BARRIER}, barrier=True)
      if inst_op in _SOPP_BRANCH:
        return _TraceInfo(INST, {"op": InstOp.JUMP if branch_taken else InstOp.JUMP_NO}, pipe="salu",
                          issue_latency=10 if branch_taken else 1, reads_scc_delay=0 if branch_taken else 1)
      return _TraceInfo(INST, {"op": InstOp.SALU}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU},
                        writes_scc_latency=9 if op_name.startswith("S_CMP") else None)
    if issubclass(inst_type, _SALU):
      return _TraceInfo(INST, {"op": InstOp.SALU}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU},
                        writes_scc_latency=9 if op_name.startswith("S_CMP") else None)
    if issubclass(inst_type, _VALU):
      op = _valu_op(op_name)
      if op is None: return _TraceInfo(VALUINST, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU})
      return _TraceInfo(INST, {"op": op}, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU}, duration=_op_duration(op))
    if issubclass(inst_type, _SMEM):
      return _TraceInfo(INST, {"op": InstOp.SMEM_RD}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU})
    op = _mem_op(inst_type, op_name)
    if op.name.startswith("LDS"):
      return _TraceInfo(INST, {"op": op}, pipe="lds", exec_cls=VMEMEXEC, exec_kwargs={"src": MemSrc.LDS}, duration=_op_duration(op))
    if op.name.startswith(("SGMEM", "FLAT")):
      return _TraceInfo(INST, {"op": op}, pipe="vmem", exec_cls=VMEMEXEC, exec_kwargs={"src": MemSrc.VMEM}, duration=_op_duration(op))
    return _TraceInfo(INST, {"op": op}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU})

  def emit(self, wave_id: int, inst, branch_taken: bool|None) -> None:
    info = self._classify(inst, branch_taken)
    if info.pkt_cls is None: return
    st, wave = self._wave(wave_id), wave_id & 0x1F
    src_vgprs = _src_vgprs(inst)
    vgpr_ready = st.vgpr_lds_ready if info.pipe == "lds" else st.vgpr_ready
    src_ready = max([vgpr_ready.get(r, st.vgpr_ready.get(r, 0)) for r in src_vgprs] +
                    [st.sgpr_ready.get(r, 0) for r in _src_sgprs(inst)] +
                    ([] if info.reads_scc_delay is None else [st.scc_ready + info.reads_scc_delay]) + [0])
    pipe_ready = getattr(st, f"{info.pipe}_ready") if info.pipe else 0
    issue = max(st.issue, pipe_ready, src_ready)
    kwargs = {"wave": wave, **info.kwargs} if info.pkt_cls in (INST, IMMEDIATE, VALUINST) else info.kwargs
    self._add(issue, info.pkt_cls, **kwargs)

    exec_time = issue + info.duration if info.pipe in {"valu", "lds", "vmem"} else issue
    if info.exec_cls is not None: self._add(exec_time, info.exec_cls, **info.exec_kwargs)
    if info.barrier: self._add(issue + 1, WAVERDY, mask=1 << wave)

    if info.pipe: setattr(st, f"{info.pipe}_ready", issue + info.issue_latency)
    ready = issue + info.duration
    for reg in _dst_vgprs(inst, _op_name(inst)):
      st.vgpr_ready[reg] = ready
      if info.pipe == "valu": st.vgpr_lds_ready[reg] = issue + 18
    for reg in _dst_sgprs(inst): st.sgpr_ready[reg] = ready
    if info.writes_scc_latency is not None: st.scc_ready = issue + info.writes_scc_latency
    st.issue, st.last_time = issue + info.issue_latency, max(st.last_time, exec_time, ready)

  def finish(self, wave_id: int) -> None:
    if wave_id not in self.started: return
    st, wave = self.waves[wave_id], wave_id & 0x1F
    self._add(max(st.issue, st.last_time + 1), WAVEEND, simd=0, wgp=0, wave=wave)

  def finalize(self) -> bytes:
    nibbles: list[int] = []
    current_time = 0
    for event in sorted(self.events, key=lambda e: (e.time, e.seq)):
      delta = event.time - current_time
      if delta < 0: raise RuntimeError("SQTT events must be timestamp sorted")
      max_delta = _max_delta(event.pkt_cls)
      if delta > max_delta:
        tail_delta = min(delta, max_delta)
        _emit_nibbles(nibbles, TS_DELTA_OR_MARK, delta=delta - tail_delta, pl=0, rt=0)
        current_time += delta - tail_delta
        delta = tail_delta
      kwargs = event.kwargs if getattr(event.pkt_cls, "delta", None) is None else {**event.kwargs, "delta": delta}
      _emit_nibbles(nibbles, event.pkt_cls, **kwargs)
      current_time += delta
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return _nibbles_to_bytes(nibbles)
