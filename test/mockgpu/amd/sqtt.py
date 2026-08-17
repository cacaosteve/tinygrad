"""SQTT packet generation for the AMD mock GPU emulator.

The mock emulator already executes decoded instructions. This helper emits the
SQTT packets for that execution with a small deterministic timing model, so
tests can compare instruction order and simple non-DRAM stalls.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
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
_VINTERP = (ir3.VINTERP, ir4.VINTERP)
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
_DS_ATOMIC_RE = re.compile(r"DS_(ADD|SUB|RSUB|INC|DEC|MIN|MAX|AND|OR|XOR|MSKOR)(_|$)")
_SALU_3_CYCLE = {"S_MUL_I32", "S_MULK_I32", "S_MUL_HI_I32", "S_MUL_HI_U32"}

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
  if issubclass(t, _DS):
    if "PERMUTE" in op_name: return InstOp.LDS_WR_2
    if is_store and "_2ADDR" in op_name: return InstOp.LDS_WR_5 if "_B64" in op_name else InstOp.LDS_WR_3
    if "CMPSTORE" in op_name: return InstOp.LDS_WR_5 if ("_B64" in op_name or "_F64" in op_name) else InstOp.LDS_WR_3
    if _DS_ATOMIC_RE.match(op_name): return InstOp.LDS_WR_3 if re.search(r"_(B|U|I|F)64", op_name) else InstOp.LDS_WR_2
    if not is_store: return InstOp.LDS_RD
    if "_ADDTID" in op_name: return InstOp.LDS_WR_1
    if "_B128" in op_name: return InstOp.LDS_WR_5
    if "_B96" in op_name: return InstOp.LDS_WR_4
    if "_B64" in op_name: return InstOp.LDS_WR_3
    return InstOp.LDS_WR_2
  if issubclass(t, _GLOBAL): return InstOp.SGMEM_WR_2 if is_store else InstOp.SGMEM_RD_1
  if issubclass(t, _FLAT): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
  if issubclass(t, _SCRATCH): return InstOp.FLAT_WR_3 if is_store else InstOp.FLAT_RD_2
  return InstOp.SALU

def _op_duration(op: InstOp|None) -> int:
  if op is None: return 1
  return int(m.group(1)) if (m:=re.search(r"_(\d+)$", op.name)) else 1

def _valu_latencies(op: InstOp) -> tuple[int, int, int]:
  duration = _op_duration(op)
  if op == InstOp.VALUB_16: return duration, 32, 32
  if op.name.startswith("VALUB_"): return duration, duration, duration
  return duration, 1, 1

def _lds_lgkm_latency(op: InstOp, pending: bool) -> int:
  if op == InstOp.LDS_RD: return 34 if pending else 31
  if op == InstOp.LDS_WR_1: return 31
  if op == InstOp.LDS_WR_2: return 33
  if op == InstOp.LDS_WR_3: return 35
  if op == InstOp.LDS_WR_4: return 38
  if op == InstOp.LDS_WR_5: return 39
  return _op_duration(op)

def _lds_exec_latency(op: InstOp) -> int:
  if op in {InstOp.LDS_WR_1, InstOp.LDS_WR_3, InstOp.LDS_WR_4, InstOp.LDS_WR_5}: return 3
  return _op_duration(op)

def _field_offset(x: Any) -> int|None:
  if x is None: return None
  if hasattr(x, "offset"): return int(x.offset)
  return int(x) if isinstance(x, int) else None

def _vgpr_offsets(x: Any) -> set[int]:
  if (off:=_field_offset(x)) is None or off < 256: return set()
  base = off - (384 if off >= 384 else 256)
  return set(range(base, base + int(getattr(x, "sz", 1))))

def _wmma_exec_latency(inst) -> int:
  blocks = {_field_offset(getattr(inst, name, None)) for name in ("vdst", "src0", "src1", "src2")}
  return min(43, 40 + len(blocks - {None}))

def _src_vgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("src0", "src1", "src2", "srcx0", "srcy0"):
    ret.update(_vgpr_offsets(getattr(inst, name, None)))
  for name in ("vsrc0", "vsrc1", "vsrc2", "vsrcx1", "vsrcy1", "data0", "data1", "addr"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None:
      ret.add(off - (384 if off >= 384 else 256) if off >= 256 else off)
  if "FMAC" in _op_name(inst) or "DOT2ACC" in _op_name(inst): ret.update(_vgpr_offsets(getattr(inst, "vdst", None)))
  return ret

def _dst_vgprs(inst, op_name: str) -> set[int]:
  ret: set[int] = set()
  if "STORE" not in op_name:
    for name in ("vdst", "vdstx", "vdsty"):
      x = getattr(inst, name, None)
      if (off:=_field_offset(x)) is not None:
        base = off - 256 if off >= 256 else off
        ret.update(range(base, base + int(getattr(x, "sz", 1))))
  if "LOAD" in op_name and (off:=_field_offset(getattr(inst, "vdata", None))) is not None: ret.add(off)
  return ret

def _src_sgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("src0", "src1", "src2", "ssrc0", "ssrc1", "sbase", "soffset", "saddr"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None and off < 128: ret.add(off)
  return ret

def _dst_sgprs(inst) -> set[int]:
  ret: set[int] = set()
  for name in ("sdst", "sdata"):
    if (off:=_field_offset(getattr(inst, name, None))) is not None and off < 128: ret.add(off)
  return ret

def _writes_vcc(inst) -> bool:
  return any(_field_offset(getattr(inst, name, None)) in {106, 107} for name in ("sdst", "sdata"))

def _writes_exec(inst, op_name: str) -> bool:
  return "SAVEEXEC" in op_name or "WREXEC" in op_name or \
    any(_field_offset(getattr(inst, name, None)) in {126, 127} for name in ("sdst", "sdata"))

@dataclass(frozen=True)
class _TraceInfo:
  pkt_cls: type[PacketType]|None
  kwargs: dict[str, Any] = field(default_factory=dict)
  pipe: str|None = None
  exec_cls: type[PacketType]|None = None
  exec_kwargs: dict[str, Any] = field(default_factory=dict)
  duration: int = 1
  issue_latency: int = 1
  dst_latency: int|None = None
  exec_latency: int = 0
  exec_release: int = 1
  exec_block: int = 0
  forward_latency: int = 0
  matrix: bool = False
  interp: bool = False
  trans: bool = False
  nop: bool = False
  vgpr_read_latency: int|None = None
  reads_scc_delay: int|None = None
  reads_vcc: bool = False
  reads_exec: bool = False
  writes_scc_latency: int|None = None
  writes_vcc_latency: int|None = None
  writes_exec_latency: int|None = None
  lgkm_latency: int|None = None
  wait_lgkm: bool = False
  wait_lgkm_extra: int = 0
  immediate_delay: int = 0
  barrier: bool = False

@dataclass
class _WaveState:
  issue: int = 2
  salu_ready: int = 2
  valu_ready: int = 2
  trans_ready: int = 2
  lds_ready: int = 2
  vmem_ready: int = 2
  control_ready: int = 2
  scc_ready: int = 2
  vcc_ready: int = 2
  exec_ready: int = 2
  lgkm_ready: int = 2
  immediate_ready: int = 0
  salu_exec_ready: int = 2
  valu_exec_ready: int = 2
  lds_exec_ready: int = 2
  vmem_exec_ready: int = 2
  control_exec_ready: int = 2
  matrix_start_ready: int = 0
  matrix_interp_ready: int = 0
  sgpr_valu_read_ready: int = 0
  trans_warm: bool = False
  had_lds: bool = False
  last_time: int = 1
  vgpr_ready: dict[int, int] = field(default_factory=dict)
  vgpr_lds_ready: dict[int, int] = field(default_factory=dict)
  vgpr_lds_pending: dict[int, bool] = field(default_factory=dict)
  vgpr_exec_ready: dict[int, int] = field(default_factory=dict)
  vgpr_forward_ready: dict[int, int] = field(default_factory=dict)
  vgpr_interp_ready: dict[int, int] = field(default_factory=dict)
  vgpr_matrix_ready: dict[int, int] = field(default_factory=dict)
  vgpr_exec_event: dict[int, int] = field(default_factory=dict)
  vgpr_producer_interp: dict[int, bool] = field(default_factory=dict)
  vgpr_producer_trans: dict[int, bool] = field(default_factory=dict)
  vgpr_producer_shiftable: dict[int, bool] = field(default_factory=dict)
  shifted_exec_events: set[int] = field(default_factory=set)
  sgpr_ready: dict[int, int] = field(default_factory=dict)
  sgpr_exec_ready: dict[int, int] = field(default_factory=dict)
  valu_exec_history: list[int] = field(default_factory=list)
  trans_exec_history: list[tuple[int, bool]] = field(default_factory=list)
  salu_exec_history: list[int] = field(default_factory=list)
  pending_delays: list[tuple[int, int]] = field(default_factory=list)

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

  def _add(self, time: int, pkt_cls: type[PacketType], **kwargs) -> int:
    self.events.append(_PacketEvent(time, self.seq, pkt_cls, kwargs))
    self.seq += 1
    return len(self.events) - 1

  def _wave(self, wave_id: int) -> _WaveState:
    if wave_id not in self.waves: self.waves[wave_id] = _WaveState()
    if wave_id not in self.started:
      self._add(1, WAVESTART, simd=0, wgp=0, wave=wave_id & 0x1F, id7=wave_id)
      self.started.add(wave_id)
    return self.waves[wave_id]

  @staticmethod
  def _delay_ready(st: _WaveState, dep: int) -> int:
    if 1 <= dep <= 4 and len(st.valu_exec_history) >= dep: return st.valu_exec_history[-dep] - 1
    if 5 <= dep <= 7 and len(st.trans_exec_history) >= dep - 4:
      exec_time, fast = st.trans_exec_history[4 - dep]
      return exec_time + (2 if fast else 1)
    if dep == 9 and st.salu_exec_history: return st.salu_exec_history[-1]
    return 0

  def _shift_exec_events(self, st: _WaveState, events: set[int]) -> None:
    for event_idx in events - st.shifted_exec_events:
      self.events[event_idx] = replace(self.events[event_idx], time=self.events[event_idx].time - 1)
      st.shifted_exec_events.add(event_idx)
      for reg, reg_event_idx in st.vgpr_exec_event.items():
        if reg_event_idx != event_idx: continue
        st.vgpr_exec_ready[reg] -= 1
        st.vgpr_forward_ready[reg] -= 1
        st.vgpr_interp_ready[reg] -= 1

  def _shift_cross_interp_producers(self, st: _WaveState, src_vgprs: set[int], interp: bool) -> None:
    events = {st.vgpr_exec_event[r] for r in src_vgprs if r in st.vgpr_exec_event and
              st.vgpr_producer_interp.get(r) != interp and st.vgpr_producer_shiftable.get(r, False)}
    self._shift_exec_events(st, events)

  def _classify(self, inst, branch_taken: bool|None) -> _TraceInfo:
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, "op") else 0, _op_name(inst)
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return _TraceInfo(None)
      if inst_op in _SOPP_IMMEDIATE:
        waits_lgkm = inst_op in {SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAIT_IDLE.value} and getattr(inst, "simm16", 0) == 0
        immediate_delay = getattr(inst, "simm16", 0) if inst_op == SOPPOp3.S_NOP.value else 0
        return _TraceInfo(IMMEDIATE, wait_lgkm=waits_lgkm, wait_lgkm_extra=1 if inst_op == SOPPOp3.S_WAIT_IDLE.value and waits_lgkm else 0,
                          immediate_delay=immediate_delay, nop=inst_op == SOPPOp3.S_NOP.value)
      if inst_op in _SOPP_BARRIER:
        if self.workgroup_waves <= 1: return _TraceInfo(IMMEDIATE)
        return _TraceInfo(INST, {"op": InstOp.BARRIER}, barrier=True)
      if inst_op in _SOPP_BRANCH:
        reads_vcc = inst_op in {SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value}
        reads_exec = inst_op in {SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}
        return _TraceInfo(INST, {"op": InstOp.JUMP if branch_taken else InstOp.JUMP_NO}, pipe="salu",
                          issue_latency=10 if branch_taken else (3 if reads_vcc or reads_exec else 1),
                          exec_block=14 if branch_taken and (getattr(inst, "simm16", 0) & 0x8000) else 0,
                          reads_scc_delay=None if reads_vcc or reads_exec else (0 if branch_taken else 1),
                          reads_vcc=reads_vcc, reads_exec=reads_exec)
      return _TraceInfo(INST, {"op": InstOp.SALU}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU},
                        exec_latency=2, forward_latency=2, writes_scc_latency=9 if op_name.startswith("S_CMP") else None)
    if issubclass(inst_type, _SALU):
      if op_name == "S_CALL_B64":
        return _TraceInfo(INST, {"op": InstOp.CALL}, pipe="control", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU},
                          issue_latency=28, exec_latency=2)
      if op_name in {"S_SETPC_B64", "S_SWAPPC_B64"}:
        return _TraceInfo(INST, {"op": InstOp.CALL}, pipe="control", issue_latency=28)
      writes_exec = _writes_exec(inst, op_name)
      salu_op = InstOp.SALU_WR_EXEC if writes_exec else InstOp.SALU
      return _TraceInfo(INST, {"op": salu_op}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU},
                        exec_latency=3 if op_name in _SALU_3_CYCLE else 2, forward_latency=2,
                        writes_scc_latency=9 if op_name.startswith("S_CMP") else None,
                        writes_vcc_latency=7 if _writes_vcc(inst) else None, writes_exec_latency=7 if writes_exec else None)
    if issubclass(inst_type, _VINTERP):
      return _TraceInfo(INST, {"op": InstOp.VINTERP}, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU},
                        exec_latency=9, forward_latency=5, interp=True)
    if issubclass(inst_type, _VALU):
      valu_op = _valu_op(op_name)
      if valu_op is None:
        if "WMMA" in op_name:
          return _TraceInfo(VALUINST, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU},
                            exec_latency=_wmma_exec_latency(inst), forward_latency=2, matrix=True)
        return _TraceInfo(VALUINST, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU},
                          exec_latency=6, vgpr_read_latency=9, forward_latency=5,
                          writes_vcc_latency=18 if op_name.startswith("V_CMP") else None)
      duration, issue_latency, dst_latency = _valu_latencies(valu_op)
      exec_latency = {InstOp.VALUT_4: 9, InstOp.VALUB_2: 10, InstOp.VALUB_4: 12, InstOp.VALUB_16: 38,
                      InstOp.VALU1_WR_EXEC: 9}[valu_op]
      return _TraceInfo(INST, {"op": valu_op}, pipe="valu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.VALU},
                        duration=duration, issue_latency=issue_latency, dst_latency=dst_latency,
                        exec_latency=exec_latency, exec_release=4 if valu_op == InstOp.VALUB_16 else 1,
                        forward_latency=10 if valu_op == InstOp.VALUT_4 else 5,
                        writes_exec_latency=18 if valu_op == InstOp.VALU1_WR_EXEC else None, trans=valu_op == InstOp.VALUT_4)
    if issubclass(inst_type, _SMEM):
      return _TraceInfo(INST, {"op": InstOp.SMEM_RD}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU})
    mem_op = _mem_op(inst_type, op_name)
    if mem_op.name.startswith("LDS"):
      is_permute = "PERMUTE" in op_name
      is_2addr_load = "_2ADDR" in op_name and "LOAD" in op_name
      is_rtn_atomic = _DS_ATOMIC_RE.match(op_name) is not None and "_RTN_" in op_name
      load_lgkm_latency = None
      if is_2addr_load: load_lgkm_latency = 35 if "_B64" in op_name else 32
      elif "LOAD" in op_name:
        if "_B128" in op_name: load_lgkm_latency = 37
        elif "_B96" in op_name: load_lgkm_latency = 36
        elif "_B64" in op_name: load_lgkm_latency = 33
      if is_rtn_atomic and re.search(r"_(B|U|I|F)64", op_name): load_lgkm_latency = 36
      wide_load = load_lgkm_latency is not None and "_B32" not in op_name
      lds_dst_latency: int|None = 8 if is_permute or (is_rtn_atomic and mem_op == InstOp.LDS_WR_2) else \
        (9 if is_rtn_atomic and mem_op == InstOp.LDS_WR_3 else (7 if mem_op == InstOp.LDS_RD else None))
      return _TraceInfo(INST, {"op": mem_op}, pipe="lds", exec_cls=VMEMEXEC, exec_kwargs={"src": MemSrc.LDS},
                        duration=3 if is_permute or is_2addr_load or wide_load else _lds_exec_latency(mem_op), dst_latency=lds_dst_latency,
                        exec_latency=3, exec_release=2, lgkm_latency=35 if is_permute else load_lgkm_latency)
    if mem_op.name.startswith(("SGMEM", "FLAT")):
      return _TraceInfo(INST, {"op": mem_op}, pipe="vmem", exec_cls=VMEMEXEC, exec_kwargs={"src": MemSrc.VMEM}, duration=_op_duration(mem_op))
    return _TraceInfo(INST, {"op": mem_op}, pipe="salu", exec_cls=ALUEXEC, exec_kwargs={"src": AluSrc.SALU})

  def emit(self, wave_id: int, inst, branch_taken: bool|None) -> None:
    inst_op = inst.op.value if hasattr(inst, "op") else 0
    if isinstance(inst, _SOPP) and inst_op == SOPPOp3.S_DELAY_ALU.value:
      st = self._wave(wave_id)
      st.trans_warm = False
      imm = getattr(inst, "simm16", 0)
      if (dep0:=imm & 0xF): st.pending_delays.append((0, dep0))
      if (dep1:=(imm >> 7) & 0xF): st.pending_delays.append(((imm >> 4) & 0x7, dep1))
      return
    info = self._classify(inst, branch_taken)
    if info.pkt_cls is None: return
    st, wave = self._wave(wave_id), wave_id & 0x1F
    src_vgprs, src_sgprs = _src_vgprs(inst), _src_sgprs(inst)
    dst_vgprs, dst_sgprs = _dst_vgprs(inst, _op_name(inst)), _dst_sgprs(inst)
    vgpr_ready = st.vgpr_lds_ready if info.pipe == "lds" else st.vgpr_ready
    src_ready = max([vgpr_ready.get(r, st.vgpr_ready.get(r, 0)) for r in src_vgprs] +
                    [st.sgpr_ready.get(r, 0) for r in src_sgprs] +
                    ([] if info.reads_scc_delay is None else [st.scc_ready + info.reads_scc_delay]) +
                    ([st.vcc_ready] if info.reads_vcc else []) + ([st.exec_ready] if info.reads_exec else []) +
                    ([st.lgkm_ready + info.wait_lgkm_extra] if info.wait_lgkm else []) + [0])
    pipe_ready = getattr(st, f"{info.pipe}_ready") if info.pipe else 0
    if info.trans: pipe_ready = max(pipe_ready, st.trans_ready)
    immediate_ready = st.immediate_ready + info.immediate_delay if info.pkt_cls == IMMEDIATE and st.immediate_ready else 0
    active_delays = [dep for skip, dep in st.pending_delays if skip == 0]
    delay_ready = max([self._delay_ready(st, dep) for dep in active_delays] + [0])
    base_issue = max(st.issue, pipe_ready, src_ready, immediate_ready, delay_ready)
    sgpr_read_ready = st.sgpr_valu_read_ready if info.pipe == "salu" and (src_sgprs or dst_sgprs) else 0
    salu_read_stall, issue = sgpr_read_ready > base_issue, max(base_issue, sgpr_read_ready)
    lds_pending = info.pipe == "lds" and st.lgkm_ready > issue
    kwargs = {"wave": wave, **info.kwargs} if info.pkt_cls in (INST, IMMEDIATE, VALUINST) else info.kwargs
    self._add(issue, info.pkt_cls, **kwargs)

    exec_time = issue
    if info.exec_cls is not None and info.pipe is not None:
      exec_ready = getattr(st, f"{info.pipe}_exec_ready")
      if info.pipe == "salu":
        dep_ready = max([st.sgpr_exec_ready.get(r, 0) + info.forward_latency for r in src_sgprs if r in st.sgpr_exec_ready] + [0])
        exec_time = max(issue + info.exec_latency + (2 if salu_read_stall and src_sgprs else 0), exec_ready, dep_ready)
      elif info.pipe == "valu":
        if not info.trans and src_vgprs and any(5 <= dep <= 7 for dep in active_delays): latency = 8
        elif not info.trans and any(st.vgpr_lds_pending.get(r, False) for r in src_vgprs): latency = 8
        elif info.trans and st.trans_warm: latency = 7
        else: latency = info.vgpr_read_latency if src_vgprs and info.vgpr_read_latency is not None else info.exec_latency
        if not info.matrix: self._shift_cross_interp_producers(st, src_vgprs, info.interp)
        if info.trans and st.had_lds:
          events = {st.vgpr_exec_event[r] for r in src_vgprs if r in st.vgpr_exec_event and
                    not st.vgpr_producer_trans.get(r, False) and st.vgpr_producer_shiftable.get(r, False)}
          self._shift_exec_events(st, events)
        interp_sources = any(st.vgpr_producer_interp.get(r, False) for r in src_vgprs)
        if not info.interp and not info.matrix and st.matrix_start_ready > issue: st.matrix_start_ready += 4 if interp_sources else 2
        dep_map = st.vgpr_matrix_ready if info.matrix else (st.vgpr_interp_ready if info.interp else st.vgpr_forward_ready)
        dep_ready = max([dep_map.get(r, 0) for r in src_vgprs] + [0])
        if info.matrix:
          matrix_start = max(issue, st.matrix_start_ready)
          exec_time = max(matrix_start + latency, dep_ready)
          st.matrix_start_ready = matrix_start + 32 + latency - 41
          st.matrix_interp_ready = matrix_start + 44
        elif info.interp:
          exec_time = max(issue + latency, exec_ready, dep_ready, st.matrix_interp_ready)
          if st.matrix_start_ready > issue: st.matrix_start_ready += 2
        else: exec_time = max(issue + latency, exec_ready, dep_ready)
      else: exec_time = max(issue + info.exec_latency, exec_ready)
      setattr(st, f"{info.pipe}_exec_ready", exec_time + info.exec_release)
    if info.pipe is not None and info.exec_block:
      setattr(st, f"{info.pipe}_exec_ready", max(getattr(st, f"{info.pipe}_exec_ready"), issue + info.exec_block))
    exec_event = self._add(exec_time, info.exec_cls, **info.exec_kwargs) if info.exec_cls is not None else None
    if info.barrier: self._add(issue + 1, WAVERDY, mask=1 << wave)

    if info.pipe:
      setattr(st, f"{info.pipe}_ready", issue + info.issue_latency)
      if info.trans: st.trans_ready = issue + 4
      if info.pipe in {"valu", "lds", "vmem"}: st.immediate_ready = max(st.immediate_ready, issue + 3)
    ready = issue + (info.duration if info.dst_latency is None else info.dst_latency)
    if lds_pending and info.dst_latency is not None: ready += 1
    if info.pipe == "lds" and dst_vgprs: st.valu_ready = max(st.valu_ready, ready)
    natural_exec_latency = info.vgpr_read_latency if src_vgprs and info.vgpr_read_latency is not None else info.exec_latency
    producer_shiftable = (info.interp or (info.vgpr_read_latency is not None and bool(src_vgprs))) and exec_time == issue + natural_exec_latency
    producer_forward_latency = info.forward_latency + (1 if info.interp and not producer_shiftable else 0)
    if info.vgpr_read_latency is not None and not src_vgprs: producer_forward_latency += 1
    for reg in dst_vgprs:
      st.vgpr_ready[reg] = ready
      st.vgpr_exec_ready[reg] = exec_time
      st.vgpr_forward_ready[reg] = exec_time + producer_forward_latency
      st.vgpr_interp_ready[reg] = exec_time + 6
      if info.matrix: st.vgpr_matrix_ready[reg] = exec_time + 34
      if exec_event is not None:
        st.vgpr_exec_event[reg] = exec_event
        st.vgpr_producer_interp[reg] = info.interp
        st.vgpr_producer_trans[reg] = info.trans
        st.vgpr_producer_shiftable[reg] = producer_shiftable
      if info.pipe == "valu": st.vgpr_lds_ready[reg] = issue + 18
      st.vgpr_lds_pending[reg] = lds_pending
    for reg in dst_sgprs:
      st.sgpr_ready[reg] = ready
      st.sgpr_exec_ready[reg] = exec_time
    if info.writes_scc_latency is not None: st.scc_ready = issue + info.writes_scc_latency
    if info.writes_vcc_latency is not None: st.vcc_ready = issue + info.writes_vcc_latency
    if info.writes_exec_latency is not None: st.exec_ready = issue + info.writes_exec_latency
    if info.pipe == "lds":
      st.had_lds = True
      op = info.kwargs.get("op")
      if isinstance(op, InstOp):
        latency = info.lgkm_latency if info.lgkm_latency is not None else _lds_lgkm_latency(op, st.lgkm_ready > issue)
        st.lgkm_ready = max(st.lgkm_ready, issue + latency)
    if info.pipe == "valu" and info.exec_cls is not None:
      if src_sgprs: st.sgpr_valu_read_ready = max(st.sgpr_valu_read_ready, exec_time)
      if info.trans: st.trans_exec_history.append((exec_time, st.trans_warm))
      else: st.valu_exec_history.append(exec_time)
    if info.pipe == "salu" and info.exec_cls is not None: st.salu_exec_history.append(exec_time)
    st.pending_delays = [(skip - 1, dep) for skip, dep in st.pending_delays if skip > 0]
    st.trans_warm = info.nop
    st.issue, st.last_time = issue + info.issue_latency, max(st.last_time, exec_time, ready)

  def finish(self, wave_id: int) -> None:
    if wave_id not in self.started: return
    st, wave = self.waves[wave_id], wave_id & 0x1F
    self._add(max(st.issue, st.last_time + 1, st.lgkm_ready + 1), WAVEEND, simd=0, wgp=0, wave=wave)

  def finalize(self) -> bytes:
    nibbles: list[int] = []
    current_time = 0
    events = list(self.events)
    drop: set[int] = set()
    alu_by_time: dict[int, dict[AluSrc, list[int]]] = {}
    for i, event in enumerate(events):
      if event.pkt_cls is ALUEXEC: alu_by_time.setdefault(event.time, {}).setdefault(event.kwargs["src"], []).append(i)
    for slots in alu_by_time.values():
      if len(slots.get(AluSrc.SALU, [])) == 1 and len(slots.get(AluSrc.VALU, [])) == 1:
        salu, valu = slots[AluSrc.SALU][0], slots[AluSrc.VALU][0]
        events[salu] = replace(events[salu], kwargs={"src": AluSrc.VALU_SALU})
        drop.add(valu)
    for event in sorted((e for i, e in enumerate(events) if i not in drop), key=lambda e: (e.time, e.pkt_cls in (ALUEXEC, VMEMEXEC), e.seq)):
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
