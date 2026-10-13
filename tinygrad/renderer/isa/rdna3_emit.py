"""RDNA3 emit helpers: waitcnt splits, clause safety (post-regalloc).

Bound lightly from `rdna3.py` — keeps the emit scan readable without circular import.
"""
from __future__ import annotations

from tinygrad.renderer.isa.rdna3_defs import TMP_VADDR

def vm_load_count(insts:list) -> int:
  return sum(1 for i in insts if (n:=getattr(i, "op_name", "")) and
             (n.startswith("GLOBAL_LOAD") or n.startswith("SCRATCH_LOAD") or
              n.startswith("BUFFER_LOAD") or n.startswith("FLAT_LOAD")))

def lgkm_store_count(insts:list) -> int:
  return sum(1 for i in insts if (n:=getattr(i, "op_name", "")) and n.startswith("DS_STORE"))

def split_lgkm_scale_and_stores(emitted:list) -> tuple[list, list]:
  i = len(emitted)
  while i > 0 and lgkm_store_count([emitted[i - 1]]): i -= 1
  return emitted[:i], emitted[i:]

def lgkm_load_count(insts:list) -> int:
  return sum(1 for i in insts if (n:=getattr(i, "op_name", "")) and n.startswith("DS_LOAD"))

def split_lgkm_scale_and_loads(emitted:list) -> tuple[list, list]:
  i = len(emitted)
  while i > 0 and lgkm_load_count([emitted[i - 1]]): i -= 1
  return emitted[:i], emitted[i:]

def split_scale_and_loads(emitted:list) -> tuple[list, list]:
  """Split dest-as-addr scalar load emit into addr ALU vs trailing VMEM loads."""
  i = len(emitted)
  while i > 0 and vm_load_count([emitted[i - 1]]): i -= 1
  return emitted[:i], emitted[i:]

def vm_store_count(insts:list) -> int:
  return sum(1 for i in insts if (n:=getattr(i, "op_name", "")) and
             (n.startswith("GLOBAL_STORE") or n.startswith("SCRATCH_STORE") or
              n.startswith("BUFFER_STORE") or n.startswith("FLAT_STORE")))

def split_scale_and_stores(emitted:list) -> tuple[list, list]:
  """Split scratch/global store emit into addr/data ALU vs trailing VMEM stores."""
  i = len(emitted)
  while i > 0 and vm_store_count([emitted[i - 1]]): i -= 1
  return emitted[:i], emitted[i:]

def tmp_vaddr_clause_safe(scales:list, loads:list) -> bool:
  # Multi-slot loads scale into TMP_VADDR. Hoisting ≥2 such scales before TMP-addr loads
  # clobbers addr (load0 sees addr1). One scale + several loads is OK (half×16 B128 pair).
  if sum(1 for s in scales if getattr(s, "vdst", None) == TMP_VADDR) <= 1: return True
  return not any(getattr(ld, "addr", None) == TMP_VADDR for ld in loads)
