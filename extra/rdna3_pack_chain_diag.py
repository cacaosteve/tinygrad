#!/usr/bin/env python3
"""Diagnose pack_chain_bias_cons reserve failures on flash DIRECT."""
from __future__ import annotations
import os
from collections import Counter

os.environ.update({
  "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "PARALLEL": "0",
  "CACHELEVEL": "0", "CCACHE": "0", "AMD_SOFT_SYSFS": "1",
})
if "CLUSTER" in os.environ:
  os.environ["AMD_CLUSTER_PACK_LLOAD"] = os.environ["CLUSTER"]

from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp, Ops
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
import tinygrad.renderer.isa.rdna3 as r
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
from tinygrad.renderer.isa.rdna3_defs import Register

stats: dict = {
  "ranked_empty_si0": 0, "ranked_ok": 0, "need": Counter(), "pool_n": Counter(),
  "first_si": Counter(), "fail_occ": 0, "fail_pool": 0, "fail_base_neg": 0,
  "samples": [], "si0_calls": 0,
}

_orig = AMDRenderer.pack_chain_bias_cons

def instrumented(self, u, cons, live):
  if not getenv("AMD_PACK_LLOAD_CHAIN", 1): return cons
  if u.op is not Ops.INS or r._iop(u) not in (AMDOps.LLOAD, AMDOps.LOAD): return cons
  if r._elem_count(u) < 2 or not isinstance(u.tag, tuple): return cons
  idx = getattr(self, "_pack_lload_index", None)
  if idx is None:
    idx = {}
    for pu in getattr(self, "_pack_chain_uops", ()):
      if pu.op is Ops.INS and r._iop(pu) is AMDOps.PACK_F16 and r._pack_f16_is_vec_load(pu) and len(pu.src) >= 2:
        for si, s in enumerate(pu.src):
          idx[s] = (pu, si)
    self._pack_lload_index = idx
  got = idx.get(u)
  if got is None: return _orig(self, u, cons, live)
  pack, si = got
  bases = getattr(self, "_pack_chain_bases", None)
  if bases is None: bases = self._pack_chain_bases = {}
  need = sum(r._reg_slots(s) for s in pack.src)
  off = sum(r._reg_slots(s) for s in pack.src[:si])
  occupied: set[int] = set()
  for p in live.values():
    if isinstance(p, Register): occupied.add(p.index)
  if id(pack) not in bases:
    pool_idxs = {rr.index for rr in u.tag[0].cons}
    ranked: list[int] = []
    n_occ = n_pool = n_neg = 0
    for rr in cons:
      base = rr.index - off
      if base < 0: n_neg += 1; continue
      if any((base + k) in occupied for k in range(need)): n_occ += 1; continue
      if not all((base + k) in pool_idxs for k in range(need)): n_pool += 1; continue
      ranked.append(base)
    if si == 0: stats["si0_calls"] += 1
    if not ranked:
      if si == 0:
        stats["ranked_empty_si0"] += 1
        stats["need"][need] += 1
        stats["pool_n"][len(pool_idxs)] += 1
        stats["fail_occ"] += n_occ
        stats["fail_pool"] += n_pool
        stats["fail_base_neg"] += n_neg
        if len(stats["samples"]) < 3:
          p_sorted = sorted(pool_idxs)
          stats["samples"].append({
            "need": need, "off": off, "cons_n": len(cons), "occ_n": len(occupied),
            "pool_n": len(pool_idxs), "pool_lo": p_sorted[:6], "pool_hi": p_sorted[-6:],
            "cons_lo": sorted(rr.index for rr in cons)[:6],
            "n_occ": n_occ, "n_pool": n_pool, "n_neg": n_neg,
          })
    else:
      stats["ranked_ok"] += 1
      stats["first_si"][si] += 1
  return _orig(self, u, cons, live)

def main() -> None:
  AMDRenderer.pack_chain_bias_cons = instrumented
  DEV.value = "AMD:AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  ren = AMDRenderer(Target("AMD", arch="gfx1100"))
  with _flash_direct_compile_env(flash_acc_small=True), Context(BEAM=0):
    prg = to_program(_amd_flash_attention(
      UOp.placeholder((32, 32, 128), dtypes.float32, 0),
      UOp.placeholder((32, 32, 128), dtypes.float16, 1),
      UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 2),
      valid_kv_len=2048, acc_small=True, k_unroll=1, k_midstore=8,
      use_acc_work=False, acc_vgpr=True, hoist_q=True), ren)
  text = next(s.content for s in elf_loader(prg.src[-1].arg)[1] if s.name == ".text")
  insts, off = [], 0
  while off < len(text):
    inst = decode_inst(text[off:], "rdna3"); insts.append(inst); off += inst.size()
    if getattr(inst, "op_name", "").lower() == "s_endpgm": break
  mc = Counter(getattr(i, "op_name", "").lower() for i in insts)
  vgpr, _, _, priv = scan_elf_regs(insts, scratch_inst_size)
  print("CLUSTER", getenv("AMD_CLUSTER_PACK_LLOAD", 0))
  print({k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()})
  print({"priv": priv, "vgpr": vgpr, "ninst": len(insts), "v_mov": mc.get("v_mov_b32_e32", 0),
         "bases": len(getattr(ren, "_pack_chain_bases", {}) or {})})

if __name__ == "__main__":
  main()
