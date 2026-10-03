#!/usr/bin/env python3
"""Attribute post-ELF MOV / dual_mov runs for flash DIRECT (WMMA A evacuate)."""
from __future__ import annotations
import os, re
from collections import Counter

os.environ.update({
  "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "PARALLEL": "0",
  "CACHELEVEL": "0", "CCACHE": "0", "AMD_SOFT_SYSFS": "1",
})

from tinygrad import dtypes
from tinygrad.helpers import Context, getenv, Target, DEV
from tinygrad.uop.ops import UOp
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.renderer.isa.amd import AMDRenderer
from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
import tinygrad.llm.kernels.amd as amd
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd.elf import scan_elf_regs, scratch_inst_size
from tinygrad.renderer.isa import rdna3_defs as D

def main() -> None:
  DEV.value = "AMD:AMD"
  getenv.cache_clear(); to_program_cache.clear(); amd._amd_flash_attention.cache_clear()
  print("PACK_F16", (D.PACK_F16_VGPR[0].index, D.PACK_F16_VGPR[-1].index, len(D.PACK_F16_VGPR)) if D.PACK_F16_VGPR else None)
  print("LLOAD", D.LLOAD_VGPR[0].index, D.LLOAD_VGPR[-1].index, len(D.LLOAD_VGPR))
  print("ACC", D.WMMA_ACC_VGPR[0].index, D.WMMA_ACC_VGPR[-1].index, len(D.WMMA_ACC_VGPR))

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
  print({"priv": priv, "vgpr": vgpr, "ninst": len(insts),
         "v_mov": mc.get("v_mov_b32_e32", 0),
         "dual_mov": sum(v for k, v in mc.items() if "dual_mov" in k)})

  # decode_inst Inst often stringifies as "v_mov_b32 v104, v86"
  vgpr_re = re.compile(r"\bv(\d+)\b")
  sample = next(i for i in insts if getattr(i, "op_name", "").lower() == "v_mov_b32_e32")
  print("sample", type(sample).__name__, str(sample)[:200], repr(sample)[:200])

  # Classify each mov-ish: LIT init vs VGPR←VGPR copy
  copy_re = re.compile(r"v_mov_b32_e32\(v\[(\d+)\], v\[(\d+)\]\)")
  dual_copy_re = re.compile(
    r"VOPD\(VOPDOp\.V_DUAL_MOV_B32, VOPDOp\.V_DUAL_MOV_B32, v\[(\d+)\], v\[(\d+)\], v\[(\d+)\], v\[(\d+)\]\)")
  n_lit = n_copy = n_dual_lit = n_dual_copy = 0
  copies: list[tuple[int, int]] = []
  for inst in insts:
    n = getattr(inst, "op_name", "").lower()
    s = str(inst)
    if n == "v_mov_b32_e32":
      m = copy_re.search(s)
      if m:
        d, src = int(m.group(1)), int(m.group(2))
        copies.append((src, d)); n_copy += 1
      else:
        n_lit += 1
    elif n == "v_dual_mov_b32":
      m = dual_copy_re.search(s)
      if m:
        d0, d1, s0, s1 = map(int, m.groups())
        copies.append((s0, d0)); copies.append((s1, d1)); n_dual_copy += 1
      else:
        n_dual_lit += 1
  print({"n_lit_mov": n_lit, "n_copy_mov": n_copy, "n_dual_lit": n_dual_lit, "n_dual_copy": n_dual_copy,
         "n_copy_pairs": len(copies)})
  # Streaks of consecutive copy mov/dual in program order
  runs: list[list[tuple[int, int]]] = []
  cur: list[tuple[int, int]] = []
  for inst in insts:
    n = getattr(inst, "op_name", "").lower()
    s = str(inst)
    got: list[tuple[int, int]] = []
    if n == "v_mov_b32_e32":
      m = copy_re.search(s)
      if m: got = [(int(m.group(2)), int(m.group(1)))]
    elif n == "v_dual_mov_b32":
      m = dual_copy_re.search(s)
      if m:
        d0, d1, s0, s1 = map(int, m.groups())
        got = [(s0, d0), (s1, d1)]
    if got:
      cur.extend(got)
    else:
      if len(cur) >= 4: runs.append(cur)
      cur = []
  if len(cur) >= 4: runs.append(cur)
  runs.sort(key=len, reverse=True)
  print("copy_runs", len(runs), "top_lens", [len(r) for r in runs[:10]])
  for r in runs[:5]:
    print("--- copy run", len(r), r[:20])
  # Dest histogram of all copies
  from collections import Counter as C
  src_c = C(s for s, _ in copies); dst_c = C(d for _, d in copies)
  print("top_src", src_c.most_common(12))
  print("top_dst", dst_c.most_common(12))

if __name__ == "__main__":
  main()
