#!/usr/bin/env python3
"""Attribute AMD_PACK_F16_HALF2_GLOBAL emit: load→lshr/mov/pack neighborhoods."""
from __future__ import annotations
import json, os, sys
from collections import Counter

def main(half2: int) -> dict:
  os.environ.update({
    "PYTHONPATH": ".", "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
    "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1",
    "AMD_PACK_F16_HALF2_GLOBAL": str(half2),
  })
  from tinygrad import dtypes
  from tinygrad.helpers import Context, getenv, Target
  from tinygrad.uop.ops import UOp
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.renderer.isa.amd import AMDRenderer
  from tinygrad.llm.kernels.amd import _amd_flash_attention, _flash_direct_compile_env
  from tinygrad.renderer.amd import decode_inst
  from tinygrad.runtime.support.elf import elf_loader
  getenv.cache_clear(); to_program_cache.clear()
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
  names = [getattr(i, "op_name", "").lower() for i in insts]
  windows = []
  shown = 0
  for i, n in enumerate(names):
    if "global_load" not in n or shown >= 2: continue
    chunk = []
    for j in range(i, min(i + 48, len(insts))):
      chunk.append(f"{j}:{names[j]}::{insts[j]}")
      if names[j] == "v_wmma_f32_16x16x16_f16": break
    windows.append(chunk)
    shown += 1
  return {
    "half2": half2,
    "counts": {k: names.count(k) for k in (
      "v_mov_b32_e32", "v_lshrrev_b32_e64", "v_pack_b32_f16", "global_load_b128",
      "v_dual_mov_b32", "v_dual_dot2acc_f32_f16")},
    "windows": windows,
  }

if __name__ == "__main__":
  h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  print(json.dumps(main(h), indent=2))
