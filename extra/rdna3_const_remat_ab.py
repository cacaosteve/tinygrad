#!/usr/bin/env python3
"""A/B: baseline vs slot-19 promote under const-MOV remat."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

from extra.rdna3_pressure_attr import compile_flash, describe, elf_metrics
from tinygrad.renderer.isa.rdna3 import AMDOps, _iop

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
AMD = ROOT / "tinygrad" / "llm" / "kernels" / "amd.py"


def sync_each(direct: bool) -> dict:
  env = os.environ.copy()
  env["DEV"] = "AMD:AMD" if direct else "AMD"
  if direct: env["AMD_FLASH_DIRECT"] = "1"
  else: env.pop("AMD_FLASH_DIRECT", None)
  env.update({"PYTHONPATH": ".", "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0"})
  p = subprocess.run(
    [PY, "extra/rdna3_fence_hw_validate.py", "--mode", "sync-each", "--case", "prefill", "--direct", str(int(direct))],
    capture_output=True, text=True, env=env, timeout=180, cwd=ROOT,
  )
  lines = [l for l in p.stdout.splitlines() if '"sync_each"' in l]
  if not lines:
    raise RuntimeError(p.stderr[-800:] or p.stdout[-800:])
  return json.loads(lines[-1])


def patch_slot19(enable: bool):
  text = AMD.read_text()
  if enable:
    out = text.replace(
      "  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
      "  acc, m_i, l_i = _reg((TM, TD), 19, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)",
      1,
    ).replace(
      'use_acc_work = bool(use_acc_small and (2 in _skip_reg) and getenv("AMD_FLASH_ACC_WORK", 1))',
      'use_acc_work = False  # slot19 promote A/B',
      1,
    )
  else:
    # restore from git
    subprocess.check_call(["git", "checkout", "--", str(AMD)], cwd=ROOT)
    return
  if out == text:
    raise RuntimeError("patch failed")
  AMD.write_text(out)


def main():
  print("=== remaining slot19 spills ===", flush=True)
  _, _, ops = compile_flash(use_acc_work=False, acc_slot=19)
  for i, u in enumerate(ops):
    if _iop(u) is AMDOps.SPILL:
      print(i, "disp", describe(u.src[0]), "val", describe(u.src[1])[:90], flush=True)

  print("=== baseline metrics ===", flush=True)
  _, prg, ops = compile_flash(use_acc_work=True, acc_slot=2)
  m = elf_metrics(prg)
  print({k: m[k] for k in ("max_vgpr", "private_segment_size", "v_mov_b32", "ninst",
                           "scratch_store_b32", "scratch_load_b128")}, flush=True)

  print("=== baseline DIRECT timing ===", flush=True)
  d = sync_each(True)
  print(d, flush=True)
  print("=== HIP timing ===", flush=True)
  h = sync_each(False)
  print(h, flush=True)

  print("=== slot19 promote timing ===", flush=True)
  try:
    patch_slot19(True)
    s19 = sync_each(True)
    print(s19, flush=True)
  finally:
    patch_slot19(False)
    print("restored amd.py", flush=True)

  out = {
    "baseline_direct": d,
    "hip": h,
    "slot19_direct": s19,
    "baseline_metrics": {k: m[k] for k in ("max_vgpr", "private_segment_size", "v_mov_b32", "ninst",
                                           "scratch_store_b32", "scratch_load_b128")},
  }
  Path("extra/rdna3_baseline_freeze/const_remat_ab.json").write_text(json.dumps(out, indent=2))
  print("WROTE const_remat_ab.json", flush=True)


if __name__ == "__main__":
  main()
