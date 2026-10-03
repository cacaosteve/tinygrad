#!/usr/bin/env python3
"""Confirm MUL_DELAY and related delay knobs."""
from __future__ import annotations
import os, pathlib, subprocess

ROOT = "/home/admin441766/tinygrad"
PY = f"{ROOT}/venv/bin/python"
CHILD = pathlib.Path(f"{ROOT}/extra/rdna3_fair_ab_knobs.py").read_text().split('CHILD = r"""')[1].split('"""')[0]

def run(label: str, extra: dict) -> None:
  env = os.environ.copy()
  env.update({"PYTHONPATH": ROOT, "AMD_SOFT_SYSFS": "1", "PARALLEL": "0", "CACHELEVEL": "0",
              "LABEL": label, "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1"})
  for k in list(env):
    if k.startswith("AMD_") and k not in ("AMD_SOFT_SYSFS", "AMD_FLASH_DIRECT"):
      env.pop(k, None)
  env.update(extra)
  p = subprocess.run([PY, "-c", CHILD], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
  print(p.stdout.strip() if p.returncode == 0 else f"FAIL {label}\n{(p.stderr or '')[-400:]}", flush=True)

if __name__ == "__main__":
  for i in range(3):
    run(f"ship{i}", {})
  for i in range(3):
    run(f"mul2_{i}", {"AMD_MUL_DELAY": "2"})
  for i in range(2):
    run(f"mul1_{i}", {"AMD_MUL_DELAY": "1"})
  for i in range(2):
    run(f"mul3_{i}", {"AMD_MUL_DELAY": "3"})
  # combo mul2 + existing
  for i in range(2):
    run(f"mul2_pre{i}", {"AMD_MUL_DELAY": "2", "AMD_WMMA_DELAY_PRE": "1"})
