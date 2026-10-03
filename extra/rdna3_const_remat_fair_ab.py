#!/usr/bin/env python3
"""Fair same-session A/B for AMD_REMAT_CONST_MOV + serial gate."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def sync_each(direct: bool, *, remat: int | None = None) -> dict:
  env = os.environ.copy()
  env["DEV"] = "AMD:AMD" if direct else "AMD"
  if direct: env["AMD_FLASH_DIRECT"] = "1"
  else: env.pop("AMD_FLASH_DIRECT", None)
  if remat is None: env.pop("AMD_REMAT_CONST_MOV", None)
  else: env["AMD_REMAT_CONST_MOV"] = str(remat)
  env.update({"PYTHONPATH": ".", "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0", "AMD_SOFT_SYSFS": "1"})
  p = subprocess.run(
    [PY, "extra/rdna3_fence_hw_validate.py", "--mode", "sync-each", "--case", "prefill", "--direct", str(int(direct))],
    capture_output=True, text=True, env=env, timeout=180, cwd=ROOT,
  )
  lines = [l for l in p.stdout.splitlines() if '"sync_each"' in l]
  if not lines:
    raise RuntimeError((p.stderr or p.stdout)[-1000:])
  return json.loads(lines[-1])


def main():
  rows = []
  for remat in (0, 1, 0, 1):  # interleaved to reduce clock drift bias
    d = sync_each(True, remat=remat)
    h = sync_each(False, remat=None)
    row = {"remat": remat, "direct_us": d["median_us"], "direct_best": d["best_us"],
           "hip_us": h["median_us"], "hip_best": h["best_us"],
           "ratio": d["median_us"] / h["median_us"]}
    rows.append(row)
    print(row, flush=True)

  print("=== serial ===", flush=True)
  env = os.environ.copy()
  env.update({"PYTHONPATH": ".", "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0",
              "AMD_SOFT_SYSFS": "1", "DEV": "AMD:AMD", "AMD_FLASH_DIRECT": "1", "AMD_REMAT_CONST_MOV": "1"})
  p = subprocess.run([PY, "extra/rdna3_serial_correctness.py"], capture_output=True, text=True,
                     env=env, timeout=600, cwd=ROOT)
  print(p.stdout[-2000:] if p.stdout else "", flush=True)
  print(p.stderr[-1000:] if p.stderr else "", flush=True)
  print("serial_exit", p.returncode, flush=True)

  out = {"rows": rows, "serial_exit": p.returncode, "serial_tail": (p.stdout or "")[-1500:]}
  Path("extra/rdna3_baseline_freeze/const_remat_fair_ab.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
