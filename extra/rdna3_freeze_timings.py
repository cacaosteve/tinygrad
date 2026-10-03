#!/usr/bin/env python3
"""Fresh-process paired sync-each timings → extra/rdna3_baseline_freeze/timings.json"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "extra/rdna3_baseline_freeze")
PY = sys.executable
rows = []
for direct, case in [(1, "prefill"), (0, "prefill"), (1, "decode"), (0, "decode")]:
  env = os.environ.copy()
  env["DEV"] = "AMD:AMD" if direct else "AMD"
  if direct: env["AMD_FLASH_DIRECT"] = "1"
  else: env.pop("AMD_FLASH_DIRECT", None)
  env.update({"PYTHONPATH": ".", "PARALLEL": "0", "CACHELEVEL": "0", "CCACHE": "0"})
  p = subprocess.run(
    [PY, "extra/rdna3_fence_hw_validate.py", "--mode", "sync-each", "--case", case, "--direct", str(direct)],
    capture_output=True, text=True, env=env, timeout=180,
  )
  lines = [l for l in p.stdout.splitlines() if '"sync_each"' in l]
  if not lines:
    print("FAIL", direct, case, p.stderr[-800:], file=sys.stderr)
    sys.exit(1)
  j = json.loads(lines[-1])
  row = {
    "direct": bool(direct), "case": case, "renderer": j["renderer"],
    "median_us": j["median_us"], "best_us": j["best_us"], "mean_us": j["mean_us"], "n": j["n"],
  }
  rows.append(row)
  print(row, flush=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "timings.json").write_text(json.dumps({"tip_note": "fresh process sync-each", "rows": rows}, indent=2))
print("WROTE", OUT / "timings.json")
