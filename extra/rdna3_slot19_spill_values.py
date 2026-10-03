#!/usr/bin/env python3
"""Re-attribute slot-19 SPILL values (src[1]) vs displacements (src[0])."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from tinygrad.renderer.isa.rdna3 import AMDOps, _iop, greg
from tinygrad.renderer.isa import Register
from tinygrad.uop.ops import Ops
from extra.rdna3_pressure_attr import compile_flash, describe, live_maps

ren, prg, ops = compile_flash(use_acc_work=False, acc_slot=19)
first_def, last_use, def_uop = live_maps(ops)
wmma = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
fw, lw = wmma[0], wmma[-1]
spill = [(i, u) for i, u in enumerate(ops) if _iop(u) is AMDOps.SPILL]
fill = [(i, u) for i, u in enumerate(ops) if _iop(u) is AMDOps.FILL]

rows = []
for i, u in spill:
  disp = u.src[0] if len(u.src) > 0 else None
  val = u.src[1] if len(u.src) > 1 else None
  try: vr = greg(val) if val is not None else None
  except Exception: vr = None
  rid = id(vr) if isinstance(vr, Register) else None
  fd, lu = first_def.get(rid), last_use.get(rid)
  du = def_uop.get(rid)
  rows.append({
    "spill_i": i,
    "disp": describe(disp),
    "val": describe(val),
    "def_i": fd,
    "last_use": lu,
    "lifetime": None if fd is None or lu is None else lu - fd,
    "def_op": (_iop(du).name if du is not None and du.op is Ops.INS else (du.op.name if du is not None else None)),
    "def_desc": describe(du) if du is not None else None,
    "cross_wmma": bool(fd is not None and lu is not None and fd < fw and lu > lw),
    "wmma_before": sum(1 for w in wmma if w < i),
    "rematerialize": bool(du is not None and ren.rematerialize(du)),
    "val_dtype": str(val.dtype) if val is not None else None,
  })

report = {
  "n_spill": len(spill),
  "n_fill": len(fill),
  "wmma_window": [fw, lw],
  "spill_phase": {
    "before": sum(1 for r in rows if r["spill_i"] < fw),
    "during": sum(1 for r in rows if fw <= r["spill_i"] <= lw),
    "after": sum(1 for r in rows if r["spill_i"] > lw),
  },
  "val_top": Counter(r["val"] for r in rows).most_common(25),
  "def_op_top": Counter(r["def_op"] for r in rows).most_common(20),
  "val_dtype_top": Counter(r["val_dtype"] for r in rows).most_common(10),
  "cross_wmma": sum(1 for r in rows if r["cross_wmma"]),
  "remat_true": sum(1 for r in rows if r["rematerialize"]),
  "lifetime_hist": Counter(r["lifetime"] for r in rows).most_common(15),
  "sample": rows[:8] + rows[-6:],
}
print(json.dumps({k: v for k, v in report.items() if k != "sample"}, indent=2))
print("SAMPLE")
for r in report["sample"]:
  print(r)
Path("extra/rdna3_baseline_freeze/slot19_spill_values.json").write_text(json.dumps(report, indent=2))
