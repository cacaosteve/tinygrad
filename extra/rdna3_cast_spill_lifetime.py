#!/usr/bin/env python3
"""Identity-based CAST:int spill lifetime for slot-19 experiment."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

from tinygrad.renderer.isa.rdna3 import AMDOps, _iop
from tinygrad.uop.ops import Ops

from extra.rdna3_pressure_attr import compile_flash, describe

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "extra/rdna3_baseline_freeze")
ren, prg, ops = compile_flash(use_acc_work=False, acc_slot=19)
wmma = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.WMMA]
spill = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.SPILL]
fill = [i for i, u in enumerate(ops) if _iop(u) is AMDOps.FILL]
fw, lw = wmma[0], wmma[-1]

cast_first: dict[int, int] = {}
cast_uses: dict[int, list[int]] = {}
for i, u in enumerate(ops):
  stack = list(u.src)
  while stack:
    s = stack.pop()
    if s.op is Ops.CAST:
      cast_first.setdefault(id(s), i)
      cast_uses.setdefault(id(s), []).append(i)
    if s.op is Ops.INS:
      stack.extend(s.src)

rows = []
for i in spill:
  val = ops[i].src[0] if ops[i].src else None
  cid = id(val) if val is not None else None
  uses = cast_uses.get(cid, [])
  last = max(uses) if uses else None
  first = cast_first.get(cid)
  off = None
  if val is not None and val.op is Ops.CAST and val.src and val.src[0].op is Ops.CONST:
    off = val.src[0].arg
  # first FILL after spill
  next_fill = next((j for j in fill if j > i), None)
  rows.append({
    "spill_i": i, "off": off, "desc": describe(val),
    "first_seen": first, "last_use": last,
    "lifetime": None if first is None or last is None else last - first,
    "cross_wmma": bool(first is not None and last is not None and first < fw and last > lw),
    "next_fill": next_fill,
    "gap_to_fill": None if next_fill is None else next_fill - i,
    "n_uses": len(uses),
  })

scratch_uses = []
for i, u in enumerate(ops):
  if _iop(u) is AMDOps.SCRATCH_ADDR:
    scratch_uses.append({
      "i": i,
      "src": describe(u.src[0] if u.src else None),
      "wmma_before": sum(1 for w in wmma if w < i),
    })

report = {
  "n_spill": len(spill),
  "wmma_window": [fw, lw],
  "cross_wmma": sum(1 for r in rows if r["cross_wmma"]),
  "lifetime_hist": Counter(r["lifetime"] for r in rows).most_common(20),
  "gap_to_fill_hist": Counter(r["gap_to_fill"] for r in rows).most_common(20),
  "off_hist": Counter(r["off"] for r in rows).most_common(30),
  "spill_phase": {
    "before": sum(1 for r in rows if r["spill_i"] < fw),
    "during": sum(1 for r in rows if fw <= r["spill_i"] <= lw),
    "after": sum(1 for r in rows if r["spill_i"] > lw),
  },
  "unique_offs": sorted({r["off"] for r in rows if r["off"] is not None}),
  "n_scratch_addr": len(scratch_uses),
  "scratch_addr_vs_wmma": Counter(s["wmma_before"] for s in scratch_uses).most_common(12),
  "rows_sample": rows[:10] + rows[-8:],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cast_spill_lifetime.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "rows_sample"}, indent=2))
