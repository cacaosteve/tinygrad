#!/usr/bin/env python3
"""Attempt SQTT capture for DIRECT flash prefill (VIZ=-2 / SQTT=1)."""
from __future__ import annotations
import json, os, pickle, sys, traceback
from collections import Counter
from pathlib import Path

import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import DEV, getenv, temp
from tinygrad.llm.kernels.amd import flash_attention

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "extra/rdna3_baseline_freeze")
OUT.mkdir(parents=True, exist_ok=True)

os.environ["DEV"] = "AMD:AMD"
os.environ["AMD_FLASH_DIRECT"] = "1"
DEV.value = "AMD:AMD"
getenv.cache_clear()

info = {
  "viz": os.environ.get("VIZ"),
  "sqtt": os.environ.get("SQTT"),
  "profile": os.environ.get("PROFILE"),
  "sqtt_buffer_size": os.environ.get("SQTT_BUFFER_SIZE"),
}

def patterned(shape, dtype=np.float32):
  c = int(np.prod(shape))
  return (((np.arange(c, dtype=np.int32) % 127) - 63) / 64.0).astype(dtype).reshape(shape)

try:
  q = Tensor(patterned((1, 32, 32, 128)), device=Device.DEFAULT).realize()
  cache = Tensor(patterned((2, 1, 8, 2048, 128), np.float16), device=Device.DEFAULT).realize()
  out = flash_attention(q, cache, 2048).realize()
  Device[Device.DEFAULT].synchronize()
  arr = out.numpy()
  info["checksum"] = float(arr.astype(np.float64).sum())
  info["finite"] = bool(np.isfinite(arr).all())
  pkl = Path(temp("profile.pkl", append_user=True))
  info["pkl"] = str(pkl)
  info["exists"] = pkl.exists()
  if pkl.exists():
    events = pickle.load(open(pkl, "rb"))
    info["kinds"] = dict(Counter(type(e).__name__ for e in events))
    (OUT / "sqtt_profile.pkl").write_bytes(pkl.read_bytes())
    slim = [e for e in events if type(e).__name__ in {"ProfilePMCEvent", "ProfileSQTTEvent", "ProfileProgramEvent"}]
    pickle.dump(slim, open(OUT / "sqtt_profile_slim.pkl", "wb"))
    info["slim"] = len(slim)
    info["sqtt_events"] = sum(1 for e in events if type(e).__name__ == "ProfileSQTTEvent")
    info["sqtt_blob_bytes"] = sum(len(getattr(e, "blob", b"")) for e in events if type(e).__name__ == "ProfileSQTTEvent")
  else:
    info["error"] = "profile.pkl not found"
except Exception as e:
  info["error"] = repr(e)
  traceback.print_exc()

(OUT / "sqtt_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
