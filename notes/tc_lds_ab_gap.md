# Closing codegen WMMA gap vs hand kernel (local WIP)

## Status (2026-07-26 night)

| Path | @4096 HW | check |
|------|------:|-------|
| register default | **~26.6k** | ok |
| `TC_LDS_AB` bounce 2×2 | **~2.4k** | ok |
| `TC_LDS_AB` + UPCAST 4×2 | — | **still wrong** (~25% cells) |

Landed: `ef23049cf` INDEX-through-RESHAPE/PERMUTE mop (necessary, not sufficient).

## Tile product 8 — root cause narrowed

Repro: `TC_LDS_AB=1 TC_UPCAST=4 TC_UPCAST_TILES=8 TC_LOCAL=2`, n=256, identity@B.
Error band = A upcast index 3 only (rows 96:128 / 224:256).

### Not the bug
- Expand `(ta,tb)` placement vs `build_range_map` (matches).
- Post-local LDS read idxs (`+0,+16,+32,+48`).
- LDS **contents** for tile 3 (forcing all WMMAs to read LDS3 → slot 3 perfect).
- Wide vs scalar LDS loads (96 scalar LLOADs still fail).
- Shared C / tid-fill / emission order.

### Is the bug
**Four distinct A packs sourced from LDS live together.** Remap tests (expand forced):

| A-tile map | slot3 | note |
|------------|------:|------|
| `[0,1,2,3]` | **93% bad** | all 4 distinct |
| `[3,3,3,3]` | **0%** | LDS3 alone OK |
| `[0,1,0,3]` | **0%** | 3 distinct, includes 3 |
| `[0,1,3,3]` / `[0,0,2,3]` / `[1,1,2,3]` | **0%** | any 3-subset OK |
| same opts, STAGE off | **0%** | register path OK with 4 A packs |

So: LDS-sourced 4-way A live ranges break tile 3; global-sourced 4-way is fine. Likely regalloc / EXTRACT-temp aliasing around `PACK_F16` when 4 LDS A packs + 2 B packs are live (post-regalloc wiring *looks* right; interference is subtler).

### Next fix (pick one)
1. **Serialize** expand_wmma STACK path: materialize WMMA results for A-tiles 0..1 into REG, then tiles 2..3 (≤2–3 distinct A live).
2. **Fix** PACK_F16 / EXTRACT temp allocation so temps cannot land in future multi-reg pack bases (see PACK v97 `EXTRACT dest=v198` before PACK v198).
3. Keep heuristic at **2×2** until fixed.

Heuristic stays capped at **2×2** under `TC_LDS_AB`.
