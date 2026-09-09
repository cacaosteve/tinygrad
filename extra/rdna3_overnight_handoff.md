# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **pending SCORE_BATCH restore** (was `6e1862726`).

## Headline (remeasured fair)

| | DIRECT (`DEV=AMD:AMD`) | HIP (`DEV=AMD`) |
|--|--:|--:|
| prefill median | **~757–790 µs** | **~324 µs** |
| decode median | **~133–138 µs** (SCORE_BATCH restore in flight) | **~108 µs** |
| WMMA / SPILL / priv / VGPR | **12 / 0 / 128 / 206** | (HIP) |

## Landed tonight

1. Prefill sticky **`ALLOW_UPCAST16=0`** + **`PACK_SLOAD_B128=1`** → SPILL 0  
2. Default **`AMD_FLASH_K_UNROLL=2`** → WMMA 12  
3. **Decode-scoped envs**: UPCAST16=1 + PACK off  
4. **Restore `AMD_FLASH_SCORE_BATCH`** (default 8) — was dropped in #18010 merge

## Dead ends / washes (this session)

- CLUSTER_SLOAD_SCAN=128, COMBINE_UNROLL=64, K_TILE_OUTER, SWIZZLE_DELAY/VALU_GAP e2e  
- Scoped K_UNROLL (qk/pv only) slower than all; hip_qk **MMU**  
- Slot-2 promote: ~same time but SPILL16 / priv256  
- ACC_SHARED: historically MMU — leave opt-in off

## Decode asm gap (still)

DIRECT partial: 207 waits, **0** delay_alu, priv 64, vgpr 103  
HIP: 146 waits, **120** delay_alu, priv 0, vgpr 85  
`AMD_SWIZZLE_DELAY=1` adds 128 delay_alu but no e2e win under park.

## Next

1. Validate SCORE_BATCH: decode A/B + serial 13/13; land if win.  
2. Prefill WMMA 12→24 without MMU; cut priv 128.  
3. Fork-only; keep soak on tip.
