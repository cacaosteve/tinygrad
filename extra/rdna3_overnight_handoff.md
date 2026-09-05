# Overnight RDNA3 — tip `4d1547216`

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.

## Headline

**Flash DIRECT ACC_SMALL** ~**672µs** (err~1e-4). HIP ~278; SDPA ~360.
Decode e2e ~49 vs HIP ~46; partial ~34 vs HIP ~29. Serial **13/13**.

## This session

1. **Confirmed tip** on 7900 XTX (`gfx1100`, `AMDRenderer`).
2. **Tried REDUCE-carried acc on promotable slot 19** — err~135, ~963µs, SPILL 110.
   Same failure mode as clearing `AMD_REG_PROMOTE_SKIP_SLOTS`. Reverted.
3. **Kept:** float4 tile work-copy + fuse `acc*(1/l)` into global store.
   SLOAD/SSTORE **96→64**, machine scratch **~97/74→65/62**, time ~flat (~672).

## Scorecard (7900)

| Workload | AMD | HIP |
|----------|-----|-----|
| Flash DIRECT | ~672 | ~278 |
| SDPA | ~360 | |
| Decode e2e | ~49 | ~46 |
| Decode partial | ~34 | ~29 |
| IQ4_XS t32 | ~66 | (see benches) |

## Do not retry

- Park from `AMD_FLASH_DIRECT` alone / ≤64 multi-pack auto-detect
- Promote slot 2 / move acc to slot 19 as REDUCE-carried → wrong numerics + spill thrash
- Fuse `acc=alpha*acc+beta*pv` as default → ~820µs (lost V-load overlap)
- `AMD_FLASH_K_UNROLL` / PV_ACC_DIRECT / ACC_SEP=0

## Next leftovers

1. Cut 21 allocator spills / remaining 64 slot-2 tile copies (toward HIP ~278)
2. Decode partial 34→29 (`s_delay_alu` / batching — see `rdna3_decode_partial_todo.md`)
3. Safer K unroll without MMU fault
4. IQ4 vgpr if HIP still ahead on matched shape
