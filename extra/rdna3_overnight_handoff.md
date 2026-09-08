# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`a688c7b9f`** (const-MOV remat).

## Headline

Flash DIRECT **opt-in**. Decode fixed (`ml_lds (WAVES,2,G)`). Prefill gap ~**2.4×**
vs HIP hand kernel on fresh processes (2026-09-08 night):

| | DIRECT | HIP |
|--|--:|--:|
| prefill median | **~670 µs** | **~280 µs** |
| machine WMMA | **6** | **24** |
| private_segment | **184 B** (was 192) | **0** |
| allocator SPILL | **14** (was 16→21) | **0** |
| decode median | ~57 µs | ~48 µs |

Soak: **58+** clean rounds on prior tip; continue on tip.

## Landed this loop

- Decode ml_lds G==SEC fix (16-wave + 32k)
- Addr remat depth-1 on flash realize (21→16 SPILL)
- Factor `K_UNROLL` scope via `AMD_FLASH_K_HIP_SCOPE`
- Remat **float/int const MOV** (softmax constants): SPILL **16→14**, priv **192→184**

## Remat dead ends (do not revive)

- `MUL` in `_pure_addr` + depth-1 → **wrong numerics** (err~0.79)
- `AMD_REMAT_ADDR_DEEP=2` → **0 spills locally**, **hangs/MMU** on 7900

## Next leftovers

1. Continuous soak on tip.
2. Remaining 14 SPILLs are nested ADD/MUL LDS bases — need safer remat binding
   or fewer live addrs (not depth-2).
3. Slot-2 scratch still 96 SLOAD/SSTORE; promote still nans.
4. 24 WMMA / full-K still hangs; PV-only expand correct but slower.
5. Do not flip DIRECT default yet.
