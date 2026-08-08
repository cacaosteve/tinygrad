# AMD ASM session checkpoint (Fri night → Sat ~01:40 PT)

Tip: `6f83d53c5` on `amd-asm-backend` / PR https://github.com/tinygrad/tinygrad/pull/17459

## Commits this stretch
- `d7b6d3cd3` — always `s_clause` scalar half loads (drop `AMD_LOAD_CLAUSE`); HW +40% @2048 vs clause off
- `6f83d53c5` — unit-test product-16 on irregular N (320/512/768)

## Best default-path numbers (method: warm≥6–8, med last half, `GlobalCounters.time_sum_s`)
| N | pattern | med | RMSE |
|---:|---|---:|---|
| 2048 | random | ~24.5k | ~9e-3 |
| 2048 | identity | ~24k | 0 |
| 4096 | random | ~60–70k (noisy) | ~1.3e-2 |
| 320/512/768 | random | — | clean on HW |
| 256 | f32 | — | ~5e-6 |

ASM vs LLVM @2048 random: ~24.4k vs ~30.5k (same launch shape).

## Failed / reverted tonight
- D16_HI TMP hi-addr: ones still hangs; random slower (~19k)
- Pack sink past B loads: blocked — B addr ADDs reuse A load VGPRs
- Epilogue store clause without distinct half regs: RMSE catastrophic (regalloc aliases)
- Store batch schedule+clause: correct but ~no GFLOPS win vs tip → reverted
- TC_LDS_TID: slower than bounce (~10k vs ~15k @2048 LDS)

## Next unfinished
Close LLVM gap @2048: need unique B-addr VGPRs so B128 can issue while A U16 in flight (or fix D16_HI properly). LDS double-buffer only if faster than register default.

## Later (~01:42)
- Hand kernel @4096: 72.6 TFLOPS
- LLVM @4096 warm≥10: ~79.2k vs ASM ~60.7k
- Pre-regalloc B-load hoist: no ISA change / no win — reverted
