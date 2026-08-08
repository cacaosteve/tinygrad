# AMD ASM session checkpoint (Aug 8 2026 ~1:15 AM PT)

Tip: `73b73f361` on amd-asm-backend → PR https://github.com/tinygrad/tinygrad/pull/17459

## Landed this stretch
- `7fb5dd9d3` assert default LOCAL=4
- `2fabba003` AMD_D16_HI hang note
- `560d3db3b` LDS skip stage for non-GLOBAL INDEX (eye@B / A@eye fallback)
- `ee8841ea4` drop dead TC_LDS_GROUP
- `73b73f361` A@eye unit coverage

## Best default-path (method: warm≥8, med last half, GlobalCounters, random@random)
- @2048 ~24.5k GFLOPS, RMSE ~9e-3
- @4096 ~60–70k GFLOPS (clock noise), RMSE ~1.3e-2
- identity@B RMSE 0

## Failed / reverted
- Dual register LOCAL 4×2: slower
- PACK_F16 sink past addr ALU: wrong (dest-as-addr)
- Soft WMMA waitcnt: noise
- AMD_D16_HI fixes (scratch/TMP): still hang ones@256+
- LOCAL=8 on p8: slower than LOCAL=4
- TC_LDS_GROUP: LDS size always exceeds

## Next
1. Close @4096 gap vs AMDLLVM/HIP (~80k) — LDS double-buffer / unique load temps for multi-clause overlap
2. Fix AMD_D16_HI properly (no in-place index scale; unique hi-addr VGPRs that stay live until wait)
3. LDS product-8+ still behind register; product-16 LDS hang
