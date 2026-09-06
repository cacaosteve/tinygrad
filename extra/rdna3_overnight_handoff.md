# Overnight RDNA3

Fork remote only: `tinygrad-cacaosteve` / `codex/rdna3-perf-coverage`.
Tip: **`608b8832c`**.

## Headline — correctness first

**Flash DIRECT remains experimental and correctness-blocked.** Prefill defaults to
SDPA unless `AMD_FLASH_DIRECT=1`. Packing / FMA_MIX / eviction stay **off**.

## Prior verdict (100-replay backend diag)

Same-graph **HIP is bit-stable**; DIRECT first out ~ulp of HIP, later launches
diverge. In-order emit + instr waits did **not** stabilize. Leading hypothesis:
dependence on **reused / uninitialized state** (not proven).

## State diag (this tip) — stop scheduling experiments

Harness: `extra/rdna3_flash_state_diag.py`

1. **fixed vs recreate** — same runtime/q/kv/out/launch args vs new buffers each time
2. **scratch history** — untouched / zero / nonzero (`0xA5`) fill of device scratch
   immediately before Flash (**not** a shipping fix if it changes behavior)
3. **phase dumps** — optional `phase_dumps` on `_amd_flash_attention` for QK /
   soft_m / soft_l / PV / pre-norm acc; compare HIP vs DIRECT at failing tile
4. Saves **first failing replay + predecessor** (also patched into backend_diag)

```bash
PYTHONPATH=.:extra python extra/rdna3_flash_state_diag.py \
  --modes recreate,fixed,scratch --phase --replays 100 --S 128
```

## Next

- If scratch init changes correctness → reads-before-writes / bounds / layout meta
- Trace first incorrect phase value through definition → spill → reload
- Decode only after DIRECT freeze replay is bit-exact vs HIP
