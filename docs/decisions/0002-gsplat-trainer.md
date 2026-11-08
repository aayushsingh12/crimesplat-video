# ADR-0002 — gsplat as the 3DGS trainer

- **Status:** accepted
- **Date:** 2026-11-07

## Context
We need a 3DGS training implementation that fits in 4 GB VRAM, supports masks, fisheye / appearance compensation, and is licence-friendly and maintained.

## Decision
Use **gsplat** (nerfstudio-project), with the **MCMC densification strategy** and a hard cap on the Gaussian count, `packed` rasterisation, and per-image appearance optimisation.

## Alternatives considered
- **Original Inria 3DGS:** reference quality, but higher memory use, no Gaussian cap, and a non-commercial licence.
- **nerfstudio `splatfacto`:** wraps gsplat. Convenient, but adds a heavy framework. We may use its data processing for comparison.
- **Commercial / closed tools (Polycam, Luma, Postshot):** not reproducible or scriptable for research, and send data to third parties.

## Consequences
- ✅ Memory-bounded training on the laptop. Apache-2.0.
- ✅ The same library runs in the cloud profile with higher caps.
- ❌ We own the training loop (masked loss, down-weighting), which is more code to maintain.
