# ADR-0001 — Static splat + tracked actors instead of 4D Gaussians

- **Status:** accepted
- **Date:** 2026-11-07

## Context
The product must show both the scene (space) and what people did (time). Two families of approaches exist:
1. **Dynamic / 4D Gaussian Splatting** (e.g., 4D-GS by Wu et al., Dynamic 3D Gaussians by Luiten et al.): Gaussians that move over time.
2. **Static scene + tracked actors:** a static splat of everything that doesn't move, with people represented as measured 3D tracks (markers or SMPL bodies) driven by a timeline.

Our inputs are mostly monocular moving cameras (body-cam) or sparse fixed cameras. Videos last minutes. People move metres. The dev GPU has 4 GB.

## Decision
Adopt **static scene + tracked actors** as the primary architecture. 4D methods are a phase-4 research extension restricted to short event clips.

## Alternatives considered
- **4D-GS (HexPlane deformation field):** benchmarked on ~10 s clips with 15–20 synced cameras or near-teleporting monocular cameras. The authors report failure on large motions and on novel views from real monocular video. A 24 GB-class GPU is expected. Rejected as primary. See the [research note](../research/4dgs-feasibility.md).
- **Dynamic 3D Gaussians (persistent per-timestep Gaussians):** needs ~27 synced, calibrated cameras plus a depth-camera initialisation, and has no monocular mode. Rejected for our footage.
- **Per-frame independent splats:** no temporal consistency, huge storage, and monocular frames can't each make a 3D scene. Rejected.

## Consequences
- ✅ Works with monocular footage, minutes long, on the laptop (capped).
- ✅ Forensically honest: every actor position comes from an observed detection with a source frame and confidence.
- ✅ Small, fast viewer payloads.
- ❌ No photoreal "free-viewpoint replay" of how bodies moved. Actors are abstractions (capsules / SMPL).
- ❌ Objects that move and stay moved (a door opened, a chair knocked over) appear in only one state, or as a blur. Mitigation: mask them and show them as annotations.
- Follow-up: phase-4 event-clip 4D spike.
