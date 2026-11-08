# ADR-0003 — Pose estimation: hloc + GLOMAP primary, feed-forward fallback

- **Status:** accepted
- **Date:** 2026-11-07

## Context
Camera poses are the foundation of everything (splat, actor lifting, evidence triangulation, the click → frame lookup). Body-cam footage is blurry and wide-angle. Multi-CCTV is wide-baseline and sparse.

## Decision
- **Primary:** hloc (SuperPoint + LightGlue, NetVLAD / MegaLoc retrieval) → GLOMAP global mapping → COLMAP bundle adjustment with a fisheye-capable camera model. Then localise all frames against the model.
- **Fallback / CCTV default:** VGGT or MASt3R-SfM on a cloud GPU, whose dense points also initialise 3DGS.
- **CCTV recommended workflow:** reconstruct a survey walk-through, then localise each CCTV camera into it.

## Alternatives considered
- COLMAP SIFT only: weaker on blur and wide baselines.
- VGGT / MASt3R only: memory-hungry (not 4 GB), non-commercial licences, less accurate than BA on long sequences.
- SLAM (ORB-SLAM3, DROID-SLAM, MegaSaM): attractive for long video and dynamic scenes. Worth an ablation, but adds a separate stack. Reconsider if SfM registration rates are poor.

## Update 2026-11-08: v0 implementation and test result
- v0 uses **COLMAP SIFT + incremental mapping via pycolmap 4.2** (no hloc yet). pycolmap 4.2 bundles GLOMAP as `global_mapping`, so both mappers are available with no extra build.
- On the synthetic room with known poses: incremental reached 4 mm mean trajectory error / 0.5° up error; GLOMAP reached 19 mm mean with a 29 cm outlier / 4.7°. GLOMAP relies on a focal-length prior, and video frames have no EXIF. **Incremental is the default**; global remains a config option (`sfm.mapper`).
- `SIMPLE_RADIAL` (one focal length) is the default for phone/handheld footage. `OPENCV` let fx and fy drift apart in testing.
- Outlier cameras (registered far from the trajectory) are dropped automatically.
- hloc (SuperPoint + LightGlue) remains the planned upgrade for blurry body-cam footage. Revisit after the first real-footage tests.

## Consequences
- ✅ Robust primary path that runs locally.
- ❌ Two pose paths to maintain and test. Coordinate conventions must be normalised (S05).
