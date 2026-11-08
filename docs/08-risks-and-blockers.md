# 08 — Risks, Blockers & Mitigations

Likelihood (L) and Impact (I) are rated High / Medium / Low.

## 1. Risk register

### Technical: reconstruction

| ID | Risk | L | I | Mitigation | Owner stage |
|---|---|---|---|---|---|
| R-01 | **4 GB VRAM** is insufficient for full-quality 3DGS and several models | H | H | MCMC with a Gaussian cap, 960 px training, `packed` mode, one model resident at a time, cloud offload profile ([ADR-0004](decisions/0004-hybrid-compute.md)) | S06, orchestrator |
| R-02 | **SfM fails** on body-cam (blur, fisheye, low light, repetitive textures) | M | H | Sharpness-based frame selection, learned features (SuperPoint + LightGlue), fisheye camera model, VGGT / MASt3R fallback, per-shot sub-scenes | S02, S04 |
| R-03 | **Pure rotation segments** (officer turns, doesn't walk) → degenerate geometry | H | M | Homography test flags `low_parallax`. Those frames are localised only and down-weighted in training. | S02, S04 |
| R-04 | **People dominate the frame** (close-range interactions) → few static features | M | H | Masks remove them, but if > 70% is masked the frame is excluded from SfM. Rely on frames before/after the interaction. | S03, S04 |
| R-05 | **Ghosting / floaters** from imperfect masks | M | M | Dilated masks, robust loss (SpotLessSplats-style), opacity regularisation, post-hoc floater pruning | S06 |
| R-06 | **Burned-in overlays, watermarks, redaction boxes** treated as geometry | H | M | Temporal-variance overlay detector, redaction detector, masked | S02, S03 |
| R-07 | **Edited videos** (news) with cuts, zooms, picture-in-picture, slow-motion replays | H | M | Shot detection, per-shot reconstruction, merge where possible. Zoom changes intrinsics → per-shot camera. Detect replays via duplicate-frame hashing. | S02, S04 |
| R-08 | **Scale ambiguity** (monocular) | H | M | Three independent estimators + user override. Report `scale_uncertainty`. Never display measurements without the uncertainty. | S05 |
| R-09 | **Multi-floor / stairs** break the single ground-plane assumption | M | M | v1: per-sub-scene plane, flag tracks on stairs. v2: piecewise ground from the splat's rendered depth / a height map. | S05, S07 |
| R-10 | **Multi-CCTV sparse views** → low-quality splat | H | M | Recommend a survey walk-through video. MASt3R/VGGT initialisation. Clearly label quality in the viewer. | S04, S06 |
| R-11 | **CCTV clock drift / no sync** | H | M | Sync via OCR of burned-in timestamps, audio cross-correlation, or a manual offset in the review tool | S01, S07 |
| R-12 | **Rolling shutter** distortion during fast motion | M | L | Down-weight high-motion frames. COLMAP has no RS model, so accept the error. | S02 |
| R-13 | **Low resolution / heavy compression** (YouTube re-encodes, 480p CCTV) | M | M | Prefer the highest-quality stream. Optional deblocking. Set accuracy expectations in the docs. | S01 |

### Technical: dynamics & understanding

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R-20 | **ID switches** in tracking (people crossing, occlusions) | H | M | ReID features, 3D-aware stitching, manual merge/split in the review tool |
| R-21 | **Feet not visible** (body-cam close range, truncation) → bad lifting | H | M | Torso-height prior, rendered splat depth, mark samples `depth`-sourced with lower confidence |
| R-22 | **VLM hallucination** of evidence or events | H | H | Detector-grounded prompts, cited-frame verification, `proposed` status + mandatory human review, confidence shown in the UI |
| R-23 | **Open-vocab detector misses** small evidence (shell casings) | H | M | Manual annotation in the review tool. Allow extra case-specific vocabulary. Run on full-res crops (tiling). |
| R-24 | **Graphic content** handled by commercial VLM APIs may be refused or policy-flagged | M | M | Provider-agnostic client. Local Qwen2.5-VL fallback. Detector-only mode. |
| R-25 | Wearer detected as a person (own arms / hands) | H | L | `wearer` mask class (person regions touching the bottom edge), excluded from tracking |

### Project / research

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R-30 | **No ground truth** for real crime footage → can't evaluate quantitatively | H | H | **Staged scenarios** with measured positions (see [09-evaluation-and-ethics.md](09-evaluation-and-ethics.md)), public datasets for sub-components |
| R-31 | **Ethics / legality** of using real body-cam footage of victims | H | H | Use staged or consented data for development and publication. Real public footage only for qualitative, non-published tests, per institutional ethics guidance. |
| R-32 | **Dependency churn** (fast-moving research repos, CUDA build breakage) | H | M | Pin versions, Docker image, adapters isolate each model |
| R-33 | Scope creep into 4DGS | M | M | ADR-0001. 4D is a phase-4 research spike ([research note](research/4dgs-feasibility.md)). |
| R-34 | Licence restrictions (non-commercial models) | M | L (academic) | Licence inventory in [07-tech-stack.md](07-tech-stack.md) |
| R-35 | Cloud GPU cost overruns | L | L | Only S06 / fallback / HMR remote. Budget alerts. Spot instances. |

### Forensic / trust

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R-40 | Analysts over-trust a photorealistic reconstruction ("the render looks real, so it's true") | H | H | Snap-to-source verification, visual distinction for proposed vs confirmed items, uncertainty display, "reconstructed / not observed" shading for regions with few supporting views |
| R-41 | Broken chain of custody (no trace from output to original evidence) | M | H | SHA-256 of the source, append-only provenance log, manifest file hashes, review log |
| R-42 | Privacy leakage via hosted demo packages | M | H | `publish` refuses packages not marked `synthetic` / `consented` |

## 2. Hard blockers (things that make a case impossible)

| Blocker | Detection | Response |
|---|---|---|
| Single static camera, no other footage | S04: all frames share ≈ identical pose / homography | Stop the reconstruction. Offer 2.5D mode: depth-image pop-out + 2D tracking only (future work). |
| < ~30 sharp, registrable frames | S02 / S04 reports | Report the failure with reasons. Suggest a better source. |
| Scene too dark (mean luminance < threshold) | S02 | Attempt low-light enhancement (optional), else fail. |
| Video is mostly occluded (camera covered, lens against a body) | S04 registered % | Partial reconstruction of the shots that work. |

## 3. Top 5 risks to retire first (phase 0–1)

1. **R-01** VRAM: measure gsplat MCMC on a 250-frame scene on the laptop.
2. **R-02** SfM on real body-cam: test on 3 public body-cam / walk-through videos.
3. **R-04 / R-05** masking quality: before/after ghost comparison.
4. **R-30** ground truth: design and film the first staged scenario early.
5. **R-22** VLM hallucination: measure precision of auto-proposed evidence on the staged scenario.
