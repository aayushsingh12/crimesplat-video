# CrimeSplat — Video Pipeline Documentation

Turning crime-scene video (body-cam, CCTV, YouTube / news / drone) into an explorable 3D Gaussian Splat reconstruction with clickable evidence and a time-driven replay of the people involved.

## Reading order

| # | Document | What it answers |
|---|---|---|
| 00 | [Overview & requirements](00-overview.md) | What we're building, for whom, and what's in and out of scope |
| 01 | [Feasibility](01-feasibility.md) | Can it be done, on which footage, and on a 4 GB GPU? |
| 02 | [High-level architecture](02-high-level-architecture.md) | System context, containers, stages, principles, deployment |
| 03 | [Data flow](03-data-flow.md) | DFD levels 0–2, runtime sequence diagrams |
| 04 | [Low-level design](04-low-level-design.md) | Per-stage algorithms, interfaces, parameters, failure modes |
| 05 | [Data model](05-data-model.md) | Scene Package format and JSON schemas |
| 06 | [Viewer design](06-viewer-design.md) | Web viewer and review tool |
| 07 | [Tech stack](07-tech-stack.md) | Libraries, models, licences |
| 08 | [Risks & blockers](08-risks-and-blockers.md) | Risk register and mitigations |
| 09 | [Evaluation & ethics](09-evaluation-and-ethics.md) | Research questions, metrics, datasets, ethics |
| 10 | [Roadmap](10-roadmap.md) | Phases and milestones |

### Decision records
| ADR | Decision |
|---|---|
| [0001](decisions/0001-static-scene-plus-tracked-actors.md) | Static splat + tracked actors instead of 4D Gaussians |
| [0002](decisions/0002-gsplat-trainer.md) | gsplat as the 3DGS trainer |
| [0003](decisions/0003-pose-estimation-strategy.md) | hloc + GLOMAP primary, VGGT / MASt3R fallback |
| [0004](decisions/0004-hybrid-compute.md) | Hybrid laptop + cloud GPU compute |
| [0005](decisions/0005-human-in-the-loop.md) | Human review of all AI-proposed content |

### Research notes
- [4D Gaussian Splatting feasibility](research/4dgs-feasibility.md): 4D-GS (Wu et al.) and Dynamic 3D Gaussians (Luiten et al.) assessed against our footage.

### How it was built
- [Build log & technical explainer](11-build-log-and-explainer.md): every command, every technology, the theory behind each stage (graduate level), and the debugging story.

### Operations
- [GPU window runbook](gpu-day-runbook.md): filming guidelines, laptop pre-checks, and the 2-day plan on the 40-series machine.

## Implementation status

v0.1 (2026-11-08) implements the static-scene path. Code is in `pipeline/` (Python) and `web/` (viewer).

| Stage (docs/04) | Status | Notes |
|---|---|---|
| S01 Ingest | ✅ | File or URL (yt-dlp), SHA-256, ffprobe. Single source only. |
| S02 Frame selection | ✅ | Sharpest frame per time bin. Shot detection warns only. |
| S03 Masking | ✅ (YOLO11-seg) | SAM 2 propagation, overlay and redaction masks: not yet |
| S04 Poses | ✅ (COLMAP SIFT + incremental) | hloc, all-frame localisation, multi-shot merge: not yet. See ADR-0003 update. |
| S05 Normalisation | ✅ | RANSAC floor, camera-height scale. Monocular depth / manual scale: not yet. |
| S06 Splat training | ✅ (gsplat MCMC, masked loss) | Appearance embedding, robust loss: not yet |
| S07–S09 Actors, evidence, events | ⏳ | Phase 2 |
| S10 Packaging | ✅ (v0 subset) | PLY (no SPZ yet), trajectory, thumbs, sparse points with visibility |
| Viewer | ✅ (v0) | Splat, camera path, frame markers, click → source frames, snap-to-view, timeline with camera-wearer position. Plain TypeScript (no React yet). |

Quick start: see [pipeline/README.md](../pipeline/README.md).

## One-paragraph summary

The pipeline ingests a video, picks sharp keyframes, masks out moving people and overlays, estimates camera poses (SfM), normalises the scene to metres with Y up, and trains a **static** Gaussian splat with gsplat. Separately, it tracks people and lifts them onto the ground plane over time; for body-cam footage the wearer *is* the camera trajectory. Detectors and a VLM propose evidence points and timeline events. A person reviews them, and everything is packaged into a static **Scene Package** that a three.js / Spark web viewer renders, with a timeline, actor trails and clickable pins that show the exact source frames.

## Conventions
- Diagrams use Mermaid (renders on GitHub and most Markdown viewers).
- Changes to architecture require a new ADR in `decisions/` using the [template](decisions/template.md).
