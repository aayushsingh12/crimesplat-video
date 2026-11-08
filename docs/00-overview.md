# 00 — Project Overview & Requirements

| Field | Value |
|---|---|
| Project | CrimeSplat — Video Pipeline |
| Parent project | Crime Scene Reconstruction using Scene Graphs |
| Status | Static-scene MVP implemented (v0.1); see [README](README.md#implementation-status) |
| Context | Academic / research |
| Last updated | 2026-11-07 |

## 1. Problem statement

Investigators often get a crime scene as **video**: police body-cam footage, CCTV, phone recordings, news or drone footage. Reviewing that video is linear and slow. It's hard to answer spatial questions like *"Where was the weapon relative to the body?"* or *"Which way did the suspect go after 3:20?"* without building a mental 3D map from a shaky 2D stream.

CrimeSplat's video pipeline turns that footage into an **explorable 3D reconstruction** built with 3D Gaussian Splatting (3DGS). On top of the reconstruction it adds:

- **Evidence points**: clickable 3D markers (body, weapon, blood, damage), each linked to the original video frames that show it.
- **Actor tracks over time**: people's positions in 3D, driven by a timeline scrubber, so a viewer can "replay" the incident in space.
- **Events**: timestamped moments (entry, struggle, weapon drawn) that sit on the timeline and are anchored in 3D.

The reference for the visual experience is the *Flight 7598 crash* reconstruction (https://flight-7598-crash.bilawalsidhu.chatgpt.site/). In that demo, a public video was turned into a splat scene you can explore, with clickable points of interest that show the source imagery.

## 2. Relationship to the parent project

The parent project has a **text pipeline** (reports → 4D panoptic scene graph) and an **image pipeline** (CLIP matching of scene photos to reports). This repository is **independent** of both:

- It takes video only and produces a self-contained *Scene Package* (see [05-data-model.md](05-data-model.md)).
- It doesn't import or depend on the text or image pipelines.
- The Scene Package's events and entities are designed so a future integration *could* map them into the scene graph. That integration is an explicit **non-goal** for this repo (see §6).

## 3. Primary user story (the "one-shot example")

> A detective loads a YouTube body-cam video of an officer entering a house where a murder took place. The pipeline extracts frames, reconstructs the house as a Gaussian splat, and produces a dashboard. The detective drags around the 3D house and sees pins for the victim's body and the knife. Clicking the knife pin shows the exact video frames in which it appears. A timeline at the bottom shows that at 0:10 the officer enters (the officer's position is the camera path itself), and that a second person moves from the hallway to the kitchen. At 3:20 a "struggle" event is marked. Scrubbing to it shows both actors' positions and lets the detective jump to the matching video moment.

## 4. Functional requirements

Priority uses MoSCoW (M = must, S = should, C = could).

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Ingest a video from a local file or URL (YouTube etc.) and record its provenance (hash, source, timestamps). | M |
| FR-2 | Reconstruct the **static** scene as a 3D Gaussian Splat that can be orbited, panned and zoomed in a web browser. | M |
| FR-3 | Remove moving people and objects from the static reconstruction (no "ghost" artifacts). | M |
| FR-4 | Estimate the camera trajectory (pose per video frame) in the reconstruction's coordinate frame. | M |
| FR-5 | Detect and track people across the video, and place them in 3D over time. | M |
| FR-6 | Timeline scrubber that animates actor positions and shows the camera (e.g., body-cam wearer) position at time *t*. | M |
| FR-7 | Evidence points: 3D markers that, when clicked, show the source frames (with the region highlighted) in which the item is visible. | M |
| FR-8 | Semi-automatic evidence and event proposal (detector + vision-language model), with a **human review step** before publishing. | M |
| FR-9 | "Click anywhere" lookup: clicking any surface in the splat returns the best source frames that observed that point. | S |
| FR-10 | "Snap to source view": move the 3D camera to the exact pose of a video frame and overlay the frame. | S |
| FR-11 | Multi-camera support: several synchronised CCTV feeds of the same space registered into one reconstruction. | S |
| FR-12 | Event detection from audio (speech transcription of body-cam audio, gunshots). | S |
| FR-13 | Approximate metric scale (metres) and gravity-aligned orientation. | S |
| FR-14 | 3D human body (SMPL) for actors instead of simple markers. | C |
| FR-15 | Measurement tool (distance between two points in metres). | C |
| FR-16 | Export a report (PDF/HTML) of evidence points and events. | C |

## 5. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Runs on the dev laptop** (RTX 3050 Laptop, 4 GB VRAM, 15 GB RAM) for a short clip in "laptop" profile. | ≤ 2 h end-to-end for a 2-min clip |
| NFR-2 | Scales to a rented cloud GPU (≥ 24 GB) with no code changes, only a config profile. | Profile switch only |
| NFR-3 | Viewer runs in a modern desktop browser on integrated graphics. | ≥ 30 FPS at 1080p, scene ≤ 50 MB |
| NFR-4 | **Reproducibility**: same input + config + seed → same Scene Package (bit-identical where the libraries allow). | Stage outputs content-hashed |
| NFR-5 | **Provenance / chain of custody**: every derived artifact traceable to the source video hash and pipeline version. | 100% of artifacts |
| NFR-6 | **Explainability**: every automated claim (evidence point, event) carries a confidence score and the source frames that support it. | 100% |
| NFR-7 | Stage-level caching and resumability: a failed stage can be re-run without redoing earlier stages. | — |
| NFR-8 | Fully open-source core; cloud APIs (VLMs) optional and swappable. | — |

## 6. Scope

### In scope (v1)
- Footage types: **body-cam / handheld**, **multi-camera CCTV with overlapping views**, **YouTube / news / drone**.
- Static-scene 3DGS plus **tracked actors** (dynamic content shown as markers, trails or SMPL bodies, not as dynamic Gaussians).
- Offline batch pipeline (CLI) plus a static web viewer.

### Explicitly out of scope (v1)
- **Single fixed CCTV camera** reconstruction. One static viewpoint gives no parallax, so there's no real 3D. Treated as future work (see [08-risks-and-blockers.md](08-risks-and-blockers.md)).
- Full **dynamic / 4D Gaussian Splatting** (deformable Gaussians). It's a research-grade risk and could be a phase-4 stretch goal.
- Real-time / live-stream processing.
- Integration with the text or image scene-graph pipelines.
- Face recognition or identity resolution. Actors are anonymous track IDs (*Person 1, Person 2*).
- Court-admissible certification. We follow forensic *good practice* but make no legal claims.

## 7. Key design decisions (summary)

Full rationale lives in [decisions/](decisions/).

1. **Static scene + tracked actors** instead of 4DGS ([ADR-0001](decisions/0001-static-scene-plus-tracked-actors.md)).
2. **gsplat** as the 3DGS trainer, for its memory efficiency on 4 GB and its Apache-2.0 license ([ADR-0002](decisions/0002-gsplat-trainer.md)).
3. **COLMAP/GLOMAP + learned features** as the primary pose estimator, with a feed-forward (VGGT / MASt3R) fallback ([ADR-0003](decisions/0003-pose-estimation-strategy.md)).
4. **Hybrid compute**: the laptop for development and light stages, a rented GPU for heavy stages ([ADR-0004](decisions/0004-hybrid-compute.md)).
5. **Human-in-the-loop** review of all AI-proposed evidence and events ([ADR-0005](decisions/0005-human-in-the-loop.md)).

## 8. Glossary

| Term | Meaning |
|---|---|
| 3DGS | 3D Gaussian Splatting: a scene is represented as millions of coloured, semi-transparent 3D Gaussians, optimised from images and rendered in real time. |
| SfM | Structure-from-Motion: estimates camera poses and a sparse 3D point cloud from overlapping images (e.g., COLMAP). |
| Pose | Camera position and orientation (extrinsics) plus lens parameters (intrinsics). |
| Parallax | Apparent shift of objects between views from different positions; needed to recover depth. |
| SMPL | Parametric 3D human body model. |
| VLM | Vision-Language Model (e.g., Qwen2.5-VL, GPT-4o, Claude, Gemini). |
| Scene Package | The pipeline's final output bundle consumed by the viewer. |
| Evidence point | A 3D-anchored annotation (e.g., "knife") with supporting frames. |
| Track | A sequence of detections of the same person across frames. |

How they relate
They form a pipeline, not competing alternatives:

Photos ──► COLMAP (SfM) ──► Camera poses + sparse point cloud
                                      │
                                      ▼
                          3D Gaussian Splatting (training)
                                      │
                                      ▼
                          Trained scene (real-time rendering)

COLMAP solves where the cameras were and produces a sparse point cloud. 
3DGS uses those poses and points as initialization — one Gaussian is seeded at each sparse point — then optimizes all Gaussian parameters via differentiable rendering against the original photos. 
The quality of the final splat depends directly on the accuracy of COLMAP's reconstruction. 
In short: SfM is the algorithm, COLMAP is the tool that runs it, and Gaussian Splatting is the downstream method that consumes COLMAP's output to build a photoreal, real-time 3D scene.