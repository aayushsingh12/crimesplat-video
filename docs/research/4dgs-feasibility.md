# Research Note — Is 4D (Dynamic) Gaussian Splatting Feasible for CrimeSplat?

| Field | Value |
|---|---|
| Papers reviewed | (A) Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024 ([arXiv 2310.08528](https://arxiv.org/abs/2310.08528)) · (B) Luiten et al., *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis*, 3DV 2024 ([arXiv 2308.09713](https://arxiv.org/abs/2308.09713)) |
| Code reviewed | (A) [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) (Apache-2.0, last news entry 2024-06-25, ~130 open issues) · (B) [JonathonLuiten/Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians) (MIT + Inria non-commercial rasteriser; data-prep code for custom captures not released) |
| Date | 2026-11-07 |
| Verdict | **4D-GS: feasible only as a bounded research extension on short event clips (cloud GPU). Dynamic 3D Gaussians: not feasible for our footage; it needs ~27 synced cameras. Usable only in a staged multi-phone lab experiment.** Neither replaces [ADR-0001](../decisions/0001-static-scene-plus-tracked-actors.md). |

## 1. What 4D-GS does

- Keeps **one canonical set of 3D Gaussians** (like normal 3DGS).
- A **Gaussian deformation field** predicts, for each Gaussian and timestamp *t*, offsets to position, rotation and scale. It has two parts:
  - a *spatial-temporal structure encoder*: a **HexPlane**, i.e., six 2D feature planes (xy, xz, yz, xt, yt, zt), multi-resolution, queried per Gaussian;
  - a tiny MLP *multi-head decoder* (Δposition, Δrotation, Δscale).
- Rendering at time *t* = deform canonical Gaussians → standard 3DGS rasterisation. That's why it stays real-time.
- Training: a static "warm-up" (plain 3DGS on SfM points), then joint optimisation of Gaussians + deformation field.

## 2. Reported results (from the paper, single RTX 3090 24 GB)

| Dataset | Setting | Frames / length | Resolution | PSNR | FPS | Train time | Storage |
|---|---|---|---|---|---|---|---|
| D-NeRF synthetic | monocular, but *near-random camera per timestamp* | 50–200 frames | 800×800 | 34.05 | 82 | 20 min | 18 MB |
| HyperNeRF vrig | real, 1–2 handheld phones | short clips | 960×540 | 25.2 | 34 | 1 h | 61 MB |
| Neu3D (DyNeRF) | real, **15–20 static synced cameras** | ~10 s (300 frames) | 1352×1014 | 31.15 | 30 | 40 min | 90 MB |

For comparison, plain 3D-GS on HyperNeRF gets 19.7 dB (it can't model motion), so 4D-GS is a real improvement *on those benchmarks*.

## 3. The paper's own limitations (§5.5 and appendix A.4)

The authors say plainly that the method struggles with:
1. **Large motions.** In the appendix's sports-dataset and teapot examples, the deformation field "may fail in modeling large motions", producing blur.
2. **Imprecise camera poses** and **absence of background points.**
3. **Monocular input:** novel-view rendering *fails* on real monocular footage (the iPhone / DyCheck examples in Fig. 16) because the model overfits the training views. They suggest depth or optical-flow priors as future work.
4. **No static/dynamic separation** in the monocular setting without extra supervision.
5. **Scale:** querying the deformation field for huge numbers of Gaussians doesn't scale to large (urban) scenes.

## 4. Mapping to our footage

| Our condition | Benchmark condition | Gap | Assessment |
|---|---|---|---|
| **Body-cam**: one moving camera, people walking metres through several rooms, minutes long | HyperNeRF: 1–2 phones, small object motion, a few seconds | Monocular ✗, large motion ✗, long duration ✗, motion blur ✗, fisheye ✗ | ❌ Whole-video 4D-GS will overfit. Dynamic people will look fine from the original camera path and break from any other viewpoint, which is exactly what the analyst wants to do. |
| **Multi-CCTV**: 3–6 wide-baseline fixed cameras | Neu3D: 15–20 closely spaced, synced, front-facing cameras | 3–5× fewer views, wide baselines, clock drift, low resolution / compression | ⚠️ Too sparse for a good 4D fit. Plausible for a short window if views overlap heavily. |
| **Drone / news** | — | Edited cuts; motion mostly static | ➖ Little dynamic content worth modelling. Static 3DGS already covers it. |
| **Duration** | 10 s clips | Our videos are 1–10 min | ❌ HexPlane's time axis has finite resolution. Minutes of footage would need chunking into ~10 s windows (the repo ships a `merge_many_4dgs.py` utility, but stitching is not addressed in the paper). |
| **Our hardware** | RTX 3090, 24 GB | RTX 3050 Laptop, **4 GB** | ❌ locally. The repo has no low-memory mode (it's based on the original Inria rasteriser, not gsplat). Plan on a cloud GPU. |
| **Software stack** | PyTorch 1.13 + CUDA 11.6, Python 3.7 | Our stack: PyTorch 2.x, CUDA 12 | ⚠️ Needs its own Docker image. It isn't co-installable with gsplat / SAM 2. |

### 4.1 The fundamental issue for forensic use

Even where 4D-GS works, a reconstructed person is **interpolated appearance**, not an observation. From any viewpoint the camera never had, the moving person's shape is *invented* by the deformation field. For a crime-scene tool that's a credibility problem: a viewer could "see" a hand position that never appeared in any frame. The static scene plus tracked actors keeps a clean line: **geometry we observed, plus positions we measured, each with a confidence and source frame.**

## 4b. Dynamic 3D Gaussians (Luiten et al.)

### What it does
- **Online, per-timestep** reconstruction. Timestep 1 is a normal 3DGS fit. For each later timestep, only each Gaussian's **position and rotation** are optimised. Colour, opacity and size stay fixed, so every Gaussian keeps a **persistent identity** over time.
- Physically inspired regularisers: **local rigidity** (neighbours move as a rigid body between timesteps), **local rotation similarity**, and **long-term local isometry** (neighbour distances are preserved over the whole sequence).
- Because Gaussians persist, the output is **dense 6-DOF tracking of every point in the scene**, not just novel views. This is the method's real selling point.

### Reported results (single RTX 3090)
| Data | Cameras | Length | PSNR | FPS | Training |
|---|---|---|---|---|---|
| CMU Panoptic Studio (PanopticSports, 6 sequences) | **27 synced, calibrated training cameras** (+4 test); first-frame point cloud **from depth cameras** | 150 timesteps | 28.7 | ~850 | ~2 h (≈4 min first timestep + ≈50 s per later timestep) |

Scenes use 200–300 k Gaussians. 3D tracking error averages about 2 cm on their benchmark.

### Requirements and limitations relevant to us
1. **Dense multi-view capture is fundamental.** The method is built on 27 synchronised, calibrated cameras. The paper is explicitly multi-camera and contrasts itself with monocular methods. There is **no monocular mode**.
2. **Persistent Gaussian set:** everything must exist at timestep 1. People or objects *entering* later (an officer coming through a door) aren't naturally handled.
3. Needs good **foreground/background masks**. The README admits its own masks are poor and hurt quality near feet and the floor.
4. Data-prep code for new captures was never released. Custom data needs our own converter.
5. Same old stack (Inria rasteriser, non-commercial). Not low-memory. Not usable on 4 GB.

### Fit with our footage
| Footage | Assessment |
|---|---|
| Body-cam (1 moving camera) | ❌ Not applicable. One view per timestep can't constrain per-timestep Gaussian motion. |
| Multi-CCTV (3–6 cameras) | ❌ / ⚠️ An order of magnitude fewer views than Panoptic Studio, uncalibrated, unsynced, compressed. Expect failure without strong priors. |
| Staged lab capture (many synced phones) | ✅ Possible as a **controlled research experiment**, e.g., 8–12 phones around a staged scene, to study what dense 4D tracking could add. |

**Conceptual takeaway worth stealing:** the *persistent identity + local rigidity* idea is exactly what our phase-4 "Event Clip 4D" module needs. Our tracked actors already give a coarse persistent identity, and a rigidity loss on person-mask Gaussians is a natural regulariser.

## 4c. Side-by-side

| | Static 3DGS + tracked actors (our plan) | 4D-GS (Wu et al.) | Dynamic 3D Gaussians (Luiten et al.) |
|---|---|---|---|
| Motion model | People as measured 3D tracks (+ optional SMPL) | Canonical Gaussians + HexPlane deformation field | Per-timestep Gaussian positions/rotations, rigidity priors |
| Works with 1 moving camera | ✅ | ⚠️ trained views only, novel views fail | ❌ |
| Works with 3–6 CCTV | ✅ (tracks) / ⚠️ (scene) | ⚠️ | ❌ |
| Minutes-long footage | ✅ | ❌ (needs ~10 s chunks) | ❌ (150 timesteps ≈ 2 h training) |
| Runs on 4 GB | ✅ (capped gsplat) | ❌ | ❌ |
| Forensic honesty | High: every position has a source frame + confidence | Low: interpolated appearance from unseen views | Medium: physically regularised, but still invented between views |
| Output for analyst | Positions, trails, events | Free-view video of a short clip | Free-view video + dense point tracks |

## 5. Recommendation

1. **Keep [ADR-0001](../decisions/0001-static-scene-plus-tracked-actors.md)** (static splat + tracked actors) as the product path. That's where FR-5/FR-6 (people over time) are met.
2. **Add an optional "Event Clip 4D" research module (phase 4)**:
   - Input: a short window around a confirmed event (e.g., the 3:20 struggle, 5–10 s), cropped to the dynamic region.
   - Initialise from our **already-trained static splat + poses** (fixing the paper's "missing background points / imprecise pose" failure modes).
   - Freeze background Gaussians. Let the deformation field act only on Gaussians inside the person masks (our S03 masks give the static/dynamic separation the paper says it lacks).
   - Add depth (Depth Anything / rendered static depth) and 2D-track supervision. Those are the priors the paper recommends.
   - Add Luiten-style **local-rigidity / isometry losses** on the dynamic Gaussians.
   - Show it in the viewer as a **clearly labelled "reconstructed motion (approximate)"** layer, viewable mainly near the original camera path.
   - Run on a cloud GPU in its own Docker image.
3. **Evaluate it as a research contribution**: held-out-frame PSNR on the event window, and novel-view quality on the multi-CCTV case vs. our actor-track baseline. That's a strong ablation for a paper / thesis chapter.
4. **Also benchmark newer monocular-dynamic methods** that build in the priors 4D-GS lacks, before committing to 4D-GS specifically: *Shape of Motion* (Wang et al., 2024), *MoSca* (Lei et al., 2024), and *MegaSaM* (2024–25) for poses in dynamic video. These are designed for casual monocular video, which is our body-cam case.

## 6. Effort estimate for the phase-4 spike

| Task | Estimate |
|---|---|
| Dockerise 4DGaussians, run on HyperNeRF + one Neu3D scene to reproduce numbers (cloud GPU) | 2–3 days |
| Converter: our Scene Package (poses, masks, static PLY) → 4DGaussians input format | 2–3 days |
| Masked / frozen-background variant + depth prior | 1–2 weeks |
| Evaluation on 2 event clips (body-cam staged + multi-camera staged) | 1 week |
| Viewer: load per-window deformed PLY sequence (export via `export_perframe_3DGS.py`) and play it in sync with the timeline | 3–5 days |

Cloud cost: roughly 20–40 GPU-hours at RTX 4090 / A10 class rates.
