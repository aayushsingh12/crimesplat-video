# 04 — Low-Level Design (Pipeline)

This document specifies each pipeline stage: its interface, algorithm, parameters, failure modes and outputs. The web app is specified in [06-viewer-design.md](06-viewer-design.md).

## 1. Repository layout (proposed)

```
crimesplat-video/
├── docs/                         # this documentation
├── pipeline/                     # Python package `crimesplat`
│   ├── crimesplat/
│   │   ├── cli.py                # Typer entry point
│   │   ├── orchestrator/         # DAG, cache, remote execution
│   │   │   ├── dag.py
│   │   │   ├── cache.py
│   │   │   └── remote.py
│   │   ├── stages/               # one module per stage, pure functions over files
│   │   │   ├── s01_ingest.py
│   │   │   ├── s02_frames.py
│   │   │   ├── s03_masking.py
│   │   │   ├── s04_poses.py
│   │   │   ├── s05_normalise.py
│   │   │   ├── s06_splat.py
│   │   │   ├── s07_actors.py
│   │   │   ├── s08_evidence.py
│   │   │   ├── s09_events.py
│   │   │   └── s10_package.py
│   │   ├── adapters/             # pluggable model wrappers (interfaces + impls)
│   │   │   ├── segmenters/       # sam2.py, yolo_seg.py
│   │   │   ├── pose/             # hloc_glomap.py, colmap.py, vggt.py, mast3r.py
│   │   │   ├── detectors/        # yolo.py, grounding_dino.py, owlv2.py
│   │   │   ├── trackers/         # botsort.py
│   │   │   ├── hmr/              # gvhmr.py
│   │   │   ├── depth/            # depth_anything.py
│   │   │   └── vlm/              # anthropic.py, gemini.py, openai.py, qwen_local.py
│   │   ├── geometry/             # transforms, ray casting, triangulation, visibility
│   │   ├── schemas/              # pydantic models == Scene Package JSON schemas
│   │   └── provenance.py
│   ├── configs/
│   │   ├── default.yaml
│   │   ├── profiles/laptop-4gb.yaml
│   │   ├── profiles/cloud-24gb.yaml
│   │   └── footage/{bodycam,cctv_multi,drone,news}.yaml
│   └── tests/
├── web/                          # Vite + TypeScript viewer / review tool
├── docker/Dockerfile
└── workspace/                    # git-ignored run outputs
```

## 2. Orchestration

### 2.1 Stage contract

```python
class Stage(Protocol):
    name: str
    version: str                      # bump when algorithm changes → invalidates cache
    requires: list[str]               # names of upstream stages
    remote_capable: bool

    def run(self, ctx: StageContext) -> StageResult: ...

@dataclass
class StageContext:
    case_id: str
    inputs: dict[str, Path]           # resolved outputs of upstream stages
    out_dir: Path                     # workspace/<case>/<stage>/<cache_key>/
    config: StageConfig               # pydantic, validated
    device: Literal["cuda", "cpu"]
    vram_budget_mb: int               # from profile; adapters must respect it
    logger: Logger
    provenance: ProvenanceWriter
```

### 2.2 Cache key

```
cache_key = sha256(
    stage.name, stage.version,
    sorted(upstream cache_keys),
    canonical_json(stage config),
    model weight hashes used by the stage,
)[:16]
```

The output dir is `workspace/<case_id>/<stage>/<cache_key>/`, with a `_SUCCESS` marker written last. Re-running with the same config is a no-op. Changing one stage's config invalidates only that stage and its downstream stages.

### 2.3 Memory discipline (4 GB profile)
- One GPU model is resident at a time. Adapters implement `load()` and `unload()`, and the orchestrator calls `torch.cuda.empty_cache()` between stages.
- Adapters read `vram_budget_mb` to pick a model size (e.g., SAM 2 `tiny` when budget < 4000).
- Images flow to the GPU in batches sized from the budget. Full image sets stay on CPU or disk.

### 2.4 CLI

```
crimesplat init <case_id> --source <url|file> [--source ...] --footage bodycam
crimesplat run <case_id> [--profile laptop-4gb] [--until splat] [--only actors] [--force evidence]
crimesplat status <case_id>               # DAG with cached / stale / failed per stage
crimesplat review <case_id>               # launches local review server
crimesplat package <case_id> [--include proposed]
crimesplat eval <case_id> --gt <gt.json>  # metrics, see 09-evaluation-and-ethics.md
```

## 3. Stage specifications

### S01 — Ingest

| | |
|---|---|
| **Input** | One or more sources (URL or path) plus optional per-source metadata (camera id, known start wall-clock time). |
| **Output** | `source_<i>.mp4` (original, untouched), `norm_<i>.mp4` (constant frame rate, H.264, ≤ 1080p), `audio_<i>.wav` (16 kHz mono), `probe_<i>.json`, provenance entries. |

Algorithm:
1. Download with `yt-dlp` (best video+audio, no re-encode). Record URL, title, uploader, upload date and the yt-dlp version.
2. Compute **SHA-256 of the original bytes** before any processing. This anchors the chain of custody.
3. `ffprobe` → codec, resolution, fps (detect VFR), duration, rotation metadata.
4. Normalise to CFR with `ffmpeg -vsync cfr`, keeping the original timestamp mapping (`pts_map.json`). VFR phone video otherwise breaks time alignment.
5. Extract audio.

Failure modes: geo-blocked or removed URLs (fail clearly), DRM (unsupported), corrupted streams (ffmpeg error → fail with log).

### S02 — Shot & frame selection

**Goal:** pick ~150–400 *keyframes* that are sharp, well-spread in viewpoint, and cover the scene. Also index *every* frame for the later stages.

1. **Shot boundary detection** with PySceneDetect (`ContentDetector`, threshold ≈ 27). Each shot is reconstructed independently first, then merged in S04. Static overlay regions (news tickers, timestamps) are found by pixel-variance analysis across the shot: pixels with near-zero temporal variance plus high edge density → `overlay_mask.png`.
2. **Letterbox / pillarbox detection** → crop rectangle.
3. **Sharpness score** per frame = variance of the Laplacian on the downscaled grey image, normalised per shot (percentile rank). Frames below the 30th percentile are rejected.
4. **Viewpoint-change selection:** compute sparse optical flow (or ORB match displacement) relative to the last keyframe. Add a new keyframe when median displacement > `flow_thresh` (≈ 5% of image width) **or** after `max_gap` seconds. Among candidates in a ±0.25 s window, choose the sharpest.
5. **Pure-rotation flag:** if a homography explains > 90% of matches between consecutive keyframes, flag the segment `low_parallax`. Those frames are kept for localisation but down-weighted in training.
6. Write `frames.parquet` (all frames) and `keyframes/<frame_id>.jpg` (full res) plus `keyframes_ds/` (training resolution, e.g., 960 px wide in the laptop profile).

Key parameters: `target_keyframes: 250`, `flow_thresh: 0.05`, `max_gap_s: 1.0`, `min_sharpness_pct: 30`.

### S03 — Dynamic masking

**Goal:** a binary mask per keyframe marking pixels that must *not* contribute to static geometry.

Mask classes (unioned for SfM and training, but stored separately):
- `person`, `vehicle`, `animal` (dynamic objects)
- `overlay` (burned-in text and logos, from S02)
- `redaction` (solid blur or black boxes from footage release. Detected as regions of very low high-frequency energy with sharp rectangular or elliptical borders.)
- `wearer` (the body-cam wearer's own arms, hands or weapon visible at the frame bottom. Detected as person masks touching the bottom edge.)

Algorithm:
1. Detect with YOLO11-seg (classes person / vehicle / animal) on keyframes at stride 1, and on all frames at the tracking fps.
2. Refine and propagate with **SAM 2 video predictor**, prompted by YOLO boxes every N frames. This gives temporally consistent masks, including partially occluded people.
3. Dilate masks by 1–2% of image width (shadows, motion-blur halos).
4. Save `masks/<frame_id>.png` (union) and `masks_by_class/`.

Laptop profile: SAM 2 `hiera_tiny`, processed in shot-chunks of ≤ 200 frames.

Note: S07 reuses the person detections. S03 writes `detections.parquet` so they aren't computed twice.

### S04 — Camera pose estimation

**Primary path (hloc + GLOMAP):**
1. **Camera model.** Footage profile `bodycam` → `OPENCV_FISHEYE`. `drone` / `news` → `OPENCV`. One shared camera per source video (`single_camera_per_folder`).
2. **Features:** SuperPoint (max 2048 keypoints), with masked regions removed before matching.
3. **Pairs:** sequential (each keyframe with the next 10–20) + retrieval (NetVLAD / MegaLoc top-20) for loop closure, e.g., when the officer re-enters a room.
4. **Matching:** LightGlue.
5. **Mapping:** GLOMAP global SfM (fast, robust). If it registers < 60% of keyframes, retry with COLMAP incremental mapping.
6. **Bundle adjustment** refining intrinsics (incl. distortion).
7. **Multiple shots / sources:** each shot maps independently. Models are merged when cross-shot retrieval finds ≥ 30 verified matches (`colmap model_merger`). Shots that don't merge are kept as separate *sub-scenes* and reported.
8. **All-frame localisation:** every non-keyframe at the tracking fps (default 10 fps) is localised against the model (SuperPoint + LightGlue → 2D–3D matches → PnP-RANSAC). Failures are filled by SLERP / linear interpolation between neighbouring successes if the gap is < 1 s, and flagged `interpolated: true`.

**Fallback path (feed-forward, cloud):** used when the primary path fails, or by default for `cctv_multi` footage:
- **VGGT** or **MASt3R-SfM** predicts poses + dense point maps directly from images. Good for wide baselines and few views.
- The dense points initialise 3DGS (InstantSplat-style), which helps a lot for sparse views.

**Multi-CCTV workflow:**
1. *Preferred:* a survey video (walk-through) is reconstructed with the primary path. Each CCTV camera is then **localised** into it (PnP from a few clear frames), giving that camera's fixed pose. CCTV frames are used for tracking only, not training.
2. *Without a survey:* MASt3R-SfM / VGGT on one or more frames per camera (people masked; ideally a frame with an empty scene, or a temporal median background image). Expect a coarse splat.

Outputs: COLMAP model (`cameras.bin`, `images.bin`, `points3D.bin`), `poses_all_frames.json`, `registration_report.json` (registered %, mean reprojection error, per-shot status).

Quality gates (warn, don't fail): registered keyframes ≥ 80%, mean reprojection error ≤ 1.5 px, track length mean ≥ 3.

### S05 — Scene normalisation

**Goal:** a canonical scene frame with Y up, metres, and the origin on the floor below the first camera position.

1. **Ground plane:** RANSAC plane fit on sparse points that are (a) below the camera trajectory and (b) roughly perpendicular to the dominant "up". A first guess for "up" is the mean camera −Y axis; body-cams are mostly upright. Refine with vanishing-point / Manhattan alignment (COLMAP `model_orientation_aligner`).
2. **Metric scale**, by priority:
   1. *Body-cam height prior:* the median camera height above the ground plane ≈ **1.45 m** (chest mount). Configurable (head mount ≈ 1.7 m).
   2. *Monocular metric depth* (Depth Anything V2 metric / UniDepth / MoGe) on ~20 keyframes. Robust median ratio between metric depth and SfM depth at sparse points.
   3. *Known objects:* the user marks a door (2.03 m) or similar in the review tool. This overrides the automatic estimates.
   
   Final scale = the user value if given, otherwise the median of the available estimates. The spread between estimates is reported as `scale_uncertainty`.
3. Apply the similarity transform to the model and write `transform.json` (4×4 + scale), so raw COLMAP coordinates stay recoverable.

### S06 — Gaussian splat training

Built on **gsplat** ([ADR-0002](decisions/0002-gsplat-trainer.md)).

| Setting | Laptop profile (4 GB) | Cloud profile (24 GB) |
|---|---|---|
| Strategy | MCMC, `cap_max = 400 000` | MCMC, `cap_max = 2 000 000` (or Default + absgrad) |
| Training resolution | 960 px wide | 1600–1920 px wide |
| Iterations | 15 000 (preview: 7 000) | 30 000 |
| SH degree | 2 | 3 |
| Rasteriser | `packed=True`, `sparse_grad` | default |
| Images on GPU | no (stream from pinned CPU memory) | yes |
| Expected time | 30–75 min | 10–25 min |

Additions on top of vanilla training:
1. **Masked loss:** L1 + D-SSIM computed only on unmasked pixels (`mask==0`). This is what removes ghosts.
2. **Fisheye:** train on undistorted images (`colmap image_undistorter`), cropping black borders. Undistortion maps are stored so overlays can be re-distorted to match source frames.
3. **Per-image appearance embedding** (exposure / white-balance compensation, as in gsplat's `app_opt`). Body-cams auto-expose wildly when moving between rooms.
4. **Low-parallax down-weighting:** frames flagged in S02 are sampled less often.
5. **Robust loss (optional):** a SpotLessSplats-style outlier mask for residual transients that S03 missed.
6. **Held-out evaluation:** every 8th keyframe is excluded from training and used for PSNR / SSIM / LPIPS (`metrics.json`).
7. **Depth maps for visibility:** after training, render a depth map per keyframe at ¼ resolution. These feed the click-anywhere visibility index (S10).

Outputs: `scene.ply`, `metrics.json`, `depth/<frame_id>.npy` (float16), training-curve logs.

### S07 — Actor tracking & 3D lifting

1. **Detection:** reuse S03 detections (YOLO11, persons) at 10 fps for all frames.
2. **Tracking:** BoT-SORT with ReID (OSNet). Camera-motion compensation is enabled, which is important for body-cam.
3. **Foot point:** bottom-centre of the bbox. If RTMPose / ViTPose keypoints are available and the ankles are visible, use the ankle midpoint instead. Truncated boxes (touching the image bottom) are marked `foot_unobserved`.
4. **Lifting:** cast a ray from the camera centre through the (undistorted) foot pixel and intersect it with the ground plane → `(x, 0, z)`. For `foot_unobserved`, use the torso centre with a height prior of 1.0 m above ground, or the rendered splat depth at the torso pixel.
5. **Multi-floor / stairs:** v1 assumes a single ground plane per sub-scene. Stairs are a known limitation (see the risks doc).
6. **Smoothing:** a constant-velocity Kalman filter with an RTS smoother in the ground plane. Outliers (jump > 3 m within 0.5 s) are rejected.
7. **Track stitching:** tracklets of the same person broken by occlusion are merged when the 3D gap is plausible (speed < 3 m/s) and the ReID cosine similarity is > 0.6.
8. **Camera wearer:** added as actor `wearer` with positions = the camera centre projected to the ground plane, plus heading = camera forward vector.
9. **Multi-CCTV:** detections from all cameras are lifted independently, then associated across cameras by 3D proximity at the same timestamp (after time sync) → fused positions.
10. **Optional SMPL (cloud):** GVHMR / WHAM per track → SMPL params in camera coordinates → transformed to the scene frame and placed so the root's ground projection matches the lifted foot point.

Output: `tracks.json` (see the data model). Positions are stored at 10 Hz. The viewer interpolates.

### S08 — Evidence proposal

1. **Prompt vocabulary** (configurable): `person lying on floor`, `knife`, `gun`, `blood`, `shell casing`, `broken glass`, `phone`, `bag`, `vehicle damage`, … plus case-specific terms the operator adds.
2. **Open-vocab detection** on keyframes: Grounding DINO / OWLv2 (`box_thresh 0.30`, `text_thresh 0.25`). This is permissive on purpose; precision comes later.
3. **SAM 2** box → mask for each detection.
4. **Cross-view association:** group detections of the same label across keyframes whose mask centroids are mutually consistent (epipolar distance < 2% of image diagonal) and whose appearance embeddings (DINOv2 crop) are similar.
5. **3D localisation:** least-squares triangulation of mask centroids across the group (≥ 2 views with baseline angle > 5°). Otherwise, unproject the centroid using the S06 rendered depth.
6. **VLM verification:** send the best 2–3 crops plus one context frame with a structured prompt → `{is_match: bool, label: str, description: str, confidence: 0–1}`. JSON output is enforced and validated, and invalid JSON is retried once.
7. **Score** = `det_conf × vlm_conf × min(1, n_views/3)`. Candidates below `0.25` are dropped. The others go to `evidence.proposed.json` with `status: "proposed"`.
8. **Supporting frames:** all keyframes (and all-frame index matches) where the triangulated point projects inside the image, passes the depth-occlusion test, and overlaps the detection mask.

### S09 — Event proposal

Sources fused on a common timeline:

| Source | Method | Example events |
|---|---|---|
| VLM over windows | 8 s windows, stride 4 s, 1–2 fps frame samples (≤ 8 frames per call), prompt with an **event taxonomy** | `entry`, `exit`, `confrontation`, `struggle`, `weapon_visible`, `person_down`, `fleeing`, `medical_aid` |
| Speech | Whisper (turbo / small) with word timestamps. A keyword / intent pass ("drop the weapon", "shots fired", "call EMS") | `verbal_command`, `shots_fired_call` |
| Audio events | PANNs / YAMNet classifier | `gunshot`, `scream`, `glass_break` |
| Track heuristics | rules over `tracks.json` | `actor_enters_scene`, `actors_in_contact` (dist < 0.8 m for > 1 s), `rapid_motion` |

Fusion: candidates of compatible types whose intervals overlap by IoU > 0.3 are merged, and confidences combine as `1 − Π(1 − c_i)`. Each event gets `actors_involved` (track IDs active and nearby) and `location` (the centroid of those actors at the event midpoint, or the camera position).

**Hallucination controls:** VLMs must cite the frame indices they base a claim on. Claims whose cited frames don't contain a relevant detection are down-weighted ×0.5. Everything stays `proposed` until reviewed.

### S10 — Packaging

1. Compress the splat: `scene.ply` → **SPZ** (Niantic, ~10× smaller) for the default viewer. Optionally a SOG/PlayCanvas or `.ksplat` variant.
2. Generate an LOD / preview splat (≤ 150 k Gaussians) for fast first paint.
3. Export camera trajectory, tracks, events and annotations to the Scene Package schema (validated with JSON Schema generated from the pydantic models).
4. **Visibility index:** per keyframe, store pose, intrinsics and a ¼-res depth (16-bit PNG, metres × 1000). The viewer uses these to answer "which frames see point X?" client-side.
5. Thumbnails: 480 px JPEGs for all keyframes, plus 1080 p for frames referenced by evidence.
6. Write `manifest.json` with versions, hashes of every file, and the provenance chain.

## 4. Configuration example

```yaml
# configs/cases/case01.yaml
case_id: case01
footage: bodycam                 # pulls configs/footage/bodycam.yaml
profile: laptop-4gb
sources:
  - uri: https://www.youtube.com/watch?v=XXXX
    camera_id: officer_1
    mount: chest                 # used by the S05 scale prior
frames:
  target_keyframes: 250
evidence:
  vocabulary_extra: ["kitchen knife", "blood smear on wall"]
vlm:
  provider: anthropic            # anthropic | gemini | openai | local_qwen | none
  model: claude-sonnet-5-5
remote:
  stages: [splat]                # run only S06 remotely
  target: runpod
```

## 5. Error handling and observability

- Each stage writes a `report.json` with `status: ok | degraded | failed`, warnings, metrics and timings. `degraded` means a fallback was used (e.g., interpolated poses, unscaled scene).
- Logs: structured JSON (structlog) per stage, plus a run-level summary in `crimesplat status`.
- Optional experiment tracking with W&B or MLflow for the research ablations (training curves, metrics).
- The pipeline **never silently drops** data. Unregistered shots, failed tracks and rejected candidates are all listed in the reports.
