# 07 — Technology Stack

Choices favour (1) open-source licences compatible with academic publication, (2) fitting in 4 GB VRAM where possible, and (3) active maintenance.

## 1. Pipeline (Python)

| Concern | Choice | Licence | Why | Alternatives considered |
|---|---|---|---|---|
| Language / runtime | Python 3.11, PyTorch 2.x, CUDA 12 | — | Ecosystem | — |
| CLI | Typer | MIT | Simple, typed | Click |
| Config | Pydantic v2 + YAML (+ OmegaConf merging) | MIT/BSD | Validation, profile overlays | Hydra (heavier) |
| Download | yt-dlp | Unlicense | De-facto standard | — |
| Video / audio | ffmpeg (CLI), PyAV | LGPL | Robust, VFR handling | OpenCV VideoCapture (VFR issues) |
| Shot detection | PySceneDetect | BSD-3 | Proven | TransNetV2 (better on gradual cuts) |
| Segmentation | SAM 2 / SAM 2.1 (hiera-tiny on laptop) | Apache-2.0 | Video propagation | Grounded-SAM 2, YOLO11-seg only |
| Detection | Ultralytics YOLO11 | **AGPL-3.0** ⚠️ | Fast, accurate | RT-DETR (Apache-2.0) if AGPL is an issue |
| Open-vocab detection | Grounding DINO / OWLv2 | Apache-2.0 | Text-prompted evidence | YOLO-World (GPL) |
| Tracking | BoT-SORT / ByteTrack (via `boxmot`) | AGPL / MIT | CMC for moving camera | OC-SORT |
| Features / matching | hloc: SuperPoint + LightGlue, NetVLAD / MegaLoc retrieval | SuperPoint weights: **non-commercial**; LightGlue: Apache | Robust on blur | DISK + LightGlue (Apache), ALIKED |
| SfM | GLOMAP + COLMAP 3.10+ (via `pycolmap`) | BSD | Standard, fisheye support | OpenMVG |
| Feed-forward 3D (fallback) | VGGT, MASt3R-SfM | VGGT: custom non-commercial / MASt3R: CC BY-NC-SA | Sparse / wide-baseline views | DUSt3R, Fast3R |
| Monocular depth | Depth Anything V2 (metric), MoGe, UniDepth | Apache / CC BY-NC (large) | Scale prior, single-view depth | ZoeDepth |
| **3DGS training** | **gsplat** (MCMC strategy) | Apache-2.0 | Memory efficiency, fisheye / app-opt, active | Inria 3DGS (non-commercial), nerfstudio splatfacto (wraps gsplat) |
| Human mesh (optional) | GVHMR / WHAM | non-commercial (SMPL licence) | World-grounded motion | 4D-Humans |
| Pose keypoints | RTMPose (MMPose) | Apache | Ankles for foot point | ViTPose |
| ASR | faster-whisper (turbo / small) | MIT | Word timestamps, fast | WhisperX (diarisation) |
| Audio events | PANNs (CNN14) | MIT | Gunshot / scream classes | YAMNet |
| VLM | Provider-agnostic client: Claude (Anthropic), Gemini, GPT-4-class, or local Qwen2.5-VL (3B/7B) | — | Evidence verification, events | — |
| Splat compression | `spz` (Niantic), `splat-transform` (PlayCanvas, for SOG) | MIT | 10× size reduction | — |
| Data | Parquet (pyarrow), JSON, NumPy | — | — | — |
| Logging / experiments | structlog, optional Weights & Biases / MLflow | — | Ablation tracking | TensorBoard |
| Packaging | uv + Docker (`nvidia/cuda:12.4-devel`) | — | Reproducible CUDA builds | conda |
| Testing | pytest, hypothesis (geometry), golden-file tests on a tiny clip | — | — | — |

### Licensing note (academic)
Several components (SuperPoint weights, VGGT, MASt3R, SMPL, Inria 3DGS) are **non-commercial**. That's fine for academic work but must be listed in the README and any paper. For a future commercial path, swap: SuperPoint → DISK / ALIKED, MASt3R/VGGT → COLMAP-only, YOLO11 → RT-DETR, SMPL → capsules.

## 2. Web app

| Concern | Choice |
|---|---|
| Build | Vite, TypeScript, pnpm |
| UI | React 18, Tailwind, Radix UI |
| 3D | three.js |
| Splats | Spark (`@sparkjsdev/spark`) — fallback `@mkkellogg/gaussian-splats-3d` |
| State | Zustand |
| Schema validation | Zod schemas generated from the pipeline's JSON Schema |
| Review server | FastAPI (shares pydantic models with the pipeline) |
| Testing | Vitest, Playwright (visual regression on a fixture package) |

## 3. Infrastructure

| Need | Choice |
|---|---|
| Local GPU | RTX 3050 Laptop 4 GB (dev, light stages, capped training) |
| Cloud GPU | RunPod / Lambda / Vast.ai / Colab Pro, RTX 4090 / A10G / L4 (24 GB), ~US$0.3–0.8 per hour |
| Sync | rclone or rsync over SSH |
| Hosting (demo, non-sensitive scenes only) | GitHub Pages / Netlify / Cloudflare Pages |
| Source control | Git + Git LFS for small fixture videos; large data outside git |
| CI | GitHub Actions: lint, type-check, unit tests on CPU; GPU tests run manually |
