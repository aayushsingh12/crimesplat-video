# GPU Window Runbook (2 days on an RTX 40-series machine)

The goal is to spend the GPU window **only on full-quality training and experiments**. Everything else (filming, SfM checks, debugging) happens on the laptop beforehand.

## 0. What is already verified (2026-11-08, on the RTX 3050 Laptop)

| Check | Result |
|---|---|
| Full pipeline `ingest → frames → masks → sfm → normalize → train → package` | ✅ runs end-to-end on a synthetic room |
| Camera poses vs. ground truth (incremental SfM) | 4 mm mean trajectory error, 0.5° up-vector error |
| Metric scale from the camera-height prior | 1.5% error |
| Splat quality, laptop profile (7k iters, 960 px) | 28.6 dB held-out PSNR, 234k Gaussians, 7.7 min, 0.5 GiB peak VRAM |
| `gpu-24gb` profile code paths (unpacked raster, SH 3, YOLO11x) | ✅ run at toy size |
| Viewer: splat, camera path, frame markers, click → source frames, snap-to-view, timeline | ✅ |

**Not yet verified:** real footage (yours), people masking on real people, and full-size `gpu-24gb` memory use.

## 1. Before the window (on the laptop)

### 1.1 Film the test videos
Film 2–4 clips. Each should be 1–3 minutes, landscape, 1080p or 4K at 30 fps.

| Do | Don't |
|---|---|
| Walk slowly. Keep moving sideways or forwards; parallax is what creates 3D. | Stand in one place and only turn (pure rotation gives no depth). |
| Keep the camera at a steady height. Note it (e.g., chest ≈ 1.45 m, eye ≈ 1.6 m). | Use digital zoom, or switch lenses mid-clip (0.5× ↔ 1×). |
| Good, even light. Textured surfaces in view. | Point at blank walls, mirrors or windows for long stretches. |
| Revisit the start position at the end (closes the loop). | Use stabilisation that crops/warps heavily, or cut / edit the clip. |
| Include one clip with a person walking through the scene (tests masking). | Let people stand in front of the camera for most of the clip. |
| One clip in "body-cam style": phone chest-mounted, walking room to room. | |

Lock exposure and focus if your phone allows it (most camera apps: long-press).

### 1.2 Run each clip on the laptop
```bash
cd pipeline
uv run crimesplat init room01 --source ~/Videos/room01.mp4 --footage handheld
uv run crimesplat run room01                 # laptop-4gb profile
uv run crimesplat status room01
uv run crimesplat publish-local room01      # then open the viewer
```
Set `normalize.camera_height_m` in `cases/room01.yaml` to your actual filming height. Use `--footage bodycam` only for genuinely wide/fisheye lenses; a phone's main lens is `handheld`.

Check each clip passes:
- `workspace/<case>/sfm/registration_report.json`: `registered_frac` ≥ 0.8, reprojection error ≤ 1.5 px.
- `workspace/<case>/normalize/report.json`: no warnings, `floor_points` in the hundreds.
- `workspace/<case>/train/eval_renders/`: held-out renders look like the left half.
- In the viewer: the floor is flat and horizontal, the camera path looks right.

Re-film any clip that fails SfM rather than debugging it on GPU time.

### 1.3 Pack for the GPU machine
- This repository (git push, or a zip without `workspace/`, `data/`, `.venv/`, `node_modules/`).
- The source videos (`data/`), plus each case's `cases/<case>.yaml`.
- Optionally the laptop's finished `workspace/<case>/sfm` and `normalize` folders. If the configs match, those stages are cached and skipped, saving 10–20 min per clip (copy the whole `workspace/<case>/` folder).

## 2. Day 1 morning: setup (target: 30 min)

```bash
bash scripts/setup.sh          # installs uv, Python env, model weights, viewer; runs `crimesplat doctor`
```
`doctor` must show ✓ for GPU, CUDA kernels, COLMAP and ffmpeg, and suggests a profile:
- ≥ 20 GB VRAM (4090, 3090) → `gpu-24gb`
- 8–16 GB (4060/4070/4080) → `gpu-8gb`

**If the machine is Windows:** use WSL2 (Ubuntu) with the Windows NVIDIA driver, then follow the same steps inside WSL. Native Windows works in principle (gsplat ships a Windows wheel), but it hasn't been tested.

**If the GPU is an RTX 50-series** (not 40-series): the CUDA 12.4 wheels don't support it. Stop and ask before spending time on it.

## 3. Day 1: full-quality runs

For each clip, from the cached laptop state:
```bash
uv run crimesplat run room01 --profile gpu-24gb
```
Changing the profile changes the `frames`/`masks`/`train` config, so those stages (and SfM after them) re-run at full quality. Expected per 2-minute clip on a 4090: frames + masks ~5 min, SfM (CPU) 10–25 min, training (30k iters) 20–40 min.

Run clips back to back in one shell loop and leave it overnight if needed:
```bash
for c in room01 room02 bodycam01; do uv run crimesplat run $c --profile gpu-24gb || echo "$c FAILED"; done
```

## 4. Day 2: experiments worth the GPU (pick by priority)

1. **Quality ablation for the report:** `train.cap_max` 500k / 1M / 2M and `train.max_long_side` 1280 / 1600 / 1920 on one clip. Record PSNR, Gaussian count, training time and peak VRAM (all in `train/metrics.json`).
2. **Masking ablation:** the people clip with `masks.enabled: false` vs. `true` (ghosting before/after). This is a figure for the paper.
3. **SfM ablation:** `sfm.mapper: global` vs. `incremental` on real footage.
4. **Stretch:** the 4D-GS event-clip spike (`docs/research/4dgs-feasibility.md`). Only if 1–3 are done.

Use separate case names for ablations (copy the case YAML, change `case_id` and one setting) so results don't overwrite each other.

## 5. Before the window closes (last 2 hours)

- Copy back `workspace/<case>/package/` for every case (the viewer needs only these), plus `train/metrics.json`, `train/eval_renders/` and `provenance.jsonl` for the write-up.
- The packages open on the laptop viewer; no GPU is needed to view them.
- Delete any real/sensitive footage from the borrowed machine.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch sees no CUDA GPU` | Driver missing or too old for CUDA 12.4 (needs ≥ 550). On WSL2, update the Windows driver. |
| CUDA out of memory in `train` | Use the next profile down, or lower `train.cap_max` / `train.max_long_side` in the case YAML. |
| SfM registers < 60% of frames | Footage problem (blur, pure rotation, featureless walls). Try `--force sfm` with `sfm.mapper: global`, or `frames.target_keyframes: 400`. |
| Floor/scale warning in `normalize` | Set `normalize.camera_height_m` correctly. Scenes without visible floor stay unitless. |
| "N shots detected" warning on an uncut clip | Fast pans can look like cuts to the shot detector. Harmless for unedited footage. |
| A stage failed | Read `workspace/<case>/<stage>/error.txt`; fix; rerun the same command (earlier stages stay cached). |
