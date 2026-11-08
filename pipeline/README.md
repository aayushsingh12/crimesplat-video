# crimesplat pipeline

Video → explorable 3D Gaussian Splat Scene Package. Design: [../docs](../docs/README.md).

## Setup

Linux or WSL2 with an NVIDIA GPU, driver ≥ 550, and ffmpeg:
```bash
bash ../scripts/setup.sh        # or: uv sync && uv run crimesplat doctor
```
No CUDA toolkit is needed: torch 2.4.1+cu124 and gsplat 1.5.3 install as prebuilt wheels (Python 3.10).

## Use

```bash
uv run crimesplat init room01 --source ~/Videos/room01.mp4 --footage handheld   # writes ../cases/room01.yaml
uv run crimesplat run room01                    # stages: ingest frames masks sfm normalize train package
uv run crimesplat run room01 --profile gpu-24gb # same case, full quality
uv run crimesplat run room01 --force train      # re-run a stage and everything after it
uv run crimesplat status room01
uv run crimesplat publish-local room01          # copy the package into ../web/public/scenes/
```
Then view it: `npm --prefix ../web run dev` and open `http://localhost:5173/?scene=scenes/room01`.

Stages are cached: re-running only redoes stages whose config (or an upstream stage) changed. Outputs, reports, errors and `provenance.jsonl` live in `../workspace/<case>/`.

## Configuration

A case resolves as `configs/footage/<footage>.yaml` < `configs/profiles/<profile>.yaml` < `cases/<case>.yaml`. Any field in `crimesplat/config.py` can be set in the case file, for example:
```yaml
normalize:
  camera_height_m: 1.6     # height you filmed at; sets metric scale
frames:
  start_s: 5
  end_s: 95
train:
  cap_max: 1000000
```

## Synthetic test scene

`tools/make_synthetic_video.py` renders a walk-through of a textured room with known camera poses (`*.gt.json`). It's for checking that SfM, scale and training work without real footage.
