"""Case configuration: defaults, overlaid by footage preset, compute profile, then the case file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class SourceConfig(BaseModel):
    uri: str  # local path or URL
    camera_id: str = "cam0"


class FramesConfig(BaseModel):
    target_keyframes: int = 250
    max_long_side: int = 1920  # keyframes are stored at this size; training may downscale further
    candidate_fps: float = 6.0  # frames decoded per second when choosing keyframes
    min_sharpness_pct: float = 30.0  # reject frames below this percentile within their time bin
    start_s: float = 0.0
    end_s: float | None = None


class MaskConfig(BaseModel):
    enabled: bool = True
    model: str = "yolo11s-seg.pt"
    # COCO ids: person, bicycle, car, motorcycle, bus, truck, bird, cat, dog, horse
    classes: list[int] = [0, 1, 2, 3, 5, 7, 14, 15, 16, 17]
    conf: float = 0.25
    dilate_frac: float = 0.015  # dilation kernel as a fraction of image width
    max_masked_frac: float = 0.7  # frames masked more than this are dropped from SfM


class SfmConfig(BaseModel):
    camera_model: str = "SIMPLE_RADIAL"  # SIMPLE_RADIAL | OPENCV | OPENCV_FISHEYE | PINHOLE
    # Incremental is slower but more accurate on video, which has no EXIF focal prior. On the
    # synthetic room: incremental 4 mm mean ATE / 0.5 deg up error, GLOMAP 19 mm (29 cm max) / 4.7 deg.
    mapper: Literal["global", "incremental"] = "incremental"
    max_image_size: int = 1600
    max_num_features: int = 8192
    sequential_overlap: int = 15
    exhaustive_below: int = 120  # use exhaustive matching when there are fewer keyframes
    min_registered_frac: float = 0.6  # below this the other mapper is tried


class NormalizeConfig(BaseModel):
    # Height of the camera above the floor, used to put the scene in metres.
    camera_height_m: float | None = 1.5


class TrainConfig(BaseModel):
    max_long_side: int = 960
    iterations: int = 15_000
    cap_max: int = 400_000
    sh_degree: int = 2
    sh_increase_every: int = 1000
    ssim_lambda: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    holdout_every: int = 8  # every Nth keyframe is held out for evaluation; 0 disables
    packed: bool = True
    log_every: int = 500
    seed: int = 0


class CaseConfig(BaseModel):
    case_id: str
    footage: str = "handheld"
    profile: str = "laptop-4gb"
    sources: list[SourceConfig]
    frames: FramesConfig = Field(default_factory=FramesConfig)
    masks: MaskConfig = Field(default_factory=MaskConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_case(path: Path, profile: str | None = None) -> CaseConfig:
    """Resolve a case file: footage preset < profile < case file < CLI profile override."""
    case = _read_yaml(path)
    footage = case.get("footage", "handheld")
    prof = profile or case.get("profile", "laptop-4gb")
    merged = _read_yaml(CONFIG_DIR / "footage" / f"{footage}.yaml")
    merged = _deep_merge(merged, _read_yaml(CONFIG_DIR / "profiles" / f"{prof}.yaml"))
    merged = _deep_merge(merged, case)
    merged["profile"] = prof
    return CaseConfig.model_validate(merged)
