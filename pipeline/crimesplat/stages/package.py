"""S10 Packaging: assemble the Scene Package the web viewer loads (see docs/05-data-model.md).

v0 contents: splat, keyframe trajectory with undistorted thumbnails, and sparse points carrying
the list of keyframes that observed them (used for click -> source frame lookup).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .. import __version__
from ..config import CaseConfig
from ..dataset import load_views
from ..workspace import sha256_file

THUMB_LONG_SIDE = 960
MAX_POINTS = 60_000
MAX_REFS_PER_POINT = 8


def run(cfg: CaseConfig, ws_root: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("scene", "cameras", "media/thumbs", "points", "provenance"):
        (out_dir / sub).mkdir(parents=True)

    sparse = ws_root / "normalize" / "sparse"
    img_dir = ws_root / "frames" / "images"
    frames = {f["name"]: f for f in json.loads((ws_root / "frames" / "frames.json").read_text())["keyframes"]}
    source = json.loads((ws_root / "ingest" / "source.json").read_text())
    transform = json.loads((ws_root / "normalize" / "transform.json").read_text())
    metrics = json.loads((ws_root / "train" / "metrics.json").read_text())
    sfm = json.loads((ws_root / "sfm" / "registration_report.json").read_text())

    shutil.copy2(ws_root / "train" / "scene.ply", out_dir / "scene" / "scene.ply")

    views = load_views(sparse, img_dir, None, THUMB_LONG_SIDE)
    traj = []
    for v in views:
        cv2.imwrite(str(out_dir / "media" / "thumbs" / v.name), v.image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        h, w = v.image.shape[:2]
        traj.append({
            "id": f"s0:{Path(v.name).stem}",
            "name": v.name,
            "t": round(frames[v.name]["t"], 3),
            "c2w": np.linalg.inv(v.w2c).round(6).tolist(),
            "K": v.K.round(3).tolist(),
            "width": w,
            "height": h,
            "keyframe": True,
            "pose_source": "sfm",
        })
    traj.sort(key=lambda f: f["t"])
    (out_dir / "cameras" / "trajectory.json").write_text(json.dumps({"frames": traj}))

    # Sparse points with their observing keyframes, longest tracks first.
    rec = pycolmap.Reconstruction(sparse)
    idx_of = {f["name"]: i for i, f in enumerate(traj)}
    pts = sorted(rec.points3D.values(), key=lambda p: -p.track.length())[:MAX_POINTS]
    xyz, rgb, refs = [], [], []
    for p in pts:
        obs = sorted({idx_of[rec.images[e.image_id].name] for e in p.track.elements
                      if rec.images[e.image_id].name in idx_of})
        if not obs:
            continue
        if len(obs) > MAX_REFS_PER_POINT:
            obs = [obs[i] for i in np.linspace(0, len(obs) - 1, MAX_REFS_PER_POINT).round().astype(int)]
        xyz.extend(np.round(p.xyz, 4).tolist())
        rgb.extend(p.color.tolist())
        refs.append(obs)
    (out_dir / "points" / "points.json").write_text(json.dumps({"xyz": xyz, "rgb": rgb, "frames": refs}))

    shutil.copy2(ws_root / "provenance.jsonl", out_dir / "provenance" / "provenance.jsonl")
    files = {str(p.relative_to(out_dir)): f"sha256:{sha256_file(p)}"
             for p in sorted(out_dir.rglob("*")) if p.is_file() and "thumbs" not in p.parts}
    manifest = {
        "schema_version": "0.1.0",
        "case_id": cfg.case_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": {"version": __version__, "profile": cfg.profile, "footage": cfg.footage},
        "units": transform["units"],
        "up": [0, 1, 0],
        "sources": [{"id": "s0", "camera_id": source["camera_id"], "origin_uri": source["uri"],
                     "sha256": source["sha256"], "duration_s": source["probe"]["duration_s"],
                     "fps": source["probe"]["fps"]}],
        "quality": {"registered_frac": sfm["registered_frac"],
                    "reproj_err_px": sfm["mean_reprojection_error_px"],
                    "psnr_holdout": metrics.get("psnr_masked"), "gaussians": metrics["gaussians"]},
        "scene": {"splat": "scene/scene.ply", "trajectory": "cameras/trajectory.json",
                  "points": "points/points.json", "thumbs": "media/thumbs/"},
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 2**20
    return {"status": "ok", "frames": len(traj), "points": len(refs), "size_mb": round(size_mb, 1)}
