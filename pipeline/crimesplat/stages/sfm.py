"""S04 Camera pose estimation with COLMAP (SIFT) and GLOMAP global mapping, via pycolmap.

Dynamic regions are excluded from feature extraction using the S03 masks. If the chosen mapper
registers too few keyframes, the other mapper is tried and the better model is kept.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from ..config import CaseConfig


def _map(kind: str, db: Path, img_dir: Path, out: Path) -> pycolmap.Reconstruction | None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if kind == "global":
        models = pycolmap.global_mapping(db, img_dir, out)
    else:
        models = pycolmap.incremental_mapping(db, img_dir, out)
    if not models:
        return None
    return max(models.values(), key=lambda r: r.num_reg_images())


def _drop_outlier_cameras(rec: pycolmap.Reconstruction, factor: float = 6.0) -> list[str]:
    """Deregister cameras far outside the trajectory (global SfM occasionally misplaces one)."""
    ids = list(rec.reg_image_ids())
    centers = np.array([rec.images[i].projection_center() for i in ids])
    d = np.linalg.norm(centers - np.median(centers, 0), axis=1)
    limit = factor * max(np.median(d), 1e-9)
    removed = []
    for i, dist in zip(ids, d):
        if dist > limit:
            removed.append(rec.images[i].name)
            rec.deregister_frame(rec.images[i].frame_id)
    return sorted(removed)


def _camera_info(cam: pycolmap.Camera) -> dict:
    return {"model": cam.model_name, "width": cam.width, "height": cam.height,
            "params_info": cam.params_info, "params": [round(float(p), 5) for p in cam.params]}


def run(cfg: CaseConfig, img_dir: Path, colmap_mask_dir: Path, excluded: list[str], out_dir: Path) -> dict:
    sc = cfg.sfm
    pycolmap.logging.minloglevel = 1  # warnings and above; the full COLMAP log is very chatty
    db = out_dir / "database.db"
    db.unlink(missing_ok=True)
    names = sorted(p.name for p in img_dir.glob("*.jpg") if p.name not in set(excluded))

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = sc.camera_model
    if colmap_mask_dir.exists():
        reader.mask_path = str(colmap_mask_dir)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = sc.max_image_size
    extraction.sift.max_num_features = sc.max_num_features
    pycolmap.extract_features(db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)

    if len(names) <= sc.exhaustive_below:
        pycolmap.match_exhaustive(db)
        matching = "exhaustive"
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = sc.sequential_overlap
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
        matching = "sequential"

    order = [sc.mapper, "incremental" if sc.mapper == "global" else "global"]
    attempts = {}
    best, best_kind = None, None
    for kind in order:
        rec = _map(kind, db, img_dir, out_dir / f"sparse_{kind}")
        n = rec.num_reg_images() if rec else 0
        attempts[kind] = n
        if rec and (best is None or n > best.num_reg_images()):
            best, best_kind = rec, kind
        if n >= sc.min_registered_frac * len(names):
            break
    if best is None:
        raise RuntimeError(f"SfM failed: no model reconstructed from {len(names)} keyframes")

    outliers = _drop_outlier_cameras(best)

    final = out_dir / "sparse"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir()
    best.write(final)

    registered = sorted(best.images[i].name for i in best.reg_image_ids())
    reg_frac = len(registered) / max(1, len(names))
    report = {
        "status": "ok" if reg_frac >= sc.min_registered_frac else "degraded",
        "mapper": best_kind,
        "matching": matching,
        "attempts_registered": attempts,
        "keyframes_in": len(names),
        "registered": len(registered),
        "registered_frac": round(reg_frac, 3),
        "points3D": best.num_points3D(),
        "mean_reprojection_error_px": round(best.compute_mean_reprojection_error(), 3),
        "mean_track_length": round(best.compute_mean_track_length(), 2),
        "camera": _camera_info(best.cameras[next(iter(best.cameras))]),
        "unregistered": sorted(set(names) - set(registered)),
        "outlier_cameras_removed": outliers,
    }
    if outliers:
        report["warnings"] = [f"removed {len(outliers)} camera(s) registered far from the trajectory"]
    (out_dir / "registration_report.json").write_text(json.dumps(report, indent=2))
    return report
