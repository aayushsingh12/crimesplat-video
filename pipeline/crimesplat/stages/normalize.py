"""S05 Scene normalisation: Y up, origin on the floor under the first camera, metres if possible.

Up is first estimated from the cameras (people hold cameras roughly upright), then refined from
the floor plane found by RANSAC over near-horizontal planes below the cameras.
Scale comes from the configured camera height above that floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap

from ..config import CaseConfig


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _basis(up: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """Rows are the new x, y (= up), z axes expressed in old coordinates (right-handed)."""
    y = _unit(up)
    x = _unit(x_hint - x_hint.dot(y) * y)
    z = np.cross(x, y)
    return np.stack([x, y, z])


def _find_floor(pts: np.ndarray, up: np.ndarray, cam_h: float, iters: int = 3000,
                max_tilt_deg: float = 12.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """RANSAC for near-horizontal planes below the cameras; returns (normal, inlier points) of the floor.

    Walls are rejected by the normal constraint. Among horizontal planes (floor, table tops,
    box tops) the lowest one with at least 30% of the best plane's support is the floor.
    """
    h = pts @ up
    below = pts[h < cam_h]
    if len(below) < 50:
        return None
    thresh = 0.01 * (cam_h - np.percentile(below @ up, 2))
    rng = np.random.default_rng(seed)
    cos_max = np.cos(np.radians(max_tilt_deg))
    planes = []  # (support, height, normal)
    for _ in range(iters):
        a, b, c = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if n.dot(up) < 0:
            n = -n
        if n.dot(up) < cos_max:
            continue
        inl = np.abs((below - a) @ n) < thresh
        planes.append((int(inl.sum()), float(a @ up), n, a))
    if not planes:
        return None
    best = max(p[0] for p in planes)
    _, _, n, a = min((p for p in planes if p[0] >= 0.3 * best), key=lambda p: p[1])
    inliers = below[np.abs((below - a) @ n) < thresh]
    if len(inliers) < 20:
        return None
    _, _, vt = np.linalg.svd(inliers - inliers.mean(0), full_matrices=False)
    normal = vt[2] if vt[2].dot(up) > 0 else -vt[2]
    return normal, inliers


def run(cfg: CaseConfig, sparse_in: Path, out_dir: Path) -> dict:
    rec = pycolmap.Reconstruction(sparse_in)
    imgs = [rec.images[i] for i in sorted(rec.reg_image_ids())]
    centers = np.array([im.projection_center() for im in imgs])
    rots = np.array([im.cam_from_world().rotation.matrix() for im in imgs])  # world -> cam
    cam_up = -rots[:, 1, :]  # camera -y axis in world coordinates (OpenCV y points down)
    cam_right = rots[:, 0, :]
    pts = np.array([p.xyz for p in rec.points3D.values()])

    R = _basis(cam_up.mean(0), cam_right.mean(0))
    warnings = []

    # Refine "up" with the floor plane normal, which is more reliable than camera tilt.
    found = _find_floor(pts, R[1], float(np.median(centers @ R[1])))
    plane_tilt_deg = None
    floor_points = 0
    if found is not None:
        n, fp = found
        plane_tilt_deg = float(np.degrees(np.arccos(np.clip(n.dot(R[1]), -1, 1))))
        R = _basis(n, R[0])
        floor = float(np.median(fp @ R[1]))
        floor_points = len(fp)
    else:
        floor = float(np.percentile(pts @ R[1], 2))
        warnings.append("no floor plane found; using the lowest points, scale is unreliable")
    ch = centers @ R[1]

    cam_above_floor = float(np.median(ch - floor))
    target = cfg.normalize.camera_height_m
    if target and cam_above_floor > 0:
        scale, units = target / cam_above_floor, "metres"
    else:
        scale, units = 1.0, "unitless"
        if target:
            warnings.append("cameras are not above the detected floor; scene left unitless")

    c0 = R @ centers[0]
    t = -scale * np.array([c0[0], floor, c0[2]])
    sim = pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), t.reshape(3, 1))
    rec.transform(sim)

    final = out_dir / "sparse"
    final.mkdir(exist_ok=True)
    rec.write(final)

    new_centers = np.array([rec.images[i].projection_center() for i in sorted(rec.reg_image_ids())])
    transform = {
        "scene_from_sfm": {"scale": scale, "rotation": R.tolist(), "translation": t.tolist()},
        "units": units,
        "camera_height_prior_m": target,
        "median_camera_height": float(np.median(new_centers[:, 1])),
        "floor_tilt_vs_camera_up_deg": plane_tilt_deg,
        "floor_points": floor_points,
        "scene_extent": float(np.linalg.norm(new_centers - new_centers.mean(0), axis=1).max()),
    }
    (out_dir / "transform.json").write_text(json.dumps(transform, indent=2))
    return {"status": "degraded" if warnings else "ok", "warnings": warnings, **transform}
