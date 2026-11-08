"""Score a case's camera poses against ground truth (e.g. from make_synthetic_video.py).

Aligns estimated camera centres to ground truth with a closed-form similarity transform
(Umeyama 1991), then reports absolute trajectory error (ATE), the residual scale (1.0 means the
metric scale from S05 was exact) and the up-vector error.

    uv run python tools/eval_poses.py ../workspace/synth ../data/synthetic_room.gt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising sum ||dst - (s R src + t)||^2 (no reflections)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs**2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def main(ws: Path, gt_path: Path, stage: str = "normalize") -> dict:
    gt = json.loads(gt_path.read_text())
    centre_of = {f["frame_index"]: np.array(f["centre"]) for f in gt["frames"]}
    rec = pycolmap.Reconstruction(ws / stage / "sparse")
    est, ref = [], []
    for i in rec.reg_image_ids():
        im = rec.images[i]
        est.append(im.projection_center())
        ref.append(centre_of[int(Path(im.name).stem[1:])])  # names are f<frame index>.jpg
    est, ref = np.array(est), np.array(ref)
    s, R, t = umeyama(est, ref)
    err = np.linalg.norm((s * (R @ est.T)).T + t - ref, axis=1)
    up_err = np.degrees(np.arccos(np.clip((R @ [0, 1, 0])[1], -1, 1)))
    return {
        "cameras": len(est),
        "ate_mean_m": round(float(err.mean()), 4),
        "ate_max_m": round(float(err.max()), 4),
        "residual_scale": round(float(s), 4),
        "up_error_deg": round(float(up_err), 2),
        "mean_height_est_m": round(float(est[:, 1].mean()), 3),
        "mean_height_gt_m": round(float(ref[:, 1].mean()), 3),
    }


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:]), indent=2))
