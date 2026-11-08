"""S03 Dynamic masking: mark people and vehicles so they don't become static geometry.

Writes two mask sets per keyframe:
  masks/<name>.png        255 = dynamic (ignored in training loss)
  colmap_masks/<name>.png COLMAP convention, 0 = ignore features (file name is <image name>.png)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import CaseConfig


def run(cfg: CaseConfig, img_dir: Path, out_dir: Path) -> dict:
    mc = cfg.masks
    mask_dir = out_dir / "masks"
    colmap_dir = out_dir / "colmap_masks"
    mask_dir.mkdir(exist_ok=True)
    colmap_dir.mkdir(exist_ok=True)

    names = sorted(p.name for p in img_dir.glob("*.jpg"))
    model = None
    if mc.enabled:
        import torch
        from ultralytics import YOLO

        model = YOLO(mc.model)
        device = 0 if torch.cuda.is_available() else "cpu"

    stats = {}
    for name in names:
        img = cv2.imread(str(img_dir / name))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if model is not None:
            res = model.predict(img, classes=mc.classes, conf=mc.conf, device=device,
                                retina_masks=True, verbose=False)[0]
            if res.masks is not None:
                m = res.masks.data.cpu().numpy().any(axis=0)
                mask = (cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) * 255)
            k = max(3, int(w * mc.dilate_frac) | 1)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        stem = Path(name).stem
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
        cv2.imwrite(str(colmap_dir / f"{name}.png"), 255 - mask)
        stats[name] = float((mask > 0).mean())

    if model is not None:
        del model
        import torch

        torch.cuda.empty_cache()

    too_masked = sorted(n for n, f in stats.items() if f > mc.max_masked_frac)
    (out_dir / "mask_stats.json").write_text(json.dumps({"masked_frac": stats, "excluded": too_masked}, indent=2))
    fracs = np.array(list(stats.values()) or [0.0])
    return {"status": "ok", "frames": len(names), "mean_masked_frac": float(fracs.mean()),
            "frames_with_dynamic": int((fracs > 0).sum()), "excluded_from_sfm": too_masked}
