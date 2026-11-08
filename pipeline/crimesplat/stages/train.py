"""S06 Gaussian splat training with gsplat's MCMC strategy.

The Gaussian count is hard-capped (`cap_max`) so memory is predictable on small GPUs. The loss
ignores masked pixels (people, vehicles, undistortion borders), which is what removes ghosts.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy
from plyfile import PlyData, PlyElement

from ..config import CaseConfig
from ..dataset import View, load_views, sparse_points

SH_C0 = 0.28209479177387814


def _knn_mean_dist(x: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    out = torch.empty(len(x), device=x.device)
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk], x)
        out[i:i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out


def _gauss_window(size: int = 11, sigma: float = 1.5, device="cuda") -> torch.Tensor:
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    return g.expand(3, 1, size, size).contiguous()


def _ssim_map(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """a, b: [1,3,H,W] in [0,1]. Returns per-pixel SSIM averaged over channels, [H,W]."""
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    var_a = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a**2
    var_b = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b**2
    cov = F.conv2d(a * b, win, padding=pad, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.mean(1)[0]


def _init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, extent: float) -> torch.nn.ParameterDict:
    means = torch.from_numpy(xyz).cuda()
    dist = _knn_mean_dist(means).clamp(min=1e-4 * extent)
    n = len(means)
    k = (sh_degree + 1) ** 2
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.rand(n, 4, device="cuda"), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device="cuda"))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).cuda() - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, k - 1, 3, device="cuda")),
    })


def _render(params, view_mat: torch.Tensor, K: torch.Tensor, w: int, h: int, sh_degree: int, packed: bool):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, view_mat[None], K[None], w, h,
        sh_degree=sh_degree, packed=packed,
    )
    return img[0].clamp(0, 1), info


def save_ply(params, path: Path) -> None:
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p["means"])
    f_dc = p["sh0"].reshape(n, 3)
    f_rest = p["shN"].transpose(0, 2, 1).reshape(n, -1)  # channel-major, as in the reference 3DGS format
    cols = (
        [("x", p["means"][:, 0]), ("y", p["means"][:, 1]), ("z", p["means"][:, 2])]
        + [(f"nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
        + [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)]
        + [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
        + [("opacity", p["opacities"])]
        + [(f"scale_{i}", p["scales"][:, i]) for i in range(3)]
        + [(f"rot_{i}", p["quats"][:, i]) for i in range(4)]
    )
    arr = np.empty(n, dtype=[(name, "f4") for name, _ in cols])
    for name, v in cols:
        arr[name] = v
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def _evaluate(params, views: list[View], sh_degree: int, packed: bool, out_dir: Path) -> dict:
    if not views:
        return {}
    psnrs = []
    render_dir = out_dir / "eval_renders"
    render_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for v in views:
            h, w = v.image.shape[:2]
            img, _ = _render(params, torch.from_numpy(v.w2c).float().cuda(), torch.from_numpy(v.K).float().cuda(),
                             w, h, sh_degree, packed)
            gt = torch.from_numpy(v.image).cuda().float() / 255
            valid = ~torch.from_numpy(v.mask).cuda()
            mse = ((img - gt) ** 2)[valid].mean().item()
            psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
            side = np.concatenate([v.image, (img.cpu().numpy() * 255).astype(np.uint8)], 1)
            cv2.imwrite(str(render_dir / v.name), side[..., ::-1])
    return {"holdout_views": len(views), "psnr_masked": round(float(np.mean(psnrs)), 2),
            "psnr_min": round(float(np.min(psnrs)), 2)}


def run(cfg: CaseConfig, sparse: Path, img_dir: Path, mask_dir: Path | None, out_dir: Path, log=print) -> dict:
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("splat training needs a CUDA GPU")

    views = load_views(sparse, img_dir, mask_dir, tc.max_long_side)
    hold = [v for i, v in enumerate(views) if tc.holdout_every and i % tc.holdout_every == tc.holdout_every - 1]
    train = [v for v in views if v not in hold]
    centers = np.array([np.linalg.inv(v.w2c)[:3, 3] for v in views])
    extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1 or 1.0

    xyz, rgb = sparse_points(sparse)
    params = _init_params(xyz, rgb, tc.sh_degree, extent)
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / tc.iterations))
    strategy = MCMCStrategy(cap_max=tc.cap_max, refine_stop_iter=min(25_000, int(tc.iterations * 0.85)))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    win = _gauss_window()

    # Images stay on the CPU (pinned) and move per step, which keeps 4 GB cards viable.
    imgs = [torch.from_numpy(v.image).pin_memory() for v in train]
    valids = [torch.from_numpy(~v.mask).pin_memory() for v in train]
    Ks = [torch.from_numpy(v.K).float().cuda() for v in train]
    w2cs = [torch.from_numpy(v.w2c).float().cuda() for v in train]

    log(f"training {len(train)} views ({len(hold)} held out), {len(xyz)} init points, "
        f"cap {tc.cap_max}, {tc.iterations} iters, extent {extent:.2f}")
    t0 = time.time()
    order = np.random.permutation(len(train))
    for step in range(tc.iterations):
        i = order[step % len(train)]
        if step % len(train) == len(train) - 1:
            order = np.random.permutation(len(train))
        gt = imgs[i].cuda(non_blocking=True).float() / 255
        valid = valids[i].cuda(non_blocking=True)
        h, w = gt.shape[:2]
        sh_deg = min(step // tc.sh_increase_every, tc.sh_degree)

        img, info = _render(params, w2cs[i], Ks[i], w, h, sh_deg, tc.packed)
        vf = valid.float()
        nvalid = vf.sum().clamp(min=1)
        l1 = ((img - gt).abs() * vf[..., None]).sum() / (nvalid * 3)
        ssim = (_ssim_map(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win) * vf).sum() / nvalid
        loss = ((1 - tc.ssim_lambda) * l1 + tc.ssim_lambda * (1 - ssim)
                + tc.opacity_reg * torch.sigmoid(params["opacities"]).mean()
                + tc.scale_reg * torch.exp(params["scales"]).mean())
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])

        if step % tc.log_every == 0 or step == tc.iterations - 1:
            mem = torch.cuda.max_memory_allocated() / 2**30
            log(f"step {step:6d}  loss {loss.item():.4f}  gaussians {len(params['means']):,}  "
                f"peak {mem:.2f} GiB  {time.time() - t0:.0f}s")

    train_s = time.time() - t0
    save_ply(params, out_dir / "scene.ply")
    metrics = {
        "status": "ok",
        "train_views": len(train),
        "gaussians": len(params["means"]),
        "iterations": tc.iterations,
        "train_seconds": round(train_s, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "resolution": list(train[0].image.shape[1::-1]),
        **_evaluate(params, hold, tc.sh_degree, tc.packed, out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    del params, optimizers
    torch.cuda.empty_cache()
    return metrics
