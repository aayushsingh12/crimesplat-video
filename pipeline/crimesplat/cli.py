"""`crimesplat` command line: init a case, run the stage pipeline with caching, inspect status."""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .config import CaseConfig, load_case
from .workspace import Workspace, canonical_hash

app = typer.Typer(add_completion=False, help="Video -> explorable 3D Gaussian Splat scene.")
console = Console()

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "cases"
WORKSPACE = REPO / "workspace"


def _ingest(cfg: CaseConfig, ws: Workspace, out: Path) -> dict:
    from .stages import ingest
    return ingest.run(cfg, out)


def _frames(cfg, ws, out):
    from .stages import frames
    src = json.loads((ws.root / "ingest" / "source.json").read_text())["file"]
    return frames.run(cfg, ws.root / "ingest" / src, out)


def _masks(cfg, ws, out):
    from .stages import masks
    return masks.run(cfg, ws.root / "frames" / "images", out)


def _sfm(cfg, ws, out):
    from .stages import sfm
    excluded = json.loads((ws.root / "masks" / "mask_stats.json").read_text())["excluded"]
    return sfm.run(cfg, ws.root / "frames" / "images", ws.root / "masks" / "colmap_masks", excluded, out)


def _normalize(cfg, ws, out):
    from .stages import normalize
    return normalize.run(cfg, ws.root / "sfm" / "sparse", out)


def _train(cfg, ws, out):
    from .stages import train
    mask_dir = ws.root / "masks" / "masks"
    return train.run(cfg, ws.root / "normalize" / "sparse", ws.root / "frames" / "images",
                     mask_dir if cfg.masks.enabled else None, out, log=console.print)


def _package(cfg, ws, out):
    from .stages import package
    return package.run(cfg, ws.root, out)


# (name, stage version, config sections that affect it, runner). Order is execution order;
# each stage depends on all earlier ones.
STAGES: list[tuple[str, str, list[str], Callable]] = [
    ("ingest", "1", ["sources"], _ingest),
    ("frames", "1", ["frames"], _frames),
    ("masks", "1", ["masks"], _masks),
    ("sfm", "1", ["sfm"], _sfm),
    ("normalize", "1", ["normalize"], _normalize),
    ("train", "1", ["train", "masks.enabled"], _train),
    ("package", "1", [], _package),
]
NAMES = [s[0] for s in STAGES]


def _section(cfg: CaseConfig, path: str):
    obj = cfg.model_dump()
    for part in path.split("."):
        obj = obj[part]
    return obj


def _keys(cfg: CaseConfig) -> dict[str, str]:
    keys, upstream = {}, ""
    for name, version, sections, _ in STAGES:
        upstream = canonical_hash([name, version, upstream, [_section(cfg, s) for s in sections]])
        keys[name] = upstream
    return keys


def _load(case_id: str, profile: Optional[str]) -> tuple[CaseConfig, Workspace]:
    path = CASES / f"{case_id}.yaml"
    cfg = load_case(path, profile)
    ws = Workspace(WORKSPACE / cfg.case_id)
    ws.root.mkdir(parents=True, exist_ok=True)
    (ws.root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    return cfg, ws


@app.command()
def init(
    case_id: str,
    source: str = typer.Option(..., help="Video file path or URL"),
    footage: str = typer.Option("handheld", help="handheld | bodycam | drone | news"),
    profile: str = typer.Option("laptop-4gb", help="laptop-4gb | gpu-8gb | gpu-24gb"),
):
    """Create cases/<case_id>.yaml."""
    CASES.mkdir(exist_ok=True)
    path = CASES / f"{case_id}.yaml"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")
    if not source.startswith(("http://", "https://")):
        source = str(Path(source).expanduser().resolve())
    path.write_text(yaml.safe_dump({"case_id": case_id, "footage": footage, "profile": profile,
                                    "sources": [{"uri": source, "camera_id": "cam0"}]}, sort_keys=False))
    console.print(f"wrote {path}")


@app.command()
def run(
    case_id: str,
    profile: Optional[str] = typer.Option(None, help="Override the case's compute profile"),
    until: Optional[str] = typer.Option(None, help=f"Stop after this stage: {', '.join(NAMES)}"),
    force: list[str] = typer.Option([], help="Re-run this stage (and everything after it)"),
):
    """Run all stages in order, skipping those whose inputs and config are unchanged."""
    cfg, ws = _load(case_id, profile)
    keys = _keys(cfg)
    forced_from = min((NAMES.index(f) for f in force), default=len(NAMES))
    for i, (name, _, _, fn) in enumerate(STAGES):
        if i < forced_from and ws.is_fresh(name, keys[name]):
            console.print(f"[dim]✓ {name} (cached)[/dim]")
        else:
            console.rule(f"[bold]{name}")
            ws.invalidate(name)
            out = ws.root / name if name != "package" else ws.root / "package"
            out.mkdir(parents=True, exist_ok=True)
            started = time.time()
            try:
                report = fn(cfg, ws, out)
            except Exception as e:
                (out / "error.txt").write_text(traceback.format_exc())
                ws.log_provenance(name, keys[name], round(time.time() - started, 1), "failed")
                console.print(f"[red]✗ {name} failed: {e}[/red]  (traceback in {out / 'error.txt'})")
                raise typer.Exit(1)
            ws.mark_done(name, keys[name], report, started)
            for w in report.get("warnings", []):
                console.print(f"[yellow]  warning: {w}[/yellow]")
            summary = {k: v for k, v in report.items() if not isinstance(v, (list, dict))}
            console.print(f"[green]✓ {name}[/green] {time.time() - started:.0f}s  {summary}")
        if name == until:
            break


@app.command()
def status(case_id: str, profile: Optional[str] = None):
    """Show which stages are cached, stale or not yet run."""
    cfg, ws = _load(case_id, profile)
    keys = _keys(cfg)
    table = Table("stage", "state", "duration", "status")
    for name in NAMES:
        key = ws.cache_key(name)
        state = "not run" if key is None else ("cached" if key == keys[name] else "stale")
        rep = ws.root / name / "report.json"
        done = ws.root / name / "_SUCCESS.json"
        dur = f"{json.loads(done.read_text())['duration_s']}s" if done.exists() else ""
        st = json.loads(rep.read_text()).get("status", "") if rep.exists() and key else ""
        table.add_row(name, state, dur, st)
    console.print(table)


@app.command()
def doctor():
    """Check the machine: GPU, CUDA kernels, COLMAP, ffmpeg. Suggests a compute profile."""
    ok = True

    def check(name: str, fn: Callable[[], str]):
        nonlocal ok
        try:
            console.print(f"[green]✓[/green] {name}: {fn()}")
        except Exception as e:
            ok = False
            console.print(f"[red]✗[/red] {name}: {e}")

    def gpu() -> str:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch sees no CUDA GPU (check the NVIDIA driver)")
        p = torch.cuda.get_device_properties(0)
        return f"{p.name}, {p.total_memory / 2**30:.1f} GiB, compute {p.major}.{p.minor}, torch {torch.__version__}"

    def kernels() -> str:
        import torch
        from gsplat import rasterization

        n = 2000
        m = torch.randn(n, 3, device="cuda") + torch.tensor([0, 0, 5.0], device="cuda")
        q = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
        K = torch.tensor([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]], device="cuda")[None]
        img, _, _ = rasterization(m, q, torch.full((n, 3), 0.05, device="cuda"), torch.ones(n, device="cuda"),
                                  torch.rand(n, 3, device="cuda"), torch.eye(4, device="cuda")[None], K, 320, 240)
        if img.sum() <= 0:
            raise RuntimeError("rasterizer returned an empty image")
        return "gsplat CUDA rasterization works"

    def colmap() -> str:
        import pycolmap

        return f"pycolmap {pycolmap.__version__} ({'CUDA' if pycolmap.has_cuda else 'CPU'} build)"

    def ffmpeg() -> str:
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe not found; install ffmpeg (e.g. `sudo apt install ffmpeg`)")
        return shutil.which("ffprobe")

    check("GPU", gpu)
    check("CUDA kernels", kernels)
    check("COLMAP", colmap)
    check("ffmpeg", ffmpeg)
    try:
        import torch

        gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        prof = "gpu-24gb" if gib >= 20 else "gpu-8gb" if gib >= 7 else "laptop-4gb"
        console.print(f"suggested profile: [bold]{prof}[/bold]  (crimesplat run <case> --profile {prof})")
    except Exception:
        pass
    if not ok:
        raise typer.Exit(1)


@app.command()
def publish_local(case_id: str):
    """Copy the case's Scene Package into web/public/scenes/ for the viewer."""
    src = WORKSPACE / case_id / "package"
    if not (src / "manifest.json").exists():
        raise typer.BadParameter("no package yet; run the pipeline first")
    dst = REPO / "web" / "public" / "scenes" / case_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    console.print(f"copied to {dst}\nopen http://localhost:5173/?scene=scenes/{case_id}")


if __name__ == "__main__":
    app()
