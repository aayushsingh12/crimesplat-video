# ADR-0004 — Hybrid compute: laptop + on-demand cloud GPU

- **Status:** accepted
- **Date:** 2026-11-07

## Context
The dev machine is an RTX 3050 Laptop with **4 GB VRAM** and 15 GB RAM. Full-quality 3DGS, VGGT / MASt3R, HMR and 7B+ VLMs don't fit.

## Decision
- All stages run locally under the `laptop-4gb` profile (smaller models, capped Gaussians, lower resolution).
- Stages flagged `remote_capable` can run on a rented 24 GB GPU under the `cloud-24gb` profile, using the **same Docker image**, with inputs and outputs synced by rclone / rsync.
- VLM calls default to an API provider, with a local fallback for sensitive data.

## Update 2026-11-08: 2-day GPU window
Instead of rented cloud GPUs, the project has access to an **RTX 40-series machine for about 2 days**. The plan becomes:
- Everything is built and smoke-tested on the laptop beforehand (done for the static pipeline).
- The window is used only for full-quality training and ablations, following [the GPU-day runbook](../gpu-day-runbook.md).
- The environment is pinned to prebuilt wheels (Python 3.10, torch 2.4.1 + CUDA 12.4, gsplat 1.5.3, pycolmap 4.2.1), so setup needs no CUDA toolkit or compilation. `scripts/setup.sh` and `crimesplat doctor` verify the machine in minutes.
- Profiles: `laptop-4gb`, `gpu-8gb` (in case the card is a 4060/4070), `gpu-24gb` (4090).
- Measured on the laptop: the laptop profile peaks at 0.5 GiB VRAM, so 4 GB is less of a constraint than feared for scenes of this size.

## Alternatives considered
- **Cloud only:** slower iteration loop, constant cost, data leaves the machine.
- **Laptop only:** caps quality, and some fallback models are impossible.
- **Buy a GPU:** out of scope for budget. Revisit if usage is heavy.

## Consequences
- ✅ Development stays local and cheap. Quality runs are available on demand.
- ❌ Profile-dependent results. Reports must always record the profile and the GPU (done via provenance).
- ❌ Sensitive data must not go to the cloud without approval (see the ethics doc).
