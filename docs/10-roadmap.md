# 10 — Roadmap & Milestones

Durations assume one primary developer working part-time on an academic schedule. Adjust as needed.

```mermaid
gantt
    title CrimeSplat Video — roadmap
    dateFormat  YYYY-MM-DD
    axisFormat  %b
    section Phase 0 · Spike
    Env + Docker + gsplat on laptop          :p0a, 2026-11-09, 7d
    Static splat from public walk-through    :p0b, after p0a, 7d
    Go / no-go review                        :milestone, m0, after p0b, 0d
    section Phase 1 · Static MVP
    Ingest, frames, masking (S01–S03)        :p1a, after m0, 14d
    Poses + normalisation (S04–S05)          :p1b, after p1a, 14d
    Splat training + packaging (S06, S10)    :p1c, after p1b, 10d
    Viewer v0 (orbit, snap-to-frame)         :p1d, after p1b, 14d
    MVP demo                                 :milestone, m1, after p1d, 0d
    section Phase 2 · Time & Evidence
    Actor tracking + lifting (S07)           :p2a, after m1, 14d
    Timeline + actor layer                   :p2b, after p2a, 10d
    Evidence proposal (S08) + pins           :p2c, after m1, 21d
    Event proposal (S09)                     :p2d, after p2c, 14d
    Review tool                              :p2e, after p2c, 14d
    Staged scenario capture #1               :p2f, after m1, 7d
    Feature-complete v1                      :milestone, m2, after p2e, 0d
    section Phase 3 · Multi-cam & Evaluation
    Multi-CCTV + survey localisation         :p3a, after m2, 21d
    Remote GPU execution                     :p3b, after m2, 7d
    Staged scenario capture #2               :p3c, after m2, 7d
    Metrics + ablations                      :p3d, after p3c, 21d
    User study                               :p3e, after p3d, 14d
    section Phase 4 · Research stretch
    4D event-clip spike (4D-GS + priors)     :p4a, after p3d, 28d
    SMPL actors (GVHMR)                      :p4b, after p3d, 14d
    Write-up / paper                         :p4c, after p3e, 28d
```

## Phase 0 — Feasibility spike (≈ 2 weeks)
**Goal:** retire R-01 and R-02.
- [ ] Docker image with CUDA 12, gsplat, pycolmap / GLOMAP, hloc, SAM 2.
- [ ] Reconstruct a public, non-sensitive house-tour / walk-through video (1–2 min) end-to-end with scripts (no orchestrator yet).
- [ ] Measure VRAM and time on the laptop at caps of 200 k / 400 k. Measure the same on one cloud GPU run.
- [ ] View the result in Spark.
- **Exit:** the go / no-go criteria in [01-feasibility.md](01-feasibility.md) §5.

## Phase 1 — Static MVP (≈ 6 weeks)
- Orchestrator with caching, the CLI, stages S01–S06 and S10, schemas v1.
- Viewer v0: load package, orbit, camera path, snap-to-source-view, click-anywhere frame lookup.
- **Demo:** body-cam-style walk-through → explorable scene with clickable source frames.

## Phase 2 — Time & evidence (≈ 6–8 weeks)
- S07 actors + timeline. S08 / S09 proposals. Review tool.
- First staged scenario filmed and processed.
- **Demo:** the full one-shot user story on staged footage.

## Phase 3 — Multi-camera & evaluation (≈ 8 weeks)
- Multi-CCTV ingest, time sync, survey-video localisation.
- Remote execution for heavy stages.
- Metrics suite (`crimesplat eval`), ablations, user study.

## Phase 4 — Research stretch (≈ 6–8 weeks)
- 4D event-clip module ([research note](research/4dgs-feasibility.md)).
- SMPL actors.
- Paper / thesis write-up.

## Definition of done (v1)
- [ ] `crimesplat run` produces a valid Scene Package from a body-cam-style video with no manual steps except review.
- [ ] All FR-M requirements met; ≥ 50% of FR-S met.
- [ ] Metrics on ≥ 2 staged scenarios reported.
- [ ] Docs updated to match the implementation, with ADRs for any deviations.
