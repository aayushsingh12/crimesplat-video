# 09 — Evaluation Plan & Ethics

## 1. Research questions

- **RQ1:** Can a static 3DGS reconstruction with dynamic-object masking be produced from unconstrained body-cam / handheld footage with quality sufficient for spatial reasoning?
- **RQ2:** How accurately can actor positions be recovered in 3D from monocular moving-camera footage via ground-plane lifting, and how does that compare to world-grounded HMR (GVHMR / WHAM)?
- **RQ3:** How precise and complete is automatic evidence and event proposal (detector + VLM), and how much reviewer time does it save?
- **RQ4:** Does the 3D spatio-temporal interface improve analysts' accuracy and speed on spatial questions vs. watching the raw video? (User study.)
- **RQ5 (stretch):** Does event-window 4D-GS improve novel-view fidelity of dynamic content over the tracked-actor baseline? ([research note](research/4dgs-feasibility.md))

## 2. Datasets

### 2.1 Staged scenarios (primary, our own)
Real crime footage has no ground truth and raises serious ethical problems, so the main evaluation uses **staged scenes**:

- 2–3 indoor locations (apartment, office corridor, lab), with consenting actors following a scripted "incident" (entry, confrontation, prop weapon dropped, exit).
- Capture **simultaneously**:
  - a chest-mounted action camera (GoPro / phone in body-cam mode, wide FOV): the system input;
  - 2–4 fixed phones or CCTV-style cameras: the multi-camera input;
  - **ground truth:** a slow, careful walk-through (or a phone LiDAR / photogrammetry scan) of the empty scene for reference geometry; tape-measured floor markers (grid or ArUco tags) for metric positions; a clapperboard / flash for sync.
- Actor ground-truth positions: annotate foot positions on the fixed cameras (calibrated to the floor markers), or use ArUco tags on actors' shoulders.
- Evidence ground truth: prop positions measured with a tape from the markers.

### 2.2 Public datasets (component benchmarks)
| Component | Dataset |
|---|---|
| 3DGS quality sanity check | Mip-NeRF 360, Tanks and Temples |
| Indoor / handheld video reconstruction | ScanNet++ (iPhone video + laser scans) |
| Egocentric + people | EgoHumans, EgoBody (if access is granted) |
| Multi-camera people tracking | WILDTRACK |
| World-grounded human motion | EMDB, RICH |
| Monocular dynamic (for RQ5) | DyCheck iPhone, HyperNeRF; Neu3D for multi-view |

### 2.3 Real public footage (qualitative only)
Public body-cam releases and news footage may be used for **qualitative robustness tests** only, never for published figures without ethics approval and a copyright review.

## 3. Metrics

| Stage | Metric | Target (staged scenes) |
|---|---|---|
| S04 poses | % keyframes registered; reprojection error; ATE vs. reference scan (after Sim(3) alignment) | ≥ 85%; ≤ 1.5 px; ATE ≤ 10 cm |
| S05 scale | Relative scale error vs. tape measurements | ≤ 10% |
| S06 splat | PSNR / SSIM / LPIPS on held-out frames; masked PSNR (static regions) | PSNR ≥ 23 dB (laptop), ≥ 25 dB (cloud) |
| S06 cleanliness | "Ghost" score: rendered opacity in regions masked in all views | qualitative + ratio |
| S07 actors | Position RMSE (m) vs GT; HOTA / IDF1 for tracking; % time tracked | RMSE ≤ 0.5 m; IDF1 ≥ 0.7 |
| S08 evidence | Precision / recall at confidence 0.5; 3D localisation error | P ≥ 0.6, R ≥ 0.8 (before review); error ≤ 0.3 m |
| S09 events | Temporal IoU-based precision / recall (tIoU ≥ 0.3) | P ≥ 0.5, R ≥ 0.7 |
| Viewer | FPS on integrated GPU; load time | ≥ 30 FPS; ≤ 10 s |
| Usability (RQ4) | Task accuracy & time; SUS score; NASA-TLX | SUS ≥ 70 |

## 4. Ablations (for a paper / thesis)

1. Masking: none vs. YOLO-seg vs. SAM 2 propagation vs. + robust loss.
2. Pose: COLMAP-SIFT vs. hloc (SP+LG) + GLOMAP vs. VGGT vs. MASt3R-SfM.
3. Training: default densification vs. MCMC at caps of 200 k / 400 k / 1 M (quality vs. VRAM curve on 4 GB vs. 24 GB).
4. Actor lifting: ground-ray vs. depth-based vs. GVHMR.
5. Evidence: detector-only vs. detector + VLM verification vs. VLM-only.
6. Multi-CCTV: with vs. without a survey walk-through.

## 5. User study (RQ4)

- **Participants:** 10–16 people (ideally criminology / forensics students).
- **Conditions:** (A) raw video + scrubbing, (B) CrimeSplat viewer. Within-subjects, counterbalanced, two staged scenarios.
- **Tasks:** "Where was the knife relative to the door?", "Which room did Person 2 go to after the confrontation?", "Estimate the distance between X and Y."
- **Measures:** correctness, time, confidence, SUS, qualitative feedback.

## 6. Ethics, privacy & forensic integrity

### 6.1 Principles
1. **Consent first.** Development and published results use staged / consented / synthetic data.
2. **Minimise real victim data.** No real victim imagery in the repository, issue trackers, demos or papers.
3. **No identity inference.** No face recognition. Actors are anonymous IDs.
4. **Assistive, not evidentiary.** The tool assists understanding. It isn't a substitute for the original footage, and the UI says so.
5. **Transparency of AI.** All AI-proposed content is labelled and requires human confirmation.

### 6.2 Data handling
- Real case data stays on encrypted local storage or an institution-approved server. Never on public hosting or unvetted third-party APIs. When the case data is sensitive, set `vlm.provider: local_qwen` or `none`.
- Sensitive-content blur in the viewer by default.
- Retention: delete workspace intermediates after packaging, unless needed for research reproducibility.

### 6.3 Approvals
- Ethics approval (IRB / institutional equivalent) for the staged-actor capture and the user study.
- Actor consent forms covering recording, 3D reconstruction and publication.
- Check the licence / terms of any public footage before use.

### 6.4 Known limitations to state in any publication
- Monocular scale uncertainty. Single-ground-plane assumption. Tracked people are approximations, not observations from novel viewpoints. VLM outputs can be wrong. Performance on real forensic footage is not guaranteed by staged results.
