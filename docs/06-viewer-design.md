# 06 — Viewer & Review Tool Design

## 1. Goals

- Reproduce the reference experience: open a URL, see the scene, orbit it, click points of interest, see the source imagery.
- Add **time**: a timeline that replays actors and the camera wearer through the static scene.
- Run on a laptop's integrated GPU at ≥ 30 FPS.

## 2. Tech choices

| Concern | Choice | Alternatives |
|---|---|---|
| Framework | Vite + TypeScript + React (UI panels only) | Svelte, plain TS |
| 3D engine | three.js | PlayCanvas engine |
| Splat renderer | **Spark** (`@sparkjsdev/spark`, three.js-native, supports SPZ / PLY / SOG, mixes splats with regular meshes) | `@mkkellogg/gaussian-splats-3d`, PlayCanvas GSplat |
| State | Zustand | Redux |
| UI kit | Tailwind + Radix primitives | MUI |
| Timeline | Custom canvas component | — |

Spark is preferred because actors, pins and camera frustums are ordinary three.js meshes. Correct depth sorting between splats and meshes matters for pins being hidden behind walls.

## 3. Screen layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Case 01 · Bodycam · 4:01   [Evidence ▾] [Layers ▾] [Measure] [Review ●]│
├───────────────────────────────────────────────┬─────────────────────────┤
│                                               │  FRAME INSPECTOR        │
│                                               │  ┌───────────────────┐  │
│            3D VIEWPORT                        │  │  source frame     │  │
│   ● pins (evidence)    ◇ actors + trails      │  │  bbox highlighted │  │
│   ▲ camera frustum (wearer at time t)         │  └───────────────────┘  │
│                                               │  01:41.3  ◀ ▶  [Snap]   │
│                                               │  Knife · weapon · 0.71  │
│                                               │  "Kitchen knife on…"    │
│                                               │  thumbnails ▢▢▢▢▢       │
├───────────────────────────────────────────────┴─────────────────────────┤
│ ▶ ❚❚  00:00 ━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 04:01   1×  ⟳ │
│ events:   ▮entry          ▮weapon          ▮▮▮struggle      ▮exit      │
│ actors:   ═══ wearer ════════════════════════════════════════════      │
│               ═══ Person 2 ═════════════════════════════               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Module design

```mermaid
flowchart TB
    PL[PackageLoader<br/>fetch manifest → validate → lazy assets] --> ST[(Zustand store<br/>time, selection, layers, mode)]
    PL --> SR[SplatLayer<br/>Spark SplatMesh, preview→full swap]
    PL --> CL[CameraPathLayer<br/>line + frustum at t]
    PL --> AL[ActorLayer<br/>capsule / SMPL, label, trail]
    PL --> EL[EvidenceLayer<br/>billboard pins, occlusion-aware]
    PL --> VI[VisibilityIndex<br/>frame poses + depth maps]
    ST <--> TL[Timeline UI]
    ST <--> FI[FrameInspector UI]
    ST --> AL & CL & EL
    PK[Picker<br/>raycast pins / actors → splat depth] --> ST
    PK --> VI
    NAV[CameraController<br/>orbit · fly · snapTo(pose)] --> ST
    RE[ReviewEditor<br/>edit mode only] --> ST
    RE -->|PATCH| API[(local review server)]
```

### 4.1 Time model
- A single `currentTime` (seconds, master clock) lives in the store. Playback advances it with `requestAnimationFrame` × playback rate.
- `ActorLayer.update(t)`: per track, binary-search the samples, then lerp position and slerp heading. Samples with `src: predicted` are drawn with reduced opacity. Tracks outside `[t_start, t_end]` are hidden (or ghosted, as a toggle).
- Trails: the last 10 s of positions as a fading polyline.
- The camera wearer is drawn as a frustum plus a small body capsule below it.

### 4.2 Picking
1. Raycast against pin sprites and actor meshes (cheap, three.js).
2. If nothing is hit, ask Spark for the splat depth at the cursor (or ray-march the sorted splats) → world point `X`.
3. `VisibilityIndex.query(X)`: for each keyframe, project `X`; reject if outside the image or `|depth_map(u,v) − z| > 0.1·z`; score = `sharpness · cos(angle) / distance`; return top-5.
4. The Frame Inspector shows the frames with a crosshair at the projected `X`.

Cost: ~250 keyframes × trivial math. Under 5 ms in JS. Depth maps are loaded lazily on first pick.

### 4.3 Snap-to-source-view
Animate the camera to the frame's `c2w` with matching FOV (from intrinsics, undistorted). Show the source frame as a screen-aligned overlay with an opacity slider. This is the strongest "is this reconstruction trustworthy?" check for an analyst.

### 4.4 Layers panel
Toggles: splat · camera path · actors · trails · evidence (confirmed / proposed) · events · grid / ground · measurement.

## 5. Review mode

Opened via `crimesplat review <case>`, which serves the app with a small local API (FastAPI or a Vite middleware) that writes `annotations/*.json`.

Reviewer actions:
- Accept / reject / edit label and description of proposed evidence and events (keyboard: `A`, `R`, `E`, `J/K` for next/prev).
- Drag a pin in 3D to fix its position. Snapping uses splat depth.
- Add a manual evidence point (click a surface → label).
- Merge or split actor tracks. Rename actors.
- Set the metric scale by clicking two points and entering the known distance (overrides S05).
- Every action appends to `review_log.jsonl` (who, when, before → after) for provenance.

## 6. Performance budget

| Item | Target |
|---|---|
| First paint (preview splat) | < 3 s on 50 Mbps |
| Full splat loaded | < 10 s |
| Frame time, integrated GPU, 400 k splats | ≤ 33 ms |
| Overlay update per frame | < 1 ms |

Fallbacks: a lower `maxSplats` on mobile or low-end devices. Disable trails if the frame time is > 33 ms.

## 7. Accessibility & UX notes
- Every 3D-only piece of information (pins, events) is also listed in a sidebar list for keyboard and screen-reader access.
- Proposed (unreviewed) items are visually distinct (dashed outline + "AI" badge) so analysts never confuse them with confirmed findings.
- Sensitive-content toggle: blur all media thumbnails by default, reveal on click (graphic footage).
