# 03 — Data Flow

Data-flow diagrams (DFDs) are given at three levels, followed by sequence diagrams for the key interactive flows.

## 1. DFD Level 0 — context

```mermaid
flowchart LR
    V[/Video sources/] -->|raw video, URL, metadata| P((0. CrimeSplat<br/>Video Pipeline))
    O[/Operator/] -->|config, review decisions| P
    P -->|Scene Package| A[/Analyst via Viewer/]
    P <-->|frames, prompts / JSON answers| X[/VLM API/]
```

## 2. DFD Level 1 — pipeline stages

```mermaid
flowchart TB
    IN[/Video file or URL/] --> P1((1 Ingest))
    P1 -->|video.mp4, audio.wav| D1[(D1 Source store)]
    P1 -->|provenance.json| D9[(D9 Provenance log)]

    D1 --> P2((2 Shot & Frame<br/>Selection))
    P2 -->|shots.json, frames.parquet| D2[(D2 Frame index)]
    P2 -->|keyframes/*.jpg| D3[(D3 Image store)]

    D3 --> P3((3 Dynamic<br/>Masking))
    D1 --> P3
    P3 -->|masks/*.png| D4[(D4 Mask store)]

    D3 & D4 --> P4((4 Pose<br/>Estimation))
    D1 -->|all frames, for localisation| P4
    P4 -->|cameras, images, points3D| D5[(D5 Sparse model)]
    P4 -->|poses_all_frames.json| D5

    D5 --> P5((5 Scene<br/>Normalisation))
    P5 -->|transform.json, ground_plane.json, scale| D5

    D3 & D4 & D5 --> P6((6 Splat<br/>Training))
    P6 -->|scene.ply, metrics.json| D6[(D6 Splat store)]

    D1 & D5 --> P7((7 Actor Tracking<br/>& Lifting))
    P7 -->|tracks.json| D7[(D7 Dynamic store)]

    D3 & D5 & D6 --> P8((8 Evidence<br/>Proposal))
    P8 -->|evidence.proposed.json| D8[(D8 Annotation store)]

    D1 & D7 --> P9((9 Event<br/>Proposal))
    P9 -->|events.proposed.json, transcript.json| D8

    D8 --> R((Human<br/>Review))
    R -->|annotations.json, status=confirmed| D8

    D6 & D7 & D8 & D5 & D2 & D9 --> P10((10 Packaging))
    P10 --> OUT[/Scene Package/]
```

### 2.1 Data stores

| ID | Store | Format | Approx. size (2-min clip) |
|---|---|---|---|
| D1 | Source store | MP4 (original + normalised), WAV | 50–500 MB |
| D2 | Frame index | `frames.parquet` (one row per *video* frame: timestamp, shot, sharpness, is_keyframe, pose_id) | < 5 MB |
| D3 | Image store | JPEG keyframes (full + downscaled) | 100–400 MB |
| D4 | Mask store | 1-bit PNG per keyframe, per class | 10–50 MB |
| D5 | Sparse model | COLMAP binary + JSON exports | 5–50 MB |
| D6 | Splat store | PLY (uncompressed), SPZ (compressed) | 50–250 MB → 10–40 MB |
| D7 | Dynamic store | JSON / Parquet | < 10 MB |
| D8 | Annotation store | JSON (proposed + reviewed) | < 5 MB |
| D9 | Provenance log | JSON-lines, append-only | < 1 MB |

## 3. DFD Level 2 — key sub-flows

### 3.1 Stage 4: pose estimation (incl. all-frame localisation)

```mermaid
flowchart LR
    K[Keyframes + masks] --> F[Feature extraction<br/>SuperPoint, masked]
    F --> PS[Pair selection<br/>sequential window + retrieval NetVLAD]
    PS --> M[Matching<br/>LightGlue]
    M --> G[Global SfM<br/>GLOMAP]
    G -->|fails / < 60% registered| FB[Fallback<br/>VGGT / MASt3R-SfM ☁️]
    G --> BA[Bundle adjustment<br/>refine intrinsics incl. fisheye]
    FB --> BA
    BA --> SM[(Sparse model)]
    SM --> L[Localise all remaining frames<br/>PnP + RANSAC vs model]
    L -->|gaps| I[Interpolate<br/>SLERP + linear, flagged]
    L & I --> PA[(poses_all_frames.json)]
```

Why localise *all* frames? The timeline needs the camera (= body-cam wearer) pose at **any** time *t*, not just at the ~250 keyframes used for training. Actor lifting also needs a pose for every frame in which an actor is detected.

### 3.2 Stage 7: actor tracking and 3D lifting

```mermaid
flowchart LR
    V[Video frames @ 10 fps] --> D[Person detection<br/>YOLO11]
    D --> T[Multi-object tracking<br/>BoT-SORT + ReID]
    T --> TR[2D tracklets]
    TR --> FP[Foot point<br/>bbox bottom-centre or<br/>ankle keypoints]
    P[(Per-frame poses)] --> RC
    GP[(Ground plane)] --> RC
    FP --> RC[Ray-cast foot point<br/>onto ground plane]
    RC --> SM[Temporal smoothing<br/>Kalman / RTS smoother]
    SM --> MERGE[Track stitching<br/>3D proximity + ReID]
    MERGE --> OUT[(tracks.json)]
    TR -.optional.-> H[GVHMR / WHAM<br/>SMPL in camera frame] -.-> AL[Align to scene frame] -.-> OUT
    CAM[(Camera trajectory)] -->|actor 'Camera wearer'| OUT
```

### 3.3 Stage 8: evidence proposal

```mermaid
flowchart LR
    KF[Keyframes] --> OD[Open-vocab detection<br/>prompt list: knife, gun, body,<br/>blood stain, shell casing, ...]
    OD --> SEG[SAM 2 box→mask]
    SEG --> ASSOC[Cross-view association<br/>epipolar + appearance]
    ASSOC --> TRI[Multi-view triangulation<br/>of mask centroids]
    TRI -->|single view only| DEP[Depth from rendered splat<br/>at mask centroid]
    TRI & DEP --> VER[VLM verification<br/>crop + context → JSON]
    VER --> RANK[Score = det_conf × vlm_conf<br/>× view_count factor]
    RANK --> OUT[(evidence.proposed.json)]
```

### 3.4 Stage 9: event proposal

```mermaid
flowchart LR
    V[Video] --> W[Sliding windows<br/>8 s, stride 4 s, 1–2 fps]
    W --> VLM[VLM: describe + classify<br/>event taxonomy → JSON]
    A[Audio] --> ASR[Whisper ASR<br/>word timestamps]
    A --> AE[Audio events<br/>gunshot / scream / glass]
    TR[(tracks.json)] --> H[Track heuristics<br/>enter / exit / proximity<br/>/ rapid motion]
    VLM & ASR & AE & H --> FUSE[Temporal fusion<br/>merge overlapping candidates]
    FUSE --> ANCH[3D anchoring<br/>actors involved → location]
    ANCH --> OUT[(events.proposed.json)]
```

## 4. Sequence diagrams (runtime)

### 4.1 Analyst clicks an evidence pin

```mermaid
sequenceDiagram
    actor U as Analyst
    participant V as Viewer
    participant O as Overlay layer
    participant F as Frame Inspector
    participant P as Scene Package (static files)

    U->>V: click on 3D pin "Knife"
    V->>O: raycast against pin sprites
    O-->>V: evidence_id = ev_007
    V->>F: open(ev_007)
    F->>P: GET frames/ev_007/*.jpg (lazy)
    P-->>F: thumbnails + bboxes
    F-->>U: gallery of supporting frames, bbox highlighted, timestamps
    U->>F: click frame @ 01:42.3
    F->>V: snapTo(camera_pose[frame_2553])
    V-->>U: 3D camera flies to the original viewpoint, frame overlaid at 50% opacity
```

### 4.2 Analyst clicks an arbitrary surface (FR-9)

```mermaid
sequenceDiagram
    actor U as Analyst
    participant V as Viewer
    participant R as Splat renderer
    participant Q as Visibility index
    participant F as Frame Inspector

    U->>V: click on the floor
    V->>R: render depth at pixel (x, y)
    R-->>V: depth d → world point X
    V->>Q: framesSeeing(X)
    Note over Q: for each camera: project X, check inside image,<br/>compare with precomputed depth map (occlusion),<br/>score = sharpness × cos(view angle) / distance
    Q-->>V: top-k frames
    V->>F: show(top-k, highlight projected X)
```

### 4.3 Timeline scrub

```mermaid
sequenceDiagram
    actor U as Analyst
    participant T as Timeline
    participant O as Overlay
    participant F as Frame Inspector

    U->>T: drag to 03:20
    T->>O: setTime(200.0 s)
    O->>O: interpolate each track at t (binary search + lerp)
    O->>O: move actor meshes, camera-wearer frustum, trails (t-10 s..t)
    T->>T: highlight active events (start ≤ t ≤ end)
    T->>F: (if "sync video" on) show source frame nearest t
```

### 4.4 Operator runs the pipeline

```mermaid
sequenceDiagram
    actor Op as Operator
    participant C as CLI
    participant W as Workspace cache
    participant L as Local GPU
    participant R as Remote GPU

    Op->>C: crimesplat run case01.yaml --profile laptop-4gb
    loop each stage in DAG order
        C->>W: key = hash(inputs, config, code_version)
        alt cache hit
            W-->>C: reuse outputs
        else miss
            alt stage.remote == true
                C->>R: sync inputs, run stage
                R-->>C: outputs
            else
                C->>L: run stage
                L-->>C: outputs
            end
            C->>W: store outputs + provenance entry
        end
    end
    C-->>Op: draft package ready → crimesplat review case01
```
