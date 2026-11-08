// Scene Package types and loading (mirrors docs/05-data-model.md, v0 subset).
import * as THREE from "three";

export interface Manifest {
  case_id: string;
  units: "metres" | "unitless";
  sources: { duration_s: number; fps: number; camera_id: string }[];
  quality: { registered_frac: number; reproj_err_px: number; psnr_holdout: number | null; gaussians: number };
  scene: { splat: string; trajectory: string; points: string; thumbs: string };
}

export interface Frame {
  id: string;
  name: string;
  t: number;
  c2w: number[][]; // OpenCV convention: x right, y down, z forward
  K: number[][];
  width: number;
  height: number;
}

export interface Points {
  xyz: Float32Array;
  frames: number[][]; // indices into trajectory frames that observed each point
}

export interface ScenePackage {
  base: string;
  manifest: Manifest;
  frames: Frame[];
  points: Points;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function loadPackage(base: string): Promise<ScenePackage> {
  const root = base.replace(/\/$/, "");
  const manifest = await getJson<Manifest>(`${root}/manifest.json`);
  const [traj, pts] = await Promise.all([
    getJson<{ frames: Frame[] }>(`${root}/${manifest.scene.trajectory}`),
    getJson<{ xyz: number[]; frames: number[][] }>(`${root}/${manifest.scene.points}`),
  ]);
  return {
    base: root,
    manifest,
    frames: traj.frames,
    points: { xyz: new Float32Array(pts.xyz), frames: pts.frames },
  };
}

export function thumbUrl(pkg: ScenePackage, f: Frame): string {
  return `${pkg.base}/${pkg.manifest.scene.thumbs}${f.name}`;
}

/** Camera pose in three.js convention (looks down -z, y up). */
export function frameMatrix(f: Frame): THREE.Matrix4 {
  const m = f.c2w;
  // Flip camera y and z axes: OpenCV -> OpenGL.
  return new THREE.Matrix4().set(
    m[0][0], -m[0][1], -m[0][2], m[0][3],
    m[1][0], -m[1][1], -m[1][2], m[1][3],
    m[2][0], -m[2][1], -m[2][2], m[2][3],
    0, 0, 0, 1,
  );
}

export function frameCentre(f: Frame): THREE.Vector3 {
  return new THREE.Vector3(f.c2w[0][3], f.c2w[1][3], f.c2w[2][3]);
}

/** Pixel position of world point X in frame f, or null if behind the camera / outside the image. */
export function project(f: Frame, X: THREE.Vector3): { u: number; v: number; depth: number } | null {
  const c = f.c2w;
  const d = [X.x - c[0][3], X.y - c[1][3], X.z - c[2][3]];
  // world -> camera = R^T (X - C)
  const xc = c[0][0] * d[0] + c[1][0] * d[1] + c[2][0] * d[2];
  const yc = c[0][1] * d[0] + c[1][1] * d[1] + c[2][1] * d[2];
  const zc = c[0][2] * d[0] + c[1][2] * d[1] + c[2][2] * d[2];
  if (zc <= 1e-3) return null;
  const u = f.K[0][0] * (xc / zc) + f.K[0][2];
  const v = f.K[1][1] * (yc / zc) + f.K[1][2];
  if (u < 0 || v < 0 || u >= f.width || v >= f.height) return null;
  return { u, v, depth: zc };
}

/**
 * Frames that saw world point X, best first. Votes come from SfM tracks of nearby sparse
 * points (these encode real visibility, including occlusion); projection is the tie-breaker
 * and the fallback when no sparse point is close.
 */
export function framesSeeing(pkg: ScenePackage, X: THREE.Vector3, radius: number, k = 12): { index: number; u: number; v: number }[] {
  const { xyz, frames: refs } = pkg.points;
  const votes = new Map<number, number>();
  const r2 = radius * radius;
  for (let i = 0; i < refs.length; i++) {
    const dx = xyz[3 * i] - X.x, dy = xyz[3 * i + 1] - X.y, dz = xyz[3 * i + 2] - X.z;
    const dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 > r2) continue;
    const w = 1 - dist2 / r2;
    for (const fi of refs[i]) votes.set(fi, (votes.get(fi) ?? 0) + w);
  }
  const out: { index: number; u: number; v: number; score: number }[] = [];
  pkg.frames.forEach((f, index) => {
    const p = project(f, X);
    if (!p) return;
    const vote = votes.get(index) ?? 0;
    if (votes.size > 0 && vote === 0) return; // other frames saw it; this one is likely occluded
    // Prefer views that are close and look at the point head-on (near the image centre).
    const centred = 1 - Math.hypot(p.u / f.width - 0.5, p.v / f.height - 0.5);
    out.push({ index, u: p.u, v: p.v, score: vote * 10 + centred / p.depth });
  });
  return out.sort((a, b) => b.score - a.score).slice(0, k);
}

/** Camera centre interpolated along the trajectory at time t (the camera wearer's position). */
export function poseAt(frames: Frame[], t: number): { pos: THREE.Vector3; quat: THREE.Quaternion } {
  let hi = frames.findIndex((f) => f.t >= t);
  if (hi <= 0) hi = hi === 0 ? 0 : frames.length - 1;
  const lo = Math.max(0, hi - 1);
  const a = frames[lo], b = frames[hi];
  const s = b.t > a.t ? THREE.MathUtils.clamp((t - a.t) / (b.t - a.t), 0, 1) : 0;
  const qa = new THREE.Quaternion().setFromRotationMatrix(frameMatrix(a));
  const qb = new THREE.Quaternion().setFromRotationMatrix(frameMatrix(b));
  return { pos: frameCentre(a).lerp(frameCentre(b), s), quat: qa.slerp(qb, s) };
}
