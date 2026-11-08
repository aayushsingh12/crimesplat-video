import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import {
  type Frame, type ScenePackage, frameCentre, frameMatrix, framesSeeing, loadPackage, poseAt, project, thumbUrl,
} from "./scene";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const canvas = $<HTMLCanvasElement>("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1013);
const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
scene.add(new SparkRenderer({ renderer }));
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const pathGroup = new THREE.Group();
const frustumGroup = new THREE.Group();
const markers = new THREE.Group();
scene.add(pathGroup, frustumGroup, markers);

let pkg: ScenePackage;
let splat: SplatMesh;
let selected: { frame: number; hits: { index: number; u: number; v: number }[] } | null = null;
let duration = 1;
let playing = false;
let wearer: THREE.Group;
let snapAnim: { from: THREE.Matrix4; to: THREE.Matrix4; fov0: number; fov1: number; t0: number } | null = null;

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement!;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement!);

function fmtTime(t: number): string {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

/** Wireframe camera pyramid for a frame, `size` metres deep. */
function frustum(f: Frame, size: number, color: number): THREE.LineSegments {
  const { K, width: w, height: h } = f;
  const corners = [[0, 0], [w, 0], [w, h], [0, h]].map(([u, v]) =>
    new THREE.Vector3(((u - K[0][2]) / K[0][0]) * size, -((v - K[1][2]) / K[1][1]) * size, -size));
  const o = new THREE.Vector3();
  const pts = corners.flatMap((c, i) => [o, c, c, corners[(i + 1) % 4]]);
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color }));
  lines.applyMatrix4(frameMatrix(f));
  return lines;
}

function buildOverlays(extent: number) {
  const pts = pkg.frames.map(frameCentre);
  pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x4cc2ff })));
  const size = extent * 0.03;
  const colour = new THREE.Color();
  pkg.frames.forEach((f, i) => {
    colour.setHSL(0.55 - 0.55 * (i / pkg.frames.length), 0.8, 0.55);
    const fr = frustum(f, size, colour.getHex());
    fr.userData.frameIndex = i;
    frustumGroup.add(fr);
  });
  // The camera wearer at the current timeline position.
  wearer = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.18, 1.0, 4, 12),
    new THREE.MeshBasicMaterial({ color: 0xf5a524, transparent: true, opacity: 0.55 }));
  wearer.add(body);
  // Pose is set every frame in setTime(); the frame only supplies the lens shape (K, size).
  const lens = frustum(pkg.frames[0], size * 2.5, 0xf5a524);
  wearer.add(lens);
  wearer.userData.body = body;
  scene.add(wearer);
}

function setTime(t: number) {
  const { pos, quat } = poseAt(pkg.frames, t);
  const body = wearer.userData.body as THREE.Mesh;
  const lens = wearer.children[1];
  lens.position.copy(pos);
  lens.quaternion.copy(quat);
  // The body hangs below the camera, standing on the floor (y = 0) when the scene is metric.
  const centreY = pkg.manifest.units === "metres" ? Math.max(0.68, pos.y - 0.7) : pos.y - 0.7;
  body.position.set(pos.x, centreY, pos.z);
  $<HTMLInputElement>("scrub").value = String(t);
  $("time-label").textContent = fmtTime(t);
}

function showFrame(index: number, uv?: { u: number; v: number }) {
  const f = pkg.frames[index];
  $("panel").hidden = false;
  $<HTMLImageElement>("frame-img").src = thumbUrl(pkg, f);
  $("frame-time").textContent = `${fmtTime(f.t)} · ${f.id}`;
  const ch = $("crosshair");
  if (uv) {
    ch.hidden = false;
    ch.style.left = `${(uv.u / f.width) * 100}%`;
    ch.style.top = `${(uv.v / f.height) * 100}%`;
  } else {
    ch.hidden = true;
  }
  selected = { frame: index, hits: selected?.hits ?? [] };
  document.querySelectorAll("#thumbs figure").forEach((el) =>
    el.classList.toggle("active", Number((el as HTMLElement).dataset.index) === index));
  setTime(f.t);
}

function showHits(title: string, hits: { index: number; u: number; v: number }[]) {
  $("panel-title").textContent = title;
  const box = $("thumbs");
  box.replaceChildren(...hits.map((h) => {
    const f = pkg.frames[h.index];
    const fig = document.createElement("figure");
    fig.dataset.index = String(h.index);
    fig.innerHTML = `<img loading="lazy" src="${thumbUrl(pkg, f)}" alt=""><figcaption>${fmtTime(f.t)}</figcaption>`;
    fig.onclick = () => showFrame(h.index, h);
    return fig;
  }));
  selected = { frame: hits[0]?.index ?? 0, hits };
  if (hits.length) showFrame(hits[0].index, hits[0]);
}

/** Fly to the frame's original viewpoint and overlay the frame so the reconstruction can be checked. */
function snapTo(index: number) {
  const f = pkg.frames[index];
  const { clientWidth: cw, clientHeight: ch } = canvas;
  const frameAspect = f.width / f.height;
  const vfov = 2 * Math.atan(f.height / 2 / f.K[1][1]);
  let fov = vfov;
  let ow = ch * frameAspect, oh = ch;
  if (cw / ch < frameAspect) {
    // Canvas is narrower than the frame: fit width, widen the vertical FOV to match.
    const hfov = 2 * Math.atan(f.width / 2 / f.K[0][0]);
    fov = 2 * Math.atan(Math.tan(hfov / 2) / (cw / ch));
    ow = cw;
    oh = cw / frameAspect;
  }
  const img = $<HTMLImageElement>("overlay");
  img.src = thumbUrl(pkg, f);
  img.style.width = `${ow}px`;
  img.style.height = `${oh}px`;
  img.style.display = "block";
  img.style.opacity = $<HTMLInputElement>("overlay-opacity").value;
  snapAnim = { from: camera.matrixWorld.clone(), to: frameMatrix(f), fov0: camera.fov, fov1: THREE.MathUtils.radToDeg(fov), t0: performance.now() };
  controls.enabled = false;
}

function clearSnap() {
  $("overlay").style.display = "none";
  controls.enabled = true;
}

function stepSnap(now: number) {
  if (!snapAnim) return;
  const s = Math.min(1, (now - snapAnim.t0) / 900);
  const e = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
  const p0 = new THREE.Vector3(), q0 = new THREE.Quaternion(), p1 = new THREE.Vector3(), q1 = new THREE.Quaternion();
  const sc = new THREE.Vector3();
  snapAnim.from.decompose(p0, q0, sc);
  snapAnim.to.decompose(p1, q1, sc);
  camera.position.copy(p0.lerp(p1, e));
  camera.quaternion.copy(q0.slerp(q1, e));
  camera.fov = THREE.MathUtils.lerp(snapAnim.fov0, snapAnim.fov1, e);
  camera.updateProjectionMatrix();
  if (s === 1) {
    // Keep orbiting around a point in front of the frame camera once the user moves again.
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    controls.target.copy(camera.position).addScaledVector(fwd, 1.0);
    snapAnim = null;
  }
}

// --- picking -----------------------------------------------------------------
const raycaster = new THREE.Raycaster();
let downAt = { x: 0, y: 0 };
canvas.addEventListener("pointerdown", (e) => { downAt = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener("pointerup", (e) => {
  if (Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return; // was a drag
  if (!controls.enabled) clearSnap();
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  raycaster.params.Line = { threshold: extent * 0.01 };

  if (frustumGroup.visible) {
    const hit = raycaster.intersectObjects(frustumGroup.children, false)[0];
    if (hit) {
      const i = hit.object.userData.frameIndex as number;
      showHits(`Frame ${pkg.frames[i].id}`, [{ index: i, u: NaN, v: NaN }]);
      showFrame(i);
      return;
    }
  }
  const hit = raycaster.intersectObject(splat, false)[0];
  if (!hit) return;
  markers.clear();
  const dot = new THREE.Mesh(new THREE.SphereGeometry(extent * 0.008, 16, 12), new THREE.MeshBasicMaterial({ color: 0xf5a524, depthTest: false }));
  dot.position.copy(hit.point);
  dot.renderOrder = 10;
  markers.add(dot);
  const hits = framesSeeing(pkg, hit.point, extent * 0.04);
  showHits(hits.length ? `${hits.length} frames saw this point` : "No frame saw this point", hits);
});

// --- UI wiring ---------------------------------------------------------------
$("panel-close").onclick = () => { $("panel").hidden = true; markers.clear(); clearSnap(); };
$("snap").onclick = () => selected && snapTo(selected.frame);
$<HTMLInputElement>("overlay-opacity").oninput = (e) => { $("overlay").style.opacity = (e.target as HTMLInputElement).value; };
$<HTMLInputElement>("show-path").onchange = (e) => { pathGroup.visible = (e.target as HTMLInputElement).checked; };
$<HTMLInputElement>("show-frustums").onchange = (e) => { frustumGroup.visible = (e.target as HTMLInputElement).checked; };
$<HTMLInputElement>("scrub").oninput = (e) => { playing = false; $("play").textContent = "▶"; setTime(Number((e.target as HTMLInputElement).value)); };
$("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; };

let extent = 1;
let last = performance.now();
function loop(now: number) {
  const dt = (now - last) / 1000;
  last = now;
  if (playing && pkg) {
    let t = Number($<HTMLInputElement>("scrub").value) + dt;
    if (t > duration) t = 0;
    setTime(t);
  }
  stepSnap(now);
  if (!snapAnim && controls.enabled) controls.update();
  renderer.render(scene, camera);
}

async function main() {
  const base = new URLSearchParams(location.search).get("scene");
  if (!base) {
    $("loading").textContent = "Add ?scene=scenes/<case_id> to the URL (run `crimesplat publish-local <case_id>` first).";
    return;
  }
  pkg = await loadPackage(base);
  const m = pkg.manifest;
  const q = m.quality;
  $("case-info").textContent = `${m.case_id} · ${pkg.frames.length} frames · ${m.units} · ` +
    `${(q.gaussians / 1000).toFixed(0)}k splats · PSNR ${q.psnr_holdout ?? "–"} dB · ` +
    `${Math.round(q.registered_frac * 100)}% registered`;

  const centres = pkg.frames.map(frameCentre);
  const box = new THREE.Box3().setFromPoints(centres);
  extent = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
  buildOverlays(extent);

  splat = new SplatMesh({ url: `${pkg.base}/${m.scene.splat}`, raycastable: true });
  scene.add(splat);
  await splat.initialized;
  $("loading").hidden = true;

  // Start at the first frame's viewpoint: interiors look like fog from outside the walls.
  const start = frameMatrix(pkg.frames[0]);
  start.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, extent * 0.25);
  duration = pkg.frames[pkg.frames.length - 1].t;
  const scrub = $<HTMLInputElement>("scrub");
  scrub.min = String(pkg.frames[0].t);
  scrub.max = String(duration);
  setTime(pkg.frames[0].t);
  resize();
  renderer.setAnimationLoop(loop);
}

main().catch((err) => {
  $("loading").textContent = `Failed to load scene: ${err.message}`;
  console.error(err);
});
// Exposed for debugging in the browser console.
Object.assign(window, { THREE, camera, project });
