// SLAM Replay app (tools/viz/replay_app.py, docs/DECISIONS.md D77, D78).
// Follow / Overview views and screenshots are rendered by the server with
// the MonoGS Gaussian rasterizer. Orbit, the free camera while it moves and
// the small rotating map are drawn here with WebGL from the map snapshot
// (one disc per Gaussian), so they turn smoothly without server round trips.
"use strict";

const RUN_COLORS = ["#5ab0e8", "#e8b04b", "#c084fc", "#7bd88f"];
const BASE_FPS = 15;          // playback at 1x, in recorded image frames per second
const KF_WINDOW = 10;         // keyframe frustums drawn
const FRUSTUM = 0.15, KF_FRUSTUM = 0.08;
const CEILING_MARGIN = 0.3;   // m above the highest camera (as render_replay.py)
const FOV = 70;               // horizontal, degrees (as render_replay.py)
const COL = { est: [0.90, 0.16, 0.16], gt: [0.24, 0.78, 0.24], kf: [0.16, 0.55, 0.90] };

const S = {
  renderer: "", runs: [], active: [], frames: [], idx: 0,
  playing: false, speed: 1, mode: "follow", color: "color",
  layers: new Set(["est", "gt", "frustum", "kf"]),
  ceiling: false, ceilingTouched: false, trail: 300, trailTouched: false,
  thumbs: true, mini: true, orbitSec: 30, ptScale: 5, crisp: true,
  free: null, freeVersion: 0, interacting: false, errLog: false,
  panel: null, token: 0,
};
const D = new Map();          // run index -> {traj: Float32Array, row: Map(frame -> row)}
const pointsCache = new Map(); // "run:snap:color" -> Promise<ArrayBuffer>
let views = [];               // per active run: DOM + GL handles

const $ = (id) => document.getElementById(id);
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return r;
}
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add("hidden"), 5000);
}
const runColor = (i) => RUN_COLORS[i % RUN_COLORS.length];
const activeRuns = () => S.active.map((i) => S.runs[i]);
const curFrame = () => S.frames[S.idx];
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–");

// ------------------------------------------------------------------ vectors

const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map((v) => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => mul(a, 1 / (norm(a) || 1));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function basis(up) {
  let e1 = cross(up, [1, 0, 0]);
  if (norm(e1) < 1e-6) e1 = cross(up, [0, 1, 0]);
  e1 = unit(e1);
  return [e1, cross(up, e1)];
}
function sphereEye(target, up, dist, el, az) {
  const [e1, e2] = basis(up);
  const h = add(mul(e1, Math.cos(az)), mul(e2, Math.sin(az)));
  return add(target, mul(add(mul(h, Math.cos(el)), mul(up, Math.sin(el))), dist));
}
const freeEye = (f, up) => sphereEye(f.target, up, f.dist, f.el, f.az);
function freeFrom(eye, target, up) {
  const [e1, e2] = basis(up);
  const d = sub(eye, target);
  const dist = norm(d);
  return { target: target.slice(), dist, el: Math.asin(Math.max(-1, Math.min(1, dot(d, up) / dist))),
           az: Math.atan2(dot(d, e2), dot(d, e1)) };
}
function orbitCamera(run, t) {
  const angle = (2 * Math.PI * t) / S.orbitSec;
  return { eye: sphereEye(run.centre, run.up, run.orbit_distance * run.extent,
                          (run.orbit_elevation * Math.PI) / 180, angle),
           target: run.centre, angle };
}

// ------------------------------------------------------------------ WebGL

function lookAt(eye, target, up) {
  const z = unit(sub(eye, target));       // camera looks along -z
  const x = unit(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0]);
}

const PT_VS = `#version 300 es
in vec3 aPos; in float aRad; in vec3 aCol; in float aOp;
uniform mat4 uView, uProj; uniform float uFocal, uScale, uCeil; uniform vec3 uUp;
out vec3 vCol; out float vCut; out float vOp;
void main() {
  vec4 pv = uView * vec4(aPos, 1.0);
  gl_Position = uProj * pv;
  // diameter = uScale sigma (the sprite edge at r = 1)
  gl_PointSize = clamp(uScale * aRad * uFocal / max(-pv.z, 1e-3), 1.0, 40.0);
  vCol = aCol; vOp = aOp; vCut = dot(aPos, uUp) - uCeil;
}`;
const PT_FS = `#version 300 es
precision mediump float;
uniform highp float uScale;  // shared with the vertex shader: same precision
in vec3 vCol; in float vCut; in float vOp; out vec4 o;
void main() {
  if (vCut > 0.0) discard;
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  // Gaussian falloff turned into MSAA coverage (alpha-to-coverage): soft
  // edges while the depth test still resolves occlusion, no sorting needed.
  // r = 1 is uScale/2 sigma from the centre: exp(-(r uScale/2)^2 / 2)
  o = vec4(vCol, vOp * exp(-0.125 * uScale * uScale * r2));
}`;
const LN_VS = `#version 300 es
in vec3 aPos; in vec3 aCol; uniform mat4 uView, uProj; out vec3 vCol;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); vCol = aCol; }`;
const LN_FS = `#version 300 es
precision mediump float; in vec3 vCol; out vec4 o; void main() { o = vec4(vCol, 1.0); }`;

class GLView {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 không khả dụng");
    this.gl = gl;
    this.pt = this.program(PT_VS, PT_FS);
    this.ln = this.program(LN_VS, LN_FS);
    this.ptVao = gl.createVertexArray();
    this.lnVao = gl.createVertexArray();
    this.buf = { pos: gl.createBuffer(), rad: gl.createBuffer(), col: gl.createBuffer(), op: gl.createBuffer(),
                 lpos: gl.createBuffer(), lcol: gl.createBuffer() };
    this.n = 0; this.nl = 0; this.pointsKey = null; this.wanted = null; this.linesKey = null;
  }
  program(vs, fs) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  attrib(prog, vao, name, buffer, size, type, normalized) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, name);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, normalized, 0, 0);
  }
  // payload of /api/points: uint32 x4 header, xyz f32, radius f32, rgb u8, opacity u8
  setPoints(key, buf) {
    const gl = this.gl;
    const n = new Uint32Array(buf, 0, 4)[0];
    const pos = new Float32Array(buf, 16, 3 * n);
    const rad = new Float32Array(buf, 16 + 12 * n, n);
    const col = new Uint8Array(buf, 16 + 16 * n, 3 * n);
    const op = new Uint8Array(buf, 16 + 19 * n, n);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.rad); gl.bufferData(gl.ARRAY_BUFFER, rad, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.col); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.op); gl.bufferData(gl.ARRAY_BUFFER, op, gl.STATIC_DRAW);
    this.attrib(this.pt, this.ptVao, "aOp", this.buf.op, 1, gl.UNSIGNED_BYTE, true);
    this.attrib(this.pt, this.ptVao, "aPos", this.buf.pos, 3, gl.FLOAT, false);
    this.attrib(this.pt, this.ptVao, "aRad", this.buf.rad, 1, gl.FLOAT, false);
    this.attrib(this.pt, this.ptVao, "aCol", this.buf.col, 3, gl.UNSIGNED_BYTE, true);
    this.n = n; this.pointsKey = key;
  }
  want(key, promise) {
    if (key === this.pointsKey || key === this.wanted) return;
    this.wanted = key;
    promise.then((buf) => { if (this.wanted === key) this.setPoints(key, buf); }).catch(() => {});
  }
  setLines(key, pos, col) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.lpos); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.lcol); gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
    this.attrib(this.ln, this.lnVao, "aPos", this.buf.lpos, 3, gl.FLOAT, false);
    this.attrib(this.ln, this.lnVao, "aCol", this.buf.lcol, 3, gl.FLOAT, false);
    this.nl = pos.length / 3; this.linesKey = key;
  }
  draw(cam, up, ceil) {
    const gl = this.gl, c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const aspect = w / h;
    const fovx = (FOV * Math.PI) / 180;
    const fovy = 2 * Math.atan(Math.tan(fovx / 2) / aspect);
    const view = lookAt(cam.eye, cam.target, up);
    const proj = perspective(fovy, aspect, 0.02, 100);
    if (this.n) {
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(this.pt);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.pt, "uView"), false, view);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.pt, "uProj"), false, proj);
      gl.uniform1f(gl.getUniformLocation(this.pt, "uFocal"), h / 2 / Math.tan(fovy / 2));
      gl.uniform1f(gl.getUniformLocation(this.pt, "uScale"), S.ptScale);
      gl.uniform1f(gl.getUniformLocation(this.pt, "uCeil"), ceil);
      gl.uniform3fv(gl.getUniformLocation(this.pt, "uUp"), up);
      gl.bindVertexArray(this.ptVao);
      gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
      gl.drawArrays(gl.POINTS, 0, this.n);
      gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    }
    if (this.nl) {
      gl.disable(gl.DEPTH_TEST);   // trajectories stay visible, as on the server renders
      gl.useProgram(this.ln);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.ln, "uView"), false, view);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.ln, "uProj"), false, proj);
      gl.bindVertexArray(this.lnVao);
      gl.drawArrays(gl.LINES, 0, this.nl);
    }
  }
}
function makeGL(canvas) {
  try { return new GLView(canvas); } catch (e) { toast(`WebGL: ${e.message}`); return null; }
}

// ------------------------------------------------------------------ run data (WebGL)

function loadTraj(i) {
  if (!D.has(i)) {
    const entry = { ready: null };
    D.set(i, entry);
    api(`/api/traj?run=${i}`).then((r) => r.arrayBuffer()).then((buf) => {
      const traj = new Float32Array(buf);
      const row = new Map();
      for (let r = 0; r < traj.length / 25; r++) row.set(Math.round(traj[r * 25]), r);
      entry.ready = { traj, row };
    }).catch((e) => toast(`Quỹ đạo: ${e.message}`));
  }
  return D.get(i).ready;
}
function snapIndex(run, f) {
  let k = 0;
  run.gaussians.forEach((g, j) => { if (g[0] <= f) k = j; });
  return k;
}
function points(i, snap, color) {
  const key = `${i}:${snap}:${color}`;
  if (!pointsCache.has(key)) {
    pointsCache.set(key, api(`/api/points?run=${i}&snap=${snap}&color=${color}`).then((r) => r.arrayBuffer()));
    while (pointsCache.size > 12) pointsCache.delete(pointsCache.keys().next().value);
  }
  return [key, pointsCache.get(key)];
}
const posOf = (t, r, off) => [t[r * 25 + off + 3], t[r * 25 + off + 7], t[r * 25 + off + 11]];
function frustumSegs(t, r, off, intr, depth) {
  const R = (i, j) => t[r * 25 + off + i * 4 + j];
  const o = posOf(t, r, off);
  const corners = [[0, 0], [intr.width, 0], [intr.width, intr.height], [0, intr.height]].map(([u, v]) => {
    const d = [((u - intr.cx) / intr.fx) * depth, ((v - intr.cy) / intr.fy) * depth, depth];
    return [0, 1, 2].map((i) => R(i, 0) * d[0] + R(i, 1) * d[1] + R(i, 2) * d[2] + o[i]);
  });
  const segs = corners.map((c) => [o, c]);
  for (let i = 0; i < 4; i++) segs.push([corners[i], corners[(i + 1) % 4]]);
  return segs;
}
// line vertices (positions, colours) of the overlays at frame f
function buildLines(run, data, f, trail, layers, withKf) {
  const pos = [], col = [];
  const push = (a, b, c) => { pos.push(...a, ...b); col.push(...c, ...c); };
  const t = data.traj, r = data.row.get(f);
  if (r === undefined) return [new Float32Array(0), new Float32Array(0)];
  let r0 = 0;
  if (trail) { while (r0 < r && t[r0 * 25] <= f - trail) r0++; }
  for (const [layer, off] of [["gt", 13], ["est", 1]]) {
    if (!layers.has(layer)) continue;
    for (let q = r0; q < r; q++) push(posOf(t, q, off), posOf(t, q + 1, off), COL[layer]);
  }
  if (withKf && layers.has("kf")) {
    for (const k of run.keyframes.filter((k) => k <= f).slice(-KF_WINDOW)) {
      const rk = data.row.get(k);
      if (rk !== undefined) for (const [a, b] of frustumSegs(t, rk, 1, run.intrinsics, KF_FRUSTUM)) push(a, b, COL.kf);
    }
  }
  if (layers.has("frustum")) {
    if (layers.has("gt")) for (const [a, b] of frustumSegs(t, r, 13, run.intrinsics, FRUSTUM)) push(a, b, COL.gt);
    for (const [a, b] of frustumSegs(t, r, 1, run.intrinsics, FRUSTUM)) push(a, b, COL.est);
  }
  return [new Float32Array(pos), new Float32Array(col)];
}

// ------------------------------------------------------------------ data

async function loadRuns() {
  const data = await (await api("/api/runs")).json();
  S.renderer = data.renderer;
  S.runs = data.runs;
  $("renderer").textContent = data.renderer === "gaussian" ? "GPU" : "CPU preview";
  $("renderer").title = data.renderer === "gaussian" ? "Follow / Overview: rasterizer Gaussian (CUDA)"
    : "Không có CUDA: Follow / Overview là bản xem thử trên CPU";
  S.active = S.active.filter((i) => i < S.runs.length);
  if (!S.active.length && S.runs.length) S.active = [0];
}
async function discover() {
  try {
    const data = await (await api("/api/discover")).json();
    $("discovered").innerHTML = data.records.map((r) => `<option value="${r.path}">${r.run}</option>`).join("");
  } catch (e) { /* optional */ }
}
function computeFrames() {
  const runs = activeRuns();
  const keep = S.frames[S.idx];
  if (!runs.length) { S.frames = []; S.idx = 0; return; }
  let common = new Set(runs[0].frames);
  for (const r of runs.slice(1)) common = new Set(r.frames.filter((f) => common.has(f)));
  S.frames = [...common].sort((a, b) => a - b);
  const j = S.frames.indexOf(keep);
  S.idx = j >= 0 ? j : Math.min(S.idx, Math.max(0, S.frames.length - 1));
}
const nKf = (run, f) => run.n_keyframes[String(f)] ?? 0;
function errorStats(run, f) {
  let cur = NaN, sum = 0, n = 0, max = 0;
  for (const [g, e] of run.error) {
    if (g > f) break;
    cur = e; sum += e; n++; if (e > max) max = e;
  }
  return { cur, mean: n ? sum / n : NaN, max };
}
const gaussiansAt = (run, f) => (run.gaussians.length ? run.gaussians[snapIndex(run, f)][1] : 0);
function lightAt(run, f) {
  let row = null;
  for (const r of run.light) if (r.frame <= f) row = r;
  return row || run.light[0] || null;
}

// ------------------------------------------------------------------ stage

function tileHTML(kind, cap) {
  return `<div class="tile" data-t="${kind}"><img data-kind="${kind}" alt=""><div class="cap">${cap}</div></div>`;
}
function buildStage() {
  const stage = $("stage");
  const runs = activeRuns();
  stage.className = runs.length > 1 ? "compare" : "single";
  stage.style.gridTemplateColumns = `repeat(${Math.max(1, runs.length)}, minmax(0, 1fr))`;
  stage.innerHTML = "";
  views = [];
  if (!runs.length) {
    stage.innerHTML = `<div class="empty-stage">Chưa mở bản ghi nào. Mở ngăn Bản ghi ở thanh bên phải.</div>`;
    return;
  }
  runs.forEach((run, k) => {
    const i = S.active[k];
    const el = document.createElement("div");
    el.className = "view";
    const albedoTile = run.has_albedo ? tileHTML("albedo", "albedo")
      : `<div class="tile" data-t="albedo"><div class="empty">không có albedo · MonoGS gốc</div><div class="cap">albedo</div></div>`;
    el.innerHTML = `
      <div class="vhead"><i class="sw" style="background:${runColor(i)}"></i><span class="name">${run.label}</span>
        <span class="sub">${run.run}</span>
        <span class="legend"><span><i style="background:var(--est)"></i>ước lượng</span>
          <span><i style="background:var(--gt)"></i>tham chiếu</span><span><i style="background:var(--kf)"></i>keyframe</span></span></div>
      <div class="viewport"><img alt=""><canvas></canvas><i class="busy"></i><span class="mode"></span></div>
      <div class="strip">${tileHTML("obs", "quan sát")}${tileHTML("render", "render")}${albedoTile}${tileHTML("depth", "độ sâu")}
        <div class="tile mini" data-t="mini"><canvas></canvas><div class="cap">bản đồ 3D</div></div></div>`;
    stage.appendChild(el);
    const vp = el.querySelector(".viewport");
    const v = { k, i, el, vp, img: vp.querySelector("img"), canvas: vp.querySelector("canvas"),
      modeLabel: vp.querySelector(".mode"), gl: null, mini: el.querySelector(".tile.mini canvas"), miniGL: null,
      tiles: [...el.querySelectorAll("img[data-kind]")], crispVersion: -1 };
    views.push(v);
    attachMouse(v);
    loadTraj(i);
  });
  layoutStage();
}
function layoutStage() {
  const stage = $("stage");
  document.querySelectorAll(".tile:not(.mini)").forEach((t) => t.classList.toggle("hidden", !S.thumbs));
  document.querySelectorAll(".tile.mini").forEach((t) => t.classList.toggle("hidden", !S.mini));
  const n = (S.thumbs ? 4 : 0) + (S.mini ? 1 : 0);
  stage.style.setProperty("--tiles", Math.max(n, 1));
  document.querySelectorAll(".strip").forEach((s) => s.classList.toggle("hidden", n === 0));
  if (stage.className === "single") {
    const H = stage.clientHeight - 16 - 22 - 6;
    const tileH = (H - (n - 1) * 6) / Math.max(n, 1);
    const w = n ? Math.max(150, Math.min((tileH * 16) / 9, stage.clientWidth * 0.3)) : 0;
    document.querySelectorAll(".view").forEach((v) => {
      v.style.gridTemplateColumns = n ? `minmax(0, 1fr) ${Math.round(w)}px` : "minmax(0, 1fr)";
    });
  }
}

// ------------------------------------------------------------------ modes

const usesGL = (v) => S.mode === "orbit"
  || (S.mode === "free" && (!S.crisp || S.interacting || v.crispVersion !== S.freeVersion));
function setMode(mode) {
  if (mode === "free" && !S.free && S.runs.length) {
    const r = activeRuns()[0];
    S.free = freeFrom(r.overview_eye, r.centre, r.up);
  }
  S.mode = mode;
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  if (!S.ceilingTouched) { S.ceiling = mode !== "follow"; $("ceiling").checked = S.ceiling; }
  if (!S.trailTouched) { S.trail = mode === "follow" ? 300 : 0; $("trail").value = String(S.trail); }
  views.forEach((v) => v.vp.classList.toggle("free", mode === "free"));
  S.freeVersion++;
  show();
}

// ------------------------------------------------------------------ server renders

function renderURL(k, w, h, f) {
  const run = activeRuns()[k];
  const p = new URLSearchParams({ run: S.active[k], frame: f, w, h, color: S.color, mode: S.mode,
    layers: [...S.layers].join(","), trail: S.trail, fov: FOV, ceiling: S.ceiling ? CEILING_MARGIN : "none" });
  if (S.mode === "free") {
    p.set("eye", freeEye(S.free, run.up).join(","));
    p.set("target", S.free.target.join(","));
  }
  if (S.mode === "orbit") p.set("angle", orbitCamera(activeRuns()[0], performance.now() / 1000).angle);
  return "/api/render?" + p;
}
function boxSize(el) {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
  const w = Math.round(el.clientWidth * dpr), h = Math.round(el.clientHeight * dpr);
  const s = Math.min(1, 1600 / w, 900 / h);
  return [Math.max(64, Math.round(w * s)), Math.max(36, Math.round(h * s))];
}
const prefetched = new Map();
async function fetchNow(url) {
  const r = await api(url);
  const blob = await r.blob();
  return { src: URL.createObjectURL(blob), n: r.headers.get("X-Gaussians") };
}
function fetchImage(url) {
  const p = prefetched.get(url);
  if (p) { prefetched.delete(url); return p; }
  return fetchNow(url);
}
function prefetch(i) {
  if (i < 0 || i >= S.frames.length || !(S.mode === "follow" || S.mode === "overview")) return;
  views.forEach((v) => {
    const [w, h] = boxSize(v.vp);
    const u = renderURL(v.k, w, h, S.frames[i]);
    if (!prefetched.has(u)) prefetched.set(u, fetchNow(u));
  });
  while (prefetched.size > 8) {
    const key = prefetched.keys().next().value;
    prefetched.get(key).then(({ src }) => URL.revokeObjectURL(src)).catch(() => {});
    prefetched.delete(key);
  }
}
function setImg(img, src) {
  return new Promise((res) => {
    const old = img.src;
    img.onload = img.onerror = () => { if (old.startsWith("blob:")) URL.revokeObjectURL(old); res(); };
    img.src = src;
  });
}
async function serverView(v, f, token) {
  const [w, h] = boxSize(v.vp);
  const version = S.freeVersion;
  v.vp.classList.add("loading");
  try {
    const { src } = await fetchImage(renderURL(v.k, w, h, f));
    if (token !== S.token) { URL.revokeObjectURL(src); return; }
    await setImg(v.img, src);
    if (S.mode === "free") v.crispVersion = version;
  } catch (e) {
    if (token === S.token) toast(`Render lỗi: ${e.message}`);
  } finally {
    v.vp.classList.remove("loading");
  }
}

// ------------------------------------------------------------------ frame display

async function show() {
  drawTimeline();
  if (!S.frames.length) return;
  const token = ++S.token;
  const f = curFrame();
  const run0 = activeRuns()[0];
  $("readout").innerHTML = `frame ${f} <span>· ${S.idx + 1}/${S.frames.length} · kf ${nKf(run0, f)}</span>`;
  const jobs = [];
  views.forEach((v) => {
    if (S.thumbs) {
      v.tiles.forEach((img) => {
        img.src = "/api/image?" + new URLSearchParams({ run: v.i, frame: f, kind: img.dataset.kind });
      });
    }
    const server = S.mode === "follow" || S.mode === "overview" || (S.mode === "free" && S.crisp && !S.interacting);
    if (server) jobs.push(serverView(v, f, token));
  });
  updatePanel();
  await Promise.all(jobs);
}

// render loop of the WebGL views: orbit, free camera while moving, small map
function tick(now) {
  const t = now / 1000;
  const f = curFrame();
  if (f !== undefined && views.length) {
    const run0 = activeRuns()[0];
    views.forEach((v) => {
      const run = S.runs[v.i];
      const gl = usesGL(v);
      v.vp.classList.toggle("gl", gl);
      v.modeLabel.textContent = S.mode === "orbit" ? "orbit · WebGL"
        : S.mode === "free" ? (gl ? "tự do · WebGL" : "tự do · Gaussian") : S.mode;
      const data = loadTraj(v.i);
      if (!data) return;
      const snap = snapIndex(run, f);
      if (gl) {
        v.gl = v.gl || makeGL(v.canvas);
        if (v.gl) {
          v.gl.want(...points(v.i, snap, S.color));
          const key = `${f}|${S.trail}|${[...S.layers]}`;
          if (v.gl.linesKey !== key) v.gl.setLines(key, ...buildLines(run, data, f, S.trail, S.layers, true));
          const cam = S.mode === "orbit" ? orbitCamera(run0, t) : { eye: freeEye(S.free, run0.up), target: S.free.target };
          v.gl.draw(cam, run.up, S.ceiling ? run.top + CEILING_MARGIN : 1e9);
        }
      }
      if (S.mini && v.mini.offsetParent !== null) {
        v.miniGL = v.miniGL || makeGL(v.mini);
        if (v.miniGL) {
          v.miniGL.want(...points(v.i, snap, S.color));
          const key = `${f}|mini|${[...S.layers]}`;
          const layers = new Set([...S.layers].filter((l) => l !== "kf"));
          if (v.miniGL.linesKey !== key) v.miniGL.setLines(key, ...buildLines(run, data, f, 0, layers, false));
          v.miniGL.draw(orbitCamera(run0, t), run.up, run.top + CEILING_MARGIN);
        }
      }
    });
  }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ playback

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function setPlayIcon() { $("play").innerHTML = `<svg><use href="#i-${S.playing ? "pause" : "play"}"/></svg>`; }
async function play() {
  if (S.playing) { S.playing = false; setPlayIcon(); return; }
  if (S.idx >= S.frames.length - 1) S.idx = 0;
  S.playing = true;
  setPlayIcon();
  while (S.playing) {
    const t0 = performance.now();
    await show();
    if (!S.playing) break;
    if (S.idx >= S.frames.length - 1) { S.playing = false; break; }
    S.idx += 1;
    prefetch(S.idx);
    prefetch(S.idx + 1);
    const wait = 1000 / (BASE_FPS * S.speed) - (performance.now() - t0);
    if (wait > 0) await sleep(wait);
  }
  setPlayIcon();
}
function seek(i) {
  S.idx = Math.max(0, Math.min(S.frames.length - 1, i));
  if (!S.playing) show();
}
function seekFrame(frame) {
  let best = 0;
  S.frames.forEach((f, i) => { if (Math.abs(f - frame) < Math.abs(S.frames[best] - frame)) best = i; });
  seek(best);
}

// ------------------------------------------------------------------ free camera

let crispTimer = null;
function settle() {
  clearTimeout(crispTimer);
  crispTimer = setTimeout(() => { S.interacting = false; if (S.mode === "free") show(); }, 280);
}
function attachMouse(v) {
  let drag = null;
  v.vp.addEventListener("contextmenu", (e) => e.preventDefault());
  v.vp.addEventListener("mousedown", (e) => {
    if (S.mode !== "free") {
      // dragging another view continues from that view with the free camera
      const r = activeRuns()[0];
      if (S.mode === "orbit") {
        const c = orbitCamera(r, performance.now() / 1000);
        S.free = freeFrom(c.eye, c.target, r.up);
      } else if (S.mode === "overview" || !S.free) S.free = freeFrom(r.overview_eye, r.centre, r.up);
      S.interacting = true;
      setMode("free");
    }
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    S.interacting = true;
    S.freeVersion++;
    views.forEach((w) => w.vp.classList.add("dragging"));
    e.preventDefault();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    drag = null;
    views.forEach((w) => w.vp.classList.remove("dragging"));
    settle();
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    const up = activeRuns()[0].up, f = S.free;
    if (drag.pan) {
      const fwd = unit(sub(f.target, freeEye(f, up)));
      const right = unit(cross(fwd, up));
      const upv = cross(right, fwd);
      f.target = add(f.target, mul(add(mul(right, -dx), mul(upv, dy)), f.dist * 0.0015));
    } else {
      f.az -= dx * 0.007;
      f.el = Math.max(-1.45, Math.min(1.45, f.el + dy * 0.007));
    }
    S.freeVersion++;
  });
  v.vp.addEventListener("wheel", (e) => {
    if (S.mode !== "free") return;
    e.preventDefault();
    S.free.dist = Math.max(0.2, Math.min(50, S.free.dist * Math.exp(e.deltaY * 0.001)));
    S.interacting = true;
    S.freeVersion++;
    settle();
  }, { passive: false });
}

// ------------------------------------------------------------------ timeline

function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}
let hoverX = null;
function drawTimeline() {
  const [g, W, H] = setupCanvas($("tl"));
  if (!S.frames.length) return;
  const x0 = S.frames[0], x1 = S.frames[S.frames.length - 1];
  const X = (f) => 4 + ((f - x0) / (x1 - x0 || 1)) * (W - 8);
  const y = Math.round(H / 2);
  g.fillStyle = "#26262b"; g.fillRect(4, y - 1, W - 8, 2);
  g.fillStyle = "#8a8a93"; g.fillRect(4, y - 1, X(curFrame()) - 4, 2);
  const run0 = activeRuns()[0];
  g.fillStyle = "rgba(40,140,230,.6)";
  for (const kf of run0 ? run0.keyframes : []) if (kf >= x0 && kf <= x1) g.fillRect(Math.round(X(kf)), y + 5, 1, 5);
  g.fillStyle = "#e8b04b"; g.fillRect(Math.round(X(curFrame())) - 1, y - 8, 2, 16);
  if (hoverX !== null) {
    const fr = Math.round(x0 + ((hoverX - 4) / (W - 8)) * (x1 - x0));
    g.fillStyle = "#f4f4f5"; g.font = "10.5px Consolas, monospace"; g.textAlign = "center";
    g.fillText(String(Math.max(x0, Math.min(x1, fr))), Math.max(20, Math.min(W - 20, hoverX)), y - 9);
  }
}
function wireTimeline() {
  const c = $("tl");
  let down = false;
  const seekAt = (e) => {
    const r = c.getBoundingClientRect();
    const x0 = S.frames[0], x1 = S.frames[S.frames.length - 1];
    seekFrame(x0 + Math.max(0, Math.min(1, (e.clientX - r.left - 4) / (r.width - 8))) * (x1 - x0));
  };
  c.addEventListener("mousedown", (e) => { down = true; seekAt(e); });
  window.addEventListener("mouseup", () => (down = false));
  c.addEventListener("mousemove", (e) => {
    hoverX = e.clientX - c.getBoundingClientRect().left;
    if (down) seekAt(e); else drawTimeline();
  });
  c.addEventListener("mouseleave", () => { hoverX = null; drawTimeline(); });
}

// ------------------------------------------------------------------ drawer

const PANEL_TITLES = { stats: "Thông số", charts: "Đồ thị", display: "Hiển thị", records: "Bản ghi" };
function openPanel(name) {
  S.panel = S.panel === name ? null : name;
  $("drawer").classList.toggle("closed", !S.panel);
  document.querySelectorAll("#rail button").forEach((b) => b.classList.toggle("on", b.dataset.open === S.panel));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("on", p.dataset.panel === S.panel));
  $("dtitle").textContent = PANEL_TITLES[S.panel] || "";
  if (S.panel === "records") buildRecords();
  requestAnimationFrame(() => { layoutStage(); show(); });
}
function updatePanel() {
  if (S.panel === "stats") drawStats();
  if (S.panel === "charts") drawCharts();
}
function drawStats() {
  const f = curFrame();
  document.querySelector('[data-panel="stats"]').innerHTML = activeRuns().map((run, k) => {
    const e = errorStats(run, f);
    const rmse = Math.sqrt(run.error.reduce((s, [, v]) => s + v * v, 0) / Math.max(run.error.length, 1));
    return `<div class="runstats"><div class="rs-head"><i class="sw" style="background:${runColor(S.active[k])}"></i>${run.label}</div>
      <div class="kv">
        <span class="k">Frame</span><span class="v big">${f}</span>
        <span class="k">Keyframe</span><span class="v big">${nKf(run, f)}</span>
        <span class="k">Gaussian trong bản đồ</span><span class="v big">${gaussiansAt(run, f).toLocaleString("vi-VN")}</span>
        <span class="k">Sai số vị trí hiện tại</span><span class="v big">${fmt(e.cur)} cm</span>
        <span class="k">Trung bình tới đây</span><span class="v">${fmt(e.mean)} cm</span>
        <span class="k">Lớn nhất tới đây</span><span class="v">${fmt(e.max)} cm</span>
        <span class="k">RMSE cả chuỗi (pose live)</span><span class="v">${fmt(rmse)} cm</span>
        <span class="k">Màu bản đồ</span><span class="v">${run.has_albedo ? "albedo" : "đã nung ánh sáng"}</span>
        <span class="k">Ghi ảnh mỗi</span><span class="v">${run.frame_every} frame</span>
        <span class="k">Thư mục</span><span class="v" title="${run.path}">${run.run}</span>
      </div></div>`;
  }).join("") || `<p class="note">Chưa có bản ghi.</p>`;
}
function buildRecords() {
  $("runlist").innerHTML = S.runs.map((run, i) => `
    <div class="rec"><input type="checkbox" data-show="${i}" ${S.active.includes(i) ? "checked" : ""} title="Hiện / ẩn">
      <div><div class="name"><i class="sw" style="background:${runColor(i)}"></i> ${run.label}</div>
        <div class="path">${run.path}</div></div>
      <button class="icon small" data-remove="${i}" title="Gỡ khỏi app (không xóa file)"><svg><use href="#i-close"/></svg></button>
    </div>`).join("") || `<p class="note">Chưa có bản ghi nào.</p>`;
  document.querySelectorAll("[data-show]").forEach((cb) => (cb.onchange = () => {
    const i = +cb.dataset.show;
    if (cb.checked) S.active = [...S.active, i].slice(-2);
    else if (S.active.length > 1) S.active = S.active.filter((x) => x !== i);
    refreshAll();
  }));
  document.querySelectorAll("[data-remove]").forEach((b) => (b.onclick = () => removeRun(+b.dataset.remove)));
}
async function removeRun(i) {
  try {
    await api("/api/remove", { method: "POST", body: JSON.stringify({ index: i }) });
    S.active = S.active.filter((x) => x !== i).map((x) => (x > i ? x - 1 : x));
    D.clear(); pointsCache.clear();
    await loadRuns();
    refreshAll();
  } catch (e) { toast(`Không gỡ được: ${e.message}`); }
}
async function openRecord() {
  const path = $("openpath").value.trim();
  if (!path) return;
  try {
    const data = await (await api("/api/load", { method: "POST", body: JSON.stringify({ path }) })).json();
    await loadRuns();
    if (!S.active.includes(data.index)) S.active = [...S.active, data.index].slice(-2);
    $("openpath").value = "";
    refreshAll();
  } catch (e) { toast(`Không mở được: ${e.message}`); }
}

// ------------------------------------------------------------------ charts

function niceTicks(lo, hi, n) {
  const step0 = (hi - lo || 1) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function lineChart(canvas, series, opt) {
  const [g, W, H] = setupCanvas(canvas);
  const box = { l: 40, r: 6, t: 6, b: 18 };
  const pw = W - box.l - box.r, ph = H - box.t - box.b;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  g.font = "10.5px Consolas, monospace";
  if (!xs.length) { g.fillStyle = "#55555c"; g.fillText(opt.empty || "không có dữ liệu", box.l, H / 2); return null; }
  const x0 = opt.xMin ?? Math.min(...xs), x1 = opt.xMax ?? Math.max(...xs);
  const X = (x) => box.l + ((x - x0) / (x1 - x0 || 1)) * pw;
  let y0 = opt.yMin ?? 0, y1 = opt.yMax ?? (Math.max(...ys) * 1.08 || 1);
  let Y = (y) => box.t + ph - ((y - y0) / (y1 - y0 || 1)) * ph;
  let ticks = niceTicks(y0, y1, 3);
  if (opt.log) {
    // decades between the smallest positive value and the largest
    const pos = ys.filter((v) => v > 0);
    const l0 = Math.floor(Math.log10(Math.min(...pos, 1))), l1 = Math.ceil(Math.log10(Math.max(...pos, 1)));
    Y = (y) => box.t + ph - ((Math.log10(Math.max(y, 10 ** l0)) - l0) / (l1 - l0 || 1)) * ph;
    ticks = [];
    for (let e = l0; e <= l1; e++) ticks.push(10 ** e);
  }
  g.strokeStyle = "#1b1b1f"; g.fillStyle = "#6b6b73"; g.lineWidth = 1;
  for (const v of ticks) {
    g.beginPath(); g.moveTo(box.l, Math.round(Y(v)) + 0.5); g.lineTo(W - box.r, Math.round(Y(v)) + 0.5); g.stroke();
    g.textAlign = "right"; g.fillText(opt.yFmt ? opt.yFmt(v) : v, box.l - 5, Y(v) + 3);
  }
  g.textAlign = "center";
  for (const v of niceTicks(x0, x1, 4)) g.fillText(opt.xFmt ? opt.xFmt(v) : v, X(v), H - 4);
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.4; g.setLineDash(s.dash || []);
    g.beginPath();
    s.points.forEach(([x, y], i) => {
      if (i === 0) g.moveTo(X(x), Y(y));
      else if (s.step) { g.lineTo(X(x), Y(s.points[i - 1][1])); g.lineTo(X(x), Y(y)); }
      else g.lineTo(X(x), Y(y));
    });
    if (s.step && s.points.length) g.lineTo(X(x1), Y(s.points[s.points.length - 1][1]));
    g.stroke();
  }
  g.setLineDash([]);
  if (opt.cursor !== undefined) {
    g.strokeStyle = "#e8b04b"; g.lineWidth = 1;
    const cx = Math.round(X(opt.cursor)) + 0.5;
    g.beginPath(); g.moveTo(cx, box.t); g.lineTo(cx, box.t + ph); g.stroke();
  }
  return { x0, x1, l: box.l, pw };
}
const chartBoxes = {};
function drawCharts() {
  const runs = activeRuns();
  const f = curFrame();
  const colorOf = (k) => runColor(S.active[k]);
  const fmax = Math.max(1, ...runs.map((r) => (r.error.length ? r.error[r.error.length - 1][0] : 0)));
  chartBoxes.error = lineChart($("chart-error"), runs.map((r, k) => ({ color: colorOf(k), points: r.error })),
    { xMin: 0, xMax: fmax, cursor: f, log: S.errLog, yFmt: (v) => (v < 1 ? v.toPrecision(1) : v) });
  chartBoxes.gauss = lineChart($("chart-gauss"), runs.map((r, k) => ({ color: colorOf(k), points: r.gaussians, step: true })),
    { xMin: 0, xMax: fmax, cursor: f, yFmt: (v) => (v >= 1000 ? `${v / 1000}k` : v) });
  const series = [], vals = [];
  runs.forEach((r, k) => {
    const row = lightAt(r, f);
    if (!row) return;
    if (row.phi) {
      const pts = (phi) => phi.map((v, i) => [i, v]);
      series.push({ color: colorOf(k), points: pts(r.light[0].phi), dash: [3, 3], width: 1 });
      series.push({ color: colorOf(k), points: pts(row.phi), width: 1.8 });
    }
    for (const [key, v] of Object.entries(row.scalars)) {
      if (!key.endsWith("n_at_bound")) vals.push(`<span class="k">${key}</span><span class="v">${v.toPrecision(4)}</span>`);
    }
  });
  chartBoxes.phi = lineChart($("chart-phi"), series, { xMin: 0, xMax: 50, yMin: 0, yMax: 1.05,
    xFmt: (v) => `${v}°`, empty: "tham số đèn cố định / không có mô hình đèn" });
  $("lightvals").innerHTML = vals.join("");
}
function chartClick(name) {
  const canvas = $(`chart-${name}`);
  canvas.addEventListener("click", (e) => {
    const b = chartBoxes[name];
    if (!b) return;
    const rect = canvas.getBoundingClientRect();
    seekFrame(b.x0 + Math.max(0, Math.min(1, (e.clientX - rect.left - b.l) / b.pw)) * (b.x1 - b.x0));
  });
}

// ------------------------------------------------------------------ actions

async function screenshot() {
  const f = curFrame();
  for (let k = 0; k < S.active.length; k++) {
    try {
      const { src } = await fetchNow(renderURL(k, 1920, 1080, f));
      const a = document.createElement("a");
      a.href = src;
      a.download = `${activeRuns()[k].run}_${S.mode}_f${f}.jpg`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(src), 5000);
    } catch (e) { toast(`Chụp ảnh lỗi: ${e.message}`); }
  }
}
function openExport() {
  $("exview").value = S.mode === "overview" ? "overview" : "follow";
  $("exstart").value = S.frames[0];
  $("exend").value = S.frames[S.frames.length - 1];
  $("exstatus").textContent = "";
  $("modal").classList.remove("hidden");
}
async function startExport() {
  try {
    const body = { runs: S.active, view: $("exview").value, start: +$("exstart").value, end: +$("exend").value };
    const data = await (await api("/api/export", { method: "POST", body: JSON.stringify(body) })).json();
    $("exstatus").textContent = `Đang xuất ${data.path}`;
    pollExport();
  } catch (e) { $("exstatus").textContent = `Lỗi: ${e.message}`; }
}
async function pollExport() {
  const st = await (await api("/api/export_status")).json();
  if (st.state === "running") {
    $("exstatus").textContent = `Đang xuất… ${Math.round(Date.now() / 1000 - st.started)} s`;
    setTimeout(pollExport, 2000);
  } else if (st.state === "done") {
    const codec = st.h264 ? "" : `<br>Mã hóa mp4v (mở bằng VLC). Cài ffmpeg trong WSL để có H.264: <code>sudo apt install ffmpeg</code>`;
    $("exstatus").innerHTML = `Xong sau ${st.seconds} s · <a href="/api/file?path=${encodeURIComponent(st.path)}">tải video</a><br>${st.windows_path}${codec}`;
    show();
  } else if (st.state === "error") {
    $("exstatus").textContent = `Lỗi: ${st.message}`;
  }
}
function refreshAll() {
  computeFrames();
  buildStage();
  if (S.panel === "records") buildRecords();
  setMode(S.mode);
}

// ------------------------------------------------------------------ wiring

function wire() {
  document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  document.querySelectorAll("#rail button").forEach((b) => (b.onclick = () => openPanel(b.dataset.open)));
  $("dclose").onclick = () => openPanel(S.panel);
  $("color").onchange = (e) => { S.color = e.target.value; show(); };
  document.querySelectorAll("[data-layer]").forEach((cb) => (cb.onchange = () => {
    if (cb.checked) S.layers.add(cb.dataset.layer); else S.layers.delete(cb.dataset.layer);
    show();
  }));
  $("ceiling").onchange = (e) => { S.ceiling = e.target.checked; S.ceilingTouched = true; show(); };
  $("trail").onchange = (e) => { S.trail = +e.target.value; S.trailTouched = true; show(); };
  $("thumbs").onchange = (e) => { S.thumbs = e.target.checked; layoutStage(); show(); };
  $("mini").onchange = (e) => { S.mini = e.target.checked; layoutStage(); show(); };
  const orbitVal = () => ($("orbitval").textContent = `${S.orbitSec} s`);
  const ptVal = () => ($("ptval").textContent = `×${S.ptScale}`);
  $("orbitsec").oninput = (e) => { S.orbitSec = +e.target.value; orbitVal(); };
  $("ptscale").oninput = (e) => { S.ptScale = +e.target.value; ptVal(); };
  orbitVal(); ptVal();
  $("crisp").onchange = (e) => { S.crisp = e.target.checked; show(); };
  $("errlog").onchange = (e) => { S.errLog = e.target.checked; drawCharts(); };
  $("speed").onchange = (e) => (S.speed = +e.target.value);
  $("play").onclick = play;
  $("prev").onclick = () => seek(S.idx - 1);
  $("next").onclick = () => seek(S.idx + 1);
  $("first").onclick = () => seek(0);
  $("last").onclick = () => seek(S.frames.length - 1);
  $("openbtn").onclick = openRecord;
  $("openpath").onkeydown = (e) => { if (e.key === "Enter") openRecord(); };
  $("shot").onclick = screenshot;
  $("exportbtn").onclick = openExport;
  $("excancel").onclick = () => $("modal").classList.add("hidden");
  $("exgo").onclick = startExport;
  chartClick("error");
  chartClick("gauss");
  wireTimeline();
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); play(); }
    else if (e.key === "ArrowLeft") seek(S.idx - 1);
    else if (e.key === "ArrowRight") seek(S.idx + 1);
    else if (e.key === "Home") seek(0);
    else if (e.key === "End") seek(S.frames.length - 1);
    else if (e.key === "Escape") { if (S.panel) openPanel(S.panel); $("modal").classList.add("hidden"); }
    else if (["1", "2", "3", "4"].includes(e.key)) setMode(["follow", "overview", "orbit", "free"][+e.key - 1]);
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { layoutStage(); show(); }, 150); });
}

(async function init() {
  wire();
  try { await loadRuns(); } catch (e) { toast(`Không kết nối được server: ${e.message}`); return; }
  discover();
  refreshAll();
  requestAnimationFrame(tick);
})();
