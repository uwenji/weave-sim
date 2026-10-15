// Rhino .3dm export (port of weavesim/export3dm.py) with rhino3dm.js (vendor/rhino3dm, loaded on first use): model,
// results, diagrams, joints and supports in a layer tree, every object named ("lath 12") and carrying user text
// (strand, group, family, values…) so Grasshopper can rebuild the data tree from the file without GUID bookkeeping; a
// companion <name>.3dm.json lists the object ids with their data.
//
//   WeaveSim
//   ├─ 0 input mesh
//   ├─ 1 model (pre)      polylines · g / nurbs · g / laths · g / knots
//   ├─ 2 result
//   │  ├─ deformed        polylines · g / nurbs · g / laths · g / displacement vectors
//   │  ├─ laths by layer  Mx My T N Fn Ft slip disp   (vertex-coloured lath meshes + legend dot)
//   │  ├─ diagrams        Mx My Vy Vx T N             (sign-coloured strips + hatch)
//   │  ├─ contact forces
//   │  └─ reactions
//   ├─ 3 joints           cuts · g / dots
//   └─ 4 supports & loads
// ("· group n <name>" only when lath groups are enabled). Written as Rhino 7 files, as the Python program does. User
// text values are formatted as Python's str() (True / False, 6.0, 1e-05), so both programs write the same strings.
import { nodeFrames, resolveSelector } from './model.js';
import { nodeValues, strandSpan, arcLengths, discreteCurvature, interp, searchsorted, knotNodePairs, fmtG } from './fields.js';
import { fixLabel } from './scene.js';
import { fkey } from './topology.js';

export const UNITS = { mm: 'Millimeters', cm: 'Centimeters', m: 'Meters', in: 'Inches' };
export const EXPORT_GROUP_COLORS = [[204, 168, 117], [77, 140, 191], [191, 102, 89], [115, 166, 102], [153, 115, 179]];
const SIGNED = new Set(['Mx', 'My', 'T', 'N', 'Ft']);
const SEGMENT_LAYERS = ['Mx', 'My', 'T', 'N'], KNOT_LAYERS = ['Fn', 'Ft', 'slip', 'state', 'pressure', 'util', 'ctype', 'dent', 'M_rot'];
const ALL_LAYERS = [...SEGMENT_LAYERS, ...KNOT_LAYERS, 'disp'];
const DIAGRAM_AXIS = { Mx: 1, Vy: 1, My: 0, Vx: 0, T: 0, N: 1 };
const VIRIDIS = [[68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38], [253, 231, 37]];
const COOLWARM = [[59, 76, 192], [124, 159, 249], [192, 212, 245], [242, 242, 242], [245, 192, 157], [222, 96, 77], [180, 4, 38]];
const POS = [77, 158, 102], NEG = [209, 97, 82];
export const EXPORT_DEFAULTS = { pre: true, result: true, layers: ['Mx', 'My', 'T', 'N', 'Fn', 'Ft', 'slip', 'disp'], diagrams: ['Mx', 'My', 'Vy', 'Vx', 'T', 'N'], joints: true, supports: true, input_mesh: true, nurbs: true, diagram_height: 0.06 };

let rhinoPromise = null;
/** rhino3dm.js (WebAssembly, ~2.6 MB), loaded once on the first export */
export function loadRhino() {
  if (!rhinoPromise) rhinoPromise = import('../vendor/rhino3dm/rhino3dm.module.min.js').then(m => m.default());
  return rhinoPromise;
}

// --- Python str() of the user text values ------------------------------------------------------------------------
const INT_KEYS = new Set(['strand', 'group', 'family', 'knots', 'k', 'knot', 'joint', 'faces']);
/** Python repr of a float: shortest round-trip digits, exponent form below 1e-4 and from 1e16, always a '.0' or exponent */
export function pyFloat(x) {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const ex = Math.abs(x).toExponential(), e = +ex.slice(ex.indexOf('e') + 1), digits = ex.slice(0, ex.indexOf('e')).replace('.', ''), sign = x < 0 ? '-' : '';
  if (e < -4 || e >= 16) return `${sign}${digits[0]}${digits.length > 1 ? '.' + digits.slice(1) : ''}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`;
  if (e >= 0) { const ip = digits.slice(0, e + 1).padEnd(e + 1, '0'), fp = digits.slice(e + 1); return `${sign}${ip}.${fp || '0'}`; }
  return `${sign}0.${'0'.repeat(-e - 1)}${digits}`;
}
const pyStr = (k, v) => (typeof v === 'boolean' ? (v ? 'True' : 'False') : typeof v === 'number' ? (INT_KEYS.has(k) ? String(Math.round(v)) : pyFloat(v)) : String(v));

// --- interpolating NURBS (export3dm.interpolating_nurbs) ------------------------------------------------------------
function findSpan(nb, p, u, U) {
  if (u >= U[nb + 1]) return nb;
  if (u <= U[p]) return p;
  let low = p, high = nb + 1, mid = (low + high) >> 1;
  while (u < U[mid] || u >= U[mid + 1]) { if (u < U[mid]) high = mid; else low = mid; mid = (low + high) >> 1; }
  return mid;
}
function basisFuns(i, u, p, U) {
  const N = new Float64Array(p + 1), left = new Float64Array(p + 1), right = new Float64Array(p + 1);
  N[0] = 1;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j]; right[j] = U[i + j] - u; let saved = 0;
    for (let r = 0; r < j; r++) { const tmp = N[r] / (right[r + 1] + left[j - r]); N[r] = saved + right[r + 1] * tmp; saved = left[j - r] * tmp; }
    N[j] = saved;
  }
  return N;
}
/** solve A X = B (A n×n, B n×3) by Gaussian elimination with partial pivoting */
function solve(A, B, n) {
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > Math.abs(A[piv * n + c])) piv = r;
    if (piv !== c) { for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; } for (let k = 0; k < 3; k++) { const t = B[3 * c + k]; B[3 * c + k] = B[3 * piv + k]; B[3 * piv + k] = t; } }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d; if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      for (let k = 0; k < 3; k++) B[3 * r + k] -= f * B[3 * c + k];
    }
  }
  for (let r = n - 1; r >= 0; r--) for (let k = 0; k < 3; k++) { let s = B[3 * r + k]; for (let c = r + 1; c < n; c++) s -= A[r * n + c] * B[3 * c + k]; B[3 * r + k] = s / A[r * n + r]; }
  return B;
}
/** clamped B-spline of `degree` through the points (chord-length parameters, de Boor averaged knots, control points from
 *  the collocation system) → {degree, knots (Rhino's n + degree − 1), points [[x,y,z]]} */
export function interpolatingNurbs(P, degree = 3) {
  const n = P.length; if (n < 2) return null;
  degree = Math.min(degree, n - 1);
  const u = new Float64Array(n); for (let i = 1; i < n; i++) u[i] = u[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1], P[i][2] - P[i - 1][2]);
  const tot = Math.max(u[n - 1], 1e-12); for (let i = 0; i < n; i++) u[i] /= tot;
  const K = []; for (let i = 0; i <= degree; i++) K.push(0);
  for (let j = 0; j < n - degree - 1; j++) { let s = 0; for (let q = j + 1; q < j + degree + 1; q++) s += u[q]; K.push(s / degree); }
  for (let i = 0; i <= degree; i++) K.push(1);
  const A = new Float64Array(n * n), B = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const span = findSpan(n - 1, degree, u[i], K), N = basisFuns(span, u[i], degree, K);
    for (let r = 0; r <= degree; r++) A[i * n + span - degree + r] = N[r];
    B[3 * i] = P[i][0]; B[3 * i + 1] = P[i][1]; B[3 * i + 2] = P[i][2];
  }
  solve(A, B, n);
  return { degree, knots: K.slice(1, -1), points: Array.from({ length: n }, (_, i) => [B[3 * i], B[3 * i + 1], B[3 * i + 2]]) };
}

// --- writer: layer / attribute bookkeeping over rhino.File3dm --------------------------------------------------------
class Writer {
  constructor(rhino, units) {
    this.r = rhino; this.f = new rhino.File3dm();
    this.f.settings().modelUnitSystem = rhino.UnitSystem[UNITS[units] || 'Millimeters'];
    this.layers = new Map(); this.index = {};
  }
  layer(path, color = null) {
    if (this.layers.has(path)) return this.layers.get(path);
    const parts = path.split('::');
    let parentId = null;
    for (let i = 1; i <= parts.length; i++) {
      const sub = parts.slice(0, i).join('::');
      if (!this.layers.has(sub)) {
        const L = new this.r.Layer(); L.name = parts[i - 1];
        if (parentId) L.parentLayerId = parentId;
        if (color && i === parts.length) L.color = { r: color[0] | 0, g: color[1] | 0, b: color[2] | 0, a: 255 };
        this.layers.set(sub, this.f.layers().add(L));
      }
      parentId = this.f.layers().findIndex(this.layers.get(sub)).id;
    }
    return this.layers.get(path);
  }
  attrs(layer, name = '', color = null, user = null, layerColor = null) {
    const a = new this.r.ObjectAttributes();
    a.layerIndex = this.layer(layer, layerColor);
    if (name) a.name = name;
    if (color) { a.objectColor = { r: color[0] | 0, g: color[1] | 0, b: color[2] | 0, a: 255 }; a.colorSource = this.r.ObjectColorSource.ColorFromObject; }
    for (const [k, v] of Object.entries(user || {})) a.setUserString(String(k), pyStr(k, v));
    return a;
  }
  add(geom, layer, name, user, color, layerColor) {
    const oid = this.f.objects().add(geom, this.attrs(layer, name, color, user, layerColor));
    this.index[oid] = { layer, name, ...(user || {}) };
    return oid;
  }
  polyline(P, layer, name, user = null, color = null, layerColor = null) { return this.add(new this.r.PolylineCurve(P.map(p => [+p[0], +p[1], +p[2]])), layer, name, user, color, layerColor); }
  nurbs(P, layer, name, user = null, layerColor = null) {
    const c = interpolatingNurbs(P); if (!c) return null;
    const nc = new this.r.NurbsCurve(c.degree, P.length);
    c.knots.forEach((k, i) => nc.knots().set(i, k));
    c.points.forEach((p, i) => nc.points().set(i, [p[0], p[1], p[2], 1.0]));
    return this.add(nc, layer, name, user, null, layerColor);
  }
  mesh(V, F, layer, name, user = null, vertexColors = null, color = null, layerColor = null) {
    const m = new this.r.Mesh();
    for (const v of V) m.vertices().add(+v[0], +v[1], +v[2]);
    for (const f of F) m.faces().addTriFace(f[0], f[1], f[2]);
    if (vertexColors) for (const c of vertexColors) m.vertexColors().add(c[0], c[1], c[2]);
    m.normals().computeNormals();
    return this.add(m, layer, name, user, color, layerColor);
  }
  line(a, b, layer, name, user = null, color = null) { return this.add(new this.r.LineCurve([+a[0], +a[1], +a[2]], [+b[0], +b[1], +b[2]]), layer, name, user, color); }
  point(p, layer, name, user = null, color = null) { return this.add(new this.r.Point([+p[0], +p[1], +p[2]]), layer, name, user, color); }
  dot(p, text, layer, name, user = null) { return this.add(new this.r.TextDot(text, [+p[0], +p[1], +p[2]]), layer, name, user, null); }
  bytes() { const wo = new this.r.File3dmWriteOptions(); wo.version = 7; return this.f.toByteArrayOptions(wo); }
}

// --- helpers ---------------------------------------------------------------------------------------------------------
const cmapRGB = (v, lo, hi, table) => {
  const t = Math.min(Math.max((v - lo) / Math.max(hi - lo, 1e-12), 0), 1) * (table.length - 1);
  const i = Math.min(Math.max(Math.floor(t), 0), table.length - 2), f = t - i;
  return [0, 1, 2].map(c => Math.trunc(table[i][c] * (1 - f) + table[i + 1][c] * f));
};
const rows3 = (A, i0, n) => Array.from({ length: n }, (_, i) => [A[3 * (i0 + i)], A[3 * (i0 + i) + 1], A[3 * (i0 + i) + 2]]);
/** sections.build for one strand: corner rings (+X+Y, −X+Y, −X−Y, +X−Y) and the faces (sides, end caps if open) */
function sectionMesh(P, R, closed, w, h) {
  const n = P.length, V = [], F = [];
  for (let i = 0; i < n; i++) {
    const ex = [0, 1, 2].map(c => R[i][c] * 0.5 * w), ey = [0, 1, 2].map(c => R[i][3 + c] * 0.5 * h), X = P[i];
    V.push(X.map((x, c) => x + ex[c] + ey[c]), X.map((x, c) => x - ex[c] + ey[c]), X.map((x, c) => x - ex[c] - ey[c]), X.map((x, c) => x + ex[c] - ey[c]));
  }
  const nseg = closed ? n : n - 1;
  for (let j = 0; j < nseg; j++) {
    const a = j, b = (j + 1) % n;
    for (const [c0, c1] of [[0, 1], [2, 3], [3, 0], [1, 2]]) { const p0 = 4 * a + c0, p1 = 4 * a + c1, q0 = 4 * b + c0, q1 = 4 * b + c1; F.push([p0, q0, q1], [p0, q1, p1]); }
  }
  if (!closed) for (const node of [0, n - 1]) { const r = 4 * node; F.push([r, r + 1, r + 2], [r, r + 2, r + 3]); }
  return { V, F };
}
const frameRows = (R9, i0, n) => Array.from({ length: n }, (_, i) => R9.subarray(9 * (i0 + i), 9 * (i0 + i) + 9));

/**
 * export: {scene, topo, mesh, st, results, contact, name} → {bytes: Uint8Array (.3dm), index (the .3dm.json object)}.
 * `st` / `results` may be null: only the pre model is written.
 */
export async function exportRhino({ scene, topo, mesh = null, st = null, results = null, contact = null, name = 'weavesim', sceneName = '' }, options = {}, rhino = null) {
  rhino = rhino || await loadRhino();
  const opt = { ...EXPORT_DEFAULTS, ...options };
  const w = new Writer(rhino, scene.units.length);
  const S = topo.nStrands, groups = Array.from({ length: S }, (_, s) => scene.strandGroup(s));
  const hybrid = scene.hybrid.enabled && scene.nGroups > 1;
  const gname = (g) => { if (!hybrid) return ''; const nm = (scene.group(g).name || '').trim(); return ` · group ${g}` + (nm && nm !== `group ${g}` ? ` ${nm}` : ''); };
  const gcol = (g) => EXPORT_GROUP_COLORS[(g - 1) % EXPORT_GROUP_COLORS.length];
  const nodesOf = (s) => topo.polylines[s].length / 3;
  const Xpre = Array.from({ length: S }, (_, s) => rows3(topo.polylines[s], 0, nodesOf(s)));
  const Rpre = Array.from({ length: S }, (_, s) => frameRows(topo.nodeFrames[s], 0, nodesOf(s)));
  let Xdef = null, Rdef = null;
  if (st) { const R9 = nodeFrames(st); Xdef = Array.from({ length: S }, (_, s) => { const { i0, n } = strandSpan(st, s); return rows3(st.X, i0, n); }); Rdef = Array.from({ length: S }, (_, s) => { const { i0, n } = strandSpan(st, s); return frameRows(R9, i0, n); }); }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const grow = (x, y, z) => { lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z); hi[0] = Math.max(hi[0], x); hi[1] = Math.max(hi[1], y); hi[2] = Math.max(hi[2], z); };
  if (mesh) for (let i = 0; i < mesh.V.length; i += 3) grow(mesh.V[i], mesh.V[i + 1], mesh.V[i + 2]);
  else for (const P of Xpre) for (const p of P) grow(...p);
  const size = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1.0;
  const userOf = (s) => { const t = topo.strands[s]; return { strand: s, group: groups[s], family: t.family, closed: !!t.closed, knots: t.edges.length }; };
  const writeModel = (root, X, R, withVectors = false) => {
    for (let s = 0; s < S; s++) {
      const g = groups[s], u = userOf(s), sec = scene.group(g).section;
      w.polyline(X[s], `${root}::polylines${gname(g)}`, `lath ${s}`, u, null, gcol(g));
      if (opt.nurbs) w.nurbs(X[s], `${root}::nurbs${gname(g)}`, `lath ${s}`, u, gcol(g));
      const sm = sectionMesh(X[s], R[s], !!topo.strands[s].closed, sec.width, sec.height);
      w.mesh(sm.V, sm.F, `${root}::laths${gname(g)}`, `lath ${s}`, { ...u, width: +sec.width, height: +sec.height }, null, null, gcol(g));
    }
    if (withVectors) for (const [fk, e] of topo.frameToKnot) {             // one vector per crossing frame (not per node)
      const s = Math.floor(fk / 1048576), k = fk % 1048576, i = topo.knotNode.get(fk), a = Xpre[s][i], b = X[s][i];
      const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      if (d > 1e-9) w.line(a, b, `${root}::displacement vectors`, `frame ${s},${k}`, { strand: s, k, knot: e, disp: d }, [60, 60, 65]);
    }
  };

  // 0 input mesh
  if (opt.input_mesh && mesh) {
    const V = rows3(mesh.V, 0, mesh.V.length / 3), F = Array.from({ length: mesh.F.length / 3 }, (_, f) => [mesh.F[3 * f], mesh.F[3 * f + 1], mesh.F[3 * f + 2]]);
    w.mesh(V, F, 'WeaveSim::0 input mesh', 'input mesh', { faces: F.length }, null, [200, 200, 200]);
  }
  // 1 model (pre)
  if (opt.pre) {
    writeModel('WeaveSim::1 model (pre)', Xpre, Rpre);
    for (const [e, kn] of topo.knots) {
      const p = topo.frameOrigin(kn.a[0], kn.a[1]), over = kn.a[0] === kn.over ? kn.a : kn.b, under = over === kn.a ? kn.b : kn.a;
      w.point(p, 'WeaveSim::1 model (pre)::knots', `knot ${e}`, { knot: e, over: `${over[0]},${over[1]}`, under: `${under[0]},${under[1]}`, boundary: !!kn.isBoundary, flipped: !!kn.flipped }, [60, 60, 65]);
    }
  }
  // 2 result
  if (opt.result && st) {
    writeModel('WeaveSim::2 result::deformed', Xdef, Rdef, true);
    if (results) {
      const pairs = knotNodePairs(st, topo);
      const layerValues = (lay) => (lay === 'disp' ? nodeValues(st, results, 'disp') : SEGMENT_LAYERS.includes(lay) ? results.segment[lay] : results.knot[lay]);
      opt.layers.forEach((lay, li) => {
        if (!ALL_LAYERS.includes(lay)) return;
        const all = layerValues(lay); if (!all || !all.length) return;
        let vmin = Infinity, vmax = -Infinity; for (const v of all) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
        if (SIGNED.has(lay)) { const m = Math.max(Math.abs(vmin), Math.abs(vmax), 1e-12); vmin = -m; vmax = m; }
        const table = SIGNED.has(lay) ? COOLWARM : VIRIDIS, root = `WeaveSim::2 result::laths by layer::${lay}`;
        const nv = nodeValues(st, results, lay, { knotPairs: pairs });
        for (let s = 0; s < S; s++) {
          const { i0, n } = strandSpan(st, s), sec = scene.group(groups[s]).section;
          const sm = sectionMesh(Xdef[s], Rdef[s], !!topo.strands[s].closed, sec.width, sec.height);
          const vals = nv.subarray(i0, i0 + n), vc = []; for (const v of vals) { const c = cmapRGB(v, vmin, vmax, table); vc.push(c, c, c, c); }
          let mn = Infinity, mx = -Infinity; for (const v of vals) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
          w.mesh(sm.V, sm.F, root, `lath ${s}`, { ...userOf(s), layer: lay, min: mn, max: mx }, vc);
        }
        const k = 1 + li * 0.6;
        w.dot([hi[0], hi[1], hi[2] + 0.05 * size * k], `${lay}  [${fmtG(vmin, 4)}, ${fmtG(vmax, 4)}]`, root, `legend ${lay}`, { layer: lay, vmin, vmax });
      });
      // diagrams
      for (const d of opt.diagrams) {
        if (!(d in DIAGRAM_AXIS)) continue;
        const col = DIAGRAM_AXIS[d], shear = d === 'Vx' || d === 'Vy';
        const v = nodeValues(st, results, d === 'Vy' ? 'Mx' : d === 'Vx' ? 'My' : d, { shear });
        let vmax = 0; for (const x of v) vmax = Math.max(vmax, Math.abs(x));
        if (!(vmax > 0)) continue;
        const amp = opt.diagram_height * size / vmax, root = `WeaveSim::2 result::diagrams::${d}`;
        for (let s = 0; s < S; s++) {
          const { i0, n } = strandSpan(st, s); if (n < 2) continue;
          const X = Xdef[s], R = Rdef[s], nv = v.subarray(i0, i0 + n);
          const Q = X.map((p, i) => p.map((x, c) => x + R[i][3 * col + c] * nv[i] * amp));
          const F = []; for (let i = 0; i < n - 1; i++) { const a = i, b = i + 1, c = n + i, e = n + i + 1; F.push([a, b, e], [a, e, c]); }
          const vc = [...nv, ...nv].map(x => (x >= 0 ? POS : NEG));
          let mn = Infinity, mx = -Infinity; for (const x of nv) { mn = Math.min(mn, x); mx = Math.max(mx, x); }
          w.mesh([...X, ...Q], F, root, `lath ${s}`, { ...userOf(s), diagram: d, min: mn, max: mx, scale: amp }, vc);
          const zig = []; for (let i = 0; i < n; i++) zig.push(X[i], Q[i]);         // base → tip → next base …
          w.polyline(zig, `${root}::hatch`, `lath ${s} hatch`, null, [60, 60, 65]);
          w.polyline(Q, `${root}::hatch`, `lath ${s} outline`, null, [60, 60, 65]);
        }
      }
      // contact forces
      if (contact && contact.mid && results.knot.Fn) {
        const Fn = results.knot.Fn; let fmax = 0; for (const v of Fn) fmax = Math.max(fmax, v);
        if (fmax > 0) {
          const L = opt.diagram_height * size / fmax;
          st.knotEdges.forEach((e, k) => {
            if (!(Fn[k] > 0)) return;
            const n = [st.nPrev[3 * k], st.nPrev[3 * k + 1], st.nPrev[3 * k + 2]], p = [contact.mid[3 * k], contact.mid[3 * k + 1], contact.mid[3 * k + 2]];
            w.line(p.map((x, c) => x - 0.5 * L * Fn[k] * n[c]), p.map((x, c) => x + 0.5 * L * Fn[k] * n[c]), 'WeaveSim::2 result::contact forces', `knot ${e}`,
              { knot: e, Fn: Fn[k], Ft: results.knot.Ft ? results.knot.Ft[k] : 0, slip: results.knot.slip ? results.knot.slip[k] : 0, state: results.knot.state && results.knot.state[k] ? 'slip' : 'stick' }, [204, 38, 38]);
          });
        }
      }
      // reactions
      if (results.reactions && results.reactions.size) {
        const mags = new Map(); for (const [fk, v] of results.reactions) mags.set(fk, Math.hypot(v[0], v[1], v[2]));
        const mmax = Math.max(...mags.values()) || 1.0, L = opt.diagram_height * size / mmax;
        for (const [fk, v] of results.reactions) {
          const s = Math.floor(fk / 1048576), k = fk % 1048576, i = st.nodeFrameIds.get(fk), p = [st.X[3 * i], st.X[3 * i + 1], st.X[3 * i + 2]], F = [v[0], v[1], v[2]];
          w.line(p.map((x, c) => x - L * F[c]), p, 'WeaveSim::2 result::reactions', `frame ${s},${k}`, { strand: s, k, Rx: F[0], Ry: F[1], Rz: F[2], R: mags.get(fk) }, [179, 26, 26]);
        }
      }
    }
  }
  // 3 joints
  if (opt.joints && scene.joints && Object.keys(scene.joints).length) {
    const X = Xdef || Xpre, R = Rdef || Rpre;
    let j = 0;
    for (const key of Object.keys(scene.joints).map(Number).sort((a, b) => a - b)) {
      const s = key; if (s >= S) continue;
      const P = X[s], n = P.length, flat = Float64Array.from(P.flat()), Sarc = arcLengths(flat, 0, n), kap = discreteCurvature(flat, 0, n), sec = scene.group(groups[s]).section;
      for (const v of scene.joints[key]) {
        const sj = Math.min(Math.max(+v, 0), Sarc[n - 1]);
        const p = [0, 1, 2].map(c => interp(sj, Sarc, P.map(q => q[c])));
        const i = Math.min(Math.max(searchsorted(Sarc, sj) - 1, 0), Math.max(n - 2, 0)), ex = R[s][i].slice(0, 3), ey = R[s][i].slice(3, 6);
        const corner = (a, b) => p.map((x, c) => x + a * 0.55 * sec.width * ex[c] + b * 0.55 * sec.height * ey[c]);
        const corners = [corner(1, 1), corner(-1, 1), corner(-1, -1), corner(1, -1)];
        const k = interp(sj, Sarc, kap), u = { joint: j, strand: s, s: sj, curvature: k, group: groups[s] };
        w.polyline([...corners, corners[0]], `WeaveSim::3 joints::cuts${gname(groups[s])}`, `joint ${j}`, u, [40, 40, 45]);
        w.dot(p, `J${String(j).padStart(2, '0')}  lath ${s}  s ${sj.toFixed(1)}  κ ${fmtG(k, 3)}`, 'WeaveSim::3 joints::dots', `joint ${j}`, u);
        j++;
      }
    }
  }
  // 4 supports & loads
  if (opt.supports) {
    for (const a of scene.anchors) {
      let fids = []; try { fids = resolveSelector(a.select, topo); } catch (e) { fids = []; }
      for (const f of fids) w.dot(topo.frameOrigin(f[0], f[1]), fixLabel(a.fix), 'WeaveSim::4 supports & loads::anchors', `anchor ${f[0]},${f[1]}`, { strand: f[0], k: f[1], fix: fixLabel(a.fix), name: a.name || '' });
    }
    const mag = (pl) => Math.hypot(...pl.F);
    const fmax = Math.max(0, ...scene.loads.point.map(mag)) || 1.0;
    scene.loads.point.forEach((pl, i) => {
      const f = [pl.frame[0] | 0, pl.frame[1] | 0]; if (!topo.hasFrame(f[0], f[1])) return;
      const p = topo.frameOrigin(f[0], f[1]);
      let F = pl.F.map(Number);
      if (pl.space === 'local') { const r = topo.knotNode.get(fkey(f[0], f[1])), R9 = topo.nodeFrames[f[0]].subarray(9 * r, 9 * r + 9); F = [0, 1, 2].map(c => R9[c] * F[0] + R9[3 + c] * F[1] + R9[6 + c] * F[2]); }
      const L = 0.08 * size / fmax, tail = p.map((x, c) => x - L * F[c]);
      w.line(tail, p, 'WeaveSim::4 supports & loads::point loads', `load ${i}`, { strand: f[0], k: f[1], F: mag(pl), space: pl.space }, [204, 38, 38]);
      w.dot(tail, `${fmtG(mag(pl), 4)} ${scene.units.force}`, 'WeaveSim::4 supports & loads::point loads', `load ${i} label`, null);
    });
  }
  const bytes = w.bytes();
  const groupsDict = {}; for (let g = 1; g <= scene.nGroups; g++) groupsDict[g] = { name: scene.group(g).name, section: scene.group(g).section, material: scene.group(g).material };
  const index = { file: `${name}.3dm`, units: scene.units.toJSON(), scene: sceneName || '', hybrid, groups: groupsDict, strand_groups: groups, objects: w.index };
  return { bytes, index, objects: Object.keys(w.index).length };
}
