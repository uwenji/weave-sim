// Kagome topology from a triangle mesh (port of weavesim/topology.py, itself a port of the Grasshopper WeaveInit v3).
// The kagome is the medial graph of the triangulation: one knot (crossing) per mesh edge, three lath segments per face.
// A strand entering a face across edge e = (p, q), having pivoted around q in the previous face, exits across the
// edge that contains p; pivots alternate, so strands are straight on a regular lattice.
// Per strand: knots (mesh edge ids in order; frame id (s, k) = k-th knot along s), polyline nodes (subdivided,
// pre-offset ±h/2 along the crossing normal with a cosine profile), node frames (X = width, Y = normal, Z = tangent).
import { PySet } from './pyset.js';

export const fkey = (s, k) => s * 1048576 + k;                     // frame id (s, k) → one number
export const fsplit = (key) => [Math.floor(key / 1048576), key % 1048576];

// --- small vector helpers (f64, numpy-compatible formulas) ------------------------------------------------------
const norm = (x, y, z) => Math.sqrt(x * x + y * y + z * z);
function normalize3(v, eps = 1e-12) { let n = norm(v[0], v[1], v[2]); if (n < eps) n = 1.0; return [v[0] / n, v[1] / n, v[2] / n]; }
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; };

/** R with Z = tangent, Y = hint made orthogonal to Z, X = Y × Z; returned as 9 numbers [X | Y | Z] */
export function frameFromTangentNormal(t, hint) {
  const z = normalize3(t);
  let y = sub3(hint, scl3(z, dot3(hint, z)));
  if (norm(y[0], y[1], y[2]) < 1e-9) { const a = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; y = sub3(a, scl3(z, dot3(a, z))); }
  y = normalize3(y);
  const x = cross3(y, z);
  return [x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]];
}

export class Topology {
  constructor(mesh, params, thickness, thicknessOf = null) {
    this.mesh = mesh; this.params = params; this.thickness = thickness; this.thicknessOf = thicknessOf;
    this.strands = [];                // {index, edges[], segments[[face, corner]], closed, family}
    this.knots = new Map();           // edge id → {edge, a:[s,k], b:[s,k], over, conflict, flipped, isBoundary}
    this.frameToKnot = new Map();     // fkey → edge id
    this.polylines = [];              // per strand Float64Array(3 N_s)
    this.nodeFrames = [];             // per strand Float64Array(9 N_s): [X | Y | Z] per node
    this.knotNode = new Map();        // fkey → node row in the strand's polyline
    this.parityConflicts = 0; this.familyCount = 0;
  }
  get nStrands() { return this.strands.length; }
  get nKnots() { return this.knots.size; }
  hasFrame(s, k) { return this.frameToKnot.has(fkey(s, k)); }
  knotOf(s, k) { return this.knots.get(this.frameToKnot.get(fkey(s, k))); }
  isOver(s, k) { const kn = this.knotOf(s, k); return kn ? kn.over === s : false; }
  frameOrigin(s, k) { const r = this.knotNode.get(fkey(s, k)), P = this.polylines[s]; return [P[3 * r], P[3 * r + 1], P[3 * r + 2]]; }
  startFrames() { return this.strands.filter(s => !s.closed).map(s => [s.index, 0]); }
  endFrames() { return this.strands.filter(s => !s.closed).map(s => [s.index, s.edges.length - 1]); }
  boundaryFrames() { const out = []; for (const kn of this.knots.values()) if (kn.isBoundary) out.push(kn.a, kn.b); return out; }
  strandFrames(s) { return this.strands[s].edges.map((_, k) => [s, k]); }
  /** mid-crossing points (mean of the two offset frame origins) and the edge ids, in sorted edge order */
  knotPoints() {
    const edges = [...this.knots.keys()].sort((a, b) => a - b);
    const pts = new Float64Array(3 * edges.length);
    edges.forEach((e, i) => { const kn = this.knots.get(e), pa = this.frameOrigin(...kn.a), pb = this.frameOrigin(...kn.b); for (let c = 0; c < 3; c++) pts[3 * i + c] = 0.5 * (pa[c] + pb[c]); });
    return { pts, edges };
  }
}

export function buildTopology(mesh, params, thickness, thicknessOf = null) {
  const topo = new Topology(mesh, params, thickness, thicknessOf);
  traceStrands(topo);
  assignFamilies(topo);
  assignParity(topo);
  applyFlips(topo);
  buildPolylines(topo);
  consistentFrames(topo);
  return topo;
}

// --- 1. tracing ---------------------------------------------------------------------------------------------------
function traceStrands(topo) {
  const mesh = topo.mesh, E = mesh.E;
  const visited = new Set();                              // face * 2^21 + corner vertex
  const vkey = (f, v) => f * 2097152 + v;
  const strands = [];
  const finish = (edges, segments, closed) => {
    if (closed && edges.length > segments.length) edges = edges.slice(0, segments.length);
    return { index: strands.length, edges, segments, closed, family: 0 };
  };
  const trace = (face, entry, pivot) => {
    const edges = [entry], segments = [];
    const first = [face, entry, pivot];
    for (;;) {
      const q = pivot;
      const p = E[2 * entry] === q ? E[2 * entry + 1] : E[2 * entry];
      const key = vkey(face, p);
      if (visited.has(key)) {
        const closed = segments.length > 0 && face === first[0] && entry === first[1] && pivot === first[2];
        return segments.length ? finish(edges, segments, closed) : null;
      }
      visited.add(key);
      segments.push([face, p]);
      const g = mesh.faceEdgeContaining(face, p, entry);
      const nxt = mesh.otherFace(g, face);
      pivot = p; entry = g;
      if (nxt < 0) { edges.push(g); return finish(edges, segments, false); }
      face = nxt;
      if (face === first[0] && entry === first[1] && pivot === first[2]) return finish(edges, segments, true);
      edges.push(g);
    }
  };
  // open strands start at boundary edges (two per boundary edge, one per endpoint as pivot)
  for (let e = 0; e < mesh.nE; e++) {
    if (!mesh.isBoundaryEdge[e]) continue;
    const f = mesh.EF[2 * e] >= 0 ? mesh.EF[2 * e] : mesh.EF[2 * e + 1];
    for (const pivot of [E[2 * e], E[2 * e + 1]]) { const st = trace(f, e, pivot); if (st) strands.push(st); }
  }
  // closed loops (closed meshes / interior cycles)
  for (let f = 0; f < mesh.nF; f++) {
    for (let c = 0; c < 3; c++) {
      const corner = mesh.F[3 * f + c];
      if (visited.has(vkey(f, corner))) continue;
      let entry = -1;
      for (let i = 0; i < 3; i++) { const e = mesh.FE[3 * f + i]; if (mesh.edgeHas(e, corner)) { entry = e; break; } }
      const otherV = E[2 * entry] === corner ? E[2 * entry + 1] : E[2 * entry];
      const st = trace(f, entry, otherV);
      if (st) strands.push(st);
    }
  }
  topo.strands = strands;
  const passes = new Map();
  for (const s of strands) s.edges.forEach((e, k) => { if (!passes.has(e)) passes.set(e, []); passes.get(e).push([s.index, k]); });
  for (const [e, fids] of passes) {
    if (fids.length !== 2) throw new Error(`edge ${e} visited ${fids.length} times; tracing invariant broken`);
    const [a, b] = fids;
    topo.knots.set(e, { edge: e, a, b, over: a[0], conflict: false, flipped: false, isBoundary: !!mesh.isBoundaryEdge[e] });
    topo.frameToKnot.set(fkey(a[0], a[1]), e);
    topo.frameToKnot.set(fkey(b[0], b[1]), e);
  }
}

// --- 2. families: greedy BFS colouring, strands sharing a face get different colours -------------------------------
function assignFamilies(topo) {
  const faceStrands = new Map();
  for (const s of topo.strands) for (const [f] of s.segments) { if (!faceStrands.has(f)) faceStrands.set(f, new PySet()); faceStrands.get(f).add(s.index); }
  const adj = new Map(topo.strands.map(s => [s.index, new PySet()]));
  for (const members of faceStrands.values()) {
    for (const a of members) { const one = new PySet(); one.add(a); adj.get(a).update(members.difference(one)); }
  }
  const colour = new Map();
  for (const root of adj.keys()) {
    if (colour.has(root)) continue;
    const dq = [root]; let head = 0;
    while (head < dq.length) {
      const s = dq[head++];
      if (colour.has(s)) continue;
      const used = new Set(); for (const n of adj.get(s)) if (colour.has(n)) used.add(colour.get(n));
      let c = 0; while (used.has(c)) c++;
      colour.set(s, c);
      for (const n of adj.get(s)) if (!colour.has(n)) dq.push(n);
    }
  }
  for (const s of topo.strands) s.family = colour.get(s.index);
  topo.familyCount = colour.size ? Math.max(...colour.values()) + 1 : 0;
}

// --- 3. over/under parity: over_s(k) = (k + φ_s) % 2 == 0; a knot (s,i)-(t,j) needs φ_s + φ_t = i + j + 1 (mod 2) --
function assignParity(topo) {
  const n = topo.nStrands, phi = new Array(n).fill(-1), adj = Array.from({ length: n }, () => []);
  const mod2 = (x) => ((x % 2) + 2) % 2;
  for (const kn of topo.knots.values()) {
    const [s, i] = kn.a, [t, j] = kn.b, req = mod2(i + j + 1);
    adj[s].push([t, req]); adj[t].push([s, req]);
  }
  for (let root = 0; root < n; root++) {
    if (phi[root] >= 0) continue;
    phi[root] = root === 0 ? mod2(topo.params.seed_strand_phase | 0) : 0;
    const dq = [root]; let head = 0;
    while (head < dq.length) {
      const s = dq[head++];
      for (const [t, req] of adj[s]) { const want = mod2(req - phi[s]); if (phi[t] < 0) { phi[t] = want; dq.push(t); } }
    }
  }
  let conflicts = 0;
  for (const kn of topo.knots.values()) {
    const [s, i] = kn.a, [t, j] = kn.b;
    const overS = mod2(i + phi[s]) === 0, overT = mod2(j + phi[t]) === 0;
    if (overS === overT) { kn.conflict = true; conflicts++; kn.over = s; }
    else kn.over = overS ? s : t;
  }
  topo.parityConflicts = conflicts;
}

function applyFlips(topo) {
  for (const f of topo.params.flip_knots || []) {
    const e = topo.frameToKnot.get(fkey(f[0] | 0, f[1] | 0));
    if (e === undefined) continue;
    const kn = topo.knots.get(e);
    kn.over = kn.over === kn.a[0] ? kn.b[0] : kn.a[0];
    kn.flipped = !kn.flipped;
  }
}

// --- 4/5. subdivision, pre-offset, frames ---------------------------------------------------------------------------
function buildPolylines(topo) {
  const mesh = topo.mesh, p = topo.params;
  const mid = (e) => [mesh.edgeMidpoints[3 * e], mesh.edgeMidpoints[3 * e + 1], mesh.edgeMidpoints[3 * e + 2]];
  const en = (e) => [mesh.edgeNormals[3 * e], mesh.edgeNormals[3 * e + 1], mesh.edgeNormals[3 * e + 2]];
  const h = p.offset_mode === 'half_height' ? topo.thickness : 0.0;
  const hOf = topo.thicknessOf;
  const sub = Math.max(1, p.subdivisions | 0);
  for (const s of topo.strands) {
    const nk = s.edges.length;
    const Nn = s.edges.map(en);
    const sign = s.edges.map(e => (topo.knots.get(e).over === s.index ? 1.0 : -1.0));
    const he = s.edges.map(e => (hOf && p.offset_mode === 'half_height' ? 0.5 * (+hOf(topo.knots.get(e).a[0]) + +hOf(topo.knots.get(e).b[0])) : h));
    const K = s.edges.map((e, k) => add3(mid(e), scl3(Nn[k], 0.5 * he[k] * sign[k])));
    const nodes = [], normals = [], knotRows = [];
    if (!s.closed && p.overhang > 0 && nk > 1) {
      const t0 = normalize3(sub3(K[1], K[0]));
      const d01 = sub3(K[1], K[0]), seg = norm(...d01) / sub;
      const m = Math.max(1, roundHalfEven(p.overhang / Math.max(seg, 1e-9)));
      for (let i = 0; i < m; i++) { const d = p.overhang * (1 - i / m); nodes.push(sub3(K[0], scl3(t0, d))); normals.push(Nn[0]); }
    }
    const K0 = s.edges.map(mid);
    const off = sign.map(sg => (0.5 * h) * sg);
    const step = 1.0 / sub;
    for (let k = 0; k < nk; k++) {
      knotRows.push(nodes.length);
      nodes.push(K[k]); normals.push(Nn[k]);
      const nxt = k + 1 < nk ? k + 1 : (s.closed ? 0 : null);
      if (nxt !== null) {
        const prv = (s.closed || k > 0) ? ((k - 1) % nk + nk) % nk : null;
        const nx2 = (s.closed || nxt + 1 < nk) ? (nxt + 1) % nk : null;
        for (let i = 1; i < sub; i++) {
          const t = i * step;
          const base = catmullRom(K0, prv, k, nxt, nx2, t);
          const nrm = normalize3(add3(scl3(Nn[k], 1 - t), scl3(Nn[nxt], t)));
          const w = 0.5 * (1.0 + Math.cos(Math.PI * t));
          const o = w * off[k] + (1 - w) * off[nxt];
          nodes.push(add3(base, scl3(nrm, o))); normals.push(nrm);
        }
      }
    }
    if (!s.closed && p.overhang > 0 && nk > 1) {
      const t1 = normalize3(sub3(K[nk - 1], K[nk - 2]));
      const seg = norm(...sub3(K[nk - 1], K[nk - 2])) / sub;
      const m = Math.max(1, roundHalfEven(p.overhang / Math.max(seg, 1e-9)));
      for (let i = 1; i <= m; i++) { nodes.push(add3(K[nk - 1], scl3(t1, p.overhang * i / m))); normals.push(Nn[nk - 1]); }
    }
    const P = new Float64Array(3 * nodes.length);
    nodes.forEach((q, i) => { P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2]; });
    topo.polylines.push(P);
    topo.nodeFrames.push(nodeFrames(P, normals, s.closed));
    knotRows.forEach((row, k) => topo.knotNode.set(fkey(s.index, k), row));
  }
}

/** centripetal Catmull-Rom point between K[i1] and K[i2]; i0 / i3 may be null (end spans) */
function catmullRom(K, i0, i1, i2, i3, t) {
  const P1 = K[i1], P2 = K[i2];
  const P0 = i0 !== null ? K[i0] : sub3(scl3(P1, 2), P2);
  const P3 = i3 !== null ? K[i3] : sub3(scl3(P2, 2), P1);
  const tj = (ti, a, b) => ti + Math.sqrt(Math.max(norm(...sub3(b, a)), 1e-12));
  const t0 = 0.0, t1 = tj(t0, P0, P1), t2 = tj(t1, P1, P2), t3 = tj(t2, P2, P3);
  const tt = t1 + t * (t2 - t1);
  const lerp = (a, b, ta, tb) => (tb - ta < 1e-12 ? a : scl3(add3(scl3(a, tb - tt), scl3(b, tt - ta)), 1 / (tb - ta)));
  const A1 = lerp(P0, P1, t0, t1), A2 = lerp(P1, P2, t1, t2), A3 = lerp(P2, P3, t2, t3);
  const B1 = lerp(A1, A2, t0, t2), B2 = lerp(A2, A3, t1, t3);
  return lerp(B1, B2, t1, t2);
}

/** node tangents (mean of the adjacent unit chords) */
function nodeTangents(P, closed) {
  const n = P.length / 3, pt = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  if (n === 1) return [[0, 0, 1]];
  const T = [];
  if (closed) { for (let i = 0; i < n; i++) T.push(normalize3(sub3(pt((i + 1) % n), pt((i - 1 + n) % n)))); return T; }
  T.push(sub3(pt(1), pt(0)));
  for (let i = 1; i < n - 1; i++) T.push(add3(normalize3(sub3(pt(i + 1), pt(i))), normalize3(sub3(pt(i), pt(i - 1)))));
  T.push(sub3(pt(n - 1), pt(n - 2)));
  return T.map(t => normalize3(t));
}

function nodeFrames(P, hints, closed) {
  const n = P.length / 3, out = new Float64Array(9 * n);
  if (n === 1) { out.set([1, 0, 0, 0, 1, 0, 0, 0, 1]); return out; }
  const T = nodeTangents(P, closed);
  for (let i = 0; i < n; i++) out.set(frameFromTangentNormal(T[i], hints[i]), 9 * i);
  return out;
}

function interp(x, xp, fp) {
  if (x <= xp[0]) return fp[0];
  const last = xp.length - 1;
  if (x >= xp[last]) return fp[last];
  let lo = 0, hi = last;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

/** knot normals from the crossing t_A × t_B of the smooth centrelines (sign-aligned with the mesh normal),
 *  interpolated between knots, frames rebuilt: the initial contact geometry is then self-consistent */
function consistentFrames(topo) {
  const mesh = topo.mesh;
  if (!mesh || !topo.knots.size) return;
  const tang = topo.polylines.map((P, s) => nodeTangents(P, topo.strands[s].closed));
  const hints = topo.polylines.map(P => Array.from({ length: P.length / 3 }, () => null));
  for (const [e, kn] of topo.knots) {
    const [sA] = kn.a, [sB] = kn.b;
    const rA = topo.knotNode.get(fkey(...kn.a)), rB = topo.knotNode.get(fkey(...kn.b));
    let n = cross3(tang[sA][rA], tang[sB][rB]);
    const nn = norm(...n), ref = [mesh.edgeNormals[3 * e], mesh.edgeNormals[3 * e + 1], mesh.edgeNormals[3 * e + 2]];
    n = nn < 1e-9 ? ref : scl3(n, 1 / nn);
    if (dot3(n, ref) < 0) n = scl3(n, -1);
    hints[sA][rA] = n; hints[sB][rB] = n;
  }
  topo.strands.forEach((st, s) => {
    const H = hints[s], known = [];
    H.forEach((v, i) => { if (v) known.push(i); });
    if (!known.length) return;
    let Hf;
    if (known.length === 1) Hf = H.map(() => H[known[0]]);
    else {
      const comp = [0, 1, 2].map(c => known.map(i => H[i][c]));
      Hf = H.map((_, i) => {
        if (i < known[0]) return H[known[0]];
        if (i > known[known.length - 1]) return H[known[known.length - 1]];
        return [0, 1, 2].map(c => interp(i, known, comp[c]));
      });
    }
    Hf = Hf.map(v => { const l = Math.max(norm(...v), 1e-12); return [v[0] / l, v[1] / l, v[2] / l]; });
    topo.nodeFrames[s] = nodeFrames(topo.polylines[s], Hf, st.closed);
  });
}

// --- stand-alone laths (tests, calibration specimens): no mesh, no knots, frames at both ends --------------------
export function topologyFromPolylines(polylines, normals = null, thickness = 30.0, params = null, knotRows = null) {
  params = params || { subdivisions: 4, overhang: 0.0, offset_mode: 'half_height', flip_knots: [], seed_strand_phase: 0 };
  const topo = new Topology(null, params, thickness);
  polylines.forEach((P0, s) => {
    const P = Float64Array.from(P0.flat ? P0.flat() : P0), n = P.length / 3;
    const rows = knotRows ? knotRows[s] : [0, n - 1];
    let nh = normals ? normals[s] : null;
    if (!nh) nh = Array.from({ length: n }, () => [0, 0, 1]);
    else if (!Array.isArray(nh[0])) nh = Array.from({ length: n }, () => nh);
    topo.strands.push({ index: s, edges: rows.map((_, k) => -(s * 1000 + k + 1)), segments: [], closed: false, family: 0 });
    topo.polylines.push(P);
    topo.nodeFrames.push(nodeFrames(P, nh, false));
    rows.forEach((row, k) => { topo.knotNode.set(fkey(s, k), row); topo.frameToKnot.set(fkey(s, k), -(s * 1000 + k + 1)); });
  });
  return topo;
}

/** the frame/knot JSON the Grasshopper WeaveInit emits (weavesim export-k2), for diffing */
export function toExportDict(topo) {
  return {
    n_strands: topo.nStrands, n_knots: topo.nKnots,
    strands: topo.strands.map(s => ({
      index: s.index, family: s.family, closed: s.closed, edges: s.edges,
      polyline: Array.from({ length: topo.polylines[s.index].length / 3 }, (_, i) => Array.from(topo.polylines[s.index].slice(3 * i, 3 * i + 3))),
      knot_nodes: s.edges.map((_, k) => topo.knotNode.get(fkey(s.index, k))),
    })),
    knots: [...topo.knots.values()].map(kn => ({ edge: kn.edge, a: kn.a, b: kn.b, over: kn.over, conflict: kn.conflict, flipped: kn.flipped, boundary: kn.isBoundary })),
    parity_conflicts: topo.parityConflicts,
  };
}
