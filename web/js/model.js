// Solver model shared by the CPU (f64) and GPU (f32) solvers: flat node / segment / joint / contact arrays built from a
// Topology (port of weavesim/solver/state.py), selectors → frame ids (selection.py), anchors and loads (goals/loads.py)
// and the per-strand section stiffness of the lath groups (Solver.strand_stiffness).
import { fkey } from './topology.js';
import { frameFromTangentNormal } from './topology.js';
import { fromFrame, toFrame, qnormalize } from './quat.js';
import { sectionStiffness, sectionArea, beamFactors, beamActivations } from './params.js';

// ------------------------------------------------------------------------------------------------ state
export function buildState(topo) {
  const S = topo.nStrands;
  const counts = topo.polylines.map(P => P.length / 3);
  const nodeOffset = new Int32Array(S + 1);
  for (let s = 0; s < S; s++) nodeOffset[s + 1] = nodeOffset[s] + counts[s];
  const n = nodeOffset[S];
  const X = new Float64Array(3 * n);
  topo.polylines.forEach((P, s) => X.set(P, 3 * nodeOffset[s]));
  const nodeStrand = new Int32Array(n);
  for (let s = 0; s < S; s++) nodeStrand.fill(s, nodeOffset[s], nodeOffset[s + 1]);

  const i0 = [], i1 = [], segStrand = [], segCounts = [], Q = [], l0 = [], j0 = [], j1 = [], jn = [];
  const q = new Float64Array(4);
  topo.strands.forEach((st, s) => {
    const off = nodeOffset[s], nn = counts[s];
    const nseg = st.closed ? nn : nn - 1;
    segCounts.push(nseg);
    const base = segCounts.slice(0, -1).reduce((a, b) => a + b, 0);
    const P = topo.polylines[s], R = topo.nodeFrames[s];
    for (let k = 0; k < nseg; k++) {
      const a = k, b = (k + 1) % nn;
      i0.push(off + a); i1.push(off + b); segStrand.push(s);
      const d = [P[3 * b] - P[3 * a], P[3 * b + 1] - P[3 * a + 1], P[3 * b + 2] - P[3 * a + 2]];
      l0.push(Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
      const hint = [R[9 * a + 3] + R[9 * b + 3], R[9 * a + 4] + R[9 * b + 4], R[9 * a + 5] + R[9 * b + 5]];
      fromFrame(frameFromTangentNormal(d, hint), q);
      Q.push(q[0], q[1], q[2], q[3]);
    }
    if (st.closed) for (let k = 0; k < nseg; k++) { j0.push(base + k); j1.push(base + (k + 1) % nseg); jn.push(off + (k + 1) % nn); }
    else for (let k = 0; k < Math.max(nseg - 1, 0); k++) { j0.push(base + k); j1.push(base + k + 1); jn.push(off + 1 + k); }
  });
  const segOffset = new Int32Array(S + 1);
  for (let s = 0; s < S; s++) segOffset[s + 1] = segOffset[s] + segCounts[s];
  const m = segOffset[S], p = j0.length;
  const L0 = Float64Array.from(l0);
  const jSeg0 = Int32Array.from(j0), jSeg1 = Int32Array.from(j1);
  const jLen = new Float64Array(p);
  for (let j = 0; j < p; j++) jLen[j] = 0.5 * (L0[jSeg0[j]] + L0[jSeg1[j]]);

  // contacts: knots in sorted edge order, u = node row of the knot (global parameter, strand-local), n_prev = A → B
  const knotEdges = Int32Array.from([...topo.knots.keys()].sort((a, b) => a - b));
  const K = knotEdges.length;
  const u = new Float64Array(2 * K), nPrev = new Float64Array(3 * K);
  const nodeFrameIds = new Map();
  for (const [fk, row] of topo.knotNode) { const s = Math.floor(fk / 1048576); nodeFrameIds.set(fk, nodeOffset[s] + row); }
  knotEdges.forEach((e, c) => {
    const kn = topo.knots.get(e);
    [kn.a, kn.b].forEach((fid, side) => {
      const [s] = fid, row = topo.knotNode.get(fkey(...fid)), nseg = segOffset[s + 1] - segOffset[s];
      u[2 * c + side] = Math.min(row, nseg);
    });
    const ia = nodeFrameIds.get(fkey(...kn.a)), ib = nodeFrameIds.get(fkey(...kn.b));
    const d = [X[3 * ib] - X[3 * ia], X[3 * ib + 1] - X[3 * ia + 1], X[3 * ib + 2] - X[3 * ia + 2]];
    const nd = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (let k = 0; k < 3; k++) nPrev[3 * c + k] = nd > 1e-12 ? d[k] / nd : topo.mesh.edgeNormals[3 * e + k];
  });
  return {
    S, n, m, p, K, X, V: new Float64Array(3 * n), X0: X.slice(), nodeOffset, nodeStrand,
    segI0: Int32Array.from(i0), segI1: Int32Array.from(i1), segStrand: Int32Array.from(segStrand), segOffset,
    Q: Float64Array.from(Q), Wt: new Float64Array(m), l0: L0,
    jSeg0, jSeg1, jNode: Int32Array.from(jn), jLen, Omega0: new Float64Array(3 * p),
    knotEdges, u, nPrev, slip: new Float64Array(K), stuck: new Uint8Array(K).fill(1), lam: new Float64Array(K),
    nodeFrameIds, iteration: 0, residual: Infinity,
    history: { kinetic: [], strain: [], bending: [], torsion: [], stretch: [], contact: [], residual: [] },
  };
}

export function cloneState(st) {
  const out = {};
  for (const [k, v] of Object.entries(st)) {
    if (ArrayBuffer.isView(v)) out[k] = v.slice();
    else if (v instanceof Map) out[k] = new Map(v);
    else if (k === 'history') out[k] = Object.fromEntries(Object.entries(v).map(([a, b]) => [a, b.slice()]));
    else out[k] = v;
  }
  return out;
}

/** hemisphere-consistent mean of the adjacent segment frames, per node → Float64Array(4n) */
export function nodeQuats(st, out = new Float64Array(4 * st.n)) {
  const ref = new Int32Array(st.n).fill(-1);
  for (let j = 0; j < st.m; j++) ref[st.segI1[j]] = j;
  for (let j = 0; j < st.m; j++) ref[st.segI0[j]] = j;
  out.fill(0);
  const Q = st.Q;
  for (const idx of [st.segI0, st.segI1]) {
    for (let j = 0; j < st.m; j++) {
      const i = idx[j], r = ref[i];
      const dot = Q[4 * j] * Q[4 * r] + Q[4 * j + 1] * Q[4 * r + 1] + Q[4 * j + 2] * Q[4 * r + 2] + Q[4 * j + 3] * Q[4 * r + 3];
      const sg = dot < 0 ? -1 : 1;
      for (let c = 0; c < 4; c++) out[4 * i + c] += sg * Q[4 * j + c];
    }
  }
  for (let i = 0; i < st.n; i++) qnormalize(out, 4 * i);
  return out;
}
/** node frames [X|Y|Z] per node → Float64Array(9n) */
export function nodeFrames(st) {
  const nq = nodeQuats(st), R = new Float64Array(9 * st.n), tmp = new Float64Array(9);
  for (let i = 0; i < st.n; i++) { toFrame(nq, 4 * i, tmp); R.set(tmp, 9 * i); }
  return R;
}

// ------------------------------------------------------------------------------------------------ selection
/** selector → list of frame ids [s, k] (selection.py) */
export function resolveSelector(sel, topo) {
  const t = sel.type;
  if (t === 'frame') { const [s, k] = sel.id; return topo.hasFrame(s, k) ? [[s | 0, k | 0]] : []; }
  if (t === 'all_start') return topo.startFrames();
  if (t === 'all_end') return topo.endFrames();
  if (t === 'all_ends') return [...topo.startFrames(), ...topo.endFrames()];
  if (t === 'strand') return topo.strandFrames(sel.strand | 0);
  if (t === 'boundary') return topo.boundaryFrames();
  if (t === 'knots') return (sel.ids || []).map(i => [i[0] | 0, i[1] | 0]).filter(([s, k]) => topo.hasFrame(s, k));
  if (t === 'region') return regionFrames(sel, topo);
  throw new Error(`unknown selector type ${t}`);
}

/** frames of the knots within a radius of the seed frame's knot: graph distance on the mesh edges (Dijkstra from the
 *  seed edge's two vertices) — the Python program uses the heat method when potpourri3d is installed */
function regionFrames(sel, topo) {
  const mesh = topo.mesh, [s0, k0] = sel.id, r = +(sel.radius || 0);
  const seedEdge = topo.frameToKnot.get(fkey(s0, k0));
  if (seedEdge === undefined || !mesh) return [];
  const nV = mesh.nV, dist = new Float64Array(nV).fill(Infinity), nbr = Array.from({ length: nV }, () => []);
  for (let e = 0; e < mesh.nE; e++) {
    const a = mesh.E[2 * e], b = mesh.E[2 * e + 1];
    const l = Math.hypot(mesh.V[3 * a] - mesh.V[3 * b], mesh.V[3 * a + 1] - mesh.V[3 * b + 1], mesh.V[3 * a + 2] - mesh.V[3 * b + 2]);
    nbr[a].push([b, l]); nbr[b].push([a, l]);
  }
  const heap = [];
  const push = (v, d) => { if (d < dist[v]) { dist[v] = d; heap.push([d, v]); heap.sort((x, y) => x[0] - y[0]); } };
  push(mesh.E[2 * seedEdge], 0); push(mesh.E[2 * seedEdge + 1], 0);
  while (heap.length) { const [d, v] = heap.shift(); if (d > dist[v]) continue; for (const [w, l] of nbr[v]) push(w, d + l); }
  const out = [];
  for (const [e, kn] of topo.knots) if (0.5 * (dist[mesh.E[2 * e]] + dist[mesh.E[2 * e + 1]]) <= r) out.push(kn.a, kn.b);
  return out;
}

/** union of fixities per frame: Map fkey → [X Y Z rX rY rZ] booleans (insertion order = resolution order) */
export function anchorMasks(anchors, topo) {
  const masks = new Map();
  for (const a of anchors) {
    for (const [s, k] of resolveSelector(a.select, topo)) {
      const key = fkey(s, k), m = masks.get(key) || [false, false, false, false, false, false];
      masks.set(key, m.map((v, i) => v || !!a.fix[i]));
    }
  }
  return masks;
}

// ------------------------------------------------------------------------------------------------ anchors, loads
export class Anchors {
  constructor(scene, topo, st) {
    const n = st.n, m = st.m;
    this.nodeFree = new Uint8Array(3 * n).fill(1);
    this.twistFree = new Uint8Array(m).fill(1);
    this.fixedNodes = [];
    this.clamps = [];                                        // [node, neighbour, segment, [unit dir node → neighbour]]
    const adj = new Map();
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      if (!adj.has(a)) adj.set(a, []); adj.get(a).push([j, b]);
      if (!adj.has(b)) adj.set(b, []); adj.get(b).push([j, a]);
    }
    for (const [key, mask] of anchorMasks(scene.anchors, topo)) {
      const nid = st.nodeFrameIds.get(key);
      if (nid === undefined) continue;
      for (let c = 0; c < 3; c++) if (mask[c]) this.nodeFree[3 * nid + c] = 0;
      if ((mask[0] || mask[1] || mask[2]) && !this.fixedNodes.includes(nid)) this.fixedNodes.push(nid);
      for (const [j, nb] of adj.get(nid) || []) {
        if (mask[5]) this.twistFree[j] = 0;
        if (mask[3] || mask[4]) {
          const t = [st.X[3 * nb] - st.X[3 * nid], st.X[3 * nb + 1] - st.X[3 * nid + 1], st.X[3 * nb + 2] - st.X[3 * nid + 2]];
          const l = Math.max(Math.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]), 1e-12);
          this.clamps.push([nid, nb, j, [t[0] / l, t[1] / l, t[2] / l]]);
        }
      }
    }
    this.reactions = new Float64Array(3 * n);
    this.fixedAny = new Uint8Array(n);
    this.allFree = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const f = this.nodeFree[3 * i] && this.nodeFree[3 * i + 1] && this.nodeFree[3 * i + 2];
      this.allFree[i] = f ? 1 : 0; this.fixedAny[i] = f ? 0 : 1;
    }
  }
}

export class Loads {
  constructor(scene, topo, st) {
    const n = st.n, u = scene.units;
    const trib = new Float64Array(n);
    for (let j = 0; j < st.m; j++) trib[st.segI0[j]] += 0.5 * st.l0[j];
    for (let j = 0; j < st.m; j++) trib[st.segI1[j]] += 0.5 * st.l0[j];
    this.trib = trib;
    this.const = new Float64Array(3 * n);
    const L = scene.loads;
    if (L.gravity.enabled) {
      const g = L.gravity.direction.map(Number), gl = Math.max(Math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), 1e-12);
      for (let c = 0; c < 3; c++) g[c] /= gl;
      const qS = Array.from({ length: st.S }, (_, s) => { const grp = scene.group(scene.strandGroup(s)); return u.selfWeightLineLoad(grp.material.density, sectionArea(grp.section)); });
      for (let i = 0; i < n; i++) { const q = qS[st.nodeStrand[i]] * trib[i]; for (let c = 0; c < 3; c++) this.const[3 * i + c] += q * g[c]; }
    }
    if (L.udl.enabled) {
      const qv = L.udl.q.map(Number);
      if (L.udl.per === 'length') { for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) this.const[3 * i + c] += trib[i] * qv[c]; }
      else {
        const area = topo.mesh ? topo.mesh.area() : 0;
        let total = 0; for (let i = 0; i < n; i++) total += trib[i];
        for (let i = 0; i < n; i++) { const w = trib[i] / Math.max(total, 1e-12) * area; for (let c = 0; c < 3; c++) this.const[3 * i + c] += w * qv[c]; }
      }
    }
    this.pointLocal = [];                                    // [node, [Fx, Fy, Fz] in the node frame]
    for (const pl of L.point) {
      const nid = st.nodeFrameIds.get(fkey(pl.frame[0] | 0, pl.frame[1] | 0));
      if (nid === undefined) continue;
      if (pl.space === 'local') this.pointLocal.push([nid, pl.F.map(Number)]);
      else for (let c = 0; c < 3; c++) this.const[3 * nid + c] += +pl.F[c];
    }
    this.total = new Float64Array(3 * n);
  }
}

// ------------------------------------------------------------------------------------------------ stiffness
/** per-segment EA / EIx / EIy / GJ (section stiffness × beam factors of the strand's lath group) and per-joint
 *  activation (flat, edge, twist) */
export function strandStiffness(scene, st) {
  const per = [], acts = [];
  for (let s = 0; s < st.S; s++) {
    const g = scene.strandGroup(s), grp = scene.group(g);
    const k = sectionStiffness(grp.section, grp.material), f = beamFactors(grp.beam);
    per.push({ EA: k.EA * f.EA, EIx: k.EIx * f.EIx, EIy: k.EIy * f.EIy, GJ: k.GJ * f.GJ });
    acts.push(g === 1 ? beamActivations(scene.beam) : beamActivations(grp.beam));
  }
  const EA = new Float64Array(st.m), EIx = new Float64Array(st.m), EIy = new Float64Array(st.m), GJ = new Float64Array(st.m);
  for (let j = 0; j < st.m; j++) { const q = per[st.segStrand[j]]; EA[j] = q.EA; EIx[j] = q.EIx; EIy[j] = q.EIy; GJ[j] = q.GJ; }
  const act = new Float64Array(3 * st.p);
  for (let j = 0; j < st.p; j++) act.set(acts[st.segStrand[st.jSeg0[j]]], 3 * j);
  return { stiff: { EA, EIx, EIy, GJ }, act };
}
