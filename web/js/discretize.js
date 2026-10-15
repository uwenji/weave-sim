// Weave discretisation: joint placement along the laths by simulated annealing (port of weavesim/discretize.py), and
// the joint helpers of the viewer (weavesim/app/joints.py): joint points, cut rectangles, piece index, joint-zone masks.
//
// Every strand longer than the stock length is cut into pieces; the joints (arc length s along the strand) are the
// variables; consecutive joints at most `max_length − joint_length` apart (kept feasible by the repair step of the
// neighbour function); the objective is a weighted sum of components normalised by their value at the initial
// equal-division placement:
//   sum_curve Σ|κ| at the joints · max_curve max|κ| · min_dist −(min distance between joints of different strands) ·
//   uniformity_range longest − shortest piece · uniformity_sum Σ|piece − mean| · structural Σ|layer| at the joints /
//   max|layer| (keeps joints out of the high-moment zones of the solved state) · clearance Σ max(0, d_c − distance to
//   the nearest crossing), d_c = lath width + joint length / 2.
// Neighbour: a random joint of a random strand moves by up to ±10 % of the local piece length; neighbours are shifted
// if the max-length constraint would break; rejected if the shift runs past the strand ends. `mll` neighbours per
// temperature step, geometric cooling, Metropolis acceptance. The random numbers come from xoshiro128** (rng.js), not
// numpy: the same seed gives a different (equally valid) annealing path than the Python program.
import { Rng } from './rng.js';
import { arcLengths, discreteCurvature, interp, searchsorted, nodeValues, strandSpan, pointAt, fmtG } from './fields.js';

export const OBJECTIVES = ['sum_curve', 'max_curve', 'min_dist', 'uniformity_range', 'uniformity_sum', 'structural', 'clearance'];
export const OBJECTIVE_LABELS = {
  sum_curve: 'min Σ curvature at joints', max_curve: 'min max curvature at joints', min_dist: 'max min distance between joints',
  uniformity_range: 'min longest − shortest piece', uniformity_sum: 'min Σ |piece − mean length|', structural: 'avoid high result values at joints',
  clearance: 'keep joints clear of crossings',
};
export const OBJECTIVE_SHORT = { sum_curve: 'Σκ', max_curve: 'maxκ', min_dist: 'dist', uniformity_range: 'range', uniformity_sum: 'Σ|Δl|', structural: 'struct', clearance: 'clear' };
export const STRUCTURAL_LAYERS = ['Mx', 'My', 'T', 'N', 'disp'];
export const DISCRETIZE_DEFAULTS = {
  max_length: 100.0, joint_length: 0.0, extra_joints: 0,
  weights: { sum_curve: 1.0, max_curve: 0.0, min_dist: 1.0, uniformity_range: 0.0, uniformity_sum: 1.0, structural: 0.0, clearance: 1.0 },
  structural_layer: 'Mx', iterations: 500, mll: 100, initial_temp: 0.05, cooling: 0.99, seed: 1,
};

/** DiscretizeParams.from_dict: defaults, weights merged, unknown keys dropped */
export function discretizeParams(d) {
  const o = structuredClone(DISCRETIZE_DEFAULTS);
  for (const k of Object.keys(DISCRETIZE_DEFAULTS)) if (d && d[k] != null && k !== 'weights') o[k] = d[k];
  if (d && d.weights) for (const [k, v] of Object.entries(d.weights)) o.weights[k] = +v;
  for (const k of ['max_length', 'joint_length', 'initial_temp', 'cooling']) o[k] = +o[k];
  for (const k of ['extra_joints', 'iterations', 'mll', 'seed']) o[k] = Math.round(+o[k]);
  return o;
}
export const effectiveMax = (p) => Math.max(p.max_length - p.joint_length, 1e-9);

// ------------------------------------------------------------------------------------------------ strand geometry
/** StrandGeometry: nodes, arc length, |κ|, the crossings' arc lengths, |structural value| at the nodes */
export class StrandGeometry {
  constructor(P, knotNodes, layerVals = null) {
    this.P = P; const n = P.length / 3; this.n = n;
    this.S = arcLengths(P, 0, n);
    this.kappa = discreteCurvature(P, 0, n);
    const kn = Int32Array.from(knotNodes).sort();
    this.knot_s = Float64Array.from(kn, i => this.S[i]);
    this.layer = layerVals ? Float64Array.from(layerVals, Math.abs) : null;
  }
  get length() { return this.S[this.n - 1]; }
  at(s, arr) { return interp(s, this.S, arr); }
  point(s) { return pointAt(this.P, 0, this.S, s); }
}

/** node rows of the crossings of every strand (frame_to_knot frames on that strand) */
export function strandKnotRows(topo) {
  const out = topo.strands.map(() => []);
  for (const fk of topo.frameToKnot.keys()) { const s = Math.floor(fk / 1048576); if (out[s]) out[s].push(topo.knotNode.get(fk)); }
  for (const r of out) r.sort((a, b) => a - b);
  return out;
}

/** geometry_from_app: one StrandGeometry per strand on positions X (the solved state), with a result layer for the
 *  structural objective (segment layers averaged to the nodes, disp as |u|) */
export function geometryFromModel(topo, st, X = st.X, results = null, layer = 'Mx') {
  const lay = results ? nodeValues(st, results, layer) : null, rows = strandKnotRows(topo);
  const out = [];
  for (let s = 0; s < st.S; s++) {
    const { i0, n } = strandSpan(st, s);
    out.push(new StrandGeometry(X.slice(3 * i0, 3 * (i0 + n)), rows[s], lay ? lay.slice(i0, i0 + n) : null));
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ annealing
export class Discretizer {
  constructor(strands, params, width = 0.0) {
    this.p = discretizeParams(params);
    this.maxL = effectiveMax(this.p);
    this.clearance = +width + 0.5 * this.p.joint_length;
    this.rng = new Rng(this.p.seed);
    this.geo = []; this.strandIds = [];
    strands.forEach((g, i) => { if (g.length > this.maxL + 1e-9) { this.geo.push(g); this.strandIds.push(i); } });
    this.L = this.geo.map(g => g.length);
    this.nJoints = this.L.map(l => Math.max(Math.ceil(l / this.maxL) - 1 + this.p.extra_joints, 0));
    let layMax = 0; for (const g of this.geo) if (g.layer && g.layer.length) for (const v of g.layer) layMax = Math.max(layMax, v);
    if (layMax > 0) for (const g of this.geo) if (g.layer) g.layer = g.layer.map(v => v / layMax);
    this.nEval = 0; this.accepted = 0;
    this.x = this.initial();
    this._setNorm();
    this._initCache();
    this.e = this._energyOf(this._components(this.stats));
    this.xbest = this.x.map(a => a.slice()); this.ebest = this.e;
    this.T = this.p.initial_temp;
    this.iteration = 0;
    this.historyBest = [this.ebest]; this.historyCurrent = [this.e];
  }
  _setNorm() {
    this.components0 = this.components(this.initial());
    this.norm = {};
    for (const [k, v] of Object.entries(this.components0)) this.norm[k] = k === 'min_dist' ? [-2.0 * this.maxL, v] : [0.0, v];
  }
  /** new weights / structural layer on a running annealer (the Python panel): renormalise, re-evaluate */
  reweight(params) {
    const p = discretizeParams(params);
    this.p.weights = p.weights; this.p.structural_layer = p.structural_layer;
    this._setNorm();
    this.e = this.energy(this.x); this.ebest = this.energy(this.xbest);
  }

  initial() { return this.L.map((l, i) => { const n = this.nJoints[i], step = l / (n + 1); return Float64Array.from({ length: n }, (_, k) => (k + 1) * step); }); }
  pieces(x = this.x) { return x.map((xi, i) => { const j = [0, ...xi, this.L[i]]; return Float64Array.from({ length: j.length - 1 }, (_, k) => j[k + 1] - j[k]); }); }
  feasible(xi, l) {
    let prev = 0;
    for (let k = 0; k <= xi.length; k++) { const v = k < xi.length ? xi[k] : l, d = v - prev; if (!(d <= this.maxL + 1e-9 && d > 1e-9)) return false; prev = v; }
    return true;
  }

  /** the objective components of a placement, computed from scratch (as discretize.py) */
  components(x) {
    const w = this.p.weights, c = {};
    const on = (k) => (w[k] || 0) > 0;
    if (on('sum_curve') || on('max_curve')) {
      let sum = 0, max = 0;
      x.forEach((xi, i) => { const g = this.geo[i]; for (const s of xi) { const k = g.at(s, g.kappa); sum += k; if (k > max) max = k; } });
      if (on('sum_curve')) c.sum_curve = sum;
      if (on('max_curve')) c.max_curve = max;
    }
    if (on('min_dist')) {
      const pts = [], sid = [];
      x.forEach((xi, i) => { for (const s of xi) { pts.push(...this.geo[i].point(s)); sid.push(i); } });
      const J = sid.length;
      if (J > 1) {
        let d2 = Infinity;
        for (let a = 0; a < J; a++) for (let b = a + 1; b < J; b++) {
          if (sid[a] === sid[b]) continue;
          const dx = pts[3 * a] - pts[3 * b], dy = pts[3 * a + 1] - pts[3 * b + 1], dz = pts[3 * a + 2] - pts[3 * b + 2], q = dx * dx + dy * dy + dz * dz;
          if (q < d2) d2 = q;
        }
        c.min_dist = Number.isFinite(d2) ? -Math.sqrt(d2) : 0.0;
      } else c.min_dist = 0.0;
    }
    if (on('uniformity_range') || on('uniformity_sum')) {
      const lens = []; for (const p of this.pieces(x)) lens.push(...p);
      if (on('uniformity_range')) c.uniformity_range = Math.max(...lens) - Math.min(...lens);
      if (on('uniformity_sum')) { const mean = lens.reduce((a, b) => a + b, 0) / lens.length; c.uniformity_sum = lens.reduce((a, v) => a + Math.abs(v - mean), 0); }
    }
    if (on('structural')) {
      let sum = 0; x.forEach((xi, i) => { const g = this.geo[i]; if (g.layer) for (const s of xi) sum += g.at(s, g.layer); });
      c.structural = sum;
    }
    if (on('clearance') && this.clearance > 0) {
      let pen = 0;
      x.forEach((xi, i) => {
        const ks = this.geo[i].knot_s; if (!ks.length || !xi.length) return;
        for (const s of xi) { let d = Infinity; for (const k of ks) d = Math.min(d, Math.abs(s - k)); pen += Math.max(0, this.clearance - d); }
      });
      c.clearance = pen;
    }
    return c;
  }

  normalised(comps) {
    const out = {};
    for (const [k, v] of Object.entries(comps)) {
      const [lo, hi] = this.norm[k] || [0.0, v];
      if (k === 'clearance') out[k] = this.clearance > 0 ? v / this.clearance : 0.0;
      else if (Math.abs(hi - lo) < 1e-12) out[k] = hi === 0 ? 0.0 : v / hi;
      else out[k] = (v - lo) / (hi - lo);
    }
    return out;
  }
  _energyOf(comps) {
    this.nEval++;
    const n = this.normalised(comps);
    let e = 0; for (const [k, v] of Object.entries(n)) e += (this.p.weights[k] || 0) * v;
    return e;
  }
  energy(x) { return this._energyOf(this.components(x)); }

  // --- incremental evaluation: a neighbour moves the joints of one strand, so only that strand's terms and its row of
  // --- the strand-pair minimum distances are recomputed (same objective as components(), summed per strand)
  _strandStats(i, xi) {
    const g = this.geo[i], n = xi.length, pts = new Float64Array(3 * n), lens = new Float64Array(n + 1);
    let sumK = 0, maxK = 0, struct = 0, clear = 0, prev = 0;
    for (let q = 0; q < n; q++) {
      const s = xi[q], k = g.at(s, g.kappa);
      sumK += k; if (k > maxK) maxK = k;
      const p = g.point(s); pts[3 * q] = p[0]; pts[3 * q + 1] = p[1]; pts[3 * q + 2] = p[2];
      if (g.layer) struct += g.at(s, g.layer);
      if (g.knot_s.length) { let d = Infinity; for (const ks of g.knot_s) d = Math.min(d, Math.abs(s - ks)); clear += Math.max(0, this.clearance - d); }
      lens[q] = s - prev; prev = s;
    }
    lens[n] = this.L[i] - prev;
    let lmin = Infinity, lmax = -Infinity, lsum = 0; for (const v of lens) { if (v < lmin) lmin = v; if (v > lmax) lmax = v; lsum += v; }
    return { n, pts, lens, lmin, lmax, lsum, sumK, maxK, struct, clear, layer: !!g.layer };
  }
  _pairRow(i, pts) {
    const G = this.geo.length, row = new Float64Array(G).fill(Infinity), ni = pts.length / 3;
    for (let t = 0; t < G; t++) {
      if (t === i) continue;
      const Q = this.stats[t].pts, nt = Q.length / 3;
      let m = Infinity;
      for (let a = 0; a < ni; a++) for (let b = 0; b < nt; b++) {
        const dx = pts[3 * a] - Q[3 * b], dy = pts[3 * a + 1] - Q[3 * b + 1], dz = pts[3 * a + 2] - Q[3 * b + 2], q = dx * dx + dy * dy + dz * dz;
        if (q < m) m = q;
      }
      row[t] = m;
    }
    return row;
  }
  _initCache() {
    const G = this.geo.length;
    this.stats = this.x.map((xi, i) => this._strandStats(i, xi));
    this.pair = new Float64Array(G * G).fill(Infinity);
    for (let i = 0; i < G; i++) { const row = this._pairRow(i, this.stats[i].pts); for (let t = 0; t < G; t++) this.pair[i * G + t] = row[t]; }
  }
  _components(stats, si = -1, row = null) {
    const w = this.p.weights, c = {}, on = (k) => (w[k] || 0) > 0, G = stats.length;
    if (on('sum_curve') || on('max_curve')) {
      let sum = 0, max = 0; for (const t of stats) { sum += t.sumK; if (t.maxK > max) max = t.maxK; }
      if (on('sum_curve')) c.sum_curve = sum;
      if (on('max_curve')) c.max_curve = max;
    }
    if (on('min_dist')) {
      let m = Infinity;
      for (let a = 0; a < G; a++) for (let b = a + 1; b < G; b++) { const v = a === si ? row[b] : b === si ? row[a] : this.pair[a * G + b]; if (v < m) m = v; }
      c.min_dist = Number.isFinite(m) ? -Math.sqrt(m) : 0.0;
    }
    if (on('uniformity_range') || on('uniformity_sum')) {
      let lo = Infinity, hi = -Infinity, sum = 0, cnt = 0;
      for (const t of stats) { if (t.lmin < lo) lo = t.lmin; if (t.lmax > hi) hi = t.lmax; sum += t.lsum; cnt += t.lens.length; }
      if (on('uniformity_range')) c.uniformity_range = hi - lo;
      if (on('uniformity_sum')) { const mean = sum / cnt; let d = 0; for (const t of stats) for (const v of t.lens) d += Math.abs(v - mean); c.uniformity_sum = d; }
    }
    if (on('structural')) { let sum = 0; for (const t of stats) if (t.layer) sum += t.struct; c.structural = sum; }
    if (on('clearance') && this.clearance > 0) { let pen = 0; for (const t of stats) pen += t.clear; c.clearance = pen; }
    return c;
  }

  /** a feasible neighbour of x (one strand's joints moved), or null; the moved strand is this.lastStrand */
  neighbour(x) {
    const cand = []; this.nJoints.forEach((n, i) => { if (n > 0) cand.push(i); });
    if (!cand.length) return null;
    const s = cand[this.rng.int(cand.length)], xi = x[s], l = this.L[s];
    const j = this.rng.int(xi.length);
    const J = [0, ...xi, l];
    const local = 0.5 * (J[j + 2] - J[j]);
    const delta = (-0.1 + 0.2 * this.rng.uniform()) * 2.0 * local;
    J[j + 1] += delta;
    if (delta < 0) {
      for (let i = j + 1; i < J.length - 1; i++) if (J[i + 1] - J[i] > this.maxL) J[i + 1] = J[i] + this.maxL;
      if (J[J.length - 1] < l - 1e-9 || J[J.length - 1] > l + 1e-9) return null;
    } else {
      for (let i = j + 1; i > 0; i--) if (J[i] - J[i - 1] > this.maxL) J[i - 1] = J[i] - this.maxL;
      if (Math.abs(J[0]) > 1e-9) return null;
    }
    const nw = Float64Array.from(J.slice(1, -1));
    if (!this.feasible(nw, l)) return null;
    const y = x.slice(); y[s] = nw; this.lastStrand = s;
    return y;
  }

  /** n temperature steps (mll neighbours each) */
  step(n = 1) {
    const G = this.geo.length;
    for (let it = 0; it < n; it++) {
      if (this.iteration >= this.p.iterations) return;
      for (let q = 0; q < Math.max(this.p.mll | 0, 1); q++) {
        const y = this.neighbour(this.x); if (!y) continue;
        const s = this.lastStrand, ts = this._strandStats(s, y[s]), row = this._pairRow(s, ts.pts);
        const stats = this.stats.slice(); stats[s] = ts;
        const e = this._energyOf(this._components(stats, s, row));
        if (e < this.e || this.rng.uniform() < Math.exp(-(e - this.e) / Math.max(this.T, 1e-12))) {
          this.x = y; this.e = e; this.accepted++; this.stats = stats;
          for (let t = 0; t < G; t++) { this.pair[s * G + t] = row[t]; this.pair[t * G + s] = row[t]; }
          if (e < this.ebest) { this.xbest = y.map(a => a.slice()); this.ebest = e; }
        }
      }
      this.T *= this.p.cooling; this.iteration++;
      this.historyBest.push(this.ebest); this.historyCurrent.push(this.e);
    }
  }
  get done() { return this.iteration >= this.p.iterations; }

  /** Discretisation: joints {strand: [s…]} of the best placement, components and energies */
  result() {
    const x = this.xbest, joints = {};
    this.strandIds.forEach((sid, i) => { joints[sid] = Array.from(x[i]); });
    const comps = this.components(x);
    return {
      joints, components_initial: this.components0, components_final: comps, normalised_initial: this.normalised(this.components0),
      normalised_final: this.normalised(comps), energy_initial: this.historyBest[0], energy_final: this.ebest,
      iterations: this.iteration, evaluations: this.nEval, params: structuredClone(this.p),
    };
  }
}

// ------------------------------------------------------------------------------------------------ joints (joints.py)
/** joints of the scene → [{s, sj, p: [x,y,z], kappa, i (segment start node row)}] on positions X */
export function jointPoints(scene, st, X = st.X) {
  const out = [];
  for (const [key, list] of Object.entries(scene.joints || {})) {
    const s = +key; if (!(s >= 0 && s < st.S) || !list || !list.length) continue;
    const { i0, n } = strandSpan(st, s), S = arcLengths(X, i0, n), k = discreteCurvature(X, i0, n);
    for (const v of list) {
      const sj = Math.min(Math.max(+v, 0), S[n - 1]);
      const i = Math.min(Math.max(searchsorted(S, sj) - 1, 0), Math.max(n - 2, 0));
      out.push({ s, sj, p: pointAt(X, i0, S, sj), kappa: interp(sj, S, k), node: i0 + i });
    }
  }
  return out;
}

/** cut rectangles (w × h × 1.1 in the section plane of the segment's start node) → Float32Array of line segments */
export function jointCuts(joints, R, sizeOf) {
  const out = new Float32Array(joints.length * 24);
  joints.forEach((j, q) => {
    const [w, h] = sizeOf(j.s), r = 9 * j.node, ex = [R[r], R[r + 1], R[r + 2]], ey = [R[r + 3], R[r + 4], R[r + 5]];
    const c = (sx, sy) => [0, 1, 2].map(k => j.p[k] + sx * 0.55 * w * ex[k] + sy * 0.55 * h * ey[k]);
    const C = [c(1, 1), c(-1, 1), c(-1, -1), c(1, -1)];
    for (let e = 0; e < 4; e++) { out.set(C[e], 24 * q + 6 * e); out.set(C[(e + 1) % 4], 24 * q + 6 * e + 3); }
  });
  return out;
}

/** piece index along each strand per node (np.searchsorted(joints, S, 'right')) → Int32Array(st.n) */
export function pieceIndex(scene, st, X = st.X) {
  const out = new Int32Array(st.n);
  for (let s = 0; s < st.S; s++) {
    const list = (scene.joints || {})[s]; if (!list || !list.length) continue;
    const js = Float64Array.from(list).sort(), { i0, n } = strandSpan(st, s), S = arcLengths(X, i0, n);
    for (let i = 0; i < n; i++) out[i0 + i] = searchsorted(js, S[i], 'right');
  }
  return out;
}

/** joints.zone_mask: nodes of strand s within ±nFrames crossings (mean crossing spacing) of its joints; null if none */
export function zoneMask(scene, topo, st, s, nFrames, X = st.X, rows = null) {
  const list = (scene.joints || {})[s]; if (!list || !list.length) return null;
  const { i0, n } = strandSpan(st, s), S = arcLengths(X, i0, n);
  const knots = (rows || strandKnotRows(topo))[s];
  let spacing;
  if (knots.length >= 2) { let sum = 0; for (let q = 1; q < knots.length; q++) sum += S[knots[q]] - S[knots[q - 1]]; spacing = sum / (knots.length - 1); }
  else spacing = S[n - 1] / Math.max(n - 1, 1) * 4;
  const m = new Uint8Array(n);
  for (const sj of list) for (let i = 0; i < n; i++) if (Math.abs(S[i] - sj) <= nFrames * spacing) m[i] = 1;
  return m;
}

/** joints.csv rows as the Python panel writes them */
export function jointsCSV(joints) {
  const f = (v) => v.toFixed(4);
  return ['strand,s,x,y,z,curvature', ...joints.map(j => `${j.s},${f(j.sj)},${f(j.p[0])},${f(j.p[1])},${f(j.p[2])},${fmtG(j.kappa, 6)}`)].join('\r\n') + '\r\n';
}
