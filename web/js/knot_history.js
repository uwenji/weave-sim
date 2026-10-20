// What the Monitor's knot charts plot: a few numbers per crossing, recorded after every chunk of iterations straight
// from the solver's arrays (no geometry snapshots, no Results build). A fixed memory budget; when it is full every
// other record is dropped and the recording stride doubles, so a long run stays covered from its first iteration at a
// coarser spacing. Stability checks are kept alongside (one entry per check).
import { fkey } from './topology.js';
import { IxEff, IyEff, sectionArea } from './params.js';

/** per-crossing channels of a record (Float32, K × NCH) */
export const KNOT_CH = ['Fn', 'Ft', 'slip', 'state', 'gap', 'pressure', 'util', 'theta', 'phiA', 'phiB', 'lambda', 'dent', 'M_rot', 'arcA', 'arcB', 'ux', 'uy', 'uz', 'sigma'];
const CH = Object.fromEntries(KNOT_CH.map((k, i) => [k, i]));
export const NCH = KNOT_CH.length;

export class KnotHistory {
  constructor({ budget = 12e6, maxRecords = 1500 } = {}) { this.budget = budget; this.maxRecords = maxRecords; this.sv = null; this.clear(); }
  clear() { this.recs = []; this.stride = 1; this.lastIt = -Infinity; this.stab = []; this.version = (this.version || 0) + 1; }

  /** bind to a (new or reset) model: the two frame nodes of every crossing, section moduli per node, arc length per segment */
  bind(sv, topo, scene) {
    this.clear();
    this.sv = sv; this.topo = topo; this.scene = scene;
    const st = sv.st, n = st.n, m = st.m, K = sv.contact ? st.K : 0;
    this.K = K;
    this.cap = Math.max(64, Math.min(this.maxRecords, Math.floor(this.budget / Math.max(4 * K * NCH + 64, 1))));
    this.kA = new Int32Array(K); this.kB = new Int32Array(K); this.overSide = new Uint8Array(K);
    for (let c = 0; c < K; c++) {
      const kn = topo.knots.get(st.knotEdges[c]);
      this.kA[c] = st.nodeFrameIds.get(fkey(...kn.a)); this.kB[c] = st.nodeFrameIds.get(fkey(...kn.b));
      this.overSide[c] = topo.isOver(kn.a[0], kn.a[1]) ? 0 : 1;
    }
    // stress σ = |N|/A + |Mx|/Wx + |My|/Wy at a node: the section of its strand, the joint at the node
    this.Ainv = new Float64Array(n); this.Wx = new Float64Array(n); this.Wy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const sec = scene.group(scene.strandGroup(st.nodeStrand[i])).section;
      this.Ainv[i] = 1 / Math.max(sectionArea(sec), 1e-300);
      this.Wx[i] = IxEff(sec) / Math.max(0.5 * sec.height, 1e-300); this.Wy[i] = IyEff(sec) / Math.max(0.5 * sec.width, 1e-300);
    }
    this.nodeJoint = new Int32Array(n).fill(-1);
    for (let j = 0; j < st.p; j++) this.nodeJoint[st.jNode[j]] = j;
    this.nodeSegs = Array.from({ length: n }, () => []);
    for (let j = 0; j < m; j++) { this.nodeSegs[st.segI0[j]].push(j); this.nodeSegs[st.segI1[j]].push(j); }
    this.arc0 = new Float64Array(m);
    for (let s = 0; s < st.S; s++) { let a = 0; for (let j = st.segOffset[s]; j < st.segOffset[s + 1]; j++) { this.arc0[j] = a; a += st.l0[j]; } }
  }

  get length() { return this.recs.length; }
  get last() { return this.recs[this.recs.length - 1] || null; }

  /** record the model's current state (every `stride` iterations unless forced); probe: {node, P0, axis} or null.
   *  Returns true when a record was taken. */
  record(force = false, probe = null) {
    const sv = this.sv; if (!sv || !this.K) return false;
    const st = sv.st, it = st.iteration, K = this.K, c0 = sv.contact;
    if (this.recs.length && it < this.last.it) this.clear();                           // the model was reset under us
    if (this.recs.length && it === this.last.it) { if (!force) return false; this.recs.pop(); }
    else if (!force && it - this.lastIt < this.stride) return false;
    const kn = new Float32Array(K * NCH), X = st.X, X0 = st.X0, bN = sv.beam.N, bM = sv.beam.M;
    const sigma = (i) => {
      let N = 0; const sg = this.nodeSegs[i]; for (const j of sg) N += bN[j]; N /= Math.max(sg.length, 1);
      const j = this.nodeJoint[i], M1 = j >= 0 ? bM[3 * j] : 0, M2 = j >= 0 ? bM[3 * j + 1] : 0;
      return Math.abs(N) * this.Ainv[i] + Math.abs(M1) / this.Wx[i] + Math.abs(M2) / this.Wy[i];
    };
    const seat = sv.scene.contact.seat_law !== 'off', rot = !!sv.scene.contact.rot_friction, aug = !!sv.scene.contact.augmented;
    for (let c = 0; c < K; c++) {
      const o = c * NCH;
      kn[o + CH.Fn] = c0.Fn[c]; kn[o + CH.Ft] = c0.Ft[c]; kn[o + CH.slip] = st.slip[c]; kn[o + CH.state] = c0.state[c]; kn[o + CH.gap] = c0.gap[c];
      kn[o + CH.pressure] = c0.pressure[c]; kn[o + CH.util] = c0.util[c]; kn[o + CH.theta] = c0.theta[c];
      kn[o + CH.phiA] = c0.phi[2 * c]; kn[o + CH.phiB] = c0.phi[2 * c + 1];
      kn[o + CH.lambda] = aug && st.lam ? st.lam[c] : NaN;
      kn[o + CH.dent] = seat ? c0.dent[2 * c] + c0.dent[2 * c + 1] + c0.dentE[c] : NaN;
      kn[o + CH.M_rot] = rot ? c0.Mrot[c] : NaN;
      for (let side = 0; side < 2; side++) {
        const u = st.u[2 * c + side], nseg = c0.nseg[2 * c + side], off = c0.segOff[2 * c + side];
        const i = Math.min(Math.max(Math.floor(u), 0), Math.max(nseg - 1, 0)), t = Math.min(Math.max(u - i, 0), 1);
        kn[o + (side ? CH.arcB : CH.arcA)] = this.arc0[off + i] + t * st.l0[off + i];
      }
      const a = this.kA[c], b = this.kB[c];
      for (let q = 0; q < 3; q++) kn[o + CH.ux + q] = 0.5 * (X[3 * a + q] - X0[3 * a + q] + X[3 * b + q] - X0[3 * b + q]);
      kn[o + CH.sigma] = Math.max(sigma(a), sigma(b));
    }
    let pt = NaN;
    if (probe) { const i = probe.node, P0 = probe.P0, d = probe.axis; pt = (X[3 * i] - P0[0]) * d[0] + (X[3 * i + 1] - P0[1]) * d[1] + (X[3 * i + 2] - P0[2]) * d[2]; }
    this.recs.push({ it, kn, pt });
    this.lastIt = it;
    if (this.recs.length > this.cap) {                                                  // drop every other, keep the newest
      const keep = this.recs.filter((_, i) => i % 2 === 0); if (keep[keep.length - 1] !== this.last) keep.push(this.last);
      this.recs = keep; this.stride *= 2;
    }
    this.version++;
    return true;
  }

  /** a stability check of the current state (stability.js) */
  addStability(it, s) { this.stab = this.stab.filter(e => e.it !== it); this.stab.push({ it, lam: s.lam, neg: s.neg, ms: s.ms, ndof: s.ndof }); this.version++; }

  /** index of the record nearest to iteration `it` */
  nearest(it) {
    const S = this.recs; if (!S.length) return -1;
    let lo = 0, hi = S.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (S[mid].it <= it) lo = mid; else hi = mid; }
    return Math.abs(S[hi].it - it) < Math.abs(S[lo].it - it) ? hi : lo;
  }
  /** channel `ch` of crossing c over the records */
  channel(c, ch) {
    const idx = CH[ch], out = new Float64Array(this.recs.length);
    for (let i = 0; i < this.recs.length; i++) out[i] = this.recs[i].kn[c * NCH + idx];
    return out;
  }
  iterations() { return Float64Array.from(this.recs, r => r.it); }
  probeTravel() { return Float64Array.from(this.recs, r => r.pt); }
}
