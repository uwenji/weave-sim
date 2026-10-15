// Dynamic relaxation in f64 on the CPU — the reference implementation (and the fallback when there is no WebGPU).
// A line-by-line port of weavesim/solver (core.py, goals/beam.py, goals/contact.py, goals/loads.py), plus the
// augmented-Lagrangian gap described in the WeaveSim README (contact.augmented / contact.al_every).
//
// One iteration:  F, T = Σ goal forces (nodes) and torques (segment frames); T splits into twist (along the chord)
// and a node couple (T × t)/l; anchors zero fixed DOFs; v += F/m, ω += T∥/I (Δt = 1, m = mass_scale·Σk); x += v;
// frames rotate by ω about their chord, then re-align Z to the new chord; Barnes kinetic damping rolls back to the
// kinetic-energy peak and restarts from rest. Convergence: max |F| over free nodes / reference force ≤ tol.
import { buildState, cloneState, nodeQuats, Anchors, Loads, strandStiffness } from './model.js';
import { toFrame, fromFrameFast, qexp, qmul, qnormalize, rotBetween } from './quat.js';
import { fkey } from './topology.js';

export const SUPPORT_SMOOTH = 0.03;                    // tanh width (≈1.7°) of the support-point sign

// numpy-style pairwise summation (the order np.sum uses), for the energy sums that steer kinetic damping
export function pairwiseSum(a, lo = 0, n = a.length) {
  if (n < 8) { let r = 0; for (let i = 0; i < n; i++) r += a[lo + i]; return r; }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] += a[lo + i + k];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = n >> 1; n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
}

const tanhSafe = Math.tanh;

// ================================================================================================= ActiveBeam
export class ActiveBeam {
  /** stiff: per-segment {EA, EIx, EIy, GJ}; activation: Float64Array(3p) (flat, edge, twist); rest: Omega0 to keep */
  constructor(st, stiff, prestress = 'flat', activation = null, rest = null) {
    const m = st.m, p = st.p;
    this.segEA = Float64Array.from(stiff.EA); this.segEIx = Float64Array.from(stiff.EIx); this.segEIy = Float64Array.from(stiff.EIy); this.segGJ = Float64Array.from(stiff.GJ);
    this.EA = this.segEA;
    this.EIx = new Float64Array(p); this.EIy = new Float64Array(p); this.GJ = new Float64Array(p);
    for (let j = 0; j < p; j++) { const s0 = st.jSeg0[j]; this.EIx[j] = this.segEIx[s0]; this.EIy[j] = this.segEIy[s0]; this.GJ[j] = this.segGJ[s0]; }
    ActiveBeam.alignFrames(st);
    const act = new Float64Array(3 * p);
    if (prestress !== 'asbuilt' && activation) for (let i = 0; i < 3 * p; i++) act[i] = Math.min(Math.max(activation[i], 0), 1);
    this.activation = act;
    if (rest) st.Omega0 = Float64Array.from(rest);
    else if (p && act.some(v => v < 1.0)) {
      const om = ActiveBeam.asBuilt(st);
      st.Omega0 = new Float64Array(3 * p);
      for (let i = 0; i < 3 * p; i++) st.Omega0[i] = (1.0 - act[i]) * om[i];
    } else st.Omega0 = new Float64Array(3 * p);
    this.N = new Float64Array(m); this.M = new Float64Array(3 * p);
    this.energy = 0; this.energyParts = { bending: 0, torsion: 0, stretch: 0 };
    this.cNode = []; this.cNb = []; this.cSeg = []; this.cDir = []; this.cLen = []; this.cM = new Float64Array(0); this.cEIx = []; this.cEIy = [];
    this._ra = new Float64Array(9); this._rb = new Float64Array(9); this._rr = new Float64Array(9);
  }

  setClamps(st, clamps) {
    this.cNode = clamps.map(c => c[0]); this.cNb = clamps.map(c => c[1]); this.cSeg = clamps.map(c => c[2]); this.cDir = clamps.map(c => c[3]);
    this.cLen = this.cSeg.map(j => 0.5 * st.l0[j]);
    this.cEIx = this.cSeg.map(j => this.segEIx[j]); this.cEIy = this.cSeg.map(j => this.segEIy[j]);
    this.cM = new Float64Array(2 * clamps.length);
  }

  /** rotate every segment frame minimally so that its Z axis is the current chord */
  static alignFrames(st) {
    const R = new Float64Array(9), Rr = new Float64Array(9), Rn = new Float64Array(9), q = new Float64Array(4), X = st.X;
    for (let j = 0; j < st.m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      let ex = X[3 * b] - X[3 * a], ey = X[3 * b + 1] - X[3 * a + 1], ez = X[3 * b + 2] - X[3 * a + 2];
      const l = Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 1e-12); ex /= l; ey /= l; ez /= l;
      toFrame(st.Q, 4 * j, R);
      rotBetween(R[6], R[7], R[8], ex, ey, ez, Rr);
      // Rn = Rr · R  (columns of R rotated)
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) Rn[3 * c + r] = Rr[3 * r] * R[3 * c] + Rr[3 * r + 1] * R[3 * c + 1] + Rr[3 * r + 2] * R[3 * c + 2];
      fromFrameFast(Rn, q);
      st.Q[4 * j] = q[0]; st.Q[4 * j + 1] = q[1]; st.Q[4 * j + 2] = q[2]; st.Q[4 * j + 3] = q[3];
    }
  }

  /** rest curvature/twist of the current geometry, per joint: (κb·m1, κb·m2, φ)/l̄ */
  static asBuilt(st) {
    const out = new Float64Array(3 * st.p), Ra = new Float64Array(9), Rb = new Float64Array(9), P = new Float64Array(9);
    for (let j = 0; j < st.p; j++) {
      const c = ActiveBeam.curv(st, j, Ra, Rb);
      const lb = st.jLen[j];
      out[3 * j] = (c.kb[0] * c.m1[0] + c.kb[1] * c.m1[1] + c.kb[2] * c.m1[2]) / lb;
      out[3 * j + 1] = (c.kb[0] * c.m2[0] + c.kb[1] * c.m2[1] + c.kb[2] * c.m2[2]) / lb;
      out[3 * j + 2] = ActiveBeam.twist(Ra, Rb, P) / lb;
    }
    return out;
  }

  static curv(st, j, Ra, Rb) {
    const X = st.X, s0 = st.jSeg0[j], s1 = st.jSeg1[j];
    const a = [X[3 * st.segI1[s0]] - X[3 * st.segI0[s0]], X[3 * st.segI1[s0] + 1] - X[3 * st.segI0[s0] + 1], X[3 * st.segI1[s0] + 2] - X[3 * st.segI0[s0] + 2]];
    const b = [X[3 * st.segI1[s1]] - X[3 * st.segI0[s1]], X[3 * st.segI1[s1] + 1] - X[3 * st.segI0[s1] + 1], X[3 * st.segI1[s1] + 2] - X[3 * st.segI0[s1] + 2]];
    const la = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), lb = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    const den = Math.max(la * lb + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]), 1e-12);
    const kb = [2 * (a[1] * b[2] - a[2] * b[1]) / den, 2 * (a[2] * b[0] - a[0] * b[2]) / den, 2 * (a[0] * b[1] - a[1] * b[0]) / den];
    toFrame(st.Q, 4 * s0, Ra); toFrame(st.Q, 4 * s1, Rb);
    const unit = (x, y, z) => { const l = Math.max(Math.sqrt(x * x + y * y + z * z), 1e-12); return [x / l, y / l, z / l]; };
    const m1 = unit(Ra[0] + Rb[0], Ra[1] + Rb[1], Ra[2] + Rb[2]);
    const m2 = unit(Ra[3] + Rb[3], Ra[4] + Rb[4], Ra[5] + Rb[5]);
    return { a, b, la, lb, den, kb, m1, m2 };
  }

  /** twist angle at a joint: X of segment a parallel-transported onto chord b vs X of segment b */
  static twist(Ra, Rb, P) {
    rotBetween(Ra[6], Ra[7], Ra[8], Rb[6], Rb[7], Rb[8], P);
    const xa = [P[0] * Ra[0] + P[1] * Ra[1] + P[2] * Ra[2], P[3] * Ra[0] + P[4] * Ra[1] + P[5] * Ra[2], P[6] * Ra[0] + P[7] * Ra[1] + P[8] * Ra[2]];
    const xb = [Rb[0], Rb[1], Rb[2]], tb = [Rb[6], Rb[7], Rb[8]];
    const cx = xa[1] * xb[2] - xa[2] * xb[1], cy = xa[2] * xb[0] - xa[0] * xb[2], cz = xa[0] * xb[1] - xa[1] * xb[0];
    const s = cx * tb[0] + cy * tb[1] + cz * tb[2];
    const c = xa[0] * xb[0] + xa[1] * xb[1] + xa[2] * xb[2];
    return Math.atan2(s, c);
  }

  addStiffness(st, kNode, kSeg) {
    for (let j = 0; j < st.m; j++) kNode[st.segI0[j]] += this.EA[j] / st.l0[j];
    for (let j = 0; j < st.m; j++) kNode[st.segI1[j]] += this.EA[j] / st.l0[j];
    if (st.p) {
      const kb = new Float64Array(st.p);
      for (let j = 0; j < st.p; j++) kb[j] = 4.0 * Math.max(this.EIx[j], this.EIy[j]) / Math.max(st.jLen[j], 1e-12) ** 3;
      for (let j = 0; j < st.p; j++) kNode[st.segI0[st.jSeg0[j]]] += kb[j];
      for (let j = 0; j < st.p; j++) kNode[st.jNode[j]] += kb[j];
      for (let j = 0; j < st.p; j++) kNode[st.segI1[st.jSeg1[j]]] += kb[j];
      for (let j = 0; j < st.p; j++) kSeg[st.jSeg0[j]] += this.GJ[j] / Math.max(st.jLen[j], 1e-12);
      for (let j = 0; j < st.p; j++) kSeg[st.jSeg1[j]] += this.GJ[j] / Math.max(st.jLen[j], 1e-12);
    }
    for (let c = 0; c < this.cNode.length; c++) { const kc = 4.0 * Math.max(this.cEIx[c], this.cEIy[c]) / Math.max(this.cLen[c], 1e-12) ** 3; kNode[this.cNb[c]] += kc; }
    for (let c = 0; c < this.cNode.length; c++) { const kc = 4.0 * Math.max(this.cEIx[c], this.cEIy[c]) / Math.max(this.cLen[c], 1e-12) ** 3; kNode[this.cNode[c]] += kc; }
  }

  apply(st, F, T) {
    const X = st.X, m = st.m, p = st.p;
    // stretch
    const fs = new Float64Array(3 * m);
    let eStretch = 0;
    const eS = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      const ex = X[3 * b] - X[3 * a], ey = X[3 * b + 1] - X[3 * a + 1], ez = X[3 * b + 2] - X[3 * a + 2];
      const L = Math.sqrt(ex * ex + ey * ey + ez * ez), Lc = Math.max(L, 1e-12);
      const N = this.EA[j] * (L - st.l0[j]) / st.l0[j];
      this.N[j] = N;
      fs[3 * j] = N * ex / Lc; fs[3 * j + 1] = N * ey / Lc; fs[3 * j + 2] = N * ez / Lc;
      eS[j] = N * (L - st.l0[j]);
    }
    for (let j = 0; j < m; j++) { const a = st.segI0[j]; F[3 * a] += fs[3 * j]; F[3 * a + 1] += fs[3 * j + 1]; F[3 * a + 2] += fs[3 * j + 2]; }
    for (let j = 0; j < m; j++) { const b = st.segI1[j]; F[3 * b] -= fs[3 * j]; F[3 * b + 1] -= fs[3 * j + 1]; F[3 * b + 2] -= fs[3 * j + 2]; }
    eStretch = 0.5 * pairwiseSum(eS);
    let eBend = 0, eTwist = 0;
    if (p) {
      const Fim = new Float64Array(3 * p), Fi = new Float64Array(3 * p), Fip = new Float64Array(3 * p), Ta = new Float64Array(3 * p), Tb = new Float64Array(3 * p);
      const eB = new Float64Array(p), eT = new Float64Array(p);
      const Ra = this._ra, Rb = this._rb, P = this._rr;
      for (let j = 0; j < p; j++) {
        const c = ActiveBeam.curv(st, j, Ra, Rb);
        const { a, b, la, lb, den, kb, m1, m2 } = c;
        const lbar = st.jLen[j];
        const k1 = (kb[0] * m1[0] + kb[1] * m1[1] + kb[2] * m1[2]) / lbar;
        const k2 = (kb[0] * m2[0] + kb[1] * m2[1] + kb[2] * m2[2]) / lbar;
        const M1 = this.EIx[j] * (k1 - st.Omega0[3 * j]), M2 = this.EIy[j] * (k2 - st.Omega0[3 * j + 1]);
        const g = [M1 * m1[0] + M2 * m2[0], M1 * m1[1] + M2 * m2[1], M1 * m1[2] + M2 * m2[2]];
        const kg = kb[0] * g[0] + kb[1] * g[1] + kb[2] * g[2];
        const ah = [a[0] / la, a[1] / la, a[2] / la], bh = [b[0] / lb, b[1] / lb, b[2] / lb];
        const bxg = [b[1] * g[2] - b[2] * g[1], b[2] * g[0] - b[0] * g[2], b[0] * g[1] - b[1] * g[0]];
        const axg = [a[1] * g[2] - a[2] * g[1], a[2] * g[0] - a[0] * g[2], a[0] * g[1] - a[1] * g[0]];
        for (let k = 0; k < 3; k++) {
          const fim = (2 * bxg[k] - (lb * ah[k] + b[k]) * kg) / den;
          const fip = (2 * axg[k] + (la * bh[k] + a[k]) * kg) / den;
          Fim[3 * j + k] = fim; Fip[3 * j + k] = fip; Fi[3 * j + k] = -(fim + fip);
        }
        eB[j] = lbar * (M1 * (k1 - st.Omega0[3 * j]) + M2 * (k2 - st.Omega0[3 * j + 1]));
        // twist
        const phi = ActiveBeam.twist(Ra, Rb, P), tau = phi / lbar;
        const Mt = this.GJ[j] * (tau - st.Omega0[3 * j + 2]);
        for (let k = 0; k < 3; k++) { Ta[3 * j + k] = Mt * Ra[6 + k]; Tb[3 * j + k] = -Mt * Rb[6 + k]; }
        eT[j] = lbar * Mt * (tau - st.Omega0[3 * j + 2]);
        this.M[3 * j] = M1; this.M[3 * j + 1] = M2; this.M[3 * j + 2] = Mt;
      }
      for (let j = 0; j < p; j++) { const i = st.segI0[st.jSeg0[j]]; for (let k = 0; k < 3; k++) F[3 * i + k] += Fim[3 * j + k]; }
      for (let j = 0; j < p; j++) { const i = st.jNode[j]; for (let k = 0; k < 3; k++) F[3 * i + k] += Fi[3 * j + k]; }
      for (let j = 0; j < p; j++) { const i = st.segI1[st.jSeg1[j]]; for (let k = 0; k < 3; k++) F[3 * i + k] += Fip[3 * j + k]; }
      eBend += 0.5 * pairwiseSum(eB);
      for (let j = 0; j < p; j++) { const s = st.jSeg0[j]; for (let k = 0; k < 3; k++) T[3 * s + k] += Ta[3 * j + k]; }
      for (let j = 0; j < p; j++) { const s = st.jSeg1[j]; for (let k = 0; k < 3; k++) T[3 * s + k] += Tb[3 * j + k]; }
      eTwist += 0.5 * pairwiseSum(eT);
    }
    const nc = this.cNode.length;
    if (nc) {
      const R = this._ra, Fnb = new Float64Array(3 * nc), eC = new Float64Array(nc);
      for (let c = 0; c < nc; c++) {
        const a = this.cDir[c], nb = this.cNb[c], nd = this.cNode[c];
        const b = [X[3 * nb] - X[3 * nd], X[3 * nb + 1] - X[3 * nd + 1], X[3 * nb + 2] - X[3 * nd + 2]];
        const lb = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        const den = Math.max(lb + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]), 1e-12);
        const kb = [2 * (a[1] * b[2] - a[2] * b[1]) / den, 2 * (a[2] * b[0] - a[0] * b[2]) / den, 2 * (a[0] * b[1] - a[1] * b[0]) / den];
        toFrame(st.Q, 4 * this.cSeg[c], R);
        const m1 = [R[0], R[1], R[2]], m2 = [R[3], R[4], R[5]], lc = this.cLen[c];
        const k1 = (kb[0] * m1[0] + kb[1] * m1[1] + kb[2] * m1[2]) / lc, k2 = (kb[0] * m2[0] + kb[1] * m2[1] + kb[2] * m2[2]) / lc;
        const M1 = this.cEIx[c] * k1, M2 = this.cEIy[c] * k2;
        const g = [M1 * m1[0] + M2 * m2[0], M1 * m1[1] + M2 * m2[1], M1 * m1[2] + M2 * m2[2]];
        const kg = kb[0] * g[0] + kb[1] * g[1] + kb[2] * g[2];
        const lbc = Math.max(lb, 1e-12), bh = [b[0] / lbc, b[1] / lbc, b[2] / lbc];
        const axg = [a[1] * g[2] - a[2] * g[1], a[2] * g[0] - a[0] * g[2], a[0] * g[1] - a[1] * g[0]];
        for (let k = 0; k < 3; k++) Fnb[3 * c + k] = (2 * axg[k] + (bh[k] + a[k]) * kg) / den;
        eC[c] = lc * (M1 * k1 + M2 * k2);
        this.cM[2 * c] = M1; this.cM[2 * c + 1] = M2;
      }
      for (let c = 0; c < nc; c++) { const i = this.cNb[c]; for (let k = 0; k < 3; k++) F[3 * i + k] += Fnb[3 * c + k]; }
      for (let c = 0; c < nc; c++) { const i = this.cNode[c]; for (let k = 0; k < 3; k++) F[3 * i + k] -= Fnb[3 * c + k]; }
      eBend += 0.5 * pairwiseSum(eC);
    }
    this.energyParts = { bending: eBend, torsion: eTwist, stretch: eStretch };
    this.energy = eStretch + eBend + eTwist;
  }
}

// ================================================================================================= PolylineContact
export class PolylineContact {
  /** secOf(s) → section {width, height}, matOf(s) → material {fc90, E90, H} of the strand's lath group */
  constructor(topo, st, secOf, params, matOf = null) {
    this.p = params;
    const K = st.K; this.K = K;
    this.strand = new Int32Array(2 * K);
    st.knotEdges.forEach((e, c) => { const kn = topo.knots.get(e); this.strand[2 * c] = kn.a[0]; this.strand[2 * c + 1] = kn.b[0]; });
    this.wk = new Float64Array(2 * K); this.hk = new Float64Array(2 * K);
    this.fc90 = new Float64Array(2 * K); this.E90 = new Float64Array(2 * K); this.Hs = new Float64Array(2 * K);
    for (let i = 0; i < 2 * K; i++) {
      const sec = secOf(this.strand[i]); this.wk[i] = +sec.width; this.hk[i] = +sec.height;
      const mt = matOf ? matOf(this.strand[i]) : null;
      this.fc90[i] = mt?.fc90 ?? 8.0; this.E90[i] = mt?.E90 ?? 800.0; this.Hs[i] = mt?.H ?? 30.0;
    }
    this.w = new Float64Array(K); this.h = new Float64Array(K);
    for (let c = 0; c < K; c++) { this.w[c] = 0.5 * (this.wk[2 * c] + this.wk[2 * c + 1]); this.h[c] = 0.5 * (this.hk[2 * c] + this.hk[2 * c + 1]); }
    this.segOff = new Int32Array(2 * K); this.nseg = new Int32Array(2 * K);
    for (let i = 0; i < 2 * K; i++) { const s = this.strand[i]; this.segOff[i] = st.segOffset[s]; this.nseg[i] = st.segOffset[s + 1] - st.segOffset[s]; }
    this.locked = new Uint8Array(K).fill(params.locked ? 1 : 0);
    this.enabled = new Uint8Array(K).fill(1);
    if (topo.params.overhang <= 0) st.knotEdges.forEach((e, c) => { if (topo.knots.get(e).isBoundary) this.enabled[c] = 0; });
    this.theta0 = new Float64Array(K).fill(NaN); this.rotSlip = new Float64Array(K);
    this.dent = new Float64Array(2 * K); this.dentE = new Float64Array(K); this.climb = new Float64Array(2 * K); this.tanpsi = new Float64Array(2 * K).fill(1);
    this.Fn = new Float64Array(K); this.Ft = new Float64Array(K); this.state = new Float64Array(K); this.gap = new Float64Array(K);
    this.ctype = new Int32Array(K); this.faceSide = new Int32Array(K).fill(-1); this.phi = new Float64Array(2 * K); this.theta = new Float64Array(K);
    this.Anom = new Float64Array(K); this.pressure = new Float64Array(K); this.util = new Float64Array(K); this.Mrot = new Float64Array(K);
    this.mid = new Float64Array(3 * K);
    this.energy = 0; this.ramp = 1.0; this.rotEnergy = 0; this.dentAfterDefault = 1500;
    // per-side evaluation scratch
    this.ev = [0, 1].map(() => ({ P: new Float64Array(3 * K), t: new Float64Array(3 * K), R: new Float64Array(9 * K), L: new Float64Array(K), seg: new Int32Array(K), tp: new Float64Array(K), a: new Int32Array(K), b: new Int32Array(K) }));
    this.initialize(st);
  }

  /** stuck material points paired at the closest points of the two centrelines (guarded Gauss–Newton sweeps) */
  initialize(st, sweeps = 3) {
    const K = this.K;
    const offsets = () => {
      this._prepare(st); this._eval(st, 0); this._eval(st, 1);
      const A = this.ev[0], B = this.ev[1];
      const St = new Float64Array(3 * K), n = new Float64Array(3 * K);
      for (let c = 0; c < K; c++) {
        const tA = [A.t[3 * c], A.t[3 * c + 1], A.t[3 * c + 2]], tB = [B.t[3 * c], B.t[3 * c + 1], B.t[3 * c + 2]];
        let nx = tA[1] * tB[2] - tA[2] * tB[1], ny = tA[2] * tB[0] - tA[0] * tB[2], nz = tA[0] * tB[1] - tA[1] * tB[0];
        const nn = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nn < 1e-6) { nx = st.nPrev[3 * c]; ny = st.nPrev[3 * c + 1]; nz = st.nPrev[3 * c + 2]; }
        else { const d = Math.max(nn, 1e-12); nx /= d; ny /= d; nz /= d; }
        if (nx * st.nPrev[3 * c] + ny * st.nPrev[3 * c + 1] + nz * st.nPrev[3 * c + 2] < 0) { nx = -nx; ny = -ny; nz = -nz; }
        const S = [B.P[3 * c] - A.P[3 * c], B.P[3 * c + 1] - A.P[3 * c + 1], B.P[3 * c + 2] - A.P[3 * c + 2]];
        const g = S[0] * nx + S[1] * ny + S[2] * nz;
        St[3 * c] = S[0] - g * nx; St[3 * c + 1] = S[1] - g * ny; St[3 * c + 2] = S[2] - g * nz;
        n[3 * c] = nx; n[3 * c + 1] = ny; n[3 * c + 2] = nz;
      }
      return { St, n };
    };
    const nrm = (St, c) => Math.sqrt(St[3 * c] ** 2 + St[3 * c + 1] ** 2 + St[3 * c + 2] ** 2);
    for (let it = 0; it < sweeps; it++) {
      const { St } = offsets();
      const A = this.ev[0], B = this.ev[1];
      const uOld = st.u.slice(), before = new Float64Array(K);
      for (let c = 0; c < K; c++) {
        const tA = [A.t[3 * c], A.t[3 * c + 1], A.t[3 * c + 2]], tB = [B.t[3 * c], B.t[3 * c + 1], B.t[3 * c + 2]];
        const taa = tA[0] * tA[0] + tA[1] * tA[1] + tA[2] * tA[2], tbb = tB[0] * tB[0] + tB[1] * tB[1] + tB[2] * tB[2], tab = tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2];
        const sa = St[3 * c] * tA[0] + St[3 * c + 1] * tA[1] + St[3 * c + 2] * tA[2], sb = St[3 * c] * tB[0] + St[3 * c + 1] * tB[1] + St[3 * c + 2] * tB[2];
        const det = taa * tbb - tab * tab, ok = det > 1e-3 && this.enabled[c];
        const ca = ok ? (sa * tbb - sb * tab) / det : 0, cb = ok ? (sb * taa - sa * tab) / det : 0;
        before[c] = nrm(St, c);
        st.u[2 * c] = Math.min(Math.max(st.u[2 * c] + ca / Math.max(A.L[c], 1e-12), 0), this.nseg[2 * c]);
        st.u[2 * c + 1] = Math.min(Math.max(st.u[2 * c + 1] - cb / Math.max(B.L[c], 1e-12), 0), this.nseg[2 * c + 1]);
      }
      const { St: St2 } = offsets();
      for (let c = 0; c < K; c++) if (nrm(St2, c) >= before[c]) { st.u[2 * c] = uOld[2 * c]; st.u[2 * c + 1] = uOld[2 * c + 1]; }
    }
    const { n } = offsets();
    st.nPrev.set(n);
  }

  lock(mask = null) { for (let c = 0; c < this.K; c++) if (!mask || mask[c]) this.locked[c] = 1; }
  unlock(mask = null) { for (let c = 0; c < this.K; c++) if (!mask || mask[c]) this.locked[c] = 0; }

  _segment(st, side, c) {
    const u = st.u[2 * c + side], n = this.nseg[2 * c + side];
    const i = Math.min(Math.max(Math.floor(u), 0), Math.max(n - 1, 0));
    const t = Math.min(Math.max(u - i, 0.0), 1.0);
    return [this.segOff[2 * c + side] + i, t, i];
  }

  addStiffness(st, kNode, kSeg) {
    const k = this.p.strength + this.p.stick_strength;
    for (let side = 0; side < 2; side++) {
      for (let c = 0; c < this.K; c++) kNode[st.segI0[this._segment(st, side, c)[0]]] += k;
      for (let c = 0; c < this.K; c++) kNode[st.segI1[this._segment(st, side, c)[0]]] += k;
    }
    if (this.p.rot_friction) {
      for (let side = 0; side < 2; side++) {
        for (let pass = 0; pass < 2; pass++) for (let c = 0; c < this.K; c++) {
          const seg = this._segment(st, side, c)[0];
          const kr = this.p.stick_strength * (0.38 ** 2) * this.wk[2 * c] * this.wk[2 * c + 1] / 0.5;
          kNode[pass ? st.segI1[seg] : st.segI0[seg]] += kr / Math.max(st.l0[seg], 1e-12) ** 2;
        }
      }
    }
    this.addTwistStiffness(st, kSeg);
  }

  /** righting stiffness of the smoothed support point (Fn + k_n h/4)(w/2)/δ plus the friction lever k_t (h/2)² */
  addTwistStiffness(st, kSeg) {
    const p = this.p, kn = this.ramp * p.strength, kt = this.ramp * p.stick_strength, tf = Math.max(p.torsion_factor, 1e-3);
    for (let side = 0; side < 2; side++) for (let c = 0; c < this.K; c++) {
      const kth = tf * ((this.Fn[c] + kn * 0.25 * this.h[c]) * (0.5 * this.w[c]) / SUPPORT_SMOOTH + kt * (0.5 * this.h[c]) ** 2);
      kSeg[this._segment(st, side, c)[0]] += kth;
    }
  }

  /** per-iteration caches: node tangents (normalised sum of the adjacent unit chords) and node frames */
  _prepare(st) {
    const X = st.X, n = st.n;
    const T = this._Tn && this._Tn.length === 3 * n ? this._Tn.fill(0) : (this._Tn = new Float64Array(3 * n));
    const eh = new Float64Array(3 * st.m);
    for (let j = 0; j < st.m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      const ex = X[3 * b] - X[3 * a], ey = X[3 * b + 1] - X[3 * a + 1], ez = X[3 * b + 2] - X[3 * a + 2];
      const l = Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 1e-12);
      eh[3 * j] = ex / l; eh[3 * j + 1] = ey / l; eh[3 * j + 2] = ez / l;
    }
    for (let j = 0; j < st.m; j++) { const a = st.segI0[j]; T[3 * a] += eh[3 * j]; T[3 * a + 1] += eh[3 * j + 1]; T[3 * a + 2] += eh[3 * j + 2]; }
    for (let j = 0; j < st.m; j++) { const b = st.segI1[j]; T[3 * b] += eh[3 * j]; T[3 * b + 1] += eh[3 * j + 1]; T[3 * b + 2] += eh[3 * j + 2]; }
    for (let i = 0; i < n; i++) { const l = Math.max(Math.sqrt(T[3 * i] ** 2 + T[3 * i + 1] ** 2 + T[3 * i + 2] ** 2), 1e-12); T[3 * i] /= l; T[3 * i + 1] /= l; T[3 * i + 2] /= l; }
    this._Qn = nodeQuats(st, this._Qn && this._Qn.length === 4 * n ? this._Qn : undefined);
  }

  /** cubic Hermite centreline (C¹) between the segment's nodes with the node tangents; frame = nlerp of the node
   *  frames, re-orthogonalised to the local tangent */
  _eval(st, side) {
    const ev = this.ev[side], X = st.X, Tn = this._Tn, Qn = this._Qn, Rn = new Float64Array(9), q = new Float64Array(4);
    for (let c = 0; c < this.K; c++) {
      const [seg, t] = this._segment(st, side, c);
      const a = st.segI0[seg], b = st.segI1[seg];
      const dx = X[3 * b] - X[3 * a], dy = X[3 * b + 1] - X[3 * a + 1], dz = X[3 * b + 2] - X[3 * a + 2];
      const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const t2 = t * t, t3 = t * t * t;
      const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
      const d00 = 6 * t2 - 6 * t, d10 = 3 * t2 - 4 * t + 1, d01 = -6 * t2 + 6 * t, d11 = 3 * t2 - 2 * t;
      const tg = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        ev.P[3 * c + k] = h00 * X[3 * a + k] + (h10 * L) * Tn[3 * a + k] + h01 * X[3 * b + k] + (h11 * L) * Tn[3 * b + k];
        tg[k] = d00 * X[3 * a + k] + (d10 * L) * Tn[3 * a + k] + d01 * X[3 * b + k] + (d11 * L) * Tn[3 * b + k];
      }
      const tl = Math.max(Math.sqrt(tg[0] * tg[0] + tg[1] * tg[1] + tg[2] * tg[2]), 1e-12);
      tg[0] /= tl; tg[1] /= tl; tg[2] /= tl;
      const sgn = (Qn[4 * a] * Qn[4 * b] + Qn[4 * a + 1] * Qn[4 * b + 1] + Qn[4 * a + 2] * Qn[4 * b + 2] + Qn[4 * a + 3] * Qn[4 * b + 3]) < 0 ? -1 : 1;
      for (let k = 0; k < 4; k++) q[k] = (1 - t) * Qn[4 * a + k] + t * sgn * Qn[4 * b + k];
      qnormalize(q);
      toFrame(q, 0, Rn);
      let y = [Rn[3], Rn[4], Rn[5]];
      const yt = y[0] * tg[0] + y[1] * tg[1] + y[2] * tg[2];
      y = [y[0] - yt * tg[0], y[1] - yt * tg[1], y[2] - yt * tg[2]];
      const yl = Math.max(Math.sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]), 1e-12);
      y[0] /= yl; y[1] /= yl; y[2] /= yl;
      const x = [y[1] * tg[2] - y[2] * tg[1], y[2] * tg[0] - y[0] * tg[2], y[0] * tg[1] - y[1] * tg[0]];
      ev.R.set([x[0], x[1], x[2], y[0], y[1], y[2], tg[0], tg[1], tg[2]], 9 * c);
      ev.t[3 * c] = tg[0]; ev.t[3 * c + 1] = tg[1]; ev.t[3 * c + 2] = tg[2];
      ev.L[c] = L; ev.seg[c] = seg; ev.tp[c] = t; ev.a[c] = a; ev.b[c] = b;
    }
  }

  /** W1: contact type per knot from the roll of each section against the contact plane (outputs; feeds W2/W3) */
  _classify(n) {
    const p = this.p, K = this.K, A = this.ev[0], B = this.ev[1];
    const cFace = new Float64Array(2 * K), Z0 = new Float64Array(2 * K), edge = new Uint8Array(2 * K);
    for (let c = 0; c < K; c++) {
      for (let side = 0; side < 2; side++) {
        const R = side === 0 ? A.R : B.R, sg = side === 0 ? 1.0 : -1.0, o = 9 * c, i = 2 * c + side;
        const nx = (n[3 * c] * R[o] + n[3 * c + 1] * R[o + 1] + n[3 * c + 2] * R[o + 2]) * sg;
        const ny = (n[3 * c] * R[o + 3] + n[3 * c + 1] * R[o + 4] + n[3 * c + 2] * R[o + 5]) * sg;
        const sx = Math.abs(tanhSafe(nx / SUPPORT_SMOOTH)), sy = Math.abs(tanhSafe(ny / SUPPORT_SMOOTH));
        edge[i] = sx * sy > p.edge_threshold ? 1 : 0;
        const yFace = Math.abs(ny) >= Math.abs(nx);
        cFace[i] = yFace ? this.wk[i] : this.hk[i];
        Z0[i] = yFace ? this.hk[i] : this.wk[i];
        this.phi[i] = Math.atan(Math.min(Math.abs(nx), Math.abs(ny)) / Math.max(Math.max(Math.abs(nx), Math.abs(ny)), 1e-12));
      }
      const tA = [A.t[3 * c], A.t[3 * c + 1], A.t[3 * c + 2]], tB = [B.t[3 * c], B.t[3 * c + 1], B.t[3 * c + 2]];
      const cr = [tA[1] * tB[2] - tA[2] * tB[1], tA[2] * tB[0] - tA[0] * tB[2], tA[0] * tB[1] - tA[1] * tB[0]];
      const sinT = Math.min(Math.max(Math.sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]), 1e-6), 1.0);
      this.theta[c] = Math.asin(sinT);
      const nEdge = edge[2 * c] + edge[2 * c + 1];
      this.ctype[c] = nEdge;
      this.faceSide[c] = nEdge === 1 ? (edge[2 * c + 1] ? 0 : 1) : -1;
      const Fn = Math.abs(this.Fn[c]);
      let Aa = cFace[2 * c] * cFace[2 * c + 1] / sinT, pres = Fn / Math.max(Aa, 1e-12), util = pres / Math.max(Math.min(this.Hs[2 * c], this.Hs[2 * c + 1]), 1e-12);
      if (nEdge === 1) {
        const fs = this.faceSide[c], es = 1 - fs;
        const ell = cFace[2 * c + fs] / sinT, kap = this.E90[2 * c + fs] / Math.max(Z0[2 * c + fs], 1e-12);
        const phiE = Math.max(this.phi[2 * c + es], 1e-4);
        const b = Math.sqrt(2.0 * Fn / Math.max(ell * kap * phiE, 1e-12));
        const full = b >= cFace[2 * c + es];
        const Afe = full ? cFace[2 * c] * cFace[2 * c + 1] / sinT : b * ell;
        const pfe = full ? Fn / Math.max(Afe, 1e-12) : kap * phiE * b;
        Aa = Afe; pres = pfe; util = pfe / Math.max(this.fc90[2 * c + es], 1e-12);
      } else if (nEdge === 2) {
        const fc = Math.min(this.fc90[2 * c], this.fc90[2 * c + 1]);
        Aa = Fn / Math.max(fc, 1e-12); pres = fc * (Fn > 0 ? 1 : 0); util = Fn > 0 ? 1.0 : 0.0;
      }
      this.Anom[c] = Aa; this.pressure[c] = pres; this.util[c] = util;
    }
    return cFace;
  }

  /** half-extent h(n) and smoothed support point r of the section of `side` facing sign·n */
  _support(R, o, n, c, sign, side, r) {
    const Xa = [R[o], R[o + 1], R[o + 2]], Ya = [R[o + 3], R[o + 4], R[o + 5]];
    const w = this.wk[2 * c + side], h_ = this.hk[2 * c + side];
    const nx = (n[0] * Xa[0] + n[1] * Xa[1] + n[2] * Xa[2]) * sign, ny = (n[0] * Ya[0] + n[1] * Ya[1] + n[2] * Ya[2]) * sign;
    const h = 0.5 * w * Math.abs(nx) + 0.5 * h_ * Math.abs(ny);
    const ax = 0.5 * w * tanhSafe(nx / SUPPORT_SMOOTH), ay = 0.5 * h_ * tanhSafe(ny / SUPPORT_SMOOTH);
    for (let k = 0; k < 3; k++) r[k] = ax * Xa[k] + ay * Ya[k];
    return h;
  }

  // --- W3: directional seat law L1′ -----------------------------------------------------------------------------
  _walls(sideCut, R, o, n, m) {
    const sg = sideCut === 0 ? 1.0 : -1.0;
    const X = [R[o], R[o + 1], R[o + 2]], Y = [R[o + 3], R[o + 4], R[o + 5]];
    const nx = (n[0] * X[0] + n[1] * X[1] + n[2] * X[2]) * sg, ny = (n[0] * Y[0] + n[1] * Y[1] + n[2] * Y[2]) * sg;
    const fx = X.map(v => Math.sign(nx) * v), fy = Y.map(v => Math.sign(ny) * v);
    const f = (fx[0] * m[0] + fx[1] * m[1] + fx[2] * m[2]) > 0 ? fx : fy;
    const cosPsi = Math.min(Math.max(Math.abs(f[0] * n[0] + f[1] * n[1] + f[2] * n[2]), 1e-6), 1.0);
    return Math.tan(Math.acos(cosPsi));
  }

  _seatFriction(c, St, n, tA, tB) {
    const p = this.p;
    let mu = p.mu;
    const norm = Math.sqrt(St[0] * St[0] + St[1] * St[1] + St[2] * St[2]);
    const dn = Math.max(norm, 1e-12), dhat = [St[0] / dn, St[1] / dn, St[2] / dn], has = norm > 1e-12;
    for (let j = 0; j < 2; j++) {
      const act = this.dent[2 * c + j] > 0 && has;
      if (!act) continue;
      const tCut = j === 0 ? tB : tA, Rcut = j === 0 ? this.ev[1].R : this.ev[0].R;
      let m = [n[1] * tCut[2] - n[2] * tCut[1], n[2] * tCut[0] - n[0] * tCut[2], n[0] * tCut[1] - n[1] * tCut[0]];
      const ml = Math.max(Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]), 1e-12); m = m.map(v => v / ml);
      const sgn = j === 0 ? 1.0 : -1.0;
      const dj = (dhat[0] * m[0] + dhat[1] * m[1] + dhat[2] * m[2]) * sgn;
      const wallDir = m.map(v => Math.sign(dj) * v * sgn);
      const tanpsi = this._walls(1 - j, Rcut, 9 * c, n, wallDir);
      this.tanpsi[2 * c + j] = tanpsi;
      const add = Math.abs(dj) * (p.mu * tanpsi < 1.0 ? (p.mu + tanpsi) / Math.max(1.0 - p.mu * tanpsi, 1e-6) - p.mu : 9.0 * p.mu);
      mu += Math.min(add, 9.0 * p.mu);
    }
    return mu;
  }

  _seatUpdate(st, c, slip, da, db, tA, tB, n, Fn, cFace, active) {
    const p = this.p;
    const inc = [da * tA[0] - db * tB[0], da * tA[1] - db * tB[1], da * tA[2] - db * tB[2]];
    for (let j = 0; j < 2; j++) {
      const tCut = j === 0 ? tB : tA;
      let m = [n[1] * tCut[2] - n[2] * tCut[1], n[2] * tCut[0] - n[0] * tCut[2], n[0] * tCut[1] - n[1] * tCut[0]];
      const ml = Math.max(Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]), 1e-12); m = m.map(v => v / ml);
      const step = (inc[0] * m[0] + inc[1] * m[1] + inc[2] * m[2]) * (j === 0 ? 1.0 : -1.0);
      const i = 2 * c + j, has = this.dent[i] > 0;
      if (has && slip) this.climb[i] += step;
      if (has && Math.abs(this.climb[i]) >= this.dent[i] / Math.max(this.tanpsi[i], 1e-6)) { this.dent[i] = 0; this.climb[i] = 0; }
    }
    const after = p.dent_after_iter >= 0 ? p.dent_after_iter : this.dentAfterDefault;
    if (st.iteration < after) return;
    const sinp0 = Math.sin(this.phi[2 * c]) * Math.cos(this.phi[2 * c]), sinp1 = Math.sin(this.phi[2 * c + 1]) * Math.cos(this.phi[2 * c + 1]);
    const sinT = Math.sin(this.theta[c]);
    if (this.ctype[c] === 2 && active && !this.locked[c]) {
      const fc = Math.min(this.fc90[2 * c], this.fc90[2 * c + 1]);
      const dEst = Math.sqrt(Math.abs(Fn) / Math.max(fc, 1e-12) * sinT * sinp0 * sinp1);
      for (let j = 0; j < 2; j++) { const cur = this.dent[2 * c + j]; this.dent[2 * c + j] = Math.max(cur, Math.min(cur + p.dent_rate * dEst, dEst)); }
    }
    if (this.ctype[c] === 1 && active && !this.locked[c]) {
      const fs = this.faceSide[c], es = 1 - fs;
      const ell = cFace[2 * c + fs] / Math.max(sinT, 1e-6);
      const dE = Math.abs(Fn) / Math.max(this.fc90[2 * c + es] * ell, 1e-12) * (es === 0 ? sinp0 : sinp1);
      this.dentE[c] = Math.max(this.dentE[c], dE);
    }
  }

  // --- W2: rotational friction about n --------------------------------------------------------------------------
  _rotationalFriction(st, T, n, active, kt) {
    const p = this.p, K = this.K, A = this.ev[0], B = this.ev[1];
    const M = new Float64Array(K), dthA = new Float64Array(K), krA = new Float64Array(K);
    for (let c = 0; c < K; c++) {
      const tA = [A.t[3 * c], A.t[3 * c + 1], A.t[3 * c + 2]], tB = [B.t[3 * c], B.t[3 * c + 1], B.t[3 * c + 2]];
      const cr = [tA[1] * tB[2] - tA[2] * tB[1], tA[2] * tB[0] - tA[0] * tB[2], tA[0] * tB[1] - tA[1] * tB[0]];
      const theta = Math.atan2(cr[0] * n[3 * c] + cr[1] * n[3 * c + 1] + cr[2] * n[3 * c + 2], tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2]);
      if (Number.isNaN(this.theta0[c]) && active[c]) this.theta0[c] = theta;
      const ok = active[c] && !Number.isNaN(this.theta0[c]);
      const sinT = Math.max(Math.sin(this.theta[c]), 1e-6);
      let req = 0.38 * Math.sqrt(this.wk[2 * c] * this.wk[2 * c + 1] / sinT);
      if (this.ctype[c] === 1) { const cw = this.faceSide[c] === 0 ? this.wk[2 * c] : this.wk[2 * c + 1]; req = 0.25 * cw / sinT; }
      else if (this.ctype[c] === 2) req = 0.38 * Math.sqrt(Math.max(this.Anom[c], 1e-12));
      const kr = kt * req * req;
      let dth = ok ? theta - this.theta0[c] : 0.0;
      dth = ((dth + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      let Mc = kr * dth;
      const Mlim = this.locked[c] ? Infinity : p.rot_factor * p.mu * Math.abs(this.Fn[c]) * req;
      if (ok && Math.abs(Mc) > Mlim) {
        const rate = Math.min(Math.max(p.slip_rate, 1e-3), 1.0);
        let adv = rate * (1.0 - Mlim / Math.max(Math.abs(Mc), 1e-12)) * dth;
        adv = Math.min(Math.max(adv, -0.05), 0.05);
        this.theta0[c] += adv; this.rotSlip[c] += Math.abs(adv); dth -= adv;
        Mc = kr * dth;
      }
      M[c] = ok ? Mc : 0.0; dthA[c] = dth; krA[c] = kr;
    }
    this.Mrot = M;
    for (let c = 0; c < K; c++) { const s = A.seg[c]; for (let k = 0; k < 3; k++) T[3 * s + k] += M[c] * n[3 * c + k]; }
    for (let c = 0; c < K; c++) { const s = B.seg[c]; for (let k = 0; k < 3; k++) T[3 * s + k] -= M[c] * n[3 * c + k]; }
    const e = new Float64Array(K); for (let c = 0; c < K; c++) e[c] = krA[c] * dthA[c] * dthA[c];
    this.rotEnergy = 0.5 * pairwiseSum(e);
  }

  apply(st, F, T) {
    const K = this.K; if (!K) return;
    const p = this.p;
    this._prepare(st); this._eval(st, 0); this._eval(st, 1);
    const A = this.ev[0], B = this.ev[1];
    const kn = this.ramp * p.strength, kt = this.ramp * p.stick_strength, seat = p.seat_law !== 'off', aug = !!p.augmented;
    const n = new Float64Array(3 * K), FA = new Float64Array(3 * K), FB = new Float64Array(3 * K), rAx = new Float64Array(3 * K), rBx = new Float64Array(3 * K);
    const slipA = new Uint8Array(K), activeA = new Uint8Array(K), daA = new Float64Array(K), dbA = new Float64Array(K);
    const ePen = new Float64Array(K), eTan = new Float64Array(K);
    const rA = [0, 0, 0], rB = [0, 0, 0];
    const tAs = [], tBs = [], Sts = [];
    for (let c = 0; c < K; c++) {
      const tA = [A.t[3 * c], A.t[3 * c + 1], A.t[3 * c + 2]], tB = [B.t[3 * c], B.t[3 * c + 1], B.t[3 * c + 2]];
      // contact normal with side persistence (independent of the twist DOF → the lever torque is conservative)
      let nx = tA[1] * tB[2] - tA[2] * tB[1], ny = tA[2] * tB[0] - tA[0] * tB[2], nz = tA[0] * tB[1] - tA[1] * tB[0];
      const nn = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (nn < 1e-6) { nx = st.nPrev[3 * c]; ny = st.nPrev[3 * c + 1]; nz = st.nPrev[3 * c + 2]; }
      else { const d = Math.max(nn, 1e-12); nx /= d; ny /= d; nz /= d; }
      if (nx * st.nPrev[3 * c] + ny * st.nPrev[3 * c + 1] + nz * st.nPrev[3 * c + 2] < 0) { nx = -nx; ny = -ny; nz = -nz; }
      st.nPrev[3 * c] = nx; st.nPrev[3 * c + 1] = ny; st.nPrev[3 * c + 2] = nz;
      n[3 * c] = nx; n[3 * c + 1] = ny; n[3 * c + 2] = nz;
      const nv = [nx, ny, nz];
      const S = [B.P[3 * c] - A.P[3 * c], B.P[3 * c + 1] - A.P[3 * c + 1], B.P[3 * c + 2] - A.P[3 * c + 2]];
      for (let k = 0; k < 3; k++) this.mid[3 * c + k] = 0.5 * (A.P[3 * c + k] + B.P[3 * c + k]);
      const g = S[0] * nx + S[1] * ny + S[2] * nz;
      const hA = this._support(A.R, 9 * c, nv, c, +1.0, 0, rA);          // A's face towards B (+n)
      const hB = this._support(B.R, 9 * c, nv, c, -1.0, 1, rB);          // B's face towards A (−n)
      let D = hA + hB;
      if (seat && p.seat_gap) D -= this.dent[2 * c] + this.dent[2 * c + 1] + this.dentE[c];
      const pen = D - g;
      let Fn = 0, active;
      if (aug) {
        const trial = st.lam[c] + kn * pen;
        active = (trial > 0 || this.locked[c]) && this.enabled[c];
        Fn = active ? (this.locked[c] ? trial : Math.max(trial, 0)) : 0;
      } else {
        active = (pen > 0 || this.locked[c]) && this.enabled[c];
        Fn = active ? kn * pen : 0.0;
      }
      activeA[c] = active ? 1 : 0;
      // tangential spring / stick–slip: decompose S_t = a tA + b tB
      const St = [S[0] - g * nx, S[1] - g * ny, S[2] - g * nz];
      const taa = tA[0] * tA[0] + tA[1] * tA[1] + tA[2] * tA[2], tbb = tB[0] * tB[0] + tB[1] * tB[1] + tB[2] * tB[2], tab = tA[0] * tB[0] + tA[1] * tB[1] + tA[2] * tB[2];
      const sa = St[0] * tA[0] + St[1] * tA[1] + St[2] * tA[2], sb = St[0] * tB[0] + St[1] * tB[1] + St[2] * tB[2];
      const det = taa * tbb - tab * tab, ok = det > 1e-8;
      const ca = ok ? (sa * tbb - sb * tab) / det : 0.0, cb = ok ? (sb * taa - sa * tab) / det : 0.0;
      const Ftv = Math.sqrt((kt * St[0]) ** 2 + (kt * St[1]) ** 2 + (kt * St[2]) ** 2);
      let muDir = p.mu;
      if (seat) muDir = this._seatFriction(c, St, nv, tA, tB);
      const limit = this.locked[c] ? Infinity : muDir * Math.abs(Fn);
      const inContact = active && ok;
      const slip = inContact && Ftv > limit;
      const free = !active && ok && this.enabled[c];
      let scale = 0;
      if (slip) scale = Math.min(Math.max(p.slip_rate, 1e-3), 1.0) * (1.0 - limit / Math.max(Ftv, 1e-12));
      if (free) scale = 1.0;
      const eps = Math.max(p.epsilon, 1e-9);
      const da = Math.min(Math.max(ca * scale, -eps), eps), db = Math.min(Math.max(cb * scale, -eps), eps);
      const moved = slip || free;
      if (moved) { st.u[2 * c] += da / Math.max(A.L[c], 1e-12); st.u[2 * c + 1] -= db / Math.max(B.L[c], 1e-12); }
      st.u[2 * c] = Math.min(Math.max(st.u[2 * c], 0.0), this.nseg[2 * c]);
      st.u[2 * c + 1] = Math.min(Math.max(st.u[2 * c + 1], 0.0), this.nseg[2 * c + 1]);
      if (slip) st.slip[c] += Math.sqrt((da * tA[0] + db * tB[0]) ** 2 + (da * tA[1] + db * tB[1]) ** 2 + (da * tA[2] + db * tB[2]) ** 2);
      st.stuck[c] = moved ? 0 : 1;
      const Se = active ? [St[0] - da * tA[0] - db * tB[0], St[1] - da * tA[1] - db * tB[1], St[2] - da * tA[2] - db * tB[2]] : [0, 0, 0];
      const fT = [kt * Se[0], kt * Se[1], kt * Se[2]];
      for (let k = 0; k < 3; k++) { FA[3 * c + k] = -Fn * nv[k] + fT[k]; FB[3 * c + k] = Fn * nv[k] - fT[k]; }
      this.Fn[c] = Fn;
      this.Ft[c] = Math.sqrt(fT[0] * fT[0] + fT[1] * fT[1] + fT[2] * fT[2]);
      this.state[c] = slip ? 1 : 0; this.gap[c] = g;
      slipA[c] = slip ? 1 : 0; daA[c] = da; dbA[c] = db;
      for (let k = 0; k < 3; k++) { rAx[3 * c + k] = rA[k]; rBx[3 * c + k] = rB[k]; }
      ePen[c] = active ? (aug ? Fn * Fn / Math.max(kn, 1e-300) : kn * pen * pen) : 0.0;
      eTan[c] = Se[0] * Se[0] + Se[1] * Se[1] + Se[2] * Se[2];
      tAs.push(tA); tBs.push(tB); Sts.push(St);
    }
    const cFace = this._classify(n);
    if (seat) for (let c = 0; c < K; c++) this._seatUpdate(st, c, slipA[c], daA[c], dbA[c], tAs[c], tBs[c], [n[3 * c], n[3 * c + 1], n[3 * c + 2]], this.Fn[c], cFace, activeA[c]);
    // distribute to the segment nodes with (1 − t, t), lever-arm torques on the segment frames
    for (let c = 0; c < K; c++) { const i = A.a[c], w = 1 - A.tp[c]; for (let k = 0; k < 3; k++) F[3 * i + k] += w * FA[3 * c + k]; }
    for (let c = 0; c < K; c++) { const i = A.b[c], w = A.tp[c]; for (let k = 0; k < 3; k++) F[3 * i + k] += w * FA[3 * c + k]; }
    for (let c = 0; c < K; c++) { const i = B.a[c], w = 1 - B.tp[c]; for (let k = 0; k < 3; k++) F[3 * i + k] += w * FB[3 * c + k]; }
    for (let c = 0; c < K; c++) { const i = B.b[c], w = B.tp[c]; for (let k = 0; k < 3; k++) F[3 * i + k] += w * FB[3 * c + k]; }
    const tf = p.torsion_factor;
    for (let c = 0; c < K; c++) {
      const s = A.seg[c], r = [rAx[3 * c], rAx[3 * c + 1], rAx[3 * c + 2]], f = [FA[3 * c], FA[3 * c + 1], FA[3 * c + 2]];
      T[3 * s] += tf * (r[1] * f[2] - r[2] * f[1]); T[3 * s + 1] += tf * (r[2] * f[0] - r[0] * f[2]); T[3 * s + 2] += tf * (r[0] * f[1] - r[1] * f[0]);
    }
    for (let c = 0; c < K; c++) {
      const s = B.seg[c], r = [rBx[3 * c], rBx[3 * c + 1], rBx[3 * c + 2]], f = [FB[3 * c], FB[3 * c + 1], FB[3 * c + 2]];
      T[3 * s] += tf * (r[1] * f[2] - r[2] * f[1]); T[3 * s + 1] += tf * (r[2] * f[0] - r[0] * f[2]); T[3 * s + 2] += tf * (r[0] * f[1] - r[1] * f[0]);
    }
    if (p.rot_friction) this._rotationalFriction(st, T, n, activeA, kt);
    this.energy = 0.5 * pairwiseSum(ePen) + 0.5 * kt * pairwiseSum(eTan) + (p.rot_friction ? this.rotEnergy : 0.0);
  }

  /** Uzawa step of the augmented Lagrangian: λ ← max(0, λ + k_n·pen) (bilateral for locked knots) = the current Fn */
  updateMultipliers(st) { for (let c = 0; c < this.K; c++) st.lam[c] = this.enabled[c] ? this.Fn[c] : 0; }
}

// ================================================================================================= Solver
/** structural deep copy (copy.deepcopy): class instances keep their prototype, typed arrays / arrays / maps / sets are
 *  copied, shared references stay shared (memo); objects in `memo` beforehand are kept as they are (the topology) */
export function deepClone(v, memo = new Map()) {
  if (v === null || typeof v !== 'object') return v;
  if (memo.has(v)) return memo.get(v);
  let out;
  if (ArrayBuffer.isView(v)) out = v instanceof DataView ? new DataView(v.buffer.slice(0)) : v.slice();
  else if (v instanceof ArrayBuffer) out = v.slice(0);
  else if (Array.isArray(v)) { out = new Array(v.length); memo.set(v, out); for (let i = 0; i < v.length; i++) out[i] = deepClone(v[i], memo); return out; }
  else if (v instanceof Map) { out = new Map(); memo.set(v, out); for (const [k, x] of v) out.set(deepClone(k, memo), deepClone(x, memo)); return out; }
  else if (v instanceof Set) { out = new Set(); memo.set(v, out); for (const x of v) out.add(deepClone(x, memo)); return out; }
  else {
    out = Object.create(Object.getPrototypeOf(v)); memo.set(v, out);
    for (const k of Reflect.ownKeys(v)) {
      const d = Object.getOwnPropertyDescriptor(v, k);
      if ('value' in d) d.value = deepClone(d.value, memo);
      Object.defineProperty(out, k, d);
    }
    return out;
  }
  memo.set(v, out);
  return out;
}

export class CpuSolver {
  constructor(scene, topo, { contacts = true } = {}) {
    this.scene = scene; this.topo = topo; this.useContacts = contacts;
    this.backend = 'cpu';
    this.st = buildState(topo);
    this.converged = false; this.kePrev = 0.0;
    this._buildGoals();
  }

  _buildGoals(contact = null, rest = null) {
    const sc = this.scene, st = this.st;
    const { stiff, act } = strandStiffness(sc, st);
    this.stiff = stiff;
    this.beam = new ActiveBeam(st, stiff, sc.solver.prestress, act, rest);
    this.loads = new Loads(sc, this.topo, st);
    this.anchors = new Anchors(sc, this.topo, st);
    this.beam.setClamps(st, this.anchors.clamps);
    if (contact) this.contact = contact;
    else {
      const secOf = (s) => sc.group(sc.strandGroup(s)).section, matOf = (s) => sc.group(sc.strandGroup(s)).material;
      this.contact = this.useContacts && st.K ? new PolylineContact(this.topo, st, secOf, sc.contact, matOf) : null;
      if (this.contact) this.contact.dentAfterDefault = (sc.solver.warmup_iter | 0) + (sc.contact.ramp_iter | 0);
    }
    if (this.contact) this.contact.p = sc.contact;
    const n = st.n, m = st.m;
    const kNode = new Float64Array(n), kSeg = new Float64Array(m);
    this.beam.addStiffness(st, kNode, kSeg);
    if (this.contact) this.contact.addStiffness(st, kNode, kSeg);
    for (let j = 0; j < m; j++) kNode[st.segI0[j]] += kSeg[j] / Math.max(st.l0[j], 1e-12) ** 2;
    for (let j = 0; j < m; j++) kNode[st.segI1[j]] += kSeg[j] / Math.max(st.l0[j], 1e-12) ** 2;
    const ms = +sc.solver.mass_scale;
    this.mass = new Float64Array(n); for (let i = 0; i < n; i++) this.mass[i] = ms * Math.max(kNode[i], 1e-12);
    this.kSegStatic = new Float64Array(m);
    this.beam.addStiffness(st, new Float64Array(n), this.kSegStatic);
    this.inertia = new Float64Array(m); for (let j = 0; j < m; j++) this.inertia[j] = ms * Math.max(kSeg[j], 1e-12);
    this.F = new Float64Array(3 * n); this.T = new Float64Array(3 * m);
  }

  reset() { this.st = buildState(this.topo); this.converged = false; this.kePrev = 0; this._buildGoals(); }

  /** independent copy (scene, state, goals, contact state); the topology is shared, it is never modified */
  clone() { return deepClone(this, new Map([[this.topo, this.topo]])); }

  /** re-read loads, anchors, stiffness and beam factors; keep positions, frames, contact state and the rest state
   *  (recomputed from the reference geometry X0 only when the activation changed) */
  rebuildGoals() {
    const st = this.st, sc = this.scene;
    let { act } = strandStiffness(sc, st);
    if (sc.solver.prestress === 'asbuilt') act = new Float64Array(act.length);
    else act = act.map(v => Math.min(Math.max(v, 0), 1));
    let rest;
    if (act.length === this.beam.activation.length && act.every((v, i) => Math.abs(v - this.beam.activation[i]) <= 1e-8 + 1e-5 * Math.abs(this.beam.activation[i]))) rest = st.Omega0.slice();
    else {
      const X = st.X.slice(), Q = st.Q.slice();
      st.X.set(st.X0);
      new ActiveBeam(st, { EA: new Float64Array(st.m).fill(1), EIx: new Float64Array(st.m).fill(1), EIy: new Float64Array(st.m).fill(1), GJ: new Float64Array(st.m).fill(1) }, sc.solver.prestress, act);
      rest = st.Omega0.slice();
      st.X.set(X); st.Q.set(Q);
    }
    this._buildGoals(this.contact, rest);
    this.converged = false; this.kePrev = 0.0;
  }

  step(nIter = 1) {
    const st = this.st, sp = this.scene.solver, n = st.n, m = st.m;
    const rampN = Math.max(this.scene.contact.ramp_iter | 0, 0);
    const F = this.F, T = this.T, Tpar = new Float64Array(m), tt = new Float64Array(3 * m);
    const Xprev = new Float64Array(3 * n), Qprev = new Float64Array(4 * m), dq = new Float64Array(4), qq = new Float64Array(4), q0 = new Float64Array(4);
    const kin = new Float64Array(3 * n), kinR = new Float64Array(m);
    let broke = false;
    for (let it = 0; it < nIter; it++) {
      if (this.contact) this.contact.ramp = rampN === 0 ? 1.0 : Math.min(1.0, 0.01 + 0.99 * st.iteration / rampN);
      F.fill(0); T.fill(0);
      this.beam.apply(st, F, T);
      // loads (const + frame-local point loads)
      const L = this.loads;
      L.total.set(L.const);
      if (L.pointLocal.length) {
        const nq = nodeQuats(st), R = new Float64Array(9);
        for (const [nid, Fv] of L.pointLocal) { toFrame(nq, 4 * nid, R); for (let k = 0; k < 3; k++) L.total[3 * nid + k] += R[k] * Fv[0] + R[3 + k] * Fv[1] + R[6 + k] * Fv[2]; }
      }
      for (let i = 0; i < 3 * n; i++) F[i] += L.total[i];
      if (this.contact) this.contact.apply(st, F, T);
      // split segment torques: parallel → twist, perpendicular → node couples
      const X = st.X;
      const cpl = new Float64Array(3 * m);
      for (let j = 0; j < m; j++) {
        const a = st.segI0[j], b = st.segI1[j];
        const ex = X[3 * b] - X[3 * a], ey = X[3 * b + 1] - X[3 * a + 1], ez = X[3 * b + 2] - X[3 * a + 2];
        const Lj = Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 1e-12);
        const tx = ex / Lj, ty = ey / Lj, tz = ez / Lj;
        tt[3 * j] = tx; tt[3 * j + 1] = ty; tt[3 * j + 2] = tz;
        const Tx = T[3 * j], Ty = T[3 * j + 1], Tz = T[3 * j + 2];
        Tpar[j] = Tx * tx + Ty * ty + Tz * tz;
        cpl[3 * j] = (Ty * tz - Tz * ty) / Lj; cpl[3 * j + 1] = (Tz * tx - Tx * tz) / Lj; cpl[3 * j + 2] = (Tx * ty - Ty * tx) / Lj;
      }
      for (let j = 0; j < m; j++) { const b = st.segI1[j]; F[3 * b] += cpl[3 * j]; F[3 * b + 1] += cpl[3 * j + 1]; F[3 * b + 2] += cpl[3 * j + 2]; }
      for (let j = 0; j < m; j++) { const a = st.segI0[j]; F[3 * a] -= cpl[3 * j]; F[3 * a + 1] -= cpl[3 * j + 1]; F[3 * a + 2] -= cpl[3 * j + 2]; }
      // anchors: reactions, then zero the fixed DOFs
      const An = this.anchors;
      An.reactions.fill(0);
      for (let i = 0; i < n; i++) if (An.fixedAny[i]) for (let k = 0; k < 3; k++) An.reactions[3 * i + k] = -F[3 * i + k];
      for (let i = 0; i < 3 * n; i++) if (!An.nodeFree[i]) F[i] = 0;
      for (let j = 0; j < m; j++) if (!An.twistFree[j]) Tpar[j] = 0;
      if (this.contact) {                               // twist inertia follows the current contact forces
        const kSeg = this.kSegStatic.slice();
        this.contact.addTwistStiffness(st, kSeg);
        for (let j = 0; j < m; j++) this.inertia[j] = sp.mass_scale * Math.max(kSeg[j], 1e-12);
      }
      // integrate
      Xprev.set(st.X); Qprev.set(st.Q);
      if (st.iteration < sp.warmup_iter && this.contact) { F.fill(0); st.V.fill(0); }
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) st.V[3 * i + k] += F[3 * i + k] / this.mass[i];
      for (let j = 0; j < m; j++) st.Wt[j] += Tpar[j] / this.inertia[j];
      if (sp.damping !== 1.0) { for (let i = 0; i < 3 * n; i++) st.V[i] *= sp.damping; for (let j = 0; j < m; j++) st.Wt[j] *= sp.damping; }
      for (let i = 0; i < 3 * n; i++) if (!An.nodeFree[i]) st.V[i] = 0;
      for (let j = 0; j < m; j++) if (!An.twistFree[j]) st.Wt[j] = 0;
      for (let i = 0; i < 3 * n; i++) st.X[i] += st.V[i];
      if (m) {
        for (let j = 0; j < m; j++) {
          const w = st.Wt[j];
          qexp(w * tt[3 * j], w * tt[3 * j + 1], w * tt[3 * j + 2], dq);
          q0[0] = st.Q[4 * j]; q0[1] = st.Q[4 * j + 1]; q0[2] = st.Q[4 * j + 2]; q0[3] = st.Q[4 * j + 3];
          qmul(dq, q0, qq); qnormalize(qq);
          st.Q[4 * j] = qq[0]; st.Q[4 * j + 1] = qq[1]; st.Q[4 * j + 2] = qq[2]; st.Q[4 * j + 3] = qq[3];
        }
        ActiveBeam.alignFrames(st);
      }
      for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) kin[3 * i + k] = this.mass[i] * st.V[3 * i + k] ** 2;
      for (let j = 0; j < m; j++) kinR[j] = this.inertia[j] * st.Wt[j] ** 2;
      let ke = 0.5 * (pairwiseSum(kin) + pairwiseSum(kinR));
      if (sp.kinetic_damping && ke < this.kePrev) {
        st.X.set(Xprev); st.Q.set(Qprev); st.V.fill(0); st.Wt.fill(0); ke = 0.0;
      }
      this.kePrev = ke;
      st.iteration += 1;
      // augmented Lagrangian: Uzawa update every al_every iterations once the ramp is over
      const c = this.contact;
      if (c && c.p.augmented && (c.p.al_every | 0) > 0 && st.iteration > rampN && (st.iteration - rampN) % (c.p.al_every | 0) === 0) c.updateMultipliers(st);
      const H = st.history;
      H.kinetic.push(ke);
      H.strain.push(this.beam.energy + (c ? c.energy : 0));
      H.bending.push(this.beam.energyParts.bending); H.torsion.push(this.beam.energyParts.torsion); H.stretch.push(this.beam.energyParts.stretch);
      H.contact.push(c ? c.energy : 0);
      // relative unbalanced force over the free nodes
      let fmax = 0;
      for (let i = 0; i < n; i++) if (An.allFree[i]) { const f = Math.sqrt(F[3 * i] ** 2 + F[3 * i + 1] ** 2 + F[3 * i + 2] ** 2); if (f > fmax) fmax = f; }
      const fref = this.forceRef();
      const resid = n ? fmax / fref : 0;
      st.residual = st.iteration > sp.warmup_iter ? resid : Infinity;
      H.residual.push(st.residual);
      if (st.residual <= sp.tol) { this.converged = true; broke = true; break; }
    }
    if (!broke) this.converged = st.residual <= sp.tol;          // Python: for … else
    return st;
  }

  forceRef() {
    const st = this.st;
    let a = 0; for (const v of this.loads.total) a = Math.max(a, Math.abs(v));
    let b = 0; for (const v of this.beam.N) b = Math.max(b, Math.abs(v));
    let c = 0; for (const v of this.beam.M) c = Math.max(c, Math.abs(v));
    let lmin = Infinity; for (const v of st.jLen) lmin = Math.min(lmin, v);
    c = st.p ? c / Math.max(lmin, 1e-12) : 0;
    let d = 0; if (this.contact) { d = -Infinity; for (const v of this.contact.Fn) d = Math.max(d, v); }
    return Math.max(a, b, c, d, 1e-12);
  }

  run(maxIter = null, callback = null) {
    maxIter = maxIter || this.scene.solver.max_iter;
    const every = Math.max(1, this.scene.solver.live_every | 0);
    let done = 0; this.converged = false;
    while (done < maxIter && !this.converged) {
      const k = Math.min(every, maxIter - done);
      this.step(k); done += k;
      if (callback) callback(this);
    }
    return this.st;
  }

  /** current geometry becomes the displacement origin; slip counters restart (fastened-structure protocol, step 2) */
  internalize() { this.st.X0 = this.st.X.slice(); this.st.slip.fill(0); }
  lockContacts(edges = null) { if (!this.contact) return; this.contact.lock(edges ? this._mask(edges) : null); this.scene.contact.locked = this.contact.locked.every(Boolean); }
  unlockContacts(edges = null) { if (!this.contact) return; this.contact.unlock(edges ? this._mask(edges) : null); this.scene.contact.locked = this.contact.locked.every(Boolean); }
  _mask(edges) { const s = new Set(edges); return Array.from(this.st.knotEdges, e => s.has(e)); }

  /** per-segment Mx My T N, per-knot contact layers, per-node displacement, reactions, energies */
  results() { return buildResults(this); }
}

/** results from any solver exposing st, beam {M, N, cNode, cM}, contact (or the GPU readback with the same fields) */
export function buildResults(sv) {
  const st = sv.st, n = st.n, m = st.m;
  const nodeM = new Float64Array(3 * n), has0 = new Uint8Array(n);
  for (let j = 0; j < st.p; j++) { const i = st.jNode[j]; nodeM[3 * i] = sv.beam.M[3 * j]; nodeM[3 * i + 1] = sv.beam.M[3 * j + 1]; nodeM[3 * i + 2] = sv.beam.M[3 * j + 2]; has0[i] = 1; }
  const hasB = has0.slice();
  sv.beam.cNode.forEach((i, c) => { nodeM[3 * i] = sv.beam.cM[2 * c]; nodeM[3 * i + 1] = sv.beam.cM[2 * c + 1]; hasB[i] = 1; });
  const segM = new Float64Array(3 * m);
  for (let j = 0; j < m; j++) for (let k = 0; k < 3; k++) segM[3 * j + k] = 0.5 * (nodeM[3 * st.segI0[j] + k] + nodeM[3 * st.segI1[j] + k]);
  for (const [cols, has] of [[[0, 1], hasB], [[2], has0]]) {
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      for (const k of cols) { if (!has[a]) segM[3 * j + k] = nodeM[3 * b + k]; }
    }
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      for (const k of cols) { if (!has[b]) segM[3 * j + k] = nodeM[3 * a + k]; }
    }
  }
  const seg = { Mx: new Float64Array(m), My: new Float64Array(m), T: new Float64Array(m), N: Float64Array.from(sv.beam.N) };
  for (let j = 0; j < m; j++) { seg.Mx[j] = segM[3 * j]; seg.My[j] = segM[3 * j + 1]; seg.T[j] = segM[3 * j + 2]; }
  const disp = new Float64Array(3 * n); for (let i = 0; i < 3 * n; i++) disp[i] = st.X[i] - st.X0[i];
  const knot = {};
  const c = sv.contact;
  if (c) {
    Object.assign(knot, { Fn: Float64Array.from(c.Fn), Ft: Float64Array.from(c.Ft), slip: Float64Array.from(st.slip), state: Float64Array.from(c.state), gap: Float64Array.from(c.gap),
      ctype: Float64Array.from(c.ctype), pressure: Float64Array.from(c.pressure), util: Float64Array.from(c.util), theta: Float64Array.from(c.theta),
      phiA: Float64Array.from({ length: st.K }, (_, k) => c.phi[2 * k]), phiB: Float64Array.from({ length: st.K }, (_, k) => c.phi[2 * k + 1]) });
    if (sv.scene.contact.seat_law !== 'off') knot.dent = Float64Array.from({ length: st.K }, (_, k) => c.dent[2 * k] + c.dent[2 * k + 1] + c.dentE[k]);
    if (sv.scene.contact.rot_friction) { knot.M_rot = Float64Array.from(c.Mrot); knot.rot_slip = Float64Array.from(c.rotSlip); }
    if (sv.scene.contact.augmented) knot.lambda = Float64Array.from(st.lam);
  } else for (const k of ['Fn', 'Ft', 'slip', 'state']) knot[k] = new Float64Array(st.K);
  const reactions = new Map();
  for (const [fk, nid] of st.nodeFrameIds) if (sv.anchors.fixedNodes.includes(nid)) reactions.set(fk, [sv.anchors.reactions[3 * nid], sv.anchors.reactions[3 * nid + 1], sv.anchors.reactions[3 * nid + 2], 0, 0, 0]);
  const H = st.history, last = (a) => (a.length ? a[a.length - 1] : 0);
  return {
    iterations: st.iteration, converged: sv.converged, residual: st.residual,
    segment: seg, knot, disp, reactions,
    energies: { strain: last(H.strain), kinetic: last(H.kinetic), beam: sv.beam.energy, bending: sv.beam.energyParts.bending, torsion: sv.beam.energyParts.torsion, stretch: sv.beam.energyParts.stretch, contact: c ? c.energy : 0 },
  };
}

export { fkey };
