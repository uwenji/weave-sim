// Stability of the current state: the eigenvalue nearest zero of the tangent stiffness and the number of unstable
// (negative) modes — the indicator of geometric failure (limit point, bifurcation, lateral–torsional buckling) for the
// Monitor's stability group.
//
//   K  = −∂g/∂q   the tangent stiffness: g the generalised forces of the solver (rod + contact + loads, with the
//                 contacts' stick / slip return mapping and multipliers frozen at the current state), q the free
//                 degrees of freedom (node translations and the 3 body-axis rotations of each segment frame)
//   A  = M^-½ K_s M^-½   K_s = ½(K + Kᵀ), M the dynamic-relaxation masses (so λ(A) is the squared frequency of the
//                 relaxation dynamics: λ → 0 at a limit point or bifurcation, λ < 0 unstable)
// K is assembled column-by-column from central differences of g with a distance-2 colouring of the stencil graph
// (a few hundred force evaluations), A is factorised by a sparse LDLᵀ (up-looking, after T. Davis's LDL) on a geometric
// nested-dissection ordering: the number of negative pivots is the number of negative eigenvalues (Sylvester's law of
// inertia), and inverse iteration on the factor gives the eigenvalue nearest zero.
import { toFrame, qexp, qmul, qnormalize } from './quat.js';
import { Rng } from './rng.js';

/** the free DOFs of a solver: [{kind: 0 node xyz | 1 rotation about a body axis, i, k, mass}] */
function dofList(sv) {
  const st = sv.st, An = sv.anchors, out = [];
  for (let i = 0; i < st.n; i++) for (let k = 0; k < 3; k++) if (An.nodeFree[3 * i + k]) out.push({ kind: 0, i, k, mass: sv.mass[i] });
  for (let j = 0; j < st.m; j++) for (let k = 0; k < 3; k++) out.push({ kind: 1, i: j, k, mass: sv.inertia3[3 * j + k] });
  return out;
}

/** generalised forces g (one per DOF) of the solver's current state, the contact state frozen (restored after) */
function makeEvaluator(sv, dofs) {
  const st = sv.st, n = st.n, m = st.m;
  const F = new Float64Array(3 * n), T = new Float64Array(3 * m), R = new Float64Array(9);
  // every typed array of the state and of the contact goal is restored after an evaluation (bookmarks, slips, dents …)
  const holders = [st, sv.contact].filter(Boolean);
  const saved = holders.map(h => Object.entries(h).filter(([, v]) => ArrayBuffer.isView(v) && !(v instanceof DataView)).map(([k, v]) => [k, v.slice()]));
  const scalars = sv.contact ? { penN: sv.contact.penN, energy: sv.contact.energy, rotEnergy: sv.contact.rotEnergy } : null;
  const restore = () => {
    holders.forEach((h, hi) => { for (const [k, v] of saved[hi]) h[k].set(v); });
    if (scalars) Object.assign(sv.contact, scalars);
  };
  const g = new Float64Array(dofs.length);
  const evaluate = () => {
    F.fill(0); T.fill(0);
    sv.beam.apply(st, F, T);
    const L = sv.loads; for (let i = 0; i < 3 * n; i++) F[i] += L.const[i];
    if (sv.contact) sv.contact.apply(st, F, T);
    for (let d = 0; d < dofs.length; d++) {
      const q = dofs[d];
      if (q.kind === 0) g[d] = F[3 * q.i + q.k];
      else { toFrame(st.Q, 4 * q.i, R); g[d] = R[3 * q.k] * T[3 * q.i] + R[3 * q.k + 1] * T[3 * q.i + 1] + R[3 * q.k + 2] * T[3 * q.i + 2]; }
    }
    restore();
    return g.slice();
  };
  return { evaluate, restore, saved };
}

/** influence sets (structural stencil, conservative): DOFs of the same strand within ±W nodes / segments, and all DOFs
 *  of both strands within ±W of each crossing's contact segment */
function stencil(sv, dofs, W = 3) {
  const st = sv.st, nd = dofs.length;
  const nodeDofs = Array.from({ length: st.n }, () => []), segDofs = Array.from({ length: st.m }, () => []);
  dofs.forEach((q, d) => (q.kind === 0 ? nodeDofs[q.i] : segDofs[q.i]).push(d));
  // strand-local index of nodes and segments
  const nodeLoc = new Int32Array(st.n), segLoc = new Int32Array(st.m);
  for (let s = 0; s < st.S; s++) { for (let i = st.nodeOffset[s]; i < st.nodeOffset[s + 1]; i++) nodeLoc[i] = i - st.nodeOffset[s]; for (let j = st.segOffset[s]; j < st.segOffset[s + 1]; j++) segLoc[j] = j - st.segOffset[s]; }
  const closed = Array.from({ length: st.S }, (_, s) => st.segOffset[s + 1] - st.segOffset[s] === st.nodeOffset[s + 1] - st.nodeOffset[s]);
  const window = (s, loc, w) => {                 // DOFs of strand s within local index loc ± w (nodes and segments)
    const out = [], nn = st.nodeOffset[s + 1] - st.nodeOffset[s], ns = st.segOffset[s + 1] - st.segOffset[s];
    for (let o = -w; o <= w; o++) {
      let a = loc + o; if (closed[s]) a = ((a % nn) + nn) % nn; if (a >= 0 && a < nn) out.push(...nodeDofs[st.nodeOffset[s] + a]);
      let b = loc + o; if (closed[s]) b = ((b % ns) + ns) % ns; if (b >= 0 && b < ns) out.push(...segDofs[st.segOffset[s] + b]);
    }
    return out;
  };
  const N = Array.from({ length: nd }, () => new Set());
  for (let d = 0; d < nd; d++) {
    const q = dofs[d], s = q.kind === 0 ? st.nodeStrand[q.i] : st.segStrand[q.i], loc = q.kind === 0 ? nodeLoc[q.i] : segLoc[q.i];
    for (const e of window(s, loc, W)) N[d].add(e);
  }
  const c0 = sv.contact;
  if (c0) for (let c = 0; c < st.K; c++) {
    const set = [];
    for (let side = 0; side < 2; side++) {
      const s = c0.strand[2 * c + side], u = st.u[2 * c + side];
      set.push(...window(s, Math.floor(u), W + 2));
    }
    for (const a of set) for (const b of set) N[a].add(b);
  }
  return N.map(x => Int32Array.from(x));
}

/** greedy distance-2 colouring: columns sharing a colour never touch a common row */
function colour(N) {
  const nd = N.length, col = new Int32Array(nd).fill(-1), mark = new Int32Array(nd + 1).fill(-1);
  let ncol = 0;
  for (let d = 0; d < nd; d++) {
    for (const r of N[d]) for (const e of N[r]) if (col[e] >= 0) mark[col[e]] = d;
    let c = 0; while (mark[c] === d) c++;
    col[d] = c; if (c + 1 > ncol) ncol = c + 1;
  }
  return { col, ncol };
}

/** geometric nested dissection on the DOF positions (one-sided separators from the stencil graph) → permutation */
function nestedDissection(pos, N, leaf = 48) {
  const nd = pos.length / 3, perm = [], side = new Int8Array(nd);
  const rec = (ids) => {
    if (ids.length <= leaf) { perm.push(...ids); return; }
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const d of ids) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], pos[3 * d + k]); hi[k] = Math.max(hi[k], pos[3 * d + k]); }
    const ax = [0, 1, 2].reduce((a, k) => (hi[k] - lo[k] > hi[a] - lo[a] ? k : a), 0);
    const sorted = ids.slice().sort((a, b) => pos[3 * a + ax] - pos[3 * b + ax]);
    const half = sorted.length >> 1, L = sorted.slice(0, half), Rr = sorted.slice(half);
    for (const d of L) side[d] = 1; for (const d of Rr) side[d] = 2;
    const sep = [], R2 = [];
    for (const d of Rr) { let touch = false; for (const e of N[d]) if (side[e] === 1) { touch = true; break; } (touch ? sep : R2).push(d); }
    for (const d of ids) side[d] = 0;
    if (!R2.length || sep.length > 0.6 * ids.length) { perm.push(...ids); return; }   // no useful cut
    rec(L); rec(R2); perm.push(...sep);
  };
  rec([...Array(nd).keys()]);
  return Int32Array.from(perm);
}

/** sparse LDLᵀ of the symmetric CSC matrix (Ap, Ai, Ax) under permutation P (after T. Davis, LDL) */
function ldl(n, Ap, Ai, Ax, P) {
  const Pinv = new Int32Array(n); for (let k = 0; k < n; k++) Pinv[P[k]] = k;
  const Parent = new Int32Array(n), Lnz = new Int32Array(n), Flag = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    Parent[k] = -1; Flag[k] = k; Lnz[k] = 0;
    const kk = P[k];
    for (let p = Ap[kk]; p < Ap[kk + 1]; p++) {
      let i = Pinv[Ai[p]];
      if (i < k) for (; Flag[i] !== k; i = Parent[i]) { if (Parent[i] === -1) Parent[i] = k; Lnz[i]++; Flag[i] = k; }
    }
  }
  const Lp = new Int32Array(n + 1); for (let k = 0; k < n; k++) Lp[k + 1] = Lp[k] + Lnz[k];
  const nnz = Lp[n];
  if (nnz > 4e7) throw new Error(`factor too large (${(nnz / 1e6).toFixed(0)} M entries)`);
  const Li = new Int32Array(nnz), Lx = new Float64Array(nnz), D = new Float64Array(n), Y = new Float64Array(n), Pattern = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    Y[k] = 0; let top = n; Flag[k] = k; Lnz[k] = 0;
    const kk = P[k];
    for (let p = Ap[kk]; p < Ap[kk + 1]; p++) {
      let i = Pinv[Ai[p]];
      if (i <= k) {
        Y[i] += Ax[p];
        let len = 0;
        for (; Flag[i] !== k; i = Parent[i]) { Pattern[len++] = i; Flag[i] = k; }
        while (len > 0) Pattern[--top] = Pattern[--len];
      }
    }
    D[k] = Y[k]; Y[k] = 0;
    for (; top < n; top++) {
      const i = Pattern[top], yi = Y[i]; Y[i] = 0;
      const p2 = Lp[i] + Lnz[i];
      for (let p = Lp[i]; p < p2; p++) Y[Li[p]] -= Lx[p] * yi;
      const lki = yi / D[i];
      D[k] -= lki * yi;
      Li[p2] = k; Lx[p2] = lki; Lnz[i]++;
    }
    if (D[k] === 0) D[k] = 1e-300;                 // exactly singular: keep going (the count and the shift handle it)
  }
  const solve = (b) => {                          // (P A Pᵀ) x = P b  →  x
    const x = new Float64Array(n);
    for (let k = 0; k < n; k++) x[k] = b[P[k]];
    for (let j = 0; j < n; j++) for (let p = Lp[j]; p < Lp[j + 1]; p++) x[Li[p]] -= Lx[p] * x[j];
    for (let j = 0; j < n; j++) x[j] /= D[j];
    for (let j = n - 1; j >= 0; j--) for (let p = Lp[j]; p < Lp[j + 1]; p++) x[j] -= Lx[p] * x[Li[p]];
    const out = new Float64Array(n); for (let k = 0; k < n; k++) out[P[k]] = x[k];
    return out;
  };
  let neg = 0; for (let k = 0; k < n; k++) if (D[k] < 0) neg++;
  return { neg, nnz, solve };
}

/** stability of the solver's current state → {lam, neg, ndof, colours, evaluations, nnz, ms, vec} (lam: eigenvalue of
 *  the mass-scaled tangent stiffness nearest zero; neg: number of negative eigenvalues) */
export function stability(sv0, { iters = 40, tol = 1e-9 } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
  const sv = sv0.clone();
  const st = sv.st;
  const dofs = dofList(sv), nd = dofs.length;
  if (!nd) return { lam: NaN, neg: 0, ndof: 0 };
  const { evaluate } = makeEvaluator(sv, dofs);
  // perturbation sizes: 1e-6 of the model size (translations), 1e-6 rad (rotations)
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < st.n; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], st.X[3 * i + k]); hi[k] = Math.max(hi[k], st.X[3 * i + k]); }
  const size = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), 1e-9), hT = 1e-6 * size, hR = 1e-6;
  const N = stencil(sv, dofs), { col, ncol } = colour(N);
  const byCol = Array.from({ length: ncol }, () => []); for (let d = 0; d < nd; d++) byCol[col[d]].push(d);
  const X0 = st.X.slice(), Q0 = st.Q.slice(), dq = new Float64Array(4), qq = new Float64Array(4), q0 = new Float64Array(4);
  const perturb = (ds, sgn) => {
    st.X.set(X0); st.Q.set(Q0);
    for (const d of ds) {
      const q = dofs[d];
      if (q.kind === 0) st.X[3 * q.i + q.k] += sgn * hT;
      else if (q.kind === 1) { const v = [0, 0, 0]; v[q.k] = sgn * hR; qexp(v[0], v[1], v[2], dq); q0.set(st.Q.subarray(4 * q.i, 4 * q.i + 4)); qmul(q0, dq, qq); qnormalize(qq); st.Q.set(qq, 4 * q.i); }
    }
  };
  // columns of K = −∂g/∂q from central differences, colour by colour
  const rows = [], cols = [], vals = [];
  for (let c = 0; c < ncol; c++) {
    const ds = byCol[c];
    perturb(ds, +1); const gp = evaluate();
    perturb(ds, -1); const gm = evaluate();
    for (const d of ds) {
      const h = dofs[d].kind === 0 ? hT : hR;
      for (const r of N[d]) { const v = -(gp[r] - gm[r]) / (2 * h); if (v !== 0) { rows.push(r); cols.push(d); vals.push(v); } }
    }
  }
  st.X.set(X0); st.Q.set(Q0);
  // A = M^-½ ½(K + Kᵀ) M^-½ in symmetric CSC (both triangles)
  const sc = Float64Array.from(dofs, q => 1 / Math.sqrt(Math.max(q.mass, 1e-300)));
  const map = new Map();
  for (let e = 0; e < vals.length; e++) {
    const r = rows[e], c = cols[e], v = 0.5 * vals[e] * sc[r] * sc[c];
    const k1 = r * nd + c, k2 = c * nd + r;
    map.set(k1, (map.get(k1) || 0) + v); map.set(k2, (map.get(k2) || 0) + v);
  }
  // CSC by the key's first index (the matrix is symmetric, so row and column roles are interchangeable)
  const counts = new Int32Array(nd + 1); for (const k of map.keys()) counts[Math.floor(k / nd) + 1]++;
  for (let i = 0; i < nd; i++) counts[i + 1] += counts[i];
  const Ap = counts.slice(), Ai = new Int32Array(map.size), Ax = new Float64Array(map.size), fill = Ap.slice(0, nd);
  for (const [k, v] of map) { const c = Math.floor(k / nd), r = k - c * nd; Ai[fill[c]] = r; Ax[fill[c]++] = v; }
  const matvec = (x) => { const y = new Float64Array(nd); for (let c = 0; c < nd; c++) for (let p = Ap[c]; p < Ap[c + 1]; p++) y[Ai[p]] += Ax[p] * x[c]; return y; };
  // ordering (DOF positions: nodes; segment midpoints for rotations) and factorisation
  const pos = new Float64Array(3 * nd);
  dofs.forEach((q, d) => {
    if (q.kind === 0) for (let k = 0; k < 3; k++) pos[3 * d + k] = X0[3 * q.i + k];
    else { const a = st.segI0[q.i], b = st.segI1[q.i]; for (let k = 0; k < 3; k++) pos[3 * d + k] = 0.5 * (X0[3 * a + k] + X0[3 * b + k]); }
  });
  const P = nestedDissection(pos, N);
  const F = ldl(nd, Ap, Ai, Ax, P);
  // inverse iteration → the eigenvalue nearest zero (Rayleigh quotient) and its mode
  const rng = new Rng(12345);
  let x = Float64Array.from({ length: nd }, () => rng.uniform() - 0.5), lam = NaN;
  const norm = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  for (let it = 0; it < iters; it++) {
    const y = F.solve(x), ny = norm(y); for (let i = 0; i < nd; i++) x[i] = y[i] / ny;
    const Ax_ = matvec(x); let rq = 0; for (let i = 0; i < nd; i++) rq += x[i] * Ax_[i];
    if (Number.isFinite(lam) && Math.abs(rq - lam) <= tol * Math.max(Math.abs(rq), 1e-300)) { lam = rq; break; }
    lam = rq;
  }
  const ms = (typeof performance !== 'undefined' ? performance : Date).now() - t0;
  return { lam, neg: F.neg, ndof: nd, colours: ncol, evaluations: 2 * ncol, nnz: F.nnz, ms, vec: x, dofs };
}
