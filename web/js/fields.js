// Fields along the laths on the solver state, shared by the colouring, the member diagrams, the discretisation and the
// Rhino export: node values of the result layers (segment layers averaged to the nodes, knot layers at the crossing
// nodes, |u|, shear = dM/ds), arc lengths, the discrete curvature of a centreline, np.interp / np.searchsorted.
// Python: export3dm._node_values, diagrams.node_values, discretize.discrete_curvature / geometry_from_app.
import { fkey } from './topology.js';

export const SEG_LAYERS = ['Mx', 'My', 'T', 'N'];

/** node / segment ranges of strand s: {i0, n, j0, m, closed} */
export function strandSpan(st, s) {
  const i0 = st.nodeOffset[s], n = st.nodeOffset[s + 1] - i0, j0 = st.segOffset[s], m = st.segOffset[s + 1] - j0;
  return { i0, n, j0, m, closed: m === n && n > 2 };
}

/** node pair (frame a, frame b) of every crossing in st.knotEdges order → Int32Array(2K) */
export function knotNodePairs(st, topo) {
  const out = new Int32Array(2 * st.K);
  for (let c = 0; c < st.K; c++) {
    const kn = topo.knots.get(st.knotEdges[c]);
    out[2 * c] = st.nodeFrameIds.get(fkey(kn.a[0], kn.a[1])); out[2 * c + 1] = st.nodeFrameIds.get(fkey(kn.b[0], kn.b[1]));
  }
  return out;
}

/** results with the internalized baseline subtracted (Results.compare: reactions are not carried over) */
export function deltaResults(r, base) {
  const sub = (a, b) => (b && b.length === a.length ? Float64Array.from(a, (v, i) => v - b[i]) : Float64Array.from(a));
  const segment = {}; for (const k of Object.keys(r.segment)) segment[k] = sub(r.segment[k], base.segment[k]);
  const knot = {}; for (const k of Object.keys(r.knot)) knot[k] = sub(r.knot[k], base.knot[k]);
  return { ...r, segment, knot, disp: sub(r.disp, base.disp), reactions: new Map(), delta: true };
}

/** node values of a layer on all nodes (Float64Array st.n). Segment layers: the mean of the adjacent segments (an end
 *  node takes its one segment; closed strands wrap); shear: d(moment)/ds between segment midpoints on X; knot layers: the
 *  crossing nodes of both strands, 0 elsewhere; disp: |u|. */
export function nodeValues(st, results, layer, { shear = false, X = st.X, knotPairs = null } = {}) {
  const out = new Float64Array(st.n);
  if (!results) return out;
  if (layer === 'disp') { const D = results.disp; for (let i = 0; i < st.n; i++) out[i] = Math.hypot(D[3 * i], D[3 * i + 1], D[3 * i + 2]); return out; }
  if (SEG_LAYERS.includes(layer)) {
    const A = results.segment[layer];
    for (let s = 0; s < st.S; s++) {
      const { i0, n, j0, m, closed } = strandSpan(st, s);
      if (n < 2 || !(m === n - 1 || closed)) continue;
      if (shear) {
        if (n < 3) continue;
        const mid = (j, c) => 0.5 * (X[3 * st.segI0[j0 + j] + c] + X[3 * st.segI1[j0 + j] + c]);
        const ds = (j1, j2) => Math.max(Math.hypot(mid(j2, 0) - mid(j1, 0), mid(j2, 1) - mid(j1, 1), mid(j2, 2) - mid(j1, 2)), 1e-12);
        if (closed) for (let i = 0; i < n; i++) { const jm = (i - 1 + m) % m; out[i0 + i] = (A[j0 + i] - A[j0 + jm]) / ds(jm, i); }
        else {
          for (let i = 1; i < n - 1; i++) out[i0 + i] = (A[j0 + i] - A[j0 + i - 1]) / ds(i - 1, i);
          out[i0] = out[i0 + 1]; out[i0 + n - 1] = out[i0 + n - 2];
        }
      } else if (closed) for (let i = 0; i < n; i++) out[i0 + i] = 0.5 * (A[j0 + (i - 1 + m) % m] + A[j0 + i]);
      else {
        for (let j = 0; j < m; j++) { out[i0 + j] += A[j0 + j]; out[i0 + j + 1] += A[j0 + j]; }
        for (let i = 1; i < n - 1; i++) out[i0 + i] *= 0.5;
      }
    }
    return out;
  }
  const K = results.knot[layer]; if (!K || !knotPairs) return out;
  for (let c = 0; c < K.length; c++) { out[knotPairs[2 * c]] = K[c]; out[knotPairs[2 * c + 1]] = K[c]; }
  return out;
}

/** arc length at the nodes of strand s on positions X (Float64Array n) */
export function arcLengths(X, i0, n) {
  const S = new Float64Array(n);
  for (let i = 1; i < n; i++) { const a = 3 * (i0 + i - 1), b = 3 * (i0 + i); S[i] = S[i - 1] + Math.hypot(X[b] - X[a], X[b + 1] - X[a + 1], X[b + 2] - X[a + 2]); }
  return S;
}

/** |κ| at the nodes of a polyline: |κb| / l̄ with κb = 2 e0×e1 / (|e0||e1| + e0·e1); 0 at the ends */
export function discreteCurvature(X, i0, n) {
  const k = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    const a = 3 * (i0 + i - 1), b = 3 * (i0 + i), c = 3 * (i0 + i + 1);
    const e0x = X[b] - X[a], e0y = X[b + 1] - X[a + 1], e0z = X[b + 2] - X[a + 2], e1x = X[c] - X[b], e1y = X[c + 1] - X[b + 1], e1z = X[c + 2] - X[b + 2];
    const l0 = Math.hypot(e0x, e0y, e0z), l1 = Math.hypot(e1x, e1y, e1z), den = Math.max(l0 * l1 + e0x * e1x + e0y * e1y + e0z * e1z, 1e-12);
    const cx = e0y * e1z - e0z * e1y, cy = e0z * e1x - e0x * e1z, cz = e0x * e1y - e0y * e1x;
    k[i] = 2 * Math.hypot(cx, cy, cz) / den / Math.max(0.5 * (l0 + l1), 1e-12);
  }
  return k;
}

/** np.searchsorted(a, v, side) on an ascending array */
export function searchsorted(a, v, side = 'left') {
  let lo = 0, hi = a.length;
  if (side === 'left') while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
  else while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] <= v) lo = mid + 1; else hi = mid; }
  return lo;
}

/** np.interp(x, xp, fp): linear, clamped to the end values */
export function interp(x, xp, fp) {
  const n = xp.length;
  if (!n) return NaN;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  const j = searchsorted(xp, x, 'right') - 1, d = xp[j + 1] - xp[j];
  if (d <= 0) return fp[j + 1];
  const t = (x - xp[j]) / d;
  return fp[j] + t * (fp[j + 1] - fp[j]);
}

/** a point at arc length s along strand nodes (X, i0) with arc lengths S */
export function pointAt(X, i0, S, s) {
  return [0, 1, 2].map(c => {
    const n = S.length;
    if (s <= S[0]) return X[3 * i0 + c];
    if (s >= S[n - 1]) return X[3 * (i0 + n - 1) + c];
    const j = searchsorted(S, s, 'right') - 1, d = S[j + 1] - S[j], t = d > 0 ? (s - S[j]) / d : 1;
    return X[3 * (i0 + j) + c] + t * (X[3 * (i0 + j + 1) + c] - X[3 * (i0 + j) + c]);
  });
}

/** Python's '%.{p}g' (optionally with a sign): 4 significant digits, trailing zeros stripped, e±XX exponents */
export function fmtG(v, p = 4, plus = false) {
  if (v == null || Number.isNaN(v)) return 'nan';
  if (!Number.isFinite(v)) return (v > 0 ? (plus ? '+' : '') + 'inf' : '-inf');
  const sign = v < 0 || Object.is(v, -0) ? '-' : plus ? '+' : '';
  const a = Math.abs(v);
  if (a === 0) return sign + '0';
  const ex = a.toExponential(p - 1), exp = +ex.slice(ex.indexOf('e') + 1);
  let s;
  if (exp < -4 || exp >= p) { let mant = ex.slice(0, ex.indexOf('e')); if (mant.includes('.')) mant = mant.replace(/\.?0+$/, ''); s = `${mant}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`; }
  else { s = a.toFixed(Math.max(0, p - 1 - exp)); if (s.includes('.')) s = s.replace(/\.?0+$/, ''); }
  return sign + s;
}
