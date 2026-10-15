// Member diagrams (port of weavesim/app/diagrams.py, Autocalcs style): a filled, hatched diagram hung on each lath in
// the plane the quantity acts in, an exaggerated deflected shape, and force arrows at crossings and supports, with
// value labels at the extremes.
//
//   Diagram   drawn along          sign colour
//   Mx        frame Y (normal)     flatwise bending moment
//   My        frame X (width)      edgewise bending moment
//   Vy / Vx   frame Y / X          shear = dM/ds of Mx / My (finite difference between segments)
//   T         frame X              torsion
//   N         frame Y              axial (tension +)
//   defl      —                    undeformed centrelines + deformed shape × exaggeration
//   Fn        —                    contact normal forces at the crossings (arrows)
//   R         —                    support reactions (arrows)
//   curv      frame Y              centreline curvature |κ| (geometry only, no results needed)
//
// buildDiagram() returns plain geometry (triangles with colours, line segments, arrows, coloured polylines, labels and
// the info line); the view draws it. Differences from the Python program: the filled strip is split where the value
// changes sign (each part one colour), and closed strands get a diagram too (Python leaves them at zero).
import { nodeValues, strandSpan, discreteCurvature, fmtG } from './fields.js';
import { DIAG_POS, DIAG_NEG, DIAG_UNDEFORMED, LOAD, ANCHOR_FULL, cmapFor, lutColor } from './colormap.js';

export const DIAGRAMS = ['none', 'defl', 'Mx', 'My', 'Vy', 'Vx', 'T', 'N', 'Fn', 'R', 'curv'];
export const DIAGRAM_LABELS = {
  none: 'off', defl: 'deflected shape', Mx: 'Mx flatwise moment', My: 'My edgewise moment', Vy: 'Vy shear (flatwise, dMx/ds)',
  Vx: 'Vx shear (edgewise, dMy/ds)', T: 'torsion', N: 'axial force', Fn: 'contact normal forces', R: 'support reactions',
  curv: 'centreline curvature |κ| (geometry only)',
};
export const DIAGRAM_AXIS = { Mx: 1, Vy: 1, My: 0, Vx: 0, T: 0, N: 1, curv: 1 };
export const HEIGHT_FRACTION = 0.06;         // auto scale: |max value| → this fraction of the model size
export const DEFL_FRACTION = 0.05;           // auto exaggeration: max |disp| → this fraction of the model size

export function diagramUnit(units, d) {
  if (d === 'curv') return `1/${units.length}`;
  if (['Mx', 'My', 'T'].includes(d)) return `${units.force}·${units.length}`;
  if (['N', 'Vx', 'Vy', 'Fn', 'R'].includes(d)) return units.force;
  if (d === 'defl') return units.length;
  return '';
}

/** node values of a member diagram on all nodes (Float64Array st.n); X = the solved positions (shear, curvature) */
export function memberValues(d, st, results, X = st.X) {
  if (d === 'curv') { const out = new Float64Array(st.n); for (let s = 0; s < st.S; s++) { const { i0, n } = strandSpan(st, s); out.set(discreteCurvature(X, i0, n), i0); } return out; }
  if (d === 'Vx' || d === 'Vy') return nodeValues(st, results, d === 'Vx' ? 'My' : 'Mx', { shear: true, X });
  return nodeValues(st, results, d);
}

/**
 * ctx: {st, topo, results (already delta'd if asked), P (display positions 3n), R (node frames 9n), X0 (reference
 * positions), size (model size), units, contact (solver contact: mid), deform, opts: {scale, labels: 0 none · 1 extremes ·
 * 2 per strand, palette, zoneMask: (s) → Uint8Array | null | undefined (joint zones on) }}
 * → {kind, tris, triCol, lines, lineCol, arrows: {from, vec, color}, polylines: [{pts, col}], labels: [{p, text}], info}
 */
export function buildDiagram(d, ctx) {
  const out = { kind: d, tris: null, triCol: null, lines: null, arrows: null, polylines: null, labels: [], info: '' };
  if (d === 'none' || !ctx.st) return out;
  if (!ctx.results && d !== 'curv') { out.info = 'run the solver to populate'; return out; }
  if (d in DIAGRAM_AXIS) member(d, ctx, out);
  else if (d === 'defl') deflection(ctx, out);
  else if (d === 'Fn') contactForces(ctx, out);
  else if (d === 'R') reactions(ctx, out);
  return out;
}

function member(d, ctx, out) {
  const { st, P, R, opts } = ctx, col = DIAGRAM_AXIS[d];
  const v = memberValues(d, st, ctx.results, st.X);
  if (opts.zoneMask) for (let s = 0; s < st.S; s++) {
    const { i0, n } = strandSpan(st, s), m = opts.zoneMask(s);
    for (let i = 0; i < n; i++) if (!m || !m[i]) v[i0 + i] = 0;
  }
  let vmax = 0; for (const x of v) vmax = Math.max(vmax, Math.abs(x));
  if (!(vmax > 0)) { out.info = `${DIAGRAM_LABELS[d]}: all zero`; return; }
  const amp = HEIGHT_FRACTION * ctx.size / vmax * opts.scale;
  const tip = (i) => [0, 1, 2].map(c => P[3 * i + c] + R[9 * i + 3 * col + c] * v[i] * amp);
  const base = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
  const tris = [], tcol = [], lines = [], labels = [];
  const tri = (a, b, c, rgb) => { tris.push(...a, ...b, ...c); for (let k = 0; k < 3; k++) tcol.push(...rgb); };
  for (let s = 0; s < st.S; s++) {
    const { i0, n, j0, m } = strandSpan(st, s);
    if (n < 2) continue;
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j0 + j], b = st.segI1[j0 + j], va = v[a], vb = v[b];
      if (va === 0 && vb === 0) continue;
      const Pa = base(a), Pb = base(b), Qa = tip(a), Qb = tip(b);
      if (va * vb < 0) {                              // split at the zero: one colour per part
        const t = va / (va - vb), Z = Pa.map((x, c) => x + t * (Pb[c] - x));
        tri(Pa, Z, Qa, va > 0 ? DIAG_POS : DIAG_NEG); tri(Z, Pb, Qb, vb > 0 ? DIAG_POS : DIAG_NEG);
      } else {
        const rgb = (va + vb) >= 0 ? DIAG_POS : DIAG_NEG;
        tri(Pa, Pb, Qb, rgb); tri(Pa, Qb, Qa, rgb);
      }
      lines.push(...Qa, ...Qb);                       // outline along the tips
    }
    for (let i = 0; i < n; i++) if (v[i0 + i] !== 0) lines.push(...base(i0 + i), ...tip(i0 + i));   // hatch: base → tip at every node
    if (opts.labels === 2) { let ib = i0; for (let i = i0; i < i0 + n; i++) if (Math.abs(v[i]) > Math.abs(v[ib])) ib = i; if (v[ib] !== 0) labels.push({ p: tip(ib), text: fmtG(v[ib], 4, true) }); }
  }
  let imax = 0, imin = 0; for (let i = 1; i < st.n; i++) { if (v[i] > v[imax]) imax = i; if (v[i] < v[imin]) imin = i; }
  if (opts.labels >= 1) { labels.push({ p: tip(imax), text: fmtG(v[imax], 4, true) }); if (v[imin] < 0) labels.push({ p: tip(imin), text: fmtG(v[imin], 4, true) }); }
  out.tris = Float32Array.from(tris); out.triCol = Float32Array.from(tcol); out.lines = Float32Array.from(lines); out.labels = labels;
  out.values = v;
  out.info = `${DIAGRAM_LABELS[d]}  max ${fmtG(v[imax], 4, true)}  min ${fmtG(v[imin], 4, true)} ${diagramUnit(ctx.units, d)}  ·  height ${(HEIGHT_FRACTION * opts.scale * 100).toFixed(0)} % of model at |max|`;
}

function deflection(ctx, out) {
  const { st, results: r, opts } = ctx, X0 = ctx.X0 || st.X0, D = r.disp;
  let dmax = 0; for (let i = 0; i < st.n; i++) dmax = Math.max(dmax, Math.hypot(D[3 * i], D[3 * i + 1], D[3 * i + 2]));
  const size = ctx.size;
  let k = (dmax > 0 ? DEFL_FRACTION * size / dmax : 1.0) * opts.scale;
  if (opts.scale <= 1.0 && dmax > 0 && dmax > DEFL_FRACTION * size) k = Math.max(k, 1.0);
  const cmap = cmapFor(opts.palette || 'default', false), polylines = [];
  let ibest = 0, mbest = -1;
  for (let s = 0; s < st.S; s++) {
    const { i0, n, closed } = strandSpan(st, s);
    if (n < 2) continue;
    const idx = Array.from({ length: n + (closed ? 1 : 0) }, (_, q) => i0 + (q % n));
    const p0 = [], p1 = [], c1 = [];
    for (const i of idx) {
      const mag = Math.hypot(D[3 * i], D[3 * i + 1], D[3 * i + 2]);
      if (mag > mbest) { mbest = mag; ibest = i; }
      for (let c = 0; c < 3; c++) { p0.push(X0[3 * i + c]); p1.push(X0[3 * i + c] + k * D[3 * i + c]); }
      c1.push(...lutColor(cmap, mag / Math.max(dmax, 1e-12)));
    }
    polylines.push({ pts: Float32Array.from(p0), col: null, color: DIAG_UNDEFORMED, width: 1.4 });
    polylines.push({ pts: Float32Array.from(p1), col: Float32Array.from(c1), width: 2.6 });
  }
  out.polylines = polylines;
  if (opts.labels) out.labels = [{ p: [0, 1, 2].map(c => X0[3 * ibest + c] + k * D[3 * ibest + c]), text: `${fmtG(mbest, 4)} ${diagramUnit(ctx.units, 'defl')}` }];
  out.legend = { cmap, lo: 0, hi: Math.max(dmax, 1e-12), title: '|disp|' };
  out.info = `deflected shape × ${fmtG(k, 3)}  ·  |disp| max ${fmtG(dmax, 4)} ${diagramUnit(ctx.units, 'defl')}`;
}

function contactForces(ctx, out) {
  const { st, results: r, opts, contact } = ctx;
  const Fn = r.knot && r.knot.Fn;
  if (!contact || !Fn) { out.info = 'contact forces: no contact goal'; return; }
  let fmax = 0, cnt = 0, ib = 0; for (let c = 0; c < Fn.length; c++) { if (Fn[c] > fmax) { fmax = Fn[c]; ib = c; } if (Fn[c] > 0) cnt++; }
  if (!(fmax > 0)) { out.info = 'contact forces: all zero'; return; }
  const L = HEIGHT_FRACTION * ctx.size / fmax * opts.scale, from = [], vec = [], n = st.nPrev;
  const mid = (c) => {
    if (contact.mid && ctx.deform === 1) return [contact.mid[3 * c], contact.mid[3 * c + 1], contact.mid[3 * c + 2]];
    const a = ctx.knotPairs[2 * c], b = ctx.knotPairs[2 * c + 1], P = ctx.P;
    return [0, 1, 2].map(q => 0.5 * (P[3 * a + q] + P[3 * b + q]));
  };
  for (let c = 0; c < Fn.length; c++) {
    if (!(Fn[c] > 0)) continue;
    const p = mid(c), v = [0, 1, 2].map(q => n[3 * c + q] * Fn[c] * L);
    from.push(...p.map((x, q) => x - 0.5 * v[q])); vec.push(...v);
  }
  out.arrows = { from: Float32Array.from(from), vec: Float32Array.from(vec), color: LOAD };
  if (opts.labels) { const p = mid(ib), v = [0, 1, 2].map(q => n[3 * ib + q] * Fn[ib] * L); out.labels = [{ p: p.map((x, q) => x + 0.5 * v[q]), text: `${fmtG(Fn[ib], 4)} ${diagramUnit(ctx.units, 'Fn')}` }]; }
  out.info = `contact normal forces  max ${fmtG(fmax, 4)} ${diagramUnit(ctx.units, 'Fn')}  ·  ${cnt} of ${Fn.length} crossings in contact`;
}

function reactions(ctx, out) {
  const { st, results: r, opts, P } = ctx;
  if (!r.reactions || !r.reactions.size) { out.info = 'reactions: none (no anchors)'; return; }
  const rows = [];
  for (const [fk, v] of r.reactions) { const i = st.nodeFrameIds.get(fk); if (i === undefined) continue; rows.push({ i, F: [v[0], v[1], v[2]], mag: Math.hypot(v[0], v[1], v[2]) }); }
  let mmax = 0, ib = 0; rows.forEach((q, k) => { if (q.mag > mmax) { mmax = q.mag; ib = k; } });
  if (!(mmax > 0)) { out.info = 'reactions: all zero'; return; }
  const L = HEIGHT_FRACTION * ctx.size / mmax * opts.scale, from = [], vec = [];
  const tail = (q) => [0, 1, 2].map(c => P[3 * q.i + c] - q.F[c] * L);
  for (const q of rows) { from.push(...tail(q)); vec.push(...q.F.map(f => f * L)); }
  out.arrows = { from: Float32Array.from(from), vec: Float32Array.from(vec), color: ANCHOR_FULL };
  if (opts.labels === 2) out.labels = rows.map(q => ({ p: tail(q), text: fmtG(q.mag, 4) }));
  else if (opts.labels === 1) out.labels = [{ p: tail(rows[ib]), text: `${fmtG(rows[ib].mag, 4)} ${diagramUnit(ctx.units, 'R')}` }];
  const tot = [0, 0, 0]; for (const q of rows) for (let c = 0; c < 3; c++) tot[c] += q.F[c];
  out.info = `reactions  max ${fmtG(mmax, 4)} ${diagramUnit(ctx.units, 'R')}  ·  ΣR = (${tot.map(t => fmtG(t, 4)).join(', ')})`;
}

