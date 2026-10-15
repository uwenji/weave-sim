// Discretisation, member-diagram node values and joint helpers against the Python reference (tests/python/disc_ref.py
// writes ref_out/disc_ref.json): the same solved state (positions and results taken from Python) → the Discretizer's
// strand selection, joint counts, curvature, structural layer, objective components, normalisation and energy; the node
// values of Mx / N / |u| / Fn / Vy; then a JS annealing run (feasibility, monotone best energy, speed).
import fs from 'node:fs';
import path from 'node:path';
import { Mesh } from '../js/mesh.js';
import { buildTopology } from '../js/topology.js';
import { Scene } from '../js/scene.js';
import { buildState } from '../js/model.js';
import { nodeValues, knotNodePairs, discreteCurvature, strandSpan } from '../js/fields.js';
import { Discretizer, geometryFromModel, discretizeParams, jointPoints, pieceIndex, zoneMask, strandKnotRows } from '../js/discretize.js';
import { EXAMPLES, REF } from './paths.mjs';

const ref = JSON.parse(fs.readFileSync(path.join(REF, 'disc_ref.json')));
let worst = 0, fails = 0;
const relerr = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
function cmpArr(label, a, b, tol = 1e-10) {
  let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i])));
  if (a.length !== b.length) { console.log(`  ${label}: length ${a.length} vs ${b.length}`); fails++; return; }
  worst = Math.max(worst, m); if (m > tol) { console.log(`  ${label}: max diff ${m.toExponential(2)}`); fails++; }
}
for (const [name, R] of Object.entries(ref)) {
  const sc = Scene.fromJSON(JSON.parse(fs.readFileSync(path.join(EXAMPLES, 'scenes', name + '.json'))), name);
  const mesh = Mesh.fromBytes(fs.readFileSync(path.resolve(path.join(EXAMPLES, 'scenes'), sc.mesh.path)), sc.mesh.path, sc.mesh.scale);
  const topo = buildTopology(mesh, sc.weave, sc.section.height, sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null);
  const st = buildState(topo);
  R.X.forEach((P, s) => st.X.set(P, 3 * st.nodeOffset[s]));
  const seg = {}; for (const k of ['Mx', 'My', 'T', 'N']) seg[k] = Float64Array.from(R.segment[k].flat());
  const results = { segment: seg, disp: Float64Array.from(R.disp.flat()), knot: { Fn: Float64Array.from(st.knotEdges, e => R.Fn[String(e)] ?? 0) }, reactions: new Map() };
  const pairs = knotNodePairs(st, topo);
  console.log(`${name}: ${st.S} strands (${R.closed.filter(Boolean).length} closed) · ${st.K} crossings`);
  // node values
  for (const [lay, opt] of [['Mx', {}], ['N', {}], ['disp', {}], ['Fn', { knotPairs: pairs }], ['Vy', { shear: true }]]) {
    const v = nodeValues(st, results, lay === 'Vy' ? 'Mx' : lay, { ...opt, knotPairs: pairs });
    cmpArr(`node values ${lay}`, v, R.nodes[lay].flat(), 1e-12);
  }
  const curv = []; for (let s = 0; s < st.S; s++) { const { i0, n } = strandSpan(st, s); curv.push(...discreteCurvature(st.X, i0, n)); }
  cmpArr('curvature', curv, R.curv.flat(), 1e-12);
  // discretizer
  const geo = geometryFromModel(topo, st, st.X, results, 'Mx');
  cmpArr('knot_s', geo.flatMap(g => [...g.knot_s]), R.knot_s.flat(), 1e-12);
  const d = new Discretizer(geo, discretizeParams(R.params), R.width);
  if (JSON.stringify(d.strandIds) !== JSON.stringify(R.strand_ids)) { console.log('  strand ids differ'); fails++; }
  if (JSON.stringify(d.nJoints) !== JSON.stringify(R.n_joints)) { console.log('  joint counts differ'); fails++; }
  cmpArr('lengths', d.L, R.L, 1e-12);
  cmpArr('structural layer', d.geo.flatMap(g => [...g.layer]), R.layer.flat(), 1e-12);
  const comp = (label, a, b) => { for (const k of Object.keys(b)) { const e = relerr(a[k], b[k]); worst = Math.max(worst, e); if (e > 1e-10 || a[k] === undefined) { console.log(`  ${label}.${k}: ${a[k]} vs ${b[k]}`); fails++; } } };
  comp('components0', d.components0, R.components0);
  comp('normalised0', d.normalised(d.components0), R.normalised0);
  if (relerr(d.historyBest[0], R.energy0) > 1e-10) { console.log(`  energy0 ${d.historyBest[0]} vs ${R.energy0}`); fails++; }
  const xt = R.x_test.map(a => Float64Array.from(a));
  comp('components(x)', d.components(xt), R.components_test);
  comp('normalised(x)', d.normalised(d.components(xt)), R.normalised_test);
  const e = d.energy(xt); if (relerr(e, R.energy_test) > 1e-10) { console.log(`  energy(x) ${e} vs ${R.energy_test}`); fails++; }
  cmpArr('pieces(x)', d.pieces(xt).flatMap(a => [...a]), R.pieces_test.flat(), 1e-12);
  if (JSON.stringify(xt.map((xi, i) => d.feasible(xi, d.L[i]))) !== JSON.stringify(R.feasible_test)) { console.log('  feasibility differs'); fails++; }
  // a JS annealing run
  const t0 = performance.now(); d.step(150); const dt = performance.now() - t0;
  let feasible = d.xbest.every((xi, i) => d.feasible(xi, d.L[i])), mono = d.historyBest.every((v, i) => i === 0 || v <= d.historyBest[i - 1]);
  console.log(`  annealing 150 steps × ${d.p.mll}: ${(dt / 1000).toFixed(2)} s, ${d.nEval} evaluations, accepted ${d.accepted}, objective ${d.historyBest[0].toFixed(4)} → ${d.ebest.toFixed(4)}  feasible ${feasible}  monotone ${mono}`);
  if (!feasible || !mono || !(d.ebest < d.historyBest[0])) fails++;
  const eFull = d.energy(d.x); if (relerr(d.e, eFull) > 1e-12) { console.log(`  incremental energy ${d.e} vs full ${eFull}`); fails++; }
  const res = d.result();
  sc.joints = res.joints;
  const jp = jointPoints(sc, st), pi = pieceIndex(sc, st), rows = strandKnotRows(topo);
  const zm = zoneMask(sc, topo, st, d.strandIds[0], 3, st.X, rows);
  console.log(`  joints ${jp.length} · pieces on strand ${d.strandIds[0]}: ${Math.max(...pi.slice(st.nodeOffset[d.strandIds[0]], st.nodeOffset[d.strandIds[0] + 1])) + 1} · zone nodes ${zm.reduce((a, b) => a + b, 0)} / ${zm.length}`);
}
console.log(fails ? `FAILED (${fails})` : `ok — worst relative difference ${worst.toExponential(2)}`);
process.exit(fails ? 1 : 0);
