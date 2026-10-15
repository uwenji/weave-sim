// Rhino export of the JS port on the Python reference state (tests/python/export_ref.py → ref_out/export_ref.json):
// the same positions, frames, contact midpoints / normals and results → ref_out/export_js_<case>.3dm (+ .json); then
// tests/python/cmp_3dm.py compares the files object by object with the Python export.
import fs from 'node:fs';
import path from 'node:path';
import { Mesh } from '../js/mesh.js';
import { buildTopology, fkey } from '../js/topology.js';
import { Scene } from '../js/scene.js';
import { buildState } from '../js/model.js';
import { exportRhino } from '../js/export3dm.js';
import { EXAMPLES, REF } from './paths.mjs';
import rhino3dm from '../vendor/rhino3dm/rhino3dm.module.min.js';

const rhino = await rhino3dm();
const ref = JSON.parse(fs.readFileSync(path.join(REF, 'export_ref.json')));
for (const [name, D] of Object.entries(ref)) {
  const sc = Scene.fromJSON(D.scene, name);
  const mesh = Mesh.fromBytes(fs.readFileSync(path.resolve(path.join(EXAMPLES, 'scenes'), sc.mesh.path)), sc.mesh.path, sc.mesh.scale);
  const topo = buildTopology(mesh, sc.weave, sc.section.height, sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null);
  const st = buildState(topo);
  st.X.set(D.X); st.Q.set(D.Q); st.X0.set(D.X0); st.nPrev.set(D.nPrev);
  if (JSON.stringify(Array.from(st.knotEdges)) !== JSON.stringify(D.knot_edges)) throw new Error('knot order differs');
  const R = D.results, seg = {};
  for (const k of ['Mx', 'My', 'T', 'N']) seg[k] = Float64Array.from(R.segment[k].flat());
  const knot = {}; for (const [k, m] of Object.entries(R.knot)) knot[k] = Float64Array.from(st.knotEdges, e => m[String(e)] ?? 0);
  const reactions = new Map(); for (const [key, v] of Object.entries(R.reactions)) { const [s, k] = key.split(',').map(Number); reactions.set(fkey(s, k), v); }
  const results = { segment: seg, knot, disp: Float64Array.from(R.node.disp.flat(2)), reactions };
  const t0 = performance.now();
  const out = await exportRhino({ scene: sc, topo, mesh, st, results, contact: { mid: Float64Array.from(D.mid) }, name: `export_js_${name}` }, {}, rhino);
  fs.writeFileSync(path.join(REF, `export_js_${name}.3dm`), out.bytes);
  fs.writeFileSync(path.join(REF, `export_js_${name}.3dm.json`), JSON.stringify(out.index, null, 1));
  console.log(`${name}: ${out.objects} objects, ${(out.bytes.length / 1e6).toFixed(2)} MB in ${((performance.now() - t0) / 1000).toFixed(2)} s`);
}
