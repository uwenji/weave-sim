// Compare the JS topology with the Python dumps written by tests/python/dump_topology.py (tests/ref_out)
import fs from 'node:fs';
import path from 'node:path';
import { Mesh } from '../js/mesh.js';
import { buildTopology, fkey } from '../js/topology.js';
import { Scene } from '../js/scene.js';
import { ROOT, EXAMPLES, REF } from './paths.mjs';

const OUT = REF, EX = EXAMPLES;
let worst = 0, fails = 0;
for (const f of fs.readdirSync(OUT).filter(f => f.startsWith('topo_'))) {
  const ref = JSON.parse(fs.readFileSync(path.join(OUT, f)));
  const name = f.slice(5, -5);
  const sc = Scene.fromJSON(ref.scene, name);
  let meshPath;
  if (name.startsWith('mesh_')) meshPath = path.join(EX, 'meshes', name.slice(5) + '.stl');
  else meshPath = path.resolve(path.join(EX, 'scenes'), sc.mesh.path);
  const mesh = Mesh.fromBytes(fs.readFileSync(meshPath), meshPath, sc.mesh.scale);
  const topo = buildTopology(mesh, sc.weave, sc.section.height, sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null);
  const errs = [];
  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (mesh.nV !== ref.mesh.nV || mesh.nF !== ref.mesh.nF || mesh.nE !== ref.mesh.nE) errs.push('mesh sizes');
  let dV = 0; for (let i = 0; i < mesh.V.length; i++) dV = Math.max(dV, Math.abs(mesh.V[i] - ref.mesh.V[i]));
  if (dV > 0) errs.push('V ' + dV);
  if (!eq(Array.from(mesh.F), ref.mesh.F)) errs.push('F');
  if (!eq(Array.from(mesh.E), ref.mesh.E)) errs.push('E');
  if (!eq(Array.from(mesh.EF), ref.mesh.EF)) errs.push('EF');
  let dN = 0; for (let i = 0; i < mesh.edgeNormals.length; i++) dN = Math.max(dN, Math.abs(mesh.edgeNormals[i] - ref.mesh.edge_normals[i]));
  if (topo.nStrands !== ref.strands.length) errs.push(`strands ${topo.nStrands} vs ${ref.strands.length}`);
  let famDiff = 0;
  ref.strands.forEach((s, i) => { const t = topo.strands[i]; if (!t) return; if (!eq(t.edges, s.edges)) errs.push(`strand ${i} edges`); if (t.closed !== s.closed) errs.push(`strand ${i} closed`); if (t.family !== s.family) famDiff++; });
  if (famDiff) errs.push(`family differs on ${famDiff} strands`);
  const kj = [...topo.knots.values()];
  if (kj.length !== ref.knots.length) errs.push('knot count');
  ref.knots.forEach((k, i) => { const t = kj[i]; if (!t) return; if (t.edge !== k.edge || !eq(t.a, k.a) || !eq(t.b, k.b) || t.over !== k.over || t.conflict !== k.conflict || t.flipped !== k.flipped || t.isBoundary !== k.boundary) errs.push(`knot ${i}`); });
  for (const [s, k, r] of ref.knot_node) if (topo.knotNode.get(fkey(s, k)) !== r) errs.push(`knot_node ${s},${k}`);
  let dP = 0, dR = 0;
  ref.polylines.forEach((P, s) => { const Q = topo.polylines[s]; if (!Q || Q.length !== P.length) { errs.push(`polyline ${s} length`); return; } for (let i = 0; i < P.length; i++) dP = Math.max(dP, Math.abs(P[i] - Q[i])); });
  ref.frames.forEach((R, s) => { const Q = topo.nodeFrames[s]; if (!Q || Q.length !== R.length) { errs.push(`frames ${s} length`); return; } for (let i = 0; i < R.length; i++) dR = Math.max(dR, Math.abs(R[i] - Q[i])); });
  if (topo.parityConflicts !== ref.parity_conflicts) errs.push('parity conflicts');
  if (topo.familyCount !== ref.family_count) errs.push(`family count ${topo.familyCount} vs ${ref.family_count}`);
  const bb = mesh.bbox().size;
  worst = Math.max(worst, dP / bb);
  if (dP / bb > 1e-12 || dR > 1e-10) errs.push(`geometry dP/size ${(dP / bb).toExponential(2)} dR ${dR.toExponential(2)}`);
  console.log(`${name.padEnd(22)} strands ${topo.nStrands} knots ${topo.nKnots}  |dP|/size ${(dP / bb).toExponential(2)}  |dR| ${dR.toExponential(2)}  |dN| ${dN.toExponential(2)}  ${errs.length ? 'FAIL ' + errs.slice(0, 6).join('; ') : 'ok'}`);
  if (errs.length) fails++;
}
console.log(fails ? `${fails} cases differ` : 'all cases match');
