// The validation cases of WeaveSim/tests/test_solver.py, built in the browser (fetches scenes and meshes over http)
import { Scene } from '../js/scene.js';
import { Mesh } from '../js/mesh.js';
import { buildTopology, topologyFromPolylines, fkey } from '../js/topology.js';
import { WEAVE, fromDict } from '../js/params.js';

export function straight(L = 1000, nseg = 10, normals = null) {
  const P = []; for (let i = 0; i <= nseg; i++) P.push(L * i / nseg, 0, 0);
  return topologyFromPolylines([P], normals ? [normals] : null, 30.0);
}
export function crossing({ load, pull, mu = 0.35, freeB = true, nseg = 10, L = 1000, h = 30 } = {}) {
  const P = [], PB = [];
  for (let i = 0; i <= nseg; i++) { const x = -L / 2 + L * i / nseg; P.push(x, 0, 0); PB.push(0, x, h); }
  const mid = Math.floor(nseg / 2);
  const topo = topologyFromPolylines([P, PB], null, h, null, [[0, mid, nseg], [0, mid, nseg]]);
  topo.knots.set(0, { edge: 0, a: [0, 1], b: [1, 1], over: 1, conflict: false, flipped: false, isBoundary: false });
  topo.frameToKnot.set(fkey(0, 1), 0); topo.frameToKnot.set(fkey(1, 1), 0);
  const sc = new Scene(); sc.loads.gravity.enabled = false; sc.contact.mu = mu;
  for (const s of [0, 1]) {
    const fix = (s === 1 && freeB) ? [0, 1, 1, 0, 0, 1] : [1, 1, 1, 0, 0, 1];
    sc.addAnchor({ type: 'frame', id: [s, 0] }, fix); sc.addAnchor({ type: 'frame', id: [s, 2] }, fix);
  }
  sc.addPointLoad([1, 1], [pull, 0, load]);
  return { sc, topo };
}
export async function caseByName(name, base = '/examples') {
  const L = 1000, F = 10;
  if (name === 'cantilever' || name === 'cantilever_edge') {
    const topo = straight(L, 10), sc = new Scene(); sc.loads.gravity.enabled = false;
    sc.addAnchor({ type: 'frame', id: [0, 0] }, [1, 1, 1, 1, 1, 1]); sc.addPointLoad([0, 1], name === 'cantilever' ? [0, 0, -F] : [0, -F, 0]);
    return { sc, topo, contacts: false };
  }
  if (name === 'torsion') {
    const nseg = 10, normals = [];
    for (let i = 0; i <= nseg; i++) { const a = Math.PI / 2 * i / nseg; normals.push([0, -Math.sin(a), Math.cos(a)]); }
    const topo = straight(L, nseg, normals), sc = new Scene(); sc.loads.gravity.enabled = false;
    sc.addAnchor({ type: 'frame', id: [0, 0] }, [1, 1, 1, 1, 1, 1]); sc.addAnchor({ type: 'frame', id: [0, 1] }, [1, 1, 1, 0, 0, 1]);
    return { sc, topo, contacts: false };
  }
  if (name.startsWith('cross_')) {
    const pull = { cross_press: 0, cross_stick: 5, cross_slip: 12, cross_locked: 12, cross_w23: 5, cross_al: 0 }[name];
    const { sc, topo } = crossing({ load: -50, pull });
    if (name === 'cross_locked') sc.contact.locked = true;
    if (name === 'cross_w23') { sc.contact.rot_friction = true; sc.contact.seat_law = 'L1'; sc.contact.dent_after_iter = 0; sc.contact.slip_rate = 0.3; }
    if (name === 'cross_al') { sc.contact.augmented = true; sc.contact.al_every = 600; sc.contact.strength = 2000; }
    return { sc, topo, contacts: true };
  }
  if (name === 'sheet') {
    const d = await (await fetch('/web/tests/ref_out/sheet_mesh.json')).json();
    const mesh = Mesh.fromArrays(d.V, d.F);
    const topo = buildTopology(mesh, fromDict(WEAVE), 30.0);
    const sc = new Scene(); sc.loads.gravity.enabled = false;
    sc.addAnchor({ type: 'all_start' }, [1, 1, 1, 0, 0, 1]); sc.addAnchor({ type: 'all_end' }, [1, 1, 1, 0, 0, 1]);
    sc.addPointLoad(d.over, [0, 0, -200.0]);
    return { sc, topo, contacts: true };
  }
  const scUrl = `${base}/scenes/${name}.json`;
  const sc = Scene.fromJSON(await (await fetch(scUrl)).json(), name);
  const meshUrl = new URL(sc.mesh.path, new URL(scUrl, location.href)).href;
  const buf = new Uint8Array(await (await fetch(meshUrl)).arrayBuffer());
  const mesh = Mesh.fromBytes(buf, meshUrl, sc.mesh.scale);
  const topo = buildTopology(mesh, sc.weave, sc.section.height, sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null);
  return { sc, topo, contacts: true };
}
