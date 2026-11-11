// Self-crossing strands (19 Oct 2026): on a genus-1 torus from try-tri-weave the laths wind round and cross themselves —
// both frames of a knot on one strand. Over / under is per frame (kn.overSide); before the fix both passes of every
// self-crossing were lifted +h/2. Checks one over pass per knot, the ±h/2 separation, alternation incl. the wrap of
// closed strands, and that a flip swaps both passes.   node web/tests/selfcross_check.mjs [mesh.obj]
import fs from 'node:fs';
import path from 'node:path';
import { Mesh } from '../js/mesh.js';
import { buildTopology, overFrame } from '../js/topology.js';
import { EXAMPLES } from './paths.mjs';
const file = process.argv[2] || path.join(EXAMPLES, 'meshes', 'genus1_torus_v6_696v.obj');
const mesh = Mesh.fromBytes(fs.readFileSync(file), file, 1.0);
const h = 0.8, P = { subdivisions: 4, overhang: 0, offset_mode: 'half_height', flip_knots: [], seed_strand_phase: 0 };
const t = buildTopology(mesh, P, h);
let fails = 0; const ok = (c, m) => { console.log(`${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fails++; };
const knots = [...t.knots.values()], selfx = knots.filter(k => k.a[0] === k.b[0]);
ok(selfx.length > 0, `${t.nStrands} strands (${t.strands.filter(s => s.closed).length} closed), ${t.nKnots} knots, ${selfx.length} self-crossings, parity conflicts ${t.parityConflicts}`);
ok(knots.every(k => t.isOver(...k.a) !== t.isOver(...k.b)), 'one over pass per knot');
let sep = 0; for (const k of knots) { const a = t.frameOrigin(...k.a), b = t.frameOrigin(...k.b), e = k.edge; const d = (a[0] - b[0]) * mesh.edgeNormals[3 * e] + (a[1] - b[1]) * mesh.edgeNormals[3 * e + 1] + (a[2] - b[2]) * mesh.edgeNormals[3 * e + 2]; if ((d > 0) !== t.isOver(...k.a) || Math.abs(d) < 0.99 * h) sep++; }
ok(sep === 0, `centrelines ±h/2 on the right side at every knot (${sep} wrong)`);
let br = 0; for (const s of t.strands) { const K = s.edges.length; for (let k = 0; k < (s.closed ? K : K - 1); k++) if (t.isOver(s.index, k) === t.isOver(s.index, (k + 1) % K)) br++; }
ok(br === 0, `over / under alternates along every strand (${br} breaks)`);
const kn = selfx[0], t1 = buildTopology(mesh, { ...P, flip_knots: [kn.a] }, h);
ok(t1.isOver(...kn.a) !== t.isOver(...kn.a) && t1.isOver(...kn.b) !== t.isOver(...kn.b), `flip of self-crossing ${kn.edge}: passes (${kn.a}) / (${kn.b}) swap`);
process.exit(fails ? 1 : 0);
