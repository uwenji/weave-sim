// The ground (6 Anchors › ground; Floor in solver_cpu.js) on the CPU f64 solver: one straight lath, no anchors.
//   rest    it starts tilted above the ground, falls, lands on one end, tips flat and rests: ΣF_ground = −Σ loads
//   stick   a sideways push below μ·W: it does not move, the ground's friction equals the push
//   slide   a push above μ·W: it slides flat, every node on the cone (|F_t| = μ F_n), ΣF_x = −μ ΣF_z
//   mu0     μ = 0: no friction force at all
//   pinned  with an anchor the rigid-body freedoms are off and the anchor + ground carry the loads together
// node web/tests/floor_check.mjs
import { Scene } from '../js/scene.js';
import { topologyFromPolylines } from '../js/topology.js';
import { CpuSolver } from '../js/solver_cpu.js';

function lath({ push = 0, mu = 0.4, anchor = false, tilt = 0.05, lift = 2 } = {}) {
  const L = 100, nseg = 10, P = [];
  for (let i = 0; i <= nseg; i++) P.push(L * i / nseg - L / 2, 0, lift + tilt * (L * i / nseg));
  const topo = topologyFromPolylines([P], null, 3.0);
  const sc = new Scene(); sc.units = new sc.units.constructor('cm', 'N');
  sc.section.width = 6; sc.section.height = 3; sc.material.E = 1e6; sc.material.G = 6e4;
  sc.contact.strength = 3000; sc.contact.stick_strength = 1000; sc.contact.ramp_iter = 0;
  sc.loads.gravity.enabled = true; sc.solver.tol = 1e-6;
  if (push) sc.loads.udl = { enabled: true, q: [push / L, 0, 0], per: 'length' };
  if (anchor) sc.addAnchor({ type: 'all_start' }, [1, 1, 1, 0, 0, 0]);
  sc.floor = { enabled: true, z: 0, mu };
  const sv = new CpuSolver(sc, topo, { contacts: false }), st = sv.st, W = [0, 0, 0];
  for (let i = 0; i < st.n; i++) for (let k = 0; k < 3; k++) W[k] += sv.loads.const[3 * i + k];
  return { sv, st, W, nseg };
}
const cone = (sv) => { let lo = Infinity, hi = 0; const f = sv.floor; for (let i = 0; i < f.n; i++) if (f.state[i]) { const r = Math.hypot(f.F[3 * i], f.F[3 * i + 1]) / Math.max(f.F[3 * i + 2], 1e-300); lo = Math.min(lo, r); hi = Math.max(hi, r); } return [lo, hi]; };
let fails = 0;
const check = (name, ok, text) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(7)} ${text}`); if (!ok) fails++; };
const f4 = (v) => v.toFixed(4);

{ // rest
  const { sv, st, W, nseg } = lath(); sv.run(20000);
  const s = sv.floor.summary(), flat = Math.abs(st.X[2] - st.X[3 * nseg + 2]);
  check('rest', sv.converged && Math.abs(s.sum[2] + W[2]) < 1e-4 * Math.abs(W[2]) && s.nodes === st.n && flat < 1e-6,
    `converged at ${st.iteration} · ΣF_z ${f4(s.sum[2])} against weight ${f4(-W[2])} · ${s.nodes}/${st.n} nodes on the ground · z of the ends ${f4(st.X[2])}, ${f4(st.X[3 * nseg + 2])} (half thickness 1.5)`);
}
{ // stick: push 3 N, μ W = 0.4 · 12.37 = 4.95 N
  const { sv, st, W } = lath({ push: 3 }); sv.run(20000);
  const s = sv.floor.summary();
  check('stick', sv.converged && s.slipping === 0 && Math.abs(s.sum[0] + W[0]) < 1e-4 * Math.abs(W[0]),
    `converged at ${st.iteration} · push ${f4(W[0])} · ground ΣF_x ${f4(s.sum[0])} · sliding nodes ${s.slipping} · |F_t|/F_n ≤ ${f4(cone(sv)[1])} (μ 0.4)`);
}
{ // slide: push 6 N > μ W
  const { sv, st, W } = lath({ push: 6 }); sv.step(3000);
  const s = sv.floor.summary(), [lo, hi] = cone(sv), x0 = st.X[0] + 50;
  check('slide', s.slipping === s.nodes && s.nodes === st.n && Math.abs(lo - 0.4) < 1e-9 && Math.abs(hi - 0.4) < 1e-9 && x0 > 1 && Math.abs(s.sum[0] + 0.4 * s.sum[2]) < 1e-6 * s.sum[2],
    `after ${st.iteration} it · moved ${f4(x0)} along the push · every node slides, |F_t|/F_n ${f4(lo)} … ${f4(hi)} · ΣF_x ${f4(s.sum[0])} = −μ ΣF_z ${f4(-0.4 * s.sum[2])}`);
}
{ // frictionless
  const { sv, st } = lath({ push: 3, mu: 0 }); sv.step(2000);
  const s = sv.floor.summary();
  check('mu0', Math.abs(s.sum[0]) < 1e-12 && st.X[0] + 50 > 1, `after ${st.iteration} it · moved ${f4(st.X[0] + 50)} · ground ΣF_x ${s.sum[0].toExponential(1)}`);
}
{ // one end pinned: no rigid-body freedoms, anchor + ground share the weight
  const { sv, st, W } = lath({ anchor: true, tilt: 0, lift: 1.5 }); sv.run(40000);
  const s = sv.floor.summary(); let R = 0; for (let i = 0; i < st.n; i++) R += sv.anchors.reactions[3 * i + 2];
  check('pinned', sv.converged && sv.residualParts.rigid === 0 && Math.abs(s.sum[2] + R + W[2]) < 1e-3 * Math.abs(W[2]),
    `converged at ${st.iteration} · ground ΣF_z ${f4(s.sum[2])} + anchor ${f4(R)} against weight ${f4(-W[2])}`);
}
console.log(fails ? `${fails} check(s) failed` : 'all ground checks passed');
process.exit(fails ? 1 : 0);
