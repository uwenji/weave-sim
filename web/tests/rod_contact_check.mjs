// Cosserat rod on the crossing cases and the example scenes (CPU f64): contact forces, slip, displacement, rod energy.
//   node web/tests/rod_contact_check.mjs [cases | scenes] [iterations]
import { caseByName } from './cases.mjs';
import { CpuSolver } from '../js/solver_cpu.js';

const what = process.argv[2] || 'cases', iters = +(process.argv[3] || 0);
const list = what === 'cases' ? [['cross_press', 6000], ['cross_stick', 6000], ['cross_slip', 6000], ['cross_locked', 6000], ['cross_w23', 6000]]
  : [['v5dome_cm', 20000], ['dome_basic', 12000]];
const sum = (a) => a.reduce((s, v) => s + v, 0), max = (a) => a.length ? Math.max(...a) : 0;
for (const [name, it0] of list) {
  const c = caseByName(name);
  const sv = new CpuSolver(c.sc, c.topo, { contacts: c.contacts });
  const t0 = Date.now(); sv.step(iters || it0); const dt = (Date.now() - t0) / 1000;
  const st = sv.st, cc = sv.contact, Fn = cc ? Array.from(cc.Fn) : [], slip = Array.from(st.slip);
  let dmax = 0; for (let i = 0; i < st.n; i++) dmax = Math.max(dmax, Math.hypot(st.X[3 * i] - st.X0[3 * i], st.X[3 * i + 1] - st.X0[3 * i + 1], st.X[3 * i + 2] - st.X0[3 * i + 2]));
  const finite = st.X.every(Number.isFinite) && st.Q.every(Number.isFinite);
  console.log(`${name.padEnd(13)} it ${String(st.iteration).padStart(6)}  residual ${st.residual.toExponential(2)}${sv.converged ? ' ✓' : '  '}  |disp| max ${dmax.toFixed(4)}  ΣFn ${sum(Fn).toFixed(2)}  max Fn ${max(Fn).toFixed(2)}  Σslip ${sum(slip).toFixed(4)}  rod energy ${sv.beam.energy.toExponential(3)} (shear ${(sv.beam.energyParts.shear || 0).toExponential(2)})  ${dt.toFixed(1)} s${finite ? '' : '  NON-FINITE STATE'}`);
}
