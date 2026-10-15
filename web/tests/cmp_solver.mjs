// CPU f64 port vs the Python solver: states after the same iteration counts (reference dumps: tests/python/dump_solver.py → tests/ref_out)
import fs from 'node:fs';
import { CpuSolver } from '../js/solver_cpu.js';
import { caseByName } from './cases.mjs';
import { ROOT, EXAMPLES, REF } from './paths.mjs';

const only = process.argv[2];
const files = fs.readdirSync(REF).filter(f => f.startsWith('solver_') && (!only || f === `solver_${only}.json`));
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const maxAbsVal = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m || 1; };
for (const f of files) {
  const ref = JSON.parse(fs.readFileSync(REF + '/' + f));
  const { sc, topo, contacts } = caseByName(ref.name, ref.scene.loads ? ref.scene : null);
  const t0 = performance.now();
  const sv = new CpuSolver(sc, topo, { contacts });
  const size = maxAbsVal(sv.st.X);
  const rows = [`${ref.name}: mass ${(maxAbs(sv.mass, ref.mass) / maxAbsVal(ref.mass)).toExponential(1)}  Ω0 ${(maxAbs(sv.st.Omega0, ref.Omega0) / maxAbsVal(ref.Omega0)).toExponential(1)}  Q0 ${maxAbs(sv.st.Q, ref.Q0).toExponential(1)}` + (ref.u0.length ? `  u0 ${maxAbs(sv.st.u, ref.u0).toExponential(1)}` : '')];
  let done = 0;
  for (const s of ref.snaps) {
    while (done < s.iteration) { sv.step(1); done++; }
    const st = sv.st;
    const dX = maxAbs(st.X, s.X) / size;
    const dQ = maxAbs(st.Q, s.Q);
    const dN = maxAbs(sv.beam.N, s.N) / maxAbsVal(s.N);
    const dM = s.M.length ? maxAbs(sv.beam.M, s.M) / maxAbsVal(s.M) : 0;
    let extra = '';
    if (s.Fn) {
      const c = sv.contact;
      extra = `  u ${maxAbs(st.u, s.u).toExponential(1)}  Fn ${(maxAbs(c.Fn, s.Fn) / maxAbsVal(s.Fn)).toExponential(1)}  gap ${maxAbs(c.gap, s.gap).toExponential(1)}  slip ${maxAbs(st.slip, s.slip).toExponential(1)}  state ${c.state.reduce((a, v, i) => a + (v !== s.state[i]), 0)}`;
      if (sc.contact.seat_law !== 'off') extra += `  dent ${maxAbs(c.dent, s.dent).toExponential(1)}`;
      if (sc.contact.rot_friction) extra += `  Mrot ${(maxAbs(c.Mrot, s.Mrot) / maxAbsVal(s.Mrot)).toExponential(1)}`;
    }
    const r = s.residual == null ? Infinity : s.residual;
    rows.push(`  it ${String(s.iteration).padStart(5)}  X ${dX.toExponential(1)}  Q ${dQ.toExponential(1)}  N ${dN.toExponential(1)}  M ${dM.toExponential(1)}  ke ${st.history.kinetic.at(-1).toExponential(3)}/${s.ke.toExponential(3)}  res ${st.residual.toExponential(2)}/${r.toExponential(2)}${extra}`);
  }
  rows.push(`  (${((performance.now() - t0) / 1000).toFixed(1)} s, ${(1e3 * (performance.now() - t0) / done).toFixed(2)} ms/it)`);
  console.log(rows.join('\n'));
}
