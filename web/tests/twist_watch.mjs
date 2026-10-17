// does a scene run away? joint twist, contact force and multiplier growth over a long relaxation (CPU f64)
//   node web/tests/twist_watch.mjs [scene=v5dome_BugTest] [iterations=60000] [report every=10000] [key=value,...]
// key=value overrides scene fields, e.g. contact.augmented=false. The 17 Oct 2026 bug: the multipliers of the pinned
// boundary crossings ratcheted (Fn 3.5 kN → 400 kN), the sections rolled past the diagonal and the laths flipped.
import { caseByName } from './cases.mjs';
import { CpuSolver, ActiveBeam } from '../js/solver_cpu.js';
const name = process.argv[2] || 'v5dome_BugTest', total = +(process.argv[3] || 60000), every = +(process.argv[4] || 10000);
const { sc, topo } = caseByName(name);
for (const m of (process.argv[5] || '').split(',').filter(Boolean)) { const [k, v] = m.split('='); const [g, f] = k.split('.'); sc[g][f] = v === 'true' ? true : v === 'false' ? false : isNaN(+v) ? v : +v; console.log(`  ${k} = ${sc[g][f]}`); }
const sv = new CpuSolver(sc, topo), st = sv.st, c = sv.contact;
console.log(`${name}: ${st.n} nodes · ${st.m} segments · ${st.p} joints · ${st.K} crossings`);
const Ra = new Float64Array(9), Rb = new Float64Array(9), P = new Float64Array(9), deg = 180 / Math.PI;
for (let it = 0; it < total; it += every) {
  sv.step(Math.min(every, total - it));
  let tw = 0, jm = 0; for (let j = 0; j < st.p; j++) { ActiveBeam.curv(st, j, Ra, Rb); const phi = Math.abs(ActiveBeam.twist(Ra, Rb, P)); if (phi > tw) { tw = phi; jm = j; } }
  let fn = 0, km = 0, lam = 0, gapMin = Infinity; for (let k = 0; k < st.K; k++) { if (c.Fn[k] > fn) { fn = c.Fn[k]; km = k; } lam = Math.max(lam, Math.abs(st.lam[k])); if (c.Fn[k] > 0) gapMin = Math.min(gapMin, c.gap[k]); }
  let d = 0; for (let i = 0; i < st.n; i++) d = Math.max(d, Math.hypot(st.X[3 * i] - st.X0[3 * i], st.X[3 * i + 1] - st.X0[3 * i + 1], st.X[3 * i + 2] - st.X0[3 * i + 2]));
  console.log(`it ${String(st.iteration).padStart(6)}  residual ${st.residual.toExponential(2)}  twist max ${(tw * deg).toFixed(1)}° (joint ${jm}, strand ${st.nodeStrand[st.jNode[jm]]})  Fn max ${fn.toFixed(0)} (knot ${st.knotEdges[km]})  |λ| max ${lam.toFixed(0)}  gap min ${gapMin.toFixed(3)}  |disp| max ${d.toFixed(2)}  energy bend/torsion/contact ${sv.beam.energyParts.bending.toExponential(2)} / ${sv.beam.energyParts.torsion.toExponential(2)} / ${c.energy.toExponential(2)}${sv.converged ? '  converged' : ''}`);
  if (sv.converged) break;
}
