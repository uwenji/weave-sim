// Stability indicator (web/js/stability.js) on cases with a known critical load:
//   Euler column (pinned–pinned, 60 × 30, L = 1000): the lowest eigenvalue falls linearly to 0 at Pcr = π²EI/L²
//   lateral–torsional buckling (cantilever strip 60 × 6, edgewise tip load at the centroid): Pcr = 4.013 √(EI GJ)/L²
//   node web/tests/stability_check.mjs [euler | ltb | dome]
import { Scene } from '../js/scene.js';
import { CpuSolver } from '../js/solver_cpu.js';
import { rectTorsionConstant } from '../js/params.js';
import { stability } from '../js/stability.js';
import { straight, caseByName } from './cases.mjs';

const which = process.argv[2] || 'euler';
const E = 1e4, G = 625;
const tip = (st) => { let i = 0; for (let k = 0; k < st.n; k++) if (st.X[3 * k] > st.X[3 * i]) i = k; return i; };
function model({ rod = 'cosserat', w = 60, h = 30, nseg = 20, anchors, load }) {
  const topo = straight(1000, nseg), sc = new Scene();
  sc.loads.gravity.enabled = false; sc.solver.tol = 1e-10; sc.section.width = w; sc.section.height = h;
  for (const [id, fix] of anchors) sc.addAnchor({ type: 'frame', id }, fix);
  const sv = new CpuSolver(sc, topo, { contacts: false }); load(sv, sv.st);
  let it = 0; while (!sv.converged && it < 200000) { sv.step(5000); it += 5000; }
  return sv;
}
if (which === 'euler') {
  const EI = E * 60 * 30 ** 3 / 12, Pe = Math.PI ** 2 * EI / 1e6;
  for (const rod of ['cosserat']) {
    let ref = null;
    for (const r of [0, 0.5, 0.9, 1.1]) {
      const sv = model({ rod, anchors: [[[0, 0], [1, 1, 1, 0, 0, 1]], [[0, 1], [0, 1, 1, 0, 0, 0]]], load: (sv, st) => { sv.loads.const[3 * tip(st)] -= r * Pe; } });
      const s = stability(sv); if (ref === null) ref = s.lam;
      console.log(`${rod.padEnd(9)} P/Pe ${r.toFixed(1)}  λ ${s.lam.toExponential(3)}  λ/λ0 ${(s.lam / ref).toFixed(4)} (expected ${(1 - r).toFixed(2)} with the Euler mode)  negative modes ${s.neg}  [${s.ndof} dof, ${s.colours} colours, factor ${s.nnz} nz, ${s.ms.toFixed(0)} ms]`);
    }
  }
}
if (which === 'ltb') {
  const w = 60, h = 6, EIz = E * w * h ** 3 / 12, GJ = G * rectTorsionConstant(w, h), EIy = E * h * w ** 3 / 12;
  const Pcr = 4.013 * Math.sqrt(EIz * GJ) / 1e6 / Math.sqrt((1 - EIz / EIy) * (1 - GJ / EIy));
  console.log(`LTB cantilever Pcr = ${Pcr.toFixed(2)} N (4.013 √(EI GJ)/L² with the pre-buckling correction)`);
  for (const rod of ['cosserat']) {
    const pts = [];
    for (const r of [0, 0.4, 0.8, 0.95, 1.05]) {
      const P = r * Pcr;
      const sv = model({ rod, w, h, nseg: 40, anchors: [[[0, 0], [1, 1, 1, 1, 1, 1]]], load: (sv, st) => { sv.loads.const[3 * tip(st) + 1] -= P; } });
      const s = stability(sv); pts.push([P, s.lam]);
      console.log(`${rod.padEnd(9)} P/Pcr ${r.toFixed(2)}  λ ${s.lam.toExponential(3)}  negative modes ${s.neg}  [${s.ndof} dof, ${s.ms.toFixed(0)} ms]`);
    }
    // zero crossing of λ(P) by linear interpolation between the last positive and the first negative point
    let cross = null; for (let i = 1; i < pts.length; i++) if (pts[i - 1][1] > 0 && pts[i][1] <= 0) cross = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * pts[i - 1][1] / (pts[i - 1][1] - pts[i][1]);
    console.log(cross ? `  λ = 0 at P ≈ ${cross.toFixed(2)} N (${(100 * (cross / Pcr - 1)).toFixed(1)} % vs Pcr)` : '  λ stays positive: no lateral–torsional instability');
  }
}
if (which === 'dome') {
  for (const rod of ['cosserat']) {
    const c = caseByName('v5dome_cm');
    const sv = new CpuSolver(c.sc, c.topo); sv.step(12000);
    const s = stability(sv);
    console.log(`v5dome ${rod.padEnd(9)} it ${sv.st.iteration} residual ${sv.st.residual.toExponential(2)}  λ ${s.lam.toExponential(3)}  negative modes ${s.neg}  [${s.ndof} dof, ${s.colours} colours, ${s.evaluations} evaluations, factor ${(s.nnz / 1e6).toFixed(2)} M nz, ${s.ms.toFixed(0)} ms]`);
  }
}
