// Cosserat rod (rod_cosserat.js) against closed-form results.
//   node web/tests/rod_bench.mjs [all | cantilever | short | beamcol | ltb | kirchhoff_limit]
// Section 60 × 30 mm (LTB: 60 × 6), E 10 000 N/mm², G 625 N/mm², κ = 5/6, L = 1000 mm unless noted.
import { Scene } from '../js/scene.js';
import { CpuSolver } from '../js/solver_cpu.js';
import { rectTorsionConstant } from '../js/params.js';
import { straight } from './cases.mjs';

const which = process.argv[2] || 'all', run = (k) => which === 'all' || which === k;
const E = 1e4, G = 625, kap = 5 / 6;
function solve({ L = 1000, nseg = 20, rod = 'cosserat', w = 60, h = 30, anchors, loads, tol = 1e-10, maxIter = 600000, GAfac = 1 }) {
  const topo = straight(L, nseg), sc = new Scene();
  sc.loads.gravity.enabled = false; sc.solver.tol = tol; sc.solver.max_iter = maxIter;
  sc.section.width = w; sc.section.height = h;
  for (const [id, fix] of anchors) sc.addAnchor({ type: 'frame', id }, fix);
  if (GAfac !== 1) { sc.beam.mode = 'advanced'; sc.beam.stiffness_shear = GAfac; }
  const sv = new CpuSolver(sc, topo, { contacts: false });
  const st = sv.st;
  loads(sv, st);
  let it = 0; while (!sv.converged && it < maxIter) { sv.step(5000); it += 5000; }
  return sv;
}
const tipNode = (st) => { let i = 0; for (let k = 0; k < st.n; k++) if (st.X[3 * k] > st.X[3 * i]) i = k; return i; };
const pct = (a, b) => `${(100 * (a / b - 1)).toFixed(2).padStart(6)} %`;
const clamp = [[[0, 0], [1, 1, 1, 1, 1, 1]]];

if (run('cantilever')) {
  console.log('\n— cantilever, tip load 10 N, Timoshenko δ = FL³/3EI + FL/κGA —');
  for (const [dir, I, lbl] of [[2, 60 * 30 ** 3 / 12, 'flatwise'], [1, 30 * 60 ** 3 / 12, 'edgewise']]) {
    const dB = 10 * 1e9 / (3 * E * I), dS = 10 * 1000 / (kap * G * 1800);
    for (const nseg of [10, 20, 40]) for (const rod of ['cosserat']) {
      const sv = solve({ nseg, rod, anchors: clamp, loads: (sv, st) => { sv.loads.const[3 * tipNode(st) + dir] -= 10; } });
      const d = -sv.st.X[3 * tipNode(sv.st) + dir];
      console.log(`${lbl} ${String(nseg).padStart(2)} seg ${rod.padEnd(9)} δ ${d.toFixed(5)} mm   vs bending only ${pct(d, dB)}   vs Timoshenko ${pct(d, dB + dS)}   (${sv.st.iteration} it)`);
    }
  }
}
if (run('short')) {
  console.log('\n— short deep cantilever L = 200 mm, edgewise 100 N: shear is 30 % of the deflection —');
  const I = 30 * 60 ** 3 / 12, dB = 100 * 200 ** 3 / (3 * E * I), dS = 100 * 200 / (kap * G * 1800);
  for (const rod of ['cosserat']) {
    const sv = solve({ L: 200, nseg: 20, rod, anchors: clamp, loads: (sv, st) => { sv.loads.const[3 * tipNode(st) + 1] -= 100; } });
    const d = -sv.st.X[3 * tipNode(sv.st) + 1];
    console.log(`${rod.padEnd(9)} δ ${d.toFixed(5)} mm   bending only ${dB.toFixed(5)}   Timoshenko ${(dB + dS).toFixed(5)}   error vs Timoshenko ${pct(d, dB + dS)}`);
  }
}
if (run('kirchhoff_limit')) {
  console.log('\n— shear-free (Kirchhoff) limit: κGA × 1, 100, 10⁴ on the short cantilever —');
  const I = 30 * 60 ** 3 / 12, dB = 100 * 200 ** 3 / (3 * E * I);
  for (const f of [1, 100, 1e4]) {
    const sv = solve({ L: 200, nseg: 20, rod: 'cosserat', GAfac: f, anchors: clamp, loads: (sv, st) => { sv.loads.const[3 * tipNode(st) + 1] -= 100; }, maxIter: 300000 });
    const d = -sv.st.X[3 * tipNode(sv.st) + 1], dS = 100 * 200 / (f * kap * G * 1800);
    console.log(`κGA × ${String(f).padEnd(5)} δ ${d.toFixed(6)}  expected ${(dB + dS).toFixed(6)}  ${pct(d, dB + dS)}  (${sv.st.iteration} it, converged ${sv.converged})`);
  }
}
if (run('beamcol')) {
  console.log('\n— pinned–pinned beam-column, sine lateral load (w_bending = 1 mm) + axial P: w = w_I/(1 − P/Pcr) —');
  for (const rod of ['cosserat']) for (const ratio of [0, 0.5, 0.8, 0.95, -0.5]) {
    const sv = solve({ rod, anchors: [[[0, 0], [1, 1, 1, 0, 0, 1]], [[0, 1], [0, 1, 1, 0, 0, 0]]], loads: (sv, st) => {
      const L = 1000, nseg = 20, EI = E * 60 * 30 ** 3 / 12, q0 = Math.PI ** 4 * EI / L ** 4, Pcr = Math.PI ** 2 * EI / L ** 2;
      for (let i = 0; i < st.n; i++) { const x = st.X[3 * i], wt = (x < 1e-9 || x > L - 1e-9) ? 0.5 : 1; sv.loads.const[3 * i + 2] -= q0 * Math.sin(Math.PI * x / L) * (L / nseg) * wt; }
      sv.loads.const[3 * tipNode(st)] -= ratio * Pcr; } });
    const st = sv.st; let im = 0; for (let i = 0; i < st.n; i++) if (Math.abs(st.X[3 * i] - 500) < Math.abs(st.X[3 * im] - 500)) im = i;
    const EI = E * 60 * 30 ** 3 / 12, Pe = Math.PI ** 2 * EI / 1e6, kGA = kap * G * 1800;
    const wI = rod === 'cosserat' ? 1 + Pe / kGA : 1, Pcr = rod === 'cosserat' ? kGA * (Math.sqrt(1 + 4 * Pe / kGA) - 1) / 2 : Pe;   // Haringx
    const P = ratio * Pe, wExp = wI / (1 - P / Pcr), w = -st.X[3 * im + 2];
    console.log(`${rod.padEnd(9)} P/Pe ${ratio.toFixed(2).padStart(5)}  w ${w.toFixed(4)}  expected ${wExp.toFixed(4)}  ${pct(w, wExp)}  (${st.iteration} it)`);
  }
}
if (run('ltb')) {
  console.log('\n— lateral–torsional buckling: cantilever strip 60 × 6 mm, L = 1000, edgewise tip load at the centroid —');
  const w = 60, h = 6, EIz = E * w * h ** 3 / 12, GJ = G * rectTorsionConstant(w, h), EIy = E * h * w ** 3 / 12;
  const Pcr = 4.013 * Math.sqrt(EIz * GJ) / 1e6;
  console.log(`Pcr = 4.013 √(EI_weak · GJ) / L² = ${Pcr.toFixed(2)} N  (EI_weak/EI_strong = ${(EIz / EIy).toFixed(3)})`);
  for (const rod of ['cosserat']) {
    const rows = [];
    for (const r of [0.3, 0.6, 0.8, 0.9, 1.1, 1.3]) {
      const P = r * Pcr;
      const sv = solve({ rod, w, h, nseg: 40, anchors: clamp, tol: 1e-9, maxIter: 400000, loads: (sv, st) => { const t = tipNode(st); sv.loads.const[3 * t + 1] -= P; sv.loads.const[3 * t + 2] -= 1e-3 * P; } });
      const st = sv.st, t = tipNode(st);
      const lat = -st.X[3 * t + 2], inpl = -st.X[3 * t + 1];
      rows.push([r, P, lat, inpl, sv.st.iteration, sv.converged]);
    }
    console.log(rod);
    for (const [r, P, lat, inpl, it, cv] of rows) console.log(`  P/Pcr ${r.toFixed(1)}  P ${P.toFixed(2).padStart(6)} N   in-plane δ ${inpl.toFixed(3).padStart(8)}   out-of-plane δ ${lat.toFixed(3).padStart(9)} mm   (${it} it${cv ? '' : ', not converged'})`);
    // lateral–torsional amplification δ = δ0 / (1 − (P/Pcr)²) with δ0 ∝ P: δ/P = c + (δ·P)/Pcr², a line of slope 1/Pcr²
    const pre = rows.filter(r => r[0] < 1 && Math.abs(r[2]) > 0);
    if (pre.length >= 3) {
      const xs = pre.map(r => r[2] * r[1]), ys = pre.map(r => r[2] / r[1]);
      const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
      let sxy = 0, sxx = 0; for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
      const slope = sxy / sxx;
      console.log(slope > 0 ? `  Southwell (LTB form) Pcr ≈ ${(1 / Math.sqrt(slope)).toFixed(2)} N  (${pct(1 / Math.sqrt(slope), Pcr)} vs 4.013 √(EI GJ)/L²; pre-buckling bending raises it by ≈ 0.6 %)` : '  no amplification: the out-of-plane deflection stays linear in P — no lateral–torsional buckling');
    }
  }
}
