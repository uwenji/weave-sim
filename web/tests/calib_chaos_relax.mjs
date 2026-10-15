// chaos ensemble including the relaxation: the woven (initial) state is perturbed by perturb × size before relax
import fs from 'node:fs';
import { loadForceExample } from './calib_cpu.mjs';
import { CpuSolver, deepClone } from '../js/solver_cpu.js';
import { applyParams, runAsync, replayForce, perturbReplica } from '../js/calibrate.js';
import { ROOT, EXAMPLES, REF } from './paths.mjs';
const perturb = +(process.argv[2] || 1e-12), R = +(process.argv[3] || 6), only = process.argv[4] || 'P1';
const cs = await loadForceExample();
const truth = JSON.parse(fs.readFileSync(EXAMPLES + '/calibration/force/synthetic_truth.json', 'utf8'));
const vals = Object.fromEntries(cs.names().map(k => [k, truth[k]])); vals.pad_stiffness = 0;
const t = cs.tests.find(x => x.name === only);
const rp = cs.replay;
for (let r = 0; r < R; r++) {
  const sc = deepClone(cs.scene); applyParams(sc, vals);
  const sv = new CpuSolver(sc, cs.topo);
  perturbReplica(sv, r, perturb);
  await runAsync(sv, rp.relax_iter);
  sv.internalize();
  const jobs = cs._jobs(vals, [t], sc)[0];
  const br = await replayForce(sv, t.probe, jobs.forces, rp.cycles, rp.step_iter, rp.avg_iter, 0);
  const res = cs.residualCycles(t, br);
  console.log(`replica ${r}: relax it ${sv.st.iteration} res ${sv.st.residual.toExponential(2)}  ${only} residual ${res.toFixed(4)}  c0 load ${br[0].d.map(v => v.toFixed(3)).join(' ')}  c1 load ${br[2].d.map(v => v.toFixed(3)).join(' ')}`);
}
