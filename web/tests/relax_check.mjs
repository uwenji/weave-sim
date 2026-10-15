// how far the relaxation of the calibration example gets: residual vs iterations (CPU f64)
import fs from 'node:fs';
import { loadForceExample } from './calib_cpu.mjs';
import { CpuSolver, deepClone } from '../js/solver_cpu.js';
import { applyParams } from '../js/calibrate.js';
import { ROOT, EXAMPLES, REF } from './paths.mjs';
const cs = await loadForceExample();
const truth = JSON.parse(fs.readFileSync(EXAMPLES + '/calibration/force/synthetic_truth.json', 'utf8'));
const vals = Object.fromEntries(cs.names().map(k => [k, truth[k]])); vals.pad_stiffness = 0;
const sc = deepClone(cs.scene); applyParams(sc, vals);
const sv = new CpuSolver(sc, cs.topo);
const rows = [];
for (let it = 0; it < 12000; it += 500) { sv.step(500); rows.push(`${sv.st.iteration}: ${sv.st.residual.toExponential(2)}${sv.converged ? ' converged' : ''}`); if (sv.converged) break; }
console.log(rows.join('\n'));
