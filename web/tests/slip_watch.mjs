// which crossings keep slipping during the relaxation of the calibration example (CPU f64)
import fs from 'node:fs';
import { loadForceExample } from './calib_cpu.mjs';
import { CpuSolver, deepClone } from '../js/solver_cpu.js';
import { applyParams } from '../js/calibrate.js';
import { ROOT, EXAMPLES, REF } from './paths.mjs';
const cs = await loadForceExample();
const truth = JSON.parse(fs.readFileSync(EXAMPLES + '/calibration/force/synthetic_truth.json', 'utf8'));
const vals = Object.fromEntries(cs.names().map(k => [k, truth[k]])); vals.pad_stiffness = 0;
const sc = deepClone(cs.scene); applyParams(sc, vals);
const sv = new CpuSolver(sc, cs.topo), st = sv.st;
sv.step(2000);
const u0 = Float64Array.from(st.u), X0 = Float64Array.from(st.X);
let slipping = new Uint32Array(st.K), resets = 0, prevKE = 0;
for (let i = 0; i < 2000; i++) { sv.step(1); for (let c = 0; c < st.K; c++) slipping[c] += sv.contact.state[c]; const ke = st.history.kinetic.at(-1); if (ke === 0 && prevKE > 0) resets++; prevKE = ke; }
const du = []; for (let c = 0; c < st.K; c++) du.push([c, Math.abs(st.u[2 * c] - u0[2 * c]) + Math.abs(st.u[2 * c + 1] - u0[2 * c + 1]), slipping[c]]);
du.sort((a, b) => b[1] - a[1]);
let dx = 0, imax = 0; for (let i = 0; i < st.n; i++) { const d = Math.hypot(st.X[3 * i] - X0[3 * i], st.X[3 * i + 1] - X0[3 * i + 1], st.X[3 * i + 2] - X0[3 * i + 2]); if (d > dx) { dx = d; imax = i; } }
const boundary = (c) => cs.topo.knots.get(st.knotEdges[c]).isBoundary;
console.log(`iterations 2000→4000: kinetic-damping resets ${resets}, max node travel ${dx.toExponential(2)} cm (node ${imax}, strand ${st.nodeStrand[imax]}), residual now ${st.residual.toExponential(2)}`);
console.log(`crossings with any slipping iteration: ${[...slipping].filter(v => v > 0).length} / ${st.K}; slipping > 50 % of the time: ${[...slipping].filter(v => v > 1000).length}`);
console.log('largest sliding (|Δu| in segments, slipping iterations, boundary):');
for (const [c, d, k] of du.slice(0, 12)) console.log(`  knot ${st.knotEdges[c]}  |Δu| ${d.toExponential(2)}  slipping ${k}  ${boundary(c) ? 'boundary' : ''}`);
