// displacement-mode replay (pinned probe push, locked crossings, pad compliance) vs the Python reference
import fs from 'node:fs';
import { loadForceExample } from './calib_cpu.mjs';
import { REF } from './paths.mjs';
const cs = await loadForceExample();
Object.assign(cs.replay, { mode: 'displacement', max_disp: 1.5, lock: true, n_steps: 5, step_iter: 1000, avg_iter: 300 });
const vals = { E: 1.4e6, activation: 0.7, pad_stiffness: 3000.0 };
const t0 = performance.now();
const r = await cs.objective(vals);
const ref = fs.existsSync(REF + '/calib_disp_truth.json') ? JSON.parse(fs.readFileSync(REF + '/calib_disp_truth.json', 'utf8')) : null;
console.log(`JS objective ${r.objective.toFixed(5)}${ref ? ` (Python ${ref.objective.toFixed(5)})` : ''}  ${((performance.now() - t0) / 1000).toFixed(1)} s`);
for (const [k, s] of Object.entries(r.sims)) {
  console.log(`  ${k} d  JS ${s.d.map(v => v.toFixed(4)).join(' ')}` + (ref ? `\n       Py ${ref.sims[k][0].map(v => v.toFixed(4)).join(' ')}` : ''));
  console.log(`  ${k} F  JS ${s.f.map(v => v.toFixed(2)).join(' ')}` + (ref ? `\n       Py ${ref.sims[k][1].map(v => v.toFixed(2)).join(' ')}` : ''));
}
