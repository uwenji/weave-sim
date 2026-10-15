// chaos ensemble of the force-mode replay on the CPU (f64): replicas from the same relaxed state, free nodes moved by
// perturb × model size — the spread of the objective is the noise floor any backend sees
import fs from 'node:fs';
import { loadForceExample } from './calib_cpu.mjs';
import { ROOT, EXAMPLES, REF } from './paths.mjs';
const perturb = +(process.argv[2] || 1e-7), R = +(process.argv[3] || 4);
const cs = await loadForceExample();
cs.replay.replicas = R; cs.replay.perturb = perturb; cs.replay.replica_stage = process.argv[4] || "relax";
const truth = JSON.parse(fs.readFileSync(EXAMPLES + '/calibration/force/synthetic_truth.json', 'utf8'));
const vals = Object.fromEntries(cs.names().map(k => [k, truth[k]])); vals.pad_stiffness = 0;
const t0 = performance.now();
const r = await cs.objective(vals);
const per = r.replicas.map(s => cs.objectiveOf(cs.residuals(s)));
console.log(`perturb ${perturb}: objective mean ${r.objective.toFixed(4)} ± ${r.objectiveStd.toFixed(4)}  replicas ${per.map(v => v.toFixed(4)).join(' ')}  (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
for (const k of Object.keys(r.residuals)) console.log(`  ${k}: ${r.residuals[k].toFixed(4)} ± ${r.spread[k].toFixed(4)}   replicas ${r.replicas.map(s => cs.residuals(s)[k].toFixed(4)).join(' ')}`);
for (const k of Object.keys(r.sims)) {
  const c0 = r.replicas.map(s => s[k][0].d.map(v => v.toFixed(3)).join(' '));
  console.log(`  ${k} c0 load, per replica:\n    ` + c0.join('\n    '));
}
