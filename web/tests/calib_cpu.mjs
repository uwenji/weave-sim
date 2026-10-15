// CPU (f64) calibration replay vs the Python reference at the synthetic truth (force mode example)
import fs from 'node:fs';
import path from 'node:path';
import { CalibrationSet, FileResolver } from '../js/calibrate.js';
import { ROOT, EXAMPLES, REF } from './paths.mjs';

const dir = EXAMPLES + '/calibration/force/';
export function nodeResolver(base) {
  return new FileResolver({ reader: async (rel, kind) => { const p = path.resolve(base, rel); if (kind === 'text') return fs.readFileSync(p, 'utf8'); const b = fs.readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); } });
}
export async function loadForceExample() {
  const calib = JSON.parse(fs.readFileSync(dir + 'calib.json', 'utf8'));
  return CalibrationSet.load(calib, nodeResolver(dir));
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const cs = await loadForceExample();
  const truth = JSON.parse(fs.readFileSync(dir + 'synthetic_truth.json', 'utf8'));
  const vals = Object.fromEntries(cs.names().map(k => [k, truth[k]])); vals.pad_stiffness = 0;
  const refPath = REF + '/calib_truth.json';
  const ref = fs.existsSync(refPath) ? JSON.parse(fs.readFileSync(refPath, 'utf8')) : null;   // tests/python/calib_ref.py writes it
  if (!ref) console.log('(no Python reference in tests/ref_out — run tests/python/calib_ref.py to compare)');
  console.log('nodes', cs.topo.nStrands, 'strands; tests', cs.tests.map(t => `${t.name} ${t.curve.length} rows`).join(', '));
  const t0 = performance.now();
  const r = await cs.objective(vals);
  console.log(`JS objective ${r.objective.toFixed(4)}${ref ? ` (Python ${ref.objective.toFixed(4)})` : ''}  ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  for (const [k, v] of Object.entries(r.residuals)) console.log(`  ${k}: JS ${v.toFixed(4)}${ref ? `  Py ${ref.residuals[k].toFixed(4)}` : ''}`);
  for (const [k, br] of Object.entries(r.sims)) {
    const pb = ref ? ref.sims[k] : null;
    br.forEach((b, i) => console.log(`  ${k} c${b.cycle} ${b.branch.padEnd(6)} JS ${b.d.map(v => v.toFixed(4)).join(' ')}` + (pb ? `\n${' '.repeat(17)}Py ${pb[i][2].map(v => v.toFixed(4)).join(' ')}` : '')));
  }
  fs.mkdirSync(REF, { recursive: true });
  fs.writeFileSync(REF + '/calib_js_cpu.json', JSON.stringify({ objective: r.objective, residuals: r.residuals, sims: r.sims }));
}
