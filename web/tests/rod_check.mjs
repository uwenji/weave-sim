// Cosserat rod (web/js/rod_cosserat.js): forces and torques against central finite differences of the energy, on a
// randomly bent, twisted and sheared strand with rest curvature and an anisotropic section.
//   node web/tests/rod_check.mjs
import { buildState } from '../js/model.js';
import { CosseratRod } from '../js/rod_cosserat.js';
import { qexp, qmul, qnormalize } from '../js/quat.js';
import { straight } from './cases.mjs';
import { Rng } from '../js/rng.js';

const rng = new Rng(11), r = () => rng.uniform ? rng.uniform() * 2 - 1 : rng.next() * 2 - 1;
const topo = straight(1000, 8), st = buildState(topo), n = st.n, m = st.m;
const EA = 1e4 * 1800, G = 625;
const stiff = { EA: new Float64Array(m).fill(EA), EIx: new Float64Array(m).fill(1e4 * 1.35e5), EIy: new Float64Array(m).fill(1e4 * 5.4e5), GJ: new Float64Array(m).fill(G * 3.7e5), GA: new Float64Array(m).fill(5 / 6 * G * 1800) };
const act = new Float64Array(3 * st.p).fill(0.4);
// as-built: bent and twisted
for (let i = 0; i < n; i++) { st.X[3 * i + 1] += 40 * Math.sin(i); st.X[3 * i + 2] += 25 * Math.cos(1.3 * i); }
const dq = new Float64Array(4), q = new Float64Array(4);
const rot = (j, v) => { qexp(v[0], v[1], v[2], dq); qmul(dq, st.Q.subarray(4 * j, 4 * j + 4), q); qnormalize(q); st.Q.set(q, 4 * j); };
for (let j = 0; j < m; j++) rot(j, [0.3 * r(), 0.3 * r(), 0.8 * r()]);
st.Q0 = st.Q.slice();
const rod = new CosseratRod(st, stiff, 'flat', act);
rod.setClamps(st, [[0, 0, [true, true, true]], [st.n - 1, m - 1, [true, false, true]]]);   // clamp joints at both ends
// deformed state
for (let i = 0; i < 3 * n; i++) st.X[i] += 5 * r();
for (let j = 0; j < m; j++) rot(j, [0.2 * r(), 0.2 * r(), 0.5 * r()]);
const energy = () => { rod.apply(st, new Float64Array(3 * n), new Float64Array(3 * m)); return rod.energy; };
const F = new Float64Array(3 * n), T = new Float64Array(3 * m);
rod.apply(st, F, T);
console.log(`clamp joints ${rod.cSeg.length}, clamp moments ${Array.from(rod.cM, v => v.toExponential(2)).join(" ")}`);
let ef = 0, fs = 0, et = 0, ts = 0;
const h = 1e-4;
for (let i = 0; i < 3 * n; i++) {
  const x0 = st.X[i]; st.X[i] = x0 + h; const ep = energy(); st.X[i] = x0 - h; const em = energy(); st.X[i] = x0;
  const fd = -(ep - em) / (2 * h); ef = Math.max(ef, Math.abs(fd - F[i])); fs = Math.max(fs, Math.abs(F[i]));
}
const hr = 1e-6;
for (let j = 0; j < m; j++) for (let k = 0; k < 3; k++) {
  const q0 = st.Q.slice(4 * j, 4 * j + 4), v = [0, 0, 0];
  v[k] = hr; rot(j, v); const ep = energy(); st.Q.set(q0, 4 * j);
  v[k] = -hr; rot(j, v); const em = energy(); st.Q.set(q0, 4 * j);
  const fd = -(ep - em) / (2 * hr); et = Math.max(et, Math.abs(fd - T[3 * j + k])); ts = Math.max(ts, Math.abs(T[3 * j + k]));
}
console.log(`forces : max |F_fd − F| / max|F| = ${(ef / fs).toExponential(2)}   (max |F| ${fs.toExponential(2)})`);
console.log(`torques: max |T_fd − T| / max|T| = ${(et / ts).toExponential(2)}   (max |T| ${ts.toExponential(2)})`);
const ok = ef / fs < 1e-5 && et / ts < 1e-5;
console.log(ok ? 'PASS' : 'FAIL'); process.exitCode = ok ? 0 : 1;
