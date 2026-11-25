// Frame inertia against the tangent stiffness: the largest eigenvalue of M⁻¹K of a sharply bent lath (CPU f64).
//
// Dynamic relaxation with kinetic damping is a gradient step x ← x + M⁻¹g whenever the damping fires every other
// iteration, so it needs λ_max(M⁻¹K) < 2; above that a mode flips for ever (a period-2 cycle, the residual stalls).
// K = −∂g/∂q is taken by central differences of the solver's own forces (node forces, body-axis frame torques), M are
// the relaxation masses and frame inertias of the last step, λ_max comes from power iteration.
//
// The case: an arc of a 60 × 30 lath pinned at both ends, straight at rest, bent by ψ per joint. Once ψ is a few tenths
// of a radian the roll of a frame works against the edgewise stiffness too (¼ EIy ψ² / l̄, rod_cosserat.js header), not
// only against GJ / l̄. Without that share in the frame inertia (the solver before 25 Nov 2026) this arc does not relax
// at all from ψ = 0.3 on — the residual grows — and the snowman test model of that day sat at λ_max = 2.8, residual 0.21.
//   node web/tests/inertia_check.mjs [root of another web/js to compare, e.g. a backup]
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const root = process.argv[2] ? path.resolve(process.argv[2]) : path.join(here, '..', 'js');
const { Scene } = await import(path.join(root, 'scene.js'));
const { topologyFromPolylines } = await import(path.join(root, 'topology.js'));
const { CpuSolver } = await import(path.join(root, 'solver_cpu.js'));
const { toFrame, qexp, qmul, qnormalize } = await import(path.join(root, 'quat.js'));

function arc(psi, nseg = 12, l = 50) {                 // polyline on a circle: the chords turn by psi at every vertex
  const P = [0, 0, 0]; let x = 0, z = 0, a = 0.5 * psi * (nseg - 1);
  for (let i = 0; i < nseg; i++) { x += l * Math.cos(a); z += l * Math.sin(a); P.push(x, 0, z); a -= psi; }
  const topo = topologyFromPolylines([P], null, 30.0);
  const sc = new Scene();                               // mm · N, 60 × 30, E 10 000
  sc.loads.gravity.enabled = false;
  sc.addAnchor({ type: 'all_start' }, [1, 1, 1, 0, 0, 0]); sc.addAnchor({ type: 'all_end' }, [1, 1, 1, 0, 0, 0]);
  return new CpuSolver(sc, topo, { contacts: false });
}
function lambdaMax(sv, iters = 400) {
  const st = sv.st, n = st.n, m = st.m, N = 3 * n + 3 * m, free = sv.anchors.nodeFree;
  const F = new Float64Array(3 * n), T = new Float64Array(3 * m), R = new Float64Array(9);
  const X0 = st.X.slice(), Q0 = st.Q.slice();
  const g = () => {
    F.fill(0); T.fill(0); sv.beam.apply(st, F, T);
    const out = new Float64Array(N);
    for (let i = 0; i < 3 * n; i++) out[i] = free[i] ? F[i] : 0;
    for (let j = 0; j < m; j++) { toFrame(st.Q, 4 * j, R); for (let k = 0; k < 3; k++) out[3 * n + 3 * j + k] = R[3 * k] * T[3 * j] + R[3 * k + 1] * T[3 * j + 1] + R[3 * k + 2] * T[3 * j + 2]; }
    st.X.set(X0); st.Q.set(Q0);
    return out;
  };
  const move = (v, e) => {
    const dq = new Float64Array(4), q0 = new Float64Array(4), qq = new Float64Array(4);
    for (let i = 0; i < 3 * n; i++) if (free[i]) st.X[i] += e * v[i];
    for (let j = 0; j < m; j++) { const o = 3 * n + 3 * j; qexp(e * v[o], e * v[o + 1], e * v[o + 2], dq); for (let k = 0; k < 4; k++) q0[k] = st.Q[4 * j + k]; qmul(q0, dq, qq); qnormalize(qq); for (let k = 0; k < 4; k++) st.Q[4 * j + k] = qq[k]; }
  };
  const M = new Float64Array(N);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) M[3 * i + k] = sv.mass[i];
  for (let j = 0; j < 3 * m; j++) M[3 * n + j] = sv.inertia3[j];
  let v = Float64Array.from({ length: N }, (_, i) => Math.sin(12.9898 * (i + 1)) * 43758.5453 % 1), lam = 0;
  for (let it = 0; it < iters; it++) {
    let nv = 0, vmax = 0; for (let i = 0; i < N; i++) { if (i < 3 * n && !free[i]) v[i] = 0; nv += M[i] * v[i] * v[i]; }
    nv = Math.sqrt(nv); for (let i = 0; i < N; i++) { v[i] /= nv; vmax = Math.max(vmax, Math.abs(v[i])); }
    const e = 1e-6 / vmax;
    move(v, e); const gp = g(); move(v, -e); const gm = g();
    let l2 = 0; const w = new Float64Array(N);
    for (let i = 0; i < N; i++) { w[i] = -(gp[i] - gm[i]) / (2 * e) / M[i]; l2 += M[i] * w[i] * w[i]; }
    l2 = Math.sqrt(l2);
    if (it > 20 && Math.abs(l2 - lam) < 1e-7 * l2) { lam = l2; break; }
    lam = l2; v = w;
  }
  return lam;
}

let fails = 0;
console.log(`solver: ${root}`);
for (const psi of [0.1, 0.3, 0.5, 0.8]) {
  const sv = arc(psi);
  sv.step(1);                                          // one step: the inertia of the state as built
  const l0 = lambdaMax(sv);
  sv.run(60000);
  const l1 = lambdaMax(sv), ok = l0 < 2 && l1 < 2 && sv.converged;
  if (!ok) fails++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} bend ${psi.toFixed(1)} rad per joint · λ_max(M⁻¹K) as built ${l0.toFixed(3)} · relaxed ${l1.toFixed(3)} (limit 2) · ${sv.converged ? 'converged at' : 'NOT converged after'} ${sv.st.iteration} it, residual ${sv.st.residual.toExponential(2)}`);
}
console.log(fails ? `${fails} check(s) failed` : 'frame inertia covers the tangent stiffness');
process.exit(fails ? 1 : 0);
