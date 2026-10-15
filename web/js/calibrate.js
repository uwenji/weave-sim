// Calibration sandbox (port of weavesim/calibrate.py): replay of UTM load tests on the simulation and fitting of
// model parameters to the measured curves — with a batched WebGPU path: P parameter vectors × T tests run as P·T
// instances of one solver in the same dispatches, so a population (Latin hypercube, CMA-ES generation, sensitivity
// sweep) costs about one replay on the GPU instead of P·T on the CPU.
//
// Replay (per test, from the relaxed structure — prestress + self-weight, geometry internalised):
//   force mode (cable pull, the lab protocol): F·axis at the probe node in n_steps increments up to the peak and back
//     to 0, `cycles` times; after each increment the structure relaxes (≤ step_iter − avg_iter iterations, stops at
//     convergence) and the probe travel along the axis is averaged over avg_iter iterations;
//     crosshead travel = probe travel + F / pad_stiffness (loading-pad / cable compliance in series)
//   displacement mode (pinned probe push): the probe node is driven along the axis, the averaged reaction is the force
// Objective = Σ weight · residual over the training tests:
//   force mode — RMS over the fitted cycles' load and unload branches of (simulated travel − measured travel at the same
//   force on the same branch of the same cycle), / measured travel range;
//   displacement mode — RMS force error on the monotone loading branch / peak force.
// Every evaluation relaxes the structure once per parameter vector and replays all tests from copies of that state.
//
// The GPU objective is a deterministic function of the parameters: fixed-point scales are fixed per calibration
// (corners of the parameter box) and forces are accumulated exactly, so a candidate gives the same value in any batch.
// CPU (f64) and GPU (f32 + fixed point) objectives differ at the level of the stick–slip chaos of the replay.
import { CpuSolver, deepClone, pairwiseSum } from './solver_cpu.js';
import { GpuSolver } from './solver_gpu.js';
import { Scene } from './scene.js';
import { Mesh } from './mesh.js';
import { buildTopology, fkey } from './topology.js';
import { UnitSystem } from './units.js';
import { parseUTMCsv, interp } from './utm.js';
import { Rng } from './rng.js';

// ------------------------------------------------------------------------------------------------ parameters
const P_ = (path) => {
  const [grp, key] = path.split('.');
  return [(sc) => sc[grp][key], (sc, v) => { sc[grp][key] = +v; }];
};
/** name → [getter, setter] on a Scene; everything the calibration may touch goes through here */
export const PARAMS = {
  E: P_('material.E'), G: P_('material.G'),
  stiffness: P_('beam.stiffness'), stiffness_twist: P_('beam.stiffness_twist'), activation: P_('beam.activation'),
  mu: P_('contact.mu'), torsion_factor: P_('contact.torsion_factor'), strength: P_('contact.strength'),
  stick_strength: P_('contact.stick_strength'), slip_rate: P_('contact.slip_rate'),
  pad_stiffness: P_('utm.pad_stiffness'),
  // review W1–W3
  rot_factor: P_('contact.rot_factor'), fc90: P_('material.fc90'), H: P_('material.H'), E90: P_('material.E90'),
};
export const LOG_SCALED = new Set(['E', 'G', 'stiffness', 'stiffness_twist', 'strength', 'stick_strength', 'pad_stiffness', 'torsion_factor', 'rot_factor', 'fc90', 'H', 'E90']);
/** physical dimension of each parameter (for unit conversion of parameter sets); the others are unit-free */
export const PARAM_DIM = { E: 'stress', G: 'stress', fc90: 'stress', H: 'stress', E90: 'stress', strength: 'force/length', stick_strength: 'force/length', pad_stiffness: 'force/length' };
export const PARAM_INFO = {
  E: 'Young’s modulus of the laths', G: 'shear modulus of the laths (torsion GJ)',
  stiffness: 'bending stiffness factor (simple beam mode)', stiffness_twist: 'twist stiffness factor (advanced beam mode)',
  activation: 'active-bending activation: share of the as-built curvature kept as prestress (0 = rests in the bent shape)',
  mu: 'Coulomb friction coefficient lath on lath', torsion_factor: 'lever-arm torque factor of the contact (swivel)',
  strength: 'normal contact penalty [force/length]', stick_strength: 'tangential stick penalty [force/length]',
  slip_rate: 'fraction of the excess tangential force turned into slip per iteration',
  pad_stiffness: 'loading pad / cable compliance in series with the probe [force/length] (0 = rigid)',
  rot_factor: 'rotational friction factor (W2)', fc90: 'compression strength perpendicular to grain (W3 seat law)',
  H: 'embedment hardening modulus (W3)', E90: 'modulus perpendicular to grain (W3)',
};
/** the transfer class of a parameter: material (a property of the timber), contact (lath on lath, same section and
 *  surface), test (belongs to the test rig, not to the structure) */
export const PARAM_CLASS = { E: 'material', G: 'material', fc90: 'material', H: 'material', E90: 'material', stiffness: 'model', stiffness_twist: 'model', activation: 'model', mu: 'contact', torsion_factor: 'contact', strength: 'contact', stick_strength: 'contact', slip_rate: 'contact', rot_factor: 'contact', pad_stiffness: 'test' };

export function applyParams(sc, values) {
  for (const [k, v] of Object.entries(values)) {
    if (!PARAMS[k]) throw new Error(`unknown calibration parameter ${k}`);
    PARAMS[k][1](sc, v);
  }
}
export function readParams(sc, names) { const o = {}; for (const k of names) o[k] = PARAMS[k][0](sc); return o; }

/** conversion factor of a parameter from units `src` to `dst` */
export function paramFactor(name, src, dst) {
  const d = PARAM_DIM[name];
  if (d === 'stress') return src.stressFactorTo(dst);
  if (d === 'force/length') return src.forceFactorTo(dst) / src.lengthFactorTo(dst);
  return 1.0;
}

// ------------------------------------------------------------------------------------------------ replay
export class Probe {
  constructor(frame, axis = [0, 0, -1], name = '') {
    this.frame = [frame[0] | 0, frame[1] | 0];
    const a = axis.map(Number), n = Math.max(Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), 1e-12);
    this.axis = a.map(v => v / n); this.name = name;
  }
  static fromDict(d) { return new Probe(d.frame, d.axis || [0, 0, -1], d.name || ''); }
  toJSON() { return { name: this.name, frame: [...this.frame], axis: [...this.axis] }; }
}

export const REPLAY_DEFAULTS = {
  mode: 'force',          // 'force': cable pull, load-controlled cycles (lab protocol); 'displacement': pinned probe push
  relax_iter: 2500,       // must exceed contact.ramp_iter + settling
  step_iter: 600,
  avg_iter: 200,          // probe displacement / reaction averaged over the last avg_iter iterations
  n_steps: 5,             // load increments per branch
  cycles: 3,              // load–unload cycles (force mode)
  fit_cycles: [1, 2],     // 0-based cycles that enter the objective (skip bedding-in)
  max_force: null,        // peak force per test; null = from the UTM curve
  workers: 2,             // (Python: tests replayed in parallel) — kept for the file format
  max_disp: null,         // displacement mode: structure travel cap
  lock: false,            // fasten the crossings (Project 3 structures only; the lab model slides freely)
  // chaos ensemble (WeaveSim web): every test is replayed `replicas` times from the relaxed state with the free nodes
  // moved by perturb × model size (replica 0 unperturbed); residuals are averaged over the replicas and their spread
  // is reported — the stick–slip replay is chaotic, a single replay is one sample of a noisy objective
  replicas: 1,
  perturb: 1e-9,
  replica_stage: 'relax', // 'relax': each replica perturbs the woven state and relaxes on its own (the relaxed stick /
                          // slip pattern is part of the chaos); 'replay': one relaxation, replicas perturbed after it
};
export function replaySettings(d) { const o = structuredClone(REPLAY_DEFAULTS); for (const k of Object.keys(REPLAY_DEFAULTS)) if (d && d[k] !== undefined) o[k] = structuredClone(d[k]); return o; }

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = () => new Promise(r => setTimeout(r, 0));
class Aborted extends Error { constructor() { super('aborted'); this.name = 'AbortError'; } }
const checkAbort = (signal) => { if (signal && signal.aborted) throw new Aborted(); };

/** Solver.run with a yield to the event loop every ~yieldMs (same iteration semantics: chunks of live_every, stop at
 *  convergence, convergence flag cleared first) */
export async function runAsync(sv, maxIter, { yieldMs = 30, signal = null } = {}) {
  maxIter = maxIter || sv.scene.solver.max_iter;
  const every = Math.max(1, sv.scene.solver.live_every | 0);
  let done = 0, t = now();
  sv.converged = false;
  while (done < maxIter && !sv.converged) {
    const k = Math.min(every, maxIter - done);
    sv.step(k); done += k;
    if (now() - t > yieldMs) { await tick(); checkAbort(signal); t = now(); }
  }
  return sv.st;
}

export function probeNode(st, probe) {
  const nid = st.nodeFrameIds.get(fkey(probe.frame[0], probe.frame[1]));
  if (nid === undefined) throw new Error(`probe ${probe.name || ''} frame (${probe.frame}) is not a crossing of this weave`);
  return nid;
}

export async function relax(sc, topo, iters, lock = false, opts = {}) {
  const sv = new CpuSolver(sc, topo);
  await runAsync(sv, iters, opts);
  sv.internalize();
  if (lock) sv.lockContacts();
  return sv;
}

/** the force schedule of a cable-pull test: [(cycle, 'load'|'unload', forces)] */
export function forceSequence(forces, cycles) {
  const up = Array.from(forces, Number), down = [...up].reverse().slice(1).concat([0.0]), seq = [];
  for (let c = 0; c < cycles; c++) { seq.push([c, 'load', up]); seq.push([c, 'unload', down]); }
  return seq;
}

/** load-controlled cable pull (CPU, f64): → [{cycle, branch, d, f}] with d = crosshead travel from the start */
export async function replayForce(svRelaxed, probe, forces, cycles, stepIter, avgIter = 200, pad = null, opts = {}) {
  const sv = svRelaxed.clone();
  sv.rebuildGoals();
  const st = sv.st, nid = probeNode(st, probe), ax = probe.axis;
  const P0 = [st.X[3 * nid], st.X[3 * nid + 1], st.X[3 * nid + 2]];
  const nAvg = Math.max(avgIter, 1), settle = Math.max(stepIter - avgIter, 1);
  const out = [];
  let dPrev = 0.0, fPrev = 0.0, applied = [0, 0, 0];
  for (const [c, branch, Fs] of forceSequence(forces, cycles)) {
    const ds = [dPrev], fs = [fPrev];
    for (const F of Fs) {
      for (let k = 0; k < 3; k++) sv.loads.const[3 * nid + k] -= applied[k];
      applied = ax.map(a => a * F);
      for (let k = 0; k < 3; k++) sv.loads.const[3 * nid + k] += applied[k];
      sv.converged = false;
      await runAsync(sv, settle, opts);
      let acc = 0.0;
      for (let i = 0; i < nAvg; i++) {
        sv.step(1);
        acc += (st.X[3 * nid] - P0[0]) * ax[0] + (st.X[3 * nid + 1] - P0[1]) * ax[1] + (st.X[3 * nid + 2] - P0[2]) * ax[2];
      }
      let d = acc / nAvg;
      if (pad) d += F / pad;
      ds.push(d); fs.push(F);
      if (opts.onIncrement) opts.onIncrement();
    }
    dPrev = ds[ds.length - 1]; fPrev = fs[fs.length - 1];
    out.push({ cycle: c, branch, d: ds, f: fs });
  }
  return out;
}

/** displacement-controlled push (CPU, f64): the probe node pinned at P0 + d·axis → {d, f} (crosshead travel, reaction) */
export async function replayDisplacement(svRelaxed, probe, disps, stepIter, pad = null, avgIter = 100, opts = {}) {
  const sv = svRelaxed.clone();
  sv.rebuildGoals();
  const st = sv.st, nid = probeNode(st, probe), ax = probe.axis, An = sv.anchors;
  for (let k = 0; k < 3; k++) An.nodeFree[3 * nid + k] = 0;
  An.fixedAny[nid] = 1; An.allFree[nid] = 0;
  if (!An.fixedNodes.includes(nid)) An.fixedNodes.push(nid);
  const P0 = [st.X[3 * nid], st.X[3 * nid + 1], st.X[3 * nid + 2]];
  const nAvg = Math.max(avgIter, 1), settle = Math.max(stepIter - avgIter, 1);
  const dOut = [0.0], fOut = [0.0];
  for (const d of disps) {
    if (d <= 0) continue;
    for (let k = 0; k < 3; k++) { st.X[3 * nid + k] = P0[k] + ax[k] * d; st.V[3 * nid + k] = 0.0; }
    sv.converged = false;
    await runAsync(sv, settle, opts);
    let acc = 0.0;
    for (let i = 0; i < nAvg; i++) { sv.step(1); acc += An.reactions[3 * nid] * ax[0] + An.reactions[3 * nid + 1] * ax[1] + An.reactions[3 * nid + 2] * ax[2]; }
    dOut.push(+d); fOut.push(acc / nAvg);
    if (opts.onIncrement) opts.onIncrement();
  }
  if (pad) for (let i = 0; i < dOut.length; i++) dOut[i] += fOut[i] / pad;
  return { d: dOut, f: fOut };
}

/** replica r > 0: move the free nodes of the (relaxed, cloned) model by perturb × model size in a fixed pseudo-random
 *  direction per node and replica (the same pattern for every parameter vector: common random numbers) */
export function perturbReplica(sv, r, perturb) {
  if (!r || !perturb) return;
  const st = sv.st, X = st.X, rng = new Rng(7919 * r + 13);
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < st.n; i++) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], X[3 * i + c]); hi[c] = Math.max(hi[c], X[3 * i + c]); }
  const eps = perturb * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const free = sv.anchors.nodeFree;
  for (let i = 0; i < 3 * st.n; i++) { const v = rng.normal() * eps; if (free[i]) X[i] += v; }
}

const maxOf = (a) => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
/** np.argsort order for objective values: NaN last (worst), ties by index */
const byValue = (f) => (a, b) => (Number.isNaN(f[a]) - Number.isNaN(f[b])) || (f[a] < f[b] ? -1 : f[a] > f[b] ? 1 : 0) || (a - b);

const linspaceTail = (a, b, n) => Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a + (i + 1) * ((b - a) / n)));
const maxAbs = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };

// ------------------------------------------------------------------------------------------------ calibration set
export class Test {
  constructor(probe, curve = null, weight = 1.0, csv = '') { this.probe = probe; this.curve = curve; this.weight = weight; this.csv = csv; }
  get name() { return this.probe.name; }
  toJSON() { return { name: this.name, frame: [...this.probe.frame], axis: [...this.probe.axis], csv: this.csv, weight: this.weight }; }
}

/** where the files of a calibration live: text(rel) / bytes(rel) / has(rel) relative to the calib.json */
export class FileResolver {
  /** files: Map name → File | ArrayBuffer | string (dropped files, matched by relative path or base name); base: URL
   *  of the calib.json for fetch(); reader: async (relPath, 'text'|'bytes') → content (Node, tests) */
  constructor({ files = null, base = null, reader = null } = {}) { this.files = files; this.base = base; this.reader = reader; }
  _find(rel) {
    if (!this.files) return undefined;
    const norm = (p) => p.replace(/\\/g, '/').replace(/^\.\//, '');
    const r = norm(rel), baseName = r.split('/').pop();
    if (this.files.has(r)) return this.files.get(r);
    for (const [k, v] of this.files) if (norm(k).split('/').pop() === baseName) return v;
    return undefined;
  }
  has(rel) { return this.files ? this._find(rel) !== undefined : true; }
  async _get(rel, kind) {
    const f = this._find(rel);
    if (f !== undefined) {
      if (typeof f === 'string') return kind === 'text' ? f : new TextEncoder().encode(f).buffer;
      if (f instanceof ArrayBuffer) return kind === 'text' ? new TextDecoder().decode(f) : f;
      if (ArrayBuffer.isView(f)) return kind === 'text' ? new TextDecoder().decode(f) : f.buffer.slice(f.byteOffset, f.byteOffset + f.byteLength);
      return kind === 'text' ? await f.text() : await f.arrayBuffer();
    }
    if (this.reader) return this.reader(rel, kind);
    if (this.base) {
      const r = await fetch(new URL(rel, this.base));
      if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`);
      return kind === 'text' ? r.text() : r.arrayBuffer();
    }
    throw new Error(`file not found: ${rel}`);
  }
  text(rel) { return this._get(rel, 'text'); }
  bytes(rel) { return this._get(rel, 'bytes'); }
}
const dirOf = (p) => { const i = p.replace(/\\/g, '/').lastIndexOf('/'); return i < 0 ? '' : p.slice(0, i + 1); };
/** a/b/../c → a/c (relative paths stay relative) */
export function joinPath(dir, rel) {
  if (/^([a-z]+:)?\//i.test(rel)) return rel;
  const parts = (dir + rel).replace(/\\/g, '/').split('/'), out = [];
  for (const p of parts) { if (p === '..' && out.length && out[out.length - 1] !== '..') out.pop(); else if (p !== '.' && p !== '') out.push(p); else if (p === '..') out.push(p); }
  return out.join('/');
}

export class CalibrationSet {
  constructor({ scene, topo, params, fixed = {}, tests = [], validation = [], replay = {}, sceneFile = '', name = 'calibration' }) {
    this.scene = scene; this.topo = topo;
    this.params = params;                         // name → [lo, hi]
    this.fixed = fixed; this.tests = tests; this.validation = validation;
    this.replay = replaySettings(replay);
    this.sceneFile = sceneFile; this.name = name;
    this.backend = 'cpu'; this.device = null;     // setBackend('gpu', device) for the batched path
    this.maxInstances = 192;                      // GPU instances per batch (parameter vectors × tests)
    this.chunk = 400;                             // iterations per GPU control read-back
    this._scales = null;
  }

  /** calib.json (parsed) + a FileResolver for the scene, its mesh and the UTM csv files */
  static async load(calib, resolver, name = 'calibration') {
    const scText = await resolver.text(calib.scene);
    const sc = Scene.fromJSON(JSON.parse(scText), calib.scene.replace(/\.json$/i, '').split('/').pop());
    const meshRel = joinPath(dirOf(calib.scene), sc.mesh.path);
    const mesh = Mesh.fromBytes(new Uint8Array(await resolver.bytes(meshRel)), meshRel, sc.mesh.scale);
    const topo = buildTopology(mesh, sc.weave, sc.section.height, sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null);
    applyParams(sc, calib.fixed || {});
    const tests = async (key) => {
      const out = [];
      for (const t of calib[key] || []) {
        let curve = null;
        if (t.csv && resolver.has(t.csv)) { try { curve = parseUTMCsv(await resolver.text(t.csv), sc.units, { source: t.csv }); } catch (e) { if (!/not found|HTTP 404|ENOENT|no such file/i.test(e.message)) throw e; } }
        out.push(new Test(Probe.fromDict(t), curve, +(t.weight ?? 1.0), t.csv || ''));
      }
      return out;
    };
    const params = Object.fromEntries(Object.entries(calib.params).map(([k, v]) => [k, [+v[0], +v[1]]]));
    for (const k of Object.keys(params)) if (!PARAMS[k]) throw new Error(`unknown calibration parameter ${k}`);
    return new CalibrationSet({ scene: sc, topo, params, fixed: calib.fixed || {}, tests: await tests('tests'), validation: await tests('validation'), replay: calib.replay || {}, sceneFile: calib.scene, name });
  }

  /** a calibration on an already loaded model (the app's current scene) */
  static fromModel(scene, topo, { params = {}, fixed = {}, tests = [], validation = [], replay = {}, sceneFile = '' } = {}) {
    return new CalibrationSet({ scene: deepClone(scene), topo, params, fixed, tests, validation, replay, sceneFile });
  }

  toJSON() {
    return { scene: this.sceneFile, params: this.params, fixed: this.fixed, tests: this.tests.map(t => t.toJSON()), validation: this.validation.map(t => { const d = t.toJSON(); delete d.weight; return d; }), replay: this.replay };
  }

  setBackend(backend, device = null) { this.backend = backend; this.device = device; this._scales = null; }
  names() { return Object.keys(this.params); }
  allTests() { return this.tests.concat(this.validation); }

  // --- encoding (unit box, log-scaled where the parameter spans decades) -------------------------------------------
  encode(values) {
    return Float64Array.from(this.names(), (k) => {
      const [lo, hi] = this.params[k], v = values[k];
      return LOG_SCALED.has(k) ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
    });
  }
  decode(x) {
    const out = {};
    this.names().forEach((k, i) => {
      const [lo, hi] = this.params[k], xi = Math.min(Math.max(x[i], 0.0), 1.0);
      out[k] = LOG_SCALED.has(k) ? Math.exp(Math.log(lo) + xi * (Math.log(hi) - Math.log(lo))) : lo + xi * (hi - lo);
    });
    return out;
  }
  /** the scene's current values of the calibrated parameters (clamped into the box) */
  center() { const v = readParams(this.scene, this.names()); for (const k of this.names()) { const [lo, hi] = this.params[k]; v[k] = Math.min(Math.max(v[k], lo), hi); } return v; }

  // --- replay jobs --------------------------------------------------------------------------------------------------
  _jobs(values, tests, sc) {
    const rp = this.replay, pad = values.pad_stiffness ?? sc.utm.pad_stiffness;
    const settle = Math.max(rp.step_iter - rp.avg_iter, 1), avg = Math.max(rp.avg_iter, 1);
    return tests.map((t) => {
      if (rp.mode === 'force') {
        const fmax = rp.max_force || (t.curve ? maxAbs(t.curve.force) : 1.0);
        const forces = linspaceTail(0, fmax, rp.n_steps);
        const seq = forceSequence(forces, rp.cycles);
        return { test: t, mode: 'force', forces, seq, sched: seq.flatMap(s => s[2]), pad, settle, avg };
      }
      let dmax;
      if (t.curve) {
        const m = t.curve.monotone();
        dmax = maxOf(m.displacement);
        if (pad) dmax = Math.max(dmax - maxAbs(m.force) / pad, 0.3 * dmax);
      } else dmax = rp.max_disp || 1.0;
      if (rp.max_disp && t.curve) dmax = Math.min(dmax, rp.max_disp);
      const disps = linspaceTail(0, dmax, rp.n_steps);
      return { test: t, mode: 'displacement', disps, sched: disps.filter(d => d > 0), pad, settle, avg };
    });
  }

  // --- simulation ---------------------------------------------------------------------------------------------------
  /** relax once with `values`, replay every test → {name: branches (force) | {d, f} (displacement)} */
  async simulate(values, tests = null, opts = {}) { return (await this.simulateBatch([values], tests, opts))[0]; }

  /** a batch of parameter vectors → [{name: result}] (GPU: one batched run; CPU: one after the other) */
  async simulateBatch(valuesList, tests = null, opts = {}) {
    tests = tests || this.tests;
    if (this.backend === 'gpu' && this.device) {
      const out = [], T = Math.max(tests.length, 1) * Math.max(this.replay.replicas | 0, 1), per = Math.max(1, Math.floor(this.maxInstances / T));
      for (let i = 0; i < valuesList.length; i += per) out.push(...(await this._simulateGPU(valuesList.slice(i, i + per), tests, { ...opts, offset: i, count: valuesList.length })));
      return out;
    }
    const out = [];
    for (let i = 0; i < valuesList.length; i++) out.push(await this._simulateCPU(valuesList[i], tests, { ...opts, offset: i, count: valuesList.length }));
    return out;
  }

  async _simulateCPU(values, tests, opts = {}) {
    const sc = deepClone(this.scene);
    applyParams(sc, values);
    const rp = this.replay, prog = opts.onProgress;
    const jobs = this._jobs(values, tests, sc);
    const nInc = jobs.reduce((a, j) => a + j.sched.length, 0);
    let inc = 0;
    const R = Math.max(rp.replicas | 0, 1), perRelax = rp.replica_stage !== 'replay', reps = [];
    const relaxOne = async (r) => {
      if (prog) prog({ stage: 'relax', candidate: opts.offset || 0, count: opts.count || 1, frac: 0 });
      const sv = new CpuSolver(deepClone(sc), this.topo);
      if (perRelax) perturbReplica(sv, r, rp.perturb);
      await runAsync(sv, rp.relax_iter, opts);
      sv.internalize();
      if (rp.lock) sv.lockContacts();
      return sv;
    };
    const sv0 = await relaxOne(0);
    const onIncrement = () => { inc++; if (prog) prog({ stage: 'replay', candidate: opts.offset || 0, count: opts.count || 1, frac: inc / Math.max(nInc * R, 1) }); };
    for (let r = 0; r < R; r++) {
      let base = sv0;
      if (r > 0) { if (perRelax) base = await relaxOne(r); else { base = sv0.clone(); perturbReplica(base, r, rp.perturb); } }
      const out = {};
      for (const j of jobs) {
        if (j.mode === 'force') out[j.test.name] = await replayForce(base, j.test.probe, j.forces, rp.cycles, rp.step_iter, rp.avg_iter, j.pad, { ...opts, onIncrement });
        else out[j.test.name] = await replayDisplacement(base, j.test.probe, j.disps, rp.step_iter, j.pad, rp.avg_iter, { ...opts, onIncrement });
      }
      reps.push(out);
    }
    return R > 1 ? reps : reps[0];
  }

  /** fixed-point scales for every GPU batch of this calibration: from the base scene and the upper corner of the box
   *  (largest stiffnesses and penalties) with the largest probe force */
  _gpuScales(tests) {
    if (this._scales) return this._scales;
    const hi = {}, lo = {};
    for (const k of this.names()) { hi[k] = this.params[k][1]; lo[k] = this.params[k][0]; }
    const models = [];
    for (const v of [hi, lo, {}]) {
      const sc = deepClone(this.scene); applyParams(sc, v);
      const md = new CpuSolver(sc, this.topo);
      if (this.replay.mode === 'force') {
        let f = this.replay.max_force || 0;
        if (!f) for (const t of this.allTests()) if (t.curve) f = Math.max(f, maxAbs(t.curve.force));
        md.protocol = { mode: 1, sched: [f || 1.0] };
      }
      models.push(md);
    }
    this._scales = GpuSolver.computeScales(models);
    return this._scales;
  }

  async _simulateGPU(valuesList, tests, opts = {}) {
    const dev = this.device, rp = this.replay, prog = opts.onProgress, signal = opts.signal;
    const scales = this._gpuScales(tests);
    const R = Math.max(rp.replicas | 0, 1), perRelax = R > 1 && rp.replica_stage !== 'replay';
    // 1 · relax every parameter vector (one instance each; one per replica when the replicas start before the relaxation)
    const models = [], owner = [];
    valuesList.forEach((v, p) => {
      for (let r = 0; r < (perRelax ? R : 1); r++) {
        const sc = deepClone(this.scene); applyParams(sc, v);
        const md = new CpuSolver(sc, this.topo);
        if (perRelax) perturbReplica(md, r, rp.perturb);
        models.push(md); owner.push([p, r]);
      }
    });
    const g = new GpuSolver(dev, models, { scales });
    try {
      const relaxIter = rp.relax_iter || this.scene.solver.max_iter;             // Python: max_iter or scene max_iter
      let done = 0, first = true;
      while (done < relaxIter) {
        checkAbort(signal);
        const k = Math.min(this.chunk, relaxIter - done);
        const ctl = await g.step(k, { force: first });
        first = false; done += k;
        if (prog) prog({ stage: 'relax', candidate: opts.offset || 0, batch: valuesList.length, count: opts.count || valuesList.length, frac: done / relaxIter });
        if (ctl.every(c => c.done === 1)) break;
      }
      await g.downloadAll(null, { history: false, rebase: false });
    } finally { g.dispose(); }
    for (const md of models) { md.internalize(); if (rp.lock) md.lockContacts(); }
    // 2 · replay: one instance per (parameter vector, test), all in the same dispatches
    const rModels = [], index = [];
    const results = valuesList.map(() => Array.from({ length: R }, () => ({})));
    models.forEach((md, mi) => {
      const [p, r0] = owner[mi];
      for (let rep = perRelax ? r0 : 0; rep < (perRelax ? r0 + 1 : R); rep++) {
        for (const j of this._jobs(valuesList[p], tests, md.scene)) {
          if (!j.sched.length) { results[p][rep][j.test.name] = j.mode === 'force' ? forceSequence([], rp.cycles).map(([c, branch]) => ({ cycle: c, branch, d: [0.0], f: [0.0] })) : { d: [0.0], f: [0.0] }; continue; }
          if (j.sched.length > 64) throw new Error(`test ${j.test.name}: ${j.sched.length} load increments — the GPU replay records at most 64 (cycles × 2 × n_steps)`);
          const r = md.clone();
          if (!perRelax) perturbReplica(r, rep, rp.perturb);
          r.rebuildGoals();
          r.protocol = { mode: j.mode === 'force' ? 1 : 2, probe: probeNode(r.st, j.test.probe), sched: j.sched, settle: j.settle, avg: j.avg, axis: j.test.probe.axis, kPad: j.pad || 0 };
          rModels.push(r); index.push({ p, rep, j });
        }
      }
    });
    if (!rModels.length) return results.map(r => (R > 1 ? r : r[0]));
    const g2 = new GpuSolver(dev, rModels, { scales });
    let outs;
    try {
      const total = Math.max(...index.map(({ j }) => j.sched.length * (j.settle + j.avg))) + 2;
      let done = 0;
      while (true) {
        checkAbort(signal);
        const k = Math.min(this.chunk, Math.max(total - done, 1));
        const ctl = await g2.step(k, { force: false });
        done += k;
        if (prog) prog({ stage: 'replay', candidate: opts.offset || 0, batch: valuesList.length, count: opts.count || valuesList.length, frac: Math.min(done / total, 1) });
        if (ctl.every(c => c.done === 1)) break;
        if (done > total + this.chunk) throw new Error('GPU replay did not finish its load protocol');
      }
      outs = await g2.readAllOutputs();
    } finally { g2.dispose(); }
    index.forEach(({ p, rep, j }, i) => {
      const o = outs[i], res = results[p][rep];
      if (j.mode === 'force') {
        let k = 0, dPrev = 0.0, fPrev = 0.0;
        const br = [];
        for (const [c, branch, Fs] of j.seq) {
          const ds = [dPrev], fs = [fPrev];
          for (const F of Fs) { let d = o[4 * k]; if (j.pad) d += F / j.pad; ds.push(d); fs.push(F); k++; }
          dPrev = ds[ds.length - 1]; fPrev = fs[fs.length - 1];
          br.push({ cycle: c, branch, d: ds, f: fs });
        }
        res[j.test.name] = br;
      } else {
        const d = [0.0], f = [0.0];
        j.sched.forEach((v, k) => { d.push(v); f.push(o[4 * k + 1]); });
        if (j.pad) for (let k = 0; k < d.length; k++) d[k] += f[k] / j.pad;
        res[j.test.name] = { d, f };
      }
    });
    return results.map(r => (R > 1 ? r : r[0]));
  }

  // --- residuals / objective ----------------------------------------------------------------------------------------
  residuals(sims, tests = null) {
    const out = {};
    for (const t of tests || this.tests) {
      if (!t.curve || !(t.name in sims)) continue;
      if (this.replay.mode === 'force') { out[t.name] = this.residualCycles(t, sims[t.name]); continue; }
      const { d, f } = sims[t.name];
      const m = t.curve.monotone();
      out[t.name] = t.curve.residual(d, f) / Math.max(maxAbs(m.force), 1e-12);
    }
    return out;
  }

  /** force mode: at every applied force level compare the simulated travel with the measured travel at that force on the
   *  same branch of the same cycle; RMS over the fitted cycles, / measured travel range */
  residualCycles(t, simBranches) {
    const utm = new Map();
    for (const b of t.curve.cycles()) utm.set(`${b.cycle}|${b.branch}`, b);
    let lo = Infinity, hi = -Infinity; for (const v of t.curve.displacement) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const dmax = Math.max(hi - lo, 1e-12);
    const key = (b) => `${b.cycle}|${b.branch}`;
    let wanted = simBranches.filter(b => this.replay.fit_cycles.includes(b.cycle) && utm.has(key(b)));
    if (!wanted.length) wanted = simBranches.filter(b => utm.has(key(b)));       // fewer measured cycles than asked
    const err = [];
    for (const b of wanted) {
      const u = utm.get(key(b));
      const order = u.f.map((_, i) => i).sort((a, c) => u.f[a] - u.f[c]);       // stable
      const fu = [], du = [];
      for (const i of order) if (!fu.length || u.f[i] !== fu[fu.length - 1]) { fu.push(u.f[i]); du.push(u.d[i]); }
      const fmin = fu[0], fmax = fu[fu.length - 1];
      for (let i = 0; i < b.f.length; i++) {
        if (b.f[i] >= fmin && b.f[i] <= fmax) err.push((b.d[i] - interp(b.f[i], fu, du)) ** 2);
      }
    }
    if (!err.length) return 10.0;                 // nothing comparable: a large finite penalty, not NaN
    return Math.sqrt(pairwiseSum(err) / err.length) / dmax;               // np.mean: pairwise summation
  }

  objectiveOf(res) { let o = 0; for (const t of this.tests) if (t.name in res) o += t.weight * res[t.name]; return o; }

  /** residuals of one simulation or of a replica ensemble (mean, with the spread across replicas) */
  score(sims, tests = null) {
    const reps = Array.isArray(sims) ? sims : [sims];
    const per = reps.map(r => this.residuals(r, tests));
    const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
    const sd = (v) => { if (v.length < 2) return 0; const m = mean(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)); };
    const residuals = {}, spread = {};
    for (const k of Object.keys(per[0])) { const v = per.map(p => p[k]); residuals[k] = reps.length > 1 ? mean(v) : v[0]; spread[k] = sd(v); }
    const objs = per.map(p => this.objectiveOf(p));
    return { objective: reps.length > 1 ? mean(objs) : objs[0], objectiveStd: sd(objs), residuals, spread, sims: reps[0], replicas: reps.length > 1 ? reps : null };
  }

  async objective(values, opts = {}) { return (await this.objectiveBatch([values], opts))[0]; }

  /** [{objective, objectiveStd, residuals, spread, sims, replicas}] for a batch of parameter vectors */
  async objectiveBatch(valuesList, opts = {}) {
    const sims = await this.simulateBatch(valuesList, this.tests, opts);
    return sims.map((s) => this.score(s));
  }
}

// ------------------------------------------------------------------------------------------------ random numbers
export { Rng };

/** n stratified samples in [0, 1]^dim (one per stratum and dimension, strata shuffled per dimension) */
export function latinHypercube(n, dim, rng) {
  const pts = Array.from({ length: n }, (_, i) => Float64Array.from({ length: dim }, () => (i + rng.uniform()) / n));
  for (let j = 0; j < dim; j++) { const p = rng.permutation(n), col = pts.map(x => x[j]); p.forEach((src, i) => { pts[i][j] = col[src]; }); }
  return pts;
}

// ------------------------------------------------------------------------------------------------ optimisers
const boxPenalty = (x) => { let s = 0; for (const v of x) s += Math.max(0, Math.abs(v - 0.5) - 0.5) ** 2; return 10.0 * s; };
const xkey = (x) => Array.from(x, (v) => v.toString()).join(',');

/** memoised batch objective: values are computed in batches (fBatch → [{score, h}]) and looked up one by one; an
 *  evaluation counts (history, log) when the optimiser consumes it — a speculative point it never asks for does not */
class BatchFn {
  constructor(fBatch, onConsume = null) { this.fBatch = fBatch; this.onConsume = onConsume; this.cache = new Map(); this.batches = 0; this.computed = 0; }
  async prefetch(X) {
    const miss = [], seen = new Set();
    for (const x of X) { const k = xkey(x); if (!this.cache.has(k) && !seen.has(k)) { seen.add(k); miss.push(x); } }
    if (miss.length) { const v = await this.fBatch(miss); miss.forEach((x, i) => this.cache.set(xkey(x), { ...v[i], used: false })); this.batches++; this.computed += miss.length; }
  }
  async value(x) {
    const k = xkey(x);
    if (!this.cache.has(k)) await this.prefetch([x]);
    const e = this.cache.get(k);
    if (!e.used) { e.used = true; if (this.onConsume) this.onConsume(e.h); }
    return e.score;
  }
}

const MAXFEV = Symbol('maxfev');
/** Nelder–Mead as scipy.optimize.minimize(method='Nelder-Mead', adaptive=False): same steps, same termination
 *  (xatol / fatol / maxfev); with `speculative` the candidate points of an iteration (reflection, expansion, outside and
 *  inside contraction) or of a shrink are evaluated in one batch — the path is the sequential one, only faster on a GPU */
export async function nelderMead(fn, sim0, { maxfev = Infinity, maxiter = null, xatol = 1e-4, fatol = 1e-4, speculative = false, onIter = null, signal = null } = {}) {
  const rho = 1, chi = 2, psi = 0.5, sigma = 0.5;
  const N = sim0[0].length;
  if (maxiter == null) maxiter = maxfev === Infinity ? N * 200 : Infinity;
  let sim = sim0.map((x) => Float64Array.from(x)), fsim = new Array(N + 1).fill(Infinity);
  let fcalls = 0;
  const func = async (x) => { if (fcalls >= maxfev) throw MAXFEV; fcalls++; return fn.value(x); };
  const sortSim = () => { const ind = fsim.map((_, i) => i).sort(byValue(fsim)); sim = ind.map(i => sim[i]); fsim = ind.map(i => fsim[i]); };
  try {
    if (speculative) await fn.prefetch(sim.slice(0, Math.min(N + 1, maxfev)));
    for (let k = 0; k < N + 1; k++) fsim[k] = await func(sim[k]);
  } catch (e) { if (e !== MAXFEV) throw e; } finally { sortSim(); }
  let iterations = 1;
  const comb = (a, xa, b, xb) => Float64Array.from(xa, (v, i) => a * v + b * xb[i]);
  while (fcalls < maxfev && iterations < maxiter) {
    checkAbort(signal);
    try {
      let dx = 0; for (let j = 1; j <= N; j++) for (let i = 0; i < N; i++) dx = Math.max(dx, Math.abs(sim[j][i] - sim[0][i]));
      let df = 0; for (let j = 1; j <= N; j++) df = Math.max(df, Math.abs(fsim[0] - fsim[j]));
      if (dx <= xatol && df <= fatol) break;
      const xbar = new Float64Array(N); for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) xbar[i] += sim[j][i]; for (let i = 0; i < N; i++) xbar[i] /= N;
      const worst = sim[N];
      const xr = comb(1 + rho, xbar, -rho, worst), xe = comb(1 + rho * chi, xbar, -rho * chi, worst);
      const xc = comb(1 + psi * rho, xbar, -psi * rho, worst), xcc = comb(1 - psi, xbar, psi, worst);
      if (speculative) await fn.prefetch([xr, xe, xc, xcc].slice(0, Math.max(maxfev - fcalls, 1)));
      const fxr = await func(xr);
      let doshrink = false;
      if (fxr < fsim[0]) {
        const fxe = await func(xe);
        if (fxe < fxr) { sim[N] = xe; fsim[N] = fxe; } else { sim[N] = xr; fsim[N] = fxr; }
      } else if (fxr < fsim[N - 1]) { sim[N] = xr; fsim[N] = fxr; }
      else if (fxr < fsim[N]) {
        const fxc = await func(xc);
        if (fxc <= fxr) { sim[N] = xc; fsim[N] = fxc; } else doshrink = true;
      } else {
        const fxcc = await func(xcc);
        if (fxcc < fsim[N]) { sim[N] = xcc; fsim[N] = fxcc; } else doshrink = true;
      }
      if (doshrink) {
        const pts = []; for (let j = 1; j <= N; j++) pts.push(Float64Array.from(sim[0], (v, i) => v + sigma * (sim[j][i] - v)));   // scipy: s0 + σ(sj − s0)
        if (speculative) await fn.prefetch(pts.slice(0, Math.max(maxfev - fcalls, 1)));
        for (let j = 1; j <= N; j++) { sim[j] = pts[j - 1]; fsim[j] = await func(sim[j]); }
      }
      iterations++;
    } catch (e) { if (e !== MAXFEV) throw e; break; } finally { sortSim(); }
    if (onIter) onIter({ iterations, fcalls, best: fsim[0], x: sim[0] });
  }
  const message = fcalls >= maxfev ? 'Maximum number of function evaluations has been exceeded.' : iterations >= maxiter ? 'Maximum number of iterations has been exceeded.' : 'Optimization terminated successfully.';
  return { x: sim[0], fun: fsim[0], nfev: fcalls, nit: iterations, message };
}

/** symmetric eigen-decomposition (cyclic Jacobi) of an n×n matrix given as array of rows → {values, vectors (columns)} */
function eigSym(A) {
  const n = A.length, a = A.map(r => Float64Array.from(r)), V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { values: a.map((r, i) => r[i]), vectors: V };
}

/** (μ/μ_w, λ)-CMA-ES (Hansen 2016 tutorial defaults) in the unit box; every generation is one batch */
export async function cmaes(fn, x0, sigma0, { lambda = null, maxEval = 200, rng = new Rng(1), tolX = 1e-3, tolFun = 1e-4, onGen = null, signal = null } = {}) {
  const n = x0.length;
  const lam = lambda || 4 + Math.floor(3 * Math.log(n)), mu = Math.floor(lam / 2);
  let w = Array.from({ length: mu }, (_, i) => Math.log((lam + 1) / 2) - Math.log(i + 1));
  const ws = w.reduce((a, b) => a + b, 0); w = w.map(v => v / ws);
  const mueff = 1 / w.reduce((a, v) => a + v * v, 0);
  const cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n), cs = (mueff + 2) / (n + mueff + 5);
  const c1 = 2 / ((n + 1.3) ** 2 + mueff), cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff));
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs, chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
  let m = Float64Array.from(x0), sigma = sigma0;
  let C = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  let B = C.map(r => Float64Array.from(r)), D = new Float64Array(n).fill(1);
  const pc = new Float64Array(n), ps = new Float64Array(n);
  let evals = 0, gen = 0, best = { f: Infinity, x: m }, message = 'maximum number of evaluations reached';
  const hist = [];
  while (evals + lam <= Math.max(maxEval, lam)) {
    checkAbort(signal);
    const Z = [], Y = [], X = [];
    for (let k = 0; k < lam; k++) {
      const z = Float64Array.from({ length: n }, () => rng.normal());
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += B[i][j] * D[j] * z[j]; y[i] = s; }
      Z.push(z); Y.push(y); X.push(Float64Array.from(m, (v, i) => v + sigma * y[i]));
    }
    await fn.prefetch(X);
    const F = []; for (const x of X) F.push(await fn.value(x));
    evals += lam; gen++;
    const idx = F.map((_, i) => i).sort(byValue(F));
    if (F[idx[0]] < best.f) best = { f: F[idx[0]], x: X[idx[0]] };
    const mOld = m;
    m = new Float64Array(n); for (let k = 0; k < mu; k++) for (let i = 0; i < n; i++) m[i] += w[k] * X[idx[k]][i];
    const yw = Float64Array.from(m, (v, i) => (v - mOld[i]) / sigma);
    // C^-1/2 yw = B D^-1 Bᵀ yw
    const t1 = new Float64Array(n); for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += B[i][j] * yw[i]; t1[j] = s / D[j]; }
    const cinv = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += B[i][j] * t1[j]; cinv[i] = s; }
    const csn = Math.sqrt(cs * (2 - cs) * mueff);
    for (let i = 0; i < n; i++) ps[i] = (1 - cs) * ps[i] + csn * cinv[i];
    const psn = Math.sqrt(ps.reduce((a, v) => a + v * v, 0));
    const hsig = psn / Math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    const ccn = Math.sqrt(cc * (2 - cc) * mueff);
    for (let i = 0; i < n; i++) pc[i] = (1 - cc) * pc[i] + hsig * ccn * yw[i];
    const Cn = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let rmu = 0; for (let k = 0; k < mu; k++) rmu += w[k] * Y[idx[k]][i] * Y[idx[k]][j];
      Cn[i][j] = (1 - c1 - cmu) * C[i][j] + c1 * (pc[i] * pc[j] + (1 - hsig) * cc * (2 - cc) * C[i][j]) + cmu * rmu;
    }
    C = Cn;
    sigma *= Math.exp((cs / damps) * (psn / chiN - 1));
    const e = eigSym(C);
    B = e.vectors; D = Float64Array.from(e.values, v => Math.sqrt(Math.max(v, 1e-20)));
    hist.push(F[idx[0]]);
    if (onGen) onGen({ gen, evals, best: best.f, mean: Array.from(m), sigma, f: F });
    const recent = hist.slice(-Math.max(3, Math.ceil(10 + 30 * n / lam)));
    if (sigma * Math.max(...D) < tolX) { message = 'step size below tolX'; break; }
    if (recent.length >= 3 && Math.max(...recent) - Math.min(...recent) < tolFun && Math.max(...F) - Math.min(...F) < tolFun) { message = 'objective flat (tolFun)'; break; }
  }
  return { x: best.x, fun: best.f, nfev: evals, ngen: gen, mean: m, sigma, message };
}

function simplexAround(x0, step) {
  const n = x0.length, S = [Float64Array.from(x0)];
  for (let i = 0; i < n; i++) { const x = Float64Array.from(x0); x[i] = x0[i] + step <= 1.0 ? x0[i] + step : x0[i] - step; S.push(x); }
  return S;
}

/** Latin-hypercube screening, then Nelder–Mead ('nm', the Python fit) or CMA-ES ('cmaes', population batches) in the
 *  unit (log-scaled) box from the best point. → {best, objective, residuals, evaluations, message, history} */
export async function fit(cs, { nInit = 8, maxEval = 60, seed = 1, method = 'nm', speculative = null, lambda = null, log = () => {}, onEval = null, onProgress = null, signal = null } = {}) {
  const rng = new Rng(seed), names = cs.names(), dim = names.length;
  const history = [];
  if (speculative == null) speculative = cs.backend === 'gpu';
  const fBatch = async (X) => {
    const vals = X.map(x => cs.decode(x));
    const t0 = now();
    const res = await cs.objectiveBatch(vals, { signal, onProgress });
    const dt = (now() - t0) / 1000;
    return res.map((r, i) => ({
      score: r.objective + boxPenalty(X[i]),
      h: { objective: r.objective, objectiveStd: r.objectiveStd, values: vals[i], residuals: r.residuals, x: Array.from(X[i]), seconds: dt / X.length },
    }));
  };
  const consume = (h) => {
    history.push(h);
    log(`eval ${String(history.length).padStart(3)}  obj ${h.objective.toFixed(4)}  ` + names.map(k => `${k}=${h.values[k].toPrecision(4)}`).join('  ') + `  (${h.seconds.toFixed(1)} s)`);
    if (onEval) onEval(h, history);
  };
  const fn = new BatchFn(fBatch, consume);
  const X0 = latinHypercube(nInit, dim, rng);
  await fn.prefetch(X0);
  const scores = []; for (const x of X0) scores.push(await fn.value(x));
  let ib = 0; scores.forEach((s, i) => { if (s < scores[ib]) ib = i; });
  const xb = X0[ib];
  let message;
  const budget = Math.max(maxEval - nInit, 1);
  if (method === 'cmaes') {
    const r = await cmaes(fn, xb, 0.2, { lambda, maxEval: budget, rng, signal });
    message = `CMA-ES: ${r.message} (${r.ngen} generations)`;
  } else {
    const r = await nelderMead(fn, simplexAround(xb, 0.15), { maxfev: budget, xatol: 0.01, fatol: 1e-3, speculative, signal });
    message = r.message;
  }
  let best = history[0];
  for (const h of history) if (h.objective < best.objective || Number.isNaN(best.objective)) best = h;
  return { best: best.values, objective: best.objective, residuals: best.residuals, evaluations: history.length, computed: fn.computed, batches: fn.batches, message, method,
    history: history.map(h => ({ objective: h.objective, ...h.values })) };
}

/** one-at-a-time sweep across each parameter's bounds (one batch on the GPU) → name → [{value, objective, residuals, sims}] */
export async function sensitivity(cs, center, n = 5, opts = {}) {
  const cands = [], tags = [];
  for (const k of cs.names()) {
    const [lo, hi] = cs.params[k];
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : i / (n - 1);                                   // np.linspace(lo, hi, 1) = [lo]
      const v = LOG_SCALED.has(k) ? Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo))) : lo + t * (hi - lo);
      cands.push({ ...center, [k]: v }); tags.push([k, v]);
    }
  }
  const res = await cs.objectiveBatch(cands, opts);
  const out = {};
  res.forEach((r, i) => { const [k, v] = tags[i]; (out[k] ||= []).push({ value: v, objective: r.objective, objectiveStd: r.objectiveStd, residuals: r.residuals, sims: r.sims }); });
  return out;
}

export async function validate(cs, values, opts = {}) {
  const sims = await cs.simulate(values, cs.validation, opts);
  const sc = cs.score(sims, cs.validation);
  return { residuals: sc.residuals, spread: sc.spread, sims: sc.sims, replicas: sc.replicas };
}

// ------------------------------------------------------------------------------------------------ synthetic experiment
function interpGrid(t, n, v) { const xp = Array.from({ length: n }, (_, i) => (n === 1 ? 0 : i / (n - 1))); return t.map(x => interp(x, xp, v)); }
const fmt = (v, d) => v.toFixed(d);

/** UTM-like csv texts for every test and validation probe from the model at `trueValues` (+ Gaussian noise of
 *  noise × peak travel (force mode) / peak force (displacement mode)); the tests' curves are replaced. → {name: csv} */
export async function makeSynthetic(cs, trueValues, { noise = 0.02, nPoints = 40, seed = 0, onProgress = null, signal = null } = {}) {
  const rng = new Rng(seed);
  const all = cs.allTests();
  let sims = await cs.simulate(trueValues, all, { onProgress, signal });
  if (Array.isArray(sims)) sims = sims[0];
  const written = {};
  const u = cs.scene.units;
  for (const t of all) {
    let dd = [], ff = [];
    if (cs.replay.mode === 'force') {
      const k = Math.max(Math.floor(nPoints / (2 * cs.replay.cycles)), 4);
      const tt = Array.from({ length: k }, (_, i) => i / k);
      for (const b of sims[t.name]) { dd.push(...interpGrid(tt, b.d.length, b.d)); ff.push(...interpGrid(tt, b.f.length, b.f)); }
      // as calibrate.make_synthetic: np.concatenate(dd + [dd[-1][-1:]]) repeats the last *interpolated* sample
      // (probably meant to close the trace at the end of the last branch — kept identical to the Python program)
      dd.push(dd[dd.length - 1]); ff.push(ff[ff.length - 1]);
      const s = noise * Math.max(maxAbs(dd), 1e-12);
      dd = dd.map(v => v + s * rng.normal());
    } else {
      const { d, f } = sims[t.name], dm = maxOf(d);
      dd = Array.from({ length: nPoints }, (_, i) => (nPoints === 1 ? 0 : dm * i / (nPoints - 1)));
      ff = dd.map(x => interp(x, d, f));
      const s = noise * Math.max(maxAbs(ff), 1e-12);
      ff = ff.map(v => v + s * rng.normal());
    }
    const rows = [`Displacement [${u.length}],Force [${u.force}]`];
    dd.forEach((a, i) => rows.push(`${fmt(a, 5)},${fmt(ff[i], 4)}`));
    const csv = rows.join('\n') + '\n';
    t.curve = parseUTMCsv(csv, u, { source: t.csv || `${t.name}.csv` });
    if (!t.csv) t.csv = `${t.name}.csv`;
    written[t.name] = csv;
  }
  return written;
}

// ------------------------------------------------------------------------------------------------ parameter sets
/** a calibrated parameter set, portable to other weave topologies and scenes: values in the calibration's units,
 *  with the material / section it was calibrated on (the domain in which the contact parameters transfer) */
export function parameterSet(cs, values, { objective = null, residuals = null, validation = null, method = '', note = '' } = {}) {
  const sc = cs.scene;
  const all = { ...cs.fixed, ...values };
  return {
    kind: 'weavesim-parameter-set', version: 1, created: new Date().toISOString(),
    units: sc.units.toJSON(),
    values: Object.fromEntries(Object.entries(all).filter(([k]) => PARAMS[k])),
    fitted: cs.names(), fixed: { ...cs.fixed },
    classes: Object.fromEntries(Object.keys(all).filter(k => PARAMS[k]).map(k => [k, PARAM_CLASS[k] || 'model'])),
    quality: { objective, residuals, validation },
    calibration: { scene: cs.sceneFile, method, tests: cs.tests.map(t => t.toJSON()), validation: cs.validation.map(t => t.toJSON()), replay: cs.replay, bounds: cs.params },
    domain: { section: structuredClone(sc.section), material: structuredClone(sc.material), contact_epsilon: sc.contact.epsilon },
    note,
  };
}

/** apply a parameter set to a scene (unit conversion by dimension); test-rig parameters (pad_stiffness) are skipped
 *  unless includeTest. → {applied: {name: value in scene units}, warnings: [...]} */
export function applyParameterSet(scene, set, { includeTest = false, only = null } = {}) {
  if (!set || set.kind !== 'weavesim-parameter-set') throw new Error('not a WeaveSim parameter set');
  const src = UnitSystem.fromJSON(set.units), dst = scene.units, applied = {}, warnings = [];
  for (const [k, v] of Object.entries(set.values || {})) {
    if (!PARAMS[k] || (only && !only.includes(k))) continue;
    if (PARAM_CLASS[k] === 'test' && !includeTest) continue;
    const val = v * paramFactor(k, src, dst);
    PARAMS[k][1](scene, val); applied[k] = val;
  }
  const dom = set.domain || {};
  if (dom.section) {
    const L = src.lengthFactorTo(dst), w = dom.section.width * L, h = dom.section.height * L;
    const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
    if (rel(scene.section.width, w) > 0.02 || rel(scene.section.height, h) > 0.02)
      warnings.push(`section ${scene.section.width}×${scene.section.height} ${dst.length} differs from the calibration's ${+w.toPrecision(4)}×${+h.toPrecision(4)}: the contact penalties (strength, stick_strength) and the activation were identified for that lath`);
  }
  if (scene.hybrid && scene.hybrid.enabled && scene.hybrid.groups.length) warnings.push('lath groups: material values went to group 1 only (the scene\'s own material); set the other groups explicitly');
  return { applied, warnings };
}
