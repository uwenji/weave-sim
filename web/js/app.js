// WeaveSim Web — the app: scene and model management, the panels, the solve loop (batched WebGPU solver or the CPU
// f64 reference), selection / picking (frames, and whole laths for the lath groups), the UTM replay, the Rhino export,
// the calibration panel (calib_ui.js), the discretisation (disc_ui.js), the fastened-structure protocol under Run
// (protocol_ui.js: Zero → settle → Lock) and the Display sidebar (display_ui.js: colours, legend, diagrams, joints). Mirrors the Python viewer (weavesim/app): File · Weave · Units & material · Contact ·
// Loads · Anchors · Solve · Discretisation · UTM · Calibration on the left, the solve / discretisation strip pinned on
// top as in LocalSH, results and display on the sidebar.
import { Scene, selectorLabel, fixLabel } from './scene.js';
import { Mesh } from './mesh.js';
import { buildTopology, fkey, fsplit } from './topology.js';
import { UnitSystem } from './units.js';
import { CpuSolver } from './solver_cpu.js';
import { GpuSolver, getDevice } from './solver_gpu.js';
import { resolveSelector, anchorMasks, nodeQuats } from './model.js';
import { toFrame } from './quat.js';
import { View } from './view.js';
import { drawPlot } from './plot.js';
import { initDisplay, LAYER_LABEL, SEG_LAYERS, KNOT_LAYERS } from './display_ui.js';
import { initDiscretize } from './disc_ui.js';
import { initProtocol } from './protocol_ui.js';
import { discretizeParams } from './discretize.js';
import { exportRhino } from './export3dm.js';
import { parseUTMCsv } from './utm.js';
import { sectionStiffness, IxEff, IyEff, JEff, sectionArea, beamFactors, beamActivations } from './params.js';
import { CalibrationSet, Test, Probe, replaySettings, REPLAY_DEFAULTS } from './calibrate.js';
import { initCalibration } from './calib_ui.js';

const $ = (id) => document.getElementById(id);
const qsa = (sel, root = document) => [...root.querySelectorAll(sel)];
export const fmt = (v, p = 4) => {
  if (v == null || !Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(p - 1) : String(+v.toPrecision(p));
};
const store = { get(k, d) { try { const v = localStorage.getItem('weavesim.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem('weavesim.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } } };

const EXAMPLES = [
  { label: 'v5 dome (cm)', path: '../examples/scenes/v5dome_cm.json' },
  { label: 'dome basic', path: '../examples/scenes/dome_basic.json' },
  { label: 'dual v5 dome (cm)', path: '../examples/scenes/dualv5dome_cm.json' },
  { label: 'dual v5 dome, lath groups', path: '../examples/scenes/dualv5dome_cm_hybrid.json' },
  { label: 'tree (cm)', path: '../examples/scenes/tree_cm.json' },
  { label: 'v5 dome failure', path: '../examples/scenes/v5dome_failure.json' },
  { label: 'calibration dome (force tests)', path: '../examples/calibration/force/dome_calib_scene.json' },
];
const AXIS_VECTORS = { 0: [0, 1], 1: [0, -1], 2: [1, 1], 3: [1, -1], 4: [2, 1], 5: [2, -1] };
/** the scene's UTM probe, or null (scenes may carry an empty {} — falsy in the Python program) */
export const probeOf = (sc) => (sc && sc.utm && sc.utm.probe && Array.isArray(sc.utm.probe.frame) && sc.utm.probe.frame.length === 2 ? sc.utm.probe : null);

export const A = {
  scene: null, mesh: null, meshBytes: null, meshName: '', topo: null, sceneName: 'scene', sceneURL: null,
  model: null, gpu: null, device: null, gpuAvailable: false, backend: 'cpu',
  running: false, busy: false, runFirst: false, runUntil: 0, gpuChunk: 64, lastSync: 0, elapsed: 0,
  results: null, baseline: null, goalSig: '', paramSig: '',
  selection: [], lathSel: new Set(), mode: 'none', stripTab: 'solve',
  utm: null, utmText: null, utmName: '', probeTrace: [], replay: null,
  view: null, dirty: { results: false },
};
let view;
if (typeof window !== 'undefined') window.weavesim = A;          // console access: weavesim.model, weavesim.scene, …

// =========================================================================================================== status
function status(msg, cls = '') { const el = $('st-msg'); el.textContent = msg; el.className = 'st dim' + (cls ? ' ' + cls : ''); }
function updateStatus() {
  const sc = A.scene, t = A.topo, m = A.model;
  $('st-units').textContent = sc ? `[${sc.units}]` : '';
  if (t) $('st-topo').innerHTML = `${t.nStrands} strands · ${t.nKnots} knots · ${t.familyCount} families` + (t.parityConflicts ? ` · <span class="warn">${t.parityConflicts} parity conflicts</span>` : '');
  else $('st-topo').textContent = 'no model';
  if (m) {
    const st = m.st, conv = m.converged;
    $('st-run').innerHTML = `· it ${st.iteration} · residual ${Number.isFinite(st.residual) ? st.residual.toExponential(2) : '—'} · ${A.elapsed.toFixed(1)} s` + (conv ? ' · <span class="ok">converged</span>' : '');
  } else $('st-run').textContent = '';
  const pill = $('st-backend');
  pill.textContent = A.backend === 'webgpu' ? `WebGPU · ${A.device?.wsInfo?.vendor || A.device?.wsInfo?.architecture || 'gpu'}` : 'CPU · f64';
  pill.className = 'st pill ' + (A.backend === 'webgpu' ? 'gpu' : 'cpu');
}

// =========================================================================================================== model
function strandHeightFn(sc) { return sc.hybrid.enabled ? (s) => sc.heightOfStrand(s) : null; }

/** a scene and its mesh bytes → topology → model */
function setScene(sc, meshBytes, meshName, opts = {}) {
  stopRun();
  A.scene = sc; A.meshBytes = meshBytes; A.meshName = meshName; A.sceneName = sc.name || 'scene';
  A.mesh = Mesh.fromBytes(meshBytes, meshName, sc.mesh.scale);
  A.selection = []; A.lathSel = new Set(); A.utm = A.utm || null; A.probeTrace = [];
  sc.discretize = discretizeParams(sc.discretize);
  if (A.mode === 'laths') setMode('laths');
  rebuildTopology(opts);
  syncPanels();
  calib?.onScene(); disc?.onScene();
}

function rebuildTopology(opts = {}) {
  const sc = A.scene;
  try {
    A.topo = buildTopology(A.mesh, sc.weave, sc.section.height, strandHeightFn(sc));
  } catch (e) { status('topology failed: ' + e.message, 'warn'); A.topo = null; return; }
  A.selection = A.selection.filter(([s, k]) => A.topo.hasFrame(s, k));
  A.lathSel = new Set([...A.lathSel].filter(s => s < A.topo.nStrands));
  resetModel(opts);
}

/** a fresh solver on the woven geometry (Reset) */
function resetModel(opts = {}) {
  stopRun();
  disposeGpu();
  if (!A.topo) return;
  A.model = new CpuSolver(A.scene, A.topo);
  A.goalSig = goalSignature(); A.paramSig = paramSignature();
  A.results = null; A.baseline = null; A.elapsed = 0; A.probeTrace = [];
  proto?.reset();
  if (A.backend === 'webgpu') makeGpu();
  view.setModel({ topo: A.topo, st: A.model.st, scene: A.scene }, { keepCamera: !!opts.keepCamera });
  view.setStrandSelection(A.lathSel);
  refreshOverlays(); display.applyColouring(); display.refreshJoints(); display.refreshDiagram(); drawChart(); updateStatus(); updateSummaries();
}

function makeGpu() {
  try {
    A.gpu = new GpuSolver(A.device, [A.model], { perSubmit: 64 });
    A.gpu.compileInfo().then((msgs) => { const err = msgs.filter(m => m.startsWith('error')); if (err.length) { console.error(err.join('\n')); status('WebGPU shader errors — switched to the CPU solver', 'warn'); setBackend('cpu'); } });
  } catch (e) { console.error(e); status('WebGPU solver failed (' + e.message + ') — CPU solver', 'warn'); A.gpu = null; A.backend = 'cpu'; }
  A.gpuChunk = 64;
}
function disposeGpu() { if (A.gpu) { A.gpu.dispose(); A.gpu = null; } }

/** pull the GPU state into the host model (positions, frames, contact state, forces, history) */
async function syncFromGpu() { if (A.gpu) await A.gpu.download(0); }

async function setBackend(b) {
  if (b === A.backend) return;
  if (b === 'webgpu' && !A.gpuAvailable) { status('WebGPU is not available in this browser (secure context: localhost or https)', 'warn'); return; }
  const wasRunning = A.running;
  stopRun();
  await waitIdle();
  if (A.gpu) { await syncFromGpu(); disposeGpu(); }
  A.backend = b;
  if (b === 'webgpu' && A.model) makeGpu();
  qsa('#backend-chips .chip').forEach(c => c.classList.toggle('on', c.dataset.backend === b));
  updateBackendReadout(); updateStatus();
  calib?.onBackend();
  if (wasRunning) startRun();
}
function updateBackendReadout() {
  const g = A.gpu;
  $('ro-backend').innerHTML = A.backend === 'webgpu'
    ? `f32 arithmetic with 64-bit fixed-point positions, frames, contact parameters and force sums${g ? ` (2^-${g.PB} length, 2^-${g.FB} force)` : ''}; exact, order-independent accumulation — bit-reproducible on this GPU. Contacts stiffer than ~k·w·6e-8 per crossing leave a residual floor: <span class="acc">f64 polish</span> finishes on the CPU.`
    : 'double precision on the CPU — the reference (same algorithm as the Python solver, agreement to rounding until the stick–slip chaos takes over)';
}
const waitIdle = async () => { while (A.busy) await new Promise(r => setTimeout(r, 10)); };
/** run fn with the solve loop held off (no chunk can start in between) */
async function exclusive(fn) { await waitIdle(); A.busy = true; try { return await fn(); } finally { A.busy = false; } }

// --- live edits (Python viewer: goal_signature → rebuild_goals) ------------------------------------------------------
function goalSignature() {
  const sc = A.scene; if (!sc) return '';
  const c = { ...sc.contact }; delete c.locked;
  return JSON.stringify({ loads: sc.loads, anchors: sc.anchors, material: sc.material, section: sc.section, beam: sc.beam, contact: c, hybrid: sc.hybrid, mass_scale: sc.solver.mass_scale, prestress: sc.solver.prestress });
}
function paramSignature() { const sc = A.scene; return sc ? JSON.stringify({ solver: sc.solver, contact: sc.contact }) : ''; }

/** apply scene edits to the running model without resetting it */
async function syncSolver() {
  if (!A.model) return;
  const sig = goalSignature(), psig = paramSignature();
  if (sig === A.goalSig && psig === A.paramSig) return;
  if (sig !== A.goalSig) {
    await syncFromGpu();
    try {
      A.model.rebuildGoals();
      if (A.gpu) { try { A.gpu.updateParams(0); } catch (e) { disposeGpu(); makeGpu(); } }
      status('live update: loads / anchors / material applied to the running state (Reset restarts from the woven geometry)');
    } catch (e) { status(`live update failed (${e.message}) — Reset`, 'warn'); }
  } else if (A.gpu) A.gpu._uploadParams(0);
  A.goalSig = sig; A.paramSig = psig;
  if (A.scene.contact.locked !== undefined && A.model.contact) {
    const want = !!A.scene.contact.locked, have = A.model.contact.locked.every(Boolean);
    if (want !== have) { await syncFromGpu(); (want ? A.model.lockContacts() : A.model.unlockContacts()); if (A.gpu) A.gpu.updateParams(0); proto?.lockChanged(want); }
  }
  refreshOverlays();
}

// =========================================================================================================== solve loop
function startRun() {
  if (!A.model) return;
  A.running = true; A.runFirst = true; A.runUntil = A.model.st.iteration + Math.max(1, A.scene.solver.max_iter | 0);
  if (proto) A.runUntil = proto.runTarget(A.runUntil);                 // a pending settle stops the run at its end
  A.model.converged = false;
  $('btn-run').textContent = 'Pause'; $('btn-run').classList.add('running');
}
function stopRun() { A.running = false; const b = $('btn-run'); if (b) { b.textContent = 'Run'; b.classList.remove('running'); } }

async function solveChunk() {
  if (A.busy || !A.model) return;
  A.busy = true;
  try {
    await syncSolver();
    const t0 = performance.now(), m = A.model, sp = A.scene.solver;
    if (A.gpu) {
      const left = A.runUntil - m.st.iteration;
      const k = Math.max(1, Math.min(A.gpuChunk, left));
      const ctl = (await A.gpu.step(k, { force: A.runFirst }))[0];
      A.runFirst = false;
      const dt = performance.now() - t0;
      A.gpuChunk = dt < 30 ? Math.min(A.gpuChunk * 2, 4096) : dt > 120 ? Math.max(8, A.gpuChunk >> 1) : A.gpuChunk;
      m.st.iteration = ctl.iter; m.st.residual = ctl.resid;
      const done = ctl.done === 1 || ctl.iter >= A.runUntil;
      if (done || performance.now() - A.lastSync > 120) { await syncFromGpu(); A.lastSync = performance.now(); afterSteps(); }
      if (done) { m.converged = ctl.done === 1 && ctl.resid <= sp.tol; stopRun(); proto.onRunEnd(); finishRun(); }
    } else {
      const every = Math.max(1, sp.live_every | 0);
      while (performance.now() - t0 < 14 && !m.converged && m.st.iteration < A.runUntil) { if (A.runFirst) { m.converged = false; A.runFirst = false; } m.step(Math.min(every, A.runUntil - m.st.iteration)); }
      afterSteps();
      if (m.converged || m.st.iteration >= A.runUntil) { stopRun(); proto.onRunEnd(); finishRun(); }
    }
    A.elapsed += (performance.now() - t0) / 1000;
  } catch (e) { console.error(e); status('solver error: ' + e.message, 'warn'); stopRun(); }
  finally { A.busy = false; }
}

/** Step: n iterations (stops early at convergence, as Solver.step) */
async function stepN(n) {
  if (!A.model || A.busy) return;
  stopRun();
  n = proto.clampStep(n);                                            // never past the end of a pending settle
  A.busy = true;
  try {
    await syncSolver();
    const t0 = performance.now();
    if (A.gpu) { await A.gpu.step(n, { force: true }); await syncFromGpu(); }
    else { A.model.converged = false; A.model.step(n); }
    A.elapsed += (performance.now() - t0) / 1000;
    afterSteps(); proto.onRunEnd(); finishRun();
  } finally { A.busy = false; }
}

function afterSteps() {
  view.updateGeometry(A.model.st);
  proto.sync();
  if (display.hasJoints()) display.refreshJointsThrottled();
  recordProbe();
  A.dirty.results = true;
  drawChart(); updateStatus();
}
function finishRun() { refreshResults(); if (display.hasJoints()) display.refreshJoints(); updateStatus(); }

/** Python _record_probe: probe travel along its axis vs the point loads on the probe frame (strip › probe) */
function recordProbe() {
  const pr = probeOf(A.scene); if (!pr || !A.model) return;
  const nid = A.model.st.nodeFrameIds.get(fkey(pr.frame[0], pr.frame[1])); if (nid === undefined) return;
  const ax = pr.axis || [0, 0, -1], n = Math.hypot(...ax) || 1, a = ax.map(v => v / n);
  const P0 = A.topo.frameOrigin(pr.frame[0], pr.frame[1]), X = A.model.st.X;
  const d = (X[3 * nid] - P0[0]) * a[0] + (X[3 * nid + 1] - P0[1]) * a[1] + (X[3 * nid + 2] - P0[2]) * a[2];
  let f = 0; for (const pl of A.scene.loads.point) if (pl.frame[0] === pr.frame[0] && pl.frame[1] === pr.frame[1]) f += pl.F[0] * a[0] + pl.F[1] * a[1] + pl.F[2] * a[2];
  const tr = A.probeTrace; if (!tr.length || tr[tr.length - 1][0] !== d) tr.push([d, f]); if (tr.length > 4000) tr.splice(0, tr.length - 4000);
}

// =========================================================================================================== chart
function drawChart() {
  if (A.stripTab === 'disc') { disc?.redraw(); return; }
  const cv = $('chart'), which = $('sel-chart').value, m = A.model, H = m?.st.history, sc = A.scene;
  const lg = $('chart-legend');
  if (which === 'utm') {
    const series = [];
    if (A.utm) series.push({ x: A.utm.displacement, y: A.utm.force, color: cssv('--fg'), dots: true, r: 1.6, label: 'UTM' });
    if (A.replay) for (const b of replayBranches(A.replay)) series.push({ x: b.d, y: b.f, color: cssv('--chart-1'), width: 1.6, label: series.some(s => s.label === 'replay') ? '' : 'replay' });
    if (A.probeTrace.length) series.push({ x: A.probeTrace.map(p => p[0]), y: A.probeTrace.map(p => p[1]), color: cssv('--chart-2'), label: 'live probe' });
    lg.innerHTML = '<span class="dim">probe force against crosshead travel</span>';
    drawPlot(cv, { series, xlabel: sc ? `travel [${sc.units.length}]` : '', compact: true, empty: 'no UTM curve or probe yet' });
  } else if (which === 'parts' && H && H.strain.length) {
    const w = chartWindow(H), mk = (arr, c, label, wd) => ({ x: w.xs, y: arr.slice(w.n0), color: cssv(c), label, width: wd });
    lg.innerHTML = `<span class="lg c1">bending</span><span class="lg c2">torsion</span><span class="lg c3">stretch</span><span class="lg c4">contact</span><span class="dim">total — [${sc.units.force}·${sc.units.length}]</span>`;
    drawPlot(cv, { series: [mk(H.strain, '--fg', '', 1.8), mk(H.bending, '--chart-1'), mk(H.torsion, '--chart-2'), mk(H.stretch, '--chart-3'), mk(H.contact, '--chart-4')], compact: true, empty: 'no iterations yet', ...w.spec });
  } else if (which === 'parts') {
    lg.innerHTML = `<span class="lg c1">bending</span><span class="lg c2">torsion</span><span class="lg c3">stretch</span><span class="lg c4">contact</span>`;
    drawPlot(cv, { series: [], compact: true, empty: 'no iterations yet' });
  } else {
    const series = []; let spec = {};
    if (H && H.residual.length) {
      const w = chartWindow(H), n0 = w.n0, n = H.residual.length, xs = w.xs;
      let kmax = 0; for (let i = n0; i < n; i++) kmax = Math.max(kmax, H.kinetic[i]);
      series.push({ x: xs, y: H.kinetic.slice(n0).map(v => v / (kmax || 1)), color: cssv('--chart-2'), width: 1.1 });
      series.push({ x: xs, y: H.residual.slice(n0), color: cssv('--chart-1'), width: 1.6 });
      series.push({ x: [xs[0], w.spec.xmax], y: [sc.solver.tol, sc.solver.tol], color: cssv('--ok'), dash: [4, 3], width: 1 });
      spec = w.spec;
    }
    lg.innerHTML = '<span class="lg c1">residual (unbalanced / max force)</span><span class="lg c2">kinetic / max</span><span class="dim">· tol</span>';
    drawPlot(cv, { series, logY: true, compact: true, empty: 'no iterations yet', ...spec });
  }
  const st = m?.st, el = $('strip-text');
  el.textContent = st ? `iteration ${st.iteration} · residual ${Number.isFinite(st.residual) ? st.residual.toExponential(3) : '—'} (tol ${fmt(sc.solver.tol, 3)})${m.converged ? ' · converged' : ''} · strain ${fmt(H.strain.at(-1) ?? 0)} ${sc.units.force}·${sc.units.length}` : 'no model';
  const pl = st && proto ? proto.line() : '';
  if (pl) { const sp = document.createElement('span'); sp.className = 'proto-line'; sp.innerHTML = pl; el.append(document.createElement('br'), sp); }
}
/** the strip chart's x window over the history (last 4000 iterations, widened to keep the latest Zero in view and
 *  to reach the planned end of a pending settle) with the protocol's phase bands */
function chartWindow(H) {
  const n = H.residual.length, last = A.model.st.iteration, first = last - n + 1;
  const pw = proto ? proto.chart(first, last) : { start: Math.max(first, last - 3999), xmax: last, bands: [], vlines: [] };
  const n0 = Math.min(Math.max(0, pw.start - first), Math.max(n - 1, 0));
  const xs = Array.from({ length: n - n0 }, (_, i) => first + n0 + i);
  const spec = { xmin: xs[0], xmax: Math.max(xs[xs.length - 1], pw.xmax), bands: pw.bands, vlines: pw.vlines };
  A.chartWin = { xmin: spec.xmin, xmax: spec.xmax, bands: spec.bands.map(b => `${b.label} ${b.x0}–${b.x1}`), vlines: spec.vlines.map(v => v.x) };   // console
  return { n0, xs, spec };
}
const cssv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// =========================================================================================================== results
async function refreshResults() {
  if (!A.model) return;
  if (A.gpu && !A.running) await syncFromGpu();
  A.results = A.model.results();
  A.dirty.results = false;
  display.applyColouring(); display.refreshDiagram(); updateExtremes(); calib?.onResults?.(); disc?.onResults();
}

function updateExtremes() {
  const r = A.results; if (!r) { $('ro-extremes').textContent = 'no results yet'; return; }
  const u = A.scene.units, rows = [], rng = (a) => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
  const mom = `${u.force}·${u.length}`;
  for (const [k, un] of [['Mx', mom], ['My', mom], ['T', mom], ['N', u.force]]) { const [lo, hi] = rng(r.segment[k]); rows.push(`${k.padEnd(4)} ${fmt(lo).padStart(10)} … ${fmt(hi).padStart(10)}  ${un}`); }
  if (r.knot.Fn && r.knot.Fn.length) {
    const mx = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    rows.push(`Fn   max ${fmt(mx(r.knot.Fn))} ${u.force}   Ft max ${fmt(mx(r.knot.Ft))} ${u.force}`);
    rows.push(`slip max ${fmt(mx(r.knot.slip))} ${u.length}   slipping ${r.knot.state.reduce((a, v) => a + (v > 0), 0)} / ${r.knot.state.length}`);
  }
  let dmax = 0; for (let i = 0; i < r.disp.length / 3; i++) dmax = Math.max(dmax, Math.hypot(r.disp[3 * i], r.disp[3 * i + 1], r.disp[3 * i + 2]));
  rows.push(`|disp| max ${fmt(dmax)} ${u.length}`);
  rows.push(`energy  bending ${fmt(r.energies.bending)}  torsion ${fmt(r.energies.torsion)}  stretch ${fmt(r.energies.stretch)}  contact ${fmt(r.energies.contact)}`);
  if (r.reactions.size) { let s = [0, 0, 0]; for (const v of r.reactions.values()) for (let c = 0; c < 3; c++) s[c] += v[c]; rows.push(`Σ reactions (${s.map(v => fmt(v, 3)).join(', ')}) ${u.force}`); }
  $('ro-extremes').textContent = rows.join('\n');
  $('sum-results').textContent = `${r.iterations} it${r.converged ? ' · converged' : ''}`;
}

/** Python Results.to_dict: per-strand segment arrays, knot values by edge id, node displacements, reactions */
function resultsJSON() {
  const r = A.results, st = A.model.st, topo = A.topo;
  const perStrand = (arr) => { const out = []; for (let s = 0; s < st.S; s++) out.push(Array.from(arr.slice(st.segOffset[s], st.segOffset[s + 1]))); return out; };
  const seg = {}; for (const k of SEG_LAYERS) seg[k] = perStrand(r.segment[k]);
  const knot = {}; for (const [k, v] of Object.entries(r.knot)) { const m = {}; st.knotEdges.forEach((e, c) => { m[String(e)] = v[c]; }); knot[k] = m; }
  const disp = []; for (let s = 0; s < st.S; s++) { const rows = []; for (let i = st.nodeOffset[s]; i < st.nodeOffset[s + 1]; i++) rows.push([r.disp[3 * i], r.disp[3 * i + 1], r.disp[3 * i + 2]]); disp.push(rows); }
  const reactions = {}; for (const [fk, v] of r.reactions) { const [s, k] = fsplit(fk); reactions[`${s},${k}`] = v; }
  return { iterations: r.iterations, converged: r.converged, note: `WeaveSim Web (${A.backend})`, energies: r.energies, segment: seg, knot, node: { disp }, reactions, units: A.scene.units.toJSON(), topology: { strands: topo.nStrands, knots: topo.nKnots } };
}

// =========================================================================================================== overlays
function frameNode(s, k) { return A.model?.st.nodeFrameIds.get(fkey(s, k)); }
function refreshOverlays() {
  if (!A.model) return;
  const sc = A.scene, st = A.model.st;
  const loads = [];
  let R = null;
  for (const pl of sc.loads.point) {
    const nid = frameNode(pl.frame[0], pl.frame[1]); if (nid === undefined) continue;
    let F = pl.F.map(Number);
    if (pl.space === 'local') { if (!R) R = nodeQuats(st); const fr = toFrame(R, 4 * nid); F = [0, 1, 2].map(c => fr[c] * F[0] + fr[3 + c] * F[1] + fr[6 + c] * F[2]); }
    loads.push({ node: nid, F });
  }
  const anchors = [];
  for (const [key, mask] of anchorMasks(sc.anchors, A.topo)) { const nid = st.nodeFrameIds.get(key); if (nid !== undefined) anchors.push({ node: nid, fix: mask }); }
  const selection = A.selection.map(([s, k]) => frameNode(s, k)).filter(v => v !== undefined);
  const probes = [];
  const pr = probeOf(sc);
  if (pr) { const nid = frameNode(pr.frame[0], pr.frame[1]); if (nid !== undefined) probes.push({ node: nid, axis: pr.axis || [0, 0, -1], label: 'probe', color: cssv('--accent') }); }
  for (const p of calib?.probeMarkers?.() || []) { const nid = frameNode(p.frame[0], p.frame[1]); if (nid !== undefined) probes.push({ node: nid, axis: p.axis, label: p.label, color: p.color }); }
  view.setOverlays({ loads, anchors, selection, probes, frames: view.visible.frames, arrowScale: +$('sl-arrow').value });
  updateSelectionReadouts(); renderLists(); updateSummaries();
}

// =========================================================================================================== selection & picking
const selKey = ([s, k]) => fkey(s, k);
function select(frames, add = false) {
  if (!add) A.selection = [];
  for (const f of frames) {
    const i = A.selection.findIndex(g => selKey(g) === selKey(f));
    if (add && i >= 0) A.selection.splice(i, 1); else if (i < 0) A.selection.push([f[0] | 0, f[1] | 0]);
  }
  refreshOverlays();
}
function updateSelectionReadouts() {
  const n = A.selection.length, last = A.selection[n - 1];
  const txt = n ? `selection: ${n} frame(s) · last (${last[0]}, ${last[1]}) ${A.topo.isOver(last[0], last[1]) ? 'over' : 'under'}` : 'select in the view: click a lath next to a crossing (its own frame), shift-click adds / removes, Esc clears';
  qsa('.sel-readout').forEach(el => { el.textContent = txt; });
  $('btn-addload').textContent = `Add to selection (${n})`; $('btn-addanchor').textContent = `Add for selection (${n})`;
  updateAssignReadout();
}
function lathsToAssign() { return [...A.lathSel].sort((a, b) => a - b); }
function updateAssignReadout() {
  if (!A.scene?.hybrid.enabled) return;
  const L = lathsToAssign(), g = +$('sel-group').value || 1;
  $('ro-assign').textContent = L.length ? `${L.length} lath${L.length === 1 ? '' : 's'}: ${L.slice(0, 14).join(', ')}${L.length > 14 ? ' …' : ''}` : 'no laths selected';
  $('btn-assign').textContent = `Assign ${L.length || ''} to group ${g}`.replace('  ', ' ');
  $('btn-assign').disabled = !L.length;
}
/** the laths selected for a group assignment (LocalSH: the others dimmed, a tube along each selected lath) */
function setLathSelection(strands) { A.lathSel = new Set(strands); view.setStrandSelection(A.lathSel); updateAssignReadout(); }

function frameAtPick(p) {
  if (!p) return null;
  if (p.kind === 'lath') return view.nearestFrame(p.strand, p.point);
  if (p.kind === 'knot') {
    const kn = A.topo.knots.get(A.model.st.knotEdges[p.c]);
    const overF = kn.a[0] === kn.over ? kn.a : kn.b, underF = overF === kn.a ? kn.b : kn.a;
    return $('sel-knotpick').value === 'under' ? underF : overF;
  }
  return null;
}

function onPick(ev) {
  const p = view.pick(ev.clientX, ev.clientY);
  if (A.mode === 'laths') {
    let s = null;
    if (p && p.kind === 'lath') s = p.strand;
    else if (p && p.kind === 'knot') { const f = frameAtPick(p); if (f) s = f[0]; }
    if (s === null) { if (!ev.shiftKey && A.lathSel.size) { setLathSelection([]); status('lath selection cleared'); } return; }
    const next = new Set(A.lathSel); if (next.has(s)) next.delete(s); else next.add(s);
    setLathSelection(next);
    status(`lath ${s} ${next.has(s) ? 'selected' : 'removed'} · ${next.size} selected · group ${A.scene.strandGroup(s)}`);
    return;
  }
  const f = frameAtPick(p);
  if (A.mode === 'flip') {
    if (!f) return;
    const flips = A.scene.weave.flip_knots.map(x => [x[0] | 0, x[1] | 0]);
    const kn = A.topo.knotOf(f[0], f[1]), key = kn.a, alt = kn.b;
    const i = flips.findIndex(x => x[0] === key[0] && x[1] === key[1]), j = flips.findIndex(x => x[0] === alt[0] && x[1] === alt[1]);
    if (i >= 0) flips.splice(i, 1); else if (j >= 0) flips.splice(j, 1); else flips.push([key[0], key[1]]);
    A.scene.weave.flip_knots = flips;
    rebuildTopology({ keepCamera: true }); updateWeaveReadout();
    status(`knot ${kn.edge} flipped — ${flips.length} flip(s)`);
    return;
  }
  if (A.mode === 'probe') {
    if (!f) return;
    const axis = [+$('nb-ax').value, +$('nb-ay').value, +$('nb-az').value];
    A.scene.utm.probe = { frame: [f[0], f[1]], axis };
    setMode('none'); refreshOverlays(); updateUtmReadout(); A.probeTrace = [];
    return;
  }
  if (calib?.pickMode?.()) { if (f) calib.onPick(f); return; }
  if (!f) { if (!ev.shiftKey) select([]); return; }
  select([f], ev.shiftKey);
}

let colourBeforeLaths = null;
function setMode(m) {
  const was = A.mode;
  A.mode = A.mode === m ? 'none' : m;
  $('btn-flipmode').classList.toggle('on', A.mode === 'flip');
  $('btn-probemode').classList.toggle('on', A.mode === 'probe');
  $('btn-lathmode').classList.toggle('on', A.mode === 'laths');
  // lath selection: colour by lath group while picking (back to the previous colouring afterwards)
  if (A.mode === 'laths' && was !== 'laths') { const c = $('sel-colour').value; if (!LAYER_LABEL[c] && c !== 'group') { colourBeforeLaths = c; $('sel-colour').value = 'group'; display.applyColouring(); } }
  if (was === 'laths' && A.mode !== 'laths') {
    if (A.lathSel.size) setLathSelection([]);                       // what is highlighted is what Assign takes: nothing outside the mode
    if (colourBeforeLaths) { $('sel-colour').value = colourBeforeLaths; colourBeforeLaths = null; display.applyColouring(); }
  }
  const banner = $('mode-banner');
  banner.hidden = A.mode === 'none';
  banner.textContent = A.mode === 'flip' ? 'FLIP — click a lath at a crossing to swap over / under (Esc leaves)' : A.mode === 'probe' ? 'PROBE — click the lath at the probe crossing (Esc leaves)'
    : A.mode === 'laths' ? 'LATHS — click laths to add / remove them, then Assign (3 Units & material) · Esc leaves' : '';
  $('gl').classList.toggle('arm', A.mode !== 'none');
}
export function setPickBanner(text) { const b = $('mode-banner'); b.hidden = !text; b.textContent = text || ''; $('gl').classList.toggle('arm', !!text); }

let hoverPending = false, hoverEv = null;
function onHover(ev) { hoverEv = ev; if (hoverPending) return; hoverPending = true; requestAnimationFrame(() => { hoverPending = false; showTooltip(hoverEv); }); }
/** hover: a small window with the local values (the Joint FEA probe) — the lath under the cursor, the position along it,
 *  the active colour layer and diagram at that point; at a crossing also its contact state */
function showTooltip(ev) {
  const tip = $('tooltip');
  if (!A.model || !ev || ev.buttons) { tip.hidden = true; return; }
  const p = view.pick(ev.clientX, ev.clientY);
  if (!p || p.kind === 'mesh') { tip.hidden = true; return; }
  const html = [];
  if (p.kind === 'lath') { const pr = display.probe(p); html.push(pr.title, ...pr.lines); }
  let c = -1;
  if (p.kind === 'knot') c = p.c;
  else { const f = view.nearestFrame(p.strand, p.point); if (f) { const nid = frameNode(f[0], f[1]); const P = view.dispX; if (Math.hypot(P[3 * nid] - p.point.x, P[3 * nid + 1] - p.point.y, P[3 * nid + 2] - p.point.z) < 1.5 * maxWidth()) c = A.model.st.knotEdges.indexOf(A.topo.frameToKnot.get(fkey(f[0], f[1]))); } }
  if (c >= 0) {
    const e = A.model.st.knotEdges[c], kn = A.topo.knots.get(e), u = A.scene.units;
    const overF = kn.a[0] === kn.over ? kn.a : kn.b, underF = overF === kn.a ? kn.b : kn.a;
    const flags = `${kn.isBoundary ? ' · boundary' : ''}${kn.flipped ? ' · flipped' : ''}${kn.conflict ? ' · <span class="warn">parity conflict</span>' : ''}`;
    if (html.length) html.push('<hr>');
    html.push(`<b>crossing ${e}</b> · over ${overF[0]} · under ${underF[0]}${flags}`);
    const r = A.results;
    if (r && r.knot.Fn && r.knot.Fn.length) {
      html.push(`Fn <b>${fmt(r.knot.Fn[c], 3)}</b> ${u.force} · Ft <b>${fmt(r.knot.Ft[c], 3)}</b> ${u.force} · ${r.knot.state[c] ? '<span class="warn">slip</span>' : 'stick'}`);
      html.push(`gap ${fmt(r.knot.gap[c], 3)} ${u.length} · slip ${fmt(r.knot.slip[c], 3)} ${u.length}`);
      if (r.knot.ctype) html.push(`${['FF', 'FE', 'EE'][r.knot.ctype[c] | 0]} · θ ${fmt(r.knot.theta[c] * 180 / Math.PI, 3)}° · p ${fmt(r.knot.pressure[c], 3)} ${u.stressLabel}`);
    }
  }
  if (!html.length) { tip.hidden = true; return; }
  tip.innerHTML = html.map(h => (h === '<hr>' ? '<div class="sep"></div>' : `<div>${h}</div>`)).join(''); tip.hidden = false;
  const rect = $('viewport').getBoundingClientRect();
  tip.style.left = Math.max(4, Math.min(ev.clientX - rect.left + 14, rect.width - tip.offsetWidth - 6)) + 'px';
  tip.style.top = Math.max(4, Math.min(ev.clientY - rect.top - 40, rect.height - tip.offsetHeight - 6)) + 'px';
}
function maxWidth() { let w = 0; for (let s = 0; s < A.model.st.S; s++) w = Math.max(w, A.scene.group(A.scene.strandGroup(s)).section.width); return w; }

// =========================================================================================================== panels: binding
/** data-bind="group.key" inputs act on the edited lath group's material / section / beam, or the scene's contact / solver */
function bindTarget(path) {
  const [grp, key] = path.split('.');
  const sc = A.scene;
  if (['material', 'section', 'beam'].includes(grp)) { const g = sc.group(+$('sel-group').value || 1); return [g[grp], key]; }
  return [sc[grp], key];
}
function readBound(el) { if (!A.scene) return; const [o, k] = bindTarget(el.dataset.bind); const v = o[k]; if (el.type === 'checkbox') el.checked = !!v; else if (el.tagName === 'SELECT') el.value = v; else el.value = typeof v === 'number' ? +v.toPrecision(6) : v ?? ''; const out = el.parentElement.querySelector('output'); if (out) out.textContent = (+v).toFixed(2); }
function writeBound(el) {
  if (!A.scene) return;
  const [o, k] = bindTarget(el.dataset.bind);
  let v;
  if (el.type === 'checkbox') v = el.checked;
  else if (el.tagName === 'SELECT') v = el.value;
  else { v = +el.value; if (!Number.isFinite(v)) return; if ('int' in el.dataset) v = Math.round(v); }
  o[k] = v;
  const out = el.parentElement.querySelector('output'); if (out) out.textContent = (+v).toFixed(2);
  const path = el.dataset.bind;
  if (path === 'section.height') { rebuildTopology({ keepCamera: true }); }
  else if (path === 'section.width') { resetModel({ keepCamera: true }); }
  onSceneEdited(path);
}
function onSceneEdited() { updateSectionReadout(); updateContactReadout(); updateSummaries(); drawChart(); calib?.onSceneEdit?.(); }

function syncPanels() {
  const sc = A.scene; if (!sc) return;
  // weave
  $('nb-subdiv').value = sc.weave.subdivisions; $('nb-overhang').value = sc.weave.overhang; $('sel-offset').value = sc.weave.offset_mode; $('nb-phase').value = sc.weave.seed_strand_phase;
  // units
  $('sel-len').value = sc.units.length; $('sel-force').value = sc.units.force;
  qsa('.u-len').forEach(e => { e.textContent = `[${sc.units.length}]`; }); qsa('.u-force').forEach(e => { e.textContent = `[${sc.units.force}]`; });
  qsa('.u-stress').forEach(e => { e.textContent = `[${sc.units.stressLabel}]`; }); qsa('.u-kpen').forEach(e => { e.textContent = `[${sc.units.force}/${sc.units.length}]`; });
  // groups
  $('ck-hybrid').checked = sc.hybrid.enabled; $('grp-ctl').hidden = !sc.hybrid.enabled; $('row-assign').hidden = !sc.hybrid.enabled;
  const gs = $('sel-group'), cur = +gs.value || 1;
  gs.innerHTML = Array.from({ length: sc.nGroups }, (_, i) => `<option value="${i + 1}">${i + 1}: ${sc.group(i + 1).name || 'group ' + (i + 1)}</option>`).join('');
  gs.value = Math.min(cur, sc.nGroups); $('nb-ngroups').value = sc.nGroups;
  $('row-grpname').hidden = !(sc.hybrid.enabled && +gs.value >= 2); if (+gs.value >= 2) $('tx-grpname').value = sc.group(+gs.value).name || '';
  qsa('[data-bind]').forEach(readBound);
  const g = sc.group(+gs.value || 1);
  $('sel-beammode').value = g.beam.mode; $('beam-simple').hidden = g.beam.mode !== 'simple'; $('beam-adv').hidden = g.beam.mode === 'simple';
  const ov = g.section.Ix != null || g.section.Iy != null || g.section.J != null;
  $('ck-override').checked = ov; $('grid-override').hidden = !ov;
  $('nb-Ix').value = +IxEff(g.section).toPrecision(6); $('nb-Iy').value = +IyEff(g.section).toPrecision(6); $('nb-J').value = +JEff(g.section).toPrecision(6);
  // loads
  const L = sc.loads;
  $('ck-grav').checked = L.gravity.enabled; ['gx', 'gy', 'gz'].forEach((k, i) => { $('nb-' + k).value = L.gravity.direction[i]; });
  $('ck-udl').checked = L.udl.enabled; ['qx', 'qy', 'qz'].forEach((k, i) => { $('nb-' + k).value = L.udl.q[i]; }); $('sel-udlper').value = L.udl.per;
  // utm
  if (probeOf(sc)) { const a = sc.utm.probe.axis || [0, 0, -1]; $('nb-ax').value = a[0]; $('nb-ay').value = a[1]; $('nb-az').value = a[2]; }
  $('sel-utmfu').value = sc.utm.units?.force || 'N'; $('sel-utmlu').value = sc.utm.units?.length || 'mm';
  updateSectionReadout(); updateContactReadout(); updateWeaveReadout(); updateUtmReadout(); renderLists(); updateSummaries(); buildReplayGrid(); updateAssignReadout();
  disc?.sync(); qsa('.u-len').forEach(e => { e.textContent = `[${sc.units.length}]`; });
  $('ro-file').textContent = `${A.sceneName} · mesh ${A.meshName} (scale ${sc.mesh.scale}) · ${A.mesh.nV} vertices, ${A.mesh.nF} faces`;
}

function updateSectionReadout() {
  const sc = A.scene; if (!sc) return;
  const g = sc.group(+$('sel-group').value || 1), s = g.section, m = g.material, u = sc.units;
  const k = sectionStiffness(s, m), f = beamFactors(g.beam), a = beamActivations(g.beam);
  $('ro-section').textContent = `Ix ${fmt(IxEff(s))}  Iy ${fmt(IyEff(s))}  J ${fmt(JEff(s))} [${u.length}⁴]\nEA ${fmt(k.EA)}  EIx ${fmt(k.EIx * f.EIx)}  EIy ${fmt(k.EIy * f.EIy)}  GJ ${fmt(k.GJ * f.GJ)}\nself-weight ${fmt(u.selfWeightLineLoad(m.density, sectionArea(s)))} ${u.force}/${u.length} · active bending ${a.map(v => Math.round(100 * v)).join(' / ')} %`;
}
function updateContactReadout() {
  const sc = A.scene, t = A.topo; if (!sc || !t) return;
  const k = sectionStiffness(sc.section, sc.material);
  let L = 0, n = 0; for (const P of t.polylines) for (let i = 3; i < P.length; i += 3) { L += Math.hypot(P[i] - P[i - 3], P[i + 1] - P[i - 2], P[i + 2] - P[i - 1]); n++; }
  const kb = 4 * k.EIx / Math.max(L / Math.max(n, 1), 1e-9) ** 3;
  $('ro-contact').textContent = `lath transverse stiffness 4EIx/l̄³ ≈ ${fmt(kb, 3)} ${sc.units.force}/${sc.units.length} (strength 20–50× that)`;
}
function updateWeaveReadout() { const t = A.topo; if (!t) return; $('ro-weave').textContent = `${t.nStrands} strands · ${t.nKnots} knots · ${t.familyCount} families\nparity conflicts ${t.parityConflicts} · flips ${A.scene.weave.flip_knots.length}`; }
function updateSummaries() {
  const sc = A.scene; if (!sc) return;
  $('sum-file').textContent = A.sceneName;
  $('sum-weave').textContent = A.topo ? `${A.topo.nStrands} strands · ${A.topo.nKnots} knots` : '';
  $('sum-material').textContent = `${sc.units} · E ${fmt(sc.material.E, 3)} · ${fmt(sc.section.width, 4)}×${fmt(sc.section.height, 4)}`;
  $('sum-contact').textContent = `μ ${fmt(sc.contact.mu, 3)} · k ${fmt(sc.contact.strength, 3)}${sc.contact.locked ? ' · locked' : ''}`;
  $('sum-loads').textContent = `${sc.loads.gravity.enabled ? 'gravity · ' : ''}${sc.loads.point.length} point`;
  $('sum-anchors').textContent = `${sc.anchors.length} anchor${sc.anchors.length === 1 ? '' : 's'}`;
  $('sum-solve').textContent = A.backend === 'webgpu' ? 'WebGPU' : 'CPU f64';
  $('sum-utm').textContent = probeOf(sc) ? `probe (${sc.utm.probe.frame})` : '';
}

function renderLists() {
  const sc = A.scene; if (!sc) return;
  const u = sc.units;
  $('list-loads').innerHTML = sc.loads.point.map((pl, i) => `<div class="item"><span class="grow">${i}: (${pl.frame}) F=[${pl.F.map(v => fmt(v, 3)).join(', ')}] ${pl.space} |F|=${fmt(Math.hypot(...pl.F), 3)} ${u.force}</span><button class="lnk" data-sel-load="${i}">select</button><button class="x" data-del-load="${i}" title="remove">×</button></div>`).join('') || '<div class="hint">no point loads</div>';
  $('list-anchors').innerHTML = sc.anchors.map((a, i) => `<div class="item"><span class="grow">${i}: ${selectorLabel(a.select).padEnd(16)} ${fixLabel(a.fix)}</span><button class="lnk" data-sel-anchor="${i}">select</button><button class="x" data-del-anchor="${i}" title="remove">×</button></div>`).join('') || '<div class="hint">no anchors — the model is free</div>';
}

// =========================================================================================================== UTM
function updateUtmReadout() {
  const sc = A.scene; if (!sc) return;
  const pr = probeOf(sc);
  $('ro-probe').textContent = pr ? `probe frame (${pr.frame[0]}, ${pr.frame[1]})` : 'no probe';
  const mx = (a) => { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; };
  $('ro-utmfile').textContent = A.utm ? `${A.utmName}: ${A.utm.length} rows · max ${fmt(mx(A.utm.force), 3)} ${sc.units.force} @ ${fmt(mx(A.utm.displacement), 3)} ${sc.units.length}` : 'no curve';
}
function loadUtmText(text, name) {
  A.utmText = text; A.utmName = name;
  const sc = A.scene;
  try {
    const head = text.split(/\r?\n/).slice(0, 60).join('\n');
    // column choices from the header row
    const delim = [',', ';', '\t'].reduce((b, d) => ((head.split(d).length > head.split(b).length) ? d : b), ',');
    const rows = head.split(/\r?\n/).map(l => l.split(delim));
    const hi = rows.findIndex((r, i) => i + 1 < rows.length && r.filter(c => c.trim() && !Number.isFinite(+c.replace(',', '.'))).length >= 2);
    const headers = hi >= 0 ? rows[hi].map(c => c.trim()) : [];
    for (const id of ['sel-utmforce', 'sel-utmdisp']) $(id).innerHTML = '<option value="">auto</option>' + headers.map(h => `<option>${h.replace(/</g, '&lt;')}</option>`).join('');
    parseUtm();
  } catch (e) { status('UTM csv: ' + e.message, 'warn'); }
}
function parseUtm() {
  if (!A.utmText || !A.scene) return;
  const sc = A.scene;
  const fileUnits = { force: $('sel-utmfu').value, length: $('sel-utmlu').value };
  try {
    // units in the header win over the selectors (as read_utm_csv); the selectors are the fallback
    A.utm = parseUTMCsv(A.utmText, sc.units, { forceCol: $('sel-utmforce').value || null, dispCol: $('sel-utmdisp').value || null, source: A.utmName, fileUnits: null });
    if (!/[([]/.test(A.utm.columns.force || '')) A.utm = parseUTMCsv(A.utmText, sc.units, { forceCol: $('sel-utmforce').value || null, dispCol: $('sel-utmdisp').value || null, source: A.utmName, fileUnits });
    sc.utm.units = fileUnits;
    status(`UTM: ${A.utm.length} rows (${A.utm.columns.force} / ${A.utm.columns.displacement})`);
  } catch (e) { A.utm = null; status('UTM csv: ' + e.message, 'warn'); }
  updateUtmReadout(); drawUtmPlot(); drawChart();
}

const REPLAY_FIELDS = [['mode', 'mode', 'sel', ['force', 'displacement']], ['relax_iter', 'relax iterations', 'int'], ['step_iter', 'iterations / increment', 'int'], ['avg_iter', 'averaged over', 'int'], ['n_steps', 'increments / branch', 'int'], ['cycles', 'cycles (force)', 'int'], ['max_force', 'peak force (blank = curve)', 'num'], ['lock', 'lock crossings', 'bool']];
let replaySet = replaySettings({ relax_iter: 2500, step_iter: 500, avg_iter: 150, n_steps: 4, cycles: 2 });
function buildReplayGrid() {
  const g = $('grid-replay');
  g.innerHTML = REPLAY_FIELDS.map(([k, label, t, opts]) => {
    const v = replaySet[k];
    if (t === 'sel') return `<label>${label}</label><select data-rp="${k}">${opts.map(o => `<option${o === v ? ' selected' : ''}>${o}</option>`).join('')}</select>`;
    if (t === 'bool') return `<label>${label}</label><input type="checkbox" data-rp="${k}"${v ? ' checked' : ''}>`;
    return `<label>${label}</label><input type="number" class="num" data-rp="${k}" value="${v ?? ''}" step="${t === 'int' ? 1 : 'any'}">`;
  }).join('');
  qsa('[data-rp]', g).forEach(el => el.addEventListener('change', () => {
    const k = el.dataset.rp;
    replaySet[k] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : (el.value === '' ? null : +el.value);
  }));
}
function replayBranches(r) { return Array.isArray(r) ? r : [{ d: r.d, f: r.f }]; }
let replayAbort = null;
async function runReplay() {
  const sc = A.scene, pr = probeOf(sc);
  if (!pr) { status('pick a probe frame first', 'warn'); return; }
  if (replaySet.mode === 'force' && !replaySet.max_force && !A.utm) { status('force mode: give a peak force or load a UTM curve', 'warn'); return; }
  stopRun(); await waitIdle();
  const axis = [+$('nb-ax').value, +$('nb-ay').value, +$('nb-az').value]; pr.axis = axis;
  const t = new Test(new Probe(pr.frame, axis, 'probe'), A.utm, 1.0, A.utmName);
  const cs = CalibrationSet.fromModel(sc, A.topo, { tests: [t], replay: replaySet });
  if (A.backend === 'webgpu' && A.device) cs.setBackend('gpu', A.device);
  replayAbort = new AbortController();
  $('btn-replay').disabled = true; $('btn-replay-stop').disabled = false;
  const t0 = performance.now();
  try {
    const sims = await cs.simulate({}, [t], { signal: replayAbort.signal, onProgress: (p) => { $('ro-utm').textContent = `${p.stage} ${(100 * p.frac).toFixed(0)} %`; } });
    A.replay = sims[t.name];
    let txt = `replay ${((performance.now() - t0) / 1000).toFixed(1)} s on ${cs.backend === 'gpu' ? 'WebGPU' : 'CPU f64'}`;
    if (A.utm) { const res = cs.residuals({ [t.name]: A.replay }, [t]); txt += ` · residual ${(100 * res[t.name]).toFixed(2)} % of the measured ${replaySet.mode === 'force' ? 'travel range' : 'peak force'}`; }
    $('ro-utm').textContent = txt;
  } catch (e) { $('ro-utm').textContent = e.name === 'AbortError' ? 'stopped' : 'replay failed: ' + e.message; if (e.name !== 'AbortError') console.error(e); }
  finally { $('btn-replay').disabled = false; $('btn-replay-stop').disabled = true; replayAbort = null; }
  drawUtmPlot(); drawChart();
}
function drawUtmPlot() {
  const sc = A.scene; if (!sc) return;
  const series = [];
  if (A.utm) series.push({ x: A.utm.displacement, y: A.utm.force, color: cssv('--fg'), dots: true, r: 1.8, label: 'UTM' });
  if (A.replay) replayBranches(A.replay).forEach((b, i) => series.push({ x: b.d, y: b.f, color: cssv('--chart-1'), width: 1.6, dots: true, line: true, r: 2, label: i ? '' : 'model', alpha: b.cycle === 0 ? 0.55 : 1 }));
  drawPlot($('utm-plot'), { series, xlabel: `crosshead travel [${sc.units.length}]`, ylabel: `force [${sc.units.force}]`, legend: 'top', empty: 'load a UTM csv and / or replay' });
}

// =========================================================================================================== files
async function readFileList(files) {
  const list = [...files];
  const by = (re) => list.filter(f => re.test(f.name));
  const calibFile = list.find(f => /calib.*\.json$/i.test(f.name));
  if (calibFile) { await calib.loadFromFiles(list); $('sec-calib').open = true; return; }
  const scenes = by(/\.json$/i), meshes = by(/\.(stl|obj)$/i), csvs = by(/\.(csv|txt)$/i);
  if (scenes.length) {
    const f = scenes[0], d = JSON.parse(await f.text());
    if (d.kind === 'weavesim-parameter-set') { calib.applyParameterSetJSON(d); return; }
    const sc = Scene.fromJSON(d, f.name.replace(/\.json$/i, ''));
    const base = sc.mesh.path.split(/[\\/]/).pop();
    const m = meshes.find(x => x.name === base) || meshes[0];
    if (!m) {
      // try next to the page's examples (a scene saved from an example keeps its relative mesh path)
      try { const r = await fetch(new URL(sc.mesh.path, location.href)); if (!r.ok) throw new Error(); setScene(sc, new Uint8Array(await r.arrayBuffer()), base); return; } catch (e) { /* ask */ }
      status(`scene ${f.name} needs its mesh ${base}: select or drop both files`, 'warn'); return;
    }
    setScene(sc, new Uint8Array(await m.arrayBuffer()), m.name);
    status(`loaded ${f.name} + ${m.name}`);
  } else if (meshes.length) {
    const m = meshes[0], sc = A.scene ? Scene.fromJSON(A.scene.toJSON(), m.name.replace(/\.\w+$/, '')) : new Scene();
    sc.mesh = { path: m.name, scale: 1.0 }; sc.weave.flip_knots = []; sc.anchors = A.scene ? sc.anchors : [];
    if (!A.scene) { sc.addAnchor({ type: 'all_start' }, [1, 1, 1, 0, 0, 0], 'starts'); sc.addAnchor({ type: 'all_end' }, [1, 1, 1, 0, 0, 0], 'ends'); }
    sc.name = m.name.replace(/\.\w+$/, '');
    setScene(sc, new Uint8Array(await m.arrayBuffer()), m.name);
    status(`new scene on ${m.name} (settings kept from the previous scene; anchors at the strand ends)`);
  }
  if (csvs.length && A.scene) { loadUtmText(await csvs[0].text(), csvs[0].name); $('sec-utm').open = true; }
}

async function loadExample(path) {
  try {
    status('loading ' + path + ' …');
    const url = new URL(path, location.href), d = await (await fetch(url)).json();
    const name = path.split('/').pop().replace(/\.json$/, '');
    const sc = Scene.fromJSON(d, name);
    const mr = await fetch(new URL(sc.mesh.path, url));
    if (!mr.ok) throw new Error(`mesh ${sc.mesh.path}: HTTP ${mr.status}`);
    A.sceneURL = url.href;
    setScene(sc, new Uint8Array(await mr.arrayBuffer()), sc.mesh.path.split('/').pop());
    status(`example ${name}`);
    store.set('example', path);
  } catch (e) {
    const net = /fetch|HTTP|NetworkError|JSON/i.test(e.message);
    if (!net) console.error(e);
    status(net ? `example not reachable (${e.message}) — serve the WeaveSim folder with web/serve.py, or open files` : `could not open the example: ${e.message}`, 'warn');
  }
}

function download(name, text, type = 'application/json') {
  const a = document.createElement('a'), url = URL.createObjectURL(new Blob([text], { type }));
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
}
export function showModal(title, hint, text, fileName) {
  $('modal-title').textContent = title; $('modal-hint').textContent = hint || ''; $('modal-text').value = text;
  const a = $('modal-download'); a.download = fileName || 'export.json'; a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  $('modal').hidden = false;
}

// =========================================================================================================== Rhino
function downloadBytes(name, bytes, type = 'application/octet-stream') {
  const a = document.createElement('a'), url = URL.createObjectURL(new Blob([bytes], { type }));
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000);
}
/** File › Export Rhino .3dm (export3dm.py): the model, and with a solved state the results, diagrams, forces */
async function export3dm() {
  if (!A.topo) { status('nothing to export: load a mesh first', 'warn'); return; }
  const ro = $('ro-3dm'), btn = $('btn-3dm');
  btn.disabled = true; ro.className = 'readout inline'; ro.textContent = 'loading rhino3dm …';
  try {
    const wasRunning = A.running; stopRun(); await waitIdle(); await syncFromGpu();
    const solved = !!A.model && A.model.st.iteration > 0;
    if (solved) A.results = A.model.results();
    const opts = {}; qsa('[data-x3dm]').forEach(c => { opts[c.dataset.x3dm] = c.checked; });
    const t0 = performance.now();
    const out = await exportRhino({ scene: A.scene, topo: A.topo, mesh: A.mesh, st: solved ? A.model.st : null, results: solved ? A.results : null, contact: solved ? A.model.contact : null, name: A.sceneName, sceneName: `${A.sceneName}.json` }, opts);
    downloadBytes(`${A.sceneName}.3dm`, out.bytes);
    setTimeout(() => download(`${A.sceneName}.3dm.json`, JSON.stringify(out.index, null, 1)), 400);
    ro.className = 'readout inline ok';
    ro.textContent = `${A.sceneName}.3dm · ${out.objects} objects · ${(out.bytes.length / 1e6).toFixed(1)} MB · ${((performance.now() - t0) / 1000).toFixed(1)} s (+ .3dm.json)` + (solved ? '' : ' — model only, no solver run yet');
    status(`Rhino → ${A.sceneName}.3dm`);
    if (wasRunning) startRun();
  } catch (e) { console.error(e); ro.className = 'readout inline warn'; ro.textContent = 'export failed: ' + (e.message || e); }
  finally { btn.disabled = false; }
}

// =========================================================================================================== init
let calib = null, display = null, disc = null, proto = null;
async function init() {
  view = new View($('gl'), $('labels')); A.view = view;
  display = initDisplay({ A, $, qsa, fmt, store, status, refreshResults: () => refreshResults() });
  disc = initDiscretize({ A, $, qsa, status, cssv, store, download, onJoints: () => { display.onJoints(); updateSummaries(); }, drawSolveChart: () => drawChart() });
  proto = initProtocol({ A, $, status, fmt, cssv, store, stopRun, startRun, exclusive, syncFromGpu, goalSignature, paramSignature,
    readLockBox: () => readBound($('ck-locked')), drawChart: () => drawChart(), updateSummaries: () => updateSummaries(),
    refreshViews: () => { if (A.model) view.updateGeometry(A.model.st); display.applyColouring(); display.refreshDiagram(); } });
  view.attachNav($('navcube'));
  const theme = store.get('theme', 'auto'); setTheme(theme);
  bindNav(); bindPanels();
  A.device = await getDevice(); A.gpuAvailable = !!A.device;
  A.backend = A.gpuAvailable && store.get('backend', 'webgpu') === 'webgpu' ? 'webgpu' : 'cpu';
  qsa('#backend-chips .chip').forEach(c => { c.classList.toggle('on', c.dataset.backend === A.backend); if (c.dataset.backend === 'webgpu' && !A.gpuAvailable) c.disabled = true; });
  updateBackendReadout(); updateStatus();
  calib = initCalibration({ A, $, qsa, status, fmt, refreshOverlays, applySceneValues: () => { syncPanels(); onSceneEdited(); }, download, showModal, setPickBanner, cssv, stopRun, waitIdle,
    openScene: (sc, bytes, name) => { setScene(sc, bytes, name); $('sel-example').value = ''; } });
  if (!A.gpuAvailable) status(window.isSecureContext ? 'no WebGPU adapter in this browser — CPU f64 solver' : 'not a secure context (http on a LAN address): WebGPU is off — use localhost or serve.py --https', 'warn');
  const sel = $('sel-example');
  sel.innerHTML = '<option value="">—</option>' + EXAMPLES.map(e => `<option value="${e.path}">${e.label}</option>`).join('');
  const q = new URLSearchParams(location.search);
  const first = q.get('scene') || store.get('example', EXAMPLES[0].path);
  sel.value = first; await loadExample(first);
  requestAnimationFrame(frame);
}

let lastResultRefresh = 0;
function frame() {
  if (A.running && !A.busy) solveChunk();
  if (A.dirty.results && !A.running && !A.busy) refreshResults();
  else if (A.dirty.results && A.running && (LAYER_LABEL[$('sel-colour').value] || display.diagramActive()) && performance.now() - lastResultRefresh > 600) {
    lastResultRefresh = performance.now(); A.results = A.model.results(); A.dirty.results = false; display.applyColouring(); display.refreshDiagram();
  }
  disc.tick();
  view.render();
  requestAnimationFrame(frame);
}

function setTheme(t) {
  if (t === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t);
  qsa('#theme-chips .chip').forEach(c => c.classList.toggle('on', c.dataset.theme === t));
  store.set('theme', t);
  view.applyTheme(); drawChart(); drawUtmPlot(); calib?.redraw?.(); disc?.redraw(); display?.applyColouring();
}

function bindNav() {
  $('nav-home').onclick = () => view.home(); $('nav-fit').onclick = () => view.fit();
  $('nav-turn').onclick = () => { view.setTurntable(!view.turntable, +$('sl-turn').value); $('nav-turn').classList.toggle('on', view.turntable); };
  $('nav-proj').onclick = () => { view.setProjection(view.projection === 'persp' ? 'ortho' : 'persp'); $('nav-proj').querySelector('.ic-persp').hidden = view.projection !== 'persp'; $('nav-proj').querySelector('.ic-ortho').hidden = view.projection === 'persp'; };
  const q = Math.PI / 2;
  $('nav-left').onclick = () => view.orbitBy(q, 0); $('nav-right').onclick = () => view.orbitBy(-q, 0); $('nav-up').onclick = () => view.orbitBy(0, -q); $('nav-down').onclick = () => view.orbitBy(0, q);
  const gl = $('gl');
  let down = null;
  gl.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
  gl.addEventListener('pointerup', (e) => { if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5 && e.button === 0) onPick(e); down = null; });
  gl.addEventListener('pointermove', onHover);
  gl.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'Escape') { if (A.mode !== 'none') setMode(A.mode); else if (calib?.pickMode?.()) calib.cancelPick(); else if (A.lathSel.size) setLathSelection([]); else select([]); }
    else if (e.key === 'h' || e.key === 'H') view.home();
    else if (e.key === 'f' || e.key === 'F') view.fit();
    else if (e.key === 't' || e.key === 'T') $('nav-turn').click();
    else if (e.key === 'p' || e.key === 'P') $('nav-proj').click();
    else if (e.key === ' ') { e.preventDefault(); $('btn-run').click(); }
  });
  // drag & drop
  const vp = $('viewport'), drop = $('drop');
  vp.addEventListener('dragover', (e) => { e.preventDefault(); drop.hidden = false; });
  vp.addEventListener('dragleave', () => { drop.hidden = true; });
  vp.addEventListener('drop', (e) => { e.preventDefault(); drop.hidden = true; readFileList(e.dataTransfer.files).catch(err => status(err.message, 'warn')); });
}

function bindPanels() {
  // strip
  $('btn-run').onclick = () => { if (A.running) stopRun(); else startRun(); };
  $('btn-step').onclick = () => stepN(Math.max(1, +$('nb-step').value | 0));
  $('btn-reset').onclick = () => resetModel({ keepCamera: true });
  $('sel-chart').onchange = drawChart;
  $('st-backend').onclick = () => setBackend(A.backend === 'webgpu' ? 'cpu' : 'webgpu');
  // file
  $('sel-example').onchange = (e) => { if (e.target.value) loadExample(e.target.value); };
  $('file-scene').onchange = (e) => { readFileList(e.target.files).catch(err => status(err.message, 'warn')); e.target.value = ''; };
  $('btn-save').onclick = () => { if (A.scene) download(`${A.sceneName}.json`, JSON.stringify(A.scene.toJSON(), null, 1)); };
  $('btn-export-results').onclick = async () => { if (!A.model) return; await refreshResults(); download(`${A.sceneName}_results.json`, JSON.stringify(resultsJSON())); };
  $('btn-shot').onclick = () => { const a = document.createElement('a'); a.href = view.screenshotDataURL(); a.download = `${A.sceneName}.png`; a.click(); };
  $('btn-3dm').onclick = export3dm;
  // weave
  const weaveEdit = () => { const w = A.scene.weave; w.subdivisions = Math.max(1, +$('nb-subdiv').value | 0); w.overhang = Math.max(0, +$('nb-overhang').value); w.offset_mode = $('sel-offset').value; w.seed_strand_phase = (+$('nb-phase').value | 0) % 2; };
  ['nb-subdiv', 'nb-overhang', 'sel-offset', 'nb-phase'].forEach(id => $(id).addEventListener('change', weaveEdit));
  $('btn-regen').onclick = () => { weaveEdit(); rebuildTopology({ keepCamera: true }); updateWeaveReadout(); updateSummaries(); };
  $('btn-flipmode').onclick = () => setMode('flip');
  $('btn-clearflips').onclick = () => { A.scene.weave.flip_knots = []; rebuildTopology({ keepCamera: true }); updateWeaveReadout(); };
  // units & groups
  const unitEdit = () => {
    const nu = new UnitSystem($('sel-len').value, $('sel-force').value);
    if (nu.equals(A.scene.units)) return;
    A.scene.convertUnits(nu);
    A.mesh = Mesh.fromBytes(A.meshBytes, A.meshName, A.scene.mesh.scale);
    if (A.utmText) parseUtm();
    rebuildTopology(); syncPanels(); status(`units → ${nu}: values converted, model rebuilt`);
  };
  $('sel-len').onchange = unitEdit; $('sel-force').onchange = unitEdit;
  $('ck-hybrid').onchange = (e) => { A.scene.hybrid.enabled = e.target.checked; if (e.target.checked && !A.scene.hybrid.groups.length) A.scene.setNGroups(2); $('sel-group').value = 1; rebuildTopology({ keepCamera: true }); syncPanels(); };
  $('nb-ngroups').onchange = (e) => { const n = Math.min(6, Math.max(1, +e.target.value | 0)); if (n !== A.scene.nGroups) { A.scene.setNGroups(n); rebuildTopology({ keepCamera: true }); syncPanels(); } };
  $('sel-group').onchange = () => { syncPanels(); updateAssignReadout(); };
  $('tx-grpname').onchange = (e) => { const g = +$('sel-group').value; if (g >= 2) { A.scene.group(g).name = e.target.value; syncPanels(); } };
  $('btn-assign').onclick = () => {
    const g = +$('sel-group').value, strands = lathsToAssign();
    if (!strands.length) { status('select laths first: Select laths… and click them in the view', 'warn'); return; }
    A.scene.assignStrands(strands, g); A.lathSel = new Set(); A.selection = [];
    rebuildTopology({ keepCamera: true }); syncPanels(); updateAssignReadout();
    status(g >= 2 ? `${strands.length} lath(s) → group ${g}` : `${strands.length} lath(s) back to group 1`);
  };
  $('btn-lathmode').onclick = () => setMode('laths');
  $('btn-selgroup').onclick = () => { if (A.mode !== 'laths') setMode('laths'); const g = +$('sel-group').value || 1; setLathSelection(Array.from({ length: A.topo.nStrands }, (_, s) => s).filter(s => A.scene.strandGroup(s) === g)); };
  $('btn-lathclear').onclick = () => setLathSelection([]);
  qsa('[data-bind]').forEach(el => { el.addEventListener(el.type === 'range' ? 'input' : 'change', () => writeBound(el)); });
  $('sel-beammode').onchange = (e) => {
    const g = A.scene.group(+$('sel-group').value || 1), b = g.beam;
    if (e.target.value === 'advanced' && b.mode === 'simple') { b.activation_flat = b.activation_edge = b.activation_twist = b.activation; b.stiffness_flat = b.stiffness_edge = b.stiffness_twist = b.stiffness; }
    b.mode = e.target.value; syncPanels(); onSceneEdited();
  };
  $('ck-override').onchange = (e) => {
    const s = A.scene.group(+$('sel-group').value || 1).section;
    if (e.target.checked) { s.Ix = IxEff(s); s.Iy = IyEff(s); s.J = JEff(s); } else { s.Ix = s.Iy = s.J = null; }
    syncPanels(); onSceneEdited();
  };
  for (const [id, k] of [['nb-Ix', 'Ix'], ['nb-Iy', 'Iy'], ['nb-J', 'J']]) $(id).onchange = (e) => { const s = A.scene.group(+$('sel-group').value || 1).section; if (s[k] != null) { s[k] = +e.target.value; onSceneEdited(); } };
  // loads
  const loadEdit = () => {
    const L = A.scene.loads;
    L.gravity.enabled = $('ck-grav').checked; L.gravity.direction = ['gx', 'gy', 'gz'].map(k => +$('nb-' + k).value);
    L.udl.enabled = $('ck-udl').checked; L.udl.q = ['qx', 'qy', 'qz'].map(k => +$('nb-' + k).value); L.udl.per = $('sel-udlper').value;
    onSceneEdited();
  };
  ['ck-grav', 'nb-gx', 'nb-gy', 'nb-gz', 'ck-udl', 'nb-qx', 'nb-qy', 'nb-qz', 'sel-udlper'].forEach(id => $(id).addEventListener('change', loadEdit));
  $('sel-loaddir').onchange = (e) => { $('load-vec').hidden = e.target.value !== 'world'; $('sel-loadaxis').hidden = e.target.value === 'world'; };
  $('btn-addload').onclick = () => {
    if (!A.selection.length) { status('select frames first', 'warn'); return; }
    const mag = +$('nb-lmag').value; let n = 0;
    for (const f of A.selection) {
      if ($('sel-loaddir').value === 'world') {
        const v = ['lx', 'ly', 'lz'].map(k => +$('nb-' + k).value), nv = Math.hypot(...v); if (nv < 1e-12) continue;
        A.scene.addPointLoad(f, v.map(c => c / nv * mag), 'world');
      } else { const [ax, sg] = AXIS_VECTORS[+$('sel-loadaxis').value], F = [0, 0, 0]; F[ax] = sg * mag; A.scene.addPointLoad(f, F, 'local'); }
      n++;
    }
    status(`added ${n} point load(s)`); refreshOverlays(); onSceneEdited();
  };
  $('list-loads').onclick = (e) => {
    const t = e.target;
    if (t.dataset.delLoad != null) { A.scene.loads.point.splice(+t.dataset.delLoad, 1); refreshOverlays(); onSceneEdited(); }
    else if (t.dataset.selLoad != null) { const pl = A.scene.loads.point[+t.dataset.selLoad]; select([pl.frame]); }
  };
  // anchors
  $('btn-sel-start').onclick = () => select(A.topo.startFrames()); $('btn-sel-end').onclick = () => select(A.topo.endFrames());
  $('btn-sel-bnd').onclick = () => select(A.topo.boundaryFrames()); $('btn-sel-clear').onclick = () => select([]);
  const fixMask = () => qsa('#fix-dofs input[data-dof]').sort((a, b) => a.dataset.dof - b.dataset.dof).map(c => (c.checked ? 1 : 0));
  $('btn-pin').onclick = () => qsa('#fix-dofs input[data-dof]').forEach(c => { c.checked = +c.dataset.dof < 3; });
  $('btn-fixed').onclick = () => qsa('#fix-dofs input[data-dof]').forEach(c => { c.checked = true; });
  $('btn-addanchor').onclick = () => {
    if (!A.selection.length) { status('select frames first', 'warn'); return; }
    const sel = A.selection.length === 1 ? { type: 'frame', id: [...A.selection[0]] } : { type: 'knots', ids: A.selection.map(f => [...f]) };
    A.scene.addAnchor(sel, fixMask()); refreshOverlays(); onSceneEdited(); status(`anchor added: ${A.selection.length} frame(s)`);
  };
  qsa('[data-anchorsel]').forEach(b => { b.onclick = () => { A.scene.addAnchor({ type: b.dataset.anchorsel }, fixMask(), b.dataset.anchorsel); refreshOverlays(); onSceneEdited(); }; });
  $('list-anchors').onclick = (e) => {
    const t = e.target;
    if (t.dataset.delAnchor != null) { A.scene.anchors.splice(+t.dataset.delAnchor, 1); refreshOverlays(); onSceneEdited(); }
    else if (t.dataset.selAnchor != null) select(resolveSelector(A.scene.anchors[+t.dataset.selAnchor].select, A.topo));
  };
  // solve
  qsa('#backend-chips .chip').forEach(c => { c.onclick = () => { store.set('backend', c.dataset.backend); setBackend(c.dataset.backend); }; });
  $('btn-polish').onclick = polish;
  // fastened-structure protocol (Zero → settle → Lock): protocol_ui.js; the 4 Contact checkbox is the same lock
  $('ck-locked').addEventListener('change', (e) => { proto.setLocked(e.target.checked); });
  // results (colour / diagram controls: display_ui.js)
  $('btn-refresh-results').onclick = refreshResults;
  // UTM
  $('btn-probemode').onclick = () => setMode('probe');
  ['nb-ax', 'nb-ay', 'nb-az'].forEach(id => $(id).addEventListener('change', () => { if (probeOf(A.scene)) { A.scene.utm.probe.axis = [+$('nb-ax').value, +$('nb-ay').value, +$('nb-az').value]; refreshOverlays(); } }));
  $('file-utm').onchange = async (e) => { const f = e.target.files[0]; if (f) loadUtmText(await f.text(), f.name); e.target.value = ''; };
  ['sel-utmforce', 'sel-utmdisp', 'sel-utmfu', 'sel-utmlu'].forEach(id => $(id).addEventListener('change', parseUtm));
  $('btn-replay').onclick = runReplay; $('btn-replay-stop').onclick = () => replayAbort?.abort();
  // view
  qsa('#theme-chips .chip').forEach(c => { c.onclick = () => setTheme(c.dataset.theme); });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { view.applyTheme(); drawChart(); drawUtmPlot(); });
  qsa('#layer-chips .chip').forEach(c => { c.onclick = () => { c.classList.toggle('on'); view.setVisible(c.dataset.layer, c.classList.contains('on')); if (c.dataset.layer === 'frames') refreshOverlays(); }; });
  $('sl-deform').oninput = (e) => { $('out-deform').textContent = e.target.value; view.setDeform(+e.target.value); display.refreshJoints(); display.refreshDiagram(); };
  $('sl-arrow').oninput = (e) => { $('out-arrow').textContent = (+e.target.value).toFixed(1); refreshOverlays(); };
  $('sl-turn').oninput = (e) => { $('out-turn').textContent = e.target.value; if (view.turntable) view.setTurntable(true, +e.target.value); };
  // modal
  $('modal-close').onclick = () => { $('modal').hidden = true; };
  $('modal-copy').onclick = () => { navigator.clipboard?.writeText($('modal-text').value); status('copied'); };
  window.addEventListener('resize', () => { drawChart(); drawUtmPlot(); calib?.redraw?.(); disc?.redraw(); });
}

/** continue from the current (GPU) state on the CPU in f64 until the residual is below the tolerance */
async function polish() {
  if (!A.model) return;
  stopRun(); await waitIdle();
  if (A.backend === 'webgpu') await setBackend('cpu');
  status('f64 polish on the CPU …');
  startRun();
}

init().catch(e => { console.error(e); status('start-up failed: ' + e.message, 'warn'); });
