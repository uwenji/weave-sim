// Calibration panel: tests (probe frame, load axis, UTM csv, weight, training / validation), the parameter box,
// replay settings, and the batched runs — objective, sensitivity sweep, fit (Latin hypercube + Nelder–Mead or CMA-ES),
// validation on held-out tests, synthetic data — with the force–travel curves, the objective history and the
// parameter-set export that carries a calibration to other weave topologies.
import { CalibrationSet, FileResolver, Test, Probe, PARAMS, PARAM_INFO, PARAM_CLASS, LOG_SCALED, replaySettings, fit, sensitivity, validate, makeSynthetic, parameterSet, applyParameterSet, readParams } from './calibrate.js';
import { parseUTMCsv } from './utm.js';
import { drawPlot } from './plot.js';
import { Scene } from './scene.js';

const PARAM_ORDER = ['E', 'G', 'activation', 'stiffness', 'stiffness_twist', 'mu', 'strength', 'stick_strength', 'torsion_factor', 'slip_rate', 'rot_factor', 'fc90', 'H', 'E90', 'pad_stiffness'];
const REPLAY_FIELDS = [['mode', 'mode', 'sel', ['force', 'displacement']], ['relax_iter', 'relax iterations', 'int'], ['step_iter', 'iterations / increment', 'int'], ['avg_iter', 'averaged over', 'int'],
  ['n_steps', 'increments / branch', 'int'], ['cycles', 'load cycles', 'int'], ['fit_cycles', 'fitted cycles (0-based)', 'list'], ['max_force', 'peak force (blank = curve)', 'num'], ['max_disp', 'travel cap (displacement)', 'num'],
  ['lock', 'lock crossings', 'bool'], ['replicas', 'chaos replicas', 'int'], ['perturb', 'replica perturbation', 'num']];
const TEST_COLORS = ['--chart-1', '--chart-3', '--chart-4', '--chart-2'];

export function initCalibration(ctx) {
  const { A, $, qsa, status, fmt, refreshOverlays, download, showModal, setPickBanner, cssv } = ctx;
  const C = { cs: null, enabled: {}, bounds: {}, running: false, abort: null, history: [], best: null, sens: null, sims: null, simValues: null, valid: null, picking: false, synthTruth: null, backend: null };
  if (typeof window !== 'undefined') window.weavesimCalib = C;     // console access
  const body = $('calib-body');
  body.innerHTML = `
    <div class="row buttons">
      <label class="btn file" title="calib.json with its scene, mesh and the UTM csv files (select them together, or a folder)"><input type="file" id="cal-files" multiple hidden>Open calibration…</label>
      <label class="btn file" title="a folder with calib.json, the scene, the csv files (and the mesh)"><input type="file" id="cal-folder" webkitdirectory hidden>Folder…</label>
      <button id="cal-example" class="btn" title="examples/calibration/force: 3 cable-pull tests + 1 validation on a dome, synthetic data">Force example</button>
      <button id="cal-new" class="btn" title="a calibration on the model in the view: add the tests below">From current scene</button>
    </div>
    <div class="readout" id="cal-ro">no calibration — open calib.json (+ scene, mesh, csv), the force example, or start from the current scene</div>
    <div id="cal-main" hidden>
      <h3>tests</h3>
      <table class="tbl" id="cal-tests"></table>
      <div class="row buttons"><button id="cal-addtest" class="btn sm toggle">+ test at a crossing…</button><button id="cal-probe2test" class="btn sm" title="the probe and UTM curve of section 9 as a new test">+ from 9 UTM</button></div>
      <h3>parameters <span class="unit" id="cal-units"></span></h3>
      <table class="tbl" id="cal-params"></table>
      <details class="sub"><summary>replay protocol</summary><div class="grid2" id="cal-replay"></div>
        <p class="hint">force mode = the cable pull: n_steps increments up to the peak force and back, cycles times; crosshead travel = probe travel + F / pad_stiffness. Chaos replicas &gt; 1 replay every test from slightly perturbed copies of the relaxed state and average the residuals — the stick–slip replay is chaotic, one replay is one sample of a noisy objective.</p></details>
      <div class="row"><label>run on</label><div class="chips inline" id="cal-backend"><button class="chip" data-b="gpu">WebGPU batch</button><button class="chip" data-b="cpu">CPU f64</button></div>
        <label for="cal-maxinst" title="GPU instances per batch (parameter vectors × tests × replicas)">batch ≤</label><input type="number" id="cal-maxinst" class="num xs" min="1" step="1" value="192"></div>
      <div class="row buttons">
        <button id="cal-obj" class="btn" title="objective at the values in the table (fitted columns at their 'value')">Objective</button>
        <button id="cal-sens" class="btn" title="one-at-a-time sweep across each fitted parameter's bounds (one GPU batch)">Sensitivity</button><input type="number" id="cal-sensn" class="num xs" min="2" max="11" value="5" title="values per parameter">
        <button id="cal-validate" class="btn" title="replay the held-out (validation) tests at the best values">Validate</button>
        <button id="cal-stop" class="btn" disabled>Stop</button>
      </div>
      <div class="row buttons">
        <button id="cal-fit" class="btn primary">Fit</button>
        <select id="cal-method"><option value="cmaes">CMA-ES (population batches)</option><option value="nm">Nelder–Mead (Python fit)</option></select>
        <label for="cal-ninit">LHS</label><input type="number" id="cal-ninit" class="num xs" min="2" step="1" value="8">
        <label for="cal-maxeval">evals</label><input type="number" id="cal-maxeval" class="num xs" min="4" step="1" value="60">
        <label for="cal-seed">seed</label><input type="number" id="cal-seed" class="num xs" step="1" value="1">
      </div>
      <div class="progress"><div id="cal-prog"></div></div>
      <div class="readout" id="cal-status">—</div>
      <div class="log" id="cal-log" hidden></div>
      <div class="plots" id="cal-plots"></div>
      <canvas id="cal-hist" class="plot sm"></canvas>
      <canvas id="cal-sensplot" class="plot sm" hidden></canvas>
      <div class="readout" id="cal-best">—</div>
      <h3>parameter set</h3>
      <div class="row buttons">
        <button id="cal-export" class="btn" title="the best values (or the table's), with units, bounds, residuals and the calibration's section — load it into other scenes">Export parameter set</button>
        <button id="cal-apply" class="btn" title="put the best values into the scene in the view (material / contact / beam; the test-rig pad stays out)">Apply to scene</button>
        <label class="btn file"><input type="file" id="cal-import" accept=".json" hidden>Import &amp; apply…</label>
      </div>
      <div class="row buttons">
        <button id="cal-synth" class="btn" title="replace the curves with model curves at the table values (+ noise): an identifiability check — fit, then compare with the values used">Synthetic data…</button>
        <label for="cal-noise">noise</label><input type="number" id="cal-noise" class="num xs" step="0.005" value="0.02">
        <button id="cal-save" class="btn">Save calib.json</button>
      </div>
      <div class="readout" id="cal-transfer"></div>
    </div>`;

  // ------------------------------------------------------------------------------------------------ loading
  async function loadFromFiles(list) {
    const files = new Map(list.map(f => [f.webkitRelativePath || f.name, f]));
    const cf = list.find(f => /calib.*\.json$/i.test(f.name));
    if (!cf) { status('no calib.json among the files', 'warn'); return; }
    const calib = JSON.parse(await cf.text());
    const relDir = (cf.webkitRelativePath || cf.name).split('/').slice(0, -1).join('/');
    const fallback = new URL('../examples/calibration/force/', location.href);
    const resolver = new FileResolver({ files, reader: async (rel, kind) => {
      // not among the files (the mesh of the example lives two folders up): try next to the page's examples
      const r = await fetch(new URL(rel, fallback)); if (!r.ok) throw new Error(`file not found: ${rel} — select it together with calib.json`);
      return kind === 'text' ? r.text() : r.arrayBuffer();
    } });
    void relDir;
    await open(calib, resolver, cf.name);
  }
  async function loadExample() {
    const base = new URL('../examples/calibration/force/calib.json', location.href);
    try {
      const calib = await (await fetch(base)).json();
      await open(calib, new FileResolver({ base }), 'examples/calibration/force/calib.json');
      const truth = await (await fetch(new URL('synthetic_truth.json', base))).json().catch(() => null);
      if (truth) { C.synthTruth = truth; log(`synthetic truth of the example: ${Object.entries(truth).map(([k, v]) => `${k}=${v}`).join('  ')}`); }
    } catch (e) { status(`calibration example not reachable (${e.message}) — serve the WeaveSim folder with web/serve.py`, 'warn'); }
  }
  async function open(calib, resolver, label) {
    status('loading calibration …');
    const cs = await CalibrationSet.load(calib, resolver, label);
    setCalibration(cs, label);
    // show its scene in the view (the probes become visible); the calibration keeps its own copy
    try {
      const meshRel = cs.scene.mesh.path;
      const sceneText = await resolver.text(calib.scene);
      const sc = Scene.fromJSON(JSON.parse(sceneText), calib.scene.replace(/\.json$/i, '').split('/').pop());
      const dir = calib.scene.includes('/') ? calib.scene.slice(0, calib.scene.lastIndexOf('/') + 1) : '';
      const bytes = new Uint8Array(await resolver.bytes(joinRel(dir, meshRel)));
      ctx.openScene?.(sc, bytes, meshRel.split('/').pop());
    } catch (e) { console.warn('calibration scene not shown:', e); }
    status(`calibration ${label}: ${cs.tests.length} tests, ${cs.validation.length} validation`);
  }
  const joinRel = (dir, rel) => { const out = []; for (const p of (dir + rel).split('/')) { if (p === '..' && out.length && out[out.length - 1] !== '..') out.pop(); else if (p && p !== '.') out.push(p); } return out.join('/'); };

  function fromScene() {
    if (!A.scene || !A.topo) return;
    const sc = A.scene, c = sc.contact;
    const params = {
      activation: [0.2, 1.0], mu: [0.15, 0.8], torsion_factor: [0.3, 5.0],
      strength: [Math.max(c.strength * 0.3, 1e-9), c.strength * 5], stick_strength: [Math.max(c.stick_strength * 0.2, 1e-9), c.stick_strength * 5],
    };
    const cs = CalibrationSet.fromModel(sc, A.topo, { params, replay: { relax_iter: Math.max(2500, (c.ramp_iter | 0) + 1000), step_iter: 500, avg_iter: 150, n_steps: 4, cycles: 2, fit_cycles: [1] }, sceneFile: `${A.sceneName}.json` });
    setCalibration(cs, `${A.sceneName} (current scene)`);
    status('calibration on the current scene: add tests (a crossing + axis + UTM csv), tick the parameters to fit');
  }

  function setCalibration(cs, label) {
    C.cs = cs; C.label = label; C.history = []; C.best = null; C.sens = null; C.sims = null; C.valid = null;
    C.enabled = {}; C.bounds = {}; C.values = readParams(cs.scene, PARAM_ORDER.filter(k => PARAMS[k]));
    for (const k of PARAM_ORDER) {
      const fitted = k in cs.params;
      C.enabled[k] = fitted;
      const v = fitted ? Math.sqrt(Math.abs(cs.params[k][0] * cs.params[k][1])) : C.values[k];
      C.bounds[k] = fitted ? [...cs.params[k]] : defaultBounds(k, C.values[k]);
      if (fitted && !(k in cs.fixed)) C.values[k] = LOG_SCALED.has(k) ? v : 0.5 * (cs.params[k][0] + cs.params[k][1]);
    }
    for (const [k, v] of Object.entries(cs.fixed)) C.values[k] = v;
    if ('pad_stiffness' in cs.params) C.values.pad_stiffness = Math.sqrt(cs.params.pad_stiffness[0] * cs.params.pad_stiffness[1]);
    C.backend = A.gpuAvailable && A.backend === 'webgpu' ? 'gpu' : 'cpu';
    $('cal-main').hidden = false;
    render(); refreshOverlays();
  }
  function defaultBounds(k, v) {
    if (k === 'activation') return [0.2, 1.0];
    if (k === 'mu') return [0.1, 0.8];
    if (k === 'slip_rate') return [0.1, 1.0];
    if (k === 'pad_stiffness') return [500, 20000];
    if (!(v > 0)) return [0.1, 10];
    return [+(v / 4).toPrecision(3), +(v * 4).toPrecision(3)];
  }

  // ------------------------------------------------------------------------------------------------ rendering
  function render() {
    const cs = C.cs; if (!cs) return;
    const u = cs.scene.units;
    $('cal-ro').textContent = `${C.label} · scene ${cs.sceneFile || '—'} · ${cs.topo.nStrands} strands, ${cs.topo.nKnots} knots · [${u}]`;
    $('cal-units').textContent = `[${u}]`;
    // tests
    const rows = cs.tests.map((t, i) => ({ t, role: 'train', i })).concat(cs.validation.map((t, i) => ({ t, role: 'valid', i })));
    $('cal-tests').innerHTML = `<tr><th>name</th><th>frame</th><th>axis</th><th>curve</th><th>w</th><th>role</th><th></th></tr>` + rows.map(({ t, role, i }) => `
      <tr data-role="${role}" data-i="${i}">
        <td><input class="txt" style="width:40px" data-f="name" value="${t.name}"></td>
        <td class="mono">${t.probe.frame[0]},${t.probe.frame[1]}</td>
        <td><input class="num" style="width:76px" data-f="axis" value="${t.probe.axis.map(v => +v.toFixed(3)).join(',')}" title="load direction x,y,z"></td>
        <td><label class="btn sm file" title="${t.csv || ''}">${t.curve ? `${t.curve.length} rows` : 'csv…'}<input type="file" accept=".csv,.txt" data-f="csv" hidden></label></td>
        <td>${role === 'train' ? `<input class="num xs" data-f="weight" value="${t.weight}">` : ''}</td>
        <td><select data-f="role"><option value="train"${role === 'train' ? ' selected' : ''}>training</option><option value="valid"${role === 'valid' ? ' selected' : ''}>validation</option></select></td>
        <td><button class="btn sm" data-f="del" title="remove">×</button></td></tr>`).join('');
    // parameters
    const sc = cs.scene;
    $('cal-params').innerHTML = `<tr><th>fit</th><th>parameter</th><th>low</th><th>high</th><th>value</th><th>best</th></tr>` + PARAM_ORDER.map(k => {
      const on = C.enabled[k], [lo, hi] = C.bounds[k], best = C.best ? C.best.best[k] : undefined;
      return `<tr class="${on ? '' : 'off'}" title="${PARAM_INFO[k] || ''} (${PARAM_CLASS[k] || 'model'})"><td><input type="checkbox" data-p="${k}" data-f="on"${on ? ' checked' : ''}></td>
        <td class="mono pname">${k}${LOG_SCALED.has(k) ? '<sup class="unit">log</sup>' : ''}</td>
        <td><input class="num xs" data-p="${k}" data-f="lo" value="${+(+lo).toPrecision(4)}"></td><td><input class="num xs" data-p="${k}" data-f="hi" value="${+(+hi).toPrecision(4)}"></td>
        <td><input class="num sm" data-p="${k}" data-f="val" value="${+(+C.values[k]).toPrecision(5)}" title="the fixed value (unticked), the centre of objective / sensitivity runs (ticked)"></td>
        <td class="mono best">${best !== undefined ? fmt(best, 4) : ''}</td></tr>`;
    }).join('');
    void sc;
    // replay
    const rp = cs.replay;
    $('cal-replay').innerHTML = REPLAY_FIELDS.map(([k, label, t, opts]) => {
      const v = rp[k];
      if (t === 'sel') return `<label>${label}</label><select data-crp="${k}">${opts.map(o => `<option${o === v ? ' selected' : ''}>${o}</option>`).join('')}</select>`;
      if (t === 'bool') return `<label>${label}</label><input type="checkbox" data-crp="${k}"${v ? ' checked' : ''}>`;
      if (t === 'list') return `<label>${label}</label><input class="num" data-crp="${k}" value="${(v || []).join(',')}">`;
      return `<label>${label}</label><input type="number" class="num" data-crp="${k}" value="${v ?? ''}" step="${t === 'int' ? 1 : 'any'}">`;
    }).join('');
    qsa('#cal-backend .chip').forEach(c => { c.classList.toggle('on', c.dataset.b === C.backend); c.disabled = c.dataset.b === 'gpu' && !A.gpuAvailable; });
    $('cal-maxinst').value = cs.maxInstances;
    renderPlots(); renderBest(); drawHistory();
    $('sum-calib').textContent = `${cs.tests.length}+${cs.validation.length} tests · ${Object.values(C.enabled).filter(Boolean).length} params`;
  }

  /** the CalibrationSet as the table says: fitted parameters and bounds, fixed values applied to its scene */
  function commit() {
    const cs = C.cs;
    cs.params = {}; const fixed = {};
    for (const k of PARAM_ORDER) {
      if (C.enabled[k]) cs.params[k] = [+C.bounds[k][0], +C.bounds[k][1]];
      else if (k in cs.fixed || C.values[k] !== readParams(cs.scene, [k])[k]) fixed[k] = +C.values[k];
    }
    for (const [k, v] of Object.entries(fixed)) PARAMS[k][1](cs.scene, v);
    cs.fixed = { ...cs.fixed, ...fixed };
    for (const k of Object.keys(cs.params)) delete cs.fixed[k];
    cs.maxInstances = Math.max(1, +$('cal-maxinst').value | 0);
    cs.setBackend(C.backend === 'gpu' && A.device ? 'gpu' : 'cpu', A.device);
    return cs;
  }
  /** table values of the fitted parameters (clamped into their boxes) */
  function centre() { const v = {}; for (const k of Object.keys(C.cs.params)) { const [lo, hi] = C.cs.params[k]; v[k] = Math.min(Math.max(+C.values[k], lo), hi); } return v; }

  // ------------------------------------------------------------------------------------------------ table events
  body.addEventListener('change', async (e) => {
    const el = e.target, cs = C.cs; if (!cs) return;
    if (el.dataset.p) {
      const k = el.dataset.p, f = el.dataset.f;
      if (f === 'on') C.enabled[k] = el.checked; else if (f === 'lo') C.bounds[k][0] = +el.value; else if (f === 'hi') C.bounds[k][1] = +el.value; else if (f === 'val') C.values[k] = +el.value;
      cs._scales = null; render(); return;
    }
    if (el.dataset.crp) {
      const k = el.dataset.crp, t = REPLAY_FIELDS.find(r => r[0] === k)[2];
      cs.replay[k] = t === 'bool' ? el.checked : t === 'sel' ? el.value : t === 'list' ? el.value.split(/[,\s]+/).filter(Boolean).map(Number) : el.value === '' ? null : (t === 'int' ? Math.round(+el.value) : +el.value);
      cs._scales = null; return;
    }
    const tr = el.closest('tr[data-role]');
    if (tr) {
      const list = tr.dataset.role === 'train' ? cs.tests : cs.validation, t = list[+tr.dataset.i], f = el.dataset.f;
      if (f === 'name') t.probe.name = el.value.trim() || t.probe.name;
      else if (f === 'axis') { const a = el.value.split(/[,\s]+/).map(Number); if (a.length === 3 && a.every(Number.isFinite)) t.probe = new Probe(t.probe.frame, a, t.probe.name); }
      else if (f === 'weight') t.weight = +el.value;
      else if (f === 'role') { list.splice(+tr.dataset.i, 1); (el.value === 'train' ? cs.tests : cs.validation).push(t); }
      else if (f === 'csv' && el.files[0]) { const file = el.files[0]; try { t.curve = parseUTMCsv(await file.text(), cs.scene.units, { source: file.name }); t.csv = file.name; } catch (err) { status(err.message, 'warn'); } }
      cs._scales = null; render(); refreshOverlays();
    }
  });
  body.addEventListener('click', (e) => {
    const el = e.target, cs = C.cs;
    if (el.dataset.f === 'del') { const tr = el.closest('tr[data-role]'); (tr.dataset.role === 'train' ? cs.tests : cs.validation).splice(+tr.dataset.i, 1); render(); refreshOverlays(); }
  });
  $('cal-files').onchange = (e) => { loadFromFiles([...e.target.files]).catch(err => status(err.message, 'warn')); e.target.value = ''; };
  $('cal-folder').onchange = (e) => { loadFromFiles([...e.target.files]).catch(err => status(err.message, 'warn')); e.target.value = ''; };
  $('cal-example').onclick = () => loadExample();
  $('cal-new').onclick = fromScene;
  $('cal-addtest').onclick = () => { C.picking = !C.picking; $('cal-addtest').classList.toggle('on', C.picking); setPickBanner(C.picking ? 'TEST — click the lath at the probe crossing (Esc leaves)' : ''); };
  $('cal-probe2test').onclick = () => {
    const pr = A.scene?.utm?.probe && Array.isArray(A.scene.utm.probe.frame) ? A.scene.utm.probe : null; if (!pr || !C.cs) { status('section 9: pick a probe (and load its csv) first', 'warn'); return; }
    const n = C.cs.tests.length + C.cs.validation.length + 1;
    C.cs.tests.push(new Test(new Probe(pr.frame, pr.axis || [0, 0, -1], `P${n}`), A.utm ? parseUTMCsv(A.utmText, C.cs.scene.units, { source: A.utmName }) : null, 1.0, A.utmName || ''));
    render(); refreshOverlays();
  };
  qsa('#cal-backend .chip').forEach(c => { c.onclick = () => { if (c.disabled) return; C.backend = c.dataset.b; render(); }; });

  // ------------------------------------------------------------------------------------------------ runs
  function log(s) { const el = $('cal-log'); el.hidden = false; el.textContent += s + '\n'; el.scrollTop = el.scrollHeight; }
  function progress(p) {
    const bar = $('cal-prog');
    if (!p) { bar.style.width = '0'; return; }
    const total = p.count || 1, done = (p.candidate || 0) + (p.batch ? (p.stage === 'relax' ? 0.25 : 0.25 + 0.75 * p.frac) * p.batch : p.stage === 'relax' ? 0.25 * p.frac : 0.25 + 0.75 * p.frac);
    bar.style.width = `${Math.min(100, 100 * done / total).toFixed(1)}%`;
    $('cal-status').textContent = `${p.stage} · ${p.batch ? `batch of ${p.batch} parameter vector(s)` : `vector ${(p.candidate || 0) + 1} / ${total}`} · ${(100 * p.frac).toFixed(0)} %`;
  }
  async function guarded(label, fn) {
    if (C.running || !C.cs) return;
    ctx.stopRun(); await ctx.waitIdle();
    C.running = true; C.abort = new AbortController();
    qsa('#cal-main button').forEach(b => { if (b.id !== 'cal-stop') b.disabled = true; }); $('cal-stop').disabled = false;
    const t0 = performance.now();
    try { commit(); await fn(C.abort.signal); $('cal-status').textContent = `${label} done in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${C.cs.backend === 'gpu' ? 'WebGPU' : 'CPU f64'}`; }
    catch (e) { $('cal-status').textContent = e.name === 'AbortError' ? `${label} stopped` : `${label} failed: ${e.message}`; if (e.name !== 'AbortError') console.error(e); }
    finally { C.running = false; C.abort = null; qsa('#cal-main button').forEach(b => { b.disabled = false; }); $('cal-stop').disabled = true; progress(null); render(); }
  }
  $('cal-stop').onclick = () => C.abort?.abort();

  $('cal-obj').onclick = () => guarded('objective', async (signal) => {
    const v = centre();
    if (!C.cs.tests.some(t => t.curve)) throw new Error('no training test has a UTM curve');
    const r = await C.cs.objective(v, { signal, onProgress: progress });
    C.sims = r.sims; C.simValues = v;
    log(`objective ${r.objective.toFixed(4)}${r.objectiveStd ? ` ± ${r.objectiveStd.toFixed(4)}` : ''}  ` + Object.entries(r.residuals).map(([k, x]) => `${k}:${x.toFixed(4)}`).join('  ') + '   at ' + Object.entries(v).map(([k, x]) => `${k}=${fmt(x, 4)}`).join(' '));
  });
  $('cal-sens').onclick = () => guarded('sensitivity', async (signal) => {
    const n = Math.min(11, Math.max(2, +$('cal-sensn').value | 0));
    if (!Object.keys(C.cs.params).length) throw new Error('tick at least one parameter');
    C.sens = await sensitivity(C.cs, centre(), n, { signal, onProgress: progress });
    for (const [k, rows] of Object.entries(C.sens)) log(`${k.padEnd(15)} ` + rows.map(r => `${fmt(r.value, 3)}:${r.objective.toFixed(3)}`).join('  '));
    drawSensitivity();
  });
  $('cal-fit').onclick = () => guarded('fit', async (signal) => {
    if (!Object.keys(C.cs.params).length) throw new Error('tick at least one parameter');
    if (!C.cs.tests.some(t => t.curve)) throw new Error('no training test has a UTM curve');
    C.history = []; C.best = null;
    const method = $('cal-method').value;
    log(`fit (${method}) of ${C.cs.names().join(', ')} on ${C.cs.tests.map(t => t.name).join(', ')} — ${C.cs.backend === 'gpu' ? 'WebGPU batches' : 'CPU f64'}`);
    const res = await fit(C.cs, { nInit: Math.max(2, +$('cal-ninit').value | 0), maxEval: Math.max(4, +$('cal-maxeval').value | 0), seed: +$('cal-seed').value | 0, method, signal, onProgress: progress, log,
      onEval: (h, hist) => { C.history = hist; if (!C.best || h.objective < C.best.objective) C.best = { best: h.values, objective: h.objective, residuals: h.residuals }; drawHistory(); renderBest(); } });
    C.best = res; C.history = res.history.map(h => ({ objective: h.objective, values: h })); C.fit = res;
    log(`best objective ${res.objective.toFixed(4)} after ${res.evaluations} evaluations (${res.computed} computed) in ${res.batches} batch(es) — ${res.message}`);
    // curves at the best values
    const r = await C.cs.objective(res.best, { signal, onProgress: progress });
    C.sims = r.sims; C.simValues = res.best;
    for (const k of Object.keys(res.best)) C.values[k] = res.best[k];
  });
  $('cal-validate').onclick = () => guarded('validation', async (signal) => {
    if (!C.cs.validation.length) throw new Error('no validation tests (set a test\'s role to validation)');
    const v = C.best ? C.best.best : centre();
    C.valid = await validate(C.cs, v, { signal, onProgress: progress });
    log('validation ' + Object.entries(C.valid.residuals).map(([k, x]) => `${k}:${x.toFixed(4)}${C.valid.spread[k] ? `±${C.valid.spread[k].toFixed(4)}` : ''}`).join('  '));
  });
  $('cal-synth').onclick = () => guarded('synthetic data', async (signal) => {
    const v = centre();
    const csvs = await makeSynthetic(C.cs, v, { noise: +$('cal-noise').value, nPoints: 40, seed: +$('cal-seed').value | 0, signal, onProgress: progress });
    C.synthTruth = { ...v }; C.sims = null;
    log(`synthetic curves at ${Object.entries(v).map(([k, x]) => `${k}=${fmt(x, 4)}`).join(' ')} (noise ${$('cal-noise').value}) — fit now and compare with these values`);
    showModal('Synthetic UTM curves', 'one csv per test (the tests now use them); save them next to calib.json to reuse', Object.entries(csvs).map(([k, t]) => `# ${k}.csv\n${t}`).join('\n'), 'synthetic_curves.txt');
  });
  $('cal-save').onclick = () => { if (!C.cs) return; commit(); download('calib.json', JSON.stringify(C.cs.toJSON(), null, 1)); };
  $('cal-export').onclick = () => {
    if (!C.cs) return; commit();
    const v = C.best ? C.best.best : centre();
    const ps = parameterSet(C.cs, v, { objective: C.best?.objective ?? null, residuals: C.best?.residuals ?? null, validation: C.valid?.residuals ?? null, method: C.fit?.method || '' });
    showModal('Parameter set', 'apply it to any scene: Calibration › Import & apply (units are converted; the loading-pad stiffness stays with the test rig)', JSON.stringify(ps, null, 1), 'weavesim_parameters.json');
  };
  $('cal-apply').onclick = () => {
    if (!C.cs || !A.scene) return; commit();
    const ps = parameterSet(C.cs, C.best ? C.best.best : centre());
    applySet(ps);
  };
  $('cal-import').onchange = async (e) => { const f = e.target.files[0]; e.target.value = ''; if (!f) return; try { applySet(JSON.parse(await f.text())); } catch (err) { status(err.message, 'warn'); } };
  function applySet(ps) {
    const { applied, warnings } = applyParameterSet(A.scene, ps);
    ctx.applySceneValues();
    const u = A.scene.units;
    $('cal-transfer').innerHTML = `applied to ${A.sceneName} [${u}]: ${Object.entries(applied).map(([k, v]) => `${k}=${fmt(v, 4)}`).join('  ')}` + (warnings.length ? `\n<span class="warn">${warnings.join('\n')}</span>` : '') + '\nmaterial / contact values apply live; the activation applies on Reset.';
    status('parameter set applied to the scene');
  }

  // ------------------------------------------------------------------------------------------------ plots
  function renderPlots() {
    const cs = C.cs, box = $('cal-plots'); if (!cs) return;
    const tests = cs.tests.concat(cs.validation);
    box.innerHTML = tests.map((t, i) => `<div><div class="cap">${t.name} (${t.probe.frame}) ${cs.validation.includes(t) ? 'validation' : 'training'}${resText(t)}</div><canvas data-plot="${i}"></canvas></div>`).join('');
    const u = cs.scene.units;
    tests.forEach((t, i) => {
      const series = [];
      if (t.curve) series.push({ x: t.curve.displacement, y: t.curve.force, color: cssv('--fg'), dots: true, r: 1.5 });
      const sim = (C.sims && C.sims[t.name]) || (C.valid && C.valid.sims && C.valid.sims[t.name]);
      if (sim) {
        const brs = Array.isArray(sim) ? sim : [{ d: sim.d, f: sim.f, cycle: 1 }];
        brs.forEach(b => series.push({ x: b.d, y: b.f, color: cssv(TEST_COLORS[0]), width: 1.5, dots: true, line: true, r: 1.8, alpha: b.cycle === 0 || !cs.replay.fit_cycles.includes(b.cycle) ? 0.45 : 1 }));
      }
      drawPlot(box.querySelector(`[data-plot="${i}"]`), { series, xlabel: `travel [${u.length}]`, compact: true, empty: t.curve ? '' : 'no csv' });
    });
  }
  function resText(t) {
    const r = C.valid && C.valid.residuals[t.name] !== undefined ? C.valid.residuals[t.name] : C.best && C.best.residuals && C.best.residuals[t.name] !== undefined ? C.best.residuals[t.name] : null;
    return r == null ? '' : ` · ${(100 * r).toFixed(1)} %`;
  }
  function drawHistory() {
    const h = C.history; const cv = $('cal-hist');
    if (!h || !h.length) { drawPlot(cv, { series: [], empty: 'objective per evaluation (fit)', compact: true }); return; }
    const xs = h.map((_, i) => i + 1), ys = h.map(x => x.objective);
    const best = []; let b = Infinity; for (const y of ys) { b = Math.min(b, y); best.push(b); }
    drawPlot(cv, { series: [{ x: xs, y: ys, color: cssv('--chart-2'), dots: true, r: 2, label: 'evaluation' }, { x: xs, y: best, color: cssv('--chart-1'), width: 1.6, label: 'best' }], xlabel: 'evaluation', legend: 'top', compact: true });
  }
  function drawSensitivity() {
    const cv = $('cal-sensplot'); cv.hidden = !C.sens; if (!C.sens) return;
    const cols = ['--chart-1', '--chart-2', '--chart-3', '--chart-4', '--accent', '--warn', '--fg'];
    const series = Object.entries(C.sens).map(([k, rows], i) => {
      const [lo, hi] = C.cs.params[k];
      const x = rows.map(r => (LOG_SCALED.has(k) ? (Math.log(r.value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (r.value - lo) / (hi - lo)));
      return { x, y: rows.map(r => r.objective), color: cssv(cols[i % cols.length]), dots: true, line: true, r: 2, label: k };
    });
    drawPlot(cv, { series, xlabel: 'position in the box (log where marked)', legend: 'top', compact: true });
  }
  function renderBest() {
    const el = $('cal-best'); if (!C.cs) return;
    const lines = [];
    if (C.best) {
      lines.push(`best objective ${C.best.objective.toFixed(4)}` + (C.best.residuals ? '   ' + Object.entries(C.best.residuals).map(([k, v]) => `${k} ${(100 * v).toFixed(1)}%`).join('  ') : ''));
      lines.push(Object.entries(C.best.best).map(([k, v]) => `${k} = ${fmt(v, 4)}${C.synthTruth && C.synthTruth[k] !== undefined ? ` (truth ${fmt(C.synthTruth[k], 4)})` : ''}`).join('\n'));
    }
    if (C.valid) lines.push('validation ' + Object.entries(C.valid.residuals).map(([k, v]) => `${k} ${(100 * v).toFixed(1)}%`).join('  '));
    el.textContent = lines.join('\n') || 'no fit yet';
  }

  // ------------------------------------------------------------------------------------------------ app hooks
  return {
    loadFromFiles,
    applyParameterSetJSON: (d) => applySet(d),
    onScene() { /* a new scene in the view: a calibration on the previous scene stays as it is */ },
    onBackend() { if (C.cs) { C.backend = A.gpuAvailable && A.backend === 'webgpu' ? 'gpu' : 'cpu'; render(); } },
    onSceneEdit() {},
    onResults() {},
    redraw() { if (C.cs) { renderPlots(); drawHistory(); drawSensitivity(); } },
    pickMode: () => C.picking,
    cancelPick() { C.picking = false; $('cal-addtest').classList.remove('on'); setPickBanner(''); },
    onPick(f) {
      const cs = C.cs; if (!cs) return;
      const n = cs.tests.length + cs.validation.length + 1;
      cs.tests.push(new Test(new Probe(f, [0, 0, -1], `P${n}`), null, 1.0, ''));
      C.picking = false; $('cal-addtest').classList.remove('on'); setPickBanner('');
      render(); refreshOverlays();
      status(`test P${n} at (${f}) — set its axis and load its csv`);
    },
    /** the calibration's probes on the model in the view (same topology only) */
    probeMarkers() {
      const cs = C.cs; if (!cs || !A.topo || cs.topo.nKnots !== A.topo.nKnots || cs.topo.nStrands !== A.topo.nStrands) return [];
      return cs.tests.map(t => ({ frame: t.probe.frame, axis: t.probe.axis, label: t.name, color: cssv('--chart-1') }))
        .concat(cs.validation.map(t => ({ frame: t.probe.frame, axis: t.probe.axis, label: t.name, color: cssv('--chart-3') })));
    },
  };
}
