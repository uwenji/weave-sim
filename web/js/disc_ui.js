// Discretisation (joint placement) in the app: the Monitor's Discretisation group (Run / Pause, Step, Clear joints, the
// annealing plot, the objective components and the piece lengths) and section 8 (stock / joint length, extra joints,
// annealing parameters, objective weights with the structural layer, joints csv). Runs time-sliced in the animation
// frame; every published placement goes to scene.joints (saved with the scene, exported to Rhino) and to the view
// (cuts, pieces colouring, joint-zone diagrams).
import { Discretizer, geometryFromModel, discretizeParams, OBJECTIVES, OBJECTIVE_LABELS, OBJECTIVE_SHORT, STRUCTURAL_LAYERS, jointPoints, jointsCSV } from './discretize.js';
import { drawPlot, watchCanvas } from './plot.js';
import { CHART_L } from './monitor_ui.js';
import { fmtG } from './fields.js';

export function initDiscretize(ctx) {
  const { A, $, qsa, status, cssv, store, download } = ctx;
  A.disc = null; A.discRunning = false; A.discResult = null;
  let lastPublish = 0;
  const P = () => (A.scene.discretize = discretizeParams(A.scene.discretize));

  const shown = () => ctx.shown('mon-disc');
  ['disc-chart', 'disc-comp', 'disc-pieces'].forEach(id => watchCanvas($(id), () => { drawChart(); drawPlots(); }));

  // --- parameters and weights ---------------------------------------------------------------------------------------
  const WEIGHT_LABEL = { sum_curve: 'Σ κ at joints', max_curve: 'max κ at joints', min_dist: 'joint spacing', uniformity_range: 'longest − shortest', uniformity_sum: 'Σ |piece − mean|', structural: 'structural', clearance: 'clear of crossings' };
  $('disc-weights').innerHTML = OBJECTIVES.map(k => `<div class="row slider"><label title="${OBJECTIVE_LABELS[k]}">${WEIGHT_LABEL[k]}</label><input type="range" min="0" max="1" step="0.01" data-w="${k}"><output class="mono"></output></div>` +
    (k === 'structural' ? `<div class="row"><label class="w" for="sel-dlayer">structural layer</label><select id="sel-dlayer">${STRUCTURAL_LAYERS.map(l => `<option>${l}</option>`).join('')}</select><span class="readout inline warn" id="ro-dneeds"></span></div>` : '')).join('');
  function sync() {
    if (!A.scene) return;
    const p = P();
    qsa('[data-disc]').forEach(el => { const v = p[el.dataset.disc]; el.value = typeof v === 'number' ? +v.toPrecision(6) : v; });
    qsa('[data-w]').forEach(el => { el.value = p.weights[el.dataset.w] ?? 0; el.parentElement.querySelector('output').textContent = (+el.value).toFixed(2); });
    $('sel-dlayer').value = p.structural_layer;
    readouts(); drawPlots();
  }
  qsa('[data-disc]').forEach(el => el.addEventListener('change', () => {
    const p = P(), k = el.dataset.disc; let v = +el.value; if (!Number.isFinite(v)) return; if ('int' in el.dataset) v = Math.round(v);
    p[k] = v;
    if (['max_length', 'joint_length', 'extra_joints'].includes(k)) { A.disc = null; A.discRunning = false; runButton(); }      // geometry / count changed: the next Run starts fresh
    else if (A.disc && ['iterations', 'mll', 'cooling'].includes(k)) A.disc.p[k] = v;                                         // the running annealer follows (Python shares the params)
    readouts();
  }));
  const reweight = () => { if (A.disc) { A.disc.reweight(P()); A.discResult = A.disc.result(); } readouts(); drawPlots(); };
  qsa('[data-w]').forEach(el => el.addEventListener('input', () => { P().weights[el.dataset.w] = +el.value; el.parentElement.querySelector('output').textContent = (+el.value).toFixed(2); }));
  qsa('[data-w]').forEach(el => el.addEventListener('change', reweight));
  $('sel-dlayer').onchange = (e) => { P().structural_layer = e.target.value; reweight(); };

  // --- the annealer -------------------------------------------------------------------------------------------------
  function start() {
    if (!A.model) return null;
    const p = P();
    const geo = geometryFromModel(A.topo, A.model.st, A.model.st.X, A.model.st.iteration > 0 ? A.results : null, p.structural_layer);
    let width = 0; for (let g = 1; g <= A.scene.nGroups; g++) width = Math.max(width, +A.scene.group(g).section.width);
    const d = new Discretizer(geo, p, width);
    A.discResult = null;
    if (!d.geo.length) { status(`no strand is longer than the stock length ${fmtG(p.max_length, 4)} ${A.scene.units.length} — nothing to cut`, 'warn'); A.disc = null; return null; }
    A.disc = d; publish();
    status(`discretisation: ${d.nJoints.reduce((a, b) => a + b, 0)} joints on ${d.geo.length} strands`);
    return d;
  }
  function publish() {
    const d = A.disc; if (!d) return;
    A.discResult = d.result();
    A.scene.joints = Object.fromEntries(Object.entries(A.discResult.joints).map(([s, v]) => [String(s), v]));
    lastPublish = performance.now();
    ctx.onJoints(); readouts(); drawChart(); drawPlots();
  }
  function steps(n) {
    if (!A.disc && !start()) return;
    A.disc.step(n); publish();
    if (A.disc.done) { A.discRunning = false; runButton(); }
  }
  function reset() {
    A.disc = null; A.discResult = null; A.discRunning = false; runButton();
    A.scene.joints = {}; ctx.onJoints(); readouts(); drawChart(); drawPlots();
  }
  /** one animation frame of a running annealing (~12 ms of temperature steps) */
  function tick() {
    if (!A.discRunning || !A.disc) return;
    const t0 = performance.now();
    while (performance.now() - t0 < 12 && !A.disc.done) A.disc.step(1);
    if (A.disc.done || performance.now() - lastPublish > 150) publish();
    if (A.disc.done) { A.discRunning = false; runButton(); status(`discretisation done: objective ${A.disc.historyBest[0].toFixed(3)} → ${A.disc.ebest.toFixed(3)}`); }
  }
  function runButton() { const b = $('btn-disc-run'); b.textContent = A.discRunning ? 'Pause' : 'Run'; b.classList.toggle('running', A.discRunning); if (A.scene) readouts(); }
  $('btn-disc-run').onclick = () => {
    if (!A.discRunning && !A.disc && !start()) return;
    if (!A.discRunning && A.disc && A.disc.done) { status('the annealing is done — Reset to start again (or raise the temperature steps)'); return; }
    A.discRunning = !A.discRunning && !!A.disc; runButton();
  };
  $('btn-disc-step').onclick = () => steps(Math.max(1, +$('nb-disc-step').value | 0));
  $('btn-disc-reset').onclick = reset;
  $('btn-joints-csv').onclick = () => { if (!A.model) return; download('joints.csv', jointsCSV(jointPoints(A.scene, A.model.st)), 'text/csv'); };

  // --- read-outs and plots ------------------------------------------------------------------------------------------
  function readouts() {
    if (!A.scene) return;
    const p = P(), d = A.disc, u = A.scene.units.length;
    let txt;
    if (d) {
      txt = `${d.nJoints.reduce((a, b) => a + b, 0)} joints · ${d.geo.length} strands cut · step ${d.iteration}/${p.iterations} · T ${d.T.toExponential(2)} · ${d.nEval} evaluations`;
      if (d.done) txt += `\ndone: objective ${d.historyBest[0].toFixed(3)} → ${d.ebest.toFixed(3)}`;
    } else txt = `cuts the laths into ${fmtG(p.max_length, 4)} ${u} pieces (joint ${fmtG(p.joint_length, 3)} ${u}) on the current geometry${A.model && A.model.st.iteration > 0 ? ' (solved)' : ''}`;
    $('disc-text').textContent = txt;
    const J = A.scene.joints || {}, nj = Object.values(J).reduce((a, v) => a + (v ? v.length : 0), 0), ns = Object.values(J).filter(v => v && v.length).length;
    $('ro-joints').textContent = nj ? `${nj} joints on ${ns} strands` : 'none';
    $('btn-joints-csv').disabled = !nj;
    $('sum-disc').textContent = nj ? `${nj} joints` : ''; $('sum-mon-disc').textContent = A.discRunning ? 'running' : nj ? `${nj} joints` : '';
    $('ro-dneeds').textContent = (p.weights.structural || 0) > 0 && !(A.results && A.model && A.model.st.iteration > 0) ? 'needs a solve' : '';
  }
  function drawChart() {
    if (!shown()) return;
    const d = A.disc, series = [], cB = cssv('--accent'), cC = cssv('--dim');
    const lg = $('disc-legend');
    if (!lg.childElementCount) lg.innerHTML = `<span class="k" style="--c:${cB}">best</span><span class="k" style="--c:${cC}">current</span>`;
    else { lg.children[0].style.setProperty('--c', cB); lg.children[1].style.setProperty('--c', cC); }
    if (d && d.historyBest.length) {
      const xs = d.historyBest.map((_, i) => i);
      series.push({ x: xs, y: d.historyCurrent, color: cC, width: 1.2 });
      series.push({ x: xs, y: d.historyBest, color: cB, width: 2 });
    }
    drawPlot($('disc-chart'), { series, L: CHART_L, empty: 'no annealing yet — Run or Step' });
  }
  function drawPlots() {
    if (!shown()) return;
    const r = A.discResult, names = r ? OBJECTIVES.filter(n => n in r.normalised_final) : [];
    const xs = names.map((_, i) => i);
    drawPlot($('disc-comp'), {
      series: names.length ? [{ x: xs.map(x => x - 0.2), y: names.map(n => r.normalised_initial[n] ?? 0), bars: true, barWidth: 0.38, color: cssv('--dim'), label: 'initial' },
        { x: xs.map(x => x + 0.2), y: names.map(n => r.normalised_final[n] ?? 0), bars: true, barWidth: 0.38, color: cssv('--accent'), label: 'final' }] : [],
      xticks: names.map((n, i) => ({ x: i, label: OBJECTIVE_SHORT[n] })), legend: 'top', ymin: 0, L: CHART_L, empty: 'objective components after a Run',
    });
    const d = A.disc, lens = d ? d.pieces(d.xbest).flatMap(a => [...a]) : [];
    let series = [];
    if (lens.length) {
      const lo = Math.min(...lens), hi = Math.max(...lens), nb = 12, w = (hi - lo) / nb || 1, counts = new Array(nb).fill(0);
      for (const v of lens) counts[Math.min(nb - 1, Math.floor((v - lo) / w))]++;
      series = [{ x: counts.map((_, i) => lo + (i + 0.5) * w), y: counts, bars: true, barWidth: w, color: cssv('--s1') }];
    }
    drawPlot($('disc-pieces'), { series, xlabel: `piece length [${A.scene ? A.scene.units.length : ''}]`, ymin: 0, L: CHART_L, empty: 'piece lengths after a Run' });
  }

  return {
    onScene() { A.disc = null; A.discResult = null; A.discRunning = false; runButton(); sync(); drawChart(); },
    onResults() { readouts(); },
    tick, sync, redraw() { drawChart(); drawPlots(); }, running: () => A.discRunning,
  };
}
