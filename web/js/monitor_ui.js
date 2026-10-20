// The Monitor: the run controls and everything watched while the model solves. A sidebar beside the view that folds
// to a narrow tab (keeping a run / pause button on its rail), docks on either side of the view and widens by its inner
// edge (the chart cards flow into two columns from ~580 px). Groups — Solver, Knots, Stability, Probe · UTM, Extremes,
// Discretisation — fold independently and only open groups are drawn, at most ten times a second during a run.
// Every chart has one anatomy (title, legend with line keys, plot) and the same left inset, so the x axes of a column
// line up; a hover on any chart against the iteration moves one crosshair through all of them (convergence, energy,
// the knot charts, the stability checks) with the values in one tooltip. The run bar carries the convergence meter:
// the residual on a log scale from its first value after the warm-up (empty) to the tolerance (full), one cell per
// decade still to go. Also here: the Setup sidebar's fold and the memory of its open section.
import { drawPlot, fmtNum, watchCanvas } from './plot.js';

/** left inset of every Monitor plot: equal insets align the plots of a column */
export const CHART_L = 42;
const MAX_POINTS = 1400;                                   // per series and draw (min / max per bucket keeps the spikes)

/** min / max decimation of arr[n0 … n) (x = first + index) to ≤ MAX_POINTS points; scale multiplies the values */
function decimate(arr, n0, n, first, scale = 1, positive = false) {
  const len = n - n0;
  if (len <= MAX_POINTS) {
    const x = new Float64Array(len), y = new Float64Array(len);
    for (let i = 0; i < len; i++) { x[i] = first + n0 + i; y[i] = arr[n0 + i] * scale; }
    return { x, y };
  }
  const nb = MAX_POINTS >> 1, per = len / nb, x = new Float64Array(2 * nb), y = new Float64Array(2 * nb);
  let k = 0;
  for (let b = 0; b < nb; b++) {
    const i0 = n0 + Math.floor(b * per), i1 = n0 + Math.min(len, Math.floor((b + 1) * per));
    let lo = Infinity, hi = -Infinity, ilo = i0, ihi = i0;
    for (let i = i0; i < i1; i++) { const v = arr[i]; if (!Number.isFinite(v) || (positive && v <= 0)) continue; if (v < lo) { lo = v; ilo = i; } if (v > hi) { hi = v; ihi = i; } }
    if (lo === Infinity) { x[k] = first + i0; y[k] = NaN; k++; continue; }
    const [ia, ib] = ilo <= ihi ? [ilo, ihi] : [ihi, ilo];
    x[k] = first + ia; y[k] = arr[ia] * scale; k++;
    if (ib !== ia) { x[k] = first + ib; y[k] = arr[ib] * scale; k++; }
  }
  return { x: x.subarray(0, k), y: y.subarray(0, k) };
}

/** ctx: {A, $, qsa, store, cssv, fmt, proto(), hist(), toggleRun(), computeStability()} */
export function initMonitor(ctx) {
  const { A, $, qsa, store, cssv, fmt } = ctx;
  const app = $('app');
  const M = { hover: null, last: 0, timer: 0, geom: {}, knots: null, disc: null, raf: 0, meterKey: '' };
  const closed = () => app.classList.contains('monitor-closed');
  const shown = (id) => !closed() && !!$(id)?.open;

  // --- layout: fold, dock, width -------------------------------------------------------------------------------
  const setOpen = (open) => {
    app.classList.toggle('monitor-closed', !open); store.set('monitorOpen', open);
    $('monitor-toggle').setAttribute('aria-expanded', String(open));
    if (open) requestAnimationFrame(() => redraw());
  };
  const setDock = (left) => { app.classList.toggle('monitor-left', left); store.set('monitorLeft', left); };
  setDock(!!store.get('monitorLeft', false)); setOpen(store.get('monitorOpen', true) !== false);
  $('monitor-toggle').onclick = () => setOpen(closed());
  $('monitor-dock').onclick = () => setDock(!app.classList.contains('monitor-left'));
  $('monitor-mini').onclick = () => ctx.toggleRun();
  const maxW = () => Math.max(300, Math.min(880, window.innerWidth - 460));
  const setWidth = (w, save = true) => {
    w = Math.round(Math.min(Math.max(w, 260), maxW()));
    app.style.setProperty('--monitor-open-w', `${w}px`);
    if (save) store.set('monitorW', w);
  };
  { const w = store.get('monitorW', null); if (w) setWidth(w, false); }
  const rz = $('monitor-resize');
  let drag = null;
  rz.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, w: $('monitor').getBoundingClientRect().width }; rz.classList.add('drag'); try { rz.setPointerCapture(e.pointerId); } catch (_) { /* synthetic */ } e.preventDefault(); });
  rz.addEventListener('pointermove', (e) => { if (!drag) return; const dx = e.clientX - drag.x; setWidth(drag.w + (app.classList.contains('monitor-left') ? dx : -dx)); });
  const endDrag = () => { if (drag) { drag = null; rz.classList.remove('drag'); } };
  rz.addEventListener('pointerup', endDrag); rz.addEventListener('pointercancel', endDrag);
  rz.addEventListener('dblclick', () => { app.style.removeProperty('--monitor-open-w'); store.set('monitorW', null); });
  rz.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const grow = (e.key === 'ArrowLeft') !== app.classList.contains('monitor-left');
    setWidth($('monitor').getBoundingClientRect().width + (grow ? 24 : -24));
  });
  // charts follow the width of the Monitor (drag, dock, window)
  let roPending = false;
  new ResizeObserver(() => { if (roPending) return; roPending = true; requestAnimationFrame(() => { roPending = false; redraw(); }); }).observe($('monitor-body'));

  // --- the Setup sidebar: fold, and the section left open --------------------------------------------------------
  const setSetup = (open) => { app.classList.toggle('setup-closed', !open); store.set('setupOpen', open); $('setup-toggle').setAttribute('aria-expanded', String(open)); };
  setSetup(store.get('setupOpen', true) !== false);
  $('setup-toggle').onclick = () => setSetup(app.classList.contains('setup-closed'));
  const secs = qsa('#sections > details.sec');
  { const saved = store.get('setupSection', null); if (typeof saved === 'string') secs.forEach((d) => { d.open = d.id === saved; }); }
  secs.forEach((d) => d.addEventListener('toggle', () => { if (d.open) store.set('setupSection', d.id); else if (!secs.some(x => x.open)) store.set('setupSection', ''); }));

  // --- groups ----------------------------------------------------------------------------------------------------
  const groups = qsa('#monitor-body > details.mon');
  { const saved = store.get('monitorGroups', null); if (Array.isArray(saved)) groups.forEach((d) => { d.open = saved.includes(d.id); }); }
  groups.forEach((d) => d.addEventListener('toggle', () => { store.set('monitorGroups', groups.filter(g => g.open).map(g => g.id)); if (d.open) requestAnimationFrame(() => redraw()); }));
  /** open a group (and the Monitor) — e.g. Probe · UTM after a replay */
  function openGroup(id) { if (closed()) setOpen(true); const d = $(id); if (d && !d.open) d.open = true; }

  // --- history toggle --------------------------------------------------------------------------------------------
  A.chartFull = !!store.get('chartFull', false);
  const histSeg = () => qsa('#seg-hist button').forEach(b => b.setAttribute('aria-pressed', String((b.dataset.h === 'all') === A.chartFull)));
  qsa('#seg-hist button').forEach(b => { b.onclick = () => { A.chartFull = b.dataset.h === 'all'; store.set('chartFull', A.chartFull); histSeg(); redraw(); }; });
  histSeg();

  // --- shared tooltip and crosshair --------------------------------------------------------------------------------
  const tipEl = $('mon-tip');
  /** rows: [{value, label, color, dash}] — the value leads, the series follows, a line key in the series colour */
  function tipShow(at, title, rows) {
    const parts = [];
    const t = document.createElement('div'); t.className = 't'; t.textContent = title; parts.push(t);
    for (const r of rows) {
      const d = document.createElement('div'); d.className = 'r';
      const k = document.createElement('i'); k.className = 'key'; k.style.setProperty('--c', r.color || cssv('--fg'));
      if (r.dash) k.style.background = `repeating-linear-gradient(90deg, ${r.color} 0 3px, transparent 3px 5px)`;
      const b = document.createElement('b'); b.textContent = r.value;
      const s = document.createElement('span'); s.textContent = r.label;
      d.append(k, b, s); parts.push(d);
    }
    tipEl.replaceChildren(...parts); tipEl.hidden = false;
    const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    let x = at.x + 14, y = at.y - h - 10;
    if (x + w > window.innerWidth - 6) x = at.x - w - 14;
    if (y < 6) y = at.y + 16;
    tipEl.style.left = `${Math.round(Math.max(4, x))}px`; tipEl.style.top = `${Math.round(y)}px`;
  }
  function tipHide() { tipEl.hidden = true; }
  /** the hovered moment: {it, src, at: {x, y}, idx?} or null; redraws the hover layer of every open chart */
  function setHover(h) {
    M.hover = h;
    if (!h) tipHide();
    if (M.raf) return;
    M.raf = requestAnimationFrame(() => { M.raf = 0; drawHoverLayer(); });
  }
  function drawHoverLayer() {
    if (shown('mon-solver')) { drawConv(); drawEnergy(); }
    if (shown('mon-stab')) drawStab();
    if (shown('mon-knots')) M.knots?.redraw();
    if (M.hover && M.hover.src === 'probe' && shown('mon-probe')) drawProbe();
  }
  const fmtV = (v, p = 4) => (Number.isFinite(v) ? fmtNum(+(+v).toPrecision(p)) : '—');
  const fmtE = (v) => (Number.isFinite(v) ? (+v).toExponential(2) : '—');
  /** pointer and keyboard on a chart whose x is the iteration: one crosshair for all of them */
  function bindIterHover(cv, key) {
    watchCanvas(cv, () => redraw());
    cv.addEventListener('pointermove', (e) => {
      const g = M.geom[key]; if (!g || !g.X) return;
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left;
      if (px < g.L - 4 || px > g.L + g.pw + 4) { if (M.hover?.src === key) setHover(null); return; }
      const it = Math.round(g.xmin + Math.min(Math.max(px - g.L, 0), g.pw) / g.pw * (g.xmax - g.xmin));
      setHover({ it, src: key, at: { x: e.clientX, y: e.clientY } });
    });
    cv.addEventListener('pointerleave', () => { if (M.hover?.src === key) setHover(null); });
    cv.addEventListener('keydown', (e) => {
      const g = M.geom[key]; if (!g || !g.X) return;
      if (e.key === 'Escape') { setHover(null); return; }
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const span = g.xmax - g.xmin, step = Math.max(1, Math.round(span / 100)) * (e.shiftKey ? 10 : 1);
      const it0 = M.hover && M.hover.src === key ? M.hover.it : g.xmax;
      const it = Math.min(Math.max(Math.round(it0 + (e.key === 'ArrowRight' ? step : -step)), g.xmin), g.xmax);
      const r = cv.getBoundingClientRect();
      setHover({ it, src: key, at: { x: r.left + g.X(it), y: r.top + 12 } });
    });
    cv.addEventListener('blur', () => { if (M.hover?.src === key) setHover(null); });
  }
  /** a legend of line keys: [{t, c, dash, dot}] */
  function legend(id, items) {
    const el = $(id); if (!el) return;
    const sig = JSON.stringify(items); if (el.__sig === sig) return; el.__sig = sig;
    el.replaceChildren(...items.map((it) => { const s = document.createElement('span'); s.className = 'k' + (it.dash ? ' dash' : '') + (it.dot ? ' dot' : ''); s.style.setProperty('--c', it.c); s.textContent = it.t; return s; }));
  }

  // --- the solver charts -----------------------------------------------------------------------------------------
  /** the x window over the history (last 4000 iterations, widened to keep the latest Zero in view and to reach the end
   *  of a pending settle; or the whole history) with the protocol's phase bands */
  function chartWindow(H) {
    const n = H.residual.length, last = A.model.st.iteration, first = last - n + 1, proto = ctx.proto();
    const pw = { ...(proto ? proto.chart(first, last) : { start: Math.max(first, last - 3999), xmax: last, bands: [], vlines: [] }) };
    if (A.chartFull) pw.start = first;
    const n0 = Math.min(Math.max(0, pw.start - first), Math.max(n - 1, 0));
    const spec = { xmin: first + n0, xmax: Math.max(last, pw.xmax), bands: pw.bands, vlines: pw.vlines };
    A.chartWin = { xmin: spec.xmin, xmax: spec.xmax, bands: spec.bands.map(b => `${b.label} ${b.x0}–${b.x1}`), vlines: spec.vlines.map(v => v.x) };   // console
    return { n0, n, first, spec };
  }
  /** the hovered iteration inside this window → {it, i} (i: index into the history) */
  function hoverIn(w) {
    const h = M.hover; if (!h || h.it == null) return null;
    const it = Math.min(Math.max(h.it, w.first + w.n0), w.first + w.n - 1);
    if (h.it < w.spec.xmin - 1 || h.it > w.spec.xmax + 1) return null;
    return { it, i: it - w.first };
  }
  function drawConv() {
    const cv = $('chart-conv'), m = A.model, H = m?.st.history, sc = A.scene;
    if (!sc) return;
    const cA = cssv('--accent'), cK = cssv('--dim'), cT = cssv('--ok'), tol = +sc.solver.tol;
    legend('lg-conv', [{ t: 'residual', c: cA }, { t: 'kinetic energy / max', c: cK }, { t: `tolerance ${fmtNum(tol)}`, c: cT, dash: true }]);
    if (!H || !H.residual.length) { M.geom.conv = drawPlot(cv, { series: [], logY: true, L: CHART_L, empty: 'no iterations yet — Run or Step' }); return; }
    const w = chartWindow(H);
    let kmax = 0; for (let i = w.n0; i < w.n; i++) if (H.kinetic[i] > kmax) kmax = H.kinetic[i];
    const ks = 1 / (kmax || 1);
    const series = [{ ...decimate(H.kinetic, w.n0, w.n, w.first, ks, true), color: cK, width: 1.4 }, { ...decimate(H.residual, w.n0, w.n, w.first, 1, true), color: cA, width: 2 },
      { x: [w.spec.xmin, w.spec.xmax], y: [tol, tol], color: cT, dash: [4, 3], width: 1 }];
    const hv = hoverIn(w), marks = [];
    if (hv) { marks.push({ x: hv.it, y: H.kinetic[hv.i] * ks, color: cK }, { x: hv.it, y: H.residual[hv.i], color: cA }); }
    M.geom.conv = drawPlot(cv, { series, logY: true, L: CHART_L, empty: 'no iterations yet', ...w.spec, crosshair: hv ? hv.it : null, marks });
    if (hv && M.hover.src === 'conv') tipShow(M.hover.at, `iteration ${hv.it}`, [{ value: fmtE(H.residual[hv.i]), label: 'residual', color: cA }, { value: fmtE(H.kinetic[hv.i] * ks), label: 'kinetic / max', color: cK }]);
  }
  const EPARTS = [['strain', 'total', '--fg', 2], ['bending', 'bending', '--s1'], ['torsion', 'torsion', '--s2'], ['stretch', 'stretch', '--s3'], ['shear', 'shear', '--s4'], ['contact', 'contact', '--s5']];
  function drawEnergy() {
    const cv = $('chart-energy'), m = A.model, H = m?.st.history, sc = A.scene;
    if (!sc) return;
    const parts = EPARTS.filter(([k]) => H && H[k] && H[k].length === H.residual.length);
    const cols = Object.fromEntries(EPARTS.map(([k, , c]) => [k, cssv(c)]));
    legend('lg-energy', (parts.length ? parts : EPARTS).map(([k, t]) => ({ t, c: cols[k] })));
    $('unit-energy').textContent = `[${sc.units.force}·${sc.units.length}]`;
    if (!H || !H.residual.length) { M.geom.energy = drawPlot(cv, { series: [], L: CHART_L, empty: 'no iterations yet' }); return; }
    const w = chartWindow(H);
    const series = parts.slice(1).concat(parts.slice(0, 1)).map(([k, , , wd]) => ({ ...decimate(H[k], w.n0, w.n, w.first), color: cols[k], width: wd || 1.6 }));
    const hv = hoverIn(w);
    const marks = hv ? parts.map(([k]) => ({ x: hv.it, y: H[k][hv.i], color: cols[k] })) : [];
    M.geom.energy = drawPlot(cv, { series, L: CHART_L, empty: 'no iterations yet', ...w.spec, crosshair: hv ? hv.it : null, marks });
    if (hv && M.hover.src === 'energy') tipShow(M.hover.at, `iteration ${hv.it}`, parts.map(([k, t]) => ({ value: fmtV(H[k][hv.i]), label: t, color: cols[k] })));
  }

  // --- stability checks --------------------------------------------------------------------------------------------
  function drawStab() {
    const cv = $('chart-stab'), hist = ctx.hist(), S = hist ? hist.stab : [];
    const ref = S.find(s => Number.isFinite(s.lam) && s.lam !== 0)?.lam;
    const pts = S.filter(s => Number.isFinite(s.lam)).sort((a, b) => a.it - b.it);
    const ys = pts.map(s => (ref ? s.lam / Math.abs(ref) : NaN)), xs = pts.map(s => s.it);
    const cOk = cssv('--s1'), cBad = cssv('--bad');
    const series = pts.length ? [{ x: xs, y: ys, color: cOk, width: 1.6, line: true }] : [];
    if (pts.length) {
      const lo = Math.min(...xs), hi = Math.max(...xs, lo + 1);
      series.push({ x: [lo, hi], y: [0, 0], color: cssv('--dim'), dash: [4, 3], width: 1 });
    }
    let marks = pts.map((s, i) => ({ x: s.it, y: ys[i], color: s.neg ? cBad : cOk }));
    let hv = null;
    if (M.hover && M.hover.it != null && pts.length) {
      let best = 0; for (let i = 1; i < pts.length; i++) if (Math.abs(pts[i].it - M.hover.it) < Math.abs(pts[best].it - M.hover.it)) best = i;
      hv = best;
    }
    M.geom.stab = drawPlot(cv, { series, L: CHART_L, marks, crosshair: M.hover && M.hover.it != null && pts.length ? M.hover.it : null,
      empty: S.length ? 'no finite eigenvalue' : 'no check yet — Check now (or tick after every run)' });
    if (hv != null && M.hover.src === 'stab') {
      const s = pts[hv];
      tipShow(M.hover.at, `check at iteration ${s.it}`, [{ value: fmtV(ys[hv], 3), label: 'λ / |λ₀|', color: s.neg ? cBad : cOk }, { value: s.lam.toExponential(3), label: 'λ softest mode', color: cssv('--dim') }, { value: String(s.neg), label: s.neg === 1 ? 'unstable mode' : 'unstable modes', color: s.neg ? cBad : cssv('--dim') }]);
    }
    const last = S.length ? S.reduce((a, b) => (b.it >= a.it ? b : a)) : null;
    $('stab-text').innerHTML = last ? `latest check, iteration ${last.it}: λ ${last.lam.toExponential(3)} · ${last.neg ? `<span class="warn">${last.neg} unstable mode${last.neg === 1 ? '' : 's'}</span>` : 'stable'} · ${last.ndof} dof, ${(last.ms / 1000).toFixed(1)} s` : '';
    $('sum-mon-stab').textContent = last ? (last.neg ? `${last.neg} unstable` : 'stable') : '';
  }

  // --- probe · UTM ---------------------------------------------------------------------------------------------------
  function drawProbe() {
    const cv = $('chart-probe'), sc = A.scene; if (!sc) return;
    const cU = cssv('--fg'), cR = cssv('--s1'), cP = cssv('--s2'), series = [], keys = [];
    if (A.utm) { series.push({ x: A.utm.displacement, y: A.utm.force, color: cU, dots: true, r: 1.6, name: 'UTM' }); keys.push({ t: 'UTM measured', c: cU, dot: true }); }
    if (A.replay) {
      const br = Array.isArray(A.replay) ? A.replay : [{ d: A.replay.d, f: A.replay.f }];
      br.forEach((b) => series.push({ x: b.d, y: b.f, color: cR, width: 1.8, name: 'replay' }));
      keys.push({ t: 'replay', c: cR });
    }
    if (A.probeTrace && A.probeTrace.length) { series.push({ x: A.probeTrace.map(p => p[0]), y: A.probeTrace.map(p => p[1]), color: cP, width: 1.8, name: 'live probe' }); keys.push({ t: 'live probe', c: cP }); }
    legend('lg-probe', keys);
    $('unit-probe').textContent = `[${sc.units.force}] · [${sc.units.length}]`;
    let marks = [];
    const h = M.hover && M.hover.src === 'probe' ? M.hover : null;
    if (h && h.pt) marks = [{ x: h.pt.x, y: h.pt.y, color: h.pt.color }];
    M.geom.probe = drawPlot(cv, { series, L: CHART_L, xlabel: `crosshead travel [${sc.units.length}]`, marks, empty: 'no UTM curve or probe yet — 9 UTM test' });
    M.probeSeries = series;
    if (h && h.pt) tipShow(h.at, h.pt.name, [{ value: fmtV(h.pt.y), label: `force [${sc.units.force}]`, color: h.pt.color }, { value: fmtV(h.pt.x), label: `travel [${sc.units.length}]`, color: cssv('--dim') }]);
    const pr = ctx.probe();
    $('sum-mon-probe').textContent = pr ? `probe (${pr.frame[0]}, ${pr.frame[1]})` : '';
  }
  {
    const cv = $('chart-probe');
    watchCanvas(cv, () => drawProbe());
    cv.addEventListener('pointermove', (e) => {
      const g = M.geom.probe, S = M.probeSeries; if (!g || !g.X || !S || !S.length) return;
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      let best = null, bd = 24 * 24;                                          // a 24 px hit radius
      for (const s of S) for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i], y = s.y[i]; if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const d = (g.X(x) - px) ** 2 + (g.Y(y) - py) ** 2; if (d < bd) { bd = d; best = { x, y, color: s.color, name: s.name }; }
      }
      if (best) setHover({ it: null, src: 'probe', at: { x: e.clientX, y: e.clientY }, pt: best }); else if (M.hover?.src === 'probe') setHover(null);
    });
    cv.addEventListener('pointerleave', () => { if (M.hover?.src === 'probe') { setHover(null); drawProbe(); } });
  }

  // --- the run bar: convergence meter and the run line ---------------------------------------------------------------
  function updateRun() {
    const m = A.model, st = m?.st, sc = A.scene, el = $('run-text'), meter = $('conv-meter');
    const mini = $('monitor-mini'); mini.classList.toggle('running', !!A.running); mini.setAttribute('aria-label', A.running ? 'Pause' : 'Run');
    if (!st || !sc) { el.textContent = 'no model'; meter.className = 'meter idle'; return; }
    const sp = sc.solver, tol = +sp.tol, r = st.residual, H = st.history;
    let r0 = NaN;
    for (let i = 0; i < H.residual.length; i++) if (Number.isFinite(H.residual[i]) && H.residual[i] > 0) { r0 = H.residual[i]; break; }
    const warm = !Number.isFinite(r);
    let frac = 0, togo = NaN, D = NaN;
    if (!warm && Number.isFinite(r0) && tol > 0) {
      const top = Math.max(r0, r);
      D = Math.log10(top / tol); togo = Math.max(0, Math.log10(Math.max(r, 1e-300) / tol));
      frac = D > 0 ? Math.min(1, Math.max(0, 1 - togo / D)) : (r <= tol ? 1 : 0);
    }
    const ok = !warm && r <= tol;
    meter.className = 'meter' + (ok ? ' ok' : '') + (warm || !Number.isFinite(D) ? ' idle' : '');
    meter.querySelector('.fill').style.width = `${(100 * (ok ? 1 : frac)).toFixed(1)}%`;
    meter.setAttribute('aria-valuenow', String(Math.round(100 * (ok ? 1 : frac))));
    const key = Number.isFinite(D) ? D.toFixed(2) : '';
    if (key !== M.meterKey) {
      M.meterKey = key;
      const ticks = [];
      // a gap j decades above the tolerance (j = 1, 2, …): the cells right of the fill are the decades still to go
      if (Number.isFinite(D) && D > 1) for (let j = 1; j < D - 1e-6; j++) { const i = document.createElement('i'); i.style.left = `${(100 * (1 - j / D)).toFixed(2)}%`; ticks.push(i); }
      meter.querySelector('.ticks').replaceChildren(...ticks);
    }
    const state = m.converged ? ' · <b>converged</b>' : warm ? ` · warm-up ${Math.min(st.iteration, sp.warmup_iter | 0)} / ${sp.warmup_iter | 0}` : Number.isFinite(togo) ? ` · ${togo.toFixed(1)} decade${Math.abs(togo - 1) < 0.05 ? '' : 's'} to go` : '';
    let html = `it <b>${st.iteration}</b> · residual <b>${warm ? '—' : r.toExponential(2)}</b> (tol ${fmt(tol, 3)})${state}`;
    const pl = ctx.proto() ? ctx.proto().line() : '';
    if (pl) html += `<br>${pl}`;
    el.innerHTML = html;
    $('sum-mon-solver').textContent = warm ? 'warm-up' : m.converged ? 'converged' : `residual ${r.toExponential(1)}`;
  }

  // --- drawing ---------------------------------------------------------------------------------------------------
  /** everything open, now */
  function redraw() {
    updateRun();
    if (shown('mon-solver')) { drawConv(); drawEnergy(); }
    if (shown('mon-stab')) drawStab();
    if (shown('mon-probe')) drawProbe();
    if (shown('mon-knots')) M.knots?.redraw(true);
    if (shown('mon-disc')) M.disc?.redraw();
  }
  /** the live charts during a run, at most every 100 ms (the run bar every call) */
  function schedule() {
    const now = performance.now(), dt = now - M.last;
    if (dt >= 100) { M.last = now; live(); return; }
    updateRun();
    if (!M.timer) M.timer = setTimeout(() => { M.timer = 0; M.last = performance.now(); live(); }, 100 - dt);
  }
  function live() {
    updateRun();
    if (shown('mon-solver')) { drawConv(); drawEnergy(); }
    if (shown('mon-probe')) drawProbe();
    if (shown('mon-knots')) M.knots?.redraw();
  }
  bindIterHover($('chart-conv'), 'conv'); bindIterHover($('chart-energy'), 'energy'); bindIterHover($('chart-stab'), 'stab');

  // stability controls
  $('btn-stab').onclick = () => { openGroup('mon-stab'); ctx.computeStability(); };
  const auto = $('ck-stabauto'); auto.checked = !!store.get('stabAuto', false); auto.onchange = () => store.set('stabAuto', auto.checked);

  /** the iteration window of the solver charts (last 4000 / all, protocol-aware) — the knot charts against the
   *  iteration show the same window, so the columns line up in time too */
  function iterWindow() { const H = A.model?.st.history; if (!H || !H.residual.length) return null; const w = chartWindow(H); return { xmin: w.spec.xmin, xmax: w.spec.xmax }; }

  return {
    redraw, schedule, updateRun, drawProbe, drawStab, shown, openGroup, setHover, hover: () => M.hover, tipShow, tipHide, iterWindow,
    setKnots(k) { M.knots = k; }, setDisc(d) { M.disc = d; }, autoStability: () => auto.checked,
    resetHover() { M.hover = null; tipHide(); },
  };
}
