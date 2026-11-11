// The Monitor's Knots group: follow chosen crossings through the solve and plot what an engineer checks at a joint —
// the friction-coefficient evolution (mobilised μ = Ft / Fn against the friction coefficient), friction use, the
// contact forces, slip and sliding along each lath, the crossing geometry (gap, roll, scissor angle), the lath stress
// next to the crossing and the embedment utilisation. Small multiples: one chart per quantity (up to six), all
// followed crossings in every chart, one colour per crossing everywhere (the chips that are the legend of all charts,
// the lines, the markers in the view), one shared x — the iteration (crosshair shared with the solver charts), the
// probe travel, or a quantity of the crossing itself (Fn: with Ft, the friction cone). Values: KnotHistory.
import { drawPlot, fmtNum, watchCanvas } from './plot.js';
import { fkey, overFrame, underFrame } from './topology.js';
import { CHART_L } from './monitor_ui.js';

export const MAX_KNOTS = 8, MAX_CHARTS = 6;
const SLOTS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const DEG = 180 / Math.PI;
const DEFAULT_CHARTS = ['mu', 'Fn', 'slip'];
const ADD_ORDER = ['mu', 'Fn', 'slip', 'disp', 'sigma', 'util', 'Ft', 'futil', 'gap', 'slideO', 'slideU', 'rollO', 'rollU', 'dtheta', 'pressure', 'ux', 'uy', 'uz', 'state'];

/** the quantities: title (the card's selector), unit, values over the records for crossing c, reference lines */
function channels(A) {
  const u = () => A.scene.units;
  const fnTiny = (S, c) => { const fn = S.channel(c, 'Fn'); let mx = 0; for (const v of fn) mx = Math.max(mx, Math.abs(v)); return 1e-6 * Math.max(mx, 1e-300); };
  const rel = (a) => { const o = Float64Array.from(a); const a0 = a.find(Number.isFinite) ?? 0; for (let i = 0; i < o.length; i++) o[i] -= a0; return o; };
  const side = (S, c, over) => (S.overSide[c] === (over ? 0 : 1) ? 'A' : 'B');
  return {
    mu: { title: 'Mobilised μ = Ft / Fn', unit: () => '–',
      get: (S, c) => { const fn = S.channel(c, 'Fn'), ft = S.channel(c, 'Ft'), tiny = fnTiny(S, c); return fn.map((v, i) => (v > tiny ? ft[i] / v : NaN)); },
      refs: () => [{ y: +A.scene.contact.mu, label: `μ ${fmtNum(+A.scene.contact.mu)}` }] },
    futil: { title: 'Friction use Ft / μFn', unit: () => '–',
      get: (S, c) => { const fn = S.channel(c, 'Fn'), ft = S.channel(c, 'Ft'), mu = Math.max(+A.scene.contact.mu, 1e-12), tiny = fnTiny(S, c); return fn.map((v, i) => (v > tiny ? ft[i] / (mu * v) : NaN)); },
      refs: () => [{ y: 1, label: 'slip at 1' }] },
    Fn: { title: 'Normal force Fn', unit: () => u().force, get: (S, c) => S.channel(c, 'Fn') },
    Ft: { title: 'Friction force Ft', unit: () => u().force, get: (S, c) => S.channel(c, 'Ft') },
    slip: { title: 'Slip, accumulated', unit: () => u().length, get: (S, c) => S.channel(c, 'slip') },
    slideO: { title: 'Sliding along the over lath', unit: () => u().length, get: (S, c) => rel(S.channel(c, 'arc' + side(S, c, true))) },
    slideU: { title: 'Sliding along the under lath', unit: () => u().length, get: (S, c) => rel(S.channel(c, 'arc' + side(S, c, false))) },
    disp: { title: 'Displacement |u|', unit: () => u().length,
      get: (S, c) => { const x = S.channel(c, 'ux'), y = S.channel(c, 'uy'), z = S.channel(c, 'uz'); return x.map((v, i) => Math.hypot(v, y[i], z[i])); } },
    ux: { title: 'Displacement u_x', unit: () => u().length, get: (S, c) => S.channel(c, 'ux') },
    uy: { title: 'Displacement u_y', unit: () => u().length, get: (S, c) => S.channel(c, 'uy') },
    uz: { title: 'Displacement u_z', unit: () => u().length, get: (S, c) => S.channel(c, 'uz') },
    gap: { title: 'Gap (centre distance)', unit: () => u().length, get: (S, c) => S.channel(c, 'gap') },
    sigma: { title: 'Lath stress σ at the crossing', unit: () => u().stressLabel, get: (S, c) => S.channel(c, 'sigma') },
    util: { title: 'Embedment p / fc90', unit: () => '–', get: (S, c) => S.channel(c, 'util'), refs: () => [{ y: 1, label: 'crushing at 1' }] },
    pressure: { title: 'Contact pressure p', unit: () => u().stressLabel, get: (S, c) => S.channel(c, 'pressure') },
    rollO: { title: 'Section roll, over lath', unit: () => '°', get: (S, c) => S.channel(c, 'phi' + side(S, c, true)).map(v => v * DEG) },
    rollU: { title: 'Section roll, under lath', unit: () => '°', get: (S, c) => S.channel(c, 'phi' + side(S, c, false)).map(v => v * DEG) },
    dtheta: { title: 'Scissor rotation Δθ', unit: () => '°', get: (S, c) => rel(S.channel(c, 'theta')).map(v => v * DEG) },
    state: { title: 'Stick 0 · slip 1', unit: () => '–', get: (S, c) => S.channel(c, 'state') },
    lambda: { title: 'AL multiplier λ', unit: () => u().force, get: (S, c) => S.channel(c, 'lambda'), needs: () => !!A.scene.contact.augmented, off: 'augmented-Lagrangian gap' },
    dent: { title: 'Seat depth (dent)', unit: () => u().length, get: (S, c) => S.channel(c, 'dent'), needs: () => A.scene.contact.seat_law !== 'off', off: 'seat law' },
    M_rot: { title: 'Scissor moment M_rot', unit: () => `${u().force}·${u().length}`, get: (S, c) => S.channel(c, 'M_rot'), needs: () => !!A.scene.contact.rot_friction, off: 'rotational friction' },
  };
}
const X_AXES = { it: 'iteration', probe: 'probe travel', disp: '|u| of the crossing', Fn: 'Fn (friction cone with Ft)', slip: 'slip' };

/** ctx: {A, $, qsa, status, cssv, store, hist(), mon, refreshOverlays, setPickBanner, probe()} */
export function initKnots(ctx) {
  const { A, $, status, cssv, store, mon } = ctx;
  const CHS = channels(A);
  const K = { knots: [], picking: false, charts: [], geom: [], data: [] };     // knots: [{edge, c, slot}]
  const colorOf = (k) => cssv(SLOTS[k.slot]);
  const hist = () => ctx.hist();

  // --- controls ---------------------------------------------------------------------------------------------------
  const xsel = $('sel-knotx');
  xsel.innerHTML = Object.entries(X_AXES).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  xsel.value = store.get('knotX', 'it'); if (!X_AXES[xsel.value]) xsel.value = 'it';
  xsel.onchange = () => { store.set('knotX', xsel.value); mon.resetHover(); redraw(true); };
  K.charts = (store.get('knotCharts', DEFAULT_CHARTS) || []).filter(k => CHS[k]).slice(0, MAX_CHARTS);
  $('btn-knotpick').onclick = () => setPicking(!K.picking);
  $('btn-knotsel').onclick = () => { addFrames(A.selection); };
  $('btn-knotclear').onclick = () => { K.knots = []; changed(); };
  $('btn-knotadd').onclick = () => {
    if (K.charts.length >= MAX_CHARTS) return;
    K.charts.push(ADD_ORDER.find(k => !K.charts.includes(k)) || 'Ft'); saveCharts(); renderCards();
  };
  $('knot-chips').addEventListener('click', (e) => { const b = e.target.closest('[data-rm]'); if (!b) return; K.knots = K.knots.filter(k => k.edge !== +b.dataset.rm); changed(); });

  function setPicking(on) {
    K.picking = on; $('btn-knotpick').classList.toggle('on', on);
    if (on) mon.openGroup('mon-knots');
    ctx.setPickBanner(on ? `KNOTS — click crossings to follow them (up to ${MAX_KNOTS}), click again to remove · Esc leaves` : '');
  }
  const edgeIndex = () => { const st = A.model?.st; if (!st) return new Map(); if (K._ei && K._ei.st === st) return K._ei.map; const map = new Map(); st.knotEdges.forEach((e, c) => map.set(e, c)); K._ei = { st, map }; return map; };
  function toggleKnot(c) {
    const st = A.model.st, e = st.knotEdges[c], i = K.knots.findIndex(k => k.edge === e);
    if (i >= 0) { K.knots.splice(i, 1); changed(); return; }
    if (K.knots.length >= MAX_KNOTS) { status(`up to ${MAX_KNOTS} crossings can be followed — remove one first`, 'warn'); return; }
    const used = new Set(K.knots.map(k => k.slot)); let slot = 0; while (used.has(slot)) slot++;
    K.knots.push({ edge: e, c, slot }); changed();
  }
  function addFrames(frames) {
    if (!frames.length) { status('select frames first (click laths next to crossings), or use Pick knots', 'warn'); return; }
    const ei = edgeIndex();
    for (const f of frames) { const e = A.topo.frameToKnot.get(fkey(f[0], f[1])); const c = ei.get(e); if (c !== undefined && !K.knots.some(k => k.edge === e)) toggleKnot(c); }
  }
  /** a pick in the view while picking: a crossing sphere, or a lath near a crossing */
  function onPick(p, frameAtPick) {
    if (!p) return;
    let c = -1;
    if (p.kind === 'knot') c = p.c;
    else if (p.kind === 'lath') { const f = frameAtPick(p); if (f) { const e = A.topo.frameToKnot.get(fkey(f[0], f[1])); c = edgeIndex().get(e) ?? -1; } }
    if (c >= 0) toggleKnot(c);
  }
  function changed() {
    store.set('knots:' + (A.sceneName || ''), K.knots.map(k => [k.edge, k.slot]));
    renderChips(); redraw(true); ctx.refreshOverlays();
  }
  /** after a new model: keep the crossings that exist in it (by mesh edge), remembered per scene */
  function onModel() {
    K._ei = null;
    const saved = K.knots.length ? K.knots.map(k => [k.edge, k.slot]) : store.get('knots:' + (A.sceneName || ''), []);
    const map = edgeIndex();
    K.knots = saved.filter(([e]) => map.has(e)).slice(0, MAX_KNOTS).map(([e, slot]) => ({ edge: e, c: map.get(e), slot }));
    renderChips(); redraw(true);
  }
  function renderChips() {
    const el = $('knot-chips');
    $('sum-mon-knots').textContent = K.knots.length ? `${K.knots.length} followed` : '';
    if (!K.knots.length) { const s = document.createElement('span'); s.className = 'empty'; s.textContent = 'No crossings followed — Pick knots, then click crossings in the view.'; el.replaceChildren(s); return; }
    el.replaceChildren(...K.knots.map((k) => {
      const kn = A.topo.knots.get(k.edge), o = overFrame(kn), un = underFrame(kn);
      const chip = document.createElement('span'); chip.className = 'kchip'; chip.title = `crossing ${k.edge}: over lath ${o[0]}, under lath ${un[0]}`;
      const key = document.createElement('i'); key.className = 'key'; key.style.setProperty('--c', colorOf(k));
      const b = document.createElement('button'); b.dataset.rm = String(k.edge); b.setAttribute('aria-label', `stop following crossing ${k.edge}`); b.textContent = '×';
      chip.append(key, document.createTextNode(`c${k.edge}`), b);
      return chip;
    }));
  }

  // --- the cards -----------------------------------------------------------------------------------------------------
  const saveCharts = () => store.set('knotCharts', K.charts);
  function renderCards() {
    const box = $('knot-cards');
    const opts = Object.entries(CHS).map(([k, v]) => `<option value="${k}">${v.title}</option>`).join('');
    box.replaceChildren(...K.charts.map((key, i) => {
      const fig = document.createElement('figure'); fig.className = 'card'; fig.dataset.i = String(i);
      fig.innerHTML = `<figcaption class="card-head"><select class="card-y" aria-label="quantity of chart ${i + 1}">${opts}</select><span class="card-unit"></span><span class="spacer"></span><span class="lgd ref"></span><button type="button" class="card-x" title="remove this chart" aria-label="remove chart ${i + 1}">×</button></figcaption><canvas class="cv" tabindex="0"></canvas>`;
      const sel = fig.querySelector('select'); sel.value = key;
      sel.onchange = () => { K.charts[i] = sel.value; saveCharts(); redraw(true); };
      fig.querySelector('.card-x').onclick = () => { K.charts.splice(i, 1); saveCharts(); mon.resetHover(); renderCards(); };
      const cv = fig.querySelector('canvas');
      cv.setAttribute('aria-label', `knot chart ${i + 1}`);
      bindHover(cv, i);
      watchCanvas(cv, () => redraw(true));
      return fig;
    }));
    $('btn-knotadd').disabled = K.charts.length >= MAX_CHARTS;
    $('knot-count').textContent = `${K.charts.length}/${MAX_CHARTS}`;
    redraw(true);
  }

  // --- data ----------------------------------------------------------------------------------------------------------
  function xValues(S, c, xk) {
    if (xk === 'it') return S.iterations();
    if (xk === 'probe') { const p = S.probeTravel(); return p.some(Number.isFinite) ? p : null; }
    return CHS[xk === 'disp' ? 'disp' : xk].get(S, c);
  }
  const xUnit = (xk) => (xk === 'it' ? '' : xk === 'Fn' ? A.scene.units.force : A.scene.units.length);

  /** the record index the shared hover points at (a hover on a knot chart carries it; elsewhere: the iteration) */
  function hoverIndex(S) {
    const h = mon.hover(); if (!h || !S.length) return -1;
    if (h.idx != null && h.src?.startsWith('k')) return h.idx;
    return h.it != null ? S.nearest(h.it) : -1;
  }

  // --- plot ----------------------------------------------------------------------------------------------------------
  let lastDraw = 0;
  function redraw(force = false) {
    if (!mon.shown('mon-knots')) return;
    const S = hist(), ver = S ? S.version : -1, h = mon.hover();
    const sig = `${ver}|${h ? `${h.it}|${h.idx}|${h.src}` : ''}`;
    if (!force && sig === K.sig && performance.now() - lastDraw < 1000) return;
    K.sig = sig; lastDraw = performance.now();
    const box = $('knot-cards'); box.hidden = !K.knots.length;              // no crossings followed: the chips say what to do
    const cards = K.knots.length ? [...box.children] : [];
    cards.forEach((fig, i) => drawCard(fig, i, S));
    text(S);
  }
  function drawCard(fig, i, S) {
    const key = K.charts[i], chn = CHS[key], cv = fig.querySelector('canvas'), xk = xsel.value;
    fig.querySelector('.card-unit').textContent = chn && A.scene ? `[${chn.unit()}]` : '';
    const refEl = fig.querySelector('.lgd.ref');
    const empty = (msg) => { K.geom[i] = drawPlot(cv, { series: [], L: CHART_L, empty: msg }); K.data[i] = []; refEl.replaceChildren(); };
    if (!chn || !A.scene) return empty('');
    if (!A.model || !S || !S.length) return empty(A.model && !A.model.contact ? 'no crossings in this model' : 'no solve yet — Run or Step');
    if (chn.needs && !chn.needs()) return empty(`not computed — switch on the ${chn.off} (4 Contact)`);
    if (!K.knots.length) return empty('Pick knots, then click crossings in the view');
    if (xk === 'probe' && !ctx.probe()) return empty('no probe — pick one in 9 UTM test');
    const data = [];
    let xmin = Infinity, xmax = -Infinity;
    // against the iteration: the window of the solver charts (records inside it, one before for continuity)
    const iw = xk === 'it' ? mon.iterWindow() : null;
    let r0 = 0;
    if (iw) { while (r0 < S.length - 1 && S.recs[r0 + 1].it <= iw.xmin) r0++; }
    for (const k of K.knots) {
      const x = xValues(S, k.c, xk); if (!x) continue;
      const y = chn.get(S, k.c);
      data.push({ k, x, y, color: colorOf(k), r0 });
      for (const v of x) if (Number.isFinite(v)) { if (v < xmin) xmin = v; if (v > xmax) xmax = v; }
    }
    if (!data.length) return empty(xk === 'probe' ? 'no probe travel recorded yet — Run with the probe set' : 'no values yet');
    const series = data.map(d => ({ x: d.r0 ? d.x.subarray(d.r0) : d.x, y: d.r0 ? d.y.subarray(d.r0) : d.y, color: d.color, width: 1.8, line: true, dots: xk !== 'it' && d.x.length < 300, r: 1.6 }));
    const span = Number.isFinite(xmin) ? [xmin, xmax > xmin ? xmax : xmin + 1] : [0, 1];
    const refs = chn.refs ? chn.refs() : [];
    for (const r of refs) series.push({ x: span, y: [r.y, r.y], color: cssv('--dim'), dash: [4, 3], width: 1 });
    if (key === 'Ft' && xk === 'Fn') { const mu = +A.scene.contact.mu; series.push({ x: [0, span[1]], y: [0, mu * span[1]], color: cssv('--dim'), dash: [4, 3], width: 1 }); refs.push({ label: `Ft = μFn` }); }
    const legendSig = refs.map(r => r.label).join('|');
    if (refEl.__sig !== legendSig) {
      refEl.__sig = legendSig;
      refEl.replaceChildren(...refs.map(r => { const s = document.createElement('span'); s.className = 'k dash'; s.style.setProperty('--c', cssv('--dim')); s.textContent = r.label; return s; }));
    }
    // hover: crosshair (against the iteration) and a marker per crossing at the hovered record
    const hi = hoverIndex(S), marks = [];
    if (hi >= 0) for (const d of data) if (Number.isFinite(d.x[hi]) && Number.isFinite(d.y[hi])) marks.push({ x: d.x[hi], y: d.y[hi], color: d.color });
    const spec = { series, L: CHART_L, marks, empty: 'no values yet' };
    if (xk === 'it') {
      spec.xmin = iw ? iw.xmin : S.recs[0].it; spec.xmax = Math.max(iw ? iw.xmax : S.last.it, spec.xmin + 1);
      if (hi >= 0) spec.crosshair = mon.hover()?.it != null && !mon.hover().src.startsWith('k') ? mon.hover().it : S.recs[hi].it;
    }
    K.geom[i] = drawPlot(cv, spec);
    K.data[i] = data;
    const h = mon.hover();
    if (h && h.src === `k${i}` && hi >= 0) {
      const xs = xk === 'it' ? '' : ` · ${X_AXES[xk].split(' (')[0]} = ${fmtNum(+(+data[0].x[hi]).toPrecision(4))} ${xUnit(xk)}`;
      const un = chn.unit() === '–' ? '' : ` ${chn.unit()}`;
      mon.tipShow(h.at, `iteration ${S.recs[hi].it}${data.length === 1 ? xs : ''}`, data.map(d => ({ value: Number.isFinite(d.y[hi]) ? `${fmtNum(+(+d.y[hi]).toPrecision(4))}${un}` : '—', label: `c${d.k.edge}${data.length > 1 && xk !== 'it' ? ` @ ${fmtNum(+(+d.x[hi]).toPrecision(3))}` : ''}`, color: d.color })));
    }
  }

  /** the values at the hovered record (or the latest), one row per crossing — the table view of the charts */
  function text(S) {
    const el = $('knot-text');
    if (!A.model || !S || !S.length || !K.knots.length) { el.replaceChildren(); return; }
    const hi = hoverIndex(S), i = hi >= 0 ? hi : S.length - 1, u = A.scene.units;
    const f = (v, p = 3) => (Number.isFinite(v) ? fmtNum(+(+v).toPrecision(p)) : '—');
    const cap = document.createElement('caption'); cap.textContent = `${hi >= 0 ? 'at' : 'latest,'} iteration ${S.recs[i].it}`;
    const cols = [['', ''], ['μ mob', ''], ['Fn', u.force], ['slip', u.length], ['|u|', u.length], ['σ', u.stressLabel]];
    const head = document.createElement('tr');
    for (const [t, un] of cols) { const th = document.createElement('th'); th.textContent = t; if (un) { const s2 = document.createElement('span'); s2.className = 'unit'; s2.textContent = ` ${un}`; th.append(s2); } head.append(th); }
    const rows = K.knots.map((k) => {
      const g = (key) => CHS[key].get(S, k.c)[i];
      const tr = document.createElement('tr');
      const td0 = document.createElement('td'); td0.className = 'kc';
      const key = document.createElement('i'); key.className = 'key'; key.style.setProperty('--c', colorOf(k));
      td0.append(key, document.createTextNode(`c${k.edge}`));
      if (g('state')) { const w = document.createElement('span'); w.className = 'warn'; w.textContent = ' slip'; w.title = 'slipping at this moment'; td0.append(w); }
      tr.append(td0);
      for (const v of [g('mu'), g('Fn'), g('slip'), g('disp'), g('sigma')]) { const td = document.createElement('td'); td.textContent = f(v); tr.append(td); }
      return tr;
    });
    const tbl = document.createElement('table'); tbl.className = 'ktbl';
    tbl.append(cap, head, ...rows);
    el.replaceChildren(tbl);
  }

  // --- hover (pointer and keyboard) ----------------------------------------------------------------------------------
  function nearestIndex(i, px, py) {
    const g = K.geom[i], S = hist(), data = K.data[i]; if (!g || !g.X || !S || !S.length || !data || !data.length) return -1;
    if (xsel.value === 'it') { const it = g.xmin + Math.min(Math.max(px - g.L, 0), g.pw) / g.pw * (g.xmax - g.xmin); return S.nearest(it); }
    let best = -1, bd = Infinity;
    for (const d of data) for (let j = 0; j < d.x.length; j++) {
      if (!Number.isFinite(d.x[j]) || !Number.isFinite(d.y[j])) continue;
      const dd = (g.X(d.x[j]) - px) ** 2 + (g.Y(d.y[j]) - py) ** 2; if (dd < bd) { bd = dd; best = j; }
    }
    return best;
  }
  function bindHover(cv, i) {
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect(), j = nearestIndex(i, e.clientX - r.left, e.clientY - r.top), S = hist();
      if (j < 0) { mon.setHover(null); return; }
      const g = K.geom[i];
      const it = xsel.value === 'it' ? Math.round(g.xmin + Math.min(Math.max(e.clientX - r.left - g.L, 0), g.pw) / g.pw * (g.xmax - g.xmin)) : S.recs[j].it;
      mon.setHover({ it, idx: j, src: `k${i}`, at: { x: e.clientX, y: e.clientY } });
    });
    cv.addEventListener('pointerleave', () => { if (mon.hover()?.src === `k${i}`) mon.setHover(null); });
    cv.addEventListener('keydown', (e) => {
      const S = hist(); if (!S || !S.length) return;
      if (e.key === 'Escape') { mon.setHover(null); return; }
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const h = mon.hover(), j0 = h && h.src === `k${i}` ? h.idx : S.length - 1;
      const j = Math.min(Math.max(j0 + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 1), 0), S.length - 1);
      const g = K.geom[i], r = cv.getBoundingClientRect(), d = K.data[i]?.[0];
      const ax = g && d && Number.isFinite(d.x[j]) ? r.left + g.X(d.x[j]) : r.left + r.width / 2;
      mon.setHover({ it: S.recs[j].it, idx: j, src: `k${i}`, at: { x: ax, y: r.top + 12 } });
    });
    cv.addEventListener('blur', () => { if (mon.hover()?.src === `k${i}`) mon.setHover(null); });
  }

  renderChips(); renderCards();
  return {
    redraw, onModel, onPick, renderChips, renderCards,
    picking: () => K.picking, cancelPick: () => setPicking(false),
    marks: () => K.knots.map(k => ({ c: k.c, color: colorOf(k), label: 'c' + k.edge })),
    knots: () => K.knots,
  };
}
