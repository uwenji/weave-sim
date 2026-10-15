// The Display sidebar (the Joint FEA "display tools" idiom, beside the view, folds to a narrow tab, docks left or
// right): lath colouring (family, pure colour, over / under, pieces, lath group, strand, or a result layer with the
// Python viewer's palettes, smooth along the laths or per segment, bands, delta, range), the member diagrams with their
// labels and the joint-zone option, and the joints overlay of the discretisation. The View section of the sidebar
// (theme, layers, deformation, arrows, turntable) is bound in app.js.
import { DIAGRAMS, DIAGRAM_LABELS, DIAGRAM_AXIS, buildDiagram, diagramUnit } from './diagrams.js';
import { deltaResults, nodeValues, knotNodePairs, fmtG } from './fields.js';
import { jointPoints, jointCuts, pieceIndex, zoneMask, strandKnotRows } from './discretize.js';
import { lutGradient, cmapFor, hex, toHex, familyColor, GROUP_COLORS, OVER_COLOR, UNDER_COLOR, PIECE_COLORS, STATE_HEX, CTYPE_HEX } from './colormap.js';

export const SEG_LAYERS = ['Mx', 'My', 'T', 'N'];
export const KNOT_LAYERS = ['Fn', 'Ft', 'slip', 'state', 'gap', 'pressure', 'util', 'ctype', 'dent', 'M_rot', 'theta', 'lambda'];
export const SIGNED = new Set(['Mx', 'My', 'T', 'N', 'Ft']);
export const CATEGORICAL = { state: { hex: STATE_HEX, names: ['stick', 'slip'] }, ctype: { hex: CTYPE_HEX, names: ['FF', 'FE', 'EE'] } };
export const LAYER_LABEL = { Mx: 'Mx flatwise moment', My: 'My edgewise moment', T: 'T torsion', N: 'N axial (+tension)', Fn: 'contact normal force', Ft: 'contact tangential force', slip: 'contact slip', state: 'stick / slip', gap: 'gap', pressure: 'contact pressure (nominal)', util: 'embedment utilisation', ctype: 'contact type', dent: 'seat depth', M_rot: 'scissor moment about n', theta: 'crossing angle θ', lambda: 'AL multiplier λ', disp: 'displacement |u|' };

/** unit of a layer in the scene units */
export function layerUnit(sel, u) {
  return { Mx: `${u.force}·${u.length}`, My: `${u.force}·${u.length}`, T: `${u.force}·${u.length}`, M_rot: `${u.force}·${u.length}`, N: u.force, Fn: u.force, Ft: u.force, lambda: u.force, slip: u.length, gap: u.length, dent: u.length, disp: u.length, pressure: u.stressLabel, util: '–', theta: '°' }[sel] || '';
}

export function initDisplay(ctx) {
  const { A, $, qsa, fmt, store } = ctx;
  const D = { d: store.get('diagram', 'none'), scale: 1, fill: 0.35, labels: 1, zone: false, zoneRange: 3 };
  if (!DIAGRAMS.includes(D.d)) D.d = 'none';
  let rowsCache = null, pairsCache = null;
  const rows = () => (rowsCache && rowsCache.topo === A.topo ? rowsCache.rows : (rowsCache = { topo: A.topo, rows: strandKnotRows(A.topo) }).rows);
  const pairs = () => (pairsCache && pairsCache.st === A.model.st ? pairsCache.p : (pairsCache = { st: A.model.st, p: knotNodePairs(A.model.st, A.topo) }).p);
  const hasJoints = () => !!A.scene && Object.values(A.scene.joints || {}).some(v => v && v.length);

  // --- the sidebar: fold to a tab, dock left / right (remembered) -------------------------------------------------
  const app = $('app');
  const setDock = (left) => { app.classList.toggle('display-left', left); store.set('displayLeft', left); };
  const setOpen = (open) => { app.classList.toggle('display-closed', !open); store.set('displayOpen', open); $('display-toggle').setAttribute('aria-expanded', String(open)); };
  setDock(!!store.get('displayLeft', false)); setOpen(store.get('displayOpen', true) !== false);
  $('display-toggle').onclick = () => setOpen(app.classList.contains('display-closed'));
  $('display-dock').onclick = () => setDock(!app.classList.contains('display-left'));

  // --- colouring --------------------------------------------------------------------------------------------------
  /** layer values per segment / node / knot; segment layers smooth → node values (mean of the adjacent segments) */
  function layerValues(layer, r) {
    if (!r) return null;
    const st = A.model.st;
    if (SEG_LAYERS.includes(layer)) return $('ck-smooth').checked ? { where: 'node', vals: nodeValues(st, r, layer) } : { where: 'segment', vals: Float64Array.from(r.segment[layer]) };
    if (layer === 'disp') return { where: 'node', vals: nodeValues(st, r, 'disp') };
    if (KNOT_LAYERS.includes(layer)) {
      const src = r.knot[layer]; if (!src) return { where: 'knot', vals: null };
      let vals = Float64Array.from(src); if (layer === 'theta') vals = vals.map(v => v * 180 / Math.PI);
      return { where: 'knot', vals };
    }
    return null;
  }
  function applyColouring() {
    if (!A.model) return;
    const view = A.view, sel = $('sel-colour').value, lg = $('legend');
    $('row-purecolour').hidden = sel !== 'plain';
    $('layer-ctl').hidden = !LAYER_LABEL[sel];
    if (!LAYER_LABEL[sel]) {
      const c = { mode: sel };
      if (sel === 'plain') c.color = hex($('in-purecolour').value);
      if (sel === 'pieces') c.pieces = pieceIndex(A.scene, A.model.st);
      view.setColouring(c); weaveLegend(sel);
      if (sel === 'pieces' && !hasJoints()) ctx.status('no joints yet — strip › Discretisation › Run');
      return;
    }
    if (!A.results) { view.setColouring({ mode: 'plain', color: [0.80, 0.80, 0.78] }); lg.hidden = true; $('ro-extremes').textContent = 'no results yet — Run or Step'; return; }
    const delta = $('ck-delta').checked && A.baseline && !CATEGORICAL[sel];
    const r = delta ? deltaResults(A.results, A.baseline) : A.results;
    const lv = layerValues(sel, r);
    if (!lv || !lv.vals) { view.setColouring({ mode: 'plain', color: [0.80, 0.80, 0.78] }); lg.hidden = true; $('ro-extremes').textContent = `${sel}: not computed in this model (switch on the matching contact option)`; return; }
    const u = A.scene.units, unit = layerUnit(sel, u);
    if (CATEGORICAL[sel]) {
      const cat = CATEGORICAL[sel];
      view.setColouring({ mode: 'layer', where: lv.where, values: lv.vals, categorical: cat.hex });
      lg.hidden = false; lg.classList.add('cat');
      $('legend-title').textContent = LAYER_LABEL[sel];
      const counts = cat.names.map((_, i) => lv.vals.reduce((a, v) => a + (Math.round(v) === i), 0));
      $('legend-bar').innerHTML = cat.names.map((nm, i) => `<span><i style="background:${cat.hex[i]}"></i>${nm} ${counts[i]}</span>`).join('');
      return;
    }
    lg.classList.remove('cat'); $('legend-bar').innerHTML = '';
    let lo = Infinity, hi = -Infinity; for (const v of lv.vals) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const signed = SIGNED.has(sel) || delta;
    if ($('ck-autorange').checked) {
      if (signed) { const mm = Math.max(Math.abs(lo), Math.abs(hi), 1e-12); lo = -mm; hi = mm; }
      if (hi <= lo) hi = lo + 1e-9;
      $('nb-vmin').value = +lo.toPrecision(4); $('nb-vmax').value = +hi.toPrecision(4);
    } else { lo = +$('nb-vmin').value; hi = +$('nb-vmax').value; if (!(hi > lo)) hi = lo + 1e-9; }
    const cmap = cmapFor($('sel-palette').value, signed), bands = $('ck-bands').checked ? 10 : 0;
    view.setColouring({ mode: 'layer', where: lv.where, values: lv.vals, cmap, lo, hi, bands });
    lg.hidden = false;
    $('legend-title').textContent = `${delta ? 'Δ ' : ''}${LAYER_LABEL[sel]} [${unit}]${lv.where === 'segment' ? ' · per segment' : ''}`;
    $('legend-bar').style.background = lutGradient(cmap, 'to right', bands);
    $('legend-lo').textContent = fmt(lo, 3); $('legend-mid').textContent = fmt((lo + hi) / 2, 3); $('legend-hi').textContent = fmt(hi, 3);
  }

  /** legend of the categorical lath colourings (family, lath group, over / under, pieces): the colours as drawn */
  function weaveLegend(sel) {
    const lg = $('legend'), view = A.view, topo = A.topo, sc = A.scene;
    const css = (rgb) => toHex(view.dark ? rgb.map(v => 0.82 * v + 0.18) : rgb);
    let title = '', items = [];
    if (sel === 'family' && topo) {
      const n = new Map(); topo.strands.forEach(s => n.set(s.family, (n.get(s.family) || 0) + 1));
      title = 'lath families (strands)'; items = [...n.keys()].sort((a, b) => a - b).map(f => ({ c: css(familyColor(f)), t: `${f >= 0 ? f : '–'} · ${n.get(f)}` }));
    } else if (sel === 'group' && topo) {
      const n = new Map(); for (let s = 0; s < topo.nStrands; s++) { const g = sc.strandGroup(s); n.set(g, (n.get(g) || 0) + 1); }
      title = 'lath groups (strands)'; items = [...n.keys()].sort((a, b) => a - b).map(g => ({ c: css(GROUP_COLORS[(g - 1) % GROUP_COLORS.length]), t: `${(sc.group(g).name || 'group ' + g).trim()} · ${n.get(g)}` }));
    } else if (sel === 'overunder') { title = 'at the nearest crossing'; items = [{ c: css(OVER_COLOR), t: 'over' }, { c: css(UNDER_COLOR), t: 'under' }]; }
    else if (sel === 'pieces' && hasJoints()) { title = 'pieces between the joints'; items = [{ c: css(PIECE_COLORS[0]), t: 'odd' }, { c: css(PIECE_COLORS[1]), t: 'even' }]; }
    if (!items.length) { lg.hidden = true; return; }
    lg.hidden = false; lg.classList.add('cat');
    $('legend-title').textContent = title;
    $('legend-bar').style.background = '';
    $('legend-bar').innerHTML = items.map(i => `<span><i style="background:${i.c}"></i>${i.t}</span>`).join('');
  }

  // --- diagrams ---------------------------------------------------------------------------------------------------
  $('diag-chips').innerHTML = DIAGRAMS.map(d => `<button class="chip${d === D.d ? ' on' : ''}" data-diag="${d}" title="${DIAGRAM_LABELS[d]}">${d === 'none' ? 'off' : d}</button>`).join('');
  qsa('#diag-chips .chip').forEach(b => { b.onclick = () => { D.d = b.dataset.diag; store.set('diagram', D.d); qsa('#diag-chips .chip').forEach(c => c.classList.toggle('on', c === b)); refreshDiagram(); }; });
  const slider = (id, out, key, digits) => { $(id).oninput = (e) => { D[key] = +e.target.value; $(out).textContent = digits >= 0 ? (+e.target.value).toFixed(digits) : e.target.value; refreshDiagram(); }; };
  slider('sl-dscale', 'out-dscale', 'scale', 2); slider('sl-dfill', 'out-dfill', 'fill', 2); slider('sl-zone', 'out-zone', 'zoneRange', -1);
  $('sel-dlabels').onchange = (e) => { D.labels = +e.target.value; refreshDiagram(); };
  $('ck-zone').onchange = (e) => { D.zone = e.target.checked; $('row-zone').hidden = !D.zone; refreshDiagram(); };

  let lastSpec = null;
  function refreshDiagram() {
    const view = A.view;
    $('diag-ctl').hidden = D.d === 'none';
    $('row-dfill').hidden = !(D.d in DIAGRAM_AXIS);
    $('joint-ctl').hidden = !hasJoints(); $('row-zone').hidden = !D.zone;
    if (!A.model || !view.dispX) return;
    if (D.d === 'none') { view.setDiagram(null); lastSpec = null; $('ro-diagram').textContent = ''; return; }
    const st = A.model.st;
    let r = A.results;
    if (r && $('ck-delta').checked && A.baseline) r = deltaResults(r, A.baseline);
    const zm = D.zone && hasJoints() && D.d in DIAGRAM_AXIS ? (s) => zoneMask(A.scene, A.topo, st, s, D.zoneRange, st.X, rows()) : null;
    const spec = buildDiagram(D.d, {
      st, topo: A.topo, results: r, P: view.dispX, R: view.nodeR, X0: st.X0, size: view.size, units: A.scene.units, contact: A.model.contact,
      deform: view.deform, knotPairs: pairs(), opts: { scale: D.scale, labels: D.labels, palette: $('sel-palette').value, zoneMask: zm },
    });
    spec.fill = D.fill; spec.scale = D.scale; lastSpec = spec;
    view.setDiagram(spec);
    $('ro-diagram').textContent = spec.info;
  }

  // --- joints overlay -------------------------------------------------------------------------------------------
  function refreshJoints() {
    const view = A.view;
    $('joint-ctl').hidden = !hasJoints();
    if (!A.model || !view.dispX) return;
    if (!hasJoints()) { view.setJoints(null); return; }
    const jp = jointPoints(A.scene, A.model.st, view.dispX);
    view.setJoints({ cuts: jointCuts(jp, view.nodeR, (s) => [view.secW[s], view.secH[s]]), points: jp.map(j => j.p), kappa: jp.map(j => j.kappa) });
  }
  /** the joints changed (discretisation): overlay, piece colouring, joint-zone diagram */
  function onJoints() {
    refreshJoints();
    if ($('sel-colour').value === 'pieces') applyColouring();
    if (D.d !== 'none' && D.zone) refreshDiagram();
  }

  // --- controls -------------------------------------------------------------------------------------------------
  ['sel-colour', 'sel-palette', 'ck-delta', 'ck-autorange', 'ck-smooth', 'ck-bands'].forEach(id => $(id).addEventListener('change', () => {
    $('nb-vmin').disabled = $('nb-vmax').disabled = $('ck-autorange').checked;
    if (LAYER_LABEL[$('sel-colour').value] && !A.results && A.model && A.model.st.iteration > 0) ctx.refreshResults(); else applyColouring();
    if (id === 'ck-delta' || id === 'sel-palette') refreshDiagram();
    store.set('colour', $('sel-colour').value);
  }));
  ['nb-vmin', 'nb-vmax'].forEach(id => $(id).addEventListener('change', applyColouring));
  $('in-purecolour').addEventListener('input', applyColouring);
  const savedColour = store.get('colour', 'family');
  if ([...$('sel-colour').options].some(o => o.value === savedColour)) $('sel-colour').value = savedColour;

  // --- hover probe (the Joint FEA idiom): the lath under the cursor and the local values ---------------------------
  /** hit {seg, point} on a lath → {title, lines} (HTML); values interpolated at the point exactly as they are drawn */
  function probe(hit) {
    const view = A.view, st = A.model.st, j = hit.seg, a = st.segI0[j], b = st.segI1[j], s = st.segStrand[j], P = view.dispX, X = st.X;
    const dx = P[3 * b] - P[3 * a], dy = P[3 * b + 1] - P[3 * a + 1], dz = P[3 * b + 2] - P[3 * a + 2], l2 = dx * dx + dy * dy + dz * dz;
    const t = l2 > 0 ? Math.min(1, Math.max(0, ((hit.point.x - P[3 * a]) * dx + (hit.point.y - P[3 * a + 1]) * dy + (hit.point.z - P[3 * a + 2]) * dz) / l2)) : 0;
    const lerp = (arr) => (1 - t) * arr[a] + t * arr[b];
    const u = A.scene.units, sc = A.scene, topo = A.topo;
    // arc length along the lath on the solved geometry
    const i0 = st.nodeOffset[s], j0 = st.segOffset[s], m = st.segOffset[s + 1] - j0;
    let sPos = 0, L = 0;
    for (let q = j0; q < j0 + m; q++) {
      const i = st.segI0[q], k = st.segI1[q], len = Math.hypot(X[3 * k] - X[3 * i], X[3 * k + 1] - X[3 * i + 1], X[3 * k + 2] - X[3 * i + 2]);
      if (q < j) sPos += len; else if (q === j) sPos += t * len;
      L += len;
    }
    void i0;
    const g = sc.strandGroup(s), gname = (sc.group(g).name || '').trim();
    const title = `<b>lath ${s}</b> · family ${topo.strands[s].family}${sc.hybrid.enabled ? ` · group ${g}${gname && gname !== `group ${g}` ? ' ' + gname : ''}` : ''}`;
    const lines = [`s ${fmtG(sPos, 4)} of ${fmtG(L, 4)} ${u.length}`];
    const c = view.colouring, sel = $('sel-colour').value;
    let shown = false;
    if (c && c.mode === 'layer' && c.values && c.where !== 'knot' && !c.categorical) {
      const v = c.where === 'node' ? lerp(c.values) : c.values[j];
      lines.push(`${$('ck-delta').checked && A.baseline ? 'Δ ' : ''}${LAYER_LABEL[sel] || sel}: <b>${fmtG(v, 4)}</b> ${layerUnit(sel, u)}`);
      shown = true;
    }
    if (lastSpec && lastSpec.values && D.d in DIAGRAM_AXIS && D.d !== sel) {
      lines.push(`${DIAGRAM_LABELS[D.d]}: <b>${fmtG(lerp(lastSpec.values), 4)}</b> ${diagramUnit(u, D.d)}`); shown = true;
    } else if (D.d === 'defl' && A.results) {
      const r = $('ck-delta').checked && A.baseline ? deltaResults(A.results, A.baseline) : A.results, Dp = r.disp;
      const da = Math.hypot(Dp[3 * a], Dp[3 * a + 1], Dp[3 * a + 2]), db = Math.hypot(Dp[3 * b], Dp[3 * b + 1], Dp[3 * b + 2]);
      lines.push(`|disp|: <b>${fmtG((1 - t) * da + t * db, 4)}</b> ${u.length}`); shown = true;
    }
    if (!shown && A.results) {
      const r = A.results.segment;
      lines.push(`Mx <b>${fmtG(r.Mx[j], 4)}</b> · My <b>${fmtG(r.My[j], 4)}</b> · T <b>${fmtG(r.T[j], 4)}</b> ${u.force}·${u.length}`);
      lines.push(`N <b>${fmtG(r.N[j], 4)}</b> ${u.force}`);
    }
    return { title, lines };
  }

  let lastJoints = 0;
  /** during a run the joints ride on the moving laths: redraw them at most every 200 ms */
  function refreshJointsThrottled() { const t = performance.now(); if (t - lastJoints > 200) { lastJoints = t; refreshJoints(); } }
  return { applyColouring, refreshDiagram, refreshJoints, refreshJointsThrottled, onJoints, probe, diagramActive: () => D.d !== 'none', state: D, hasJoints };
}
