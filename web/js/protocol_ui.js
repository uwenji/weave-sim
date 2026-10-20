// Load-test protocol, in the Monitor's run bar (the step is called Zero here, as
// in zeroing the gauges before a load test). Zero takes the current (relaxed) geometry as the reference — displacement
// and slip restart from zero, so the zigzag of the input mesh that the relaxation took out is not counted as load
// displacement — and lets the solver settle for the given iterations, the button filling as it goes. At the end the
// settled state becomes the reference and the solver pauses (Run → Pause, nothing else): add the load and Run, and
// Results › delta is the load displacement alone. The drift during the settle is reported. Zero clicked while it settles
// stops the settle. The crossings are never locked by the protocol — fastened joints are the explicit *lock all
// crossings* box in 4 Contact (setLocked), which the chart shows as a blue *locked* band. The phases are kept as
// iteration bands for the convergence chart, in the idiom of the colour-annealing chart: grey relax, sand zero (dashed line at
// the planned end), blue locked.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** '#rrggbb' / '#rgb' / 'rgb(…)' → rgba with alpha a */
export function rgba(col, a) {
  const s = String(col || '').trim();
  let m = /^#([0-9a-f]{3})$/i.exec(s);
  if (m) { const h = m[1]; return `rgba(${parseInt(h[0] + h[0], 16)},${parseInt(h[1] + h[1], 16)},${parseInt(h[2] + h[2], 16)},${a})`; }
  m = /^#([0-9a-f]{6})/i.exec(s);
  if (m) { const v = parseInt(m[1], 16); return `rgba(${v >> 16 & 255},${v >> 8 & 255},${v & 255},${a})`; }
  m = /^rgba?\(([^)]+)\)/i.exec(s);
  if (m) { const [r, g, b] = m[1].split(',').map(t => parseFloat(t)); return `rgba(${r},${g},${b},${a})`; }
  return s || `rgba(128,128,128,${a})`;
}

/** ctx: {A, $, status, fmt, cssv, store, stopRun, startRun, exclusive(fn), syncFromGpu, goalSignature, paramSignature,
 *  readLockBox, refreshViews, drawChart, updateSummaries} */
export function initProtocol(ctx) {
  const { A, $, status, fmt, cssv, store } = ctx;
  const P = { phase: null, phases: [], last: null };
  A.protocol = P;                                                    // console / tests: weavesim.protocol

  const it = () => (A.model ? A.model.st.iteration : 0);
  const contact = () => (A.model && A.model.contact && A.model.contact.locked.length ? A.model.contact : null);
  const isLocked = () => { const c = contact(); return !!c && c.locked.every(Boolean); };
  const units = () => esc(A.scene ? A.scene.units.length : '');
  const steps = () => Math.max(0, Math.round(+$('nb-zero-steps').value || 0));
  /** lath thickness (largest over the groups): a settle drift above a tenth of it means the relaxation had not finished */
  const thickness = () => { const sc = A.scene; let h = sc.section.height; if (sc.hybrid && sc.hybrid.enabled) for (let g = 1; g <= sc.nGroups; g++) h = Math.max(h, sc.group(g).section.height); return h; };
  const moving = (l) => !!(l && l.n && !l.stopped && l.drift > 0.1 * thickness());

  /** largest node displacement from the reference (the drift since Zero) */
  function drift() {
    const st = A.model.st, X = st.X, X0 = st.X0; let d = 0;
    for (let i = 0; i < st.n; i++) {
      const dx = X[3 * i] - X0[3 * i], dy = X[3 * i + 1] - X0[3 * i + 1], dz = X[3 * i + 2] - X0[3 * i + 2], v = dx * dx + dy * dy + dz * dz;
      if (v > d) d = v;
    }
    return Math.sqrt(d);
  }
  function slipMax() { const s = A.model.st.slip; let m = 0; for (let k = 0; k < s.length; k++) m = Math.max(m, Math.abs(s[k])); return m; }

  const openLock = (i) => { if (!P.phases.some(p => p.kind === 'locked' && p.end == null)) P.phases.push({ kind: 'locked', i0: i, end: null }); };
  const closeLock = (i) => { for (const p of P.phases) if (p.kind === 'locked' && p.end == null) p.end = Math.max(i, p.i0); };

  // ----------------------------------------------------------------------------------------- actions (host state synced)
  /** the current state becomes the reference: X0 = X, slip 0 (solver.internalize on the host and the GPU) */
  function zeroNow() {
    const m = A.model;
    m.internalize(); if (A.gpu) A.gpu.internalize(0);
    A.baseline = m.results(); A.results = A.baseline; A.probeTrace = [];
  }
  function lockNow(on) {
    const m = A.model; if (!contact()) return;
    if (on) m.lockContacts(); else m.unlockContacts();
    A.goalSig = ctx.goalSignature(); A.paramSig = ctx.paramSignature();
    if (A.gpu) A.gpu.updateParams(0);
    ctx.readLockBox();
    if (on) openLock(it()); else closeLock(it());
  }

  /** end of the settle: the settled state is the reference, the solver pauses (inside the solve loop); the crossings
   *  are left as they are (locked only if 4 Contact says so) */
  function complete() {
    const ph = P.phase, i = it();
    if (ph) { ph.end = i; ph.drift = drift(); ph.slip = slipMax(); ph.conv = !!A.model.converged; P.phase = null; }
    zeroNow();
    P.last = ph || { kind: 'zero', i0: i, i1: i, end: i, n: 0, drift: 0, slip: 0, conv: false };
    ctx.stopRun();
    const u = A.scene.units.length, l = P.last, mv = moving(l);
    const how = ph ? `settled ${l.end - l.i0} it${l.conv ? ' (converged)' : ''}, drift ${fmt(l.drift, 3)} ${u}${mv ? ' — still moving: relax longer before loading' : ''}` : 'zeroed';
    status(`${how} · reference at it ${i}${isLocked() ? ' (crossings locked)' : ''} — add the load, then Run; Results › delta measures from here`, mv ? 'warn' : '');
    sync(); ctx.drawChart(); ctx.updateSummaries();
  }

  async function onZero() {
    if (!A.model) return;
    if (P.phase) { await stopSettle(); return; }
    ctx.stopRun();
    const n = steps(); store.set('zeroSteps', n);
    await ctx.exclusive(async () => {
      await ctx.syncFromGpu();
      const i0 = it();
      closeLock(i0);                       // a locked stretch ends where a new settle begins (the crossings stay as they are)
      zeroNow();
      if (n > 0) { P.phase = { kind: 'zero', i0, i1: i0 + n, end: null, n }; P.phases.push(P.phase); }
      else complete();
    });
    ctx.refreshViews();
    if (P.phase) { status(`zeroed at it ${P.phase.i0} — settling ${n} iterations, then the solver pauses`); ctx.startRun(); }
    sync(); ctx.drawChart(); ctx.updateSummaries();
  }

  async function stopSettle() {
    ctx.stopRun();
    let ph = null;
    await ctx.exclusive(async () => {
      await ctx.syncFromGpu();
      ph = P.phase; if (!ph) return;
      ph.end = it(); ph.stopped = true; ph.drift = drift(); ph.slip = slipMax(); P.last = ph; P.phase = null;
    });
    if (ph) status(`settle stopped at ${ph.end - ph.i0} / ${ph.n} — the reference stays at it ${ph.i0}`);
    sync(); ctx.drawChart();
  }

  async function setLocked(on) {
    if (!A.model) return;
    if (!contact()) { status('no crossings to lock', 'warn'); sync(); return; }
    await ctx.exclusive(async () => { await ctx.syncFromGpu(); lockNow(on); });
    status(on ? 'crossings locked at their sliding positions (fastened)' : 'crossings free to slide');
    sync(); ctx.drawChart(); ctx.updateSummaries();
  }

  // ----------------------------------------------------------------------------------------- solve-loop hooks
  /** the run's stop target: the end of the settle while one is pending */
  const runTarget = (until) => (P.phase ? P.phase.i1 : until);
  /** Step never runs past the end of the settle */
  const clampStep = (n) => (P.phase ? Math.max(1, Math.min(n, P.phase.i1 - it())) : n);
  /** after a run / step stopped (host synced, inside the solve loop): finish the settle if it reached its end */
  function onRunEnd() { if (P.phase && A.model && (it() >= P.phase.i1 || A.model.converged)) complete(); }
  /** the crossings were locked / unlocked elsewhere (4 Contact checkbox via the live-edit sync) */
  function lockChanged(on) { if (on) openLock(it()); else closeLock(it()); sync(); }
  function reset() { P.phase = null; P.phases = []; P.last = null; sync(); }

  // ----------------------------------------------------------------------------------------- display
  function sync() {
    const bz = $('btn-zero'), ph = P.phase;
    bz.classList.toggle('on', !!ph);
    bz.setAttribute('aria-pressed', ph ? 'true' : 'false');
    bz.style.setProperty('--p', ph ? `${Math.min(100, Math.max(0, 100 * (it() - ph.i0) / Math.max(ph.n, 1))).toFixed(1)}%` : '0%');
    bz.disabled = !A.model;
  }

  /** x window and phase bands for the solver charts. first / last: iterations of the oldest and newest history entry.
   *  The window keeps the latest Zero in view (with some relax before it) and reaches to the planned end. */
  function chart(first, last) {
    let start = Math.max(first, last - 3999), xmax = last;
    const ph = P.phase; if (ph) xmax = Math.max(xmax, ph.i1);
    const zs = P.phases.filter(p => p.kind === 'zero'), lastZ = zs[zs.length - 1];
    if (lastZ && last - lastZ.i0 <= 20000) {
      const span = Math.max(xmax - lastZ.i0, 1);
      start = Math.min(start, Math.max(first, Math.floor(lastZ.i0 - Math.max(0.25 * span, 100))));
    }
    const dim = cssv('--dim'), ink = cssv('--zero'), lk = cssv('--lock'), bands = [], vlines = [];
    if (ph) vlines.push({ x: ph.i1, color: ink });
    else if (lastZ && lastZ.end != null) vlines.push({ x: lastZ.end, color: dim });
    // a dashed line at the end of the axis would sit on the frame, and a lock opened at the last iteration would have no
    // width yet: leave a little room after them
    const pad = 0.03 * Math.max(xmax - start, 10);
    for (const v of vlines) if (v.x > xmax - pad) xmax = v.x + pad;
    for (const p of P.phases) if (p.kind === 'locked' && p.end == null && p.i0 > xmax - pad) xmax = p.i0 + pad;
    if (zs.length && zs[0].i0 > start) bands.push({ x0: start, x1: zs[0].i0, fill: rgba(dim, 0.1), label: 'relax', ink: dim });
    for (const p of P.phases) {
      const x1 = p.kind === 'zero' ? (p.end ?? p.i1) : (p.end ?? xmax);
      if (!(x1 > p.i0)) continue;
      if (p.kind === 'zero') bands.push({ x0: p.i0, x1, fill: rgba(ink, 0.17), label: 'zero', ink });
      else bands.push({ x0: p.i0, x1, fill: rgba(lk, 0.08), label: 'locked', ink: lk });
    }
    return { start, xmax, bands, vlines };
  }

  /** the protocol line of the run bar (html) */
  function line() {
    if (!A.model) return '';
    const u = units(), ph = P.phase;
    if (ph) return `<span class="zero">zero · settle ${Math.min(it() - ph.i0, ph.n)} / ${ph.n}</span> · drift ${fmt(drift(), 3)} ${u} · slip ${fmt(slipMax(), 3)} ${u}`;
    const l = P.last; if (!l) return '';
    if (l.stopped) return `<span class="zero">settle stopped</span> at ${l.end - l.i0} / ${l.n} · reference at it ${l.i0} · drift ${fmt(l.drift, 3)} ${u}`;
    const dr = `settle drift ${fmt(l.drift, 3)} ${u}, slip ${fmt(l.slip, 3)} ${u}${l.conv ? ' (converged)' : ''}`;
    const settle = !l.n ? '' : moving(l) ? ` · <span class="warn">${dr} — still moving, relax longer</span>` : ` · ${dr}`;
    const head = isLocked() ? `locked at it ${l.end}` : `reference at it ${l.end}`;
    const next = it() === l.end ? ' · add the load, then Run' : ' · delta from here';
    return `<span class="lock">${head}</span>${settle}${next}`;
  }

  // ----------------------------------------------------------------------------------------- bindings
  $('nb-zero-steps').value = store.get('zeroSteps', store.get('intSteps', +$('nb-zero-steps').value || 1000));
  $('nb-zero-steps').addEventListener('change', () => store.set('zeroSteps', steps()));
  $('btn-zero').onclick = () => { onZero().catch(e => { console.error(e); status('Zero failed: ' + e.message, 'warn'); }); };
  sync();

  return { runTarget, clampStep, onRunEnd, lockChanged, reset, sync, chart, line, setLocked, isLocked, active: () => !!P.phase };
}
