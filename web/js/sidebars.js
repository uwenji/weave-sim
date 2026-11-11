// The three sidebars (Setup, Monitor, Display): drag-to-resize on the inner edge, and keeping the view usable on
// narrow windows (iPad, a phone in landscape, a narrow desktop window). Layout and fold / dock state are classes on
// #app and CSS variables (style.css); this module only measures and toggles.
//
// Narrow windows (wider than the stacked phone layout, ≤ 820 px): the view has a floor (CSS --view-min, 280 px) and
// the open sidebars shrink to their CSS min-width first; when even that does not fit, the sidebar opened longest ago
// folds to its rail (through its own toggle, so the fold is remembered like a click). Opening a sidebar on a narrow
// window therefore folds another one — the sidebars behave like an accordion there — and the row never overflows
// while two or more sidebars are open.

/** the inner edge of a sidebar widens / narrows it: pointer drag (with capture), double-click resets the default,
 *  arrow keys when focused. `cssVar` is the width property set on #app, `key` its localStorage key, `leftOf()` whether
 *  the sidebar sits left of the view, `lim` {min, floor, max, reserve}: the width stays within
 *  [min, max(floor, min(max, window − reserve))]. */
export function makeResizable({ app, store, rz, el, cssVar, key, leftOf, lim }) {
  const maxW = () => Math.max(lim.floor, Math.min(lim.max, window.innerWidth - lim.reserve));
  const setWidth = (w, save = true) => {
    w = Math.round(Math.min(Math.max(w, lim.min), maxW()));
    app.style.setProperty(cssVar, `${w}px`);
    if (save) store.set(key, w);
  };
  { const w = store.get(key, null); if (w) setWidth(w, false); }
  let drag = null;
  rz.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, w: el.getBoundingClientRect().width }; rz.classList.add('drag'); try { rz.setPointerCapture(e.pointerId); } catch (_) { /* synthetic */ } e.preventDefault(); });
  rz.addEventListener('pointermove', (e) => { if (!drag) return; const dx = e.clientX - drag.x; setWidth(drag.w + (leftOf() ? dx : -dx)); });
  const endDrag = () => { if (drag) { drag = null; rz.classList.remove('drag'); } };
  rz.addEventListener('pointerup', endDrag); rz.addEventListener('pointercancel', endDrag);
  rz.addEventListener('dblclick', () => { app.style.removeProperty(cssVar); store.set(key, null); });
  rz.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const grow = (e.key === 'ArrowLeft') !== leftOf();
    setWidth(el.getBoundingClientRect().width + (grow ? 24 : -24));
  });
  return { setWidth };
}

/** narrow windows: fold the sidebar opened longest ago while the row of sidebars and view overflows the window (or the
 *  view is narrower than minView px). The initial priority (folded first → last): Display, Setup, Monitor. */
export function initSidebarFit({ app, viewport, minView = 200, stackedQuery = '(max-width: 820px)' }) {
  const bars = [
    { cls: 'display-closed', toggle: 'display-toggle' },
    { cls: 'setup-closed', toggle: 'setup-toggle' },
    { cls: 'monitor-closed', toggle: 'monitor-toggle' },
  ];
  const isOpen = (b) => !app.classList.contains(b.cls);
  const stamp = new Map(bars.map((b, i) => [b, i]));         // larger = opened more recently
  let seq = bars.length, open = new Set(bars.filter(isOpen));
  const stacked = window.matchMedia(stackedQuery);
  let fitting = false;
  const cols = ['panel', 'monitor', 'viewport', 'display'].map(id => document.getElementById(id));
  const tooNarrow = () => {
    let w = 0; for (const el of cols) w += el.getBoundingClientRect().width;
    return w > app.clientWidth + 1 || viewport.clientWidth < minView;
  };
  function fit() {
    if (fitting || stacked.matches) return;
    fitting = true;
    try {
      for (let guard = 0; guard < bars.length && tooNarrow(); guard++) {
        const opened = bars.filter(isOpen);
        if (opened.length <= 1) break;
        opened.sort((a, b) => stamp.get(a) - stamp.get(b));
        document.getElementById(opened[0].toggle).click();     // the sidebar's own toggle: remembered like a click
      }
    } finally { fitting = false; }
  }
  new MutationObserver(() => {
    const now = new Set(bars.filter(isOpen));
    for (const b of now) if (!open.has(b)) stamp.set(b, ++seq);  // just opened: the last to be folded
    open = now;
    fit();
  }).observe(app, { attributes: true, attributeFilter: ['class'] });
  let pending = false;
  window.addEventListener('resize', () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; fit(); }); });
  // a sidebar widened by its resizer can also squeeze the view
  new ResizeObserver(() => { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; fit(); }); } }).observe(viewport);
  requestAnimationFrame(fit);
  return { fit };
}
