// Small canvas plots in the app's idiom (theme colours from the CSS variables, crisp on HiDPI): line / dot series,
// linear or log y axis, nice ticks, a compact legend. Used by the solve strip, the UTM overlay and the calibration.
const css = (name, fb) => (typeof getComputedStyle === 'function' ? getComputedStyle(document.documentElement).getPropertyValue(name).trim() : '') || fb;

export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { const d = Math.abs(lo) || 1; lo -= 0.5 * d; hi += 0.5 * d; }
  const raw = (hi - lo) / Math.max(n, 1), mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v);
  return { ticks: out, step };
}
export function fmtNum(v, step = null) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(a >= 1e5 ? 1 : 0).replace('e+', 'e');
  if (step != null) { const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9))); return v.toFixed(d); }
  return +v.toPrecision(4) + '';
}

/** spec: {series: [{x, y, color, width, dots, dash, label, alpha, bars, barWidth}], xlabel, ylabel, logY, xmin, xmax,
 *  ymin, ymax, legend (true | 'top'), vlines: [{x, color}], bands: [{x0, x1, fill, label, ink}], xticks: [{x, label}]
 *  (categorical axis), note, empty}. Bar series draw a rectangle from 0 to y around each x (barWidth in x units).
 *  Bands shade an x range behind the grid (phases of a run), labelled at their top left when there is room. */
export function drawPlot(canvas, spec) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = canvas.clientWidth || canvas.width, H = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const fg = css('--fg', '#212324'), dim = css('--dim', '#70737a'), grid = css('--border', '#dcdcd9');
  const font = `10.5px ${css('--font-mono', 'monospace')}`;
  g.font = font;
  const series = (spec.series || []).filter(s => s && s.x && s.x.length);
  const logY = !!spec.logY;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const s of series) for (let i = 0; i < s.x.length; i++) {
    const x = s.x[i], y = s.y[i];
    if (!Number.isFinite(x) || !Number.isFinite(y) || (logY && y <= 0)) continue;
    const hw = s.bars ? 0.5 * (s.barWidth || 0.8) : 0;
    xmin = Math.min(xmin, x - hw); xmax = Math.max(xmax, x + hw); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    if (s.bars) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); }
  }
  if (spec.xmin != null) xmin = spec.xmin; if (spec.xmax != null) xmax = spec.xmax;
  if (spec.ymin != null) ymin = spec.ymin; if (spec.ymax != null) ymax = spec.ymax;
  const empty = !Number.isFinite(xmin) || !Number.isFinite(ymin);
  if (empty) { xmin = 0; xmax = 1; ymin = logY ? 1e-6 : 0; ymax = 1; }
  if (xmax <= xmin) xmax = xmin + 1;
  if (logY) { ymin = Math.max(ymin, 1e-300); if (ymax <= ymin) ymax = ymin * 10; } else if (ymax <= ymin) { const d = Math.abs(ymin) || 1; ymin -= 0.5 * d; ymax += 0.5 * d; }
  else { const pad = 0.06 * (ymax - ymin); if (spec.ymin == null) ymin -= pad; if (spec.ymax == null) ymax += pad; }
  const L = spec.compact ? 36 : 44, R = 8, T = spec.legend === 'top' ? 16 : 6, B = spec.xlabel ? 28 : 18;
  const pw = Math.max(W - L - R, 10), ph = Math.max(H - T - B, 10);
  const ly0 = logY ? Math.log10(ymin) : 0, ly1 = logY ? Math.log10(ymax) : 0;
  const X = (x) => L + (x - xmin) / (xmax - xmin) * pw;
  const Y = (y) => T + ph - (logY ? (Math.log10(Math.max(y, 1e-300)) - ly0) / Math.max(ly1 - ly0, 1e-12) : (y - ymin) / (ymax - ymin)) * ph;
  // phase bands behind the grid and the series
  const bands = empty ? [] : (spec.bands || []).filter(b => b && b.x1 > b.x0 && b.x1 > xmin && b.x0 < xmax);
  for (const b of bands) { const xa = X(Math.max(b.x0, xmin)), xb = X(Math.min(b.x1, xmax)); g.fillStyle = b.fill; g.fillRect(xa, T, Math.max(xb - xa, 1), ph); }
  // grid + ticks
  g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = dim; g.textBaseline = 'middle';
  g.textAlign = 'center';
  if (spec.xticks) for (const t of spec.xticks) { const x = Math.round(X(t.x)) + 0.5; g.fillText(t.label, x, T + ph + 8); }
  else {
    const xt = niceTicks(xmin, xmax, Math.max(2, Math.floor(pw / 70)));
    for (const v of xt.ticks) { const x = Math.round(X(v)) + 0.5; g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.globalAlpha = 0.5; g.stroke(); g.globalAlpha = 1; g.fillText(fmtNum(v, xt.step), x, T + ph + 8); }
  }
  g.textAlign = 'right';
  if (logY) {
    const d0 = Math.ceil(ly0 - 1e-9), d1 = Math.floor(ly1 + 1e-9), every = Math.max(1, Math.ceil((d1 - d0 + 1) / Math.max(2, Math.floor(ph / 22))));
    for (let d = d0; d <= d1; d += every) { const y = Math.round(Y(10 ** d)) + 0.5; g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.globalAlpha = 0.5; g.stroke(); g.globalAlpha = 1; g.fillText(`1e${d}`, L - 4, y); }
  } else {
    const yt = niceTicks(ymin, ymax, Math.max(2, Math.floor(ph / 24)));
    for (const v of yt.ticks) { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.globalAlpha = 0.5; g.stroke(); g.globalAlpha = 1; g.fillText(fmtNum(v, yt.step), L - 4, y); }
  }
  g.strokeStyle = dim; g.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
  if (spec.xlabel) { g.textAlign = 'center'; g.fillStyle = dim; g.fillText(spec.xlabel, L + pw / 2, H - 6); }
  if (spec.ylabel) { g.save(); g.translate(9, T + ph / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillStyle = dim; g.fillText(spec.ylabel, 0, 0); g.restore(); }
  for (const v of spec.vlines || []) { if (v.x < xmin || v.x > xmax) continue; g.strokeStyle = v.color || dim; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(X(v.x) + 0.5, T); g.lineTo(X(v.x) + 0.5, T + ph); g.stroke(); g.setLineDash([]); }
  // series (clipped to the plot area)
  g.save(); g.beginPath(); g.rect(L, T, pw, ph); g.clip();
  for (const s of series) {
    g.strokeStyle = s.color || fg; g.fillStyle = s.color || fg; g.lineWidth = s.width || 1.4; g.globalAlpha = s.alpha ?? 1;
    g.setLineDash(s.dash || []);
    if (s.bars) {
      const hw = 0.5 * (s.barWidth || 0.8), y0 = Y(Math.max(ymin, Math.min(ymax, 0)));
      for (let i = 0; i < s.x.length; i++) { const x = s.x[i], y = s.y[i]; if (!Number.isFinite(x) || !Number.isFinite(y)) continue; const xa = X(x - hw), xb = X(x + hw), yv = Y(y); g.fillRect(Math.min(xa, xb) + 0.5, Math.min(y0, yv), Math.max(Math.abs(xb - xa) - 1, 1), Math.max(Math.abs(yv - y0), 1)); }
      g.globalAlpha = 1; g.setLineDash([]); continue;
    }
    if (!s.dots || s.line) {
      g.beginPath(); let pen = false;
      const step = Math.max(1, Math.floor(s.x.length / (2 * pw)));          // decimate long histories to ~2 points per pixel
      for (let i = 0; i < s.x.length; i += (i + step < s.x.length ? step : 1)) {
        const x = s.x[i], y = s.y[i];
        if (!Number.isFinite(x) || !Number.isFinite(y) || (logY && y <= 0)) { pen = false; continue; }
        if (pen) g.lineTo(X(x), Y(y)); else { g.moveTo(X(x), Y(y)); pen = true; }
      }
      g.stroke();
    }
    g.setLineDash([]);
    if (s.dots) {
      const r = s.r || 2.2;
      for (let i = 0; i < s.x.length; i++) { const x = s.x[i], y = s.y[i]; if (!Number.isFinite(x) || !Number.isFinite(y) || (logY && y <= 0)) continue; g.beginPath(); g.arc(X(x), Y(y), r, 0, 2 * Math.PI); g.fill(); }
    }
    g.globalAlpha = 1;
  }
  g.restore();
  // band labels over the series, with a halo in the panel colour
  if (bands.length) {
    g.textAlign = 'left'; g.textBaseline = 'middle'; g.lineJoin = 'round';
    const halo = css('--panel', '#fbfbfa');
    for (const b of bands) {
      if (!b.label) continue;
      const xa = X(Math.max(b.x0, xmin)), xb = X(Math.min(b.x1, xmax)), tw = g.measureText(b.label).width;
      if (xb - xa < tw + 8) continue;
      g.lineWidth = 3; g.strokeStyle = halo; g.globalAlpha = 0.8; g.strokeText(b.label, xa + 4, T + 9); g.globalAlpha = 1;
      g.fillStyle = b.ink || dim; g.fillText(b.label, xa + 4, T + 9);
    }
  }
  // legend
  const labelled = series.filter(s => s.label);
  if (spec.legend && labelled.length) {
    g.textAlign = 'left'; g.textBaseline = 'middle';
    let x = L + 6; const y = spec.legend === 'top' ? 8 : T + 9;
    for (const s of labelled) {
      g.fillStyle = s.color || fg; g.globalAlpha = s.alpha ?? 1; g.fillRect(x, y - 1.5, 10, 3); g.globalAlpha = 1;
      g.fillStyle = dim; g.fillText(s.label, x + 13, y);
      x += 13 + g.measureText(s.label).width + 12;
    }
  }
  if (spec.note) { g.textAlign = 'right'; g.fillStyle = dim; g.fillText(spec.note, L + pw - 4, T + ph - 9); }
  if (empty && spec.empty) { g.textAlign = 'center'; g.fillStyle = dim; g.fillText(spec.empty, L + pw / 2, T + ph / 2); }
  return { X, Y, L, T, pw, ph, xmin, xmax, ymin, ymax };
}
