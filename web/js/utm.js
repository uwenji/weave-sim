// UTM (universal testing machine) CSV import — Mecmesin OmniTest exports and generic force / displacement tables
// (port of weavesim/utm.py). Finds the header row, picks the columns by name (or by the names given), reads units
// from headers such as "Force (N)" / "Travel [mm]" and converts to the scene's units.
import { FORCE_TO_N, LENGTH_TO_M } from './units.js';

const FORCE_KEYS = ['force', 'load', 'kraft'];
const DISP_KEYS = ['displacement', 'travel', 'extension', 'deflection', 'position', 'stroke', 'distance'];
const TIME_KEYS = ['time'];

export class UTMCurve {
  constructor(displacement, force, time = null, source = '', columns = {}) {
    this.displacement = Float64Array.from(displacement); this.force = Float64Array.from(force);
    this.time = time ? Float64Array.from(time) : null; this.source = source; this.columns = columns;
  }
  get length() { return this.force.length; }
  /** loading branch only: the running maximum of displacement */
  monotone() {
    const d = [], f = [], t = [];
    let run = -Infinity;
    for (let i = 0; i < this.length; i++) {
      const v = this.displacement[i];
      const prev = run; run = Math.max(run, v);
      if (i === 0 || run > prev) { d.push(this.displacement[i]); f.push(this.force[i]); if (this.time) t.push(this.time[i]); }
    }
    return new UTMCurve(d, f, this.time ? t : null, this.source, this.columns);
  }
  /** load / unload branches of a cyclic test: [{cycle, branch: 'load'|'unload', d, f}], turning points found on the
   *  displacement with a hysteresis band of tol × range */
  cycles(tol = 0.02) {
    const d = this.displacement, f = this.force, n = d.length;
    if (n < 3) return [{ cycle: 0, branch: 'load', d: Array.from(d), f: Array.from(f) }];
    let lo = Infinity, hi = -Infinity; for (const v of d) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const band = tol * Math.max(hi - lo, 1e-12);
    const out = [];
    let start = 0, rising = true, ext = d[0], iext = 0, cyc = 0;
    const seg = (a, b) => ({ d: Array.from(d.slice(a, b)), f: Array.from(f.slice(a, b)) });
    for (let i = 1; i < n; i++) {
      if (rising) {
        if (d[i] > ext) { ext = d[i]; iext = i; }
        else if (ext - d[i] > band) { out.push({ cycle: cyc, branch: 'load', ...seg(start, iext + 1) }); start = iext; rising = false; ext = d[i]; iext = i; }
      } else {
        if (d[i] < ext) { ext = d[i]; iext = i; }
        else if (d[i] - ext > band) { out.push({ cycle: cyc, branch: 'unload', ...seg(start, iext + 1) }); start = iext; rising = true; ext = d[i]; iext = i; cyc++; }
      }
    }
    out.push({ cycle: cyc, branch: rising ? 'load' : 'unload', ...seg(start, n) });
    return out.filter(b => b.d.length >= 2);
  }
  /** RMS force residual of a simulated curve sampled on the UTM displacement grid (monotone branch) */
  residual(simD, simF) {
    const m = this.monotone();
    let s = 0, k = 0;
    for (let i = 0; i < m.length; i++) {
      const v = interpOrNaN(m.displacement[i], simD, simF);
      if (!Number.isNaN(v)) { s += (v - m.force[i]) ** 2; k++; }
    }
    return k ? Math.sqrt(s / k) : NaN;
  }
}

/** np.interp with left/right = NaN (xp must increase) */
export function interpOrNaN(x, xp, fp) {
  const n = xp.length; if (!n || x < xp[0] || x > xp[n - 1]) return NaN;
  return interp(x, xp, fp);
}
/** np.interp (clamped ends) */
export function interp(x, xp, fp) {
  const n = xp.length; if (!n) return NaN;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  return (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]) * (x - xp[lo]) + fp[lo];
}

const UNIT_RE = /[([]\s*([a-zA-Zµ]+)\s*[)\]]/;
/** the column delimiter: tab or semicolon when it splits the first lines into the same number of fields (exports with a
 *  decimal comma, e.g. Mecmesin VectorPro "0,533;0;-212,73" — the comma outnumbers the semicolon there), else the most
 *  frequent of , ; tab */
function delimiter(text) {
  const lines = text.slice(0, 16384).split(/\r?\n/).filter(l => l.trim()).slice(0, 25);
  const consistent = (d) => {                     // the most common field count is nonzero and shared by ≥ 70 % of the lines
    const c = lines.map(l => l.split(d).length - 1), freq = new Map();
    for (const v of c) if (v > 0) freq.set(v, (freq.get(v) || 0) + 1);
    let best = 0; for (const n of freq.values()) best = Math.max(best, n);
    return c.length > 0 && best >= 0.7 * c.length;
  };
  for (const d of ['\t', ';']) if (consistent(d)) return d;
  const head = text.slice(0, 8192);
  const counts = { ',': head.split(',').length - 1, ';': head.split(';').length - 1, '\t': head.split('\t').length - 1 };
  let best = ','; for (const k of [',', ';', '\t']) if (counts[k] > counts[best]) best = k;
  return counts[best] > 0 ? best : ',';
}
/** Python float() of a cell (decimal comma allowed): nan / inf tokens included → number, or undefined if not a number */
function pyFloat(s) {
  const t = String(s).trim().replace(',', '.');
  if (t === '') return undefined;
  if (/^[+-]?nan$/i.test(t)) return NaN;
  if (/^[+-]?inf(inity)?$/i.test(t)) return t.startsWith('-') ? -Infinity : Infinity;
  if (!/^[+-]?(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?$/.test(t)) return undefined;
  const v = Number(t.replace(/_/g, ''));
  return Number.isNaN(v) ? undefined : v;
}
const isNum = (s) => pyFloat(s) !== undefined;
function findHeader(rows) {
  for (let i = 0; i < Math.min(rows.length - 1, 200); i++) {
    const r = rows[i], nxt = rows[i + 1];
    if (r.filter(c => c.trim() && !isNum(c)).length >= 2 && nxt.filter(isNum).length >= 2) return i;
  }
  return 0;
}
function pick(headers, keys, override) {
  if (override) {
    const i = headers.findIndex(h => h.trim().toLowerCase() === String(override).trim().toLowerCase());
    if (i >= 0) return i;
    const n = parseInt(override, 10); return Number.isFinite(n) ? n : null;
  }
  const low = headers.map(h => h.toLowerCase());
  for (const k of keys) { const i = low.findIndex(h => h.includes(k)); if (i >= 0) return i; }
  return null;
}
function splitCSV(line, delim) {
  const out = []; let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
    else if (ch === delim && !q) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur); return out;
}

/** text of a CSV → UTMCurve in the scene units `to` ({L, F} of a UnitSystem) */
export function parseUTMCsv(text, to, { forceCol = null, dispCol = null, fileUnits = null, source = '' } = {}) {
  text = text.replace(/^﻿/, '');
  const delim = delimiter(text);
  const rows = text.split(/\r?\n/).map(l => splitCSV(l, delim)).filter(r => r.some(c => c.trim()));
  if (rows.length < 2) throw new Error(`${source}: no data rows`);
  const hi = findHeader(rows), headers = rows[hi].map(c => c.trim()), data = rows.slice(hi + 1);
  const fi = pick(headers, FORCE_KEYS, forceCol), di = pick(headers, DISP_KEYS, dispCol), ti = pick(headers, TIME_KEYS, null);
  if (fi == null || di == null) throw new Error(`${source}: could not identify the force / displacement columns in ${JSON.stringify(headers)}`);
  const col = (i) => data.map(r => { const v = r[i] == null ? undefined : pyFloat(r[i]); return v === undefined ? NaN : v; });
  let F = col(fi), D = col(di), T = ti != null ? col(ti) : null;
  const ok = F.map((v, i) => !Number.isNaN(v) && !Number.isNaN(D[i]));
  F = F.filter((_, i) => ok[i]); D = D.filter((_, i) => ok[i]); if (T) T = T.filter((_, i) => ok[i]);
  const unitOf = (h) => { const m = UNIT_RE.exec(h); return m ? m[1] : null; };
  const fu = fileUnits?.force || unitOf(headers[fi]) || 'N', lu = fileUnits?.length || unitOf(headers[di]) || 'mm';
  const fFac = (FORCE_TO_N[fu] ?? 1.0) / to.F, lFac = (LENGTH_TO_M[lu] ?? 1e-3) / to.L;
  // an absolute crosshead position (e.g. Mecmesin "Displacement (mm)" from −212.7 to −112.7) becomes travel from the
  // first sample; records that already start near zero (|d0| ≤ 5 % of the range) are kept as they are
  let d0 = 0; if (D.length) { let lo = Infinity, hi = -Infinity; for (const v of D) { lo = Math.min(lo, v); hi = Math.max(hi, v); } if (Math.abs(D[0]) > 0.05 * Math.max(hi - lo, 1e-300)) d0 = D[0]; }
  return new UTMCurve(D.map(v => (v - d0) * lFac), F.map(v => v * fFac), T, source, { force: headers[fi], displacement: headers[di], force_unit: fu, length_unit: lu, zeroed: d0 });
}
