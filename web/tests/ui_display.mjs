// Headless check of the display features (SwiftShader): ground shadow, lath colours (family / layer smooth / bands),
// member diagrams with labels, the discretisation (Monitor group, annealing, joints, pieces, joint-zone diagram), lath
// selection for the groups (LocalSH idiom), the Rhino export (download → ref_out/ui/<scene>.3dm), zoom responsiveness,
// the Display sidebar (fold, dock left). Screenshots → ref_out/ui/d_*.png.
//   node web/tests/ui_display.mjs [out] [cpu|webgpu] [scene query]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, REF, launchChromium } from './paths.mjs';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.stl': 'application/octet-stream', '.csv': 'text/csv', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  let p = path.join(ROOT, u);
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
  fs.createReadStream(p).pipe(res);
}).listen(0);
const port = server.address().port;
const out = process.argv[2] || REF + '/ui';
fs.mkdirSync(out, { recursive: true });
const backend = process.argv[3] || 'cpu';
const browser = await launchChromium();
const context = await browser.newContext({ viewport: { width: 1600, height: 960 }, deviceScaleFactor: 1, acceptDownloads: true });
const page = await context.newPage();
const errors = [];
page.on('console', m => { const t = m.text(); if (t.includes('experimental on this platform')) return; if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${t}`); if (m.type() !== 'debug') console.log(`[${m.type()}] ${t}`); });
page.on('pageerror', e => { errors.push('PAGEERROR ' + e.message); console.log('PAGEERROR', e.message, e.stack); });
await page.addInitScript((b) => { try { localStorage.clear(); localStorage.setItem('weavesim.displayOpen', 'true'); localStorage.setItem('weavesim.backend', JSON.stringify(b)); } catch (e) {} }, backend);
await page.goto(`http://localhost:${port}/web/index.html${process.argv[4] ? '?' + process.argv[4] : ''}`);
const shot = async (name) => { await page.waitForTimeout(300); await page.screenshot({ path: `${out}/d_${name}.png` }); console.log('shot', name); };
const text = (sel) => page.$eval(sel, el => el.textContent);
const select = async (sel, v) => { await page.selectOption(sel, v); await page.waitForTimeout(400); };
await page.waitForFunction(() => document.getElementById('st-topo').textContent.includes('strands'), null, { timeout: 120000 });
console.log('status:', await text('#statusbar'));
await page.waitForTimeout(800);
await shot('01_loaded');

// --- zoom: a wheel event must render right away (it used to wait for the next orbit) ---------------------------------
const zoom = await page.evaluate(async () => {
  const v = window.weavesim.view; let renders = 0; const r0 = v.renderer.render.bind(v.renderer); v.renderer.render = (...a) => { renders++; return r0(...a); };
  await new Promise(r => setTimeout(r, 300)); const before = renders, d0 = v.viewDist();
  const rect = v.canvas.getBoundingClientRect();
  v.canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: -240, clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2, bubbles: true, cancelable: true }));
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  return { renders: renders - before, d0, d1: v.viewDist() };
});
console.log(`zoom: distance ${zoom.d0.toFixed(1)} → ${zoom.d1.toFixed(1)}, frames drawn in 2 rAF after the wheel: ${zoom.renders}`);
if (!(zoom.renders >= 1 && zoom.d1 < zoom.d0)) errors.push('zoom did not render');

// --- run the solver a little ---------------------------------------------------------------------------------------
await page.fill('#nb-step', '1500'); await page.click('#btn-step');
await page.waitForFunction(() => window.weavesim.model && window.weavesim.model.st.iteration >= 1500 && !window.weavesim.busy, null, { timeout: 600000 });
await page.waitForTimeout(1200);
console.log('after step:', await text('#run-text'));
await select('#sel-colour', 'Mx');
console.log('legend:', await text('#legend-title'), '|', await text('#legend-lo'), await text('#legend-hi'));
const lathMat = () => page.evaluate(() => window.weavesim.view.layers.laths.material === window.weavesim.view.matLut ? 'lut' : 'vertex colours');
console.log('lath material:', await lathMat());
await shot('02_Mx_smooth');
await page.click('#ck-bands'); await page.waitForTimeout(400); await shot('03_Mx_bands'); await page.click('#ck-bands');
await page.click('#ck-smooth'); await page.waitForTimeout(400); await shot('04_Mx_segments'); await page.click('#ck-smooth');
await select('#sel-colour', 'Fn'); await shot('05_Fn_knots');
await select('#sel-colour', 'family');
console.log('family legend:', await text('#legend-title'), '|', await page.$eval('#legend-bar', el => [...el.querySelectorAll('span')].map(sp => sp.querySelector('i').style.background + ' ' + sp.textContent).join(' | ')));

// --- hover probe -------------------------------------------------------------------------------------------------------
const latPt = (s, frac = 0.5) => page.evaluate(([s, frac]) => {
  const A = window.weavesim, v = A.view, st = A.model.st, r = v.canvas.getBoundingClientRect();
  const i = st.nodeOffset[s] + Math.floor((st.nodeOffset[s + 1] - st.nodeOffset[s] - 1) * frac), P = v.dispX, R = v.nodeR, h = v.secH[s];
  const j = st.segOffset[s] + Math.min(i - st.nodeOffset[s], st.segOffset[s + 1] - st.segOffset[s] - 1), a = st.segI0[j], b = st.segI1[j];
  const q = [0, 1, 2].map(c => 0.5 * (P[3 * a + c] + P[3 * b + c]) + 0.5 * h * 0.5 * (R[9 * a + 3 + c] + R[9 * b + 3 + c]));
  const V = v.camera.position.clone().set(q[0], q[1], q[2]).project(v.camera);
  return { x: r.left + (V.x + 1) / 2 * r.width, y: r.top + (1 - V.y) / 2 * r.height };
}, [s, frac]);
await select('#sel-colour', 'Mx');
let hp = await latPt(4, 0.37); await page.mouse.move(hp.x, hp.y); await page.waitForTimeout(250); await page.mouse.move(hp.x + 1, hp.y); await page.waitForTimeout(400);
console.log('probe (Mx layer):', await page.$eval('#tooltip', el => (el.hidden ? '(hidden)' : [...el.children].map(c => c.textContent).join(' / '))));
await shot('05b_probe');
await page.click('#diag-chips [data-diag="Vy"]'); await page.waitForTimeout(500);
await page.mouse.move(hp.x, hp.y + 1); await page.waitForTimeout(400);
console.log('probe (Mx + diagram Vy):', await page.$eval('#tooltip', el => (el.hidden ? '(hidden)' : [...el.children].map(c => c.textContent).join(' / '))));
await page.click('#diag-chips [data-diag="none"]'); await select('#sel-colour', 'family');
await page.mouse.move(hp.x + 2, hp.y); await page.waitForTimeout(400);
console.log('probe (no layer):', await page.$eval('#tooltip', el => (el.hidden ? '(hidden)' : [...el.children].map(c => c.textContent).join(' / '))));
await page.mouse.move(5, 5);

// --- diagrams --------------------------------------------------------------------------------------------------------
for (const d of ['Mx', 'defl', 'Fn', 'R', 'curv', 'Vy']) {
  await page.click(`#diag-chips [data-diag="${d}"]`); await page.waitForTimeout(500);
  const n = await page.$$eval('#labels .dlabel', els => els.filter(e => e.style.display !== 'none').length);
  console.log(`diagram ${d}: ${await text('#ro-diagram')} · ${n} labels`);
  if (['Mx', 'defl'].includes(d)) await shot(`06_diagram_${d}`);
}
await page.selectOption('#sel-dlabels', '2'); await page.click('#diag-chips [data-diag="Mx"]'); await page.waitForTimeout(500);
console.log('per-strand labels:', await page.$$eval('#labels .dlabel', els => els.length));
await page.selectOption('#sel-dlabels', '1'); await page.dispatchEvent('#sel-dlabels', 'change');

// --- discretisation --------------------------------------------------------------------------------------------------
await page.evaluate(() => { document.getElementById('mon-disc').open = true; });
await page.evaluate(() => { document.querySelectorAll('#sections > details.sec').forEach(d => { d.open = d.id === 'sec-disc'; }); });
await page.evaluate(() => { const el = document.getElementById('nb-diter'); el.value = 150; el.dispatchEvent(new Event('change', { bubbles: true })); });
await page.click('#btn-disc-run');
await page.waitForFunction(() => !window.weavesim.discRunning && window.weavesim.disc && window.weavesim.disc.done, null, { timeout: 300000 });
await page.waitForTimeout(600);
console.log('discretisation:', (await text('#disc-text')).replace(/\n/g, ' | '), '|', await text('#ro-joints'));
await shot('07_disc');
await select('#sel-colour', 'pieces'); await shot('08_pieces');
await page.click('#ck-zone'); await page.waitForTimeout(500);
console.log('zone diagram:', await text('#ro-diagram'));
await shot('09_zone');
await page.click('#ck-zone'); await page.click('#diag-chips [data-diag="none"]');
await select('#sel-colour', 'family');
await page.evaluate(() => { document.getElementById('mon-disc').open = false; });

// --- lath selection for the groups ------------------------------------------------------------------------------------
await page.evaluate(() => { document.querySelectorAll('#sections > details.sec').forEach(d => { d.open = d.id === 'sec-material'; }); });
const hybrid = await page.$eval('#ck-hybrid', el => el.checked);
if (!hybrid) { await page.click('#ck-hybrid'); await page.waitForTimeout(800); }
await page.click('#btn-lathmode');
const pts = await page.evaluate(() => {
  const A = window.weavesim, v = A.view, st = A.model.st, r = v.canvas.getBoundingClientRect(), out = [];
  for (const s of [2, 9]) {
    const i = st.nodeOffset[s] + Math.floor((st.nodeOffset[s + 1] - st.nodeOffset[s]) / 2) + 1, P = v.dispX, R = v.nodeR, h = v.secH[s];
    const V = v.camera.position.clone().set(P[3 * i] + 0.5 * h * R[9 * i + 3], P[3 * i + 1] + 0.5 * h * R[9 * i + 4], P[3 * i + 2] + 0.5 * h * R[9 * i + 5]).project(v.camera);
    out.push({ s, x: r.left + (V.x + 1) / 2 * r.width, y: r.top + (1 - V.y) / 2 * r.height });
  }
  return out;
});
for (const p of pts) { await page.mouse.click(p.x, p.y); await page.waitForTimeout(300); }
console.log('lath selection:', await text('#ro-assign'), '| button:', await text('#btn-assign'), '| colour:', await page.$eval('#sel-colour', el => el.value));
await shot('10_lath_selection');
// a click on the carrier inside the dome (not a lath) clears: nothing highlighted, no tube
const blank = await page.evaluate(() => { const v = window.weavesim.view, r = v.canvas.getBoundingClientRect(); for (let y = 0.45; y < 0.8; y += 0.01) for (let x = 0.4; x < 0.6; x += 0.01) { const px = r.left + x * r.width, py = r.top + y * r.height, p = v.pick(px, py); if (p && p.kind === 'mesh') return { x: px, y: py }; } return null; });
if (blank) { await page.mouse.click(blank.x, blank.y); await page.waitForTimeout(300); }
const cleared = await page.evaluate(() => ({ sel: window.weavesim.lathSel.size, view: window.weavesim.view.strandSel.size, tube: !!window.weavesim.view.layers.path }));
console.log('click on the carrier:', JSON.stringify(cleared), await text('#ro-assign'));
if (cleared.sel || cleared.view || cleared.tube) errors.push('lath selection not cleared by a click on the carrier');
for (const p of pts) { await page.mouse.click(p.x, p.y); await page.waitForTimeout(300); }
await page.keyboard.press('Escape'); await page.waitForTimeout(300);
const afterEsc = await page.evaluate(() => ({ mode: window.weavesim.mode, sel: window.weavesim.lathSel.size, tube: !!window.weavesim.view.layers.path, colour: document.getElementById('sel-colour').value }));
console.log('after Esc:', JSON.stringify(afterEsc));
if (afterEsc.sel || afterEsc.tube || afterEsc.mode !== 'none') errors.push('lath selection still highlighted after leaving the mode');
await shot('10b_after_esc');

// --- Rhino export ----------------------------------------------------------------------------------------------------
await page.evaluate(() => { document.querySelectorAll('#sections > details.sec').forEach(d => { d.open = d.id === 'sec-file'; }); });
const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.click('#btn-3dm')]);
const file = path.join(out, dl.suggestedFilename()); await dl.saveAs(file);
await page.waitForFunction(() => /objects|failed/.test(document.getElementById('ro-3dm').textContent), null, { timeout: 120000 });
console.log('rhino:', await text('#ro-3dm'), '→', file, fs.statSync(file).size, 'bytes');

// --- sidebar: fold, dock left ------------------------------------------------------------------------------------------
await page.click('#display-toggle'); await page.waitForTimeout(400); await shot('11_sidebar_folded');
await page.click('#display-toggle'); await page.click('#display-dock'); await page.waitForTimeout(400); await shot('12_sidebar_left');
await page.click('#display-dock');
console.log('ERRORS:', errors.length ? '\n' + errors.join('\n') : 'none');
await browser.close(); server.close();
