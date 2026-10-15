// Fastened-structure protocol on the WebGPU solver (headless Chromium, SwiftShader): relax → Zero (settle) →
// the crossings lock on the host and the GPU, the settled state is the reference on both, the solver pauses; a load and
// a run give a delta; strip screenshots (settling, locked, loaded) for a look at the bands and the buttons.
// node tests/ui_protocol.mjs [outdir]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
import { ROOT, launchChromium } from './paths.mjs';
const out = process.argv[2] || null;
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.csv': 'text/csv', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => { const p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); }).listen(0);
const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
const errors = [];
page.on('console', m => { const t = m.text(); if (m.type() === 'error' && !t.includes('ERR_TUNNEL') && !t.includes('status of 404')) { errors.push(t); console.log('CONSOLE ERROR', t); } });
page.on('pageerror', e => { errors.push('PAGEERROR ' + e.message); console.log('PAGEERROR', e.message, e.stack); });
await page.addInitScript(() => { try { localStorage.setItem('weavesim.backend', '"webgpu"'); localStorage.setItem('weavesim.theme', '"light"'); } catch (e) {} });
await page.goto(`http://localhost:${server.address().port}/web/index.html`);
const W = (fn, arg) => page.evaluate(fn, arg);
const ok = (c, msg) => { console.log((c ? 'ok   ' : 'FAIL ') + msg); if (!c) errors.push('FAIL ' + msg); };
const shot = async (name) => { if (out) { fs.mkdirSync(out, { recursive: true }); await page.locator('#strip').screenshot({ path: path.join(out, name) }); } };
await page.waitForFunction(() => document.getElementById('st-topo').textContent.includes('strands') && window.weavesim.gpu, null, { timeout: 120000 });
ok(await W(() => window.weavesim.backend === 'webgpu'), 'WebGPU backend');
// relax
await page.fill('#nb-step', '400'); await page.click('#btn-step');
await page.waitForFunction(() => window.weavesim.model.st.iteration >= 400 && !window.weavesim.busy, null, { timeout: 300000 });
const i0 = await W(() => window.weavesim.model.st.iteration);
// Zero with a settle long enough to catch it running
await page.fill('#nb-zero-steps', '600'); await page.dispatchEvent('#nb-zero-steps', 'change');
await page.click('#btn-zero');
await page.waitForFunction((a) => window.weavesim.model.st.iteration >= a + 250 || !window.weavesim.protocol.phase, i0, { timeout: 300000 });
const mid = await W(() => ({ on: document.getElementById('btn-zero').classList.contains('on'), fill: document.getElementById('btn-zero').style.getPropertyValue('--p'), phase: !!window.weavesim.protocol.phase, text: document.getElementById('strip-text').textContent }));
ok(!mid.phase || (mid.on && parseFloat(mid.fill) > 0), `settling on the GPU: button lit, fill ${mid.fill} · ${mid.text.split('N·cm').pop()}`);
await shot('strip_settling.png');
await page.waitForFunction(() => !window.weavesim.protocol.phase && !window.weavesim.running && !window.weavesim.busy, null, { timeout: 300000 });
await page.waitForTimeout(300);
await shot('strip_locked.png');
// host and GPU agree: reference = settled state, slip zero, every crossing locked
const r = await W(async () => {
  const A = window.weavesim, st = A.model.st, P = A.protocol;
  const host = { it: st.iteration, X: Array.from(st.X), X0: Array.from(st.X0), slip: Array.from(st.slip), lock: Array.from(A.model.contact.locked) };
  await A.gpu.download(0);                                      // read the GPU state back over the host copy
  const dev = { X: Array.from(st.X), X0: Array.from(st.X0), slip: Array.from(st.slip), lock: Array.from(A.model.contact.locked) };
  let dX = 0, dX0 = 0, dRef = 0; for (let i = 0; i < host.X.length; i++) { dX = Math.max(dX, Math.abs(host.X[i] - dev.X[i])); dX0 = Math.max(dX0, Math.abs(host.X0[i] - dev.X0[i])); dRef = Math.max(dRef, Math.abs(dev.X[i] - dev.X0[i])); }
  return { it: host.it, last: { ...P.last }, phases: P.phases.map(p => `${p.kind} ${p.i0}-${p.end}`), dX, dX0, dRef, slipH: Math.max(0, ...host.slip.map(Math.abs)), slipD: Math.max(0, ...dev.slip.map(Math.abs)),
    lockH: host.lock.every(Boolean), lockD: dev.lock.every(Boolean), led: document.getElementById('btn-lock').getAttribute('aria-pressed'), text: document.getElementById('strip-text').textContent, win: A.chartWin };
});
ok(r.last.i0 === i0 && r.it === r.last.end && r.it <= i0 + 600, `settle ${r.last.i0} → ${r.last.end} (${r.last.conv ? 'converged' : 'planned end'}), drift ${r.last.drift.toPrecision(3)} · paused`);
ok(r.lockH && r.lockD && r.led === 'true', 'every crossing locked on the host and on the GPU · Lock light on');
ok(r.dRef === 0 && r.dX0 === 0 && r.slipH === 0 && r.slipD === 0, `the GPU reference is the settled state: max |x − x0| ${r.dRef}, host/GPU x0 ${r.dX0}, slip ${r.slipD}`);
ok(r.win.bands.join('|').includes(`zero ${i0}–${r.it}`) && r.win.vlines.includes(r.it), `chart bands ${r.win.bands.join(' | ')} · dashed at ${r.win.vlines}`);
ok(r.text.includes('still moving') === r.last.drift > 0.1 * (await W(() => window.weavesim.scene.section.height)), `drift flag (relaxed only 400 it): …${r.text.split('N·cm').pop()}`);
// a load and a run: the crossings stay locked, the delta is the load displacement
await W(() => { window.weavesim.selection = [[11, 5]]; });
await W(() => { document.getElementById('sec-loads').open = true; });
await page.selectOption('#sel-loaddir', 'world'); await page.fill('#nb-lmag', '300'); await page.click('#btn-addload'); await page.waitForTimeout(200);
await page.fill('#nb-step', '400'); await page.click('#btn-step');
await page.waitForFunction((a) => window.weavesim.model.st.iteration >= a + 400 && !window.weavesim.busy, r.it, { timeout: 300000 });
await page.waitForTimeout(300);
await shot('strip_loaded.png');
const l = await W(() => { const A = window.weavesim, st = A.model.st; let d = 0; for (let i = 0; i < st.X.length; i++) d = Math.max(d, Math.abs(st.X[i] - st.X0[i])); let s = 0; for (const v of st.slip) s = Math.max(s, Math.abs(v));
  return { d, s, lock: A.model.contact.locked.every(Boolean), bands: A.chartWin.bands, text: document.getElementById('strip-text').textContent }; });
ok(l.lock && l.d > 0, `loaded: crossings still locked, max displacement from the reference ${l.d.toPrecision(3)}, slip ${l.s.toPrecision(3)}`);
ok(l.bands.some(b => b.startsWith('locked ')) && l.text.includes('delta from here'), `bands ${l.bands.join(' | ')} · …${l.text.split('N·cm').pop()}`);
// the dark theme draws too
await W(() => { document.documentElement.setAttribute('data-theme', 'dark'); window.dispatchEvent(new Event('resize')); });
await page.waitForTimeout(300); await shot('strip_loaded_dark.png');
console.log(errors.length ? 'ERRORS:\n' + errors.join('\n') : 'ALL OK');
await browser.close(); server.close();
