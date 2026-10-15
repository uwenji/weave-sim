// Headless UI check (SwiftShader WebGPU): load the app, run, switch layers, open the calibration example; screenshots
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, EXAMPLES, REF, launchChromium } from './paths.mjs';
// serves the WeaveSim folder: /web/… and /examples/…
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
const steps = (process.argv[3] || 'load,run,layers,calib').split(',');
const backend = process.argv[4] || 'webgpu';
const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const errors = [];
page.on('console', m => { const t = m.text(); if (t.includes('experimental on this platform')) return; if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${t}`); console.log(`[${m.type()}] ${t}`); });
page.on('pageerror', e => { errors.push('PAGEERROR ' + e.message); console.log('PAGEERROR', e.message, e.stack); });
await page.addInitScript((b) => { try { localStorage.setItem('weavesim.backend', JSON.stringify(b)); } catch (e) {} }, backend);
await page.goto(`http://localhost:${port}/web/index.html${process.argv[5] ? '?' + process.argv[5] : ''}`);
const shot = async (name) => { await page.screenshot({ path: `${out}/${name}.png` }); console.log('shot', name); };
const text = (sel) => page.$eval(sel, el => el.textContent);
await page.waitForFunction(() => document.getElementById('st-topo').textContent.includes('strands'), null, { timeout: 120000 });
console.log('status:', await text('#statusbar'));
if (steps.includes('load')) { await page.waitForTimeout(800); await shot('01_loaded'); }
if (steps.includes('run')) {
  await page.click('#btn-run');
  await page.waitForTimeout(+(process.env.RUN_MS || 20000));
  await page.click('#btn-run');
  await page.waitForTimeout(1500);
  console.log('after run:', await text('#statusbar'), '|', await text('#strip-text'));
  await shot('02_run');
}
if (steps.includes('layers')) {
  for (const layer of ['Mx', 'Fn', 'state', 'disp']) {
    await page.selectOption('#sel-colour', layer);
    await page.waitForTimeout(1200);
    console.log(layer, '→', (await text('#ro-extremes')).split('\n')[0], '| legend:', await text('#legend-title'));
  }
  await page.selectOption('#sel-colour', 'Mx'); await page.waitForTimeout(800);
  await shot('03_layer_Mx');
}
if (steps.includes('calib')) {
  await page.evaluate(() => { document.querySelectorAll('details.sec').forEach(d => { d.open = d.id === 'sec-calib'; }); });
  await page.click('#cal-example');
  await page.waitForFunction(() => !document.getElementById('cal-main').hidden, null, { timeout: 120000 });
  await page.waitForTimeout(1500);
  console.log('calib:', await text('#cal-ro'));
  await shot('04_calib');
}
if (steps.includes('calibrun')) {
  // a short protocol so the software GPU finishes: objective at the table values, then a tiny CMA-ES fit
  await page.evaluate(() => {
    const set = (k, v) => { const el = document.querySelector(`[data-crp="${k}"]`); el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); };
    set('relax_iter', 400); set('step_iter', 80); set('avg_iter', 20); set('n_steps', 2); set('cycles', 2);
  });
  await page.click('#cal-obj');
  await page.waitForFunction(() => /objective (done|failed|stopped)/.test(document.getElementById('cal-status').textContent), null, { timeout: 900000 });
  console.log('objective:', await text('#cal-status'), '|', (await text('#cal-log')).trim().split('\n').pop());
  await page.evaluate(() => { document.getElementById('cal-maxeval').value = 6; document.getElementById('cal-ninit').value = 3; });
  if (process.env.FIT) {
    await page.click('#cal-fit');
    await page.waitForFunction(() => /fit (done|failed|stopped)/.test(document.getElementById('cal-status').textContent), null, { timeout: 3000000 });
    console.log('fit:', await text('#cal-status'));
    console.log((await text('#cal-log')).trim().split('\n').slice(-4).join('\n'));
    console.log('best:', await text('#cal-best'));
  }
  await page.evaluate(() => { const p = document.getElementById('panel'); p.scrollTop = document.getElementById('cal-params').offsetTop - 60; });
  await page.waitForTimeout(500);
  await shot('06_calib_params');
  await page.evaluate(() => { const p = document.getElementById('panel'); p.scrollTop = document.getElementById('cal-plots').offsetTop - 200; });
  await page.waitForTimeout(500);
  await shot('07_calib_plots');
}
if (steps.includes('pick')) {
  // screen position of crossing 40 (mid of its two frame nodes)
  const box = await page.evaluate(() => {
    const A = window.weavesim, v = A.view, st = A.model.st, c = 40, P = v.dispX;
    const ia = v.knotNodes[2 * c], ib = v.knotNodes[2 * c + 1];
    const p = { x: 0.5 * (P[3 * ia] + P[3 * ib]), y: 0.5 * (P[3 * ia + 1] + P[3 * ib + 1]), z: 0.5 * (P[3 * ia + 2] + P[3 * ib + 2]) };
    const r = v.canvas.getBoundingClientRect();
    const V = v.camera.position.clone().set(p.x, p.y, p.z).project(v.camera);
    return { x: r.left + (V.x + 1) / 2 * r.width, y: r.top + (1 - V.y) / 2 * r.height };
  });
  await page.mouse.click(box.x, box.y);
  await page.waitForTimeout(500);
  console.log('selection:', (await page.$$eval('.sel-readout', els => els.map(e => e.textContent)))[0]);
  await page.mouse.move(box.x + 3, box.y + 2); await page.waitForTimeout(500);
  console.log('tooltip:', await page.$eval('#tooltip', el => (el.hidden ? '(hidden)' : el.textContent)));
  await shot('05_pick');
}
console.log('ERRORS:', errors.length ? '\n' + errors.join('\n') : 'none');
await browser.close(); server.close();
