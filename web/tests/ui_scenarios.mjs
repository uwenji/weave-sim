// UI scenarios (headless Chromium): backend switching, lath groups on the GPU, fastened protocol (Zero → settle →
// Lock, the Lock toggle) + delta, live edits,
// unit change, flip mode, load / anchor editing, UTM csv + replay, save / reopen a scene
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
import { ROOT, REF, launchChromium } from './paths.mjs';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.csv': 'text/csv', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => { let p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)); if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(res); }).listen(0);
const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const errors = [];
page.on('console', m => { const t = m.text(); if (m.type() === 'error' && !t.includes('ERR_TUNNEL') && !t.includes('status of 404')) { errors.push(t); console.log('CONSOLE ERROR', t); } });
page.on('pageerror', e => { errors.push('PAGEERROR ' + e.message); console.log('PAGEERROR', e.message, e.stack); });
await page.addInitScript(() => { try { localStorage.setItem('weavesim.displayOpen', 'true'); localStorage.setItem('weavesim.backend', '"cpu"'); } catch (e) {} });
await page.goto(`http://localhost:${server.address().port}/web/index.html`);
const W = (fn, arg) => page.evaluate(fn, arg);
const status = () => W(() => document.getElementById('st-msg').textContent + ' | ' + document.getElementById('st-run').textContent);
await page.waitForFunction(() => document.getElementById('st-topo').textContent.includes('strands'), null, { timeout: 60000 });
await W(() => { document.querySelectorAll('#sections > details.sec').forEach(d => { d.open = true; }); });
const ok = (c, msg) => { console.log((c ? 'ok   ' : 'FAIL ') + msg); if (!c) errors.push('FAIL ' + msg); };
// 1 run on the CPU, switch to the GPU mid-way, run on, switch back
await page.fill('#nb-step', '300'); await page.click('#btn-step'); await page.waitForTimeout(500);
let it = await W(() => window.weavesim.model.st.iteration); ok(it === 300, `CPU step 300 → iteration ${it}`);
const X1 = await W(() => Array.from(window.weavesim.model.st.X.slice(0, 6)));
await page.click('#st-backend'); await page.waitForFunction(() => window.weavesim.backend === 'webgpu' && window.weavesim.gpu, null, { timeout: 60000 });
await page.fill('#nb-step', '100'); await page.click('#btn-step'); await page.waitForFunction(() => window.weavesim.model.st.iteration >= 400, null, { timeout: 120000 });
it = await W(() => window.weavesim.model.st.iteration); ok(it === 400, `GPU continues from the CPU state: iteration ${it}`);
await page.click('#st-backend'); await page.waitForFunction(() => window.weavesim.backend === 'cpu' && !window.weavesim.gpu, null, { timeout: 60000 });
await page.click('#btn-step'); await page.waitForTimeout(800);
it = await W(() => window.weavesim.model.st.iteration); ok(it === 500, `back on the CPU: iteration ${it}`);
// 2 load-test protocol (Monitor run bar): Zero → settle → the solver pauses (crossings untouched); a click while it
//   settles stops it; the 4 Contact box locks / unlocks; then a load at a selected frame, a step, the delta layer
const PS = () => W(() => { const A = window.weavesim, P = A.protocol, st = A.model.st, bi = document.getElementById('btn-zero');
  let d = 0; for (let i = 0; i < 3 * st.n; i++) d = Math.max(d, Math.abs(st.X[i] - st.X0[i]));
  let sl = 0; for (const v of st.slip) sl = Math.max(sl, Math.abs(v));
  return { it: st.iteration, running: A.running, phase: P.phase && { ...P.phase }, last: P.last && { ...P.last }, phases: P.phases.map(p => `${p.kind} ${p.i0}-${p.end}`),
    locked: A.model.contact.locked.every(Boolean), any: A.model.contact.locked.some(Boolean), on: bi.classList.contains('on'), fill: bi.style.getPropertyValue('--p'),
    box: document.getElementById('ck-locked').checked, d, sl, text: document.getElementById('run-text').textContent, win: A.chartWin }; });
let it0 = await W(() => window.weavesim.model.st.iteration);
await page.fill('#nb-zero-steps', '100000'); await page.dispatchEvent('#nb-zero-steps', 'change');
await page.click('#btn-zero'); await page.waitForTimeout(500);
let ps = await PS();
ok(ps.phase && ps.phase.i0 === it0 && ps.phase.i1 === it0 + 100000 && ps.running && ps.on && /zero · settle \d+ \/ 100000/.test(ps.text),
  `settle running from it ${it0}: button lit (fill ${ps.fill}), line "${ps.text.split('converged').pop().split('·').slice(-3).join('·').trim()}"`);
ok(ps.win && ps.win.bands.some(b => b.startsWith('zero ' + it0)) && ps.win.xmax >= it0 + 100000, `chart: bands ${JSON.stringify(ps.win?.bands)} · x to ${ps.win?.xmax}`);
await page.click('#btn-zero'); await page.waitForFunction(() => !window.weavesim.protocol.phase && !window.weavesim.running && !window.weavesim.busy, null, { timeout: 60000 });
ps = await PS();
ok(ps.last && (ps.last.stopped || ps.last.conv) && !ps.any, `a second click stops the settle at ${ps.last?.end - ps.last?.i0} it (stopped ${!!ps.last?.stopped}, crossings free)`);
it0 = ps.it;
await page.fill('#nb-zero-steps', '120'); await page.dispatchEvent('#nb-zero-steps', 'change');
await page.click('#btn-zero');
await page.waitForFunction(() => !window.weavesim.protocol.phase && !window.weavesim.running && !window.weavesim.busy && window.weavesim.protocol.last && window.weavesim.protocol.last.i0 >= 0 && !window.weavesim.protocol.last.stopped, null, { timeout: 120000 });
ps = await PS();
ok(ps.last.i0 === it0 && ps.last.end === ps.it && ps.it <= it0 + 120 && ps.it > it0, `settled ${ps.it - it0} it (planned 120${ps.last.conv ? ', converged' : ''}) and paused`);
ok(!ps.any && !ps.box, 'end of the settle: the crossings are untouched (free), 4 Contact box unticked');
ok(ps.d === 0 && ps.sl === 0, `the settled state is the reference: max |X − X0| ${ps.d}, slip ${ps.sl}`);
ok(/reference at it \d+ · settle drift/.test(ps.text) && ps.text.includes('add the load, then Run'), `line: …${ps.text.slice(-110)}`);
ok(ps.phases.join(',').includes(`zero ${it0}-${ps.it}`) && !ps.phases.some(p => p.startsWith('locked')), `phases ${ps.phases.join(' | ')}`);
await W(() => { document.getElementById('sec-contact').open = true; });
await page.check('#ck-locked'); await page.waitForTimeout(400);
ps = await PS(); ok(ps.locked && ps.box && ps.phases.includes(`locked ${ps.it}-null`), '4 Contact box ticked: every crossing locked, locked band opens');
await page.uncheck('#ck-locked'); await page.waitForTimeout(400);
ps = await PS(); ok(!ps.any && !ps.box, '4 Contact box unticked: crossings free');
await page.check('#ck-locked'); await page.waitForTimeout(400);
ps = await PS(); ok(ps.locked && ps.box, 'ticked: locked again');
await W(() => { document.getElementById('sec-loads').open = true; });        // Setup is an accordion: one section open
await W(() => { window.weavesim.selection = [[11, 5]]; });
await page.selectOption('#sel-loaddir', 'world'); await page.fill('#nb-lmag', '200');
await page.click('#btn-addload'); await page.waitForTimeout(300);
ok(await W(() => window.weavesim.scene.loads.point.length === 1), 'point load added to the selection');
await page.fill('#nb-step', '600'); await page.click('#btn-step'); await page.waitForTimeout(2500);
await page.selectOption('#sel-colour', 'disp'); await page.check('#ck-delta'); await page.waitForTimeout(800);
const legend = await W(() => document.getElementById('legend-title').textContent);
ok(legend.startsWith('Δ displacement'), `delta legend: ${legend}`);
// 3 live edit: E doubled → rebuildGoals without reset (iteration continues)
it = await W(() => window.weavesim.model.st.iteration);
await W(() => { document.getElementById('sec-material').open = true; });
const E0 = await W(() => window.weavesim.scene.material.E);
await page.fill('#nb-E', String(2 * E0)); await page.dispatchEvent('#nb-E', 'change');
await page.fill('#nb-step', '50'); await page.click('#btn-step'); await page.waitForTimeout(800);
const it2 = await W(() => window.weavesim.model.st.iteration);
ok(it2 === it + 50 && (await W(() => window.weavesim.model.stiff.EA[0])) > 0, `live E edit kept the state (iteration ${it} → ${it2}) · ${await W(() => document.getElementById('st-msg').textContent)}`);
// 4 units: cm → mm, values converted, model rebuilt
const w0 = await W(() => window.weavesim.scene.section.width);
await page.selectOption('#sel-len', 'mm'); await page.waitForTimeout(1500);
const w1 = await W(() => window.weavesim.scene.section.width), E1 = await W(() => window.weavesim.scene.material.E);
ok(Math.abs(w1 - 10 * w0) < 1e-9 && Math.abs(E1 - 2 * E0 / 100) < 1e-6 * E1, `units cm → mm: width ${w0} → ${w1}, E ${2 * E0} → ${E1}`);
// 5 lath groups example on the GPU
await page.click('#st-backend'); await page.waitForFunction(() => window.weavesim.backend === 'webgpu', null, { timeout: 60000 });
await W(() => { document.getElementById('sec-file').open = true; });
await page.selectOption('#sel-example', '../examples/scenes/dualv5dome_cm_hybrid.json');
await page.waitForFunction(() => window.weavesim.sceneName === 'dualv5dome_cm_hybrid' && window.weavesim.model && window.weavesim.model.st.iteration === 0, null, { timeout: 120000 });
console.log('   status after load:', await W(() => document.getElementById('st-msg').textContent));
ok(await W(() => window.weavesim.scene.hybrid.enabled && window.weavesim.scene.nGroups > 1), `lath groups scene: ${await W(() => window.weavesim.scene.nGroups)} groups`);
await page.fill('#nb-step', '600'); await page.click('#btn-step'); await page.waitForFunction(() => window.weavesim.model.st.iteration >= 600, null, { timeout: 600000 });
const res = await W(() => window.weavesim.model.st.residual);
ok(Number.isFinite(res), `GPU run on lath groups: residual ${res}`);
// compare with the CPU for the same 200 iterations
const cmp = await W(async () => {
  const { CpuSolver } = await import('/web/js/solver_cpu.js');
  const A = window.weavesim, cpu = new CpuSolver(A.scene, A.topo); cpu.step(600);
  let dx = 0, s = 0; for (let i = 0; i < cpu.st.X.length; i++) { dx = Math.max(dx, Math.abs(cpu.st.X[i] - A.model.st.X[i])); s = Math.max(s, Math.abs(cpu.st.X[i] - cpu.st.X0[i])); }
  return { dx, disp: s };
});
ok(cmp.dx < 1e-3 * cmp.disp + 1e-9, `lath groups GPU vs CPU after 600 it: |ΔX| ${cmp.dx.toExponential(2)} (displacement ${cmp.disp.toExponential(2)})`);
// 6 flip a crossing (rebuilds), save the scene, reopen it from the saved json
await page.click('#st-backend'); await page.waitForFunction(() => window.weavesim.backend === 'cpu', null, { timeout: 60000 });
await W(() => { document.getElementById('sec-weave').open = true; });
await page.click('#btn-flipmode');
const pt = await W(() => { const v = window.weavesim.view, P = v.dispX, c = 10, ia = v.knotNodes[2 * c], ib = v.knotNodes[2 * c + 1]; const r = v.canvas.getBoundingClientRect(); const V = v.camera.position.clone().set(0.5 * (P[3 * ia] + P[3 * ib]), 0.5 * (P[3 * ia + 1] + P[3 * ib + 1]), 0.5 * (P[3 * ia + 2] + P[3 * ib + 2])).project(v.camera); return { x: r.left + (V.x + 1) / 2 * r.width, y: r.top + (1 - V.y) / 2 * r.height }; });
await page.mouse.click(pt.x, pt.y); await page.waitForTimeout(1500);
const flips = await W(() => window.weavesim.scene.weave.flip_knots.length);
ok(flips === 1, `flip mode: ${flips} flip(s) · ${await W(() => document.getElementById('st-msg').textContent)}`);
await page.keyboard.press('Escape');
await W(() => { document.getElementById('sec-file').open = true; });
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#btn-save')]);
const saved = path.join(REF, 'saved_scene.json'); await dl.saveAs(saved);
const sj = JSON.parse(fs.readFileSync(saved, 'utf8'));
ok(sj.version === 1 && sj.weave.flip_knots.length === 1 && sj.hybrid.enabled, `saved scene: version ${sj.version}, flips ${sj.weave.flip_knots.length}, groups ${sj.hybrid.groups.length + 1}`);
// 7 UTM: probe + csv + replay (CPU, short protocol)
await page.selectOption('#sel-example', '../examples/calibration/force/dome_calib_scene.json');
await page.waitForFunction(() => window.weavesim.sceneName === 'dome_calib_scene' && window.weavesim.model && window.weavesim.model.st.iteration === 0, null, { timeout: 120000 });
await W(() => { window.weavesim.scene.utm.probe = { frame: [11, 5], axis: [0, 0, -1] }; });
await page.setInputFiles('#file-utm', path.join(ROOT, 'examples/calibration/force/P1.csv')); await page.waitForTimeout(800);
ok(await W(() => !!window.weavesim.utm && window.weavesim.utm.length === 41), `UTM csv: ${await W(() => document.getElementById('ro-utmfile').textContent)}`);
await W(() => { const set = (k, v) => { const el = document.querySelector(`#grid-replay [data-rp="${k}"]`); el.value = v; el.dispatchEvent(new Event('change')); }; set('relax_iter', 300); set('step_iter', 60); set('avg_iter', 20); set('n_steps', 2); set('cycles', 1); });
await W(() => { document.getElementById('sec-utm').open = true; });
await page.click('#btn-replay');
const tk = setInterval(async () => console.log('   …', await W(() => document.getElementById('ro-utm').textContent + ' | ' + JSON.stringify(window.weavesim.scene.utm.probe))), 10000);
await page.waitForFunction(() => /replay .* s on|failed|stopped/.test(document.getElementById('ro-utm').textContent), null, { timeout: 300000 });
clearInterval(tk);
ok((await W(() => document.getElementById('ro-utm').textContent)).includes('residual'), `UTM replay: ${await W(() => document.getElementById('ro-utm').textContent)}`);
await page.screenshot({ path: path.join(REF, 'ui', '08_utm.png') });
console.log('ERRORS:', errors.length ? '\n' + errors.join('\n') : 'none');
await browser.close(); server.close();
