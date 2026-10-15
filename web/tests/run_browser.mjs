// Headless Chromium (SwiftShader WebGPU) runner for the browser checks: node tests/run_browser.mjs <page> [query]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, EXAMPLES, REF, launchChromium } from './paths.mjs';
// serves the WeaveSim folder: /web/… and /examples/…
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.stl': 'application/octet-stream', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const p = path.join(ROOT, u);
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': types[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
  fs.createReadStream(p).pipe(res);
}).listen(0);
const port = server.address().port;
const page_ = process.argv[2] || 'web/tests/gpu_check.html', query = process.argv[3] || '';
const timeout = +(process.argv[4] || 600000);
const browser = await launchChromium();
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (!t.includes('experimental on this platform')) console.log(t); });
page.on('pageerror', e => console.log('PAGEERROR', e.message));
await page.goto(`http://localhost:${port}/${page_}${query ? '?' + query : ''}`);
try { await page.waitForFunction(() => window.__done === true, null, { timeout }); } catch (e) { console.log('TIMEOUT'); }
await browser.close(); server.close();
