// Where the Node checks find the WeaveSim examples and the Python reference dumps (tests/python/*.py write them)
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = process.env.WEAVESIM_ROOT || path.resolve(here, '..', '..');          // the WeaveSim folder
export const EXAMPLES = path.join(ROOT, 'examples');
export const REF = process.env.WEAVESIM_REF || path.join(here, 'ref_out');                // Python reference dumps

/** headless Chromium for the browser checks: Playwright from node_modules (npm i -D playwright) or the sandbox copy;
 *  on Linux without a GPU the SwiftShader WebGPU adapter is used */
export async function launchChromium() {
  const fs = await import('node:fs');
  let pw;
  try { pw = await import('playwright'); } catch (e) { pw = await import('/opt/npm-tools/node_modules/playwright/index.mjs'); }
  const sandboxChrome = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const executablePath = process.env.CHROME_PATH || (fs.existsSync(sandboxChrome) ? sandboxChrome : undefined);
  const linux = process.platform === 'linux';
  const args = ['--enable-unsafe-webgpu'].concat(linux ? ['--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface', '--use-angle=swiftshader'] : []);
  return pw.chromium.launch({ headless: true, executablePath, args });
}
