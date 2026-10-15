// Batched dynamic relaxation on WebGPU. B instances of one topology run in the same dispatches (grid y = instance):
// one interactive model (B = 1), or a calibration population × UTM tests. The host-side model of every instance is a
// CpuSolver (setup, rebuilds after scene edits, results): upload() copies its state and parameters to the GPU,
// download() copies the GPU state back into it, so CpuSolver.results() / rebuildGoals() work on the GPU state.
// See wgsl.js for the kernels and the precision scheme.
import { solverWGSL } from './wgsl.js';

const WG = 64, HN = 4096, NOUT = 64, NCT = 32, NIP = 108;
const TWO32 = 4294967296;

export async function getDevice() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const want = { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize };
    const device = await adapter.requestDevice({ requiredLimits: want });
    try { device.wsInfo = adapter.info || {}; } catch (e) { /* read-only on some builds */ }
    device.lost.then((e) => console.warn('WebGPU device lost:', e.message));
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', e.error ? e.error.message : e));   // a rejected command buffer is otherwise silent
    return device;
  } catch (e) { console.warn('WebGPU unavailable:', e); return null; }
}

const ceilLog2 = (x) => Math.ceil(Math.log2(Math.max(x, 1e-30)));
// compiled pipelines per device and shader source (a calibration builds many solvers of the same sizes)
const PIPES = new WeakMap();
const clampi = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** integer of x · 2^bits (x f64) as [lo, hi] u32 words (two's complement) */
function toI64(x, bits) {
  const v = Math.round(x * 2 ** bits);                        // power-of-two scaling is exact; |v| < 2^63
  let big = BigInt(v);
  if (big < 0n) big += 1n << 64n;
  return [Number(big & 0xffffffffn), Number((big >> 32n) & 0xffffffffn)];
}
/** [lo, hi] → f64 value · 2^-bits */
function fromI64(lo, hi, bits) {
  const neg = hi >= 0x80000000;
  let v;
  if (neg) { const l = (~lo >>> 0) + 1, carry = l >= TWO32 ? 1 : 0; v = -(((~hi >>> 0) + carry) * TWO32 + (l % TWO32)); }
  else v = hi * TWO32 + lo;
  return v * 2 ** -bits;
}
/** exact integer difference (a − b) of two [lo, hi] pairs as f64 · 2^-bits */
function diffI64(alo, ahi, blo, bhi, bits) {
  let lo = alo - blo, borrow = 0;
  if (lo < 0) { lo += TWO32; borrow = 1; }
  let hi = (ahi - bhi - borrow) >>> 0;
  return fromI64(lo >>> 0, hi, bits);
}

export class GpuSolver {
  /** models: CpuSolver[] sharing one topology (the first defines the sizes); opts.perSubmit: iterations per command
   *  buffer; opts.scales: fixed-point scales {PB, FB, TB, KB} (default: from the models, see computeScales) */
  constructor(device, models, opts = {}) {
    this.dev = device; this.models = models; this.B = models.length; this.backend = 'webgpu';
    this.perSubmit = opts.perSubmit || 64;
    const m0 = models[0], st = m0.st;
    const N = st.n, M = st.m, P = st.p, K = m0.contact ? st.K : 0, C = m0.anchors.clamps.length, B = this.B;
    for (const md of models) if (md.st.n !== N || md.st.m !== M || (md.contact ? md.st.K : 0) !== K || md.anchors.clamps.length !== C) throw new Error('GpuSolver: all instances must share one topology, contact setting and anchors');
    this.sizes = { N, M, P, K, C };
    Object.assign(this, opts.scales || GpuSolver.computeScales(models));
    // ---- layout
    const L = { N, M, P, C, K, HN, NOUT };
    L.NWG_P = Math.ceil((M + N) / WG); L.NWG_F = Math.ceil((P + C + K) / WG); L.NWG_N = Math.ceil(N / WG); L.NWG_S = Math.ceil(M / WG);
    L.NWG = Math.max(1, L.NWG_P, L.NWG_F, L.NWG_N, L.NWG_S);
    const alloc = (obj, spec) => { let o = 0; const off = {}; for (const [k, n] of spec) { off[k] = o; o += Math.max(n, 0); } obj.size = Math.max(o, 4); return off; };
    L.t = alloc(L.tS = {}, [['seg0', M], ['seg1', M], ['j0', P], ['j1', P], ['jn', P], ['cn', C], ['cb', C], ['cs', C], ['nsa', N], ['nsb', N], ['koff', 2 * K], ['knseg', 2 * K], ['ken', K]]);
    L.g = alloc(L.gS = {}, [['l0', M], ['jlen', P], ['clen', C], ['wk', 2 * K], ['hk', 2 * K], ['kw', K], ['kh', K],
      ['stiff', B * M * 4], ['om', B * P * 3], ['mass', B * N], ['kseg', B * M], ['lc', B * N * 3], ['ll', B * N * 3], ['cei', B * C * 2], ['cdir', B * C * 3],
      ['fc90', B * K * 2], ['e90', B * K * 2], ['hs', B * K * 2]]);
    L.i = alloc(L.iS = {}, [['x', B * N * 6], ['xp', B * N * 6], ['x0', B * N * 6], ['eref', B * M * 6], ['u', B * K * 4], ['slip', B * K * 2], ['fix', B * N], ['tfix', B * M], ['lock', B * K], ['kout', B * K], ['probe0', B * 6], ['q', B * M * 8], ['qp', B * M * 8]]);
    L.f = alloc(L.fS = {}, [['v', B * N * 3], ['w', B * M], ['nprev', B * K * 3], ['lam', B * K], ['th0', B * K], ['rslip', B * K],
      ['dent', B * K * 2], ['dente', B * K], ['climb', B * K * 2], ['tanpsi', B * K * 2], ['fn', B * K], ['ft', B * K], ['gap', B * K], ['phi', B * K * 2], ['theta', B * K],
      ['anom', B * K], ['pres', B * K], ['util', B * K], ['mrot', B * K], ['mid', B * K * 3], ['nax', B * M], ['mj', B * P * 3], ['cm', B * C * 2], ['react', B * N * 3],
      ['e', B * M * 3], ['lseg', B * M], ['r', B * M * 9], ['tn', B * N * 3], ['qn', B * N * 4], ['tpar', B * M], ['inert', B * M], ['eref', B * M * 3], ['c0', B * M],
      ['hist', B * HN * 8], ['out', B * NOUT * 4]]);
    L.a = alloc(L.aS = {}, [['f', B * N * 6], ['t', B * M * 6], ['k', B * M * 2]]);
    this.L = L;
    // ---- buffers
    const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (words, label) => device.createBuffer({ size: Math.max(16, words * 4), usage: SU, label });
    this.bTopo = mk(L.tS.size, 'topo'); this.bGeo = mk(L.gS.size, 'geo'); this.bIp = mk(B * NIP, 'ip');
    this.bIst = mk(L.iS.size, 'ist'); this.bFst = mk(L.fS.size, 'fst'); this.bAcc = mk(L.aS.size, 'acc');
    this.bPart = mk(B * 4 * L.NWG * 8, 'part'); this.bCtl = mk(B * NCT, 'ctl');
    this.geo = new Float32Array(L.gS.size); this.ipArr = new ArrayBuffer(B * NIP * 4);
    this._uploadTopo();
    // ---- pipelines (cached per device and source)
    this.code = solverWGSL(L);
    let cache = PIPES.get(device);
    if (!cache) PIPES.set(device, cache = new Map());
    let ent = cache.get(this.code);
    if (!ent) {
      const module = device.createShaderModule({ code: this.code, label: 'weavesim solver' });
      const layout = device.createBindGroupLayout({ entries: ['read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'storage', 'storage', 'storage', 'storage'].map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: t } })) });
      const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      const pipes = {};
      for (const e of ['k_prep', 'k_force', 'k_seg', 'k_node', 'k_segint', 'k_ctrl']) pipes[e] = device.createComputePipeline({ layout: pl, compute: { module, entryPoint: e }, label: e });
      ent = { module, layout, pipes };
      cache.set(this.code, ent);
      if (cache.size > 8) cache.delete(cache.keys().next().value);
    }
    this.module = ent.module; this.pipes = ent.pipes;
    this.bg = device.createBindGroup({ layout: ent.layout, entries: [this.bTopo, this.bGeo, this.bIp, this.bIst, this.bFst, this.bAcc, this.bPart, this.bCtl].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.ctlHost = new Array(B).fill(null);
    this.histRead = new Array(B).fill(0);
    for (let b = 0; b < B; b++) this.upload(b);
  }

  /** compile-time check: resolves to the shader's compilation messages (errors abort the solver) */
  async compileInfo() { const info = await this.module.getCompilationInfo(); return info.messages.map(m => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`); }

  // ------------------------------------------------------------------------------------------- fixed-point scales
  /** fixed-point scales for a set of models of one topology: positions 2^-PB, forces 2^-FB, torques 2^-TB, contact
   *  twist stiffness 2^-KB. A batch whose results must not depend on its composition (calibration: the same parameter
   *  vector gives the same objective in any batch) passes scales computed once, from the corners of the parameter box. */
  static computeScales(models) {
    const st = models[0].st;
    let R = 1e-6; for (const v of st.X) R = Math.max(R, Math.abs(v));
    const PB = clampi(59 - ceilLog2(R * 4), 16, 56);                 // positions: |X|·2^PB < 2^61
    let EA = 0, EI = 0, GJ = 0, kp = 0, h = 0, w = 0, Lsum = 0, lmin = Infinity;
    for (const md of models) {
      for (const v of md.stiff.EA) EA = Math.max(EA, v); for (const v of md.stiff.EIx) EI = Math.max(EI, v); for (const v of md.stiff.EIy) EI = Math.max(EI, v); for (const v of md.stiff.GJ) GJ = Math.max(GJ, v);
      kp = Math.max(kp, md.scene.contact.strength, md.scene.contact.stick_strength);
      if (md.contact) { for (const v of md.contact.hk) h = Math.max(h, v); for (const v of md.contact.wk) w = Math.max(w, v); }
      let s = 0; for (const v of md.loads.const) s += Math.abs(v);
      if (md.protocol && md.protocol.mode === 1) { let f = 0; for (const v of md.protocol.sched) f = Math.max(f, Math.abs(v)); s += f; }
      Lsum = Math.max(Lsum, s);
    }
    for (const v of st.l0) lmin = Math.min(lmin, v);
    if (!Number.isFinite(lmin)) lmin = 1;
    const Fmax = Math.max(EA * 1e-2, kp * Math.max(h, w), Lsum * 10, 4 * EI / (lmin * lmin), 1e-6);
    const Tmax = Math.max(EI / lmin, GJ / lmin, Fmax * Math.max(w, h, lmin), 1e-6);
    const Kmax = Math.max((Fmax + kp * h) * w / 0.03 * 10, kp * h * h, 1e-6);
    return {
      PB,
      FB: clampi(55 - ceilLog2(Fmax), 8, 44),                         // |F|·2^FB stays below 2^55 (2^8 headroom)
      TB: clampi(55 - ceilLog2(Tmax), 8, 44),
      KB: clampi(55 - ceilLog2(Kmax), 8, 44),
    };
  }
  get scales() { return { PB: this.PB, FB: this.FB, TB: this.TB, KB: this.KB }; }

  _uploadTopo() {
    const { N, M, P, K, C } = this.sizes, L = this.L, md = this.models[0], st = md.st;
    const T = new Uint32Array(L.tS.size), t = L.t, NONE = 0xffffffff;
    T.set(st.segI0, t.seg0); T.set(st.segI1, t.seg1); T.set(st.jSeg0, t.j0); T.set(st.jSeg1, t.j1); T.set(st.jNode, t.jn);
    const cl = md.anchors.clamps;
    cl.forEach((c, i) => { T[t.cn + i] = c[0]; T[t.cb + i] = c[1]; T[t.cs + i] = c[2]; });
    T.fill(NONE, t.nsa, t.nsa + N); T.fill(NONE, t.nsb, t.nsb + N);
    for (let j = 0; j < M; j++) { T[t.nsa + st.segI0[j]] = j; T[t.nsb + st.segI1[j]] = j; }
    if (md.contact) {
      const c = md.contact;
      for (let i = 0; i < 2 * K; i++) { T[t.koff + i] = c.segOff[i]; T[t.knseg + i] = c.nseg[i]; }
      for (let k = 0; k < K; k++) T[t.ken + k] = c.enabled[k];
    }
    this.dev.queue.writeBuffer(this.bTopo, 0, T);
    const G = this.geo, g = L.g;
    for (let j = 0; j < M; j++) G[g.l0 + j] = st.l0[j];
    for (let j = 0; j < P; j++) G[g.jlen + j] = st.jLen[j];
    md.beam.cLen.forEach((v, i) => { G[g.clen + i] = v; });
    if (md.contact) {
      const c = md.contact;
      for (let i = 0; i < 2 * K; i++) { G[g.wk + i] = c.wk[i]; G[g.hk + i] = c.hk[i]; }
      for (let k = 0; k < K; k++) { G[g.kw + k] = c.w[k]; G[g.kh + k] = c.h[k]; }
    }
    this.jlenMin = P ? Math.min(...st.jLen) : 1;
  }

  // ------------------------------------------------------------------------------------------- host → device
  /** copy instance b's model (state, stiffness, masses, loads, anchors, contact state, parameters) to the GPU */
  upload(b) {
    const md = this.models[b], st = md.st, { N, M, P, K, C } = this.sizes, L = this.L, G = this.geo, g = L.g, B = this.B;
    // geo, per instance
    for (let j = 0; j < M; j++) { const o = g.stiff + (b * M + j) * 4; G[o] = md.stiff.EA[j]; G[o + 1] = md.stiff.EIx[j]; G[o + 2] = md.stiff.EIy[j]; G[o + 3] = md.stiff.GJ[j]; }
    for (let i = 0; i < 3 * P; i++) G[g.om + b * P * 3 + i] = st.Omega0[i];
    for (let i = 0; i < N; i++) G[g.mass + b * N + i] = md.mass[i];
    for (let j = 0; j < M; j++) G[g.kseg + b * M + j] = md.kSegStatic[j];
    for (let i = 0; i < 3 * N; i++) G[g.lc + b * N * 3 + i] = md.loads.const[i];
    G.fill(0, g.ll + b * N * 3, g.ll + (b + 1) * N * 3);
    for (const [nid, Fv] of md.loads.pointLocal) for (let c = 0; c < 3; c++) G[g.ll + (b * N + nid) * 3 + c] += Fv[c];
    md.beam.cNode.forEach((_, c) => {
      G[g.cei + (b * C + c) * 2] = md.beam.cEIx[c]; G[g.cei + (b * C + c) * 2 + 1] = md.beam.cEIy[c];
      for (let k = 0; k < 3; k++) G[g.cdir + (b * C + c) * 3 + k] = md.beam.cDir[c][k];
    });
    if (md.contact) for (let i = 0; i < 2 * K; i++) { G[g.fc90 + b * K * 2 + i] = md.contact.fc90[i]; G[g.e90 + b * K * 2 + i] = md.contact.E90[i]; G[g.hs + b * K * 2 + i] = md.contact.Hs[i]; }
    this.dev.queue.writeBuffer(this.bGeo, 0, G);
    this._uploadParams(b);
    this._uploadState(b);
  }

  _uploadParams(b) {
    const md = this.models[b], sc = md.scene, sp = sc.solver, cp = sc.contact;
    const dv = new DataView(this.ipArr, b * NIP * 4, NIP * 4);
    const f = (w, v) => dv.setFloat32(4 * w, v, true), u = (w, v) => dv.setUint32(4 * w, v >>> 0, true);
    f(0, sp.mass_scale); f(1, sp.damping); u(2, sp.kinetic_damping ? 1 : 0); u(3, sp.warmup_iter | 0);
    f(4, sp.tol); u(5, Math.max(cp.ramp_iter | 0, 0)); u(6, md.contact ? 1 : 0); u(7, cp.augmented ? 1 : 0);
    u(8, cp.al_every | 0); f(9, cp.mu); f(10, cp.strength); f(11, cp.stick_strength);
    f(12, cp.torsion_factor); f(13, cp.epsilon); f(14, cp.slip_rate); u(15, cp.rot_friction ? 1 : 0);
    f(16, cp.rot_factor); u(17, cp.seat_law !== 'off' ? 1 : 0); u(18, cp.seat_gap ? 1 : 0); f(19, cp.dent_rate);
    u(20, cp.dent_after_iter >= 0 ? cp.dent_after_iter : (md.contact ? md.contact.dentAfterDefault : 0)); f(21, cp.edge_threshold); f(22, this.jlenMin); u(23, 0);
    f(24, 2 ** (this.PB - 32)); f(25, 2 ** -this.PB); f(26, 2 ** (this.FB - 32)); f(27, 2 ** -this.FB);
    f(28, 2 ** (this.TB - 32)); f(29, 2 ** -this.TB); f(30, 2 ** (this.KB - 32)); f(31, 2 ** -this.KB);
    const pr = md.protocol || null;
    u(32, pr ? pr.mode : 0); u(33, pr ? pr.probe : 0); u(34, pr ? pr.sched.length : 0); u(35, pr ? pr.settle : 0);
    u(36, pr ? pr.avg : 0); u(37, 0); u(38, 0); u(39, 0);
    f(40, pr ? pr.axis[0] : 0); f(41, pr ? pr.axis[1] : 0); f(42, pr ? pr.axis[2] : -1); f(43, pr ? pr.kPad || 0 : 0);
    for (let i = 0; i < 64; i++) f(44 + i, pr && i < pr.sched.length ? pr.sched[i] : 0);
    this.dev.queue.writeBuffer(this.bIp, b * NIP * 4, this.ipArr, b * NIP * 4, NIP * 4);
  }

  _uploadState(b) {
    const md = this.models[b], st = md.st, { N, M, P, K, C } = this.sizes, L = this.L, PB = this.PB;
    const i = L.i;
    const put64 = (arr, o, x, bits) => { const [lo, hi] = toI64(x, bits); arr[o] = lo; arr[o + 1] = hi; };
    const xw = new Uint32Array(N * 6), x0w = new Uint32Array(N * 6);
    for (let n = 0; n < N; n++) for (let c = 0; c < 3; c++) { put64(xw, n * 6 + 2 * c, st.X[3 * n + c], PB); put64(x0w, n * 6 + 2 * c, st.X0[3 * n + c], PB); }
    this._wI(i.x + b * N * 6, xw); this._wI(i.xp + b * N * 6, xw); this._wI(i.x0 + b * N * 6, x0w);
    this._writeEref(b, xw);
    if (K) {
      const uw = new Uint32Array(K * 4), sw = new Uint32Array(K * 2), lk = new Uint32Array(K);
      for (let k = 0; k < 2 * K; k++) { const uu = st.u[k], hi = Math.floor(uu); uw[2 * k] = Math.min(Math.round((uu - hi) * TWO32), TWO32 - 1) >>> 0; uw[2 * k + 1] = hi >>> 0; }
      for (let k = 0; k < K; k++) { put64(sw, 2 * k, st.slip[k], PB); lk[k] = md.contact ? md.contact.locked[k] : 0; }
      this._wI(i.u + b * K * 4, uw); this._wI(i.slip + b * K * 2, sw); this._wI(i.lock + b * K, lk);
    }
    const fx = new Uint32Array(N), tf = new Uint32Array(M);
    for (let n = 0; n < N; n++) { const a = md.anchors.nodeFree; let bits = (a[3 * n] ? 0 : 1) | (a[3 * n + 1] ? 0 : 2) | (a[3 * n + 2] ? 0 : 4); if (bits) bits |= 8; fx[n] = bits; }
    for (let j = 0; j < M; j++) tf[j] = md.anchors.twistFree[j] ? 0 : 1;
    if (md.protocol && md.protocol.mode === 2) fx[md.protocol.probe] = 15;
    this._wI(i.fix + b * N, fx); this._wI(i.tfix + b * M, tf);
    if (md.protocol) { const p0 = new Uint32Array(6); for (let c = 0; c < 3; c++) put64(p0, 2 * c, st.X[3 * md.protocol.probe + c], PB); this._wI(i.probe0 + b * 6, p0); }
    // float state
    const f = L.f, F = (o, arr) => this.dev.queue.writeBuffer(this.bFst, o * 4, Float32Array.from(arr));
    F(f.v + b * N * 3, st.V); F(f.w + b * M, st.Wt);
    const qw = new Uint32Array(M * 8);
    for (let k = 0; k < 4 * M; k++) put64(qw, 2 * k, st.Q[k], 60);
    this._wI(i.q + b * M * 8, qw); this._wI(i.qp + b * M * 8, qw);
    F(f.inert + b * M, md.inertia);
    if (K && md.contact) {
      const c = md.contact;
      F(f.nprev + b * K * 3, st.nPrev); F(f.lam + b * K, st.lam); F(f.th0 + b * K, Array.from(c.theta0, v => (Number.isNaN(v) ? 1e30 : v)));
      F(f.rslip + b * K, c.rotSlip); F(f.dent + b * K * 2, c.dent); F(f.dente + b * K, c.dentE); F(f.climb + b * K * 2, c.climb); F(f.tanpsi + b * K * 2, c.tanpsi);
      F(f.fn + b * K, c.Fn);
    }
    // control: iteration, kinetic-energy peak, flags; the protocol starts in its settle phase with increment 0
    const ct = new ArrayBuffer(NCT * 4), dv = new DataView(ct);
    dv.setUint32(0, st.iteration, true); dv.setUint32(4, 0, true); dv.setUint32(8, 0, true); dv.setUint32(12, 0, true);
    dv.setFloat32(16, md.kePrev, true); dv.setFloat32(24, Number.isFinite(st.residual) ? st.residual : 3e38, true);
    dv.setUint32(48, st.iteration, true);
    const pr = md.protocol;
    if (pr && !pr.sched.length) throw new Error('GPU replay: empty load schedule');
    if (pr && pr.mode === 1) { dv.setFloat32(80, pr.axis[0] * pr.sched[0], true); dv.setFloat32(84, pr.axis[1] * pr.sched[0], true); dv.setFloat32(88, pr.axis[2] * pr.sched[0], true); dv.setFloat32(92, pr.sched[0], true); }
    if (pr && pr.mode === 2) dv.setFloat32(92, pr.sched[0], true);
    this.dev.queue.writeBuffer(this.bCtl, b * NCT * 4, ct);
    this.ctlHost[b] = { iter: st.iteration, done: 0, rollback: 0, resid: st.residual };
    this.histRead[b] = 0;
    if (pr && pr.mode === 2 && pr.sched.length) this._placeProbe(b, pr.sched[0]);
  }

  _wI(o, arr) { this.dev.queue.writeBuffer(this.bIst, o * 4, arr); }

  /** displacement control: the probe node sits at P0 + v·axis */
  _placeProbe(b, v) {
    const md = this.models[b], pr = md.protocol, PB = this.PB, w = new Uint32Array(6);
    for (let c = 0; c < 3; c++) { const [lo, hi] = toI64(md.st.X[3 * pr.probe + c] + pr.axis[c] * v, PB); w[2 * c] = lo; w[2 * c + 1] = hi; }
    this._wI(this.L.i.x + (b * this.sizes.N + pr.probe) * 6, w); this._wI(this.L.i.xp + (b * this.sizes.N + pr.probe) * 6, w);
  }

  /** reference edge vectors for the compensated stretch: E = X[i1] − X[i0] (exact integers), its f32 copy and
   *  c0 = |E|² − l0² in f64 — rebasing them on the current geometry keeps |δ| small, so ext stays exact to f32 */
  _writeEref(b, xw) {
    const md = this.models[b], st = md.st, { M } = this.sizes, L = this.L, PB = this.PB;
    const ew = new Uint32Array(M * 6), e32 = new Float32Array(M * 3), c0 = new Float32Array(M);
    for (let j = 0; j < M; j++) {
      const a = st.segI0[j], c = st.segI1[j];
      let s2 = 0;
      for (let k = 0; k < 3; k++) {
        const oa = a * 6 + 2 * k, oc = c * 6 + 2 * k;
        let lo = xw[oc] - xw[oa], borrow = 0; if (lo < 0) { lo += TWO32; borrow = 1; }
        const hi = (xw[oc + 1] - xw[oa + 1] - borrow) >>> 0;
        ew[j * 6 + 2 * k] = lo >>> 0; ew[j * 6 + 2 * k + 1] = hi;
        const v = fromI64(lo >>> 0, hi, PB);
        e32[3 * j + k] = v; s2 += v * v;
      }
      c0[j] = s2 - st.l0[j] * st.l0[j];
    }
    this._wI(L.i.eref + b * M * 6, ew);
    this.dev.queue.writeBuffer(this.bFst, (L.f.eref + b * M * 3) * 4, e32);
    this.dev.queue.writeBuffer(this.bFst, (L.f.c0 + b * M) * 4, c0);
  }

  /** after a scene edit: copy the instance's rebuilt model parameters (stiffness, masses, loads, anchors, rest state,
   *  contact parameters) without touching the dynamic state on the GPU */
  updateParams(b) {
    const md = this.models[b], { N, M, P, K, C } = this.sizes, L = this.L, G = this.geo, g = L.g;
    for (let j = 0; j < M; j++) { const o = g.stiff + (b * M + j) * 4; G[o] = md.stiff.EA[j]; G[o + 1] = md.stiff.EIx[j]; G[o + 2] = md.stiff.EIy[j]; G[o + 3] = md.stiff.GJ[j]; }
    for (let i = 0; i < 3 * P; i++) G[g.om + b * P * 3 + i] = md.st.Omega0[i];
    for (let i = 0; i < N; i++) G[g.mass + b * N + i] = md.mass[i];
    for (let j = 0; j < M; j++) G[g.kseg + b * M + j] = md.kSegStatic[j];
    for (let i = 0; i < 3 * N; i++) G[g.lc + b * N * 3 + i] = md.loads.const[i];
    G.fill(0, g.ll + b * N * 3, g.ll + (b + 1) * N * 3);
    for (const [nid, Fv] of md.loads.pointLocal) for (let c = 0; c < 3; c++) G[g.ll + (b * N + nid) * 3 + c] += Fv[c];
    if (md.anchors.clamps.length !== C) throw new Error('the anchors changed the number of clamps: rebuild the GPU solver');
    md.beam.cNode.forEach((_, c) => { G[g.cei + (b * C + c) * 2] = md.beam.cEIx[c]; G[g.cei + (b * C + c) * 2 + 1] = md.beam.cEIy[c]; for (let k = 0; k < 3; k++) G[g.cdir + (b * C + c) * 3 + k] = md.beam.cDir[c][k]; });
    this.dev.queue.writeBuffer(this.bGeo, 0, G);
    const fx = new Uint32Array(N), tf = new Uint32Array(M);
    for (let n = 0; n < N; n++) { const a = md.anchors.nodeFree; let bits = (a[3 * n] ? 0 : 1) | (a[3 * n + 1] ? 0 : 2) | (a[3 * n + 2] ? 0 : 4); if (bits) bits |= 8; fx[n] = bits; }
    for (let j = 0; j < M; j++) tf[j] = md.anchors.twistFree[j] ? 0 : 1;
    this._wI(L.i.fix + b * N, fx); this._wI(L.i.tfix + b * M, tf);
    if (md.contact) this._wI(L.i.lock + b * K, Uint32Array.from(md.contact.locked));
    this._uploadParams(b);
    // a rebuild resets the kinetic-energy peak and the convergence flag (Solver.rebuild_goals)
    this.dev.queue.writeBuffer(this.bCtl, (b * NCT + 4) * 4, new Float32Array([0]));
    this.dev.queue.writeBuffer(this.bCtl, (b * NCT + 1) * 4, new Uint32Array([0]));
    this.dev.queue.writeBuffer(this.bFst, (L.f.inert + b * M) * 4, Float32Array.from(md.inertia));
  }

  // ------------------------------------------------------------------------------------------- running
  /** run up to nIter iterations on the selected instances (default all); stops early per instance at convergence
   *  (opts.force: clear the convergence flag first, like Solver.run). Resolves with the control read-back. */
  async step(nIter, opts = {}) {
    const B = this.B, insts = opts.instances || [...Array(B).keys()], dev = this.dev;
    for (const b of insts) {
      const c = this.ctlHost[b];
      const w = new Uint32Array([c.iter + nIter]);
      dev.queue.writeBuffer(this.bCtl, (b * NCT + 12) * 4, w);
      if (opts.force !== false) dev.queue.writeBuffer(this.bCtl, (b * NCT + 1) * 4, new Uint32Array([0]));
    }
    const L = this.L, { N, M, P, K, C } = this.sizes;
    const groups = [['k_prep', L.NWG_P], ['k_force', L.NWG_F], ['k_seg', L.NWG_S], ['k_node', L.NWG_N], ['k_segint', L.NWG_S], ['k_ctrl', 1]];
    let left = nIter;
    while (left > 0) {
      const k = Math.min(this.perSubmit, left); left -= k;
      const enc = dev.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.bg);
      for (let it = 0; it < k; it++) for (const [name, nwg] of groups) { if (!nwg) continue; pass.setPipeline(this.pipes[name]); pass.dispatchWorkgroups(nwg, B); }
      pass.end();
      dev.queue.submit([enc.finish()]);
    }
    return this.readControl();
  }

  async _read(buffer, offsetWords, words) {
    const rb = this.dev.createBuffer({ size: Math.max(16, words * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offsetWords * 4, rb, 0, words * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = rb.getMappedRange().slice(0, words * 4);
    rb.unmap(); rb.destroy();
    return out;
  }

  async readControl() {
    const buf = await this._read(this.bCtl, 0, this.B * NCT);
    const dv = new DataView(buf), out = [];
    for (let b = 0; b < this.B; b++) {
      const o = b * NCT * 4, U = (w) => dv.getUint32(o + 4 * w, true), F = (w) => dv.getFloat32(o + 4 * w, true);
      const c = { iter: U(0), done: U(1), rollback: U(2), alUpdate: U(3), kePrev: F(4), ke: F(5), resid: F(6) >= 1e38 ? Infinity : F(6), fref: F(7),
        eStretch: F(8), eBend: F(9), eTors: F(10), eCont: F(11), stopAt: U(12), histN: U(13), phase: U(14), inc: U(15) };
      this.ctlHost[b] = c; out.push(c);
    }
    return out;
  }

  /** the energy / residual history written since the last call, per instance (appended to model.st.history) */
  async readHistory(b) {
    const c = this.ctlHost[b], done = c.histN, from = this.histRead[b];
    const n = Math.min(done - from, HN);
    if (n <= 0) return 0;
    const buf = new Float32Array(await this._read(this.bFst, this.L.f.hist + b * HN * 8, HN * 8));
    const H = this.models[b].st.history;
    for (let k = done - n; k < done; k++) {
      // entry of iteration it (1-based) sits at (it − 1) % HN; the k-th recorded entry here is iteration c.iter − (done − 1 − k)
      const it = c.iter - (done - 1 - k), o = ((it - 1) % HN) * 8;
      H.kinetic.push(buf[o]); H.strain.push(buf[o + 1]); H.bending.push(buf[o + 2]); H.torsion.push(buf[o + 3]); H.stretch.push(buf[o + 4]); H.contact.push(buf[o + 5]);
      H.residual.push(buf[o + 6] >= 1e38 ? Infinity : buf[o + 6]);
    }
    this.histRead[b] = done;
    return n;
  }

  /** protocol outputs (travel, reaction, applied value, iteration) per increment */
  async readOutputs(b) {
    const buf = new Float32Array(await this._read(this.bFst, this.L.f.out + b * NOUT * 4, NOUT * 4));
    return buf;
  }
  /** the protocol outputs of every instance in one read: Float32Array[B] of NOUT × (travel, reaction, value, iteration) */
  async readAllOutputs() {
    const buf = new Float32Array(await this._read(this.bFst, this.L.f.out, this.B * NOUT * 4));
    return Array.from({ length: this.B }, (_, b) => buf.slice(b * NOUT * 4, (b + 1) * NOUT * 4));
  }
  /** download every instance (or the listed ones) with one read of the state buffers */
  async downloadAll(instances = null, opts = {}) {
    const L = this.L;
    await this.readControl();
    const bufs = { iBuf: new Uint32Array(await this._read(this.bIst, 0, L.iS.size)), fBuf: new Float32Array(await this._read(this.bFst, 0, L.fS.size)) };
    const out = [];
    for (const b of instances || [...Array(this.B).keys()]) out.push(await this.download(b, { ...opts, bufs }));
    return out;
  }

  /** copy the GPU state of instance b into its CpuSolver model (positions, frames, contacts, outputs, energies) */
  async download(b, { rebase = true, bufs = null, history = true } = {}) {
    const md = this.models[b], st = md.st, { N, M, P, K, C } = this.sizes, L = this.L, PB = this.PB;
    const c = (bufs && this.ctlHost[b]) || (await this.readControl())[b];
    const rb = c.rollback === 1;
    const iBuf = bufs ? bufs.iBuf : new Uint32Array(await this._read(this.bIst, 0, L.iS.size));
    const fBuf = bufs ? bufs.fBuf : new Float32Array(await this._read(this.bFst, 0, L.fS.size));
    const xo = (rb ? L.i.xp : L.i.x) + b * N * 6;
    for (let n = 0; n < N; n++) for (let k = 0; k < 3; k++) { const o = xo + n * 6 + 2 * k; st.X[3 * n + k] = fromI64(iBuf[o], iBuf[o + 1], PB); }
    const x0o = L.i.x0 + b * N * 6;
    for (let n = 0; n < N; n++) for (let k = 0; k < 3; k++) { const o = x0o + n * 6 + 2 * k; st.X0[3 * n + k] = fromI64(iBuf[o], iBuf[o + 1], PB); }
    const qo = (rb ? L.i.qp : L.i.q) + b * M * 8;
    for (let i = 0; i < 4 * M; i++) st.Q[i] = fromI64(iBuf[qo + 2 * i], iBuf[qo + 2 * i + 1], 60);
    if (rb) { st.V.fill(0); st.Wt.fill(0); }
    else { for (let i = 0; i < 3 * N; i++) st.V[i] = fBuf[L.f.v + b * N * 3 + i]; for (let j = 0; j < M; j++) st.Wt[j] = fBuf[L.f.w + b * M + j]; }
    for (let j = 0; j < M; j++) md.beam.N[j] = fBuf[L.f.nax + b * M + j];
    for (let i = 0; i < 3 * P; i++) md.beam.M[i] = fBuf[L.f.mj + b * P * 3 + i];
    for (let i = 0; i < 2 * C; i++) md.beam.cM[i] = fBuf[L.f.cm + b * C * 2 + i];
    for (let i = 0; i < 3 * N; i++) md.anchors.reactions[i] = fBuf[L.f.react + b * N * 3 + i];
    md.beam.energyParts = { bending: c.eBend, torsion: c.eTors, stretch: c.eStretch };
    md.beam.energy = c.eBend + c.eTors + c.eStretch;
    if (K && md.contact) {
      const ct = md.contact;
      for (let k = 0; k < 2 * K; k++) { const o = L.i.u + b * K * 4 + 2 * k; st.u[k] = iBuf[o + 1] >= 0x80000000 ? 0 : iBuf[o + 1] + iBuf[o] / TWO32; }
      for (let k = 0; k < K; k++) {
        const o = L.i.slip + b * K * 2 + 2 * k; st.slip[k] = fromI64(iBuf[o], iBuf[o + 1], PB);
        const kb = b * K + k, ko = iBuf[L.i.kout + kb];
        ct.state[k] = ko & 1; st.stuck[k] = (ko >> 1) & 1; ct.ctype[k] = (ko >> 2) & 3; const fs = (ko >> 4) & 3; ct.faceSide[k] = fs === 3 ? -1 : fs;
        ct.Fn[k] = fBuf[L.f.fn + kb]; ct.Ft[k] = fBuf[L.f.ft + kb]; ct.gap[k] = fBuf[L.f.gap + kb]; ct.theta[k] = fBuf[L.f.theta + kb];
        ct.Anom[k] = fBuf[L.f.anom + kb]; ct.pressure[k] = fBuf[L.f.pres + kb]; ct.util[k] = fBuf[L.f.util + kb]; ct.Mrot[k] = fBuf[L.f.mrot + kb];
        ct.rotSlip[k] = fBuf[L.f.rslip + kb]; ct.dentE[k] = fBuf[L.f.dente + kb];
        const t0 = fBuf[L.f.th0 + kb]; ct.theta0[k] = t0 > 1e29 ? NaN : t0;
        st.lam[k] = c.alUpdate ? (ct.enabled[k] ? ct.Fn[k] : 0) : fBuf[L.f.lam + kb];
        ct.locked[k] = iBuf[L.i.lock + kb];
        for (let q = 0; q < 2; q++) { ct.phi[2 * k + q] = fBuf[L.f.phi + kb * 2 + q]; ct.dent[2 * k + q] = fBuf[L.f.dent + kb * 2 + q]; ct.climb[2 * k + q] = fBuf[L.f.climb + kb * 2 + q]; ct.tanpsi[2 * k + q] = fBuf[L.f.tanpsi + kb * 2 + q]; }
        for (let q = 0; q < 3; q++) { st.nPrev[3 * k + q] = fBuf[L.f.nprev + kb * 3 + q]; ct.mid[3 * k + q] = fBuf[L.f.mid + kb * 3 + q]; }
      }
      ct.energy = c.eCont;
    }
    st.iteration = c.iter; st.residual = c.resid; md.kePrev = c.kePrev;
    md.converged = c.done === 1 && c.resid <= md.scene.solver.tol;
    if (history) await this.readHistory(b);
    else this.histRead[b] = c.histN;
    if (rebase) {                                                  // keep the stretch reference close to the geometry
      const xw = new Uint32Array(N * 6); xw.set(iBuf.subarray(xo, xo + N * 6));
      this._writeEref(b, xw);
    }
    return md;
  }

  /** current geometry becomes the displacement origin; slip counters restart (on the GPU). WebGPU rejects a copy within
   *  one buffer (the whole command buffer is dropped), so x goes to x0 through a scratch buffer. */
  internalize(b) {
    const { N, K } = this.sizes, L = this.L, rb = this.ctlHost[b] && this.ctlHost[b].rollback === 1, bytes = N * 24;
    if (!this.bTmp || this.bTmp.size < bytes) {
      try { this.bTmp?.destroy(); } catch (e) { /* ignore */ }
      this.bTmp = this.dev.createBuffer({ label: 'tmp', size: Math.max(16, bytes), usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    }
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.bIst, ((rb ? L.i.xp : L.i.x) + b * N * 6) * 4, this.bTmp, 0, bytes);
    enc.copyBufferToBuffer(this.bTmp, 0, this.bIst, (L.i.x0 + b * N * 6) * 4, bytes);
    if (K) enc.clearBuffer(this.bIst, (L.i.slip + b * K * 2) * 4, K * 8);
    this.dev.queue.submit([enc.finish()]);
  }

  dispose() { for (const b of [this.bTopo, this.bGeo, this.bIp, this.bIst, this.bFst, this.bAcc, this.bPart, this.bCtl, this.bTmp]) try { b?.destroy(); } catch (e) { /* ignore */ } }
}

export { toI64, fromI64, diffI64 };
