// WGSL source of the batched dynamic-relaxation solver (one dispatch per kernel, B instances along the y grid axis).
//
// Precision scheme (WebGPU has no f64):
//  · node positions are 64-bit fixed-point integers (vec2<u32>, 2^-PB length units): the velocity is added exactly,
//    every edge / stencil vector is an exact integer difference converted to f32 once — so the f32 error is relative
//    to the local size (a segment, a crossing), never to the coordinate (a 3 m model in f32 alone would lose 0.2 µm
//    per coordinate and several newtons of axial force per lath);
//  · axial strain is computed against a per-segment reference edge vector (exact integer + f32 copy + f64 offset c0):
//    ext = (c0 + 2E·δ + δ·δ) / (L + l0), δ = exact integer change since the reference — no cancellation in L − l0;
//  · contact parameters u are 64-bit fixed point (segment index | 32-bit fraction), slips add exactly; the Hermite
//    point is evaluated at the 24-bit part of t plus a first-order term for the remaining 8 bits;
//  · forces, torques and the contact twist stiffness are accumulated as 64-bit fixed-point integers with atomics:
//    sums are exact and independent of thread order, so a run is bit-for-bit reproducible on a given GPU;
//  · reductions (kinetic energy, residual, energies) run in a fixed tree order.
export function solverWGSL(L) {
  const u = (n) => `${n >>> 0}u`;
  return /* wgsl */`
// ---------------------------------------------------------------------------------------------- sizes
const N: u32 = ${u(L.N)}; const M: u32 = ${u(L.M)}; const P: u32 = ${u(L.P)}; const C: u32 = ${u(L.C)}; const K: u32 = ${u(L.K)};
const NONE: u32 = 0xffffffffu;
const SUPPORT_SMOOTH: f32 = 0.03;
const TWIST_RATE_MAX: f32 = 0.015;      // 0.5 · SUPPORT_SMOOTH: roll step per iteration (solver_cpu.js)
const AL_FREE_MIN: f32 = 0.25;          // free weight below which a contact point counts as support-controlled (no multiplier)
const PI: f32 = 3.14159265358979;
const HN: u32 = ${u(L.HN)};
const NWG: u32 = ${u(L.NWG)};
const NSLOT: u32 = 4u;
// topo (u32)
const T_SEG0 = ${u(L.t.seg0)}; const T_SEG1 = ${u(L.t.seg1)}; const T_J0 = ${u(L.t.j0)}; const T_J1 = ${u(L.t.j1)}; const T_JN = ${u(L.t.jn)};
const T_CN = ${u(L.t.cn)}; const T_CB = ${u(L.t.cb)}; const T_CS = ${u(L.t.cs)}; const T_NSA = ${u(L.t.nsa)}; const T_NSB = ${u(L.t.nsb)};
const T_KOFF = ${u(L.t.koff)}; const T_KNSEG = ${u(L.t.knseg)}; const T_KEN = ${u(L.t.ken)};
// geo (f32): shared, then per instance
const G_L0 = ${u(L.g.l0)}; const G_JLEN = ${u(L.g.jlen)}; const G_CLEN = ${u(L.g.clen)}; const G_WK = ${u(L.g.wk)}; const G_HK = ${u(L.g.hk)};
const G_KW = ${u(L.g.kw)}; const G_KH = ${u(L.g.kh)};
const G_STIFF = ${u(L.g.stiff)}; const G_OM = ${u(L.g.om)}; const G_MASS = ${u(L.g.mass)}; const G_KSEG = ${u(L.g.kseg)};
const G_LC = ${u(L.g.lc)}; const G_LL = ${u(L.g.ll)}; const G_CEI = ${u(L.g.cei)}; const G_CDIR = ${u(L.g.cdir)};
const G_FC90 = ${u(L.g.fc90)}; const G_E90 = ${u(L.g.e90)}; const G_HS = ${u(L.g.hs)};
// ist (u32): integer state per instance
const I_X = ${u(L.i.x)}; const I_XP = ${u(L.i.xp)}; const I_X0 = ${u(L.i.x0)}; const I_EREF = ${u(L.i.eref)}; const I_U = ${u(L.i.u)};
const I_SLIP = ${u(L.i.slip)}; const I_FIX = ${u(L.i.fix)}; const I_TFIX = ${u(L.i.tfix)}; const I_LOCK = ${u(L.i.lock)}; const I_KOUT = ${u(L.i.kout)};
const I_PROBE0 = ${u(L.i.probe0)}; const I_Q = ${u(L.i.q)}; const I_QP = ${u(L.i.qp)};
// quaternions are 64-bit fixed point too (2^-60): frame updates are added in delta form, so a frame that does not move
// does not change by a single bit — f32 frames re-rounded every iteration inject ~1e-7 rad of roll noise, which the
// stiff twist mode turns into kinetic energy that trips the kinetic-damping reset and stalls convergence
const QINV: f32 = 8.673617379884035e-19; const QSC: f32 = 268435456.0;
// fst (f32): float state per instance
const F_V = ${u(L.f.v)}; const F_W = ${u(L.f.w)}; const F_NPREV = ${u(L.f.nprev)};
const F_LAM = ${u(L.f.lam)}; const F_TH0 = ${u(L.f.th0)}; const F_RSLIP = ${u(L.f.rslip)}; const F_DENT = ${u(L.f.dent)}; const F_DENTE = ${u(L.f.dente)};
const F_CLIMB = ${u(L.f.climb)}; const F_TANPSI = ${u(L.f.tanpsi)}; const F_FN = ${u(L.f.fn)}; const F_FT = ${u(L.f.ft)}; const F_GAP = ${u(L.f.gap)};
const F_PHI = ${u(L.f.phi)}; const F_THETA = ${u(L.f.theta)}; const F_ANOM = ${u(L.f.anom)}; const F_PRES = ${u(L.f.pres)}; const F_UTIL = ${u(L.f.util)};
const F_MROT = ${u(L.f.mrot)}; const F_MID = ${u(L.f.mid)}; const F_NAX = ${u(L.f.nax)}; const F_MJ = ${u(L.f.mj)}; const F_CM = ${u(L.f.cm)};
const F_REACT = ${u(L.f.react)}; const F_E = ${u(L.f.e)}; const F_LSEG = ${u(L.f.lseg)}; const F_R = ${u(L.f.r)}; const F_TN = ${u(L.f.tn)};
const F_QN = ${u(L.f.qn)}; const F_TPAR = ${u(L.f.tpar)}; const F_INERT = ${u(L.f.inert)}; const F_EREF = ${u(L.f.eref)}; const F_C0 = ${u(L.f.c0)};
const F_HIST = ${u(L.f.hist)}; const F_OUT = ${u(L.f.out)}; const NOUT = ${u(L.NOUT)};
const F_PENACC = ${u(L.f.penacc)};
// acc (atomic u32): fixed-point accumulators per instance
const A_F = ${u(L.a.f)}; const A_T = ${u(L.a.t)}; const A_K = ${u(L.a.k)};

struct IP {                         // per-instance parameters (host-written)
  mass_scale: f32, damping: f32, kin_damp: u32, warmup: u32,
  tol: f32, ramp_iter: u32, contacts: u32, aug: u32,
  al_every: u32, mu: f32, strength: f32, stick: f32,
  torsion_factor: f32, epsilon: f32, slip_rate: f32, rot_fric: u32,
  rot_factor: f32, seat: u32, seat_gap: u32, dent_rate: f32,
  dent_after: u32, edge_thr: f32, jlen_min: f32, pad0: u32,
  psc: f32, pinv: f32, fsc: f32, finv: f32,             // positions: value·2^-PB, sc = 2^(PB-32); forces 2^-FB
  tsc: f32, tinv: f32, ksc: f32, kinv: f32,             // torques, twist stiffness
  // UTM protocol (mode 0: free run, 1: force-controlled cable pull, 2: displacement-controlled push)
  mode: u32, probe: u32, n_inc: u32, settle: u32,
  avg: u32, pad1: u32, pad2: u32, pad3: u32,
  axis_x: f32, axis_y: f32, axis_z: f32, k_pad: f32,
  sched: array<f32, 64>,            // force (mode 1) or structure travel (mode 2) per increment
};
struct CT {                         // per-instance control (device-written; host reads, resets flags)
  iter: u32, done: u32, rollback: u32, al_update: u32,
  ke_prev: f32, ke: f32, resid: f32, fref: f32,
  e_stretch: f32, e_bend: f32, e_tors: f32, e_cont: f32,
  stop_at: u32, hist_n: u32, phase: u32, inc: u32,
  it_phase: u32, acc_n: u32, pad_a: u32, pad_b: u32,
  pfx: f32, pfy: f32, pfz: f32, cur_f: f32,
  // UTM read-out sums over the averaging window, exact 64-bit fixed point: probe travel Σ(X − P0) per axis
  // (2^-PB) and the reaction along the probe axis (2^-FB) — one rounding when the average is formed
  adx: vec2<u32>, ady: vec2<u32>, adz: vec2<u32>, arr: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> topo: array<u32>;
@group(0) @binding(1) var<storage, read> geo: array<f32>;
@group(0) @binding(2) var<storage, read> ip: array<IP>;
@group(0) @binding(3) var<storage, read_write> ist: array<u32>;
@group(0) @binding(4) var<storage, read_write> fst: array<f32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(7) var<storage, read_write> ctl: array<CT>;

var<workgroup> wg_act: u32;
var<workgroup> rs: array<vec4<f32>, 64>;
var<workgroup> rm: array<vec4<f32>, 64>;


// ---------------------------------------------------------------------------------------------- accurate elementary functions
// WGSL allows sin/cos an absolute error of 2^-11 and atan2 4096 ULP (SwiftShader and Metal fast-math use such
// approximations); these use only + − × ÷, exact range reduction and Cephes single-precision polynomials (~1–2 ULP).
const PIO2_1: f32 = 1.5703125; const PIO2_2: f32 = 4.837512969970703125e-4; const PIO2_3: f32 = 7.549789954891882e-8;
const PIO2F: f32 = 1.5707963267948966; const PIO4F: f32 = 0.7853981633974483;
fn sqrt_acc(x: f32) -> f32 { let s = sqrt(x); return select(0.0, s + 0.5 * (x - s * s) / s, s > 0.0); }
fn len_acc(v: vec3<f32>) -> f32 { return sqrt_acc(dot(v, v)); }
fn sincos_acc(x: f32) -> vec2<f32> {
  let k = round(x * 0.6366197723675814);
  let r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
  let z = r * r;
  let sn = r + r * z * (-1.6666654611e-1 + z * (8.3321608736e-3 + z * -1.9515295891e-4));
  let cs = 1.0 - 0.5 * z + z * z * (4.166664568298827e-2 + z * (-1.388731625493765e-3 + z * 2.443315711809948e-5));
  let q = i32(k) & 3;
  if (q == 0) { return vec2<f32>(sn, cs); }
  if (q == 1) { return vec2<f32>(cs, -sn); }
  if (q == 2) { return vec2<f32>(-sn, -cs); }
  return vec2<f32>(-cs, sn);
}
fn sin_acc(x: f32) -> f32 { return sincos_acc(x).x; }
fn atan_acc(x0: f32) -> f32 {
  var x = abs(x0); var y = 0.0;
  if (x > 2.414213562373095) { y = PIO2F; x = -1.0 / x; }
  else if (x > 0.4142135623730950) { y = PIO4F; x = (x - 1.0) / (x + 1.0); }
  let z = x * x;
  y = y + (x + x * z * (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1));
  return select(y, -y, x0 < 0.0);
}
fn atan2_acc(y: f32, x: f32) -> f32 {
  let ax = abs(x); let ay = abs(y);
  if (ax == 0.0 && ay == 0.0) { return 0.0; }
  var r: f32;
  if (ay <= ax) { r = atan_acc(ay / ax); } else { r = PIO2F - atan_acc(ax / ay); }
  if (x < 0.0) { r = 3.141592653589793 - r; }
  return select(r, -r, y < 0.0);
}
fn asin_acc(x: f32) -> f32 { return atan2_acc(x, sqrt_acc(max((1.0 - x) * (1.0 + x), 0.0))); }
fn exp_acc(x: f32) -> f32 {
  let k = floor(x * 1.4426950408889634 + 0.5);
  let r = (x - k * 0.693359375) + k * 2.12194440e-4;
  let z = r * r;
  let p = (((((1.9875691500e-4 * r + 1.3981999507e-3) * r + 8.3334519073e-3) * r + 4.1665795894e-2) * r + 1.6666665459e-1) * r + 5.0000001201e-1) * z + r + 1.0;
  return ldexp(p, i32(k));
}
fn tanh_acc(x: f32) -> f32 {
  let a = abs(x);
  if (a > 9.0) { return select(1.0, -1.0, x < 0.0); }
  if (a >= 0.625) { let e = exp_acc(2.0 * a); let t = 1.0 - 2.0 / (e + 1.0); return select(t, -t, x < 0.0); }
  let z = x * x;
  return x + x * z * ((((-5.70498872745e-3 * z + 2.06390887954e-2) * z - 5.37397155531e-2) * z + 1.33314422036e-1) * z - 3.33332819422e-1);
}

// ---------------------------------------------------------------------------------------------- 64-bit integers
fn i64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let lo = a.x + b.x; return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x)); }
fn i64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let lo = a.x - b.x; return vec2<u32>(lo, a.y - b.y - select(0u, 1u, a.x < b.x)); }
fn i64_neg(a: vec2<u32>) -> vec2<u32> { let lo = ~a.x + 1u; return vec2<u32>(lo, ~a.y + select(0u, 1u, lo == 0u)); }
fn i64_isneg(a: vec2<u32>) -> bool { return (a.y >> 31u) == 1u; }
/** integer · inv (inv = 2^-FB) as f32: sign-magnitude, so small negative values keep their precision */
fn i64_f(a: vec2<u32>, inv: f32) -> f32 {
  let neg = i64_isneg(a);
  let m = select(a, i64_neg(a), neg);
  let f = (f32(m.y) * 4294967296.0 + f32(m.x)) * inv;
  return select(f, -f, neg);
}
/** round-toward-zero of f · 2^FB as a 64-bit integer; sc = 2^(FB-32); saturates at ±2^63 */
fn f_i64(f: f32, sc: f32) -> vec2<u32> {
  let a = min(abs(f) * sc, 2147483520.0);
  let hi = floor(a);
  let m = vec2<u32>(u32((a - hi) * 4294967296.0), u32(hi));
  return select(m, i64_neg(m), f < 0.0);
}
fn ld64(o: u32) -> vec2<u32> { return vec2<u32>(ist[o], ist[o + 1u]); }
fn st64(o: u32, v: vec2<u32>) { ist[o] = v.x; ist[o + 1u] = v.y; }
/** acc += v (exact 64-bit add from two 32-bit atomics: the carry is decided by the value the low word had) */
fn acc_add(w: u32, v: vec2<u32>) {
  if (v.x == 0u && v.y == 0u) { return; }
  let old = atomicAdd(&acc[w], v.x);
  let carry = select(0u, 1u, old + v.x < old);
  atomicAdd(&acc[w + 1u], v.y + carry);
}
fn acc_take(w: u32, inv: f32) -> f32 {
  let v = vec2<u32>(atomicLoad(&acc[w]), atomicLoad(&acc[w + 1u]));
  atomicStore(&acc[w], 0u); atomicStore(&acc[w + 1u], 0u);
  return i64_f(v, inv);
}
fn addF(b: u32, i: u32, f: vec3<f32>, sc: f32) {
  let w = A_F + (b * N + i) * 6u;
  acc_add(w, f_i64(f.x, sc)); acc_add(w + 2u, f_i64(f.y, sc)); acc_add(w + 4u, f_i64(f.z, sc));
}
fn addT(b: u32, j: u32, t: vec3<f32>, sc: f32) {
  let w = A_T + (b * M + j) * 6u;
  acc_add(w, f_i64(t.x, sc)); acc_add(w + 2u, f_i64(t.y, sc)); acc_add(w + 4u, f_i64(t.z, sc));
}

// ---------------------------------------------------------------------------------------------- positions
fn xoff(b: u32, i: u32, prev: bool) -> u32 { return select(I_X, I_XP, prev) + (b * N + i) * 6u; }
/** X_i − X_j (f32, exact integer difference converted once) */
fn pdiff(b: u32, i: u32, j: u32, prev: bool, inv: f32) -> vec3<f32> {
  let oi = xoff(b, i, prev); let oj = xoff(b, j, prev);
  return vec3<f32>(i64_f(i64_sub(ld64(oi), ld64(oj)), inv), i64_f(i64_sub(ld64(oi + 2u), ld64(oj + 2u)), inv), i64_f(i64_sub(ld64(oi + 4u), ld64(oj + 4u)), inv));
}
/** absolute position (display / outputs only) */
fn pabs(b: u32, i: u32, prev: bool, inv: f32) -> vec3<f32> {
  let o = xoff(b, i, prev);
  return vec3<f32>(i64_f(ld64(o), inv), i64_f(ld64(o + 2u), inv), i64_f(ld64(o + 4u), inv));
}

// ---------------------------------------------------------------------------------------------- quaternions (w, x, y, z)
fn ldq(o: u32) -> vec4<f32> { return vec4<f32>(fst[o], fst[o + 1u], fst[o + 2u], fst[o + 3u]); }
fn stq(o: u32, q: vec4<f32>) { fst[o] = q.x; fst[o + 1u] = q.y; fst[o + 2u] = q.z; fst[o + 3u] = q.w; }
fn qoff(b: u32, j: u32, prev: bool) -> u32 { return select(I_Q, I_QP, prev) + (b * M + j) * 8u; }
/** segment quaternion (fixed point) as f32 */
fn ldqi(o: u32) -> vec4<f32> { return vec4<f32>(i64_f(ld64(o), QINV), i64_f(ld64(o + 2u), QINV), i64_f(ld64(o + 4u), QINV), i64_f(ld64(o + 6u), QINV)); }
fn qconj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(q.x, -q.y, -q.z, -q.w); }
fn qnorm(q: vec4<f32>) -> vec4<f32> { let n = sqrt_acc(dot(q, q)); return select(q / n, q, n == 0.0); }
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w, a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                   a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y, a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}
struct Fr { x: vec3<f32>, y: vec3<f32>, z: vec3<f32> };
fn qframe(q0: vec4<f32>) -> Fr {
  let q = qnorm(q0); let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  return Fr(vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
            vec3<f32>(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
            vec3<f32>(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)));
}
/** frame → quaternion (Shepperd, the per-iteration variant) */
fn frameq(f: Fr) -> vec4<f32> {
  let m00 = f.x.x; let m10 = f.x.y; let m20 = f.x.z; let m01 = f.y.x; let m11 = f.y.y; let m21 = f.y.z; let m02 = f.z.x; let m12 = f.z.y; let m22 = f.z.z;
  let t = m00 + m11 + m22;
  var q: vec4<f32>;
  if (t > 0.0) { let s = sqrt(max(t + 1.0, 0.0)) * 2.0; q = vec4<f32>(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s); }
  else if (m00 >= m11 && m00 >= m22) { let s = sqrt(max(1.0 + m00 - m11 - m22, 0.0)) * 2.0; q = vec4<f32>((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s); }
  else if (m11 >= m22) { let s = sqrt(max(1.0 + m11 - m00 - m22, 0.0)) * 2.0; q = vec4<f32>((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s); }
  else { let s = sqrt(max(1.0 + m22 - m00 - m11, 0.0)) * 2.0; q = vec4<f32>((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s); }
  return qnorm(q);
}
/** minimal rotation taking unit a onto unit b, applied to v */
fn rotb(a: vec3<f32>, b: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  let w = cross(a, b); let c = dot(a, b); let s2 = dot(w, w);
  let fac = select(0.5, (1.0 - c) / max(s2, 1e-24), s2 > 1e-24);
  let kv = cross(w, v);
  return v + kv + fac * cross(w, kv);
}
fn unit3(v: vec3<f32>) -> vec3<f32> { return v / max(len_acc(v), 1e-12); }
fn ld3(o: u32) -> vec3<f32> { return vec3<f32>(fst[o], fst[o + 1u], fst[o + 2u]); }
fn st3(o: u32, v: vec3<f32>) { fst[o] = v.x; fst[o + 1u] = v.y; fst[o + 2u] = v.z; }
fn ldfr(o: u32) -> Fr { return Fr(ld3(o), ld3(o + 3u), ld3(o + 6u)); }
fn gld3(o: u32) -> vec3<f32> { return vec3<f32>(geo[o], geo[o + 1u], geo[o + 2u]); }

fn inst_live(b: u32) -> u32 { return select(0u, 1u, ctl[b].done == 0u && ctl[b].iter < ctl[b].stop_at); }
fn ramp_of(b: u32) -> f32 {
  let rn = ip[b].ramp_iter;
  return select(min(1.0, 0.01 + 0.99 * f32(ctl[b].iter) / f32(rn)), 1.0, rn == 0u);
}
fn reduce_store(b: u32, kern: u32, wg: u32, lid: u32, s: vec4<f32>, m: vec4<f32>) {
  rs[lid] = s; rm[lid] = m; workgroupBarrier();
  for (var k = 32u; k > 0u; k = k >> 1u) {
    if (lid < k) { rs[lid] = rs[lid] + rs[lid + k]; rm[lid] = max(rm[lid], rm[lid + k]); }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let o = ((b * NSLOT + kern) * NWG + wg) * 8u;
    part[o] = rs[0].x; part[o + 1u] = rs[0].y; part[o + 2u] = rs[0].z; part[o + 3u] = rs[0].w;
    part[o + 4u] = rm[0].x; part[o + 5u] = rm[0].y; part[o + 6u] = rm[0].z; part[o + 7u] = rm[0].w;
  }
}

// ============================================================================================== 1 · prep
// segments: (rollback), chord, frame, compensated stretch → node forces; nodes: (rollback), tangent, mean frame
@compute @workgroup_size(64)
fn k_prep(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  let rb = ctl[b].rollback == 1u;
  let pinv = ip[b].pinv; let psc = ip[b].psc;
  var s_st = 0.0; var m_n = 0.0;
  let t = gid.x;
  if (t < M) {
    let j = t;
    let a = topo[T_SEG0 + j]; let c = topo[T_SEG1 + j];
    let oq = qoff(b, j, false);
    if (rb) { let op = qoff(b, j, true); for (var k = 0u; k < 8u; k++) { ist[oq + k] = ist[op + k]; } fst[F_W + b * M + j] = 0.0; }
    let q = ldqi(select(oq, qoff(b, j, true), rb));
    let e = pdiff(b, c, a, rb, pinv);
    let Ls = len_acc(e);
    st3(F_E + (b * M + j) * 3u, e); fst[F_LSEG + b * M + j] = Ls;
    let fr = qframe(q);
    let orr = F_R + (b * M + j) * 9u;
    st3(orr, fr.x); st3(orr + 3u, fr.y); st3(orr + 6u, fr.z);
    // δ = (X_c − X_a) − E_ref exactly, then ext = |E_ref + δ| − l0 without cancellation
    let oc = xoff(b, c, rb); let oa = xoff(b, a, rb); let oe = I_EREF + (b * M + j) * 6u;
    let d = vec3<f32>(i64_f(i64_sub(i64_sub(ld64(oc), ld64(oa)), ld64(oe)), pinv),
                      i64_f(i64_sub(i64_sub(ld64(oc + 2u), ld64(oa + 2u)), ld64(oe + 2u)), pinv),
                      i64_f(i64_sub(i64_sub(ld64(oc + 4u), ld64(oa + 4u)), ld64(oe + 4u)), pinv));
    let E = ld3(F_EREF + (b * M + j) * 3u);
    let l0 = geo[G_L0 + j];
    let num = fst[F_C0 + b * M + j] + 2.0 * dot(E, d) + dot(d, d);
    let ext = num / (Ls + l0);
    let EA = geo[G_STIFF + (b * M + j) * 4u];
    let Nn = EA * ext / l0;
    fst[F_NAX + b * M + j] = Nn;
    let fs = Nn * (e / max(Ls, 1e-12));
    addF(b, a, fs, ip[b].fsc); addF(b, c, -fs, ip[b].fsc);
    s_st = 0.5 * Nn * ext; m_n = abs(Nn);
  } else if (t < M + N) {
    let i = t - M;
    let ox = xoff(b, i, false);
    if (rb) {
      let op = xoff(b, i, true);
      for (var k = 0u; k < 6u; k++) { ist[ox + k] = ist[op + k]; }
      st3(F_V + (b * N + i) * 3u, vec3<f32>(0.0));
    }
    let sa = topo[T_NSA + i]; let sb = topo[T_NSB + i];
    var T = vec3<f32>(0.0);
    var qa = vec4<f32>(0.0); var qb = vec4<f32>(0.0);
    if (sa != NONE) { T = T + unit3(pdiff(b, topo[T_SEG1 + sa], i, rb, pinv)); qa = ldqi(qoff(b, sa, rb)); }
    if (sb != NONE) { T = T + unit3(pdiff(b, i, topo[T_SEG0 + sb], rb, pinv)); qb = ldqi(qoff(b, sb, rb)); }
    st3(F_TN + (b * N + i) * 3u, unit3(T));
    let qref = select(qb, qa, sa != NONE);
    var accq = vec4<f32>(0.0);
    if (sa != NONE) { accq = accq + select(qa, -qa, dot(qa, qref) < 0.0); }
    if (sb != NONE) { accq = accq + select(qb, -qb, dot(qb, qref) < 0.0); }
    stq(F_QN + (b * N + i) * 4u, qnorm(accq));
  }
  reduce_store(b, 0u, wid.x, lid.x, vec4<f32>(s_st, 0.0, 0.0, 0.0), vec4<f32>(m_n, 0.0, 0.0, 0.0));
}

// ============================================================================================== 2 · forces
// joints (bending + twist), clamps, contacts
struct Ev { P: vec3<f32>, t: vec3<f32>, R: Fr, L: f32, seg: u32, tp: f32, a: u32, c: u32 };

fn useg(b: u32, k: u32, side: u32) -> vec3<u32> {     // (segment, fraction bits, at-end flag) of u on that side
  let v = ld64(I_U + (b * K + k) * 4u + 2u * side);
  let ns = topo[T_KNSEG + 2u * k + side];
  var i = v.y; var fr = v.x; var endf = 0u;
  if (i64_isneg(v)) { i = 0u; fr = 0u; }
  else if (i >= ns) { i = max(ns, 1u) - 1u; fr = 0u; endf = 1u; }
  return vec3<u32>(topo[T_KOFF + 2u * k + side] + i, fr, endf);
}

/** free weight of a contact point: (1 − t)² on the segment's first node + t² on its second; a node fixed in all three
 *  translations counts 0 (PolylineContact.freeWeight) */
fn free_weight(b: u32, e: Ev) -> f32 {
  let fa = (ist[I_FIX + b * N + e.a] & 7u) == 7u; let fc = (ist[I_FIX + b * N + e.c] & 7u) == 7u;
  return select((1.0 - e.tp) * (1.0 - e.tp), 0.0, fa) + select(e.tp * e.tp, 0.0, fc);
}
/** cubic Hermite (C¹) centreline on a side, in coordinates relative to node o; frame = nlerp of node frames */
fn evalside(b: u32, k: u32, side: u32, o: u32, pinv: f32) -> Ev {
  let us = useg(b, k, side);
  let seg = us.x;
  let a = topo[T_SEG0 + seg]; let c = topo[T_SEG1 + seg];
  let th = f32(us.y & 0xffffff00u) * 2.3283064365386963e-10;      // 24-bit part of t
  let tl = f32(us.y & 0xffu) * 2.3283064365386963e-10;              // remaining 8 bits (first-order term)
  var t = th; var tlo = tl;
  if (us.z == 1u) { t = 1.0; tlo = 0.0; }
  let xa = pdiff(b, a, o, false, pinv); let xc = pdiff(b, c, o, false, pinv);
  let Ls = fst[F_LSEG + b * M + seg];
  let ta = ld3(F_TN + (b * N + a) * 3u); let tc = ld3(F_TN + (b * N + c) * 3u);
  let t2 = t * t; let t3 = t2 * t;
  let h00 = 2.0 * t3 - 3.0 * t2 + 1.0; let h10 = t3 - 2.0 * t2 + t; let h01 = -2.0 * t3 + 3.0 * t2; let h11 = t3 - t2;
  let d00 = 6.0 * t2 - 6.0 * t; let d10 = 3.0 * t2 - 4.0 * t + 1.0; let d01 = -6.0 * t2 + 6.0 * t; let d11 = 3.0 * t2 - 2.0 * t;
  let dP = d00 * xa + (d10 * Ls) * ta + d01 * xc + (d11 * Ls) * tc;
  let Pp = h00 * xa + (h10 * Ls) * ta + h01 * xc + (h11 * Ls) * tc + tlo * dP;
  let tg = unit3(dP);
  var qa = ldq(F_QN + (b * N + a) * 4u); var qc = ldq(F_QN + (b * N + c) * 4u);
  qc = select(qc, -qc, dot(qa, qc) < 0.0);
  let tt = t + tlo;
  let fr = qframe(qnorm((1.0 - tt) * qa + tt * qc));
  let y = unit3(fr.y - dot(fr.y, tg) * tg);
  let x = cross(y, tg);
  return Ev(Pp, tg, Fr(x, y, tg), Ls, seg, tt, a, c);
}

/** half-extent h(n) of the section facing sign·n and its smoothed support point r */
fn support(R: Fr, n: vec3<f32>, sg: f32, w: f32, h: f32) -> vec4<f32> {
  let nx = dot(n, R.x) * sg; let ny = dot(n, R.y) * sg;
  let hh = 0.5 * w * abs(nx) + 0.5 * h * abs(ny);
  let r = (0.5 * w * tanh_acc(nx / SUPPORT_SMOOTH)) * R.x + (0.5 * h * tanh_acc(ny / SUPPORT_SMOOTH)) * R.y;
  return vec4<f32>(r, hh);
}
fn wallTan(sideCut: u32, R: Fr, n: vec3<f32>, m: vec3<f32>) -> f32 {
  let sg = select(-1.0, 1.0, sideCut == 0u);
  let nx = dot(n, R.x) * sg; let ny = dot(n, R.y) * sg;
  let fx = sign(nx) * R.x; let fy = sign(ny) * R.y;
  let f = select(fy, fx, dot(fx, m) > 0.0);
  let cp = clamp(abs(dot(f, n)), 1e-6, 1.0);
  return sqrt_acc(max((1.0 - cp) * (1.0 + cp), 0.0)) / cp;               // tan(acos(cp))
}
fn u_add(b: u32, k: u32, side: u32, du: f32) {
  let o = I_U + (b * K + k) * 4u + 2u * side;
  var v = i64_add(ld64(o), f_i64(du, 1.0));
  let ns = topo[T_KNSEG + 2u * k + side];
  if (i64_isneg(v)) { v = vec2<u32>(0u, 0u); }
  else if (v.y > ns || (v.y == ns && v.x > 0u)) { v = vec2<u32>(0u, ns); }
  st64(o, v);
}

@compute @workgroup_size(64)
fn k_force(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  let pinv = ip[b].pinv; let fsc = ip[b].fsc; let tsc = ip[b].tsc;
  var s = vec4<f32>(0.0); var mx = vec4<f32>(0.0, -3e38, 0.0, 0.0);
  let t = gid.x;
  if (t < P) {
    // ---------------------------------------------------------------- joint: bending (Bergou) and twist
    let j = t;
    let s0 = topo[T_J0 + j]; let s1 = topo[T_J1 + j];
    let a = ld3(F_E + (b * M + s0) * 3u); let bb = ld3(F_E + (b * M + s1) * 3u);
    let la = fst[F_LSEG + b * M + s0]; let lb = fst[F_LSEG + b * M + s1];
    let den = max(la * lb + dot(a, bb), 1e-12);
    let kb = 2.0 * cross(a, bb) / den;
    let Ra = ldfr(F_R + (b * M + s0) * 9u); let Rb = ldfr(F_R + (b * M + s1) * 9u);
    let m1 = unit3(Ra.x + Rb.x); let m2 = unit3(Ra.y + Rb.y);
    let lbar = geo[G_JLEN + j];
    let om = gld3(G_OM + (b * P + j) * 3u);
    let so = G_STIFF + (b * M + s0) * 4u;
    let EIx = geo[so + 1u]; let EIy = geo[so + 2u]; let GJ = geo[so + 3u];
    let k1 = dot(kb, m1) / lbar; let k2 = dot(kb, m2) / lbar;
    let M1 = EIx * (k1 - om.x); let M2 = EIy * (k2 - om.y);
    let g = M1 * m1 + M2 * m2; let kg = dot(kb, g);
    let Fim = (2.0 * cross(bb, g) - (lb * (a / la) + bb) * kg) / den;
    let Fip = (2.0 * cross(a, g) + (la * (bb / lb) + a) * kg) / den;
    addF(b, topo[T_SEG0 + s0], Fim, fsc); addF(b, topo[T_JN + j], -(Fim + Fip), fsc); addF(b, topo[T_SEG1 + s1], Fip, fsc);
    // twist: the swing–twist split of conj(Qa)·Qb about the local Z, with Qb − ±Qa taken exactly in fixed point
    // (equals the angle between Ra.x parallel-transported onto chord b and Rb.x)
    let oa = qoff(b, s0, false); let ob = qoff(b, s1, false);
    let qaf = ldqi(oa); let qbf = ldqi(ob);
    let flip = dot(qaf, qbf) < 0.0;
    var dq = vec4<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
      let va = ld64(oa + 2u * k); let vb = ld64(ob + 2u * k);
      dq[k] = i64_f(select(i64_sub(vb, va), i64_add(vb, va), flip), QINV);
    }
    let rq = qmul(qconj(qaf), dq) * select(1.0, -1.0, flip);
    let phi = 2.0 * atan2_acc(rq.w, 1.0 + rq.x);
    let tau = phi / lbar;
    let Mt = GJ * (tau - om.z);
    addT(b, s0, Mt * Ra.z, tsc); addT(b, s1, -Mt * Rb.z, tsc);
    st3(F_MJ + (b * P + j) * 3u, vec3<f32>(M1, M2, Mt));
    s.x = 0.5 * lbar * (M1 * (k1 - om.x) + M2 * (k2 - om.y));
    s.y = 0.5 * lbar * Mt * (tau - om.z);
    mx.x = max(max(abs(M1), abs(M2)), abs(Mt));
  } else if (t < P + C) {
    // ---------------------------------------------------------------- clamp: boundary joint against the frozen chord
    let c = t - P;
    let nd = topo[T_CN + c]; let nb = topo[T_CB + c]; let sg = topo[T_CS + c];
    let a = gld3(G_CDIR + (b * C + c) * 3u);
    let bv = pdiff(b, nb, nd, false, pinv);
    let lb = len_acc(bv);
    let den = max(lb + dot(a, bv), 1e-12);
    let kb = 2.0 * cross(a, bv) / den;
    let R = ldfr(F_R + (b * M + sg) * 9u);
    let lc = geo[G_CLEN + c];
    let EIx = geo[G_CEI + (b * C + c) * 2u]; let EIy = geo[G_CEI + (b * C + c) * 2u + 1u];
    let k1 = dot(kb, R.x) / lc; let k2 = dot(kb, R.y) / lc;
    let M1 = EIx * k1; let M2 = EIy * k2;
    let g = M1 * R.x + M2 * R.y; let kg = dot(kb, g);
    let Fnb = (2.0 * cross(a, g) + (bv / max(lb, 1e-12) + a) * kg) / den;
    addF(b, nb, Fnb, fsc); addF(b, nd, -Fnb, fsc);
    fst[F_CM + (b * C + c) * 2u] = M1; fst[F_CM + (b * C + c) * 2u + 1u] = M2;
    s.x = 0.5 * lc * (M1 * k1 + M2 * k2);
  } else if (t < P + C + K) {
    // ---------------------------------------------------------------- contact at a crossing
    let k = t - P - C;
    let kk = b * K + k;
    let pr = ip[b];
    let enabled = topo[T_KEN + k] == 1u;
    let locked = ist[I_LOCK + kk] == 1u;
    let ramp = ramp_of(b);
    let kn = ramp * pr.strength; let kt = ramp * pr.stick;
    let o = topo[T_SEG0 + useg(b, k, 0u).x];                      // stencil origin: first node of A's segment
    let A = evalside(b, k, 0u, o, pinv);
    let B = evalside(b, k, 1u, o, pinv);
    let tA = A.t; let tB = B.t;
    let sinT = clamp(len_acc(cross(tA, tB)), 1e-6, 1.0);
    let wA = geo[G_WK + 2u * k]; let wB = geo[G_WK + 2u * k + 1u]; let hA = geo[G_HK + 2u * k]; let hB = geo[G_HK + 2u * k + 1u];
    let fcA = geo[G_FC90 + (b * K + k) * 2u]; let fcB = geo[G_FC90 + (b * K + k) * 2u + 1u];
    if (ctl[b].al_update == 1u) {
      // safeguarded Uzawa step (PolylineContact.alValue): window-mean penetration, no multiplier at a support-controlled
      // crossing (contact point on a fixed node), |λ| ≤ the crushing capacity of the face overlap
      let penMean = fst[F_PENACC + kk] / f32(max(pr.al_every, 1u));
      let fwA = free_weight(b, A); let fwB = free_weight(b, B);
      let pinned = fwA < AL_FREE_MIN || fwB < AL_FREE_MIN;
      let cap = min(fcA, fcB) * wA * wB / sinT;
      var lv = fst[F_LAM + kk] + kn * penMean;
      if (!locked) { lv = max(lv, 0.0); }
      lv = clamp(lv, -cap, cap);
      fst[F_LAM + kk] = select(lv, 0.0, !enabled || pinned);
      fst[F_PENACC + kk] = 0.0;
    }
    var n = cross(tA, tB);
    let nn = len_acc(n);
    let np = ld3(F_NPREV + kk * 3u);
    n = select(n / max(nn, 1e-12), np, nn < 1e-6);
    n = select(n, -n, dot(n, np) < 0.0);
    st3(F_NPREV + kk * 3u, n);
    let S = B.P - A.P;
    st3(F_MID + kk * 3u, pabs(b, o, false, pinv) + 0.5 * (A.P + B.P));
    let g = dot(S, n);
    let supA = support(A.R, n, 1.0, wA, hA); let supB = support(B.R, n, -1.0, wB, hB);
    var D = supA.w + supB.w;
    let seat = pr.seat == 1u;
    let od = F_DENT + kk * 2u;
    if (seat && pr.seat_gap == 1u) { D = D - (fst[od] + fst[od + 1u] + fst[F_DENTE + kk]); }
    let pen = D - g;
    if (ctl[b].iter >= pr.ramp_iter) { fst[F_PENACC + kk] = fst[F_PENACC + kk] + pen; }
    var Fn = 0.0; var act = false;
    if (pr.aug == 1u) {
      let trial = fst[F_LAM + kk] + kn * pen;
      act = (trial > 0.0 || locked) && enabled;
      Fn = select(0.0, select(max(trial, 0.0), trial, locked), act);
    } else {
      act = (pen > 0.0 || locked) && enabled;
      Fn = select(0.0, kn * pen, act);
    }
    let St = S - g * n;
    let taa = dot(tA, tA); let tbb = dot(tB, tB); let tab = dot(tA, tB);
    let sa = dot(St, tA); let sb = dot(St, tB);
    let det = taa * tbb - tab * tab; let ok = det > 1e-8;
    let ca = select(0.0, (sa * tbb - sb * tab) / det, ok); let cb = select(0.0, (sb * taa - sa * tab) / det, ok);
    let Ftv = kt * len_acc(St);
    // directional seat friction (W3)
    var mu = pr.mu;
    if (seat) {
      let nrm = len_acc(St); let dhat = St / max(nrm, 1e-12);
      for (var jj = 0u; jj < 2u; jj++) {
        if (fst[od + jj] > 0.0 && nrm > 1e-12) {
          let tCut = select(tA, tB, jj == 0u);
          var Rcut = A.R; if (jj == 0u) { Rcut = B.R; }
          let m = unit3(cross(n, tCut));
          let sgn = select(-1.0, 1.0, jj == 0u);
          let dj = dot(dhat, m) * sgn;
          let tp = wallTan(1u - jj, Rcut, n, sign(dj) * m * sgn);
          fst[F_TANPSI + kk * 2u + jj] = tp;
          let add = abs(dj) * select(9.0 * pr.mu, (pr.mu + tp) / max(1.0 - pr.mu * tp, 1e-6) - pr.mu, pr.mu * tp < 1.0);
          mu = mu + min(add, 9.0 * pr.mu);
        }
      }
    }
    let limit = select(mu * abs(Fn), 3e38, locked);
    let slip = act && ok && Ftv > limit;
    let free = !act && ok && enabled;
    var sc = 0.0;
    if (slip) { sc = clamp(pr.slip_rate, 1e-3, 1.0) * (1.0 - limit / max(Ftv, 1e-12)); }
    if (free) { sc = 1.0; }
    let eps = max(pr.epsilon, 1e-9);
    let da = clamp(ca * sc, -eps, eps); let db = clamp(cb * sc, -eps, eps);
    if (slip || free) {
      u_add(b, k, 0u, da / max(A.L, 1e-12));
      u_add(b, k, 1u, -db / max(B.L, 1e-12));
    }
    if (slip) {
      let os = I_SLIP + kk * 2u;
      st64(os, i64_add(ld64(os), f_i64(len_acc(da * tA + db * tB), pr.psc)));
    }
    let Se = select(vec3<f32>(0.0), St - da * tA - db * tB, act);
    let fT = kt * Se;
    let FA = -Fn * n + fT; let FB = Fn * n - fT;
    fst[F_FN + kk] = Fn; fst[F_FT + kk] = len_acc(fT); fst[F_GAP + kk] = g;
    // classification (W1): contact type, nominal area, pressure, utilisation
    var cface = vec2<f32>(0.0); var z0 = vec2<f32>(0.0); var edge = vec2<u32>(0u); var phiv = vec2<f32>(0.0);
    for (var sd = 0u; sd < 2u; sd++) {
      var R = B.R; if (sd == 0u) { R = A.R; }
      let sgs = select(-1.0, 1.0, sd == 0u);
      let wv = select(wB, wA, sd == 0u); let hv = select(hB, hA, sd == 0u);
      let nx = dot(n, R.x) * sgs; let ny = dot(n, R.y) * sgs;
      let e = abs(tanh_acc(nx / SUPPORT_SMOOTH)) * abs(tanh_acc(ny / SUPPORT_SMOOTH)) > pr.edge_thr;
      let yf = abs(ny) >= abs(nx);
      cface[sd] = select(hv, wv, yf); z0[sd] = select(wv, hv, yf); edge[sd] = select(0u, 1u, e);
      phiv[sd] = atan_acc(min(abs(nx), abs(ny)) / max(max(abs(nx), abs(ny)), 1e-12));
    }
    fst[F_PHI + kk * 2u] = phiv.x; fst[F_PHI + kk * 2u + 1u] = phiv.y;
    let theta = asin_acc(sinT);
    fst[F_THETA + kk] = theta;
    let nEdge = edge.x + edge.y;
    var faceSide = 3u;                                              // 3 = none (−1)
    if (nEdge == 1u) { faceSide = select(1u, 0u, edge.y == 1u); }
    let Fa = abs(Fn);
    var Aa = cface.x * cface.y / sinT;
    var pres = Fa / max(Aa, 1e-12);
    var util = pres / max(min(geo[G_HS + (b * K + k) * 2u], geo[G_HS + (b * K + k) * 2u + 1u]), 1e-12);
    if (nEdge == 1u) {
      let fs = faceSide; let es = 1u - fs;
      let ell = cface[fs] / sinT;
      let kap = geo[G_E90 + (b * K + k) * 2u + fs] / max(z0[fs], 1e-12);
      let phiE = max(phiv[es], 1e-4);
      let bw = sqrt_acc(2.0 * Fa / max(ell * kap * phiE, 1e-12));
      let full = bw >= cface[es];
      let Afe = select(bw * ell, cface.x * cface.y / sinT, full);
      let pfe = select(kap * phiE * bw, Fa / max(Afe, 1e-12), full);
      Aa = Afe; pres = pfe; util = pfe / max(select(fcB, fcA, es == 0u), 1e-12);
    } else if (nEdge == 2u) {
      let fc = min(fcA, fcB);
      Aa = Fa / max(fc, 1e-12); pres = select(0.0, fc, Fa > 0.0); util = select(0.0, 1.0, Fa > 0.0);
    }
    fst[F_ANOM + kk] = Aa; fst[F_PRES + kk] = pres; fst[F_UTIL + kk] = util;
    // seat law (W3): climb / escape of existing notches, then rate-limited dent formation
    if (seat) {
      let inc = da * tA - db * tB;
      for (var jj = 0u; jj < 2u; jj++) {
        let tCut = select(tA, tB, jj == 0u);
        let m = unit3(cross(n, tCut));
        let stp = dot(inc, m) * select(-1.0, 1.0, jj == 0u);
        let has = fst[od + jj] > 0.0;
        let oc = F_CLIMB + kk * 2u + jj;
        if (has && slip) { fst[oc] = fst[oc] + stp; }
        if (has && abs(fst[oc]) >= fst[od + jj] / max(fst[F_TANPSI + kk * 2u + jj], 1e-6)) { fst[od + jj] = 0.0; fst[oc] = 0.0; }
      }
      if (ctl[b].iter >= pr.dent_after) {
        let c0v = sincos_acc(phiv.x); let c1v = sincos_acc(phiv.y);
        let sp0 = c0v.x * c0v.y; let sp1 = c1v.x * c1v.y; let st_ = sin_acc(theta);
        if (nEdge == 2u && act && !locked) {
          let dEst = sqrt_acc(Fa / max(min(fcA, fcB), 1e-12) * st_ * sp0 * sp1);
          for (var jj = 0u; jj < 2u; jj++) { let cur = fst[od + jj]; fst[od + jj] = max(cur, min(cur + pr.dent_rate * dEst, dEst)); }
        }
        if (nEdge == 1u && act && !locked) {
          let fs = faceSide; let es = 1u - fs;
          let ell = cface[fs] / max(st_, 1e-6);
          let dE = Fa / max(select(fcB, fcA, es == 0u) * ell, 1e-12) * select(sp1, sp0, es == 0u);
          fst[F_DENTE + kk] = max(fst[F_DENTE + kk], dE);
        }
      }
    }
    // distribute to the segment nodes (1 − t, t) and the lever-arm torques
    addF(b, A.a, (1.0 - A.tp) * FA, fsc); addF(b, A.c, A.tp * FA, fsc);
    addF(b, B.a, (1.0 - B.tp) * FB, fsc); addF(b, B.c, B.tp * FB, fsc);
    let tf = pr.torsion_factor;
    addT(b, A.seg, tf * cross(supA.xyz, FA), tsc); addT(b, B.seg, tf * cross(supB.xyz, FB), tsc);
    // rotational friction about n (W2)
    var eRot = 0.0;
    if (pr.rot_fric == 1u) {
      let thc = atan2_acc(dot(cross(tA, tB), n), dot(tA, tB));
      let o0 = F_TH0 + kk;
      if (fst[o0] > 1e29 && act) { fst[o0] = thc; }
      let okr = act && fst[o0] < 1e29;
      let sinT2 = max(sin_acc(theta), 1e-6);
      var req = 0.38 * sqrt_acc(wA * wB / sinT2);
      if (nEdge == 1u) { req = 0.25 * select(wB, wA, faceSide == 0u) / sinT2; }
      else if (nEdge == 2u) { req = 0.38 * sqrt_acc(max(Aa, 1e-12)); }
      let kr = kt * req * req;
      var dth = select(0.0, thc - fst[o0], okr);
      dth = dth + PI; dth = dth - floor(dth / (2.0 * PI)) * (2.0 * PI); dth = dth - PI;
      var Mr = kr * dth;
      let Mlim = select(pr.rot_factor * pr.mu * Fa * req, 3e38, locked);
      if (okr && abs(Mr) > Mlim) {
        let adv = clamp(clamp(pr.slip_rate, 1e-3, 1.0) * (1.0 - Mlim / max(abs(Mr), 1e-12)) * dth, -0.05, 0.05);
        fst[o0] = fst[o0] + adv; fst[F_RSLIP + kk] = fst[F_RSLIP + kk] + abs(adv); dth = dth - adv;
        Mr = kr * dth;
      }
      Mr = select(0.0, Mr, okr);
      fst[F_MROT + kk] = Mr;
      addT(b, A.seg, Mr * n, tsc); addT(b, B.seg, -Mr * n, tsc);
      eRot = 0.5 * kr * dth * dth;
    }
    // packed outputs: bit0 slip, bit1 stuck, bits 2-3 contact type, bits 4-5 face side
    ist[I_KOUT + kk] = select(0u, 1u, slip) | select(0u, 2u, !(slip || free)) | (nEdge << 2u) | (faceSide << 4u);
    // contact twist stiffness for the segment inertia (post-update segments, as add_twist_stiffness)
    let tfk = max(pr.torsion_factor, 1e-3);
    let kth = tfk * ((Fn + kn * 0.25 * geo[G_KH + k]) * (0.5 * geo[G_KW + k]) / SUPPORT_SMOOTH + kt * (0.5 * geo[G_KH + k]) * (0.5 * geo[G_KH + k]));
    let ksc = pr.ksc;
    acc_add(A_K + (b * M + useg(b, k, 0u).x) * 2u, f_i64(kth, ksc));
    acc_add(A_K + (b * M + useg(b, k, 1u).x) * 2u, f_i64(kth, ksc));
    let ePen = select(0.0, select(kn * pen * pen, Fn * Fn / max(kn, 1e-30), pr.aug == 1u), act);
    s.z = 0.5 * ePen + 0.5 * kt * dot(Se, Se) + eRot;
    mx.y = Fn;
  }
  reduce_store(b, 1u, wid.x, lid.x, s, mx);
}

// ============================================================================================== 3 · segment torques
@compute @workgroup_size(64)
fn k_seg(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  let j = gid.x;
  if (j >= M) { return; }
  let tinv = ip[b].tinv;
  let w = A_T + (b * M + j) * 6u;
  let T = vec3<f32>(acc_take(w, tinv), acc_take(w + 2u, tinv), acc_take(w + 4u, tinv));
  let ktw = acc_take(A_K + (b * M + j) * 2u, ip[b].kinv);
  let e = ld3(F_E + (b * M + j) * 3u); let Ls = max(fst[F_LSEG + b * M + j], 1e-12);
  let tt = e / Ls;
  var tpar = dot(T, tt);
  let cpl = cross(T, tt) / Ls;
  addF(b, topo[T_SEG1 + j], cpl, ip[b].fsc); addF(b, topo[T_SEG0 + j], -cpl, ip[b].fsc);
  if (ist[I_TFIX + b * M + j] == 1u) { tpar = 0.0; }
  fst[F_TPAR + b * M + j] = tpar;
  if (ip[b].contacts == 1u) { fst[F_INERT + b * M + j] = ip[b].mass_scale * max(geo[G_KSEG + b * M + j] + ktw, 1e-12); }
}

// ============================================================================================== 4 · nodes: loads, anchors, integrate
@compute @workgroup_size(64)
fn k_node(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  var s = vec4<f32>(0.0); var mx = vec4<f32>(0.0);
  let i = gid.x;
  if (i < N) {
    let pr = ip[b];
    let w = A_F + (b * N + i) * 6u;
    var F = vec3<f32>(acc_take(w, pr.finv), acc_take(w + 2u, pr.finv), acc_take(w + 4u, pr.finv));
    var load = gld3(G_LC + (b * N + i) * 3u);
    let ll = gld3(G_LL + (b * N + i) * 3u);
    if (ll.x != 0.0 || ll.y != 0.0 || ll.z != 0.0) { let R = qframe(ldq(F_QN + (b * N + i) * 4u)); load = load + R.x * ll.x + R.y * ll.y + R.z * ll.z; }
    if (pr.mode == 1u && i == pr.probe) { load = load + vec3<f32>(ctl[b].pfx, ctl[b].pfy, ctl[b].pfz); }
    F = F + load;
    mx.x = max(max(abs(load.x), abs(load.y)), abs(load.z));
    let fx = ist[I_FIX + b * N + i];
    st3(F_REACT + (b * N + i) * 3u, select(vec3<f32>(0.0), -F, (fx & 8u) != 0u));
    if ((fx & 1u) != 0u) { F.x = 0.0; }
    if ((fx & 2u) != 0u) { F.y = 0.0; }
    if ((fx & 4u) != 0u) { F.z = 0.0; }
    if (fx == 0u) { mx.y = len_acc(F); }
    let ov = F_V + (b * N + i) * 3u;
    var V = ld3(ov);
    if (ctl[b].iter < pr.warmup && pr.contacts == 1u) { F = vec3<f32>(0.0); V = vec3<f32>(0.0); }
    let m = geo[G_MASS + b * N + i];
    V = (V + F / m) * pr.damping;
    if ((fx & 1u) != 0u) { V.x = 0.0; }
    if ((fx & 2u) != 0u) { V.y = 0.0; }
    if ((fx & 4u) != 0u) { V.z = 0.0; }
    st3(ov, V);
    let ox = xoff(b, i, false); let op = xoff(b, i, true);
    for (var k = 0u; k < 6u; k++) { ist[op + k] = ist[ox + k]; }
    st64(ox, i64_add(ld64(ox), f_i64(V.x, pr.psc)));
    st64(ox + 2u, i64_add(ld64(ox + 2u), f_i64(V.y, pr.psc)));
    st64(ox + 4u, i64_add(ld64(ox + 4u), f_i64(V.z, pr.psc)));
    s.x = m * dot(V, V);
  }
  reduce_store(b, 2u, wid.x, lid.x, s, mx);
}

// ============================================================================================== 5 · segment frames: twist, re-align to the new chord
@compute @workgroup_size(64)
fn k_segint(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  var s = vec4<f32>(0.0);
  let j = gid.x;
  if (j < M) {
    let pr = ip[b];
    let oq = qoff(b, j, false);
    let op = qoff(b, j, true);
    for (var k = 0u; k < 8u; k++) { ist[op + k] = ist[oq + k]; }
    let q0 = ldqi(oq);
    let ow = F_W + b * M + j;
    let inert = fst[F_INERT + b * M + j];
    var W = clamp(fst[ow] + fst[F_TPAR + b * M + j] / inert, -TWIST_RATE_MAX, TWIST_RATE_MAX) * pr.damping;
    if (ist[I_TFIX + b * M + j] == 1u) { W = 0.0; }
    fst[ow] = W;
    // Q ← align(t0 → t1) · exp(W t0) · Q, applied as an increment: (1 + d2)(1 + d1) − 1 = d1 + d2 + d2 d1
    let t0 = unit3(ld3(F_E + (b * M + j) * 3u));
    let hw = 0.5 * W;
    let s4 = sin_acc(0.5 * hw);
    let d1 = vec4<f32>(-2.0 * s4 * s4, t0 * sin_acc(hw));
    let t1 = unit3(pdiff(b, topo[T_SEG1 + j], topo[T_SEG0 + j], false, pr.pinv));
    let v = cross(t0, t1); let sv = len_acc(v);
    var d2 = vec4<f32>(0.0);
    if (sv > 0.0) { let th = atan2_acc(sv, dot(t0, t1)); let s8 = sin_acc(0.25 * th); d2 = vec4<f32>(-2.0 * s8 * s8, (v / sv) * sin_acc(0.5 * th)); }
    let dd = d1 + d2 + qmul(d2, d1);
    let dQ = qmul(dd, q0);
    let x = 2.0 * dot(q0, dQ) + dot(dQ, dQ);
    let rt = sqrt_acc(1.0 + x);
    let dtot = dQ - (x / (rt * (1.0 + rt))) * (q0 + dQ);
    for (var k = 0u; k < 4u; k++) { st64(oq + 2u * k, i64_add(ld64(oq + 2u * k), f_i64(dtot[k], QSC))); }
    s.x = inert * W * W;
  }
  reduce_store(b, 3u, wid.x, lid.x, s, vec4<f32>(0.0));
}

// ============================================================================================== 6 · control (one workgroup per instance)
var<workgroup> cs: array<vec4<f32>, 64>;
var<workgroup> cm: array<vec4<f32>, 64>;
var<workgroup> tot: array<vec4<f32>, 8>;
@compute @workgroup_size(64)
fn k_ctrl(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  for (var kern = 0u; kern < NSLOT; kern++) {
    var ss = vec4<f32>(0.0); var mm = vec4<f32>(0.0, -3e38, 0.0, 0.0);
    if (kern != 1u) { mm.y = 0.0; }
    let nwg = select(select(select(NWG_SI, NWG_N, kern == 2u), NWG_F, kern == 1u), NWG_P, kern == 0u);
    for (var w = lid.x; w < nwg; w += 64u) {
      let o = ((b * NSLOT + kern) * NWG + w) * 8u;
      ss = ss + vec4<f32>(part[o], part[o + 1u], part[o + 2u], part[o + 3u]);
      mm = max(mm, vec4<f32>(part[o + 4u], part[o + 5u], part[o + 6u], part[o + 7u]));
    }
    cs[lid.x] = ss; cm[lid.x] = mm; workgroupBarrier();
    for (var k = 32u; k > 0u; k = k >> 1u) {
      if (lid.x < k) { cs[lid.x] = cs[lid.x] + cs[lid.x + k]; cm[lid.x] = max(cm[lid.x], cm[lid.x + k]); }
      workgroupBarrier();
    }
    if (lid.x == 0u) { tot[2u * kern] = cs[0]; tot[2u * kern + 1u] = cm[0]; }
    workgroupBarrier();
  }
  if (lid.x != 0u) { return; }
  let pr = ip[b];
  var c = ctl[b];
  let sP = tot[0]; let mP = tot[1]; let sF = tot[2]; let mF = tot[3]; let sN = tot[4]; let mN = tot[5]; let sS = tot[6];
  var ke = 0.5 * (sN.x + sS.x);
  var rollback = 0u;
  if (pr.kin_damp == 1u && ke < c.ke_prev) { rollback = 1u; ke = 0.0; }
  c.rollback = rollback;
  c.ke_prev = ke; c.ke = ke;
  c.iter = c.iter + 1u;
  var fref = max(mN.x, mP.x);
  if (P > 0u) { fref = max(fref, mF.x / max(pr.jlen_min, 1e-12)); }
  if (pr.contacts == 1u && K > 0u) { fref = max(fref, mF.y); }
  fref = max(fref, 1e-12);
  var resid = mN.y / fref;
  if (c.iter <= pr.warmup) { resid = 3e38; }
  c.resid = resid; c.fref = fref;
  c.e_stretch = sP.x; c.e_bend = sF.x; c.e_tors = sF.y; c.e_cont = sF.z;
  c.al_update = select(0u, 1u, pr.aug == 1u && pr.al_every > 0u && c.iter > pr.ramp_iter && ((c.iter - pr.ramp_iter) % pr.al_every) == 0u);
  let ho = F_HIST + (b * HN + (c.iter - 1u) % HN) * 8u;
  fst[ho] = ke; fst[ho + 1u] = sP.x + sF.x + sF.y + sF.z; fst[ho + 2u] = sF.x; fst[ho + 3u] = sF.y;
  fst[ho + 4u] = sP.x; fst[ho + 5u] = sF.z; fst[ho + 6u] = resid; fst[ho + 7u] = f32(c.iter);
  c.hist_n = c.hist_n + 1u;
  let conv = resid <= pr.tol;
  if (pr.mode == 0u) {
    if (conv) { c.done = 1u; }
  } else {
    // UTM replay: settle (≤ settle iterations, stops early at convergence), then average the probe read-out over avg
    // iterations, record (travel, force), next increment — the protocol of weavesim.calibrate.replay_force / replay
    let probe = pr.probe;
    let ax = vec3<f32>(pr.axis_x, pr.axis_y, pr.axis_z);
    c.it_phase = c.it_phase + 1u;
    if (c.phase == 0u) {
      if (conv || c.it_phase >= pr.settle) {
        c.phase = 1u; c.it_phase = 0u; c.acc_n = 0u;
        c.adx = vec2<u32>(0u); c.ady = vec2<u32>(0u); c.adz = vec2<u32>(0u); c.arr = vec2<u32>(0u);
      }
    } else if (c.phase == 1u) {
      let o = xoff(b, probe, rollback == 1u);
      let p0 = I_PROBE0 + b * 6u;
      c.adx = i64_add(c.adx, i64_sub(ld64(o), ld64(p0)));
      c.ady = i64_add(c.ady, i64_sub(ld64(o + 2u), ld64(p0 + 2u)));
      c.adz = i64_add(c.adz, i64_sub(ld64(o + 4u), ld64(p0 + 4u)));
      c.arr = i64_add(c.arr, f_i64(dot(ld3(F_REACT + (b * N + probe) * 3u), ax), pr.fsc));
      c.acc_n = c.acc_n + 1u;
      if (c.it_phase >= pr.avg) {
        let oo = F_OUT + (b * NOUT + c.inc) * 4u;
        let nn = f32(c.acc_n);
        let dav = dot(vec3<f32>(i64_f(c.adx, pr.pinv), i64_f(c.ady, pr.pinv), i64_f(c.adz, pr.pinv)), ax) / nn;
        let rav = i64_f(c.arr, pr.finv) / nn;
        fst[oo] = dav; fst[oo + 1u] = rav; fst[oo + 2u] = c.cur_f; fst[oo + 3u] = f32(c.iter);
        c.inc = c.inc + 1u; c.it_phase = 0u; c.phase = 0u;
        if (c.inc >= pr.n_inc) { c.phase = 2u; c.done = 1u; }
        else {
          let v = pr.sched[min(c.inc, 63u)];
          c.cur_f = v;
          if (pr.mode == 1u) { c.pfx = ax.x * v; c.pfy = ax.y * v; c.pfz = ax.z * v; }
          else {
            // displacement control: put the probe node at P0 + v·axis (exactly), at rest
            let ox = xoff(b, probe, false);
            st64(ox, i64_add(ld64(p0), f_i64(ax.x * v, pr.psc)));
            st64(ox + 2u, i64_add(ld64(p0 + 2u), f_i64(ax.y * v, pr.psc)));
            st64(ox + 4u, i64_add(ld64(p0 + 4u), f_i64(ax.z * v, pr.psc)));
            st3(F_V + (b * N + probe) * 3u, vec3<f32>(0.0));
            if (rollback == 1u) { let op = xoff(b, probe, true); for (var k = 0u; k < 6u; k++) { ist[op + k] = ist[ox + k]; } }
          }
        }
      }
    }
  }
  ctl[b] = c;
}
`.replace(/NWG_P/g, `${L.NWG_P}u`).replace(/NWG_F/g, `${L.NWG_F}u`).replace(/NWG_N/g, `${L.NWG_N}u`).replace(/NWG_SI/g, `${L.NWG_S}u`);
}
