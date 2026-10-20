// WGSL source of the batched dynamic-relaxation solver (one dispatch per kernel, B instances along the y grid axis).
// The rod is the discrete Cosserat (Simo–Reissner) rod of rod_cosserat.js — node positions, one full rotation per
// segment (quaternion, not slaved to the chord) and a 3-axis body angular velocity per segment. One iteration:
//   k_prep   segments: frame R, strain σ = Rᵀe/l0 − e3 → force n = R diag(GA, GA, EA) σ on the end nodes and the
//            torque e × n on the frame; nodes: tangent and mean frame (contact evaluation, frame-local loads)
//   k_force  joints: Ψ = log(R_aᵀR_b), m = B (Ψ/l̄ − Ω0), torques ±R_a J_l⁻ᵀ(Ψ) m; clamp joints (rotation anchors)
//            against the frozen frames; contacts (forces, lever and friction torques on the segment frames, righting
//            stiffness)
//   k_int    nodes: loads, reactions, anchors, v += F/m, x += v; segments: inertia per body axis (static + the contact
//            share of this iteration), ω += Rᵀ T / I, roll-rate limit, q ← q ⊗ exp(ω)
//   k_ctrl   reductions, Barnes kinetic damping (roll back to the kinetic-energy peak), residual, history, UTM protocol
//
// Precision scheme (WebGPU has no f64):
//  · node positions are 64-bit fixed-point integers (vec2<u32>, 2^-PB length units): the velocity is added exactly,
//    every edge / stencil vector is an exact integer difference converted to f32 once — so the f32 error is relative
//    to the local size (a segment, a crossing), never to the coordinate (a 3 m model in f32 alone would lose 0.2 µm
//    per coordinate and several newtons of axial force per lath);
//  · quaternions are 64-bit fixed point too (2^-60), updated in delta form (a frame that does not move does not change
//    by a bit), and every relative rotation — joint (q_a, q_b), clamp (frozen frame, q) — is formed from the exact
//    integer difference of the two quaternions: Ψ carries f32 precision relative to the (small) relative rotation,
//    not to the unit quaternion (f32 quaternions alone: ~1e-7 rad of bending noise);
//  · the segment strain comes from the exact mismatch μ = e − l0 t, t = |q|² R(q̂) e3 formed from the fixed-point
//    quaternion by exact 64 × 64-bit integer products, and the exact norm defect η = |q|² − 1:
//      l0 σ = R(q̂)ᵀ μ + l0 η e3 — f32 precision relative to the strain itself (an f32 frame and edge alone would leave
//    ~1e-7·EA of axial-force and ~1e-7·κGA of shear-force noise per segment: a residual floor on smooth beams);
//  · torques are accumulated in the body axes of the segment they act on: a joint adds J to segment a exactly and
//    −R(Δq)ᵀ J to segment b with the exact relative rotation Δq (clamp joints likewise), so the bending moments that
//    cancel at equilibrium carry no f32 frame-rounding noise; contact torques are rotated by the segment frame once;
//  · contact parameters u are 64-bit fixed point (segment index | 32-bit fraction), slips add exactly; the Hermite
//    point is evaluated at the 24-bit part of t plus a first-order term for the remaining 8 bits;
//  · forces, torques and the contact righting stiffness are accumulated as 64-bit fixed-point integers with atomics:
//    sums are exact and independent of thread order, so a run is bit-for-bit reproducible on a given GPU;
//  · reductions (kinetic energy, residual, energies) run in a fixed tree order.
/** 64-bit integer arithmetic on vec2<u32> (two's complement, x = low word, y = high word): add, subtract, negate,
 *  f32 conversions, the exact (a · b) >> 60 product — shared with tests/gpu_int_check.html */
export const I64_WGSL = /* wgsl */`
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
fn i64_shl1(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x << 1u, (a.y << 1u) | (a.x >> 31u)); }
/** 32 × 32 → 64-bit unsigned product from 16-bit halves (WGSL has no wide multiply) */
fn mul32(a: u32, b: u32) -> vec2<u32> {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let mid = lh + hl; let cm = select(0u, 0x10000u, mid < lh);      // carry of the middle sum (bit 48 of the product)
  let lo = ll + (mid << 16u); let cl = select(0u, 1u, lo < ll);
  return vec2<u32>(lo, hh + (mid >> 16u) + cm + cl);
}
/** (a · b) >> 60 of two signed 64-bit integers, truncated toward zero (|a · b| < 2^123): the product of two 2^-60
 *  fixed-point numbers in 2^-60 units, or of a 2^-PB and a 2^-60 number in 2^-PB units — exact to one unit */
fn i64_mul60(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let na = i64_isneg(a); let nb = i64_isneg(b);
  let A = select(a, i64_neg(a), na); let B = select(b, i64_neg(b), nb);
  let p00 = mul32(A.x, B.x); let p01 = mul32(A.x, B.y); let p10 = mul32(A.y, B.x); let p11 = mul32(A.y, B.y);
  // |a·b| = p11·2^64 + (p01 + p10)·2^32 + p00 as 32-bit words w0 … w3 (w0 is below the result)
  var w1 = p00.y + p01.x; var c = select(0u, 1u, w1 < p01.x);
  w1 = w1 + p10.x; c = c + select(0u, 1u, w1 < p10.x);
  var w2 = p11.x + c; var c2 = select(0u, 1u, w2 < c);
  w2 = w2 + p01.y; c2 = c2 + select(0u, 1u, w2 < p01.y);
  w2 = w2 + p10.y; c2 = c2 + select(0u, 1u, w2 < p10.y);
  let w3 = p11.y + c2;
  let r = vec2<u32>((w1 >> 28u) | (w2 << 4u), (w2 >> 28u) | (w3 << 4u));
  return select(r, i64_neg(r), na != nb);
}
`;

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
const HN: u32 = ${u(L.HN)}; const HS: u32 = ${u(L.HS)};
const NWG: u32 = ${u(L.NWG)};
const NSLOT: u32 = 3u;
// topo (u32)
const T_SEG0 = ${u(L.t.seg0)}; const T_SEG1 = ${u(L.t.seg1)}; const T_J0 = ${u(L.t.j0)}; const T_J1 = ${u(L.t.j1)}; const T_CS = ${u(L.t.cs)};
const T_NSA = ${u(L.t.nsa)}; const T_NSB = ${u(L.t.nsb)}; const T_KOFF = ${u(L.t.koff)}; const T_KNSEG = ${u(L.t.knseg)}; const T_KEN = ${u(L.t.ken)};
const T_L0I = ${u(L.t.l0i)};            // rest lengths as 64-bit integers (2^-PB)
// geo (f32): shared, then per instance
const G_L0 = ${u(L.g.l0)}; const G_JLEN = ${u(L.g.jlen)}; const G_CLEN = ${u(L.g.clen)}; const G_WK = ${u(L.g.wk)}; const G_HK = ${u(L.g.hk)};
const G_KW = ${u(L.g.kw)}; const G_KH = ${u(L.g.kh)};
const G_STIFF = ${u(L.g.stiff)}; const G_OM = ${u(L.g.om)}; const G_MASS = ${u(L.g.mass)}; const G_KROT = ${u(L.g.krot)};
const G_LC = ${u(L.g.lc)}; const G_LL = ${u(L.g.ll)}; const G_CB = ${u(L.g.cb)};
const G_FC90 = ${u(L.g.fc90)}; const G_E90 = ${u(L.g.e90)}; const G_HS = ${u(L.g.hs)};
// ist (u32): integer state per instance
const I_X = ${u(L.i.x)}; const I_XP = ${u(L.i.xp)}; const I_X0 = ${u(L.i.x0)}; const I_U = ${u(L.i.u)};
const I_SLIP = ${u(L.i.slip)}; const I_FIX = ${u(L.i.fix)}; const I_LOCK = ${u(L.i.lock)}; const I_KOUT = ${u(L.i.kout)};
const I_PROBE0 = ${u(L.i.probe0)}; const I_Q = ${u(L.i.q)}; const I_QP = ${u(L.i.qp)}; const I_CQ = ${u(L.i.cq)};
// quaternions: value · 2^-60 (QINV) and the f32 → fixed-point scale 2^(60−32) (QSC)
const QINV: f32 = 8.673617379884035e-19; const QSC: f32 = 268435456.0;
// fst (f32): float state per instance
const F_V = ${u(L.f.v)}; const F_W = ${u(L.f.w)}; const F_NPREV = ${u(L.f.nprev)};
const F_LAM = ${u(L.f.lam)}; const F_TH0 = ${u(L.f.th0)}; const F_RSLIP = ${u(L.f.rslip)}; const F_DENT = ${u(L.f.dent)}; const F_DENTE = ${u(L.f.dente)};
const F_CLIMB = ${u(L.f.climb)}; const F_TANPSI = ${u(L.f.tanpsi)}; const F_FN = ${u(L.f.fn)}; const F_FT = ${u(L.f.ft)}; const F_GAP = ${u(L.f.gap)};
const F_PHI = ${u(L.f.phi)}; const F_THETA = ${u(L.f.theta)}; const F_ANOM = ${u(L.f.anom)}; const F_PRES = ${u(L.f.pres)}; const F_UTIL = ${u(L.f.util)};
const F_MROT = ${u(L.f.mrot)}; const F_MID = ${u(L.f.mid)}; const F_NAX = ${u(L.f.nax)}; const F_VSH = ${u(L.f.vsh)}; const F_MJ = ${u(L.f.mj)}; const F_CM = ${u(L.f.cm)};
const F_REACT = ${u(L.f.react)}; const F_LSEG = ${u(L.f.lseg)}; const F_R = ${u(L.f.r)}; const F_TN = ${u(L.f.tn)};
const F_QN = ${u(L.f.qn)};
const F_HIST = ${u(L.f.hist)}; const F_OUT = ${u(L.f.out)}; const NOUT = ${u(L.NOUT)};
const F_PENACC = ${u(L.f.penacc)};
// acc (atomic u32): fixed-point accumulators per instance — node forces (world), segment torques (body axes of the
// segment), contact righting stiffness
const A_F = ${u(L.a.f)}; const A_T = ${u(L.a.t)}; const A_K = ${u(L.a.k)};

struct IP {                         // per-instance parameters (host-written)
  mass_scale: f32, damping: f32, kin_damp: u32, warmup: u32,
  tol: f32, ramp_iter: u32, contacts: u32, aug: u32,
  al_every: u32, mu: f32, strength: f32, stick: f32,
  torsion_factor: f32, epsilon: f32, slip_rate: f32, rot_fric: u32,
  rot_factor: f32, seat: u32, seat_gap: u32, dent_rate: f32,
  dent_after: u32, edge_thr: f32, jlen_min: f32, l_ref: f32,   // l_ref = min(l̄ ∪ l0): torque residual scale
  psc: f32, pinv: f32, fsc: f32, finv: f32,             // positions: value·2^-PB, sc = 2^(PB-32); forces 2^-FB
  tsc: f32, tinv: f32, ksc: f32, kinv: f32,             // torques, righting stiffness
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
  it_phase: u32, acc_n: u32, e_shear: f32, res_t: f32,   // shear energy, torque part of the residual
  pfx: f32, pfy: f32, pfz: f32, cur_f: f32,
  // UTM read-out sums over the averaging window, exact 64-bit fixed point: probe travel Σ(X − P0) per axis
  // (2^-PB) and the reaction along the probe axis (2^-FB) — one rounding when the average is formed
  adx: vec2<u32>, ady: vec2<u32>, adz: vec2<u32>, arr: vec2<u32>,
  res_f: f32, pad_c: u32, pad_d: u32, pad_e: u32,     // force part of the residual
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
${I64_WGSL}
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
/** torque on the frame of segment j, in its body axes */
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
/** quaternion in fixed point (8 words at o) as f32 */
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
fn frot(R: Fr, v: vec3<f32>) -> vec3<f32> { return v.x * R.x + v.y * R.y + v.z * R.z; }        // R v
fn frotT(R: Fr, v: vec3<f32>) -> vec3<f32> { return vec3<f32>(dot(R.x, v), dot(R.y, v), dot(R.z, v)); }   // Rᵀ v
/** q_a* ⊗ q_b − 1 of two fixed-point quaternions (offsets oa, ob), formed from the exact integer difference q_b ∓ q_a
 *  (the sign that keeps q_a·q_b ≥ 0, as relRotation): the vector part carries f32 precision relative to the relative
 *  rotation; the scalar part of the product is 1 + .x */
fn qrel(oa: u32, ob: u32) -> vec4<f32> {
  let qa = ldqi(oa); let qb = ldqi(ob);
  let flip = dot(qa, qb) < 0.0;
  var dq = vec4<f32>(0.0);
  for (var k = 0u; k < 4u; k++) {
    let va = ld64(oa + 2u * k); let vb = ld64(ob + 2u * k);
    dq[k] = i64_f(select(i64_sub(vb, va), i64_add(vb, va), flip), QINV);
  }
  return qmul(qconj(qa), dq) * select(1.0, -1.0, flip);
}
/** rotation vector Ψ = log(q_a* ⊗ q_b) (material axes) and θ = |Ψ| from r = q_a* ⊗ q_b − 1 (qrel; relRotation) */
fn relrot(r: vec4<f32>) -> vec4<f32> {
  let w = 1.0 + r.x; let v = r.yzw;
  let s = len_acc(v);
  let th = 2.0 * atan2_acc(s, w);
  let f = select(2.0 / max(w, 1e-12), th / s, s > 1e-12);
  return vec4<f32>(f * v, th);
}
/** R(Δq)ᵀ x for Δq = 1 + r (qrel): the body axes of q_b from those of q_a, x − 2w (v × x) + 2 v × (v × x) */
fn rotT(r: vec4<f32>, x: vec3<f32>) -> vec3<f32> { let v = r.yzw; let vx = cross(v, x); return x - 2.0 * (1.0 + r.x) * vx + 2.0 * cross(v, vx); }
/** J_l⁻ᵀ(Ψ) m = m + ½ Ψ × m + β Ψ × (Ψ × m),  β = 1/θ² − cot(θ/2)/(2θ): the series below θ = 1 (the closed form
 *  cancels in f32 for small θ), the half-angle closed form above */
fn jlt(psi: vec3<f32>, th: f32, m: vec3<f32>) -> vec3<f32> {
  var beta: f32;
  if (th < 1.0) {
    let t2 = th * th;
    beta = 0.08333333333333333 + t2 * (1.388888888888889e-3 + t2 * (3.306878306878307e-5 + t2 * (8.267195767195767e-7 + t2 * 2.087675698786810e-8)));
  } else {
    let sc = sincos_acc(0.5 * th);
    beta = 1.0 / (th * th) - sc.y / (2.0 * th * sc.x);
  }
  let c = cross(psi, m);
  return m + 0.5 * c + beta * cross(psi, c);
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
// segments: (rollback), chord, frame, stretch / shear → end-node forces and the frame torque e × n;
// nodes: (rollback), tangent, mean frame
@compute @workgroup_size(64)
fn k_prep(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  let rb = ctl[b].rollback == 1u;
  let pinv = ip[b].pinv;
  var s = vec4<f32>(0.0); var mx = vec4<f32>(0.0);
  let t = gid.x;
  if (t < M) {
    let j = t;
    let a = topo[T_SEG0 + j]; let c = topo[T_SEG1 + j];
    let oq = qoff(b, j, false);
    if (rb) { let op = qoff(b, j, true); for (var k = 0u; k < 8u; k++) { ist[oq + k] = ist[op + k]; } st3(F_W + (b * M + j) * 3u, vec3<f32>(0.0)); }
    let oqs = select(oq, qoff(b, j, true), rb);                     // this iteration's frame (restored on a roll-back)
    let e = pdiff(b, c, a, rb, pinv);
    let Ls = len_acc(e);
    fst[F_LSEG + b * M + j] = Ls;
    let R = qframe(ldqi(oqs));
    let orr = F_R + (b * M + j) * 9u;
    st3(orr, R.x); st3(orr + 3u, R.y); st3(orr + 6u, R.z);
    // exact strain mismatch (header): t = |q|² R(q̂) e3 and η = |q|² − 1 from 64-bit products of the quaternion
    // (2^-60), μ = e − l0 t in 2^-PB, then l0 σ = R(q̂)ᵀ μ + l0 η e3
    let qw = ld64(oqs); let qx = ld64(oqs + 2u); let qy = ld64(oqs + 4u); let qz = ld64(oqs + 6u);
    let ww = i64_mul60(qw, qw); let xx = i64_mul60(qx, qx); let yy = i64_mul60(qy, qy); let zz = i64_mul60(qz, qz);
    let tx = i64_shl1(i64_add(i64_mul60(qx, qz), i64_mul60(qw, qy)));
    let ty = i64_shl1(i64_sub(i64_mul60(qy, qz), i64_mul60(qw, qx)));
    let tz = i64_sub(i64_add(ww, zz), i64_add(xx, yy));
    let eta = i64_f(i64_sub(i64_add(i64_add(ww, xx), i64_add(yy, zz)), vec2<u32>(0u, 0x10000000u)), QINV);
    let L0 = vec2<u32>(topo[T_L0I + 2u * j], topo[T_L0I + 2u * j + 1u]);
    let oc = xoff(b, c, rb); let oa = xoff(b, a, rb);
    let mu = vec3<f32>(i64_f(i64_sub(i64_sub(ld64(oc), ld64(oa)), i64_mul60(L0, tx)), pinv),
                       i64_f(i64_sub(i64_sub(ld64(oc + 2u), ld64(oa + 2u)), i64_mul60(L0, ty)), pinv),
                       i64_f(i64_sub(i64_sub(ld64(oc + 4u), ld64(oa + 4u)), i64_mul60(L0, tz)), pinv));
    let l0 = geo[G_L0 + j];
    let ls = frotT(R, mu) + vec3<f32>(0.0, 0.0, l0 * eta);                 // l0 σ (body axes)
    let sg = ls / l0;
    let so = G_STIFF + (b * M + j) * 5u;
    let GA = geo[so + 4u];
    let nb = vec3<f32>(GA * sg.x, GA * sg.y, geo[so] * sg.z);              // (V1, V2, N) in the body axes
    let n = frot(R, nb);
    addF(b, a, n, ip[b].fsc); addF(b, c, -n, ip[b].fsc);
    addT(b, j, cross(vec3<f32>(ls.x, ls.y, l0 + ls.z), nb), ip[b].tsc);   // e × n = R (Rᵀe × Rᵀn)
    fst[F_NAX + b * M + j] = nb.z; fst[F_VSH + (b * M + j) * 2u] = nb.x; fst[F_VSH + (b * M + j) * 2u + 1u] = nb.y;
    s.x = 0.5 * l0 * nb.z * sg.z; s.y = 0.5 * l0 * (nb.x * sg.x + nb.y * sg.y); mx.x = abs(nb.z);
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
  reduce_store(b, 0u, wid.x, lid.x, s, mx);
}

// ============================================================================================== 2 · forces
// joints (bending + twist), clamp joints, contacts
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
    // ---------------------------------------------------------------- joint: bending and twist, Ψ = log(R_aᵀR_b)
    let j = t;
    let s0 = topo[T_J0 + j]; let s1 = topo[T_J1 + j];
    let r = qrel(qoff(b, s0, false), qoff(b, s1, false));
    let rr = relrot(r);
    let psi = rr.xyz;
    let lbar = geo[G_JLEN + j];
    let so = G_STIFF + (b * M + s0) * 5u;
    let d = psi / lbar - gld3(G_OM + (b * P + j) * 3u);
    let m = vec3<f32>(geo[so + 1u], geo[so + 2u], geo[so + 3u]) * d;      // (EIx, EIy, GJ)(κ − κ0)
    let J = jlt(psi, rr.w, m);                                            // τ = R_a J: J in a's axes, R(Δq)ᵀ J in b's
    addT(b, s0, J, tsc); addT(b, s1, -rotT(r, J), tsc);
    st3(F_MJ + (b * P + j) * 3u, m);
    s.x = 0.5 * lbar * (m.x * d.x + m.y * d.y);
    s.y = 0.5 * lbar * m.z * d.z;
    mx.x = max(max(abs(m.x), abs(m.y)), abs(m.z));
  } else if (t < P + C) {
    // ---------------------------------------------------------------- clamp joint: frozen frame c → segment frame
    let c = t - P;
    let j = topo[T_CS + c];
    let r = qrel(I_CQ + (b * C + c) * 8u, qoff(b, j, false));
    let rr = relrot(r);
    let psi = rr.xyz;
    let lc = geo[G_CLEN + c];
    let d = psi / lc;
    let m = gld3(G_CB + (b * C + c) * 3u) * d;                            // per-axis stiffness, 0 on a free axis
    addT(b, j, -rotT(r, jlt(psi, rr.w, m)), tsc);                         // −R_c J in the segment's axes
    fst[F_CM + (b * C + c) * 2u] = m.x; fst[F_CM + (b * C + c) * 2u + 1u] = m.y;
    s.x = 0.5 * lc * (m.x * d.x + m.y * d.y);
    s.y = 0.5 * lc * m.z * d.z;
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
    // distribute to the segment nodes (1 − t, t); lever-arm torques on the segment frames (in full, body axes)
    addF(b, A.a, (1.0 - A.tp) * FA, fsc); addF(b, A.c, A.tp * FA, fsc);
    addF(b, B.a, (1.0 - B.tp) * FB, fsc); addF(b, B.c, B.tp * FB, fsc);
    let tf = pr.torsion_factor;
    let RA = ldfr(F_R + (b * M + A.seg) * 9u); let RB = ldfr(F_R + (b * M + B.seg) * 9u);
    var TA = tf * cross(supA.xyz, FA); var TB = tf * cross(supB.xyz, FB);
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
      TA = TA + Mr * n; TB = TB - Mr * n;
      eRot = 0.5 * kr * dth * dth;
    }
    addT(b, A.seg, frotT(RA, TA), tsc); addT(b, B.seg, frotT(RB, TB), tsc);
    // packed outputs: bit0 slip, bit1 stuck, bits 2-3 contact type, bits 4-5 face side
    ist[I_KOUT + kk] = select(0u, 1u, slip) | select(0u, 2u, !(slip || free)) | (nEdge << 2u) | (faceSide << 4u);
    // contact righting stiffness for the frame inertia (post-update segments, as addTwistStiffness)
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

// ============================================================================================== 3 · integrate
// nodes: loads, reactions, anchors, v += F/m, x += v; segments: ω += Rᵀ T / I (body axes), q ← q ⊗ exp(ω)
@compute @workgroup_size(64)
fn k_int(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  var s = vec4<f32>(0.0); var mx = vec4<f32>(0.0);
  let t = gid.x;
  let pr = ip[b];
  if (t < N) {
    let i = t;
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
    mx.y = len_acc(F);                                              // force residual: every node, fixed DOFs zeroed
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
  } else if (t < N + M) {
    let j = t - N;
    let w = A_T + (b * M + j) * 6u;
    let tm = vec3<f32>(acc_take(w, pr.tinv), acc_take(w + 2u, pr.tinv), acc_take(w + 4u, pr.tinv));   // Rᵀ T (body axes)
    let ktw = acc_take(A_K + (b * M + j) * 2u, pr.kinv);
    mx.z = len_acc(tm);
    // inertia per body axis: joints, clamps and shear lever (static) + the contact righting share of this iteration
    let I3 = pr.mass_scale * max(gld3(G_KROT + (b * M + j) * 3u) + vec3<f32>(ktw), vec3<f32>(1e-12));
    let ow = F_W + (b * M + j) * 3u;
    var W = ld3(ow) + tm / I3;
    W.z = clamp(W.z, -TWIST_RATE_MAX, TWIST_RATE_MAX);
    W = W * pr.damping;
    st3(ow, W);
    // q ← q ⊗ exp(W) (body frame), applied as an increment dQ = q ⊗ (exp(W) − 1) and renormalised in delta form
    let oq = qoff(b, j, false); let op = qoff(b, j, true);
    for (var k = 0u; k < 8u; k++) { ist[op + k] = ist[oq + k]; }
    let q0 = ldqi(oq);
    let a = len_acc(W);
    let s4 = sin_acc(0.25 * a);
    let dQ = qmul(q0, vec4<f32>(-2.0 * s4 * s4, W * select(0.5, sin_acc(0.5 * a) / a, a > 1e-12)));
    let x = 2.0 * dot(q0, dQ) + dot(dQ, dQ);
    let rt = sqrt_acc(1.0 + x);
    let dtot = dQ - (x / (rt * (1.0 + rt))) * (q0 + dQ);
    for (var k = 0u; k < 4u; k++) { st64(oq + 2u * k, i64_add(ld64(oq + 2u * k), f_i64(dtot[k], QSC))); }
    s.y = dot(I3, W * W);
  }
  reduce_store(b, 2u, wid.x, lid.x, s, mx);
}

// ============================================================================================== 4 · control (one workgroup per instance)
var<workgroup> cs: array<vec4<f32>, 64>;
var<workgroup> cm: array<vec4<f32>, 64>;
var<workgroup> tot: array<vec4<f32>, 6>;
@compute @workgroup_size(64)
fn k_ctrl(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let b = wid.y;
  if (lid.x == 0u) { wg_act = inst_live(b); }
  if (workgroupUniformLoad(&wg_act) == 0u) { return; }
  for (var kern = 0u; kern < NSLOT; kern++) {
    var ss = vec4<f32>(0.0); var mm = vec4<f32>(0.0);
    if (kern == 1u) { mm.y = -3e38; }                               // max Fn may be negative (locked crossings)
    let nwg = select(select(NWG_I, NWG_F, kern == 1u), NWG_P, kern == 0u);
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
  // k_prep: (stretch, shear | max |N|); k_force: (bending, torsion, contact | max |M|, max Fn); k_int: (Σ m v², Σ I ω² |
  // max |load|, max |F|, max |Rᵀ T|)
  let sP = tot[0]; let mP = tot[1]; let sF = tot[2]; let mF = tot[3]; let sI = tot[4]; let mI = tot[5];
  var ke = 0.5 * (sI.x + sI.y);
  var rollback = 0u;
  if (pr.kin_damp == 1u && ke < c.ke_prev) { rollback = 1u; ke = 0.0; }
  c.rollback = rollback;
  c.ke_prev = ke; c.ke = ke;
  c.iter = c.iter + 1u;
  var fref = max(mI.x, mP.x);
  if (P > 0u) { fref = max(fref, mF.x / max(pr.jlen_min, 1e-12)); }
  if (pr.contacts == 1u && K > 0u) { fref = max(fref, mF.y); }
  fref = max(fref, 1e-12);
  let rf = mI.y / fref; let rt = mI.z / (fref * pr.l_ref);
  var resid = max(rf, rt);
  if (c.iter <= pr.warmup) { resid = 3e38; }
  c.resid = resid; c.fref = fref; c.res_f = rf; c.res_t = rt;
  c.e_stretch = sP.x; c.e_shear = sP.y; c.e_bend = sF.x; c.e_tors = sF.y; c.e_cont = sF.z;
  c.al_update = select(0u, 1u, pr.aug == 1u && pr.al_every > 0u && c.iter > pr.ramp_iter && ((c.iter - pr.ramp_iter) % pr.al_every) == 0u);
  let ho = F_HIST + (b * HN + (c.iter - 1u) % HN) * HS;
  fst[ho] = ke; fst[ho + 1u] = sP.x + sP.y + sF.x + sF.y + sF.z; fst[ho + 2u] = sF.x; fst[ho + 3u] = sF.y;
  fst[ho + 4u] = sP.x; fst[ho + 5u] = sP.y; fst[ho + 6u] = sF.z; fst[ho + 7u] = resid; fst[ho + 8u] = f32(c.iter);
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
`.replace(/NWG_P/g, `${L.NWG_P}u`).replace(/NWG_F/g, `${L.NWG_F}u`).replace(/NWG_I/g, `${L.NWG_I}u`);
}
