// Discrete Cosserat (Simo–Reissner) rod: the shear-deformable rod of theory §3 — the rod model of the solvers.
//
// Unknowns: node positions x_i and one full rotation R_j (quaternion Q_j) per segment — the segment frame is no longer
// slaved to its chord, so the section can shear (frame Z ≠ chord), roll and bend in one consistent energy:
//
//   segment j (nodes a → b, edge e = x_b − x_a, rest length l0):
//     σ = R_jᵀ e / l0 − e3                       (σ1, σ2: shear along the width X and the normal Y; σ3: stretch)
//     E_s = ½ l0 (κGA σ1² + κGA σ2² + EA σ3²),    n = R_j S σ  (internal force)
//     F_a += n,  F_b −= n,  τ_j += e × n
//   joint i between segments a → b (Voronoi length l̄):
//     Ψ = log(R_aᵀ R_b)  (rotation vector in the material axes, the same in a and b),  κ = Ψ / l̄
//     E_b = ½ l̄ (κ − κ0)ᵀ B (κ − κ0),  B = diag(EIx, EIy, GJ),  m = B (κ − κ0)
//     τ_a += R_a J_l⁻ᵀ(Ψ) m,  τ_b −= R_a J_l⁻ᵀ(Ψ) m,   J_l⁻ᵀ m = m + ½ Ψ × m + β Ψ × (Ψ × m)
//
// Forces and torques are the exact gradient of E_s + E_b (finite-difference checked in web/tests/rod_check.mjs).
// The J_l⁻ᵀ term carries what a Kirchhoff rod with frames slaved to the chords leaves out: for EIx ≠ EIy, m is not
// parallel to Ψ, and Ψ × m is the roll moment of anisotropic bending; the twist–bending coupling comes from the same term
// and from the frames being free. Lateral–torsional buckling, shear deflection (12EI/κGAL²) and the Kirchhoff limit (GA → ∞) all follow.
// Torques T are world vectors per segment; the solver integrates a 3-axis body angular velocity per segment.
import { toFrame } from './quat.js';

export const SHEAR_KAPPA = 5 / 6;                     // shear coefficient of a rectangle

/** rotation vector Ψ = log(q_a* ⊗ q_b) (material axes) of two unit quaternions Q[4a..], Q[4b..] → out[3]; returns θ */
export function relRotation(Q, a, b, out) {
  const aw = Q[4 * a], ax = -Q[4 * a + 1], ay = -Q[4 * a + 2], az = -Q[4 * a + 3];
  const bw = Q[4 * b], bx = Q[4 * b + 1], by = Q[4 * b + 2], bz = Q[4 * b + 3];
  let w = aw * bw - ax * bx - ay * by - az * bz;
  let x = aw * bx + ax * bw + ay * bz - az * by;
  let y = aw * by - ax * bz + ay * bw + az * bx;
  let z = aw * bz + ax * by - ay * bx + az * bw;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.sqrt(x * x + y * y + z * z), th = 2 * Math.atan2(s, w);
  const f = s > 1e-12 ? th / s : 2 / Math.max(w, 1e-12);
  out[0] = f * x; out[1] = f * y; out[2] = f * z;
  return th;
}

export class CosseratRod {
  /** stiff: per-segment {EA, EIx, EIy, GJ, GA}; activation: Float64Array(3p) (flat, edge, twist); rest: Omega0 to keep */
  constructor(st, stiff, prestress = 'flat', activation = null, rest = null) {
    const m = st.m, p = st.p;
    this.kind = 'cosserat';
    this.segEA = Float64Array.from(stiff.EA); this.segEIx = Float64Array.from(stiff.EIx); this.segEIy = Float64Array.from(stiff.EIy); this.segGJ = Float64Array.from(stiff.GJ);
    this.segGA = stiff.GA ? Float64Array.from(stiff.GA) : Float64Array.from(stiff.EA, v => SHEAR_KAPPA * v / 16);
    this.EA = this.segEA; this.GA = this.segGA;
    this.EIx = new Float64Array(p); this.EIy = new Float64Array(p); this.GJ = new Float64Array(p);
    for (let j = 0; j < p; j++) { const s0 = st.jSeg0[j]; this.EIx[j] = this.segEIx[s0]; this.EIy[j] = this.segEIy[s0]; this.GJ[j] = this.segGJ[s0]; }
    if (!st.Q0 || st.Q0.length !== st.Q.length) st.Q0 = st.Q.slice();
    const act = new Float64Array(3 * p);
    if (prestress !== 'asbuilt' && activation) for (let i = 0; i < 3 * p; i++) act[i] = Math.min(Math.max(activation[i], 0), 1);
    this.activation = act;
    if (rest) st.Omega0 = Float64Array.from(rest);
    else if (p && act.some(v => v < 1.0)) {
      const om = CosseratRod.asBuilt(st, st.Q0);
      st.Omega0 = new Float64Array(3 * p);
      for (let i = 0; i < 3 * p; i++) st.Omega0[i] = (1.0 - act[i]) * om[i];
    } else st.Omega0 = new Float64Array(3 * p);
    this.N = new Float64Array(m); this.V = new Float64Array(2 * m); this.M = new Float64Array(3 * p);
    this.energy = 0; this.energyParts = { bending: 0, torsion: 0, stretch: 0, shear: 0 };
    this.cNode = []; this.cSeg = []; this.cLen = []; this.cB = []; this.cQ = new Float64Array(0); this.cM = new Float64Array(0);
    this._R = new Float64Array(9 * m);
  }

  /** clamps (rX rY rZ anchors, Anchors.rotClamps): a boundary joint between the segment's frame and its frame at this
   *  rebuild, frozen, with half a segment of Voronoi length (second-order accurate in the segment length; freezing the
   *  segment's rotation instead loses half a segment of bending at the support). Per axis: rX → EIx (flatwise), rY → EIy (edgewise), rZ → GJ (twist). */
  setClamps(st, rotClamps = []) {
    const nc = rotClamps.length;
    this.cNode = rotClamps.map(c => c[0]); this.cSeg = rotClamps.map(c => c[1]);
    this.cLen = this.cSeg.map(j => 0.5 * st.l0[j]);
    this.cB = rotClamps.map(([, j, mk]) => [mk[0] ? this.segEIx[j] : 0, mk[1] ? this.segEIy[j] : 0, mk[2] ? this.segGJ[j] : 0]);
    this.cQ = new Float64Array(4 * nc);
    this.cSeg.forEach((j, c) => this.cQ.set(st.Q.subarray(4 * j, 4 * j + 4), 4 * c));
    this.cM = new Float64Array(2 * nc);
  }

  /** rest curvature/twist Ψ/l̄ per joint of the frames Q (default: the reference frames Q0) */
  static asBuilt(st, Q = st.Q0 || st.Q) {
    const out = new Float64Array(3 * st.p), psi = new Float64Array(3);
    for (let j = 0; j < st.p; j++) {
      relRotation(Q, st.jSeg0[j], st.jSeg1[j], psi);
      const lb = st.jLen[j];
      out[3 * j] = psi[0] / lb; out[3 * j + 1] = psi[1] / lb; out[3 * j + 2] = psi[2] / lb;
    }
    return out;
  }

  /** forces on the nodes (F, 3n) and world torques on the segment frames (T, 3m) */
  apply(st, F, T) {
    const X = st.X, Q = st.Q, m = st.m, p = st.p, R = this._R;
    for (let j = 0; j < m; j++) toFrame(Q, 4 * j, R.subarray(9 * j, 9 * j + 9));
    let eStretch = 0, eShear = 0, eBend = 0, eTwist = 0;
    for (let j = 0; j < m; j++) {
      const a = st.segI0[j], b = st.segI1[j], o = 9 * j, l0 = st.l0[j];
      const ex = X[3 * b] - X[3 * a], ey = X[3 * b + 1] - X[3 * a + 1], ez = X[3 * b + 2] - X[3 * a + 2];
      const s1 = (R[o] * ex + R[o + 1] * ey + R[o + 2] * ez) / l0;
      const s2 = (R[o + 3] * ex + R[o + 4] * ey + R[o + 5] * ez) / l0;
      const s3 = (R[o + 6] * ex + R[o + 7] * ey + R[o + 8] * ez) / l0 - 1.0;
      const n1 = this.GA[j] * s1, n2 = this.GA[j] * s2, n3 = this.EA[j] * s3;
      const nx = n1 * R[o] + n2 * R[o + 3] + n3 * R[o + 6];
      const ny = n1 * R[o + 1] + n2 * R[o + 4] + n3 * R[o + 7];
      const nz = n1 * R[o + 2] + n2 * R[o + 5] + n3 * R[o + 8];
      F[3 * a] += nx; F[3 * a + 1] += ny; F[3 * a + 2] += nz;
      F[3 * b] -= nx; F[3 * b + 1] -= ny; F[3 * b + 2] -= nz;
      T[3 * j] += ey * nz - ez * ny; T[3 * j + 1] += ez * nx - ex * nz; T[3 * j + 2] += ex * ny - ey * nx;
      this.N[j] = n3; this.V[2 * j] = n1; this.V[2 * j + 1] = n2;
      eStretch += 0.5 * l0 * n3 * s3; eShear += 0.5 * l0 * (n1 * s1 + n2 * s2);
    }
    const psi = new Float64Array(3);
    for (let j = 0; j < p; j++) {
      const sa = st.jSeg0[j], sb = st.jSeg1[j], lb = st.jLen[j], o = 9 * sa;
      const th = relRotation(Q, sa, sb, psi);
      const d0 = psi[0] / lb - st.Omega0[3 * j], d1 = psi[1] / lb - st.Omega0[3 * j + 1], d2 = psi[2] / lb - st.Omega0[3 * j + 2];
      const m0 = this.EIx[j] * d0, m1 = this.EIy[j] * d1, m2 = this.GJ[j] * d2;
      // J_l⁻ᵀ(Ψ) m = m + ½ Ψ×m + β Ψ×(Ψ×m),  β = 1/θ² − (1 + cos θ)/(2θ sin θ)  (→ 1/12 + θ²/720)
      const beta = th < 1e-3 ? 1 / 12 + th * th / 720 : 1 / (th * th) - (1 + Math.cos(th)) / (2 * th * Math.sin(th));
      const c0 = psi[1] * m2 - psi[2] * m1, c1 = psi[2] * m0 - psi[0] * m2, c2 = psi[0] * m1 - psi[1] * m0;
      const e0 = psi[1] * c2 - psi[2] * c1, e1 = psi[2] * c0 - psi[0] * c2, e2 = psi[0] * c1 - psi[1] * c0;
      const j0 = m0 + 0.5 * c0 + beta * e0, j1 = m1 + 0.5 * c1 + beta * e1, j2 = m2 + 0.5 * c2 + beta * e2;
      const tx = j0 * R[o] + j1 * R[o + 3] + j2 * R[o + 6];
      const ty = j0 * R[o + 1] + j1 * R[o + 4] + j2 * R[o + 7];
      const tz = j0 * R[o + 2] + j1 * R[o + 5] + j2 * R[o + 8];
      T[3 * sa] += tx; T[3 * sa + 1] += ty; T[3 * sa + 2] += tz;
      T[3 * sb] -= tx; T[3 * sb + 1] -= ty; T[3 * sb + 2] -= tz;
      this.M[3 * j] = m0; this.M[3 * j + 1] = m1; this.M[3 * j + 2] = m2;
      eBend += 0.5 * lb * (m0 * d0 + m1 * d1); eTwist += 0.5 * lb * m2 * d2;
    }
    // clamp joints: frozen frame c (side a) → segment frame j (side b)
    if (this.cSeg.length) {
      const Rc = new Float64Array(9), QQ = new Float64Array(8);
      for (let c = 0; c < this.cSeg.length; c++) {
        const j = this.cSeg[c], lc = this.cLen[c], B = this.cB[c];
        QQ.set(this.cQ.subarray(4 * c, 4 * c + 4), 0); QQ.set(Q.subarray(4 * j, 4 * j + 4), 4);
        const th = relRotation(QQ, 0, 1, psi);
        toFrame(QQ, 0, Rc);
        const d0 = psi[0] / lc, d1 = psi[1] / lc, d2 = psi[2] / lc;
        const m0 = B[0] * d0, m1 = B[1] * d1, m2 = B[2] * d2;
        const beta = th < 1e-3 ? 1 / 12 + th * th / 720 : 1 / (th * th) - (1 + Math.cos(th)) / (2 * th * Math.sin(th));
        const c0 = psi[1] * m2 - psi[2] * m1, c1 = psi[2] * m0 - psi[0] * m2, c2 = psi[0] * m1 - psi[1] * m0;
        const e0 = psi[1] * c2 - psi[2] * c1, e1 = psi[2] * c0 - psi[0] * c2, e2 = psi[0] * c1 - psi[1] * c0;
        const j0 = m0 + 0.5 * c0 + beta * e0, j1 = m1 + 0.5 * c1 + beta * e1, j2 = m2 + 0.5 * c2 + beta * e2;
        T[3 * j] -= j0 * Rc[0] + j1 * Rc[3] + j2 * Rc[6];
        T[3 * j + 1] -= j0 * Rc[1] + j1 * Rc[4] + j2 * Rc[7];
        T[3 * j + 2] -= j0 * Rc[2] + j1 * Rc[5] + j2 * Rc[8];
        this.cM[2 * c] = m0; this.cM[2 * c + 1] = m1;
        eBend += 0.5 * lc * (m0 * d0 + m1 * d1); eTwist += 0.5 * lc * m2 * d2;
      }
    }
    this.energyParts = { bending: eBend, torsion: eTwist, stretch: eStretch, shear: eShear };
    this.energy = eStretch + eShear + eBend + eTwist;
  }

  /** node masses: the stretch/shear spring of each segment (bending reaches the nodes through the frames) */
  addStiffness(st, kNode, kSeg) {
    for (let j = 0; j < st.m; j++) {
      const k = (this.EA[j] + this.GA[j]) / st.l0[j];
      kNode[st.segI0[j]] += k; kNode[st.segI1[j]] += k;
    }
  }

  /** rotational stiffness per segment and material axis (3m): joints B/l̄, clamps B/l_c, the shear lever κGA·L²/l0 about
   *  X and Y (fictitious inertias of the dynamic relaxation; built with the goals, before any force evaluation) */
  addRotStiffness(st, kRot) {
    const X = st.X;
    for (let j = 0; j < st.p; j++) {
      const lb = Math.max(st.jLen[j], 1e-12), B = [this.EIx[j], this.EIy[j], this.GJ[j]];
      for (const s of [st.jSeg0[j], st.jSeg1[j]]) for (let k = 0; k < 3; k++) kRot[3 * s + k] += B[k] / lb;
    }
    for (let c = 0; c < this.cSeg.length; c++) for (let k = 0; k < 3; k++) kRot[3 * this.cSeg[c] + k] += this.cB[c][k] / Math.max(this.cLen[c], 1e-12);
    for (let j = 0; j < st.m; j++) {
      const a = st.segI0[j], b = st.segI1[j];
      const L = Math.hypot(X[3 * b] - X[3 * a], X[3 * b + 1] - X[3 * a + 1], X[3 * b + 2] - X[3 * a + 2]);
      const k = this.GA[j] * L * L / st.l0[j];
      kRot[3 * j] += k; kRot[3 * j + 1] += k;
    }
  }
}
