// Parameter groups (port of weavesim/params.py). All values are in scene units.
// Section axes: X = lath width (in surface) → Ix = w h³/12 (flatwise, weak); Y = surface normal (thickness)
// → Iy = h w³/12 (edgewise, strong); Z = strand tangent → J (Roark rectangle).
// Plain objects are used so a scene.json round-trips unchanged; the functions below derive what the solver needs.

export const MATERIAL = { E: 10000.0, G: 625.0, density: 700.0, E90: 800.0, fc90: 8.0, H: 30.0 };
export const SECTION = { width: 60.0, height: 30.0, Ix: null, Iy: null, J: null };
export const CONTACT = {
  mu: 0.35, strength: 1e4, stick_strength: 1e4, torsion_factor: 1.0, epsilon: 15.0, ramp_iter: 1500, slip_rate: 1.0,
  locked: false, rot_friction: false, rot_factor: 1.0, seat_law: 'off', seat_gap: true, dent_rate: 0.01,
  dent_after_iter: -1, edge_threshold: 0.5,
  // augmented-Lagrangian gap (README "5 Contact"): Fn = max(0, λ + strength·pen), λ ← max(0, λ + strength·pen)
  // every al_every iterations once the penalty ramp is over
  augmented: false, al_every: 600,
};
export const BEAM = {
  mode: 'simple', activation: 1.0, stiffness: 1.0, activation_flat: 1.0, activation_edge: 1.0, activation_twist: 1.0,
  stiffness_flat: 1.0, stiffness_edge: 1.0, stiffness_twist: 1.0, stiffness_axial: 1.0,
};
export const SOLVER = {
  max_iter: 5000, tol: 1e-4, damping: 1.0, kinetic_damping: true, mass_scale: 1.5, warmup_iter: 300, beam_weight: 1.0,
  prestress: 'flat', live_every: 10,
};
export const WEAVE = { subdivisions: 4, overhang: 0.0, offset_mode: 'half_height', flip_knots: [], seed_strand_phase: 0 };

/** a fresh object with the defaults, overridden by the known keys of d (unknown keys are dropped, as in Python) */
export function fromDict(defaults, d) {
  const out = structuredClone(defaults);
  if (d) for (const k of Object.keys(defaults)) if (k in d && d[k] !== undefined) out[k] = structuredClone(d[k]);
  return out;
}

export function rectTorsionConstant(a, b) {
  if (a < b) [a, b] = [b, a];
  const r = b / a;
  return a * b ** 3 * (1.0 / 3.0 - 0.21 * r * (1.0 - r ** 4 / 12.0));
}
export const sectionArea = (s) => s.width * s.height;
export const IxEff = (s) => (s.Ix != null ? s.Ix : s.width * s.height ** 3 / 12.0);
export const IyEff = (s) => (s.Iy != null ? s.Iy : s.height * s.width ** 3 / 12.0);
export const JEff = (s) => (s.J != null ? s.J : rectTorsionConstant(s.width, s.height));
export function sectionStiffness(s, m) {
  return { EA: m.E * sectionArea(s), EIx: m.E * IxEff(s), EIy: m.E * IyEff(s), GJ: m.G * JEff(s) };
}

const clamp01 = (v) => Math.min(Math.max(+v, 0.0), 1.0);
/** (flatwise, edgewise, twist) prestress shares in [0, 1] */
export function beamActivations(b) {
  if (b.mode === 'advanced') return [clamp01(b.activation_flat), clamp01(b.activation_edge), clamp01(b.activation_twist)];
  const a = clamp01(b.activation); return [a, a, a];
}
/** stiffness multipliers keyed like sectionStiffness() */
export function beamFactors(b) {
  if (b.mode === 'advanced') return { EA: +b.stiffness_axial, EIx: +b.stiffness_flat, EIy: +b.stiffness_edge, GJ: +b.stiffness_twist };
  const k = +b.stiffness; return { EA: 1.0, EIx: k, EIy: k, GJ: k };
}

// --- unit conversion --------------------------------------------------------------------------------------------
export function rescaleMaterial(m, src, dst) {
  const s = src.stressFactorTo(dst);
  return { ...m, E: m.E * s, G: m.G * s, E90: m.E90 * s, fc90: m.fc90 * s, H: m.H * s };
}
export function rescaleSection(sec, src, dst) {
  const L = src.lengthFactorTo(dst), f4 = L ** 4;
  return { width: sec.width * L, height: sec.height * L, Ix: sec.Ix == null ? null : sec.Ix * f4, Iy: sec.Iy == null ? null : sec.Iy * f4, J: sec.J == null ? null : sec.J * f4 };
}
/** strength / stick_strength are penalties [force/length]; epsilon is a length; the rest is unit-free */
export function rescaleContact(c, src, dst) {
  const L = src.lengthFactorTo(dst), F = src.forceFactorTo(dst);
  return { ...c, strength: c.strength * F / L, stick_strength: c.stick_strength * F / L, epsilon: c.epsilon * L };
}
export function rescaleWeave(w, src, dst) { return { ...w, overhang: w.overhang * src.lengthFactorTo(dst) }; }
