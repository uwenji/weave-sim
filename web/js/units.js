// Unit system: a (length, force) pair; everything in a scene is stored in scene units (port of weavesim/units.py).
export const LENGTH_TO_M = { mm: 1e-3, cm: 1e-2, m: 1.0 };
export const FORCE_TO_N = { N: 1.0, kN: 1e3 };
export const G_SI = 9.80665;

export class UnitSystem {
  constructor(length = 'mm', force = 'N') {
    if (!(length in LENGTH_TO_M)) throw new Error(`unknown length unit ${length}`);
    if (!(force in FORCE_TO_N)) throw new Error(`unknown force unit ${force}`);
    this.length = length; this.force = force;
  }
  get L() { return LENGTH_TO_M[this.length]; }                 // scene length → m
  get F() { return FORCE_TO_N[this.force]; }                   // scene force → N
  get stress() { return this.F / (this.L * this.L); }          // scene stress → Pa
  get stressLabel() { return `${this.force}/${this.length}²`; }
  get gravity() { return G_SI / this.L; }                      // g in scene length / s²
  lengthFactorTo(o) { return this.L / o.L; }
  forceFactorTo(o) { return this.F / o.F; }
  stressFactorTo(o) { return this.stress / o.stress; }
  /** weight per unit length [force / length] of a section of `area` (scene length²) */
  selfWeightLineLoad(density, area) {
    const areaSI = area * this.L * this.L;
    const qSI = density * areaSI * G_SI;                       // N/m
    return qSI / this.F * this.L;
  }
  equals(o) { return !!o && o.length === this.length && o.force === this.force; }
  toJSON() { return { length: this.length, force: this.force }; }
  static fromJSON(d) { d = d || {}; return new UnitSystem(d.length || 'mm', d.force || 'N'); }
  toString() { return `${this.length}·${this.force}`; }
}
