// Seeded random numbers shared by the calibration (Latin hypercube, synthetic noise) and the discretisation (annealing).
// xoshiro128** seeded by splitmix32. The numpy generator of the Python program is not reproduced: the same seed gives
// other samples (a run is reproducible within WeaveSim Web).
function splitmix32(a) {
  return () => { a |= 0; a = (a + 0x9e3779b9) | 0; let t = a ^ (a >>> 16); t = Math.imul(t, 0x21f0aaad); t ^= t >>> 15; t = Math.imul(t, 0x735a2d97); return (t ^ (t >>> 15)) >>> 0; };
}
const rotl = (x, k) => (x << k) | (x >>> (32 - k));
/** xoshiro128** seeded by splitmix32 (the numpy generator is not reproduced: the same seed gives other samples) */
export class Rng {
  constructor(seed = 1) { const sm = splitmix32(seed >>> 0); this.s = new Uint32Array([sm(), sm(), sm(), sm()]); this._n = null; }
  u32() {
    const s = this.s, r = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0, t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11);
    return r;
  }
  uniform() { return ((this.u32() >>> 5) * 67108864 + (this.u32() >>> 6)) / 9007199254740992; }
  /** integer in [0, n) */
  int(n) { return Math.min(n - 1, Math.floor(this.uniform() * n)); }
  normal() {
    if (this._n !== null) { const v = this._n; this._n = null; return v; }
    let u = 0; while (u === 0) u = this.uniform();
    const v = this.uniform(), r = Math.sqrt(-2 * Math.log(u));
    this._n = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  permutation(n) { const p = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(this.uniform() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; }
}
