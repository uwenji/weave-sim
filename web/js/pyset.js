// A CPython-compatible set of small non-negative ints (hash(i) = i): the same table, probing and resize rules as
// Objects/setobject.c, so iterating it gives the order Python gives. Only used where the Python program's result
// depends on set iteration order (the greedy family colouring in topology._assign_families), so that the lath
// families here carry the same numbers as in the Python viewer.
const LINEAR_PROBES = 9, PERTURB_SHIFT = 5, MINSIZE = 8;

export class PySet {
  constructor() { this.mask = MINSIZE - 1; this.keys = new Array(MINSIZE).fill(null); this.fill = 0; this.used = 0; }

  get size() { return this.used; }

  has(key) {
    let i = key & this.mask, perturb = key;
    for (;;) {
      const probes = i + LINEAR_PROBES <= this.mask ? LINEAR_PROBES : 0;
      for (let p = 0, j = i; p <= probes; p++, j++) {
        const k = this.keys[j];
        if (k === null) return false;
        if (k === key) return true;
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & this.mask;
    }
  }

  add(key) {
    let i = key & this.mask, perturb = key;
    for (;;) {
      const probes = i + LINEAR_PROBES <= this.mask ? LINEAR_PROBES : 0;
      for (let p = 0, j = i; p <= probes; p++, j++) {
        const k = this.keys[j];
        if (k === null) {
          this.keys[j] = key; this.fill++; this.used++;
          if (this.fill * 5 < this.mask * 3) return;
          this._resize(this.used > 50000 ? this.used * 2 : this.used * 4);
          return;
        }
        if (k === key) return;
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & this.mask;
    }
  }

  static _insertClean(keys, mask, key) {
    let i = key & mask, perturb = key;
    for (;;) {
      if (keys[i] === null) { keys[i] = key; return; }
      if (i + LINEAR_PROBES <= mask) for (let j = 1; j <= LINEAR_PROBES; j++) if (keys[i + j] === null) { keys[i + j] = key; return; }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  _resize(minused) {
    let newsize = MINSIZE; while (newsize <= minused) newsize *= 2;
    const old = this.keys, nk = new Array(newsize).fill(null), nm = newsize - 1;
    for (const k of old) if (k !== null) PySet._insertClean(nk, nm, k);
    this.keys = nk; this.mask = nm; this.fill = this.used;
  }

  *[Symbol.iterator]() { for (const k of this.keys) if (k !== null) yield k; }

  /** set − other (other a PySet), CPython's set_difference */
  difference(other) {
    const r = new PySet();
    if ((this.used >> 2) > other.size) {                  // set_copy_and_difference: copy, then discard (no dummies in our use)
      for (const k of this) if (!other.has(k)) r.add(k);  // (discarding leaves dummies; equivalent order for our small sets)
      return r;
    }
    for (const k of this) if (!other.has(k)) r.add(k);
    return r;
  }

  /** self |= other (other a PySet), CPython's set_merge */
  update(other) {
    if (other === this || other.used === 0) return;
    if ((this.fill + other.used) * 5 >= this.mask * 3) this._resize((this.used + other.used) * 2);
    if (this.fill === 0 && this.mask === other.mask && other.fill === other.used) {
      this.keys = other.keys.slice(); this.fill = other.fill; this.used = other.used; return;
    }
    if (this.fill === 0) {
      for (const k of other.keys) if (k !== null) PySet._insertClean(this.keys, this.mask, k);
      this.fill = other.used; this.used = other.used; return;
    }
    for (const k of other.keys) if (k !== null) this.add(k);
  }
}
