// Scene: everything needed to reproduce a run (port of weavesim/scene.py). Reads and writes the same scene.json
// as the Python program (version 1); paths inside the file are relative to the file's own location.
import { UnitSystem } from './units.js';
import {
  MATERIAL, SECTION, CONTACT, BEAM, SOLVER, WEAVE, fromDict, rescaleMaterial, rescaleSection, rescaleContact, rescaleWeave,
} from './params.js';

export const SCENE_VERSION = 1;
export const DOF_NAMES = ['X', 'Y', 'Z', 'rX', 'rY', 'rZ'];

const selectorFrom = (d) => { d = d || {}; const s = { type: d.type || 'frame' }; for (const k of ['id', 'strand', 'ids', 'radius']) if (d[k] != null) s[k] = structuredClone(d[k]); return s; };
export function selectorLabel(s) {
  if (s.type === 'frame') return `frame (${s.id})`;
  if (s.type === 'strand') return `strand ${s.strand}`;
  if (s.type === 'knots') return `${(s.ids || []).length} frames`;
  if (s.type === 'region') return `region r=${s.radius} @ (${s.id})`;
  return s.type;
}
export const fixLabel = (fix) => DOF_NAMES.map((n, i) => (fix[i] ? n : '·')).join('');

export class Scene {
  constructor() {
    this.units = new UnitSystem();
    this.mesh = { path: '', scale: 1.0 };
    this.weave = fromDict(WEAVE);
    this.material = fromDict(MATERIAL);
    this.section = fromDict(SECTION);
    this.contact = fromDict(CONTACT);
    this.beam = fromDict(BEAM);
    this.solver = fromDict(SOLVER);
    this.discretize = null;                     // kept verbatim for the Python tools (joint placement)
    this.joints = {};
    this.hybrid = { enabled: false, groups: [], assign: {} };
    this.loads = { gravity: { enabled: true, direction: [0, 0, -1] }, udl: { enabled: false, q: [0, 0, 0], per: 'length' }, point: [] };
    this.anchors = [];
    this.utm = { path: null, mode: 'overlay', probe: null, columns: { force: null, displacement: null }, units: { force: 'N', length: 'mm' }, pad_stiffness: 0.0 };
    this.name = 'scene';
    this.extra = {};                            // unknown top-level keys, written back unchanged
  }

  // --- lath groups (hybrid weave) ---------------------------------------------------------------------------
  get nGroups() { return this.hybrid.enabled ? 1 + this.hybrid.groups.length : 1; }
  /** group g (1-based); group 1 is a view on the scene's own material / section / beam */
  group(g) {
    if (g <= 1 || !this.hybrid.enabled || g - 2 >= this.hybrid.groups.length) return { name: 'group 1', material: this.material, section: this.section, beam: this.beam };
    return this.hybrid.groups[g - 2];
  }
  strandGroup(s) {
    if (!this.hybrid.enabled) return 1;
    const g = +(this.hybrid.assign[s] ?? 1);
    return g >= 2 && g <= this.nGroups ? g : 1;
  }
  heightOfStrand(s) { return +this.group(this.strandGroup(s)).section.height; }
  setNGroups(n) {
    n = Math.max(1, n | 0);
    while (1 + this.hybrid.groups.length < n) {
      const k = 2 + this.hybrid.groups.length;
      this.hybrid.groups.push({ name: `group ${k}`, material: structuredClone(this.material), section: structuredClone(this.section), beam: structuredClone(this.beam) });
    }
    this.hybrid.groups.length = Math.max(n - 1, 0);
    for (const s of Object.keys(this.hybrid.assign)) if (this.hybrid.assign[s] > n) delete this.hybrid.assign[s];
  }
  assignStrands(strands, g) { for (const s of strands) { if (g >= 2) this.hybrid.assign[s] = g; else delete this.hybrid.assign[s]; } }

  // --- units --------------------------------------------------------------------------------------------------
  convertUnits(nu) {
    const old = this.units; if (old.equals(nu)) return;
    const L = old.lengthFactorTo(nu), F = old.forceFactorTo(nu);
    this.material = rescaleMaterial(this.material, old, nu);
    this.section = rescaleSection(this.section, old, nu);
    for (const g of this.hybrid.groups) { g.material = rescaleMaterial(g.material, old, nu); g.section = rescaleSection(g.section, old, nu); }
    this.contact = rescaleContact(this.contact, old, nu);
    this.weave = rescaleWeave(this.weave, old, nu);
    this.mesh.scale *= L;
    if (this.discretize) { if (this.discretize.max_length != null) this.discretize.max_length *= L; if (this.discretize.joint_length != null) this.discretize.joint_length *= L; }
    for (const k of Object.keys(this.joints)) this.joints[k] = this.joints[k].map(v => v * L);
    for (const p of this.loads.point) p.F = p.F.map(f => f * F);
    this.loads.udl.q = this.loads.udl.per === 'length' ? this.loads.udl.q.map(q => q * F / L) : this.loads.udl.q.map(q => q * F / (L * L));
    this.units = nu;
  }

  // --- serialisation --------------------------------------------------------------------------------------------
  toJSON() {
    const d = {
      version: SCENE_VERSION,
      units: this.units.toJSON(),
      mesh: { path: this.mesh.path, scale: this.mesh.scale },
      weave: { ...this.weave, flip_knots: this.weave.flip_knots.map(f => [f[0] | 0, f[1] | 0]) },
      material: this.material, section: this.section, contact: this.contact, beam: this.beam, solver: this.solver,
      ...(this.discretize ? { discretize: this.discretize } : {}),
      joints: this.joints,
      hybrid: { enabled: this.hybrid.enabled, groups: this.hybrid.groups, assign: Object.fromEntries(Object.entries(this.hybrid.assign).map(([k, v]) => [String(k), v | 0])) },
      loads: this.loads,
      anchors: this.anchors.map(a => ({ select: a.select, fix: a.fix.map(f => (f ? 1 : 0)), name: a.name || '' })),
      utm: this.utm,
      ...this.extra,
    };
    return structuredClone(d);
  }

  static fromJSON(d, name = 'scene') {
    d = d || {};
    const v = d.version ?? SCENE_VERSION;
    if (v > SCENE_VERSION) throw new Error(`scene version ${v} newer than supported ${SCENE_VERSION}`);
    const sc = new Scene();
    sc.name = name;
    sc.units = UnitSystem.fromJSON(d.units);
    const mesh = d.mesh || {};
    sc.mesh = { path: mesh.path || '', scale: +(mesh.scale ?? 1.0) };
    sc.weave = fromDict(WEAVE, d.weave);
    sc.weave.flip_knots = (sc.weave.flip_knots || []).map(f => [f[0] | 0, f[1] | 0]);
    sc.material = fromDict(MATERIAL, d.material);
    const md = d.material || {};
    if (['E90', 'fc90', 'H'].some(k => !(k in md))) {           // older scene: the defaults are N/mm²
      const s = new UnitSystem('mm', 'N').stressFactorTo(sc.units);
      for (const k of ['E90', 'fc90', 'H']) if (!(k in md)) sc.material[k] = MATERIAL[k] * s;
    }
    sc.section = fromDict(SECTION, d.section);
    sc.contact = fromDict(CONTACT, d.contact);
    sc.beam = fromDict(BEAM, d.beam);
    sc.solver = fromDict(SOLVER, d.solver);
    sc.discretize = d.discretize ? structuredClone(d.discretize) : null;
    sc.joints = d.joints ? structuredClone(d.joints) : {};
    const h = d.hybrid || {};
    sc.hybrid = {
      enabled: !!h.enabled,
      groups: (h.groups || []).map(g => ({ name: g.name || '', material: fromDict(MATERIAL, g.material), section: fromDict(SECTION, g.section), beam: fromDict(BEAM, g.beam) })),
      assign: Object.fromEntries(Object.entries(h.assign || {}).map(([k, v]) => [k | 0, v | 0])),
    };
    const L = d.loads || {};
    sc.loads = {
      gravity: { enabled: L.gravity?.enabled ?? true, direction: [...(L.gravity?.direction || [0, 0, -1])] },
      udl: { enabled: !!L.udl?.enabled, q: [...(L.udl?.q || [0, 0, 0])], per: L.udl?.per || 'length' },
      point: (L.point || []).map(p => ({ frame: [p.frame[0] | 0, p.frame[1] | 0], F: p.F.map(Number), space: p.space || 'world', name: p.name || '' })),
    };
    sc.anchors = (d.anchors || []).map(a => ({ select: selectorFrom(a.select), fix: (a.fix || [1, 1, 1, 0, 0, 0]).map(f => (f ? 1 : 0)), name: a.name || '' }));
    const u = d.utm || {};
    sc.utm = {
      path: u.path ?? null, mode: u.mode || 'overlay', probe: u.probe ? structuredClone(u.probe) : null,
      columns: { force: null, displacement: null, ...(u.columns || {}) }, units: { force: 'N', length: 'mm', ...(u.units || {}) },
      pad_stiffness: +(u.pad_stiffness ?? 0),
    };
    const known = new Set(['version', 'units', 'mesh', 'weave', 'material', 'section', 'contact', 'beam', 'solver', 'discretize', 'joints', 'hybrid', 'loads', 'anchors', 'utm']);
    for (const k of Object.keys(d)) if (!known.has(k)) sc.extra[k] = structuredClone(d[k]);
    return sc;
  }

  addPointLoad(frame, F, space = 'world', name = '') { const p = { frame: [frame[0] | 0, frame[1] | 0], F: F.map(Number), space, name }; this.loads.point.push(p); return p; }
  addAnchor(select, fix, name = '') { const a = { select: selectorFrom(select), fix: fix.map(f => (f ? 1 : 0)), name }; this.anchors.push(a); return a; }
}
