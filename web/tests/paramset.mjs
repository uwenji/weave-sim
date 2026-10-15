// parameter sets: export from a calibration (cm·N), apply to scenes in other units / with another section
import assert from 'node:assert/strict';
import { loadForceExample } from './calib_cpu.mjs';
import { parameterSet, applyParameterSet, paramFactor } from '../js/calibrate.js';
import { Scene } from '../js/scene.js';
import { UnitSystem } from '../js/units.js';
const cs = await loadForceExample();
const best = { mu: 0.4, torsion_factor: 1.5, strength: 12000, stick_strength: 9000, activation: 0.8, pad_stiffness: 2500 };
const ps = JSON.parse(JSON.stringify(parameterSet(cs, best, { objective: 0.12, residuals: { P1: 0.05 }, method: 'cmaes' })));
assert.equal(ps.kind, 'weavesim-parameter-set'); assert.deepEqual(ps.units, { length: 'cm', force: 'N' });
assert.equal(ps.values.E, 1e6); assert.equal(ps.values.mu, 0.4); assert.equal(ps.classes.pad_stiffness, 'test');
// same units: values as they are, the pad stays with the test rig
const a = new Scene(); a.units = new UnitSystem('cm', 'N'); a.section = { ...cs.scene.section };
let r = applyParameterSet(a, ps);
assert.equal(a.contact.mu, 0.4); assert.equal(a.contact.strength, 12000); assert.equal(a.material.E, 1e6); assert.equal(a.beam.activation, 0.8);
assert.ok(!('pad_stiffness' in r.applied)); assert.equal(r.warnings.length, 0);
// mm·N: E [N/cm²] → ×0.01 [N/mm²]; strength [N/cm] → ×0.1 [N/mm]; mu unit-free
const b = new Scene(); b.units = new UnitSystem('mm', 'N'); b.section = { width: 20, height: 8, Ix: null, Iy: null, J: null };
r = applyParameterSet(b, ps);
assert.ok(Math.abs(b.material.E - 1e4) < 1e-9, b.material.E); assert.ok(Math.abs(b.contact.strength - 1200) < 1e-9); assert.equal(b.contact.mu, 0.4);
assert.equal(r.warnings.length, 0, r.warnings.join());                 // 2×0.8 cm = 20×8 mm: same lath
// another section: flagged
const c = new Scene(); c.units = new UnitSystem('mm', 'N'); c.section = { width: 60, height: 30, Ix: null, Iy: null, J: null };
r = applyParameterSet(c, ps);
assert.equal(r.warnings.length, 1); console.log('warning:', r.warnings[0]);
assert.ok(Math.abs(paramFactor('E', new UnitSystem('m', 'kN'), new UnitSystem('mm', 'N')) - 1e-3) < 1e-15);
console.log('parameter sets: ok');
