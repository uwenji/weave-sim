# Calibration plan — what to measure, fix and fit (October 2026)

Goal: one parameter set per **material + lath stock + forming process** that transfers across
weave topologies (dome → column → tree). A parameter can transfer only if it describes the
material, the crossing physics or the forming, not the specimen; every parameter that depends on
the mesh, the span or the solver is fixed by rule and never fitted.

## 1 Measured on the UTM / in the lab → `fixed`

| parameter | where | test |
|---|---|---|
| `E` | material | lath three/four-point bending (flatwise), also gives `G` if a torsion rig exists; else G = E/16 |
| `density` | material | weigh the laths |
| `fc90`, `E90` | material (embedment) | compression ⊥ grain on lath stock; `H` by ball indentation (EN 1534) |
| `mu` | contact | double-lap shear test on the lath stock, same surface finish and moisture |
| `pad_stiffness` | utm | cable + fixture stiffness: pull against a rigid block (or set from the measured cable) |
| section `width`, `height` | section | callipers; use per lath group for hybrid weaves |

## 2 Fixed by rule → not fitted (specimen / solver parameters)

| parameter | rule | why not fit |
|---|---|---|
| `strength`, `stick_strength` | 20 × 4 EIx / L_span³ (L_span = mean crossing spacing); `augmented: true`, `al_every` 600 (1200 for soft models) | numerical penalties; with the multiplier the gap is exact, so they no longer stand in for crossing compliance |
| `epsilon`, `ramp_iter`, `warmup_iter`, `mass_scale` | solver defaults | numerics |
| `slip_rate` | 0.3 | regularisation, not physics |
| `edge_threshold` | 0.5 | classification only |
| `subdivisions`, `overhang` | model | geometry |

The scenes `v5dome_cm`, `dualv5dome_cm`, `tree_cm` carry these values now (strength 2 300 / 4 500 / 4 700, AL on).

## 3 Fitted on the dome tests → the transferable set

Fit in this order, each stage with the previous ones fixed:

1. **`activation`** (0…1) — the forming process: how much of the woven curvature is stored as
   prestress (steamed / pre-bent laths relax part of it). Material + process, transfers.
   Identified by the loading-branch stiffness at several probes.
2. **`torsion_factor`** — lever-arm torsion scaling at the crossing. Identified by the difference
   between probes near and far from the supports (it changes the twist share). Transfers.
3. **`rot_factor`** (only with `rot_friction: true`) — rotational friction limit about the crossing
   normal. Identified by the column / skip-weave tests that shear in the surface; on a dome it is
   weakly visible. Transfers.
4. **`mu`** — keep the shear-test value fixed; fit it only as a check (± 20 % band). The unloading
   branch identifies it; if the fitted value leaves the band the shear-test surface differs from the
   weave surface (moisture, finish) and the shear test should be repeated, not the fit trusted.
5. **`fc90`, `H`** (only with `seat_law: "L1"`) — fit only against the crossing-jig tests of the
   review's §11 (press and pull), never against the dome: the dome sees them through the dents'
   Fn loss, which is confounded with activation.

`E`, `strength`, `stick_strength`, `pad_stiffness` must not be in `params`: E is measured, the
penalties are numerical, the pad/cable is a fixture.

## 4 Transfer test (dome → column / tree)

1. Fit on the dome probes (3 training + 1 validation) with the calib.json below.
2. Copy `material`, `contact.mu / torsion_factor / rot_factor`, `beam.activation` into the column
   scene; recompute `strength` from the column's crossing spacing (rule above); keep the column's
   own anchors, loads, section and `pad_stiffness` if the fixture differs.
3. Replay the column tests with no fitting: the residual on the column is the transfer error. Report
   it next to the dome validation residual. If the column error is much larger than the dome's,
   the parameter that does not transfer is the one that changes the column most in the
   sensitivity sweep — run `calibrate sensitivity` on the column scene with the dome's values as
   centre to find it.

## 5 calib.json for the dome (template)

```json
{
  "scene": "dome_lab.json",
  "fixed": {"E": <measured>, "G": <measured or E/16>, "mu": <shear test>, "fc90": <measured>, "H": <measured>,
            "pad_stiffness": <measured>, "strength": <20·4EIx/L³>, "stick_strength": <same>},
  "params": {"activation": [0.2, 1.0], "torsion_factor": [0.3, 3.0]},
  "tests": [...3 probes...], "validation": [...1 probe...],
  "replay": {"mode": "force", "cycles": 3, "fit_cycles": [1, 2], "n_steps": 6, "workers": 4, "lock": false}
}
```

Two free parameters: Nelder–Mead with 30–40 evaluations resolves this; add `rot_factor` when the
column tests exist and `mu` only as the check in §3.
