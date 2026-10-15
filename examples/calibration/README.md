# UTM calibration — how to run

Which parameters to measure, fix and fit, and how to test transfer to another topology: **CALIBRATION_PLAN.md**.

Two synthetic experiments are included so the pipeline can be run before lab data exists.
Both use the v5dome scene in cm · N (laths 2 × 0.8 cm, E 1e6 N/cm²).

| folder   | protocol                                                     | status |
|----------|--------------------------------------------------------------|--------|
| `force/` | **lab protocol**: free crossings, cable pull at the crossing along gravity, load–unload cycles, probe displacement read | current |
| `.` (root) | pinned-probe push, locked crossings, reaction read            | superseded, kept for reference |

## 1. Install (once, in the weavesim environment)

```bash
pip install -e ".[dev,plot]"      # matplotlib is only needed for `calibrate plot`
```

## 2. Run the synthetic experiment (`force/`)

All paths inside `calib.json` are relative to the folder that holds it, so run from anywhere:

```bash
cd WeaveSim/examples/calibration/force

# optional — regenerate the synthetic "UTM" curves from known parameters (≈ 2 min)
weavesim calibrate synth calib.json \
    --true '{"mu":0.45,"torsion_factor":1.5,"strength":15000,"stick_strength":8000,"activation":0.7,"pad_stiffness":3000}' \
    --noise 0.01

# one-at-a-time sensitivity sweep, 3 values per parameter (18 evaluations ≈ 30 min)
weavesim calibrate sensitivity calib.json --n 3

# fit: 8 Latin-hypercube points + Nelder–Mead (40 evaluations ≈ 1 h), then validation on P4
weavesim calibrate fit calib.json --n-init 8 --max-eval 40

# plot measured vs fitted (and vs the truth, for the synthetic case) → fit_curves.png (≈ 4 min)
weavesim calibrate plot calib.json --true "$(cat synthetic_truth.json)"
```

One evaluation = one relaxation (2 500 iterations) + every test replayed from that state
(2 cycles × 2 branches × 4 increments × 500 iterations each), tests in parallel
(`replay.workers`). ≈ 100 s on 2 cores; set `"workers": 4` on a 4-core Mac.

Outputs (in the folder of calib.json): `calibrate_<action>.log`, `sensitivity.json`,
`fit_log.csv` (every evaluation: parameters, objective, per-test residual), `fit.json`
(best, residuals, validation, full history), `fit_curves.png`.

The included results: `synthetic_truth.json` → `fit.json` reached objective 0.20 after 40
evaluations while the truth scores 0.022 — the optimiser, not the model, is the limit; see
the project note / ARCHITECTURE §6.2.

## 3. Use real lab data

1. One CSV per probe with the **whole cyclic trace** (loading and unloading, all cycles):
   two columns, displacement and force, any header the reader recognises (`Displacement`,
   `Travel`, `Load`, `Force` …); units via `scene.utm.units` or the CSV header. Drops from
   the mechanical setting are harmless (cycles are split with a 2 % hysteresis band).
2. Copy `force/calib.json`, point `"scene"` at your model's scene.json (anchors and gravity
   as in the lab), set each test's `"frame"` to the crossing the cable was attached to —
   pick the knot in the viewer: the hover tooltip shows `frame (strand, k)` — and `"axis"`
   to the cable direction (gravity: `[0, 0, -1]` for z-up).
3. `"fixed"`: E from the lath bending test; cable/fixture stiffness and the crossing
   compliance (`strength`, `stick_strength`) from their own tests when available — every
   series compliance left free widens the valley the optimiser has to resolve.
4. `"replay"`: `cycles` = number of lab cycles, `fit_cycles` = the ones to fit (0-based;
   skip the bedding-in cycle), `max_force` null (taken from the CSV peak), `n_steps` 4–8.
5. Run `sensitivity` first (which parameters move the curves?), then `fit`, then `plot`.
   Keep one probe out of `tests` as `validation`.

`calib.json` keys: `scene`, `params` {name: [lo, hi]}, `fixed` {name: value}, `tests` /
`validation` [{name, frame, axis, csv, weight}], `replay` {mode, relax_iter, step_iter,
avg_iter, n_steps, cycles, fit_cycles, max_force, workers, lock}. Fittable names:
`E G stiffness stiffness_twist activation mu torsion_factor strength stick_strength slip_rate pad_stiffness`.
