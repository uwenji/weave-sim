# How to start — WeaveSim Web

## 1 · On your Mac

```bash
conda activate weavesim                 # any Python 3 works; nothing to install
cd ".../330_Modelling/WeaveSim"
python web/serve.py
```

Open <http://localhost:8090/web/> in Chrome (or Edge, or Safari 26+). The pill at the top right says which solver runs:
`WebGPU · apple` is the GPU solver, `CPU · f64` the double-precision reference — click the pill (or section 7 Solve) to
switch. Opening `index.html` by double-click does not work: the page loads its modules and the example scenes over
http. `serve.py` serves the whole WeaveSim folder (so `examples/` is reachable) and sends no-cache headers, so edited
files show up on a plain reload. Another port: `python web/serve.py --port 9000`. Stop with Ctrl-C.

### Other devices on the same Wi-Fi

`serve.py` prints a second address, `http://192.168.x.x:8090/web/`. Over plain http another device gets the CPU solver
only (browsers enable WebGPU on localhost or https). For WebGPU there, run `python web/serve.py --https` and accept the
certificate warning once (the certificate is made with openssl on the first start).

## 2 · First five minutes

The window has three sidebars around the 3-D view: **Setup** on the left (the numbered sections, one open at a time),
**Monitor** and **Display** on the right. Each folds to a narrow tab with its header; `⇄` moves Monitor or Display to
the other side of the view; drag the inner edge of the Monitor to widen it (two columns of charts from ~580 px).
The **i** icons explain a control on hover (or tap).

1. **File** — the example list loads `v5 dome (cm)` at the start; any WeaveSim `scene.json` opens with its mesh (select
   both files, or drop them on the view). A mesh alone starts a new scene with the current settings and anchors at the
   strand ends.
2. **Monitor** — the run bar on top stays put: `Run` / `Pause` (Space), `Step` n iterations, `max` iterations per
   Run, `Reset` (back to the woven geometry: discards the solve, its history, the followed values and the Zero
   reference), and `Zero` [settle iterations] for the load test (6). The bar under them is the convergence meter: it
   fills as the residual falls from its first value to the tolerance, one cell per decade still to go. Below, groups
   that fold independently: *Solver* (convergence: residual and kinetic energy, log scale; energy: total and its
   parts), *Knots*, *Stability*, *Probe · UTM*, *Extremes* (the ranges of the results) and *Discretisation* (7). Hover
   a chart for its values: against the iteration, one crosshair runs through all the charts.
3. **Knots** (Monitor) — `Pick knots`, then click crossings in the view (up to 8; `+ selection` takes the crossings of
   the selected frames). Each followed crossing keeps one colour everywhere: its chip, its line in every chart and its
   marker in the view. Every chart shows one quantity for all followed crossings — the mobilised friction coefficient
   μ = Ft / Fn against μ, Fn, Ft, slip, sliding along each lath, |u|, the lath stress σ next to the crossing, the
   embedment p / fc90, gap, roll, scissor angle … — `+ chart` adds one (up to 6), the title picks the quantity, `×`
   removes it. *against* sets the shared x axis: the iteration, the probe travel, |u|, Fn (with Ft: the friction cone)
   or slip. The table under the charts lists the values at the hovered moment.
4. **Stability** (Monitor) — `Check now` measures the softest mode of the tangent stiffness of the current state (0 at
   a limit point or buckling, negative when unstable); *after every run* repeats it at the end of every Run / Step.
5. **Display** — *Colour*: the laths by family, a pure colour, over / under, pieces (joints), lath group or strand, or
   by a result layer: Mx, My, T, N, a crossing quantity (Fn, Ft, slip, stick / slip, gap, pressure, utilisation,
   contact type, dent, M_rot, θ, λ) or the displacement, *smooth* (node values along each lath) or per segment,
   *bands*, *delta* (after `Zero`) and a fixed range. *Diagram*: the member diagrams (Mx, My, Vy, Vx, T, N, curvature)
   hung on the laths, the deflected shape, contact forces and reactions, with value labels at the extremes or per lath.
   *View*: theme, layers (laths, centrelines, knots, mesh, wire, shadow, joints, loads, anchors, probes, frames),
   deformation ×, arrow size, turntable. **Hover** a lath: a small window shows the lath, the position along it and the
   local value of the colour layer and the diagram (at a crossing also its forces).
6. **Load test** (Monitor run bar) — relax (`Run`), then `Zero`: the relaxed geometry becomes the reference
   (displacement and slip zero — the zigzag of the input mesh that the relaxation took out is not counted), the solver
   settles for the given iterations (the button fills, the convergence chart shades the *zero* band, the drift shows in
   the run bar — above a tenth of the lath thickness it is flagged: relax longer; click it again to stop), then the
   settled state becomes the reference and the solver pauses. Add the load, `Run`, and colour by *delta*. `0`
   iterations zeroes at once. The crossings are never locked by Zero: for fastened joints tick *lock all crossings* in
   4 Contact (the chart shows it as the blue *locked* band).
7. **Joints** (8 Discretisation for the parameters, Monitor › Discretisation to run) — stock length, joint length,
   extra joints, the objective weights (curvature at the joints, spacing, uniform pieces, keep clear of crossings,
   *structural*: away from high values of a result layer — solve first). `Run` anneals the joint positions on the
   current geometry; the cuts appear on the laths, *pieces* colours them, the Display › Diagram option *around joints
   only* shows the diagram within ±n crossings of the joints. `export joints (csv)`; the joints are saved with the scene.
8. **Loads and anchors** (5, 6) — click a lath next to a crossing to select that frame (shift adds / removes), then
   `Add to selection`. Anchors: quick-select all start / end / boundary frames, tick the fixed DOFs. Loads, anchors,
   material, beam and contact apply live to the running model; section, weave and units rebuild it.
9. **Lath groups** (3) — tick *lath groups*, choose the group to edit, `Select laths…` and click laths in the view: the
   selected laths turn orange with a dark line along them, the others fade (click a lath again to remove it, click
   anywhere else to clear, `laths of group` takes the group's current laths, Esc leaves). `Assign … to group n`.
10. **Rhino** (1 File) — `Export Rhino .3dm`: the model, and after a solve the deformed laths, the laths by result
   layer, the diagrams, contact forces, reactions, joints, supports and loads in a layer tree with user text on every
   object (Rhino 7 format) + a `.3dm.json` index. Chrome may ask once to allow two downloads.
11. **Rod and precision** — every lath is a Cosserat rod (bending about both axes, twist, stretch and shear; the roll
   moment of anisotropic bending, so lateral–torsional buckling is captured) on both solvers. The GPU solver is f32
   with 64-bit fixed-point positions, frames and force sums (bit-reproducible). Very stiff contacts leave a small
   residual floor on the GPU: `f64 polish` (7 Solve) finishes on the CPU from the GPU state.
12. **View cube** — click a face / edge / corner, drag to orbit; ⌂ home, ⤢ fit, ⟳ turntable, perspective /
   orthographic. Keys H F T P. Wheel: zoom, right-drag: pan.

## 3 · A UTM test on the model (section 9)

`Pick probe…` and click the loaded crossing; set the load axis (e.g. `0 0 -1`); `Open UTM csv…` (Mecmesin / generic
tables: the force and travel columns and their units are found in the header; choose them if not). `Replay test` runs
the lab protocol on the model (force mode = the cable pull: load steps up to the peak and back, cycles times); the
Monitor's *Probe · UTM* group plots it over the measurement, the run's residual is in section 9.

## 4 · Calibration (section 10)

* `Force example` loads `examples/calibration/force` (3 cable-pull tests + 1 validation test on a dome, synthetic
  curves). `Open calibration…` takes a `calib.json` with its scene, mesh and csv files (select
  them together, or use `Folder…`). `From current scene` starts one on the model in the view: `+ test at a crossing…`,
  then set each test's axis and csv. Tests with different load directions are fitted together.
* Tick the parameters to fit and set their boxes (log-scaled where marked). Unticked values are fixed.
* `Objective` evaluates the table values; `Sensitivity` sweeps every ticked parameter across its box (one GPU batch);
  `Fit` = Latin hypercube, then CMA-ES (population batches, tolerant of a noisy objective) or Nelder–Mead; `Validate` replays the held-out tests at the best values. `Synthetic data…` replaces the curves by the model at
  the table values (+ noise) — fit afterwards to see whether the tests identify the parameters.
* **Chaos replicas** (replay protocol): the stick–slip replay is chaotic, so one replay is one sample of a noisy
  objective. With replicas = 4–8 each test is replayed from slightly perturbed copies and the residuals are averaged
  (costs replicas × the instances — cheap on the GPU).
* `Export parameter set` saves the fitted values with units, bounds, residuals and the section they were calibrated
  on. Open any other scene (another topology, another mesh) and `Import & apply…`: units are converted, the loading-pad
  stiffness (test rig) stays out, and a different lath section is flagged.

## 5 · Check your GPU

<http://localhost:8090/web/tests/selftest.html> → `Run`: GPU against CPU on the validation cases, reproducibility, and
the batched throughput (instances × iterations per second for 1 … 64 instances). `Copy report` puts the text on the
clipboard.
