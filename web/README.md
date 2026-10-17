# WeaveSim Web

WeaveSim in the browser: the same scenes, topology and solver as the Python program, with a batched WebGPU solver for
large weaves and calibration populations, the CPU solver in double precision as the reference, and a calibration sandbox
that fits model parameters to UTM load tests and exports them as a parameter set for other weave topologies.

Start: see [HOW_TO_START.md](HOW_TO_START.md) — `python web/serve.py`, then <http://localhost:8090/web/>.

## What is in it

| file | what it does |
|---|---|
| `index.html`, `style.css` | the app: status bar, solve / discretisation strip (pinned), sections 1 File … 10 Calibration, 3-D view with the navigation cube, the colour legend and the value labels, the Display sidebar (folds to a tab, docks left or right) — the idiom of the WeaveSim viewer, LocalSH and the Joint FEA apps |
| `theory.html` | how the solver works, for reviewers and for readers without the maths (opened from *11 Theory* at the end of the menu): the Cosserat and Kirchhoff rod, second-order theory, the sliding contact point and reverse kinematics, the section-aware gap, friction, embedment, dynamic relaxation and its safeguards (the 17 Oct 2026 multiplier fix, pass-through) — every equation with its code location, interactive figures, light and dark, a plain-words reading mode; one self-contained page, KaTeX from `vendor/katex` |
| `js/app.js` | scene / model management, panels, solve loop, picking (frames; whole laths for the lath groups), UTM replay, Rhino export |
| `js/display_ui.js` | the Display sidebar: lath colouring and legend, member diagrams, joints overlay |
| `js/disc_ui.js` | the discretisation: strip tab, section 8, plots, joints csv |
| `js/protocol_ui.js` | the load-test protocol under Run: Zero → settle → pause (the 4 Contact box locks), the phase bands of the strip chart |
| `js/calib_ui.js` | the calibration panel |
| `js/view.js` | three.js view: laths as swept sections on the current state (colour-map texture per fragment: exact gradients), centrelines, crossings, loads, anchors, probes, selection, diagrams, joints, lath selection, soft ground shadow |
| `js/diagrams.js` | member diagrams (diagrams.py): Mx My Vy Vx T N curvature strips, deflected shape, contact forces, reactions, labels |
| `js/discretize.js` | joint placement by simulated annealing (discretize.py) and the joint helpers (joints.py) |
| `js/fields.js` | node values of the result layers, shear, arc length, curvature (shared by colours, diagrams, joints, export) |
| `js/export3dm.js` | Rhino .3dm export (export3dm.py) with rhino3dm.js (`vendor/rhino3dm`, loaded on the first export) |
| `js/scene.js`, `params.js`, `units.js` | `scene.json` (version 1) read and written exactly as the Python program; unit conversion |
| `js/mesh.js`, `topology.js`, `pyset.js` | STL / OBJ, the kagome tracer — knot, strand and frame ids identical to the Python program (CPython set-iteration order emulated for the family colouring) |
| `js/model.js`, `quat.js` | solver state, selectors, anchors, loads, section stiffness of the lath groups |
| `js/solver_cpu.js` | the solver in f64: ActiveBeam (discrete elastic rod: stretch, anisotropic bending, twist, clamps, activation), PolylineContact (global parameter u, Hermite centreline, section-aware gap, lever-arm torque, Coulomb stick / slip, lock, W1 contact type, W2 rotational friction, W3 seat law, augmented Lagrangian), dynamic relaxation with kinetic damping |
| `js/wgsl.js`, `js/solver_gpu.js` | the same solver as WGSL kernels, B instances per dispatch (see *Precision* and *Batching*) |
| `js/utm.js`, `js/calibrate.js` | UTM csv import, the replay protocols, objective, sensitivity, fit (Latin hypercube + Nelder–Mead or CMA-ES), validation, synthetic data, parameter sets |
| `js/plot.js`, `js/colormap.js`, `js/navcube.js`, `js/rng.js` | plots (lines, dots, bars), colour maps and palettes (the Python viewer's: matplotlib maps, family / group / over-under colours), the view cube (from LocalSH), seeded random numbers |
| `vendor/` | three.js r170 (+ OrbitControls, wide lines), rhino3dm.js 8.17 (MIT, McNeel), KaTeX 0.16.47 (MIT; woff2 fonts only, for `theory.html`) |
| `tests/` | Node checks against the Python reference, browser checks of the GPU solver, `selftest.html` (precision, reproducibility and throughput on your GPU) |

## Precision

WebGPU has no f64. The GPU solver keeps everything that accumulates or cancels in 64-bit integers and does the local
arithmetic in f32:

* node positions are 64-bit fixed point (2^-PB model units, PB chosen from the model size: 2^-50 cm ≈ 1e-15 cm for a
  2 m dome); the velocity is added exactly, and every edge or stencil vector is an exact integer difference converted
  to f32 once — the f32 error is relative to the local size (a segment, a crossing), never to the coordinate;
* segment frames (quaternions) are 64-bit fixed point (2^-60) updated in delta form: a frame that does not move does
  not change by a bit (f32 frames re-rounded every iteration inject ~1e-7 rad of roll noise that the stiff twist mode
  turns into kinetic energy, which trips the kinetic-damping reset and stalls convergence);
* axial strain is computed against a per-segment reference edge (exact integer + f32 copy + f64 offset), so
  `L − l0` never cancels;
* contact parameters u are 64-bit fixed point (segment index | 32-bit fraction), slips add exactly;
* forces, torques and the contact twist stiffness are accumulated with 64-bit fixed-point atomics: sums are exact and
  independent of the thread order, so a run is **bit-for-bit reproducible** on a given GPU, and a calibration candidate
  gives the same objective in any batch;
* the UTM read-outs (probe travel, reaction) are summed in 64-bit fixed point over the averaging window;
* WGSL's built-in `sin`, `cos`, `atan2` may be very inaccurate (the spec allows 2^-11 absolute for sin/cos and 4096 ULP
  for atan2; SwiftShader uses that latitude): the solver uses its own polynomial versions accurate to ~1 ULP.

Validation (Node and headless Chromium with SwiftShader, `tests/`):

* topology: 12 cases identical to the Python program (strands, knots, frames, families, offsets);
* CPU f64 against the Python solver: agreement to rounding (1e-15 relative) until the stick–slip dynamics amplify the
  rounding — that divergence is in the Python program itself (two Python runs 1e-15 apart separate the same way);
* GPU against CPU f64: beams to ~1e-7 relative (cantilever, edgewise, torsion), single crossings (stick, slip, locked,
  W2/W3) to f32 level, the v5 dome tracks f64 to 1e-5 through 1500 iterations; later the woven scenes separate by the
  same chaos (aggregates within 1 %);
* contact floor: with a very stiff contact penalty the f32 contact forces carry noise of about k·w·6e-8 per crossing,
  which can hold the relative residual above a 1e-4 tolerance (single crossing with k = 1e4 N/mm: 1.8e-4). **f64 polish**
  (section 7) continues on the CPU from the GPU state;
* UTM replay: the JS replay reproduces the Python replay to 4 decimals (all 16 increments of P1 in the force example),
  6× faster on the CPU (23 s vs 136 s per objective evaluation); the GPU replay from the same relaxed state reproduces
  the CPU replay to 4 decimals through the first load cycle.

## Batching (GPU)

One solver holds B instances of one topology (same nodes, segments, crossings, anchors) with their own parameters and
state; every kernel is dispatched once per iteration for all of them (grid y = instance), six dispatches per iteration.
The interactive model is B = 1. A calibration batch is B = parameter vectors × tests × replicas: a Latin-hypercube
screening, a CMA-ES generation or a whole sensitivity sweep is one run. The fixed-point scales of a calibration are set
once from the corners of the parameter box, so results do not depend on the batch composition. Real-GPU throughput
could not be measured where this was written (software GPU only) — `tests/selftest.html` measures it on your machine.

Within a run, iterations go to the GPU in command buffers of 64 iterations; WebGPU exposes one queue per device, so an
interactive run and a calibration share it and the driver interleaves them.

## Display, diagrams, joints, Rhino (the Python viewer's Results panel, Discretisation bar and export)

* **Colours.** Result layers are drawn with a colour-map texture looked up per fragment from a scalar per lath vertex:
  *smooth* (default) interpolates the node values along each lath (segment values averaged to the nodes, as the Python
  ribbons and the Rhino export do), off gives one colour per segment (the Python sections), *bands* quantises to 10
  steps. All colours are converted from sRGB to linear before WebGL lights them, so a colour is the Python viewer's
  colour (the first version handed sRGB values to WebGL as linear: pale, washed-out gradients). The result palettes and
  the lath colourings pure colour, over / under, pieces and lath group are those of `weavesim/app/theme.py` and
  `viewer.py`; the lath families use Paul Tol's high-contrast scheme as LocalSH (#004488 deep blue, #DDAA33 warm gold,
  #BB5566 muted crimson), then his light scheme for further families (#EE8866, #EEDD88, #FFAABB, #77AADD, #99DDFF,
  #44BB99), lifted 18 % toward white on the dark canvas; "by strand" uses his muted scheme. The legend lists the
  families / groups with their strand counts. Crossing layers colour the crossings and fade along the adjacent segments.
* **Hover probe** (the Joint FEA idiom): a small window at the cursor with the lath (family, group), the position s
  along it, the value of the colour layer and of the diagram at that point — interpolated exactly as drawn — or the
  section forces Mx My T N when neither is shown; next to a crossing also its contact state (Fn, Ft, stick / slip, gap,
  slip, contact type).
* **Load-test protocol** (strip, under Run; Python: `1 Internalize`): `Zero`
  takes the relaxed geometry as the reference (X0 = X, slip 0) and settles for the given iterations, then takes the
  settled state as the reference again and pauses (Run → Pause), so Results › *delta*
  after the load is the load displacement alone; the drift during the settle is reported. The crossings are not
  touched: fastened joints are the *lock all crossings* box in 4 Contact (17 Oct 2026 — the strip's `Lock` button was
  removed; it doubled the box and read like a solver freeze). The strip chart shades the phases in the
  colour-annealing idiom — grey *relax*, sand *zero* with a
  dashed line at the planned end, blue *locked* while the box is ticked — and keeps the latest Zero in view. On the GPU the reference
  copy goes through a scratch buffer (WebGPU rejects a copy within one buffer; until 15 Oct the GPU kept its old
  reference, so delta on WebGPU was measured from the earlier state).
* **Diagrams** (Display › Diagram): Mx, My, Vy, Vx, T, N and the centreline curvature as filled, hatched strips along
  the frame axis the quantity acts in (height 6 % of the model at |max| × scale), the deflected shape (undeformed grey,
  deformed coloured by |u|, exaggerated to 5 % of the model), contact normal forces and reactions as arrows, value labels
  at the extremes or per lath, *delta* against the internalised state, and — with joints — *around joints only*
  (±n crossings). Two deliberate differences from `diagrams.py`: the strip is split where the value changes sign (each
  part one colour), and closed strands get a diagram (Python leaves them at zero).
* **Discretisation** (strip › Discretisation, section 8): `discretize.py` ported line by line — the same strands cut,
  joint counts, curvature, structural layer, objective components, normalisation and energy as Python on the same
  state (`tests/disc_check.mjs`: 3e-15). The annealing evaluates a neighbour incrementally (only the moved strand's
  terms and its row of strand-pair distances; 50× faster, same objective); its random numbers are xoshiro128**, so a
  seed gives another (equally valid) path than numpy. The joints go to `scene.joints` (saved with the scene and read by
  the Python program), show as cut rectangles on the laths and drive *pieces* and *around joints only*.
* **Rhino** (File › Export Rhino .3dm): `export3dm.py` ported with rhino3dm.js — the same layer tree, object names,
  user text (formatted as Python's `str()`), meshes, interpolating NURBS (collocation solve, de Boor knots), vertex
  colours, diagrams, forces, joints, supports and loads, and the `.3dm.json` index. On the same state the JS file
  matches the Python file object by object (`tests/export_check.mjs` + `tests/python/cmp_3dm.py`: 1001 and 2837
  objects, geometry to 1e-12, identical vertex colours and user text up to the last digit of some floats).
* **Lath groups** (3 Units & material): `Select laths…` picks whole laths (LocalSH: the selected laths in the selection
  colour with a dark line along them, the others faded; a click on anything else clears, leaving the mode clears);
  `Assign … to group n` rebuilds the model with the group's section, as the Python program, and shows the new group
  colours at once.

## Calibration sandbox (section 10)

1. **Tests**: a probe crossing (frame), the load axis, the UTM csv, a weight, training or validation. Load a `calib.json`
   of the Python program with its files (or a folder), start from the force example, or from the scene in the view
   (pick crossings, load the csv files).
2. **Parameters**: tick what is fitted, set the box (log-scaled where marked); unticked values are fixed.
3. **Protocol**: force mode is the lab cable pull (n_steps up to the peak force and back, cycles times; crosshead
   travel = probe travel + F / pad_stiffness); displacement mode drives the probe.
4. **Objective / Sensitivity / Fit / Validate**: fit = Latin hypercube, then Nelder–Mead (the Python fit, same steps and
   termination; on the GPU the candidate points of an iteration run in one batch) or CMA-ES (each generation one batch).
5. **Chaos replicas**: the replay is chaotic (stick–slip); with replicas > 1 every test is replayed from slightly
   perturbed copies (1e-9 of the model size) and the residuals are averaged; the spread is shown. See *Findings*.
6. **Parameter set**: export the fitted values with units, classes (material / contact / model / test rig), bounds,
   residuals and the section they were calibrated on; *Import & apply* converts units and writes them into any scene —
   the transfer to other topologies. The loading-pad stiffness belongs to the test rig and is not transferred.

## Findings while porting (worth knowing for the calibration)

* **The force example's relaxation does not converge.** Over 12 000 iterations the residual stays between 1e-2 and 0.7
  (tolerance 1e-4); between iterations 2000 and 4000 there are ~100 kinetic-damping resets, 111 of 130 crossings slip at
  some point and a node on strand 11 (P1's strand) travels 0.7 cm — as much as the measured probe travel. The replays
  start from a creeping state. Same in the Python program (the port reproduces it).
* **The objective is noisy.** At the synthetic truth, 8 replicas whose woven state was perturbed by 1e-9 of the model
  size give objectives 0.13 … 0.21 (mean 0.170, σ 0.030; P2 0.014 … 0.066, P3 0.011 … 0.067). A 1e-6 difference of
  the relaxed state (one crossing stuck instead of slipping) moves P1's travel at 450 N by 4–14 %. CPU f64, GPU and
  Python land on different, equally valid samples (objective 0.13, 0.24 and 0.15 at the truth); from the *same* relaxed
  state the GPU replay reproduces the CPU replay to 4 decimals through the first cycle. Use replicas for a defined
  objective; CMA-ES tolerates the noise better than Nelder–Mead.
* `make_synthetic` (Python) closes a force-mode trace with `dd[-1][-1:]`, i.e. it repeats the last *interpolated* sample
  rather than the end of the last branch (probably meant `d[-1:]`); the web version does the same, so synthetic files
  agree. `fit` starts Nelder–Mead from `np.argmin` of the screening scores, which picks a NaN score if there is one; the
  web version skips NaN.
* In the Python sources on disk the augmented-Lagrangian gap (README, panels, `tree_cm.json`) is missing from
  `params.py` / `contact.py`; the web version implements it as the README describes (`augmented`, `al_every`).
* 17 Oct 2026 — multiplier safeguards and roll rate limit (`solver_cpu.js` `PolylineContact.alValue`,
  `TWIST_RATE_MAX`; `wgsl.js` contact kernel / `k_segint`; `solver_gpu.js` mirrors the update in `download`, state field
  `penacc`): the Uzawa step uses the window-mean penetration, gives no multiplier to a crossing whose contact point sits
  on a fixed node (free weight < `AL_FREE_MIN` = 0.25) and caps |λ| at fc90·wA·wB/sinθ; the segment frames roll by at
  most SUPPORT_SMOOTH/2 per iteration (also in Python `core.py`). Before: λ at the pinned boundary knots ratcheted without
  bound (`v5dome_BugTest.json`: Fn 3.5 kN → 400 kN in 100 k iterations, laths flipped and passed through each other;
  `v5dome_cm` with AL on never converged). The Python reference dumps (`tests/python/dump_solver.py`) need regenerating
  for the roll rate limit.
* `UTMRef.from_dict` in the Python program drops `pad_stiffness`; the web version keeps it.

An independent review of the calibration port against `calibrate.py` / `utm.py` / scipy found 12 differences, all
fixed: Nelder–Mead now sorts NaN last and shrinks with scipy's arithmetic (bit-identical paths on 12 test problems),
speculative points that Nelder–Mead never uses no longer count as evaluations, the CMA-ES weights follow Hansen's
tutorial for odd λ, csv cells `nan` / `inf` parse as in Python, long curves no longer overflow the stack, a sensitivity
sweep with one value uses the lower bound (np.linspace), a missing csv loads as "no curve", `relax_iter` 0 falls back to
`max_iter` on the GPU too, and empty load schedules are handled.

## Differences from the Python program

* region selections use graph distance on the mesh edges (the Python program uses the heat method when potpourri3d is
  installed); the report (`app/report.py`) is not ported;
* random numbers (Latin hypercube, synthetic noise, the discretisation's annealing) come from a seeded xoshiro128**,
  not numpy — the same seed gives other samples;
* diagrams split at sign changes and are drawn on closed strands too (see *Display*);
* Nelder–Mead reuses evaluations it already has (the Python fit evaluates the best LHS point a second time).

## Tests

```bash
node web/tests/cmp_topology.mjs        # topology vs the Python dumps (needs the reference dumps, see tests/)
node web/tests/cmp_solver.mjs          # CPU f64 solver vs Python
node web/tests/calib_cpu.mjs           # calibration replay vs Python
node web/tests/disc_check.mjs          # discretisation, diagram node values, joint helpers vs Python
node web/tests/export_check.mjs        # Rhino export on the Python state → python web/tests/python/cmp_3dm.py compares
```

`tests/selftest.html` (open it from the running server) checks the GPU against the CPU, reproducibility and
throughput on your GPU.
