# WeaveSim Web — checks

Paths: `paths.mjs` (the WeaveSim folder = two levels up, override with `WEAVESIM_ROOT`; Python reference dumps in
`tests/ref_out`, override with `WEAVESIM_REF`).

## In the browser (open from the running server, `python web/serve.py`)

| page | what it checks |
|---|---|
| `selftest.html` | GPU against CPU f64 on the validation cases, bit-reproducibility, batched throughput on your GPU — the one to run first |
| `gpu_check.html?cases=…&iters=…` | the validation cases one by one (cantilever, edgewise, torsion, crossings stick / slip / locked / W2-W3 / AL, scenes) |
| `gpu_trace.html?case=…` | CPU and GPU states side by side at checkpoints (where a divergence starts) |
| `stats_check.html?case=…&iters=…` | aggregate statistics CPU f64 vs a perturbed CPU run vs GPU (chaos vs error) |
| `calib_det.html` | a calibration candidate gives bit-identical replays in any GPU batch; `?replicas=3` exercises the chaos replicas |
| `calib_gpu.html`, `calib_gpu_p1.html` | the force example's replay on the GPU against CPU and Python (slow on a software GPU) |

## Node (CPU, f64)

```bash
python web/tests/python/dump_topology.py      # Python reference dumps → tests/ref_out (needs the weavesim environment)
python web/tests/python/dump_solver.py
python web/tests/python/calib_ref.py
node web/tests/cmp_topology.mjs               # topology: identical to Python (12 cases)
node web/tests/cmp_solver.mjs [case]          # solver states against Python at checkpoints
node web/tests/calib_cpu.mjs                  # force-mode replay against Python (example at the synthetic truth)
node web/tests/calib_disp.mjs                 # displacement-mode replay (locked, pad) against Python
node web/tests/paramset.mjs                   # parameter sets: units, transfer warnings
node web/tests/calib_chaos.mjs 1e-9 8 relax   # objective spread of chaos replicas
node web/tests/relax_check.mjs                # residual of the example's relaxation over 12 000 iterations
node web/tests/slip_watch.mjs                 # which crossings keep slipping during that relaxation
python web/tests/python/disc_ref.py           # discretisation reference (solved state, components, energy)
node web/tests/disc_check.mjs                 # discretisation / diagram node values / joints against it
python web/tests/python/export_ref.py         # Python Rhino export of a solved state (needs pip install rhino3dm)
node web/tests/export_check.mjs               # the JS export of the same state
python web/tests/python/cmp_3dm.py            # object-by-object comparison of the two .3dm files
```

## Headless browser runs (optional; `npm i -D playwright`)

`run_browser.mjs <page> [query]` runs one of the pages above headless and prints its console; `ui_check.mjs` and
`ui_scenarios.mjs` drive the app (load, run, layers, picking, backend switch, lath groups on the GPU, fastened protocol (Zero →
settle → Lock, stop, the Lock toggle and the 4 Contact box),
live edits, units, flips, save, UTM replay, calibration objective and fit) and take screenshots into `ref_out/ui`;
`ui_protocol.mjs [out]` runs the fastened-structure protocol on WebGPU (settle, lock on host and GPU, the settled state
as the reference on both, a load, the phase bands; strip screenshots in light and dark);
`ui_display.mjs [out] [cpu|webgpu]` checks the display features (zoom redraw, shadow, smooth / banded / per-segment
colours, the diagrams and labels, discretisation, pieces, joint zones, lath selection, Rhino download, sidebar fold /
dock).
