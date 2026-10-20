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
| `gpu_parity.html?cases=…&long=…` | GPU against CPU f64 per case: the trajectory at checkpoints next to a perturbed CPU run (the yardstick of the chaos), and the end state after a long run |
| `gpu_rod_bench.html` | the clamped cantilever and the short deep beam on the GPU against the CPU and Timoshenko |
| `gpu_diag.html` | where CPU and GPU frames differ: the segments with the largest frame difference, split into bending / shear and roll |
| `gpu_int_check.html` | the 64-bit integer helpers of the kernels against BigInt |
| `gpu_bench.html?case=…&B=1,4,16` | throughput: B instances of one scene in the same dispatches, next to the CPU |
| `calib_small.html` | the batched calibration path on a small self-contained case, CPU against GPU |
| `calib_gpu.html`, `calib_gpu_p1.html` | the force example's replay on the GPU against the CPU (`calib_gpu.html` also against a Python dump; slow on a software GPU) |

## Node (CPU, f64)

```bash
python web/tests/python/dump_topology.py      # Python reference dumps → tests/ref_out (needs the weavesim environment)
python web/tests/python/dump_solver.py
python web/tests/python/calib_ref.py
node web/tests/cmp_topology.mjs               # topology: identical to Python (12 cases)
node web/tests/cmp_solver.mjs [case]          # solver states against Python at checkpoints (*)
node web/tests/calib_cpu.mjs                  # force-mode replay against Python (example at the synthetic truth) (*)
node web/tests/calib_disp.mjs                 # displacement-mode replay (locked, pad) against Python (*)
node web/tests/paramset.mjs                   # parameter sets: units, transfer warnings
node web/tests/calib_chaos.mjs 1e-9 8 relax   # objective spread of chaos replicas
node web/tests/relax_check.mjs                # residual of the example's relaxation over 12 000 iterations
node web/tests/slip_watch.mjs                 # which crossings keep slipping during that relaxation
node web/tests/twist_watch.mjs [scene] [it] [every] [k=v,…]   # joint twist, Fn and multiplier growth over a long run (runaway check)
node web/tests/rod_check.mjs                  # Cosserat rod: forces and torques against finite differences of the energy
node web/tests/rod_bench.mjs [all|cantilever|short|beamcol|ltb|kirchhoff_limit]   # the Cosserat rod against closed forms (Timoshenko, Haringx, LTB, the shear-free limit)
node web/tests/rod_contact_check.mjs [cases|scenes]   # the Cosserat rod on the crossing cases and the example domes
node web/tests/stability_check.mjs [euler|ltb|dome]   # stability check: Euler column, lateral–torsional buckling, dome timing
python web/tests/python/disc_ref.py           # discretisation reference (solved state, components, energy)
node web/tests/disc_check.mjs                 # discretisation / diagram node values / joints against it
python web/tests/python/export_ref.py         # Python Rhino export of a solved state (needs pip install rhino3dm)
node web/tests/export_check.mjs               # the JS export of the same state
python web/tests/python/cmp_3dm.py            # object-by-object comparison of the two .3dm files
```

(*) The web solvers use the Cosserat rod since 20 Oct 2026; the Python program keeps the shear-free (Kirchhoff) rod
with frames slaved to the chords. These three checks therefore no longer agree to rounding: they remain as plumbing
checks (case set-up, replay protocol, units) and as a measure of the difference between the two rods. Topology,
discretisation and the Rhino export do not depend on the rod and still match Python exactly.

## Headless browser runs (optional; `npm i -D playwright`)

`run_browser.mjs <page> [query]` runs one of the pages above headless and prints its console; `ui_check.mjs` and
`ui_scenarios.mjs` drive the app (load, run, layers, picking, backend switch, lath groups on the GPU, fastened protocol (Zero →
settle, stop, the 4 Contact lock box),
live edits, units, flips, save, UTM replay, calibration objective and fit) and take screenshots into `ref_out/ui`;
`ui_protocol.mjs [out]` runs the fastened-structure protocol on WebGPU (settle, lock on host and GPU, the settled state
as the reference on both, a load, the phase bands; Monitor screenshots in light and dark);
`ui_display.mjs [out] [cpu|webgpu]` checks the display features (zoom redraw, shadow, smooth / banded / per-segment
colours, the diagrams and labels, discretisation in the Monitor, pieces, joint zones, lath selection, Rhino download,
sidebar fold / dock).
