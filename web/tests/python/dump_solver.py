"""Reference runs of the Python solver for the JS port: states after given iteration counts."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))   # the WeaveSim folder
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))   # where the JS tests look
REF.mkdir(parents=True, exist_ok=True)
import json, sys, copy
from pathlib import Path
import numpy as np
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests')); sys.path.insert(0, str(ROOT / 'examples'))
from weavesim import Mesh
from weavesim.scene import Scene, Selector
from weavesim.solver import Solver
from test_solver import straight, crossing
from make_meshes import tri_lattice
from weavesim import WeaveParams, build_topology

OUT = REF

def snap(sv):
    st = sv.state
    d = {'iteration': st.iteration, 'residual': (None if not np.isfinite(st.residual) else st.residual), 'converged': bool(sv.converged),
         'X': st.Xf.ravel().tolist(), 'Q': st.Q.ravel().tolist(), 'Wt': st.Wt.tolist(), 'V': st.Vf.ravel().tolist(),
         'ke': st.kinetic_energy[-1] if st.kinetic_energy else 0.0, 'se': st.strain_energy[-1] if st.strain_energy else 0.0,
         'N': sv.beam.N.tolist(), 'M': sv.beam.M.ravel().tolist()}
    if sv.contact is not None:
        c = sv.contact
        d.update({'u': st.u.ravel().tolist(), 'slip': st.slip.tolist(), 'Fn': c.Fn.tolist(), 'Ft': c.Ft.tolist(), 'gap': c.gap.tolist(), 'state': c.state.tolist(),
                  'ctype': c.ctype.tolist(), 'pressure': c.pressure.tolist(), 'Mrot': c.M_rot.tolist(), 'dent': c.dent.ravel().tolist(), 'dentE': c.dent_e.tolist(), 'nPrev': st.n_prev.ravel().tolist()})
    r = sv.results()
    d['res'] = {'Mx': np.concatenate(r.segment['Mx']).tolist() if r.segment['Mx'] else [], 'My': np.concatenate(r.segment['My']).tolist() if r.segment['My'] else [],
                'T': np.concatenate(r.segment['T']).tolist() if r.segment['T'] else [], 'N': np.concatenate(r.segment['N']).tolist() if r.segment['N'] else [],
                'reactions': {f'{s},{k}': v.tolist() for (s, k), v in r.reactions.items()}}
    d['hist_ke'] = st.kinetic_energy[-50:]
    return d

def run_case(name, sc, topo, checkpoints, contacts=True, mutate=None):
    sv = Solver(sc, topo, contacts=contacts)
    out = {'name': name, 'scene': sc.to_dict(), 'mass': sv.mass.tolist(), 'inertia0': sv.inertia.tolist(), 'Omega0': sv.state.Omega0.ravel().tolist(),
           'u0': sv.state.u.ravel().tolist() if sv.contact is not None else [], 'nPrev0': sv.state.n_prev.ravel().tolist(), 'Q0': sv.state.Q.ravel().tolist(), 'snaps': []}
    done = 0
    for cp in checkpoints:
        if mutate and cp == 'mutate':
            mutate(sv); out['snaps'].append({'mutate': True}); continue
        while done < cp and not (sv.converged and cp > 10**8):
            sv.step(1); done += 1
        out['snaps'].append(snap(sv))
    (OUT / f'solver_{name}.json').write_text(json.dumps(out))
    s = out['snaps'][-1]
    print(name, 'iters', s['iteration'], 'resid', s['residual'], 'conv', s['converged'], 'se', s['se'])

cps_short = [1, 2, 5, 10, 50, 100, 300, 1000, 3000]
# cantilever flatwise
L, F = 1000.0, 10.0
topo = straight(L, 10); sc = Scene(); sc.loads.gravity.enabled = False
sc.add_anchor(Selector(type="frame", id=[0, 0]), [1, 1, 1, 1, 1, 1]); sc.add_point_load((0, 1), [0, 0, -F])
run_case('cantilever', sc, topo, cps_short + [10000, 40000], contacts=False)
# edgewise
topo = straight(L, 10); sc = Scene(); sc.loads.gravity.enabled = False
sc.add_anchor(Selector(type="frame", id=[0, 0]), [1, 1, 1, 1, 1, 1]); sc.add_point_load((0, 1), [0, -F, 0])
run_case('cantilever_edge', sc, topo, cps_short + [10000], contacts=False)
# torsion held
nseg = 10; ang = np.linspace(0, np.pi / 2, nseg + 1)
normals = np.stack([np.zeros(nseg + 1), -np.sin(ang), np.cos(ang)], 1)
topo = straight(L, nseg, normals=normals); sc = Scene(); sc.loads.gravity.enabled = False
sc.add_anchor(Selector(type="frame", id=[0, 0]), [1] * 6); sc.add_anchor(Selector(type="frame", id=[0, 1]), [1, 1, 1, 0, 0, 1])
run_case('torsion', sc, topo, cps_short, contacts=False)
# crossings
sc, topo = crossing(load=-50.0, pull=0.0); run_case('cross_press', sc, topo, cps_short + [10000, 20000])
sc, topo = crossing(load=-50.0, pull=5.0); run_case('cross_stick', sc, topo, cps_short + [10000, 20000])
sc, topo = crossing(load=-50.0, pull=12.0); run_case('cross_slip', sc, topo, cps_short + [4000])
sc, topo = crossing(load=-50.0, pull=12.0); sc.contact.locked = True; run_case('cross_locked', sc, topo, cps_short + [4000])
sc, topo = crossing(load=-50.0, pull=5.0); sc.contact.rot_friction = True; sc.contact.seat_law = 'L1'; sc.contact.dent_after_iter = 0; sc.contact.slip_rate = 0.3
run_case('cross_w23', sc, topo, cps_short + [6000])
# woven sheet
V, Fc = tri_lattice(6, 5, 1200.0); m = Mesh.from_arrays(V, Fc)
topo = build_topology(m, WeaveParams(subdivisions=4), thickness=30.0)
pts, edges = topo.knot_points(); cen = pts.mean(axis=0); k = int(np.argmin(np.linalg.norm(pts[:, :2] - cen[:2], axis=1)))
kn = topo.knots[edges[k]]; over = kn.a if kn.over == kn.a[0] else kn.b
sc = Scene(); sc.loads.gravity.enabled = False
sc.add_anchor(Selector(type="all_start"), [1, 1, 1, 0, 0, 1]); sc.add_anchor(Selector(type="all_end"), [1, 1, 1, 0, 0, 1])
sc.add_point_load(over, [0, 0, -200.0])
np.save(REF / 'sheet_V.npy', V); np.save(REF / 'sheet_F.npy', Fc)
json.dump({'V': V.ravel().tolist(), 'F': Fc.ravel().tolist(), 'over': list(over)}, open(REF / 'sheet_mesh.json', 'w'))
run_case('sheet', sc, topo, cps_short + [5000])
# example scenes
for nm, iters in [('v5dome_cm', [1, 10, 100, 300, 600, 1500, 3000, 5000]), ('dome_basic', [1, 10, 100, 300, 1000, 2000]), ('dualv5dome_cm_hybrid', [1, 10, 100, 500, 1500])]:
    sc = Scene.load(ROOT / 'examples' / 'scenes' / f'{nm}.json')
    mesh = Mesh.load(sc.mesh_file, scale=sc.mesh_scale)
    topo = sc.build_topology(mesh)
    run_case(nm, sc, topo, iters)
