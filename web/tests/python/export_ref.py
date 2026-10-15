"""Reference Rhino export of the Python program (weavesim/export3dm.py) for tests/export_check.mjs: a short solve of an
example with joints, the .3dm written by Python, and the state the JS export starts from (positions, frames, contact
midpoints and normals, results)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))
REF.mkdir(parents=True, exist_ok=True)
import json, sys, types
import numpy as np
sys.path.insert(0, str(ROOT))
for name in ('polyscope', 'polyscope.imgui', 'polyscope.implot'):
    sys.modules.setdefault(name, types.ModuleType(name))
from weavesim import Mesh
from weavesim.scene import Scene
from weavesim.solver import Solver
from weavesim.discretize import Discretizer, geometry_from_app
from weavesim.export3dm import export_3dm, ExportOptions

EX = ROOT / 'examples'
out = {}
for case, iters in (('v5dome_cm', 300), ('dome_basic', 200)):
    sc = Scene.load(EX / 'scenes' / f'{case}.json')
    mesh = Mesh.load(sc.mesh_file, scale=sc.mesh_scale)
    topo = sc.build_topology(mesh)
    sv = Solver(sc, topo)
    sv.step(iters)
    r = sv.results()
    # joints from a short annealing on the solved state, a point load in the frame axes for the local branch
    L = np.array([np.linalg.norm(np.diff(P, axis=0), axis=1).sum() for P in sv.state.X])
    sc.discretize.max_length = float(np.median(L) * 0.45)
    d = Discretizer(geometry_from_app(topo, sv.state.X, r, 'Mx'), sc.discretize, width=sc.section.width)
    d.step(20)
    sc.joints = d.result().joints
    f0 = next(iter(topo.frame_to_knot))
    sc.add_point_load(f0, [0.0, -3.0, 1.0], space='local')
    path = REF / f'export_py_{case}.3dm'
    export_3dm(sc, topo, path, state=sv.state, results=r, options=ExportOptions(), mesh=mesh, contact=sv.contact)
    st = sv.state
    out[case] = {
        'scene': sc.to_dict(), 'X': st.Xf.ravel().tolist(), 'Q': st.Q.ravel().tolist(), 'X0': st.X0f.ravel().tolist(), 'nPrev': st.n_prev.ravel().tolist(),
        'mid': np.asarray(sv.contact.mid).ravel().tolist(), 'knot_edges': [int(e) for e in st.knot_edges],
        'results': r.to_dict(), 'file': path.name,
    }
    print(case, 'written', path.name, 'joints', sum(len(v) for v in sc.joints.values()))
(REF / 'export_ref.json').write_text(json.dumps(out))
