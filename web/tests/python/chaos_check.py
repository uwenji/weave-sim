"""Is the late divergence chaos? Python vs Python with a 1e-15 relative perturbation of the initial positions."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))   # the WeaveSim folder
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))   # where the JS tests look
REF.mkdir(parents=True, exist_ok=True)
import sys, numpy as np
sys.path.insert(0, str(ROOT))
from weavesim import Mesh
from weavesim.scene import Scene
from weavesim.solver import Solver
for nm in ['dome_basic', 'dualv5dome_cm_hybrid']:
    sc = Scene.load(ROOT / 'examples' / 'scenes' / f'{nm}.json')
    mesh = Mesh.load(sc.mesh_file, scale=sc.mesh_scale); topo = sc.build_topology(mesh)
    a = Solver(sc, topo); b = Solver(sc, topo)
    b.state.Xf *= (1 + 1e-15)
    size = np.abs(a.state.Xf).max()
    for cp in [300, 1000, 1500, 2000]:
        while a.state.iteration < cp:
            a.step(1); b.step(1)
        print(nm, cp, 'dX/size %.1e' % (np.abs(a.state.Xf - b.state.Xf).max() / size), 'res %.2e %.2e' % (a.state.residual, b.state.residual), flush=True)
