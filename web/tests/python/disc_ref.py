"""Reference for the discretisation, diagram node values and joint helpers of the JS port (tests/disc_check.mjs):
a short Python solve of an example, then the Discretizer's components / normalisation / energy on that state."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))   # the WeaveSim folder
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))
REF.mkdir(parents=True, exist_ok=True)
import json, sys, types
import numpy as np
sys.path.insert(0, str(ROOT))
for name in ('polyscope', 'polyscope.imgui', 'polyscope.implot'):     # the viewer package imports polyscope; stub it
    sys.modules.setdefault(name, types.ModuleType(name))
from weavesim import Mesh
from weavesim.scene import Scene
from weavesim.solver import Solver
from weavesim.discretize import Discretizer, DiscretizeParams, geometry_from_app, discrete_curvature
from weavesim.export3dm import _node_values

EX = ROOT / 'examples'
out = {}
for case, iters in (('v5dome_cm', 300), ('dualv5dome_cm_hybrid', 200)):
    sc = Scene.load(EX / 'scenes' / f'{case}.json')
    mesh = Mesh.load(sc.mesh_file, scale=sc.mesh_scale)
    topo = sc.build_topology(mesh)
    sv = Solver(sc, topo)
    sv.step(iters)
    r = sv.results()
    X = sv.state.X
    L = np.array([np.linalg.norm(np.diff(P, axis=0), axis=1).sum() for P in X])
    params = DiscretizeParams(max_length=float(np.median(L) * 0.45), joint_length=0.6 * sc.section.width, extra_joints=1,
                              weights={k: 1.0 for k in ['sum_curve', 'max_curve', 'min_dist', 'uniformity_range', 'uniformity_sum', 'structural', 'clearance']},
                              structural_layer='Mx', seed=3)
    geo = geometry_from_app(topo, X, r, 'Mx')
    width = max(sc.group(g).section.width for g in range(1, sc.n_groups + 1))
    d = Discretizer(geo, params, width=width)
    rng = np.random.default_rng(7)
    x_test = []
    for xi, l in zip(d.x, d.L):
        y = np.clip(xi + rng.uniform(-0.2, 0.2, len(xi)) * (l / (len(xi) + 1)), 0.01, l - 0.01)
        x_test.append(np.sort(y))
    nodes = {}
    for lay, kw in (('Mx', {}), ('N', {}), ('disp', {}), ('Fn', {}), ('Vy', {'shear': True})):
        src = 'Mx' if lay == 'Vy' else lay
        nodes[lay] = [_node_values(r, topo, src, s, len(X[s]), shear=kw.get('shear', False), X=X[s]).tolist() for s in range(topo.n_strands)]
    out[case] = {
        'iterations': iters, 'closed': [bool(s.closed) for s in topo.strands],
        'X': [P.ravel().tolist() for P in X],
        'segment': {k: [a.tolist() for a in v] for k, v in r.segment.items()},
        'disp': [a.ravel().tolist() for a in r.node['disp']],
        'Fn': {str(e): v for e, v in r.knot['Fn'].items()},
        'params': params.to_dict(), 'width': width,
        'strand_ids': d.strand_ids, 'L': d.L.tolist(), 'n_joints': d.n_joints,
        'kappa': [g.kappa.tolist() for g in geo], 'knot_s': [g.knot_s.tolist() for g in geo],
        'layer': [g.layer.tolist() if g.layer is not None else None for g in d.geo],
        'components0': d.components0, 'normalised0': d.normalised(d.components0), 'energy0': d.history_best[0],
        'x_test': [a.tolist() for a in x_test], 'components_test': d.components(x_test), 'normalised_test': d.normalised(d.components(x_test)),
        'energy_test': float(sum(d.p.weights.get(k, 0.0) * v for k, v in d.normalised(d.components(x_test)).items())),
        'pieces_test': [a.tolist() for a in d.pieces(x_test)],
        'feasible_test': [bool(d.feasible(xi, l)) for xi, l in zip(x_test, d.L)],
        'curv': [discrete_curvature(P).tolist() for P in X],
        'nodes': nodes,
    }
    print(case, 'strands cut', len(d.geo), 'joints', sum(d.n_joints), 'energy0', d.history_best[0], 'components0', d.components0)
(REF / 'disc_ref.json').write_text(json.dumps(out))
print('written', REF / 'disc_ref.json')
