"""Dump Python topology for every example (reference for the JS port)."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))   # the WeaveSim folder
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))   # where the JS tests look
REF.mkdir(parents=True, exist_ok=True)
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(ROOT))
from weavesim import Mesh, WeaveParams, build_topology
from weavesim.scene import Scene

EX = ROOT / 'examples'
OUT = REF
cases = []
for sc_path in sorted((EX / 'scenes').glob('*.json')):
    sc = Scene.load(sc_path)
    cases.append((sc_path.stem, sc.mesh_file, sc.mesh_scale, sc))
for m in ['sheet.stl', 'v5column.stl', 'dualdome.stl', 'dome.stl', 'v5dome.stl', 'tree.stl']:
    sc = Scene()
    cases.append(('mesh_' + m[:-4], EX / 'meshes' / m, 1.0, sc))
for name, mp, scale, sc in cases:
    mesh = Mesh.load(mp, scale=scale)
    topo = sc.build_topology(mesh)
    d = {
        'mesh': {'nV': int(mesh.n_vertices), 'nF': int(mesh.n_faces), 'nE': int(mesh.n_edges),
                 'V': mesh.V.ravel().tolist(), 'F': mesh.F.ravel().tolist(), 'E': mesh.E.ravel().tolist(), 'EF': mesh.EF.ravel().tolist(),
                 'edge_normals': mesh.edge_normals.ravel().tolist()},
        'strands': [{'index': s.index, 'family': s.family, 'closed': s.closed, 'edges': s.edges} for s in topo.strands],
        'knots': [{'edge': kn.edge, 'a': list(kn.a), 'b': list(kn.b), 'over': kn.over, 'conflict': kn.conflict, 'flipped': kn.flipped, 'boundary': kn.is_boundary} for kn in topo.knots.values()],
        'knot_node': [[s, k, r] for (s, k), r in topo.knot_node.items()],
        'polylines': [P.ravel().tolist() for P in topo.polylines],
        'frames': [R.transpose(0, 2, 1).reshape(-1).tolist() for R in topo.node_frames],   # per node [X | Y | Z]
        'parity_conflicts': topo.parity_conflicts, 'family_count': topo.family_count,
        'scene': sc.to_dict(),
    }
    (OUT / f'topo_{name}.json').write_text(json.dumps(d))
    print(name, 'V', mesh.n_vertices, 'F', mesh.n_faces, 'E', mesh.n_edges, 'strands', topo.n_strands, 'knots', topo.n_knots, 'conflicts', topo.parity_conflicts, 'fam', topo.family_count)
