"""Compare the JS and Python Rhino exports (ref_out/export_js_<case>.3dm vs export_py_<case>.3dm): units, the layer tree
(full paths, colours), and every object matched by (layer, name, occurrence): type, colour, user text and geometry."""
import os, sys, json
from collections import defaultdict
from pathlib import Path as _P
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))
import numpy as np
import rhino3dm as r

def key_objects(f):
    layers = {i: f.Layers[i].FullPath for i in range(len(f.Layers))}
    out, seen = {}, defaultdict(int)
    for o in f.Objects:
        a = o.Attributes
        k = (layers[a.LayerIndex], a.Name)
        out[k + (seen[k],)] = o
        seen[k] += 1
    return out, layers

def geom(g):
    if isinstance(g, r.PolylineCurve):
        return 'pl', np.array([[g.Point(i).X, g.Point(i).Y, g.Point(i).Z] for i in range(g.PointCount)])
    if isinstance(g, r.NurbsCurve):
        P = np.array([[g.Points[i].X, g.Points[i].Y, g.Points[i].Z, g.Points[i].W] for i in range(len(g.Points))])
        K = np.array([g.Knots[i] for i in range(len(g.Knots))])
        return 'nc', (g.Degree, P, K)
    if isinstance(g, r.Mesh):
        V = np.array([[g.Vertices[i].X, g.Vertices[i].Y, g.Vertices[i].Z] for i in range(len(g.Vertices))])
        F = np.array([list(g.Faces[i])[:3] for i in range(len(g.Faces))])
        C = np.array([list(g.VertexColors[i])[:3] for i in range(len(g.VertexColors))]) if len(g.VertexColors) else None
        return 'mesh', (V, F, C)
    if isinstance(g, r.LineCurve):
        return 'line', np.array([[g.PointAtStart.X, g.PointAtStart.Y, g.PointAtStart.Z], [g.PointAtEnd.X, g.PointAtEnd.Y, g.PointAtEnd.Z]])
    if isinstance(g, r.Point):
        return 'pt', np.array([g.Location.X, g.Location.Y, g.Location.Z])
    if isinstance(g, r.TextDot):
        return 'dot', (g.Text, np.array([g.Point.X, g.Point.Y, g.Point.Z]))
    return type(g).__name__, None

fails = 0
for case in sys.argv[1:] or ['v5dome_cm', 'dome_basic']:
    fp, fj = r.File3dm.Read(str(REF / f'export_py_{case}.3dm')), r.File3dm.Read(str(REF / f'export_js_{case}.3dm'))
    op, lp = key_objects(fp)
    oj, lj = key_objects(fj)
    errs = []
    if fp.Settings.ModelUnitSystem != fj.Settings.ModelUnitSystem: errs.append('units')
    lcp = {fp.Layers[i].FullPath: tuple(fp.Layers[i].Color) for i in range(len(fp.Layers))}
    lcj = {fj.Layers[i].FullPath: tuple(fj.Layers[i].Color) for i in range(len(fj.Layers))}
    if set(lcp) != set(lcj): errs.append(f'layers only py {sorted(set(lcp) - set(lcj))[:5]} only js {sorted(set(lcj) - set(lcp))[:5]}')
    for k in set(lcp) & set(lcj):
        if lcp[k] != lcj[k]: errs.append(f'layer colour {k} {lcp[k]} vs {lcj[k]}')
    if set(op) != set(oj):
        errs.append(f'objects only py {len(set(op) - set(oj))} {sorted(set(op) - set(oj))[:4]} only js {len(set(oj) - set(op))} {sorted(set(oj) - set(op))[:4]}')
    worst = defaultdict(float)
    ustr = 0
    for k in sorted(set(op) & set(oj)):
        a, b = op[k], oj[k]
        A, B = a.Attributes, b.Attributes
        if tuple(A.ObjectColor) != tuple(B.ObjectColor) or A.ColorSource != B.ColorSource: errs.append(f'{k} colour {A.ObjectColor} vs {B.ObjectColor}')
        ua, ub = dict(A.GetUserStrings()), dict(B.GetUserStrings())
        if set(ua) != set(ub): errs.append(f'{k} user keys {sorted(ua)} vs {sorted(ub)}')
        for uk in set(ua) & set(ub):
            if ua[uk] != ub[uk]:
                try:
                    x, y = float(ua[uk]), float(ub[uk])
                    rel = abs(x - y) / max(abs(x), 1e-9)
                    worst['user ' + uk] = max(worst['user ' + uk], rel)
                    if rel > 1e-6: errs.append(f'{k} user {uk} {ua[uk]} vs {ub[uk]}')
                except ValueError:
                    errs.append(f'{k} user {uk} {ua[uk]!r} vs {ub[uk]!r}')
                ustr += 1
        ta, ga = geom(a.Geometry); tb, gb = geom(b.Geometry)
        if ta != tb: errs.append(f'{k} type {ta} vs {tb}'); continue
        if ta in ('pl', 'line', 'pt'):
            if np.shape(ga) != np.shape(gb): errs.append(f'{k} shape'); continue
            worst[ta] = max(worst[ta], float(np.abs(ga - gb).max()))
        elif ta == 'nc':
            if ga[0] != gb[0] or ga[1].shape != gb[1].shape: errs.append(f'{k} nurbs shape'); continue
            worst['nurbs points'] = max(worst['nurbs points'], float(np.abs(ga[1] - gb[1]).max()))
            worst['nurbs knots'] = max(worst['nurbs knots'], float(np.abs(ga[2] - gb[2]).max()))
        elif ta == 'mesh':
            (Va, Fa, Ca), (Vb, Fb, Cb) = ga, gb
            if Va.shape != Vb.shape or Fa.shape != Fb.shape or not (Fa == Fb).all(): errs.append(f'{k} mesh topology {Va.shape} {Vb.shape}'); continue
            worst['mesh vertices'] = max(worst['mesh vertices'], float(np.abs(Va - Vb).max()))
            if (Ca is None) != (Cb is None): errs.append(f'{k} vertex colours present'); continue
            if Ca is not None:
                d = int(np.abs(Ca - Cb).max()); worst['vertex colour'] = max(worst['vertex colour'], d)
                if d > 1: errs.append(f'{k} vertex colours differ by {d}')
        elif ta == 'dot':
            if ga[0] != gb[0]: errs.append(f'{k} dot text {ga[0]!r} vs {gb[0]!r}')
            worst['dot'] = max(worst['dot'], float(np.abs(ga[1] - gb[1]).max()))
    for k, v in worst.items():
        if not k.startswith('user') and k != 'vertex colour' and v > 1e-6: errs.append(f'{k} max diff {v:.3g}')
    print(f'{case}: {len(op)} / {len(oj)} objects · {len(lcp)} layers · worst ' + ', '.join(f'{k} {v:.2g}' for k, v in sorted(worst.items())) + (f' · {ustr} user strings differ in the last digits' if ustr else ''))
    for e in errs[:25]: print('  ', e)
    if len(errs) > 25: print(f'   … {len(errs) - 25} more')
    fails += len(errs)
    ip, ij = json.loads((REF / f'export_py_{case}.3dm.json').read_text()), json.loads((REF / f'export_js_{case}.3dm.json').read_text())
    for k in ('units', 'hybrid', 'strand_groups'):
        if ip[k] != ij[k]: print('   index', k, 'differs'); fails += 1
    if len(ip['objects']) != len(ij['objects']): print('   index object counts', len(ip['objects']), len(ij['objects'])); fails += 1
print('FAILED' if fails else 'ok')
sys.exit(1 if fails else 0)
