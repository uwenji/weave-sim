"""Reference: Python objective + replay curves at the synthetic truth (force mode), for the JS/GPU port."""
import os
from pathlib import Path as _P
ROOT = _P(os.environ.get('WEAVESIM_ROOT', _P(__file__).resolve().parents[3]))   # the WeaveSim folder
REF = _P(os.environ.get('WEAVESIM_REF', _P(__file__).resolve().parents[1] / 'ref_out'))   # where the JS tests look
REF.mkdir(parents=True, exist_ok=True)
import json, sys, time
sys.path.insert(0, str(ROOT))
from weavesim.calibrate import CalibrationSet
import numpy as np
cs = CalibrationSet.load(ROOT / 'examples' / 'calibration' / 'force' / 'calib.json')
cs.replay.workers = 1
truth = json.load(open(ROOT / 'examples' / 'calibration' / 'force' / 'synthetic_truth.json'))
vals = {k: truth[k] for k in cs.names()}
vals['pad_stiffness'] = 0.0
t0 = time.time()
obj, res, sims = cs.objective(vals)
print('objective at truth', obj, res, '%.1f s' % (time.time() - t0), flush=True)
out = {'values': vals, 'objective': obj, 'residuals': res,
       'sims': {name: [[c, b, d.tolist(), f.tolist()] for c, b, d, f in br] for name, br in sims.items()}}
json.dump(out, open(REF / 'calib_truth.json', 'w'))
