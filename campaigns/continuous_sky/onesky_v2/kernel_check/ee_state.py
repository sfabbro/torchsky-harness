import sys
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/claude/work/kernel_check")
import kc_common as K
K.STATE = sys.argv[1]
import q1_kernel as Q
fit = K.Fit()
for f in range(6):
    es = np.nonzero(fit.e_filt == f)[0]
    dets, cnt = np.unique(fit.e_det[es], return_counts=True); det = int(dets[np.argmax(cnt)])
    e_rot = int(es[fit.e_det[es] == det][0]); lat = fit.ref_lattice_xi(det).numpy()
    try:
        o = Q.measure(fit, f, det, lat[4], e_rot, full=False)
        o = o[0] if isinstance(o, tuple) else o
        ee = o["encircled_about_origin_levels_norm"]
        print(K.FILTERS[f], "kernel fraction inside r =", {k: round(v, 3) for k, v in ee.items()})
    except Exception as e:
        print(K.FILTERS[f], "failed:", repr(e)[:200])
