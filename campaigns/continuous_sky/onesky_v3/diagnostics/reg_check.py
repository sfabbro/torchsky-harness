"""Registration of a response-only fit, checked against the joint fit: per exposure, translation (response-only) minus translation (joint).
The two frames differ by one common shift, so the median is removed; what is left is the registration error of the response-only fit."""
import sys
import collections
import numpy as np
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]
def terms(run):
    t = SkyFit.load(f"{S}/fits/{run}", device="cpu").exposure_terms
    return {k: (np.array(v["affine"][4:6]) * 1e3, v["split"], v["gain"]) for k, v in t.items()}
J = terms("x_joint_base")
def det(n): return n.split("_")[-1] if n.startswith("jw") else "hst " + n[:6]
for run, sel in (("x_h2jk_base", "jw"), ("x_h2jk_direct", "jw")):
    R = terms(run)
    names = [n for n in R if n in J and n.startswith(sel)]
    d = np.array([R[n][0] - J[n][0] for n in names])
    med = np.median(d, 0)
    r = d - med
    rad = np.hypot(r[:, 0], r[:, 1])
    print(f"\n{run}: {len(names)} exposures in both fits; common shift ({med[0]:+.1f}, {med[1]:+.1f}) mas; radial error after removing it: median {np.median(rad):.1f}, 68% {np.percentile(rad,68):.1f}, 90% {np.percentile(rad,90):.1f}, max {rad.max():.1f} mas; beyond 30 mas: {(rad>30).sum()}, beyond 100 mas: {(rad>100).sum()}")
    by = collections.defaultdict(list)
    for n, x, q in zip(names, r, rad): by[(det(n), R[n][1])].append((x[0], x[1], q))
    for k in sorted(by):
        a = np.array(by[k]); print(f"   {k[0]:12s} {k[1]:5s} n {len(a):3d}  mean ({a[:,0].mean():+7.1f}, {a[:,1].mean():+7.1f})  rms about it ({a[:,0].std():5.1f}, {a[:,1].std():5.1f})  median radial {np.median(a[:,2]):6.1f} max {a[:,2].max():6.1f}")
