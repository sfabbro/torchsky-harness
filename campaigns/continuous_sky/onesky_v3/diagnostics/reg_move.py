import sys
import re
import numpy as np
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]
def terms(run):
    return {k: np.array(v["affine"][4:6]) * 1e3 for k, v in SkyFit.load(f"{S}/fits/{run}", device="cpu").exposure_terms.items()}
J = terms("x_joint_base")
for run, sel in (("x_h2j_base", "jw"), ("x_j2h_base", "i")):
    R = terms(run); C = {}
    for l in open(f"{S}/{run}.log"):
        m = re.search(r"coarse registration to the given sky (\S+): shift \(([-+\d.]+), ([-+\d.]+)\) mas", l)
        if m: C[m.group(1)] = np.array([float(m.group(2)), float(m.group(3))])
        m = re.search(r"coarse registration to the given sky (\S+): REFUSED", l)
        if m: C[m.group(1)] = np.zeros(2)
    names = [n for n in R if n in J and n in C and n.startswith(sel)]
    fin = np.array([R[n] for n in names]); coa = np.array([C[n] for n in names]); jo = np.array([J[n] for n in names])
    for lab, sub in (("all", np.ones(len(names), bool)),) + ((("short channel", np.array(["long" not in n for n in names])), ("long channel", np.array(["long" in n for n in names]))) if sel == "jw" else ()):
        f, c, j = fin[sub], coa[sub], jo[sub]
        off = np.median(f - j, 0)
        ec, ef = np.hypot(*(c - j - np.median(c - j, 0)).T), np.hypot(*(f - j - off).T)
        mv = np.hypot(*(f - c).T)
        print(f"{run} {lab}: n {sub.sum()}; coarse error (about its own median) median {np.median(ec):.1f} mas, 90% {np.percentile(ec,90):.1f}; final error median {np.median(ef):.1f}, 90% {np.percentile(ef,90):.1f}; moved by the fit: median {np.median(mv):.1f}, 90% {np.percentile(mv,90):.1f}, max {mv.max():.1f} mas; median(coarse - joint) ({np.median(c-j,0)[0]:+.1f}, {np.median(c-j,0)[1]:+.1f})")
