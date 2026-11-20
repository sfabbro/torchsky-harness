"""Stage 2: learned-kernel azimuthal profiles (reusing kernel_check loaders). Writes kern_<key>.npz to the cache."""
import sys
import time
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/claude/work/kernel_check")
from prep import CACHE, STATES
import kc_common as K
import q1_kernel as Q
def run(key, name):
    s = torch.load(f"/home/claude/work/states/{name}_state.pt", map_location="cpu", weights_only=False)
    sk = s["skies"][0]
    if "A" not in sk:  # item sky: dummy keys so that Fit can load it (kernel does not use them)
        N = sk["resid"].shape[-1]; sk["A"] = torch.zeros(1, 12, N, N); sk["pre"] = torch.ones(12, N, N)
    tmp = f"/tmp/claude-0/-home-claude/286b9f86-e33d-5aff-95e8-42cb96d595fe/scratchpad/tmp_{key}_state.pt"; torch.save(s, tmp); K.STATE = tmp
    fit = K.Fit(); out = {}
    for f in range(6):
        es = np.nonzero(fit.e_filt == f)[0]; dets, cnt = np.unique(fit.e_det[es], return_counts=True); det = int(dets[np.argmax(cnt)])
        e_rot = int(es[fit.e_det[es] == det][0]); lat = fit.ref_lattice_xi(det).numpy()
        t = time.time(); o, (r, prof) = Q.measure(fit, f, det, lat[4], e_rot, full=False)
        out["r"] = r; out[f"p{f}"] = prof; out[f"tot{f}"] = o["integral_continuous_over_levels"]
        print(key, K.FILTERS[f], "det", det, round(time.time() - t, 1), "s; cont integral/quadrature", round(o["integral_continuous_over_levels"], 4), flush=True)
    np.savez(f"{CACHE}/kern_{key}.npz", **out)
if __name__ == "__main__":
    run(sys.argv[1], STATES.get(sys.argv[1], sys.argv[1]))
