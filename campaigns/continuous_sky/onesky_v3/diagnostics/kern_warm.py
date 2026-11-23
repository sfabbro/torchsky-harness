"""How well, and in how many steps, does the kernel field take the official stamps? Real network sizes (from the joint fit), fresh weights."""
import sys
import copy
import time
import numpy as np
import torch
from torchsky.map.exposure_fit import SkyFit
from torchsky.map.exposure_fit._fit import warm_kernel, _start_target
from torchsky.map.exposure_fit._kernel import KernelNet
S = sys.argv[1]; steps = [int(x) for x in sys.argv[2].split(",")]
torch.set_num_threads(4)
fit = SkyFit.load(f"{S}/fits/x_joint_base", device="cpu"); op = fit.operator
bands = list(fit.bands); lib = np.load(f"{S}/psflib/nircam_epsf.npz")
def stamp_of(name):
    st = [lib[k].mean(0) for k in lib.files if k.endswith("/" + name)]
    n = min(s.shape[0] for s in st)
    st = [s[(s.shape[0] - n) // 2:(s.shape[0] - n) // 2 + n, (s.shape[1] - n) // 2:(s.shape[1] - n) // 2 + n] for s in st]
    m = np.mean(st, 0); yy, xx = np.mgrid[0:n, 0:n]
    cen = ((m * xx).sum() / m.sum() - (n - 1) / 2, (m * yy).sum() / m.sum() - (n - 1) / 2)
    return torch.tensor(m / m.sum()), cen
PIX = {b: (0.0310 if b in ("F090W", "F150W", "F200W") else 0.0630 if b.startswith("F2") or b in ("F356W", "F444W") else 0.13) for b in bands}
PIX.update(F105W=0.13, F125W=0.13, F140W=0.13, F160W=0.13)
finfo, starts = {}, {}
for f, b in enumerate(bands):
    ins = int(op.e_inst[(op.e_filt == f).nonzero()[0, 0]])
    finfo[f] = {"pixel_arcsec": PIX[b], "diameter_m": 6.5 if ins == 0 else 2.4, "inst": ins}
    if any(k.endswith("/" + b) for k in lib.files):
        st, cen = stamp_of(b); starts[f] = (st, 0.25 * PIX[b]); print(b, "stamp", tuple(st.shape), "centroid offset [stamp px]", np.round(cen, 2))
g = torch.linspace(-0.6, 0.6, 121); V0, V1 = torch.meshgrid(g, g, indexing="xy"); us = torch.stack([V0, V1], -1).reshape(-1, 2); r = us.norm(dim=-1).numpy()
def deradial(a):
    out = a.copy(); b = np.digitize(r, np.linspace(0, 0.86, 44))
    for k in np.unique(b): out[b == k] -= a[b == k].mean()
    return out
for n in steps:
    m = copy.deepcopy(op); torch.manual_seed(0)
    m.net = KernelNet(m.net.enc, m.net.dh, m.net.hidden, m.net.layers)
    cfg = copy.copy(m.cfg); cfg.warm_kernel_steps = n
    t0 = time.time(); lines = []
    warm_kernel(m, cfg, finfo, log=lines.append, starts=starts)
    out = []
    for f in sorted(starts):
        with torch.no_grad():
            lk = m.net(m.net.enc(us), m.mean_codes(f)[:1])[0]
            tgt, ins = _start_target(starts[f][0], starts[f][1], us, torch.zeros(us.shape[0]))
        sel = (r > 0.08) & (r < 0.55)
        d = (lk - tgt).numpy(); d = d - d.mean()
        out.append(f"{bands[f]} azimuthal corr {np.corrcoef(deradial(lk.numpy())[sel], deradial(tgt.numpy())[sel])[0, 1]:+.2f} rms log error inside 0.6 arcsec {d.std():.2f}")
    print(f"{n} steps, {time.time() - t0:.0f} s: " + lines[-1][-150:]); print("   " + "; ".join(out))
