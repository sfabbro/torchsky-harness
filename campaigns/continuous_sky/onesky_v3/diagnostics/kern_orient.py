"""Which orientation of the official stamp matches the kernel the joint fit learned? Azimuthal structure only (radial mean removed)."""
import sys
import numpy as np
import torch
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]
fit = SkyFit.load(f"{S}/fits/{sys.argv[2]}", device="cpu"); m = fit.operator
lib = np.load(f"{S}/psflib/nircam_epsf.npz")
bands = list(fit.bands); print("bands", bands, "instruments", fit.instruments)
g = torch.linspace(-0.6, 0.6, 121)
V0, V1 = torch.meshgrid(g, g, indexing="xy")  # V0 varies along columns of the array, V1 along rows
us = torch.stack([V0, V1], -1).reshape(-1, 2)
r = us.norm(dim=-1).numpy()
def deradial(a):
    out = a.copy(); b = np.digitize(r, np.linspace(0, 0.86, 44))
    for k in np.unique(b): out[b == k] -= a[b == k].mean()
    return out
PIX = {"F090W": 0.0310, "F150W": 0.0310, "F200W": 0.0310, "F277W": 0.0630, "F356W": 0.0630, "F444W": 0.0630}
for name in ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W"):
    f = bands.index(name)
    with torch.no_grad():
        h = m.mean_codes(f)[:1]
        lk = m.net(m.net.enc(us), h)[0].numpy()
    stamps = [lib[k].mean(0) for k in lib.files if k.endswith('/' + name)]
    res = {}
    for lab, T in (("identity", np.eye(2)), ("point reflection", -np.eye(2)), ("flip x", np.diag([-1, 1])), ("flip y", np.diag([1, -1])),
                   ("transpose", np.array([[0, 1], [1, 0]])), ("anti-transpose", -np.array([[0, 1], [1, 0]])), ("rot +90", np.array([[0, -1], [1, 0]])), ("rot -90", np.array([[0, 1], [-1, 0]]))):
        w = us.numpy() @ T.T  # stamp coordinates (x, y) [arcsec] read at kernel offset v
        val = 0
        for st in stamps:
            n = st.shape[0]; pitch = 4.0 / n; c = (n - 1) / 2  # the stamps span 4 arcsec
            x, y = w[:, 0] / pitch + c, w[:, 1] / pitch + c
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); fx, fy = x - x0, y - y0
            val = val + (st[y0, x0] * (1 - fx) * (1 - fy) + st[y0, x0 + 1] * fx * (1 - fy) + st[y0 + 1, x0] * (1 - fx) * fy + st[y0 + 1, x0 + 1] * fx * fy) / pitch**2
        a, b = deradial(np.log(val)), deradial(lk)
        sel = (r > 0.08) & (r < 0.55)
        res[lab] = float(np.corrcoef(a[sel], b[sel])[0, 1])
    print(name, " ".join(f"{k} {v:+.2f}" for k, v in sorted(res.items(), key=lambda kv: -kv[1])))
