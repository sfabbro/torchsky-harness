"""Did the kernel stay near the official image? For each fit and filter: correlation of the learned mean kernel with the official stamp
(point reflection, as the start reads it), in log density with the radial mean removed (azimuthal structure, 0.08 to 0.55 arcsec), and
the rms difference in log density inside 0.3 arcsec after removing a constant."""
import sys
import numpy as np
import torch
from torchsky.map.exposure_fit import SkyFit
S = sys.argv[1]
lib = np.load(f"{S}/psflib/nircam_epsf.npz")
g = torch.linspace(-0.6, 0.6, 121); V0, V1 = torch.meshgrid(g, g, indexing="xy"); us = torch.stack([V0, V1], -1).reshape(-1, 2); r = us.norm(dim=-1).numpy()
def deradial(a):
    out = a.copy(); b = np.digitize(r, np.linspace(0, 0.86, 44))
    for k in np.unique(b): out[b == k] -= a[b == k].mean()
    return out
def official(name):
    val = 0; w = -us.numpy()
    for st in [lib[k].mean(0) for k in lib.files if k.endswith("/" + name)]:
        n = st.shape[0]; pitch = 4.0 / n; c = (n - 1) / 2
        x, y = w[:, 0] / pitch + c, w[:, 1] / pitch + c
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); fx, fy = x - x0, y - y0
        val = val + (st[y0, x0] * (1 - fx) * (1 - fy) + st[y0, x0 + 1] * fx * (1 - fy) + st[y0 + 1, x0] * (1 - fx) * fy + st[y0 + 1, x0 + 1] * fx * fy) / pitch**2
    return np.log(val)
OFF = {b: official(b) for b in ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")}
sel, core = (r > 0.08) & (r < 0.55), r < 0.3
for run in sys.argv[2:]:
    fit = SkyFit.load(f"{S}/fits/{run}", device="cpu"); m = fit.operator; bands = list(fit.bands); out = []
    for name in OFF:
        with torch.no_grad():
            lk = m.net(m.net.enc(us), m.mean_codes(bands.index(name))[:1])[0].numpy()
        d = lk - OFF[name]; d = d[core] - d[core].mean()
        out.append(f"{name} {np.corrcoef(deradial(lk)[sel], deradial(OFF[name])[sel])[0, 1]:+.2f} / {d.std():.2f}")
    print(f"{run:22s} " + "  ".join(out))
