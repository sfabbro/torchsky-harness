"""v24: the zip renderers must place a point source's response where the explicit model's convolve_basis does."""
import sys, torch, torch.nn as nn
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v24 as S
H, os_, n = 32, 4, 7
no = n * os_
m = S.ZipSkyModel.__new__(S.ZipSkyModel); nn.Module.__init__(m)
class Pt(nn.Module):  # field: narrow bump at (16, 16) for coefficient 0
    size = H
    def forward(self, tx, ty):
        v = torch.exp(-((tx - 16) ** 2 + (ty - 16) ** 2) / (2 * 0.3 ** 2))
        return torch.stack([v], -1)
m.sky = Pt()
m.register_buffer("lam_nodes", torch.tensor([[1.0]])); m.register_buffer("w_nodes", torch.ones(1, 1)); m.phi = torch.ones(1, 1, 1)
for conv in ("explicit", "v20"):
    S.ZIP_PSF_CONVENTION = conv
    Ib = torch.zeros(1, 1, n, n); Ib[0, 0, n // 2 + 2, n // 2 + 1] = 1.0  # sky-grid kernel tap at dy=+2, dx=+1
    I = Ib.repeat_interleave(os_, -1).repeat_interleave(os_, -2) / os_ ** 2  # same tap on the fine grid
    if conv == "explicit":
        I = I.flip(-2, -1)
    m.I, m.no, m.os = I, no, os_
    cdf = torch.cumsum(I.flatten(2), -1); m.ee = cdf[..., -1].clone(); m.cdf = cdf / cdf[..., -1:]
    m.render_mode = "mc"; m._R = None
    m.build_kernels(2, {(0, 0)}); m.render_mode = "det"
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(H).float(), indexing="ij")
    with torch.no_grad():
        img = m.core(0, 0, xx.flatten(), yy.flatten()).view(H, H)
    pk = divmod(int(img.argmax()), H)
    S0 = torch.zeros(1, H, H); S0[0, 16, 16] = 1
    ref = S.convolve_basis(S0, Ib)[0, 0]
    print(conv, "zip det response peak (y,x)", pk, "| explicit convolve_basis peak", divmod(int(ref.argmax()), H))
