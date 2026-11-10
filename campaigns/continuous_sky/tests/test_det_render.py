"""Deterministic zip renderer vs the v20 Monte Carlo sampler, on a synthetic asymmetric PSF and a random field."""
import sys, math, torch, torch.nn as nn
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v21 as S
torch.manual_seed(1)
H, K, Q, os_, n = 64, 3, 2, 4, 11
no = n * os_
m = S.ZipSkyModel.__new__(S.ZipSkyModel); nn.Module.__init__(m)
m.sky = S.ZipField(H, K, levels=(8, 16, 32, 64), feat=4, hidden=32, scale=1.0)
for G in m.sky.grids:
    G.data.normal_(0, 1.0)  # structure down to the pixel scale
m.register_buffer("lam_nodes", torch.tensor([[1.0, 1.5]]))
m.register_buffer("w_nodes", torch.full((1, Q), 1.0 / Q))
m.phi = torch.rand(1, Q, K)
u = (torch.arange(no) - (no - 1) / 2) / os_
yy, xx = torch.meshgrid(u, u, indexing="ij")
I = torch.stack([torch.exp(-((xx - 1.3) ** 2 / 2 / 0.8 ** 2 + (yy + 0.6) ** 2 / 2 / 0.5 ** 2)) + 0.1 * torch.exp(-((xx + 2) ** 2 + yy ** 2) / 2),
                 torch.exp(-((xx + 0.7) ** 2 / 2 / 0.6 ** 2 + (yy - 1.1) ** 2 / 2 / 0.9 ** 2))])[:, None]  # (Q, D=1, no, no)
I = I / I.sum((-2, -1), keepdim=True) * torch.tensor([0.9, 0.8])[:, None, None, None]
m.I, m.no, m.os = I, no, os_
cdf = torch.cumsum(I.flatten(2), -1); m.ee = cdf[..., -1].clone(); m.cdf = cdf / cdf[..., -1:]
m.build_kernels(2, {(0, 0)})
x = torch.rand(400) * 40 + 12; y = torch.rand(400) * 40 + 12
with torch.no_grad():
    gen = torch.Generator().manual_seed(0)
    mc = torch.stack([m.core(0, 0, x, y, n=512, gen=gen) for _ in range(8)]).mean(0)  # 4096 samples
    m.render_mode = "det"; det = m.core(0, 0, x, y)
    m.build_kernels(1, {(0, 0)}); m._R = None; det4 = m.core(0, 0, x, y)
scale = mc.abs().mean()
print("mean |MC| %.4f" % scale)
print("det os2 - MC: mean %.2e  rms %.2e (rel rms %.2e)" % ((det - mc).mean(), (det - mc).pow(2).mean().sqrt(), (det - mc).pow(2).mean().sqrt() / scale))
print("det os1 - MC: mean %.2e  rms %.2e (rel rms %.2e)" % ((det4 - mc).mean(), (det4 - mc).pow(2).mean().sqrt(), (det4 - mc).pow(2).mean().sqrt() / scale))
# MC noise level of the 4096-sample reference
with torch.no_grad():
    m.render_mode = "mc"
    g2 = torch.Generator().manual_seed(7)
    a1 = m.core(0, 0, x, y, n=512, gen=g2); a2 = m.core(0, 0, x, y, n=512, gen=g2)
print("MC 512-sample rms difference / sqrt(2) / sqrt(8): %.2e" % ((a1 - a2).pow(2).mean().sqrt() / math.sqrt(2) / math.sqrt(8)))
# shifted-kernel sign check: a mirrored PSF must give a different answer
m.I = I.flip(-1, -2); m.build_kernels(2, {(0, 0)}); m._R = None; m.render_mode = "det"
with torch.no_grad(): detm = m.core(0, 0, x, y)
print("sensitivity: mirrored-PSF det - MC rel rms %.2e (must be >> the above)" % ((detm - mc).pow(2).mean().sqrt() / scale))
m.I = I
# gradient path as in fit_zip
m.render_mode = "det"; m.build_kernels(2, {(0, 0)})
R_full = m.render(); m._R = {k: v.detach().requires_grad_(True) for k, v in R_full.items()}
l = m.core(0, 0, x, y).pow(2).sum(); l.backward()
torch.autograd.backward([R_full[(0, 0)]], [m._R[(0, 0)].grad])
g_split = [p.grad.clone() for p in m.sky.parameters()]
for p in m.sky.parameters(): p.grad = None
m._R = m.render(); l2 = m.core(0, 0, x, y).pow(2).sum(); l2.backward()
print("gradient split vs direct, max rel diff: %.2e" % max(float((a - p.grad).abs().max() / p.grad.abs().max().clamp(min=1e-30)) for a, p in zip(g_split, m.sky.parameters())))
