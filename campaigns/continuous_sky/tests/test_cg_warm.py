import sys, time, torch, torch.nn as nn
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v27 as S
torch.manual_seed(0); H, K = 96, 8
A0 = torch.randn(K, H, H) * 0.5
yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(H).float(), indexing="ij")
for _ in range(15):
    y, x = torch.randint(5, H - 5, (2,)); amp = 10 ** (torch.rand(1) * 4)
    A0 += amp * torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2 / 0.7 ** 2) * torch.rand(K, 1, 1)
m = S.ZipSkyModel.__new__(S.ZipSkyModel); nn.Module.__init__(m)
m.sky = S.BSplineField(H, K, spacings=(8, 4, 2, 1), scale=20.0)
m.register_buffer("lam_nodes", torch.ones(1, 1))
t = time.time(); l = S.cg_warm_start(m, A0, 1.0, 300); print("CG 300 it: mse/sig2 %.4g, %.0f s" % (float(l), time.time() - t))
print("check grid_coeffs after:", float((m.grid_coeffs() - A0).pow(2).mean()))
