"""BSplineField (v23) tests, no data needed. Run: python3 test_bspline.py"""
import sys, math, time
import numpy as np, torch, torch.nn as nn
from scipy.interpolate import BSpline
sys.path.insert(0, "/home/claude/work/scripts")
import smacs_four_v23 as S

torch.manual_seed(0); np.random.seed(0)
B3 = BSpline.basis_element(np.arange(-2.0, 3.0), extrapolate=False)  # centred cubic B-spline, support (-2, 2)
b3 = lambda t: np.nan_to_num(B3(t))

# (a) partition of unity
for h in (8, 4, 2, 1, 0.5, 3):
    f = S.BSplineField(32, 5, spacings=(h,))
    for c in f.coefs: c.data.fill_(1.0)
    p = torch.rand(20000) * 32 - 0.5; q = torch.rand(20000) * 32 - 0.5
    p[:3], q[:3] = torch.tensor([-0.5, 31.5, 0.0]), torch.tensor([-0.5, 31.5, 31.5])  # corners / edges included
    err = (f(p, q) - 1).abs().max().item()
    assert err < 1e-6, (h, err)
print("(a) partition of unity OK (max err %.1e over spacings 8,4,2,1,0.5,3)" % err)

# (b) known cubic spline, direct reference (scipy basis elements, slow double loop over knots)
size, K, h = 20, 3, 2.5
f = S.BSplineField(size, K, spacings=(h,), scale=1.7)
f.coefs[0].data.normal_()
c = f.coefs[0].detach().double().numpy()
n = c.shape[1]
assert n == math.ceil(size / h) + 5
x = np.random.rand(300) * size - 0.5; y = np.random.rand(300) * size - 0.5
ref = np.zeros((300, K))
for i in range(n):
    for j in range(n):
        xi, yj = -0.5 + (i - 2) * h, -0.5 + (j - 2) * h
        wgt = b3((x - xi) / h) * b3((y - yj) / h)
        ref += wgt[:, None] * c[:, i, j][None]
ref *= 1.7
got = f(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).detach().double().numpy()
e = np.abs(got - ref).max(); print("(b) vs reference, max abs diff %.2e (field rms %.2f)" % (e, ref.std()))
assert e < 1e-5
# outside: constant extension beyond the margin
far = f(torch.tensor([-100.0, -50.0]), torch.tensor([-100.0, 3.0])); far2 = f(torch.tensor([-20.0, -30.0]), torch.tensor([-70.0, 3.0]))
assert torch.allclose(far, far2, atol=1e-6), "clamped extension not constant"

# (c) linearity
f = S.BSplineField(24, 4, spacings=(6, 3, 1.5))
c1 = [torch.randn_like(p) for p in f.coefs]; c2 = [torch.randn_like(p) for p in f.coefs]
p = torch.rand(500) * 30 - 3; q = torch.rand(500) * 30 - 3
def ev(cs):
    for p_, c_ in zip(f.coefs, cs): p_.data.copy_(c_)
    return f(p, q)
a_, b_ = 0.7, -2.3
lhs = ev([a_ * u + b_ * v for u, v in zip(c1, c2)]); rhs = a_ * ev(c1) + b_ * ev(c2)
e = (lhs - rhs).abs().max().item(); print("(c) linearity max err %.2e" % e); assert e < 1e-5

# (d) gradcheck, float64, tiny field; includes gradient w.r.t. positions and chunking path
f = S.BSplineField(6, 2, spacings=(3, 1.5), chunk=7).double()
for pp in f.coefs: pp.data.normal_()
px = (torch.rand(20, dtype=torch.float64) * 7 - 1).requires_grad_(); py = (torch.rand(20, dtype=torch.float64) * 7 - 1).requires_grad_()
cs = [pp for pp in f.coefs]
fn = lambda px, py, *cs: f._eval(px, py, *cs).T
assert torch.autograd.gradcheck(fn, (px, py, *cs), eps=1e-6, atol=1e-6)
f.zero_grad(); g1 = torch.autograd.grad(f(px, py).pow(2).sum(), list(f.coefs))  # chunk=7 -> checkpointed path
f.chunk = 100; g2 = torch.autograd.grad(f(px, py).pow(2).sum(), list(f.coefs))
assert all(torch.allclose(u, v) for u, v in zip(g1, g2))
print("(d) gradcheck float64 OK (positions and coefficients); chunked/checkpointed gradient == unchunked")

# (e) plug into ZipSkyModel; deterministic renderer as in test_det_render.py
H, Kc, Q, os_, nn_ = 64, 3, 2, 4, 11
no = nn_ * os_
m = S.ZipSkyModel.__new__(S.ZipSkyModel); nn.Module.__init__(m)
m.sky = S.BSplineField(H, Kc, spacings=(8, 4, 2, 1, 0.5), scale=1.0)
for pp in m.sky.coefs: pp.data.normal_()
m.register_buffer("lam_nodes", torch.tensor([[1.0, 1.5]])); m.register_buffer("w_nodes", torch.full((1, Q), 1.0 / Q))
m.phi = torch.rand(1, Q, Kc)
u = (torch.arange(no) - (no - 1) / 2) / os_
yy, xx = torch.meshgrid(u, u, indexing="ij")
I = torch.stack([torch.exp(-((xx - 1.3) ** 2 / 2 / 0.8 ** 2 + (yy + 0.6) ** 2 / 2 / 0.5 ** 2)),
                 torch.exp(-((xx + 0.7) ** 2 / 2 / 0.6 ** 2 + (yy - 1.1) ** 2 / 2 / 0.9 ** 2))])[:, None]
I = I / I.sum((-2, -1), keepdim=True) * torch.tensor([0.9, 0.8])[:, None, None, None]
m.I, m.no, m.os = I, no, os_
cdf = torch.cumsum(I.flatten(2), -1); m.ee = cdf[..., -1].clone(); m.cdf = cdf / cdf[..., -1:]
A = m.grid_coeffs(); assert A.shape == (Kc, H, H), A.shape
m.build_kernels(2, {(0, 0)}); m.render_mode = "det"
R = m.render(); assert R[(0, 0)].shape == (H * 2, H * 2) and torch.isfinite(R[(0, 0)]).all()
x = torch.rand(100) * 40 + 12; y = torch.rand(100) * 40 + 12
with torch.no_grad():
    det = m.core(0, 0, x, y)
    m.render_mode = "mc"; mc = torch.stack([m.core(0, 0, x, y, n=512, gen=torch.Generator().manual_seed(s)) for s in range(8)]).mean(0)
print("(e) grid_coeffs", tuple(A.shape), "render", tuple(R[(0, 0)].shape), "det vs MC rel rms %.2e (white-noise field, informational)" % ((det - mc).pow(2).mean().sqrt() / mc.abs().mean()))
m.render_mode = "det"; m._R = None; m.render()
m._R = m.render(); m.core(0, 0, x, y).pow(2).sum().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.sky.parameters()); print("    gradients reach all levels through the renderer")

# (f) warm-start-style fit
def fit(sp, target, steps=400):
    f = S.BSplineField(64, 1, spacings=sp)
    gx, gy = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")  # gx varies with row; use as (y, x)
    ty, tx = gx.flatten(), gy.flatten()
    opt = torch.optim.Adam(f.parameters(), lr=5e-2)
    for _ in range(steps):
        opt.zero_grad(); l = (f(tx, ty).T.reshape(1, 64, 64) - target).pow(2).mean(); l.backward(); opt.step()
    return float(l)
tgt = torch.randn(1, 64, 64)
m1, m2 = fit((4, 2), tgt), fit((4, 2, 1, 0.5), tgt)
print("(f) mse spacings 4,2: %.4f ; 4,2,1,0.5: %.4f" % (m1, m2)); assert m2 < m1

# (g) timing
f = S.BSplineField(512, 8, spacings=(8, 4, 2, 1, 0.5))
for pp in f.coefs: pp.data.normal_()
npar = sum(p.numel() for p in f.parameters())
N = 1_600_000
tx = torch.rand(N) * 512 - 0.5; ty = torch.rand(N) * 512 - 0.5
torch.set_num_threads(torch.get_num_threads())
t = time.perf_counter(); f(tx, ty).pow(2).sum().backward(); dt = time.perf_counter() - t
print("(g) forward+backward, N=1.6e6, K=8, spacings 8,4,2,1,0.5, CPU (%d threads): %.2f s" % (torch.get_num_threads(), dt))
print("params size=512 K=8 default spacings:", npar, [tuple(p.shape) for p in f.coefs])
print("ALL OK")
