"""Sanity checks for profiles.py: Gaussian reproduction, flux conservation, unit integral of psi, timing."""
import math, time, sys, os
import numpy as np, torch
from scipy.special import erf
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profiles as P


def items_from(x, y, w1, w2, th, f, latent=None):
    l1 = 0.5 * np.log(np.asarray(w1) ** 2 - P.FLOOR ** 2 + 1e-12); l2 = 0.5 * np.log(np.asarray(w2) ** 2 - P.FLOOR ** 2 + 1e-12)
    init = dict(x=np.asarray(x, float), y=np.asarray(y, float), la=l1, lb=l2, th=np.asarray(th, float), lf=np.log(f), bg=0.0)
    if latent is not None: init['z'] = latent
    return P.Items(init, latent is not None)


def exact_axis_gauss(n, x, y, s, f):
    e = np.arange(n + 1.0); s2 = s * math.sqrt(2)
    ex = 0.5 * (erf((e[1:] - x) / s2) - erf((e[:-1] - x) / s2)); ey = 0.5 * (erf((e[1:] - y) / s2) - erf((e[:-1] - y) / s2))
    return f * np.outer(ey, ex)


n = 64
print('1) fixed Gaussian vs exact pixel integrals (round Gaussian, erf), relative max error per pixel / peak, flux ratio')
psi_g = P.Psi('gauss')
for s in [0.45, 0.6, 0.8, 1.2, 2.0, 4.0, 7.0]:
    it = items_from([30.3], [31.7], [s], [s], [0.3], [1000.0])
    with torch.no_grad(): img = P.render(it, psi_g, n, with_bg=False).numpy()
    ref = exact_axis_gauss(n, 30.3, 31.7, s, 1000.0)
    print(f'  width {s:4.2f}: max|err|/peak = {np.abs(img - ref).max() / ref.max():.2e}, sum/flux = {img.sum() / 1000:.6f}')

print('2) elliptical rotated Gaussian vs brute-force 16x16 sub-sampled reference')
for (a, b, th) in [(3.0, 0.5, 0.7), (6.0, 2.0, 2.0), (1.0, 0.45, 1.1)]:
    it = items_from([32.4], [30.9], [a], [b], [th], [500.0])
    with torch.no_grad(): img = P.render(it, psi_g, n, with_bg=False).numpy()
    o = 16; xs = (np.arange(n * o) + 0.5) / o; X, Y = np.meshgrid(xs - 32.4, xs - 30.9)
    u = (np.cos(th) * X + np.sin(th) * Y) / a; v = (-np.sin(th) * X + np.cos(th) * Y) / b
    ref = 500 * np.exp(-0.5 * (u * u + v * v)) / (2 * np.pi * a * b)
    ref = ref.reshape(n, o, n, o).mean((1, 3))
    print(f'  ({a},{b},{th}): max|err|/peak = {np.abs(img - ref).max() / ref.max():.2e}, sum/flux = {img.sum() / 500:.6f}')

print('3) learned psi with random (large) weights: unit integral by brute-force 2-D quadrature, and rendered flux')
torch.manual_seed(3)
psi = P.Psi('learned')
with torch.no_grad():
    psi.l2.weight.normal_(0, 0.6)
z = torch.randn(5, P.D_LAT)
with torch.no_grad():
    tab = psi.tables(z)
    g = torch.linspace(-P.R_SUP, P.R_SUP, 1601); h = g[1] - g[0]
    U1, U2 = torch.meshgrid(g, g, indexing='xy')
    for k in range(5):
        v = psi.evaluate(U1[None], U2[None], tab[k:k + 1])
        print(f'  latent {k}: integral = {float(v.sum() * h * h):.5f}, min value = {float(v.min()):.2e}')
    for s in [0.45, 0.7, 1.0, 2.0, 5.0]:
        it = items_from([32.2] * 5, [31.6] * 5, [s] * 5, [s * 1.3] * 5, [0.4] * 5, [1000.0] * 5, latent=z.numpy())
        sums = [float(P.render(it, psi, n, with_bg=False, idx_subset=torch.tensor([k])).sum()) / 1000 for k in range(5)]
        # reference with 16x16 point samples per pixel of the same psi
        print(f'  width {s}: rendered sum/flux = {np.round(sums, 4)}')
    print('  pixel-integration error of the sub-sampling rule, sharpest random profile, vs 16x16 sub-samples:')
    for s in [0.45, 0.7, 1.0, 1.6, 3.0]:
        it = items_from([32.2], [31.6], [s], [s], [0.4], [1000.0], latent=z[:1].numpy())
        img = P.render(it, psi, n, with_bg=False).numpy()
        o = 16; xs = (torch.arange(n * o, dtype=torch.float32) + 0.5) / o
        X, Y = torch.meshgrid(xs - 32.2, xs - 31.6, indexing='xy'); c, sn = math.cos(0.4), math.sin(0.4)
        ref = psi.evaluate(((c * X + sn * Y) / s)[None], ((-sn * X + c * Y) / s)[None], tab[:1])[0] * 1000 / s / s
        ref = ref.reshape(n, o, n, o).mean((1, 3)).numpy()
        print(f'    width {s}: max|err|/peak = {np.abs(img - ref).max() / ref.max():.2e}, rms err/peak = {np.sqrt(((img-ref)**2).mean())/ref.max():.2e}')

print('4) timing at n=256 for a lattice seeding')
for mode, budget in [('gauss', 15000), ('learned', 15000), ('learned', 45000)]:
    psi = P.Psi(mode); N = P.n_items_for(budget, mode == 'learned', psi.n_weights())
    it = P.Items(P.lattice_init(256, N, mode == 'learned'), mode == 'learned')
    t0 = time.time()
    for _ in range(3):
        img = P.render(it, psi, 256); img.sum().backward()
    print(f'  {mode} budget {budget}: {N} items, {(time.time() - t0) / 3:.3f} s per forward+backward')
