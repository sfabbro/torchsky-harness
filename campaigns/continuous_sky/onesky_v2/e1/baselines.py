"""Baseline image representations for experiment E1: pixel grid, Gaussians, hash-grid neural field.

All are fitted to data['noisy'] (or data['truth']) with per-pixel weights 1/sigma^2 and saved with
common.save_result. Usage example:
    python baselines.py --method gauss_adaptive --budget 15000 --steps 2000 --lr 1.0
Declared approximations are listed in BASELINES_REPORT.md and next to the code concerned.
"""
import argparse, math, os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
torch.set_num_threads(1)
torch.set_flush_denormal(True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FLOOR = common.FLOOR_SIGMA      # 0.45 px, the declared resolution floor
MAX_WIDTH = 7.8                 # compute cap: widths limited to 7.8 px so 4 widths + pixel variance fit a 65 x 65 patch
PIX_VAR = 1.0 / 12.0            # variance of a box pixel; added to every Gaussian variance


def still_falling(hist):
    """hist: list of loss values at equal spacing. True if mean of last 10% is >1% below mean of 80-90%."""
    h = np.asarray(hist)
    if len(h) < 20: return True
    k = max(1, len(h) // 10)
    a = h[-2 * k:-k].mean(); b = h[-k:].mean()
    return bool((a - b) / b > 0.01)


# ----------------------------------------------------------------------------------------- grid
def fit_grid(data, budget, target='noisy', max_iter=400, tol=1e-7):
    """G x G coefficient grid (G = floor(sqrt(budget))), bicubic read at pixel centres
    (torch grid_sample 'bicubic', border padding; its kernel is Keys with a=-0.75, which passes through the nodes).
    Linear least squares solved by conjugate gradients on the weighted normal equations."""
    n = data['n']; G = int(math.floor(math.sqrt(budget)))
    y = torch.tensor(data[target], dtype=torch.float64).reshape(1, 1, n, n)
    w = torch.tensor(1.0 / data['sigma'] ** 2, dtype=torch.float64).reshape(1, 1, n, n)
    c1 = (torch.arange(n, dtype=torch.float64) + 0.5) / n * 2 - 1
    grid = torch.stack(torch.meshgrid(c1, c1, indexing='xy'), -1)[None]     # (1,n,n,2) as (x,y)
    def A(c): return F.grid_sample(c, grid, mode='bicubic', padding_mode='border', align_corners=False)
    def At(v):
        c = torch.zeros(1, 1, G, G, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad((A(c) * v).sum(), c)[0]
    def N(c): return At(w * A(c))
    t0 = time.time()
    b = At(w * y); c = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r * r).sum(); b2 = (b * b).sum(); hist = []
    it = 0
    for it in range(1, max_iter + 1):
        Np = N(p); al = rs / (p * Np).sum(); c = c + al * p; r = r - al * Np
        rn = (r * r).sum(); hist.append(float((w * (A(c) - y) ** 2).mean()) if it % 5 == 0 else None)
        if rn / b2 < tol ** 2: break
        p = r + rn / rs * p; rs = rn
    model = A(c).reshape(n, n).numpy()
    h = [x for x in hist if x is not None]
    loss = float((w * (A(c) - y) ** 2).mean())
    return model, G * G, time.time() - t0, dict(iterations=it, final_loss=loss, still_falling=bool(math.sqrt(float(rn / b2)) > 1e-3),
            relative_normal_residual=math.sqrt(float(rn / b2)), G=G, interpolation='bicubic (grid_sample, a=-0.75)', solver='conjugate gradients')


# ------------------------------------------------------------------------------------ gaussians
PATCH_H = [4, 8, 16, 32, 64, 128, 256]     # patch half-widths (patch sizes 9,17,33,65,129,257,513)


def halton(m, base):
    out = np.zeros(m)
    for i in range(m):
        f = 1.0; r = 0.0; k = i + 1
        while k > 0:
            f /= base; r += f * (k % base); k //= base
        out[i] = r
    return out


class Gaussians(nn.Module):
    """Sum of N anisotropic Gaussians + constant background.
    Per Gaussian: x, y (pixel units, pixel i centred at i+0.5), la, lb (log widths), theta, lf (log flux).
    width = sqrt(0.45^2 + exp(2 l))  (so the width is never below 0.45 px, and = exp(l) when large);
    flux = exp(lf) (kept positive; used instead of softplus because fluxes span 4 decades).
    Pixel integration (approximate): add 1/12 px^2 to each variance. Each Gaussian is evaluated on a square patch of
    half-width ~ 4 major widths (sizes 9..513) and scatter-added."""
    def __init__(self, n, init):
        super().__init__()
        self.n = n
        for k in ['x', 'y', 'la', 'lb', 'th', 'lf']:
            setattr(self, k, nn.Parameter(torch.tensor(init[k], dtype=torch.float32)))
        self.bg = nn.Parameter(torch.tensor(float(init['bg'])))
        self.N = len(init['x'])

    def widths(self):
        la = torch.clamp(self.la, max=math.log(MAX_WIDTH)); lb = torch.clamp(self.lb, max=math.log(MAX_WIDTH))
        a = torch.sqrt(FLOOR ** 2 + torch.exp(2 * la)); b = torch.sqrt(FLOOR ** 2 + torch.exp(2 * lb))
        return a, b

    def forward(self):
        n = self.n; a, b = self.widths()
        a2 = a * a + PIX_VAR; b2 = b * b + PIX_VAR
        need = torch.ceil(4 * torch.sqrt(torch.maximum(a2, b2))).detach()
        flux = torch.exp(self.lf); cth = torch.cos(self.th); sth = torch.sin(self.th)
        out = torch.zeros(n * n, dtype=torch.float32)
        bucket = torch.full((self.N,), len(PATCH_H) - 1, dtype=torch.long)
        for k in range(len(PATCH_H) - 1, -1, -1):
            bucket[need <= PATCH_H[k]] = k
        for k, h in enumerate(PATCH_H):
            idx = torch.nonzero(bucket == k).squeeze(1)
            if len(idx) == 0: continue
            P = 2 * h + 1; chunk = max(1, 1_200_000 // (P * P))
            off = torch.arange(-h, h + 1)
            for s in range(0, len(idx), chunk):
                i = idx[s:s + chunk]
                x = self.x[i]; y = self.y[i]
                cx = torch.floor(x.detach()).long(); cy = torch.floor(y.detach()).long()
                ix = cx[:, None] + off[None]; iy = cy[:, None] + off[None]            # (m,P)
                dx = (ix.float() + 0.5) - x[:, None]; dy = (iy.float() + 0.5) - y[:, None]
                # q/2 = 0.5*A*dx^2 + 0.5*C*dy^2 + Bc*dx*dy, built from 1D pieces so only three (m,P,P) passes remain
                ca = cth[i]; sa = sth[i]; ia = 1.0 / a2[i]; ib = 1.0 / b2[i]
                A_ = ca * ca * ia + sa * sa * ib; C_ = sa * sa * ia + ca * ca * ib; Bc = ca * sa * (ia - ib)
                okx = ((ix >= 0) & (ix < n)); oky = ((iy >= 0) & (iy < n))
                ex = 0.5 * A_[:, None] * dx * dx + (~okx).float() * 1e4 + (0.5 * torch.log(a2[i] * b2[i]) + math.log(2 * math.pi) - torch.log(flux[i]))[:, None]
                ey = 0.5 * C_[:, None] * dy * dy + (~oky).float() * 1e4
                arg = ex[:, None, :] + ey[:, :, None] + (Bc[:, None] * dx)[:, None, :] * dy[:, :, None]
                flat = (iy.clamp(0, n - 1) * n)[:, :, None] + ix.clamp(0, n - 1)[:, None, :]
                out = out.index_add(0, flat.reshape(-1), torch.exp(-arg).reshape(-1))
        return (out + self.bg).reshape(n, n)


LATTICE_FRAC = [0.02, 0.18, 0.80]


def lattice_init(n, N, rng, init_sb=0.3):
    """Data-independent seeding: three square lattices (coarse, medium, fine; 2%, 18%, 80% of N) plus the
    remainder on a Halton point set at the fine scale. Width = half the lattice spacing; flux gives each level
    a surface brightness of init_sb (an arbitrary constant, not read from the data)."""
    fr = LATTICE_FRAC; xs, ys, ss = [], [], []
    left = N
    for f in fr:
        m = max(1, int(math.floor(math.sqrt(f * N)))); m = min(m, int(math.floor(math.sqrt(left))))
        sp = n / m; g = (np.arange(m) + 0.5) * sp
        X, Y = np.meshgrid(g, g); xs.append(X.ravel()); ys.append(Y.ravel()); ss.append(np.full(m * m, 0.5 * sp)); left -= m * m
    if left > 0:
        sp = n / math.sqrt(N * fr[-1]); xs.append(halton(left, 2) * n); ys.append(halton(left, 3) * n); ss.append(np.full(left, 0.5 * sp))
    x = np.concatenate(xs); y = np.concatenate(ys); s = np.concatenate(ss)
    assert len(x) == N, (len(x), N)
    th = rng.uniform(0, np.pi, N)
    s = np.minimum(s, MAX_WIDTH)
    return dict(x=x, y=y, la=np.log(s), lb=np.log(s), th=th, lf=np.log(init_sb * (2 * s) ** 2), bg=1.0)


def fit_gauss(data, budget, adaptive, target='noisy', steps=2000, lr=1.0, thr=3.0, every=0.12,
              max_move=0.3, seed=0, time_limit=200.0, verbose=False):
    n = data['n']; N = (budget - 1) // 6
    rng = np.random.default_rng(seed)
    model = Gaussians(n, lattice_init(n, N, rng))
    y = torch.tensor(data[target], dtype=torch.float32); w = torch.tensor(1.0 / data['sigma'] ** 2, dtype=torch.float32)
    base = dict(x=0.05, y=0.05, la=0.03, lb=0.03, th=0.05, lf=0.05, bg=0.1)
    opt = torch.optim.Adam([dict(params=[getattr(model, k)], lr=base[k] * lr, name=k) for k in base])
    hist = []; t0 = time.time(); moved_total = 0; it = 0; n_events = 0
    next_event = every
    while True:
        prog = max(it / steps, (time.time() - t0) / time_limit)      # progress: iteration budget or wall-clock, whichever is further on
        if prog >= 1.0: break
        f = 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups: g['lr'] = base[g['name']] * lr * f
        it += 1
        opt.zero_grad(); img = model(); loss = (w * (img - y) ** 2).mean(); loss.backward(); opt.step()
        if it % 10 == 0: hist.append(float(loss.detach()))
        if adaptive and prog >= next_event and prog <= 0.75:
            next_event += every; n_events += 1
            res = gaussian_filter(np.clip(y.numpy() - img.detach().numpy(), 0, None), 2.0)
            with torch.no_grad():
                flux = torch.exp(model.lf); low = torch.nonzero(flux < thr).squeeze(1)
                if len(low) > 0:
                    low = low[torch.argsort(flux[low])][:int(max_move * N)]
                    p = res.ravel() + 1e-12; p /= p.sum(); pick = rng.choice(n * n, size=len(low), p=p)
                    model.x[low] = torch.tensor((pick % n) + rng.uniform(0, 1, len(low)), dtype=torch.float32)
                    model.y[low] = torch.tensor((pick // n) + rng.uniform(0, 1, len(low)), dtype=torch.float32)
                    model.la[low] = math.log(0.5); model.lb[low] = math.log(0.5)
                    model.th[low] = torch.tensor(rng.uniform(0, np.pi, len(low)), dtype=torch.float32)
                    model.lf[low] = math.log(2 * thr)
                    for g in opt.param_groups:
                        if g['name'] in ('x', 'y', 'la', 'lb', 'th', 'lf'):
                            st = opt.state[g['params'][0]]
                            if 'exp_avg' in st: st['exp_avg'][low] = 0; st['exp_avg_sq'][low] = 0
                    moved_total += len(low)
        if verbose and it % 100 == 0: print(it, float(loss), round(time.time() - t0, 1), flush=True)
    timed_out = (time.time() - t0) / time_limit >= 1.0 and it < steps
    with torch.no_grad(): img = model().numpy()
    secs = time.time() - t0
    fl = hist[-1] if hist else float(loss)
    ex = dict(iterations=it, final_loss=fl, still_falling=still_falling(hist), n_gaussians=N, lr_multiplier=lr, steps_planned=steps,
              timed_out=timed_out, relocated_total=moved_total if adaptive else 0, relocation_threshold_flux=thr if adaptive else None,
              relocation_every_fraction_of_run=every if adaptive else None, relocation_events=n_events, time_limit_s=time_limit, flux_param='exp', min_width=FLOOR, pixel_variance_added=PIX_VAR, max_width=MAX_WIDTH, lattice_fractions=LATTICE_FRAC)
    return img, 6 * N + 1, secs, ex


# ------------------------------------------------------------------------------------- hash grid
def hash_levels(L, rmin, rmax):
    b = math.exp((math.log(rmax) - math.log(rmin)) / (L - 1))
    return [int(round(rmin * b ** l)) for l in range(L)]


def mlp_params(L, F_, H=32): return (L * F_ * H + H) + (H * H + H) + (H + 1)


def pick_T(budget, res, F_=2, H=32):
    avail = budget - mlp_params(len(res), F_, H)
    best = 1
    for T in range(1, 4_000_000):
        tot = sum(min((r + 1) ** 2, T) for r in res) * F_
        if tot > avail: break
        best = T
        if T > max((r + 1) ** 2 for r in res): break
    return best


class HashField(nn.Module):
    """Multi-resolution hash grid (Instant-NGP style) + MLP (2 x 32, ReLU) + softplus output * out_scale.
    Pixel positions are fixed, so corner indices/weights for all pixels (and sub-samples) are precomputed."""
    def __init__(self, n, budget, L=12, rmin=8, rmax=512, F_=2, H=32, sub=1, out_scale=10.0):
        super().__init__()
        self.n = n; self.res = hash_levels(L, rmin, rmax); self.T = pick_T(budget, self.res, F_, H)
        self.L, self.F_, self.out_scale, self.sub = L, F_, out_scale, sub
        sizes = [min((r + 1) ** 2, self.T) for r in self.res]; self.sizes = sizes
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.table = nn.Parameter(torch.empty(int(offs[-1]), F_).uniform_(-1e-4, 1e-4))
        self.net = nn.Sequential(nn.Linear(L * F_, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))
        # precompute corners at sub-sample positions (sub x sub per pixel, or 1 at the centre)
        if sub == 1: o = np.array([0.5])
        else: o = (np.arange(sub) + 0.5) / sub
        px = (np.arange(n)[:, None] + o[None, :]).reshape(-1) / n                       # (n*sub,) in [0,1]
        XX, YY = np.meshgrid(px, px)                                                    # (n*sub, n*sub)
        # order: pixel-major with sub-samples last -> (n,sub,n,sub) -> (n*n, sub*sub)
        def to_pix(a): return a.reshape(n, sub, n, sub).transpose(0, 2, 1, 3).reshape(n * n, sub * sub)
        XX = to_pix(XX); YY = to_pix(YY)
        idx = np.zeros((n * n, sub * sub, L, 4), dtype=np.int64); wt = np.zeros((n * n, sub * sub, L, 4), dtype=np.float32)
        for l, r in enumerate(self.res):
            px_ = XX * r; py_ = YY * r; ix = np.floor(px_).astype(np.int64); iy = np.floor(py_).astype(np.int64)
            fx = (px_ - ix).astype(np.float32); fy = (py_ - iy).astype(np.float32)
            for c, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                vx = ix + dx; vy = iy + dy
                if (r + 1) ** 2 <= self.T: h = vy * (r + 1) + vx
                else: h = (vx ^ (vy * 2654435761)) % self.T
                idx[:, :, l, c] = h + offs[l]
                wt[:, :, l, c] = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
        self.idx = torch.tensor(idx); self.wt = torch.tensor(wt)

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def forward(self, pix):
        """pix: LongTensor of pixel indices (flat). Returns model value per pixel (mean over sub-samples)."""
        idx = self.idx[pix]; wt = self.wt[pix]                                   # (B,S,L,4)
        f = (self.table[idx] * wt[..., None]).sum(3)                             # (B,S,L,F)
        B, S = f.shape[:2]
        o = F.softplus(self.net(f.reshape(B * S, -1))).reshape(B, S) * self.out_scale
        return o.mean(1)


def fit_hash(data, budget, target='noisy', steps=3000, lr=1e-2, batch=16384, sub=1, seed=0, time_limit=200.0,
             out_scale=10.0, verbose=False):
    n = data['n']; torch.manual_seed(seed)
    model = HashField(n, budget, sub=sub, out_scale=out_scale)
    y = torch.tensor(data[target], dtype=torch.float32).reshape(-1); w = torch.tensor(1.0 / data['sigma'] ** 2, dtype=torch.float32).reshape(-1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(seed); hist = []; t0 = time.time(); it = 0
    while True:
        prog = max(it / steps, (time.time() - t0) / time_limit)
        if prog >= 1.0: break
        for gr in opt.param_groups: gr['lr'] = lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * prog)))
        it += 1
        pix = torch.randint(0, n * n, (batch,), generator=g) if batch < n * n else torch.arange(n * n)
        opt.zero_grad(); loss = (w[pix] * (model(pix) - y[pix]) ** 2).mean(); loss.backward(); opt.step()
        if it % 10 == 0: hist.append(float(loss.detach()))
        if verbose and it % 100 == 0: print(it, float(loss), round(time.time() - t0, 1), flush=True)
    timed_out = (time.time() - t0) / time_limit >= 1.0 and it < steps
    with torch.no_grad():
        img = torch.cat([model(torch.arange(s, min(s + 8192, n * n))) for s in range(0, n * n, 8192)]).reshape(n, n).numpy()
        full = float((w * (torch.tensor(img).reshape(-1) - y) ** 2).mean())
    secs = time.time() - t0
    ex = dict(iterations=it, final_loss=full, final_minibatch_loss_last=hist[-1] if hist else None, still_falling=still_falling(hist),
              lr=lr, batch=batch, subsamples=f'{sub}x{sub}', levels=model.L, resolutions=model.res, table_size_T=model.T,
              steps_planned=steps, timed_out=timed_out, time_limit_s=time_limit, out_scale=out_scale, features_per_level=2, hidden='2x32')
    return img, model.n_params(), secs, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True); ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--target', default='noisy'); ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=None); ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16384); ap.add_argument('--sub', type=int, default=1)
    ap.add_argument('--out', default=None, help='results dir; default results/. Use a scratch dir for development.')
    ap.add_argument('--tag', default=''); ap.add_argument('--time_limit', type=float, default=200.0)
    ap.add_argument('--verbose', action='store_true'); ap.add_argument('--thr', type=float, default=3.0)
    ap.add_argument('--every', type=float, default=0.12)
    a = ap.parse_args()
    data = common.make_data(common.make_spec(0), n=a.n, zoom=1.0)
    if a.method == 'grid': img, npar, secs, ex = fit_grid(data, a.budget, a.target)
    elif a.method in ('gauss_lattice', 'gauss_adaptive'):
        img, npar, secs, ex = fit_gauss(data, a.budget, a.method == 'gauss_adaptive', a.target, a.steps, a.lr or 1.0, thr=a.thr,
                                        every=a.every, time_limit=a.time_limit, verbose=a.verbose)
    elif a.method == 'hash':
        img, npar, secs, ex = fit_hash(data, a.budget, a.target, a.steps, a.lr or 1e-2, a.batch, a.sub, time_limit=a.time_limit, verbose=a.verbose)
    else: raise SystemExit('unknown method')
    assert npar <= a.budget or a.budget > 65000, (npar, a.budget)
    variant = 'seed0_zoom1.0' + a.tag
    r = common.save_result(a.method, a.budget, variant, img, data, n_params_image=npar, fit_seconds=secs, target=a.target,
                           extra=ex, outdir=a.out)
    print(json.dumps(dict(method=a.method, budget=a.budget, target=a.target, npar=npar, secs=round(secs, 1), chi2_truth=r['chi2_truth'],
                          psnr=r['psnr'], extra={k: ex[k] for k in ('iterations', 'final_loss', 'still_falling')})))


if __name__ == '__main__':
    main()
