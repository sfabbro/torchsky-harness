"""Items with a learned, size-scaled profile (experiment E1).

Model:  I(x) = background + sum_n a_n * psi( D_n^-1 R_n^T (x - x_n) ; z_n ) / (w1_n * w2_n)
  * x_n: position (pixel units; pixel i spans [i, i+1), centre i+0.5); R_n: rotation by theta_n
  * D_n = diag(w1_n, w2_n): the item's size along its two axes. w = sqrt(0.45^2 + exp(2 l)), so no item is
    narrower than the declared floor 0.45 px; log size of an item = log sqrt(w1 w2); axis ratio = w2/w1.
    (Same width parametrisation as the Gaussian baseline in baselines.py, so the per-item count matches: 6.)
  * a_n = exp(log flux) >= 0;  z_n: latent of dimension D_LAT (only for the learned profile).
  * psi(u; z) in polar coordinates (rho, phi) of u, support rho < R_SUP:
        psi = exp(gamma(l; z)) * W(rho) * [1 + sum_{m=1..M} A_m(l; z) cos(m phi) + B_m(l; z) sin(m phi)] / N(z)
    with l = log(1 + rho / RHO0). gamma, A_m, B_m are tabulated on T_KNOTS knots uniform in l (linear
    interpolation), produced by a shared network z -> tanh hidden layer -> tables. Constraints:
      - non-negativity: sum_m sqrt(A_m^2 + B_m^2) <= tanh(.) < 1 at each knot, and linear interpolation keeps
        this between knots (the norm is convex), so the bracket is > 0 everywhere;
      - unit integral: the harmonics integrate to zero over phi, so N(z) = 2 pi int exp(gamma) W rho drho, a
        1-D integral done per item by a 400-point midpoint rule on [0, R_SUP].
    W = 1 for rho < 0.85 R_SUP and a cos^2 taper to 0 at R_SUP (bounded support).
    Fixed-Gaussian mode ('gauss'): psi = exp(-rho^2/2) [rho < R_SUP] / (2 pi (1 - exp(-R_SUP^2/2))), no latent.
  * Pixel integration: each item is point-sampled on o x o sub-samples per pixel and averaged, with
    o = clip(ceil(3 / min width), 1, 7) (width 0.45 px -> 7x7 sub-samples, width >= 3 px -> 1), so the
    sub-sample spacing is at most 1/3 of the item's width, i.e. about one node spacing of the psi table.
    This is an approximation of the pixel integral; its error is measured in `selftest`.
  * Each item is evaluated only on a square patch of side P >= 2 R_SUP * max width + 2 pixels (sizes bucketed),
    clamped to lie inside the image, and scatter-added (index_add_) into the flattened image.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

FLOOR = common.FLOOR_SIGMA
R_SUP = 3.0
RHO0 = 1.0
K_GRID = 19
H_GRID = 2 * R_SUP / (K_GRID - 1)
T_KNOTS = 12
M_HARM = 3
D_LAT = 4
HIDDEN = 12
L_MAX = math.log1p(R_SUP / RHO0)
PATCH_SIZES = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512]
CHUNK = 600_000          # max sub-samples per rendering chunk


def knot_rho():
    l = torch.linspace(0, L_MAX, T_KNOTS, dtype=torch.float64)
    return RHO0 * torch.expm1(l)


def window(rho):
    t = ((rho / R_SUP - 0.85) / 0.15).clamp(0, 1)
    return torch.cos(0.5 * math.pi * t) ** 2


class Psi(nn.Module):
    """Shared profile network. mode 'learned' or 'gauss'.
    Learned: the polar form (see module docstring) is evaluated per item on a K_GRID x K_GRID Cartesian table over
    [-R_SUP, R_SUP]^2 (spacing H_GRID psi units); psi itself is DEFINED as the bilinear interpolant of that table.
    Hence: psi >= 0 exactly (bilinear interpolation of non-negative nodes), and its integral is exactly
    H_GRID^2 * sum(nodes) (the table is zero on its border), so normalising the nodes gives unit integral exactly.
    The table spacing fixes the bandwidth of psi in its own coordinates."""
    def __init__(self, mode='learned', seed=0):
        super().__init__()
        self.mode = mode
        if mode == 'learned':
            g = torch.Generator().manual_seed(seed)
            self.l1 = nn.Linear(D_LAT, HIDDEN)
            self.l2 = nn.Linear(HIDDEN, T_KNOTS * (1 + 2 * M_HARM))
            with torch.no_grad():
                self.l1.weight.copy_(torch.randn(HIDDEN, D_LAT, generator=g) * 0.5)
                self.l1.bias.copy_(torch.randn(HIDDEN, generator=g) * 0.1)
                self.l2.weight.copy_(torch.randn(*self.l2.weight.shape, generator=g) * 0.02)
                b = torch.zeros(1 + 2 * M_HARM, T_KNOTS)
                b[0] = (-0.5 * knot_rho() ** 2).float()        # start from a Gaussian radial profile
                self.l2.bias.copy_(b.reshape(-1))
            # fixed linear maps from knot tables to the Cartesian nodes
            gr = torch.linspace(-R_SUP, R_SUP, K_GRID, dtype=torch.float64)
            U1, U2 = torch.meshgrid(gr, gr, indexing='xy')                  # row index = u2, column = u1
            rho = torch.sqrt(U1 ** 2 + U2 ** 2).reshape(-1); phi = torch.atan2(U2, U1).reshape(-1)
            lq = torch.log1p(rho / RHO0) / L_MAX * (T_KNOTS - 1)
            i0 = lq.floor().clamp(0, T_KNOTS - 2).long(); w = (lq - i0).clamp(0, 1)
            Mi = torch.zeros(K_GRID * K_GRID, T_KNOTS, dtype=torch.float64)
            ar = torch.arange(K_GRID * K_GRID)
            Mi[ar, i0] = 1 - w; Mi[ar, i0 + 1] = w
            Hm = []
            for m in range(1, M_HARM + 1):
                Hm.append(Mi * torch.cos(m * phi)[:, None]); Hm.append(Mi * torch.sin(m * phi)[:, None])
            self.register_buffer('M0', Mi.T.contiguous().float())                     # (T, K^2)
            self.register_buffer('MH', torch.cat(Hm, 1).T.contiguous().float())       # (2M T, K^2)
            win = window(rho) * (rho < R_SUP)
            self.register_buffer('win', win.float())

    def n_weights(self):
        return sum(p.numel() for p in self.parameters()) if self.mode == 'learned' else 0

    def tables(self, z):
        """z (m, D_LAT) -> normalised node tables (m, 1, K, K)."""
        out = self.l2(torch.tanh(self.l1(z))).view(-1, 1 + 2 * M_HARM, T_KNOTS)
        gam = out[:, 0]
        ab = out[:, 1:].reshape(-1, M_HARM, 2, T_KNOTS)
        amp = torch.sqrt((ab ** 2).sum(2) + 1e-8)                        # (m, M, T)
        tot = amp.sum(1, keepdim=True)                                  # (m, 1, T)
        ab = ab * (0.98 * torch.tanh(tot) / tot)[:, :, None]            # sum of harmonic amplitudes < 0.98
        lg = gam @ self.M0                                              # (m, K^2)
        lg = lg - lg.max(1, keepdim=True)[0].detach()
        G = torch.exp(lg) * self.win * (1 + ab.reshape(-1, 2 * M_HARM * T_KNOTS) @ self.MH)
        G = G / (G.sum(1, keepdim=True) * H_GRID ** 2)
        return G.view(-1, 1, K_GRID, K_GRID)

    def evaluate(self, u1, u2, tab=None):
        """psi at coordinates u (m,S,S) for m items. tab from tables() (learned mode)."""
        if self.mode == 'gauss':
            r2 = u1 * u1 + u2 * u2
            return torch.exp(-0.5 * r2) * (r2 < R_SUP ** 2) / (2 * math.pi * (1 - math.exp(-0.5 * R_SUP ** 2)))
        grid = torch.stack([u1, u2], -1) * (1.0 / R_SUP)
        return F.grid_sample(tab, grid, mode='bilinear', padding_mode='zeros', align_corners=True)[:, 0]


class Items(nn.Module):
    def __init__(self, init, use_latent):
        super().__init__()
        for k in ['x', 'y', 'la', 'lb', 'th', 'lf']:
            setattr(self, k, nn.Parameter(torch.tensor(init[k], dtype=torch.float32)))
        self.use_latent = use_latent
        if use_latent:
            self.z = nn.Parameter(torch.tensor(init['z'], dtype=torch.float32))
        self.bg = nn.Parameter(torch.tensor(float(init.get('bg', 0.0))))
        self.N = len(init['x'])

    def per_item(self):
        return 6 + (D_LAT if self.use_latent else 0)

    def n_params(self):
        return self.N * self.per_item() + 1

    def widths(self):
        return torch.sqrt(FLOOR ** 2 + torch.exp(2 * self.la)), torch.sqrt(FLOOR ** 2 + torch.exp(2 * self.lb))


def subsample_factor(wmin):
    return torch.clamp(torch.ceil(3.0 / wmin), 1, 7).long()


def render(items: Items, psi: Psi, n: int, with_bg=True, idx_subset=None):
    w1, w2 = items.widths()
    wmax = torch.maximum(w1, w2).detach(); wmin = torch.minimum(w1, w2).detach()
    need = torch.ceil(2 * R_SUP * wmax + 2).long()
    ps = torch.tensor(PATCH_SIZES)
    pb = torch.searchsorted(ps, need).clamp(max=len(PATCH_SIZES) - 1)
    o = subsample_factor(wmin)
    flux = torch.exp(items.lf)
    cth, sth = torch.cos(items.th), torch.sin(items.th)
    tab_all = psi.tables(items.z) if psi.mode == 'learned' else None
    out = torch.zeros(n * n, dtype=torch.float32)
    key = pb * 8 + o
    sel = torch.arange(items.N) if idx_subset is None else idx_subset
    for kv in torch.unique(key[sel]).tolist():
        idx = sel[key[sel] == kv]
        P = min(PATCH_SIZES[kv // 8], n); oo = kv % 8; S = P * oo
        chunk = max(1, CHUNK // (S * S))
        for c0 in range(0, len(idx), chunk):
            i = idx[c0:c0 + chunk]
            x = items.x[i]; y = items.y[i]
            ox = torch.clamp(torch.floor(x.detach()).long() - P // 2, 0, n - P)
            oy = torch.clamp(torch.floor(y.detach()).long() - P // 2, 0, n - P)
            sub = (torch.arange(S, dtype=torch.float32) + 0.5) / oo
            dx = (ox[:, None].float() + sub[None]) - x[:, None]          # (m,S)
            dy = (oy[:, None].float() + sub[None]) - y[:, None]
            DX = dx[:, None, :]; DY = dy[:, :, None]
            c = cth[i][:, None, None]; s = sth[i][:, None, None]
            a = w1[i][:, None, None]; b = w2[i][:, None, None]
            u1 = (c * DX + s * DY) / a; u2 = (-s * DX + c * DY) / b
            val = psi.evaluate(u1, u2, None if tab_all is None else tab_all[i])
            val = val * (flux[i] / (w1[i] * w2[i]))[:, None, None]
            if oo > 1:
                val = val.view(len(i), P, oo, P, oo).mean((2, 4))
            ar = torch.arange(P)
            pix = (oy[:, None, None] + ar[None, :, None]) * n + (ox[:, None, None] + ar[None, None, :])
            out = out.index_add(0, pix.reshape(-1), val.reshape(-1))
    img = out.view(n, n)
    return img + items.bg if with_bg else img


# ------------------------------------------------------------------------------------- seeding
def n_items_for(budget, use_latent, shared=0):
    return (budget - shared - 1) // (6 + (D_LAT if use_latent else 0))


def _square(n, cnt, offset=0.5):
    k = int(math.ceil(math.sqrt(cnt)))
    g = (np.arange(k) + offset) * n / k
    X, Y = np.meshgrid(g, g)
    # take cnt of the k*k cells, spread evenly (every j-th cell) when cnt < k*k
    sel = np.round(np.linspace(0, k * k - 1, cnt)).astype(int)
    return X.ravel()[sel], Y.ravel()[sel], n / k


def lattice_init(n, N, use_latent, seed=0):
    """Data-independent seeding (identical for learned and fixed-Gaussian profiles):
      1. coarse full-coverage square lattices with spacing = 2 x width, widths 32, 16, 8, 4 px (coarsest first),
         added while their total stays <= 25% of N;
      2. one third of the remaining items on a square lattice with width = half its spacing (covering, medium);
      3. the other two thirds on a square lattice offset by half a cell with width 1 px (compact).
    Flux of every item: 0.5 x (its lattice spacing)^2 (an arbitrary constant, not read from the data); theta 0;
    latent ~ N(0, 0.3^2) from a fixed random generator."""
    rng = np.random.default_rng(seed)
    xs, ys, ws, fl = [], [], [], []
    used = 0; w = 32.0
    while w >= 4.0:
        k = int(round(n / (2 * w))); cnt = k * k
        if used + cnt > 0.25 * N: break
        X, Y, sp = _square(n, cnt)
        xs.append(X); ys.append(Y); ws.append(np.full(cnt, w)); fl.append(np.full(cnt, 0.5 * sp * sp)); used += cnt; w /= 2
    rest = N - used; n_med = rest // 3; n_cmp = rest - n_med
    if n_med > 0:
        X, Y, sp = _square(n, n_med)
        xs.append(X); ys.append(Y); ws.append(np.full(n_med, max(0.6, sp / 2))); fl.append(np.full(n_med, 0.5 * sp * sp))
    if n_cmp > 0:
        X, Y, sp = _square(n, n_cmp, offset=0.0)
        X = (X + 0.5 * sp) % n; Y = (Y + 0.5 * sp) % n
        xs.append(X); ys.append(Y); ws.append(np.full(n_cmp, 1.0)); fl.append(np.full(n_cmp, 0.5 * sp * sp))
    x = np.concatenate(xs); y = np.concatenate(ys); wd = np.concatenate(ws); f = np.concatenate(fl)
    l = 0.5 * np.log(np.maximum(wd ** 2 - FLOOR ** 2, 1e-4))
    init = dict(x=x, y=y, la=l, lb=l.copy(), th=np.zeros(N), lf=np.log(f), bg=0.0)
    if use_latent:
        init['z'] = rng.normal(0, 0.3, (N, D_LAT))
    return init


# ------------------------------------------------------------------------------------- fitting
LR = dict(x=0.2, y=0.2, la=0.03, lb=0.03, th=0.03, lf=0.05, z=0.03, bg=0.1)
LR_PSI = 5e-3


def still_falling(hist):
    h = np.asarray(hist)
    if len(h) < 20: return True
    k = max(1, len(h) // 10)
    a = h[-2 * k:-k].mean(); b = h[-k:].mean()
    return bool((a - b) / b > 0.01)


def make_opt(items_list, psi, train_psi):
    groups = []
    for it in items_list:
        for k, lr in LR.items():
            if k == 'z' and not it.use_latent: continue
            groups.append(dict(params=[getattr(it, k)], lr=lr, name=k))
    if train_psi:
        groups.append(dict(params=list(psi.parameters()), lr=LR_PSI, name='psi'))
    return torch.optim.Adam(groups)


def relocate(items: Items, opt, model_img, data_t, sig_t, thr=1.0, rng=None, max_frac=0.05):
    """Adaptive step: items with signal-to-noise flux / (sigma_local * sqrt(4 pi w1 w2 + 1)) < thr are moved to
    positions drawn in proportion to the Gaussian-smoothed (1 px) positive residual in noise units; they restart
    round (width 1 px), theta 0, latent 0, flux = smoothed residual (counts/px) x 2 pi x 1.5^2.
    At most max_frac of the items per call (lowest S/N first)."""
    n = data_t.shape[0]
    with torch.no_grad():
        w1, w2 = items.widths()
        xi = items.x.clamp(0, n - 1e-3).long(); yi = items.y.clamp(0, n - 1e-3).long()
        snr = torch.exp(items.lf) / (sig_t[yi, xi] * torch.sqrt(4 * math.pi * w1 * w2 + 1))
        low = torch.nonzero(snr < thr).squeeze(1)
        if len(low) == 0: return 0
        low = low[torch.argsort(snr[low])][: int(max_frac * items.N)]
        res = ((data_t - model_img) / sig_t).numpy()
        sm = np.maximum(gaussian_filter(res, 1.0), 0)
        sm_c = gaussian_filter((data_t - model_img).numpy(), 1.0)
        p = sm.ravel() / sm.sum()
        pick = rng.choice(n * n, size=len(low), p=p)
        py, px = pick // n, pick % n
        items.x[low] = torch.tensor(px + rng.uniform(0, 1, len(low)), dtype=torch.float32)
        items.y[low] = torch.tensor(py + rng.uniform(0, 1, len(low)), dtype=torch.float32)
        l1 = 0.5 * math.log(1.0 - FLOOR ** 2)
        items.la[low] = l1; items.lb[low] = l1; items.th[low] = 0.0
        items.lf[low] = torch.tensor(np.log(np.maximum(sm_c[py, px], 0.5) * 2 * math.pi * 1.5 ** 2), dtype=torch.float32)
        if items.use_latent: items.z[low] = 0.0
        for p0 in [items.x, items.y, items.la, items.lb, items.th, items.lf] + ([items.z] if items.use_latent else []):
            st = opt.state.get(p0, {})
            if 'exp_avg' in st:
                st['exp_avg'][low] = 0; st['exp_avg_sq'][low] = 0
    return len(low)


def fit(datas, budget_items, psi, train_psi, adaptive, target='noisy', time_limit=210.0, seed=0, verbose=False,
        n_iter=None, init_items=None):
    """Fit items (one set per image in `datas`) and optionally psi. Returns items list, history, info."""
    use_latent = psi.mode == 'learned'
    items_list = []
    for k, d in enumerate(datas):
        init = lattice_init(d['n'], budget_items, use_latent, seed=seed + k)
        items_list.append(Items(init, use_latent))
    for p in psi.parameters(): p.requires_grad_(train_psi)
    opt = make_opt(items_list, psi, train_psi)
    tg = [torch.tensor(d[target], dtype=torch.float32) for d in datas]
    sg = [torch.tensor(d['sigma'], dtype=torch.float32) for d in datas]
    wt = [1.0 / s ** 2 for s in sg]
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time(); hist = []; moved = 0
    # learning-rate schedule by elapsed time (cosine from 1 to 0.05 over time_limit), or by iteration if n_iter given
    for gp in opt.param_groups: gp['lr0'] = gp['lr']
    reloc_at = [0.2, 0.35, 0.5]; it = 0
    while True:
        fr = (it / n_iter) if n_iter else (time.time() - t0) / time_limit
        if fr >= 1: break
        it += 1
        sc = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * fr))
        for gp in opt.param_groups: gp['lr'] = gp['lr0'] * sc
        opt.zero_grad()
        loss = 0.0; imgs = []
        for k, d in enumerate(datas):
            img = render(items_list[k], psi, d['n'])
            imgs.append(img.detach())
            loss = loss + ((img - tg[k]) ** 2 * wt[k]).mean() / len(datas)
        loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        if adaptive and reloc_at and fr >= reloc_at[0]:
            reloc_at.pop(0)
            for k in range(len(datas)):
                moved += relocate(items_list[k], opt, imgs[k], tg[k], sg[k], rng=rng)
        if verbose and it % 50 == 0:
            print(it, round(float(loss), 4), round(time.time() - t0, 1), flush=True)
    secs = time.time() - t0
    with torch.no_grad():
        finals = [render(items_list[k], psi, d['n']).numpy().astype(np.float64) for k, d in enumerate(datas)]
    info = dict(iterations=it, schedule='cosine over wall-clock time limit' if not n_iter else f'cosine over {n_iter} iterations',
                final_loss=hist[-1], still_falling=still_falling(hist), relocated_total=int(moved), seconds=secs)
    return items_list, finals, info


def item_table(items: Items):
    with torch.no_grad():
        w1, w2 = items.widths()
        d = dict(x=items.x.numpy().tolist(), y=items.y.numpy().tolist(), logsize=(0.5 * torch.log(w1 * w2)).numpy().tolist(),
                 w1=w1.numpy().tolist(), w2=w2.numpy().tolist(), th=items.th.numpy().tolist(), flux=torch.exp(items.lf).numpy().tolist())
        if items.use_latent: d['z'] = items.z.numpy().tolist()
    return d


def size_stats(items: Items):
    with torch.no_grad():
        w1, w2 = items.widths(); ls = 0.5 * torch.log(w1 * w2).numpy(); f = torch.exp(items.lf).numpy()
    q = np.percentile(np.exp(ls), [5, 50, 95]).tolist()
    return dict(size_px_p5_p50_p95=[round(v, 3) for v in q], frac_items_at_floor=float(np.mean(np.exp(ls) < 0.5)),
                flux_total=float(f.sum()))


PSI_FILE = os.path.join(HERE, 'results', 'profiles_psi_joint_seeds0-1-2.pt')
CFG = dict(R_SUP=R_SUP, RHO0=RHO0, T_KNOTS=T_KNOTS, M_HARM=M_HARM, D_LAT=D_LAT, HIDDEN=HIDDEN,
           subsampling='o=clip(ceil(3/min width),1,7) per axis', psi_table=f'{K_GRID}x{K_GRID} bilinear over [-{R_SUP},{R_SUP}]^2', relocation='S/N<1, at most 5% of items, at 20/35/50% of iterations', width_floor=FLOOR, lr=LR, lr_psi=LR_PSI)


def run_single(method, budget, spec_seed, zoom, target, time_limit, verbose, save_items=False):
    """T1 (learned psi trained on this image, counted in the budget), T2/T3 frozen psi, fixed Gaussian."""
    data = common.make_data(common.make_spec(spec_seed), n=256, zoom=zoom)
    variant = f'seed{spec_seed}_zoom{zoom}'
    adaptive = method.endswith('adaptive')
    if method.startswith('profile_frozen'):
        psi = Psi('learned'); psi.load_state_dict(torch.load(PSI_FILE)); train = False
        shared_in_budget = 0; nsh = 0
    elif method.startswith('profile_fixedgauss'):
        psi = Psi('gauss'); train = False; shared_in_budget = 0; nsh = 0
    else:
        psi = Psi('learned'); train = True; shared_in_budget = psi.n_weights(); nsh = psi.n_weights()
    N = n_items_for(budget, psi.mode == 'learned', shared_in_budget)
    items, imgs, info = fit([data], N, psi, train, adaptive, target, time_limit, verbose=verbose)
    it = items[0]
    extra = dict(info, n_items=N, per_item=it.per_item(), psi_mode=psi.mode, psi_weights=Psi('learned').n_weights(),
                 psi_trained_on=('this image' if train else ('seeds 0,1,2 (frozen)' if 'frozen' in method else 'fixed Gaussian')),
                 config=CFG, **size_stats(it))
    if method.startswith('profile_frozen'): extra['frozen_psi_weights_not_in_budget'] = psi.n_weights()
    r = common.save_result(method, budget, variant, imgs[0], data, n_params_image=it.n_params(), n_params_shared=nsh,
                           fit_seconds=info['seconds'], target=target, extra=extra)
    if save_items:
        with open(os.path.join(HERE, 'results', f'profiles_items__{method}__{budget}__{variant}__{target}.json'), 'w') as f:
            json.dump(item_table(it), f)
    print(method, budget, variant, target, 'chi2_truth', round(r['chi2_truth'], 4), 'psnr', round(r['psnr'], 2),
          'iters', info['iterations'], 'falling', info['still_falling'], 's', round(info['seconds'], 1), flush=True)
    return r


def run_joint(budget, time_limit, verbose):
    """T2 step 1: train psi jointly on skies 0, 1, 2 (each with its own items at this image budget)."""
    datas = [common.make_data(common.make_spec(s), n=256, zoom=1.0) for s in (0, 1, 2)]
    psi = Psi('learned')
    N = n_items_for(budget, True, 0)
    items, imgs, info = fit(datas, N, psi, True, adaptive=True, target='noisy', time_limit=time_limit, verbose=verbose)
    torch.save(psi.state_dict(), PSI_FILE)
    sc = [common.score(imgs[k], datas[k])['chi2_truth'] for k in range(3)]
    meta = dict(info, budget_items_per_sky=budget, n_items_per_sky=N, chi2_truth_per_sky=sc, psi_weights=psi.n_weights(), config=CFG)
    with open(PSI_FILE.replace('.pt', '.json'), 'w') as f: json.dump(meta, f, indent=1)
    print('joint psi', meta, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--budget', type=int, default=15000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--zoom', type=float, default=1.0)
    ap.add_argument('--target', default='noisy')
    ap.add_argument('--time_limit', type=float, default=200.0)
    ap.add_argument('--save_items', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if a.method == 'joint':
        run_joint(a.budget, a.time_limit, a.verbose)
    else:
        run_single(a.method, a.budget, a.seed, a.zoom, a.target, a.time_limit, a.verbose, a.save_items)
