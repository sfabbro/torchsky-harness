"""Comparator: the representation and fitting recipe of "Spectro-Spatial Survey Fusion" (Vassilakis, Cranmer & Belokurov,
arXiv:2611.21798), re-implemented for calibrated space-telescope exposures and scored by protocol v1.

It is an independent re-implementation written from the paper and from a reading of the authors' released snapshot
(github.com/GeorgeVassilakis/survey-fusion-paper-code, Apache-2.0, commit ac581e2; no code was copied or executed). Settings quoted
below are theirs unless marked ADAPTED.

  sky        F(x, lambda) = sum_j F_j G_j(x) s_j(lambda): anisotropic Gaussians (centre, two log-sigmas, angle), flux = softplus(raw);
             s_j(lambda) = softplus(c_j) . T(ln lambda - shift_j): 32 templates = softplus(raw) on 96 knots uniform in ln lambda over the
             union of the bandpasses, ends pinned to their neighbours, linear in wavelength between knots, zero outside.
  instrument GIVEN and held: header astrometry; a constant background per exposure subtracted beforehand; the point-spread function of
             each (detector, filter) as a SIGNED mixture of Gaussians fitted to a supplied stamp that already holds the pixel response
             (their fit: Adam on means / log-sigmas / angles with the weights re-solved by constrained ridge least squares).
  render     closed form at pixel centres: each sky Gaussian convolved with each PSF term through the exposure's local affine map;
             pairs kept when the pixel is within 18 native pixels plus 8 sigma of the component.
  loss       Gaussian with the variance recomputed from the current model and held in the gradient, the data variance where
             |residual| > 10 sigma; no prior.
  fit        batches of 10,000 pixels drawn by importance (5% uniform), exact reweighting; Adam (0.9, 0.999, 1e-8) on the rows touched;
             clone / split / prune every 100 steps from step 600, growth until step 15,000 with the threshold frozen at the 70th
             percentile of the first event; 11 epochs.

ADAPTED, each stated in the log:
  * the sky is in MJy/sr and arcsec on the tangent plane of the box (theirs: nanomaggies, a normalised plane);
  * one PSF mixture per (detector, filter), no variation across the detector; --psf-terms terms (theirs: 4 eigenimages x 10);
  * the variance model: their CCD model needs gain and read noise; here var(mu) = ERR^2 + a_e (mu - d) with a_e the slope of ERR^2 against
    the data of exposure e, floored at ERR^2 / 4;
  * the background is the exposure's median inside the box (theirs: the survey's sky model). With --fit-background 1 a constant per
    exposure is fitted on top of it (found necessary on SMACS: cuts of one box cover different parts of a cluster, so their medians are
    not one level, and the equal-time run of 23 Nov scored 2.8 to 3.5 on faint pixels for that reason alone);
  * the starting number of components and the cap follow the size of the field (--n-init, --max-components); theirs were 100,000 and 10
    million for 2 square degrees;
  * step sizes that their code expresses in units of the field are kept at their value in native pixels (position: 0.1 pixel);
  * band averages by a 64-node quadrature of the photon-weighted curve (theirs: the exact integral of the piecewise-linear spectrum);
  * no spectra are fitted (an images-only run, which their code also supports).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

PIX_SUPPORT = 18.0  # culling radius [native pixels]
N_SIGMA = 8.0
NQ = 64


def inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float64)
    return (y + torch.log(-torch.expm1(-y))).float()


# ----------------------------------------------------------------------------------------------------------------- PSF as a signed mixture
def cov3(log_s, th):
    e1, e2 = torch.exp(2 * log_s[..., 0]), torch.exp(2 * log_s[..., 1])
    c, s = torch.cos(th), torch.sin(th)
    return c * c * e1 + s * s * e2, c * s * (e1 - e2), s * s * e1 + c * c * e2


def gauss2(dx, dy, a, b, d):
    det = a * d - b * b
    q = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
    return torch.exp(-0.5 * q) / (2 * math.pi * torch.sqrt(det))


def fit_psf_mixture(stamp, pitch, k, steps=3500, seeds=(41, 83, 127), max_offset=10.0, max_sigma=5.0, min_sigma=0.25, half=25.0, dev="cpu"):
    """Signed Gaussian mixture of a PSF stamp (unit sum; `pitch` native pixels per sample; centred). Their recipe: Adam lr 0.025 with a cosine
    to 10%, gradient clip 10, weights by ridge least squares constrained to the stamp's integral, three seeds, best kept. Offsets and sigmas
    are in native pixels. Returns weights (k,), means (k, 2), covariances (k, 3) [native pixels], relative L2 error."""
    n = stamp.shape[0]
    c = (n - 1) / 2
    g = (torch.arange(n, dtype=torch.float64) - c) * pitch
    keep = g.abs() <= half + 1e-9
    g = g[keep]
    st = torch.as_tensor(stamp, dtype=torch.float64)[keep][:, keep]
    st = st / st.sum() / pitch**2  # density per native pixel^2, unit integral on the kept region
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    X, Y, T = xx.flatten().to(dev), yy.flatten().to(dev), st.flatten().to(dev)
    tot = float(T.sum() * pitch**2)
    best = None
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        rad = torch.logspace(math.log10(0.6), math.log10(4.5), k, dtype=torch.float64)
        mean = (0.3 * torch.randn(k, 2, generator=gen, dtype=torch.float64)).to(dev).requires_grad_(True)
        ls = (torch.log(rad)[:, None].repeat(1, 2) + 0.05 * torch.randn(k, 2, generator=gen, dtype=torch.float64)).to(dev).requires_grad_(True)
        th = (math.pi * torch.rand(k, generator=gen, dtype=torch.float64)).to(dev).requires_grad_(True)
        opt = torch.optim.Adam([mean, ls, th], lr=0.025)

        def model():
            m = max_offset * torch.tanh(mean / max_offset)
            s = ls.clamp(math.log(min_sigma), math.log(max_sigma))
            a, b, d = cov3(s, th)
            B = gauss2(X[None] - m[:, :1], Y[None] - m[:, 1:], a[:, None], b[:, None], d[:, None])  # (k, P)
            # weights: ridge least squares with the sum fixed to the stamp's integral (their constrained_weights)
            G = B @ B.T + 1e-7 * torch.eye(k, dtype=B.dtype, device=B.device) * (B @ B.T).diagonal().mean()
            r = B @ T
            ones = torch.ones(k, dtype=B.dtype, device=B.device)
            Gi_r, Gi_1 = torch.linalg.solve(G, r), torch.linalg.solve(G, ones)
            w = Gi_r + Gi_1 * (tot - ones @ Gi_r) / (ones @ Gi_1)
            return w, m, (a, b, d), w @ B

        for it in range(steps):
            for gr in opt.param_groups:
                gr["lr"] = 0.025 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / steps)))
            w, m, cv, pred = model()
            loss = ((pred - T) ** 2).sum() / (T**2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mean, ls, th], 10.0)
            opt.step()
        with torch.no_grad():
            w, m, cv, pred = model()
            err = float(torch.sqrt(((pred - T) ** 2).sum() / (T**2).sum()))
        if best is None or err < best[3]:
            best = (w.detach().float().cpu(), m.detach().float().cpu(), torch.stack(cv, -1).detach().float().cpu(), err)
    return best


def stamp_fwhm(stamp, pitch):
    """FWHM [native pixels] of a centred stamp from the area above half maximum (a circle of equal area)."""
    s = np.asarray(stamp, np.float64)
    return 2.0 * math.sqrt(float((s >= 0.5 * s.max()).sum()) * pitch**2 / math.pi)


# ----------------------------------------------------------------------------------------------------------------- the field
class Field:
    def __init__(self, n, K, G, lam_lo, lam_hi, dev):
        self.dev, self.K, self.G = dev, K, G
        z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
        self.p = {"mu": z(n, 2), "log_s": z(n, 2), "theta": z(n), "raw_flux": z(n) - 4.0, "coef": torch.randn(n, K, device=dev), "shift": z(n)}
        self.l0, self.l1 = math.log(lam_lo), math.log(lam_hi)
        self.scale = 0.5 * (self.l1 - self.l0)  # ln(1 + z) = scale x shift, as in their code
        lam_n = torch.linspace(-1.0, 1.0, G)
        slopes = torch.linspace(-2.0, 2.0, K)
        shape = torch.exp(slopes[:, None] * lam_n[None])
        shape = shape / shape.mean(1, keepdim=True)
        self.raw_t = (inv_softplus(shape[:, 1:-1]) + 0.01 * torch.randn(K, G - 2)).to(dev).requires_grad_(True)
        self.knots = torch.exp(torch.linspace(self.l0, self.l1, G, device=dev))
        self.m = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.p.items()}

    @property
    def n(self):
        return self.p["mu"].shape[0]

    def templates(self):
        t = F.softplus(self.raw_t)
        return torch.cat([t[:, :1], t, t[:, -1:]], 1)  # ends pinned to their neighbours

    def band_amplitudes(self, coef, shift, lam_q, w_q):
        """(M, nb): photon-weighted band averages of the spectra of M components. lam_q, w_q (nb, NQ)."""
        vals = F.softplus(coef) @ self.templates()  # (M, G): each component's spectrum at the knots
        x = torch.log(lam_q)[None] - (self.scale * shift)[:, None, None]  # (M, nb, NQ): ln wavelength in the template frame
        u = (x - self.l0) / (self.l1 - self.l0) * (self.G - 1)
        g0 = u.floor().clamp(0, self.G - 2).long()
        lo, hi = self.knots[g0], self.knots[g0 + 1]
        w = ((torch.exp(x) - lo) / (hi - lo)).clamp(0.0, 1.0)  # linear in wavelength between log-spaced knots
        inside = (u >= 0) & (u <= self.G - 1)
        M = coef.shape[0]
        flat = g0.reshape(M, -1)
        v0 = torch.gather(vals, 1, flat).view_as(w)
        v1 = torch.gather(vals, 1, flat + 1).view_as(w)
        s = torch.where(inside, v0 * (1 - w) + v1 * w, torch.zeros_like(w))
        return (s * w_q[None]).sum(-1)


def cell_table(mu, reach, half, cell):
    """Cell lists: every component is listed in each cell its square of half side `reach` touches. Returns (sorted component ids, starts,
    counts, cells per axis)."""
    nc = int(math.ceil(2 * half / cell))
    lo = ((mu - reach[:, None] + half) / cell).floor().long().clamp(0, nc - 1)
    hi = ((mu + reach[:, None] + half) / cell).floor().long().clamp(0, nc - 1)
    span = (hi - lo + 1).amax(1)
    comp, cid = [], []
    for s in torch.unique(span).tolist():
        idx = torch.nonzero(span == s).squeeze(1)
        off = torch.arange(s, device=mu.device)
        cx = lo[idx, 0, None] + off[None]
        cy = lo[idx, 1, None] + off[None]
        okx, oky = cx <= hi[idx, 0, None], cy <= hi[idx, 1, None]
        cc = cy[:, :, None] * nc + cx[:, None, :]
        ok = oky[:, :, None] & okx[:, None, :]
        comp.append(idx[:, None, None].expand_as(cc)[ok])
        cid.append(cc[ok])
    comp, cid = torch.cat(comp), torch.cat(cid)
    order = torch.argsort(cid)
    counts = torch.bincount(cid, minlength=nc * nc)
    starts = torch.cumsum(counts, 0) - counts
    return comp[order], starts, counts, nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="SMACS")
    ap.add_argument("--split", default="dither")
    ap.add_argument("--box-id", type=int, default=0)
    ap.add_argument("--method-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-root", required=True)
    ap.add_argument("--svo-cache", required=True)
    ap.add_argument("--psf-lib", required=True, help="npz of effective PSF stamps: key '<DET>/<BAND>' -> (nodes, N, N), unit sum")
    ap.add_argument("--psf-oversample", type=int, default=4)
    ap.add_argument("--psf-terms", type=int, default=40)
    ap.add_argument("--psf-fit-steps", type=int, default=3500)
    ap.add_argument("--n-init", type=int, default=60000)
    ap.add_argument("--max-components", type=int, default=120000)
    ap.add_argument("--epochs", type=float, default=11.0)
    ap.add_argument("--budget-min", type=float, default=0.0, help="> 0: keep fitting after the epochs until this many minutes")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=1000, help="pixels per pair chunk")
    ap.add_argument("--fit-background", type=int, default=0, help="1: one constant per exposure is fitted with the field (start: the median), and a "
                    "held-out exposure's constant is the median of its residual. Not in their recipe; needed when exposures cover different "
                    "parts of the box, since the median of a cut then depends on what it covers")
    ap.add_argument("--max-pairs", type=int, default=1500000, help="a chunk with more (pixel, component) pairs is halved until it fits: bounds memory, changes no result")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-exposures", type=int, default=0)
    ap.add_argument("--synthetic-psf", type=float, default=0.0, help="tests: a Gaussian PSF of this sigma [native pixels] instead of --psf-lib")
    a = ap.parse_args()

    from torchsky.filter_profiles import bandpass_from_svo
    from torchsky.instrument import instrument_description
    from torchsky.map.exposure_fit import load_exposures
    from torchsky.map.exposure_fit._observations import observations_to_exposures

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "fit.log"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    t_all = time.perf_counter()
    log(__doc__.split("ADAPTED")[0].strip().splitlines()[0])
    sp = json.load(open(os.path.join(a.protocol_root, f"splits_{a.field}.json")))
    box = next(b for b in sp["boxes"] if int(b["id"]) == a.box_id)
    half = box["size_arcsec"] / 2 + 1.0
    split = sp["splits"][a.split]
    held, train = set(split["heldout"]), set(split["train"])
    ex = [e for e in sp["exposures"] if e["rootname"] in held | train and str(a.box_id) in {str(k) for k in e.get("boxes", {})}]
    if a.max_exposures > 0:
        ex = [e for e in ex if e["rootname"] in train][: a.max_exposures] + [e for e in ex if e["rootname"] in held][: max(2, a.max_exposures // 4)]
    by_root = {e["rootname"]: e for e in ex}
    curves = {}
    for f in sorted({e["filter"] for e in ex}):
        spec = bandpass_from_svo(band_name=f, facility="JWST", instrument="NIRCam", filter_id=f"JWST/NIRCam.{f}", prefer="any", cache_dir=a.svo_cache)
        curves[f] = (torch.as_tensor(np.asarray(spec["transmission_wavelength"], np.float64) / 1e4), torch.as_tensor(np.asarray(spec["transmission_throughput"], np.float64)))
    obs = load_exposures([e["path"] for e in ex], [instrument_description("NIRCAM").with_filter_curves(curves)], (box["ra"], box["dec"], half))
    for o in obs:
        root = next(r for r in by_root if str(o.header["TSNAME"]).startswith(r))
        o.header["TSROOT"] = root
        if root in held:
            o.header["TSSPLIT"] = "held"
    recs, _ = observations_to_exposures(obs, n_lambda=5)
    roots = [str(o.header["TSROOT"]) for o in obs]
    fit_stage(a, recs, obs, roots, half, dev, log)
    log(f"all done in {(time.perf_counter() - t_all) / 60:.1f} min")


def fit_stage(a, recs, obs, roots, half, dev, log, psf_stamps=None):
    """Everything after loading (separate so that a test can call it with synthetic records and PSF stamps)."""
    bands = sorted({r.bandpass.name for r in recs}, key=lambda f: recs[[q.bandpass.name for q in recs].index(f)].filt)
    nb = len(bands)
    bp = {r.bandpass.name: r.bandpass for r in recs}
    lam_lo = min(float(b.lam_um[b.T > 1e-3 * b.T.max()].min()) for b in bp.values())
    lam_hi = max(float(b.lam_um[b.T > 1e-3 * b.T.max()].max()) for b in bp.values())
    lam_q, w_q = [], []
    for f in bands:
        b = bp[f]
        keep = b.T > 1e-3 * b.T.max()
        lam = np.linspace(b.lam_um[keep].min(), b.lam_um[keep].max(), NQ)
        w = np.interp(lam, b.lam_um, b.T) * lam  # photon weight
        lam_q.append(lam)
        w_q.append(w / w.sum())
    lam_q = torch.tensor(np.stack(lam_q), dtype=torch.float32, device=dev)
    w_q = torch.tensor(np.stack(w_q), dtype=torch.float32, device=dev)

    # ---- PSF mixtures, one per (detector, band), carried to the sky by each exposure's mean Jacobian [arcsec / pixel]
    lib = None if psf_stamps is not None or a.synthetic_psf > 0 else np.load(a.psf_lib)
    mix, fwhm_arcsec = {}, []
    for r in recs:
        key = (r.det_name, r.bandpass.name)
        if key in mix:
            continue
        if psf_stamps is not None:
            st, pitch = psf_stamps[key], 1.0 / a.psf_oversample
        elif a.synthetic_psf > 0:
            g = (np.arange(-100, 101) / 4.0) ** 2
            st, pitch = np.exp(-0.5 * (g[:, None] + g[None]) / a.synthetic_psf**2), 0.25
        else:
            det = {"NRCBLONG": "NRCB5", "NRCALONG": "NRCA5"}.get(r.det_name, r.det_name)
            st, pitch = np.asarray(lib[f"{det}/{r.bandpass.name}"], np.float64).mean(0), 1.0 / a.psf_oversample
        w, m, cv, err = fit_psf_mixture(st / st.sum(), pitch, a.psf_terms, steps=a.psf_fit_steps, dev=dev)
        fw = stamp_fwhm(st, pitch)
        mix[key] = (w, m, cv, err, fw)
        log(f"PSF {key}: {a.psf_terms} signed Gaussians, relative L2 error {err:.4f} (their gate: median 0.015, maximum 0.06), FWHM {fw:.2f} native pixels, "
            f"negative weight fraction {float(w.clamp(max=0).abs().sum() / w.abs().sum()):.2f}")
    E = len(recs)
    Kp = a.psf_terms
    Pw, Pm, Pc = torch.zeros(E, Kp), torch.zeros(E, Kp, 2), torch.zeros(E, Kp, 3)
    pixscale, e_band = torch.zeros(E), torch.zeros(E, dtype=torch.long)
    for i, r in enumerate(recs):
        w, m, cv, err, fw = mix[(r.det_name, r.bandpass.name)]
        J = torch.as_tensor(np.asarray(r.extra["J"]), dtype=torch.float32)  # sky offset [arcsec] = J @ pixel offset
        S = torch.stack([torch.stack([cv[:, 0], cv[:, 1]], -1), torch.stack([cv[:, 1], cv[:, 2]], -1)], -2)  # (Kp, 2, 2)
        Ss = J @ S @ J.T
        Pw[i], Pm[i] = w, m @ J.T
        Pc[i] = torch.stack([Ss[:, 0, 0], Ss[:, 0, 1], Ss[:, 1, 1]], -1)
        pixscale[i] = float(abs(torch.linalg.det(J)) ** 0.5)
        e_band[i] = bands.index(r.bandpass.name)
        fwhm_arcsec.append(fw * float(pixscale[i]))
    Pw, Pm, Pc, pixscale, e_band = Pw.to(dev), Pm.to(dev), Pc.to(dev), pixscale.to(dev), e_band.to(dev)
    pix0 = float(pixscale.min())  # the model pixel: the finest native pixel
    fwhm_med = float(np.median(fwhm_arcsec))

    # ---- pixel tables (training exposures), background = the exposure's median in the box, variance model
    tr = [i for i, r in enumerate(recs) if r.split == "train"]
    cols = {k: [] for k in ("d", "err", "x", "y", "e", "alpha", "score")}
    bkg = torch.zeros(E)
    for i, r in enumerate(recs):
        v = r.valid
        d, s = r.d[v].float(), r.sig[v].float()
        bkg[i] = d.median()
        if i not in tr:
            continue
        A = torch.stack([torch.ones_like(d), d], 1).double()
        sol = torch.linalg.lstsq(A, (s.double() ** 2)[:, None]).solution[:, 0]
        alpha = float(sol[1].clamp(min=0.0))
        src = d - bkg[i]
        sc = ((src - src.median()) / s).clamp(min=0.0)
        cols["d"].append(src)
        cols["err"].append(s)
        cols["x"].append(r.X[v].float())
        cols["y"].append(r.Y[v].float())
        cols["e"].append(torch.full((d.numel(),), i, dtype=torch.long))
        cols["alpha"].append(torch.full((d.numel(),), alpha))
        # their sampler: exposure in proportion to its pixels, pixel by clipped significance with a 5% uniform share
        n_loc = d.numel()
        cols["score"].append(0.95 * sc / sc.sum().clamp(min=1e-30) + 0.05 / n_loc)
    T = {k: torch.cat(v).to(dev) for k, v in cols.items()}
    n_tot = T["d"].numel()
    counts = torch.tensor([c.numel() for c in cols["d"]], dtype=torch.float64)
    p_exp = torch.repeat_interleave(counts / counts.sum(), counts.long()).to(dev)
    n_loc = torch.repeat_interleave(counts, counts.long()).to(dev)
    prob = p_exp * T["score"].double()
    cdf = torch.cumsum(prob, 0)
    cdf = cdf / cdf[-1]
    isw = (1.0 / (n_loc * T["score"].double())).float()  # reweighting 1 / (n_local p)
    bkg = bkg.to(dev)
    log(f"{len(tr)} training exposures, {n_tot} fitted pixels, bands {bands}, spectral range {lam_lo:.3f}-{lam_hi:.3f} um, model pixel {pix0:.4f} arcsec, "
        f"median PSF FWHM {fwhm_med:.3f} arcsec")

    # ---- initial components from image peaks (their seeding: 7 x 7 local maxima, brightest first, merged within 3 pixels, widths 1, 2, 4, 8 pixels)
    per_exp = max(32, -(-a.n_init // max(len(tr), 1)))
    px, py, pv, pb = [], [], [], []
    for i in tr:
        r = recs[i]
        img = torch.where(r.valid, r.d.float() - float(bkg[i]), torch.full_like(r.d.float(), -1e30))
        mx = F.max_pool2d(img[None, None], 7, 1, 3)[0, 0]
        ok = (img >= mx) & r.valid
        vals = img[ok]
        top = torch.argsort(vals, descending=True)[:per_exp]
        px.append(r.X[ok][top].float())
        py.append(r.Y[ok][top].float())
        pv.append(vals[top])
        pb.append(torch.full((top.numel(),), int(e_band[i])))
    px, py, pv, pb = torch.cat(px), torch.cat(py), torch.cat(pv), torch.cat(pb)
    order = torch.argsort(pv, descending=True)
    px, py, pv, pb = px[order], py[order], pv[order], pb[order]
    cellq = torch.stack([(px / (3 * pix0)).round(), (py / (3 * pix0)).round()], 1).long()
    _, first = np.unique(cellq.numpy(), axis=0, return_index=True)  # one peak per 3-pixel cell, the brightest (sorted input)
    first = torch.as_tensor(np.sort(first))[: a.n_init]
    px, py, pv, pb = px[first], py[first], pv[first].clamp(min=1e-9), pb[first]
    n_pk = px.numel()
    n0 = a.n_init
    fld = Field(n0, 32, 96, lam_lo, lam_hi, dev)
    assign = torch.arange(n0) % n_pk
    occ = torch.arange(n0) // n_pk
    sig0 = pix0 * 2.0 ** (occ % 4).float()
    jit = torch.randn(n0, 2) * sig0[:, None] * (occ > 0).float()[:, None]
    with torch.no_grad():
        fld.p["mu"].copy_((torch.stack([px, py], 1)[assign] + jit).clamp(-half, half).to(dev))
        fld.p["log_s"].copy_(torch.log(sig0)[:, None].repeat(1, 2).to(dev))
        amp = fld.band_amplitudes(fld.p["coef"], fld.p["shift"], lam_q, w_q)  # (n0, nb)
        dens = 1.0 / (2 * math.pi * sig0**2)
        tot = torch.zeros(n_pk).index_add_(0, assign, (dens * amp[torch.arange(n0), pb[assign]].cpu()))
        fld.p["raw_flux"].copy_(inv_softplus((pv / tot.clamp(min=1e-30))[assign].clamp(min=1e-9)).to(dev))
    log(f"seeded {n0} components on {n_pk} peaks ({per_exp} per exposure before merging)")

    # ---- optimiser settings (theirs; position in native pixels)
    lr = {"mu": 0.1 * pix0, "log_s": 5e-3, "theta": 1e-3, "raw_flux": 5e-2, "coef": 2.5e-3, "shift": 2.5e-3 / 20}
    opt_t = torch.optim.Adam([fld.raw_t], lr=2.5e-3, betas=(0.9, 0.999), eps=1e-8)
    steps_epoch = n_tot // a.batch
    n_steps = int(a.epochs * steps_epoch) if a.max_steps <= 0 else a.max_steps
    grow_scale = 3.0 * fwhm_med
    prune_scale = 636.0 * pix0
    g_acc, g_cnt = torch.zeros(fld.n, device=dev), torch.zeros(fld.n, device=dev)
    thr = None
    table = None
    band_sig = torch.stack([T["err"][e_band[T["e"]] == b].median() for b in range(nb)])
    R_e = PIX_SUPPORT * pixscale  # (E,) culling radius [arcsec]
    R_max = float(R_e.max())
    SLACK, REBUILD, CELL = 10 * 0.1 * pix0, 10, 0.5

    # optional: one background constant per exposure, in units of its band's median error, fitted by Adam with the field
    db = torch.zeros(E, device=dev, requires_grad=bool(a.fit_background))
    db_scale = band_sig[e_band]
    opt_b = torch.optim.Adam([db], lr=2e-2) if a.fit_background else None

    SPLIT = "split"  # render's answer when a chunk holds more pairs than --max-pairs: the caller halves the chunk
    stat = {"pairs_max": 0, "halved": 0}

    def pieces(ii):
        """Render the pixels ii with gradients, in as many pieces as --max-pairs needs. The loss is a sum over pixels and gradients are
        accumulated by the caller, so the pieces change nothing but the memory."""
        stack = [ii]
        while stack:
            cur = stack.pop()
            out = render(T["x"][cur], T["y"][cur], T["e"][cur], Kp, True)
            if out[1] is SPLIT:
                h = cur.numel() // 2
                stack += [cur[h:], cur[:h]]
                stat["halved"] += 1
                continue
            yield cur, out[0], out[1], out[2]

    def render(idx_pix_x, idx_pix_y, e_ids, P, with_grad):
        """Source model at pixels of sky positions (x, y) in exposures e_ids, with parameters P gathered for the rows `rows`.
        Returns mu (n,), and the rows used with the leaves to differentiate."""
        nonlocal table
        cid = ((idx_pix_y + half) / CELL).floor().long().clamp(0, table[3] - 1) * table[3] + ((idx_pix_x + half) / CELL).floor().long().clamp(0, table[3] - 1)
        cnt = table[2][cid]
        tot_c = int(cnt.sum())
        if idx_pix_x.numel() > 1 and tot_c > 8 * a.max_pairs:  # candidates before culling: too many to even list
            return None, SPLIT, None
        mu_out = torch.zeros(idx_pix_x.numel(), device=dev)
        if tot_c == 0:
            return mu_out, None, None
        pi = torch.repeat_interleave(torch.arange(cid.numel(), device=dev), cnt)
        within = torch.arange(tot_c, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        cj = table[0][table[1][cid][pi] + within]
        with torch.no_grad():
            dx, dy = idx_pix_x[pi] - fld.p["mu"][cj, 0], idx_pix_y[pi] - fld.p["mu"][cj, 1]
            reach = N_SIGMA * torch.exp(fld.p["log_s"][cj]).amax(1) + R_e[e_ids[pi]]
            ok = dx * dx + dy * dy <= reach * reach
        pi, cj = pi[ok], cj[ok]
        if pi.numel() == 0:
            return mu_out, None, None
        if idx_pix_x.numel() > 1 and pi.numel() > a.max_pairs:
            return None, SPLIT, None
        stat["pairs_max"] = max(stat["pairs_max"], pi.numel())
        rows, inv = torch.unique(cj, return_inverse=True)
        leaves = {k: v[rows].detach().requires_grad_(with_grad) for k, v in fld.p.items()}
        a_, b_, d_ = cov3(leaves["log_s"], leaves["theta"])
        amp = F.softplus(leaves["raw_flux"])[:, None] * fld.band_amplitudes(leaves["coef"], leaves["shift"], lam_q, w_q)  # (M, nb)
        e = e_ids[pi]
        dx = idx_pix_x[pi] - leaves["mu"][inv, 0]
        dy = idx_pix_y[pi] - leaves["mu"][inv, 1]
        val = torch.zeros(pi.numel(), device=dev)
        for t0 in range(0, P, 8):  # PSF terms in groups of 8 to bound memory
            sl = slice(t0, min(t0 + 8, P))
            m, c = Pm[:, sl][e], Pc[:, sl][e]  # slice the terms before gathering by exposure: memory is pairs x 8 terms, not pairs x all terms
            g = gauss2(dx[:, None] - m[..., 0], dy[:, None] - m[..., 1], a_[inv, None] + c[..., 0], b_[inv, None] + c[..., 1], d_[inv, None] + c[..., 2])
            val = val + (Pw[:, sl][e] * g).sum(1)
        contrib = val * amp[inv, e_band[e]]
        mu_out = mu_out.index_add(0, pi, contrib)
        return mu_out, rows, leaves

    def rebuild():
        nonlocal table
        with torch.no_grad():
            reach = N_SIGMA * torch.exp(fld.p["log_s"]).amax(1) + R_max + SLACK
            table = cell_table(fld.p["mu"], reach, half, CELL)

    def resize(keep=None, add=None):
        """Apply a population change: keep (index of survivors) then append rows `add` (dict of tensors); moments of new rows are zero."""
        nonlocal g_acc, g_cnt
        for k in fld.p:
            pk, mk, vk = fld.p[k], fld.m[k], fld.v[k]
            if keep is not None:
                pk, mk, vk = pk[keep], mk[keep], vk[keep]
            if add is not None:
                pk = torch.cat([pk, add[k]])
                mk = torch.cat([mk, torch.zeros_like(add[k])])
                vk = torch.cat([vk, torch.zeros_like(add[k])])
            fld.p[k], fld.m[k], fld.v[k] = pk, mk, vk
        g_acc, g_cnt = torch.zeros(fld.n, device=dev), torch.zeros(fld.n, device=dev)

    rebuild()
    t0 = time.perf_counter()
    step, hist = 0, []
    budget = a.budget_min * 60.0
    while True:
        el = time.perf_counter() - t0
        if step >= n_steps and (budget <= 0 or el >= budget):
            break
        if step % REBUILD == 0:
            rebuild()
        draw = torch.searchsorted(cdf, torch.rand(a.batch, device=dev, dtype=torch.float64)).clamp(max=n_tot - 1)
        grads = {k: torch.zeros_like(v) for k, v in fld.p.items()}
        touched = torch.zeros(fld.n, dtype=torch.bool, device=dev)
        opt_t.zero_grad()
        if opt_b is not None:
            opt_b.zero_grad()
        loss_sum, chi_sum = 0.0, 0.0
        for ii, mu, rows, leaves in (piece for c0 in range(0, a.batch, a.chunk) for piece in pieces(draw[c0 : c0 + a.chunk])):
            d, err = T["d"][ii], T["err"][ii]
            if opt_b is not None:
                mu = mu + (db * db_scale)[T["e"][ii]]
            with torch.no_grad():
                var_m = (err**2 + T["alpha"][ii] * (mu - d)).clamp(min=0.25 * err**2)
                var = torch.where((d - mu).abs() > 10.0 * err, err**2, var_m)
            nll = 0.5 * (d - mu) ** 2 / var + 0.5 * torch.log(2 * math.pi * var)
            loss = (isw[ii] * nll).sum() / a.batch
            chi_sum += float((((d - mu) / err) ** 2).sum().detach())
            loss_sum += float(loss.detach())
            if rows is None:
                if opt_b is not None:
                    gb = torch.autograd.grad(loss, [db])[0]
                    db.grad = gb if db.grad is None else db.grad + gb
                continue
            gl = torch.autograd.grad(loss, list(leaves.values()) + [fld.raw_t] + ([db] if opt_b is not None else []), allow_unused=True)
            if opt_b is not None:
                gl, gb = gl[:-1], gl[-1]
                if gb is not None:
                    db.grad = gb if db.grad is None else db.grad + gb
            for k, g in zip(leaves, gl[:-1]):
                grads[k].index_add_(0, rows, g)
            if gl[-1] is not None:
                fld.raw_t.grad = gl[-1] if fld.raw_t.grad is None else fld.raw_t.grad + gl[-1]
            touched[rows] = True
        # density-control statistic: position gradient per model pixel, summed over visits (before any clipping)
        if step < 15000:
            gn = (grads["mu"] * pix0).norm(dim=1)
            g_acc += torch.where(touched, gn, torch.zeros_like(gn))
            g_cnt += touched.float()
        torch.nn.utils.clip_grad_norm_([fld.raw_t], 1.0)
        opt_t.step()
        if opt_b is not None:
            opt_b.step()
        rows = torch.nonzero(touched).squeeze(1)
        pos_decay = 0.01 ** (min(step, 30000) / 30000.0)
        bc1, bc2 = 1 - 0.9 ** (step + 1), 1 - 0.999 ** (step + 1)
        with torch.no_grad():
            for k in fld.p:
                g = grads[k][rows]
                fld.m[k][rows] = 0.9 * fld.m[k][rows] + 0.1 * g
                fld.v[k][rows] = 0.999 * fld.v[k][rows] + 0.001 * g * g
                stepk = lr[k] * (pos_decay if k == "mu" else 1.0)
                fld.p[k][rows] -= stepk * (fld.m[k][rows] / bc1) / ((fld.v[k][rows] / bc2).sqrt() + 1e-8)
        step += 1
        if step > 500 and step % 100 == 0:
            with torch.no_grad():
                n_before = fld.n
                sig = torch.exp(fld.p["log_s"])
                flux = F.softplus(fld.p["raw_flux"])
                ampb = fld.band_amplitudes(fld.p["coef"], fld.p["shift"], lam_q, w_q)
                var_psf = (fwhm_med / 2.355) ** 2 + pix0**2 / 12
                peak = flux[:, None] * ampb / (2 * math.pi * torch.sqrt((sig[:, 0] ** 2 + var_psf) * (sig[:, 1] ** 2 + var_psf)))[:, None]
                dead = (peak / band_sig[None]).amax(1) < 1e-8
                n_clone = n_split = 0
                add = None
                keep = ~dead
                if step < 15000:
                    mean_g = g_acc / g_cnt.clamp(min=1)
                    if thr is None:
                        vis = mean_g[g_cnt > 0]
                        thr = float(torch.quantile(vis[torch.randperm(vis.numel(), device=dev)[:1000000]], 0.7)) if vis.numel() else float("inf")
                        log(f"growth threshold frozen at {thr:.4e} (70th percentile of the mean position gradient at step {step})")
                    high = (mean_g > thr) & (g_cnt > 0) & ~dead
                    room = a.max_components - int(keep.sum())
                    if int(high.sum()) > room:  # cap: the largest gradients are kept
                        hi_idx = torch.nonzero(high).squeeze(1)
                        sel = hi_idx[torch.topk(mean_g[hi_idx], max(room, 0)).indices]
                        high = torch.zeros_like(high)
                        high[sel] = True
                    small = sig.amax(1) <= grow_scale
                    cl, spl = high & small, high & ~small
                    n_clone, n_split = int(cl.sum()), int(spl.sum())
                    half_flux = inv_softplus((0.5 * flux).clamp(min=1e-30).cpu()).to(dev)
                    new = {k: [] for k in fld.p}
                    if n_clone:
                        fld.p["raw_flux"][cl] = half_flux[cl]
                        for k in fld.p:
                            new[k].append(fld.p[k][cl].clone())
                    if n_split:
                        a_, b_, d_ = cov3(fld.p["log_s"][spl], fld.p["theta"][spl])
                        Lc = torch.linalg.cholesky(torch.stack([torch.stack([a_, b_], -1), torch.stack([b_, d_], -1)], -2))
                        for _ in range(2):
                            ch = {k: fld.p[k][spl].clone() for k in fld.p}
                            ch["mu"] = ch["mu"] + torch.einsum("nij,nj->ni", Lc, torch.randn(n_split, 2, device=dev))
                            ch["log_s"] = ch["log_s"] - math.log(1.6)
                            ch["raw_flux"] = half_flux[spl]
                            for k in fld.p:
                                new[k].append(ch[k])
                        keep = keep & ~spl
                    if n_clone or n_split:
                        add = {k: torch.cat(v) for k, v in new.items()}
                resize(torch.nonzero(keep).squeeze(1), add)
                wide = torch.exp(fld.p["log_s"]).amax(1) > prune_scale
                if bool(wide.any()):
                    resize(torch.nonzero(~wide).squeeze(1))
                rebuild()
            if step % 1000 == 0 or n_clone or n_split:
                log(f"refine step {step}: {n_before} -> {fld.n} components (cloned {n_clone}, split {n_split}, dead {int(dead.sum())})")
        if step % 250 == 0 or step == 1:
            el = time.perf_counter() - t0
            hist.append([step, el, loss_sum, chi_sum / a.batch, fld.n])
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            log(f"step {step:7d} of {n_steps} t {el:8.1f}s loss {loss_sum:.4f} chi2/px(batch, pipeline error) {chi_sum / a.batch:.3f} components {fld.n} "
                f"({el / step:.3f} s/step) largest piece {stat['pairs_max']} pairs, pieces halved {stat['halved']}, cuda peak {peak:.2f} GB")
            stat["pairs_max"] = 0
    fit_s = time.perf_counter() - t0
    if opt_b is not None:
        q = torch.quantile(db.detach()[torch.as_tensor(tr, device=dev)], torch.tensor([0.05, 0.5, 0.95], device=dev)).tolist()
        log(f"fitted background constants of the training exposures, in units of the band's median error: 5/50/95% {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}")
    log(f"fit done: {step} steps ({step / max(steps_epoch, 1):.1f} epochs) in {fit_s / 60:.1f} min, {fld.n} components")
    torch.save({"p": {k: v.cpu() for k, v in fld.p.items()}, "raw_t": fld.raw_t.detach().cpu(), "bands": bands, "lam": [lam_lo, lam_hi], "bkg": bkg.cpu(),
                "bkg_fitted": (db.detach() * db_scale).cpu()},
               os.path.join(a.out, "state.pt"))

    # ---- held-out exposures: the fixed field through the exposure's own PSF and astrometry, plus its own median background
    recs_out = {}
    pdir = os.path.join(a.protocol_root, "pred", a.field, a.method_name, a.split)
    os.makedirs(pdir, exist_ok=True)
    rebuild()
    for i, r in enumerate(recs):
        if r.split != "held":
            continue
        v = r.valid
        xs, ys = r.X[v].float().to(dev), r.Y[v].float().to(dev)
        out = torch.zeros(xs.numel(), device=dev)
        with torch.no_grad():
            stack = [(c0, min(c0 + a.chunk, xs.numel())) for c0 in range(0, xs.numel(), a.chunk)]
            while stack:
                lo, hi = stack.pop()
                e_ids = torch.full((hi - lo,), i, dtype=torch.long, device=dev)
                res = render(xs[lo:hi], ys[lo:hi], e_ids, Kp, False)
                if res[1] is SPLIT:
                    stack += [(lo, (lo + hi) // 2), ((lo + hi) // 2, hi)]
                    continue
                out[lo:hi] = res[0]
        b_i = float((r.d[v].float() - out.cpu()).median()) if a.fit_background else float(bkg[i])
        m = torch.full(r.d.shape, float("nan"))
        m[v] = out.cpu() + b_i
        chi = float((((r.d[v].float() - m[v]) / r.sig[v].float()) ** 2).mean())
        ny, nx = r.det_shape
        y0, x0 = r.cut0
        full = np.full((ny, nx), np.nan, np.float32)
        val = np.zeros((ny, nx), bool)
        full[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = m.numpy()
        val[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = np.isfinite(m.numpy())
        meta = {"method": a.method_name, "fitted_on_heldout": ["background constant (median of the residual)" if a.fit_background else "background constant (median)"], "chi2": chi}
        np.savez_compressed(os.path.join(pdir, f"{roots[i]}.npz"), model=full, model_valid=val, meta=json.dumps(meta))
        recs_out[roots[i]] = {"chi2": chi, "band": r.bandpass.name}
        log(f"held-out {roots[i]} {r.bandpass.name}: chi2/N (pipeline error, own mask) {chi:.3f}")
    json.dump({"args": vars(a), "steps": step, "epochs": step / max(steps_epoch, 1), "fit_seconds": fit_s, "components": fld.n, "fitted_pixels": n_tot,
               "psf_fit": {f"{k[0]}/{k[1]}": {"rel_l2": v[3], "fwhm_pix": v[4]} for k, v in mix.items()}, "history": hist, "heldout": recs_out,
               "cuda_peak_gb": (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=str)
    return fld


if __name__ == "__main__":
    main()
