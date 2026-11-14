"""Shared loading code for the independent check of the learned kernel of the fit fieldop_smacs_fix_200_s0.

Only the model DEFINITION is taken from the package that made the fit (/home/claude/arc/fieldop_latest/fieldop: network classes, the
level quadrature used for the package's own normalisation, the spectral basis). Every measurement (grids, integrals, profiles, renders,
photometry) is written here and in kernel_check.py, independently of the earlier agent's code.

Conventions read from the package (model.py, kernel.py, render.py, sky.py, bandpass.py):
  k(u) = exp(l(v, h)) * T(|u|) / Z,  v = R_e^T u (detector-aligned frame, arcsec), T = C1 taper from 8.5" to 10" (kernel extent 10"),
  Z = sum over the nested level quadrature of exp(l) * window * pitch^2  (unit integral on that quadrature, approximation F7).
  c(x) = int k(u) s(x + u) du  (a correlation),  model = sum_q w_q exp(gamma_q) c_q + background  (input level 'cal': MJy/sr).
"""
import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
sys.path.insert(0, "/home/claude/arc/fieldop_latest")
from fieldop.kernel import Codes, KernelNet, LogPolarEncoding, lam_feats  # noqa: E402
from fieldop.render import Levels, kernel_level_values, smooth_down  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = "/home/claude/work/e2_learned/dl/fieldop_state.pt"
FILTERS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
DETS = ["NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG"]
PIX = {0: 0.031, 1: 0.063}          # nominal NIRCam pixel scales [arcsec]; the run used header medians (not saved) - approximation
RENDER_PITCH = {0: 0.005, 1: 0.01}  # from fieldop.log (sky pitch 0.04 / 8 and / 4)
EXTENT = 10.0
SKY_PITCH = 0.04
BOX = dict(ra=110.8268, dec=-73.4544, size=20.48)
# published NIRCam PSF FWHM (JWST user documentation, optical PSF without pixel), arcsec
PUB_FWHM = {"F090W": 0.033, "F150W": 0.050, "F200W": 0.066, "F277W": 0.092, "F356W": 0.116, "F444W": 0.145}


class Fit:
    """The saved state with the kernel network, the code network and the throughput network rebuilt from the package classes."""

    def __init__(self):
        s = torch.load(STATE, map_location="cpu", weights_only=False)
        self.s = s
        enc = LogPolarEncoding(r0=0.5 * min(RENDER_PITCH.values()), r_max=EXTENT, n_rad=8, m_max=12)
        self.net = KernelNet(enc, dh=16, hidden=64, layers=3)
        self.net.load_state_dict({k[4:]: v for k, v in s.items() if k.startswith("net.")}, strict=True)
        self.codes = Codes(6, 1, 5, 123, s["codes.curves"], "curve", dh=16, hidden=64)
        self.codes.load_state_dict({k[6:]: v for k, v in s.items() if k.startswith("codes.")}, strict=True)
        self.net.eval()
        self.codes.eval()
        self.tau = {k[8:]: float(F.softplus(v)) for k, v in s.items() if k.startswith("tau.raw.")}
        self.e_filt = s["e_filt"].numpy()
        self.e_chan = s["e_chan"].numpy()
        self.e_det = s["e_det"].numpy()
        self.R = s["e_R"]
        self.A = s["skies"][0]["A"][0]            # (K, N, N) coefficients (one view; pre already folded? see sky_coeffs)
        self.pre = s["skies"][0]["pre"]
        self.N = self.A.shape[-1]
        self.levels = {c: Levels(RENDER_PITCH[c], 2, 24, EXTENT, a=0.5, taper=0.85) for c in (0, 1)}

    # ---------------------------------------------------------------- sky
    def sky_coeffs(self):
        """SkyGrid.coeffs(): stored parameter A is coefficients / pre (sky.py), so coefficients = A * pre."""
        return self.A * self.pre

    def band_image(self, f):
        """Band image on the sky nodes = sum_k Phi_band[f, k] A_k  (MJy/sr, equal-weight mean over the 5 wavelength nodes, gamma = 0)."""
        return torch.einsum("k,khw->hw", self.s["Phi_band"][f], self.sky_coeffs())

    def node_xy(self):
        """Tangent-plane coordinates of the sky nodes [arcsec]: X to the WEST (north-up pixel x axis, CD1_1 < 0), Y to the north."""
        g = (torch.arange(self.N, dtype=torch.float64) - (self.N - 1) / 2) * SKY_PITCH
        return g  # X of column j, Y of row i

    # ---------------------------------------------------------------- codes and kernel
    def ref_lattice_xi(self, det):
        """3x3 reference lattice of a detector (training footprint), from the state."""
        m = self.s["ref_det"].numpy() == det
        return self.s["ref_xi"][m]

    def code(self, f, det=None, xi=(0.0, 0.0), e=None, lam=None):
        """h (1, 16): pivot code (+ tau_lambda term) + tau_space * normalised detector code + tau_exp * z_e (if e given)."""
        with torch.no_grad():
            ft, it = torch.tensor([f]), torch.tensor([0])
            hp = self.codes.hbar_at(self.s["piv"][f].reshape(1), ft, it)
            h = hp
            if lam is not None:
                h = h + self.tau["k_lambda"] * (self.codes.hbar_at(torch.tensor([float(lam)]), ft, it) - hp)
            if det is not None:
                ref = self.codes.hdet_raw(self.s["ref_det"], self.s["ref_xi"])
                mu = ref.mean(0)
                sc = torch.sqrt(((ref - mu) ** 2).sum(-1).mean() / ref.shape[-1] + 1e-12)
                hd = (self.codes.hdet_raw(torch.tensor([det]), torch.tensor([list(map(float, xi))])) - mu) / sc
                h = h + self.tau["k_space"] * hd
            if e is not None:
                h = h + self.tau["k_exp"] * self.codes.z[e][None]
        return h

    def logk(self, v, h, chunk=200000):
        """l(v, h) (unnormalised log density) at detector-frame offsets v (P, 2) arcsec."""
        out = []
        with torch.no_grad():
            for i in range(0, v.shape[0], chunk):
                out.append(self.net(self.net.enc(v[i:i + chunk].float()), h)[0].double())
        return torch.cat(out)

    def logZ_levels(self, h, chan, R):
        """Package normalisation: log of the sum over the nested level quadrature (render.kernel_level_values)."""
        with torch.no_grad():
            _, logZ = kernel_level_values(self.net, self.levels[chan], R.float(), h)
        return float(logZ[0])

    def blur_moment(self, f_ref=0, pm=0.005):
        """Core second moment of the mean kernel of the blur-reference filter (window of width 2 sqrt(floor)), averaged over its
        wavelength nodes, written here from the description of F18 (model.blur_moment). Returns (M, floor)."""
        floor = SKY_PITCH ** 2 / 3.0
        Rw = 2 * math.sqrt(floor)
        nh = int(math.ceil(4 * Rw / pm))
        g = torch.arange(-nh, nh + 1, dtype=torch.float64) * pm
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        u = torch.stack([xx.flatten(), yy.flatten()], -1)
        r2 = (u ** 2).sum(-1)
        Ms = []
        for q in range(self.s["lam"].shape[1]):
            h = self.code(f_ref, lam=float(self.s["lam"][f_ref, q]))
            lg = self.logk(u, h) - 0.5 * r2 / Rw ** 2
            w = torch.softmax(lg, 0)
            Ms.append(float((w * r2).sum()))
        return float(np.mean(Ms)), floor

    # ---------------------------------------------------------------- throughput (for absolute levels in Q4)
    def gamma(self, e):
        """log throughput at the 5 nodes for exposure e: gbar + tau_g_exp u_e - gauge mean m (detector term tau_g_space = 1e-6: dropped).
        Written from model.gamma_at / gauge_mean."""
        s = self.s
        with torch.no_grad():
            def mlp_apply(prefix, x):
                x = F.silu(F.linear(x, s[prefix + ".0.weight"], s[prefix + ".0.bias"]))
                x = F.silu(F.linear(x, s[prefix + ".2.weight"], s[prefix + ".2.bias"]))
                return F.linear(x, s[prefix + ".4.weight"], s[prefix + ".4.bias"])

            def gbar(lam, f):
                fin = F.linear(s["codes.curves"][f], s["g_fin.weight"], s["g_fin.bias"])
                inst = s["g_inst.weight"][0]
                x = torch.cat([lam_feats(lam), fin.expand(lam.numel(), -1), inst.expand(lam.numel(), -1)], -1)
                return mlp_apply("gbar", x)[:, 0]
            lamf = s["lam"].flatten()
            tg = self.tau.get("g_exp", 0.0)
            gref = s["gref"].numpy()
            G = torch.stack([gbar(lamf, int(self.e_filt[i])) + tg * s["u"][i] for i in gref])     # (n_ref, nL)
            Tw = s["T_ref"][s["e_filt"][s["gref"]]]
            den = Tw.sum(0)
            m = torch.where(den > 0, (Tw * G).sum(0) / den.clamp(min=1e-12), torch.zeros_like(den)).view(6, -1)
            f = int(self.e_filt[e])
            return gbar(s["lam"][f], f) + tg * s["u"][e] - m[f]
