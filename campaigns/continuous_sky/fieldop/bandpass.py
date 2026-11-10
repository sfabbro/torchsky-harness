"""Filter curves: lambda quadrature nodes (F4) and the curve INPUT vector of the kernel / throughput conditioning.

The lambda integral of the forward model, for filter f with published (SVO) transmission T_f:
    int dlambda A(lambda) [k * s](lambda)  with  A = alpha * g,  g(xi, lambda, t) = T_f(lambda) / (lambda N_f) * exp(gamma(xi, lambda, t))
N_f = int T_f dlambda / lambda (photon-counting f_nu normalisation: a flat f_nu sky gives the cal surface brightness when gamma = 0).
Quadrature: Q nodes at the mid-quantiles (q + 0.5)/Q of the density T_f(lambda) / (lambda N_f) over the filter support, weights 1/Q
(as v40 load_bandpasses).  The nodes span the support; the measure T/(lambda N) is carried by the node placement, exp(gamma) is
evaluated at the nodes.  A dense reference rule (many nodes) is available for convergence tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

CURVE_LO, CURVE_HI, CURVE_N = 0.55, 5.6, 48  # um: fixed log-lambda grid of the filter-curve input vector


@dataclass
class Filter:
    name: str
    instrument: str
    lam_um: np.ndarray        # curve wavelengths (sorted)
    T: np.ndarray             # transmission (>= 0)
    nodes: np.ndarray = field(default=None)   # (Q,) quadrature nodes [um]
    weights: np.ndarray = field(default=None)  # (Q,)
    pivot: float = 0.0        # photon-weighted mean of ln lambda, exponentiated
    curve_vec: np.ndarray = field(default=None)  # (CURVE_N,) T on the fixed grid, max-normalised

    def T_at(self, lam):
        return np.interp(lam, self.lam_um, self.T, left=0.0, right=0.0)


def quantile_nodes(lam, T, Q, support_frac=1e-3):
    lam = np.asarray(lam, np.float64)
    T = np.clip(np.asarray(T, np.float64), 0, None)
    keep = T >= support_frac * T.max()
    i0, i1 = np.argmax(keep), len(keep) - np.argmax(keep[::-1])
    lam, T = lam[max(i0 - 1, 0):min(i1 + 1, len(lam))], T[max(i0 - 1, 0):min(i1 + 1, len(lam))]
    dens = T / lam
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(lam))])
    cdf /= cdf[-1]
    qs = (np.arange(Q) + 0.5) / Q
    return np.interp(qs, cdf, lam), np.full(Q, 1.0 / Q)


def finish_filter(f: Filter, Q: int) -> Filter:
    o = np.argsort(f.lam_um)
    f.lam_um, f.T = np.asarray(f.lam_um, np.float64)[o], np.clip(np.asarray(f.T, np.float64)[o], 0, None)
    f.nodes, f.weights = quantile_nodes(f.lam_um, f.T, Q)
    dens = f.T / f.lam_um
    f.pivot = float(math.exp(np.trapezoid(dens * np.log(f.lam_um), f.lam_um) / np.trapezoid(dens, f.lam_um)))
    grid = np.exp(np.linspace(math.log(CURVE_LO), math.log(CURVE_HI), CURVE_N))
    v = f.T_at(grid)
    f.curve_vec = v / max(v.max(), 1e-12)
    return f


def box_filter(name, instrument, lo, hi, Q, edge=0.03, n=400):
    """Synthetic filter: flat top between lo and hi with smooth edges (tests)."""
    lam = np.linspace(lo * 0.85, hi * 1.15, n)
    T = 0.5 * (np.tanh((lam - lo) / (edge * lo)) - np.tanh((lam - hi) / (edge * hi)))
    return finish_filter(Filter(name, instrument, lam, T), Q)


def svo_filter(name, instrument, svo_id, Q, cache_dir=None):
    """Published curve from SVO through torchsky (cached on disk)."""
    from torchsky.filter_profiles import bandpass_from_svo
    fac, rest = svo_id.split("/", 1)
    ins, filt = rest.rsplit(".", 1)
    spec = bandpass_from_svo(band_name=filt, facility=fac, instrument=ins, filter_id=svo_id, prefer="any", cache_dir=cache_dir)
    lam = np.asarray(spec["transmission_wavelength"], np.float64) / 1e4
    T = np.asarray(spec["transmission_throughput"], np.float64)
    return finish_filter(Filter(name, instrument, lam, T), Q)


class FilterSet:
    """Ordered filters; tensors of nodes / weights / curve vectors / pivots for the model."""

    def __init__(self, filters):
        self.filters = list(filters)
        self.names = [f.name for f in self.filters]
        self.index = {f.name: i for i, f in enumerate(self.filters)}
        Q = {len(f.nodes) for f in self.filters}
        assert len(Q) == 1, "same Q for every filter"
        self.Q = Q.pop()

    def __len__(self):
        return len(self.filters)

    def tensors(self, dev):
        lam = torch.tensor(np.stack([f.nodes for f in self.filters]), dtype=torch.float32, device=dev)
        w = torch.tensor(np.stack([f.weights for f in self.filters]), dtype=torch.float32, device=dev)
        curve = torch.tensor(np.stack([f.curve_vec for f in self.filters]), dtype=torch.float32, device=dev)
        piv = torch.tensor([f.pivot for f in self.filters], dtype=torch.float32, device=dev)
        return lam, w, curve, piv

    def T_table(self, lam_flat):
        """T_f(lambda) for every filter f at the given wavelengths (normalised by each curve's max): (nf, M) numpy."""
        return np.stack([f.T_at(lam_flat) / max(f.T.max(), 1e-12) for f in self.filters])
