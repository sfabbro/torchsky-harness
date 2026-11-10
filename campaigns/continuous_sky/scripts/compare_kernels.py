#!/usr/bin/env python
"""Compare the effective kernels of the field operator (opfield.py) with the learned E2-data tables (smacs_four_v38.py), per (detector, band).

    python scripts/compare_kernels.py OPFIELD_RUN E2_RUN [--key learned] [--out DIR] [--rcmp 8] [--fmax 0.4] [--e2-file FILE]

Inputs
  OPFIELD_RUN/kernels/effective_kernels.npz + kernels.json   keys  "<instrument>|<det>|<band>|<roll>|<field>", field in
        grid (sky-frame raster), pos (sky-frame node offsets, target px), mass (band-integrated, unit sum), and, from the 7 Nov version on (regenerate
        an older run with `opfield.py --eval-only --init-state ... --args-from ...`):
        vdet (CANONICAL stencil coordinates, [:, 0] = column, [:, 1] = row: NOT detector px unless the learned Lambda = I), M = J Lambda (pos = M vdet),
        J (exposure detector -> target-px Jacobian; from the 7 Nov evening version on), massq (per-lambda_q masses), lamq, wq.
        TRUE detector-px node offsets = J^-1 pos = Lambda vdet (this tool). Files without '|J' use J = s R(roll) with s = pixscale / target px (stated approximation).
  E2_RUN/explicit_lib_3d_psf_groups.npy   ['detector_frame']['DET/BAND'] -> {oversample (4), half_window_det_px, library, learned, table}: square
        arrays [row, col], mass per sample on a lattice of pitch 1/oversample detector px centred at N // 2 (unit sum for 'learned' and 'library').
  Both are DETECTOR-frame objects in detector pixels; the opfield kernel of each roll group is brought to the detector frame with J^-1 pos (no rotation of the
  sky-frame grid is needed or used). An earlier version used the canonical vdet as detector px: wrong by the learned Lambda (about 2.7x for SW).

What is computed per (detector, band) (rolls of the opfield kernel are averaged in the detector frame; their scatter is reported as the roll dependence)
  1. common lattice: the opfield core lattice (pitch p = core pitch, nodes at multiples of p, |v| <= R_cmp = min(core radius, --rcmp)); the E2 table is
     integrated over the SAME cells (exact box integral of its piecewise-constant mass, centred on the E2 core centroid so that the centroids coincide);
     both are renormalised to unit sum inside the window.
  2. the common blur (the declared gauge: sky (x) g with K (x) g^-1): ONE isotropic Gaussian variance s [px^2] per CHANNEL, fitted in Fourier space by
     weighted least squares ln|F_op / F_E2| = -2 pi^2 s f^2 over f <= --fmax cycles/px (weights |F_op|^2 |F_E2|^2, |F| > 0.1), pooled over all cells of the
     channel. s > 0: the field-operator kernels are broader than E2's by that variance (E2 (x) g(s) is compared); s < 0: op (x) g(|s|) is compared.
     A residual kernel difference is reported before and after the gauge fit: L1 = sum |dK| (fraction of the window mass), rms and max of dK relative to
     the E2 peak.
  3. enclosed energy of both (each about its own core centroid; nodes sorted by radius, linear interpolation of the cumulative mass: a log-polar ring
     is a step, the interpolation halves the step error), the ratio and difference.
  4. azimuthal Fourier amplitude |sum m_i exp(-i m phi_i)| / sum m_i for m = 2 and 6 in annuli 2-6, 6-12, 12-24 px (detector-frame azimuth, node masses
     of the opfield kernel, sample masses of the E2 table), with the finite-sample floor sqrt(sum m_i^2) / sum m_i; m = 6 is the hexagonal-pupil spike
     pattern, m = 2 the horizontal strut / elongation. The log-polar wings have 24 azimuths per ring: m < 12 is representable, spikes narrower than 15 deg are not.
  5. lambda dependence: per lambda_q the core rms radius (r <= R_cmp) of the opfield kernel and its log-log slope against lambda (diffraction: 1); the
     E2 value only where its dump has per-node tables ('tables_nodes' / 'lam_nodes_um', chromatic runs).

APPROXIMATIONS: the opfield node mass is the cell integral, so on its own lattice the comparison is exact in the core; beyond the core radius the nodes are
coarser than the E2 lattice and only integrated quantities (EE, azimuthal amplitudes) are compared. A single isotropic blur per channel cannot absorb
band-dependent differences (reported as the residual). Windowing at R_cmp truncates both kernels identically but the Fourier fit then sees the window
edge. The E2 table is treated as mass per sample cell (its use as a continuous interpolant in smacs_four_v38 adds a smoothing of order the oversampled pitch
squared / 12 that is NOT removed). Rolls are averaged with equal weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

PIXSCALE_DEFAULT = {"NIRCAM_SW": 0.0311, "NIRCAM_LW": 0.063, "WFC3_IR": 0.128, "ACS_WFC": 0.05, "WFC3_UVIS": 0.04}
EE_R = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
EE_W = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
AZ_ANN = ((2.0, 6.0), (6.0, 12.0), (12.0, 24.0))


def infer_channel(det, band=""):
    d = det.upper()
    if "LONG" in d:
        return "NIRCAM_LW"
    if d.startswith("NRC"):
        return "NIRCAM_SW"
    if d.startswith("WFC3_IR"):
        return "WFC3_IR"
    if d.startswith("ACS"):
        return "ACS_WFC"
    return "WFC3_UVIS" if d.startswith("WFC3") else "UNKNOWN"


# ------------------------------------------------------------------------------------------------ loading
def legacy_jacobian(roll_deg, pixscale_arcsec, target_arcsec):
    """APPROXIMATION (files without a saved '|J'): the exposure's detector -> target-pixel Jacobian taken as s R(theta), s = pixscale / target pixel,
    theta = the roll of the group key (rounded to 1 deg), parity +1 and no skew. Checked against the header Jacobian of the held-out dump of the
    7 Nov JWST run: columns agree to 0.3 % in scale and 0.2 deg in angle (JWST detectors: parity +1). Not valid for HST parity or a rolled-and-skewed WCS."""
    th = math.radians(roll_deg)
    s_ = pixscale_arcsec / target_arcsec
    return s_ * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def detector_offsets(pos, J):
    """TRUE detector-px node offsets from the sky-frame offsets (target px): v_det = J^-1 pos ( = Lambda x canonical stencil coordinates ).
    The stored 'vdet' is the CANONICAL stencil coordinate (before the learned Lambda), which is NOT detector px unless Lambda = I."""
    return pos @ np.linalg.inv(J).T


def load_opfield(run):
    z = np.load(os.path.join(run, "kernels", "effective_kernels.npz"), allow_pickle=False)
    meta = {}
    mp = os.path.join(run, "kernels", "kernels.json")
    kj = json.load(open(mp)) if os.path.exists(mp) else {}
    for r in kj.get("kernels", []):
        meta[r["group"]] = r
    tgt = float(kj.get("target_px_arcsec", 0.04))
    groups = defaultdict(dict)
    for k in z.files:
        base, field = k.rsplit("|", 1)
        groups[base][field] = z[k]
    out = defaultdict(list)
    legacy = 0
    for base, f in groups.items():
        inst, det, band, roll = base.split("|")
        if "vdet" not in f:
            legacy += 1
            continue
        m = meta.get(base, {})
        ps = m.get("pixscale_arcsec_per_det_px")
        if "J" in f:
            J, jsrc = f["J"].astype(np.float64), "saved"
        else:
            J, jsrc = legacy_jacobian(float(roll), ps or PIXSCALE_DEFAULT.get(m.get("channel") or infer_channel(det, band), 0.04), tgt), "roll+pixscale (approximation)"
        pos = f["pos"].astype(np.float64)
        Lc = np.linalg.inv(J) @ f["M"].astype(np.float64)       # the learned Lambda actually applied (M = J Lambda)
        out[(det, band)].append({"group": base, "instrument": inst, "roll": float(roll), "vcan": f["vdet"].astype(np.float64),
                                 "vdet": detector_offsets(pos, J), "J": J, "J_source": jsrc, "Lc": Lc, "mass": f["mass"].astype(np.float64),
                                 "massq": f.get("massq"), "lamq": f.get("lamq"), "wq": f.get("wq"), "channel": m.get("channel") or infer_channel(det, band),
                                 "pixscale": ps})
    if legacy and not out:
        raise SystemExit(f"{run}: the kernels file has no detector-frame keys (vdet): written by an older opfield.py. Regenerate with "
                         "`opfield.py --eval-only --init-state RUN/opfield_best.pt --args-from RUN/metrics.json --out NEWDIR ...` (same data / args).")
    return out, kj.get("stencil", {})


def load_e2(path, key):
    f = path if os.path.isfile(path) else os.path.join(path, "explicit_lib_3d_psf_groups.npy")
    if not os.path.exists(f):
        cand = [x for x in os.listdir(path) if x.endswith("_psf_groups.npy")]
        if len(cand) != 1:
            raise SystemExit(f"{path}: expected explicit_lib_3d_psf_groups.npy or exactly one *_psf_groups.npy, found {cand}")
        f = os.path.join(path, cand[0])
    g = np.load(f, allow_pickle=True).item()["detector_frame"]
    out = {}
    for lab, ent in g.items():
        det, band = lab.split("/", 1)
        if key not in ent:
            continue
        K = np.asarray(ent[key], np.float64)
        out[(det, band)] = {"K": K / K.sum(), "os": float(ent["oversample"]), "R": float(ent.get("half_window_det_px", (K.shape[0] // 2) / ent["oversample"])),
                            "lam": ent.get("lam_um"), "nodes": ent.get("tables_nodes"), "lam_nodes": ent.get("lam_nodes_um")}
    return out, f


# ------------------------------------------------------------------------------------------------ geometry helpers
def table_points(K, os_):
    N = K.shape[0]
    ax = (np.arange(N) - N // 2) / os_
    xx, yy = np.meshgrid(ax, ax)     # xx varies along the columns (axis 1), yy along the rows (axis 0)
    return np.stack([xx.ravel(), yy.ravel()], 1), K.ravel()


def centroid(pts, m, rmax=3.0):
    r = np.hypot(pts[:, 0], pts[:, 1])
    s = r <= rmax
    return (m[s, None] * pts[s]).sum(0) / m[s].sum()


class BoxTable:
    """Exact box integrals of a piecewise-constant mass table (summed-area table with bilinear interpolation of the cumulative)."""

    def __init__(self, K, os_):
        self.N, self.os = K.shape[0], float(os_)
        self.S = np.zeros((self.N + 1, self.N + 1))
        self.S[1:, 1:] = K.cumsum(0).cumsum(1)

    def _F(self, t, s):
        N = self.N
        t, s = np.clip(t, 0, N), np.clip(s, 0, N)
        i, j = np.minimum(np.floor(t).astype(int), N - 1), np.minimum(np.floor(s).astype(int), N - 1)
        ft, fs = t - i, s - j
        S = self.S
        return S[j, i] * (1 - ft) * (1 - fs) + S[j, i + 1] * ft * (1 - fs) + S[j + 1, i] * (1 - ft) * fs + S[j + 1, i + 1] * ft * fs

    def box(self, cx, cy, side):
        e = lambda u: u * self.os + self.N // 2 + 0.5  # noqa: E731
        x0, x1, y0, y1 = e(cx - side / 2), e(cx + side / 2), e(cy - side / 2), e(cy + side / 2)
        return self._F(x1, y1) - self._F(x0, y1) - self._F(x1, y0) + self._F(x0, y0)


def lattice_from_nodes(vdet, m, p, n, nc=None):
    """Dense (2n+1)^2 array [row, col] of the core-node masses on the DETECTOR-px lattice of pitch p whose origin is the physical position of node 0
    (the canonical stencil starts at the origin: lattice point (i, j) is at vdet[0] + p (i, j)). vdet = TRUE detector-px offsets (J^-1 pos).
    If Lambda = I the core nodes sit exactly on lattice points and the result is exact; otherwise the nodes (canonical lattice mapped by Lambda) are
    off-lattice and each mass is split bilinearly over the four neighbouring lattice points (mass-conserving; an extra smoothing of at most p^2/6
    per axis, absorbed by the common-blur gauge; a sparse comb if Lambda stretches the stencil beyond 1). Only the first nc (core) nodes are used."""
    nn_ = len(m) if nc is None else int(nc)
    v = (vdet[:nn_] - vdet[0][None]) / p
    mm = m[:nn_]
    A = np.zeros((2 * n + 1, 2 * n + 1))
    fx, fy = v[:, 0] + n, v[:, 1] + n
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    wx, wy = fx - x0, fy - y0
    used = 0
    for dxx, dyy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        xi, yi = x0 + dxx, y0 + dyy
        ok = (xi >= 0) & (xi <= 2 * n) & (yi >= 0) & (yi <= 2 * n)
        np.add.at(A, (yi[ok], xi[ok]), mm[ok] * w[ok])
        used += int(ok.sum())
    return A, used


def detect_core_pitch(v):
    r = np.hypot(v[:, 0], v[:, 1])
    xs = np.unique(np.round(v[r < 2.5, 0], 4))
    d = np.diff(xs)
    d = d[d > 1e-3]
    return float(d.min()) if len(d) else 1.0


def ee_curve(pts, m, radii):
    r = np.hypot(pts[:, 0], pts[:, 1])
    o = np.argsort(r, kind="stable")
    c = np.cumsum(m[o]) / m.sum()
    rs = r[o]
    # one value per distinct radius (ring): the cumulative AFTER the ring; interpolation between ring radii
    rr = np.round(rs, 6)
    idx = np.searchsorted(rr, np.unique(rr), side="right") - 1
    return np.interp(radii, np.concatenate([[0.0], rs[idx]]), np.concatenate([[0.0], c[idx]]))


def az_amp(pts, m, r1, r2, mm):
    r = np.hypot(pts[:, 0], pts[:, 1])
    s = (r >= r1) & (r < r2)
    if s.sum() < 8 or m[s].sum() <= 0:
        return None, None
    ph = np.arctan2(pts[s, 1], pts[s, 0])
    w = m[s]
    amp = abs((w * np.exp(-1j * mm * ph)).sum()) / w.sum()
    floor = math.sqrt((w ** 2).sum()) / w.sum()
    return float(amp), float(floor)


def core_rms(pts, m, rmax):
    r = np.hypot(pts[:, 0], pts[:, 1])
    s = r <= rmax
    w = m[s] / m[s].sum()
    return float(math.sqrt((w * r[s] ** 2).sum() / 2.0))


# ------------------------------------------------------------------------------------------------ Fourier gauge
def fft_pad(A, pad=2):
    n = A.shape[0]
    M = int(2 ** math.ceil(math.log2(n * pad)))
    F = np.fft.fft2(A, s=(M, M))
    f = np.fft.fftfreq(M)          # cycles per lattice step
    return F, f


def blur_centered(A, s_var, p):
    """As blur_grid but with the array centre as the origin (circular shift handled by zero padding on both sides)."""
    if s_var <= 0:
        return A
    n = A.shape[0]
    M = int(2 ** math.ceil(math.log2(n * 2)))
    P = np.zeros((M, M))
    o = (M - n) // 2
    P[o:o + n, o:o + n] = A
    F = np.fft.fft2(P)
    f = np.fft.fftfreq(M) / p
    G = np.exp(-2 * math.pi ** 2 * s_var * (f[None, :] ** 2 + f[:, None] ** 2))
    B = np.real(np.fft.ifft2(F * G))[o:o + n, o:o + n]
    return B


def fit_blur_terms(A_op, A_e2, p, fmax):
    """Normal-equation terms (sum w x y, sum w x^2, n) for ln|F_op/F_E2| = -2 pi^2 s f^2 from two lattice arrays (unit window sum)."""
    Fo, f = fft_pad(A_op)
    Fe, _ = fft_pad(A_e2)
    fx = f[None, :] / p
    fy = f[:, None] / p
    f2 = fx ** 2 + fy ** 2
    ao, ae = np.abs(Fo), np.abs(Fe)
    ok = (f2 <= fmax ** 2) & (f2 > 0) & (ao > 0.1) & (ae > 0.1)
    if ok.sum() < 5:
        return 0.0, 0.0, 0
    x = -2 * math.pi ** 2 * f2[ok]
    y = np.log(ao[ok] / ae[ok])
    w = (ao[ok] * ae[ok]) ** 2
    return float((w * x * y).sum()), float((w * x * x).sum()), int(ok.sum())


def diff_stats(A, B):
    d = A - B
    return {"L1_fraction_of_window_mass": float(np.abs(d).sum()), "rms_over_E2_peak": float(np.sqrt((d ** 2).mean()) / B.max()),
            "max_abs_over_E2_peak": float(np.abs(d).max() / B.max()), "peak_ratio_op_over_E2": float(A.max() / B.max())}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("opfield_run")
    ap.add_argument("e2_run")
    ap.add_argument("--key", default="learned", choices=("learned", "table", "library"))
    ap.add_argument("--e2-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rcmp", type=float, default=8.0, help="radius [detector px] of the lattice window (<= the opfield core radius)")
    ap.add_argument("--fmax", type=float, default=0.4, help="highest spatial frequency [cycles / detector px] of the blur fit")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    op, sten = load_opfield(a.opfield_run)
    e2, e2f = load_e2(a.e2_file or a.e2_run, a.key)
    out = a.out or f"compare_kernels_{os.path.basename(os.path.normpath(a.opfield_run))}_vs_{os.path.basename(os.path.normpath(a.e2_run))}"
    os.makedirs(out, exist_ok=True)
    cells = sorted(set(op) & set(e2))
    if not cells:
        raise SystemExit(f"no common (detector, band): opfield {sorted(op)[:6]}..., E2 {sorted(e2)[:6]}...")
    p_core = float(sten.get("core_pitch_det_px") or 0) or detect_core_pitch(np.concatenate([g["vcan"] for gs in op.values() for g in gs]))
    rc_core = float(sten.get("core_radius_det_px") or 0) or 3.0
    rcmp = min(a.rcmp, rc_core)
    n = int(round(rcmp / p_core))
    win = np.arange(-n, n + 1) * p_core
    wx, wy = np.meshgrid(win, win)
    wmask = np.hypot(wx, wy) <= n * p_core + 1e-9
    res = {"opfield_run": a.opfield_run, "e2_file": e2f, "e2_key": a.key, "core_pitch_det_px": p_core, "window_radius_det_px": n * p_core, "cells": {}, "channels": {},
           "only_opfield": sorted(set(op) - set(e2)), "only_e2": sorted(set(e2) - set(op))}
    prep = {}
    chan_terms = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for cell in cells:
        gs, ent = op[cell], e2[cell]
        chan = gs[0]["channel"]
        # detector-frame node masses averaged over rolls (equal weights); roll scatter from the core lattices
        lat_roll, ee_roll = [], []
        for g in gs:
            A, nn = lattice_from_nodes(g["vdet"], g["mass"], p_core, n, sten.get("core_points"))
            A = A * wmask
            lat_roll.append(A / A.sum())
            ee_roll.append(ee_curve(g["vdet"], g["mass"], EE_R))
        A_op = np.mean(lat_roll, 0)
        roll_scatter = float(np.sqrt(np.mean([np.sum((x - A_op) ** 2) for x in lat_roll]))) if len(lat_roll) > 1 else 0.0
        pts_e, m_e = table_points(ent["K"], ent["os"])
        c_e = centroid(pts_e, m_e)
        c_o = np.mean([centroid(g["vdet"], g["mass"]) for g in gs], 0)
        bt = BoxTable(ent["K"], ent["os"])
        v0 = np.mean([g["vdet"][0] for g in gs], 0)     # physical offset of the lattice origin from the zero-centroid (= -cbar)
        B_e2 = bt.box(wx + v0[0] + c_e[0], wy + v0[1] + c_e[1], p_core) * wmask
        B_e2 = B_e2 / B_e2.sum()
        t = fit_blur_terms(A_op, B_e2, p_core, a.fmax)
        ct = chan_terms[chan]
        ct[0] += t[0]; ct[1] += t[1]; ct[2] += t[2]; ct[3] += 1
        prep[cell] = dict(A_op=A_op, B_e2=B_e2, chan=chan, gs=gs, ent=ent, pts_e=pts_e, m_e=m_e, c_e=c_e, c_o=c_o, ee_op=np.mean(ee_roll, 0), roll_scatter=roll_scatter)
    s_chan = {}
    for chan, (num, den, nf, nc) in chan_terms.items():
        s = num / den if den > 0 else 0.0
        s_chan[chan] = s
        res["channels"][chan] = {"blur_variance_s_px2": s, "sigma_g_px": math.sqrt(abs(s)), "sign": "op broader than E2 (E2 blurred)" if s > 0 else "op sharper than E2 (op blurred)",
                                 "n_cells": nc, "n_fourier_points": nf}
    # per cell: gauge-fixed residual, EE, azimuth, lambda
    for cell in cells:
        P = prep[cell]
        s = s_chan[P["chan"]]
        A, B = P["A_op"], P["B_e2"]
        if s > 0:
            Ag, Bg = A, blur_centered(B, s, p_core)
        else:
            Ag, Bg = blur_centered(A, -s, p_core), B
        Ag, Bg = Ag * wmask, Bg * wmask
        Ag, Bg = Ag / Ag.sum(), Bg / Bg.sum()
        Sv = np.linalg.svd(P["gs"][0]["Lc"], compute_uv=False)
        r = {"channel": P["chan"], "n_rolls": len(P["gs"]), "J_source": P["gs"][0]["J_source"], "Lambda_singular_values": Sv.tolist(),
             "core_extent_det_px": float(Sv.max() * rc_core), "roll_scatter_L2": P["roll_scatter"], "core_centroid_op": P["c_o"].tolist(), "core_centroid_E2": P["c_e"].tolist(),
             "residual_raw": diff_stats(A, B), "residual_after_gauge": diff_stats(Ag, Bg),
             "core_rms_op_px": core_rms(P["gs"][0]["vdet"], P["gs"][0]["mass"], n * p_core), "core_rms_E2_px": core_rms(P["pts_e"] - P["c_e"], P["m_e"], n * p_core)}
        # radial profile of the gauge-fixed residual (lattice nodes)
        rr = np.hypot(wx, wy)[wmask]
        d = (Ag - Bg)[wmask]
        edges = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12])
        edges = edges[edges <= rr.max() + 1e-9]
        ib = np.clip(np.digitize(rr, edges) - 1, 0, len(edges) - 1)
        # EE of the gauge-fixed lattices INSIDE the comparison window (window-normalised: mass beyond the window radius is not counted in either kernel)
        rw = np.hypot(wx, wy)
        r["EE_after_gauge_window"] = {"r_det_px": EE_W.tolist(), "op": [float(Ag[rw <= x].sum() / Ag.sum()) for x in EE_W], "E2": [float(Bg[rw <= x].sum() / Bg.sum()) for x in EE_W],
                                      "window_radius_det_px": n * p_core}
        r["residual_radial"] = {"r_edges_px": edges.tolist(), "sum_dK_in_bin": [float(d[ib == k].sum()) for k in range(len(edges))],
                                "sum_abs_dK_in_bin": [float(np.abs(d[ib == k]).sum()) for k in range(len(edges))]}
        # EE (node-based, full radius) against the E2 table
        pts_c = P["pts_e"] - P["c_e"]
        ee_e = ee_curve(pts_c, P["m_e"], EE_R)
        ee_o = P["ee_op"]
        ps = P["gs"][0]["pixscale"] or PIXSCALE_DEFAULT.get(P["chan"], 0.04)
        r["pixscale_arcsec_per_det_px"] = ps
        r["EE"] = {"r_det_px": EE_R.tolist(), "r_arcsec": (EE_R * ps).tolist(), "op": ee_o.tolist(), "E2": ee_e.tolist(), "op_minus_E2": (ee_o - ee_e).tolist()}
        # azimuthal power
        az = {}
        for (r1, r2) in AZ_ANN:
            for mm in (2, 6):
                ao_, fo_ = az_amp(P["gs"][0]["vdet"], P["gs"][0]["mass"], r1, r2, mm)
                ae_, fe_ = az_amp(pts_c, P["m_e"], r1, r2, mm)
                az[f"m{mm}_r{r1:g}-{r2:g}"] = {"op": ao_, "op_floor": fo_, "E2": ae_, "E2_floor": fe_}
        r["azimuthal_amplitude"] = az
        # lambda dependence
        g0 = P["gs"][0]
        if g0["massq"] is not None and g0["lamq"] is not None:
            lam = np.asarray(g0["lamq"], float)
            sig = np.array([core_rms(g0["vdet"], g0["massq"][q].astype(float), n * p_core) for q in range(len(lam))])
            sl = float(np.polyfit(np.log(lam), np.log(sig), 1)[0]) if len(lam) > 2 and np.all(sig > 0) else None
            lam_e = {"lam_um": lam.tolist(), "core_rms_op_px": sig.tolist(), "loglog_slope_op": sl}
            if P["ent"]["nodes"] is not None and P["ent"]["lam_nodes"] is not None:
                ln_ = np.asarray(P["ent"]["lam_nodes"], float)
                se = np.array([core_rms(pts_c, np.asarray(t, float).ravel() / np.asarray(t, float).sum(), n * p_core) for t in P["ent"]["nodes"]])
                lam_e.update(lam_nodes_E2_um=ln_.tolist(), core_rms_E2_px=se.tolist(), loglog_slope_E2=float(np.polyfit(np.log(ln_), np.log(se), 1)[0]) if len(ln_) > 2 else None)
            r["lambda"] = lam_e
        res["cells"][f"{cell[0]}/{cell[1]}"] = r
    # per channel summary of the residual and of the azimuthal ratios
    for chan in res["channels"]:
        cs = [v for v in res["cells"].values() if v["channel"] == chan]
        res["channels"][chan]["mean_L1_raw"] = float(np.mean([c["residual_raw"]["L1_fraction_of_window_mass"] for c in cs]))
        res["channels"][chan]["mean_L1_after_gauge"] = float(np.mean([c["residual_after_gauge"]["L1_fraction_of_window_mass"] for c in cs]))
        res["channels"][chan]["mean_EE_op_minus_E2"] = np.mean([c["EE"]["op_minus_E2"] for c in cs], 0).tolist()
        res["channels"][chan]["EE_r_det_px"] = EE_R.tolist()
        res["channels"][chan]["EE_after_gauge_window"] = {"r_det_px": EE_W.tolist(), "op": np.mean([c["EE_after_gauge_window"]["op"] for c in cs], 0).tolist(),
                                                         "E2": np.mean([c["EE_after_gauge_window"]["E2"] for c in cs], 0).tolist()}
        res["channels"][chan]["EE_node_based"] = {"r_det_px": EE_R.tolist(), "op": np.mean([c["EE"]["op"] for c in cs], 0).tolist(), "E2": np.mean([c["EE"]["E2"] for c in cs], 0).tolist()}
        res["channels"][chan]["Lambda_singular_values_mean"] = np.mean([c["Lambda_singular_values"] for c in cs], 0).tolist()
        res["channels"][chan]["core_extent_det_px_mean"] = float(np.mean([c["core_extent_det_px"] for c in cs]))
        for key in ("m6_r2-6", "m6_r6-12", "m6_r12-24", "m2_r2-6", "m2_r6-12", "m2_r12-24"):
            vo = [c["azimuthal_amplitude"][key]["op"] for c in cs if c["azimuthal_amplitude"][key]["op"] is not None]
            ve = [c["azimuthal_amplitude"][key]["E2"] for c in cs if c["azimuthal_amplitude"][key]["E2"] is not None]
            fo = [c["azimuthal_amplitude"][key]["op_floor"] for c in cs if c["azimuthal_amplitude"][key]["op"] is not None]
            fe = [c["azimuthal_amplitude"][key]["E2_floor"] for c in cs if c["azimuthal_amplitude"][key]["E2"] is not None]
            if vo and ve:
                res["channels"][chan]["azimuthal_" + key] = {"op_mean": float(np.mean(vo)), "E2_mean": float(np.mean(ve)), "ratio": float(np.mean(vo) / max(np.mean(ve), 1e-12)),
                                                             "op_floor": float(np.mean(fo)), "E2_floor": float(np.mean(fe)),
                                                             "significant": bool(np.mean(vo) > 2 * np.mean(fo) or np.mean(ve) > 2 * np.mean(fe))}
    json.dump(res, open(os.path.join(out, "compare_kernels.json"), "w"), indent=1, default=float)
    if not a.no_plots:
        make_plots(res, prep, out, n, p_core, s_chan)
    for chan, c in res["channels"].items():
        print(f"{chan}: common blur s = {c['blur_variance_s_px2']:+.4f} px^2 (sigma {c['sigma_g_px']:.3f} px; {c['sign']}); mean L1 residual raw {c['mean_L1_raw']:.4f} -> after gauge {c['mean_L1_after_gauge']:.4f}; "
              f"mean EE(op - E2) at {EE_R[[1, 3, 6, 8, 10]].tolist()} px: {[round(c['mean_EE_op_minus_E2'][i], 4) for i in (1, 3, 6, 8, 10)]}")
        e_ = c["EE_after_gauge_window"]
        print(f"   Lambda singular values (mean) {np.round(c['Lambda_singular_values_mean'], 3).tolist()} => dense core covers r <= {c['core_extent_det_px_mean']:.2f} det px; EE after the gauge, inside the "
              f"{res['window_radius_det_px']:g} px window, at {e_['r_det_px']}: op {np.round(e_['op'], 3).tolist()} E2 {np.round(e_['E2'], 3).tolist()}")
        for key in ("m6_r2-6", "m6_r6-12", "m6_r12-24", "m2_r2-6", "m2_r6-12", "m2_r12-24"):
            if "azimuthal_" + key in c:
                v = c["azimuthal_" + key]
                print(f"   azimuthal {key}: op {v['op_mean']:.4f}  E2 {v['E2_mean']:.4f}  ratio {v['ratio']:.2f}  (floors {v['op_floor']:.3f} / {v['E2_floor']:.3f}"
                      + ("" if v["significant"] else "; both below 2x the floor: no signal") + ")")
    print(f"wrote {out}/compare_kernels.json")


def make_plots(res, prep, out, n, p, s_chan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = list(prep)[:8]
    fig, axs = plt.subplots(len(cells), 4, figsize=(11, 2.5 * len(cells)), squeeze=False)
    for i, c in enumerate(cells):
        P = prep[c]
        s = s_chan[P["chan"]]
        A, B = P["A_op"], P["B_e2"]
        Ag, Bg = (A, blur_centered(B, s, p)) if s > 0 else (blur_centered(A, -s, p), B)
        wm = np.hypot(*np.meshgrid(np.arange(-n, n + 1) * p, np.arange(-n, n + 1) * p)) <= n * p + 1e-9
        Ag, Bg = Ag * wm, Bg * wm
        Ag, Bg = Ag / Ag.sum(), Bg / Bg.sum()
        ext = [-(n + .5) * p, (n + .5) * p] * 2
        for j, (im, t, cm, vm) in enumerate(((Ag, "op" + (" (x) g" if s < 0 else ""), "magma", None), (Bg, "E2" + (" (x) g" if s > 0 else ""), "magma", None))):
            axs[i, j].imshow(np.log10(np.maximum(im, 1e-6)), origin="lower", extent=ext, cmap=cm, vmin=-5, vmax=np.log10(im.max()))
            axs[i, j].set_title(f"{c[0]}/{c[1]} {t} (log10)", fontsize=7)
        d = Ag - Bg
        v = np.abs(d).max() + 1e-12
        axs[i, 2].imshow(d, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
        axs[i, 2].set_title("op - E2 after gauge", fontsize=7)
        ee = res["cells"][f"{c[0]}/{c[1]}"]["EE"]
        axs[i, 3].plot(ee["r_det_px"], ee["op"], "o-", label="op")
        axs[i, 3].plot(ee["r_det_px"], ee["E2"], "s-", label="E2")
        axs[i, 3].set_xscale("log")
        axs[i, 3].set_title("EE (detector px)", fontsize=7)
        if i == 0:
            axs[i, 3].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kernels_core_ee.png"), dpi=100)
    plt.close(fig)
    ch = list(res["channels"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for c in ch:
        axs[0].plot(EE_R, res["channels"][c]["mean_EE_op_minus_E2"], "o-", label=c)
    axs[0].set_xscale("log")
    axs[0].axhline(0, color="grey", lw=0.5)
    axs[0].set_title("mean EE(op) - EE(E2) per channel", fontsize=8)
    axs[0].legend(fontsize=6)
    keys = [f"m{m}_r{r1:g}-{r2:g}" for (r1, r2) in AZ_ANN for m in (6, 2)]
    for k, c in enumerate(ch):
        cs = [v for v in res["cells"].values() if v["channel"] == c]
        yo = [np.nanmean([x["azimuthal_amplitude"][kk]["op"] if x["azimuthal_amplitude"][kk]["op"] is not None else np.nan for x in cs]) for kk in keys]
        ye = [np.nanmean([x["azimuthal_amplitude"][kk]["E2"] if x["azimuthal_amplitude"][kk]["E2"] is not None else np.nan for x in cs]) for kk in keys]
        axs[1].plot(range(len(keys)), yo, "o-", label=f"{c} op")
        axs[1].plot(range(len(keys)), ye, "s--", label=f"{c} E2")
    axs[1].set_xticks(range(len(keys)))
    axs[1].set_xticklabels(keys, rotation=60, fontsize=6)
    axs[1].set_yscale("log")
    axs[1].set_title("azimuthal amplitude |c_m|/c_0", fontsize=8)
    axs[1].legend(fontsize=5)
    for c, v in res["cells"].items():
        if "lambda" in v:
            axs[2].plot(v["lambda"]["lam_um"], v["lambda"]["core_rms_op_px"], "o-", label=c)
    axs[2].set_title("core rms radius vs lambda (op)", fontsize=8)
    if len(res["cells"]) <= 10:
        axs[2].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kernels_channel_summary.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
