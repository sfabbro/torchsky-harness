"""Tests for scripts/smacs_four_v33.py (roll-aware PSF: --psf-frame detector, --pixel-box). Run with the venv that has torchsky:
    python tests/test_v33.py [name ...]
(a) without the flags v33 is bit-identical to v32 (tests/check_v32_equiv.py: NIRCam cal files, three variants, stdout, cubes, states, npy, metrics)
(b) J_e estimation: synthetic exposures with known rotation / scale / parity / anisotropy; real HST samples vs ORIENTAT and PA_V3
(c) a rotationally asymmetric kernel rendered for theta = 0 and 90 degrees is the 90-degree-rotated image (all three PSF classes, box on and off)
(d) two-roll scene: a star seen at two rolls with a spiky detector-frame PSF; sky-frame kernel vs detector-frame kernel
(e) pixel box: delta source through a Gaussian PSF (sigma 0.5 detector px) with 3.2x undersampled pixels vs brute-force fine-grid integration
(f) end-to-end --quick run, NIRCam + two HST samples, --psf-frame detector --pixel-box --spectral rbf
(g) groups: unseen theta creates a group on the fly; prepare(groups=[id]) equals the slice of prepare(); refusals"""
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import smacs_four_v33 as S  # noqa: E402

CPU = torch.device("cpu")
SAMPLES = os.path.join(ROOT, "hst", "samples")
HST_ALL = os.path.join(ROOT, "testdata", "hst_all")
NIR = os.path.join(ROOT, "testdata", "nircam")
NIR_ALL = os.path.join(ROOT, "testdata", "nircam_all")
SVO = os.path.join(ROOT, "testdata", "svo_cache")
RA0, DEC0 = 110.8268, -73.4544


def have_torchsky():
    try:
        import torchsky  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def group(b, theta, sx=1.0, sy=1.0, parity=1, box=False, shear=0.0):
    """A PSF group dict like PSFGroups builds it."""
    M = [[sx, shear * sx], [shear * sy, sy]]
    return {"id": 0, "b": b, "band": S.REG.bands[b], "theta_deg": theta, "parity": parity, "M": M, "sx": sx, "sy": sy,
            "S": (max(1, int(math.ceil(2.0 * max(sx, sy)))) if box else 1)}


# ---------------------------------------------------------------------------------------------------- (a)
def test_a_equivalence():
    if not have_torchsky():
        print("(a) SKIPPED: torchsky not available")
        return
    out = os.path.join(HERE, "check_v32_equiv_output.txt")
    p = subprocess.run([sys.executable, os.path.join(HERE, "check_v32_equiv.py")], capture_output=True, text=True)
    open(out, "w").write(p.stdout + p.stderr)
    print("(a) v32 vs v33 without the flags (tests/check_v32_equiv.py; log tests/check_v32_equiv_output.txt):")
    for ln in p.stdout.splitlines():
        print("    " + ln[:230])
    assert p.returncode == 0 and "\nEQUIVALENT" in p.stdout, (p.stdout + p.stderr)[-3000:]


# ---------------------------------------------------------------------------------------------------- (b)
def synthetic_exposure(theta_deg, sx, sy, parity=1, shear=0.0, nrow=300, ncol=260, t0=(100.3, 77.1), noise=0.0, distort=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    rr, cc = torch.meshgrid(torch.arange(nrow), torch.arange(ncol), indexing="ij")
    row, col = rr.flatten().double(), cc.flatten().double()
    t = math.radians(theta_deg)
    R = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
    Mm = torch.tensor([[sx, shear * sx], [shear * sy, sy]], dtype=torch.float64)
    Fm = torch.diag(torch.tensor([1.0, float(parity)], dtype=torch.float64))
    J = R @ Mm @ Fm
    u = torch.stack([col - ncol / 2, row - nrow / 2], 1)
    xy = u @ J.T + torch.tensor(t0, dtype=torch.float64)
    xy = xy + distort * torch.stack([(u[:, 0] / 100) ** 2, (u[:, 1] / 100) ** 2 * 0.5], 1)
    if noise:
        xy = xy + noise * torch.randn(xy.shape, generator=g, dtype=torch.float64)
    e = {"row": row.long(), "col": col.long(), "tx": xy[:, 0].float(), "ty": xy[:, 1].float(), "d": torch.zeros(row.numel()), "name": "syn", "band": "F150W"}
    return e, J.numpy()


def test_b_jacobian_synthetic():
    cases = [(0.0, 1.0, 1.0, 1, 0.0), (37.25, 0.775, 0.775, 1, 0.0), (-120.7, 3.384, 3.020, 1, 0.002), (73.789, 1.241, 1.2396, 1, 0.039),
             (145.1, 0.775, 0.775, -1, 0.0), (-33.3, 3.2, 3.2, -1, 0.001), (179.9, 1.5, 1.5, 1, 0.0)]
    print("(b) J_e on synthetic exposures (float32 coordinates, 78 000 pixels):")
    worst_t, worst_s = 0.0, 0.0
    for th, sx, sy, par, sh in cases:
        e, J = synthetic_exposure(th, sx, sy, par, sh)
        j = S.estimate_jacobian(e)
        dth = (j["theta_deg"] - th + 180.0) % 360.0 - 180.0
        ds = abs(j["scale"] / math.sqrt(abs(np.linalg.det(J))) - 1.0)
        dsx, dsy = abs(j["sx"] / sx - 1.0), abs(j["sy"] / sy - 1.0)
        worst_t, worst_s = max(worst_t, abs(dth)), max(worst_s, ds, dsx, dsy)
        print(f"    theta {th:8.3f} -> {j['theta_deg']:8.3f} (err {dth:+.2e} deg) parity {par:+d}->{j['parity']:+d} scale rel err {ds:.1e} "
              f"sx {dsx:.1e} sy {dsy:.1e} shear {sh:.4f}->{j['shear']:.4f} fit rms {j['fit_rms_px']:.1e} px")
        assert abs(dth) < 0.01 and j["parity"] == par and ds < 1e-4 and dsx < 1e-4 and dsy < 1e-4, (th, dth, ds)
        assert abs(j["shear"] - sh) < 1e-3
    # centre-pixel noise and mild distortion: the linear fit still recovers rotation and scale
    e, J = synthetic_exposure(61.0, 3.2, 3.2, 1, 0.0, noise=0.05, distort=0.4)
    j = S.estimate_jacobian(e)
    print(f"    with 0.05 px coordinate noise + quadratic distortion: theta err {j['theta_deg'] - 61.0:+.4f} deg, scale rel err {j['scale'] / 3.2 - 1:+.1e}")
    assert abs(j["theta_deg"] - 61.0) < 0.05 and abs(j["scale"] / 3.2 - 1) < 2e-3
    print(f"    worst over the clean cases: theta {worst_t:.1e} deg (limit 0.01), scale {worst_s:.1e} (limit 1e-4)")


def test_b2_jacobian_real_hst():
    if not have_torchsky():
        print("(b2) SKIPPED: torchsky not available")
        return
    import hst_io
    from astropy.io import fits
    th = S.target_header(RA0, DEC0, 512, 0.04)
    files = sorted(glob.glob(os.path.join(HST_ALL, "*.fits")))
    ex = hst_io.load_hst_exposures(files, th, CPU, pad=24)
    print(f"(b2) J_e of the real HST samples ({len(ex)} chips of {len(files)} files, 512 px / 0.04 arcsec target grid, north up east left):")
    print(f"    {'chip':16s} {'theta':>8s} {'phi_y':>8s} {'ORIENTAT':>9s} {'PA_V3':>8s} {'th-ORI':>7s} {'phiy-ORI':>8s} {'th-PA_V3':>9s} {'sx':>6s} {'sy':>6s} {'shear':>7s} {'rms px':>7s}")
    rel = {}
    for e in ex:
        j = S.estimate_jacobian(e)
        J = np.array(j["J"])
        phi_y = math.degrees(math.atan2(-J[0, 1], J[1, 1]))  # direction of the detector +y axis, from target +y (north) towards target -x (east)
        ori = float(e["header"]["ORIENTAT"])
        pa = float(fits.getheader(e["path"], 0)["PA_V3"])
        d_ori = (j["theta_deg"] - ori + 180) % 360 - 180
        d_phy = (phi_y - ori + 180) % 360 - 180
        d_pa = (j["theta_deg"] - pa) % 360
        print(f"    {e['name']:16s} {j['theta_deg']:8.3f} {phi_y:8.3f} {ori:9.3f} {pa:8.3f} {d_ori:7.3f} {d_phy:8.3f} {d_pa:9.3f} {j['sx']:6.3f} {j['sy']:6.3f} {j['shear']:7.4f} {j['fit_rms_px']:7.3f}")
        assert j["parity"] == 1
        rel.setdefault(e["instrument"], []).append((d_ori, d_phy, d_pa))
        assert abs(d_phy) < 0.5, (e["name"], d_phy)  # the y-axis direction IS ORIENTAT
    for inst, v in rel.items():
        v = np.array(v)
        print(f"    {inst}: theta - ORIENTAT = {v[:, 0].mean():+.3f} (range {v[:, 0].min():+.3f}..{v[:, 0].max():+.3f}) deg; "
              f"y-axis angle - ORIENTAT within {np.abs(v[:, 1]).max():.3f} deg; theta - PA_V3 = {v[:, 2].mean():.3f} (range {v[:, 2].min():.3f}..{v[:, 2].max():.3f}) mod 360")
    assert all(abs(x[0]) < 0.2 for x in rel["WFC3/IR"]) and all(abs(x[2] - 44.73) < 0.3 for x in rel["WFC3/IR"])
    assert all(abs(x[2] - 180.1) < 0.4 for x in rel["ACS/WFC"])
    print("    relation: theta_e = angle of the detector +y axis east of north (= ORIENTAT) for orthogonal detectors (WFC3/IR: +0.08 deg); "
          "for the skewed ACS/WFC axes theta_e is the polar-decomposition rotation, i.e. the mean of the x- and y-axis angles = ORIENTAT + half the "
          "axis skew (+2.0 / +2.6 deg for chips 1 / 2, whose y axes are at ORIENTAT within 0.3 deg); theta_e - PA_V3 = 44.73 deg (WFC3/IR) and 180.1 deg (ACS/WFC), "
          "constants of the instrument, parity +1 (target and detector have the same handedness) for all")
    nir = S.load_exposures(NIR_ALL, th, CPU, pad=24)
    print("    NIRCam module B:")
    for e in nir:
        j = S.estimate_jacobian(e)
        pa = float(fits.getheader(e["path"], "SCI")["PA_V3"])
        print(f"      {e['name'][-22:]:24s} {e['band']} theta {j['theta_deg']:8.3f} PA_V3 {pa:8.3f} theta-PA_V3 {(j['theta_deg'] - pa + 180) % 360 - 180:+.3f} scale {j['scale']:.4f} parity {j['parity']:+d}")
        assert j["parity"] == 1 and abs((j["theta_deg"] - pa + 180) % 360 - 180) < 1.0


# ---------------------------------------------------------------------------------------------------- (c)
def _chromatic(nb_bands=("F150W", "F444W")):
    S.set_registry(S.build_registry(list(nb_bands)))
    torch.manual_seed(3)
    psf = S.ChromaticPSF(S.REG.bands, 1, 0.04, nk=4, n=31)
    with torch.no_grad():  # make the field strongly asymmetric: random output layer, random first-layer weights
        for p in psf.kern.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return psf


def test_c_rotation_of_kernels():
    S.PSFG = S.PSFGroups(0.5, False)
    print("(c) kernel rendered at theta = 90 deg vs the 90-degree-rotated theta = 0 image (max |diff| / max |kernel|):")
    psfs = {"ChromaticPSF": _chromatic()}
    S.set_registry(S.build_registry(["F150W", "F444W"]))
    torch.manual_seed(4)
    ev = S.ExplicitVarPSF(torch.tensor([1.5, 2.0]), 1, 4, n=31)
    ev.res.data = 0.02 * torch.randn_like(ev.res)
    psfs["ExplicitVarPSF"] = ev
    iv = S.ImplicitVarPSF(2, 1, 4, n=31)
    with torch.no_grad():
        for p in iv.kern.parameters():
            p.add_(0.3 * torch.randn_like(p))
    psfs["ImplicitVarPSF"] = iv
    for name, psf in psfs.items():
        for box in (False, True):
            S.PSFG = S.PSFGroups(0.5, box)
            sx = sy = 3.2
            g0 = [group(0, 0.0, sx, sy, 1, box), group(1, 0.0, sx, sy, 1, box)]
            g90 = [group(0, 90.0, sx, sy, 1, box), group(1, 90.0, sx, sy, 1, box)]
            with torch.no_grad():
                B0, B90 = S.psf_group_kernels(psf, g0), S.psf_group_kernels(psf, g90)
            err = float((B90 - torch.rot90(B0, k=-1, dims=(-2, -1))).abs().max() / B0.abs().max())
            asym = float((B0 - torch.rot90(B0, k=-1, dims=(-2, -1))).abs().max() / B0.abs().max())
            print(f"    {name:15s} box {'on ' if box else 'off'}: |K90 - rot90(K0)| = {err:.1e}   (asymmetry of K0 itself {asym:.2f}; S = {g0[0]['S']})")
            assert err < 1e-5 and asym > 0.05, (name, box, err, asym)
            # parity: theta 0, parity -1 is the row-flipped kernel
            gm = [group(0, 0.0, sx, sy, -1, box), group(1, 0.0, sx, sy, -1, box)]
            with torch.no_grad():
                Bm = S.psf_group_kernels(psf, gm)
            errp = float((Bm - B0.flip(-2)).abs().max() / B0.abs().max())
            assert errp < 1e-5, (name, "parity", errp)
    # theta = 0, no box reproduces the unrotated basis
    S.PSFG = S.PSFGroups(0.5, False)
    with torch.no_grad():
        e0 = float((S.psf_group_kernels(psfs["ChromaticPSF"], [group(0, 0.0), group(1, 0.0)]) - psfs["ChromaticPSF"].basis()).abs().max())
        e0e = float((S.psf_group_kernels(psfs["ExplicitVarPSF"], [group(0, 0.0), group(1, 0.0)]) - psfs["ExplicitVarPSF"].basis()).abs().max())
    print(f"    theta = 0, no box vs psf.basis(): ChromaticPSF {e0:.1e}, ExplicitVarPSF {e0e:.1e}")
    assert e0 < 1e-6 and e0e < 1e-6
    # spectral path: per-node kernels, rotated rows
    psf = psfs["ChromaticPSF"]
    lam = torch.tensor([[1.3, 1.5, 1.7], [4.0, 4.4, 4.8]])
    with torch.no_grad():
        Ba = S.psf_group_kernels(psf, [group(0, 0.0), group(1, 0.0)], lam)
        Bb = S.psf_group_kernels(psf, [group(0, 90.0), group(1, 90.0)], lam)
    errs = float((Bb - torch.rot90(Ba, k=-1, dims=(-2, -1))).abs().max() / Ba.abs().max())
    print(f"    spectral rows (basis_lam, 2 bands x 3 nodes): |K90 - rot90(K0)| = {errs:.1e}, shape {tuple(Ba.shape)}")
    assert errs < 1e-5 and Ba.shape[0] == 6
    # a generic angle: unit sum of the core, finite, rotation back approximately restores (bilinear, ExplicitVarPSF) / exactly (MLP)
    with torch.no_grad():
        Bg = S.psf_group_kernels(psfs["ChromaticPSF"], [group(0, 33.0), group(1, -71.5)])
        Be = S.psf_group_kernels(psfs["ExplicitVarPSF"], [group(0, 33.0), group(1, -71.5)])
    print(f"    generic angles: core sums ChromaticPSF {Bg[:, 0].sum((-2, -1)).tolist()}, ExplicitVarPSF {Be[:, 0].sum((-2, -1)).tolist()}, "
          f"zero-sum residual kernels max |sum| {float(Bg[:, 1:].sum((-2, -1)).abs().max()):.1e} / {float(Be[:, 1:].sum((-2, -1)).abs().max()):.1e}")
    assert torch.isfinite(Bg).all() and torch.isfinite(Be).all()


# ---------------------------------------------------------------------------------------------------- (d)
def spiky(xi, scale=1.0):
    """Analytic rotationally asymmetric detector-frame PSF (unnormalised): Gaussian core + four spikes of different length/amplitude."""
    x, y = xi[..., 0], xi[..., 1]
    f = torch.exp(-0.5 * (x ** 2 + y ** 2) / 3.0 ** 2)
    for (ex, ey, amp, L) in ((1, 0, 0.16, 9.0), (0, 1, 0.10, 6.0), (-1, 0, 0.06, 4.0), (0, -1, 0.13, 8.0)):
        al = x * ex + y * ey
        pe = -x * ey + y * ex
        f = f + amp * torch.exp(-0.5 * pe ** 2 / 1.5 ** 2) * torch.exp(-al.clamp(min=0) / L) * (al > 0).float() * torch.exp(-0.5 * (al.clamp(max=0) / 1.0) ** 2)
    return f


def true_kernel(n=31):
    r = torch.arange(n, dtype=torch.float32) - n // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    K = spiky(torch.stack([xx, yy], -1))
    return K / K.sum()


def render_scene(psf, groups, sky, data):
    out = []
    for gdef, e in zip(groups, data):
        B = S.psf_group_kernels(psf, [gdef])  # 1 nk n n
        CB = S.convolve_basis(sky, B)[0]  # nk H W
        out.append(S.sample(CB, e["x"], e["y"])[0])
    return out


def two_roll_scene(rolls=(0.0, 70.0), npix=41, flux=7000.0, sig_n=1.0, seed=0):
    """Star at the sky position c0 seen at two rolls; detector pixel (col, row) of roll theta sits at c0 + R(theta) u + (0.27, -0.13) on the
    sky grid. Data = flux * [true detector-frame kernel rotated to the roll] + noise, rendered by the module's own operator (kernel rotation by
    bilinear resampling, convolution, bilinear sampling); the rotation operator itself is checked against analytic functions in (c) and (e)."""
    Hs = 81
    c0 = torch.tensor([40.0, 40.0])
    rng = torch.Generator().manual_seed(seed)
    data = []
    for th in rolls:
        t = math.radians(th)
        R = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        rr, cc = torch.meshgrid(torch.arange(npix), torch.arange(npix), indexing="ij")
        u = torch.stack([cc.flatten() - npix // 2, rr.flatten() - npix // 2], 1).float()
        p = c0 + u @ R.T + torch.tensor([0.27, -0.13])
        data.append({"theta": th, "x": p[:, 0], "y": p[:, 1]})
    S.set_registry(S.build_registry(["F150W"]))
    S.PSFG = S.PSFGroups(0.5, False)
    truth = S.ExplicitVarPSF(torch.tensor([3.0]), 1, nk=1, n=31)
    truth.res.data[0, 0] = true_kernel() - S.gauss_kernel(torch.tensor([3.0]), 31)[0]
    sky = torch.zeros(1, Hs, Hs)
    sky[0, int(c0[1]), int(c0[0])] = flux
    with torch.no_grad():
        ms = render_scene(truth, [group(0, e["theta"]) for e in data], sky, data)
    for e, m in zip(data, ms):
        e["truth"] = m
        e["d"] = m + sig_n * torch.randn(m.shape, generator=rng)
    return Hs, c0, data, flux, sig_n


def fit_scene(frame, Hs, c0, data, flux, sig_n, steps=60):
    S.set_registry(S.build_registry(["F150W"]))
    S.PSFG = S.PSFGroups(0.5, False)
    psf = S.ExplicitVarPSF(torch.tensor([3.0]), 1, nk=1, n=31)
    sky = torch.zeros(1, Hs, Hs)
    sky[0, int(c0[1]), int(c0[0])] = flux
    groups = [group(0, e["theta"] if frame == "detector" else 0.0) for e in data]
    opt = torch.optim.LBFGS([psf.res], lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        l = sum(((e["d"] - m) ** 2).sum() for e, m in zip(data, render_scene(psf, groups, sky, data))) / (2 * sig_n ** 2)
        l.backward()
        return l
    for _ in range(6):
        opt.step(closure)
    with torch.no_grad():
        ms = render_scene(psf, groups, sky, data)
    n = sum(e["d"].numel() for e in data)
    chi = sum(float(((e["d"] - m) ** 2).sum()) for e, m in zip(data, ms)) / sig_n ** 2 / n
    chis = []
    for e, m in zip(data, ms):  # chi2 on the pixels where the truth is above 5 noise sigma
        sel = e["truth"] > 5 * sig_n
        chis.append(float(((e["d"] - m)[sel] ** 2).sum()) / sig_n ** 2 / int(sel.sum()))
    return chi, chis, psf


def test_d_two_roll_scene():
    Hs, c0, data, flux, sig_n = two_roll_scene()
    n = sum(e["d"].numel() for e in data)
    truth_chi = sum(float(((e["d"] - e["truth"]) ** 2).sum()) for e in data) / sig_n ** 2 / n
    print(f"(d) two-roll scene: one star (flux {flux}, core peak S/N {flux * float(true_kernel().max()) / sig_n:.0f}) at rolls "
          f"{[e['theta'] for e in data]} deg, {data[0]['d'].numel()} pixels per roll, sky known, PSF kernel (31x31 ExplicitVarPSF grid) fitted by L-BFGS")
    print(f"    chi2/N of the generating model itself: {truth_chi:.3f}")
    chi_s, chis_s, _ = fit_scene("sky", Hs, c0, data, flux, sig_n)
    chi_d, chis_d, psf_d = fit_scene("detector", Hs, c0, data, flux, sig_n)
    print(f"    sky-frame kernel     (one kernel for both rolls): chi2/N = {chi_s:.3f}; on the pixels with model > 5 sigma in each roll {chis_s[0]:.2f}, {chis_s[1]:.2f}")
    print(f"    detector-frame kernel (rotated per roll)         : chi2/N = {chi_d:.3f}; on the pixels with model > 5 sigma in each roll {chis_d[0]:.2f}, {chis_d[1]:.2f}")
    assert chi_d < 1.15 * truth_chi + 0.02 and max(chis_d) < 1.3, (chi_d, chis_d)
    assert chi_s > chi_d + 0.3 and min(chis_s) > 2.0 and max(chis_d) < 1.3, (chi_s, chi_d, chis_s)
    # the fitted kernel is the truth in the detector frame
    K = true_kernel()
    Kf = psf_d.basis()[0, 0].detach()
    print(f"    fitted detector-frame kernel vs truth: max |diff| / peak = {float((Kf - K).abs().max() / K.max()):.3f}")


# ---------------------------------------------------------------------------------------------------- (e)
class GaussPSF:
    """Analytic Gaussian PSF in the detector frame with the module's rotation / box interface (unnormalised: peak 1/(2 pi sigma^2))."""

    def __init__(self, sigma_target_px, n=61):
        self.sig, self.n = sigma_target_px, n
        self.res = torch.zeros(1)

    def basis(self, rot=None):
        r = torch.arange(self.n, dtype=torch.float32) - self.n // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(-1, 2)
        out = []
        for m in range(rot["band"].numel()):
            pts = S.rot_xy(xy, rot["cs"][m], rot["parity"][m], rot["shifts"][m])
            g = torch.exp(-0.5 * pts.pow(2).sum(-1) / self.sig ** 2) / (2 * math.pi * self.sig ** 2)
            out.append(g.reshape(-1, self.n, self.n).mean(0))
        return torch.stack(out)[:, None]


def test_e_pixel_box():
    S.set_registry(S.build_registry(["F150W"]))
    pix = 3.2  # WFC3/IR 0.128 arcsec in 0.04 arcsec target px
    sig = 0.5 * pix  # sigma = 0.5 detector px
    n = 61
    Hs = 81
    c = 40
    psf = GaussPSF(sig, n)
    print(f"(e) delta source, Gaussian PSF sigma = {sig:.2f} target px = 0.5 detector px, detector pixel {pix} target px (box S = {int(math.ceil(2 * pix))}):")
    sky = torch.zeros(1, Hs, Hs)
    sky[0, c, c] = 1.0
    worst = 0.0
    for th in (0.0, 20.0, 57.3):
        S.PSFG = S.PSFGroups(0.5, True)
        g = group(0, th, pix, pix, 1, True)
        B = S.psf_group_kernels(psf, [g])  # (1, 1, n, n) box-averaged
        CB = S.convolve_basis(sky, B)[0, 0]  # CB[y, x] = K_eff(source - pixel) (correlation convention)
        S.PSFG = S.PSFGroups(0.5, False)
        CB0 = S.convolve_basis(sky, S.psf_group_kernels(psf, [group(0, th, pix, pix, 1, False)]))[0, 0]
        # brute force in the SKY frame: pixel centred on grid node p, footprint = p + J u, J = pix R(theta), 80 x 80 midpoint sub-positions
        t = math.radians(th)
        R = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
        nf = 80
        uu = (torch.arange(nf, dtype=torch.float64) + 0.5) / nf - 0.5
        uy, ux = torch.meshgrid(uu, uu, indexing="ij")
        u = torch.stack([ux.flatten(), uy.flatten()], 1)
        pts = pix * (u @ R.T)  # offsets of the footprint points from the pixel centre (sky grid)
        ref = torch.zeros(31, 31, dtype=torch.float64)
        for iy in range(31):
            for ix in range(31):
                p = torch.tensor([ix - 15.0, iy - 15.0], dtype=torch.float64)  # pixel node relative to the source (source - pixel = -p)
                d = pts + p  # footprint points relative to the source
                ref[iy, ix] = (torch.exp(-0.5 * d.pow(2).sum(1) / sig ** 2) / (2 * math.pi * sig ** 2)).mean()
        # CB[y, x] with the source at (c, c): pixel node (x, y) -> p = (x - c, y - c)
        model = CB[c - 15:c + 16, c - 15:c + 16].double()
        model0 = CB0[c - 15:c + 16, c - 15:c + 16].double()
        err = float((model - ref).abs().max() / ref.max())
        err0 = float((model0 - ref).abs().max() / ref.max())
        worst = max(worst, err)
        print(f"    roll {th:5.1f} deg: |model - brute force| / peak = {err:.2e} with the box (S=7);  without the box (kernel point-sampled) {err0:.2e}")
        assert err < 0.01, err
        assert err0 > 0.05  # the box matters at this sampling
        if th == 0.0:  # closed form: product of erf differences
            from math import erf, sqrt
            ex = torch.tensor([[erf((x + pix / 2) / (sqrt(2) * sig)) - erf((x - pix / 2) / (sqrt(2) * sig)) for x in range(-15, 16)]], dtype=torch.float64)
            cf = (ex.T @ ex) / 4.0 / pix ** 2
            errc = float((model - cf).abs().max() / cf.max())
            print(f"                     closed form (erf products), theta = 0: {errc:.2e}")
            assert errc < 0.01
    print(f"    worst: {worst:.2e} (limit 1e-2)")


# ---------------------------------------------------------------------------------------------------- (f)
def run_v33(args, out_name, data_dir=NIR, hst=True):
    out = os.path.join(ROOT, "runs", out_name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "smacs_four_v33.py"), "--data-dir", data_dir, "--out", out, "--size", "128", "--quick",
           "--no-fallback", "--boot-b", "50", "--budget-explicit", "1000", "--budget-implicit", "1000"]
    if hst:
        cmd += ["--hst-dir", SAMPLES, "--hst-instruments", "WFC3/IR,ACS/WFC", "--hst-dq-policy", "default"]
    t0 = time.time()
    p = subprocess.run(cmd + args, capture_output=True, text=True)
    open(os.path.join(out, "stdout.log"), "w").write(p.stdout + p.stderr)
    return p, out, time.time() - t0


def test_f_end_to_end_quick():
    if not (have_torchsky() and os.path.isdir(NIR) and os.path.isdir(SAMPLES)):
        print("(f) SKIPPED: torchsky or test data not available")
        return
    p, out, dt = run_v33(["--no-holdout", "--score-all", "--explicit-psf", "chrom", "--psf-model", "chrom", "--spectral", "rbf", "--svo-cache", SVO,
                          "--spec-q", "3", "--stripes", "1", "--psf-frame", "detector", "--pixel-box"], "v33_quick_detector")
    log = p.stdout + p.stderr
    print(f"(f) end-to-end --quick, NIRCam + WFC3/IR + ACS samples, --psf-frame detector --pixel-box --spectral rbf (chromatic PSF): exit {p.returncode} in {dt:.0f} s; log {out}/stdout.log")
    for ln in log.splitlines():
        if any(k in ln for k in ("PSF frame", "  group", "band registry", "TEST ", "Traceback", "Error", "failed")):
            print("    " + ln[:300])
    assert p.returncode == 0, log[-3000:]
    assert "held-out eval failed" not in log
    M = json.load(open(os.path.join(out, "metrics.json")))
    pf = M["psf_frame"]
    assert pf["frame"] == "detector" and pf["pixel_box"] and pf["n_train_groups"] >= len(M["band_registry"])
    hp = M["metrics"]["heldout_per_instrument"]
    for method, per in hp.items():
        assert set(per) == {"NIRCam", "ACS/WFC", "WFC3/IR"} and all(np.isfinite(v["chi2"]) for v in per.values()), (method, per)
    from astropy.io import fits
    for f in sorted(glob.glob(os.path.join(out, "cubes", "*.fits"))):
        assert np.isfinite(fits.getdata(f)).all(), f
    for f in glob.glob(os.path.join(out, "*.pt")):
        sd = torch.load(f, weights_only=False)
        assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()), f
    assert os.path.exists(os.path.join(out, "explicit_chrom_3d_psf_groups.npy")) or glob.glob(os.path.join(out, "*_psf_groups.npy"))
    print("    groups:", [(g["band"], g["theta_deg"], g["S"], g["n_exp"]) for g in pf["groups"]])
    print("    metrics: heldout_chi2", {k: round(v, 3) for k, v in M["metrics"]["heldout_chi2"].items()})
    for k, v in hp.items():
        print("    per instrument", k, {i: round(x["chi2"], 3) for i, x in v.items()})
    print("    all cubes and state dicts finite")


# ---------------------------------------------------------------------------------------------------- (g)
def test_g_groups_and_refusals():
    S.set_registry(S.build_registry(["F150W", "F444W"]))
    torch.manual_seed(5)
    # exposures: two rolls of F150W, one of F444W
    def exp(band, th, name, sc=0.775):
        e, _ = synthetic_exposure(th, sc, sc, 1, 0.0, nrow=60, ncol=60)
        e.update(band=band, name=name, det=0)
        e["jac"] = S.estimate_jacobian(e)
        return e
    ex = [exp("F150W", 10.1, "a"), exp("F150W", 10.2, "b"), exp("F150W", 70.0, "c"), exp("F444W", 10.1, "d", 1.57)]
    G = S.PSFGroups(0.5, True)
    ids = [G.add(e, True) for e in ex[:3]] + [None]
    G.add(ex[3], True)
    G.freeze_train()
    assert G.n_train == 3 and ids[0] == ids[1] != ids[2], (G.n_train, ids)
    print(f"(g) groups: theta 10.1 and 10.2 -> one 0.5-degree group; 70.0 -> a second; F444W -> a third ({G.n_train} groups, S = {[g['S'] for g in G.groups]})")
    unseen = exp("F150W", 123.4, "e")
    gid = G.group_id(unseen)
    assert gid == 3 and G.n_train == 3 and len(G.groups) == 4, gid
    S.PSFG = G
    psf = S.ExplicitVarPSF(torch.tensor([1.5, 2.0]), 1, 4, n=31)
    psf.res.data = 0.02 * torch.randn_like(psf.res)
    sky = S.GridSky(torch.randn(2, 64, 64))
    m = S.SkyModel(sky, psf, None, None, S.REG.index)
    S.PSF_FRAME = "detector"
    try:
        with torch.no_grad():
            _, CB = m.prepare()
            _, CB1 = m.prepare(groups=[1])
            _, CBu = m.prepare(groups=[gid])
            _, CBo = m.prepare(groups="own")
        assert CB.shape == (3, 4, 64, 64) and CBu.shape == (1, 4, 64, 64) and CBo.shape == (2, 4, 64, 64)
        assert torch.equal(CB[1:2], CB1)
        cache = {}
        c1 = S.group_cb(m, unseen, cache)
        assert torch.equal(c1, CBu[0]) and S.group_cb(m, unseen, cache) is c1
        print(f"    prepare(): CB {tuple(CB.shape)} for the 3 training groups; prepare(groups=[1]) equals its slice; the unseen-theta group (id {gid}) "
              f"is convolved on demand and cached ({tuple(c1.shape)}); 'own' (theta = 0) {tuple(CBo.shape)}")
        # predict reads the batch's group
        e = {"band": "F150W", "tx": torch.tensor([20.3, 30.1]), "ty": torch.tensor([21.0, 33.3]), "dx": torch.zeros(2), "dy": torch.zeros(2), "pgrp": 2}
        v = S.sample(CB[2], e["tx"], e["ty"])
        assert v.shape == (4, 2)
    finally:
        S.PSF_FRAME = "sky"
    # theta = 0 groups (one per band, no box) reproduce the sky-frame prepare() for both PSF classes, plain and spectral
    S.set_registry(S.build_registry(["F150W", "F444W"]))
    chrom = _chromatic()
    basis = S.RBFSpectralBasis(8, learned=False)
    lam_nodes = torch.tensor([[1.35, 1.5, 1.65], [4.1, 4.4, 4.7]])
    w_nodes = torch.full((2, 3), 1.0 / 3)
    A0 = torch.randn(8, 64, 64)
    for pname, pp in (("ExplicitVarPSF", psf), ("ChromaticPSF", chrom)):
        for kind in ("plain", "spectral"):
            sky_ = S.GridSky(torch.randn(2, 64, 64)) if kind == "plain" else S.SpectralSky(S.GridSky(A0), basis, lam_nodes, w_nodes)
            m_ = S.SkyModel(sky_, pp, None, None, S.REG.index)
            S.PSF_FRAME = "sky"
            with torch.no_grad():
                _, CBs = m_.prepare()
            S.PSFG = S.PSFGroups(0.5, False)
            S.PSFG.groups = [group(0, 0.0), group(1, 0.0)]
            S.PSFG.n_train = 2
            S.PSF_FRAME = "detector"
            try:
                with torch.no_grad():
                    _, CBd = m_.prepare()
            finally:
                S.PSF_FRAME = "sky"
            err = float((CBs - CBd).abs().max() / CBs.abs().max())
            print(f"    theta = 0 group per band vs sky-frame prepare(): {pname:15s} {kind:8s} max rel diff {err:.1e}")
            assert err < 1e-5, (pname, kind, err)
    # refusals
    class A:  # noqa: D401
        pixel_box = True
        psf_frame = "sky"
    try:
        S.configure_psf_frame(A)
        raise AssertionError("pixel-box without detector frame accepted")
    except SystemExit as exc:
        print("    refusal:", exc)
    class B:
        pixel_box = False
        psf_frame = "detector"
        zip = True
    try:
        S.configure_psf_frame(B)
        raise AssertionError("zip with detector frame accepted")
    except SystemExit as exc:
        print("    refusal:", exc)
    S.configure_psf_frame(type("C", (), {})())
    assert S.PSF_FRAME == "sky" and not S.PIXEL_BOX


def test_g2_cli_refusal():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "smacs_four_v33.py"), "--out", "/tmp/none", "--pixel-box"], capture_output=True, text=True)
    assert p.returncode != 0 and "--pixel-box needs --psf-frame detector" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-500:]
    print("    CLI: --pixel-box without --psf-frame detector exits with:", (p.stderr.strip().splitlines() or [""])[-1][:120])


if __name__ == "__main__":
    tests = [test_a_equivalence, test_b_jacobian_synthetic, test_b2_jacobian_real_hst, test_c_rotation_of_kernels, test_d_two_roll_scene,
             test_e_pixel_box, test_f_end_to_end_quick, test_g_groups_and_refusals, test_g2_cli_refusal]
    only = sys.argv[1:]
    for t in tests:
        if only and not any(o in t.__name__ for o in only):
            continue
        t()
    print("\nALL TESTS PASSED")
