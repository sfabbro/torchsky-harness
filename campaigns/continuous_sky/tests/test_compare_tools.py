"""Synthetic tests of scripts/compare_operators.py and scripts/compare_kernels.py (CPU, a few seconds each).

compare_operators: two fake runs on the same exposures. Truth = Gaussian core + Gaussian wings + 6 angular spikes (1/r). A = the truth model (chi2 ~ 1),
B = the same core with half the wings and 40 per cent of the spikes. Expected: B has the larger chi2, the excess sits in shoulders / wings, the mean-z azimuthal profile of B in the
3-8 / 8-15 px annuli carries an m = 6 modulation that A lacks; A and B agree on the data; row / column / position tables are filled.

compare_kernels: E2 table = Gaussian (sigma_E2) + spikes on a 4x lattice (a centroid offset is put in on purpose); op nodes on the real opfield Stencil
(core 10 px at pitch 0.5, log-polar wings) with cell-integrated masses of the same kernel blurred by a KNOWN Gaussian (variance s_true) and two rolls with
different recentring cbar. Expected: the fitted common blur recovers s_true, the residual after the gauge is far below the raw one, the m = 6 amplitudes
agree; with the spikes removed from the op kernel the op m = 6 amplitude collapses; with the E2 table sharper than op the sign of s flips.

Run: venvT/bin/python tests/test_compare_tools.py
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_kernels as CK  # noqa: E402
import compare_operators as CO  # noqa: E402


# ------------------------------------------------------------------------------------------------ compare_operators
def psf_value(ox, oy, wings=1.0, spikes=1.0, sig=1.3):
    """core + Gaussian wings (x `wings`) + 6 angular spikes ~ 1/r (x `spikes`)"""
    r = np.hypot(ox, oy) + 1e-9
    core = np.exp(-0.5 * r ** 2 / sig ** 2) / (2 * np.pi * sig ** 2)
    s2 = 5.0
    ph = np.arctan2(oy, ox)
    ang = sum(np.exp(-0.5 * (np.angle(np.exp(1j * (ph - k * np.pi / 3))) / 0.07) ** 2) for k in range(6))
    return 0.80 * core + wings * 0.12 * np.exp(-0.5 * r ** 2 / s2 ** 2) / (2 * np.pi * s2 ** 2) + spikes * 0.08 * ang / (r + 1.0) / 60.0 * (r > 2)


def make_fake_runs(tmp, n_exp=3, seed=0):
    rng = np.random.default_rng(seed)
    runA, runB = os.path.join(tmp, "runA"), os.path.join(tmp, "runB")
    for r, m in ((runA, "opfield"), (runB, "explicit_lib_3d")):
        os.makedirs(os.path.join(r, "heldout_pixels", m))
    H, W = 240, 260
    rr, cc = np.mgrid[0:H, 0:W]
    for ie in range(n_exp):
        pos = []
        while len(pos) < 14:
            y, x = rng.uniform(25, H - 25), rng.uniform(25, W - 25)
            if all(math.hypot(y - a, x - b) > 40 for a, b in pos):
                pos.append((y, x))
        truth = np.full((H, W), 20.0)
        modelA = truth.copy()
        modelB = truth.copy()
        for (y, x) in pos:
            F = rng.uniform(3e3, 2e4)
            oy, ox = rr - y, cc - x
            m = (np.abs(oy) < 30) & (np.abs(ox) < 30)
            truth[m] += F * psf_value(ox[m], oy[m])
            modelA[m] += F * psf_value(ox[m], oy[m])
            modelB[m] += F * psf_value(ox[m], oy[m], wings=0.5, spikes=0.4)
        sigma = 3.0 + 0.05 * np.sqrt(np.abs(truth))
        data = truth + rng.normal(0, 1, truth.shape) * sigma
        dx = (cc / (W - 1) * 2 - 1).ravel()
        dy = (rr / (H - 1) * 2 - 1).ravel()
        for run, mod, meth in ((runA, modelA, "opfield"), (runB, modelB, "explicit_lib_3d")):
            np.savez_compressed(os.path.join(run, "heldout_pixels", meth, f"exp{ie}.npz"), name=f"exp{ie}.fits", band="F150W", instrument="NIRCam", det="NRCB3", det_index=0,
                                split="test", method=meth, mjd=59000.0 + ie, roll_deg=10.0, parity=1, pixscale_arcsec=0.031, gain=1.0, noise_scale=1.0,
                                row=rr.ravel().astype(np.int32), col=cc.ravel().astype(np.int32), dx=dx.astype(np.float32), dy=dy.astype(np.float32),
                                tx=cc.ravel().astype(np.float32), ty=rr.ravel().astype(np.float32), tx_hdr=cc.ravel().astype(np.float32), ty_hdr=rr.ravel().astype(np.float32),
                                data=data.ravel().astype(np.float32), model=mod.ravel().astype(np.float32), model_sky=(mod - 20.0).ravel().astype(np.float32),
                                model_bkg=np.full(H * W, 20.0, np.float32), sigma=sigma.ravel().astype(np.float32))
    return runA, runB


def test_compare_operators():
    with tempfile.TemporaryDirectory() as tmp:
        runA, runB = make_fake_runs(tmp)
        out = os.path.join(tmp, "out")
        CO.main([runA, runB, "--out", out, "--peak-snr", "50"])
        R = json.load(open(os.path.join(out, "compare_operators.json")))
        allg = R["groups"]["all"]
        cat = allg["cat"]
        assert cat["total_chi2_per_pixel_B"] > cat["total_chi2_per_pixel_A"] + 0.01, cat
        assert 0.97 < cat["total_chi2_per_pixel_A"] < 1.06, cat["total_chi2_per_pixel_A"]            # the true model is at chi2 ~ 1
        d = np.array(cat["delta_chi2_over_total_B"])
        assert d[1] + d[2] < 0 and (d[1] + d[2]) < -0.003, d  # B (second run) is worse in shoulders + wings: A - B < 0
        assert all(abs(d[k]) < 0.003 for k in (4,)), d                                                     # nothing happens far from the sources
        rad = allg["radial"]
        zA, zB = np.array(rad["mean_z2A"]), np.array(rad["mean_z2B"])
        big = np.array(rad['n']) > 3000
        assert np.nanmax(np.abs(zA - 1.0)[big]) < 0.1 and (zB[4:9] > zA[4:9] + 0.03).all(), (zA, zB, rad['n'])
        az = allg["azimuth_1"]       # annulus 3-8 px
        aA, aB = az["amp_mean_z_A"]["m6"], az["amp_mean_z_B"]["m6"]
        assert aB > 3 * aA and aB > 0.08, (aA, aB)
        assert len(az["mean_zA"]) == 16
        for k in ("snr", "pos", "row", "col"):
            assert k in allg
        assert R["n_common_exposures"] == 3 and R["per_exposure"][0]["max_abs_data_diff_over_sigma"] < 1e-6
        for f in ("1_chi2_snr.png", "2_peaks_radial.png", "3_peaks_azimuth.png", "4_position.png", "5_readout.png", "6_chi2_fraction.png"):
            assert os.path.getsize(os.path.join(out, f)) > 2000, f
        n_used = sum(r["n_peaks_used"] for r in R["per_exposure"])
        print(f"  compare_operators: chi2/px A {cat['total_chi2_per_pixel_A']:.3f} B {cat['total_chi2_per_pixel_B']:.3f}; (A-B)/total_B by category {np.round(d, 4).tolist()}; "
              f"m=6 amplitude of mean z (3-8 px) A {aA:.3f} B {aB:.3f}; {n_used} stamps")
        # null_stacks.py must read the same dumps unchanged
        import null_stacks as NS
        method, exps = NS.load_run(runA)
        assert method == "opfield" and len(exps) == 3 and np.isfinite(NS.whiten(exps[0])).all()


# ------------------------------------------------------------------------------------------------ compare_kernels
def gauss_cell_mass(cx, cy, side, sig):
    """exact integral of a unit 2-D Gaussian (sigma) over the box of side `side` centred at (cx, cy)"""
    f = lambda c: 0.5 * (erf((c + side / 2) / (sig * math.sqrt(2))) - erf((c - side / 2) / (sig * math.sqrt(2))))  # noqa: E731
    return f(cx) * f(cy)


def spike_density(x, y):
    r = np.hypot(x, y) + 1e-9
    ph = np.arctan2(y, x)
    ang = sum(np.exp(-0.5 * (np.angle(np.exp(1j * (ph - k * np.pi / 3))) / 0.12) ** 2) for k in range(6))
    return ang / r * ((r > 2) & (r < 26))


def build_inputs(tmp, sig_e2=1.2, s_true=0.09, spikes_e2=True, spikes_op=True, eps=0.06, c_e=(0.15, -0.1), Lc=None, save_J=True):
    import opfield as O
    e2dir, opdir = os.path.join(tmp, "e2"), os.path.join(tmp, "op")
    os.makedirs(e2dir)
    os.makedirs(os.path.join(opdir, "kernels"))
    os_, R = 4, 40.0
    N = 2 * int(math.ceil(R * os_)) + 1
    ax = (np.arange(N) - N // 2) / os_
    xx, yy = np.meshgrid(ax, ax)
    # spikes normalised on the fine lattice
    sp = spike_density(xx, yy)
    sp = sp / sp.sum()

    def kernel_samples(sig, spikes, shift):
        g = np.array([[gauss_cell_mass(x - shift[0], y - shift[1], 1.0 / os_, sig) for x in ax] for y in ax])
        s = spike_density(xx - shift[0], yy - shift[1])
        s = s / s.sum()
        K = (1 - eps) * g + (eps * s if spikes else 0.0)
        return K / K.sum()

    K_e2 = kernel_samples(sig_e2, spikes_e2, c_e)
    groups = {"NRCB3/F090W": {"det": "NRCB3", "band": "F090W", "node_index": 1, "lam_um": 0.9, "oversample": 4, "half_window_det_px": R, "library": K_e2, "learned": K_e2, "table": K_e2}}
    np.save(os.path.join(e2dir, "explicit_lib_3d_psf_groups.npy"), {"detector_frame": groups}, allow_pickle=True)
    # opfield kernel: same family, variance sig_e2^2 + s_true; the nodes on the real stencil
    sig_op = math.sqrt(max(sig_e2 ** 2 + s_true, 1e-4))     # s_true < 0: the op kernel is SHARPER
    st = O.Stencil(10.0, 0.5, 1.25, 24, 48.0)
    nk = st.nk_for(48.0)
    npz, rows = {}, []
    for k, (roll, cbar) in enumerate(((-35, (0.08, -0.05)), (62, (0.05, 0.02)))):
        v = st.v[:nk]
        vdet = v - np.array(cbar)[None]                       # CANONICAL stencil coordinates
        Lm = np.eye(2) if Lc is None else np.asarray(Lc, float)   # learned Lambda: physical detector-px offsets u = Lambda vdet
        u = vdet @ Lm.T
        side = np.sqrt(st.dv[:nk] * abs(np.linalg.det(Lm)))
        g = np.array([gauss_cell_mass(x, y, sd, sig_op) for (x, y), sd in zip(u, side)])
        s = spike_density(u[:, 0], u[:, 1]) * st.dv[:nk] * abs(np.linalg.det(Lm))
        s = s / s.sum()
        m = (1 - eps) * g + (eps * s if spikes_op else 0.0)
        m = m / m.sum()
        name = f"NIRCam|NRCB3|F090W|{roll}"
        J = np.array([[0.7, 0.2], [-0.2, 0.7]], np.float64)
        M = (J @ Lm).astype(np.float32)
        if save_J:
            npz[name + "|J"] = J.astype(np.float32)
        npz[name + "|vdet"] = vdet.astype(np.float32)
        npz[name + "|mass"] = m.astype(np.float32)
        npz[name + "|pos"] = (vdet @ M.T).astype(np.float32)
        npz[name + "|grid"] = np.zeros((3, 3), np.float32)
        npz[name + "|M"] = M
        lam = np.array([0.85, 0.9, 0.95], np.float32)
        npz[name + "|lamq"] = lam
        npz[name + "|wq"] = np.array([0.3, 0.4, 0.3], np.float32)
        npz[name + "|massq"] = np.stack([m] * 3).astype(np.float32)
        rows.append({"group": name, "instrument": "NIRCam", "det": "NRCB3", "band": "F090W", "roll_deg": roll, "channel": "NIRCAM_SW", "pixscale_arcsec_per_det_px": 0.031})
    np.savez_compressed(os.path.join(opdir, "kernels", "effective_kernels.npz"), **npz)
    json.dump({"kernels": rows, "stencil": {"wings": "logpolar", "core_pitch_det_px": 0.5, "core_radius_det_px": 10.0}}, open(os.path.join(opdir, "kernels", "kernels.json"), "w"))
    return opdir, e2dir


def run_kernels(tmp, **kw):
    opdir, e2dir = build_inputs(tmp, **kw)
    out = os.path.join(tmp, "out")
    CK.main([opdir, e2dir, "--out", out])
    return json.load(open(os.path.join(out, "compare_kernels.json"))), out


def test_compare_kernels_recovers_blur_and_spikes():
    with tempfile.TemporaryDirectory() as tmp:
        R, out = run_kernels(tmp, s_true=0.09)
        ch = R["channels"]["NIRCAM_SW"]
        c = R["cells"]["NRCB3/F090W"]
        assert abs(ch["blur_variance_s_px2"] - 0.09) < 0.02, ch
        assert c["residual_after_gauge"]["L1_fraction_of_window_mass"] < 0.5 * c["residual_raw"]["L1_fraction_of_window_mass"], c
        assert c["residual_after_gauge"]["max_abs_over_E2_peak"] < 0.05
        assert abs(c["core_centroid_E2"][0] - 0.15) < 0.03 and abs(c["core_centroid_E2"][1] + 0.1) < 0.03
        a6, a2 = c["azimuthal_amplitude"]["m6_r6-12"], c["azimuthal_amplitude"]["m2_r6-12"]
        assert a6["E2"] > 0.1 and 0.6 < a6["op"] / a6["E2"] < 1.4, a6
        assert a2["E2"] < 0.2 * a6["E2"] and a2["op"] < 0.2 * a6["op"], (a2, a6)
        assert c["n_rolls"] == 2 and "lambda" in c and len(c["lambda"]["lam_um"]) == 3
        ee = np.array(c["EE"]["op_minus_E2"])
        assert np.abs(ee[[6, 7, 8, 9, 10]]).max() < 0.01 and ee[1] < -0.02, ee   # EE agrees beyond 6 px and shows the broader op core at 1-2 px
        for f in ("kernels_core_ee.png", "kernels_channel_summary.png"):
            assert os.path.getsize(os.path.join(out, f)) > 2000
        print(f"  compare_kernels: fitted s = {ch['blur_variance_s_px2']:.4f} px^2 (true 0.09); L1 raw {c['residual_raw']['L1_fraction_of_window_mass']:.4f} -> "
              f"gauge {c['residual_after_gauge']['L1_fraction_of_window_mass']:.4f}; m=6 (6-12 px) op {a6['op']:.3f} E2 {a6['E2']:.3f}; E2 centroid {np.round(c['core_centroid_E2'], 3).tolist()}")


def test_compare_kernels_sign_and_missing_spikes():
    with tempfile.TemporaryDirectory() as tmp:
        R, _ = run_kernels(tmp, s_true=-0.09, spikes_op=False)
        ch = R["channels"]["NIRCAM_SW"]
        c = R["cells"]["NRCB3/F090W"]
        assert abs(ch["blur_variance_s_px2"] + 0.09) < 0.07 and "sharper" in ch["sign"], ch   # looser: without the spikes the op core carries 6 per cent more of the window mass
        a6 = c["azimuthal_amplitude"]["m6_r6-12"]
        assert a6["op"] < 0.25 * a6["E2"], a6
        print(f"  compare_kernels (op sharper, no spikes): s = {ch['blur_variance_s_px2']:.4f} (true -0.09); m=6 op {a6['op']:.3f} vs E2 {a6['E2']:.3f}")


def test_compare_kernels_lambda_not_identity():
    """Regression for the 7 Nov bug: the stored vdet is CANONICAL stencil coordinates; with the learned Lambda != I (here ~0.37 x, sheared: 2.7x
    shrink as in the real SW run) the detector-px offsets are J^-1 pos = Lambda vdet. The old code read vdet as detector px and found the op kernel
    ~2.7x too broad."""
    Lc = [[0.40, 0.10], [-0.06, 0.33]]
    with tempfile.TemporaryDirectory() as tmp:
        R, _ = run_kernels(tmp, s_true=0.09, Lc=Lc, spikes_e2=False, spikes_op=False, eps=0.0)
        ch = R["channels"]["NIRCAM_SW"]
        c = R["cells"]["NRCB3/F090W"]
        sig_op = math.sqrt(1.2 ** 2 + 0.09)
        assert 0.9 < c["core_rms_op_px"] / sig_op < 1.05, (c["core_rms_op_px"], sig_op)      # old code: ~3.3 px
        assert abs(ch["blur_variance_s_px2"] - 0.09) < 0.08, ch                              # + <= p^2/6 = 0.04 from the bilinear lattice splat; old code: ~ +9
        sv = np.array(c["Lambda_singular_values"])
        assert np.allclose(sv, np.linalg.svd(np.array(Lc), compute_uv=False), rtol=2e-4), sv
        ee = np.array(c["EE"]["op_minus_E2"])
        assert np.abs(ee[[4, 5, 6]]).max() < 0.02, ee
        print(f"  compare_kernels (Lambda != I): core rms op {c['core_rms_op_px']:.3f} px (expected {sig_op:.3f}), s = {ch['blur_variance_s_px2']:.4f} (true 0.09 + splat)")


def test_legacy_jacobian_fallback():
    J = CK.legacy_jacobian(145.0, 0.031, 0.04)
    assert abs(np.linalg.det(J) - (0.031 / 0.04) ** 2) < 1e-12
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert np.allclose(CK.detector_offsets(v @ J.T, J), v)


def test_compare_kernels_legacy_file_message():
    with tempfile.TemporaryDirectory() as tmp:
        opdir, e2dir = build_inputs(tmp)
        z = dict(np.load(os.path.join(opdir, "kernels", "effective_kernels.npz")))
        z = {k: v for k, v in z.items() if not k.endswith(("|vdet", "|M", "|massq", "|lamq", "|wq"))}
        np.savez_compressed(os.path.join(opdir, "kernels", "effective_kernels.npz"), **z)
        try:
            CK.main([opdir, e2dir, "--out", os.path.join(tmp, "o")])
        except SystemExit as exc:
            assert "eval-only" in str(exc)
        else:
            raise AssertionError("legacy file accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL OK")
