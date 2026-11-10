"""Tests of the fully learned instrument track (scripts/learned_instrument.py, scripts/jwst_lw_sky.py).

(a) synthetic: known sky, known rotated elliptical PSF + 1.2 arcsec box (independent fine quadrature), known soft saturation, per-frame
    offsets; the learned model reaches the noise on held-out frames (new roll angles), its linear part matches the true kernel
    (correlation > 0.99) and the structured baseline cannot fit the nonlinearity (large residuals at bright pixels).
(b) patch sampler: rotation follows J, parity, NaN masking, NaN-aware pre-smoothing.
(c) small real run on CPU: 4 AORs, a few frames each, quick config, both models (needs /home/claude/work/li_data).
Run:  /tmp/claude-0/-home-claude/e41a3836-de0d-574f-a748-03c2fea2b582/scratchpad/venvT/bin/python -m pytest tests/test_learned_instrument.py -s
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import learned_instrument as li  # noqa: E402
from jwst_lw_sky import FrozenSky  # noqa: E402

torch.set_num_threads(2)
DEV = torch.device("cpu")
LI_DATA = os.path.join(ROOT, "li_data")


# ------------------------------------------------------------------------------------------------ synthetic world
def rot(theta_deg):
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


FINE = 0.15  # arcsec, step of the truth quadrature (the 1.2 arcsec pixel box is 8 steps)
NF = 53      # +-3.9 arcsec


def true_kernel_fine():
    """Elliptical Gaussian (sigma 0.85 x 0.55 arcsec, angle 30 deg in the detector frame) ⊗ 1.2 arcsec box on a 0.15 arcsec grid,
    +-3.9 arcsec (53 x 53), unit sum. Independent of the 0.3 arcsec patch discretisation of the model."""
    from scipy.ndimage import uniform_filter

    n = NF + 8
    ax = (np.arange(n) - (n - 1) / 2) * FINE
    vv, uu = np.meshgrid(ax, ax, indexing="ij")
    R = rot(30.0)
    a = R[0, 0] * uu + R[1, 0] * vv
    b = R[0, 1] * uu + R[1, 1] * vv
    G = np.exp(-0.5 * ((a / 0.85) ** 2 + (b / 0.55) ** 2))
    K = uniform_filter(G, size=8, mode="constant")  # box of 8 x 0.15 = 1.2 arcsec
    c = (n - NF) // 2
    K = K[c: c + NF, c: c + NF]
    return K / K.sum()


def coarse_kernel():
    """True kernel on the model's 25 x 25 patch grid (0.3 arcsec = every second fine sample), flattened, unit sum."""
    k = true_kernel_fine()[2:51:2, 2:51:2].reshape(-1)
    return k / k.sum()


class Synth:
    E_SAT = 25.0
    NOISE = 0.3   # raw units; sky amplitudes are O(1-100), so faint sources are 3-30 sigma
    COLOUR = (0.7, 0.3)

    def __init__(self, seed=0, n_train=60, n_test=6, det=32, sky_px=480):
        rng = np.random.default_rng(seed)
        self.rng, self.det, self.H = rng, det, sky_px
        yy, xx = np.mgrid[:sky_px, :sky_px].astype(np.float32)
        bands = []
        for b in range(2):
            img = np.zeros((sky_px, sky_px), np.float32)
            for _ in range(90):
                x, y = rng.uniform(30, sky_px - 30, 2)
                s = rng.uniform(2.0, 7.0)
                amp = min(float(np.exp(rng.normal(2.6, 1.3))), 500.0)
                r2 = (xx - x) ** 2 + (yy - y) ** 2
                m = r2 < (6 * s) ** 2
                img[m] += amp * np.exp(-0.5 * r2[m] / s ** 2)
            bands.append(img)
        self.sky = FrozenSky(["F356W", "F444W"], np.stack(bands), np.ones((2, sky_px, sky_px), np.float32), None, 0.12, [3.57, 4.40])
        self.skyT_truth = li.SkyTensor(self.sky, DEV, 0.3, prefilter=False)
        self.K = torch.from_numpy(true_kernel_fine().reshape(-1)).float()
        self.offs_fine = li.patch_offsets(NF, FINE)
        angles = [15.0 * i for i in range(n_train)] + [7.5 + 55.0 * j for j in range(n_test)]
        self.frames = [self.make_frame(i, th) for i, th in enumerate(angles)]
        self.train = self.frames[:n_train]
        self.test = self.frames[n_train:]

    def make_frame(self, gid, theta):
        det, rng = self.det, self.rng
        ii, jj = np.mgrid[:det, :det]
        col, row = jj.ravel().astype(np.float64), ii.ravel().astype(np.float64)
        R = rot(theta)
        s = 10.0  # sky px per detector px (1.2 / 0.12)
        c0 = np.array([self.H / 2, self.H / 2]) + rng.uniform(-15, 15, 2)
        d = np.stack([col - (det - 1) / 2, row - (det - 1) / 2], 1)
        pos = (c0[None] + s * d @ R.T).astype(np.float32)
        J = np.tile((s * R).reshape(1, 4), (pos.shape[0], 1)).astype(np.float32)
        pos_t, J_t = torch.from_numpy(pos), torch.from_numpy(J)
        e = np.zeros(pos.shape[0], np.float32)
        for k in range(0, pos.shape[0], 96):
            v, ok = self.skyT_truth.sample(pos_t[k: k + 96], J_t[k: k + 96], self.offs_fine)
            mix = sum(c * v[:, b] for b, c in enumerate(self.COLOUR))  # [B, NF*NF]
            e[k: k + 96] = (mix * self.K[None]).sum(1).numpy()
        n_e = e / (1.0 + e / self.E_SAT)  # soft saturation (true nonlinearity)
        b_i = float(rng.normal(0, 3.0))
        dat = (n_e + b_i + rng.normal(0, self.NOISE, e.shape)).astype(np.float32)
        return dict(name=f"syn{gid}", path="", aor=f"A{gid // 8}", program="P", ch=0, mjd=57000.0 + gid, texp=100.0, dce=gid, ny=det, nx=det,
                    d=dat, unc=np.full_like(dat, self.NOISE), good=np.ones(dat.shape, bool), pos=pos, J=J, row=row.astype(np.int64), col=col.astype(np.int64),
                    gid=gid, e_true=e, theta=theta)


def small_cfg():
    return li.Cfg(hidden=48, K=8, batch=2048, steps=1800, nuis_steps=150, lr=3e-3, lr_kernel=2e-3, bright_frac=0.3, det_size=32,
                  gate=10.0, smooth=0.1, bank_l2=0.1, g_l2=0.05)


def fit_model(kind, world, cfg):
    """Same code path as learned_instrument.main: prepare frames, pool of the train frames, train, held-out evaluation with a nuisance fit."""
    import copy
    torch.manual_seed(0)
    frames = copy.deepcopy(world.frames)  # prepare_frames rescales in place
    train, test = frames[:len(world.train)], frames[len(world.train):]
    unit = world.NOISE
    skyT = li.SkyTensor(world.sky, DEV, cfg.pitch, True, scale=unit)
    ik = li.init_kernel(skyT.names, cfg.patch, cfg.pitch, skyT.nb)
    li.add_time_features(frames)
    li.prepare_frames(frames, skyT, cfg, ik, unit, DEV)
    tables = li.frame_tables(frames, DEV)
    model = li.build_model(kind, skyT.nb, len(frames), cfg, tables, ik).to(DEV)
    with torch.no_grad():
        for fr in train:
            model.frame_b.weight[fr["gid"], 0] = fr["b0"]
    pool = li.Pool(train, DEV)
    info = _train(model, pool, skyT, cfg)
    recs = li.evaluate(model, test, skyT, cfg, DEV)
    return model, skyT, recs, info


def _train(model, pool, skyT, cfg):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        class Null:
            def write(self, s): pass
            def flush(self): pass
        return li.train(model, pool, skyT, cfg, 0.0, d, Null())


@pytest.fixture(scope="module")
def synth():
    world = Synth(n_train=60)
    cfg = small_cfg()
    out = {}
    for kind in ("learned", "structured"):
        w = world
        c = cfg if kind == "learned" else __import__("dataclasses").replace(cfg, steps=800)  # the baseline plateaus quickly (and a step is 3x dearer)
        model, skyT, recs, info = fit_model(kind, w, c)
        out[kind] = dict(model=model, recs=recs, metrics=li.compute_metrics(recs)["all"], info=info, world=w)
        m = out[kind]["metrics"]
        print(f"\n[{kind}] steps {info['steps']}  {info['samples_per_s']:.0f} samples/s  held-out chi2: all {m['all']['chi2_mean']:.3f} "
              f"(n={m['all']['n']}), bright {m['bright'].get('chi2_mean', float('nan')):.3f} (n={m['bright']['n']}), "
              f"median-robust {m['all']['chi2_median']:.3f}")
    return out


def test_a_synthetic_learned_reaches_noise(synth):
    m = synth["learned"]["metrics"]
    assert m["all"]["n"] > 1000 and m["bright"]["n"] > 100
    assert m["all"]["chi2_mean"] < 1.3, m["all"]
    assert m["bright"]["chi2_mean"] < 1.6, m["bright"]


def test_a_synthetic_structured_cannot_fit_nonlinearity(synth):
    ml, ms = synth["learned"]["metrics"], synth["structured"]["metrics"]
    assert ms["bright"]["chi2_mean"] > 4.0, ms["bright"]
    assert ms["bright"]["chi2_mean"] > 3.0 * ml["bright"]["chi2_mean"]
    # most bright pixels of the baseline are off by more than 3 sigma; for the learned model almost none are
    def frac_off(kind):
        z = np.concatenate([((r["d"] - r["pred"]) / r["unc"])[r["score"] & (r["src"] / r["unc"] > 5)] for r in synth[kind]["recs"]])
        return float(np.mean(np.abs(z) > 3.0))
    fs, fl = frac_off("structured"), frac_off("learned")
    print(f"\nfraction of bright pixels with |resid| > 3 sigma: structured {fs:.3f}, learned {fl:.3f}")
    assert fs > 0.3 and fl < 0.15


def test_a_synthetic_linear_part_matches_true_psf(synth):
    model, w = synth["learned"]["model"], synth["learned"]["world"]
    W = model.effective_kernel(0, [0.0, 0.0], (w.frames[0]["mjd"] - li.MJD_REF) / li.MJD_SCALE).numpy()  # [nb, P2]
    Kc = coarse_kernel()
    truth = np.stack([c * Kc for c in w.COLOUR])
    r = np.corrcoef(W.reshape(-1), truth.reshape(-1))[0, 1]
    # per band too
    rb = [np.corrcoef(W[b], truth[b])[0, 1] for b in range(2)]
    # colour mixing: ratio of the integrated weights
    ratio = W[1].sum() / W[0].sum()
    print(f"\nlinear-part correlation with the true kernel: {r:.4f} (per band {rb[0]:.4f}, {rb[1]:.4f}); band weight ratio {ratio:.3f} (true {0.3 / 0.7:.3f})")
    assert r > 0.99


def test_a_synthetic_predictions_are_per_frame_offset_free(synth):
    # roll invariance: held-out score does not depend on the roll angle beyond noise
    recs = synth["learned"]["recs"]
    chi = [li.compute_metrics([r])["all"]["all"]["chi2_mean"] for r in recs]
    print("per held-out frame chi2:", np.round(chi, 3))
    assert max(chi) < 1.8


# ------------------------------------------------------------------------------------------------ (b) sampler
def _sky(data, names=("F356W",), prefilter=False):
    data = np.asarray(data, np.float32)
    data = data[None] if data.ndim == 2 else data
    fs = FrozenSky(list(names), data, np.ones_like(data), None, 0.12, [3.5] * data.shape[0])
    return li.SkyTensor(fs, DEV, 0.3, prefilter=prefilter)


def _J(theta, scale=1.0, flip=False):
    R = rot(theta) * scale
    if flip:
        R = R @ np.diag([1.0, -1.0])
    return torch.tensor(R.reshape(1, 4), dtype=torch.float32)


@pytest.mark.parametrize("theta,flip", [(0, False), (33.0, False), (-110.0, False), (47.0, True)])
def test_b_patch_follows_jacobian_linear_sky(theta, flip):
    H = 200
    yy, xx = np.mgrid[:H, :H].astype(np.float32)
    a, b = 0.7, -0.4
    sk = _sky(a * xx + b * yy + 3.0)
    P = 25
    offs = li.patch_offsets(P, 0.3)
    pos = torch.tensor([[100.3, 90.6]])
    J = _J(theta, 2.5, flip)
    v, ok = sk.sample(pos, J, offs)
    assert ok.all()
    ou, ov = offs[:, 0].numpy(), offs[:, 1].numpy()
    Jn = J[0].numpy().reshape(2, 2)
    sx = pos[0, 0].item() + Jn[0, 0] * ou + Jn[0, 1] * ov
    sy = pos[0, 1].item() + Jn[1, 0] * ou + Jn[1, 1] * ov
    assert np.allclose(v[0, 0].numpy(), a * sx + b * sy + 3.0, atol=2e-4)
    # row-major (v, u) layout: first axis of the PxP patch is the detector ROW offset
    assert np.allclose(offs.reshape(P, P, 2)[3, 7].numpy(), [(7 - 12) * 0.25, (3 - 12) * 0.25])


@pytest.mark.parametrize("theta", [25.0, 90.0, -60.0])
def test_b_rotation_follows_J_elongated_source(theta):
    """Sky blob elongated along the direction theta, sampled with J = s R(theta), equals the same blob elongated along x sampled with J = s I."""
    H, s = 241, 3.0
    yy, xx = np.mgrid[:H, :H].astype(np.float32)
    c = (H - 1) / 2

    def blob(phi):
        R = rot(phi)
        a = R[0, 0] * (xx - c) + R[1, 0] * (yy - c)  # coordinate along the blob axis
        b = R[0, 1] * (xx - c) + R[1, 1] * (yy - c)
        return np.exp(-0.5 * ((a / 14.0) ** 2 + (b / 4.0) ** 2))

    offs = li.patch_offsets(25, 0.3)
    pos = torch.tensor([[c, c]])
    ref, _ = _sky(blob(0.0)).sample(pos, _J(0.0, s), offs)
    rotd, ok = _sky(blob(theta)).sample(pos, _J(theta, s), offs)
    assert ok.all()
    err = (ref - rotd).abs().max().item()
    # a patch sampled WITHOUT following the rotation would differ strongly
    wrong, _ = _sky(blob(theta)).sample(pos, _J(0.0, s), offs)
    assert err < 0.02, err
    assert (ref - wrong).abs().max().item() > 0.2


def test_b_nan_masking_and_out_of_grid():
    H = 60
    img = np.full((H, H), 2.0, np.float32)
    img[20:25, 30:35] = np.nan
    sk = _sky(img)
    offs = li.patch_offsets(7, 1.2)  # integer detector offsets -> J = I samples exactly on pixel centres
    v, ok = sk.sample(torch.tensor([[30.0, 22.0], [10.0, 40.0], [30.5, 22.5], [1.0, 1.0]]), _J(0.0), offs)
    o = offs.numpy().astype(int)
    for k, (px, py) in enumerate([(30, 22), (10, 40)]):
        for q in range(offs.shape[0]):
            x, y = px + o[q, 0], py + o[q, 1]
            expect = not (20 <= y < 25 and 30 <= x < 35)
            assert bool(ok[k, 0, q]) == expect
            assert float(v[k, 0, q]) == (2.0 if expect else 0.0)
    # half-pixel positions: a sample is valid only if all four neighbours are finite
    px, py = 30.5, 22.5
    for q in range(offs.shape[0]):
        x, y = px + o[q, 0], py + o[q, 1]
        nb = [(int(math.floor(x)) + i, int(math.floor(y)) + j) for i in (0, 1) for j in (0, 1)]
        expect = all(not (20 <= yy_ < 25 and 30 <= xx_ < 35) for xx_, yy_ in nb) and all(0 <= xx_ < H and 0 <= yy_ < H for xx_, yy_ in nb)
        assert bool(ok[2, 0, q]) == expect
    assert not ok[3, 0, :].all()  # patch hanging off the grid corner is partly invalid
    assert (v[~ok] == 0).all()


def test_b_prefilter_is_nan_aware():
    img = np.full((80, 80), 5.0, np.float32)
    img[30:40, 30:40] = np.nan
    sk = _sky(img, prefilter=True)
    m = sk.mask[0, 0].numpy() > 0
    assert not m[30:40, 30:40].any() and m.sum() == 80 * 80 - 100
    assert np.allclose(sk.vals[0, 0].numpy()[m], 5.0, atol=1e-4)  # normalised convolution: no leakage of the NaN hole as zeros


def test_b_irac_mask_bits_match_pixelstore():
    try:
        import pixelstore
    except Exception:
        pytest.skip("pixelstore not importable")
    assert pixelstore.IRAC_FATAL_BITS == li.IRAC_FATAL_BITS == 32520


def test_split_deterministic_and_distinct_programs():
    import csv
    rows = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "datasearch", "manifest_spitzer_union.csv"))):
        rows[r["aor"]] = r["program"].split("_")[-1]
    assert len(rows) == 11
    s1, s2 = li.make_split(rows), li.make_split(dict(reversed(list(rows.items()))))
    assert s1["test"] == s2["test"] and s1["val"] == s2["val"]
    assert len(s1["test"]) == 2 and len(s1["val"]) == 1
    progs = [rows[a] for a in s1["test"] + s1["val"]]
    assert len(set(progs)) == 3
    assert rows[s1["train"][0]] and not set(s1["train"]) & set(s1["test"] + s1["val"])
    assert "90233" not in progs  # single-AOR program stays in training
    print("\nsplit:", {k: s1[k] for k in ("test", "val")}, progs)


# ------------------------------------------------------------------------------------------------ jwst_lw_sky
def test_jwst_lw_sky_single_exposure(tmp_path):
    f = os.path.join(ROOT, "testdata", "nircam", "jw02736001001_02103_00004_nrcblong_cal.fits")
    if not os.path.exists(f):
        pytest.skip("no NIRCam test file")
    import jwst_lw_sky as jl
    out = jl.build([f], str(tmp_path), bands=("F356W",), scale_arcsec=0.12)
    sky = jl.load_sky(out)
    d, w = sky.data[0], sky.weight[0]
    fin = np.isfinite(d)
    assert 0.2 < fin.mean() < 0.9 and ((w > 0) == fin).all()
    assert abs(np.nanmedian(d)) < 0.02  # background subtracted per exposure
    assert sky.scale_arcsec == pytest.approx(0.12)


# ------------------------------------------------------------------------------------------------ (c) real small run
@pytest.mark.skipif(not os.path.exists(os.path.join(LI_DATA, "sky", "jwst_lw_sky.fits")), reason="li_data (IRAC samples + coarse coadd) missing")
def test_c_real_small_run_cpu(tmp_path):
    out = str(tmp_path / "run")
    li.main(["--sky", os.path.join(LI_DATA, "sky", "jwst_lw_sky.fits"), "--irac-root", os.path.join(LI_DATA, "irac"), "--out", out,
             "--aors", "48186112,58000128,58467840,66697472", "--model", "both", "--quick", "--max-frames-per-aor", "3", "--steps", "80", "--nuis-steps", "30", "--max-png", "2"])
    m = json.load(open(os.path.join(out, "metrics.json")))
    split = json.load(open(os.path.join(out, "split.json")))
    assert len(split["test"]) == 2 and len(split["val"]) == 1
    assert len({split["programs"][a] for a in split["test"] + split["val"]}) == 3
    u = m["usable_pixels"]
    assert sum(v["n_usable"] for v in u.values()) > 1000
    bad = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, float) and not math.isfinite(o):
            bad.append(path)
    walk(m["models"])
    assert not bad, bad
    for kind in ("learned", "structured"):
        t = m["models"][kind]["metrics"]["test"]["all"]["all"]
        assert t["n"] > 100 and np.isfinite(t["chi2_mean"]) and np.isfinite(t["chi2_median"])
        pngs = [f for f in os.listdir(os.path.join(out, kind)) if f.startswith("resid_") and f.endswith(".png")]
        assert len(pngs) >= 1
    assert m["models"]["learned"]["parameters"]["total"] > m["models"]["structured"]["parameters"]["total"]
