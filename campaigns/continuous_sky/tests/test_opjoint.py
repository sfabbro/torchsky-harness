"""CPU tests of scripts/opjoint.py on a purpose-built analytic world (two fields, two instruments, Gaussian PSFs of different width per instrument).

 1. opfield.py is unchanged (sha256 in tests/opfield.sha256; written on first run)
 2. inner sky steps leave every global parameter, buffer, nuisance row and the operator Adam state bit-identical
 3. a single tile with --inner-sky 0 runs and the loss decreases (same loop as opfield.train)
 4. joint fit of a rich + a thin field beats the thin field alone on the thin field's held-out dithers (T1)
 5. leave-field-out: local-only refit of a left-out field works and leaves every global parameter bit-identical

 6. [world 2] two fields (X with two tiles, Y with one), two instruments with different PSF widths AND pixel scales (1 and 2 target px per detector px):
    joint fit vs each single-field fit on held-out dithers (T1); the shared operator is recovered (kernel width of both instruments from the joint fit)
 7. [world 2] leave-one-field-out (Y): local-only refit, global parameters bit-identical, score close to the joint fit; optional fine-tune of everything runs
 8. [world 2] T5 held-out tile (local-only refit with the operator fixed) against the control (operator refitted on that tile alone)
 9. memory swap: gradients after swapping tiles (sky and pool leave the device and come back) equal the plain ones, bit for bit; a fit that never swapped agrees; pool chunks tile the pool exactly
10. an exposure shared by two tiles has ONE global row (z, pe, stripes); private ids for T5 tiles
11. fields config (JSON) parsing and validation; legacy flags give the same structure
12. one tile, --inner-sky 0 versus opfield.train for the same steps, seed and batches: max |difference| of every parameter is reported and asserted
Run: /tmp/claude-0/-home-claude/e41a3836-de0d-574f-a748-03c2fea2b582/scratchpad/venvTorch/bin/python tests/test_opjoint.py
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import opfield as O  # noqa: E402
import opjoint as J  # noqa: E402

V = O.V
torch.set_num_threads(4)
DEV = torch.device("cpu")
SIG_PSF = {"iA": 1.1, "iB": 1.9}
BAND_INST = {"bA": "iA", "bB": "iB"}
SB = 1.0       # intrinsic blob width (target px)
NOISE = 0.02
SIZE = 28


def truth_sky(seed, size=SIZE, n=7):
    g = np.random.RandomState(seed)
    return [(g.uniform(3, size - 3), g.uniform(3, size - 3), g.uniform(0.6, 2.0)) for _ in range(n)]


def render(blobs, x, y, sig_psf):
    s2 = SB ** 2 + sig_psf ** 2
    out = torch.zeros_like(x)
    for bx, by, A in blobs:
        out = out + A * (SB ** 2 / s2) * torch.exp(-0.5 * ((x - bx) ** 2 + (y - by) ** 2) / s2)
    return out


def make_tile_exps(tag, blobs, plan, seed, size=SIZE):
    """plan = {band: n_exposures}. Exposure = size x size detector pixels, dithered by a sub-pixel offset."""
    g = np.random.RandomState(seed)
    exps = []
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    for band, n in plan.items():
        for k in range(n):
            ox, oy = g.uniform(-0.5, 0.5, 2)
            tx, ty = (xx + ox).reshape(-1), (yy + oy).reshape(-1)
            clean = render(blobs, tx, ty, SIG_PSF[BAND_INST[band]])
            d = clean + NOISE * torch.from_numpy(g.randn(clean.numel()).astype(np.float32))
            e = dict(name=f"{tag}_{band}_{k}_cal.fits", band=band, detname="D" + band, d=d, ivar=torch.full_like(d, 1.0 / NOISE ** 2), tx=tx.clone(), ty=ty.clone(),
                     dx=(xx.reshape(-1) - size / 2) / size, dy=(yy.reshape(-1) - size / 2) / size, row=yy.reshape(-1).long(), col=xx.reshape(-1).long(), bkg0=0.0,
                     jac=dict(J=[[1.0, 0.0], [0.0, 1.0]], theta_deg=0.0, parity=1, scale=1.0), mjd=59000.0 + k, exptime=100.0, path=None, holdout=False)
            exps.append(e)
    return exps


def classical(exps, thead, kernel, scale, dev, bands=None):
    H, W = int(thead["NAXIS2"]), int(thead["NAXIS1"])
    bands = bands or ["bA", "bB"]
    cube = torch.zeros(len(bands), H, W)
    cov = {}
    for j, b in enumerate(bands):
        num, den = torch.zeros(H * W), torch.zeros(H * W)
        for e in exps:
            if e["band"] != b or e.get("holdout") or e.get("val"):
                continue
            ix = e["tx"].round().long().clamp(0, W - 1)
            iy = e["ty"].round().long().clamp(0, H - 1)
            num.index_add_(0, iy * W + ix, e["d"])
            den.index_add_(0, iy * W + ix, torch.ones_like(e["d"]))
        cube[j] = (num / den.clamp(min=1)).view(H, W)
        cov[b] = (den.view(H, W) > 0)
    classical.last_cover = cov
    return cube


V.estimate_jacobian = lambda e: e["jac"]


def reg():
    return O.OpReg(["bA", "bB"], [1.5, 2.5], ["iA", "iB"], ["CA", "CB"], [False, False])


def small_args(out, extra=()):
    return O.parse_args(["--out", str(out), "--core-radius", "4.5", "--core-pitch", "0.5", "--wing-ratio", "1.6", "--wing-nang", "8", "--wing-arcsec", "0.8", "--hidden", "24",
                         "--n-bank", "4", "--shape-hidden", "24", "--scale", "0.1", "--spec-q", "3", "--batch", "512", "--bright-frac", "0.3", "--nuis-steps", "40",
                         "--nuis-batch", "512", "--log-every", "1000", "--c2f-start-arcsec", "0.2", "--c2f-frac", "0.3", "--quick"] + list(extra))


def make_ctx(tmp, fields, extra=(), aj_extra=(), roles=None):
    """fields = {name: (blobs_seed, plan, holdout_plan)}; returns ctx."""
    tiles = []
    for name, (seed, plan, hold) in fields.items():
        blobs = truth_sky(seed)
        exps = make_tile_exps(name, blobs, plan, seed)
        ho = make_tile_exps(name + "_ho", blobs, hold, seed + 1000)
        for e in ho:
            e["holdout"] = True
        thead = {"NAXIS1": SIZE, "NAXIS2": SIZE}
        t = J.Tile(name, name, thead, exps + ho, SIZE, 0, sel=None, role=(roles or {}).get(name, "train"))
        tiles.append(t)
    a = small_args(tmp, extra)
    aj, _ = J.parse_args(["--out", str(tmp), "--no-apertures", "--inner-sky", "2", "--visit-cycles", "5", "--field-floor", "0.3"] + list(aj_extra) + list(extra))
    aj.apertures = False
    ctx = J.assemble(tiles, a, aj, DEV, tmp, reg=reg(), bandpass=(torch.tensor([[1.45, 1.5, 1.55], [2.45, 2.5, 2.55]]), torch.full((2, 3), 1 / 3.0)), classical_fn=classical)
    return ctx


def fresh_fit(ctx, tiles=None, tag="main"):
    return J.Fit(ctx, ctx.tiles if tiles is None else tiles, tag)


def train_cycles(fit, out, cycles, K=None):
    fit.a.steps, fit.a.budget_min = cycles, 0.0
    fit.a.ckpt_every_min = 1e9
    fit.a.val_every_min = 0.0
    LOG = {}
    fit.train(out, LOG, resume=False)
    return LOG


def t1_src(fit, t, steps=40):
    recs, *_ = fit.score_tile(t, "hold", steps, tag="TEST ")
    p = J.TF.pool_records(recs)
    return p


# ------------------------------------------------------------------------------------------------ tests
def test_opfield_unchanged():
    p = os.path.join(ROOT, "scripts", "opfield.py")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    f = os.path.join(HERE, "opfield.sha256")
    if not os.path.exists(f):
        open(f, "w").write(h + "  opfield.py\n")
        print("  opfield.sha256 written (first run)")
    else:
        assert open(f).read().split()[0] == h, "opfield.py changed"
    print("  opfield.py sha256", h[:16])


def test_inner_steps_global_identical():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(tmp, {"F1": (1, {"bA": 3, "bB": 3}, {"bA": 1})})
        fit = fresh_fit(ctx)
        train_cycles(fit, tmp, 4)
        r = fit.selfcheck(ctx.tiles[0], 3)
        assert r["ok"], r
        assert r["d_global"] == 0.0 and r["d_nuisance"] == 0.0 and r["d_sky"] > 0 and r["operator_adam_unchanged"]


def test_single_tile_k0_trains():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(tmp, {"F1": (1, {"bA": 3, "bB": 3}, {"bA": 1})}, aj_extra=("--inner-sky", "0"))
        fit = fresh_fit(ctx)
        LOG = train_cycles(fit, tmp, 60)
        h = LOG["train"]["main"]["loss_history"]
        assert LOG["train"]["main"]["cycles"] == 60 and np.isfinite(h[-1][1])
        print("  K=0 loss history", [round(x[1], 2) for x in h[:1] + h[-1:]])


def run_pair(cycles=400):
    with tempfile.TemporaryDirectory() as tmp:
        rich = (1, {"bA": 8, "bB": 8}, {})
        thin = (2, {"bA": 2, "bB": 2}, {"bA": 2, "bB": 2})
        out = {}
        ctx = make_ctx(tmp, {"RICH": rich, "THIN": thin})
        fit = fresh_fit(ctx)
        train_cycles(fit, tmp, cycles)
        thin_t = next(t for t in ctx.tiles if t.key == "THIN")
        out["joint"] = t1_src(fit, thin_t)
        ctx2 = make_ctx(tmp, {"THIN": thin})
        fit2 = fresh_fit(ctx2)
        train_cycles(fit2, tmp, cycles)
        out["separate"] = t1_src(fit2, ctx2.tiles[0])
        return out


def test_joint_beats_separate():
    r = run_pair()
    j, s = r["joint"], r["separate"]
    print(f"  thin field held-out chi2 (all): joint {j['chi2_all']:.3f} vs separate {s['chi2_all']:.3f}; robust joint {j['robust_all']:.3f} vs {s['robust_all']:.3f}")
    assert j["chi2_all"] < s["chi2_all"]


def test_leave_out_local_only():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_ctx(tmp, {"RICH": (1, {"bA": 6, "bB": 6}, {}), "NEW": (3, {"bA": 3, "bB": 3}, {"bA": 1, "bB": 1})}, roles={"NEW": "leftout"})
        fit = fresh_fit(ctx, ctx.tiles)
        assert [t.key for t in fit.train_tiles] == ["RICH"]
        train_cycles(fit, tmp, 40)
        new = next(t for t in ctx.tiles if t.key == "NEW")
        info = fit.local_refit(new, 60, "LEAVE-OUT")
        assert info["global_param_max_abs_delta"] == 0.0, info
        p = t1_src(fit, new)
        assert np.isfinite(p["chi2_all"])
        print(f"  left-out field: global delta {info['global_param_max_abs_delta']}, held-out chi2 {p['chi2_all']:.3f}")


# ================================================================================================ world 2: two fields, two instruments (PSF width and pixel scale differ)
PIX = {"iA": 1.0, "iB": 2.0}          # target px per detector px
SIG2 = {"iA": 1.1, "iB": 3.0}         # PSF sigma [target px]


def make_tile_exps2(tag, blobs, plan, seed, size=SIZE):
    g = np.random.RandomState(seed)
    exps = []
    for band, n in plan.items():
        inst = BAND_INST[band]
        s = PIX[inst]
        nd = int(round(size / s))
        yy, xx = torch.meshgrid(torch.arange(nd).float(), torch.arange(nd).float(), indexing="ij")
        for k in range(n):
            ox, oy = g.uniform(-0.5, 0.5, 2) * s
            tx, ty = (s * (xx + 0.5) - 0.5 + ox).reshape(-1), (s * (yy + 0.5) - 0.5 + oy).reshape(-1)
            clean = render(blobs, tx, ty, SIG2[inst])
            d = clean + NOISE * torch.from_numpy(g.randn(clean.numel()).astype(np.float32))
            exps.append(dict(name=f"{tag}_{band}_{k}_cal.fits", band=band, detname="D" + band, d=d, ivar=torch.full_like(d, 1.0 / NOISE ** 2), tx=tx.clone(), ty=ty.clone(),
                             dx=(xx.reshape(-1) - nd / 2) / nd, dy=(yy.reshape(-1) - nd / 2) / nd, row=yy.reshape(-1).long(), col=xx.reshape(-1).long(), bkg0=0.0,
                             jac=dict(J=[[s, 0.0], [0.0, s]], theta_deg=0.0, parity=1, scale=s), mjd=59000.0 + k, exptime=100.0, path=None, holdout=False))
    return exps


def make_ctx2(tmp, specs, extra=(), aj_extra=()):
    """specs: list of dict(key, field, seed, plan, hold, role). The sky of a tile is its own random blob field (seed)."""
    tiles = []
    for sp in specs:
        blobs = truth_sky(sp["seed"])
        exps = make_tile_exps2(sp["key"], blobs, sp["plan"], sp["seed"])
        ho = make_tile_exps2(sp["key"] + "_ho", blobs, sp.get("hold", {}), sp["seed"] + 1000)
        for e in ho:
            e["holdout"] = True
        tiles.append(J.Tile(sp["key"], sp["field"], {"NAXIS1": SIZE, "NAXIS2": SIZE}, exps + ho, SIZE, 0, sel=None, role=sp.get("role", "train")))
    a = small_args(tmp, extra)
    aj, _ = J.parse_args(["--out", str(tmp), "--no-apertures", "--inner-sky", "2", "--visit-cycles", "5", "--field-floor", "0.3"] + list(aj_extra) + list(extra))
    aj.apertures = False
    for d in ("cubes", "kernels"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    return J.assemble(tiles, a, aj, DEV, tmp, reg=reg(), bandpass=(torch.tensor([[1.45, 1.5, 1.55], [2.45, 2.5, 2.55]]), torch.full((2, 3), 1 / 3.0)), classical_fn=classical)


def pooled(fit, ctx, keys, steps=40):
    recs = []
    for t in ctx.tiles:
        if t.key in keys:
            r, *_ = fit.score_tile(t, "hold", steps, tag="TEST ")
            recs += r
    return J.TF.pool_records(recs)


WORLD2 = dict(X1=dict(key="X1", field="X", seed=11, plan={"bA": 4, "bB": 8}, hold={"bA": 2, "bB": 3}),
              X2=dict(key="X2", field="X", seed=12, plan={"bA": 4, "bB": 8}, hold={"bA": 2, "bB": 3}),
              Y=dict(key="Y", field="Y", seed=2, plan={"bA": 2, "bB": 4}, hold={"bA": 2, "bB": 3}),
              X3=dict(key="X3", field="X", seed=13, plan={"bA": 2, "bB": 4}, hold={"bA": 2, "bB": 3}))
CYC = 250
_cache = {}


def world2_fits():
    """joint (X1 X2 Y), X only (X1 X2; Y left out; X3 held-out tile), Y only. Cached: 4 fits are the cost of this file."""
    if "r" in _cache:
        return _cache["r"]
    tmp = tempfile.mkdtemp()
    W = WORLD2
    r = {"tmp": tmp}
    ctx_j = make_ctx2(tmp, [W["X1"], W["X2"], W["Y"]])
    fj = fresh_fit(ctx_j)
    train_cycles(fj, tmp, CYC)
    r["joint"] = {"X": pooled(fj, ctx_j, {"X1", "X2"}), "Y": pooled(fj, ctx_j, {"Y"})}
    r["joint_fit"], r["joint_ctx"] = fj, ctx_j
    # per-instrument kernel widths of the joint fit (second moment of the effective kernel, target px)
    ctx_y = make_ctx2(tmp, [W["Y"]])
    fy = fresh_fit(ctx_y)
    train_cycles(fy, tmp, CYC)
    r["sepY"] = pooled(fy, ctx_y, {"Y"})
    out_x = os.path.join(tmp, "x")
    os.makedirs(out_x)
    for d in ("cubes", "kernels"):
        os.makedirs(os.path.join(out_x, d))
    ctx_x = make_ctx2(out_x, [W["X1"], W["X2"], dict(W["Y"], role="leftout"), dict(W["X3"], role="heldout")],
                      aj_extra=("--ht-steps", "120", "--ht-control", "--ht-control-cycles", str(CYC // 2), "--lo-steps", "120", "--lo-ft-cycles", "30"))
    ctx_x.aj.leave_out_field = "Y"
    fx = fresh_fit(ctx_x)
    train_cycles(fx, out_x, CYC)
    r["sepX"] = pooled(fx, ctx_x, {"X1", "X2"})
    LOG = {"args": {}, "metrics": {}, "approximations": [], "timings": {}}
    tm = J.run_all(ctx_x, LOG, out_x, fit=fx)
    r["x_tm"], r["x_ctx"], r["x_log"] = tm, ctx_x, LOG
    _cache["r"] = r
    return r


def test_two_fields_joint_beats_single_field():
    r = world2_fits()
    jx, sx, jy, sy = r["joint"]["X"], r["sepX"], r["joint"]["Y"], r["sepY"]
    print(f"  field X (2 tiles) held-out chi2 all: joint {jx['chi2_all']:.3f} vs X-only {sx['chi2_all']:.3f}; robust {jx['robust_all']:.3f} vs {sx['robust_all']:.3f}")
    print(f"  field Y (1 tile)  held-out chi2 all: joint {jy['chi2_all']:.3f} vs Y-only {sy['chi2_all']:.3f}; robust {jy['robust_all']:.3f} vs {sy['robust_all']:.3f}")
    assert jy["chi2_all"] < sy["chi2_all"] and jx["chi2_all"] < sx["chi2_all"]
    # three tiles, two fields, two instruments in one table
    ctx = r["joint_ctx"]
    assert len(ctx.tiles) == 3 and {t.field for t in ctx.tiles} == {"X", "Y"} and ctx.reg.instruments == ["iA", "iB"]
    assert all(t.cycles > 0 for t in ctx.tiles), [t.cycles for t in ctx.tiles]
    # the shared operator learned both pixel scales: effective Gaussian width of the fitted kernels in target px
    m = r["joint_fit"].model
    assert m.nuis.z.shape[0] == ctx.gt.n


def test_leave_one_field_out_close_to_joint():
    r = world2_fits()
    lo = r["x_tm"]["leave_out"]
    ent = lo["tiles"]["Y"]
    assert ent["global_param_max_abs_delta"] == 0.0
    ly, jy, sy = ent["T1"]["chi2_all"], r["joint"]["Y"]["chi2_all"], r["sepY"]["chi2_all"]
    print(f"  Y left out: local-only T1 chi2 all {ly:.3f}; joint {jy:.3f}; Y-only {sy:.3f}; with fine-tune {ent['T1_finetune']['chi2_all']:.3f}")
    assert ly < sy, "an operator learned on field X should serve Y better than Y alone"
    assert ly < 1.25 * jy, "leave-field-out is close to the joint fit"
    assert np.isfinite(ent["T1_finetune"]["chi2_all"]) and ent["finetune_cycles"] == 30


def test_t5_beats_control():
    r = world2_fits()
    t5 = r["x_tm"]["T5"]["tiles"]["X3"]
    a_, c_ = t5["T5"]["chi2_all"], t5["T5_control"]["chi2_all"]
    print(f"  T5 held-out tile X3: operator fixed {a_:.3f} (robust {t5['T5']['robust_all']:.3f}) vs control (operator refitted on the tile alone) {c_:.3f} ({t5['T5_control']['robust_all']:.3f})")
    assert t5["global_param_max_abs_delta"] == 0.0
    assert a_ < c_


def test_swap_gradients_and_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        W = WORLD2
        ctx = make_ctx2(tmp, [W["X1"], W["X2"], W["Y"]])
        fit = fresh_fit(ctx)
        train_cycles(fit, tmp, 6)
        tA, tB, tC = ctx.tiles
        m = fit.model

        def grads(f, t, bt):
            f.model.zero_grad(set_to_none=True)
            lv = f.model.make_levels(0.0)
            out_ = f.model(bt, f.model.nuis, lv)
            data, _ = f.data_loss(bt, out_, 0.5, 0.0)
            (data + f.sky_regs()).backward()
            g = {n: p.grad.detach().clone() for n, p in f.model.named_parameters() if p.grad is not None}
            return g

        fit.activate(tA)
        bt = fit.pool.draw(256, 0.3, torch.Generator().manual_seed(3))
        g0 = grads(fit, tA, bt)
        assert any(n == "sky" for n in g0) and len(g0) > 10
        fit.activate(tB)
        fit.activate(tC)
        assert fit.skies[tA.key].sky.device.type == "cpu" and fit.skies[tB.key].sky.device.type == "cpu"
        fit.activate(tA)
        g1 = grads(fit, tA, bt)
        assert set(g0) == set(g1) and all(torch.equal(g0[k], g1[k]) for k in g0), [k for k in g0 if not torch.equal(g0[k], g1[k])]
        # a fit that never swapped (single tile) with the same operator, nuisances and sky
        f1 = J.Fit(ctx, [tA], "plain")
        f1.load_op_state(fit.op_state())
        f1.activate(tA)
        with torch.no_grad():
            f1.model.sky.copy_(fit.model.sky)
        g2 = grads(f1, tA, bt)
        d = max(float((g0[k] - g2[k]).abs().max()) for k in g0)
        print(f"  gradients: swapped vs plain-in-place identical ({len(g0)} tensors); vs a never-swapped fit max |diff| {d:.1e}")
        assert d == 0.0
        # pool chunks tile the pool exactly: sum of per-pixel loss gradient over the chunks = over the whole pool
        store = tA.get_store()
        N = store["d"].numel()
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
        parts = [perm[c::3] for c in range(3)]
        assert sum(int(p.numel()) for p in parts) == N and torch.equal(torch.cat(parts).sort().values, torch.arange(N))
        full = J.GPool(store, None, DEV)
        tot = None
        for idx in [None] + parts:
            pool = full if idx is None else J.GPool(store, idx, DEV)
            fit.model.zero_grad(set_to_none=True)
            acc = 0.0
            for s0 in range(0, pool.N, 400):
                bt_ = pool.get(torch.arange(s0, min(pool.N, s0 + 400)))
                out_ = fit.model(bt_, fit.model.nuis, fit.model.make_levels(0.0))
                nll, _ = O.nll_terms(bt_["d"], out_["pred"], bt_["sp"], torch.zeros_like(out_["h"]), torch.full_like(out_["pi"], O.PI_INIT))
                nll.sum().backward()
                acc += float(nll.sum().detach())
            gsky = fit.model.sky.grad.detach().clone()
            if idx is None:
                tot, ref = (acc, gsky), (acc, gsky.abs().max())
            else:
                tot = (tot[0] - acc, tot[1] - gsky)
        print(f"  chunks: whole pool minus the 3 chunks: residual loss {tot[0]:.2e} (of {ref[0]:.1f}), residual sky gradient {float(tot[1].abs().max()):.2e} (of {float(ref[1]):.2e})")
        assert abs(tot[0]) < 1e-4 * abs(ref[0]) and float(tot[1].abs().max()) < 1e-4 * float(ref[1])


def test_shared_exposure_ids():
    with tempfile.TemporaryDirectory() as tmp:
        W = WORLD2
        blobs = truth_sky(11)
        e1 = make_tile_exps2("S", blobs, {"bA": 3, "bB": 3}, 5)
        e2 = [dict(e) for e in e1[:4]] + make_tile_exps2("S2", blobs, {"bA": 1}, 6)       # four exposures appear in both tiles
        t1 = J.Tile("T1", "X", {"NAXIS1": SIZE, "NAXIS2": SIZE}, e1, SIZE, 0, sel=None)
        t2 = J.Tile("T2", "X", {"NAXIS1": SIZE, "NAXIS2": SIZE}, e2, SIZE, 0, sel=None)
        t3 = J.Tile("T3", "Y", {"NAXIS1": SIZE, "NAXIS2": SIZE}, [dict(e) for e in make_tile_exps2("H", blobs, {"bA": 3, "bB": 3}, 7)], SIZE, 0, sel=None, role="heldout")
        a = small_args(tmp)
        aj, _ = J.parse_args(["--out", str(tmp), "--no-apertures"])
        aj.apertures = False
        ctx = J.assemble([t1, t2, t3], a, aj, DEV, tmp, reg=reg(), bandpass=(torch.tensor([[1.45, 1.5, 1.55], [2.45, 2.5, 2.55]]), torch.full((2, 3), 1 / 3.0)), classical_fn=classical)
        gt = ctx.gt
        assert gt.n == 6 + 1 + 6, gt.n      # 6 + 1 new in T2, 6 private in T3
        for e in e1[:4]:
            assert t1.eid[e["name"]] == t2.eid[e["name"]]
        assert len(set(t3.eid.values()) & (set(t1.eid.values()) | set(t2.eid.values()))) == 0 and all("@T3" in gt.keys[i] for i in t3.eid.values())
        # a tile's own train exposure ids
        assert set(t1.train_eids) >= set(t2.train_eids) & set(t1.train_eids) and len(set(t1.train_eids) & set(t2.train_eids)) == 4
        fit = J.Fit(ctx, [t1, t2, t3], "main")
        assert fit.train_eids == sorted(set(t1.train_eids) | set(t2.train_eids))


def test_fields_config():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"fields": [{"name": "A", "type": "plan", "plan": "/x/plan.json", "tiles": [1, 2], "shifts_from": "/x/s.json", "args": {"halo_px": 32}},
                          {"name": "B", "type": "frames", "args": {"data_dir": "/d", "hst_dir": "/h", "ra": 1.0, "dec": -2.0, "coarse_shift": 1.0}}]}
        p = os.path.join(tmp, "f.json")
        json.dump(cfg, open(p, "w"))
        aj, a = J.parse_args(["--out", tmp, "--fields-config", p, "--holdout-field", "B"])
        assert aj.fields_list == ["A", "B"] and aj.leave_out_field == "B" and aj.field_specs[0]["tiles"] == [1, 2]
        fa = J.field_args(a, aj.field_specs[1])
        assert fa.data_dir == "/d" and fa.coarse_shift == 1.0 and fa.ra == 1.0 and a.data_dir != "/d"
        assert J.field_args(a, aj.field_specs[0]).halo_px == 32
        bad = dict(cfg, fields=[dict(cfg["fields"][1], args={"no_such_flag": 1})])
        json.dump(bad, open(p, "w"))
        aj2, a2 = J.parse_args(["--out", tmp, "--fields-config", p])
        try:
            J.field_args(a2, aj2.field_specs[0])
            raise AssertionError("unknown arg accepted")
        except SystemExit:
            pass
        json.dump({"fields": [{"name": "A", "type": "plan"}]}, open(p, "w"))
        try:
            J.parse_args(["--out", tmp, "--fields-config", p])
            raise AssertionError("plan field without plan accepted")
        except SystemExit:
            pass
        aj3, _ = J.parse_args(["--out", tmp, "--plan", "/x/p.json"])
        assert [s_["name"] for s_ in aj3.field_specs] == ["SMACS", "LMC"] and aj3.field_specs[1]["plan"] == "/x/p.json"


def test_single_tile_vs_opfield_train():
    """One tile, K=0, steps N: opjoint.Fit.train versus opfield.train on the same model init, pool, generator, learning rates. Reports the largest parameter difference."""
    with tempfile.TemporaryDirectory() as tmp:
        blobs = truth_sky(1)
        exps = make_tile_exps2("E", blobs, {"bA": 3, "bB": 5}, 1) + make_tile_exps2("E_ho", blobs, {"bA": 1}, 99)
        for e in exps[-1:]:
            e["holdout"] = True
        exps_o = copy.deepcopy(exps)
        a = small_args(tmp)
        aj, _ = J.parse_args(["--out", str(tmp), "--no-apertures", "--inner-sky", "0", "--visit-cycles", "1000"])
        aj.apertures = False
        t = J.Tile("E", "E", {"NAXIS1": SIZE, "NAXIS2": SIZE}, exps, SIZE, 0, sel=None)
        ctx = J.assemble([t], a, aj, DEV, tmp, reg=reg(), bandpass=(torch.tensor([[1.45, 1.5, 1.55], [2.45, 2.5, 2.55]]), torch.full((2, 3), 1 / 3.0)), classical_fn=classical)
        N = 25
        fit = fresh_fit(ctx)
        lr_sky, lr_bkg, B = fit.lr_sky_of(t), fit.lr_bkg, fit.batch_size()
        train_cycles(fit, tmp, N)
        # opfield side
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        mo = O.OpField(a, ctx.reg, ctx.gt.meta, ctx.stencil, t.A0.clone().to(DEV), ctx.Phi_nodes, ctx.w_nodes, ctx.lam_nodes, len(ctx.dets), ctx.scale_ch, DEV, det_names=ctx.dets)
        mo.set_train_rows(t.train_eids)
        for e in exps_o:
            e["det"] = ctx.dets.index(e["detname"])
        pool = O.Pool(exps_o, t.train_eids, t.src_masks, ctx.band_index, DEV)
        a2 = copy.copy(a)
        a2.steps, a2.budget_min, a2.val_every_min = N, 0.0, 0.0
        O.train(mo, pool, exps_o, t.train_eids, [], a2, B, lr_sky, lr_bkg, t.sig_b, t.src_masks, ctx.band_index, None, ctx.reg, Path(tmp), {}, DEV)
        po = dict(mo.named_parameters())
        pj = dict(fit.model.named_parameters())
        worst, name = 0.0, ""
        for k, v in po.items():
            if k == "sky":
                continue
            d = float((v.detach() - pj[k].detach()).abs().max())
            if d > worst:
                worst, name = d, k
        dsky = float((po["sky"].detach() - fit.skies[t.key].sky.detach()).abs().max())
        print(f"  K=0, 1 tile, {N} steps: max |opjoint - opfield| over operator/nuisance parameters {worst:.2e} ({name}); sky {dsky:.2e}")
        assert worst == 0.0 and dsky == 0.0     # bit-identical on the CPU in this configuration


if __name__ == "__main__":
    which = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not which or any(w in name for w in which)):
            print(name)
            fn()
    print("all ok")
