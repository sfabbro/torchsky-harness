"""Golden step files of the model (written for the minimal model of MINIMAL_PRESET_2026-11-18.md, section 6, which is the only model of this package).

One fixed batch of one small synthetic problem: the full parameter state, the batch, the scalar loss of the joint step
(data term + prior of the per-exposure effects, as in train.fit) and the gradient of that loss with respect to every parameter, saved to a .pt file.
Three configurations: one instrument; two instruments; one instrument with row and column stripes.

    PYTHONNOUSERSITE=1 python tests/oracle/golden_step.py                # writes tests/oracle/results/golden_<name>.pt for the three configurations
                                                                         # (do NOT run it in the pruned package without need: the three files are the reference's)
    PYTHONNOUSERSITE=1 python tests/oracle/golden_step.py --check        # recomputes from the files and compares (the same as tests/test_oracle_golden.py)

A file holds: "state" (model.state_dict(): every parameter and buffer, the sky included), "batch" (the patches of the step: exposure id, pixel window,
importance weight and every array of the patch's pixel dictionary), "loss" (total) with "loss_parts", "grads" (name -> tensor, or None for a parameter
that the loss does not reach: every parameter of a switched-off term, and the variability scales' raw values only when their axis is on), "recipe" (how the
synthetic problem is made again: used by the test, not needed by a re-implementation), "config" (the options as a plain dict).

Float64: inside float64_mode() (default dtype float64, torch.Tensor.float patched to give float64, the level grids of the renderer and every floating tensor
of the exposures converted), see FLOAT64_NOTE. The package itself is not edited. The render jitter is OFF in the golden step (it is a random shift of the finest kernel grid; a re-implementation is checked on the unjittered loss).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(HERE, "results")

from fieldop import cli, train  # noqa: E402
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402

NAMES = ("one_instrument", "two_instruments", "one_instrument_stripes")
FLOAT64_NOTE = ("float64. The package hard-codes float32 in a few places (Exposure.pix and region_nodes call .float(), the level grids of render.Levels are built as "
                "float32); inside float64_mode() those are made float64 from outside (torch.Tensor.float patched, level grids converted), nothing in fieldop/ is edited. "
                "Checked by central finite differences on a few scalar parameters (key fd_check), which agree to 1e-6 only if no float32 step is left.")


@contextlib.contextmanager
def float64_mode():
    """float64 for everything built and run inside the block; restores the default dtype, torch.Tensor.float and the item rasteriser's module settings."""
    from fieldop import sky as _sky
    old = (torch.get_default_dtype(), torch.Tensor.float, _sky.PYR_HALF, _sky.ITEM_CHUNK)
    torch.set_default_dtype(torch.float64)
    torch.Tensor.float = lambda self: self.to(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old[0])
        torch.Tensor.float = old[1]
        _sky.PYR_HALF, _sky.ITEM_CHUNK = old[2:]


def instruments(name):
    a = SynthInstrument("A", 0.10, 24, 6.5, 0.07, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)])
    if name == "two_instruments":
        return [a, SynthInstrument("B", 0.12, 20, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    return [a]


def to_double(exps):
    for e in exps:
        for k in ("d", "sig", "X", "Y"):
            v = getattr(e, k)
            if v is not None and torch.is_floating_point(v):
                setattr(e, k, v.to(torch.float64))
        for k, v in list(e.extra.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                e.extra[k] = v.to(torch.float64)


def add_stripes(exps, seed=5):
    """Declare row and column stripes on every exposure and put a known pattern into the data (0.3 noise units rms in rows and in columns)."""
    g = torch.Generator().manual_seed(seed)
    for e in exps:
        ny, nx = e.shape
        e.d = e.d + 0.3 * torch.randn(ny, 1, generator=g, dtype=e.d.dtype) + 0.3 * torch.randn(1, nx, generator=g, dtype=e.d.dtype)
        e.stripes = True
        e.bkg0 = float(e.d.median())


def build_problem(name, seed=0):
    """Call inside float64_mode(). (model, exps, cfg, pool, recipe). Deterministic: the field, the model start and the perturbation of the zero-started parameters depend on `seed` only."""
    torch.manual_seed(seed)
    insts = instruments(name)
    exps, fset, meta, _, _ = make_field(seed=1, n_per_filter=4, n_held=1, insts=insts, centre_range=0.4, rot_choices=(0, 1, 2, 3), patch_px=12,
                                        truth_half=1.0)
    to_double(exps)
    if name == "one_instrument_stripes":
        add_stripes(exps)
        from fieldop.data import build_meta
        meta = build_meta(exps, ["A"], ["A"], [i.name for i in insts])
    kw = dict(sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12, n_lambda=3, k_hidden=16,
              item_count=300, batch_patches=6, steps=0, budget_min=1e6, log_every=2, seed=seed, held_steps=6, held_reg_steps=4,
              gauge_ref_inst=0 if len(insts) > 1 else None)
    cfg = default_config(**kw)
    cli.apply_sky_module_options(cfg)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    model.double()
    for lv in model.levels_ch:                     # the renderer's level grids are built as float32
        lv.u = [u.double() for u in lv.u]
        lv.w = [w.double() for w in lv.w]
    # parameters that start at exactly zero (last layer of the throughput net, codes, gains, affine, stripes, noise scales) get a small seeded value, so that
    # every gradient of the kept terms is generic
    g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.startswith("sky.") or n.startswith("tau."):
                continue
            if bool((p == 0).all()):
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        model.bkg[:, 1:] += 0.05 * torch.randn(model.bkg[:, 1:].shape, generator=g, dtype=model.bkg.dtype)
    train.set_prior_pixels(cfg, exps, log=lambda s: None)
    tr_cuts = [i for i, e in enumerate(exps) if e.split == "train"]          # cuts == exposures here (one box)
    pool = train.PatchPool(exps, tr_cuts, cfg)
    pool.refresh(model)
    recipe = {"name": name, "seed": seed, "float64": True}
    return model, exps, cfg, pool, recipe


def joint_loss(model, exps, cfg, items, pool_mean_valid):
    """The scalar of the joint step of train.fit: importance-weighted mean data loss of the batch + prior of the batch's exposure effects / prior pixels
    (the sky regulariser of the reference package is zero for an item sky and is gone). Jitter off."""
    P = model.prepare()
    data, npx, _ = train.batch_loss(model, exps, items, P, cfg, pool_mean_valid, jitter=False)
    eids = train.batch_eids(exps, items, model.dev)
    pri = model.priors(eids) / cfg.prior_pixels
    return data + pri, {"data": data.detach(), "prior": pri.detach()}


def batch_arrays(exps, items):
    out = []
    for eid, sl, w in items:
        pix = exps[eid].pix(sl)
        out.append({"eid": int(exps[eid].eid), "cut": int(eid), "window": [int(v) for v in sl], "weight": float(w),
                    "pix": {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in pix.items()}})
    return out


def fd_check(model, exps, cfg, items, mean_valid, loss, grads_by_name, eps=1e-6):
    """Central finite difference of the loss along one element of a few scalar parameters against the autograd gradient: {name[index]: (autograd, numeric)}."""
    out = {}
    e0 = int(exps[items[0][0]].eid)                                   # an exposure that is in the batch
    d0, f0 = int(model.e_det[e0]), int(model.e_filt[e0])
    picks = [("hs", (d0, f0)), ("bkg", (e0, 1)), ("bkg", (e0, 0)), ("u", (e0,)), ("codes.z", (e0, 0)), ("affine", (e0, 4)), ("affine", (e0, 1)), ("floor_raw", (0,)),
             ("gbar.4.bias", (0,)), ("codes.hbar.4.bias", (0,)), ("net.hid.0.bias", (3,))]
    if exps[items[0][0]].stripes:
        picks += [("stripe_rows", (e0, 3)), ("stripe_cols", (e0, 5))]
    params = dict(model.named_parameters())
    for n, idx in picks:
        if n not in params or grads_by_name.get(n) is None:
            continue
        p = params[n]
        with torch.no_grad():
            p[idx] += eps
        lp = float(joint_loss(model, exps, cfg, items, mean_valid)[0])
        with torch.no_grad():
            p[idx] -= 2 * eps
        lm = float(joint_loss(model, exps, cfg, items, mean_valid)[0])
        with torch.no_grad():
            p[idx] += eps
        out[f"{n}{list(idx)}"] = (float(grads_by_name[n][idx]), (lp - lm) / (2 * eps))
        if abs(out[f"{n}{list(idx)}"][0]) < 1e-12:
            out[f"{n}{list(idx)}"] += ("gradient is zero",)
    return out


def make_golden(name):
    with float64_mode():
        return _make_golden(name)


def _make_golden(name):
    model, exps, cfg, pool, recipe = build_problem(name)
    for draw_seed in range(200):                # the first draw with at least 4 exposures and every filter in it
        items = pool.draw(cfg.batch_patches, np.random.default_rng(draw_seed))
        if len({exps[i[0]].eid for i in items}) >= 4 and len({exps[i[0]].filt for i in items}) == len(model.fset):
            break
    recipe["draw_seed"] = draw_seed
    params = dict(model.named_parameters())
    loss, parts = joint_loss(model, exps, cfg, items, pool.mean_valid)
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    gd = {k: (None if g is None else g.detach().clone()) for k, g in zip(params, grads)}
    fd = fd_check(model, exps, cfg, items, pool.mean_valid, loss, gd)
    return {"name": name, "dtype": "float64", "torch": torch.__version__, "float64_note": FLOAT64_NOTE, "recipe": recipe, "config": cli.plain_config(cfg),
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}, "batch": batch_arrays(exps, items), "pool_mean_valid": pool.mean_valid,
            "prior_pixels": cfg.prior_pixels, "loss": float(loss.detach()), "loss_parts": {k: float(v) for k, v in parts.items()},
            "grads": gd, "fd_check": fd,
            "switched_off_parameters": sorted(n for n, g in gd.items() if g is None),
            "jitter": False}


def recompute(rec):
    with float64_mode():
        return _recompute(rec)


def load_surviving_state(model, state):
    """Load the saved state into the pruned model. Tensors of terms that no longer exist (in the file only) are skipped; a tensor the model has and the file lacks
    is an error (a surviving parameter or buffer must be in the file)."""
    own = model.state_dict()
    missing = sorted(k for k in own if k not in state)
    assert not missing, f"the golden file lacks surviving tensors: {missing}"
    model.load_state_dict({k: state[k] for k in own})
    return sorted(k for k in state if k not in own)


def _recompute(rec):
    """Rebuild the problem from the recipe, load the saved state, and recompute (loss, grads) on the saved batch; also checks the saved batch arrays
    equal the rebuilt ones."""
    model, exps, cfg, pool, _ = build_problem(rec["recipe"]["name"], rec["recipe"]["seed"])
    load_surviving_state(model, rec["state"])
    items = [(b["cut"], tuple(b["window"]), b["weight"]) for b in rec["batch"]]
    for b in rec["batch"]:
        for k, v in exps[b["cut"]].pix(tuple(b["window"])).items():
            if torch.is_tensor(v):
                assert torch.equal(v, b["pix"][k]), f"saved batch differs from the rebuilt one: {k}"
    params = dict(model.named_parameters())
    loss, parts = joint_loss(model, exps, cfg, items, rec["pool_mean_valid"])
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    return float(loss.detach()), {k: (None if g is None else g.detach()) for k, g in zip(params, grads)}


NOISE = 1e-12          # a parameter whose largest gradient element is below this fraction of the largest element of the whole step has a gradient that is zero
#                        up to rounding (the gauge directions of the throughput network and the item angles: 1e-15 to 1e-18 against 15 for the largest element)


def relative_gap(a, b):
    """max |a - b| / max |b|: the largest-element relative error of one parameter's gradient."""
    return float((a - b).abs().max()) / max(float(b.abs().max()), 1e-300)


def compare(rec, loss, grads, tol=1e-12, detail=None):
    """(worst gap, list of failures). The loss and the gradient of every parameter the pruned model has are compared with the file's; a parameter that is only in
    the file (a deleted term) is skipped; a parameter of the model that is not in the file is a failure.
    A gradient with a largest element above NOISE x the largest element of the step is compared by its relative gap (must be <= tol). A gradient below that is
    rounding noise in both versions (a gauge direction): it is compared by its absolute gap against tol x the largest element of the step.
    `detail` (a dict, optional) receives the numbers of rule B: loss gap, largest relative gap of the real gradients, largest absolute gap of the noise-level
    ones in units of the largest element, how many gradients are bit-identical, how many were compared."""
    lgap = abs(loss - rec["loss"]) / max(abs(rec["loss"]), 1e-300)
    worst = lgap
    live = [g for k, g in rec["grads"].items() if k in grads and g is not None]
    gmax = max(float(g.abs().max()) for g in live)
    bad = [(k, "surviving parameter missing from the file") for k in grads if k not in rec["grads"]]
    n_cmp = n_bits = n_noise = 0
    real_max = noise_max = 0.0
    for k, g1 in grads.items():
        if k not in rec["grads"]:
            continue
        g0 = rec["grads"][k]
        if (g0 is None) != (g1 is None):
            bad.append((k, "None mismatch"))
            continue
        if g0 is None:
            continue
        n_cmp += 1
        n_bits += int(torch.equal(g1, g0))
        if float(g0.abs().max()) < NOISE * gmax:
            n_noise += 1
            r = float((g1 - g0).abs().max()) / gmax
            noise_max = max(noise_max, r)
        else:
            r = relative_gap(g1, g0)
            real_max = max(real_max, r)
        worst = max(worst, r)
        if r > tol:
            bad.append((k, r))
    if lgap > tol:
        bad.append(("loss", lgap))
    if detail is not None:
        detail.update(loss_gap=lgap, loss_bit_identical=bool(loss == rec["loss"]), grad_rel_gap_max=real_max, noise_level_gradients=n_noise,
                      noise_level_abs_gap_over_largest_element=noise_max, grads_compared=n_cmp, grads_bit_identical=n_bits,
                      parameters_only_in_file=sorted(k for k in rec["grads"] if k not in grads))
    return worst, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--names", default=",".join(NAMES))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    for name in a.names.split(","):
        path = os.path.join(RESULTS, f"golden_{name}.pt")
        if a.check:
            rec = torch.load(path, weights_only=False)
            loss, grads = recompute(rec)
            det = {}
            worst, bad = compare(rec, loss, grads, tol=1e-12, detail=det)
            print(f"{name}: loss {rec['loss']:.17g} recomputed {loss:.17g}; worst relative gap {worst:.3e}; bad {bad}")
            print(f"   {det}")
        else:
            rec = make_golden(name)
            torch.save(rec, path)
            n_nz = sum(1 for g in rec["grads"].values() if g is not None)
            print(f"{name}: loss {rec['loss']:.12g} {rec['loss_parts']}; {len(rec['grads'])} parameters, {n_nz} with a gradient; "
                  f"{os.path.getsize(path) / 1e6:.2f} MB -> {path}")
            for k, v in rec["fd_check"].items():
                print(f"   finite difference {k}: autograd {v[0]:.9g}, numeric {v[1]:.9g}, relative gap {abs(v[0] - v[1]) / max(abs(v[0]), 1e-30):.2e} {v[2:]}")


if __name__ == "__main__":
    main()
