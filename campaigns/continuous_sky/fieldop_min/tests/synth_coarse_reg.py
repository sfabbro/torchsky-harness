"""Build B, test 1: two-instrument synthetic field with known translations of 0.2 to 0.8 arcsec injected on the second instrument.

Recovery by --coarse-reg xcorr (error against the injected translation, in pixels of instrument B) and the held-out score after a short fit,
against the same field without injected translations. Written as functions used by tests/test_coarse_reg.py; run as a script it prints the numbers.
    PYTHONNOUSERSITE=1 python tests/synth_coarse_reg.py [--steps 60]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fieldop import train  # noqa: E402
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.coarse_reg import coarse_register  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.synth import SynthInstrument, make_field  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402

PIX_B = 0.12


def field(inject, seed=3):
    insts = [SynthInstrument("A", 0.08, 56, 6.5, 0.05, 1.5, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", PIX_B, 40, 6.5, 0.10, 2.4, filters=[("B1", 2.0, 2.8)])]
    return make_field(seed=seed, n_per_filter=4, n_held=1, insts=insts, centre_range=0.3, rot_choices=(0, 1, 2, 3), patch_px=14,
                      truth_half=3.0, n_pts=70, inst_shift={"B": (0.1, 0.6)} if inject else None), insts


def build(inject, coarse, steps=0, log=lambda s: None, seed=3):
    torch.manual_seed(0)
    (exps, fset, meta, truth, insts_o), insts = field(inject, seed)
    cfg = default_config(gauge_ref_inst=0, sky_pitch=0.08, render_pitch=0.04, level_nh=8, kernel_extent=0.8, region_px=12,
                         n_lambda=3, k_hidden=16, stripes=0, batch_patches=3, steps=steps, budget_min=1e6, tau_mode="fixed",
                         log_every=1000, seed=0, held_steps=8, held_reg_steps=6, warm_kernel_steps=30, reg_prepass_steps=30,
                         reg_prepass_iters=1, blur_floor=0.0, coarse_reg="xcorr" if coarse else "off", patch_arcsec=2.0)
    sky, _ = make_sky(exps, fset, cfg, 3.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    finfo = {f: {"pixel_arcsec": insts[0].pixel if f < 2 else PIX_B, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
    train.warm_kernel(model, cfg, finfo, log=log)
    return model, exps, cfg


def recovery(model, exps, gauge_free=False):
    """Error of the translation of each B exposure against (a) the injected translation alone and (b) the full target: injected + the exposure's
    own header error (extra["true_shift"], 0.03 arcsec rms in this field) minus the mean header error of the A training exposures (the starting sky
    is built from them at their header positions, so it sits at the mean of their errors). gauge_free: the error is taken after removing its mean
    over all training exposures (the fit's common-translation gauge removes the mean translation, so only differences are determined)."""
    sc = model.aff_scale[4:6]
    mA = np.mean([e.extra["true_shift"] for e in exps if e.inst == 0 and e.split == "train"], 0)
    tg = lambda e: np.asarray(e.extra["inj_shift"]) + np.asarray(e.extra["true_shift"]) - mA  # noqa: E731
    tt = lambda e: (model.affine[e.eid, 4:6] * sc).detach().numpy()  # noqa: E731
    com = np.mean([tt(e) - tg(e) for e in exps if e.split == "train"], 0) if gauge_free else np.zeros(2)
    e_inj, e_full, inj = [], [], []
    for e in exps:
        if e.inst == 1:
            e_inj.append(tt(e) - np.asarray(e.extra["inj_shift"]))
            e_full.append(tt(e) - tg(e) - com)
            inj.append(np.hypot(*e.extra["inj_shift"]))
    f = lambda a: float(np.hypot(a[:, 0], a[:, 1]).max())  # noqa: E731
    e_inj, e_full = np.array(e_inj), np.array(e_full)
    return dict(n=len(e_inj), max_err_vs_injected_arcsec=f(e_inj), max_err_vs_full_target_arcsec=f(e_full),
                rms_err_vs_full_target_arcsec=float(np.sqrt((e_full ** 2).sum(1).mean())), max_err_vs_full_target_pix=f(e_full) / PIX_B,
                injected_min=float(min(inj)), injected_max=float(max(inj)), per_exposure_err_mas=(e_full * 1e3).round(1).tolist())


def heldout(model, exps, cfg):
    held = [e.eid for e in exps if e.split == "held"]
    stage = train.heldout_stage(model, exps, held, cfg, log=lambda s: None)
    sc = {}
    for c in held:
        e = exps[c]
        sc[e.name] = stage["pred"](e)[3]
    allr = train.summarise_scores(sc)["written"]
    b = [sc[e.name] for e in exps if e.split == "held" and e.inst == 1]
    nb = sum(r["n"] for r in b)
    return dict(all=allr["chi2_input_noise"], B=sum(r["chi2"] * r["n"] for r in b) / nb)


def run(inject, coarse, steps, log=lambda s: None):
    model, exps, cfg = build(inject, coarse, steps, log)
    ts = TileSet.single(model, exps, cfg, log)
    rec = coarse_register(model, exps, cfg, log=log, ts=ts, inst_names=["A", "B"]) if coarse else {}
    rec_after = recovery(model, exps, gauge_free=bool(coarse))
    train.register_prepass(model, exps, cfg, log=log, ts=ts)
    out = dict(coarse=rec, recovery_after_coarse=rec_after)
    if steps:
        train.fit(model, exps, cfg, log=log, ts=ts)
    out["recovery_after_fit"] = recovery(model, exps, gauge_free=True)
    out["heldout_chi2"] = heldout(model, exps, cfg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--which", default="inj_coarse,control,inj_nocoarse")
    a = ap.parse_args()
    for w in a.which.split(","):
        t0 = time.perf_counter()
        inject, coarse = {"inj_coarse": (1, 1), "control": (0, 0), "inj_nocoarse": (1, 0), "control_coarse": (0, 1)}[w]
        r = run(inject, coarse, a.steps, log=print if w == "inj_coarse" else (lambda s: None))
        print(f"== {w}: inject {inject} coarse {coarse}: held-out chi2/px {r['heldout_chi2']}; recovery after coarse {r['recovery_after_coarse']}; "
              f"after fit {r['recovery_after_fit']}  [{time.perf_counter() - t0:.0f} s]", flush=True)
