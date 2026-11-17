"""Step benchmark: milliseconds per training step and peak memory of the field-operator fit on the small synthetic field, on the CPU or on a GPU.

    python tests/bench_step.py --sky-form grid  --batch-patches 4 8 --steps 10
    python tests/bench_step.py --sky-form items --item-count 1500 6000 --batch-patches 4 --n-per-filter 8 16 --device cuda

Every option after the sizes takes one or several values; all combinations are run, one line each. Sizes:
  --batch-patches   patches per step (the cost grows about in proportion)
  --n-per-filter    exposures per filter in the field (6 train + 1 validation + 1 held out at 8); the cost per step should hardly change
  --item-count      items of the item sky (item sky only)
  --sky-pitch       sky grid pitch [arcsec] (smaller = larger grid; the field is 3.2 arcsec across)
Timing: a first fit of 2 steps warms up (kernel, allocator); then fits of N and N + --steps steps are timed and the cost of the extra steps
is (t2 - t1) / --steps, so the set-up of fit() (patch pool, optimisers) is excluded. On a GPU the clock is read after torch.cuda.synchronize().
Peak memory: torch.cuda.max_memory_allocated on a GPU; on the CPU the peak resident size of the process (it includes the set-up and the
earlier configurations of the same run, so list the sizes from small to large).
Needs only the package (default: the folder above tests/) and torch; the data are synthetic; the model is not warmed up (the cost of a step does
not depend on that). One thread on the CPU unless --threads says otherwise.
"""
import argparse
import itertools
import json
import os
import resource
import sys
import time

import torch


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), help="folder holding fieldop/")
    ap.add_argument("--sky-form", choices=["grid", "items"], default="grid")
    ap.add_argument("--batch-patches", type=int, nargs="+", default=[4])
    ap.add_argument("--n-per-filter", type=int, nargs="+", default=[8])
    ap.add_argument("--item-count", type=int, nargs="+", default=[1500])
    ap.add_argument("--sky-pitch", type=float, nargs="+", default=[0.06])
    ap.add_argument("--steps", type=int, default=10, help="steps timed")
    ap.add_argument("--device", default="auto", help="cpu, cuda or auto (cuda when available)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--item-raster", default="direct", choices=["direct", "pyramid"])
    return ap.parse_args()


def build(a, dev, bp, npf, ic, pitch):
    from fieldop.cli import default_config, make_sky
    from fieldop.model import FieldOperator
    from fieldop.synth import SynthInstrument, make_field
    torch.manual_seed(0)
    insts = [SynthInstrument("A", 0.10, 40, 6.5, 0.07, 1.5, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)]),
             SynthInstrument("B", 0.12, 34, 6.5, 0.10, 2.4, space_grad=0.15, chromatic=1.0, flat_grad=0.08, filters=[("B1", 2.0, 2.8)])]
    exps, fset, meta, _, _ = make_field(seed=3, n_per_filter=npf, n_val=1, n_held=1, insts=insts, shift_rms=0.03, centre_range=0.6, dev=dev,
                                        rot_choices=(0, 1, 2, 3), patch_px=20, truth_half=1.6)
    kw = dict(sky_pitch=pitch, render_pitch=0.02, level_nh=12, kernel_extent=1.2, region_px=13, n_lambda=3, k_hidden=32, pattern_grid=0, stripes=0,
              batch_patches=bp, inner_sky=1, budget_min=1e6, steps=2, tau_mode="fixed", val_every=2, log_every=10 ** 9, is_snr=10.0, seed=0)
    if a.sky_form == "items":
        kw.update(sky_form="items", item_count=ic, item_raster=a.item_raster)
    cfg = default_config(**kw)
    sky, _ = make_sky(exps, fset, cfg, 2.8, dev)
    model = FieldOperator(cfg, fset, meta, sky, torch.device(dev)).to(dev)
    return model, exps, cfg


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.abspath(a.pkg))
    from fieldop.train import fit
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cuda = dev.startswith("cuda")

    def clock():
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    ics = a.item_count if a.sky_form == "items" else [0]
    for bp, npf, ic, pitch in itertools.product(a.batch_patches, a.n_per_filter, ics, a.sky_pitch):
        model, exps, cfg = build(a, dev, bp, npf, ic, pitch)
        quiet = lambda s: None  # noqa: E731
        cfg.steps = 2
        fit(model, exps, cfg, log=quiet)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        cfg.steps = 3
        t0 = clock()
        fit(model, exps, cfg, log=quiet)
        t1 = clock() - t0
        cfg.steps = 3 + a.steps
        t0 = clock()
        fit(model, exps, cfg, log=quiet)
        t2 = clock() - t0
        ms = (t2 - t1) / a.steps * 1e3
        peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        n_train = sum(e.split == "train" for e in exps)
        row = {"device": dev, "sky_form": a.sky_form, "batch_patches": bp, "train_exposures": n_train, "items": ic or None, "sky_pitch": pitch,
               "ms_per_step": round(ms, 1), "peak_memory_MiB": round(peak, 0), "memory_kind": "cuda allocated" if cuda else "process resident"}
        print(json.dumps(row), flush=True)
        del model, exps


if __name__ == "__main__":
    main()
