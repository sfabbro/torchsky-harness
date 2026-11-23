"""The torchsky library path on real exposures, scored by protocol v1: load_exposures -> fit_sky -> predict_exposure -> pred/*.npz.

    python run_library_fit.py --field SMACS --split dither --method-name lib_base_s0 --out /arc/.../onesky_v3/lib_base_s0 \
        --protocol-root /arc/projects/ots/adass2026/protocol/v1 --svo-cache /arc/projects/ots/adass2026/svo_cache --budget-min 200

It uses only the installed torchsky (public names, plus torchsky.map.exposure_fit.experimental for the arms that are not public yet: --internal
key=value replaces entries of the private settings for this process, through experimental.internal_settings).
What it writes: <out>/fit (SkyFit.save), <out>/metrics.json, and one protocol prediction per held-out exposure.

Cross-instrument test (22 Nov 2026): --instruments keeps the exposures of the named instruments only; --sky-from DIR takes the sky of the
fit saved in DIR, holds it fixed, and fits only the response of the selected exposures (torchsky's experimental fit_response_to_sky). Every
held-out exposure also gets a side file <out>/xpred/<rootname>.npz (data, pipeline error, valid mask, model, model noise, on the cut), which
score_cross.py reads: HST exposures have no protocol mask. metrics.json gains "response": per (instrument, band) the learned mean
log-throughput against the published curve and two widths of the learned kernel.

Concept latents (22 Nov 2026): --latents "pca:2,ae:2,aeshape:3" (with --load-fit) replaces, one variant at a time, every item's intrinsic
description by its reconstruction from d numbers and predicts the held-out exposures again, under the method name <method>_<kind><d>:
  pca      the item's spectrum shape (its K log coefficients minus their mean) projected on d flux-weighted principal directions;
  ae       the same vector through a learned non-linear bottleneck of d numbers (two small networks, trained here on the items);
  aeshape  the spectrum shape and the two log widths together through a bottleneck of d numbers.
An item then costs: position (2), orientation (1), amplitude (1) and d latents (plus its two widths for pca and ae) instead of 5 + K.
The item's total coefficient sum is kept exactly. Shortcuts: the held-out exposures' own terms are the ones fitted with the uncompressed sky;
weights are the items' coefficient sums (bright items count more); nothing is refitted after compression, so this is a lower bound on what
a fit made with the bottleneck in place would reach.

Shortcuts, stated: one box (--box-id); a held-out exposure's own terms are fitted by predict_exposure with cross-fitting (each pixel is
predicted by terms fitted on the other half of the 4 x 4 pixel blocks); the kernel widths are read from the band-averaged stamp inside
0.5 arcsec, without the per-exposure code.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import time

import numpy as np
import torch

SVO = {"NIRCAM": ("JWST", "NIRCam"), "WFC3_IR": ("HST", "WFC3_IR"), "ACS_WFC": ("HST", "ACS_WFC")}


def parse_value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="SMACS")
    ap.add_argument("--split", default="dither")
    ap.add_argument("--splits", default="", help="splits file (default <protocol-root>/splits_<FIELD>.json)")
    ap.add_argument("--box-id", type=int, default=0)
    ap.add_argument("--method-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-root", required=True)
    ap.add_argument("--svo-cache", required=True)
    ap.add_argument("--budget-min", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--items", type=int, default=60000)
    ap.add_argument("--sky-pitch", type=float, default=0.04)
    ap.add_argument("--item-floor", type=float, default=0.0, help="smallest item width [arcsec]; 0 = 0.4 sky pitch")
    ap.add_argument("--reference-instrument", type=int, default=None)
    ap.add_argument("--max-exposures", type=int, default=0, help="smoke runs: keep the first N exposures of each of train and held-out")
    ap.add_argument("--train-exclude-filters", default="", help="comma list: no training exposure of these filters (their held-out stay)")
    ap.add_argument("--internal", action="append", default=[], help="key=value or key:value override of a private setting (repeatable)")
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--load-fit", default="", help="directory written by SkyFit.save: skip the fit and predict from it (same split and exposures)")
    ap.add_argument("--latents", default="", help='with --load-fit: comma list of kind:d (pca, ae, aeshape), e.g. "pca:1,pca:2,ae:2,aeshape:3"')
    ap.add_argument("--instruments", default="", help="comma list (NIRCAM, WFC3_IR, ACS_WFC): keep only these instruments' exposures")
    ap.add_argument("--train-half", type=int, default=-1, help="0 or 1: keep every second training exposure of each (filter, detector), sorted by name; the two "
                    "halves share no exposure and are scored on the same held-out exposures (split-half test of what the data determine)")
    ap.add_argument("--flat-noise", type=int, default=0, help="1: the pipeline error of every exposure is replaced by ONE number, its median over the "
                    "faintest fifth of the exposure's valid pixels, so that the Poisson noise of the sources is not in the weights (experiment of 23 Nov)")
    ap.add_argument("--noise-floor-max", type=float, default=-1.0, help="upper bound of the learned noise term proportional to the model's signal "
                    "(library default 0.2); 1e-6 switches that term off")
    ap.add_argument("--spectral-k", type=int, default=0, help="number of spectral bumps per item (0: the library's default, 12), spread over the union of the filters")
    ap.add_argument("--kernel-start", default="", help="npz of official point-source images, key '<DETECTOR>/<FILTER>' -> (wavelength nodes, N, N), "
                    "sampled at 1/--kernel-start-oversample of a detector pixel. The kernel of each NIRCam filter found there STARTS from the mean "
                    "image over nodes and detectors (torchsky's experimental fit_sky_with_kernel_start); nothing holds it there afterwards")
    ap.add_argument("--kernel-start-oversample", type=int, default=4)
    ap.add_argument("--kernel-start-wfc3ir", default="", help="npz of the published WFC3/IR point-source images (Anderson's library), key 'WFC3IR_<FILTER>' -> "
                    "(positions, N, N) at a quarter of a pixel. The kernel of each WFC3/IR filter STARTS from the mean over positions, cropped to the central "
                    "33 x 33 samples (4 pixels each side: beyond, the published values are at their noise, a quarter of them negative) and floored at 2e-6")
    ap.add_argument("--shifts-from", default="", help="with --sky-from: directory of a fit that holds BOTH these exposures and those of the --sky-from fit (a joint "
                    "fit). Each exposure starts at the translation that fit found for it, moved into the frame of the given sky; no search is made")
    ap.add_argument("--sky-from", default="", help="directory written by SkyFit.save: its sky is held fixed and only the response of these exposures is fitted")
    ap.add_argument("--train-filters", default="", help="comma list: training exposures of these filters only (held-out exposures of every filter stay)")
    a = ap.parse_args()

    from torchsky.filter_profiles import bandpass_from_svo
    from torchsky.instrument import instrument_description
    from torchsky.map.exposure_fit import SkyFit, SkyFitConfig, fit_sky, load_exposures, predict_exposure
    from torchsky.map.exposure_fit import experimental
    import torchsky

    t_all = time.perf_counter()
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "fit.log"), "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    internal = {}
    for kv in a.internal:
        k, v = kv.replace(":", "=", 1).split("=", 1)  # key:value also accepted: the session launcher drops an env value that holds "="
        internal[k] = parse_value(v)
    settings_scope = contextlib.ExitStack()  # the override lasts for the whole process
    try:
        settings_scope.enter_context(experimental.internal_settings(internal))
    except ValueError as err:
        raise SystemExit(f"--internal: {err}")
    log(f"torchsky {torchsky.__version__} torch {torch.__version__} cuda {torch.version.cuda}; private settings changed: {internal}")

    sp = json.load(open(a.splits or os.path.join(a.protocol_root, f"splits_{a.field}.json")))
    box = next(b for b in sp["boxes"] if int(b["id"]) == a.box_id)
    half = box["size_arcsec"] / 2 + 1.0  # the prototype's data margin
    split = sp["splits"][a.split]
    held, train = set(split["heldout"]), set(split["train"])
    ex = [e for e in sp["exposures"] if e["rootname"] in held | train and str(a.box_id) in {str(k) for k in e.get("boxes", {})}]
    keep_inst = [s.strip().upper() for s in a.instruments.split(",") if s.strip()]
    if keep_inst:
        ex = [e for e in ex if e.get("instrument", "NIRCAM") in keep_inst]
    only = [f for f in a.train_filters.split(",") if f.strip()]
    if only:
        ex = [e for e in ex if e["filter"] in only or e["rootname"] in held]
    excl = [f for f in a.train_exclude_filters.split(",") if f.strip()]
    if excl:
        ex = [e for e in ex if not (e["filter"] in excl and e["rootname"] in train and e["rootname"] not in held)]
    if a.train_half in (0, 1):
        groups = {}
        for e in sorted((e for e in ex if e["rootname"] in train and e["rootname"] not in held), key=lambda e: (e["filter"], e.get("detector", ""), e["rootname"])):
            groups.setdefault((e["filter"], e.get("detector", "")), []).append(e["rootname"])
        drop = {r for g in groups.values() for i, r in enumerate(g) if i % 2 != a.train_half}
        ex = [e for e in ex if e["rootname"] not in drop]
    if a.max_exposures > 0:
        ex = [e for e in ex if e["rootname"] in train][: a.max_exposures] + [e for e in ex if e["rootname"] in held][: max(2, a.max_exposures // 4)]
    by_root = {e["rootname"]: e for e in ex}
    insts = sorted({e.get("instrument", "NIRCAM") for e in ex})
    descs = []
    for key in insts:
        fac, ins = SVO[key]
        curves = {}
        for f in sorted({e["filter"] for e in ex if e.get("instrument", "NIRCAM") == key}):
            spec = bandpass_from_svo(band_name=f, facility=fac, instrument=ins, filter_id=f"{fac}/{ins}.{f}", prefer="any", cache_dir=a.svo_cache)
            lam = torch.as_tensor(np.asarray(spec["transmission_wavelength"], np.float64) / 1e4)
            curves[f] = (lam, torch.as_tensor(np.asarray(spec["transmission_throughput"], np.float64)))
        descs.append(instrument_description(key).with_filter_curves(curves))
    log(f"{a.field} box {box}: {len(ex)} exposure files ({sum(e['rootname'] in train for e in ex)} train, {sum(e['rootname'] in held for e in ex)} "
        f"held-out), instruments {insts}, filters {sorted({e['filter'] for e in ex})}")

    t0 = time.perf_counter()
    obs = load_exposures([e["path"] for e in ex], descs, (box["ra"], box["dec"], half))
    if a.flat_noise:
        flat = []
        for o in obs:
            v = o.mask.bool() & torch.isfinite(o.image) & torch.isfinite(o.err) & (o.err > 0)
            faint = v & (o.image <= torch.quantile(o.image[v].float(), 0.2))
            c = float(o.err[faint].median())
            o.err = torch.full_like(o.err, c)
            flat.append(c)
        log(f"flat noise: the pipeline error of each exposure replaced by its blank-sky median ({min(flat):.4g} to {max(flat):.4g} MJy/sr over {len(flat)} exposures)")
    n_held = 0
    for o in obs:
        name = str(o.header["TSNAME"])
        root = next((r for r in by_root if name.startswith(r)), None)
        if root is None:
            raise SystemExit(f"observation {name} matches no rootname of the split")
        o.header["TSGROUP"] = by_root[root].get("dither_group", root)
        if by_root[root].get("visit_key"):
            o.header["TSVISIT"] = str(by_root[root]["visit_key"])
        o.header["TSROOT"] = root
        if root in held:
            o.header["TSSPLIT"] = "held"
            n_held += 1
    log(f"loaded {len(obs)} observations with pixels in the box ({n_held} held-out) in {time.perf_counter() - t0:.0f} s")

    cfg = SkyFitConfig(sky_pitch=a.sky_pitch, item_count=a.items, item_width_floor=a.item_floor, budget_minutes=a.budget_min, steps=a.steps,
                       seed=a.seed, device=a.device, reference_instrument=a.reference_instrument, **({"spectral_k": a.spectral_k} if a.spectral_k else {}),
                       **({"noise_floor_max": a.noise_floor_max} if a.noise_floor_max > 0 else {}))
    t0 = time.perf_counter()
    if a.load_fit:
        fit = SkyFit.load(a.load_fit, device=a.device)
        log(f"loaded the fit of {a.load_fit}; no fitting")
    elif a.sky_from:
        src = SkyFit.load(a.sky_from, device=a.device)
        cfg = dataclasses.replace(src.config, budget_minutes=a.budget_min, steps=a.steps, seed=a.seed, device=a.device, reference_instrument=None)
        log(f"response-only fit against the sky of {a.sky_from} (instruments {src.instruments}, bands {src.bands}); sky geometry from that fit: "
            f"pitch {cfg.sky_pitch}, {cfg.item_count} items, floor {cfg.item_width_floor}")
        if a.shifts_from:
            # A control for the cross-instrument test: registration is taken from a fit that saw both instruments, so that what is measured
            # is the given sky, not the search for it. A position in the joint frame is x + t_joint; the same exposure of the given sky's own
            # instrument sits at x + t_given in the given sky's frame, so frame offset = median over those exposures of (t_given - t_joint).
            ref = SkyFit.load(a.shifts_from, device="cpu").exposure_terms
            given = src.exposure_terms
            both = sorted(n for n in given if n in ref)
            if len(both) < 3:
                raise SystemExit(f"--shifts-from: only {len(both)} exposures are in both {a.shifts_from} and {a.sky_from}")
            dt = np.array([[given[n]["affine"][4 + i] - ref[n]["affine"][4 + i] for i in (0, 1)] for n in both])
            off = np.median(dt, 0)
            sc = 1.4826 * np.median(np.abs(dt - off), 0)
            log(f"known shifts from {a.shifts_from}: frame offset ({off[0] * 1e3:+.1f}, {off[1] * 1e3:+.1f}) mas from {len(both)} exposures in both fits "
                f"(scatter {sc[0] * 1e3:.1f}, {sc[1] * 1e3:.1f} mas)")
            for o in obs:
                name = str(o.header["TSNAME"])
                if name not in ref:
                    raise SystemExit(f"--shifts-from: exposure {name} is not in {a.shifts_from}")
                o.header["TSSHIFTX"] = float(ref[name]["affine"][4] + off[0])
                o.header["TSSHIFTY"] = float(ref[name]["affine"][5] + off[1])
        fit = experimental.fit_response_to_sky(obs, cfg, src, internal=internal, log=log)
        del src
        fit.save(os.path.join(a.out, "fit"))
    elif a.kernel_start or a.kernel_start_wfc3ir:
        ks = {}
        if a.kernel_start_wfc3ir:
            lib2 = np.load(a.kernel_start_wfc3ir)
            for band in sorted({str(o.band_name) for o in obs if str(o.header["TSINST"]) == "WFC3_IR"}):
                if f"WFC3IR_{band}" not in lib2.files:
                    continue
                m = np.asarray(lib2[f"WFC3IR_{band}"], np.float64).mean(0)
                c, h = (m.shape[0] - 1) // 2, 16
                m = np.clip(m[c - h:c + h + 1, c - h:c + h + 1], 2e-6, None)
                ks[("WFC3_IR", band)] = (torch.tensor(m / m.sum(), dtype=torch.float32), 0.25)
            log(f"kernel start from {a.kernel_start_wfc3ir}: WFC3/IR {sorted(b for i, b in ks if i == 'WFC3_IR')} (mean over 9 positions, central 33 x 33 samples)")
        lib = np.load(a.kernel_start) if a.kernel_start else None
        present = sorted({str(o.band_name) for o in obs if str(o.header["TSINST"]) == "NIRCAM"}) if lib is not None else []
        for band in present:
            st = [np.asarray(lib[k], np.float64).mean(0) for k in lib.files if k.endswith("/" + band)]
            if not st:
                continue
            n = min(x.shape[0] for x in st)  # detectors differ by 1.5% in pixel scale, so their stamps differ in size: centre-crop, then mean
            st = [x[(x.shape[0] - n) // 2:(x.shape[0] - n) // 2 + n, (x.shape[1] - n) // 2:(x.shape[1] - n) // 2 + n] for x in st]
            m = np.mean(st, 0)
            ks[("NIRCAM", band)] = (torch.tensor(m / m.sum(), dtype=torch.float32), 1.0 / a.kernel_start_oversample)
        log(f"kernel start from {a.kernel_start}: NIRCam {sorted(b for i, b in ks if i == 'NIRCAM')} (mean over wavelength nodes and detectors)")
        fit = experimental.fit_sky_with_kernel_start(obs, cfg, ks, internal=internal, log=log)
        fit.save(os.path.join(a.out, "fit"))
    else:
        fit = fit_sky(obs, cfg, log=log)
        fit.save(os.path.join(a.out, "fit"))
    fit_s = time.perf_counter() - t0
    rep = fit.report
    log(f"fit done in {fit_s / 60:.1f} min: {rep.get('n_steps')} steps, {rep.get('s_per_step')} s/step, dilation max {rep.get('dilation_max')}, "
        f"noise floor {rep.get('noise_floor_final')}")
    with torch.no_grad():
        sky = fit.operator.sky
        w = sky.widths(0)
        compact = int((w.amax(1) < 2.0 * sky.pitch).sum())
        log(f"items: {sky.n_items}; larger width below 2 sky pitches: {compact} ({100.0 * compact / sky.n_items:.0f}%); smaller width quantiles "
            f"[arcsec] 5/50/95%: {[round(float(q), 4) for q in torch.quantile(w.amin(1).double().cpu(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))]}")

    response = {}
    try:
        for key, v in experimental.response_summary(fit).items():
            mlg = v["mean_log_throughput"]
            response[key] = {"mean_log_throughput": mlg, "kernel_r50_arcsec": v["kernel_r50_arcsec"],
                             "kernel_core_rms_arcsec": v["kernel_core_rms_arcsec"]}
            log(f"response {key}: mean log-throughput against the published curve {mlg:+.4f} (flux factor {v['flux_factor']:.3f}); kernel radius of "
                f"half the light inside 0.5 arcsec {1e3 * v['kernel_r50_arcsec']:.1f} mas; rms radius per axis inside 0.15 arcsec "
                f"{1e3 * v['kernel_core_rms_arcsec']:.1f} mas")
    except Exception as err:  # a read-out only: never lose the predictions of a finished fit to it
        log(f"response summary failed: {err!r}")

    def predict_all(method):
        recs = {}
        xdir = os.path.join(a.out, "xpred" if method == a.method_name else "xpred_" + method)
        os.makedirs(xdir, exist_ok=True)
        pdir = os.path.join(a.protocol_root, "pred", a.field, method, a.split)
        os.makedirs(pdir, exist_ok=True)
        t0 = time.perf_counter()
        for o in obs:
            if o.header.get("TSSPLIT") != "held":
                continue
            p = predict_exposure(fit, o, cross_fit=True)
            m = p.model.detach().float().cpu().numpy()
            ny, nx = int(o.header["TSDETNY"]), int(o.header["TSDETNX"])
            y0, x0 = int(round(o.detector_y0_pix)), int(round(o.detector_x0_pix))
            full = np.full((ny, nx), np.nan, np.float32)
            sig = np.full((ny, nx), np.nan, np.float32)  # the model's own noise at each pixel: what calibrated intervals are built from
            val = np.zeros((ny, nx), bool)
            full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
            sig[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = p.sigma.detach().float().cpu().numpy()
            val[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = np.isfinite(m)
            meta = {"method": method, "fitted_on_heldout": ["background", "affine", "gain", "kernel_code", "stripes"], "cross_fit": True,
                    "chi2": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in p.chi2.items() if isinstance(v, (int, float, str))}}
            if str(o.header["TSINST"]) == "NIRCAM":  # the protocol's exposures; other instruments have no protocol mask and are scored from xpred
                np.savez_compressed(os.path.join(pdir, f"{o.header['TSROOT']}.npz"), model=full, model_valid=val, sigma=sig, meta=json.dumps(meta))
            np.savez_compressed(os.path.join(xdir, f"{o.header['TSROOT']}.npz"), data=o.image.detach().float().cpu().numpy(),
                                err=o.err.detach().float().cpu().numpy(), valid=o.mask.detach().cpu().numpy().astype(bool), model=m,
                                sigma=p.sigma.detach().float().cpu().numpy(), band=str(o.band_name), instrument=str(o.header["TSINST"]),
                                group=str(o.header.get("TSGROUP", "")), y0=y0, x0=x0)
            recs[str(o.header["TSROOT"])] = meta["chi2"] | {"band": o.band_name}
            log(f"held-out {o.header['TSROOT']} {o.band_name}: chi2/N (pipeline error, own mask) {meta['chi2'].get('chi2')}")
        log(f"predicted {len(recs)} held-out exposures in {(time.perf_counter() - t0) / 60:.1f} min -> {pdir}")
        return recs

    recs = {} if a.no_predict else predict_all(a.method_name)
    latent_recs = {}
    variants = [v.strip() for v in a.latents.split(",") if v.strip()]
    if variants and not a.load_fit:
        raise SystemExit("--latents needs --load-fit (a finished fit)")
    if variants:
        sky = fit.operator.sky
        keep = {k: getattr(sky, k).data.clone() for k in ("lc", "lw")}
        for v in variants:
            kind, d = v.split(":")
            for k, t in keep.items():
                getattr(sky, k).data.copy_(t)
            with torch.no_grad():
                r, z = experimental.compress_items(sky, fit.operator.Phi_band, kind, int(d), a.seed, log)
            name = f"{a.method_name}_{kind}{d}"
            np.savez_compressed(os.path.join(a.out, f"latents_{kind}{d}.npz"), z=z.numpy(), pos=sky.pos.data[0].float().cpu().numpy(),
                                log_amp=torch.logsumexp(keep["lc"][0].double(), 1).float().cpu().numpy())
            r["heldout"] = predict_all(name)
            latent_recs[name] = r
        for k, t in keep.items():
            getattr(sky, k).data.copy_(t)
        sky._mut += 1
        sky._cache = None

    json.dump({"args": vars(a), "internal": internal, "report": rep, "response": response, "latents": latent_recs, "fit_seconds": fit_s, "heldout": recs,
               "total_seconds": time.perf_counter() - t_all,
               "torchsky": torchsky.__version__, "torch": torch.__version__,
               "cuda_peak_gb": (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=str)
    log(f"all done in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
