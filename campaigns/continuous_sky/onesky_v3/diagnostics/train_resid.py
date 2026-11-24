"""How well a saved fit reproduces each exposure it was given, training exposures included: one row per exposure.

    python train_resid.py --load-fit RUN/fit --field LMC --split dither --method-name NAME --out RUN --protocol-root ROOT --svo-cache DIR

Why: a held-out score says nothing about one training exposure. An exposure at a wrong position, or a bad one, shows here as a residual far
above that of the other exposures of its visit. Written for the LMC exposure that the registration placed 1.9 arcsec from its own visit.

What is computed, per exposure e with row k of the fit: the model m at every pixel of the cut, rendered with the fit's own terms of row k
(position, gain, background, kernel code; NIRCam stripes re-solved in closed form on the residual, as every prediction of the library does),
then chi2 / N = mean ((d - m) / sigma)^2 over the valid pixels, sigma = the pipeline's error. Given for all valid pixels and for the bright
ones (d / sigma > 50). Also the translation the fit holds for the exposure (model.affine[k, 4:6], gauge included).

Stated limits: for a training exposure this is a residual on pixels the fit saw (optimistic, and not comparable with a held-out score); the
stripes absorb any row or column structure. It uses private parts of torchsky (the operator and the internal renderer), so it follows the
working tree the fit was made with. The command-line arguments mirror run_library_fit.py so that the same job script launches it; the fit is
never changed and nothing is written but <out>/train_resid.json.
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import torch

SVO = {"NIRCAM": ("JWST", "NIRCam"), "WFC3_IR": ("HST", "WFC3_IR"), "ACS_WFC": ("HST", "ACS_WFC")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="SMACS")
    ap.add_argument("--split", default="dither")
    ap.add_argument("--splits", default="")
    ap.add_argument("--box-id", type=int, default=0)
    ap.add_argument("--method-name", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-root", required=True)
    ap.add_argument("--svo-cache", required=True)
    ap.add_argument("--load-fit", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from torchsky.filter_profiles import bandpass_from_svo
    from torchsky.instrument import instrument_description
    from torchsky.map.exposure_fit import SkyFit, load_exposures
    from torchsky.map.exposure_fit import _fit as F
    from torchsky.map.exposure_fit._observations import observation_to_record
    from torchsky.map.exposure_fit._skyfit import _module_state

    sp = json.load(open(a.splits or os.path.join(a.protocol_root, f"splits_{a.field}.json")))
    box = next(b for b in sp["boxes"] if int(b["id"]) == a.box_id)
    half = box["size_arcsec"] / 2 + 1.0
    split = sp["splits"][a.split]
    held, train = set(split["heldout"]), set(split["train"])
    ex = [e for e in sp["exposures"] if e["rootname"] in held | train and str(a.box_id) in {str(k) for k in e.get("boxes", {})}]
    by_root = {e["rootname"]: e for e in ex}
    descs = []
    for key in sorted({e.get("instrument", "NIRCAM") for e in ex}):
        fac, ins = SVO[key]
        curves = {}
        for f in sorted({e["filter"] for e in ex if e.get("instrument", "NIRCAM") == key}):
            spec = bandpass_from_svo(band_name=f, facility=fac, instrument=ins, filter_id=f"{fac}/{ins}.{f}", prefer="any", cache_dir=a.svo_cache)
            lam = torch.as_tensor(np.asarray(spec["transmission_wavelength"], np.float64) / 1e4)
            curves[f] = (lam, torch.as_tensor(np.asarray(spec["transmission_throughput"], np.float64)))
        descs.append(instrument_description(key).with_filter_curves(curves))

    t0 = time.perf_counter()
    obs = load_exposures([e["path"] for e in ex], descs, (box["ra"], box["dec"], half))
    print(f"loaded {len(obs)} observations in {time.perf_counter() - t0:.0f} s", flush=True)
    fit = SkyFit.load(a.load_fit, device=a.device)
    names, model = fit.exposure_names, fit.operator
    meta, cfg, dev = model.meta, model.cfg, model.dev
    rows = []
    with _module_state(domain=fit._domain(), item_chunk=cfg.item_chunk), torch.no_grad():
        P = {k: v.detach() for k, v in model.prepare().items()}
        shifts = (model.affine * model.aff_scale)[:, 4:6].cpu().numpy()
        for o in obs:
            name = str(o.header["TSNAME"])
            root = next((r for r in by_root if name.startswith(r)), None)
            if name not in names or root is None:
                print(f"skipped {name}: not in the fit", flush=True)
                continue
            k = names.index(name)
            rec = observation_to_record(o, filt=meta["filt"][k], inst=meta["inst"][k], chan=meta["chan"][k], det=meta["det"][k],
                                        bandpass=model.fset.filters[meta["filt"][k]], name=name)
            e = dataclasses.replace(rec, eid=k, cid=k, extra=dict(rec.extra), d=rec.d.to(dev), sig=rec.sig.to(dev), valid=rec.valid.to(dev),
                                    X=rec.X.to(dev), Y=rec.Y.to(dev))
            m, pv, _ = F.predict_exposure(model, e, P, cfg)
            v = e.valid & pv & torch.isfinite(m)
            z2 = ((e.d - m) / e.sig)[v].double() ** 2
            snr = (e.d / e.sig)[v]
            bright = snr > 50
            row = {"name": name, "root": root, "split": "held" if root in held else "train", "visit": str(by_root[root].get("visit_key", "")),
                   "filter": str(o.band_name), "n": int(v.sum()), "chi2_n": float(z2.mean()) if int(v.sum()) else None,
                   "n_bright": int(bright.sum()), "chi2_n_bright": float(z2[bright].mean()) if int(bright.sum()) else None,
                   "median_abs_pull": float(z2.sqrt().median()) if int(v.sum()) else None,
                   "shift_mas": [round(1e3 * float(shifts[k, 0]), 1), round(1e3 * float(shifts[k, 1]), 1)]}
            rows.append(row)
            print(f"{row['split']:5s} visit {row['visit']} {name} {row['filter']}: chi2/N {row['chi2_n']:.3f} (N {row['n']}), bright {row['chi2_n_bright']} "
                  f"(N {row['n_bright']}), shift {row['shift_mas']} mas", flush=True)
    json.dump({"fit": a.load_fit, "rows": rows}, open(os.path.join(a.out, "train_resid.json"), "w"), indent=1)
    print(f"wrote {len(rows)} rows to {os.path.join(a.out, 'train_resid.json')}", flush=True)


if __name__ == "__main__":
    main()
