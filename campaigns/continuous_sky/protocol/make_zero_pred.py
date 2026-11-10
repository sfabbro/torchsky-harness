#!/usr/bin/env python3
"""make_zero_pred.py -- reference method "zero" of protocol v1: model = 0 everywhere (the scorer adds its own background offset).

Writes pred/<FIELD>/zero/<SPLIT>/<rootname>.npz (model float32 zeros, model_valid all True, meta) for every held-out exposure of
every split in splits_<FIELD>.json.  It fits nothing to the held-out data.  Its score is the sanity floor of the protocol: chi2/N of
a pure sky model is (S/N)^2-dominated, i.e. huge for bright pixels and ~1 (after the offset) only where snr_ref < 1.
"""
import argparse
import json
import os

import numpy as np

ROOT = "/arc/projects/ots/adass2026/protocol/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--splits", default="", help="comma list; default all")
    a = ap.parse_args()
    sp = json.load(open(os.path.join(a.root, f"splits_{a.field}.json")))
    ex = {e["rootname"]: e for e in sp["exposures"]}
    names = a.splits.split(",") if a.splits else list(sp["splits"])
    for s in names:
        d = os.path.join(a.root, "pred", a.field, "zero", s)
        os.makedirs(d, exist_ok=True)
        for r in sp["splits"][s]["heldout"]:
            shape = np.load(os.path.join(a.root, "heldout", a.field, r + ".npz"))["valid"].shape
            np.savez_compressed(os.path.join(d, r + ".npz"), model=np.zeros(shape, np.float32), model_valid=np.ones(shape, bool),
                                meta=json.dumps(dict(method="zero", fitted_on_heldout="nothing")))
        print(f"{a.field}/zero/{s}: {len(sp['splits'][s]['heldout'])} predictions", flush=True)


if __name__ == "__main__":
    main()
