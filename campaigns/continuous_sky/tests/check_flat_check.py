"""End-to-end check of flat_check.py in learned-map mode with a CONSTRUCTED state (no real gain fit exists yet).

The state = the reference SMACS state (fieldop_fix_200, no gain map) + a gain map for one detector built like model.PixelGain would store it:
box, observed mask, counts and information from flat_check's footprint prediction, and logg = the official log flat (gauge removed) +
Gaussian noise of sigma_p = 1 / sqrt(I_p) per pixel. flat_check must then report a correlation equal to its own expected correlation at the
noise limit (within sampling error) and a slope ~ 1.

    python tests/check_flat_check.py --ref-state STATE --metrics metrics.json --cal-dir CAL --rate-dir RATE --roots ... --out-state fake.pt
then  python flat_check.py --state fake.pt --cal-dir CAL --rate-dir RATE --roots ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import flat_check as FC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-state", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--cal-dir", required=True)
    ap.add_argument("--rate-dir", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-state", required=True)
    a = ap.parse_args()
    st = torch.load(a.ref_state, map_location="cpu", weights_only=False)
    metrics = json.load(open(a.metrics))
    flats, det = FC.load_conv(a.cal_dir, a.rate_dir, a.roots)
    box, ob, cnt, inf_, _ = FC.footprint_from_ref(st, metrics, det)
    r0, c0, nr, nc = box
    meta = {"dets": metrics["info"]["dets"], "filters": metrics["info"]["filters"]}
    w = FC.filter_weights(st, meta, det)
    have = [f for f in flats if f in w]
    ws = np.array([w[f] for f in have])
    truth = sum(wi * flats[f][r0:r0 + nr, c0:c0 + nc] for wi, f in zip(ws / ws.sum(), have))
    good = ob & np.isfinite(truth)
    tp = FC.project(np.where(good, truth, 0.0), good, box, 2)
    rng = np.random.default_rng(1)
    logg = np.where(good, tp + rng.normal(size=tp.shape) / np.sqrt(np.maximum(inf_, 1e-30)), 0.0)
    st["pixgain.logg"] = torch.tensor(logg.ravel(), dtype=torch.float32)
    st["pixgain.obs"] = torch.tensor(ob.ravel())
    st["pixgain.count"] = torch.tensor(cnt.ravel(), dtype=torch.float32)
    st["pixgain.info"] = torch.tensor(inf_.ravel(), dtype=torch.float32)
    st["pixel_gain_meta"] = {"mode": "detector", "labels": [det], "boxes": [list(box)], "offsets": [0], "poly_order": 2, "input_level": "rate",
                             "det_names": meta["dets"], "filters": meta["filters"], "gauge": {"constructed": True}}
    torch.save(st, a.out_state)
    print("wrote", a.out_state, det, box)


if __name__ == "__main__":
    main()
