#!/usr/bin/env python3
"""Remove every TRAINING exposure of one filter from a list of exposures (protocol v1 format); the held-out set is unchanged, so the
held-out exposures of that filter become a band the fit never saw in training.  usage: make_filter_holdout_splits.py IN FILTER OUT"""
import sys
import json
sin, filt, out = sys.argv[1:4]
sp = json.load(open(sin))
by = {e["rootname"]: e for e in sp["exposures"]}
drop = set()
for k, v in sp["splits"].items():
    if "train" in v:
        d = {r for r in v["train"] if by.get(r, {}).get("filter") == filt}
        v["train"] = [r for r in v["train"] if r not in d]; v["n_train"] = len(v["train"])
        v["rule"] = v.get("rule", "") + f" | training exposures of {filt} removed ({len(d)}): unseen-band test"
        drop |= d if k == "dither" else set()
        print(k, "removed", len(d), "train left", v["n_train"], "held-out of that filter", sum(by[r]["filter"] == filt for r in v.get("heldout", []) if r in by))
json.dump(sp, open(out, "w"), indent=1)
