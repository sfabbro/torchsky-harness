"""Splits for the cross-instrument test on SMACS: the joint JWST + HST WFC3/IR list of 18 Nov, with HST exposures held out as well.

    python make_cross_splits.py splits_SMACS_jwst_wfc3ir.json splits_SMACS_cross.json

The input list has every HST exposure in training. Here, in each (visit, filter) group of HST exposures with at least 3 members, one in four
(at least one) is held out, drawn with a fixed seed. Groups of 2 keep both for training. JWST held-out exposures are unchanged (protocol v1).

Splits written (rootnames):
  dither        JWST training + HST training / JWST held-out + HST held-out. A run selects instruments with the runner's --instruments.
  filter_F200W  as in the input, with the HST held-out exposures moved from training to held-out.

Shortcut, stated: HST held-out exposures have no protocol mask and no reference mosaic; they are scored by score_cross.py from the runner's
side files (data, pipeline error, valid mask, model), in classes defined by a named reference method.
"""

import collections
import json
import sys

import numpy as np

src, out = sys.argv[1:3]
sp = json.load(open(src))
rng = np.random.default_rng(20261122)
groups = collections.defaultdict(list)
for e in sp["exposures"]:
    if e["instrument"] != "NIRCAM":
        groups[e["dither_group"]].append(e["rootname"])
held = []
for g, roots in sorted(groups.items()):
    roots = sorted(roots)
    if len(roots) >= 3:
        held += [roots[i] for i in sorted(rng.choice(len(roots), size=max(1, len(roots) // 4), replace=False).tolist())]
held_set = set(held)
for name in ("dither", "filter_F200W"):
    s = sp["splits"][name]
    s["train"] = [r for r in s["train"] if r not in held_set]
    s["heldout"] = list(s["heldout"]) + held
    s["n_train"], s["n_heldout"] = len(s["train"]), len(s["heldout"])
    s["rule"] = s.get("rule", "") + " | HST: one in four exposures of each (visit, filter) group of 3 or more held out (22 Nov 2026)"
sp["splits"] = {k: v for k, v in sp["splits"].items() if k in ("dither", "filter_F200W")}
sp["notes"] = (sp.get("notes") or []) + [f"make_cross_splits.py (22 Nov 2026): {len(held)} HST exposures held out: {held}"]
json.dump(sp, open(out, "w"), indent=1)
by = {e["rootname"]: e for e in sp["exposures"]}
print("HST held-out", len(held), dict(collections.Counter(by[r]["filter"] for r in held)))
for name, s in sp["splits"].items():
    print(name, "train", dict(collections.Counter((by[r]["instrument"], by[r]["filter"]) for r in s["train"])))
    print(name, "held ", dict(collections.Counter((by[r]["instrument"], by[r]["filter"]) for r in s["heldout"])))
