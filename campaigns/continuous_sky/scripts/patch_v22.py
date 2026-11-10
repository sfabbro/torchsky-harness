"""Build smacs_four_v22.py from v21: Q5a scoring upgrade (4 Nov 2026).
  * --n-test N : N test dithers per band (N=1 reproduces the v21 single-detector-image holdout exactly)
  * heldout_chi2.per_exposure : per scored exposure sums and photometry pulls
  * bootstrap_scores() : stratified (by band) bootstrap over scored exposures, stored in LOG["metrics"]["heldout_bootstrap"]
  * out/holdout_selection.json : {"holdout": [...], "validation": [...]} (reusable with --holdout-from)
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


# ---------------------------------------------------------------- 1. name parser, selection, bootstrap (module level)
NEW_FUNCS = r'''# ----------------------------------------------------------------------------- v22: names, selection, bootstrap
import re as _re

# [NNN_]jwPPPPPOOOVVV_GGSAA_EEEEE_<detector>_...   (PPPPP programme, OOO observation, VVV visit, GGSAA visit group /
# parallel seq / activity, EEEEE exposure number = dither position, detector e.g. nrcb1 or nrcblong).  The optional NNN_
# is the sorted-file index prefix present on the files of this data set.
_NAME_RE = _re.compile(r"^(?:(?P<idx>\d+)_)?jw(?P<prog>\d{5})(?P<obs>\d{3})(?P<vis>\d{3})_(?P<grp>[0-9a-z]{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)(?:_|\.|$)",
                       _re.IGNORECASE)


def parse_exposure_name(name):
    """Parse a JWST NIRCam file name. Returns a dict (prog, obs, vis, grp, exp, det, idx, visit, dither) or None.
    visit  = (obs+vis, grp): the observation/visit number plus the visit-group/activity code, so that the key
             separates both different JWST visits and different activity groups inside one visit;
    dither = (visit, exp): all detectors that share an exposure number of one visit (one dither / time stamp)."""
    m = _NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return None
    g = m.groupdict()
    out = {"prog": g["prog"], "obs": g["obs"], "vis": g["vis"], "grp": g["grp"], "exp": g["exp"], "det": g["det"].lower(),
           "idx": int(g["idx"]) if g["idx"] is not None else None}
    out["visit"] = (g["obs"] + g["vis"], g["grp"])
    out["dither"] = (out["visit"], g["exp"])
    return out


def dither_key(name):
    p = parse_exposure_name(name)
    return p["dither"] if p else (("?", "?"), str(name))  # unparsable: every file is its own dither


def visit_key(name):
    p = parse_exposure_name(name)
    return p["visit"] if p else ("?", "?")


def select_exposures(exps, n_test=1, n_val=0, holdout_names=None, no_holdout=False, bands=None):
    """Sets e["holdout"] / e["val"] on every exposure dict (keys needed: name, band, d) and returns
    {"holdout": [names], "validation": [names], "warnings": [...]}.
    * no_holdout: nothing held out (v21).  holdout_names: reuse a list (v21; --holdout-from).
    * n_test == 1 and no holdout_names: v21 rule, the single detector image with most pixels per band (bands with >= 3 images).
    * n_test > 1: per band, n_test *dithers* (all detector images sharing visit + exposure number), chosen round-robin over the
      visits of that band (visits ordered by their best dither), inside a visit by most pixels, ties by key. Deterministic.
      At least 2 dithers are always left to training/validation; if the band has fewer, fewer are held out (warning).
    * validation (n_val rounds): v21 rule on what is left, one detector image per band and round, only while >= 4 images remain."""
    bands = BANDS if bands is None else bands
    warnings = []
    for e in exps:
        e["holdout"] = False
        e["val"] = False
    if no_holdout:
        pass
    elif holdout_names is not None:
        names = set(holdout_names)
        for e in exps:
            e["holdout"] = e["name"] in names
    elif n_test <= 1:
        for b in bands:
            bl = [e for e in exps if e["band"] == b]
            if len(bl) >= 3:  # hold out the exposure covering the most of the field
                max(bl, key=lambda e: e["d"].numel())["holdout"] = True
    else:
        for b in bands:
            bl = [e for e in exps if e["band"] == b]
            dith = {}
            for e in bl:
                dith.setdefault(dither_key(e["name"]), []).append(e)
            n_eff = min(n_test, len(dith) - 2)
            if n_eff < n_test:
                warnings.append(f"{b}: {len(dith)} dithers, holding out {max(n_eff, 0)} instead of {n_test}")
            if n_eff < 1:
                continue
            px = {k: sum(e["d"].numel() for e in v) for k, v in dith.items()}
            byvis = {}
            for k in dith:
                byvis.setdefault(k[0], []).append(k)
            for v in byvis.values():
                v.sort(key=lambda k: (-px[k], k))
            order = sorted(byvis, key=lambda v: (-px[byvis[v][0]], v))
            chosen = []
            while len(chosen) < n_eff:
                for v in order:
                    if byvis[v] and len(chosen) < n_eff:
                        chosen.append(byvis[v].pop(0))
            for k in chosen:
                for e in dith[k]:
                    e["holdout"] = True
    if not no_holdout:
        for _ in range(n_val):  # validation: early stopping / selection
            for b in bands:
                bl = [e for e in exps if e["band"] == b and not e["holdout"] and not e["val"]]
                if len(bl) >= 4:
                    max(bl, key=lambda e: e["d"].numel())["val"] = True
    return {"holdout": [e["name"] for e in exps if e["holdout"]], "validation": [e["name"] for e in exps if e["val"]],
            "warnings": warnings}


_BOOT_STATS = ("chi2_all", "chi2_src", "robust_all", "robust_src", "flux_err")


def _pooled_scores(recs, pulls):
    """Pooled statistics exactly as heldout_chi2 pools them: sums over exposures / total pixels; flux error = NMAD of pooled pulls
    (torch median convention, as nmad_ in heldout_chi2)."""
    n = sum(r["n"] for r in recs)
    ns = sum(r["n_src"] for r in recs)
    out = {"chi2_all": sum(r["chi2_sum"] for r in recs) / max(n, 1),
           "chi2_src": (sum(r["chi2_src_sum"] for r in recs) / ns) if ns else float("nan"),
           "robust_all": sum(r["robust_sum"] for r in recs) / max(n, 1),
           "robust_src": (sum(r["robust_src_sum"] for r in recs) / ns) if ns else float("nan")}
    v = torch.cat(pulls) if pulls else torch.zeros(0)
    out["flux_err"] = float(1.4826 * (v - v.median()).abs().median()) if v.numel() else float("nan")
    return out


def bootstrap_scores(per_exposure, B=1000, seed=0, cluster="dither"):
    """Stratified (by band) cluster bootstrap over scored exposures (the records of heldout_chi2.per_exposure; one record = one
    detector image). cluster="dither" (default): the resampling unit is a dither, i.e. all records of a band sharing dither_key(name)
    (detectors of one dither share pointing and background); cluster="image": the unit is one detector image. Each replicate
    redraws, within every band, as many units as the band has, with replacement, and recomputes the pooled statistics. Returns {stat: {"est", "p16", "p84", "p2.5", "p97.5"}, "n_exposures": {band: n},
    "single_exposure_bands": [...], "single_dither_bands": [...], "B", "seed"}."""
    assert cluster in ("dither", "image")
    by = {}  # band -> list of units; a unit is a list of records
    for r in per_exposure:
        by.setdefault(r["band"], [])
    if cluster == "dither":
        dd = {}
        for r in per_exposure:
            dd.setdefault((r["band"], dither_key(r["name"])), []).append(r)
        for (b, _), v in sorted(dd.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            by[b].append(v)
    else:
        for r in per_exposure:
            by[r["band"]].append([r])
    pul = {id(r): torch.as_tensor(r.get("pulls", []), dtype=torch.float32).reshape(-1) for r in per_exposure}
    est = _pooled_scores(list(per_exposure), [pul[id(r)] for r in per_exposure])
    rng = np.random.default_rng(seed)
    bands = sorted(by)
    samp = np.full((B, len(_BOOT_STATS)), np.nan)
    for k in range(B):
        recs, pulls = [], []
        for b in bands:
            idx = rng.integers(0, len(by[b]), len(by[b]))
            for j in idx:
                for r in by[b][j]:
                    recs.append(r)
                    pulls.append(pul[id(r)])
        sc = _pooled_scores(recs, pulls)
        samp[k] = [sc[t] for t in _BOOT_STATS]
    res = {}
    for j, t in enumerate(_BOOT_STATS):
        col = samp[:, j]
        col = col[np.isfinite(col)]
        q = np.percentile(col, [2.5, 16, 84, 97.5]) if col.size else [float("nan")] * 4
        res[t] = {"est": est[t], "p16": float(q[1]), "p84": float(q[2]), "p2.5": float(q[0]), "p97.5": float(q[3])}
    nimg = {b: sum(len(u) for u in by[b]) for b in bands}
    res["n_exposures"] = nimg
    res["n_dithers"] = {b: len({dither_key(r["name"]) for u in by[b] for r in u}) for b in bands}
    res["single_exposure_bands"] = [b for b in bands if nimg[b] == 1]
    res["single_dither_bands"] = [b for b in bands if res["n_dithers"][b] == 1]
    res["cluster"], res["B"], res["seed"] = cluster, B, seed
    res["note"] = ("resampling unit = " + ("one dither (all its scored detector images)" if cluster == "dither" else "one detector image")
                   + "; dithers of one visit are treated as independent. Bands in single_dither_bands (cluster mode) or "
                   "single_exposure_bands contribute no resampling variance (their variance is not seen).")
    return res


def boot_line(name, bs):
    def f(t):
        d = bs[t]
        return f"{d['est']:.4f} [{d['p16']:.4f}, {d['p84']:.4f}]"
    msg = (f"BOOT {name}: src a {f('chi2_src')} | all {f('chi2_all')} | robust src {f('robust_src')} | robust all {f('robust_all')} "
           f"| flux err {f('flux_err')}  (68% interval, B={bs['B']})")
    msg += " | n_dithers " + ",".join(f"{b}:{n}" for b, n in bs["n_dithers"].items())
    one = bs["single_dither_bands"] if bs["cluster"] == "dither" else bs["single_exposure_bands"]
    if one:
        msg += f" | SINGLE-{'DITHER' if bs['cluster'] == 'dither' else 'EXPOSURE'} bands (variance not seen): {','.join(one)}"
    return msg


'''
rep("def rgb_from_cube(", NEW_FUNCS + "def rgb_from_cube(")

# ---------------------------------------------------------------- 2. per-exposure records in heldout_chi2
rep("""    resid = {}
    for i in hold_idx:
        e = exps[i]
        sub = [e]
""", """    resid = {}
    per_exposure = []  # v22: one record per scored exposure
    heldout_chi2.per_exposure = per_exposure
    for i in hold_idx:
        e = exps[i]
        sub = [e]
""")
rep("""acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
""", """acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
            pe = {"name": e["name"], "band": e["band"], "det": int(e["det"]), "n": int(r.numel()), "n_src": 0, "chi2_sum": float(chi.sum()),
                  "chi2_src_sum": 0.0, "robust_sum": float(rh.sum()), "robust_src_sum": 0.0, "pulls": []}
            per_exposure.append(pe)
""")
rep("""acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
""", """acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
                pe["n_src"], pe["chi2_src_sum"], pe["robust_src_sum"] = int(sm.sum()), float(chi[sm].sum()), float(rh[sm].sum())
""")
rep("""P["dm"].append(dm.cpu()); P["pull"].append(pull.cpu()); P["dx"].append(dx_.cpu()); P["dy"].append(dy_.cpu())
""", """P["dm"].append(dm.cpu()); P["pull"].append(pull.cpu()); P["dx"].append(dx_.cpu()); P["dy"].append(dy_.cpu())
                        pe["pulls"] = [float(x) for x in pull.cpu().tolist()]
""")

# ---------------------------------------------------------------- 3. flags
rep("""    ap.add_argument("--patience", type=int""", """    ap.add_argument("--n-test", type=int, default=1, help="test exposures per band: 1 = v21 (one detector image); N>1 = N dithers (all their detector images), spread over visits")
    ap.add_argument("--boot-b", type=int, default=1000, help="bootstrap replicates for the held-out scores")
    ap.add_argument("--patience", type=int""")

# ---------------------------------------------------------------- 4. selection in run()
rep("""    for e in exps:
        e["holdout"] = False
    if getattr(a, "no_holdout", False):
        pass  # refit on every exposure (scored in-sample with --score-all)
    elif getattr(a, "holdout_from", None):
        names = set(json.load(open(a.holdout_from))["holdout"])
        for e in exps:
            e["holdout"] = e["name"] in names
    else:
        for b in BANDS:
            bl = [e for e in exps if e["band"] == b]
            if len(bl) >= 3:  # hold out the exposure covering the most of the field
                max(bl, key=lambda e: e["d"].numel())["holdout"] = True
    for e in exps:
        e["val"] = False
    for _ in range(0 if getattr(a, "no_holdout", False) else getattr(a, "n_val", 0)):  # validation: early stopping / selection
        for b in BANDS:
            bl = [e for e in exps if e["band"] == b and not e["holdout"] and not e["val"]]
            if len(bl) >= 4:
                max(bl, key=lambda e: e["d"].numel())["val"] = True
""", """    sel = select_exposures(exps, n_test=getattr(a, "n_test", 1), n_val=getattr(a, "n_val", 0),
                           holdout_names=(json.load(open(a.holdout_from))["holdout"] if getattr(a, "holdout_from", None) else None),
                           no_holdout=getattr(a, "no_holdout", False))
    for w_ in sel["warnings"]:
        print("selection warning:", w_, flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump({"holdout": sel["holdout"], "validation": sel["validation"], "warnings": sel["warnings"]},
              open(Path(out) / "holdout_selection.json", "w"), indent=1)
""")

# ---------------------------------------------------------------- 5. bootstrap in the TEST loop
rep("""        resid_all[name] = rs
        tick(f"TEST {name}:""", """        resid_all[name] = rs
        per_exp = list(heldout_chi2.per_exposure)
        LOG["metrics"].setdefault("heldout_per_exposure", {})[name] = [{k: v for k, v in r_.items() if k != "pulls"} for r_ in per_exp]
        try:
            bs = bootstrap_scores(per_exp, B=getattr(a, "boot_b", 1000), seed=0)
            LOG["metrics"].setdefault("heldout_bootstrap", {})[name] = bs
            tick(boot_line(name, bs))
        except Exception as exc:  # noqa: BLE001
            tick(f"bootstrap failed for {name}: {exc!r}")
        tick(f"TEST {name}:""")

open(dst, "w").write(s)
print("wrote", dst)
