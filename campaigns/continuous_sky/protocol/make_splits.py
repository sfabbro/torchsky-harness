#!/usr/bin/env python3
"""make_splits.py -- fixed train / held-out splits of the evaluation protocol v1 (see INTERFACE.md).

Writes  <out>/splits_<FIELD>.json  for FIELD in {SMACS, LMC}.

Splits (every held-out list is fixed once; seed and rule are recorded in the JSON)
  dither          2 whole dither positions held out per (filter, scoring box).  A dither position is one telescope pointing in
                  one filter: ALL detector images of that exposure number and filter (SW module detectors or the LW detector).
                  SMACS: the legacy choice of /arc/projects/ots/adass2026/scripts/holdout_q5_cal.json is kept verbatim (2 dither
                  positions per band, 24 detector images) so that old runs stay comparable.  LMC: seeded random choice.
  visit           one whole visit (programme+observation+visit number, in one filter) held out per (filter, box) where the box is
                  covered by >= 2 visits in that filter.  Where it is impossible the (filter, box) is listed under "unavailable".
  epoch (LMC)     programme 1476 (2022) vs 4447 (2023/24): "epoch" = train on 1476, hold out 4447; "epoch_reverse" = the opposite.
  filter_<F>      filter F held out entirely (default F200W, a placeholder until more filters arrive; --heldout-filters).

APPROXIMATIONS / CONVENTIONS (stated, not hidden)
  A1  Scoring box membership: LMC from the planner's per-tile exposure lists (cover_frac of the plan, footprint from the sampled
      array boundary with the full WCS); SMACS from the SCI-header (SIP) WCS of each _cal on a 21x21 grid of the box.  Header WCS
      errors (up to 1.8" in LMC programme 1476) are NOT corrected here, so cover fractions near 0 or 1 are uncertain by that much.
  A2  An image "overlaps" a box when its cover_frac > 0.  A dither position "covers" a box when the summed cover_frac of its detector images
      (the detectors of one pointing do not overlap; capped at 1) is >= 0.5; the SMACS box straddles detector gaps, so no single detector
      image covers it.  A visit covers a box when one of its dither positions does.  Dither / visit units are eligible for a box when
      they cover it.  Held-out lists contain the images of the
      held-out units that overlap >= 1 box (the only ones that can be scored); the training list excludes ALL images of a held-out
      unit (also those that overlap no box), except in the SMACS legacy split (train = everything except the 24 legacy files).
  A3  Held-out units are global: an image held out for one box is held out for every box it overlaps (and is scored there too).
  A4  Each (filter, box) keeps >= --min-train-dithers (default 3) dither positions / >= 1 visit for training; if the requested 2 /
      1 would break that, fewer are held out and it is recorded ("short").  Units already held out for another box that cover this
      box count towards its quota, so a box can end up with more than 2.
  A5  Only the filters in --filters (default the six SMACS filters) are processed; files of other filters are ignored for now.
  A6  Exposure metadata come from the primary headers at run time; the identifiers (programme/observation/visit/exposure) from
      the file name, which is the JWST convention jwPPPPPOOOVVV_GGSAA_EEEEE_<detector>_cal.fits.
"""
import argparse
import datetime
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ARC = "/arc/projects/ots/adass2026"
SMACS_DIR = "/arc/projects/ots/torchsky-data/jwst_smacs0723/cal"
LMC_DIRS = ["/arc/projects/ots/torchsky-data/aggregate/1476_nircam", "/arc/projects/ots/torchsky-data/aggregate/4447_nircam"]
SMACS_BOX = dict(id=0, ra=110.8268, dec=-73.4544, size_arcsec=20.48)
SIX_FILTERS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
NAME_RE = re.compile(r"^jw(\d{5})(\d{3})(\d{3})_(\w{5})_(\d{5})_(\w+?)_cal\.fits$")
COVER_MIN, SEED_DEFAULT = 0.5, 20261110


# ----------------------------------------------------------------------------- geometry
def box_cover_frac(sci_header, box, n=21):
    """Fraction of the box (n x n sample points, tangent-plane square, north up) inside the detector array, header (SIP) WCS."""
    from astropy.wcs import WCS
    w = WCS(sci_header)
    h = box["size_arcsec"] / 2.0
    g = np.linspace(-h, h, n)
    xi, eta = np.meshgrid(g, g)
    xi, eta = np.radians(xi.ravel() / 3600.0), np.radians(eta.ravel() / 3600.0)
    ra0, dec0 = math.radians(box["ra"]), math.radians(box["dec"])
    # inverse gnomonic projection (xi east, eta north)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * math.sin(dec0) + np.where(rho > 0, eta * np.sin(c) * math.cos(dec0) / rho, 0.0))
        ra = ra0 + np.arctan2(xi * np.sin(c), rho * math.cos(dec0) * np.cos(c) - eta * math.sin(dec0) * np.sin(c))
    px, py = w.all_world2pix(np.degrees(ra), np.degrees(dec), 0)
    nx, ny = sci_header["NAXIS1"], sci_header["NAXIS2"]
    ok = (px >= -0.5) & (px <= nx - 0.5) & (py >= -0.5) & (py <= ny - 0.5)
    return float(ok.mean())


# ----------------------------------------------------------------------------- exposure table
def read_one(path, boxes_for_coverage):
    from astropy.io import fits
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    with fits.open(path, memmap=True) as hl:
        h0 = hl[0].header
        rec = dict(file=os.path.basename(path), rootname=os.path.basename(path)[:-len("_cal.fits")], path=path,
                   program=m.group(1), observation=m.group(2), visit=m.group(3), activity=m.group(4), exposure=m.group(5),
                   detector=str(h0.get("DETECTOR", m.group(6).upper())), filter=str(h0.get("FILTER", "")).strip(),
                   pupil=str(h0.get("PUPIL", "")).strip(), mjd=float(h0.get("EXPSTART", float("nan"))),
                   pa_v3=float("nan"), instrument=str(h0.get("INSTRUME", "")))
        sh = hl["SCI"].header
        rec["pa_v3"] = float(sh.get("PA_V3", float("nan")))
        if boxes_for_coverage:
            rec["boxes"] = {str(b["id"]): box_cover_frac(sh, b) for b in boxes_for_coverage}
    return rec


def list_cals(dirs):
    out = []
    for d in dirs:
        out += sorted(glob.glob(os.path.join(d, "*_cal.fits")))
    return out


def build_exposures(field, files, boxes, plan, filters, workers):
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda p: read_one(p, boxes if field == "SMACS" else None), files))
    recs = [r for r in recs if r]
    skipped = defaultdict(int)
    keep = []
    for r in recs:
        if r["instrument"] != "NIRCAM" or r["filter"] not in filters or r["pupil"] not in ("CLEAR", ""):
            skipped[f'{r["instrument"]}/{r["filter"]}/{r["pupil"]}'] += 1
            continue
        keep.append(r)
    if field == "LMC":                           # box membership from the plan (A1)
        cf = defaultdict(dict)
        for t in plan["tiles"]:
            for b, lst in t["exposures"].items():
                for e in lst:
                    cf[e["file"]][str(t["id"])] = float(e["cover_frac"])
        for r in keep:
            r["boxes"] = cf.get(r["file"], {})
    keep.sort(key=lambda r: r["rootname"])
    for r in keep:
        r["boxes"] = {k: round(v, 4) for k, v in r["boxes"].items() if v > 0}
        r["dither_group"] = group_key(r)
        r["visit_key"] = f'{r["program"]}{r["observation"]}{r["visit"]}'
    return keep, dict(skipped)


def group_key(r):
    """Dither position = exposure (visit, activity, exposure number) in one filter."""
    return f'jw{r["program"]}{r["observation"]}{r["visit"]}_{r["activity"]}_{r["exposure"]}|{r["filter"]}'


# ----------------------------------------------------------------------------- unit selection
def choose_units(rng, boxes, filters, elig, n_per, min_train, preset=()):
    """Greedy seeded choice of held-out units.  elig[(box, filter)] = sorted list of unit keys able to cover that box.
    Returns (heldout set, report dict).  See APPROXIMATIONS A3/A4."""
    covers = defaultdict(set)
    for (b, f), us in elig.items():
        for u in us:
            covers[u].add((b, f))
    H = set(preset)

    def safe(u):
        return all(len([x for x in elig[k] if x not in H and x != u]) >= min_train for k in covers[u])

    order = sorted(elig, key=lambda k: (len(elig[k]), k))        # most constrained first
    rep = {}
    for (b, f) in order:
        cand = elig[(b, f)]
        have = [u for u in cand if u in H]
        want = min(n_per, max(0, len(cand) - min_train))
        pool = sorted(u for u in cand if u not in H)
        rng.shuffle(pool)
        for u in pool:
            if len([x for x in cand if x in H]) >= want:
                break
            if safe(u):
                H.add(u)
        got = [u for u in cand if u in H]
        rep[f"{f}/box{b}"] = dict(n_candidates=len(cand), n_heldout=len(got), requested=n_per,
                                  short=len(got) < n_per, already_from_other_boxes=len(have))
    return H, rep


# ----------------------------------------------------------------------------- main
def make_field(field, a):
    plan = None
    if field == "SMACS":
        boxes = [SMACS_BOX]
        files = list_cals([SMACS_DIR])
    else:
        plan = json.load(open(a.plan))
        boxes = [dict(id=t["id"], ra=t["ra"], dec=t["dec"], size_arcsec=t["size_arcsec"]) for t in plan["tiles"]]
        files = list_cals(LMC_DIRS)
    print(f"{field}: {len(files)} _cal files on disk", flush=True)
    exps, skipped = build_exposures(field, files, boxes, plan, a.filters, a.workers)
    print(f"{field}: {len(exps)} NIRCam exposures in {a.filters}; skipped (other filters/instruments): {skipped}", flush=True)
    by_root = {r["rootname"]: r for r in exps}
    all_roots = [r["rootname"] for r in exps]
    rng = random.Random(f"{a.seed}/{field}")
    bids = [str(b["id"]) for b in boxes]
    in_box = lambda r: any(k in r["boxes"] for k in bids)          # overlaps >= 1 box (A2)
    gimgs = defaultdict(list)
    for r in exps:
        gimgs[r["dither_group"]].append(r)
    gcov = {(g, b): min(1.0, sum(r["boxes"].get(b, 0.0) for r in rs)) for g, rs in gimgs.items() for b in bids}
    vcov = defaultdict(float)
    for (g, b), c in gcov.items():
        r0 = gimgs[g][0]
        k = (f'{r0["visit_key"]}|{r0["filter"]}', b)
        vcov[k] = max(vcov[k], c)
    gprog = {g: rs[0]["program"] for g, rs in gimgs.items()}
    splits, notes = {}, []

    def finish(name, rule, train, held, extra=None):
        d = dict(rule=rule, n_train=len(train), n_heldout=len(held), train=sorted(train), heldout=sorted(held))
        d.update(extra or {})
        splits[name] = d

    # ---------------- dither
    groups = defaultdict(list)
    for r in exps:
        groups[r["dither_group"]].append(r["rootname"])
    elig = {(b, f): sorted(g for g in gimgs if g.endswith("|" + f) and gcov[(g, b)] >= COVER_MIN) for b in bids for f in a.filters}
    elig = {k: v for k, v in elig.items() if v}
    if field == "SMACS" and os.path.exists(a.legacy):
        leg = json.load(open(a.legacy))["holdout"]
        leg_roots = [x[:-len("_cal.fits")] for x in leg]
        miss = [x for x in leg_roots if x not in by_root]
        if miss:
            raise SystemExit(f"legacy held-out files not found in the SMACS table: {miss}")
        held_groups = sorted({by_root[x]["dither_group"] for x in leg_roots})
        per_filter = defaultdict(set)
        for g in held_groups:
            per_filter[g.split("|")[1]].add(g)
        extra_overlap = sorted(r["rootname"] for g in held_groups for r in exps if r["dither_group"] == g and in_box(r) and r["rootname"] not in leg_roots)
        finish("dither", "SMACS: legacy 2 dither positions per band (holdout_q5_cal.json, 24 detector images) kept verbatim so old runs stay comparable",
               [x for x in all_roots if x not in leg_roots], leg_roots,
               dict(source=a.legacy, dither_positions_per_filter={f: sorted(g.split("|")[0] for g in gs) for f, gs in sorted(per_filter.items())},
                    images_of_heldout_positions_overlapping_box_but_not_in_legacy_list=extra_overlap, seed=None,
                    n_candidate_positions_per_filter={f: len(elig.get((bids[0], f), [])) for f in a.filters}))
        if extra_overlap:
            notes.append(f"SMACS dither: {len(extra_overlap)} images of held-out dither positions overlap the box but are not in the legacy list (they stay in training)")
    else:
        H, rep = choose_units(rng, bids, a.filters, elig, a.n_dither, a.min_train_dithers)
        held = [x for g in H for x in groups[g] if in_box(by_root[x])]
        train = [x for x in all_roots if by_root[x]["dither_group"] not in H]
        finish("dither", f"{a.n_dither} whole dither positions per (filter, box); seeded random; min {a.min_train_dithers} kept for training",
               train, held, dict(seed=a.seed, heldout_dither_positions=sorted(H), by_box_filter=rep))

    # ---------------- visit
    vis = defaultdict(list)
    for r in exps:
        vis[f'{r["visit_key"]}|{r["filter"]}'].append(r["rootname"])
    velig = {(b, f): sorted(v for (v, bb), c in vcov.items() if bb == b and v.endswith("|" + f) and c >= COVER_MIN) for b in bids for f in a.filters}
    velig = {k: v for k, v in velig.items() if v}
    unavailable = sorted(f"{f}/box{b}" for (b, f), v in velig.items() if len(v) < 2)
    H, rep = choose_units(rng, bids, a.filters, velig, 1, 1)
    held = [x for g in H for x in vis[g] if in_box(by_root[x])]
    train = [x for x in all_roots if f'{by_root[x]["visit_key"]}|{by_root[x]["filter"]}' not in H]
    finish("visit", "one whole visit (all its images in that filter) per (filter, box) where the box has >= 2 covering visits; seeded random",
           train, held, dict(seed=a.seed, heldout_visits=sorted(H), unavailable_single_visit=unavailable, by_box_filter=rep))
    if unavailable:
        notes.append(f"visit split impossible (only one covering visit) for {len(unavailable)} (filter, box) pairs: {unavailable}")

    # ---------------- epoch (LMC)
    if field == "LMC":
        progs = sorted({r["program"] for r in exps})
        if progs == ["01476", "04447"] or (len(progs) == 2 and "01476" in progs and "04447" in progs):
            for name, hold, tr in (("epoch", "04447", "01476"), ("epoch_reverse", "01476", "04447")):
                train = [x for x in all_roots if by_root[x]["program"] == tr]
                held = [x for x in all_roots if by_root[x]["program"] == hold and in_box(by_root[x])]
                both = {f"{f}/box{b}" for b in bids for f in a.filters
                        if all(any(gprog[g] == p and g.endswith("|" + f) and c >= COVER_MIN for (g, bb), c in gcov.items() if bb == b)
                               for p in ("01476", "04447"))}
                finish(name, f"train on programme {int(tr)}, hold out every image of programme {int(hold)} that overlaps a box",
                       train, held, dict(train_programme=tr, heldout_programme=hold, filter_box_covered_by_both_programmes=sorted(both),
                                         filter_box_not_in_both=sorted(f"{f}/box{b}" for b in bids for f in a.filters if f"{f}/box{b}" not in both)))
        else:
            notes.append(f"epoch split skipped: programmes found {progs}")

    # ---------------- filter
    for f in a.heldout_filters:
        if f not in a.filters:
            notes.append(f"filter split {f} skipped: not in --filters")
            continue
        finish(f"filter_{f}", f"filter {f} held out entirely (placeholder until more filters arrive); images that overlap a box are scored",
               [x for x in all_roots if by_root[x]["filter"] != f], [x for x in all_roots if by_root[x]["filter"] == f and in_box(by_root[x])])

    out = dict(field=field, protocol="v1", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               seed=a.seed, filters=a.filters, min_train_dithers=a.min_train_dithers, cover_min=COVER_MIN, boxes=boxes,
               n_exposures=len(exps), skipped_other_filters_or_instruments=skipped, notes=notes,
               approximations=[l.strip() for l in __doc__.split("APPROXIMATIONS / CONVENTIONS")[1].splitlines() if l.strip()],
               exposures=exps,
               splits=splits)
    p = os.path.join(a.out, f"splits_{field}.json")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    for k, v in splits.items():
        print(f"  {field}/{k}: train {v['n_train']}  heldout {v['n_heldout']}", flush=True)
    print(f"wrote {p}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fields", default="SMACS,LMC")
    ap.add_argument("--out", default=f"{ARC}/protocol/v1")
    ap.add_argument("--plan", default=f"{ARC}/p1/plan_lmc.json")
    ap.add_argument("--legacy", default=f"{ARC}/scripts/holdout_q5_cal.json")
    ap.add_argument("--filters", default=",".join(SIX_FILTERS))
    ap.add_argument("--heldout-filters", default="F200W", help="comma list; one split filter_<F> each (placeholder default F200W)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n-dither", type=int, default=2)
    ap.add_argument("--min-train-dithers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.filters = [x for x in a.filters.split(",") if x]
    a.heldout_filters = [x for x in a.heldout_filters.split(",") if x]
    for f in a.fields.split(","):
        make_field(f, a)


if __name__ == "__main__":
    main()
