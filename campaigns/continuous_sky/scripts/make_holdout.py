#!/usr/bin/env python
"""Held-out split generator for smacs_four_v38.py --holdout-from (tests T1-T4 of psf-geometry-audit.md).

The output is a JSON file {"holdout": [names], "note": ..., "test": "T2", "params": {...}, "summary": {...}, "warnings": [...]}. v38 reads only
the "holdout" list (read_holdout_names); the other keys document the split. Names are what select_exposures matches: an HST ROOTNAME (holds out all
its chips, like holdout_hst.txt) and a JWST file name (one detector image; every detector of a dither is listed).

Inputs (any combination; records are merged, duplicates by name dropped):
  --manifest CSV      HST manifest (hst/hst_imaging_unique.csv: proposal_id, instrument, detector, filter, root/obs_id, product_filename, product_type,
                      t_start_mjd). Only FLT/FLC rows of WFC3/IR, ACS/WFC, WFC3/UVIS are used (what the v38 loader supports); one record per rootname.
  --jwst-dirs D1,D2   directories with *_cal.fits (--jwst-glob): instrument, filter, MJD, roll from the headers (needs astropy; --no-headers = names only:
                      instrument from the detector in the name, band group = instrument + detector, no MJD, no roll).
  --names FILE        one exposure name per line (JWST file names and/or HST rootnames); instrument inferred from the name as far as possible.
  --hst-dir DIR       HST FITS files (<root>_flt.fits / _flc.fits): roll = PA_V3 of the primary header (else ORIENTAT of the first SCI extension).
  --roll-csv CSV      name,roll_deg overrides / supplies rolls (HST rootname or JWST file name).
  --instruments LIST  only these instruments are considered (others never held out), e.g. "WFC3/IR,ACS/WFC,NIRISS".

Splits (approximations are stated per mode; band = instrument/detector + filter):
  t1  dither level (the v22 rule of _auto_holdout with n_test > 1): per band, --n-dithers dithers (HST: one rootname = all chips; JWST: one visit +
      exposure number = all detectors) chosen round-robin over the visits; at least 2 dithers always train. APPROXIMATION: v22 picks the dither with
      the most loaded pixels, which needs the data; here the first dither in name order is taken (deterministic).
  t2  whole visits with distinct roll angles: per instrument the roll angles (PA_V3, circular) are clustered (single linkage, --roll-tol deg, default 2);
      the --t2-n-rolls (default 1) most isolated clusters are held out (all their visits, or the --t2-visits-per-roll largest); isolation = distance to
      the nearest TRAINING cluster, reported, warned below --min-separation (default 5 deg). A band whose exposures all lie at the held-out roll is
      returned to training (it could not be tested). Needs rolls: --hst-dir, --jwst-dirs or --roll-csv.
  t3  held-out epoch: every exposure of --test-programs is held out, --train-programs train (other programs stay in training; restrict the run with
      --hst-programs). Bands (instrument/detector + filter) with no training exposure are NOT held out (their operator would be untrained): listed
      under "dropped_uncovered". --both-directions also writes <out>_reverse.json with the roles swapped.
  t4  new-instrument transfer: for --new-instrument X, per band of X the exposures are split about half/half by visit (balanced by exposure count,
      largest first) or, if the band has one visit, by dither (alternating); the test half is held out, all other instruments train (nothing of theirs
      is listed). A band that cannot be split (one dither) trains entirely and is listed under "not_testable". Use with v38 --freeze-sky-from.

Examples:
  python scripts/make_holdout.py t3 --manifest hst/hst_imaging_unique.csv --test-programs 16729 --train-programs 14096 --both-directions --out hold_t3.json
  python scripts/make_holdout.py t4 --manifest hst/hst_imaging_unique.csv --new-instrument WFC3/IR --out hold_t4_wfc3ir.json
  python scripts/make_holdout.py t4 --jwst-dirs /arc/.../2736_niriss,/arc/.../4368_niriss --new-instrument NIRISS --out hold_t4_niriss.json
  python scripts/make_holdout.py t2 --manifest hst/hst_imaging_unique.csv --hst-dir /arc/projects/ots/torchsky-data/smacs0723_union/hst --out hold_t2.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

JWST_RE = re.compile(r"^(?:(?P<idx>\d+)_)?jw(?P<prog>\d{5})(?P<obs>\d{3})(?P<vis>\d{3})_(?P<grp>[0-9a-z]{5})_(?P<exp>\d{5})_(?P<det>[a-z0-9]+)(?:_|\.|$)", re.I)
HST_RE = re.compile(r"^(?P<root>[a-z][a-z0-9]{8})(?:_(?:flt|flc|sci\d+))?(?:\.fits)?$", re.I)
SUPPORTED_HST = {"WFC3/IR", "ACS/WFC", "WFC3/UVIS"}


# ----------------------------------------------------------------------------------------------------------------- records
def jwst_instrument_from_det(det):
    d = det.lower()
    return "NIRCam" if d.startswith("nrc") else ("NIRISS" if d.startswith("nis") else ("MIRI" if d.startswith("miri") else "JWST"))


def hst_visit(root):
    """visit id of ipppssoot: instrument letter + ppp (programme) + ss (visit): root[:6]."""
    return root[:6].lower()


def make_record(name, kind, **kw):
    r = {"name": name, "kind": kind, "program": None, "instrument": "?", "detector": "", "filter": "", "mjd": None, "visit": None, "dither": None,
         "roll": None, "files": [name]}
    r.update(kw)
    r["band"] = f"{r['instrument']}:{r['filter'] or r['detector']}"
    return r


def hst_records_from_manifest(path):
    recs, skipped = {}, {"unsupported_instrument_or_product": 0}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            inst = f"{row['instrument']}/{row['detector']}".upper()
            if inst not in SUPPORTED_HST or row.get("product_type", "FLT").upper() not in ("FLT", "FLC"):
                skipped["unsupported_instrument_or_product"] += 1
                continue
            root = (row.get("obs_id") or row["product_filename"].split("_")[0] or row.get("root")).strip().lower()  # 9-char ipppssoot (the manifest "root" column is 8 chars and matched nothing: fixed 7 Nov)
            if root in recs:
                continue
            recs[root] = make_record(root, "hst", program=int(row["proposal_id"]), instrument=inst, detector=row["detector"].upper(),
                                     filter=row["filter"].strip().upper(), mjd=float(row["t_start_mjd"]), visit=hst_visit(root), dither=root)
    return list(recs.values()), skipped


def jwst_records_from_dirs(dirs, glob_pat, headers=True):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, glob_pat))):
            nm = os.path.basename(f)
            m = JWST_RE.match(nm)
            if not m:
                continue
            g = m.groupdict()
            r = make_record(nm, "jwst", program=int(g["prog"]), instrument=jwst_instrument_from_det(g["det"]), detector=g["det"].upper(),
                            visit=(g["obs"] + g["vis"], g["grp"]), dither=((g["obs"] + g["vis"], g["grp"]), g["exp"]))
            if headers:
                try:
                    from astropy.io import fits
                    with fits.open(f, memmap=False) as h:
                        h0 = h[0].header
                        sh = h["SCI"].header
                    ins = {"NIRCAM": "NIRCam", "NIRISS": "NIRISS", "MIRI": "MIRI"}.get(str(h0.get("INSTRUME", "")).upper(), r["instrument"])
                    flt, pup = str(h0.get("FILTER", "")).upper(), str(h0.get("PUPIL", "")).upper()
                    clear = ("", "CLEAR", "CLEARP", "OPEN", "N/A", "NONE")
                    filt = pup if (flt in clear and pup not in clear) else flt
                    roll = sh.get("PA_APER", h0.get("PA_V3"))
                    r.update(instrument=ins, filter=filt, detector=str(h0.get("DETECTOR", r["detector"])).upper(),
                             mjd=float(sh.get("MJD-AVG", h0.get("EXPSTART", "nan"))), roll=None if roll is None else float(roll))
                    r["band"] = f"{ins}:{filt or r['detector']}"
                except Exception as exc:  # noqa: BLE001
                    print(f"warning: {nm}: header not read ({exc}); name only", file=sys.stderr)
            recs.append(r)
    return recs


def records_from_names(path):
    recs = []
    for ln in open(path):
        nm = ln.strip()
        if not nm or nm.startswith("#"):
            continue
        m = JWST_RE.match(nm)
        if m:
            g = m.groupdict()
            recs.append(make_record(os.path.basename(nm), "jwst", program=int(g["prog"]), instrument=jwst_instrument_from_det(g["det"]), detector=g["det"].upper(),
                                    visit=(g["obs"] + g["vis"], g["grp"]), dither=((g["obs"] + g["vis"], g["grp"]), g["exp"])))
            continue
        m = HST_RE.match(nm)
        if m:
            root = m.group("root").lower()
            ins = {"i": "WFC3", "j": "ACS", "o": "STIS"}.get(root[0], "HST")
            recs.append(make_record(root, "hst", instrument=ins, program=None, visit=hst_visit(root), dither=root))
    return recs


def read_hst_roll(hst_dir, root):
    from astropy.io import fits
    for suf in ("_flc.fits", "_flt.fits"):
        p = os.path.join(hst_dir, root + suf)
        if os.path.exists(p):
            with fits.open(p, memmap=False) as h:
                v = h[0].header.get("PA_V3")
                if v is None and len(h) > 1:
                    v = h[1].header.get("ORIENTAT")
            return None if v is None else float(v)
    return None


def attach_rolls(recs, hst_dir=None, roll_csv=None):
    if hst_dir:
        for r in recs:
            if r["kind"] == "hst" and r["roll"] is None:
                r["roll"] = read_hst_roll(hst_dir, r["name"])
    if roll_csv:
        tab = {}
        for row in csv.reader(open(roll_csv)):
            if len(row) >= 2 and row[0] and not row[0].startswith("#") and row[0] != "name":
                tab[row[0].strip()] = float(row[1])
        for r in recs:
            if r["name"] in tab:
                r["roll"] = tab[r["name"]]


def select(recs, instruments):
    if not instruments:
        return recs
    want = {x.strip().upper() for x in instruments.split(",") if x.strip()}
    return [r for r in recs if r["instrument"].upper() in want]


def groups(recs, key):
    out = {}
    for r in recs:
        out.setdefault(key(r), []).append(r)
    return out


def band_key(r):
    return r["band"]


def finish(mode, test_recs, recs, params, warnings, extra=None, note=""):
    names = sorted({f for r in test_recs for f in r["files"]})
    tst = {r["name"] for r in test_recs}
    per_band = {}
    for r in recs:
        b = per_band.setdefault(r["band"], {"n_test": 0, "n_train": 0})
        b["n_test" if r["name"] in tst else "n_train"] += 1
    out = {"holdout": names, "test": mode, "note": note, "params": params, "summary": {"n_holdout_names": len(names), "n_records": len(recs), "per_band": per_band},
           "warnings": warnings, "generator": "scripts/make_holdout.py"}
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------------------------------------------- T1
def split_t1(recs, n_dithers=2):
    test, warn = [], []
    for band, rs in sorted(groups(recs, band_key).items()):
        dith = groups(rs, lambda r: r["dither"])
        n_eff = min(n_dithers, len(dith) - 2)
        if n_eff < n_dithers:
            warn.append(f"{band}: {len(dith)} dithers, holding out {max(n_eff, 0)} instead of {n_dithers}")
        if n_eff < 1:
            continue
        byvis = {}
        for k in sorted(dith, key=str):
            byvis.setdefault(str(dith[k][0]["visit"]), []).append(k)
        order = sorted(byvis, key=lambda v: (-len(byvis[v]), v))
        chosen = []
        while len(chosen) < n_eff:
            for v in order:
                if byvis[v] and len(chosen) < n_eff:
                    chosen.append(byvis[v].pop(0))
        for k in chosen:
            test += dith[k]
    return test, warn


# ----------------------------------------------------------------------------------------------------------------- T2
def circ_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def roll_clusters(rolls, tol):
    """single-linkage clusters of circular angles: list of lists of indices into `rolls`."""
    order = sorted(range(len(rolls)), key=lambda i: rolls[i] % 360.0)
    clusters = []
    for i in order:
        if clusters and circ_dist(rolls[clusters[-1][-1]], rolls[i]) <= tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    if len(clusters) > 1 and circ_dist(rolls[clusters[0][0]], rolls[clusters[-1][-1]]) <= tol:  # wrap-around
        clusters[0] = clusters.pop() + clusters[0]
    return clusters


def split_t2(recs, tol=2.0, n_rolls=1, visits_per_roll=0, min_sep=5.0):
    test, warn, info = [], [], {}
    nor = [r for r in recs if r["roll"] is None]
    if nor:
        raise SystemExit(f"T2 needs a roll for every record; {len(nor)} have none (first: {nor[0]['name']}): give --hst-dir (PA_V3), --jwst-dirs (headers) or --roll-csv")
    for inst, rs in sorted(groups(recs, lambda r: r["instrument"]).items()):
        rolls = [r["roll"] for r in rs]
        cl = roll_clusters(rolls, tol)
        mean = []
        for c in cl:
            ang = [math.radians(rolls[i]) for i in c]
            mean.append(math.degrees(math.atan2(sum(math.sin(x) for x in ang), sum(math.cos(x) for x in ang))) % 360.0)
        if len(cl) < 2:
            warn.append(f"{inst}: only one roll cluster ({mean[0]:.1f} deg): no T2 possible")
            continue
        iso = [min(circ_dist(mean[i], mean[j]) for j in range(len(cl)) if j != i) for i in range(len(cl))]
        # most isolated first; ties: more exposures, then roll
        pick = sorted(range(len(cl)), key=lambda i: (-iso[i], -len(cl[i]), mean[i]))[:max(1, min(n_rolls, len(cl) - 1))]
        train_clusters = [i for i in range(len(cl)) if i not in pick]
        info[inst] = {"clusters_deg": [round(m, 2) for m in mean], "n_exposures": [len(c) for c in cl], "held_out_clusters": [round(mean[i], 2) for i in pick], "separation_deg": {}}
        for i in pick:
            sep = min(circ_dist(mean[i], mean[j]) for j in train_clusters)
            info[inst]["separation_deg"][f"{mean[i]:.2f}"] = round(sep, 2)
            if sep < min_sep:
                warn.append(f"{inst}: held-out roll {mean[i]:.1f} deg is only {sep:.1f} deg from a training roll (< {min_sep}): not a distinct roll")
            sel = [rs[k] for k in cl[i]]
            if visits_per_roll > 0:
                byv = groups(sel, lambda r: str(r["visit"]))
                keep = sorted(byv, key=lambda v: (-len(byv[v]), v))[:visits_per_roll]
                sel = [r for v in keep for r in byv[v]]
            test += sel
    # a band entirely at the held-out roll cannot be tested: return it to training
    tst = {r["name"] for r in test}
    dropped = []
    for band, rs in groups(recs, band_key).items():
        if rs and all(r["name"] in tst for r in rs):
            dropped.append(band)
            test = [r for r in test if r["band"] != band]
            warn.append(f"{band}: every exposure is at the held-out roll, returned to training (not testable)")
    return test, warn, {"roll_clusters": info, "not_testable_bands": dropped}


# ----------------------------------------------------------------------------------------------------------------- T3
def split_t3(recs, test_programs, train_programs):
    test_p, train_p = set(test_programs), set(train_programs)
    if test_p & train_p:
        raise SystemExit("--test-programs and --train-programs overlap")
    if any(r["program"] is None for r in recs):
        raise SystemExit("T3 needs the programme of every record (manifest, or JWST names); --names with HST rootnames carries none")
    test = [r for r in recs if r["program"] in test_p]
    train = [r for r in recs if r["program"] in train_p]
    covered = {r["band"] for r in train}
    dropped = sorted({r["band"] for r in test if r["band"] not in covered})
    warn = [f"{b}: no training exposure of this band in programmes {sorted(train_p)}: its exposures stay in training, not held out" for b in dropped]
    test = [r for r in test if r["band"] in covered]
    if not test:
        warn.append("no held-out exposure left: check the programme numbers")
    return test, warn, {"dropped_uncovered": dropped, "train_programs": sorted(train_p), "test_programs": sorted(test_p)}


# ----------------------------------------------------------------------------------------------------------------- T4
def split_t4(recs, new_instrument):
    want = new_instrument.strip().upper()
    xs = [r for r in recs if r["instrument"].upper() == want]
    if not xs:
        raise SystemExit(f"no records of instrument {new_instrument!r}; present: {sorted({r['instrument'] for r in recs})}")
    test, warn, not_testable, how = [], [], [], {}
    for band, rs in sorted(groups(xs, band_key).items()):
        dith = groups(rs, lambda r: r["dither"])
        visits = groups(rs, lambda r: str(r["visit"]))
        if len(visits) >= 2:
            units = sorted(visits.items(), key=lambda kv: (-len({r["dither"] for r in kv[1]}), kv[0]))
            n_te = n_tr = 0
            te = []
            for v, vr in units:
                n = len({r["dither"] for r in vr})
                if n_te < n_tr:
                    te += vr
                    n_te += n
                else:
                    n_tr += n
            if not te or n_tr == 0:  # e.g. two visits where the first took the train side and the second also: force the split
                te = units[-1][1]
            test += te
            how[band] = f"by visit ({len(visits)} visits)"
        elif len(dith) >= 2:
            keys = sorted(dith, key=str)
            for k in keys[1::2]:
                test += dith[k]
            how[band] = f"by dither within one visit ({len(dith)} dithers)"
        else:
            not_testable.append(band)
            warn.append(f"{band}: a single dither, cannot be split: trains entirely")
    return test, warn, {"new_instrument": new_instrument, "split": how, "not_testable": not_testable,
                        "new_instrument_train_names": sorted({f for r in xs if r["name"] not in {t['name'] for t in test} for f in r["files"]}),
                        "intended_use": "run A (sky) without this instrument's data; run B with --freeze-sky-from run A, --holdout-from this file"}


# ----------------------------------------------------------------------------------------------------------------- CLI
def build_records(args):
    recs, info = [], {}
    if args.manifest:
        r, sk = hst_records_from_manifest(args.manifest)
        recs += r
        info["manifest_skipped"] = sk
    if args.jwst_dirs:
        recs += jwst_records_from_dirs([d for d in args.jwst_dirs.split(",") if d], args.jwst_glob, headers=not args.no_headers)
    if args.names:
        recs += records_from_names(args.names)
    seen, uniq = set(), []
    for r in recs:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)
    recs = select(uniq, args.instruments)
    attach_rolls(recs, args.hst_dir, args.roll_csv)
    return recs, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("t1", "t2", "t3", "t4"))
    ap.add_argument("--manifest")
    ap.add_argument("--jwst-dirs")
    ap.add_argument("--jwst-glob", default="*_cal.fits")
    ap.add_argument("--no-headers", action="store_true", help="JWST: use the file names only")
    ap.add_argument("--names")
    ap.add_argument("--hst-dir")
    ap.add_argument("--roll-csv")
    ap.add_argument("--instruments")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dithers", type=int, default=2, help="t1: dithers held out per band")
    ap.add_argument("--roll-tol", type=float, default=2.0, help="t2: roll clustering tolerance [deg]")
    ap.add_argument("--t2-n-rolls", type=int, default=1, help="t2: roll clusters held out per instrument")
    ap.add_argument("--t2-visits-per-roll", type=int, default=0, help="t2: at most this many visits of a held-out roll (0 = all)")
    ap.add_argument("--min-separation", type=float, default=5.0, help="t2: warn when a held-out roll is closer than this to a training roll [deg]")
    ap.add_argument("--test-programs", help="t3: comma list of programme ids held out")
    ap.add_argument("--train-programs", help="t3: comma list of programme ids that train")
    ap.add_argument("--both-directions", action="store_true", help="t3: also write <out>_reverse.json with test and train swapped")
    ap.add_argument("--new-instrument", help="t4: instrument whose exposures are split half/half (WFC3/IR, ACS/WFC, WFC3/UVIS, NIRISS, NIRCam, MIRI)")
    args = ap.parse_args(argv)
    recs, info = build_records(args)
    if not recs:
        raise SystemExit("no exposure records: give --manifest, --jwst-dirs or --names")
    params = {k: v for k, v in vars(args).items() if v not in (None, False)}
    outs = []
    if args.mode == "t1":
        test, warn = split_t1(recs, args.n_dithers)
        outs.append((args.out, finish("T1", test, recs, params, warn, note="dither-level held-out exposures (v22 rule, name order instead of pixel count)")))
    elif args.mode == "t2":
        test, warn, extra = split_t2(recs, args.roll_tol, args.t2_n_rolls, args.t2_visits_per_roll, args.min_separation)
        outs.append((args.out, finish("T2", test, recs, params, warn, extra, "whole visits at roll angles absent from training")))
    elif args.mode == "t3":
        if not (args.test_programs and args.train_programs):
            raise SystemExit("t3 needs --test-programs and --train-programs")
        tp = [int(x) for x in args.test_programs.split(",")]
        rp = [int(x) for x in args.train_programs.split(",")]
        test, warn, extra = split_t3(recs, tp, rp)
        outs.append((args.out, finish("T3", test, recs, params, warn, extra, f"epoch held out: programmes {tp} test, {rp} train")))
        if args.both_directions:
            test2, warn2, extra2 = split_t3(recs, rp, tp)
            stem, ext = os.path.splitext(args.out)
            outs.append((f"{stem}_reverse{ext}", finish("T3", test2, recs, params, warn2, extra2, f"epoch held out (reverse): programmes {rp} test, {tp} train")))
    else:
        if not args.new_instrument:
            raise SystemExit("t4 needs --new-instrument")
        test, warn, extra = split_t4(recs, args.new_instrument)
        outs.append((args.out, finish("T4", test, recs, params, warn, extra, f"new-instrument transfer: half of {args.new_instrument} held out, everything else trains")))
    for path, obj in outs:
        obj["inputs"] = info
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1)
        s = obj["summary"]
        print(f"{path}: {obj['test']} {s['n_holdout_names']} names held out of {s['n_records']} exposure records")
        for b, v in sorted(s["per_band"].items()):
            print(f"   {b:28s} test {v['n_test']:3d}  train {v['n_train']:3d}")
        for w in obj["warnings"]:
            print("   WARNING:", w)
    return outs


if __name__ == "__main__":
    main()
