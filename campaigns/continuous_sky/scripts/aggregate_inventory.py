"""Inventory (and optional download) of public calibration / survey programmes for learning instrument operators in aggregate
(claude/operator-aggregate-design-memo.md). Queries MAST for level-2 per-detector products (JWST *_cal, HST *_flt/_flc) of the listed
programmes, writes a CSV of every product and a JSON summary (files and GB per programme, instrument, filter), and with --fetch DIR
downloads the products matching --fetch-filters (and --fetch-programs) into DIR/<programme>_<instrument>/.

Approximations: programme list from the memo (IDs verified there against STScI pages, except 1501/1515 known only through 4471/4475);
MAST metadata is taken as is (no check of the pointing against a field position); the per-file size is MAST's 'size' column."""
import argparse
import json
import os
import sys
import time

PROGRAMS = {  # id: (mission, note)
    "1476": ("JWST", "NIRCam LMC calibration field, Cycle 1 flats/photometry/distortion"),
    "4447": ("JWST", "NIRCam LMC calibration field, Cycle 2"),
    "1501": ("JWST", "NIRISS LMC astrometric field, Cycle 1"),
    "4471": ("JWST", "NIRISS LMC astrometric field, Cycle 2"),
    "1515": ("JWST", "NIRISS F200W monitoring, Cycle 1"),
    "4475": ("JWST", "NIRISS F200W monitoring, Cycle 2"),
    "1334": ("JWST", "ERS M92 (NIRCam)"),
    "1837": ("JWST", "PRIMER (extragalactic control)"),
    "11928": ("HST", "omega Cen WFC3/IR"), "12340": ("HST", "omega Cen WFC3/IR"), "12353": ("HST", "omega Cen WFC3/IR"),
    "12694": ("HST", "omega Cen WFC3/IR"), "12714": ("HST", "omega Cen WFC3/IR"), "17974": ("HST", "omega Cen WFC3 monitor"),
    "11445": ("HST", "WFC3/IR on-orbit verification, 47 Tuc / LMC"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--programs", default=",".join(PROGRAMS))
    ap.add_argument("--fetch", default=None, help="download directory (omit = inventory only)")
    ap.add_argument("--fetch-programs", default="1476,4447")
    ap.add_argument("--fetch-filters", default="F090W,F150W,F200W,F277W,F356W,F444W")
    ap.add_argument("--fetch-max", type=int, default=100000)
    a = ap.parse_args()
    from astroquery.mast import Observations
    os.makedirs(a.out, exist_ok=True)
    rows, summary = [], {}
    for pid in a.programs.split(","):
        mission = PROGRAMS.get(pid, ("JWST", ""))[0]
        try:
            obs = Observations.query_criteria(proposal_id=pid, obs_collection=mission, dataproduct_type="image")
        except Exception as exc:  # noqa: BLE001
            log("query failed", pid, repr(exc)[:200]); continue
        if len(obs) == 0:
            log(pid, "no observations"); summary[pid] = {"note": "no observations"}; continue
        try:
            prods = Observations.get_product_list(obs)
        except Exception as exc:  # noqa: BLE001
            log("product list failed", pid, repr(exc)[:200]); continue
        if mission == "JWST":
            sel = prods[(prods["productSubGroupDescription"] == "CAL") & (prods["calib_level"] == 2)]
        else:
            sel = prods[[str(s) in ("FLT", "FLC") for s in prods["productSubGroupDescription"]]]
        obs_by_id = {str(o["obsid"]): o for o in obs}
        s = {}
        for p in sel:
            o = obs_by_id.get(str(p["parent_obsid"])) or obs_by_id.get(str(p["obsID"]))
            inst = str(o["instrument_name"]) if o is not None else "?"
            filt = str(o["filters"]) if o is not None else "?"
            t = float(o["t_min"]) if o is not None and o["t_min"] is not None else float("nan")
            size = int(p["size"]) if p["size"] is not None else 0
            rows.append({"program": pid, "mission": mission, "instrument": inst, "filters": filt, "mjd": t, "file": str(p["productFilename"]),
                         "uri": str(p["dataURI"]), "size": size})
            k = f"{inst}|{filt}"
            s.setdefault(k, [0, 0]); s[k][0] += 1; s[k][1] += size
        summary[pid] = {"note": PROGRAMS.get(pid, ("", ""))[1], "n_obs": len(obs),
                        "per_instrument_filter": {k: {"files": v[0], "GB": round(v[1] / 1e9, 2)} for k, v in sorted(s.items())}}
        log(pid, mission, len(obs), "obs;", sum(v[0] for v in s.values()), "files;", round(sum(v[1] for v in s.values()) / 1e9, 1), "GB")
    import csv
    with open(os.path.join(a.out, "aggregate_inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["program", "mission", "instrument", "filters", "mjd", "file", "uri", "size"])
        w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(a.out, "aggregate_inventory.json"), "w"), indent=1)
    if not a.fetch:
        return
    want_p, want_f = set(a.fetch_programs.split(",")), set(a.fetch_filters.split(","))
    def band_ok(filters):  # exact match: tokens other than CLEAR must be exactly one wanted filter (excludes F150W2, filter+pupil combos, masks, grisms)
        toks = [t.strip() for t in filters.split(";") if t.strip() and t.strip() != "CLEAR"]
        return len(toks) == 1 and toks[0] in want_f
    todo = [r for r in rows if r["program"] in want_p and band_ok(r["filters"]) and "NIRCAM/IMAGE" in r["instrument"] or
            (r["program"] in want_p and band_ok(r["filters"]) and "NIRISS/IMAGE" in r["instrument"])][: a.fetch_max]
    log("fetching", len(todo), "files,", round(sum(r["size"] for r in todo) / 1e9, 1), "GB")
    n = 0
    for r in todo:
        d = os.path.join(a.fetch, f"{r['program']}_{r['instrument'].split('/')[0].lower()}")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, r["file"])
        if os.path.exists(dst) and os.path.getsize(dst) == r["size"]:
            continue
        for attempt in range(3):
            try:
                Observations.download_file(r["uri"], local_path=dst)
                break
            except Exception as exc:  # noqa: BLE001
                log("retry", r["file"], repr(exc)[:120]); time.sleep(5)
        n += 1
        if n % 50 == 0:
            log("downloaded", n)
    log("done", n)


if __name__ == "__main__":
    sys.exit(main())
