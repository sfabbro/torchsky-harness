"""Table of protocol-v1 scores of several methods, from their score files (scores/<FIELD>/<METHOD>/<SPLIT>.json or a run's score_<split>.json).

    python collect_scores.py label=path/to/score.json [label=path ...]

Prints markdown: chi2/N without the scorer's offset, per bin of the official mosaic's signal-to-noise, with the 68% interval of the bootstrap
over dither groups, then the headline (signal-to-noise 5 and above) per filter. A cell is "-" when the file has no pixels there."""
import json
import sys


def cell(b):
    key = "chi2_per_pixel" if b and "chi2_per_pixel" in b else "value"
    if not b or b.get(key) is None:
        return "-"
    lo, hi = b.get("lo"), b.get("hi")
    return f"{b[key]:.3f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None and hi is not None else "")


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    data = {lab: json.load(open(path)) for lab, path in runs}
    first = next(iter(data.values()))
    bins = first["bins"]
    print("| Method | exposures | headline | " + " | ".join(bins) + " | fractional residual rms above 50 |")
    print("|---|---|---|" + "---|" * (len(bins) + 1))
    for lab, d in data.items():
        wo = d["levels"]["field"]["all"]["without_offset"]
        by = {b["snr_ref"]: b for b in wo["bins"]}
        head = wo.get("headline_chi2_snr_ref_ge5")
        print(f"| {lab} | {d['n_scored_exposures']} of {d['n_heldout_listed']} | {cell(head)} | " + " | ".join(cell(by.get(b)) for b in bins) + f" | {cell(wo.get('frac_resid_rms_snr_ref_gt50'))} |")
    filters = sorted({f for d in data.values() for f in d["levels"]["filter"]})
    print("\n| Method | " + " | ".join(filters) + " |")
    print("|---|" + "---|" * len(filters))
    for lab, d in data.items():
        row = []
        for f in filters:
            wo = d["levels"]["filter"].get(f, {}).get("without_offset", {})
            row.append(cell(wo.get("headline_chi2_snr_ref_ge5")))
        print(f"| {lab} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
