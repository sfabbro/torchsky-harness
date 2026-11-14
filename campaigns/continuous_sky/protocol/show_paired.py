import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); print("== reference:", d["reference"])
    for m, M in d["methods"].items():
        x = M["levels"]["field"]["all"]
        for off in ("with_offset", "without_offset"):
            rows = [x[off]["headline"]] + x[off]["bins"]
            print(f"  {m} [{off}]")
            for r in rows:
                dv = r["diff"]
                print(f"    S/N {r['snr_ref']:>6}  n={r['n_pixels']:>8}  method {r['score_method']['value']:8.3f}  ref {r['score_reference']['value']:8.3f}  diff {dv['value']:+8.3f} [{dv['lo68']:+.3f}, {dv['hi68']:+.3f}]  P(better) {dv['frac_better']:.2f}")
