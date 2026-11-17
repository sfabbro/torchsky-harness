"""Tables for the sky-zero change (AUDIT_2026-11-17.md section F, TRUTH_REPORT.md section 6): reads the per-realisation result files of
run_truth.py and prints, per configuration (sky form, number of training exposures):

* empty-region total offset (fit sky + fitted mean background - true background - true sky), in ideal-noise units per resolution element,
  with its standard error from the realisation means and from the positions;
* the same paired with the ideal estimator on the same noise (fit - ideal: removes most of the noise common to both);
* the raw sky offset (sky as returned, no background correction) and the fitted mean background minus the truth [surface brightness];
* rms of the total error in ideal-noise units, and the ratio of the 32- to the 8-exposure rms in flux units (sqrt(N) scaling = 0.5);
* matched-aperture flux ratio and bias in source noise units, S/N 3 to 10 and 10 to 100 (bins by the S/N of each N);
* held-out score / noise-only score.

    python tests/truth/sky_zero_table.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truth_problem as tp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [  # label, file pattern (with {f} = g / i and {n} = 8 / 32)
    ("median (today)", "results/{f}_n{n}_s?.json"),
    ("masked", "results/sky_zero/mk_{f}_n{n}_s?.json"),
    ("masked + margin 3", "results/sky_zero/mm_{f}_n{n}_s?.json"),
]


def stats(runs):
    ms = [r["measure"] for r in runs]
    sig = np.array(ms[0]["emp"]["sigma_ideal"])
    ref = np.array(ms[0]["emp"]["ref_sky"])
    iref = np.array(ms[0]["emp"]["ideal_ref"])
    Z = np.array([(np.array(m["emp"]["rec"]) - ref) / sig for m in ms])
    Zr = np.array([(np.array(m["emp"]["rec_raw"]) - ref) / sig for m in ms])
    I = np.array([(np.array(m["emp"]["ideal"]) - iref) / sig for m in ms])
    D = Z - I
    R = len(runs)
    sem_r = lambda A: float(A.mean(1).std(ddof=1) / np.sqrt(R)) if R > 1 else float("nan")  # noqa: E731
    sem_p = lambda A: float(A.mean(0).std(ddof=1) / np.sqrt(A.shape[1]))  # noqa: E731
    out = {"R": R, "mean": Z.mean(), "sem_r": sem_r(Z), "sem_p": sem_p(Z), "paired": D.mean(), "paired_sem_r": sem_r(D), "paired_sem_p": sem_p(D),
           "raw": Zr.mean(), "bkg_off": float(np.mean([m["bkg_offset"] for m in ms])), "rms": float(np.sqrt((Z ** 2).mean())),
           "rms_ideal": float(np.sqrt((I ** 2).mean())), "sigma_flux": float(sig.mean()), "sd": float(Z.std()),
           "held": [round(r["ratio_to_floor"], 4) for r in runs]}
    src = tp.summarise_sources(ms)
    for lo, key in ((3, "s3"), (10, "s10")):
        rows = [r for r in src if r["bin"][0] >= lo and (lo == 10 or r["bin"][0] < 10)]
        n = sum(r["n_sources"] for r in rows)
        out[key + "_ratio"] = sum(r["rec_matched"]["mean_ratio"] * r["n_sources"] for r in rows) / n
        out[key + "_bias"] = sum(r["rec_matched"]["mean_bias_noise_units"] * r["n_sources"] for r in rows) / n
        out[key + "_ideal_bias"] = sum(r["ideal"]["mean_bias_noise_units"] * r["n_sources"] for r in rows) / n
        out[key + "_sem"] = float(np.sqrt(sum((r["rec_matched"]["sem_pooled"] * r["n_sources"]) ** 2 for r in rows)) / n)
    return out


def main():
    res = {}
    for label, pat in CONFIGS:
        for f in ("g", "i"):
            for n in (8, 32):
                files = sorted(glob.glob(os.path.join(HERE, pat.format(f=f, n=n))))
                if not files:
                    continue
                res[(label, f, n)] = stats([json.load(open(p)) for p in files])
    print("| convention | sky | N | R | total offset (real.) (pos.) | fit - ideal (real.) (pos.) | raw offset | bkg - truth [s.b.] | rms | rms ideal | "
          "rms 32/8 flux units | matched ratio S/N 3-10 | bias 3-10 | matched ratio S/N >= 10 | bias >= 10 | held-out / floor |")
    for (label, f, n), s in res.items():
        k8 = res.get((label, f, 8))
        sc = ""
        if n == 32 and k8:
            sc = "%.2f" % (s["rms"] * s["sigma_flux"] / (k8["rms"] * k8["sigma_flux"]))
        print(f"| {label} | {'grid' if f == 'g' else 'items'} | {n} | {s['R']} | {s['mean']:+.2f} ({s['sem_r']:.2f}) ({s['sem_p']:.2f}) | "
              f"{s['paired']:+.2f} ({s['paired_sem_r']:.2f}) ({s['paired_sem_p']:.2f}) | {s['raw']:+.2f} | {s['bkg_off']:+.4f} | {s['rms']:.2f} | "
              f"{s['rms_ideal']:.2f} | {sc} | {s['s3_ratio']:.2f} +- {s['s3_sem']:.2f} | {s['s3_bias']:+.2f} (ideal {s['s3_ideal_bias']:+.2f}) | "
              f"{s['s10_ratio']:.3f} | {s['s10_bias']:+.2f} (ideal {s['s10_ideal_bias']:+.2f}) | {s['held']} |")
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(os.path.join(HERE, "results/sky_zero/table.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
