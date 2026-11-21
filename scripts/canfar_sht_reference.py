"""CPU SHT reference timings (ducc0, healpy) and a torchsky-vs-ducc0 check.

Run by the ``backend-matrix-cuda`` CANFAR mode when TORCHSKY_BM_SHT_REFERENCE=1.

* ``timings``: ducc0.sht.experimental adjoint_synthesis / synthesis and healpy
  map2alm / alm2map (iter=0, no weights) on the host CPU with ``--threads``
  threads at each nside, lmax = mmax = 2 * nside, spin 0, RING. Each call is
  repeated ``--reps`` times; min and median wall seconds are recorded (the first
  call is reported separately as ``first_s``).
* ``correctness``: torchsky.sphere.spectral map2alm / alm2map (iter=0) on
  ``--device`` in float64 / complex128 against ducc0 on CPU (healpy-normalised:
  map2alm = adjoint_synthesis * 4 pi / npix). Reports max |diff| / max |ref| and
  ||diff||_2 / ||ref||_2. Same inputs for both: a random alm (seeded, m = 0
  imaginary parts zero) and the band-limited map ducc0 synthesises from it.

Prints one JSON document per mode and writes it to ``--json-out``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["timings", "correctness"])
    p.add_argument("--nsides", default="64,256,512")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json-out", required=True)
    return p.parse_args()


ARGS = _parse()
# healpy (libsharp/ducc inside) reads OMP_NUM_THREADS at import.
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)

import ducc0  # noqa: E402
import numpy as np  # noqa: E402


def _geom(nside: int) -> dict:
    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return {k: info[k] for k in ("theta", "phi0", "nphi", "ringstart")}


def _random_alm(lmax: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    alm[: lmax + 1] = alm[: lmax + 1].real  # m = 0 block (healpy order)
    return alm.astype(np.complex128)


def _ducc_synthesis(
    alm: np.ndarray, nside: int, lmax: int, nthreads: int
) -> np.ndarray:
    out = ducc0.sht.experimental.synthesis(
        alm=alm[None, :],
        lmax=lmax,
        mmax=lmax,
        spin=0,
        nthreads=nthreads,
        **_geom(nside),
    )
    return out[0]


def _ducc_map2alm(m: np.ndarray, nside: int, lmax: int, nthreads: int) -> np.ndarray:
    out = ducc0.sht.experimental.adjoint_synthesis(
        map=m[None, :], lmax=lmax, mmax=lmax, spin=0, nthreads=nthreads, **_geom(nside)
    )
    return out[0] * (4.0 * np.pi / m.size)


def _time(fn, reps: int) -> dict:
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {
        "first_s": first,
        "min_s": min(ts),
        "median_s": statistics.median(ts),
        "reps": reps,
    }


def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def timings() -> dict:
    import healpy as hp

    rows = []
    for nside in (int(s) for s in ARGS.nsides.split(",")):
        lmax = 2 * nside
        alm = _random_alm(lmax, seed=nside)
        m = _ducc_synthesis(alm, nside, lmax, ARGS.threads)
        geom = _geom(nside)  # cached outside the timed calls
        syn = lambda: ducc0.sht.experimental.synthesis(  # noqa: E731
            alm=alm[None, :],
            lmax=lmax,
            mmax=lmax,
            spin=0,
            nthreads=ARGS.threads,
            **geom,
        )
        adj = lambda: ducc0.sht.experimental.adjoint_synthesis(  # noqa: E731
            map=m[None, :], lmax=lmax, mmax=lmax, spin=0, nthreads=ARGS.threads, **geom
        )
        h_m2a = lambda: hp.map2alm(m, lmax=lmax, mmax=lmax, iter=0, use_weights=False)  # noqa: E731
        h_a2m = lambda: hp.alm2map(alm, nside, lmax=lmax, mmax=lmax)  # noqa: E731
        for lib, op, fn in (
            ("ducc0", "synthesis (alm2map)", syn),
            ("ducc0", "adjoint_synthesis (map2alm)", adj),
            ("healpy", "alm2map", h_a2m),
            ("healpy", "map2alm iter=0", h_m2a),
        ):
            rec = {
                "lib": lib,
                "op": op,
                "nside": nside,
                "lmax": lmax,
                "npix": 12 * nside * nside,
            }
            rec.update(_time(fn, ARGS.reps))
            rows.append(rec)
            print(json.dumps(rec), file=sys.stderr)
    return {
        "mode": "timings",
        "threads": ARGS.threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu": _cpu_name(),
        "os_cpu_count": os.cpu_count(),
        "ducc0": ducc0.__version__,
        "healpy": hp.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "rows": rows,
    }


def _rel(diff: np.ndarray, ref: np.ndarray) -> dict:
    return {
        "rel_max": float(np.max(np.abs(diff)) / np.max(np.abs(ref))),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
    }


def correctness() -> dict:
    import torch
    import torchsky
    from torchsky.sphere import spectral

    dev = torch.device(ARGS.device)
    rows = []
    for nside in (int(s) for s in ARGS.nsides.split(",")):
        lmax = 2 * nside
        alm = _random_alm(lmax, seed=1000 + nside)
        m_ref = _ducc_synthesis(alm, nside, lmax, ARGS.threads)
        alm_ref = _ducc_map2alm(m_ref, nside, lmax, ARGS.threads)
        m_ts = spectral.alm2map(
            torch.from_numpy(alm).to(dev), nside, lmax=lmax, mmax=lmax
        )
        alm_ts = spectral.map2alm(
            torch.from_numpy(m_ref).to(dev), lmax=lmax, mmax=lmax, iter=0
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        m_ts = m_ts.detach().cpu().numpy().reshape(-1)
        alm_ts = alm_ts.detach().cpu().numpy().reshape(-1)
        rows.append(
            {
                "op": "alm2map vs ducc0 synthesis",
                "nside": nside,
                "lmax": lmax,
                "torchsky_dtype": "float64",
                **_rel(m_ts - m_ref, m_ref),
            }
        )
        rows.append(
            {
                "op": "map2alm(iter=0) vs ducc0 adjoint_synthesis*4pi/npix",
                "nside": nside,
                "lmax": lmax,
                "torchsky_dtype": "complex128",
                **_rel(alm_ts - alm_ref, alm_ref),
            }
        )
        print(json.dumps(rows[-2]), json.dumps(rows[-1]), file=sys.stderr)
    return {
        "mode": "correctness",
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(0)
        if dev.type == "cuda"
        else _cpu_name(),
        "torch": torch.__version__,
        "torchsky": getattr(torchsky, "__version__", None),
        "torchsky_file": torchsky.__file__,
        "ducc0": ducc0.__version__,
        "ducc0_threads": ARGS.threads,
        "rows": rows,
    }


def main() -> None:
    doc = timings() if ARGS.mode == "timings" else correctness()
    text = json.dumps(doc, indent=2)
    print(text)
    with open(ARGS.json_out, "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
