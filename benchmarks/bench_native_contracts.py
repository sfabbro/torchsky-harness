#!/usr/bin/env python3
"""Adversarial contract probes for the installed torchsky subject.

The scientific gates in this repository compare valid-domain results against
reference implementations. They do not exercise malformed structural inputs,
invalid projection domains, tensor-layout contracts, concurrency, or high-order
spectral cases. This script adds that missing layer.

Every probe runs in its own subprocess with a hard timeout, so a crash or hang is
recorded as a failed row instead of killing the driver. A probe passes only when
its child exits 0; the child raises AssertionError when the expected contract is
violated.

Usage (the harness environment must have the candidate subject installed):
    pixi run -e test -- python benchmarks/bench_native_contracts.py \
        --json-out benchmark_results/native_contracts.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
import time
from pathlib import Path

# Each entry is a self-contained child process. Keep assertions in the child so a
# failing contract produces a nonzero return code.
PROBES: dict[str, str] = {
    "empty_ring_synthesis": r"""
import torch
from torchsky import _ops
s = torch.empty((1, 1, 0), dtype=torch.complex128)
st = torch.empty((0,), dtype=torch.int64)
ln = torch.empty((0,), dtype=torch.int64)
out = _ops.spectral_ring_fourier_synthesis(s, st, ln, 0)
assert tuple(out.shape) == (1, 0), out.shape
assert bool(torch.isfinite(out).all())
print("empty ring synthesis:", tuple(out.shape))
""",
    "empty_healpix_arrays": r"""
import torch
from torchsky.sphere import healpix as hp
p = torch.empty((0,), dtype=torch.int64)
lon, lat = hp.pix2ang(4, p, lonlat=True)
assert tuple(lon.shape) == (0,) and tuple(lat.shape) == (0,)
lon0 = torch.empty((0,), dtype=torch.float64)
pix = hp.ang2pix(4, lon0, lon0, lonlat=True)
assert tuple(pix.shape) == (0,)
print("empty HEALPix arrays ok")
""",
    "invalid_healpix_coordinates": r"""
import torch
from torchsky.sphere import healpix as hp
lon = torch.tensor([float("nan"), 10.0, 10.0], dtype=torch.float64)
lat = torch.tensor([0.0, 95.0, -95.0], dtype=torch.float64)
try:
    pix = hp.ang2pix(4, lon, lat, lonlat=True)
except Exception:
    print("invalid coordinates raised")
else:
    raise AssertionError(f"invalid coordinates returned pixels {pix.tolist()}")
""",
    "tan_back_hemisphere": r"""
import numpy as np, torch
from astropy.wcs import WCS as AW
from torchsky.wcs import WCS
h = {"NAXIS":2,"NAXIS1":64,"NAXIS2":64,"CRPIX1":32.0,"CRPIX2":32.0,
     "CRVAL1":0.0,"CRVAL2":0.0,"CDELT1":0.01,"CDELT2":0.01,
     "CTYPE1":"RA---TAN","CTYPE2":"DEC--TAN"}
aw = AW(h); tw = WCS(h)
ra = np.array([91.0, 180.0, 270.0]); dec = np.zeros(3)
ax, ay = aw.all_world2pix(ra, dec, 0)
tx, ty = tw.world_to_pixel(torch.tensor(ra, dtype=torch.float64),
                           torch.tensor(dec, dtype=torch.float64))
tx = tx.detach().numpy(); ty = ty.detach().numpy()
astropy_invalid = ~np.isfinite(ax)
torchsky_finite = np.isfinite(tx)
assert np.array_equal(astropy_invalid, ~torchsky_finite), (
    f"domain mismatch: astropy_invalid={astropy_invalid.tolist()} "
    f"torchsky_finite={torchsky_finite.tolist()}"
)
print("TAN invalid-domain parity ok")
""",
    "mol_polar_roundtrip": r"""
import numpy as np, torch
from astropy.wcs import WCS as AW
from torchsky.wcs import WCS
h = {"NAXIS":2,"NAXIS1":4096,"NAXIS2":4096,"CRPIX1":2048.0,"CRPIX2":2048.0,
     "CRVAL1":0.0,"CRVAL2":0.0,"CD1_1":-1.0,"CD1_2":0.0,"CD2_1":0.0,"CD2_2":1.0,
     "CTYPE1":"RA---MOL","CTYPE2":"DEC--MOL"}
aw = AW(h); tw = WCS(h)
ra = np.array([10.0, 20.0, 30.0]); dec = np.array([89.9, 89.99, 89.999])
x, y = aw.all_world2pix(ra, dec, 0)
rx, ry = tw.world_to_pixel(torch.tensor(ra, dtype=torch.float64),
                           torch.tensor(dec, dtype=torch.float64))
rra, rdec = tw.pixel_to_world(rx, ry)
rdec = rdec.detach().numpy(); a_dec = np.asarray(aw.all_pix2world(x, y, 0)[1])
err = float(np.max(np.abs(rdec - a_dec)))
assert np.isfinite(rdec).all(), rdec
assert err < 1.0e-5, f"MOL polar dec error {err} deg"
print("MOL polar max dec error deg:", err)
""",
    "generalized_hpx": r"""
import numpy as np, torch
from astropy.wcs import WCS as AW
from torchsky.wcs import WCS
h = {"NAXIS":2,"NAXIS1":4096,"NAXIS2":4096,"CRPIX1":2048.0,"CRPIX2":2048.0,
     "CRVAL1":0.0,"CRVAL2":0.0,"CUNIT1":"deg","CUNIT2":"deg",
     "CTYPE1":"RA---HPX","CTYPE2":"DEC--HPX",
     "CD1_1":-1.0,"CD1_2":0.0,"CD2_1":0.0,"CD2_2":1.0,
     "PV2_1":6.0,"PV2_2":4.0}
aw = AW(h); tw = WCS(h)
rng = np.random.default_rng(0)
ra = rng.uniform(0, 360, 2000); dec = rng.uniform(-89, 89, 2000)
ax, ay = aw.all_world2pix(ra, dec, 0)
tx, ty = tw.world_to_pixel(torch.tensor(ra), torch.tensor(dec))
tx = tx.detach().numpy(); ty = ty.detach().numpy()
finite = np.isfinite(ax) & np.isfinite(tx) & np.isfinite(ty)
assert finite.sum() >= 1990, f"only {int(finite.sum())} finite HPX points"
diff = float(np.nanmax(np.hypot(tx[finite] - ax[finite], ty[finite] - ay[finite])))
assert diff < 1.0e-6, f"HPX H=6,K=4 pixel diff {diff}"
print("HPX H=6,K=4 max pixel diff:", diff)
""",
    "invalid_tpv_metadata": r"""
from torchsky.wcs import WCS
h = {"NAXIS":2,"NAXIS1":8,"NAXIS2":8,"CRPIX1":4.0,"CRPIX2":4.0,
     "CRVAL1":0.0,"CRVAL2":0.0,"CD1_1":1e-3,"CD1_2":0.0,"CD2_1":0.0,"CD2_2":1e-3,
     "CTYPE1":"RA---TPV","CTYPE2":"DEC--TPV",
     "PV1_1":1.0,"PV1_39":"not-a-float"}
try:
    WCS(h)
except Exception:
    print("malformed TPV metadata raised")
else:
    raise AssertionError("malformed TPV metadata was accepted")
""",
    "scalar_sht_high_order": r"""
import numpy as np, torch
from torchsky.sphere.spectral import alm2map, alm_size
lmax = 256
nside = 128
rng = np.random.default_rng(0)
alm = torch.tensor(
    rng.normal(size=alm_size(lmax)) + 1j * rng.normal(size=alm_size(lmax)),
    dtype=torch.complex128,
)
m = alm2map(alm, nside=nside, lmax=lmax)
assert bool(torch.isfinite(m).all()), "non-finite SHT output at lmax=mmax=256"
print("high-order scalar SHT max:", float(m.abs().max()))
""",
    "spin_sht_concurrency": r"""
import threading, torch
from torchsky.sphere.spectral import alm2map_spin, map2alm_spin
nside = 32; lmax = 32; spin = 2
npix = 12 * nside * nside
qu = torch.randn((2, npix), dtype=torch.float64)
errors = []
def work():
    try:
        ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
        m = alm2map_spin((ae, ab), nside=nside, spin=spin, lmax=lmax)
        assert bool(torch.isfinite(m).all())
    except BaseException as exc:  # noqa: BLE001 - reported as a probe row
        errors.append(repr(exc))
threads = [threading.Thread(target=work) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=60)
assert not any(t.is_alive() for t in threads), "concurrent spin SHT hung"
assert not errors, errors
print("concurrent spin SHT ok")
""",
    "tiny_arc_geometry": r"""
import torch
from torchsky.sphere.geom import spherical_polygon_area, spherical_polygon_contains
lon = torch.tensor([0.0, 2e-8, 0.0], dtype=torch.float64)
lat = torch.tensor([0.0, 0.0, 2e-8], dtype=torch.float64)
area = spherical_polygon_area(lon, lat)
assert float(area) > 0.0, area
hit = spherical_polygon_contains(
    torch.tensor(0.5e-8), torch.tensor(0.5e-8), lon, lat
)
assert bool(hit), hit
print("tiny arc area:", float(area))
""",
    "dispatcher_healpix_contract": r"""
import torch
import torchsky._ops as ops
assert ops.has_spectral_dispatcher() and ops.has_fixed_dispatcher()
lon = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
lat = torch.tensor([-5.0, 0.0, 5.0, 85.0], dtype=torch.float64)
direct = torch.ops.torchsky.healpix_ang2pix_ring(64, lon, lat)
legacy = ops.healpix_ang2pix_ring(64, lon, lat)
assert torch.equal(direct, legacy), (direct, legacy)
strided_lon = torch.empty(8, dtype=torch.float64)[::2]
strided_lat = torch.empty(8, dtype=torch.float64)[1::2]
strided_lon.copy_(lon)
strided_lat.copy_(lat)
strided = torch.ops.torchsky.healpix_ang2pix_ring(64, strided_lon, strided_lat)
assert torch.equal(strided, direct), (strided, direct)
empty = torch.ops.torchsky.healpix_ang2pix_ring(
    64, torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
)
assert tuple(empty.shape) == (0,), empty.shape
print("dispatcher healpix contract ok")
""",
    "noncontiguous_wcs": r"""
import numpy as np, torch
from torchsky.wcs import WCS
h = {"NAXIS":2,"NAXIS1":16,"NAXIS2":16,"CRPIX1":8.0,"CRPIX2":8.0,
     "CRVAL1":150.0,"CRVAL2":2.0,"CDELT1":-1e-3,"CDELT2":1e-3,
     "CTYPE1":"RA---TAN","CTYPE2":"DEC--TAN"}
w = WCS(h)
base = torch.randn(6, 8, dtype=torch.float64)
x = base[:, ::2].contiguous()
y = base[:, 1::2].contiguous()
view = torch.empty(6, 8, dtype=torch.float64)[:, ::2]
view.copy_(x)
x_nc = view
ra1, dec1 = w.pixel_to_world(x, y)
ra2, dec2 = w.pixel_to_world(x_nc, y)
assert bool(torch.isfinite(ra1).all()) and bool(torch.isfinite(dec1).all())
assert torch.allclose(ra1, ra2, equal_nan=True, atol=0.0)
assert torch.allclose(dec1, dec2, equal_nan=True, atol=0.0)
print("noncontiguous WCS input ok")
""",
}


def _run_probe(name: str, timeout: int) -> dict[str, object]:
    code = textwrap.dedent(PROBES[name])
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        return {
            "case": name,
            "status": "OK" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "signal": -proc.returncode if proc.returncode < 0 else None,
            "elapsed_s": round(elapsed, 3),
            "stdout": proc.stdout.strip()[-4000:],
            "stderr": proc.stderr.strip()[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "case": name,
            "status": "FAIL",
            "returncode": None,
            "signal": None,
            "elapsed_s": round(elapsed, 3),
            "stdout": (exc.stdout or "")[-4000:]
            if isinstance(exc.stdout, str)
            else "",
            "stderr": "TIMEOUT",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/native_contracts.json"),
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="run only the named probe(s)",
    )
    args = parser.parse_args(argv)

    names = [name for name in PROBES if not args.probe or name in args.probe]
    rows = [_run_probe(name, args.timeout) for name in names]
    subject = {}
    try:
        import torch
        import torchsky

        subject = {
            "torchsky_version": torchsky.__version__,
            "torchsky_import_path": str(Path(torchsky.__file__).resolve()),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "python": sys.version,
        }
    except Exception as exc:  # noqa: BLE001 - provenance only
        subject = {"import_error": repr(exc)}

    failures = [row["case"] for row in rows if row["status"] != "OK"]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    # The evidence claim evaluator expects a top-level row list with ``case``
    # and ``status``; keep provenance in a sidecar so the contract rows remain
    # machine-checkable.
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    sidecar = args.json_out.with_suffix(args.json_out.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "subject": subject,
                "failures": failures,
                "probe_count": len(rows),
            },
            indent=2,
        )
        + "\n"
    )

    for row in rows:
        print(
            f"{row['case']}: {row['status']} ({row.get('elapsed_s')} s)"
            + (
                f" rc={row['returncode']} signal={row['signal']}"
                if row["status"] != "OK"
                else ""
            )
        )
        if row["status"] != "OK" and row["stderr"]:
            print(textwrap.indent(str(row["stderr"]), "  "))
    print(f"json: {args.json_out}")
    print(f"provenance: {sidecar}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
