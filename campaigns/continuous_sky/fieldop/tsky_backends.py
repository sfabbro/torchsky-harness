"""Optional torchsky back ends of the prototype (each behind its own command-line switch; the defaults keep the prototype's own code).

  A. --item-impl torchsky      item sky rasterised by torchsky.scene.GaussianItemScene            (item_raster_torchsky)
  B. --rate-wcs telescope      count-rate pixel positions from torchsky.wcs.TelescopeFrameDetectorToSky (telescope_wcs_for_rate_file)
  C. --loader torchfits        exposure files read by torchsky.map.field.field_observation_from_exposure_file (read_exposure_file_torchfits)
  D. --render-impl torchsky    NOT WIRED IN: see TORCHSKY_INTEGRATION_REPORT.md (render_nested_psf_reference is the measurement helper only)

torchsky is imported lazily, only when a switch asks for it, so the default path never imports it through this module.  If an import
fails the error says which switch needs it and that the default value of the switch needs no torchsky piece.
"""
from __future__ import annotations

import math

import numpy as np
import torch

TSKY_SRC_HINT = "set PYTHONPATH to the torchsky 'src' folder (the one that holds the 'torchsky' package)"


def _need(feature, what):
    try:
        return __import__(what, fromlist=["_"])
    except Exception as exc:  # noqa: BLE001  (any import failure, including a broken compiled extension of torchfits)
        raise RuntimeError(f"{feature} needs {what}, which could not be imported ({type(exc).__name__}: {exc}). {TSKY_SRC_HINT}. "
                           f"The default value of this switch does not use it.") from exc


# ============================================================================= A. items through GaussianItemScene
# Prototype coordinates: X to the west, Y to the north [arcsec], grid node (i, j) at X = x0 + j p, Y = x0 + i p; item width w0 along
# (cos a, sin a) in (X, Y), w1 across it.  A tangent plane at (RA0, DEC0) with CDELT1 = -p, CDELT2 = +p arcsec and the centre pixel on X = Y = 0
# maps pixel (j, i) to exactly these coordinates, so the pixel grid of the scene IS the prototype's grid.  The scene's angle is east of
# north, major axis along (sin t, cos t) in (east, north) = (-sin t, cos t) in (west, north); equating with (cos a, sin a): t = a - 90 deg.
_RA0, _DEC0 = 150.0, 2.0


def grid_header(N, pitch):
    s = float(pitch) / 3600.0
    c = (N - 1) / 2 + 1.0
    return {"NAXIS": 2, "NAXIS1": int(N), "NAXIS2": int(N), "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
            "CRVAL1": _RA0, "CRVAL2": _DEC0, "CRPIX1": c, "CRPIX2": c, "CDELT1": -s, "CDELT2": s}


class _SceneCall(torch.nn.Module):
    """Holds a GaussianItemScene so that torch.func.functional_call can substitute NON-leaf tensors for its parameters (the scene wraps what
    it is given in nn.Parameter, which would cut the graph to the prototype's own parameters)."""

    def __init__(self, scene, header):
        super().__init__()
        self.scene = scene
        self.header = header

    def forward(self):
        return self.scene.render_coefficients(self.header)


def _sky_chunk():
    from . import sky as _sky
    return _sky.ITEM_CHUNK


def _scene_for(sky, n_items, K, dtype):
    key = (n_items, K, sky.raster_mode, float(sky.floor), int(sky.N), float(sky.pitch), dtype, int(_sky_chunk()))
    cache = sky.__dict__.setdefault("_tsky_cache", {})
    if key not in cache:
        from torchsky.wcs import WCS
        from torchsky.scene.gaussians import GaussianItemScene
        from . import sky as _sky
        z = torch.zeros(n_items, dtype=torch.float64)
        gp = torch.stack([torch.ones(n_items), torch.ones(n_items), torch.zeros(n_items)], 1).double()
        scene = GaussianItemScene(z + _RA0, z + _DEC0, gp, torch.ones(n_items, K, dtype=dtype), min_sigma_arcsec=float(sky.floor),
                                  mode=sky.raster_mode, pixel_integration=True, chunk_elements=int(_sky.ITEM_CHUNK))
        hdr = grid_header(sky.N, sky.pitch)
        cache.clear()
        cache[key] = (_SceneCall(scene, hdr), WCS(hdr))
    return cache[key]


def item_raster_torchsky(sky, pos, lw, ang, lc):
    """(K, N, N) coefficient planes of the items of ONE view through GaussianItemScene; same parameters as ItemSky.raster (pos (n, 2) [arcsec],
    lw (n, 2) log widths before the floor, ang (n,) [rad], lc (n, K) log coefficients).  Width floor, cap, positivity are the prototype's:
    sigma = exp(min(lw, log max_width)) enters the scene, which adds the floor in quadrature (the prototype's widths(): sqrt(floor^2 + exp(2 lw)))."""
    _need("--item-impl torchsky", "torchsky.scene.gaussians")
    from torch.func import functional_call
    if sky.patch_shape == "sheared":
        raise RuntimeError("--item-impl torchsky does not implement --item-patch sheared (not ported to GaussianItemScene); use --item-patch square")
    n, K = pos.shape[0], lc.shape[1]
    call, wcs = _scene_for(sky, n, K, lc.dtype)
    p64 = pos.double()
    pix = (p64 - sky.x0) / sky.pitch
    world = wcs.pixel_to_world_differentiable(pix, origin=0)             # (n, 2) degrees, differentiable in pos
    sigma = torch.exp(lw.clamp(max=math.log(sky.max_width))).double()
    params = {"scene.lon": world[:, 0], "scene.lat": world[:, 1],
              "scene.sigma_arcsec": sigma, "scene.angle_deg": torch.rad2deg(ang.double()) - 90.0,
              "scene.coefficients": torch.exp(lc)}
    # the scene returns flux per pixel (sum of a plane = coefficient); the prototype's coefficient grid is a density per arcsec^2 (p^2 sum g = 1)
    return functional_call(call, params, ()) / (sky.pitch * sky.pitch)


# ============================================================================= B. count-rate positions from the telescope frame
_SIAF_CACHE = {}


def read_headers(path, which):
    """Headers (plain dicts) of the given extensions; through torchfits when it imports, else through astropy (so that --rate-wcs telescope does
    not depend on torchfits)."""
    try:
        import torchfits
        return [dict(torchfits.read_header(str(path), w)) for w in which]
    except Exception:  # noqa: BLE001
        from astropy.io import fits
        with fits.open(str(path), memmap=False) as h:
            return [{k: h[w].header[k] for k in h[w].header.keys() if k and k not in ("COMMENT", "HISTORY", "")} for w in which]


def siaf_polynomial(aperture_name, instrument="NIRCam"):
    """(coeffs_x, coeffs_y, x_sci_ref, y_sci_ref, degree) of a SIAF aperture, from pysiaf (installed with the prototype's environment, read only)."""
    pysiaf = _need("--rate-wcs telescope", "pysiaf")
    if instrument not in _SIAF_CACHE:
        _SIAF_CACHE[instrument] = pysiaf.Siaf(instrument)
    ap = _SIAF_CACHE[instrument][aperture_name]
    c = ap.get_polynomial_coefficients()
    return (np.asarray(c["Sci2IdlX"], np.float64), np.asarray(c["Sci2IdlY"], np.float64), float(ap.XSciRef), float(ap.YSciRef),
            int(ap.Sci2IdlDeg))


class TelescopeRateWCS:
    """The detector-to-sky map of a count-rate file built from its OWN header (pointing, roll, parity, V3 angle) and the aperture polynomial.
    Offers the one method the prototype calls on a WCS, pixel_to_world_differentiable(pixels (n, 2), origin=0) -> (n, 2) degrees.
    Aperture name: APERNAME of the PRIMARY header (assumed to be the SIAF name; NRCBLONG files say NRCB5_FULL, which is the SIAF name).
    Neglected: velocity aberration (VA_SCALE), the filter-dependent offset and distortion of the pipeline's reference files, time dependence."""

    def __init__(self, rate_path, chunk=262144):
        tfr = _need("--rate-wcs telescope", "torchsky.wcs")
        self.path = str(rate_path)
        sci, prim = read_headers(self.path, ("SCI", 0))
        self.aperture = str(prim.get("APERNAME", "")).strip().upper()
        if not self.aperture:
            raise ValueError(f"{rate_path}: no APERNAME in the primary header; --rate-wcs telescope needs the SIAF aperture name")
        cx, cy, xr, yr, self.degree = siaf_polynomial(self.aperture)
        shape = (int(sci["NAXIS2"]), int(sci["NAXIS1"]))
        self.transform = tfr.TelescopeFrameDetectorToSky.from_header(sci, cx, cy, x_sci_ref=xr, y_sci_ref=yr, detector_shape=shape)
        self.chunk = int(chunk)
        self.va_scale = float(sci.get("VA_SCALE", float("nan")))
        # pixel solid angle at the aperture reference pixel [arcsec^2] from the polynomial's own Jacobian (rate files carry no PIXAR_A2)
        ref = torch.tensor([[xr - 1.0, yr - 1.0]], dtype=torch.float64)
        _, jac = self.transform.distortion.evaluate_with_jacobian(ref)
        self.pixar_a2 = float(torch.linalg.det(jac[0]).abs())

    @torch.no_grad()
    def pixel_to_world_differentiable(self, pixels, origin=0):
        p = torch.as_tensor(pixels, dtype=torch.float64) - float(origin)
        out = []
        for s in range(0, p.shape[0], self.chunk):
            q = p[s:s + self.chunk]
            ra, dec = self.transform.pixel_to_world(q[:, 0], q[:, 1])
            out.append(torch.stack([ra, dec], -1))
        return torch.cat(out) if out else p.new_zeros(0, 2)


# ============================================================================= C. exposure files through torchsky's reader
LOADER_FALLBACK_MESSAGE = ("--loader torchfits: torchfits or torchsky.map.field.field_observation_from_exposure_file could not be imported ({err}); "
                           "FALLING BACK to the astropy reader (--loader astropy) for this run.")
_LOADER_STATE = {"checked": False, "ok": False, "err": ""}


def torchfits_reader_available():
    """True when the torchfits reader can be used.  The import can fail (torchfits is a compiled extension: a missing or mismatched wheel, or an
    unreadable shared library, raises ImportError / OSError at import); the caller then falls back to astropy with a message."""
    if not _LOADER_STATE["checked"]:
        _LOADER_STATE["checked"] = True
        try:
            import torchfits  # noqa: F401
            from torchsky.map.field.exposure_io import field_observation_from_exposure_file  # noqa: F401
            _LOADER_STATE["ok"] = True
        except Exception as exc:  # noqa: BLE001
            _LOADER_STATE["err"] = f"{type(exc).__name__}: {exc}"
    return _LOADER_STATE["ok"]


def read_exposure_file_torchfits(path, rate_path, dq_ignore_bits, rate_wcs):
    """The dict of data.read_exposure_file, read through field_observation_from_exposure_file.  Differences from the astropy dict (all tested):
    sci holds 0 (not the file's value, possibly NaN) at pixels the reader marks unusable; dq is not returned (the reader folds it into the mask:
    key 'usable' = finite science, finite positive error, DQ & ~dq_ignore_bits == 0); everything at usable pixels is identical."""
    import torchfits
    from torchsky.map.field.exposure_io import field_observation_from_exposure_file
    from torchsky.map.quality import DQPolicy
    from torchsky.wcs import WCS
    src = rate_path or path
    reject = 0xFFFFFFFF & ~int(dq_ignore_bits)
    obs = field_observation_from_exposure_file(src, dq_policy=DQPolicy(reject_bits=reject), dtype=torch.float64)
    sci = obs.image.numpy().astype(np.float32)
    err = obs.err.numpy().astype(np.float32)
    usable = obs.mask.numpy()
    names = [str(torchfits.read_extname(str(src), i) or "").strip().upper() for i in range(1, int(torchfits.read_num_hdus(str(src))))]
    area = torchfits.read(str(src), hdu=1 + names.index("AREA")).numpy().astype(np.float32) if "AREA" in names else None
    sh = obs.header
    if rate_path and rate_wcs == "cal":
        sh = {k: v for k, v in dict(torchfits.read_header(str(path), "SCI")).items()}
    hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
    h0 = dict(torchfits.read_header(str(src), 0))
    return dict(sci=sci, err=err, dq=None, usable=usable, area=area, w=WCS(header=hd), pixar_a2=float(sh.get("PIXAR_A2", float("nan"))),
                detector=str(h0.get("DETECTOR", "")).strip().upper(), instrument=str(h0.get("INSTRUME", "")).strip().upper(),
                header_hd=hd)
