"""Parity tests for extended WCS projections (AZP, SZP, AIR, PAR, BON, COP, COE, COD, COO).

Validates torch-native WCS transforms against Astropy/WCSLIB fixture behavior
for forward (pixel->world) and inverse (world->pixel) directions.
"""

import platform

import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky import WCS as TorchWCS


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
    pv2_1: float | None = None,
    pv2_2: float | None = None,
    cd_scale: float = 0.01,
) -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CD1_1"] = -cd_scale
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = cd_scale

    if pv2_1 is not None:
        header["PV2_1"] = pv2_1
    if pv2_2 is not None:
        header["PV2_2"] = pv2_2

    return header


def _sample_pixels(
    header: dict,
    n_side: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0


# --- Zenithal projections ---

@pytest.mark.parametrize("projection,pv2_1,pv2_2", [
    ("AZP", 0.0, None),      # AZP with mu=0 (standard perspective)
    ("AZP", 2.0, None),      # AZP with mu=2
    ("AZP", 0.0, 20.0),      # AZP with tilt gamma=20 (PV2_2)
    ("SZP", 0.0, None),      # SZP with mu=0 (defaults: phi_c=0, theta_c=90)
    ("SZP", 3.0, None),      # SZP with mu=3
    ("AIR", 90.0, None),     # AIR with theta_b=90 (default)
    ("AIR", 60.0, None),     # AIR with theta_b=60
])
@pytest.mark.parametrize("origin", [0, 1])
def test_zenithal_extended_forward_matches_astropy(
    projection: str, pv2_1: float, pv2_2: float | None, origin: int
) -> None:
    header = _make_header(projection, pv2_1=pv2_1, pv2_2=pv2_2)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for {projection}"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("projection,pv2_1,pv2_2", [
    ("AZP", 0.0, None),
    ("AZP", 2.0, None),
    ("SZP", 0.0, None),
    ("SZP", 3.0, None),
    ("AIR", 90.0, None),
    ("AIR", 60.0, None),
])
@pytest.mark.parametrize("origin", [0, 1])
def test_zenithal_extended_inverse_matches_astropy(
    projection: str, pv2_1: float, pv2_2: float | None, origin: int
) -> None:
    header = _make_header(projection, pv2_1=pv2_1, pv2_2=pv2_2)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for {projection}"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- Pseudocylindrical (PAR) ---

@pytest.mark.parametrize("origin", [0, 1])
def test_par_forward_matches_astropy(origin: int) -> None:
    header = _make_header("PAR", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), "No finite parity points for PAR"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("origin", [0, 1])
def test_par_inverse_matches_astropy(origin: int) -> None:
    header = _make_header("PAR", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), "No finite inverse points for PAR"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- Bonne (BON) ---

@pytest.mark.parametrize("pv2_1", [30.0, 45.0, 60.0])
@pytest.mark.parametrize("origin", [0, 1])
def test_bon_forward_matches_astropy(pv2_1: float, origin: int) -> None:
    header = _make_header("BON", pv2_1=pv2_1, cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for BON pv2_1={pv2_1}"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("pv2_1", [30.0, 45.0, 60.0])
@pytest.mark.parametrize("origin", [0, 1])
def test_bon_inverse_matches_astropy(pv2_1: float, origin: int) -> None:
    header = _make_header("BON", pv2_1=pv2_1, cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for BON pv2_1={pv2_1}"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- Conic projections (COP, COE, COD, COO) ---

# All conic projections follow the WCSLIB convention: PV2_1 = sigma
# (reference / cone-apex latitude), PV2_2 = delta (half-difference of
# standard parallels).  The two standard parallels are sigma - delta
# and sigma + delta.
@pytest.mark.parametrize("projection,pv2_1,pv2_2", [
    ("COP", 45.0, 0.0),     # Single standard parallel
    ("COP", 30.0, 10.0),    # Two-standard: sigma=30, delta=10 → 20°,40°
    ("COE", 45.0, 0.0),
    ("COE", 30.0, 10.0),
    ("COD", 45.0, 0.0),
    ("COD", 30.0, 10.0),    # Two-standard: sigma=30, delta=10 → 20°,40°
    ("COO", 45.0, 0.0),
    ("COO", 30.0, 10.0),
])
@pytest.mark.parametrize("origin", [0, 1])
def test_conic_forward_matches_astropy(
    projection: str, pv2_1: float, pv2_2: float, origin: int
) -> None:
    # Conic projections need CRVAL2 near the standard parallel
    header = _make_header(
        projection, crval2=pv2_1, pv2_1=pv2_1, pv2_2=pv2_2, cd_scale=0.02,
    )
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for {projection}"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("projection,pv2_1,pv2_2", [
    ("COP", 45.0, 0.0),
    ("COP", 30.0, 10.0),
    ("COE", 45.0, 0.0),
    ("COE", 30.0, 10.0),
    ("COD", 45.0, 0.0),
    ("COD", 30.0, 10.0),
    ("COO", 45.0, 0.0),
    ("COO", 30.0, 10.0),
])
@pytest.mark.parametrize("origin", [0, 1])
def test_conic_inverse_matches_astropy(
    projection: str, pv2_1: float, pv2_2: float, origin: int
) -> None:
    header = _make_header(
        projection, crval2=pv2_1, pv2_1=pv2_1, pv2_2=pv2_2, cd_scale=0.02,
    )
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for {projection}"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- PCO (American polyconic) ---

@pytest.mark.parametrize("origin", [0, 1])
def test_pco_forward_matches_astropy(origin: int) -> None:
    header = _make_header("PCO", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), "No finite parity points for PCO"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("origin", [0, 1])
def test_pco_inverse_matches_astropy(origin: int) -> None:
    header = _make_header("PCO", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), "No finite inverse points for PCO"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- XPH (HEALPix polar / butterfly) ---

@pytest.mark.parametrize("origin", [0, 1])
def test_xph_forward_matches_astropy(origin: int) -> None:
    header = _make_header("XPH", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), "No finite parity points for XPH"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("origin", [0, 1])
def test_xph_inverse_matches_astropy(origin: int) -> None:
    header = _make_header("XPH", cd_scale=0.5)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), "No finite inverse points for XPH"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- NCP (obsolete SIN slant) ---

@pytest.mark.parametrize("crval2", [30.0, 45.0, 60.0])
@pytest.mark.parametrize("origin", [0, 1])
def test_ncp_forward_matches_astropy(crval2: float, origin: int) -> None:
    header = _make_header("NCP", crval2=crval2, cd_scale=0.01)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for NCP crval2={crval2}"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-5)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-5)


@pytest.mark.parametrize("crval2", [30.0, 45.0, 60.0])
@pytest.mark.parametrize("origin", [0, 1])
def test_ncp_inverse_matches_astropy(crval2: float, origin: int) -> None:
    header = _make_header("NCP", crval2=crval2, cd_scale=0.01)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for NCP crval2={crval2}"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-4)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-4)


# --- Quad-cube (TSC, CSC, QSC) ---

@pytest.mark.parametrize("projection", ["TSC", "CSC", "QSC"])
@pytest.mark.parametrize("origin", [0, 1])
def test_quadcube_forward_matches_astropy(projection: str, origin: int) -> None:
    # Stay near face-1 centre so samples remain on the cross layout.
    header = _make_header(projection, cd_scale=0.3)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=origin,
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for {projection}"
    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    atol = 1e-6 if projection == "CSC" else 1e-5
    np.testing.assert_allclose(dra, 0.0, atol=atol)
    np.testing.assert_allclose(ddec, 0.0, atol=atol)


@pytest.mark.parametrize("projection", ["TSC", "CSC", "QSC"])
@pytest.mark.parametrize("origin", [0, 1])
def test_quadcube_inverse_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection, cd_scale=0.3)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for {projection}"
    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid), torch.from_numpy(dec_valid), origin=origin,
    )
    atol = 1e-8 if projection == "CSC" else 1e-4
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=atol)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=atol)




def test_csc_cpu_matches_wcslib() -> None:
    """CSC CPU (batched ATen path) agrees with WCSLIB to double-precision rounding.

    torchsky 87f8c3ce rounds the CSC polynomial with uncontracted float32
    ``a + x*b``, identically on every device. On x86-64 that is bit-for-bit what
    WCSLIB computes (CANFAR: 2.8e-14 deg), so 1e-12 deg is the parity bound.
    On arm64 the astropy-bundled WCSLIB is compiled with FMA contraction, so the
    oracle itself sits one float32 ulp (~8 mas, measured 8.4e-7 deg on Apple
    silicon) away; bound that at 3e-6 deg (1 mas = 2.78e-7 deg).
    """
    header = _make_header("CSC", cd_scale=0.3)
    awcs = AstropyWCS(header)
    x, y = _sample_pixels(header, n_side=15)
    python = TorchWCS(dict(header))
    ra_p, dec_p = python.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)
    ra_a, dec_a = awcs.all_pix2world(x, y, 0)
    dra = _ra_wrap_delta(ra_p.numpy(), ra_a)
    ddec = dec_p.numpy() - dec_a
    max_err_deg = float(max(np.nanmax(np.abs(dra)), np.nanmax(np.abs(ddec))))
    fma_oracle = platform.machine().lower() in {"arm64", "aarch64"}
    assert max_err_deg < (3.0e-6 if fma_oracle else 1.0e-12)
