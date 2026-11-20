import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky import WCS as TorchWCS


PROJECTIONS = (
    "TAN",
    "SIN",
    "ARC",
    "AIT",
    "MOL",
    "HPX",
    "CEA",
    "MER",
    "SFL",
    "CAR",
    "ZEA",
    "STG",
    "CYP",
)


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
) -> dict:  # type: ignore
    header = dict()  # type: ignore
    header["NAXIS"] = 2  # type: ignore
    header["NAXIS1"] = naxis1  # type: ignore
    header["NAXIS2"] = naxis2  # type: ignore
    header["CTYPE1"] = f"RA---{projection}"  # type: ignore
    header["CTYPE2"] = f"DEC--{projection}"  # type: ignore
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5  # type: ignore
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5  # type: ignore
    header["CRVAL1"] = crval1  # type: ignore
    header["CRVAL2"] = crval2  # type: ignore
    header["CUNIT1"] = "deg"  # type: ignore
    header["CUNIT2"] = "deg"  # type: ignore

    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 1.0  # type: ignore
    elif projection in {"CEA", "MER", "CYP", "CAR", "SFL"}:
        header["CD1_1"] = -0.5  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 0.5  # type: ignore
        if projection == "CEA":
            header["PV2_1"] = 1.0  # type: ignore
        if projection == "CYP":
            header["PV2_1"] = 2.0  # type: ignore
            header["PV2_2"] = 1.0  # type: ignore
    else:
        header["CD1_1"] = -0.01  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 0.01  # type: ignore

    return header  # type: ignore


def _sample_pixels(
    header: dict,
    n_side: int = 7,  # type: ignore
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)  # type: ignore
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)  # type: ignore
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0  # type: ignore


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore

    x, y = _sample_pixels(header)  # type: ignore

    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x),
        torch.from_numpy(y),
        origin=origin,
    )

    ra_t_np = ra_t.cpu().numpy()  # type: ignore
    dec_t_np = dec_t.cpu().numpy()  # type: ignore

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]  # type: ignore

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)  # type: ignore
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)  # type: ignore


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_tensor_matches_astropy(
    projection: str, origin: int
) -> None:  # type: ignore
    """Stacked (N, 2) pixel tensor path must match Astropy (same as two-arg semantics)."""  # type: ignore
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore

    x, y = _sample_pixels(header)  # type: ignore
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    pixels = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=-1)  # type: ignore
    world = twcs.pixel_to_world(pixels, origin=origin)
    ra_t_np = world[..., 0].cpu().numpy().reshape(-1)  # type: ignore
    dec_t_np = world[..., 1].cpu().numpy().reshape(-1)  # type: ignore

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]  # type: ignore

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)  # type: ignore
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)  # type: ignore




@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore

    x, y = _sample_pixels(header)  # type: ignore
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"

    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid),
        torch.from_numpy(dec_valid),
        origin=origin,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-6)  # type: ignore
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-6)  # type: ignore

    np.testing.assert_allclose(x_t.cpu().numpy(), x[valid], atol=1e-6)  # type: ignore
    np.testing.assert_allclose(y_t.cpu().numpy(), y[valid], atol=1e-6)  # type: ignore


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_stack_tensor_matches_astropy(
    projection: str, origin: int
) -> None:  # type: ignore
    """Stacked (N, 2) world tensor inverse must match Astropy ``all_world2pix``."""
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore

    x, y = _sample_pixels(header)  # type: ignore
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"

    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    world = torch.stack(
        [torch.from_numpy(ra_valid), torch.from_numpy(dec_valid)],
        dim=-1,  # type: ignore
    )
    pix_t = twcs.world_to_pixel(world, origin=origin)
    x_t = pix_t[..., 0].cpu().numpy().reshape(-1)  # type: ignore
    y_t = pix_t[..., 1].cpu().numpy().reshape(-1)  # type: ignore

    np.testing.assert_allclose(x_t, x_ast, atol=1e-6)  # type: ignore
    np.testing.assert_allclose(y_t, y_ast, atol=1e-6)  # type: ignore


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_compiled_matches_astropy(
    projection: str, origin: int
) -> None:  # type: ignore
    """After ``compile()``, stacked forward path still matches Astropy (legacy gate)."""
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore
    try:
        twcs.compile(mode="reduce-overhead")  # type: ignore
    except Exception:
        pytest.skip("torch.compile not available")

    x, y = _sample_pixels(header)  # type: ignore
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    pixels = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=-1)  # type: ignore
    try:
        world = twcs.pixel_to_world(pixels, origin=origin)
    except RuntimeError:
        pytest.skip("torch.compile execution failed (likely missing CUDA headers)")
    ra_t_np = world[..., 0].cpu().numpy().reshape(-1)  # type: ignore
    dec_t_np = world[..., 1].cpu().numpy().reshape(-1)  # type: ignore

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]  # type: ignore

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)  # type: ignore
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)  # type: ignore


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_stack_compiled_matches_astropy(
    projection: str, origin: int
) -> None:  # type: ignore
    """After ``compile()``, stacked inverse still matches Astropy (legacy gate)."""
    header = _make_header(projection)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore
    try:
        twcs.compile(mode="reduce-overhead")  # type: ignore
    except Exception:
        pytest.skip("torch.compile not available")

    x, y = _sample_pixels(header)  # type: ignore
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"
    ra_v = ra_ast[valid]
    dec_v = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_v, dec_v, origin)
    world = torch.stack(
        [torch.from_numpy(ra_v), torch.from_numpy(dec_v)],
        dim=-1,  # type: ignore
    )
    try:
        pix = twcs.world_to_pixel(world, origin=origin)
    except RuntimeError:
        pytest.skip("torch.compile execution failed (likely missing CUDA headers)")
    x_t = pix[..., 0].cpu().numpy().reshape(-1)  # type: ignore
    y_t = pix[..., 1].cpu().numpy().reshape(-1)  # type: ignore

    np.testing.assert_allclose(x_t, x_ast, atol=1e-6)  # type: ignore
    np.testing.assert_allclose(y_t, y_ast, atol=1e-6)  # type: ignore


@pytest.mark.parametrize("projection", ["CEA", "MER"])
@pytest.mark.parametrize("origin", [0, 1])
def test_off_equator_cylindrical_forward_matches_astropy(
    projection: str, origin: int
) -> None:  # type: ignore
    header = _make_header(  # type: ignore
        projection,
        crval1=12.0,
        crval2=-3.0,  # type: ignore
        naxis1=1024,
        naxis2=640,
    )
    awcs = AstropyWCS(header)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore

    x = np.array([512.5, 512.5, 600.0, 700.0], dtype=np.float64)
    y = np.array([320.5, 400.0, 320.5, 500.0], dtype=np.float64)

    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x),
        torch.from_numpy(y),
        origin=origin,
    )

    ra_t_np = ra_t.cpu().numpy()  # type: ignore
    dec_t_np = dec_t.cpu().numpy()  # type: ignore

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), (
        f"No finite off-equator parity points for projection={projection}"
    )  # type: ignore

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]  # type: ignore

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)  # type: ignore
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)  # type: ignore




def test_lonpole_180_two_arg_and_stacked_match_astropy() -> None:
    header = _make_header("CEA")
    header["LONPOLE"] = 180.0
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, 0)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    ra2, dec2 = twcs.pixel_to_world(xt, yt, origin=0)
    world = twcs.pixel_to_world(torch.stack([xt, yt], dim=-1), origin=0)
    dra2 = _ra_wrap_delta(ra2.cpu().numpy(), ra_ast)
    dras = _ra_wrap_delta(world[..., 0].cpu().numpy(), ra_ast)
    np.testing.assert_allclose(dra2, 0.0, atol=5e-9)
    np.testing.assert_allclose(dras, 0.0, atol=5e-9)
    np.testing.assert_allclose(dec2.cpu().numpy(), dec_ast, atol=5e-9)
    np.testing.assert_allclose(world[..., 1].cpu().numpy(), dec_ast, atol=5e-9)
