#!/usr/bin/env python3
"""
Small-N WCS dispatch matrix: Astropy vs torchsky call shapes (legacy reference).

Measures median wall time per batch for the same pixel sample using:
  - Astropy ``all_pix2world`` / ``all_world2pix`` (NumPy reference),
  - torchsky ``pixel_to_world(x, y)`` (pure-torch two-argument path),
  - torchsky ``pixel_to_world(pixels)`` with ``pixels.shape == (N, 2)`` (tensor transform path),
  - same stacked forward call after ``wcs.compile()`` (``torch.compile``-d ``pixels -> rel -> world``
    graph for FITS ``origin`` in ``{0, 1}``),
  - torchsky ``world_to_pixel(ra, dec)``, ``world_to_pixel(world)`` stacked, and stacked after
    ``compile()`` (``torch.compile``-d inverse graph).

Emits median ms per batch for each path plus max |ΔRA|,|ΔDec| vs Astropy on the stacked
path (sanity). **Do not assume torchsky is faster than Astropy at small ``N``:** on a
typical laptop, Astropy ``all_pix2world`` can still win wall-clock for hundreds of
points; the stacked + ``compile()`` row is there to measure the **Torch-compiled**
graph cost/benefit for integration with tensor pipelines, not as a universal speed win.
Not a CI gate by default.

Example::

    pixi run -e bench-all -- python benchmarks/bench_wcs_small_n_paths.py --cases TAN,SIN --n-points 256 --runs 11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from torchsky_harness.benchmarks.bench_wcs import (  # noqa: E402
    CASES,
    _make_header,
    _resolve_device,
    _sample_pixels,
    _time_many,
)
from torchsky import WCS as TorchWCS


def _median_ms(fn: Any, *, runs: int, device: torch.device) -> float:
    return float(_time_many(fn, runs, sync_device=device) * 1000.0)


def _parity_max_arcsec(
    awcs: AstropyWCS,
    twcs: TorchWCS,
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: int,
) -> dict[str, float]:
    ra_a, dec_a = awcs.all_pix2world(x, y, origin)
    pixels = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        ],
        dim=-1,
    )
    world = twcs.pixel_to_world(pixels, origin=float(origin))
    ra_t = world[..., 0].detach().cpu().numpy().reshape(-1)
    dec_t = world[..., 1].detach().cpu().numpy().reshape(-1)
    valid = (
        np.isfinite(ra_a) & np.isfinite(dec_a) & np.isfinite(ra_t) & np.isfinite(dec_t)
    )
    if not np.any(valid):
        return {"max_dra_deg": float("nan"), "max_ddec_deg": float("nan")}
    dra = ((ra_t[valid] - ra_a[valid] + 180.0) % 360.0) - 180.0
    ddec = dec_t[valid] - dec_a[valid]
    return {
        "max_abs_dra_deg": float(np.max(np.abs(dra))),
        "max_abs_ddec_deg": float(np.max(np.abs(ddec))),
    }


def _parity_inverse_max_pix(
    awcs: AstropyWCS,
    twcs: TorchWCS,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    origin: int,
) -> dict[str, float]:
    x_a, y_a = awcs.all_world2pix(ra, dec, origin)
    world = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(ra)),
            torch.from_numpy(np.ascontiguousarray(dec)),
        ],
        dim=-1,
    )
    pix = twcs.world_to_pixel(world, origin=float(origin))
    x_t = pix[..., 0].detach().cpu().numpy().reshape(-1)
    y_t = pix[..., 1].detach().cpu().numpy().reshape(-1)
    valid = np.isfinite(x_a) & np.isfinite(y_a) & np.isfinite(x_t) & np.isfinite(y_t)
    if not np.any(valid):
        return {"max_dx_pix": float("nan"), "max_dy_pix": float("nan")}
    return {
        "max_dx_pix": float(np.max(np.abs(x_t[valid] - x_a[valid]))),
        "max_dy_pix": float(np.max(np.abs(y_t[valid] - y_a[valid]))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        default="TAN,SIN,ARC",
        help="Comma-separated keys from torchsky_harness.benchmarks.bench_wcs.CASES",
    )
    p.add_argument("--n-points", type=int, default=256)
    p.add_argument("--runs", type=int, default=11)
    p.add_argument("--origin", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    device = _resolve_device(args.device)
    case_keys = [k.strip() for k in args.cases.split(",") if k.strip()]
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for key in case_keys:
        case = CASES[key]
        header = _make_header(case)
        awcs = AstropyWCS(header)
        twcs_cpu = TorchWCS(dict(header))
        twcs = TorchWCS(dict(header)).to(device)
        x, y = _sample_pixels(header, args.n_points, rng, case, "interior")
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        yt = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        pixels = torch.stack([xt, yt], dim=-1)

        parity = _parity_max_arcsec(awcs, twcs_cpu, x, y, origin=args.origin)
        ra_ref, dec_ref = awcs.all_pix2world(x, y, args.origin)
        ra_t = torch.from_numpy(np.ascontiguousarray(ra_ref)).to(device)
        dec_t = torch.from_numpy(np.ascontiguousarray(dec_ref)).to(device)
        world = torch.stack([ra_t, dec_t], dim=-1)
        parity_inv = _parity_inverse_max_pix(
            awcs, twcs_cpu, ra_ref, dec_ref, origin=args.origin
        )

        cpu = torch.device("cpu")
        ms_ast = _median_ms(
            lambda: awcs.all_pix2world(x, y, args.origin),
            runs=args.runs,
            device=cpu,
        )
        ms_xy = _median_ms(
            lambda: twcs.pixel_to_world(xt, yt, origin=args.origin),
            runs=args.runs,
            device=device,
        )
        ms_stack = _median_ms(
            lambda: twcs.pixel_to_world(pixels, origin=args.origin),
            runs=args.runs,
            device=device,
        )

        ms_ast_w2p = _median_ms(
            lambda: awcs.all_world2pix(ra_ref, dec_ref, args.origin),
            runs=args.runs,
            device=cpu,
        )
        ms_w2p_xy = _median_ms(
            lambda: twcs.world_to_pixel(ra_t, dec_t, origin=args.origin),
            runs=args.runs,
            device=device,
        )
        ms_w2p_stack = _median_ms(
            lambda: twcs.world_to_pixel(world, origin=args.origin),
            runs=args.runs,
            device=device,
        )

        twcs_c = TorchWCS(dict(header)).to(device)
        pixels_c = pixels.clone()
        world_c = world.clone()
        try:
            twcs_c.compile(mode="reduce-overhead")
            # Inductor can throw on first invoke after compile() returns.
            twcs_c.pixel_to_world(pixels_c, origin=args.origin)
            twcs_c.world_to_pixel(world_c, origin=args.origin)
        except Exception as e:  # pragma: no cover - env specific
            ms_stack_c = float("nan")
            ms_w2p_stack_c = float("nan")
            compile_note = f"compile_failed:{type(e).__name__}"
        else:
            compile_note = "ok"
            ms_stack_c = _median_ms(
                lambda: twcs_c.pixel_to_world(pixels_c, origin=args.origin),
                runs=args.runs,
                device=device,
            )
            ms_w2p_stack_c = _median_ms(
                lambda: twcs_c.world_to_pixel(world_c, origin=args.origin),
                runs=args.runs,
                device=device,
            )

        row = {
            "case": key,
            "projection": case.projection,
            "n_points": int(args.n_points),
            "device": str(device),
            "origin": int(args.origin),
            "median_ms_astropy_all_pix2world": ms_ast,
            "median_ms_torchsky_pixel_to_world_xy": ms_xy,
            "median_ms_torchsky_pixel_to_world_stack": ms_stack,
            "median_ms_torchsky_pixel_to_world_stack_compiled": ms_stack_c,
            "torchsky_xy_vs_astropy": ms_ast / ms_xy if ms_xy > 0 else float("nan"),
            "torchsky_stack_vs_astropy": ms_ast / ms_stack
            if ms_stack > 0
            else float("nan"),
            "torchsky_stack_compiled_vs_astropy": ms_ast / ms_stack_c
            if ms_stack_c > 0 and np.isfinite(ms_stack_c)
            else float("nan"),
            "median_ms_astropy_all_world2pix": ms_ast_w2p,
            "median_ms_torchsky_world_to_pixel_xy": ms_w2p_xy,
            "median_ms_torchsky_world_to_pixel_stack": ms_w2p_stack,
            "median_ms_torchsky_world_to_pixel_stack_compiled": ms_w2p_stack_c,
            "torchsky_w2p_xy_vs_astropy": ms_ast_w2p / ms_w2p_xy
            if ms_w2p_xy > 0
            else float("nan"),
            "torchsky_w2p_stack_vs_astropy": ms_ast_w2p / ms_w2p_stack
            if ms_w2p_stack > 0
            else float("nan"),
            "torchsky_w2p_stack_compiled_vs_astropy": ms_ast_w2p / ms_w2p_stack_c
            if ms_w2p_stack_c > 0 and np.isfinite(ms_w2p_stack_c)
            else float("nan"),
            "compile": compile_note,
            **parity,
            **parity_inv,
        }
        rows.append(row)

    for r in rows:
        print(
            f"{r['case']:8s}  N={r['n_points']:<6d}  "
            f"ast_p2w={r['median_ms_astropy_all_pix2world']:.4f}ms  "
            f"ts_p2w_xy={r['median_ms_torchsky_pixel_to_world_xy']:.4f}ms  "
            f"ts_p2w_st={r['median_ms_torchsky_pixel_to_world_stack']:.4f}ms  "
            f"ts_p2w_st+c={r['median_ms_torchsky_pixel_to_world_stack_compiled']:.4f}ms  "
            f"ast_w2p={r['median_ms_astropy_all_world2pix']:.4f}ms  "
            f"ts_w2p_xy={r['median_ms_torchsky_world_to_pixel_xy']:.4f}ms  "
            f"ts_w2p_st={r['median_ms_torchsky_world_to_pixel_stack']:.4f}ms  "
            f"ts_w2p_st+c={r['median_ms_torchsky_world_to_pixel_stack_compiled']:.4f}ms  "
            f"|dra|max={r['max_abs_dra_deg']:.3e}°  |ddec|max={r['max_abs_ddec_deg']:.3e}°  "
            f"|dx|max={r['max_dx_pix']:.3e}px  |dy|max={r['max_dy_pix']:.3e}px  "
            f"compile={r['compile']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
