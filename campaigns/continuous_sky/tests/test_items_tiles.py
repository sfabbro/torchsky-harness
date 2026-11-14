"""Several sky patches (tiles) with the item sky: two tiles (synth.make_field_tiles), one ItemSky per tile, --sky-swap 1, registration
pre-pass (rebuilds every tile's warm start -> reseeds the items), a short fit, held-out fit, and saving / loading the per-tile sky states
(the cli.py format). Smoke test only (CPU, about a minute): it checks that the machinery runs and that state round-trips exactly.
Run:  python tests/test_items_tiles.py
"""
import io
import os
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fieldop.cli import default_config, make_sky  # noqa: E402
from fieldop.model import FieldOperator  # noqa: E402
from fieldop.sky import ItemSky  # noqa: E402
from fieldop.synth import make_field_tiles  # noqa: E402
from fieldop.tiles import TileSet  # noqa: E402
from fieldop.train import fit, fit_heldout, register_prepass, warm_kernel  # noqa: E402


def test_items_two_tiles():
    torch.manual_seed(0)
    cuts, fset, meta, truth, insts, tiles, half_data = make_field_tiles(seed=0, n_wide=4, n_side=2)
    cfg = default_config(sky_pitch=0.06, render_pitch=0.02, level_nh=10, kernel_extent=1.0, region_px=13, n_lambda=3, k_hidden=16,
                         warm_kernel_steps=20, pattern_grid=0, stripes=0, batch_patches=3, inner_sky=1, budget_min=1e6, steps=12,
                         tau_mode="fixed", held_steps=6, held_reg_steps=4, log_every=4, reg_prepass_steps=4, reg_prepass_iters=1,
                         data_margin=0.3, sky_margin=0.6, sky_swap=1, visit_steps=3, sky_form="items", item_count=300, item_smooth_pitch=0.3,
                         item_relocate=1)
    half = half_data - cfg.data_margin + cfg.sky_margin
    skies = {}
    for k in sorted(tiles):
        cs = [e for e in cuts if e.tile == k]
        for e in cs:
            e.to_dev("cpu")
        skies[k], N = make_sky(cs, fset, cfg, half, "cpu")
        for e in cs:
            e.release()
    assert all(isinstance(s, ItemSky) for s in skies.values())
    model = FieldOperator(cfg, fset, meta, skies[0], torch.device("cpu"))
    ts = TileSet(model, cuts, skies, cfg, log=lambda s: None)
    finfo = {f: {"pixel_arcsec": 0.10 if f < 2 else 0.12, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
    warm_kernel(model, cfg, finfo, log=lambda s: None)
    register_prepass(model, cuts, cfg, log=print, ts=ts)
    rec = fit(model, cuts, cfg, log=print, ts=ts)
    held = [c for c, e in enumerate(cuts) if e.split == "held"]
    fit_heldout(model, cuts, held, cfg, log=print, ts=ts)
    assert rec["n_steps"] == 12
    # save / load (as cli.py: per-tile state dicts on the host)
    st = {k: {kk: vv.detach().cpu() for kk, vv in s_.state_dict().items()} for k, s_ in skies.items()}
    buf = io.BytesIO()
    torch.save(st, buf)
    buf.seek(0)
    st2 = torch.load(buf)
    for k, s_ in skies.items():
        s_.to("cpu")
        A1 = s_.coeffs().detach().clone()
        fresh = ItemSky(torch.zeros_like(A1), s_.pitch, s_.basis, s_.n_items, n_levels=s_.n_levels, smooth_pitch=s_.smooth_pitch)
        fresh.register_buffer("sig_b", torch.zeros_like(s_.sig_b))
        fresh.load_state_dict(st2[k])
        A2 = fresh.coeffs().detach()
        err = float((A1 - A2).abs().max())
        print(f"tile {k}: {s_.n_items} items, {s_.n_fitted()} fitted numbers; reload max |dA| = {err:.2e}; visits {ts.stats[k]}")
        assert err == 0.0
    print("relocations:", rec.get("relocated"))


if __name__ == "__main__":
    test_items_two_tiles()
