"""Default path unchanged with several tiles: the two-tile synthetic field (synth.make_field_tiles), grid sky, --sky-swap 1, registration
pre-pass, 12 steps, held-out fit; prints a checksum of the final model and sky states. Run with the untouched and the modified package:
    python tests/tiles_grid_equiv.py untouched     python tests/tiles_grid_equiv.py ."""
import hashlib
import os
import sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.abspath(sys.argv[1]))
from fieldop.cli import default_config, make_sky
from fieldop.model import FieldOperator
from fieldop.synth import make_field_tiles
from fieldop.tiles import TileSet
from fieldop.train import fit, fit_heldout, register_prepass, warm_kernel
torch.manual_seed(0)
cuts, fset, meta, truth, insts, tiles, half_data = make_field_tiles(seed=0, n_wide=4, n_side=2)
cfg = default_config(sky_pitch=0.06, render_pitch=0.02, level_nh=10, kernel_extent=1.0, region_px=13, n_lambda=3, k_hidden=16, warm_kernel_steps=20,
                     pattern_grid=0, stripes=0, batch_patches=3, inner_sky=1, budget_min=1e6, steps=12, tau_mode="fixed", held_steps=6, held_reg_steps=4,
                     log_every=4, reg_prepass_steps=4, reg_prepass_iters=1, data_margin=0.3, sky_margin=0.6, sky_swap=1, visit_steps=3)
half = half_data - cfg.data_margin + cfg.sky_margin
skies = {}
for k in sorted(tiles):
    cs = [e for e in cuts if e.tile == k]
    for e in cs: e.to_dev("cpu")
    skies[k], N = make_sky(cs, fset, cfg, half, "cpu")
    for e in cs: e.release()
model = FieldOperator(cfg, fset, meta, skies[0], torch.device("cpu"))
ts = TileSet(model, cuts, skies, cfg, log=lambda s: None)
finfo = {f: {"pixel_arcsec": 0.10 if f < 2 else 0.12, "diameter_m": 6.5, "inst": 0 if f < 2 else 1} for f in range(len(fset))}
warm_kernel(model, cfg, finfo, log=lambda s: None)
register_prepass(model, cuts, cfg, log=lambda s: None, ts=ts)
rec = fit(model, cuts, cfg, log=lambda s: None, ts=ts)
fit_heldout(model, cuts, [c for c, e in enumerate(cuts) if e.split == "held"], cfg, log=lambda s: None, ts=ts)
h = hashlib.sha256()
st = {k: v for k, v in model.state_dict().items() if not k.startswith("sky.")}
for k in sorted(st): h.update(k.encode()); h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
for t in sorted(skies):
    for k, v in sorted(skies[t].state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
print(sys.argv[1], "losses", [round(s[2], 6) for s in rec["steps"]], "state sha256", h.hexdigest())
