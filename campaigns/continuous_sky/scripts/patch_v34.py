"""v33 -> v34 (6 Nov 2026): experiment E1 of psf-geometry-audit.md, sky halo.
Problem (audit #4): data pixels were kept in a ring of 24 target px OUTSIDE the size x size sky grid; there the model sampled
the convolved sky through bilinear zero padding, i.e. predicted background only (about 16% of a fully covering exposure).
--halo-px H (default 0 = v33 behaviour, bit-identical): the sky grid becomes (size + 2H)^2 with the same centre; the data
selection is unchanged (pixels within the ORIGINAL box +- 24 px, i.e. pad = 24 - H in the enlarged grid), so every kept pixel
lies at least H - 24 px inside the sky grid. Use H >= 24 + psf_n // 2 so that the kernel never reads beyond the grid.
The bright-pixel (S/N > 5) masks are zeroed outside the original box, so the source-pixel metric uses the same region as v33.
Sky pixels further than ~(H - 24 - psf_n/2) px outside the original box are unconstrained (no data reach them); they only
exist so that sources just outside the box can be modelled by the pixels that see their wings."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


rep('''    ap.add_argument("--load-dir", default=None,''',
    '''    ap.add_argument("--halo-px", type=int, default=0, help="v34: enlarge the sky grid by this many target px on every side; data selection unchanged")
    ap.add_argument("--load-dir", default=None,''')

rep('''    thead = target_header(a.ra, a.dec, a.size, a.scale)
    LOG["target"] = thead

    pad_ = int(round(24 * 0.04 / a.scale))
    exps = load_all_exposures(a, thead, dev, pad_)
    counts = {b: sum(1 for e in exps if e["band"] == b and e["d"].numel() > 0.5 * (a.size * a.scale / 0.031) ** 2 * (0.031 / 0.063) ** 2) for b in NIRCAM_BANDS}''',
    '''    a.size_data = a.size  # v34: the data box; the sky grid adds a halo of a.halo_px on every side
    if getattr(a, "halo_px", 0) > 0:
        a.size = a.size + 2 * a.halo_px
        LOG["halo"] = {"halo_px": a.halo_px, "size_data": a.size_data, "size_sky": a.size}
    thead = target_header(a.ra, a.dec, a.size, a.scale)
    LOG["target"] = thead

    pad_ = int(round(24 * 0.04 / a.scale)) - getattr(a, "halo_px", 0)
    exps = load_all_exposures(a, thead, dev, pad_)
    counts = {b: sum(1 for e in exps if e["band"] == b and e["d"].numel() > 0.5 * (a.size_data * a.scale / 0.031) ** 2 * (0.031 / 0.063) ** 2) for b in NIRCAM_BANDS}''')

rep('''    src_masks = torch.stack(src_masks)
''', '''    src_masks = torch.stack(src_masks)
    if getattr(a, "halo_px", 0) > 0:  # v34: bright-pixel metric on the original box only (same region as v33)
        hp = a.halo_px
        inner = torch.zeros_like(src_masks[0])
        inner[hp:hp + a.size_data, hp:hp + a.size_data] = True
        src_masks = src_masks & inner
''')
open(dst, "w").write(s)
print("ok", dst)
