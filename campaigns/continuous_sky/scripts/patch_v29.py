"""v28 -> v29: JWST/NIRCam + HST in one forward model (no pre-registration of exposures).

Principle: every exposure (NIRCam cal chip, HST flt/flc chip) stays in its own detector frame; its header WCS (HST: SIP + D2IM +
NPOL/CPDIS lookup tables via hst_io) gives only the starting map (tx, ty); the fit learns the per-exposure transforms (Geometry),
a learned PSF and one shared sky s(alpha, delta, lambda), and predicts every exposure's pixels. Data are MJy/sr per nominal pixel
for both telescopes, so the model has no pixel-area factor.

Changes (each one is an exact-string replacement with a count assert; with no --hst-* flag the run is the v28 run):
  1. band registry (BandRegistry, REG) replaces BANDS / FWHM_ARCSEC / LAM_UM / CHANNEL; every use goes through REG.
  2. --lam-lo / --lam-hi (spectral domain of the sky field).
  3. ChromaticPSF: lambda/D from the band's telescope diameter; channel one-hot over the registry's channels.
  4. --hst-dir/--hst-manifest/--hst-programs/--hst-instruments/--hst-filters/--hst-mjd-min/--hst-mjd-max/--hst-dq-policy.
  5. classical(): HST bands use an inverse-variance weighted tent/bilinear splat of the loaded pixels (approximation).
  6. 1/f stripes only for NIRCam exposures; the detection image uses the NIRCam SW bands by name.
  7. --holdout-from together with --n-test N; HST names parse (a dither = one exposure = all chips of one rootname).
  8. per-instrument roll-up of the held-out scores (TEST / BOOT lines, LOG), only printed when more than one instrument is present.
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old))
    s = s.replace(old, new)


def rep_between(start, end, new):
    """Replace s[start:end) (start and end each occur once; end is searched after start)."""
    global s
    assert s.count(start) == 1, (start[:70], s.count(start))
    i = s.index(start)
    j = s.index(end, i)
    assert s.count(end) >= 1
    s = s[:i] + new + s[j:]


# ------------------------------------------------------------------------------------------------ 0. docstring
rep('''Outputs (OUT): cubes/*.fits, rgb/*.png, figure PNGs, metrics.json
"""''', '''Outputs (OUT): cubes/*.fits, rgb/*.png, figure PNGs, metrics.json

v29: HST (WFC3/IR, ACS/WFC, WFC3/UVIS flt/flc) joins JWST/NIRCam in the same fit. Exposures are NOT registered: each stays in its
detector frame, the header WCS is the starting map only, the learned Geometry/PSF/sky do the rest (see patch_v29.py).
"""''')

# ------------------------------------------------------------------------------------------------ 1. band registry
rep('''BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
FWHM_ARCSEC = {"F090W": 0.030, "F150W": 0.049, "F200W": 0.064, "F277W": 0.091, "F356W": 0.115, "F444W": 0.145}
LAM_UM = {"F090W": 0.902, "F150W": 1.501, "F200W": 1.990, "F277W": 2.786, "F356W": 3.563, "F444W": 4.421}  # pivot
CHANNEL = {"F090W": 0, "F150W": 0, "F200W": 0, "F277W": 1, "F356W": 1, "F444W": 1}  # NIRCam SW / LW
''', '''# ----------------------------------------------------------------------------- v29: band registry
# v28 kept four NIRCam-only module tables (BANDS, FWHM_ARCSEC, LAM_UM, CHANNEL). They are now the NIRCam rows of a registry built from
# the data that were loaded (NIRCam bands as before, HST bands from hst_io), read through REG everywhere.
NIRCAM_BANDS = ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
NIRCAM_FWHM_ARCSEC = {"F090W": 0.030, "F150W": 0.049, "F200W": 0.064, "F277W": 0.091, "F356W": 0.115, "F444W": 0.145}
NIRCAM_LAM_UM = {"F090W": 0.902, "F150W": 1.501, "F200W": 1.990, "F277W": 2.786, "F356W": 3.563, "F444W": 4.421}  # pivot
NIRCAM_CHANNEL = {"F090W": "NIRCAM_SW", "F150W": "NIRCAM_SW", "F200W": "NIRCAM_SW",
                  "F277W": "NIRCAM_LW", "F356W": "NIRCAM_LW", "F444W": "NIRCAM_LW"}
CHANNELS = ["NIRCAM_SW", "NIRCAM_LW", "WFC3_IR", "ACS_WFC", "WFC3_UVIS"]  # canonical order = one-hot order (NIRCam first)
CHANNEL_DIAMETER_M = {"NIRCAM_SW": 6.5, "NIRCAM_LW": 6.5, "WFC3_IR": 2.4, "ACS_WFC": 2.4, "WFC3_UVIS": 2.4}
HST_CHANNEL = {"WFC3/IR": "WFC3_IR", "ACS/WFC": "ACS_WFC", "WFC3/UVIS": "WFC3_UVIS"}
LOD_COEF = 0.0317  # lambda/D [arcsec per um] for D = 6.5 m (v28 value); scaled by 6.5 / D for other telescopes
D_REF_M = 6.5


def nominal_fwhm_arcsec(lam_um, diameter_m, pixel_arcsec):
    """Starting FWHM for an HST band (NIRCam keeps its table): Airy FWHM 1.028 lambda/D added in quadrature to half the detector
    pixel. A starting value for the Gaussian PSF core / rigid model only (WFC3/IR F160W: 0.156 arcsec vs 0.15 published)."""
    lod = 0.206265 * lam_um / diameter_m
    return round(math.hypot(1.028 * lod, 0.5 * pixel_arcsec), 4)


class BandRegistry:
    """Per band: pivot lambda [um], telescope diameter [m], nominal FWHM [arcsec], channel (str) and its id in self.channels, SVO
    filter id (for load_bandpasses), target-grid PSF kernel size [px], instrument label (roll-up of the scores) and facility.
    Band order: NIRCam bands (canonical order) first, then HST bands sorted by wavelength. self.channels is the one-hot basis of the
    learned PSF: the channels in use, in canonical order; both NIRCam channels whenever any NIRCam band is present (the v28 shapes)."""

    def __init__(self, entries):
        self.entries = [dict(e) for e in entries]
        self.bands = [e["band"] for e in self.entries]
        assert len(set(self.bands)) == len(self.bands), self.bands
        self.index = {b: k for k, b in enumerate(self.bands)}
        col = lambda key: {e["band"]: e[key] for e in self.entries}
        self.lam_um, self.fwhm_arcsec, self.channel = col("lam_um"), col("fwhm_arcsec"), col("channel")
        self.diameter_m, self.svo_id, self.kernel_px = col("diameter_m"), col("svo_id"), col("kernel_px")
        self.instrument, self.facility = col("instrument"), col("facility")
        used = set(self.channel.values())
        if any(c.startswith("NIRCAM") for c in used):
            used |= {"NIRCAM_SW", "NIRCAM_LW"}
        self.channels = [c for c in CHANNELS if c in used]
        self.chan_id = {b: self.channels.index(c) for b, c in self.channel.items()}
        self.is_nircam = {b: c.startswith("NIRCAM") for b, c in self.channel.items()}
        self.is_hst = {b: not v for b, v in self.is_nircam.items()}
        self.instruments = list(dict.fromkeys(self.instrument[b] for b in self.bands))
        self.multi = len(self.instruments) > 1
        self.psf_n = max(self.kernel_px.values()) if self.kernel_px else 31

    def summary(self):
        return [{"band": b, "instrument": self.instrument[b], "channel": self.channel[b], "chan_id": self.chan_id[b],
                 "lam_um": self.lam_um[b], "diameter_m": self.diameter_m[b], "fwhm_arcsec": self.fwhm_arcsec[b],
                 "svo_id": self.svo_id[b], "kernel_px": self.kernel_px[b]} for b in self.bands]


def _nircam_entry(b, psf_n):
    ch = NIRCAM_CHANNEL[b]
    return {"band": b, "lam_um": NIRCAM_LAM_UM[b], "diameter_m": CHANNEL_DIAMETER_M[ch], "fwhm_arcsec": NIRCAM_FWHM_ARCSEC[b],
            "channel": ch, "svo_id": f"JWST/NIRCam.{b}", "kernel_px": psf_n, "instrument": "NIRCam", "facility": "JWST"}


def _hst_entry(info, psf_n):
    """info: band, instrument ("WFC3/IR", "ACS/WFC", "WFC3/UVIS"), filter, photplam [Angstrom], svo_id (None for a clear filter)."""
    import hst_io  # the same directory as this script
    bi = hst_io.hst_band_info(info["instrument"], info["filter"])
    if info.get("photplam") in (None, 0):
        raise ValueError(f"HST band {info['band']}: no PHOTPLAM in the headers, cannot place it in wavelength")
    lam = float(info["photplam"]) / 1e4
    return {"band": info["band"], "lam_um": lam, "diameter_m": bi["diameter_m"],
            "fwhm_arcsec": nominal_fwhm_arcsec(lam, bi["diameter_m"], bi["pixel_scale_arcsec"]),
            "channel": HST_CHANNEL[bi["instrument"]], "svo_id": info.get("svo_id") or bi["svo_id"], "kernel_px": psf_n,
            "instrument": bi["instrument"], "facility": "HST"}


def build_registry(nircam_bands, hst_infos=(), psf_n=31):
    """nircam_bands: names (any subset of NIRCAM_BANDS, put in canonical order); hst_infos: list of dicts for _hst_entry."""
    nir = [b for b in NIRCAM_BANDS if b in set(nircam_bands)]
    hst = sorted((_hst_entry(i, psf_n) for i in hst_infos), key=lambda e: (e["lam_um"], e["band"]))
    return BandRegistry([_nircam_entry(b, psf_n) for b in nir] + hst)


def registry_from_exposures(exps, psf_n=31):
    """Registry of the bands present in a loaded exposure list (HST exposures carry instrument/filter/photplam/svo_id)."""
    nir = {e["band"] for e in exps if not e.get("instrument") and e["band"] in NIRCAM_BANDS}
    by = {}
    for e in exps:
        if e.get("instrument"):
            by.setdefault(e["band"], []).append(e)
    infos = []
    for b, es in by.items():
        pl = [float(e["photplam"]) for e in es if e.get("photplam")]
        infos.append({"band": b, "instrument": es[0]["instrument"], "filter": es[0]["filter"],
                      "photplam": float(np.median(pl)) if pl else None, "svo_id": es[0].get("svo_id")})
    return build_registry(sorted(nir), infos, psf_n)


REG = build_registry(NIRCAM_BANDS)  # default = the v28 NIRCam registry; run() replaces it by the registry of the loaded data


def set_registry(reg):
    global REG
    REG = reg
    return reg


def uses_stripes(band):
    """1/f row/column stripes are a NIRCam readout effect: NIRCam exposures only."""
    return REG.is_nircam.get(band, True)


def detection_bands(reg=None):
    """Indices of the bands of the detection image (aperture catalogue, splat seeds): the NIRCam SW bands, by name; if there are
    none (HST-only), the three shortest-wavelength bands."""
    reg = reg or REG
    sw = [k for k, b in enumerate(reg.bands) if reg.channel[b] == "NIRCAM_SW"]
    if sw:
        return sw
    return sorted(sorted(range(len(reg.bands)), key=lambda k: reg.lam_um[reg.bands[k]])[:3])


def require_nircam_only(what):
    hst = [b for b in REG.bands if REG.is_hst[b]]
    if hst:
        raise SystemExit(f"{what} is NIRCam-only (6.5 m pupil model) and cannot be used with HST bands present: {hst}")
''')

rep('''LOG: dict = {"timings": {}, "metrics": {}}''', '''LOG: dict = {"timings": {}, "metrics": {}}
ARCSEC2_TO_SR = (math.pi / 648000.0) ** 2''')

# ------------------------------------------------------------------------------------------------ exposure_band
rep('''    if f in BANDS:
        return f
    if p in BANDS:
        return p
    return None''', '''    if f in NIRCAM_BANDS:
        return f
    if p in NIRCAM_BANDS:
        return p
    return None''')

# ------------------------------------------------------------------------------------------------ 4. loading, 5. splat coadd
rep('''# ----------------------------------------------------------------------------- classical
def classical(exps, thead, kernel, pixfrac, device):''', r'''# ----------------------------------------------------------------------------- v29: HST selection / loading
def _csv_list(v):
    return [x.strip() for x in str(v).split(",") if x.strip()] if v else []


def _norm_inst(x):
    return x.strip().upper().replace("_", "/").replace(" ", "")


def parse_dq_policy(v):
    v = str(v).strip().lower()
    return int(v) if v.lstrip("-").isdigit() else v


def select_hst_files(hst_dir, manifest=None, programs=None, instruments=None, filters=None, mjd_min=None, mjd_max=None):
    """HST flt/flc files to load. With a manifest (CSV like hst_imaging_unique.csv: proposal_id, instrument, detector, filter,
    obs_id, product_filename, product_type, t_start_mjd) the selection is made on the table; without, on the primary headers of
    *_flt.fits / *_flc.fits in hst_dir. programs/instruments/filters: comma lists ("WFC3/IR,ACS/WFC"); mjd_min/max: EXPSTART bounds.
    Only WFC3/IR, ACS/WFC, WFC3/UVIS are kept (the loader supports no others); when both flt and flc exist for one exposure the flc
    (CTE-corrected) is used. Returns (files, info)."""
    import csv

    import hst_io
    progs = {int(x) for x in _csv_list(programs)} or None
    insts = {_norm_inst(x) for x in _csv_list(instruments)} or None
    filts = {x.upper() for x in _csv_list(filters)} or None
    rows = []
    if manifest:
        with open(manifest, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({"path": os.path.join(hst_dir, r["product_filename"]), "inst": f"{r['instrument']}/{r['detector']}".upper(),
                             "filt": r["filter"].strip().upper(), "prog": int(r["proposal_id"]), "mjd": float(r["t_start_mjd"])})
    else:
        for f in sorted(glob.glob(os.path.join(hst_dir, "*_fl[tc].fits"))):
            h0 = fits.getheader(f, 0)
            rows.append({"path": f, "inst": hst_io._identify(h0), "filt": hst_io._filter_name(h0),
                         "prog": int(h0.get("PROPOSID", -1) or -1), "mjd": float(h0.get("EXPSTART", float("nan")))})
    n0 = len(rows)
    ok = [r for r in rows if r["inst"] in hst_io.DQ_GOOD_BITS and r["path"].endswith(("_flt.fits", "_flc.fits"))]
    n_unsupported = len(rows) - len(ok)
    ok = [r for r in ok if (progs is None or r["prog"] in progs) and (insts is None or r["inst"] in insts)
          and (filts is None or r["filt"] in filts) and (mjd_min is None or r["mjd"] >= mjd_min) and (mjd_max is None or r["mjd"] <= mjd_max)]
    best = {}
    for r in ok:  # one file per exposure (root), flc preferred over flt
        root = os.path.basename(r["path"]).split("_")[0]
        if root not in best or r["path"].endswith("_flc.fits"):
            best[root] = r
    sel = sorted(best.values(), key=lambda r: (r["filt"], os.path.basename(r["path"])))
    missing = [r["path"] for r in sel if not os.path.exists(r["path"])]
    files = [r["path"] for r in sel if os.path.exists(r["path"])]
    return files, {"n_rows": n0, "n_unsupported_or_other_product": n_unsupported, "n_selected": len(sel), "n_missing_files": len(missing),
                   "missing_files": [os.path.basename(p) for p in missing][:50], "n_files": len(files)}


def load_all_exposures(a, thead, dev, pad):
    """JWST exactly as v28, plus HST through hst_io when --hst-dir is given; one merged list (JWST first, HST by wavelength, name)."""
    exps = load_exposures(a.data_dir, thead, dev, pad=pad)
    if not getattr(a, "hst_dir", None):
        if any(getattr(a, k, None) for k in ("hst_manifest", "hst_programs", "hst_instruments", "hst_filters", "hst_mjd_min", "hst_mjd_max")):
            raise SystemExit("--hst-manifest/--hst-programs/--hst-instruments/--hst-filters/--hst-mjd-* need --hst-dir")
        return exps
    import hst_io
    files, info = select_hst_files(a.hst_dir, a.hst_manifest, a.hst_programs, a.hst_instruments, a.hst_filters, a.hst_mjd_min, a.hst_mjd_max)
    LOG["hst_selection"] = info
    if info["n_missing_files"]:
        tick(f"HST: {info['n_missing_files']} selected files not found in {a.hst_dir} (skipped): {info['missing_files'][:5]}...")
    hexps = hst_io.load_hst_exposures(files, thead, dev, pad=pad, dq_policy=parse_dq_policy(a.hst_dq_policy), calibrate_noise=CALIBRATE_NOISE)
    hexps.sort(key=lambda e: (e["photplam"] or 0.0, e["band"], e["name"]))
    perb = {}
    for e in hexps:
        perb[e["band"]] = perb.get(e["band"], 0) + 1
    LOG["hst_selection"]["loaded_chips_per_band"] = perb
    tick(f"HST: {len(files)} files selected, {len(hexps)} chips overlap the target grid; per band {perb}")
    return exps + hexps


def splat_coadd(exps, thead, device, radius=None):
    """APPROXIMATE classical coadd for HST bands (torchsky resampling from file headers does not know the HST lookup-table
    distortions): inverse-variance weighted splat of the loaded pixels (tx, ty, d - bkg0, ivar), which already carry the full header
    WCS, onto the target grid. Each pixel is spread with a tent (bilinear) kernel of radius h = max(1, input pixel / target pixel)
    target pixels: h = 1 is plain bilinear splatting (2 x 2 target pixels); for an undersampled detector (WFC3/IR 0.128 arcsec on a
    0.04 arcsec grid, h = 3.2) a plain bilinear splat leaves most target pixels unfilled, so the tent follows the input footprint.
    No flux/area conservation (surface brightness in, surface brightness out), no PSF matching, no outlier rejection.
    Returns (image with NaN where nothing landed, summed weight)."""
    H, W = int(thead["NAXIS2"]), int(thead["NAXIS1"])
    s_out = abs(float(thead["CD2_2"])) * 3600.0
    num = torch.zeros(H * W, dtype=torch.float64, device=device)
    den = torch.zeros_like(num)
    for e in exps:
        s_in = math.sqrt(float(e["pixar_sr_nominal"]) / ARCSEC2_TO_SR) if e.get("pixar_sr_nominal") else s_out
        h = max(1.0, s_in / s_out) if radius is None else float(radius)  # radius: test override (1.0 = plain bilinear)
        tx, ty = e["tx"].double().to(device), e["ty"].double().to(device)
        iv, val = e["ivar"].double().to(device), (e["d"] - e["bkg0"]).double().to(device)
        ok = torch.isfinite(val) & torch.isfinite(iv) & (iv > 0) & torch.isfinite(tx) & torch.isfinite(ty)
        tx, ty, iv, val = tx[ok], ty[ok], iv[ok], val[ok]
        x0, y0 = torch.floor(tx), torch.floor(ty)
        m = int(math.ceil(h)) - 1
        for oy in range(-m, m + 2):
            iy = y0 + oy
            wy = (1.0 - (iy - ty).abs() / h).clamp(min=0.0)
            for ox in range(-m, m + 2):
                ix = x0 + ox
                w = wy * (1.0 - (ix - tx).abs() / h).clamp(min=0.0)
                inside = (w > 0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                if not bool(inside.any()):
                    continue
                lin = (iy[inside] * W + ix[inside]).long()
                wi = w[inside] * iv[inside]
                num.index_add_(0, lin, wi * val[inside])
                den.index_add_(0, lin, wi)
    return (num / den).reshape(H, W).float(), den.reshape(H, W).float()


def band_median_sigma(c, cover=None):
    """(median, 1.4826 MAD) of a band image: all pixels (NIRCam, as v28) or the covered pixels only (HST splat coadds)."""
    v = c if cover is None or not bool(cover.any()) else c[cover]
    m = v.median()
    return m, 1.4826 * (v - m).abs().median()


# ----------------------------------------------------------------------------- classical
def classical(exps, thead, kernel, pixfrac, device):''')

rep('''    out = {}
    for b in BANDS:
        obs = []
''', '''    out, cover = {}, {}
    for b in REG.bands:
        if REG.is_hst[b]:  # v29: approximate splat coadd from the loaded pixels (SWarp-like and Drizzle-like are the same image)
            img, wsum = splat_coadd([e for e in exps if e["band"] == b and not e["holdout"] and not e.get("val")], thead, device)
            cover[b] = (wsum > 0).to(device)
            out[b] = torch.nan_to_num(img).to(device)
            continue
        cover[b] = None
        obs = []
''')
rep('''        out[b] = torch.nan_to_num(res.data.float()).to(device)
    return torch.stack([out[b] for b in BANDS])''', '''        out[b] = torch.nan_to_num(res.data.float()).to(device)
    classical.last_cover = cover
    return torch.stack([out[b] for b in REG.bands])''')

# ------------------------------------------------------------------------------------------------ 3. ChromaticPSF
rep('''    detector. No per-band free kernels: bands constrain each other through lambda."""

    def __init__(self, bands, ndet, scale_arcsec, nk=4, n=31, hidden=96):
        super().__init__()
        self.n, self.nk, self.nb = n, nk, len(bands)
        lam = torch.tensor([LAM_UM[b] for b in bands])
        self.register_buffer("loglam", (torch.log(lam) - math.log(2.0)) / 0.8)
        self.register_buffer("chan", torch.nn.functional.one_hot(torch.tensor([CHANNEL[b] for b in bands]), 2).float())
        lod = 0.0317 * lam  # lambda/D in arcsec for D = 6.5 m
        self.scale_arcsec = scale_arcsec
        self.register_buffer("chan_idx", torch.tensor([CHANNEL[b] for b in bands]))
        self.register_buffer("pix_per_lod", scale_arcsec / lod)
        self.kern = nn.Sequential(nn.Linear(2 * 2 * 6 + 2 * 2 * 4 + 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nk))
        self.det_emb = nn.Embedding(ndet, 4)
        self.wnet = nn.Sequential(nn.Linear(2 * 2 * 3 + 3 + 4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, nk - 1))''',
    '''    detector. No per-band free kernels: bands constrain each other through lambda.
    v29: one network for all instruments. lambda/D uses the band's telescope diameter (6.5 m NIRCam, 2.4 m HST; the per-channel diameter
    is a non-persistent buffer so the state dict keeps the v28 layout), the channel one-hot runs over the registry's channels
    (NIRCam SW/LW, WFC3/IR, ACS/WFC, WFC3/UVIS), and the conditioning is (log lambda, channel one-hot)."""

    def __init__(self, bands, ndet, scale_arcsec, nk=4, n=31, hidden=96, reg=None):
        super().__init__()
        reg = reg or REG
        self.n, self.nk, self.nb = n, nk, len(bands)
        self.nchan = len(reg.channels)
        lam = torch.tensor([reg.lam_um[b] for b in bands])
        cid = torch.tensor([reg.chan_id[b] for b in bands])
        self.register_buffer("loglam", (torch.log(lam) - math.log(2.0)) / 0.8)
        self.register_buffer("chan", torch.nn.functional.one_hot(cid, self.nchan).float())
        self.register_buffer("chan_D", torch.tensor([CHANNEL_DIAMETER_M[c] for c in reg.channels]), persistent=False)
        lod = LOD_COEF * lam * (D_REF_M / self.chan_D[cid])  # lambda/D in arcsec (exactly 0.0317 lambda for D = 6.5 m)
        self.scale_arcsec = scale_arcsec
        self.register_buffer("chan_idx", cid.clone())
        self.register_buffer("pix_per_lod", scale_arcsec / lod)
        self.kern = nn.Sequential(nn.Linear(2 * 2 * 6 + 2 * 2 * 4 + 1 + self.nchan, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, nk))
        self.det_emb = nn.Embedding(ndet, 4)
        self.wnet = nn.Sequential(nn.Linear(2 * 2 * 3 + 1 + self.nchan + 4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, nk - 1))''')
rep('''        """Basis kernels at arbitrary wavelengths (um) and NIRCam channels (0 = SW, 1 = LW): (M, nk, n, n)."""
        lod = 0.0317 * lam_um
        loglam = (torch.log(lam_um) - math.log(2.0)) / 0.8
        chan = torch.nn.functional.one_hot(chan_idx.long(), 2).float()''', '''        """Basis kernels at arbitrary wavelengths (um) and registry channel ids: (M, nk, n, n)."""
        lod = LOD_COEF * lam_um * (D_REF_M / self.chan_D[chan_idx.long()])
        loglam = (torch.log(lam_um) - math.log(2.0)) / 0.8
        chan = torch.nn.functional.one_hot(chan_idx.long(), self.nchan).float()''')
rep('''            cond = torch.cat([loglam[b].view(1, 1), chan[b].view(1, 2)], 1).expand(P, 3)''',
    '''            cond = torch.cat([loglam[b].view(1, 1), chan[b].view(1, -1)], 1).expand(P, 1 + self.nchan)''')
rep('''        cond = torch.cat([self.loglam[b].view(1, 1), self.chan[b].view(1, 2)], 1).expand(N, 3)''',
    '''        cond = torch.cat([self.loglam[b].view(1, 1), self.chan[b].view(1, -1)], 1).expand(N, 1 + self.nchan)''')

# ------------------------------------------------------------------------------------------------ 6. stripes: NIRCam only
rep('''        if self.stripes is not None:
            m = m + self.stripes(vec(i, e["dx"]), e["row"], e["col"])
        return m''', '''        if self.stripes is not None and uses_stripes(e["band"]):  # v29: 1/f stripes are a NIRCam effect
            m = m + self.stripes(vec(i, e["dx"]), e["row"], e["col"])
        return m''')
rep('''        if self.stripes is not None:
            m = m + self.stripes(vec(i, dx_), e["row"][idx], e["col"][idx])
        return m''', '''        if self.stripes is not None and uses_stripes(e["band"]):
            m = m + self.stripes(vec(i, dx_), e["row"][idx], e["col"][idx])
        return m''')
rep('''            if getattr(args_global, "stripes", 0):  # same 1/f treatment for every method: row/col weighted means''',
    '''            if getattr(args_global, "stripes", 0) and uses_stripes(e["band"]):  # same 1/f treatment for every method: row/col weighted means (NIRCam)''')

# ------------------------------------------------------------------------------------------------ band loops
rep('''    for b in BANDS:
        sel = [i for i in idx if exps[i]["band"] == b]''', '''    for b in REG.bands:
        sel = [i for i in idx if exps[i]["band"] == b]''')
rep('''        planes = []
        for b in BANDS:''', '''        planes = []
        for b in REG.bands:''')
rep('''        if len(planes) == len(BANDS):''', '''        if len(planes) == len(REG.bands):''')

# ------------------------------------------------------------------------------------------------ 8. per-instrument scores
rep('''    per_band = {}
    resid = {}
''', '''    per_band = {}
    per_inst = {}  # v29: roll-up by instrument
    resid = {}
''')
rep('''            acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
''', '''            acc["chi2"] += float(chi.sum()); acc["robust"] += float(rh.sum()); acc["n"] += r.numel(); acc["n_exp"] += 1
            ia = per_inst.setdefault(REG.instrument.get(e["band"], "?"), {"chi2": 0.0, "robust": 0.0, "n": 0, "chi2_src": 0.0, "robust_src": 0.0, "n_src": 0, "n_exp": 0})
            ia["chi2"] += float(chi.sum()); ia["robust"] += float(rh.sum()); ia["n"] += r.numel(); ia["n_exp"] += 1
''')
rep('''                  "chi2_src_sum": 0.0, "robust_sum": float(rh.sum()), "robust_src_sum": 0.0, "pulls": []}
''', '''                  "chi2_src_sum": 0.0, "robust_sum": float(rh.sum()), "robust_src_sum": 0.0, "pulls": []}
            if REG.multi:
                pe["instrument"] = REG.instrument.get(e["band"], "?")
''')
rep('''                acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
''', '''                acc["chi2_src"] += float(chi[sm].sum()); acc["robust_src"] += float(rh[sm].sum()); acc["n_src"] += int(sm.sum())
                ia["chi2_src"] += float(chi[sm].sum()); ia["robust_src"] += float(rh[sm].sum()); ia["n_src"] += int(sm.sum())
''')
rep('''    for bnd, acc in per_band.items():  # per-band means over every scored exposure of that band''',
    '''    for bnd, acc in list(per_band.items()) + list(per_inst.items()):  # per-band (and per-instrument) means over every scored exposure''')
rep('''                         "per_band": per_band}''', '''                         "per_band": per_band, "per_instrument": per_inst}''')

rep('''def bootstrap_scores(per_exposure, B=1000, seed=0, cluster="dither"):''', '''def _record_instrument(r):
    return r.get("instrument") or REG.instrument.get(r["band"], "?")


def bootstrap_scores(per_exposure, B=1000, seed=0, cluster="dither"):''')
rep('''    samp = np.full((B, len(_BOOT_STATS)), np.nan)
    for k in range(B):''', '''    samp = np.full((B, len(_BOOT_STATS)), np.nan)
    insts = sorted({_record_instrument(r) for r in per_exposure})
    samp_i = np.full((B, len(insts), len(_BOOT_STATS)), np.nan) if len(insts) > 1 else None  # v29: per-instrument replicates
    for k in range(B):''')
rep('''        sc = _pooled_scores(recs, pulls)
        samp[k] = [sc[t] for t in _BOOT_STATS]
''', '''        sc = _pooled_scores(recs, pulls)
        samp[k] = [sc[t] for t in _BOOT_STATS]
        if samp_i is not None:
            for j_, ins in enumerate(insts):
                sel_ = [q_ for q_, r_ in enumerate(recs) if _record_instrument(r_) == ins]
                if sel_:
                    sci_ = _pooled_scores([recs[q_] for q_ in sel_], [pulls[q_] for q_ in sel_])
                    samp_i[k, j_] = [sci_[t] for t in _BOOT_STATS]
''')
rep('''    nimg = {b: sum(len(u) for u in by[b]) for b in bands}
''', '''    if samp_i is not None:
        res["per_instrument"] = {}
        for j_, ins in enumerate(insts):
            recs_i = [r for r in per_exposure if _record_instrument(r) == ins]
            est_i = _pooled_scores(recs_i, [pul[id(r)] for r in recs_i])
            d_ = {}
            for q_, t in enumerate(_BOOT_STATS):
                col = samp_i[:, j_, q_]
                col = col[np.isfinite(col)]
                p_ = np.percentile(col, [16, 84]) if col.size else [float("nan")] * 2
                d_[t] = {"est": est_i[t], "p16": float(p_[0]), "p84": float(p_[1])}
            d_["n_images"] = len(recs_i)
            res["per_instrument"][ins] = d_
    nimg = {b: sum(len(u) for u in by[b]) for b in bands}
''')
rep('''    msg += " | n_dithers " + ",".join(f"{b}:{n}" for b, n in bs["n_dithers"].items())
''', '''    msg += " | n_dithers " + ",".join(f"{b}:{n}" for b, n in bs["n_dithers"].items())
    for ins, d_ in bs.get("per_instrument", {}).items():  # v29
        msg += f" | {ins} (n={d_['n_images']}): src {d_['chi2_src']['est']:.4f} [{d_['chi2_src']['p16']:.4f}, {d_['chi2_src']['p84']:.4f}] all {d_['chi2_all']['est']:.4f} flux err {d_['flux_err']['est']:.4f}"
''')

# ------------------------------------------------------------------------------------------------ 7. names and selection
rep('''    m = _NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return None
    g = m.groupdict()''', '''    m = _NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return _parse_hst_name(name)
    g = m.groupdict()''')
rep('''def parse_exposure_name(name):''', r'''_HST_NAME_RE = _re.compile(r"^(?P<root>[a-z][a-z0-9]{8})_sci(?P<ext>\d+)$", _re.IGNORECASE)  # hst_io chip names: <rootname>_sci<ext>


def _parse_hst_name(name):
    """HST chip name <ipppssoot>_sci<N> (i instrument, ppp programme, ss visit, oo observation, t transmission). visit = instrument +
    programme + visit (root[:6]); dither = one exposure = all chips (sci1, sci2) of one rootname."""
    m = _HST_NAME_RE.match(os.path.basename(str(name)))
    if m is None:
        return None
    root = m.group("root").lower()
    visit = (root[:6], "hst")
    return {"prog": root[1:4], "obs": root[4:6], "vis": root[4:6], "grp": "hst", "exp": root[6:], "det": "sci" + m.group("ext"), "idx": None,
            "visit": visit, "dither": (visit, root[6:]), "root": root}


def hst_root(name):
    p = _parse_hst_name(name)
    return p["root"] if p else None


def read_holdout_names(path):
    """Names to hold out from a file: a metrics.json / holdout_selection.json ({"holdout": [...]}), a JSON list, or one name per line."""
    txt = open(path).read()
    try:
        obj = json.loads(txt)
    except ValueError:
        return [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return list(obj["holdout"]) if isinstance(obj, dict) else list(obj)


def parse_exposure_name(name):''')

rep('''def select_exposures(exps, n_test=1, n_val=0, holdout_names=None, no_holdout=False, bands=None):''',
    '''def select_exposures(exps, n_test=1, n_val=0, holdout_names=None, no_holdout=False, bands=None, fill_missing_bands=False):''')
rep('''    * validation (n_val rounds): v21 rule on what is left, one detector image per band and round, only while >= 4 images remain."""
    bands = BANDS if bands is None else bands''', '''    * validation (n_val rounds): v21 rule on what is left, one detector image per band and round, only while >= 4 images remain.
    * v29: holdout_names + fill_missing_bands: the names are held out (exact chip name, or an HST rootname = all its chips) and every
      band with no held-out exposure after that gets the n_test rule above (n_test == 1: v21 rule; > 1: that many dithers)."""
    bands = REG.bands if bands is None else bands''')
rep_between('''    if no_holdout:
        pass
    elif holdout_names is not None:''', '''    if not no_holdout:
        for _ in range(n_val):  # validation: early stopping / selection''', '''    if no_holdout:
        pass
    else:
        todo = bands
        if holdout_names is not None:
            names = set(holdout_names)
            for e in exps:
                e["holdout"] = e["name"] in names or hst_root(e["name"]) in names
            todo = [b for b in bands if fill_missing_bands and not any(e["holdout"] for e in exps if e["band"] == b)]
            for b in todo:
                warnings.append(f"{b}: no held-out exposure in the holdout file, selected by the n_test={n_test} rule")
        _auto_holdout(exps, todo, n_test, warnings)
''')
rep('''def select_exposures(exps,''', '''def _auto_holdout(exps, bands, n_test, warnings):
    """The v21 (n_test <= 1) / v22 (n_test > 1, dithers over visits) holdout rules on the given bands (unchanged from v28)."""
    if n_test <= 1:
        for b in bands:
            bl = [e for e in exps if e["band"] == b]
            if len(bl) >= 3:  # hold out the exposure covering the most of the field
                max(bl, key=lambda e: e["d"].numel())["holdout"] = True
        return
    for b in bands:
        bl = [e for e in exps if e["band"] == b]
        dith = {}
        for e in bl:
            dith.setdefault(dither_key(e["name"]), []).append(e)
        n_eff = min(n_test, len(dith) - 2)
        if n_eff < n_test:
            warnings.append(f"{b}: {len(dith)} dithers, holding out {max(n_eff, 0)} instead of {n_test}")
        if n_eff < 1:
            continue
        px = {k: sum(e["d"].numel() for e in v) for k, v in dith.items()}
        byvis = {}
        for k in dith:
            byvis.setdefault(k[0], []).append(k)
        for v in byvis.values():
            v.sort(key=lambda k: (-px[k], k))
        order = sorted(byvis, key=lambda v: (-px[byvis[v][0]], v))
        chosen = []
        while len(chosen) < n_eff:
            for v in order:
                if byvis[v] and len(chosen) < n_eff:
                    chosen.append(byvis[v].pop(0))
        for k in chosen:
            for e in dith[k]:
                e["holdout"] = True


def select_exposures(exps,''')

# ------------------------------------------------------------------------------------------------ rgb
rep('''def rgb_from_cube(cube, stretch, Q=8.0, norm=None):
    if norm is not None:
        cube = cube / norm[:, None, None]
    b = cube[0] + cube[1]
    g = cube[2] + cube[3]
    r = cube[4] + cube[5]''', '''def rgb_groups(n):
    """Three channel groups (blue, green, red) of band indices: the bands in wavelength order split in three contiguous parts
    ([0,1], [2,3], [4,5] for the six NIRCam bands, as v28)."""
    order = sorted(range(n), key=lambda k: REG.lam_um[REG.bands[k]]) if n == len(REG.bands) else list(range(n))
    if n < 3:
        return [[order[0]], [order[min(1, n - 1)]], [order[-1]]]
    return [list(map(int, ch)) for ch in np.array_split(np.array(order), 3)]


def _group_sum(cube, idx):  # sum of the group, rescaled to a two-band sum (factor 1.0 for two bands: v28 values)
    acc = cube[idx[0]]
    for j in idx[1:]:
        acc = acc + cube[j]
    return acc if len(idx) == 2 else acc * (2.0 / len(idx))


def rgb_from_cube(cube, stretch, Q=8.0, norm=None):
    if norm is not None:
        cube = cube / norm[:, None, None]
    gb, gg, gr = rgb_groups(cube.shape[0])
    b = _group_sum(cube, gb)
    g = _group_sum(cube, gg)
    r = _group_sum(cube, gr)''')

# ------------------------------------------------------------------------------------------------ bandpasses
rep('''def load_bandpasses(bands, q_nodes, cache_dir=None):''', '''def load_bandpasses(bands, q_nodes, cache_dir=None, reg=None):''')
rep('''    lam, wts, curves = [], [], {}
    for b in bands:
        spec = bandpass_from_svo(band_name=b, facility="JWST", instrument="NIRCam", filter_id=f"JWST/NIRCam.{b}",
                                 prefer="any", cache_dir=cache_dir)''', '''    reg = reg or REG
    lam, wts, curves = [], [], {}
    for b in bands:
        sid = reg.svo_id[b]  # NIRCam: JWST/NIRCam.<filter>; HST: the loader's id (HST/ACS_WFC.F814W, HST/WFC3_IR.F160W, ...)
        if not sid:
            raise ValueError(f"band {b} has no SVO filter id (clear filter?): cannot build its bandpass")
        fac, rest = sid.split("/", 1)
        ins, filt = rest.rsplit(".", 1)
        spec = bandpass_from_svo(band_name=filt, facility=fac, instrument=ins, filter_id=sid, prefer="any", cache_dir=cache_dir)''')

# ------------------------------------------------------------------------------------------------ pupil PSF (NIRCam only)
rep('''    from pupil_psf import PupilPSF
    det_chan = [1 if "LONG" in d.upper() else 0 for d in dets]
    psf = PupilPSF(BANDS, [LAM_UM[b] for b in BANDS], [CHANNEL[b] for b in BANDS], det_chan, a.scale, n=a.psf_n).to(dev)
    b0 = band_index[BANDS[0]]''', '''    from pupil_psf import PupilPSF
    require_nircam_only("the pupil PSF")
    det_chan = [1 if "LONG" in d.upper() else 0 for d in dets]
    psf = PupilPSF(REG.bands, [REG.lam_um[b] for b in REG.bands], [REG.chan_id[b] for b in REG.bands], det_chan, a.scale, n=a.psf_n).to(dev)
    b0 = band_index[REG.bands[0]]''')
rep('''"flip": best[2], "stars_yx": stars, "band": BANDS[b0]}''', '''"flip": best[2], "stars_yx": stars, "band": REG.bands[b0]}''')
rep('''from {len(stars)} stars in {BANDS[b0]}; "''', '''from {len(stars)} stars in {REG.bands[b0]}; "''')

# ------------------------------------------------------------------------------------------------ capacity control
rep('''                per_band = {BANDS[j]: {"all": float(rb[j].pow(2).mean()), "bright": float(rb[j][src_masks[j]].pow(2).mean()),
                                       "bright_max_abs": float(rb[j][src_masks[j]].abs().max())} for j in range(len(BANDS))}''',
    '''                per_band = {REG.bands[j]: {"all": float(rb[j].pow(2).mean()), "bright": float(rb[j][src_masks[j]].pow(2).mean()),
                                           "bright_max_abs": float(rb[j][src_masks[j]].abs().max())} for j in range(len(REG.bands))}''')

# ------------------------------------------------------------------------------------------------ splat seeds (detection bands)
rep('''        D = (init[:3] / sig_b[:3, None, None]).sum(0) / math.sqrt(3)
        D = D - D.median()''', '''        sw = detection_bands()  # NIRCam SW bands by name
        D = (init[sw] / sig_b[sw][:, None, None]).sum(0) / math.sqrt(len(sw))
        D = D - D.median()''')

# ------------------------------------------------------------------------------------------------ 2./4. arguments and main
rep('''    ap.add_argument("--data-dir", default="/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned")''',
    '''    ap.add_argument("--data-dir", default="/arc/projects/ots/torchsky-data/jwst_smacs0723/aligned")
    ap.add_argument("--hst-dir", default=None, help="v29: directory with HST *_flt.fits / *_flc.fits; enables the HST exposures")
    ap.add_argument("--hst-manifest", default=None, help="CSV (like hst_imaging_unique.csv) selecting the HST files in --hst-dir; without it every flt/flc in the directory is filtered on its headers")
    ap.add_argument("--hst-programs", default=None, help="comma list of HST proposal ids")
    ap.add_argument("--hst-instruments", default=None, help='comma list, e.g. "WFC3/IR,ACS/WFC" (also WFC3/UVIS)')
    ap.add_argument("--hst-filters", default=None, help="comma list of HST filter names, e.g. F160W,F814W")
    ap.add_argument("--hst-mjd-min", type=float, default=None, help="earliest EXPSTART (MJD) of the HST exposures")
    ap.add_argument("--hst-mjd-max", type=float, default=None, help="latest EXPSTART (MJD) of the HST exposures")
    ap.add_argument("--hst-dq-policy", default="default", help='HST DQ: "default" (drizzlepac good bits per instrument), "strict" (DQ == 0) or an integer good-bit sum')
    ap.add_argument("--lam-lo", type=float, default=0.70, help="lower end (um) of the spectral domain of the sky field (HST optical needs ~0.35)")
    ap.add_argument("--lam-hi", type=float, default=5.30, help="upper end (um) of the spectral domain of the sky field")''')
rep('''    ap.add_argument("--n-test", type=int, default=1, help="test exposures per band: 1 = v21 (one detector image); N>1 = N dithers (all their detector images), spread over visits")''',
    '''    ap.add_argument("--n-test", type=int, default=None, help="test exposures per band: 1 (default) = v21 (one detector image); N>1 = N dithers (all their detector images; HST: one exposure = all its chips), spread over visits. With --holdout-from: given explicitly, bands without a held-out exposure in the file get this rule")''')
rep('''    ap.add_argument("--holdout-from", default=None, help="metrics.json whose 'holdout' exposure names to reuse")''',
    '''    ap.add_argument("--holdout-from", default=None, help="metrics.json / holdout_selection.json ('holdout' names), JSON list or text file (one name per line; an HST rootname holds out all its chips) to reuse")''')
rep('''    global args_global, SAVE_DIR, CALIBRATE_NOISE, HUBER_DELTA, LAM_NEG
''', '''    global args_global, SAVE_DIR, CALIBRATE_NOISE, HUBER_DELTA, LAM_NEG, LAM_LO, LAM_HI
    LAM_LO, LAM_HI = a.lam_lo, a.lam_hi
''')
rep('''    exps = load_exposures(a.data_dir, thead, dev, pad=int(round(24 * 0.04 / a.scale)))
    counts = {b: sum(1 for e in exps if e["band"] == b and e["d"].numel() > 0.5 * (a.size * a.scale / 0.031) ** 2 * (0.031 / 0.063) ** 2) for b in BANDS}''',
    '''    pad_ = int(round(24 * 0.04 / a.scale))
    exps = load_all_exposures(a, thead, dev, pad_)
    counts = {b: sum(1 for e in exps if e["band"] == b and e["d"].numel() > 0.5 * (a.size * a.scale / 0.031) ** 2 * (0.031 / 0.063) ** 2) for b in NIRCAM_BANDS}''')
rep('''        LOG["centre_fallback"] = True
        exps = load_exposures(a.data_dir, thead, dev, pad=int(round(24 * 0.04 / a.scale)))''', '''        LOG["centre_fallback"] = True
        exps = load_all_exposures(a, thead, dev, pad_)''')

# ------------------------------------------------------------------------------------------------ run()
rep('''def run(exps, thead, dev, a, out, classical_fn=None):
    dets = sorted({e["detname"] for e in exps})''', '''def run(exps, thead, dev, a, out, classical_fn=None):
    global LAM_LO, LAM_HI
    LAM_LO, LAM_HI = getattr(a, "lam_lo", LAM_LO), getattr(a, "lam_hi", LAM_HI)
    set_registry(registry_from_exposures(exps, psf_n=getattr(a, "psf_n", 31)))
    LOG["band_registry"] = REG.summary()
    LOG["lam_domain_um"] = [LAM_LO, LAM_HI]
    tick("band registry: " + ", ".join(f"{b} [{REG.instrument[b]} {REG.channel[b]} {REG.lam_um[b]:.3f}um D={REG.diameter_m[b]}m fwhm={REG.fwhm_arcsec[b]}\\"]" for b in REG.bands))
    if REG.multi and (getattr(a, "explicit_psf", "") == "pupil" or getattr(a, "psf_model", "") == "pupil" or getattr(a, "zip", False)):
        require_nircam_only("--explicit-psf/--psf-model pupil and --zip")
    if getattr(a, "spectral", "none") != "none":
        lo_b = min(REG.lam_um.values())
        if lo_b < LAM_LO or max(REG.lam_um.values()) > LAM_HI:
            raise SystemExit(f"band pivots span {lo_b:.3f}-{max(REG.lam_um.values()):.3f} um, outside the spectral domain "
                             f"[{LAM_LO}, {LAM_HI}] um: set --lam-lo/--lam-hi")
    dets = sorted({e["detname"] for e in exps})''')
rep('''    counts = {b: sum(1 for e in exps if e["band"] == b) for b in BANDS}
    LOG["exposures_per_band"] = counts''', '''    counts = {b: sum(1 for e in exps if e["band"] == b) for b in REG.bands}
    LOG["exposures_per_band"] = counts
    LOG["exposures_per_instrument"] = {ins: sum(1 for e in exps if REG.instrument[e["band"]] == ins) for ins in REG.instruments}''')
rep('''    sel = select_exposures(exps, n_test=getattr(a, "n_test", 1), n_val=getattr(a, "n_val", 0),
                           holdout_names=(json.load(open(a.holdout_from))["holdout"] if getattr(a, "holdout_from", None) else None),
                           no_holdout=getattr(a, "no_holdout", False))''', '''    n_test_arg = getattr(a, "n_test", None)
    sel = select_exposures(exps, n_test=1 if n_test_arg is None else n_test_arg, n_val=getattr(a, "n_val", 0),
                           holdout_names=(read_holdout_names(a.holdout_from) if getattr(a, "holdout_from", None) else None),
                           no_holdout=getattr(a, "no_holdout", False), fill_missing_bands=n_test_arg is not None)''')
rep('''    band_index = {b: k for k, b in enumerate(BANDS)}

    # ---------------- classical (torchsky)''', '''    band_index = dict(REG.index)

    # ---------------- classical (torchsky)''')
rep('''    tick("classical coadds done")
    sig_b = torch.stack([1.4826 * (c - c.median()).abs().median() for c in swarp]).clamp(min=1e-6)
    results = {"swarp": swarp, "drizzle": driz}
    init = driz if getattr(a, "init", "swarp") == "drizzle" else swarp
    src_masks = torch.stack([(c - c.median()) > 5 * sb for c, sb in zip(swarp, sig_b)])''', '''    tick("classical coadds done")
    cover = dict(getattr(cfn, "last_cover", {}))  # HST splat coadds: bool coverage per band (None / absent = all pixels, as v28)
    if any(v is not None for v in cover.values()):
        hb = [b for b in REG.bands if REG.is_hst[b]]
        LOG["hst_coadd_approximation"] = ("HST bands " + ",".join(hb) + ": SWarp-like and Drizzle-like are the SAME inverse-variance weighted "
                                          "tent-splat of the loaded pixels (tx, ty from the header WCS incl. lookup tables); approximation, "
                                          "no flux conservation / PSF matching / outlier rejection; statistics on covered pixels only")
        tick("APPROXIMATION: " + LOG["hst_coadd_approximation"])
    stats = [band_median_sigma(c, cover.get(b)) for b, c in zip(REG.bands, swarp)]
    sig_b = torch.stack([s_ for _, s_ in stats]).clamp(min=1e-6)
    results = {"swarp": swarp, "drizzle": driz}
    init = driz if getattr(a, "init", "swarp") == "drizzle" else swarp
    src_masks = []
    for (m_, _), b, c, sb in zip(stats, REG.bands, swarp, sig_b):
        sm_ = (c - m_) > 5 * sb
        if cover.get(b) is not None:
            sm_ = sm_ & cover[b]
        src_masks.append(sm_)
    src_masks = torch.stack(src_masks)''')
rep('''        Dd = ((swarp[:3] - swarp[:3].flatten(1).median(1).values[:, None, None]) / sig_b[:3, None, None]).sum(0) / math.sqrt(3)''',
    '''        sw = detection_bands()  # v29: the NIRCam SW bands by name (F090W, F150W, F200W)
        Dd = ((swarp[sw] - swarp[sw].flatten(1).median(1).values[:, None, None]) / sig_b[sw][:, None, None]).sum(0) / math.sqrt(len(sw))''')
rep('''"detection": "SWarp-like F090W+F150W+F200W peak > 5 sigma, no neighbour within 0.4 arcsec; scored if S/N > 20 in the exposure"}''',
    '''"detection": f"SWarp-like {'+'.join(REG.bands[k] for k in sw)} peak > 5 sigma, no neighbour within 0.4 arcsec; scored if S/N > 20 in the exposure"}''')
rep('''    sig0 = torch.tensor([FWHM_ARCSEC[b] / a.scale / 2.3548 for b in BANDS], device=dev)''',
    '''    sig0 = torch.tensor([REG.fwhm_arcsec[b] / a.scale / 2.3548 for b in REG.bands], device=dev)''')
rep('''        lam_nodes, w_nodes, curves = load_bandpasses(BANDS, a.spec_q, a.svo_cache)''',
    '''        lam_nodes, w_nodes, curves = load_bandpasses(REG.bands, a.spec_q, a.svo_cache)
        if float(lam_nodes.min()) < LAM_LO or float(lam_nodes.max()) > LAM_HI:
            tick(f"WARNING: bandpass nodes span {float(lam_nodes.min()):.3f}-{float(lam_nodes.max()):.3f} um, beyond the sky domain [{LAM_LO}, {LAM_HI}]")''')
rep('''for k, b in enumerate(BANDS)}
        LOG["bandpass_curves"] = curves''', '''for k, b in enumerate(REG.bands)}
        LOG["bandpass_curves"] = curves''')
rep('''            epsf = ChromaticPSF(BANDS, ndet, a.scale, a.nk, n=a.psf_n).to(dev)''', '''            epsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)''')
rep('''        from pupil_psf import PupilPSF
        assert spec, "--zip needs --spectral rbf"''', '''        from pupil_psf import PupilPSF
        require_nircam_only("--zip (pupil PSF)")
        assert spec, "--zip needs --spectral rbf"''')
rep('''        zpsf = PupilPSF(BANDS, [LAM_UM[b] for b in BANDS], [CHANNEL[b] for b in BANDS], det_chan, a.scale, n=a.psf_n).to(dev)''',
    '''        zpsf = PupilPSF(REG.bands, [REG.lam_um[b] for b in REG.bands], [REG.chan_id[b] for b in REG.bands], det_chan, a.scale, n=a.psf_n).to(dev)''')
rep('''            fsky = SirenSky(a.size, len(BANDS), hidden''', '''            fsky = SirenSky(a.size, len(REG.bands), hidden''')
rep('''                fsky = FieldSky(a.size, len(BANDS), scale=sig_b * 20).to(dev)''', '''                fsky = FieldSky(a.size, len(REG.bands), scale=sig_b * 20).to(dev)''')
rep('''        ipsf = ImplicitVarPSF(len(BANDS), ndet, a.nk, n=a.psf_n).to(dev)''', '''        ipsf = ImplicitVarPSF(len(REG.bands), ndet, a.nk, n=a.psf_n).to(dev)''')
rep('''            ipsf = ChromaticPSF(BANDS, ndet, a.scale, a.nk, n=a.psf_n).to(dev)''', '''            ipsf = ChromaticPSF(REG.bands, ndet, a.scale, a.nk, n=a.psf_n).to(dev)''')
rep('''    if getattr(a, "updec_dir", None):
        updec = load_updec(a.updec_dir, thead, dev)''', '''    if getattr(a, "updec_dir", None) and REG.multi:
        tick("WARNING: --updec-dir products exist for NIRCam only; skipped because HST bands are present")
    elif getattr(a, "updec_dir", None):
        updec = load_updec(a.updec_dir, thead, dev)''')
rep('''        LOG["metrics"].setdefault("heldout_per_band", {})[name] = L["per_band"]
''', '''        LOG["metrics"].setdefault("heldout_per_band", {})[name] = L["per_band"]
        if REG.multi:
            LOG["metrics"].setdefault("heldout_per_instrument", {})[name] = L["per_instrument"]
''')
rep('''        tick(f"TEST {name}: chi2 all {c2:.4f} src {c2s:.4f} | robust all {L['robust']:.4f} src {L['robust_src']:.4f}")
''', '''        inst_txt = ""
        if REG.multi:  # v29: instrument roll-up (and per-band lines) of the held-out scores
            f_ = lambda v: "nan" if v is None else f"{v:.4f}"
            inst_txt = "".join(f" | {ins} (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} robust all {f_(v['robust'])} src {f_(v['robust_src'])}"
                               for ins, v in L["per_instrument"].items())
            for b_, v in L["per_band"].items():
                tick(f"TEST-BAND {name} {b_} [{REG.instrument[b_]}] (n_exp {v['n_exp']}): chi2 all {f_(v['chi2'])} src {f_(v['chi2_src'])} "
                     f"robust all {f_(v['robust'])} src {f_(v['robust_src'])}")
        tick(f"TEST {name}: chi2 all {c2:.4f} src {c2s:.4f} | robust all {L['robust']:.4f} src {L['robust_src']:.4f}" + inst_txt)
''')
rep('''    disp_sig = torch.full((len(BANDS),), 0.06 / a.scale / 2.3548, device=dev)''', '''    disp_sig = torch.full((len(REG.bands),), 0.06 / a.scale / 2.3548, device=dev)''')
rep('''groups=len(BANDS))[0]''', '''groups=len(REG.bands))[0]''')
rep('''                lam_out = torch.exp(torch.linspace(math.log(0.8), math.log(5.0), 25, device=dev))''',
    '''                lo_out = 0.8 if LAM_LO >= 0.70 else LAM_LO * 8.0 / 7.0  # inside the domain, as v28's 0.8-5.0 inside 0.70-5.30
                hi_out = 5.0 if LAM_HI <= 5.30 else LAM_HI * 5.0 / 5.3
                lam_out = torch.exp(torch.linspace(math.log(lo_out), math.log(hi_out), 25, device=dev))''')
rep('''            for b in BANDS:
                k = band_index[b]''', '''            for b in REG.bands:
                k = band_index[b]''')
rep('''dict(zip(BANDS, [float(x) for x in (m.psf.log_sig.exp() * 2.3548 * a.scale).cpu()]))''',
    '''dict(zip(REG.bands, [float(x) for x in (m.psf.log_sig.exp() * 2.3548 * a.scale).cpu()]))''')
rep('''    ref = (swarp / norm[:, None, None]).sum(0) / 3
''', '''    ref = (swarp / norm[:, None, None]).sum(0) / 3 * (6.0 / len(REG.bands))  # factor 1.0 for the six NIRCam bands
''')

open(dst, "w").write(s)
print("wrote", dst)
