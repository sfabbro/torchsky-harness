"""v37 -> v38 (7 Nov 2026): the data-diversity and held-out-test infrastructure of the audit's T1-T7 (psf-geometry-audit.md, addendum).
Every new flag defaults to the v37 behaviour; without them v38 is bit-identical to v37 (tests/check_v38_equiv.py).

1. --extra-jwst-dirs D1,D2,... (--extra-jwst-glob "*_cal.fits", --include-miri)   MORE JWST DATA
   JWST calibrated exposures of other programmes / instruments that overlap the target box are loaded IN ADDITION to --data-dir by the same
   loader as v37 (SCI/ERR/DQ, torchsky WCS from the SCI header, DQ == 0, same pixel selection and units: MJy/sr per pixel as given).
   Instrument, detector and band come from the primary header (INSTRUME, DETECTOR, FILTER, PUPIL, EXP_TYPE). The filter may be in FILTER (PUPIL
   CLEAR/CLEARP) or in PUPIL (FILTER CLEAR); if both name something other than a clear position the combination is skipped (e.g. F444W + F405N).
   Only imaging exposures (EXP_TYPE ending in _IMAGE) are read. A band is an (instrument, filter) pair: NIRCam keeps its bare name (F150W,
   shared with --data-dir), NIRISS bands are NIRISS_F150W, MIRI bands MIRI_F770W; SVO ids JWST/NIRCam.F150W, JWST/NIRISS.F150W, JWST/MIRI.F770W.
   Registry additions: NIRCam F115W, F182M, F210M, F300M, F410M (SW/LW channels as the existing bands; module A detectors NRCA1-4, NRCALONG are
   ordinary detector names); one channel NIRISS (all 12 imaging filters, 0.0656 arcsec/px) and one MIRI; band order NIRCam, NIRISS, MIRI, HST.
   Registry numbers are NOT measured: pivot wavelengths are entered from memory of the JDox filter tables (about 1 per cent; they only order the bands,
   condition the PSF networks and set the starting width; the sky uses the SVO bandpass nodes); starting FWHM = 1.028 lambda/D with D = 6.5 m
   (diffraction only, no pixel term, as the existing NIRCam rows; NIRISS short-wavelength and MIRI are pixel-limited, so this is a start value only).
   uses_stripes() stays NIRCam-only: NIRISS 1/f noise is NOT modelled (not trivial: readout direction and amplitude were not checked).
   MIRI is excluded unless --include-miri: pixel 0.11 arcsec (2.75 target px at 0.04), PSF FWHM 0.27-0.8 arcsec so the 3 arcsec window holds
   only 4-11 FWHM (the wings need more), wavelengths 5.6-25.5 um need --lam-hi, and its cruciform / droop / scattered-light are unmodelled.
   NOT done / approximations: the registry pixel scale is only used to warn when the WCS Jacobian of an exposure differs by > 3 per cent; JWST
   pixel-area maps are NOT applied (as for NIRCam in v37); the physics PSF library (--lib-source file) has no NIRISS/MIRI entry and stops with
   an error: use --lib-source gauss for them. The SCI-header WCS (SIP) convention of --data-dir is required: files with only a gwcs are skipped (logged).
   --data-dir none loads no main-directory JWST exposures (NIRISS-only runs; add --no-fallback).

2. --freeze-sky-from RUN_DIR|FILE (--freeze-instruments-from RUN_DIR|FILE)   FROZEN-SKY TRANSFER (T4)
   The explicit spectral grid sky (sky.coeffs.s, K x H x W, and sky.basis.*) is loaded from <name>_state.pt (else <name>_best.pt, else the single
   explicit*3d*_state.pt) of an earlier run and frozen; only the operators are fitted. The sky is s(x, y, lambda) = sum_k a_k(x, y) phi_k(lambda); a band
   sees it through its SVO bandpass nodes (SpectralSky.nodes), so a band the source run never had needs NO new sky parameter. The only band-shaped
   sky tensors are the BUFFERS lam_nodes, w_nodes (nb, Q), which are not loaded and are recomputed for this run's bands (stated in the log). The
   loader checks that the target grid (ra, dec, scale, size, halo), spec_k, lam_lo/lam_hi (from the source metrics.json) and every tensor shape agree
   and stops with a message otherwise (no partial or silent loading). Only the grid sky (--sky-coeffs grid) is supported.
   --freeze-instruments-from: operators of instruments already fitted are copied from the source state BY NAME and kept fixed: per-exposure
   parameters (bkg.pe, bkg.logg, geom.M, geom.t, stripes.r/c) by exposure name, per-detector ones (bkg.pd, geom.D, psf.det_emb rows) by detector name,
   library tables psf.tab by (detector, band), residual networks by channel. Whole tensors (tables, networks) are frozen with requires_grad = False;
   rows of shared tensors are frozen by zeroing their gradient (Adam with a zero gradient and zero moments does not move them). A table whose
   shape differs from the source (another window or oversampling) stays free and is reported. Library PSF only (--psf-model lib).
   APPROXIMATION: the frozen rows still enter the loss; the sky convolution is recomputed (with autograd) every step, so a frozen-sky run costs about as
   much as a normal one. Exposures of already fitted instruments cost time but not accuracy: load only the new instrument's data to save it.

3. --dump-heldout-pixels   RESIDUAL PIXELS FOR THE NULL STACKS (T7)
   In the final scoring of every fitted model (not the periodic validation, not swarp/drizzle) one compressed npz per scored exposure is written to
   OUT/heldout_pixels/<model>/<exposure>.npz with, per pixel: row, col (detector), dx, dy (normalised detector position), tx, ty (position on the
   target grid AFTER the held-out registration refit), tx_hdr, ty_hdr (before it, with the static distortion), data, model (total: background +
   stripes + gain x sky), model_sky (gain x convolved sky only), model_bkg (model - model_sky), sigma (1/sqrt(ivar): the pipeline ERR, times
   noise_scale if --calibrate-noise), and per exposure name, band, instrument, detector, det_index, split (test/val/train), mjd (JWST: MJD-AVG; HST:
   EXPSTART = start, not mid-point), roll_deg (rotation theta of the exposure's Jacobian, estimate_jacobian), parity, pixscale_arcsec, gain, noise_scale.
   Old bug (heldout_resid.npy): coords were keyed by band with the LAST scored exposure of the band, resid with the FIRST: they refer to different
   exposures. With the flag the coords are the first exposure's (consistent with resid); without it the bug is kept so that v38 stays bit-identical.
"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


# ===================================================================================== docstring
rep('''detector frame, the header WCS is the starting map only, the learned Geometry/PSF/sky do the rest (see patch_v29.py).
"""''', '''detector frame, the header WCS is the starting map only, the learned Geometry/PSF/sky do the rest (see patch_v29.py).

v38: --extra-jwst-dirs (NIRISS, NIRCam module A and medium bands, optionally MIRI), --freeze-sky-from / --freeze-instruments-from (frozen-sky transfer, T4),
--dump-heldout-pixels (T7 residual pixels). All off by default: bit-identical to v37. See patch_v38.py for every approximation.
"""''')

# ===================================================================================== registry
rep('''CHANNELS = ["NIRCAM_SW", "NIRCAM_LW", "WFC3_IR", "ACS_WFC", "WFC3_UVIS"]  # canonical order = one-hot order (NIRCam first)
CHANNEL_DIAMETER_M = {"NIRCAM_SW": 6.5, "NIRCAM_LW": 6.5, "WFC3_IR": 2.4, "ACS_WFC": 2.4, "WFC3_UVIS": 2.4}
''', r'''CHANNELS = ["NIRCAM_SW", "NIRCAM_LW", "NIRISS", "MIRI", "WFC3_IR", "ACS_WFC", "WFC3_UVIS"]  # canonical order = one-hot order (NIRCam first); v38: + NIRISS, MIRI
CHANNEL_DIAMETER_M = {"NIRCAM_SW": 6.5, "NIRCAM_LW": 6.5, "NIRISS": 6.5, "MIRI": 6.5, "WFC3_IR": 2.4, "ACS_WFC": 2.4, "WFC3_UVIS": 2.4}
# ---- v38: JWST bands beyond the six of the --data-dir (see patch_v38.py). Pivot wavelengths [um] entered from memory of the JDox filter tables
# (about 1 per cent; they only order the bands, condition the PSF networks and set the starting width: the sky uses the SVO bandpass nodes).
NIRCAM_EXTRA_LAM_UM = {"F115W": 1.154, "F182M": 1.845, "F210M": 2.093, "F300M": 2.989, "F410M": 4.082}
NIRCAM_EXTRA_CHANNEL = {"F115W": "NIRCAM_SW", "F182M": "NIRCAM_SW", "F210M": "NIRCAM_SW", "F300M": "NIRCAM_LW", "F410M": "NIRCAM_LW"}
NIRCAM_ALL_LAM_UM = dict(NIRCAM_LAM_UM, **NIRCAM_EXTRA_LAM_UM)
NIRCAM_ALL_CHANNEL = dict(NIRCAM_CHANNEL, **NIRCAM_EXTRA_CHANNEL)
NIRCAM_ALL = sorted(NIRCAM_ALL_LAM_UM, key=lambda b: NIRCAM_ALL_LAM_UM[b])  # F090W F115W F150W F182M F200W F210M F277W F300M F356W F410M F444W
NIRISS_LAM_UM = {"F090W": 0.903, "F115W": 1.148, "F140M": 1.404, "F150W": 1.493, "F158M": 1.586, "F200W": 1.993, "F277W": 2.762, "F356W": 3.593,
                 "F380M": 3.826, "F430M": 4.282, "F444W": 4.434, "F480M": 4.815}
MIRI_LAM_UM = {"F560W": 5.6, "F770W": 7.7, "F1000W": 10.0, "F1130W": 11.3, "F1280W": 12.8, "F1500W": 15.0, "F1800W": 18.0, "F2100W": 21.0, "F2550W": 25.5}
JWST_OTHER_LAM_UM = {"NIRISS": NIRISS_LAM_UM, "MIRI": MIRI_LAM_UM}
JWST_OTHER_CHANNEL = {"NIRISS": "NIRISS", "MIRI": "MIRI"}
CHANNEL_PIXEL_SCALE = {"NIRCAM_SW": 0.031, "NIRCAM_LW": 0.063, "NIRISS": 0.0656, "MIRI": 0.11}  # nominal detector pixel scale [arcsec]
# detector name (header DETECTOR) -> (instrument, channel). NIRCam module A and B; NIRISS imaging; MIRI imager. (MIRI IFU detectors are not imaging.)
JWST_DETECTORS = dict([(f"NRC{m}{i}", ("NIRCam", "NIRCAM_SW")) for m in "AB" for i in "1234"] + [(f"NRC{m}LONG", ("NIRCam", "NIRCAM_LW")) for m in "AB"]
                      + [("NIS", ("NIRISS", "NIRISS")), ("MIRIMAGE", ("MIRI", "MIRI"))])
JWST_INSTRUMENTS = {"NIRCAM": "NIRCam", "NIRISS": "NIRISS", "MIRI": "MIRI"}
CLEAR_POSITIONS = ("", "CLEAR", "CLEARP", "OPEN", "N/A", "NONE")


def jwst_fwhm_start(lam_um, diameter_m=6.5):
    """Starting FWHM [arcsec] of the extra JWST bands: 1.028 lambda/D (the formula behind the existing NIRCam table; diffraction only)."""
    return round(1.028 * 0.206265 * lam_um / diameter_m, 3)
''')

rep('''        self.is_hst = {b: not v for b, v in self.is_nircam.items()}
''', '''        self.is_hst = {b: self.facility[b] == "HST" for b in self.bands}  # v38: JWST NIRISS / MIRI are neither NIRCam nor HST (v37: not is_nircam)
        self.pixel_scale = {b: CHANNEL_PIXEL_SCALE.get(self.channel[b]) for b in self.bands}  # v38: nominal arcsec / detector px (JWST channels only)
''')

rep('''def _nircam_entry(b, psf_n):
    ch = NIRCAM_CHANNEL[b]
    return {"band": b, "lam_um": NIRCAM_LAM_UM[b], "diameter_m": CHANNEL_DIAMETER_M[ch], "fwhm_arcsec": NIRCAM_FWHM_ARCSEC[b],
            "channel": ch, "svo_id": f"JWST/NIRCam.{b}", "kernel_px": psf_n, "instrument": "NIRCam", "facility": "JWST"}
''', '''def _nircam_entry(b, psf_n):
    ch = NIRCAM_ALL_CHANNEL[b]
    return {"band": b, "lam_um": NIRCAM_ALL_LAM_UM[b], "diameter_m": CHANNEL_DIAMETER_M[ch],
            "fwhm_arcsec": NIRCAM_FWHM_ARCSEC[b] if b in NIRCAM_FWHM_ARCSEC else jwst_fwhm_start(NIRCAM_ALL_LAM_UM[b]),
            "channel": ch, "svo_id": f"JWST/NIRCam.{b}", "kernel_px": psf_n, "instrument": "NIRCam", "facility": "JWST"}


def _jwst_other_entry(info, psf_n):
    """info: band (INSTRUMENT_FILTER), instrument (NIRISS | MIRI), filter. v38."""
    ins, f = info["instrument"], info["filter"]
    ch = JWST_OTHER_CHANNEL[ins]
    lam = JWST_OTHER_LAM_UM[ins][f]
    return {"band": info["band"], "lam_um": lam, "diameter_m": CHANNEL_DIAMETER_M[ch], "fwhm_arcsec": jwst_fwhm_start(lam), "channel": ch,
            "svo_id": f"JWST/{ins}.{f}", "kernel_px": psf_n, "instrument": ins, "facility": "JWST"}
''')

rep('''def build_registry(nircam_bands, hst_infos=(), psf_n=31):
    """nircam_bands: names (any subset of NIRCAM_BANDS, put in canonical order); hst_infos: list of dicts for _hst_entry."""
    nir = [b for b in NIRCAM_BANDS if b in set(nircam_bands)]
    hst = sorted((_hst_entry(i, psf_n) for i in hst_infos), key=lambda e: (e["lam_um"], e["band"]))
    return BandRegistry([_nircam_entry(b, psf_n) for b in nir] + hst)
''', '''def build_registry(nircam_bands, hst_infos=(), psf_n=31, jwst_other=()):
    """nircam_bands: names (any subset of NIRCAM_ALL, put in canonical order); hst_infos: list of dicts for _hst_entry; jwst_other (v38): dicts for
    _jwst_other_entry (NIRISS, MIRI), placed after NIRCam and before HST, by channel then wavelength."""
    nir = [b for b in NIRCAM_ALL if b in set(nircam_bands)]
    hst = sorted((_hst_entry(i, psf_n) for i in hst_infos), key=lambda e: (e["lam_um"], e["band"]))
    jw = sorted((_jwst_other_entry(i, psf_n) for i in jwst_other), key=lambda e: (CHANNELS.index(e["channel"]), e["lam_um"], e["band"]))
    return BandRegistry([_nircam_entry(b, psf_n) for b in nir] + jw + hst)
''')

rep('''    nir = {e["band"] for e in exps if not e.get("instrument") and e["band"] in NIRCAM_BANDS}
    by = {}
    for e in exps:
        if e.get("instrument"):
            by.setdefault(e["band"], []).append(e)
''', '''    nir = {e["band"] for e in exps if not e.get("instrument") and e["band"] in NIRCAM_BANDS}
    by, jw_other = {}, {}
    for e in exps:
        if e.get("facility") == "JWST" and e.get("instrument"):  # v38: --extra-jwst-dirs exposures
            if e["instrument"] == "NIRCam":
                nir.add(e["band"])
            else:
                jw_other[e["band"]] = {"band": e["band"], "instrument": e["instrument"], "filter": e["filter"]}
        elif e.get("instrument"):
            by.setdefault(e["band"], []).append(e)
''')
rep('''    return build_registry(sorted(nir), infos, psf_n)
''', '''    return build_registry(sorted(nir), infos, psf_n, jwst_other=list(jw_other.values()))
''')

rep('''    hst = [b for b in REG.bands if REG.is_hst[b]]
    if hst:
        raise SystemExit(f"{what} is NIRCam-only (6.5 m pupil model) and cannot be used with HST bands present: {hst}")''',
    '''    hst = [b for b in REG.bands if REG.is_hst[b]]
    if hst:
        raise SystemExit(f"{what} is NIRCam-only (6.5 m pupil model) and cannot be used with HST bands present: {hst}")
    other = [b for b in REG.bands if not REG.is_nircam[b]]  # v38: NIRISS / MIRI bands
    if other:
        raise SystemExit(f"{what} is NIRCam-only (6.5 m pupil model) and cannot be used with NIRISS/MIRI bands present: {other}")''')

# ===================================================================================== loader
rep('''def load_exposures(data_dir, thead, device, pad=24):
    from torchsky.wcs import WCS

    tw = WCS(header=thead)
    size = thead["NAXIS1"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    exps = []
    for f in files:
        b = exposure_band(f)
        if b is None:
            continue
''', r'''def jwst_header_info(h0, include_miri=False):
    """v38: instrument / detector / band of a JWST primary header (INSTRUME, DETECTOR, FILTER, PUPIL, EXP_TYPE). Returns a dict; "skip" in it = reason
    the file is not used. NIRISS/NIRCam: the filter is in FILTER (PUPIL clear) or in PUPIL (FILTER clear); both set to non-clear = skipped."""
    g = lambda k: str(h0.get(k, "") or "").strip().upper()
    ins_h, det, f, p, et, tel = g("INSTRUME"), g("DETECTOR"), g("FILTER"), g("PUPIL"), g("EXP_TYPE"), g("TELESCOP")
    out = {"instrument_hdr": ins_h, "detector": det, "filter_hdr": f, "pupil_hdr": p, "exp_type": et}
    if tel and tel != "JWST":
        return dict(out, skip=f"TELESCOP {tel} is not JWST")
    inst = JWST_INSTRUMENTS.get(ins_h)
    if inst is None:
        return dict(out, skip=f"INSTRUME {ins_h!r} is not NIRCam, NIRISS or MIRI")
    out["instrument"] = inst
    if et and not et.endswith("_IMAGE"):
        return dict(out, skip=f"EXP_TYPE {et} is not imaging")
    if inst == "MIRI" and not include_miri:
        return dict(out, skip="MIRI excluded (pixel 0.11 arcsec, PSF 0.27-0.8 arcsec FWHM vs a 3 arcsec window, wavelengths beyond the sky domain): use --include-miri")
    table = NIRCAM_ALL_LAM_UM if inst == "NIRCam" else JWST_OTHER_LAM_UM[inst]
    cands = [x for x in (f, p) if x in table]
    others = [x for x in (f, p) if x not in table]
    if len(cands) != 1 or any(x not in CLEAR_POSITIONS for x in others):
        return dict(out, skip=f"unsupported FILTER/PUPIL combination {f!r}/{p!r} for {inst} imaging")
    filt = cands[0]
    if det not in JWST_DETECTORS or JWST_DETECTORS[det][0] != inst:
        return dict(out, skip=f"DETECTOR {det!r} is not a known {inst} imaging detector")
    band = filt if inst == "NIRCam" else f"{inst}_{filt}"
    chan = NIRCAM_ALL_CHANNEL[filt] if inst == "NIRCam" else JWST_OTHER_CHANNEL[inst]
    if inst == "NIRCam" and JWST_DETECTORS[det][1] != chan:
        return dict(out, skip=f"{filt} is a {chan} band but DETECTOR {det} is {JWST_DETECTORS[det][1]}")
    out.update(band=band, filter=filt, channel=chan, lam_um=table[filt], svo_id=f"JWST/{inst}.{filt}", pixel_scale_arcsec=CHANNEL_PIXEL_SCALE[chan])
    return out


def jwst_file_info(path, include_miri=False):
    return jwst_header_info(fits.getheader(path, 0), include_miri)


def load_exposures(data_dir, thead, device, pad=24, extra=None):
    """extra (v38, --extra-jwst-dirs): None = v37 (NIRCam bands of --data-dir, every *.fits); else a dict {"glob", "include_miri", "skipped": []}:
    files matching the glob are read through jwst_header_info (any JWST imaging instrument) and their exposure dicts carry instrument, facility,
    filter, mjd, jwst_extra; skipped files are listed with the reason in extra["skipped"]."""
    from torchsky.wcs import WCS

    tw = WCS(header=thead)
    size = thead["NAXIS1"]
    files = sorted(glob.glob(os.path.join(data_dir, "*.fits" if extra is None else extra["glob"])))
    exps = []
    for f in files:
        meta = {}
        if extra is None:
            b = exposure_band(f)
            if b is None:
                continue
        else:
            info = jwst_file_info(f, extra["include_miri"])
            if info.get("skip"):
                extra["skipped"].append([os.path.basename(f), info["skip"]])
                continue
            b = info["band"]
            meta = {"instrument": info["instrument"], "facility": "JWST", "filter": info["filter"], "jwst_extra": True}
''')
rep('''        hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        try:
            w = WCS(header=hd)
        except Exception as exc:  # noqa: BLE001
            print("skip (wcs)", os.path.basename(f), exc)
            continue
''', '''        hd = {k: sh[k] for k in sh.keys() if k and k not in ("COMMENT", "HISTORY", "")}
        if extra is not None:
            meta["mjd"] = float(hd.get("MJD-AVG", hd.get("MJD-BEG", float("nan"))))
            if "CTYPE1" not in hd:
                extra["skipped"].append([os.path.basename(f), "no FITS WCS (CTYPE1) in the SCI header: gwcs-only file, run the same alignment as for --data-dir"])
                continue
        try:
            w = WCS(header=hd)
        except Exception as exc:  # noqa: BLE001
            print("skip (wcs)", os.path.basename(f), exc)
            if extra is not None:
                extra["skipped"].append([os.path.basename(f), f"WCS from the SCI header failed: {exc}"])
            continue
''')
rep('''        if not inside.any():
            continue
        x0, x1 = max(int(xx.ravel()[inside].min()) - 32, 0), min(int(xx.ravel()[inside].max()) + 33, nx)''',
    '''        if not inside.any():
            if extra is not None:
                extra["skipped"].append([os.path.basename(f), "no overlap with the target box"])
            continue
        x0, x1 = max(int(xx.ravel()[inside].min()) - 32, 0), min(int(xx.ravel()[inside].max()) + 33, nx)''')
rep('''        if keep.sum() < 2000:
            continue
        dk, ek = d[keep], e[keep]''', '''        if keep.sum() < 2000:
            if extra is not None:
                extra["skipped"].append([os.path.basename(f), f"only {int(keep.sum())} usable pixels in the target box"])
            continue
        dk, ek = d[keep], e[keep]''')
rep('''            "detname": str(fits.getheader(f, 0).get("DETECTOR", "")).strip().upper() or os.path.basename(f).split("_")[4],
        })
    return exps
''', '''            "detname": str(fits.getheader(f, 0).get("DETECTOR", "")).strip().upper() or os.path.basename(f).split("_")[4],
            **meta,
        })
    return exps


def load_extra_jwst(a, thead, dev, pad):
    """v38: exposures of --extra-jwst-dirs (comma list), after those of --data-dir. Logs per directory and per band what was read and what skipped."""
    dirs = _csv_list(a.extra_jwst_dirs)
    out, rep_ = [], {}
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"--extra-jwst-dirs: {d} is not a directory")
        ctx = {"glob": a.extra_jwst_glob, "include_miri": bool(a.include_miri), "skipped": []}
        ex = load_exposures(d, thead, dev, pad=pad, extra=ctx)
        perb = {}
        for e in ex:
            perb[e["band"]] = perb.get(e["band"], 0) + 1
        rep_[d] = {"n_files_matching": len(glob.glob(os.path.join(d, a.extra_jwst_glob))), "n_loaded": len(ex), "per_band": perb,
                   "per_detector": {dn: sum(1 for e in ex if e["detname"] == dn) for dn in sorted({e["detname"] for e in ex})},
                   "n_skipped": len(ctx["skipped"]), "skipped_reasons": {r: sum(1 for _, rr in ctx["skipped"] if rr == r) for r in sorted({rr for _, rr in ctx["skipped"]})},
                   "skipped_first": ctx["skipped"][:20]}
        tick(f"extra JWST {d}: {len(ex)} exposures loaded, per band {perb}; {len(ctx['skipped'])} files skipped {rep_[d]['skipped_reasons']}")
        out += ex
    LOG["extra_jwst"] = {"dirs": rep_, "glob": a.extra_jwst_glob, "include_miri": bool(a.include_miri),
                         "note": "NIRISS/MIRI 1/f not modelled (stripes NIRCam-only); pixel-area maps not applied; registry wavelengths/pixel scales entered by hand"}
    return out
''')

rep('''    exps = load_exposures(a.data_dir, thead, dev, pad=pad)
    if not getattr(a, "hst_dir", None):''', '''    exps = [] if str(a.data_dir).lower() == "none" else load_exposures(a.data_dir, thead, dev, pad=pad)  # v38: "none" = no main-directory JWST
    if getattr(a, "extra_jwst_dirs", None):
        exps = exps + load_extra_jwst(a, thead, dev, pad)
        dup = sorted({e["name"] for e in exps if sum(1 for x in exps if x["name"] == e["name"]) > 1})
        if dup:
            raise SystemExit(f"--extra-jwst-dirs: duplicate exposure names across directories (they would collide in --holdout-from): {dup[:5]}")
    if not getattr(a, "hst_dir", None):''')

# ===================================================================================== arguments
rep('''    ap.add_argument("--seed", type=int, default=None, help="v37: seed torch/numpy/random at start (GPU kernels not deterministic)")
    a = ap.parse_args()''', '''    ap.add_argument("--seed", type=int, default=None, help="v37: seed torch/numpy/random at start (GPU kernels not deterministic)")
    ap.add_argument("--extra-jwst-dirs", default=None, help="v38: comma list of directories with further JWST *_cal.fits (NIRISS, NIRCam module A / medium bands, ...) loaded in addition to --data-dir (use --data-dir none for none)")
    ap.add_argument("--extra-jwst-glob", default="*_cal.fits", help="v38: file pattern inside --extra-jwst-dirs")
    ap.add_argument("--include-miri", action="store_true", help="v38: also read MIRI imaging (excluded by default: 0.11 arcsec pixels, large PSF vs the window, wavelengths need --lam-hi)")
    ap.add_argument("--freeze-sky-from", default=None, help="v38: run directory (or state file) whose grid sky (sky.coeffs, sky.basis) is loaded and frozen; only the operators are fitted (T4)")
    ap.add_argument("--freeze-instruments-from", default=None, help="v38: with --freeze-sky-from: copy and freeze the operators of the exposures / detectors / (detector, band) tables that exist in this run directory (matched by name)")
    ap.add_argument("--dump-heldout-pixels", action="store_true", help="v38: write OUT/heldout_pixels/<model>/<exposure>.npz (per-pixel data, model, sigma, coordinates, MJD, roll) for every scored exposure (T7)")
    a = ap.parse_args()''')
rep('''    configure_psf_frame(a)
    global args_global, SAVE_DIR, CALIBRATE_NOISE''', '''    configure_psf_frame(a)
    validate_v38(a)
    global args_global, SAVE_DIR, CALIBRATE_NOISE''')

# ===================================================================================== run(): hooks
rep('''        setup_psf_frame(exps, train_idx, a, dev)

    if getattr(a, "coarse_shift_jwst", 0.0) > 0:''', '''        setup_psf_frame(exps, train_idx, a, dev)
    if getattr(a, "extra_jwst_dirs", None):
        check_pixel_scales(exps, a)

    if getattr(a, "coarse_shift_jwst", 0.0) > 0:''')
rep('''        if getattr(a, "load_dir", None):
            load_state_checked(m, Path(a.load_dir) / f"{name}_state.pt", dev)
''', '''        if getattr(a, "freeze_sky_from", None):
            apply_frozen_sky(m, a, name)
            if getattr(a, "freeze_instruments_from", None):
                apply_frozen_instruments(m, a, name, dets, exps)
        if getattr(a, "load_dir", None):
            load_state_checked(m, Path(a.load_dir) / f"{name}_state.pt", dev)
''')
rep('''            c2, c2s, rs = heldout_chi2(obj, exps, score_idx, band_index, ndet, steps=20 if a.quick else 150, src_masks=src_masks)
''', '''            set_dump_ctx(a, out, name, obj)
            try:
                c2, c2s, rs = heldout_chi2(obj, exps, score_idx, band_index, ndet, steps=20 if a.quick else 150, src_masks=src_masks)
            finally:
                clear_dump_ctx()
''')
rep('''    hold_geo = {exps[i]["band"]: (exps[i]["tx"].cpu().numpy(), exps[i]["ty"].cpu().numpy()) for i in score_idx}
''', '''    hold_geo = {exps[i]["band"]: (exps[i]["tx"].cpu().numpy(), exps[i]["ty"].cpu().numpy()) for i in score_idx}
    if getattr(a, "dump_heldout_pixels", False):  # v38: coords of the FIRST scored exposure of each band, as resid (the old dict kept the last one)
        hold_geo = {}
        for i in score_idx:
            hold_geo.setdefault(exps[i]["band"], (exps[i]["tx"].cpu().numpy(), exps[i]["ty"].cpu().numpy()))
''')
rep('''            per_exposure.append(pe)
''', '''            per_exposure.append(pe)
            if DUMP_CTX is not None and isinstance(model_or_image, SkyModel):  # v38: --dump-heldout-pixels
                xr_, yr_ = geom(0, e["det"], tx0, ty0, e["dx"], e["dy"])
                dump_exposure_pixels(e, r, m, solve_with.gain * c, xr_, yr_, tx0, ty0, float(solve_with.gain))
''')

# ===================================================================================== new code
NEW = r'''# ----------------------------------------------------------------------------- v38: validation, pixel-scale check, frozen sky, pixel dumps
def validate_v38(a):
    """Flag combinations of v38 (idempotent)."""
    if getattr(a, "include_miri", False) and not getattr(a, "extra_jwst_dirs", None):
        raise SystemExit("--include-miri needs --extra-jwst-dirs")
    if getattr(a, "extra_jwst_glob", "*_cal.fits") != "*_cal.fits" and not getattr(a, "extra_jwst_dirs", None):
        raise SystemExit("--extra-jwst-glob needs --extra-jwst-dirs")
    if getattr(a, "freeze_instruments_from", None) and not getattr(a, "freeze_sky_from", None):
        raise SystemExit("--freeze-instruments-from needs --freeze-sky-from")
    if getattr(a, "freeze_sky_from", None):
        if getattr(a, "spectral", "none") != "rbf":
            raise SystemExit("--freeze-sky-from needs --spectral rbf (the transfer is through the spectral basis)")
        if getattr(a, "sky_coeffs", "grid") != "grid":
            raise SystemExit("--freeze-sky-from supports --sky-coeffs grid only")
        if not getattr(a, "skip_implicit", False):
            raise SystemExit("--freeze-sky-from is explicit-only: add --skip-implicit")
        if getattr(a, "load_dir", None) or getattr(a, "zip", False) or getattr(a, "rigid", False) or getattr(a, "freeze_instrument_from", None) or getattr(a, "uq", False):
            raise SystemExit("--freeze-sky-from does not combine with --load-dir, --zip, --rigid, --freeze-instrument-from or --uq")
        if getattr(a, "freeze_instruments_from", None) and getattr(a, "explicit_psf", "") != "lib":
            raise SystemExit("--freeze-instruments-from supports the library PSF only (--psf-model lib --explicit-psf lib)")


def check_pixel_scales(exps, a):
    """Warn when the WCS Jacobian of an exposure differs by > 3 per cent from the registry's nominal pixel scale (JWST channels). Log only."""
    bad, n = [], 0
    for e in exps:
        ps = REG.pixel_scale.get(e["band"])
        if ps is None:
            continue
        j = e.get("jac") or estimate_jacobian(e)
        got = j["scale"] * a.scale
        n += 1
        if abs(got / ps - 1.0) > 0.03:
            bad.append([e["name"], round(got, 5), ps])
    LOG["pixel_scale_check"] = {"n_checked": n, "n_off_by_more_than_3pct": len(bad), "examples": bad[:10]}
    for nm, got, ps in bad[:10]:
        tick(f"WARNING: {nm}: WCS pixel scale {got} arcsec vs registry {ps}")
    tick(f"pixel-scale check against the registry: {n} JWST exposures, {len(bad)} differ by more than 3 per cent")
    if any(REG.channel[b] == "MIRI" for b in REG.bands):
        tick(f"WARNING: MIRI bands present: the PSF window is {a.psf_n * a.scale:.2f} arcsec (3 arcsec = 4-11 FWHM at 5.6-25 um, wings are cut) and --lam-hi must cover 25.5 um")


def _state_file(src, name):
    p = Path(src)
    if p.is_file():
        return p
    for suf in ("_state.pt", "_best.pt"):
        if (p / f"{name}{suf}").exists():
            return p / f"{name}{suf}"
    c = sorted(p.glob("explicit*3d*_state.pt"))
    if len(c) == 1:
        return c[0]
    raise SystemExit(f"{src}: no {name}_state.pt / {name}_best.pt and {'no' if not c else 'several'} explicit*3d*_state.pt ({[x.name for x in c]}): pass the state file itself")


def _source_meta(path):
    for fn in ("metrics.json", "metrics_partial.json"):
        q = Path(path).parent / fn
        if q.exists():
            return json.load(open(q))
    return None


def _same_arg(v0, v1):
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(v0) and num(v1):
        return abs(v0 - v1) <= 1e-9 * max(1.0, abs(v0))
    return v0 == v1


def apply_frozen_sky(m, a, name):
    """--freeze-sky-from: load sky.coeffs / sky.basis from an earlier run and freeze every sky parameter (see patch_v38.py)."""
    path = _state_file(a.freeze_sky_from, name)
    sd = torch.load(path, map_location="cpu")
    meta = _source_meta(path)
    rep_ = {"state_file": str(path), "source_metrics": meta is not None}
    if meta is not None and "args" in meta:
        sa = meta["args"]
        bad = [f"{k}: source {sa[k]} vs this run {getattr(a, k)}" for k in ("ra", "dec", "scale", "size", "halo_px", "spec_k", "lam_lo", "lam_hi", "sky_coeffs", "spectral", "target_fwhm")
               if k in sa and hasattr(a, k) and not _same_arg(sa[k], getattr(a, k))]
        if bad:
            raise SystemExit(f"--freeze-sky-from {path}: the source run is not on the same grid / spectral domain: " + "; ".join(bad))
        rep_["checked_args"] = "ra dec scale size halo_px spec_k lam_lo lam_hi sky_coeffs spectral target_fwhm identical"
    else:
        tick(f"WARNING: no metrics.json next to {path}: lam_lo/lam_hi and the grid could not be compared, only tensor shapes are checked")
        rep_["checked_args"] = "NOT CHECKED (no metrics.json)"
    mine = {k: v for k, v in m.state_dict().items() if k.startswith("sky.")}
    band_buffers = [k for k in mine if k in ("sky.lam_nodes", "sky.w_nodes")]
    load, errs = {}, []
    for k, v in mine.items():
        if k in band_buffers:
            continue
        if k not in sd:
            errs.append(f"{k} missing in the source")
        elif tuple(sd[k].shape) != tuple(v.shape):
            errs.append(f"{k}: source {tuple(sd[k].shape)} vs this run {tuple(v.shape)}")
        else:
            load[k] = sd[k]
    unexpected = [k for k in sd if k.startswith("sky.") and k not in mine]
    if errs or unexpected:
        raise SystemExit(f"--freeze-sky-from {path}: sky state does not match: {errs}; source-only sky tensors {unexpected} (a grid sky with the same K, H, W and basis is required)")
    names = [n for n, _ in m.sky.named_parameters()]
    assert all(n.startswith(("coeffs.", "basis.")) for n in names), names  # no band-indexed sky parameter exists
    m.load_state_dict(load, strict=False)
    for p_ in m.sky.parameters():
        p_.requires_grad_(False)
    rep_.update(loaded=sorted(load), frozen_sky_parameters=names, band_buffers_not_loaded=band_buffers,
                band_buffer_shapes={k: list(mine[k].shape) for k in band_buffers},
                source_band_buffer_shapes={k: list(sd[k].shape) for k in band_buffers if k in sd},
                note="s(x,y,lambda) = sum_k a_k(x,y) phi_k(lambda): the coefficients carry no band index; lam_nodes / w_nodes (nb, Q) are buffers recomputed from the SVO "
                     "bandpasses of this run's bands, so bands unseen by the source run need no new sky parameter")
    LOG["freeze_sky"] = rep_
    nfree = sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
    tick(f"{name}: sky frozen from {path}: {sum(v.numel() for v in load.values()) / 1e6:.2f} M sky values, {len(names)} sky parameter tensors, "
         f"band buffers {band_buffers} recomputed for {len(REG.bands)} bands; {nfree / 1e6:.3f} M operator parameters free")


def _row_mask_hook(param, rows):
    """Freeze the given rows (leading index) of a shared parameter by zeroing their gradient."""
    if not rows or not param.requires_grad:
        return
    mask = torch.ones(param.shape[0])
    mask[list(rows)] = 0.0
    param.register_hook(lambda g, mk=mask: g * mk.to(g.device).view(-1, *([1] * (g.dim() - 1))))


def apply_frozen_instruments(m, a, name, dets, exps):
    """--freeze-instruments-from: copy and freeze operator parameters of exposures / detectors / (detector, band) tables present in the source run."""
    path = _state_file(a.freeze_instruments_from, name)
    sd = torch.load(path, map_location="cpu")
    meta = _source_meta(path)
    if meta is None or not all(k in meta for k in ("detectors", "band_registry", "noise_scale")):
        raise SystemExit(f"--freeze-instruments-from {path}: the source run's metrics.json (detectors, band_registry, noise_scale) is needed to match by name")
    s_dets = {n: i for i, n in enumerate(meta["detectors"])}
    s_bands = {r["band"]: i for i, r in enumerate(meta["band_registry"])}
    s_exps = {n: i for i, n in enumerate(meta["noise_scale"])}
    P = dict(m.named_parameters())
    rep_ = {"state_file": str(path), "per_exposure": {}, "per_detector": {}, "tables": {}, "networks": {}}
    mine_e = [(i, s_exps[e["name"]]) for i, e in enumerate(exps) if e["name"] in s_exps]
    mine_d = [(d, s_dets[dn]) for d, dn in enumerate(dets) if dn in s_dets]
    with torch.no_grad():
        for key in ("bkg.pe", "bkg.logg", "geom.M", "geom.t", "stripes.r", "stripes.c"):
            if key not in P or key not in sd:
                continue
            if tuple(sd[key].shape[1:]) != tuple(P[key].shape[1:]):
                rep_["per_exposure"][key] = f"shape {tuple(sd[key].shape[1:])} vs {tuple(P[key].shape[1:])}: left free"
                continue
            for i, j in mine_e:
                P[key][i] = sd[key][j].to(P[key].device)
            _row_mask_hook(P[key], [i for i, _ in mine_e])
            rep_["per_exposure"][key] = len(mine_e)
        for key in ("bkg.pd", "geom.D", "psf.det_emb.weight"):
            if key not in P or key not in sd:
                continue
            if tuple(sd[key].shape[1:]) != tuple(P[key].shape[1:]):
                rep_["per_detector"][key] = f"shape {tuple(sd[key].shape[1:])} vs {tuple(P[key].shape[1:])}: left free"
                continue
            for d, j in mine_d:
                P[key][d] = sd[key][j].to(P[key].device)
            _row_mask_hook(P[key], [d for d, _ in mine_d])
            rep_["per_detector"][key] = [dets[d] for d, _ in mine_d]
        for tk, prm in (list(m.psf.tab.items()) if hasattr(m.psf, "tab") else []):
            d, b = (int(x) for x in tk.split("_"))
            lab = f"{dets[d]}/{REG.bands[b]}"
            if dets[d] in s_dets and REG.bands[b] in s_bands:
                skey = f"psf.tab.{s_dets[dets[d]]}_{s_bands[REG.bands[b]]}"
                if skey in sd and tuple(sd[skey].shape) == tuple(prm.shape):
                    prm.copy_(sd[skey].to(prm.device))
                    prm.requires_grad_(False)
                    rep_["tables"][lab] = "frozen"
                else:
                    rep_["tables"][lab] = "shape differs from the source (window / oversampling): left free" if skey in sd else "absent in the source state: left free"
            else:
                rep_["tables"][lab] = "new (detector, band): free"
        for ch in (list(m.psf.nets.keys()) if hasattr(m.psf, "nets") else []):
            pre = f"psf.nets.{ch}."
            keys = [k for k in P if k.startswith(pre)]
            if keys and all(k in sd and tuple(sd[k].shape) == tuple(P[k].shape) for k in keys):
                for k in keys:
                    P[k].copy_(sd[k].to(P[k].device))
                    P[k].requires_grad_(False)
                rep_["networks"][ch] = "frozen"
            else:
                rep_["networks"][ch] = "not in the source (or shapes differ): free"
    new_e = [e["name"] for e in exps if e["name"] not in s_exps]
    rep_["n_exposures_frozen"], rep_["n_exposures_new"] = len(mine_e), len(new_e)
    rep_["new_exposures_per_instrument"] = {ins: sum(1 for e in exps if e["name"] not in s_exps and REG.instrument[e["band"]] == ins) for ins in REG.instruments}
    LOG["freeze_instruments"] = rep_
    tick(f"{name}: instruments frozen from {path}: {len(mine_e)} exposures and {len(mine_d)} detectors matched by name, "
         f"tables {sum(v == 'frozen' for v in rep_['tables'].values())}/{len(rep_['tables'])} frozen; {len(new_e)} new exposures {rep_['new_exposures_per_instrument']} "
         f"(rows of shared tensors are frozen by gradient masks)")


DUMP_CTX = None


def set_dump_ctx(a, out, name, obj):
    global DUMP_CTX
    DUMP_CTX = None
    if getattr(a, "dump_heldout_pixels", False) and isinstance(obj, SkyModel):
        d = Path(out) / "heldout_pixels" / name
        d.mkdir(parents=True, exist_ok=True)
        DUMP_CTX = {"dir": d, "method": name, "n": 0}


def clear_dump_ctx():
    global DUMP_CTX
    if DUMP_CTX is not None:
        tick(f"--dump-heldout-pixels: {DUMP_CTX['n']} exposure files written to {DUMP_CTX['dir']}")
    DUMP_CTX = None


def exposure_mjd(e):
    """JWST: MJD-AVG of the SCI header (mid-exposure); HST: EXPSTART (start of the exposure, not the mid-point); nan if neither is found."""
    if e.get("mjd") is not None and np.isfinite(e["mjd"]):
        return float(e["mjd"])
    hd = e.get("header") or {}
    for k in ("MJD-AVG", "MJD-BEG"):
        if k in hd:
            return float(hd[k])
    try:
        return float(fits.getheader(e["path"], 0).get("EXPSTART", float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")


def dump_exposure_pixels(e, r, m, sky_total, xr, yr, tx0, ty0, gain):
    j = e.get("jac") or estimate_jacobian(e)
    f32 = lambda t: t.detach().float().cpu().numpy()
    split = "test" if e.get("holdout") else ("val" if e.get("val") else "train")
    stem = Path(e["name"]).name
    stem = stem[:-5] if stem.endswith(".fits") else stem
    np.savez_compressed(
        DUMP_CTX["dir"] / f"{stem}.npz", name=e["name"], band=e["band"], instrument=REG.instrument.get(e["band"], "?"), det=e["detname"], det_index=int(e["det"]),
        split=split, method=DUMP_CTX["method"], mjd=exposure_mjd(e), roll_deg=float(j["theta_deg"]), parity=int(j["parity"]),
        pixscale_arcsec=float(j["scale"] * args_global.scale), gain=gain, noise_scale=float(e.get("noise_scale", 1.0)),
        row=e["row"].cpu().numpy().astype(np.int32), col=e["col"].cpu().numpy().astype(np.int32), dx=f32(e["dx"]), dy=f32(e["dy"]),
        tx=f32(xr), ty=f32(yr), tx_hdr=f32(tx0), ty_hdr=f32(ty0), data=f32(e["d"]), model=f32(m), model_sky=f32(sky_total), model_bkg=f32(m - sky_total),
        sigma=f32(1.0 / e["ivar"].sqrt()))
    DUMP_CTX["n"] += 1


'''
rep('''def run(exps, thead, dev, a, out, classical_fn=None):
''', NEW + '''def run(exps, thead, dev, a, out, classical_fn=None):
''')
open(dst, "w").write(s)
print("ok", dst)
