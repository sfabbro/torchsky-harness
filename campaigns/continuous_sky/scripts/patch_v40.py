"""v39 -> v40 (8 Nov 2026): start from (and optionally freeze) learned library-PSF tables of ANOTHER run: frozen / fine-tuned transfer (experiment P1).
Without the new flags v40 is bit-identical to v39 (tests/check_v40_equiv.py).

--load-tables-from INSTRUMENT_STATE.pt   (needs --psf-model lib --explicit-psf lib --lib-learn-table, no --lib-chromatic)
    INSTRUMENT_STATE.pt is the file written by scripts/tilefit.py (format "tilefit-instrument", version 1): "tables" maps "<DETECTOR>/<BAND>" to a dict
    with "theta" (the raw softplus parameter of the table, (N, N)), "oversample", "N". For every (detector, band) table of THIS run the entry of the same
    detector NAME (header DETECTOR, upper case) and band is looked up; a matched table of identical shape and oversampling is copied into the run's
    parameter (psf.tab["<d>_<b>"], so the table starts from the learned one instead of the generic Gaussian). Reported in the log and in
    metrics.json["load_tables"]: matched, shape-mismatched (left at the start), absent (left at the start). The run stops if NOTHING matches.
--freeze-tables   (needs --load-tables-from)
    The matched tables get requires_grad False: frozen transfer. Without it the tables are the starting point and keep learning at the usual table rate
    (--psf-lr, default 1e-2: for a fine-tune from a good start a smaller --psf-lr, e.g. 2e-3, is advisable; not changed here).
Everything else (sky, per-exposure geometry / background / gain, static distortion and detector background pattern, stripes) is fitted on this run's data as usual.

APPROXIMATIONS / LIMITS
  * Only the achromatic learnable tables (one per detector and band, v36/v38 E2-data) are transferred; --lib-chromatic is refused.
  * Matching is by detector name and band name only: no check that the source run used the same window (--psf-window-arcsec) or pixel scale. The oversampling must
    be equal; the table size N may differ by a few samples (it follows ceil(R x oversample) of the first PSF group's pixel scale): then the unit-sum table is centre-cropped
    or zero-padded to this run's size (same sample positions), renormalised, and its raw parameter recomputed with the start's floor (1e-8 of the peak); the cropped mass
    fraction is logged in metrics.json["load_tables"]["resized"]. A window that differs by more than the edge samples is NOT supported silently: it is cropped / padded the same way. A source table that was never updated in its run (trained == False,
    its pair never had a training exposure) is still loaded; the log lists the source's `n_updates` per matched table (0 = it is the generic start).
  * The per-exposure and per-detector distortion / background parameters of the source are NOT transferred (SMACS exposures are different exposures; the
    detector patterns of the source are specific to its field: transferring them would also need a decision on the background pattern; left free).
  * The PSF gauges / priors (--psf-centre, --lib-table-tv) apply to loaded tables as to any table.
"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


rep('''endpoint tables interpolated linearly in lambda). Default none: bit-identical to v38. No lambda/D physics imposed. See patch_v39.py.
"""''', '''endpoint tables interpolated linearly in lambda). Default none: bit-identical to v38. No lambda/D physics imposed. See patch_v39.py.

v40: --load-tables-from INSTRUMENT_STATE.pt [--freeze-tables]: start from (and optionally freeze) learned library-PSF tables of another run (scripts/tilefit.py),
matched by detector name and band. Default off: bit-identical to v39. See patch_v40.py.
"""''')

rep('''    a = ap.parse_args()
    if a.seed is not None:''', '''    ap.add_argument("--load-tables-from", default=None, metavar="INSTRUMENT_STATE.pt",
                    help="v40: start the learnable library-PSF tables from the file written by tilefit.py (matched by detector name and band; needs --lib-learn-table)")
    ap.add_argument("--freeze-tables", action="store_true", help="v40: with --load-tables-from: freeze the loaded tables (frozen transfer); without it they keep learning (fine-tune)")
    a = ap.parse_args()
    if a.seed is not None:''')

rep('''    validate_v38(a)
    global args_global''', '''    validate_v38(a)
    validate_v40(a)
    global args_global''')

rep('''DUMP_CTX = None


def set_dump_ctx''', '''def validate_v40(a):
    """Flag combinations of v40 (idempotent)."""
    if getattr(a, "freeze_tables", False) and not getattr(a, "load_tables_from", None):
        raise SystemExit("--freeze-tables needs --load-tables-from")
    if getattr(a, "load_tables_from", None):
        if not LIB_ON or not getattr(a, "lib_learn_table", False):
            raise SystemExit("--load-tables-from needs --psf-model lib --explicit-psf lib --lib-learn-table")
        if getattr(a, "lib_chromatic", "none") != "none":
            raise SystemExit("--load-tables-from does not support --lib-chromatic (achromatic tables only)")
        if getattr(a, "freeze_instruments_from", None) or getattr(a, "freeze_instrument_from", None) or getattr(a, "load_dir", None):
            raise SystemExit("--load-tables-from does not combine with --freeze-instruments-from, --freeze-instrument-from or --load-dir")
        if not os.path.exists(a.load_tables_from):
            raise SystemExit(f"--load-tables-from {a.load_tables_from}: no such file")


def apply_loaded_tables(m, a, name, dets):
    """--load-tables-from / --freeze-tables: copy the learned tables of a tilefit.py instrument_state.pt into psf.tab, matched by (detector name, band)."""
    st = torch.load(a.load_tables_from, map_location="cpu", weights_only=False)
    if not isinstance(st, dict) or st.get("format") != "tilefit-instrument":
        raise SystemExit(f"--load-tables-from {a.load_tables_from}: not a tilefit instrument_state.pt (format 'tilefit-instrument' missing)")
    src_tabs = st["tables"]
    rep_ = {"state_file": str(a.load_tables_from), "freeze": bool(a.freeze_tables), "matched": [], "shape_mismatch": [], "absent": [], "source_n_pairs": len(src_tabs)}
    n_match = 0
    with torch.no_grad():
        for tk, prm in m.psf.tab.items():
            d, b = (int(x) for x in tk.split("_"))
            lab = f"{dets[d]}/{REG.bands[b]}"
            ent = src_tabs.get(lab)
            if ent is None:
                rep_["absent"].append(lab)
                continue
            th = ent["theta"]
            if int(ent.get("oversample", m.psf.os)) != int(m.psf.os):
                rep_["shape_mismatch"].append([lab, list(th.shape), list(prm.shape), ent.get("oversample"), int(m.psf.os)])
                continue
            if tuple(th.shape) != tuple(prm.shape):
                # the table size follows ceil(R x oversample) of the first PSF group's pixel scale, which differs by a few samples between data sets; the lattice is centred
                # at N // 2 with the same pitch (1 / oversample detector px), so a centred crop / zero pad keeps every sample at the same detector offset
                Ns, Nt = int(th.shape[-1]), int(prm.shape[-1])
                T = ent["table"].double()
                if Nt < Ns:
                    o = (Ns - Nt) // 2
                    Tn = T[o:o + Nt, o:o + Nt]
                else:
                    o = (Nt - Ns) // 2
                    Tn = torch.zeros(Nt, Nt, dtype=T.dtype)
                    Tn[o:o + Ns, o:o + Ns] = T
                lost = 1.0 - float(Tn.sum() / T.sum())
                Tn = Tn / Tn.sum()
                y = Tn.clamp(min=1e-8 * float(Tn.max()))
                th = torch.log(torch.expm1(y)).float()
                rep_.setdefault("resized", []).append([lab, Ns, Nt, lost])
            prm.copy_(th.to(prm.device, prm.dtype))
            if a.freeze_tables:
                prm.requires_grad_(False)
            rep_["matched"].append([lab, int(ent.get("n_updates", -1))])
            n_match += 1
    LOG["load_tables"] = rep_
    tick(f"{name}: --load-tables-from {a.load_tables_from}: {n_match} of {len(m.psf.tab)} tables matched by (detector, band) and loaded"
         f"{' and FROZEN' if a.freeze_tables else ' (fine-tune: they keep learning)'}; shape mismatch {len(rep_['shape_mismatch'])}, absent in the source {len(rep_['absent'])} "
         f"(left at the generic start): absent {rep_['absent'][:12]}; source updates per matched table {rep_['matched'][:12]}")
    if n_match == 0:
        raise SystemExit("--load-tables-from: no (detector, band) table matched; check detector names / bands / window / oversampling of the source run")


DUMP_CTX = None


def set_dump_ctx''')

rep('''            freeze_instrument(m, Path(a.freeze_instrument_from) / "explicit_chrom_state.pt", dev, name)
''', '''            freeze_instrument(m, Path(a.freeze_instrument_from) / "explicit_chrom_state.pt", dev, name)
        if getattr(a, "load_tables_from", None):  # v40
            apply_loaded_tables(m, a, name, dets)
''')
open(dst, "w").write(s)
print("ok")
