"""v23 -> v24: fix the PSF orientation of the zip model (Monte Carlo sampler and deterministic renderer).
The explicit model's convolve_basis computes a correlation (a point source at p0 appears at p0 - delta for a kernel tap
at delta); the zip sampler drew s(x - delta), i.e. a convolution (p0 + delta). The frozen instrument (pupil PSF:
orientation search and wavefront) was fitted in the explicit convention, so v20-v23 zip runs saw that PSF rotated by
180 degrees. The fitted pupil PSF is far from point-symmetric (L1 asymmetry 0.3-0.65 per band, odd aberrations).
v24 flips the tabulated PSF once in build_samplers (default), so both renderers match the explicit model."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """        self.I = I  # (nbQ, D, no, no), kept for the deterministic renderer (v21)
"""
new = """        if ZIP_PSF_CONVENTION == "explicit":  # v24: correlation convention of convolve_basis (see patch_v24.py)
            I = I.flip(-2, -1)
        self.I = I  # (nbQ, D, no, no), kept for the deterministic renderer (v21)
"""
assert s.count(old) == 1; s = s.replace(old, new)
old = "\nT0 = time.perf_counter()\n"
assert s.count(old) == 1
s = s.replace(old, old + 'ZIP_PSF_CONVENTION = "explicit"  # v24; "v20" reproduces the old (180-degree rotated) zip PSF\n')
old = """    ap.add_argument("--zip-render-os", type=int, default=2"""
assert s.count(old) == 1
s = s.replace(old, """    ap.add_argument("--zip-psf-convention", choices=("explicit", "v20"), default="explicit", help="v24: zip PSF orientation; v20 = the old 180-degree rotated one")
""" + old)
old = """        if a.zip_render == "det":
            m.build_kernels("""
assert s.count(old) == 1
s = s.replace(old, """        LOG["zip_psf_convention"] = ZIP_PSF_CONVENTION
""" + old)
old = """    global args_global, SAVE_DIR, CALIBRATE_NOISE, HUBER_DELTA, LAM_NEG
"""
assert s.count(old) == 1
s = s.replace(old, old + """    global ZIP_PSF_CONVENTION
    ZIP_PSF_CONVENTION = a.zip_psf_convention
""")
open(dst, "w").write(s)
print("wrote", dst)
