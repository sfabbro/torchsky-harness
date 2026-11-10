"""v24 -> v25: one PSF convention flag for every model (5 Nov 2026).
convolve_basis (explicit models) computes a correlation: m(x) = sum_d B(d) S(x + d). The pupil PSF's orientation search
(PupilPSF.fit_orientation) matches the kernel directly to star images, i.e. assumes a convolution m(x) = sum_d B(d) S(x - d).
--psf-convention convolution flips B inside convolve_basis (explicit models) and keeps the zip PSF unflipped (as in v20);
--psf-convention correlation (default) is the v19/v24 behaviour for both. Learned PSFs absorb either convention; the
physical pupil PSF does not (its fitted wavefront has large odd terms)."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


rep('ZIP_PSF_CONVENTION = "explicit"  # v24; "v20" reproduces the old (180-degree rotated) zip PSF\n',
    'ZIP_PSF_CONVENTION = "explicit"  # v24; "v20" reproduces the old (180-degree rotated) zip PSF\n'
    'PSF_CONVENTION = "correlation"  # v25: correlation (v19/v24 explicit) | convolution (pupil orientation fit, v20 zip)\n')
rep('''    nb, nk, n, _ = B.shape
    H, W = S.shape[-2:]
    p = n // 2''', '''    if PSF_CONVENTION == "convolution":  # v25: true convolution m(x) = sum_d B(d) S(x - d)
        B = B.flip(-2, -1)
    nb, nk, n, _ = B.shape
    H, W = S.shape[-2:]
    p = n // 2''')
rep('''        if ZIP_PSF_CONVENTION == "explicit":  # v24: correlation convention of convolve_basis (see patch_v24.py)''',
    '''        if ZIP_PSF_CONVENTION == "explicit" and PSF_CONVENTION == "correlation":  # v24/v25: match convolve_basis''')
rep('''    ap.add_argument("--zip-psf-convention"''',
    '''    ap.add_argument("--psf-convention", choices=("correlation", "convolution"), default="correlation", help="v25: how every model applies its PSF; convolution matches the pupil orientation fit")
    ap.add_argument("--zip-psf-convention"''')
rep('''    global ZIP_PSF_CONVENTION
    ZIP_PSF_CONVENTION = a.zip_psf_convention
''', '''    global ZIP_PSF_CONVENTION, PSF_CONVENTION
    ZIP_PSF_CONVENTION = a.zip_psf_convention
    PSF_CONVENTION = a.psf_convention
    LOG["psf_convention"] = PSF_CONVENTION
''')
open(dst, "w").write(s)
print("wrote", dst)
