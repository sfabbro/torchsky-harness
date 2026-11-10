# SMACS J0723 field: data beyond NIRCam ERO + HST RELICS (5 Nov 2026)

Archives queried on 5 Nov 2026, for the cone at RA 110.8268, Dec −73.4544:

- MAST: 2′ cone, astroquery `Observations.query_region`.
- ESO: ObsCore and raw TAP, 0.03°.
- IRSA: ObsCore TAP, footprint contains the centre.
- NOIRLab Astro Data Archive: DECam instcal within ±1° boxes.

Counts are archive rows, not exposures. Program titles were not all checked.

## Imaging

| Data | Bands | Epoch | Archive | Why it is useful |
|---|---|---|---|---|
| JWST NIRCam GO 4043 | F090W, F115W (39.5 ks), F444W, plus grism | 2024.4 | MAST | deep; a second NIRCam epoch |
| JWST NIRCam GO 6882 (VENUS?) | F150W, F182M, F210M, F300M, F410M, F444W | 2026.4 | MAST | medium bands sharpen s(λ); a third epoch |
| JWST NIRCam 12774 | F115W, F150W, F300M, F444W (short) | 2026.5 | MAST | epoch, bands |
| JWST NIRISS imaging 2736 / 4368 | F115W, F200W / F150W | 2022.5 / 2024.7 | MAST | a second JWST instrument at the same epoch and bands: the cleanest shared-sky test |
| JWST MIRI 2736 | F770W, F1000W, F1500W, F1800W | 2022.5 | MAST | λ to 18 µm, a different detector and PSF |
| HST WFC3/UVIS 14534, 16567, 16729 | F606W | 2017–2022 | MAST | a third HST camera; time |
| HST WFC3/IR 16729, 18297 | F105W / F110W, F160W | 2022 / 2026.6 | MAST (already on /arc) | time |
| VISTA GCAV (198.A-2008) | Y, J, Ks (~2 h per band) | 2017–2020 | ESO | deepest ground NIR; seeing-limited |
| VISTA VHS | J, Ks (240 s) | 2016 | ESO | shallow |
| HAWK-I 0102.A-0754 / SOFI 092.A-0857 | Ks | 2019 / 2014 | ESO | ground NIR |
| OmegaCAM 112.266S.001 | g, r (4.5 ks each) | 2023–2024 | ESO | ground optical |
| DECam (several programs) | u, g, r, i, z, N (~36 exposures) | 2016–2024 | NOIRLab | ground optical, shallow |
| FORS2 091.A-0284, WFI, EFOSC | R, GG435 | 2011–2013 | ESO | older ground optical |
| ERIS 113.26SV / 114.27LA | K (AO) | 2024–2025 | ESO | AO PSF, small field |
| Spitzer IRAC (SRELICS) | 3.6, 4.5 µm | ~2016 | IRSA | overlaps NIRCam LW at 1.7″ |
| SPHEREx QR2/QR3 | 102 channels, 0.75–5 µm, 6″ pixels | 2025–2026 | IRSA | spectral leverage at low resolution |
| WISE / NEOWISE | W1–W4; W1/W2 with ~760 epochs | 2010–2024 | IRSA | time, mid-IR |
| GALEX, Swift UVOT, 2MASS | FUV/NUV; UVW2, UVM2, UVW1; JHK | 1999–2011 | MAST / IRSA | UV and shallow NIR |

## Spectroscopy (needs a dispersed or IFU forward operator)

| Data | Notes |
|---|---|
| NIRSpec IFU (2957, 2969, 2959) | PRISM, G140M, G395H; 2024 |
| NIRSpec MSA (2736), NIRSpec slit (12774) | multi-object and slit spectra |
| NIRISS WFSS (2736, 4368), NIRCam grism (4043) | slitless; works as imaging through a dispersion operator |
| MUSE (0102.A-0718) | one cube, 2.6 ks, 2019 |
| KMOS, X-shooter (2023), ALMA bands 4–7 | other spectra |

## Not available

- Euclid Q1 does not cover the field; DR1 has not been checked.
- Rubin has no public data here yet.
- UNIONS/CFHT cannot reach Dec −73.

Gaia is used only as an external check, never inside the model.
