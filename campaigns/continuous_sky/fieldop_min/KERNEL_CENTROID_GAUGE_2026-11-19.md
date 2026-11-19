# Kernel-centroid gauge, 19 November 2026

A convention ("gauge") that fixes the position of the learned kernels (point-spread functions, PSFs) against the sky, added to the field
operator in the research prototype (`/home/claude/work/fieldop_min/fieldop/`) and, identically, in the library
(`/home/claude/torchsky/src/torchsky/map/exposure_fit/`). It is behind one setting, off by default: `--kernel-centroid-gauge {0,1}` in the
prototype's parser, `"kernel_centroid_gauge"` in the library's private settings mapping. With it off every fit is bit for bit what it was.
CPU only, synthetic data only; no git commit, no network, no remote job. `/home/claude/work/fieldop_tsky/` was not touched.

Units: arcsec for lengths in formulas, mas (milliarcseconds) in tables.

## 1. Summary

* **The degeneracy, as coded.** Moving the kernel of each kernel class $c$ (one per filter and instrument) by a detector-frame vector
  $\delta_c$, moving every exposure's translation by $-R_e\delta_{c(e)}$ and the sky by any common $a$ (with the translations by $+a$)
  leaves every prediction unchanged, at any number of orientations and instruments. With the existing common-shift gauge (training
  translations of zero mean) this leaves $2N_{\rm class}$ free directions, and the sky moves along them by
  $a = \langle R_e\delta_{c(e)}\rangle_{\rm train}$. Two corrections to the brief: (i) because every exposure has its own free
  translation, a kernel shift that differs between filters is *also* a null direction of this model (it trades position between a filter's
  kernel and that filter's translations, without moving the sky); (ii) several orientations do not break the degeneracy; they decide how
  much of a detector-frame kernel drift reaches the sky. At one orientation (SMACS 0723) all of it does; at four balanced orientations
  almost none does. The one place where the model is not invariant is the blur rule (section 2.4).
* **Numerical check.** A physical kernel shift (the continuous kernel field evaluated at $v-\delta$) compensated through the translations
  changes predictions by the render's quadrature error only: $5\times10^{-5}$ of the signal (0.03 sigma) at a render pitch of 0.005 arcsec,
  against $1.2\times10^{-2}$ (1 mas) and $1.2\times10^{-1}$ (10 mas) without compensation. At the synthetic field's own render pitch
  (0.02 arcsec) the same quadrature error is 0.2 to 2 per cent; the render's own phase jitter changes predictions by 2.2 per cent rms.
* **The convention.** Every class's mean kernel (code at the pivot, no per-exposure term), as rendered, has zero first moment inside a
  Gaussian window of sigma $2\times$ its generic core width. It is applied by an exact projection after each joint optimiser step that
  puts the shift into the translations of that class and the training mean of those changes into the sky. The translation part is exact
  to float32 round-off ($4\times10^{-7}$ of the signal for a 20 mas projection); the sky part is exact up to the item rasteriser
  ($4\times10^{-4}$ of the signal per mas of sky shift; the per-step sky shift was up to about 0.5 mas in the first steps and 1 to 7
  microarcseconds at the end of the synthetic fits).
* **Bit identity.** Prototype against library: 30 steps off and on, 150 steps off and on, PASS (66 tensors off, 67 on, held-out
  chi-squared and loss logs equal). Off, the numbers are those recorded before the change.
* **Synthetic demonstration** (section 5, disjoint halves, 400 steps, two seeds). At one orientation, the part of the sky shift between
  halves that registration does not explain is (+3.1, -4.8) ± (2.3, 2.6) and (-8.1, -5.7) ± (2.1, 1.9) mas with the gauge off, equal to
  the kernel-frame difference within about 1 mas; with it on, (-0.3, 0.0) ± (2.0, 2.7) and (-2.3, -0.6) ± (2.0, 1.9) mas (reduced
  chi-squared against zero 7.4 off, 0.4 on). At four orientations the drift stays in the translations and neither setting shows it. The
  short synthetic fits do not reproduce a 17 to 20 mas kernel-frame shift between halves (at most about 5 mas, from detector-frame drifts
  of 1 to 16 mas per class).
* **Side effects** (section 6, 300 steps, two seeds). Held-out chi-squared +0.21 and +0.09 per cent, flux ratios within 0.002, final
  dilation within 0.03, against +1.4 and -7.4 per cent, 0.04 and 0.3 to 1.1 for another fit seed: inside the optimisation noise.
* **Open.** Section 9: real data; the blur rule's dependence on the network's origin (possibly the driver of the drift); a tie to catalogue
  stars.

## 2. The degeneracy

### 2.1 The model as coded

For exposure $e$ (filter $f(e)$, instrument $i(e)$, kernel class $c(e) = (f, i)$), pixel $\xi$, wavelength nodes $\lambda_q$:

```latex
m_e(\xi) = \frac1Q\sum_q w_q\,e^{\gamma_e(\lambda_q)}\int d^2u\;k_e(u)\;s\big(\Omega_e(\xi)+u,\lambda_q\big) + B_e(\xi),
\qquad
\Omega_e(\xi) = x_e(\xi) + M_e\,(x_e(\xi)-r_e) + t_e ,
```

```latex
k_e(u) = Z_e^{-1}\,\exp \ell\!\big(R_e^{\mathsf T}u/\sigma;\;h_e\big),\qquad h_e = \bar h_{c(e)} + \tau z_e ,
\qquad
s(x,\lambda) = \sum_n \Big[\sum_k c_{nk}\phi_k(\lambda)\Big]\,G(x-x_n;\Sigma_n).
```

$x_e(\xi)$ is the header position plus the fixed registration `shift0`, $M_e$ the per-exposure affine matrix, $t_e$ the learned
translation, $R_e$ the exposure's rotation (detector to sky, with parity), $\sigma$ the common-blur dilation, $\bar h_c$ the class code at
the filter's pivot wavelength (the same code at every $\lambda_q$ of the filter, so a class has one kernel at all its wavelengths),
$z_e$ the per-exposure code, $Z_e$ the normalisation (unit integral inside the normalisation radius), $x_n$ the item positions (one position
per item at all wavelengths). Existing gauges: throughput of the reference instrument, $\sum_{e\in T}t_e = 0$ (common shift, $T$ the
training exposures), background constants by the median rule, blur floor through $\sigma$, flux inside the normalisation radius.

### 2.2 The transformation

For detector-frame vectors $\delta_c$ (one per class) and a sky vector $a$:

```latex
k_e(u)\;\to\;k_e\big(u - R_e\delta_{c(e)}\big),\qquad x_n\to x_n + a,\qquad t_e\to t_e + a - R_e\delta_{c(e)}\quad(\text{every }e).
```

Then, with $v = u - R_e\delta_{c(e)}$,

```latex
\int d^2u\;k_e(u-R_e\delta)\;s\big(\Omega_e + a - R_e\delta + u - a\big) = \int d^2v\;k_e(v)\;s(\Omega_e+v)
```

for every exposure, pixel and wavelength node: every prediction is unchanged. Nothing else changes: $Z_e$ is an integral over the plane
(section 2.4 for the normalisation window), the throughput, backgrounds and noise terms do not see it, the translations have no prior
(the affine prior acts on $M_e$ only), the code prior acts on $z_e$, not on positions.

* **Orientations.** $R_e$ enters only through the per-exposure translation change, which is free per exposure: the identity holds for any
  set of orientations. What the orientations decide is the sky part once the common-shift gauge is imposed (below).
* **Per-filter shifts.** $\delta_c$ is free per class. A kernel shift that differs between filters is a null direction here because each
  exposure's translation absorbs it. It would be a real per-filter offset only in a model whose translations are shared across filters
  (one per visit or pointing); in this model a per-filter pointing offset and a per-filter kernel centroid cannot be told apart.
* **Wavelength.** The shift must be the same at every $\lambda_q$ of a class (positions and items are achromatic). It is, because a class's
  kernel is the same at all its nodes. A chromatic kernel shift within a filter is not a null direction.
* **Per-exposure codes.** In function space a per-exposure kernel shift $\varepsilon_e$ through $z_e$ with $t_e\to t_e - R_e\varepsilon_e$ is
  also null; the $N(0,1)$ prior on $z_e$ pins it only indirectly. The convention does not touch it.
* **Affine terms.** $M_e$ is not involved ($t_e$ enters additively after it). A kernel *magnification* would trade with $M_e$, but a shift
  does not.

### 2.3 What remains free under the existing gauges

The common-shift gauge $\sum_T t_e = 0$ fixes $a$:

```latex
a(\delta) = \frac{1}{|T|}\sum_{e\in T} R_e\,\delta_{c(e)} = \sum_c \frac{n_c}{|T|}\,\bar R_c\,\delta_c ,\qquad \bar R_c = \text{mean } R_e \text{ over the training exposures of class } c .
```

So $2N_{\rm class}$ directions remain free and the sky follows them by $a(\delta)$. At one orientation ($\bar R_c = R$ for all $c$) a common
detector-frame drift $\delta$ moves the sky by $R\delta$, all of it: this is the SMACS 0723 situation (six bands, one orientation), and a
fit-to-fit difference of this drift is a common sky shift in every band. At four orientations used equally $\bar R_c \approx 0$ and the
same drift lands in the translations, not in the sky. Holding one reference class does not fix the sky: the other classes still move it
by their share $n_c\bar R_c\delta_c/|T|$.

### 2.4 Where the coded model is not exactly invariant

1. **The blur rule.** The dilation $\sigma$ is set from the core second moment of the blur-reference class's kernel **about the network's
   origin** (`blur_moment`). A kernel moved off the origin by $\delta$ has a larger moment about the origin (by about $|\delta|^2$ inside
   the window), so when the floor is active ($\sigma>1$) a shift of that class changes $\sigma$ and so every kernel. Measured on the cached
   synthetic state (floor $1.2\times10^{-3}$ arcsec$^2$, $\sigma = 1.174$): a shift of 1 mas changes $M_{\rm ref}$ by $-1.4$ per cent and
   $\sigma$ to 1.183; 10 mas, $\sigma = 1.156$; 20 mas, $\sigma = 1.000$ (floor no longer active). With the floor inactive the
   direction is null. With it active it is not null, and the rule rewards an off-centre kernel (it may then be sharper for the same
   floor), which is a candidate *driver* of the drift, not only a lack of restoring force (section 9).
2. **The render's quadrature.** The kernel is sampled on fixed nested grids and convolved on them; a kernel shift by a fraction of the grid
   pitch is not a translation of the sampled problem. Section 3 measures it.
3. **Windows centred on the origin.** The level windows, the extent taper and the normalisation window $n(r)$ (radius 2 arcsec on real
   data) are centred on $u=0$. The normalisation changes at first order by $\delta\cdot\int k\,\nabla n$, zero for a centro-symmetric
   kernel and otherwise tiny (the kernel at 2 arcsec times a few mas).
4. **The item rasteriser.** Items are rasterised onto a grid read through a tent interpolant, so moving every item by $a$ is not an exact
   translation of the continuous sky: $4\times10^{-4}$ of the signal per mas on the synthetic field (sky pitch 0.06 arcsec).

## 3. Numerical check of the degeneracy

Script: `/tmp/claude-0/-home-claude/286b9f86-e33d-5aff-95e8-42cb96d595fe/scratchpad/check_degeneracy.py` (and `check_quadrature.py`,
`probe_blur.py` there; scratch, not part of either code base). State: the synthetic field of `compare_port.py` (two instruments, three
filters, 8 exposures each, render pitch 0.02 arcsec, sky pitch 0.06 arcsec) after warm start, registration and 30 steps with the gauge on;
one version with exposures at four orientations, one at a single orientation. The kernel shift is **physical**: a wrapper on the network's
encoding evaluates $\ell((v-\delta)/\sigma)$, so the render samples a moved kernel. Predictions are every pixel of every patch of every
exposure, no render jitter, dilation held. Numbers: rms of the change over rms of the signal, and the largest change in units of the pixel
sigma.

Four orientations (dilation 1.174):

| transformation | 1 mas | 10 mas | 20 mas |
|---|---|---|---|
| one class shifted, its translations compensate | 2.2e-3 (0.7 sigma) | 8.6e-3 (4.7) | 1.2e-2 (4.9) |
| every class shifted differently, translations compensate | 5.5e-3 (2.1) | 2.0e-2 (7.4) | 2.5e-2 (13.5) |
| same, training-translation mean held, the sky takes it | 5.5e-3 (2.1) | 2.0e-2 (7.8) | 2.5e-2 (13.8) |
| sky and every translation moved together (existing null direction) | 4.4e-4 (0.3) | 3.8e-3 (2.2) | 6.0e-3 (2.1) |
| control: one class shifted, nothing compensates | 4.7e-3 (1.6) | 5.3e-2 (16.9) | 1.0e-1 (33.3) |
| control: common detector-frame shift compensated by the sky alone | 1.5e-2 (6.9) | 1.5e-1 (58) | 3.0e-1 (126) |
| control: one class shifted, compensated by the sky alone | 1.0e-2 (3.0) | 1.0e-1 (30) | 2.1e-1 (64) |

One orientation (dilation 1.196): compensated by the translations 3.2e-3, 1.1e-2, 7.1e-3 (one class) and 8.2e-3, 2.4e-2, 1.8e-2 (every class);
common detector-frame shift compensated by the sky alone 7.2e-3, 2.5e-2, 1.8e-2, i.e. at one orientation that is also a null direction (the
same level as the translation-compensated ones), whereas at four orientations it is 15 to 30 times larger.

The residual of the compensated rows is the render's quadrature error, not a failure of the identity. Same state, same 10 mas shift of
every class, finer render grids (`check_quadrature.py`; the encoding's core radius held at 0.01 arcsec):

| render pitch | compensated, 1 mas | compensated, 10 mas | uncompensated, 1 mas | uncompensated, 10 mas |
|---|---|---|---|---|
| 0.02 arcsec | 5.5e-3 (2.1 sigma) | 2.0e-2 (7.4) | 1.2e-2 | 1.3e-1 |
| 0.01 arcsec | 2.1e-3 (0.6) | 2.2e-3 (0.9) | 1.2e-2 | 1.2e-1 |
| 0.005 arcsec | **6.7e-5 (0.03)** | **5.2e-5 (0.03)** | 1.2e-2 | 1.2e-1 |

So the identity holds to $5\times10^{-5}$ when the quadrature resolves the learned kernel, and the synthetic field's render (pitch 0.02
arcsec for a kernel core of about the same width) carries a phase error of the same order as the signal of a 1 mas shift: drawing the
level-0 grid's phase at random, as training does, changes predictions by 2.2 per cent rms (11 sigma at most). On real data the render
pitch is the detector pixel divided by 6 (`--render-os-pix`), finer relative to the kernel.

## 4. The convention

### 4.1 What is held

For each kernel class $c$ (every (filter, instrument) pair of the exposures, held out ones included), with $d_c$ the class's offset:

```latex
\mu_c(d_c) \;=\; \frac{\sum_j w_j\,W_c(w_j)\,\hat k_c(w_j-d_c)}{\sum_j W_c(w_j)\,\hat k_c(w_j-d_c)} \;=\; 0,
\qquad \hat k_c(w) = \exp\ell\big(w/\sigma;\,\bar h_c\big),\qquad W_c(w) = \exp\!\big(-|w|^2/2\rho_c^2\big),
```

```latex
\rho_c = 2\sqrt{\big(0.42\,\lambda_{\rm piv}/D\big)^2 + p^2/12},\qquad w_j \in \{\rho_c\,(i,j)/8 : |i|,|j|\le 32\}\ \ (\pm 4\rho_c).
```

* **Which kernels.** The class's mean kernel: the code at the pivot wavelength with no per-exposure term ($z=0$), the same kernel the blur
  rule uses. Every class separately, so all $2N_{\rm class}$ null directions of section 2.3 are removed and nothing else (each projection
  moves only along them). There is no reference class: with each class held, the sky frame is fixed by the common-shift gauge and the
  header positions, $a = 0$ by construction.
* **Which frame.** Detector frame, physical arcsec, the kernel as rendered (after the dilation, with its offset).
* **Wavelength weights.** None needed: a class's kernel is the same at all its wavelength nodes (code at the pivot), so the moment at every
  node is the same and any weighting gives this one.
* **Window.** A Gaussian of sigma $\rho_c$: twice the generic core width of the class (diffraction $0.42\lambda/D$ and the pixel box,
  inputs only, the formula of the blur floor and of the warm start). On SMACS: F090W 0.030 arcsec, F444W 0.12 arcsec;
  synthetic: 0.065, 0.076, 0.094 arcsec. The moment is computed on the continuous kernel field on its own fine grid (pitch $\rho_c/8$),
  not on the render's level grids, so it does not depend on the render pitch or the level layout.
* **Why it is well defined with wings to 10 arcsec.** The window weight at 1 arcsec is $e^{-0.5(1/0.12)^2}\approx10^{-15}$ even for
  F444W: the held moment depends on the core only, not on the wings, the extent taper, the level hand-overs or the normalisation radius.
  An unwindowed first moment would not be: for wings falling as $r^{-3}$ the integrand $r\,k(r)\,r\,dr$ falls only as $r^{-1}$, so the
  plain centroid is set by the far wings and the truncation at the kernel extent. The condition $\mu_c=0$ is the fixed point of the
  usual re-centred windowed centroid; its Jacobian $\partial\mu_c/\partial d_c$ is close to $\rho_c^2/(\rho_c^2+s^2)\,I$ for a core of
  width $s$ (about 0.8 here), far from singular, and it is solved exactly by Newton steps (below).

### 4.2 The projection

Each class has an offset $d_c$ (a persistent buffer, zero at the start). The rendered kernel is the network's kernel moved by $d_c$ in the
detector frame. The render applies it as the equivalent shift of its read positions, $\Omega_e \to \Omega_e + R_e d_{c(e)}$ (exactly
equivalent in the continuum by section 2.2, and free of the quadrature's phase error that moving the kernel samples would bring). After the
joint step of every optimiser iteration (after the operator, nuisance and sky updates, the pedestal and common-shift projections):

```latex
d_c^{+} = d_c - \sum_{\rm 4\ Newton\ steps} J_c^{-1}\mu_c,\quad J_c = \partial\mu_c/\partial d_c\ (\text{autograd, }2\times2);\qquad \delta_c = d_c^+ - d_c ,
```

```latex
\Delta t_e = -R_e\,\delta_{c(e)},\qquad \bar a = \frac1{|T|}\sum_{e\in T}\Delta t_e,\qquad
t_e \leftarrow t_e + \Delta t_e - \bar a\ \ (\text{every } e),\qquad x_n \leftarrow x_n - \bar a .
```

This is the transformation of section 2.2 with $\delta_c$ and $a=-\bar a$: function-preserving; the training translations keep zero mean
exactly. Like the other projections it leaves the optimiser's moment estimates untouched. It changes no prior and constrains nothing in
the response's shape: the network itself is not modified.

### 4.3 Interactions

* **Common-shift gauge.** Unchanged and still exact; together the two fix the frame: the header frame on average over the training
  exposures, with centred kernels. (Without the new gauge the frame is the header frame minus the sky-frame mean kernel centroid, which is
  free.)
* **Blur rule.** The offset is outside it: the blur moment is still evaluated on the network's kernel about the network's origin, so the
  projection does not change $\sigma$ and stays function-preserving. The dilation is computed fresh in the projection, from the updated
  network, so the moment is held for the kernel the next step renders. But the network itself can still move away from its own origin (the
  offset follows it), and with the floor active the blur rule still rewards that (section 2.4). The convention fixes the frame; it does not
  make the blur rule translation-invariant.
* **Normalisation radius.** No interaction: the window is 0.03 to 0.2 arcsec, the radius 2 arcsec; the flux convention is untouched.
* **Several orientations or instruments.** The degeneracy is exact there too (section 2.2), so the convention removes a null direction there
  as well, not only selects a frame. What differs is how much a drift had moved the sky (section 2.3).
* **Where it only selects a frame.** For the blur-reference class when the floor is active, the direction is not exactly null in the
  model; the convention still holds that class's rendered kernel centred, i.e. it picks one frame among nearly equivalent ones, while the
  blur rule keeps its own (origin-dependent) behaviour. The quadrature error (section 3) plays no role in the projection, which moves
  read positions, not kernel samples.
* **Several tiles.** Not implemented: the fit refuses to start with the setting on and more than one tile (the sky part would have to move
  every tile's sky per shift component). The library is one tile only anyway.

### 4.4 Is it worth holding every class rather than the common part only?

Holding only the $n_c$-weighted mean $\sum_c n_c\bar R_c\delta_c=0$ would fix the sky with two numbers and leave the per-class kernel and
translation split free. Holding every class costs nothing more, also centres every band's kernel product (`SkyFit.response.kernel` returns
the rendered, moved kernel), and makes the reported per-exposure translations the positions of the light centroid. At one orientation the
two choices fix the same sky frame.

## 5. Synthetic demonstration: disjoint halves, off and on

Script: `/home/claude/work/fieldop_min/tests/demo_kernel_centroid_halves.py` (not collected by pytest; about 10 minutes per fit on this
machine). Field of `compare_port.py`; per filter the held-out exposure is shared and the other 7 training exposures alternate between
half 0 (4) and half 1 (3); 400 steps; seeds 0 and 1 (field and fit). Per fit: $F$ = common offset of the fitted sky against the truth
(iterated windowed centroids at isolated true point sources, minus the same estimator on the true sky rasterised on the same grid;
12 source-band measurements); $a_{\rm reg}$ = training mean of $(\text{shift0}_e + t_e - \text{true shift}_e)$, the frame set by registration
and the translation gauge; $\kappa$ = training mean of $R_e\mu_{c(e)}$, the sky-frame windowed centroid of the rendered kernels (zero with the
gauge on). Model: $F = a_{\rm reg} + \kappa$ (the true kernel is centred). Between halves $\Delta F = \Delta a_{\rm reg} + \Delta\kappa$.

Between-half differences (half 0 minus half 1) [mas], x and y; $\Delta F$ error from the scatter of the 12 per-source differences:

| orientations | seed | gauge | $\Delta F$ (sky shift between halves) | $\Delta a_{\rm reg}$ (registration frame) | $\Delta\kappa$ (kernel frame) | $\Delta F - \Delta a_{\rm reg}$ |
|---|---|---|---|---|---|---|
| one (as SMACS) | 0 | off | +2.3 ± 2.3, +6.8 ± 2.6 | -0.8, +11.6 | +3.7, -4.6 | **+3.1 ± 2.3, -4.8 ± 2.6** |
| one | 0 | on | -1.1 ± 2.0, +11.6 ± 2.7 | -0.8, +11.6 | 0, 0 | **-0.3 ± 2.0, 0.0 ± 2.7** |
| one | 1 | off | +15.4 ± 2.1, +3.6 ± 1.9 | +23.5, +9.2 | -5.1, -4.5 | **-8.1 ± 2.1, -5.7 ± 1.9** |
| one | 1 | on | +21.2 ± 2.0, +8.6 ± 1.9 | +23.5, +9.2 | 0, 0 | **-2.3 ± 2.0, -0.6 ± 1.9** |
| four | 0 | off | -8.7 ± 2.4, +12.8 ± 1.4 | -9.9, +12.7 | +0.7, -0.3 | +1.2 ± 2.4, +0.1 ± 1.4 |
| four | 0 | on | -9.7 ± 2.4, +13.1 ± 1.4 | -9.9, +12.7 | 0, 0 | +0.2 ± 2.4, +0.4 ± 1.4 |
| four | 1 | off | -2.7 ± 2.5, +18.4 ± 3.5 | -4.1, +19.6 | -0.1, +0.6 | +1.4 ± 2.5, -1.2 ± 3.5 |
| four | 1 | on | -2.6 ± 2.5, +17.8 ± 3.5 | -4.1, +19.6 | 0, 0 | +1.5 ± 2.5, -1.8 ± 3.5 |

Reading:

1. **The free direction moves in synthetic fits too.** With the gauge off, the windowed centroid of the network's kernels drifts in the
   detector frame by 1 to 16 mas in 400 steps (per class, e.g. seed 1 half 0 at one orientation: A1 -15.9, -4.6; A2 -8.6, -3.4; B1 -4.7,
   -3.4 mas), differently in each half. With the gauge on the offsets that compensate it are of the same size (up to 17 mas) and the
   rendered kernels stay centred (held moment after the last projection at most $9\times10^{-6}$ mas).
2. **At one orientation the drift reaches the sky and the gauge removes it.** Off, the part of the half-to-half sky shift that the
   registration does not explain is (+3.1, -4.8) and (-8.1, -5.7) mas, the latter 3 to 4 times its error, and it equals the kernel-frame
   difference $\Delta\kappa$ to about 1 mas in each component ((+3.7, -4.6) and (-5.1, -4.5) mas). On, it is (-0.3, 0.0) and
   (-2.3, -0.6) mas, within about one error of zero. Pooled over the two seeds, the reduced chi-squared of $\Delta F-\Delta a_{\rm reg}$
   against zero is 7.4 off and 0.4 on (4 numbers each).
3. **At four orientations the drift stays in the translations** ($\Delta\kappa$ below 0.7 mas off) and the gauge changes nothing visible,
   as section 2.3 predicts.
4. **What the synthetic test does not show.** It does not reproduce a 17 to 20 mas sky shift between halves from the kernel alone; its
   largest kernel-frame difference is about 5 mas in 400 steps. The large raw $\Delta F$ here (up to 23 mas) is the *registration* frame:
   each half's translations have zero mean, and the halves' exposures have independent true pointing errors (30 mas rms per exposure,
   so 10 to 15 mas between half means). That part is a property of the common-shift convention and is untouched by the new gauge; on the
   SMACS data it is excluded by the 37 shared exposures, which register to both skies within 3 mas (a registration-frame difference
   would show up in them, a kernel-frame one does not).
5. Errors are the scatter of 12 per-source differences (6 isolated point sources in 2 or 3 bands, not independent between bands), so
   they are approximate. The model $F = a_{\rm reg}+\kappa$ holds per fit to 0 to 3 mas (a common estimator offset, the same off and on).
6. The fits used the synthetic field's coarse render (pitch 0.02 arcsec) and reached dilations of 1.2 to 2.5, larger than real fits (1.0
   to 1.4).

## 6. Side effects at equal steps

Full synthetic field (all 7 training exposures per filter, four orientations), 300 steps, the held-out stage as configured; same field and
fit seed off and on, plus one more fit seed with the gauge off on the same field (fit seed 7) as the scale of the optimisation noise.
Script: `/tmp/claude-0/-home-claude/286b9f86-e33d-5aff-95e8-42cb96d595fe/scratchpad/sidefx.py`. Held-out chi-squared: mean squared
residual over the input sigma, pixel-weighted over the three held-out exposures. Flux ratio: aperture flux (radius 0.2 arcsec, local
annulus subtracted) on the fitted band image over the true band flux, median of 4 isolated point sources per band (a relative measure: the
aperture misses the wings, the same off and on).

| field seed | run | held-out chi-squared | per exposure (A1, A2, B1) | flux ratio A1, A2, B1 | final dilation (max) | final batch loss |
|---|---|---|---|---|---|---|
| 0 | off | 2.6986 | 4.789, 1.615, 1.304 | 0.9332, 0.8786, 0.8350 | 3.015 (3.293) | 0.8563 |
| 0 | **on** | 2.7042 | 4.801, 1.619, 1.305 | 0.9330, 0.8784, 0.8343 | 2.988 (3.253) | 0.8565 |
| 0 | off, fit seed 7 | 2.7360 | 4.975, 1.515, 1.327 | 0.9274, 0.8830, 0.8370 | 1.911 (2.143) | 0.9013 |
| 1 | off | 2.9670 | 5.112, 1.683, 1.776 | 0.9531, 0.9108, 0.9107 | 1.373 (1.385) | 0.8293 |
| 1 | **on** | 2.9698 | 5.118, 1.684, 1.776 | 0.9526, 0.9089, 0.9093 | 1.348 (1.356) | 0.8298 |
| 1 | off, fit seed 7 | 2.7479 | 4.791, 1.698, 1.374 | 0.9542, 0.9304, 0.9483 | 1.677 (2.147) | 1.3435 |

On against off: held-out chi-squared +0.21 and +0.09 per cent, flux ratios within 0.002, dilation within 0.03. Another fit seed on the same
field changes the held-out chi-squared by +1.4 and -7.4 per cent, the flux ratios by up to 0.04 and the dilation by 0.3 to 1.1. So the
gauge's effect on held-out prediction, fluxes and the blur dilation is well inside the optimisation noise, as a pure gauge should be.
(The on and off fits stay close because they share every random number; the projections are small perturbations along a null direction.)
In these fits the projection steps were up to 1.6 mas per class early in the fit and 0.01 mas per step at the end (rms 0.29 mas over 300
steps), the sky shifts it applied 0.5 to 7 microarcseconds per step at the end and 1 to 10 mas in total; the held moment after a
projection stayed below $10^{-5}$ mas.

## 7. Implementation

Same numerical code in both places (the library's is the prototype's block, formatted by ruff). Lines are those of the files as left.

| what | prototype (`/home/claude/work/fieldop_min/fieldop/`) | library (`/home/claude/torchsky/src/torchsky/map/exposure_fit/`) |
|---|---|---|
| setting, default 0 | `cli.py` 121-124 (`--kernel-centroid-gauge {0,1}`) | `_config.py` 66 (`"kernel_centroid_gauge": 0` in `_INTERNAL_SETTINGS`) |
| switch in the operator (nothing made when off) | `model.py` 165-168 | `_response.py` 266-269 |
| classes, offsets `kc_off` (persistent buffer), window, grid | `model.py` 216-243 (`KC_NEWTON`, `KC_WINDOW`, `_init_kernel_centroid`) | `_response.py` 318-363 |
| held moment $\mu_c$ | `model.py` 245-259 (`kernel_centroid`) | `_response.py` 365-383 |
| Newton solve | `model.py` 261-274 (`kernel_centroid_solve`) | `_response.py` 385-403 |
| projection (translations, sky, statistics on the device) | `model.py` 276-312 (`project_kernel_centroid`) | `_response.py` 405-443 |
| record for the log and the report | `model.py` 314-326 (`kernel_centroid_report`) | `_response.py` 445-470 |
| render: read positions moved by $R_e d_c$ | `model.py` 478-479 (`render_batch`) | `_response.py` 663-664 |
| gauge report: held moment and offsets per class | `model.py` 597-602 (`gauge_report`) | `_response.py` 859-871 |
| fit loop: refusal with several tiles; projection after the joint step; log line; end-of-fit record and line | `train.py` 386-390, 499-500, 537-541, 560-567 | `_fit.py` 522-527, 663-664, 724-729, 758-771 |
| fit report | `metrics.json`: `fit.kernel_centroid_gauge`, `gauges.kernel_centroid_*` | `_skyfit.py` 609-613: `SkyFit.report["kernel_centroid_gauge"]` (only when on) |
| kernel product includes the offset | (no reader in the prototype) | `_readers.py` 88-90 (`kernel_stamp`) |
| approximation list | `__init__.py`, entry F43 | `docs/models/exposure_fit.md`, Stated Approximations |

The log line at every logged step (with the setting on) adds `kcen held <largest held moment after the last projection> mas step <largest
class step of the last projection> (max <largest so far>) mas sky-shift total [x, y] mas`. At the end of the fit one line gives, per class,
the offset $d_c$ [mas] and the window, then the number of projections, the largest and rms class step, the held moment after the last
projection and the total sky shift applied. `rec["kernel_centroid_gauge"]` (prototype `metrics.json` under `fit`; library
`SkyFit.report`) holds the same numbers. The statistics stay on the device and are read only at logged steps. The setting is saved with a
fit (library: in the manifest's private settings; prototype: in `metrics.json` `config`), and a state saved with it on carries `kc_off`, so
a reloaded fit (and the uncertainty step, which renders through `render_batch`) renders the same kernels. A state saved with the setting on
cannot be loaded with it off (unexpected key), which is intended.

Cost per step: per class four Newton steps, each one forward pass of the kernel network on 4,225 points and two backward passes for the
2 x 2 Jacobian, plus one more forward pass for the residual. On the synthetic CPU fit this was not measurable against the 1.2 s step.

`compare_port.py` (`/home/claude/work/port_check/`) has a new switch `--kernel-centroid-gauge {0,1}`, which sets the option on both sides and
also requires the two end-of-fit gauge records to be equal.

## 8. Tests and checks

**New tests.**

* Prototype `tests/test_kernel_centroid_gauge.py` (3 tests, 12 s): (1) off by default, nothing made (no `kc_off`, no `kc_*` in the state);
  (2) on a small field with four orientations: offsets moved by ±20 mas through the translations change predictions by less than
  $2\times10^{-6}$ of the signal (measured $4\times10^{-7}$), the translation part of the projection likewise, the held moment afterwards
  below $10^{-7}$ arcsec for every class; the full projection after a step-sized perturbation (0.05 mas per class) changes predictions by
  less than $10^{-5}$ and the training-translation mean by less than $10^{-9}$ arcsec; a 1 mas kernel offset alone changes them by more
  than $10^{-4}$ (the check can fail); (3) a 6-step fit with the gauge on projects 6 times and reports the record, held moment below
  $10^{-4}$ mas in the gauge report.
* Library `tests/test_exposure_fit_centroid_gauge.py` (4 tests, 15 s): off by default (`_INTERNAL_SETTINGS`), nothing made and no report
  entry; a 6-step fit with the setting on reports the gauge, every class's held moment below $10^{-7}$ arcsec, and the kernel stamp of
  `SkyFit.response.kernel` (the rendered, moved kernel) has a windowed centroid below 2 mas on its own 0.04 arcsec quadrature; the
  projection preserves predictions (as in the prototype); save and load keep the offsets.
* `tests/test_exposure_fit_config.py`: the pinned private settings gain `kernel_centroid_gauge: 0` (81 entries).
* The synthetic demonstration (section 5) is `tests/demo_kernel_centroid_halves.py` in the prototype (not collected; 10 minutes per fit).

**Off changes nothing.** The off path makes nothing (no buffer, no attribute read by the render, no random number drawn), and:

* `compare_port.py --steps 30`: `66 identical, 0 different ... PASS`, held-out chi-squared 28.230878829956055, 2.888465166091919,
  3.076887845993042 (as before the change), loss log equal, `RESULT: PASS`.
* `compare_port.py --steps 150 --reference-state .../reference_minimal_items150_state.pt`: prototype against library `66 identical,
  0 different ... PASS`; recorded reference (the unpruned package's run) against the prototype `66 identical, 0 different ... PASS`;
  `RESULT: PASS`.
* The golden steps (`tests/test_oracle_golden.py`, loss and every gradient to $10^{-12}$ relative) pass in the full prototype run below.

**On: the two code bases stay bit-identical.** `compare_port.py --steps 30 --kernel-centroid-gauge 1`: `67 identical, 0 different, 67
reference and 67 new tensors -> PASS`, held-out chi-squared equal (28.23281478881836, 2.8885693550109863, 3.077021360397339), loss log
equal, end-of-fit gauge records equal, `RESULT: PASS`. `--steps 150 --kernel-centroid-gauge 1`: `67 identical, 0 different ... PASS`,
held-out chi-squared equal, records equal, `RESULT: PASS`.

**Suites.**

* Prototype, full suite (not a subset): `python -m pytest tests -q -x --ignore=tests/truth` -> `111 passed, 13 warnings in 639.42s
  (0:10:39)` (run while two fits were using the other core; 7:18 alone on 19 Nov).
* Library: `python -m pytest tests/test_exposure_fit_*.py -q` -> `65 passed, 10 warnings in 203.59s (0:03:23)`.
* Ruff 0.15.20 in the library: `python -m ruff check .` -> `All checks passed!`; `python -m ruff format --check` on the seven touched
  files -> `7 files already formatted`. No variable named `l` was added.
* All Python was run as `PYTHONNOUSERSITE=1 PYTHONPATH=/home/claude/torchsky/src python ...` as asked (not through Pixi).

## 9. Real data: the run, what it should show, and what remains open

**The real-data run.** The package to run is this folder's `fieldop` (the frozen `fieldop_tsky` does not have the option), packed as the
usual versioned tarball. The two half fits of 19 November are repeated with one extra argument and nothing else changed:

```
--kernel-centroid-gauge 1
```

i.e. `-e "ARGS=<the 19 Nov arguments, including --train-half 0> --kernel-centroid-gauge 1"`, and the same with `--train-half 1`. The fit's
log then carries `kcen held ... step ... sky-shift total [x, y] mas` on every logged line and, at the end, one line `kernel-centroid gauge:
F090W/0 offset [x, y] mas (window 0.030 arcsec); ...`; `metrics.json` holds the record under `fit.kernel_centroid_gauge` and the held moments
under `gauges`. The extra cost is a few tens of small kernel-network evaluations per step (six classes), small against the render. The
products and `diff-files` (mode halves) are made as on 19 November.

**What it should show.** In `diff-files` on the two halves, `FRAMEDX`, `FRAMEDY` (-17 to -20 mas in x and +3 to +7 mas in y on 19 November,
common to all bands) going to within their error of zero, while the 37 shared exposures still register to both skies within a few mas
(they did before; with the gauge, kernels and sky no longer move together, so a remaining frame difference would show up there). The
offsets $d_c$ at the end of each half fit say how far the network's kernels had drifted: at one orientation, the difference of their
rotated means between the halves is the frame difference the gauge removed and should be about the 19 November `FRAMEDX`, `FRAMEDY`.
If instead the frame difference stays, it is a registration-frame difference (the zero-mean convention over different exposure sets)
or something else, not this gauge's direction. The flux-scale and width terms (`ZPDIFF`, `WIDTHDIF`) are not expected to change: they
are not this direction.

**Every approximation.**

1. The convention holds the windowed centroid of each class's mean kernel (code at the pivot, no per-exposure code term). Per-exposure
   kernels can still move against their translations through their codes (a function-space null direction held only by the code prior).
2. The window (sigma $2\times$ the generic diffraction and pixel width) is a choice; another window gives another, equally valid frame,
   differing from this one by the kernel's asymmetry inside the core (for a symmetric core, none). It is fixed from inputs only.
3. The held moment is computed on a fixed grid (pitch $\rho_c/8$ to $\pm4\rho_c$); round-off of the moment about $10^{-8}$ arcsec.
4. Four Newton steps per projection, not iterated to a tolerance (no device read in the loop): the held moment after a projection was below
   $10^{-5}$ mas in every fit here, and below $10^{-4}$ mas after an artificial 20 mas jump (the test's bound).
5. The projection's sky part (moving every item by $\bar a$) is exact up to the item rasteriser ($4\times10^{-4}$ of the signal per mas of
   sky shift on the synthetic field); in a fit $\bar a$ is 1 microarcsecond to 1 mas per step.
6. The render applies the offset as a shift of its read positions, which is the kernel shift exactly in the continuum; a kernel-sample
   shift would differ from it by the render's quadrature phase error (section 3), which the model has anyway.
7. The blur rule still measures the network's kernel about the network's origin (section 2.4).
8. One tile only (the fit refuses otherwise).
9. The synthetic demonstration has 12 source-band measurements per fit and two seeds; its errors are approximate (section 5).

**Open.**

1. The real-data check above (a GPU job; not run here).
2. **The blur rule and the drift.** With the floor active, the blur rule's moment about the network's origin makes an off-centre kernel
   cheaper (it can be sharper for the same floor): this may be what drives the network's kernels off their origin (1 to 16 mas in 400
   synthetic steps, and the 17 to 20 mas between the SMACS halves). The gauge removes the consequence for the frame, not the cause. Making
   the blur rule use the moment of the rendered, centred kernel (evaluate `blur_moment` with the offset) would make it translation
   invariant; it is a change to the blur convention, and with the floor active the projection would then no longer be exactly
   function-preserving (the dilation would follow the offset). Not done; worth a separate test.
3. **A tie of the frame to catalogue stars** (allowed by the owner): with the kernels centred, the sky frame is the header frame averaged
   over the training exposures (the common-shift gauge). Replacing that average by a fit of a common sky offset to reference-catalogue
   stars (for example Gaia, with proper motions to the epoch) would tie the absolute frame, and would also remove the registration-frame
   part that differs between disjoint halves (section 5, point 4).
4. Several tiles: the sky part of the projection would have to move every tile's sky, per connected component of the common-shift gauge.
5. Whether to hold only the $n_c$-weighted common part instead of every class (section 4.4): equivalent for the sky frame at one
   orientation; the per-class choice also centres the band kernels and the translations.
6. Documentation: `docs/models/exposure_fit.md` (Stated Approximations) has three sentences, stating the setting is off until validated
   on real data. No changelog entry (the setting is private and off).
