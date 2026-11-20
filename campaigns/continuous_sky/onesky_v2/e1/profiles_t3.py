"""T3 analysis: match items of the frozen-profile fits of sky 10 across zooms (scene coordinates) and compare
log size and latent. Matching rule (declared): an item at zoom 1 with flux S/N proxy flux > FMIN counts and
size > 1 px is matched to the nearest item at zoom z within 0.5 x its size (scene units) whose flux is within a
factor 1.5 and whose size is within a factor 2; each zoom-z item used at most once."""
import json
import math
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

FMIN = 200.0
n = 256


def load(z):
    it = json.load(open(os.path.join(HERE, 'results', f'profiles_items__profile_frozen_lattice__15000__seed10_zoom{z}__noisy.json')))
    x = common.L / 2 - n / (2 * z) + np.array(it['x']) / z
    y = common.L / 2 - n / (2 * z) + np.array(it['y']) / z
    ls_pix = np.array(it['logsize'])
    return dict(x=x, y=y, ls_pix=ls_pix, size_scene=np.exp(ls_pix) / z, flux=np.array(it['flux']), z=np.array(it['z']),
                w1=np.array(it['w1']), w2=np.array(it['w2']))


ref = load(1.0)
out = {}
zl = ref['z']
rng = np.random.default_rng(0)
ii = rng.integers(0, len(zl), 4000); jj = rng.integers(0, len(zl), 4000)
spread_all = float(np.median(np.linalg.norm(zl[ii] - zl[jj], axis=1)))
for zoom in (1.4, 2.0, 2.8):
    o = load(zoom)
    half = n / (2 * zoom); lo, hi = common.L / 2 - half + 2 / zoom, common.L / 2 + half - 2 / zoom
    sel = np.where((ref['flux'] > FMIN) & (np.exp(ref['ls_pix']) > 1.0) & (ref['x'] > lo + ref['size_scene'] * 3) &
                   (ref['x'] < hi - ref['size_scene'] * 3) & (ref['y'] > lo + ref['size_scene'] * 3) & (ref['y'] < hi - ref['size_scene'] * 3))[0]
    tree = cKDTree(np.c_[o['x'], o['y']]); used = set(); pairs = []
    for i in sel[np.argsort(-ref['flux'][sel])]:
        cand = tree.query_ball_point([ref['x'][i], ref['y'][i]], 0.5 * ref['size_scene'][i])
        best = None
        for j in cand:
            if j in used: continue
            fr = o['flux'][j] / ref['flux'][i]; sr = o['size_scene'][j] / ref['size_scene'][i]
            if 1 / 1.5 < fr < 1.5 and 0.5 < sr < 2:
                d = math.hypot(o['x'][j] - ref['x'][i], o['y'][j] - ref['y'][i])
                if best is None or d < best[0]: best = (d, j)
        if best: used.add(best[1]); pairs.append((i, best[1]))
    pairs = np.array(pairs).reshape(-1, 2)
    if len(pairs) == 0:
        out[zoom] = dict(n_candidates=int(len(sel)), n_matched=0); continue
    a, b = pairs[:, 0], pairs[:, 1]
    dls = o['ls_pix'][b] - ref['ls_pix'][a]
    dz = np.linalg.norm(ref['z'][a] - o['z'][b], axis=1)
    perm = rng.permutation(len(b)); dz_null = np.linalg.norm(ref['z'][a] - o['z'][b[perm]], axis=1)
    out[zoom] = dict(n_candidates=int(len(sel)), n_matched=int(len(pairs)), log_zoom=math.log(zoom),
                     dlogsize_pix_median=float(np.median(dls)), dlogsize_pix_iqr=[float(np.percentile(dls, 25)), float(np.percentile(dls, 75))],
                     latent_dist_median=float(np.median(dz)), latent_dist_null_shuffled_matches=float(np.median(dz_null)),
                     latent_spread_all_items_median_pair=spread_all, ratio_matched_to_spread=float(np.median(dz) / spread_all))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, 'results', 'profiles_t3_matching.json'), 'w'), indent=1)
for z in (1.0, 1.4, 2.0, 2.8):
    r = json.load(open(os.path.join(HERE, 'results', f'profile_frozen_lattice__15000__seed10_zoom{z}__noisy.json')))
    print(z, 'chi2_truth', round(r['chi2_truth'], 4), 'psnr', round(r['psnr'], 2))
