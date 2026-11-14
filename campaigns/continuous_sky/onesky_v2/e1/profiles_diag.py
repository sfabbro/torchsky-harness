"""Diagnostic (not a budget-matched run): separate the effect of item count from the effect of the learned profile
on the noise-free seed-0 image at the 15000 budget. (a) frozen joint psi with the SAME number of items as the
Gaussian (2499, i.e. 4 more numbers per item than the budget allows); (b) fixed Gaussian with the learned run's
item count (1384)."""
import sys, os, json, torch, numpy as np
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profiles as P, common
d = common.make_data(common.make_spec(0), n=256)
case = sys.argv[1]
if case == 'frozen2499':
    psi = P.Psi('learned'); psi.load_state_dict(torch.load(P.PSI_FILE)); N = 2499
else:
    psi = P.Psi('gauss'); N = 1384
items, imgs, info = P.fit([d], N, psi, False, False, 'truth', 220.0)
r = common.score(imgs[0], d)
print(case, N, 'chi2_truth', round(r['chi2_truth'], 4), 'iters', info['iterations'], 'falling', info['still_falling'])
json.dump(dict(case=case, n_items=N, chi2_truth=r['chi2_truth'], psnr=r['psnr'], frac_err_by_band=r['frac_err_by_band'], **info),
          open(os.path.join(P.HERE, 'results', f'profiles_diag_{case}.json'), 'w'), indent=1)
