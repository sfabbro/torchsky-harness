"""Medium and small test fields (section 5 of SOLVER_DIAGNOSIS_2026-11-19.md): python tests/solver_repro/medium_field.py JSON_OPTIONS
(environment: MODES=filter,fourier; NPROBES=1). Logs of the 19 Nov runs: logs/medium_t5a.log (two LW exposures, Q = 2, both modes),
t5b (three LW exposures, Q = 2, the filter mode before the probe option existed), t5c (three LW, Q = 3, one probe), t5d (three LW, Q = 2, 9 probes)."""
import os; import sys; import torch; import json; import time; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); torch.set_num_threads(1)
from repro_field import build_model
from fieldop import uncertainty as U
kw=json.loads(sys.argv[1])
t0=time.time()
model,exps,cfg,truth,o=build_model(gap_centre_filter="F090W", **kw)
for m in os.environ.get("MODES","filter,fourier").split(","):
    import os; op=U.SkyNormalOperator(model,exps,cfg,rho=10.0,precond_mode=m,log=lambda s:None,tile=64,batch_px=16000,n_probes=int(os.environ.get("NPROBES","1")))
    if m=="filter": print("probes",[len(v) for v in op.probe_list.values()], flush=True)
    x,info=U.pcg(op,op.residual_rhs(),tol=1e-6,maxiter=60 if m=="fourier" else 100)
    print(m, 'N',op.N,'npix',op.n_pix,'kappa', [round(k,6) for k in op.kappa[:3]], 'hist', [f"{h:.1e}" for h in info['history']], 'final', f"{info['rel_residual']:.2e}", 'ritz', round(info['ritz_min'],4), round(info['ritz_max'],1), 't/app %.2f'%(op.t_apply/op.n_apply), flush=True)
print('total %.0f s'%(time.time()-t0))
