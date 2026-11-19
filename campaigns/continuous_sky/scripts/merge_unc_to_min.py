"""Copy fieldop_tsky's uncertainty.py into fieldop_min with the four edits the pruned package needs (names of deleted terms)."""
s=open('/home/claude/work/fieldop_tsky/fieldop/uncertainty.py').read()
a='''            Lb = legendre6(pix["xi"])
            G = self.model.gain_of(t.eid, pix)
            return Lb * G[:, None] if G is not None else Lb
'''
assert s.count(a)==1; s=s.replace(a,'''            return legendre6(pix["xi"])
''')
for l in ['    _data.RATE_WCS = cfg.rate_wcs\n','    _data.LOADER = getattr(cfg, "loader", "astropy")\n',
          '    model.attach_pixel_gain(exps, cfg, fset_names=fset.names, det_names=info["dets"])\n',
          '    model.attach_filter_shift(exps, cfg, fset_names=fset.names, det_names=info["dets"])\n']:
    assert s.count(l)==1, l; s=s.replace(l,'')
open('/home/claude/work/fieldop_min/fieldop/uncertainty.py','w').write(s)
tn=open('/home/claude/work/fieldop_tsky/tests/test_uncertainty.py').read().splitlines(keepends=True)
tm=open('/home/claude/work/fieldop_min/tests/test_uncertainty.py').read().splitlines(keepends=True)
i=[k for k,l in enumerate(tn) if '6. real-data adaptation' in l][0]
j=[k for k,l in enumerate(tm) if '6. real-data adaptation' in l][0]
open('/home/claude/work/fieldop_min/tests/test_uncertainty.py','w').write(''.join(tm[:j])+''.join(tn[i:]))
print("merged")
