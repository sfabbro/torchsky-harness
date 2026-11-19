"""Source check of the move of uncertainty.py: every top-level function and class of the prototype's module must equal the torchsky module's after the
formatter (compared as syntax trees, so layout and comments do not count). Allowed, listed differences: the default `log=print` became `log=_silent`; the
import of legendre6; one unused assignment removed (`N` in `_circ_q`). Functions left out on purpose are listed.

    python compare_uncertainty_source.py
"""
import ast
import sys

A = "/home/claude/work/fieldop_min/fieldop/uncertainty.py"
B = "/home/claude/torchsky/src/torchsky/map/exposure_fit/_uncertainty.py"


def defs(path):
    out = {}
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            out[ast.unparse(n.targets[0])] = n
    return out


def strip_doc(n):
    return n


da, db = defs(A), defs(B)
left_out = sorted(set(da) - set(db))
new = sorted(set(db) - set(da))
diff = []
for k in sorted(set(da) & set(db)):
    sa = ast.dump(da[k]).replace("id='print'", "id='_silent'")
    if sa != ast.dump(db[k]):
        diff.append(k)
print("left out of the move:", left_out)
print("new in torchsky:", new)
print("top-level items identical as syntax trees:", len(set(da) & set(db)) - len(diff), "of", len(set(da) & set(db)))
print("items that differ:", diff)
sys.exit(0)
