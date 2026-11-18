"""Which lines of fieldop/*.py run, by workload (sys.monitoring, Python 3.12+; no extra package). Used to count the lines that are dead under --minimal.

    python tests/oracle/line_hits.py pytest OUT.json [pytest arguments]        # hits per test file (a restart at every test)
    python tests/oracle/line_hits.py script TAG OUT.json script.py [arguments]  # hits of one script run, under one tag

OUT.json: {"executable": {file: [lines]}, "hits": {tag: {file: [lines]}}}. Overhead: one callback per line and test, then the location is disabled.
"""
from __future__ import annotations

import glob
import json
import os
import runpy
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PKG = os.path.join(ROOT, "fieldop")
LINES = os.path.join(HERE, "results", "lines")
TOOL = 3

HITS: dict = defaultdict(lambda: defaultdict(set))
TAG = ["untagged"]


def executable_lines(path):
    src = open(path).read()
    top = compile(src, path, "exec")
    out, stack = set(), [top]
    while stack:
        c = stack.pop()
        out.update(ln for _, _, ln in c.co_lines() if ln is not None and ln > 0)
        stack.extend(k for k in c.co_consts if hasattr(k, "co_lines"))
    return sorted(out)


def start():
    mon = sys.monitoring
    mon.use_tool_id(TOOL, "fieldop-lines")

    def cb(code, line):
        f = code.co_filename
        if f.startswith(PKG):
            HITS[TAG[0]][os.path.basename(f)].add(line)
        return mon.DISABLE
    mon.register_callback(TOOL, mon.events.LINE, cb)
    mon.set_events(TOOL, mon.events.LINE)


def restart():
    sys.monitoring.restart_events()


def stop():
    sys.monitoring.set_events(TOOL, 0)
    sys.monitoring.free_tool_id(TOOL)


def dump(path):
    ex = {os.path.basename(p): executable_lines(p) for p in sorted(glob.glob(os.path.join(PKG, "*.py")))}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump({"executable": ex, "hits": {t: {f: sorted(v) for f, v in d.items()} for t, d in HITS.items()}}, open(path, "w"))


class Plugin:
    def pytest_runtest_setup(self, item):
        TAG[0] = os.path.basename(str(item.fspath))
        restart()


def main():
    mode = sys.argv[1]
    sys.path.insert(0, ROOT)
    if mode == "pytest":
        import pytest
        out = sys.argv[2]
        start()
        code = pytest.main(sys.argv[3:], plugins=[Plugin()])
        stop()
        dump(out)
        sys.exit(int(code))
    if mode == "script":
        tag, out, script = sys.argv[2], sys.argv[3], sys.argv[4]
        TAG[0] = tag
        sys.argv = [script] + sys.argv[5:]
        start()
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            stop()
            dump(out)
        return
    raise SystemExit("modes: pytest, script")


if __name__ == "__main__":
    main()
