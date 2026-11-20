#!/usr/bin/env python
"""Summarise pytest JUnit XML files: outcome counts, failures, and skip reasons.

Used by the CANFAR ``torchsky-tests-cuda`` mode and by local device runs of
torchsky's own test suite. Each input is ``LABEL=PATH``; the output JSON keeps
per-label counts, every failure/error with its first message line, and skip
reasons grouped by text. Skips whose reason names an absent accelerator are
flagged ``device_absent`` so a scorecard can list them separately.

    python scripts/summarize_junit.py main=main.xml map=map.xml --json-out s.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# Skip reasons that exist only because the accelerator is not present on the
# host. Matched case-insensitively against the skip message.
# "needs float64; MPS has none" is a capability skip on a present device, not
# an absent one, so the patterns name absence explicitly.
DEVICE_ABSENT_PATTERNS = (
    r"\b(cuda|mps|gpu|accelerator)\b[^;]*\b(unavailable|not available|absent|missing)\b",
    r"\b(requires|needs|no)\s+(a\s+)?(cuda|mps|gpu|accelerator)\b",
)
_DEVICE_RE = [re.compile(p, re.IGNORECASE) for p in DEVICE_ABSENT_PATTERNS]


def device_absent(reason: str) -> bool:
    return any(rx.search(reason) for rx in _DEVICE_RE)


def _first_line(text: str | None, limit: int = 400) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def _error_line(text: str | None, limit: int = 400) -> str:
    """The most informative line of a traceback: the last ``E   ...`` line."""
    if not text:
        return ""
    e_lines = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("E ")]
    if e_lines:
        return e_lines[-1][:limit]
    return _first_line(text, limit)


def _test_file(case: ET.Element) -> str:
    file_attr = case.get("file")
    if file_attr:
        return file_attr
    classname = case.get("classname", "")
    parts = classname.split(".")
    # pytest classnames look like tests.test_foo or tests.test_foo.TestBar
    mods = [p for p in parts if p.startswith("test_")]
    if mods:
        idx = parts.index(mods[0])
        return "/".join(parts[: idx + 1]) + ".py"
    return classname


def summarise(path: Path) -> dict:
    root = ET.parse(path).getroot()
    counts = Counter()
    failures = []
    skips = defaultdict(list)
    xfails = []
    for case in root.iter("testcase"):
        counts["total"] += 1
        test_id = f"{_test_file(case)}::{case.get('name')}"
        record = {
            "file": _test_file(case),
            "classname": case.get("classname"),
            "name": case.get("name"),
            "time_s": float(case.get("time") or 0.0),
        }
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        if failure is not None or error is not None:
            node = failure if failure is not None else error
            kind = "failed" if failure is not None else "error"
            counts[kind] += 1
            record.update(
                outcome=kind,
                type=node.get("type"),
                message=_first_line(node.get("message")),
                error=_error_line(node.text) or _first_line(node.get("message")),
            )
            failures.append(record)
        elif skipped is not None:
            if (skipped.get("type") or "") == "pytest.xfail":
                counts["xfailed"] += 1
                xfails.append({**record, "reason": skipped.get("message", "")})
            else:
                counts["skipped"] += 1
                reason = skipped.get("message") or _first_line(skipped.text)
                skips[reason].append(test_id)
        else:
            counts["passed"] += 1
    skip_groups = [
        {
            "reason": reason,
            "count": len(ids),
            "device_absent": device_absent(reason),
            "examples": ids[:5],
        }
        for reason, ids in sorted(skips.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return {
        "path": str(path),
        "counts": {
            k: counts.get(k, 0)
            for k in ("total", "passed", "failed", "error", "skipped", "xfailed")
        },
        "device_absent_skips": sum(g["count"] for g in skip_groups if g["device_absent"]),
        "failures": failures,
        "skip_groups": skip_groups,
        "xfails": xfails,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="LABEL=junit.xml")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--meta", action="append", default=[], help="KEY=VALUE to record")
    args = parser.parse_args(argv)
    out: dict = {"meta": dict(m.split("=", 1) for m in args.meta), "suites": {}}
    for item in args.inputs:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"expected LABEL=PATH, got {item!r}")
        p = Path(path)
        if not p.is_file():
            out["suites"][label] = {"path": path, "missing": True}
            continue
        out["suites"][label] = summarise(p)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for label, s in out["suites"].items():
        if s.get("missing"):
            print(f"{label}: missing {s['path']}")
            continue
        c = s["counts"]
        print(
            f"{label}: total={c['total']} passed={c['passed']} failed={c['failed']} "
            f"error={c['error']} skipped={c['skipped']} xfailed={c['xfailed']} "
            f"device_absent_skips={s['device_absent_skips']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
