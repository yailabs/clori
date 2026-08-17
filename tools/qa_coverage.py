#!/usr/bin/env python3
"""Produce a local line/function coverage diagnostic for production owners."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFLAGS = (
    "-O0 -g --coverage -std=c11 -Wall -Wextra -pedantic -Wstrict-prototypes "
    "-Wmissing-prototypes -Wmissing-declarations -Wshadow -Wformat=2 -Wundef "
    "-Wvla -pthread"
)


def run(argv: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def main() -> int:
    identity = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    run_root = ROOT / "build/qa/coverage/runs" / identity
    build_root = run_root / "build"
    run_root.mkdir(parents=True)
    run(
        [
            "make",
            f"BUILD_DIR={build_root.relative_to(ROOT)}",
            "NVCC=__yvex_nvcc_unavailable__",
            f"CFLAGS={CFLAGS}",
            "LDFLAGS=--coverage",
            "test-core",
        ]
    )

    totals: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(dir=run_root, prefix="gcov-") as temporary:
        output_root = Path(temporary)
        for data_file in sorted((build_root / "obj/src").rglob("*.gcda")):
            run(["gcov", "--json-format", "--preserve-paths", str(data_file)], cwd=output_root)
        for report_path in sorted(output_root.glob("*.gcov.json.gz")):
            with gzip.open(report_path, "rt", encoding="utf-8") as stream:
                report = json.load(stream)
            for source in report.get("files", []):
                path = Path(source["file"])
                try:
                    relative = path.resolve().relative_to(ROOT)
                except ValueError:
                    continue
                if not relative.parts or relative.parts[0] != "src":
                    continue
                domain = relative.parts[1] if len(relative.parts) > 1 else "root"
                facts = totals.setdefault(
                    domain,
                    {"lines_total": 0, "lines_covered": 0, "functions_total": 0,
                     "functions_covered": 0},
                )
                lines = source.get("lines", [])
                functions = source.get("functions", [])
                facts["lines_total"] += len(lines)
                facts["lines_covered"] += sum(int(line.get("count", 0)) > 0 for line in lines)
                facts["functions_total"] += len(functions)
                facts["functions_covered"] += sum(
                    int(function.get("execution_count", 0)) > 0 for function in functions
                )

    overall = {key: sum(facts[key] for facts in totals.values())
               for key in ("lines_total", "lines_covered", "functions_total", "functions_covered")}
    summary = {
        "schema": "yvex.qa.coverage.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "overall": overall,
        "domains": dict(sorted(totals.items())),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    line_percent = 100.0 * overall["lines_covered"] / max(overall["lines_total"], 1)
    function_percent = 100.0 * overall["functions_covered"] / max(overall["functions_total"], 1)
    print(
        f"coverage: lines {overall['lines_covered']}/{overall['lines_total']} ({line_percent:.1f}%) "
        f"functions {overall['functions_covered']}/{overall['functions_total']} "
        f"({function_percent:.1f}%)"
    )
    print(f"coverage: summary {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
