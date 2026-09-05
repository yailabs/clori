#!/usr/bin/env python3
"""Property and mutation checks for the canonical QA registry seam."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import generate_qa_registry  # noqa: E402
import qa  # noqa: E402


def expect_refusal(registry: dict, message: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
        json.dump(registry, stream)
        stream.flush()
        try:
            generate_qa_registry.load_and_validate(ROOT, Path(stream.name))
        except generate_qa_registry.RegistryError:
            return
    raise AssertionError(message)


def sample_report(*, stability: dict | None) -> dict:
    value = {
        "schema": qa.EVIDENCE_SCHEMA,
        "run_identity": "source-stability-property",
        "source_commit": "1" * 40,
        "source_state": "clean",
        "summary": {"counts": {state: 0 for state in ["PASS", "FAIL", "SKIP", "BLOCKED", "ERROR"]}},
        "results": [],
    }
    if stability is not None:
        value["source_stability"] = stability
    return value


def check_source_stability() -> None:
    clean = {"head": "1" * 40, "state": "clean", "delta_identity": "a" * 64}
    if not qa.source_stability(clean, dict(clean))["valid"]:
        raise AssertionError("unchanged QA source snapshot was invalidated")

    moved_head = dict(clean, head="2" * 40)
    if qa.source_stability(clean, moved_head)["valid"]:
        raise AssertionError("changed HEAD retained valid QA evidence")

    dirty_delta = dict(clean, state="dirty", delta_identity="b" * 64)
    invalid = qa.source_stability(clean, dirty_delta)
    if invalid["valid"] or invalid["changed_fields"] != ["state", "delta_identity"]:
        raise AssertionError("changed dirty source delta retained valid QA evidence")

    with tempfile.TemporaryDirectory() as directory:
        repository = Path(directory)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.email", "qa@example.invalid"], cwd=repository,
                       check=True)
        subprocess.run(["git", "config", "user.name", "QA Property"], cwd=repository,
                       check=True)
        (repository / "tracked.c").write_text("int tracked;\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.c"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
        before = qa.source_snapshot(repository)
        (repository / "untracked.c").write_text("int untracked;\n", encoding="utf-8")
        after = qa.source_snapshot(repository)
        if qa.source_stability(before, after)["valid"]:
            raise AssertionError("untracked source input retained valid QA evidence")

    with tempfile.TemporaryDirectory() as directory:
        invalid_path = Path(directory) / "invalid.json"
        invalid_path.write_text(json.dumps(sample_report(stability=invalid)), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = qa.report(str(invalid_path))
        if status == 0 or "SOURCE MUTATED / EVIDENCE INVALID" not in output.getvalue():
            raise AssertionError("invalid QA evidence was not visible and non-zero")

        legacy_path = Path(directory) / "legacy.json"
        legacy_path.write_text(json.dumps(sample_report(stability=None)), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            if qa.report(str(legacy_path)) != 0:
                raise AssertionError("legacy evidence without optional stability fields was refused")


def main() -> int:
    source = ROOT / "config/qa/registry.json"
    registry, tests = generate_qa_registry.load_and_validate(ROOT, source)
    build_consumers = [
        item for item in tests
        if item["runner"]["kind"] in {"c-unit", "c-cuda"} or
        (item["runner"]["kind"] == "command" and item["runner"]["argv"][0] == "make")
    ]
    if any("build-tree" not in item["resources"] for item in build_consumers):
        raise AssertionError("native or Make-backed test did not acquire the build-tree resource")
    first = generate_qa_registry.projections(registry, tests)
    second = generate_qa_registry.projections(registry, tests)
    if first != second:
        raise AssertionError("QA projections are not deterministic")
    check_source_stability()

    duplicate = copy.deepcopy(registry)
    duplicate["tests"].append(copy.deepcopy(duplicate["tests"][0]))
    expect_refusal(duplicate, "duplicate test ID was accepted")

    unknown_lane = copy.deepcopy(registry)
    unknown_lane["tests"][0]["lanes"] = ["not-a-lane"]
    expect_refusal(unknown_lane, "unknown lane was accepted")

    malformed_function = copy.deepcopy(registry)
    c_test = next(item for item in malformed_function["tests"] if item["runner"]["kind"] == "c-unit")
    c_test["runner"]["function"] = "unsafe_test_function"
    expect_refusal(malformed_function, "unregistered C function was accepted")

    invalid_result = copy.deepcopy(registry)
    invalid_result["result_states"] = ["PASS", "FAIL"]
    expect_refusal(invalid_result, "incomplete result taxonomy was accepted")

    invalid_resource = copy.deepcopy(registry)
    invalid_resource["resources"]["cuda-device"]["scope"] = "process"
    expect_refusal(invalid_resource, "invalid resource scope was accepted")

    missing_make_target = copy.deepcopy(registry)
    missing_make_target["make_target_inventory"].pop()
    expect_refusal(missing_make_target, "orphan Make QA target was accepted")

    if len(tests) < 80:
        raise AssertionError("registry unexpectedly lost the established test corpus")
    print(f"qa registry properties: ok tests={len(tests)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
