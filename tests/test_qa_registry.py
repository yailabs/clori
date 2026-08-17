#!/usr/bin/env python3
"""Property and mutation checks for the canonical QA registry seam."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import generate_qa_registry  # noqa: E402


def expect_refusal(registry: dict, message: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
        json.dump(registry, stream)
        stream.flush()
        try:
            generate_qa_registry.load_and_validate(ROOT, Path(stream.name))
        except generate_qa_registry.RegistryError:
            return
    raise AssertionError(message)


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
