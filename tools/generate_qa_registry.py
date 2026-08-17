#!/usr/bin/env python3
"""Validate the canonical QA registry and generate runner/build projections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn


SCHEMA = "yvex.qa.registry.v1"
IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
FUNCTION = re.compile(r"^yvex_(?:cuda_)?test_[a-z0-9_]+$")
EVIDENCE = {
    "unit",
    "component",
    "integration",
    "structural",
    "numeric",
    "reference",
    "runtime",
    "transactional",
    "cuda",
    "sanitizer",
    "live",
    "performance",
    "release",
    "coverage",
    "static",
    "property",
}
RESULT_STATES = {"PASS", "FAIL", "SKIP", "BLOCKED", "ERROR"}
RUNNER_KINDS = {"c-unit", "c-cuda", "command"}
DECISIONS = {"KEEP", "MOVE", "SPLIT", "CONSOLIDATE", "GENERATE", "REPLACE", "DELETE"}
MAKE_TARGET_ROLES = {"aggregate", "compatibility", "diagnostic", "fixture-plan"}
TEST_KEYS = {
    "id",
    "title",
    "owner",
    "domain",
    "evidence",
    "lanes",
    "runner",
    "paths",
    "build_targets",
    "requirements",
    "resources",
    "timeout_seconds",
    "cost",
    "deterministic_seed",
    "repeat",
    "fixture",
    "hermetic",
    "claims",
    "sanitizers",
    "inventory_decision",
    "legacy_targets",
    "runner_groups",
    "requirement_policy",
}


class RegistryError(ValueError):
    """One registry field violates the QA schema."""


def fail(where: str, message: str) -> NoReturn:
    raise RegistryError(f"{where}: {message}")


def text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        fail(where, "must be a non-empty string")
    return value


def string_list(value: Any, where: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        fail(where, "must be a string array")
    result = [text(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        fail(where, "contains duplicate values")
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(str(path), f"cannot read strict UTF-8 JSON: {exc}")
    if not isinstance(value, dict):
        fail(str(path), "top level must be an object")
    return value


def repository_files(root: Path, pathspec: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", pathspec],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        fail("git", result.stderr.strip() or "cannot enumerate tracked files")
    return sorted(result.stdout.splitlines())


def apply_defaults(registry: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    defaults = registry.get("defaults", {})
    if not isinstance(defaults, dict):
        fail("defaults", "must be an object")
    merged = dict(defaults)
    merged.update(item)
    resources = list(merged.get("resources", []))
    runner = merged.get("runner")
    if (isinstance(runner, dict) and runner.get("kind") == "command" and
            runner.get("argv") and Path(runner["argv"][0]).name == "make" and
            "build-tree" not in resources):
        resources.append("build-tree")
    merged["resources"] = resources
    return merged


def validate_lane_catalog(registry: dict[str, Any]) -> set[str]:
    lanes = registry.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        fail("lanes", "must be a non-empty object")
    for lane, value in lanes.items():
        if not IDENTIFIER.fullmatch(lane):
            fail(f"lanes.{lane}", "invalid lane ID")
        if not isinstance(value, dict):
            fail(f"lanes.{lane}", "must be an object")
        required = {"title", "description", "gate", "environment", "cost"}
        if set(value) != required:
            fail(f"lanes.{lane}", f"fields must be {sorted(required)}")
        text(value["title"], f"lanes.{lane}.title")
        text(value["description"], f"lanes.{lane}.description")
        text(value["environment"], f"lanes.{lane}.environment")
        text(value["cost"], f"lanes.{lane}.cost")
        if not isinstance(value["gate"], bool):
            fail(f"lanes.{lane}.gate", "must be boolean")
    return set(lanes)


def validate_resources(registry: dict[str, Any]) -> set[str]:
    resources = registry.get("resources")
    if not isinstance(resources, dict):
        fail("resources", "must be an object")
    for resource, value in resources.items():
        if not IDENTIFIER.fullmatch(resource):
            fail(f"resources.{resource}", "invalid resource ID")
        if not isinstance(value, dict) or set(value) != {"exclusive", "scope", "description"}:
            fail(f"resources.{resource}", "requires exclusive, scope, and description")
        if not isinstance(value["exclusive"], bool):
            fail(f"resources.{resource}.exclusive", "must be boolean")
        if value["scope"] not in {"worktree", "host"}:
            fail(f"resources.{resource}.scope", "must be worktree or host")
        text(value["description"], f"resources.{resource}.description")
    return set(resources)


def validate_runner(item: dict[str, Any], where: str, root: Path) -> None:
    runner = item.get("runner")
    if not isinstance(runner, dict):
        fail(f"{where}.runner", "must be an object")
    kind = text(runner.get("kind"), f"{where}.runner.kind")
    if kind not in RUNNER_KINDS:
        fail(f"{where}.runner.kind", f"must be one of {sorted(RUNNER_KINDS)}")
    allowed = {"kind", "function", "argv", "env"}
    unknown = set(runner) - allowed
    if unknown:
        fail(f"{where}.runner", f"unknown fields {sorted(unknown)}")
    if kind in {"c-unit", "c-cuda"}:
        function = text(runner.get("function"), f"{where}.runner.function")
        if not FUNCTION.fullmatch(function):
            fail(f"{where}.runner.function", "invalid C test function")
        if "argv" in runner or "env" in runner:
            fail(f"{where}.runner", "C runner cannot define argv or env")
    else:
        argv = string_list(runner.get("argv"), f"{where}.runner.argv", allow_empty=False)
        if not argv:
            fail(f"{where}.runner.argv", "cannot be empty")
        env = runner.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                            for key, value in env.items()):
            fail(f"{where}.runner.env", "must be a string map")
    for path in string_list(item.get("paths"), f"{where}.paths", allow_empty=False):
        if path.startswith("/") or ".." in Path(path).parts or not (root / path).is_file():
            fail(f"{where}.paths", f"invalid repository file {path!r}")


def validate_tests(registry: dict[str, Any], root: Path, lanes: set[str], resources: set[str]) -> list[dict[str, Any]]:
    raw_tests = registry.get("tests")
    if not isinstance(raw_tests, list) or not raw_tests:
        fail("tests", "must be a non-empty array")
    tests: list[dict[str, Any]] = []
    ids: set[str] = set()
    legacy_targets: set[str] = set()
    functions: set[str] = set()
    for index, raw in enumerate(raw_tests):
        where = f"tests[{index}]"
        if not isinstance(raw, dict):
            fail(where, "must be an object")
        item = apply_defaults(registry, raw)
        unknown = set(item) - TEST_KEYS
        if unknown:
            fail(where, f"unknown fields {sorted(unknown)}")
        test_id = text(item.get("id"), f"{where}.id")
        if not IDENTIFIER.fullmatch(test_id) or test_id in ids:
            fail(f"{where}.id", "invalid or duplicate stable test ID")
        ids.add(test_id)
        text(item.get("title"), f"{where}.title")
        text(item.get("owner"), f"{where}.owner")
        text(item.get("domain"), f"{where}.domain")
        evidence = set(string_list(item.get("evidence"), f"{where}.evidence", allow_empty=False))
        if not evidence <= EVIDENCE:
            fail(f"{where}.evidence", f"unknown values {sorted(evidence - EVIDENCE)}")
        selected_lanes = set(string_list(item.get("lanes"), f"{where}.lanes"))
        if not selected_lanes <= lanes:
            fail(f"{where}.lanes", f"unknown lanes {sorted(selected_lanes - lanes)}")
        selected_resources = set(string_list(item.get("resources"), f"{where}.resources"))
        if not selected_resources <= resources:
            fail(f"{where}.resources", f"unknown resources {sorted(selected_resources - resources)}")
        validate_runner(item, where, root)
        string_list(item.get("build_targets"), f"{where}.build_targets")
        requirements = item.get("requirements")
        if not isinstance(requirements, dict) or set(requirements) != {"hardware", "tools", "assets"}:
            fail(f"{where}.requirements", "requires hardware, tools, and assets")
        if requirements["hardware"] not in {"cpu", "cuda", "sm121"}:
            fail(f"{where}.requirements.hardware", "must be cpu, cuda, or sm121")
        string_list(requirements["tools"], f"{where}.requirements.tools")
        string_list(requirements["assets"], f"{where}.requirements.assets")
        if item.get("requirement_policy") not in {"block", "skip"}:
            fail(f"{where}.requirement_policy", "must be block or skip")
        if not isinstance(item.get("timeout_seconds"), int) or item["timeout_seconds"] <= 0:
            fail(f"{where}.timeout_seconds", "must be positive integer")
        if item.get("cost") not in {"tiny", "small", "medium", "large", "very-large"}:
            fail(f"{where}.cost", "invalid cost class")
        if not isinstance(item.get("repeat"), int) or item["repeat"] <= 0:
            fail(f"{where}.repeat", "must be positive integer")
        if item.get("deterministic_seed") is not None and not isinstance(item["deterministic_seed"], int):
            fail(f"{where}.deterministic_seed", "must be integer or null")
        if not isinstance(item.get("hermetic"), bool):
            fail(f"{where}.hermetic", "must be boolean")
        text(item.get("fixture"), f"{where}.fixture")
        string_list(item.get("claims"), f"{where}.claims", allow_empty=False)
        string_list(item.get("sanitizers"), f"{where}.sanitizers")
        if item.get("inventory_decision") not in DECISIONS:
            fail(f"{where}.inventory_decision", f"must be one of {sorted(DECISIONS)}")
        for target in string_list(item.get("legacy_targets"), f"{where}.legacy_targets"):
            if target in legacy_targets:
                fail(f"{where}.legacy_targets", f"duplicate target {target}")
            legacy_targets.add(target)
        string_list(item.get("runner_groups"), f"{where}.runner_groups")
        kind = item["runner"]["kind"]
        if kind in {"c-unit", "c-cuda"}:
            function = item["runner"]["function"]
            if function in functions:
                fail(f"{where}.runner.function", "duplicate C function")
            functions.add(function)
        tests.append(item)
    return sorted(tests, key=lambda item: item["id"])


def validate_c_function_parity(registry: dict[str, Any], tests: list[dict[str, Any]], root: Path) -> None:
    expected = {
        item["runner"]["function"]
        for item in tests
        if item["runner"]["kind"] in {"c-unit", "c-cuda"}
    }
    exempt = set(string_list(registry.get("registry_exempt_functions", []),
                             "registry_exempt_functions"))
    definitions: set[str] = set()
    pattern = re.compile(r"\bint\s+(yvex_(?:cuda_)?test_[a-z0-9_]+)\s*\(")
    for path in repository_files(root, "tests/unit"):
        if path.endswith(".c"):
            definitions.update(pattern.findall((root / path).read_text(errors="strict")))
    if definitions - exempt != expected:
        fail(
            "tests",
            f"C registration parity failed missing={sorted(definitions - exempt - expected)} "
            f"stale={sorted(expected - (definitions - exempt))}",
        )


def validate_command_inventory(registry: dict[str, Any], tests: list[dict[str, Any]], root: Path) -> None:
    registered = {path for item in tests for path in item["paths"]}
    exempt = set(string_list(registry.get("inventory_exempt_paths", []), "inventory_exempt_paths"))
    candidates = {
        path
        for path in repository_files(root, "tests")
        if Path(path).suffix in {".sh", ".py"}
    }
    missing = candidates - registered - exempt
    stale = {path for path in exempt if not (root / path).is_file()}
    if missing or stale:
        fail("inventory", f"command parity failed missing={sorted(missing)} stale_exempt={sorted(stale)}")


def validate_make_target_inventory(registry: dict[str, Any], tests: list[dict[str, Any]],
                                   root: Path) -> list[dict[str, Any]]:
    raw = registry.get("make_target_inventory")
    if not isinstance(raw, list):
        fail("make_target_inventory", "must be an array")
    known_tests = {item["id"] for item in tests}
    known_lanes = set(registry["lanes"])
    entries: list[dict[str, Any]] = []
    targets: set[str] = set()
    for index, item in enumerate(raw):
        where = f"make_target_inventory[{index}]"
        required = {"target", "role", "canonical_tests", "canonical_lanes", "decision", "reason"}
        if not isinstance(item, dict) or set(item) != required:
            fail(where, f"fields must be {sorted(required)}")
        target = text(item["target"], f"{where}.target")
        if target in targets:
            fail(f"{where}.target", "duplicate target")
        targets.add(target)
        if item["role"] not in MAKE_TARGET_ROLES:
            fail(f"{where}.role", f"must be one of {sorted(MAKE_TARGET_ROLES)}")
        canonical_tests = set(string_list(item["canonical_tests"], f"{where}.canonical_tests"))
        canonical_lanes = set(string_list(item["canonical_lanes"], f"{where}.canonical_lanes"))
        if canonical_tests - known_tests:
            fail(f"{where}.canonical_tests", f"unknown tests {sorted(canonical_tests - known_tests)}")
        if canonical_lanes - known_lanes:
            fail(f"{where}.canonical_lanes", f"unknown lanes {sorted(canonical_lanes - known_lanes)}")
        if item["decision"] not in DECISIONS:
            fail(f"{where}.decision", f"must be one of {sorted(DECISIONS)}")
        text(item["reason"], f"{where}.reason")
        entries.append(item)

    make_targets: set[str] = set()
    pattern = re.compile(r"^((?:test|check|smoke)(?:-[a-z0-9][a-z0-9_-]*)?):", re.MULTILINE)
    make_targets.update(pattern.findall((root / "Makefile").read_text(encoding="utf-8")))
    implicit = {target for item in tests for target in item["legacy_targets"]}
    missing = make_targets - implicit - targets
    # Registry legacy targets are materialized through the generated generic
    # Make rule and therefore need not have handwritten literal target lines.
    stale = targets - make_targets
    if missing or stale:
        fail("make_target_inventory",
             f"parity failed missing={sorted(missing)} stale={sorted(stale)}")
    return sorted(entries, key=lambda item: item["target"])


def c_string(value: str) -> str:
    return json.dumps(value)


def render_declarations(tests: list[dict[str, Any]]) -> str:
    functions = sorted(
        item["runner"]["function"]
        for item in tests
        if item["runner"]["kind"] in {"c-unit", "c-cuda"}
    )
    lines = [
        "/* Generated by tools/generate_qa_registry.py; do not edit. */",
        "#ifndef YVEX_TEST_DECLARATIONS_H",
        "#define YVEX_TEST_DECLARATIONS_H",
        "",
    ]
    lines.extend(f"int {function}(void);" for function in functions)
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def table_tests(tests: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "unit":
        return [item for item in tests if item["runner"]["kind"] == "c-unit"]
    if group == "cuda":
        return [item for item in tests if item["runner"]["kind"] == "c-cuda"]
    return [item for item in tests if group in item["runner_groups"]]


def render_table(tests: list[dict[str, Any]], group: str) -> str:
    selected = table_tests(tests, group)
    symbol = group.replace("-", "_")
    lines = ["/* Generated by tools/generate_qa_registry.py; do not edit. */"]
    lines.append(f"static const struct yvex_test_entry yvex_{symbol}_tests[] = {{")
    for item in selected:
        legacy = item["id"].split(".", 1)[-1]
        lines.append(
            f"    {{{c_string(item['id'])}, {c_string(legacy)}, {item['runner']['function']}}},"
        )
    lines.extend(
        [
            "};",
            f"static const size_t yvex_{symbol}_test_count =",
            f"    sizeof(yvex_{symbol}_tests) / sizeof(yvex_{symbol}_tests[0]);",
            "",
        ]
    )
    return "\n".join(lines)


def make_assignment(name: str, values: list[str]) -> str:
    if not values:
        return f"{name} :=\n"
    lines = [f"{name} := \\"]
    for index, value in enumerate(values):
        suffix = " \\" if index + 1 < len(values) else ""
        lines.append(f"\t{value}{suffix}")
    return "\n".join(lines) + "\n"


def render_make(tests: list[dict[str, Any]], identity: str) -> str:
    lines = [
        "# Generated by tools/generate_qa_registry.py; do not edit.",
        f"QA_REGISTRY_IDENTITY := {identity}",
        "",
    ]
    groups = sorted({group for item in tests for group in item["runner_groups"]})
    for group in groups:
        sources = sorted({item["paths"][0] for item in table_tests(tests, group)})
        lines.append(make_assignment(f"QA_{group.upper().replace('-', '_')}_TEST_SRCS", sources).rstrip())
        lines.append("")
    aliases = sorted(target for item in tests for target in item["legacy_targets"])
    lines.append(make_assignment("QA_LEGACY_TARGETS", aliases).rstrip())
    lines.append("")
    c_unit_aliases = sorted(
        target for item in tests if item["runner"]["kind"] == "c-unit"
        for target in item["legacy_targets"]
    )
    lines.append(make_assignment("QA_C_UNIT_LEGACY_TARGETS", c_unit_aliases).rstrip())
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_inventory(tests: list[dict[str, Any]]) -> str:
    fields = [
        "test_id",
        "path",
        "owner",
        "domain",
        "evidence",
        "lanes",
        "runner",
        "hardware",
        "assets",
        "cost",
        "hermetic",
        "fixture",
        "claim",
        "decision",
    ]
    rows: list[list[str]] = []
    for item in tests:
        for path in item["paths"]:
            rows.append(
                [
                    item["id"],
                    path,
                    item["owner"],
                    item["domain"],
                    ",".join(item["evidence"]),
                    ",".join(item["lanes"]),
                    item["runner"]["kind"],
                    item["requirements"]["hardware"],
                    ",".join(item["requirements"]["assets"]),
                    item["cost"],
                    "yes" if item["hermetic"] else "no",
                    item["fixture"],
                    ",".join(item["claims"]),
                    item["inventory_decision"],
                ]
            )
    output = ["\t".join(fields)]
    output.extend("\t".join(row) for row in sorted(rows))
    return "\n".join(output) + "\n"


def render_make_target_inventory(entries: list[dict[str, Any]]) -> str:
    fields = ["target", "role", "canonical_tests", "canonical_lanes", "decision", "reason"]
    lines = ["\t".join(fields)]
    for item in entries:
        lines.append("\t".join([
            item["target"], item["role"], ",".join(item["canonical_tests"]),
            ",".join(item["canonical_lanes"]), item["decision"], item["reason"],
        ]))
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    if path.is_file() and path.read_text() == content:
        os.utime(path, None)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def projections(registry: dict[str, Any], tests: list[dict[str, Any]]) -> dict[str, str]:
    material = json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(material).hexdigest()
    groups = {"unit", "cuda", "quant", "artifact"}
    result = {
        "test_declarations.h": render_declarations(tests),
        "registry.mk": render_make(tests, identity),
        "inventory.tsv": render_inventory(tests),
        "make_targets.tsv": render_make_target_inventory(registry["make_target_inventory"]),
        "registry.sha256": identity + "\n",
    }
    for group in sorted(groups):
        result[f"{group}_registry.inc"] = render_table(tests, group)
    return result


def load_and_validate(root: Path, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = load_json(path)
    allowed = {
        "schema",
        "schema_version",
        "result_states",
        "defaults",
        "lanes",
        "resources",
        "tests",
        "inventory_exempt_paths",
        "registry_exempt_functions",
        "make_target_inventory",
    }
    unknown = set(registry) - allowed
    if unknown:
        fail("registry", f"unknown fields {sorted(unknown)}")
    if registry.get("schema") != SCHEMA or registry.get("schema_version") != 1:
        fail("registry", f"requires {SCHEMA} version 1")
    if set(string_list(registry.get("result_states"), "result_states")) != RESULT_STATES:
        fail("result_states", f"must be exactly {sorted(RESULT_STATES)}")
    lanes = validate_lane_catalog(registry)
    resources = validate_resources(registry)
    tests = validate_tests(registry, root, lanes, resources)
    validate_c_function_parity(registry, tests, root)
    validate_command_inventory(registry, tests, root)
    registry["make_target_inventory"] = validate_make_target_inventory(registry, tests, root)
    return registry, tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("config/qa/registry.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    registry_path = arguments.registry if arguments.registry.is_absolute() else root / arguments.registry
    output_dir = arguments.output_dir if arguments.output_dir.is_absolute() else root / arguments.output_dir
    try:
        registry, tests = load_and_validate(root, registry_path)
        expected = projections(registry, tests)
        if arguments.check:
            stale = [name for name, content in expected.items()
                     if not (output_dir / name).is_file() or (output_dir / name).read_text() != content]
            if stale:
                fail("generated", f"stale projections {stale}")
        else:
            for name, content in expected.items():
                write_atomic(output_dir / name, content)
    except RegistryError as exc:
        raise SystemExit(f"qa registry: {exc}") from exc
    print(f"qa registry: ok tests={len(tests)} lanes={len(registry['lanes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
