#!/usr/bin/env python3
"""Validate the canonical operator registry, generator, and audit reconciliation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config/operator/registry.json"
GENERATOR = ROOT / "tools/generate_operator_registry.py"
AUDIT = ROOT / "docs/audits/operator-surface-ec7dcc"
GENERATED = ROOT / "build/generated/operator"
FORBIDDEN_TOP_LEVEL = {
    "evidence",
    "graph",
    "quant",
    "source",
    "tensor",
    "tokenizer",
    "eval",
    "bench",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_registry() -> dict[str, object]:
    with REGISTRY.open(encoding="utf-8") as source:
        return json.load(source)


def invoke(
    registry: pathlib.Path,
    output: pathlib.Path,
    check: bool = False,
    audit: pathlib.Path | None = None,
    migration: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        "python3",
        str(GENERATOR),
        "--registry",
        str(registry),
        "--output",
        str(output),
    ]
    if audit is not None:
        command.extend(["--audit-root", str(audit)])
    if migration is not None:
        command.extend(["--migration-output", str(migration)])
    if check:
        command.append("--check")
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def mutation_failure(registry: dict[str, object], mutate, expected: str) -> None:
    candidate = copy.deepcopy(registry)
    mutate(candidate)
    with tempfile.TemporaryDirectory(prefix="yvex-registry-refusal-") as temporary:
        root = pathlib.Path(temporary)
        source = root / "registry.json"
        source.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = invoke(source, root / "generated")
    require(result.returncode == 2, f"mutation unexpectedly passed: {expected}")
    require(expected in result.stderr, f"missing refusal {expected!r}: {result.stderr}")


def operation(registry: dict[str, object], operation_id: str) -> dict[str, object]:
    rows = registry["operations"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["operation_id"] == operation_id)


def test_generation(registry: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="yvex-registry-generation-") as temporary:
        first = pathlib.Path(temporary) / "first"
        second = pathlib.Path(temporary) / "second"
        result = invoke(REGISTRY, first)
        require(result.returncode == 0, result.stderr)
        result = invoke(REGISTRY, second)
        require(result.returncode == 0, result.stderr)
        products = ("registry.h", "registry.c", "registry.sha256")
        for name in products:
            require((first / name).read_bytes() == (second / name).read_bytes(), f"nondeterministic {name}")
        require(invoke(REGISTRY, first, check=True).returncode == 0, "fresh products rejected")
        (first / "registry.c").write_text("stale\n", encoding="utf-8")
        require(invoke(REGISTRY, first, check=True).returncode == 1, "stale product accepted")
    normalized = json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    identity = hashlib.sha256(normalized).hexdigest()
    require((GENERATED / "registry.sha256").read_text(encoding="utf-8").strip() == identity,
            "generated registry identity is stale")
    generated_source = (GENERATED / "registry.c").read_text(encoding="utf-8")
    for forbidden in (
        "yvex_artifact_",
        "yvex_backend_",
        "yvex_generation_",
        "yvex_graph_",
        "yvex_protocol_",
        "yvex_runtime_",
        "yvex_server_",
        "malloc(",
        "fopen(",
    ):
        require(forbidden not in generated_source,
                f"generated descriptors contain behavior: {forbidden}")
    generated_object = ROOT / "build/obj/generated/operator/registry.o"
    require(generated_object.is_file(), "compiled registry descriptor object is missing")
    undefined = subprocess.run(
        ["nm", "-u", str(generated_object)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(undefined.returncode == 0, undefined.stderr)
    require(not undefined.stdout.strip(),
            f"generated descriptor object has behavior dependencies: {undefined.stdout}")


def test_refusals(registry: dict[str, object]) -> None:
    mutation_failure(registry, lambda row: row.update(schema_version=2), "registry.schema")
    mutation_failure(registry, lambda row: row.update(unexpected=True), "unknown field 'unexpected'")
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(summmary="typo"),
        "unknown field 'summmary'",
    )
    mutation_failure(
        registry,
        lambda row: row["operations"].append(copy.deepcopy(row["operations"][0])),
        "duplicate operation ID",
    )

    def duplicate_path(row: dict[str, object]) -> None:
        source = operation(row, "server.status")
        target = operation(row, "server.models")
        target["command_path"] = list(source["command_path"])

    mutation_failure(registry, duplicate_path, "duplicate canonical path")

    def alias_collision(row: dict[str, object]) -> None:
        operation(row, "server.status")["aliases"] = [
            {"path": ["server", "models"], "deprecation": "current"}
        ]

    mutation_failure(registry, alias_collision, "alias collides")

    def duplicate_flag(row: dict[str, object]) -> None:
        operation(row, "server.status")["flags"] = [
            {"name": "--json", "value_type": "boolean", "takes_value": False}
        ]

    mutation_failure(registry, duplicate_flag, "duplicate flag")

    def conflicting_flag_type(row: dict[str, object]) -> None:
        operation(row, "server.status")["flags"] = [
            {"name": "--json", "value_type": "number", "takes_value": True}
        ]

    mutation_failure(registry, conflicting_flag_type, "conflicting flag types/defaults")

    def unknown_flag_field(row: dict[str, object]) -> None:
        operation(row, "server.status")["flags"] = [
            {
                "name": "--strict-test",
                "value_type": "boolean",
                "takes_value": False,
                "surprise": True,
            }
        ]

    mutation_failure(registry, unknown_flag_field, "unknown field 'surprise'")

    def unknown_argument_field(row: dict[str, object]) -> None:
        target = next(item for item in row["operations"] if item.get("arguments"))
        target["arguments"][0]["surprise"] = True

    mutation_failure(registry, unknown_argument_field, "unknown field 'surprise'")

    def invalid_argument_order(row: dict[str, object]) -> None:
        operation(row, "server.status")["arguments"] = [
            {"name": "optional", "multiplicity": "optional"},
            {"name": "required", "multiplicity": "one", "required": True},
        ]

    mutation_failure(registry, invalid_argument_order, "required argument cannot follow")

    def unknown_relation(row: dict[str, object]) -> None:
        operation(row, "server.status")["flags"] = [
            {
                "name": "--extra",
                "value_type": "boolean",
                "takes_value": False,
                "conflicts": ["--missing"],
            }
        ]

    mutation_failure(registry, unknown_relation, "unknown related flag")
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(test_owner="none"),
        "requires test and documentation owners",
    )
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(adapter_id="graph"),
        "unknown runtime-client adapter",
    )
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(protocol_operation="unknown"),
        "unknown protocol operation",
    )
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(renderer_id="unknown"),
        "unknown renderer",
    )
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(command_path=["eval"]),
        "forbidden top-level namespace",
    )
    mutation_failure(
        registry,
        lambda row: operation(row, "server.status").update(summary="run yvex-dev"),
        "references retired executable",
    )
    mutation_failure(
        registry,
        lambda row: row["flag_sets"].update(orphan=[]),
        "orphan flag set",
    )

    def unknown_flag_set_field(row: dict[str, object]) -> None:
        target = next(value for value in row["flag_sets"].values() if isinstance(value, dict))
        target["surprise"] = []

    mutation_failure(registry, unknown_flag_set_field, "unknown field 'surprise'")


def read_tsv(name: str) -> list[dict[str, str]]:
    with (AUDIT / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def test_audit_reconciliation(registry: dict[str, object]) -> None:
    commands = read_tsv("commands.tsv")
    flags = read_tsv("flags.tsv")
    operations = read_tsv("operations.tsv")
    require(len(commands) == 70, "frozen command count changed")
    require(len(flags) == 426, "frozen command/flag count changed")
    require(len(operations) == 99, "frozen operation count changed")
    rows = registry["operations"]
    assert isinstance(rows, list)
    by_id = {row["operation_id"]: row for row in rows}
    unmatched = sorted({row["operation_id"] for row in operations} - set(by_id))
    require(not unmatched, f"unmatched audit operations: {unmatched}")
    for row in rows:
        if row.get("deprecation_state") != "removed":
            continue
        successors = row.get("superseded_by", [])
        require(successors, f"removed operation has no successor: {row['operation_id']}")
        require(all(successor in by_id for successor in successors),
                f"unknown successor for {row['operation_id']}")
        require(all(by_id[successor].get("deprecation_state") == "current" for successor in successors),
                f"removed successor for {row['operation_id']}")
    command_ids = {row["command_id"] for row in commands}
    require(all(row["command_id"] in command_ids for row in flags),
            "flag row has no audited command owner")
    require(sum(row.get("lane") == "offline-engine" and row.get("CLI_projection") for row in rows) >= 39,
            "offline capabilities were not preserved")
    require(sum(row.get("lane") == "runtime-client" and row.get("CLI_projection") for row in rows) >= 17,
            "client capabilities were not preserved")
    require(any(row.get("operation_id") == "server.host" and
                row.get("lane") == "daemon-entrypoint" and row.get("CLI_projection")
                for row in rows), "foreground server entrypoint is not projected")
    slash = {row.get("slash_projection") for row in rows if row.get("slash_projection") != "none"}
    require(slash == {"/help", "/status", "/models", "/memory", "/context", "/sessions",
                      "/session", "/new", "/attach", "/detach", "/reset", "/close",
                      "/cancel", "/quit", "/nothink", "/think", "/think-max"},
            f"unexpected slash catalog: {sorted(slash)}")
    slash_aliases = {alias for row in rows for alias in row.get("slash_aliases", [])}
    require(slash_aliases == {"/exit"},
            f"unexpected slash aliases: {sorted(slash_aliases)}")
    with tempfile.TemporaryDirectory(prefix="yvex-audit-reconciliation-") as temporary:
        root = pathlib.Path(temporary)
        first = root / "first.md"
        second = root / "second.md"
        generated = root / "generated"
        result = invoke(REGISTRY, generated, audit=AUDIT, migration=first)
        require(result.returncode == 0, result.stderr)
        result = invoke(REGISTRY, generated, audit=AUDIT, migration=second)
        require(result.returncode == 0, result.stderr)
        require(first.read_bytes() == second.read_bytes(), "nondeterministic migration matrix")
        require(first.read_bytes() ==
                (ROOT / "docs/migrations/command-architecture-v1.md").read_bytes(),
                "tracked migration matrix is stale")


def test_compiled_discovery(registry: dict[str, object]) -> None:
    result = subprocess.run(
        [str(ROOT / "yvex"), "help", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, result.stderr)
    require("\x1b" not in result.stdout and "/home/" not in result.stdout,
            "machine discovery leaked terminal or private-path data")
    discovery = json.loads(result.stdout)
    require(discovery["schema"] == "yvex.command.discovery.v1", "discovery schema")
    require(discovery["registry_identity"] == (GENERATED / "registry.sha256").read_text().strip(),
            "compiled registry identity")
    operations = discovery["operations"]
    require(len(operations) == len(registry["operations"]), "compiled operation coverage")
    defaults = registry["operation_defaults"]
    flag_sets = registry["flag_sets"]
    assert isinstance(defaults, dict) and isinstance(flag_sets, dict)
    expected: dict[str, dict[str, object]] = {}
    for raw in registry["operations"]:
        assert isinstance(raw, dict)
        row = dict(defaults)
        row.update(raw)
        flags: list[dict[str, object]] = []
        for set_name in [*registry.get("global_flag_sets", []), *row.get("flag_sets", [])]:
            flag_set = flag_sets[set_name]
            if isinstance(flag_set, dict):
                flags.extend({"name": name, "value_type": "delegated", "takes_value": True}
                             for name in flag_set.get("values", []))
                flags.extend({"name": name, "value_type": "boolean", "takes_value": False}
                             for name in flag_set.get("booleans", []))
                flags.extend({"name": name, "value_type": "delegated", "takes_value": True,
                              "multiplicity": "repeatable"}
                             for name in flag_set.get("repeatable_values", []))
            else:
                assert isinstance(flag_set, list)
                flags.extend(flag_set)
        flags.extend(row.get("flags", []))
        expected[row["operation_id"]] = {**row, "expanded_flags": flags}
    for actual in operations:
        source = expected[actual["operation_id"]]
        require(actual["command_path"] == " ".join(source.get("command_path", [])),
                f"discovery command path: {actual['operation_id']}")
        require(actual["aliases"] == [" ".join(alias["path"])
                                      for alias in source.get("aliases", [])],
                f"discovery aliases: {actual['operation_id']}")
        require(actual["summary"] == source["summary"],
                f"discovery summary: {actual['operation_id']}")
        require(actual["input_schema"] == source["input_schema"] and
                actual["result_schema"] == source["result_schema"],
                f"discovery schemas: {actual['operation_id']}")
        require(actual["side_effects"] == source["side_effects"],
                f"discovery side effects: {actual['operation_id']}")
        require(len(actual["arguments"]) == len(source.get("arguments", [])),
                f"discovery argument coverage: {actual['operation_id']}")
        slash_arguments = source.get("slash_arguments", source.get("arguments", []))
        require(len(actual["slash_arguments"]) == len(slash_arguments),
                f"discovery slash argument coverage: {actual['operation_id']}")
        require([flag["name"] for flag in actual["flags"]] ==
                [flag["name"] for flag in source["expanded_flags"]],
                f"discovery flag coverage: {actual['operation_id']}")
        require(actual["projections"]["protocol"] == source.get("protocol_operation", "none"),
                f"discovery protocol projection: {actual['operation_id']}")
        require(actual["test_owner"] == source["test_owner"] and
                actual["documentation_owner"] == source["documentation_owner"],
                f"discovery owners: {actual['operation_id']}")
    projected = [row for row in operations if row["projections"]["cli"]]
    for row in projected:
        first = row["command_path"].split(" ", 1)[0]
        require(first not in FORBIDDEN_TOP_LEVEL, f"forbidden projection: {row['command_path']}")


def test_completion() -> None:
    outputs: dict[str, str] = {}
    for shell in ("bash", "zsh", "fish"):
        command = [str(ROOT / "yvex"), "completion", shell]
        first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        require(first.returncode == 0, first.stderr)
        require(first.stdout == second.stdout, f"nondeterministic {shell} completion")
        require("yvex-dev" not in first.stdout and "yvex-openai" not in first.stdout,
                f"retired executable in {shell} completion")
        require("'server'" in first.stdout and "server status" in first.stdout and
                "--ctx" in first.stdout,
                f"{shell} completion is not context aware")
        outputs[shell] = first.stdout
    with tempfile.TemporaryDirectory(prefix="yvex-completion-") as temporary:
        root = pathlib.Path(temporary)
        bash = root / "yvex.bash"
        bash.write_text(outputs["bash"], encoding="utf-8")
        require(subprocess.run(["bash", "-n", str(bash)], check=False).returncode == 0,
                "invalid bash completion")
        for shell in ("zsh", "fish"):
            executable = shutil.which(shell)
            if not executable:
                continue
            source = root / f"yvex.{shell}"
            source.write_text(outputs[shell], encoding="utf-8")
            require(subprocess.run([executable, "-n", str(source)], check=False).returncode == 0,
                    f"invalid {shell} completion")


def main() -> int:
    registry = read_registry()
    test_generation(registry)
    test_refusals(registry)
    test_audit_reconciliation(registry)
    test_compiled_discovery(registry)
    test_completion()
    print("operator registry: schema/generation/refusal/audit/discovery checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
