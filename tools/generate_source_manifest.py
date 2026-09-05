#!/usr/bin/env python3
"""Project the governed source inventory into deterministic Make variables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PRODUCTION_SUFFIXES = {".c", ".cu", ".h"}
MANIFEST_FIELDS = 9


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"source manifest: {message}")


def relative_files(root: Path, roots: tuple[str, ...], suffixes: set[str]) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for name in roots
        for path in (root / name).rglob("*")
        if path.is_file() and path.suffix in suffixes
    )


def load_owners(root: Path, manifest: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[str] = set()
    with manifest.open(newline="") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.startswith("#") or not raw.strip():
                continue
            fields = next(csv.reader([raw], delimiter="\t"))
            if len(fields) != MANIFEST_FIELDS:
                fail(f"{manifest}:{number}: expected {MANIFEST_FIELDS} fields")
            path = fields[0]
            if path in seen:
                fail(f"duplicate owner path: {path}")
            if not (root / path).is_file():
                fail(f"owned path does not exist: {path}")
            seen.add(path)
            rows.append(fields)

    actual = set(relative_files(root, ("src", "include"), PRODUCTION_SUFFIXES))
    registered = {row[0] for row in rows}
    if actual != registered:
        fail(
            "owner parity failed: "
            f"missing={sorted(actual - registered)} stale={sorted(registered - actual)}"
        )
    return sorted(rows, key=lambda row: row[0])


def tracked_test_sources(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", "tests"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        paths = result.stdout.splitlines()
    else:
        paths = relative_files(root, ("tests",), {".c"})
    return sorted(path for path in paths if path.endswith(".c") and (root / path).is_file())


def classify_production(rows: list[list[str]]) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {
        "OWNED_PRODUCTION_SRCS": [],
        "OWNED_PRODUCTION_HEADERS": [],
        "CORE_SRCS": [],
        "YVEX_SRCS": [],
        "OPENAI_ADAPTER_SRCS": [],
        "CUDA_SRCS": [],
        "CUDA_CU_SRCS": [],
        "CLI_COMMAND_SRCS": [],
        "CLI_INPUT_SRCS": [],
        "CLI_MODEL_ARTIFACT_SRCS": [],
        "CLI_RENDER_SRCS": [],
        "CLI_IO_SRCS": [],
    }
    for row in rows:
        path = row[0]
        suffix = Path(path).suffix
        if suffix == ".h":
            classes["OWNED_PRODUCTION_HEADERS"].append(path)
            continue
        classes["OWNED_PRODUCTION_SRCS"].append(path)
        if path.startswith("src/cli/"):
            classes["YVEX_SRCS"].append(path)
            if path.startswith("src/cli/commands/"):
                classes["CLI_COMMAND_SRCS"].append(path)
            elif path.startswith("src/cli/input/"):
                classes["CLI_INPUT_SRCS"].append(path)
            elif path.startswith("src/cli/model_artifacts/"):
                classes["CLI_MODEL_ARTIFACT_SRCS"].append(path)
            elif path.startswith("src/cli/render/"):
                classes["CLI_RENDER_SRCS"].append(path)
            elif path.startswith("src/cli/io/") and not path.endswith("/client.c"):
                classes["CLI_IO_SRCS"].append(path)
        elif path.startswith("src/server/openai/"):
            classes["OPENAI_ADAPTER_SRCS"].append(path)
        elif path.startswith("src/backend/cuda/") and suffix == ".cu":
            classes["CUDA_CU_SRCS"].append(path)
        elif path.startswith("src/backend/cuda/"):
            classes["CUDA_SRCS"].append(path)
        else:
            classes["CORE_SRCS"].append(path)

    product = (
        classes["CORE_SRCS"]
        + classes["YVEX_SRCS"]
        + classes["OPENAI_ADAPTER_SRCS"]
        + classes["CUDA_SRCS"]
        + classes["CUDA_CU_SRCS"]
    )
    if sorted(product) != classes["OWNED_PRODUCTION_SRCS"] or len(product) != len(set(product)):
        fail("production classification is incomplete or overlapping")
    return classes


def classify_tests(paths: list[str]) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {
        "TEST_UNIT_SRCS": [],
        "TEST_REFERENCE_SRCS": [],
        "CUDA_TEST_UNIT_SRCS": [],
        "TEST_RUNNER_SRCS": [],
        "TEST_LIVE_SRCS": [],
        "TEST_INTEGRATION_SRCS": [],
    }
    for path in paths:
        if path.startswith("tests/reference/"):
            key = "TEST_REFERENCE_SRCS"
        elif path.startswith("tests/unit/cuda/"):
            key = "CUDA_TEST_UNIT_SRCS"
        elif path in {"tests/test.c", "tests/test_cuda.c"} or path.endswith("_runner.c"):
            key = "TEST_RUNNER_SRCS"
        elif path.startswith("tests/unit/"):
            key = "TEST_UNIT_SRCS"
        elif path.startswith("tests/live/"):
            key = "TEST_LIVE_SRCS"
        elif path.startswith("tests/integration/"):
            key = "TEST_INTEGRATION_SRCS"
        else:
            fail(f"unclassified tracked C test: {path}")
        classes[key].append(path)
    classified = [path for values in classes.values() for path in values]
    if sorted(classified) != paths or len(classified) != len(set(classified)):
        fail("test classification is incomplete or overlapping")
    return classes


def make_assignment(name: str, values: list[str]) -> str:
    if not values:
        return f"{name} :=\n"
    lines = [f"{name} := \\"]
    lines.extend(
        f"\t{value}{' \\' if index + 1 < len(values) else ''}"
        for index, value in enumerate(values)
    )
    return "\n".join(lines) + "\n"


def render_make(root: Path, rows: list[list[str]]) -> str:
    variables = classify_production(rows)
    variables.update(classify_tests(tracked_test_sources(root)))
    material = "\n".join("\t".join(row) for row in rows).encode()
    identity = hashlib.sha256(material).hexdigest()
    sections = [
        "# Generated by tools/generate_source_manifest.py; do not edit.",
        f"SOURCE_MANIFEST_IDENTITY := {identity}",
        "",
    ]
    for name, values in variables.items():
        sections.append(make_assignment(name, sorted(values)).rstrip())
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_family_header(rows: list[list[str]]) -> str:
    prefix = "src/graph/families/"
    names = [
        Path(row[0]).stem
        for row in rows
        if row[0].startswith(prefix) and Path(row[0]).suffix == ".c"
    ]
    if len(names) != len(set(names)):
        fail("graph family provider names are not unique")
    lines = [
        "/* Generated by tools/generate_source_manifest.py; do not edit. */",
        "#ifndef YVEX_GENERATED_SOURCE_FAMILIES_H_INCLUDED",
        "#define YVEX_GENERATED_SOURCE_FAMILIES_H_INCLUDED",
        "",
        f"#define YVEX_GRAPH_FAMILY_DESCRIPTOR_COUNT {len(names)}u",
    ]
    if names:
        lines.append("#define YVEX_GRAPH_FAMILY_DESCRIPTORS(X) \\")
        lines.extend(
            f"    X({name}){' \\' if index + 1 < len(names) else ''}"
            for index, name in enumerate(names)
        )
    else:
        lines.append("#define YVEX_GRAPH_FAMILY_DESCRIPTORS(X)")
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    if path.is_file() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("config/source_owners.tsv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-header", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    rows = load_owners(root, manifest)
    expected = render_make(root, rows)
    family_header = arguments.family_header
    if family_header and not family_header.is_absolute():
        family_header = root / family_header
    expected_family = render_family_header(rows) if family_header else None
    if arguments.check:
        if not output.is_file() or output.read_text() != expected:
            fail(f"generated projection is stale: {output}")
        if (family_header and
                (not family_header.is_file() or family_header.read_text() != expected_family)):
            fail(f"generated family projection is stale: {family_header}")
        print(f"source manifest: ok ({expected.count(chr(10))} lines)")
        return 0
    write_atomic(output, expected)
    if family_header and expected_family is not None:
        write_atomic(family_header, expected_family)
    return 0


if __name__ == "__main__":
    sys.exit(main())
