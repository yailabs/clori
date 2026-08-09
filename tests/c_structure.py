#!/usr/bin/env python3
"""Canonical structural audit for YVEX-owned C and CUDA code.

The scanner is deliberately dependency-free.  It lexes enough C structure to
identify includes, top-level function declarations/definitions, comments,
balanced scopes, and source ownership without pretending to compile C.  The
compiler remains the authority for language correctness; this tool owns the
repository's structural policy and machine-readable inventory.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "config/c_policy.json"
GROUPS = ("ownership", "layout", "architecture", "natural")


@dataclass(frozen=True)
class Comment:
    start: int
    end: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class Include:
    line: int
    delimiter: str
    name: str


@dataclass(frozen=True)
class Function:
    name: str
    start: int
    body_start: int
    end: int
    start_line: int
    end_line: int
    is_static: bool
    signature: str
    body: str

    @property
    def lines(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class CUnit:
    path: Path
    text: str
    masked: str
    comment_stripped: str
    comments: list[Comment]
    includes: list[Include]
    functions: list[Function]


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    offsets.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    return offsets


def line_number(offsets: Sequence[int], position: int) -> int:
    return bisect.bisect_right(offsets, position)


def lex_c(text: str) -> tuple[str, str, list[Comment]]:
    """Mask literals/comments while preserving byte positions and newlines."""

    masked = list(text)
    comment_stripped = list(text)
    offsets = line_offsets(text)
    comments: list[Comment] = []
    index = 0
    size = len(text)
    state = "code"
    comment_start = -1

    while index < size:
        char = text[index]
        following = text[index + 1] if index + 1 < size else ""
        if state == "code":
            if char == "/" and following == "*":
                comment_start = index
                masked[index] = masked[index + 1] = " "
                comment_stripped[index] = comment_stripped[index + 1] = " "
                index += 2
                state = "block-comment"
                continue
            if char == "/" and following == "/":
                comment_start = index
                masked[index] = masked[index + 1] = " "
                comment_stripped[index] = comment_stripped[index + 1] = " "
                index += 2
                state = "line-comment"
                continue
            if char == '"':
                masked[index] = " "
                index += 1
                state = "string"
                continue
            if char == "'":
                masked[index] = " "
                index += 1
                state = "character"
                continue
            index += 1
            continue

        if state == "block-comment":
            if char == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                comment_stripped[index] = comment_stripped[index + 1] = " "
                end = index + 2
                comments.append(
                    Comment(
                        comment_start,
                        end,
                        line_number(offsets, comment_start),
                        line_number(offsets, end - 1),
                        text[comment_start:end],
                    )
                )
                index = end
                state = "code"
                continue
            if char != "\n":
                masked[index] = " "
                comment_stripped[index] = " "
            index += 1
            continue

        if state == "line-comment":
            if char == "\n":
                comments.append(
                    Comment(
                        comment_start,
                        index,
                        line_number(offsets, comment_start),
                        line_number(offsets, max(comment_start, index - 1)),
                        text[comment_start:index],
                    )
                )
                state = "code"
                index += 1
                continue
            masked[index] = " "
            comment_stripped[index] = " "
            index += 1
            continue

        quote = '"' if state == "string" else "'"
        if char == "\\":
            masked[index] = " "
            if index + 1 < size and text[index + 1] != "\n":
                masked[index + 1] = " "
            index += 2
            continue
        if char == quote:
            masked[index] = " "
            state = "code"
            index += 1
            continue
        if char != "\n":
            masked[index] = " "
        index += 1

    if state == "line-comment":
        comments.append(
            Comment(
                comment_start,
                size,
                line_number(offsets, comment_start),
                line_number(offsets, max(comment_start, size - 1)),
                text[comment_start:size],
            )
        )

    return "".join(masked), "".join(comment_stripped), merge_line_comments(text, comments)


def merge_line_comments(text: str, comments: Sequence[Comment]) -> list[Comment]:
    merged: list[Comment] = []
    for comment in comments:
        if (
            merged
            and merged[-1].text.lstrip().startswith("//")
            and comment.text.lstrip().startswith("//")
            and not text[merged[-1].end : comment.start].strip()
        ):
            previous = merged.pop()
            merged.append(
                Comment(
                    previous.start,
                    comment.end,
                    previous.start_line,
                    comment.end_line,
                    text[previous.start : comment.end],
                )
            )
        else:
            merged.append(comment)
    return merged


def signature_start(masked: str, lower: int, upper: int) -> int:
    """Return the first meaningful line after directives/comments/blank space."""

    candidate = lower
    position = lower
    while position < upper:
        end = masked.find("\n", position, upper)
        if end < 0:
            end = upper
        stripped = masked[position:end].strip()
        if not stripped or stripped.startswith("#"):
            candidate = min(upper, end + 1)
        position = end + 1
    while candidate < upper and masked[candidate].isspace():
        candidate += 1
    return candidate


def function_candidate(masked: str, lower: int, brace: int) -> tuple[str, int, bool] | None:
    line_start = masked.rfind("\n", 0, brace) + 1
    if re.match(r"\s*#", masked[line_start:brace]):
        return None
    cursor = brace - 1
    while cursor >= lower and masked[cursor].isspace():
        cursor -= 1
    if cursor < lower or masked[cursor] != ")":
        return None

    nesting = 1
    cursor -= 1
    while cursor >= lower and nesting:
        if masked[cursor] == ")":
            nesting += 1
        elif masked[cursor] == "(":
            nesting -= 1
        cursor -= 1
    if nesting:
        return None

    name_end = cursor + 1
    while cursor >= lower and masked[cursor].isspace():
        cursor -= 1
    name_end = cursor + 1
    while cursor >= lower and (masked[cursor].isalnum() or masked[cursor] == "_"):
        cursor -= 1
    name = masked[cursor + 1 : name_end]
    if not name or name in {"if", "for", "while", "switch", "sizeof", "_Alignof"}:
        return None

    start = signature_start(masked, lower, brace)
    signature = masked[start:brace]
    if "=" in signature or re.search(r"\btypedef\b", signature):
        return None
    if re.match(r"\s*#", signature):
        return None
    is_static = bool(re.search(r"\bstatic\b", signature[: max(0, cursor + 1 - start)]))
    return name, start, is_static


def find_functions(text: str, masked: str) -> list[Function]:
    offsets = line_offsets(text)
    functions: list[Function] = []
    depth = 0
    lower = 0
    active: tuple[str, int, bool, int] | None = None

    for index, char in enumerate(masked):
        if char == "{":
            if depth == 0:
                candidate = function_candidate(masked, lower, index)
                if candidate:
                    active = (*candidate, index)
                else:
                    active = None
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                if active:
                    name, start, is_static, body_start = active
                    end = index + 1
                    functions.append(
                        Function(
                            name,
                            start,
                            body_start,
                            end,
                            line_number(offsets, start),
                            line_number(offsets, index),
                            is_static,
                            text[start:body_start],
                            text[body_start:end],
                        )
                    )
                lower = index + 1
                active = None
            continue
        if depth == 0 and char == ";":
            lower = index + 1
    return functions


def parse_includes(text: str) -> list[Include]:
    expression = re.compile(r"^\s*#\s*include\s*([\"<])([^\">]+)[\">]", re.MULTILINE)
    return [
        Include(text.count("\n", 0, match.start()) + 1, match.group(1), match.group(2))
        for match in expression.finditer(text)
    ]


def parse_unit(path: Path) -> CUnit:
    text = path.read_text(errors="ignore")
    masked, comment_stripped, comments = lex_c(text)
    return CUnit(
        path,
        text,
        masked,
        comment_stripped,
        comments,
        parse_includes(text),
        find_functions(text, masked),
    )


def adjacent_comment(unit: CUnit, position: int) -> Comment | None:
    prior = [comment for comment in unit.comments if comment.end <= position]
    if not prior:
        return None
    comment = prior[-1]
    if unit.text[comment.end:position].strip():
        return None
    return comment


def declarations(unit: CUnit) -> set[str]:
    found: set[str] = set()
    expression = re.compile(
        r"(?<!\(\*)\b(yvex_[A-Za-z0-9_]+)\s*\([^;{}]*\)\s*;",
        re.DOTALL,
    )
    for match in expression.finditer(unit.masked):
        prefix = unit.masked[max(0, match.start() - 80) : match.start()]
        if re.search(r"\bstatic\s+(?:inline\s+)?$", prefix):
            continue
        found.add(match.group(1))
    return found


def data_declarations(unit: CUnit) -> set[str]:
    found: set[str] = set()
    expression = re.compile(
        r"\bextern\s+(?:const\s+)?(?:struct\s+)?[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\s*\*\s*|\s+)(yvex_[A-Za-z0-9_]+)(?:\s*\[[^;]*\])?\s*;"
    )
    for match in expression.finditer(unit.masked):
        found.add(match.group(1))
    return found


def strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, set()):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        if len(component) > 1:
            cycles.append(sorted(component))

    for node in graph:
        if node not in indices:
            visit(node)
    return sorted(cycles)


def path_has_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


class Audit:
    def __init__(self, policy_path: Path):
        self.policy = json.loads(policy_path.read_text())
        if self.policy.get("schema_version") != 1:
            raise ValueError("unsupported C policy schema")
        suffixes = set(self.policy["production_suffixes"])
        self.files = sorted(
            path
            for root_name in self.policy["production_roots"]
            for path in (ROOT / root_name).rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
        self.units = {relative(path): parse_unit(path) for path in self.files}
        self.translation_units = {
            name: unit for name, unit in self.units.items() if unit.path.suffix in {".c", ".cu"}
        }
        self.headers = {
            name: unit for name, unit in self.units.items() if unit.path.suffix == ".h"
        }
        self.test_translation_units = {
            relative(path): parse_unit(path)
            for path in sorted((ROOT / "tests").rglob("*"))
            if path.is_file() and path.suffix in {".c", ".cu"}
        }
        self.manifest_rows, self.manifest_errors = self.load_manifest()
        self.manifest = {row[0]: row for row in self.manifest_rows}
        self.include_targets = self.resolve_includes()
        self._make_database: str | None = None

    def load_manifest(self) -> tuple[list[list[str]], list[str]]:
        path = ROOT / self.policy["manifest"]["path"]
        rows: list[list[str]] = []
        errors: list[str] = []
        if not path.is_file():
            return rows, [f"missing ownership manifest: {relative(path)}"]
        with path.open(newline="") as stream:
            for number, raw in enumerate(stream, 1):
                if raw.startswith("#") or not raw.strip():
                    continue
                fields = next(csv.reader([raw], delimiter="\t"))
                if len(fields) != self.policy["manifest"]["fields"]:
                    errors.append(f"manifest line {number}: expected nine fields")
                    continue
                rows.append(fields)
        return rows, errors

    def resolve_include(self, source: str, include: Include) -> str | None:
        name = include.name
        if name.startswith("yvex/"):
            candidate = ROOT / "include" / name
            return relative(candidate) if candidate.is_file() else None
        if name.startswith("src/") or name.startswith("include/"):
            candidate = ROOT / name
            return relative(candidate) if candidate.is_file() else None
        if include.delimiter == '"':
            local = (ROOT / source).parent / name
            if local.is_file():
                return relative(local)
            matches = [path for path in self.files if path.name == name]
            if len(matches) == 1:
                return relative(matches[0])
        return None

    def resolve_includes(self) -> dict[str, list[tuple[Include, str | None]]]:
        return {
            source: [(include, self.resolve_include(source, include)) for include in unit.includes]
            for source, unit in self.units.items()
        }

    def make_database(self) -> str:
        if self._make_database is None:
            result = subprocess.run(
                ["make", "-pn"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self._make_database = result.stdout
        return self._make_database

    def make_variable(self, name: str) -> list[str]:
        match = re.search(rf"^{re.escape(name)} := (.*)$", self.make_database(), re.MULTILINE)
        return match.group(1).split() if match else []

    def header_tier(self, path: str) -> str | None:
        tiers = self.policy["header_tiers"]
        if path_has_prefix(path, tiers["internal_root"]):
            return "internal"
        if Path(path).parent == Path(tiers["public_root"]):
            return "public"
        if path_has_prefix(path, tiers["source_root"]):
            return "source"
        return None

    def leading_contract(self, unit: CUnit) -> Comment | None:
        if not unit.comments:
            return None
        first = unit.comments[0]
        if unit.text[: first.start].strip():
            return None
        return first

    def commentary_paths(self) -> list[Path]:
        """Return every first-party code or build file governed by commentary policy."""
        policy = self.policy["commentary"]
        suffixes = set(policy["suffixes"])
        paths: set[Path] = set()
        for root_name in policy["roots"]:
            root = ROOT / root_name
            if not root.exists():
                continue
            paths.update(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in suffixes
            )
        paths.update(ROOT / name for name in policy["build_files"] if (ROOT / name).is_file())
        excluded = tuple(policy["excluded_prefixes"])
        return sorted(path for path in paths if not relative(path).startswith(excluded))

    @staticmethod
    def normalize_comment(text: str) -> str:
        lines = []
        for raw in text.splitlines():
            line = re.sub(r"\s*\*/\s*$", "", raw)
            line = re.sub(r"^\s*(?:/\*+|\*+|//+|#+)\s?", "", line).strip()
            if line:
                lines.append(line)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def commentary_comments(self) -> Iterator[tuple[str, int, str]]:
        """Yield C, Python, shell, and Make comments without scanning literals as prose."""
        for path in self.commentary_paths():
            name = relative(path)
            if path.suffix in {".c", ".cu", ".h"}:
                for comment in parse_unit(path).comments:
                    yield name, comment.start_line, self.normalize_comment(comment.text)
                continue
            if path.suffix == ".py":
                try:
                    tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                    for token in tokens:
                        if token.type == tokenize.COMMENT and not token.string.startswith("#!"):
                            yield name, token.start[0], token.string.lstrip("#").strip()
                except (SyntaxError, tokenize.TokenError) as exc:
                    yield name, 0, f"COMMENTARY_SCANNER_PARSE_ERROR {exc}"
                continue
            for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#") and not stripped.startswith("#!"):
                    yield name, number, stripped.lstrip("#").strip()

    def metrics(self) -> dict[str, object]:
        physical = blank = comment_only = code = executable = 0
        functions: list[tuple[str, Function]] = []
        hard_width = self.policy["limits"]["hard_line_width"]
        long_lines = 0
        file_lines: dict[str, int] = {}
        suffixes: Counter[str] = Counter()
        for name, unit in self.units.items():
            raw = unit.text.splitlines()
            stripped = unit.comment_stripped.splitlines()
            file_lines[name] = len(raw)
            suffixes[unit.path.suffix] += 1
            physical += len(raw)
            for original, visible in zip(raw, stripped):
                if not original.strip():
                    blank += 1
                elif not visible.strip():
                    comment_only += 1
                else:
                    code += 1
                if len(original.expandtabs(8)) > hard_width:
                    long_lines += 1
            functions.extend((name, function) for function in unit.functions)
            if unit.path.suffix in {".c", ".cu"}:
                for function in unit.functions:
                    body = unit.comment_stripped[function.body_start : function.end]
                    for line in body.splitlines():
                        visible = line.strip()
                        if (
                            visible
                            and not visible.startswith("#")
                            and not re.fullmatch(r"[{}]+;?", visible)
                        ):
                            executable += 1

        tiers = Counter(self.header_tier(path) or "invalid" for path in self.headers)
        commented = 0
        for name, function in functions:
            comment = adjacent_comment(self.units[name], function.start)
            if comment:
                commented += 1

        archive = self.archive_snapshot()
        largest_file = max(file_lines, key=file_lines.get, default="")
        largest_function = max(functions, key=lambda item: item[1].lines, default=None)
        basenames = [len(Path(name).name) for name in self.units]
        consumers = self.header_consumers()
        one_consumer_headers = sorted(
            name
            for name in self.headers
            if self.header_tier(name) in {"internal", "source"} and len(consumers[name]) == 1
        )
        subsystem_counts = Counter(row[1] for row in self.manifest_rows)
        owner_counts = Counter(row[2] for row in self.manifest_rows)
        include_facts = self.include_snapshot()
        return {
            "production_files": len(self.units),
            "translation_units": len(self.translation_units),
            "headers": len(self.headers),
            "files_by_suffix": dict(sorted(suffixes.items())),
            "files_by_subsystem": dict(sorted(subsystem_counts.items())),
            "header_tiers": dict(tiers),
            "physical_lines": physical,
            "code_lines": code,
            "executable_lines": executable,
            "comment_only_lines": comment_only,
            "blank_lines": blank,
            "production_bytes": sum(len(unit.text.encode()) for unit in self.units.values()),
            "largest_file": {"path": largest_file, "lines": file_lines.get(largest_file, 0)},
            "functions": len(functions),
            "translation_unit_functions": sum(
                len(unit.functions) for unit in self.translation_units.values()
            ),
            "header_defined_functions": sum(len(unit.functions) for unit in self.headers.values()),
            "functions_over_100": sum(function.lines > 100 for _, function in functions),
            "functions_over_200": sum(function.lines > 200 for _, function in functions),
            "functions_over_500": sum(function.lines > 500 for _, function in functions),
            "maximum_function_lines": max((function.lines for _, function in functions), default=0),
            "largest_function": {
                "path": largest_function[0] if largest_function else "",
                "name": largest_function[1].name if largest_function else "",
                "line": largest_function[1].start_line if largest_function else 0,
                "lines": largest_function[1].lines if largest_function else 0,
            },
            "functions_with_adjacent_commentary": commented,
            "interface_headers_with_module_commentary": sum(
                self.leading_contract(unit) is not None for unit in self.headers.values()
            ),
            "reviewed_commentary_files": len(self.commentary_paths()),
            "hard_width_violations": long_lines,
            "same_stem_pairs": len(self.same_stem_pairs()),
            "include_facts": include_facts,
            "bare_quoted_includes": include_facts["bare_quoted"],
            "umbrella_source_users": include_facts["umbrella_source_users"],
            "one_consumer_nonpublic_headers": len(one_consumer_headers),
            "one_consumer_nonpublic_header_paths": one_consumer_headers,
            "average_basename_characters": (
                round(sum(basenames) / len(basenames), 3) if basenames else 0
            ),
            "maximum_basename_characters": max(basenames, default=0),
            "semantic_owners": len({row[2] for row in self.manifest_rows}),
            "files_per_semantic_owner": {
                "maximum": max(owner_counts.values(), default=0),
                "histogram": dict(sorted(Counter(owner_counts.values()).items())),
            },
            "duplicate_function_blocks": self.duplicate_block_snapshot(),
            "archive": archive,
            "symbols": self.symbol_snapshot(),
        }

    def header_consumers(self) -> dict[str, set[str]]:
        consumers: dict[str, set[str]] = defaultdict(set)
        for source, edges in self.include_targets.items():
            if source not in self.translation_units:
                continue
            for _include, target in edges:
                if target in self.headers:
                    consumers[target].add(source)
        return consumers

    def include_snapshot(self) -> dict[str, int]:
        by_basename: dict[str, list[str]] = defaultdict(list)
        for name in self.units:
            by_basename[Path(name).name].append(name)
        bare = ambiguous = unresolved_owned = umbrella = 0
        for source, edges in self.include_targets.items():
            for include, target in edges:
                if include.delimiter == '"' and "/" not in include.name:
                    bare += 1
                    if len(by_basename[include.name]) > 1:
                        ambiguous += 1
                if (
                    include.delimiter == '"'
                    and include.name.startswith(("src/", "include/"))
                    and target is None
                ):
                    unresolved_owned += 1
                if source.startswith("src/") and include.name == "yvex/api.h":
                    umbrella += 1
        return {
            "bare_quoted": bare,
            "ambiguous_quoted": ambiguous,
            "unresolved_owned": unresolved_owned,
            "umbrella_source_users": umbrella,
        }

    def duplicate_block_snapshot(self) -> dict[str, object]:
        minimum = self.policy["analysis"]["duplicate_block_minimum_tokens"]
        groups: dict[str, list[str]] = defaultdict(list)
        for name, unit in self.units.items():
            for function in unit.functions:
                tokens = re.findall(
                    r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9A-Fa-f]+|[0-9]+(?:\.[0-9]+)?|[^\s]",
                    unit.masked[function.body_start : function.end],
                )
                if len(tokens) < minimum:
                    continue
                digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
                groups[digest].append(f"{name}:{function.start_line}:{function.name}")
        duplicates = [locations for locations in groups.values() if len(locations) > 1]
        duplicates.sort(key=lambda locations: (-len(locations), locations))
        return {
            "groups": len(duplicates),
            "functions": sum(len(locations) for locations in duplicates),
            "sample": duplicates[:20],
        }

    def same_stem_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for name in self.translation_units:
            header = str(Path(name).with_suffix(".h"))
            if header in self.headers:
                pairs.append((name, header))
        return sorted(pairs)

    def unadmitted_same_stem_pairs(self) -> list[tuple[str, str]]:
        result = []
        for source, header in self.same_stem_pairs():
            row = self.manifest.get(header)
            if row and (
                row[5] == "generated-abi"
                or (row[3] == "backend" and row[5] in {"platform-interface", "backend-kernel"})
            ):
                continue
            result.append((source, header))
        return result

    def archive_snapshot(self) -> dict[str, object]:
        archive = ROOT / self.policy["archive"]["path"]
        if not archive.is_file():
            return {"present": False}
        result = subprocess.run(
            ["ar", "t", str(archive)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        members = result.stdout.splitlines() if result.returncode == 0 else []
        return {
            "present": True,
            "readable": result.returncode == 0,
            "members": len(members),
            "unique_members": len(set(members)),
            "duplicate_excess": len(members) - len(set(members)),
            "duplicate_names": {
                name: count for name, count in Counter(members).items() if count > 1
            },
        }

    def nm_symbols(self) -> tuple[dict[str, list[str]], list[str]]:
        archive = ROOT / self.policy["archive"]["path"]
        if not archive.is_file():
            return {}, []
        result = subprocess.run(
            ["nm", "-A", "-g", "--defined-only", str(archive)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        definitions: dict[str, list[str]] = defaultdict(list)
        foreign: list[str] = []
        namespace = self.policy["symbols"]["namespace"]
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 3 or not re.fullmatch(r"[A-ZTRDB]", fields[-2]):
                continue
            symbol = fields[-1]
            owner = " ".join(fields[:-2])
            if symbol.startswith(namespace):
                definitions[symbol].append(owner)
            elif not symbol.startswith(("_", ".")):
                foreign.append(symbol)
        return dict(definitions), sorted(set(foreign))

    def nm_undefined_consumers(self) -> dict[str, list[str]]:
        """Return production object consumers after preprocessing and macro expansion."""
        archive = ROOT / self.policy["archive"]["path"]
        if not archive.is_file():
            return {}
        result = subprocess.run(
            ["nm", "-A", "-u", str(archive)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        consumers: dict[str, set[str]] = defaultdict(set)
        namespace = self.policy["symbols"]["namespace"]
        for line in result.stdout.splitlines():
            match = re.match(r"^(.*?):\s+U\s+(\S+)$", line)
            if not match:
                continue
            owner, symbol = match.groups()
            if symbol.startswith(namespace):
                consumers[symbol].add(owner)
        return {symbol: sorted(owners) for symbol, owners in consumers.items()}

    def public_declarations(self) -> set[str]:
        return {
            symbol
            for name, unit in self.headers.items()
            if self.header_tier(name) == "public"
            for symbol in declarations(unit)
        }

    def private_declarations(self) -> set[str]:
        return {
            symbol
            for name, unit in self.headers.items()
            if self.header_tier(name) in {"internal", "source"}
            for symbol in declarations(unit) | data_declarations(unit)
        }

    def symbol_snapshot(self) -> dict[str, object]:
        definitions, foreign = self.nm_symbols()
        public = self.public_declarations()
        defined = set(definitions)
        return {
            "public_function_declarations": len(public),
            "library_globals": len(defined),
            "public_defined": len(public & defined),
            "nonpublic_globals": len(defined - public),
            "duplicate_definitions": sum(len(owners) > 1 for owners in definitions.values()),
            "foreign_globals": len(foreign),
        }

    def ownership_violations(self) -> list[str]:
        errors = list(self.manifest_errors)
        actual = set(self.units)
        paths = [row[0] for row in self.manifest_rows]
        registered = set(paths)
        if actual != registered:
            errors.append(
                "manifest parity: "
                f"missing={sorted(actual - registered)} stale={sorted(registered - actual)}"
            )
        duplicates = sorted(path for path, count in Counter(paths).items() if count != 1)
        if duplicates:
            errors.append(f"duplicate manifest paths: {duplicates}")

        manifest_policy = self.policy["manifest"]
        owners: dict[str, list[str]] = defaultdict(list)
        owner_partitions: Counter[tuple[str, str]] = Counter()
        families: dict[str, list[str]] = defaultdict(list)
        family_headers: dict[str, list[str]] = defaultdict(list)
        family_subsystems: Counter[tuple[str, str]] = Counter()
        for row in self.manifest_rows:
            path, subsystem, owner, scope, visibility, boundary, consumers, partition, exception = row
            if not subsystem or not owner or not boundary or not consumers:
                errors.append(f"incomplete manifest row: {path}")
            if scope not in manifest_policy["scopes"]:
                errors.append(f"invalid scope: {path}: {scope}")
            if visibility not in manifest_policy["visibilities"]:
                errors.append(f"invalid visibility: {path}: {visibility}")
            if partition not in manifest_policy["partitions"]:
                errors.append(f"invalid partition: {path}: {partition}")
            if exception != manifest_policy["allowed_exception"]:
                errors.append(f"unadmitted exception: {path}: {exception}")
            owners[owner].append(path)
            owner_partitions[(owner, partition)] += 1

            tier = self.header_tier(path) if path.endswith(".h") else None
            if tier == "public" and (visibility != "public" or boundary != "public-abi"):
                errors.append(f"public header has non-public manifest facts: {path}")
            if tier in {"internal", "source"} and visibility != "private":
                errors.append(f"non-public header is marked public: {path}")

            if scope == "family":
                family = Path(path).stem
                if path.endswith((".c", ".cu")):
                    if not re.match(
                        r"^src/(model|graph|backend/cuda)/families/[^/]+\.(c|cu)$", path
                    ):
                        errors.append(f"invalid family source path: {path}")
                    if re.search(
                        r"_(plan|execute|sink|numeric|internal|reference|report)\.(c|cu)$", path
                    ):
                        errors.append(f"family phase file: {path}")
                    families[family].append(path)
                    family_subsystems[(family, subsystem)] += 1
                elif path.endswith(".h"):
                    if not re.match(r"^include/yvex/internal/families/[^/]+\.h$", path):
                        errors.append(f"invalid family header path: {path}")
                    family_headers[family].append(path)
                else:
                    errors.append(f"invalid family owner path: {path}")
            elif "/families/" in path:
                errors.append(f"family path lacks family scope: {path}")

        maximum = self.policy["limits"]["files_per_semantic_owner"]
        for owner, owner_paths in sorted(owners.items()):
            if len(owner_paths) > maximum:
                errors.append(f"fragmented semantic owner: {owner}: {owner_paths}")
        for key, count in sorted(owner_partitions.items()):
            if count > 1:
                errors.append(f"duplicate owner partition: {key}: {count}")

        family_limit = self.policy["limits"]["family_sources"]
        family_header_limit = self.policy["limits"]["family_headers"]
        subsystem_limit = self.policy["limits"]["family_sources_per_subsystem"]
        for family, family_paths in sorted(families.items()):
            if len(family_paths) > family_limit:
                errors.append(f"family budget exceeded: {family}: {family_paths}")
        for key, count in sorted(family_subsystems.items()):
            if count > subsystem_limit:
                errors.append(f"family subsystem budget exceeded: {key}: {count}")
        for family, header_paths in sorted(family_headers.items()):
            if len(header_paths) > family_header_limit:
                errors.append(f"family header budget exceeded: {family}: {header_paths}")

        consumers = self.header_consumers()
        minimum = manifest_policy["private_header_minimum_translation_unit_consumers"]
        special = set(self.policy["header_tiers"]["source_special_boundaries"])
        private_by_subsystem: dict[str, list[str]] = defaultdict(list)
        for path in sorted(self.headers):
            if self.header_tier(path) != "source":
                continue
            row = self.manifest.get(path)
            boundary = row[5] if row else ""
            if Path(path).name == self.policy["header_tiers"]["source_private_basename"]:
                if row:
                    private_by_subsystem[row[1]].append(path)
                if len(consumers[path]) < minimum:
                    errors.append(
                        f"private header lacks {minimum} translation-unit consumers: "
                        f"{path}: {sorted(consumers[path])}"
                    )
            elif boundary not in special:
                errors.append(f"source header is neither private.h nor platform/generated ABI: {path}")
            elif boundary in {"platform-interface", "backend-kernel"} and (not row or row[3] != "backend"):
                errors.append(f"platform/backend header lacks backend scope: {path}")
        for subsystem, private_headers in sorted(private_by_subsystem.items()):
            if len(private_headers) > 1:
                errors.append(
                    f"subsystem owns multiple private.h contracts: {subsystem}: {private_headers}"
                )
        return errors

    def layout_violations(self) -> list[str]:
        errors: list[str] = []
        limits = self.policy["limits"]
        suffixes = set(self.policy["owned_suffixes"])
        owned = sorted(
            path
            for root_name in self.policy["owned_roots"]
            if (ROOT / root_name).exists()
            for path in (ROOT / root_name).rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
        for path in owned:
            name = relative(path)
            basename = path.name
            stem = path.stem
            if basename.startswith("yvex_"):
                errors.append(f"project prefix in filename: {name}")
            if len(basename) > limits["basename_characters"]:
                errors.append(f"basename exceeds limit: {name}")
            if not re.fullmatch(r"[a-z0-9_]+", stem):
                errors.append(f"filename is not lowercase snake_case: {name}")
            if re.search(r"_(internal|private)\.(c|h|cu|sh|py)$", basename):
                errors.append(f"forbidden phase/private filename: {name}")
            if len(Path(name).parts) > limits["path_components"]:
                errors.append(f"path exceeds component limit: {name}")
            if path.parent.name and path.parent.name in stem.split("_"):
                errors.append(f"basename repeats immediate directory: {name}")

        metrics = self.metrics()
        for key in (
            "production_files",
            "translation_units",
            "headers",
            "physical_lines",
            "code_lines",
            "executable_lines",
            "semantic_owners",
        ):
            if int(metrics[key]) > limits[key]:
                errors.append(f"{key} exceeds policy: {metrics[key]} > {limits[key]}")

        tier_counts = Counter(self.header_tier(path) or "invalid" for path in self.headers)
        for tier, limit_key in (
            ("public", "public_headers"),
            ("internal", "internal_headers"),
            ("source", "source_headers"),
        ):
            if tier_counts[tier] > limits[limit_key]:
                errors.append(
                    f"{tier} header count exceeds policy: {tier_counts[tier]} > {limits[limit_key]}"
                )
        if tier_counts["invalid"]:
            invalid = [path for path in self.headers if self.header_tier(path) is None]
            errors.append(f"headers outside admitted tiers: {invalid}")

        for name, unit in self.units.items():
            lines = unit.text.splitlines()
            maximum = (
                limits["header_lines"] if name in self.headers else limits["translation_unit_lines"]
            )
            if len(lines) > maximum:
                errors.append(f"file exceeds line limit: {name}: {len(lines)} > {maximum}")
            for number, line in enumerate(lines, 1):
                width = len(line.expandtabs(8))
                if width > limits["hard_line_width"]:
                    errors.append(
                        f"hard line width exceeded: {name}:{number}: "
                        f"{width} > {limits['hard_line_width']}"
                    )

        pairs = self.unadmitted_same_stem_pairs()
        if pairs:
            errors.append(f"same-stem implementation/header pairs: {pairs}")

        include_policy = self.policy["include_policy"]
        umbrella = self.policy["header_tiers"]["umbrella"]
        for source, edges in self.include_targets.items():
            tier = self.header_tier(source) if source in self.headers else None
            duplicate_includes = sorted(
                name for name, count in Counter(include.name for include, _target in edges).items()
                if count > 1
            )
            if duplicate_includes:
                errors.append(f"duplicate includes: {source}: {duplicate_includes}")
            for include, target in edges:
                if include.delimiter == '"':
                    if "/" not in include.name:
                        errors.append(f"bare quoted include: {source}:{include.line}: {include.name}")
                    elif not include.name.startswith(include_policy["source_prefix"]):
                        errors.append(
                            f"quoted include is not repository-qualified source path: "
                            f"{source}:{include.line}: {include.name}"
                        )
                    if target is None:
                        errors.append(f"unresolved owned include: {source}:{include.line}: {include.name}")
                elif include.name.startswith(include_policy["public_prefix"]):
                    if target is None:
                        errors.append(f"missing YVEX header: {source}:{include.line}: {include.name}")
                if source.startswith("src/") and include.name == "yvex/api.h":
                    errors.append(f"production source imports public umbrella: {source}:{include.line}")
                if tier == "public" and target:
                    target_tier = self.header_tier(target)
                    if target_tier != "public":
                        errors.append(f"public header depends on {target_tier}: {source} -> {target}")
                if source == umbrella and target and self.header_tier(target) != "public":
                    errors.append(f"umbrella exposes non-public header: {target}")

        forwarding_allow = set(include_policy["forwarding_header_allow"])
        for name, unit in self.headers.items():
            if name in forwarding_allow:
                continue
            meaningful: list[str] = []
            for line in unit.comment_stripped.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    if re.match(r"#\s*(?:include|pragma\s+once|ifn?def|endif)\b", stripped):
                        continue
                    if re.match(r"#\s*define\s+[A-Z0-9_]+(?:_H)?\s*$", stripped):
                        continue
                meaningful.append(stripped)
            if not meaningful:
                errors.append(f"forwarding/empty compatibility header: {name}")

        errors.extend(self.make_layout_violations())
        errors.extend(self.archive_violations())
        errors.extend(self.header_compile_violations())
        errors.extend(self.stale_path_violations())
        return errors

    def make_layout_violations(self) -> list[str]:
        errors: list[str] = []
        makefile = (ROOT / "Makefile").read_text()
        style_path = ROOT / ".clang-format"
        if not style_path.is_file():
            errors.append("canonical .clang-format is missing")
        else:
            style = style_path.read_text()
            column = re.search(r"^ColumnLimit:\s*(\d+)\s*$", style, re.MULTILINE)
            expected_column = self.policy["limits"]["preferred_line_width"]
            if not column or int(column.group(1)) != expected_column:
                errors.append(
                    ".clang-format ColumnLimit is not canonical: "
                    f"{column.group(1) if column else 'missing'} != {expected_column}"
                )
        roots = re.search(r"^CPPFLAGS \?= (.*)$", makefile, re.MULTILINE)
        observed = [token[2:] for token in roots.group(1).split() if token.startswith("-I")] if roots else []
        expected = self.policy["include_policy"]["make_include_roots"]
        if observed != expected:
            errors.append(f"Makefile include roots are not canonical: {observed} != {expected}")
        cflags = re.search(
            r"^CFLAGS\s*(?:[:+?]?=)\s*(.*)$", self.make_database(), re.MULTILINE
        )
        observed_cflags = cflags.group(1).split() if cflags else []
        missing_warnings = [
            warning
            for warning in self.policy["include_policy"]["required_c_warnings"]
            if warning not in observed_cflags
        ]
        if missing_warnings:
            errors.append(f"Makefile lacks strict C warnings: {missing_warnings}")
        if "$(OBJ_DIR)/%.o: %.c" not in makefile or "@mkdir -p $(@D)" not in makefile:
            errors.append("Makefile does not preserve source-relative object paths")
        if re.search(r"(?:notdir|basename)\s+\$\([^)]*SRCS", makefile):
            errors.append("Makefile flattens source object paths")

        declared: list[str] = []
        for variable in self.policy["archive"]["source_variables"]:
            values = [value for value in self.make_variable(variable) if value.startswith("src/")]
            declared.extend(values)
        duplicates = sorted(name for name, count in Counter(declared).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate Makefile source entries: {duplicates}")
        actual = set(self.translation_units)
        declared_set = set(declared)
        if actual != declared_set:
            errors.append(
                "Makefile/source parity: "
                f"missing={sorted(actual - declared_set)} stale={sorted(declared_set - actual)}"
            )
        objects = ["build/obj/" + source.rsplit(".", 1)[0] + ".o" for source in actual]
        if len(objects) != len(set(objects)):
            errors.append("source-relative object collision")
        return errors

    def header_compile_violations(self) -> list[str]:
        errors: list[str] = []
        compilers = (
            ("C", shlex.split(os.environ.get("CC", "cc")), "c11", "c"),
            ("C++", shlex.split(os.environ.get("CXX", "c++")), "c++17", "c++"),
        )
        common_flags = [
            "-D_FILE_OFFSET_BITS=64",
            "-D_POSIX_C_SOURCE=200809L",
            "-Iinclude",
            "-I.",
            "-fsyntax-only",
            "-",
        ]
        for name in sorted(self.headers):
            tier = self.header_tier(name)
            include = name.removeprefix("include/")
            source = (
                f"#include <{include}>\n"
                if name.startswith("include/")
                else f'#include "{name}"\n'
            )
            for language, compiler, standard, input_kind in compilers:
                if language == "C++" and tier != "public":
                    continue
                flags = [f"-std={standard}", "-x", input_kind, *common_flags]
                try:
                    result = subprocess.run(
                        compiler + flags,
                        cwd=ROOT,
                        input=source,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                except OSError as failure:
                    return [f"header {language} compiler cannot start: {failure}"]
                if result.returncode:
                    diagnostic = " ".join(result.stderr.strip().splitlines()[:3])
                    errors.append(
                        f"{tier} header is not {language} self-contained: {name}: {diagnostic}"
                    )
        return errors

    def archive_violations(self) -> list[str]:
        errors: list[str] = []
        archive = ROOT / self.policy["archive"]["path"]
        if not archive.is_file():
            return errors
        result = subprocess.run(
            ["ar", "t", str(archive)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode:
            return [f"archive cannot be listed: {result.stderr.strip()}"]
        members = result.stdout.splitlines()
        duplicates = sorted(name for name, count in Counter(members).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate archive member identities: {duplicates}")
        prefix = self.policy["archive"]["member_prefix"]
        short = [name for name in members if not name.startswith(prefix)]
        if short:
            errors.append(f"archive members lost source-relative identity: {short}")
        expected = self.make_variable(self.policy["archive"]["object_variable"])
        if expected and Counter(expected) != Counter(members):
            errors.append(
                "archive/object parity: "
                f"missing={sorted((Counter(expected) - Counter(members)).elements())} "
                f"extra={sorted((Counter(members) - Counter(expected)).elements())}"
            )
        return errors

    def stale_path_violations(self) -> list[str]:
        errors: list[str] = []
        documents = [
            ROOT / name
            for name in ("AGENTS.md", "ROADMAP.md", "CONTRIBUTING.md", "README.md")
        ]
        documents.extend((ROOT / "docs").rglob("*.md"))
        expression = re.compile(r"`((?:src|include|tests|config)/[^` ]+)`")
        for document in documents:
            if not document.is_file():
                continue
            for number, line in enumerate(document.read_text(errors="ignore").splitlines(), 1):
                for reference in expression.findall(line):
                    reference = reference.rstrip(".,:;")
                    if any(token in reference for token in ("*", "[", "<", "{")):
                        continue
                    if not (ROOT / reference).exists():
                        errors.append(
                            f"stale owned path: {relative(document)}:{number}: {reference}"
                        )
        return errors

    def architecture_violations(self) -> list[str]:
        errors: list[str] = []
        graph: dict[str, set[str]] = {name: set() for name in self.units}
        subsystem_graph: dict[str, set[str]] = defaultdict(set)
        for source, edges in self.include_targets.items():
            for include, target in edges:
                if source.startswith("src/") and include.name.startswith("tests/"):
                    errors.append(f"production includes tests: {source}:{include.line}")
                if not target or target not in graph:
                    continue
                graph[source].add(target)
                source_row = self.manifest.get(source)
                target_row = self.manifest.get(target)
                if source_row and target_row and source_row[1] != target_row[1]:
                    subsystem_graph[source_row[1]].add(target_row[1])
                for rule in self.policy["dependencies"]["forbidden_edges"]:
                    if path_has_prefix(source, rule["from"]) and path_has_prefix(target, rule["to"]):
                        errors.append(f"forbidden dependency: {source} -> {target}")
                if source.startswith("src/") and target.startswith("tests/"):
                    errors.append(f"production includes tests: {source} -> {target}")
                if (
                    source.startswith("src/graph/")
                    and "/families/" not in source
                    and "/families/" in target
                ):
                    errors.append(f"generic graph depends on family implementation: {source} -> {target}")

        family_tokens = {
            Path(row[0]).stem
            for row in self.manifest_rows
            if row[3] == "family"
        }
        family_tokens.update(
            token
            for name in tuple(family_tokens)
            if (token := re.sub(r"_v?[0-9].*$", "", name))
        )
        allowed_family_symbols = set(self.policy["symbols"]["family_entrypoints"])
        for name, unit in self.headers.items():
            row = self.manifest.get(name)
            if row and row[3] == "family":
                continue
            identifiers = set(re.findall(r"\byvex_[A-Za-z0-9_]+\b", unit.masked))
            leaked = sorted(
                identifier
                for identifier in identifiers - allowed_family_symbols
                if any(token in identifier for token in family_tokens)
            )
            if leaked:
                errors.append(f"family implementation contract leaks through {name}: {leaked}")

        for cycle in strongly_connected(graph):
            errors.append(f"include cycle: {cycle}")
        for cycle in strongly_connected(dict(subsystem_graph)):
            errors.append(f"cross-subsystem include cycle: {cycle}")

        for root in self.policy["dependencies"]["planning_roots"]:
            for name, unit in self.translation_units.items():
                if not path_has_prefix(name, root):
                    continue
                for call in self.policy["dependencies"]["payload_io_calls"]:
                    if re.search(rf"\b{re.escape(call)}\s*\(", unit.masked):
                        errors.append(f"planning code performs payload I/O: {name}: {call}")

        errors.extend(self.abi_violations())
        return errors

    def abi_violations(self) -> list[str]:
        errors: list[str] = []
        definitions, foreign = self.nm_symbols()
        if not definitions and not (ROOT / self.policy["archive"]["path"]).exists():
            return errors
        public = self.public_declarations()
        private = self.private_declarations()
        defined = set(definitions)
        limits = self.policy["limits"]
        if len(public) > limits["public_function_declarations"]:
            errors.append(
                f"public declaration count increased: {len(public)} > "
                f"{limits['public_function_declarations']}"
            )
        for symbol in sorted(public):
            count = len(definitions.get(symbol, []))
            if count != 1:
                errors.append(f"public function definition cardinality: {symbol}: {count}")
            for fragment in self.policy["symbols"]["forbidden_public_fragments"]:
                if fragment in symbol:
                    errors.append(f"internal/test/family helper exposed publicly: {symbol}")
        public_identifiers = {
            identifier
            for name, unit in self.headers.items()
            if self.header_tier(name) == "public"
            for identifier in re.findall(r"\byvex_[A-Za-z0-9_]+\b", unit.masked)
        }
        for identifier in sorted(public_identifiers):
            for fragment in self.policy["symbols"]["forbidden_public_fragments"]:
                if fragment in identifier:
                    errors.append(f"internal/test/family identifier exposed publicly: {identifier}")
        duplicates = sorted(symbol for symbol, owners in definitions.items() if len(owners) != 1)
        if duplicates:
            errors.append(f"duplicate global definitions: {duplicates}")
        if foreign:
            errors.append(f"globally visible symbols lack YVEX namespace: {foreign}")
        if len(defined) > limits["library_global_symbols"]:
            errors.append(f"library globals exceed policy: {len(defined)}")

        nonpublic = defined - public
        if len(nonpublic) > limits["nonpublic_global_symbols"]:
            errors.append(
                f"non-public globals exceed policy: {len(nonpublic)} > "
                f"{limits['nonpublic_global_symbols']}"
            )
        undeclared = sorted(nonpublic - private)
        if undeclared:
            errors.append(f"non-public globals lack internal/private declarations: {undeclared}")

        symbol_consumers: dict[str, list[str]] = defaultdict(list)
        consumer_units = dict(self.translation_units)
        consumer_units.update(self.test_translation_units)
        for name, unit in consumer_units.items():
            for symbol in set(re.findall(r"\byvex_[A-Za-z0-9_]+\b", unit.masked)):
                symbol_consumers[symbol].append(name)
        for symbol, owners in self.nm_undefined_consumers().items():
            symbol_consumers[symbol].extend(owners)
        for symbol in sorted(nonpublic & private):
            consumers = sorted(set(symbol_consumers.get(symbol, [])))
            if len(consumers) < 2:
                errors.append(f"non-public global lacks cross-TU consumer: {symbol}: {consumers}")

        for entrypoint in self.policy["symbols"]["family_entrypoints"]:
            if len(definitions.get(entrypoint, [])) != 1:
                errors.append(f"family entrypoint cardinality: {entrypoint}")
        return errors

    def natural_violations(self) -> list[str]:
        errors: list[str] = []
        policy = self.policy["commentary"]
        duplicates: dict[str, list[str]] = defaultdict(list)

        for name, unit in self.units.items():
            for function in unit.functions:
                if function.lines > self.policy["limits"]["function_lines"]:
                    errors.append(
                        f"function exceeds line limit: {name}:{function.start_line}: "
                        f"{function.name}: {function.lines}"
                    )
                if function.is_static and function.name.startswith(self.policy["symbols"]["namespace"]):
                    errors.append(f"private static function retains public prefix: {name}: {function.name}")

        for name, unit in self.headers.items():
            if self.header_tier(name) not in policy["interface_header_tiers"]:
                continue
            comment = self.leading_contract(unit)
            normalized = self.normalize_comment(comment.text) if comment else ""
            if len(normalized) < policy["interface_module_comment_minimum_characters"]:
                errors.append(f"interface header lacks module commentary: {name}")

        old_label = re.compile(
            r"(?:^|\s)(?:" + "|".join(re.escape(label) for label in policy["obsolete_labels"])
            + r")\s*:\s*",
            re.IGNORECASE,
        )
        todo = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
        commented_code = re.compile(
            r"(?:#\s*(?:include|define|if|endif)\b|"
            r"(?:return|break|continue|goto)\b[^;]*;|"
            r"(?:if|for|while|switch)\s*\([^)]*\)\s*\{?|"
            r"[A-Za-z_][A-Za-z0-9_]*\s*\([^;{}]*\)\s*;)",
        )
        for name, line, comment in self.commentary_comments():
            if not comment:
                continue
            location = f"{name}:{line}"
            if comment.startswith("COMMENTARY_SCANNER_PARSE_ERROR"):
                errors.append(f"commentary scan failed: {location}: {comment}")
                continue
            if old_label.search(comment):
                errors.append(f"obsolete commentary label: {location}")
            lowered = comment.lower()
            for fragment in policy["forbidden_boilerplate_fragments"]:
                if fragment.lower() in lowered:
                    errors.append(f"synthetic commentary boilerplate: {location}: {fragment}")
            for stale in policy["stale_topology_names"]:
                if stale.lower() in lowered:
                    errors.append(f"stale topology name in commentary: {location}: {stale}")
            marker = todo.search(comment)
            if marker and not re.search(policy["admitted_todo_pattern"], comment):
                errors.append(f"ownerless maintenance marker: {location}: {marker.group(0)}")
            if commented_code.fullmatch(comment.strip()):
                errors.append(f"commented-out code: {location}")
            normalized = re.sub(r"\s+", " ", lowered).strip()
            if len(normalized) >= policy["duplicate_minimum_characters"]:
                duplicates[normalized].append(location)

        for locations in duplicates.values():
            if len(locations) > 1:
                errors.append(f"duplicated commentary: {locations}")

        errors.extend(self.output_and_shell_violations())
        errors.extend(self.claim_violations())
        return errors

    def output_and_shell_violations(self) -> list[str]:
        errors: list[str] = []
        output = re.compile(r"\b(?:printf|fprintf|vprintf|vfprintf|puts|fputs|putchar|fputc|perror)\s*\(")
        for name, unit in self.translation_units.items():
            if output.search(unit.masked):
                row = self.manifest.get(name)
                boundary = row[5] if row else ""
                admitted = (
                    name.startswith("src/cli/io/")
                    or (row is not None and row[3] == "entrypoint")
                    or boundary in {"transactional-io", "file-serialization"}
                )
                if not admitted:
                    errors.append(f"unowned direct output: {name}")
            if any(path_has_prefix(name, root) for root in self.policy["dependencies"]["domain_roots"]):
                for call in self.policy["dependencies"]["shell_calls"]:
                    if re.search(rf"\b{re.escape(call)}\s*\(", unit.masked):
                        errors.append(f"domain owner shells out: {name}: {call}")
        return errors

    def claim_violations(self) -> list[str]:
        errors: list[str] = []
        for item in self.policy["required_literals"]:
            found = False
            for root_name in item["roots"]:
                root = ROOT / root_name
                if not root.exists():
                    continue
                paths = [root] if root.is_file() else root.rglob("*")
                for path in paths:
                    if path.is_file() and item["text"] in path.read_text(errors="ignore"):
                        found = True
                        break
                if found:
                    break
            if not found:
                errors.append(f"required semantic literal is absent: {item['text']}")

        expression = re.compile(
            r"\b(" + "|".join(
                re.escape(name) for name in self.policy["forbidden_capability_assignments"]
            ) + r")\s*=\s*1\b"
        )
        for root_name in ("src", "tests/live"):
            root = ROOT / root_name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {".c", ".h", ".cu"}:
                    if expression.search(path.read_text(errors="ignore")):
                        errors.append(f"unsupported capability promoted: {relative(path)}")
        return errors

    def violations(self) -> dict[str, list[str]]:
        return {
            "ownership": self.ownership_violations(),
            "layout": self.layout_violations(),
            "architecture": self.architecture_violations(),
            "natural": self.natural_violations(),
        }

    def group_violations(self, group: str) -> list[str]:
        checks = {
            "ownership": self.ownership_violations,
            "layout": self.layout_violations,
            "architecture": self.architecture_violations,
            "natural": self.natural_violations,
        }
        return checks[group]()


def self_test() -> None:
    sample = """/*
 * The fixture exercises comment-aware parsing without imposing prose templates.
 */
#include <stddef.h>
/*
 * Addition is kept out of a macro so the scanner sees one ordinary definition.
 */
int fixture_add(int left, int right)
{
    return left + right;
}

static int zero(void) { return 0; }
"""
    masked, stripped, comments = lex_c(sample)
    functions = find_functions(sample, masked)
    assert [function.name for function in functions] == ["fixture_add", "zero"]
    assert functions[0].lines == 4
    assert functions[1].is_static
    unit = CUnit(Path("fixture.c"), sample, masked, stripped, comments, parse_includes(sample), functions)
    assert adjacent_comment(unit, functions[0].start) is not None
    assert "macro" in adjacent_comment(unit, functions[0].start).text
    assert parse_includes(sample)[0].name == "stddef.h"
    assert strongly_connected({"a": {"b"}, "b": {"a"}}) == [["a", "b"]]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.json"
        path.write_text(json.dumps({"functions": len(functions)}))
        assert json.loads(path.read_text())["functions"] == 2
    print("c structure self-test: ok")


def emit_snapshot(audit: Audit, pretty: bool, maximum: int) -> None:
    violations = audit.violations()
    document = {
        "schema_version": 1,
        "policy_sha256": hashlib.sha256(
            json.dumps(audit.policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "metrics": audit.metrics(),
        "violations": {
            group: {"count": len(items), "sample": items[:maximum]}
            for group, items in violations.items()
        },
    }
    print(json.dumps(document, indent=2 if pretty else None, sort_keys=True))


def run_check(audit: Audit, groups: Sequence[str], maximum: int) -> int:
    failed = False
    for group in groups:
        violations = audit.group_violations(group)
        if violations:
            failed = True
            print(f"c structure {group}: {len(violations)} violation(s)", file=sys.stderr)
            for violation in violations[:maximum]:
                print(f"  {violation}", file=sys.stderr)
            if len(violations) > maximum:
                print(f"  ... {len(violations) - maximum} more", file=sys.stderr)
        else:
            print(f"c structure {group}: ok")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="enforce one or more policy groups")
    check.add_argument("groups", nargs="+", choices=GROUPS)
    check.add_argument("--max-errors", type=int, default=50)

    snapshot = subparsers.add_parser("snapshot", help="emit the complete structural inventory")
    snapshot.add_argument("--pretty", action="store_true")
    snapshot.add_argument("--max-errors", type=int, default=100)

    subparsers.add_parser("self-test", help="exercise the dependency-free lexer and parser")
    arguments = parser.parse_args(argv)
    if arguments.command == "self-test":
        self_test()
        return 0

    policy_path = arguments.policy
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    audit = Audit(policy_path)
    if arguments.command == "snapshot":
        emit_snapshot(audit, arguments.pretty, arguments.max_errors)
        return 0
    return run_check(audit, arguments.groups, arguments.max_errors)


if __name__ == "__main__":
    raise SystemExit(main())
