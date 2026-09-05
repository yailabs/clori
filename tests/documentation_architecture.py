#!/usr/bin/env python3
"""Validate the small current YVEX documentation surface."""

from __future__ import annotations

import re
import hashlib
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "NOTICE.md",
    "SECURITY.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/architecture/system.md",
    "docs/architecture/compilation.md",
    "docs/architecture/runtime.md",
    "docs/architecture/commands.md",
    "docs/contracts/artifacts.md",
    "docs/contracts/c-api.md",
    "docs/contracts/events-telemetry.md",
    "docs/contracts/local-protocol.md",
    "docs/contracts/runtime.md",
    "docs/development/agentic-engineering.md",
    "docs/development/qa.md",
    "docs/development/source-ownership.md",
    "docs/model-families/integration.md",
    "docs/model-families/deepseek-v4-flash.md",
    "docs/model-families/minimax-h3.md",
    "docs/model-families/mamba2.md",
    "docs/openai-compatibility.md",
    "docs/operator-runbook.md",
    "docs/releases/doctrine.md",
    "docs/releases/v0.1.md",
}
RETIRED_PATHS = {
    "PROJECT.md",
    "MODEL_ARTIFACTS.md",
    ".agents/skills/engineering-worklog",
    "config/documentation_owners.tsv",
    "config/frozen_documents.tsv",
    "docs/audits",
    "docs/doctrine",
    "docs/migrations",
    "docs/milestones",
    "docs/operations",
    "docs/worklog",
    "docs/archive",
    "docs/ledger",
    "docs/agentic",
    "docs/development/documentation-policy.md",
}


def fail(message: str) -> None:
    print(f"documentation architecture: {message}", file=sys.stderr)
    raise SystemExit(1)


def markdown_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        line
        for line in result.stdout.splitlines()
        if line and (ROOT / line).is_file() and not line.startswith("build/")
    }


def github_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith(("~~~", "```")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", raw)
        if not match:
            continue
        heading = re.sub(r"!?\[([^]]*)\]\([^)]+\)", r"\1", match.group(1))
        heading = re.sub(r"<[^>]+>", "", heading)
        slug = heading.replace("`", "").replace("*", "").strip().lower()
        slug = re.sub(r"[^\w\- ]", "", slug, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", slug)
        ordinal = counts.get(slug, 0)
        counts[slug] = ordinal + 1
        anchors.add(slug if ordinal == 0 else f"{slug}-{ordinal}")
    return anchors


def link_targets(text: str) -> list[str]:
    """Current docs use inline/reference Markdown and HTML image/link targets."""
    text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", text)
    pattern = re.compile(r"!?\[[^]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
    references = re.findall(r"(?m)^\s*\[[^]]+\]:\s*(\S+)", text)
    html = re.findall(r"\b(?:src|href)=[\"']([^\"']+)[\"']", text)
    return pattern.findall(text) + references + html


def check_links(paths: set[str]) -> set[Path]:
    anchor_cache: dict[Path, set[str]] = {}
    destinations: set[Path] = set()
    for relative in sorted(paths):
        source = ROOT / relative
        for raw_target in link_targets(source.read_text(encoding="utf-8")):
            target = raw_target.strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            target, _, fragment = target.partition("#")
            destination = source if not target else (source.parent / unquote(target)).resolve()
            try:
                destination.relative_to(ROOT)
            except ValueError:
                fail(f"{relative} links outside repository: {raw_target}")
            if destination.is_dir():
                destination = destination / "README.md"
            if not destination.exists():
                fail(f"{relative} has unresolved link: {raw_target}")
            destinations.add(destination)
            if fragment and destination.suffix == ".md":
                anchors = anchor_cache.setdefault(destination, github_anchors(destination))
                if fragment not in anchors:
                    fail(f"{relative} has unresolved anchor: {raw_target}")
    return destinations


def check_assets(destinations: set[Path]) -> None:
    """Visual projections need real readers, paired sources, and accessible SVGs."""
    assets = set((ROOT / "docs/diagrams").glob("*"))
    assets.update((ROOT / "docs").glob("*.svg"))
    digests: dict[str, Path] = {}
    for path in sorted(assets):
        if path not in destinations:
            fail(f"unconsumed documentation asset: {path.relative_to(ROOT)}")
        if path.suffix == ".mmd" and path.with_suffix(".svg") not in assets:
            fail(f"diagram lacks SVG projection: {path.name}")
        if path.suffix != ".svg":
            continue
        if path.parent.name == "diagrams" and path.with_suffix(".mmd") not in assets:
            fail(f"diagram lacks editable source: {path.name}")
        text = path.read_text(encoding="utf-8")
        for marker in ('<svg ', '<title ', '<desc ', 'role="img"'):
            if marker not in text:
                fail(f"inaccessible SVG {path.name}: absent {marker}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in digests:
            fail(f"duplicate visual assets: {digests[digest].name}, {path.name}")
        digests[digest] = path


def check_current_truth(paths: set[str]) -> None:
    for required in REQUIRED:
        if required not in paths:
            fail(f"missing current documentation owner: {required}")
    for retired in RETIRED_PATHS:
        if (ROOT / retired).exists():
            fail(f"retired governance surface remains: {retired}")

    active = []
    for relative in sorted(paths):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "Active Next:" in text:
            active.append(relative)
        if relative in {"README.md", "CONTRIBUTING.md", "SECURITY.md", "SUPPORT.md"}:
            if "`yvexd`" in text:
                fail(f"public guidance names the retired daemon executable: {relative}")
        if (
            "PROJECT.md" in text
            and relative != "ROADMAP.md"
            and not relative.startswith("docs/decisions/")
        ):
            fail(f"current documentation points at retired PROJECT.md: {relative}")
    if active != ["ROADMAP.md"]:
        fail(f"Active Next must exist only in ROADMAP.md: {active}")

    for relative in ("README.md", "ROADMAP.md", "docs/README.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if re.search(r"^\|\s*`?[AH]\d{2}`?\s*\|", text, re.MULTILINE):
            fail(f"public entry copies private alignment rows: {relative}")
    method = (ROOT / "docs/development/agentic-engineering.md").read_text(encoding="utf-8")
    for section in ("## Authorities", "## Verify the delivery", "## Decide progression",
                    "## Documentation lifecycle"):
        if section not in method:
            fail(f"engineering method lacks authority: {section}")

    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    active_rows = re.findall(r"^\|\s*\d+\s*\|\s*`([A-Z0-9.]+)`\s*\|\s*`active`", roadmap, re.MULTILINE)
    next_rows = re.findall(r"^Active Next:\s*(\S+)\s*$", roadmap, re.MULTILINE)
    if len(active_rows) != 1 or next_rows != active_rows:
        fail(f"ROADMAP active milestone mismatch: active={active_rows} next={next_rows}")

    server = (ROOT / "include/yvex/server.h").read_text(encoding="utf-8")
    match = re.search(r"#define YVEX_LOCAL_PROTOCOL_VERSION (\d+)u", server)
    if not match:
        fail("public local protocol version is absent")
    contract = (ROOT / "docs/contracts/local-protocol.md").read_text(encoding="utf-8")
    if f"YVEX_LOCAL_PROTOCOL_VERSION = {match.group(1)}" not in contract:
        fail("local protocol document does not match the public header")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "## Why YVEX",
        "## Quick start",
        "## Product boundary",
        "## Documentation",
        "## Current limits",
    ):
        if phrase not in readme:
            fail(f"README lacks current entry section: {phrase}")
    if re.search(r"V010\.|POST010\.|/home/|/Users/|\$HOME/", readme):
        fail("README contains project-control or machine-local detail")
    if re.search(
        r"production-ready|blazing fast|state of the art|enterprise-grade|"
        r"seamless|cutting-edge|revolutionary",
        readme,
        re.IGNORECASE,
    ):
        fail("README contains unsupported marketing language")


def main() -> int:
    paths = markdown_paths()
    check_current_truth(paths)
    check_assets(check_links(paths))
    print(f"documentation architecture: ok (markdown={len(paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
