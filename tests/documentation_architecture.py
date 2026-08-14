#!/usr/bin/env python3
"""Validate YVEX documentation ownership, navigation, and claim boundaries."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
OWNERS = ROOT / "config/documentation_owners.tsv"
FROZEN = ROOT / "config/frozen_documents.tsv"
BASELINE = "51a5c087eafe857d71df1566ce90c2f87a2fcfc1"
WORKLOG_FIELDS = [
    "Date",
    "Type",
    "Milestone",
    "Branch",
    "Baseline",
    "Checkpoint",
    "Subsystem",
    "Model family",
    "Hardware",
    "Evidence",
    "Comparability",
    "Publishability",
]
WORKLOG_SECTIONS = [
    "Before",
    "Problem",
    "Causal analysis",
    "Decision",
    "Implementation",
    "After",
    "Evidence",
    "Remaining limitations",
    "Why it matters",
]


def fail(message: str) -> None:
    print(f"documentation architecture: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_tsv(path: Path, header: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing manifest: {path.relative_to(ROOT)}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != header:
            fail(f"{path.relative_to(ROOT)} has wrong header: {reader.fieldnames}")
        rows = list(reader)
    for number, row in enumerate(rows, 2):
        if any(value is None or value == "" for value in row.values()):
            fail(f"{path.relative_to(ROOT)}:{number} has an empty or malformed field")
    return rows


def markdown_paths() -> set[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        line
        for line in result.stdout.splitlines()
        if line and (ROOT / line).is_file() and "build/" not in line
    }


def github_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("```") or raw.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", raw)
        if not match:
            continue
        heading = match.group(1)
        heading = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", heading)
        heading = heading.replace("`", "").replace("*", "").replace("_", "")
        slug = heading.strip().lower()
        slug = re.sub(r"[^\w\- ]", "", slug, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", slug)
        ordinal = counts.get(slug, 0)
        counts[slug] = ordinal + 1
        anchors.add(slug if ordinal == 0 else f"{slug}-{ordinal}")
    return anchors


def check_links(paths: set[str]) -> None:
    link_re = re.compile(r"!?\[[^]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
    anchor_cache: dict[Path, set[str]] = {}
    incoming: set[str] = set()

    for relative in sorted(paths):
        source = ROOT / relative
        text = source.read_text(encoding="utf-8")
        for raw_target in link_re.findall(text):
            target = raw_target.strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            target, _, fragment = target.partition("#")
            target = unquote(target)
            destination = source if target == "" else (source.parent / target).resolve()
            try:
                destination.relative_to(ROOT)
            except ValueError:
                fail(f"{relative} links outside repository: {raw_target}")
            if destination.is_dir():
                destination = destination / "README.md"
            if not destination.exists():
                fail(f"{relative} has unresolved link: {raw_target}")
            if destination.is_file() and destination.suffix == ".md":
                incoming.add(str(destination.relative_to(ROOT)))
            if fragment and destination.suffix == ".md":
                anchors = anchor_cache.setdefault(destination, github_anchors(destination))
                if fragment not in anchors:
                    fail(f"{relative} has unresolved anchor: {raw_target}")

    owner_rows = read_tsv(
        OWNERS,
        ["path", "class", "authority_mode", "audience", "lifecycle", "canonical_subject"],
    )
    exempt = {
        "audit",
        "decision",
        "migration",
        "milestone",
        "test-support",
        "legal",
        "worklog",
    }
    for row in owner_rows:
        path = row["path"]
        if (
            path.startswith("docs/")
            and path != "docs/README.md"
            and row["class"] not in exempt
            and row["lifecycle"] in {"living", "unreleased"}
            and path not in incoming
        ):
            fail(f"orphan living document has no incoming Markdown link: {path}")


def check_owners() -> list[dict[str, str]]:
    header = ["path", "class", "authority_mode", "audience", "lifecycle", "canonical_subject"]
    rows = read_tsv(OWNERS, header)
    paths = [row["path"] for row in rows]
    if paths != sorted(paths):
        fail("documentation ownership manifest is not path-sorted")
    if len(paths) != len(set(paths)):
        fail("documentation ownership manifest contains duplicate paths")
    actual = markdown_paths()
    expected = set(paths)
    if actual != expected:
        fail(
            "Markdown ownership mismatch; missing="
            f"{sorted(actual - expected)} extra={sorted(expected - actual)}"
        )

    classes = {
        "entry",
        "doctrine",
        "reference",
        "architecture",
        "family",
        "contract",
        "operations",
        "development",
        "project-control",
        "milestone",
        "decision",
        "audit",
        "migration",
        "release",
        "legal",
        "contribution",
        "test-support",
        "worklog",
    }
    modes = {"canonical", "projection", "frozen"}
    lifecycles = {"living", "frozen", "accepted", "planned", "retained", "unreleased"}
    subjects: set[str] = set()
    for row in rows:
        if row["class"] not in classes:
            fail(f"unknown document class for {row['path']}: {row['class']}")
        if row["authority_mode"] not in modes:
            fail(f"unknown authority mode for {row['path']}: {row['authority_mode']}")
        if row["lifecycle"] not in lifecycles:
            fail(f"unknown lifecycle for {row['path']}: {row['lifecycle']}")
        if row["canonical_subject"] in subjects:
            fail(f"duplicate canonical subject: {row['canonical_subject']}")
        subjects.add(row["canonical_subject"])
        if not (ROOT / row["path"]).is_file():
            fail(f"owned document is absent: {row['path']}")
        if row["class"] == "audit" and row["authority_mode"] != "frozen":
            fail(f"audit is not frozen: {row['path']}")

    controls = [row["path"] for row in rows if row["class"] == "project-control"]
    if controls != ["ROADMAP.md"]:
        fail(f"ROADMAP.md is not the unique project-control document: {controls}")
    return rows


def check_frozen() -> None:
    rows = read_tsv(FROZEN, ["path", "sha256"])
    paths = [row["path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("frozen-document manifest is unsorted or contains duplicates")
    audit_files = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "docs/audits").rglob("*")
        if path.is_file()
    }
    if set(paths) != audit_files:
        fail(
            "frozen audit mismatch; missing="
            f"{sorted(audit_files - set(paths))} extra={sorted(set(paths) - audit_files)}"
        )
    for row in rows:
        path = ROOT / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            fail(f"frozen document changed: {row['path']}")


def check_engineering_worklogs(rows: list[dict[str, str]]) -> None:
    skill_path = ROOT / ".agents/skills/engineering-worklog/SKILL.md"
    if not skill_path.is_file():
        fail("repository engineering-worklog skill is absent")
    skill = skill_path.read_text(encoding="utf-8")
    if not re.match(
        r"^---\nname: engineering-worklog\ndescription: .+\n---\n", skill
    ):
        fail("engineering-worklog skill lacks canonical discovery metadata")
    for phrase in (
        "checkpoint",
        "repair",
        "performance",
        "closure",
        "build/worklog/",
        "docs/worklog/",
        "private-draft",
        "reviewed",
        "public-safe",
        "Do not generate a worklog for formatting-only edits",
        "Do not generate or publish visuals",
    ):
        if phrase not in skill:
            fail(f"engineering-worklog skill lacks required contract: {phrase}")
    if (ROOT / ".codex/skills/engineering-worklog").exists():
        fail("engineering-worklog uses unsupported repository skill location")

    records = [row for row in rows if row["class"] == "worklog"]
    if not records:
        fail("documentation ownership has no retained engineering worklog")
    for row in records:
        path = row["path"]
        if not re.fullmatch(
            r"docs/worklog/\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*\.md",
            path,
        ):
            fail(f"worklog path is not canonical: {path}")
        text = (ROOT / path).read_text(encoding="utf-8")
        metadata: list[tuple[str, str]] = []
        for match in re.finditer(r"^\| ([^|]+?) \| (.+?) \|$", text, re.MULTILINE):
            field, value = match.groups()
            if field not in {"Field", "---"}:
                metadata.append((field, value.strip("`")))
            if len(metadata) == len(WORKLOG_FIELDS):
                break
        if [field for field, _ in metadata] != WORKLOG_FIELDS:
            fail(f"worklog metadata fields are missing or out of order: {path}")
        values = dict(metadata)
        event_date = Path(path).name[:10]
        if values["Date"] != event_date:
            fail(f"worklog date does not match its path: {path}")
        if values["Type"] not in {"checkpoint", "repair", "performance", "closure"}:
            fail(f"worklog has invalid trigger type: {path}")
        for field in ("Baseline", "Checkpoint"):
            if not re.fullmatch(r"[0-9a-f]{40}", values[field]):
                fail(f"worklog {field.lower()} is not a full commit identity: {path}")
        if values["Comparability"] not in {
            "directly comparable",
            "approximately comparable",
            "characterization only",
            "not-applicable",
        }:
            fail(f"worklog has invalid comparability: {path}")
        if values["Publishability"] not in {"private-draft", "reviewed", "public-safe"}:
            fail(f"worklog has invalid publishability: {path}")
        for section in WORKLOG_SECTIONS:
            if f"## {section}" not in text:
                fail(f"worklog lacks required section {section}: {path}")
        if values["Publishability"] == "public-safe" and re.search(
            r"(?:/home/|/Users/|\$HOME/|BEGIN [A-Z ]*PRIVATE KEY|"
            r"(?:api[_-]?key|token|password)\s*[:=]\s*\S+)",
            text,
            re.IGNORECASE,
        ):
            fail(f"public-safe worklog contains sensitive local material: {path}")


def check_migration() -> None:
    inventory_path = ROOT / "docs/audits/documentation-architecture-51a5c/inventory.tsv"
    header = [
        "path",
        "kind",
        "baseline_class",
        "baseline_authority",
        "conflict",
        "disposition",
        "final_owner",
        "claim_effect",
    ]
    rows = read_tsv(inventory_path, header)
    recorded = [row["path"] for row in rows]
    if recorded != sorted(recorded) or len(recorded) != len(set(recorded)):
        fail("baseline documentation inventory is unsorted or contains duplicates")

    available = subprocess.run(
        ["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    root_surfaces = {
        ".github/pull_request_template.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "MODEL_ARTIFACTS.md",
        "NOTICE.md",
        "README.md",
        "ROADMAP.md",
        "tests/vectors/README.md",
    }
    if available.returncode == 0:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", BASELINE],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        baseline = sorted(
            path
            for path in result.stdout.splitlines()
            if path.startswith("docs/") or path in root_surfaces
        )
        if recorded != baseline:
            fail(
                "baseline documentation inventory mismatch; missing="
                f"{sorted(set(baseline) - set(recorded))} "
                f"extra={sorted(set(recorded) - set(baseline))}"
            )
    elif len(recorded) != 48:
        fail("frozen baseline inventory does not contain 48 surfaces")

    migration = (ROOT / "docs/migrations/documentation-architecture-v1.md").read_text(
        encoding="utf-8"
    )
    for row in rows:
        if f"| `{row['path']}` |" not in migration:
            fail(f"migration record omits baseline path: {row['path']}")
        if row["claim_effect"] != "preserved":
            fail(f"baseline claim effect was not preserved: {row['path']}")
        for owner in row["final_owner"].split(";"):
            if not (ROOT / owner).exists():
                fail(f"migration owner does not exist for {row['path']}: {owner}")


def check_project_control() -> None:
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    if len(roadmap.splitlines()) > 350:
        fail("ROADMAP.md exceeds 350 lines")
    active_next = re.findall(r"^Active Next: (\S+)$", roadmap, flags=re.MULTILINE)
    if active_next != ["V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0"]:
        fail(f"unexpected Active Next: {active_next}")
    active_rows = re.findall(
        r"^\| \d+ \| `([^`]+)` \| `active` \|", roadmap, flags=re.MULTILINE
    )
    if active_rows != active_next:
        fail(f"active milestone/Active Next mismatch: {active_rows}/{active_next}")
    required = {
        "V010.DOCS.INFORMATION.ARCHITECTURE.0": "complete",
        "V010.REPO.CODE.COMMENTARY.0": "complete",
        "V010.OPERATOR.REPL.CONSOLE.0": "complete",
        "V010.REBASE.DEEPSEEK.DSPARK.0": "complete",
        "V010.PRODUCT.ARCHITECTURE.REFOUNDATION.0": "complete",
        "V010.REPO.ARCHITECTURE.COMPRESSION.0": "complete",
        "V010.CORE.COMPILATION.FAMILY.CONSOLIDATION.0": "complete",
        "V010.DEVELOPMENT.ENGINEERING.WORKLOG.0": "complete",
        "V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0": "active",
        "V010.EVAL.DEEPSEEK.0": "blocked",
        "V010.BENCH.DEEPSEEK.0": "not-measured",
        "V010.RELEASE.0": "blocked",
    }
    for milestone, state in required.items():
        pattern = rf"\| `?{re.escape(milestone)}`? \| `{state}` \|"
        if not re.search(pattern, roadmap):
            fail(f"ROADMAP.md lacks {milestone}={state}")
    if (ROOT / "PROJECT.md").exists():
        fail("retired PROJECT.md exists")
    owners = read_tsv(
        OWNERS,
        ["path", "class", "authority_mode", "audience", "lifecycle", "canonical_subject"],
    )
    for row in owners:
        path = row["path"]
        if path == "ROADMAP.md" or row["class"] in {"audit", "migration"}:
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        if re.search(r"^Active Next\s*[:=]", text, re.MULTILINE):
            fail(f"Active Next appears outside ROADMAP.md: {path}")
        if row["class"] == "milestone" and re.search(
            r"^V\d+\.[A-Z0-9_.]+\s*=\s*(?:active|blocked|complete|partial|not-measured)",
            text,
            re.MULTILINE,
        ):
            fail(f"milestone contract owns live state: {path}")


def check_content(rows: list[dict[str, str]]) -> None:
    required_files = [
        "README.md",
        "ROADMAP.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
        "docs/README.md",
        "docs/doctrine/principles.md",
        "docs/doctrine/glossary.md",
        "docs/doctrine/evidence.md",
        "docs/reference/verified-inference.md",
        "docs/architecture/system.md",
        "docs/contracts/runtime.md",
        "docs/model-families/deepseek-v4-flash.md",
        "docs/model-families/qwen.md",
        "docs/model-families/gemma.md",
        "docs/development/documentation-policy.md",
        "docs/migrations/documentation-architecture-v1.md",
        "docs/releases/doctrine.md",
        "docs/releases/v0.1.md",
    ]
    for path in required_files:
        if not (ROOT / path).is_file():
            fail(f"required documentation owner is absent: {path}")

    retired = [
        "MODEL_ARTIFACTS.md",
        "docs/api.md",
        "docs/cli-output-architecture.md",
        "docs/contract.md",
        "docs/model-families.md",
        "docs/reference-architecture.md",
        "docs/runbooks/README.md",
        "docs/runbooks/common.md",
        "docs/runbooks/deepseek.md",
        "docs/system-target.md",
        "docs/topology-closure-audit.md",
        "docs/v010-release-doctrine.md",
    ]
    for path in retired:
        if (ROOT / path).exists():
            fail(f"retired documentation path remains: {path}")

    allowed_flat = {"README.md", "operator-runbook.md", "openai-compatibility.md"}
    flat = {path.name for path in (ROOT / "docs").glob("*.md")}
    if flat != allowed_flat:
        fail(f"unexpected flat documentation surface: {sorted(flat)}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "# YVEX",
        "## Why YVEX",
        "## Quick start",
        "## Product boundary",
        "## Documentation",
        "## Current limits",
        "## License",
    ):
        if heading not in readme:
            fail(f"README.md lacks {heading}")
    if not 80 <= len(readme.splitlines()) <= 220:
        fail(f"README.md length is not compact: {len(readme.splitlines())}")
    for forbidden in ("V010.", "POST010.", "/home/", "/Users/", "$HOME/", "Terminal 1 —"):
        if forbidden in readme:
            fail(f"README.md contains non-entry detail: {forbidden}")
    for phrase in ("production-ready", "release-ready", "enterprise-grade", "state of the art"):
        if phrase.lower() in readme.lower():
            fail(f"README.md contains unsupported marketing claim: {phrase}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if "## Unreleased" not in changelog or "### Changed" not in changelog:
        fail("CHANGELOG.md lacks Unreleased/Changed structure")
    if re.search(r"\b(?:V010|POST010)\.", changelog) or re.search(r"\bwave\b", changelog, re.I):
        fail("CHANGELOG.md contains milestone or wave chronology")

    glossary = (ROOT / "docs/doctrine/glossary.md").read_text(encoding="utf-8")
    terms = [
        "Source snapshot",
        "Model family",
        "Model target",
        "Logical model",
        "Transformation plan",
        "Physical variant",
        "Complete artifact",
        "Supported artifact",
        "Admission",
        "Materialization",
        "Runtime binding",
        "Runtime model",
        "Runtime session",
        "Persistent state",
        "Workspace",
        "Semantic graph",
        "Executable graph",
        "Launch graph",
        "Capability",
        "Evidence",
        "Evaluation",
        "Benchmark",
        "Release qualification",
    ]
    for term in terms:
        if f"| {term} |" not in glossary:
            fail(f"glossary lacks canonical term: {term}")

    by_path = {row["path"]: row for row in rows}
    excluded_classes = {"audit", "migration", "decision", "milestone", "worklog"}
    stale_paths = (
        "docs/reference-architecture.md",
        "docs/contract.md",
        "docs/api.md",
        "docs/model-families.md",
        "docs/v010-release-doctrine.md",
        "MODEL_ARTIFACTS.md",
    )
    old_commands = re.compile(
        r"(?:\./)?yvex (?:evidence|graph|quant|source|tensor|tokenizer)(?:\s|`)|"
        r"(?:\./)?yvex runtime (?:input|context)(?:\s|`)"
    )
    for path, row in by_path.items():
        if row["class"] in excluded_classes or path == "CHANGELOG.md":
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        if old_commands.search(text):
            fail(f"active document retains pre-command-architecture grammar: {path}")
        if any(stale in text for stale in stale_paths):
            fail(f"active document points to a retired documentation path: {path}")
        if path not in {"ROADMAP.md"} and "PROJECT.md" in text:
            fail(f"active document points to retired PROJECT.md: {path}")
        if row["class"] not in {"audit", "milestone"} and re.search(
            r"(?:^|\s)(?:you>|assistant>)", text, re.MULTILINE
        ):
            fail(f"active document presents transitional REPL roles: {path}")

    deepseek = (ROOT / "docs/model-families/deepseek-v4-flash.md").read_text(encoding="utf-8")
    deepseek_words = " ".join(deepseek.split())
    if (
        "DeepSeek-V4-Flash-DSpark is the sole complete YVEX "
        "source-to-streamed-text vertical"
    ) not in deepseek_words:
        fail("DeepSeek record lacks exact current capability stage")
    for phrase in ("model behavior or quality evaluation", "release qualification"):
        if phrase not in deepseek:
            fail(f"DeepSeek record lacks non-claim: {phrase}")
    for family in ("qwen", "gemma"):
        text = (ROOT / f"docs/model-families/{family}.md").read_text(encoding="utf-8")
        if "unsupported runtime family" not in text:
            fail(f"{family} record does not state unsupported runtime stage")

    if "The server-backed console uses one `yvex>` prompt" not in readme:
        fail("README does not describe the current server-backed console")
    if "Ctrl-D exits cleanly" not in readme:
        fail("README omits the current console EOF contract")
    if "functional but transitional" in readme:
        fail("README still describes the implemented console as transitional")
    if "hidden chain of thought" not in (ROOT / "docs/milestones/runtime-console-repl.md").read_text(
        encoding="utf-8"
    ):
        fail("console milestone lacks hidden-reasoning non-claim")


def main() -> int:
    rows = check_owners()
    paths = {row["path"] for row in rows}
    check_frozen()
    check_engineering_worklogs(rows)
    check_migration()
    check_links(paths)
    check_project_control()
    check_content(rows)
    print(
        "documentation architecture: ok "
        f"(markdown={len(paths)} canonical={sum(r['authority_mode'] == 'canonical' for r in rows)} "
        f"frozen={sum(r['authority_mode'] == 'frozen' for r in rows)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
