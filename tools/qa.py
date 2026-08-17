#!/usr/bin/env python3
"""Resolve, execute, and report canonical YVEX QA obligations."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import fcntl
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator, NoReturn

import generate_qa_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config/qa/registry.json"
OBLIGATIONS_PATH = ROOT / "config/qa/obligations.json"
EVIDENCE_SCHEMA = "yvex.qa.evidence.v1"
PLAN_SCHEMA = "yvex.qa.plan.v1"
BUILD_LOCK = threading.Lock()


class QaError(RuntimeError):
    """The QA request or environment is invalid."""


def fail(message: str) -> NoReturn:
    raise QaError(message)


def run_capture(argv: list[str], *, check: bool = True) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        if check:
            fail(f"cannot execute {argv[0]}: {exc}")
        return ""
    if check and result.returncode != 0:
        fail(result.stderr.strip() or result.stdout.strip() or f"command failed: {argv}")
    return result.stdout.strip()


def load_registry() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        return generate_qa_registry.load_and_validate(ROOT, REGISTRY_PATH)
    except generate_qa_registry.RegistryError as exc:
        fail(f"registry invalid: {exc}")


def load_obligations(registry: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(OBLIGATIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read obligation policy: {exc}")
    allowed = {"schema", "schema_version", "fallback", "change_classes"}
    if not isinstance(value, dict) or set(value) != allowed:
        fail("obligation policy has invalid top-level fields")
    if value["schema"] != "yvex.qa.obligations.v1" or value["schema_version"] != 1:
        fail("unsupported obligation policy schema")
    lanes = set(registry["lanes"])
    seen: set[str] = set()
    for index, item in enumerate(value["change_classes"]):
        where = f"change_classes[{index}]"
        required = {"id", "patterns", "lanes", "tests", "reason"}
        if not isinstance(item, dict) or set(item) != required:
            fail(f"{where} fields must be {sorted(required)}")
        if item["id"] in seen:
            fail(f"duplicate change class {item['id']}")
        seen.add(item["id"])
        if not item["patterns"] or not all(isinstance(pattern, str) for pattern in item["patterns"]):
            fail(f"{where}.patterns must be a non-empty string array")
        if not set(item["lanes"]) <= lanes:
            fail(f"{where} references unknown lane")
        if not isinstance(item["reason"], str) or not item["reason"]:
            fail(f"{where}.reason must be non-empty")
    if not set(value["fallback"]["lanes"]) <= lanes:
        fail("fallback references unknown lane")
    known_tests = {item["id"] for item in registry["tests"]}
    referenced_tests = set(value["fallback"]["tests"])
    for item in value["change_classes"]:
        referenced_tests.update(item["tests"])
    if referenced_tests - known_tests:
        fail(f"obligation policy references unknown tests: {sorted(referenced_tests - known_tests)}")
    return value


def source_owners() -> dict[str, str]:
    owners: dict[str, str] = {}
    with (ROOT / "config/source_owners.tsv").open(newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            owners[row[0]] = row[2]
    return owners


def changed_paths(base: str) -> list[str]:
    run_capture(["git", "rev-parse", "--verify", f"{base}^{{commit}}"])
    paths = set(run_capture(["git", "diff", "--name-only", f"{base}...HEAD"]).splitlines())
    paths.update(run_capture(["git", "diff", "--name-only"]).splitlines())
    paths.update(run_capture(["git", "diff", "--name-only", "--cached"]).splitlines())
    paths.update(
        run_capture(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    )
    return sorted(path for path in paths if path)


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def selected_by_lanes(tests: list[dict[str, Any]], lanes: set[str]) -> set[str]:
    return {item["id"] for item in tests if lanes.intersection(item["lanes"])}


def build_plan(
    registry: dict[str, Any], tests: list[dict[str, Any]], obligations: dict[str, Any], base: str
) -> dict[str, Any]:
    paths = changed_paths(base)
    owners = source_owners()
    reasons: dict[str, list[str]] = {}
    classes: dict[str, list[str]] = {}
    lanes: set[str] = set()
    explicit_tests: set[str] = set()
    for path in paths:
        matched = [item for item in obligations["change_classes"] if matches(path, item["patterns"])]
        if not matched:
            fallback = obligations["fallback"]
            lanes.update(fallback["lanes"])
            explicit_tests.update(fallback["tests"])
            reasons.setdefault(path, []).append(f"fallback: {fallback['reason']}")
            classes[path] = ["unknown"]
            continue
        classes[path] = []
        for item in matched:
            classes[path].append(item["id"])
            lanes.update(item["lanes"])
            explicit_tests.update(item["tests"])
            owner = owners.get(path)
            owner_fact = f" owner={owner}" if owner else ""
            reasons.setdefault(path, []).append(f"{item['id']}{owner_fact}: {item['reason']}")
    known_ids = {item["id"] for item in tests}
    unknown = explicit_tests - known_ids
    if unknown:
        fail(f"obligation policy references unknown tests: {sorted(unknown)}")
    selected = selected_by_lanes(tests, lanes) | explicit_tests
    plan = {
        "schema": PLAN_SCHEMA,
        "base": run_capture(["git", "rev-parse", f"{base}^{{commit}}"]),
        "head": run_capture(["git", "rev-parse", "HEAD"]),
        "paths": paths,
        "classes": classes,
        "required_lanes": sorted(lanes),
        "required_tests": sorted(selected),
        "reasons": reasons,
    }
    material = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["plan_identity"] = hashlib.sha256(material).hexdigest()
    return plan


def tool_path(name: str) -> str | None:
    value = shutil.which(name)
    if value:
        return value
    if name == "nvcc" and Path("/usr/local/cuda/bin/nvcc").is_file():
        return "/usr/local/cuda/bin/nvcc"
    return None


def hardware_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {"cuda_present": Path("/dev/nvidiactl").exists(), "sm121_present": False}
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        facts["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        facts["cuda_present"] = facts["cuda_present"] or (
            result.returncode == 0 and bool(result.stdout.strip())
        )
        facts["sm121_present"] = result.returncode == 0 and "12.1" in result.stdout
    return facts


def doctor(registry: dict[str, Any]) -> dict[str, Any]:
    assets = sorted(
        {asset for item in registry["tests"] for asset in item.get("requirements", {}).get("assets", [])}
    )
    tools = sorted(
        {tool for item in registry["tests"] for tool in item.get("requirements", {}).get("tools", [])}
    )
    return {
        "compiler": run_capture([os.environ.get("CC", "cc"), "--version"], check=False).splitlines()[:1],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tools": {name: tool_path(name) for name in tools},
        "hardware": hardware_facts(),
        "assets": {
            name: {
                "configured": bool(os.environ.get(name)),
                "value_kind": "configured" if os.environ.get(name) else "absent",
            }
            for name in assets
        },
        "build_root_writable": os.access(ROOT / "build", os.W_OK) if (ROOT / "build").exists()
        else os.access(ROOT, os.W_OK),
        "temp_root": str(Path(tempfile.gettempdir()).resolve()),
    }


def requirement_failure(item: dict[str, Any], facts: dict[str, Any]) -> str | None:
    requirement = item["requirements"]
    missing_tools = [name for name in requirement["tools"] if not facts["tools"].get(name)]
    if missing_tools:
        return f"missing tools: {', '.join(missing_tools)}"
    hardware = requirement["hardware"]
    if hardware == "cuda" and not facts["hardware"]["cuda_present"]:
        return "CUDA device unavailable"
    if hardware == "sm121" and not facts["hardware"]["sm121_present"]:
        return "SM121 device unavailable"
    missing_assets = [name for name in requirement["assets"] if not os.environ.get(name)]
    if missing_assets:
        return f"missing configured assets: {', '.join(missing_assets)}"
    for name in requirement["assets"]:
        value = os.environ[name]
        if "://" not in value and not Path(value).exists():
            return f"configured asset does not exist: {name}"
    return None


def source_delta_identity() -> tuple[str, str]:
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest.update(diff)
    untracked = run_capture(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    for path in sorted(path for path in untracked if "__pycache__/" not in path and not path.endswith(".pyc")):
        digest.update(path.encode() + b"\0")
        digest.update((ROOT / path).read_bytes())
    identity = digest.hexdigest()
    empty = hashlib.sha256(b"").hexdigest()
    return ("clean" if identity == empty else "dirty", identity)


def build_identity() -> str:
    material = {
        "cc": os.environ.get("CC", "cc"),
        "python": platform.python_version(),
        "nvcc": run_capture([tool_path("nvcc") or "nvcc", "--version"], check=False).splitlines()[-1:],
        "cflags": os.environ.get("CFLAGS", ""),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def resource_locks(registry: dict[str, Any], resources: list[str]) -> Iterator[None]:
    streams = []
    try:
        for resource in sorted(resources):
            definition = registry["resources"][resource]
            if not definition["exclusive"]:
                continue
            if definition["scope"] == "host":
                runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))
                if not runtime_root.is_absolute() or not runtime_root.is_dir() or runtime_root.is_symlink():
                    fail(f"unsafe QA runtime root: {runtime_root}")
                lock_root = runtime_root / f"yvex-qa-{os.getuid()}" / "locks"
            else:
                lock_root = ROOT / "build/qa/locks"
            lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root_stat = lock_root.stat()
            if lock_root.is_symlink() or root_stat.st_uid != os.getuid():
                fail(f"unsafe QA lock root: {lock_root}")
            os.chmod(lock_root, 0o700)
            lock_path = lock_root / f"{resource}.lock"
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.fchmod(descriptor, 0o600)
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.getuid():
                os.close(descriptor)
                fail(f"unsafe QA resource lock: {lock_path}")
            stream = os.fdopen(descriptor, "a+")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def command_for(item: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    runner = item["runner"]
    environment = dict(os.environ)
    if runner["kind"] == "c-unit":
        if "unit-workspace" not in item["resources"]:
            environment["YVEX_TEST_DISABLE_WORKSPACE_LOCK"] = "1"
        environment["YVEX_TEST_FILTER"] = item["id"]
        return [str(ROOT / "build/tests/test")], environment
    if runner["kind"] == "c-cuda":
        return [str(ROOT / "build/tests/test_cuda")], {
            **environment,
            "YVEX_CUDA_TEST_FILTER": item["id"],
        }
    environment.update(runner.get("env", {}))
    return list(runner["argv"]), environment


def build_for(item: dict[str, Any], log) -> tuple[bool, str]:
    with BUILD_LOCK:
        kind = item["runner"]["kind"]
        targets = list(item["build_targets"])
        if kind == "c-unit":
            targets.append("build/tests/test")
        elif kind == "c-cuda":
            targets.append("cuda")
        if not targets:
            return True, ""
        result = subprocess.run(
            ["make", *targets],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode == 0, f"build exited {result.returncode}"


def execute_one(
    registry: dict[str, Any], item: dict[str, Any], facts: dict[str, Any], run_root: Path, verbose: bool
) -> dict[str, Any]:
    start = time.monotonic()
    missing = requirement_failure(item, facts)
    if missing:
        status = "SKIP" if item["requirement_policy"] == "skip" else "BLOCKED"
        return {
            "test_id": item["id"],
            "status": status,
            "duration_seconds": 0.0,
            "reason": missing,
            "command": [],
            "log": None,
            "repeat_results": [],
            "fixture": item["fixture"],
            "evidence": item["evidence"],
        }
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{item['id']}.log"
    repeats: list[dict[str, Any]] = []
    command: list[str] = []
    status = "PASS"
    reason = ""
    with resource_locks(registry, item["resources"]), log_path.open("w", encoding="utf-8") as log:
        built, build_reason = build_for(item, log)
        if not built:
            status = "ERROR"
            reason = build_reason
        else:
            command, environment = command_for(item)
            for index in range(item["repeat"]):
                repeat_start = time.monotonic()
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=item["timeout_seconds"],
                        check=False,
                    )
                    repeat_status = "PASS" if result.returncode == 0 else "FAIL"
                    repeat_reason = "" if result.returncode == 0 else f"process exited {result.returncode}"
                except subprocess.TimeoutExpired:
                    repeat_status = "ERROR"
                    repeat_reason = f"timeout after {item['timeout_seconds']} seconds"
                repeats.append(
                    {
                        "index": index,
                        "status": repeat_status,
                        "duration_seconds": round(time.monotonic() - repeat_start, 6),
                        "reason": repeat_reason,
                    }
                )
                if repeat_status != "PASS":
                    status = repeat_status
                    reason = repeat_reason
                    break
    if verbose and log_path.is_file():
        print(log_path.read_text(errors="replace"), end="")
    return {
        "test_id": item["id"],
        "status": status,
        "duration_seconds": round(time.monotonic() - start, 6),
        "reason": reason,
        "command": command,
        "log": str(log_path.relative_to(ROOT)),
        "repeat_results": repeats,
        "fixture": item["fixture"],
        "evidence": item["evidence"],
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def execute(
    registry: dict[str, Any], tests: list[dict[str, Any]], selected_ids: list[str],
    *, lane: str | None, plan: dict[str, Any] | None, verbose: bool, jobs: int
) -> int:
    if jobs <= 0:
        fail("--jobs must be a positive integer")
    lookup = {item["id"]: item for item in tests}
    selected = [lookup[test_id] for test_id in selected_ids]
    source_state, delta_identity = source_delta_identity()
    started = dt.datetime.now(dt.timezone.utc)
    run_material = f"{started.isoformat()}\0{run_capture(['git','rev-parse','HEAD'])}\0{','.join(selected_ids)}"
    run_identity = hashlib.sha256(run_material.encode()).hexdigest()
    run_root = ROOT / "build/qa/runs" / run_identity
    run_root.mkdir(parents=True, exist_ok=False)
    facts = doctor(registry)
    print(f"PLAN {len(selected)} tests" + (f" · lane {lane}" if lane else ""), flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for index, item in enumerate(selected, 1):
            print(f"RUN  [{index}/{len(selected)}] {item['id']}", flush=True)
            futures[executor.submit(execute_one, registry, item, facts, run_root, verbose)] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Preserve an evidence result for orchestration defects.
                result = {
                    "test_id": item["id"], "status": "ERROR", "duration_seconds": 0.0,
                    "reason": f"orchestrator exception: {exc}", "command": [], "log": None,
                    "repeat_results": [], "fixture": item["fixture"], "evidence": item["evidence"],
                }
            results.append(result)
            suffix = f" · {result['reason']}" if result["reason"] else ""
            print(f"{result['status']:<7} {item['id']} · {result['duration_seconds']:.3f}s{suffix}",
                  flush=True)
    result_order = {test_id: index for index, test_id in enumerate(selected_ids)}
    results.sort(key=lambda result: result_order[result["test_id"]])
    counts = {state: sum(result["status"] == state for result in results)
              for state in ["PASS", "FAIL", "SKIP", "BLOCKED", "ERROR"]}
    finished = dt.datetime.now(dt.timezone.utc)
    registry_identity = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": 1,
        "run_identity": run_identity,
        "invocation": ["python3", str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        "registry_identity": registry_identity,
        "source_commit": run_capture(["git", "rev-parse", "HEAD"]),
        "source_state": source_state,
        "source_delta_identity": delta_identity,
        "build_identity": build_identity(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "host": facts,
        "selected_lane": lane,
        "resolved_obligations": plan,
        "test_ids": selected_ids,
        "results": results,
        "summary": {"counts": counts, "total": len(results)},
    }
    evidence_path = ROOT / "build/qa/evidence" / f"{run_identity}.json"
    atomic_json(evidence_path, report)
    atomic_json(ROOT / "build/qa/evidence/latest.json", report)
    print(
        "SUMMARY " + " ".join(f"{state}={counts[state]}" for state in counts)
        + f" · evidence {evidence_path.relative_to(ROOT)}", flush=True
    )
    return 1 if counts["FAIL"] or counts["BLOCKED"] or counts["ERROR"] else 0


def resolve_selection(
    registry: dict[str, Any], tests: list[dict[str, Any]], target: str | None,
    legacy_target: str | None
) -> tuple[list[str], str | None]:
    lookup = {item["id"]: item for item in tests}
    if legacy_target:
        selected = [item["id"] for item in tests if legacy_target in item["legacy_targets"]]
        if not selected:
            fail(f"unknown legacy target {legacy_target}")
        return selected, None
    target = target or "fast"
    if target in registry["lanes"]:
        return sorted(selected_by_lanes(tests, {target})), target
    if target in lookup:
        return [target], None
    fail(f"unknown lane or test ID {target}")


def print_plan(plan: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(f"QA PLAN · {plan['base'][:12]} -> {plan['head'][:12]}")
    for path in plan["paths"]:
        print(f"  {path}")
        for reason in plan["reasons"][path]:
            print(f"    -> {reason}")
    print(f"required lanes: {', '.join(plan['required_lanes']) or 'none'}")
    print(f"required tests: {len(plan['required_tests'])}")


def report(path: str) -> int:
    report_path = ROOT / "build/qa/evidence/latest.json" if path == "latest" else Path(path)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        value = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read report: {exc}")
    if value.get("schema") != EVIDENCE_SCHEMA:
        fail("unsupported evidence schema")
    counts = value["summary"]["counts"]
    print(f"run {value['run_identity']}")
    print(f"source {value['source_commit']} ({value['source_state']})")
    print(" ".join(f"{state}={counts[state]}" for state in counts))
    for result in value["results"]:
        if result["status"] != "PASS":
            print(f"{result['status']} {result['test_id']}: {result['reason']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--lane")
    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("test_id")
    subparsers.add_parser("doctor")
    subparsers.add_parser("validate")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--changed", required=True, metavar="BASE")
    plan_parser.add_argument("--json", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("target", nargs="?")
    run_parser.add_argument("--changed", metavar="BASE")
    run_parser.add_argument("--legacy-target")
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--jobs", type=int, default=1)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("path", nargs="?", default="latest")
    arguments = parser.parse_args()
    try:
        registry, tests = load_registry()
        registry["tests"] = tests
        obligations = load_obligations(registry)
        if arguments.command == "validate":
            print(f"qa: registry valid tests={len(tests)} lanes={len(registry['lanes'])}")
            return 0
        if arguments.command == "list":
            selected = tests if not arguments.lane else [item for item in tests if arguments.lane in item["lanes"]]
            for item in selected:
                print(f"{item['id']}\t{','.join(item['lanes'])}\t{item['title']}")
            return 0
        if arguments.command == "explain":
            item = next((item for item in tests if item["id"] == arguments.test_id), None)
            if not item:
                fail(f"unknown test ID {arguments.test_id}")
            print(json.dumps(item, indent=2, sort_keys=True))
            return 0
        if arguments.command == "doctor":
            print(json.dumps(doctor(registry), indent=2, sort_keys=True))
            return 0
        if arguments.command == "plan":
            plan_value = build_plan(registry, tests, obligations, arguments.changed)
            print_plan(plan_value, json_output=arguments.json)
            return 0
        if arguments.command == "report":
            return report(arguments.path)
        if arguments.changed:
            plan_value = build_plan(registry, tests, obligations, arguments.changed)
            return execute(
                registry,
                tests,
                plan_value["required_tests"],
                lane=None,
                plan=plan_value,
                verbose=arguments.verbose,
                jobs=arguments.jobs,
            )
        selected, lane = resolve_selection(registry, tests, arguments.target, arguments.legacy_target)
        return execute(registry, tests, selected, lane=lane, plan=None, verbose=arguments.verbose,
                       jobs=arguments.jobs)
    except QaError as exc:
        print(f"qa: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
