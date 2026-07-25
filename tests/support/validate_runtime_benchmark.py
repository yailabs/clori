#!/usr/bin/env python3
"""Validate retained target-scale runtime benchmark evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
import sys


HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
MODES = ("eager", "piecewise", "full")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def csv_map(path: pathlib.Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    require(rows and rows[0] == ["field", "value"], f"{path}: invalid CSV header")
    require(all(len(row) == 2 for row in rows[1:]), f"{path}: invalid CSV row")
    result = dict(rows[1:])
    require(len(result) == len(rows) - 1, f"{path}: duplicate CSV field")
    return result


def baseline_fields(path: pathlib.Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as stream:
        lines = [line.rstrip("\n") for line in stream]
    require(lines and lines[0] == "YVEX_RUNTIME_BENCHMARK_BASELINE\t5",
            f"{path}: not schema five")
    fields = dict(line.split("\t", 1) for line in lines[1:] if "\t" in line)
    require(len(fields) == len(lines) - 1, f"{path}: duplicate or malformed field")
    identity = fields.get("identity", "")
    require(HEX64.fullmatch(identity) is not None, f"{path}: invalid identity")
    require(fields.get("device_timing_available") == "1",
            f"{path}: CUDA device timing unavailable")
    device = [int(fields[name], 16) for name in (
        "device_minimum_ns", "device_p50_ns", "device_p90_ns", "device_p95_ns",
        "device_p99_ns", "device_maximum_ns",
    )]
    require(0 < device[0] <= device[1] <= device[2] <= device[3] <= device[4] <= device[5],
            f"{path}: device distribution is invalid")
    mean = int(fields["device_mean_ns"], 16)
    require(device[0] <= mean <= device[5], f"{path}: device mean is invalid")
    require(fields.get("benchmark_scope") == "attention_component",
            f"{path}: benchmark scope drift")
    return fields


def exact_svg(path: pathlib.Path, identity: str, current: str, baseline: str,
              commit: str, source_state: str, mode: str, device: str) -> None:
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == identity, f"{path}: digest mismatch")
    text = data.decode("utf-8")
    for fact in (
        'data-chart-schema="5"', 'data-chart-design="2"',
        "Attention execution profile", "Steady-state latency",
        "Cold-start composition", "Memory &amp; movement", "Steady-state contract",
        "CUDA event timing", f"current={current}", f"baseline={baseline}",
        f"BUILD {commit[:10]}", f"SOURCE {source_state}", device, "decode",
        mode.upper(),
        "release-attention-set",
    ):
        require(fact in text, f"{path}: missing SVG provenance {fact!r}")


def common(result: dict[str, object] | dict[str, str], mode: str, binding: pathlib.Path) -> None:
    expected = {
        "status": "complete", "backend": "cuda", "scope": "full",
        "operation_scope": "release-attention-set", "phase": "decode",
        "requested_mode": mode, "selected_mode": mode,
        "benchmark_scope": "attention_component",
        "component_benchmark_status": "measured",
        "correctness_status": "pass", "structural_runtime_status": "pass",
        "performance_status": "measured",
    }
    for key, value in expected.items():
        require(str(result.get(key)).lower() == value, f"{mode}: {key} mismatch")
    counts = {
        "layers_executed": 43, "bindings_executed": 634,
        "swa_layers_executed": 2, "csa_layers_executed": 21,
        "hca_layers_executed": 20, "warmup_count": 3,
        "benchmark_sample_count": 20, "execution_dispatch_count": 23,
        "artifact_hash_passes": 1,
        "warm_artifact_hash_passes": 0, "runtime_source_headers_read": 0,
        "runtime_source_payload_bytes_read": 0, "runtime_transform_plans_built": 0,
        "runtime_quant_plans_built": 0, "runtime_writer_plans_built": 0,
        "runtime_model_builds": 1, "runtime_descriptor_builds": 1,
        "semantic_graph_builds": 1, "executable_graph_builds": 1,
        "warm_weight_artifact_reads": 0, "warm_weight_upload_bytes": 0,
        "warm_host_allocations": 0, "warm_device_allocations": 0,
        "warm_device_frees": 0,
    }
    for key, value in counts.items():
        require(int(result.get(key, -1)) == value, f"{mode}: {key} mismatch")
    require(str(result.get("cuda_device")) == "NVIDIA GB10",
            f"{mode}: unexpected CUDA device")
    require(int(result.get("compute_capability_major", -1)) == 12 and
            int(result.get("compute_capability_minor", -1)) == 1,
            f"{mode}: unexpected compute capability")
    require(int(result.get("resident_binding_count", 0)) == 806,
            f"{mode}: incomplete attention-envelope residency")
    require(int(result.get("resident_encoded_bytes", 0)) == 5_766_703_652,
            f"{mode}: resident encoded-byte accounting drift")
    require(int(result.get("workspace_bytes", 0)) > 0,
            f"{mode}: missing session workspace evidence")
    for key in (
        "artifact_identity", "runtime_binding_identity", "runtime_descriptor_identity",
        "execution_descriptor_identity", "benchmark_identity", "benchmark_chart_identity",
    ):
        require(HEX64.fullmatch(str(result.get(key, ""))) is not None,
                f"{mode}: invalid {key}")
    require(pathlib.Path(str(result.get("runtime_binding_path", ""))).resolve() == binding,
            f"{mode}: runtime binding path drift")
    require(str(result.get("runtime_generation_ready")).lower() == "false",
            f"{mode}: generation claim promoted")
    require(str(result.get("benchmark_correctness_precondition_passed")).lower() == "true",
            f"{mode}: correctness precondition missing")
    require(str(result.get("benchmark_runtime_precondition_passed")).lower() == "true",
            f"{mode}: runtime precondition missing")
    require(str(result.get("benchmark_device_timing_available")).lower() == "true",
            f"{mode}: CUDA event timing unavailable")
    device = [float(result.get(name, -1.0)) for name in (
        "benchmark_device_minimum_seconds", "benchmark_device_p50_seconds",
        "benchmark_device_p90_seconds", "benchmark_device_p95_seconds",
        "benchmark_device_p99_seconds", "benchmark_device_maximum_seconds",
    )]
    require(0.0 < device[0] <= device[1] <= device[2] <= device[3] <=
            device[4] <= device[5],
            f"{mode}: invalid CUDA event timing distribution")
    require(device[0] <= float(result.get("benchmark_device_mean_seconds", -1.0)) <= device[5],
            f"{mode}: invalid CUDA event timing mean")
    graph_count = int(result.get("cuda_graph_count", -1))
    piece_count = int(result.get("cuda_graph_piece_count", -1))
    captures = int(result.get("cuda_graph_capture_count", -1))
    replays = int(result.get("cuda_graph_replay_count", -1))
    launches = int(result.get("cuda_graph_launch_count", -1))
    kernel_nodes = int(result.get("cuda_graph_kernel_node_count", -1))
    if mode == "eager":
        require((graph_count, captures, replays, launches, kernel_nodes) == (0, 0, 0, 0, 0),
                f"{mode}: eager path reported CUDA Graph work")
    elif mode == "full":
        require(graph_count == 43 and piece_count == 1 and captures == graph_count,
                f"{mode}: full graph inventory mismatch")
        require(replays == graph_count * 23 and launches == replays and kernel_nodes > 0,
                f"{mode}: full graph replay accounting mismatch")
    else:
        require(graph_count > 43 and piece_count == graph_count and captures == graph_count,
                f"{mode}: piecewise graph decomposition mismatch")
        require(replays == graph_count * 23 and launches == replays and kernel_nodes > 0,
                f"{mode}: piecewise graph replay accounting mismatch")


def validate_mode(root: pathlib.Path, binding: pathlib.Path, mode: str) -> tuple[str, str]:
    primary_path = root / f"{mode}.json"
    comparison_path = root / f"{mode}-comparison.csv"
    baseline_path = root / f"{mode}.yvex-benchmark"
    primary_svg = root / f"{mode}.svg"
    comparison_svg = root / f"{mode}-comparison.svg"
    with primary_path.open(encoding="utf-8") as stream:
        primary = json.load(stream)
    comparison = csv_map(comparison_path)
    common(primary, mode, binding)
    common(comparison, mode, binding)
    fields = baseline_fields(baseline_path)
    identity = fields["identity"]
    workspace = int(str(primary["workspace_bytes"]))
    state = int(str(primary["state_allocated_bytes"]))
    expected_peak = (int(str(primary["host_resident_bytes"])) + workspace +
                     int(str(primary["pinned_host_peak_bytes"])) + state)
    require(int(fields["host_workspace_bytes"], 16) == workspace,
            f"{mode}: session workspace mislabeled")
    require(int(fields["state_bytes"], 16) == state, f"{mode}: state bytes drift")
    require(int(fields["peak_host_bytes"], 16) == expected_peak,
            f"{mode}: peak host accounting incomplete")
    require(primary.get("benchmark_identity") == identity, f"{mode}: baseline content drift")
    require(primary.get("benchmark_baseline_written") is True, f"{mode}: baseline not published")
    require(comparison.get("benchmark_baseline_compatible") == "true",
            f"{mode}: comparison incompatible")
    require(comparison.get("benchmark_baseline_identity") == identity,
            f"{mode}: comparison baseline drift")
    require(primary.get("artifact_identity") == comparison.get("artifact_identity"),
            f"{mode}: artifact identity drift")
    require(primary.get("runtime_binding_identity") == comparison.get("runtime_binding_identity"),
            f"{mode}: binding identity drift")
    for result, path, expected_baseline in (
        (primary, primary_svg, "none"),
        (comparison, comparison_svg, identity),
    ):
        require(str(result.get("benchmark_chart_generated")).lower() == "true",
                f"{path}: chart not generated")
        require(pathlib.Path(str(result.get("benchmark_chart_path", ""))).resolve() == path,
                f"{path}: chart path drift")
        require(int(result.get("benchmark_chart_file_bytes", -1)) == path.stat().st_size,
                f"{path}: chart size drift")
        exact_svg(
            path, str(result["benchmark_chart_identity"]), str(result["benchmark_identity"]),
            expected_baseline, str(result["benchmark_current_commit"]),
            str(result["benchmark_current_source_state"]), mode, str(result["cuda_device"]),
        )
        require(COMMIT.fullmatch(str(result["benchmark_current_commit"])) is not None,
                f"{path}: invalid commit provenance")
        require(str(result["benchmark_current_source_state"]) in ("clean", "dirty"),
                f"{path}: invalid source provenance")
    return str(primary["artifact_identity"]), str(primary["runtime_binding_identity"])


def main() -> int:
    require(len(sys.argv) == 3, "usage: validate_runtime_benchmark.py EVIDENCE_DIR BINDING")
    root = pathlib.Path(sys.argv[1]).resolve(strict=True)
    binding = pathlib.Path(sys.argv[2]).resolve(strict=True)
    require(root.is_dir() and binding.is_file(), "evidence directory and binding are required")
    identities = [validate_mode(root, binding, mode) for mode in MODES]
    require(len({value[0] for value in identities}) == 1, "artifact identity differs by mode")
    require(len({value[1] for value in identities}) == 1, "binding identity differs by mode")
    print(f"runtime benchmark evidence: schema=5 modes={len(MODES)} charts=6 validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"runtime benchmark evidence invalid: {exc}", file=sys.stderr)
        raise SystemExit(1)
