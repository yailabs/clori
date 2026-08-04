#!/usr/bin/env python3
"""Exercise bounded MiniMax-H3 intake with deterministic offline evidence."""

from __future__ import annotations

import copy
import csv
import json
import pathlib
import shutil
import struct
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/minimax_h3_intake.py"
REVISION = "a" * 40
MAX_HEADER = 64 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def safetensors_bytes(header: dict[str, object]) -> bytes:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_end = max(
        (
            descriptor["data_offsets"][1]
            for name, descriptor in header.items()
            if name != "__metadata__"
        ),
        default=0,
    )
    return struct.pack("<Q", len(encoded)) + encoded + bytes([0xA5]) * payload_end


def write_fixture(root: pathlib.Path, mutation: str = "valid") -> None:
    files: dict[str, bytes] = {
        "FL2VA/model_index.json": json_bytes(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "processor": ["transformers", "AutoProcessor"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
            }
        ),
        "FL2VA/processor/tokenizer.json": json_bytes({"model": {"type": "BPE"}}),
        "FL2VA/text_encoder/config.json": json_bytes(
            {"architectures": ["Qwen3VLForConditionalGeneration"], "hidden_size": 8}
        ),
        "LICENSE": b"Synthetic research fixture license.\n",
        "README.md": b"Excluded model card.\n",
        "Ref2VA/config.json": json_bytes({"excluded": True}),
    }
    first_header: dict[str, object] = {
        "encoder.a": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]},
        "encoder.scale": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }
    second_header: dict[str, object] = {
        "encoder.b": {"dtype": "F16", "shape": [3], "data_offsets": [0, 6]},
        "__metadata__": {"format": "pt"},
    }
    weight_map = {
        "encoder.a": "model-00001-of-00002.safetensors",
        "encoder.b": "model-00002-of-00002.safetensors",
        "encoder.scale": "model-00001-of-00002.safetensors",
    }
    total_size = 18
    if mutation == "duplicate":
        second_header = {
            "encoder.a": {"dtype": "F16", "shape": [3], "data_offsets": [0, 6]}
        }
    elif mutation == "missing-shard":
        weight_map["encoder.b"] = "missing.safetensors"
    elif mutation == "stale-index":
        del weight_map["encoder.b"]
        total_size = 12
    elif mutation == "overlap":
        first_header["encoder.scale"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        }
    first_path = "FL2VA/text_encoder/model-00001-of-00002.safetensors"
    second_path = "FL2VA/text_encoder/model-00002-of-00002.safetensors"
    files[first_path] = safetensors_bytes(first_header)
    files[second_path] = safetensors_bytes(second_header)
    files["FL2VA/text_encoder/model.safetensors.index.json"] = json_bytes(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    )
    if mutation == "malformed-length":
        files[first_path] = struct.pack("<Q", len(files[first_path]) + 100) + files[first_path][8:]
    elif mutation == "excessive-header":
        files[first_path] = struct.pack("<Q", MAX_HEADER + 1) + b"{}"

    (root / "files").mkdir(parents=True)
    tree: list[dict[str, object]] = []
    directories: set[str] = set()
    for relative, data in files.items():
        path = root / "files" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        parent = pathlib.PurePosixPath(relative).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
        tree.append({"type": "file", "oid": "blob-" + relative, "size": len(data), "path": relative})
    for directory in directories:
        tree.append({"type": "directory", "oid": "tree-" + directory, "size": 0, "path": directory})
    (root / "repository.json").write_bytes(
        json_bytes({"sha": REVISION, "cardData": {"license": "other"}})
    )
    (root / "tree.json").write_bytes(json_bytes(sorted(tree, key=lambda row: str(row["path"]))))


def invoke(
    fixture: pathlib.Path,
    output: pathlib.Path,
    *,
    check: bool = False,
    repo: str = "MiniMaxAI/MiniMax-H3",
) -> subprocess.CompletedProcess[str]:
    command = [
        "python3",
        str(TOOL),
        "--repo",
        repo,
        "--revision",
        REVISION,
        "--subdir",
        "FL2VA",
        "--output",
        str(output),
        "--fixture-dir",
        str(fixture),
    ]
    if check:
        command.append("--check")
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def file_snapshot(root: pathlib.Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_valid_and_deterministic(temporary: pathlib.Path) -> None:
    fixture = temporary / "fixture"
    write_fixture(fixture)
    first = temporary / "first"
    second = temporary / "second"
    result = invoke(fixture, first)
    require(result.returncode == 0, result.stderr)
    result = invoke(fixture, second)
    require(result.returncode == 0, result.stderr)
    require(file_snapshot(first) == file_snapshot(second), "independent runs are not byte-identical")
    result = invoke(fixture, first, check=True)
    require(result.returncode == 0, result.stderr)

    intake = json.loads((first / "intake-result.json").read_text(encoding="utf-8"))
    require(intake["tensor_payload_bytes_downloaded"] == 0, "fixture payload bytes were reported read")
    require(intake["tensors"] == 3, "valid fixture tensor count is wrong")
    require(intake["shards"] == 2, "valid fixture shard count is wrong")
    with (first / "downloads.tsv").open(encoding="utf-8", newline="") as stream:
        downloads = list(csv.DictReader(stream, delimiter="\t"))
    header_downloads = [row for row in downloads if row["category"].startswith("safetensors-header-")]
    require(len(header_downloads) == 4, "each shard must have exactly two bounded header reads")
    for row in header_downloads:
        require(row["requested_range"].startswith("bytes="), "header read is missing an explicit range")
    output_bytes = b"".join(file_snapshot(first).values())
    require(bytes([0xA5]) * 16 not in output_bytes, "synthetic tensor payload escaped into output")

    stale = first / "intake-result.json"
    stale.write_text("{}\n", encoding="utf-8")
    result = invoke(fixture, first, check=True)
    require(result.returncode == 2 and "stale output" in result.stderr, "--check accepted stale output")


def test_refusal(temporary: pathlib.Path, mutation: str, expected: str) -> None:
    fixture = temporary / mutation
    write_fixture(fixture, mutation)
    result = invoke(fixture, temporary / f"output-{mutation}")
    require(result.returncode == 2, f"{mutation} unexpectedly passed")
    require(expected in result.stderr, f"{mutation} missing refusal {expected!r}: {result.stderr}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="yvex-minimax-intake-") as directory:
        temporary = pathlib.Path(directory)
        test_valid_and_deterministic(temporary)
        cases = {
            "duplicate": "duplicate tensor name",
            "missing-shard": "missing shard",
            "stale-index": "shard absent from index",
            "malformed-length": "declared safetensors header exceeds file size",
            "excessive-header": "safetensors header exceeds",
            "overlap": "overlapping offsets",
        }
        for mutation, expected in cases.items():
            test_refusal(temporary, mutation, expected)
        fixture = temporary / "url-refusal"
        write_fixture(fixture)
        result = invoke(fixture, temporary / "url-output", repo="http://huggingface.co/MiniMaxAI/MiniMax-H3")
        require(result.returncode == 2, "non-HTTPS source URL unexpectedly passed")
        require("must use HTTPS" in result.stderr, "non-HTTPS refusal is missing")


if __name__ == "__main__":
    main()
