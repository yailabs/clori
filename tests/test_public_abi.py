#!/usr/bin/env python3
"""Reject silent drift in explicitly versioned installed C records."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

# Each entry ties the current declaration tokens and LP64 layout to one explicit
# schema decision. Updating an entry is therefore a reviewable ABI migration,
# not a mechanical consequence of editing an installed header.
RECORDS = {
    "yvex_provider_request": (
        "include/yvex/provider.h", "YVEX_PROVIDER_SCHEMA_V3", 3, 584,
        "3a388fb2e598e86e1f5e5bcc8870ac46061489808a8e97becde33c5b321e5f9b"),
    "yvex_provider_output": (
        "include/yvex/provider.h", "YVEX_PROVIDER_SCHEMA_V1", 1, 344,
        "82baeac3ca91c06196a7cc2aa053aecd99c86f98e9902155a0c638629703cb59"),
    "yvex_imatrix_data_summary": (
        "include/yvex/quant.h", "YVEX_IMATRIX_DATA_SCHEMA_VERSION", 1, 216,
        "0d034156e6add39bfea3f9b6b7aa8c71cec95b7fb77ff89b6780cfd2cfedfb4c"),
    "yvex_quant_policy_rule": (
        "include/yvex/quant.h", "YVEX_QUANT_POLICY_SCHEMA_VERSION", 2, 128,
        "e1b749492c046976b5d5119fbdbf90732930245c787e6669a94e783f3cd20d74"),
    "yvex_quant_policy_summary": (
        "include/yvex/quant.h", "YVEX_QUANT_POLICY_SCHEMA_VERSION", 2, 144,
        "8008bea6ecfd46feccef371a6425a8681ce3e9659d47a7806a2486bea192f455"),
    "yvex_server_event": (
        "include/yvex/server.h", "YVEX_RUNTIME_EVENT_SCHEMA_VERSION", 4, 904,
        "a9cec914e39a2d7ab4ccbee797b6592b25f1432e46c8f7a27fdff6bc3907aaf3"),
    "yvex_server_metrics": (
        "include/yvex/server.h", "YVEX_RUNTIME_METRICS_SCHEMA_VERSION", 3, 216,
        "0b811120863697b41e7c6c3999746c7c06ce41d2578a317512a18e85c61e1bdd"),
    "yvex_server_options": (
        "include/yvex/server.h", "YVEX_SERVER_OPTIONS_SCHEMA_CURRENT", 4, 88,
        "8073cc5ab362027b9e7696e4dc66012adb523204bfc7c306c66ac398a2ab05ab"),
    "yvex_server_engine_options": (
        "include/yvex/server.h", "YVEX_SERVER_ENGINE_SCHEMA_CURRENT", 2, 136,
        "6741157212a73013a0dfb0c3113fd23994bb5dccbb7146a7d3a0dbf1e80f5f86"),
    "yvex_server_engine_summary": (
        "include/yvex/server.h", "YVEX_SERVER_ENGINE_SCHEMA_CURRENT", 2, 728,
        "92deac5947e27a2426866816f8b60b1c2d83dffc135f52a7c97e0b01939d103f"),
    "yvex_server_summary": (
        "include/yvex/server.h", "YVEX_SERVER_SUMMARY_SCHEMA_V1", 1, 832,
        "187bd214ba857f775c22be5165f40471de9805f5017e515afe8892950e95eae6"),
    "yvex_client_partial_turn": (
        "include/yvex/server.h", "YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1", 1, 392,
        "d3c7abc6ab4c65dfb69f4a3828ee49dd3b4804309fddcb6dc36aa8543ed6c1ee"),
    "yvex_console_status": (
        "include/yvex/server.h", "YVEX_CONSOLE_STATUS_SCHEMA_V1", 1, 496,
        "83e3767d5d2fda381a82a65ddf8d894918355161e6ff23410a27a2680f112d15"),
    "yvex_client_state_checkpoint": (
        "include/yvex/server.h", "YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1", 1, 296,
        "d7fd385982bb24fc091d6702c0def19e1cae0070cd6717ef2211775b9aecb444"),
    "yvex_client_media_result": (
        "include/yvex/server.h", "YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V1", 1, 856,
        "812c35fbd87c5f621d3a8b52ffbe39c32cdcc41580bf562dc07fd8b6ff6123b2"),
    "yvex_client_request": (
        "include/yvex/server.h", "YVEX_LOCAL_PROTOCOL_VERSION", 14, 920,
        "e564c4dfa3c471b5efa0a4a493c63452d9a92d24cd8c273a58d13c5017f9ff41"),
    "yvex_client_message": (
        "include/yvex/server.h", "YVEX_LOCAL_PROTOCOL_VERSION", 14, 10024,
        "7427bba88c63a0443c240058c0117c2e6f2169fce79854305a3b4b372408759c"),
    "yvex_tokenizer_plan_summary": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_PLAN_SCHEMA_V3", 3, 864,
        "cca1ce6ec52182dcee89615b486db711590c2bcdb05a94117deecda5f695beb4"),
    "yvex_tokenizer_encode_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_EXECUTION_SCHEMA_V1", 1, 328,
        "2326e86e866ae2cd28b38d44ea02952132c7fcd1efdbda696faf21328cd59a28"),
    "yvex_tokenizer_decode_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_EXECUTION_SCHEMA_V1", 1, 240,
        "b3b9386de94f29cc1be1c835b966d140b6a3f1aa0ca802a84147c707df9ea511"),
    "yvex_tokenizer_fragment": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_DECODER_SCHEMA_V1", 1, 256,
        "e143bdd815a24a35d457ef8e3d082647bd85160dc5011e12b57827f66c9a79d3"),
    "yvex_token_sequence_summary": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_APPEND_SCHEMA_V1", 1, 168,
        "c87d60c73d1e3e4e09d9918646d1e20bf25e01f0007d149d43bbbff07d32c28b"),
    "yvex_tokenizer_provider_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V2", 2, 128,
        "114c115096e6e628c8a2c07bb890c4645357bec99e6b36ad2d1adbfdf4766090"),
}


def declaration_digest(header: str, record: str) -> str:
    text = (ROOT / header).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)
    matches = [
        body
        for body, name in re.findall(
            r"typedef\s+struct\s*\{(.*?)\}\s*([A-Za-z_]\w*)\s*;",
            text,
            flags=re.DOTALL,
        )
        if name == record
    ]
    if len(matches) != 1:
        raise ValueError(f"{record}: expected one installed declaration, found {len(matches)}")
    tokens = re.findall(r"[A-Za-z_]\w*|\d+[A-Za-z0-9_]*|[^\s]", matches[0])
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def installed_versioned_records() -> set[str]:
    records: set[str] = set()
    for header in (ROOT / "include/yvex").glob("*.h"):
        text = header.read_text(encoding="utf-8")
        text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)
        for match in re.finditer(
            r"typedef\s+struct\s*\{(.*?)\}\s*([A-Za-z_]\w*)\s*;",
            text,
            flags=re.DOTALL,
        ):
            tokens = re.findall(r"[A-Za-z_]\w*|\d+[A-Za-z0-9_]*|[^\s]", match.group(1))
            if tokens[:3] == ["unsigned", "int", "schema_version"]:
                records.add(match.group(2))
    return records


def compiler_source() -> str:
    lines = [
        "#include <limits.h>",
        "#include <stddef.h>",
        "#include <yvex/provider.h>",
        "#include <yvex/quant.h>",
        "#include <yvex/server.h>",
        "#include <yvex/tokenizer.h>",
        "#if defined(__cplusplus)",
        "#define ABI_ASSERT(condition, message) static_assert(condition, message)",
        "#else",
        "#define ABI_ASSERT(condition, message) _Static_assert(condition, message)",
        "#endif",
        'ABI_ASSERT(CHAR_BIT == 8, "public ABI requires 8-bit bytes");',
        'ABI_ASSERT(sizeof(void *) == 8, "public ABI manifest requires 64-bit pointers");',
        'ABI_ASSERT(sizeof(unsigned int) == 4, "public ABI requires 32-bit unsigned int");',
        'ABI_ASSERT(sizeof(unsigned long long) == 8, "public ABI requires 64-bit unsigned long long");',
        'ABI_ASSERT(sizeof(double) == 8, "public ABI requires 64-bit double");',
        'ABI_ASSERT(YVEX_SERVER_OPTIONS_SCHEMA_V3 == 3u, "legacy v3 identity changed");',
        'ABI_ASSERT(YVEX_SERVER_OPTIONS_SCHEMA_V4 == 4u, "server options v4 identity changed");',
        'ABI_ASSERT(YVEX_SERVER_ENGINE_SCHEMA_V1 == 1u, "legacy engine v1 identity changed");',
        'ABI_ASSERT(YVEX_SERVER_ENGINE_SCHEMA_V2 == 2u, "engine v2 identity changed");',
        'ABI_ASSERT(YVEX_RUNTIME_EVENT_SCHEMA_V3 == 3u, "legacy event v3 identity changed");',
        'ABI_ASSERT(YVEX_RUNTIME_EVENT_SCHEMA_V4 == 4u, "event v4 identity changed");',
        'ABI_ASSERT(YVEX_LOCAL_PROTOCOL_VERSION == 14u, "local protocol identity changed");',
        'ABI_ASSERT(YVEX_SERVER_ENGINE_NONE == 0, "engine-kind none value changed");',
        'ABI_ASSERT(YVEX_SERVER_ENGINE_TEXT == 1, "engine-kind text value changed");',
        'ABI_ASSERT(YVEX_SERVER_ENGINE_MEDIA == 2, "engine-kind media value changed");',
        'ABI_ASSERT(YVEX_SERVER_EXECUTION_NOT_APPLICABLE == 0, '
        '"execution-strategy n/a value changed");',
        'ABI_ASSERT(YVEX_SERVER_EXECUTION_TARGET_ONLY == 1, '
        '"execution-strategy target value changed");',
        'ABI_ASSERT(YVEX_SERVER_EXECUTION_SPECULATIVE == 2, '
        '"execution-strategy speculative value changed");',
    ]
    for record, (_, version_macro, version, size, _) in RECORDS.items():
        lines.extend(
            [
                f'ABI_ASSERT({version_macro} == {version}u, "{record} schema changed");',
                f'ABI_ASSERT(sizeof({record}) == {size}u, "{record} layout changed");',
                f'ABI_ASSERT(offsetof({record}, schema_version) == 0u, '
                f'"{record} schema prefix changed");',
            ]
        )
    option_fields = {
        "schema_version": (0, 4),
        "socket_path": (8, 8),
        "request_queue_capacity": (16, 8),
        "worker_count": (24, 8),
        "maximum_engines": (32, 8),
        "openai_timeout_ms": (40, 8),
        "openai_port": (48, 2),
        "trace_level": (52, 4),
        "console": (56, 4),
        "trace_content": (60, 4),
        "openai_enabled": (64, 4),
        "model_loader": (72, 8),
        "model_loader_context": (80, 8),
    }
    for field, (offset, size) in option_fields.items():
        lines.extend(
            [
                f'ABI_ASSERT(offsetof(yvex_server_options, {field}) == {offset}u, '
                f'"server options {field} offset changed");',
                f'ABI_ASSERT(sizeof(((yvex_server_options *)0)->{field}) == {size}u, '
                f'"server options {field} type changed");',
            ]
        )
    return "\n".join(lines) + "\n"


def compile_contract(language: str, compiler: str, standard: str) -> list[str]:
    command = shlex.split(compiler) + [
        f"-std={standard}",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-D_FILE_OFFSET_BITS=64",
        "-D_POSIX_C_SOURCE=200809L",
        "-Iinclude",
        "-I.",
        "-fsyntax-only",
        "-x",
        language,
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            input=compiler_source(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as failure:
        return [f"{language} compiler cannot start: {failure}"]
    if result.returncode:
        return [f"{language} ABI compile failed: {result.stderr.strip()}"]
    return []


def main() -> int:
    errors: list[str] = []
    discovered = installed_versioned_records()
    untracked = sorted(discovered - RECORDS.keys())
    stale = sorted(RECORDS.keys() - discovered)
    if untracked:
        errors.append(f"versioned installed records missing from manifest: {untracked}")
    if stale:
        errors.append(f"ABI manifest records no longer installed: {stale}")
    for record, (header, _, _, _, expected) in RECORDS.items():
        try:
            actual = declaration_digest(header, record)
        except (OSError, ValueError) as failure:
            errors.append(str(failure))
            continue
        if actual != expected:
            errors.append(
                f"{record}: declaration changed without an explicit ABI manifest/version decision"
            )
    errors.extend(compile_contract("c", os.environ.get("CC", "cc"), "c11"))
    errors.extend(compile_contract("c++", os.environ.get("CXX", "c++"), "c++17"))
    if errors:
        for error in errors:
            print(f"public ABI: {error}", file=sys.stderr)
        return 1
    print(f"public ABI: {len(RECORDS)} versioned records stable in C and C++")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
