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
        "72083625e7033a734875bd042885e278ff84626da8e48e469850adbaabf548ea"),
    "yvex_provider_output": (
        "include/yvex/provider.h", "YVEX_PROVIDER_SCHEMA_V1", 1, 344,
        "ae3a0a9ad988467cd7e27c7052a999401bcb91df908d1d801ada1582bd9d3f44"),
    "yvex_imatrix_data_summary": (
        "include/yvex/quant.h", "YVEX_IMATRIX_DATA_SCHEMA_VERSION", 1, 216,
        "4b46211d7dcfe10b5044a792e2c29ef304d67f464f5e92c4117a01a700b08cbc"),
    "yvex_quant_policy_rule": (
        "include/yvex/quant.h", "YVEX_QUANT_POLICY_SCHEMA_VERSION", 2, 128,
        "6481bfdfb89d47208fa1c452af48da9fee5d30353e5b025da83a09f686804bb3"),
    "yvex_quant_policy_summary": (
        "include/yvex/quant.h", "YVEX_QUANT_POLICY_SCHEMA_VERSION", 2, 144,
        "0abbe27da9653e134ccd9d72c450ed9c8c421f0d8be42d4a00a6801564707bf0"),
    "yvex_server_event": (
        "include/yvex/server.h", "YVEX_RUNTIME_EVENT_SCHEMA_VERSION", 3, 904,
        "1656dc28a152d1cb3cf896637fa418ee3d6fcc0029af156ef210ad17891ca3b7"),
    "yvex_server_metrics": (
        "include/yvex/server.h", "YVEX_RUNTIME_METRICS_SCHEMA_VERSION", 3, 216,
        "ae3862b3b675ad48174f8d4f8238e37cfe2f87d2a5e88d5ec000b2409cc474ca"),
    "yvex_server_options": (
        "include/yvex/server.h", "YVEX_SERVER_OPTIONS_SCHEMA_CURRENT", 4, 88,
        "c14728a52058ffe119b4f3abcde8cac42c7fe62cb89bcfe3d85d70f8e180e257"),
    "yvex_server_engine_options": (
        "include/yvex/server.h", "YVEX_SERVER_ENGINE_SCHEMA_V1", 1, 128,
        "5842c0f71785cb8a8405a78f5fe3a99ea7ead527465a3f0fb57c7bd0cfa762e2"),
    "yvex_server_engine_summary": (
        "include/yvex/server.h", "YVEX_SERVER_ENGINE_SCHEMA_V1", 1, 720,
        "48a11618df62b8c2feb086fbda7c8f06ceed5170cb913c86a77546a861ced528"),
    "yvex_server_summary": (
        "include/yvex/server.h", "YVEX_SERVER_SUMMARY_SCHEMA_V1", 1, 832,
        "b7906cd8bf1e5f43bcc8580c979ff6447abb2e1fefc94913d588fa65c603ac66"),
    "yvex_client_partial_turn": (
        "include/yvex/server.h", "YVEX_CLIENT_PARTIAL_TURN_SCHEMA_V1", 1, 392,
        "9877c412a6bc23eda32becbdcc9b45b8f8a5b9e288be8f1530f03e026568d5c3"),
    "yvex_console_status": (
        "include/yvex/server.h", "YVEX_CONSOLE_STATUS_SCHEMA_V1", 1, 496,
        "c7a04e166f72d68123b2da718d3c7ef0a44cbed445959b4ba570563d2f40d1ce"),
    "yvex_client_state_checkpoint": (
        "include/yvex/server.h", "YVEX_CLIENT_STATE_CHECKPOINT_SCHEMA_V1", 1, 296,
        "cd2fc01ca1f85d05c9bf10cc13b00b1d24d9b1763c654c78acd0d61283db4058"),
    "yvex_client_media_result": (
        "include/yvex/server.h", "YVEX_CLIENT_MEDIA_RESULT_SCHEMA_V1", 1, 856,
        "de0ddcbee4fbda54d79a13c7cf8a4ee6ef8fcc09c694e1eb293dfe5713ae58bc"),
    "yvex_client_request": (
        "include/yvex/server.h", "YVEX_LOCAL_PROTOCOL_VERSION", 13, 920,
        "364c5afa2a79ae58ba4fa2629c9c172a9723235690db3b9e2323ef16766593f5"),
    "yvex_client_message": (
        "include/yvex/server.h", "YVEX_LOCAL_PROTOCOL_VERSION", 13, 10016,
        "6da40c51bbb75df94cf19bc1a4f4e4124ec85f37ed1b3daf097bed8c2ac2b182"),
    "yvex_tokenizer_plan_summary": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_PLAN_SCHEMA_V3", 3, 864,
        "0dffc80772b1119db3d2004c7f989775280c461b389f805307c3a2465876fb31"),
    "yvex_tokenizer_encode_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_EXECUTION_SCHEMA_V1", 1, 328,
        "c37d6705fc8b79f31ed298fd536b4e51082ac9aebce3f2a8aebaf5c29657b14f"),
    "yvex_tokenizer_decode_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_EXECUTION_SCHEMA_V1", 1, 240,
        "e3d9694def271ea2bb78e5850657efd26cbc609f9049b8f7eef5683df3e9fa37"),
    "yvex_tokenizer_fragment": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_DECODER_SCHEMA_V1", 1, 256,
        "c750a7793cdece8c52c9877299eb1ce2841f2d050a171d4cfdb5380a91b88f72"),
    "yvex_token_sequence_summary": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_APPEND_SCHEMA_V1", 1, 168,
        "66e74f93367bb5a766ba4fd6e2fff3aa33891506ee14edb79ea426d06442053d"),
    "yvex_tokenizer_provider_result": (
        "include/yvex/tokenizer.h", "YVEX_TOKENIZER_PROVIDER_RESULT_SCHEMA_V2", 2, 128,
        "fba58ed4c1432c2fca11b34df8efc353a5d0acf1e51cacf0fc9fc07ebad11aff"),
}


def declaration_digest(header: str, record: str) -> str:
    text = (ROOT / header).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)
    matches = re.findall(
        r"typedef\s+struct\s*\{(.*?)\}\s*" + re.escape(record) + r"\s*;",
        text,
        flags=re.DOTALL,
    )
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
