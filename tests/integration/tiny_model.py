#!/usr/bin/env python3
"""Emit the deterministic GGUF used by the real CPU composition test."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


UINT32 = 4
INT32 = 5
FLOAT32 = 6
STRING = 8
ARRAY = 9
GGML_F32 = 0
GGML_I32 = 26
ALIGNMENT = 32


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return u64(len(encoded)) + encoded


def scalar(value_type: int, value: object) -> bytes:
    if value_type == UINT32:
        return u32(int(value))
    if value_type == INT32:
        return struct.pack("<i", int(value))
    if value_type == FLOAT32:
        return struct.pack("<f", float(value))
    if value_type == STRING:
        return text(str(value))
    raise ValueError(f"unsupported metadata type {value_type}")


def metadata(key: str, value_type: int, value: object) -> bytes:
    return text(key) + u32(value_type) + scalar(value_type, value)


def array(key: str, value_type: int, values: list[object]) -> bytes:
    return (
        text(key)
        + u32(ARRAY)
        + u32(value_type)
        + u64(len(values))
        + b"".join(scalar(value_type, value) for value in values)
    )


def align(payload: bytes) -> bytes:
    return payload + b"\0" * ((-len(payload)) % ALIGNMENT)


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    qtype: int = GGML_F32

    @property
    def payload(self) -> bytes:
        elements = 1
        for dim in self.dims:
            elements *= dim
        return b"\0" * (elements * 4)


VOCABULARY_SIZE = 261


TENSORS = (
    Tensor("tiny.token_embedding", (4, VOCABULARY_SIZE)),
    Tensor("tiny.hc_head_function", (4, 1)),
    Tensor("tiny.hc_head_base", (1, 1)),
    Tensor("tiny.hc_head_scale", (1, 1)),
    Tensor("tiny.output_norm", (4, 1)),
    Tensor("tiny.output_head", (4, VOCABULARY_SIZE)),
    Tensor("tiny.layer.0.attention_sinks", (1, 1)),
    Tensor("tiny.layer.0.attention_norm", (4, 1)),
    Tensor("tiny.layer.0.hc_attention_function", (4, 3)),
    Tensor("tiny.layer.0.hc_attention_base", (3, 1)),
    Tensor("tiny.layer.0.hc_attention_scale", (3, 1)),
    Tensor("tiny.layer.0.attention_q_a_norm", (4, 1)),
    Tensor("tiny.layer.0.attention_kv_norm", (4, 1)),
    Tensor("tiny.layer.0.attention_q_a", (4, 4)),
    Tensor("tiny.layer.0.attention_q_b", (4, 4)),
    Tensor("tiny.layer.0.attention_kv", (4, 4)),
    Tensor("tiny.layer.0.attention_out_a", (4, 4)),
    Tensor("tiny.layer.0.attention_out_b", (4, 4)),
    Tensor("tiny.layer.0.ffn_norm", (4, 1)),
    Tensor("tiny.layer.0.hc_ffn_function", (4, 3)),
    Tensor("tiny.layer.0.hc_ffn_base", (3, 1)),
    Tensor("tiny.layer.0.hc_ffn_scale", (3, 1)),
    Tensor("tiny.layer.0.router", (4, 2)),
    Tensor("tiny.layer.0.router_table", (1, VOCABULARY_SIZE), GGML_I32),
    Tensor("tiny.layer.0.expert_gate", (4, 2)),
    Tensor("tiny.layer.0.expert_up", (4, 2)),
    Tensor("tiny.layer.0.expert_down", (1, 8)),
    Tensor("tiny.layer.0.shared_gate", (4, 1)),
    Tensor("tiny.layer.0.shared_up", (4, 1)),
    Tensor("tiny.layer.0.shared_down", (1, 4)),
)


def build() -> bytes:
    def byte_codepoint(byte: int) -> int:
        if 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255:
            return byte
        extra = sum(
            not (33 <= candidate <= 126 or 161 <= candidate <= 172 or 174 <= candidate <= 255)
            for candidate in range(byte)
        )
        return 256 + extra

    byte_tokens = [chr(byte_codepoint(byte)) for byte in range(256)]
    tokens = ["ok", *byte_tokens, "<eos>", "</think>", "<think>", "<unk>"]
    token_types = [1, *([6] * 256), 3, 3, 3, 2]
    assert len(tokens) == VOCABULARY_SIZE
    facts = [
        metadata("general.architecture", STRING, "tiny-executable"),
        metadata("general.name", STRING, "YVEX tiny executable vertical"),
        metadata("general.file_type", UINT32, 0),
        metadata("general.alignment", UINT32, ALIGNMENT),
        metadata("tiny.context_length", UINT32, 8),
        metadata("tokenizer.ggml.model", STRING, "yvex-fixture-simple"),
        metadata("tokenizer.ggml.pre", STRING, "tiny"),
        metadata("tokenizer.huggingface.json", STRING, "{}"),
        metadata("yvex.tokenizer.config.json", STRING, "{}"),
        array("tokenizer.ggml.tokens", STRING, tokens),
        array("tokenizer.ggml.scores", FLOAT32, [0.0] * len(tokens)),
        array("tokenizer.ggml.token_type", INT32, token_types),
        array("tokenizer.ggml.merges", STRING, ["o k"]),
        metadata("tokenizer.ggml.eos_token_id", UINT32, 257),
        metadata("tokenizer.ggml.unknown_token_id", UINT32, 260),
    ]
    offsets: list[int] = []
    data = b""
    for tensor in TENSORS:
        data = align(data)
        offsets.append(len(data))
        data += tensor.payload
    directory = b""
    for tensor, offset in zip(TENSORS, offsets, strict=True):
        directory += text(tensor.name) + u32(len(tensor.dims))
        directory += b"".join(u64(dim) for dim in tensor.dims)
        directory += u32(tensor.qtype) + u64(offset)
    header = b"GGUF" + u32(3) + u64(len(TENSORS)) + u64(len(facts))
    return align(header + b"".join(facts) + directory) + align(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="flip one trailing artifact byte for the admission-refusal fixture",
    )
    args = parser.parse_args()
    payload = bytearray(build())
    if args.corrupt:
        payload[-1] ^= 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
