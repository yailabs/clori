#!/usr/bin/env python3
# Regenerates the tiny deterministic GGUF fixtures consumed by parser tests. The binary outputs are
# tracked test vectors; this script is maintenance tooling rather than a production writer.

from __future__ import annotations

import struct
from pathlib import Path


OUT = Path(__file__).resolve().parent
MAGIC = b"GGUF"
VERSION = 3

UINT8 = 0
INT8 = 1
UINT16 = 2
INT16 = 3
UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10
INT64 = 11
FLOAT64 = 12

GGML_TYPE_F32 = 0


def u8(value: int) -> bytes:
    return struct.pack("<B", value)


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def gguf_string(text: bytes | str) -> bytes:
    if isinstance(text, str):
        text = text.encode("utf-8")
    return u64(len(text)) + text


def scalar_value(value_type: int, value) -> bytes:
    if value_type == UINT8:
        return struct.pack("<B", value)
    if value_type == INT8:
        return struct.pack("<b", value)
    if value_type == UINT16:
        return struct.pack("<H", value)
    if value_type == INT16:
        return struct.pack("<h", value)
    if value_type == UINT32:
        return struct.pack("<I", value)
    if value_type == INT32:
        return struct.pack("<i", value)
    if value_type == FLOAT32:
        return struct.pack("<f", value)
    if value_type == BOOL:
        return struct.pack("<B", 1 if value else 0)
    if value_type == STRING:
        return gguf_string(value)
    if value_type == UINT64:
        return struct.pack("<Q", value)
    if value_type == INT64:
        return struct.pack("<q", value)
    if value_type == FLOAT64:
        return struct.pack("<d", value)
    raise ValueError(value_type)


def kv(key: str, value_type: int, value) -> bytes:
    return gguf_string(key) + u32(value_type) + scalar_value(value_type, value)


def kv_array(key: str, element_type: int, values: list) -> bytes:
    payload = gguf_string(key) + u32(ARRAY) + u32(element_type) + u64(len(values))
    for value in values:
        payload += scalar_value(element_type, value)
    return payload


def tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    payload = gguf_string(name)
    payload += u32(len(dims))
    for dim in dims:
        payload += u64(dim)
    payload += u32(ggml_type)
    payload += u64(offset)
    return payload


def align(data: bytes, alignment: int) -> bytes:
    pad = (alignment - (len(data) % alignment)) % alignment
    return data + (b"\x00" * pad)


def file_bytes(metadata: list[bytes], tensors: list[bytes], tensor_data: bytes = b"", alignment: int = 32) -> bytes:
    data = MAGIC + u32(VERSION) + u64(len(tensors)) + u64(len(metadata))
    data += b"".join(metadata)
    data += b"".join(tensors)
    if tensors:
        data = align(data, alignment)
        data += tensor_data
    return data


def write(name: str, data: bytes) -> None:
    (OUT / name).write_bytes(data)


def valid_metadata() -> list[bytes]:
    return [
        kv("general.architecture", STRING, "llama"),
        kv("general.name", STRING, "yvex-test"),
        kv("llama.context_length", UINT32, 4096),
        kv("general.file_type", UINT32, 0),
        kv("general.alignment", UINT32, 32),
    ]


def tokenizer_metadata(
    model: str = "yvex-fixture-simple",
    include_tokens: bool = True,
    scores: list[float] | None = None,
    token_types: list[int] | None = None,
    special_id: int | None = None,
) -> list[bytes]:
    tokens = ["<unk>", "<bos>", "<eos>", "hello", " ", "world", "Write", " C"]
    if scores is None:
        scores = [0.0] * len(tokens)
    if token_types is None:
        token_types = [2, 3, 3, 1, 1, 1, 1, 1]

    metadata = [
        kv("general.architecture", STRING, "llama"),
        kv("general.name", STRING, "yvex-tokenizer-test"),
        kv("llama.context_length", UINT32, 4096),
        kv("general.file_type", UINT32, 0),
        kv("general.alignment", UINT32, 32),
        kv("tokenizer.ggml.model", STRING, model),
    ]
    if include_tokens:
        metadata.append(kv_array("tokenizer.ggml.tokens", STRING, tokens))
    metadata.append(kv_array("tokenizer.ggml.scores", FLOAT32, scores))
    metadata.append(kv_array("tokenizer.ggml.token_type", INT32, token_types))
    metadata.extend([
        kv("tokenizer.ggml.bos_token_id", UINT32, 1),
        kv("tokenizer.ggml.eos_token_id", UINT32, 2),
        kv("tokenizer.ggml.unknown_token_id", UINT32, 0 if special_id is None else special_id),
    ])
    return metadata


def tiny_tensor_file(metadata: list[bytes]) -> bytes:
    return file_bytes(
        metadata,
        [tensor_info("token_embd.weight", [4, 8], GGML_TYPE_F32, 0)],
        b"\x00" * (4 * 8 * 4),
    )


def main() -> None:
    write(
        "valid-metadata-tensors.gguf",
        file_bytes(
            valid_metadata(),
            [tensor_info("token_embd.weight", [4, 8], GGML_TYPE_F32, 0)],
            b"\x00" * (4 * 8 * 4),
        ),
    )

    write(
        "metadata-unknown-type.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1) + gguf_string("general.name") + u32(99),
    )
    write(
        "metadata-string-oob.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1) + gguf_string("general.name") + u32(STRING) + u64(100),
    )
    write(
        "metadata-array-oob.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1)
        + gguf_string("general.tags") + u32(ARRAY) + u32(UINT32) + u64(3) + u32(1),
    )
    write(
        "metadata-nested.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1)
        + gguf_string("general.tags") + u32(ARRAY) + u32(ARRAY) + u64(1),
    )
    write(
        "metadata-bool-invalid.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1) + gguf_string("general.flag") + u32(BOOL) + u8(2),
    )
    write(
        "metadata-empty-key.gguf",
        MAGIC + u32(VERSION) + u64(0) + u64(1) + u64(0) + u32(UINT32) + u32(1),
    )

    metadata = valid_metadata()
    write(
        "tensor-name-out-of-bounds.gguf",
        MAGIC + u32(VERSION) + u64(1) + u64(len(metadata)) + b"".join(metadata) + u64(100),
    )
    write(
        "tensor-rank-zero.gguf",
        file_bytes(metadata, [gguf_string("token_embd.weight") + u32(0) + u32(GGML_TYPE_F32) + u64(0)]),
    )
    write(
        "tensor-rank-unsupported.gguf",
        file_bytes(metadata, [tensor_info("token_embd.weight", [1, 2, 3, 4, 5], GGML_TYPE_F32, 0)]),
    )
    write(
        "tensor-dim-zero.gguf",
        file_bytes(metadata, [tensor_info("token_embd.weight", [4, 0], GGML_TYPE_F32, 0)]),
    )
    write(
        "tensor-dim-overflow.gguf",
        file_bytes(metadata, [tensor_info("token_embd.weight", [1 << 63, 3], GGML_TYPE_F32, 0)]),
    )
    write(
        "tensor-offset-misaligned.gguf",
        file_bytes(metadata, [tensor_info("token_embd.weight", [4, 8], GGML_TYPE_F32, 1)], b"\x00" * 128),
    )
    write(
        "tensor-offset-out-of-bounds.gguf",
        file_bytes(metadata, [tensor_info("token_embd.weight", [4, 8], GGML_TYPE_F32, 1024)], b"\x00" * 128),
    )

    write("valid-tokenizer-simple.gguf", tiny_tensor_file(tokenizer_metadata()))
    write("tokenizer-missing-tokens.gguf", tiny_tensor_file(tokenizer_metadata(include_tokens=False)))
    write("tokenizer-type-len.gguf", tiny_tensor_file(tokenizer_metadata(token_types=[2, 3])))
    write("tokenizer-bad-score-len.gguf", tiny_tensor_file(tokenizer_metadata(scores=[0.0, 0.0])))
    write("tokenizer-bad-special-id.gguf", tiny_tensor_file(tokenizer_metadata(special_id=99)))
    write("tokenizer-unsupported-arch.gguf", tiny_tensor_file(tokenizer_metadata(model="mystery-tokenizer")))


if __name__ == "__main__":
    main()
