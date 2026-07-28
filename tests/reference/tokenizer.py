#!/usr/bin/env python3
"""Independent DeepSeek tokenizer/prompt oracle; production never imports this module."""
import importlib.util
import ast
import pathlib
import re
import subprocess
import sys

import tokenizers
from tokenizers import Tokenizer


def ids_from_cli(yvex, artifact, text):
    result = subprocess.run(
        [yvex, "tokenize", artifact, "--text", text],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r"^ids:(.*)$", result.stdout, re.MULTILINE)
    if not match:
        raise AssertionError("YVEX token IDs absent")
    return [int(value) for value in match.group(1).split()]


def text_from_cli(yvex, artifact, token_ids):
    result = subprocess.run(
        [yvex, "detokenize", artifact, "--ids", ",".join(map(str, token_ids))],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r'^text: (".*")$', result.stdout, re.MULTILINE)
    if not match:
        raise AssertionError("YVEX decoded text absent")
    escaped = ast.literal_eval(match.group(1))
    return escaped.encode("latin-1").decode("utf-8")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} SOURCE_DIR YVEX ARTIFACT")
    source, yvex, artifact = map(pathlib.Path, sys.argv[1:])
    if tokenizers.__version__ != "0.20.3":
        raise AssertionError(f"unexpected tokenizers version {tokenizers.__version__}")
    oracle = Tokenizer.from_file(str(source / "tokenizer.json"))
    corpus = [
        "", "hello world", "  repeated   spaces\nnext\tline", "café e\u0301",
        "😀🧠", "你好世界", "こんにちは世界", "Привет мир", "مرحبا بالعالم",
        "नमस्ते दुनिया", "1234567890", "<think>hello</think>",
        "<｜User｜>Hello<｜Assistant｜></think>",
    ]
    for text in corpus:
        expected = oracle.encode(text, add_special_tokens=False).ids
        actual = ids_from_cli(str(yvex), str(artifact), text)
        if actual != expected:
            raise AssertionError((text, expected, actual))
        if expected:
            decoded = text_from_cli(str(yvex), str(artifact), expected)
            expected_decoded = oracle.decode(expected, skip_special_tokens=False)
            if decoded != expected_decoded:
                raise AssertionError((expected, expected_decoded, decoded))
    module_path = source / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    expected_prompt = module.encode_messages(messages, "chat")
    result = subprocess.run(
        [str(yvex), "prompt", str(artifact), "--system", "policy", "--user", "hi",
         "--assistant", "ok", "--user", "next", "--tokens"],
        check=True, capture_output=True, text=True,
    )
    rendered = result.stdout.split("rendered:\n", 1)[1].split("\ntokens:", 1)[0]
    if rendered != expected_prompt:
        raise AssertionError((expected_prompt, rendered))
    tool_messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "call"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "one"},
        {"role": "tool", "content": "two"},
    ]
    expected_tool_prompt = module.encode_messages(tool_messages, "chat")
    result = subprocess.run(
        [str(yvex), "prompt", str(artifact), "--system", "policy", "--user", "call",
         "--assistant", "working", "--tool", "one", "--tool", "two", "--tokens"],
        check=True, capture_output=True, text=True,
    )
    rendered = result.stdout.split("rendered:\n", 1)[1].split("\ntokens:", 1)[0]
    if rendered != expected_tool_prompt:
        raise AssertionError((expected_tool_prompt, rendered))
    expected_thinking = module.encode_messages(
        [{"role": "user", "content": "reason"}], "thinking")
    result = subprocess.run(
        [str(yvex), "prompt", str(artifact), "--user", "reason", "--thinking"],
        check=True, capture_output=True, text=True,
    )
    rendered = result.stdout.split("rendered:\n", 1)[1].split("\nprompt_identity:", 1)[0]
    if rendered != expected_thinking:
        raise AssertionError((expected_thinking, rendered))
    print("tokenizer_reference=tokenizers-0.20.3 cases=13 prompt_cases=3 "
          "encode_decode_parity=pass")


if __name__ == "__main__":
    main()
