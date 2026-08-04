#!/usr/bin/env python3
"""Acquire bounded MiniMax-H3 FL2VA evidence or the authorized immutable source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import stat
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import Message
from typing import Any


TOOL_SCHEMA = "yvex.minimax-h3.intake.v1"
TOOL_VERSION = 1
ACQUISITION_SCHEMA = "yvex.source-acquisition.v1"
ACQUISITION_MANIFEST = "yvex-source-acquisition.json"
AUTHORIZATION_ENV = "YVEX_MINIMAX_H3_LOCAL_RESEARCH_AUTHORIZED"
REQUIRED_SUBDIR = "FL2VA"
MAX_API_BYTES = 16 * 1024 * 1024
MAX_TREE_PAGES = 128
MAX_TREE_ENTRIES = 20_000
MAX_METADATA_FILE_BYTES = 64 * 1024 * 1024
MAX_METADATA_TOTAL_BYTES = 256 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 1_000_000
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


class IntakeError(ValueError):
    """One intake input or source invariant is invalid."""


def fail(message: str) -> None:
    raise IntakeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def acquisition_identity(repo: str, revision: str, subdir: str, files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for value in (ACQUISITION_SCHEMA, repo, revision, subdir):
        identity_text(digest, value)
    digest.update(struct.pack("<Q", len(files)))
    for row in files:
        identity_text(digest, row["path"])
        digest.update(struct.pack("<Q", row["expected_size"]))
        identity_text(digest, row["expected_sha256"])
        identity_text(digest, row["actual_sha256"])
        identity_text(digest, row["git_oid"])
        identity_text(digest, row["lfs_oid"])
        identity_text(digest, row["xet_hash"])
        identity_text(digest, row["classification"])
        identity_text(digest, row["component"])
    return digest.hexdigest()


def require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{where}: expected an object")
    return value


def require_array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{where}: expected an array")
    return value


def require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{where}: expected non-empty text")
    return value


def require_uint(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        fail(f"{where}: expected a non-negative integer")
    return value


def allowed_delivery_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    hostname = hostname.lower().rstrip(".")
    return (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
        or hostname == "hf.co"
        or hostname.endswith(".hf.co")
    )


def safe_url_identity(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


class RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only within recognized Hugging Face delivery domains."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or not allowed_delivery_host(parsed.hostname):
            fail(f"refused redirect to unrecognized delivery host {parsed.hostname!r}")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and parsed.hostname != "huggingface.co":
            redirected.remove_header("Authorization")
        return redirected


@dataclass(frozen=True)
class Download:
    category: str
    source: str
    requested_range: str
    byte_count: int
    digest: str


class Provider:
    def __init__(self) -> None:
        self.downloads: list[Download] = []

    def repository(self, repo: str, revision: str) -> dict[str, Any]:
        raise NotImplementedError

    def tree(self, repo: str, revision: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def file_bytes(self, repo: str, revision: str, path: str, category: str, limit: int) -> bytes:
        raise NotImplementedError

    def file_range(
        self, repo: str, revision: str, path: str, start: int, end: int, category: str
    ) -> bytes:
        raise NotImplementedError

    def stream_file(
        self,
        repo: str,
        revision: str,
        path: str,
        offset: int,
        stream: Any,
        digest: Any,
        expected_size: int,
    ) -> int:
        raise NotImplementedError

    def record(self, category: str, source: str, requested_range: str, data: bytes) -> None:
        self.downloads.append(Download(category, source, requested_range, len(data), sha256(data)))


class NetworkProvider(Provider):
    def __init__(self, token: str | None) -> None:
        super().__init__()
        self.token = token
        self.opener = urllib.request.build_opener(RestrictedRedirect())

    def request(
        self,
        url: str,
        category: str,
        limit: int,
        requested_range: str = "",
        headers: dict[str, str] | None = None,
        require_partial: bool = False,
    ) -> tuple[bytes, Message]:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not allowed_delivery_host(parsed.hostname):
            fail(f"refused non-Hugging-Face HTTPS source {safe_url_identity(url)!r}")
        request = urllib.request.Request(url, headers={"User-Agent": "yvex-minimax-h3-intake/1"})
        if headers:
            for name, value in headers.items():
                request.add_header(name, value)
        if self.token:
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        try:
            with self.opener.open(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > limit:
                    fail(f"{category}: response exceeds {limit} bytes")
                if require_partial and response.status != 206:
                    fail(f"{category}: server ignored the bounded range request")
                data = response.read(limit + 1)
                if len(data) > limit:
                    fail(f"{category}: response exceeds {limit} bytes")
                final_host = urllib.parse.urlsplit(response.geturl()).hostname
                if not allowed_delivery_host(final_host):
                    fail(f"{category}: response arrived from an unrecognized host")
                response_headers = response.headers
        except IntakeError:
            raise
        except (OSError, ValueError, urllib.error.URLError) as exc:
            fail(f"{category}: acquisition failed for {safe_url_identity(url)}: {exc}")
        self.record(category, safe_url_identity(url), requested_range, data)
        return data, response_headers

    def repository(self, repo: str, revision: str) -> dict[str, Any]:
        quoted_repo = urllib.parse.quote(repo, safe="/")
        quoted_revision = urllib.parse.quote(revision, safe="")
        url = (
            f"https://huggingface.co/api/models/{quoted_repo}/revision/{quoted_revision}"
            "?expand%5B%5D=sha&expand%5B%5D=cardData"
        )
        data, _ = self.request(url, "repository-metadata", MAX_API_BYTES)
        try:
            return require_mapping(json.loads(data), "repository metadata")
        except (UnicodeError, json.JSONDecodeError) as exc:
            fail(f"repository metadata: invalid JSON: {exc}")

    def tree(self, repo: str, revision: str) -> list[dict[str, Any]]:
        quoted_repo = urllib.parse.quote(repo, safe="/")
        url = (
            f"https://huggingface.co/api/models/{quoted_repo}/tree/{revision}"
            "?recursive=true"
        )
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _ in range(MAX_TREE_PAGES):
            if url in seen:
                fail("repository tree: pagination cycle")
            seen.add(url)
            data, headers = self.request(url, "repository-tree", MAX_API_BYTES)
            try:
                rows = require_array(json.loads(data), "repository tree page")
            except (UnicodeError, json.JSONDecodeError) as exc:
                fail(f"repository tree: invalid JSON: {exc}")
            for row in rows:
                result.append(require_mapping(row, "repository tree row"))
            if len(result) > MAX_TREE_ENTRIES:
                fail(f"repository tree: exceeds {MAX_TREE_ENTRIES} entries")
            next_url = None
            for candidate, attributes in parse_link_header(headers.get("Link", "")):
                if attributes.get("rel") == "next":
                    next_url = candidate
                    break
            if next_url is None:
                return result
            parsed = urllib.parse.urlsplit(next_url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                fail("repository tree: refused pagination host")
            url = next_url
        fail(f"repository tree: exceeds {MAX_TREE_PAGES} pages")

    def resolve_url(self, repo: str, revision: str, path: str) -> str:
        quoted_repo = urllib.parse.quote(repo, safe="/")
        quoted_path = urllib.parse.quote(path, safe="/")
        return f"https://huggingface.co/{quoted_repo}/resolve/{revision}/{quoted_path}"

    def file_bytes(self, repo: str, revision: str, path: str, category: str, limit: int) -> bytes:
        data, _ = self.request(self.resolve_url(repo, revision, path), category, limit)
        return data

    def file_range(
        self, repo: str, revision: str, path: str, start: int, end: int, category: str
    ) -> bytes:
        if start < 0 or end < start:
            fail(f"{path}: invalid requested range")
        requested = f"bytes={start}-{end}"
        data, headers = self.request(
            self.resolve_url(repo, revision, path),
            category,
            end - start + 1,
            requested,
            {"Range": requested},
            require_partial=True,
        )
        expected = f"bytes {start}-{end}/"
        if not headers.get("Content-Range", "").startswith(expected):
            fail(f"{path}: invalid Content-Range for {requested}")
        if len(data) != end - start + 1:
            fail(f"{path}: short response for {requested}")
        return data

    def stream_file(
        self,
        repo: str,
        revision: str,
        path: str,
        offset: int,
        stream: Any,
        digest: Any,
        expected_size: int,
    ) -> int:
        if offset < 0 or offset > expected_size:
            fail(f"{path}: invalid resume offset")
        if offset == expected_size:
            return 0
        url = self.resolve_url(repo, revision, path)
        headers = {"User-Agent": "yvex-minimax-h3-intake/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-{expected_size - 1}"
        request = urllib.request.Request(url, headers=headers)
        if self.token:
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        transferred = 0
        try:
            with self.opener.open(request, timeout=120) as response:
                final_host = urllib.parse.urlsplit(response.geturl()).hostname
                if not allowed_delivery_host(final_host):
                    fail(f"source-payload: response arrived from an unrecognized host for {path}")
                if offset:
                    expected_range = f"bytes {offset}-{expected_size - 1}/{expected_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        fail(f"{path}: server refused exact resume range")
                elif response.status not in {200, 206}:
                    fail(f"{path}: unexpected source response status {response.status}")
                length = response.headers.get("Content-Length")
                remaining = expected_size - offset
                if length is not None and int(length) != remaining:
                    fail(f"{path}: response length does not match immutable tree size")
                while transferred < remaining:
                    chunk = response.read(min(1024 * 1024, remaining - transferred))
                    if not chunk:
                        fail(f"{path}: short source payload response")
                    stream.write(chunk)
                    digest.update(chunk)
                    transferred += len(chunk)
                if response.read(1):
                    fail(f"{path}: source payload exceeds immutable tree size")
        except IntakeError:
            raise
        except (OSError, ValueError, urllib.error.URLError) as exc:
            fail(f"source-payload: acquisition failed for {safe_url_identity(url)}: {exc}")
        return transferred


class FixtureProvider(Provider):
    def __init__(self, root: pathlib.Path) -> None:
        super().__init__()
        self.root = root.resolve()

    def fixture_read(self, relative: str, category: str, limit: int) -> bytes:
        path = self.root / relative
        try:
            size = path.stat().st_size
            if size > limit:
                fail(f"{category}: fixture exceeds {limit} bytes")
            data = path.read_bytes()
        except OSError as exc:
            fail(f"{category}: cannot read fixture {relative!r}: {exc}")
        self.record(category, f"fixture:{relative}", "", data)
        return data

    def repository(self, repo: str, revision: str) -> dict[str, Any]:
        data = self.fixture_read("repository.json", "repository-metadata", MAX_API_BYTES)
        try:
            return require_mapping(json.loads(data), "fixture repository metadata")
        except (UnicodeError, json.JSONDecodeError) as exc:
            fail(f"fixture repository metadata: invalid JSON: {exc}")

    def tree(self, repo: str, revision: str) -> list[dict[str, Any]]:
        data = self.fixture_read("tree.json", "repository-tree", MAX_API_BYTES)
        try:
            rows = require_array(json.loads(data), "fixture repository tree")
        except (UnicodeError, json.JSONDecodeError) as exc:
            fail(f"fixture repository tree: invalid JSON: {exc}")
        if len(rows) > MAX_TREE_ENTRIES:
            fail(f"repository tree: exceeds {MAX_TREE_ENTRIES} entries")
        return [require_mapping(row, "fixture repository tree row") for row in rows]

    def source_path(self, path: str) -> pathlib.Path:
        candidate = (self.root / "files" / path).resolve()
        expected = (self.root / "files").resolve()
        if expected not in candidate.parents:
            fail(f"fixture path escapes root: {path!r}")
        return candidate

    def file_bytes(self, repo: str, revision: str, path: str, category: str, limit: int) -> bytes:
        source = self.source_path(path)
        try:
            size = source.stat().st_size
            if size > limit:
                fail(f"{category}: fixture file exceeds {limit} bytes")
            data = source.read_bytes()
        except OSError as exc:
            fail(f"{category}: cannot read fixture file {path!r}: {exc}")
        self.record(category, f"fixture:{path}", "", data)
        return data

    def file_range(
        self, repo: str, revision: str, path: str, start: int, end: int, category: str
    ) -> bytes:
        source = self.source_path(path)
        requested = f"bytes={start}-{end}"
        try:
            with source.open("rb") as stream:
                stream.seek(start)
                data = stream.read(end - start + 1)
        except OSError as exc:
            fail(f"{category}: cannot read fixture file {path!r}: {exc}")
        if len(data) != end - start + 1:
            fail(f"{path}: short fixture response for {requested}")
        self.record(category, f"fixture:{path}", requested, data)
        return data

    def stream_file(
        self,
        repo: str,
        revision: str,
        path: str,
        offset: int,
        stream: Any,
        digest: Any,
        expected_size: int,
    ) -> int:
        source = self.source_path(path)
        try:
            if source.stat().st_size != expected_size:
                fail(f"{path}: fixture size does not match immutable tree size")
            if offset == expected_size:
                return 0
            with source.open("rb") as input_stream:
                input_stream.seek(offset)
                transferred = 0
                while True:
                    chunk = input_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    transferred += len(chunk)
        except OSError as exc:
            fail(f"source-payload: cannot read fixture file {path!r}: {exc}")
        return transferred


def parse_link_header(value: str) -> list[tuple[str, dict[str, str]]]:
    links: list[tuple[str, dict[str, str]]] = []
    for item in value.split(","):
        item = item.strip()
        if not item.startswith("<") or ">" not in item:
            continue
        url, remainder = item[1:].split(">", 1)
        attributes: dict[str, str] = {}
        for field in remainder.split(";"):
            if "=" not in field:
                continue
            name, raw = field.strip().split("=", 1)
            attributes[name.lower()] = raw.strip().strip('"')
        links.append((url, attributes))
    return links


def parse_repo(value: str) -> str:
    if "://" in value:
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme != "https":
            fail("source repository URL must use HTTPS")
        if parsed.hostname != "huggingface.co" or parsed.query or parsed.fragment:
            fail("source repository URL must identify huggingface.co without query or fragment")
        value = parsed.path.strip("/")
    if not REPO_RE.fullmatch(value):
        fail("source repository must be an owner/name Hugging Face repository")
    return value


def normalize_tree(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        path = require_text(row.get("path"), f"tree[{index}].path")
        kind = require_text(row.get("type"), f"tree[{index}].type")
        if path.startswith("/") or ".." in pathlib.PurePosixPath(path).parts:
            fail(f"repository tree: unsafe path {path!r}")
        if path in seen:
            fail(f"repository tree: duplicate path {path!r}")
        seen.add(path)
        if kind not in {"file", "directory"}:
            fail(f"repository tree: unsupported entry type {kind!r}")
        size = require_uint(row.get("size", 0), f"tree[{index}].size")
        lfs = row.get("lfs")
        lfs_oid = ""
        lfs_size: int | None = None
        if lfs is not None:
            lfs_row = require_mapping(lfs, f"tree[{index}].lfs")
            lfs_oid = require_text(lfs_row.get("oid"), f"tree[{index}].lfs.oid")
            lfs_size = require_uint(lfs_row.get("size"), f"tree[{index}].lfs.size")
            if kind == "file" and lfs_size != size:
                fail(f"repository tree: LFS size mismatch for {path!r}")
        normalized.append(
            {
                "lfs_oid": lfs_oid,
                "lfs_size": lfs_size,
                "oid": require_text(row.get("oid"), f"tree[{index}].oid"),
                "path": path,
                "size": size,
                "type": kind,
                "xet_hash": row.get("xetHash", "") if isinstance(row.get("xetHash", ""), str) else "",
            }
        )
    return sorted(normalized, key=lambda row: row["path"])


def component_for(subdir: str, path: str) -> str:
    relative = pathlib.PurePosixPath(path).relative_to(subdir)
    if len(relative.parts) < 2:
        fail(f"{path}: safetensors file has no component owner")
    return relative.parts[0]


def acquisition_component(subdir: str, path: str) -> str:
    relative = pathlib.PurePosixPath(path).relative_to(subdir)
    return relative.parts[0] if len(relative.parts) > 1 else "pipeline"


def is_license(path: str) -> bool:
    name = pathlib.PurePosixPath(path).name.lower()
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    return "license" in name and suffix in {"", ".md", ".txt", ".rst"}


def metadata_category(subdir: str, path: str) -> str | None:
    if is_license(path):
        return "license"
    prefix = f"{subdir}/"
    if not path.startswith(prefix):
        return None
    pure = pathlib.PurePosixPath(path)
    if path.endswith(".safetensors.index.json"):
        return "safetensors-index"
    if pure.suffix.lower() == ".json":
        if any(part in {"processor", "tokenizer", "text_encoder"} for part in pure.parts):
            return "tokenizer-processor-metadata"
        return "configuration"
    if pure.suffix.lower() in {".txt", ".model"} and any(
        part in {"processor", "tokenizer", "text_encoder"} for part in pure.parts
    ):
        return "tokenizer-processor-metadata"
    return None


def output_metadata_path(category: str, source: str) -> pathlib.PurePosixPath:
    if category == "license":
        return pathlib.PurePosixPath("licenses") / source
    return pathlib.PurePosixPath("configs") / source


def parse_safetensors(
    provider: Provider,
    repo: str,
    revision: str,
    subdir: str,
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = row["path"]
    component = component_for(subdir, path)
    prefix = provider.file_range(repo, revision, path, 0, 7, "safetensors-header-prefix")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length > MAX_SAFETENSORS_HEADER_BYTES:
        fail(f"{path}: safetensors header exceeds {MAX_SAFETENSORS_HEADER_BYTES} bytes")
    if header_length < 2:
        fail(f"{path}: malformed safetensors header length {header_length}")
    file_size = require_uint(row["size"], f"{path}.size")
    if 8 + header_length > file_size:
        fail(f"{path}: declared safetensors header exceeds file size")
    raw = provider.file_range(
        repo,
        revision,
        path,
        8,
        7 + header_length,
        "safetensors-header-json",
    )
    try:
        header = require_mapping(json.loads(raw), f"{path} header")
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"{path}: invalid safetensors header JSON: {exc}")
    payload_size = file_size - 8 - header_length
    tensors: list[dict[str, Any]] = []
    positive_ranges: list[tuple[int, int, str]] = []
    for name in sorted(header):
        if name == "__metadata__":
            require_mapping(header[name], f"{path}.__metadata__")
            continue
        descriptor = require_mapping(header[name], f"{path}:{name}")
        dtype = require_text(descriptor.get("dtype"), f"{path}:{name}.dtype")
        if dtype not in DTYPE_BYTES:
            fail(f"{path}:{name}: unsupported or unknown dtype {dtype!r}")
        shape_raw = require_array(descriptor.get("shape"), f"{path}:{name}.shape")
        shape = [require_uint(value, f"{path}:{name}.shape") for value in shape_raw]
        offsets = require_array(descriptor.get("data_offsets"), f"{path}:{name}.data_offsets")
        if len(offsets) != 2:
            fail(f"{path}:{name}: data_offsets must contain two values")
        start = require_uint(offsets[0], f"{path}:{name}.data_offsets[0]")
        end = require_uint(offsets[1], f"{path}:{name}.data_offsets[1]")
        if end < start:
            fail(f"{path}:{name}: malformed descending offsets")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected = elements * DTYPE_BYTES[dtype]
        if end - start != expected:
            fail(f"{path}:{name}: payload span does not match dtype and shape")
        if end > payload_size:
            fail(f"{path}:{name}: payload span exceeds declared shard size")
        if end > start:
            positive_ranges.append((start, end, name))
        tensors.append(
            {
                "component": component,
                "declared_payload_span": end - start,
                "dtype": dtype,
                "elements": elements,
                "end": end,
                "name": name,
                "shape": shape,
                "shard": path,
                "start": start,
            }
        )
        if len(tensors) > MAX_TENSORS:
            fail(f"tensor inventory exceeds {MAX_TENSORS} entries")
    positive_ranges.sort()
    for previous, current in zip(positive_ranges, positive_ranges[1:]):
        if current[0] < previous[1]:
            fail(f"{path}: overlapping offsets for {previous[2]!r} and {current[2]!r}")
    header_record = {
        "component": component,
        "file_size": file_size,
        "header": header,
        "header_json_bytes": header_length,
        "payload_bytes": payload_size,
        "shard": path,
    }
    return header_record, tensors


def reconcile_indexes(
    subdir: str,
    files: dict[str, dict[str, Any]],
    metadata: dict[str, bytes],
    tensors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_component: dict[str, dict[str, dict[str, Any]]] = {}
    for tensor in tensors:
        component = tensor["component"]
        component_tensors = by_component.setdefault(component, {})
        name = tensor["name"]
        if name in component_tensors:
            fail(f"{component}: duplicate tensor name {name!r}")
        component_tensors[name] = tensor
    shard_components: dict[str, set[str]] = {}
    for path in files:
        if path.startswith(f"{subdir}/") and path.endswith(".safetensors"):
            shard_components.setdefault(component_for(subdir, path), set()).add(path)
    indexes: dict[str, tuple[str, dict[str, str], dict[str, Any]]] = {}
    for path, data in metadata.items():
        if not path.endswith(".safetensors.index.json"):
            continue
        component = component_for(subdir, path)
        if component in indexes:
            fail(f"{component}: multiple shard indexes are not admitted")
        try:
            document = require_mapping(json.loads(data), path)
        except (UnicodeError, json.JSONDecodeError) as exc:
            fail(f"{path}: invalid shard-index JSON: {exc}")
        raw_map = require_mapping(document.get("weight_map"), f"{path}.weight_map")
        weight_map: dict[str, str] = {}
        parent = pathlib.PurePosixPath(path).parent
        for name, shard_name in raw_map.items():
            tensor_name = require_text(name, f"{path}.weight_map key")
            relative_shard = require_text(shard_name, f"{path}.weight_map[{tensor_name!r}]")
            resolved = str(parent / relative_shard)
            if component_for(subdir, resolved) != component:
                fail(f"{path}: inconsistent component attribution for {relative_shard!r}")
            weight_map[tensor_name] = resolved
        indexes[component] = (path, weight_map, document)
    validation: list[dict[str, Any]] = []
    for component in sorted(shard_components):
        shards = shard_components[component]
        component_tensors = by_component.get(component, {})
        if component not in indexes:
            if len(shards) > 1:
                fail(f"{component}: multiple shards have no shard index")
            validation.append(
                {
                    "component": component,
                    "index": None,
                    "indexed": False,
                    "shards": len(shards),
                    "tensors": len(component_tensors),
                    "validated": True,
                }
            )
            continue
        index_path, weight_map, document = indexes[component]
        referenced_shards = set(weight_map.values())
        missing = sorted(referenced_shards - set(files))
        if missing:
            fail(f"{index_path}: missing shard {missing[0]!r}")
        stale = sorted(shards - referenced_shards)
        if stale:
            fail(f"{index_path}: shard absent from index {stale[0]!r}")
        absent = sorted(set(component_tensors) - set(weight_map))
        if absent:
            fail(f"{index_path}: tensor absent from index {absent[0]!r}")
        unknown = sorted(set(weight_map) - set(component_tensors))
        if unknown:
            fail(f"{index_path}: indexed tensor absent from shard headers {unknown[0]!r}")
        for name, shard in sorted(weight_map.items()):
            if component_tensors[name]["shard"] != shard:
                fail(f"{index_path}: stale shard reference for tensor {name!r}")
        metadata_row = document.get("metadata", {})
        if metadata_row is not None:
            metadata_map = require_mapping(metadata_row, f"{index_path}.metadata")
            if "total_size" in metadata_map:
                declared = require_uint(metadata_map["total_size"], f"{index_path}.metadata.total_size")
                actual = sum(tensor["declared_payload_span"] for tensor in component_tensors.values())
                if declared != actual:
                    fail(f"{index_path}: index total_size does not match tensor spans")
        validation.append(
            {
                "component": component,
                "index": index_path,
                "indexed": True,
                "shards": len(shards),
                "tensors": len(component_tensors),
                "validated": True,
            }
        )
    extra_indexes = sorted(set(indexes) - set(shard_components))
    if extra_indexes:
        fail(f"{indexes[extra_indexes[0]][0]}: index has no component shards")
    return validation


def write_tsv(columns: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return stream.getvalue().encode("utf-8")


def build_products(
    provider: Provider, repo_input: str, requested_revision: str, subdir: str
) -> dict[pathlib.PurePosixPath, bytes]:
    repo = parse_repo(repo_input)
    if subdir != REQUIRED_SUBDIR:
        fail(f"subdirectory must be exactly {REQUIRED_SUBDIR!r}")
    repository = provider.repository(repo, requested_revision)
    revision = require_text(repository.get("sha"), "repository.sha").lower()
    if not COMMIT_RE.fullmatch(revision):
        fail("remote source revision did not resolve to an immutable full commit hash")
    tree = normalize_tree(provider.tree(repo, revision))
    files = {row["path"]: row for row in tree if row["type"] == "file"}
    if not any(path.startswith(f"{subdir}/") for path in files):
        fail(f"repository has no files beneath {subdir!r}")
    source_tree_material = {
        "repository": repo,
        "revision": revision,
        "tree": tree,
    }
    source_tree_identity = sha256(canonical_json(source_tree_material))
    selected_metadata: dict[str, bytes] = {}
    products: dict[pathlib.PurePosixPath, bytes] = {}
    total_metadata = 0
    source_rows: list[dict[str, Any]] = []
    for path, row in sorted(files.items()):
        category = metadata_category(subdir, path)
        admitted = path.startswith(f"{subdir}/")
        source_rows.append(
            {
                "admitted_fl2va": "true" if admitted else "false",
                "lfs_oid": row["lfs_oid"],
                "oid": row["oid"],
                "path": path,
                "selected_category": category or "",
                "size": row["size"],
                "xet_hash": row["xet_hash"],
            }
        )
        if category is None:
            continue
        if not admitted and category != "license":
            fail(f"unexpected selected file outside {subdir}: {path!r}")
        data = provider.file_bytes(repo, revision, path, category, MAX_METADATA_FILE_BYTES)
        if len(data) != row["size"]:
            fail(f"{path}: downloaded metadata size does not match repository tree")
        total_metadata += len(data)
        if total_metadata > MAX_METADATA_TOTAL_BYTES:
            fail(f"selected metadata exceeds {MAX_METADATA_TOTAL_BYTES} bytes")
        selected_metadata[path] = data
        products[output_metadata_path(category, path)] = data
    shard_rows = [
        row
        for row in tree
        if row["type"] == "file"
        and row["path"].startswith(f"{subdir}/")
        and row["path"].endswith(".safetensors")
    ]
    if not shard_rows:
        fail(f"repository has no safetensors shards beneath {subdir!r}")
    headers: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    for row in shard_rows:
        header, shard_tensors = parse_safetensors(provider, repo, revision, subdir, row)
        headers.append(header)
        tensors.extend(shard_tensors)
        if len(tensors) > MAX_TENSORS:
            fail(f"tensor inventory exceeds {MAX_TENSORS} entries")
        relative = pathlib.PurePosixPath(row["path"] + ".header.json")
        products[pathlib.PurePosixPath("safetensors-headers") / relative] = pretty_json(header)
    validation = reconcile_indexes(subdir, files, selected_metadata, tensors)
    tensors.sort(key=lambda row: (row["component"], row["shard"], row["name"]))
    components: list[dict[str, Any]] = []
    dtype_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    for component in sorted({row["component"] for row in tensors}):
        selected = [row for row in tensors if row["component"] == component]
        component_shards = sorted({row["shard"] for row in selected})
        dtypes: dict[str, dict[str, int]] = {}
        shapes: dict[str, dict[str, int]] = {}
        for tensor in selected:
            dtype = dtypes.setdefault(tensor["dtype"], {"tensors": 0, "elements": 0, "payload_bytes": 0})
            dtype["tensors"] += 1
            dtype["elements"] += tensor["elements"]
            dtype["payload_bytes"] += tensor["declared_payload_span"]
            shape_key = "x".join(str(value) for value in tensor["shape"]) or "scalar"
            shape = shapes.setdefault(shape_key, {"tensors": 0, "elements": 0, "payload_bytes": 0})
            shape["tensors"] += 1
            shape["elements"] += tensor["elements"]
            shape["payload_bytes"] += tensor["declared_payload_span"]
        largest = sorted(
            selected,
            key=lambda row: (-row["declared_payload_span"], row["name"]),
        )[:10]
        component_files = sorted(path for path in selected_metadata if path.startswith(f"{subdir}/{component}/"))
        components.append(
            {
                "component": component,
                "dtypes": dtypes,
                "elements": sum(row["elements"] for row in selected),
                "largest_tensors": [
                    {
                        "dtype": row["dtype"],
                        "name": row["name"],
                        "payload_bytes": row["declared_payload_span"],
                        "shape": row["shape"],
                    }
                    for row in largest
                ],
                "metadata_files": component_files,
                "payload_bytes": sum(row["declared_payload_span"] for row in selected),
                "shard_files": component_shards,
                "shards": len(component_shards),
                "source_file_bytes": sum(files[path]["size"] for path in component_shards + component_files),
                "tensors": len(selected),
            }
        )
        for dtype_name, counts in sorted(dtypes.items()):
            dtype_rows.append({"component": component, "dtype": dtype_name, **counts})
        for shape_name, counts in sorted(shapes.items()):
            shape_rows.append({"component": component, "shape": shape_name, **counts})
    tensor_rows = [
        {
            **row,
            "shape": "x".join(str(value) for value in row["shape"]) or "scalar",
        }
        for row in tensors
    ]
    downloads = [
        {
            "byte_count": row.byte_count,
            "category": row.category,
            "requested_range": row.requested_range,
            "sha256": row.digest,
            "source": row.source,
        }
        for row in provider.downloads
    ]
    category_bytes: dict[str, int] = {}
    for row in downloads:
        category_bytes[row["category"]] = category_bytes.get(row["category"], 0) + row["byte_count"]
    license_rows = [
        {"path": path, "sha256": sha256(data), "size": len(data)}
        for path, data in sorted(selected_metadata.items())
        if is_license(path)
    ]
    repository_record = {
        "declared_license": repository.get("cardData", {}).get("license")
        if isinstance(repository.get("cardData"), dict)
        else None,
        "repository": repo,
        "requested_revision": requested_revision,
        "resolved_revision": revision,
        "source_url": f"https://huggingface.co/{repo}",
    }
    identity_material = {
        "components": components,
        "licenses": license_rows,
        "metadata": [
            {"path": path, "sha256": sha256(data), "size": len(data)}
            for path, data in sorted(selected_metadata.items())
        ],
        "repository": repo,
        "revision": revision,
        "schema": TOOL_SCHEMA,
        "source_tree_identity": source_tree_identity,
        "subdir": subdir,
        "tensors": tensor_rows,
    }
    intake_identity = sha256(canonical_json(identity_material))
    excluded_top_levels = sorted(
        {
            pathlib.PurePosixPath(path).parts[0]
            for path in files
            if not path.startswith(f"{subdir}/") and not is_license(path)
        }
    )
    result = {
        "byte_counts": category_bytes,
        "components": len(components),
        "declared_license": repository_record["declared_license"],
        "downloads": len(downloads),
        "excluded_top_levels": excluded_top_levels,
        "fl2va_files": sum(1 for path in files if path.startswith(f"{subdir}/")),
        "intake_identity": intake_identity,
        "license_files": license_rows,
        "payload_bytes": sum(row["declared_payload_span"] for row in tensors),
        "repository": repo,
        "repository_files": len(files),
        "reproduction_command": (
            f"python3 tools/minimax_h3_intake.py --repo {repo} --revision {revision} "
            f"--subdir {subdir} --output <OUTPUT>"
        ),
        "revision": revision,
        "safetensors_header_bytes_downloaded": category_bytes.get("safetensors-header-prefix", 0)
        + category_bytes.get("safetensors-header-json", 0),
        "schema": TOOL_SCHEMA,
        "shards": len(shard_rows),
        "source_tree_identity": source_tree_identity,
        "subdir": subdir,
        "tensor_payload_bytes_downloaded": 0,
        "tensors": len(tensors),
        "tool_version": TOOL_VERSION,
    }
    products[pathlib.PurePosixPath("repository.json")] = pretty_json(repository_record)
    products[pathlib.PurePosixPath("source-tree.json")] = pretty_json(source_tree_material)
    products[pathlib.PurePosixPath("source-files.tsv")] = write_tsv(
        ["path", "size", "oid", "lfs_oid", "xet_hash", "admitted_fl2va", "selected_category"],
        source_rows,
    )
    products[pathlib.PurePosixPath("component-summary.json")] = pretty_json(
        {"components": components, "schema": TOOL_SCHEMA}
    )
    products[pathlib.PurePosixPath("tensor-inventory.tsv")] = write_tsv(
        [
            "component",
            "shard",
            "name",
            "dtype",
            "shape",
            "elements",
            "start",
            "end",
            "declared_payload_span",
        ],
        tensor_rows,
    )
    products[pathlib.PurePosixPath("dtype-summary.tsv")] = write_tsv(
        ["component", "dtype", "tensors", "elements", "payload_bytes"], dtype_rows
    )
    products[pathlib.PurePosixPath("shape-summary.tsv")] = write_tsv(
        ["component", "shape", "tensors", "elements", "payload_bytes"], shape_rows
    )
    products[pathlib.PurePosixPath("shard-index-validation.json")] = pretty_json(
        {"components": validation, "schema": TOOL_SCHEMA}
    )
    products[pathlib.PurePosixPath("downloads.tsv")] = write_tsv(
        ["category", "source", "requested_range", "byte_count", "sha256"], downloads
    )
    products[pathlib.PurePosixPath("intake-result.json")] = pretty_json(result)
    return products


def materialize(staging: pathlib.Path, products: dict[pathlib.PurePosixPath, bytes]) -> None:
    for relative, data in sorted(products.items(), key=lambda item: str(item[0])):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)


def compare_products(output: pathlib.Path, staging: pathlib.Path) -> None:
    expected = sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file())
    for relative in expected:
        actual = output / relative
        if not actual.is_file():
            fail(f"check failed: missing output {relative}")
        if actual.read_bytes() != (staging / relative).read_bytes():
            fail(f"check failed: stale output {relative}")


def acquisition_paths(tree: list[dict[str, Any]], subdir: str) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for row in tree:
        if row["type"] != "file":
            continue
        path = row["path"]
        if path.startswith(f"{subdir}/") or path in {"LICENSE", "docs/QA-about-License.md"}:
            admitted.append(row)
    admitted.sort(key=lambda row: row["path"])
    if not admitted or not any(row["path"].endswith(".safetensors") for row in admitted):
        fail("authorized acquisition has no FL2VA safetensors payloads")
    return admitted


def ensure_directory_chain(root: pathlib.Path, relative_parent: pathlib.PurePosixPath) -> None:
    current = root
    if current.exists():
        mode = os.lstat(current).st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            fail(f"source destination is not a real directory: {current}")
    else:
        current.mkdir()
    for part in relative_parent.parts:
        if part in {"", ".", ".."}:
            fail(f"unsafe source destination component {part!r}")
        current = current / part
        if current.exists():
            mode = os.lstat(current).st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                fail(f"source destination component is not a real directory: {part!r}")
        else:
            current.mkdir()


def ensure_absolute_directory_chain(path: pathlib.Path) -> None:
    if not path.is_absolute() or path == pathlib.Path("/"):
        fail("source destination must be an absolute bounded directory")
    current = pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        if part in {"", ".", ".."}:
            fail(f"unsafe source destination component {part!r}")
        current /= part
        if current.exists():
            mode = os.lstat(current).st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                fail(f"source destination component is not a real directory: {part!r}")
        else:
            current.mkdir()


def safe_destination(root: pathlib.Path, source_path: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(source_path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        fail(f"unsafe acquired source path {source_path!r}")
    ensure_directory_chain(root, relative.parent)
    target = root.joinpath(*relative.parts)
    if target.parent.resolve() != root.joinpath(*relative.parent.parts).resolve():
        fail(f"acquired source path escapes destination: {source_path!r}")
    return target


def hash_existing(path: pathlib.Path, expected_maximum: int) -> tuple[Any, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        mode = os.lstat(path).st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            fail(f"refused non-regular partial or source file {path.name!r}")
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > expected_maximum:
                    fail(f"{path.name}: existing file exceeds immutable tree size")
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot verify existing source file {path.name!r}: {exc}")
    return digest, size


def expected_content_sha256(row: dict[str, Any]) -> str:
    oid = row.get("lfs_oid", "")
    if not oid:
        return ""
    if oid.startswith("sha256:"):
        oid = oid[7:]
    if not re.fullmatch(r"[0-9a-f]{64}", oid):
        fail(f"{row['path']}: unsupported LFS content identity")
    return oid


def acquire_one(
    provider: Provider,
    repo: str,
    revision: str,
    row: dict[str, Any],
    output: pathlib.Path,
    check: bool,
) -> tuple[dict[str, Any], int]:
    path = row["path"]
    target = safe_destination(output, path)
    expected_size = row["size"]
    expected_sha256 = expected_content_sha256(row)
    partial = target.with_name(f"{target.name}.part")
    transferred = 0
    if check:
        if partial.exists():
            fail(f"check failed: incomplete partial file {path}.part")
        if not target.exists():
            fail(f"check failed: missing acquired source {path}")
        digest, size = hash_existing(target, expected_size)
    elif target.exists():
        if partial.exists():
            fail(f"{path}: final and partial files both exist")
        digest, size = hash_existing(target, expected_size)
    else:
        if partial.exists():
            digest, size = hash_existing(partial, expected_size)
        else:
            digest, size = hashlib.sha256(), 0
        try:
            with partial.open("ab") as stream:
                transferred = provider.stream_file(
                    repo, revision, path, size, stream, digest, expected_size
                )
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            fail(f"{path}: cannot publish partial source bytes: {exc}")
        size += transferred
        if size != expected_size:
            fail(f"{path}: acquired size does not match immutable tree")
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            fail(f"{path}: acquired SHA-256 does not match LFS identity")
        os.replace(partial, target)
    actual_sha256 = digest.hexdigest()
    if size != expected_size:
        fail(f"{path}: source size mismatch")
    if expected_sha256 and actual_sha256 != expected_sha256:
        fail(f"{path}: source SHA-256 mismatch")
    component = (
        acquisition_component(REQUIRED_SUBDIR, path)
        if path.startswith(f"{REQUIRED_SUBDIR}/")
        else "license"
    )
    return (
        {
            "actual_sha256": actual_sha256,
            "actual_size": size,
            "classification": "shard" if path.endswith(".safetensors") else "metadata",
            "component": component,
            "expected_sha256": expected_sha256,
            "expected_size": expected_size,
            "git_oid": row["oid"],
            "lfs_oid": row["lfs_oid"],
            "path": path,
            "verified": True,
            "xet_hash": row["xet_hash"],
        },
        transferred,
    )


def acquire_source(args: argparse.Namespace) -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        fail(f"authorized acquisition requires {AUTHORIZATION_ENV}=1")
    repo = parse_repo(args.repo)
    if args.subdir != REQUIRED_SUBDIR:
        fail(f"subdirectory must be exactly {REQUIRED_SUBDIR!r}")
    if not COMMIT_RE.fullmatch(args.revision.lower()):
        fail("authorized acquisition requires an explicit immutable full revision")
    output = args.output
    ensure_absolute_directory_chain(output)
    provider: Provider = (
        FixtureProvider(args.fixture_dir) if args.fixture_dir is not None else NetworkProvider(os.environ.get("HF_TOKEN"))
    )
    repository = provider.repository(repo, args.revision)
    revision = require_text(repository.get("sha"), "repository.sha").lower()
    if revision != args.revision.lower():
        fail("authorized acquisition revision did not resolve exactly")
    tree = normalize_tree(provider.tree(repo, revision))
    rows = acquisition_paths(tree, args.subdir)
    files: list[dict[str, Any]] = []
    transferred = 0
    for row in rows:
        record, byte_count = acquire_one(provider, repo, revision, row, output, args.check)
        files.append(record)
        transferred += byte_count
        print(
            f"source acquisition file: {record['path']} bytes={record['actual_size']} "
            f"transferred={byte_count}",
            flush=True,
        )
    source_identity = acquisition_identity(repo, revision, args.subdir, files)
    manifest = {
        "acquisition_complete": True,
        "acquisition_identity": source_identity,
        "admitted_subtree": args.subdir,
        "authorization_gate": AUTHORIZATION_ENV,
        "files": files,
        "metadata_bytes": sum(row["actual_size"] for row in files if row["classification"] == "metadata"),
        "repository": repo,
        "revision": revision,
        "schema": ACQUISITION_SCHEMA,
        "shard_bytes": sum(row["actual_size"] for row in files if row["classification"] == "shard"),
        "shards": sum(row["classification"] == "shard" for row in files),
        "source_bytes": sum(row["actual_size"] for row in files),
    }
    encoded = pretty_json(manifest)
    manifest_path = output / ACQUISITION_MANIFEST
    if args.check:
        if not manifest_path.is_file() or manifest_path.read_bytes() != encoded:
            fail(f"check failed: stale output {ACQUISITION_MANIFEST}")
    else:
        temporary = output / f".{ACQUISITION_MANIFEST}.tmp"
        temporary.write_bytes(encoded)
        os.replace(temporary, manifest_path)
    print(
        f"source acquisition: files={len(files)} shards={manifest['shards']} "
        f"bytes={manifest['source_bytes']} transferred={transferred} identity={source_identity}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face repository ID or HTTPS URL")
    parser.add_argument("--revision", required=True, help="source revision to resolve and pin")
    parser.add_argument("--subdir", required=True, help="must be FL2VA")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="external evidence directory")
    parser.add_argument("--check", action="store_true", help="verify byte-identical existing products")
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="acquire and verify the complete authorized FL2VA source allowlist",
    )
    parser.add_argument("--fixture-dir", type=pathlib.Path, help="use a local offline fixture")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.acquire:
        acquire_source(args)
        return
    output = args.output.resolve()
    if output == pathlib.Path("/"):
        fail("output must not be the filesystem root")
    if args.check and not output.is_dir():
        fail("check requires an existing output directory")
    if not args.check and output.exists():
        fail("output already exists; use a new directory or --check")
    provider: Provider
    if args.fixture_dir is not None:
        provider = FixtureProvider(args.fixture_dir)
    else:
        provider = NetworkProvider(os.environ.get("HF_TOKEN"))
    products = build_products(provider, args.repo, args.revision, args.subdir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.intake-", dir=output.parent))
    try:
        materialize(staging, products)
        if args.check:
            compare_products(output, staging)
        else:
            os.replace(staging, output)
            staging = pathlib.Path()
    finally:
        if staging != pathlib.Path() and staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(sys.argv[1:] if argv is None else argv))
    except IntakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
