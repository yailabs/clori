#!/usr/bin/env python3
"""Project catalog facts and reviewed evidence into an immutable release record.

This offline engineering tool neither registers models nor publishes files.
Integrity receipts come from explicit full-file verification, never catalog READY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


SCHEMA = "yvex.model.release.v1"
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def evidence_files(records):
    require(isinstance(records, list) and records, "missing evidence references")
    for record in records:
        path = Path(record["path"])
        require(HEX256.fullmatch(record["sha256"]), "invalid evidence checksum")
        require(path.stat().st_size <= 32 * 1024 * 1024, "evidence must be bounded metadata")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                f"changed evidence: {path}")


def select_model(catalog, assessment):
    require(catalog.get("schema") == "yvex.model.list.v3", "unsupported catalog schema")
    matches = [m for m in catalog["models"] if m["identity"] == assessment["logical_identity"]]
    require(len(matches) == 1, "logical identity missing or ambiguous in catalog")
    model = matches[0]
    source = assessment["upstream"]
    require(REVISION.fullmatch(source["revision"]), "upstream revision must be immutable")
    keys = ("provider", "repository", "revision")
    matches = [s for s in model["sources"] if all(s[k] == source[k] for k in keys)]
    require(len(matches) == 1, "wrong or ambiguous upstream source for logical model")
    return model, {k: matches[0][k] for k in keys}


def release_path(value):
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts
            and str(path) == value and "\\" not in value, "unsafe release filename")
    return value


def project_file(item, model, receipts, upstream, source_binding=None):
    matches = [r for r in model["representations"] if r["path"] == item["path"]]
    require(len(matches) == 1, "file is not an unambiguous catalog representation")
    registered = matches[0]
    matches = [r for r in receipts if r["path"] == item["path"]]
    require(len(matches) == 1, "missing or duplicate integrity receipt")
    receipt = matches[0]
    require(receipt["status"] == "PASS" and receipt["stable"] is True, "failed integrity receipt")
    sha = receipt["sha256"]
    require(HEX256.fullmatch(sha), "invalid artifact checksum")
    before, after = receipt["stat_before"], receipt["stat_after"]
    st = Path(item["path"]).stat()
    current = {key: getattr(st, key) for key in STAT_FIELDS}
    require(before == after == current, "artifact changed after full verification")
    require(receipt["bytes_read"] == st.st_size > 0, "incomplete full-file checksum")
    require(registered["size_bytes"] == st.st_size, "catalog size mismatch")
    require(not registered["identity"] or registered["identity"] == sha, "catalog digest mismatch")
    require(registered["format"].lower() == receipt["format"].lower(), "catalog format mismatch")
    require(receipt["tensor_count"] == registered["tensor_count"] > 0, "catalog tensor mismatch")
    require(sum(receipt["types"].values()) == receipt["tensor_count"], "incomplete tensor types")
    for key in ("bounds_valid", "unique_tensor_names", "nonoverlap", "descriptors_equal_to_forensic"):
        require(receipt[key] is True, f"invalid tensor structure: {key}")
    metadata = receipt["metadata"]
    repositories = [metadata[k] for k in ("general.source.huggingface.repository", "general.source.repository")
                    if k in metadata]
    revisions = [metadata[k] for k in ("yvex.source.revision", "general.source.revision") if k in metadata]
    require(all(value == upstream["repository"] for value in repositories), "artifact upstream repository mismatch")
    require(all(value == upstream["revision"] for value in revisions), "artifact upstream revision mismatch")
    upstream_link = "EXACT_EMBEDDED" if repositories and revisions else "UNRESOLVED"
    if source_binding:
        evidence_files([source_binding])
        bound = read_json(source_binding["path"])
        require(bound.get("source_verified") is True, "source binding is not verified")
        require(bound.get("repository") == upstream["repository"]
                and bound.get("revision") == upstream["revision"], "source binding upstream mismatch")
        snapshot = metadata.get("yvex.source.snapshot.identity")
        require(snapshot and snapshot == bound.get("source_snapshot_identity"), "source snapshot binding mismatch")
        upstream_link = "EXACT_SOURCE_BINDING"
    count, index = item["shard_count"], item["shard_index"]
    require(type(count) is int and type(index) is int and 1 <= index <= count, "invalid shard ordering")
    require(receipt["shard_count"] == count, "header shard count mismatch")
    if count > 1:
        require(metadata.get("split.no") == index - 1, "header shard index mismatch")
    for key in ("header_sha256", "tensor_descriptor_sha256", "tensor_manifest_sha256"):
        require(HEX256.fullmatch(receipt[key]), f"missing tensor identity: {key}")
    return {
        "sha256": sha, "filename": release_path(item["release_path"]),
        "size_bytes": st.st_size, "allocated_size": receipt["allocated_size"],
        "format": receipt["format"], "component": item["component"],
        "shard_index": index, "shard_count": count,
        "tensor_count": receipt["tensor_count"], "tensor_types": receipt["types"],
        "header_sha256": receipt["header_sha256"],
        "tensor_descriptor_sha256": receipt["tensor_descriptor_sha256"],
        "tensor_manifest_sha256": receipt["tensor_manifest_sha256"],
        "representation_metadata": metadata,
        "upstream_link": upstream_link,
        "locations": [{"kind": "local", "path": item["path"]}],
        "integrity": {k: receipt[k] for k in ("started_at", "completed_at", "duration_seconds")},
    }


def check_set(files, model, scope):
    require(files, "empty artifact set")
    require(len({f["filename"] for f in files}) == len(files), "duplicate release filename")
    require(len({f["sha256"] for f in files}) == len(files), "duplicate physical release bytes")
    for component in {f["component"] for f in files}:
        shards = [f for f in files if f["component"] == component]
        counts = {f["shard_count"] for f in shards}
        require(len(counts) == 1, "inconsistent shard counts")
        count = next(iter(counts))
        require(sorted(f["shard_index"] for f in shards) == list(range(1, count + 1)),
                "incomplete or duplicate shard set")
    require(scope in ("model", "composite", "component"), "unknown release scope")
    if scope == "composite":
        expected = {c["path"] for c in model["components"]}
        actual = {f["locations"][0]["path"] for f in files}
        require(expected and expected == actual, "composite omits or substitutes catalog component")
    elif scope == "model":
        require(not model["components"], "composite must declare its component scope")


def qualify_gate(name, gate, hashes, upstream):
    require(gate.get("status") in ("PASS", "BLOCKED"), f"invalid {name} status")
    evidence_files(gate["evidence"])
    if gate["status"] == "BLOCKED":
        require(gate.get("reason"), f"unexplained {name} blocker")
        return {"code": "BLOCKED_" + name.upper(), "reason": gate["reason"]}
    require(sorted(gate.get("artifact_sha256", [])) == hashes, f"{name} refers to other artifact bytes")
    require(gate.get("upstream") == upstream, f"{name} refers to another source")
    if name == "license":
        for key in ("license", "weight_scope", "redistribution", "derivatives", "obligations"):
            require(gate.get(key), f"license review missing {key}")
    elif name == "lineage":
        require(gate.get("stages"), "no proven transformation stages")
        for stage in gate["stages"]:
            for key in ("input_identity", "output_identity", "tool", "tool_revision", "proof"):
                require(stage.get(key), f"lineage missing {key}")
            require(REVISION.fullmatch(stage["tool_revision"]), "unresolved producing tool revision")
    else:
        for key in ("yvex_revision", "source_tree", "machine", "hardware", "environment",
                    "backend", "configuration", "lifecycle", "date"):
            require(gate.get(key), f"exact runtime validation missing {key}")
        require(REVISION.fullmatch(gate["yvex_revision"]), "unresolved validation revision")
        require(gate.get("source_stable") is True, "validation source changed during execution")
    return None


def project(catalog, assessment, receipts):
    require(assessment.get("schema") == "yvex.model.release.assessment.v1", "unsupported assessment schema")
    model, upstream = select_model(catalog, assessment)
    files = sorted([project_file(f, model, receipts, upstream, assessment.get("source_binding"))
                    for f in assessment["files"]],
                   key=lambda f: f["filename"])
    check_set(files, model, assessment["scope"])
    hashes = sorted(f["sha256"] for f in files)
    blockers = []
    for name in ("license", "lineage", "validation"):
        blocker = qualify_gate(name, assessment[name], hashes, upstream)
        if blocker:
            blockers.append(blocker)
    if any(f["upstream_link"] == "UNRESOLVED" for f in files):
        blockers.append({"code": "BLOCKED_PROVENANCE", "reason":
                         "Embedded snapshot identity lacks a proven binding to the catalog upstream revision"})
    repository = assessment["proposed_repository"]
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository), "invalid proposed repository")
    identity = [{k: f[k] for k in ("sha256", "size_bytes", "component", "shard_index", "shard_count")}
                for f in files]
    identity.sort(key=lambda f: (f["component"], f["shard_index"]))
    return {
        "schema": SCHEMA, "artifact_set_identity": "sha256:" + digest(identity),
        "logical_model": {k: model[k] for k in ("identity", "name", "family")},
        "upstream": upstream, "scope": assessment["scope"], "files": files,
        "source_binding": assessment.get("source_binding"),
        "license": assessment["license"], "lineage": assessment["lineage"],
        "validation": assessment["validation"], "observability": assessment["observability"],
        "publication": {"proposed_repository": repository, "remote_locations": [],
                        "status": blockers[0]["code"] if blockers else "READY_TO_PUBLISH",
                        "blockers": blockers},
        "projection": {"catalog_schema": catalog["schema"], "catalog_sha256": digest(catalog),
                       "assessment_sha256": digest(assessment), "integrity_sha256": digest(receipts)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="fresh model list --json projection")
    parser.add_argument("--assessment", required=True, help="reviewed evidence with exact subject identities")
    parser.add_argument("--integrity", required=True, help="explicit full-file verification receipts")
    parser.add_argument("--output", required=True, help="new immutable JSON report; must not exist")
    args = parser.parse_args()
    try:
        result = project(read_json(args.catalog), read_json(args.assessment), read_json(args.integrity))
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"release projection: {exc}\n")
    print(result["publication"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
