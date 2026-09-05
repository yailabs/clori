#!/usr/bin/env python3
"""Reject stale bytes, source substitutions, incomplete sets and borrowed validation."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import sys
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location("model_release", Path(__file__).parents[1] / "tools/model_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseProjection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.path.write_bytes(b"bounded artifact fixture")
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        st = self.path.stat()
        stat = {k: getattr(st, k) for k in release.STAT_FIELDS}
        self.source = {"provider": "huggingface", "repository": "org/model", "revision": "a" * 40}
        self.catalog = {"schema": "yvex.model.list.v3", "models": [{
            "identity": "family:fixture/model:test", "name": "test", "family": "fixture",
            "sources": [self.source], "components": [], "representations": [{
                "path": str(self.path), "identity": sha, "format": "gguf", "size_bytes": st.st_size,
                "tensor_count": 1}]}]}
        self.receipts = [{"path": str(self.path), "sha256": sha, "status": "PASS", "stable": True,
            "stat_before": stat, "stat_after": stat, "bytes_read": st.st_size, "tensor_count": 1,
            "types": {"F32": 1}, "bounds_valid": True, "unique_tensor_names": True, "nonoverlap": True,
            "descriptors_equal_to_forensic": True, "format": "GGUF", "shard_count": 1,
            "allocated_size": st.st_blocks * 512, "header_sha256": "b" * 64,
            "tensor_descriptor_sha256": "c" * 64, "tensor_manifest_sha256": "d" * 64,
            "metadata": {"general.source.huggingface.repository": "org/model", "yvex.source.revision": "a" * 40},
            "started_at": "2026-09-05T00:00:00Z", "completed_at": "2026-09-05T00:00:01Z",
            "duration_seconds": 1}]
        self.evidence = Path(self.tmp.name) / "evidence.json"
        self.evidence.write_text('{"scope":"software fixture only"}')
        ref = {"path": str(self.evidence), "sha256": hashlib.sha256(self.evidence.read_bytes()).hexdigest()}
        blocked = {"status": "BLOCKED", "reason": "Historical proof absent", "evidence": [ref]}
        self.assessment = {"schema": "yvex.model.release.assessment.v1",
            "logical_identity": "family:fixture/model:test", "upstream": self.source,
            "scope": "model", "proposed_repository": "yailabs/test-GGUF", "observability": {},
            "files": [{"path": str(self.path), "release_path": "model.gguf", "component": "model",
                       "shard_index": 1, "shard_count": 1}],
            "license": copy.deepcopy(blocked), "lineage": copy.deepcopy(blocked),
            "validation": copy.deepcopy(blocked)}

    def project(self):
        return release.project(self.catalog, self.assessment, self.receipts)

    def test_blockers_survive_and_identity_ignores_distribution_name(self):
        first = self.project()
        self.assessment["proposed_repository"] = "yailabs/different-name"
        self.assessment["files"][0]["release_path"] = "renamed.gguf"
        second = self.project()
        self.assertEqual(first["artifact_set_identity"], second["artifact_set_identity"])
        self.assertEqual(len(first["publication"]["blockers"]), 3)
        self.assertEqual(first["publication"]["remote_locations"], [])

    def test_changed_same_length_bytes_rejected(self):
        self.path.write_bytes(b"x" * self.path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.project()

    def test_wrong_upstream_revision_rejected(self):
        self.receipts[0]["metadata"]["yvex.source.revision"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            self.project()

    def test_catalog_checksum_disagreement_rejected(self):
        self.catalog["models"][0]["representations"][0]["identity"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.project()

    def test_missing_shard_rejected(self):
        self.assessment["files"][0]["shard_count"] = 2
        self.receipts[0]["shard_count"] = 2
        self.receipts[0]["metadata"]["split.no"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete or duplicate shard"):
            self.project()

    def test_wrong_shard_header_rejected(self):
        self.assessment["files"][0]["shard_count"] = 2
        self.receipts[0]["shard_count"] = 2
        self.receipts[0]["metadata"]["split.no"] = 1
        with self.assertRaisesRegex(ValueError, "header shard index"):
            self.project()

    def test_composite_requires_every_current_component(self):
        self.assessment["scope"] = "composite"
        self.catalog["models"][0]["components"] = [{"path": str(self.path)}, {"path": "/missing/component.gguf"}]
        with self.assertRaisesRegex(ValueError, "omits or substitutes"):
            self.project()

    def test_borrowed_validation_rejected(self):
        gate = self.assessment["validation"]
        gate.update(status="PASS", artifact_sha256=["0" * 64], upstream=self.source)
        with self.assertRaisesRegex(ValueError, "other artifact bytes"):
            self.project()

    def test_changed_evidence_rejected(self):
        self.evidence.write_text('{"different":"evidence"}')
        with self.assertRaisesRegex(ValueError, "changed evidence"):
            self.project()

    def test_unsafe_filename_rejected(self):
        self.assessment["files"][0]["release_path"] = "../model.gguf"
        with self.assertRaisesRegex(ValueError, "unsafe release"):
            self.project()

    def test_duplicate_json_authority_rejected(self):
        self.evidence.write_text('{"status":"BLOCKED","status":"PASS"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            release.read_json(self.evidence)

    def test_structural_corruption_rejected(self):
        self.receipts[0]["bounds_valid"] = False
        with self.assertRaisesRegex(ValueError, "invalid tensor structure"):
            self.project()

    def use_current_plan(self):
        plan = Path(self.tmp.name) / "physical.plan"
        manifest = Path(self.tmp.name) / "tensors.csv"
        manifest.write_text("name,dtype\nweight,F32\n")
        identities = {"profile_identity": "1" * 64,
                      "required_payload_identity": "2" * 64,
                      "transform_identity": "3" * 64}
        plan.write_text("yvex.physical_variant_plan.v1\nplan_schema=2\ndecision_count=1\n"
                        + "".join(f"{key}={value}\n" for key, value in identities.items()))
        receipt = self.receipts[0]
        receipt.update(descriptors_equal_to_forensic=False, descriptors_match_sealed_plan=True,
                       descriptor_authority={"kind": "yvex.physical_variant_plan.v1",
                           "path": str(plan), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
                       fresh_tensor_manifest=str(manifest),
                       tensor_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        receipt["metadata"].update({"yvex.quant.profile.identity": identities["profile_identity"],
                                    "yvex.source.payload.identity": identities["required_payload_identity"],
                                    "yvex.transform.identity": identities["transform_identity"]})
        return plan

    def test_new_artifact_uses_current_plan_not_historical_offsets(self):
        self.use_current_plan()
        self.assertEqual(self.project()["files"][0]["sha256"], self.receipts[0]["sha256"])

    def test_changed_descriptor_plan_rejected(self):
        plan = self.use_current_plan()
        plan.write_text(plan.read_text().replace("decision_count=1", "decision_count=2"))
        with self.assertRaisesRegex(ValueError, "changed evidence"):
            self.project()

    def test_current_plan_cannot_bind_other_transformation(self):
        self.use_current_plan()
        self.receipts[0]["metadata"]["yvex.transform.identity"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "plan identity mismatch"):
            self.project()

    def test_current_plan_requires_actual_descriptor_match(self):
        self.use_current_plan()
        self.receipts[0]["descriptors_match_sealed_plan"] = False
        with self.assertRaisesRegex(ValueError, "sealed-plan descriptor"):
            self.project()

    def test_unknown_embedded_source_stays_blocked(self):
        self.receipts[0]["metadata"] = {}
        result = self.project()
        self.assertIn("BLOCKED_PROVENANCE", [b["code"] for b in result["publication"]["blockers"]])

    def test_source_snapshot_binding_rejects_another_snapshot(self):
        self.receipts[0]["metadata"] = {"yvex.source.snapshot.identity": "1" * 64}
        binding = Path(self.tmp.name) / "source.json"
        binding.write_text(json.dumps({"source_verified": True, "repository": "org/model",
                                      "revision": "a" * 40, "source_snapshot_identity": "2" * 64}))
        self.assessment["source_binding"] = {
            "path": str(binding), "sha256": hashlib.sha256(binding.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "snapshot binding mismatch"):
            self.project()

    def test_ready_requires_all_exact_gates(self):
        for name in ("license", "lineage", "validation"):
            self.assessment[name].update(status="PASS", artifact_sha256=[self.receipts[0]["sha256"]],
                                         upstream=self.source)
        self.assessment["license"].update(license="MIT", weight_scope="explicit weights",
                                           redistribution="permitted", derivatives="permitted",
                                           obligations=["include license"])
        self.assessment["lineage"]["stages"] = [{"input_identity": "a" * 64,
            "output_identity": self.receipts[0]["sha256"], "tool": "fixture", "tool_revision": "b" * 40,
            "proof": "software fixture; not real model qualification"}]
        self.assessment["validation"].update(yvex_revision="c" * 40, source_tree="d" * 40,
            source_stable=True, machine="fixture", hardware="CPU", environment={"OS": "fixture"},
            backend="cpu", configuration={"tokens": 1}, lifecycle=["open", "generate", "close"],
            date="2026-09-05T00:00:00Z")
        self.assertEqual(self.project()["publication"]["status"], "READY_TO_PUBLISH")
        self.assessment["validation"]["source_stable"] = False
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.project()


build_spec = importlib.util.spec_from_file_location("model_release_build", Path(__file__).parents[1] / "tools/model_release_build.py")
build = importlib.util.module_from_spec(build_spec)
build_spec.loader.exec_module(build)


class ObservedBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / 'new.gguf'
        self.request = {'stage': 'fixture', 'logical_identity': 'fixture', 'upstream': {},
            'argv': [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b"new bytes")', str(self.output)],
            'outputs': [str(self.output)], 'storage_root': str(self.root), 'required_free_bytes': 0}

    def run_build(self, states=None):
        with patch.object(build, 'source_state', side_effect=states or [
                {'tracked_contents_sha256': 'a'}, {'tracked_contents_sha256': 'a'}]):
            return build.observe(self.request, self.root / 'evidence', self.root, .01)

    def test_observed_output_and_real_failure(self):
        result = self.run_build()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['outputs'][0]['sha256'], hashlib.sha256(b'new bytes').hexdigest())
        self.assertGreater(result['duration_seconds'], 0)
        self.assertIsNone(result['peak_working_storage_bytes'])

    def test_existing_artifact_preserved(self):
        self.output.write_bytes(b'previous')
        with self.assertRaisesRegex(ValueError, 'existing output'):
            self.run_build()
        self.assertEqual(self.output.read_bytes(), b'previous')

    def test_space_refusal_precedes_execution(self):
        self.request['required_free_bytes'] = 2**100
        with self.assertRaisesRegex(ValueError, 'free-space'):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_failed_process_cannot_produce_pass(self):
        self.request['argv'] = [sys.executable, '-c', 'raise SystemExit(9)']
        result = self.run_build()
        self.assertEqual(result['exit_code'], 9)
        self.assertEqual(result['status'], 'BLOCKED')

    def test_changed_source_invalidates_build(self):
        result = self.run_build([{'tracked_contents_sha256': 'a'}, {'tracked_contents_sha256': 'b'}])
        self.assertEqual(result['status'], 'BLOCKED')

    def test_input_evidence_tampering_prevents_execution(self):
        evidence = self.root / 'input.json'
        evidence.write_text('{}')
        self.request['input_evidence'] = [{'path': str(evidence), 'sha256': '0' * 64}]
        with self.assertRaisesRegex(ValueError, 'input evidence'):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_input_changed_during_build_invalidates_output(self):
        evidence = self.root / 'input.json'
        evidence.write_text('{}')
        self.request['input_evidence'] = [{'path': str(evidence),
            'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}]
        self.request['argv'] = [sys.executable, '-c',
            'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("changed"); '
            'pathlib.Path(sys.argv[2]).write_bytes(b"new bytes")', str(evidence), str(self.output)]
        result = self.run_build()
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertFalse(result['inputs_stable'])


if __name__ == "__main__":
    unittest.main()
