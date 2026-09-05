# Model release evidence

Release qualification projects the existing model library, source identities and
artifact records. It does not create another catalog or change runtime admission.
The engineering tool `tools/model_release.py` consumes a fresh
`yvex model list --json` document, explicit full-file integrity receipts and a
reviewed assessment. It writes a new `yvex.model.release.v1` JSON record and
refuses to overwrite an existing record. Generated records and evidence stay
outside Git, under the operator's evidence or release preparation directory.

The record preserves logical identity, immutable upstream repository/revision,
ordered component/shard files, full SHA-256, sizes, tensor/header identities,
transformation evidence, license obligations, exact runtime qualification,
preparation observability and proposed distribution location. The artifact-set
identity hashes a canonical list of component, shard order, size and content
digest. Local paths, proposed repository names, hardware and timestamps do not
participate in that identity. Publication locations remain empty until a later
authorized publication verifies the exact remote bytes.

The assessment schema is `yvex.model.release.assessment.v1`. Required fields are
`logical_identity`, `upstream`, `scope`, `files`, `license`, `lineage`,
`validation`, `observability` and `proposed_repository`. Each file names its
catalog path, proposed release path, component and one-based shard index/count.
Scope is `model`, `composite` or `component`; composite qualification must include
every current catalog component. Alternate quantizations receive separate exact
artifact-set records and can share one proposed distribution repository.

Integrity receipts contain current before/after device, inode, size, mtime and
ctime, full bytes read, SHA-256, bounded format checks and tensor descriptors.
They must come from an explicit full-file verification, with recorded start,
completion and duration. Receipt reuse checks current stat identity, registered
size/digest, format, tensor counts, ranges, shard metadata and exact upstream
metadata. It never treats a profile, directory name or payload-only digest as a
full-file checksum. Receipts are local engineering evidence, not signatures or
protection against a malicious evidence author. Publication must independently
verify the intended bytes; stat equality alone is not remote integrity proof.

Some admitted component formats embed a source snapshot identity instead of a
repository/revision pair. An optional `source_binding` references a checksummed
canonical source inspection report with `source_verified`, `repository`,
`revision` and `source_snapshot_identity`. These must agree with the catalog and
artifact. A missing binding remains `BLOCKED_PROVENANCE`; conflicting provenance
is an error and cannot be overridden by an assessment.

License, lineage and validation gates each carry `PASS` or `BLOCKED`, an explicit
reason when blocked, and checksummed evidence references. Passing gates must name
the exact full-file checksum set and upstream identity. License review records
weight scope, redistribution/derivative terms and obligations. Proven lineage
records input/output identities, producing tool revision and proof; an unrecorded
command stays unrecorded. Runtime evidence identifies YVEX revision/tree, source
stability, machine, hardware, environment, backend, effective configuration,
successful lifecycle and date. Catalog `READY`, a loadable profile, parser
agreement or another artifact's execution cannot satisfy these gates.

The output is `READY_TO_PUBLISH` only when all gates close. Otherwise it retains
every blocker and uses the first blocker as its summary state. Invalid integrity,
source substitutions, missing shards/components or stale evidence reject the
projection itself. Upstream-only sources, fixtures and development alternatives
belong in the qualification triage; they need no artificial public release record.

Observability records download, conversion, quantization and validation duration,
peak working storage and final artifact storage. Missing historical values are
null with a reason. Current checksum or execution time does not fill a historical
preparation timing gap. Qualification neither uploads nor modifies model bytes,
provider caches, runtime sessions or distribution repositories.
