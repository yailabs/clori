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
participate in that identity. Publication locations remain empty until an
authorized publication verifies the exact remote bytes.

Verified publication belongs to the existing artifact registry. Disk schema
`yvex.models.local.v8` adds publication records joining logical identity and an
existing artifact SHA-256/size to provider, repository, immutable Hub commit,
remote filename/digest and public manifest filename/digest. The installed
`yvex_model_publication` record has its own schema version 1; existing registry
entry and catalog summary layouts are unchanged. Readers still accept older
registry schemas. Older writers reject v8 instead of silently discarding remote
locations. Removing the final artifact reference cannot save an orphaned
publication. A catalog join with a different logical identity fails closed.

The CLI projects these facts through `yvex.model.list.v4` and `yvex.model.v4`.
Each representation exposes `local_available` and `remote_locations`; publication
does not change source origin, runtime admission, working-set membership or
artifact identity. Local availability checks a regular file and its recorded
size; it is not a fresh integrity or runtime qualification claim. Remote facts
survive absent local payloads. Eviction and rehydration are separate operations.

`tools/model_release.py --publication RECEIPT.json` requires a qualified release,
a fresh v4 catalog containing matching remote locations, and a checksummed
`yvex.hub.publication.receipt.v1` evidence record. The receipt binds an exact
repository commit, authoritative remote file sizes/SHA-256, public manifest
identities, complete file set, license, card metadata/rendering and independently
verified public visibility. It projects `PUBLISHED` without changing artifact-set
identity. The tool still accepts v3 catalogs for historical qualification; it
does not perform uploads or independently authenticate receipt authors. Hub
operations and public inspection produce the receipt; the projection rejects
disagreement with the canonical catalog or changed evidence files.

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

A reconstructed artifact can instead bind its fresh tensor manifest to a
checksummed `yvex.physical_variant_plan.v1` through `descriptor_authority` and
`descriptors_match_sealed_plan`. The projection checks the current plan's source,
transformation, profile and tensor count against the receipt. Historical offsets
are not an authority for newly emitted bytes. Changed plans or tensor manifests
invalidate the projection.

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
preparation timing gap. The projection neither uploads nor modifies model bytes,
provider caches, runtime sessions or distribution repositories.

Controlled reconstruction can record a new build with
`tools/model_release_build.py --request REQUEST.json --out-dir NEW_DIRECTORY`.
The request declares `stage`, `logical_identity`, `upstream`, exact `argv`,
expected new `outputs`, `storage_root`, `required_free_bytes`, optional
`working_paths` and checksum-bound `input_evidence`. The observer refuses existing
outputs and insufficient space, executes without a shell, preserves raw logs,
measures monotonic duration, hashes new outputs and records executable and source
identity before/after, including the observer and input evidence. Shared source
or input changes invalidate the receipt. Arguments
and command output must contain no credentials; the observer never dumps the
environment. Sampled allocated high-water is explicitly a lower bound, not an
exact peak. The receipt is preparation evidence for a release assessment and
does not grant structural, numerical or runtime qualification.

The DeepSeek family admission owner contains the exact admitted physical release
identities. Profile identity and extent select the expected catalog record;
selection alone never verifies payload bytes. Binding preparation still requires
full-file identity verification and current compiler physical compatibility.
This remains separate from publication eligibility and the local model library.
