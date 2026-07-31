# Documentation Architecture Baseline Audit

Status: frozen point-in-time evidence

Baseline: `51a5c087eafe857d71df1566ce90c2f87a2fcfc1`

Date: 2026-07-31

## Purpose

This audit records the documentation system immediately after canonical
command-architecture closure and before information-architecture migration. It
explains what existed, where authorities conflicted, how every tracked
documentation surface was disposed, which terminology changed, which claims
were preserved, and which non-documentation gaps remain.

It is not a living documentation index, command authority, capability ledger,
or project-control surface.

## Method

Discovery used the exact Git tree at the baseline commit, not the working tree:

```sh
git ls-tree -r --name-only 51a5c087eafe857d71df1566ce90c2f87a2fcfc1
git show 51a5c087eafe857d71df1566ce90c2f87a2fcfc1:PATH
```

The audit classified root entry documents, every tracked Markdown file under
`docs/`, legal material, diagram sources/renderings, frozen audit tables, and
the vector-test README. It inspected headings, internal links, named
authorities, project-state language, command spellings, model-family stages,
release claims, and current code/registry owners.

Dynamic product execution was excluded. The resident daemon was not queried,
restarted, or used for generation. The accepted command registry, protocol v4,
and repository tests were treated as implementation inputs.

## Baseline counts

| Surface | Count |
| --- | ---: |
| Tracked documentation surfaces inventoried | 48 |
| Markdown documents | 33 |
| Existing frozen operator-audit files | 11 |
| Diagram and logo sources/renderings | 9 |
| Root entry/contribution/control/legal surfaces | 8 |
| Flat living `docs/*.md` documents | 10 |
| Canonical glossary | 0 |
| Public changelog | 0 |
| Documentation-governance owner | 0 |
| Machine-readable documentation ownership rows | 0 |

## Principal findings

- README combined public entry, detailed three-terminal operation, doctrine,
  implemented architecture, DeepSeek internals, build policy, and navigation.
- The reference architecture contained current YVEX diagrams and an
  implementation/milestone traceability appendix inside an otherwise
  implementation-independent paper.
- The model-family document mixed the common integration contract, three
  family stages, candidate matrices, implementation facts, and non-claims in
  1,455 lines.
- The runtime contract mixed runtime, source/filesystem, command, graph,
  protocol, telemetry, benchmark, and claim contracts.
- Release doctrine combined stable gate meanings with the exact v0.1 target
  and current readiness narrative.
- `system-target.md` combined current filesystem ownership with a project-like
  gap table and implemented capability narrative.
- Strong command and OpenAI documents were already bound as command-registry
  documentation owners and were kept at their stable paths.
- The two operator milestone contracts embedded live-state transition blocks,
  including `Active Next`, despite `ROADMAP.md` being the sole live authority.
- Frozen audits were accurate point-in-time evidence but lacked a byte-level
  immutability guard.

Detailed findings are in [`findings.md`](findings.md). The complete baseline
mapping is in [`inventory.tsv`](inventory.tsv), terminology migration in
[`terminology.tsv`](terminology.tsv), and unresolved non-documentation facts in
[`gaps.tsv`](gaps.tsv).

## Completeness

Every one of the 48 baseline surfaces has exactly one row in `inventory.tsv`.
Every one of the 33 baseline Markdown documents has a keep, rewrite, move,
split, merge, freeze, or delete disposition. The final public migration record
maps the same paths to current canonical owners.

The 11 existing files under `operator-surface-ec7dcc/` were not modified. The
topology audit moved without content change. Frozen checksums are owned by
`config/frozen_documents.tsv`.

## Preserved claims

The migration preserves, without promotion:

- one complete DeepSeek source-to-streamed-text vertical on CPU and the
  admitted mixed GB10 path;
- one public command executable and one persistent runtime executable;
- private local protocol v4 and bounded OpenAI compatibility profile v1;
- complete artifacts outside Git that are not supported/release artifacts;
- partial performance state with warm optimization still open;
- Qwen and Gemma source/header candidate-role evidence only;
- no model evaluation, full-model release benchmark, or release qualification.

## Result

The repository now has one authority map, one glossary, one evidence doctrine,
separate reference and implemented architecture, separate family records,
separate normative contracts and operator procedures, stable release surfaces,
and repository validation for ownership, links, frozen evidence, terminology,
claims, and project control.
