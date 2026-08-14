---
name: engineering-worklog
description: Create evidence-backed semantic worklogs for material YVEX checkpoints, repairs, performance changes, cutovers, and closures. Use before meaningful final handoffs or when asked for a worklog or communication seed; skip trivial or generated-only changes.
---

# Engineering Worklog

Capture the engineering delta that future maintainers and communication work need without
turning a Git diff into a narrative or inventing evidence. Treat
[`AGENTS.md`](../../../AGENTS.md) and the repository evidence doctrine as higher-level
constraints.

## Classify the trigger

Choose exactly one primary type:

- `checkpoint`: a meaningful evidenced capability or intermediate state became real;
- `repair`: a significant defect or false assumption was understood and corrected;
- `performance`: a causal improvement was measured under comparable conditions;
- `closure`: an accepted wave or milestone boundary closed.

Do not generate a worklog for formatting-only edits, typo fixes, mechanical renames,
generated-file churn, insignificant housekeeping, or trivial test maintenance. State
`Engineering worklog: not required (trivial change)` in the handoff instead.

## Establish the evidence boundary

1. Read the current repository instructions and evidence doctrine.
2. Resolve the actual branch, `HEAD`, baseline or merge base, commit range, working-tree diff,
   and staged diff. Never assume a previously reported SHA.
3. Identify the semantic owner, before-state, concrete problem, causal evidence, decision,
   implementation invariant, after-state, and remaining limitations.
4. Inspect the tests, measurements, traces, review findings, and operator observations already
   produced by the delivery. Do not rerun expensive GPU work solely for a worklog.
5. Label missing proof as `evidence gap`; do not smooth uncertainty into a claim.
6. Treat Git paths, line counts, and commit statistics as secondary evidence only.

For performance claims, require comparable identity, workload, hardware, and method. Mark each
quantitative comparison `directly comparable`, `approximately comparable`, or
`characterization only`. If comparability is absent, report observations without a delta claim.

## Choose draft or durable output

Write intermediate material under `build/worklog/`. The build tree is ignored and is not an
authority.

Promote a record to `docs/worklog/YYYY-MM-DD-<short-semantic-name>.md` only when all of these
are true:

- it represents a material trigger with durable engineering value;
- its baseline and checkpoint are stable commit identities;
- its semantic claims are supported or explicitly marked as gaps;
- its remaining limitations are explicit;
- it has been intentionally selected for repository retention.

Use the event date, lowercase hyphenated words, and a concise semantic name. Do not bulk-create
historical records. A durable record is documentary history, never runtime, compiler, project-
control, benchmark, evaluation, or release authority.

## Write the canonical record

Use Markdown with one H1 title followed by this metadata table in this order:

```markdown
| Field | Value |
| --- | --- |
| Date | YYYY-MM-DD |
| Type | checkpoint / repair / performance / closure |
| Milestone | stable ID or `not-applicable` |
| Branch | branch that produced the change |
| Baseline | full commit SHA |
| Checkpoint | full commit SHA |
| Subsystem | semantic owner or bounded owner set |
| Model family | family name or `not-applicable` |
| Hardware | exact relevant hardware or `not-applicable` |
| Evidence | lowest truthful evidence classes |
| Comparability | directly comparable / approximately comparable / characterization only / not-applicable |
| Publishability | private-draft / reviewed / public-safe |
```

Then write these required sections:

1. `## Before` — behavior, ownership, measurements, limitations, and assumptions that were
   actually true.
2. `## Problem` — the concrete correctness, architecture, performance, memory, UX, or evidence
   deficiency.
3. `## Causal analysis` — what the available evidence establishes; state uncertainty.
4. `## Decision` — the chosen owner and invariant, plus materially relevant rejected
   alternatives.
5. `## Implementation` — the semantic change; files and commits are secondary evidence.
6. `## After` — the new admitted truth, without promoting a higher capability.
7. `## Evidence` — tests, oracles, sanitizers, profiles, measurements, identities, and gaps.
8. `## Remaining limitations` — explicit non-claims and later owners.
9. `## Why it matters` — one short technical consequence without promotion.

Add `## Quantitative delta` only when numerical facts exist. Do not fabricate an empty table.

## Create communication projections

When useful, append `## Communication projections` with:

- `### Short update`: concise mechanism plus evidence, without hype;
- `### Longer post seed`: a short ordered explanation, not a complete marketing article;
- `### Article seed`: title, thesis, three to six sections, and strongest evidence;
- `### Visual candidates`: concrete tables, diagrams, or plots worth producing later;
- `### Quoteable technical facts`: a small set copied or faithfully projected from the
  canonical record.

Every projection must be derivable from the canonical sections above it. Do not introduce a
number, conclusion, or capability only in a projection. Do not generate or publish visuals,
posts, articles, or social content automatically.

## Apply authorship and sensitivity discipline

Prefer project-centered language when agency is ambiguous: `YVEX changed...`, not an
unsupported claim that one person wrote every implementation detail. Attribute an owner when
the record actually establishes that decision.

Use these publication states:

- `private-draft`: incomplete review or sensitive local detail remains;
- `reviewed`: technical content was reviewed, but public-safety review is incomplete;
- `public-safe`: no secrets, credentials, private URLs, inappropriate host/path details,
  personal or client data, unreleased third-party information, or unsupported claims remain.

Publication state is metadata only. It never triggers publication. Normalize irrelevant local
paths in public projections. Refuse `public-safe` while an evidence gap affects a headline
claim.

## Validate and hand off

For a durable record:

1. add it to `config/documentation_owners.tsv` as class `worklog`, retained lifecycle;
2. link the worklog collection from `docs/README.md` without copying its facts;
3. run the skill validator and documentation/project-control guards required by
   [`docs/development/documentation-policy.md`](../../../docs/development/documentation-policy.md);
4. inspect the final diff for secrets, unsupported claims, and projection-only facts.

Include this compact block in the final handoff:

```text
Engineering worklog
generated: yes/no
path: path or not-applicable
trigger: checkpoint/repair/performance/closure/not-applicable
publishability: private-draft/reviewed/public-safe/not-applicable
strongest_delta: one evidence-backed before/after fact or not-applicable
```

Do not paste the full record into the handoff unless requested.
