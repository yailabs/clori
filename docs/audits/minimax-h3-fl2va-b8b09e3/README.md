# MiniMax-H3 FL2VA Intake Audit

Status: frozen at research intake closure

Main baseline: `69181458cf38cb7455f2ad73d7139fd57e3aa25b`

Source revision: `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08`

This audit binds the bounded source, component, tensor-header, architecture,
operation-gap, and GB10 feasibility evidence used by the
[MiniMax-H3 family record](../../model-families/minimax-h3.md). It is
point-in-time research evidence, not product capability, project state, legal
advice, or release authority.

## Scope and method

The source is `MiniMaxAI/MiniMax-H3` at the immutable revision above. The
working subgraph is exactly `FL2VA/`, initially for `t2va`. `Ref2VA/`, root
hosted-system components, context/regeneration/2K variants, reference
checkpoints, media assets, and every non-FL2VA weight are excluded.

`tools/minimax_h3_intake.py` used one serial HTTPS acquisition stream. It
enumerated source-bound Hugging Face tree metadata, downloaded selected JSON,
tokenizer/processor text metadata, license files, and safetensors indexes, then
read each safetensors length prefix and JSON header with exact HTTP Range
requests. It did not fetch tensor payload bytes. Redirects were limited to
recognized Hugging Face delivery domains and credentials, when present, were
not forwarded to delivery hosts.

The external evidence root used at execution was
`/home/dgmothx/lab/models/intake/minimax-h3/<FULL_REVISION>/`. The absolute
path has no identity role. Canonical identity material contains repository,
immutable revision, `FL2VA`, normalized source tree, selected metadata hashes,
component summaries, and tensor descriptors.

## Identities

| Evidence | SHA-256 or identity |
| --- | --- |
| Normalized source tree | `91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e` |
| Aggregate external intake | `1e7db0167eafc1e43ecaad37897198cd54f838398907b686211886dbc288b662` |
| External `source-tree.json` | `d25aff4f8e75f7bc047697850022273ada8d8910bef06a1843ebfdd352ba9e5a` |
| External `source-files.tsv` | `a7cf175b5f93e8af3556f7d9b318d2aa03023bc85a0ff892c0571c7ce5893f3f` |
| External `component-summary.json` | `65258bb2997e84396267128b156d3c01c68b0b7756280b1891b922ca618e9c6f` |
| External `dtype-summary.tsv` | `0d6a245c976e3586d14520e11dcc66961029f0c80fc18e3fb5eca6cbcb579a44` |
| External `shape-summary.tsv` | `c031257ce2162be35476e1170ef1a05b43a5f76a831593b56c9fe7078551ab83` |
| External shard-index validation | `20bc8c004fa2768537edfab1352936834edaf661318302f6fe3104540ffea2d4` |
| External intake result | `a440814d75cd3c321e249f938f3c46168c5d410291f65f62692d9048c65af814` |
| Source `LICENSE` | `59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44` |
| Source license Q&A | `26900a2868636f886d241efe94002ad11d858b3c3eb005e58c0c2ad60f0de7ae` |

The repository declares license identifier `other`. The license and Q&A facts
are recorded in the family record without an eligibility conclusion.

## Acquisition result

- 280 repository files observed; 81 FL2VA files admitted.
- 34,718,145 metadata/config/license/index bytes downloaded.
- 375,696 safetensors prefix/header bytes downloaded.
- 0 tensor payload bytes downloaded.
- 29 shards reconciled: 14 text encoder, 13 Omni-Transformer, one visual VAE,
  and one audio VAE.
- 3,240 unique component-qualified tensors reconciled to their headers and
  shard indexes with no missing, stale, duplicate, malformed, overlapping, or
  unknown-dtype entry.
- 69,235,580,593 parameter elements and 144,016,000,740 declared payload bytes.

The complete untracked tensor inventory remains in the external evidence
root. [`components.tsv`](components.tsv) retains bounded component totals and
contracts; it does not copy per-tensor rows. [`operation-gap.tsv`](operation-gap.tsv)
records all 46 inspected operation boundaries and the exact YVEX owner used
for comparison. [`metrics.json`](metrics.json) retains counts, hashes,
memory formulas, assumptions, alternative scores, and reproduction commands.

## Findings and decisions

Simultaneous original-dtype residency is infeasible because the source tensor
payload alone exceeds 128 GB. Staged original-dtype residency is
conditionally feasible in a 90.8–111.1 GB modeled peak only with exact phase
release, memory-efficient attention, VAE tiling, and request-geometry budgets.
Staging plus numerically proven selective transforms models a 59–84 GB peak.
Neither conditional result is runtime evidence or a speed claim.

The gap inventory classifies 2 operations as available generic, 14 as bounded
extensions, 22 as new generic primitives, 3 as backend-fusion candidates, 4
as family composition, and the pipeline-owned iteration schedule as unknown.
Full attention is a correctness
and capacity blocker: existing YVEX attention is row-wise F32 with explicit
scratch, while H3 needs batched full-sequence attention whose naive BF16
score/probability matrices exceed memory at an illustrative 768p geometry.

Alternative B is conditionally selected: one composite model target binds
identity-bearing component artifacts, a typed phase DAG, residency plan, and
output contract. It is not one composite artifact. The common runtime,
artifact, graph, and backend ownership boundaries remain unchanged.

The selected next slice is
`R010.MINIMAX.H3.FL2VA.IR.0`: full source-to-architecture/Transformation IR
without numerical execution. It scores highest because it can close exact
component/role/identity ownership before introducing attention, convolution,
scheduler, residency, or media abstractions.

## Reproduction

Use an arbitrary external root; its path does not enter the identity:

```sh
python3 tools/minimax_h3_intake.py \
  --repo MiniMaxAI/MiniMax-H3 \
  --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 \
  --subdir FL2VA \
  --output <EXTERNAL_ROOT>/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08

python3 tools/minimax_h3_intake.py \
  --repo MiniMaxAI/MiniMax-H3 \
  --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 \
  --subdir FL2VA \
  --output <EXTERNAL_ROOT>/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 \
  --check

PYTHONDONTWRITEBYTECODE=1 python3 tests/test_minimax_h3_intake.py
```

The live generation and the second live `--check` pass completed
byte-identically. Offline fixture generation completed twice independently
with byte-identical output and all required refusal cases.

## Frozen sibling hashes

| File | SHA-256 |
| --- | --- |
| `metrics.json` | `2da54590d20e4c11ac0fa8f27209d82e32421ba4e3a58273895eb8338f74b97d` |
| `components.tsv` | `03befe64018273c10743f2e4a4c1edd39e20b305b93cf274a4b8b642530a7f3e` |
| `operation-gap.tsv` | `8c9e189e4a9b5a2fd82e9f307165772224d01f8ddf9202b8e995578f20c2281a` |

## Progression and non-claims

`progression_decision: proceed`

`downstream_safe: true`

The selected source/IR consumer does not consume unknown numerical schedule,
attention, VAE, media, or runtime behavior. Those are deferred depth with
explicit later executable consumers, not capability implied by this audit.
There are no intake gate blockers, boundary incompletenesses, or local
external blockers. Legal authorization remains outside engineering evidence.

This audit does not prove MiniMax-H3 artifact emission, materialization,
runtime execution, residency, audio/video generation, synchronized output,
Diffusers/SGLang/vLLM parity, model quality, speed, 768p practicality, 2K,
Ref2VA, H3-Context-IR, H3-Regenerate-2K, commercial or redistribution
eligibility, evaluation, benchmark, release support, or a second complete
model-family vertical.
