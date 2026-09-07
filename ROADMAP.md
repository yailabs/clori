# YVEX Roadmap

Status: living public project control
Release target: v0.1.0

This file is the sole live authority for macro progression, accepted project
boundaries, release-gate state, and the next engineering action. Technical
contracts belong to their code and documentation owners; implementation
chronology belongs to Git, issues, and pull requests.

## Accepted foundation

YVEX has one native compiler/runtime and persistent local host. Source
provenance, artifacts, deployment compatibility, engine generations, typed
multipart content, directional capabilities, model leases, transactional
session state, resource accounting, and execution measurement have common
owners. Multiple fitting engines can coexist without replacing a primary
conversation. Cooperative scheduling does not imply continuous batching.

The first integrated reassessment closes two generic ownership boundaries:
source-declared logical model relations replace catalog family exceptions,
and a private platform adapter isolates terminal mechanics from client
semantics. Linux execution is qualified; portable interfaces do not establish
Windows or macOS execution. The compiler/state/backend substrate is retained
where its existing ownership is sound.

The accepted family boundaries are deliberately unequal:

- DeepSeek reaches source-to-hosted text and speculative execution.
- Qwen's admitted specialization reaches hybrid stateful text execution,
  without claiming its unexecuted vision components.
- MiniMax reaches composite media execution and publication; full-scale
  numerical and behavioral correctness remains an open boundary.
- Mamba2 reaches pinned acquisition, exact source roles, common transactional
  recurrent state, and component numerics. It remains partial and refuses
  READY: no complete decoder/artifact, load, or hosted generation.

The [family integration contract](docs/model-families/integration.md#current-family-boundaries)
links the evidence owners. No source recognition or generic plumbing promotes
another family to executable support.

## Current sequence

Only one row is active. States distinguish implemented boundaries from partial
work, dependency-blocked work, and missing measurements. Published milestone
IDs are not reused; retired completed sequences remain in Git history rather
than a second current ledger.

| Order | Milestone | State | Owned after-state | Depends on |
| ---: | --- | --- | --- | --- |
| 1 | `MAINTENANCE.ARCHITECTURE.REASSESSMENT.0` | `complete` | Source-qualified logical relations and platform-isolated client lifetimes, adversarial QA and a repeated whole-model performance control. | Accepted integrated foundation and documentation ownership |
| 2 | `SPECTRUM.MAMBA2.REPAIR.0` | `active` | Accepted evidence remains PARTIAL. Resolve source authorities and the SSM-only compiled decoder, then earn artifact, deployment, and hosted generation evidence. | Qualified reassessment; repair the same boundary |
| 3 | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | `partial` | Improve real warm execution under controlled phase/resource evidence without weakening numerical or lifecycle semantics. | Measured bottleneck selection; explicit resumption |
| 4 | `V010.EVAL.DEEPSEEK.0` | `blocked` | Repeatable behavior, quality, tokenizer, long-context, and refusal evaluation over the admitted product path. | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| 5 | `V010.BENCH.DEEPSEEK.0` | `not-measured` | Identity-bound full-model latency, throughput, memory, and reliability evidence. | `V010.EVAL.DEEPSEEK.0` |
| 6 | `V010.RELEASE.0` | `blocked` | All release gates close together; no missing evidence is relabelled pass. | `V010.BENCH.DEEPSEEK.0` |

Active Next: SPECTRUM.MAMBA2.REPAIR.0

The next delivery repairs Mamba2's existing boundary. An active work item is
not family promotion: its evidence remains PARTIAL until the product path is
qualified. This is not permission to begin another architecture, performance
campaign, or application integration.

## Open boundaries

### Architecture and model progression

Mamba2 exposes unresolved tokenizer/special-token and normalization authority,
and decoder assumptions that remain too attention/FFN-shaped for pure SSM.
The [Mamba2 record](docs/model-families/mamba2.md) owns the exact refusal and
highest evidence stage. A generic recurrent primitive does not close a model.

Future architecture work is selected from demonstrated pressure on execution,
state, topology, or typed input/output. Public direction is not the private
candidate ledger: unacquired reference names and planning rows confer no
capability. Cross-project application roles remain outside YVEX.

### Performance

DeepSeek long-decode behavior, Qwen steady decode, and MiniMax evaluation cost
are characterization candidates, not optimization conclusions. Select work
from controlled identities, phase attribution, rolling versus cumulative
rates, and resource ownership. Do not transfer old timings to a new tree or
treat component improvements as whole-model gains.

[GB10 targets](docs/development/gb10-targets.md) owns workload definitions,
engineering budgets, and the remaining empirical representation barrier.
No target number is a measured release result.

### Release

The v0.1 target remains DeepSeek text execution on the admitted GB10 path.
Other executable families do not automatically enter that release scope.
[Release doctrine](docs/releases/doctrine.md) defines the independent gates;
the [v0.1 record](docs/releases/v0.1.md) defines version-specific obligations.

```text
model_behavior_evaluation_ready=0
full_model_release_benchmark_ready=0
release_qualification_ready=0
```

## Progression discipline

The [engineering method](docs/development/agentic-engineering.md) explains how
a delivery establishes or falsifies a property and selects `proceed`,
`repair_same_boundary`, `complete_evidence`, or `blocked_external`.
Completing a prompt or passing software QA does not itself permit promotion.
Branch epochs are coordination history, not model-family ownership.
