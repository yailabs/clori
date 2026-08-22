# Expert Worklist Prefill Admission Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-18 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `c916962399464b8c7bc5068b5afbdf904a779d29` |
| Checkpoint | `8ba4459ccce60867a462f6aa90ff4327469a9198` |
| Subsystem | runtime prefill admission and CUDA MoE expert worklists |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | CUDA refusal test; compute-sanitizer; complete-model generation; ASan/LeakSanitizer/UBSan |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

Physical Execution IR v4 and runtime binding v14 sealed the admissible expert
worklist widths, but the generation prefill loop still treated the configured
prefill chunk as sufficient authority. A 15-token reasoning prompt could enter
one transformer execution wider than the compiled worklist policy.

The CUDA worklist builder correctly rejected that unsupported width. Execution
nevertheless continued into grouped MoE kernels, which consumed an order buffer
that had not been initialized by a successful worklist build. The foreground
server closed during prefill and the operator observed a segmentation fault.

## Problem

The compiler-sealed width policy was authoritative in the worklist contract but
was not enforced at every consumer boundary. The runtime created an invalid
physical chunk, the backend did not refuse it before graph mutation, and the
kernels assumed that upstream worklist validation had succeeded.

This was a correctness and failure-contract defect, not a request to enlarge
the admitted execution width. Weakening the policy or manufacturing rows would
repeat the previously rejected fake-width Tensor Core regime.

## Causal analysis

The exact operator prompt `mi spieghi come funziona un llm` in explicit
reasoning mode produced 15 prompt tokens. The configured chunk admitted all 15,
while the compiled compatible-row envelope was smaller. A blocking CUDA run
localized the failure to the attention completion boundary. Compute-sanitizer
then reported an invalid global read in `yvex_moe_grouped_up_rows` while
dereferencing the selected expert through an uninitialized order entry.

Source inspection established the sequence:

1. runtime selected a chunk wider than the compiled worklist policy;
2. `yvex_expert_worklist_build_cuda` returned an unsupported-width status;
3. grouped execution still launched;
4. the kernel read `order[ordered_pair]` before checking the failed status and
   before proving that the resulting source-pair index was bounded.

The backend therefore had two missing fail-closed boundaries: a host preflight
before numerical mutation and a device-side guard against a failed producer.

## Decision

Preserve Physical Execution IR v4 and binding v14 as the width authority.
Generation caps each physical prefill chunk to the compiler-sealed compatible
batch width. CUDA validates the requested row width against the same sealed
policy before worklist construction or graph mutation. Grouped kernels also
observe the device status first and bound the source-pair index before reading
selected experts.

This is layered defense, not duplicated planning. Runtime forms only admitted
chunks; the backend refuses an invalid compiled operation; the kernel remains
safe if an earlier device producer fails.

## Implementation

Checkpoint `8ba4459ccce60867a462f6aa90ff4327469a9198` makes the runtime copy the
model's compatible-batch width and use it as an upper bound for prefill chunks.
The CUDA MoE owner refuses a row count absent from the sealed policy mask with
`YVEX_ERR_UNSUPPORTED` before output publication or graph execution.

Grouped-up and grouped-down kernels now stop on a non-zero worklist status,
validate mandatory geometry before division, and reject an out-of-range source
pair before using it to index the selected-expert array. The focused CUDA test
seals a policy that excludes the requested width and proves refusal leaves the
completion result and output tensor unmodified.

The complete-model reasoning fixture now uses the original multiword prompt so
the formerly unsafe prefill geometry remains production-reachable in QA.

## After

The 15-token prompt is executed as three compiler-admitted prefill chunks in
both target-only and DSpark generation. The same explicit reasoning request
completes prefill and generation without closing the foreground server.

An unsupported width is a typed pre-mutation refusal. A failed device worklist
producer cannot cause grouped kernels to read uninitialized order data.

## Quantitative delta

| Fact | Before | After | Evidence class |
| --- | --- | --- | --- |
| 15-token prefill physical chunks | one unsupported width | three admitted chunks | directly comparable execution fact |
| Unsupported-width backend behavior | worklist failure followed by kernel execution | typed refusal before graph/output mutation | directly comparable contract fact |
| Compute-sanitizer | invalid global read | zero reported errors | directly comparable prompt and artifact |
| Complete-model generation lane | operator-observed server crash | PASS | runtime qualification |

No throughput comparison is promoted from this repair. The live lane repeatedly
opens the complete model and is not a performance fixture.

## Evidence

- CUDA refusal evidence
  `build/qa/evidence/79a87f0a5a9bd7d4790717e4c5ad9d64285b5f2d8d22dec3774dba8a4692bc5d.json`
  passed the focused native owner.
- Complete-model generation evidence
  `build/qa/evidence/36f26541997d33a4cf2cb3c96306843ee1924e8e1388cbd5ee2272d081e80a9a.json`
  passed target-only parity, repeated stochastic identity, DSpark verification
  and CLI reachability using artifact identity
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
  and binding identity
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Runtime sanitizer evidence
  `build/qa/evidence/2cf4ea27284b1a35a66b04760aad1798df527f499115eaac67f86337a7509ee2.json`
  passed ASan, LeakSanitizer and UBSan coverage.
- Quant sanitizer evidence
  `build/qa/evidence/1e79bb57868c9fb5c63a9a66c905b7b6677b5678fe6aa29a3f377d4d082f7a6e.json`
  passed its admitted sanitizer boundary.
- A direct compute-sanitizer replay of the exact failing prompt completed with
  `ERROR SUMMARY: 0 errors`; the raw tool transcript remains an untracked
  operator asset.
- Target-only and DSpark server replays both preserved the foreground process
  after the 15-token prompt.

## Remaining limitations

- This repair does not enlarge the compiled compatible batch width or create
  additional real execution width.
- It does not increase Tensor Core routed-pair coverage, change qtype policy or
  claim a performance improvement.
- Complete-model admission still materializes a large process-resident model;
  loading/residency economics remain a separate measured owner.
- The repair does not close GB10 continuous batching, deep-context,
  persistence, evaluation, benchmark or release gates.

## Why it matters

Compiler-sealed execution width now remains authoritative from prefill planning
through CUDA launch, and a rejected worklist fails as a typed operation instead
of becoming an out-of-bounds device read and a dead server.

## Communication projections

### Short update

YVEX repaired a DeepSeek reasoning-mode crash by carrying the compiler-sealed
expert-worklist width into prefill chunking and enforcing it again before CUDA
mutation. The original 15-token prompt now runs as three admitted chunks, while
unsupported widths fail closed and compute-sanitizer reports zero errors.

### Quoteable technical facts

- "Prefill chunks cannot exceed the compiler-sealed compatible-row envelope."
- "Unsupported expert-worklist width is refused before CUDA graph or output
  mutation."
- "Grouped MoE kernels stop before consuming order metadata from a failed
  worklist build."
