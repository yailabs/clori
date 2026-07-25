#!/usr/bin/env sh
set -eu

fail() {
  printf 'docs surface: %s\n' "$1" >&2
  exit 1
}

require_file() {
  test -f "$1" || fail "missing file: $1"
}

require_text() {
  file=$1
  value=$2
  grep -nF -- "$value" "$file" >/dev/null ||
    fail "$file missing required text: $value"
}

require_pattern() {
  file=$1
  pattern=$2
  grep -nE -- "$pattern" "$file" >/dev/null ||
    fail "$file missing required pattern: $pattern"
}

reject_text() {
  file=$1
  value=$2
  if grep -nF -- "$value" "$file" >/dev/null; then
    fail "$file retains forbidden text: $value"
  fi
}

require_before() {
  file=$1
  first=$2
  second=$3
  first_line=$(grep -nF -- "$first" "$file" | head -n 1 | cut -d: -f1)
  second_line=$(grep -nF -- "$second" "$file" | head -n 1 | cut -d: -f1)
  test -n "$first_line" && test -n "$second_line" ||
    fail "$file cannot order missing sections: $first / $second"
  test "$first_line" -lt "$second_line" ||
    fail "$file section order is wrong: $first must precede $second"
}

for file in \
  README.md \
  AGENTS.md \
  PROJECT.md \
  MODEL_ARTIFACTS.md \
  NOTICE.md \
  docs/api.md \
  docs/contract.md \
  docs/model-families.md \
  docs/operator-runbook.md \
  docs/cli-output-architecture.md \
  docs/reference-architecture.md \
  docs/v010-release-doctrine.md \
  docs/topology-closure-audit.md \
  docs/system-target.md \
  docs/runbooks/README.md \
  docs/runbooks/deepseek.md \
  docs/runbooks/common.md
do
  require_file "$file"
done

require_file config/attention_quality.tsv

test -d docs/runbooks || fail "missing directory: docs/runbooks"
test ! -e docs/spine.md || fail "obsolete project-control path exists"
test -z "$(find docs -maxdepth 1 -type d -name repair -print -quit)" ||
  fail "temporary project-control directory exists"
test ! -e docs/runbooks/glm.md || fail "unsupported GLM runbook remains"

archive_path=docs/arc""hive
for path in \
  docs/README.md \
  docs/backend-contract.md \
  docs/cli-commands.md \
  docs/cli-interface-spine.md \
  docs/cli-runtime.md \
  docs/runtime-filesystem.md \
  "$archive_path" \
  docs/legacy \
  docs/spines
do
  test ! -e "$path" || fail "obsolete or archive path exists: $path"
done

runbook_count=$(find docs/runbooks -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')
test "$runbook_count" -eq 3 || fail "unexpected runbook count: $runbook_count"

unexpected_doc=$(find docs -maxdepth 1 -type f -name '*.md' \
  ! -name api.md \
  ! -name contract.md \
  ! -name model-families.md \
  ! -name operator-runbook.md \
  ! -name cli-output-architecture.md \
  ! -name reference-architecture.md \
  ! -name v010-release-doctrine.md \
  ! -name topology-closure-audit.md \
  ! -name system-target.md \
  -print -quit)
test -z "$unexpected_doc" || fail "unexpected canonical document: $unexpected_doc"

for heading in \
  "## 0. Repository contract" \
  "## 1. Directory is the namespace" \
  "## 2. Semantic owner admission" \
  "## 3. Machine-readable ownership" \
  "## 4. Generic and family boundaries" \
  "## 5. C interfaces, symbols, and contracts" \
  "## 6. Dependency DAG" \
  "## 7. Canonical capability ownership" \
  "## 8. CLI, reports, and output" \
  "## 9. Evidence and claim discipline" \
  "## 10. Tests and validation" \
  "## 11. Project control and closure" \
  "## 12. Final rule"
do
  require_text AGENTS.md "$heading"
done

require_text AGENTS.md 'A new file is not an implementation convenience.'
require_text AGENTS.md '`config/source_owners.tsv` is the canonical source-ownership manifest.'
require_text AGENTS.md '`tests/test_source_ownership.sh` rejects missing and duplicate registrations,'
require_text AGENTS.md '`tests/test_architecture_boundaries.sh`:'
require_text AGENTS.md 'basenames are at most 32 characters including the extension;'
require_text AGENTS.md 'maximum of three production'
require_text AGENTS.md 'Explicit user authorization is required before:'
require_text AGENTS.md 'source-relative object paths are mandatory'
require_text AGENTS.md '`PROJECT.md` is the sole project-control authority.'
require_text AGENTS.md 'Rank and state remain distinct. There'
require_text AGENTS.md 'is exactly one active milestone and exactly one Active Next.'
require_text AGENTS.md 'DeepSeek-V4-Flash is the'
require_text AGENTS.md 'v0.1.0 release target; it is not automatically supported'
require_text AGENTS.md '`TRACK.COMPILATION` owns artifact-neutral transformation semantics,'
require_text AGENTS.md 'Quantization is not a GGUF artifact.'
require_text AGENTS.md 'tensor proof artifact'
require_text AGENTS.md 'complete model artifact'
require_text AGENTS.md 'supported model artifact'
require_text AGENTS.md '### Commit format'
require_text AGENTS.md 'New commits use Conventional Commits:'
require_text AGENTS.md 'Runtime benchmark baselines and CSV/JSON evidence are'
require_text AGENTS.md 'A curated deterministic SVG derived from that evidence may be'
require_text AGENTS.md 'tracked under `docs/assets/` as documentation'
require_text AGENTS.md '### Progression admissibility'
require_text AGENTS.md '### Six-pass vertical iteration'
require_text AGENTS.md '### Quality and evaluation taxonomy'
for quality_class in \
  'Software tests' \
  'Numerical conformance' \
  'Runtime qualification' \
  'Component benchmarks' \
  'Model behavior evaluation' \
  'Model quality evaluation' \
  'Agent runtime evaluation' \
  'Release qualification'
do
  require_text AGENTS.md "$quality_class"
done
require_text AGENTS.md 'No generic `qa_passed` or `qa_ready` fact'
require_text AGENTS.md 'may collapse these classes.'

for classification in \
  gate_blocker \
  boundary_incomplete \
  evidence_gap \
  deferred_depth \
  optimization_debt \
  generalization_debt \
  external_blocker
do
  require_text AGENTS.md "\`$classification\`"
done

for decision in proceed repair_same_boundary complete_evidence blocked_external
do
  require_text AGENTS.md "\`$decision\`"
done

for pass in \
  'PASS 1 — Vertical closure' \
  'PASS 2 — Correctness and ownership hardening' \
  'PASS 3 — Memory and residency optimization' \
  'PASS 4 — Kernel and execution optimization' \
  'PASS 5 — Evaluation and benchmark' \
  'PASS 6 — Multi-family generalization'
do
  require_text AGENTS.md "$pass"
done

require_text AGENTS.md 'progression_decision: proceed | repair_same_boundary | complete_evidence | blocked_external'
require_text AGENTS.md 'downstream_safe: true | false'
require_text AGENTS.md 'per-family runtimes, duplicated storage or registries, target-string branches,'

require_pattern README.md '^# YVEX$'
require_text README.md '[Project status](PROJECT.md)'
require_text README.md '[`PROJECT.md`](PROJECT.md) is the sole live authority'

# The README is the stable public system definition. Guard its architecture,
# section order, claim discipline, and real operator surface without freezing
# volatile release evidence into the opening copy.
for heading in \
  '## What YVEX owns' \
  '## System architecture' \
  '## Design invariants' \
  '## Release vertical: DeepSeek-V4-Flash on NVIDIA GB10' \
  '## Verified implementation snapshot' \
  '## Current executable surfaces' \
  '## Build products and validation' \
  '## Repository architecture' \
  '## Documentation map' \
  '## License'
do
  require_text README.md "$heading"
done

require_before README.md '## What YVEX owns' '## System architecture'
require_before README.md '## System architecture' '## Design invariants'
require_before README.md '## Design invariants' \
  '## Release vertical: DeepSeek-V4-Flash on NVIDIA GB10'
require_before README.md '## Release vertical: DeepSeek-V4-Flash on NVIDIA GB10' \
  '## Verified implementation snapshot'
require_before README.md '## Verified implementation snapshot' \
  '## Current executable surfaces'

for definition in \
  'native C/CUDA inference system' \
  'pinned open-weight' \
  'model sources through admitted family profiles' \
  'identity-bound artifacts' \
  'runtime contracts.' \
  'Family profiles define model-specific topology' \
  'Common owners provide reusable verification'
do
  require_text README.md "$definition"
done

for term in Implements Executes Admits Supports Targets Plans; do
  require_text README.md "**$term**"
done

for boundary in \
  'Verified source snapshot' \
  'Family semantics + logical model' \
  'Exact tensor roles + Transformation IR' \
  'Physical profile + lowering' \
  'Quantization + encoding' \
  'Artifact construction + identity' \
  'Admission + materialization' \
  'Runtime binding' \
  'runtime model' \
  'execution session' \
  'Residency + memory plan' \
  'Semantic graph' \
  'Executable graph' \
  'Persistent model state' \
  'prefill or decode' \
  'Backend dispatch + launch graph' \
  'Output head + logits' \
  'Sampling + token append + stop policy' \
  'Detokenization + text' \
  'Identity-bound execution evidence' \
  'Evaluation' \
  'Benchmark' \
  'Release admission'
do
  require_text README.md "$boundary"
done

for invariant in \
  'Identity-bound derivation.' \
  'Logical and physical separation.' \
  'Planning before byte execution.' \
  'Family policy through typed boundaries.' \
  'Fail-closed admission.' \
  'Transactional publication.' \
  'Explicit resource ownership.' \
  'Backend execution without model inference.' \
  'Evidence scoped to the executed boundary.' \
  'Operator reachability.'
do
  require_text README.md "$invariant"
done

require_text README.md \
  '[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)'
require_text README.md 'is the sole v0.1.0 release target.'
for evidence in \
  '46 verified source shards and 69,187 exact source contributions' \
  '1,360 emitted terminal tensors' \
  'selected complete GGUF of approximately 102.4 GB' \
  '43 main attention layers and 634 core attention bindings' \
  'complete attention core and envelope execution on CPU' \
  'NVIDIA GB10 CUDA path'
do
  require_text README.md "$evidence"
done

for snapshot in \
  '| Persistent attention state | Admitted for all 43 DeepSeek attention layers' \
  '| Activation-driven attention prefill | Versioned 43-layer activation bundles' \
  '| Tokenizer-backed model prefill and model decode | Unsupported |' \
  '| FFN/MoE and complete transformer composition | Unsupported |' \
  '| Logits, sampling, and text generation | Unsupported |' \
  '| Evaluation | Blocked |' \
  'full-model benchmark is not measured' \
  '| Release | Blocked |'
do
  require_text README.md "$snapshot"
done
require_text README.md 'complete model artifact'
require_text README.md 'supported model artifact'

for command in \
  './yvex commands' \
  './yvex graph attention --help' \
  './yvex graph attention prepare' \
  './yvex graph attention describe' \
  './yvex graph attention execute' \
  './yvex graph attention compare' \
  './yvex graph attention state exercise' \
  './yvex graph attention benchmark'
do
  require_text README.md "$command"
done
require_text README.md 'canonical diagnostic activation'
require_text README.md 'versioned tensor-file activation bundle'
require_text README.md '--input tensor-file'
require_text README.md 'Prompt-backed prefill, model decode, and generation'

readme_commands=$(grep -E '^[[:space:]]*\./yvex([[:space:]]|$)' README.md |
  sed 's/^[[:space:]]*//')
while IFS= read -r command; do
  test -n "$command" || continue
  case "$command" in
    './yvex commands' | \
    './yvex graph attention --help' | \
    './yvex graph attention prepare \' | \
    './yvex graph attention describe \' | \
    './yvex graph attention execute \' | \
    './yvex graph attention compare \' | \
    './yvex graph attention state exercise \' | \
    './yvex graph attention benchmark \') ;;
    *) fail "README contains an unregistered operator command: $command" ;;
  esac
done <<EOF
$readme_commands
EOF

if grep -nE '^[[:space:]]*\./yvex (generate|prefill|decode|logits|sample|serve|fullmodel|materialize)([[:space:]]|$)' README.md; then
  fail 'README contains an unsupported operator command'
fi

require_text docs/contract.md '### DeepSeek Attention Operator Contract'
require_text docs/contract.md '## Production Activation-Prefill Contract'
require_text docs/api.md '### Internal DeepSeek Attention Operator Boundary'
require_text docs/api.md '### Internal Activation-Prefill Boundary'
require_text docs/contract.md '## Quality, Qualification, Benchmark, And Evaluation Contract'
require_text docs/api.md '## Qualification, Benchmark, And Chart Contract'
require_text docs/contract.md '### Attention Component Benchmark'
require_text docs/contract.md 'Model behavior,'
require_text docs/contract.md 'agent runtime/evaluation,'
require_text docs/api.md 'benchmark compare --baseline OLD'
require_text docs/api.md 'benchmark compare --max-regression-bps N'
require_text docs/runbooks/deepseek.md '--chart "$EVIDENCE/$MODE.svg"'
require_text docs/runbooks/deepseek.md './yvex graph attention qualify'
require_text docs/runbooks/deepseek.md './yvex graph attention benchmark compare'
require_text docs/runbooks/deepseek.md '--max-regression-bps 500'
require_text docs/runbooks/deepseek.md 'not model evaluation, agent evaluation or full-model benchmark'
require_text docs/runbooks/README.md \
  'production attention commands, benchmark charts, unsupported'
require_text docs/operator-runbook.md 'software acceptance, numerical'
require_text docs/operator-runbook.md 'agent-runtime evaluation'

for project_fact in \
  'model_behavior_evaluation_ready=0' \
  'model_quality_evaluation_ready=0' \
  'agent_runtime_ready=0' \
  'agent_tool_execution_ready=0' \
  'agent_evaluation_ready=0' \
  'release_qualification_ready=0' \
  'schema-v5 `attention_component`' \
  'V010.EVAL.DEEPSEEK.0` | DeepSeek | `blocked`' \
  'V010.BENCH.DEEPSEEK.0` | DeepSeek / DGX Spark | `not-measured`' \
  'V010.RELEASE.0` | DeepSeek v0.1.0 | `blocked`'
do
  require_text PROJECT.md "$project_fact"
done

if grep -nE '\b(qa_ready|qa_passed)[[:space:]]*=[[:space:]]*1\b' \
  src/*.c src/*/*.c include/yvex/*.h include/yvex/internal/*.h PROJECT.md README.md \
  2>/dev/null; then
  fail 'generic QA fact attempts to promote capability'
fi
if grep -nE 'agent_(runtime|tool_execution|evaluation)_ready=1|benchmark_scope[[:space:]]*=[[:space:]]*(model|generation|agent)' \
  PROJECT.md README.md config/attention_quality.tsv; then
  fail 'unsupported agent or higher benchmark capability is promoted'
fi
if grep -nE 'TRACK\.AGENT|V010\.AGENT\.' PROJECT.md; then
  fail 'agent track or milestone was introduced without scope authority'
fi

# Exact release evidence remains guarded at its live project-control owner.
for evidence in \
  '46/46 safetensors headers, 69,187 unique tensor records' \
  '177,680,573,600 bytes with identity `f16e800c0d7383ee76cb2e2fa8bdd674bab29c017cba64eaba85c39016e257ca`' \
  '102,408,545,440 bytes with identity `01b2bed4f070d0a3fdb02e546764b3a49cb69886eebe17b4877d20294725682c`' \
  '68 metadata entries, exact tokenizer material, and all 1,360 tensors' \
  '102,396,843,592 encoded payload bytes through bounded file-backed/staged access' \
  '33,792 expert subviews'
do
  require_text PROJECT.md "$evidence"
done

for stale in \
  '## Engineering Method' \
  'reasoning LLM' \
  'coding agent' \
  'candidate patch' \
  'Pareto' \
  'Only trusted source payload streaming is implemented' \
  '| Transformation IR | active — not implemented |' \
  '| Quantization and reference dequantization | blocked |' \
  '| GGUF writer and complete artifact | blocked |'
do
  reject_text README.md "$stale"
done

reject_text README.md '**YVEX is a native C inference engine'
reject_text README.md 'Active Next:'
reject_text README.md 'YVEX currently generates DeepSeek text'
reject_text README.md 'YVEX supports DeepSeek-V4-Flash'
reject_text README.md 'YVEX compiles arbitrary models'
reject_text README.md 'YVEX selects Pareto-optimal variants'
reject_text README.md 'YVEX executes the complete transformer'
reject_text README.md 'YVEX is release-ready'
reject_text README.md 'DeepSeek device residency is complete'
reject_text README.md 'DeepSeek model runtime is complete'
reject_text README.md 'DeepSeek CUDA attention is complete'

if grep -niE \
  'production-ready|blazing fast|state of the art|enterprise-grade|seamless|powerful|cutting-edge|revolutionary' \
  README.md; then
  fail 'README contains forbidden marketing language'
fi

if grep -nE 'V010\.|POST010\.' README.md; then
  fail 'README exposes internal project-control IDs'
fi
if grep -nE 'former .* owner|former CLI .* adapter' docs/topology-closure-audit.md; then
  fail 'historical topology evidence lost its commit-qualified path'
fi
if test -n "$(git ls-files '*.yvex-benchmark')"; then
  fail 'generated benchmark baseline is tracked'
fi
unexpected_svg=$(git ls-files '*.svg' |
  grep -Ev '^docs/assets/benchmarks/attention/(eager|eager-comparison|piecewise|piecewise-comparison|full|full-comparison)[.]svg$' ||
  true)
if test -n "$unexpected_svg"; then
  fail "unexpected generated SVG is tracked: $unexpected_svg"
fi
if grep -nE '(/home/|/Users/|\$HOME/)' README.md; then
  fail 'README exposes a local filesystem path'
fi
if grep -nF 'flowchart LR' README.md; then
  fail 'README architecture diagrams must remain top-down'
fi

mermaid_count=$(grep -c '^```mermaid$' README.md)
test "$mermaid_count" -ge 1 ||
  fail "README must contain a top-down architecture diagram: $mermaid_count"
flowchart_td_count=$(grep -c '^flowchart TD$' README.md)
test "$flowchart_td_count" -eq "$mermaid_count" ||
  fail "README Mermaid diagrams must be top-down: $flowchart_td_count/$mermaid_count"

primary_mermaid=$(awk '
/^```mermaid$/ && !seen {
  seen = 1
  inside = 1
  next
}
inside && /^```$/ { exit }
inside { print }
' README.md)
printf '%s\n' "$primary_mermaid" | grep -F 'flowchart TD' >/dev/null ||
  fail 'README primary architecture diagram is not top-down'
if printf '%s\n' "$primary_mermaid" |
  grep -niE 'deepseek|deepseek4|active|blocked|unsupported|complete'; then
  fail 'README primary architecture diagram is family-specific or stateful'
fi

awk '
/^```mermaid$/ {
  if (in_mermaid) exit 1
  in_mermaid = 1
  opened++
  next
}
in_mermaid && /^```$/ {
  in_mermaid = 0
  closed++
}
END {
  if (in_mermaid || opened != closed) exit 1
}
' README.md || fail "README Mermaid fences are unbalanced"

for target in $(grep -oE '\]\([^)]+\)' README.md |
  sed 's/^](//; s/)$//' |
  grep -Ev '^(https?://|mailto:|#)' || true)
do
  target=${target%%#*}
  test -e "$target" || fail "README local link does not resolve: $target"
done

sh tests/test_project_ledger.sh

project=PROJECT.md
for heading in \
  "## 1. Authority And Update Contract" \
  "## 2. Rank, State, And Proof Semantics" \
  "## 3. Product, Release, And Engineering Scope" \
  "## 4. Current Hard Truth" \
  "## 5. Active Work And Critical Path" \
  "## 6. Family Capability Matrix" \
  "## 7. Track Registry And Dashboard" \
  "## 8. First-Class Milestone Roadmap" \
  "## 9. Complete Track/Wave Ledger" \
  "## 10. Evidence Lanes" \
  "### 10.1 Decommission Obligations" \
  "## 11. Release Gates" \
  "## 12. Reference Architecture Ownership" \
  "## 13. Explicit Non-Claims" \
  "## 14. Version Sequence" \
  "## 15. Documentation Ownership And Cutover" \
  "## 16. Agent Start Checklist"
do
  require_text "$project" "$heading"
done

require_text "$project" '`PROJECT.md` is the single project-control authority for YVEX.'
require_text "$project" 'Current proof stage:'
require_text "$project" 'YVEX generates real text with DeepSeek-V4-Flash on the DGX Spark CUDA backend'
require_text "$project" '$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash'
require_text "$project" 'deepseek4-v4-flash'
require_text "$project" 'Qwen, Gemma, and dense/common work already implemented remains active'
require_text "$project" 'The exact DeepSeek-V4-Flash source projects to one immutable typed IR'
require_text "$project" 'One immutable IR-derived requirement set reconciles exactly against all 69,187 tensors'
require_text "$project" 'The sealed artifact-neutral Transformation IR now projects to 1,360 immutable GGUF lowering descriptors'
require_text "$project" 'Manifest v3 binds every shard to its authoritative Hugging Face Git LFS SHA-256'
require_text "$project" 'all 69,187 contributions and mapping identity `1aecbbe25b04de0d` remain exact'
require_text "$project" 'Production C contains no fallback PTX.'
require_text "$project" 'A no-`nvcc` build refuses every kernel before dispatch'
require_text "$project" 'Complete resident DeepSeek attention and stable double-buffered session state are admitted on the generated-bundle GB10 path'
require_text "$project" '`attention_execution_supported=1`, `attention_cuda_execution_ready=1`, and'
require_text "$project" 'complete DeepSeek SWA/CSA/HCA attention core and its immediate envelope are admitted'
require_text "$project" 'a supported DeepSeek-V4-Flash model artifact; the admitted artifacts are consumed by attention but have not passed the complete runtime and release gates;'
require_text "$project" 'complete-model, MoE, or output-head residency; only the admitted attention weight pack and session-owned persistent attention state are resident;'
require_text "$project" 'an execution-complete DeepSeek transformer runtime; the common attention runtime, persistent state, and activation-driven attention prefill are admitted but embedding, cross-layer hidden-state propagation, full-model prefill, FFN/MoE, complete layer composition, and final model output are not;'
require_text "$project" 'persistent_kv_ready=1'
reject_text "$project" 'complete GGUF writer, complete-model emission, writer-reader roundtrip, or artifact support admission;'
require_text "$project" '| Recovered IDs | 631 |'
require_text "$project" '| Explicit new IDs | 50 |'
require_text "$project" '| Canonical IDs | 681 |'
require_text "$project" '| First-class milestones | 44 |'
require_text "$project" '### 3.5 Model Compilation Boundaries'
require_text "$project" '| `TRACK.COMPILATION` | Artifact-neutral transformation IR'
require_text "$project" 'verified source facts'
require_text "$project" 'verified payload session'
require_text "$project" 'V010.MODEL.TRANSFORM.IR.0'
require_text "$project" 'family roles or transformation semantics. Quantization consumes canonical'
require_text "$project" 'transformation truth and may not rediscover source names, roles, aggregation'
require_text "$project" 'memory, persistent KV, and temporary scratch. Completed source payload'
require_text "$project" 'streaming is build-time source access; it is not inference-time SSD expert'

for category in \
  "Selected embedding and segment commands" \
  "Persistent KV" \
  "Diagnostic decode" \
  "Fixture logits and sampling" \
  "Bounded diagnostic generation" \
  "Selected-artifact support levels" \
  "Report-only fullmodel surfaces" \
  "Stale target, help, and claim tests"
do
  require_text "$project" "$category"
done
reject_text "$project" 'decide later'

for term in \
  "Tensor proof artifact" \
  "Complete model artifact" \
  "Supported model artifact"
do
  require_text "$project" "$term"
done

for rank in milestone capability evidence subtask migration future; do
  require_text "$project" "| \`$rank\` |"
done

if grep -nE '^### 9\.[0-9]+ TRACK\.(ARCHITECTURE|EXECUTION|MODELS|PROJECT|CLAIMS|TOPOLOGY)$' "$project"; then
  fail "attempted recovery track name became canonical"
fi

reference=docs/reference-architecture.md
require_pattern "$reference" '^# Reference Architecture for Verified Transformer Inference$'
require_text "$reference" 'Status: implementation-agnostic research architecture'
require_text "$reference" 'This document defines a conformance model.'
require_text "$reference" 'This appendix is intentionally separated from the implementation-agnostic'
require_text "$reference" 'It owns the external engineering baseline and maps'
require_text "$reference" 'It does not own project'
require_text "$reference" 'state, milestone state, dependency order, capability claims, or Active Next;'
require_text "$reference" '## 21. Architectural Invariants'
require_text "$reference" '## 22. Conformance Criteria'
require_text "$reference" '## References'
for source in \
  'vLLM architecture' \
  'vllm/model_executor/models/deepseek_v4.py' \
  '96a04cb13f9c3ed86028e090784a9eb059cf5318' \
  'python/sglang/srt/models/deepseek_v4.py' \
  'GGUF specification' \
  'e920c523e3b8a0163fe498af5bf90df35ff51d25' \
  'conversion/deepseek.py' \
  'src/models/deepseek4.cpp' \
  'TensorRT-LLM architecture' \
  'NVIDIA/cutlass' \
  'Driver API module management' \
  'execution control' \
  'DeepSeek [V4 technical report v1]' \
  '60d8d70770c6776ff598c94bb586a859a38244f1' \
  'deepseek-ai/FlashMLA' \
  'a2bcc5c86678b72a86b7aadc29b643a5ce63c747' \
  '0893eac771d532b7110f1f7581d3f4cd0b9172bf' \
  '02cedf6e4e421ac48d271452bf3836cb57caf297' \
  '1e7394829291360bdcf07036cbe5411631d2d33b' \
  'ae754e9ed8b650e78b921906b2ba8af65ea408ab' \
  '80ebbc396aee40eedc1d829222f3362d10fa4c6c' \
  'SSD expert streaming' \
  'quality validation'
do
  require_text "$reference" "$source"
done
require_text "$reference" 'src/backend/cuda/'
require_text "$reference" 'src/model/families/'
require_text "$reference" '8df14cfc8c8a09b4e57f082e59593a3abce4ffb3'
require_text "$reference" 'V010.REBASE.DEEPSEEK.0'
require_text "$reference" 'V010.GGUF.ARTIFACT.ABI.1'
require_text "$reference" 'V010.RUNTIME.DEEPSEEK.MOE.0'

doctrine=docs/v010-release-doctrine.md
require_text "$doctrine" '`PROJECT.md` owns the v0.1.0 product target and active execution sequence.'
require_text "$doctrine" '## Gate State Ownership'
require_text "$doctrine" '`PROJECT.md` is the sole owner of current gate state, milestone state,'
require_text "$doctrine" 'The target is currently unsupported.'
reject_text "$doctrine" '## Current State'

require_text MODEL_ARTIFACTS.md 'Tensor proof artifact'
require_text MODEL_ARTIFACTS.md 'Complete model artifact'
require_text MODEL_ARTIFACTS.md 'Supported model artifact'
require_text MODEL_ARTIFACTS.md 'Two complete DeepSeek-V4-Flash model artifacts currently exist outside the'
require_text MODEL_ARTIFACTS.md 'neither is a'
require_text MODEL_ARTIFACTS.md 'supported model artifact:'
reject_text MODEL_ARTIFACTS.md 'No such complete model artifact currently exists.'
require_text MODEL_ARTIFACTS.md 'decommission obligations and consuming milestones are recorded in'
require_text MODEL_ARTIFACTS.md 'GGUF is the'
require_text MODEL_ARTIFACTS.md 'v0.1.0 release lowering, not the identity of the logical model.'

deepseek_lines=$(wc -l < docs/runbooks/deepseek.md | tr -d ' ')
test "$deepseek_lines" -le 100 || fail "DeepSeek runbook is not short: $deepseek_lines"
require_text docs/runbooks/deepseek.md '$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash'
require_text docs/runbooks/deepseek.md 'deepseek4-v4-flash'
require_text docs/runbooks/deepseek.md 'There is no supported DeepSeek generation command to run yet.'
require_text docs/runbooks/deepseek.md 'Current milestone state, dependencies, gates, and Active Next live only in'
if grep -nE '\./yvex (generate|prefill|decode|logits|sample|fullmodel|materialize)' docs/runbooks/deepseek.md; then
  fail "DeepSeek runbook retains an unsupported generation execution lane"
fi

require_text docs/system-target.md 'Authority: filesystem and module topology; current project state belongs only'
require_text docs/system-target.md '## GGUF Structural Reader Boundary'
require_text docs/system-target.md '## GGUF Qtype ABI Boundary'
require_text docs/system-target.md '| Transformation plan | sealed artifact-neutral IR binds all 69,187 source values to 1,360 terminal tensors'
require_text docs/system-target.md '| GGUF writer | deterministic v3 plan and transactional file writer complete |'
require_text docs/system-target.md '| Runtime descriptor | immutable DeepSeek descriptor binds all 1,360 admitted tensors and topology facts and is consumed by the common attention runtime and persistent-state recipes |'
reject_text docs/system-target.md '| Transformation plan | no artifact-neutral transformation IR exists |'
require_text docs/topology-closure-audit.md 'point-in-time inventory'
require_text docs/topology-closure-audit.md '`PROJECT.md` owns when each finding is removed or'
require_text docs/cli-output-architecture.md '## Project State Ownership'
require_text docs/model-families.md 'exact v0.1.0 target'
require_text docs/model-families.md 'sealed Transformation IR, complete quantization, two admitted complete artifacts'
require_text docs/model-families.md 'complete CPU/GB10 attention, and session-owned persistent attention state exist; tokenizer-backed prefill, MoE, transformer execution, and generation remain unsupported'
reject_text docs/model-families.md 'no artifact-neutral transformation plan, payload conversion, complete model artifact, or runtime path'
require_text docs/contract.md 'These are implementation facts, not a runtime progress ladder.'
require_text docs/contract.md 'defined only by `PROJECT.md`.'
require_text docs/contract.md '### Model Compilation Contract'
require_text docs/contract.md 'Source payload streaming remains build-time access and does'
require_text docs/api.md 'decommission obligations in `PROJECT.md`'
for retired_header in runtime.h generation.h metrics.h; do
  require_text docs/api.md "$retired_header"
done
require_text docs/api.md 'incompatible pre-release ABI cutover'
require_text PROJECT.md '`V010.RUNTIME.1` is the completed cutover'
require_text PROJECT.md 'Historical diagnostic runtime/generation state'

repair_path=$(printf 'docs/%s' repair)
if rg -nF "$repair_path" AGENTS.md PROJECT.md MODEL_ARTIFACTS.md docs tests Makefile; then
  fail "obsolete temporary project-control path is still referenced"
fi

repair_word=repair
spine_word=spine
repair_phrase="$repair_word $spine_word"
if rg -niF "$repair_phrase" AGENTS.md PROJECT.md MODEL_ARTIFACTS.md docs tests Makefile; then
  fail "obsolete temporary project-control document is still referenced"
fi

if rg -n '^Active Next:' AGENTS.md MODEL_ARTIFACTS.md docs Makefile; then
  fail "project state is duplicated outside PROJECT.md"
fi

if rg -nF 'Active Next: V010.' Makefile tests/test_project_ledger.sh; then
  fail "current milestone is hard-coded in a generic guard"
fi

echo "docs surface: ok (project_control=PROJECT.md taxonomy=owned)"
