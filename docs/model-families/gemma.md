# Gemma Technical Record

Status: source and candidate-role evidence; unsupported runtime family

This record owns current Gemma facts. Gemma is outside the v0.1 release target
and is not a generation-capable YVEX family.

## Current evidence

`gemma-4-12b-it` is a backend-neutral source target with configuration and
tokenizer facts, a header-metadata model-class profile, a header-only tensor
inventory, and header-derived candidate mappings into dense canonical role
labels.

Downloaded Gemma targets such as `gemma-4-31b-it` reuse the same source path.
Header evidence recognizes embedding, attention, dense MLP, normalization,
layer-scalar, final-norm, separate output-head, and config-proven tied
output-head candidates under `model.language_model`. A tied-head candidate
remains report evidence until artifact and logits owners consume it.

Gemma pressures a future dense runtime and CPU/CUDA physical-shape path.
Backend pressure is not target identity and does not establish CUDA support.

## Missing boundaries

Gemma still requires complete role validation, an artifact contract, complete
artifact production/admission, a runtime descriptor and binding, dense family
adapter coverage, graph and backend lowering, persistent state, tokenizer-to-
text generation, evaluation, and benchmark evidence.

## Non-claims

Source download, header inventory, lexical mapping, model-class reports, and
tied-head candidates do not establish source payload trust, a complete
artifact, runtime execution, generation, evaluation, benchmark, or release
support.
