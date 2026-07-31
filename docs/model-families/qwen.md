# Qwen Technical Record

Status: source and candidate-role evidence; unsupported runtime family

This record owns current Qwen facts. Qwen is outside the v0.1 release target
and is not a generation-capable YVEX family.

## Current evidence

`qwen3-8b` is a backend-neutral source target with configuration and tokenizer
facts, a header-metadata model-class profile, a header-only tensor inventory,
and header-derived candidate mappings into canonical role labels.

Downloaded Qwen targets such as `qwen3-6-35b-a3b` reuse the same source path.
Header evidence recognizes embedding, final norm, attention Q/K/V/O, Q/K
norms, linear-attention tensors, MoE router, routed experts, and shared experts
under `model.language_model`. These are mapping candidates, not validated
runtime roles.

Qwen may pressure dense and sparse/MoE composition, tokenizer portability, and
future Metal memory/lowering work. Backend pressure is not target identity and
does not establish Metal support.

## Missing boundaries

Qwen still requires complete role validation, an artifact contract, complete
artifact production/admission, a runtime descriptor and binding, family
adapter coverage, graph and backend lowering, persistent state, tokenizer-to-
text generation, evaluation, and benchmark evidence.

## Non-claims

Source download, header inventory, lexical mapping, model-class reports, and
quantization-policy reports do not establish source payload trust, a complete
artifact, runtime execution, generation, evaluation, benchmark, or release
support.
