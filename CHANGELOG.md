# Changelog

All externally meaningful YVEX changes are recorded here. The project is not
yet released; entries remain under **Unreleased** until release qualification
and a version tag are accepted.

This changelog follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
without treating every internal milestone, test, or refactor as a public
change. Git history preserves implementation chronology.

## Unreleased

### Added

- DeepSeek-V4-Flash-DSpark as the sole current DeepSeek source target, with
  target-only reference generation and target-verified speculative generation
  in the same resident runtime model and session authority.
- Complete daemon-backed DeepSeek-V4-Flash source-to-streamed-text execution
  through native and bounded OpenAI-compatible local surfaces.
- Exact server-owned multi-turn sessions with committed-prefix reuse,
  cancellation, partial-progress truth, and one persistent model lifecycle.
- Registry-driven command discovery, advanced help, JSON discovery, and Bash,
  Zsh, and Fish completion.
- A daemon-backed `yvex>` console with composed attachment state, live prefill
  progress, direct streamed output, typed turn metrics, registry-derived slash
  completion, semantic watch/human trace, and clean Ctrl-C/Ctrl-D handling.

### Changed

- Consolidated the product topology to the public `yvex` command and the
  long-lived `yvexd` host; the OpenAI-compatible listener now runs inside the
  daemon.
- Replaced implementation-era top-level command buckets with the canonical
  `compile`, `artifact`, `inspect`, `execute`, `profile`, and `system`
  projections.
- Advanced the private local protocol to version 4, separating selected model
  configuration from the live runtime model and removing false artifact/model
  facade operations.
- Advanced the private local protocol to version 5 for typed speculative-cycle,
  accepted-prefix, and committed-only usage facts; version 4 is refused.
- Made hosted startup registry-first: `model list` reports complete startup
  profiles, `model select NAME` resolves one profile without path flags, and
  `runtime start` opens the selected model without environment variables.
- Made the human terminal surface compact and semantic: startup announces the
  selected model before admission; REPL attachment facts and commands use a
  stable vertical hierarchy while turn metrics remain compact; TTY color
  respects `NO_COLOR`; Ctrl-L clears and redraws active input; and categorized
  operational watch separates signal from connection churn and detailed
  trace/profile output.
- Reorganized documentation by authority and lifecycle, with canonical
  terminology, family records, contracts, operator procedures, frozen audits,
  and validated migration paths.

### Removed

- Retired the separate `yvex-dev` and `yvex-openai` product executables.
- Removed the old top-level `evidence`, `graph`, `quant`, `source`, `tensor`,
  and `tokenizer` command namespaces; migration hints do not execute hidden
  aliases.

### Security

- The hosted protocol and OpenAI-compatible endpoint remain local-only and
  fail closed. Authentication, TLS, CORS, and remote exposure are not part of
  the current compatibility profile.
