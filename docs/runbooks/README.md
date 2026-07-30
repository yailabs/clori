# YVEX Runbooks

Runbooks contain executable operator procedures for current behavior. They do
not record delivery progress or create capability.

## Files

- `deepseek.md` records the exact v0.1.0 source, admitted artifact and runtime
  binding, production attention commands, benchmark charts, unsupported
  generation boundary, and safe filesystem checks.
- `common.md` records repository validation, claim checks, artifact guardrails,
  and operator-local cleanup boundaries.

Normative family architecture lives in `../model-families.md`. Project control
and current sequencing live only in `../../ROADMAP.md`.

## Rules

- Use only commands that exist in the current checkout.
- State requirements, writes, rerun safety, stop point, and capability boundary.
- Do not present tensor proof artifacts as complete or supported model
  artifacts.
- Do not present reports, fixtures, selected segments, or bounded control loops
  as runtime generation.
- Keep operator-local source, artifacts, reports, registries, logs, and build
  output outside git.
