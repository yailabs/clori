## Boundary

Describe the problem, the affected owner, and the resulting behavior.
Link the issue or decision record when applicable.

## Reachability

Name the changed operator command, API, or other consumer. For documentation,
policy, or build-only changes, explain that scope instead.

## Validation

- QA plan command and identity:
- Required lanes and structured evidence:
- PASS / FAIL / SKIP / BLOCKED / ERROR summary:
- Remaining failures or missing prerequisites:

Include focused failure, cleanup, cancellation, numerical, or performance
evidence when the changed contract requires it. Sanitize all attached logs;
keep credentials, model payloads, and private data out of the diff and report.

## Claims and progression

State any capability, compatibility, or readiness change and the remaining
non-claims. For milestone work, include the progression decision, downstream
safety, and any `ROADMAP.md` or decision-record update. Mark these inapplicable
when the delivery does not change a milestone.

## Review checklist

- [ ] The diff preserves unrelated work and contains only owned changes.
- [ ] Required tests ran; missing or failed evidence is reported explicitly.
- [ ] Documentation and third-party attribution match the changed contract.
- [ ] No credentials, model payloads, generated packages, or runtime dumps are included.

For vulnerabilities, use the private reporting process in `SECURITY.md`
before opening a public pull request.
