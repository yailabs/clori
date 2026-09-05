# Security Policy

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/yailabs/yvex/security/advisories/new)
to contact the repository maintainers. Do not disclose an unpatched
vulnerability in a public issue, pull request, or discussion.

Include the affected commit, platform and backend, a minimal reproduction,
expected and observed behavior, and the impact across the trust boundary.
Prefer synthetic inputs. Remove credentials, personal data, private prompts
and responses, and machine-specific paths. Do not upload model weights,
complete runtime dumps, or third-party data you cannot redistribute.

Maintainers use the private report to investigate and coordinate a fix and
public advisory when appropriate. There is no guaranteed response time,
backport window, or bounty program. For ordinary defects and usage questions,
see [SUPPORT.md](SUPPORT.md).

## Supported development line

YVEX is under development toward v0.1.0. Security fixes target the current
`main` branch; no released or long-term support series is currently declared.
Report an issue even if it was discovered on an older revision, and identify
that revision. [ROADMAP.md](ROADMAP.md) owns release qualification state.

## Local trust boundary

- `yvex serve` runs with the operator's filesystem and device privileges. Run
  it as an ordinary user with access only to the models and data it needs.
- The native Unix-domain socket uses private filesystem permissions and
  validates that the peer UID owns the runtime on supported platforms.
- The optional OpenAI-compatible HTTP listener binds to `127.0.0.1`. It has
  no client authentication or TLS. Loopback is reachable by other local
  processes and users; it does not provide the native socket's UID isolation.
  Do not publish it through a proxy, tunnel, container port mapping, or public
  interface, or use it as an isolation boundary on an untrusted shared host.
- Source, tokenizer, media, and artifact parsers process external data.
  Identity checks and admission validate their declared contracts; they do
  not sandbox parser execution, establish a publisher's trustworthiness, or
  guarantee that generated output is safe.
- Provider acquisition may contact external services and use separately
  installed provider tools and credentials. Local inference does not imply
  that acquisition is offline. Keep those tools and system dependencies
  updated and use trusted model sources.

The implemented transport and lifecycle contracts are in
[local protocol](docs/contracts/local-protocol.md),
[OpenAI compatibility](docs/openai-compatibility.md), and the
[operator runbook](docs/operator-runbook.md).

## Development and disclosure hygiene

Never commit credentials, model payloads, private service environments, or raw
operator traces. Review changes and sanitize logs before attaching evidence to
GitHub. Automated scanning supplements that review and does not establish
that a repository or artifact is free of secrets or vulnerabilities.

If a credential is exposed, revoke or rotate it with its provider first;
removing the current file does not revoke copies in Git history or logs.
Coordinate any history or published-artifact cleanup with the maintainers.
