# Command and operation architecture

Status: current implemented architecture

YVEX is one inference/compiler/runtime product with four distinct process
roles: finite offline work, one persistent hosted runtime, native local
clients, and external compatibility clients. Command spelling never changes a
process role according to TTY state, loaded engines, or host presence.

## Domain objects

| Object | Authority |
| --- | --- |
| Model | logical semantic identity shared by related representations |
| Source | acquired payload, exact revision, provenance, and verification state |
| Artifact | immutable compiled physical package |
| Profile | durable deployment configuration naming an artifact, backend, and admitted runtime policy |
| Engine | process-local loaded engine generation owned by the host |
| Session | mutable conversation/generation state bound to one exact engine generation |
| Host | foreground process owning engines, sessions, scheduling, execution, listeners, and logs |

These names are not aliases. In particular, a profile is not a loaded engine.
The default logical-model projection contains deployable models only; source
records without authenticated lineage remain visible under `source`, and one
model with several profiles still appears once.

## Product grammar

```text
yvex
yvex help [COMMAND...]
yvex help --advanced
yvex help --json
yvex version

yvex serve
yvex chat

yvex host status|logs|stop
yvex host memory
yvex engine list|show|load|unload
yvex session list|new|show|attach|detach|reset|cancel|close

yvex model list|show
yvex source list|show|inspect|verify|acquire
yvex artifact list|show|verify|status|materialize
yvex profile list|show|create|verify|remove|scan
yvex compile ...
yvex inspect ...
yvex bench ...
```

Bare `yvex` prints the compact command map and exits successfully. It never
starts, attaches to, or probes a host and never enters chat. Interactive human
generation requires `yvex chat`. Programmatic generation uses the private
typed protocol through native clients or the admitted loopback provider API;
there is no public one-shot generation command.

The old `run`, `server`, `execute`, `profile`-as-a-verb, and implementation
top-level namespaces are not aliases. Obvious retired spellings fail with a
bounded migration hint and never dispatch their former operation.

## Process roles

### Foreground host

`yvex serve` always attempts to become the persistent foreground host. It owns
the private Unix socket, engine manager, scheduler, sessions, backend
execution, telemetry, and optional loopback OpenAI listener. A one-time TTY
boot report is followed by an ordinary line-oriented host log stream. The
process does not read an administrative command language from stdin, own the
alternate screen, clear scrollback, or become a client.

If a healthy host already owns the socket, `yvex serve` refuses and identifies
the socket and inspection command. It never attaches to that host. Native
administration is performed from another process with `host`, `engine`, and
`session` commands.

### Native chat client

`yvex chat` is the sole public REPL. It requires a TTY and connects to the
private local protocol. It opens no artifacts, initializes no CUDA state,
starts no host, and loads no engine. A missing host produces the explicit
`yvex serve` remediation; a missing engine points to interactive `yvex engine
load`.

The linear editor preserves scrollback and owns bounded history, UTF-8
deletion, bracketed paste, resize redraw, cancellation, and terminal
restoration. Registry-derived slash operations are limited to conversation and
session use:

```text
/help /status /context
/new /sessions /session /attach /detach /reset /cancel /close
/think /think-max /nothink
/quit
```

Reasoning policy persists for the attached session until changed. The client
does not describe it as a next-turn-only setting. Load, unload, compile,
acquire, and host-stop operations are deliberately absent from the REPL.

### Hosted administration

`host` addresses the already-running process. `engine load` on a terminal
offers a linear model/deployment chooser; its numeric rows are temporary
rendering conveniences and resolve to one exact profile identity before the
native request. `engine load PROFILE` is the deterministic automation form.
Both create an immutable process-local engine generation; `engine unload
ENGINE` drains that generation without stopping the host. Session commands
address server-owned mutable state and cross the same private protocol. None of
these clients opens a package directly.

### Offline work

Model and source commands inspect semantic/provenance truth. Artifact commands
inspect immutable compiled packages. Profile commands inspect deployment
configuration. `compile` owns the admitted high-level source-to-artifact path;
specialized compiler phases remain discoverable below it. Bounded component
execution and measurement live under `bench`; read-only engineering evidence
lives under `inspect`. Offline commands neither require nor start a host unless
their named domain is explicitly hosted.

## Protocol planes

Native commands and chat use private local protocol v16 over a UID-owned Unix
socket. That protocol carries YVEX engine generations, sessions, KV identity,
lifecycle, typed progress, cancellation, resource facts, and telemetry.

External compatibility consumers use the separate loopback HTTP OpenAI
profile. The adapter owns no model, engine, session, KV, scheduler, or backend
state and never parses CLI output. Native clients do not route through HTTP.

## Canonical operation authority

`config/operator/registry.json` is the strict `yvex.operator.registry.v1`
operation authority. The build
validates it and emits immutable descriptors plus a content identity. Those
descriptors drive parsing, dispatch, human help, JSON discovery, shell
completion, and the chat slash catalog. Domain owners retain capability,
semantic validation, and typed result authority; the registry does not create
support by naming an operation.

Default help projects only the product map. `help --advanced` exposes admitted
advanced and engineering leaves. `help --json` is a stable structured
projection with exact operation and command identities. Human and machine
renderers consume the same typed result and machine output contains no ANSI.

## Output and errors

Product results go to stdout and product errors to stderr. Human views respect
TTY detection and `NO_COLOR`; redirected and structured output is
non-decorative. Submission-time errors surface immediately. Host/backend
failures are returned through the correct typed completion boundary rather
than being inferred from log prose.

The foreground host is the logging producer. `yvex host logs` projects its
typed stream from another process; `--json` selects the machine JSONL
projection and `--follow` makes continuous intent explicit. No second logging
authority or administrative REPL exists.

## Non-claims

This local product architecture does not establish public HTTP serving,
authentication, TLS, remote security, full OpenAI compatibility, distributed
serving, model quality, benchmark authority, or release qualification.
