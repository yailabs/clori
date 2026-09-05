# 0007 — External terminal editor ownership

Date: 2026-09-05
Status: accepted

## Context

The chat protocol client previously implemented byte input, raw mode, cursor
movement, history navigation, paste framing and redraw alongside product
commands, sessions and generation. REPLAI now supplies that terminal substrate
through an independently qualified C ABI, including the existing line-oriented
presentation grammar. Keeping two live editors would leave ownership ambiguous.

## Decision

`src/cli/io/client.c` consumes REPLAI C ABI 1 through its installed header. The
exact revision, Git tree and downloaded archive checksum are owned by
`config/replai.json`: revision `df5538c718b8d068432032e7fb116fb8bfab158e`.
No floating branch, vendored header, committed native artifact or runtime
fallback editor exists. The adjacent REPLAI repository is not required.

The product statically links `libreplai_c.a`. System link flags come from the
staged `replai.pc`. This preserves the single executable package and introduces
no runtime loader requirement for a REPLAI shared library. The private prefix
contains the producer's original header, native artifacts, license and a
checksum receipt. Product packages retain its license and build receipt.

`make client` builds the pinned external producer when needed. The build requires
current stable Rust/Cargo, Python 3, pkg-config and the existing native compiler.
The producer downloads the immutable archive over HTTPS, checks its SHA-256,
builds with `cargo build --locked --release -p replai-c`, and invokes REPLAI's
own staging command. Source and Cargo target files live in a temporary external
cache workspace, not in Git. A failed preparation never publishes a partial
prefix. Repeated builds verify installed checksums and reuse the installation.

`REPLAI_PREFIX` can select an empty installation location. `REPLAI_SOURCE` can
select a clean adjacent checkout **at the same exact pin**; its source is not
modified and Cargo output remains external. Incompatible source revisions,
header ABI, receipts, missing artifacts or changed artifact bytes fail the
build. Changing the pin requires a fresh prefix and consumer requalification.

The adapter owns one opaque handle per chat lifetime. Each input opens the
terminal with a literal host label/state suffix, polls events, and closes before
product dispatch. Submission copies exact UTF-8 into a C-owned command buffer.
YVEX admits nonempty non-command history and suppresses consecutive duplicates;
REPLAI owns navigation and returning to the original draft/cursor. The YVEX
operation registry selects a unique slash replacement; the library validates and
applies it. Ambiguous/missing completions preserve the draft.

Keyboard interrupts while editing return generic events; YVEX clears the draft
and applies its repeated-interrupt exit policy. Host-delivered SIGINT is observed
by a minimal YVEX handler and delivered to the library from ordinary control
flow. Resize installs no YVEX SIGWINCH handler; REPLAI polls dimensions. EOF and
input failure are distinct. Ctrl-D deletes at the cursor on nonempty input and
exits on empty input, retaining the qualified editor correction.

Generation starts only after the input interaction has restored captured termios
and released duplicated descriptors. The existing YVEX generation signal thread,
protocol cancellation request, quiet-output termios scope and queued-key discard
policy remain product-owned. No prompt is concurrently editable during generation.
Reconnect, attachments, command grammar, engine/session binding, reasoning and
product output retain their existing owners.

## Consequences and evidence

The old live `repl_read_line`, byte insertion/deletion, escape reader, scalar
column count, redraw and history-navigation implementation is removed. Product
formatting, palette use outside editing, generation terminal suppression and
semantic helpers with historical `repl_` names remain for the later removal audit;
they are not an alternate input loop.

`tests/repl_pty.sh` retains its existing attachments, streaming, cancellation,
reconnect and linear-surface coverage. Its production-process extension observes
exact protocol input after Unicode/grapheme edits, history return, multiline CRLF
paste and slash completion. It checks prompt bytes, color-disable rules, resize,
Ctrl-L, editing interrupts and exact captured termios. TTY descriptors are 5
while editing, 3 during generation, and 5 after reopening; repeated turns retain
the same count. Optional `YVEX_REPL_MEMCHECK=valgrind make test-repl` runs the real
chat process under a memory checker and fails on invalid accesses/definite leaks.

The existing tiny vertical also drives `yvex chat` against `yvex serve` and the
real compiled CPU decoder: input `a` produces `okokok`, advances session position
to 5, returns to the prompt, resets through a product command, and repeats. This
is bounded executable composition evidence, not model quality or a large-model
performance result. No weights are downloaded for terminal qualification.

## Alternatives considered

Runtime shared linkage was unnecessary for the current executable package.
Vendoring, copying the binding, a local editor fallback and a toy acceptance
consumer would not prove external ownership. Extending REPLAI with product
semantics was rejected; ABI 1 expresses this cutover without a library change.
