# Changelog

Externally meaningful changes remain under **Unreleased** until release
qualification and a version tag are accepted. This is a consolidated public
change record, not an implementation diary; Git preserves intermediate
protocol migrations, experiments, and refactor chronology. Current gate state
belongs to [ROADMAP](ROADMAP.md).

## Unreleased

### Model and source lifecycle

- Provider-neutral discovery and exact-revision inspection, including Hugging
  Face search and product acquisition of one coherent Safetensors or GGUF
  representation. Consolidated and sharded alternatives are not blindly
  downloaded together; dry-run and resumable operation control remain distinct
  from preparation.
- Separate acquired source, immutable package, READY deployment, and loaded
  engine facts in the logical model catalog. Preparation cannot promote a
  source-only family to executable support.
- DeepSeek-V4-Flash-DSpark source-to-hosted text with target-only and
  target-verified speculative execution under one model/state authority.
- Qwen's admitted hybrid attention/recurrent text specialization and MiniMax's
  staged composite media execution. Their different numerical and product
  evidence limits remain in the [family records](docs/model-families/integration.md#current-family-boundaries).
- Mamba2 exact acquired-source/role interpretation and portable transactional
  selective-SSD component execution. This is partial support: no complete
  decoder/artifact, READY deployment, or hosted generation.

### Hosting and application surfaces

- One foreground `yvex serve` host can start with zero engines and retain
  transport while independently loading and unloading exact engine generations.
  Multiple fitting engines, explicit ensure-active leases, directional
  capabilities, and `model active` expose authoritative runtime state.
- Engine-bound retained sessions, exact prefix reuse, partial-progress truth,
  cancellation, reset, bounded state checkpoints, and copy-on-write session
  fork. Live leases/sessions prevent premature engine retirement.
- Ordered typed multipart content with durable identity and derivation
  provenance; bounded local media references avoid JSON base64 expansion.
  Repeated `/attach PATH` staging in `yvex chat` preserves session identity.
  Unsupported input/model combinations fail before numerical execution.
- Registry-driven command grammar, discovery, advanced help, and shell
  completion. Bare `yvex` prints the product map; `yvex chat` is the explicit
  linear REPL and `host logs` is the shared operational event stream.
- Source-authored reasoning/final/tool channels, exact prompt policy, and
  committed-only output. The bounded loopback
  [OpenAI compatibility profile](docs/openai-compatibility.md) supports its
  declared Chat Completions and Responses subset; YVEX never executes tools.

### Execution and measurement

- Immutable package/storage meaning is separated from deployment-specific
  backend, representation, implementation, width, and crossover choices.
  Binding readers reject legacy records that cannot preserve that distinction.
- Typed transactional state unifies lifecycle without conflating KV,
  convolution, recurrent, draft, RNG, decoder, and media geometry.
  Compatible-operation batching and cooperative runnable concurrency remain
  distinct from global ready-sequence continuous batching.
- CUDA device-native logits, admitted greedy/stochastic selection and
  speculative correction, bounded result transfer, and session-stream
  synchronization. CPU and detailed reference paths remain explicit.
  These changes do not claim entirely device-side generation.
- Verified-reopen artifact leases and artifact-backed UMA addressability avoid
  treating repeated payload hashing or full anonymous copies as mandatory
  startup work. Composite components use the same admission owner.
- Resource reports distinguish mapping, proven addressability, explicit
  allocations, prepared resources, arenas/workspace, typed session state,
  current/peak memory, and process RSS. Unknown physical UMA residency is not
  converted into a false zero GPU working-set claim.
- Execution accounting publishes scoped phase/counter facts, cumulative and
  bounded rolling rates, unattributed/overlapping time, and bounded
  server-authored progress. Detailed profiling and operational observability
  remain distinct measurement configurations.
- Hardware-significant execution choices and kernel identities are bound to
  the admitted deployment. Performance changes require controlled
  characterization; no component optimization establishes release throughput.

### Compatibility and removals

- The [native protocol contract](docs/contracts/local-protocol.md) is v20.
  Its multipart/model-control wire semantics require matching peers;
  earlier fixed layouts are refused rather than reinterpreted.
- Installed ABI records have independent schema/layout identities, including
  server capacity and distinct engine-kind versus text-strategy facts.
  [C API contracts](docs/contracts/c-api.md) own exact versions and migration
  behavior; runtime bindings and package IR are separately versioned.
- Removed the separate daemon/developer/OpenAI executables, implicit
  bare-command chat, public one-shot generation, persisted implicit model
  selection, and stdin-driven server administration. Advanced source,
  artifact, compile, profile, and benchmark operations remain discoverable
  through `yvex help --advanced`; they are not another hosted runtime.
- The foreground host refuses duplicate startup instead of silently attaching
  as a client. Exact unsupported backend, strategy, or capability requests
  fail closed rather than changing execution.

### Security and release limits

The native socket and compatibility listener retain their local trust
boundary. Authentication, TLS, remote exposure, and full upstream API parity
are not provided by the compatibility profile. See [SECURITY](SECURITY.md).

Operational execution and software QA do not establish model quality,
release-path benchmarks, or release readiness. The
[release doctrine](docs/releases/doctrine.md) keeps those gates separate.
