# Operating DeepSeek-V4-Flash

DeepSeek-V4-Flash is the first complete hosted YVEX vertical. Family identities
and architecture facts live in the
[DeepSeek technical record](../model-families/deepseek-v4-flash.md); current
gates live in [`ROADMAP.md`](../../ROADMAP.md).

## Product path

Export the admitted DeepSeek artifact and binding, then follow the explicit
first-start procedure in the [operator runbook](../operator-runbook.md). That
procedure starts `yvexd` directly; configured model defaults are optional and
are not artifact admission.

After `runtime.ready`, observe and use the same host from two additional
terminals:

```sh
./yvex runtime watch
./yvex chat --session main
```

The daemon opens one model and retains the complete encoded model payload in
one immutable process-lifetime host arena together with tokenizer, attention,
materialization, output-head, and plan resources. Each named session owns
independent DeepSeek persistent state and exact prompt/token continuation.

On turn two, the host renders and encodes the complete expected conversation,
proves that the committed token ledger is its exact prefix, and prefills only
the new suffix. An incompatible prefix refuses; reset is explicit.

## One-shot path

```sh
./yvex run --strategy stochastic --temperature 0.8 --top-k 50 --top-p 0.95 --min-p 0.05 --typical-p 1.0 --seed 42 "Explain attention in one sentence."
```

Sampling remains common-host even when Transformer, MoE, persistent state, and
output-head projection execute on CUDA. Streamed fragments are sent only after
sampled-token decode commit and incremental detokenization commit.

## Application-provider path

The same daemon exposes the bounded loopback profile without loading another
DeepSeek model. After `runtime.ready`, verify it:

```sh
curl -fsS http://127.0.0.1:8001/health
```

Discover the exact admitted model identifier with one command:

```sh
curl -fsS http://127.0.0.1:8001/v1/models
```

OpenAI Python and JavaScript SDKs use
`base_url=http://127.0.0.1:8001/v1`. Chat Completions and Responses translate
typed messages through the DeepSeek tokenizer/prompt owner; the adapter never
constructs DeepSeek control-token syntax. Function tools return typed calls for
the application to execute. See the
[bounded compatibility profile](../openai-compatibility.md).

## Offline engineering path

The engine-linked offline lane of `yvex` retains direct proof surfaces:

```sh
./yvex execute attention prepare --help
./yvex execute transformer run --help
./yvex execute transformer decode --help
./yvex execute transformer logits --help
./yvex execute transformer sample --help
./yvex execute transformer generate --help

./yvex inspect tokenizer --help
./yvex execute tokenizer encode --help
./yvex compile quant preset --help
./yvex compile quant plan --help
./yvex artifact materialize --help
```

These are finite engineering and conformance operations, not hosted-generation
aliases. They may open the engine directly but never create a persistent model
authority.

## Runtime evidence

For a hosted two-turn proof record:

- artifact, physical variant, binding, model, session, and turn identities;
- model/artifact open and residency build counts;
- first and second prompt token counts;
- exact reusable prefix and second-turn suffix counts;
- sampled-token/decode-input equality;
- final positions and persistent-state digests;
- TTFT, prefill/decode timing and memory snapshots;
- cancellation, detach/reconnect, reset, and shutdown results.

Logs, traces, prompts, model output, artifacts, bindings, and model files remain
untracked external operator assets.

## Non-claims

Hosted generation and local compatibility evidence are not model-quality
evaluation, full-model benchmark evidence, release qualification, full OpenAI
service equivalence, public serving, remote security, continuous batching, MTP,
or speculative execution.
