# YVEX OpenAI Compatibility Profile v2

Status: normative implemented compatibility contract

Authority: HTTP request/response/SSE compatibility projected by
`src/server/openai/` and `src/provider/`. Producer: the loopback listener inside
`yvexd`. Consumers: explicitly configured local OpenAI-compatible clients. The
adapter follows the hosted runtime lifecycle and owns no model, session, KV,
worker, or telemetry authority.

`yvex.openai.compat.v2` is a bounded, local application-provider profile. It
adapts OpenAI-compatible HTTP/JSON/SSE requests to YVEX local protocol v7 and
the existing `yvexd` model host. It is not a claim of full OpenAI API or OpenAI
service equivalence.

The profile was audited on 2026-07-29 against the official OpenAI references
for [Models](https://developers.openai.com/api/reference/resources/models),
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events),
and [function calling](https://developers.openai.com/api/docs/guides/function-calling).
Those moving interfaces do not expand this explicitly versioned YVEX subset.

## Process and trust boundary

```text
application or SDK
  -> loopback HTTP/1.1
  -> yvexd OpenAI adapter
  -> provider-neutral request over YVEX protocol v7
  -> yvexd session and generation owners
```

The adapter is source-separated from runtime mathematics, opens no second model
or artifact, owns no KV, and executes no application tool. `yvexd` remains the
only model host and process. The adapter refuses non-loopback bind addresses;
authentication, TLS, CORS, and remote exposure are outside this profile.

When the selected runtime profile uses DSpark, the adapter receives only
target-verified committed fragments from the same server turn. Draft
candidates are never emitted over JSON/SSE and never enter compatibility
usage. This internal execution mode does not change compatibility profile v2.

Profile v2 maps `reasoning_effort` values `none`, `high`, and `max` to the
admitted source-authored model policies. Chat responses and deltas expose
explicit model-emitted text as `reasoning_content`; Responses objects use the
same documented YVEX field and the
`response.reasoning_content.delta`/`response.reasoning_content.done` events.
Final `content` remains separate. The adapter never emits `<think>` delimiters,
folds explicit reasoning into final content, infers reasoning from prose, or
exposes hidden runtime state. A profile or model without the typed capability
refuses `high` and `max`.

For ordinary multi-turn thinking requests, the prompt owner applies the
source-authored `drop_thinking` rule and omits prior assistant reasoning before
the last user turn. When tools are present, it retains reasoning and tool
continuity, merges tool results into the DeepSeek user block, and orders them
by the preceding call IDs. HTTP state stores typed fields; it never reconstructs
this history from rendered final text.

The normal registry-backed runtime start enables the default loopback listener.
Select a startup-ready model and start the host; the listener is prepared
before model admission and begins accepting requests only after
`runtime.ready`:

```sh
./yvex model list
./yvex model select deepseek4-v4-flash-dspark-runtime-iq2xxs
./yvex runtime start
```

The alias is illustrative and must be replaced by a startup-ready local entry.
Adapter-to-runtime frame I/O has a bounded 600000 ms default timeout; local
operators may override it with `--openai-timeout-ms` for their admitted workload.

SDKs use `base_url=http://127.0.0.1:8001/v1` and any local non-secret API-key
placeholder.

## Endpoint matrix

| Method and path | Profile status | YVEX mapping |
| --- | --- | --- |
| `GET /health` | supported, YVEX extension | adapter and runtime readiness |
| `GET /v1/models` | supported | loaded daemon model list containing one model |
| `GET /v1/models/{id}` | supported | exact loaded-model lookup |
| `POST /v1/chat/completions` | supported subset | ephemeral YVEX session and typed turn |
| `POST /v1/responses` | supported subset | typed turn plus bounded response-state mapping |

Embeddings, audio, images, files, batches, fine-tuning, moderation, Realtime,
hosted tools, legacy Assistants, and all other paths refuse.

## Chat Completions request profile

Supported fields are:

| Field | Accepted form |
| --- | --- |
| `model` | exact ID returned by `/v1/models` |
| `messages` | ordered `developer`, `system`, `user`, `assistant`, and `tool` messages; assistant history may carry `reasoning_content` and bounded typed tool calls |
| `stream` | boolean |
| `stream_options.include_usage` | boolean |
| `temperature` | finite value from 0 through 2 |
| `top_p` | finite value from 0 through 1 |
| `seed` | non-negative integer |
| `stop` | one string or at most four bounded strings |
| `max_tokens`, `max_completion_tokens` | positive bounded integer |
| `n` | exactly 1 |
| `reasoning_effort` | `none`, `high`, or `max` |
| `tools` | bounded function definitions |
| `tool_choice` | `none`, `auto`, `required`, or one named function |
| `parallel_tool_calls` | boolean; true admits multiple source-authored calls |
| `response_format` | `text` or `json_object` |

Unknown fields and unsupported known fields refuse; they are not ignored.
Multimodal content, `n > 1`, penalties, logprobs, modalities, prediction, and
service-tier controls are outside the profile.

A non-stream response contains one assistant choice, separate final and
reasoning content, an exact finish reason, prompt/completion/total usage, and a
YVEX completion-metrics object. A stream emits `chat.completion.chunk` objects,
an initial assistant-role delta, ordered `reasoning_content`, final-content, or
tool-call deltas, the terminal finish reason, an optional usage chunk, and
`data: [DONE]` only after successful completion.

## Responses request profile

Supported fields are:

| Field | Accepted form |
| --- | --- |
| `model` | exact loaded-model ID |
| `input` | text string or text/function-result item array |
| `instructions` | text instructions |
| `max_output_tokens` | positive bounded integer |
| `temperature`, `top_p` | same ranges as Chat Completions |
| `stream` | boolean |
| `reasoning_effort` | `none`, `high`, or `max` |
| `tools` | flat Responses function definitions |
| `tool_choice` | none, auto, required, or one named function |
| `parallel_tool_calls` | boolean |
| `previous_response_id` | live record created by this daemon instance |
| `store` | false only |
| `background` | false only |

The admitted streaming sequence uses the applicable subset of:

```text
response.created
response.output_item.added
response.content_part.added
response.reasoning_content.delta
response.output_text.delta
response.reasoning_content.done
response.output_text.done
response.content_part.done
response.function_call_arguments.delta
response.function_call_arguments.done
response.output_item.done
response.completed | response.incomplete | response.failed
```

Every event has an ordered `sequence_number`. `previous_response_id` names a
bounded in-memory adapter record tied to an existing YVEX session and model.
Records do not survive daemon restart and never reconstruct hidden state from
text. A successful continuation consumes the prior response ID and replaces it
with the returned successor ID; branching an already-mutated KV session is
refused rather than replayed implicitly.

`store=true`, background execution, hosted tools, web/file search, code
interpreter, computer use, image generation, MCP, multimodal input, and remote
conversation objects refuse.

## Function tools

The supported tool type is `function`. Definitions contain a name, optional
description, JSON-schema parameters, and `strict=false`. Profile v2 admits a
bounded ordered set of calls when `parallel_tool_calls=true`; call IDs must be
unique, and results are merged into the source-authored user/tool-result form
in call order.

YVEX returns a stable call ID, function name, and arguments that validate as a
JSON object. The application validates and executes the function, then returns
the result with the exact call ID. The tokenizer/family prompt owner renders
tool definitions, calls, and results; the HTTP adapter never invents model
control-token syntax.

YVEX never executes application tools.

`strict=true`, malformed or duplicate arguments, duplicate or unknown call
IDs, and unmatched named choices refuse. A prose suggestion to call a tool is
not promoted into a function-call object.

## Structured output and stop strings

`response_format={"type":"json_object"}` requests JSON through the admitted
prompt policy and validates the complete result as one JSON object with no
trailing non-whitespace bytes. Malformed JSON fails; the adapter never repairs
it. JSON Schema and constrained decoding are not part of profile v2.

Stop strings are bounded and matched across generated fragment boundaries
before provider publication. Matched bytes are suppressed from the application
result. Model-committed tokens remain committed and are never falsely rolled
back.

## Usage and finish mapping

| YVEX fact | Compatibility fact |
| --- | --- |
| EOS, tokenizer stop, or matched stop string | `stop` |
| maximum output tokens | `length` |
| typed function call | `tool_calls` |
| cancelled request | cancellation/error path |
| model, tokenizer, JSON, or output failure | typed error |

`prompt_tokens`, `completion_tokens`, and `total_tokens` retain their standard
meanings. `completion_tokens_details.reasoning_tokens` (or the Responses
equivalent) reports the reasoning subset. `yvex_completion_metrics` separately
reports reasoning/final counts and rates, time to first reasoning/final token,
phase durations, and total completion time/rate. Reused-prefix counts remain
YVEX telemetry facts and never replace prompt usage. DSpark proposal,
verification, accepted-prefix, and rejection counters remain YVEX telemetry
facts; `completion_tokens` counts only target-verified tokens committed by the
runtime.

## HTTP and error contract

The adapter accepts bounded HTTP/1.1 with an explicit `Content-Length`, bounded
headers and body, strict UTF-8 JSON, no duplicate keys, no trailing data, and no
silent type coercion. Transfer-encoded request bodies refuse. One connection
serves one request in profile v2.

Errors use the OpenAI-compatible envelope:

```json
{"error":{"message":"...","type":"...","param":null,"code":"..."}}
```

The admitted status classes include invalid request (400), model not found
(404), incompatible state (409), request too large (413), unsupported parameter
(422), queue full (429), internal failure (500), runtime unavailable (503), and
gateway timeout (504). Once SSE headers are committed, failure is represented
by the admitted terminal stream error/failure event rather than a fictional
replacement HTTP status.

Default adapter and daemon telemetry excludes prompt and response content.
Provider request, YVEX request, session, turn, endpoint, and stream facts remain
correlatable through typed identities.

## Verified clients and non-claims

Compatibility acceptance pins the official OpenAI Python and JavaScript SDK
versions recorded by the closure evidence and exercises standard `base_url`
configuration only. Bet-tennis is an unchanged external consumer used as
acceptance pressure; it is not an implementation dependency.

This profile does not establish full OpenAI API compatibility, Anthropic
compatibility, public serving, authentication, TLS, remote security, hosted
tool execution, model quality, benchmark evidence, or release qualification.
