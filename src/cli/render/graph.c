/* Render typed graph reports only through CLI I/O; projection never establishes readiness. */
#include "src/cli/render/private.h"
#include "src/cli/io/private.h"
#include "src/cli/model_artifacts/private.h"
#include <stddef.h>
#include <string.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/sampling.h>
static const char *const literal_lines_0[] = {
    "usage: yvex execute attention prepare --target TARGET",
    "           [--source DIR --source-manifest FILE]",
    "           [--physical-variant-plan FILE --quant-preset NAME|--quant-policy FILE]",
    "           [--imatrix-manifest FILE]",
    "       yvex inspect attention describe --target TARGET",
    "       yvex inspect attention capabilities --target TARGET --backend cpu|cuda",
    "       yvex inspect attention plan --target TARGET --backend cpu|cuda",
    "       yvex execute attention run --target TARGET --backend cpu|cuda",
    "       yvex execute attention compare --target TARGET",
    "       yvex inspect attention state|validate|exercise --target TARGET",
    "       yvex inspect attention residency --target TARGET --backend cpu|cuda",
    "       yvex execute attention capture|replay --target TARGET",
    "       yvex profile attention cuda-graph list|inspect|warmup|update|invalidate|release --target TARGET",
    "       yvex profile attention trace|profile|benchmark|qualify --target TARGET --backend cpu|cuda",
    "       yvex profile attention compare --baseline FILE --current FILE",
    "       yvex execute moe --target TARGET --artifact FILE --runtime-binding FILE",
    "           --backend cpu|cuda --input tensor-file --input-file FILE --scope full",
    "           token IDs are numeric routing input; they do not establish tokenizer support",
    "       yvex execute transformer run --target TARGET --artifact FILE --runtime-binding FILE",
    "           --backend cpu|cuda --phase prefill --input token-ids --input-file FILE",
    "           --chunk-tokens N --context-capacity N --progress off",
    "           numeric token IDs do not establish tokenizer, logits, decode, or generation support",
    "       yvex execute transformer decode --target TARGET --artifact FILE --runtime-binding FILE",
    "           --backend cpu|cuda --input token-ids --input-file FILE",
    "           --prefill-tokens N --prefill-chunk-tokens N --context-capacity N --progress off",
    "           teacher-forced numeric tokens do not establish logits, sampling, or generation",
    "       yvex execute transformer logits --target TARGET --artifact FILE --runtime-binding FILE",
    "           --backend cpu|cuda --input token-ids --input-file FILE",
    "           --prefill-tokens N --prefill-chunk-tokens N --context-capacity N --progress off",
    "           complete raw vocabulary logits do not establish sampling or generation",
    "       yvex execute transformer sample --target TARGET --artifact FILE --runtime-binding FILE",
    "           --strategy greedy | --strategy stochastic --seed N [sampling filters]",
    "           selected token IDs are not appended, decoded, detokenized, or generated",
    "       yvex execute transformer generate --target TARGET --artifact FILE --runtime-binding FILE",
    "           --backend cpu|cuda --user TEXT --max-new-tokens N --prefill-chunk-tokens N",
    "           [--models-root DIR] [--artifact FILE] [--runtime-binding FILE] [--runtime-binding-dir DIR]",
    "           [--backend cpu|cuda] [--phase prefill|decode|mixed|verify]",
    "           [--mode eager|piecewise|full|auto] [--scope quick|full]",
    "           [--operation-scope core|envelope|release-attention-set]",
    "           [--tokens N] [--warmup N] [--repeat N] [--trace-level none|summary|stages|full]",
    "           [--progress auto|plain|off] [--max-host-bytes N] [--max-device-bytes N]",
    "           [--require-mode] [--capture-bucket ID] [--baseline FILE] [--current FILE]",
    "           [--write-baseline]",
    "           [--chart PATH.svg]",
    "           benchmark compare: [--max-regression-bps N]",
    "           [--probe canonical] [--compare-backends] [--output normal|table|audit|json|csv]",
    "       bounded selectors: [--layer N] [--layer-start N --layer-count 1]",
    "           [--class swa|csa|hca] [--position N] [--history-tokens N]",
    "       activation prefill: [--input tensor-file --input-file FILE]",
    "           [--chunk-tokens N] [--context-capacity N]",
    "       reserved controls refuse until their typed runtime owners exist: [multi-layer ranges]",
    "           [--local-capacity N] [--compressed-capacity N] [--indexer-capacity N]",
    "",
    "example: yvex execute attention run --target deepseek4-v4-flash-dspark --backend cpu --scope quick",
    "boundary: attention commands execute canonical activations over admitted weights and session-persistent "
        "state; they are not prompt execution, transformer composition, or generation"
};
#define ATTENTION_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_graph_attention_operator_result, MEMBER), ""}
#define ATTENTION_TIMING(KEY, PHASE) \
    {KEY, YVEX_CLI_FIELD_DOUBLE, \
     offsetof(yvex_graph_attention_operator_result, lifecycle_seconds) + sizeof(double) * (PHASE), ""}
#define ATTENTION_BENCHMARK_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_graph_attention_operator_result, benchmark) + \
                    offsetof(yvex_runtime_benchmark_operator_summary, MEMBER), ""}
#define ATTENTION_PROBE_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_graph_attention_operator_result, probe) + \
                    offsetof(yvex_attention_probe_result, MEMBER), ""}
#define ATTENTION_CAPABILITY(KEY, MEMBER) \
    {KEY, YVEX_CLI_FIELD_BOOL, \
     offsetof(yvex_graph_attention_operator_result, capabilities) + \
         offsetof(yvex_runtime_capabilities, MEMBER), ""}
#define FIELD_COUNT(FIELDS) (sizeof(FIELDS) / sizeof((FIELDS)[0]))
#define MOE_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_moe_operator_result, MEMBER), ""}
#define MOE_EXECUTION_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_moe_operator_result, execution) + \
                    offsetof(yvex_runtime_moe_result, MEMBER), ""}
#define MOE_QTYPE_FIELD(KEY, QTYPE) \
    {KEY, YVEX_CLI_FIELD_U64, offsetof(yvex_moe_operator_result, execution) + \
         offsetof(yvex_runtime_moe_result, qtype_counts) + \
         (QTYPE) * sizeof(unsigned long long), ""}
static const yvex_cli_field_spec moe_fields[] = {
    MOE_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    MOE_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    MOE_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    MOE_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    MOE_FIELD("backend", YVEX_CLI_FIELD_TEXT_ARRAY, backend),
    MOE_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_identity),
    MOE_FIELD("runtime_binding_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_binding_identity),
    MOE_FIELD("runtime_descriptor_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_descriptor_identity),
    MOE_FIELD("runtime_numeric_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_numeric_identity),
    MOE_FIELD("moe_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, moe_plan_identity),
    MOE_FIELD("layers", YVEX_CLI_FIELD_U64, layer_count),
    MOE_FIELD("tokens", YVEX_CLI_FIELD_U64, token_count),
    MOE_FIELD("hash_router_layers", YVEX_CLI_FIELD_U64, hash_router_count),
    MOE_FIELD("learned_router_layers", YVEX_CLI_FIELD_U64, learned_router_count),
    MOE_FIELD("routed_experts", YVEX_CLI_FIELD_U64, routed_experts),
    MOE_FIELD("shared_experts", YVEX_CLI_FIELD_U64, shared_experts),
    MOE_FIELD("experts_per_token", YVEX_CLI_FIELD_U64, experts_per_token),
    MOE_EXECUTION_FIELD("layers_executed", YVEX_CLI_FIELD_U64, layers_executed),
    MOE_EXECUTION_FIELD("hash_router_executions", YVEX_CLI_FIELD_U64, hash_router_executions),
    MOE_EXECUTION_FIELD("learned_router_executions", YVEX_CLI_FIELD_U64, learned_router_executions),
    MOE_EXECUTION_FIELD("routed_expert_executions", YVEX_CLI_FIELD_U64, routed_expert_executions),
    MOE_EXECUTION_FIELD("shared_expert_executions", YVEX_CLI_FIELD_U64, shared_expert_executions),
    MOE_EXECUTION_FIELD("expert_subviews_accessed", YVEX_CLI_FIELD_U64, expert_subviews_accessed),
    MOE_EXECUTION_FIELD("encoded_bytes_read", YVEX_CLI_FIELD_U64, encoded_bytes_read),
    MOE_EXECUTION_FIELD("h2d_bytes", YVEX_CLI_FIELD_U64, host_to_device_bytes),
    MOE_EXECUTION_FIELD("d2h_bytes", YVEX_CLI_FIELD_U64, device_to_host_bytes),
    MOE_EXECUTION_FIELD("kernel_launches", YVEX_CLI_FIELD_U64, kernel_launches),
    MOE_EXECUTION_FIELD("upload_count", YVEX_CLI_FIELD_U64, upload_count),
    MOE_EXECUTION_FIELD("selected_expert_cache_hits", YVEX_CLI_FIELD_U64, cache_hits),
    MOE_EXECUTION_FIELD("selected_expert_cache_misses", YVEX_CLI_FIELD_U64, cache_misses),
    MOE_QTYPE_FIELD("qtype_f32_weight_accesses", YVEX_GGUF_QTYPE_F32),
    MOE_QTYPE_FIELD("qtype_q8_0_weight_accesses", YVEX_GGUF_QTYPE_Q8_0),
    MOE_QTYPE_FIELD("qtype_q2_k_weight_accesses", YVEX_GGUF_QTYPE_Q2_K),
    MOE_QTYPE_FIELD("qtype_iq2_xxs_weight_accesses", YVEX_GGUF_QTYPE_IQ2_XXS),
    MOE_QTYPE_FIELD("qtype_bf16_weight_accesses", YVEX_GGUF_QTYPE_BF16),
    MOE_QTYPE_FIELD("qtype_i32_weight_accesses", YVEX_GGUF_QTYPE_I32),
    MOE_EXECUTION_FIELD("input_identity", YVEX_CLI_FIELD_TEXT_ARRAY, input_identity),
    MOE_EXECUTION_FIELD("routing_digest", YVEX_CLI_FIELD_TEXT_ARRAY, routing_digest),
    MOE_EXECUTION_FIELD("routed_digest", YVEX_CLI_FIELD_TEXT_ARRAY, routed_digest),
    MOE_EXECUTION_FIELD("shared_digest", YVEX_CLI_FIELD_TEXT_ARRAY, shared_digest),
    MOE_EXECUTION_FIELD("combined_output_digest", YVEX_CLI_FIELD_TEXT_ARRAY, combined_output_digest),
    MOE_EXECUTION_FIELD("execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY, execution_identity),
    MOE_FIELD("moe_plan_ready", YVEX_CLI_FIELD_BOOL, moe_plan_ready),
    MOE_FIELD("moe_router_ready", YVEX_CLI_FIELD_BOOL, moe_router_ready),
    MOE_FIELD("moe_routed_expert_ready", YVEX_CLI_FIELD_BOOL, moe_routed_expert_ready),
    MOE_FIELD("moe_shared_expert_ready", YVEX_CLI_FIELD_BOOL, moe_shared_expert_ready),
    MOE_FIELD("moe_block_ready", YVEX_CLI_FIELD_BOOL, moe_block_ready),
    MOE_FIELD("moe_prefill_composed", YVEX_CLI_FIELD_BOOL, moe_prefill_composed),
    MOE_FIELD("moe_decode_composed", YVEX_CLI_FIELD_BOOL, moe_decode_composed),
    MOE_FIELD("transformer_ready", YVEX_CLI_FIELD_BOOL, transformer_ready),
    MOE_FIELD("generation_ready", YVEX_CLI_FIELD_BOOL, generation_ready),
    MOE_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
    MOE_FIELD("completed", YVEX_CLI_FIELD_BOOL, completed),
};
#define TRANSFORMER_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_transformer_operator_result, MEMBER), ""}
#define TRANSFORMER_EXECUTION_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_transformer_operator_result, execution) + \
                    offsetof(yvex_runtime_transformer_result, MEMBER), ""}
static const yvex_cli_field_spec transformer_fields[] = {
    TRANSFORMER_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    TRANSFORMER_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    TRANSFORMER_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    TRANSFORMER_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    TRANSFORMER_FIELD("backend", YVEX_CLI_FIELD_TEXT_ARRAY, backend),
    TRANSFORMER_FIELD("phase", YVEX_CLI_FIELD_TEXT_ARRAY, phase),
    TRANSFORMER_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_identity),
    TRANSFORMER_FIELD("runtime_binding_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_binding_identity),
    TRANSFORMER_FIELD("transformer_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, transformer_plan_identity),
    TRANSFORMER_EXECUTION_FIELD("input_identity", YVEX_CLI_FIELD_TEXT_ARRAY, input_identity),
    TRANSFORMER_EXECUTION_FIELD("token_start", YVEX_CLI_FIELD_U64, token_start),
    TRANSFORMER_EXECUTION_FIELD("token_count", YVEX_CLI_FIELD_U64, token_count),
    TRANSFORMER_EXECUTION_FIELD("chunk_count", YVEX_CLI_FIELD_U64, chunk_count),
    TRANSFORMER_EXECUTION_FIELD("committed_prefix", YVEX_CLI_FIELD_U64, committed_prefix),
    TRANSFORMER_EXECUTION_FIELD("position_before", YVEX_CLI_FIELD_U64, position_before),
    TRANSFORMER_EXECUTION_FIELD("position_after", YVEX_CLI_FIELD_U64, position_after),
    TRANSFORMER_EXECUTION_FIELD("generation_before", YVEX_CLI_FIELD_U64, generation_before),
    TRANSFORMER_EXECUTION_FIELD("generation_after", YVEX_CLI_FIELD_U64, generation_after),
    TRANSFORMER_FIELD("hidden_width", YVEX_CLI_FIELD_U64, hidden_width),
    TRANSFORMER_FIELD("expanded_width", YVEX_CLI_FIELD_U64, expanded_width),
    TRANSFORMER_FIELD("layers", YVEX_CLI_FIELD_U64, layer_count),
    TRANSFORMER_EXECUTION_FIELD("embedding_rows", YVEX_CLI_FIELD_U64, embedding_rows),
    TRANSFORMER_EXECUTION_FIELD("embedding_bytes", YVEX_CLI_FIELD_U64, embedding_bytes),
    TRANSFORMER_EXECUTION_FIELD("layers_executed", YVEX_CLI_FIELD_U64, layers_executed),
    TRANSFORMER_EXECUTION_FIELD("swa_layers", YVEX_CLI_FIELD_U64, swa_layers),
    TRANSFORMER_EXECUTION_FIELD("csa_layers", YVEX_CLI_FIELD_U64, csa_layers),
    TRANSFORMER_EXECUTION_FIELD("hca_layers", YVEX_CLI_FIELD_U64, hca_layers),
    TRANSFORMER_EXECUTION_FIELD("hash_router_executions", YVEX_CLI_FIELD_U64, hash_routers),
    TRANSFORMER_EXECUTION_FIELD("learned_router_executions", YVEX_CLI_FIELD_U64, learned_routers),
    TRANSFORMER_EXECUTION_FIELD("routed_expert_executions", YVEX_CLI_FIELD_U64, routed_experts),
    TRANSFORMER_EXECUTION_FIELD("shared_expert_executions", YVEX_CLI_FIELD_U64, shared_experts),
    TRANSFORMER_EXECUTION_FIELD("h2d_bytes", YVEX_CLI_FIELD_U64, h2d_bytes),
    TRANSFORMER_EXECUTION_FIELD("d2h_bytes", YVEX_CLI_FIELD_U64, d2h_bytes),
    TRANSFORMER_EXECUTION_FIELD("kernel_launches", YVEX_CLI_FIELD_U64, kernel_launches),
    TRANSFORMER_EXECUTION_FIELD("embedding_digest", YVEX_CLI_FIELD_TEXT_ARRAY, embedding_digest),
    TRANSFORMER_EXECUTION_FIELD("routing_digest", YVEX_CLI_FIELD_TEXT_ARRAY, routing_digest),
    TRANSFORMER_EXECUTION_FIELD("layer_digest", YVEX_CLI_FIELD_TEXT_ARRAY, layer_digest),
    TRANSFORMER_EXECUTION_FIELD("final_expanded_digest", YVEX_CLI_FIELD_TEXT_ARRAY, final_expanded_digest),
    TRANSFORMER_EXECUTION_FIELD("normalized_hidden_digest", YVEX_CLI_FIELD_TEXT_ARRAY, normalized_hidden_digest),
    TRANSFORMER_EXECUTION_FIELD("persistent_state_digest", YVEX_CLI_FIELD_TEXT_ARRAY, persistent_state_digest),
    TRANSFORMER_EXECUTION_FIELD("execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY, execution_identity),
    TRANSFORMER_FIELD("embedding_ready", YVEX_CLI_FIELD_BOOL, embedding_ready),
    TRANSFORMER_FIELD("transformer_plan_ready", YVEX_CLI_FIELD_BOOL, transformer_plan_ready),
    TRANSFORMER_FIELD("transformer_block_ready", YVEX_CLI_FIELD_BOOL, transformer_block_ready),
    TRANSFORMER_FIELD("transformer_stack_ready", YVEX_CLI_FIELD_BOOL, transformer_stack_ready),
    TRANSFORMER_FIELD("transformer_final_head_ready", YVEX_CLI_FIELD_BOOL, transformer_final_head_ready),
    TRANSFORMER_FIELD("transformer_final_norm_ready", YVEX_CLI_FIELD_BOOL, transformer_final_norm_ready),
    TRANSFORMER_FIELD("transformer_hidden_state_ready", YVEX_CLI_FIELD_BOOL, transformer_hidden_state_ready),
    TRANSFORMER_FIELD("full_model_prefill_ready", YVEX_CLI_FIELD_BOOL, full_model_prefill_ready),
    TRANSFORMER_FIELD("transformer_ready", YVEX_CLI_FIELD_BOOL, transformer_ready),
    TRANSFORMER_FIELD("single_token_transformer_component_ready", YVEX_CLI_FIELD_BOOL,
                      single_token_transformer_component_ready),
    TRANSFORMER_FIELD("model_decode_ready", YVEX_CLI_FIELD_BOOL, model_decode_ready),
    TRANSFORMER_FIELD("logits_ready", YVEX_CLI_FIELD_BOOL, logits_ready),
    TRANSFORMER_FIELD("sampling_ready", YVEX_CLI_FIELD_BOOL, sampling_ready),
    TRANSFORMER_FIELD("tokenizer_runtime_ready", YVEX_CLI_FIELD_BOOL, tokenizer_runtime_ready),
    TRANSFORMER_FIELD("generation_ready", YVEX_CLI_FIELD_BOOL, generation_ready),
    TRANSFORMER_FIELD("model_behavior_evaluation_ready", YVEX_CLI_FIELD_BOOL, model_behavior_evaluation_ready),
    TRANSFORMER_FIELD("release_qualification_ready", YVEX_CLI_FIELD_BOOL, release_qualification_ready),
    TRANSFORMER_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
    TRANSFORMER_FIELD("completed", YVEX_CLI_FIELD_BOOL, completed),
};
#define DECODE_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_decode_operator_result, MEMBER), ""}
#define DECODE_EXECUTION_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_decode_operator_result, decode) + \
                    offsetof(yvex_runtime_decode_result, MEMBER), ""}
static const yvex_cli_field_spec decode_fields[] = {
    DECODE_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    DECODE_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    DECODE_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    DECODE_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    DECODE_FIELD("backend", YVEX_CLI_FIELD_TEXT_ARRAY, backend),
    DECODE_FIELD("phase", YVEX_CLI_FIELD_TEXT_ARRAY, phase),
    DECODE_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_identity),
    DECODE_FIELD("runtime_binding_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_binding_identity),
    DECODE_FIELD("transformer_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, transformer_plan_identity),
    DECODE_FIELD("hidden_width", YVEX_CLI_FIELD_U64, hidden_width),
    DECODE_FIELD("layers", YVEX_CLI_FIELD_U64, layer_count),
    DECODE_FIELD("prefill_tokens_committed", YVEX_CLI_FIELD_U64, prefill_tokens_committed),
    DECODE_EXECUTION_FIELD("input_identity", YVEX_CLI_FIELD_TEXT_ARRAY, input_identity),
    DECODE_EXECUTION_FIELD("decode_steps_requested", YVEX_CLI_FIELD_U64, requested_steps),
    DECODE_EXECUTION_FIELD("decode_steps_completed", YVEX_CLI_FIELD_U64, completed_steps),
    DECODE_EXECUTION_FIELD("partial", YVEX_CLI_FIELD_BOOL, partial),
    DECODE_EXECUTION_FIELD("has_incomplete_step", YVEX_CLI_FIELD_BOOL, has_incomplete_step),
    DECODE_EXECUTION_FIELD("first_incomplete_step", YVEX_CLI_FIELD_U64, first_incomplete_step),
    DECODE_EXECUTION_FIELD("initial_committed_prefix", YVEX_CLI_FIELD_U64, initial_committed_prefix),
    DECODE_EXECUTION_FIELD("final_committed_prefix", YVEX_CLI_FIELD_U64, final_committed_prefix),
    DECODE_EXECUTION_FIELD("generation_before", YVEX_CLI_FIELD_U64, generation_before),
    DECODE_EXECUTION_FIELD("generation_after", YVEX_CLI_FIELD_U64, generation_after),
    DECODE_EXECUTION_FIELD("layers_executed", YVEX_CLI_FIELD_U64, layers_executed),
    DECODE_EXECUTION_FIELD("swa_layers", YVEX_CLI_FIELD_U64, swa_layers),
    DECODE_EXECUTION_FIELD("csa_layers", YVEX_CLI_FIELD_U64, csa_layers),
    DECODE_EXECUTION_FIELD("hca_layers", YVEX_CLI_FIELD_U64, hca_layers),
    DECODE_EXECUTION_FIELD("hash_router_executions", YVEX_CLI_FIELD_U64, hash_routers),
    DECODE_EXECUTION_FIELD("learned_router_executions", YVEX_CLI_FIELD_U64, learned_routers),
    DECODE_EXECUTION_FIELD("routed_expert_executions", YVEX_CLI_FIELD_U64, routed_experts),
    DECODE_EXECUTION_FIELD("shared_expert_executions", YVEX_CLI_FIELD_U64, shared_experts),
    DECODE_EXECUTION_FIELD("h2d_bytes", YVEX_CLI_FIELD_U64, h2d_bytes),
    DECODE_EXECUTION_FIELD("d2h_bytes", YVEX_CLI_FIELD_U64, d2h_bytes),
    DECODE_EXECUTION_FIELD("kernel_launches", YVEX_CLI_FIELD_U64, kernel_launches),
    DECODE_EXECUTION_FIELD("aggregate_hidden_digest", YVEX_CLI_FIELD_TEXT_ARRAY, aggregate_hidden_digest),
    DECODE_EXECUTION_FIELD("aggregate_state_digest", YVEX_CLI_FIELD_TEXT_ARRAY, aggregate_state_digest),
    DECODE_EXECUTION_FIELD("decode_execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY, decode_execution_identity),
    DECODE_FIELD("decode_step_ready", YVEX_CLI_FIELD_BOOL, decode_step_ready),
    DECODE_FIELD("decode_repeat_ready", YVEX_CLI_FIELD_BOOL, decode_repeat_ready),
    DECODE_FIELD("decode_hidden_state_ready", YVEX_CLI_FIELD_BOOL, decode_hidden_state_ready),
    DECODE_FIELD("decode_partial_progress_ready", YVEX_CLI_FIELD_BOOL, decode_partial_progress_ready),
    DECODE_FIELD("moe_decode_composed", YVEX_CLI_FIELD_BOOL, moe_decode_composed),
    DECODE_FIELD("model_decode_ready", YVEX_CLI_FIELD_BOOL, model_decode_ready),
    DECODE_FIELD("logits_ready", YVEX_CLI_FIELD_BOOL, logits_ready),
    DECODE_FIELD("output_head_ready", YVEX_CLI_FIELD_BOOL, output_head_ready),
    DECODE_FIELD("sampling_ready", YVEX_CLI_FIELD_BOOL, sampling_ready),
    DECODE_FIELD("tokenizer_runtime_ready", YVEX_CLI_FIELD_BOOL, tokenizer_runtime_ready),
    DECODE_FIELD("generation_ready", YVEX_CLI_FIELD_BOOL, generation_ready),
    DECODE_FIELD("model_behavior_evaluation_ready", YVEX_CLI_FIELD_BOOL, model_behavior_evaluation_ready),
    DECODE_FIELD("full_model_benchmark_ready", YVEX_CLI_FIELD_BOOL, full_model_benchmark_ready),
    DECODE_FIELD("release_qualification_ready", YVEX_CLI_FIELD_BOOL, release_qualification_ready),
    DECODE_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
    DECODE_FIELD("completed", YVEX_CLI_FIELD_BOOL, completed),
};
#define LOGITS_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_logits_operator_result, MEMBER), ""}
#define LOGITS_PLAN_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_logits_operator_result, plan) + \
                    offsetof(yvex_runtime_logits_plan_summary, MEMBER), ""}
#define LOGITS_EXECUTION_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_logits_operator_result, execution) + \
                    offsetof(yvex_runtime_logits_result, MEMBER), ""}
static const yvex_cli_field_spec logits_fields[] = {
    LOGITS_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    LOGITS_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    LOGITS_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    LOGITS_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    LOGITS_FIELD("backend", YVEX_CLI_FIELD_TEXT_ARRAY, backend),
    LOGITS_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_identity),
    LOGITS_FIELD("runtime_binding_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                 runtime_binding_identity),
    LOGITS_FIELD("transformer_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                 transformer_plan_identity),
    LOGITS_PLAN_FIELD("output_head_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                      output_head_plan_identity),
    LOGITS_PLAN_FIELD("output_head_tensor_id", YVEX_CLI_FIELD_U64,
                      output_head_tensor_id),
    LOGITS_PLAN_FIELD("output_head_qtype", YVEX_CLI_FIELD_I32, qtype),
    LOGITS_PLAN_FIELD("hidden_width", YVEX_CLI_FIELD_U64, hidden_width),
    LOGITS_PLAN_FIELD("vocabulary_size", YVEX_CLI_FIELD_U64, vocabulary_size),
    LOGITS_PLAN_FIELD("output_head_rows", YVEX_CLI_FIELD_U64, row_count),
    LOGITS_PLAN_FIELD("output_head_columns", YVEX_CLI_FIELD_U64, row_width),
    LOGITS_PLAN_FIELD("output_head_row_bytes", YVEX_CLI_FIELD_U64, row_bytes),
    LOGITS_PLAN_FIELD("output_head_encoded_bytes", YVEX_CLI_FIELD_U64, encoded_bytes),
    LOGITS_FIELD("output_head_host_resident_bytes", YVEX_CLI_FIELD_U64,
                 output_head_host_bytes),
    LOGITS_FIELD("output_head_device_resident_bytes", YVEX_CLI_FIELD_U64,
                 output_head_device_bytes),
    LOGITS_FIELD("output_head_upload_bytes", YVEX_CLI_FIELD_U64,
                 output_head_upload_bytes),
    LOGITS_FIELD("output_head_upload_count", YVEX_CLI_FIELD_U64,
                 output_head_upload_count),
    LOGITS_FIELD("prefill_logits_rows", YVEX_CLI_FIELD_U64, prefill_logits_rows),
    LOGITS_FIELD("decode_logits_rows", YVEX_CLI_FIELD_U64, decode_logits_rows),
    LOGITS_EXECUTION_FIELD("logits_rows_requested", YVEX_CLI_FIELD_U64, requested_rows),
    LOGITS_EXECUTION_FIELD("logits_rows_completed", YVEX_CLI_FIELD_U64, completed_rows),
    LOGITS_EXECUTION_FIELD("first_incomplete_row", YVEX_CLI_FIELD_U64,
                           first_incomplete_row),
    LOGITS_EXECUTION_FIELD("final_source_position", YVEX_CLI_FIELD_U64,
                           final_source_position),
    LOGITS_EXECUTION_FIELD("partial", YVEX_CLI_FIELD_BOOL, partial),
    LOGITS_EXECUTION_FIELD("aggregate_logits_digest", YVEX_CLI_FIELD_TEXT_ARRAY,
                           aggregate_logits_digest),
    LOGITS_EXECUTION_FIELD("logits_execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                           execution_identity),
    LOGITS_FIELD("output_head_binding_ready", YVEX_CLI_FIELD_BOOL,
                 output_head_binding_ready),
    LOGITS_FIELD("output_head_residency_ready", YVEX_CLI_FIELD_BOOL,
                 output_head_residency_ready),
    LOGITS_FIELD("logits_cpu_ready", YVEX_CLI_FIELD_BOOL, logits_cpu_ready),
    LOGITS_FIELD("logits_cuda_ready", YVEX_CLI_FIELD_BOOL, logits_cuda_ready),
    LOGITS_FIELD("logits_prefill_ready", YVEX_CLI_FIELD_BOOL, logits_prefill_ready),
    LOGITS_FIELD("logits_decode_ready", YVEX_CLI_FIELD_BOOL, logits_decode_ready),
    LOGITS_FIELD("logits_full_vocabulary_ready", YVEX_CLI_FIELD_BOOL,
                 logits_full_vocabulary_ready),
    LOGITS_FIELD("logits_hidden_contract_ready", YVEX_CLI_FIELD_BOOL,
                 logits_hidden_contract_ready),
    LOGITS_FIELD("logits_partial_progress_ready", YVEX_CLI_FIELD_BOOL,
                 logits_partial_progress_ready),
    LOGITS_FIELD("logits_ready", YVEX_CLI_FIELD_BOOL, logits_ready),
    LOGITS_FIELD("sampling_ready", YVEX_CLI_FIELD_BOOL, sampling_ready),
    LOGITS_FIELD("tokenizer_runtime_ready", YVEX_CLI_FIELD_BOOL,
                 tokenizer_runtime_ready),
    LOGITS_FIELD("generation_ready", YVEX_CLI_FIELD_BOOL, generation_ready),
    LOGITS_FIELD("model_behavior_evaluation_ready", YVEX_CLI_FIELD_BOOL,
                 model_behavior_evaluation_ready),
    LOGITS_FIELD("full_model_benchmark_ready", YVEX_CLI_FIELD_BOOL,
                 full_model_benchmark_ready),
    LOGITS_FIELD("release_qualification_ready", YVEX_CLI_FIELD_BOOL,
                 release_qualification_ready),
    LOGITS_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
    LOGITS_FIELD("completed", YVEX_CLI_FIELD_BOOL, completed),
};
#define SAMPLE_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_sampling_operator_result, MEMBER), ""}
#define SAMPLE_POLICY(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_sampling_operator_result, policy) + \
                    offsetof(yvex_runtime_sampling_policy, MEMBER), ""}
#define SAMPLE_EXEC(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_sampling_operator_result, execution) + \
                    offsetof(yvex_runtime_sampling_execution, MEMBER), ""}
#define SAMPLE_BOOL(MEMBER) SAMPLE_FIELD(#MEMBER, YVEX_CLI_FIELD_BOOL, MEMBER)
static const yvex_cli_field_spec sampling_fields[] = {
    SAMPLE_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    SAMPLE_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    SAMPLE_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    SAMPLE_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    SAMPLE_FIELD("logits_backend", YVEX_CLI_FIELD_TEXT_ARRAY, logits_backend),
    SAMPLE_FIELD("sampling_execution_kind", YVEX_CLI_FIELD_TEXT_ARRAY, sampling_execution_kind),
    SAMPLE_FIELD("samples", YVEX_CLI_FIELD_U64, sample_count),
    SAMPLE_FIELD("prefill_samples", YVEX_CLI_FIELD_U64, prefill_samples),
    SAMPLE_FIELD("decode_samples", YVEX_CLI_FIELD_U64, decode_samples),
    SAMPLE_FIELD("strategy", YVEX_CLI_FIELD_TEXT_ARRAY, strategy),
    SAMPLE_POLICY("temperature", YVEX_CLI_FIELD_DOUBLE, temperature),
    SAMPLE_POLICY("top_k", YVEX_CLI_FIELD_U64, top_k),
    SAMPLE_POLICY("top_p", YVEX_CLI_FIELD_DOUBLE, top_p),
    SAMPLE_POLICY("min_p", YVEX_CLI_FIELD_DOUBLE, min_p),
    SAMPLE_POLICY("typical_p", YVEX_CLI_FIELD_DOUBLE, typical_p),
    SAMPLE_POLICY("seed", YVEX_CLI_FIELD_U64, seed),
    SAMPLE_POLICY("rng_algorithm", YVEX_CLI_FIELD_U32, rng_algorithm),
    SAMPLE_POLICY("rng_version", YVEX_CLI_FIELD_U32, rng_version),
    SAMPLE_POLICY("filter_order_version", YVEX_CLI_FIELD_U32, filter_order_version),
    SAMPLE_POLICY("policy_identity", YVEX_CLI_FIELD_TEXT_ARRAY, policy_identity),
    SAMPLE_EXEC("sampling_requested_samples", YVEX_CLI_FIELD_U64, requested_samples),
    SAMPLE_EXEC("sampling_completed_samples", YVEX_CLI_FIELD_U64, completed_samples),
    SAMPLE_EXEC("sampling_first_incomplete_sample", YVEX_CLI_FIELD_U64, first_incomplete_sample),
    SAMPLE_EXEC("sampling_partial", YVEX_CLI_FIELD_BOOL, partial),
    SAMPLE_EXEC("sampling_identity", YVEX_CLI_FIELD_TEXT_ARRAY, aggregate_sampling_identity),
    SAMPLE_FIELD("workspace_bytes", YVEX_CLI_FIELD_U64, workspace_bytes),
    SAMPLE_FIELD("workspace_generation", YVEX_CLI_FIELD_U64, workspace_generation),
    SAMPLE_FIELD("warm_sampling_allocations", YVEX_CLI_FIELD_U64, warm_workspace_allocations),
    SAMPLE_BOOL(sampling_source_contract_ready), SAMPLE_BOOL(sampling_policy_ready), SAMPLE_BOOL(sampling_greedy_ready),
    SAMPLE_BOOL(sampling_temperature_ready),
    SAMPLE_BOOL(sampling_top_k_ready), SAMPLE_BOOL(sampling_top_p_ready),
    SAMPLE_BOOL(sampling_min_p_ready), SAMPLE_BOOL(sampling_typical_ready),
    SAMPLE_BOOL(sampling_stochastic_ready), SAMPLE_BOOL(sampling_seed_reproducibility_ready),
    SAMPLE_BOOL(sampling_real_logits_ready), SAMPLE_BOOL(sampling_partial_progress_ready),
    SAMPLE_BOOL(sampling_ready), SAMPLE_BOOL(persistent_state_unchanged),
    SAMPLE_BOOL(token_append_ready), SAMPLE_BOOL(tokenizer_runtime_ready),
    SAMPLE_BOOL(eos_policy_ready), SAMPLE_BOOL(stop_policy_ready),
    SAMPLE_BOOL(detokenization_ready), SAMPLE_BOOL(generation_ready),
    SAMPLE_BOOL(cuda_sampling_ready), SAMPLE_BOOL(model_behavior_evaluation_ready),
    SAMPLE_BOOL(full_model_benchmark_ready), SAMPLE_BOOL(release_qualification_ready),
    SAMPLE_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
    SAMPLE_FIELD("completed", YVEX_CLI_FIELD_BOOL, completed),
};
static const yvex_cli_field_spec attention_base_fields[] = {
    ATTENTION_FIELD("command", YVEX_CLI_FIELD_TEXT_ARRAY, command),
    ATTENTION_FIELD("status", YVEX_CLI_FIELD_TEXT_ARRAY, status),
    ATTENTION_FIELD("target", YVEX_CLI_FIELD_TEXT_ARRAY, target),
    ATTENTION_FIELD("backend", YVEX_CLI_FIELD_TEXT_ARRAY, backend),
    ATTENTION_FIELD("scope", YVEX_CLI_FIELD_TEXT_ARRAY, scope),
    ATTENTION_FIELD("operation_scope", YVEX_CLI_FIELD_TEXT_ARRAY, operation_scope),
    ATTENTION_FIELD("phase", YVEX_CLI_FIELD_TEXT_ARRAY, phase),
    ATTENTION_FIELD("trace_policy", YVEX_CLI_FIELD_TEXT_ARRAY, trace_policy),
    ATTENTION_FIELD("requested_mode", YVEX_CLI_FIELD_TEXT_ARRAY, requested_mode),
    ATTENTION_FIELD("selected_mode", YVEX_CLI_FIELD_TEXT_ARRAY, selected_mode),
    ATTENTION_FIELD("selection_reason", YVEX_CLI_FIELD_TEXT_ARRAY, selection_reason),
    ATTENTION_FIELD("artifact_path", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_path),
    ATTENTION_FIELD("runtime_binding_path", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_binding_path),
};
static const yvex_cli_field_spec attention_target_fields[] = {
    ATTENTION_FIELD("family", YVEX_CLI_FIELD_TEXT_ARRAY, family),
    ATTENTION_FIELD("input_class", YVEX_CLI_FIELD_TEXT_ARRAY, input_class),
};
static const yvex_cli_field_spec attention_admission_fields[] = {
    ATTENTION_FIELD("execution_class", YVEX_CLI_FIELD_TEXT_ARRAY, execution_class),
    ATTENTION_FIELD("weights_class", YVEX_CLI_FIELD_TEXT_ARRAY, weights_class),
    ATTENTION_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_identity),
    ATTENTION_FIELD("runtime_binding_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_binding_identity),
    ATTENTION_FIELD("runtime_model_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_model_identity),
    ATTENTION_FIELD("activation_input_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                    activation_input_identity),
    ATTENTION_FIELD("artifact_bytes_hashed", YVEX_CLI_FIELD_U64, artifact_bytes_hashed),
    ATTENTION_FIELD("artifact_identity_verified", YVEX_CLI_FIELD_BOOL, artifact_identity_verified),
    ATTENTION_FIELD("materialization_identity", YVEX_CLI_FIELD_TEXT_ARRAY, materialization_identity),
    ATTENTION_FIELD("logical_model_identity", YVEX_CLI_FIELD_TEXT_ARRAY, logical_model_identity),
    ATTENTION_FIELD("runtime_numeric_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_numeric_identity),
    ATTENTION_FIELD("runtime_descriptor_identity", YVEX_CLI_FIELD_TEXT_ARRAY, runtime_descriptor_identity),
    ATTENTION_FIELD("attention_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, attention_plan_identity),
    ATTENTION_FIELD("semantic_graph_identity", YVEX_CLI_FIELD_TEXT_ARRAY, semantic_graph_identity),
    ATTENTION_FIELD("executable_graph_identity", YVEX_CLI_FIELD_TEXT_ARRAY, executable_graph_identity),
    ATTENTION_FIELD("main_layers_total", YVEX_CLI_FIELD_U64, main_layers_total),
    ATTENTION_FIELD("bindings_total", YVEX_CLI_FIELD_U64, bindings_total),
    ATTENTION_CAPABILITY("attention_execution_supported", attention_core_ready),
    ATTENTION_FIELD("attention_cuda_execution_ready", YVEX_CLI_FIELD_BOOL, attention_cuda_execution_ready),
    ATTENTION_CAPABILITY("attention_state_delta_ready", attention_state_delta_ready),
    ATTENTION_CAPABILITY("attention_weight_residency_ready", attention_weight_residency_ready),
    ATTENTION_CAPABILITY("attention_trace_ready", attention_trace_ready),
    ATTENTION_CAPABILITY("attention_profile_ready", attention_profile_ready),
    ATTENTION_CAPABILITY("attention_benchmark_ready", attention_benchmark_ready),
};
static const yvex_cli_field_spec attention_capability_fields[] = {
    ATTENTION_CAPABILITY("attention_semantics_ready", attention_semantics_ready),
    ATTENTION_CAPABILITY("attention_core_ready", attention_core_ready),
    ATTENTION_CAPABILITY("attention_envelope_ready", attention_envelope_ready),
    ATTENTION_CAPABILITY("cpu_prefill_eager_ready", cpu_prefill_eager_ready),
    ATTENTION_CAPABILITY("cpu_decode_eager_ready", cpu_decode_eager_ready),
    ATTENTION_CAPABILITY("cuda_prefill_eager_ready", cuda_prefill_eager_ready),
    ATTENTION_CAPABILITY("cuda_decode_eager_ready", cuda_decode_eager_ready),
    ATTENTION_CAPABILITY("cuda_prefill_piecewise_graph_ready", cuda_prefill_piecewise_graph_ready),
    ATTENTION_CAPABILITY("cuda_decode_piecewise_graph_ready", cuda_decode_piecewise_graph_ready),
    ATTENTION_CAPABILITY("cuda_prefill_full_graph_ready", cuda_prefill_full_graph_ready),
    ATTENTION_CAPABILITY("cuda_decode_full_graph_ready", cuda_decode_full_graph_ready),
    ATTENTION_CAPABILITY("attention_workspace_ready", attention_workspace_ready),
    ATTENTION_CAPABILITY("attention_operator_ready", attention_operator_ready),
    ATTENTION_CAPABILITY("mixed_attention_ready", mixed_attention_ready),
    ATTENTION_CAPABILITY("speculative_attention_ready", speculative_attention_ready),
};
static const yvex_cli_field_spec attention_execution_fields[] = {
    ATTENTION_FIELD("execution_descriptor_identity", YVEX_CLI_FIELD_TEXT_ARRAY, execution_descriptor_identity),
    ATTENTION_PROBE_FIELD("attention_execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY, attention_execution_identity),
    ATTENTION_PROBE_FIELD("layers_executed", YVEX_CLI_FIELD_U64, layers_executed),
    ATTENTION_PROBE_FIELD("bindings_executed", YVEX_CLI_FIELD_U64, bindings_executed),
    ATTENTION_PROBE_FIELD("swa_layers_executed", YVEX_CLI_FIELD_U64, swa_layers_executed),
    ATTENTION_PROBE_FIELD("csa_layers_executed", YVEX_CLI_FIELD_U64, csa_layers_executed),
    ATTENTION_PROBE_FIELD("hca_layers_executed", YVEX_CLI_FIELD_U64, hca_layers_executed),
    ATTENTION_PROBE_FIELD("topk_selected", YVEX_CLI_FIELD_U64, topk_selected),
    ATTENTION_PROBE_FIELD("hca_ratio", YVEX_CLI_FIELD_U64, hca_ratio),
    ATTENTION_FIELD("warmup_count", YVEX_CLI_FIELD_U64, warmup_count),
    ATTENTION_FIELD("repeat_count", YVEX_CLI_FIELD_U64, repeat_count),
    ATTENTION_PROBE_FIELD("tensor_output_digest", YVEX_CLI_FIELD_TEXT_ARRAY, tensor_output_digest),
    ATTENTION_PROBE_FIELD("state_delta_digest", YVEX_CLI_FIELD_TEXT_ARRAY, state_delta_digest),
    ATTENTION_FIELD("execution_evidence_digest", YVEX_CLI_FIELD_TEXT_ARRAY, execution_evidence_digest),
    ATTENTION_FIELD("execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY, execution_identity),
    ATTENTION_FIELD("execution_dispatch_count", YVEX_CLI_FIELD_U64, execution_dispatch_count),
    ATTENTION_FIELD("prefill_chunk_count", YVEX_CLI_FIELD_U64, prefill_chunk_count),
    ATTENTION_FIELD("committed_prefix", YVEX_CLI_FIELD_U64, committed_prefix),
    ATTENTION_FIELD("trace_stage_count", YVEX_CLI_FIELD_U64, trace_stage_count),
    ATTENTION_FIELD("trace_value_count", YVEX_CLI_FIELD_U64, trace_value_count),
};
static const yvex_cli_field_spec attention_state_fields[] = {
    ATTENTION_FIELD("state_layout_identity", YVEX_CLI_FIELD_TEXT_ARRAY, state_layout_identity),
    ATTENTION_FIELD("state_content_identity", YVEX_CLI_FIELD_TEXT_ARRAY, state_content_identity),
    ATTENTION_FIELD("state_residency_identity", YVEX_CLI_FIELD_TEXT_ARRAY, state_residency_identity),
    ATTENTION_FIELD("state_layer_count", YVEX_CLI_FIELD_U64, state_layer_count),
    ATTENTION_FIELD("state_prepared_layer_count", YVEX_CLI_FIELD_U64, state_prepared_layer_count),
    ATTENTION_FIELD("state_allocated_bytes", YVEX_CLI_FIELD_U64, state_allocated_bytes),
    ATTENTION_FIELD("state_capacity", YVEX_CLI_FIELD_U64, state_capacity),
    ATTENTION_FIELD("state_committed_sequence_length", YVEX_CLI_FIELD_U64,
                    state_committed_sequence_length),
    ATTENTION_FIELD("state_next_position", YVEX_CLI_FIELD_U64, state_next_position),
    ATTENTION_FIELD("state_generation", YVEX_CLI_FIELD_U64, state_generation),
    ATTENTION_FIELD("state_residency_generation", YVEX_CLI_FIELD_U64,
                    state_residency_generation),
    ATTENTION_FIELD("state_device_bytes", YVEX_CLI_FIELD_U64, state_device_bytes),
    ATTENTION_FIELD("state_upload_bytes", YVEX_CLI_FIELD_U64, state_upload_bytes),
    ATTENTION_FIELD("state_upload_count", YVEX_CLI_FIELD_U64, state_upload_count),
    ATTENTION_FIELD("state_commit_count", YVEX_CLI_FIELD_U64, state_commit_count),
    ATTENTION_FIELD("state_abort_count", YVEX_CLI_FIELD_U64, state_abort_count),
    ATTENTION_FIELD("state_cancellation_count", YVEX_CLI_FIELD_U64, state_cancellation_count),
    ATTENTION_FIELD("state_reset_count", YVEX_CLI_FIELD_U64, state_reset_count),
    ATTENTION_FIELD("state_sealed", YVEX_CLI_FIELD_BOOL, state_sealed),
    ATTENTION_FIELD("state_persistent", YVEX_CLI_FIELD_BOOL, state_persistent),
    ATTENTION_FIELD("state_position_consistent", YVEX_CLI_FIELD_BOOL,
                    state_position_consistent),
    ATTENTION_FIELD("state_cuda_ready", YVEX_CLI_FIELD_BOOL, state_cuda_ready),
    ATTENTION_FIELD("state_transaction_active", YVEX_CLI_FIELD_BOOL, state_transaction_active),
    ATTENTION_FIELD("state_validation_passed", YVEX_CLI_FIELD_BOOL, state_validation_passed),
    ATTENTION_FIELD("state_read_after_write_verified", YVEX_CLI_FIELD_BOOL,
                    state_read_after_write_verified),
    ATTENTION_FIELD("state_clear_reuse_verified", YVEX_CLI_FIELD_BOOL,
                    state_clear_reuse_verified),
};
static const yvex_cli_field_spec attention_runtime_fields[] = {
    ATTENTION_FIELD("artifact_hash_passes", YVEX_CLI_FIELD_U64, artifact_hash_passes),
    ATTENTION_FIELD("warm_artifact_hash_passes", YVEX_CLI_FIELD_U64, warm_artifact_hash_passes),
    ATTENTION_FIELD("runtime_source_headers_read", YVEX_CLI_FIELD_U64, runtime_source_headers_read),
    ATTENTION_FIELD("runtime_source_payload_bytes_read", YVEX_CLI_FIELD_U64, runtime_source_payload_bytes_read),
    ATTENTION_FIELD("runtime_transform_plans_built", YVEX_CLI_FIELD_U64, runtime_transform_plans_built),
    ATTENTION_FIELD("runtime_quant_plans_built", YVEX_CLI_FIELD_U64, runtime_quant_plans_built),
    ATTENTION_FIELD("runtime_writer_plans_built", YVEX_CLI_FIELD_U64, runtime_writer_plans_built),
    ATTENTION_FIELD("runtime_model_builds", YVEX_CLI_FIELD_U64, runtime_model_builds),
    ATTENTION_FIELD("runtime_descriptor_builds", YVEX_CLI_FIELD_U64, runtime_descriptor_builds),
    ATTENTION_FIELD("semantic_graph_builds", YVEX_CLI_FIELD_U64, semantic_graph_builds),
    ATTENTION_FIELD("executable_graph_builds", YVEX_CLI_FIELD_U64, executable_graph_builds),
};
static const yvex_cli_field_spec attention_timing_fields[] = {
    ATTENTION_TIMING("artifact_open_seconds", YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN),
    ATTENTION_TIMING("artifact_hash_seconds", YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH),
    ATTENTION_TIMING("artifact_admission_seconds", YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION),
    ATTENTION_TIMING("binding_open_seconds", YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN),
    ATTENTION_TIMING("materialization_open_seconds", YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN),
    ATTENTION_TIMING("runtime_model_seal_seconds", YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL),
    ATTENTION_TIMING("resident_weight_prepare_seconds", YVEX_RUNTIME_LIFECYCLE_RESIDENCY),
    ATTENTION_TIMING("backend_open_seconds", YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN),
    ATTENTION_TIMING("workspace_prepare_seconds", YVEX_RUNTIME_LIFECYCLE_WORKSPACE_PREPARE),
    ATTENTION_TIMING("graph_warmup_seconds", YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP),
    ATTENTION_TIMING("graph_capture_seconds", YVEX_RUNTIME_LIFECYCLE_GRAPH_CAPTURE),
    ATTENTION_TIMING("graph_instantiate_seconds", YVEX_RUNTIME_LIFECYCLE_GRAPH_INSTANTIATE),
    ATTENTION_TIMING("execution_seconds", YVEX_RUNTIME_LIFECYCLE_EXECUTION),
    ATTENTION_TIMING("publication_seconds", YVEX_RUNTIME_LIFECYCLE_PUBLICATION),
    ATTENTION_TIMING("cleanup_seconds", YVEX_RUNTIME_LIFECYCLE_CLEANUP),
    ATTENTION_FIELD("total_seconds", YVEX_CLI_FIELD_DOUBLE, total_seconds),
};
static const yvex_cli_field_spec attention_residency_fields[] = {
    ATTENTION_FIELD("residency_identity", YVEX_CLI_FIELD_TEXT_ARRAY, residency_identity),
    ATTENTION_FIELD("workspace_identity", YVEX_CLI_FIELD_TEXT_ARRAY, workspace_identity),
    ATTENTION_FIELD("resident_binding_count", YVEX_CLI_FIELD_U64, resident_binding_count),
    ATTENTION_FIELD("resident_encoded_bytes", YVEX_CLI_FIELD_U64, resident_encoded_bytes),
    ATTENTION_FIELD("host_resident_bytes", YVEX_CLI_FIELD_U64, host_resident_bytes),
    ATTENTION_FIELD("device_resident_bytes", YVEX_CLI_FIELD_U64, device_resident_bytes),
    ATTENTION_FIELD("workspace_bytes", YVEX_CLI_FIELD_U64, workspace_bytes),
    ATTENTION_FIELD("pinned_host_bytes", YVEX_CLI_FIELD_U64, pinned_host_bytes),
    ATTENTION_FIELD("pinned_host_peak_bytes", YVEX_CLI_FIELD_U64, pinned_host_peak_bytes),
    ATTENTION_FIELD("pinned_host_residency", YVEX_CLI_FIELD_BOOL, pinned_host_residency),
    ATTENTION_FIELD("upload_bytes", YVEX_CLI_FIELD_U64, upload_bytes),
    ATTENTION_FIELD("upload_count", YVEX_CLI_FIELD_U64, upload_count),
    ATTENTION_FIELD("warm_weight_artifact_reads", YVEX_CLI_FIELD_U64, warm_weight_artifact_reads),
    ATTENTION_FIELD("warm_weight_upload_bytes", YVEX_CLI_FIELD_U64, warm_weight_upload_bytes),
    ATTENTION_FIELD("warm_h2d_bytes", YVEX_CLI_FIELD_U64, warm_h2d_bytes),
    ATTENTION_FIELD("warm_d2h_bytes", YVEX_CLI_FIELD_U64, warm_d2h_bytes),
    ATTENTION_FIELD("warm_host_allocations", YVEX_CLI_FIELD_U64, warm_host_allocations),
    ATTENTION_FIELD("warm_device_allocations", YVEX_CLI_FIELD_U64, warm_device_allocations),
    ATTENTION_FIELD("warm_device_frees", YVEX_CLI_FIELD_U64, warm_device_frees),
};
static const yvex_cli_field_spec attention_benchmark_fields[] = {
    ATTENTION_FIELD("benchmark_sample_count", YVEX_CLI_FIELD_U64, benchmark_sample_count),
    ATTENTION_FIELD("benchmark_first_execution_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_first_execution_seconds),
    ATTENTION_FIELD("benchmark_minimum_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
    ATTENTION_FIELD("benchmark_p50_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_P50]),
    ATTENTION_FIELD("benchmark_p90_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_P90]),
    ATTENTION_FIELD("benchmark_p95_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_P95]),
    ATTENTION_FIELD("benchmark_p99_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_P99]),
    ATTENTION_FIELD("benchmark_maximum_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
    ATTENTION_FIELD("benchmark_mean_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_host_seconds[YVEX_RUNTIME_BENCHMARK_MEAN]),
    ATTENTION_FIELD("benchmark_standard_deviation_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_host_seconds[
                    YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]),
    ATTENTION_FIELD("benchmark_device_timing_available", YVEX_CLI_FIELD_BOOL, benchmark_device_timing_available),
    ATTENTION_BENCHMARK_FIELD("benchmark_identity", YVEX_CLI_FIELD_TEXT_ARRAY, identity),
    ATTENTION_BENCHMARK_FIELD("benchmark_current_commit", YVEX_CLI_FIELD_TEXT_ARRAY, current_commit),
    ATTENTION_BENCHMARK_FIELD("benchmark_current_source_state", YVEX_CLI_FIELD_TEXT_ARRAY, current_source_state),
};
static const yvex_cli_field_spec attention_benchmark_device_fields[] = {
    ATTENTION_FIELD("benchmark_device_minimum_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
    ATTENTION_FIELD("benchmark_device_p50_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_P50]),
    ATTENTION_FIELD("benchmark_device_p90_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_P90]),
    ATTENTION_FIELD("benchmark_device_p95_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_P95]),
    ATTENTION_FIELD("benchmark_device_p99_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_P99]),
    ATTENTION_FIELD("benchmark_device_maximum_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
    ATTENTION_FIELD("benchmark_device_mean_seconds", YVEX_CLI_FIELD_DOUBLE,
                    benchmark_device_seconds[YVEX_RUNTIME_BENCHMARK_MEAN]),
    ATTENTION_FIELD("benchmark_device_standard_deviation_seconds", YVEX_CLI_FIELD_DOUBLE, benchmark_device_seconds[
                    YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]),
};
static const yvex_cli_field_spec attention_benchmark_publication_fields[] = {
    ATTENTION_BENCHMARK_FIELD("benchmark_path", YVEX_CLI_FIELD_TEXT_ARRAY, path),
    ATTENTION_BENCHMARK_FIELD("benchmark_baseline_written", YVEX_CLI_FIELD_BOOL, baseline_written),
    ATTENTION_BENCHMARK_FIELD("benchmark_file_bytes", YVEX_CLI_FIELD_U64, file_bytes),
};
static const yvex_cli_field_spec attention_benchmark_baseline_fields[] = {
    ATTENTION_BENCHMARK_FIELD("benchmark_baseline_identity", YVEX_CLI_FIELD_TEXT_ARRAY, baseline_identity),
    ATTENTION_BENCHMARK_FIELD("benchmark_baseline_commit", YVEX_CLI_FIELD_TEXT_ARRAY, baseline_commit),
    ATTENTION_BENCHMARK_FIELD("benchmark_baseline_source_state", YVEX_CLI_FIELD_TEXT_ARRAY, baseline_source_state),
    ATTENTION_BENCHMARK_FIELD("benchmark_path", YVEX_CLI_FIELD_TEXT_ARRAY, path),
    ATTENTION_BENCHMARK_FIELD("benchmark_baseline_compatible", YVEX_CLI_FIELD_BOOL, baseline_compatible),
    ATTENTION_BENCHMARK_FIELD("benchmark_cold_delta_seconds", YVEX_CLI_FIELD_DOUBLE, cold_delta_seconds),
    ATTENTION_BENCHMARK_FIELD("benchmark_minimum_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
    ATTENTION_BENCHMARK_FIELD("benchmark_p50_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_P50]),
    ATTENTION_BENCHMARK_FIELD("benchmark_p90_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_P90]),
    ATTENTION_BENCHMARK_FIELD("benchmark_p95_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_P95]),
    ATTENTION_BENCHMARK_FIELD("benchmark_p99_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_P99]),
    ATTENTION_BENCHMARK_FIELD("benchmark_maximum_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
    ATTENTION_BENCHMARK_FIELD("benchmark_mean_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    host_delta_seconds[YVEX_RUNTIME_BENCHMARK_MEAN]),
};
static const yvex_cli_field_spec attention_benchmark_device_baseline_fields[] = {
    ATTENTION_BENCHMARK_FIELD("benchmark_device_minimum_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_p50_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_P50]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_p90_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_P90]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_p95_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_P95]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_p99_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_P99]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_maximum_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_mean_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[YVEX_RUNTIME_BENCHMARK_MEAN]),
    ATTENTION_BENCHMARK_FIELD("benchmark_device_standard_deviation_delta_seconds", YVEX_CLI_FIELD_DOUBLE,
                    device_delta_seconds[
                                  YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]),
};
static const yvex_cli_field_spec attention_benchmark_chart_fields[] = {
    ATTENTION_BENCHMARK_FIELD("benchmark_chart_generated", YVEX_CLI_FIELD_BOOL, chart_generated),
    ATTENTION_BENCHMARK_FIELD("benchmark_chart_path", YVEX_CLI_FIELD_TEXT_ARRAY, chart_path),
    ATTENTION_BENCHMARK_FIELD("benchmark_chart_identity", YVEX_CLI_FIELD_TEXT_ARRAY, chart_identity),
    ATTENTION_BENCHMARK_FIELD("benchmark_chart_file_bytes", YVEX_CLI_FIELD_U64, chart_file_bytes),
};
static const yvex_cli_field_spec attention_benchmark_context_fields[] = {
    ATTENTION_FIELD("benchmark_scope", YVEX_CLI_FIELD_TEXT_ARRAY, benchmark_scope),
    ATTENTION_FIELD("attention_class", YVEX_CLI_FIELD_TEXT_ARRAY, attention_class),
    ATTENTION_FIELD("requested_token_count", YVEX_CLI_FIELD_U64, requested_token_count),
    ATTENTION_FIELD("requested_history_tokens", YVEX_CLI_FIELD_U64, requested_history_tokens),
    ATTENTION_FIELD("requested_layer_start", YVEX_CLI_FIELD_U64, requested_layer_start),
    ATTENTION_FIELD("requested_layer_count", YVEX_CLI_FIELD_U64, requested_layer_count),
    ATTENTION_FIELD("component_benchmark_status", YVEX_CLI_FIELD_TEXT_ARRAY,
                    quality_status[YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK]),
    ATTENTION_FIELD("correctness_status", YVEX_CLI_FIELD_TEXT_ARRAY, quality_status[YVEX_RUNTIME_QUALITY_CORRECTNESS]),
    ATTENTION_FIELD("structural_runtime_status", YVEX_CLI_FIELD_TEXT_ARRAY,
                    quality_status[YVEX_RUNTIME_QUALITY_STRUCTURAL]),
    ATTENTION_FIELD("performance_status", YVEX_CLI_FIELD_TEXT_ARRAY, quality_status[YVEX_RUNTIME_QUALITY_PERFORMANCE]),
    ATTENTION_FIELD("benchmark_correctness_precondition_passed", YVEX_CLI_FIELD_BOOL,
                    benchmark_correctness_precondition_passed),
    ATTENTION_FIELD("benchmark_runtime_precondition_passed", YVEX_CLI_FIELD_BOOL,
                    benchmark_runtime_precondition_passed),
};
static const yvex_cli_field_spec attention_benchmark_regression_fields[] = {
    ATTENTION_BENCHMARK_FIELD("benchmark_regression_policy_enabled", YVEX_CLI_FIELD_BOOL,
                              regression_policy_enabled),
    ATTENTION_BENCHMARK_FIELD("benchmark_regression_policy_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                              regression_policy_identity),
    ATTENTION_BENCHMARK_FIELD("benchmark_comparison_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                              comparison_identity),
    ATTENTION_BENCHMARK_FIELD("benchmark_regression_basis_points", YVEX_CLI_FIELD_U64,
                              regression_basis_points),
    ATTENTION_BENCHMARK_FIELD("benchmark_performance_passed", YVEX_CLI_FIELD_BOOL,
                              performance_passed),
};
static const yvex_cli_field_spec attention_qualification_fields[] = {
    ATTENTION_FIELD("qualification_identity", YVEX_CLI_FIELD_TEXT_ARRAY, qualification_identity),
    ATTENTION_FIELD("quality_matrix_identity", YVEX_CLI_FIELD_TEXT_ARRAY, quality_matrix_identity),
    ATTENTION_FIELD("software_contract_status", YVEX_CLI_FIELD_TEXT_ARRAY,
                    quality_status[YVEX_RUNTIME_QUALITY_SOFTWARE]),
    ATTENTION_FIELD("numerical_conformance_status", YVEX_CLI_FIELD_TEXT_ARRAY,
                    quality_status[YVEX_RUNTIME_QUALITY_NUMERICAL]),
    ATTENTION_FIELD("runtime_qualification_status", YVEX_CLI_FIELD_TEXT_ARRAY,
                    quality_status[YVEX_RUNTIME_QUALITY_QUALIFICATION]),
    ATTENTION_FIELD("model_behavior_evaluation_available", YVEX_CLI_FIELD_BOOL, model_behavior_evaluation_available),
    ATTENTION_FIELD("model_quality_evaluation_available", YVEX_CLI_FIELD_BOOL, model_quality_evaluation_available),
    ATTENTION_FIELD("agent_runtime_available", YVEX_CLI_FIELD_BOOL, agent_runtime_available),
    ATTENTION_FIELD("agent_evaluation_available", YVEX_CLI_FIELD_BOOL, agent_evaluation_available),
    ATTENTION_FIELD("release_qualification_available", YVEX_CLI_FIELD_BOOL, release_qualification_available),
};
static const yvex_cli_field_spec attention_cuda_fields[] = {
    ATTENTION_PROBE_FIELD("cuda_device", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_device),
    ATTENTION_FIELD("cuda_driver", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_driver),
    ATTENTION_FIELD("cuda_build_identity", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_build_identity),
    ATTENTION_FIELD("capture_bucket", YVEX_CLI_FIELD_TEXT_ARRAY, capture_bucket),
    ATTENTION_PROBE_FIELD("compute_capability_major", YVEX_CLI_FIELD_I32, cuda_compute_capability_major),
    ATTENTION_PROBE_FIELD("compute_capability_minor", YVEX_CLI_FIELD_I32, cuda_compute_capability_minor),
    ATTENTION_PROBE_FIELD("kernel_launches", YVEX_CLI_FIELD_U64, kernel_launches),
    ATTENTION_PROBE_FIELD("peak_device_bytes", YVEX_CLI_FIELD_U64, peak_device_bytes),
    ATTENTION_PROBE_FIELD("h2d_bytes", YVEX_CLI_FIELD_U64, h2d_bytes),
    ATTENTION_PROBE_FIELD("d2h_bytes", YVEX_CLI_FIELD_U64, d2h_bytes),
    ATTENTION_FIELD("cuda_launch_graph_identity", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_launch_graph_identity),
    ATTENTION_FIELD("cuda_graph_exec_identity", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_graph_exec_identity),
    ATTENTION_FIELD("cuda_graph_count", YVEX_CLI_FIELD_U64, cuda_graph_count),
    ATTENTION_FIELD("cuda_graph_piece_count", YVEX_CLI_FIELD_U64, cuda_graph_piece_count),
    ATTENTION_FIELD("cuda_graph_capture_count", YVEX_CLI_FIELD_U64, cuda_graph_capture_count),
    ATTENTION_FIELD("cuda_graph_instantiate_count", YVEX_CLI_FIELD_U64, cuda_graph_instantiate_count),
    ATTENTION_FIELD("cuda_graph_replay_count", YVEX_CLI_FIELD_U64, cuda_graph_replay_count),
    ATTENTION_FIELD("cuda_graph_launch_count", YVEX_CLI_FIELD_U64, cuda_graph_launch_count),
    ATTENTION_FIELD("cuda_graph_node_count", YVEX_CLI_FIELD_U64, cuda_graph_node_count),
    ATTENTION_FIELD("cuda_graph_kernel_node_count", YVEX_CLI_FIELD_U64, cuda_graph_kernel_node_count),
    ATTENTION_FIELD("cuda_graph_memcpy_node_count", YVEX_CLI_FIELD_U64, cuda_graph_memcpy_node_count),
    ATTENTION_FIELD("cuda_graph_memset_node_count", YVEX_CLI_FIELD_U64, cuda_graph_memset_node_count),
    ATTENTION_FIELD("cuda_graph_update_count", YVEX_CLI_FIELD_U64, cuda_graph_update_count),
    ATTENTION_FIELD("cuda_graph_update_pending_count", YVEX_CLI_FIELD_U64, cuda_graph_update_pending_count),
    ATTENTION_FIELD("cuda_graph_registry_scope", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_graph_registry_scope),
    ATTENTION_FIELD("cuda_graph_registry_count", YVEX_CLI_FIELD_U64, cuda_graph_registry_count),
    ATTENTION_FIELD("cuda_graph_registry_index", YVEX_CLI_FIELD_U64, cuda_graph_registry_index),
    ATTENTION_FIELD("cuda_graph_registry_affected_count", YVEX_CLI_FIELD_U64, cuda_graph_registry_affected_count),
    ATTENTION_FIELD("cuda_graph_entry_compatibility_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                    cuda_graph_entry_compatibility_identity),
    ATTENTION_FIELD("cuda_graph_entry_state", YVEX_CLI_FIELD_I32, cuda_graph_entry_state),
    ATTENTION_FIELD("cuda_graph_entry_reason", YVEX_CLI_FIELD_I32, cuda_graph_entry_reason),
    ATTENTION_FIELD("cuda_graph_entry_capture_mode", YVEX_CLI_FIELD_I32, cuda_graph_entry_capture_mode),
    ATTENTION_FIELD("cuda_graph_entry_uploaded", YVEX_CLI_FIELD_BOOL, cuda_graph_entry_uploaded),
    ATTENTION_FIELD("cuda_graph_entry_update_requested", YVEX_CLI_FIELD_BOOL, cuda_graph_entry_update_requested),
    ATTENTION_FIELD("cuda_graph_capture_elapsed_ns", YVEX_CLI_FIELD_U64, cuda_graph_capture_elapsed_ns),
    ATTENTION_FIELD("cuda_graph_instantiate_elapsed_ns", YVEX_CLI_FIELD_U64, cuda_graph_instantiate_elapsed_ns),
    ATTENTION_FIELD("cuda_graph_last_update_elapsed_ns", YVEX_CLI_FIELD_U64, cuda_graph_last_update_elapsed_ns),
    ATTENTION_FIELD("cuda_graph_last_replay_elapsed_ns", YVEX_CLI_FIELD_U64, cuda_graph_last_replay_elapsed_ns),
    ATTENTION_FIELD("cuda_graph_invalidation_count", YVEX_CLI_FIELD_U64, cuda_graph_invalidation_count),
};
static const yvex_cli_field_spec attention_comparison_fields[] = {
    ATTENTION_PROBE_FIELD("cpu_output_digest", YVEX_CLI_FIELD_TEXT_ARRAY, cpu_output_digest),
    ATTENTION_PROBE_FIELD("cuda_output_digest", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_output_digest),
    ATTENTION_PROBE_FIELD("cpu_state_delta_digest", YVEX_CLI_FIELD_TEXT_ARRAY, cpu_state_delta_digest),
    ATTENTION_PROBE_FIELD("cuda_state_delta_digest", YVEX_CLI_FIELD_TEXT_ARRAY, cuda_state_delta_digest),
    ATTENTION_PROBE_FIELD("comparison_contract_identity", YVEX_CLI_FIELD_TEXT_ARRAY, comparison_contract_identity),
    ATTENTION_PROBE_FIELD("comparison_values", YVEX_CLI_FIELD_U64, comparison_values),
    ATTENTION_PROBE_FIELD("comparison_output_values", YVEX_CLI_FIELD_U64, comparison_output_values),
    ATTENTION_PROBE_FIELD("comparison_state_values", YVEX_CLI_FIELD_U64, comparison_state_values),
    ATTENTION_PROBE_FIELD("comparison_finite_values", YVEX_CLI_FIELD_U64, comparison_finite_values),
    ATTENTION_PROBE_FIELD("comparison_nonfinite_values", YVEX_CLI_FIELD_U64, comparison_nonfinite_values),
    ATTENTION_PROBE_FIELD("comparison_maximum_absolute_error", YVEX_CLI_FIELD_DOUBLE,
                    comparison_maximum_absolute_error),
    ATTENTION_PROBE_FIELD("comparison_maximum_relative_error", YVEX_CLI_FIELD_DOUBLE,
                    comparison_maximum_relative_error),
    ATTENTION_PROBE_FIELD("comparison_rmse", YVEX_CLI_FIELD_DOUBLE, comparison_rmse),
    ATTENTION_PROBE_FIELD("comparison_passed", YVEX_CLI_FIELD_BOOL, comparison_passed),
    ATTENTION_PROBE_FIELD("output_digest_equal", YVEX_CLI_FIELD_BOOL, output_digest_equal),
    ATTENTION_PROBE_FIELD("state_delta_digest_equal", YVEX_CLI_FIELD_BOOL, state_delta_digest_equal),
    ATTENTION_PROBE_FIELD("bitwise_equality_observed", YVEX_CLI_FIELD_BOOL, bitwise_equality_observed),
    ATTENTION_PROBE_FIELD("bitwise_equality_required", YVEX_CLI_FIELD_BOOL, bitwise_equality_required),
};
static const yvex_cli_field_spec attention_failure_fields[] = {
    ATTENTION_FIELD("first_failing_stage", YVEX_CLI_FIELD_TEXT_ARRAY, first_failing_stage),
    ATTENTION_PROBE_FIELD("first_failing_layer", YVEX_CLI_FIELD_U64, first_failing_layer),
    ATTENTION_PROBE_FIELD("first_failing_coordinate", YVEX_CLI_FIELD_U64, first_failing_coordinate),
};
static const yvex_cli_field_spec attention_provenance_fields[] = {
    ATTENTION_FIELD("source_snapshot_identity", YVEX_CLI_FIELD_TEXT_ARRAY, source_snapshot_identity),
    ATTENTION_FIELD("payload_identity", YVEX_CLI_FIELD_TEXT_ARRAY, payload_identity),
    ATTENTION_FIELD("artifact_transform_identity", YVEX_CLI_FIELD_TEXT_ARRAY, artifact_transform_identity),
    ATTENTION_FIELD("logical_transform_identity", YVEX_CLI_FIELD_TEXT_ARRAY, logical_transform_identity),
    ATTENTION_PROBE_FIELD("payload_bytes_read", YVEX_CLI_FIELD_U64, payload_bytes_read),
};
static const yvex_cli_field_spec attention_compatibility_fields[] = {
    ATTENTION_FIELD("current_writer_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, current_writer_plan_identity),
    ATTENTION_FIELD("payload_plan_identity", YVEX_CLI_FIELD_TEXT_ARRAY, payload_plan_identity),
    ATTENTION_FIELD("payload_byte_identity", YVEX_CLI_FIELD_TEXT_ARRAY, payload_byte_identity),
    ATTENTION_FIELD("physical_payload_compatible", YVEX_CLI_FIELD_BOOL, physical_payload_compatible),
    ATTENTION_FIELD("artifact_rebuild_required", YVEX_CLI_FIELD_BOOL, artifact_rebuild_required),
    ATTENTION_FIELD("materialization_rebuild_required", YVEX_CLI_FIELD_BOOL, materialization_rebuild_required),
    ATTENTION_FIELD("tensor_inventory_equal", YVEX_CLI_FIELD_BOOL, tensor_inventory_equal),
    ATTENTION_FIELD("qtype_equal", YVEX_CLI_FIELD_BOOL, qtype_equal),
    ATTENTION_FIELD("layout_equal", YVEX_CLI_FIELD_BOOL, layout_equal),
    ATTENTION_FIELD("offset_equal", YVEX_CLI_FIELD_BOOL, offset_equal),
    ATTENTION_FIELD("payload_digest_equal", YVEX_CLI_FIELD_BOOL, payload_digest_equal),
};
static const yvex_cli_field_spec attention_reachability_fields[] = {
    ATTENTION_FIELD("operator_command_available", YVEX_CLI_FIELD_BOOL, operator_command_available),
    ATTENTION_FIELD("production_api_available", YVEX_CLI_FIELD_BOOL, production_api_available),
    ATTENTION_FIELD("internal_live_runner_available", YVEX_CLI_FIELD_BOOL, internal_live_runner_available),
    ATTENTION_FIELD("end_user_generation_available", YVEX_CLI_FIELD_BOOL, end_user_generation_available),
};
static const yvex_cli_field_spec attention_reason_field[] = {
    ATTENTION_FIELD("failure_code", YVEX_CLI_FIELD_TEXT_ARRAY, failure_code),
    ATTENTION_FIELD("failure_where", YVEX_CLI_FIELD_TEXT_ARRAY, failure_where),
    ATTENTION_FIELD("reason", YVEX_CLI_FIELD_TEXT_ARRAY, reason),
};
static const yvex_cli_field_spec attention_final_field[] = {
    ATTENTION_FIELD("activation_prefill_ready", YVEX_CLI_FIELD_BOOL,
                    activation_prefill_ready),
    ATTENTION_FIELD("prefill_persistent_state_ready", YVEX_CLI_FIELD_BOOL,
                    prefill_persistent_state_ready),
    ATTENTION_FIELD("full_model_prefill_ready", YVEX_CLI_FIELD_BOOL,
                    full_model_prefill_ready),
    ATTENTION_CAPABILITY("persistent_kv_ready", persistent_kv_ready),
    ATTENTION_CAPABILITY("transformer_ready", transformer_ready),
    ATTENTION_CAPABILITY("runtime_generation_ready", generation_ready),
};
typedef enum {
    ATTENTION_FIELDS_ALWAYS = 1u << 0,
    ATTENTION_FIELDS_TARGET = 1u << 1,
    ATTENTION_FIELDS_ADMITTED = 1u << 2,
    ATTENTION_FIELDS_COMPLETED = 1u << 3,
    ATTENTION_FIELDS_CUDA = 1u << 4,
    ATTENTION_FIELDS_COMPARISON = 1u << 5,
    ATTENTION_FIELDS_COMPARISON_FAILURE = 1u << 6,
    ATTENTION_FIELDS_DETAILED_ADMISSION = 1u << 7,
    ATTENTION_FIELDS_COMPATIBILITY = 1u << 8,
    ATTENTION_FIELDS_TIMING = 1u << 9,
    ATTENTION_FIELDS_RESIDENCY = 1u << 10,
    ATTENTION_FIELDS_STATE = 1u << 11,
    ATTENTION_FIELDS_BENCHMARK = 1u << 12,
    ATTENTION_FIELDS_BENCHMARK_PUBLICATION = 1u << 13,
    ATTENTION_FIELDS_BENCHMARK_BASELINE = 1u << 14,
    ATTENTION_FIELDS_BENCHMARK_CHART = 1u << 15,
    ATTENTION_FIELDS_REASON = 1u << 16,
    ATTENTION_FIELDS_BENCHMARK_DEVICE = 1u << 17,
    ATTENTION_FIELDS_BENCHMARK_BASELINE_DEVICE = 1u << 18,
    ATTENTION_FIELDS_BENCHMARK_CONTEXT = 1u << 19,
    ATTENTION_FIELDS_QUALIFICATION = 1u << 20,
    ATTENTION_FIELDS_BENCHMARK_REGRESSION = 1u << 21
} attention_field_condition;
typedef struct {
    const yvex_cli_field_spec *fields;
    size_t count;
    attention_field_condition condition;
    int final;
} attention_field_group;
typedef enum {
    ATTENTION_PRESENCE_TEXT,
    ATTENTION_PRESENCE_TARGET,
    ATTENTION_PRESENCE_BOOL,
    ATTENTION_PRESENCE_U64,
    ATTENTION_PRESENCE_DOUBLE,
    ATTENTION_PRESENCE_COMPARISON_FAILURE
} attention_presence_kind;
typedef struct {
    attention_field_condition condition;
    size_t offset;
    attention_presence_kind kind;
    int detailed;
} attention_presence_rule;
#define ATTENTION_GROUP(FIELDS, CONDITION) \
    {FIELDS, FIELD_COUNT(FIELDS), CONDITION, 0}
static const attention_field_group attention_field_groups[] = {
    ATTENTION_GROUP(attention_base_fields, ATTENTION_FIELDS_ALWAYS),
    ATTENTION_GROUP(attention_target_fields, ATTENTION_FIELDS_TARGET),
    ATTENTION_GROUP(attention_admission_fields, ATTENTION_FIELDS_ADMITTED),
    ATTENTION_GROUP(attention_capability_fields, ATTENTION_FIELDS_ADMITTED),
    ATTENTION_GROUP(attention_execution_fields, ATTENTION_FIELDS_COMPLETED),
    ATTENTION_GROUP(attention_runtime_fields, ATTENTION_FIELDS_ADMITTED),
    ATTENTION_GROUP(attention_timing_fields, ATTENTION_FIELDS_TIMING),
    ATTENTION_GROUP(attention_residency_fields, ATTENTION_FIELDS_RESIDENCY),
    ATTENTION_GROUP(attention_state_fields, ATTENTION_FIELDS_STATE),
    ATTENTION_GROUP(attention_benchmark_fields, ATTENTION_FIELDS_BENCHMARK),
    ATTENTION_GROUP(attention_benchmark_device_fields,
                    ATTENTION_FIELDS_BENCHMARK_DEVICE),
    ATTENTION_GROUP(attention_benchmark_publication_fields,
                    ATTENTION_FIELDS_BENCHMARK_PUBLICATION),
    ATTENTION_GROUP(attention_benchmark_baseline_fields,
                    ATTENTION_FIELDS_BENCHMARK_BASELINE),
    ATTENTION_GROUP(attention_benchmark_device_baseline_fields,
                    ATTENTION_FIELDS_BENCHMARK_BASELINE_DEVICE),
    ATTENTION_GROUP(attention_benchmark_chart_fields, ATTENTION_FIELDS_BENCHMARK_CHART),
    ATTENTION_GROUP(attention_benchmark_context_fields,
                    ATTENTION_FIELDS_BENCHMARK_CONTEXT),
    ATTENTION_GROUP(attention_benchmark_regression_fields,
                    ATTENTION_FIELDS_BENCHMARK_REGRESSION),
    ATTENTION_GROUP(attention_qualification_fields, ATTENTION_FIELDS_QUALIFICATION),
    ATTENTION_GROUP(attention_cuda_fields, ATTENTION_FIELDS_CUDA),
    ATTENTION_GROUP(attention_comparison_fields, ATTENTION_FIELDS_COMPARISON),
    ATTENTION_GROUP(attention_failure_fields, ATTENTION_FIELDS_COMPARISON_FAILURE),
    ATTENTION_GROUP(attention_provenance_fields, ATTENTION_FIELDS_DETAILED_ADMISSION),
    ATTENTION_GROUP(attention_compatibility_fields, ATTENTION_FIELDS_COMPATIBILITY),
    ATTENTION_GROUP(attention_reachability_fields, ATTENTION_FIELDS_ALWAYS),
    ATTENTION_GROUP(attention_reason_field, ATTENTION_FIELDS_REASON),
    {attention_final_field, FIELD_COUNT(attention_final_field), ATTENTION_FIELDS_ALWAYS, 1},
};
static const attention_presence_rule attention_presence_rules[] = {
    {ATTENTION_FIELDS_TARGET, offsetof(yvex_graph_attention_operator_result, family), ATTENTION_PRESENCE_TARGET, 0},
{ATTENTION_FIELDS_ADMITTED, offsetof(yvex_graph_attention_operator_result, attention_plan_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_COMPLETED, offsetof(yvex_graph_attention_operator_result, completed), ATTENTION_PRESENCE_BOOL, 0},
{ATTENTION_FIELDS_CUDA, offsetof(yvex_graph_attention_operator_result, probe.cuda_device),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_COMPARISON, offsetof(yvex_graph_attention_operator_result, probe.comparison_available),
     ATTENTION_PRESENCE_BOOL, 0},
    {ATTENTION_FIELDS_COMPARISON_FAILURE, 0u, ATTENTION_PRESENCE_COMPARISON_FAILURE, 0},
{ATTENTION_FIELDS_DETAILED_ADMISSION, offsetof(yvex_graph_attention_operator_result, attention_plan_identity),
     ATTENTION_PRESENCE_TEXT, 1},
    {ATTENTION_FIELDS_COMPATIBILITY, offsetof(yvex_graph_attention_operator_result, current_writer_plan_identity),
     ATTENTION_PRESENCE_TEXT, 1},
    {ATTENTION_FIELDS_TIMING, offsetof(yvex_graph_attention_operator_result, benchmark_sample_count),
     ATTENTION_PRESENCE_U64, 0},
    {ATTENTION_FIELDS_RESIDENCY, offsetof(yvex_graph_attention_operator_result, residency_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_STATE, offsetof(yvex_graph_attention_operator_result, state_layout_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_BENCHMARK, offsetof(yvex_graph_attention_operator_result, benchmark_sample_count),
     ATTENTION_PRESENCE_U64, 0},
    {ATTENTION_FIELDS_BENCHMARK_DEVICE,
     offsetof(yvex_graph_attention_operator_result, benchmark_device_timing_available),
     ATTENTION_PRESENCE_BOOL, 0},
    {ATTENTION_FIELDS_BENCHMARK_PUBLICATION, offsetof(yvex_graph_attention_operator_result, benchmark.baseline_written),
     ATTENTION_PRESENCE_BOOL, 0},
    {ATTENTION_FIELDS_BENCHMARK_BASELINE, offsetof(yvex_graph_attention_operator_result, benchmark.baseline_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_BENCHMARK_BASELINE_DEVICE,
     offsetof(yvex_graph_attention_operator_result, benchmark.device_timing_available),
     ATTENTION_PRESENCE_BOOL, 0},
    {ATTENTION_FIELDS_BENCHMARK_CHART, offsetof(yvex_graph_attention_operator_result, benchmark.chart_generated),
     ATTENTION_PRESENCE_BOOL, 0},
    {ATTENTION_FIELDS_BENCHMARK_CONTEXT, offsetof(yvex_graph_attention_operator_result, benchmark_scope),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_BENCHMARK_REGRESSION,
     offsetof(yvex_graph_attention_operator_result, benchmark.comparison_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_QUALIFICATION, offsetof(yvex_graph_attention_operator_result, qualification_identity),
     ATTENTION_PRESENCE_TEXT, 0},
    {ATTENTION_FIELDS_REASON, offsetof(yvex_graph_attention_operator_result, reason), ATTENTION_PRESENCE_TEXT, 0},
};
#undef ATTENTION_GROUP
#undef ATTENTION_CAPABILITY
#undef ATTENTION_PROBE_FIELD
#undef ATTENTION_BENCHMARK_FIELD
#undef ATTENTION_TIMING
static int graph_attention_rule_present(
    const attention_presence_rule *rule,
    const yvex_graph_attention_operator_result *result)
{
    const unsigned char *value = (const unsigned char *)result + rule->offset;
    switch (rule->kind) {
    case ATTENTION_PRESENCE_TEXT: return *(const char *)value != '\0';
    case ATTENTION_PRESENCE_TARGET: return strcmp((const char *)value, "unavailable") != 0;
    case ATTENTION_PRESENCE_BOOL: return *(const int *)value != 0;
    case ATTENTION_PRESENCE_U64: return *(const unsigned long long *)value != 0ull;
    case ATTENTION_PRESENCE_DOUBLE: return *(const double *)value > 0.0;
    case ATTENTION_PRESENCE_COMPARISON_FAILURE:
        return result->probe.comparison_available && !result->probe.comparison_passed;
    }
    return 0;
}
static unsigned int graph_attention_visible_groups(
    const yvex_graph_attention_operator_result *result, int detailed)
{
    unsigned int visible = ATTENTION_FIELDS_ALWAYS;
    size_t index;
    for (index = 0u; index < FIELD_COUNT(attention_presence_rules); ++index)
        if ((!attention_presence_rules[index].detailed || detailed) &&
            graph_attention_rule_present(&attention_presence_rules[index], result))
            visible |= (unsigned int)attention_presence_rules[index].condition;
    return visible;
}
static int graph_attention_emit(FILE *fp,
                                int json,
                                const yvex_graph_attention_operator_result *result,
                                const yvex_cli_field_spec *fields,
                                size_t count,
                                int comma)
{
    int rc = json ? yvex_cli_json_fields(fp, result, fields, count, comma)
                  : yvex_cli_out_fields(fp, result, fields, count);
    return rc < 0 ? YVEX_ERR_IO : rc;
}
static int graph_attention_csv_cell(FILE *fp, const char *text)
{
    const unsigned char *cursor = (const unsigned char *)(text ? text : "");
    if (yvex_cli_out_char(fp, '"') < 0) return YVEX_ERR_IO;
    for (; *cursor; ++cursor) {
        if (*cursor == '"' && yvex_cli_out_char(fp, '"') < 0) return YVEX_ERR_IO;
        if (yvex_cli_out_char(fp, (char)*cursor) < 0) return YVEX_ERR_IO;
    }
    return yvex_cli_out_char(fp, '"') < 0 ? YVEX_ERR_IO : YVEX_OK;
}
static int graph_csv_field(FILE *fp, const void *object,
                           const yvex_cli_field_spec *field)
{
    const unsigned char *base = (const unsigned char *)object;
    const void *value = base + field->offset;
    char number[64];
    const char *text = number;
    switch (field->kind) {
    case YVEX_CLI_FIELD_TEXT:
        text = *(const char *const *)value;
        break;
    case YVEX_CLI_FIELD_TEXT_ARRAY:
        text = (const char *)value;
        break;
    case YVEX_CLI_FIELD_U64:
        (void)snprintf(number, sizeof(number), "%llu", *(const unsigned long long *)value);
        break;
    case YVEX_CLI_FIELD_U32:
        (void)snprintf(number, sizeof(number), "%u", *(const unsigned int *)value);
        break;
    case YVEX_CLI_FIELD_I32:
        (void)snprintf(number, sizeof(number), "%d", *(const int *)value);
        break;
    case YVEX_CLI_FIELD_BOOL:
        text = *(const int *)value ? "true" : "false";
        break;
    case YVEX_CLI_FIELD_DOUBLE:
        (void)snprintf(number, sizeof(number), "%.17g", *(const double *)value);
        break;
    case YVEX_CLI_FIELD_FLOAT9:
        (void)snprintf(number, sizeof(number), "%.9g", *(const double *)value);
        break;
    case YVEX_CLI_FIELD_HEX64:
        (void)snprintf(number, sizeof(number), "%016llx",
                       *(const unsigned long long *)value);
        break;
    default:
        return YVEX_ERR_UNSUPPORTED;
    }
    if (!text || !text[0]) text = field->fallback ? field->fallback : "unknown";
    if (graph_attention_csv_cell(fp, field->key) != YVEX_OK ||
        yvex_cli_out_char(fp, ',') < 0 || graph_attention_csv_cell(fp, text) != YVEX_OK ||
        yvex_cli_out_char(fp, '\n') < 0)
        return YVEX_ERR_IO;
    return YVEX_OK;
}
static int graph_attention_render_fields(FILE *fp,
                                         yvex_graph_report_mode mode,
                                         const yvex_graph_attention_operator_result *result)
{
    int json = mode == YVEX_GRAPH_REPORT_MODE_JSON;
    int csv = mode == YVEX_GRAPH_REPORT_MODE_CSV;
    int detailed = json || csv || mode == YVEX_GRAPH_REPORT_MODE_AUDIT;
    unsigned int visible = graph_attention_visible_groups(result, detailed);
    size_t index, field_index;
    int rc = YVEX_OK;
    if (csv && yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
    if (json) yvex_cli_json_begin(fp);
    for (index = 0; rc == YVEX_OK && index < FIELD_COUNT(attention_field_groups); ++index) {
        const attention_field_group *group = &attention_field_groups[index];
        if (!(visible & (unsigned int)group->condition)) continue;
        if (csv) {
            for (field_index = 0; field_index < group->count; ++field_index)
                if (graph_csv_field(fp, result, &group->fields[field_index]) != YVEX_OK)
                    return YVEX_ERR_IO;
        } else {
            rc = graph_attention_emit(fp, json, result, group->fields, group->count, !group->final);
        }
    }
    if (json) yvex_cli_json_end(fp);
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
int yvex_graph_attention_render(FILE *fp,
                                yvex_graph_report_mode mode,
                                const yvex_graph_attention_operator_result *result)
{
    if (!fp || !result) return YVEX_ERR_INVALID_ARG;
    return graph_attention_render_fields(fp, mode, result);
}
int yvex_graph_moe_render(FILE *fp, yvex_graph_report_mode mode,
                          const yvex_moe_operator_result *result)
{
    size_t index;
    int rc;
    if (!fp || !result) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
        if (yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
        for (index = 0; index < FIELD_COUNT(moe_fields); ++index)
            if (graph_csv_field(fp, result, &moe_fields[index]) != YVEX_OK)
                return YVEX_ERR_IO;
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        rc = yvex_cli_json_fields(fp, result, moe_fields, FIELD_COUNT(moe_fields), 0);
        yvex_cli_json_end(fp);
    } else {
        rc = yvex_cli_out_fields(fp, result, moe_fields, FIELD_COUNT(moe_fields));
    }
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
int yvex_graph_transformer_render(FILE *fp, yvex_graph_report_mode mode,
                                  const yvex_transformer_operator_result *result)
{
    size_t index;
    int rc;
    if (!fp || !result) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
        if (yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
        for (index = 0; index < FIELD_COUNT(transformer_fields); ++index)
            if (graph_csv_field(fp, result, &transformer_fields[index]) != YVEX_OK)
                return YVEX_ERR_IO;
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        rc = yvex_cli_json_fields(fp, result, transformer_fields,
                                  FIELD_COUNT(transformer_fields), 0);
        yvex_cli_json_end(fp);
    } else {
        rc = yvex_cli_out_fields(fp, result, transformer_fields,
                                 FIELD_COUNT(transformer_fields));
    }
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
/*
 * Render ordered per-step decode evidence after the flat operator summary.
 *
 * No identity or capability derivation.
 */
static int graph_decode_steps_render(FILE *fp, yvex_graph_report_mode mode,
                                     const yvex_decode_operator_result *result)
{
    unsigned long long index;
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        if (yvex_cli_out_puts(fp, "  \"steps\": [\n") < 0) return YVEX_ERR_IO;
        for (index = 0ull; index < result->step_count; ++index) {
            const yvex_runtime_decode_step_result *step = &result->steps[index];
            if (yvex_cli_out_writef(
                    fp,
                    "    {\"ordinal\":%llu,\"token_id\":%u,\"position_before\":%llu,"
                    "\"position_after\":%llu,\"generation_before\":%llu,"
                    "\"generation_after\":%llu,\"hidden_digest\":\"%s\","
                    "\"state_digest\":\"%s\",\"transformer_execution_identity\":\"%s\","
                    "\"decode_step_identity\":\"%s\"}%s\n",
                    step->step_ordinal, step->token_id, step->position_before,
                    step->position_after, step->generation_before,
                    step->generation_after, step->normalized_hidden_digest,
                    step->persistent_state_digest,
                    step->transformer_execution_identity,
                    step->decode_step_identity,
                    index + 1ull < result->step_count ? "," : "") < 0)
                return YVEX_ERR_IO;
        }
        yvex_cli_out_line(fp, "  ]");
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    for (index = 0ull; index < result->step_count; ++index) {
        const yvex_runtime_decode_step_result *step = &result->steps[index];
        if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
            if (yvex_cli_out_writef(
                    fp, "\"step.%llu\",\"token=%u position=%llu:%llu generation=%llu:%llu "
                        "hidden=%s state=%s identity=%s\"\n",
                    index, step->token_id, step->position_before,
                    step->position_after, step->generation_before,
                    step->generation_after, step->normalized_hidden_digest,
                    step->persistent_state_digest, step->decode_step_identity) < 0)
                return YVEX_ERR_IO;
        } else if (yvex_cli_out_writef(
                       fp, "step.%llu: token=%u position=%llu:%llu generation=%llu:%llu "
                           "hidden=%s state=%s identity=%s\n",
                       index, step->token_id, step->position_before,
                       step->position_after, step->generation_before,
                       step->generation_after, step->normalized_hidden_digest,
                       step->persistent_state_digest, step->decode_step_identity) < 0)
            return YVEX_ERR_IO;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
int yvex_graph_decode_render(FILE *fp, yvex_graph_report_mode mode,
                             const yvex_decode_operator_result *result)
{
    size_t index;
    int rc;
    if (!fp || !result || (result->step_count && !result->steps))
        return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
        if (yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
        for (index = 0; index < FIELD_COUNT(decode_fields); ++index)
            if (graph_csv_field(fp, result, &decode_fields[index]) != YVEX_OK)
                return YVEX_ERR_IO;
        return graph_decode_steps_render(fp, mode, result);
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        rc = yvex_cli_json_fields(fp, result, decode_fields,
                                  FIELD_COUNT(decode_fields), 1);
        if (rc == YVEX_OK) rc = graph_decode_steps_render(fp, mode, result);
        yvex_cli_json_end(fp);
    } else {
        rc = yvex_cli_out_fields(fp, result, decode_fields,
                                 FIELD_COUNT(decode_fields));
        if (rc == YVEX_OK) rc = graph_decode_steps_render(fp, mode, result);
    }
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
/*
 * Render bounded per-row logits evidence without dumping raw vocabulary values.
 *
 * Writes only bounded scalar and identity evidence.
 */
static int graph_logits_rows_render(FILE *fp, yvex_graph_report_mode mode,
                                    const yvex_logits_operator_result *result)
{
    unsigned long long index;
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        if (yvex_cli_out_puts(fp, "  \"rows\": [\n") < 0) return YVEX_ERR_IO;
        for (index = 0ull; index < result->row_count; ++index) {
            const yvex_runtime_logits_row_result *row = &result->rows[index];
            if (yvex_cli_out_writef(
                    fp,
                    "    {\"ordinal\":%llu,\"source_phase\":\"%s\","
                    "\"source_position\":%llu,\"logits_count\":%llu,"
                    "\"finite_count\":%llu,\"minimum_logit\":%.9g,"
                    "\"maximum_logit\":%.9g,\"raw_logits_digest\":\"%s\","
                    "\"output_head_residency_identity\":\"%s\","
                    "\"backend_execution_identity\":\"%s\","
                    "\"logits_row_identity\":\"%s\"}%s\n",
                    index,
                    row->source_phase == YVEX_LOGITS_SOURCE_PREFILL ? "prefill" : "decode",
                    row->source_position, row->logits_count, row->finite_count,
                    row->minimum_logit, row->maximum_logit, row->raw_logits_digest,
                    row->output_head_residency_identity,
                    row->backend_execution_identity, row->logits_row_identity,
                    index + 1ull < result->row_count ? "," : "") < 0)
                return YVEX_ERR_IO;
        }
        yvex_cli_out_line(fp, "  ]");
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    for (index = 0ull; index < result->row_count; ++index) {
        const yvex_runtime_logits_row_result *row = &result->rows[index];
        if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
            if (yvex_cli_out_writef(
                    fp, "\"row.%llu\",\"phase=%s position=%llu values=%llu finite=%llu "
                        "digest=%s identity=%s\"\n",
                    index,
                    row->source_phase == YVEX_LOGITS_SOURCE_PREFILL ? "prefill" : "decode",
                    row->source_position, row->logits_count, row->finite_count,
                    row->raw_logits_digest, row->logits_row_identity) < 0)
                return YVEX_ERR_IO;
        } else if (yvex_cli_out_writef(
                       fp, "row.%llu: phase=%s position=%llu values=%llu finite=%llu "
                           "digest=%s identity=%s\n",
                       index,
                       row->source_phase == YVEX_LOGITS_SOURCE_PREFILL ? "prefill" : "decode",
                       row->source_position, row->logits_count, row->finite_count,
                       row->raw_logits_digest, row->logits_row_identity) < 0)
            return YVEX_ERR_IO;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
int yvex_graph_logits_render(FILE *fp, yvex_graph_report_mode mode,
                             const yvex_logits_operator_result *result)
{
    size_t index;
    int rc;
    if (!fp || !result || (result->row_count && !result->rows))
        return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
        if (yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
        for (index = 0u; index < FIELD_COUNT(logits_fields); ++index)
            if (graph_csv_field(fp, result, &logits_fields[index]) != YVEX_OK)
                return YVEX_ERR_IO;
        return graph_logits_rows_render(fp, mode, result);
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        rc = yvex_cli_json_fields(fp, result, logits_fields,
                                  FIELD_COUNT(logits_fields), 1);
        if (rc == YVEX_OK) rc = graph_logits_rows_render(fp, mode, result);
        yvex_cli_json_end(fp);
    } else {
        rc = yvex_cli_out_fields(fp, result, logits_fields,
                                 FIELD_COUNT(logits_fields));
        if (rc == YVEX_OK) rc = graph_logits_rows_render(fp, mode, result);
    }
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
static int graph_sampling_rows(FILE *fp, yvex_graph_report_mode mode,
                               const yvex_sampling_operator_result *result)
{
    unsigned long long index;
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON)
        yvex_cli_out_puts(fp, "  \"selected_tokens\": [\n");
    for (index = 0ull; index < result->sample_count; ++index) {
        const yvex_runtime_sampling_result *row = &result->samples[index];
        const char *phase = row->source_phase == YVEX_LOGITS_SOURCE_PREFILL ? "prefill" : "decode";
        if (mode == YVEX_GRAPH_REPORT_MODE_JSON)
            yvex_cli_out_writef(fp, "    {\"ordinal\":%llu,\"phase\":\"%s\",\"position\":%llu,"
                                "\"token_id\":%u,\"logit\":%.9g,\"probability\":%.17g,"
                                "\"log_probability\":%.17g,\"candidates\":%llu,"
                                "\"source_identity\":\"%s\",\"candidate_identity\":\"%s\","
                                "\"rng_before\":\"%s\",\"rng_after\":\"%s\","
                                "\"identity\":\"%s\"}%s\n",
                                index, phase, row->source_position, row->selected_token_id,
                                row->selected_logit, row->selected_probability,
                                row->selected_log_probability, row->final_candidate_count,
                                row->source_identity, row->candidate_set_identity,
                                row->rng_state_before_identity, row->rng_state_after_identity,
                                row->selected_token_identity,
                                index + 1ull < result->sample_count ? "," : "");
        else
            yvex_cli_out_writef(fp, mode == YVEX_GRAPH_REPORT_MODE_CSV
                ? "\"sample.%llu\",\"phase=%s position=%llu token=%u logit=%.9g "
                  "probability=%.17g log_probability=%.17g candidates=%llu "
                  "source=%s candidate=%s rng_before=%s rng_after=%s identity=%s\"\n"
                : "sample.%llu: phase=%s position=%llu token=%u logit=%.9g "
                  "probability=%.17g log_probability=%.17g candidates=%llu "
                  "source=%s candidate=%s rng_before=%s rng_after=%s identity=%s\n",
                index, phase, row->source_position, row->selected_token_id,
                row->selected_logit, row->selected_probability,
                row->selected_log_probability, row->final_candidate_count,
                row->source_identity, row->candidate_set_identity,
                row->rng_state_before_identity, row->rng_state_after_identity,
                row->selected_token_identity);
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) yvex_cli_out_line(fp, "  ]");
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
int yvex_graph_sampling_render(FILE *fp, yvex_graph_report_mode mode,
                               const yvex_sampling_operator_result *result)
{
    size_t index;
    int rc;
    if (!fp || !result || (result->sample_count && !result->samples))
        return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_CSV) {
        if (yvex_cli_out_writef(fp, "field,value\n") < 0) return YVEX_ERR_IO;
        for (index = 0u; index < FIELD_COUNT(sampling_fields); ++index)
            if (graph_csv_field(fp, result, &sampling_fields[index]) != YVEX_OK)
                return YVEX_ERR_IO;
        return graph_sampling_rows(fp, mode, result);
    }
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        rc = yvex_cli_json_fields(fp, result, sampling_fields,
                                  FIELD_COUNT(sampling_fields), 1);
        if (rc == YVEX_OK) rc = graph_sampling_rows(fp, mode, result);
        yvex_cli_json_end(fp);
    } else {
        rc = yvex_cli_out_fields(fp, result, sampling_fields,
                                 FIELD_COUNT(sampling_fields));
        if (rc == YVEX_OK) rc = graph_sampling_rows(fp, mode, result);
    }
    return rc < 0 || ferror(fp) ? YVEX_ERR_IO : rc;
}
static const char *const generation_roofline_names[] = {
    "prefill-layer", "decode-layer", "verify-sweep", "draft-sweep",
    "output-head", "state-promotion", "batched-decode"};
static int graph_generation_roofline_json(FILE *fp, const yvex_runtime_generation_result *run)
{
    const yvex_execution_roofline_ledger *ledger = &run->roofline;
    unsigned long long index;
    if (!run->roofline_available)
        return yvex_cli_out_puts(fp, "  \"roofline\": null,\n") < 0 ? YVEX_ERR_IO : YVEX_OK;
    if (yvex_cli_out_writef(
            fp, "  \"roofline\": {\"schema\":%u,\"identity\":\"%s\","
                "\"measured_phase_mask\":%llu,\"missing_phase_mask\":%llu,"
                "\"rooflined_phase_mask\":%llu,\"priority_provisional\":%s,"
                "\"phases\":[\n", ledger->schema_version, ledger->identity,
            ledger->measured_phase_mask, ledger->missing_phase_mask,
            ledger->rooflined_phase_mask,
            ledger->priority_provisional ? "true" : "false") < 0) return YVEX_ERR_IO;
    for (index = 0ull; index < ledger->phase_count; ++index) {
        const yvex_execution_phase_roofline *phase = &ledger->phases[index];
        const yvex_execution_phase_measurement *value = &phase->measurement;
        if (yvex_cli_out_writef(
                fp, "    {\"name\":\"%s\",\"available\":%s,\"roofline_available\":%s,"
                    "\"fact_mask\":%llu,\"missing_fact_mask\":%llu,"
                    "\"active_weight_bytes\":%llu,\"state_bytes\":%llu,"
                    "\"activation_bytes\":%llu,\"temporary_bytes\":%llu,"
                    "\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"d2d_bytes\":%llu,"
                    "\"kernel_count\":%llu,\"synchronization_count\":%llu,"
                    "\"occupancy_parts_per_million\":%llu,\"minimum_memory_time_ns\":%llu,"
                    "\"duration_ns\":%llu,\"work_units\":%llu,\"committed_tokens\":%llu,"
                    "\"active_device_bytes\":%llu,\"transfer_bytes\":%llu,"
                    "\"measured_bytes_per_second\":%llu,\"roofline_utilization_ppm\":%llu,"
                    "\"optimization_headroom_ns\":%llu,\"priority\":%llu}%s\n",
                generation_roofline_names[index], phase->available ? "true" : "false",
                phase->roofline_available ? "true" : "false", value->fact_mask,
                phase->missing_fact_mask, value->active_weight_bytes, value->state_bytes,
                value->activation_bytes, value->temporary_bytes, value->h2d_bytes,
                value->d2h_bytes, value->d2d_bytes, value->kernel_count,
                value->synchronization_count, value->occupancy_parts_per_million,
                phase->minimum_memory_time_ns, value->measured_duration_ns, value->work_units,
                value->committed_tokens, phase->active_device_bytes, phase->transfer_bytes,
                phase->measured_bytes_per_second, phase->roofline_utilization_parts_per_million,
                phase->optimization_headroom_ns, phase->optimization_priority,
                index + 1ull < ledger->phase_count ? "," : "") < 0) return YVEX_ERR_IO;
    }
    return yvex_cli_out_puts(fp, "  ]},\n") < 0 ? YVEX_ERR_IO : YVEX_OK;
}
static int graph_generation_roofline_audit(FILE *fp, const yvex_runtime_generation_result *run)
{
    const yvex_execution_roofline_ledger *ledger = &run->roofline;
    unsigned long long index;
    if (!run->roofline_available) return YVEX_OK;
    if (yvex_cli_out_writef(
            fp, "roofline: identity=%s measured=%llu missing=%llu rooflined=%llu provisional=%s\n",
            ledger->identity, ledger->measured_phase_mask, ledger->missing_phase_mask,
            ledger->rooflined_phase_mask, ledger->priority_provisional ? "true" : "false") < 0)
        return YVEX_ERR_IO;
    for (index = 0ull; index < ledger->phase_count; ++index) {
        const yvex_execution_phase_roofline *phase = &ledger->phases[index];
        const yvex_execution_phase_measurement *value = &phase->measurement;
        if (yvex_cli_out_writef(
                fp, "roofline.%s: available=%s facts=%llu missing=%llu "
                    "weight=%llu state=%llu activation=%llu temporary=%llu "
                    "movement=%llu/%llu/%llu kernels=%llu syncs=%llu occupancy_ppm=%llu "
                    "minimum_ns=%llu duration_ns=%llu work=%llu committed=%llu "
                    "active=%llu transfer=%llu measured_bps=%llu utilization_ppm=%llu "
                    "headroom_ns=%llu priority=%llu\n",
                generation_roofline_names[index], phase->available ? "true" : "false",
                value->fact_mask, phase->missing_fact_mask, value->active_weight_bytes,
                value->state_bytes, value->activation_bytes, value->temporary_bytes,
                value->h2d_bytes, value->d2h_bytes, value->d2d_bytes, value->kernel_count,
                value->synchronization_count, value->occupancy_parts_per_million,
                phase->minimum_memory_time_ns, value->measured_duration_ns, value->work_units,
                value->committed_tokens, phase->active_device_bytes, phase->transfer_bytes,
                phase->measured_bytes_per_second, phase->roofline_utilization_parts_per_million,
                phase->optimization_headroom_ns, phase->optimization_priority) < 0)
            return YVEX_ERR_IO;
    }
    return YVEX_OK;
}
int yvex_graph_generation_render(FILE *fp, yvex_graph_report_mode mode,
                                 const yvex_generation_operator_result *result)
{
    const yvex_runtime_generation_result *run; unsigned long long index;
    if (!fp || !result || (result->token_count && !result->tokens)) return YVEX_ERR_INVALID_ARG;
    run = &result->execution;
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        yvex_cli_json_field_str(fp, "command", result->command, 1);
        yvex_cli_json_field_str(fp, "status", result->status, 1);
        yvex_cli_json_field_str(fp, "reason", result->reason, 1);
        yvex_cli_json_field_str(fp, "target", result->target, 1);
        yvex_cli_json_field_str(fp, "family", result->family, 1);
        yvex_cli_json_field_str(fp, "backend", result->backend, 1);
        yvex_cli_json_field_str(fp, "generation_plan_identity", result->plan.generation_plan_identity, 1);
        yvex_cli_json_field_str(fp, "generation_execution_identity", run->generation_execution_identity, 1);
        yvex_cli_json_field_str(fp, "prompt_identity", run->prompt_identity, 1);
        yvex_cli_json_field_str(
            fp, "execution_mode",
            run->execution_mode == YVEX_GENERATION_MODE_DSPARK
                ? "dspark" : "target-only",
            1);
        yvex_cli_json_field_str(fp, "speculation_policy_identity",
                                run->speculation_policy_identity, 1);
        if (yvex_cli_out_writef(fp,
                "  \"prompt_tokens\": %llu,\n  \"prefill_chunks\": %llu,\n"
                "  \"sampled_tokens\": %llu,\n  \"model_committed_tokens\": %llu,\n"
                "  \"decode_steps\": %llu,\n  \"logits_projections\": %llu,\n"
                "  \"sampling_draws\": %llu,\n  \"generated_text_bytes\": %llu,\n"
                "  \"draft_cycles\": %llu,\n  \"draft_forwards\": %llu,\n"
                "  \"proposed_tokens\": %llu,\n"
                "  \"selected_verification_tokens\": %llu,\n"
                "  \"target_verifications\": %llu,\n"
                "  \"accepted_draft_tokens\": %llu,\n"
                "  \"rejected_draft_tokens\": %llu,\n"
                "  \"discarded_draft_tokens\": %llu,\n"
                "  \"target_correction_or_bonus_tokens\": %llu,\n"
                "  \"maximum_accepted_prefix\": %llu,\n"
                "  \"mean_accepted_prefix\": %.9f,\n"
                "  \"confidence_logit_count\": %llu,\n"
                "  \"confidence_logit_minimum\": %.9f,\n"
                "  \"confidence_logit_maximum\": %.9f,\n"
                "  \"confidence_logit_mean\": %.9f,\n"
                "  \"draft_seconds\": %.9f,\n"
                "  \"verification_seconds\": %.9f,\n"
                "  \"speculative_commit_seconds\": %.9f,\n"
                "  \"effective_committed_tokens_per_second\": %.9f,\n"
                "  \"final_position\": %llu,\n  \"final_generation\": %llu,\n",
                run->prompt_token_count, run->prefill_chunk_count, run->sampled_token_count,
                run->model_committed_token_count, run->decode_step_count,
                run->logits_projection_count, run->sampling_draw_count,
                run->generated_text_bytes, run->draft_cycle_count,
                run->draft_forward_count, run->proposed_token_count,
                run->selected_verification_token_count,
                run->target_verification_count,
                run->accepted_draft_token_count,
                run->rejected_draft_token_count,
                run->discarded_draft_token_count,
                run->target_correction_or_bonus_token_count,
                run->maximum_accepted_prefix, run->mean_accepted_prefix,
                run->confidence_logit_count,
                run->confidence_logit_minimum,
                run->confidence_logit_maximum,
                run->confidence_logit_mean,
                (double)run->draft_ns / 1000000000.0,
                (double)run->verification_ns / 1000000000.0,
                (double)run->speculative_commit_ns / 1000000000.0,
                run->effective_committed_tokens_per_second,
                run->final_position,
                run->final_persistent_generation) < 0) return YVEX_ERR_IO;
        yvex_cli_json_field_str(fp, "generated_text_digest", run->generated_text_digest, 1);
        yvex_cli_json_field_str(fp, "persistent_state_digest", run->final_persistent_state_digest, 1);
        yvex_cli_json_field_str(fp, "stop_reason",
                                yvex_runtime_generation_stop_reason_name(run->stop_reason), 1);
        yvex_cli_json_field_bool(fp, "generation_ready", result->generation_ready, 1);
        yvex_cli_json_field_bool(fp, "cli_generate_ready", result->cli_generate_ready, 1);
        if (yvex_cli_out_writef(fp,
                "  \"profile\": {\"schema\":%u,\"mode\":\"%s\",\"identity\":\"%s\","
                "\"phases_ns\":{", run->profile.schema_version,
                runtime_profile_mode_name(run->profile.mode),
                run->profile.profile_identity) < 0) return YVEX_ERR_IO;
        for (index = 0ull; index < YVEX_RUNTIME_PROFILE_PHASE_COUNT; ++index)
            if (yvex_cli_out_writef(fp, "\"%s\":%llu%s",
                    runtime_profile_phase_name((yvex_runtime_profile_phase)index),
                    run->profile.phase_ns[index],
                    index + 1ull == YVEX_RUNTIME_PROFILE_PHASE_COUNT ? "" : ",") < 0)
                return YVEX_ERR_IO;
        yvex_cli_out_puts(fp, "},\"counters\":{");
        for (index = 0ull; index < YVEX_RUNTIME_PROFILE_COUNTER_COUNT; ++index)
            if (yvex_cli_out_writef(fp, "\"%s\":%llu%s",
                    runtime_profile_counter_name((yvex_runtime_profile_counter)index),
                    run->profile.counters[index],
                    index + 1ull == YVEX_RUNTIME_PROFILE_COUNTER_COUNT ? "" : ",") < 0)
                return YVEX_ERR_IO;
        yvex_cli_out_puts(fp, "}},\n");
        if (graph_generation_roofline_json(fp, run) != YVEX_OK) return YVEX_ERR_IO;
        yvex_cli_out_puts(fp, "  \"generated_tokens\": [\n");
        for (index = 0ull; index < result->token_count; ++index) {
            const yvex_runtime_generation_token_result *token = &result->tokens[index];
            if (yvex_cli_out_writef(fp,
                    "    {\"ordinal\":%llu,\"token_id\":%u,\"decode_input_id\":%u,"
                    "\"decode_submitted\":%s,\"model_committed\":%s,"
                    "\"terminal\":%s,\"suppressed\":%s,\"position_before\":%llu,"
                    "\"position_after\":%llu,\"text_bytes\":%llu,\"identity\":\"%s\"}%s\n",
                    token->ordinal, token->sampled_token_id, token->decode_input_token_id,
                    token->decode_submitted ? "true" : "false",
                    token->model_committed ? "true" : "false", token->terminal ? "true" : "false",
                    token->suppressed ? "true" : "false", token->position_before,
                    token->position_after, token->text_byte_count, token->token_step_identity,
                    index + 1ull < result->token_count ? "," : "") < 0) return YVEX_ERR_IO;
        }
        yvex_cli_out_line(fp, "  ]");
        yvex_cli_json_end(fp);
    } else {
        if (yvex_cli_out_writef(fp,
                "status: %s\nprompt_tokens: %llu\nsampled_tokens: %llu\n"
                "model_committed_tokens: %llu\nexecution_mode: %s\n"
                "generated_text_bytes: %llu\n"
                "generated_text_digest: %s\nstop_reason: %s\nreason: %s\n",
                result->status, run->prompt_token_count, run->sampled_token_count,
                run->model_committed_token_count,
                run->execution_mode == YVEX_GENERATION_MODE_DSPARK
                    ? "dspark" : "target-only",
                run->generated_text_bytes, run->generated_text_digest,
                yvex_runtime_generation_stop_reason_name(run->stop_reason),
                result->reason[0] ? result->reason : "none") < 0)
            return YVEX_ERR_IO;
        if (run->execution_mode == YVEX_GENERATION_MODE_DSPARK &&
            yvex_cli_out_writef(
                fp, "speculation: proposed=%llu accepted=%llu rejected=%llu "
                    "verifications=%llu max_prefix=%llu confidence=%llu "
                    "[%.6g,%.6g] mean=%.6g\n",
                run->proposed_token_count, run->accepted_draft_token_count,
                run->rejected_draft_token_count,
                run->target_verification_count,
                run->maximum_accepted_prefix, run->confidence_logit_count,
                run->confidence_logit_minimum,
                run->confidence_logit_maximum,
                run->confidence_logit_mean) < 0)
            return YVEX_ERR_IO;
        if (mode == YVEX_GRAPH_REPORT_MODE_AUDIT &&
            graph_generation_roofline_audit(fp, run) != YVEX_OK) return YVEX_ERR_IO;
        for (index = 0ull; index < result->token_count; ++index)
            if (yvex_cli_out_writef(fp, "token.%llu: id=%u committed=%s terminal=%s text_bytes=%llu\n",
                    index, result->tokens[index].sampled_token_id,
                    result->tokens[index].model_committed ? "true" : "false",
                    result->tokens[index].terminal ? "true" : "false",
                    result->tokens[index].text_byte_count) < 0) return YVEX_ERR_IO;
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}
int yvex_graph_render_help(FILE *fp)
{
    yvex_cli_out_lines(fp, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
    return YVEX_OK;
}
static const char *const literal_lines_2[] = {
    "generation_ready: false", "generation: unsupported-full-model", "benchmark_status: not-measured"};
static const char *const literal_lines_3[] = {
    "prefill_descriptor: unsupported-full-transformer-prefill", "prefill.requires_embedding: true",
    "prefill.requires_attention_qkv: true", "prefill.requires_real_kv_writes: true",
    "prefill.requires_mlp_or_moe: true", "prefill.requires_layer_scheduler: true",
    "prefill.current_status: unsupported", "prefill.blocker: real transformer prefill not implemented",
    "decode_descriptor: unsupported-full-model-decode", "decode.mode_required: baseline-autoregressive",
    "decode.requires_prefill_state: true", "decode.requires_kv_read: true", "decode.requires_layer_execution: true",
    "decode.current_status: unsupported", "decode.blocker: full model decode not implemented",
    "logits_descriptor: unsupported-real-output-head-logits",
    "sampling_descriptor: unsupported-real-vocabulary-sampling", "residency_requirements_status: planned",
    "residency_plan: descriptor-only-no-allocation"};
static const char *const literal_lines_4[] = {
    "ssd_staged_required_bytes: planned", "kv_required_bytes: planned", "scratch_required_bytes: planned",
    "context_requirements_status: planned", "max_context: metadata-or-unknown", "requested_context: not-requested",
    "context_policy: planned", "position_policy: rope-or-family-specific-planned", "rope_policy: planned",
    "kv_requirements_status: unsupported", "kv_layout: planned", "kv_dtype: planned", "kv_layers: unknown",
    "kv_heads: unknown", "kv_head_dim: unknown", "kv_capacity_status: unsupported-full-transformer-kv",
    "kv.required: true", "kv.real_attention_writes: false", "kv.runtime_status: unsupported",
    "kv_write_ready: false", "kv_read_ready: false", "logits_requirements_status: unsupported"};
static const char *const literal_lines_5[] = {
    "logits_buffer_required: true", "real_output_head_logits: false", "logits_ready: false",
    "logits.blocker: real output-head logits runtime unsupported"};
static const char *const literal_lines_6[] = {
    "special_token_policy: planned", "eos_backed_stop: unsupported", "stop_token_text_matching: unsupported",
    "tokenizer_quality_generation: unsupported"};
static const char *const literal_lines_7[] = {
    "backend.primitive_rope: implemented-fixture", "backend.primitive_attention: implemented-fixture",
    "backend.primitive_matmul: implemented-fixture", "backend.primitive_mlp: implemented-fixture",
    "backend.full_transformer_integration: unsupported", "backend_allocation_attempted: false"};
static const char *const literal_lines_8[] = {
    "prefill_ready: false", "decode_ready: false", "sampling_ready: false", "cleanup_attempted: false",
    "cleanup_status: not-needed"};
static const char *const literal_pair_15[] = { "full_runtime_model: false", "full_model_execution: unsupported"};
static const char *const literal_pair_16[] = { "fullmodel: descriptor", "status: fullmodel-descriptor"};
typedef struct {
    const char *role;
    const char *collection;
} descriptor_role_collection;
static const descriptor_role_collection descriptor_role_collections[] = {
    {"token_embedding", "embedding"}, {"attention_norm", "normalization"},
{"post_attention_norm", "normalization"}, {"final_norm", "normalization"},
    {"q_projection", "attention"}, {"k_projection", "attention"},
{"v_projection", "attention"}, {"o_projection", "attention"},
    {"mlp_gate", "mlp"}, {"mlp_up", "mlp"}, {"mlp_down", "mlp"}, {"moe_router", "moe"},
{"moe_expert_gate", "moe"}, {"moe_expert_up", "moe"}, {"moe_expert_down", "moe"}, {"output_head", "output"},
    {"tokenizer_metadata", "tokenizer-runtime-input"},
};
static const char *fullmodel_descriptor_role_collection(const char *role)
{
    size_t index;
    if (!role) return "unknown";
    for (index = 0; index < sizeof(descriptor_role_collections) /
                                  sizeof(descriptor_role_collections[0]); ++index) {
        if (strcmp(role, descriptor_role_collections[index].role) == 0)
            return descriptor_role_collections[index].collection;
    }
    return "unknown";
}
static const char *fullmodel_descriptor_role_residency(const char *role,
                                                       const char *backend,
                                                       int present)
{
    if (!present) return "not-planned";
    if (role && strcmp(role, "tokenizer_metadata") == 0) return "host-runtime-metadata";
    return backend && strcmp(backend, "cuda") == 0 ? "cuda-resident-planned" : "cpu-resident-planned";
}
static void fullmodel_print_descriptor_role(yvex_model_context *ctx,
                                            const yvex_fullmodel_collections *collections,
                                            const char *role,
                                            const char *backend)
{
    const yvex_tensor_info *tensor = NULL;
    char dims[128];
    int present = 0;
    if (role && strcmp(role, "tokenizer_metadata") == 0) {
        present = collections && collections->has_tokenizer_metadata;
    } else {
        tensor = fullmodel_descriptor_find_tensor(ctx, role);
        present = tensor != NULL;
    }
    dims[0] = '\0';
    if (tensor) dims_to_text(tensor->dims, tensor->rank, dims, sizeof(dims));
    yvex_cli_out_writef(stdout, "role.%s.status: %s\n", role ? role : "unknown", present ? "present" : "missing");
    yvex_cli_out_writef(stdout, "role.%s.tensor: %s\n", role ? role : "unknown",
           tensor && tensor->name ? tensor->name : present ? "metadata" : "none");
    yvex_cli_out_writef(stdout, "role.%s.shape: %s\n", role ? role : "unknown",
           tensor ? dims : present ? "metadata" : "unknown");
    yvex_cli_out_writef(stdout, "role.%s.dtype: %s\n", role ? role : "unknown",
           tensor ? yvex_dtype_name(tensor->dtype) : present ? "metadata" : "unknown");
    yvex_cli_out_writef(stdout, "role.%s.qtype: %s\n", role ? role : "unknown",
           tensor ? yvex_dtype_name(tensor->dtype) : present ? "metadata" : "unknown");
    yvex_cli_out_writef(stdout, "role.%s.bytes: %llu\n", role ? role : "unknown",
           tensor ? tensor->storage_bytes : 0ull);
    yvex_cli_out_writef(stdout, "role.%s.collection: %s\n", role ? role : "unknown",
           fullmodel_descriptor_role_collection(role));
    yvex_cli_out_writef(stdout, "role.%s.residency_expectation: %s\n", role ? role : "unknown",
           fullmodel_descriptor_role_residency(role, backend, present));
    yvex_cli_out_writef(stdout, "role.%s.runtime_consumer: %s\n", role ? role : "unknown",
           present ? "planned" : "blocked-missing-role");
}
/*
 * Render one descriptor collection and its exact runtime requirements.
 *
 * Borrowed collection identity, accounting, requirements, and blocker facts. Rendering does not
 * make the collection resident or executable.
 */
static void fullmodel_print_descriptor_collection(const char *name,
                                                  unsigned long long count,
                                                  unsigned long long bytes,
                                                  int required_for_prefill,
                                                  int required_for_decode,
                                                  int required_for_logits,
                                                  int required_for_generation,
                                                  const char *runtime_consumer,
                                                  const char *blocker)
{
    yvex_cli_out_writef(stdout, "collection.%s.status: %s\n", name, count > 0ull ? "present" : "missing");
    yvex_cli_out_writef(stdout, "collection.%s.tensor_count: %llu\n", name, count);
    yvex_cli_out_writef(stdout, "collection.%s.byte_count: %llu\n", name, bytes);
    yvex_cli_out_writef(stdout, "collection.%s.required_for_prefill: %s\n", name,
        required_for_prefill ? "true" : "false");
    yvex_cli_out_writef(stdout, "collection.%s.required_for_decode: %s\n", name,
        required_for_decode ? "true" : "false");
    yvex_cli_out_writef(stdout, "collection.%s.required_for_logits: %s\n", name,
        required_for_logits ? "true" : "false");
    yvex_cli_out_writef(stdout, "collection.%s.required_for_generation: %s\n", name,
        required_for_generation ? "true" : "false");
    yvex_cli_out_writef(stdout, "collection.%s.runtime_consumer: %s\n", name,
        runtime_consumer ? runtime_consumer : "planned");
    yvex_cli_out_writef(stdout, "collection.%s.blocker: %s\n", name, blocker && blocker[0] ? blocker : "none");
}
typedef struct {
    const char *key;
    size_t flag_offset;
    const char *available;
    const char *unavailable;
} graph_requirement_spec;
#define GRAPH_FIXED ((size_t)-1)
#define GRAPH_ATTENTION ((size_t)-2)
#define GRAPH_MLP ((size_t)-3)
#define GRAPH_NORMALIZATION ((size_t)-4)
#define GRAPH_FLAG(member_) offsetof(yvex_fullmodel_collections, member_)
static const graph_requirement_spec graph_requirements[] = {
    {"graph_requirements_status", GRAPH_FIXED, "blocked", NULL}, {"required_graph_ops", GRAPH_FIXED,
     "embedding-lookup,rmsnorm,q-projection,k-projection,v-projection,rope-position,attention-score,"
     "causal-mask,softmax,attention-value-accumulation,o-projection,residual-add,mlp-gate-up-down,"
     "activation,moe-router,expert-dispatch,expert-accumulation,final-norm,output-head-projection", NULL},
    {"unsupported_graph_ops", GRAPH_FIXED,
     "full-transformer-attention,real-layer-scheduler,real-moe-router,real-expert-dispatch,"
     "real-output-head-projection", NULL},
    {"required_backend_ops", GRAPH_FIXED,
     "tensor-read,rmsnorm,matmul,rope,attention,softmax,activation,residual-add,kv-read,kv-write", NULL},
    {"unsupported_backend_ops", GRAPH_FIXED,
     "full-transformer-runtime-integration,real-attention-backed-kv,real-output-head-logits", NULL},
    {"graph.embedding_lookup", GRAPH_FLAG(has_token_embedding), "planned-real-tensor", "missing-tensor"},
{"graph.rmsnorm", GRAPH_NORMALIZATION, "implemented-selected-segment", "missing-tensor"},
    {"graph.q_projection", GRAPH_FLAG(has_attention_q), "planned", "missing-tensor"},
{"graph.k_projection", GRAPH_FLAG(has_attention_k), "planned", "missing-tensor"},
    {"graph.v_projection", GRAPH_FLAG(has_attention_v), "planned", "missing-tensor"},
{"graph.rope_position_op", GRAPH_FIXED, "implemented-primitive", NULL},
    {"graph.attention_primitive", GRAPH_FIXED, "implemented-fixture", NULL},
{"graph.full_transformer_attention", GRAPH_ATTENTION, "unsupported", "missing-tensor"},
    {"graph.o_projection", GRAPH_FLAG(has_attention_out), "planned", "missing-tensor"},
{"graph.residual_add", GRAPH_FIXED, "planned", NULL},
    {"graph.mlp_primitive", GRAPH_FIXED, "implemented-fixture", NULL},
{"graph.full_transformer_mlp", GRAPH_MLP, "unsupported", "missing-tensor"},
    {"graph.moe_router", GRAPH_FLAG(has_moe_router), "planned", "missing-tensor"},
{"graph.expert_dispatch", GRAPH_FLAG(has_moe_expert), "planned", "missing-tensor"},
    {"graph.output_head_projection", GRAPH_FLAG(has_output_head), "planned", "missing-tensor"},
};
#undef GRAPH_FLAG
#undef GRAPH_NORMALIZATION
#undef GRAPH_MLP
#undef GRAPH_ATTENTION
#undef GRAPH_FIXED
static void fullmodel_print_descriptor_graph_requirements(const yvex_fullmodel_collections *collections)
{
    size_t index;
    for (index = 0; index < sizeof(graph_requirements) / sizeof(graph_requirements[0]); ++index) {
        const graph_requirement_spec *spec = &graph_requirements[index];
        int available = spec->flag_offset == (size_t)-1;
        if (spec->flag_offset == (size_t)-2) available = fullmodel_has_attention_collection(collections);
        else if (spec->flag_offset == (size_t)-3) available = fullmodel_has_mlp_collection(collections);
        else if (spec->flag_offset == (size_t)-4)
            available = fullmodel_has_normalization_collection(collections);
        else if (!available && collections)
            available = *(const int *)((const unsigned char *)collections + spec->flag_offset);
        yvex_cli_out_writef(stdout, "%s: %s\n", spec->key,
                            available ? spec->available : spec->unavailable);
    }
}
typedef enum descriptor_blocker_rule {
    DESCRIPTOR_BLOCKER_COUNT,
    DESCRIPTOR_BLOCKER_ATTENTION,
    DESCRIPTOR_BLOCKER_MLP,
    DESCRIPTOR_BLOCKER_TOKENIZER,
    DESCRIPTOR_BLOCKER_UNKNOWN,
    DESCRIPTOR_BLOCKER_FIXED
} descriptor_blocker_rule;
typedef struct descriptor_collection_spec {
    const char *name;
    const char *descriptor_name;
    const char *descriptor_missing_status;
    size_t count_offset;
    size_t bytes_offset;
    unsigned int required_mask;
    descriptor_blocker_rule blocker_rule;
    const char *missing_blocker;
    const char *runtime_consumer;
} descriptor_collection_spec;
#define COLLECTION_OFF(member_) offsetof(yvex_fullmodel_collections, member_)
#define REQUIRED_MASK(prefill_, decode_, logits_, generation_) \
    ((unsigned int)(prefill_) | ((unsigned int)(decode_) << 1u) | \
     ((unsigned int)(logits_) << 2u) | ((unsigned int)(generation_) << 3u))
static const char *const descriptor_roles[] = {
    "token_embedding", "attention_norm", "post_attention_norm", "final_norm",
    "q_projection", "k_projection", "v_projection", "o_projection",
    "mlp_gate", "mlp_up", "mlp_down", "moe_router", "moe_expert_gate",
    "moe_expert_up", "moe_expert_down", "output_head", "tokenizer_metadata", "unknown"};
static const descriptor_collection_spec descriptor_collections[] = {
    {"embedding", "embedding", "missing", COLLECTION_OFF(embedding), COLLECTION_OFF(embedding_bytes),
REQUIRED_MASK(1, 1, 0, 1), DESCRIPTOR_BLOCKER_COUNT,
     "embedding collection missing", "planned"},
    {"normalization", "normalization", "missing", COLLECTION_OFF(normalization), COLLECTION_OFF(normalization_bytes),
REQUIRED_MASK(1, 1, 1, 1), DESCRIPTOR_BLOCKER_COUNT,
     "normalization collection missing", "planned"},
    {"attention", "attention", "missing", COLLECTION_OFF(attention), COLLECTION_OFF(attention_bytes),
REQUIRED_MASK(1, 1, 0, 1), DESCRIPTOR_BLOCKER_ATTENTION,
     "attention Q/K/V/O tensors missing", "planned"},
    {"mlp", "mlp", "missing", COLLECTION_OFF(mlp), COLLECTION_OFF(mlp_bytes),
REQUIRED_MASK(1, 1, 0, 1), DESCRIPTOR_BLOCKER_MLP, "MLP tensors missing", "planned"},
    {"moe", "moe", "planned-or-missing", COLLECTION_OFF(moe), COLLECTION_OFF(moe_bytes),
REQUIRED_MASK(1, 1, 0, 1), DESCRIPTOR_BLOCKER_COUNT,
     "MoE tensors missing or not identified", "planned"},
    {"output", "output", "missing", COLLECTION_OFF(output), COLLECTION_OFF(output_bytes),
REQUIRED_MASK(0, 1, 1, 1), DESCRIPTOR_BLOCKER_COUNT, "output head missing", "planned"},
    {"tokenizer-runtime-input", "tokenizer", "missing", COLLECTION_OFF(tokenizer), COLLECTION_OFF(tokenizer_bytes),
REQUIRED_MASK(1, 1, 1, 1), DESCRIPTOR_BLOCKER_TOKENIZER,
     "tokenizer metadata missing", "planned"},
    {"kv-cache-runtime", "kv", "unsupported-real-attention-backed-kv", (size_t)-1, (size_t)-1,
REQUIRED_MASK(1, 1, 0, 1),
     DESCRIPTOR_BLOCKER_FIXED, "real attention-backed KV writes unsupported", "unsupported"},
    {"unknown", NULL, NULL, COLLECTION_OFF(unknown), COLLECTION_OFF(unknown_bytes),
REQUIRED_MASK(0, 0, 0, 0), DESCRIPTOR_BLOCKER_UNKNOWN, "unknown tensor role", "unsupported"},
};
#undef REQUIRED_MASK
#undef COLLECTION_OFF
static int descriptor_collection_ready(const descriptor_collection_spec *spec,
                                       const yvex_fullmodel_collections *collections,
                                       unsigned long long count)
{
    switch (spec->blocker_rule) {
    case DESCRIPTOR_BLOCKER_ATTENTION: return fullmodel_has_attention_collection(collections);
    case DESCRIPTOR_BLOCKER_MLP: return fullmodel_has_mlp_collection(collections);
    case DESCRIPTOR_BLOCKER_TOKENIZER:
        return collections && collections->has_tokenizer_metadata;
    case DESCRIPTOR_BLOCKER_UNKNOWN: return count == 0ull;
    case DESCRIPTOR_BLOCKER_FIXED: return 0;
    case DESCRIPTOR_BLOCKER_COUNT: return count > 0ull;
    }
    return 0;
}
static void fullmodel_print_descriptor_inventory(
    yvex_model_context *ctx,
    const yvex_fullmodel_collections *collections,
    const char *backend)
{
    size_t i;
    for (i = 0; i < sizeof(descriptor_roles) / sizeof(descriptor_roles[0]); ++i) {
        fullmodel_print_descriptor_role(ctx, collections, descriptor_roles[i], backend);
    }
    for (i = 0; i < sizeof(descriptor_collections) / sizeof(descriptor_collections[0]); ++i) {
        const descriptor_collection_spec *spec = &descriptor_collections[i];
        unsigned long long count = cli_collection_value(collections, spec->count_offset);
        if (spec->descriptor_name)
            yvex_cli_out_writef(stdout, "%s_descriptor: %s\n", spec->descriptor_name,
                                descriptor_collection_ready(spec, collections, count)
                                    ? "present" : spec->descriptor_missing_status);
    }
    for (i = 0; i < sizeof(descriptor_collections) / sizeof(descriptor_collections[0]); ++i) {
        const descriptor_collection_spec *spec = &descriptor_collections[i];
        unsigned long long count = cli_collection_value(collections, spec->count_offset);
        unsigned long long bytes = cli_collection_value(collections, spec->bytes_offset);
        int ready = descriptor_collection_ready(spec, collections, count);
        fullmodel_print_descriptor_collection(
            spec->name, count, bytes, (spec->required_mask & 1u) != 0u,
            (spec->required_mask & 2u) != 0u, (spec->required_mask & 4u) != 0u,
            (spec->required_mask & 8u) != 0u, spec->runtime_consumer,
            ready ? "none" : spec->missing_blocker);
    }
}
void fullmodel_print_descriptor_report(const yvex_cli_fullmodel_options *options,
                                              yvex_model_ref *ref,
                                              yvex_model_context *ctx,
                                              const char *target_id,
                                              const char *target_class,
                                              unsigned long long artifact_bytes,
                                              yvex_arch arch,
                                              unsigned long long tensor_count,
                                              unsigned long long total_tensor_bytes,
                                              const yvex_fullmodel_collections *collections,
                                              const char *role_coverage,
                                              const char *missing_roles,
                                              const char *unsupported_roles,
                                              int selected_target)
{
    yvex_fullmodel_backend_fit fit;
    const char *backend = options && options->backend ? options->backend : "cpu";
    int descriptor_complete = role_coverage &&
                              (strcmp(role_coverage, "complete") == 0 ||
                               strcmp(role_coverage, "observed") == 0);
    const char *descriptor_status = selected_target ? "partial" :
                                    (descriptor_complete ? "complete" : "partial");
    const char *materialization_plan_status = selected_target ? "partial" : "ready";
    const char *materialization_proof_status = selected_target ? "refused-selected-runtime-slice" :
                                               (descriptor_complete
                                                    ? "available-controlled-tiny-proof"
                                                    : "blocked-missing-roles");
    const char *full_materialization = selected_target ? "refused-selected-runtime-slice" :
                                      (descriptor_complete
                                           ? "controlled-tiny-proof-available"
                                           : "planned");
    unsigned long long cuda_bytes = strcmp(backend, "cuda") == 0 ? total_tensor_bytes : 0ull;
    unsigned long long cpu_bytes = strcmp(backend, "cuda") == 0 ? 0ull : total_tensor_bytes;
    fullmodel_probe_backend_fit(backend, total_tensor_bytes, &fit);
    yvex_cli_out_lines(stdout, literal_pair_16, sizeof(literal_pair_16) / sizeof(literal_pair_16[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n", ref && ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target_id ? target_id : "path");
    yvex_cli_out_writef(stdout, "target_class: %s\n", target_class ? target_class : "candidate-GGUF-path");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "format: %s\n", options && options->format ? options->format : "text");
    yvex_cli_out_writef(stdout, "artifact_identity_status: %s\n", fullmodel_identity_status(ref, artifact_bytes));
    yvex_cli_out_writef(stdout, "tensor_inventory_status: pass\n");
    yvex_cli_out_writef(stdout, "materialization_plan_status: %s\n", materialization_plan_status);
    yvex_cli_out_writef(stdout, "materialization_proof_status: %s\n", materialization_proof_status);
    yvex_cli_out_writef(stdout, "runtime_descriptor: report-only\n");
    yvex_cli_out_writef(stdout, "runtime_descriptor_status: %s\n", descriptor_status);
    yvex_cli_out_writef(stdout, "runtime_descriptor_kind: fullmodel-planning\n");
    yvex_cli_out_writef(stdout, "family: %s\n", fullmodel_family_from_arch(arch));
    yvex_cli_out_writef(stdout, "architecture: %s\n", yvex_arch_name(arch));
    yvex_cli_out_writef(stdout, "model_class: %s\n",
        selected_target ? "selected-runtime-slice" : "descriptor-only-candidate");
    yvex_cli_out_lines(stdout, literal_pair_15, sizeof(literal_pair_15) / sizeof(literal_pair_15[0]));
    yvex_cli_out_writef(stdout, "full_model_materialization: %s\n", full_materialization);
    yvex_cli_out_lines(stdout, literal_lines_2, sizeof(literal_lines_2) / sizeof(literal_lines_2[0]));
    yvex_cli_out_writef(stdout, "tensor_count: %llu\n", tensor_count);
    yvex_cli_out_writef(stdout, "total_tensor_bytes: %llu\n", total_tensor_bytes);
    yvex_cli_out_writef(stdout, "tensor_role_map_status: %s\n", descriptor_status);
    yvex_cli_out_writef(stdout, "tensor_collection_map_status: %s\n", descriptor_status);
    yvex_cli_out_writef(stdout, "required_role_coverage: %s\n",
        descriptor_complete ? "complete" : (role_coverage ? role_coverage : "partial"));
    yvex_cli_out_writef(stdout, "missing_required_roles: %s\n", missing_roles ? missing_roles : "unknown");
    yvex_cli_out_writef(stdout, "unsupported_required_roles: %s\n", unsupported_roles ? unsupported_roles : "unknown");
    yvex_cli_out_writef(stdout, "unknown_role_count: %llu\n", collections ? collections->unknown : 0ull);
    fullmodel_print_descriptor_inventory(ctx, collections, backend);
    fullmodel_print_descriptor_graph_requirements(collections);
    yvex_cli_out_lines(stdout, literal_lines_3, sizeof(literal_lines_3) / sizeof(literal_lines_3[0]));
    yvex_cli_out_writef(stdout, "cpu_resident_required_bytes: %llu\n", cpu_bytes);
    yvex_cli_out_writef(stdout, "cuda_resident_required_bytes: %llu\n", cuda_bytes);
    yvex_cli_out_writef(stdout, "host_staged_required_bytes: %llu\n", strcmp(backend,
        "cuda") == 0 ? total_tensor_bytes : 0ull);
    yvex_cli_out_lines(stdout, literal_lines_4, sizeof(literal_lines_4) / sizeof(literal_lines_4[0]));
    yvex_cli_out_writef(stdout, "output_head_present: %s\n",
        collections && collections->has_output_head ? "true" : "false");
    yvex_cli_out_writef(stdout, "output_head_tensor: %s\n",
           fullmodel_descriptor_find_tensor(ctx, "output_head")
               ? fullmodel_descriptor_find_tensor(ctx, "output_head")->name
               : "none");
    yvex_cli_out_writef(stdout, "output_head_dtype: %s\n",
           fullmodel_descriptor_find_tensor(ctx, "output_head")
               ? yvex_dtype_name(fullmodel_descriptor_find_tensor(ctx, "output_head")->dtype)
               : "unknown");
    yvex_cli_out_writef(stdout, "vocab_size: %s\n",
        collections && collections->has_output_head ? "from-output-head-shape" : "unknown");
    yvex_cli_out_lines(stdout, literal_lines_5, sizeof(literal_lines_5) / sizeof(literal_lines_5[0]));
    yvex_cli_out_writef(stdout, "tokenizer_requirements_status: %s\n",
           collections && collections->has_tokenizer_metadata ? "partial" : "blocked");
    yvex_cli_out_writef(stdout, "tokenizer_metadata_present: %s\n",
           collections && collections->has_tokenizer_metadata ? "true" : "false");
    yvex_cli_out_lines(stdout, literal_lines_6, sizeof(literal_lines_6) / sizeof(literal_lines_6[0]));
    yvex_cli_out_writef(stdout, "backend_requirements_status: %s\n", fit.available ? "planned" : "unsupported");
    yvex_cli_out_writef(stdout, "backend.cpu.available: true\n");
    yvex_cli_out_writef(stdout, "backend.cuda.context_available: %s\n",
        yvex_backend_cuda_context_available() ? "true" : "false");
    yvex_cli_out_writef(stdout, "backend.memory_known: %s\n", fit.memory_known ? "true" : "false");
    yvex_cli_out_writef(stdout, "backend.required_bytes: %llu\n", fit.required_bytes);
    yvex_cli_out_writef(stdout, "backend.fit_status: %s\n", fit.fit_status);
    yvex_cli_out_writef(stdout, "backend.fit_reason: %s\n", fit.fit_reason);
    yvex_cli_out_lines(stdout, literal_lines_7, sizeof(literal_lines_7) / sizeof(literal_lines_7[0]));
    yvex_cli_out_writef(stdout, "runtime_blockers: %s\n",
           selected_target
               ? "full runtime tensor set incomplete; attention Q/K/V/O tensors missing; MLP/MoE tensors "
                   "missing; output head missing; real transformer prefill unsupported; real attention-"
                   "backed KV writes unsupported; full model decode unsupported; real output-head logits "
                   "unsupported; real vocabulary sampling unsupported; full model execution unsupported"
               : "real transformer prefill unsupported; real attention-backed KV writes unsupported; full "
                   "model decode unsupported; real output-head logits runtime unsupported; real vocabulary "
                   "sampling unsupported; full model execution unsupported");
    yvex_cli_out_writef(stdout, "descriptor_blockers: %s\n",
           selected_target
               ? "selected-runtime-slice is partial descriptor only"
               : "runtime family adapter boundary remains planned");
    yvex_cli_out_lines(stdout, literal_lines_8, sizeof(literal_lines_8) / sizeof(literal_lines_8[0]));
    fullmodel_print_descriptor_phases(descriptor_status, descriptor_status, NULL);
}
