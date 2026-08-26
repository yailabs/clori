/*
 * Provide help and typed normal/table/audit rendering for model artifact reports.
 *
 * This renderer formats typed facts only and uses src/cli/io writers. Rendered reports do not
 * imply artifact emission, runtime generation, benchmark evidence, or release readiness.
 */
#include "src/cli/render/private.h"
#include "src/cli/model_artifacts/private.h"
#include "src/cli/io/private.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *key;
    const char *role;
} fullmodel_role_projection;

static const fullmodel_role_projection family_runtime_roles[] = {
    {"token_embedding_role", "token_embedding"},
    {"attention_norm_role", "attention_norm"},
    {"post_attention_norm_role", "post_attention_norm"},
    {"final_norm_role", "final_norm"},
    {"q_projection_role", "q_projection"},
    {"k_projection_role", "k_projection"},
    {"v_projection_role", "v_projection"},
    {"o_projection_role", "o_projection"},
    {"mlp_gate_role", "mlp_gate"},
    {"mlp_up_role", "mlp_up"},
    {"mlp_down_role", "mlp_down"},
    {"moe_router_role", "moe_router"},
    {"output_head_role", "output_head"},
    {"tokenizer_metadata_role", "tokenizer_metadata"},
};

static const yvex_render_field_spec prepare_identity_fields[] = {
    {"target_id", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, target_id), "unknown"},
    {"family", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, family), "unknown"},
    {"provider", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, provider), "unknown"},
    {"repo_id", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, repo_id), "unknown"},
    {"revision", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, revision), "unknown"},
    {"models_root", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, models_root), "unknown"},
    {"source_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, source_path), "unknown"},
    {"source_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, source_status), "unknown"},
    {"source_manifest_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, source_manifest_path), "unknown"},
    {"native_inventory_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, native_inventory_path), "unknown"},
    {"model_class_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, model_class_status), "unknown"},
    {"tensor_map_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, tensor_map_path), "unknown"},
    {"tensor_map_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, tensor_map_status), "unknown"},
    {"output_head_map_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, output_head_map_path), "unknown"},
    {"output_head_map_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, output_head_map_status), "unknown"},
    {"tokenizer_map_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, tokenizer_map_path), "unknown"},
    {"tokenizer_map_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, tokenizer_map_status), "unknown"},
    {"artifact_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, artifact_status), "unknown"},
    {"expected_artifact_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, expected_artifact_path), "unknown"},
    {"artifact_plan_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, artifact_plan_status), "unknown"},
    {"artifact_emission_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, artifact_emission_status), "unknown"},
    {"artifact_identity_status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, artifact_identity_status), "unknown"},
    {"prepare_blocker_count", YVEX_RENDER_FIELD_U32,
     offsetof(yvex_models_prepare_source_report, blocker_count), NULL},
    {"top_blocker", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, top_blocker), "unknown"},
    {"download_registry_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, download_registry_path), "unknown"},
    {"download_report_path", YVEX_RENDER_FIELD_TEXT_ARRAY,
     offsetof(yvex_models_prepare_source_report, download_report_path), "unknown"},
    {"downloaded_target_resolved", YVEX_RENDER_FIELD_BOOL,
     offsetof(yvex_models_prepare_source_report, downloaded_target_resolved), NULL},
};

static const yvex_render_field_spec prepare_result_fields[] = {
    {"reason", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, reason), "unknown"},
    {"next", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, next), "unknown"},
    {"status", YVEX_RENDER_FIELD_TEXT,
     offsetof(yvex_models_prepare_source_report, final_status), "unknown"},
};

#define MATERIALIZE_REPORT_FIELD(key_, kind_, member_, fallback_) \
    {key_, kind_, offsetof(fullmodel_materialize_report, member_), fallback_}

static const yvex_render_field_spec materialize_admission_fields[] = {
    MATERIALIZE_REPORT_FIELD("artifact_identity_status", YVEX_RENDER_FIELD_TEXT,
                             artifact_identity_status, "not-checked"),
    MATERIALIZE_REPORT_FIELD("tensor_inventory_status", YVEX_RENDER_FIELD_TEXT,
                             tensor_inventory_status, "unknown"),
    MATERIALIZE_REPORT_FIELD("required_role_coverage", YVEX_RENDER_FIELD_TEXT,
                             required_role_coverage, "partial"),
    MATERIALIZE_REPORT_FIELD("missing_required_roles", YVEX_RENDER_FIELD_TEXT,
                             missing_required_roles, "unknown"),
    MATERIALIZE_REPORT_FIELD("unsupported_required_roles", YVEX_RENDER_FIELD_TEXT,
                             unsupported_required_roles,
                             "runtime-family-adapter,real-transformer-prefill,real-attention-backed-KV,"
                                 "real-DeepSeek-decode,real-output-head-logits,real-vocabulary-sampling"),
    MATERIALIZE_REPORT_FIELD("placement_plan_status", YVEX_RENDER_FIELD_TEXT,
                             placement_plan_status, "unknown"),
    MATERIALIZE_REPORT_FIELD("memory_budget_status", YVEX_RENDER_FIELD_TEXT,
                             memory_budget_status, "unknown"),
    MATERIALIZE_REPORT_FIELD("backend_preflight_status", YVEX_RENDER_FIELD_TEXT,
                             backend_preflight_status, "unknown"),
    MATERIALIZE_REPORT_FIELD("materialization_mode", YVEX_RENDER_FIELD_TEXT,
                             materialization_mode, "none"),
    MATERIALIZE_REPORT_FIELD("full_model_materialization", YVEX_RENDER_FIELD_TEXT,
                             full_model_materialization, "failed"),
    MATERIALIZE_REPORT_FIELD("full_model_materialization_proof", YVEX_RENDER_FIELD_TEXT,
                             full_model_materialization_proof, "fail"),
};

static const yvex_render_field_spec materialize_lifecycle_fields[] = {
    MATERIALIZE_REPORT_FIELD("phase", YVEX_RENDER_FIELD_TEXT, phase, "failed"),
    MATERIALIZE_REPORT_FIELD("failed_phase", YVEX_RENDER_FIELD_TEXT, failed_phase, "none"),
    MATERIALIZE_REPORT_FIELD("failed_reason", YVEX_RENDER_FIELD_TEXT, failed_reason, "none"),
    MATERIALIZE_REPORT_FIELD("cleanup_attempted", YVEX_RENDER_FIELD_TEXT,
                             cleanup_attempted, "false"),
    MATERIALIZE_REPORT_FIELD("cleanup_status", YVEX_RENDER_FIELD_TEXT,
                             cleanup_status, "not-needed"),
    MATERIALIZE_REPORT_FIELD("cleanup_idempotent", YVEX_RENDER_FIELD_TEXT,
                             cleanup_idempotent, "true"),
    MATERIALIZE_REPORT_FIELD("owned_state_released", YVEX_RENDER_FIELD_TEXT,
                             owned_state_released, "true"),
    MATERIALIZE_REPORT_FIELD("partial_materialization", YVEX_RENDER_FIELD_TEXT,
                             partial_materialization, "false"),
};

static const yvex_render_field_spec materialize_accounting_fields[] = {
    MATERIALIZE_REPORT_FIELD("materialized_tensor_count", YVEX_RENDER_FIELD_U64,
                             materialized_tensor_count, NULL),
    MATERIALIZE_REPORT_FIELD("materialized_tensor_bytes", YVEX_RENDER_FIELD_U64,
                             materialized_tensor_bytes, NULL),
    MATERIALIZE_REPORT_FIELD("refused_tensor_count", YVEX_RENDER_FIELD_U64,
                             refused_tensor_count, NULL),
    MATERIALIZE_REPORT_FIELD("skipped_tensor_count", YVEX_RENDER_FIELD_U64,
                             skipped_tensor_count, NULL),
    MATERIALIZE_REPORT_FIELD("required_tensor_count", YVEX_RENDER_FIELD_U64,
                             required_tensor_count, NULL),
    MATERIALIZE_REPORT_FIELD("required_tensor_bytes", YVEX_RENDER_FIELD_U64,
                             required_tensor_bytes, NULL),
    MATERIALIZE_REPORT_FIELD("peak_planned_bytes", YVEX_RENDER_FIELD_U64,
                             peak_planned_bytes, NULL),
    MATERIALIZE_REPORT_FIELD("cpu_resident_bytes", YVEX_RENDER_FIELD_U64,
                             cpu_resident_bytes, NULL),
    MATERIALIZE_REPORT_FIELD("cuda_resident_bytes", YVEX_RENDER_FIELD_U64,
                             cuda_resident_bytes, NULL),
    MATERIALIZE_REPORT_FIELD("residency_plan", YVEX_RENDER_FIELD_TEXT,
                             residency_plan, "not-planned"),
    MATERIALIZE_REPORT_FIELD("runtime_blockers", YVEX_RENDER_FIELD_TEXT,
                             runtime_blockers, "runtime family adapter not implemented"),
};

#undef MATERIALIZE_REPORT_FIELD

static const char *const literal_pair_0[] = { "family: unknown", "family_detected: unknown"};

static const char *const literal_pair_1[] = { "family_runtime: report", "status: fullmodel-family-runtime-fail"};

static const char *const literal_pair_2[] = { "fullmodel: descriptor", "status: fullmodel-descriptor-fail"};

static const char *const literal_pair_3[] = { "family: unknown", "family_detected: unknown"};

static const char *const literal_pair_4[] = { "family_runtime: report", "status: fullmodel-family-runtime-fail"};

static const char *const literal_pair_5[] = { "fullmodel: descriptor", "status: fullmodel-descriptor-fail"};

static const char *const literal_pair_6[] = { "family: glm", "family_detected: glm"};

static const char *const literal_pair_7[] = {
    "family_runtime: report", "status: fullmodel-family-runtime-unsupported"};

static const char *const literal_pair_8[] = { "fullmodel: descriptor", "status: fullmodel-descriptor-unsupported"};

static const char *const literal_pair_9[] = {
    "plan_kind: full-model-materialization", "plan_source: source-target-without-YVEX-GGUF"};

static const char *const literal_pair_10[] = {
    "fullmodel: materialization-plan", "status: fullmodel-materialization-plan-unsupported"};

static const char *const literal_pair_11[] = { "fullmodel: report", "status: fullmodel-report-unsupported"};

static const char *const literal_pair_12[] = {
    "boundary: descriptor report-only, no runtime execution", "status: fullmodel-descriptor"};

static const char *const literal_pair_13[] = {
    "boundary: plan-only, no materialization", "status: fullmodel-materialization-plan"};

static const char *const literal_pair_14[] = {
    "graph_requirement_status: blocked", "runtime_blocker_status: blocked"};

static const char *const literal_lines_0[] = {
    "prefill_ready: false", "decode_ready: false", "logits_ready: false", "sampling_ready: false",
    "full_model_execution: unsupported", "full_model_materialization: planned", "full_runtime_descriptor: planned",
    "generation_ready: false", "generation: unsupported-full-model", "benchmark_status: not-measured"};

static const char *const literal_lines_9[] = {
    "family_adapter: unsupported", "family_adapter_status: unsupported", "family_runtime_stage: report-only",
    "runtime_claim: unsupported", "generation: unsupported-full-model", "benchmark_status: not-measured",
    "descriptor_status: unavailable", "descriptor_source: fullmodel-descriptor-facts", "full_runtime_model: false",
    "full_model_execution: unsupported", "generation_ready: false", "runtime_execution_ready: false",
    "runtime_blockers: unsupported or unknown runtime family adapter",
    "next_required_rows: FAMILY.RUNTIME.0,ATTENTION.CLASS.0,CONTEXT.CLASS.0,KV.CACHE.0,MOE.CLASS.0,family-"
        "specific-runtime-target",
    "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_10[] = {
    "family_runtime_stage: report-only", "runtime_claim: unsupported", "generation: unsupported-full-model",
    "benchmark_status: not-measured"};

static const char *const literal_lines_11[] = {
    "descriptor_source: fullmodel-descriptor-facts", "full_runtime_model: false",
    "full_model_execution: unsupported", "generation_ready: false"};

static const char *const literal_lines_12[] = {
    "position_rule_status: planned", "kv_rule_status: blocked", "moe_rule_status: blocked"};

static const char *const literal_lines_13[] = {
    "attention_family: deepseek-family-attention-planned", "attention_type: unknown-family-specific",
    "attention_heads: unknown", "kv_heads: unknown", "head_dim: unknown", "attention_q_required: true",
    "attention_k_required: true", "attention_v_required: true", "attention_o_required: true", "rope_required: true",
    "rope_status: planned", "rope_base: unknown", "rope_scaling: unknown", "mask_required: true",
    "mask_rule: causal-or-family-specific-planned", "context_policy: planned", "attention_runtime_ready: false",
    "kv_required: true", "kv_layout: planned", "kv_dtype: planned",
    "kv_capacity_status: unsupported-full-transformer-kv", "kv_write_ready: false", "kv_read_ready: false",
    "moe_required: family-specific-planned", "router_required: family-specific-planned"};

static const char *const literal_lines_14[] = {
    "moe_expert_count: unknown", "expert_count: unknown", "moe_active_expert_count: unknown",
    "active_expert_count: unknown", "moe_shared_experts: unknown", "shared_expert_status: planned",
    "moe_dispatch_ready: false",
    "moe_blockers: router logits, top-k routing, expert dispatch, and expert accumulation are not implemented",
    "output_head_required: true"};

static const char *const literal_lines_15[] = { "logits_projection_ready: false", "real_output_head_logits: false",
    "logits_blockers: real output-head logits runtime unsupported",
    "required_graph_ops: embedding-lookup,rmsnorm,q-projection,k-projection,v-projection,rope-position,"
        "attention-score,causal-mask,softmax,attention-value-accumulation,o-projection,residual-add,mlp,moe-"
        "router,expert-dispatch,final-norm,output-head-projection",
    "implemented_graph_primitives: rope,attention-fixture,matmul,mlp-fixture,controlled-block,controlled-"
        "layers,selected-embedding,selected-rmsnorm-segment",
    "unsupported_graph_ops: full-attention-from-model-tensors,full-transformer-block-from-model-tensors,"
        "full-layer-stack,real-moe-router,real-expert-dispatch,real-output-head-projection",
    "graph.rope_primitive: implemented", "graph.attention_primitive: implemented-fixture",
    "graph.matmul_primitive: implemented", "graph.mlp_primitive: implemented-fixture",
    "graph.full_attention_from_model_tensors: unsupported",
    "graph.full_transformer_block_from_model_tensors: unsupported", "graph.full_layer_stack: unsupported"};

static const char *const literal_lines_16[] = {
    "prefill_ready: false", "decode_ready: false", "logits_ready: false", "sampling_ready: false",
    "runtime_execution_ready: false"};

static const char *const literal_lines_17[] = {
    "next_required_rows: ATTENTION.CLASS.0,CONTEXT.CLASS.0,KV.CACHE.0,MOE.CLASS.0,"
        "FAMILY.RUNTIME.DeepSeek.detail,real-transformer-prefill,real-decode,real-output-head-logits,real-"
        "vocabulary-sampling,GEN.DEEPSEEK.0",
    "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_18[] = {
    "full_model_execution: unsupported", "generation_ready: false", "generation: unsupported-full-model",
    "benchmark_status: not-measured"};

static const char *const literal_lines_19[] = {
    "target_class: official-source-huge-model", "source_artifact_class: official safetensors",
    "target_artifact_class: future YVEX-produced GGUF", "artifact_exists: false", "artifact_bytes: 0",
    "artifact_identity_status: not-applicable", "tensor_count: 0",
    "tensor_inventory_status: not-performed-source-only-target", "metadata_status: not-performed",
    "architecture: glm", "family: glm", "model_class: huge-MoE-source-target",
    "fullmodel_inventory: unsupported-source-only", "qtype_summary: none", "dtype_summary: none",
    "total_tensor_bytes: 0", "estimated_cpu_resident_bytes: unknown", "estimated_cuda_resident_bytes: unknown",
    "estimated_kv_bytes: planned", "estimated_scratch_bytes: planned", "estimated_total_runtime_bytes: unknown"};

static const char *const literal_lines_20[] = {
    "backend_placement_status: not-performed", "cpu_placement: unsupported-source-only",
    "cuda_placement: unsupported-source-only"};

static const char *const literal_lines_21[] = {
    "cuda_memory_status: unavailable", "residency_plan: future-YVEX-produced-artifact-required",
    "tensor_collections_status: not-performed", "collection_detected: no", "collection_supported: false",
    "runtime_consumer: unsupported", "embedding_tensors: 0", "normalization_tensors: 0", "attention_tensors: 0",
    "kv_cache_requirements: planned", "mlp_tensors: 0", "moe_tensors: 0", "output_tensors: 0",
    "tokenizer_tensors: 0", "required_role_coverage: none", "missing_required_roles: YVEX-produced-GGUF-artifact",
    "unsupported_required_roles: GLM-runtime-family-mapping,GLM-YVEX-produced-GGUF-emission,GLM-runtime-execution",
    "runtime_blockers: source-only target; YVEX-produced GGUF emission planned; full model runtime unsupported"};

static const char *const literal_lines_22[] = {
    "target_class: official-source-huge-model", "artifact_exists: false", "artifact_bytes: 0",
    "artifact_identity_status: not-applicable", "tensor_inventory_status: not-performed-source-only-target",
    "tensor_count: 0", "total_tensor_bytes: 0"};

static const char *const literal_lines_23[] = {
    "plan_status: unsupported", "materialization_plan_ready: false", "materialization_attempted: false",
    "full_materialization_proof: false", "full_model_execution: unsupported", "generation_ready: false",
    "generation: unsupported-full-model", "benchmark_status: not-measured"};

static const char *const literal_lines_24[] = {
    "plan_tensor_count: 0", "plan_tensor_bytes: 0", "plan_collection_count: 0", "plan_phase_count: 1",
    "plan_blocker_count: 1", "plan_cleanup_required: false", "plan_cleanup_phases: none",
    "backend_available: unknown", "backend_memory_known: false", "backend_memory_total_bytes: unknown",
    "backend_memory_available_bytes: unknown", "backend_required_bytes: 0", "backend_fit_status: unsupported",
    "backend_fit_reason: source-only target has no YVEX-produced GGUF tensor inventory",
    "backend_allocation_attempted: false"};

static const char *const literal_lines_25[] = {
    "cleanup_plan_required: false", "cleanup_plan_phases: none", "cleanup_idempotent_required: true",
    "cleanup_failure_policy: preserve-failure-report", "next_required_row: FULLMODEL.2",
    "proof_ready_for_fullmodel_2: false",
    "fullmodel_2_blockers: YVEX-produced GGUF artifact missing; full tensor inventory unavailable"};

static const char *const literal_lines_26[] = { "format: text", "artifact_identity_status: not-applicable",
    "tensor_inventory_status: not-performed-source-only-target",
    "materialization_plan_status: unsupported-source-only", "materialization_proof_status: unsupported-source-only",
    "runtime_descriptor: report-only", "runtime_descriptor_status: unsupported",
    "runtime_descriptor_kind: fullmodel-planning", "family: glm", "architecture: glm",
    "model_class: huge-MoE-source-target", "full_runtime_model: false", "full_model_execution: unsupported",
    "full_model_materialization: unsupported-source-only", "generation_ready: false",
    "generation: unsupported-full-model", "benchmark_status: not-measured", "tensor_role_map_status: not-performed",
    "tensor_collection_map_status: not-performed", "required_role_coverage: none",
    "missing_required_roles: YVEX-produced-GGUF-artifact",
    "unsupported_required_roles: GLM-runtime-family-mapping,GLM-YVEX-produced-GGUF-emission,GLM-runtime-execution",
    "unknown_role_count: 0", "embedding_descriptor: not-performed-source-only",
    "normalization_descriptor: not-performed-source-only", "attention_descriptor: not-performed-source-only",
    "mlp_descriptor: not-performed-source-only", "moe_descriptor: not-performed-source-only",
    "output_descriptor: not-performed-source-only", "tokenizer_descriptor: not-performed-source-only",
    "kv_descriptor: unsupported-source-only", "prefill_descriptor: unsupported-source-only",
    "decode_descriptor: unsupported-source-only", "logits_descriptor: unsupported-source-only",
    "sampling_descriptor: unsupported-source-only", "graph_requirements_status: unsupported-source-only",
    "required_graph_ops: planned-after-YVEX-produced-GGUF", "unsupported_graph_ops: GLM-full-transformer-runtime",
    "required_backend_ops: planned-after-YVEX-produced-GGUF", "unsupported_backend_ops: GLM-runtime-execution",
    "residency_requirements_status: unsupported-source-only",
    "residency_plan: future-YVEX-produced-artifact-required", "cpu_resident_required_bytes: unknown",
    "cuda_resident_required_bytes: unknown", "host_staged_required_bytes: unknown",
    "ssd_staged_required_bytes: planned", "kv_required_bytes: planned", "scratch_required_bytes: planned",
    "context_requirements_status: planned", "max_context: unknown", "requested_context: not-requested",
    "context_policy: planned", "position_policy: planned", "rope_policy: planned",
    "kv_requirements_status: unsupported-source-only", "kv_layout: planned", "kv_dtype: planned",
    "kv_layers: unknown", "kv_heads: unknown", "kv_head_dim: unknown",
    "kv_capacity_status: unsupported-source-only", "kv_write_ready: false", "kv_read_ready: false",
    "logits_requirements_status: unsupported-source-only", "output_head_present: false",
    "output_head_dtype: unknown", "vocab_size: unknown", "logits_buffer_required: true",
    "real_output_head_logits: false", "logits_ready: false",
    "runtime_blockers: source-only target; YVEX-produced GGUF emission planned; GLM runtime unsupported",
    "descriptor_blockers: source-only target has no YVEX-produced GGUF tensor inventory", "prefill_ready: false",
    "decode_ready: false", "sampling_ready: false", "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_27[] = {
    "family_adapter: unsupported-source-only", "family_adapter_status: unsupported",
    "family_runtime_stage: report-only", "runtime_claim: unsupported", "generation: unsupported-full-model",
    "benchmark_status: not-measured", "descriptor_status: unsupported-source-only",
    "descriptor_source: not-performed-source-only-target", "full_runtime_model: false",
    "full_model_execution: unsupported", "generation_ready: false", "role_adapter_status: not-performed",
    "collection_adapter_status: not-performed", "attention_rule_status: unsupported-source-only",
    "position_rule_status: unsupported-source-only", "kv_rule_status: unsupported-source-only",
    "moe_rule_status: unsupported-source-only", "mlp_rule_status: unsupported-source-only",
    "output_head_rule_status: unsupported-source-only", "tokenizer_rule_status: unsupported-source-only",
    "graph_requirement_status: unsupported-source-only", "runtime_blocker_status: blocked",
    "tensor_inventory_status: not-performed-source-only-target", "token_embedding_role: not-performed-source-only",
    "attention_norm_role: not-performed-source-only", "q_projection_role: not-performed-source-only",
    "k_projection_role: not-performed-source-only", "v_projection_role: not-performed-source-only",
    "o_projection_role: not-performed-source-only", "output_head_role: not-performed-source-only",
    "tokenizer_metadata_role: not-performed-source-only", "attention_family: glm-family-planned",
    "attention_type: unknown-source-only", "attention_runtime_ready: false", "kv_required: true",
    "kv_layout: planned", "kv_dtype: planned", "kv_capacity_status: unsupported-source-only",
    "kv_write_ready: false", "kv_read_ready: false", "moe_required: true", "router_required: true",
    "router_present: false", "expert_tensors_present: false", "moe_expert_count: unknown",
    "moe_active_expert_count: unknown", "moe_shared_experts: unknown", "moe_dispatch_ready: false",
    "output_head_required: true", "output_head_present: false", "output_head_tensor: none", "vocab_size: unknown",
    "logits_projection_ready: false", "real_output_head_logits: false",
    "required_graph_ops: planned-after-YVEX-produced-GGUF",
    "implemented_graph_primitives: none-for-source-only-target",
    "unsupported_graph_ops: GLM-full-transformer-runtime", "graph.full_attention_from_model_tensors: unsupported",
    "graph.full_transformer_block_from_model_tensors: unsupported", "graph.full_layer_stack: unsupported",
    "full_transformer_graph_ready: false", "prefill_ready: false", "decode_ready: false", "logits_ready: false",
    "sampling_ready: false", "runtime_execution_ready: false",
    "runtime_blockers: source-only target; YVEX-produced GGUF emission planned; GLM runtime family mapping "
        "planned; GLM runtime unsupported",
    "next_required_rows: OWI.HUGE.0,MODEL.CLASS.3,TENSOR.COLLECTION.2,ATTENTION.CLASS.0,KV.CACHE.0,"
        "MOE.CLASS.0,GLM-YVEX-produced-GGUF,GLM-runtime-family-mapping",
    "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_28[] = {
    "plan_status: failed", "materialization_plan_ready: false", "materialization_attempted: false",
    "full_materialization_proof: false", "full_model_execution: unsupported", "generation_ready: false",
    "generation: unsupported-full-model", "benchmark_status: not-measured", "plan_id: unavailable",
    "plan_kind: full-model-materialization", "plan_source: tensor-inventory"};

static const char *const literal_lines_29[] = {
    "plan_tensor_count: 0", "plan_tensor_bytes: 0", "plan_collection_count: 0", "plan_phase_count: 1",
    "plan_blocker_count: 1", "plan_cleanup_required: false", "plan_cleanup_phases: none",
    "backend_available: unknown", "backend_memory_known: false", "backend_memory_total_bytes: unknown",
    "backend_memory_available_bytes: unknown", "backend_required_bytes: 0", "backend_fit_status: unknown",
    "backend_fit_reason: inventory failed before backend fit preflight", "backend_allocation_attempted: false"};

static const char *const literal_lines_30[] = {
    "cleanup_plan_required: false", "cleanup_plan_phases: none", "cleanup_idempotent_required: true",
    "cleanup_failure_policy: preserve-failure-report", "next_required_row: FULLMODEL.2",
    "proof_ready_for_fullmodel_2: false",
    "fullmodel_2_blockers: tensor inventory unavailable; materialization plan unavailable"};

static const char *const literal_lines_31[] = {
    "artifact_identity_status: unavailable", "tensor_inventory_status: failed",
    "materialization_plan_status: unavailable", "materialization_proof_status: unavailable",
    "runtime_descriptor: report-only", "runtime_descriptor_status: fail",
    "runtime_descriptor_kind: fullmodel-planning", "family: unknown", "architecture: unknown",
    "model_class: unresolved", "full_runtime_model: false", "full_model_execution: unsupported",
    "full_model_materialization: unavailable", "generation_ready: false", "generation: unsupported-full-model",
    "benchmark_status: not-measured", "tensor_role_map_status: unavailable",
    "tensor_collection_map_status: unavailable", "required_role_coverage: none", "missing_required_roles: artifact",
    "unsupported_required_roles: full-runtime-model", "runtime_blockers: artifact path missing",
    "descriptor_blockers: artifact path missing", "prefill_ready: false", "decode_ready: false",
    "logits_ready: false", "sampling_ready: false", "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_32[] = {
    "family_adapter: unavailable", "family_adapter_status: failed", "family_runtime_stage: report-only",
    "runtime_claim: unsupported", "generation: unsupported-full-model", "benchmark_status: not-measured",
    "descriptor_status: fail", "descriptor_source: unavailable", "full_runtime_model: false",
    "full_model_execution: unsupported", "generation_ready: false", "runtime_execution_ready: false",
    "runtime_blockers: artifact path missing", "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_33[] = {
    "target_class: unresolved-artifact", "source_artifact_class: unknown", "target_artifact_class: GGUF artifact",
    "artifact_exists: false", "artifact_bytes: 0", "artifact_identity_status: unavailable", "tensor_count: 0",
    "tensor_inventory_status: failed", "metadata_status: failed", "architecture: unknown", "family: unknown",
    "model_class: unresolved", "fullmodel_inventory: unavailable", "qtype_summary: none", "dtype_summary: none",
    "total_tensor_bytes: 0", "estimated_cpu_resident_bytes: unknown", "estimated_cuda_resident_bytes: unknown",
    "estimated_kv_bytes: planned", "estimated_scratch_bytes: planned", "estimated_total_runtime_bytes: unknown"};

static const char *const literal_lines_34[] = {
    "backend_placement_status: failed-missing-artifact", "cpu_placement: unavailable",
    "cuda_placement: unavailable"};

static const char *const literal_lines_35[] = {
    "cuda_memory_status: unavailable", "residency_plan: unavailable", "tensor_collections_status: unavailable",
    "collection_detected: no", "collection_supported: false", "runtime_consumer: unsupported",
    "embedding_tensors: 0", "normalization_tensors: 0", "attention_tensors: 0", "kv_cache_requirements: planned",
    "mlp_tensors: 0", "moe_tensors: 0", "output_tensors: 0", "tokenizer_tensors: 0", "required_role_coverage: none",
    "missing_required_roles: artifact", "unsupported_required_roles: full-runtime-model",
    "runtime_blockers: artifact path missing"};

static const char *const literal_lines_36[] = {
    "artifact_identity_status: not-checked", "tensor_inventory_status: failed",
    "materialization_plan_status: failed", "materialization_proof_status: unavailable",
    "runtime_descriptor: report-only", "runtime_descriptor_status: fail",
    "runtime_descriptor_kind: fullmodel-planning", "family: unknown", "architecture: unknown",
    "model_class: parse-failed", "full_runtime_model: false", "full_model_execution: unsupported",
    "full_model_materialization: unavailable", "generation_ready: false", "generation: unsupported-full-model",
    "benchmark_status: not-measured", "tensor_role_map_status: unavailable",
    "tensor_collection_map_status: unavailable", "required_role_coverage: none",
    "missing_required_roles: parseable-GGUF-tensor-directory", "unsupported_required_roles: full-runtime-model",
    "runtime_blockers: GGUF metadata or tensor directory parse failed",
    "descriptor_blockers: GGUF metadata or tensor directory parse failed", "prefill_ready: false",
    "decode_ready: false", "logits_ready: false", "sampling_ready: false", "cleanup_attempted: false",
    "cleanup_status: not-needed"};

static const char *const literal_lines_37[] = {
    "family_adapter: unavailable", "family_adapter_status: failed", "family_runtime_stage: report-only",
    "runtime_claim: unsupported", "generation: unsupported-full-model", "benchmark_status: not-measured",
    "descriptor_status: fail", "descriptor_source: fullmodel-descriptor-facts", "full_runtime_model: false",
    "full_model_execution: unsupported", "generation_ready: false", "tensor_inventory_status: failed",
    "runtime_execution_ready: false", "runtime_blockers: GGUF metadata or tensor directory parse failed",
    "cleanup_attempted: false", "cleanup_status: not-needed"};

static const char *const literal_lines_38[] = {
    "target_class: GGUF-artifact", "source_artifact_class: unknown", "target_artifact_class: GGUF artifact",
    "artifact_exists: true"};

static const char *const literal_lines_39[] = {
    "artifact_identity_status: not-checked", "tensor_count: 0", "tensor_inventory_status: failed",
    "metadata_status: failed", "architecture: unknown", "family: unknown", "model_class: parse-failed",
    "fullmodel_inventory: unavailable", "qtype_summary: none", "dtype_summary: none", "total_tensor_bytes: 0",
    "estimated_cpu_resident_bytes: unknown", "estimated_cuda_resident_bytes: unknown",
    "estimated_kv_bytes: planned", "estimated_scratch_bytes: planned", "estimated_total_runtime_bytes: unknown"};

static const char *const literal_lines_40[] = {
    "backend_placement_status: failed-parse", "cpu_placement: unavailable", "cuda_placement: unavailable"};

static const char *const literal_lines_41[] = {
    "cuda_memory_status: unavailable", "residency_plan: unavailable", "tensor_collections_status: failed",
    "collection_detected: no", "collection_supported: false", "runtime_consumer: unsupported",
    "embedding_tensors: 0", "normalization_tensors: 0", "attention_tensors: 0", "kv_cache_requirements: planned",
    "mlp_tensors: 0", "moe_tensors: 0", "output_tensors: 0", "tokenizer_tensors: 0", "required_role_coverage: none",
    "missing_required_roles: parseable-GGUF-tensor-directory", "unsupported_required_roles: full-runtime-model",
    "runtime_blockers: GGUF metadata or tensor directory parse failed"};

static const char *const literal_lines_42[] = {
    "alias: yvex inspect model full plan --model FILE_OR_ALIAS [options]", "\nExamples:",
    "  yvex inspect model full report --model deepseek4-v4-flash-dspark-selected-embed --backend cpu",
    "  yvex inspect model full report --model deepseek4-v4-flash-dspark-selected-embed-rmsnorm "
    "--backend cpu --limit-tensors 8",
    "  yvex inspect model full report --model ./candidate.gguf --target deepseek4-v4-flash-dspark --backend cuda",
    "  yvex inspect model full materialization-plan --model "
        "deepseek4-v4-flash-dspark-selected-embed-rmsnorm --backend cpu "
        "--residency resident",
    "  yvex inspect model full materialization-plan --model ./candidate.gguf "
        "--target deepseek4-v4-flash-dspark --backend "
        "cuda --residency hybrid",
    "  yvex inspect model full materialize --model ./tiny-fullish.gguf --backend cpu --limit-bytes 1048576",
    "  yvex inspect model full materialize --model deepseek4-v4-flash-dspark-selected-embed-rmsnorm --backend cpu",
    "  yvex inspect model full descriptor --model ./candidate.gguf --target deepseek4-v4-flash-dspark "
        "--backend cpu --limit-tensors 40",
    "  yvex inspect model full family-runtime --model deepseek4-v4-flash-dspark-selected-embed-rmsnorm "
        "--family deepseek --backend cpu",
    "\nfullmodel report:", "  inventory and placement pressure report."};

static const char *const literal_lines_43[] = {
    "\nfullmodel materialization-plan:", "  planned placement phases and materialization preflight only.",
    "\nfullmodel materialization proof:",
    "  controlled proof over a tiny full-ish GGUF tensor set, or a clean refusal for selected/runtime-"
        "slice and incomplete artifacts.",
    "\nfullmodel descriptor:",
    "  planning/reporting boundary for tensor roles, inspect tensor collections, residency expectations, graph "
        "requirements, prefill/KV/decode/logits/sampling requirements, output-head/tokenizer requirements, "
        "backend requirements, and blockers.",
    "\nfullmodel family-runtime:",
    "  maps descriptor facts into model-family runtime adapter facts. DeepSeek is the first concrete "
        "family report target. Qwen/Metal remains planned unless separately implemented.",
    "\nFullmodel report reads GGUF metadata and tensor-directory facts only. Materialization-plan reuses "
        "those inventory facts to plan collection placement, residency, backend fit, blockers, and cleanup. "
        "Materialize allocates and releases only the bounded required proof tensors that pass role coverage "
        "and byte-limit checks. Descriptor builds a runtime requirement report only. Family-runtime maps "
        "descriptor facts into family-specific tensor roles, collection adapters, attention/KV/MoE/output "
        "requirements, graph requirements, blockers, and next-row dependencies. These reports do not execute "
        "the model, materialize full weights, run graph execution, write real KV, produce real logits, sample "
        "real vocabulary tokens, generate, evaluate, benchmark, or report throughput. They report why full "
        "transformer prefill, decode, logits, and generation remain unsupported.",
    "Boundary: no full model execution, no inference readiness, no DeepSeek generation, no provider "
        "generation, no streaming generation, no eval, no benchmark, no throughput."};

typedef struct {
    const char *model;
    const char *resolved_path;
    const char *target_id;
    const char *target_class;
    const char *backend;
} fullmodel_failure_identity;

typedef struct {
    const char *const *lines;
    size_t count;
} fullmodel_line_group;

typedef struct {
    const yvex_render_field_spec *fields;
    size_t count;
} fullmodel_field_group;

typedef struct {
    fullmodel_line_group descriptor_title;
    fullmodel_line_group descriptor_body;
    fullmodel_line_group family_title;
    fullmodel_line_group family_identity;
    fullmodel_line_group family_body;
    fullmodel_line_group generic_identity;
    fullmodel_line_group generic_backend;
    fullmodel_line_group generic_runtime;
    const char *descriptor_phase;
    const char *family_phase;
    const char *target_class;
    int artifact_bytes_visible;
} fullmodel_failure_layout;

typedef struct {
    const char *empty;
    const char *model;
    const char *resolved_path;
    const char *target_id;
    const char *target_class;
    const char *backend;
    const char *family_detected;
    const char *family_requested;
    const char *adapter_status;
    const char *artifact_identity_status;
    const char *attention_rule_status;
    const char *mlp_rule_status;
    const char *output_head_rule_status;
    const char *tokenizer_rule_status;
    const char *moe_expert_roles;
    const char *router_present;
    const char *expert_tensors_present;
    const char *output_head_present;
    const char *output_head_tensor;
    const char *vocab_size;
    const char *graph_attention_status;
    const char *backend_requirements_status;
    const char *backend_available;
    const char *backend_memory_known;
    const char *backend_fit_status;
    const char *missing_required_roles;
    const char *unsupported_required_roles;
    const char *runtime_blockers;
    unsigned long long tensor_count;
    unsigned long long total_tensor_bytes;
    unsigned long long backend_required_bytes;
} fullmodel_runtime_view;

#define FULLMODEL_LINE_GROUP(lines_) {lines_, sizeof(lines_) / sizeof((lines_)[0])}
#define FAILURE_IDENTITY_FIELD(key_, member_, fallback_) \
    {key_, YVEX_RENDER_FIELD_TEXT, offsetof(fullmodel_failure_identity, member_), fallback_}

static const yvex_render_field_spec failure_core_fields[] = {
    FAILURE_IDENTITY_FIELD("model", model, ""),
    FAILURE_IDENTITY_FIELD("model_resolved_path", resolved_path, ""),
    FAILURE_IDENTITY_FIELD("target_id", target_id, "unknown"),
};

static const yvex_render_field_spec failure_class_fields[] = {
    FAILURE_IDENTITY_FIELD("target_class", target_class, "unresolved-artifact"),
    FAILURE_IDENTITY_FIELD("backend", backend, "cpu"),
};

#define RUNTIME_TEXT(key_, member_, fallback_) \
    {key_, YVEX_RENDER_FIELD_TEXT, offsetof(fullmodel_runtime_view, member_), fallback_}
#define RUNTIME_U64(key_, member_) \
    {key_, YVEX_RENDER_FIELD_U64, offsetof(fullmodel_runtime_view, member_), NULL}

static const yvex_render_field_spec runtime_identity_fields[] = {
    RUNTIME_TEXT("status", empty, "fullmodel-family-runtime"),
    RUNTIME_TEXT("model", model, ""),
    RUNTIME_TEXT("model_resolved_path", resolved_path, ""),
    RUNTIME_TEXT("target_id", target_id, "path"),
    RUNTIME_TEXT("target_class", target_class, "candidate-GGUF-path"),
    RUNTIME_TEXT("backend", backend, "cpu"),
    RUNTIME_TEXT("family", empty, "deepseek"),
    RUNTIME_TEXT("family_detected", family_detected, "unknown"),
    RUNTIME_TEXT("family_requested", family_requested, "auto"),
    RUNTIME_TEXT("family_adapter", empty, "deepseek-runtime-report"),
    RUNTIME_TEXT("family_adapter_status", adapter_status, "partial"),
};

static const yvex_render_field_spec runtime_descriptor_fields[] = {
    RUNTIME_TEXT("descriptor_status", adapter_status, "partial"),
};

static const yvex_render_field_spec runtime_tensor_fields[] = {
    RUNTIME_U64("tensor_count", tensor_count),
    RUNTIME_U64("total_tensor_bytes", total_tensor_bytes),
    RUNTIME_TEXT("artifact_identity_status", artifact_identity_status, "not-checked"),
};

static const yvex_render_field_spec runtime_rule_fields[] = {
    RUNTIME_TEXT("role_adapter_status", adapter_status, "partial"),
    RUNTIME_TEXT("collection_adapter_status", adapter_status, "partial"),
    RUNTIME_TEXT("attention_rule_status", attention_rule_status, "blocked-missing-qkv"),
    RUNTIME_TEXT("attention_rules", attention_rule_status, "blocked-missing-qkv"),
};

static const yvex_render_field_spec runtime_family_rule_fields[] = {
    RUNTIME_TEXT("mlp_rule_status", mlp_rule_status, "blocked-missing-mlp"),
    RUNTIME_TEXT("output_head_rule_status", output_head_rule_status, "blocked-missing-output-head"),
    RUNTIME_TEXT("tokenizer_rule_status", tokenizer_rule_status, "blocked-missing-tokenizer-metadata"),
};

static const yvex_render_field_spec runtime_moe_fields[] = {
    RUNTIME_TEXT("moe_expert_roles", moe_expert_roles, "missing"),
};

static const yvex_render_field_spec runtime_presence_fields[] = {
    RUNTIME_TEXT("router_present", router_present, "false"),
    RUNTIME_TEXT("moe_router_present", router_present, "false"),
    RUNTIME_TEXT("expert_tensors_present", expert_tensors_present, "false"),
};

static const yvex_render_field_spec runtime_output_fields[] = {
    RUNTIME_TEXT("output_head_present", output_head_present, "false"),
    RUNTIME_TEXT("output_head_tensor", output_head_tensor, "none"),
    RUNTIME_TEXT("vocab_size", vocab_size, "unknown"),
};

static const yvex_render_field_spec runtime_graph_fields[] = {
    RUNTIME_TEXT("graph.full_transformer_attention", graph_attention_status, "missing-tensor"),
    RUNTIME_TEXT("full_transformer_graph_ready", empty, "false"),
};

static const yvex_render_field_spec runtime_backend_fields[] = {
    RUNTIME_TEXT("backend_requirements_status", backend_requirements_status, "unsupported"),
    RUNTIME_TEXT("backend_available", backend_available, "false"),
    RUNTIME_TEXT("backend_memory_known", backend_memory_known, "false"),
    RUNTIME_U64("backend_required_bytes", backend_required_bytes),
    RUNTIME_TEXT("backend_fit_status", backend_fit_status, "unknown"),
    RUNTIME_TEXT("backend_allocation_attempted", empty, "false"),
};

static const yvex_render_field_spec runtime_blocker_fields[] = {
    RUNTIME_TEXT("missing_required_roles", missing_required_roles, "unknown"),
    RUNTIME_TEXT("unsupported_required_roles", unsupported_required_roles, "unknown"),
};

static const yvex_render_field_spec runtime_final_fields[] = {
    RUNTIME_TEXT("runtime_blockers", runtime_blockers, "unknown"),
};

#define FULLMODEL_FIELD_GROUP(fields_) {fields_, sizeof(fields_) / sizeof((fields_)[0])}

static const fullmodel_field_group runtime_field_groups[] = {
    FULLMODEL_FIELD_GROUP(runtime_identity_fields),
    FULLMODEL_FIELD_GROUP(runtime_descriptor_fields),
    FULLMODEL_FIELD_GROUP(runtime_tensor_fields),
    FULLMODEL_FIELD_GROUP(runtime_rule_fields),
    FULLMODEL_FIELD_GROUP(runtime_family_rule_fields),
    FULLMODEL_FIELD_GROUP(runtime_moe_fields),
    FULLMODEL_FIELD_GROUP(runtime_presence_fields),
    FULLMODEL_FIELD_GROUP(runtime_output_fields),
    FULLMODEL_FIELD_GROUP(runtime_graph_fields),
    FULLMODEL_FIELD_GROUP(runtime_backend_fields),
    FULLMODEL_FIELD_GROUP(runtime_blocker_fields),
    FULLMODEL_FIELD_GROUP(runtime_final_fields),
};

#undef FULLMODEL_FIELD_GROUP

#undef RUNTIME_U64
#undef RUNTIME_TEXT

static const fullmodel_failure_layout failure_layouts[] = {
    {
        FULLMODEL_LINE_GROUP(literal_pair_5), FULLMODEL_LINE_GROUP(literal_lines_31),
        FULLMODEL_LINE_GROUP(literal_pair_4), FULLMODEL_LINE_GROUP(literal_pair_3),
        FULLMODEL_LINE_GROUP(literal_lines_32), FULLMODEL_LINE_GROUP(literal_lines_33),
        FULLMODEL_LINE_GROUP(literal_lines_34), FULLMODEL_LINE_GROUP(literal_lines_35),
        "resolve-model", "resolve-model", "unresolved-artifact", 0,
    },
    {
        FULLMODEL_LINE_GROUP(literal_pair_2), FULLMODEL_LINE_GROUP(literal_lines_36),
        FULLMODEL_LINE_GROUP(literal_pair_1), FULLMODEL_LINE_GROUP(literal_pair_0),
        FULLMODEL_LINE_GROUP(literal_lines_37), FULLMODEL_LINE_GROUP(literal_lines_38),
        FULLMODEL_LINE_GROUP(literal_lines_40), FULLMODEL_LINE_GROUP(literal_lines_41),
        "tensor-inventory", "load-descriptor", "GGUF-artifact", 1,
    },
};

static const fullmodel_materialize_report materialize_failure_templates[] = {
    {
        .status = "fullmodel-materialize-fail",
        .target_class = "unresolved-artifact",
        .artifact_identity_status = "unavailable",
        .tensor_inventory_status = "failed",
        .required_role_coverage = "none",
        .missing_required_roles = "artifact",
        .unsupported_required_roles = "full-runtime-model",
        .placement_plan_status = "failed",
        .memory_budget_status = "not-performed",
        .backend_preflight_status = "not-performed",
        .materialization_mode = "none",
        .full_model_materialization = "failed",
        .full_model_materialization_proof = "fail",
        .phase = "failed",
        .failed_phase = "resolve-model",
        .cleanup_attempted = "false",
        .cleanup_status = "not-needed",
        .cleanup_idempotent = "true",
        .owned_state_released = "true",
        .partial_materialization = "false",
        .residency_plan = "unavailable",
        .runtime_blockers = "artifact path missing",
    },
    {
        .status = "fullmodel-materialize-fail",
        .target_class = "GGUF-artifact",
        .artifact_identity_status = "not-checked",
        .tensor_inventory_status = "failed",
        .required_role_coverage = "none",
        .missing_required_roles = "parseable-GGUF-tensor-directory",
        .unsupported_required_roles = "full-runtime-model",
        .placement_plan_status = "failed",
        .memory_budget_status = "not-performed",
        .backend_preflight_status = "not-performed",
        .materialization_mode = "none",
        .full_model_materialization = "failed",
        .full_model_materialization_proof = "fail",
        .phase = "failed",
        .failed_phase = "tensor-inventory",
        .cleanup_attempted = "false",
        .cleanup_status = "not-needed",
        .cleanup_idempotent = "true",
        .owned_state_released = "true",
        .partial_materialization = "false",
        .residency_plan = "unavailable",
        .runtime_blockers = "GGUF metadata or tensor directory parse failed",
    },
};

static const fullmodel_materialize_report source_only_materialize_template = {
    .status = "fullmodel-materialize-unsupported",
    .model_resolved_path = "source-only-target",
    .target_class = "official-source-huge-model",
    .artifact_identity_status = "not-applicable",
    .tensor_inventory_status = "not-performed-source-only-target",
    .required_role_coverage = "none",
    .missing_required_roles = "YVEX-produced-GGUF-artifact",
    .unsupported_required_roles =
        "GLM-runtime-family-mapping,GLM-YVEX-produced-GGUF-emission,GLM-runtime-execution",
    .placement_plan_status = "unsupported",
    .memory_budget_status = "not-performed",
    .backend_preflight_status = "not-performed",
    .materialization_mode = "source-only-refusal",
    .full_model_materialization = "unsupported-source-only",
    .full_model_materialization_proof = "unsupported",
    .phase = "failed",
    .failed_phase = "resolve-model",
    .failed_reason = "YVEX-produced-GGUF-artifact-missing",
    .cleanup_attempted = "false",
    .cleanup_status = "not-needed",
    .cleanup_idempotent = "true",
    .owned_state_released = "true",
    .partial_materialization = "false",
    .residency_plan = "future-YVEX-produced-artifact-required",
    .runtime_blockers =
        "source-only target; YVEX-produced GGUF emission planned; full model runtime unsupported",
};

static const char *const fullmodel_usage_lines[] = {
    "usage: yvex inspect model full report --model FILE_OR_ALIAS [--backend cpu|cuda] [--target TARGET] "
        "[--limit-tensors N] [--registry FILE] [--audit | --output normal|table|audit]",
    "usage: yvex inspect model full materialization-plan --model FILE_OR_ALIAS [--backend cpu|cuda] "
        "[--residency resident|host-staged|ssd-staged|hybrid] [--target TARGET] [--limit-tensors N] "
        "[--registry FILE] [--audit | --output normal|table|audit]",
    "usage: yvex inspect model full materialize --model FILE_OR_ALIAS [--backend cpu|cuda] [--registry FILE] "
        "[--dry-run] [--plan-only] [--require-role ROLE] [--require-collection COLLECTION] "
        "[--limit-bytes N] [--fail-after-phase PHASE] [--report-dir DIR] "
        "[--audit | --output normal|table|audit]",
    "usage: yvex inspect model full descriptor --model FILE_OR_ALIAS [--backend cpu|cuda] [--target TARGET] "
        "[--format text] [--limit-tensors N] [--require-role ROLE] [--require-collection COLLECTION] "
        "[--include-blockers] [--include-placement] [--include-graph] [--include-kv] "
        "[--include-logits] [--audit | --output normal|table|audit]",
    "usage: yvex inspect model full family-runtime --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] "
        "[--backend cpu|cuda] [--include-blockers] [--include-roles] [--include-graph] [--include-kv] "
        "[--include-moe] [--include-output] [--audit | --output normal|table|audit]",
};

typedef enum {
    FAMILY_PHASE_PASS,
    FAMILY_PHASE_ADAPTER,
    FAMILY_PHASE_BLOCKED,
    FAMILY_PHASE_FAILURE_MARKER,
} family_phase_kind;

typedef struct {
    const char *name;
    family_phase_kind kind;
} family_phase_spec;

static const family_phase_spec family_runtime_phases[] = {
    {"preflight", FAMILY_PHASE_PASS}, {"resolve-model", FAMILY_PHASE_PASS},
    {"resolve-family", FAMILY_PHASE_PASS}, {"load-descriptor", FAMILY_PHASE_PASS},
    {"family-profile", FAMILY_PHASE_PASS}, {"role-adapter", FAMILY_PHASE_ADAPTER},
    {"collection-adapter", FAMILY_PHASE_ADAPTER}, {"attention-rules", FAMILY_PHASE_BLOCKED},
    {"position-rules", FAMILY_PHASE_BLOCKED}, {"kv-rules", FAMILY_PHASE_BLOCKED},
    {"moe-rules", FAMILY_PHASE_BLOCKED}, {"mlp-rules", FAMILY_PHASE_BLOCKED},
    {"output-head-rules", FAMILY_PHASE_BLOCKED}, {"tokenizer-rules", FAMILY_PHASE_BLOCKED},
    {"graph-requirements", FAMILY_PHASE_BLOCKED},
    {"runtime-phase-blockers", FAMILY_PHASE_BLOCKED}, {"adapter-report", FAMILY_PHASE_PASS},
    {"complete", FAMILY_PHASE_PASS}, {"failed", FAMILY_PHASE_FAILURE_MARKER},
    {"cleanup", FAMILY_PHASE_PASS},
};

static const char *const materialize_phases[] = {
    "preflight", "resolve-model", "artifact-identity", "tensor-inventory", "role-coverage",
    "placement-plan", "memory-budget", "backend-preflight", "materialize-embedding",
    "materialize-normalization", "materialize-attention", "materialize-mlp", "materialize-moe",
    "materialize-output", "materialize-tokenizer", "cleanup", "complete",
};

#undef FAILURE_IDENTITY_FIELD
#undef FULLMODEL_LINE_GROUP

void model_prepare_source_report_render(
    const yvex_models_prepare_source_report *report,
    yvex_models_output_mode mode)
{
    yvex_render_out out;

    if (!report) return;
    if (mode != YVEX_MODELS_OUTPUT_AUDIT) {
        render_out_init(&out, stdout, YVEX_RENDER_MODE_PORCELAIN);
        render_report_title(&out, "models prepare", report->target_id, "blocked");
        render_fields3(&out, "family", report->family,
                       "source", report->source_status,
                       "artifact", report->artifact_status);
        render_fields2(&out, "plan", "full-gguf planned",
                       "emission", report->artifact_emission_status);
        render_top_blocker(&out, report->top_blocker);
        render_next(&out, report->next);
        render_kv(&out, "boundary",
                  "prepare dry-run only; no artifact emission/runtime/generation");
        return;
    }
    render_out_init(&out, stdout, YVEX_RENDER_MODE_AUDIT);
    render_section(&out, "models: prepare");
    render_object_fields(stdout, report, prepare_identity_fields,
                         sizeof(prepare_identity_fields) /
                             sizeof(prepare_identity_fields[0]));
    model_stage_print("target", "unsupported");
    model_print_runtime_generation("not-performed");
    render_object_fields(stdout, report, prepare_result_fields,
                         sizeof(prepare_result_fields) /
                             sizeof(prepare_result_fields[0]));
}

void print_fullmodel_common_boundaries(void)
{
    yvex_cli_out_lines(stdout, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
}

static void fullmodel_print_field_group(const void *object, fullmodel_field_group group)
{
    render_object_fields(stdout, object, group.fields, group.count);
}

static void fullmodel_print_family_runtime_phases(const char *adapter_status,
                                                  const char *failure_phase)
{
    unsigned int i;
    int failed_seen = 0;

    for (i = 0; i < sizeof(family_runtime_phases) / sizeof(family_runtime_phases[0]); ++i) {
        const family_phase_spec *phase = &family_runtime_phases[i];
        const char *status = "pass";
        if (failure_phase && strcmp(failure_phase, phase->name) == 0) {
            status = "fail";
            failed_seen = 1;
        } else if (failed_seen) {
            status = "skipped";
        } else if (phase->kind == FAMILY_PHASE_ADAPTER) {
            status = adapter_status ? adapter_status : "partial";
        } else if (phase->kind == FAMILY_PHASE_BLOCKED) {
            status = "blocked";
        } else if (phase->kind == FAMILY_PHASE_FAILURE_MARKER && !failure_phase) {
            status = "skipped";
        }
        model_phase_print("family_runtime_phase", i, phase->name, status, "planned");
    }
}

static const char *fullmodel_attention_rule_status(const yvex_fullmodel_collections *collections)
{
    if (!collections ||
        !collections->has_attention_q ||
        !collections->has_attention_k ||
        !collections->has_attention_v ||
        !collections->has_attention_out) {
        return "blocked-missing-qkv";
    }
    return "blocked-full-transformer-integration";
}

static int fullmodel_print_unsupported_family(
    const yvex_cli_fullmodel_options *options,
    const yvex_model_ref *ref,
    const char *target_id,
    const char *target_class,
    const char *backend,
    const char *requested,
    const char *detected)
{
    const char *status = strcmp(detected, "unknown") == 0
                             ? "fullmodel-family-runtime-fail"
                             : "fullmodel-family-runtime-unsupported";

    yvex_cli_out_writef(stdout, "status: %s\n", status);
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n", ref && ref->path ? ref->path : "");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target_id ? target_id : "path");
    yvex_cli_out_writef(stdout, "target_class: %s\n",
                        target_class ? target_class : "candidate-GGUF-path");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "family: %s\n", detected);
    yvex_cli_out_writef(stdout, "family_detected: %s\n", detected);
    yvex_cli_out_writef(stdout, "family_requested: %s\n", requested);
    yvex_cli_out_lines(stdout, literal_lines_9, sizeof(literal_lines_9) / sizeof(literal_lines_9[0]));
    fullmodel_print_family_runtime_phases("unsupported", "resolve-family");
    yvex_cli_out_writef(stdout,
                        "reason: requested family is not supported by family-runtime report\n");
    return 5;
}

int fullmodel_print_family_runtime_report(const yvex_cli_fullmodel_options *options,
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
    const yvex_tensor_info *output_head;
    fullmodel_runtime_view view;
    size_t role_index;
    const char *backend = options && options->backend ? options->backend : "cpu";
    const char *requested = model_requested_family(options ? options->family : NULL);
    const char *detected = fullmodel_detect_family(options, arch, target_id);
    const char *adapter_status = selected_target ? "partial" :
                                 (role_coverage && strcmp(role_coverage, "complete") == 0
                                      ? "complete"
                                      : "partial");
    int has_mlp = fullmodel_has_mlp_collection(collections);
    int has_output = collections && collections->has_output_head;
    int supported_family = fullmodel_family_request_matches(requested, detected) &&
                           strcmp(detected, "deepseek") == 0;

    fullmodel_probe_backend_fit(backend, total_tensor_bytes, &fit);
    output_head = fullmodel_descriptor_find_tensor(ctx, "output_head");
    view = (fullmodel_runtime_view){
        .model = options && options->model ? options->model : "",
        .resolved_path = ref && ref->path ? ref->path : "",
        .target_id = target_id,
        .target_class = target_class,
        .backend = backend,
        .family_detected = detected,
        .family_requested = requested,
        .adapter_status = adapter_status,
        .artifact_identity_status = fullmodel_identity_status(ref, artifact_bytes),
        .attention_rule_status = fullmodel_attention_rule_status(collections),
        .mlp_rule_status = has_mlp ? "blocked-full-transformer-integration" : "blocked-missing-mlp",
        .output_head_rule_status = has_output ? "blocked-logits-runtime" : "blocked-missing-output-head",
        .tokenizer_rule_status = collections && collections->has_tokenizer_metadata
                                     ? "partial" : "blocked-missing-tokenizer-metadata",
        .moe_expert_roles = collections && collections->has_moe_expert ? "present" : "missing",
        .router_present = collections && collections->has_moe_router ? "true" : "false",
        .expert_tensors_present = collections && collections->has_moe_expert ? "true" : "false",
        .output_head_present = has_output ? "true" : "false",
        .output_head_tensor = output_head ? output_head->name : "none",
        .vocab_size = has_output ? "from-output-head-shape" : "unknown",
        .graph_attention_status = fullmodel_has_attention_collection(collections)
                                      ? "unsupported" : "missing-tensor",
        .backend_requirements_status = fit.available ? "planned" : "unsupported",
        .backend_available = fit.available ? "true" : "false",
        .backend_memory_known = fit.memory_known ? "true" : "false",
        .backend_fit_status = fit.fit_status,
        .missing_required_roles = missing_roles,
        .unsupported_required_roles = unsupported_roles,
        .runtime_blockers = selected_target
                                ? "selected runtime slice is incomplete; attention Q/K/V/O tensors missing; "
                                  "output head missing; real transformer prefill unsupported; real attention-"
                                  "backed KV unsupported; real DeepSeek decode unsupported; real output-head "
                                  "logits unsupported; real vocabulary sampling unsupported"
                                : "real transformer prefill unsupported; real attention-backed KV unsupported; "
                                  "real DeepSeek decode unsupported; real output-head logits unsupported; real "
                                  "vocabulary sampling unsupported",
        .tensor_count = tensor_count,
        .total_tensor_bytes = total_tensor_bytes,
        .backend_required_bytes = fit.required_bytes,
    };

    yvex_cli_out_writef(stdout, "family_runtime: report\n");
    if (!supported_family) {
        return fullmodel_print_unsupported_family(options, ref, target_id,
                                                   target_class, backend,
                                                   requested, detected);
    }

    fullmodel_print_field_group(&view, runtime_field_groups[0]);
    yvex_cli_out_lines(stdout, literal_lines_10, sizeof(literal_lines_10) / sizeof(literal_lines_10[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[1]);
    yvex_cli_out_lines(stdout, literal_lines_11, sizeof(literal_lines_11) / sizeof(literal_lines_11[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[2]);
    fullmodel_print_field_group(&view, runtime_field_groups[3]);
    yvex_cli_out_lines(stdout, literal_lines_12, sizeof(literal_lines_12) / sizeof(literal_lines_12[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[4]);
    yvex_cli_out_lines(stdout, literal_pair_14, sizeof(literal_pair_14) / sizeof(literal_pair_14[0]));

    for (role_index = 0u;
         role_index < sizeof(family_runtime_roles) / sizeof(family_runtime_roles[0]);
         ++role_index) {
        const fullmodel_role_projection *role = &family_runtime_roles[role_index];
        yvex_cli_out_writef(stdout, "%s: %s\n", role->key,
                            fullmodel_role_status_from_tensor(ctx, collections, role->role));
    }
    fullmodel_print_field_group(&view, runtime_field_groups[5]);
    yvex_cli_out_lines(stdout, literal_lines_13, sizeof(literal_lines_13) / sizeof(literal_lines_13[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[6]);
    yvex_cli_out_lines(stdout, literal_lines_14, sizeof(literal_lines_14) / sizeof(literal_lines_14[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[7]);
    yvex_cli_out_lines(stdout, literal_lines_15, sizeof(literal_lines_15) / sizeof(literal_lines_15[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[8]);
    fullmodel_print_field_group(&view, runtime_field_groups[9]);
    fullmodel_print_field_group(&view, runtime_field_groups[10]);
    yvex_cli_out_lines(stdout, literal_lines_16, sizeof(literal_lines_16) / sizeof(literal_lines_16[0]));
    fullmodel_print_field_group(&view, runtime_field_groups[11]);
    yvex_cli_out_lines(stdout, literal_lines_17, sizeof(literal_lines_17) / sizeof(literal_lines_17[0]));
    fullmodel_print_family_runtime_phases(adapter_status, NULL);
    return 0;
}

static void fullmodel_print_materialize_phase_set(const char *terminal_phase,
                                                  const char *failed_phase)
{
    int failed_seen = 0;
    unsigned int i;

    for (i = 0; i < sizeof(materialize_phases) / sizeof(materialize_phases[0]); ++i) {
        const char *phase = materialize_phases[i];
        const char *status = "planned";
        if (failed_phase && failed_phase[0] && strcmp(failed_phase, phase) == 0) {
            status = "fail";
            failed_seen = 1;
        } else if (!failed_seen &&
                   terminal_phase &&
                   (strcmp(terminal_phase, "complete") == 0 ||
                    strcmp(terminal_phase, phase) == 0)) {
            status = "pass";
        } else if (failed_seen) {
            status = "skipped";
        }
        if (strcmp(phase, "materialize-moe") == 0 && !failed_seen) {
            status = "skipped";
        }
        model_phase_print("materialization_phase", i, phase, status, "planned");
    }
}

void fullmodel_print_materialize_report(const fullmodel_materialize_report *report)
{
    static const fullmodel_materialize_report empty_report;
    const yvex_cli_fullmodel_options *options;

    if (!report) report = &empty_report;
    options = report->options;

    if (options && options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        yvex_cli_out_writef(stdout, "fullmodel-materialize: %s model=%s backend=%s\n",
               report && report->status ? report->status : "fullmodel-materialize-fail",
               options->model ? options->model : "",
               options->backend ? options->backend : "cpu");
        yvex_cli_out_writef(stdout, "reason: %s\n", report && report->failed_reason ? report->failed_reason : "none");
        yvex_cli_out_writef(stdout, "bytes: materialized=%llu required=%llu\n",
               report ? report->materialized_tensor_bytes : 0ull,
               report ? report->required_tensor_bytes : 0ull);
        yvex_cli_out_writef(stdout, "cleanup: %s\n",
            report && report->cleanup_status ? report->cleanup_status : "not-needed");
        yvex_cli_out_writef(stdout, "boundary: bounded proof/refusal only, no full model execution\n");
        yvex_cli_out_writef(stdout, "status: %s\n",
            report && report->status ? report->status : "fullmodel-materialize-fail");
        return;
    }

    yvex_cli_out_writef(stdout, "fullmodel: materialize\n");
    yvex_cli_out_writef(stdout, "status: %s\n",
                        report->status ? report->status : "fullmodel-materialize-fail");
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n",
                        report->model_resolved_path ? report->model_resolved_path : "");
    yvex_cli_out_writef(stdout, "target_id: %s\n", report->target_id ? report->target_id : "path");
    yvex_cli_out_writef(stdout, "target_class: %s\n",
                        report->target_class ? report->target_class : "candidate-GGUF-path");
    yvex_cli_out_writef(stdout, "backend: %s\n", options && options->backend ? options->backend : "cpu");
    yvex_cli_out_writef(stdout, "dry_run: %s\n", options && options->dry_run ? "true" : "false");
    yvex_cli_out_writef(stdout, "plan_only: %s\n", options && options->plan_only ? "true" : "false");
    yvex_cli_out_writef(stdout, "report_dir: %s\n", options && options->report_dir ? options->report_dir : "none");
    render_object_fields(stdout, report, materialize_admission_fields,
                         sizeof(materialize_admission_fields) /
                             sizeof(materialize_admission_fields[0]));
    yvex_cli_out_lines(stdout, literal_lines_18, sizeof(literal_lines_18) / sizeof(literal_lines_18[0]));
    render_object_fields(stdout, report, materialize_lifecycle_fields,
                         sizeof(materialize_lifecycle_fields) /
                             sizeof(materialize_lifecycle_fields[0]));
    render_object_fields(stdout, report, materialize_accounting_fields,
                         sizeof(materialize_accounting_fields) /
                             sizeof(materialize_accounting_fields[0]));
    fullmodel_print_materialize_phase_set(report->phase ? report->phase : "failed",
                                          report->failed_phase);
}

void fullmodel_print_report_normal(const yvex_cli_fullmodel_options *options,
                                          const char *status,
                                          const char *target_id,
                                          const char *target_class,
                                          const char *role_coverage,
                                          const char *top_blocker,
                                          const char *next)
{
    yvex_cli_out_writef(stdout, "fullmodel: report model=%s backend=%s\n",
           options && options->model ? options->model : "",
           options && options->backend ? options->backend : "cpu");
    yvex_cli_out_writef(stdout, "status: %s\n", status ? status : "report-only");
    yvex_cli_out_writef(stdout, "target: %s class=%s\n",
           target_id ? target_id : "path",
           target_class ? target_class : "unknown");
    yvex_cli_out_writef(stdout, "role_coverage: %s\n", role_coverage ? role_coverage : "partial");
    yvex_cli_out_writef(stdout, "top_blocker: %s\n",
        top_blocker ? top_blocker : "missing-full-runtime-tensor-coverage");
    yvex_cli_out_writef(stdout, "next: %s\n", next ? next : "tensor/source/artifact row required");
    yvex_cli_out_writef(stdout, "boundary: report-only, no full model execution\n");
}

void fullmodel_print_plan_normal(const yvex_cli_fullmodel_options *options,
                                        const char *status,
                                        const char *target_class,
                                        const char *fit,
                                        const char *top_blocker)
{
    yvex_cli_out_writef(stdout, "materialization-plan: %s model=%s backend=%s\n",
           status ? status : "blocked",
           options && options->model ? options->model : "",
           options && options->backend ? options->backend : "cpu");
    yvex_cli_out_writef(stdout, "residency: %s\n", options && options->residency ? options->residency : "resident");
    yvex_cli_out_writef(stdout, "class: %s\n", target_class ? target_class : "unknown");
    yvex_cli_out_writef(stdout, "fit: %s\n", fit ? fit : "unknown");
    yvex_cli_out_writef(stdout, "top_blocker: %s\n",
        top_blocker ? top_blocker : "full-runtime candidate artifact required");
    yvex_cli_out_lines(stdout, literal_pair_13, sizeof(literal_pair_13) / sizeof(literal_pair_13[0]));
}

void fullmodel_print_descriptor_normal(const yvex_cli_fullmodel_options *options,
                                              const char *target_id,
                                              const char *target_class,
                                              const char *role_coverage,
                                              const char *missing_roles)
{
    yvex_cli_out_writef(stdout, "report: fullmodel-descriptor\n");
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "backend: %s target=%s class=%s\n",
           options && options->backend ? options->backend : "cpu",
           target_id ? target_id : "path",
           target_class ? target_class : "unknown");
    yvex_cli_out_writef(stdout, "role_coverage: %s\n", role_coverage ? role_coverage : "partial");
    yvex_cli_out_writef(stdout, "top_blocker: %s\n",
           missing_roles && strcmp(missing_roles, "none") == 0
               ? "runtime integration missing"
               : "missing-full-runtime-tensor-coverage");
    yvex_cli_out_lines(stdout, literal_pair_12, sizeof(literal_pair_12) / sizeof(literal_pair_12[0]));
}

void fullmodel_print_family_runtime_normal(const yvex_cli_fullmodel_options *options,
                                                  const char *target_id,
                                                  const char *target_class,
                                                  const char *role_coverage,
                                                  const char *missing_roles)
{
    yvex_cli_out_writef(stdout, "report: family-runtime\n");
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "family: %s backend=%s\n",
           options && options->family ? options->family : "auto",
           options && options->backend ? options->backend : "cpu");
    yvex_cli_out_writef(stdout, "target: %s class=%s\n",
           target_id ? target_id : "path",
           target_class ? target_class : "unknown");
    yvex_cli_out_writef(stdout, "status: %s\n", role_coverage ? role_coverage : "partial");
    yvex_cli_out_writef(stdout, "top_blocker: %s\n",
           missing_roles && strcmp(missing_roles, "none") == 0
               ? "runtime family adapter missing"
               : "missing family runtime tensor roles");
    yvex_cli_out_writef(stdout, "boundary: report-only, no runtime execution\n");
}

int print_fullmodel_source_only_report(const char *target,
                                              const char *backend)
{
    yvex_cli_out_lines(stdout, literal_pair_11, sizeof(literal_pair_11) / sizeof(literal_pair_11[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", target ? target : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: source-only-target\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target ? target : "");
    yvex_cli_out_lines(stdout, literal_lines_19, sizeof(literal_lines_19) / sizeof(literal_lines_19[0]));
    yvex_cli_out_writef(stdout, "backend: %s\n", backend ? backend : "cpu");
    yvex_cli_out_lines(stdout, literal_lines_20, sizeof(literal_lines_20) / sizeof(literal_lines_20[0]));
    yvex_cli_out_writef(stdout, "cuda_context_available: %s\n",
        yvex_backend_cuda_context_available() ? "true" : "false");
    yvex_cli_out_lines(stdout, literal_lines_21, sizeof(literal_lines_21) / sizeof(literal_lines_21[0]));
    print_fullmodel_common_boundaries();
    return 0;
}

typedef enum {
    DESCRIPTOR_PHASE_PASS,
    DESCRIPTOR_PHASE_ROLE,
    DESCRIPTOR_PHASE_COLLECTION,
    DESCRIPTOR_PHASE_PLANNED,
    DESCRIPTOR_PHASE_BLOCKED,
    DESCRIPTOR_PHASE_FAILURE_MARKER
} descriptor_phase_kind;

typedef struct {
    const char *name;
    descriptor_phase_kind kind;
} descriptor_phase_spec;

static const descriptor_phase_spec descriptor_phases[] = {
    {"preflight", DESCRIPTOR_PHASE_PASS}, {"resolve-model", DESCRIPTOR_PHASE_PASS},
{"artifact-identity", DESCRIPTOR_PHASE_PASS}, {"tensor-inventory", DESCRIPTOR_PHASE_PASS},
    {"role-map", DESCRIPTOR_PHASE_ROLE}, {"collection-map", DESCRIPTOR_PHASE_COLLECTION},
{"shape-requirements", DESCRIPTOR_PHASE_PASS},
    {"residency-requirements", DESCRIPTOR_PHASE_PLANNED}, {"graph-requirements", DESCRIPTOR_PHASE_PLANNED},
{"prefill-requirements", DESCRIPTOR_PHASE_BLOCKED}, {"kv-requirements", DESCRIPTOR_PHASE_BLOCKED},
    {"decode-requirements", DESCRIPTOR_PHASE_BLOCKED}, {"logits-requirements", DESCRIPTOR_PHASE_BLOCKED},
{"sampling-requirements", DESCRIPTOR_PHASE_BLOCKED}, {"tokenizer-requirements", DESCRIPTOR_PHASE_PASS},
    {"backend-requirements", DESCRIPTOR_PHASE_PLANNED},
{"blocker-report", DESCRIPTOR_PHASE_PASS}, {"descriptor-build", DESCRIPTOR_PHASE_PASS},
    {"complete", DESCRIPTOR_PHASE_PASS}, {"failed", DESCRIPTOR_PHASE_FAILURE_MARKER},
{"cleanup", DESCRIPTOR_PHASE_PASS},
};

static const char *descriptor_phase_status(descriptor_phase_kind kind,
                                           const char *role_status,
                                           const char *collection_status,
                                           int failed_seen,
                                           int failure_here,
                                           int has_failure)
{
    if (failure_here) return "fail";
    if (failed_seen) return "skipped";
    if (kind == DESCRIPTOR_PHASE_ROLE) return role_status ? role_status : "partial";
    if (kind == DESCRIPTOR_PHASE_COLLECTION)
        return collection_status ? collection_status : "partial";
    if (kind == DESCRIPTOR_PHASE_PLANNED) return "planned";
    if (kind == DESCRIPTOR_PHASE_BLOCKED) return "blocked";
    if (kind == DESCRIPTOR_PHASE_FAILURE_MARKER && !has_failure) return "skipped";
    return "pass";
}

void fullmodel_print_descriptor_phases(const char *role_status,
                                       const char *collection_status,
                                       const char *failure_phase)
{
    size_t index;
    int failed_seen = 0;

    for (index = 0; index < sizeof(descriptor_phases) / sizeof(descriptor_phases[0]); ++index) {
        const descriptor_phase_spec *phase = &descriptor_phases[index];
        int failure_here = failure_phase && strcmp(failure_phase, phase->name) == 0;
        const char *status = descriptor_phase_status(phase->kind, role_status, collection_status,
                                                     failed_seen, failure_here, failure_phase != NULL);
        model_phase_print("descriptor_phase", (unsigned int)index, phase->name, status, "planned");
        failed_seen |= failure_here;
    }
}

int print_fullmodel_source_only_plan(const yvex_cli_fullmodel_options *options,
                                            const char *target)
{
    const char *backend = options && options->backend ? options->backend : "cpu";
    const char *residency = options && options->residency ? options->residency : "resident";

    yvex_cli_out_lines(stdout, literal_pair_10, sizeof(literal_pair_10) / sizeof(literal_pair_10[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : target ? target : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: source-only-target\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target ? target : "");
    yvex_cli_out_lines(stdout, literal_lines_22, sizeof(literal_lines_22) / sizeof(literal_lines_22[0]));
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "residency: %s\n", residency);
    yvex_cli_out_lines(stdout, literal_lines_23, sizeof(literal_lines_23) / sizeof(literal_lines_23[0]));
    yvex_cli_out_writef(stdout, "plan_id: fullmodel-materialization:%s:%s:%s\n",
           target ? target : "source-only-target", backend, residency);
    yvex_cli_out_lines(stdout, literal_pair_9, sizeof(literal_pair_9) / sizeof(literal_pair_9[0]));
    yvex_cli_out_writef(stdout, "plan_backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "plan_residency: %s\n", residency);
    yvex_cli_out_lines(stdout, literal_lines_24, sizeof(literal_lines_24) / sizeof(literal_lines_24[0]));
    fullmodel_print_phase(0u, "preflight", "unsupported",
                          0ull, 0ull, "not-applicable", 1, 1,
                          "YVEX-produced GGUF artifact required before planning");
    (void)fullmodel_print_blocker(0u, "artifact", "fatal",
                                  "YVEX-produced GGUF artifact required before materialization planning",
                                  1, 1);
    yvex_cli_out_lines(stdout, literal_lines_25, sizeof(literal_lines_25) / sizeof(literal_lines_25[0]));
    return 5;
}

int print_fullmodel_source_only_materialize(const yvex_cli_fullmodel_options *options,
                                                   const char *target)
{
    fullmodel_materialize_report report = source_only_materialize_template;
    report.options = options;
    report.target_id = target ? target : "source-only-target";
    fullmodel_print_materialize_report(&report);
    return 5;
}

int print_fullmodel_source_only_descriptor(const yvex_cli_fullmodel_options *options,
                                                  const char *target)
{
    const char *backend = options && options->backend ? options->backend : "cpu";

    yvex_cli_out_lines(stdout, literal_pair_8, sizeof(literal_pair_8) / sizeof(literal_pair_8[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : target ? target : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: source-only-target\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target ? target : "");
    yvex_cli_out_writef(stdout, "target_class: official-source-huge-model\n");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_lines(stdout, literal_lines_26, sizeof(literal_lines_26) / sizeof(literal_lines_26[0]));
    fullmodel_print_descriptor_phases("unsupported", "unsupported", "resolve-model");
    return 5;
}

int print_fullmodel_source_only_family_runtime(const yvex_cli_fullmodel_options *options,
                                                      const char *target)
{
    const char *backend = options && options->backend ? options->backend : "cpu";
    const char *requested = model_requested_family(options ? options->family : NULL);

    yvex_cli_out_lines(stdout, literal_pair_7, sizeof(literal_pair_7) / sizeof(literal_pair_7[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : target ? target : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: source-only-target\n");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target ? target : "");
    yvex_cli_out_writef(stdout, "target_class: official-source-huge-model\n");
    yvex_cli_out_writef(stdout, "backend: %s\n", backend);
    yvex_cli_out_lines(stdout, literal_pair_6, sizeof(literal_pair_6) / sizeof(literal_pair_6[0]));
    yvex_cli_out_writef(stdout, "family_requested: %s\n", requested);
    yvex_cli_out_lines(stdout, literal_lines_27, sizeof(literal_lines_27) / sizeof(literal_lines_27[0]));
    fullmodel_print_family_runtime_phases("unsupported", "resolve-model");
    return 5;
}

static void print_fullmodel_failed_plan_fields(const yvex_cli_fullmodel_options *options,
                                               const char *phase,
                                               const char *reason,
                                               unsigned long long artifact_bytes)
{
    const char *backend = options && options->backend ? options->backend : "cpu";
    const char *residency = options && options->residency ? options->residency : "resident";

    yvex_cli_out_writef(stdout, "residency: %s\n", residency);
    yvex_cli_out_lines(stdout, literal_lines_28, sizeof(literal_lines_28) / sizeof(literal_lines_28[0]));
    yvex_cli_out_writef(stdout, "plan_backend: %s\n", backend);
    yvex_cli_out_writef(stdout, "plan_residency: %s\n", residency);
    yvex_cli_out_lines(stdout, literal_lines_29, sizeof(literal_lines_29) / sizeof(literal_lines_29[0]));
    fullmodel_print_phase(0u, phase ? phase : "preflight", "blocked",
                          0ull, artifact_bytes, "not-applicable", 1, 1,
                          reason ? reason : "inventory failed");
    (void)fullmodel_print_blocker(0u, "inventory", "fatal",
                                  reason ? reason : "inventory failed",
                                  1, 1);
    yvex_cli_out_lines(stdout, literal_lines_30, sizeof(literal_lines_30) / sizeof(literal_lines_30[0]));
}

static void fullmodel_print_line_group(fullmodel_line_group group)
{
    yvex_cli_out_lines(stdout, group.lines, group.count);
}

static int fullmodel_print_failure(const yvex_cli_fullmodel_options *options,
                                   const yvex_model_ref *ref,
                                   const char *resolved_path,
                                   const char *reason,
                                   unsigned long long artifact_bytes,
                                   int rc,
                                   unsigned int layout_index)
{
    const fullmodel_failure_layout *layout = &failure_layouts[layout_index];
    const char *fallback_reason = layout_index ? "parse failed" : "artifact path does not exist";
    const char *target_id = options && options->target ? options->target
                            : ref && ref->alias && ref->alias[0] ? ref->alias
                            : layout_index ? "path" : "unknown";
    fullmodel_failure_identity identity = {
        options && options->model ? options->model : "", resolved_path, target_id,
        layout->target_class, options && options->backend ? options->backend : "cpu",
    };
    int is_plan = options && options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN;
    int is_materialize = options && options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZE;
    int is_descriptor = options && options->command == YVEX_FULLMODEL_COMMAND_DESCRIPTOR;
    int is_family = options && options->command == YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME;

    reason = reason ? reason : fallback_reason;
    if (is_materialize) {
        fullmodel_materialize_report report = materialize_failure_templates[layout_index];
        report.options = options;
        report.model_resolved_path = resolved_path;
        report.target_id = target_id;
        report.failed_reason = reason;
        fullmodel_print_materialize_report(&report);
        if (layout->artifact_bytes_visible) yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", artifact_bytes);
        yvex_cli_out_writef(stdout, "reason: %s\n", reason);
        return exit_for_status(rc);
    }
    if (is_descriptor) {
        fullmodel_print_line_group(layout->descriptor_title);
        render_object_fields(stdout, &identity, failure_core_fields,
                             sizeof(failure_core_fields) / sizeof(failure_core_fields[0]));
        render_object_fields(stdout, &identity, failure_class_fields,
                             sizeof(failure_class_fields) / sizeof(failure_class_fields[0]));
        fullmodel_print_line_group(layout->descriptor_body);
        fullmodel_print_descriptor_phases("fail", "fail", layout->descriptor_phase);
        if (layout->artifact_bytes_visible) yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", artifact_bytes);
        yvex_cli_out_writef(stdout, "reason: %s\n", reason);
        return exit_for_status(rc);
    }
    if (is_family) {
        fullmodel_print_line_group(layout->family_title);
        render_object_fields(stdout, &identity, failure_core_fields,
                             sizeof(failure_core_fields) / sizeof(failure_core_fields[0]));
        render_object_fields(stdout, &identity, failure_class_fields,
                             sizeof(failure_class_fields) / sizeof(failure_class_fields[0]));
        fullmodel_print_line_group(layout->family_identity);
        yvex_cli_out_writef(stdout, "family_requested: %s\n",
                            model_requested_family(options ? options->family : NULL));
        fullmodel_print_line_group(layout->family_body);
        fullmodel_print_family_runtime_phases("failed", layout->family_phase);
        if (layout->artifact_bytes_visible) yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", artifact_bytes);
        yvex_cli_out_writef(stdout, "reason: %s\n", reason);
        return exit_for_status(rc);
    }
    yvex_cli_out_writef(stdout, "fullmodel: %s\n", is_plan ? "materialization-plan" : "report");
    yvex_cli_out_writef(stdout, "status: %s\n",
                        is_plan ? "fullmodel-materialization-plan-fail" : "fullmodel-report-fail");
    render_object_fields(stdout, &identity, failure_core_fields,
                         sizeof(failure_core_fields) / sizeof(failure_core_fields[0]));
    fullmodel_print_line_group(layout->generic_identity);
    if (layout_index) {
        yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", artifact_bytes);
        yvex_cli_out_lines(stdout, literal_lines_39, sizeof(literal_lines_39) / sizeof(literal_lines_39[0]));
    }
    yvex_cli_out_writef(stdout, "backend: %s\n", identity.backend);
    fullmodel_print_line_group(layout->generic_backend);
    yvex_cli_out_writef(stdout, "cuda_context_available: %s\n",
                        yvex_backend_cuda_context_available() ? "true" : "false");
    fullmodel_print_line_group(layout->generic_runtime);
    print_fullmodel_common_boundaries();
    if (is_plan) {
        print_fullmodel_failed_plan_fields(options, layout_index ? "tensor-directory" : "preflight",
                                           reason, artifact_bytes);
    }
    yvex_cli_out_writef(stdout, "reason: %s\n", reason);
    return exit_for_status(rc);
}

int print_fullmodel_missing_report(const yvex_cli_fullmodel_options *options,
                                          const char *resolved_path)
{
    return fullmodel_print_failure(options, NULL, resolved_path ? resolved_path : "",
                                   "artifact path does not exist", 0ull, YVEX_ERR_IO, 0u);
}

int print_fullmodel_parse_failure_report(const yvex_cli_fullmodel_options *options,
                                                const yvex_model_ref *ref,
                                                const char *reason,
                                                int rc)
{
    unsigned long long artifact_bytes = 0ull;

    fullmodel_file_size(ref && ref->path ? ref->path : "", &artifact_bytes);
    return fullmodel_print_failure(options, ref, ref && ref->path ? ref->path : "", reason,
                                   artifact_bytes, rc, 1u);
}

void yvex_fullmodel_help(FILE *fp)
{
    yvex_cli_out_lines(fp, fullmodel_usage_lines,
                       sizeof(fullmodel_usage_lines) / sizeof(fullmodel_usage_lines[0]));
    yvex_cli_out_lines(fp, literal_lines_42, sizeof(literal_lines_42) / sizeof(literal_lines_42[0]));
    yvex_cli_out_line(fp, "  Default output is compact. Use --audit for full diagnostic fields.");
    yvex_cli_out_lines(fp, literal_lines_43, sizeof(literal_lines_43) / sizeof(literal_lines_43[0]));
}

typedef struct {
    const yvex_remote_model *model;
    unsigned long long index;
} remote_catalog_row;

static int remote_catalog_row_compare(const void *left, const void *right)
{
    const remote_catalog_row *a = left;
    const remote_catalog_row *b = right;
    const char *a_family = a->model->family[0] ? a->model->family : "unknown";
    const char *b_family = b->model->family[0] ? b->model->family : "unknown";
    int order = strcmp(a_family, b_family);

    return order ? order : strcmp(a->model->repository, b->model->repository);
}

static void remote_parameter_text(char *out,
                                  size_t capacity,
                                  const yvex_remote_model *model)
{
    double billions;

    if (!model->parameter_count_known) {
        snprintf(out, capacity, "unknown");
        return;
    }
    billions = (double)model->parameter_count / 1000000000.0;
    if (billions >= 1.0)
        snprintf(out, capacity, "%.1fB", billions);
    else
        snprintf(out, capacity, "%.1fM", (double)model->parameter_count / 1000000.0);
}

static void remote_representation_classes(char *out,
                                          size_t capacity,
                                          const yvex_remote_catalog *catalog,
                                          unsigned long long model_index)
{
    const yvex_remote_model *model = yvex_remote_catalog_at(catalog, model_index);
    unsigned int index;

    out[0] = '\0';
    if (!model) return;
    for (index = 0u; index < model->representation_count; ++index) {
        const yvex_model_representation *representation =
            yvex_remote_catalog_representation_at(catalog, model_index, index);
        size_t used = strlen(out);
        if (!representation || strstr(out, representation->format)) continue;
        snprintf(out + used, capacity - used, "%s%s", used ? "," : "",
                 representation->format);
    }
    if (!out[0]) snprintf(out, capacity, "unknown");
}

static int remote_catalog_render_table(FILE *fp,
                                       const yvex_remote_catalog *catalog,
                                       int representations)
{
    unsigned long long count = yvex_remote_catalog_count(catalog);
    unsigned long long row;
    remote_catalog_row *rows;
    const char *last_family = NULL;

    if (representations && count == 1u) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, 0u);
        char parameters[32];
        unsigned int index;

        remote_parameter_text(parameters, sizeof(parameters), model);
        yvex_cli_out_writef(fp, "MODEL  %s\n", model->repository);
        yvex_cli_out_writef(fp, "family=%s  revision=%s  parameters=%s  gated=%s\n\n",
                            model->family[0] ? model->family : "unknown",
                            model->resolved_revision[0] ? model->resolved_revision : "unknown",
                            parameters, model->gated_known ? (model->gated ? "yes" : "no")
                                                          : "unknown");
        yvex_cli_out_writef(fp, "%-23s %-13s %-18s %10s %6s %-9s %s\n",
                            "REPRESENTATION", "FORMAT", "PRECISION/QTYPE", "SIZE", "FILES",
                            "LOCAL", "YVEX COMPATIBILITY");
        for (index = 0u; index < model->representation_count; ++index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, 0u, index);
            char size[32];
            if (!representation) continue;
            if (representation->size_known)
                model_download_format_bytes(size, sizeof(size), representation->size_bytes);
            else
                snprintf(size, sizeof(size), "unknown");
            yvex_cli_out_writef(fp, "%-23.23s %-13.13s %-18.18s %10.10s %6llu %-9s %s%s\n",
                                representation->identity, representation->format,
                                representation->precision[0] ? representation->precision
                                                              : "unknown",
                                size, representation->file_count,
                                representation->local ? "yes" : "no",
                                representation->compatibility,
                                representation->provisional ? " (provisional)" : "");
        }
        yvex_cli_out_writef(fp, "\nsupport: %s\nreason: %s\n",
                            yvex_model_support_stage_name(model->support_stage),
                            model->support_reason);
        yvex_cli_out_writef(fp, "provider_files: %u (use --audit or --json for exact paths)\n",
                            model->available_file_count);
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    rows = calloc(count ? (size_t)count : 1u, sizeof(*rows));
    if (!rows) return YVEX_ERR_NOMEM;
    for (row = 0u; row < count; ++row) {
        rows[row].model = yvex_remote_catalog_at(catalog, row);
        rows[row].index = row;
    }
    qsort(rows, (size_t)count, sizeof(*rows), remote_catalog_row_compare);
    yvex_cli_out_writef(fp, "REMOTE MODELS  count=%llu\n", count);
    for (row = 0u; row < count; ++row) {
        const yvex_remote_model *model = rows[row].model;
        const char *family = model->family[0] ? model->family : "unknown";
        char parameters[32];
        char classes[64];

        if (!last_family || strcmp(last_family, family) != 0) {
            yvex_cli_out_writef(fp, "\nFAMILY %s\n", family);
            yvex_cli_out_writef(fp, "%-44s %9s %-18s %-7s %-22s %s\n", "MODEL / REPOSITORY",
                                "PARAMS", "REPRESENTATIONS", "GATED", "YVEX STAGE", "LOCAL");
            last_family = family;
        }
        remote_parameter_text(parameters, sizeof(parameters), model);
        remote_representation_classes(classes, sizeof(classes), catalog, rows[row].index);
        yvex_cli_out_writef(fp, "%-44.44s %9s %-18.18s %-7s %-22.22s %s\n",
                            model->repository, parameters, classes,
                            model->gated_known ? (model->gated ? "yes" : "no") : "unknown",
                            yvex_model_support_stage_name(model->support_stage),
                            model->local ? "yes" : "no");
    }
    free(rows);
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

static int remote_catalog_render_audit(FILE *fp, const yvex_remote_catalog *catalog)
{
    unsigned long long model_index;

    yvex_cli_out_writef(fp, "remote_models: %llu\n", yvex_remote_catalog_count(catalog));
    for (model_index = 0u; model_index < yvex_remote_catalog_count(catalog); ++model_index) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, model_index);
        unsigned int representation_index;
        yvex_cli_out_writef(fp, "model[%llu]:\n", model_index);
        yvex_cli_out_writef(fp, "  provider: %s\n  repository: %s\n  author: %s\n",
                            model->provider, model->repository,
                            model->author[0] ? model->author : "unknown");
        yvex_cli_out_writef(fp, "  revision_reference: %s\n  resolved_revision: %s\n",
                            model->revision_reference[0] ? model->revision_reference : "unknown",
                            model->resolved_revision[0] ? model->resolved_revision : "unknown");
        yvex_cli_out_writef(fp, "  family: %s\n  family_evidence: %s\n  architecture: %s\n",
                            model->family[0] ? model->family : "unknown",
                            model->family_evidence[0] ? model->family_evidence : "unknown",
                            model->architecture[0] ? model->architecture : "unknown");
        yvex_cli_out_writef(fp, "  base_model: %s\n  lineage_relation: %s\n",
                            model->base_model[0] ? model->base_model : "unknown",
                            model->lineage_relation[0] ? model->lineage_relation : "unknown");
        yvex_cli_out_writef(fp, "  support_stage: %s\n  support_reason: %s\n  local: %s\n",
                            yvex_model_support_stage_name(model->support_stage),
                            model->support_reason, model->local ? "true" : "false");
        for (representation_index = 0u; representation_index < model->representation_count;
             ++representation_index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, model_index,
                                                       representation_index);
            yvex_cli_out_writef(
                fp,
                "  representation[%u]: identity=%s format=%s precision=%s evidence=%s "
                "files=%llu bytes=%llu size_known=%s selector=%s local=%s compatibility=%s "
                "recommendation=%s\n",
                representation_index, representation->identity, representation->format,
                representation->precision[0] ? representation->precision : "unknown",
                representation->precision_evidence[0] ? representation->precision_evidence
                                                       : "unknown",
                representation->file_count, representation->size_bytes,
                representation->size_known ? "true" : "false",
                representation->file_pattern[0] ? representation->file_pattern : "multiple",
                representation->local ? "true" : "false", representation->compatibility,
                representation->recommendation);
        }
        for (representation_index = 0u; representation_index < model->available_file_count;
             ++representation_index) {
            const yvex_remote_file *file =
                yvex_remote_catalog_file_at(catalog, model_index, representation_index);
            yvex_cli_out_writef(fp, "  file[%u]: path=%s kind=%s representation=%s bytes=%llu "
                                    "size_known=%s\n",
                                representation_index, file->path,
                                yvex_remote_file_kind_name(file->kind), file->representation,
                                file->size_bytes, file->size_known ? "true" : "false");
        }
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

static int remote_catalog_render_json(FILE *fp, const yvex_remote_catalog *catalog)
{
    unsigned long long model_index;

    yvex_cli_out_fputs("{\"schema\":\"yvex.remote-model-catalog.v1\",\"models\":[", fp);
    for (model_index = 0u; model_index < yvex_remote_catalog_count(catalog); ++model_index) {
        const yvex_remote_model *model = yvex_remote_catalog_at(catalog, model_index);
        unsigned int representation_index;
        if (model_index) yvex_cli_out_fputs(",", fp);
        yvex_cli_out_fputs("{\"provider\":", fp);
        yvex_file_json_write_string(fp, model->provider);
        yvex_cli_out_fputs(",\"repository\":", fp);
        yvex_file_json_write_string(fp, model->repository);
        yvex_cli_out_fputs(",\"resolved_revision\":", fp);
        yvex_file_json_write_string(fp, model->resolved_revision);
        yvex_cli_out_fputs(",\"family\":", fp);
        yvex_file_json_write_string(fp, model->family);
        yvex_cli_out_fputs(",\"architecture\":", fp);
        yvex_file_json_write_string(fp, model->architecture);
        yvex_cli_out_fputs(",\"base_model\":", fp);
        yvex_file_json_write_string(fp, model->base_model);
        yvex_cli_out_fputs(",\"support_stage\":", fp);
        yvex_file_json_write_string(fp, yvex_model_support_stage_name(model->support_stage));
        yvex_cli_out_writef(fp, ",\"local\":%s,\"gated_known\":%s,\"gated\":%s,"
                                "\"parameter_count_known\":%s,\"parameter_count\":%llu,"
                                "\"representations\":[",
                            model->local ? "true" : "false",
                            model->gated_known ? "true" : "false",
                            model->gated ? "true" : "false",
                            model->parameter_count_known ? "true" : "false",
                            model->parameter_count);
        for (representation_index = 0u; representation_index < model->representation_count;
             ++representation_index) {
            const yvex_model_representation *representation =
                yvex_remote_catalog_representation_at(catalog, model_index,
                                                       representation_index);
            if (representation_index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{\"identity\":", fp);
            yvex_file_json_write_string(fp, representation->identity);
            yvex_cli_out_fputs(",\"format\":", fp);
            yvex_file_json_write_string(fp, representation->format);
            yvex_cli_out_fputs(",\"precision\":", fp);
            yvex_file_json_write_string(fp, representation->precision);
            yvex_cli_out_fputs(",\"file_pattern\":", fp);
            yvex_file_json_write_string(fp, representation->file_pattern);
            yvex_cli_out_fputs(",\"compatibility\":", fp);
            yvex_file_json_write_string(fp, representation->compatibility);
            yvex_cli_out_writef(fp, ",\"file_count\":%llu,\"size_bytes\":%llu,"
                                    "\"size_known\":%s,\"provisional\":%s,\"local\":%s}",
                                representation->file_count, representation->size_bytes,
                                representation->size_known ? "true" : "false",
                                representation->provisional ? "true" : "false",
                                representation->local ? "true" : "false");
        }
        yvex_cli_out_fputs("],\"files\":[", fp);
        for (representation_index = 0u; representation_index < model->available_file_count;
             ++representation_index) {
            const yvex_remote_file *file =
                yvex_remote_catalog_file_at(catalog, model_index, representation_index);
            if (representation_index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{\"path\":", fp);
            yvex_file_json_write_string(fp, file->path);
            yvex_cli_out_fputs(",\"kind\":", fp);
            yvex_file_json_write_string(fp, yvex_remote_file_kind_name(file->kind));
            yvex_cli_out_fputs(",\"representation\":", fp);
            yvex_file_json_write_string(fp, file->representation);
            yvex_cli_out_writef(fp, ",\"size_bytes\":%llu,\"size_known\":%s}",
                                file->size_bytes, file->size_known ? "true" : "false");
        }
        yvex_cli_out_fputs("]}", fp);
    }
    yvex_cli_out_fputs("]}\n", fp);
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

int yvex_remote_catalog_render(FILE *fp,
                               const yvex_remote_catalog *catalog,
                               yvex_model_catalog_output_mode mode,
                               int representations)
{
    if (!fp || !catalog) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON)
        return remote_catalog_render_json(fp, catalog);
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT)
        return remote_catalog_render_audit(fp, catalog);
    return remote_catalog_render_table(fp, catalog, representations);
}

int yvex_local_catalog_render(FILE *fp,
                              const yvex_local_model_catalog *catalog,
                              yvex_cli_engine_state_resolver engine_state,
                              const void *engine_context,
                              int engine_host_observed,
                              yvex_model_catalog_output_mode mode)
{
    unsigned long long index;

    if (!fp || !catalog) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_JSON) {
        yvex_cli_out_writef(fp,
                            "{\"schema\":\"yvex.local-model-catalog.v1\","
                            "\"engine_host_observed\":%s,\"models\":[",
                            engine_host_observed ? "true" : "false");
        for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
            const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
            const char *state = engine_state ? engine_state(model, engine_context)
                                             : model->engine_state;
            if (index) yvex_cli_out_fputs(",", fp);
            yvex_cli_out_fputs("{\"name\":", fp);
            yvex_file_json_write_string(fp, model->name);
            yvex_cli_out_fputs(",\"family\":", fp);
            yvex_file_json_write_string(fp, model->family);
            yvex_cli_out_fputs(",\"kind\":", fp);
            yvex_file_json_write_string(
                fp, model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package" : "acquired-source");
            yvex_cli_out_fputs(",\"representation\":", fp);
            yvex_file_json_write_string(fp, model->representation);
            yvex_cli_out_fputs(",\"package_state\":", fp);
            yvex_file_json_write_string(fp, model->package_state);
            yvex_cli_out_fputs(",\"verification_state\":", fp);
            yvex_file_json_write_string(fp, model->verification_state);
            yvex_cli_out_fputs(",\"engine_state\":", fp);
            yvex_file_json_write_string(fp, state);
            yvex_cli_out_fputs(",\"blocker\":", fp);
            yvex_file_json_write_string(fp, model->blocker);
            yvex_cli_out_writef(fp, ",\"size_bytes\":%llu,\"size_known\":%s,"
                                    "\"package_ready\":%s}",
                                model->size_bytes, model->size_known ? "true" : "false",
                                model->package_ready ? "true" : "false");
        }
        yvex_cli_out_fputs("]}\n", fp);
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    if (mode == YVEX_MODEL_CATALOG_OUTPUT_AUDIT) {
        yvex_cli_out_writef(fp, "local_models: %llu\n",
                            yvex_local_model_catalog_count(catalog));
        yvex_cli_out_writef(fp, "engine_host_observed: %s\n",
                            engine_host_observed ? "true" : "false");
        for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
            const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
            const char *state = engine_state ? engine_state(model, engine_context)
                                             : model->engine_state;
            yvex_cli_out_writef(fp, "model[%llu]: name=%s family=%s kind=%s representation=%s "
                                    "package_state=%s verification=%s engine=%s backend=%s "
                                    "bytes=%llu blocker=%s path=%s\n",
                                index, model->name, model->family,
                                model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package"
                                                                        : "acquired-source",
                                model->representation, model->package_state,
                                model->verification_state, state,
                                model->backend[0] ? model->backend : "unknown", model->size_bytes,
                                model->blocker[0] ? model->blocker : "none", model->path);
        }
        return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
    }
    yvex_cli_out_writef(fp, "LOCAL MODELS  count=%llu\n\n",
                        yvex_local_model_catalog_count(catalog));
    yvex_cli_out_writef(fp,
                        "%-40s %-12s %-8s %-16s %-18s %-18s %10s %-12s %s\n",
                        "MODEL", "FAMILY", "KIND", "REPRESENTATION", "PACKAGE STATE",
                        "VERIFICATION", "SIZE", "ENGINE", "BACKEND / BLOCKER");
    for (index = 0u; index < yvex_local_model_catalog_count(catalog); ++index) {
        const yvex_local_model *model = yvex_local_model_catalog_at(catalog, index);
        const char *state = engine_state ? engine_state(model, engine_context)
                                         : model->engine_state;
        char size[32];
        if (model->size_known)
            model_download_format_bytes(size, sizeof(size), model->size_bytes);
        else
            snprintf(size, sizeof(size), "unknown");
        yvex_cli_out_writef(fp, "%-40.40s %-12.12s %-8.8s %-16.16s %-18.18s %-18.18s "
                                "%10.10s %-12.12s %s%s%s\n",
                            model->name, model->family,
                            model->kind == YVEX_LOCAL_MODEL_PACKAGE ? "package" : "source",
                            model->representation,
                            model->package_state, model->verification_state, size,
                            state,
                            model->backend[0] ? model->backend : "-",
                            model->blocker[0] ? " · " : "", model->blocker);
    }
    return ferror(fp) ? YVEX_ERR_IO : YVEX_OK;
}

#define COMPONENT_FIELD(KEY, KIND, MEMBER) \
    {KEY, KIND, offsetof(yvex_cli_component_result, MEMBER), ""}
#define COMPONENT_EXECUTION_FIELD(KEY, KIND, MEMBER)                         \
    {KEY, KIND, offsetof(yvex_cli_component_result, execution) +             \
                    offsetof(yvex_component_execution_result, MEMBER), ""}

int yvex_component_render(FILE *fp, yvex_graph_report_mode mode,
                          const yvex_cli_component_result *result)
{
    static const yvex_cli_field_spec fields[] = {
        COMPONENT_FIELD("status", YVEX_CLI_FIELD_TEXT, status),
        COMPONENT_FIELD("target", YVEX_CLI_FIELD_TEXT, target),
        COMPONENT_FIELD("component", YVEX_CLI_FIELD_TEXT, component),
        COMPONENT_FIELD("backend", YVEX_CLI_FIELD_TEXT, backend),
        COMPONENT_EXECUTION_FIELD("artifact_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                                  artifact_identity),
        COMPONENT_EXECUTION_FIELD("execution_identity", YVEX_CLI_FIELD_TEXT_ARRAY,
                                  execution_identity),
        COMPONENT_EXECUTION_FIELD("batch", YVEX_CLI_FIELD_U64, batch),
        COMPONENT_EXECUTION_FIELD("output_values", YVEX_CLI_FIELD_U64,
                                  output_values),
        COMPONENT_EXECUTION_FIELD("output_rank", YVEX_CLI_FIELD_U32,
                                  output_rank),
        COMPONENT_EXECUTION_FIELD("tensor_reads", YVEX_CLI_FIELD_U64,
                                  tensor_reads),
        COMPONENT_EXECUTION_FIELD("payload_bytes_read", YVEX_CLI_FIELD_U64,
                                  payload_bytes_read),
        COMPONENT_EXECUTION_FIELD("peak_workspace_bytes", YVEX_CLI_FIELD_U64,
                                  peak_workspace_bytes),
        COMPONENT_FIELD("output_path", YVEX_CLI_FIELD_TEXT, output_path),
        COMPONENT_FIELD("published_bytes", YVEX_CLI_FIELD_U64, published_bytes),
        COMPONENT_FIELD("published", YVEX_CLI_FIELD_BOOL, published),
    };

    if (!fp || !result) return YVEX_ERR_INVALID_ARG;
    if (mode == YVEX_GRAPH_REPORT_MODE_NORMAL)
        return yvex_cli_out_writef(
                   fp, "Component execution: %s target=%s component=%s values=%llu "
                       "output=%s identity=%s\n",
                   result->status, result->target, result->component,
                   result->execution.output_values, result->output_path,
                   result->execution.execution_identity) < 0
                   ? YVEX_ERR_IO : YVEX_OK;
    if (mode == YVEX_GRAPH_REPORT_MODE_JSON) {
        yvex_cli_json_begin(fp);
        if (yvex_cli_json_fields(fp, result, fields,
                                 sizeof(fields) / sizeof(fields[0]), 0) != 0)
            return YVEX_ERR_IO;
        yvex_cli_json_end(fp);
        return YVEX_OK;
    }
    return yvex_cli_out_fields(fp, result, fields, sizeof(fields) / sizeof(fields[0])) == 0
               ? YVEX_OK : YVEX_ERR_IO;
}

#undef COMPONENT_EXECUTION_FIELD
#undef COMPONENT_FIELD
