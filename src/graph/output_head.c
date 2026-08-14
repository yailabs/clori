/* Compile the immutable output-head projection independently of runtime workspaces. */
#include <yvex/internal/compiler.h>

#include <stdint.h>
#include <string.h>

#include <yvex/internal/core.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/transformer.h>

static int output_head_refuse(yvex_error *err, yvex_status status,
                              const char *reason)
{
    yvex_error_set(err, status, "graph.output-head", reason);
    return status;
}

static int output_head_identity(yvex_runtime_logits_plan_summary *summary)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!summary) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.plan.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, summary->family_adapter_version) ||
        !yvex_sha256_update_text(&hash, summary->artifact_identity) ||
        !yvex_sha256_update_text(&hash, summary->materialization_identity) ||
        !yvex_sha256_update_text(&hash, summary->logical_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_numeric_identity) ||
        !yvex_sha256_update_text(&hash, summary->runtime_descriptor_identity) ||
        !yvex_sha256_update_text(&hash, summary->transformer_plan_identity) ||
        !yvex_sha256_update_u64(&hash, summary->output_head_tensor_id) ||
        !yvex_sha256_update_u64(&hash, summary->role) ||
        !yvex_sha256_update_u64(&hash, summary->qtype) ||
        !yvex_sha256_update_u64(&hash, summary->row_width) ||
        !yvex_sha256_update_u64(&hash, summary->row_count) ||
        !yvex_sha256_update_u64(&hash, summary->row_bytes) ||
        !yvex_sha256_update_u64(&hash, summary->encoded_bytes) ||
        !yvex_sha256_update_u64(&hash, summary->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, summary->hidden_width) ||
        !yvex_sha256_update_u64(
            &hash, (unsigned int)summary->separate_output_head) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)summary->output_head_bias) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, summary->output_head_plan_identity);
    return 1;
}

int yvex_output_head_plan_validate(
    const yvex_runtime_logits_plan_summary *summary, yvex_error *err)
{
    yvex_runtime_logits_plan_summary canonical;
    char expected[YVEX_SHA256_HEX_CAP];
    if (!summary || summary->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        !summary->family_adapter_id || !summary->family_adapter_version ||
        summary->role != YVEX_TENSOR_ROLE_OUTPUT_HEAD || !summary->row_width ||
        !summary->row_count || !summary->row_bytes || !summary->encoded_bytes ||
        summary->row_count != summary->vocabulary_size ||
        summary->row_width != summary->hidden_width ||
        !summary->separate_output_head || summary->output_head_bias ||
        !yvex_sha256_hex_valid(summary->output_head_plan_identity))
        return output_head_refuse(err, YVEX_ERR_FORMAT,
                                  "compiled output-head plan is malformed");
    canonical = *summary;
    memcpy(expected, summary->output_head_plan_identity, sizeof(expected));
    canonical.output_head_plan_identity[0] = '\0';
    if (!output_head_identity(&canonical) ||
        strcmp(canonical.output_head_plan_identity, expected) != 0)
        return output_head_refuse(err, YVEX_ERR_STATE,
                                  "compiled output-head plan identity is stale");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_output_head_plan_build(
    yvex_runtime_logits_plan_summary *out,
    unsigned long long family_adapter_id,
    unsigned long long family_adapter_version,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const yvex_transformer_plan *transformer,
    const yvex_logits_family_policy *policy, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime =
        yvex_runtime_descriptor_summary_get(descriptor);
    const yvex_materialization_summary *material =
        yvex_materialization_session_summary(materialization);
    const yvex_transformer_plan_summary *transformer_summary =
        yvex_transformer_plan_summary_get(transformer);
    const yvex_runtime_tensor_binding *row = yvex_runtime_descriptor_find_role(
        descriptor, YVEX_TENSOR_ROLE_OUTPUT_HEAD, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX);
    const yvex_runtime_tensor_binding *embedding =
        yvex_runtime_descriptor_find_role(
            descriptor, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
            YVEX_TENSOR_SCOPE_GLOBAL, YVEX_MATERIALIZATION_NO_INDEX,
            YVEX_MATERIALIZATION_NO_INDEX);
    const yvex_materialized_tensor_binding *binding = row
        ? yvex_materialization_session_tensor_at(materialization, row->tensor_id)
        : NULL;
    const yvex_gguf_qtype_geometry *geometry = binding
        ? yvex_gguf_qtype_geometry_find(binding->qtype) : NULL;
    const yvex_quant_numeric_capability *numeric = binding
        ? yvex_quant_numeric_capability_at(binding->qtype) : NULL;
    unsigned long long blocks, row_bytes, encoded_bytes;
    if (out) memset(out, 0, sizeof(*out));
    if (!out || !runtime || !material || !transformer_summary || !policy ||
        !family_adapter_id || !family_adapter_version ||
        policy->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        !policy->separate_output_head || policy->tied_output_head ||
        policy->output_head_bias)
        return output_head_refuse(
            err, YVEX_ERR_FORMAT,
            "family output-head policy is unavailable or incompatible");
    if (!row || !binding || !embedding ||
        binding->tensor_id == embedding->tensor_id ||
        binding->role != YVEX_TENSOR_ROLE_OUTPUT_HEAD ||
        binding->row_width != transformer_summary->hidden_width ||
        binding->row_count != transformer_summary->vocabulary_size ||
        !geometry || !geometry->block_size || !geometry->bytes_per_block ||
        binding->row_width % geometry->block_size || !numeric ||
        !numeric->dedicated_cpu_compute_available ||
        !numeric->dedicated_cuda_compute_available)
        return output_head_refuse(
            err, YVEX_ERR_FORMAT,
            "exact separate output-head binding or qtype compute is unavailable");
    blocks = binding->row_width / geometry->block_size;
    if (!yvex_core_u64_mul(blocks, geometry->bytes_per_block, &row_bytes) ||
        !yvex_core_u64_mul(row_bytes, binding->row_count, &encoded_bytes) ||
        encoded_bytes != binding->encoded_bytes)
        return output_head_refuse(
            err, YVEX_ERR_FORMAT,
            "output-head encoded geometry is inconsistent");
    out->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    out->family_adapter_id = family_adapter_id;
    out->family_adapter_version = family_adapter_version;
    out->output_head_tensor_id = binding->tensor_id;
    out->role = binding->role;
    out->qtype = binding->qtype;
    out->row_width = binding->row_width;
    out->row_count = binding->row_count;
    out->row_bytes = row_bytes;
    out->encoded_bytes = encoded_bytes;
    out->vocabulary_size = transformer_summary->vocabulary_size;
    out->hidden_width = transformer_summary->hidden_width;
    out->separate_output_head = policy->separate_output_head;
    out->output_head_bias = policy->output_head_bias;
    yvex_core_text_copy(out->artifact_identity, sizeof(out->artifact_identity),
                        material->artifact_identity);
    yvex_core_text_copy(out->materialization_identity,
                        sizeof(out->materialization_identity),
                        material->plan_identity);
    yvex_core_text_copy(out->logical_model_identity,
                        sizeof(out->logical_model_identity),
                        runtime->logical_model_identity);
    yvex_core_text_copy(out->runtime_numeric_identity,
                        sizeof(out->runtime_numeric_identity),
                        runtime->runtime_numeric_identity);
    yvex_core_text_copy(out->runtime_descriptor_identity,
                        sizeof(out->runtime_descriptor_identity),
                        runtime->runtime_descriptor_identity);
    yvex_core_text_copy(out->transformer_plan_identity,
                        sizeof(out->transformer_plan_identity),
                        transformer_summary->transformer_plan_identity);
    if (!output_head_identity(out))
        return output_head_refuse(err, YVEX_ERR_STATE,
                                  "output-head plan identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}
