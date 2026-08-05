/* Speculative candidates remain untrusted until target comparison; this owner returns only a
 * committable prefix and never publishes model, RNG, tokenizer, transcript, or session state. */
#include "src/runtime/private.h"
#include <yvex/internal/decode.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>
typedef struct {
    const yvex_materialized_tensor_binding *binding;
    const unsigned char *encoded;
    unsigned long long encoded_bytes, row_bytes;
} speculation_weight;
struct yvex_runtime_speculation_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    yvex_runtime_transformer_context *target_transformer, *draft_transformer;
    yvex_runtime_logits_context *target_logits, *verification_logits;
    yvex_runtime_sampling_context *target_sampling, *draft_sampling, *verification_sampling;
    yvex_runtime_sampling_transaction *target_rng, *draft_rng;
    yvex_runtime_sampling_policy sampling_policy;
    yvex_runtime_speculation_options options;
    yvex_speculation_family_policy policy;
    speculation_weight feature_projection, feature_norm, markov_embedding;
    speculation_weight markov_output, confidence;
    yvex_backend *device_backend;
    yvex_device_tensor *device_feature_input, *device_feature_projected;
    yvex_device_tensor *device_feature_normalized, *device_feature_norm;
    yvex_device_tensor *device_adjusted_logits;
    float *feature_projected, *feature_norm_weights, *target_features;
    float *draft_hidden, *draft_pre_normalized, *target_hidden;
    float *base_logits, *adjusted_logits, *markov_bias;
    float *draft_probabilities, *target_probabilities, *markov_embedding_values;
    unsigned int *draft_input_ids;
    unsigned long long vocabulary_size, hidden_width, workspace_bytes;
    unsigned long long pending_position, pending_committed_count;
    unsigned long long pending_verified_prefix_count;
    unsigned int pending_tokens[YVEX_SPECULATION_MAX_BLOCK + 2u], target_token_ids[YVEX_SPECULATION_MAX_BLOCK + 1u];
    yvex_runtime_transformer_result pending_verification_target;
    char pending_source_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char pending_sampling_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char pending_cycle_identity[YVEX_SPECULATION_IDENTITY_CAP];
    char projected_feature_identity[YVEX_SPECULATION_IDENTITY_CAP];
    const yvex_runtime_commit_participant *publication;
    int device_draft_selection, publication_prepared, cycle_pending, verification_staged;
};
static void speculation_pending_clear(yvex_runtime_speculation_context *context);
static int speculation_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.speculation", reason);
    return status;
}
static int speculation_hash_finish(yvex_sha256 *hash, char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static int speculation_markov_row_identity(const yvex_runtime_speculation_context *context,
    unsigned int token,
    char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    const yvex_materialized_tensor_binding *binding = context
        ? context->markov_embedding.binding : NULL;
    if (!binding || !context->model_view || !context->model_view->binding || token >= binding->row_count)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.speculation.markov-row.v1") ||
        !yvex_sha256_update_text(&hash, context->model_view->binding->identity) ||
        !yvex_sha256_update_u64(&hash, binding->tensor_id) ||
        !yvex_sha256_update_u64(&hash, binding->qtype) || !yvex_sha256_update_u64(&hash, token))
        return 0;
    return speculation_hash_finish(&hash, output);
}
static int speculation_cuda_physical_add(yvex_execution_physical_facts *physical,
    const yvex_backend_cuda_operation_facts *facts, yvex_error *err)
{
    yvex_execution_memory_facts memory = {0};
    if (yvex_execution_memory_facts_add(
            &memory, facts->active_weight_bytes, facts->state_bytes, facts->activation_bytes,
            facts->temporary_bytes, facts->compulsory_memory_facts_available,
            !facts->compulsory_memory_facts_available, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_execution_physical_facts_add(
        physical, &memory, facts->h2d_bytes, facts->d2h_bytes, facts->d2d_bytes,
        facts->kernel_launches, facts->device_synchronizations, err);
}
static int speculation_draft_sampling_policy(const yvex_runtime_sampling_policy *target_policy,
    unsigned long long vocabulary_size, yvex_runtime_sampling_policy *draft_policy,
    yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    unsigned long long seed = 0ull;
    if (draft_policy) memset(draft_policy, 0, sizeof(*draft_policy));
    if (!target_policy || !draft_policy || !vocabulary_size ||
        !yvex_sha256_hex_valid(target_policy->policy_identity))
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "draft sampling requires one sealed target policy");
    *draft_policy = *target_policy;
    if (target_policy->strategy == YVEX_SAMPLING_STRATEGY_STOCHASTIC) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(&hash, "yvex.speculation.draft-rng-domain.v1") ||
            !yvex_sha256_update_text(&hash, target_policy->policy_identity) ||
            !yvex_sha256_final(&hash, digest))
            return speculation_refuse(err, YVEX_ERR_STATE,
                                      "draft sampling seed derivation failed");
        for (index = 0u; index < sizeof(seed); ++index)
            seed = (seed << 8u) | digest[index];
        draft_policy->seed = seed;
    }
    draft_policy->policy_identity[0] = '\0';
    return yvex_runtime_sampling_policy_seal(draft_policy, vocabulary_size, err);
}
static int speculation_weight_bind(yvex_runtime_speculation_context *context, yvex_tensor_role role,
                                   speculation_weight *out,
                                   yvex_error *err)
{
    const yvex_runtime_descriptor_summary *summary =
        yvex_runtime_descriptor_summary_get(context->model_view->descriptor);
    const yvex_runtime_tensor_binding *match = NULL;
    unsigned long long index, bytes = 0ull;
    if (!summary || !out)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculation weight owner is unavailable");
    for (index = 0ull; index < summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *row = yvex_runtime_descriptor_tensor_at(
            context->model_view->descriptor, index);
        if (row && row->role == role && row->scope == YVEX_TENSOR_SCOPE_DRAFT) {
            if (match)
                return speculation_refuse(
                    err, YVEX_ERR_FORMAT, "speculation role has duplicate bindings");
            match = row;
        }
    }
    if (!match || !match->binding || !match->binding->row_count ||
        match->binding->encoded_bytes % match->binding->row_count ||
        yvex_runtime_residency_binding_view(
            context->model_view->residency, match->binding, &out->encoded,
            &bytes, err) != YVEX_OK || bytes != match->binding->encoded_bytes)
        return speculation_refuse(
            err, YVEX_ERR_STATE, "speculation role is not resident and exact");
    out->binding = match->binding;
    out->encoded_bytes = bytes;
    out->row_bytes = bytes / match->binding->row_count;
    return YVEX_OK;
}
static int speculation_weight_decode_row(const speculation_weight *weight,
    unsigned long long row, float *values, unsigned long long capacity, yvex_error *err)
{
    const yvex_gguf_qtype_geometry *geometry;
    yvex_quant_failure failure;
    unsigned long long blocks, block;
    if (!weight || !weight->binding || !values ||
        row >= weight->binding->row_count ||
        capacity < weight->binding->row_width)
        return speculation_refuse(
            err, YVEX_ERR_BOUNDS, "speculation decoded row capacity is invalid");
    geometry = yvex_gguf_qtype_geometry_find(weight->binding->qtype);
    if (!geometry || !geometry->block_size ||
        weight->binding->row_width % geometry->block_size ||
        (blocks = weight->binding->row_width / geometry->block_size) == 0ull ||
        blocks * geometry->bytes_per_block != weight->row_bytes)
        return speculation_refuse(err, YVEX_ERR_FORMAT, "speculation weight row is not block exact");
    for (block = 0ull; block < blocks; ++block) {
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_decode_block(
                weight->binding->qtype, weight->encoded + row * weight->row_bytes +
                    block * geometry->bytes_per_block,
                geometry->bytes_per_block,
                values + block * geometry->block_size,
                geometry->block_size, &failure, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    return YVEX_OK;
}
static int speculation_context_geometry(yvex_runtime_speculation_context *context, yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *logits =
        yvex_runtime_logits_plan_summary_get(context->target_logits);
    if (!logits || logits->vocabulary_size != context->vocabulary_size ||
        logits->hidden_width != context->hidden_width ||
        context->feature_projection.binding->row_width !=
            context->policy.concatenated_feature_width ||
        context->feature_projection.binding->row_count != context->hidden_width ||
        context->feature_norm.binding->row_width != context->hidden_width ||
        context->feature_norm.binding->row_count != 1ull ||
        context->markov_embedding.binding->row_count != context->vocabulary_size ||
        context->markov_embedding.binding->row_width != context->policy.markov_rank ||
        context->markov_output.binding->row_count != context->vocabulary_size ||
        context->markov_output.binding->row_width != context->policy.markov_rank ||
        context->confidence.binding->row_count != 1ull ||
        context->confidence.binding->row_width !=
            context->hidden_width + context->policy.markov_rank)
        return speculation_refuse(
            err, YVEX_ERR_FORMAT, "DSpark runtime weight geometry is inconsistent");
    return YVEX_OK;
}
static int speculation_context_buffers(yvex_runtime_speculation_context *context, yvex_error *err)
{
    unsigned long long block = context->policy.block_size;
    unsigned long long hidden_rows, target_hidden_rows, feature_rows;
    unsigned long long draft_rows, logits_rows, target_probability_rows, probabilities;
    unsigned long long markov_values, total = 0ull;
#define SPEC_ADD(count_) \
    do { \
        unsigned long long bytes_; \
        if (!yvex_core_u64_mul((count_), sizeof(float), &bytes_) || \
            !yvex_core_u64_add(total, bytes_, &total)) goto overflow; \
    } while (0)
    if (!yvex_core_u64_mul(block, context->hidden_width, &hidden_rows) ||
        !yvex_core_u64_mul(block + 2ull, context->hidden_width, &target_hidden_rows) ||
        !yvex_core_u64_mul(block + 2ull,
                           context->policy.concatenated_feature_width,
                           &feature_rows) ||
        !yvex_core_u64_mul(block, context->vocabulary_size, &draft_rows) ||
        !yvex_core_u64_mul(block + 1ull, context->vocabulary_size, &logits_rows) ||
        !yvex_core_u64_mul(context->verification_logits ? 0ull : block + 1ull,
                           context->vocabulary_size, &target_probability_rows) ||
        !yvex_core_u64_add(draft_rows, target_probability_rows, &probabilities) ||
        !yvex_core_u64_mul(block, context->policy.markov_rank, &markov_values))
        goto overflow;
    SPEC_ADD(target_hidden_rows);
    SPEC_ADD(context->hidden_width);
    SPEC_ADD(feature_rows);
    SPEC_ADD(hidden_rows);
    SPEC_ADD(hidden_rows);
    SPEC_ADD(target_hidden_rows);
    SPEC_ADD(logits_rows);
    SPEC_ADD(context->vocabulary_size);
    SPEC_ADD(context->vocabulary_size);
    SPEC_ADD(probabilities);
    SPEC_ADD(markov_values);
    if (!yvex_core_u64_add(total, block * sizeof(unsigned int), &total) ||
        total > SIZE_MAX ||
        (context->options.maximum_host_bytes &&
         total > context->options.maximum_host_bytes))
        goto overflow;
    context->feature_projected = yvex_core_calloc((size_t)target_hidden_rows, sizeof(float));
    context->feature_norm_weights = yvex_core_calloc(
        (size_t)context->hidden_width, sizeof(float));
    context->target_features = yvex_core_calloc((size_t)feature_rows, sizeof(float));
    context->draft_hidden = yvex_core_calloc((size_t)hidden_rows, sizeof(float));
    context->draft_pre_normalized = yvex_core_calloc((size_t)hidden_rows, sizeof(float));
    context->target_hidden = yvex_core_calloc((size_t)target_hidden_rows, sizeof(float));
    /* Disjoint draft/verify phases share one block-plus-one arena without row-localizing width-N. */
    context->base_logits = yvex_core_calloc((size_t)logits_rows, sizeof(float));
    context->adjusted_logits = yvex_core_calloc(
        (size_t)context->vocabulary_size, sizeof(float));
    context->markov_bias = yvex_core_calloc((size_t)context->vocabulary_size, sizeof(float));
    context->draft_probabilities = yvex_core_calloc((size_t)probabilities, sizeof(float));
    context->target_probabilities = target_probability_rows ? context->draft_probabilities + draft_rows : NULL;
    context->markov_embedding_values = yvex_core_calloc((size_t)markov_values, sizeof(float));
    context->draft_input_ids = yvex_core_calloc((size_t)block, sizeof(unsigned int));
    if (!context->feature_projected || !context->feature_norm_weights ||
        !context->target_features ||
        !context->draft_hidden || !context->draft_pre_normalized ||
        !context->target_hidden || !context->base_logits ||
        !context->adjusted_logits || !context->markov_bias ||
        !context->draft_probabilities || !context->markov_embedding_values ||
        !context->draft_input_ids)
        return speculation_refuse(
            err, YVEX_ERR_NOMEM, "DSpark runtime workspace allocation failed");
    context->workspace_bytes = total;
    return speculation_weight_decode_row(&context->feature_norm, 0ull,
        context->feature_norm_weights, context->hidden_width, err);
overflow:
    return speculation_refuse(err, YVEX_ERR_BOUNDS, "DSpark runtime workspace extent overflowed");
#undef SPEC_ADD
}
static int speculation_device_tensor_open(yvex_backend *backend, yvex_device_tensor **out,
    const char *name, unsigned long long elements, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long bytes;
    if (!backend || !out || !name || !yvex_core_u64_mul(elements, sizeof(float), &bytes))
        return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                  "DSpark device feature workspace extent overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = elements;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(backend, &descriptor, out, err);
}
static int speculation_context_device_buffers(yvex_runtime_speculation_context *context,
    yvex_error *err)
{
    const yvex_runtime_session_view *view;
    unsigned long long rows, input_elements, output_elements;
    unsigned long long total_elements, total_bytes;
    int rc;
    if (context->options.backend != YVEX_BACKEND_KIND_CUDA ||
        context->options.execution_profile->evidence != YVEX_EXECUTION_EVIDENCE_PRODUCTION)
        return YVEX_OK;
    view = yvex_runtime_session_view_get(context->session);
    rows = context->policy.block_size + 2ull;
    if (!view || !view->backend)
        return speculation_refuse(
            err, YVEX_ERR_STATE, "DSpark CUDA feature workspace backend is unavailable");
    if (!yvex_core_u64_mul(rows, context->policy.concatenated_feature_width,
                           &input_elements) ||
        !yvex_core_u64_mul(rows, context->hidden_width, &output_elements) ||
        !yvex_core_u64_mul(output_elements, 2ull, &total_elements) ||
        !yvex_core_u64_add(total_elements, input_elements, &total_elements) ||
        !yvex_core_u64_add(total_elements, context->hidden_width, &total_elements) ||
        !yvex_core_u64_add(total_elements, context->device_draft_selection
                                               ? context->vocabulary_size : 0ull,
                           &total_elements) ||
        !yvex_core_u64_mul(total_elements, sizeof(float), &total_bytes))
        return speculation_refuse(
            err, YVEX_ERR_BOUNDS, "DSpark CUDA feature workspace is not admitted");
    if (yvex_backend_memory_can_add(view->backend, total_bytes, "CUDA",
            "runtime.speculation.device-workspace", err) != YVEX_OK)
        return yvex_error_code(err);
    context->device_backend = view->backend;
    rc = speculation_device_tensor_open(view->backend, &context->device_feature_input,
                                        "dspark_feature_input", input_elements, err);
    if (rc == YVEX_OK)
        rc = speculation_device_tensor_open(view->backend, &context->device_feature_projected,
                                            "dspark_feature_projected", output_elements, err);
    if (rc == YVEX_OK)
        rc = speculation_device_tensor_open(view->backend, &context->device_feature_normalized,
                                            "dspark_feature_normalized", output_elements, err);
    if (rc == YVEX_OK)
        rc = speculation_device_tensor_open(view->backend, &context->device_feature_norm,
                                            "dspark_feature_norm", context->hidden_width, err);
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = speculation_device_tensor_open(
            view->backend, &context->device_adjusted_logits, "dspark_adjusted_logits",
            context->vocabulary_size, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_tensor_write(view->backend, context->device_feature_norm,
            context->feature_norm_weights, context->device_feature_norm->bytes, err);
    return rc;
}
int yvex_runtime_speculation_context_open(
    yvex_runtime_speculation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session, yvex_runtime_transformer_context *target_transformer,
    yvex_runtime_logits_context *target_logits, yvex_runtime_sampling_context *target_sampling,
    const yvex_runtime_sampling_policy *sampling_policy,
    const yvex_runtime_speculation_options *options, yvex_error *err)
{
    yvex_runtime_speculation_context *context = NULL;
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_logits_options logits_options = {0};
    yvex_runtime_sampling_options sampling_options = {0};
    yvex_runtime_sampling_policy draft_policy = {0};
    const yvex_transformer_plan_summary *target_plan;
    const yvex_runtime_descriptor_summary *descriptor;
    unsigned long long draft_context_capacity;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !target_transformer || !target_logits ||
        !target_sampling || !sampling_policy || !options ||
        !options->execution_profile || !options->shape_registry ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) || !options->context_capacity)
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG, "complete DSpark runtime owners are required");
    context = yvex_core_calloc(1u, sizeof(*context));
    if (!context)
        return speculation_refuse(err, YVEX_ERR_NOMEM, "DSpark context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->target_transformer = target_transformer;
    context->target_logits = target_logits;
    context->target_sampling = target_sampling;
    context->sampling_policy = *sampling_policy;
    context->options = *options;
    descriptor = context->model_view ? yvex_runtime_descriptor_summary_get(
        context->model_view->descriptor) : NULL;
    target_plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(target_transformer));
    if (!context->model_view || !descriptor || !target_plan ||
        yvex_runtime_transformer_context_session(target_transformer) != session ||
        !context->model_view->adapter || !context->model_view->adapter->speculation_policy ||
        !context->model_view->adapter->speculation_policy(descriptor, &context->policy) ||
        context->policy.schema_version != YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 ||
        !context->policy.target_verification_required || !context->policy.parallel_block_backbone ||
        !context->policy.sequential_markov || !context->policy.shares_output_head ||
        !context->model_view->draft_attention ||
        descriptor->draft_layer_count != context->policy.draft_layer_count ||
        target_plan->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER ||
        context->policy.block_size < 2ull ||
        context->policy.block_size > YVEX_SPECULATION_MAX_BLOCK) {
        rc = speculation_refuse(err, YVEX_ERR_UNSUPPORTED,
                                "runtime binding does not admit complete DSpark execution");
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    if (!target_plan->maximum_context || options->context_capacity > target_plan->maximum_context ||
        !yvex_core_u64_add(options->context_capacity, context->policy.block_size + 2ull,
                           &draft_context_capacity) ||
        draft_context_capacity > target_plan->maximum_context) {
        rc = speculation_refuse(err, YVEX_ERR_UNSUPPORTED,
            "DSpark requires bounded draft lookahead beyond the admitted target context");
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    context->vocabulary_size = target_plan->vocabulary_size;
    context->hidden_width = target_plan->hidden_width;
    context->device_draft_selection = options->backend == YVEX_BACKEND_KIND_CUDA &&
        options->execution_profile->evidence == YVEX_EXECUTION_EVIDENCE_PRODUCTION &&
        sampling_policy->strategy == YVEX_SAMPLING_STRATEGY_GREEDY;
    transformer_options.maximum_host_bytes = options->maximum_host_bytes;
    transformer_options.maximum_device_bytes = options->maximum_device_bytes;
    /* Draft queries attend one another, so reserve ephemeral lookahead beyond target-visible
     * context; accepted target state remains bounded by the caller's capacity. */
    transformer_options.context_capacity = draft_context_capacity;
    transformer_options.workspace_token_capacity = context->policy.block_size + 2ull;
    transformer_options.tensor_scope = YVEX_TENSOR_SCOPE_DRAFT;
    transformer_options.cancel_requested = options->cancel_requested;
    transformer_options.cancel_context = options->cancel_context;
    transformer_options.evidence_level = runtime_attention_evidence(options->execution_profile->evidence);
    transformer_options.device_hidden_output = context->device_draft_selection;
    transformer_options.device_pre_normalized_output = context->device_draft_selection;
    transformer_options.execution_profile = options->execution_profile;
    transformer_options.shape_registry = options->shape_registry;
    rc = yvex_runtime_transformer_context_open(
        &context->draft_transformer, model, session, &transformer_options, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_admit_shared_draft_plan(
            target_logits, yvex_runtime_transformer_context_plan(context->draft_transformer), err);
    sampling_options.maximum_vocabulary_size = context->vocabulary_size;
    sampling_options.maximum_rows = context->policy.block_size;
    sampling_options.maximum_host_bytes = options->maximum_host_bytes;
    sampling_options.cancel_requested = options->cancel_requested;
    sampling_options.cancel_context = options->cancel_context;
    sampling_options.device_selection = context->device_draft_selection;
    if (rc == YVEX_OK)
        rc = speculation_draft_sampling_policy(
            sampling_policy, context->vocabulary_size, &draft_policy, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_open(
            &context->draft_sampling, yvex_runtime_logits_plan_summary_get(target_logits),
            &draft_policy, &sampling_options, err);
    if (rc == YVEX_OK && context->device_draft_selection) {
        logits_options.maximum_rows = context->policy.block_size + 1ull;
        logits_options.maximum_host_bytes = options->maximum_host_bytes;
        logits_options.maximum_device_bytes = options->maximum_device_bytes;
        logits_options.evidence_profile = options->execution_profile->evidence;
        logits_options.device_selection = 1;
        logits_options.execution_profile = options->execution_profile;
        logits_options.cancel_requested = options->cancel_requested;
        logits_options.cancel_context = options->cancel_context;
        rc = yvex_runtime_logits_context_open(&context->verification_logits, model, session,
            yvex_runtime_transformer_context_plan(target_transformer), &logits_options, err);
        sampling_options.maximum_rows = context->policy.block_size + 1ull;
        sampling_options.device_selection = 1;
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_context_open(&context->verification_sampling,
                yvex_runtime_logits_plan_summary_get(context->verification_logits), sampling_policy,
                &sampling_options, err);
    }
#define BIND(role_, member_) \
    if (rc == YVEX_OK) rc = speculation_weight_bind(context, (role_), \
                                                     &context->member_, err)
    BIND(context->policy.feature_projection_role, feature_projection);
    BIND(context->policy.feature_norm_role, feature_norm);
    BIND(context->policy.markov_embedding_role, markov_embedding);
    BIND(context->policy.markov_output_role, markov_output);
    BIND(context->policy.confidence_role, confidence);
#undef BIND
    if (rc == YVEX_OK) rc = speculation_context_geometry(context, err);
    if (rc == YVEX_OK) rc = speculation_context_buffers(context, err);
    if (rc == YVEX_OK) rc = speculation_context_device_buffers(context, err);
    if (rc != YVEX_OK) {
        (void)yvex_runtime_speculation_context_close(&context, NULL);
        return rc;
    }
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
}
const yvex_speculation_family_policy *yvex_runtime_speculation_policy_get(
    const yvex_runtime_speculation_context *context)
{
    return context ? &context->policy : NULL;
}
static int speculation_project_target_features(yvex_runtime_speculation_context *context,
    const float *features, unsigned long long token_count,
    yvex_runtime_transformer_result *result, yvex_error *err)
{
    unsigned long long token, row, output_elements;
    if (!context || !features || !token_count || !result ||
        token_count > context->policy.block_size + 2ull ||
        !yvex_core_u64_mul(token_count, context->hidden_width, &output_elements))
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG, "DSpark feature projection extent is invalid");
    if (context->device_feature_input) {
        const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(
            yvex_runtime_transformer_context_plan(context->draft_transformer));
        yvex_backend_cuda_operation_facts facts = {0};
        yvex_runtime_transformer_result candidate = *result;
        yvex_execution_memory_facts memory = result->memory;
        yvex_device_tensor input = *context->device_feature_input;
        yvex_device_tensor projected = *context->device_feature_projected;
        yvex_device_tensor normalized = *context->device_feature_normalized;
        unsigned long long input_elements;
        unsigned long long input_bytes, output_bytes, norm_activation;
        unsigned long long h2d, d2h, kernels, uploads, downloads, synchronizations;
        int resident_input = context->device_feature_input->is_written;
        int rc;
        if (!plan || !yvex_core_u64_mul(
                         token_count, context->policy.concatenated_feature_width,
                         &input_elements) ||
            !yvex_core_u64_mul(input_elements, sizeof(float), &input_bytes) ||
            !yvex_core_u64_mul(output_elements, sizeof(float), &output_bytes) ||
            !yvex_core_u64_mul(output_bytes, 2ull, &norm_activation))
            return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                      "DSpark CUDA feature projection extent overflowed");
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return speculation_refuse(err, YVEX_ERR_CANCELLED,
                                      "DSpark feature projection was cancelled");
        input.rank = projected.rank = normalized.rank = 2u;
        input.dims[0] = projected.dims[0] = normalized.dims[0] = token_count;
        input.dims[1] = context->policy.concatenated_feature_width;
        projected.dims[1] = normalized.dims[1] = context->hidden_width;
        input.bytes = input_bytes;
        projected.bytes = normalized.bytes = output_bytes;
        rc = resident_input ? YVEX_OK : yvex_backend_tensor_write(
            context->device_backend, &input, features, input_bytes, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_cuda_encoded_matvec(
                context->device_backend, context->feature_projection.encoded,
                context->feature_projection.encoded_bytes,
                context->feature_projection.binding->qtype,
                context->hidden_width,
                context->policy.concatenated_feature_width,
                context->feature_projection.row_bytes, token_count, &input,
                NULL, 0ull, NULL, &projected, &facts, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_op_rms_norm(context->device_backend, &projected,
                context->device_feature_norm,
                (float)plan->output_norm_epsilon, &normalized, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_tensor_read(context->device_backend, &normalized,
                context->feature_projected, output_bytes, err);
        if (rc != YVEX_OK) return rc;
        if (yvex_execution_memory_facts_add(
                &memory, facts.active_weight_bytes, facts.state_bytes,
                facts.activation_bytes, facts.temporary_bytes,
                facts.compulsory_memory_facts_available,
                !facts.compulsory_memory_facts_available, err) != YVEX_OK ||
            yvex_execution_memory_facts_add(
                &memory, context->device_feature_norm->bytes, 0ull,
                norm_activation, 0ull, 1ull, 0ull, err) != YVEX_OK)
            return yvex_error_code(err);
        if (!yvex_core_u64_add(facts.h2d_bytes,
                               resident_input ? 0ull : input_bytes, &h2d) ||
            !yvex_core_u64_add(facts.d2h_bytes, output_bytes, &d2h) ||
            !yvex_core_u64_add(facts.kernel_launches, 1ull, &kernels) ||
            !yvex_core_u64_add(facts.upload_count, resident_input ? 0ull : 1ull, &uploads) ||
            !yvex_core_u64_add(facts.download_count, 1ull, &downloads) ||
            !yvex_core_u64_add(facts.device_synchronizations,
                               resident_input ? 2ull : 3ull, &synchronizations) ||
            !yvex_core_u64_add(candidate.h2d_bytes, h2d, &candidate.h2d_bytes) ||
            !yvex_core_u64_add(candidate.d2h_bytes, d2h, &candidate.d2h_bytes) ||
            !yvex_core_u64_add(candidate.d2d_bytes, facts.d2d_bytes, &candidate.d2d_bytes) ||
            !yvex_core_u64_add(candidate.kernel_launches, kernels, &candidate.kernel_launches) ||
            !yvex_core_u64_add(candidate.upload_count, uploads, &candidate.upload_count) ||
            !yvex_core_u64_add(candidate.download_count, downloads, &candidate.download_count) ||
            !yvex_core_u64_add(
                candidate.stream_synchronizations,
                facts.stream_synchronizations,
                &candidate.stream_synchronizations) ||
            !yvex_core_u64_add(candidate.device_synchronizations, synchronizations,
                               &candidate.device_synchronizations))
            return speculation_refuse(err, YVEX_ERR_BOUNDS,
                                      "DSpark CUDA feature accounting overflowed");
        candidate.memory = memory;
        *result = candidate;
        context->device_feature_input->is_written = 1;
        context->device_feature_projected->is_written = 1;
        context->device_feature_normalized->is_written = 1;
        if (!yvex_execution_f32_digest(
                "yvex.runtime.speculation.projected-features.v2",
                context->feature_projected, output_elements,
                context->projected_feature_identity))
            return speculation_refuse(err, YVEX_ERR_STATE,
                                      "DSpark projected feature identity could not be sealed");
        return YVEX_OK;
    }
    for (token = 0ull; token < token_count; ++token) {
        const float *source = features + token * context->policy.concatenated_feature_width;
        float *projected = context->feature_projected + token * context->hidden_width;
        for (row = 0ull; row < context->hidden_width; ++row) {
            yvex_quant_failure failure = {0};
            if (context->options.cancel_requested &&
                context->options.cancel_requested(context->options.cancel_context))
                return speculation_refuse(err, YVEX_ERR_CANCELLED,
                                          "DSpark feature projection was cancelled");
            if (yvex_quant_cpu_dot(
                    context->feature_projection.binding->qtype,
                    context->feature_projection.encoded +
                        row * context->feature_projection.row_bytes,
                    (size_t)context->feature_projection.row_bytes, source,
                    context->policy.concatenated_feature_width, &projected[row],
                    &failure, err) != YVEX_OK)
                return yvex_error_code(err);
        }
        if (yvex_transformer_feature_normalize(
                projected, context->hidden_width,
                context->feature_norm_weights,
                yvex_transformer_plan_summary_get(
                    yvex_runtime_transformer_context_plan(
                        context->draft_transformer))->output_norm_epsilon,
                err) != YVEX_OK)
            return yvex_error_code(err);
    }
    if (!yvex_execution_f32_digest(
            "yvex.runtime.speculation.projected-features.v2",
            context->feature_projected, output_elements,
            context->projected_feature_identity))
        return speculation_refuse(err, YVEX_ERR_STATE,
                                  "DSpark projected feature identity could not be sealed");
    return YVEX_OK;
}
static int speculation_weight_matvec(yvex_runtime_speculation_context *context,
    const speculation_weight *weight, const float *input, float *output, yvex_error *err)
{
    unsigned long long row;
    for (row = 0ull; row < weight->binding->row_count; ++row) {
        yvex_quant_failure failure = {0};
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return speculation_refuse(err, YVEX_ERR_CANCELLED,
                                      "DSpark matrix projection was cancelled");
        if (yvex_quant_cpu_dot(
                weight->binding->qtype,
                weight->encoded + row * weight->row_bytes,
                (size_t)weight->row_bytes, input, weight->binding->row_width,
                &output[row], &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        if (!isfinite(output[row]))
            return speculation_refuse(err, YVEX_ERR_FORMAT,
                                      "DSpark matrix projection is non-finite");
    }
    return YVEX_OK;
}
static int speculation_transformer_execute(yvex_runtime_speculation_context *context,
    yvex_runtime_transformer_context *transformer,
    const unsigned int *token_ids, unsigned long long token_count,
    unsigned long long position, int candidate_block_visible,
    int retain_prefix_checkpoints,
    yvex_attention_transaction_disposition disposition,
    const unsigned long long *feature_layers,
    unsigned long long feature_layer_count, float *normalized,
    float *pre_normalized, float *features, unsigned long long feature_row_offset,
    yvex_runtime_transformer_result *result, yvex_error *err)
{
    const yvex_transformer_plan_summary *plan =
        yvex_transformer_plan_summary_get(
            yvex_runtime_transformer_context_plan(transformer));
    yvex_transformer_input_summary summary = {0};
    yvex_transformer_input *input = NULL;
    yvex_runtime_transformer_request request = {0};
    yvex_runtime_transformer_output output = {0};
    unsigned long long value_count, feature_count = 0ull;
    int rc, device_output = transformer == context->draft_transformer && context->device_draft_selection;
    if (!plan || !token_ids || !token_count || (!normalized && !device_output) || !result ||
        !yvex_core_u64_mul(token_count, plan->hidden_width, &value_count) ||
        (features && (!feature_layers || !feature_layer_count ||
                      !yvex_core_u64_mul(value_count, feature_layer_count,
                                         &feature_count))) ||
        (!features && feature_layer_count))
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "speculative transformer geometry is invalid");
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = position;
    summary.token_count = token_count;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    rc = yvex_transformer_input_seal(&summary, token_ids, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(&input, &summary, token_ids, err);
    request.chunk_tokens = token_count;
    request.backend = context->options.backend;
    request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    request.transaction_disposition = disposition;
    request.candidate_block_visible = candidate_block_visible;
    request.retain_prefix_checkpoints = retain_prefix_checkpoints;
    request.feature_layer_ordinals = feature_layers;
    request.feature_layer_count = feature_layer_count;
    output.normalized_hidden = device_output ? NULL : normalized;
    output.capacity = device_output ? 0ull : value_count;
    output.pre_normalized_hidden = device_output ? NULL : pre_normalized;
    output.pre_normalized_capacity = device_output ? 0ull : pre_normalized ? value_count : 0ull;
    output.features = features;
    output.feature_capacity = feature_count;
    if (features && context->device_feature_input) {
        output.device_features = context->device_feature_input;
        output.device_feature_row_offset = feature_row_offset;
        output.device_feature_row_stride = context->policy.concatenated_feature_width;
        context->device_feature_input->is_written = 0;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(transformer, input, &request,
                                              &output, result, err);
    if (rc != YVEX_OK && output.device_features)
        output.device_features->is_written = 0;
    else if (rc == YVEX_OK && output.device_features &&
             !output.device_features->is_written)
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target feature device publication is incomplete");
    yvex_transformer_input_close(&input);
    if (rc == YVEX_OK && disposition == YVEX_ATTENTION_TRANSACTION_ABORT &&
        (result->position_before != position ||
         result->position_after != position ||
         result->generation_after != result->generation_before))
        rc = speculation_refuse(
            err, YVEX_ERR_STATE,
            "speculative transformer candidate state escaped abort");
    return rc;
}
static int speculation_project_draft_base(yvex_runtime_speculation_context *context,
    const yvex_runtime_transformer_result *draft,
    yvex_runtime_logits_row_result *rows, unsigned long long count,
    yvex_runtime_logits_result *execution, yvex_error *err)
{
    const yvex_transformer_plan *plan = yvex_runtime_transformer_context_plan(
        context->draft_transformer);
    yvex_runtime_logits_source sources[YVEX_SPECULATION_MAX_BLOCK] = {{0}};
    yvex_output_head_batch_request request = {0};
    unsigned long long row;
    int device_output = context->device_draft_selection;
    for (row = 0ull; row < count; ++row) {
        int rc = yvex_runtime_logits_source_from_draft(context->target_logits, &sources[row],
            plan, draft,
            device_output ? NULL : context->draft_hidden,
            device_output ? 0ull : count * context->hidden_width, row, err);
        if (rc != YVEX_OK) return rc;
    }
    request.schema_version = YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1;
    request.row_count = count;
    request.output_vocabulary = context->vocabulary_size;
    request.backend = context->options.backend;
    request.result_class = device_output ? YVEX_OUTPUT_HEAD_RESULT_DEVICE_LOGITS
                                         : YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS;
    request.selection_policy = YVEX_OUTPUT_HEAD_SELECTION_RAW;
    request.evidence_profile = context->options.execution_profile->evidence;
    request.execution_class = device_output ? YVEX_EXECUTION_CLASS_DEVICE_NATIVE
                                            : YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.execution_profile_identity = context->options.execution_profile->identity;
    return yvex_runtime_logits_execute_rows(
        context->target_logits, &request, sources, device_output ? NULL : context->base_logits,
        device_output ? 0ull : count * context->vocabulary_size, rows, count, execution, err);
}
static unsigned int speculation_argmax(const float *values, unsigned long long count);
static int speculation_draft_one(
    yvex_runtime_speculation_context *context, unsigned long long ordinal,
    const yvex_runtime_transformer_result *draft,
    unsigned int previous_token, const yvex_runtime_logits_row_result *base_result,
    double uniform, yvex_execution_physical_facts *physical,
    char proposal_identity[YVEX_SPECULATION_IDENTITY_CAP],
    unsigned int *selected, float *confidence, yvex_error *err)
{
    float *markov = context->markov_embedding_values +
                    ordinal * context->policy.markov_rank;
    float *confidence_input = context->feature_projected;
    yvex_runtime_logits_row_result adjusted = {0};
    yvex_runtime_sampling_source source = {0};
    yvex_runtime_sampling_distribution_result distribution = {0};
    yvex_runtime_sampling_result selection = {0};
    yvex_backend_cuda_operation_facts gather_facts = {0}, device_facts = {0};
    yvex_backend_cuda_operation_facts confidence_facts = {0};
    yvex_execution_memory_facts no_memory = {0};
    yvex_device_tensor markov_input = {0}, additive = {0}, pre_normalized = {0};
    yvex_device_tensor device_confidence = {0};
    char adjustment_identity[YVEX_SPECULATION_IDENTITY_CAP];
    unsigned long long pre_offset = 0ull;
    unsigned int token = UINT32_MAX;
    int rc = context->device_draft_selection
        ? (speculation_markov_row_identity(context, previous_token, adjustment_identity) ? YVEX_OK
           : speculation_refuse(err, YVEX_ERR_STATE, "device Markov input identity failed"))
        : speculation_weight_decode_row(&context->markov_embedding, previous_token, markov,
                                        context->policy.markov_rank, err);
    if (rc == YVEX_OK && context->device_draft_selection &&
        (!draft || !draft->pre_normalized_hidden_device_available ||
         draft->device_pre_normalized_hidden.rows != draft->token_count ||
         ordinal >= draft->token_count ||
         !yvex_core_u64_mul(ordinal, context->hidden_width, &pre_offset) ||
         !yvex_core_u64_add(pre_offset, draft->device_pre_normalized_hidden.element_offset, &pre_offset) ||
         !yvex_backend_tensor_f32_subview(draft->device_pre_normalized_hidden.tensor, pre_offset,
                                          context->hidden_width, &pre_normalized) ||
         !yvex_backend_tensor_f32_subview(context->device_feature_projected, 0ull,
                                          context->policy.markov_rank, &markov_input) ||
         !yvex_backend_tensor_f32_subview(context->device_feature_projected,
                                          context->policy.markov_rank, 1ull, &device_confidence) ||
         !yvex_backend_tensor_f32_subview(base_result->device_logits.tensor,
                                          base_result->device_logits.element_offset,
                                          context->vocabulary_size, &additive)))
        rc = speculation_refuse(err, YVEX_ERR_BOUNDS, "device Markov projection extent is invalid");
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_backend_cuda_encoded_gather(
            context->device_backend, context->markov_embedding.encoded,
            context->markov_embedding.encoded_bytes, context->markov_embedding.binding->qtype,
            context->markov_embedding.binding->row_count,
            context->policy.markov_rank, context->markov_embedding.row_bytes,
            &previous_token, 1ull, &markov_input, &gather_facts, err);
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_backend_cuda_encoded_matvec(
            context->device_backend, context->markov_output.encoded,
            context->markov_output.encoded_bytes,
            context->markov_output.binding->qtype, context->vocabulary_size,
            context->policy.markov_rank, context->markov_output.row_bytes,
            1ull, &markov_input, NULL, 0ull, &additive,
            context->device_adjusted_logits, &device_facts, err);
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_runtime_logits_additive_adjust(
            context->target_logits, base_result, NULL, NULL, NULL, 0ull,
            context->device_adjusted_logits, 0ull, adjustment_identity, &adjusted, err);
    if (rc == YVEX_OK && !context->device_draft_selection)
        rc = speculation_weight_matvec(context, &context->markov_output,
                                       markov, context->markov_bias, err);
    if (rc == YVEX_OK && !context->device_draft_selection)
        rc = yvex_runtime_logits_additive_adjust(
            context->target_logits, base_result,
            context->base_logits + ordinal * context->vocabulary_size,
            context->markov_bias, context->adjusted_logits, context->vocabulary_size,
            NULL, 0ull, NULL, &adjusted, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_source_from_logits(
            context->draft_sampling, &source,
            context->device_draft_selection ? NULL : context->adjusted_logits,
            context->device_draft_selection ? 0ull : context->vocabulary_size,
            &adjusted, err);
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_runtime_sampling_select(context->draft_sampling, &source, &selection, err);
    if (rc == YVEX_OK && !context->device_draft_selection)
        rc = yvex_runtime_sampling_distribution(
            context->draft_sampling, &source,
            context->draft_probabilities + ordinal * context->vocabulary_size,
            context->vocabulary_size, &distribution, err);
    if (rc == YVEX_OK) {
        if (context->device_draft_selection) {
            token = selection.selected_token_id;
            yvex_runtime_identity_copy(proposal_identity, selection.selected_token_identity);
        } else {
            token = context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                    ? speculation_argmax(
                          context->draft_probabilities +
                              ordinal * context->vocabulary_size,
                          context->vocabulary_size)
                    : yvex_speculation_distribution_sample(
                          context->draft_probabilities +
                              ordinal * context->vocabulary_size,
                          NULL, context->vocabulary_size, uniform, 0);
        }
    }
    if (rc == YVEX_OK && token == UINT32_MAX)
        rc = speculation_refuse(err, YVEX_ERR_FORMAT, "draft distribution has no selectable mass");
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_backend_cuda_encoded_matvec(
            context->device_backend, context->confidence.encoded,
            context->confidence.encoded_bytes, context->confidence.binding->qtype,
            1ull, context->confidence.binding->row_width, context->confidence.row_bytes,
            1ull, &pre_normalized, &markov_input, context->hidden_width, NULL,
            &device_confidence, &confidence_facts, err);
    if (rc == YVEX_OK && context->device_draft_selection)
        rc = yvex_backend_tensor_read(context->device_backend, &device_confidence,
                                      confidence, sizeof(*confidence), err);
    if (rc == YVEX_OK && !context->device_draft_selection) {
        memcpy(confidence_input, context->draft_pre_normalized + ordinal * context->hidden_width,
               (size_t)context->hidden_width * sizeof(float));
        memcpy(confidence_input + context->hidden_width, markov,
               (size_t)context->policy.markov_rank * sizeof(float));
        rc = speculation_weight_matvec(
            context, &context->confidence, confidence_input, confidence, err);
    }
    if (rc == YVEX_OK && context->device_draft_selection &&
        (speculation_cuda_physical_add(physical, &gather_facts, err) != YVEX_OK ||
         speculation_cuda_physical_add(physical, &device_facts, err) != YVEX_OK ||
         speculation_cuda_physical_add(physical, &confidence_facts, err) != YVEX_OK ||
         yvex_execution_physical_facts_add(
             physical, &no_memory, 0ull, selection.d2h_bytes, 0ull,
             selection.kernel_launches, selection.device_synchronizations, err) != YVEX_OK ||
         yvex_execution_physical_facts_add(
             physical, &no_memory, 0ull, sizeof(*confidence), 0ull, 0ull, 1ull, err) != YVEX_OK))
        rc = yvex_error_code(err);
    if (rc == YVEX_OK) *selected = token;
    return rc;
}
static unsigned int speculation_argmax(const float *values, unsigned long long count)
{
    unsigned long long index, selected = 0ull;
    for (index = 1ull; index < count; ++index)
        if (values[index] > values[selected]) selected = index;
    return (unsigned int)selected;
}
static int speculation_phase_physical(const yvex_runtime_transformer_result *transformer,
    const yvex_runtime_logits_result *execution, yvex_execution_physical_facts *facts,
    yvex_error *err)
{
    yvex_execution_physical_facts candidate = {0};
    unsigned long long transformer_sync;
    int rc;
    if (!transformer || !execution || !execution->completed || !execution->completed_rows || !facts)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark physical accounting requires complete owners");
    if (!yvex_core_u64_add(transformer->stream_synchronizations,
                           transformer->device_synchronizations,
                           &transformer_sync))
        return speculation_refuse(err, YVEX_ERR_BOUNDS, "DSpark physical accounting overflowed");
    rc = yvex_execution_physical_facts_add(
        &candidate, &transformer->memory, transformer->h2d_bytes, transformer->d2h_bytes,
        transformer->d2d_bytes, transformer->kernel_launches, transformer_sync, err);
    if (rc == YVEX_OK)
        rc = yvex_execution_physical_facts_add(
            &candidate, &execution->physical.memory, execution->physical.h2d_bytes,
            execution->physical.d2h_bytes,
            execution->physical.d2d_bytes, execution->physical.kernel_count,
            execution->physical.synchronization_count, err);
    if (rc != YVEX_OK) return rc;
    *facts = candidate;
    return YVEX_OK;
}
static int speculation_execute_draft(yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result draft = {0};
    yvex_runtime_logits_row_result rows[YVEX_SPECULATION_MAX_BLOCK] = {{0}};
    yvex_runtime_logits_result logits_execution = {0};
    yvex_runtime_sampling_uniform_result draw_result = {0};
    yvex_sha256 hash;
    unsigned long long started, index, draft_count = context->policy.block_size;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int rc;
    context->draft_input_ids[0] = request->conditioning_token_id;
    for (index = 1ull; index < draft_count; ++index)
        context->draft_input_ids[index] = (unsigned int)context->policy.noise_token_id;
    started = yvex_core_monotonic_ns();
    /* Position -1 features belong to the input whose target distribution produced the conditioning
     * token. The reference stores them before embedding that token beside noise; reprocessing it
     * shifts both caches and every proposal. Execute the complete source-authored block: attention
     * is non-causal within it, so truncation changes verified hidden state. The request extent
     * limits only the prefix handed to the verifier. */
    rc = speculation_transformer_execute(
        context, context->draft_transformer, context->draft_input_ids,
        draft_count, request->position, 1, 0,
        YVEX_ATTENTION_TRANSACTION_ABORT, NULL, 0ull,
        context->draft_hidden, context->draft_pre_normalized, NULL, 0ull,
        &draft, err);
    if (rc == YVEX_OK)
        rc = speculation_project_draft_base(
            context, &draft, rows, draft_count, &logits_execution, err);
    if (rc == YVEX_OK)
        rc = speculation_phase_physical(&draft, &logits_execution, &result->draft_physical, err);
    if (rc == YVEX_OK &&
        context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_STOCHASTIC)
        rc = yvex_runtime_sampling_transaction_begin(
            context->draft_sampling, &context->draft_rng, err);
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(
             &hash, context->device_draft_selection
                        ? "yvex.runtime.speculation.draft.device-greedy.v1"
                        : "yvex.runtime.speculation.draft.v1") ||
         !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
         !yvex_sha256_update_text(&hash, draft.execution_identity) ||
         !yvex_sha256_update_u64(&hash, request->position) ||
         !yvex_sha256_update_u64(&hash, draft_count) ||
         !yvex_sha256_update_u64(&hash, request->candidate_count)))
        rc = speculation_refuse(
            err, YVEX_ERR_STATE, "draft execution identity initialization failed");
    for (index = 0ull; rc == YVEX_OK && index < draft_count; ++index) {
        char proposal_identity[YVEX_SPECULATION_IDENTITY_CAP];
        unsigned int previous = index ? result->candidate_token_ids[index - 1ull]
                                      : request->conditioning_token_id;
        double uniform = 0.0;
        if (context->draft_rng)
            rc = yvex_runtime_sampling_transaction_uniforms(
                context->draft_rng, &uniform, 1ull, &draw_result, err);
        if (rc == YVEX_OK) result->draft_rng_draw_count += draw_result.draw_count;
        if (rc == YVEX_OK)
            rc = speculation_draft_one(
                context, index, &draft, previous, &rows[index], uniform,
                &result->draft_physical, proposal_identity,
                &result->candidate_token_ids[index],
                &result->confidence_logits[index], err);
        if (rc == YVEX_OK && context->device_draft_selection &&
            (!yvex_sha256_update_u64(&hash, result->candidate_token_ids[index]) ||
             !yvex_sha256_update_text(&hash, proposal_identity)))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE, "draft distribution identity failed");
    }
    result->draft_ns = yvex_core_monotonic_ns() - started;
    for (index = 0ull; rc == YVEX_OK && !context->device_draft_selection && index < draft_count;
         ++index) {
        char probabilities[YVEX_SPECULATION_IDENTITY_CAP];
        if (!yvex_execution_f32_digest(
                "yvex.runtime.speculation.draft-distribution.v1",
                context->draft_probabilities + index * context->vocabulary_size,
                context->vocabulary_size, probabilities) ||
            !yvex_sha256_update_u64(&hash, result->candidate_token_ids[index]) ||
            !yvex_sha256_update_text(&hash, probabilities))
            rc = speculation_refuse(err, YVEX_ERR_STATE, "draft distribution identity failed");
    }
    if (rc == YVEX_OK) {
        char confidence[YVEX_SPECULATION_IDENTITY_CAP];
        if (!yvex_execution_f32_digest("yvex.runtime.speculation.confidence.v1",
                result->confidence_logits, draft_count, confidence) ||
            !yvex_sha256_update_text(&hash, confidence))
            rc = speculation_refuse(err, YVEX_ERR_STATE, "draft confidence identity failed");
    }
    if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest))
        rc = speculation_refuse(
            err, YVEX_ERR_STATE, "draft execution identity finalization failed");
    if (rc == YVEX_OK)
        yvex_sha256_hex(digest, result->draft_execution_identity);
    return rc;
}
static int speculation_verify_target(yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_runtime_logits_context *logits_owner = context->verification_logits
        ? context->verification_logits : context->target_logits;
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_logits_source sources[YVEX_SPECULATION_MAX_BLOCK + 1u] = {{0}};
    yvex_runtime_logits_row_result logits[YVEX_SPECULATION_MAX_BLOCK + 1u] = {{0}};
    yvex_runtime_logits_result logits_execution = {0};
    yvex_runtime_sampling_source sampling_sources[YVEX_SPECULATION_MAX_BLOCK + 1u] = {{0}};
    yvex_runtime_sampling_result selections[YVEX_SPECULATION_MAX_BLOCK + 1u] = {{0}};
    yvex_runtime_sampling_execution sampling_execution = {0};
    yvex_output_head_batch_request output_head = {0};
    unsigned int verification_tokens[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long started, row;
    yvex_runtime_model_failure failure = {0};
    int device_verification = context->verification_logits != NULL, acquired = 0, rc;
    verification_tokens[0] = request->conditioning_token_id;
    memcpy(verification_tokens + 1u, result->candidate_token_ids,
           (size_t)request->candidate_count * sizeof(*verification_tokens));
    started = yvex_core_monotonic_ns();
    /* Row zero follows the target-authored conditioning token; the final row supplies correction
     * or bonus when the complete draft survives. */
    rc = yvex_runtime_session_begin(context->session, &failure, err);
    acquired = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_transformer_execute(
            context, context->target_transformer, verification_tokens,
            request->candidate_count + 1ull, request->position, 0, 1,
            YVEX_ATTENTION_TRANSACTION_STAGE,
            context->policy.target_feature_layers,
            context->policy.target_feature_layer_count,
            context->target_hidden, NULL, context->target_features, 0ull,
            &target, err);
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(
             &hash, device_verification
                        ? "yvex.runtime.speculation.verify.device-greedy.v1"
                        : "yvex.runtime.speculation.verify.v1") ||
         !yvex_sha256_update_text(&hash, context->pending_source_identity) ||
         !yvex_sha256_update_text(&hash, target.execution_identity)))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target verification identity initialization failed");
    for (row = 0ull; rc == YVEX_OK && row <= request->candidate_count; ++row)
        rc = yvex_runtime_logits_source_from_transformer(
            logits_owner, &sources[row], &target, context->target_hidden,
            (request->candidate_count + 1ull) * context->hidden_width, row, err);
    output_head.schema_version = YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1;
    output_head.row_count = request->candidate_count + 1ull;
    output_head.output_vocabulary = context->vocabulary_size;
    output_head.backend = context->options.backend;
    output_head.result_class = device_verification ? YVEX_OUTPUT_HEAD_RESULT_DEVICE_LOGITS
                                                   : YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS;
    output_head.selection_policy = YVEX_OUTPUT_HEAD_SELECTION_RAW;
    output_head.evidence_profile = context->options.execution_profile->evidence;
    output_head.execution_class = device_verification ? YVEX_EXECUTION_CLASS_DEVICE_NATIVE
                                                      : YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    output_head.execution_profile_identity = context->options.execution_profile->identity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_execute_rows(
            logits_owner, &output_head, sources,
            device_verification ? NULL : context->base_logits,
            device_verification ? 0ull
                                : (request->candidate_count + 1ull) * context->vocabulary_size,
            logits, request->candidate_count + 1ull, &logits_execution, err);
    if (rc == YVEX_OK)
        rc = speculation_phase_physical(
            &target, &logits_execution,
            &result->verification_physical, err);
    for (row = 0ull; rc == YVEX_OK && row <= request->candidate_count; ++row)
        rc = yvex_runtime_sampling_source_from_logits(
            device_verification ? context->verification_sampling : context->target_sampling,
            &sampling_sources[row],
            device_verification ? NULL : context->base_logits + row * context->vocabulary_size,
            device_verification ? 0ull : context->vocabulary_size, &logits[row], err);
    if (rc == YVEX_OK && device_verification)
        rc = yvex_runtime_sampling_execute(
            context->verification_sampling, sampling_sources, request->candidate_count + 1ull,
            selections, YVEX_SPECULATION_MAX_BLOCK + 1ull, &sampling_execution, err);
    for (row = 0ull; rc == YVEX_OK && row <= request->candidate_count; ++row) {
        if (device_verification) {
            yvex_execution_memory_facts no_memory = {0};
            context->target_token_ids[row] = selections[row].selected_token_id;
            rc = yvex_execution_physical_facts_add(
                &result->verification_physical, &no_memory, 0ull,
                selections[row].d2h_bytes, 0ull, selections[row].kernel_launches,
                selections[row].device_synchronizations, err);
        } else {
            yvex_runtime_sampling_distribution_result distribution = {0};
            rc = yvex_runtime_sampling_distribution(
                context->target_sampling, &sampling_sources[row],
                context->target_probabilities + row * context->vocabulary_size,
                context->vocabulary_size, &distribution, err);
            if (rc == YVEX_OK &&
                !yvex_sha256_update_text(&hash, distribution.distribution_identity))
                rc = speculation_refuse(err, YVEX_ERR_STATE,
                                        "target verification identity update failed");
        }
        if (rc == YVEX_OK && device_verification &&
            !yvex_sha256_update_text(&hash, selections[row].selected_token_identity))
            rc = speculation_refuse(err, YVEX_ERR_STATE,
                                    "target verification identity update failed");
    }
    result->verification_ns = yvex_core_monotonic_ns() - started;
    if (rc == YVEX_OK && !yvex_sha256_final(&hash, digest))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target verification identity finalization failed");
    if (rc == YVEX_OK) {
        yvex_sha256_hex(digest, result->verification_execution_identity);
        result->target_verification_count = 1ull;
        context->pending_verification_target = target;
        context->verification_staged = 1;
    }
    if (acquired && rc != YVEX_OK)
        rc = yvex_runtime_session_finish_scope(context->session,
            YVEX_TENSOR_SCOPE_GLOBAL, YVEX_ATTENTION_TRANSACTION_ABORT, rc, err);
    return rc;
}
static int speculation_target_draws(yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result,
    double uniforms[YVEX_SPECULATION_MAX_BLOCK], double *correction,
    yvex_error *err)
{
    yvex_runtime_sampling_uniform_result draw_result = {0};
    unsigned long long index;
    int rc = context->target_rng
                 ? YVEX_OK
                 : yvex_runtime_sampling_transaction_begin(
                       context->target_sampling, &context->target_rng, err);
    /* DeepSpec draws the whole mask before taking its contiguous prefix. Consuming every position
     * keeps target RNG stable when early rejection makes later draws observationally inert. */
    for (index = 0ull; rc == YVEX_OK && index < request->candidate_count;
         ++index) {
        unsigned int candidate = result->candidate_token_ids[index];
        double q;
        rc = yvex_runtime_sampling_transaction_uniforms(
            context->target_rng, &uniforms[index], 1ull, &draw_result, err);
        if (rc == YVEX_OK) result->target_rng_draw_count += draw_result.draw_count;
        q = context->draft_probabilities[
            index * context->vocabulary_size + candidate];
        if (rc == YVEX_OK && q <= 0.0)
            rc = speculation_refuse(err, YVEX_ERR_FORMAT,
                                    "draft proposal has zero admitted probability");
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_transaction_uniforms(
            context->target_rng, correction, 1ull, &draw_result, err);
    if (rc == YVEX_OK) result->target_rng_draw_count += draw_result.draw_count;
    return rc;
}
static int speculation_accept_cycle(yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    yvex_speculation_acceptance_request acceptance = {0};
    unsigned int target_tokens[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    unsigned int accepted[YVEX_SPECULATION_MAX_BLOCK + 1u] = {0};
    double uniforms[YVEX_SPECULATION_MAX_BLOCK] = {0.0};
    double correction = 0.0;
    unsigned long long started, index;
    int rc = YVEX_OK;
    acceptance.schema_version = YVEX_SPECULATION_SCHEMA_V1;
    acceptance.kind = context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY
                          ? YVEX_SPECULATION_ACCEPT_GREEDY
                          : YVEX_SPECULATION_ACCEPT_STOCHASTIC;
    acceptance.candidate_count = request->candidate_count;
    acceptance.vocabulary_size = context->vocabulary_size;
    acceptance.distribution_stride = context->vocabulary_size;
    acceptance.candidate_token_ids = result->candidate_token_ids;
    acceptance.draft_probabilities = context->draft_probabilities;
    acceptance.target_probabilities = context->target_probabilities;
    acceptance.acceptance_uniforms = uniforms;
    if (acceptance.kind == YVEX_SPECULATION_ACCEPT_GREEDY) {
        if (context->verification_logits)
            memcpy(target_tokens, context->target_token_ids,
                   (size_t)(request->candidate_count + 1ull) * sizeof(*target_tokens));
        else
            for (index = 0ull; index <= request->candidate_count; ++index)
                target_tokens[index] = speculation_argmax(
                    context->target_probabilities + index * context->vocabulary_size,
                    context->vocabulary_size);
        acceptance.target_token_ids = target_tokens;
    } else {
        rc = speculation_target_draws(
            context, request, result, uniforms, &correction, err);
        acceptance.correction_uniform = correction;
    }
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = yvex_speculation_accept(
            &acceptance, accepted,
            YVEX_SPECULATION_MAX_BLOCK + 1ull, &result->acceptance, err);
    result->acceptance_ns = yvex_core_monotonic_ns() - started;
    if (rc == YVEX_OK) {
        memcpy(result->committed_token_ids, accepted,
               (size_t)result->acceptance.committed_count * sizeof(*accepted));
        result->committed_count = result->acceptance.committed_count;
    }
    return rc;
}
static int speculation_cycle_identity(const yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result)
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(&hash, "yvex.runtime.speculation.cycle.v1") &&
           yvex_sha256_update_text(&hash, context->policy.policy_identity) &&
           yvex_sha256_update_text(&hash, context->pending_cycle_identity) &&
           yvex_sha256_update_u64(&hash, request->position) &&
           yvex_sha256_update_u64(&hash, request->conditioning_token_id) &&
           yvex_sha256_update_u64(&hash, result->draft_proposed_count) &&
           yvex_sha256_update_u64(&hash, result->candidate_count) &&
           yvex_sha256_update_text(&hash, result->draft_execution_identity) &&
           yvex_sha256_update_text(&hash, result->verification_execution_identity) &&
           yvex_sha256_update_text(&hash,
                                   result->acceptance.acceptance_identity) &&
           speculation_hash_finish(&hash, result->cycle_identity);
}
int yvex_runtime_speculation_cycle(yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_cycle_request *request,
    yvex_runtime_speculation_cycle_result *result, yvex_error *err)
{
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !request || !result || !context->cycle_pending ||
        context->pending_committed_count != 1ull ||
        context->pending_position != request->position ||
        context->pending_tokens[0] != request->conditioning_token_id ||
        !request->candidate_count ||
        request->candidate_count > context->policy.block_size ||
        request->candidate_count > context->policy.accepted_prefix_maximum ||
        request->conditioning_token_id >= context->vocabulary_size ||
        !yvex_sha256_hex_valid(context->pending_source_identity) ||
        !yvex_sha256_hex_valid(context->pending_sampling_identity))
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark cycle request does not continue its target anchor");
    result->draft_proposed_count = context->policy.block_size;
    result->candidate_count = request->candidate_count;
    result->conditioning_token_id = request->conditioning_token_id;
    result->draft_started = 1;
    rc = request->phase_sink
             ? request->phase_sink(
                   request->phase_context,
                   YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_STARTED, 0ull, err)
             : YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_execute_draft(context, request, result, err);
    if (rc == YVEX_OK) result->draft_completed = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_DRAFT_COMPLETED,
            result->draft_ns, err);
    if (rc == YVEX_OK) result->verification_started = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_STARTED, 0ull, err);
    if (rc == YVEX_OK)
        rc = speculation_verify_target(context, request, result, err);
    if (rc == YVEX_OK) result->verification_completed = 1;
    if (rc == YVEX_OK && request->phase_sink)
        rc = request->phase_sink(
            request->phase_context,
            YVEX_RUNTIME_SPECULATION_PHASE_VERIFICATION_COMPLETED,
            result->verification_ns, err);
    if (rc == YVEX_OK)
        rc = speculation_accept_cycle(context, request, result, err);
    if (rc == YVEX_OK && !speculation_cycle_identity(context, request, result))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "DSpark cycle identity derivation failed");
    if (rc == YVEX_OK) {
        result->completed = 1;
        context->pending_committed_count = result->committed_count + 1ull;
        context->pending_verified_prefix_count =
            result->acceptance.accepted_draft_count + 1ull;
        memcpy(context->pending_tokens + 1u, result->committed_token_ids,
               (size_t)result->committed_count *
                   sizeof(*context->pending_tokens));
        yvex_runtime_identity_copy(context->pending_cycle_identity,
                                   result->cycle_identity);
        context->cycle_pending = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (context->verification_staged) {
        yvex_error primary = err ? *err : (yvex_error){0};
        int finish_rc = yvex_runtime_session_finish_scope(
            context->session, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_ATTENTION_TRANSACTION_ABORT, rc, err);
        context->verification_staged = 0;
        if (finish_rc != YVEX_OK) rc = finish_rc;
        else if (err) *err = primary;
    }
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    speculation_pending_clear(context);
    return rc;
}
int yvex_runtime_speculation_target_step(
    yvex_runtime_speculation_context *context, unsigned long long position,
    const float *target_probabilities, unsigned long long probability_capacity,
    const char *distribution_identity,
    yvex_runtime_speculation_target_step_result *result, yvex_error *err)
{
    yvex_runtime_sampling_uniform_result draw = {0};
    yvex_sha256 hash;
    double uniform = 0.0;
    unsigned int token;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result || context->cycle_pending ||
        !target_probabilities || probability_capacity < context->vocabulary_size ||
        position >= context->options.context_capacity ||
        !yvex_sha256_hex_valid(distribution_identity) ||
        !yvex_speculation_distribution_valid(target_probabilities,
                                             context->vocabulary_size))
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "target step requires one admitted distribution and no pending cycle");
    if (context->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY)
        token = speculation_argmax(target_probabilities,
                                   context->vocabulary_size);
    else {
        rc = yvex_runtime_sampling_transaction_begin(
            context->target_sampling, &context->target_rng, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_sampling_transaction_uniforms(
                context->target_rng, &uniform, 1ull, &draw, err);
        token = rc == YVEX_OK
                    ? yvex_speculation_distribution_sample(
                          target_probabilities, NULL,
                          context->vocabulary_size, uniform, 0)
                    : UINT32_MAX;
    }
    if (rc == YVEX_OK && token == UINT32_MAX)
        rc = speculation_refuse(err, YVEX_ERR_FORMAT,
                                "target distribution has no selectable mass");
    yvex_sha256_init(&hash);
    if (rc == YVEX_OK &&
        (!yvex_sha256_update_text(
             &hash, "yvex.runtime.speculation.target-step.v1") ||
         !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
         !yvex_sha256_update_text(&hash, distribution_identity) ||
         !yvex_sha256_update_u64(&hash, position) ||
         !yvex_sha256_update_u64(&hash, token) ||
         !speculation_hash_finish(&hash, result->cycle_identity)))
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "target step identity derivation failed");
    if (rc == YVEX_OK) {
        result->completed = 1;
        result->position = position;
        result->token_id = token;
        result->target_rng_draw_count = draw.draw_count;
        yvex_runtime_identity_copy(result->source_distribution_identity, distribution_identity);
        yvex_runtime_identity_copy(result->sampling_identity, result->cycle_identity);
        context->pending_position = position;
        context->pending_committed_count = 1ull;
        context->pending_tokens[0] = token;
        yvex_runtime_identity_copy(context->pending_source_identity, distribution_identity);
        yvex_runtime_identity_copy(context->pending_sampling_identity, result->cycle_identity);
        yvex_runtime_identity_copy(context->pending_cycle_identity,
                                   result->cycle_identity);
        context->cycle_pending = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    return rc;
}
static void speculation_pending_clear(yvex_runtime_speculation_context *context)
{
    if (!context) return;
    context->cycle_pending = 0;
    context->pending_position = 0ull;
    context->pending_committed_count = 0ull;
    context->pending_verified_prefix_count = 0ull;
    memset(context->pending_tokens, 0, sizeof(context->pending_tokens));
    memset(&context->pending_verification_target, 0,
           sizeof(context->pending_verification_target));
    context->pending_source_identity[0] = '\0';
    context->pending_sampling_identity[0] = '\0';
    context->pending_cycle_identity[0] = '\0';
    context->verification_staged = 0;
}
static int speculation_rng_prepare(void *opaque, yvex_error *err) {
    yvex_runtime_speculation_context *context = opaque;
    int rc = YVEX_OK;
    if (context->target_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->target_rng, err);
    if (rc == YVEX_OK && context->draft_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->draft_rng, err);
    if (rc == YVEX_OK && context->publication) {
        rc = context->publication->prepare(context->publication->context, err);
        context->publication_prepared = rc == YVEX_OK;
    }
    if (rc != YVEX_OK) {
        (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
        (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    }
    return rc;
}
static void speculation_rng_publish(void *opaque) {
    yvex_runtime_speculation_context *context = opaque;
    yvex_runtime_sampling_transaction_publish_commit(&context->target_rng);
    yvex_runtime_sampling_transaction_publish_commit(&context->draft_rng);
    if (context->publication_prepared)
        context->publication->publish(context->publication->context);
    context->publication = NULL;
    context->publication_prepared = 0;
    speculation_pending_clear(context);
}
static void speculation_rng_cancel(void *opaque) {
    yvex_runtime_speculation_context *context = opaque;
    (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    if (context->publication)
        context->publication->cancel(context->publication->context);
    context->publication = NULL;
    context->publication_prepared = 0;
    speculation_pending_clear(context);
}
static int speculation_stage_draft_features(
    yvex_runtime_speculation_context *context, unsigned long long token_start,
    unsigned long long token_count, yvex_runtime_transformer_core_commit_result *result,
    yvex_error *err)
{
    const yvex_device_tensor *resident = context
        ? context->device_feature_normalized : NULL;
    int rc;
    if (context && context->device_feature_normalized) {
        if (!context->device_feature_normalized->is_written)
            return speculation_refuse(
                err, YVEX_ERR_STATE,
                "DSpark normalized device features are unpublished");
    }
    rc = yvex_runtime_transformer_stage_core_features(
        context->draft_transformer, token_start, context->feature_projected,
        resident, context->projected_feature_identity, token_count, result, err);
    if (rc == YVEX_OK && resident && !result->device_input_consumed)
        rc = speculation_refuse(err, YVEX_ERR_STATE,
                                "DSpark draft core ignored its resident feature input");
    return rc;
}
static int speculation_stage_tokens(
    yvex_runtime_speculation_context *context, const unsigned int *token_ids,
    unsigned long long token_start, unsigned long long token_count,
    yvex_runtime_transformer_result *target, yvex_runtime_transformer_core_commit_result *draft,
    yvex_error *err)
{
    int rc = speculation_transformer_execute(
        context, context->target_transformer, token_ids,
        token_count, token_start, 0, 0,
        YVEX_ATTENTION_TRANSACTION_STAGE,
        context->policy.target_feature_layers,
        context->policy.target_feature_layer_count, context->target_hidden,
        NULL, context->target_features, 0ull, target, err);
    if (rc == YVEX_OK)
        rc = speculation_project_target_features(
            context, context->target_features, token_count, target, err);
    if (rc == YVEX_OK)
        rc = speculation_stage_draft_features(
            context, token_start, token_count, draft, err);
    return rc;
}
int yvex_runtime_speculation_prefill(
    yvex_runtime_speculation_context *context, const unsigned int *token_ids,
    unsigned long long token_start, unsigned long long token_count,
    float *normalized_hidden, unsigned long long normalized_hidden_capacity,
    yvex_runtime_transformer_result *target_result,
    yvex_runtime_speculation_feature_result *draft_result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_transformer_core_commit_result draft = {0};
    yvex_runtime_speculation_feature_result prepared = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_sha256 hash;
    unsigned long long feature_values, synchronizations;
    int acquired = 0, rc;
    if (target_result) memset(target_result, 0, sizeof(*target_result));
    if (draft_result) memset(draft_result, 0, sizeof(*draft_result));
    if (!context || !token_ids || !token_count ||
        token_count > context->policy.block_size + 2ull || !normalized_hidden ||
        !target_result || !draft_result || context->cycle_pending ||
        token_count > context->options.context_capacity ||
        token_start > context->options.context_capacity - token_count ||
        !yvex_core_u64_mul(token_count, context->hidden_width, &feature_values) ||
        normalized_hidden_capacity < feature_values)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark prefill extent is not admitted");
    rc = yvex_runtime_session_begin(context->session, &failure, err);
    acquired = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_stage_tokens(context, token_ids, token_start, token_count,
                                      &target, &draft, err);
    if (rc == YVEX_OK) {
        prepared.token_start = token_start;
        prepared.token_count = token_count;
        prepared.position_after = draft.position_after;
        yvex_runtime_identity_copy(prepared.persistent_state_digest,
                                   draft.persistent_state_digest);
        if (target.position_after != token_start + token_count ||
            draft.position_after != token_start + token_count ||
            !yvex_sha256_hex_valid(target.persistent_state_digest) ||
            !yvex_sha256_hex_valid(prepared.persistent_state_digest) ||
            !yvex_sha256_hex_valid(context->projected_feature_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "DSpark prefill staged identity is incomplete");
        else
            yvex_runtime_identity_copy(prepared.projected_feature_digest,
                                       context->projected_feature_identity);
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(target.stream_synchronizations,
                            target.device_synchronizations,
                            &synchronizations) ||
         yvex_execution_physical_facts_add(
             &prepared.physical, &target.memory, target.h2d_bytes,
             target.d2h_bytes, target.d2d_bytes, target.kernel_launches,
             synchronizations, err) != YVEX_OK ||
         yvex_execution_physical_facts_add(
             &prepared.physical, &draft.physical.memory,
             draft.physical.h2d_bytes, draft.physical.d2h_bytes,
             draft.physical.d2d_bytes, draft.physical.kernel_count,
             draft.physical.synchronization_count, err) != YVEX_OK))
        rc = yvex_error_is_set(err)
                 ? yvex_error_code(err)
                 : speculation_refuse(
                       err, YVEX_ERR_BOUNDS,
                       "DSpark prefill physical accounting overflowed");
    if (rc == YVEX_OK) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(
                &hash, "yvex.runtime.speculation.prefill.v1") ||
            !yvex_sha256_update_text(&hash, context->policy.policy_identity) ||
            !yvex_sha256_update_text(&hash, target.execution_identity) ||
            !yvex_sha256_update_text(&hash, draft.execution_identity) ||
            !yvex_sha256_update_text(&hash, prepared.projected_feature_digest) ||
            !yvex_sha256_update_text(&hash, prepared.persistent_state_digest) ||
            !yvex_sha256_update_u64(&hash, token_start) ||
            !yvex_sha256_update_u64(&hash, token_count) ||
            !speculation_hash_finish(&hash, prepared.execution_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "DSpark prefill identity derivation failed");
    }
    if (acquired)
        rc = yvex_runtime_session_finish_coordinated(context->session, rc, NULL, err);
    if (rc != YVEX_OK) return rc;
    memcpy(normalized_hidden, context->target_hidden,
           (size_t)feature_values * sizeof(float));
    *target_result = target;
    prepared.completed = 1;
    *draft_result = prepared;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int speculation_commit_identity(const yvex_runtime_speculation_context *context,
    const yvex_runtime_speculation_commit_result *result,
    char output[YVEX_SPECULATION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return yvex_sha256_update_text(
               &hash, "yvex.runtime.speculation.commit.v1") &&
           yvex_sha256_update_text(&hash, context->policy.policy_identity) &&
           yvex_sha256_update_text(
               &hash, result->target_execution_identity) &&
           yvex_sha256_update_text(
               &hash, result->draft_execution_identity) &&
           yvex_sha256_update_text(&hash, result->target_state_identity) &&
           yvex_sha256_update_text(&hash, result->draft_state_identity) &&
           yvex_sha256_update_u64(&hash, result->token_start) &&
           yvex_sha256_update_u64(&hash, result->token_count) &&
           speculation_hash_finish(&hash, output);
}
/* A short tail may leave no draft room. Its target-authored anchor still advances both states,
 * but is not a verified candidate prefix and never enters prefix promotion. */
static int speculation_commit_target_step(
    yvex_runtime_speculation_context *context, float *final_hidden,
    const yvex_runtime_commit_participant *publication,
    yvex_runtime_speculation_commit_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_transformer_core_commit_result draft = {0};
    yvex_runtime_commit_participant participant = {
        .context = context, .prepare = speculation_rng_prepare,
        .publish = speculation_rng_publish, .cancel = speculation_rng_cancel};
    yvex_runtime_model_failure failure = {0};
    unsigned long long started = yvex_core_monotonic_ns();
    int acquired = 0, rc;
    rc = yvex_runtime_session_begin(context->session, &failure, err);
    acquired = rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = speculation_stage_tokens(
            context, context->pending_tokens, context->pending_position, 1ull,
            &target, &draft, err);
    if (rc == YVEX_OK) {
        result->token_start = context->pending_position;
        result->token_count = 1ull;
        result->position_after = context->pending_position + 1ull;
        result->target_extension_count = 1ull;
        result->target_result = target;
        yvex_runtime_identity_copy(result->cycle_identity,
                                   context->pending_cycle_identity);
        yvex_runtime_identity_copy(result->target_execution_identity, target.execution_identity);
        yvex_runtime_identity_copy(result->draft_execution_identity,
                                   draft.execution_identity);
        yvex_runtime_identity_copy(result->target_state_identity,
                                   target.persistent_state_digest);
        yvex_runtime_identity_copy(result->draft_state_identity,
                                   draft.persistent_state_digest);
        if (target.position_after != result->position_after ||
            draft.position_after != result->position_after ||
            !yvex_sha256_hex_valid(result->target_state_identity) ||
            !yvex_sha256_hex_valid(result->draft_state_identity) ||
            !speculation_commit_identity(context, result,
                                         result->commit_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "target-authored DSpark tail has incomplete staged identity");
    }
    if (acquired) {
        context->publication = publication;
        context->publication_prepared = 0;
        rc = yvex_runtime_session_finish_coordinated(
            context->session, rc, &participant, err);
    }
    if (rc != YVEX_OK) return rc;
    memcpy(final_hidden, context->target_hidden,
           (size_t)context->hidden_width * sizeof(*final_hidden));
    result->commit_ns = yvex_core_monotonic_ns() - started;
    result->replayed_target_token_count = 0ull;
    result->completed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int speculation_promoted_target_result(yvex_runtime_speculation_context *context,
    unsigned long long committed_count,
    const yvex_runtime_transformer_result *extension,
    yvex_runtime_transformer_result *target, yvex_error *err)
{
    const yvex_runtime_session_view *view =
        yvex_runtime_session_view_get(context->session);
    yvex_graph_attention_state_summary state = {0};
    yvex_sha256 hash;
    unsigned long long hidden_values, feature_values;
    char normalized[YVEX_SPECULATION_IDENTITY_CAP];
    char features[YVEX_SPECULATION_IDENTITY_CAP];
    char execution[YVEX_SPECULATION_IDENTITY_CAP];
    if (!view || !view->attention_state_provider ||
        !view->attention_state_provider->summary ||
        view->attention_state_provider->summary(
            view->attention_state_provider->context, &state, err) != YVEX_OK ||
        !state.staged_batch_complete ||
        state.staged_next_position != context->pending_position + committed_count ||
        !yvex_core_u64_mul(committed_count, context->hidden_width,
                           &hidden_values) ||
        !yvex_core_u64_mul(committed_count,
                           context->policy.concatenated_feature_width,
                           &feature_values))
        return speculation_refuse(
            err, YVEX_ERR_STATE,
            "promoted target prefix is not a complete staged transaction");
    memset(target, 0, sizeof(*target));
    target->phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    target->token_start = target->position_before = context->pending_position;
    target->token_count = committed_count;
    target->committed_prefix = target->position_after = state.staged_next_position;
    target->generation_before =
        context->pending_verification_target.generation_before;
    target->generation_after = state.staged_generation;
    target->chunk_count = 1ull + (extension ? extension->chunk_count : 0ull);
    target->feature_layer_count = context->policy.target_feature_layer_count;
    target->feature_row_count = committed_count;
    yvex_runtime_identity_copy(
        target->input_identity,
        context->pending_verification_target.input_identity);
    yvex_runtime_identity_copy(target->persistent_state_digest,
                               state.staged_state_content_identity);
    if (!yvex_execution_f32_digest(
            "yvex.transformer.normalized-hidden.v1", context->target_hidden,
            hidden_values, normalized) ||
        !yvex_execution_f32_digest(
            "yvex.transformer.target-features.v1", context->target_features,
            feature_values, features))
        return speculation_refuse(
            err, YVEX_ERR_STATE,
            "promoted target prefix values could not be identity-bound");
    yvex_runtime_identity_copy(target->normalized_hidden_digest, normalized);
    yvex_runtime_identity_copy(target->feature_digest, features);
    target->normalized_hidden_host_available = 1;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(
            &hash, "yvex.runtime.speculation.promoted-target.v1") ||
        !yvex_sha256_update_text(
            &hash, context->pending_verification_target.execution_identity) ||
        (extension && !yvex_sha256_update_text(
                           &hash, extension->execution_identity)) ||
        !yvex_sha256_update_text(&hash, target->normalized_hidden_digest) ||
        !yvex_sha256_update_text(&hash, target->feature_digest) ||
        !yvex_sha256_update_text(&hash, target->persistent_state_digest) ||
        !yvex_sha256_update_u64(&hash, context->pending_position) ||
        !yvex_sha256_update_u64(&hash, committed_count) ||
        !speculation_hash_finish(&hash, execution))
        return speculation_refuse(
            err, YVEX_ERR_STATE,
            "promoted target execution identity could not be sealed");
    yvex_runtime_identity_copy(target->execution_identity, execution);
    target->completed = 1;
    return YVEX_OK;
}
int yvex_runtime_speculation_commit_prefix(
    yvex_runtime_speculation_context *context,
    unsigned long long committed_count, float *final_hidden,
    unsigned long long final_hidden_capacity,
    const yvex_runtime_commit_participant *publication,
    yvex_runtime_speculation_commit_result *result, yvex_error *err)
{
    yvex_runtime_transformer_result target = {0};
    yvex_runtime_transformer_result extension = {0};
    yvex_runtime_transformer_core_commit_result draft = {0};
    yvex_runtime_commit_participant participant = {
        .context = context, .prepare = speculation_rng_prepare,
        .publish = speculation_rng_publish, .cancel = speculation_rng_cancel};
    unsigned long long final_values, base_count;
    unsigned long long started, promotion_started, extension_started = 0ull;
    int extension_required, rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result || !context->cycle_pending || !committed_count ||
        !final_hidden ||
        !yvex_core_u64_mul(committed_count, context->hidden_width,
                           &final_values) ||
        final_values > SIZE_MAX / sizeof(*final_hidden) ||
        final_hidden_capacity < final_values ||
        (publication && (!publication->prepare || !publication->publish ||
                         !publication->cancel)) ||
        committed_count > context->pending_committed_count ||
        committed_count > context->policy.block_size + 2ull ||
        committed_count > context->options.context_capacity ||
        context->pending_position > context->options.context_capacity - committed_count ||
        ((!context->verification_staged ||
          !context->pending_verified_prefix_count ||
          committed_count > context->pending_verified_prefix_count + 1ull) &&
         !(committed_count == 1ull &&
           context->pending_committed_count == 1ull &&
           !context->verification_staged &&
           !context->pending_verified_prefix_count)))
        return speculation_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "committed prefix is not an admitted pending DSpark result");
    if (!context->verification_staged)
        return speculation_commit_target_step(
            context, final_hidden, publication, result, err);
    result->token_start = context->pending_position;
    result->token_count = committed_count;
    yvex_runtime_identity_copy(result->cycle_identity,
                               context->pending_cycle_identity);
    context->publication = publication;
    context->publication_prepared = 0;
    started = yvex_core_monotonic_ns();
    base_count = committed_count < context->pending_verified_prefix_count
                     ? committed_count
                     : context->pending_verified_prefix_count;
    extension_required = committed_count > base_count;
    result->verified_prefix_count = base_count;
    result->target_extension_count = (unsigned long long)extension_required;
    promotion_started = yvex_core_monotonic_ns();
    rc = yvex_runtime_session_select_attention_prefix(
        context->session, YVEX_TENSOR_SCOPE_GLOBAL, base_count,
        (unsigned long long)extension_required, &result->promotion_physical, err);
    result->promotion_ns = yvex_core_monotonic_ns() - promotion_started;
    if (rc == YVEX_OK)
        result->promoted_target_token_count = base_count;
    if (rc == YVEX_OK && extension_required) {
        unsigned long long hidden_offset = base_count * context->hidden_width;
        unsigned long long feature_offset =
            base_count * context->policy.concatenated_feature_width;
        extension_started = yvex_core_monotonic_ns();
        rc = speculation_transformer_execute(
            context, context->target_transformer,
            context->pending_tokens + base_count, 1ull,
            context->pending_position + base_count, 0, 0,
            YVEX_ATTENTION_TRANSACTION_STAGE,
            context->policy.target_feature_layers,
            context->policy.target_feature_layer_count,
            context->target_hidden + hidden_offset, NULL,
            context->target_features + feature_offset, base_count,
            &extension, err);
        result->target_extension_ns =
            yvex_core_monotonic_ns() - extension_started;
    }
    if (rc == YVEX_OK)
        rc = speculation_promoted_target_result(
            context, committed_count,
            extension_required ? &extension : NULL, &target, err);
    if (rc == YVEX_OK)
        rc = speculation_project_target_features(
            context, context->target_features, committed_count, &target, err);
    if (rc == YVEX_OK)
        rc = speculation_stage_draft_features(
            context, context->pending_position, committed_count, &draft, err);
    if (rc == YVEX_OK) {
        result->position_after = context->pending_position + committed_count;
        result->target_result = target;
        yvex_runtime_identity_copy(result->target_execution_identity,
                                   target.execution_identity);
        yvex_runtime_identity_copy(result->draft_execution_identity,
                                   draft.execution_identity);
        yvex_runtime_identity_copy(result->target_state_identity,
                                   target.persistent_state_digest);
        yvex_runtime_identity_copy(result->draft_state_identity,
                                   draft.persistent_state_digest);
        if (target.position_after != result->position_after ||
            draft.position_after != result->position_after ||
            !yvex_sha256_hex_valid(result->target_state_identity) ||
            !yvex_sha256_hex_valid(result->draft_state_identity) ||
            !speculation_commit_identity(context, result,
                                         result->commit_identity))
            rc = speculation_refuse(
                err, YVEX_ERR_STATE,
                "speculative staged prefix identity is incomplete");
    }
    rc = yvex_runtime_session_finish_coordinated(
        context->session, rc, &participant, err);
    if (rc != YVEX_OK) return rc;
    result->commit_ns = yvex_core_monotonic_ns() - started;
    /* The transformer result authenticates the complete committed prefix. Retain every matching
     * hidden row until the next anchor distribution is sealed; the last row alone breaks that
     * producer/digest contract on the second cycle. */
    memcpy(final_hidden, context->target_hidden,
           (size_t)final_values * sizeof(*final_hidden));
    result->replayed_target_token_count = 0ull;
    result->completed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
int yvex_runtime_speculation_cycle_abort(
    yvex_runtime_speculation_context *context, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context)
        return speculation_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "DSpark cycle owner is required");
    if (!context->cycle_pending) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (context->verification_staged) {
        rc = yvex_runtime_session_finish_scope(
            context->session, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_ATTENTION_TRANSACTION_ABORT, YVEX_OK, err);
        context->verification_staged = 0;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_transaction_abort(&context->target_rng, err);
    else
        (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
    if (context->draft_rng) {
        int abort_rc = yvex_runtime_sampling_transaction_abort(
            &context->draft_rng, rc == YVEX_OK ? err : NULL);
        if (rc == YVEX_OK) rc = abort_rc;
    }
    speculation_pending_clear(context);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}
int yvex_runtime_speculation_finish_terminal(
    yvex_runtime_speculation_context *context,
    const yvex_runtime_commit_participant *publication, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !context->cycle_pending ||
        !context->pending_committed_count || !publication ||
        context->verification_staged ||
        !publication->prepare || !publication->publish ||
        !publication->cancel)
        return speculation_refuse(
            err, YVEX_ERR_STATE,
            "terminal completion requires pending target and publication state");
    if (context->target_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->target_rng, err);
    if (rc == YVEX_OK && context->draft_rng)
        rc = yvex_runtime_sampling_transaction_prepare_commit(
            context->draft_rng, err);
    if (rc == YVEX_OK)
        rc = publication->prepare(publication->context, err);
    if (rc == YVEX_OK) {
        yvex_runtime_sampling_transaction_publish_commit(&context->target_rng);
        yvex_runtime_sampling_transaction_publish_commit(&context->draft_rng);
        publication->publish(publication->context);
        speculation_pending_clear(context);
        yvex_error_clear(err);
    } else {
        publication->cancel(publication->context);
        (void)yvex_runtime_sampling_transaction_abort(&context->target_rng, NULL);
        (void)yvex_runtime_sampling_transaction_abort(&context->draft_rng, NULL);
    }
    return rc;
}
int yvex_runtime_speculation_context_close(
    yvex_runtime_speculation_context **context, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->cycle_pending) {
        rc = yvex_runtime_speculation_cycle_abort(*context, err);
        if (rc != YVEX_OK) return rc;
    }
    if ((*context)->device_feature_normalized)
        rc = yvex_backend_tensor_release((*context)->device_backend,
                                         &(*context)->device_feature_normalized, err);
    if (rc == YVEX_OK && (*context)->device_adjusted_logits)
        rc = yvex_backend_tensor_release((*context)->device_backend,
                                         &(*context)->device_adjusted_logits, err);
    if (rc == YVEX_OK && (*context)->device_feature_projected)
        rc = yvex_backend_tensor_release((*context)->device_backend,
                                         &(*context)->device_feature_projected, err);
    if (rc == YVEX_OK && (*context)->device_feature_input)
        rc = yvex_backend_tensor_release((*context)->device_backend,
                                         &(*context)->device_feature_input, err);
    if (rc == YVEX_OK && (*context)->device_feature_norm)
        rc = yvex_backend_tensor_release((*context)->device_backend,
                                         &(*context)->device_feature_norm, err);
    if (rc != YVEX_OK) return rc;
    if ((*context)->verification_sampling)
        rc = yvex_runtime_sampling_context_close(&(*context)->verification_sampling, err);
    if (rc == YVEX_OK && (*context)->verification_logits)
        rc = yvex_runtime_logits_context_close(&(*context)->verification_logits, err);
    if (rc == YVEX_OK && (*context)->draft_sampling)
        rc = yvex_runtime_sampling_context_close(&(*context)->draft_sampling, err);
    if (rc == YVEX_OK && (*context)->draft_transformer)
        rc = yvex_runtime_transformer_context_close(
            &(*context)->draft_transformer, err);
    if (rc != YVEX_OK) return rc;
    yvex_core_free((*context)->draft_input_ids);
    yvex_core_free((*context)->markov_embedding_values);
    yvex_core_free((*context)->draft_probabilities);
    yvex_core_free((*context)->markov_bias);
    yvex_core_free((*context)->adjusted_logits);
    yvex_core_free((*context)->base_logits);
    yvex_core_free((*context)->target_hidden);
    yvex_core_free((*context)->draft_pre_normalized);
    yvex_core_free((*context)->draft_hidden);
    yvex_core_free((*context)->target_features);
    yvex_core_free((*context)->feature_norm_weights);
    yvex_core_free((*context)->feature_projected);
    memset(*context, 0, sizeof(**context));
    yvex_core_free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
