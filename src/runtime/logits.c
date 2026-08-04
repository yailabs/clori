/*
 * Make prefill and decode hidden rows directly consumable by the sampling owner.
 *
 * One complete canonical vocabulary row is published only after every value is finite.
 * Family-neutral runtime execution from typed normalized hidden state to raw F32 logits.
 */
#include <yvex/internal/logits.h>

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>

struct yvex_runtime_logits_plan {
    yvex_runtime_logits_plan_summary summary;
    const yvex_materialized_tensor_binding *binding;
};

struct yvex_runtime_logits_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    const yvex_runtime_session_view *session_view;
    yvex_runtime_logits_plan plan;
    yvex_runtime_logits_options options;
    const unsigned char *resident_head;
    unsigned long long resident_head_bytes;
    float *candidate;
    yvex_device_tensor *device_hidden, *device_logits;
    pthread_mutex_t mutex;
    unsigned long long execution_count;
    char shared_draft_plan_identity[YVEX_SHA256_HEX_CAP];
    int mutex_ready, busy, invalidated, shared_draft_plan_admitted;
};

/*
 * Admit and project logits readiness from the exact immutable output-head residency.
 *
 * Narrows output-head binding readiness to resident implementation truth.
 */
int yvex_runtime_logits_residency_admit(
    yvex_runtime_capabilities *capabilities,
    const yvex_runtime_residency_summary *residency)
{
    if (!capabilities || !residency ||
        (capabilities->output_head_binding_ready && !residency->output_head_complete))
        return 0;
    capabilities->output_head_binding_ready =
        capabilities->output_head_binding_ready && residency->output_head_complete;
    return 1;
}

void yvex_runtime_logits_capabilities_invalidate(yvex_runtime_capabilities *capabilities)
{
    if (!capabilities) return;
    capabilities->output_head_binding_ready = capabilities->output_head_projection_ready =
        capabilities->logits_cpu_ready = capabilities->logits_cuda_ready =
        capabilities->logits_prefill_ready = capabilities->logits_decode_ready =
        capabilities->logits_full_vocabulary_ready = capabilities->logits_hidden_contract_ready =
        capabilities->logits_partial_progress_ready = capabilities->logits_ready = 0;
}

static int logits_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.logits", reason);
    return status;
}

static int logits_hash_values(yvex_sha256 *hash, const float *values,
                              unsigned long long count)
{
    unsigned long long index;
    if (!hash || (!values && count)) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, bits)) return 0;
    }
    return 1;
}

static int logits_values_digest(const char *domain, const float *values,
                                unsigned long long count,
                                char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (output) output[0] = '\0';
    if (!domain || !values || !count || !output) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !logits_hash_values(&hash, values, count) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

/*
 * Derive the pointer-free output-head plan identity.
 *
 * Publishes only its canonical identity field.
 */
static int logits_plan_identity(yvex_runtime_logits_plan_summary *summary)
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
        !yvex_sha256_update_u64(&hash, (unsigned int)summary->separate_output_head) ||
        !yvex_sha256_update_u64(&hash, (unsigned int)summary->output_head_bias) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, summary->output_head_plan_identity);
    return 1;
}

static int logits_plan_build(yvex_runtime_logits_plan *plan,
                             const yvex_runtime_model_view *view,
                             const yvex_transformer_plan *transformer_plan,
                             yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime;
    const yvex_transformer_plan_summary *transformer;
    const yvex_runtime_tensor_binding *row;
    const yvex_runtime_tensor_binding *embedding;
    const yvex_materialized_tensor_binding *binding;
    const yvex_gguf_qtype_geometry *geometry;
    const yvex_quant_numeric_capability *numeric;
    yvex_logits_family_policy policy;
    unsigned long long blocks, row_bytes, encoded_bytes;
    if (!plan || !view || !view->adapter || !view->adapter->logits_policy ||
        !view->adapter->logits_policy(&policy) ||
        policy.schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        !policy.separate_output_head || policy.tied_output_head ||
        policy.output_head_bias)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "family output-head policy is unavailable or incompatible");
    runtime = yvex_runtime_descriptor_summary_get(view->descriptor);
    transformer = yvex_transformer_plan_summary_get(transformer_plan);
    row = yvex_runtime_descriptor_find_role(
        view->descriptor, YVEX_TENSOR_ROLE_OUTPUT_HEAD, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX);
    embedding = yvex_runtime_descriptor_find_role(
        view->descriptor, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX);
    binding = row ? yvex_materialization_session_tensor_at(
                        view->materialization, row->tensor_id) : NULL;
    geometry = binding ? yvex_gguf_qtype_geometry_find(binding->qtype) : NULL;
    numeric = binding ? yvex_quant_numeric_capability_at(binding->qtype) : NULL;
    if (!runtime || !transformer || !row || !binding || !embedding ||
        binding->tensor_id == embedding->tensor_id ||
        binding->role != YVEX_TENSOR_ROLE_OUTPUT_HEAD ||
        binding->row_width != transformer->hidden_width ||
        binding->row_count != transformer->vocabulary_size ||
        !geometry || !geometry->block_size || !geometry->bytes_per_block ||
        binding->row_width % geometry->block_size ||
        !numeric || !numeric->dedicated_cpu_compute_available ||
        !numeric->dedicated_cuda_compute_available)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "exact separate output-head binding or qtype compute is unavailable");
    blocks = binding->row_width / geometry->block_size;
    if (!yvex_core_u64_mul(blocks, geometry->bytes_per_block, &row_bytes) ||
        !yvex_core_u64_mul(row_bytes, binding->row_count, &encoded_bytes) ||
        encoded_bytes != binding->encoded_bytes)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "output-head encoded geometry is inconsistent");
    memset(plan, 0, sizeof(*plan));
    plan->binding = binding;
    plan->summary.schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    plan->summary.family_adapter_id = view->adapter->adapter_id;
    plan->summary.family_adapter_version = view->adapter->adapter_version;
    plan->summary.output_head_tensor_id = binding->tensor_id;
    plan->summary.role = binding->role;
    plan->summary.qtype = binding->qtype;
    plan->summary.row_width = binding->row_width;
    plan->summary.row_count = binding->row_count;
    plan->summary.row_bytes = row_bytes;
    plan->summary.encoded_bytes = encoded_bytes;
    plan->summary.vocabulary_size = transformer->vocabulary_size;
    plan->summary.hidden_width = transformer->hidden_width;
    plan->summary.separate_output_head = policy.separate_output_head;
    plan->summary.output_head_bias = policy.output_head_bias;
    yvex_runtime_identity_copy(plan->summary.artifact_identity,
                               view->binding->artifact_identity);
    yvex_runtime_identity_copy(plan->summary.materialization_identity,
                               view->binding->materialization_identity);
    yvex_runtime_identity_copy(plan->summary.logical_model_identity,
                               runtime->logical_model_identity);
    yvex_runtime_identity_copy(plan->summary.runtime_numeric_identity,
                               runtime->runtime_numeric_identity);
    yvex_runtime_identity_copy(plan->summary.runtime_descriptor_identity,
                               runtime->runtime_descriptor_identity);
    yvex_runtime_identity_copy(plan->summary.transformer_plan_identity,
                               transformer->transformer_plan_identity);
    if (!logits_plan_identity(&plan->summary))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "output-head plan identity derivation failed");
    return YVEX_OK;
}

static int logits_device_open(yvex_runtime_logits_context *context,
                              yvex_device_tensor **out, const char *name,
                              unsigned long long elements, yvex_error *err)
{
    yvex_backend_tensor_desc descriptor = {0};
    unsigned long long bytes;
    if (!yvex_core_u64_mul(elements, sizeof(float), &bytes))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "logits device tensor extent overflowed");
    descriptor.name = name;
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = elements;
    descriptor.bytes = bytes;
    return yvex_backend_tensor_alloc(context->session_view->backend,
                                     &descriptor, out, err);
}

/*
 * Open one reusable logits context over borrowed model/session owners.
 *
 * Borrows resident head and allocates stable local CPU/CUDA workspaces.
 */
int yvex_runtime_logits_context_open(
    yvex_runtime_logits_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_transformer_plan *transformer_plan,
    const yvex_runtime_logits_options *options, yvex_error *err)
{
    yvex_runtime_logits_context *context = NULL;
    yvex_runtime_model_summary model_summary;
    yvex_runtime_residency_summary residency;
    unsigned long long candidate_bytes, device_bytes;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !transformer_plan || !options ||
        !options->maximum_rows ||
        options->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        (options->device_greedy_selection &&
         (!options->execution_profile ||
          options->execution_profile->backend != YVEX_BACKEND_KIND_CUDA ||
          !yvex_sha256_hex_valid(options->execution_profile->identity))))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "logits requires model, session, transformer plan, and row budget");
    context = (yvex_runtime_logits_context *)calloc(1u, sizeof(*context));
    if (!context) return logits_refuse(err, YVEX_ERR_NOMEM,
                                       "logits context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    if (!context->model_view || !context->session_view ||
        context->session_view->model != model ||
        yvex_runtime_model_summary_copy(model, &model_summary, err) != YVEX_OK ||
        !model_summary.sealed || !model_summary.valid) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "logits model/session pairing is invalid");
        goto failure;
    }
    rc = logits_plan_build(&context->plan, context->model_view,
                           transformer_plan, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_runtime_residency_snapshot(
        context->model_view->residency, &residency, NULL, NULL, err);
    if (rc != YVEX_OK || !residency.output_head_complete ||
        !yvex_sha256_hex_valid(residency.output_head_residency_identity)) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "sealed output-head residency is unavailable");
        goto failure;
    }
    rc = yvex_runtime_residency_binding_view(
        context->model_view->residency, context->plan.binding,
        &context->resident_head, &context->resident_head_bytes, err);
    if (rc != YVEX_OK ||
        context->resident_head_bytes != context->plan.summary.encoded_bytes)
        goto failure;
    if (!yvex_core_u64_mul(context->plan.summary.vocabulary_size,
                           sizeof(float), &candidate_bytes) ||
        candidate_bytes > SIZE_MAX ||
        (options->maximum_host_bytes && candidate_bytes + sizeof(*context) >
                                                options->maximum_host_bytes)) {
        rc = logits_refuse(err, YVEX_ERR_NOMEM,
                           "logits host workspace exceeds its budget");
        goto failure;
    }
    context->candidate = (float *)malloc((size_t)candidate_bytes);
    if (!context->candidate) {
        rc = logits_refuse(err, YVEX_ERR_NOMEM,
                           "logits candidate allocation failed");
        goto failure;
    }
    if (yvex_backend_kind_of(context->session_view->backend) ==
        YVEX_BACKEND_KIND_CUDA) {
        if (!yvex_core_u64_add(candidate_bytes,
                               context->plan.summary.hidden_width * sizeof(float),
                               &device_bytes) ||
            (options->maximum_device_bytes &&
             residency.device_resident_bytes + device_bytes >
                 options->maximum_device_bytes)) {
            rc = logits_refuse(err, YVEX_ERR_NOMEM,
                               "logits CUDA workspace exceeds its budget");
            goto failure;
        }
        rc = logits_device_open(context, &context->device_hidden,
                                "logits.hidden", context->plan.summary.hidden_width, err);
        if (rc == YVEX_OK)
            rc = logits_device_open(context, &context->device_logits,
                                    "logits.output", context->plan.summary.vocabulary_size, err);
        if (rc != YVEX_OK) goto failure;
    }
    if (pthread_mutex_init(&context->mutex, NULL) != 0) {
        rc = logits_refuse(err, YVEX_ERR_STATE,
                           "logits context synchronization initialization failed");
        goto failure;
    }
    context->mutex_ready = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
failure:
    (void)yvex_runtime_logits_context_close(&context, NULL);
    return rc;
}

const yvex_runtime_logits_plan_summary *yvex_runtime_logits_plan_summary_get(
    const yvex_runtime_logits_context *context)
{
    return context ? &context->plan.summary : NULL;
}

int yvex_runtime_logits_admit_shared_draft_plan(
    yvex_runtime_logits_context *context,
    const yvex_transformer_plan *draft_plan, yvex_error *err)
{
    const yvex_transformer_plan_summary *draft =
        yvex_transformer_plan_summary_get(draft_plan);
    const yvex_runtime_descriptor_summary *runtime =
        context && context->model_view
            ? yvex_runtime_descriptor_summary_get(context->model_view->descriptor) : NULL;
    yvex_speculation_family_policy policy;
    if (!context || !draft || !runtime || !context->model_view || !context->model_view->adapter ||
        !context->model_view->adapter->speculation_policy ||
        !context->model_view->adapter->speculation_policy(runtime, &policy) ||
        policy.schema_version != YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1 ||
        !policy.shares_output_head || draft->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
        draft->hidden_width != context->plan.summary.hidden_width ||
        draft->vocabulary_size != context->plan.summary.vocabulary_size ||
        strcmp(draft->logical_model_identity,
               context->plan.summary.logical_model_identity) != 0 ||
        strcmp(draft->runtime_numeric_identity,
               context->plan.summary.runtime_numeric_identity) != 0 ||
        strcmp(draft->runtime_descriptor_identity,
               context->plan.summary.runtime_descriptor_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "draft plan cannot consume the resident shared output head");
    yvex_runtime_identity_copy(context->shared_draft_plan_identity,
                               draft->transformer_plan_identity);
    context->shared_draft_plan_admitted = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int logits_source_identity(yvex_runtime_logits_source *source)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!source) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.source.v2") ||
        !yvex_sha256_update_u64(&hash, source->schema_version) ||
        !yvex_sha256_update_u64(&hash, source->source_phase) ||
        !yvex_sha256_update_u64(&hash, source->source_position) ||
        !yvex_sha256_update_u64(&hash, source->row_count) ||
        !yvex_sha256_update_u64(&hash, source->hidden_width) ||
        !yvex_sha256_update_u64(&hash, source->host_values_available) ||
        !yvex_sha256_update_u64(&hash, source->device_values_available) ||
        !yvex_sha256_update_text(&hash, source->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, source->runtime_binding_identity) ||
        !yvex_sha256_update_text(&hash, source->transformer_plan_identity) ||
        !yvex_sha256_update_text(&hash, source->transformer_execution_identity) ||
        !yvex_sha256_update_text(&hash, source->normalized_hidden_digest) ||
        (source->device_values_available &&
         (!yvex_sha256_update_text(
              &hash, source->device_hidden.execution_profile_identity) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.model_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.session_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.state_generation) ||
          !yvex_sha256_update_u64(&hash, source->device_hidden.element_offset))) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, source->source_identity);
    return 1;
}

/*
 * Initialize one source from exact borrowed model and plan identities.
 *
 * Live context, phase, position, producer identity, and finite hidden row. Publishes one
 * identity-bound borrowed source view.
 */
static int logits_source_begin(const yvex_runtime_logits_context *context,
                               yvex_runtime_logits_source *source,
                               yvex_logits_source_phase phase,
                               unsigned long long position,
                               const float *hidden,
                               const yvex_execution_device_view *device_hidden,
                               const char *producer_hidden_digest,
                               const char *plan_identity,
                               const char *producer_identity,
                               yvex_error *err)
{
    yvex_runtime_model_summary model;
    if (!context || !source || (!hidden == !device_hidden) ||
        !producer_hidden_digest || !producer_identity ||
        !yvex_sha256_hex_valid(producer_hidden_digest) ||
        !yvex_sha256_hex_valid(producer_identity) ||
        (device_hidden &&
         yvex_execution_device_view_validate(device_hidden, err) != YVEX_OK) ||
        yvex_runtime_model_summary_copy(context->model, &model, err) != YVEX_OK)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "logits producer source facts are invalid");
    memset(source, 0, sizeof(*source));
    source->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    source->source_phase = phase;
    source->source_position = position;
    source->row_count = 1ull;
    source->hidden_width = context->plan.summary.hidden_width;
    source->normalized_hidden = hidden;
    source->host_values_available = hidden != NULL;
    source->device_values_available = device_hidden != NULL;
    if (device_hidden) source->device_hidden = *device_hidden;
    yvex_runtime_identity_copy(source->runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(source->runtime_binding_identity,
                               context->model_view->binding->identity);
    yvex_runtime_identity_copy(source->transformer_plan_identity, plan_identity);
    yvex_runtime_identity_copy(source->transformer_execution_identity,
                               producer_identity);
    if (hidden &&
        !logits_values_digest("yvex.transformer.normalized-hidden.v1", hidden,
                              source->hidden_width,
                              source->normalized_hidden_digest))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden host row could not be sealed");
    if (device_hidden)
        yvex_runtime_identity_copy(source->normalized_hidden_digest,
                                   producer_hidden_digest);
    if (!logits_source_identity(source) ||
        !yvex_sha256_hex_valid(source->source_identity))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity could not be sealed");
    return YVEX_OK;
}

int yvex_runtime_logits_source_from_transformer(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err)
{
    yvex_execution_device_view device_row;
    char complete_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long complete_values, row_offset, first_device_row;
    if (!context || !source || !producer || !producer->completed ||
        producer->phase != YVEX_TRANSFORMER_PHASE_PREFILL ||
        !producer->token_count || row_ordinal >= producer->token_count ||
        !yvex_core_u64_mul(producer->token_count,
                           context->plan.summary.hidden_width, &complete_values) ||
        !yvex_core_u64_mul(row_ordinal, context->plan.summary.hidden_width,
                           &row_offset))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "Transformer normalized-hidden publication is incompatible");
    if (producer->normalized_hidden_host_available) {
        if (!normalized_hidden || hidden_capacity < complete_values ||
            !logits_values_digest(
                "yvex.transformer.normalized-hidden.v1", normalized_hidden,
                complete_values, complete_digest) ||
            strcmp(complete_digest, producer->normalized_hidden_digest) != 0)
            return logits_refuse(
                err, YVEX_ERR_FORMAT,
                "Transformer host hidden publication is incompatible");
        return logits_source_begin(
            context, source, YVEX_LOGITS_SOURCE_PREFILL,
            producer->token_start + row_ordinal, normalized_hidden + row_offset,
            NULL, producer->normalized_hidden_digest,
            context->plan.summary.transformer_plan_identity,
            producer->execution_identity, err);
    }
    if (!producer->normalized_hidden_device_available || normalized_hidden ||
        hidden_capacity || producer->device_hidden.rows > producer->token_count)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "Transformer device hidden publication is incompatible");
    first_device_row = producer->token_count - producer->device_hidden.rows;
    if (row_ordinal < first_device_row) return logits_refuse(
        err, YVEX_ERR_BOUNDS,
        "requested hidden row is no longer resident in the device publication");
    device_row = producer->device_hidden;
    if (!yvex_core_u64_mul(row_ordinal - first_device_row,
                           context->plan.summary.hidden_width, &row_offset) ||
        !yvex_core_u64_add(device_row.element_offset, row_offset,
                           &device_row.element_offset))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "device hidden row offset overflowed");
    device_row.rows = 1ull;
    if (yvex_execution_device_view_validate(&device_row, err) != YVEX_OK)
        return yvex_error_code(err);
    return logits_source_begin(
        context, source, YVEX_LOGITS_SOURCE_PREFILL,
        producer->token_start + row_ordinal, NULL, &device_row,
        producer->normalized_hidden_digest,
        context->plan.summary.transformer_plan_identity,
        producer->execution_identity, err);
}

int yvex_runtime_logits_source_from_decode(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_runtime_decode_step_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    yvex_error *err)
{
    char digest[YVEX_SHA256_HEX_CAP];
    if (!context || !source || !producer || !producer->completed ||
        producer->position_after != producer->position_before + 1ull)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "decode normalized-hidden publication is incompatible");
    if (producer->normalized_hidden_host_available) {
        if (!normalized_hidden ||
            hidden_capacity < context->plan.summary.hidden_width ||
            !logits_values_digest(
                "yvex.transformer.normalized-hidden.v1", normalized_hidden,
                context->plan.summary.hidden_width, digest) ||
            strcmp(digest, producer->normalized_hidden_digest) != 0)
            return logits_refuse(
                err, YVEX_ERR_FORMAT,
                "decode host hidden publication is incompatible");
        return logits_source_begin(
            context, source, YVEX_LOGITS_SOURCE_DECODE,
            producer->position_before, normalized_hidden, NULL,
            producer->normalized_hidden_digest,
            context->plan.summary.transformer_plan_identity,
            producer->transformer_execution_identity, err);
    }
    if (!producer->normalized_hidden_device_available || normalized_hidden ||
        hidden_capacity || producer->device_hidden.rows != 1ull ||
        yvex_execution_device_view_validate(&producer->device_hidden, err) != YVEX_OK)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "decode device hidden publication is incompatible");
    return logits_source_begin(
        context, source, YVEX_LOGITS_SOURCE_DECODE,
        producer->position_before, NULL, &producer->device_hidden,
        producer->normalized_hidden_digest,
        context->plan.summary.transformer_plan_identity,
        producer->transformer_execution_identity, err);
}

int yvex_runtime_logits_source_from_draft(
    const yvex_runtime_logits_context *context,
    yvex_runtime_logits_source *source,
    const yvex_transformer_plan *draft_plan,
    const yvex_runtime_transformer_result *producer,
    const float *normalized_hidden, unsigned long long hidden_capacity,
    unsigned long long row_ordinal, yvex_error *err)
{
    const yvex_transformer_plan_summary *draft =
        yvex_transformer_plan_summary_get(draft_plan);
    char complete_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long complete_values, row_offset;
    if (!context || !source || !draft || !producer || !producer->completed ||
        producer->phase != YVEX_TRANSFORMER_PHASE_PREFILL ||
        !context->shared_draft_plan_admitted ||
        strcmp(draft->transformer_plan_identity,
               context->shared_draft_plan_identity) != 0 ||
        !producer->token_count || row_ordinal >= producer->token_count ||
        !yvex_core_u64_mul(producer->token_count, draft->hidden_width,
                           &complete_values) ||
        hidden_capacity < complete_values || !normalized_hidden ||
        !logits_values_digest("yvex.transformer.normalized-hidden.v1",
                              normalized_hidden, complete_values,
                              complete_digest) ||
        strcmp(complete_digest, producer->normalized_hidden_digest) != 0 ||
        !yvex_core_u64_mul(row_ordinal, draft->hidden_width, &row_offset))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "draft normalized-hidden publication is incompatible");
    return logits_source_begin(
        context, source, YVEX_LOGITS_SOURCE_DRAFT,
        producer->token_start + row_ordinal, normalized_hidden + row_offset,
        NULL, producer->normalized_hidden_digest,
        draft->transformer_plan_identity, producer->execution_identity, err);
}

/*
 * Validate one normalized-hidden source against exact producing identities.
 *
 * Refuses stale model/plan/producer identity, geometry, or payload digest.
 */
static int logits_source_validate(const yvex_runtime_logits_context *context,
                                  const yvex_runtime_logits_source *source,
                                  yvex_error *err)
{
    yvex_runtime_model_summary model;
    yvex_runtime_logits_source canonical;
    char digest[YVEX_SHA256_HEX_CAP];
    if (!context || !source || source->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        source->source_phase > YVEX_LOGITS_SOURCE_DRAFT || source->row_count != 1ull ||
        source->hidden_width != context->plan.summary.hidden_width ||
        source->host_values_available + source->device_values_available != 1 ||
        source->host_values_available != (source->normalized_hidden != NULL) ||
        yvex_runtime_model_summary_copy(context->model, &model, err) != YVEX_OK ||
        strcmp(source->runtime_model_identity, model.runtime_model_identity) != 0 ||
        strcmp(source->runtime_binding_identity,
               context->model_view->binding->identity) != 0 ||
        ((strcmp(source->transformer_plan_identity,
                 context->plan.summary.transformer_plan_identity) != 0) &&
         (!context->shared_draft_plan_admitted ||
          source->source_phase != YVEX_LOGITS_SOURCE_DRAFT ||
          strcmp(source->transformer_plan_identity,
                 context->shared_draft_plan_identity) != 0)) ||
        !yvex_sha256_hex_valid(source->transformer_execution_identity))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity or geometry is stale");
    if (source->host_values_available &&
        (!logits_values_digest("yvex.transformer.normalized-hidden.v1",
                               source->normalized_hidden, source->hidden_width,
                               digest) ||
         strcmp(digest, source->normalized_hidden_digest) != 0))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden host row digest is stale");
    if (source->device_values_available &&
        (source->device_hidden.kind != YVEX_EXECUTION_DEVICE_HIDDEN ||
         source->device_hidden.backend != context->session_view->backend ||
         source->device_hidden.rows != 1ull ||
         source->device_hidden.columns != source->hidden_width ||
         strcmp(source->device_hidden.runtime_model_identity,
                source->runtime_model_identity) != 0 ||
         !context->options.execution_profile ||
         strcmp(source->device_hidden.execution_profile_identity,
                context->options.execution_profile->identity) != 0 ||
         yvex_execution_device_view_validate(&source->device_hidden, err) != YVEX_OK))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden device row is stale");
    canonical = *source;
    canonical.source_identity[0] = '\0';
    if (!logits_source_identity(&canonical) ||
        strcmp(canonical.source_identity, source->source_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "normalized-hidden source identity is not canonical");
    return YVEX_OK;
}

static int logits_project_cpu(yvex_runtime_logits_context *context,
                              const float *hidden, yvex_error *err)
{
    unsigned long long row;
    for (row = 0ull; row < context->plan.summary.row_count; ++row) {
        yvex_quant_failure failure;
        if (context->options.cancel_requested &&
            context->options.cancel_requested(context->options.cancel_context))
            return logits_refuse(err, YVEX_ERR_CANCELLED,
                                 "logits CPU projection was cancelled");
        memset(&failure, 0, sizeof(failure));
        if (yvex_quant_cpu_dot(
                context->plan.summary.qtype,
                context->resident_head + row * context->plan.summary.row_bytes,
                (size_t)context->plan.summary.row_bytes, hidden,
                context->plan.summary.hidden_width, &context->candidate[row],
                &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        if (!isfinite(context->candidate[row]))
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "CPU output-head projection produced non-finite logits");
    }
    return YVEX_OK;
}

/*
 * Execute one full resident output-head projection on CUDA without CPU fallback.
 *
 * CUDA context buffers, encoded resident head, and admitted hidden row. Bounded H2D, device
 * matvec, synchronization, and candidate D2H.
 */
static int logits_project_cuda(yvex_runtime_logits_context *context,
                               const yvex_runtime_logits_source *source,
                               yvex_runtime_logits_row_result *result,
                               yvex_error *err)
{
    unsigned long long hidden_bytes, logits_bytes;
    yvex_backend_cuda_operation_facts facts;
    yvex_device_tensor borrowed_hidden;
    const yvex_device_tensor *device_hidden = context->device_hidden;
    int rc;
    if (!context->device_hidden || !context->device_logits ||
        !yvex_core_u64_mul(context->plan.summary.hidden_width, sizeof(float),
                           &hidden_bytes) ||
        !yvex_core_u64_mul(context->plan.summary.vocabulary_size, sizeof(float),
                           &logits_bytes))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "stable logits CUDA buffers are unavailable");
    if (source->device_values_available) {
        if (!yvex_backend_tensor_f32_subview(
                source->device_hidden.tensor,
                source->device_hidden.element_offset,
                context->plan.summary.hidden_width, &borrowed_hidden))
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "device hidden row cannot be borrowed");
        device_hidden = &borrowed_hidden;
        rc = YVEX_OK;
    } else {
        rc = yvex_backend_tensor_write(
            context->session_view->backend, context->device_hidden,
            source->normalized_hidden, hidden_bytes, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_encoded_matvec(
            context->session_view->backend, context->resident_head,
            context->resident_head_bytes, context->plan.summary.qtype,
            context->plan.summary.row_count, context->plan.summary.row_width,
            context->plan.summary.row_bytes, device_hidden,
            context->device_logits, &facts, err);
    if (rc == YVEX_OK && !context->options.device_greedy_selection)
        rc = yvex_backend_tensor_read(context->session_view->backend,
                                      context->device_logits,
                                      context->candidate, logits_bytes, err);
    if (rc == YVEX_OK) {
        result->h2d_bytes = source->device_values_available ? 0ull : hidden_bytes;
        result->d2h_bytes = facts.d2h_bytes +
                            (context->options.device_greedy_selection ? 0ull : logits_bytes);
        result->kernel_launches = facts.kernel_launches;
        result->device_synchronizations = facts.device_synchronizations +
                                          !source->device_values_available +
                                          !context->options.device_greedy_selection;
    }
    return rc;
}

/*
 * Seal complete row evidence after all vocabulary values are finite.
 *
 * Any non-finite or identity failure leaves the row incomplete.
 */
static int logits_device_view_build(
    yvex_runtime_logits_context *context,
    const yvex_runtime_residency_summary *residency,
    yvex_execution_device_view *view, yvex_error *err)
{
    const yvex_attention_state_provider *provider =
        context ? context->session_view->attention_state_provider : NULL;
    if (!context || !residency || !residency->generation)
        return logits_refuse(err, YVEX_ERR_STATE,
                             "device logits generations are unavailable");
    return yvex_runtime_device_view_bind(
        view, YVEX_EXECUTION_DEVICE_LOGITS, context->model, context->session,
        provider, context->options.execution_profile, context->device_logits,
        0ull, 1ull, context->plan.summary.vocabulary_size, err);
}

static int logits_row_identity_build(yvex_runtime_logits_row_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!result) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.row.v2") ||
        !yvex_sha256_update_u64(&hash, result->source_phase) ||
        !yvex_sha256_update_u64(&hash, result->source_position) ||
        !yvex_sha256_update_u64(&hash, result->vocabulary_size) ||
        !yvex_sha256_update_u64(&hash, result->host_values_available) ||
        !yvex_sha256_update_u64(&hash, result->device_values_available) ||
        !yvex_sha256_update_u64(&hash, result->evidence_profile) ||
        !yvex_sha256_update_text(&hash, result->source_hidden_digest) ||
        !yvex_sha256_update_text(&hash, result->output_head_plan_identity) ||
        !yvex_sha256_update_text(&hash, result->output_head_residency_identity) ||
        !yvex_sha256_update_text(&hash, result->backend_execution_identity))
        return 0;
    if (result->host_values_available &&
        !yvex_sha256_update_text(&hash, result->raw_logits_digest))
        return 0;
    if (result->device_values_available &&
        (!yvex_sha256_update_text(
             &hash, result->device_logits.execution_profile_identity) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.model_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.session_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.state_generation) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.element_offset) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.rows) ||
         !yvex_sha256_update_u64(&hash, result->device_logits.columns)))
        return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, result->logits_row_identity);
    return 1;
}

static int logits_row_finish(yvex_runtime_logits_context *context,
                             const yvex_runtime_logits_source *source,
                             yvex_backend_kind backend,
                             yvex_runtime_logits_row_result *result,
                             yvex_error *err)
{
    yvex_runtime_residency_summary residency;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, scan_bytes;
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    if (yvex_runtime_residency_snapshot(context->model_view->residency,
                                        &residency, NULL, NULL, err) != YVEX_OK)
        return yvex_error_code(err);
    result->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    result->evidence_profile = context->options.evidence_profile;
    result->source_phase = source->source_phase;
    result->source_position = source->source_position;
    result->vocabulary_size = context->plan.summary.vocabulary_size;
    result->hidden_width = context->plan.summary.hidden_width;
    result->logits_count = context->plan.summary.vocabulary_size;
    yvex_runtime_identity_copy(result->source_hidden_digest,
                               source->normalized_hidden_digest);
    yvex_runtime_identity_copy(result->output_head_plan_identity,
                               context->plan.summary.output_head_plan_identity);
    yvex_runtime_identity_copy(result->output_head_residency_identity,
                               residency.output_head_residency_identity);
    if (context->options.device_greedy_selection) {
        result->device_values_available = 1;
        if (logits_device_view_build(context, &residency,
                                     &result->device_logits, err) != YVEX_OK)
            return yvex_error_code(err);
    } else {
        for (index = 0ull; index < result->vocabulary_size; ++index) {
            float value = context->candidate[index];
            if (!isfinite(value))
                return logits_refuse(
                    err, YVEX_ERR_FORMAT,
                    "output-head projection produced non-finite logits");
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
        if (!yvex_core_u64_mul(result->vocabulary_size, sizeof(float),
                               &scan_bytes) ||
            !logits_values_digest("yvex.runtime.raw-logits.v1",
                                  context->candidate, result->logits_count,
                                  result->raw_logits_digest))
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "raw logits evidence derivation failed");
        result->host_values_available = result->finite_count_available =
            result->range_available = result->raw_digest_available = 1;
        result->finite_count = result->logits_count;
        result->minimum_logit = minimum;
        result->maximum_logit = maximum;
        result->full_array_host_scan_bytes = scan_bytes;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.backend-evidence.v2") ||
        !yvex_sha256_update_u64(&hash, backend) ||
        !yvex_sha256_update_u64(&hash, result->h2d_bytes) ||
        !yvex_sha256_update_u64(&hash, result->d2h_bytes) ||
        !yvex_sha256_update_u64(&hash, result->kernel_launches) ||
        !yvex_sha256_update_u64(&hash, result->host_values_available) ||
        !yvex_sha256_update_u64(&hash, result->device_values_available) ||
        !yvex_sha256_update_u64(&hash, result->evidence_profile) ||
        (result->host_values_available &&
         !yvex_sha256_update_text(&hash, result->raw_logits_digest)) ||
        (result->device_values_available &&
         !yvex_sha256_update_text(
             &hash, result->device_logits.execution_profile_identity)) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits backend evidence identity derivation failed");
    yvex_sha256_hex(digest, result->backend_execution_identity);
    if (!logits_row_identity_build(result))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits row identity derivation failed");
    result->completed = 1;
    return YVEX_OK;
}

/*
 * Independently re-admit one published complete logits row for a downstream owner.
 *
 * Stale geometry, payload, plan, or field-wise row identity refuses before consumption.
 */
int yvex_runtime_logits_row_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *result, yvex_error *err)
{
    yvex_runtime_logits_row_result canonical;
    char raw_digest[YVEX_SHA256_HEX_CAP];
    if (!plan || !result || !result->completed ||
        result->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        result->source_phase > YVEX_LOGITS_SOURCE_DRAFT ||
        result->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC ||
        !plan->vocabulary_size || result->vocabulary_size != plan->vocabulary_size ||
        result->logits_count != plan->vocabulary_size ||
        result->host_values_available == result->device_values_available ||
        strcmp(result->output_head_plan_identity,
               plan->output_head_plan_identity) != 0 ||
        !yvex_sha256_hex_valid(result->source_hidden_digest) ||
        !yvex_sha256_hex_valid(result->output_head_residency_identity) ||
        !yvex_sha256_hex_valid(result->backend_execution_identity))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "complete logits row geometry or payload is stale");
    if (result->host_values_available &&
        (!logits || logits_capacity < result->logits_count ||
         !result->finite_count_available || !result->range_available ||
         !result->raw_digest_available ||
         result->finite_count != result->logits_count ||
         !logits_values_digest("yvex.runtime.raw-logits.v1", logits,
                               result->logits_count, raw_digest) ||
         strcmp(raw_digest, result->raw_logits_digest) != 0))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "host logits evidence is stale");
    if (result->device_values_available &&
        (result->finite_count_available || result->range_available ||
         result->raw_digest_available || result->finite_count ||
         yvex_execution_device_view_validate(&result->device_logits, err) !=
             YVEX_OK))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "device logits view is stale");
    canonical = *result;
    canonical.logits_row_identity[0] = '\0';
    if (!logits_row_identity_build(&canonical) ||
        strcmp(canonical.logits_row_identity, result->logits_row_identity) != 0)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "logits row identity is not canonical");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_logits_additive_adjust(
    const yvex_runtime_logits_context *context, const float *base_logits,
    unsigned long long base_capacity,
    const yvex_runtime_logits_row_result *base_result,
    const float *additive_logits, unsigned long long additive_capacity,
    float *adjusted_logits, unsigned long long adjusted_capacity,
    yvex_runtime_logits_row_result *result, yvex_error *err)
{
    const yvex_runtime_logits_plan_summary *plan =
        context ? &context->plan.summary : NULL;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char additive_digest[YVEX_SHA256_HEX_CAP];
    unsigned long long index;
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    if (result) memset(result, 0, sizeof(*result));
    if (!plan || !base_logits || !base_result || !additive_logits ||
        !adjusted_logits || !result || base_result->source_phase != YVEX_LOGITS_SOURCE_DRAFT ||
        additive_capacity < plan->vocabulary_size ||
        adjusted_capacity < plan->vocabulary_size ||
        yvex_runtime_logits_row_validate(plan, base_logits, base_capacity,
                                         base_result, err) != YVEX_OK)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "draft additive logits require one admitted base row");
    for (index = 0ull; index < plan->vocabulary_size; ++index) {
        double value = (double)base_logits[index] + additive_logits[index];
        if (!isfinite(additive_logits[index]) || !isfinite(value) ||
            value < -FLT_MAX || value > FLT_MAX)
            return logits_refuse(err, YVEX_ERR_FORMAT,
                                 "draft additive logits produced a non-finite value");
        adjusted_logits[index] = (float)value;
        if (adjusted_logits[index] < minimum) minimum = adjusted_logits[index];
        if (adjusted_logits[index] > maximum) maximum = adjusted_logits[index];
    }
    *result = *base_result;
    result->minimum_logit = minimum;
    result->maximum_logit = maximum;
    result->h2d_bytes = result->d2h_bytes = result->kernel_launches = 0ull;
    result->device_synchronizations = 0ull;
    if (!yvex_core_u64_mul(plan->vocabulary_size, sizeof(float), &index) ||
        !yvex_core_u64_add(result->full_array_host_scan_bytes, index,
                           &result->full_array_host_scan_bytes))
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "draft adjusted logits evidence extent overflowed");
    if (!logits_values_digest("yvex.runtime.draft-logits-bias.v1", additive_logits,
                              plan->vocabulary_size, additive_digest) ||
        !logits_values_digest("yvex.runtime.raw-logits.v1", adjusted_logits,
                              plan->vocabulary_size, result->raw_logits_digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted logits digest derivation failed");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.additive.v1") ||
        !yvex_sha256_update_text(&hash, base_result->logits_row_identity) ||
        !yvex_sha256_update_text(&hash, additive_digest) ||
        !yvex_sha256_update_text(&hash, result->raw_logits_digest) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted backend identity derivation failed");
    yvex_sha256_hex(digest, result->backend_execution_identity);
    result->logits_row_identity[0] = '\0';
    if (!logits_row_identity_build(result))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "draft adjusted row identity derivation failed");
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_runtime_logits_result_validate(
    const yvex_runtime_logits_plan_summary *plan, const float *logits,
    unsigned long long logits_capacity,
    const yvex_runtime_logits_row_result *rows,
    unsigned long long row_capacity,
    const yvex_runtime_logits_result *result, yvex_error *err)
{
    unsigned long long valid_count, index;
    if (!plan || !logits || !rows || !result ||
        result->schema_version != YVEX_RUNTIME_LOGITS_SCHEMA_V1 ||
        !plan->vocabulary_size || !result->completed_rows ||
        result->completed_rows > result->requested_rows ||
        row_capacity < result->completed_rows ||
        !yvex_core_u64_mul(result->completed_rows,
                           plan->vocabulary_size, &valid_count) ||
        logits_capacity != valid_count)
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "repeated logits prefix extent is not canonical");
    if ((result->completed &&
         (result->partial || result->completed_rows != result->requested_rows ||
          result->first_incomplete_row != result->requested_rows)) ||
        (result->partial &&
         (result->completed || result->completed_rows >= result->requested_rows ||
          result->first_incomplete_row != result->completed_rows)) ||
        (!result->completed && !result->partial))
        return logits_refuse(err, YVEX_ERR_FORMAT,
                             "repeated logits completion directory is inconsistent");
    for (index = 0ull; index < result->completed_rows; ++index)
        if (yvex_runtime_logits_row_validate(
                plan, logits + index * plan->vocabulary_size,
                plan->vocabulary_size, &rows[index], err) != YVEX_OK)
            return yvex_error_code(err);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int logits_enter(yvex_runtime_logits_context *context, yvex_error *err)
{
    if (!context || !context->mutex_ready ||
        pthread_mutex_lock(&context->mutex) != 0)
        return logits_refuse(err, YVEX_ERR_STATE, "logits context lock failed");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    return YVEX_OK;
}

static void logits_leave(yvex_runtime_logits_context *context, int completed)
{
    if (context && context->mutex_ready &&
        pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (completed) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
}

int yvex_runtime_logits_project(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *source, yvex_backend_kind backend,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *result, yvex_error *err)
{
    unsigned long long output_bytes;
    int rc = logits_enter(context, err);
    if (result) memset(result, 0, sizeof(*result));
    if (rc != YVEX_OK) return rc;
    if (!result ||
        (!context->options.device_greedy_selection &&
         (!logits || logits_capacity < context->plan.summary.vocabulary_size)) ||
        backend != yvex_backend_kind_of(context->session_view->backend))
        rc = logits_refuse(err, YVEX_ERR_INVALID_ARG,
                           "logits output capacity or backend is incompatible");
    if (rc == YVEX_OK) rc = logits_source_validate(context, source, err);
    if (rc == YVEX_OK && context->options.cancel_requested &&
        context->options.cancel_requested(context->options.cancel_context))
        rc = logits_refuse(err, YVEX_ERR_CANCELLED,
                           "logits projection was cancelled before execution");
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CPU &&
        !source->host_values_available)
        rc = logits_refuse(err, YVEX_ERR_UNSUPPORTED,
                           "CPU logits require an explicit host hidden row");
    if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CPU)
        rc = logits_project_cpu(context, source->normalized_hidden, err);
    else if (rc == YVEX_OK && backend == YVEX_BACKEND_KIND_CUDA)
        rc = logits_project_cuda(context, source, result, err);
    if (rc == YVEX_OK)
        rc = logits_row_finish(context, source, backend, result, err);
    if (rc == YVEX_OK && !context->options.device_greedy_selection &&
        yvex_core_u64_mul(context->plan.summary.vocabulary_size, sizeof(float),
                          &output_bytes))
        memcpy(logits, context->candidate, (size_t)output_bytes);
    else if (rc == YVEX_OK && !context->options.device_greedy_selection)
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits publication extent overflowed");
    logits_leave(context, rc == YVEX_OK);
    return rc;
}

static int logits_batch_contract(
    const yvex_runtime_logits_context *context,
    const yvex_output_head_batch_request *request,
    char identity[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const yvex_runtime_logits_plan_summary *plan = context ? &context->plan.summary : NULL;
    if (!plan || !request || request->schema_version != YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1 ||
        !request->row_count || request->row_count > context->options.maximum_rows ||
        request->output_vocabulary != plan->vocabulary_size ||
        request->backend != yvex_backend_kind_of(context->session_view->backend) ||
        request->result_class != YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS ||
        request->selection_policy != YVEX_OUTPUT_HEAD_SELECTION_RAW ||
        request->execution_class != YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE ||
        request->evidence_profile != context->options.evidence_profile ||
        (context->options.execution_profile &&
         (!request->execution_profile_identity ||
          strcmp(request->execution_profile_identity,
                 context->options.execution_profile->identity) != 0)) ||
        (!context->options.execution_profile && request->execution_profile_identity))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "output-head batch contract is incompatible");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.output-head-batch.v1") ||
        !yvex_sha256_update_text(&hash, plan->output_head_plan_identity) ||
        !yvex_sha256_update_u64(&hash, request->row_count) ||
        !yvex_sha256_update_u64(&hash, request->output_vocabulary) ||
        !yvex_sha256_update_u64(&hash, request->backend) ||
        !yvex_sha256_update_u64(&hash, request->result_class) ||
        !yvex_sha256_update_u64(&hash, request->selection_policy) ||
        !yvex_sha256_update_u64(&hash, request->evidence_profile) ||
        !yvex_sha256_update_u64(&hash, request->execution_class) ||
        !yvex_sha256_update_text(
            &hash, request->execution_profile_identity
                       ? request->execution_profile_identity : "uncompiled-reference") ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "output-head batch contract identity failed");
    yvex_sha256_hex(digest, identity);
    return YVEX_OK;
}

/*
 * Execute an ordered output-head graph with complete-row partial-progress semantics. The portable
 * backend is row-local internally, but row count and result policy belong to this batch contract.
 */
int yvex_runtime_logits_execute_rows(
    yvex_runtime_logits_context *context,
    const yvex_output_head_batch_request *request,
    const yvex_runtime_logits_source *sources,
    float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, required;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !request || !sources || context->options.device_greedy_selection ||
        !rows || row_capacity < request->row_count || !result ||
        !yvex_core_u64_mul(request->row_count, context->plan.summary.vocabulary_size,
                           &required) || !logits || logits_capacity < required)
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "ordered logits request or caller storage is invalid");
    rc = logits_batch_contract(context, request,
                               result->output_head_contract_identity, err);
    if (rc != YVEX_OK) return rc;
    result->schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1;
    result->requested_rows = request->row_count;
    result->first_incomplete_row = request->row_count;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.aggregate.v1"))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "aggregate logits hash initialization failed");
    for (index = 0ull; index < request->row_count; ++index) {
        rc = yvex_runtime_logits_project(
            context, &sources[index], request->backend,
            logits + index * context->plan.summary.vocabulary_size,
            context->plan.summary.vocabulary_size, &rows[index], err);
        if (rc != YVEX_OK) {
            result->partial = index != 0ull;
            result->first_incomplete_row = index;
            break;
        }
        result->completed_rows++;
        result->final_source_position = sources[index].source_position;
        if (!yvex_sha256_update_text(&hash, rows[index].raw_logits_digest)) {
            rc = logits_refuse(err, YVEX_ERR_STATE,
                               "aggregate logits digest update failed");
            result->first_incomplete_row = index + 1ull;
            break;
        }
    }
    if (!yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "aggregate logits digest finalization failed");
    yvex_sha256_hex(digest, result->aggregate_logits_digest);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.logits.execution.v2") ||
        !yvex_sha256_update_text(&hash, result->output_head_contract_identity) ||
        !yvex_sha256_update_u64(&hash, result->requested_rows) ||
        !yvex_sha256_update_u64(&hash, result->completed_rows) ||
        !yvex_sha256_update_u64(&hash, result->first_incomplete_row) ||
        !yvex_sha256_update_u64(&hash, result->final_source_position) ||
        !yvex_sha256_update_text(&hash, result->aggregate_logits_digest) ||
        !yvex_sha256_final(&hash, digest))
        return logits_refuse(err, YVEX_ERR_STATE,
                             "logits execution identity derivation failed");
    yvex_sha256_hex(digest, result->execution_identity);
    result->completed = result->completed_rows == result->requested_rows;
    return rc;
}

int yvex_runtime_logits_execute(
    yvex_runtime_logits_context *context,
    const yvex_runtime_logits_source *sources, unsigned long long row_count,
    yvex_backend_kind backend, float *logits, unsigned long long logits_capacity,
    yvex_runtime_logits_row_result *rows, unsigned long long row_capacity,
    yvex_runtime_logits_result *result, yvex_error *err)
{
    yvex_output_head_batch_request request = {0};
    request.schema_version = YVEX_OUTPUT_HEAD_BATCH_SCHEMA_V1;
    request.row_count = row_count;
    request.output_vocabulary = context ? context->plan.summary.vocabulary_size : 0ull;
    request.backend = backend;
    request.result_class = YVEX_OUTPUT_HEAD_RESULT_HOST_LOGITS;
    request.selection_policy = YVEX_OUTPUT_HEAD_SELECTION_RAW;
    request.evidence_profile = context ? context->options.evidence_profile
                                       : YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.execution_profile_identity = context && context->options.execution_profile
                                             ? context->options.execution_profile->identity : NULL;
    return yvex_runtime_logits_execute_rows(
        context, &request, sources, logits, logits_capacity, rows, row_capacity, result, err);
}

/*
 * Release logits-local buffers while preserving borrowed model/session owners.
 *
 * Busy or backend cleanup refusal retains retryable ownership.
 */
int yvex_runtime_logits_context_close(yvex_runtime_logits_context **context,
                                      yvex_error *err)
{
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->mutex_ready && pthread_mutex_lock(&(*context)->mutex) == 0) {
        if ((*context)->busy) {
            (void)pthread_mutex_unlock(&(*context)->mutex);
            return logits_refuse(err, YVEX_ERR_STATE,
                                 "busy logits context cannot close");
        }
        (void)pthread_mutex_unlock(&(*context)->mutex);
    }
    if ((*context)->device_logits)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_logits, err);
    if (rc == YVEX_OK && (*context)->device_hidden)
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_hidden, err);
    if (rc != YVEX_OK) return rc;
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    free((*context)->candidate);
    memset(*context, 0, sizeof(**context));
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int logits_transformer_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_transformer_context_close(
        (yvex_runtime_transformer_context **)opaque, err);
}

static int logits_input_slice(yvex_transformer_input **out,
                              const yvex_transformer_input_summary *base,
                              const unsigned int *tokens,
                              unsigned long long offset,
                              unsigned long long count,
                              yvex_error *err)
{
    yvex_transformer_input_summary summary;
    if (!out || !base || !tokens || !count || offset > base->token_count ||
        count > base->token_count - offset)
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "logits token-input slice is invalid");
    summary = *base;
    summary.token_start = base->token_start + offset;
    summary.token_count = count;
    summary.payload_bytes = 0ull;
    summary.payload_digest[0] = summary.input_identity[0] = '\0';
    if (yvex_transformer_input_seal(&summary, tokens, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_transformer_input_open_memory(out, &summary, tokens, err);
}

void yvex_runtime_logits_operator_result_release(yvex_logits_operator_result *result)
{
    if (!result) return;
    free(result->rows);
    free(result->raw_logits);
    result->rows = NULL;
    result->raw_logits = NULL;
    result->raw_logits_count = 0ull;
    result->row_count = 0ull;
}

static void logits_operator_refuse(yvex_logits_operator_result *result,
                                   const yvex_error *err)
{
    if (!result) return;
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err)
                            ? yvex_error_message(err)
                            : "logits execution refused");
}

static void logits_operator_publish_facts(
    yvex_logits_operator_result *result,
    const yvex_transformer_plan_summary *transformer_plan,
    const yvex_runtime_model_view *model_view,
    const yvex_runtime_logits_context *logits_context,
    yvex_backend_kind backend, int completed)
{
    yvex_runtime_residency_summary residency;
    yvex_error ignored;
    if (result && transformer_plan && model_view && logits_context) {
        const yvex_runtime_logits_plan_summary *logits_plan =
            yvex_runtime_logits_plan_summary_get(logits_context);
        if (logits_plan) result->plan = *logits_plan;
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->adapter->family_name);
        yvex_runtime_identity_copy(result->artifact_identity,
                                   model_view->binding->artifact_identity);
        yvex_runtime_identity_copy(result->runtime_binding_identity,
                                   model_view->binding->identity);
        yvex_runtime_identity_copy(result->transformer_plan_identity,
                                   transformer_plan->transformer_plan_identity);
        result->row_count = result->execution.completed_rows;
        result->prefill_logits_rows = result->row_count ? 1ull : 0ull;
        result->decode_logits_rows = result->row_count - result->prefill_logits_rows;
    }
    memset(&residency, 0, sizeof(residency));
    yvex_error_clear(&ignored);
    if (result && model_view && yvex_runtime_residency_snapshot(
            model_view->residency, &residency, NULL, NULL, &ignored) == YVEX_OK) {
        result->output_head_host_bytes = residency.output_head_encoded_bytes;
        if (backend == YVEX_BACKEND_KIND_CUDA) {
            result->output_head_device_bytes = residency.output_head_encoded_bytes;
            result->output_head_upload_bytes = residency.output_head_encoded_bytes;
            result->output_head_upload_count = 1ull;
        }
    }
    if (result && completed) {
        result->output_head_binding_ready = result->output_head_residency_ready = 1;
        result->logits_cpu_ready = result->logits_cuda_ready = 1;
        result->logits_prefill_ready = result->logits_decode_ready = 1;
        result->logits_full_vocabulary_ready = result->logits_hidden_contract_ready = 1;
        result->logits_partial_progress_ready = result->logits_ready = 1;
    }
}

static int logits_operator_finish(yvex_logits_operator_result *result, int rc,
                                  yvex_error *err)
{
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        logits_operator_refuse(result, err);
    }
    return rc;
}

/*
 * Publish only the completed raw-logits prefix after repeated execution.
 *
 * Extent overflow preserves caller ownership and returns typed refusal.
 */
static int logits_operator_publish_raw(
    yvex_logits_operator_result *result, float **raw_logits,
    unsigned long long raw_capacity, unsigned long long row_capacity,
    const yvex_runtime_logits_plan_summary *plan, int rc, yvex_error *err)
{
    unsigned long long valid_logits_count;
    yvex_error validation_error;
    int validation_rc;
    if (!*raw_logits || !result->execution.completed_rows)
        return rc;
    if (!plan || !yvex_core_u64_mul(result->execution.completed_rows,
                                    plan->vocabulary_size,
                                    &valid_logits_count) ||
        valid_logits_count > raw_capacity) {
        if (rc != YVEX_OK) return rc;
        return logits_refuse(err, YVEX_ERR_BOUNDS,
                             "completed raw logits prefix overflowed");
    }
    result->raw_logits = *raw_logits;
    result->raw_logits_count = valid_logits_count;
    *raw_logits = NULL;
    yvex_error_clear(&validation_error);
    validation_rc = yvex_runtime_logits_result_validate(
        plan, result->raw_logits, result->raw_logits_count,
        result->rows, row_capacity, &result->execution, &validation_error);
    if (validation_rc != YVEX_OK) {
        free(result->raw_logits);
        result->raw_logits = NULL;
        result->raw_logits_count = 0ull;
        if (rc == YVEX_OK) {
            if (err) *err = validation_error;
            return validation_rc;
        }
    }
    return rc;
}

/*
 * Execute one shared-context prefill/decode/logits operator workflow.
 *
 * Cleanup leases preserve exact ownership; completed raw rows remain caller-owned evidence.
 */
int yvex_runtime_logits_operator_execute(
    const yvex_logits_operator_request *request,
    yvex_logits_operator_result *result,
    yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err)
{
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_logits_options logits_options = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_transformer_context *transformer = NULL;
    yvex_runtime_decode_context *decode = NULL;
    yvex_runtime_logits_context *logits_context = NULL;
    yvex_transformer_input *input = NULL, *prefill_input = NULL, *decode_input = NULL;
    const yvex_transformer_input_summary *input_summary;
    const yvex_transformer_plan_summary *plan;
    const yvex_runtime_model_view *model_view;
    const unsigned int *tokens;
    yvex_transformer_input_limits limits = {0};
    yvex_runtime_transformer_request prefill_request = {0};
    yvex_runtime_transformer_output prefill_output = {0};
    yvex_runtime_transformer_result prefill_result = {0};
    yvex_runtime_decode_request decode_request = {0};
    yvex_runtime_decode_output decode_output = {0};
    yvex_runtime_decode_result decode_result = {0};
    yvex_runtime_logits_source *sources = NULL;
    yvex_runtime_decode_step_result *decode_steps = NULL;
    float *prefill_hidden = NULL, *decode_hidden = NULL, *raw_logits = NULL;
    unsigned long long prefill_count, decode_count, hidden_count, row_count, logits_count;
    yvex_error primary = {0}, cleanup_error;
    int adopted = 0, rc, close_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path || !request->runtime_binding_path ||
        !request->input_path || !request->prefill_tokens ||
        !request->prefill_chunk_tokens || !request->context_capacity ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA))
        return logits_refuse(err, YVEX_ERR_INVALID_ARG,
                             "complete logits operator arguments are required");
    yvex_core_text_copy(result->command, sizeof(result->command),
                        "execute transformer logits");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(
        &cleanup, &model_request, &session_request, &model, &session, &failure, err);
    limits.maximum_file_bytes = request->maximum_host_bytes
                                    ? request->maximum_host_bytes : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_file(&input, request->input_path,
                                              &limits, err);
    transformer_options.maximum_host_bytes = request->maximum_host_bytes;
    transformer_options.maximum_device_bytes = request->maximum_device_bytes;
    transformer_options.context_capacity = request->context_capacity;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_open(
            &transformer, model, session, &transformer_options, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(
            cleanup, transformer, logits_transformer_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    input_summary = yvex_transformer_input_summary_get(input);
    tokens = yvex_transformer_input_token_ids(input);
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(transformer));
    model_view = yvex_runtime_model_view_get(model);
    if (rc == YVEX_OK &&
        (!input_summary || !tokens || !plan || !model_view || input_summary->token_start ||
         request->prefill_tokens >= input_summary->token_count ||
         request->context_capacity < input_summary->token_count))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits prefill/decode split or capacity is invalid");
    prefill_count = request->prefill_tokens;
    decode_count = input_summary ? input_summary->token_count - prefill_count : 0ull;
    row_count = decode_count + 1ull;
    if (rc == YVEX_OK)
        rc = logits_input_slice(&prefill_input, input_summary, tokens, 0ull,
                                prefill_count, err);
    if (rc == YVEX_OK)
        rc = logits_input_slice(&decode_input, input_summary, tokens + prefill_count,
                                prefill_count, decode_count, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(prefill_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float)))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits prefill hidden extent overflowed");
    if (rc == YVEX_OK) {
        prefill_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        prefill_output.normalized_hidden = prefill_hidden;
        prefill_output.capacity = hidden_count;
        if (!prefill_hidden) rc = logits_refuse(err, YVEX_ERR_NOMEM,
                                                "prefill hidden allocation failed");
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(decode_count, plan->hidden_width, &hidden_count) ||
         hidden_count > SIZE_MAX / sizeof(float) || decode_count > SIZE_MAX))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "logits decode hidden extent overflowed");
    if (rc == YVEX_OK) {
        decode_hidden = (float *)calloc((size_t)hidden_count, sizeof(float));
        decode_steps = (yvex_runtime_decode_step_result *)calloc(
            (size_t)decode_count, sizeof(*decode_steps));
        sources = (yvex_runtime_logits_source *)calloc((size_t)row_count,
                                                       sizeof(*sources));
        result->rows = (yvex_runtime_logits_row_result *)calloc(
            (size_t)row_count, sizeof(*result->rows));
        if (!decode_hidden || !decode_steps || !sources || !result->rows)
            rc = logits_refuse(err, YVEX_ERR_NOMEM,
                               "logits operator directory allocation failed");
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(row_count, plan->vocabulary_size, &logits_count) ||
         logits_count > SIZE_MAX / sizeof(float)))
        rc = logits_refuse(err, YVEX_ERR_BOUNDS,
                           "raw logits output extent overflowed");
    if (rc == YVEX_OK) {
        raw_logits = (float *)calloc((size_t)logits_count, sizeof(float));
        if (!raw_logits) rc = logits_refuse(err, YVEX_ERR_NOMEM,
                                            "raw logits output allocation failed");
    }
    decode_options.maximum_steps = decode_count;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_open(
            &decode, transformer, session, &decode_options, err);
    logits_options.maximum_rows = row_count;
    logits_options.maximum_host_bytes = request->maximum_host_bytes;
    logits_options.maximum_device_bytes = request->maximum_device_bytes;
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_open(
            &logits_context, model, session,
            yvex_runtime_transformer_context_plan(transformer),
            &logits_options, err);
    prefill_request.chunk_tokens = request->prefill_chunk_tokens;
    prefill_request.backend = request->backend;
    prefill_request.phase = YVEX_TRANSFORMER_PHASE_PREFILL;
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_execute(
            transformer, prefill_input, &prefill_request, &prefill_output,
            &prefill_result, err);
    decode_output.normalized_hidden = decode_hidden;
    decode_output.normalized_hidden_capacity = hidden_count;
    decode_output.steps = decode_steps;
    decode_output.step_capacity = decode_count;
    decode_request.backend = request->backend;
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(decode, decode_input, &decode_request,
                                         &decode_output, &decode_result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_source_from_transformer(
            logits_context, &sources[0], &prefill_result, prefill_hidden,
            prefill_count * plan->hidden_width, prefill_count - 1ull, err);
    for (unsigned long long index = 0ull; rc == YVEX_OK && index < decode_count; ++index)
        rc = yvex_runtime_logits_source_from_decode(
            logits_context, &sources[index + 1ull], &decode_steps[index],
            decode_hidden + index * plan->hidden_width, plan->hidden_width, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_execute(
            logits_context, sources, row_count, request->backend,
            raw_logits, logits_count, result->rows, row_count,
            &result->execution, err);
    rc = logits_operator_publish_raw(
        result, &raw_logits, logits_count, row_count,
        yvex_runtime_logits_plan_summary_get(logits_context), rc, err);
    logits_operator_publish_facts(result, plan, model_view, logits_context,
                                  request->backend, rc == YVEX_OK);
    primary = err ? *err : (yvex_error){0};
    close_rc = yvex_runtime_logits_context_close(&logits_context, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    close_rc = yvex_runtime_decode_context_close(&decode, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    yvex_transformer_input_close(&decode_input);
    yvex_transformer_input_close(&prefill_input);
    yvex_transformer_input_close(&input);
    free(prefill_hidden); free(decode_hidden); free(decode_steps);
    free(sources); free(raw_logits);
    if (!adopted && transformer) {
        close_rc = yvex_runtime_transformer_context_close(&transformer, &cleanup_error);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    }
    close_rc = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_rc != YVEX_OK) { rc = close_rc; primary = cleanup_error; }
    if (cleanup) *retained_cleanup = cleanup;
    if (err && rc != YVEX_OK) *err = primary;
    return logits_operator_finish(result, rc, err);
}
