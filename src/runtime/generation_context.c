/*
 * A generation context admits one model/session pair and owns the finite composition of lower
 * transformer, logits, sampling, tokenizer, and speculative resources. Construction publishes
 * nothing until every lower plan and lifecycle primitive is ready; close drains an admitted turn
 * before releasing those resources in reverse dependency order.
 */
#include "src/runtime/private.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <build_commit.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>

static int generation_context_refuse(yvex_error *err, yvex_status status,
                                     const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}

static int generation_stops_open(yvex_runtime_generation_context *context,
                                 unsigned long long vocabulary_size,
                                 yvex_error *err)
{
    unsigned long long index;
    if (!context->options.additional_stop_token_count) return YVEX_OK;
    if (!context->options.additional_stop_token_ids ||
        context->options.additional_stop_token_count > SIZE_MAX / sizeof(unsigned int))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "additional stop-token extent is invalid");
    context->additional_stops = yvex_core_calloc(
        (size_t)context->options.additional_stop_token_count,
        sizeof(*context->additional_stops));
    if (!context->additional_stops)
        return generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "additional stop-token allocation failed");
    for (index = 0ull; index < context->options.additional_stop_token_count; ++index) {
        unsigned int token = context->options.additional_stop_token_ids[index];
        unsigned long long scan;
        if (token >= vocabulary_size)
            return generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "additional stop token is outside vocabulary");
        for (scan = 0ull; scan < index; ++scan)
            if (context->additional_stops[scan] == token)
                return generation_context_refuse(
                    err, YVEX_ERR_FORMAT,
                    "additional stop tokens contain a duplicate");
        context->additional_stops[index] = token;
    }
    context->options.additional_stop_token_ids = context->additional_stops;
    return YVEX_OK;
}

static int generation_execution_profile_build(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    const yvex_physical_execution_summary *physical =
        yvex_physical_execution_ir_summary(context->model_view->physical_execution);
    const yvex_runtime_binding_summary *binding = context->model_view->binding;
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(context->session);
    yvex_runtime_session_summary session;
    yvex_compiled_execution_profile_request request = {0};
    yvex_backend_cuda_attention_graph_summary cuda = {0};
    const char *kernel_bundle = YVEX_BUILD_IDENTITY;
    char hardware[YVEX_EXECUTION_TEXT_CAP];
    int rc;

    if (!physical || !binding || !session_view || !session_view->backend ||
        yvex_runtime_session_summary_copy(context->session, &session, err) != YVEX_OK)
        return generation_context_refuse(
            err, YVEX_ERR_STATE, "execution profile owners are unavailable");
    if (context->options.backend == YVEX_BACKEND_KIND_CUDA) {
        rc = yvex_backend_cuda_attention_graph_summary_get(
            session_view->backend, &cuda, err);
        if (rc != YVEX_OK || !yvex_sha256_hex_valid(cuda.cuda_build_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE, "CUDA kernel bundle identity is unavailable");
        kernel_bundle = cuda.cuda_build_identity;
        (void)snprintf(hardware, sizeof(hardware), "portable-cuda-sm%d%d",
                       session.compute_capability_major,
                       session.compute_capability_minor);
    } else {
        yvex_core_text_copy(hardware, sizeof(hardware), "portable-cpu");
    }
    request.schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1;
    request.logical_model_identity = binding->logical_model_identity;
    request.physical_variant_identity = binding->profile_identity;
    request.physical_execution_identity = physical->identity;
    request.artifact_identity = binding->artifact_identity;
    request.materialization_identity = binding->materialization_identity;
    request.runtime_binding_identity = binding->identity;
    request.kernel_bundle_identity = kernel_bundle;
    request.hardware_profile = hardware;
    request.backend = context->options.backend;
    request.device_index = session.device_index;
    request.compute_major = session.compute_capability_major;
    request.compute_minor = session.compute_capability_minor;
    request.context_capacity = context->options.context_capacity;
    request.generation_mode = context->options.mode == YVEX_GENERATION_MODE_DSPARK
                                  ? YVEX_EXECUTION_GENERATION_SPECULATIVE
                                  : YVEX_EXECUTION_GENERATION_TARGET_ONLY;
    request.workload = YVEX_EXECUTION_WORKLOAD_INTERACTIVE;
    request.evidence = context->options.evidence_profile;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.host_stochastic_reference =
        context->options.sampling_policy.strategy != YVEX_SAMPLING_STRATEGY_GREEDY;
    request.token_local_moe_reference = 1;
    request.eager_attention_reference = 1;
    return yvex_compiled_execution_profile_seal(
        &request, &context->execution_profile, err);
}

static int generation_plan_build(yvex_runtime_generation_context *context,
                                 yvex_error *err)
{
    yvex_runtime_model_summary model;
    const yvex_transformer_plan_summary *transformer;
    const yvex_runtime_logits_plan_summary *logits;
    const yvex_tokenizer_plan_summary *tokenizer;
    yvex_runtime_generation_plan_summary plan;
    if (yvex_runtime_model_summary_copy(context->model, &model, err) != YVEX_OK)
        return yvex_error_code(err);
    transformer = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    logits = yvex_runtime_logits_plan_summary_get(context->logits);
    tokenizer = yvex_tokenizer_plan_summary_get(context->tokenizer);
    if (!transformer || !logits || !tokenizer || !tokenizer->sealed ||
        !tokenizer->runtime_bound ||
        tokenizer->vocabulary_size != transformer->vocabulary_size ||
        tokenizer->vocabulary_size != logits->vocabulary_size ||
        strcmp(transformer->transformer_plan_identity,
               logits->transformer_plan_identity) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lower-owner plans are incompatible");
    memset(&plan, 0, sizeof(plan));
    plan.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V4;
    plan.backend = context->options.backend;
    plan.mode = context->options.mode;
    plan.context_capacity = context->options.context_capacity;
    plan.prefill_chunk_tokens = context->options.prefill_chunk_tokens;
    plan.maximum_new_tokens = context->options.maximum_new_tokens;
    plan.maximum_output_bytes = context->options.maximum_output_bytes;
    plan.trace_policy = (unsigned int)context->options.trace_policy;
    plan.evidence_profile = context->execution_profile.evidence;
    plan.execution_class = context->execution_profile.execution_class;
    yvex_runtime_identity_copy(plan.runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(plan.runtime_binding_identity,
                               context->model_view->binding->identity);
    yvex_runtime_identity_copy(plan.runtime_descriptor_identity,
                               model.runtime_descriptor_identity);
    yvex_runtime_identity_copy(plan.tokenizer_plan_identity,
                               tokenizer->tokenizer_plan_identity);
    yvex_runtime_identity_copy(plan.prompt_policy_identity,
                               tokenizer->prompt_policy_identity);
    yvex_runtime_identity_copy(plan.transformer_plan_identity,
                               transformer->transformer_plan_identity);
    yvex_runtime_identity_copy(plan.logits_plan_identity,
                               logits->output_head_plan_identity);
    yvex_runtime_identity_copy(plan.sampling_policy_identity,
                               context->options.sampling_policy.policy_identity);
    yvex_runtime_identity_copy(plan.kernel_bundle_identity,
                               context->execution_profile.kernel_bundle_identity);
    yvex_runtime_identity_copy(plan.execution_profile_identity,
                               context->execution_profile.identity);
    yvex_core_text_copy(plan.hardware_profile, sizeof(plan.hardware_profile),
                        context->execution_profile.hardware_profile);
    if (context->speculation) {
        const yvex_speculation_family_policy *policy =
            yvex_runtime_speculation_policy_get(context->speculation);
        if (!policy || !yvex_sha256_hex_valid(policy->policy_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "DSpark policy identity is unavailable");
        yvex_runtime_identity_copy(plan.speculation_policy_identity,
                                   policy->policy_identity);
    }
    if (!yvex_runtime_generation_stop_identity(
            tokenizer, context->additional_stops,
            context->options.additional_stop_token_count,
            plan.stop_policy_identity) ||
        !yvex_runtime_generation_plan_identity(
            &plan, plan.generation_plan_identity))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation plan identity derivation failed");
    context->plan = plan;
    return YVEX_OK;
}

static int generation_execution_owners_open(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_options *options,
    const yvex_runtime_logits_plan_summary **logits_plan, yvex_error *err)
{
    yvex_runtime_transformer_options transformer = {0};
    yvex_runtime_logits_options logits = {0};
    yvex_runtime_sampling_options sampling = {0};
    yvex_runtime_speculation_options speculation = {0};
    int rc;

    transformer.maximum_host_bytes = options->maximum_host_bytes;
    transformer.maximum_device_bytes = options->maximum_device_bytes;
    transformer.context_capacity = options->context_capacity;
    transformer.workspace_token_capacity = options->prefill_chunk_tokens;
    if (options->mode == YVEX_GENERATION_MODE_DSPARK &&
        transformer.workspace_token_capacity < YVEX_SPECULATION_MAX_BLOCK + 2ull)
        transformer.workspace_token_capacity = YVEX_SPECULATION_MAX_BLOCK + 2ull;
    transformer.cancel_requested = options->cancel_requested;
    transformer.cancel_context = options->cancel_context;
    transformer.evidence_level =
        runtime_attention_evidence(options->evidence_profile);
    transformer.device_hidden_output =
        options->backend == YVEX_BACKEND_KIND_CUDA &&
        options->mode == YVEX_GENERATION_MODE_TARGET_ONLY &&
        options->sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY &&
        options->evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    transformer.execution_profile = &context->execution_profile;
    transformer.shape_registry = context->execution_shapes;
    rc = yvex_runtime_transformer_context_open(
        &context->transformer, context->model, context->session, &transformer, err);
    logits.maximum_rows = options->mode == YVEX_GENERATION_MODE_DSPARK
                              ? YVEX_SPECULATION_MAX_BLOCK + 1ull
                              : 1ull;
    logits.maximum_host_bytes = options->maximum_host_bytes;
    logits.maximum_device_bytes = options->maximum_device_bytes;
    logits.evidence_profile = options->evidence_profile;
    logits.device_greedy_selection = transformer.device_hidden_output;
    logits.execution_profile = &context->execution_profile;
    logits.cancel_requested = options->cancel_requested;
    logits.cancel_context = options->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_open(
            &context->logits, context->model, context->session,
            yvex_runtime_transformer_context_plan(context->transformer),
            &logits, err);
    *logits_plan = rc == YVEX_OK
                       ? yvex_runtime_logits_plan_summary_get(context->logits)
                       : NULL;
    if (rc == YVEX_OK && !*logits_plan)
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE, "runtime logits plan is unavailable");
    if (rc != YVEX_OK) return rc;
    sampling.maximum_vocabulary_size = (*logits_plan)->vocabulary_size;
    sampling.maximum_rows = logits.maximum_rows;
    sampling.maximum_host_bytes = options->maximum_host_bytes;
    sampling.cancel_requested = options->cancel_requested;
    sampling.cancel_context = options->cancel_context;
    rc = yvex_runtime_sampling_context_open(
        &context->sampling, *logits_plan, &context->options.sampling_policy,
        &sampling, err);
    if (rc != YVEX_OK || options->mode != YVEX_GENERATION_MODE_DSPARK)
        return rc;
    speculation.backend = options->backend;
    speculation.context_capacity = options->context_capacity;
    speculation.maximum_host_bytes = options->maximum_host_bytes;
    speculation.maximum_device_bytes = options->maximum_device_bytes;
    speculation.cancel_requested = options->cancel_requested;
    speculation.cancel_context = options->cancel_context;
    speculation.execution_profile = &context->execution_profile;
    speculation.shape_registry = context->execution_shapes;
    return yvex_runtime_speculation_context_open(
        &context->speculation, context->model, context->session,
        context->transformer, context->logits, context->sampling,
        &context->options.sampling_policy, &speculation, err);
}

int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_decode_options decode_options = {0};
    yvex_tokenizer_decode_options decoder_options = {0};
    const yvex_runtime_logits_plan_summary *logits_plan;
    unsigned long long hidden_bytes, logits_bytes;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !model || !session || !options ||
        options->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V4 ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        options->mode > YVEX_GENERATION_MODE_DSPARK ||
        !options->context_capacity || !options->prefill_chunk_tokens ||
        !options->maximum_new_tokens || !options->maximum_output_bytes ||
        options->trace_policy > YVEX_RUNTIME_TRACE_FULL ||
        options->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete bounded generation options are required");
    context = yvex_core_calloc(1u, sizeof(*context));
    if (!context)
        return generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "generation context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->tokenizer = context->model_view ? context->model_view->tokenizer : NULL;
    context->options = *options;
    atomic_init(&context->lifecycle, 0u);
    atomic_init(&context->admission_failures, 0ull);
    if (!context->model_view || !context->tokenizer ||
        !yvex_runtime_session_view_get(session) ||
        yvex_runtime_session_view_get(session)->model != model) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation model, session, and tokenizer are not paired");
        goto failure;
    }
    rc = generation_stops_open(
        context, yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    context->options.sampling_policy = options->sampling_policy;
    rc = yvex_runtime_sampling_policy_seal(
        &context->options.sampling_policy,
        yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_execution_profile_build(context, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_execution_shape_registry_open(
        &context->execution_shapes, 128ull, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_execution_owners_open(
        context, options, &logits_plan, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_plan_build(context, err);
    if (rc != YVEX_OK) goto failure;
    decode_options.maximum_steps = options->maximum_new_tokens;
    decode_options.cancel_requested = options->cancel_requested;
    decode_options.cancel_context = options->cancel_context;
    rc = yvex_runtime_decode_context_open(
        &context->decode, context->transformer, session, &decode_options, err);
    if (rc != YVEX_OK) goto failure;
    decoder_options.skip_special_tokens = 1;
    decoder_options.require_complete_utf8 = 1;
    decoder_options.cancelled = options->cancel_requested;
    decoder_options.cancel_context = options->cancel_context;
    rc = yvex_tokenizer_decoder_open(
        &context->decoder, context->tokenizer, &decoder_options, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_token_sequence_open(
        &context->sequence, options->maximum_new_tokens, err);
    if (rc != YVEX_OK) goto failure;
    context->hidden_count = logits_plan->hidden_width;
    context->logits_count = logits_plan->vocabulary_size;
    if ((context->speculation &&
         !yvex_core_u64_mul(context->hidden_count,
                            YVEX_SPECULATION_MAX_BLOCK + 2ull,
                            &context->hidden_count)) ||
        !yvex_core_u64_mul(context->hidden_count, sizeof(float), &hidden_bytes) ||
        !yvex_core_u64_mul(context->logits_count, sizeof(float), &logits_bytes) ||
        !yvex_core_u64_add(hidden_bytes, logits_bytes,
                           &context->workspace_bytes) ||
        (context->speculation &&
         !yvex_core_u64_add(context->workspace_bytes, logits_bytes,
                            &context->workspace_bytes)) ||
        context->workspace_bytes > SIZE_MAX ||
        (options->maximum_host_bytes &&
         context->workspace_bytes > options->maximum_host_bytes)) {
        rc = generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "generation-local workspace exceeds its budget");
        goto failure;
    }
    context->hidden = yvex_core_calloc(
        (size_t)context->hidden_count, sizeof(float));
    context->logits_row = yvex_core_calloc(
        (size_t)context->logits_count, sizeof(float));
    if (context->speculation)
        context->anchor_probabilities = yvex_core_calloc(
            (size_t)context->logits_count, sizeof(float));
    if (!context->hidden || !context->logits_row ||
        (context->speculation && !context->anchor_probabilities)) {
        rc = generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "generation-local workspace allocation failed");
        goto failure;
    }
    if (pthread_mutex_init(&context->drain_mutex, NULL) != 0) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lifecycle mutex initialization failed");
        goto failure;
    }
    context->drain_mutex_ready = 1;
    if (pthread_cond_init(&context->drain_condition, NULL) != 0) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lifecycle condition initialization failed");
        goto failure;
    }
    context->drain_condition_ready = 1;
    context->continuation_allowed = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;

failure:
    if (context) {
        yvex_error cleanup;
        yvex_error_clear(&cleanup);
        yvex_token_sequence_close(&context->sequence);
        yvex_tokenizer_decoder_close(&context->decoder);
        (void)yvex_runtime_decode_context_close(&context->decode, &cleanup);
        (void)yvex_runtime_speculation_context_close(
            &context->speculation, &cleanup);
        (void)yvex_runtime_sampling_context_close(&context->sampling, &cleanup);
        (void)yvex_runtime_logits_context_close(&context->logits, &cleanup);
        (void)yvex_runtime_transformer_context_close(
            &context->transformer, &cleanup);
        yvex_execution_shape_registry_close(&context->execution_shapes);
        if (context->drain_condition_ready)
            (void)pthread_cond_destroy(&context->drain_condition);
        if (context->drain_mutex_ready)
            (void)pthread_mutex_destroy(&context->drain_mutex);
        yvex_core_free(context->logits_row);
        yvex_core_free(context->anchor_probabilities);
        yvex_core_free(context->hidden);
        yvex_core_free(context->additional_stops);
        yvex_core_free(context);
    }
    return rc;
}

const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context)
{
    return context ? &context->plan : NULL;
}

int yvex_runtime_generation_context_close(
    yvex_runtime_generation_context **context, yvex_error *err)
{
    yvex_runtime_generation_context *owner;
    unsigned int observed, desired;
    int rc;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *context;
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    while (!(observed & YVEX_GENERATION_LIFECYCLE_CLOSING)) {
        desired = observed | YVEX_GENERATION_LIFECYCLE_CLOSING;
        if (atomic_compare_exchange_weak_explicit(
                &owner->lifecycle, &observed, desired,
                memory_order_acq_rel, memory_order_acquire))
            break;
    }
    if (owner->drain_mutex_ready) {
        if (pthread_mutex_lock(&owner->drain_mutex) != 0)
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "generation close drain lock failed");
        while (atomic_load_explicit(&owner->lifecycle, memory_order_acquire) &
               YVEX_GENERATION_LIFECYCLE_ACTIVE) {
            if (!owner->drain_condition_ready ||
                pthread_cond_wait(&owner->drain_condition,
                                  &owner->drain_mutex) != 0) {
                (void)pthread_mutex_unlock(&owner->drain_mutex);
                return generation_context_refuse(
                    err, YVEX_ERR_STATE,
                    "generation close drain failed");
            }
        }
        (void)pthread_mutex_unlock(&owner->drain_mutex);
    }
    rc = yvex_runtime_speculation_context_close(&owner->speculation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_close(&owner->sampling, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_close(&owner->logits, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_close(&owner->decode, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_close(&owner->transformer, err);
    if (rc != YVEX_OK) return rc;
    yvex_execution_shape_registry_close(&owner->execution_shapes);
    yvex_tokenizer_decoder_close(&owner->decoder);
    yvex_token_sequence_close(&owner->sequence);
    if (owner->drain_condition_ready &&
        pthread_cond_destroy(&owner->drain_condition) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation close condition cleanup failed");
    owner->drain_condition_ready = 0;
    if (owner->drain_mutex_ready &&
        pthread_mutex_destroy(&owner->drain_mutex) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation close mutex cleanup failed");
    owner->drain_mutex_ready = 0;
    atomic_store_explicit(&owner->lifecycle,
                          YVEX_GENERATION_LIFECYCLE_CLOSED,
                          memory_order_release);
    yvex_core_free(owner->logits_row);
    yvex_core_free(owner->anchor_probabilities);
    yvex_core_free(owner->hidden);
    yvex_core_free(owner->additional_stops);
    memset(owner, 0, sizeof(*owner));
    yvex_core_free(owner);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
