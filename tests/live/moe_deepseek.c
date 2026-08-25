/*
 * Exercises real routed/shared expert execution without copying production algorithms. The test
 * consumes one production plan and only the production runtime API. Test-only consumer over the
 * admitted external GGUF and runtime binding.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The production model path retains F32 activations until a separately
 * identity-bound whole-stack policy admits compression. Forensic execution
 * also retains the tighter canonical-order accumulation contract. */
#define MOE_LIVE_PRODUCTION_ABSOLUTE_TOLERANCE 6.0e-3
#define MOE_LIVE_FORENSIC_ABSOLUTE_TOLERANCE 3.0e-3
#define MOE_LIVE_RELATIVE_TOLERANCE 3.0e-3

typedef struct {
    yvex_moe_input_summary summary;
    yvex_moe_input_layer_record *records;
    float *activations;
    unsigned int *token_ids;
    yvex_moe_input *input;
} live_input;

typedef struct {
    yvex_moe_layer_result result;
    float *combined, *routed, *shared, *post, *combination;
} live_layer_output;

typedef struct { unsigned long long calls, cancel_at; } live_cancel_control;

static void live_fail(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "moe_live step=%s status=%d where=%s reason=%s\n", step, rc,
            err ? yvex_error_where(err) : "", err ? yvex_error_message(err) : "");
}

static void live_input_close(live_input *input)
{
    if (!input) return;
    yvex_moe_input_close(&input->input);
    free(input->records);
    free(input->activations);
    free(input->token_ids);
    memset(input, 0, sizeof(*input));
}

static float live_value(unsigned long long layer, unsigned long long column)
{
    unsigned long long value = (layer * 131ull + column * 7ull + 29ull) % 521ull;
    return ((float)value - 260.0f) / 2048.0f;
}

static int live_input_open(live_input *input, const yvex_model_engine *model,
                           const yvex_moe_plan *plan, const char *path, yvex_error *err)
{
    const yvex_model_engine_view *view = yvex_model_engine_view_get(model);
    const yvex_runtime_binding_summary *binding = view ? view->binding : NULL;
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
    unsigned long long layer_index, total = 0ull, offset = 0ull;
    int rc = YVEX_OK;

    memset(input, 0, sizeof(*input));
    if (!binding || !summary || summary->layer_count != 43ull) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.moe.input",
                       "the admitted 43-layer MoE plan is required");
        return YVEX_ERR_STATE;
    }
    input->records = calloc((size_t)summary->layer_count, sizeof(*input->records));
    for (layer_index = 0ull; input->records && layer_index < summary->layer_count;
         ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, layer_index);
        if (!layer || !yvex_core_u64_add(total, layer->expanded_width, &total)) {
            rc = YVEX_ERR_BOUNDS;
            goto fail;
        }
    }
    if (!input->records || total > (unsigned long long)(SIZE_MAX / sizeof(float))) {
        rc = input->records ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM;
        goto fail;
    }
    input->activations = malloc((size_t)total * sizeof(float));
    input->token_ids = malloc(sizeof(*input->token_ids));
    if (!input->activations || !input->token_ids) {
        rc = YVEX_ERR_NOMEM;
        goto fail;
    }
    input->summary.schema_version = YVEX_MOE_INPUT_SCHEMA_V1;
    input->summary.token_count = 1ull;
    input->summary.layer_count = summary->layer_count;
    input->summary.activation_payload_bytes = total * sizeof(float);
    input->summary.token_id_payload_bytes = sizeof(*input->token_ids);
    yvex_runtime_identity_copy(input->summary.logical_model_identity,
                               binding->logical_model_identity);
    yvex_runtime_identity_copy(input->summary.runtime_numeric_identity,
                               binding->runtime_numeric_identity);
    yvex_runtime_identity_copy(input->summary.runtime_descriptor_identity,
                               binding->runtime_descriptor_identity);
    yvex_runtime_identity_copy(input->summary.moe_plan_identity,
                               binding->moe_plan_identity);
    input->token_ids[0] = 1u;
    for (layer_index = 0ull; layer_index < summary->layer_count; ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, layer_index);
        yvex_moe_input_layer_record *record = &input->records[layer_index];
        unsigned long long column;
        record->ordinal = layer_index;
        record->layer_index = layer->layer_index;
        record->width = record->stride = layer->expanded_width;
        record->payload_offset = offset * sizeof(float);
        record->payload_bytes = layer->expanded_width * sizeof(float);
        yvex_runtime_identity_copy(record->layer_identity, layer->layer_identity);
        for (column = 0ull; column < layer->expanded_width; ++column)
            input->activations[offset + column] = live_value(layer_index, column);
        offset += layer->expanded_width;
    }
    rc = yvex_moe_input_seal(&input->summary, input->records, input->activations,
                             input->token_ids, err);
    if (rc == YVEX_OK)
        rc = yvex_moe_input_open_memory(&input->input, &input->summary, input->records,
                                        input->activations, input->token_ids, err);
    if (rc == YVEX_OK)
        rc = yvex_moe_input_write(path, &input->summary, input->records,
                                  input->activations, input->token_ids, err);
    if (rc == YVEX_OK) return YVEX_OK;
fail:
    live_input_close(input);
    if (!yvex_error_is_set(err))
        yvex_error_set(err, (yvex_status)rc, "test.moe.input",
                       "live MoE input allocation or geometry failed");
    return rc;
}

static int live_context_open(yvex_runtime_execution_session **session,
                             yvex_runtime_moe_context **context,
                             yvex_model_engine *model, yvex_backend_kind backend,
                             const yvex_runtime_moe_options *requested_options,
                             yvex_error *err)
{
    yvex_runtime_session_open_request request = {0};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_moe_options options = {0};
    int rc;
    request.backend = backend;
    if (requested_options) options = *requested_options;
    rc = yvex_runtime_session_open(session, model, &request, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_moe_context_open(context, model, *session, &options, NULL, err);
    return rc;
}

static int live_cancel_requested(void *opaque)
{
    live_cancel_control *control = (live_cancel_control *)opaque;
    return ++control->calls >= control->cancel_at;
}

static int live_context_close(yvex_runtime_moe_context **context,
                              yvex_runtime_execution_session **session,
                              yvex_error *err)
{
    yvex_error cleanup;
    int rc = yvex_runtime_moe_context_close(context, err), close_rc;
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    return rc;
}

static int live_layer_output_open(live_layer_output *output,
                                  const yvex_moe_layer_plan *layer)
{
    memset(output, 0, sizeof(*output));
    output->combined = calloc((size_t)layer->hidden_width, sizeof(float));
    output->routed = calloc((size_t)layer->hidden_width, sizeof(float));
    output->shared = calloc((size_t)layer->hidden_width, sizeof(float));
    output->post = calloc((size_t)layer->residual_streams, sizeof(float));
    output->combination = calloc((size_t)(layer->residual_streams * layer->residual_streams),
                                 sizeof(float));
    if (!output->combined || !output->routed || !output->shared || !output->post ||
        !output->combination) return 0;
    output->result.combined_output = output->combined;
    output->result.combined_capacity = layer->hidden_width;
    output->result.routed_output = output->routed;
    output->result.routed_capacity = layer->hidden_width;
    output->result.shared_output = output->shared;
    output->result.shared_capacity = layer->hidden_width;
    output->result.post = output->post;
    output->result.post_capacity = layer->residual_streams;
    output->result.combination = output->combination;
    output->result.combination_capacity = layer->residual_streams * layer->residual_streams;
    return 1;
}

static void live_layer_output_close(live_layer_output *output)
{
    if (!output) return;
    free(output->combination);
    free(output->post);
    free(output->shared);
    free(output->routed);
    free(output->combined);
    memset(output, 0, sizeof(*output));
}

static int live_layer_compare(const yvex_moe_layer_plan *layer,
                              const live_layer_output *cpu,
                              const live_layer_output *cuda, const char *profile,
                              double absolute_tolerance, double *maximum, double *rmse)
{
    unsigned long long index, first_failure = ULLONG_MAX;
    double squared = 0.0;
    int within = 1;
    *maximum = 0.0;
    for (index = 0ull; index < layer->experts_per_token; ++index)
        if (cpu->result.router.selected_experts[index] !=
            cuda->result.router.selected_experts[index]) {
            fprintf(stderr, "moe_live selection_mismatch profile=%s layer=%llu rank=%llu "
                            "cpu=%llu cuda=%llu\n", profile, layer->layer_index, index,
                    cpu->result.router.selected_experts[index],
                    cuda->result.router.selected_experts[index]);
            return 0;
        }
    for (index = 0ull; index < layer->hidden_width; ++index) {
        double left = cpu->combined[index], right = cuda->combined[index];
        double difference = fabs(left - right);
        double scale = fmax(fabs(left), fabs(right));
        if (!isfinite(left) || !isfinite(right) ||
            difference > absolute_tolerance +
                             MOE_LIVE_RELATIVE_TOLERANCE * scale) {
            within = 0;
            if (first_failure == ULLONG_MAX) first_failure = index;
        }
        if (difference > *maximum) *maximum = difference;
        squared += difference * difference;
    }
    *rmse = sqrt(squared / (double)layer->hidden_width);
    if (!within)
        fprintf(stderr, "moe_live numeric_mismatch profile=%s layer=%llu first=%llu "
                        "max_abs=%.17g rmse=%.17g cpu=%.9g cuda=%.9g\n",
                profile, layer->layer_index, first_failure, *maximum, *rmse,
                cpu->combined[first_failure], cuda->combined[first_failure]);
    return within;
}

static int live_representative(yvex_runtime_moe_context *cpu_context,
                               yvex_runtime_moe_context *cuda_context,
                               const yvex_moe_plan *plan, const live_input *input,
                               const char *profile, double absolute_tolerance,
                               yvex_error *err)
{
    const unsigned long long layers[] = {0ull, 3ull, 8ull};
    unsigned long long case_index;
    for (case_index = 0ull; case_index < sizeof(layers) / sizeof(layers[0]); ++case_index) {
        unsigned long long ordinal = layers[case_index], stride;
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, ordinal);
        const float *values;
        live_layer_output cpu = {0}, cuda = {0};
        double maximum = 0.0, rmse = 0.0;
        int rc;
        if (!layer || !live_layer_output_open(&cpu, layer) ||
            !live_layer_output_open(&cuda, layer)) {
            live_layer_output_close(&cuda);
            live_layer_output_close(&cpu);
            return YVEX_ERR_NOMEM;
        }
        rc = yvex_moe_input_layer_view(input->input, ordinal, &values, &stride, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_moe_execute_layer(cpu_context, ordinal, values,
                                                input->token_ids[0], 1,
                                                &cpu.result, err);
        if (rc == YVEX_OK)
            rc = yvex_runtime_moe_execute_layer(cuda_context, ordinal, values,
                                                input->token_ids[0], 1,
                                                &cuda.result, err);
        if (rc == YVEX_OK &&
            !live_layer_compare(layer, &cpu, &cuda, profile, absolute_tolerance,
                                &maximum, &rmse)) {
            yvex_error_setf(err, YVEX_ERR_FORMAT, "test.moe.cpu-cuda",
                            "representative MoE CPU/CUDA comparison failed at layer %llu",
                            ordinal);
            rc = YVEX_ERR_FORMAT;
        }
        if (rc == YVEX_OK)
            printf("moe_profile=%s layer=%llu router=%s selected=%llu "
                   "max_abs=%.17g rmse=%.17g\n", profile, ordinal,
                   layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID
                                ? "hash" : "learned",
                   cpu.result.router.selected_count, maximum, rmse);
        live_layer_output_close(&cuda);
        live_layer_output_close(&cpu);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int live_qtypes_match(const yvex_moe_plan *plan,
                             const yvex_runtime_moe_result *result)
{
    unsigned long long expected[YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP] = {0};
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
    unsigned long long layer_index;
    if (!summary || !result) return 0;
    for (layer_index = 0ull; layer_index < summary->layer_count; ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, layer_index);
        unsigned int slot;
        if (!layer) return 0;
        for (slot = 0u; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
            unsigned long long accesses = 1ull;
            if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR) continue;
            if (layer->qtypes[slot] >= YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP) return 0;
            if (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
                slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN)
                accesses = layer->experts_per_token;
            expected[layer->qtypes[slot]] += accesses;
        }
    }
    for (layer_index = 0ull; layer_index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP;
         ++layer_index)
        if (expected[layer_index] != result->qtype_counts[layer_index]) return 0;
    return 1;
}

static int live_full_cuda(yvex_runtime_moe_context *context, const yvex_moe_plan *plan,
                          const live_input *input, yvex_error *err)
{
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
    const yvex_moe_layer_plan *first = yvex_moe_plan_layer_at(plan, 0ull);
    yvex_runtime_moe_output output = {0};
    yvex_runtime_moe_result result;
    unsigned long long rows, combined, post, combination;
    int rc;
    if (!summary || !first ||
        !yvex_core_u64_mul(summary->layer_count, input->summary.token_count, &rows) ||
        !yvex_core_u64_mul(rows, first->hidden_width, &combined) ||
        !yvex_core_u64_mul(rows, first->residual_streams, &post) ||
        !yvex_core_u64_mul(post, first->residual_streams, &combination))
        return YVEX_ERR_BOUNDS;
    output.combined_capacity = combined;
    output.post_capacity = post;
    output.combination_capacity = combination;
    output.combined_outputs = calloc((size_t)combined, sizeof(float));
    output.post = calloc((size_t)post, sizeof(float));
    output.combination = calloc((size_t)combination, sizeof(float));
    if (!output.combined_outputs || !output.post || !output.combination) rc = YVEX_ERR_NOMEM;
    else rc = yvex_runtime_moe_execute(context, input->input, &output, &result, err);
    if (rc == YVEX_OK &&
        (!result.completed || result.layers_executed != 43ull ||
         result.hash_router_executions != 3ull ||
         result.learned_router_executions != 40ull ||
         result.routed_expert_executions != 258ull ||
         result.shared_expert_executions != 43ull ||
         result.expert_subviews_accessed != 774ull ||
         !live_qtypes_match(plan, &result))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.moe.full-cuda",
                       "full CUDA MoE execution counters are incomplete");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        printf("moe_full_cuda_layers=%llu hash=%llu learned=%llu routed=%llu shared=%llu "
               "subviews=%llu combined_output_digest=%s routing_digest=%s\n",
               result.layers_executed, result.hash_router_executions,
               result.learned_router_executions, result.routed_expert_executions,
               result.shared_expert_executions, result.expert_subviews_accessed,
               result.combined_output_digest, result.routing_digest);
    free(output.combination);
    free(output.post);
    free(output.combined_outputs);
    return rc;
}

static int live_cancellation(yvex_model_engine *model, const yvex_moe_plan *plan,
                             const live_input *input, yvex_error *err)
{
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_moe_context *context = NULL;
    const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(plan, 0ull);
    const float *values = NULL;
    unsigned long long stride = 0ull, index;
    live_layer_output output = {0};
    live_cancel_control control = {0ull, 2ull};
    yvex_runtime_moe_options options = {0};
    yvex_error cleanup;
    int rc, close_rc;
    options.cancel_requested = live_cancel_requested;
    options.cancel_context = &control;
    rc = live_context_open(&session, &context, model, YVEX_BACKEND_KIND_CPU,
                           &options, err);
    if (rc == YVEX_OK && (!layer || !live_layer_output_open(&output, layer)))
        rc = YVEX_ERR_NOMEM;
    if (rc == YVEX_OK)
        rc = yvex_moe_input_layer_view(input->input, 0ull, &values, &stride, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_moe_execute_layer(context, 0ull, values,
                                            input->token_ids[0], 1,
                                            &output.result, err);
    if (rc != YVEX_ERR_CANCELLED) {
        if (rc == YVEX_OK)
            yvex_error_set(err, YVEX_ERR_STATE, "test.moe.cancel",
                           "post-selection cancellation was not observed");
        rc = YVEX_ERR_STATE;
    } else {
        rc = YVEX_OK;
        for (index = 0ull; index < layer->hidden_width; ++index)
            if (output.combined[index] != 0.0f) rc = YVEX_ERR_STATE;
        if (rc == YVEX_OK) rc = yvex_runtime_moe_context_reset(context, err);
        if (rc == YVEX_OK) yvex_error_clear(err);
    }
    live_layer_output_close(&output);
    yvex_error_clear(&cleanup);
    close_rc = live_context_close(&context, &session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    if (rc == YVEX_OK) printf("moe_cancellation_after_selection=refused_and_reusable\n");
    return rc;
}

int main(int argc, char **argv)
{
    yvex_model_engine_open_request request = {0};
    yvex_model_engine_failure failure = {0};
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *cpu_session = NULL, *cuda_session = NULL;
    yvex_runtime_execution_session *forensic_session = NULL;
    yvex_runtime_moe_context *cpu_context = NULL, *cuda_context = NULL;
    yvex_runtime_moe_context *forensic_context = NULL;
    yvex_runtime_moe_options forensic_options = {0};
    const yvex_moe_plan *plan;
    live_input input;
    yvex_error err, cleanup;
    int rc, close_rc;

    if (argc != 4) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING MOE_INPUT_OUTPUT\n", argv[0]);
        return 2;
    }
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    memset(&input, 0, sizeof(input));
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash-dspark";
    request.residency_backend = YVEX_BACKEND_KIND_CUDA;
    forensic_options.evidence_level = YVEX_ATTENTION_EVIDENCE_FULL;
    rc = yvex_model_engine_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK)
        rc = live_context_open(&cpu_session, &cpu_context, model,
                               YVEX_BACKEND_KIND_CPU, NULL, &err);
    if (rc == YVEX_OK)
        rc = live_context_open(&cuda_session, &cuda_context, model,
                               YVEX_BACKEND_KIND_CUDA, NULL, &err);
    if (rc == YVEX_OK)
        rc = live_context_open(&forensic_session, &forensic_context, model,
                               YVEX_BACKEND_KIND_CUDA, &forensic_options, &err);
    plan = yvex_runtime_moe_context_plan(cpu_context);
    if (rc == YVEX_OK) rc = live_input_open(&input, model, plan, argv[3], &err);
    if (rc == YVEX_OK)
        rc = live_representative(cpu_context, cuda_context, plan, &input,
                                 "production",
                                 MOE_LIVE_PRODUCTION_ABSOLUTE_TOLERANCE, &err);
    if (rc == YVEX_OK)
        rc = live_representative(cpu_context, forensic_context, plan, &input,
                                 "forensic",
                                 MOE_LIVE_FORENSIC_ABSOLUTE_TOLERANCE, &err);
    if (forensic_context || forensic_session) {
        yvex_error_clear(&cleanup);
        close_rc = live_context_close(&forensic_context, &forensic_session, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    }
    if (rc == YVEX_OK) rc = live_full_cuda(cuda_context, plan, &input, &err);
    if (rc == YVEX_OK) rc = live_cancellation(model, plan, &input, &err);
    if (rc != YVEX_OK) live_fail("execution", rc, &err);
    live_input_close(&input);
    if (forensic_context || forensic_session) {
        yvex_error_clear(&cleanup);
        close_rc = live_context_close(&forensic_context, &forensic_session, &cleanup);
        if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    }
    yvex_error_clear(&cleanup);
    close_rc = live_context_close(&cuda_context, &cuda_session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_context_close(&cpu_context, &cpu_session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_model_engine_close(&model);
    if (model && rc == YVEX_OK) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        printf("moe_input=%s\nmoe_plan_ready=1\nmoe_block_ready=1\n"
               "transformer_ready=0\ngeneration_ready=0\n", argv[3]);
    return rc == YVEX_OK ? 0 : 1;
}
