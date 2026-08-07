/*
 * Exercises one real token traverses embedding, 43 blocks, final head/norm, and persistent
 * state. Both backends consume one identity-bound token bundle through production APIs.
 * Test-only consumer over the admitted external GGUF and runtime binding.
 */
#include <yvex/internal/transformer.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/logits.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRANSFORMER_LIVE_ABSOLUTE_TOLERANCE 8.0e-3
#define TRANSFORMER_LIVE_RELATIVE_TOLERANCE 8.0e-3

typedef struct {
    yvex_runtime_execution_session *session;
    yvex_runtime_transformer_context *context;
    float *hidden, *features;
    yvex_runtime_transformer_result result;
} live_execution;

static void live_fail(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "transformer_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
}

static int live_execution_open(live_execution *execution, yvex_runtime_model *model,
                               yvex_backend_kind backend, yvex_error *err)
{
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options options = {0};
    yvex_runtime_model_failure failure = {0};
    memset(execution, 0, sizeof(*execution));
    session_request.backend = backend;
    options.context_capacity = 2ull;
    options.workspace_token_capacity = 2ull;
    options.evidence_level = YVEX_ATTENTION_EVIDENCE_NONE;
    if (yvex_runtime_session_open(&execution->session, model, &session_request,
                                  &failure, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_runtime_transformer_context_open(
        &execution->context, model, execution->session, &options, err);
}

static int live_execution_close(live_execution *execution, yvex_error *err)
{
    yvex_error cleanup;
    int rc, close_rc;
    free(execution->hidden);
    execution->hidden = NULL;
    free(execution->features);
    execution->features = NULL;
    rc = yvex_runtime_transformer_context_close(&execution->context, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&execution->session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup;
    }
    return rc;
}

static int live_input_open(yvex_transformer_input **input,
                           yvex_transformer_input_summary *summary,
                           const yvex_runtime_model *model,
                           const yvex_transformer_plan *plan,
                           const unsigned int *token_ids,
                           unsigned long long token_count,
                           const char *path, yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_runtime_binding_summary *binding = view ? view->binding : NULL;
    const yvex_transformer_plan_summary *plan_summary =
        yvex_transformer_plan_summary_get(plan);
    int rc;
    if (!binding || !plan_summary || plan_summary->layer_count != 43ull ||
        !token_ids || !token_count)
        return YVEX_ERR_STATE;
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary->token_count = token_count;
    summary->vocabulary_size = plan_summary->vocabulary_size;
    yvex_runtime_identity_copy(summary->logical_model_identity,
                               binding->logical_model_identity);
    yvex_runtime_identity_copy(summary->runtime_numeric_identity,
                               binding->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary->runtime_descriptor_identity,
                               binding->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary->transformer_plan_identity,
                               plan_summary->transformer_plan_identity);
    rc = yvex_transformer_input_seal(summary, token_ids, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(input, summary, token_ids, err);
    if (rc == YVEX_OK && path)
        rc = yvex_transformer_input_write(path, summary, token_ids, err);
    return rc;
}

static int live_execute(live_execution *execution, const yvex_transformer_input *input,
                        yvex_backend_kind backend, unsigned long long chunk_tokens,
                        yvex_error *err)
{
    unsigned long long *feature_layers = NULL;
    const yvex_transformer_plan_summary *plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(execution->context));
    const yvex_transformer_input_summary *input_summary =
        yvex_transformer_input_summary_get(input);
    yvex_runtime_transformer_request request = {
        .chunk_tokens = chunk_tokens,
        .backend = backend,
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
    yvex_runtime_transformer_output output = {0};
    unsigned long long output_count, feature_count, layer;
    int rc;
    if (!plan || !input_summary ||
        !yvex_core_u64_mul(input_summary->token_count, plan->hidden_width, &output_count) ||
        !yvex_core_u64_mul(output_count, plan->layer_count, &feature_count) ||
        output_count > SIZE_MAX / sizeof(float) ||
        feature_count > SIZE_MAX / sizeof(float) ||
        plan->layer_count > SIZE_MAX / sizeof(*feature_layers)) return YVEX_ERR_STATE;
    execution->hidden = calloc((size_t)output_count, sizeof(float));
    execution->features = calloc((size_t)feature_count, sizeof(float));
    feature_layers = malloc((size_t)plan->layer_count * sizeof(*feature_layers));
    if (!execution->hidden || !execution->features || !feature_layers) {
        free(feature_layers);
        return YVEX_ERR_NOMEM;
    }
    for (layer = 0ull; layer < plan->layer_count; ++layer)
        feature_layers[layer] = layer;
    request.feature_layer_ordinals = feature_layers;
    request.feature_layer_count = plan->layer_count;
    output.normalized_hidden = execution->hidden;
    output.capacity = output_count;
    output.features = execution->features;
    output.feature_capacity = feature_count;
    rc = yvex_runtime_transformer_execute(execution->context, input, &request,
                                          &output, &execution->result, err);
    free(feature_layers);
    if (rc == YVEX_OK &&
        (!execution->result.completed ||
         execution->result.layers_executed != 43ull * execution->result.chunk_count ||
         execution->result.swa_layers != 2ull * execution->result.chunk_count ||
         execution->result.csa_layers != 21ull * execution->result.chunk_count ||
         execution->result.hca_layers != 20ull * execution->result.chunk_count ||
         execution->result.hash_routers != 3ull * input_summary->token_count ||
         execution->result.learned_routers != 40ull * input_summary->token_count ||
         execution->result.routed_experts != 258ull * input_summary->token_count ||
         execution->result.shared_experts != 43ull * input_summary->token_count ||
         execution->result.feature_layer_count != plan->layer_count ||
         execution->result.feature_row_count != input_summary->token_count ||
         execution->result.position_after != input_summary->token_count ||
         !yvex_sha256_hex_valid(execution->result.normalized_hidden_digest) ||
         !yvex_sha256_hex_valid(execution->result.persistent_state_digest))) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.transformer.structure",
                       "full transformer execution counters are incomplete");
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

static int live_values_compare(const float *left_values, const float *right_values,
                               unsigned long long width, double *maximum, double *rmse,
                               unsigned long long *first_mismatch,
                               double *first_left, double *first_right)
{
    unsigned long long index;
    double squared = 0.0;
    int matched = 1;
    *maximum = 0.0;
    if (first_mismatch) *first_mismatch = width;
    if (first_left) *first_left = 0.0;
    if (first_right) *first_right = 0.0;
    for (index = 0ull; index < width; ++index) {
        double left = left_values[index], right = right_values[index];
        double difference = fabs(left - right);
        double scale = fmax(fabs(left), fabs(right));
        if (!isfinite(left) || !isfinite(right) ||
            difference > TRANSFORMER_LIVE_ABSOLUTE_TOLERANCE +
                             TRANSFORMER_LIVE_RELATIVE_TOLERANCE * scale) {
            if (matched) {
                if (first_mismatch) *first_mismatch = index;
                if (first_left) *first_left = left;
                if (first_right) *first_right = right;
            }
            matched = 0;
        }
        if (!isfinite(difference)) {
            squared = INFINITY;
            *maximum = INFINITY;
            continue;
        }
        if (difference > *maximum) *maximum = difference;
        squared += difference * difference;
    }
    *rmse = sqrt(squared / (double)width);
    return matched;
}

static int live_compare(const live_execution *cpu, const live_execution *cuda,
                        unsigned long long width, double *maximum, double *rmse,
                        unsigned long long *first_mismatch,
                        double *first_left, double *first_right)
{
    return live_values_compare(cpu->hidden, cuda->hidden, width, maximum, rmse,
                               first_mismatch, first_left, first_right);
}

static unsigned long long live_first_layer_mismatch(
    const live_execution *cpu, const live_execution *cuda,
    unsigned long long layer_count, unsigned long long width,
    double *maximum, double *rmse)
{
    unsigned long long layer;
    for (layer = 0ull; layer < layer_count; ++layer)
        if (!live_values_compare(cpu->features + layer * width,
                                 cuda->features + layer * width,
                                 width, maximum, rmse, NULL, NULL, NULL))
            break;
    return layer;
}

static int live_argmax_compare(
    yvex_runtime_model *model, const live_execution *cpu,
    const live_execution *cuda, unsigned int *cpu_token,
    unsigned int *cuda_token, float *cpu_margin, float *cuda_margin,
    double *logit_maximum, double *logit_rmse, double *probability_tv,
    yvex_error *err)
{
    yvex_runtime_logits_options options = {0};
    yvex_runtime_logits_context *context = NULL;
    yvex_runtime_logits_source cpu_source, cuda_source;
    yvex_runtime_logits_row_result row;
    const yvex_runtime_logits_plan_summary *summary;
    float *cpu_logits = NULL, *cuda_logits = NULL;
    unsigned long long index, width;
    unsigned int cpu_second = 0u, cuda_second = 0u;
    double squared = 0.0, cpu_maximum, cuda_maximum;
    double cpu_mass = 0.0, cuda_mass = 0.0;
    int rc;
    options.maximum_rows = 1ull;
    options.evidence_profile = YVEX_EXECUTION_EVIDENCE_FORENSIC;
    rc = yvex_runtime_logits_context_open(
        &context, model, cpu->session,
        yvex_runtime_transformer_context_plan(cpu->context), &options, err);
    summary = rc == YVEX_OK ? yvex_runtime_logits_plan_summary_get(context) : NULL;
    width = summary ? summary->vocabulary_size : 0ull;
    if (rc == YVEX_OK && (!width || width > SIZE_MAX / sizeof(float))) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "test.transformer.logits",
                       "output-head vocabulary exceeds comparison bounds");
        rc = YVEX_ERR_BOUNDS;
    }
    if (rc == YVEX_OK) {
        cpu_logits = malloc((size_t)width * sizeof(*cpu_logits));
        cuda_logits = malloc((size_t)width * sizeof(*cuda_logits));
        if (!cpu_logits || !cuda_logits) {
            yvex_error_set(err, YVEX_ERR_NOMEM, "test.transformer.logits",
                           "output-head comparison allocation failed");
            rc = YVEX_ERR_NOMEM;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_source_from_transformer(
            context, &cpu_source, &cpu->result, cpu->hidden,
            cpu->result.token_count * summary->hidden_width,
            cpu->result.token_count - 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_source_from_transformer(
            context, &cuda_source, &cuda->result, cuda->hidden,
            cuda->result.token_count * summary->hidden_width,
            cuda->result.token_count - 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_project(
            context, &cpu_source, YVEX_BACKEND_KIND_CPU,
            cpu_logits, width, &row, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_project(
            context, &cuda_source, YVEX_BACKEND_KIND_CPU,
            cuda_logits, width, &row, err);
    if (rc == YVEX_OK) {
        *cpu_token = *cuda_token = 0u;
        *logit_maximum = 0.0;
        for (index = 1ull; index < width; ++index) {
            if (cpu_logits[index] > cpu_logits[*cpu_token])
                *cpu_token = (unsigned int)index;
            if (cuda_logits[index] > cuda_logits[*cuda_token])
                *cuda_token = (unsigned int)index;
        }
        for (index = 0ull; index < width; ++index) {
            double difference = fabs((double)cpu_logits[index] - cuda_logits[index]);
            if (difference > *logit_maximum) *logit_maximum = difference;
            squared += difference * difference;
        }
        *logit_rmse = sqrt(squared / (double)width);
        cpu_second = *cpu_token == 0u ? 1u : 0u;
        cuda_second = *cuda_token == 0u ? 1u : 0u;
        for (index = 0ull; index < width; ++index) {
            if (index != *cpu_token && cpu_logits[index] > cpu_logits[cpu_second])
                cpu_second = (unsigned int)index;
            if (index != *cuda_token && cuda_logits[index] > cuda_logits[cuda_second])
                cuda_second = (unsigned int)index;
        }
        *cpu_margin = cpu_logits[*cpu_token] - cpu_logits[cpu_second];
        *cuda_margin = cuda_logits[*cuda_token] - cuda_logits[cuda_second];
        cpu_maximum = cpu_logits[*cpu_token];
        cuda_maximum = cuda_logits[*cuda_token];
        for (index = 0ull; index < width; ++index) {
            cpu_mass += exp((double)cpu_logits[index] - cpu_maximum);
            cuda_mass += exp((double)cuda_logits[index] - cuda_maximum);
        }
        *probability_tv = 0.0;
        for (index = 0ull; index < width; ++index)
            *probability_tv += fabs(
                exp((double)cpu_logits[index] - cpu_maximum) / cpu_mass -
                exp((double)cuda_logits[index] - cuda_maximum) / cuda_mass);
        *probability_tv *= 0.5;
    }
    free(cuda_logits);
    free(cpu_logits);
    if (yvex_runtime_logits_context_close(&context, rc == YVEX_OK ? err : NULL) != YVEX_OK &&
        rc == YVEX_OK)
        rc = yvex_error_code(err);
    return rc;
}

static int live_history_changes(const float *with_history, const float *without_history,
                                unsigned long long width)
{
    unsigned long long index;
    for (index = 0ull; index < width; ++index)
        if (with_history[index] != without_history[index]) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    yvex_runtime_model_open_request request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_model *model = NULL;
    yvex_transformer_input *input = NULL, *cli_input = NULL, *empty_input = NULL;
    yvex_transformer_input_summary input_summary, auxiliary_summary;
    live_execution cpu = {0}, cpu_steps = {0}, cpu_empty = {0}, cuda = {0};
    const yvex_transformer_plan_summary *plan;
    const unsigned int tokens[] = {1u, 2u}, cli_token[] = {1u}, empty_token[] = {2u};
    yvex_error err, cleanup;
    double maximum = 0.0, rmse = 0.0;
    const char *step = "model-open";
    unsigned long long first_mismatch = 0ull;
    unsigned long long first_layer = 0ull;
    double first_left = 0.0, first_right = 0.0;
    double layer_maximum = 0.0, layer_rmse = 0.0;
    double logit_maximum = 0.0, logit_rmse = 0.0, probability_tv = 0.0;
    float cpu_margin = 0.0f, cuda_margin = 0.0f;
    unsigned int cpu_token = 0u, cuda_token = 0u;
    int rc, close_rc, hidden_match, state_match;
    if (argc != 4) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING TOKEN_INPUT_OUTPUT\n", argv[0]);
        return 2;
    }
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash-dspark";
    request.residency_backend = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK) {
        step = "cpu-context-open";
        rc = live_execution_open(&cpu, model, YVEX_BACKEND_KIND_CPU, &err);
    }
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(cpu.context));
    if (rc == YVEX_OK) {
        step = "input-open";
        rc = live_input_open(&input, &input_summary, model,
                             yvex_runtime_transformer_context_plan(cpu.context),
                             tokens, 2ull, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "cli-input-open";
        rc = live_input_open(&cli_input, &auxiliary_summary, model,
                             yvex_runtime_transformer_context_plan(cpu.context),
                             cli_token, 1ull, argv[3], &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-execute";
        rc = live_execute(&cpu, input, YVEX_BACKEND_KIND_CPU, 2ull, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-step-context-open";
        rc = live_execution_open(&cpu_steps, model, YVEX_BACKEND_KIND_CPU, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-step-execute";
        rc = live_execute(&cpu_steps, input, YVEX_BACKEND_KIND_CPU, 1ull, &err);
    }
    if (rc == YVEX_OK &&
        (!live_compare(&cpu, &cpu_steps, plan->hidden_width * 2ull, &maximum, &rmse,
                       NULL, NULL, NULL) ||
         strcmp(cpu.result.persistent_state_digest,
                cpu_steps.result.persistent_state_digest) != 0)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.transformer.chunk-equivalence",
                       "whole chunk and repeated one-token chunks diverged");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "empty-input-open";
        rc = live_input_open(&empty_input, &auxiliary_summary, model,
                             yvex_runtime_transformer_context_plan(cpu.context),
                             empty_token, 1ull, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-empty-context-open";
        rc = live_execution_open(&cpu_empty, model, YVEX_BACKEND_KIND_CPU, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-empty-execute";
        rc = live_execute(&cpu_empty, empty_input, YVEX_BACKEND_KIND_CPU, 1ull, &err);
    }
    if (rc == YVEX_OK && !live_history_changes(
            cpu.hidden + plan->hidden_width, cpu_empty.hidden, plan->hidden_width)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.transformer.causality",
                       "committed first-token history did not affect the second token");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "cuda-context-open";
        rc = live_execution_open(&cuda, model, YVEX_BACKEND_KIND_CUDA, &err);
    }
    if (rc == YVEX_OK) {
        step = "cuda-execute";
        rc = live_execute(&cuda, input, YVEX_BACKEND_KIND_CUDA, 2ull, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-cuda-compare";
        hidden_match = plan && live_compare(
            &cpu, &cuda, plan->hidden_width * 2ull, &maximum, &rmse,
            &first_mismatch, &first_left, &first_right);
        first_layer = plan ? live_first_layer_mismatch(
                                 &cpu, &cuda, plan->layer_count,
                                 plan->hidden_width, &layer_maximum, &layer_rmse)
                           : 0ull;
        state_match = strcmp(cpu.result.persistent_state_digest,
                             cuda.result.persistent_state_digest) == 0;
        rc = live_argmax_compare(model, &cpu, &cuda, &cpu_token, &cuda_token,
                                 &cpu_margin, &cuda_margin, &logit_maximum,
                                 &logit_rmse, &probability_tv, &err);
        if (rc == YVEX_OK && (!hidden_match || !state_match)) {
            fprintf(stderr,
                    "transformer_live numeric first=%llu cpu=%.9g cuda=%.9g max_abs=%.9g "
                    "rmse=%.9g first_layer=%llu layer_max=%.9g layer_rmse=%.9g "
                    "cpu_token=%u cuda_token=%u cpu_margin=%.9g cuda_margin=%.9g "
                    "logit_max=%.9g logit_rmse=%.9g tv=%.9g\n",
                    first_mismatch, first_left, first_right, maximum, rmse,
                    first_layer, layer_maximum, layer_rmse, cpu_token, cuda_token,
                    cpu_margin, cuda_margin, logit_maximum, logit_rmse,
                    probability_tv);
            yvex_error_setf(
                &err, YVEX_ERR_FORMAT, "test.transformer.cpu-cuda",
                "CPU/CUDA hidden comparison failed (state=%d first_layer=%llu "
                "max_abs=%.9g rmse=%.9g cpu_token=%u cuda_token=%u tv=%.9g)",
                state_match, first_layer, maximum, rmse, cpu_token, cuda_token,
                probability_tv);
            rc = YVEX_ERR_FORMAT;
        }
    }
    if (rc != YVEX_OK) live_fail(step, rc, &err);
    else
        printf("transformer_layers=43 chunks=1 tokens=2 swa=2 csa=21 hca=20 "
               "hash=6 learned=80 routed=516 shared=86 max_abs=%.17g rmse=%.17g\n"
               "chunk_equivalence=pass causal_history=pass first_layer=%llu "
               "layer_max_abs=%.17g layer_rmse=%.17g cpu_token=%u cuda_token=%u "
               "logit_max_abs=%.17g logit_rmse=%.17g probability_tv=%.17g\n"
               "cpu_hidden_digest=%s\ncuda_hidden_digest=%s\nkv_digest=%s\n",
               maximum, rmse, first_layer, layer_maximum, layer_rmse, cpu_token,
               cuda_token, logit_maximum, logit_rmse, probability_tv,
               cpu.result.normalized_hidden_digest, cuda.result.normalized_hidden_digest,
               cuda.result.persistent_state_digest);
    yvex_transformer_input_close(&empty_input);
    yvex_transformer_input_close(&cli_input);
    yvex_transformer_input_close(&input);
    yvex_error_clear(&cleanup);
    close_rc = live_execution_close(&cuda, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_execution_close(&cpu_empty, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_execution_close(&cpu_steps, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_execution_close(&cpu, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_runtime_model_close(&model);
    if (model && rc == YVEX_OK) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        printf("transformer_input=%s\ntransformer_ready=1\n"
               "model_decode_ready=0\nlogits_ready=0\ngeneration_ready=0\n", argv[3]);
    return rc == YVEX_OK ? 0 : 1;
}
