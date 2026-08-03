/*
 * Exercises two real teacher-forced decode steps consume prior KV and publish normalized hidden
 * rows. Both backends reuse one warm transformer/session and commit exactly once per decode
 * token. Test-only consumer over the admitted external GGUF and runtime binding.
 */
#include <yvex/internal/decode.h>
#include <yvex/internal/runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECODE_LIVE_ABSOLUTE_TOLERANCE 8.0e-3
#define DECODE_LIVE_RELATIVE_TOLERANCE 8.0e-3

typedef struct {
    yvex_runtime_execution_session *session;
    yvex_runtime_transformer_context *transformer;
    yvex_runtime_decode_context *decode;
    float *prefill_hidden, *decode_hidden;
    yvex_runtime_decode_step_result steps[3];
    yvex_runtime_transformer_result prefill;
    yvex_runtime_decode_result result;
    yvex_runtime_session_summary before_decode, after_decode;
} live_decode;

static void live_fail(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "decode_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
}

static int live_decode_open(live_decode *execution, yvex_runtime_model *model,
                            yvex_backend_kind backend,
                            int (*decode_cancel)(void *), void *cancel_context,
                            yvex_error *err)
{
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_transformer_options transformer_options = {0};
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_model_failure failure = {0};
    memset(execution, 0, sizeof(*execution));
    session_request.backend = backend;
    transformer_options.context_capacity = 4ull;
    decode_options.maximum_steps = 3ull;
    decode_options.cancel_requested = decode_cancel;
    decode_options.cancel_context = cancel_context;
    if (yvex_runtime_session_open(&execution->session, model, &session_request,
                                  &failure, err) != YVEX_OK)
        return yvex_error_code(err);
    if (yvex_runtime_transformer_context_open(
            &execution->transformer, model, execution->session,
            &transformer_options, err) != YVEX_OK)
        return yvex_error_code(err);
    return yvex_runtime_decode_context_open(
        &execution->decode, execution->transformer, execution->session,
        &decode_options, err);
}

static int live_decode_close(live_decode *execution, yvex_error *err)
{
    yvex_error cleanup;
    int rc, close_rc;
    free(execution->prefill_hidden);
    free(execution->decode_hidden);
    execution->prefill_hidden = execution->decode_hidden = NULL;
    rc = yvex_runtime_decode_context_close(&execution->decode, err);
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_transformer_context_close(
        &execution->transformer, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = yvex_runtime_session_close(&execution->session, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; *err = cleanup; }
    return rc;
}

static int live_input_open(yvex_transformer_input **input,
                           const yvex_transformer_plan_summary *plan,
                           const unsigned int *tokens,
                           unsigned long long start,
                           unsigned long long count,
                           const char *path, yvex_error *err)
{
    yvex_transformer_input_summary summary;
    int rc;
    memset(&summary, 0, sizeof(summary));
    summary.schema_version = YVEX_TRANSFORMER_INPUT_SCHEMA_V1;
    summary.token_start = start;
    summary.token_count = count;
    summary.vocabulary_size = plan->vocabulary_size;
    yvex_runtime_identity_copy(summary.logical_model_identity,
                               plan->logical_model_identity);
    yvex_runtime_identity_copy(summary.runtime_numeric_identity,
                               plan->runtime_numeric_identity);
    yvex_runtime_identity_copy(summary.runtime_descriptor_identity,
                               plan->runtime_descriptor_identity);
    yvex_runtime_identity_copy(summary.transformer_plan_identity,
                               plan->transformer_plan_identity);
    rc = yvex_transformer_input_seal(&summary, tokens, err);
    if (rc == YVEX_OK)
        rc = yvex_transformer_input_open_memory(input, &summary, tokens, err);
    if (rc == YVEX_OK && path)
        rc = yvex_transformer_input_write(path, &summary, tokens, err);
    return rc;
}

static int live_decode_run(live_decode *execution,
                           const yvex_transformer_input *prefill_input,
                           const yvex_transformer_input *decode_input,
                           yvex_backend_kind backend,
                           unsigned long long width, yvex_error *err)
{
    yvex_runtime_transformer_request prefill_request = {
        .chunk_tokens = 1ull,
        .backend = backend,
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
    yvex_runtime_transformer_output prefill_output;
    yvex_runtime_decode_request decode_request = {.backend = backend};
    yvex_runtime_decode_output decode_output;
    int rc;
    execution->prefill_hidden = (float *)calloc((size_t)width, sizeof(float));
    execution->decode_hidden = (float *)calloc((size_t)(2ull * width), sizeof(float));
    if (!execution->prefill_hidden || !execution->decode_hidden) return YVEX_ERR_NOMEM;
    prefill_output.normalized_hidden = execution->prefill_hidden;
    prefill_output.capacity = width;
    decode_output.normalized_hidden = execution->decode_hidden;
    decode_output.normalized_hidden_capacity = 2ull * width;
    decode_output.steps = execution->steps;
    decode_output.step_capacity = 2ull;
    rc = yvex_runtime_transformer_execute(
        execution->transformer, prefill_input, &prefill_request,
        &prefill_output, &execution->prefill, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(
            execution->session, &execution->before_decode, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(
            execution->decode, decode_input, &decode_request,
            &decode_output, &execution->result, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_summary_copy(
            execution->session, &execution->after_decode, err);
    if (rc == YVEX_OK &&
        (!execution->result.completed || execution->result.partial ||
         execution->result.requested_steps != 2ull ||
         execution->result.completed_steps != 2ull ||
         execution->result.initial_committed_prefix != 1ull ||
         execution->result.final_committed_prefix != 3ull ||
         execution->result.generation_after !=
             execution->result.generation_before + 2ull ||
         execution->result.layers_executed != 86ull ||
         execution->result.swa_layers != 4ull ||
         execution->result.csa_layers != 42ull ||
         execution->result.hca_layers != 40ull ||
         execution->result.hash_routers != 6ull ||
         execution->result.learned_routers != 80ull ||
         execution->result.routed_experts != 516ull ||
         execution->result.shared_experts != 86ull ||
         execution->after_decode.workspace_generation !=
             execution->before_decode.workspace_generation ||
         execution->after_decode.workspace_bytes !=
             execution->before_decode.workspace_bytes ||
         execution->after_decode.host_workspace_bytes !=
             execution->before_decode.host_workspace_bytes ||
         execution->after_decode.device_workspace_bytes !=
             execution->before_decode.device_workspace_bytes ||
         strcmp(execution->after_decode.workspace_identity,
                execution->before_decode.workspace_identity) != 0 ||
         execution->after_decode.workspace_capacity_failure_count !=
             execution->before_decode.workspace_capacity_failure_count)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.decode.structure",
                       "repeated decode structural evidence is incomplete");
        rc = YVEX_ERR_FORMAT;
    }
    return rc;
}

static int live_compare(const live_decode *cpu, const live_decode *cuda,
                        unsigned long long count, double *maximum, double *rmse)
{
    unsigned long long index;
    double squared = 0.0;
    *maximum = 0.0;
    for (index = 0ull; index < count; ++index) {
        double left = cpu->decode_hidden[index], right = cuda->decode_hidden[index];
        double difference = fabs(left - right);
        double scale = fmax(fabs(left), fabs(right));
        if (!isfinite(left) || !isfinite(right) ||
            difference > DECODE_LIVE_ABSOLUTE_TOLERANCE +
                             DECODE_LIVE_RELATIVE_TOLERANCE * scale)
            return 0;
        if (difference > *maximum) *maximum = difference;
        squared += difference * difference;
    }
    *rmse = sqrt(squared / (double)count);
    return 1;
}

static int live_reference_run(live_decode *execution,
                              const yvex_transformer_input *stream,
                              unsigned long long width, yvex_error *err)
{
    yvex_runtime_transformer_request request = {
        .chunk_tokens = 1ull,
        .backend = YVEX_BACKEND_KIND_CPU,
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
    yvex_runtime_transformer_output output;
    execution->prefill_hidden = (float *)calloc((size_t)(3ull * width), sizeof(float));
    if (!execution->prefill_hidden) return YVEX_ERR_NOMEM;
    output.normalized_hidden = execution->prefill_hidden;
    output.capacity = 3ull * width;
    return yvex_runtime_transformer_execute(
        execution->transformer, stream, &request, &output,
        &execution->prefill, err);
}

typedef struct { unsigned int calls; } live_cancel_state;

static int live_cancel_after_two(void *opaque)
{
    live_cancel_state *state = (live_cancel_state *)opaque;
    return ++state->calls > 2u;
}

/* Prove cancellation after two steps publishes exact typed partial progress. */
static int live_partial_run(live_decode *execution,
                            const yvex_transformer_input *prefill_input,
                            const yvex_transformer_input *decode_input,
                            unsigned long long width, yvex_error *err)
{
    yvex_runtime_transformer_request prefill_request = {
        .chunk_tokens = 1ull,
        .backend = YVEX_BACKEND_KIND_CPU,
        .phase = YVEX_TRANSFORMER_PHASE_PREFILL};
    yvex_runtime_transformer_output prefill_output;
    yvex_runtime_decode_request decode_request = {.backend = YVEX_BACKEND_KIND_CPU};
    yvex_runtime_decode_output decode_output;
    int rc;
    execution->prefill_hidden = (float *)calloc((size_t)width, sizeof(float));
    execution->decode_hidden = (float *)calloc((size_t)(3ull * width), sizeof(float));
    if (!execution->prefill_hidden || !execution->decode_hidden) return YVEX_ERR_NOMEM;
    prefill_output.normalized_hidden = execution->prefill_hidden;
    prefill_output.capacity = width;
    decode_output.normalized_hidden = execution->decode_hidden;
    decode_output.normalized_hidden_capacity = 3ull * width;
    decode_output.steps = execution->steps;
    decode_output.step_capacity = 3ull;
    rc = yvex_runtime_transformer_execute(
        execution->transformer, prefill_input, &prefill_request,
        &prefill_output, &execution->prefill, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_execute(
            execution->decode, decode_input, &decode_request,
            &decode_output, &execution->result, err);
    if (rc != YVEX_ERR_CANCELLED || !execution->result.partial ||
        execution->result.completed || execution->result.completed_steps != 2ull ||
        execution->result.first_incomplete_step != 2ull ||
        execution->result.final_committed_prefix != 3ull ||
        !execution->steps[0].completed || !execution->steps[1].completed ||
        execution->steps[2].completed) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.decode.partial",
                       "decode cancellation did not preserve exact two-step progress");
        return YVEX_ERR_FORMAT;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int main(int argc, char **argv)
{
    yvex_runtime_model_open_request request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_model *model = NULL;
    yvex_transformer_input *stream = NULL, *prefill = NULL, *decode = NULL;
    yvex_transformer_input *mutated_prefill = NULL, *mutated_decode = NULL;
    live_decode cpu = {0}, cuda = {0}, reference = {0}, partial = {0}, prefix = {0};
    const yvex_transformer_plan_summary *plan = NULL;
    const unsigned int tokens[] = {1u, 2u, 3u};
    const unsigned int prefix_token[] = {4u};
    const unsigned int decode_mutation[] = {4u, 3u, 5u};
    yvex_error err, cleanup;
    double maximum = 0.0, rmse = 0.0;
    live_cancel_state cancel_state = {0};
    const char *step = "model-open";
    int rc, close_rc;
    if (argc != 4) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING TOKEN_STREAM_OUTPUT\n", argv[0]);
        return 2;
    }
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash-dspark";
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc == YVEX_OK) {
        step = "cpu-open";
        rc = live_decode_open(&cpu, model, YVEX_BACKEND_KIND_CPU, NULL, NULL, &err);
    }
    plan = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(cpu.transformer));
    if (rc == YVEX_OK) { step = "stream-open"; rc = live_input_open(&stream, plan, tokens, 0ull, 3ull, argv[3], &err); }
    if (rc == YVEX_OK) { step = "prefill-open"; rc = live_input_open(&prefill, plan, tokens, 0ull, 1ull, NULL, &err); }
    if (rc == YVEX_OK) {
        step = "decode-open";
        rc = live_input_open(&decode, plan, tokens + 1u, 1ull, 2ull, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "mutated-prefix-open";
        rc = live_input_open(&mutated_prefill, plan, prefix_token, 0ull, 1ull,
                             NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "mutated-decode-open";
        rc = live_input_open(&mutated_decode, plan, decode_mutation, 1ull, 3ull,
                             NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "cpu-decode";
        rc = live_decode_run(&cpu, prefill, decode, YVEX_BACKEND_KIND_CPU,
                             plan->hidden_width, &err);
    }
    if (rc == YVEX_OK) {
        step = "reference-open";
        rc = live_decode_open(&reference, model, YVEX_BACKEND_KIND_CPU,
                              NULL, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "reference-execute";
        rc = live_reference_run(&reference, stream, plan->hidden_width, &err);
    }
    if (rc == YVEX_OK &&
        (memcmp(cpu.decode_hidden,
                reference.prefill_hidden + plan->hidden_width,
                (size_t)(2ull * plan->hidden_width) * sizeof(float)) != 0 ||
         strcmp(cpu.result.aggregate_state_digest,
                reference.prefill.persistent_state_digest) != 0)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.decode.sequence-equivalence",
                       "decode steps diverged from equivalent transformer sequence");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "prefix-mutation-open";
        rc = live_decode_open(&prefix, model, YVEX_BACKEND_KIND_CPU,
                              NULL, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "prefix-mutation-execute";
        rc = live_decode_run(&prefix, mutated_prefill, decode,
                             YVEX_BACKEND_KIND_CPU, plan->hidden_width, &err);
    }
    if (rc == YVEX_OK &&
        strcmp(cpu.steps[0].normalized_hidden_digest,
               prefix.steps[0].normalized_hidden_digest) == 0) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.decode.prefix-causality",
                       "mutated committed prefix did not change decode output");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "partial-open";
        rc = live_decode_open(&partial, model, YVEX_BACKEND_KIND_CPU,
                              live_cancel_after_two, &cancel_state, &err);
    }
    if (rc == YVEX_OK) {
        step = "partial-execute";
        rc = live_partial_run(&partial, prefill, mutated_decode,
                              plan->hidden_width, &err);
    }
    if (rc == YVEX_OK &&
        (strcmp(cpu.steps[0].normalized_hidden_digest,
                partial.steps[0].normalized_hidden_digest) == 0 ||
         strcmp(cpu.steps[1].normalized_hidden_digest,
                partial.steps[1].normalized_hidden_digest) == 0)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.decode.token-causality",
                       "mutated decode token did not change current and later outputs");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        step = "cuda-open";
        rc = live_decode_open(&cuda, model, YVEX_BACKEND_KIND_CUDA,
                              NULL, NULL, &err);
    }
    if (rc == YVEX_OK) {
        step = "cuda-decode";
        rc = live_decode_run(&cuda, prefill, decode, YVEX_BACKEND_KIND_CUDA,
                             plan->hidden_width, &err);
    }
    if (rc == YVEX_OK &&
        (!live_compare(&cpu, &cuda, 2ull * plan->hidden_width, &maximum, &rmse) ||
         strcmp(cpu.result.aggregate_state_digest,
                cuda.result.aggregate_state_digest) != 0)) {
        yvex_error_set(&err, YVEX_ERR_FORMAT, "test.decode.cpu-cuda",
                       "CPU/CUDA repeated decode comparison failed");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc != YVEX_OK) live_fail(step, rc, &err);
    else
        printf("decode_steps=2 layers=86 swa=4 csa=42 hca=40 hash=6 learned=80 "
               "routed=516 shared=86 max_abs=%.17g rmse=%.17g\n"
               "cpu_hidden_digest=%s\ncuda_hidden_digest=%s\nkv_digest=%s\n"
               "sequence_equivalence=pass prefix_causality=pass "
               "token_causality=pass partial_progress=pass warm_reuse=pass\n"
               "model_decode_ready=1 logits_ready=0 generation_ready=0\n",
               maximum, rmse, cpu.result.aggregate_hidden_digest,
               cuda.result.aggregate_hidden_digest,
               cuda.result.aggregate_state_digest);
    yvex_transformer_input_close(&mutated_decode);
    yvex_transformer_input_close(&mutated_prefill);
    yvex_transformer_input_close(&decode);
    yvex_transformer_input_close(&prefill);
    yvex_transformer_input_close(&stream);
    yvex_error_clear(&cleanup);
    close_rc = live_decode_close(&cuda, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_decode_close(&prefix, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_decode_close(&partial, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_decode_close(&reference, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_error_clear(&cleanup);
    close_rc = live_decode_close(&cpu, &cleanup);
    if (rc == YVEX_OK && close_rc != YVEX_OK) { rc = close_rc; err = cleanup; }
    yvex_runtime_model_close(&model);
    if (model && rc == YVEX_OK) rc = YVEX_ERR_STATE;
    return rc == YVEX_OK ? 0 : 1;
}
