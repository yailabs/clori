/*
 * Exercises activation prefill over real DeepSeek weights without creating another executor.
 * Every input covers 43 admitted layers and every execution uses session-persistent state.
 * Test-only consumer of the production runtime-prefill API and admitted external artifact.
 */
#include <yvex/internal/core.h>
#include <yvex/internal/runtime_prefill.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    yvex_runtime_activation_input_summary summary;
    yvex_runtime_activation_layer_record *records;
    float *payload;
    yvex_runtime_activation_input *input;
} live_activation;

typedef struct {
    const yvex_runtime_activation_prefill_result *result;
    unsigned long long cancel_after_chunks;
} live_cancel;

static int live_fail(const char *step, int rc, const yvex_error *err)
{
    fprintf(stderr, "prefill_live step=%s status=%d where=%s reason=%s\n",
            step, rc, err ? yvex_error_where(err) : "",
            err ? yvex_error_message(err) : "");
    return 0;
}

static void live_activation_close(live_activation *activation)
{
    if (!activation) return;
    yvex_runtime_activation_input_close(&activation->input);
    free(activation->records);
    free(activation->payload);
    memset(activation, 0, sizeof(*activation));
}

static float live_activation_value(unsigned long long layer,
                                   unsigned long long token,
                                   unsigned long long column,
                                   unsigned int variant)
{
    unsigned long long value =
        (layer * 131ull + token * 17ull + column * 7ull + 19ull) % 257ull;
    float result = ((float)value - 128.0f) / 512.0f;
    if (variant && token == 0ull)
        result += (column % 2ull ? -1.0f : 1.0f) / 64.0f;
    return result;
}

static int live_activation_open(live_activation *activation,
                                const yvex_runtime_model *model,
                                unsigned long long start,
                                unsigned long long tokens,
                                unsigned int variant, yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_runtime_binding_summary *binding = view ? view->binding : NULL;
    const yvex_attention_plan *plan = view ? view->attention : NULL;
    unsigned long long layers = yvex_attention_plan_layer_count(plan);
    unsigned long long index, total = 0ull, offset = 0ull;
    int rc = YVEX_OK;

    memset(activation, 0, sizeof(*activation));
    if (!binding || !plan || layers != 43ull || !tokens) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.prefill.activation",
                       "the admitted 43-layer runtime model is required");
        return YVEX_ERR_STATE;
    }
    activation->records = calloc((size_t)layers, sizeof(*activation->records));
    if (!activation->records) return YVEX_ERR_NOMEM;
    for (index = 0ull; index < layers; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(plan, index);
        unsigned long long elements;
        if (!layer || !layer->hidden_dimension ||
            !yvex_core_u64_mul(tokens, layer->hidden_dimension, &elements) ||
            !yvex_core_u64_add(total, elements, &total)) {
            rc = YVEX_ERR_BOUNDS;
            goto fail;
        }
    }
    if (total > (unsigned long long)(SIZE_MAX / sizeof(float))) {
        rc = YVEX_ERR_BOUNDS;
        goto fail;
    }
    activation->payload = malloc((size_t)total * sizeof(float));
    if (!activation->payload) {
        rc = YVEX_ERR_NOMEM;
        goto fail;
    }
    activation->summary.schema_version =
        YVEX_RUNTIME_ACTIVATION_INPUT_SCHEMA_V1;
    activation->summary.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
    activation->summary.token_start = start;
    activation->summary.token_count = tokens;
    activation->summary.layer_count = layers;
    activation->summary.payload_bytes = total * sizeof(float);
#define COPY_BINDING(member)                                                   \
    yvex_runtime_identity_copy(activation->summary.member, binding->member)
    COPY_BINDING(logical_model_identity);
    COPY_BINDING(runtime_numeric_identity);
    COPY_BINDING(runtime_descriptor_identity);
    COPY_BINDING(attention_plan_identity);
#undef COPY_BINDING
    for (index = 0ull; index < layers; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(plan, index);
        yvex_runtime_activation_layer_record *record =
            &activation->records[index];
        unsigned long long token, column, elements =
            tokens * layer->hidden_dimension;
        record->ordinal = index;
        record->layer_index = layer->layer_index;
        record->width = record->stride = layer->hidden_dimension;
        record->payload_offset = offset * sizeof(float);
        record->payload_bytes = elements * sizeof(float);
        rc = yvex_runtime_activation_layer_identity_compute(
            binding->attention_plan_identity, index, layer,
            YVEX_ATTENTION_OPERATION_CORE, record->layer_identity, err);
        if (rc != YVEX_OK) goto fail;
        for (token = 0ull; token < tokens; ++token)
            for (column = 0ull; column < layer->hidden_dimension; ++column)
                activation->payload[offset + token * layer->hidden_dimension +
                                    column] =
                    live_activation_value(index, start + token, column,
                                          variant);
        offset += elements;
    }
    rc = yvex_runtime_activation_input_seal(
        &activation->summary, activation->records, activation->payload, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_activation_input_open_memory(
            &activation->input, &activation->summary, activation->records,
            activation->payload, err);
    if (rc == YVEX_OK) return YVEX_OK;
fail:
    live_activation_close(activation);
    return rc;
}

static int live_cancel_requested(void *context)
{
    live_cancel *cancel = (live_cancel *)context;
    return !cancel || !cancel->result ||
           cancel->result->chunk_count >= cancel->cancel_after_chunks;
}

static int live_session_open(yvex_runtime_execution_session **session,
                             yvex_runtime_model *model,
                             yvex_backend_kind backend, yvex_error *err)
{
    yvex_runtime_session_open_request request;
    yvex_runtime_model_failure failure;
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    request.backend = backend;
    return yvex_runtime_session_open(session, model, &request, &failure, err);
}

static int live_execute(yvex_runtime_model *model,
                        yvex_runtime_execution_session *session,
                        const live_activation *activation,
                        yvex_backend_kind backend,
                        unsigned long long chunk_tokens,
                        yvex_runtime_activation_prefill_result *result,
                        live_cancel *cancel, yvex_error *err)
{
    yvex_runtime_activation_prefill_request request;
    yvex_runtime_model_failure failure;
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    request.backend = backend;
    request.mode = YVEX_RUNTIME_MODE_EAGER;
    request.operation_scope = YVEX_RUNTIME_SCOPE_ATTENTION_CORE;
    request.chunk_tokens = chunk_tokens;
    request.context_capacity = 2ull;
    if (cancel) {
        request.cancel_requested = live_cancel_requested;
        request.cancel_context = cancel;
    }
    return yvex_runtime_activation_prefill_execute(
        model, session, activation->input, &request, result, &failure, err);
}

static int live_session_close(yvex_runtime_execution_session **session,
                              yvex_error *err)
{
    int rc = yvex_runtime_session_close(session, err);
    return rc == YVEX_OK && !*session ? YVEX_OK : rc;
}

/* Prove whole-chunk, subchunk, clear, cancellation, and rollback on CPU. */
static int live_cpu_suite(yvex_runtime_model *model,
                          const live_activation *full,
                          const live_activation *prefix_a,
                          const live_activation *prefix_b,
                          const live_activation *suffix, yvex_error *err)
{
    yvex_runtime_execution_session *whole = NULL, *chunked = NULL;
    yvex_runtime_execution_session *causal_a = NULL, *causal_b = NULL;
    yvex_runtime_execution_session *cancelled = NULL, *failed = NULL;
    yvex_runtime_activation_prefill_result whole_result, chunked_result;
    yvex_runtime_activation_prefill_result prefix_result, suffix_a, suffix_b;
    yvex_runtime_activation_prefill_result clear_result, cancelled_result;
    yvex_runtime_activation_prefill_result failed_result;
    yvex_runtime_model_failure failure;
    live_cancel cancellation;
    yvex_error cleanup_error;
    int rc = YVEX_OK, close_rc;

#define OPEN_SESSION(owner)                                                    \
    do {                                                                       \
        if (rc == YVEX_OK)                                                     \
            rc = live_session_open(&(owner), model, YVEX_BACKEND_KIND_CPU, err); \
    } while (0)
    OPEN_SESSION(whole);
    OPEN_SESSION(chunked);
    OPEN_SESSION(causal_a);
    OPEN_SESSION(causal_b);
    OPEN_SESSION(cancelled);
    OPEN_SESSION(failed);
#undef OPEN_SESSION
    if (rc == YVEX_OK)
        rc = live_execute(model, whole, full, YVEX_BACKEND_KIND_CPU, 2ull,
                          &whole_result, NULL, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, chunked, full, YVEX_BACKEND_KIND_CPU, 1ull,
                          &chunked_result, NULL, err);
    if (rc == YVEX_OK &&
        (strcmp(whole_result.tensor_output_digest,
                chunked_result.tensor_output_digest) != 0 ||
         strcmp(whole_result.persistent_state_digest,
                chunked_result.persistent_state_digest) != 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.prefill.chunk-equivalence",
                       "whole and repeated subchunk output/state digests differ");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) printf("prefill_cpu_chunk_equivalence_stage=pass\n");
    memset(&failure, 0, sizeof(failure));
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_reset_persistent_state(
            chunked, &failure, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, chunked, full, YVEX_BACKEND_KIND_CPU, 2ull,
                          &clear_result, NULL, err);
    if (rc == YVEX_OK &&
        strcmp(clear_result.tensor_output_digest,
               whole_result.tensor_output_digest) != 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.prefill.clear",
                       "cleared session output differs from a fresh session");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) printf("prefill_cpu_clear_stage=pass\n");
    if (rc == YVEX_OK)
        rc = live_execute(model, causal_a, prefix_a, YVEX_BACKEND_KIND_CPU,
                          1ull, &prefix_result, NULL, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, causal_a, suffix, YVEX_BACKEND_KIND_CPU, 1ull,
                          &suffix_a, NULL, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, causal_b, prefix_b, YVEX_BACKEND_KIND_CPU,
                          1ull, &prefix_result, NULL, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, causal_b, suffix, YVEX_BACKEND_KIND_CPU, 1ull,
                          &suffix_b, NULL, err);
    if (rc == YVEX_OK &&
        (strcmp(suffix_a.tensor_output_digest,
                suffix_b.tensor_output_digest) == 0 ||
         strcmp(suffix_a.persistent_state_digest,
                suffix_b.persistent_state_digest) == 0)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.prefill.causality",
                       "distinct committed prefixes did not change the suffix");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) printf("prefill_cpu_causality_stage=pass\n");
    cancellation.result = &cancelled_result;
    cancellation.cancel_after_chunks = 1ull;
    if (rc == YVEX_OK) {
        int cancel_rc = live_execute(
            model, cancelled, full, YVEX_BACKEND_KIND_CPU, 1ull,
            &cancelled_result, &cancellation, err);
        if (cancel_rc != YVEX_ERR_CANCELLED ||
            cancelled_result.committed_prefix != 1ull) {
            yvex_error_set(err, YVEX_ERR_STATE, "test.prefill.cancellation",
                           "cancellation did not preserve one committed chunk");
            rc = YVEX_ERR_STATE;
        } else
            yvex_error_clear(err);
    }
    if (rc == YVEX_OK) printf("prefill_cpu_cancellation_stage=pass\n");
    if (rc == YVEX_OK)
        (void)setenv("YVEX_TEST_RUNTIME_PREFILL_FAIL_LAYER", "7", 1);
    if (rc == YVEX_OK) {
        int fail_rc = live_execute(
            model, failed, full, YVEX_BACKEND_KIND_CPU, 1ull,
            &failed_result, NULL, err);
        (void)unsetenv("YVEX_TEST_RUNTIME_PREFILL_FAIL_LAYER");
        if (fail_rc != YVEX_ERR_STATE || failed_result.committed_prefix != 0ull) {
            yvex_error_set(err, YVEX_ERR_STATE, "test.prefill.rollback",
                           "layer failure published partial chunk state");
            rc = YVEX_ERR_STATE;
        } else
            yvex_error_clear(err);
    }
    if (rc == YVEX_OK) printf("prefill_cpu_rollback_stage=pass\n");
    if (rc == YVEX_OK)
        printf("prefill_cpu_whole_subchunk_equivalent=1\n"
               "prefill_cpu_causal_history=1\n"
               "prefill_cpu_clear_reuse=1\n"
               "prefill_cpu_cancelled_prefix=1\n"
               "prefill_cpu_layer_failure_rollback=1\n");
#define CLOSE_SESSION(owner)                                                   \
    do {                                                                       \
        yvex_error_clear(&cleanup_error);                                      \
        close_rc = live_session_close(&(owner), &cleanup_error);               \
        if (rc == YVEX_OK && close_rc != YVEX_OK) {                            \
            rc = close_rc;                                                     \
            *err = cleanup_error;                                              \
        }                                                                      \
    } while (0)
    CLOSE_SESSION(failed);
    CLOSE_SESSION(cancelled);
    CLOSE_SESSION(causal_b);
    CLOSE_SESSION(causal_a);
    CLOSE_SESSION(chunked);
    CLOSE_SESSION(whole);
#undef CLOSE_SESSION
    return rc;
}

static int live_cuda_suite(yvex_runtime_model *model,
                           const live_activation *full, yvex_error *err)
{
    yvex_runtime_execution_session *cpu = NULL, *cuda = NULL;
    yvex_runtime_activation_prefill_result cpu_result, cuda_result;
    yvex_error cleanup_error;
    int rc = live_session_open(&cpu, model, YVEX_BACKEND_KIND_CPU, err);
    int close_rc;
    if (rc == YVEX_OK)
        rc = live_execute(model, cpu, full, YVEX_BACKEND_KIND_CPU, 1ull,
                          &cpu_result, NULL, err);
    if (rc == YVEX_OK)
        rc = live_session_open(&cuda, model, YVEX_BACKEND_KIND_CUDA, err);
    if (rc == YVEX_OK)
        rc = live_execute(model, cuda, full, YVEX_BACKEND_KIND_CUDA, 1ull,
                          &cuda_result, NULL, err);
    if (rc == YVEX_OK &&
        (strcmp(cpu_result.tensor_output_digest,
                cuda_result.tensor_output_digest) != 0 ||
         strcmp(cpu_result.persistent_state_digest,
                cuda_result.persistent_state_digest) != 0 ||
         cpu_result.layers_executed != 86ull ||
         cuda_result.layers_executed != 86ull ||
         cpu_result.bindings_executed != 1268ull ||
         cuda_result.bindings_executed != 1268ull)) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "test.prefill.cpu-cuda",
                       "CPU and CUDA output/state or scale counters differ");
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK)
        printf("prefill_cpu_cuda_output_equal=1\n"
               "prefill_cpu_cuda_state_equal=1\n"
               "prefill_chunks=2\n"
               "prefill_layers_per_chunk=43\n"
               "prefill_bindings_per_chunk=634\n");
    yvex_error_clear(&cleanup_error);
    close_rc = live_session_close(&cuda, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup_error;
    }
    yvex_error_clear(&cleanup_error);
    close_rc = live_session_close(&cpu, &cleanup_error);
    if (rc == YVEX_OK && close_rc != YVEX_OK) {
        rc = close_rc;
        *err = cleanup_error;
    }
    return rc;
}

int main(int argc, char **argv)
{
    yvex_runtime_model_open_request request;
    yvex_runtime_model_failure failure;
    yvex_runtime_model *model = NULL;
    live_activation full, prefix_a, prefix_b, suffix;
    yvex_error err;
    int rc;

    if (argc != 4) {
        fprintf(stderr, "usage: %s ARTIFACT RUNTIME_BINDING ACTIVATION_OUTPUT\n",
                argv[0]);
        return 2;
    }
    (void)setvbuf(stdout, NULL, _IOLBF, 0);
    memset(&request, 0, sizeof(request));
    memset(&failure, 0, sizeof(failure));
    memset(&full, 0, sizeof(full));
    memset(&prefix_a, 0, sizeof(prefix_a));
    memset(&prefix_b, 0, sizeof(prefix_b));
    memset(&suffix, 0, sizeof(suffix));
    request.artifact_path = argv[1];
    request.runtime_binding_path = argv[2];
    request.target_id = "deepseek4-v4-flash-dspark";
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc != YVEX_OK || !model) {
        live_fail("model-open", rc, &err);
        return 1;
    }
    rc = live_activation_open(&full, model, 0ull, 2ull, 0u, &err);
    if (rc == YVEX_OK)
        rc = live_activation_open(&prefix_a, model, 0ull, 1ull, 0u, &err);
    if (rc == YVEX_OK)
        rc = live_activation_open(&prefix_b, model, 0ull, 1ull, 1u, &err);
    if (rc == YVEX_OK)
        rc = live_activation_open(&suffix, model, 1ull, 1ull, 0u, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_activation_input_write(
            argv[3], &full.summary, full.records, full.payload, &err);
    if (rc == YVEX_OK)
        rc = live_cpu_suite(
            model, &full, &prefix_a, &prefix_b, &suffix, &err);
    if (rc == YVEX_OK && !getenv("YVEX_PREFILL_CPU_ONLY"))
        rc = live_cuda_suite(model, &full, &err);
    if (rc != YVEX_OK) live_fail("execution", rc, &err);
    live_activation_close(&suffix);
    live_activation_close(&prefix_b);
    live_activation_close(&prefix_a);
    live_activation_close(&full);
    yvex_runtime_model_close(&model);
    if (model) {
        fprintf(stderr, "prefill_live step=model-close status=state\n");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        printf("prefill_activation_file=%s\n"
               "prefill_input_class=typed_activation_tensor_file\n"
               "prefill_full_model_ready=0\n"
               "prefill_generation_ready=0\n", argv[3]);
    }
    return rc == YVEX_OK ? 0 : 1;
}
