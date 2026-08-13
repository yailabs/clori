/*
 * All generated files live under one uniquely owned /tmp directory. The fixture proves storage
 * semantics with a tiny admitted GGUF.
 */
#define _GNU_SOURCE
#include <yvex/artifact.h>
#include <yvex/gguf.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/artifact_lowering.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/runtime_prefix.h>
#include <yvex/internal/runtime_state_store.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/runtime/private.h"
#include "tests/test.h"

#define TEST_BINDING_HEADER_BYTES 88u
#define TEST_BINDING_IDENTITY_OFFSET 24u
#define TEST_BINDING_CAPABILITY_FIELDS 41u

typedef struct {
    yvex_artifact *artifact;
    yvex_gguf *gguf;
    yvex_tensor_table *tensors;
    yvex_artifact_lowering_map *map;
    yvex_complete_artifact_admission admission;
    yvex_artifact_physical_compatibility compatibility;
    yvex_materialization_plan *materialization_plan;
    yvex_materialization_session *materialization;
    yvex_runtime_descriptor *descriptor;
    yvex_semantic_model_ir *semantic_model;
    yvex_operator_graph_ir *operator_graph;
    yvex_physical_execution_ir *physical_execution;
    yvex_compiled_model_plan *compiled_plan;
    yvex_attention_plan *attention;
    yvex_runtime_capabilities capabilities;
    yvex_transformer_family_policy transformer_policy;
    yvex_logits_family_policy logits_policy;
    yvex_speculation_family_policy speculation_policy;
} binding_fixture;

static int runtime_model_open_fixture(
    const binding_fixture *fixture,
    const yvex_runtime_binding_prepare_result *prepared,
    yvex_runtime_model **model, yvex_runtime_model_failure *failure,
    yvex_error *err);

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned int ready;
    int released;
} runtime_thread_gate;

typedef struct {
    yvex_runtime_execution_session *session;
    runtime_thread_gate *gate;
    int begin_status;
    yvex_runtime_model_failure_code failure_code;
} runtime_execute_thread;

typedef struct {
    yvex_runtime_model *model;
    runtime_thread_gate *gate;
    yvex_runtime_execution_session *session;
    int open_status;
    yvex_runtime_model_failure_code failure_code;
} runtime_open_thread;

static int runtime_thread_gate_init(runtime_thread_gate *gate)
{
    memset(gate, 0, sizeof(*gate));
    if (pthread_mutex_init(&gate->mutex, NULL) != 0)
        return 0;
    if (pthread_cond_init(&gate->condition, NULL) != 0) {
        (void)pthread_mutex_destroy(&gate->mutex);
        return 0;
    }
    return 1;
}

static void runtime_thread_gate_wait_ready(runtime_thread_gate *gate, unsigned int count)
{
    (void)pthread_mutex_lock(&gate->mutex);
    while (gate->ready < count)
        (void)pthread_cond_wait(&gate->condition, &gate->mutex);
    (void)pthread_mutex_unlock(&gate->mutex);
}

static void runtime_thread_gate_release(runtime_thread_gate *gate)
{
    (void)pthread_mutex_lock(&gate->mutex);
    gate->released = 1;
    (void)pthread_cond_broadcast(&gate->condition);
    (void)pthread_mutex_unlock(&gate->mutex);
}

static void runtime_thread_gate_destroy(runtime_thread_gate *gate)
{
    (void)pthread_cond_destroy(&gate->condition);
    (void)pthread_mutex_destroy(&gate->mutex);
}

static void *runtime_execute_thread_main(void *argument)
{
    runtime_execute_thread *thread = (runtime_execute_thread *)argument;
    yvex_runtime_model_failure failure;
    yvex_error err;

    memset(&failure, 0, sizeof(failure));
    thread->begin_status = yvex_runtime_session_begin(thread->session, &failure, &err);
    thread->failure_code = failure.code;
    (void)pthread_mutex_lock(&thread->gate->mutex);
    thread->gate->ready++;
    (void)pthread_cond_broadcast(&thread->gate->condition);
    while (!thread->gate->released)
        (void)pthread_cond_wait(&thread->gate->condition, &thread->gate->mutex);
    (void)pthread_mutex_unlock(&thread->gate->mutex);
    if (thread->begin_status == YVEX_OK)
        thread->begin_status = yvex_runtime_session_finish(
            thread->session, YVEX_OK, &err);
    return NULL;
}

static void *runtime_open_thread_main(void *argument)
{
    runtime_open_thread *thread = (runtime_open_thread *)argument;
    yvex_runtime_session_open_request request;
    yvex_runtime_model_failure failure;
    yvex_error err;

    memset(&failure, 0, sizeof(failure));
    (void)pthread_mutex_lock(&thread->gate->mutex);
    thread->gate->ready++;
    (void)pthread_cond_broadcast(&thread->gate->condition);
    while (!thread->gate->released)
        (void)pthread_cond_wait(&thread->gate->condition, &thread->gate->mutex);
    (void)pthread_mutex_unlock(&thread->gate->mutex);
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    thread->open_status = yvex_runtime_session_open(
        &thread->session, thread->model, &request, &failure, &err);
    thread->failure_code = failure.code;
    return NULL;
}

static const yvex_graph_execution_binding *runtime_fixture_execution(void)
{
    return yvex_graph_execution_find(0ull, 0ull, "deepseek4-v4-flash-dspark");
}

static const yvex_family_compiler_adapter *runtime_fixture_compiler(void)
{
    return yvex_compiler_family_deepseek_v4();
}

static int runtime_state_prepare_fixture(
    const yvex_runtime_model *model,
    const yvex_attention_state_provider *provider,
    unsigned long long layer_index, yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_attention_summary *summary = view
        ? yvex_attention_plan_summary(view->attention) : NULL;
    const yvex_attention_layer_plan *layer = view
        ? yvex_attention_plan_layer_at(view->attention, layer_index) : NULL;
    yvex_attention_state_recipe_request request = {0};
    yvex_attention_state_recipe recipe = {0};
    int rc;

    if (!summary || !layer) return YVEX_ERR_STATE;
    request.layer_ordinal = layer_index;
    request.final_position = layer->sliding_window
                                 ? layer->sliding_window - 1ull : 1ull;
    request.attention_plan_identity = summary->attention_plan_identity;
    rc = yvex_attention_state_recipe_build(
        layer, &request, &recipe, failure, err);
    if (rc != YVEX_OK) return rc;
    return provider && provider->prepare
               ? provider->prepare(provider->context, layer_index, &recipe,
                                   NULL, failure, err)
               : YVEX_ERR_STATE;
}

static int runtime_state_begin_fixture(
    const yvex_runtime_model *model,
    const yvex_attention_state_provider *provider,
    unsigned long long layer_index, unsigned long long token_position,
    unsigned long long token_count, yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_attention_layer_plan *layer = view
        ? yvex_attention_plan_layer_at(view->attention, layer_index) : NULL;
    const yvex_attention_history_view *history = NULL;
    return provider && provider->begin && layer
               ? provider->begin(provider->context, layer_index, layer, NULL,
                                 token_position, token_count, NULL, &history,
                                 failure, err)
               : YVEX_ERR_STATE;
}

typedef struct injected_state_control injected_state_control;
typedef struct {
    injected_state_control *control;
    yvex_graph_attention_state_summary summary;
    yvex_execution_capacity_plan capacity;
    yvex_attention_state_recipe recipe;
    int commit_prepared;
} injected_state;
struct injected_state_control {
    injected_state *active;
    unsigned int opens, summaries, configures, commits, aborts;
    unsigned int invalidations, releases, discards;
    int fail_open_after_publish, malformed_success, fail_discard_once;
    int fail_summary_once, fail_configure_once, fail_commit_once, fail_abort_once;
    int fail_invalidate_once, fail_release_once;
};

static int injected_state_prepare(
    void *context, unsigned long long layer_index,
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_history_view *initial_history,
    yvex_attention_failure *failure, yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)initial_history;
    (void)failure;
    if (!state || !recipe || recipe->layer_index != layer_index) return YVEX_ERR_INVALID_ARG;
    state->summary.layer_count = 1ull;
    state->summary.prepared_layer_count = 1ull;
    state->summary.allocated_bytes = 1ull;
    state->recipe = *recipe;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_summary(void *context,
                                  yvex_graph_attention_state_summary *out,
                                  yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    if (!state || !out) return YVEX_ERR_INVALID_ARG;
    state->control->summaries++;
    if (state->control->fail_summary_once) {
        state->control->fail_summary_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.summary",
                       "injected state summary failure");
        return YVEX_ERR_STATE;
    }
    *out = state->summary;
    yvex_error_clear(err);
    return YVEX_OK;
}

static const yvex_execution_capacity_plan *injected_state_capacity(void *context)
{
    injected_state *state = (injected_state *)context;
    return state && state->capacity.schema_version ? &state->capacity : NULL;
}

static const yvex_attention_state_recipe *injected_state_recipe(
    void *context, unsigned long long layer_index)
{
    injected_state *state = (injected_state *)context;
    return state && layer_index == 0ull && state->recipe.schema_version
               ? &state->recipe
               : NULL;
}

static int injected_state_identity(void *context, unsigned long long layer_index,
                                   char output[YVEX_SHA256_HEX_CAP],
                                   yvex_error *err)
{
    (void)context;
    (void)layer_index;
    memset(output, 'a', YVEX_SHA256_HEX_CAP - 1u);
    output[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    yvex_error_clear(err);
    return YVEX_OK;
}
static const yvex_attention_history_view injected_state_history = {.immutable = 1};

static const yvex_attention_history_view *injected_state_view(
    void *context, unsigned long long layer_index, yvex_attention_state_view_kind kind)
{
    (void)context;
    (void)layer_index;
    (void)kind;
    return &injected_state_history;
}
static int injected_state_begin(
    void *context, unsigned long long layer_index,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *initial_history,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation,
    const yvex_attention_history_view **history,
    yvex_attention_failure *failure, yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)layer_index;
    (void)layer;
    (void)token_position;
    (void)token_count;
    (void)cancellation;
    (void)failure;
    if (!state || !history) return YVEX_ERR_INVALID_ARG;
    state->summary.transaction_active = 1;
    state->summary.candidate_active = 1;
    *history = initial_history;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_select_prefix(
    void *context, unsigned long long prefix_count,
    unsigned long long extension_count, yvex_attention_failure *failure,
    yvex_error *err)
{
    (void)context;
    (void)prefix_count;
    (void)extension_count;
    (void)failure;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_stage(
    void *context, const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)publication;
    (void)cancellation;
    (void)failure;
    if (!state || !state->summary.transaction_active) return YVEX_ERR_STATE;
    state->summary.candidate_active = 0;
    state->summary.staged_layer_count = 1ull;
    if (state_delta_identity) {
        memset(state_delta_identity, 'b', YVEX_SHA256_HEX_CAP - 1u);
        state_delta_identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_prepare_commit(void *context,
                                         yvex_attention_failure *failure,
                                         yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)failure;
    state->control->commits++;
    if (state->control->fail_commit_once) {
        state->control->fail_commit_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.commit", "injected commit failure");
        return YVEX_ERR_STATE;
    }
    state->commit_prepared = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void injected_state_publish_commit(void *context) {
    injected_state *state = (injected_state *)context;
    if (!state || !state->commit_prepared) return;
    state->summary.transaction_active = 0;
    state->summary.staged_layer_count = 0ull;
    state->summary.commit_count++;
    state->commit_prepared = 0;
}

static void injected_state_cancel_commit(void *context) {
    injected_state *state = (injected_state *)context;
    if (state) state->commit_prepared = 0;
}

static int injected_state_commit(void *context, yvex_attention_failure *failure,
                                 yvex_error *err) {
    int rc = injected_state_prepare_commit(context, failure, err);
    if (rc == YVEX_OK) injected_state_publish_commit(context);
    return rc;
}

static int injected_state_abort(void *context, yvex_attention_failure *failure,
                                yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)failure;
    state->control->aborts++;
    if (state->control->fail_abort_once) {
        state->control->fail_abort_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.abort", "injected abort failure");
        return YVEX_ERR_STATE;
    }
    state->summary.transaction_active = 0;
    state->summary.candidate_active = 0;
    state->summary.staged_layer_count = 0ull;
    state->summary.abort_count++;
    return YVEX_OK;
}

static int injected_state_reset(void *context, yvex_attention_failure *failure,
                                yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)failure;
    if (!state) return YVEX_ERR_INVALID_ARG;
    state->summary.transaction_active = 0;
    state->summary.candidate_active = 0;
    state->summary.staged_layer_count = 0ull;
    state->summary.reset_count++;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_configure_pages(
    void *context, const yvex_execution_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    (void)failure;
    if (!state || !capacity ||
        capacity->schema_version != YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1)
        return YVEX_ERR_INVALID_ARG;
    state->control->configures++;
    if (state->control->fail_configure_once) {
        state->control->fail_configure_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.pages",
                       "injected state paging failure");
        return YVEX_ERR_STATE;
    }
    state->summary.paged = 1;
    state->summary.paging_configured = 1;
    state->capacity = *capacity;
    yvex_core_text_copy(
        state->summary.capacity_plan_identity,
        sizeof(state->summary.capacity_plan_identity), capacity->identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_invalidate(void *context, yvex_error *err)
{
    injected_state *state = (injected_state *)context;
    state->control->invalidations++;
    if (state->control->fail_invalidate_once) {
        state->control->fail_invalidate_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.invalidate",
                       "injected invalidate failure");
        return YVEX_ERR_STATE;
    }
    state->summary.invalidated = 1;
    state->summary.cancelled = 1;
    state->summary.generation++;
    return YVEX_OK;
}

static int injected_state_restore(
    void *context, const yvex_attention_state_checkpoint *checkpoint,
    yvex_attention_failure *failure, yvex_error *err)
{
    (void)context;
    (void)checkpoint;
    (void)failure;
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "test.state.restore",
                   "injected provider does not persist state");
    return YVEX_ERR_UNSUPPORTED;
}

static int injected_state_prefix_capture(
    void *context, unsigned long long maximum_bytes,
    yvex_attention_state_prefix **prefix, yvex_attention_failure *failure,
    yvex_error *err)
{
    (void)context;
    (void)maximum_bytes;
    (void)failure;
    if (prefix) *prefix = NULL;
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "test.state.prefix",
                   "injected provider does not share prefixes");
    return YVEX_ERR_UNSUPPORTED;
}

static int injected_state_prefix_attach(
    void *context, const yvex_attention_state_prefix *prefix,
    yvex_attention_failure *failure, yvex_error *err)
{
    (void)context;
    (void)prefix;
    (void)failure;
    yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "test.state.prefix",
                   "injected provider does not attach prefixes");
    return YVEX_ERR_UNSUPPORTED;
}

static int injected_state_release(void **context, yvex_error *err)
{
    injected_state *state = context ? (injected_state *)*context : NULL;
    if (!state) return YVEX_OK;
    state->control->releases++;
    if (state->control->fail_release_once) {
        state->control->fail_release_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.release", "injected release failure");
        return YVEX_ERR_STATE;
    }
    state->control->active = NULL;
    free(state);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_factory_open(
    void *context, const yvex_attention_plan *plan,
    unsigned long long maximum_host_bytes,
    yvex_attention_state_provider *out,
    yvex_attention_failure *failure, yvex_error *err)
{
    injected_state_control *control = (injected_state_control *)context;
    injected_state *state;
    (void)plan;
    (void)maximum_host_bytes;
    (void)failure;
    if (!control || !out || control->active) return YVEX_ERR_INVALID_ARG;
    state = (injected_state *)calloc(1u, sizeof(*state));
    if (!state) return YVEX_ERR_NOMEM;
    state->control = control;
    state->summary.schema_version = YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V4;
    state->summary.sealed = 1;
    memset(state->summary.state_layout_identity, 'c', YVEX_SHA256_HEX_CAP - 1u);
    state->summary.state_layout_identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    control->active = state;
    control->opens++;
    *out = (yvex_attention_state_provider){
        .schema_version = YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V8,
        .context = state,
        .configure_pages = injected_state_configure_pages,
        .prepare = injected_state_prepare,
        .summary = injected_state_summary,
        .capacity = injected_state_capacity,
        .recipe = injected_state_recipe,
        .view = injected_state_view,
        .identity = injected_state_identity,
        .begin = injected_state_begin,
        .stage = injected_state_stage,
        .select_prefix = injected_state_select_prefix,
        .prepare_commit = injected_state_prepare_commit,
        .publish_commit = injected_state_publish_commit,
        .cancel_commit = injected_state_cancel_commit,
        .commit = injected_state_commit,
        .abort = injected_state_abort,
        .reset = injected_state_reset,
        .restore = injected_state_restore,
        .prefix_capture = injected_state_prefix_capture,
        .prefix_attach = injected_state_prefix_attach,
        .invalidate = injected_state_invalidate,
        .release = injected_state_release};
    if (control->malformed_success) {
        control->malformed_success = 0;
        out->commit = NULL;
    }
    if (control->fail_open_after_publish) {
        control->fail_open_after_publish = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.open",
                       "injected factory failure after candidate publication");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int injected_state_factory_discard(
    void *context, yvex_attention_state_provider *candidate, yvex_error *err)
{
    injected_state_control *control = (injected_state_control *)context;
    injected_state *state;
    if (!control || !candidate) return YVEX_ERR_INVALID_ARG;
    control->discards++;
    if (control->fail_discard_once) {
        control->fail_discard_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.discard",
                       "injected factory candidate discard failure");
        return YVEX_ERR_STATE;
    }
    state = candidate->context ? (injected_state *)candidate->context : control->active;
    if (state && state != control->active) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.state.discard",
                       "factory candidate does not own the active state");
        return YVEX_ERR_STATE;
    }
    free(state);
    control->active = NULL;
    memset(candidate, 0, sizeof(*candidate));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int copy_regular_file(const char *source, const char *destination)
{
    unsigned char buffer[4096];
    int input = -1, output = -1;
    int result = 0;

    input = open(source, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (input < 0) goto done;
    output = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (output < 0) goto done;
    for (;;) {
        ssize_t got = read(input, buffer, sizeof(buffer));
        size_t offset = 0u;
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) goto done;
        if (got == 0) break;
        while (offset < (size_t)got) {
            ssize_t written = write(output, buffer + offset, (size_t)got - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) goto done;
            offset += (size_t)written;
        }
    }
    result = fsync(output) == 0;

done:
    if (output >= 0) (void)close(output);
    if (input >= 0) (void)close(input);
    if (!result) (void)unlink(destination);
    return result;
}

static int regular_files_equal(const char *left_path, const char *right_path)
{
    yvex_core_file_result result;
    unsigned char *left = NULL, *right = NULL;
    size_t left_count = 0u, right_count = 0u;
    yvex_error err;
    int equal = 0;

    memset(&result, 0, sizeof(result));
    if (yvex_core_file_read_snapshot(left_path, 64u * 1024u * 1024u,
                                     &left, &left_count, &result, &err) != YVEX_OK)
        goto done;
    memset(&result, 0, sizeof(result));
    if (yvex_core_file_read_snapshot(right_path, 64u * 1024u * 1024u,
                                     &right, &right_count, &result, &err) != YVEX_OK)
        goto done;
    equal = left_count == right_count && memcmp(left, right, left_count) == 0;

done:
    free(right);
    free(left);
    return equal;
}

static int test_binding_u64(const unsigned char *bytes, size_t count, size_t offset,
                            unsigned long long *value)
{
    unsigned int index;

    if (!bytes || !value || offset > count || 8u > count - offset) return 0;
    *value = 0ull;
    for (index = 0u; index < 8u; ++index)
        *value |= (unsigned long long)bytes[offset + index] << (index * 8u);
    return 1;
}

static int test_binding_text_skip(const unsigned char *bytes, size_t count, size_t *offset,
                                  size_t *payload_offset)
{
    unsigned long long length;

    if (!offset || !test_binding_u64(bytes, count, *offset, &length) ||
        length > (unsigned long long)SIZE_MAX || *offset + 8u > count ||
        (size_t)length > count - (*offset + 8u))
        return 0;
    if (payload_offset) *payload_offset = *offset + 8u;
    *offset += 8u + (size_t)length;
    return 1;
}

static int test_binding_u64_skip(size_t count, size_t *offset, size_t field_count)
{
    if (!offset || *offset > count || field_count > (count - *offset) / 8u) return 0;
    *offset += field_count * 8u;
    return 1;
}

static int test_binding_offsets(const unsigned char *file, size_t count,
                                size_t *format_text, size_t *format_version,
                                size_t *capability_value, size_t *compatibility_schema)
{
    size_t offset = TEST_BINDING_HEADER_BYTES;
    unsigned int index;

    if (count < TEST_BINDING_HEADER_BYTES ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        24u > count - offset)
        return 0;
    offset += 24u;
    if (!test_binding_text_skip(file, count, &offset, format_text) ||
        8u > count - offset)
        return 0;
    *format_version = offset;
    offset += 8u;
    for (index = 0u; index < 5u; ++index)
        if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    *capability_value = offset;
    if (!test_binding_u64_skip(
            count, &offset, TEST_BINDING_CAPABILITY_FIELDS)) return 0;
    *compatibility_schema = offset;
    return 8u <= count - offset;
}

static int test_binding_graph_identity_offsets(
    const unsigned char *file, size_t count,
    size_t *semantic_identity, size_t *executable_identity)
{
    size_t offset = TEST_BINDING_HEADER_BYTES;

    if (!semantic_identity || !executable_identity ||
        count < TEST_BINDING_HEADER_BYTES ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_u64_skip(count, &offset, 3u) ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_u64_skip(count, &offset, 1u) ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_text_skip(file, count, &offset, semantic_identity) ||
        !test_binding_text_skip(file, count, &offset, executable_identity))
        return 0;
    return *semantic_identity <= count && 64u <= count - *semantic_identity &&
           *executable_identity <= count && 64u <= count - *executable_identity;
}

static int test_binding_material_count_offset(const unsigned char *file, size_t count,
                                              size_t *material_count)
{
    size_t offset = TEST_BINDING_HEADER_BYTES;
    unsigned int index;

    if (!material_count || count < TEST_BINDING_HEADER_BYTES ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_u64_skip(count, &offset, 3u) ||
        !test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_u64_skip(count, &offset, 1u))
        return 0;
    for (index = 0u; index < 5u; ++index)
        if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    if (!test_binding_text_skip(file, count, &offset, NULL) ||
        !test_binding_u64_skip(count, &offset, TEST_BINDING_CAPABILITY_FIELDS) ||
        !test_binding_u64_skip(count, &offset, 7u))
        return 0;
    for (index = 0u; index < 11u; ++index)
        if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    if (!test_binding_u64_skip(count, &offset, 8u) ||
        !test_binding_u64_skip(count, &offset, 8u))
        return 0;
    for (index = 0u; index < 11u; ++index)
        if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    if (!test_binding_u64_skip(count, &offset, 9u)) return 0;
    for (index = 0u; index < 2u; ++index)
        if (!test_binding_text_skip(file, count, &offset, NULL)) return 0;
    if (!test_binding_u64_skip(
            count, &offset, 23u + 2u * YVEX_MATERIALIZATION_QTYPE_CAP))
        return 0;
    *material_count = offset;
    return 8u <= count - offset;
}

static void test_binding_put_u64(unsigned char *bytes, size_t offset,
                                 unsigned long long value)
{
    unsigned int index;
    for (index = 0u; index < 8u; ++index)
        bytes[offset + index] = (unsigned char)(value >> (index * 8u));
}

static void test_binding_put_u32(unsigned char *bytes, size_t offset,
                                 unsigned int value)
{
    unsigned int index;
    for (index = 0u; index < 4u; ++index)
        bytes[offset + index] = (unsigned char)(value >> (index * 8u));
}

static int rewrite_state_file_fixture(const char *path, size_t offset,
                                      unsigned int mode,
                                      unsigned long long value)
{
    yvex_core_file_result snapshot_result = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned char *snapshot = NULL;
    size_t snapshot_count = 0u, cursor = 0u;
    int descriptor = -1, result = 0;

    if (!path ||
        yvex_core_file_read_snapshot(path, 64u * 1024u * 1024u,
                                     &snapshot, &snapshot_count,
                                     &snapshot_result, NULL) != YVEX_OK ||
        snapshot_count < YVEX_SHA256_DIGEST_BYTES ||
        mode > 2u || offset >= snapshot_count - YVEX_SHA256_DIGEST_BYTES ||
        (mode == 0u &&
         8u > snapshot_count - YVEX_SHA256_DIGEST_BYTES - offset))
        goto done;
    if (mode == 1u)
        snapshot[offset] = snapshot[offset] == (unsigned char)'0'
                               ? (unsigned char)'1'
                               : (unsigned char)'0';
    else if (mode == 0u)
        test_binding_put_u64(snapshot, offset, value);
    else snapshot[offset] = 0xffu;
    if (mode < 2u) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update(&hash, snapshot,
                                snapshot_count - YVEX_SHA256_DIGEST_BYTES) ||
            !yvex_sha256_final(&hash, digest))
            goto done;
        memcpy(snapshot + snapshot_count - sizeof(digest), digest,
               sizeof(digest));
    }
    descriptor = open(path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) goto done;
    while (cursor < snapshot_count) {
        ssize_t written = pwrite(descriptor, snapshot + cursor,
                                 snapshot_count - cursor, (off_t)cursor);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) goto done;
        cursor += (size_t)written;
    }
    result = fsync(descriptor) == 0 && close(descriptor) == 0;
    descriptor = -1;

done:
    if (descriptor >= 0) (void)close(descriptor);
    free(snapshot);
    return result;
}

static int rewrite_attention_artifact_fixture(const char *path)
{
    static const char original_name[] = "token_embd.weight";
    static const char attention_name[] = "blk.0.attn_sinks.weight";
    yvex_artifact_options options = {0};
    yvex_core_file_result snapshot_result = {0};
    const yvex_gguf_tensor_info *tensor;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    unsigned char *snapshot = NULL, *rewritten = NULL;
    unsigned long long data_offset, encoded_bytes = 64ull * 4ull, name_length;
    size_t snapshot_count = 0u, rewritten_count, record_offset = SIZE_MAX, cursor;
    unsigned int matches = 0u;
    yvex_error err;
    int descriptor = -1, result = 0;

    if (!path) return 0;
    options.path = path;
    options.readonly = 1;
    if (yvex_artifact_open(&artifact, &options, &err) != YVEX_OK ||
        yvex_gguf_open(&gguf, artifact, &err) != YVEX_OK)
        goto done;
    tensor = yvex_gguf_tensor_at(gguf, 0ull);
    data_offset = yvex_gguf_tensor_data_offset(gguf);
    if (yvex_gguf_tensor_count(gguf) != 1ull || !tensor ||
        strcmp(tensor->name, original_name) != 0 || tensor->rank != 2u ||
        tensor->storage_bytes != 128ull || data_offset > (unsigned long long)SIZE_MAX ||
        encoded_bytes > (unsigned long long)SIZE_MAX - data_offset)
        goto done;
    yvex_gguf_close(gguf);
    gguf = NULL;
    yvex_artifact_close(artifact);
    artifact = NULL;
    if (yvex_core_file_read_snapshot(path, 4096u, &snapshot, &snapshot_count,
                                     &snapshot_result, &err) != YVEX_OK ||
        data_offset > snapshot_count)
        goto done;
    for (cursor = 0u; cursor + 8u + sizeof(original_name) - 1u <= (size_t)data_offset;
         ++cursor) {
        if (test_binding_u64(snapshot, (size_t)data_offset, cursor, &name_length) &&
            name_length == sizeof(original_name) - 1u &&
            memcmp(snapshot + cursor + 8u, original_name, sizeof(original_name) - 1u) == 0) {
            record_offset = cursor;
            matches++;
        }
    }
    if (matches != 1u) goto done;
    rewritten_count = (size_t)(data_offset + encoded_bytes);
    rewritten = calloc(rewritten_count, 1u);
    if (!rewritten) goto done;
    memcpy(rewritten, snapshot, snapshot_count < rewritten_count
                                    ? snapshot_count : rewritten_count);
    memset(rewritten + record_offset, 0, (size_t)data_offset - record_offset);
    cursor = record_offset;
    test_binding_put_u64(rewritten, cursor, sizeof(attention_name) - 1u);
    cursor += 8u;
    memcpy(rewritten + cursor, attention_name, sizeof(attention_name) - 1u);
    cursor += sizeof(attention_name) - 1u;
    test_binding_put_u32(rewritten, cursor, 1u);
    cursor += 4u;
    test_binding_put_u64(rewritten, cursor, 64ull);
    cursor += 8u;
    test_binding_put_u32(rewritten, cursor, YVEX_GGUF_QTYPE_F32);
    cursor += 4u;
    test_binding_put_u64(rewritten, cursor, 0ull);
    if (cursor + 8u > (size_t)data_offset) goto done;
    descriptor = open(path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0 || ftruncate(descriptor, (off_t)rewritten_count) != 0)
        goto done;
    cursor = 0u;
    while (cursor < rewritten_count) {
        ssize_t written = pwrite(descriptor, rewritten + cursor,
                                 rewritten_count - cursor, (off_t)cursor);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) goto done;
        cursor += (size_t)written;
    }
    result = fsync(descriptor) == 0 && close(descriptor) == 0;
    descriptor = -1;

done:
    if (descriptor >= 0) (void)close(descriptor);
    free(rewritten);
    free(snapshot);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return result;
}

static int test_binding_readdress(const char *path, unsigned char *file, size_t count,
                                  char addressed_path[YVEX_PATH_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char identity[YVEX_SHA256_HEX_CAP], directory[YVEX_PATH_CAP];
    const char *slash = strrchr(path, '/');
    const char *domain;
    unsigned long long schema;
    int descriptor, result = 0;
    size_t offset = 0u, directory_length;

    if (!slash || count < TEST_BINDING_HEADER_BYTES ||
        !test_binding_u64(file, count, 8u, &schema) ||
        schema != YVEX_RUNTIME_BINDING_SCHEMA_CURRENT)
        return 0;
    domain = "yvex.runtime.binding.v12";
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_u64(&hash, schema) ||
        !yvex_sha256_update(&hash, file + TEST_BINDING_HEADER_BYTES,
                            count - TEST_BINDING_HEADER_BYTES) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, identity);
    memcpy(file + TEST_BINDING_IDENTITY_OFFSET, identity, 64u);
    descriptor = open(path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) return 0;
    while (offset < count) {
        ssize_t written = pwrite(descriptor, file + offset, count - offset, (off_t)offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) goto done;
        offset += (size_t)written;
    }
    if (fsync(descriptor) != 0) goto done;
    directory_length = (size_t)(slash - path);
    if (directory_length >= sizeof(directory)) goto done;
    memcpy(directory, path, directory_length);
    directory[directory_length] = '\0';
    if (snprintf(addressed_path, YVEX_PATH_CAP, "%s/%s%s", directory, identity,
                 YVEX_RUNTIME_BINDING_SUFFIX) >= YVEX_PATH_CAP)
        goto done;
    result = close(descriptor) == 0 && rename(path, addressed_path) == 0;
    descriptor = -1;

done:
    if (descriptor >= 0) (void)close(descriptor);
    return result;
}

static int fixture_artifact_open(binding_fixture *fixture, const char *path)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;
    rc = yvex_artifact_open(&fixture->artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&fixture->gguf, fixture->artifact, &err);
    if (rc == YVEX_OK)
        rc = yvex_tensor_table_from_gguf(&fixture->tensors, fixture->gguf, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "runtime binding fixture open failed: %s\n", yvex_error_message(&err));
    return rc == YVEX_OK;
}

static int fixture_admission_build(binding_fixture *fixture)
{
    yvex_complete_artifact_admission *value = &fixture->admission;
    const yvex_gguf_tensor_info *tensor;
    yvex_artifact_file_identity identity;
    yvex_error err;

    memset(value, 0, sizeof(*value));
    memset(&identity, 0, sizeof(identity));
    if (yvex_artifact_identity_read_open(fixture->artifact, &identity, &err) != YVEX_OK)
        return 0;
    tensor = yvex_gguf_tensor_at(fixture->gguf, 0ull);
    if (yvex_gguf_tensor_count(fixture->gguf) != 1ull || !tensor ||
        tensor->storage_bytes != 256ull)
        return 0;
    value->artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    value->tensor_count = yvex_tensor_table_count(fixture->tensors);
    value->file_bytes = yvex_artifact_size(fixture->artifact);
    value->payload_bytes = tensor->storage_bytes;
    value->source_snapshot_identity = 0x1122334455667788ull;
    value->mapping_identity = 0x8877665544332211ull;
    value->tokenizer_complete = 1;
    value->native_reader_accepted = 1;
    value->official_reader_accepted = 1;
    value->payload_integrity_accepted = 1;
    value->materialization_input_ready = 1;
    value->artifact_identity_verified = 1;
    value->artifact_bytes_hashed = value->file_bytes;
    value->complete = 1;
    (void)snprintf(value->artifact_path, sizeof(value->artifact_path), "%s",
                   yvex_artifact_path(fixture->artifact));
    (void)snprintf(value->payload_identity, sizeof(value->payload_identity), "%064x", 2);
    (void)snprintf(value->transform_identity, sizeof(value->transform_identity), "%064x", 3);
    (void)snprintf(value->profile_identity, sizeof(value->profile_identity), "%064x", 4);
    (void)snprintf(value->profile_name, sizeof(value->profile_name), "fixture-profile-v1");
    (void)snprintf(value->quant_execution_identity,
                   sizeof(value->quant_execution_identity), "%064x", 5);
    (void)snprintf(value->payload_plan_identity,
                   sizeof(value->payload_plan_identity), "%064x", 6);
    (void)snprintf(value->payload_byte_identity,
                   sizeof(value->payload_byte_identity), "%064x", 7);
    (void)snprintf(value->writer_plan_identity,
                   sizeof(value->writer_plan_identity), "%064x", 8);
    (void)snprintf(value->artifact_identity, sizeof(value->artifact_identity), "%s",
                   identity.sha256);
    (void)snprintf(value->admission_identity, sizeof(value->admission_identity), "%064x", 10);
    (void)snprintf(value->official_reader_revision,
                   sizeof(value->official_reader_revision), "fixture-reader-v1");
    return yvex_artifact_snapshot_get(fixture->artifact, &value->file_snapshot, NULL) == YVEX_OK;
}

static void fixture_compatibility_build(binding_fixture *fixture)
{
    yvex_artifact_physical_compatibility *value = &fixture->compatibility;
    const yvex_complete_artifact_admission *admission = &fixture->admission;

    memset(value, 0, sizeof(*value));
    value->schema_version = YVEX_ARTIFACT_PHYSICAL_COMPATIBILITY_SCHEMA_VERSION;
    value->source_snapshot_identity = admission->source_snapshot_identity;
    value->mapping_identity = admission->mapping_identity;
    value->tensor_count = admission->tensor_count;
    value->tensors_compared = admission->tensor_count;
    value->payload_bytes = admission->payload_bytes;
    (void)snprintf(value->writer_plan_identity, sizeof(value->writer_plan_identity), "%064x", 11);
    (void)snprintf(value->admitted_writer_plan_identity,
                   sizeof(value->admitted_writer_plan_identity), "%s",
                   admission->writer_plan_identity);
    (void)snprintf(value->artifact_identity, sizeof(value->artifact_identity), "%s",
                   admission->artifact_identity);
    (void)snprintf(value->payload_identity, sizeof(value->payload_identity), "%s",
                   admission->payload_identity);
    (void)snprintf(value->writer_transform_identity,
                   sizeof(value->writer_transform_identity), "%s",
                   runtime_fixture_execution()->logical_transform_identity);
    (void)snprintf(value->admitted_transform_identity,
                   sizeof(value->admitted_transform_identity), "%s",
                   admission->transform_identity);
    (void)snprintf(value->writer_profile_identity, sizeof(value->writer_profile_identity), "%s",
                   admission->profile_identity);
    (void)snprintf(value->admitted_profile_identity,
                   sizeof(value->admitted_profile_identity), "%s",
                   admission->profile_identity);
    (void)snprintf(value->quant_execution_identity,
                   sizeof(value->quant_execution_identity), "%s",
                   admission->quant_execution_identity);
    (void)snprintf(value->payload_plan_identity, sizeof(value->payload_plan_identity), "%s",
                   admission->payload_plan_identity);
    (void)snprintf(value->payload_byte_identity, sizeof(value->payload_byte_identity), "%s",
                   admission->payload_byte_identity);
    value->physical_payload_compatible = 1;
    value->tensor_inventory_equal = 1;
    value->qtype_equal = 1;
    value->layout_equal = 1;
    value->offset_equal = 1;
    value->payload_digest_equal = 1;
}

/* Construct one identity-bearing synthetic plan through the canonical import owner. */
static int fixture_attention_build(binding_fixture *fixture)
{
    const yvex_materialization_summary *materialization;
    const yvex_runtime_descriptor_summary *runtime;
    yvex_attention_summary summary;
    yvex_attention_layer_plan layer;
    yvex_attention_failure failure;
    yvex_error err;

    materialization = yvex_materialization_session_summary(fixture->materialization);
    runtime = yvex_runtime_descriptor_summary_get(fixture->descriptor);
    if (!materialization || !runtime) return 0;
    memset(&summary, 0, sizeof(summary));
    memset(&layer, 0, sizeof(layer));
    summary.status = YVEX_ATTENTION_STATUS_EXECUTION_READY;
    summary.tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
    (void)snprintf(summary.artifact_identity, sizeof(summary.artifact_identity), "%s",
                   runtime->artifact_identity);
    (void)snprintf(summary.materialization_plan_identity,
                   sizeof(summary.materialization_plan_identity), "%s",
                   materialization->plan_identity);
    (void)snprintf(summary.logical_model_identity, sizeof(summary.logical_model_identity), "%s",
                   runtime->logical_model_identity);
    (void)snprintf(summary.runtime_descriptor_identity,
                   sizeof(summary.runtime_descriptor_identity), "%s",
                   runtime->runtime_descriptor_identity);
    (void)snprintf(summary.runtime_numeric_identity,
                   sizeof(summary.runtime_numeric_identity), "%s",
                   runtime->runtime_numeric_identity);
    summary.layer_count = 1ull;
    summary.swa_layer_count = 1ull;
    summary.required_binding_count = 1ull;
    summary.payload_bytes_bound = materialization->payload_bytes;
    summary.qtype_binding_counts[YVEX_GGUF_QTYPE_F32] = 1ull;
    summary.history_contract_ready = 1;
    summary.state_delta_contract_ready = 1;
    summary.cpu_reference_ready = 1;
    summary.full_execution_ready = 1;
    summary.cuda_execution_ready = 1;
    layer.ordinal = 0ull;
    layer.layer_index = 0ull;
    layer.predictor_index = YVEX_MATERIALIZATION_NO_INDEX;
    layer.tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER;
    layer.attention_class = YVEX_ATTENTION_CLASS_SWA;
    layer.compute_contract = YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1;
    layer.sliding_window = 4ull;
    layer.query_heads = 1ull;
    layer.kv_heads = 1ull;
    layer.head_dimension = 4ull;
    layer.query_lora_rank = 4ull;
    layer.hidden_dimension = 4ull;
    layer.residual_expanded_width = 4ull;
    layer.required_binding_count = 1ull;
    layer.payload_bytes_bound = materialization->payload_bytes;
    if (!yvex_attention_plan_identity_compute(
            &summary, &layer, 1ull, summary.attention_plan_identity))
        return 0;
    if (yvex_attention_plan_import(
            &fixture->attention, &summary, &layer, 1ull, fixture->materialization,
            fixture->descriptor, &failure, &err) != YVEX_OK) {
        fprintf(stderr, "runtime binding attention import failed: %s\n",
                yvex_error_message(&err));
        return 0;
    }
    return 1;
}

static int fixture_model_execution_build(
    yvex_model_execution_descriptor *model, yvex_error *err)
{
    char logical[YVEX_SHA256_HEX_CAP], source[YVEX_SHA256_HEX_CAP];
    char schedule[YVEX_SHA256_HEX_CAP], state[YVEX_SHA256_HEX_CAP];
    yvex_model_execution_descriptor_request request = {0};

    (void)snprintf(logical, sizeof(logical), "%064x", 31);
    (void)snprintf(source, sizeof(source), "%064x", 32);
    (void)snprintf(schedule, sizeof(schedule), "%064x", 33);
    (void)snprintf(state, sizeof(state), "%064x", 34);
    request.schema_version = YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1;
    request.logical_model_identity = logical;
    request.source_model_identity = source;
    request.attention_schedule_identity = schedule;
    request.persistent_state_identity = state;
    request.maximum_context = 4ull;
    request.original_context = 4ull;
    request.rope_scaling = YVEX_MODEL_ROPE_SCALING_NONE;
    request.rope_theta = 10000ull;
    request.rope_scaling_factor = 1ull;
    request.layer_count = 1ull;
    request.hidden_width = 4ull;
    request.vocabulary_size = 4ull;
    request.attention_heads = 1ull;
    request.kv_heads = 1ull;
    request.head_width = 4ull;
    request.swa_layers = 1ull;
    request.sliding_window = 4ull;
    request.normalization_epsilon = 1e-6;
    request.output_input_width = 4ull;
    request.output_vocabulary_size = 4ull;
    request.persistent_state_class_mask =
        YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING);
    request.eos_token_id = 1ull;
    return yvex_model_execution_descriptor_seal(&request, model, err) == YVEX_OK;
}

static void fixture_disable_unavailable_execution(
    yvex_runtime_capabilities *capabilities)
{
    capabilities->moe_plan_ready = 0;
    capabilities->moe_router_ready = 0;
    capabilities->moe_routed_expert_ready = 0;
    capabilities->moe_shared_expert_ready = 0;
    capabilities->moe_block_ready = 0;
    capabilities->transformer_ready = 0;
    capabilities->output_head_binding_ready = 0;
    capabilities->output_head_projection_ready = 0;
    capabilities->logits_cpu_ready = 0;
    capabilities->logits_cuda_ready = 0;
    capabilities->logits_prefill_ready = 0;
    capabilities->logits_decode_ready = 0;
    capabilities->logits_full_vocabulary_ready = 0;
    capabilities->logits_hidden_contract_ready = 0;
    capabilities->logits_partial_progress_ready = 0;
    capabilities->logits_ready = 0;
}

static int fixture_execution_compile(binding_fixture *fixture, yvex_error *err)
{
    const yvex_family_compiler_adapter *compiler = runtime_fixture_compiler();
    const yvex_runtime_descriptor_summary *descriptor =
        yvex_runtime_descriptor_summary_get(fixture->descriptor);
    yvex_compiled_model_plan_request plan = {0};
    if (!compiler || !descriptor || !compiler->execution_capabilities ||
        !compiler->transformer_policy || !compiler->logits_policy ||
        !compiler->speculation_policy || !compiler->graph ||
        !compiler->execution_capabilities(&fixture->capabilities) ||
        !compiler->transformer_policy(
            descriptor, &fixture->transformer_policy) ||
        !compiler->logits_policy(&fixture->logits_policy) ||
        !compiler->speculation_policy(
            descriptor, &fixture->speculation_policy))
        return 0;
    fixture_disable_unavailable_execution(&fixture->capabilities);
    if (!yvex_runtime_capabilities_contract_valid(&fixture->capabilities) ||
        yvex_physical_execution_ir_build(
            &fixture->physical_execution, fixture->materialization,
            fixture->descriptor, fixture->admission.profile_identity,
            compiler->physical_execution_policy, err) != YVEX_OK)
        return 0;
    plan.operator_graph = fixture->operator_graph;
    plan.materialization = fixture->materialization;
    plan.descriptor = fixture->descriptor;
    plan.attention = fixture->attention;
    plan.graph = compiler->graph();
    plan.family_adapter_id = compiler->adapter_id;
    plan.family_adapter_version = compiler->adapter_version;
    plan.capabilities = fixture->capabilities;
    plan.transformer_policy = fixture->transformer_policy;
    plan.logits_policy = fixture->logits_policy;
    return yvex_compiled_model_plan_build(
               &fixture->compiled_plan, &plan, err) == YVEX_OK;
}

static int fixture_build(binding_fixture *fixture, const char *artifact_path,
                         int include_model_execution)
{
    yvex_materialization_options options;
    yvex_materialization_projection projection;
    yvex_materialization_failure material_failure;
    yvex_model_execution_descriptor model_execution;
    yvex_runtime_descriptor_family_facts family = {0};
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_error err;
    int rc;

    memset(fixture, 0, sizeof(*fixture));
    if (!fixture_artifact_open(fixture, artifact_path))
        return 0;
    if (yvex_test_deepseek_map_fixture_build(&fixture->map) != YVEX_OK || !fixture->map)
        return 0;
    rc = yvex_materialization_project_artifact_lowering(fixture->map, &projection, &err);
    if (rc != YVEX_OK || !fixture_admission_build(fixture)) return 0;
    fixture->admission.mapping_identity = projection.mapping_identity;
    fixture_compatibility_build(fixture);
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 16u;
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &fixture->materialization_plan, &fixture->admission, fixture->artifact,
            fixture->gguf, fixture->tensors, &projection, &options, &material_failure, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "runtime binding materialization plan failed: %s\n",
                yvex_error_message(&err));
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &fixture->materialization, fixture->materialization_plan,
            fixture->artifact, &options, &material_failure, &err);
    if (rc != YVEX_OK && fixture->materialization_plan)
        fprintf(stderr, "runtime binding materialization open failed: %s\n",
                yvex_error_message(&err));
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            fixture->materialization, &material_failure, &err);
    if (rc == YVEX_OK && include_model_execution) {
        if (!fixture_model_execution_build(&model_execution, &err))
            rc = yvex_error_code(&err);
        else {
            family.logical_model_identity = model_execution.logical_model_identity;
            family.runtime_numeric_identity = model_execution.source_model_identity;
            family.runtime_hadamard_revision = "fixture-hadamard-v1";
            family.runtime_numeric_schema_version = 1u;
            family.runtime_compute_policy_count = 1ull;
            family.layer_count = model_execution.layer_count;
            family.vocabulary_size = model_execution.vocabulary_size;
            family.model_execution = &model_execution;
        }
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_descriptor_build(
            &fixture->descriptor, &fixture->admission, fixture->materialization,
            include_model_execution ? &family : NULL, &descriptor_failure, &err);
    if (rc != YVEX_OK && fixture->materialization)
        fprintf(stderr, "runtime binding descriptor build failed: %s\n",
                yvex_error_message(&err));
    if (rc == YVEX_OK && !fixture_attention_build(fixture)) rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK) {
        const yvex_family_compiler_adapter *compiler = runtime_fixture_compiler();
        const yvex_runtime_descriptor_summary *summary =
            yvex_runtime_descriptor_summary_get(fixture->descriptor);
        yvex_semantic_model_ir_request semantic = {
            .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
            .family_adapter_id = compiler ? compiler->adapter_id : 0ull,
            .family_adapter_version = compiler ? compiler->adapter_version : 0ull,
            .target_id = "runtime-binding-fixture",
            .source_model_identity = summary
                ? summary->model_execution.source_model_identity : NULL,
            .logical_model_identity = summary
                ? summary->model_execution.logical_model_identity : NULL,
            .semantic_payload_identity = summary
                ? summary->model_execution.identity : NULL,
            .execution_descriptor = summary
                ? &summary->model_execution : NULL};
        rc = yvex_semantic_model_ir_seal(
            &fixture->semantic_model, &semantic, &err);
        if (rc == YVEX_OK)
            rc = yvex_operator_graph_ir_build_transformer(
                &fixture->operator_graph, fixture->semantic_model,
                fixture->attention, NULL, &err);
        if (rc == YVEX_OK && !fixture_execution_compile(fixture, &err))
            rc = YVEX_ERR_STATE;
    }
    return rc == YVEX_OK;
}

static void fixture_close(binding_fixture *fixture)
{
    if (!fixture) return;
    yvex_operator_graph_ir_close(&fixture->operator_graph);
    yvex_semantic_model_ir_close(&fixture->semantic_model);
    yvex_compiled_model_plan_close(&fixture->compiled_plan);
    yvex_physical_execution_ir_close(&fixture->physical_execution);
    yvex_attention_plan_close(fixture->attention);
    yvex_runtime_descriptor_close(fixture->descriptor);
    yvex_materialization_session_close(fixture->materialization);
    yvex_materialization_plan_close(fixture->materialization_plan);
    if (fixture->map)
        yvex_model_register_deepseek_v4()->lowering.map->close(fixture->map);
    yvex_tensor_table_close(fixture->tensors);
    yvex_gguf_close(fixture->gguf);
    yvex_artifact_close(fixture->artifact);
    memset(fixture, 0, sizeof(*fixture));
}

static int variant_path(const char *root, const char *variant, const char *basename,
                        char directory[YVEX_PATH_CAP], char path[YVEX_PATH_CAP])
{
    if (snprintf(directory, YVEX_PATH_CAP, "%s/%s", root, variant) >= YVEX_PATH_CAP ||
        mkdir(directory, 0700) != 0 ||
        snprintf(path, YVEX_PATH_CAP, "%s/%s", directory, basename) >= YVEX_PATH_CAP)
        return 0;
    return 1;
}

static int directory_has_suffix(const char *path, const char *suffix)
{
    struct dirent *entry;
    DIR *directory = opendir(path);
    int found = 0;
    size_t suffix_length = suffix ? strlen(suffix) : 0u;

    if (!directory || !suffix_length) return 1;
    while ((entry = readdir(directory)) != NULL) {
        size_t name_length = strlen(entry->d_name);
        if (name_length >= suffix_length &&
            strcmp(entry->d_name + name_length - suffix_length, suffix) == 0) {
            found = 1;
            break;
        }
    }
    (void)closedir(directory);
    return found;
}

static int directory_has_temporary(const char *path)
{
    return directory_has_suffix(path, ".tmp");
}

static int fixture_binding_request(const binding_fixture *fixture, const char *directory,
                                   yvex_runtime_binding_prepare_request *request)
{
    const yvex_graph_execution_binding *execution = runtime_fixture_execution();
    const yvex_family_compiler_adapter *compiler = runtime_fixture_compiler();
    const yvex_runtime_descriptor_summary *descriptor;
    yvex_conversation_protocol tokenizer_source;

    memset(request, 0, sizeof(*request));
    request->directory = directory;
    request->admission = &fixture->admission;
    request->physical_compatibility = &fixture->compatibility;
    request->materialization = fixture->materialization;
    request->runtime_descriptor = fixture->descriptor;
    request->operator_graph = fixture->operator_graph;
    request->physical_execution = fixture->physical_execution;
    request->compiled_plan = fixture->compiled_plan;
    request->attention_plan = fixture->attention;
    descriptor = yvex_runtime_descriptor_summary_get(fixture->descriptor);
    if (!execution || !compiler || !descriptor || !compiler->graph ||
        !compiler->graph() ||
        !compiler->execution_capabilities ||
        !compiler->transformer_policy || !compiler->logits_policy ||
        !compiler->speculation_policy || !compiler->tokenizer_policy) return 0;
    request->family_adapter_id = execution->adapter_id;
    request->family_adapter_version = execution->adapter_version;
    request->artifact_format = "GGUF";
    request->artifact_format_version = 3u;
    request->logical_transform_identity =
        execution->logical_transform_identity;
    request->capabilities = fixture->capabilities;
    request->transformer_policy = fixture->transformer_policy;
    request->logits_policy = fixture->logits_policy;
    request->speculation_policy = fixture->speculation_policy;
    tokenizer_source = *yvex_model_deepseek_v4_conversation();
    tokenizer_source.vocabulary_size = descriptor->model_execution.vocabulary_size;
    tokenizer_source.base_vocabulary_size = descriptor->model_execution.vocabulary_size;
    tokenizer_source.merge_count = 0u;
    tokenizer_source.added_token_count = 0u;
    tokenizer_source.special_token_count = 0u;
    if (yvex_tokenizer_family_policy_compile(
            &request->tokenizer_policy, &tokenizer_source,
            YVEX_TOKENIZER_KIND_FIXTURE_SIMPLE, YVEX_TOKENIZER_MODEL_FIXTURE,
            YVEX_TOKENIZER_PROMPT_CONVERSATION, NULL) != YVEX_OK) return 0;
    return
           yvex_runtime_capabilities_contract_valid(&request->capabilities);
}

static int test_runtime_capability_contract(void)
{
    yvex_runtime_capabilities eager = {0}, mutated;
    char eager_identity[YVEX_SHA256_HEX_CAP], mutated_identity[YVEX_SHA256_HEX_CAP];

    eager.attention_semantics_ready = 1;
    eager.attention_core_ready = 1;
    eager.cpu_prefill_eager_ready = 1;
    eager.cpu_decode_eager_ready = 1;
    eager.cuda_eager_implemented = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_capabilities_contract_valid(&eager) &&
            !eager.cuda_piecewise_graph_implemented &&
            !eager.cuda_full_graph_implemented &&
            yvex_runtime_capabilities_identity(&eager, eager_identity),
        "eager-only capability declaration is valid without graph promotion");
    mutated = eager;
    mutated.attention_trace_ready = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_capabilities_identity(&mutated, mutated_identity) &&
            strcmp(eager_identity, mutated_identity) != 0,
        "one valid capability mutation changes its canonical identity");
    mutated = eager;
    mutated.cuda_eager_implemented = 0;
    mutated.cuda_full_graph_implemented = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "full graph implementation without eager is invalid");
    memset(&mutated, 0, sizeof(mutated));
    mutated.attention_semantics_ready = 1;
    mutated.attention_envelope_ready = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "attention envelope without core is invalid");
    mutated = eager;
    mutated.attention_weight_residency_ready = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "pre-admission residency readiness is invalid");
    mutated = eager;
    mutated.attention_workspace_ready = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "pre-admission workspace readiness is invalid");
    mutated = eager;
    mutated.persistent_kv_ready = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "pre-admission persistent KV readiness is invalid");
    mutated = eager;
    mutated.generation_ready = 1;
    YVEX_TEST_ASSERT(!yvex_runtime_capabilities_contract_valid(&mutated),
                     "pre-admission generation readiness is invalid");
    mutated = eager;
    mutated.attention_core_ready = 2;
    YVEX_TEST_ASSERT(
        !yvex_runtime_capabilities_identity(&mutated, mutated_identity),
        "non-binary capability fields have no canonical identity");
    return 0;
}

static int test_physical_activation_policy(const binding_fixture *fixture)
{
    const yvex_family_compiler_adapter *compiler = runtime_fixture_compiler();
    const yvex_physical_execution_decision *base;
    const yvex_physical_execution_decision *encoded_decision;
    yvex_runtime_tensor_binding *runtime_binding;
    yvex_materialized_tensor_binding *materialized;
    yvex_physical_execution_ir *encoded = NULL;
    yvex_physical_execution_policy policy;
    yvex_runtime_binding candidate = {0};
    yvex_error err;
    unsigned long long original_experts, backing_bytes = 0ull;
    yvex_tensor_role original_role;
    int rc;

    base = yvex_physical_execution_ir_decision_at(fixture->physical_execution, 0ull);
    runtime_binding = (yvex_runtime_tensor_binding *)
        yvex_runtime_descriptor_tensor_at(fixture->descriptor, 0ull);
    materialized = runtime_binding
                       ? (yvex_materialized_tensor_binding *)runtime_binding->binding
                       : NULL;
    YVEX_TEST_ASSERT(compiler && compiler->physical_execution_policy && base && materialized,
                     "physical activation fixture has one compiled decision");
    policy = *compiler->physical_execution_policy;
    policy.schema_version = YVEX_PHYSICAL_EXECUTION_POLICY_SCHEMA_V2;
    policy.activation = YVEX_EXECUTION_ACTIVATION_DEVICE_F32;
    policy.encoded_activation_consumer_mask =
        1ull << (unsigned int)YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP;
    policy.derived_asset_qtype_mask = 1ull << base->canonical_qtype;
    original_experts = materialized->expert_count;
    original_role = runtime_binding->role;
    materialized->expert_count = 2ull;
    runtime_binding->role = YVEX_TENSOR_ROLE_MOE_EXPERT_GATE;
    rc = yvex_physical_execution_ir_build(
        &encoded, fixture->materialization, fixture->descriptor,
        fixture->admission.profile_identity, &policy, &err);
    materialized->expert_count = original_experts;
    runtime_binding->role = original_role;
    encoded_decision = yvex_physical_execution_ir_decision_at(encoded, 0ull);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && encoded_decision &&
            encoded_decision->consumer == YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP &&
            encoded_decision->activation == YVEX_EXECUTION_ACTIVATION_DEVICE_ENCODED &&
            encoded_decision->expert_count == 2ull &&
            encoded_decision->derived_asset_required &&
            encoded_decision->layout == YVEX_EXECUTION_LAYOUT_DERIVED_BACKEND &&
            strcmp(yvex_physical_execution_ir_summary(encoded)->identity,
                   yvex_physical_execution_ir_summary(fixture->physical_execution)->identity) != 0,
        "V2 physical policy seals consumer-selected encoded activation into identity");
    candidate.summary.tensor_count = 1ull;
    candidate.materialized = materialized;
    candidate.physical_execution = encoded;
    YVEX_TEST_ASSERT(
        yvex_runtime_private_residency_backing_bytes(
            &candidate, NULL, YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED,
            &backing_bytes, &err) == YVEX_ERR_UNSUPPORTED && !backing_bytes,
        "compiled derived layout refuses an incompatible host residency policy");
    yvex_physical_execution_ir_close(&encoded);
    policy.schema_version = YVEX_PHYSICAL_EXECUTION_POLICY_SCHEMA_V1;
    rc = yvex_physical_execution_ir_build(
        &encoded, fixture->materialization, fixture->descriptor,
        fixture->admission.profile_identity, &policy, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_INVALID_ARG && !encoded,
        "V1 physical policy refuses an unrepresentable encoded consumer mask");
    return 0;
}

/* Validate prepare, independent reopen, all three imports, and atomic conflict. */
static int test_prepare_reopen_import(const binding_fixture *fixture, const char *directory,
                                      yvex_runtime_binding_prepare_result *prepared,
                                      yvex_runtime_binding **binding_out)
{
    yvex_runtime_binding_prepare_request request;
    yvex_runtime_binding_prepare_request mutated_request;
    yvex_runtime_binding_prepare_result conflict_result;
    yvex_runtime_binding_prepare_result rejected_result;
    yvex_runtime_binding_prepare_result mutated_result;
    yvex_artifact_physical_compatibility rejected_compatibility;
    yvex_artifact_physical_compatibility mutated_compatibility;
    yvex_runtime_binding *mutated = NULL;
    yvex_runtime_binding_failure failure;
    yvex_materialization_options options;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_attention_plan *attention = NULL;
    yvex_attention_plan *draft_attention = NULL;
    const yvex_physical_execution_ir *physical = NULL;
    const yvex_physical_execution_decision *physical_decision;
    const yvex_transformer_family_policy *transformer;
    const yvex_logits_family_policy *logits;
    const yvex_speculation_family_policy *speculation;
    const yvex_tokenizer_family_policy *tokenizer_policy;
    const yvex_runtime_descriptor_summary *imported_descriptor;
    yvex_runtime_binding_summary summary;
    yvex_core_file_result file_result;
    unsigned char *before = NULL, *after = NULL;
    size_t before_count = 0u, after_count = 0u;
    yvex_error err;
    char capability_identity[YVEX_SHA256_HEX_CAP];
    int rc;

    YVEX_TEST_ASSERT(test_physical_activation_policy(fixture) == 0,
                     "physical activation policy is versioned and identity-bound");
    {
        const yvex_operator_graph_summary *operators =
            yvex_operator_graph_ir_summary(fixture->operator_graph);
        const yvex_operator_node *attention_node =
            yvex_operator_graph_ir_node_at(fixture->operator_graph, 1ull);
        YVEX_TEST_ASSERT(
            operators && operators->maximum_context == 4ull &&
                operators->target_layer_count == 1ull &&
                operators->draft_layer_count == 0ull &&
                operators->node_count == 5ull && operators->edge_count == 6ull &&
                operators->state_class_mask ==
                    YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING) &&
                attention_node && attention_node->kind == YVEX_OPERATOR_ATTENTION &&
                attention_node->state_read_mask ==
                    YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING),
            "canonical operator graph owns target topology and mutable state edges");
    }
    YVEX_TEST_ASSERT(fixture_binding_request(fixture, directory, &request),
                     "runtime binding request declares adapter capabilities");
    memset(&rejected_result, 0, sizeof(rejected_result));
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE", "1", 1) == 0,
        "runtime binding validation fault enabled");
    rc = yvex_runtime_binding_prepare(&request, &rejected_result, &failure, &err);
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE") == 0,
        "runtime binding validation fault disabled");
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && !rejected_result.published &&
            failure.code == YVEX_RUNTIME_BINDING_FAILURE_FORMAT &&
            strcmp(failure.field, "candidate-validation") == 0 &&
            !directory_has_temporary(directory) &&
            !directory_has_suffix(directory, YVEX_RUNTIME_BINDING_SUFFIX),
        "candidate validation failure publishes nothing and removes its temporary");
    mutated_request = request;
    mutated_request.physical_execution = NULL;
    memset(&rejected_result, 0, sizeof(rejected_result));
    rc = yvex_runtime_binding_prepare(
        &mutated_request, &rejected_result, &failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_INVALID_ARG && !rejected_result.published &&
            failure.code == YVEX_RUNTIME_BINDING_FAILURE_INVALID_ARGUMENT &&
            strcmp(failure.field, "request") == 0 &&
            !directory_has_suffix(directory, YVEX_RUNTIME_BINDING_SUFFIX),
        "runtime binding refuses an uncompiled physical execution input");
    {
        const yvex_runtime_descriptor_summary *fixture_descriptor =
            yvex_runtime_descriptor_summary_get(fixture->descriptor);
        const yvex_family_compiler_adapter *compiler = runtime_fixture_compiler();
        yvex_semantic_model_ir_request semantic_request = {
            .schema_version = YVEX_SEMANTIC_MODEL_IR_SCHEMA_V1,
            .family_adapter_id = compiler->adapter_id,
            .family_adapter_version = compiler->adapter_version,
            .target_id = "runtime-binding-identity-mismatch",
            .source_model_identity =
                fixture_descriptor->model_execution.source_model_identity,
            .logical_model_identity =
                fixture_descriptor->model_execution.logical_model_identity,
            .semantic_payload_identity =
                fixture_descriptor->model_execution.identity,
            .execution_descriptor = &fixture_descriptor->model_execution};
        yvex_semantic_model_ir *other_semantic = NULL;
        yvex_operator_graph_ir *other_graph = NULL;
        YVEX_TEST_ASSERT(
            yvex_semantic_model_ir_seal(
                &other_semantic, &semantic_request, &err) == YVEX_OK &&
            yvex_operator_graph_ir_build_transformer(
                &other_graph, other_semantic, fixture->attention,
                NULL, &err) == YVEX_OK,
            "distinct canonical operator graph fixture built");
        mutated_request = request;
        mutated_request.operator_graph = other_graph;
        memset(&rejected_result, 0, sizeof(rejected_result));
        rc = yvex_runtime_binding_prepare(
            &mutated_request, &rejected_result, &failure, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_STATE && !rejected_result.published &&
                failure.code == YVEX_RUNTIME_BINDING_FAILURE_IDENTITY &&
                strcmp(failure.field, "operator-graph") == 0,
            "runtime binding refuses an operator graph not sealed by its compiled plan");
        yvex_operator_graph_ir_close(&other_graph);
        yvex_semantic_model_ir_close(&other_semantic);
    }
    rc = yvex_runtime_binding_prepare(&request, prepared, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && prepared->published, "runtime binding prepared");
    YVEX_TEST_ASSERT(strlen(prepared->summary.identity) == 64u,
                     "runtime binding content identity");
    rc = yvex_runtime_binding_open(
        binding_out, prepared->path, &summary, NULL, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "runtime binding reopened");
    YVEX_TEST_ASSERT(strcmp(summary.identity, prepared->summary.identity) == 0,
                     "reopened runtime binding identity");
    YVEX_TEST_ASSERT(summary.schema_version == YVEX_RUNTIME_BINDING_SCHEMA_CURRENT,
                     "reopened runtime binding schema");
    YVEX_TEST_ASSERT(
        yvex_sha256_hex_is_valid(summary.semantic_graph_identity) &&
            yvex_sha256_hex_is_valid(summary.executable_graph_identity) &&
            strcmp(summary.semantic_graph_identity,
                   yvex_operator_graph_ir_summary(
                       fixture->operator_graph)->identity) == 0 &&
            strcmp(summary.semantic_graph_identity,
                   summary.executable_graph_identity) != 0,
        "runtime binding owner derives distinct graph identities without caller assertions");
    YVEX_TEST_ASSERT(
        yvex_runtime_capabilities_identity(&request.capabilities, capability_identity) &&
            strcmp(summary.execution_capability_identity, capability_identity) == 0 &&
            strcmp(prepared->summary.execution_capability_identity,
                   capability_identity) == 0 &&
            memcmp(&summary.capabilities, &request.capabilities,
                   sizeof(summary.capabilities)) == 0,
        "reopened binding preserves the authenticated adapter capability matrix");
    YVEX_TEST_ASSERT(
        summary.physical_compatibility.physical_payload_compatible == 1 &&
            summary.physical_compatibility.artifact_rebuild_required == 0 &&
            summary.physical_compatibility.materialization_rebuild_required == 0 &&
            strcmp(summary.physical_compatibility.writer_plan_identity,
                   fixture->compatibility.writer_plan_identity) == 0 &&
            strcmp(summary.physical_compatibility.payload_byte_identity,
                   fixture->admission.payload_byte_identity) == 0,
        "reopened runtime binding preserves exact physical compatibility proof");
    YVEX_TEST_ASSERT(
        strcmp(summary.artifact_transform_identity, fixture->admission.transform_identity) == 0 &&
            strcmp(summary.logical_transform_identity,
                   runtime_fixture_execution()->logical_transform_identity) == 0,
        "runtime binding separates artifact and logical transform identities");
    YVEX_TEST_ASSERT(summary.tensor_count == 1ull && summary.layer_count == 1ull,
                     "reopened runtime binding record counts");
    yvex_materialization_options_default(&options);
    rc = yvex_runtime_binding_import_materialization(
        *binding_out, fixture->artifact, &options, &plan, &session, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "binding materialization imported");
    rc = yvex_runtime_binding_import_graph(
        *binding_out, session, &descriptor, &attention, &draft_attention,
        &physical, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !draft_attention && physical,
                     "binding runtime graph imported without an invented draft plan");
    YVEX_TEST_ASSERT(
        yvex_physical_execution_ir_summary(physical)->schema_version ==
                YVEX_PHYSICAL_EXECUTION_SCHEMA_V2 &&
            strcmp(yvex_physical_execution_ir_summary(physical)->identity,
               summary.physical_execution_identity) == 0 &&
            summary.physical_execution_decision_count == summary.tensor_count,
        "binding imports persisted physical execution truth");
    YVEX_TEST_ASSERT(
        yvex_runtime_binding_policies(
            *binding_out, &transformer, &logits, &speculation),
        "binding exposes one compiled policy envelope");
    imported_descriptor = yvex_runtime_descriptor_summary_get(descriptor);
    physical_decision = yvex_physical_execution_ir_decision_at(physical, 0ull);
    YVEX_TEST_ASSERT(
        transformer && imported_descriptor &&
            transformer->hidden_width == imported_descriptor->model_execution.hidden_width &&
            transformer->residual_streams ==
                imported_descriptor->model_execution.residual_streams && logits &&
            logits->separate_output_head && speculation &&
            speculation->block_size == imported_descriptor->model_execution.proposal_width &&
            speculation->target_feature_layer_count ==
                imported_descriptor->model_execution.target_feature_count &&
            yvex_sha256_hex_is_valid(speculation->policy_identity),
        "binding imports authenticated compiled execution policies");
    YVEX_TEST_ASSERT(
        physical_decision && imported_descriptor &&
            physical_decision->maximum_context ==
                imported_descriptor->model_execution.maximum_context &&
            physical_decision->supported_width_mask == 2ull &&
            physical_decision->activation ==
                runtime_fixture_compiler()->physical_execution_policy->activation &&
            strcmp(physical_decision->kernel_family,
                   runtime_fixture_compiler()->physical_execution_policy
                       ->dense_kernel_family) == 0,
        "physical execution derives semantic bounds and preserves selected policy");
    tokenizer_policy = yvex_runtime_binding_tokenizer_policy(*binding_out);
    YVEX_TEST_ASSERT(
        tokenizer_policy &&
            tokenizer_policy->family_adapter_id == summary.family_adapter_id &&
            tokenizer_policy->family_adapter_version == summary.family_adapter_version &&
            tokenizer_policy->vocabulary_size ==
                imported_descriptor->model_execution.vocabulary_size &&
            yvex_tokenizer_family_policy_validate(tokenizer_policy, &err) == YVEX_OK,
        "binding imports the authenticated compiled tokenizer policy");
    YVEX_TEST_ASSERT(
        strcmp(yvex_runtime_descriptor_summary_get(descriptor)->runtime_descriptor_identity,
               summary.runtime_descriptor_identity) == 0,
        "imported descriptor identity");
    YVEX_TEST_ASSERT(
        strcmp(yvex_attention_plan_summary(attention)->attention_plan_identity,
               summary.attention_plan_identity) == 0 &&
            yvex_attention_plan_summary(attention)->required_envelope_binding_count ==
                yvex_attention_plan_summary(fixture->attention)
                    ->required_envelope_binding_count,
        "imported attention identity and envelope binding count");
    yvex_attention_plan_close(attention);
    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);

    mutated_request = request;
    mutated_request.capabilities.attention_trace_ready =
        !mutated_request.capabilities.attention_trace_ready;
    YVEX_TEST_ASSERT(
        yvex_runtime_capabilities_contract_valid(&mutated_request.capabilities),
        "capability mismatch fixture remains internally valid");
    YVEX_TEST_ASSERT(
        yvex_runtime_capabilities_admitted_by(&mutated_request.capabilities,
                                              &request.capabilities) &&
            !yvex_runtime_capabilities_admitted_by(&request.capabilities,
                                                   &mutated_request.capabilities),
        "artifact capability subsets admit while unsupported promotions refuse");

    rejected_compatibility = fixture->compatibility;
    rejected_compatibility.payload_digest_equal = 0;
    mutated_request = request;
    mutated_request.physical_compatibility = &rejected_compatibility;
    memset(&rejected_result, 0, sizeof(rejected_result));
    rc = yvex_runtime_binding_prepare(
        &mutated_request, &rejected_result, &failure, &err);
    YVEX_TEST_ASSERT(
        rc != YVEX_OK && !rejected_result.published &&
            failure.code == YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY &&
            strcmp(failure.field, "compatibility-verdict") == 0,
        "mutated physical compatibility verdict refuses before publication");
    rejected_compatibility = fixture->compatibility;
    (void)snprintf(rejected_compatibility.writer_transform_identity,
                   sizeof(rejected_compatibility.writer_transform_identity), "%064x", 12);
    mutated_request.physical_compatibility = &rejected_compatibility;
    memset(&rejected_result, 0, sizeof(rejected_result));
    rc = yvex_runtime_binding_prepare(
        &mutated_request, &rejected_result, &failure, &err);
    YVEX_TEST_ASSERT(
        rc != YVEX_OK && !rejected_result.published &&
            failure.code == YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY &&
            strcmp(failure.field, "writer-transform-identity") == 0,
        "mutated compatibility identity refuses before publication");

    mutated_request = request;
    mutated_request.logical_transform_identity =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    mutated_compatibility = fixture->compatibility;
    (void)snprintf(mutated_compatibility.writer_transform_identity,
                   sizeof(mutated_compatibility.writer_transform_identity), "%s",
                   mutated_request.logical_transform_identity);
    mutated_request.physical_compatibility = &mutated_compatibility;
    memset(&mutated_result, 0, sizeof(mutated_result));
    rc = yvex_runtime_binding_prepare(
        &mutated_request, &mutated_result, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && mutated_result.published &&
                         strcmp(mutated_result.summary.identity,
                                prepared->summary.identity) != 0,
                     "logical transform identity changes binding content address");
    rc = yvex_runtime_binding_open(
        &mutated, mutated_result.path, &mutated_result.summary, NULL, &failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            strcmp(mutated_result.summary.logical_transform_identity,
                   mutated_request.logical_transform_identity) == 0,
        "mutated logical transform identity reopens from canonical body");
    yvex_runtime_binding_close(mutated);
    mutated = NULL;
    YVEX_TEST_ASSERT(unlink(mutated_result.path) == 0,
                     "mutated logical transform binding cleaned");

    memset(&file_result, 0, sizeof(file_result));
    YVEX_TEST_ASSERT(
        yvex_core_file_read_snapshot(prepared->path, 64u * 1024u * 1024u,
                                     &before, &before_count, &file_result,
                                     &err) == YVEX_OK &&
            setenv("YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE", "1", 1) == 0,
        "published binding snapshot and validation fault prepared");
    memset(&conflict_result, 0, sizeof(conflict_result));
    rc = yvex_runtime_binding_prepare(&request, &conflict_result, &failure, &err);
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_BINDING_VALIDATE_FAILURE") == 0 &&
            yvex_core_file_read_snapshot(prepared->path, 64u * 1024u * 1024u,
                                         &after, &after_count, &file_result,
                                         &err) == YVEX_OK,
        "published binding snapshot reopened after candidate refusal");
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && !conflict_result.published &&
            failure.code == YVEX_RUNTIME_BINDING_FAILURE_FORMAT &&
            before_count == after_count && memcmp(before, after, before_count) == 0 &&
            !directory_has_temporary(directory),
        "candidate refusal preserves the pre-existing destination byte-for-byte");
    free(before);
    free(after);
    before = NULL;
    after = NULL;

    memset(&conflict_result, 0, sizeof(conflict_result));
    rc = yvex_runtime_binding_prepare(&request, &conflict_result, &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "duplicate content-address publication refused");
    YVEX_TEST_ASSERT(failure.code == YVEX_RUNTIME_BINDING_FAILURE_CONFLICT,
                     "publication conflict failure code");
    YVEX_TEST_ASSERT(!conflict_result.published && !directory_has_temporary(directory),
                     "conflict removes only its temporary file");
    return 0;
}

static int test_compiled_model_binding_v12(const char *root)
{
    binding_fixture fixture;
    yvex_runtime_binding_prepare_request request;
    yvex_runtime_binding_prepare_result prepared = {0};
    yvex_runtime_binding_summary summary;
    yvex_runtime_binding_failure failure;
    yvex_runtime_binding *binding = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_generation_context *generation = NULL;
    yvex_runtime_generation_options generation_options = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_model_failure model_failure;
    const yvex_runtime_descriptor_summary *descriptor;
    char directory[YVEX_PATH_CAP], artifact_path[YVEX_PATH_CAP];
    yvex_error err;

    YVEX_TEST_ASSERT(
        variant_path(root, "model-v12", "runtime.gguf", directory, artifact_path) &&
            copy_regular_file("tests/fixtures/gguf/valid-tokenizer-simple.gguf",
                              artifact_path) &&
            rewrite_attention_artifact_fixture(artifact_path),
        "v12 runtime artifact fixture created");
    YVEX_TEST_ASSERT(fixture_build(&fixture, artifact_path, 1),
                     "v12 runtime binding fixture built");
    descriptor = yvex_runtime_descriptor_summary_get(fixture.descriptor);
    YVEX_TEST_ASSERT(descriptor &&
                         descriptor->model_execution.schema_version ==
                             YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1,
                     "v12 fixture owns a sealed model execution descriptor");
    YVEX_TEST_ASSERT(fixture_binding_request(&fixture, directory, &request) &&
                         yvex_runtime_binding_prepare(
                             &request, &prepared, &failure, &err) == YVEX_OK,
                     "v12 runtime binding prepared");
    YVEX_TEST_ASSERT(yvex_runtime_binding_open(
                         &binding, prepared.path, &summary, NULL, &failure, &err) == YVEX_OK &&
                         summary.schema_version == YVEX_RUNTIME_BINDING_SCHEMA_CURRENT &&
                         strcmp(summary.model_execution_identity,
                                descriptor->model_execution.identity) == 0 &&
                         summary.semantic_maximum_context ==
                             descriptor->model_execution.maximum_context,
                     "v12 reader authenticates canonical operator and execution records");
    yvex_runtime_binding_close(binding);
    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         &fixture, &prepared, &model, &model_failure, &err) == YVEX_OK,
                     "v12 runtime model instantiates compiled execution geometry");
    session_request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &session_request,
                         &model_failure, &err) == YVEX_OK,
                     "v12 runtime session opens for capacity admission");
    generation_options.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    generation_options.backend = YVEX_BACKEND_KIND_CPU;
    generation_options.mode = YVEX_GENERATION_MODE_TARGET_ONLY;
    generation_options.context_capacity = 5ull;
    generation_options.prefill_chunk_tokens = 1ull;
    generation_options.maximum_new_tokens = 1ull;
    generation_options.maximum_output_bytes = 64ull;
    generation_options.evidence_profile = YVEX_EXECUTION_EVIDENCE_PRODUCTION;
    generation_options.sampling_policy = (yvex_runtime_sampling_policy){
        .schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1,
        .strategy = YVEX_SAMPLING_STRATEGY_GREEDY,
        .temperature = 1.0,
        .top_p = 1.0,
        .typical_p = 1.0};
    YVEX_TEST_ASSERT(yvex_runtime_generation_context_open(
                         &generation, model, session, &generation_options,
                         &err) == YVEX_ERR_BOUNDS && !generation,
                     "semantic context maximum refuses before state mutation");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "v12 capacity session closes");
    yvex_runtime_model_close(&model);
    fixture_close(&fixture);
    (void)unlink(prepared.path);
    (void)unlink(artifact_path);
    (void)rmdir(directory);
    return 0;
}

static int test_corruption_refusals(const yvex_runtime_binding_prepare_result *prepared,
                                    const char *root)
{
    const char *basename = strrchr(prepared->path, '/');
    const char *variants[] = {
        "truncated", "tail", "legacy-schema", "stale", "preoperator-v11"};
    const yvex_runtime_binding_failure_code expected[] = {
        YVEX_RUNTIME_BINDING_FAILURE_TRUNCATED,
        YVEX_RUNTIME_BINDING_FAILURE_TRAILING_DATA,
        YVEX_RUNTIME_BINDING_FAILURE_SCHEMA,
        YVEX_RUNTIME_BINDING_FAILURE_IDENTITY,
        YVEX_RUNTIME_BINDING_FAILURE_COMPATIBILITY};
    char directories[5][YVEX_PATH_CAP];
    char paths[5][YVEX_PATH_CAP];
    struct stat status;
    unsigned char value;
    unsigned int i;

    basename = basename ? basename + 1 : prepared->path;
    for (i = 0u; i < 5u; ++i) {
        int fd;
        yvex_runtime_binding *binding = NULL;
        yvex_runtime_binding_failure failure;
        yvex_error err;
        int rc;

        YVEX_TEST_ASSERT(
            variant_path(root, variants[i], basename, directories[i], paths[i]),
            "runtime binding variant path");
        YVEX_TEST_ASSERT(copy_regular_file(prepared->path, paths[i]),
                         "runtime binding variant copied");
        fd = open(paths[i], O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        YVEX_TEST_ASSERT(fd >= 0 && fstat(fd, &status) == 0,
                         "runtime binding variant opened");
        if (i == 0u) {
            YVEX_TEST_ASSERT(ftruncate(fd, status.st_size - 1) == 0,
                             "runtime binding truncated");
        } else if (i == 1u) {
            value = 0xa5u;
            YVEX_TEST_ASSERT(lseek(fd, 0, SEEK_END) >= 0 && write(fd, &value, 1u) == 1,
                             "runtime binding tail appended");
        } else if (i == 2u) {
            value = 3u;
            YVEX_TEST_ASSERT(pwrite(fd, &value, 1u, 8) == 1,
                             "legacy runtime binding schema written");
        } else if (i == 3u) {
            YVEX_TEST_ASSERT(pread(fd, &value, 1u, 96) == 1,
                             "runtime binding stale byte read");
            value ^= 0x5au;
            YVEX_TEST_ASSERT(pwrite(fd, &value, 1u, 96) == 1,
                             "runtime binding stale byte written");
        } else {
            unsigned char legacy_header[16] = "YVRBND11";
            test_binding_put_u64(legacy_header, 8u,
                                 YVEX_RUNTIME_BINDING_SCHEMA_CURRENT - 1u);
            YVEX_TEST_ASSERT(pwrite(fd, legacy_header, sizeof(legacy_header), 0) ==
                                 (ssize_t)sizeof(legacy_header),
                             "pre-operator-graph v11 header written");
        }
        YVEX_TEST_ASSERT(close(fd) == 0, "runtime binding variant closed");
        rc = yvex_runtime_binding_open(&binding, paths[i], NULL, NULL, &failure, &err);
        YVEX_TEST_ASSERT(rc != YVEX_OK && binding == NULL,
                         "corrupt runtime binding refused");
        YVEX_TEST_ASSERT(failure.code == expected[i],
                         "corrupt runtime binding typed failure");
        YVEX_TEST_ASSERT(unlink(paths[i]) == 0 && rmdir(directories[i]) == 0,
                         "corrupt runtime binding fixture cleaned");
    }
    return 0;
}

static int test_canonical_refusals(const yvex_runtime_binding_prepare_result *prepared,
                                   const char *root)
{
    const char *basename = strrchr(prepared->path, '/');
    const char *variants[] = {
        "u32-overflow", "noncanonical-text", "record-byte-budget",
        "nonbinary-capability"};
    char directories[4][YVEX_PATH_CAP], paths[4][YVEX_PATH_CAP];
    unsigned int index;

    basename = basename ? basename + 1 : prepared->path;
    for (index = 0u; index < 4u; ++index) {
        yvex_core_file_result file_result;
        yvex_runtime_binding_failure failure;
        yvex_runtime_binding *binding = NULL;
        unsigned char *file = NULL;
        char addressed_path[YVEX_PATH_CAP];
        size_t count = 0u, format_text, format_version, capability_value;
        size_t compatibility_schema;
        size_t material_count;
        yvex_error err;
        int rc;

        YVEX_TEST_ASSERT(
            variant_path(root, variants[index], basename, directories[index], paths[index]) &&
                copy_regular_file(prepared->path, paths[index]),
            "canonical refusal fixture copied");
        memset(&file_result, 0, sizeof(file_result));
        YVEX_TEST_ASSERT(
            yvex_core_file_read_snapshot(paths[index], 64u * 1024u * 1024u,
                                         &file, &count, &file_result, &err) == YVEX_OK &&
                test_binding_offsets(file, count, &format_text, &format_version,
                                     &capability_value, &compatibility_schema) &&
                test_binding_material_count_offset(file, count, &material_count),
            "canonical refusal offsets decoded");
        if (index == 0u)
            test_binding_put_u64(file, compatibility_schema,
                                 (unsigned long long)UINT_MAX + 1ull);
        else if (index == 1u)
            file[format_text] = '\0';
        else if (index == 2u)
            test_binding_put_u64(file, material_count, 1048576ull);
        else
            test_binding_put_u64(file, capability_value, 2ull);
        YVEX_TEST_ASSERT(test_binding_readdress(paths[index], file, count, addressed_path),
                         "canonical refusal fixture readdressed");
        free(file);
        rc = yvex_runtime_binding_open(
            &binding, addressed_path, NULL, NULL, &failure, &err);
        YVEX_TEST_ASSERT(
            rc == (index == 2u ? YVEX_ERR_BOUNDS : YVEX_ERR_FORMAT) && !binding &&
                failure.code == (index == 2u ? YVEX_RUNTIME_BINDING_FAILURE_BOUNDS
                                             : YVEX_RUNTIME_BINDING_FAILURE_FORMAT),
                         "authenticated noncanonical binding refused");
        YVEX_TEST_ASSERT(unlink(addressed_path) == 0 && rmdir(directories[index]) == 0,
                         "canonical refusal fixture cleaned");
    }
    return 0;
}

/* Prove a valid outer content hash cannot authenticate either false graph identity. */
static int test_graph_identity_refusals(
    const yvex_runtime_binding_prepare_result *prepared, const char *root)
{
    static const char *variants[] = {"semantic-graph", "executable-graph"};
    static const char *fields[] = {
        "semantic-graph-identity", "executable-graph-identity"};
    const char *basename = strrchr(prepared->path, '/');
    char directories[2][YVEX_PATH_CAP], paths[2][YVEX_PATH_CAP];
    unsigned int index;

    basename = basename ? basename + 1 : prepared->path;
    for (index = 0u; index < 2u; ++index) {
        yvex_core_file_result file_result = {0};
        yvex_runtime_binding_failure failure;
        yvex_runtime_binding *binding = NULL;
        unsigned char *file = NULL;
        char addressed_path[YVEX_PATH_CAP];
        size_t count = 0u, semantic, executable;
        yvex_error err;
        int rc;

        YVEX_TEST_ASSERT(
            variant_path(root, variants[index], basename, directories[index], paths[index]) &&
                copy_regular_file(prepared->path, paths[index]) &&
                yvex_core_file_read_snapshot(paths[index], 64u * 1024u * 1024u,
                                             &file, &count, &file_result, &err) == YVEX_OK &&
                test_binding_graph_identity_offsets(
                    file, count, &semantic, &executable),
            "graph identity mutation fixture decoded");
        memset(file + (index == 0u ? semantic : executable),
               index == 0u ? 'd' : 'e', 64u);
        YVEX_TEST_ASSERT(test_binding_readdress(paths[index], file, count, addressed_path),
                         "graph identity mutation receives a valid outer content address");
        free(file);
        rc = yvex_runtime_binding_open(
            &binding, addressed_path, NULL, NULL, &failure, &err);
        YVEX_TEST_ASSERT(
            rc == YVEX_ERR_FORMAT && !binding &&
                failure.code == YVEX_RUNTIME_BINDING_FAILURE_IDENTITY &&
                strcmp(failure.field, fields[index]) == 0,
            "independent reopen rejects one canonically false graph identity");
        YVEX_TEST_ASSERT(unlink(addressed_path) == 0 && rmdir(directories[index]) == 0,
                         "graph identity mutation fixture cleaned");
    }
    return 0;
}

static int test_artifact_copy_portability(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared,
    const char *root)
{
    char copy_path[YVEX_PATH_CAP], binding_directory[YVEX_PATH_CAP];
    yvex_complete_artifact_admission portable_admission;
    yvex_runtime_binding_prepare_request request;
    yvex_runtime_binding_prepare_result portable_result;
    yvex_runtime_binding *portable_binding = NULL;
    yvex_runtime_binding_failure binding_failure;
    yvex_artifact_options artifact_options;
    yvex_artifact *replacement = NULL;
    yvex_artifact_snapshot replacement_snapshot;
    yvex_materialization_options options;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure material_failure;
    const yvex_materialized_tensor_binding *tensor;
    yvex_error err;
    unsigned char byte;
    int descriptor, rc;

    YVEX_TEST_ASSERT(
        snprintf(copy_path, sizeof(copy_path), "%s/artifact-copy.gguf", root) <
                (int)sizeof(copy_path) &&
            snprintf(binding_directory, sizeof(binding_directory), "%s/portable-binding", root) <
                (int)sizeof(binding_directory) &&
            mkdir(binding_directory, 0700) == 0,
        "portable binding paths prepared");
    YVEX_TEST_ASSERT(copy_regular_file(yvex_artifact_path(fixture->artifact), copy_path),
                         "replacement artifact copied");
    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = copy_path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&replacement, &artifact_options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         yvex_artifact_snapshot_get(
                             replacement, &replacement_snapshot, &err) == YVEX_OK &&
                         (replacement_snapshot.device != fixture->admission.file_snapshot.device ||
                          replacement_snapshot.inode != fixture->admission.file_snapshot.inode),
                     "replacement artifact opened with independent snapshot");
    portable_admission = fixture->admission;
    portable_admission.file_snapshot = replacement_snapshot;
    (void)snprintf(portable_admission.artifact_path,
                   sizeof(portable_admission.artifact_path), "%s", copy_path);
    YVEX_TEST_ASSERT(fixture_binding_request(fixture, binding_directory, &request),
                     "portable binding request declares adapter capabilities");
    request.admission = &portable_admission;
    memset(&portable_result, 0, sizeof(portable_result));
    rc = yvex_runtime_binding_prepare(
        &request, &portable_result, &binding_failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && portable_result.published &&
            strcmp(portable_result.summary.identity, prepared->summary.identity) == 0 &&
            regular_files_equal(portable_result.path, prepared->path),
        "byte-identical artifact copies produce one portable binding identity");
    rc = yvex_runtime_binding_open(
        &portable_binding, portable_result.path, NULL, NULL, &binding_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "portable binding independently reopened");
    yvex_materialization_options_default(&options);
    rc = yvex_runtime_binding_import_materialization(
        portable_binding, replacement, &options, &plan, &session, &binding_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && plan && session,
                     "portable binding leases the independently reopened artifact snapshot");
    tensor = yvex_materialization_session_tensor_at(session, 0ull);
    descriptor = open(copy_path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(descriptor >= 0 && pread(descriptor, &byte, 1u, 0) == 1,
                     "leased artifact opened for drift mutation");
    byte ^= 0x5au;
    YVEX_TEST_ASSERT(pwrite(descriptor, &byte, 1u, 0) == 1 && close(descriptor) == 0,
                     "leased artifact mutated after materialization commit");
    rc = yvex_materialization_session_read(
        session, tensor, 0ull, &byte, 1u, &material_failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         material_failure.code == YVEX_MATERIALIZATION_FAILURE_SNAPSHOT_DRIFT,
                     "process-local replacement snapshot drift invalidates the imported lease");
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_runtime_binding_close(portable_binding);
    yvex_artifact_close(replacement);
    YVEX_TEST_ASSERT(unlink(portable_result.path) == 0 &&
                         rmdir(binding_directory) == 0 && unlink(copy_path) == 0,
                     "portable binding and replacement artifact cleaned");
    return 0;
}

static int runtime_model_open_fixture(const binding_fixture *fixture,
                                      const yvex_runtime_binding_prepare_result *prepared,
                                      yvex_runtime_model **model,
                                      yvex_runtime_model_failure *failure,
                                      yvex_error *err)
{
    yvex_runtime_model_open_request request;

    memset(&request, 0, sizeof(request));
    request.artifact_path = yvex_artifact_path(fixture->artifact);
    request.runtime_binding_path = prepared->path;
    request.target_id = runtime_fixture_execution()->target_id;
    return yvex_runtime_model_open(model, &request, failure, err);
}

typedef struct {
    unsigned long long events[YVEX_RUNTIME_LIFECYCLE_COUNT];
    unsigned long long hash_completed, hash_total;
    int cancel_hash, restrict_memory_after_hash, memory_restricted;
} runtime_progress_fixture;

static int runtime_progress_collect(void *opaque, yvex_runtime_lifecycle_phase phase,
                                    unsigned long long completed,
                                    unsigned long long total)
{
    runtime_progress_fixture *progress = (runtime_progress_fixture *)opaque;

    if ((unsigned int)phase >= YVEX_RUNTIME_LIFECYCLE_COUNT)
        return 0;
    progress->events[phase]++;
    if (phase == YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH) {
        progress->hash_completed = completed;
        progress->hash_total = total;
        if (progress->restrict_memory_after_hash && total && completed == total &&
            !progress->memory_restricted &&
            setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "1", 1) == 0)
            progress->memory_restricted = 1;
        if (progress->cancel_hash && completed)
            return 0;
    }
    return 1;
}

static int test_runtime_model_progress(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model_open_request request;
    yvex_runtime_model_failure failure;
    runtime_progress_fixture progress;
    yvex_runtime_binding *binding = NULL;
    yvex_runtime_binding_summary binding_summary;
    yvex_complete_artifact_admission binding_admission;
    yvex_runtime_binding_failure binding_failure;
    yvex_runtime_model *model = NULL;
    yvex_runtime_model_summary summary;
    yvex_error err;
    unsigned long long reserve, transient, baseline_required;
    int capacity_rc;

    memset(&request, 0, sizeof(request));
    memset(&progress, 0, sizeof(progress));
    request.artifact_path = yvex_artifact_path(fixture->artifact);
    request.runtime_binding_path = prepared->path;
    request.target_id = runtime_fixture_execution()->target_id;
    request.progress = runtime_progress_collect;
    request.progress_context = &progress;
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_OK && model,
                     "runtime progress model opens");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK,
                     "runtime progress model summary copies safely");
    YVEX_TEST_ASSERT(progress.events[YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN] == 1ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN] == 2ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL] == 2ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] > 1ull &&
                         progress.hash_completed == fixture->admission.file_bytes &&
                         progress.hash_total == fixture->admission.file_bytes,
                     "runtime progress reports exact cold hash coverage");
    YVEX_TEST_ASSERT(summary.total_seconds >= 0.0 &&
                         summary.lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] >= 0.0 &&
                         summary.lifecycle_seconds[
                             YVEX_RUNTIME_LIFECYCLE_ARTIFACT_ADMISSION] > 0.0 &&
                         summary.lifecycle_seconds[
                             YVEX_RUNTIME_LIFECYCLE_MATERIALIZATION_OPEN] >= 0.0,
                     "runtime model retains typed phase timing");
    yvex_runtime_model_close(&model);
    memset(&binding_failure, 0, sizeof(binding_failure));
    YVEX_TEST_ASSERT(
        yvex_runtime_binding_open(
            &binding, prepared->path, &binding_summary, &binding_admission,
            &binding_failure, &err) == YVEX_OK && binding &&
            runtime_binding_maximum_tensor_bytes(
                binding, &transient),
        "runtime binding exposes one exact transient tensor extent");
    yvex_runtime_binding_close(binding);
    binding = NULL;
    memset(&progress, 0, sizeof(progress));
    reserve = yvex_runtime_private_system_reserve(128ull * 1024ull * 1024ull * 1024ull);
    YVEX_TEST_ASSERT(
        reserve == 16ull * 1024ull * 1024ull * 1024ull &&
            setenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES", "137438953472", 1) == 0 &&
            setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "137438953472", 1) == 0,
        "model admission installs deterministic proportional reserve facts");
    baseline_required = fixture->admission.payload_bytes +
                        YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE + transient;
    request.maximum_host_bytes = baseline_required - 1ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_model_open(&model, &request, &failure, &err) == YVEX_ERR_BOUNDS &&
            !model && failure.code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION &&
            strcmp(failure.field, "model-host-budget") == 0 &&
            failure.expected == baseline_required &&
            failure.actual == request.maximum_host_bytes &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 0ull,
        "model open preserves the minimum reserve inside a configured host budget");
    request.maximum_host_bytes = 0ull;
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "17179869183", 1) == 0,
        "inject capacity below the proportional system reserve");
    capacity_rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    YVEX_TEST_ASSERT(
        capacity_rc == YVEX_ERR_BOUNDS && !model &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION &&
            strcmp(failure.field, "system-memory-capacity") == 0 &&
            failure.expected == fixture->admission.payload_bytes + reserve + transient &&
            failure.actual == 17179869183ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 0ull,
        "model open derives a proportional reserve from effective system capacity");
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES") == 0 &&
            unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES") == 0,
        "model admission clears deterministic proportional reserve facts");
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "1", 1) == 0,
                     "inject system memory capacity refusal");
    capacity_rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES") == 0,
                     "clear system memory capacity refusal");
    YVEX_TEST_ASSERT(
        capacity_rc == YVEX_ERR_BOUNDS &&
            !model && failure.code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION &&
            strcmp(failure.field, "system-memory-capacity") == 0 &&
            failure.expected > fixture->admission.payload_bytes && failure.actual == 1ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 0ull,
        "model open preserves the system reserve before artifact mutation");
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "18446744073709551615", 1) == 0 &&
            setenv("YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES", "1", 1) == 0,
        "inject process memory capacity refusal");
    capacity_rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES") == 0 &&
            unsetenv("YVEX_TEST_RUNTIME_CGROUP_AVAILABLE_MEMORY_BYTES") == 0,
        "clear process memory capacity refusal");
    YVEX_TEST_ASSERT(
        capacity_rc == YVEX_ERR_BOUNDS && !model &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION &&
            strcmp(failure.field, "process-memory-capacity") == 0 &&
            failure.expected > fixture->admission.payload_bytes && failure.actual == 1ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 0ull,
        "model open preserves the cgroup reserve before artifact mutation");
    memset(&progress, 0, sizeof(progress));
    progress.restrict_memory_after_hash = 1;
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES", "137438953472", 1) == 0 &&
            setenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES", "137438953472", 1) == 0,
        "inject sufficient first-pass memory capacity");
    capacity_rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_TOTAL_MEMORY_BYTES") == 0 &&
            unsetenv("YVEX_TEST_RUNTIME_AVAILABLE_MEMORY_BYTES") == 0,
        "clear just-in-time memory-capacity injection");
    YVEX_TEST_ASSERT(
        capacity_rc == YVEX_ERR_BOUNDS && !model && progress.memory_restricted &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_ALLOCATION &&
            strcmp(failure.field, "system-memory-capacity") == 0 &&
            failure.actual == 1ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN] == 1ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] > 1ull &&
            progress.events[YVEX_RUNTIME_LIFECYCLE_RESIDENCY] == 0ull,
        "model open rechecks live capacity after hashing and before residency mutation");
    memset(&progress, 0, sizeof(progress));
    progress.cancel_hash = 1;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_MODEL_CLEANUP_FAILURE", "1", 1) == 0,
                     "inject failed-open model cleanup refusal");
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_ERR_STATE && model &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_CLEANUP &&
                         strcmp(failure.field, "model-open-cleanup") == 0,
                     "failed model cleanup publishes its exact retry owner");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_MODEL_CLEANUP_FAILURE") == 0,
                     "clear failed-open model cleanup refusal");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model, "failed-open model cleanup retry discharges ownership");
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_ERR_CANCELLED && !model,
                     "runtime hash progress cancellation refuses model publication");
    return 0;
}

static int test_runtime_family_neutrality(void)
{
    const yvex_graph_execution_binding *deepseek =
        yvex_graph_execution_find(0ull, 0ull, "deepseek4-v4-flash-dspark");
    const yvex_family_compiler_adapter *compiler =
        yvex_compiler_family_deepseek_v4();
    yvex_compilation_runtime_binding_request request = {0};
    yvex_compilation_runtime_binding_result rejected = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(deepseek != NULL && deepseek->api != NULL,
                     "compiled family identity resolves through graph execution registry");
    YVEX_TEST_ASSERT(strcmp(deepseek->operator_family_key, "deepseek") == 0 &&
                         strcmp(deepseek->source_manifest_filename,
                                YVEX_SOURCE_RELEASE_MANIFEST_LEAF) == 0 &&
                         strcmp(deepseek->operator_artifact_filename,
                                YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME) == 0 &&
                         deepseek->model && deepseek->compiler == compiler &&
                         compiler && compiler->binding_pipeline &&
                         compiler->binding_compile == yvex_family_binding_compile &&
                         compiler->binding_pipeline->source_open &&
                         compiler->binding_pipeline->semantic_model_build &&
                         deepseek->model() == yvex_model_register_deepseek_v4(),
                     "compiler preparation facts remain separate from execution facts");
    YVEX_TEST_ASSERT(yvex_runtime_binding_compile_publish(
                         compiler, &request, rejected.path,
                         &rejected.published, &err) == YVEX_ERR_INVALID_ARG &&
                         !rejected.published && !rejected.path[0],
                     "generic publication owner refuses incomplete family compiler products");
    YVEX_TEST_ASSERT(yvex_graph_execution_find(
                         deepseek->adapter_id, deepseek->adapter_version, NULL) == deepseek,
                     "compiled adapter identity selects one immutable execution binding");
    YVEX_TEST_ASSERT(yvex_graph_execution_find(
                         0ull, 0ull, "not-a-runtime-family") == NULL &&
                         yvex_graph_execution_find(deepseek->adapter_id, 0ull, NULL) == NULL,
                     "unknown target and execution version are refused");
    return 0;
}

static int test_runtime_model_verified_reopen(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared,
    const char *root)
{
    char cache_root[YVEX_PATH_CAP], receipt_root[YVEX_PATH_CAP], artifact_dir[YVEX_PATH_CAP];
    yvex_runtime_model_open_request request;
    yvex_runtime_model_failure failure;
    runtime_progress_fixture progress;
    yvex_runtime_model *model = NULL;
    yvex_runtime_model_summary summary;
    yvex_artifact_reopen_lease lease;
    yvex_error err;
    unsigned char corrupt = 0u;
    int count, descriptor;

    count = snprintf(cache_root, sizeof(cache_root), "%s/reopen-cache", root);
    YVEX_TEST_ASSERT(count > 0 && count < (int)sizeof(cache_root),
                     "runtime verified-reopen cache root is bounded");
    memset(&request, 0, sizeof(request));
    request.artifact_path = yvex_artifact_path(fixture->artifact);
    request.runtime_binding_path = prepared->path;
    request.target_id = runtime_fixture_execution()->target_id;
    request.artifact_reopen_cache_root = cache_root;
    request.progress = runtime_progress_collect;
    request.progress_context = &progress;
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_OK && model &&
                         yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK &&
                         summary.artifact_hash_passes == 1ull &&
                         summary.artifact_verified_reopen_passes == 0ull &&
                         summary.artifact_reopen_cache_failures == 0ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] > 1ull,
                     "first runtime open verifies every artifact byte and publishes a lease");
    yvex_runtime_model_close(&model);
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_OK && model &&
                         yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK &&
                         summary.artifact_hash_passes == 0ull &&
                         summary.artifact_verified_reopen_passes == 1ull &&
                         summary.artifact_reopen_cache_failures == 0ull &&
                         summary.artifact_bytes_hashed == 0ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] == 0ull,
                     "unchanged runtime open consumes exact prior verification without rehashing");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(yvex_artifact_reopen_lease_check(
                         fixture->artifact, fixture->admission.artifact_identity,
                         cache_root, &lease, &err) == YVEX_OK && lease.verified,
                     "runtime published lease remains independently verifiable");
    descriptor = open(lease.path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(descriptor >= 0 && pwrite(descriptor, &corrupt, 1u, 0) == 1 &&
                         fsync(descriptor) == 0 && close(descriptor) == 0,
                     "runtime verified-reopen corruption is durable");
    memset(&progress, 0, sizeof(progress));
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_OK && model &&
                         yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK &&
                         summary.artifact_hash_passes == 1ull &&
                         summary.artifact_verified_reopen_passes == 0ull &&
                         summary.artifact_reopen_cache_failures == 1ull &&
                         progress.events[YVEX_RUNTIME_LIFECYCLE_ARTIFACT_HASH] > 1ull,
                     "invalid runtime lease falls back to complete verification visibly");
    yvex_runtime_model_close(&model);
    count = snprintf(artifact_dir, sizeof(artifact_dir), "%s/artifact-reopen/%s",
                     cache_root, fixture->admission.artifact_identity);
    YVEX_TEST_ASSERT(count > 0 && count < (int)sizeof(artifact_dir),
                     "runtime verified-reopen artifact directory is bounded");
    count = snprintf(receipt_root, sizeof(receipt_root), "%s/artifact-reopen", cache_root);
    YVEX_TEST_ASSERT(count > 0 && count < (int)sizeof(receipt_root) &&
                         unlink(lease.path) == 0 && rmdir(artifact_dir) == 0 &&
                         rmdir(receipt_root) == 0 && rmdir(cache_root) == 0,
                     "runtime verified-reopen cache is cleaned narrowly");
    return 0;
}

static int test_runtime_model_session_reuse(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *first = NULL, *second = NULL, *undersized = NULL;
    yvex_runtime_model_failure failure;
    yvex_attention_failure attention_failure;
    yvex_runtime_session_open_request request;
    yvex_runtime_model_summary model_summary;
    yvex_runtime_session_summary session_summary, copied_summary;
    const yvex_attention_workspace_summary *workspace_summary;
    const yvex_attention_state_provider *state;
    yvex_graph_attention_state_summary state_summary;
    yvex_materialization_session *materialization;
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_access_summary before, after_warm;
    yvex_core_execution_observation execution_before, execution_after, execution_delta;
    unsigned char output[16];
    float *first_span, *replayed_span;
    yvex_error err;
    int rc;

    yvex_core_execution_observation_snapshot(&execution_before);
    rc = runtime_model_open_fixture(fixture, prepared, &model, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && model != NULL, "binding-backed runtime model opens");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &model_summary, &err) == YVEX_OK &&
                         model_summary.sealed && model_summary.valid,
                     "runtime model sealed and valid");
    YVEX_TEST_ASSERT(model_summary.artifact_hash_passes == 1ull &&
                         model_summary.artifact_bytes_hashed == fixture->admission.file_bytes,
                     "runtime model performs one cold artifact hash");
    YVEX_TEST_ASSERT(model_summary.gguf_directory_parses == 1ull &&
                         model_summary.runtime_binding_parses == 1ull &&
                         model_summary.runtime_model_builds == 1ull &&
                         model_summary.runtime_descriptor_builds == 1ull &&
                         model_summary.semantic_graph_builds == 1ull &&
                         model_summary.executable_graph_builds == 1ull,
                     "runtime model builds each immutable owner once");
    YVEX_TEST_ASSERT(model_summary.attention_layer_count == 1ull &&
                         model_summary.attention_binding_count == 1ull,
                     "bounded runtime model retains its exact graph counts");
    YVEX_TEST_ASSERT(model_summary.capabilities.attention_core_ready == 1 &&
                         model_summary.capabilities.attention_envelope_ready == 1 &&
                         model_summary.capabilities.cpu_prefill_eager_ready == 1 &&
                         model_summary.capabilities.cuda_eager_implemented == 1 &&
                         model_summary.capabilities.cuda_piecewise_graph_implemented == 1 &&
                         model_summary.capabilities.cuda_full_graph_implemented == 1 &&
                         model_summary.capabilities.attention_workspace_ready == 0 &&
                         model_summary.capabilities.attention_state_delta_ready == 1 &&
                         model_summary.capabilities.attention_trace_ready == 1 &&
                         model_summary.capabilities.attention_profile_ready == 1 &&
                         model_summary.capabilities.attention_benchmark_ready == 1 &&
                         model_summary.capabilities.cuda_prefill_eager_ready == 0 &&
                         model_summary.capabilities.cuda_decode_eager_ready == 0 &&
                         model_summary.capabilities.cuda_prefill_piecewise_graph_ready == 0 &&
                         model_summary.capabilities.cuda_prefill_full_graph_ready == 0 &&
                         model_summary.capabilities.persistent_kv_ready == 0 &&
                         model_summary.capabilities.generation_ready == 0,
                     "runtime model publishes implementation facts without resource promotion");

    materialization = yvex_runtime_model_view_get(model)->materialization;
    binding = yvex_materialization_session_tensor_at(materialization, 0ull);
    YVEX_TEST_ASSERT(binding && binding->encoded_bytes == 256ull &&
                         binding->role == YVEX_TENSOR_ROLE_ATTENTION_SINKS,
                     "runtime model materialization binding available");
    YVEX_TEST_ASSERT(yvex_materialization_session_access_summary(
                         materialization, &before, &err) == YVEX_OK,
                     "runtime materialization initial access summary");
    YVEX_TEST_ASSERT(yvex_materialization_session_read(
                         materialization, binding, 3ull, output, sizeof(output), NULL,
                         &err) == YVEX_OK &&
                         yvex_materialization_session_read(
                             materialization, binding, 19ull, output, sizeof(output), NULL,
                             &err) == YVEX_OK,
                     "runtime warm reads use resident provider");
    YVEX_TEST_ASSERT(yvex_materialization_session_access_summary(
                         materialization, &after_warm, &err) == YVEX_OK &&
                         after_warm.artifact_read_calls == before.artifact_read_calls &&
                         after_warm.resident_read_calls == before.resident_read_calls + 2ull,
                     "runtime warm reads perform zero artifact access");

    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.maximum_host_bytes = 1ull;
    rc = yvex_runtime_session_open(&undersized, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !undersized &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_GRAPH,
                     "undersized session host budget refuses before publication");
    request.maximum_host_bytes = 0ull;
    rc = yvex_runtime_session_open(&first, model, &request, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_session_open(&second, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && first && second, "two warm CPU sessions open");
    YVEX_TEST_ASSERT(yvex_runtime_session_summary_copy(
                         first, &session_summary, &err) == YVEX_OK,
                     "first runtime session summary copies safely");
    workspace_summary = yvex_attention_workspace_summary_get(
        yvex_runtime_session_view_get(first)->attention_workspace);
    YVEX_TEST_ASSERT(workspace_summary && session_summary.workspace_bytes > 0ull &&
                         workspace_summary->capacity_bytes == session_summary.workspace_bytes &&
                         session_summary.capabilities.attention_workspace_ready == 1 &&
                         session_summary.capabilities.cpu_prefill_eager_ready == 1 &&
                         session_summary.capabilities.cuda_prefill_eager_ready == 0 &&
                         session_summary.capabilities.cuda_prefill_full_graph_ready == 0 &&
                         session_summary.warm_host_allocations == 0ull,
                     "session derives one cold arena and reports zero warm heap allocation");
    state = yvex_runtime_session_view_get(first)->attention_state_provider;
    YVEX_TEST_ASSERT(
        runtime_state_prepare_fixture(model, state, 0ull, &attention_failure,
                                      &err) == YVEX_OK,
        "first session prepares one bounded attention state layer");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(first, &failure, &err) == YVEX_OK,
                     "first session acquires execution");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(first, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_BUSY,
                     "same mutable session refuses concurrent execution");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(second, &failure, &err) == YVEX_OK,
                     "different mutable session executes independently");
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_begin(
            yvex_runtime_session_view_get(first)->attention_workspace, &err) == YVEX_OK,
        "session arena begins its first warm graph lifetime");
    first_span = (float *)yvex_attention_workspace_calloc(
        yvex_runtime_session_view_get(first)->attention_workspace, 16ull, sizeof(float));
    YVEX_TEST_ASSERT(first_span &&
                         yvex_attention_workspace_rewind(
                             yvex_runtime_session_view_get(first)->attention_workspace,
                             0ull, &err) == YVEX_OK &&
                         yvex_attention_workspace_finish(
                             yvex_runtime_session_view_get(first)->attention_workspace,
                             &err) == YVEX_OK,
                     "first warm graph lifetime retires without freeing arena storage");
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_begin(
            yvex_runtime_session_view_get(first)->attention_workspace, &err) == YVEX_OK,
        "session arena begins a second warm graph lifetime");
    replayed_span = (float *)yvex_attention_workspace_calloc(
        yvex_runtime_session_view_get(first)->attention_workspace, 16ull, sizeof(float));
    YVEX_TEST_ASSERT(replayed_span == first_span &&
                         yvex_attention_workspace_rewind(
                             yvex_runtime_session_view_get(first)->attention_workspace,
                             0ull, &err) == YVEX_OK &&
                         yvex_attention_workspace_finish(
                             yvex_runtime_session_view_get(first)->attention_workspace,
                             &err) == YVEX_OK,
                     "warm graph replay reuses the same stable arena address");
    YVEX_TEST_ASSERT(yvex_runtime_session_finish(second, YVEX_OK, &err) == YVEX_OK,
                     "second session finishes without cleanup failure");
    YVEX_TEST_ASSERT(
        runtime_state_begin_fixture(model, state, 0ull, 0ull, 1ull,
                                    &attention_failure, &err) == YVEX_OK &&
            yvex_runtime_session_finish(first, YVEX_ERR_IO, &err) == YVEX_ERR_IO &&
            state->summary(state->context, &state_summary, &err) == YVEX_OK &&
            !state_summary.transaction_active && state_summary.abort_count == 1ull &&
            state_summary.components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].entry_count == 0ull,
        "failed execution aborts its candidate without changing committed state");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(first, &failure, &err) == YVEX_OK &&
                         yvex_runtime_session_finish(
                             first, YVEX_ERR_CANCELLED, &err) == YVEX_ERR_CANCELLED,
                     "request-scoped cancellation is accounted at execution finish");
    YVEX_TEST_ASSERT(yvex_runtime_session_summary_copy(
                         first, &session_summary, &err) == YVEX_OK &&
                         session_summary.failure_count == 1ull &&
                         session_summary.cancellation_count == 1ull &&
                         !session_summary.invalidated &&
                         session_summary.warm_artifact_hash_passes == 0ull &&
                         session_summary.warm_weight_artifact_reads == 0ull &&
                         session_summary.warm_host_allocations == 0ull &&
                         session_summary.workspace_allocation_count == 2ull &&
                         session_summary.workspace_peak_bytes == 16ull * sizeof(float) &&
                         session_summary.workspace_peak_bytes <= session_summary.workspace_bytes &&
                         session_summary.workspace_capacity_failure_count == 0ull,
                     "warm session counters preserve trust and refusal facts");
    YVEX_TEST_ASSERT(yvex_runtime_session_summary_copy(
                         second, &session_summary, &err) == YVEX_OK &&
                         session_summary.execution_count == 1ull,
                     "second session success accounted independently");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_begin(second, &failure, &err) == YVEX_OK &&
            setenv("YVEX_TEST_RUNTIME_SESSION_COUNTER_OVERFLOW", "1", 1) == 0 &&
            yvex_runtime_session_finish(second, YVEX_OK, &err) == YVEX_ERR_BOUNDS &&
            unsetenv("YVEX_TEST_RUNTIME_SESSION_COUNTER_OVERFLOW") == 0 &&
            yvex_runtime_session_summary_copy(second, &copied_summary, &err) == YVEX_OK &&
            copied_summary.invalidated,
        "counter overflow is fail-closed rather than success-shaped");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_begin(first, &failure, &err) == YVEX_OK &&
            runtime_state_begin_fixture(model, state, 0ull, 0ull, 1ull,
                                        &attention_failure, &err) == YVEX_OK &&
            yvex_runtime_session_finish(first, YVEX_OK, &err) == YVEX_ERR_STATE &&
            state->summary(state->context, &state_summary, &err) == YVEX_OK &&
            !state_summary.transaction_active && state_summary.abort_count == 2ull,
        "success refuses and aborts an unpublished state candidate");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &model_summary, &err) == YVEX_OK &&
                         model_summary.artifact_hash_passes == 1ull &&
                         model_summary.runtime_model_builds == 1ull &&
                         model_summary.runtime_descriptor_builds == 1ull,
                     "warm sessions do not rebuild immutable runtime model");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&second, &err) == YVEX_OK && !second,
                     "second runtime session closes without cleanup failure");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&first, &err) == YVEX_OK && !first,
                     "first runtime session closes without cleanup failure");
    yvex_core_execution_observation_snapshot(&execution_after);
    YVEX_TEST_ASSERT(
        yvex_core_execution_observation_delta(
            &execution_before, &execution_after, &execution_delta) &&
            !execution_delta.source_headers_read &&
            !execution_delta.source_payload_bytes_read &&
            !execution_delta.transform_plans_built &&
            !execution_delta.quant_plans_built &&
            !execution_delta.writer_plans_built,
        "runtime model and warm sessions enter no source or planning owner");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(NULL, &err) == YVEX_OK,
                     "null runtime session close is idempotent");
    yvex_runtime_model_close(&model);
    return 0;
}

static int test_runtime_concurrent_session_isolation(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *sessions[2] = {NULL, NULL};
    yvex_runtime_model_failure failure;
    runtime_thread_gate open_gate, execute_gate;
    runtime_open_thread openings[2];
    runtime_execute_thread contexts[2];
    pthread_t open_threads[2], execute_threads[2];
    yvex_runtime_session_summary summary;
    yvex_error err;
    unsigned int index;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for concurrent sessions");
    YVEX_TEST_ASSERT(runtime_thread_gate_init(&open_gate),
                     "concurrent open gate initialized");
    memset(openings, 0, sizeof(openings));
    for (index = 0u; index < 2u; ++index) {
        openings[index].model = model;
        openings[index].gate = &open_gate;
        YVEX_TEST_ASSERT(pthread_create(
                             &open_threads[index], NULL, runtime_open_thread_main,
                             &openings[index]) == 0,
                         "concurrent session-open thread created");
    }
    runtime_thread_gate_wait_ready(&open_gate, 2u);
    runtime_thread_gate_release(&open_gate);
    for (index = 0u; index < 2u; ++index) {
        YVEX_TEST_ASSERT(pthread_join(open_threads[index], NULL) == 0,
                         "concurrent session-open thread joined");
        YVEX_TEST_ASSERT(openings[index].open_status == YVEX_OK &&
                             openings[index].failure_code ==
                                 YVEX_RUNTIME_MODEL_FAILURE_NONE &&
                             openings[index].session != NULL,
                         "independent runtime session opens concurrently");
        sessions[index] = openings[index].session;
    }
    runtime_thread_gate_destroy(&open_gate);

    YVEX_TEST_ASSERT(runtime_thread_gate_init(&execute_gate),
                     "concurrent execution gate initialized");
    memset(contexts, 0, sizeof(contexts));
    for (index = 0u; index < 2u; ++index) {
        contexts[index].session = sessions[index];
        contexts[index].gate = &execute_gate;
        YVEX_TEST_ASSERT(pthread_create(
                             &execute_threads[index], NULL, runtime_execute_thread_main,
                             &contexts[index]) == 0,
                         "concurrent execution thread created");
    }
    runtime_thread_gate_wait_ready(&execute_gate, 2u);
    YVEX_TEST_ASSERT(contexts[0].begin_status == YVEX_OK &&
                         contexts[1].begin_status == YVEX_OK,
                     "different sessions hold execution ownership concurrently");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(
                         sessions[0], &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_BUSY,
                     "same session refuses a concurrent execution owner");
    runtime_thread_gate_release(&execute_gate);
    for (index = 0u; index < 2u; ++index)
        YVEX_TEST_ASSERT(pthread_join(execute_threads[index], NULL) == 0,
                         "concurrent execution thread joined");
    for (index = 0u; index < 2u; ++index) {
        YVEX_TEST_ASSERT(yvex_runtime_session_summary_copy(
                             sessions[index], &summary, &err) == YVEX_OK &&
                             summary.execution_count == 1ull && !summary.busy,
                         "concurrent session completion remains isolated");
    }
    runtime_thread_gate_destroy(&execute_gate);
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&sessions[1], &err) == YVEX_OK,
                     "second concurrent session closes cleanly");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&sessions[0], &err) == YVEX_OK,
                     "first concurrent session closes cleanly");
    yvex_runtime_model_close(&model);
    return 0;
}

static int test_runtime_concurrent_close_drain(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *anchor = NULL;
    yvex_runtime_session_open_request request;
    yvex_runtime_model_failure failure;
    runtime_thread_gate execute_gate, open_gate;
    runtime_execute_thread execution;
    runtime_open_thread opening;
    pthread_t execute_thread, open_thread;
    yvex_error err;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for concurrent close drain");
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &anchor, model, &request, &failure, &err) == YVEX_OK,
                     "anchor session reserves model lifetime");
    YVEX_TEST_ASSERT(runtime_thread_gate_init(&execute_gate) &&
                         runtime_thread_gate_init(&open_gate),
                     "close-drain gates initialized");
    memset(&execution, 0, sizeof(execution));
    execution.session = anchor;
    execution.gate = &execute_gate;
    YVEX_TEST_ASSERT(pthread_create(
                         &execute_thread, NULL, runtime_execute_thread_main,
                         &execution) == 0,
                     "anchor execution thread created");
    runtime_thread_gate_wait_ready(&execute_gate, 1u);
    YVEX_TEST_ASSERT(execution.begin_status == YVEX_OK,
                     "anchor execution owns session before close");

    memset(&opening, 0, sizeof(opening));
    opening.model = model;
    opening.gate = &open_gate;
    YVEX_TEST_ASSERT(pthread_create(
                         &open_thread, NULL, runtime_open_thread_main, &opening) == 0,
                     "concurrent session-open thread created");
    runtime_thread_gate_wait_ready(&open_gate, 1u);
    runtime_thread_gate_release(&open_gate);
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model, "close clears the owner during concurrent drain");
    YVEX_TEST_ASSERT(pthread_join(open_thread, NULL) == 0,
                     "concurrent session-open thread joined");
    YVEX_TEST_ASSERT(
        (opening.open_status == YVEX_OK && opening.session != NULL) ||
            (opening.open_status != YVEX_OK && opening.session == NULL &&
             opening.failure_code == YVEX_RUNTIME_MODEL_FAILURE_BUSY),
        "session open either publishes before close or refuses the draining model");
    if (opening.session)
        YVEX_TEST_ASSERT(yvex_runtime_session_close(&opening.session, &err) == YVEX_OK,
                         "concurrent opened session closes cleanly");
    runtime_thread_gate_release(&execute_gate);
    YVEX_TEST_ASSERT(pthread_join(execute_thread, NULL) == 0,
                     "anchor execution thread joined");
    runtime_thread_gate_destroy(&open_gate);
    runtime_thread_gate_destroy(&execute_gate);
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&anchor, &err) == YVEX_OK,
                     "anchor runtime session closes after drain");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model, "post-drain repeated model close is idempotent");
    return 0;
}

static int test_runtime_model_close_drain(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL, *refused = NULL;
    yvex_runtime_execution_session *retained;
    yvex_runtime_model_failure failure;
    yvex_runtime_session_open_request request;
    yvex_runtime_session_summary summary;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for close drain");
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "session opens before model drain");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model, "close clears the model handle while a session drains");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(session, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_BUSY,
                     "draining model refuses existing-session dispatch");
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &refused, model, &request, &failure, &err) != YVEX_OK && !refused,
                     "cleared owner cannot open a new session");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK,
                     "drained runtime session closes cleanly");
    yvex_runtime_model_close(&model);
    yvex_runtime_model_close(NULL);
    YVEX_TEST_ASSERT(!model, "post-drain and null-address model closes are idempotent");

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model reopens for deferred release retry");
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "final session reserves model before deferred release retry");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model &&
                         setenv("YVEX_TEST_RUNTIME_MODEL_CLEANUP_FAILURE", "1", 1) == 0,
                     "model close delegates one injected final release to its session");
    retained = session;
    rc = yvex_runtime_session_close(&session, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session == retained &&
                         strcmp(yvex_error_where(&err), "runtime.model.release") == 0 &&
                         yvex_runtime_session_summary_copy(
                             session, &summary, &err) == YVEX_OK &&
                         !summary.open,
                     "failed final model release retains the exact closed session owner");
    rc = yvex_runtime_session_close(&session, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session == retained &&
                         strcmp(yvex_error_where(&err), "runtime.model.release") == 0,
                     "repeated release failure retries without discharging the model twice");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_MODEL_CLEANUP_FAILURE") == 0 &&
                         yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "final retry releases the deferred model and session exactly once");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK,
                     "post-discharge session close is idempotent");
    return 0;
}

static int test_runtime_session_owner_retry(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_session_open_request request;
    yvex_runtime_model_failure failure;
    yvex_attention_failure attention_failure;
    yvex_runtime_session_summary summary;
    yvex_graph_attention_state_summary state_summary;
    const yvex_attention_state_provider *state;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for session owner retry");
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "CPU session opens before checked cleanup fault");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE", "1", 1) == 0,
                     "inject runtime session pre-release cleanup failure");
    rc = yvex_runtime_session_close(&session, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session &&
                         yvex_runtime_session_summary_copy(
                             session, &summary, &err) == YVEX_OK &&
                         !summary.open,
                     "session close retains one closing owner after pre-release failure");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE") == 0 &&
                         yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "session close retry discharges retained ownership exactly once");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_SESSION_OPEN_FAILURE", "1", 1) == 0 &&
                         setenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE", "1", 1) == 0,
                     "inject failed-open and pre-release cleanup failure");
    rc = yvex_runtime_session_open(&session, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_CLEANUP &&
                         yvex_runtime_session_summary_copy(
                             session, &summary, &err) == YVEX_OK &&
                         !summary.open,
                     "failed-open cleanup returns its retryable closing owner");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_SESSION_OPEN_FAILURE") == 0 &&
                         unsetenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE") == 0 &&
                         yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "failed-open session owner closes after the fault clears");
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "CPU session reopens for state and reservation cleanup faults");
    state = yvex_runtime_session_view_get(session)->attention_state_provider;
    YVEX_TEST_ASSERT(
        runtime_state_prepare_fixture(model, state, 0ull, &attention_failure,
                                      &err) == YVEX_OK &&
            yvex_runtime_session_begin(session, &failure, &err) == YVEX_OK &&
            runtime_state_begin_fixture(model, state, 0ull, 0ull, 1ull,
                                        &attention_failure, &err) == YVEX_OK &&
            setenv("YVEX_TEST_RUNTIME_STATE_ABORT_FAILURE", "1", 1) == 0,
        "active candidate is ready for injected rollback failure");
    rc = yvex_runtime_session_finish(session, YVEX_ERR_IO, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_STATE &&
            yvex_runtime_session_summary_copy(session, &summary, &err) == YVEX_OK &&
            state->summary(state->context, &state_summary, &err) == YVEX_OK &&
            summary.invalidated && state_summary.transaction_active &&
            state_summary.components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].entry_count == 0ull,
        "rollback failure is observable and preserves committed state");
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_STATE_ABORT_FAILURE") == 0 &&
            state->abort(state->context, &attention_failure, &err) == YVEX_OK,
        "state rollback retry clears the retained candidate");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(!model &&
                         setenv("YVEX_TEST_RUNTIME_SESSION_UNRESERVE_FAILURE", "1", 1) == 0,
                     "model drain waits on an injected reservation discharge failure");
    rc = yvex_runtime_session_close(&session, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session,
                     "reservation discharge failure retains the final model owner");
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_SESSION_UNRESERVE_FAILURE") == 0 &&
            yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
        "reservation discharge retry releases the session and deferred model exactly once");
    return 0;
}

static int test_runtime_state_factory_candidate_cleanup(
    const binding_fixture *fixture,
    const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_model_failure failure;
    injected_state_control control = {0};
    yvex_attention_state_provider_factory factory = {
        .context = &control,
        .open = injected_state_factory_open,
        .discard = injected_state_factory_discard,
    };
    yvex_runtime_session_open_request request = {
        .backend = YVEX_BACKEND_KIND_CPU,
        .attention_state_factory = &factory,
    };
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for factory candidate cleanup");
    control.fail_open_after_publish = 1;
    rc = yvex_runtime_session_open(&session, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && !session && !control.active &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_GRAPH &&
                         control.opens == 1u && control.discards == 1u,
                     "failed factory open discards its published opaque candidate");

    control.malformed_success = 1;
    rc = yvex_runtime_session_open(&session, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !session && !control.active &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_GRAPH &&
                         control.opens == 2u && control.discards == 2u,
                     "malformed factory success discards ownership without provider release");

    control.fail_open_after_publish = 1;
    control.fail_discard_once = 1;
    rc = yvex_runtime_session_open(&session, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && session && control.active &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_CLEANUP &&
                         control.opens == 3u && control.discards == 3u,
                     "discard failure returns an owned session candidate for cleanup retry");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session &&
                         !control.active && control.discards == 4u && control.releases == 0u,
                     "cleanup retry discharges factory ownership without leaking active state");
    yvex_runtime_model_close(&model);
    return 0;
}

static int test_runtime_injected_state_provider(
    const binding_fixture *fixture,
    const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_model_failure failure;
    yvex_runtime_session_summary summary;
    yvex_runtime_state_promotion_facts promotion;
    yvex_execution_capacity_plan capacity = {0};
    injected_state_control control, partial_target, partial_draft;
    yvex_attention_state_provider_factory factory, partial_factory;
    yvex_runtime_session_open_request request, partial_request;
    yvex_attention_failure state_failure;
    yvex_error err;
    int rc;

    memset(&control, 0, sizeof(control));
    factory = (yvex_attention_state_provider_factory){
        .context = &control,
        .open = injected_state_factory_open,
        .discard = injected_state_factory_discard,
    };
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.attention_state_factory = &factory;
    capacity.schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    memset(capacity.identity, 'd', YVEX_SHA256_HEX_CAP - 1u);
    capacity.identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for injected state provider");

    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK &&
                         control.active && control.opens == 1u &&
                         yvex_runtime_session_configure_persistent_pages(
                             session, &capacity, &failure, &err) == YVEX_OK &&
                         control.configures == 1u &&
                         control.active->summary.paging_configured,
                     "session configures paging through the injected provider factory");
    control.active->summary.transaction_active = 1;
    control.active->summary.staged_layer_count = 1ull;
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(session, &failure, &err) == YVEX_OK &&
                         yvex_runtime_session_select_attention_prefix(
                             session, YVEX_TENSOR_SCOPE_GLOBAL, 1ull, 0ull,
                             &promotion, &err) == YVEX_OK &&
                         promotion.schema_version ==
                             YVEX_RUNTIME_STATE_PROMOTION_FACTS_SCHEMA_V1 &&
                         promotion.available && promotion.physical.memory.complete &&
                         promotion.physical.memory.measured_operations == 1ull &&
                         !promotion.physical.memory.active_weight_bytes &&
                         !promotion.physical.memory.state_bytes &&
                         !promotion.physical.memory.activation_bytes &&
                         !promotion.physical.memory.temporary_bytes &&
                         yvex_runtime_session_finish(session, YVEX_OK, &err) == YVEX_OK &&
                         control.commits == 1u && control.active->summary.commit_count == 1ull,
                     "CPU prefix promotion reports complete zero device spans and commits");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session &&
                         control.releases == 1u && !control.active,
                     "successful injected provider closes through its owner");

    memset(&partial_target, 0, sizeof(partial_target));
    memset(&partial_draft, 0, sizeof(partial_draft));
    partial_factory = (yvex_attention_state_provider_factory){
        .context = &partial_target,
        .open = injected_state_factory_open,
        .discard = injected_state_factory_discard,
    };
    partial_request = request;
    partial_request.attention_state_factory = &partial_factory;
    YVEX_TEST_ASSERT(
        yvex_runtime_session_open(
            &session, model, &partial_request, &failure, &err) == YVEX_OK &&
            injected_state_factory_open(
                &partial_draft, NULL, 0ull,
                &session->draft_attention_state_provider, &state_failure,
                &err) == YVEX_OK,
        "session opens independently owned target and draft state providers");
    session->draft_attention_state_provider_ready = 1;
    session->view.draft_attention_state_provider =
        &session->draft_attention_state_provider;
    partial_draft.fail_configure_once = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_session_configure_persistent_pages(
            session, &capacity, &failure, &err) == YVEX_ERR_STATE &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_GRAPH &&
            partial_target.configures == 1u && partial_draft.configures == 1u &&
            partial_target.invalidations == 1u &&
            partial_draft.invalidations == 1u &&
            yvex_runtime_session_summary_copy(
                session, &summary, &err) == YVEX_OK && summary.invalidated,
        "draft paging failure invalidates a partially configured state pair");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_close(&session, &err) == YVEX_OK && !session &&
            !partial_target.active && !partial_draft.active,
        "partially configured state providers release through their owners");

    control.fail_commit_once = 1;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "injected provider reopens for commit failure");
    control.fail_configure_once = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_session_configure_persistent_pages(
            session, &capacity, &failure, &err) == YVEX_ERR_STATE &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_GRAPH &&
            control.configures == 2u && !control.active->summary.invalidated,
        "paging configuration failure returns before session-state mutation");
    control.active->summary.transaction_active = 1;
    control.active->summary.staged_layer_count = 1ull;
    rc = yvex_runtime_session_begin(session, &failure, &err);
    if (rc == YVEX_OK) rc = yvex_runtime_session_finish(session, YVEX_OK, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && control.commits == 2u &&
                         control.aborts == 1u &&
                         !control.active->summary.transaction_active &&
                         yvex_runtime_session_summary_copy(
                             session, &summary, &err) == YVEX_OK && summary.invalidated,
                     "commit failure aborts candidate state and invalidates the session");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "commit-failed injected provider releases cleanly");

    control.fail_abort_once = 1;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "injected provider reopens for abort retry");
    control.active->summary.transaction_active = 1;
    control.active->summary.staged_layer_count = 1ull;
    rc = yvex_runtime_session_begin(session, &failure, &err);
    if (rc == YVEX_OK) rc = yvex_runtime_session_finish(session, YVEX_ERR_IO, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && control.aborts == 3u &&
                         !control.active->summary.transaction_active,
                     "session retries injected provider abort before returning cleanup failure");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "abort-retried injected provider releases cleanly");

    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "injected provider reopens for summary failure abort");
    control.active->summary.transaction_active = 1;
    control.active->summary.staged_layer_count = 1ull;
    control.fail_summary_once = 1;
    rc = yvex_runtime_session_begin(session, &failure, &err);
    if (rc == YVEX_OK) rc = yvex_runtime_session_finish(session, YVEX_OK, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && control.aborts == 4u &&
                         !control.active->summary.transaction_active &&
                         yvex_runtime_session_summary_copy(
                             session, &summary, &err) == YVEX_OK && summary.invalidated,
                     "summary failure still aborts the live candidate and invalidates the session");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "summary-failed provider releases after its candidate abort");

    control.fail_release_once = 1;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "injected provider reopens for release retry");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_ERR_STATE && session &&
                         control.active && yvex_runtime_session_close(
                             &session, &err) == YVEX_OK && !session && !control.active,
                     "injected provider release failure retains and then discharges ownership");
    yvex_runtime_model_close(&model);
    return 0;
}

typedef struct {
    unsigned int calls;
    int fail_once;
} runtime_cleanup_dependent;

static int runtime_cleanup_dependent_release(void **context, yvex_error *err)
{
    runtime_cleanup_dependent *dependent =
        context ? (runtime_cleanup_dependent *)*context : NULL;

    if (!dependent) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    dependent->calls++;
    if (dependent->fail_once) {
        dependent->fail_once = 0;
        yvex_error_set(err, YVEX_ERR_STATE, "test.runtime.cleanup.dependent",
                       "injected dependent cleanup failure");
        return YVEX_ERR_STATE;
    }
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Prove one opaque lease transfers complete model/session ownership across cleanup faults. */
static int test_runtime_cleanup_lease_retry(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_cleanup_lease *lease = NULL;
    yvex_runtime_model_open_request model_request;
    yvex_runtime_session_open_request session_request;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_model_failure failure;
    runtime_cleanup_dependent dependent = {.fail_once = 1};
    runtime_cleanup_dependent occupied = {0};
    yvex_runtime_session_summary session_summary;
    yvex_error err;
    int rc;

    memset(&model_request, 0, sizeof(model_request));
    model_request.artifact_path = yvex_artifact_path(fixture->artifact);
    model_request.runtime_binding_path = prepared->path;
    model_request.target_id = runtime_fixture_execution()->target_id;
    memset(&session_request, 0, sizeof(session_request));
    session_request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(
        yvex_runtime_cleanup_lease_acquire(
            &lease, &model_request, &session_request, &model, &session,
            &failure, &err) == YVEX_OK && model && session,
        "cleanup lease acquires one model and session atomically");
    YVEX_TEST_ASSERT(yvex_runtime_cleanup_lease_close(&lease, &err) == YVEX_OK && !lease,
                     "cleanup lease releases both owners exactly once");

    YVEX_TEST_ASSERT(
        yvex_runtime_cleanup_lease_acquire(
            &lease, &model_request, NULL, &model, NULL, &failure, &err) == YVEX_OK &&
            yvex_runtime_cleanup_lease_session_open(
                lease, &session_request, &session, &failure, &err) == YVEX_OK,
        "cleanup lease supports staged session acquisition");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE", "1", 1) == 0,
                     "inject lease-owned session cleanup failure");
    rc = yvex_runtime_cleanup_lease_close(&lease, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && lease && model && session,
                     "cleanup failure retains both lease-owned handles");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE") == 0 &&
                         yvex_runtime_cleanup_lease_close(&lease, &err) == YVEX_OK && !lease,
                     "cleanup lease retry discharges both handles");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_SESSION_OPEN_FAILURE", "1", 1) == 0 &&
                         setenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE", "1", 1) == 0,
                     "inject lease acquisition and cleanup failures");
    rc = yvex_runtime_cleanup_lease_acquire(
        &lease, &model_request, &session_request, &model, &session, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && lease &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_CLEANUP &&
                         !model && !session,
                     "failed acquisition publishes its complete retryable lease");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_SESSION_OPEN_FAILURE") == 0 &&
                         unsetenv("YVEX_TEST_RUNTIME_SESSION_CLEANUP_FAILURE") == 0 &&
                         yvex_runtime_cleanup_lease_close(&lease, &err) == YVEX_OK && !lease,
                     "failed acquisition lease closes after faults clear");

    YVEX_TEST_ASSERT(
        yvex_runtime_cleanup_lease_acquire(
            &lease, &model_request, &session_request, &model, &session,
            &failure, &err) == YVEX_OK &&
            yvex_runtime_cleanup_lease_adopt(
                lease, &dependent, runtime_cleanup_dependent_release, &err) == YVEX_OK,
        "cleanup lease adopts one model-dependent owner");
    YVEX_TEST_ASSERT(
        yvex_runtime_cleanup_lease_adopt(
            lease, &occupied, runtime_cleanup_dependent_release, &err) ==
                YVEX_ERR_INVALID_ARG &&
            occupied.calls == 0u,
        "occupied cleanup slot refuses a second dependent without transfer");
    rc = yvex_runtime_cleanup_lease_close(&lease, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_STATE && lease && dependent.calls == 1u &&
            strcmp(yvex_error_where(&err), "test.runtime.cleanup.dependent") == 0 &&
            yvex_runtime_session_summary_copy(session, &session_summary, &err) == YVEX_OK &&
            session_summary.open,
        "dependent cleanup failure retains lease, session, and model for retry");
    YVEX_TEST_ASSERT(
        yvex_runtime_cleanup_lease_close(&lease, &err) == YVEX_OK && !lease &&
            dependent.calls == 2u && occupied.calls == 0u,
        "dependent cleanup retry precedes and then discharges runtime owners");
    YVEX_TEST_ASSERT(yvex_runtime_cleanup_lease_close(&lease, &err) == YVEX_OK,
                     "discharged cleanup lease close is idempotent");
    return 0;
}

static int runtime_state_store_capacity_fixture(
    const char *model_execution_identity,
    yvex_execution_capacity_plan *capacity, yvex_error *err)
{
    yvex_execution_hardware_profile hardware = {0};
    yvex_execution_workload_profile workload = {0};
    yvex_execution_capacity_plan_request request = {0};
    yvex_execution_state_class_request states[YVEX_MODEL_STATE_CLASS_COUNT] = {{0}};
    unsigned long long index;

    hardware.schema_version = YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1;
    hardware.backend = YVEX_BACKEND_KIND_CPU;
    hardware.admitted_fact_mask =
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_MEMORY) |
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_PAGING);
    hardware.total_memory_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    hardware.usable_memory_bytes = 12ull * 1024ull * 1024ull * 1024ull;
    hardware.host_page_bytes = hardware.device_page_bytes = 4096ull;
    memcpy(hardware.name, "state-store-cpu", sizeof("state-store-cpu"));
    if (yvex_execution_hardware_profile_seal(&hardware, err) != YVEX_OK)
        return yvex_error_code(err);

    workload.schema_version = YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1;
    workload.kind = YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    workload.minimum_session_context = 8ull;
    workload.requested_session_context = 128ull;
    workload.concurrent_sequences = 1ull;
    workload.logical_batch_tokens = workload.prefill_chunk_tokens = 8ull;
    workload.attention_microbatch_rows = workload.moe_row_tile = 1ull;
    workload.output_head_rows = 1ull;
    workload.system_reserve_bytes = YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE;
    workload.latency_priority = 1;
    workload.durable_state = 1;
    memcpy(workload.name, "state-store", sizeof("state-store"));
    if (yvex_execution_workload_profile_seal(&workload, err) != YVEX_OK)
        return yvex_error_code(err);

    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        states[index].state_class = (yvex_model_state_class)index;
        states[index].extent = YVEX_EXECUTION_STATE_EXTENT_CONTEXT;
        states[index].logical_block_tokens = 1ull;
        states[index].bytes_per_block = 4096ull;
        states[index].alignment_bytes = 4096ull;
        states[index].kernel_tile_tokens = 1ull;
        states[index].promotion_granularity_tokens = 1ull;
        states[index].page_table_entry_bytes = 8ull;
    }
    states[YVEX_MODEL_STATE_CANDIDATE_DELTA].extent =
        YVEX_EXECUTION_STATE_EXTENT_CANDIDATE;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].extent =
        YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].shared = 1;
    states[YVEX_MODEL_STATE_PREFIX_CHECKPOINT].copy_on_write = 1;

    request.schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    request.model_execution_identity = model_execution_identity;
    request.semantic_maximum_context = 128ull;
    request.candidate_width = 1ull;
    request.semantic_state_class_mask =
        (1ull << YVEX_MODEL_STATE_CLASS_COUNT) - 1ull;
    request.hardware = &hardware;
    request.workload = &workload;
    request.model_bytes = 1024ull * 1024ull;
    request.state_classes = states;
    request.state_class_count = YVEX_MODEL_STATE_CLASS_COUNT;
    request.workspace_bytes = 1024ull * 1024ull;
    request.scheduler_bytes = request.graph_bytes = 4096ull;
    return yvex_execution_capacity_plan_build(&request, capacity, err);
}

static int test_runtime_probe_consumer_boundary(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared,
    const char *root)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_execution_session *forked_session = NULL;
    yvex_runtime_session_prefix *prefix = NULL;
    const yvex_runtime_model_view *model_view;
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_runtime_session_open_request session_request;
    yvex_graph_attention_capacity_request capacity_request;
    yvex_attention_probe_request probe_request;
    yvex_attention_probe_result result;
    yvex_runtime_model_failure model_failure;
    yvex_attention_failure attention_failure;
    yvex_attention_failure draft_attention_failure;
    yvex_graph_attention_state_summary state_before, state_after, forked_state;
    yvex_graph_attention_state_summary draft_state;
    yvex_runtime_state_residency_summary state_residency;
    yvex_runtime_state_store_summary saved_state, restored_state;
    yvex_runtime_session_prefix_summary prefix_summary, attached_prefix;
    yvex_runtime_state_store_payload inspected_payload = {0};
    yvex_execution_capacity_plan persisted_capacity;
    yvex_runtime_session_summary session_summary;
    const yvex_attention_history_view *stable_view;
    const yvex_attention_layer_plan *state_layer;
    const yvex_attention_history_view *candidate_history = NULL;
    yvex_attention_publication publication = {0};
    float state_values[4] = {0.25f, -0.25f, 0.5f, -0.5f};
    unsigned int state_token = 7u;
    static const unsigned char semantic_payload[] =
        "session-prefix-and-rng-checkpoint";
    char state_delta[YVEX_SHA256_HEX_CAP];
    yvex_error err;
    char state_path[YVEX_PATH_CAP], corrupt_path[YVEX_PATH_CAP];
    static const size_t fault_offsets[] = {8u, 48u, 336u, 256u};
    static const unsigned int fault_modes[] = {0u, 1u, 1u, 2u};
    unsigned int fault;
    int rc;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &model_failure, &err) == YVEX_OK,
                     "runtime model opens for session-scoped probe ABI");
    memset(&session_request, 0, sizeof(session_request));
    session_request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &session_request, &model_failure, &err) == YVEX_OK,
                     "CPU session opens for session-scoped probe ABI");
    model_view = yvex_runtime_model_view_get(model);
    YVEX_TEST_ASSERT(
        model_view && model_view->binding &&
            strcmp(model_view->binding->identity, prepared->summary.identity) == 0 &&
            strcmp(model_view->binding->executable_graph_identity,
                   prepared->summary.executable_graph_identity) == 0,
        "runtime model retains its immutable binding summary after open returns");
    YVEX_TEST_ASSERT(
        runtime_state_store_capacity_fixture(
            model_view->binding->model_execution_identity,
            &persisted_capacity, &err) == YVEX_OK &&
            yvex_runtime_session_configure_persistent_pages(
                session, &persisted_capacity, &model_failure, &err) == YVEX_OK,
        "state checkpoint fixture admits canonical physical capacity before storage");
    memset(&capacity_request, 0, sizeof(capacity_request));
    capacity_request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    capacity_request.token_count = 1ull;
    capacity_request.execution_count = 1ull;
    capacity_request.select_layer = 1;
    capacity_request.layer_start = 0ull;
    YVEX_TEST_ASSERT(
        yvex_graph_attention_capacity_plan_build(
            &capacity, yvex_runtime_model_view_get(model)->attention,
            &capacity_request, &err) == YVEX_OK,
        "one-layer capacity plan seals for the runtime consumer");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_prepare_attention_probe_state(
            session, NULL, capacity, &attention_failure, &err) == YVEX_ERR_STATE,
        "state preparation refuses a session/model mismatch");
    rc = yvex_runtime_session_prepare_attention_probe_state(
        session, model, capacity, &attention_failure, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "state checkpoint fixture preparation failed: %s\n",
                yvex_error_message(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "matching runtime owners prepare canonical probe state");
    stable_view = yvex_runtime_session_view_get(session)->attention_state_provider->view(
        yvex_runtime_session_view_get(session)->attention_state_provider->context, 0ull,
        YVEX_ATTENTION_STATE_VIEW_COMMITTED);
    YVEX_TEST_ASSERT(stable_view, "persistent state exposes its initial committed view");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(session)->attention_state_provider->summary(
            yvex_runtime_session_view_get(session)->attention_state_provider->context,
            &state_before, &err) == YVEX_OK &&
            yvex_runtime_state_residency_summary_copy(
                yvex_runtime_session_view_get(session)->state_residency,
                &state_residency, &err) == YVEX_OK &&
            yvex_runtime_session_summary_copy(session, &session_summary, &err) == YVEX_OK,
        "probe preparation seals readable persistent provider, residency, and session summaries");
    YVEX_TEST_ASSERT(
        state_before.persistent && state_before.position_consistent &&
            state_before.prepared_layer_count == 1ull &&
            state_residency.sealed && state_residency.layer_count == 1ull &&
            !state_residency.cuda_ready && !state_residency.host_bytes &&
            !state_residency.device_bytes && session_summary.capabilities.persistent_kv_ready,
        "CPU session admits persistent state only after complete plan and residency coverage");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_prepare_persistent_state(
            session, capacity, &model_failure, &err) == YVEX_ERR_STATE &&
            yvex_runtime_session_reset_persistent_state(
                session, &model_failure, &err) == YVEX_OK &&
            yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                yvex_runtime_session_view_get(session)->attention_state_provider->context,
                &state_after, &err) == YVEX_OK &&
            state_after.generation == state_before.generation + 1ull &&
            state_after.reset_count == state_before.reset_count + 1ull &&
            stable_view == yvex_runtime_session_view_get(session)->attention_state_provider->view(
                yvex_runtime_session_view_get(session)->attention_state_provider->context, 0ull,
                YVEX_ATTENTION_STATE_VIEW_COMMITTED),
        "persistent state refuses duplicate seal and clears through allocation-stable reuse");
    memset(&probe_request, 0, sizeof(probe_request));
    memset(&result, 0, sizeof(result));
    probe_request.backend = YVEX_BACKEND_KIND_CPU;
    probe_request.logical_model_identity = "caller-owned-runtime-identity";
    probe_request.probe = YVEX_ATTENTION_PROBE_CANONICAL_V2;
    probe_request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    probe_request.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
    probe_request.token_count = 1ull;
    probe_request.select_layer = 1;
    probe_request.layer_ordinal = 0ull;
    rc = yvex_runtime_attention_probe_execute(
        session, NULL, &probe_request, &result, &model_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !result.layers_executed,
                     "session-scoped execution refuses a session/model mismatch");
    rc = yvex_runtime_attention_probe_execute(
        session, model, &probe_request, &result, &model_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !result.layers_executed &&
                         strcmp(yvex_error_where(&err), "runtime.attention.execute") == 0,
                     "session-scoped execution refuses caller-injected runtime resources");
    probe_request.logical_model_identity = NULL;
    probe_request.workspace =
        yvex_runtime_session_view_get(session)->attention_workspace;
    rc = yvex_runtime_attention_probe_execute(
        session, model, &probe_request, &result, &model_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && !result.layers_executed &&
                         strcmp(yvex_error_where(&err), "runtime.attention.execute") == 0,
                     "session-scoped execution refuses a caller-injected workspace");
    state_layer = yvex_attention_plan_layer_at(model_view->attention, 0ull);
    YVEX_TEST_ASSERT(state_layer, "state checkpoint fixture layer is available");
    publication.owned = publication.complete = publication.prefix_addressable = 1;
    (void)snprintf(publication.execution_identity,
                   sizeof(publication.execution_identity), "%064x", 91);
    publication.layer_index = 0ull;
    publication.attention_class = state_layer->attention_class;
    publication.token_position = 0ull;
    publication.token_count = 1ull;
    publication.token_ids = &state_token;
    publication.kv_width = state_layer->head_dimension;
    publication.raw_kv = state_values;
    YVEX_TEST_ASSERT(
        yvex_runtime_session_begin(session, &model_failure, &err) == YVEX_OK,
        "state checkpoint fixture owns one session transaction");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(session)->attention_state_provider->begin(
            yvex_runtime_session_view_get(session)->attention_state_provider->context,
            0ull, state_layer, NULL, 0ull, 1ull, NULL,
            &candidate_history, &attention_failure, &err) == YVEX_OK &&
            candidate_history,
        "state checkpoint fixture begins one candidate prefix");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(session)->attention_state_provider->stage(
            yvex_runtime_session_view_get(session)->attention_state_provider->context,
            &publication, NULL, state_delta, &attention_failure, &err) == YVEX_OK,
        "state checkpoint fixture stages one candidate prefix");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(session)->attention_state_provider->commit(
            yvex_runtime_session_view_get(session)->attention_state_provider->context,
            &attention_failure, &err) == YVEX_OK &&
            yvex_runtime_session_finish(session, YVEX_ERR_IO, &err) == YVEX_ERR_IO &&
            yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                yvex_runtime_session_view_get(session)->attention_state_provider->context,
                &state_before, &err) == YVEX_OK &&
            state_before.committed_sequence_length == 1ull,
        "successful state transaction commits one prefix before persistence");
    memset(&draft_attention_failure, 0, sizeof(draft_attention_failure));
    YVEX_TEST_ASSERT(
        yvex_attention_state_provider_open_persistent(
            model_view->attention, 1ull << 20u,
            &session->draft_attention_state_provider,
            &draft_attention_failure, &err) == YVEX_OK,
        "runtime fixture opens a pristine optional draft scope");
    session->draft_attention_state_provider_ready = 1;
    session->view.draft_attention_state_provider =
        &session->draft_attention_state_provider;
    YVEX_TEST_ASSERT(
        session->draft_attention_state_provider.summary(
            session->draft_attention_state_provider.context,
            &draft_state, &err) == YVEX_OK &&
            !draft_state.prepared_layer_count &&
            !draft_state.committed_sequence_length,
        "optional draft scope remains physically pristine");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_prefix_capture(
            session, 1ull, &prefix, &prefix_summary, &model_failure, &err) ==
                YVEX_ERR_BOUNDS &&
            !prefix,
        "runtime prefix capture refuses an insufficient explicit byte budget");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_prefix_capture(
            session, 1ull << 20u, &prefix, &prefix_summary,
            &model_failure, &err) == YVEX_OK &&
            prefix && prefix_summary.schema_version ==
                          YVEX_RUNTIME_SESSION_PREFIX_SCHEMA_V1 &&
            prefix_summary.scope_count == 1ull &&
            prefix_summary.committed_sequence_length == 1ull &&
            prefix_summary.shared_bytes &&
            yvex_sha256_hex_valid(prefix_summary.prefix_identity),
        "runtime captures one identity-bound committed target prefix");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_open(
            &forked_session, model, &session_request, &model_failure, &err) ==
            YVEX_OK,
        "empty runtime session opens before prefix attachment");
    memset(&draft_attention_failure, 0, sizeof(draft_attention_failure));
    YVEX_TEST_ASSERT(
        yvex_attention_state_provider_open_persistent(
            model_view->attention, 1ull << 20u,
            &forked_session->draft_attention_state_provider,
            &draft_attention_failure, &err) == YVEX_OK,
        "fork destination opens the same pristine optional draft scope");
    forked_session->draft_attention_state_provider_ready = 1;
    forked_session->view.draft_attention_state_provider =
        &forked_session->draft_attention_state_provider;
    rc = yvex_runtime_session_prefix_attach(
        forked_session, prefix, &attached_prefix, &model_failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            strcmp(attached_prefix.prefix_identity,
                   prefix_summary.prefix_identity) == 0,
        "empty runtime session attaches the identity-bound prefix");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(forked_session)
                    ->attention_state_provider->summary(
                        yvex_runtime_session_view_get(forked_session)
                            ->attention_state_provider->context,
                        &forked_state, &err) == YVEX_OK &&
            forked_state.committed_sequence_length == 1ull &&
            strcmp(forked_state.state_content_identity,
                   state_before.state_content_identity) == 0 &&
            forked_session->draft_attention_state_provider.summary(
                forked_session->draft_attention_state_provider.context,
                &draft_state, &err) == YVEX_OK &&
            !draft_state.prepared_layer_count &&
            !draft_state.committed_sequence_length,
        "attached target prefix preserves content and leaves optional draft pristine");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_prepare_persistent_state(
            forked_session, capacity, &model_failure, &err) == YVEX_OK,
        "forked prefix seals its independent runtime residency before execution");
    yvex_runtime_session_prefix_close(&prefix);
    YVEX_TEST_ASSERT(
        yvex_runtime_session_reset_persistent_state(
            forked_session, &model_failure, &err) == YVEX_OK &&
            yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                yvex_runtime_session_view_get(session)
                    ->attention_state_provider->context,
                &state_after, &err) == YVEX_OK &&
            state_after.committed_sequence_length == 1ull &&
            strcmp(state_after.state_content_identity,
                   state_before.state_content_identity) == 0 &&
            yvex_runtime_session_close(&forked_session, &err) == YVEX_OK &&
            !forked_session,
        "forked state survives prefix-handle close and resets without mutating source");
    YVEX_TEST_ASSERT(
        snprintf(state_path, sizeof(state_path), "%s/state.yvex", root) <
                (int)sizeof(state_path) &&
            snprintf(corrupt_path, sizeof(corrupt_path), "%s/state-corrupt.yvex", root) <
                (int)sizeof(corrupt_path),
        "state checkpoint fixture paths are bounded");
    rc = yvex_runtime_session_state_save(
        session, state_path, semantic_payload, sizeof(semantic_payload),
        &saved_state, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "state checkpoint save failed: %s\n",
                yvex_error_message(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "state checkpoint publishes one immutable file");
    YVEX_TEST_ASSERT(
        saved_state.schema_version == YVEX_RUNTIME_STATE_STORE_SCHEMA_V2 &&
            saved_state.scope_count == 1ull &&
            saved_state.committed_sequence_length == 1ull &&
            saved_state.payload_bytes == sizeof(semantic_payload) &&
            yvex_sha256_hex_valid(saved_state.payload_identity) &&
            yvex_sha256_hex_valid(saved_state.file_digest),
        "state checkpoint reports exact model-bound evidence");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_state_save(
            session, state_path, NULL, 0ull, &restored_state,
            &err) == YVEX_ERR_STATE,
        "state checkpoint refuses replacement publication");
    YVEX_TEST_ASSERT(
            yvex_runtime_session_state_inspect(
                session, state_path, saved_state.file_bytes,
                &inspected_payload, &restored_state, &err) == YVEX_OK &&
            inspected_payload.byte_count == sizeof(semantic_payload) &&
            memcmp(inspected_payload.bytes, semantic_payload,
                   sizeof(semantic_payload)) == 0 &&
            strcmp(inspected_payload.payload_identity,
                   saved_state.payload_identity) == 0 &&
            yvex_runtime_session_reset_persistent_state(
            session, &model_failure, &err) == YVEX_OK &&
            yvex_runtime_session_state_restore(
                session, state_path, saved_state.file_bytes,
                saved_state.committed_sequence_length,
                saved_state.file_digest, saved_state.payload_identity,
                &restored_state,
                &err) == YVEX_OK &&
            restored_state.committed_sequence_length == 1ull &&
            strcmp(restored_state.file_digest, saved_state.file_digest) == 0 &&
            yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                yvex_runtime_session_view_get(session)->attention_state_provider->context,
                &state_after, &err) == YVEX_OK &&
            state_after.committed_sequence_length == 1ull &&
            strcmp(state_after.state_content_identity,
                   state_before.state_content_identity) == 0,
        "state checkpoint restores the exact committed prefix after reset");
    yvex_runtime_state_store_payload_close(&inspected_payload);
    for (fault = 0u; fault < 4u; ++fault) {
        YVEX_TEST_ASSERT(
            copy_regular_file(state_path, corrupt_path) &&
                rewrite_state_file_fixture(
                    corrupt_path, fault_offsets[fault], fault_modes[fault], 99ull) &&
                yvex_runtime_session_state_restore(
                    session, corrupt_path, saved_state.file_bytes,
                    saved_state.committed_sequence_length, NULL, NULL,
                    &restored_state, &err) == YVEX_ERR_FORMAT &&
                yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                    yvex_runtime_session_view_get(session)->attention_state_provider->context,
                    &state_after, &err) == YVEX_OK &&
                state_after.committed_sequence_length == 1ull &&
                strcmp(state_after.state_content_identity,
                       state_before.state_content_identity) == 0 &&
                unlink(corrupt_path) == 0,
            "state checkpoint mismatch refuses without partial publication");
    }
    yvex_graph_attention_capacity_plan_close(&capacity);
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "probe consumer session closes without staged state");
    yvex_runtime_model_close(&model);
    YVEX_TEST_ASSERT(unlink(state_path) == 0,
                     "state checkpoint fixtures clean up");
    return 0;
}

static void runtime_paged_capacity_fixture(yvex_execution_capacity_plan *capacity)
{
    unsigned long long index;

    memset(capacity, 0, sizeof(*capacity));
    capacity->schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    capacity->per_session_maximum = 128ull;
    capacity->state_pool_bytes = 1ull << 30u;
    capacity->candidate_reserve_bytes = 1ull << 30u;
    capacity->state_class_count = YVEX_MODEL_STATE_CLASS_COUNT;
    memset(capacity->model_execution_identity, 'a', YVEX_SHA256_HEX_CAP - 1u);
    memset(capacity->hardware_profile_identity, 'b', YVEX_SHA256_HEX_CAP - 1u);
    memset(capacity->workload_profile_identity, 'c', YVEX_SHA256_HEX_CAP - 1u);
    memset(capacity->identity, 'd', YVEX_SHA256_HEX_CAP - 1u);
    for (index = 0ull; index < capacity->state_class_count; ++index) {
        yvex_execution_state_class_plan *state = &capacity->state_classes[index];

        state->state_class = (yvex_model_state_class)index;
        state->extent = YVEX_EXECUTION_STATE_EXTENT_CONTEXT;
        state->logical_block_tokens = 1ull;
        state->bytes_per_block = 8ull;
        state->page_tokens = 128ull;
        state->page_bytes = 1024ull;
    }
}

/* CUDA staging must not read virtual history pages before their provider admission. */
static int test_runtime_paged_state_cuda_pack(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_state_residency *residency = NULL;
    yvex_backend *backend = NULL;
    const yvex_attention_state_provider *provider;
    const yvex_graph_attention_capacity_layer *layer;
    yvex_graph_attention_capacity_plan *attention_capacity = NULL;
    yvex_graph_attention_capacity_request attention_request = {0};
    yvex_execution_capacity_plan page_capacity;
    yvex_runtime_state_residency_summary summary;
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_model_failure model_failure;
    yvex_attention_failure attention_failure;
    yvex_backend_options backend_options = {0};
    yvex_error err;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &model_failure, &err) == YVEX_OK,
                     "runtime model opens for paged CUDA residency packing");
    session_request.backend = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &session_request, &model_failure, &err) == YVEX_OK,
                     "runtime session opens for paged CUDA residency packing");
    provider = yvex_runtime_session_view_get(session)->attention_state_provider;
    runtime_paged_capacity_fixture(&page_capacity);
    YVEX_TEST_ASSERT(yvex_runtime_session_configure_persistent_pages(
                         session, &page_capacity, &model_failure, &err) == YVEX_OK,
                     "runtime provider admits class-specific virtual pages");
    attention_request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    attention_request.token_count = 1ull;
    attention_request.execution_count = 1ull;
    attention_request.select_layer = 1;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &attention_capacity, yvex_runtime_model_view_get(model)->attention,
                         &attention_request, &err) == YVEX_OK,
                     "one-layer paged state capacity seals");
    layer = yvex_graph_attention_capacity_plan_layer(attention_capacity, 0ull);
    YVEX_TEST_ASSERT(layer && layer->selected &&
                         provider->prepare(provider->context, 0ull, &layer->recipe,
                                           NULL, &attention_failure, &err) == YVEX_OK,
                     "empty committed and candidate histories reserve inaccessible tails");
    backend_options.kind = YVEX_BACKEND_KIND_CUDA;
    if (yvex_backend_open(&backend, &backend_options, &err) == YVEX_OK) {
        unsigned long long committed_pages;
        YVEX_TEST_ASSERT(yvex_runtime_state_residency_prepare(
                             &residency, backend, attention_capacity, provider,
                             0ull, ULLONG_MAX, 0ull, ULLONG_MAX, &err) == YVEX_OK &&
                             yvex_runtime_state_residency_summary_copy(
                                 residency, &summary, &err) == YVEX_OK,
                         "native CUDA state residency prepares from provider pages");
        if (yvex_backend_virtual_tensor_supported(backend)) {
            YVEX_TEST_ASSERT(summary.paged && !summary.host_bytes,
                             "native CUDA state uses page-backed device storage");
            YVEX_TEST_ASSERT(summary.virtual_device_bytes && summary.page_granularity,
                             "native CUDA state reports virtual geometry");
            YVEX_TEST_ASSERT(!summary.device_bytes && !summary.page_commit_count,
                             "empty CUDA state reserves no physical pages");
            YVEX_TEST_ASSERT(yvex_runtime_state_residency_transition(
                                 residency, provider, NULL, 0ull, 1ull,
                                 YVEX_RUNTIME_STATE_BEGIN, &err) == YVEX_OK &&
                                 yvex_runtime_state_residency_summary_copy(
                                     residency, &summary, &err) == YVEX_OK,
                             "candidate growth admits its CUDA pages before execution");
            YVEX_TEST_ASSERT(summary.page_commit_count,
                             "native CUDA state commits admitted physical pages");
            YVEX_TEST_ASSERT(summary.device_bytes ==
                                 summary.page_commit_count * summary.page_granularity,
                             "native CUDA state accounts every physical granule");
            committed_pages = summary.page_commit_count;
            YVEX_TEST_ASSERT(yvex_runtime_state_residency_reset(residency, &err) == YVEX_OK &&
                                 yvex_runtime_state_residency_summary_copy(
                                     residency, &summary, &err) == YVEX_OK &&
                                 !summary.device_bytes &&
                                 summary.page_release_count == committed_pages,
                             "state reset releases physical pages but retains virtual geometry");
        } else {
            YVEX_TEST_ASSERT(!summary.paged && summary.host_bytes == summary.device_bytes,
                             "CUDA without VMM exposes its explicit full-bank fallback");
        }
        YVEX_TEST_ASSERT(yvex_runtime_state_residency_close(&residency, &err) == YVEX_OK &&
                             yvex_backend_close_checked(&backend, &err) == YVEX_OK,
                         "native CUDA state residency closes exactly");
    } else {
        YVEX_TEST_ASSERT(yvex_error_code(&err) == YVEX_ERR_UNSUPPORTED,
                         "unavailable CUDA refuses paged runtime state explicitly");
    }
    yvex_graph_attention_capacity_plan_close(&attention_capacity);
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK,
                     "paged state test session closes");
    yvex_runtime_model_close(&model);
    return 0;
}

static int runtime_cuda_test_ready(int *ready)
{
    yvex_backend *backend = NULL;
    yvex_backend_options options;
    yvex_backend_capability_result capability;
    yvex_error err;
    int rc;

    *ready = 0;
    if (!yvex_backend_cuda_available()) return 0;
    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&backend, &options, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_query_capability(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &capability, &err);
    if (rc == YVEX_OK &&
        capability.reason == YVEX_BACKEND_CAPABILITY_REASON_KERNEL_BUNDLE_ABSENT) {
        fprintf(stderr, "SKIP: generated CUDA bundle is absent\n");
    } else if (rc == YVEX_OK && capability.state == YVEX_BACKEND_CAPABILITY_SUPPORTED) {
        *ready = 1;
    } else {
        rc = rc == YVEX_OK ? YVEX_ERR_BACKEND : rc;
    }
    if (yvex_backend_close_checked(&backend, &err) != YVEX_OK) return YVEX_ERR_STATE;
    return rc;
}

static int test_runtime_model_cuda_residency_claim(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model_open_request request = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_residency_summary summary = {0};
    yvex_runtime_model *model = NULL;
    const yvex_runtime_model_view *view;
    yvex_error err;
    int ready, rc = runtime_cuda_test_ready(&ready);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA model residency test probes the native bundle");
    if (!ready) {
        fprintf(stderr, "SKIP: CUDA unavailable for runtime model residency\n");
        return 0;
    }
    request.artifact_path = yvex_artifact_path(fixture->artifact);
    request.runtime_binding_path = prepared->path;
    request.target_id = runtime_fixture_execution()->target_id;
    request.residency_backend = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_HOST_REGISTER_FAILURE", "1", 1) == 0,
                     "install artifact registration failure injection");
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_HOST_REGISTER_FAILURE") == 0,
                     "clear artifact registration failure injection");
    if (rc != YVEX_ERR_BACKEND || model ||
        strcmp(yvex_error_where(&err), "cuda.residency.register") != 0 ||
        strcmp(failure.field, "model-residency") != 0)
        fprintf(stderr,
                "artifact prefetch refusal rc=%d model=%p where=%s message=%s failure=%u field=%s\n",
                rc, (void *)model, yvex_error_where(&err), yvex_error_message(&err),
                (unsigned int)failure.code, failure.field);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_BACKEND && !model &&
            strcmp(yvex_error_where(&err), "cuda.residency.register") == 0 &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_MATERIALIZATION &&
            strcmp(failure.field, "model-residency") == 0,
        "artifact-backed CUDA placement refuses when registration cannot complete");
    rc = yvex_runtime_model_open(&model, &request, &failure, &err);
    if (rc != YVEX_OK)
        fprintf(stderr, "CUDA runtime residency open failed: %s (%s)\n",
                yvex_error_message(&err), yvex_error_where(&err));
    YVEX_TEST_ASSERT(rc == YVEX_OK && model,
                     "runtime model claims its admitted CUDA residency owner");
    view = yvex_runtime_model_view_get(model);
    YVEX_TEST_ASSERT(view && view->residency &&
                         yvex_runtime_residency_snapshot(
                             view->residency, &summary, NULL, NULL, &err) == YVEX_OK &&
                         summary.cuda_ready &&
                         summary.placement ==
                               YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED &&
                           !summary.host_resident_bytes && !summary.device_resident_bytes &&
                           summary.artifact_backed_bytes == summary.cuda_addressable_bytes &&
                           summary.cuda_pageable_map_count == 1ull &&
                           summary.cuda_pageable_map_bytes == summary.artifact_backed_bytes &&
                           summary.cuda_host_registration_count == 1ull &&
                           !summary.cuda_pageable_prefetch_count &&
                           !summary.cuda_pageable_prefetch_bytes &&
                           !summary.cuda_managed_allocation_count &&
                           !summary.cuda_managed_bytes,
                     "runtime model publishes exact registered artifact-backed CUDA facts");
    yvex_runtime_model_close(&model);
    return 0;
}

/* Prove one CUDA resident pack is shared by isolated session contexts until owner release. */
static int test_runtime_cuda_session_cleanup_retry(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_backend *owner = NULL, *first = NULL, *second = NULL;
    yvex_device_tensor *weights = NULL, *session_weights = NULL;
    yvex_backend_options options;
    yvex_backend_tensor_desc descriptor;
    yvex_backend_memory_stats stats;
    unsigned char *host = NULL;
    unsigned long long first_address = 0ull, second_address = 0ull;
    yvex_error err;
    int ready, rc;

    (void)fixture;
    (void)prepared;
    rc = runtime_cuda_test_ready(&ready);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "CUDA bundle refusal remains fail-closed");
    if (!ready) {
        fprintf(stderr, "SKIP: CUDA unavailable for shared runtime residency\n");
        return 0;
    }
    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.name = "shared-runtime-residency";
    descriptor.dtype = YVEX_DTYPE_I8;
    descriptor.rank = 1u;
    descriptor.dims[0] = descriptor.bytes = 32ull;
    YVEX_TEST_ASSERT(
        yvex_backend_open(&owner, &options, &err) == YVEX_OK &&
            yvex_backend_resident_alloc(
                owner, &descriptor, &weights, &host, &err) == YVEX_OK &&
            yvex_backend_resident_attach(
                owner, host, descriptor.bytes, weights, 1ull, &err) == YVEX_OK &&
            yvex_backend_open_shared_cuda(&first, owner, 0ull, &err) == YVEX_OK &&
            yvex_backend_open_shared_cuda(&second, owner, 0ull, &err) == YVEX_OK,
        "one CUDA owner opens two isolated shared-context sessions");
    descriptor.name = "session-owned-residency";
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(first, &descriptor, &session_weights, &err) == YVEX_OK &&
            yvex_backend_resident_attach(
                second, host, descriptor.bytes, session_weights, 1ull, &err) ==
                YVEX_ERR_INVALID_ARG &&
            yvex_backend_tensor_release(first, &session_weights, &err) == YVEX_OK,
        "one session cannot lend its device tensor to another session");
    descriptor.name = "shared-runtime-residency";
    YVEX_TEST_ASSERT(
        yvex_backend_resident_attach(first, host, descriptor.bytes, weights, 1ull, &err) == YVEX_OK &&
            yvex_backend_resident_attach(second, host, descriptor.bytes, weights, 1ull, &err) == YVEX_OK &&
            yvex_backend_resident_resolve(first, host, descriptor.bytes, &first_address) ==
                YVEX_BACKEND_RESIDENT_HIT &&
            yvex_backend_resident_resolve(second, host, descriptor.bytes, &second_address) ==
                YVEX_BACKEND_RESIDENT_HIT &&
            first_address == second_address && first_address != 0ull && first != second,
        "both sessions resolve the same stable read-only device range through isolated state");
    rc = yvex_backend_tensor_release(owner, &weights, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && weights &&
                         yvex_backend_resident_resolve(
                             first, host, descriptor.bytes, &first_address) ==
                             YVEX_BACKEND_RESIDENT_HIT &&
                         yvex_backend_resident_resolve(
                             second, host, descriptor.bytes, &second_address) ==
                             YVEX_BACKEND_RESIDENT_HIT &&
                         first_address == second_address,
                     "model owner cannot release resident bytes while sessions borrow them");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(first, &err) == YVEX_OK,
                     "first borrowed resident mapping detaches");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&first, &err) == YVEX_OK && !first &&
                         yvex_backend_resident_resolve(
                             second, host, descriptor.bytes, &second_address) ==
                             YVEX_BACKEND_RESIDENT_HIT && second_address == first_address,
                     "first session close leaves the owner pack valid for the second");
    YVEX_TEST_ASSERT(yvex_backend_get_memory_stats(owner, &stats, &err) == YVEX_OK &&
                         stats.allocation_count == 1ull,
                     "model owner uploads one resident allocation for both sessions");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(second, &err) == YVEX_OK,
                     "second borrowed resident mapping detaches");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&second, &err) == YVEX_OK && !second,
                     "second isolated session closes without releasing owner bytes");
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(owner, &err) == YVEX_OK,
                     "model owner detaches after all session mappings");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "tensor-alloc", 1) == 0,
                     "inject model-owned CUDA residency release failure");
    rc = yvex_backend_tensor_release(owner, &weights, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && weights,
                     "failed model release retains its exact resident tensor owner");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear model-owned CUDA release failure");
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(owner, &weights, &err) == YVEX_OK && !weights &&
                         yvex_backend_close_checked(&owner, &err) == YVEX_OK && !owner,
                     "model-owned pack and context release exactly once after both sessions");
    return 0;
}

static int test_runtime_cuda_workspace_transaction(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_runtime_session_open_request request;
    yvex_graph_attention_capacity_request capacity_request;
    yvex_runtime_model_failure failure;
    yvex_runtime_session_summary before, after;
    runtime_execute_thread execution;
    runtime_thread_gate gate;
    pthread_t thread;
    yvex_error err;
    unsigned long long row_one_bytes, row_two_bytes;
    int ready, rc;

    rc = runtime_cuda_test_ready(&ready);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "workspace CUDA bundle refusal remains fail-closed");
    if (!ready) return 0;
    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens for transactional CUDA workspace");
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_runtime_session_open(&session, model, &request, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "admitted CUDA runtime session opens for workspace transaction");
    memset(&capacity_request, 0, sizeof(capacity_request));
    capacity_request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    capacity_request.token_count = 1ull;
    capacity_request.execution_count = 1ull;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &capacity, yvex_runtime_model_view_get(model)->attention,
                         &capacity_request, &err) == YVEX_OK,
                     "transactional workspace capacity plan seals");
    {
        const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
        const yvex_runtime_binding *binding = view->compiled_binding;
        YVEX_TEST_ASSERT(
            yvex_runtime_private_attention_workspace_required(
                &binding->attention, binding->layers,
                binding->summary.layer_count, capacity,
                YVEX_ATTENTION_EXECUTION_EAGER,
                YVEX_ATTENTION_OPERATION_ENVELOPE,
                YVEX_ATTENTION_EVIDENCE_NONE, 1ull, 1, &row_one_bytes,
                &err) == YVEX_OK &&
                yvex_runtime_private_attention_workspace_required(
                    &binding->attention, binding->layers,
                    binding->summary.layer_count, capacity,
                    YVEX_ATTENTION_EXECUTION_EAGER,
                    YVEX_ATTENTION_OPERATION_ENVELOPE,
                    YVEX_ATTENTION_EVIDENCE_NONE, 2ull, 1,
                    &row_two_bytes, &err) == YVEX_OK &&
                row_two_bytes > row_one_bytes &&
                yvex_runtime_private_attention_workspace_required(
                    &binding->attention, binding->layers,
                    binding->summary.layer_count, capacity,
                    YVEX_ATTENTION_EXECUTION_EAGER,
                    YVEX_ATTENTION_OPERATION_ENVELOPE,
                    YVEX_ATTENTION_EVIDENCE_NONE, 0ull, 1,
                    &row_two_bytes, &err) == YVEX_ERR_INVALID_ARG,
            "physical row capacity scales staging independently of state capacity");
    }
    YVEX_TEST_ASSERT(yvex_runtime_session_summary_copy(session, &before, &err) == YVEX_OK,
                     "capture session summary before workspace transaction");
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, (yvex_runtime_execution_mode)-1,
        YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE, YVEX_ATTENTION_EVIDENCE_NONE,
        capacity, capacity_request.token_count, 0ull, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT &&
                         yvex_runtime_session_summary_copy(session, &after, &err) == YVEX_OK &&
                         memcmp(&before, &after, sizeof(before)) == 0,
                     "negative workspace mode refuses before session mutation");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_RUNTIME_WORKSPACE_FAILURE", "capability", 1) == 0 &&
                         setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE",
                                "host-workspace-pre-release", 1) == 0,
                     "inject workspace publication and pre-release cleanup failures");
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, YVEX_RUNTIME_MODE_EAGER, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
        YVEX_ATTENTION_EVIDENCE_NONE, capacity, capacity_request.token_count,
        0ull, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_CLEANUP &&
                         yvex_runtime_session_summary_copy(session, &after, &err) == YVEX_OK &&
                         memcmp(&before, &after, sizeof(before)) == 0,
                     "workspace cleanup failure preserves the complete prior summary");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear workspace pre-release cleanup failure");
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, YVEX_RUNTIME_MODE_EAGER, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
        YVEX_ATTENTION_EVIDENCE_NONE, capacity, capacity_request.token_count,
        0ull, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_BACKEND &&
                         yvex_runtime_session_summary_copy(session, &after, &err) == YVEX_OK &&
                         memcmp(&before, &after, sizeof(before)) == 0,
                     "workspace retry drains prior ownership and rolls back publication atomically");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_RUNTIME_WORKSPACE_FAILURE") == 0,
                     "clear workspace publication failure");
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, YVEX_RUNTIME_MODE_EAGER, YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
        YVEX_ATTENTION_EVIDENCE_NONE, capacity, capacity_request.token_count,
        0ull, &failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_runtime_session_summary_copy(session, &after, &err) == YVEX_OK &&
            after.capabilities.attention_workspace_ready == 1 &&
            after.capabilities.cuda_prefill_eager_ready == 1 &&
            after.capabilities.cuda_decode_eager_ready == 1 &&
            after.host_workspace_owned && after.host_workspace_pinned &&
            after.host_workspace_bytes > 0ull && after.device_workspace_bytes > 0ull,
        "real resident binding admits CUDA workspace after complete rollback");
    before = after;
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, YVEX_RUNTIME_MODE_EAGER,
        YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
        YVEX_ATTENTION_EVIDENCE_NONE, capacity, capacity_request.token_count,
        0ull, &failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            yvex_runtime_session_summary_copy(session, &after, &err) == YVEX_OK &&
            after.workspace_generation == before.workspace_generation &&
            after.workspace_allocation_count == before.workspace_allocation_count &&
            after.host_workspace_bytes == before.host_workspace_bytes &&
            after.device_workspace_bytes == before.device_workspace_bytes &&
            strcmp(after.workspace_identity, before.workspace_identity) == 0,
        "an exact second capacity owner adopts the sealed CUDA workspace without allocation");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_begin(session, &failure, &err) == YVEX_OK &&
            yvex_runtime_session_prepare_attention_probe_state(
                session, model, capacity, NULL, &err) == YVEX_OK &&
            yvex_runtime_session_prepare_attention_workspace(
                session, YVEX_RUNTIME_MODE_EAGER,
                YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
                YVEX_ATTENTION_EVIDENCE_NONE, capacity,
                capacity_request.token_count, 0ull, &failure, &err) ==
                YVEX_OK &&
            yvex_runtime_session_finish(session, YVEX_OK, &err) == YVEX_OK,
        "the transaction owner may prepare persistent state and shared workspace");
    YVEX_TEST_ASSERT(runtime_thread_gate_init(&gate),
                     "foreign workspace-owner gate initializes");
    memset(&execution, 0, sizeof(execution));
    execution.session = session;
    execution.gate = &gate;
    YVEX_TEST_ASSERT(pthread_create(&thread, NULL, runtime_execute_thread_main,
                                    &execution) == 0,
                     "foreign execution owner starts");
    runtime_thread_gate_wait_ready(&gate, 1u);
    rc = yvex_runtime_session_prepare_attention_workspace(
        session, YVEX_RUNTIME_MODE_EAGER,
        YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE,
        YVEX_ATTENTION_EVIDENCE_NONE, capacity, capacity_request.token_count,
        0ull, &failure, &err);
    YVEX_TEST_ASSERT(
        execution.begin_status == YVEX_OK && rc == YVEX_ERR_STATE &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_BUSY,
        "a foreign thread cannot mutate an owned session workspace");
    runtime_thread_gate_release(&gate);
    YVEX_TEST_ASSERT(pthread_join(thread, NULL) == 0 &&
                         execution.begin_status == YVEX_OK,
                     "foreign execution owner releases the session");
    runtime_thread_gate_destroy(&gate);
    yvex_graph_attention_capacity_plan_close(&capacity);
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK && !session,
                     "transactional workspace session closes cleanly");
    yvex_runtime_model_close(&model);
    return 0;
}

static int test_runtime_model_compiled_execution(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model_open_request request;
    yvex_runtime_model *model = NULL;
    yvex_runtime_model_failure failure;
    char target[] = "compiled-fixture";
    yvex_error err;

    memset(&request, 0, sizeof(request));
    request.artifact_path = yvex_artifact_path(fixture->artifact);
    request.runtime_binding_path = prepared->path;
    request.target_id = "";
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) != YVEX_OK && !model &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_INVALID_ARGUMENT,
                     "runtime model refuses a missing presentation target before artifact work");
    request.target_id = target;
    YVEX_TEST_ASSERT(yvex_runtime_model_open(
                         &model, &request, &failure, &err) == YVEX_OK && model,
                     "runtime model instantiates compiled execution without family lookup");
    memset(target, 'x', sizeof(target) - 1u);
    target[sizeof(target) - 1u] = '\0';
    YVEX_TEST_ASSERT(
        yvex_runtime_model_view_get(model)->graph == &yvex_attention_execution_api &&
            strcmp(yvex_runtime_model_view_get(model)->target_id,
                   "compiled-fixture") == 0,
        "sealed runtime model owns its presentation target and generic execution API");
    yvex_runtime_model_close(&model);
    return 0;
}

static int test_runtime_model_snapshot_drift(
    const binding_fixture *fixture, const yvex_runtime_binding_prepare_result *prepared)
{
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_model_failure failure;
    yvex_attention_failure attention_failure;
    yvex_runtime_session_open_request request;
    yvex_attention_state_provider_factory factory;
    injected_state_control control;
    yvex_runtime_model_summary summary;
    yvex_runtime_session_summary session_summary;
    yvex_graph_attention_state_summary state_before, state_after;
    unsigned long long checks_before_drift;
    yvex_error err;
    int fd, truncate_rc, close_rc;

    YVEX_TEST_ASSERT(runtime_model_open_fixture(
                         fixture, prepared, &model, &failure, &err) == YVEX_OK,
                     "runtime model opens before drift");
    memset(&control, 0, sizeof(control));
    factory = (yvex_attention_state_provider_factory){
        .context = &control,
        .open = injected_state_factory_open,
        .discard = injected_state_factory_discard,
    };
    memset(&request, 0, sizeof(request));
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.attention_state_factory = &factory;
    YVEX_TEST_ASSERT(yvex_runtime_session_open(
                         &session, model, &request, &failure, &err) == YVEX_OK,
                     "runtime session opens before drift");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_view_get(session)->attention_state_provider->summary(
            yvex_runtime_session_view_get(session)->attention_state_provider->context,
            &state_before, &err) == YVEX_OK,
        "runtime session state generation captured before drift");
    YVEX_TEST_ASSERT(runtime_state_prepare_fixture(
                         model, yvex_runtime_session_view_get(session)->attention_state_provider,
                         0ull, &attention_failure, &err) == YVEX_OK &&
                         yvex_runtime_session_begin(session, &failure, &err) == YVEX_OK &&
                         runtime_state_begin_fixture(
                             model, yvex_runtime_session_view_get(session)->attention_state_provider,
                             0ull, 0ull, 1ull, &attention_failure, &err) == YVEX_OK,
                     "runtime state owns an active candidate before drift invalidation");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK,
                     "runtime model snapshot copies immediately before drift");
    checks_before_drift = summary.drift_checks;
    fd = open(yvex_artifact_path(fixture->artifact), O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(fd >= 0, "runtime artifact fixture opened for drift");
    truncate_rc = ftruncate(fd, (off_t)fixture->admission.file_bytes - 1);
    close_rc = close(fd);
    YVEX_TEST_ASSERT(truncate_rc == 0 && close_rc == 0,
                     "runtime artifact fixture truncated after model seal");
    YVEX_TEST_ASSERT(
        (control.fail_invalidate_once = 1) == 1 &&
            yvex_runtime_model_validate(model, &failure, &err) != YVEX_OK &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_DRIFT &&
            control.invalidations == 0u && control.active->summary.transaction_active,
        "busy drift latches cancellation without racing the active provider candidate");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_finish(session, YVEX_ERR_IO, &err) == YVEX_ERR_STATE &&
            control.aborts == 1u && control.invalidations == 1u &&
            !control.active->summary.transaction_active,
        "quiescent finish aborts the candidate before exposing deferred invalidation failure");
    YVEX_TEST_ASSERT(
        yvex_runtime_model_validate(model, &failure, &err) != YVEX_OK &&
            failure.code == YVEX_RUNTIME_MODEL_FAILURE_DRIFT,
        "invalid model remains refused after deferred session invalidation");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK &&
                         !summary.valid && summary.invalidation_count == 1ull &&
                         summary.drift_checks == checks_before_drift + 2ull,
                     "runtime drift invalidates model once");
    YVEX_TEST_ASSERT(
        yvex_runtime_session_summary_copy(session, &session_summary, &err) == YVEX_OK &&
            yvex_runtime_session_view_get(session)->attention_state_provider->summary(
                yvex_runtime_session_view_get(session)->attention_state_provider->context,
                &state_after, &err) == YVEX_OK &&
            session_summary.invalidated && session_summary.cancelled &&
            !session_summary.residency_generation && !session_summary.workspace_generation &&
            !state_after.invalidated && !state_after.cancelled &&
            state_after.generation == state_before.generation,
        "failed deferred invalidation remains explicitly pending after candidate abort");
    YVEX_TEST_ASSERT(yvex_runtime_session_begin(session, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_MODEL_FAILURE_DRIFT,
                     "invalidated model refuses warm session dispatch");
    YVEX_TEST_ASSERT(yvex_runtime_model_summary_copy(model, &summary, &err) == YVEX_OK &&
                         summary.invalidation_count == 1ull,
                     "repeated drift refusal does not duplicate invalidation");
    YVEX_TEST_ASSERT(yvex_runtime_session_close(&session, &err) == YVEX_OK &&
                         control.invalidations == 2u && !control.active,
                     "session close retries deferred provider invalidation before release");
    yvex_runtime_model_close(&model);
    return 0;
}

int yvex_test_runtime_binding(void)
{
    binding_fixture fixture;
    yvex_materialization_projection projection = {0};
    yvex_runtime_binding_prepare_result prepared;
    yvex_runtime_binding *binding = NULL;
    char root[] = "/tmp/yvex-runtime-binding-test-XXXXXX";
    char artifact_path[YVEX_PATH_CAP];
    yvex_error err;
    int rc = 1;

    memset(&prepared, 0, sizeof(prepared));
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        yvex_materialization_project_artifact_lowering(NULL, &projection, &err) ==
                YVEX_ERR_INVALID_ARG &&
            strcmp(yvex_error_where(&err), "materialization.lowering") == 0,
        "materialization projection refuses a missing generic lowering map");
    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "runtime binding temporary root");
    YVEX_TEST_ASSERT(snprintf(artifact_path, sizeof(artifact_path), "%s/runtime.gguf", root) <
                         (int)sizeof(artifact_path) &&
                         copy_regular_file("tests/fixtures/gguf/valid-tokenizer-simple.gguf",
                                           artifact_path) &&
                         rewrite_attention_artifact_fixture(artifact_path),
                     "runtime artifact fixture copied and bound to one attention tensor");
    YVEX_TEST_ASSERT(fixture_build(&fixture, artifact_path, 1),
                     "runtime binding fixture built");
    if (test_runtime_capability_contract() != 0) goto done;
    if (test_prepare_reopen_import(&fixture, root, &prepared, &binding) != 0) goto done;
    if (test_corruption_refusals(&prepared, root) != 0) goto done;
    if (test_canonical_refusals(&prepared, root) != 0) goto done;
    if (test_graph_identity_refusals(&prepared, root) != 0) goto done;
    if (test_artifact_copy_portability(&fixture, &prepared, root) != 0) goto done;
    if (test_compiled_model_binding_v12(root) != 0) goto done;
    if (test_runtime_family_neutrality() != 0) goto done;
    if (test_runtime_model_compiled_execution(&fixture, &prepared) != 0) goto done;
    if (test_runtime_model_progress(&fixture, &prepared) != 0) goto done;
    if (test_runtime_model_verified_reopen(&fixture, &prepared, root) != 0) goto done;
    if (test_runtime_model_session_reuse(&fixture, &prepared) != 0) goto done;
    if (test_runtime_concurrent_session_isolation(&fixture, &prepared) != 0) goto done;
    if (test_runtime_concurrent_close_drain(&fixture, &prepared) != 0) goto done;
    if (test_runtime_model_close_drain(&fixture, &prepared) != 0) goto done;
    if (test_runtime_session_owner_retry(&fixture, &prepared) != 0) goto done;
    if (test_runtime_state_factory_candidate_cleanup(&fixture, &prepared) != 0) goto done;
    if (test_runtime_injected_state_provider(&fixture, &prepared) != 0) goto done;
    if (test_runtime_cleanup_lease_retry(&fixture, &prepared) != 0) goto done;
    if (test_runtime_probe_consumer_boundary(&fixture, &prepared, root) != 0) goto done;
    if (test_runtime_paged_state_cuda_pack(&fixture, &prepared) != 0) goto done;
    if (test_runtime_model_cuda_residency_claim(&fixture, &prepared) != 0) goto done;
    if (test_runtime_cuda_session_cleanup_retry(&fixture, &prepared) != 0) goto done;
    if (test_runtime_cuda_workspace_transaction(&fixture, &prepared) != 0) goto done;
    if (test_runtime_model_snapshot_drift(&fixture, &prepared) != 0) goto done;
    rc = 0;

done:
    yvex_runtime_binding_close(binding);
    fixture_close(&fixture);
    if (prepared.path[0]) (void)unlink(prepared.path);
    (void)unlink(artifact_path);
    (void)rmdir(root);
    return rc;
}
