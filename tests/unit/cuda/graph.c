/*
 * CUDA launch-graph lifecycle tests. Proves optional Driver admission,
 * capture/instantiate/upload/replay/update, explicit inventory and identities, typed refusal,
 * and retryable cleanup.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>
#include <yvex/internal/backend.h>

#include "src/backend/cuda/private.h"
#include "tests/test.h"

static void make_desc(yvex_backend_tensor_desc *desc, const char *name)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = YVEX_DTYPE_F32;
    desc->rank = 1u;
    desc->dims[0] = 8ull;
    desc->bytes = 8ull * (unsigned long long)sizeof(float);
}

static void make_count_desc(yvex_backend_tensor_desc *desc, const char *name,
                            yvex_dtype dtype, unsigned long long count, size_t width)
{
    memset(desc, 0, sizeof(*desc));
    desc->name = name;
    desc->dtype = dtype;
    desc->rank = 1u;
    desc->dims[0] = count;
    desc->bytes = count * (unsigned long long)width;
}

static int open_cuda(yvex_backend **out)
{
    yvex_backend_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(out, &options, &err);
    if (rc == YVEX_ERR_UNSUPPORTED) {
        fprintf(stderr, "SKIP: CUDA unavailable: %s\n", yvex_error_message(&err));
        return 77;
    }
    YVEX_TEST_ASSERT(rc == YVEX_OK, "open CUDA backend for graph lifecycle");
    return 0;
}

static int test_attention_piece_inventory(void)
{
    unsigned long long core_swa = 0ull, core_csa = 0ull;
    unsigned long long envelope_swa = 0ull, envelope_csa = 0ull;
    yvex_cuda_attention_stage stage;

    for (stage = YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_PRE;
         stage < YVEX_CUDA_ATTENTION_STAGE_COUNT; ++stage) {
        core_swa += (unsigned long long)cuda_attention_piece_active(
            YVEX_BACKEND_ATTENTION_SCOPE_CORE,
            YVEX_BACKEND_ATTENTION_SWA, stage);
        core_csa += (unsigned long long)cuda_attention_piece_active(
            YVEX_BACKEND_ATTENTION_SCOPE_CORE,
            YVEX_BACKEND_ATTENTION_CSA, stage);
        envelope_swa += (unsigned long long)cuda_attention_piece_active(
            YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE,
            YVEX_BACKEND_ATTENTION_SWA, stage);
        envelope_csa += (unsigned long long)cuda_attention_piece_active(
            YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE,
            YVEX_BACKEND_ATTENTION_CSA, stage);
    }
    YVEX_TEST_ASSERT(
        core_swa == 2ull && core_csa == 3ull &&
            envelope_swa == 4ull && envelope_csa == 5ull,
        "piecewise attention inventory contains only active semantic stages");
    YVEX_TEST_ASSERT(
        !cuda_attention_piece_active(
            YVEX_BACKEND_ATTENTION_SCOPE_CORE,
            YVEX_BACKEND_ATTENTION_HCA,
            YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_PRE) &&
            !cuda_attention_piece_active(
                YVEX_BACKEND_ATTENTION_SCOPE_CORE,
                YVEX_BACKEND_ATTENTION_HCA,
                YVEX_CUDA_ATTENTION_STAGE_ENVELOPE_POST) &&
            !cuda_attention_piece_active(
                YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE,
                YVEX_BACKEND_ATTENTION_HCA,
                YVEX_CUDA_ATTENTION_STAGE_COUNT),
        "piecewise attention inventory rejects inactive and invalid stages");
    return 0;
}

static CUresult (*deferred_real_free)(CUdeviceptr);
static CUdeviceptr deferred_expected_pointer;
static unsigned int deferred_free_calls;
static unsigned int deferred_true_frees;
static CUresult (*module_real_unload)(CUmodule);
static CUmodule module_expected_handle;
static unsigned int module_unload_calls;
static unsigned int module_true_unloads;
static unsigned int module_unload_failures;

static CUresult fail_once_before_device_free(CUdeviceptr pointer)
{
    ++deferred_free_calls;
    if (deferred_free_calls == 1u)
        return YVEX_CUDA_ERROR_NOT_INITIALIZED;
    if (pointer == deferred_expected_pointer)
        ++deferred_true_frees;
    return deferred_real_free(pointer);
}

static CUresult fail_before_module_unload(CUmodule module)
{
    ++module_unload_calls;
    if (module_unload_failures) {
        module_expected_handle = module;
        --module_unload_failures;
        return YVEX_CUDA_ERROR_NOT_INITIALIZED;
    }
    if (module == module_expected_handle)
        ++module_true_unloads;
    return module_real_unload(module);
}

/* Prove failed raw cleanup transfers exact ownership/accounting to checked backend retry. */
static int test_deferred_raw_release(void)
{
    const unsigned long long bytes = 64ull;
    yvex_backend_memory_stats before, deferred, after;
    yvex_cuda_backend_state *state;
    yvex_cuda_work work;
    yvex_backend *backend = NULL;
    yvex_error err;
    CUdeviceptr pointer = 0u;
    int retained_ok;
    int rc = open_cuda(&backend);

    if (rc != 0)
        return rc;
    state = yvex_cuda_state(backend);
    memset(&work, 0, sizeof(work));
    rc = yvex_backend_get_memory_stats(backend, &before, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "snapshot raw cleanup accounting");
    rc = yvex_backend_memory_can_add(
        backend, bytes, "CUDA", "cuda.test.deferred.alloc", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "admit raw cleanup fixture bytes");
    rc = yvex_cuda_set_current(backend, "cuda.test.deferred.context", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "select raw cleanup fixture context");
    rc = yvex_cuda_status(&state->driver,
                          state->driver.cuMemAlloc_v2(&pointer, (size_t)bytes),
                          "cuda.test.deferred.alloc", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && pointer != 0u,
                     "allocate raw cleanup fixture before Driver fault");
    backend_memory_acquire(backend, bytes);
    work.backend = backend;
    work.state = state;
    work.variant = YVEX_BACKEND_VARIANT_TENSOR_ALLOC;
    work.pointers[0] = pointer;
    work.sizes[0] = bytes;
    work.count = 1u;
    work.current_bytes = bytes;
    deferred_real_free = state->driver.cuMemFree_v2;
    deferred_expected_pointer = pointer;
    deferred_free_calls = 0u;
    deferred_true_frees = 0u;
    state->driver.cuMemFree_v2 = fail_once_before_device_free;

    rc = yvex_cuda_work_cleanup(&work, &err);
    retained_ok = rc == YVEX_ERR_BACKEND && deferred_free_calls == 1u &&
                  deferred_true_frees == 0u && work.count == 0u &&
                  work.current_bytes == 0ull &&
                  state->deferred_release_count == 1u &&
                  state->deferred_release_bytes == bytes &&
                  state->deferred_releases[0].pointer == pointer;
    YVEX_TEST_ASSERT(yvex_backend_get_memory_stats(backend, &deferred, &err) == YVEX_OK,
                     "inspect accounting retained by deferred raw owner");
    retained_ok = retained_ok &&
                  deferred.allocated_bytes == before.allocated_bytes + bytes &&
                  deferred.allocation_count == before.allocation_count + 1ull;

    rc = yvex_cuda_deferred_release_drain(backend, &err);
    state->driver.cuMemFree_v2 = deferred_real_free;
    YVEX_TEST_ASSERT(rc == YVEX_OK && deferred_free_calls == 2u &&
                         deferred_true_frees == 1u,
                     "retry performs the first and only true Driver free");
    YVEX_TEST_ASSERT(state->deferred_release_count == 0u &&
                         state->deferred_release_bytes == 0ull,
                     "successful retry empties deferred raw ownership");
    YVEX_TEST_ASSERT(yvex_backend_get_memory_stats(backend, &after, &err) == YVEX_OK &&
                         after.allocated_bytes == before.allocated_bytes &&
                         after.allocation_count == before.allocation_count,
                     "true Driver free restores exact live allocation accounting");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !backend, "close deferred-release test backend");
    YVEX_TEST_ASSERT(retained_ok,
                     "pre-free failure preserves pointer and bytes under backend ownership");
    return 0;
}

static int test_module_release_retry(void)
{
    yvex_backend *backend = NULL;
    yvex_cuda_backend_state *state;
    CUfunction expected_function;
    unsigned int expected_modules;
    yvex_error err;
    int retained_ok;
    int rc = open_cuda(&backend);

    if (rc != 0)
        return rc;
    state = yvex_cuda_state(backend);
    expected_modules = state->module_count;
    module_real_unload = state->driver.cuModuleUnload;
    module_expected_handle = state->modules[0];
    expected_function = state->rope_function;
    module_unload_calls = 0u;
    module_true_unloads = 0u;
    module_unload_failures = 1u;
    state->driver.cuModuleUnload = fail_before_module_unload;
    rc = yvex_backend_close_checked(&backend, &err);
    retained_ok = rc == YVEX_ERR_BACKEND && backend != NULL &&
                  yvex_cuda_state(backend) == state &&
                  state->module_count == expected_modules &&
                  state->modules[0] == module_expected_handle &&
                  state->rope_function == expected_function &&
                  module_unload_calls == 1u && module_true_unloads == 0u;
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !backend &&
                         module_unload_calls == expected_modules + 1u &&
                         module_true_unloads == 1u,
                     "checked-close retry performs one true module unload");
    YVEX_TEST_ASSERT(retained_ok,
                     "pre-unload failure preserves module and function ownership");
    return 0;
}

static int test_module_rollback_retry(void)
{
    yvex_backend *backend = NULL;
    yvex_cuda_backend_state *state;
    CUmodule admitted_module;
    CUfunction admitted_function;
    unsigned int expected_modules;
    yvex_error err;
    int retained_ok, rc = open_cuda(&backend);

    if (rc != 0) return rc;
    state = yvex_cuda_state(backend);
    expected_modules = state->module_count;
    admitted_module = state->modules[0];
    admitted_function = state->rope_function;
    YVEX_TEST_ASSERT(yvex_cuda_kernel_bundle_admit(backend, &err) == YVEX_OK &&
                         state->module_count == expected_modules &&
                         state->modules[0] == admitted_module &&
                         state->rope_function == admitted_function,
                     "repeated bundle admission preserves the admitted module");
    YVEX_TEST_ASSERT(yvex_cuda_kernel_bundle_close(backend, &err) == YVEX_OK,
                     "release admitted bundle before rollback fixture");
    module_real_unload = state->driver.cuModuleUnload;
    module_expected_handle = NULL;
    module_unload_calls = module_true_unloads = 0u;
    module_unload_failures = 1u;
    state->driver.cuModuleUnload = fail_before_module_unload;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", "symbol", 1) == 0,
                     "inject symbol refusal before rollback unload");
    rc = yvex_cuda_kernel_bundle_admit(backend, &err);
    retained_ok = rc == YVEX_ERR_BACKEND &&
                  state->module_count == expected_modules && state->modules[0] &&
                  module_unload_calls == 0u && module_true_unloads == 0u;
    module_expected_handle = state->modules[0];
    YVEX_TEST_ASSERT(yvex_cuda_kernel_bundle_admit(backend, &err) == YVEX_ERR_STATE &&
                         state->modules[0] == module_expected_handle && module_unload_calls == 0u,
                     "bundle re-admission refuses retained cleanup ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE") == 0 &&
                         yvex_backend_close_checked(&backend, &err) == YVEX_ERR_BACKEND &&
                         backend && module_unload_calls == 1u && module_true_unloads == 0u,
                     "first checked close retains the rejected module after Driver refusal");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&backend, &err) == YVEX_OK && !backend &&
                         module_unload_calls == expected_modules + 1u &&
                         module_true_unloads == 1u,
                     "checked backend close retries the retained rejected module");
    YVEX_TEST_ASSERT(retained_ok,
                     "failed rollback unload preserves module ownership");
    return 0;
}

/* Prove a shared-open rollback publishes its failed backend until close can retry. */
static int test_shared_open_rollback_retry(void)
{
    yvex_backend *owner = NULL, *retained = NULL;
    yvex_cuda_backend_state *owner_state, *retained_state;
    yvex_backend_capability_result capability;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *tensor = NULL;
    yvex_error err;
    yvex_backend_capability aggregate;
    yvex_backend_operation_variant variant;
    unsigned int expected_modules;
    int retained_ok, rc = open_cuda(&owner);

    if (rc != 0) return rc;
    owner_state = yvex_cuda_state(owner);
    expected_modules = owner_state->module_count;
    module_real_unload = owner_state->driver.cuModuleUnload;
    module_expected_handle = NULL;
    module_unload_calls = module_true_unloads = 0u;
    module_unload_failures = 2u;
    owner_state->driver.cuModuleUnload = fail_before_module_unload;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_BUNDLE_FAILURE", "symbol", 1) == 0,
                     "inject shared bundle admission refusal");
    rc = yvex_backend_open_shared_cuda(&retained, owner, 0ull, &err);
    retained_state = yvex_cuda_state(retained);
    retained_ok = rc == YVEX_ERR_BACKEND && retained && retained_state &&
                  yvex_backend_status_of(retained) == YVEX_BACKEND_STATUS_FAILED &&
                  retained_state->modules[0] == module_expected_handle &&
                  retained_state->module_count == expected_modules &&
                  module_unload_calls == 1u &&
                  module_true_unloads == 0u;
    for (variant = YVEX_BACKEND_VARIANT_TENSOR_ALLOC;
         variant < YVEX_BACKEND_VARIANT_COUNT; ++variant) {
        rc = yvex_backend_query_capability(retained, variant, &capability, &err);
        retained_ok = retained_ok && rc == YVEX_OK &&
                      capability.state == YVEX_BACKEND_CAPABILITY_FAILED &&
                      capability.reason == YVEX_BACKEND_CAPABILITY_REASON_CLEANUP_FAILED &&
                      yvex_cuda_require_capability(
                          retained, variant, "cuda.test.retained_cleanup", &err) ==
                          YVEX_ERR_BACKEND;
    }
    for (aggregate = YVEX_BACKEND_CAP_TENSOR_ALLOC;
         aggregate <= YVEX_BACKEND_CAP_OP_ATTENTION; ++aggregate)
        retained_ok = retained_ok && !yvex_backend_supports(retained, aggregate);
    make_desc(&desc, "retained_cleanup_refusal");
    rc = yvex_backend_tensor_alloc(retained, &desc, &tensor, &err);
    retained_ok = retained_ok && rc == YVEX_ERR_STATE && !tensor;
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_BUNDLE_FAILURE") == 0,
                     "clear shared bundle admission refusal");
    module_unload_failures = 0u;
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&retained, &err) == YVEX_OK && !retained &&
                         module_unload_calls == expected_modules + 1u &&
                         module_true_unloads == 1u,
                     "shared cleanup owner retries one true module unload");
    owner_state->driver.cuModuleUnload = module_real_unload;
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&owner, &err) == YVEX_OK && !owner,
                     "shared rollback context owner closes independently");
    YVEX_TEST_ASSERT(retained_ok,
                     "shared-open cleanup failure publishes the exact failed owner");
    return 0;
}

static int test_shared_context_close_order(void)
{
    yvex_backend_capability_result capability;
    yvex_backend *owner = NULL, *child = NULL, *blocked = NULL;
    yvex_error err;
    int rc = open_cuda(&owner);

    if (rc != 0) return rc;
    rc = yvex_backend_open_shared_cuda(&child, owner, 0ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && child,
                     "open one shared CUDA session child");
    rc = yvex_backend_query_capability(
        child, YVEX_BACKEND_VARIANT_TENSOR_ALLOC, &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         capability.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "shared CUDA child is executable before owner close refusal");
    rc = yvex_backend_close_checked(&owner, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && owner,
                     "context owner close refuses while a shared child is live");
    rc = yvex_backend_open_shared_cuda(&blocked, owner, 0ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE && !blocked,
                     "terminal owner close admission refuses every new shared child");
    rc = yvex_backend_query_capability(
        child, YVEX_BACKEND_VARIANT_TENSOR_ALLOC, &capability, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         capability.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "owner close refusal preserves the live shared child");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&child, &err) == YVEX_OK && !child,
                     "shared CUDA child closes before its context owner");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&owner, &err) == YVEX_OK && !owner,
                     "context owner closes after its final shared child");
    return 0;
}

static int test_capability_projection(yvex_backend *backend)
{
    yvex_backend_cuda_graph_capability graph;
    yvex_backend_capability_result eager;
    yvex_error err;
    int rc;

    rc = yvex_backend_cuda_graph_query(backend, &graph, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "query CUDA graph capability");
    YVEX_TEST_ASSERT(graph.schema == YVEX_BACKEND_CUDA_GRAPH_SCHEMA,
                     "CUDA graph capability schema");
    YVEX_TEST_ASSERT(graph.state == YVEX_BACKEND_CUDA_GRAPH_OPEN,
                     "CUDA graph Driver lifecycle available");
    YVEX_TEST_ASSERT(graph.stream_api_available && graph.graph_api_available &&
                     graph.update_api_available,
                     "CUDA stream, graph, and update APIs available");
    YVEX_TEST_ASSERT(graph.async_memory_available && graph.async_copy_available &&
                     graph.pinned_host_memory_available && graph.event_timing_available,
                     "optional asynchronous, pinned, and event APIs loaded");

    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_API", "absent", 1) == 0,
                     "inject absent CUDA graph API");
    rc = yvex_backend_cuda_graph_query(backend, &graph, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     graph.state == YVEX_BACKEND_CUDA_GRAPH_UNAVAILABLE &&
                     graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_STREAM_API_UNAVAILABLE,
                     "optional graph API absence is typed");
    rc = yvex_backend_query_capability(backend, YVEX_BACKEND_VARIANT_ROPE_F32,
                                       &eager, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     eager.state == YVEX_BACKEND_CAPABILITY_SUPPORTED,
                     "optional graph API absence preserves eager admission");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_API") == 0,
                     "clear absent CUDA graph API injection");
    return 0;
}

typedef struct {
    yvex_backend *backend;
    const yvex_device_tensor *input;
    yvex_device_tensor *output;
    float *staged_input;
    float *staged_output;
    unsigned long long position;
    CUfunction replay_function;
    yvex_cuda_work work;
} attention_graph_fixture;

typedef struct {
    yvex_backend *backend;
    yvex_device_tensor *input;
    yvex_device_tensor *output;
    unsigned long long position;
    unsigned int prepare_count;
    int prepare_failure;
} rope_graph_fixture;

typedef struct {
    yvex_backend *backend;
    yvex_device_tensor *before_kv, *before_score, *token_kv, *token_score, *ape;
    yvex_device_tensor *after_kv, *after_score, *compressed, *status;
    unsigned long long cursor;
} rolling_graph_fixture;

static int enqueue_rope_fixture(void *opaque, int enqueue_kernels, yvex_error *err)
{
    rope_graph_fixture *fixture = (rope_graph_fixture *)opaque;
    yvex_cuda_backend_state *state = yvex_cuda_state(fixture->backend);
    CUdeviceptr input;
    CUdeviceptr output;
    unsigned long long width = 8ull;
    unsigned long long position = fixture->position;
    float inverse_root = 0.1f;
    void *params[5];

    if (!enqueue_kernels) {
        input = yvex_cuda_tensor_ptr(fixture->input);
        output = yvex_cuda_tensor_ptr(fixture->output);
        params[0] = &input;
        params[1] = &output;
        params[2] = &width;
        params[3] = &position;
        params[4] = &inverse_root;
        return yvex_cuda_graph_kernel_update(
            fixture->backend, YVEX_BACKEND_VARIANT_ROPE_F32,
            state->rope_function, 1u, 128u, 0u, params,
            "cuda.rope.launch", err);
    }
    return yvex_backend_op_rope(fixture->backend, fixture->input, fixture->position,
                                10000.0f, fixture->output, err);
}

static int enqueue_empty_fixture(void *opaque, int enqueue_kernels, yvex_error *err)
{
    (void)opaque;
    (void)enqueue_kernels;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int prepare_rope_fixture(void *opaque, yvex_error *err)
{
    rope_graph_fixture *fixture = (rope_graph_fixture *)opaque;
    fixture->prepare_count++;
    if (fixture->prepare_failure) {
        yvex_error_set(err, YVEX_ERR_BACKEND, "cuda.graph.fixture.prepare",
                       "injected graph preamble failure");
        return YVEX_ERR_BACKEND;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int enqueue_rolling_fixture(void *opaque, int enqueue_kernels, yvex_error *err)
{
    rolling_graph_fixture *fixture = (rolling_graph_fixture *)opaque;
    yvex_cuda_backend_state *state = yvex_cuda_state(fixture->backend);
    CUdeviceptr before_kv = yvex_cuda_tensor_ptr(fixture->before_kv);
    CUdeviceptr before_score = yvex_cuda_tensor_ptr(fixture->before_score);
    CUdeviceptr token_kv = yvex_cuda_tensor_ptr(fixture->token_kv);
    CUdeviceptr token_score = yvex_cuda_tensor_ptr(fixture->token_score);
    CUdeviceptr ape = yvex_cuda_tensor_ptr(fixture->ape);
    CUdeviceptr after_kv = yvex_cuda_tensor_ptr(fixture->after_kv);
    CUdeviceptr after_score = yvex_cuda_tensor_ptr(fixture->after_score);
    CUdeviceptr compressed = yvex_cuda_tensor_ptr(fixture->compressed);
    CUdeviceptr status = yvex_cuda_tensor_ptr(fixture->status);
    unsigned long long ratio = 4ull, head = 2ull, width = 2ull, slots = 4ull;
    int overlap = 0, emit = 0;
    void *params[] = {
        &before_kv, &before_score, &token_kv, &token_score, &ape,
        &after_kv, &after_score, &compressed, &ratio, &head, &width,
        &slots, &fixture->cursor, &overlap, &emit, &status
    };
    const char *stage = "cuda.graph.fixture.rolling";

    if (!state || !state->deepseek_rolling_function) {
        yvex_error_set(err, YVEX_ERR_STATE, stage,
                       "admitted rolling kernel function is required");
        return YVEX_ERR_STATE;
    }
    return enqueue_kernels
        ? yvex_cuda_launch(
              fixture->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
              state->deepseek_rolling_function, 1u, 256u, 0u, params, stage, err)
        : yvex_cuda_graph_kernel_update(
              fixture->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
              state->deepseek_rolling_function, 1u, 256u, 0u, params, stage, err);
}

/* Prove a warm graph replay updates the rolling insertion cursor. */
static int test_rolling_cursor_update(yvex_backend *backend)
{
    const float token[2] = {1.25f, -2.5f};
    float zero_state[8] = {0}, zero_pair[2] = {0}, result[8] = {0};
    int zero_status = 0;
    rolling_graph_fixture fixture;
    yvex_backend_cuda_graph_info info;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor **tensors[] = {
        &fixture.before_kv, &fixture.before_score, &fixture.after_kv,
        &fixture.after_score
    };
    yvex_error err;
    size_t i;
    int rc;

    memset(&fixture, 0, sizeof(fixture));
    fixture.backend = backend;
    for (i = 0u; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        make_count_desc(&desc, "rolling_state", YVEX_DTYPE_F32, 8ull, sizeof(float));
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_alloc(backend, &desc, tensors[i], &err) == YVEX_OK &&
                yvex_backend_tensor_write(
                    backend, *tensors[i], zero_state, sizeof(zero_state), &err) == YVEX_OK,
            "allocate and initialize rolling state fixture");
    }
#define ALLOC_PAIR(field, label, values)                                                   \
    do {                                                                                    \
        make_count_desc(&desc, label, YVEX_DTYPE_F32, 2ull, sizeof(float));                 \
        YVEX_TEST_ASSERT(                                                                   \
            yvex_backend_tensor_alloc(backend, &desc, &fixture.field, &err) == YVEX_OK &&   \
                yvex_backend_tensor_write(backend, fixture.field, values,                    \
                                          sizeof(zero_pair), &err) == YVEX_OK,               \
            "allocate and initialize rolling vector fixture");                              \
    } while (0)
    ALLOC_PAIR(token_kv, "rolling_token_kv", token);
    ALLOC_PAIR(token_score, "rolling_token_score", zero_pair);
    ALLOC_PAIR(ape, "rolling_ape", zero_pair);
    ALLOC_PAIR(compressed, "rolling_compressed", zero_pair);
#undef ALLOC_PAIR
    make_count_desc(&desc, "rolling_status", YVEX_DTYPE_I32, 1ull, sizeof(int));
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &fixture.status, &err) == YVEX_OK &&
            yvex_backend_tensor_write(
                backend, fixture.status, &zero_status, sizeof(zero_status), &err) == YVEX_OK,
        "allocate and initialize rolling status fixture");

    rc = yvex_cuda_graph_execute(
        backend, "cuda-rolling-dynamic-cursor-v1", NULL, enqueue_rolling_fixture,
        &fixture, 8u, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG,
                     "graph execution refuses unsupported execution flags");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_EVENT_FAILURE", "synchronize", 1) == 0,
                     "inject an event synchronization failure");
    fixture.cursor = 3ull;
    rc = yvex_cuda_graph_execute(
        backend, "cuda-rolling-dynamic-cursor-v1", NULL, enqueue_rolling_fixture,
        &fixture, 0, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && info.capture_count == 1ull &&
                         info.last_device_elapsed_ns == 0ull,
                     "capture one real rolling transition without device timing");
    fixture.cursor = 0ull;
    rc = yvex_cuda_graph_execute(
        backend, "cuda-rolling-dynamic-cursor-v1", NULL, enqueue_rolling_fixture,
        &fixture, 0, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && info.replay_count == 2ull &&
                         info.last_device_elapsed_ns == 0ull,
                     "replay rolling transition without timing synchronization");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_EVENT_FAILURE") == 0,
                     "clear event synchronization failure injection");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_read(
            backend, fixture.after_kv, result, sizeof(result), &err) == YVEX_OK &&
            result[0] == token[0] && result[1] == token[1] &&
            result[6] == 0.0f && result[7] == 0.0f,
        "warm replay applies the current rolling cursor");

    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &fixture.status, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fixture.compressed, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fixture.ape, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fixture.token_score, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fixture.token_kv, &err) == YVEX_OK,
        "release rolling vector fixtures");
    for (i = 0u; i < sizeof(tensors) / sizeof(tensors[0]); ++i)
        YVEX_TEST_ASSERT(
            yvex_backend_tensor_release(backend, tensors[i], &err) == YVEX_OK,
            "release rolling state fixture");
    return 0;
}

/* Prove graph pieces borrow the session stream and publish only after its explicit barrier. */
static int test_shared_graph_completion(yvex_backend *backend)
{
    const float input_data[8] = {0.25f, -0.5f, 1.0f, -1.5f, 2.0f, -2.5f, 3.0f, -3.5f};
    float graph_data[8] = {0}, eager_data[8] = {0};
    yvex_backend_cuda_graph_info info;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *graph_output = NULL;
    yvex_device_tensor *piece_output = NULL;
    yvex_device_tensor *eager_output = NULL;
    rope_graph_fixture fixture = {0};
    rope_graph_fixture next_piece = {0};
    yvex_error err;
    int device_wide = 1;
    int rc;

    make_desc(&desc, "shared_graph_input");
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &desc, &input, &err) == YVEX_OK,
                     "allocate shared-stream graph input");
    make_desc(&desc, "shared_graph_output");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &graph_output, &err) == YVEX_OK,
        "allocate shared-stream graph output");
    make_desc(&desc, "shared_graph_piece");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &piece_output, &err) == YVEX_OK,
        "allocate ordered graph-piece output");
    make_desc(&desc, "shared_graph_eager");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &eager_output, &err) == YVEX_OK,
        "allocate shared-stream eager output");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_write(backend, input, input_data, sizeof(input_data), &err) == YVEX_OK,
        "write shared-stream graph input");
    fixture = (rope_graph_fixture){
        .backend = backend, .input = input, .output = graph_output, .position = 7ull
    };
    rc = yvex_cuda_graph_execute(
        backend, "cuda-shared-stream-rope-v1", NULL, enqueue_rope_fixture, &fixture,
        YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
            YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION,
        &info, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && info.shared_launch_stream && info.completion_pending &&
            info.capture_count == 1ull && info.replay_count == 1ull &&
            info.synchronize_count == 0ull && info.last_replay_elapsed_ns == 0ull &&
            info.last_device_elapsed_ns == 0ull,
        "shared-stream graph capture leaves completion to its transaction owner");
    rc = yvex_cuda_launch_synchronize(
        backend, YVEX_BACKEND_VARIANT_ROPE_F32, &device_wide,
        "cuda.test.shared-graph.capture", &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && device_wide == 0,
                     "shared-stream graph completes through one scoped barrier");

    fixture.position = 8ull;
    rc = yvex_cuda_graph_execute(
        backend, "cuda-shared-stream-rope-v1", NULL, enqueue_rope_fixture, &fixture,
        YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
            YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION,
        &info, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && info.shared_launch_stream && info.completion_pending &&
            info.capture_count == 1ull && info.replay_count == 2ull &&
            info.synchronize_count == 0ull,
        "warm shared-stream replay updates parameters without a graph-local barrier");
    next_piece = (rope_graph_fixture){
        .backend = backend, .input = graph_output, .output = piece_output, .position = 9ull
    };
    rc = yvex_cuda_graph_execute(
        backend, "cuda-shared-stream-rope-piece-v1", NULL, enqueue_rope_fixture, &next_piece,
        YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
            YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION,
        &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && info.completion_pending &&
                         info.capture_count == 1ull && info.synchronize_count == 0ull,
                     "a later graph piece captures behind pending work on the same stream");
    device_wide = 1;
    YVEX_TEST_ASSERT(
        yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ROPE_F32, &device_wide,
            "cuda.test.shared-graph.replay", &err) == YVEX_OK &&
            device_wide == 0 &&
            yvex_backend_tensor_read(
                backend, piece_output, graph_data, sizeof(graph_data), &err) == YVEX_OK,
        "one barrier completes and exposes every ordered graph piece");
    YVEX_TEST_ASSERT(
        yvex_backend_op_rope(backend, input, fixture.position, 10000.0f,
                             graph_output, &err) == YVEX_OK &&
            yvex_backend_op_rope(backend, graph_output, next_piece.position, 10000.0f,
                                 eager_output, &err) == YVEX_OK &&
            yvex_cuda_launch_synchronize(
                backend, YVEX_BACKEND_VARIANT_ROPE_F32, &device_wide,
                "cuda.test.shared-graph.reference", &err) == YVEX_OK &&
            yvex_backend_tensor_read(
                backend, eager_output, eager_data, sizeof(eager_data), &err) == YVEX_OK &&
            memcmp(graph_data, eager_data, sizeof(graph_data)) == 0,
        "deferred shared-stream replay matches eager numerical execution exactly");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &eager_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &piece_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &graph_output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK,
        "release shared-stream graph fixtures");
    return 0;
}

/* Prove allocation-stable dynamic work runs before both capture and replay. */
static int test_graph_prepare_lifecycle(yvex_backend *backend)
{
    rope_graph_fixture fixture = {0};
    yvex_backend_cuda_graph_info info;
    yvex_backend_tensor_desc desc;
    yvex_error err;
    int rc;

    fixture.backend = backend;
    fixture.position = 5ull;
    make_desc(&desc, "graph_prepare_input");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &fixture.input, &err) == YVEX_OK,
        "allocate graph preamble input");
    make_desc(&desc, "graph_prepare_output");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_alloc(backend, &desc, &fixture.output, &err) == YVEX_OK,
        "allocate graph preamble output");
    rc = yvex_cuda_graph_execute(
        backend, "cuda-graph-prepare-lifecycle-v1", prepare_rope_fixture,
        enqueue_rope_fixture, &fixture, 1, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && fixture.prepare_count == 1u &&
                         info.capture_count == 1ull && info.replay_count == 1ull,
                     "graph preamble precedes initial capture");
    rc = yvex_cuda_graph_execute(
        backend, "cuda-graph-prepare-lifecycle-v1", prepare_rope_fixture,
        enqueue_rope_fixture, &fixture, 1, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && fixture.prepare_count == 2u &&
                         info.capture_count == 1ull && info.replay_count == 2ull,
                     "graph preamble repeats without recapture");
    fixture.prepare_failure = 1;
    rc = yvex_cuda_graph_execute(
        backend, "cuda-graph-prepare-lifecycle-v1", prepare_rope_fixture,
        enqueue_rope_fixture, &fixture, 1, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && fixture.prepare_count == 3u,
                     "graph preamble failure refuses launch");
    fixture.prepare_failure = 0;
    rc = yvex_cuda_graph_execute(
        backend, "cuda-graph-prepare-lifecycle-v1", prepare_rope_fixture,
        enqueue_rope_fixture, &fixture, 1, &info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && info.capture_count == 1ull &&
                         info.replay_count == 3ull,
                     "preamble failure preserves the admitted executable");
    YVEX_TEST_ASSERT(
        yvex_backend_tensor_release(backend, &fixture.output, &err) == YVEX_OK &&
            yvex_backend_tensor_release(backend, &fixture.input, &err) == YVEX_OK,
        "release graph preamble tensors");
    return 0;
}

/*
 * Enqueue pinned H2D, memset, production kernel, and D2H nodes for registry proof.
 *
 * Stable fixture and capture/replay enqueue policy. Captures the complete bounded transfer/kernel
 * unit only during initial/update capture. Returns exact copy or production-kernel refusal without
 * CPU fallback. Replay preparation performs no Driver work and owns no allocation.
 */
static int enqueue_attention_fixture(void *opaque, int enqueue_kernels, yvex_error *err)
{
    attention_graph_fixture *fixture = (attention_graph_fixture *)opaque;
    yvex_backend_attention_failure failure;
    CUdeviceptr input;
    CUdeviceptr output;
    unsigned long long width = 8ull;
    unsigned long long position = fixture->position;
    float inverse_root = 0.1f;
    void *params[5];
    int rc;

    fixture->work.prepare_only = !enqueue_kernels;
    if (!enqueue_kernels) {
        input = yvex_cuda_tensor_ptr(fixture->input);
        output = yvex_cuda_tensor_ptr(fixture->output);
        params[0] = &input;
        params[1] = &output;
        params[2] = &width;
        params[3] = &position;
        params[4] = &inverse_root;
        return yvex_cuda_graph_kernel_update(
            fixture->backend, YVEX_BACKEND_VARIANT_ROPE_F32,
            fixture->replay_function ? fixture->replay_function
                                     : yvex_cuda_state(fixture->backend)->rope_function,
            1u, 128u, 0u,
            params, "cuda.rope.launch", err);
    }
    rc = yvex_cuda_attention_operations_get()->initialize(
        &fixture->work, yvex_cuda_tensor_ptr(fixture->input), 8u * sizeof(float),
        fixture->staged_input, 0, "cuda.graph.fixture.h2d", &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_attention_operations_get()->initialize(
            &fixture->work, yvex_cuda_tensor_ptr(fixture->output), 8u * sizeof(float),
            NULL, 1, "cuda.graph.fixture.memset", &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_op_rope(fixture->backend, fixture->input, fixture->position, 10000.0f,
                                  fixture->output, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_attention_operations_get()->download(
            &fixture->work, fixture->staged_output,
            yvex_cuda_tensor_ptr(fixture->output), 8u * sizeof(float),
            "cuda.graph.fixture.d2h", &failure, err);
    return rc;
}

static int test_attention_graph_configuration(yvex_backend *backend)
{
    const float input_data[8] = {1.0f, -2.0f, 3.0f, -4.0f,
                                 5.0f, -6.0f, 7.0f, -8.0f};
    float dynamic_data[8] = {0};
    unsigned char *resident_data = NULL;
    attention_graph_fixture fixture;
    yvex_backend_attention_job job;
    yvex_backend_cuda_attention_graph_entry entry;
    yvex_backend_cuda_attention_graph_summary summary;
    yvex_backend_cuda_graph_info graph_info;
    yvex_backend_memory_stats memory_before;
    yvex_backend_memory_stats memory_after;
    yvex_cuda_backend_state *state;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *output = NULL;
    yvex_device_tensor *resident = NULL;
    yvex_device_tensor *workspace = NULL;
    yvex_error err;
    char first_key[160];
    char full_key[160];
    char dynamic_key[160];
    char canonical_graph_identity[YVEX_SHA256_HEX_BYTES];
    char bundle_identity[YVEX_SHA256_HEX_BYTES];
    static const char *const capture_faults[] = {
        "stream-create", "capture-begin", "capture-end", "inventory",
        "exec-identity", "instantiate", "upload"
    };
    unsigned long long count = 0ull;
    unsigned long long failed_synchronize_count = 0ull;
    unsigned long long configuration_hits;
    unsigned int configuration_count;
    size_t fault_index;
    unsigned int stage;
    int rc;

    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, (yvex_backend_cuda_graph_registry_action)-1, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_INVALID_ARG && count == 0ull,
                     "negative registry action refuses without invalidation");
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_PREFILL,
        YVEX_BACKEND_CUDA_ATTENTION_EAGER, "attention-config-eager-v1",
        "not-applicable", 0ull, 0ull, 0ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "configure eager CUDA attention mode");
    rc = yvex_backend_cuda_attention_graph_summary_get(backend, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.configured &&
                     summary.selected_mode == YVEX_BACKEND_CUDA_ATTENTION_EAGER &&
                     summary.graph_count == 0ull,
                     "eager attention mode reports no fabricated graphs");
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_DECODE,
        YVEX_BACKEND_CUDA_ATTENTION_FULL, "attention-config-full-v1",
        "decode-1", 4ull, 1ull, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "full graph mode refuses without stable residency and workspace");

    make_desc(&desc, "attention_resident");
    rc = backend->vtable->resident_alloc(
        backend, &desc, &resident, &resident_data, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate stable managed resident range");
    memset(resident_data, 0, (size_t)desc.bytes);
    make_desc(&desc, "attention_workspace");
    rc = yvex_backend_tensor_alloc(backend, &desc, &workspace, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate stable workspace range");
    rc = yvex_backend_resident_attach(backend, resident_data, desc.bytes,
                                      resident, 7ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "attach stable attention residency");
    rc = yvex_backend_workspace_attach(backend, workspace, 9ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "attach stable attention workspace");
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_PREFILL,
        YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE, "attention-config-piecewise-v1",
        "prefill-4", 4ull, 1ull, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "configure piecewise CUDA attention mode");
    rc = yvex_backend_cuda_attention_graph_summary_get(backend, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.configured &&
                     summary.selected_mode == YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE &&
                     strcmp(summary.compatibility_identity,
                            "attention-config-piecewise-v1") == 0 &&
                     strcmp(summary.capture_bucket, "prefill-4") == 0 &&
                     summary.driver_version > 0 &&
                     yvex_sha256_hex_valid(summary.cuda_build_identity) &&
                     summary.graph_count == 0ull && summary.capture_count == 0ull,
                     "piecewise mode reports configuration without fabricated capture evidence");
    make_desc(&desc, "attention_graph_input");
    rc = yvex_backend_tensor_alloc(backend, &desc, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate captured attention fixture input");
    make_desc(&desc, "attention_graph_output");
    rc = yvex_backend_tensor_alloc(backend, &desc, &output, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "allocate captured attention fixture output");
    rc = yvex_backend_tensor_write(backend, input, input_data, sizeof(input_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "admit captured attention fixture input");
    rc = yvex_backend_host_workspace_prepare_owned(backend, 64ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "prepare pinned capture fixture staging");
    memset(&fixture, 0, sizeof(fixture));
    fixture.backend = backend;
    fixture.input = input;
    fixture.output = output;
    fixture.position = 7ull;
    fixture.work.backend = backend;
    fixture.work.state = yvex_cuda_state(backend);
    fixture.work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_acquire(backend, 32ull, 16ull,
                                            (void **)&fixture.staged_input) ==
                YVEX_BACKEND_RESIDENT_HIT &&
            yvex_backend_host_workspace_acquire(backend, 32ull, 16ull,
                                                (void **)&fixture.staged_output) ==
                YVEX_BACKEND_RESIDENT_HIT,
        "bind stable pinned input and output staging spans");
    memcpy(fixture.staged_input, input_data, sizeof(input_data));
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:empty-capture-v1",
        NULL, enqueue_empty_fixture, NULL, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED,
                     "production registry rejects an empty Driver capture");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 0ull,
                     "empty capture leaves no registry ownership");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_ATTENTION_FAILURE", "copy-output", 1) == 0,
                     "inject captured output-copy failure");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:failed-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "captured output-copy fault refuses complete graph admission");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_ATTENTION_FAILURE") == 0,
                     "clear captured output-copy failure");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 0ull,
                     "failed capture leaves no partial registry ownership");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_ATTENTION_FAILURE", "copy-output", 1) == 0 &&
                         setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "graph-destroy", 1) == 0,
                     "inject enqueue failure before abandoned-graph destroy");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:abort-destroy-retry-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "abandoned-graph destroy failure is typed");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_ATTENTION_FAILURE") == 0,
                     "clear enqueue failure before retained cleanup retry");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "failed abandoned-graph destroy retains one registry owner");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_FAILED &&
                         entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
                     "retained abandoned graph exposes cleanup failure state");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear abandoned-graph destroy failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "retained abandoned graph releases exactly once after retry");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 0ull,
                     "abandoned-graph cleanup retry empties the registry");
    for (fault_index = 0u;
         fault_index < sizeof(capture_faults) / sizeof(capture_faults[0]);
         ++fault_index) {
        YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE",
                                capture_faults[fault_index], 1) == 0,
                         "inject partial graph preparation failure");
        rc = yvex_cuda_graph_execute(
            backend, "attention-config-piecewise-v1:retryable-fault-v1",
            NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
        YVEX_TEST_ASSERT(rc != YVEX_OK, "partial graph preparation failure is typed");
        rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK && count == 0ull,
                         "partial graph preparation leaves no registry entry");
        YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                         "clear partial graph preparation failure");
        rc = yvex_cuda_graph_execute(
            backend, "attention-config-piecewise-v1:retryable-fault-v1",
            NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "graph preparation retry succeeds");
        rc = yvex_backend_cuda_attention_graph_registry_apply(
            backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE, &count, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                         "retry graph ownership releases exactly once");
    }
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.inventory.kernel_node_count > 0ull &&
                     graph_info.inventory.memcpy_node_count >= 2ull &&
                     graph_info.inventory.memset_node_count > 0ull &&
                     graph_info.capture_elapsed_ns > 0ull &&
                     graph_info.instantiate_elapsed_ns > 0ull &&
                     graph_info.last_replay_elapsed_ns > 0ull,
                     "captured attention unit inventories transfers, memset, kernel, and timings");
    memcpy(canonical_graph_identity, graph_info.launch_graph_identity,
           sizeof(canonical_graph_identity));
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "attention registry counts the real captured unit");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !entry.update_requested &&
                     entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED &&
                     entry.graph.inventory.memcpy_node_count >= 2ull &&
                     entry.graph.inventory.memset_node_count > 0ull,
                     "attention registry inspection returns exact graph inventory");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 1ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS,
                     "attention registry inspection refuses missing entry");
    rc = yvex_backend_get_memory_stats(backend, &memory_before, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "snapshot memory events before warm graph replay");
    fixture.position = 8ull;
    rc = yvex_backend_op_rope(backend, input, fixture.position, 10000.0f, output, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "execute advanced-position eager reference");
    rc = yvex_backend_tensor_read(backend, output, dynamic_data, sizeof(dynamic_data), &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "read advanced-position eager reference");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.capture_count == 1ull &&
                     graph_info.instantiate_count == 1ull &&
                     graph_info.upload_count == 1ull && graph_info.replay_count == 2ull,
                     "warm graph replay performs no capture, instantiate, or upload");
    YVEX_TEST_ASSERT(memcmp(dynamic_data, fixture.staged_output, sizeof(dynamic_data)) == 0,
                     "warm replay updates dynamic kernel parameters without recapture");
    rc = yvex_backend_get_memory_stats(backend, &memory_after, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                     memory_after.allocation_events == memory_before.allocation_events &&
                     memory_after.release_events == memory_before.release_events,
                     "warm graph replay performs zero backend allocation or release events");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "request an injected incompatible production update");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "update", 1) == 0,
                     "inject CUDA graph-exec update incompatibility");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED,
                     "production registry preserves the admitted graph after update refusal");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && entry.update_requested &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED &&
                         entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_UPDATE_INCOMPATIBLE,
                     "registry exposes retryable update refusal without replacing the executable");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear CUDA graph-exec update incompatibility");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "request compatible update for every attention graph");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.update_count == 1ull &&
                     graph_info.replay_count == 3ull &&
                     graph_info.last_update_elapsed_ns > 0ull,
                     "next production execution performs real compatible graph-exec update");
    rc = yvex_backend_cuda_attention_graph_summary_get(backend, &summary, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && summary.graph_count == 1ull &&
                     summary.update_count == 1ull && summary.update_pending_count == 0ull &&
                     summary.memcpy_node_count >= 2ull && summary.memset_node_count > 0ull,
                     "attention summary aggregates updated transfer-complete graph evidence");
    state = yvex_cuda_state(backend);
    configuration_count = state->attention_configuration_count;
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_DECODE,
        YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE, "attention-config-piecewise-v1",
        "decode-1", 8ull, 2ull, 2ull, &err);
    if (rc == YVEX_OK)
        rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && count == 1ull &&
            state->attention_configuration_count == configuration_count + 1u &&
            state->attention_configurations[
                state->attention_active_configuration].phase ==
                YVEX_BACKEND_ATTENTION_PHASE_DECODE,
        "shape reconfiguration retains compatible graph ownership by phase");
    YVEX_TEST_ASSERT(
        yvex_cuda_attention_configuration_active(
            state, YVEX_BACKEND_ATTENTION_PHASE_PREFILL) == NULL,
        "an active decode shape cannot satisfy a prefill dispatch");
    configuration_count = state->attention_configuration_count;
    configuration_hits = state->attention_configuration_hits;
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_PREFILL,
        YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE, "attention-config-piecewise-v1",
        "prefill-4", 4ull, 1ull, 1ull, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK &&
            state->attention_configuration_count == configuration_count &&
            state->attention_configuration_hits == configuration_hits + 1ull &&
            state->attention_configurations[
                state->attention_active_configuration].phase ==
                YVEX_BACKEND_ATTENTION_PHASE_PREFILL,
        "returning to an admitted phase selects its immutable capacity record");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "request graph update under permuted Driver node enumeration");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_NODE_ORDER", "reverse", 1) == 0,
                     "permute captured Driver node enumeration");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_NODE_ORDER") == 0,
                     "clear captured Driver node permutation");
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.update_count == 2ull &&
                     graph_info.replay_count == 4ull &&
                     strcmp(graph_info.launch_graph_identity,
                            canonical_graph_identity) == 0,
                     "canonical graph identity and replay survive Driver node permutation");
    fixture.replay_function = yvex_cuda_state(backend)->rms_norm_f32_function;
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    fixture.replay_function = NULL;
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "replay refuses a mutated captured kernel function before launch");
    memcpy(bundle_identity, yvex_cuda_state(backend)->kernel_bundle_identity,
           sizeof(bundle_identity));
    yvex_cuda_state(backend)->kernel_bundle_identity[0] =
        bundle_identity[0] == '0' ? '1' : '0';
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "request update under mutated kernel bundle identity");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.update_count == 3ull &&
                     graph_info.replay_count == 5ull &&
                     strcmp(graph_info.launch_graph_identity,
                            canonical_graph_identity) != 0,
                     "launch graph identity binds the generated kernel bundle");
    memcpy(yvex_cuda_state(backend)->kernel_bundle_identity, bundle_identity,
           sizeof(bundle_identity));
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "request update after restoring admitted kernel bundle identity");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.update_count == 4ull &&
                     graph_info.replay_count == 6ull &&
                     strcmp(graph_info.launch_graph_identity,
                            canonical_graph_identity) == 0,
                     "restored bundle and launch schedule restore canonical graph identity");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "launch", 1) == 0,
                     "inject production registry launch failure");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "production registry launch failure is typed");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_FAILED &&
                         entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_LAUNCH_FAILED,
                     "registry exposes failed launch without a successful replay increment");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear production registry launch failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "invalidate launch-failed graph before recapture");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 1ull,
                     "invalidation retains one non-executable registry owner for safe recapture");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_INVALIDATED &&
                         entry.graph.graph_exec_identity[0] == '\0',
                     "invalidated registry owner exposes no executable graph identity");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "recapture production graph after launch failure");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "synchronize", 1) == 0,
                     "inject graph replay synchronization failure");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "graph replay synchronization failure is typed");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear graph replay synchronization failure");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_FAILED &&
                         entry.graph.reason ==
                             YVEX_BACKEND_CUDA_GRAPH_REASON_SYNCHRONIZE_FAILED,
                     "failed synchronization poisons the cached executable");
    failed_synchronize_count = entry.graph.synchronize_count;
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "poisoned executable refuses stale replay before quiescence");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "quiesce", 1) == 0,
                     "inject cleanup quiescence failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "failed quiescence preserves poisoned cleanup ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear cleanup quiescence failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "retry quiesces and invalidates failed replay");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_INVALIDATED &&
                         entry.graph.synchronize_count == failed_synchronize_count + 1ull,
                     "cleanup proves stream quiescence before handle destruction");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "quiesced invalidation permits safe graph recapture and reuse");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:atomic-peer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "capture peer graph for atomic invalidation proof");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "exec-destroy", 1) == 0,
                     "inject registry-wide executable cleanup failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "registry-wide cleanup failure is typed after atomic poisoning");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_FAILED &&
                         entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
                     "first registry graph remains poisoned after cleanup failure");
    rc = yvex_backend_cuda_attention_graph_registry_get(backend, 1ull, &entry, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK &&
                         entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_FAILED &&
                         entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
                     "peer registry graph is poisoned before fallible cleanup");
    rc = yvex_cuda_graph_execute(
        backend, "attention-config-piecewise-v1:unit-transfer-v1",
        NULL, enqueue_attention_fixture, &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "cleanup-failed registry graph refuses stale replay");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear registry-wide executable cleanup failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "retry completes poisoned registry invalidation");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "stream-destroy", 1) == 0,
                     "inject attention registry stream cleanup failure");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && count == 0ull,
                     "registry release failure preserves exact retry ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear attention registry stream cleanup failure");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 2ull,
                     "failed registry cleanup keeps its entry discoverable");
    rc = yvex_backend_cuda_attention_graph_registry_apply(
        backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 2ull,
                     "release removes exact attention registry ownership");
    rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && count == 0ull,
                     "released attention registry is empty");
    rc = yvex_backend_host_workspace_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "detach captured pinned staging workspace");
    rc = yvex_backend_tensor_release(backend, &output, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "release captured attention fixture output");
    rc = yvex_backend_tensor_release(backend, &input, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "release captured attention fixture input");
    memset(&job, 0, sizeof(job));
    job.phase = YVEX_BACKEND_ATTENTION_PHASE_PREFILL;
    for (stage = 0u; stage < YVEX_CUDA_ATTENTION_STAGE_COUNT; ++stage) {
        char piece_key[160];

        rc = yvex_cuda_attention_graph_key(
            backend, &job, stage, stage + 1u, piece_key, &err);
        YVEX_TEST_ASSERT(rc == YVEX_OK && piece_key[0] != '\0',
                         "every canonical attention stage interval is admitted");
        if (stage == 0u)
            memcpy(first_key, piece_key, sizeof(first_key));
        else
            YVEX_TEST_ASSERT(strcmp(first_key, piece_key) != 0,
                             "piecewise stage intervals have distinct graph keys");
    }
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, full_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && full_key[0] != '\0' &&
                     strcmp(first_key, full_key) != 0,
                     "full canonical attention stage interval is admitted distinctly");
    backend->state_residency_generation = 7ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(full_key, dynamic_key) == 0,
                     "persistent-state generation preserves allocation-stable graph compatibility");
    backend->state_residency_generation = 0ull;
    job.token_position = 3ull;
    job.local_count = 3ull;
    job.compressed_count = 1ull;
    job.indexer_count = 1ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(full_key, dynamic_key) == 0,
                     "advancing history preserves one topology-capacity graph key");
    job.compression_ratio = 4ull;
    job.token_count = 1ull;
    job.token_position = 1ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, full_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "compression phase enters graph compatibility");
    job.token_position = 5ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(full_key, dynamic_key) == 0,
                     "equivalent compression phases reuse one launch topology");
    job.token_position = 2ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(full_key, dynamic_key) == 0,
                     "non-emitting compression phases reuse one launch topology");
    job.token_position = 3ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(full_key, dynamic_key) == 0,
                     "captured compressor predicates emission without changing topology");
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COMPRESS, dynamic_key, &err);
    job.token_position = 1ull;
    rc = rc == YVEX_OK
             ? yvex_cuda_attention_graph_key(
                   backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COMPRESS,
                   first_key, &err)
             : rc;
    YVEX_TEST_ASSERT(rc == YVEX_OK && strcmp(first_key, dynamic_key) == 0,
                     "non-compression pieces ignore emission topology");
    job.local_count = 5ull;
    job.sliding_window = 5ull;
    job.candidate_block_visible = 1;
    YVEX_TEST_ASSERT(yvex_cuda_attention_local_capacity(
                         &state->attention_configurations[state->attention_active_configuration],
                         &job, 0) == 5ull && yvex_cuda_attention_local_capacity(
                         &state->attention_configurations[state->attention_active_configuration],
                         &job, 1) == 4ull,
                     "execution storage and published target state retain distinct capacities");
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "candidate-visible history admits the reserved transaction slot");
    job.candidate_block_visible = 0;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "full initial ring admits phase-local target trimming");
    job.candidate_block_visible = 1;
    job.local_count = 6ull;
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT, dynamic_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS,
                     "candidate-visible history beyond the sliding window refuses");
    memset(&job, 0, sizeof(job));
    rc = yvex_cuda_attention_graph_key(
        backend, &job, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT + 1u, full_key, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_STATE,
                     "attention graph interval beyond canonical stage count refuses");
    rc = yvex_backend_cuda_attention_configure(
        backend, YVEX_BACKEND_ATTENTION_PHASE_DECODE,
        YVEX_BACKEND_CUDA_ATTENTION_FULL, "attention-config-full-v2",
        "decode-1", 4ull, 1ull, 1ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK,
                     "new compatibility identity invalidates before full mode configuration");
    yvex_backend_workspace_detach(backend);
    YVEX_TEST_ASSERT(yvex_backend_resident_detach(backend, &err) == YVEX_OK,
                     "detach stable attention residency");
    rc = yvex_backend_tensor_release(backend, &workspace, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "release stable workspace range");
    rc = yvex_backend_tensor_release(backend, &resident, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "release stable resident range");
    return 0;
}

static int test_host_workspace_lifecycle(yvex_backend *backend)
{
    yvex_backend_host_workspace_summary summary;
    void *first = NULL;
    void *second = NULL;
    yvex_error err;
    int rc;

    rc = yvex_backend_host_workspace_prepare_owned(backend, 257ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "prepare owned host staging arena");
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_summary_get(backend, &summary) &&
            summary.attached && summary.owned && summary.pinned &&
            summary.capacity == 257ull && summary.used == 0ull &&
            summary.peak == 0ull && summary.generation == 1ull &&
            summary.allocation_count == 1ull,
        "owned host arena exposes exact cold preparation facts");
    rc = yvex_backend_host_workspace_acquire(backend, 32ull, 16ull, &first);
    YVEX_TEST_ASSERT(rc == YVEX_BACKEND_RESIDENT_HIT && first,
                     "acquire aligned host staging range");
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_acquire(backend, 257ull, 1ull,
                                            &second) ==
                YVEX_BACKEND_RESIDENT_MISS &&
            !second,
        "host staging refuses capacity overflow without resizing");
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_acquire(backend, 1ull, 3ull, &second) ==
                YVEX_BACKEND_RESIDENT_INVALID &&
            !second,
        "host staging refuses non-power-of-two alignment");
    backend_host_workspace_reset(backend);
    rc = yvex_backend_host_workspace_acquire(backend, 257ull, 1ull, &second);
    YVEX_TEST_ASSERT(rc == YVEX_BACKEND_RESIDENT_HIT && second == first,
                     "warm rewind reuses the exact owned address");
    backend_host_workspace_reset(backend);
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_summary_get(backend, &summary) &&
            summary.used == 0ull && summary.peak == 257ull &&
            summary.allocation_count == 1ull,
        "two acquisitions preserve one allocation and exact peak evidence");
    rc = yvex_backend_host_workspace_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "detach owned host staging arena");
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_summary_get(backend, &summary) &&
            !summary.attached && summary.capacity == 0ull &&
            summary.allocation_count == 0ull,
        "detach releases owned storage and clears every arena fact");

    rc = yvex_backend_host_workspace_prepare_owned(backend, 128ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "prepare standalone owned host staging arena");
    YVEX_TEST_ASSERT(
        yvex_backend_host_workspace_summary_get(backend, &summary) &&
            summary.attached && summary.owned && summary.pinned &&
            summary.capacity == 128ull &&
            summary.allocation_count == 1ull,
        "standalone cold arena reports its single owned allocation");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "host-workspace", 1) == 0,
                     "inject page-locked host cleanup failure");
    rc = yvex_backend_host_workspace_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND,
                     "page-locked cleanup failure is typed after physical release");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear page-locked host cleanup failure");
    rc = yvex_backend_host_workspace_detach(backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "page-locked host cleanup retry is idempotent");
    return 0;
}

/* Prove checked backend close retains registry graph ownership across a stream-release fault. */
static int test_backend_graph_close_retry(void)
{
    yvex_backend_cuda_attention_graph_entry entry;
    yvex_backend_cuda_graph_info graph_info;
    yvex_backend_tensor_desc desc;
    yvex_device_tensor *input = NULL;
    yvex_device_tensor *output = NULL;
    yvex_backend *backend = NULL;
    rope_graph_fixture fixture;
    yvex_error err;
    int rc = open_cuda(&backend);

    if (rc != 0) return rc;
    make_desc(&desc, "close_retry_input");
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &desc, &input, &err) == YVEX_OK,
                     "allocate production graph close-retry input");
    make_desc(&desc, "close_retry_output");
    YVEX_TEST_ASSERT(yvex_backend_tensor_alloc(backend, &desc, &output, &err) == YVEX_OK,
                     "allocate production graph close-retry output");
    YVEX_TEST_ASSERT(
        yvex_backend_cuda_attention_configure(
            backend, YVEX_BACKEND_ATTENTION_PHASE_DECODE,
            YVEX_BACKEND_CUDA_ATTENTION_EAGER, "cuda-close-retry-v1",
            "not-applicable", 0ull, 0ull, 0ull, &err) == YVEX_OK,
        "configure registry identity for checked backend close retry");
    fixture = (rope_graph_fixture){
        .backend = backend, .input = input, .output = output, .position = 7ull
    };
    rc = yvex_cuda_graph_execute(
        backend, "cuda-close-retry-v1:rope", NULL, enqueue_rope_fixture,
        &fixture, 1, &graph_info, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && graph_info.inventory.kernel_node_count == 1ull,
                     "capture real production kernel through the registry");
    YVEX_TEST_ASSERT(
        yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err) == YVEX_OK &&
            entry.graph.state == YVEX_BACKEND_CUDA_GRAPH_INSTANTIATED,
        "production registry owns the graph before checked close");
    YVEX_TEST_ASSERT(yvex_backend_tensor_release(backend, &output, &err) == YVEX_OK &&
                         yvex_backend_tensor_release(backend, &input, &err) == YVEX_OK,
                     "release graph fixture tensors before backend close");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_GRAPH_FAILURE", "stream-destroy", 1) == 0,
                     "inject checked backend graph cleanup failure");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_BACKEND && backend &&
            yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, &err) == YVEX_OK &&
            entry.graph.reason == YVEX_BACKEND_CUDA_GRAPH_REASON_CLEANUP_FAILED,
        "checked backend close preserves registry graph ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_GRAPH_FAILURE") == 0,
                     "clear checked backend graph cleanup failure");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !backend,
                     "checked backend close retry releases the registry graph once");
    return 0;
}

/* Prove checked backend close retains pinned host ownership before physical release. */
static int test_backend_host_close_retry(void)
{
    yvex_backend_host_workspace_summary summary;
    yvex_backend *backend = NULL;
    yvex_error err;
    int rc = open_cuda(&backend);

    if (rc != 0) return rc;
    YVEX_TEST_ASSERT(yvex_backend_host_workspace_prepare_owned(backend, 128ull, &err) == YVEX_OK,
                     "prepare pinned workspace for checked backend close retry");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE",
                            "host-workspace-pre-release", 1) == 0,
                     "inject checked backend pinned-host pre-release failure");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && backend &&
                         yvex_backend_host_workspace_summary_get(
                             backend, &summary) &&
                         summary.attached && summary.owned && summary.capacity == 128ull,
                     "checked backend close retains the unreleased pinned workspace");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear checked backend pinned-host pre-release failure");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&backend, &err) == YVEX_OK && !backend,
                     "checked backend close retry releases pinned ownership once");
    return 0;
}

static int test_backend_detach(void)
{
    yvex_backend *backend = NULL;
    yvex_error err;
    int rc = open_cuda(&backend);

    if (rc != 0) return rc;
    rc = yvex_backend_host_workspace_prepare_owned(backend, 128ull, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "prepare owned staging for checked backend cleanup");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_CLEANUP_FAILURE", "host-workspace", 1) == 0,
                     "inject checked backend host cleanup failure");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && !backend,
                     "checked backend close reports cleanup failure and nulls ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_CLEANUP_FAILURE") == 0,
                     "clear checked backend host cleanup failure");
    return 0;
}

/* Prove session-stream creation, scoped completion, and checked cleanup refusal. */
static int test_execution_stream_lifecycle(void)
{
    yvex_backend_options options = {0};
    yvex_backend *backend = NULL;
    yvex_error err;
    int device_wide = 1;
    int rc;
    options.kind = YVEX_BACKEND_KIND_CUDA;
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_STREAM_FAILURE", "create", 1) == 0,
                     "inject CUDA execution stream creation failure");
    rc = yvex_backend_open(&backend, &options, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && !backend,
                     "execution stream creation failure rolls backend open back");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_STREAM_FAILURE") == 0,
                     "clear CUDA execution stream creation failure");
    YVEX_TEST_ASSERT(yvex_backend_open(&backend, &options, &err) == YVEX_OK && backend,
                     "open CUDA backend with one execution stream");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_SYNC_FAILURE", "encoded-attention", 1) == 0,
                     "inject launch-stream completion failure");
    rc = yvex_cuda_launch_synchronize(
        backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
        "cuda.test.execution-stream", &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && device_wide == 0 &&
                         strstr(yvex_error_message(&err), "synchronization failure") != NULL,
                     "launch-stream completion fails closed without a context-wide barrier");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_SYNC_FAILURE") == 0,
                     "clear launch-stream completion failure");
    YVEX_TEST_ASSERT(setenv("YVEX_TEST_CUDA_STREAM_FAILURE", "destroy", 1) == 0,
                     "inject CUDA execution stream cleanup failure");
    rc = yvex_backend_close_checked(&backend, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BACKEND && backend,
                     "checked close retains failed execution stream ownership");
    YVEX_TEST_ASSERT(unsetenv("YVEX_TEST_CUDA_STREAM_FAILURE") == 0,
                     "clear CUDA execution stream cleanup failure");
    YVEX_TEST_ASSERT(yvex_backend_close_checked(&backend, &err) == YVEX_OK && !backend,
                     "checked close retry releases the execution stream once");
    return 0;
}

static int test_attention_failure_preserves_cause(void)
{
    const yvex_cuda_attention_operations *operations =
        yvex_cuda_attention_operations_get();
    yvex_backend_attention_failure failure = {0};
    yvex_error err;
    int rc;

    yvex_error_set(&err, YVEX_ERR_BOUNDS, "cuda.attention.specific",
                   "specific attention capacity was exceeded");
    rc = operations->fail(
        &failure, YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE,
        "cuda.attention.wrapper", 7ull, 5ull, &err, YVEX_ERR_BOUNDS,
        "generic attention completion failed");
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_BOUNDS &&
            failure.code == YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE &&
            failure.expected == 7ull && failure.actual == 5ull &&
            strcmp(yvex_error_where(&err), "cuda.attention.specific") == 0 &&
            strstr(yvex_error_message(&err), "specific attention capacity") != NULL,
        "attention failure facts preserve the originating CUDA cause");
    return 0;
}

int yvex_cuda_test_graph(void)
{
    yvex_backend *backend = NULL;
    int rc = open_cuda(&backend);

    if (rc != 0) return rc;
    YVEX_TEST_ASSERT(test_attention_piece_inventory() == 0,
                     "CUDA attention piecewise semantic inventory");
    YVEX_TEST_ASSERT(test_capability_projection(backend) == 0,
                     "CUDA graph capability projection");
    YVEX_TEST_ASSERT(test_host_workspace_lifecycle(backend) == 0,
                     "CUDA host workspace lifecycle");
    YVEX_TEST_ASSERT(test_attention_graph_configuration(backend) == 0,
                     "CUDA attention graph mode configuration");
    YVEX_TEST_ASSERT(test_rolling_cursor_update(backend) == 0,
                     "CUDA rolling cursor replay update");
    YVEX_TEST_ASSERT(test_shared_graph_completion(backend) == 0,
                     "CUDA shared-stream graph completion");
    YVEX_TEST_ASSERT(test_graph_prepare_lifecycle(backend) == 0,
                     "CUDA graph dynamic preamble lifecycle");
    rc = yvex_backend_close_checked(&backend, NULL);
    YVEX_TEST_ASSERT(rc == YVEX_OK && !backend, "checked CUDA backend cleanup");
    rc = test_deferred_raw_release();
    if (rc != 0)
        return rc;
    rc = test_module_release_retry();
    if (rc != 0)
        return rc;
    rc = test_module_rollback_retry();
    if (rc != 0)
        return rc;
    rc = test_shared_open_rollback_retry();
    if (rc != 0)
        return rc;
    rc = test_shared_context_close_order();
    if (rc != 0)
        return rc;
    rc = test_backend_graph_close_retry();
    if (rc != 0)
        return rc;
    rc = test_backend_host_close_retry();
    if (rc != 0)
        return rc;
    rc = test_execution_stream_lifecycle();
    if (rc != 0)
        return rc;
    rc = test_attention_failure_preserves_cause();
    if (rc != 0)
        return rc;
    return test_backend_detach();
}
