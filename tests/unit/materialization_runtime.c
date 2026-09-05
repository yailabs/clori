/*
 * Uses only tiny GGUF fixtures and synthetic complete-admission facts bound to the opened
 * fixture snapshot. Fixture materialization is not full runtime support.
 */
#include <yvex/internal/artifact.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/runtime.h>

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/api.h>

#include "tests/test.h"

typedef struct {
    unsigned long long tensor_id, bytes;
    const unsigned char *data;
    unsigned long long note_calls, note_bytes, detach_calls;
} fixture_resident_span;
typedef struct {
    yvex_materialization_session *session;
    const yvex_materialized_tensor_binding *binding;
    unsigned long long iterations;
    int failed;
} fixture_resident_read_thread;
typedef struct {
    int mismatch;
} fixture_terminal_projection;

static int fixture_terminal_find(
    const void *context, const char *name, yvex_materialization_terminal *out)
{
    const fixture_terminal_projection *fixture =
        (const fixture_terminal_projection *)context;

    if (!fixture || !name || !out || strcmp(name, "token_embd.weight") != 0) return 0;
    *out = (yvex_materialization_terminal){
        0ull,
        fixture->mismatch ? YVEX_TENSOR_ROLE_OUTPUT_HEAD : YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
        YVEX_TENSOR_COLLECTION_GLOBAL, YVEX_TENSOR_SCOPE_GLOBAL,
        YVEX_MATERIALIZATION_NO_INDEX, YVEX_MATERIALIZATION_NO_INDEX, 0ull};
    return 1;
}

static int fixture_resident_resolve(const void *context,
                                    const yvex_materialized_tensor_binding *binding,
                                    const unsigned char **data, unsigned long long *bytes)
{
    const fixture_resident_span *span = (const fixture_resident_span *)context;

    if (!span || !binding || binding->tensor_id != span->tensor_id)
        return YVEX_MATERIALIZATION_READ_MISS;
    *data = span->data;
    *bytes = span->bytes;
    return YVEX_MATERIALIZATION_READ_HIT;
}

static int fixture_resident_note(const void *context, unsigned long long bytes)
{
    fixture_resident_span *span = (fixture_resident_span *)context;

    if (!span || span->note_calls == ULLONG_MAX || span->note_bytes > ULLONG_MAX - bytes)
        return 0;
    span->note_calls++;
    span->note_bytes += bytes;
    return 1;
}

static void fixture_resident_detached(const void *context)
{
    fixture_resident_span *span = (fixture_resident_span *)context;

    if (span) span->detach_calls++;
}

static void *fixture_resident_read_worker(void *context)
{
    fixture_resident_read_thread *thread = (fixture_resident_read_thread *)context;
    unsigned char output[16];
    unsigned long long index;

    for (index = 0ull; index < thread->iterations; ++index) {
        yvex_materialization_failure failure;
        yvex_error err;
        unsigned long long offset = index % (thread->binding->encoded_bytes - sizeof(output) + 1ull);

        if (yvex_materialization_session_read(thread->session, thread->binding, offset,
                                              output, sizeof(output), &failure, &err) != YVEX_OK ||
            output[0] != 0x5a || output[sizeof(output) - 1u] != 0x5a) {
            thread->failed = 1;
            break;
        }
    }
    return NULL;
}

static void fill_fixture_admission(const yvex_artifact *artifact,
                                   const yvex_tensor_table *tensors,
                                   yvex_complete_artifact_admission *admission)
{
    memset(admission, 0, sizeof(*admission));
    admission->artifact_class = YVEX_ARTIFACT_CLASS_COMPLETE_YVEX;
    admission->tensor_count = yvex_tensor_table_count(tensors);
    admission->file_bytes = yvex_artifact_size(artifact);
    admission->payload_bytes = 128ull;
    admission->materialization_input_ready = 1;
    admission->tokenizer_complete = 1;
    admission->complete = 1;
    admission->runtime_supported = 0;
    (void)snprintf(admission->artifact_path,
                   sizeof(admission->artifact_path), "%s",
                   yvex_artifact_path(artifact));
    (void)snprintf(admission->artifact_identity,
                   sizeof(admission->artifact_identity),
                   "1111111111111111111111111111111111111111111111111111111111111111");
    (void)snprintf(admission->profile_identity,
                   sizeof(admission->profile_identity),
                   "fixture-profile");
    (void)snprintf(admission->writer_plan_identity,
                   sizeof(admission->writer_plan_identity),
                   "fixture-writer-plan");
    (void)yvex_artifact_snapshot_get(artifact, &admission->file_snapshot, NULL);
}

static int open_fixture(yvex_artifact **artifact,
                        yvex_gguf **gguf,
                        yvex_tensor_table **tensors)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = "tests/fixtures/gguf/valid-tokenizer-simple.gguf";
    options.readonly = 1;
    options.map = 0;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(artifact, &options, &err);
    if (rc == YVEX_OK) rc = yvex_gguf_open(gguf, *artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(tensors, *gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "fixture open failed: %s: %s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        yvex_tensor_table_close(*tensors);
        yvex_gguf_close(*gguf);
        yvex_artifact_close(*artifact);
        return 1;
    }
    return 0;
}

static int test_materialization_fixture(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_materialization_options options;
    fixture_terminal_projection projection_context = {0};
    yvex_materialization_projection projection = {
        YVEX_MATERIALIZATION_PROJECTION_SCHEMA_VERSION, 1ull, 1ull,
        &projection_context, fixture_terminal_find, 1};
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure failure;
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_runtime_descriptor_family_facts family;
    const yvex_materialization_summary *summary;
    const yvex_runtime_descriptor_summary *descriptor_summary;
    const yvex_materialized_tensor_binding *binding;
    unsigned char bytes[16];
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture(&artifact, &gguf, &tensors) == 0,
                     "open materialization fixture");
    fill_fixture_admission(artifact, tensors, &admission);
    admission.mapping_identity = projection.mapping_identity;
    yvex_materialization_options_default(&options);
    options.max_chunk_bytes = 16u;

    rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, &projection, &options,
        &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "plan builds");
    summary = yvex_materialization_plan_summary(plan);
    YVEX_TEST_ASSERT(summary != NULL, "plan summary");
    YVEX_TEST_ASSERT(summary->status == YVEX_MATERIALIZATION_STATUS_PLANNED,
                     "plan status");
    YVEX_TEST_ASSERT(summary->tensor_count == 1ull, "plan tensor count");
    YVEX_TEST_ASSERT(summary->payload_bytes == 128ull, "plan payload bytes");
    YVEX_TEST_ASSERT(summary->file_backed_bytes_owned == 128ull,
                     "file backed bytes");
    YVEX_TEST_ASSERT(summary->execution_ready == 0, "execution not ready");
    YVEX_TEST_ASSERT(strlen(summary->plan_identity) == 64u,
                     "plan identity");

    rc = yvex_materialization_session_open(
        &session, plan, artifact, &options, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "session open");
    binding = yvex_materialization_session_tensor_at(session, 0ull);
    YVEX_TEST_ASSERT(binding != NULL && strcmp(binding->name, "token_embd.weight") == 0,
                     "session preserves the canonical binding order");
    YVEX_TEST_ASSERT(binding->encoded_bytes == 128ull, "binding bytes");
    YVEX_TEST_ASSERT(binding->access_mode == YVEX_MATERIALIZATION_ACCESS_FILE_RANGE ||
                         binding->access_mode ==
                             YVEX_MATERIALIZATION_ACCESS_BACKEND_CANDIDATE_FILE_RANGE,
                     "binding access");
    rc = yvex_materialization_session_commit(session, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "session commit");
    rc = yvex_materialization_session_read(
        session, binding, 0ull, bytes, sizeof(bytes), &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "session read");
    rc = yvex_materialization_session_walk_payload(
        session, NULL, NULL, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "session full walk");
    summary = yvex_materialization_session_summary(session);
    YVEX_TEST_ASSERT(summary->payload_bytes_accessed >= 144ull,
                     "payload access counted");
    YVEX_TEST_ASSERT(summary->peak_executor_owned_bytes == 16ull,
                     "bounded staging peak");

    memset(&family, 0, sizeof(family));
    family.logical_model_identity =
        "1111111111111111111111111111111111111111111111111111111111111111";
    family.runtime_numeric_identity =
        "2222222222222222222222222222222222222222222222222222222222222222";
    family.runtime_hadamard_revision =
        "Dao-AILab/fast-hadamard-transform:v1.1.0.post2:"
        "e7706faf8d1c3b9f241e36860640ad1dac644ede";
    rc = yvex_runtime_descriptor_build_projected(
        &descriptor, &admission, session, &family, &projection,
        &descriptor_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "runtime descriptor builds");
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    YVEX_TEST_ASSERT(descriptor_summary != NULL, "descriptor summary");
    YVEX_TEST_ASSERT(descriptor_summary->status == YVEX_RUNTIME_DESCRIPTOR_STATUS_READY,
                     "descriptor status");
    YVEX_TEST_ASSERT(descriptor_summary->tensor_count == 1ull,
                     "descriptor tensor count");
    YVEX_TEST_ASSERT(descriptor_summary->graph_execution_ready == 0,
                     "graph not ready");
    YVEX_TEST_ASSERT(descriptor_summary->generation_ready == 0,
                     "generation not ready");
    YVEX_TEST_ASSERT(
        strcmp(descriptor_summary->runtime_hadamard_revision,
               family.runtime_hadamard_revision) == 0,
        "runtime descriptor preserves the complete Hadamard authority commit");
    YVEX_TEST_ASSERT(yvex_runtime_descriptor_tensor_at(descriptor, 0ull) != NULL &&
                         strcmp(yvex_runtime_descriptor_tensor_at(descriptor, 0ull)->binding->name,
                                "token_embd.weight") == 0,
                     "descriptor preserves canonical tensor order");

    yvex_runtime_descriptor_close(descriptor);
    descriptor = NULL;
    projection_context.mismatch = 1;
    rc = yvex_runtime_descriptor_build_projected(
        &descriptor, &admission, session, &family, &projection,
        &descriptor_failure, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_FORMAT && !descriptor &&
            descriptor_failure.code == YVEX_RUNTIME_DESCRIPTOR_FAILURE_ARCHITECTURE,
        "runtime descriptor refuses a terminal projection that changes semantic role");
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_materialization_refusals(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_materialization_options options;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure failure;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture(&artifact, &gguf, &tensors) == 0,
                     "open refusal fixture");
    fill_fixture_admission(artifact, tensors, &admission);
    yvex_materialization_options_default(&options);
    admission.complete = 0;
    rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL, &options,
        &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "incomplete admission refused");
    YVEX_TEST_ASSERT(failure.code == YVEX_MATERIALIZATION_FAILURE_ADMISSION,
                     "admission failure code");
    fill_fixture_admission(artifact, tensors, &admission);
    admission.file_snapshot.size += 1ull;
    rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL, &options,
        &failure, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "snapshot drift refused");
    YVEX_TEST_ASSERT(failure.code == YVEX_MATERIALIZATION_FAILURE_SNAPSHOT_DRIFT,
                     "snapshot failure code");

    fill_fixture_admission(artifact, tensors, &admission);
    options.max_chunk_bytes = 8u;
    options.cancel_after_first_chunk = 1;
    rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL, &options,
        &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "plan for cancellation");
    rc = yvex_materialization_session_open(
        &session, plan, artifact, &options, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "session for cancellation");
    rc = yvex_materialization_session_commit(session, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "commit for cancellation");
    rc = yvex_materialization_session_walk_payload(
        session, NULL, NULL, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_CANCELLED, "walk cancelled");
    YVEX_TEST_ASSERT(failure.code == YVEX_MATERIALIZATION_FAILURE_CANCELLED,
                     "cancel failure code");
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

/* Prove one verified component can own a deterministic staged resident pack. */
static int test_component_residency(void)
{
    static const char component_identity[] =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_materialization_options materialization_options;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_residency *first = NULL, *second = NULL;
    yvex_runtime_residency_options options = {0};
    yvex_runtime_residency_failure failure;
    yvex_runtime_residency_summary first_summary, second_summary;
    yvex_backend_options backend_options = {0};
    yvex_backend *wrong_backend = NULL;
    yvex_materialization_access_summary before, after;
    const yvex_materialized_tensor_binding *binding;
    const unsigned char *arena = NULL, *resident = NULL, *borrowed = NULL;
    unsigned char expected[128];
    unsigned long long arena_bytes = 0ull, resident_bytes = 0ull;
    yvex_error err;
    int uploaded = 0, rc;

    YVEX_TEST_ASSERT(open_fixture(&artifact, &gguf, &tensors) == 0,
                     "open component residency fixture");
    fill_fixture_admission(artifact, tensors, &admission);
    yvex_materialization_options_default(&materialization_options);
    rc = yvex_materialization_plan_build(
        &plan, &admission, artifact, gguf, tensors, NULL,
        &materialization_options, &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &materialization_options,
            &materialization_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            session, &materialization_failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "component residency materialization committed");
    binding = yvex_materialization_session_tensor_at(session, 0ull);
    YVEX_TEST_ASSERT(binding && binding->encoded_bytes == 128ull,
                     "component residency binding is exact");
    YVEX_TEST_ASSERT(
        yvex_artifact_read_at(artifact, binding->absolute_offset, expected,
                              sizeof(expected), &err) == YVEX_OK,
        "component residency reference bytes are readable");
    YVEX_TEST_ASSERT(
        yvex_runtime_component_residency_prepare(
            &first, session, "mutable-component", &options, &failure, &err) ==
                YVEX_ERR_INVALID_ARG && !first &&
            failure.code == YVEX_RUNTIME_RESIDENCY_FAILURE_INVALID_ARGUMENT,
        "component residency refuses a mutable semantic identity");
    options.maximum_host_bytes = 127ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_residency_prepare(
            &first, session, component_identity, &options, &failure, &err) ==
                YVEX_ERR_NOMEM && !first &&
            failure.code == YVEX_RUNTIME_RESIDENCY_FAILURE_BUDGET,
        "component residency refuses a host budget below exact payload bytes");
    options.maximum_host_bytes = 128ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_component_residency_prepare(
            &first, session, component_identity, &options, &failure, &err) == YVEX_OK &&
            first,
        "component residency prepares within its exact host budget");
    YVEX_TEST_ASSERT(
        yvex_runtime_residency_snapshot(
            first, &first_summary, &arena, &arena_bytes, &err) == YVEX_OK &&
            first_summary.sealed && first_summary.attached && first_summary.host_ready &&
            first_summary.schema_version == YVEX_RUNTIME_RESIDENCY_SCHEMA_V7 &&
            first_summary.placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED &&
            first_summary.host_locked && first_summary.model_complete &&
            first_summary.binding_count == 1ull && first_summary.model_binding_count == 1ull &&
            first_summary.encoded_bytes == 128ull && !first_summary.mapped_package_bytes &&
            first_summary.prepared_bytes == 128ull && arena && arena_bytes == 128ull &&
            yvex_sha256_hex_valid(first_summary.payload_digest) &&
            yvex_sha256_hex_valid(first_summary.residency_identity),
        "component residency publishes complete deterministic host facts");
    backend_options.kind = YVEX_BACKEND_KIND_CPU;
    YVEX_TEST_ASSERT(
        yvex_backend_open(&wrong_backend, &backend_options, &err) == YVEX_OK &&
            yvex_runtime_residency_cuda_session_attach(
                first, &wrong_backend, 1ull, &uploaded, &second_summary, &err) ==
                YVEX_ERR_INVALID_ARG &&
            wrong_backend && !uploaded &&
            yvex_backend_close_checked(&wrong_backend, &err) == YVEX_OK && !wrong_backend,
        "component residency refuses a prepared non-CUDA owner without consuming it");
    YVEX_TEST_ASSERT(
        yvex_runtime_residency_binding_view(
            first, binding, &resident, &resident_bytes, &err) == YVEX_OK &&
            resident == arena && resident_bytes == binding->encoded_bytes &&
            memcmp(resident, expected, sizeof(expected)) == 0,
        "component residency resolves the exact encoded tensor span");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_access_summary(session, &before, &err) == YVEX_OK &&
            yvex_materialization_session_borrow(
                session, binding, 0ull, (size_t)binding->encoded_bytes,
                &borrowed, &materialization_failure, &err) == YVEX_OK &&
            borrowed == resident &&
            yvex_materialization_session_access_summary(session, &after, &err) == YVEX_OK &&
            after.artifact_read_calls == before.artifact_read_calls &&
            after.resident_read_calls == before.resident_read_calls + 1ull,
        "component materialization borrows resident bytes without reopening the artifact");
    YVEX_TEST_ASSERT(yvex_runtime_residency_close(&first, &err) == YVEX_OK && !first,
                     "component residency detaches and closes");
    YVEX_TEST_ASSERT(
        yvex_runtime_component_residency_prepare(
            &second, session, component_identity, &options, &failure, &err) == YVEX_OK &&
            yvex_runtime_residency_snapshot(
                second, &second_summary, NULL, NULL, &err) == YVEX_OK &&
            strcmp(second_summary.payload_digest, first_summary.payload_digest) == 0 &&
            strcmp(second_summary.residency_identity,
                   first_summary.residency_identity) == 0,
        "repeated component residency has byte-identical semantic identities");
    YVEX_TEST_ASSERT(
        yvex_runtime_residency_invalidate(second, &err) == YVEX_OK &&
            yvex_runtime_residency_binding_view(
                second, binding, &resident, &resident_bytes, &err) == YVEX_ERR_STATE &&
            yvex_runtime_residency_close(&second, &err) == YVEX_OK && !second,
        "component residency invalidation refuses subsequent tensor resolution");
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

/* Prove a sealed resident provider replaces only its exact tensor's physical reads. */
static int test_materialization_resident_provider(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_materialization_options options;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_failure failure;
    const yvex_materialized_tensor_binding *binding;
    yvex_materialization_access_summary before, after;
    yvex_materialization_read_provider provider;
    fixture_resident_span span;
    fixture_resident_read_thread thread_contexts[4];
    pthread_t threads[4];
    unsigned char resident[128], output[16];
    const unsigned char *borrowed = NULL;
    unsigned int created = 0u, index;
    yvex_error err;
    int wrong_owner;
    int rc;

    YVEX_TEST_ASSERT(open_fixture(&artifact, &gguf, &tensors) == 0,
                     "open resident provider fixture");
    fill_fixture_admission(artifact, tensors, &admission);
    yvex_materialization_options_default(&options);
    rc = yvex_materialization_plan_build(&plan, &admission, artifact, gguf, tensors,
                                         NULL, &options, &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(&session, plan, artifact, &options,
                                               &failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(session, &failure, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "resident provider materialization committed");
    binding = yvex_materialization_session_tensor_at(session, 0ull);
    YVEX_TEST_ASSERT(binding && binding->encoded_bytes == sizeof(resident),
                     "resident provider binding geometry");
    memset(resident, 0x5a, sizeof(resident));
    memset(&span, 0, sizeof(span));
    span.tensor_id = binding->tensor_id;
    span.bytes = binding->encoded_bytes;
    span.data = resident;
    provider.context = &span;
    provider.resolve = fixture_resident_resolve;
    provider.note_access = fixture_resident_note;
    provider.detached = fixture_resident_detached;
    YVEX_TEST_ASSERT(
        yvex_materialization_session_access_summary(session, &before, &err) == YVEX_OK,
        "initial access summary");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_attach_read_provider(session, &provider, &failure, &err) ==
            YVEX_OK,
        "resident provider attached");
    memset(output, 0, sizeof(output));
    YVEX_TEST_ASSERT(
        yvex_materialization_session_read(session, binding, 7ull, output, sizeof(output),
                                          &failure, &err) == YVEX_OK,
        "resident provider read");
    YVEX_TEST_ASSERT(output[0] == 0x5a && output[sizeof(output) - 1u] == 0x5a,
                     "resident bytes returned");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_borrow(session, binding, 11ull, sizeof(output),
                                            &borrowed, &failure, &err) == YVEX_OK &&
            borrowed == resident + 11u,
        "resident subrange borrowed without copy");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_access_summary(session, &after, &err) == YVEX_OK,
        "resident access summary");
    YVEX_TEST_ASSERT(after.artifact_read_calls == before.artifact_read_calls &&
                         after.artifact_bytes_read == before.artifact_bytes_read,
                     "resident hit performs zero artifact reads");
    YVEX_TEST_ASSERT(after.resident_read_calls == before.resident_read_calls + 2ull &&
                         after.resident_bytes_read ==
                             before.resident_bytes_read + 2ull * sizeof(output),
                     "resident hit accounted exactly");
    before = after;
    memset(thread_contexts, 0, sizeof(thread_contexts));
    for (index = 0u; index < 4u; ++index) {
        thread_contexts[index].session = session;
        thread_contexts[index].binding = binding;
        thread_contexts[index].iterations = 32ull;
        if (pthread_create(&threads[index], NULL, fixture_resident_read_worker,
                           &thread_contexts[index]) != 0)
            break;
        created++;
    }
    for (index = 0u; index < created; ++index)
        (void)pthread_join(threads[index], NULL);
    YVEX_TEST_ASSERT(created == 4u, "concurrent resident reader threads created");
    for (index = 0u; index < created; ++index)
        YVEX_TEST_ASSERT(!thread_contexts[index].failed,
                         "concurrent resident reader completed exactly");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_access_summary(session, &after, &err) == YVEX_OK &&
            after.resident_read_calls == before.resident_read_calls + 128ull &&
            after.resident_bytes_read == before.resident_bytes_read + 2048ull &&
            after.artifact_read_calls == before.artifact_read_calls &&
            span.note_calls == 130ull && span.note_bytes == 2080ull,
        "concurrent resident reads and provider callbacks are synchronized without loss");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_attach_read_provider(session, &provider, &failure, &err) !=
            YVEX_OK,
        "provider replacement refused");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_detach_read_provider(session, &wrong_owner, &failure, &err) !=
            YVEX_OK,
        "foreign provider detach refused");
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_MATERIALIZATION_CLEANUP_FAILURE", "provider-detach-lock", 1) == 0,
        "provider detach synchronization failure injected");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_detach_read_provider(session, &span, &failure, &err) ==
                YVEX_ERR_STATE &&
            failure.code == YVEX_MATERIALIZATION_FAILURE_LIFECYCLE &&
            span.detach_calls == 0ull,
        "failed provider detach retains ownership without callback");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_read(session, binding, 0ull, output, sizeof(output),
                                          &failure, &err) == YVEX_OK &&
            output[0] == 0x5a,
        "failed provider detach keeps the resident reader usable");
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_MATERIALIZATION_CLEANUP_FAILURE") == 0,
        "provider detach synchronization failure cleared");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_detach_read_provider(session, &span, &failure, &err) ==
                YVEX_OK &&
            span.detach_calls == 1ull,
        "owned provider detach retry notifies exactly once");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_read(session, binding, 0ull, output, sizeof(output),
                                          &failure, &err) == YVEX_OK,
        "physical read resumes after detach");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_access_summary(session, &after, &err) == YVEX_OK &&
            after.artifact_read_calls == before.artifact_read_calls + 1ull,
        "post-detach physical read accounted");
    YVEX_TEST_ASSERT(
        yvex_materialization_session_attach_read_provider(session, &provider, &failure, &err) ==
            YVEX_OK,
        "resident provider reattaches before exclusive session teardown");
    yvex_materialization_session_close(session);
    YVEX_TEST_ASSERT(span.detach_calls == 2ull,
                     "exclusive session teardown notifies the reattached provider once");
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

int yvex_test_materialization_runtime(void)
{
    if (test_materialization_fixture() != 0) return 1;
    if (test_materialization_refusals() != 0) return 1;
    if (test_component_residency() != 0) return 1;
    if (test_materialization_resident_provider() != 0) return 1;
    return 0;
}
