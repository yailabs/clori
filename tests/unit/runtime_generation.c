/*
 * Keep the common generation owner in focused and sanitizer aggregates. Focused tests never
 * fabricate a production model/session or teacher-forced generation path. Sanitizer-visible
 * contract proof; complete composition belongs to the live runner.
 */
#include "tests/test.h"

#include <pthread.h>
#include <string.h>

#include <yvex/internal/generation.h>

#include "src/runtime/private.h"

static const char profile_id_a[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char profile_id_b[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
static const char profile_id_c[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
static const char profile_id_d[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
static const char profile_id_e[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
static const char profile_id_f[] =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned long long ready;
    int released;
} generation_scheduler_gate;

typedef struct {
    runtime_engine_scheduler *scheduler;
    runtime_engine_work ticket;
    generation_scheduler_gate *gate;
    int result;
} generation_scheduler_job;

static void generation_scheduler_key(yvex_execution_compatibility_key *key,
                                     unsigned long long layer)
{
    memset(key, 0, sizeof(*key));
    key->schema_version = YVEX_EXECUTION_COMPATIBILITY_SCHEMA_V2;
    key->phase = YVEX_EXECUTION_PHASE_DECODE;
    key->operation = YVEX_EXECUTION_COMPATIBILITY_MOE;
    key->backend_kind = 1u;
    key->tensor_scope = 1u;
    key->execution_class = 1u;
    key->engine_generation = 1ull;
    key->layer_ordinal = layer;
    key->row_width = 64ull;
    key->admitted_width = 4ull;
}

static int generation_scheduler_execute(runtime_engine_work *const *tickets,
                                        unsigned long long ticket_count,
                                        yvex_error *err)
{
    unsigned long long index, rows = 0ull;
    for (index = 0ull; index < ticket_count; ++index) {
        generation_scheduler_job *job = tickets[index]->context;
        if (!job || tickets[index] != &job->ticket ||
            (index && !yvex_execution_compatibility_keys_match(
                          &tickets[0]->key, &tickets[index]->key, err))) {
            yvex_error_set(err, YVEX_ERR_STATE, "test.engine-scheduler",
                           "incompatible ticket entered one physical batch");
            return YVEX_ERR_STATE;
        }
        rows += tickets[index]->row_count;
    }
    if (!rows || rows != tickets[0]->actual_width || rows > 4ull) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.engine-scheduler",
                       "physical batch width is not exact");
        return YVEX_ERR_STATE;
    }
    tickets[0]->worklists.schema_version =
        YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1;
    tickets[0]->worklists.worklist_count = 1ull;
    tickets[0]->worklists.pair_count = rows * 2ull;
    tickets[0]->worklists.bucket_count = rows;
    tickets[0]->worklists.maximum_bucket_population = rows;
    tickets[0]->worklists.narrow_pairs = rows * 2ull;
    tickets[0]->worklists.population_histogram[rows] = rows;
    yvex_error_clear(err);
    return YVEX_OK;
}

static void *generation_scheduler_submit(void *opaque)
{
    generation_scheduler_job *job = opaque;
    yvex_error err;
    (void)pthread_mutex_lock(&job->gate->mutex);
    job->gate->ready++;
    (void)pthread_cond_broadcast(&job->gate->condition);
    while (!job->gate->released)
        (void)pthread_cond_wait(&job->gate->condition, &job->gate->mutex);
    (void)pthread_mutex_unlock(&job->gate->mutex);
    job->result = yvex_runtime_private_engine_scheduler_submit(
        job->scheduler, &job->ticket, &err);
    return NULL;
}

static int generation_test_engine_scheduling(void)
{
    runtime_engine_scheduler *scheduler = NULL;
    yvex_engine_scheduler_summary summary;
    generation_scheduler_job jobs[8];
    pthread_t threads[8];
    generation_scheduler_gate gate;
    yvex_error err;
    unsigned long long index;
    memset(&gate, 0, sizeof(gate));
    memset(jobs, 0, sizeof(jobs));
    YVEX_TEST_ASSERT(pthread_mutex_init(&gate.mutex, NULL) == 0 &&
                         pthread_cond_init(&gate.condition, NULL) == 0,
                     "test submission gate should initialize");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_open(&scheduler, 8ull, &err) == YVEX_OK,
        "engine scheduler should open");
    for (index = 0ull; index < 8ull; ++index) {
        jobs[index].scheduler = scheduler;
        jobs[index].gate = &gate;
        jobs[index].ticket.row_count = 1ull;
        jobs[index].ticket.execute = generation_scheduler_execute;
        jobs[index].ticket.context = &jobs[index];
        generation_scheduler_key(&jobs[index].ticket.key,
                                 index < 4ull ? 0ull : 1ull);
        YVEX_TEST_ASSERT(
            yvex_execution_compatibility_key_validate(
                &jobs[index].ticket.key, &err) == YVEX_OK &&
                pthread_create(&threads[index], NULL, generation_scheduler_submit,
                               &jobs[index]) == 0,
            "compatible batch test ticket should start");
    }
    (void)pthread_mutex_lock(&gate.mutex);
    while (gate.ready != 8ull)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    gate.released = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    do {
        YVEX_TEST_ASSERT(
            yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) ==
                YVEX_OK,
            "queued compatible batch facts should remain inspectable");
    } while (summary.queued != 8ull);
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_start(scheduler, &err) == YVEX_OK,
        "compatible batch worker should start after deterministic queueing");
    for (index = 0ull; index < 8ull; ++index) {
        (void)pthread_join(threads[index], NULL);
        YVEX_TEST_ASSERT(
            jobs[index].result == YVEX_OK &&
                jobs[index].ticket.actual_width == 4ull &&
                jobs[index].ticket.group_size == 4ull,
            "compatible rows should form exact same-operation physical batches");
    }
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) == YVEX_OK &&
            summary.submissions == 8ull && summary.physical_batches == 2ull &&
            summary.submissions_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 8ull &&
            summary.physical_batches_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 2ull &&
            summary.executed_rows_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 8ull &&
            summary.multi_source_batches == 2ull && summary.executed_rows == 8ull &&
            summary.maximum_width == 4ull && summary.multi_source_rows == 8ull &&
            summary.maximum_multi_source_width == 4ull &&
            summary.maximum_source_count == 4ull &&
            summary.multi_source_worklists == 2ull &&
            summary.multi_source_expert_pairs == 16ull &&
            summary.maximum_multi_source_bucket_population == 4ull &&
            !summary.multi_source_matrix_tile_eligible_pairs &&
            !summary.multi_source_matrix_tile_executed_pairs &&
            summary.multi_source_narrow_pairs == 16ull &&
            summary.multi_source_population_histogram[4] == 8ull &&
            !summary.failures,
        "batch summary should expose real width without cross-operation merging");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_close(&scheduler, &err) == YVEX_OK,
        "engine scheduler should close cleanly");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int generation_test_bounded_batch_coalescing(void)
{
    runtime_engine_scheduler *scheduler = NULL;
    yvex_engine_scheduler_summary summary = {0};
    generation_scheduler_gate gate;
    generation_scheduler_job job;
    pthread_t thread;
    yvex_error err;
    unsigned long long attempt;
    memset(&gate, 0, sizeof(gate));
    memset(&job, 0, sizeof(job));
    gate.released = 1;
    YVEX_TEST_ASSERT(pthread_mutex_init(&gate.mutex, NULL) == 0 &&
                         pthread_cond_init(&gate.condition, NULL) == 0,
                     "bounded coalescing gate should initialize");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_open(&scheduler, 4ull, &err) == YVEX_OK &&
            yvex_runtime_private_engine_scheduler_set_producers(scheduler, 1ull, &err) ==
                YVEX_OK &&
            yvex_runtime_private_engine_scheduler_start(scheduler, &err) == YVEX_OK,
        "one ready producer should start bounded arrival coalescing");
    job.scheduler = scheduler;
    job.gate = &gate;
    job.ticket.row_count = 1ull;
    job.ticket.coalescing_limit_ns = 1000000ull;
    job.ticket.execute = generation_scheduler_execute;
    job.ticket.context = &job;
    job.ticket.kind = RUNTIME_ENGINE_WORK_RENDEZVOUS;
    generation_scheduler_key(&job.ticket.key, 0ull);
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_validate(&job.ticket.key, &err) == YVEX_OK &&
            pthread_create(&thread, NULL, generation_scheduler_submit, &job) == 0,
        "one producer should submit into the bounded rendezvous");
    for (attempt = 0ull; attempt < 100000ull; ++attempt) {
        YVEX_TEST_ASSERT(
            yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) ==
                YVEX_OK,
            "bounded coalescing state should remain inspectable");
        if (summary.coalescing_waits) break;
    }
    (void)pthread_join(thread, NULL);
    YVEX_TEST_ASSERT(
        job.result == YVEX_OK &&
            yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) ==
                YVEX_OK &&
            summary.registered_producers == 1ull &&
            summary.coalescing_waits == 1ull &&
            summary.coalescing_timeouts == 1ull && summary.coalescing_ns &&
            summary.rendezvous_submissions == 1ull &&
            summary.rendezvous_steps == 1ull &&
            summary.submissions_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 1ull &&
            summary.rendezvous_steps_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 1ull &&
            summary.multi_source_rendezvous == 0ull &&
            summary.maximum_rendezvous_width == 1ull &&
            summary.physical_batches == 0ull,
        "the first ready quantum waits briefly for a peer, then executes width one");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_close(&scheduler, &err) == YVEX_OK,
        "bounded coalescing owner should close cleanly");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int generation_test_incompatible_arrival_releases_impossible_wait(void)
{
    runtime_engine_scheduler *scheduler = NULL;
    yvex_engine_scheduler_summary summary = {0};
    generation_scheduler_gate gate;
    generation_scheduler_job jobs[2];
    pthread_t threads[2];
    yvex_error err;
    unsigned long long attempt;
    memset(&gate, 0, sizeof(gate));
    memset(jobs, 0, sizeof(jobs));
    gate.released = 1;
    YVEX_TEST_ASSERT(pthread_mutex_init(&gate.mutex, NULL) == 0 &&
                         pthread_cond_init(&gate.condition, NULL) == 0,
                     "incompatible-arrival gate should initialize");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_open(&scheduler, 4ull, &err) == YVEX_OK &&
            yvex_runtime_private_engine_scheduler_set_producers(scheduler, 4ull, &err) ==
                YVEX_OK &&
            yvex_runtime_private_engine_scheduler_start(scheduler, &err) == YVEX_OK,
        "four-producer engine scheduler should start");
    jobs[0].scheduler = scheduler;
    jobs[0].gate = &gate;
    jobs[0].ticket.row_count = 1ull;
    jobs[0].ticket.coalescing_limit_ns = 100000000ull;
    jobs[0].ticket.execute = generation_scheduler_execute;
    jobs[0].ticket.context = &jobs[0];
    generation_scheduler_key(&jobs[0].ticket.key, 0ull);
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_validate(&jobs[0].ticket.key, &err) ==
                YVEX_OK &&
            pthread_create(&threads[0], NULL, generation_scheduler_submit,
                           &jobs[0]) == 0,
        "declared width-four ticket should enter coalescing");
    for (attempt = 0ull; attempt < 100000ull; ++attempt) {
        YVEX_TEST_ASSERT(
            yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) ==
                YVEX_OK,
            "coalescing wait should remain observable");
        if (summary.coalescing_waits) break;
    }
    YVEX_TEST_ASSERT(summary.coalescing_waits == 1ull,
                     "declared width-four ticket should wait for peers");
    jobs[1].scheduler = scheduler;
    jobs[1].gate = &gate;
    jobs[1].ticket.row_count = 1ull;
    jobs[1].ticket.coalescing_limit_ns = 1ull;
    jobs[1].ticket.execute = generation_scheduler_execute;
    jobs[1].ticket.context = &jobs[1];
    generation_scheduler_key(&jobs[1].ticket.key, 1ull);
    jobs[1].ticket.key.phase = YVEX_EXECUTION_PHASE_VERIFY;
    YVEX_TEST_ASSERT(
        yvex_execution_compatibility_key_validate(&jobs[1].ticket.key, &err) ==
                YVEX_OK &&
            pthread_create(&threads[1], NULL, generation_scheduler_submit,
                           &jobs[1]) == 0,
        "incompatible ticket should wake but not satisfy coalescing");
    (void)pthread_join(threads[0], NULL);
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_set_producers(scheduler, 1ull, &err) ==
            YVEX_OK,
        "completed incompatible producer should leave the ready population");
    (void)pthread_join(threads[1], NULL);
    YVEX_TEST_ASSERT(
        jobs[0].result == YVEX_OK && jobs[1].result == YVEX_OK &&
            jobs[0].ticket.actual_width == 1ull &&
            jobs[1].ticket.actual_width == 1ull &&
            yvex_runtime_private_engine_scheduler_snapshot(scheduler, &summary, &err) ==
                YVEX_OK &&
            summary.submissions == 2ull && summary.physical_batches == 2ull &&
            summary.submissions_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 1ull &&
            summary.submissions_by_phase[YVEX_EXECUTION_PHASE_VERIFY] == 1ull &&
            summary.physical_batches_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 1ull &&
            summary.physical_batches_by_phase[YVEX_EXECUTION_PHASE_VERIFY] == 1ull &&
            summary.executed_rows_by_phase[YVEX_EXECUTION_PHASE_DECODE] == 1ull &&
            summary.executed_rows_by_phase[YVEX_EXECUTION_PHASE_VERIFY] == 1ull &&
            summary.phase_mismatches &&
            !summary.multi_source_batches && !summary.multi_source_rows &&
            !summary.maximum_multi_source_width && !summary.maximum_source_count &&
            summary.coalescing_waits == 2ull &&
            summary.coalescing_timeouts == 1ull,
        "incompatible work stays separate under bounded engine scheduling");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_scheduler_close(&scheduler, &err) == YVEX_OK,
        "incompatible-arrival batch owner should close cleanly");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int generation_test_active_scheduler_producers(void)
{
    yvex_model_engine model = {0};
    yvex_engine_scheduler_summary summary = {0};
    yvex_error err;
    YVEX_TEST_ASSERT(pthread_mutex_init(&model.lifecycle_mutex, NULL) == 0,
                     "model engine scheduler lock should initialize");
    model.lifecycle_mutex_ready = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_acquire(&model, 2ull, 2ull, 2ull, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_acquire(&model, 2ull, 2ull, 2ull, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_acquire(&model, 2ull, 2ull, 2ull, &err) ==
                YVEX_OK &&
            model.engine_scheduler_references == 3ull,
        "idle context references must not consume admitted producer width");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_producer_enter(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_producer_enter(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_producer_enter(&model, &err) ==
                YVEX_ERR_BOUNDS &&
            yvex_runtime_private_engine_scheduler_snapshot(
                model.engine_scheduler, &summary, &err) == YVEX_OK &&
            summary.registered_producers == 2ull &&
            model.engine_scheduler_producers == 2ull,
        "only active executions may consume bounded producer width");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_producer_leave(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_producer_leave(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_release(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_release(&model, &err) ==
                YVEX_OK &&
            yvex_runtime_private_model_scheduler_release(&model, &err) ==
                YVEX_OK &&
            !model.engine_scheduler && !model.engine_scheduler_references &&
            !model.engine_scheduler_producers,
        "active producers and idle references should drain independently");
    (void)pthread_mutex_destroy(&model.lifecycle_mutex);
    return 0;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned long long entered;
    int released;
} generation_progress_gate;

typedef struct {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    generation_progress_gate *gate;
    int result;
} generation_progress_job;

static void *generation_progress_run(void *opaque)
{
    generation_progress_job *job = opaque;
    runtime_engine_progress_lease lease = {0};
    yvex_error err;
    job->result = yvex_runtime_private_engine_progress_enter(
        job->model, job->session, YVEX_ENGINE_PROGRESS_DECODE,
        NULL, NULL, &lease, &err);
    if (job->result != YVEX_OK) return NULL;
    (void)pthread_mutex_lock(&job->gate->mutex);
    job->gate->entered++;
    (void)pthread_cond_broadcast(&job->gate->condition);
    while (!job->gate->released)
        (void)pthread_cond_wait(&job->gate->condition, &job->gate->mutex);
    (void)pthread_mutex_unlock(&job->gate->mutex);
    job->result = yvex_runtime_private_engine_progress_leave(
        &lease, YVEX_OK, &err);
    return NULL;
}

static int generation_progress_cancelled(void *opaque)
{
    return opaque && *(const int *)opaque;
}

static int generation_test_ready_progress_scheduling(void)
{
    yvex_model_engine model = {0};
    yvex_runtime_execution_session sessions[2] = {{0}};
    generation_progress_job jobs[2] = {{0}};
    generation_progress_gate gate = {0};
    runtime_engine_progress_lease lease = {0};
    yvex_engine_scheduler_summary summary = {0};
    pthread_t threads[2];
    yvex_error err;
    unsigned long long index;
    int cancel_requested = 0;
    YVEX_TEST_ASSERT(
        pthread_mutex_init(&model.lifecycle_mutex, NULL) == 0 &&
            pthread_mutex_init(&gate.mutex, NULL) == 0 &&
            pthread_cond_init(&gate.condition, NULL) == 0,
        "ready-progress test owners should initialize");
    model.lifecycle_mutex_ready = 1;
    model.summary.engine_generation = 7ull;
    for (index = 0ull; index < 2ull; ++index) {
        sessions[index].engine = &model;
        sessions[index].summary.engine_generation = 7ull;
        sessions[index].batch_source_ordinal = index + 1ull;
        jobs[index].model = &model;
        jobs[index].gate = &gate;
    }
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_acquire(
            &model, 2ull, 2ull, 2ull, &err) == YVEX_OK,
        "engine ready-progress scheduler should open");
    jobs[0].session = jobs[1].session = &sessions[0];
    YVEX_TEST_ASSERT(
        pthread_create(&threads[0], NULL, generation_progress_run, &jobs[0]) == 0 &&
            pthread_create(&threads[1], NULL, generation_progress_run, &jobs[1]) == 0,
        "same-sequence ready work should submit concurrently");
    (void)pthread_mutex_lock(&gate.mutex);
    while (gate.entered != 1ull)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    (void)pthread_mutex_unlock(&gate.mutex);
    do {
        YVEX_TEST_ASSERT(
            yvex_model_engine_scheduler_summary_copy(
                &model, &summary, &err) == YVEX_OK,
            "same-sequence exclusion should remain observable");
    } while (summary.ready_sequence_work != 1ull);
    YVEX_TEST_ASSERT(
        summary.active_sequences == 1ull && summary.sequence_conflicts,
        "one sequence generation cannot advance concurrently");
    (void)pthread_mutex_lock(&gate.mutex);
    gate.released = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    for (index = 0ull; index < 2ull; ++index) {
        (void)pthread_join(threads[index], NULL);
        YVEX_TEST_ASSERT(jobs[index].result == YVEX_OK,
                         "serialized same-sequence work should complete");
    }
    (void)pthread_mutex_lock(&gate.mutex);
    gate.entered = 0ull;
    gate.released = 0;
    (void)pthread_mutex_unlock(&gate.mutex);
    jobs[0].session = &sessions[0];
    jobs[1].session = &sessions[1];
    YVEX_TEST_ASSERT(
        pthread_create(&threads[0], NULL, generation_progress_run, &jobs[0]) == 0 &&
            pthread_create(&threads[1], NULL, generation_progress_run, &jobs[1]) == 0,
        "distinct ready sequences should submit concurrently");
    (void)pthread_mutex_lock(&gate.mutex);
    while (gate.entered != 2ull)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    gate.released = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    for (index = 0ull; index < 2ull; ++index) {
        (void)pthread_join(threads[index], NULL);
        YVEX_TEST_ASSERT(jobs[index].result == YVEX_OK,
                         "distinct ready sequences should complete");
    }
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_progress_enter(
            &model, &sessions[0], YVEX_ENGINE_PROGRESS_DECODE,
            NULL, NULL, &lease, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_transition(
                &lease, YVEX_ENGINE_PROGRESS_DRAFT, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_transition(
                &lease, YVEX_ENGINE_PROGRESS_VERIFY, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_transition(
                &lease, YVEX_ENGINE_PROGRESS_CORRECTION, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_transition(
                &lease, YVEX_ENGINE_PROGRESS_PUBLICATION, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_leave(
                &lease, YVEX_OK, &err) == YVEX_OK &&
            yvex_model_engine_scheduler_summary_copy(
                &model, &summary, &err) == YVEX_OK &&
            summary.sequence_capacity == 2ull &&
            summary.admitted_maximum_width == 2ull &&
            summary.maximum_active_sequences == 2ull &&
            summary.progress_submissions == 9ull &&
            summary.progress_transitions == 4ull &&
            summary.progress_completions == 9ull &&
            summary.progress_submissions_by_kind[YVEX_ENGINE_PROGRESS_DECODE] == 5ull &&
            summary.progress_submissions_by_kind[YVEX_ENGINE_PROGRESS_DRAFT] == 1ull &&
            summary.progress_submissions_by_kind[YVEX_ENGINE_PROGRESS_VERIFY] == 1ull &&
            summary.progress_submissions_by_kind[YVEX_ENGINE_PROGRESS_CORRECTION] == 1ull &&
            summary.progress_submissions_by_kind[YVEX_ENGINE_PROGRESS_PUBLICATION] == 1ull &&
            !summary.ready_sequence_work && !summary.active_sequences,
        "one engine authority should expose bounded phase progression");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_progress_enter(
            &model, &sessions[0], YVEX_ENGINE_PROGRESS_DECODE,
            generation_progress_cancelled, &cancel_requested,
            &lease, &err) == YVEX_OK,
        "active sequence work should admit before cancellation");
    cancel_requested = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_progress_transition(
            &lease, YVEX_ENGINE_PROGRESS_DRAFT, &err) ==
                YVEX_ERR_CANCELLED &&
            yvex_runtime_private_engine_progress_leave(
                &lease, YVEX_ERR_CANCELLED, &err) ==
                YVEX_ERR_CANCELLED &&
            yvex_model_engine_scheduler_summary_copy(
                &model, &summary, &err) == YVEX_OK &&
            summary.progress_submissions == 11ull &&
            summary.progress_completions == 10ull &&
            summary.progress_cancellations == 1ull &&
            !summary.ready_sequence_work && !summary.active_sequences,
        "queued phase cancellation should release one exact sequence lease");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_release(&model, &err) == YVEX_OK,
        "ready-progress scheduler should drain cleanly");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    (void)pthread_mutex_destroy(&model.lifecycle_mutex);
    return 0;
}

static int generation_test_runnable_capacity(void)
{
    yvex_model_engine model = {0};
    yvex_runtime_execution_session sessions[2] = {{0}};
    generation_progress_job job = {0};
    generation_progress_gate gate = {0};
    runtime_engine_progress_lease active = {0};
    yvex_engine_scheduler_summary summary = {0};
    pthread_t thread;
    yvex_error err;
    unsigned long long attempt;
    YVEX_TEST_ASSERT(
        pthread_mutex_init(&model.lifecycle_mutex, NULL) == 0 &&
            pthread_mutex_init(&gate.mutex, NULL) == 0 &&
            pthread_cond_init(&gate.condition, NULL) == 0,
        "runnable-capacity test owners should initialize");
    model.lifecycle_mutex_ready = 1;
    model.summary.engine_generation = 11ull;
    for (attempt = 0ull; attempt < 2ull; ++attempt) {
        sessions[attempt].engine = &model;
        sessions[attempt].summary.engine_generation = 11ull;
        sessions[attempt].batch_source_ordinal = attempt + 1ull;
    }
    YVEX_TEST_ASSERT(
        yvex_runtime_private_model_scheduler_acquire(
            &model, 1ull, 2ull, 1ull, &err) == YVEX_OK &&
            yvex_runtime_private_engine_progress_enter(
                &model, &sessions[0], YVEX_ENGINE_PROGRESS_DECODE,
                NULL, NULL, &active, &err) == YVEX_OK,
        "one-wide engine should admit a larger runnable population");
    job.model = &model;
    job.session = &sessions[1];
    job.gate = &gate;
    YVEX_TEST_ASSERT(
        pthread_create(&thread, NULL, generation_progress_run, &job) == 0,
        "independent work should wait for one-wide physical execution");
    for (attempt = 0ull; attempt < 100000ull; ++attempt) {
        YVEX_TEST_ASSERT(
            yvex_model_engine_scheduler_summary_copy(
                &model, &summary, &err) == YVEX_OK,
            "runnable scheduler pressure should remain inspectable");
        if (summary.ready_sequence_work == 1ull) break;
    }
    YVEX_TEST_ASSERT(
        attempt < 100000ull && summary.sequence_capacity == 1ull &&
            summary.runnable_capacity == 2ull &&
            summary.cooperative_width == 1ull &&
            summary.maximum_active_sequences == 1ull,
        "runnable admission must remain distinct from physical sequence width");
    YVEX_TEST_ASSERT(
        yvex_runtime_private_engine_progress_leave(
            &active, YVEX_OK, &err) == YVEX_OK,
        "first physical quantum should release its lease");
    (void)pthread_mutex_lock(&gate.mutex);
    while (gate.entered != 1ull)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    gate.released = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    (void)pthread_join(thread, NULL);
    YVEX_TEST_ASSERT(
        job.result == YVEX_OK &&
            yvex_runtime_private_model_scheduler_release(&model, &err) == YVEX_OK,
        "queued independent work should run after the admitted safe point");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    (void)pthread_mutex_destroy(&model.lifecycle_mutex);
    return 0;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned long long entered, chronology[32], chronology_count;
    int run;
} cooperative_execution_gate;

typedef struct {
    yvex_runtime_execution_coordinator *scheduler;
    cooperative_execution_gate *gate;
    unsigned long long handle, steps, cancel_after;
    int cancelled, result;
} cooperative_execution_job;

static int cooperative_execution_cancelled(void *opaque)
{
    cooperative_execution_job *job = opaque;
    return job && job->cancelled;
}

static void *cooperative_execution_run(void *opaque)
{
    cooperative_execution_job *job = opaque;
    yvex_runtime_execution_lease lease = {0};
    yvex_error err;
    unsigned long long step;
    job->result = yvex_runtime_execution_enter(
        job->scheduler, 19ull, job->handle, YVEX_ENGINE_PROGRESS_COMPONENT,
        cooperative_execution_cancelled, job, &lease, &err);
    if (job->result != YVEX_OK) return NULL;
    (void)pthread_mutex_lock(&job->gate->mutex);
    job->gate->entered++;
    (void)pthread_cond_broadcast(&job->gate->condition);
    while (!job->gate->run)
        (void)pthread_cond_wait(&job->gate->condition, &job->gate->mutex);
    (void)pthread_mutex_unlock(&job->gate->mutex);
    for (step = 0ull; step < job->steps; ++step) {
        (void)pthread_mutex_lock(&job->gate->mutex);
        job->gate->chronology[job->gate->chronology_count++] = job->handle;
        (void)pthread_mutex_unlock(&job->gate->mutex);
        if (job->cancel_after == step + 1ull) job->cancelled = 1;
        if (job->cancelled) {
            job->result = yvex_runtime_execution_leave(
                &lease, YVEX_ERR_CANCELLED, &err);
            return NULL;
        }
        if (step + 1ull < job->steps &&
            yvex_runtime_execution_yield_requested(&lease)) {
            job->result = yvex_runtime_execution_yield_resume(&lease, &err);
            if (job->result != YVEX_OK) return NULL;
        }
    }
    job->result = yvex_runtime_execution_leave(&lease, YVEX_OK, &err);
    return NULL;
}

static int cooperative_execution_case(
    unsigned long long first_steps, unsigned long long first_cancel_after,
    unsigned long long second_steps, unsigned long long second_cancel_after,
    yvex_engine_scheduler_summary *summary,
    unsigned long long chronology[32], unsigned long long *chronology_count)
{
    yvex_runtime_execution_coordinator *scheduler = NULL;
    cooperative_execution_gate gate = {0};
    cooperative_execution_job jobs[2] = {{0}};
    pthread_t threads[2];
    yvex_error err;
    unsigned long long attempt;
    YVEX_TEST_ASSERT(
        pthread_mutex_init(&gate.mutex, NULL) == 0 &&
            pthread_cond_init(&gate.condition, NULL) == 0 &&
            yvex_runtime_execution_coordinator_open(
                &scheduler, 4ull, 1ull, &err) == YVEX_OK,
        "single-width cooperative scheduler should open");
    jobs[0] = (cooperative_execution_job){
        scheduler, &gate, 1ull, first_steps, first_cancel_after, 0, YVEX_OK};
    jobs[1] = (cooperative_execution_job){
        scheduler, &gate, 2ull, second_steps, second_cancel_after, 0, YVEX_OK};
    YVEX_TEST_ASSERT(
        pthread_create(&threads[0], NULL, cooperative_execution_run, &jobs[0]) == 0,
        "first long cooperative request should start");
    (void)pthread_mutex_lock(&gate.mutex);
    while (gate.entered != 1ull)
        (void)pthread_cond_wait(&gate.condition, &gate.mutex);
    (void)pthread_mutex_unlock(&gate.mutex);
    YVEX_TEST_ASSERT(
        pthread_create(&threads[1], NULL, cooperative_execution_run, &jobs[1]) == 0,
        "second cooperative request should become runnable");
    for (attempt = 0ull; attempt < 100000ull; ++attempt) {
        YVEX_TEST_ASSERT(
            yvex_runtime_execution_coordinator_summary_copy(
                scheduler, summary, &err) == YVEX_OK,
            "cooperative ready queue should remain inspectable");
        if (summary->ready_sequence_work == 1ull) break;
    }
    YVEX_TEST_ASSERT(summary->ready_sequence_work == 1ull,
                     "second request should wait behind one active quantum");
    (void)pthread_mutex_lock(&gate.mutex);
    gate.run = 1;
    (void)pthread_cond_broadcast(&gate.condition);
    (void)pthread_mutex_unlock(&gate.mutex);
    (void)pthread_join(threads[0], NULL);
    (void)pthread_join(threads[1], NULL);
    YVEX_TEST_ASSERT(
        jobs[0].result == (first_cancel_after ? YVEX_ERR_CANCELLED : YVEX_OK) &&
            jobs[1].result == (second_cancel_after ? YVEX_ERR_CANCELLED : YVEX_OK) &&
            yvex_runtime_execution_coordinator_summary_copy(
                scheduler, summary, &err) == YVEX_OK,
        "cooperative requests should reach their exact terminal states");
    memcpy(chronology, gate.chronology,
           (size_t)(gate.chronology_count * sizeof(*chronology)));
    *chronology_count = gate.chronology_count;
    YVEX_TEST_ASSERT(
        yvex_runtime_execution_coordinator_close(&scheduler, &err) == YVEX_OK,
        "cooperative scheduler should drain cleanly");
    (void)pthread_cond_destroy(&gate.condition);
    (void)pthread_mutex_destroy(&gate.mutex);
    return 0;
}

static int generation_test_cooperative_fairness(void)
{
    yvex_engine_scheduler_summary summary = {0};
    unsigned long long chronology[32], count = 0ull;
    if (cooperative_execution_case(3ull, 2ull, 4ull, 0ull, &summary,
                                   chronology, &count) != 0)
        return 1;
    YVEX_TEST_ASSERT(
        count == 6ull && chronology[0] == 1ull && chronology[1] == 2ull &&
            chronology[2] == 1ull && chronology[3] == 2ull &&
            summary.cooperative_width == 1ull &&
            summary.maximum_active_sequences == 1ull &&
            summary.maximum_ready_sequence_work == 1ull &&
            summary.cooperative_yields == 2ull &&
            summary.cooperative_resumes == 2ull &&
            summary.progress_cancellations == 1ull &&
            summary.scheduler_wait_nanoseconds &&
            summary.maximum_quantum_nanoseconds,
        "two long requests should alternate before one cancels and the other continues");
    memset(&summary, 0, sizeof(summary));
    memset(chronology, 0, sizeof(chronology));
    count = 0ull;
    if (cooperative_execution_case(6ull, 0ull, 1ull, 0ull, &summary,
                                   chronology, &count) != 0)
        return 1;
    YVEX_TEST_ASSERT(
        count == 7ull && chronology[0] == 1ull && chronology[1] == 2ull &&
            summary.cooperative_yields == 1ull &&
            summary.cooperative_resumes == 1ull &&
            !summary.progress_cancellations,
        "one short request should run at the first safe point of a long request");
    memset(&summary, 0, sizeof(summary));
    memset(chronology, 0, sizeof(chronology));
    count = 0ull;
    if (cooperative_execution_case(4ull, 0ull, 3ull, 2ull, &summary,
                                   chronology, &count) != 0)
        return 1;
    YVEX_TEST_ASSERT(
        count == 6ull && chronology[0] == 1ull && chronology[1] == 2ull &&
            chronology[2] == 1ull && chronology[3] == 2ull &&
            chronology[4] == 1ull && chronology[5] == 1ull &&
            summary.cooperative_yields == 3ull &&
            summary.cooperative_resumes == 3ull &&
            summary.progress_cancellations == 1ull,
        "cancelling the second request should not stall the first request");
    memset(&summary, 0, sizeof(summary));
    memset(chronology, 0, sizeof(chronology));
    count = 0ull;
    if (cooperative_execution_case(12ull, 0ull, 12ull, 0ull, &summary,
                                   chronology, &count) != 0)
        return 1;
    YVEX_TEST_ASSERT(
        count == 24ull && summary.progress_submissions == 24ull &&
            summary.progress_completions == 24ull &&
            summary.cooperative_yields == 22ull &&
            summary.cooperative_resumes == 22ull &&
            !summary.ready_sequence_work && !summary.active_sequences,
        "repeated yield and resume should finish without retained scheduler work");
    return 0;
}

typedef struct {
    yvex_runtime_execution_coordinator *scheduler;
    yvex_runtime_execution_lease lease;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered, proceed, result;
} cooperative_drain_job;

typedef struct {
    yvex_runtime_execution_coordinator **scheduler;
    int result;
} cooperative_close_job;

static void *cooperative_drain_run(void *opaque)
{
    cooperative_drain_job *job = opaque;
    yvex_error err;
    job->result = yvex_runtime_execution_enter(
        job->scheduler, 23ull, 1ull, YVEX_ENGINE_PROGRESS_COMPONENT,
        NULL, NULL, &job->lease, &err);
    if (job->result != YVEX_OK) return NULL;
    (void)pthread_mutex_lock(&job->mutex);
    job->entered = 1;
    (void)pthread_cond_broadcast(&job->condition);
    while (!job->proceed)
        (void)pthread_cond_wait(&job->condition, &job->mutex);
    (void)pthread_mutex_unlock(&job->mutex);
    if (!yvex_runtime_execution_yield_requested(&job->lease)) {
        job->result = yvex_runtime_execution_leave(&job->lease, YVEX_OK, &err);
        return NULL;
    }
    job->result = yvex_runtime_execution_yield_resume(&job->lease, &err);
    return NULL;
}

static void *cooperative_close_run(void *opaque)
{
    cooperative_close_job *job = opaque;
    yvex_error err;
    job->result = yvex_runtime_execution_coordinator_close(job->scheduler, &err);
    return NULL;
}

static int generation_test_cooperative_drain(void)
{
    yvex_runtime_execution_coordinator *scheduler = NULL;
    cooperative_drain_job work = {0};
    cooperative_close_job close = {&scheduler, YVEX_ERR_STATE};
    pthread_t worker, closer;
    yvex_error err;
    unsigned long long attempt;
    YVEX_TEST_ASSERT(
        yvex_runtime_execution_coordinator_open(
            &scheduler, 2ull, 1ull, &err) == YVEX_OK &&
            pthread_mutex_init(&work.mutex, NULL) == 0 &&
            pthread_cond_init(&work.condition, NULL) == 0,
        "cooperative drain fixture should open");
    work.scheduler = scheduler;
    YVEX_TEST_ASSERT(
        pthread_create(&worker, NULL, cooperative_drain_run, &work) == 0,
        "cooperative drain work should start");
    (void)pthread_mutex_lock(&work.mutex);
    while (!work.entered)
        (void)pthread_cond_wait(&work.condition, &work.mutex);
    (void)pthread_mutex_unlock(&work.mutex);
    YVEX_TEST_ASSERT(
        pthread_create(&closer, NULL, cooperative_close_run, &close) == 0,
        "scheduler drain should start concurrently with active work");
    for (attempt = 0ull; attempt < 100000ull; ++attempt)
        if (yvex_runtime_execution_yield_requested(&work.lease)) break;
    YVEX_TEST_ASSERT(
        attempt < 100000ull,
        "a draining scheduler should request the next admitted safe point");
    (void)pthread_mutex_lock(&work.mutex);
    work.proceed = 1;
    (void)pthread_cond_broadcast(&work.condition);
    (void)pthread_mutex_unlock(&work.mutex);
    (void)pthread_join(worker, NULL);
    (void)pthread_join(closer, NULL);
    YVEX_TEST_ASSERT(
        work.result == YVEX_ERR_STATE && close.result == YVEX_OK && !scheduler &&
            !work.lease.scheduler,
        "active work should abandon its candidate at a drain safe point");
    (void)pthread_cond_destroy(&work.condition);
    (void)pthread_mutex_destroy(&work.mutex);
    return 0;
}

static int generation_test_stop_taxonomy(void)
{
    unsigned int reason;
    for (reason = YVEX_GENERATION_STOP_NONE;
         reason <= YVEX_GENERATION_STOP_OUTPUT_FAILURE; ++reason)
        YVEX_TEST_ASSERT(
            strcmp(yvex_runtime_generation_stop_reason_name(
                       (yvex_runtime_generation_stop_reason)reason),
                   "unknown") != 0,
            "generation stop reason must have a stable name");
    YVEX_TEST_ASSERT(strcmp(yvex_runtime_generation_stop_reason_name(
                               (yvex_runtime_generation_stop_reason)99),
                           "unknown") == 0,
                     "unknown generation stop reason must remain explicit");
    return 0;
}

static int generation_test_refusals(void)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_generation_result result;
    yvex_runtime_generation_options options;
    yvex_generation_operator_result operator_result;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    int complete = 1;
    yvex_error err;
    memset(&result, 0, sizeof(result));
    memset(&options, 0, sizeof(options));
    memset(&operator_result, 0, sizeof(operator_result));
    YVEX_TEST_ASSERT(yvex_runtime_generation_context_open(
                         &context, NULL, NULL, &options, &err) ==
                         YVEX_ERR_INVALID_ARG && !context,
                     "generation context must refuse absent model/session");
    YVEX_TEST_ASSERT(yvex_runtime_generation_execute(
                         NULL, NULL, NULL, 0ull, NULL, 0ull, &result, NULL,
                         &err) ==
                         YVEX_ERR_INVALID_ARG,
                     "generation execute must refuse absent context");
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_turn_begin(
            NULL, NULL, NULL, 0ull, NULL, 0ull, &result, &err) ==
            YVEX_ERR_INVALID_ARG,
        "incremental turn begin must refuse absent ownership");
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_turn_advance(NULL, 1ull, &complete, &err) ==
                YVEX_ERR_INVALID_ARG &&
            !complete,
        "incremental turn advance must refuse absent active work");
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_turn_finish(NULL, &err) == YVEX_ERR_STATE,
        "incremental turn finish must refuse absent completed work");
    YVEX_TEST_ASSERT(yvex_runtime_generation_result_validate(
                         NULL, NULL, 0ull, NULL, 0ull, &result, &err) ==
                         YVEX_ERR_FORMAT,
                     "generation result validation must fail closed");
    YVEX_TEST_ASSERT(yvex_runtime_generation_operator_execute(
                         NULL, &operator_result, &cleanup, &err) ==
                         YVEX_ERR_INVALID_ARG && !cleanup,
                     "generation operator must refuse absent request");
    YVEX_TEST_ASSERT(yvex_runtime_generation_context_close(&context, &err) ==
                         YVEX_OK && !context,
                     "generation close must be idempotent for empty ownership");
    yvex_runtime_generation_operator_result_release(&operator_result);
    return 0;
}

static int generation_test_execution_identity_excludes_measurement(void)
{
    yvex_runtime_generation_result result;
    yvex_runtime_generation_evidence evidence;
    char before[YVEX_SHA256_HEX_CAP], after[YVEX_SHA256_HEX_CAP];
    memset(&result, 0, sizeof(result));
    memset(&evidence, 0, sizeof(evidence));
    result.schema_version = YVEX_RUNTIME_GENERATION_RESULT_SCHEMA_V5;
    result.execution_mode = YVEX_GENERATION_MODE_SPECULATIVE;
    result.draft_cycle_count = 2ull;
    result.proposed_token_count = 8ull;
    result.accepted_draft_token_count = 6ull;
    result.rejected_draft_token_count = 2ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, before),
        "generation execution identity must admit a bounded result");
    result.draft_ns = 123456ull;
    result.verification_ns = 654321ull;
    result.speculative_commit_ns = 777ull;
    result.mean_accepted_prefix = 3.0;
    result.effective_committed_tokens_per_second = 0.75;
    evidence.roofline_available = 1;
    memset(evidence.roofline.identity, 'a', YVEX_SHA256_HEX_CAP - 1u);
    evidence.roofline.identity[YVEX_SHA256_HEX_CAP - 1u] = '\0';
    YVEX_TEST_ASSERT(
        evidence.roofline_available &&
            yvex_runtime_generation_execution_identity(&result, NULL, after) &&
            strcmp(before, after) == 0,
        "measurement values must not alter semantic execution identity");
    result.speculation_source_boundary_token_count = 1ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, after) &&
            strcmp(before, after) != 0,
        "source-boundary continuation must alter execution identity");
    result.speculation_source_boundary_token_count = 0ull;
    result.proposed_token_count++;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_execution_identity(&result, NULL, after) &&
            strcmp(before, after) != 0,
        "semantic speculative counters must alter execution identity");
    return 0;
}

static int generation_test_evidence_sidecar_validation(void)
{
    yvex_runtime_generation_plan_summary plan = {0};
    yvex_runtime_generation_evidence evidence = {0}, mutated;
    yvex_error err;
    evidence.schema_version = YVEX_RUNTIME_GENERATION_EVIDENCE_SCHEMA_V1;
    YVEX_TEST_ASSERT(
        yvex_runtime_profile_begin(
            &evidence.profile, YVEX_RUNTIME_PROFILE_SUMMARY,
            YVEX_RUNTIME_PROFILE_GENERATION, YVEX_BACKEND_KIND_CPU,
            profile_id_a, profile_id_b, profile_id_c, profile_id_d,
            profile_id_e, profile_id_f, &err) == YVEX_OK &&
            yvex_runtime_profile_finish(&evidence.profile, &err) == YVEX_OK &&
            yvex_runtime_generation_evidence_validate(
                &plan, &evidence, &err) == YVEX_OK,
        "generation evidence validates independently from semantic output");
    mutated = evidence;
    mutated.profile.counters[YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES]++;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_evidence_validate(
            &plan, &mutated, &err) == YVEX_ERR_FORMAT,
        "mutated generation evidence fails closed without changing generation identity");
    mutated = evidence;
    mutated.schema_version++;
    YVEX_TEST_ASSERT(
        yvex_runtime_generation_evidence_validate(
            &plan, &mutated, &err) == YVEX_ERR_FORMAT,
        "unknown generation evidence schema fails closed");
    return 0;
}

static int generation_test_plan_binds_workload_profile(void)
{
    yvex_runtime_generation_plan_summary plan;
    char before[YVEX_SHA256_HEX_CAP], after[YVEX_SHA256_HEX_CAP];
    memset(&plan, 0, sizeof(plan));
    plan.schema_version = YVEX_RUNTIME_GENERATION_PLAN_SCHEMA_CURRENT;
    plan.producer_kind = YVEX_EXECUTION_PLAN_TRANSFORMER;
    strcpy(plan.producer_plan_identity,
           "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    strcpy(plan.workload_profile_identity,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, before),
                     "generation plan must bind a workload profile");
    plan.trace_policy = YVEX_RUNTIME_TRACE_FULL;
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, after) &&
                         strcmp(before, after) == 0,
                     "observability policy must not alter semantic execution identity");
    plan.trace_policy = YVEX_RUNTIME_TRACE_NONE;
    plan.workload_profile_identity[0] = 'b';
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, after) &&
                         strcmp(before, after) != 0,
                     "workload profile changes must alter generation plan identity");
    strcpy(plan.workload_profile_identity,
           "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    plan.producer_kind = YVEX_EXECUTION_PLAN_DECODER;
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, after) &&
                         strcmp(before, after) != 0,
                     "generation producer kind must alter generation plan identity");
    plan.producer_kind = YVEX_EXECUTION_PLAN_TRANSFORMER;
    plan.producer_plan_identity[0] = 'd';
    YVEX_TEST_ASSERT(yvex_runtime_generation_plan_identity(&plan, after) &&
                         strcmp(before, after) != 0,
                     "generation producer identity must alter generation plan identity");
    return 0;
}

static int generation_test_decode_profile_projection(void)
{
    yvex_runtime_decode_step_result decode = {0};
    yvex_runtime_profile_record profile;
    yvex_error err;
    decode.completed = 1;
    decode.kernel_launches = 19ull;
    decode.accelerated_matrix_launches = 3ull;
    decode.attention_ns = 11000ull;
    decode.attention_device_ns = 7000ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_profile_begin(
            &profile, YVEX_RUNTIME_PROFILE_STAGES,
            YVEX_RUNTIME_PROFILE_DECODE, YVEX_BACKEND_KIND_CUDA,
            profile_id_a, profile_id_b, profile_id_c, profile_id_d,
            profile_id_e, profile_id_f, &err) == YVEX_OK &&
            yvex_runtime_generation_profile_decode(&profile, &decode, &err) ==
                YVEX_OK &&
            profile.counters[YVEX_RUNTIME_PROFILE_KERNEL_LAUNCHES] == 19ull &&
            profile.counters[YVEX_RUNTIME_PROFILE_ACCELERATED_MATRIX_LAUNCHES] == 3ull &&
            profile.phase_ns[YVEX_RUNTIME_PROFILE_ATTENTION] == 11000ull,
        "decode profile preserves launch classes and non-perturbative host timing");
    return 0;
}

typedef struct {
    unsigned int id, prepare_count, publish_count, abort_count;
    unsigned int *events, *event_count;
    int fail_prepare, fail_abort;
} generation_transaction_probe;

static void generation_transaction_event(
    generation_transaction_probe *probe, unsigned int phase)
{
    probe->events[(*probe->event_count)++] = phase + probe->id;
}

static int generation_transaction_prepare(void *opaque, yvex_error *err)
{
    generation_transaction_probe *probe = opaque;
    probe->prepare_count++;
    generation_transaction_event(probe, 10u);
    if (probe->fail_prepare) {
        yvex_error_set(err, YVEX_ERR_STATE, "test.transaction.prepare",
                       "injected transaction prepare failure");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void generation_transaction_publish(void *opaque)
{
    generation_transaction_probe *probe = opaque;
    probe->publish_count++;
    generation_transaction_event(probe, 20u);
}

static int generation_transaction_abort(void *opaque, yvex_error *err)
{
    generation_transaction_probe *probe = opaque;
    probe->abort_count++;
    generation_transaction_event(probe, 30u);
    if (probe->fail_abort) {
        yvex_error_set(err, YVEX_ERR_IO, "test.transaction.abort",
                       "injected transaction abort failure");
        return YVEX_ERR_IO;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int generation_test_transaction_participants(void)
{
    generation_transaction_probe probes[3] = {{0}};
    yvex_runtime_transaction_participant participants[3];
    unsigned int events[12] = {0}, event_count = 0u, index;
    const unsigned int published[] = {11u, 12u, 13u, 21u, 22u, 23u};
    const unsigned int aborted[] = {11u, 12u, 33u, 32u, 31u};
    yvex_error err;
    for (index = 0u; index < 3u; ++index) {
        probes[index].id = index + 1u;
        probes[index].events = events;
        probes[index].event_count = &event_count;
        participants[index] = (yvex_runtime_transaction_participant){
            .context = &probes[index],
            .prepare = generation_transaction_prepare,
            .publish = generation_transaction_publish,
            .abort = generation_transaction_abort};
    }
    YVEX_TEST_ASSERT(
        yvex_runtime_transaction_resolve(
            participants, 3u, YVEX_OK, &err) == YVEX_OK &&
            event_count == sizeof(published) / sizeof(published[0]) &&
            memcmp(events, published, sizeof(published)) == 0,
        "transaction participants prepare before ordered publication");
    for (index = 0u; index < 3u; ++index) {
        probes[index].prepare_count = 0u;
        probes[index].publish_count = 0u;
        probes[index].abort_count = 0u;
    }
    memset(events, 0, sizeof(events));
    event_count = 0u;
    probes[1].fail_prepare = 1;
    YVEX_TEST_ASSERT(
        yvex_runtime_transaction_resolve(
            participants, 3u, YVEX_OK, &err) == YVEX_ERR_STATE &&
            event_count == sizeof(aborted) / sizeof(aborted[0]) &&
            memcmp(events, aborted, sizeof(aborted)) == 0 &&
            !probes[0].publish_count && !probes[1].publish_count &&
            !probes[2].publish_count,
        "prepare failure aborts every participant in reverse order without publication");
    probes[1].fail_prepare = 0;
    probes[2].fail_abort = 1;
    event_count = 0u;
    YVEX_TEST_ASSERT(
        yvex_runtime_transaction_resolve(
            participants, 3u, YVEX_ERR_CANCELLED, &err) == YVEX_ERR_IO,
        "abort failure replaces the primary status with exact cleanup failure");
    YVEX_TEST_ASSERT(
        yvex_runtime_transaction_resolve(
            participants, YVEX_RUNTIME_TRANSACTION_PARTICIPANT_CAP + 1u,
            YVEX_OK, &err) == YVEX_ERR_INVALID_ARG,
        "transaction participant population is bounded");
    return 0;
}

int yvex_test_runtime_generation(void)
{
    if (generation_test_engine_scheduling() != 0) return 1;
    if (generation_test_bounded_batch_coalescing() != 0) return 1;
    if (generation_test_incompatible_arrival_releases_impossible_wait() != 0)
        return 1;
    if (generation_test_active_scheduler_producers() != 0) return 1;
    if (generation_test_ready_progress_scheduling() != 0) return 1;
    if (generation_test_runnable_capacity() != 0) return 1;
    if (generation_test_cooperative_fairness() != 0) return 1;
    if (generation_test_cooperative_drain() != 0) return 1;
    if (generation_test_stop_taxonomy() != 0) return 1;
    if (generation_test_refusals() != 0) return 1;
    if (generation_test_execution_identity_excludes_measurement() != 0) return 1;
    if (generation_test_evidence_sidecar_validation() != 0) return 1;
    if (generation_test_plan_binds_workload_profile() != 0) return 1;
    if (generation_test_decode_profile_projection() != 0) return 1;
    if (generation_test_transaction_participants() != 0) return 1;
    return 0;
}
