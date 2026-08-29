/* Engine-owned scheduling facts for compiler-sealed executable work. */
#ifndef INCLUDE_YVEX_INTERNAL_ENGINE_SCHEDULER_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_ENGINE_SCHEDULER_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/execution_batch.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_ENGINE_PROGRESS_PREFILL = 0,
    YVEX_ENGINE_PROGRESS_DECODE,
    YVEX_ENGINE_PROGRESS_DRAFT,
    YVEX_ENGINE_PROGRESS_VERIFY,
    YVEX_ENGINE_PROGRESS_CORRECTION,
    YVEX_ENGINE_PROGRESS_PUBLICATION,
    YVEX_ENGINE_PROGRESS_COMPONENT,
    YVEX_ENGINE_PROGRESS_KIND_COUNT
} yvex_engine_progress_kind;

struct yvex_model_engine;

typedef struct {
    int enabled;
    unsigned long long admitted_maximum_width, sequence_capacity;
    unsigned long long ready_sequence_work, active_sequences;
    unsigned long long maximum_ready_sequence_work, maximum_active_sequences;
    unsigned long long progress_submissions, progress_transitions;
    unsigned long long progress_completions, progress_cancellations;
    unsigned long long progress_failures;
    unsigned long long sequence_conflicts;
    unsigned long long progress_submissions_by_kind[
        YVEX_ENGINE_PROGRESS_KIND_COUNT];
    unsigned long long progress_completions_by_kind[
        YVEX_ENGINE_PROGRESS_KIND_COUNT];
    unsigned long long submissions, physical_batches, multi_source_batches;
    unsigned long long submitted_rows, executed_rows, maximum_width;
    unsigned long long multi_source_rows, maximum_multi_source_width;
    unsigned long long maximum_source_count;
    unsigned long long multi_source_worklists, multi_source_expert_pairs;
    unsigned long long maximum_multi_source_bucket_population;
    unsigned long long multi_source_matrix_tile_eligible_pairs;
    unsigned long long multi_source_matrix_tile_executed_pairs;
    unsigned long long multi_source_narrow_pairs;
    unsigned long long multi_source_population_histogram[
        YVEX_EXPERT_WORKLIST_HISTOGRAM_CAP];
    unsigned long long cancellations, failures, active, queued;
    unsigned long long registered_producers, coalescing_limit_ns, coalescing_waits;
    unsigned long long coalescing_timeouts, coalescing_ns;
    unsigned long long compatibility_candidates, compatibility_mismatches;
    unsigned long long phase_mismatches, layer_mismatches, operation_mismatches;
    unsigned long long geometry_mismatches, profile_mismatches, identity_mismatches;
    unsigned long long rendezvous_limit_ns, rendezvous_submissions;
    unsigned long long rendezvous_steps, multi_source_rendezvous;
    unsigned long long maximum_rendezvous_width;
    unsigned long long submissions_by_phase[YVEX_EXECUTION_PHASE_COUNT];
    unsigned long long physical_batches_by_phase[YVEX_EXECUTION_PHASE_COUNT];
    unsigned long long executed_rows_by_phase[YVEX_EXECUTION_PHASE_COUNT];
    unsigned long long rendezvous_steps_by_phase[YVEX_EXECUTION_PHASE_COUNT];
} yvex_engine_scheduler_summary;

int yvex_model_engine_scheduler_maximum_width_copy(
    const struct yvex_model_engine *model, unsigned long long *width,
    yvex_error *err);
int yvex_model_engine_scheduler_summary_copy(
    const struct yvex_model_engine *model,
    yvex_engine_scheduler_summary *out, yvex_error *err);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_ENGINE_SCHEDULER_H_INCLUDED */
