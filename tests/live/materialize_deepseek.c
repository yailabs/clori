/*
 * Plan-only mode reads zero tensor payload bytes; live mode reads payload through
 * materialization bindings with one bounded reusable buffer. This runner proves materialization
 * and descriptor readiness only.
 */
#define _POSIX_C_SOURCE 200809L

#include <yvex/internal/artifact.h>
#include <yvex/internal/compiler_source.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <yvex/api.h>

static const char *materialization_status_name(yvex_materialization_status status) {
    static const char *const names[] = {"refused", "planned", "committed", "aborted"};

    return (unsigned int)status < sizeof(names) / sizeof(names[0]) ? names[status] : "refused";
}

typedef struct {
    struct timespec begin;
    unsigned long long last_bytes;
} materialize_live_progress;

static unsigned long long elapsed_ns(const struct timespec *begin,
                                     const struct timespec *end)
{
    if (end->tv_nsec >= begin->tv_nsec)
        return (unsigned long long)(end->tv_sec - begin->tv_sec) *
                   1000000000ull +
               (unsigned long long)(end->tv_nsec - begin->tv_nsec);
    return (unsigned long long)(end->tv_sec - begin->tv_sec - 1) *
               1000000000ull + 1000000000ull -
           (unsigned long long)(begin->tv_nsec - end->tv_nsec);
}

static int path_join_selected(char *out, size_t out_size,
                              const char *models_root)
{
    int written;
    if (!out || !out_size || !models_root) return 0;
    written = snprintf(out, out_size, "%s/deepseek/%s", models_root,
                       YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME);
    return written >= 0 && (size_t)written < out_size;
}

static void print_materialization_failure(const char *phase,
                                          const yvex_materialization_failure *failure,
                                          const yvex_error *err)
{
    fprintf(stderr,
            "%s_failure=%s tensor=%llu name=%s expected=%llu actual=%llu offset=%llu where=%s message=%s\n",
            phase, yvex_materialization_failure_name(failure->code),
            failure->tensor_index, failure->tensor_name, failure->expected,
            failure->actual, failure->offset, yvex_error_where(err),
            yvex_error_message(err));
}

static void print_descriptor_failure(const char *phase,
                                     const yvex_runtime_descriptor_failure *failure,
                                     const yvex_error *err)
{
    fprintf(stderr,
            "%s_failure=%s tensor=%llu name=%s expected=%llu actual=%llu where=%s message=%s\n",
            phase, yvex_runtime_descriptor_failure_name(failure->code),
            failure->tensor_index, failure->tensor_name, failure->expected,
            failure->actual, yvex_error_where(err), yvex_error_message(err));
}

static void progress_report(void *opaque,
                            const yvex_materialization_summary *summary,
                            const yvex_materialized_tensor_binding *binding)
{
    materialize_live_progress *progress =
        (materialize_live_progress *)opaque;
    const unsigned long long interval = 4ull * 1024ull * 1024ull * 1024ull;
    struct timespec now;
    unsigned long long elapsed;
    double rate;
    if (!progress || !summary || !binding) return;
    if (summary->payload_bytes_accessed < summary->payload_bytes &&
        summary->payload_bytes_accessed - progress->last_bytes < interval)
        return;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed = elapsed_ns(&progress->begin, &now);
    rate = elapsed ? (double)summary->payload_bytes_accessed * 1e9 /
                         (double)elapsed : 0.0;
    printf("progress_phase=materialize-walk tensor=%llu name=%s accessed=%llu total=%llu calls=%llu elapsed_seconds=%.3f bytes_per_second=%.3f\n",
           binding->tensor_id, binding->name,
           summary->payload_bytes_accessed, summary->payload_bytes,
           summary->access_calls, (double)elapsed / 1e9, rate);
    fflush(stdout);
    progress->last_bytes = summary->payload_bytes_accessed;
}

static int count_expert_subviews(yvex_materialization_session *session,
                                 unsigned long long *subviews)
{
    unsigned long long count = 0ull;
    unsigned long long i;
    yvex_materialization_failure failure;
    yvex_error err;

    for (i = 0ull; ; ++i) {
        const yvex_materialized_tensor_binding *binding =
            yvex_materialization_session_tensor_at(session, i);
        if (!binding) break;
        if (binding->expert_count > 1ull) {
            yvex_materialized_expert_subview view;
            unsigned long long expert;
            for (expert = 0ull; expert < binding->expert_count; ++expert) {
                if (yvex_materialization_session_expert_subview(
                        session, binding, expert, &view, &failure, &err) !=
                    YVEX_OK) {
                    print_materialization_failure("expert-subview", &failure, &err);
                    return 1;
                }
                count++;
            }
        }
    }
    *subviews = count;
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = "full";
    const char *source_path;
    const char *models_root;
    const char *manifest_path;
    char artifact_path[YVEX_ARTIFACT_PATH_CAP];
    yvex_deepseek_payload_handoff_options handoff_options;
    yvex_deepseek_payload_handoff *handoff = NULL;
    yvex_deepseek_payload_failure handoff_failure;
    yvex_artifact_options artifact_options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_materialization_options options;
    yvex_materialization_projection projection;
    yvex_materialization_plan *plan = NULL;
    yvex_materialization_session *session = NULL;
    yvex_materialization_session *second = NULL;
    yvex_materialization_failure materialization_failure;
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_runtime_descriptor_failure descriptor_failure;
    yvex_semantic_model_ir *semantic_model = NULL;
    const yvex_materialization_summary *plan_summary;
    const yvex_materialization_summary *session_summary;
    const yvex_runtime_descriptor_summary *descriptor_summary;
    unsigned long long expert_subviews = 0ull;
    materialize_live_progress progress;
    yvex_error err;
    int rc;

    if (argc == 5 && strcmp(argv[1], "--plan-only") == 0) {
        mode = "plan-only";
        source_path = argv[2];
        models_root = argv[3];
        manifest_path = argv[4];
    } else if (argc == 4) {
        source_path = argv[1];
        models_root = argv[2];
        manifest_path = argv[3];
    } else {
        fprintf(stderr,
                "usage: materialize_deepseek [--plan-only] SOURCE MODELS_ROOT SOURCE_MANIFEST\n");
        return 2;
    }
    if (!path_join_selected(artifact_path, sizeof(artifact_path), models_root)) {
        fprintf(stderr, "artifact_path_build=fail\n");
        return 1;
    }
    memset(&handoff_options, 0, sizeof(handoff_options));
    handoff_options.source_path = source_path;
    handoff_options.models_root = models_root;
    handoff_options.manifest_path = manifest_path;
    yvex_source_payload_budget_default(&handoff_options.budget);
    handoff_options.budget.maximum_open_handles = 32u;
    handoff_options.budget.maximum_streams = 16u;
    handoff_options.budget.maximum_inflight_host_bytes =
        handoff_options.budget.chunk_bytes *
        handoff_options.budget.maximum_streams;
    handoff_options.chunk_bytes = handoff_options.budget.chunk_bytes;
    handoff_options.page_bytes = handoff_options.budget.page_bytes;
    rc = yvex_model_register_deepseek_v4()->payload.open(
        &handoff, &handoff_options, &handoff_failure, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "handoff_failure=%d where=%s message=%s\n",
                handoff_failure.code, yvex_error_where(&err),
                yvex_error_message(&err));
        return 1;
    }
    rc = yvex_compiler_family_deepseek_v4()->binding_pipeline->semantic_model_build(
        &semantic_model,
        yvex_model_register_deepseek_v4()->payload.verification(handoff), &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "semantic_model_failure where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = artifact_path;
    artifact_options.readonly = 1;
    artifact_options.map = 0;
    rc = yvex_artifact_open(&artifact, &artifact_options, &err);
    if (rc == YVEX_OK &&
        yvex_artifact_size(artifact) != YVEX_SELECTED_DEEPSEEK_FILE_BYTES) {
        fprintf(stderr, "artifact_size_mismatch expected=%llu actual=%llu\n",
                YVEX_SELECTED_DEEPSEEK_FILE_BYTES,
                yvex_artifact_size(artifact));
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "artifact_open_failure where=%s message=%s\n",
                yvex_error_where(&err), yvex_error_message(&err));
        yvex_semantic_model_ir_close(&semantic_model);
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        yvex_tensor_table_close(tensors);
        yvex_gguf_close(gguf);
        yvex_artifact_close(artifact);
        return 1;
    }
    rc = yvex_artifact_admit_deepseek(
        artifact, &admission, &admission_failure, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "admission_failure=%s field=%s expected=%llu actual=%llu message=%s\n",
                yvex_artifact_admission_code_name(admission_failure.code),
                admission_failure.field, admission_failure.expected,
                admission_failure.actual, yvex_error_message(&err));
        goto cleanup_fail;
    }
    yvex_materialization_options_default(&options);
    options.require_terminal_projection = 1;
    options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    options.backend_resident_budget_bytes = 0ull;
    options.future_graph_scratch_reserve_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    options.future_kv_reserve_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    rc = yvex_deepseek_materialization_projection(
        yvex_model_register_deepseek_v4()->payload.map(handoff), &projection, &err);
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &plan, &admission, artifact, gguf, tensors, &projection, &options,
            &materialization_failure, &err);
    if (rc != YVEX_OK) {
        print_materialization_failure("plan", &materialization_failure, &err);
        goto cleanup_fail;
    }
    plan_summary = yvex_materialization_plan_summary(plan);
    printf("mode=%s\n", mode);
    printf("artifact_path=%s\n", artifact_path);
    printf("artifact_identity=%s\n", admission.artifact_identity);
    printf("materialization_plan_identity=%s\n",
           plan_summary->plan_identity);
    printf("tensor_count=%llu\n", plan_summary->tensor_count);
    printf("payload_bytes=%llu\n", plan_summary->payload_bytes);
    printf("file_backed_bytes=%llu\n", plan_summary->file_backed_bytes);
    printf("staged_cache_bytes=%llu\n", plan_summary->staged_cache_bytes);
    printf("backend_candidate_bytes=%llu\n",
           plan_summary->backend_candidate_bytes);
    printf("qtype_0_bytes=%llu\n", plan_summary->qtype_bytes[0]);
    printf("qtype_8_bytes=%llu\n", plan_summary->qtype_bytes[8]);
    printf("qtype_10_bytes=%llu\n", plan_summary->qtype_bytes[10]);
    printf("qtype_30_bytes=%llu\n", plan_summary->qtype_bytes[30]);
    if (strcmp(mode, "plan-only") == 0) {
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &options, &materialization_failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_materialization_session_commit(
                session, &materialization_failure, &err);
        if (rc != YVEX_OK) {
            print_materialization_failure("commit", &materialization_failure, &err);
            goto cleanup_fail;
        }
    } else {
        memset(&progress, 0, sizeof(progress));
        (void)clock_gettime(CLOCK_MONOTONIC, &progress.begin);
        rc = yvex_materialization_session_open(
            &session, plan, artifact, &options, &materialization_failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_materialization_session_commit(
                session, &materialization_failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_materialization_session_walk_payload(
                session, progress_report, &progress,
                &materialization_failure, &err);
        if (rc != YVEX_OK) {
            print_materialization_failure("live-walk", &materialization_failure, &err);
            goto cleanup_fail;
        }
        rc = yvex_materialization_session_open(
            &second, plan, artifact, &options, &materialization_failure, &err);
        if (rc == YVEX_OK)
            rc = yvex_materialization_session_commit(
                second, &materialization_failure, &err);
        if (rc != YVEX_OK) {
            print_materialization_failure("second-lifecycle", &materialization_failure, &err);
            goto cleanup_fail;
        }
    }
    if (count_expert_subviews(session, &expert_subviews) != 0)
        goto cleanup_fail;
    rc = yvex_runtime_descriptor_build_deepseek(
        &descriptor, &admission, session,
        yvex_model_register_deepseek_v4()->payload.map(handoff), semantic_model,
        &descriptor_failure, &err);
    if (rc != YVEX_OK) {
        print_descriptor_failure("descriptor", &descriptor_failure, &err);
        goto cleanup_fail;
    }
    session_summary = yvex_materialization_session_summary(session);
    descriptor_summary = yvex_runtime_descriptor_summary_get(descriptor);
    printf("materialization_status=%s\n",
           materialization_status_name(session_summary->status));
    printf("payload_bytes_accessed=%llu\n",
           session_summary->payload_bytes_accessed);
    printf("access_calls=%llu\n", session_summary->access_calls);
    printf("peak_executor_owned_bytes=%llu\n",
           session_summary->peak_executor_owned_bytes);
    printf("expert_subviews=%llu\n", expert_subviews);
    printf("runtime_descriptor_status=%u\n", (unsigned int)descriptor_summary->status);
    printf("runtime_descriptor_identity=%s\n",
           descriptor_summary->runtime_descriptor_identity);
    printf("runtime_numeric_identity=%s\n",
           descriptor_summary->runtime_numeric_identity);
    printf("runtime_numeric_schema_version=%u\n",
           descriptor_summary->runtime_numeric_schema_version);
    printf("runtime_hadamard_revision=%s\n",
           descriptor_summary->runtime_hadamard_revision);
    printf("runtime_tensor_count=%llu\n", descriptor_summary->tensor_count);
    printf("runtime_missing_bindings=%llu\n",
           descriptor_summary->missing_required_bindings);
    printf("runtime_duplicate_bindings=%llu\n",
           descriptor_summary->duplicate_bindings);
    printf("runtime_generation_ready=%d\n",
           descriptor_summary->generation_ready);

    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(second);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    yvex_semantic_model_ir_close(&semantic_model);
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return 0;

cleanup_fail:
    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(second);
    yvex_materialization_session_close(session);
    yvex_materialization_plan_close(plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    yvex_semantic_model_ir_close(&semantic_model);
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return 1;
}
