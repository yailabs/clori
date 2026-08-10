/*
 * Plan-only mode reads zero payload and creates no output file; live output is external and
 * publication follows every validation gate. This runner proves complete artifacts but never
 * materializes them.
 */
#define _POSIX_C_SOURCE 200809L
#include <yvex/internal/artifact.h>
#include <yvex/internal/compiler_source.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/operator_graph.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/families/deepseek_v4.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define SELECTED_EXECUTION_IDENTITY \
    "b81f3c5d670737bf20c938e635a1bffdbb0d60f885f994225a02225bb7ba51db"
#define ARTIFACT_WORKERS 16u
#define ARTIFACT_EXECUTOR_BUDGET (64u * 1024u * 1024u)

typedef struct {
    yvex_gguf_file_sink *sink;
    unsigned long long total_bytes;
    struct timespec begin;
    atomic_int stop;
} artifact_progress_context;

static const char *artifact_class_name(yvex_artifact_class artifact_class) {
    switch (artifact_class) {
    case YVEX_ARTIFACT_CLASS_TENSOR_PROOF:
        return "tensor-proof-artifact";
    case YVEX_ARTIFACT_CLASS_EXTERNAL_UNADMITTED:
        return "external-unadmitted-artifact";
    case YVEX_ARTIFACT_CLASS_COMPLETE_YVEX:
        return "complete-yvex-artifact";
    default:
        return "refused";
    }
}

typedef struct {
    struct timespec begin;
    unsigned long long last_reported_bytes;
} artifact_roundtrip_progress_context;

typedef struct {
    yvex_quant_execution_summary execution;
    yvex_gguf_file_sink_summary emission;
    yvex_gguf_roundtrip_summary roundtrip;
    yvex_artifact_official_reader_fact official;
    yvex_complete_artifact_admission admission;
    char profile_name[64];
    char profile_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char writer_plan_identity[YVEX_GGUF_WRITER_IDENTITY_CAP];
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char execution_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char path[YVEX_ARTIFACT_PATH_CAP];
    unsigned long long quantize_write_elapsed_ns;
    unsigned long long finalize_elapsed_ns;
    unsigned long long native_roundtrip_elapsed_ns;
    unsigned long long official_reader_elapsed_ns;
    unsigned long long publish_admission_elapsed_ns;
    int published;
    int complete;
} artifact_live_result;

static unsigned long long artifact_elapsed_ns(const struct timespec *begin,
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

static int artifact_writer_plan_build(
    yvex_gguf_writer_plan **out, const yvex_quant_plan *quant,
    const yvex_deepseek_payload_handoff *handoff,
    const yvex_gguf_writer_plan_options *options,
    yvex_gguf_writer_failure *failure, yvex_error *error)
{
    yvex_gguf_writer_plan_request request;

    memset(&request, 0, sizeof(request));
    request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    request.quant_plan = quant;
    request.options = options;
    request.input.complete.lowering = yvex_model_deepseek_writer_lowering_api();
    request.input.complete.lowering_context =
        yvex_model_register_deepseek_v4()->payload.map(handoff);
    request.input.complete.verification =
        yvex_model_register_deepseek_v4()->payload.verification(handoff);
    return yvex_gguf_writer_plan_build(out, &request, failure, error);
}

static int artifact_path_build(char *out, size_t out_bytes,
                               const char *models_root,
                               const char *basename)
{
    int written;
    if (!out || !out_bytes || !models_root || !basename) return 0;
    written = snprintf(out, out_bytes, "%s/deepseek/%s",
                       models_root, basename);
    return written >= 0 && (size_t)written < out_bytes;
}

static void *artifact_progress_entry(void *opaque)
{
    artifact_progress_context *context =
        (artifact_progress_context *)opaque;
    unsigned int seconds = 0u;
    while (!atomic_load_explicit(&context->stop, memory_order_relaxed)) {
        struct timespec delay = {1, 0};
        yvex_gguf_file_sink_summary summary;
        struct timespec now;
        unsigned long long elapsed;
        double rate;
        double eta;
        (void)nanosleep(&delay, NULL);
        if (atomic_load_explicit(&context->stop, memory_order_relaxed)) break;
        seconds++;
        if (seconds % 30u != 0u ||
            yvex_gguf_file_sink_summary_get(context->sink, &summary) !=
                YVEX_OK)
            continue;
        (void)clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed = artifact_elapsed_ns(&context->begin, &now);
        rate = elapsed ? (double)summary.encoded_bytes_written * 1e9 /
                             (double)elapsed : 0.0;
        eta = rate > 0.0 && summary.encoded_bytes_written < context->total_bytes
            ? (double)(context->total_bytes -
                       summary.encoded_bytes_written) / rate : 0.0;
        printf("progress_phase=quantize-write committed=%llu source_ranges=%llu source_bytes=%llu encoded=%llu physical_file_bytes=%llu planned_file_bytes=%llu chunks=%llu elapsed_seconds=%.3f bytes_per_second=%.3f eta_seconds=%.3f\n",
               summary.committed_terminals,
               summary.source_ranges_committed,
               summary.source_bytes_committed,
               summary.encoded_bytes_written,
               summary.physical_write_bytes,
               summary.planned_file_bytes,
               summary.output_chunks, (double)elapsed / 1e9, rate, eta);
        fflush(stdout);
    }
    return NULL;
}

static void artifact_roundtrip_progress(
    void *opaque,
    const yvex_gguf_roundtrip_summary *summary,
    unsigned long long planned_file_bytes)
{
    artifact_roundtrip_progress_context *context =
        (artifact_roundtrip_progress_context *)opaque;
    const unsigned long long interval = 4ull * 1024ull * 1024ull * 1024ull;
    struct timespec now;
    unsigned long long elapsed;
    double rate;
    double eta;

    if (!context || !summary || !planned_file_bytes ||
        (summary->bytes_hashed < planned_file_bytes &&
         summary->bytes_hashed - context->last_reported_bytes < interval))
        return;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed = artifact_elapsed_ns(&context->begin, &now);
    rate = elapsed ? (double)summary->bytes_hashed * 1e9 /
                         (double)elapsed : 0.0;
    eta = rate > 0.0 && summary->bytes_hashed < planned_file_bytes
        ? (double)(planned_file_bytes - summary->bytes_hashed) / rate : 0.0;
    printf("progress_phase=native-roundtrip terminals=%llu hashed=%llu total=%llu read_calls=%llu elapsed_seconds=%.3f bytes_per_second=%.3f eta_seconds=%.3f\n",
           summary->terminals_verified, summary->bytes_hashed,
           planned_file_bytes, summary->read_calls,
           (double)elapsed / 1e9, rate, eta);
    fflush(stdout);
    context->last_reported_bytes = summary->bytes_hashed;
}

static void artifact_print_quant_failure(const char *phase,
                                         const yvex_quant_failure *failure,
                                         const yvex_error *error)
{
    fprintf(stderr,
            "%s_failure=%u terminal=%llu source=%llu row=%llu block=%llu expected=%llu actual=%llu qtype=%u operation=%u status=%s where=%s message=%s\n",
            phase, (unsigned int)failure->code,
            failure->terminal_ordinal, failure->source_index,
            failure->row_index, failure->block_index, failure->expected,
            failure->actual, failure->qtype,
            (unsigned int)failure->operation,
            yvex_status_name(yvex_error_code(error)),
            yvex_error_where(error), yvex_error_message(error));
}

static int artifact_plan_compatibility(
    const char *artifact_path,
    const yvex_gguf_writer_plan *writer)
{
    yvex_artifact_options options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_complete_artifact_admission admission;
    yvex_artifact_admission_failure admission_failure;
    yvex_artifact_physical_compatibility compatibility;
    yvex_artifact_compatibility_failure failure;
    yvex_error error;
    int rc;

    memset(&options, 0, sizeof(options));
    memset(&admission_failure, 0, sizeof(admission_failure));
    memset(&failure, 0, sizeof(failure));
    options.path = artifact_path;
    options.readonly = 1;
    yvex_error_clear(&error);
    rc = yvex_artifact_open(&artifact, &options, &error);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &error);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admit_deepseek(
            artifact, &admission, &admission_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_artifact_admission_identity_verify(
            artifact, &admission, NULL, NULL, &admission_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_artifact_physical_compatibility_validate(
            writer, &admission, artifact, gguf, &compatibility, &failure,
            &error);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "physical_compatibility_failure=%u field=%s tensor=%llu name=%s expected=%llu actual=%llu where=%s message=%s\n",
                (unsigned int)failure.code, failure.field,
                failure.tensor_index, failure.tensor_name, failure.expected,
                failure.actual, yvex_error_where(&error),
                yvex_error_message(&error));
    } else {
        printf("physical_payload_compatible=%d\n",
               compatibility.physical_payload_compatible);
        printf("artifact_rebuild_required=%d\n",
               compatibility.artifact_rebuild_required);
        printf("materialization_rebuild_required=%d\n",
               compatibility.materialization_rebuild_required);
        printf("tensor_inventory_equal=%d\n",
               compatibility.tensor_inventory_equal);
        printf("qtype_equal=%d\n", compatibility.qtype_equal);
        printf("layout_equal=%d\n", compatibility.layout_equal);
        printf("offset_equal=%d\n", compatibility.offset_equal);
        printf("payload_digest_equal=%d\n",
               compatibility.payload_digest_equal);
        printf("compatibility_tensors_compared=%llu\n",
               compatibility.tensors_compared);
        printf("compatibility_payload_bytes_read=%llu\n",
               compatibility.payload_bytes_read);
        printf("compatibility_writer_plan_identity=%s\n",
               compatibility.writer_plan_identity);
        printf("compatibility_artifact_identity=%s\n",
               compatibility.artifact_identity);
    }
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}

static int artifact_plan_one(
    const yvex_deepseek_payload_handoff *handoff,
    yvex_quant_profile_kind profile,
    const char *execution_identity,
    const char *artifact_path)
{
    yvex_quant_plan *quant = NULL;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_gguf_writer_plan *repeat_writer = NULL;
    yvex_quant_failure quant_failure;
    yvex_gguf_writer_failure writer_failure;
    yvex_gguf_writer_plan_options writer_options;
    yvex_error error;
    const yvex_quant_plan_summary *quant_summary;
    const yvex_gguf_writer_plan_summary *writer_summary;
    const yvex_gguf_writer_plan_summary *repeat_summary;
    const unsigned char *prefix;
    const unsigned char *repeat_prefix;
    size_t prefix_bytes;
    size_t repeat_prefix_bytes;
    unsigned int qtype;
    int rc;

    yvex_error_clear(&error);
    rc = yvex_quant_plan_build_deepseek_profile(
        &quant, yvex_model_register_deepseek_v4()->payload.transform_ir(handoff),
        yvex_model_register_deepseek_v4()->payload.binding(handoff),
        yvex_model_register_deepseek_v4()->payload.map(handoff), profile, NULL,
        &quant_failure, &error);
    if (rc != YVEX_OK) {
        artifact_print_quant_failure("quant-plan", &quant_failure, &error);
        return 1;
    }
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_options.required_execution_identity = execution_identity;
    rc = artifact_writer_plan_build(
        &writer, quant, handoff, &writer_options, &writer_failure, &error);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "writer_plan_failure=%u metadata=%llu tensor=%llu name=%s expected=%llu actual=%llu status=%s where=%s message=%s\n",
                (unsigned int)writer_failure.code,
                writer_failure.metadata_index, writer_failure.tensor_index,
                writer_failure.name, writer_failure.expected,
                writer_failure.actual,
                yvex_status_name(yvex_error_code(&error)),
                yvex_error_where(&error), yvex_error_message(&error));
        yvex_quant_plan_release(&quant);
        return 1;
    }
    quant_summary = yvex_quant_plan_summary_get(quant);
    writer_summary = yvex_gguf_writer_plan_summary_get(writer);
    rc = artifact_writer_plan_build(
        &repeat_writer, quant, handoff, &writer_options, &writer_failure, &error);
    repeat_summary = yvex_gguf_writer_plan_summary_get(repeat_writer);
    prefix = yvex_gguf_writer_plan_prefix(writer, &prefix_bytes);
    repeat_prefix = yvex_gguf_writer_plan_prefix(
        repeat_writer, &repeat_prefix_bytes);
    if (rc != YVEX_OK || !writer_summary || !repeat_summary || !prefix ||
        !repeat_prefix || prefix_bytes != repeat_prefix_bytes ||
        memcmp(prefix, repeat_prefix, prefix_bytes) != 0 ||
        strcmp(writer_summary->writer_plan_identity,
               repeat_summary->writer_plan_identity) != 0) {
        fprintf(stderr,
                "writer_plan_determinism_failure first=%s repeat=%s\n",
                writer_summary ? writer_summary->writer_plan_identity : "",
                repeat_summary ? repeat_summary->writer_plan_identity : "");
        yvex_gguf_writer_plan_release(&repeat_writer);
        yvex_gguf_writer_plan_release(&writer);
        yvex_quant_plan_release(&quant);
        return 1;
    }
    printf("profile_name=%s\n", quant_summary->profile_name);
    printf("profile_identity=%s\n", quant_summary->profile_identity);
    printf("writer_plan_identity=%s\n",
           writer_summary->writer_plan_identity);
    printf("writer_plan_repeat_identity=%s\n",
           repeat_summary->writer_plan_identity);
    printf("writer_plan_deterministic=1\n");
    printf("metadata_count=%llu\n", writer_summary->metadata_count);
    printf("tokenizer_tokens=%llu\n",
           writer_summary->tokenizer_token_count);
    printf("tokenizer_merges=%llu\n",
           writer_summary->tokenizer_merge_count);
    printf("structural_bytes=%llu\n", writer_summary->structural_bytes);
    printf("pre_data_padding_bytes=%llu\n",
           writer_summary->pre_data_padding_bytes);
    printf("tensor_payload_bytes=%llu\n",
           writer_summary->tensor_payload_bytes);
    printf("tensor_padding_bytes=%llu\n",
           writer_summary->tensor_padding_bytes);
    printf("predicted_file_bytes=%llu\n",
           writer_summary->final_file_bytes);
    printf("payload_bytes_read=%llu\n", writer_summary->payload_bytes_read);
    printf("writer_plan_owned_bytes=%llu\n", writer_summary->owned_bytes);
    for (qtype = 0u; qtype <= YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID;
         ++qtype) {
        if (!writer_summary->qtype_tensor_counts[qtype]) continue;
        printf("profile_qtype_%u_name=%s\n", qtype,
               yvex_gguf_qtype_name(qtype));
        printf("profile_qtype_%u_tensors=%llu\n", qtype,
               writer_summary->qtype_tensor_counts[qtype]);
        printf("profile_qtype_%u_bytes=%llu\n", qtype,
               writer_summary->qtype_payload_bytes[qtype]);
    }
    if (artifact_path && artifact_plan_compatibility(artifact_path, writer) != 0)
        rc = YVEX_ERR_FORMAT;
    yvex_gguf_writer_plan_release(&repeat_writer);
    yvex_gguf_writer_plan_release(&writer);
    yvex_quant_plan_release(&quant);
    return rc == YVEX_OK ? 0 : 1;
}

/* Verifies one official-reader process against an unchanged regular inode. */
static int artifact_official_check(
    const char *checker,
    const char *path,
    const yvex_gguf_writer_plan_summary *plan,
    yvex_artifact_official_reader_fact *out)
{
    char file_size[32];
    char metadata_count[32];
    char tensor_count[32];
    char token_count[32];
    char merge_count[32];
    struct stat before;
    struct stat after;
    pid_t child;
    int status = 0;
    int waited = 0;

    memset(out, 0, sizeof(*out));
    if (!checker || !checker[0] || !path || !plan ||
        stat(path, &before) != 0 || !S_ISREG(before.st_mode)) return 0;
    (void)snprintf(file_size, sizeof(file_size), "%llu",
                   plan->final_file_bytes);
    (void)snprintf(metadata_count, sizeof(metadata_count), "%llu",
                   plan->metadata_count);
    (void)snprintf(tensor_count, sizeof(tensor_count), "%llu",
                   plan->tensor_count);
    (void)snprintf(token_count, sizeof(token_count), "%llu",
                   plan->tokenizer_token_count);
    (void)snprintf(merge_count, sizeof(merge_count), "%llu",
                   plan->tokenizer_merge_count);
    child = fork();
    if (child < 0) return 0;
    if (child == 0) {
        execl(checker, checker, path, file_size, metadata_count, tensor_count,
              token_count, merge_count, (char *)NULL);
        _exit(127);
    }
    do {
        if (waitpid(child, &status, 0) >= 0) {
            waited = 1;
            break;
        }
    } while (errno == EINTR);
    if (!waited || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        stat(path, &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
        return 0;
    (void)snprintf(out->revision, sizeof(out->revision), "%s",
                   YVEX_GGUF_OFFICIAL_READER_REVISION);
    out->metadata_count = plan->metadata_count;
    out->tensor_count = plan->tensor_count;
    out->file_bytes = plan->final_file_bytes;
    out->file_device = (unsigned long long)after.st_dev;
    out->file_inode = (unsigned long long)after.st_ino;
    out->file_mtime_seconds = (long long)after.st_mtim.tv_sec;
    out->file_mtime_nanoseconds = (long long)after.st_mtim.tv_nsec;
    out->file_ctime_seconds = (long long)after.st_ctim.tv_sec;
    out->file_ctime_nanoseconds = (long long)after.st_ctim.tv_nsec;
    out->accepted = 1;
    return 1;
}

static int artifact_structure_one(
    const yvex_deepseek_payload_handoff *handoff,
    yvex_quant_profile_kind profile,
    const char *execution_identity,
    const char *checker)
{
    yvex_quant_plan *quant = NULL;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_quant_failure quant_failure;
    yvex_gguf_writer_failure writer_failure;
    yvex_gguf_writer_plan_options writer_options;
    yvex_artifact_official_reader_fact official;
    yvex_error error;
    const yvex_gguf_writer_plan_summary *summary;
    const unsigned char *prefix;
    size_t prefix_bytes;
    size_t delivered = 0u;
    char path[256] = {0};
    int fd = -1;
    int rc;

    yvex_error_clear(&error);
    rc = yvex_quant_plan_build_deepseek_profile(
        &quant, yvex_model_register_deepseek_v4()->payload.transform_ir(handoff),
        yvex_model_register_deepseek_v4()->payload.binding(handoff),
        yvex_model_register_deepseek_v4()->payload.map(handoff), profile, NULL,
        &quant_failure, &error);
    if (rc != YVEX_OK) goto cleanup;
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_options.required_execution_identity = execution_identity;
    rc = artifact_writer_plan_build(
        &writer, quant, handoff, &writer_options, &writer_failure, &error);
    if (rc != YVEX_OK) goto cleanup;
    summary = yvex_gguf_writer_plan_summary_get(writer);
    prefix = yvex_gguf_writer_plan_prefix(writer, &prefix_bytes);
    {
        int written = snprintf(
            path, sizeof(path), "/tmp/yvex-gguf-structure-%ld-%u.gguf",
            (long)getpid(), (unsigned int)profile);
        if (!summary || !prefix || summary->final_file_bytes > LLONG_MAX ||
            written < 0 || (size_t)written >= sizeof(path)) {
            rc = YVEX_ERR_BOUNDS;
            goto cleanup;
        }
    }
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        rc = YVEX_ERR_IO;
        goto cleanup;
    }
    while (delivered < prefix_bytes) {
        ssize_t wrote = pwrite(fd, prefix + delivered,
                               prefix_bytes - delivered,
                               (off_t)delivered);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) {
            rc = YVEX_ERR_IO;
            goto cleanup;
        }
        delivered += (size_t)wrote;
    }
    if (ftruncate(fd, (off_t)summary->final_file_bytes) != 0 ||
        fsync(fd) != 0) {
        rc = YVEX_ERR_IO;
        goto cleanup;
    }
    if (close(fd) != 0) {
        fd = -1;
        rc = YVEX_ERR_IO;
        goto cleanup;
    }
    fd = -1;
    if (!artifact_official_check(checker, path, summary, &official)) {
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    printf("sparse_structure_profile=%s file_bytes=%llu official_revision=%s result=accepted\n",
           summary->profile_name, summary->final_file_bytes,
           official.revision);
    rc = YVEX_OK;

cleanup:
    if (fd >= 0) close(fd);
    if (path[0]) (void)unlink(path);
    yvex_gguf_writer_plan_release(&writer);
    yvex_quant_plan_release(&quant);
    return rc == YVEX_OK ? 0 : 1;
}

static int artifact_execute_one(
    yvex_deepseek_payload_handoff *handoff,
    yvex_quant_profile_kind profile,
    const yvex_quant_plan *provided_quant,
    const yvex_imatrix_data *imatrix,
    const char *required_execution_identity,
    const char *destination,
    const char *checker,
    int publish,
    artifact_live_result *result)
{
    yvex_quant_plan *owned_quant = NULL;
    const yvex_quant_plan *quant = provided_quant;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_gguf_file_sink *file_sink = NULL;
    yvex_quant_output_sink output_sink;
    yvex_quant_executor_options executor_options;
    yvex_quant_failure quant_failure;
    yvex_gguf_writer_failure writer_failure;
    yvex_gguf_file_failure file_failure;
    yvex_gguf_roundtrip_failure roundtrip_failure;
    yvex_artifact_admission_failure admission_failure;
    yvex_gguf_writer_plan_options writer_options;
    yvex_gguf_file_sink_options file_options;
    yvex_gguf_roundtrip_options roundtrip_options;
    artifact_roundtrip_progress_context roundtrip_progress;
    yvex_artifact_admission_request admission_request;
    artifact_progress_context progress;
    pthread_t progress_thread;
    const yvex_quant_plan_summary *quant_summary;
    const yvex_gguf_writer_plan_summary *writer_summary;
    yvex_error error;
    struct timespec begin;
    struct timespec end;
    struct timespec phase_begin;
    struct timespec phase_end;
    char temporary_path[YVEX_ARTIFACT_PATH_CAP];
    yvex_gguf_file_sink_summary preflight;
    int progress_started = 0;
    int rc;

    memset(result, 0, sizeof(*result));
    yvex_error_clear(&error);
    rc = quant ? YVEX_OK : yvex_quant_plan_build_deepseek_profile(
        &owned_quant, yvex_model_register_deepseek_v4()->payload.transform_ir(handoff),
        yvex_model_register_deepseek_v4()->payload.binding(handoff),
        yvex_model_register_deepseek_v4()->payload.map(handoff), profile, NULL,
        &quant_failure, &error);
    if (!quant) quant = owned_quant;
    if (rc != YVEX_OK) {
        artifact_print_quant_failure("quant-plan", &quant_failure, &error);
        goto cleanup;
    }
    quant_summary = yvex_quant_plan_summary_get(quant);
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_options.required_execution_identity = required_execution_identity;
    rc = artifact_writer_plan_build(
        &writer, quant, handoff, &writer_options, &writer_failure, &error);
    if (rc != YVEX_OK) {
        fprintf(stderr, "writer_plan_failure=%u tensor=%llu message=%s\n",
                (unsigned int)writer_failure.code,
                writer_failure.tensor_index, yvex_error_message(&error));
        goto cleanup;
    }
    writer_summary = yvex_gguf_writer_plan_summary_get(writer);
    yvex_gguf_file_sink_options_default(&file_options);
    file_options.destination_path = destination;
    printf("emission_preflight_profile=%s destination=%s expected_payload=%llu predicted_file=%llu\n",
           quant_summary->profile_name, destination,
           writer_summary->tensor_payload_bytes,
           writer_summary->final_file_bytes);
    fflush(stdout);
    rc = yvex_gguf_file_sink_create(
        &file_sink, writer, quant, &file_options, &file_failure, &error);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "file_sink_create_failure=%u system=%d expected=%llu actual=%llu message=%s\n",
                (unsigned int)file_failure.code,
                file_failure.system_error, file_failure.expected,
                file_failure.actual, yvex_error_message(&error));
        goto cleanup;
    }
    (void)snprintf(temporary_path, sizeof(temporary_path), "%s",
                   yvex_gguf_file_sink_temporary_path(file_sink));
    memset(&preflight, 0, sizeof(preflight));
    if (yvex_gguf_file_sink_summary_get(file_sink, &preflight) != YVEX_OK) {
        fprintf(stderr, "file_preflight_summary=unavailable\n");
        rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    printf("emission_preflight_available_bytes=%llu required_safe_bytes=%llu preallocated=%d destination_conflict=0 filesystem_device=%llu temporary_path=%s\n",
           preflight.available_bytes, preflight.required_safe_bytes,
           preflight.preallocated, preflight.file_device, temporary_path);
    fflush(stdout);
    yvex_gguf_file_sink_adapter(file_sink, &output_sink);
    yvex_quant_executor_options_default(&executor_options);
    executor_options.worker_count = ARTIFACT_WORKERS;
    executor_options.maximum_owned_bytes = ARTIFACT_EXECUTOR_BUDGET;
    executor_options.imatrix = imatrix;
    memset(&progress, 0, sizeof(progress));
    progress.sink = file_sink;
    progress.total_bytes = writer_summary->tensor_payload_bytes;
    atomic_init(&progress.stop, 0);
    (void)clock_gettime(CLOCK_MONOTONIC, &begin);
    progress.begin = begin;
    if (pthread_create(&progress_thread, NULL,
                       artifact_progress_entry, &progress) == 0)
        progress_started = 1;
    rc = yvex_quant_execute(
        quant, &output_sink, &executor_options, &result->execution,
        &quant_failure, &error);
    atomic_store_explicit(&progress.stop, 1, memory_order_relaxed);
    if (progress_started) (void)pthread_join(progress_thread, NULL);
    (void)clock_gettime(CLOCK_MONOTONIC, &end);
    result->quantize_write_elapsed_ns = artifact_elapsed_ns(&begin, &end);
    if (rc != YVEX_OK) {
        artifact_print_quant_failure("quant-execute", &quant_failure, &error);
        goto cleanup;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &phase_begin);
    rc = yvex_gguf_file_sink_finalize(
        file_sink, &result->emission, &file_failure, &error);
    (void)clock_gettime(CLOCK_MONOTONIC, &phase_end);
    result->finalize_elapsed_ns = artifact_elapsed_ns(
        &phase_begin, &phase_end);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "file_finalize_failure=%u terminal=%llu expected=%llu actual=%llu message=%s\n",
                (unsigned int)file_failure.code,
                file_failure.terminal_ordinal, file_failure.expected,
                file_failure.actual, yvex_error_message(&error));
        goto cleanup;
    }
    memset(&roundtrip_progress, 0, sizeof(roundtrip_progress));
    (void)clock_gettime(CLOCK_MONOTONIC, &roundtrip_progress.begin);
    yvex_gguf_roundtrip_options_default(&roundtrip_options);
    roundtrip_options.progress = artifact_roundtrip_progress;
    roundtrip_options.progress_context = &roundtrip_progress;
    (void)clock_gettime(CLOCK_MONOTONIC, &phase_begin);
    rc = yvex_gguf_roundtrip_validate(
        temporary_path, writer, yvex_gguf_file_sink_digest(file_sink),
        &roundtrip_options, &result->roundtrip, &roundtrip_failure, &error);
    (void)clock_gettime(CLOCK_MONOTONIC, &phase_end);
    result->native_roundtrip_elapsed_ns = artifact_elapsed_ns(
        &phase_begin, &phase_end);
    if (rc != YVEX_OK) {
        fprintf(stderr,
                "native_roundtrip_failure=%u tensor=%llu offset=%llu expected=%llu actual=%llu message=%s\n",
                (unsigned int)roundtrip_failure.code,
                roundtrip_failure.tensor_index,
                roundtrip_failure.file_offset, roundtrip_failure.expected,
                roundtrip_failure.actual, yvex_error_message(&error));
        goto cleanup;
    }
    if (publish) (void)clock_gettime(CLOCK_MONOTONIC, &phase_begin);
    if (publish && !artifact_official_check(
            checker, temporary_path, writer_summary, &result->official)) {
        fprintf(stderr, "official_reader_failure=refused path=%s revision=%s\n",
                temporary_path, YVEX_GGUF_OFFICIAL_READER_REVISION);
        rc = YVEX_ERR_FORMAT;
        goto cleanup;
    }
    if (publish) {
        (void)clock_gettime(CLOCK_MONOTONIC, &phase_end);
        result->official_reader_elapsed_ns = artifact_elapsed_ns(
            &phase_begin, &phase_end);
    }
    if (publish) {
        (void)clock_gettime(CLOCK_MONOTONIC, &phase_begin);
        rc = yvex_gguf_file_sink_publish(
            file_sink, &result->roundtrip, &result->emission,
            &file_failure, &error);
        if (rc != YVEX_OK) {
            fprintf(stderr, "file_publish_failure=%u message=%s\n",
                    (unsigned int)file_failure.code,
                    yvex_error_message(&error));
            goto cleanup;
        }
        memset(&admission_request, 0, sizeof(admission_request));
        admission_request.artifact_path = destination;
        admission_request.writer_plan = writer;
        admission_request.emission = &result->emission;
        admission_request.native_roundtrip = &result->roundtrip;
        admission_request.official_reader = &result->official;
        rc = yvex_complete_artifact_admit(
            &admission_request, &result->admission,
            &admission_failure, &error);
        if (rc != YVEX_OK) {
            fprintf(stderr,
                    "artifact_admission_failure=%s field=%s expected=%llu actual=%llu message=%s\n",
                    yvex_artifact_admission_code_name(admission_failure.code),
                    admission_failure.field, admission_failure.expected,
                    admission_failure.actual, yvex_error_message(&error));
            if (yvex_gguf_file_sink_withdraw(
                    file_sink, &file_failure, &error) != YVEX_OK)
                fprintf(stderr, "artifact_withdraw_failure=%u message=%s\n",
                        (unsigned int)file_failure.code,
                        yvex_error_message(&error));
            goto cleanup;
        }
        {
            yvex_artifact_descriptor_fact descriptor;
            yvex_model_complete_artifact_gate_fact model_gate;
            if (!yvex_artifact_descriptor_from_admission(
                    &result->admission, &descriptor) ||
                !descriptor.materialization_input_ready ||
                descriptor.runtime_supported ||
                yvex_model_artifact_gate_from_admission(
                    &result->admission, &model_gate, &error) != YVEX_OK ||
                !model_gate.complete_artifact_admitted ||
                model_gate.execution_ready) {
                fprintf(stderr,
                        "artifact_descriptor_cutover=refused\n");
                (void)yvex_gguf_file_sink_withdraw(
                    file_sink, &file_failure, &error);
                rc = YVEX_ERR_STATE;
                goto cleanup;
            }
        }
        (void)clock_gettime(CLOCK_MONOTONIC, &phase_end);
        result->publish_admission_elapsed_ns = artifact_elapsed_ns(
            &phase_begin, &phase_end);
        result->published = 1;
        (void)snprintf(result->path, sizeof(result->path), "%s", destination);
    }
    (void)snprintf(result->profile_name, sizeof(result->profile_name), "%s",
                   quant_summary->profile_name);
    (void)snprintf(result->profile_identity,
                   sizeof(result->profile_identity), "%s",
                   quant_summary->profile_identity);
    (void)snprintf(result->writer_plan_identity,
                   sizeof(result->writer_plan_identity), "%s",
                   writer_summary->writer_plan_identity);
    (void)snprintf(result->artifact_identity,
                   sizeof(result->artifact_identity), "%s",
                   result->roundtrip.artifact_identity);
    (void)snprintf(result->execution_identity,
                   sizeof(result->execution_identity), "%s",
                   result->emission.execution_identity);
    result->complete = 1;
    printf("emission_profile=%s\n", result->profile_name);
    printf("emission_profile_identity=%s\n", result->profile_identity);
    printf("writer_plan_identity=%s\n", result->writer_plan_identity);
    printf("quant_execution_identity=%s\n", result->execution_identity);
    printf("artifact_identity=%s\n", result->artifact_identity);
    printf("artifact_path=%s\n", publish ? destination : "temporary-discard");
    printf("artifact_file_bytes=%llu\n", result->roundtrip.file_bytes);
    printf("artifact_payload_bytes=%llu\n",
           result->roundtrip.payload_bytes_verified);
    printf("artifact_bytes_hashed=%llu\n", result->roundtrip.bytes_hashed);
    printf("artifact_terminals_committed=%llu\n",
           result->emission.committed_terminals);
    printf("artifact_terminals_aborted=%llu\n",
           result->emission.aborted_terminals);
    printf("artifact_source_values=%llu\n",
           result->execution.source_values_consumed);
    printf("artifact_source_ranges=%llu\n",
           result->execution.source_ranges_read);
    printf("artifact_source_bytes=%llu\n",
           result->execution.payload_bytes_read);
    printf("artifact_physical_write_bytes=%llu\n",
           result->emission.physical_write_bytes);
    printf("artifact_writer_plan_owned_bytes=%llu\n",
           writer_summary->owned_bytes);
    printf("artifact_file_sink_peak_owned_bytes=%llu\n",
           result->emission.peak_owned_bytes);
    printf("artifact_peak_executor_bytes=%zu\n",
           result->execution.peak_owned_bytes);
    printf("artifact_quantize_write_nanoseconds=%llu\n",
           result->quantize_write_elapsed_ns);
    printf("artifact_finalize_nanoseconds=%llu\n",
           result->finalize_elapsed_ns);
    printf("artifact_native_roundtrip_nanoseconds=%llu\n",
           result->native_roundtrip_elapsed_ns);
    printf("artifact_official_reader_nanoseconds=%llu\n",
           result->official_reader_elapsed_ns);
    printf("artifact_publish_admission_nanoseconds=%llu\n",
           result->publish_admission_elapsed_ns);
    printf("artifact_native_roundtrip=accepted\n");
    printf("artifact_official_roundtrip=%s\n",
           publish ? "accepted" : "not-required");
    printf("artifact_admission=%s\n",
           publish ? artifact_class_name(
                         result->admission.artifact_class)
                   : "not-published");
    fflush(stdout);
    rc = YVEX_OK;

cleanup:
    if (progress_started && !atomic_load_explicit(
            &progress.stop, memory_order_relaxed)) {
        atomic_store_explicit(&progress.stop, 1, memory_order_relaxed);
        (void)pthread_join(progress_thread, NULL);
    }
    yvex_gguf_file_sink_release(&file_sink);
    yvex_gguf_writer_plan_release(&writer);
    yvex_quant_plan_release(&owned_quant);
    return rc == YVEX_OK && result->complete ? 0 : 1;
}

static int artifact_semantic_model_build(
    yvex_semantic_model_ir **out, const yvex_source_verification *verification,
    yvex_error *err)
{
    const yvex_family_compiler_adapter *adapter = yvex_compiler_family_deepseek_v4();
    const yvex_family_binding_pipeline *pipeline =
        adapter ? adapter->binding_pipeline : NULL;

    if (!pipeline || !pipeline->semantic_model_build) return YVEX_ERR_STATE;
    return pipeline->semantic_model_build(out, verification, err);
}

static int artifact_variant_bind(
    const yvex_deepseek_payload_handoff *handoff, const yvex_quant_plan *quant,
    const artifact_live_result *emitted, const char *binding_directory,
    char binding_path[YVEX_PATH_CAP])
{
    const yvex_model_family_api *model = yvex_model_register_deepseek_v4();
    const yvex_family_compiler_adapter *adapter =
        yvex_compiler_family_deepseek_v4();
    const yvex_graph_compiler_api *graph = adapter && adapter->graph
        ? adapter->graph() : NULL;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_tensor_table *tensors = NULL;
    yvex_materialization_plan *materialization_plan = NULL;
    yvex_materialization_session *materialization = NULL;
    yvex_semantic_model_ir *semantic_model = NULL;
    yvex_operator_graph_ir *operator_graph = NULL;
    yvex_physical_execution_ir *physical_execution = NULL;
    yvex_compiled_model_plan *compiled_plan = NULL;
    yvex_runtime_descriptor *descriptor = NULL;
    yvex_attention_plan *attention = NULL;
    yvex_attention_plan *draft_attention = NULL;
    yvex_gguf_writer_plan *writer = NULL;
    yvex_artifact_options artifact_options = {0};
    yvex_materialization_options materialization_options;
    yvex_materialization_projection materialization_projection;
    yvex_materialization_failure materialization_failure = {0};
    yvex_runtime_descriptor_failure descriptor_failure = {0};
    yvex_attention_failure attention_failure = {0};
    yvex_gguf_writer_failure writer_failure = {0};
    yvex_artifact_compatibility_failure compatibility_failure = {0};
    yvex_artifact_physical_compatibility compatibility;
    yvex_gguf_writer_plan_options writer_options;
    yvex_gguf_writer_plan_request writer_request = {0};
    yvex_runtime_binding_prepare_request prepare = {0};
    yvex_compiled_model_plan_request compiled_request = {0};
    yvex_runtime_binding_prepare_result prepared = {0};
    yvex_runtime_binding_failure binding_failure = {0};
    const yvex_transform_ir_summary *transform = yvex_transform_ir_summary_get(
        model->payload.transform_ir(handoff));
    const yvex_gguf_writer_plan_summary *writer_summary;
    const yvex_materialization_summary *materialization_summary;
    yvex_error error;
    int rc;

    yvex_error_clear(&error);
    if (!binding_directory || !binding_directory[0] || !transform || !adapter ||
        !adapter->execution_capabilities || !adapter->transformer_policy ||
        !adapter->logits_policy || !adapter->speculation_policy ||
        !adapter->tokenizer_policy || !graph) {
        fprintf(stderr, "variant_binding_preflight=refused\n");
        return 1;
    }
    artifact_options.path = emitted->path;
    artifact_options.readonly = 1;
    rc = yvex_artifact_open(&artifact, &artifact_options, &error);
    if (rc == YVEX_OK) rc = yvex_gguf_open(&gguf, artifact, &error);
    if (rc == YVEX_OK) rc = yvex_tensor_table_from_gguf(&tensors, gguf, &error);
    yvex_materialization_options_default(&materialization_options);
    materialization_options.require_terminal_projection = 1;
    materialization_options.max_chunk_bytes = 16ull * 1024ull * 1024ull;
    materialization_options.cache_budget_bytes = 256ull * 1024ull * 1024ull;
    materialization_options.future_graph_scratch_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    materialization_options.future_kv_reserve_bytes =
        2ull * 1024ull * 1024ull * 1024ull;
    if (rc == YVEX_OK)
        rc = yvex_deepseek_materialization_projection(
            model->payload.map(handoff), &materialization_projection, &error);
    if (rc == YVEX_OK)
        rc = yvex_materialization_plan_build(
            &materialization_plan, &emitted->admission, artifact, gguf, tensors,
            &materialization_projection, &materialization_options,
            &materialization_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_open(
            &materialization, materialization_plan, artifact, &materialization_options,
            &materialization_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_materialization_session_commit(
            materialization, &materialization_failure, &error);
    if (rc == YVEX_OK)
        rc = artifact_semantic_model_build(
            &semantic_model, model->payload.verification(handoff), &error);
    if (rc == YVEX_OK)
        rc = yvex_runtime_descriptor_build_deepseek(
            &descriptor, &emitted->admission, materialization, model->payload.map(handoff),
            semantic_model, &descriptor_failure, &error);
    if (rc == YVEX_OK)
        rc = graph->plan_build(
            &attention, semantic_model, materialization, descriptor,
            &attention_failure, &error);
    if (rc == YVEX_OK && graph->draft_plan_build)
        rc = graph->draft_plan_build(
            &draft_attention, semantic_model, materialization, descriptor,
            &attention_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_operator_graph_ir_build_transformer(
            &operator_graph, semantic_model, attention, draft_attention, &error);
    yvex_gguf_writer_plan_options_default(&writer_options);
    writer_request.input_class = YVEX_GGUF_WRITER_INPUT_COMPLETE_ARTIFACT;
    writer_request.quant_plan = quant;
    writer_request.options = &writer_options;
    writer_request.input.complete.lowering = yvex_model_deepseek_writer_lowering_api();
    writer_request.input.complete.lowering_context = model->payload.map(handoff);
    writer_request.input.complete.verification = model->payload.verification(handoff);
    if (rc == YVEX_OK)
        rc = yvex_gguf_writer_plan_build(
            &writer, &writer_request, &writer_failure, &error);
    if (rc == YVEX_OK)
        rc = yvex_artifact_physical_compatibility_validate(
            writer, &emitted->admission, artifact, gguf, &compatibility,
            &compatibility_failure, &error);
    writer_summary = yvex_gguf_writer_plan_summary_get(writer);
    if (rc == YVEX_OK && (!writer_summary || !compatibility.physical_payload_compatible ||
                          !adapter->execution_capabilities(&prepare.capabilities) ||
                          !yvex_runtime_capabilities_contract_valid(&prepare.capabilities) ||
                          !adapter->transformer_policy(
                              yvex_runtime_descriptor_summary_get(descriptor),
                              &prepare.transformer_policy) ||
                          !adapter->logits_policy(&prepare.logits_policy) ||
                          !adapter->speculation_policy(
                              yvex_runtime_descriptor_summary_get(descriptor),
                              &prepare.speculation_policy)))
        rc = YVEX_ERR_STATE;
    if (rc == YVEX_OK)
        rc = yvex_physical_execution_ir_build(
            &physical_execution, materialization, descriptor,
            emitted->admission.profile_identity,
            adapter->physical_execution_policy, &error);
    if (rc == YVEX_OK) {
        compiled_request.operator_graph = operator_graph;
        compiled_request.materialization = materialization;
        compiled_request.descriptor = descriptor;
        compiled_request.attention = attention;
        compiled_request.draft_attention = draft_attention;
        compiled_request.graph = graph;
        compiled_request.family_adapter_id = adapter->adapter_id;
        compiled_request.family_adapter_version = adapter->adapter_version;
        compiled_request.capabilities = prepare.capabilities;
        compiled_request.transformer_policy = prepare.transformer_policy;
        compiled_request.logits_policy = prepare.logits_policy;
        rc = yvex_compiled_model_plan_build(
            &compiled_plan, &compiled_request, &error);
    }
    if (rc == YVEX_OK) {
        prepare.directory = binding_directory;
        prepare.admission = &emitted->admission;
        prepare.physical_compatibility = &compatibility;
        prepare.materialization = materialization;
        prepare.runtime_descriptor = descriptor;
        prepare.operator_graph = operator_graph;
        prepare.physical_execution = physical_execution;
        prepare.compiled_plan = compiled_plan;
        prepare.attention_plan = attention;
        prepare.draft_attention_plan = draft_attention;
        prepare.family_adapter_id = adapter->adapter_id;
        prepare.family_adapter_version = adapter->adapter_version;
        prepare.artifact_format = "gguf";
        prepare.artifact_format_version = writer_summary->gguf_version;
        prepare.logical_transform_identity = transform->transform_identity;
        if (!adapter->tokenizer_policy(&prepare.tokenizer_policy, &error))
            rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK) {
        rc = yvex_runtime_binding_prepare(
            &prepare, &prepared, &binding_failure, &error);
    }
    materialization_summary = yvex_materialization_session_summary(materialization);
    if (rc == YVEX_OK) {
        (void)snprintf(binding_path, YVEX_PATH_CAP, "%s", prepared.path);
        printf("variant_materialization_identity=%s\n", materialization_summary->plan_identity);
        printf("variant_materialization_tensors=%llu\n", materialization_summary->tensor_count);
        printf("variant_materialization_payload_bytes=%llu\n",
               materialization_summary->payload_bytes);
        printf("variant_iq2_xxs_bytes=%llu\n",
               materialization_summary->qtype_bytes[YVEX_GGUF_QTYPE_IQ2_XXS]);
        printf("variant_runtime_binding_path=%s\n", prepared.path);
        printf("variant_runtime_binding_identity=%s\n", prepared.summary.identity);
        printf("variant_runtime_binding_schema=%u\n", prepared.summary.schema_version);
        printf("variant_materialization_ready=1\nvariant_runtime_binding_ready=1\n");
    } else {
        fprintf(stderr, "variant_binding_failure status=%s where=%s message=%s\n",
                yvex_status_name(yvex_error_code(&error)), yvex_error_where(&error),
                yvex_error_message(&error));
    }
    yvex_gguf_writer_plan_release(&writer);
    yvex_compiled_model_plan_close(&compiled_plan);
    yvex_physical_execution_ir_close(&physical_execution);
    yvex_operator_graph_ir_close(&operator_graph);
    yvex_semantic_model_ir_close(&semantic_model);
    yvex_attention_plan_close(attention);
    yvex_attention_plan_close(draft_attention);
    yvex_runtime_descriptor_close(descriptor);
    yvex_materialization_session_close(materialization);
    yvex_materialization_plan_close(materialization_plan);
    yvex_tensor_table_close(tensors);
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc == YVEX_OK ? 0 : 1;
}

int main(int argc, char **argv)
{
    yvex_deepseek_payload_handoff_options options;
    yvex_deepseek_payload_handoff *handoff = NULL;
    yvex_deepseek_payload_failure failure;
    yvex_error error;
    artifact_live_result reference;
    artifact_live_result selected;
    artifact_live_result deterministic;
    artifact_live_result variant;
    yvex_quant_policy *variant_policy = NULL;
    yvex_imatrix_data *variant_imatrix = NULL;
    yvex_quant_plan *variant_quant = NULL;
    yvex_imatrix_data_summary variant_imatrix_summary;
    yvex_quant_failure quant_failure;
    char reference_path[YVEX_ARTIFACT_PATH_CAP];
    char selected_path[YVEX_ARTIFACT_PATH_CAP];
    char deterministic_path[YVEX_ARTIFACT_PATH_CAP];
    const char *checker;
    int argument = 1;
    int plan_only = 0;
    int structure_only = 0;
    int variant_only = 0;
    const char *variant_destination = NULL;
    int rc;

    if (argc > 1 && strcmp(argv[1], "--plan-only") == 0) {
        plan_only = 1;
        argument++;
    } else if (argc > 1 && strcmp(argv[1], "--structure-only") == 0) {
        structure_only = 1;
        argument++;
    } else if (argc > 1 && strcmp(argv[1], "--variant") == 0) {
        variant_only = 1;
        argument++;
    }
    if ((!variant_only && argc - argument != 3) ||
        (variant_only && argc - argument != 4)) {
        fprintf(stderr,
                "usage: %s [--plan-only|--structure-only] SOURCE MODELS_ROOT MANIFEST\n"
                "       %s --variant SOURCE MODELS_ROOT MANIFEST DESTINATION\n",
                argv[0],
                argv[0]);
        return 2;
    }
    if (variant_only) variant_destination = argv[argument + 3];
    memset(&options, 0, sizeof(options));
    options.source_path = argv[argument];
    options.models_root = argv[argument + 1];
    options.manifest_path = argv[argument + 2];
    yvex_source_payload_budget_default(&options.budget);
    options.budget.maximum_open_handles = 32u;
    options.budget.maximum_streams = ARTIFACT_WORKERS;
    options.budget.maximum_inflight_host_bytes =
        options.budget.chunk_bytes * options.budget.maximum_streams;
    options.chunk_bytes = options.budget.chunk_bytes;
    options.page_bytes = options.budget.page_bytes;
    yvex_error_clear(&error);
    rc = yvex_model_register_deepseek_v4()->payload.open(
        &handoff, &options, &failure, &error);
    if (rc != YVEX_OK) {
        fprintf(stderr, "handoff_failure=%s where=%s message=%s\n",
                yvex_model_register_deepseek_v4()->payload.failure_name(failure.code),
                yvex_error_where(&error), yvex_error_message(&error));
        return 1;
    }
    if (variant_only) {
        yvex_imatrix_data_options imatrix_options;
        const char *preset = getenv("YVEX_QUANT_PRESET");
        const char *imatrix_path = getenv("YVEX_IMATRIX_PATH");
        const char *binding_directory = getenv("YVEX_VARIANT_BINDING_DIR");
        char binding_path[YVEX_PATH_CAP];

        if (!preset || !preset[0]) preset = YVEX_DEEPSEEK_QUANT_DSPARK_PROFILE_NAME;
        memset(&variant_imatrix_summary, 0, sizeof(variant_imatrix_summary));
        memset(&imatrix_options, 0, sizeof(imatrix_options));
        imatrix_options.path = imatrix_path;
        imatrix_options.source_model_identity =
            YVEX_DEEPSEEK_QUANT_IMATRIX_SOURCE_IDENTITY;
        imatrix_options.calibration_dataset_identity =
            YVEX_DEEPSEEK_QUANT_IMATRIX_DATASET_IDENTITY;
        imatrix_options.producer = "llama.cpp-imatrix";
        imatrix_options.producer_version = 1u;
        imatrix_options.maximum_mapped_bytes = 1024u * 1024u * 1024u;
        rc = imatrix_path && imatrix_path[0]
                 ? yvex_quant_policy_preset_open(&variant_policy, preset, &error)
                 : YVEX_ERR_INVALID_ARG;
        if (rc == YVEX_OK)
            rc = yvex_imatrix_data_open(&variant_imatrix, &imatrix_options, &error);
        if (rc == YVEX_OK)
            rc = yvex_imatrix_data_get_summary(
                variant_imatrix, &variant_imatrix_summary, &error);
        if (rc == YVEX_OK)
            rc = yvex_quant_plan_build_deepseek_policy(
                &variant_quant,
                yvex_model_register_deepseek_v4()->payload.transform_ir(handoff),
                yvex_model_register_deepseek_v4()->payload.binding(handoff),
                yvex_model_register_deepseek_v4()->payload.map(handoff), variant_policy,
                variant_imatrix_summary.imatrix_identity, NULL, &quant_failure, &error);
        checker = getenv("YVEX_GGML_CHECKER");
        if (!checker || !checker[0]) checker = "build/tests/ggml_gguf_check";
        if (rc == YVEX_OK) {
            printf("mode=complete-physical-variant-emission\n");
            rc = artifact_execute_one(
                handoff, YVEX_QUANT_PROFILE_RELEASE_Q8_Q2, variant_quant, variant_imatrix,
                NULL, variant_destination, checker, 1, &variant);
        }
        if (rc == YVEX_OK)
            rc = artifact_variant_bind(
                handoff, variant_quant, &variant, binding_directory, binding_path);
        if (rc == YVEX_OK) {
            printf("variant_policy_identity=%s\n",
                   yvex_quant_plan_summary_get(variant_quant)->policy_identity);
            printf("variant_imatrix_identity=%s\n", variant_imatrix_summary.imatrix_identity);
            printf("variant_artifact_identity=%s\n", variant.artifact_identity);
            printf("variant_materialization_input_ready=1\n");
        } else {
            fprintf(stderr, "variant_emission_failure where=%s message=%s\n",
                    yvex_error_where(&error), yvex_error_message(&error));
        }
        yvex_quant_plan_release(&variant_quant);
        yvex_imatrix_data_close(variant_imatrix);
        yvex_quant_policy_close(variant_policy);
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return rc == YVEX_OK ? 0 : 1;
    }
    if (plan_only) {
        printf("mode=plan-only\n");
        rc = artifact_plan_one(
            handoff, YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, NULL, NULL);
        if (rc == 0 && artifact_path_build(
                selected_path, sizeof(selected_path), argv[argument + 1],
                "deepseek-v4-flash-dspark-q8_0-q2_k-v1.gguf"))
            rc = artifact_plan_one(
                handoff, YVEX_QUANT_PROFILE_RELEASE_Q8_Q2,
                SELECTED_EXECUTION_IDENTITY, selected_path);
        else if (rc == 0)
            rc = 1;
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return rc;
    }
    if (!artifact_path_build(
            reference_path, sizeof(reference_path), argv[argument + 1],
            "deepseek-v4-flash-dspark-source-faithful-v1.gguf") ||
        !artifact_path_build(
            selected_path, sizeof(selected_path), argv[argument + 1],
            "deepseek-v4-flash-dspark-q8_0-q2_k-v1.gguf") ||
        !artifact_path_build(
            deterministic_path, sizeof(deterministic_path),
            argv[argument + 1],
            "deepseek-v4-flash-dspark-q8_0-q2_k-v1.determinism.gguf")) {
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return 1;
    }
    checker = getenv("YVEX_GGML_CHECKER");
    if (!checker || !checker[0]) checker = "build/tests/ggml_gguf_check";
    if (structure_only) {
        printf("mode=sparse-structure-only\n");
        rc = artifact_structure_one(
            handoff, YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, NULL, checker);
        if (rc == 0)
            rc = artifact_structure_one(
                handoff, YVEX_QUANT_PROFILE_RELEASE_Q8_Q2,
                SELECTED_EXECUTION_IDENTITY, checker);
        yvex_model_register_deepseek_v4()->payload.close(handoff);
        return rc;
    }
    printf("mode=complete-artifact-emission\n");
    rc = artifact_execute_one(
        handoff, YVEX_QUANT_PROFILE_SOURCE_FAITHFUL, NULL, NULL, NULL,
        reference_path, checker, 1, &reference);
    if (rc == 0)
        rc = artifact_execute_one(
            handoff, YVEX_QUANT_PROFILE_RELEASE_Q8_Q2,
            NULL, NULL, SELECTED_EXECUTION_IDENTITY, selected_path, checker, 1,
            &selected);
    if (rc == 0)
        rc = artifact_execute_one(
            handoff, YVEX_QUANT_PROFILE_RELEASE_Q8_Q2,
            NULL, NULL, SELECTED_EXECUTION_IDENTITY, deterministic_path, checker, 0,
            &deterministic);
    if (rc == 0 &&
        (strcmp(selected.artifact_identity,
                deterministic.artifact_identity) != 0 ||
         strcmp(selected.execution_identity,
                deterministic.execution_identity) != 0)) {
        fprintf(stderr,
                "deterministic_identity_failure selected=%s repeated=%s\n",
                selected.artifact_identity, deterministic.artifact_identity);
        rc = 1;
    }
    if (rc == 0) {
        printf("reference_artifact_identity=%s\n",
               reference.artifact_identity);
        printf("selected_artifact_identity=%s\n",
               selected.artifact_identity);
        printf("deterministic_artifact_identity=%s\n",
               deterministic.artifact_identity);
        printf("deterministic_execution_identity=%s\n",
               deterministic.execution_identity);
        printf("deterministic_serialization=accepted\n");
        printf("test_artifact_profiles_emitted=2\n");
        printf("materialization_input_ready=1\n");
        printf("runtime_supported=0\n");
    }
    yvex_model_register_deepseek_v4()->payload.close(handoff);
    return rc;
}
