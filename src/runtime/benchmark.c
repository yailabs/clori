/* Owner: runtime.benchmark.
 * Owns: immutable runtime benchmark baselines, compatibility comparison, and deterministic SVG evidence.
 * Does not own: measurement, benchmark admission, runtime execution, CLI parsing, or capability promotion.
 * Invariants: canonical records are bounded, content-addressed, independently reopenable, and never overwritten.
 * Boundary: this file-serialization lifecycle consumes typed execution facts after production execution completes.
 * Purpose: persist trustworthy external benchmark evidence and compare equivalent measurements
 *   under an optional caller-owned regression policy.
 * Inputs: typed identities, counters, timings, and external paths. Effects: publishes only unique owned files.
 * Failure: typed refusal preserves existing files and removes only the exact temporary owned by the failed call. */

#define _GNU_SOURCE
#include <yvex/internal/benchmark.h>
#include <yvex/internal/runtime.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <build_commit.h>

#define BENCHMARK_FILE_MAX (32u * 1024u)
#define BENCHMARK_CHART_MAX (64u * 1024u)
#define BENCHMARK_EMPTY_SOURCE_DELTA \
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

typedef yvex_core_bytes benchmark_bytes;
typedef yvex_graph_attention_operator_result bench_result;
typedef yvex_runtime_benchmark_baseline bench_record;
typedef yvex_runtime_benchmark_key bench_key;
typedef yvex_runtime_benchmark_metrics bench_metrics;
typedef enum {
    BENCHMARK_FIELD_COMMIT = 0,
    BENCHMARK_FIELD_SOURCE_STATE,
    BENCHMARK_FIELD_SHA256,
    BENCHMARK_FIELD_TEXT_16,
    BENCHMARK_FIELD_TEXT_32,
    BENCHMARK_FIELD_TEXT_64,
    BENCHMARK_FIELD_TEXT_128,
    BENCHMARK_FIELD_BOOL,
    BENCHMARK_FIELD_U64
} benchmark_field_kind;
typedef struct {
    const char *name;
    size_t offset;
    benchmark_field_kind kind;
} benchmark_field;
static const size_t benchmark_field_widths[] = {
    YVEX_RUNTIME_BENCHMARK_COMMIT_CAP, YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP,
    YVEX_SHA256_HEX_BYTES, YVEX_RUNTIME_BENCHMARK_MODE_CAP,
    YVEX_RUNTIME_BENCHMARK_SCOPE_CAP, YVEX_RUNTIME_BENCHMARK_BUCKET_CAP,
    YVEX_RUNTIME_BENCHMARK_TEXT_CAP, sizeof(int), sizeof(unsigned long long)};
typedef struct {
    size_t destination, capacity, source;
} benchmark_text_projection;
typedef struct {
    size_t destination, source;
} benchmark_u64_projection;
typedef struct {
    unsigned int index, percentile;
} benchmark_percentile;
typedef struct {
    unsigned int index;
    const char *value;
} benchmark_status;

static const benchmark_field bench_key_fields[] = {
    {"commit", offsetof(bench_record, key.commit), BENCHMARK_FIELD_COMMIT},
    {"build_source_state", offsetof(bench_record, key.build_source_state), BENCHMARK_FIELD_SOURCE_STATE},
    {"source_delta_identity", offsetof(bench_record, key.source_delta_identity), BENCHMARK_FIELD_SHA256},
{"build_identity", offsetof(bench_record, key.build_identity), BENCHMARK_FIELD_SHA256},
{"benchmark_scope", offsetof(bench_record, key.benchmark_scope), BENCHMARK_FIELD_TEXT_32},
    {"artifact_identity", offsetof(bench_record, key.artifact_identity), BENCHMARK_FIELD_SHA256},
    {"materialization_identity", offsetof(bench_record, key.materialization_identity), BENCHMARK_FIELD_SHA256},
    {"runtime_binding_identity", offsetof(bench_record, key.runtime_binding_identity), BENCHMARK_FIELD_SHA256},
{"logical_model_identity", offsetof(bench_record, key.logical_model_identity), BENCHMARK_FIELD_SHA256},
    {"runtime_numeric_identity", offsetof(bench_record, key.runtime_numeric_identity), BENCHMARK_FIELD_SHA256},
    {"runtime_descriptor_identity", offsetof(bench_record, key.runtime_descriptor_identity), BENCHMARK_FIELD_SHA256},
    {"semantic_graph_identity", offsetof(bench_record, key.semantic_graph_identity), BENCHMARK_FIELD_SHA256},
{"executable_graph_identity", offsetof(bench_record, key.executable_graph_identity), BENCHMARK_FIELD_SHA256},
    {"execution_descriptor_identity", offsetof(bench_record, key.execution_descriptor_identity),
     BENCHMARK_FIELD_SHA256},
    {"residency_identity", offsetof(bench_record, key.residency_identity), BENCHMARK_FIELD_SHA256},
{"workspace_identity", offsetof(bench_record, key.workspace_identity), BENCHMARK_FIELD_SHA256},
    {"state_layout_identity", offsetof(bench_record, key.state_layout_identity), BENCHMARK_FIELD_SHA256},
{"kernel_bundle_identity", offsetof(bench_record, key.kernel_bundle_identity), BENCHMARK_FIELD_SHA256},
    {"machine_identity", offsetof(bench_record, key.machine_identity), BENCHMARK_FIELD_TEXT_128},
{"cpu_model", offsetof(bench_record, key.cpu_model), BENCHMARK_FIELD_TEXT_128},
{"gpu_model", offsetof(bench_record, key.gpu_model), BENCHMARK_FIELD_TEXT_128},
    {"device", offsetof(bench_record, key.device), BENCHMARK_FIELD_TEXT_128},
{"driver", offsetof(bench_record, key.driver), BENCHMARK_FIELD_TEXT_128},
    {"cuda_build", offsetof(bench_record, key.cuda_build), BENCHMARK_FIELD_TEXT_128},
{"mode", offsetof(bench_record, key.mode), BENCHMARK_FIELD_TEXT_16},
    {"phase", offsetof(bench_record, key.phase), BENCHMARK_FIELD_TEXT_16},
{"scope", offsetof(bench_record, key.scope), BENCHMARK_FIELD_TEXT_32},
    {"coverage", offsetof(bench_record, key.coverage), BENCHMARK_FIELD_TEXT_16},
    {"attention_class", offsetof(bench_record, key.attention_class), BENCHMARK_FIELD_TEXT_16},
    {"trace_policy", offsetof(bench_record, key.trace_policy), BENCHMARK_FIELD_TEXT_16},
{"capture_bucket", offsetof(bench_record, key.capture_bucket), BENCHMARK_FIELD_TEXT_64},
    {"memory_bytes", offsetof(bench_record, key.memory_bytes), BENCHMARK_FIELD_U64},
    {"compute_capability_major", offsetof(bench_record, key.compute_capability_major), BENCHMARK_FIELD_U64},
    {"compute_capability_minor", offsetof(bench_record, key.compute_capability_minor), BENCHMARK_FIELD_U64},
{"layer_start", offsetof(bench_record, key.layer_start), BENCHMARK_FIELD_U64},
{"layer_count", offsetof(bench_record, key.layer_count), BENCHMARK_FIELD_U64},
    {"token_count", offsetof(bench_record, key.token_count), BENCHMARK_FIELD_U64},
    {"history_tokens", offsetof(bench_record, key.history_tokens), BENCHMARK_FIELD_U64},
    {"warmup_count", offsetof(bench_record, key.warmup_count), BENCHMARK_FIELD_U64},
{"iteration_count", offsetof(bench_record, key.iteration_count), BENCHMARK_FIELD_U64},
};
static const benchmark_field benchmark_metric_fields[] = {
    {"cold_total_ns", offsetof(bench_record, metrics.cold_total_ns), BENCHMARK_FIELD_U64},
{"cold_artifact_ns", offsetof(bench_record, metrics.cold_artifact_ns), BENCHMARK_FIELD_U64},
    {"cold_binding_ns", offsetof(bench_record, metrics.cold_binding_ns), BENCHMARK_FIELD_U64},
{"cold_model_ns", offsetof(bench_record, metrics.cold_model_ns), BENCHMARK_FIELD_U64},
    {"cold_residency_ns", offsetof(bench_record, metrics.cold_residency_ns), BENCHMARK_FIELD_U64},
{"cold_workspace_ns", offsetof(bench_record, metrics.cold_workspace_ns), BENCHMARK_FIELD_U64},
    {"cold_graph_warmup_ns", offsetof(bench_record, metrics.cold_graph_warmup_ns), BENCHMARK_FIELD_U64},
{"cold_graph_capture_ns", offsetof(bench_record, metrics.cold_graph_capture_ns), BENCHMARK_FIELD_U64},
    {"cold_graph_instantiate_ns", offsetof(bench_record, metrics.cold_graph_instantiate_ns), BENCHMARK_FIELD_U64},
{"first_execution_ns", offsetof(bench_record, metrics.first_execution_ns), BENCHMARK_FIELD_U64},
    {"publication_ns", offsetof(bench_record, metrics.publication_ns), BENCHMARK_FIELD_U64},
{"cleanup_ns", offsetof(bench_record, metrics.cleanup_ns), BENCHMARK_FIELD_U64},
    {"minimum_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
     BENCHMARK_FIELD_U64},
    {"p50_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P50]), BENCHMARK_FIELD_U64},
{"p90_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P90]), BENCHMARK_FIELD_U64},
    {"p95_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P95]), BENCHMARK_FIELD_U64},
{"p99_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P99]), BENCHMARK_FIELD_U64},
    {"maximum_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
     BENCHMARK_FIELD_U64},
    {"mean_ns", offsetof(bench_record, metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN]), BENCHMARK_FIELD_U64},
    {"standard_deviation_ns", offsetof(bench_record,
                                      metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]),
     BENCHMARK_FIELD_U64},
    {"device_timing_available", offsetof(bench_record, metrics.device_timing.available), BENCHMARK_FIELD_BOOL},
    {"device_minimum_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MINIMUM]),
     BENCHMARK_FIELD_U64},
    {"device_p50_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P50]),
     BENCHMARK_FIELD_U64},
    {"device_p90_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P90]),
     BENCHMARK_FIELD_U64},
    {"device_p95_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P95]),
     BENCHMARK_FIELD_U64},
    {"device_p99_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P99]),
     BENCHMARK_FIELD_U64},
    {"device_maximum_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM]),
     BENCHMARK_FIELD_U64},
    {"device_mean_ns", offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN]),
     BENCHMARK_FIELD_U64},
    {"device_standard_deviation_ns",
     offsetof(bench_record, metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]),
     BENCHMARK_FIELD_U64},
    {"artifact_bytes_hashed", offsetof(bench_record, metrics.artifact_bytes_hashed), BENCHMARK_FIELD_U64},
{"artifact_bytes_read", offsetof(bench_record, metrics.artifact_bytes_read), BENCHMARK_FIELD_U64},
    {"weight_bytes_read", offsetof(bench_record, metrics.weight_bytes_read), BENCHMARK_FIELD_U64},
{"resident_encoded_bytes", offsetof(bench_record, metrics.resident_encoded_bytes), BENCHMARK_FIELD_U64},
    {"resident_h2d_bytes", offsetof(bench_record, metrics.resident_h2d_bytes), BENCHMARK_FIELD_U64},
{"h2d_bytes", offsetof(bench_record, metrics.h2d_bytes), BENCHMARK_FIELD_U64},
{"d2h_bytes", offsetof(bench_record, metrics.d2h_bytes), BENCHMARK_FIELD_U64},
    {"warm_weight_reads", offsetof(bench_record, metrics.warm_weight_reads), BENCHMARK_FIELD_U64},
{"warm_upload_bytes", offsetof(bench_record, metrics.warm_upload_bytes), BENCHMARK_FIELD_U64},
    {"warm_host_allocations", offsetof(bench_record, metrics.warm_host_allocations), BENCHMARK_FIELD_U64},
{"warm_device_allocations", offsetof(bench_record, metrics.warm_device_allocations), BENCHMARK_FIELD_U64},
    {"warm_device_frees", offsetof(bench_record, metrics.warm_device_frees), BENCHMARK_FIELD_U64},
    {"last_dispatch_kernel_launches", offsetof(bench_record, metrics.last_dispatch_kernel_launches),
     BENCHMARK_FIELD_U64},
    {"cuda_graph_launches", offsetof(bench_record, metrics.cuda_graph_launches), BENCHMARK_FIELD_U64},
{"cuda_graph_captures", offsetof(bench_record, metrics.cuda_graph_captures), BENCHMARK_FIELD_U64},
    {"cuda_graph_replays", offsetof(bench_record, metrics.cuda_graph_replays), BENCHMARK_FIELD_U64},
{"cuda_graph_nodes", offsetof(bench_record, metrics.cuda_graph_nodes), BENCHMARK_FIELD_U64},
    {"peak_host_bytes", offsetof(bench_record, metrics.peak_host_bytes), BENCHMARK_FIELD_U64},
    {"last_dispatch_peak_device_bytes", offsetof(bench_record, metrics.last_dispatch_peak_device_bytes),
     BENCHMARK_FIELD_U64},
    {"resident_bytes", offsetof(bench_record, metrics.resident_bytes), BENCHMARK_FIELD_U64},
    {"host_workspace_bytes", offsetof(bench_record, metrics.host_workspace_bytes), BENCHMARK_FIELD_U64},
    {"state_bytes", offsetof(bench_record, metrics.state_bytes), BENCHMARK_FIELD_U64},
};
static const yvex_runtime_benchmark_key bench_key_defaults = {
    .commit = YVEX_BUILD_COMMIT,
    .build_source_state = YVEX_BUILD_SOURCE_STATE,
    .source_delta_identity = YVEX_BUILD_SOURCE_DELTA_IDENTITY,
    .build_identity = YVEX_BUILD_IDENTITY,
    .benchmark_scope = "attention_component",
};
static const benchmark_text_projection bench_key_result_fields[] = {
    {offsetof(bench_key, artifact_identity), YVEX_SHA256_HEX_BYTES, offsetof(bench_result, artifact_identity)},
    {offsetof(bench_key, materialization_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, materialization_identity)},
    {offsetof(bench_key, runtime_binding_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, runtime_binding_identity)},
    {offsetof(bench_key, logical_model_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, logical_model_identity)},
    {offsetof(bench_key, runtime_numeric_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, runtime_numeric_identity)},
    {offsetof(bench_key, runtime_descriptor_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, runtime_descriptor_identity)},
    {offsetof(bench_key, semantic_graph_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, semantic_graph_identity)},
    {offsetof(bench_key, executable_graph_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, executable_graph_identity)},
    {offsetof(bench_key, execution_descriptor_identity), YVEX_SHA256_HEX_BYTES,
     offsetof(bench_result, execution_descriptor_identity)},
    {offsetof(bench_key, residency_identity), YVEX_SHA256_HEX_BYTES, offsetof(bench_result, residency_identity)},
    {offsetof(bench_key, workspace_identity), YVEX_SHA256_HEX_BYTES, offsetof(bench_result, workspace_identity)},
    {offsetof(bench_key, state_layout_identity), YVEX_SHA256_HEX_BYTES, offsetof(bench_result, state_layout_identity)},
    {offsetof(bench_key, mode), YVEX_RUNTIME_BENCHMARK_MODE_CAP, offsetof(bench_result, selected_mode)},
{offsetof(bench_key, phase), YVEX_RUNTIME_BENCHMARK_MODE_CAP, offsetof(bench_result, phase)},
    {offsetof(bench_key, scope), YVEX_RUNTIME_BENCHMARK_SCOPE_CAP, offsetof(bench_result, operation_scope)},
{offsetof(bench_key, coverage), YVEX_RUNTIME_BENCHMARK_MODE_CAP, offsetof(bench_result, scope)},
    {offsetof(bench_key, attention_class), YVEX_RUNTIME_BENCHMARK_MODE_CAP, offsetof(bench_result, attention_class)},
    {offsetof(bench_key, trace_policy), YVEX_RUNTIME_BENCHMARK_MODE_CAP, offsetof(bench_result, trace_policy)},
    {offsetof(bench_key, capture_bucket), YVEX_RUNTIME_BENCHMARK_BUCKET_CAP, offsetof(bench_result, capture_bucket)},
};
static const benchmark_u64_projection benchmark_metric_result_fields[] = {
    {offsetof(bench_metrics, artifact_bytes_hashed), offsetof(bench_result, artifact_bytes_hashed)},
{offsetof(bench_metrics, artifact_bytes_read), offsetof(bench_result, artifact_bytes_hashed)},
    {offsetof(bench_metrics, weight_bytes_read), offsetof(bench_result, resident_encoded_bytes)},
{offsetof(bench_metrics, resident_encoded_bytes), offsetof(bench_result, resident_encoded_bytes)},
    {offsetof(bench_metrics, resident_h2d_bytes), offsetof(bench_result, upload_bytes)},
{offsetof(bench_metrics, warm_weight_reads), offsetof(bench_result, warm_weight_artifact_reads)},
    {offsetof(bench_metrics, warm_upload_bytes), offsetof(bench_result, warm_weight_upload_bytes)},
{offsetof(bench_metrics, warm_host_allocations), offsetof(bench_result, warm_host_allocations)},
    {offsetof(bench_metrics, warm_device_allocations), offsetof(bench_result, warm_device_allocations)},
{offsetof(bench_metrics, warm_device_frees), offsetof(bench_result, warm_device_frees)},
    {offsetof(bench_metrics, cuda_graph_launches), offsetof(bench_result, cuda_graph_launch_count)},
{offsetof(bench_metrics, cuda_graph_captures), offsetof(bench_result, cuda_graph_capture_count)},
    {offsetof(bench_metrics, cuda_graph_replays), offsetof(bench_result, cuda_graph_replay_count)},
{offsetof(bench_metrics, cuda_graph_nodes), offsetof(bench_result, cuda_graph_node_count)},
    {offsetof(bench_metrics, host_workspace_bytes), offsetof(bench_result, workspace_bytes)},
{offsetof(bench_metrics, state_bytes), offsetof(bench_result, state_allocated_bytes)},
};
static const benchmark_u64_projection bench_key_result_u64_fields[] = {
    {offsetof(bench_key, layer_start), offsetof(bench_result, requested_layer_start)},
    {offsetof(bench_key, layer_count), offsetof(bench_result, requested_layer_count)},
    {offsetof(bench_key, token_count), offsetof(bench_result, requested_token_count)},
    {offsetof(bench_key, history_tokens), offsetof(bench_result, requested_history_tokens)},
    {offsetof(bench_key, warmup_count), offsetof(bench_result, warmup_count)},
    {offsetof(bench_key, iteration_count), offsetof(bench_result, benchmark_sample_count)},
};
static const benchmark_percentile benchmark_percentiles[] = {
    {YVEX_RUNTIME_BENCHMARK_P50, 50u}, {YVEX_RUNTIME_BENCHMARK_P90, 90u},
    {YVEX_RUNTIME_BENCHMARK_P95, 95u}, {YVEX_RUNTIME_BENCHMARK_P99, 99u},
};
static const benchmark_status benchmark_measured_statuses[] = {
    {YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK, "measured"},
    {YVEX_RUNTIME_QUALITY_CORRECTNESS, "pass"},
    {YVEX_RUNTIME_QUALITY_STRUCTURAL, "pass"},
    {YVEX_RUNTIME_QUALITY_PERFORMANCE, "measured"},
};
enum {
    BENCHMARK_KEY_FIELD_COUNT = sizeof(bench_key_fields) / sizeof(bench_key_fields[0]),
    BENCHMARK_METRIC_FIELD_COUNT = sizeof(benchmark_metric_fields) / sizeof(benchmark_metric_fields[0]),
    BENCHMARK_KEY_TEXT_COUNT = sizeof(bench_key_result_fields) / sizeof(bench_key_result_fields[0]),
    BENCHMARK_KEY_U64_COUNT = sizeof(bench_key_result_u64_fields) / sizeof(bench_key_result_u64_fields[0]),
    BENCHMARK_METRIC_PROJECTION_COUNT =
        sizeof(benchmark_metric_result_fields) / sizeof(benchmark_metric_result_fields[0]),
    BENCHMARK_PERCENTILE_COUNT = sizeof(benchmark_percentiles) / sizeof(benchmark_percentiles[0]),
    BENCHMARK_MEASURED_STATUS_COUNT =
        sizeof(benchmark_measured_statuses) / sizeof(benchmark_measured_statuses[0]),
};
static const yvex_runtime_benchmark_failure_code benchmark_file_codes[] = {
    YVEX_RUNTIME_BENCHMARK_FAILURE_PUBLISH,
    YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_BENCHMARK_FAILURE_PATH,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CREATE,
    YVEX_RUNTIME_BENCHMARK_FAILURE_WRITE,
    YVEX_RUNTIME_BENCHMARK_FAILURE_SYNC,
    YVEX_RUNTIME_BENCHMARK_FAILURE_SYNC,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CONFLICT,
    YVEX_RUNTIME_BENCHMARK_FAILURE_PUBLISH,
    YVEX_RUNTIME_BENCHMARK_FAILURE_SYNC,
    YVEX_RUNTIME_BENCHMARK_FAILURE_OPEN,
    YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS,
    YVEX_RUNTIME_BENCHMARK_FAILURE_READ,
    YVEX_RUNTIME_BENCHMARK_FAILURE_READ,
    YVEX_RUNTIME_BENCHMARK_FAILURE_READ,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CLEANUP,
};
enum { BENCHMARK_FILE_CODE_COUNT = sizeof(benchmark_file_codes) / sizeof(benchmark_file_codes[0]) };
/* Purpose: publish one stable typed benchmark refusal and clear no caller-owned evidence. */
static int benchmark_reject(yvex_runtime_benchmark_failure *failure,
                            yvex_runtime_benchmark_failure_code code,
                            const char *field, unsigned long long expected,
                            unsigned long long actual, yvex_status status,
                            const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
        if (field) yvex_core_text_copy(failure->field, sizeof(failure->field), field);
    }
    yvex_error_set(err, status, "runtime_benchmark", reason);
    return status;
}
/* Purpose: reject one malformed benchmark field through the canonical typed failure owner. */
static int benchmark_field_reject(yvex_runtime_benchmark_failure *failure,
                                  const char *field, unsigned long long expected,
                                  unsigned long long actual, yvex_status status,
                                  const char *reason, yvex_error *err)
{
    return benchmark_reject(
        failure, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, field, expected, actual, status, reason, err);
}
/* Purpose: reject one failed boolean invariant without repeating its expected/actual encoding. */
static int benchmark_fail(
    yvex_runtime_benchmark_failure *failure, yvex_runtime_benchmark_failure_code code,
    const char *field, yvex_status status, const char *reason, yvex_error *err)
{
    return benchmark_reject(failure, code, field, 1ull, 0ull, status, reason, err);
}
/* Purpose: reject one malformed boolean field through the canonical field vocabulary. */
static int benchmark_field_fail(
    yvex_runtime_benchmark_failure *failure, const char *field, yvex_status status,
    const char *reason, yvex_error *err)
{
    return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, field, status, reason, err);
}
/* Purpose: append one complete immutable ASCII string without hand-maintained byte counts. */
static int bytes_text(benchmark_bytes *bytes, const char *text)
{
    return text && yvex_core_bytes_append(bytes, text, strlen(text));
}
/* Purpose: finalize one canonical SHA-256 construction into its fixed hexadecimal identity.
 * Inputs: active hash and fixed-capacity output. Effects: writes one lowercase identity.
 * Failure: returns false when finalization fails. Boundary: hashes no native object storage. */
static int benchmark_hash_finish(yvex_sha256 *hash, char output[YVEX_SHA256_HEX_BYTES])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
/* Purpose: append formatted ASCII while rejecting truncation and allocation failure.
 * Inputs: buffer, format, and scalar arguments. Effects: appends one complete formatted span.
 * Failure: false without count advance. Boundary: callers own validation and ordering. */
static int bytes_format(benchmark_bytes *bytes, const char *format, ...)
{
    char stack[512];
    va_list args, copy;
    int needed;
    va_start(args, format);
    va_copy(copy, args);
    needed = vsnprintf(stack, sizeof(stack), format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(args);
        return 0;
    }
    if ((size_t)needed < sizeof(stack)) {
        va_end(args);
        return yvex_core_bytes_append(bytes, stack, (size_t)needed);
    }
    if (!yvex_core_bytes_reserve(bytes, (size_t)needed)) {
        va_end(args);
        return 0;
    }
    needed = vsnprintf((char *)bytes->data + bytes->count, bytes->capacity - bytes->count, format, args);
    va_end(args);
    if (needed < 0 || (size_t)needed >= bytes->capacity - bytes->count) return 0;
    bytes->count += (size_t)needed;
    return 1;
}
/* Purpose: validate one bounded printable field that enters identity and text serialization. */
static int text_valid(const char *text, size_t capacity)
{
    size_t index, length;
    if (!text || !memchr(text, '\0', capacity)) return 0;
    length = strlen(text);
    if (!length) return 0;
    for (index = 0u; index < length; ++index) {
        unsigned char value = (unsigned char)text[index];
        if (value < 0x20u || value > 0x7eu || value == '\t') return 0;
    }
    return 1;
}
/* Purpose: copy one benchmark text field without accepting truncation. */
static int benchmark_text_copy(char *output, size_t capacity, const char *text)
{
    return output && text && capacity && snprintf(output, capacity, "%s", text) < (int)capacity;
}
/* Purpose: convert one finite nonnegative duration to canonical nanoseconds. */
static int benchmark_seconds_ns(double seconds, unsigned long long *output)
{
    double integral_seconds, fractional_seconds;
    unsigned long long integral_ns, fractional_ns;
    if (!output || !isfinite(seconds) || seconds < 0.0) return 0;
    fractional_seconds = modf(seconds, &integral_seconds);
    if (integral_seconds > (double)(ULLONG_MAX / 1000000000ull)) return 0;
    fractional_ns = (unsigned long long)(fractional_seconds * 1000000000.0 + 0.5);
    return yvex_core_u64_mul((unsigned long long)integral_seconds, 1000000000ull, &integral_ns) &&
           yvex_core_u64_add(integral_ns, fractional_ns, output);
}

typedef struct {
    double values[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
} benchmark_seconds_distribution;

typedef struct {
    unsigned int first, count;
} benchmark_lifecycle_group;

static const benchmark_lifecycle_group benchmark_cold_groups[] = {
    {YVEX_RUNTIME_LIFECYCLE_ARTIFACT_OPEN, 3u}, {YVEX_RUNTIME_LIFECYCLE_BINDING_OPEN, 2u},
{YVEX_RUNTIME_LIFECYCLE_MODEL_SEAL, 1u}, {YVEX_RUNTIME_LIFECYCLE_RESIDENCY, 1u},
    {YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN, 2u}, {YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP, 1u},
{YVEX_RUNTIME_LIFECYCLE_GRAPH_CAPTURE, 1u}, {YVEX_RUNTIME_LIFECYCLE_GRAPH_INSTANTIATE, 1u},
};
enum { BENCHMARK_COLD_GROUP_COUNT = sizeof(benchmark_cold_groups) / sizeof(benchmark_cold_groups[0]) };
/* Purpose: impose one total numeric ordering over finite benchmark samples. */
static int benchmark_sample_compare(const void *left, const void *right)
{
    const double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}
/* Purpose: validate, sort, and summarize one positive finite timing sample set.
 * Inputs: mutable samples, count, and summary. Effects: validates, sorts, and summarizes the samples.
 * Failure: false for invalid samples. Boundary: computes distributions without policy or publication. */
static int benchmark_samples_summarize(double *samples, unsigned long long count,
                                       benchmark_seconds_distribution *summary)
{
    double sum = 0.0, squared = 0.0;
    unsigned long long index;
    size_t percentile;
    if (!samples || !summary || !count || count > (unsigned long long)(SIZE_MAX / sizeof(*samples)))
        return 0;
    for (index = 0ull; index < count; ++index)
        if (!isfinite(samples[index]) || samples[index] <= 0.0) return 0;
    qsort(samples, (size_t)count, sizeof(*samples), benchmark_sample_compare);
    for (index = 0ull; index < count; ++index) sum += samples[index];
    summary->values[YVEX_RUNTIME_BENCHMARK_MINIMUM] = samples[0];
    for (percentile = 0u; percentile < BENCHMARK_PERCENTILE_COUNT; ++percentile)
        summary->values[benchmark_percentiles[percentile].index] =
            samples[(count * benchmark_percentiles[percentile].percentile + 99ull) / 100ull - 1ull];
    summary->values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] = samples[count - 1ull];
    summary->values[YVEX_RUNTIME_BENCHMARK_MEAN] = sum / (double)count;
    for (index = 0ull; index < count; ++index) {
        const double delta = samples[index] - summary->values[YVEX_RUNTIME_BENCHMARK_MEAN];
        squared += delta * delta;
    }
    summary->values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION] = sqrt(squared / (double)count);
    return isfinite(sum) && isfinite(squared) &&
           isfinite(summary->values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION]);
}
/* Purpose: publish host and optional device benchmark distributions through one owner.
 * Inputs: host/device samples, count, and result. Effects: publishes complete timing distributions.
 * Failure: invalid samples publish nothing. Boundary: statistics promote no execution capability. */
int yvex_runtime_benchmark_samples_finish(
    double *host_seconds, double *device_seconds, unsigned long long count,
    int device_requested, yvex_graph_attention_operator_result *result, yvex_error *err)
{
    benchmark_seconds_distribution host, device = {0};
    double first_execution;
    size_t status;
    if (!result || (device_requested != 0 && device_requested != 1) || !host_seconds || !count)
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "host-samples",
                                count, 0ull, YVEX_ERR_BOUNDS,
                                "benchmark samples must be positive, finite, and complete", err);
    first_execution = host_seconds[0];
    if (!benchmark_samples_summarize(host_seconds, count, &host))
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "host-samples",
                                count, 0ull, YVEX_ERR_BOUNDS,
                                "benchmark samples must be positive, finite, and complete", err);
    if (device_requested && !benchmark_samples_summarize(device_seconds, count, &device))
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "device-samples",
                                count, 0ull, YVEX_ERR_BOUNDS,
                                "device benchmark samples must be positive and finite", err);
    result->benchmark_sample_count = count;
    memcpy(result->benchmark_host_seconds, host.values, sizeof(result->benchmark_host_seconds));
    result->benchmark_first_execution_seconds = first_execution;
    result->benchmark_device_timing_available = device_requested;
    memcpy(result->benchmark_device_seconds, device.values, sizeof(result->benchmark_device_seconds));
    yvex_core_text_copy(result->benchmark_scope, sizeof(result->benchmark_scope), "attention_component");
    for (status = 0u; status < BENCHMARK_MEASURED_STATUS_COUNT; ++status)
        yvex_core_text_copy(result->quality_status[benchmark_measured_statuses[status].index],
                            24u, benchmark_measured_statuses[status].value);
    result->benchmark_correctness_precondition_passed = 1;
    result->benchmark_runtime_precondition_passed = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: identify the CPU execution host for an identity-bound baseline.
 * Inputs: bounded labels and error output. Effects: reads kernel host facts into both labels.
 * Failure: typed refusal if unavailable/oversized. Boundary: does not probe runtime capability. */
static int benchmark_cpu_identity(char device[YVEX_RUNTIME_BENCHMARK_TEXT_CAP],
                                  char driver[YVEX_RUNTIME_BENCHMARK_TEXT_CAP],
                                  unsigned long long *memory_bytes,
                                  yvex_error *err)
{
    struct utsname host;
    long processors, pages, page_size;
    int device_length, driver_length;

    if (uname(&host) != 0 || (processors = sysconf(_SC_NPROCESSORS_ONLN)) <= 0 ||
        (pages = sysconf(_SC_PHYS_PAGES)) <= 0 || (page_size = sysconf(_SC_PAGESIZE)) <= 0 ||
        !memory_bytes || !yvex_core_u64_mul((unsigned long long)pages,
                                            (unsigned long long)page_size, memory_bytes))
        return benchmark_fail(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "device", YVEX_ERR_IO,
            "CPU benchmark host identity is unavailable", err);
    device_length = snprintf(device, YVEX_RUNTIME_BENCHMARK_TEXT_CAP,
                             "cpu:%s:%s:%ld", host.nodename, host.machine, processors);
    driver_length = snprintf(
        driver, YVEX_RUNTIME_BENCHMARK_TEXT_CAP, "kernel:%s:%s", host.sysname, host.release);
    if (device_length < 0 || device_length >= (int)YVEX_RUNTIME_BENCHMARK_TEXT_CAP ||
        driver_length < 0 || driver_length >= (int)YVEX_RUNTIME_BENCHMARK_TEXT_CAP)
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                                "device", YVEX_RUNTIME_BENCHMARK_TEXT_CAP, 0ull,
                                YVEX_ERR_BOUNDS, "CPU benchmark identity is too long", err);
    return YVEX_OK;
}
/* Purpose: project immutable execution and build facts into one benchmark key.
 * Inputs: production result and key. Effects: copies bounded execution and machine facts.
 * Failure: unavailable/oversized facts refuse. Boundary: neither seals evidence nor applies policy. */
static int benchmark_attention_key(const yvex_graph_attention_operator_result *result,
                                   yvex_runtime_benchmark_key *key, yvex_error *err)
{
    char cpu_device[YVEX_RUNTIME_BENCHMARK_TEXT_CAP] = {0};
    char cpu_driver[YVEX_RUNTIME_BENCHMARK_TEXT_CAP] = {0};
    unsigned long long memory_bytes = 0ull;
    const int cuda = strcmp(result->backend, "cuda") == 0;
    const char *device = result->probe.cuda_device, *driver = result->cuda_driver;
    unsigned char *destination = (unsigned char *)key;
    const unsigned char *source = (const unsigned char *)result;
    size_t index;

    if (benchmark_cpu_identity(cpu_device, cpu_driver, &memory_bytes, err) != YVEX_OK)
        return yvex_error_code(err);
    *key = bench_key_defaults;
    for (index = 0u; index < BENCHMARK_KEY_TEXT_COUNT; ++index) {
        const benchmark_text_projection *field = &bench_key_result_fields[index];
        if (!benchmark_text_copy((char *)destination + field->destination, field->capacity,
                                 (const char *)source + field->source))
            return benchmark_fail(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "key", YVEX_ERR_BOUNDS,
                "attention benchmark key exceeds its fixed capacity", err);
    }
    if (!benchmark_text_copy(key->kernel_bundle_identity, sizeof(key->kernel_bundle_identity),
                             cuda ? result->cuda_build_identity : YVEX_BUILD_IDENTITY) ||
        !benchmark_text_copy(key->machine_identity, sizeof(key->machine_identity),
                             cuda ? device : cpu_device) ||
        !benchmark_text_copy(key->cpu_model, sizeof(key->cpu_model), cpu_device) ||
        !benchmark_text_copy(key->gpu_model, sizeof(key->gpu_model),
                             cuda ? device : "not-applicable") ||
        !benchmark_text_copy(key->device, sizeof(key->device), cuda ? device : cpu_device) ||
        !benchmark_text_copy(key->driver, sizeof(key->driver), cuda ? driver : cpu_driver) ||
        !benchmark_text_copy(key->cuda_build, sizeof(key->cuda_build),
                             cuda ? result->cuda_build_identity : "not-applicable"))
        return benchmark_fail(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD, "machine", YVEX_ERR_BOUNDS,
            "attention benchmark host facts exceed their capacity", err);
    key->memory_bytes = memory_bytes;
    key->compute_capability_major = (unsigned long long)result->probe.cuda_compute_capability_major;
    key->compute_capability_minor = (unsigned long long)result->probe.cuda_compute_capability_minor;
    for (index = 0u; index < BENCHMARK_KEY_U64_COUNT; ++index)
        memcpy(destination + bench_key_result_u64_fields[index].destination,
               source + bench_key_result_u64_fields[index].source, sizeof(unsigned long long));
    return YVEX_OK;
}
/* Purpose: convert one complete seconds distribution into canonical nanoseconds. */
static int benchmark_timing_project(
    const benchmark_seconds_distribution *source, int available,
    yvex_runtime_benchmark_timing_distribution *target)
{
    size_t index;

    memset(target, 0, sizeof(*target));
    target->available = available;
    for (index = 0u; index < YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT; ++index) {
        if ((!available && source->values[index] != 0.0) ||
            (available && !benchmark_seconds_ns(source->values[index], &target->values[index])))
            return 0;
    }
    return !available || target->values[YVEX_RUNTIME_BENCHMARK_MINIMUM] != 0ull;
}
/* Purpose: project measured lifecycle and latency values into canonical nanoseconds.
 * Inputs: production timings and metrics. Effects: converts finite durations under one rounding contract.
 * Failure: invalid/unrepresentable timing refuses. Boundary: does not measure, publish, or interpret. */
static int benchmark_attention_times(const yvex_graph_attention_operator_result *result,
                                     yvex_runtime_benchmark_metrics *metrics,
                                     yvex_error *err)
{
    benchmark_seconds_distribution host, device;
    unsigned long long *cold_ns = &metrics->cold_artifact_ns;
    double cold_seconds[BENCHMARK_COLD_GROUP_COUNT] = {0};
    double total = 0.0;
    size_t group, phase;

    memcpy(host.values, result->benchmark_host_seconds, sizeof(host.values));
    memcpy(device.values, result->benchmark_device_seconds, sizeof(device.values));
    for (group = 0u; group < BENCHMARK_COLD_GROUP_COUNT; ++group) {
        for (phase = 0u; phase < benchmark_cold_groups[group].count; ++phase)
            cold_seconds[group] += result->lifecycle_seconds[
                benchmark_cold_groups[group].first + phase];
        total += cold_seconds[group];
        if (!benchmark_seconds_ns(cold_seconds[group], &cold_ns[group]))
            goto timing_failure;
    }
    if (!benchmark_seconds_ns(total, &metrics->cold_total_ns) ||
        !benchmark_seconds_ns(result->benchmark_first_execution_seconds, &metrics->first_execution_ns) ||
        !benchmark_seconds_ns(result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_PUBLICATION],
                              &metrics->publication_ns) ||
        !benchmark_seconds_ns(result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_CLEANUP],
                              &metrics->cleanup_ns) ||
        !benchmark_timing_project(&host, 1, &metrics->host_timing))
        goto timing_failure;
    if (result->benchmark_device_timing_available != 0 &&
        result->benchmark_device_timing_available != 1)
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                                "device-timing-availability", 1ull,
                                (unsigned long long)result->benchmark_device_timing_available,
                                YVEX_ERR_FORMAT, "device timing availability is not canonical", err);
    if (!benchmark_timing_project(&device, result->benchmark_device_timing_available,
                                  &metrics->device_timing))
        return benchmark_reject(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS,
                                "device-timing", result->benchmark_device_timing_available, 0ull,
                                YVEX_ERR_BOUNDS,
                                "device timing values disagree with availability", err);
    return YVEX_OK;

timing_failure:
    return benchmark_fail(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS, "timing", YVEX_ERR_BOUNDS,
        "attention benchmark timing is not representable", err);
}
/* Purpose: project resource and dispatch counters without reconstructing runtime truth.
 * Inputs: production counters and metrics. Effects: copies counters and derives checked memory extents.
 * Failure: accounting overflow refuses. Boundary: copied evidence cannot infer capability. */
static int benchmark_attention_metrics(const yvex_graph_attention_operator_result *result,
                                       yvex_runtime_benchmark_metrics *metrics,
                                       yvex_error *err)
{
    const unsigned char *source = (const unsigned char *)result;
    unsigned char *destination = (unsigned char *)metrics;
    unsigned long long peak;
    size_t index;

    for (index = 0u; index < BENCHMARK_METRIC_PROJECTION_COUNT; ++index)
        memcpy(destination + benchmark_metric_result_fields[index].destination,
               source + benchmark_metric_result_fields[index].source,
               sizeof(unsigned long long));
    metrics->h2d_bytes = result->probe.h2d_bytes;
    metrics->d2h_bytes = result->probe.d2h_bytes;
    metrics->last_dispatch_kernel_launches = result->probe.kernel_launches;
    metrics->last_dispatch_peak_device_bytes = result->probe.peak_device_bytes;
    if (!yvex_core_u64_add(result->host_resident_bytes, result->device_resident_bytes,
                           &metrics->resident_bytes) ||
        !yvex_core_u64_add(result->host_resident_bytes, result->workspace_bytes, &peak) ||
        !yvex_core_u64_add(peak, result->pinned_host_peak_bytes, &peak) ||
        !yvex_core_u64_add(peak, result->state_allocated_bytes, &metrics->peak_host_bytes))
        return benchmark_fail(NULL, YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS, "memory", YVEX_ERR_BOUNDS,
            "attention benchmark memory accounting overflowed", err);
    return YVEX_OK;
}
/* Purpose: validate a full lowercase hexadecimal commit identity.
 * Inputs: fixed-capacity NUL-terminated commit field. Effects: none.
 * Failure: false for noncanonical hex. Boundary: validation does not discover repository state. */
static int commit_valid(const char commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP])
{
    return commit && strnlen(commit, YVEX_RUNTIME_BENCHMARK_COMMIT_CAP) == 40u &&
           strspn(commit, "0123456789abcdef") == 40u;
}
/* Purpose: admit only the two source-tree states captured by the generated build provenance. */
static int source_state_valid(const char state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP])
{
    return state && (strcmp(state, "clean") == 0 || strcmp(state, "dirty") == 0);
}

typedef struct {
    const char *name;
    size_t offset;
    const char *const *values;
    size_t count;
} benchmark_vocabulary;

static const char *const benchmark_modes[] = {"eager", "piecewise", "full"};
static const char *const benchmark_phases[] = {"prefill", "decode"};
static const char *const benchmark_scopes[] = {"core", "envelope", "release-attention-set"};
static const char *const benchmark_coverages[] = {"quick", "full"};
static const char *const benchmark_classes[] = {"swa", "csa", "hca", "mixed"};
static const char *const benchmark_traces[] = {"none", "summary", "stages", "full"};
static const benchmark_vocabulary benchmark_vocabularies[] = {
    {"mode", offsetof(bench_key, mode), benchmark_modes, 3u},
{"phase", offsetof(bench_key, phase), benchmark_phases, 2u},
    {"scope", offsetof(bench_key, scope), benchmark_scopes, 3u},
{"coverage", offsetof(bench_key, coverage), benchmark_coverages, 2u},
    {"attention_class", offsetof(bench_key, attention_class), benchmark_classes, 4u},
{"trace_policy", offsetof(bench_key, trace_policy), benchmark_traces, 4u},
};
enum {
    BENCHMARK_VOCABULARY_COUNT =
        sizeof(benchmark_vocabularies) / sizeof(benchmark_vocabularies[0])
};
/* Purpose: name the first execution vocabulary field outside the canonical benchmark domain.
 * Inputs: one fully populated benchmark key. Effects: none.
 * Failure: names the first invalid field. Boundary: neither repairs values nor infers support. */
static const char *key_vocabulary_mismatch(const yvex_runtime_benchmark_baseline *record)
{
    const unsigned char *key = (const unsigned char *)&record->key;
    size_t rule, value;

    if (strcmp(record->key.benchmark_scope, "attention_component") != 0)
        return "benchmark_scope";
    for (rule = 0u; rule < BENCHMARK_VOCABULARY_COUNT; ++rule) {
        const benchmark_vocabulary *vocabulary = &benchmark_vocabularies[rule];
        const char *text = (const char *)key + vocabulary->offset;
        for (value = 0u; value < vocabulary->count; ++value)
            if (strcmp(text, vocabulary->values[value]) == 0) break;
        if (value == vocabulary->count) return vocabulary->name;
    }
    return NULL;
}
/* Purpose: validate the single canonical key-field schema used by hash, I/O, and equality.
 * Inputs: complete record plus typed refusal outputs. Effects: writes refusal state only.
 * Failure: rejects the first malformed text field. Boundary: geometry and metrics validate separately. */
static int key_validate(const yvex_runtime_benchmark_baseline *record,
                        yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    const unsigned char *base = (const unsigned char *)record;
    const char *vocabulary;
    size_t index;

    if (!record)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "key",
            YVEX_ERR_INVALID_ARG, "benchmark key is required", err);
    for (index = 0u; index < BENCHMARK_KEY_FIELD_COUNT; ++index) {
        const benchmark_field *field = &bench_key_fields[index];
        const char *text = (const char *)(base + field->offset);
        int valid = field->kind == BENCHMARK_FIELD_U64 ||
                    (field->kind == BENCHMARK_FIELD_COMMIT && commit_valid(text)) ||
                    (field->kind == BENCHMARK_FIELD_SOURCE_STATE && source_state_valid(text)) ||
                    (field->kind == BENCHMARK_FIELD_SHA256 && yvex_sha256_hex_valid(text)) ||
                    (field->kind >= BENCHMARK_FIELD_TEXT_16 && field->kind <= BENCHMARK_FIELD_TEXT_128 &&
                     text_valid(text, benchmark_field_widths[field->kind]));
        if (!valid) {
            unsigned long long expected = field->kind == BENCHMARK_FIELD_COMMIT ? 40ull :
                                          field->kind == BENCHMARK_FIELD_SHA256 ? 64ull : 1ull;
            return benchmark_field_reject(
                failure, field->name, expected, strnlen(text, benchmark_field_widths[field->kind]),
                YVEX_ERR_FORMAT,
                field->kind == BENCHMARK_FIELD_COMMIT
                    ? "benchmark commit must be a full lowercase SHA"
                : field->kind == BENCHMARK_FIELD_SOURCE_STATE
                    ? "benchmark source state must be clean or dirty"
                : field->kind == BENCHMARK_FIELD_SHA256
                    ? "benchmark identity field is invalid"
                    : "benchmark text field is invalid",
                err);
        }
    }
    if ((strcmp(record->key.build_source_state, "clean") == 0) !=
        (strcmp(record->key.source_delta_identity, BENCHMARK_EMPTY_SOURCE_DELTA) == 0))
        return benchmark_field_fail(failure, "source_delta_identity", YVEX_ERR_FORMAT,
            "benchmark source state and exact delta identity disagree", err);
    vocabulary = key_vocabulary_mismatch(record);
    if (vocabulary)
        return benchmark_field_fail(failure, vocabulary, YVEX_ERR_FORMAT,
            "benchmark execution vocabulary is not canonical", err);
    if (!record->key.iteration_count)
        return benchmark_field_fail(failure, "iteration_count", YVEX_ERR_BOUNDS,
            "benchmark iteration count must be positive", err);
    return YVEX_OK;
}
/* Purpose: append canonical schema fields to a hash without hashing native structure bytes.
 * Inputs: hash, record, and field table. Effects: appends each scalar once in canonical order.
 * Failure: false on hash failure. Boundary: excludes pointers, padding, native layout, and paths. */
static int fields_hash(yvex_sha256 *hash, const yvex_runtime_benchmark_baseline *record,
                       const benchmark_field *fields, size_t count)
{
    const unsigned char *base = (const unsigned char *)record;
    size_t index;

    for (index = 0u; index < count; ++index) {
        unsigned long long value;
        if (fields[index].kind == BENCHMARK_FIELD_BOOL) {
            int boolean;
            memcpy(&boolean, base + fields[index].offset, sizeof(boolean));
            if (!yvex_sha256_update_u64(hash, (unsigned long long)boolean)) return 0;
            continue;
        }
        if (fields[index].kind != BENCHMARK_FIELD_U64) {
            if (!yvex_sha256_update_text(hash, (const char *)(base + fields[index].offset)))
                return 0;
            continue;
        }
        memcpy(&value, base + fields[index].offset, sizeof(value));
        if (!yvex_sha256_update_u64(hash, value)) return 0;
    }
    return 1;
}
/* Purpose: serialize canonical schema fields in their single declared order.
 * Inputs: byte sink, record, and field table. Effects: appends one canonical line per field.
 * Failure: false on format/allocation/bounds. Boundary: validation and identity remain lifecycle-owned. */
static int fields_serialize(benchmark_bytes *bytes,
                            const yvex_runtime_benchmark_baseline *record,
                            const benchmark_field *fields, size_t count)
{
    const unsigned char *base = (const unsigned char *)record;
    size_t index;

    for (index = 0u; index < count; ++index) {
        unsigned long long value;
        if (fields[index].kind == BENCHMARK_FIELD_BOOL) {
            int boolean;
            memcpy(&boolean, base + fields[index].offset, sizeof(boolean));
            if (!bytes_format(bytes, "%s\t%d\n", fields[index].name, boolean)) return 0;
            continue;
        }
        if (fields[index].kind != BENCHMARK_FIELD_U64) {
            if (!bytes_format(bytes, "%s\t%s\n", fields[index].name,
                              (const char *)(base + fields[index].offset)))
                return 0;
            continue;
        }
        memcpy(&value, base + fields[index].offset, sizeof(value));
        if (!bytes_format(bytes, "%s\t%016llx\n", fields[index].name, value)) return 0;
    }
    return 1;
}
/* Purpose: validate one host or device distribution under an exact availability contract.
 * Inputs: immutable canonical timing values and the backend-required availability bit. Effects: none.
 * Failure: false for residue, disorder, zero minimum, or overflow. Boundary: assigns no policy. */
static int timing_valid(const yvex_runtime_benchmark_timing_distribution *timing,
                        int expected_available)
{
    const unsigned long long *value = timing->values;
    size_t index;

    if (timing->available != expected_available) return 0;
    if (!timing->available) {
        for (index = 0u; index < YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT; ++index)
            if (value[index]) return 0;
        return 1;
    }
    return value[YVEX_RUNTIME_BENCHMARK_MINIMUM] != 0ull &&
           value[YVEX_RUNTIME_BENCHMARK_MINIMUM] <= value[YVEX_RUNTIME_BENCHMARK_P50] &&
           value[YVEX_RUNTIME_BENCHMARK_P50] <= value[YVEX_RUNTIME_BENCHMARK_P90] &&
           value[YVEX_RUNTIME_BENCHMARK_P90] <= value[YVEX_RUNTIME_BENCHMARK_P95] &&
           value[YVEX_RUNTIME_BENCHMARK_P95] <= value[YVEX_RUNTIME_BENCHMARK_P99] &&
           value[YVEX_RUNTIME_BENCHMARK_P99] <= value[YVEX_RUNTIME_BENCHMARK_MAXIMUM] &&
           value[YVEX_RUNTIME_BENCHMARK_MEAN] >= value[YVEX_RUNTIME_BENCHMARK_MINIMUM] &&
           value[YVEX_RUNTIME_BENCHMARK_MEAN] <= value[YVEX_RUNTIME_BENCHMARK_MAXIMUM] &&
           value[YVEX_RUNTIME_BENCHMARK_MAXIMUM] <= (unsigned long long)LLONG_MAX &&
           value[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION] <=
               (unsigned long long)LLONG_MAX;
}
/* Purpose: validate ordered timing, resource, and execution-mode benchmark evidence.
 * Inputs: benchmark record and refusal outputs. Effects: writes refusal state only on failure.
 * Failure: rejects inconsistent evidence. Boundary: establishes consistency, not performance policy. */
static int metrics_validate(const yvex_runtime_benchmark_baseline *record,
                            yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    const yvex_runtime_benchmark_metrics *metrics;
    const yvex_runtime_benchmark_timing_distribution *device;
    const unsigned long long *cold_phase;
    unsigned long long cold = 0ull;
    unsigned int graph_fields, phase;
    int eager, cuda;

    if (!record)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "metrics",
            YVEX_ERR_INVALID_ARG, "benchmark metrics are required", err);
    metrics = &record->metrics;
    device = &metrics->device_timing;
    if (!metrics->cold_total_ns || !timing_valid(&metrics->host_timing, 1) ||
        metrics->cold_total_ns > (unsigned long long)LLONG_MAX)
        return benchmark_field_fail(failure, "metrics", YVEX_ERR_BOUNDS,
            "benchmark nanosecond statistics are invalid or unordered", err);
    cuda = strcmp(record->key.cuda_build, "not-applicable") != 0;
    if (!timing_valid(device, cuda))
        return benchmark_field_reject(
            failure, "device-timing", cuda ? 1ull : 0ull,
            (unsigned long long)(device->available == 1), YVEX_ERR_BOUNDS,
            "device timing availability or distribution is invalid", err);
    if (!metrics->artifact_bytes_hashed || !metrics->resident_encoded_bytes ||
        !metrics->resident_bytes || !metrics->state_bytes)
        return benchmark_field_fail(failure, "structural-evidence", YVEX_ERR_BOUNDS,
            "benchmark structural byte evidence is incomplete", err);
    cold_phase = &metrics->cold_artifact_ns;
    for (phase = 0u; phase < 8u; ++phase)
        if (!yvex_core_u64_add(cold, cold_phase[phase], &cold))
            return benchmark_field_reject(
                failure, "resource-accounting", metrics->cold_total_ns, cold,
                YVEX_ERR_BOUNDS, "benchmark cold-phase accounting overflowed", err);
    if (cold != metrics->cold_total_ns ||
        metrics->resident_encoded_bytes > metrics->resident_bytes ||
        metrics->peak_host_bytes < metrics->host_workspace_bytes ||
        metrics->peak_host_bytes < metrics->state_bytes)
        return benchmark_field_reject(
            failure, "resource-accounting", metrics->cold_total_ns, cold,
            YVEX_ERR_BOUNDS,
            "benchmark phase or owned-memory accounting is inconsistent", err);
    graph_fields = (metrics->cuda_graph_launches != 0ull) +
                   (metrics->cuda_graph_captures != 0ull) +
                   (metrics->cuda_graph_replays != 0ull) +
                   (metrics->cuda_graph_nodes != 0ull);
    eager = strcmp(record->key.mode, "eager") == 0;
    if ((eager && graph_fields) || (!eager && graph_fields != 4u))
        return benchmark_field_reject(
            failure, "execution-mode-evidence", eager ? 0ull : 4ull, graph_fields,
            YVEX_ERR_STATE,
            eager ? "eager benchmark cannot claim CUDA Graph evidence"
                  : "graph benchmark mode requires four complete CUDA Graph counters",
            err);
    if (metrics->warm_weight_reads || metrics->warm_upload_bytes ||
        metrics->warm_host_allocations || metrics->warm_device_allocations ||
        metrics->warm_device_frees)
        return benchmark_field_reject(
            failure, "steady-state", 0ull, 1ull, YVEX_ERR_STATE,
            "benchmark steady-state allocation or weight-transfer invariant failed", err);
    return YVEX_OK;
}
/* Purpose: hash one record using canonical scalar values rather than C object representation.
 * Inputs: validated record and fixed-capacity identity output. Effects: writes one lowercase SHA-256 identity.
 * Failure: false on hash failure. Boundary: excludes pointers, padding, native layout, and paths. */
static int baseline_identity(const yvex_runtime_benchmark_baseline *record,
                             char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.benchmark.baseline.v5") ||
        !yvex_sha256_update_u64(&hash, record->schema_version) ||
        !fields_hash(&hash, record, bench_key_fields, BENCHMARK_KEY_FIELD_COUNT) ||
        !fields_hash(&hash, record, benchmark_metric_fields, BENCHMARK_METRIC_FIELD_COUNT) ||
        !benchmark_hash_finish(&hash, output))
        return 0;
    return 1;
}
/* Purpose: prove a borrowed record is sealed under its exact canonical identity.
 * Inputs: immutable record plus optional typed refusal outputs. Effects: writes only refusal state on failure.
 * Failure: rejects schema/content/identity mismatch. Boundary: never changes borrowed evidence. */
static int baseline_validate(const yvex_runtime_benchmark_baseline *record,
                             yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_BYTES];
    int rc;

    if (!record)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "record",
            YVEX_ERR_INVALID_ARG, "benchmark record is required", err);
    if (record->schema_version != YVEX_RUNTIME_BENCHMARK_SCHEMA_V5)
        return benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA, "schema",
                                YVEX_RUNTIME_BENCHMARK_SCHEMA_V5, record->schema_version,
                                YVEX_ERR_UNSUPPORTED, "benchmark schema is unsupported", err);
    if ((rc = key_validate(record, failure, err)) != YVEX_OK ||
        (rc = metrics_validate(record, failure, err)) != YVEX_OK)
        return rc;
    if (!baseline_identity(record, identity) || strcmp(identity, record->identity) != 0)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_IDENTITY, "baseline_identity",
            YVEX_ERR_STATE, "benchmark baseline identity does not match canonical content", err);
    return YVEX_OK;
}
/* Purpose: serialize one validated record into the exact version-five line format.
 * Inputs: sealed record and byte buffer. Effects: appends each field in fixed order and encoding.
 * Failure: returns false on bound or allocation failure. Boundary: serialization does not publish or reopen files. */
static int baseline_serialize(const yvex_runtime_benchmark_baseline *record,
                              benchmark_bytes *bytes)
{
    bytes->maximum = BENCHMARK_FILE_MAX;
    bytes->initial_capacity = 1024u;
    return bytes_format(bytes, "YVEX_RUNTIME_BENCHMARK_BASELINE\t5\n") &&
           bytes_format(bytes, "identity\t%s\n", record->identity) &&
           fields_serialize(bytes, record, bench_key_fields, BENCHMARK_KEY_FIELD_COUNT) &&
           fields_serialize(bytes, record, benchmark_metric_fields, BENCHMARK_METRIC_FIELD_COUNT);
}
/* Purpose: translate core file mechanics into the benchmark evidence failure vocabulary.
 * Inputs: one typed core lifecycle stage. Effects: returns a domain code without changing either failure object.
 * Failure: unknown stages fail closed. Boundary: preserves filesystem status and evidence bytes. */
static yvex_runtime_benchmark_failure_code benchmark_file_code(yvex_core_file_stage stage)
{
    return (unsigned int)stage < BENCHMARK_FILE_CODE_COUNT
               ? benchmark_file_codes[stage] : YVEX_RUNTIME_BENCHMARK_FAILURE_PUBLISH;
}
/* Purpose: name the exact owned cleanup operation retained by the core file lifecycle. */
static const char *benchmark_cleanup_field(yvex_core_file_cleanup_stage stage)
{
    switch (stage) {
    case YVEX_CORE_FILE_CLEANUP_FILE_CLOSE: return "cleanup-file-close";
    case YVEX_CORE_FILE_CLEANUP_TEMPORARY_UNLINK: return "cleanup-temporary-unlink";
    case YVEX_CORE_FILE_CLEANUP_DESTINATION_UNLINK: return "cleanup-destination-unlink";
    case YVEX_CORE_FILE_CLEANUP_DIRECTORY_SYNC: return "cleanup-directory-sync";
    case YVEX_CORE_FILE_CLEANUP_NONE: break;
    }
    return "file-cleanup";
}
/* Purpose: transactionally publish exact bytes through one no-symlink, no-replace file lifecycle.
 * Inputs: safe path, exact immutable bytes, optional lifecycle fault controls, and typed
 * publication/refusal outputs.
 * Effects: creates, writes, syncs, atomically links, and directory-syncs one file.
 * Failure: removes only its owned candidate. Boundary: bytes are canonical and faults remain explicit. */
static int publish_bytes(const char *path, const void *data, size_t count,
                         const yvex_core_file_faults *faults,
                         yvex_runtime_benchmark_publication *result,
                         yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    yvex_core_file_result file_result;
    int rc;

    if (result) memset(result, 0, sizeof(*result));
    if (!result || !data || !count || !path || path[0] != '/')
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_PATH, "path", YVEX_ERR_INVALID_ARG,
            "benchmark publication requires an absolute external path", err);
    memset(&file_result, 0, sizeof(file_result));
    rc = yvex_core_file_publish_noreplace(
        path, data, count, faults, NULL, NULL, &file_result, err);
    if (file_result.cleanup_stage != YVEX_CORE_FILE_CLEANUP_NONE)
        return benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_CLEANUP,
                                benchmark_cleanup_field(file_result.cleanup_stage), 0ull,
                                (unsigned long long)file_result.cleanup_system_error,
                                YVEX_ERR_IO, "benchmark file cleanup failed", err);
    if (rc != YVEX_OK)
        return benchmark_reject(failure, benchmark_file_code(file_result.stage),
                                "file-lifecycle", file_result.expected, file_result.actual, (yvex_status)rc,
                                "benchmark file lifecycle failed", err);
    result->published = 1;
    result->file_bytes = count;
    yvex_core_text_copy(result->path, sizeof(result->path), path);
    yvex_error_clear(err);
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}
/* Purpose: extract one exact key/value line in canonical order without accepting duplicates.
 * Inputs: bounded cursor, key, and value output. Effects: terminates one line and advances once.
 * Failure: false for malformed lines. Boundary: callers own field capacity and type parsing. */
static int parse_line(char **cursor, char *end, const char *key, char **value)
{
    char *line_end, *separator;
    size_t key_length = strlen(key);

    if (!cursor || !*cursor || *cursor >= end) return 0;
    line_end = memchr(*cursor, '\n', (size_t)(end - *cursor));
    if (!line_end) return 0;
    separator = memchr(*cursor, '\t', (size_t)(line_end - *cursor));
    if (!separator || (size_t)(separator - *cursor) != key_length ||
        memcmp(*cursor, key, key_length) != 0 || separator + 1 == line_end)
        return 0;
    *separator = '\0';
    *line_end = '\0';
    *value = separator + 1;
    *cursor = line_end + 1;
    return 1;
}
/* Purpose: decode every field in one schema table from canonical ordered text lines.
 * Inputs: cursor, record, and field table. Effects: parses one ordered line into each declared field.
 * Failure: rejects order/size/encoding. Boundary: semantic validation follows complete parsing. */
static int fields_parse(char **cursor, char *end,
                        yvex_runtime_benchmark_baseline *record,
                        const benchmark_field *fields, size_t count)
{
    unsigned char *base = (unsigned char *)record;
    size_t index;

    for (index = 0u; index < count; ++index) {
        const benchmark_field *field = &fields[index];
        char *value, *tail;
        unsigned long long parsed;
        if (!parse_line(cursor, end, field->name, &value)) return 0;
        if (field->kind == BENCHMARK_FIELD_BOOL) {
            int boolean;
            if ((value[0] != '0' && value[0] != '1') || value[1]) return 0;
            boolean = value[0] == '1';
            memcpy(base + field->offset, &boolean, sizeof(boolean));
            continue;
        }
        if (field->kind != BENCHMARK_FIELD_U64) {
            if (strlen(value) >= benchmark_field_widths[field->kind]) return 0;
            yvex_core_text_copy((char *)(base + field->offset), benchmark_field_widths[field->kind], value);
            continue;
        }
        if (strlen(value) != 16u) return 0;
        errno = 0;
        parsed = strtoull(value, &tail, 16);
        if (errno || tail != value + 16u || *tail) return 0;
        memcpy(base + field->offset, &parsed, sizeof(parsed));
    }
    return 1;
}
/* Purpose: parse every canonical baseline field from one mutable bounded file buffer.
 * Inputs: exact mutable bytes and record. Effects: tokenizes and fills each v5 field once.
 * Failure: false for missing/reordered/trailing data. Boundary: reopen owns identity and byte equality. */
static int baseline_parse(char *data, size_t count, yvex_runtime_benchmark_baseline *record)
{
    char *cursor = data, *end = data + count, *header, *identity;

    memset(record, 0, sizeof(*record));
    if (!parse_line(&cursor, end, "YVEX_RUNTIME_BENCHMARK_BASELINE", &header))
        return 0;
    if (header[0] < '1' || header[0] > '5' || header[1]) return 0;
    record->schema_version = (unsigned int)(header[0] - '0');
    if (record->schema_version != YVEX_RUNTIME_BENCHMARK_SCHEMA_V5) return 0;
    if (!parse_line(&cursor, end, "identity", &identity) ||
        strlen(identity) >= sizeof(record->identity) ||
        !fields_parse(&cursor, end, record, bench_key_fields, BENCHMARK_KEY_FIELD_COUNT) ||
        !fields_parse(&cursor, end, record, benchmark_metric_fields, BENCHMARK_METRIC_FIELD_COUNT) ||
        cursor != end)
        return 0;
    record->metrics.host_timing.available = 1;
    yvex_core_text_copy(record->identity, sizeof(record->identity), identity);
    return 1;
}
/* Purpose: read one stable regular file through a no-symlink path and exact bounded snapshot.
 * Inputs: safe path and owned outputs. Effects: allocates and fills one exact byte buffer.
 * Failure: rejects unsafe, changed, or unreadable files. Boundary: callers own parsing and identity. */
static int read_file(const char *path, char **data, size_t *count,
                     yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    yvex_core_file_result file_result;
    unsigned char *bytes = NULL;
    int rc;

    if (!path || path[0] != '/')
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_PATH, "path", YVEX_ERR_INVALID_ARG,
            "benchmark evidence requires an absolute external path", err);
    memset(&file_result, 0, sizeof(file_result));
    rc = yvex_core_file_read_snapshot(path, BENCHMARK_FILE_MAX, &bytes, count,
                                      &file_result, err);
    if (rc != YVEX_OK)
        return benchmark_reject(failure, benchmark_file_code(file_result.stage),
                                "file-lifecycle", file_result.expected, file_result.actual, (yvex_status)rc,
                                "benchmark file snapshot failed", err);
    *data = (char *)bytes;
    return YVEX_OK;
}
/* Purpose: compare workload compatibility while retaining commit and source state provenance.
 * Inputs: two validated records. Effects: compares canonical keys without mutation.
 * Failure: names the first incompatibility. Boundary: commits may differ; build/source identities may not. */
static const char *keys_mismatch(const yvex_runtime_benchmark_baseline *left,
                                 const yvex_runtime_benchmark_baseline *right)
{
    const unsigned char *left_base = (const unsigned char *)left;
    const unsigned char *right_base = (const unsigned char *)right;
    size_t index;

    for (index = 0u; index < BENCHMARK_KEY_FIELD_COUNT; ++index) {
        const benchmark_field *field = &bench_key_fields[index];
        const void *left_value = left_base + field->offset;
        const void *right_value = right_base + field->offset;
        if (field->kind == BENCHMARK_FIELD_COMMIT) continue;
        if (memcmp(left_value, right_value, benchmark_field_widths[field->kind]) != 0)
            return field->name;
    }
    return NULL;
}
/* Purpose: produce one signed measured delta after metrics validation bounded both operands. */
static long long metric_delta(unsigned long long current, unsigned long long baseline)
{
    return (long long)current - (long long)baseline;
}
typedef struct {
    size_t offset;
    int inverse, device;
} benchmark_regression_field;

static const benchmark_regression_field benchmark_regression_fields[] = {
    {offsetof(bench_metrics, host_timing.values[YVEX_RUNTIME_BENCHMARK_P50]), 0, 0},
    {offsetof(bench_metrics, device_timing.values[YVEX_RUNTIME_BENCHMARK_P50]), 0, 1},
    {offsetof(bench_metrics, host_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN]), 1, 0},
    {offsetof(bench_metrics, device_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN]), 1, 1},
    {offsetof(bench_metrics, peak_host_bytes), 0, 0},
    {offsetof(bench_metrics, last_dispatch_peak_device_bytes), 0, 1},
    {offsetof(bench_metrics, h2d_bytes), 0, 0}, {offsetof(bench_metrics, d2h_bytes), 0, 0},
    {offsetof(bench_metrics, warm_host_allocations), 0, 0},
    {offsetof(bench_metrics, warm_device_allocations), 0, 1},
    {offsetof(bench_metrics, warm_device_frees), 0, 1},
    {offsetof(bench_metrics, last_dispatch_kernel_launches), 0, 0},
    {offsetof(bench_metrics, cuda_graph_launches), 0, 0},
};
enum {
    BENCHMARK_REGRESSION_FIELD_COUNT =
        sizeof(benchmark_regression_fields) / sizeof(benchmark_regression_fields[0])
};

/* Purpose: classify one measured ratio against a shared caller-selected basis-point ceiling. */
static int regression_exceeded(unsigned long long current, unsigned long long baseline,
                               unsigned long long basis_points, int inverse)
{
    if (!inverse)
        return (long double)current * 10000.0L > (long double)baseline * (10000.0L + (long double)basis_points);
    return basis_points < 10000ull && (long double)baseline * 10000.0L <
               (long double)current * (long double)(10000ull - basis_points);
}
/* Purpose: derive one canonical threshold policy identity without hashing native storage. */
static int regression_policy_identity(
    const yvex_runtime_benchmark_regression_policy *policy,
    char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;

    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.benchmark.regression-policy.v1") ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)policy->enabled) ||
        !yvex_sha256_update_u64(&hash, policy->basis_points))
        return 0;
    return benchmark_hash_finish(&hash, output);
}
/* Purpose: evaluate configured performance dimensions without affecting correctness status.
 * Inputs: compatible metrics and threshold policy. Effects: publishes identity-bound performance status.
 * Failure: invalid policy/counters refuse. Boundary: no threshold means measured, not release-passed. */
static int regression_policy_apply(
    const yvex_runtime_benchmark_baseline *current,
    const yvex_runtime_benchmark_baseline *baseline,
    const yvex_runtime_benchmark_regression_policy *requested,
    yvex_runtime_benchmark_comparison *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    yvex_runtime_benchmark_regression_policy policy = {0};
    const unsigned char *current_metrics = (const unsigned char *)&current->metrics;
    const unsigned char *baseline_metrics = (const unsigned char *)&baseline->metrics;
    yvex_sha256 hash;
    size_t index;

    if (requested) policy = *requested;
    if ((policy.enabled != 0 && policy.enabled != 1) ||
        (!policy.enabled && policy.basis_points))
        return benchmark_field_reject(failure, "regression_policy", 1ull,
                                      (unsigned long long)policy.enabled,
                                      YVEX_ERR_INVALID_ARG,
                                      "benchmark regression policy is not canonical", err);
    result->regression_policy = policy;
    if (!regression_policy_identity(&policy, result->regression_policy_identity))
        return benchmark_field_fail(failure, "regression_policy_identity", YVEX_ERR_STATE,
            "benchmark regression policy identity failed", err);
    result->performance_passed = 1;
    for (index = 0u; policy.enabled && index < BENCHMARK_REGRESSION_FIELD_COUNT; ++index) {
        const benchmark_regression_field *field = &benchmark_regression_fields[index];
        unsigned long long current_value, baseline_value;
        if (field->device && !current->metrics.device_timing.available) continue;
        memcpy(&current_value, current_metrics + field->offset, sizeof(current_value));
        memcpy(&baseline_value, baseline_metrics + field->offset, sizeof(baseline_value));
        if (regression_exceeded(current_value, baseline_value,
                                policy.basis_points, field->inverse))
            result->performance_passed = 0;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.benchmark.comparison.v1") ||
        !yvex_sha256_update_text(&hash, current->identity) ||
        !yvex_sha256_update_text(&hash, baseline->identity) ||
        !yvex_sha256_update_text(&hash, result->regression_policy_identity) ||
        !yvex_sha256_update_u64(&hash, (unsigned long long)result->performance_passed) ||
        !benchmark_hash_finish(&hash, result->comparison_identity))
        return benchmark_field_fail(failure, "comparison_identity", YVEX_ERR_STATE,
            "benchmark comparison identity failed", err);
    return YVEX_OK;
}
/* Purpose: escape one bounded identity/device label for deterministic SVG text. */
static int svg_text(benchmark_bytes *bytes, const char *text)
{
    const char *cursor, *start;
    for (cursor = start = text; *cursor; ++cursor) {
        const char *replacement = NULL;
        if (*cursor == '&') replacement = "&amp;";
        else if (*cursor == '<') replacement = "&lt;";
        else if (*cursor == '>') replacement = "&gt;";
        else if (*cursor == '\"') replacement = "&quot;";
        else if (*cursor == '\'') replacement = "&apos;";
        if (replacement) {
            if (!yvex_core_bytes_append(bytes, start, (size_t)(cursor - start)) ||
                !bytes_text(bytes, replacement))
                return 0;
            start = cursor + 1;
        }
    }
    return yvex_core_bytes_append(bytes, start, (size_t)(cursor - start));
}
/* Purpose: scale one bar with integer half-up rounding that is identical across ABIs. */
static unsigned int chart_height(unsigned long long value, unsigned long long maximum,
                                 unsigned int extent)
{
    unsigned long long scale, quotient, remainder;
    unsigned int height;

    if (!value || !maximum || !extent) return 0u;
    if (value >= maximum) return extent;
    scale = (unsigned long long)extent * 2ull;
    quotient = maximum / scale;
    remainder = maximum % scale;
    for (height = 0u; height < extent; ++height) {
        unsigned long long factor = (unsigned long long)height * 2ull + 1ull;
        unsigned long long tail = remainder * factor;
        unsigned long long threshold = quotient * factor + tail / scale + (tail % scale != 0ull);
        if (value < threshold) break;
    }
    return height ? height : 1u;
}

static const char *const benchmark_chart_percentiles[] = {
    "min", "p50", "p90", "p99", "max"};
/* Purpose: hash canonical chart inputs independently from formatting or destination path. */
static int chart_bytes_identity(const benchmark_bytes *svg,
                                char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    yvex_sha256_init(&hash);
    return svg && yvex_sha256_update(&hash, svg->data, svg->count) &&
           benchmark_hash_finish(&hash, output);
}
/* Purpose: render five warm-latency percentiles for current and optional baseline evidence.
 * Inputs: SVG buffer and sealed records. Effects: appends deterministic percentile bars and legend.
 * Failure: false on bounds/allocation. Boundary: displays raw values without policy. */
static int chart_warm(benchmark_bytes *svg, const yvex_runtime_benchmark_baseline *current,
                      const yvex_runtime_benchmark_baseline *baseline)
{
    typedef struct {
        const yvex_runtime_benchmark_timing_distribution *timing;
        const char *style;
        unsigned int offset, width;
    } chart_series;
    const chart_series series[] = {
        {&current->metrics.host_timing, "current", 0u, 30u}, {&current->metrics.device_timing, "device", 10u, 10u},
{baseline ? &baseline->metrics.host_timing : NULL, "baseline", 32u, 30u},
{baseline ? &baseline->metrics.device_timing : NULL, "device_baseline", 42u, 10u},
    };
    unsigned long long maximum = 0ull;
    size_t index, series_index;

    for (series_index = 0u; series_index < sizeof(series) / sizeof(series[0]); ++series_index)
        if (series[series_index].timing && series[series_index].timing->available &&
            series[series_index].timing->values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] > maximum)
            maximum = series[series_index].timing->values[YVEX_RUNTIME_BENCHMARK_MAXIMUM];
    if (!bytes_text(svg,
                    "<text x=\"410\" y=\"142\" class=\"section\">WARM HOST / DEVICE</text>"
                    "<line x1=\"410\" y1=\"360\" x2=\"900\" y2=\"360\" class=\"axis\"/>"))
        return 0;
    for (index = 0u; index < 5u; ++index) {
        unsigned int x = 436u + (unsigned int)index * 94u;
        if (!bytes_format(svg, "<text x=\"%u\" y=\"382\" class=\"label\">%s</text>",
                          x + 27u, benchmark_chart_percentiles[index]))
            return 0;
        for (series_index = 0u; series_index < sizeof(series) / sizeof(series[0]); ++series_index) {
            const chart_series *item = &series[series_index];
            unsigned int height;
            if (!item->timing || !item->timing->available) continue;
            height = chart_height(item->timing->values[index], maximum, 190u);
            if (!bytes_format(svg,
                              "<rect x=\"%u\" y=\"%u\" width=\"%u\" height=\"%u\" class=\"%s\"/>",
                              x + item->offset, 360u - height, item->width, height, item->style))
                return 0;
        }
    }
    if (!bytes_text(svg,
                    "<circle cx=\"650\" cy=\"411\" r=\"6\" class=\"current\"/>"
                    "<text x=\"663\" y=\"416\" class=\"meta\">current</text>"))
        return 0;
    if (baseline &&
        !bytes_text(svg,
                    "<circle cx=\"750\" cy=\"411\" r=\"6\" class=\"baseline\"/>"
                    "<text x=\"763\" y=\"416\" class=\"meta\">baseline</text>"))
        return 0;
    if (!current->metrics.device_timing.available)
        return bytes_text(svg,
                          "<text x=\"650\" y=\"438\" class=\"meta\">DEVICE TIMING UNAVAILABLE</text>");
    return bytes_text(svg,
                      "<circle cx=\"850\" cy=\"411\" r=\"6\" class=\"device\"/>"
                      "<text x=\"863\" y=\"416\" class=\"meta\">device event</text>");
}
/* Purpose: render one exact resource counter as a current/baseline horizontal comparison. */
static int chart_resource_row(benchmark_bytes *svg, const char *label, unsigned int y,
                              unsigned long long current, unsigned long long baseline,
                              int has_baseline)
{
    unsigned long long maximum = current > baseline ? current : baseline;
    unsigned int current_width = chart_height(current, maximum, 220u);
    unsigned int baseline_width = chart_height(baseline, maximum, 220u);

    if (!bytes_format(svg,
                      "<text x=\"64\" y=\"%u\" class=\"meta\">%s</text>"
                      "<rect x=\"180\" y=\"%u\" width=\"%u\" height=\"7\" class=\"current\"/>"
                      "<text x=\"414\" y=\"%u\" class=\"metric\">%llu B</text>",
                      y + 7u, label, y, current_width, y + 7u, current))
        return 0;
    return !has_baseline ||
           bytes_format(svg,
                        "<rect x=\"180\" y=\"%u\" width=\"%u\" height=\"7\" class=\"baseline\"/>"
                        "<text x=\"414\" y=\"%u\" class=\"baseline_metric\">%llu B</text>",
                        y + 9u, baseline_width, y + 16u, baseline);
}
/* Purpose: visualize resource and launch evidence retained by benchmark schema five.
 * Inputs: current and optional baseline. Effects: appends resource and launch evidence.
 * Failure: false on bounded serialization. Boundary: measured counters receive no grade. */
static int chart_resources(benchmark_bytes *svg,
                           const yvex_runtime_benchmark_baseline *current,
                           const yvex_runtime_benchmark_baseline *baseline)
{
    const yvex_runtime_benchmark_metrics *metrics = &current->metrics;
    const yvex_runtime_benchmark_metrics *prior = baseline ? &baseline->metrics : NULL;

    if (!bytes_text(svg,
                    "<line x1=\"64\" y1=\"448\" x2=\"900\" y2=\"448\" class=\"axis\"/>"
                    "<text x=\"64\" y=\"477\" class=\"section\">STRUCTURAL EVIDENCE</text>"))
        return 0;
    if (!chart_resource_row(svg, "resident", 492u, metrics->resident_bytes,
                            prior ? prior->resident_bytes : 0ull, prior != NULL) ||
        !chart_resource_row(svg, "host workspace", 518u, metrics->host_workspace_bytes,
                            prior ? prior->host_workspace_bytes : 0ull, prior != NULL) ||
        !chart_resource_row(svg, "encoded weight pack", 544u,
                            metrics->resident_encoded_bytes,
                            prior ? prior->resident_encoded_bytes : 0ull,
                            prior != NULL) ||
        !chart_resource_row(svg, "resident H2D", 570u, metrics->resident_h2d_bytes,
                            prior ? prior->resident_h2d_bytes : 0ull, prior != NULL) ||
        !chart_resource_row(svg, "attention state", 596u, metrics->state_bytes,
                            prior ? prior->state_bytes : 0ull, prior != NULL))
        return 0;
    return bytes_format(
        svg,
        "<text x=\"600\" y=\"505\" class=\"meta\">last dispatch kernels %llu - graph launches %llu</text>"
        "<text x=\"600\" y=\"528\" class=\"meta\">captures %llu - replays %llu - nodes %llu</text>"
        "<text x=\"600\" y=\"551\" class=\"meta\">warm alloc host/device %llu/%llu</text>"
        "<text x=\"600\" y=\"574\" class=\"meta\">warm reads %llu - uploads %llu B</text>"
        "<text x=\"600\" y=\"600\" class=\"meta\">state %llu B</text>",
        metrics->last_dispatch_kernel_launches, metrics->cuda_graph_launches,
        metrics->cuda_graph_captures, metrics->cuda_graph_replays,
        metrics->cuda_graph_nodes, metrics->warm_host_allocations,
        metrics->warm_device_allocations, metrics->warm_weight_reads,
        metrics->warm_upload_bytes, metrics->state_bytes);
}
/* Purpose: assemble deterministic SVG bytes bound to benchmark and optional baseline identities.
 * Inputs: sealed records and SVG buffer. Effects: appends one dependency-free SVG.
 * Failure: false on escaping/format/bounds. Boundary: hashes only the complete document. */
static int chart_serialize(const yvex_runtime_benchmark_baseline *current,
                           const yvex_runtime_benchmark_baseline *baseline,
                           benchmark_bytes *svg)
{
    svg->maximum = BENCHMARK_CHART_MAX;
    svg->initial_capacity = 1024u;
    if (!bytes_format(svg,
                      "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"760\" viewBox=\"0 0 960 760\""
                      " data-chart-schema=\"5\">"
                      "<style>text{font-family:ui-monospace,SFMono-Regular,monospace;fill:#12212b}"
                      ".title{font-size:24px;font-weight:700}"
                      ".section{font-size:14px;font-weight:700;letter-spacing:1px}"
                      ".meta{font-size:11px;fill:#52636d}.label{font-size:11px;text-anchor:middle}"
                      ".value{font-size:10px;text-anchor:middle}.metric{font-size:10px;fill:#007f73}"
                      ".baseline_metric{font-size:10px;fill:#a86b12}.axis{stroke:#b8c4ca;stroke-width:1}"
                      ".current{fill:#007f73}.baseline{fill:#e9a23b}"
                      ".device{fill:#2e5aac}.device_baseline{fill:#7b65aa}</style>"
                      "<rect width=\"960\" height=\"760\" fill=\"#f4f1e8\"/>"
                      "<text x=\"64\" y=\"62\" class=\"title\">YVEX ATTENTION RUNTIME</text>"
                      "<text x=\"64\" y=\"89\" class=\"meta\">") ||
        !svg_text(svg, current->key.device) ||
        !bytes_text(svg, " - ") || !svg_text(svg, current->key.phase) ||
        !bytes_text(svg, " - ") || !svg_text(svg, current->key.mode) ||
        !bytes_text(svg, " - ") || !svg_text(svg, current->key.scope) ||
        !bytes_format(svg,
                      "</text><text x=\"64\" y=\"142\" class=\"section\">COLD PREPARATION</text>"
                      "<text x=\"64\" y=\"180\" class=\"meta\">total %llu ns</text>"
                      "<text x=\"64\" y=\"205\" class=\"meta\">artifact %llu ns</text>"
                      "<text x=\"64\" y=\"230\" class=\"meta\">residency %llu ns</text>"
                      "<text x=\"64\" y=\"255\" class=\"meta\">graph %llu ns</text>",
                      current->metrics.cold_total_ns, current->metrics.cold_artifact_ns,
                      current->metrics.cold_residency_ns,
                      current->metrics.cold_graph_warmup_ns +
                          current->metrics.cold_graph_capture_ns +
                          current->metrics.cold_graph_instantiate_ns) ||
        !chart_warm(svg, current, baseline) || !chart_resources(svg, current, baseline) ||
        !bytes_format(svg,
                      "<line x1=\"64\" y1=\"626\" x2=\"900\" y2=\"626\" class=\"axis\"/>"
                      "<text x=\"64\" y=\"650\" class=\"meta\">current %s</text>"
                      "<text x=\"64\" y=\"670\" class=\"meta\">build %s - source %s</text>"
                      "<text x=\"64\" y=\"700\" class=\"meta\">baseline %s</text>"
                      "<text x=\"64\" y=\"720\" class=\"meta\">build %s - source %s</text>"
                      "</svg>\n",
                      current->identity, current->key.commit, current->key.build_source_state,
                      baseline ? baseline->identity : "none",
                      baseline ? baseline->key.commit : "none",
                      baseline ? baseline->key.build_source_state : "none"))
        return 0;
    return 1;
}
/* Purpose: seal one mutable benchmark record under its canonical content identity.
 * Inputs: complete record. Effects: validates fields and writes only its identity.
 * Failure: leaves identity empty and names the field. Boundary: sealing does not admit capability. */
int yvex_runtime_benchmark_baseline_seal(yvex_runtime_benchmark_baseline *record,
                                         yvex_runtime_benchmark_failure *failure,
                                         yvex_error *err)
{
    int rc;
    if (!record)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "record",
            YVEX_ERR_INVALID_ARG, "benchmark record is required", err);
    record->identity[0] = '\0';
    if (record->schema_version != YVEX_RUNTIME_BENCHMARK_SCHEMA_V5)
        return benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA, "schema",
                                YVEX_RUNTIME_BENCHMARK_SCHEMA_V5, record->schema_version,
                                YVEX_ERR_UNSUPPORTED, "benchmark schema is unsupported", err);
    if ((rc = key_validate(record, failure, err)) != YVEX_OK ||
        (rc = metrics_validate(record, failure, err)) != YVEX_OK)
        return rc;
    if (!baseline_identity(record, record->identity))
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_IDENTITY, "identity",
            YVEX_ERR_STATE, "benchmark identity construction failed", err);
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: construct one sealed benchmark record from completed runtime attention evidence.
 * Inputs: production result with samples/identities. Effects: fills and seals caller-owned evidence.
 * Failure: invalid facts refuse. Boundary: runtime projects evidence; CLI owns paths and rendering. */
int yvex_runtime_benchmark_baseline_from_attention(
    const yvex_graph_attention_operator_result *result,
    yvex_runtime_benchmark_baseline *record,
    yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    int rc;

    if (!record || !result || !result->completed || !result->benchmark_sample_count ||
        !yvex_sha256_hex_valid(result->execution_descriptor_identity))
        return benchmark_fail(
            failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "attention-result",
            YVEX_ERR_INVALID_ARG, "completed identity-bound benchmark samples are required", err);
    memset(record, 0, sizeof(*record));
    record->schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V5;
    rc = benchmark_attention_key(result, &record->key, err);
    if (rc == YVEX_OK) rc = benchmark_attention_times(result, &record->metrics, err);
    if (rc == YVEX_OK) rc = benchmark_attention_metrics(result, &record->metrics, err);
    return rc == YVEX_OK ? yvex_runtime_benchmark_baseline_seal(record, failure, err) : rc;
}
/* Purpose: transactionally publish one sealed benchmark baseline without replacement.
 * Inputs: safe path and sealed record. Effects: atomically publishes one canonical file.
 * Failure: preserves destinations and removes owned temp. Boundary: stores evidence without grading it. */
int yvex_runtime_benchmark_baseline_write(
    const char *path, const yvex_runtime_benchmark_baseline *record,
    yvex_runtime_benchmark_publication *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    benchmark_bytes bytes = {0};
    int rc = baseline_validate(record, failure, err);
    if (rc != YVEX_OK) return rc;
    if (!baseline_serialize(record, &bytes)) {
        free(bytes.data);
        return benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS,
                                "serialization", BENCHMARK_FILE_MAX, bytes.count, YVEX_ERR_BOUNDS,
                                "benchmark serialization exceeded its bound", err);
    }
    rc = publish_bytes(path, bytes.data, bytes.count, NULL, result, failure, err);
    if (rc == YVEX_OK)
        yvex_core_text_copy(result->identity, sizeof(result->identity), record->identity);
    free(bytes.data);
    return rc;
}
/* Purpose: independently reopen and authenticate one canonical benchmark baseline.
 * Inputs: safe path and record output. Effects: publishes only after stable canonical authentication.
 * Failure: typed format/drift/identity refusal. Boundary: validates evidence without executing it. */
int yvex_runtime_benchmark_baseline_open(const char *path,
                                         yvex_runtime_benchmark_baseline *record,
                                         yvex_runtime_benchmark_failure *failure,
                                         yvex_error *err)
{
    yvex_runtime_benchmark_baseline parsed;
    benchmark_bytes canonical = {0};
    char *data = NULL, *parse_copy = NULL;
    size_t count = 0u;
    int rc;

    if (record) memset(record, 0, sizeof(*record));
    if (!record)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "record",
            YVEX_ERR_INVALID_ARG, "benchmark reopen result is required", err);
    rc = read_file(path, &data, &count, failure, err);
    if (rc != YVEX_OK) return rc;
    parse_copy = (char *)malloc(count + 1u);
    if (!parse_copy) {
        rc = benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_READ, "allocation",
                              count, 0ull, YVEX_ERR_NOMEM,
                              "benchmark parse buffer allocation failed", err);
        goto done;
    }
    memcpy(parse_copy, data, count + 1u);
    if (!baseline_parse(parse_copy, count, &parsed)) {
        if (parsed.schema_version == YVEX_RUNTIME_BENCHMARK_SCHEMA_V1 ||
            parsed.schema_version == YVEX_RUNTIME_BENCHMARK_SCHEMA_V2 ||
            parsed.schema_version == YVEX_RUNTIME_BENCHMARK_SCHEMA_V3 ||
            parsed.schema_version == YVEX_RUNTIME_BENCHMARK_SCHEMA_V4)
            rc = benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA, "schema",
                                  YVEX_RUNTIME_BENCHMARK_SCHEMA_V5,
                                  parsed.schema_version, YVEX_ERR_UNSUPPORTED,
                                  "legacy benchmark schema requires provenance-bound regeneration", err);
        else
            rc = benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_FORMAT, "format", YVEX_ERR_FORMAT,
                "benchmark file is malformed", err);
        goto done;
    }
    if ((rc = baseline_validate(&parsed, failure, err)) != YVEX_OK) goto done;
    if (!baseline_serialize(&parsed, &canonical) || canonical.count != count ||
        memcmp(canonical.data, data, count) != 0) {
        rc = benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_FORMAT,
                              "canonical-bytes", count, canonical.count, YVEX_ERR_FORMAT,
                              "benchmark file bytes are not canonical", err);
        goto done;
    }
    *record = parsed;
    rc = YVEX_OK;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);

done:
    free(canonical.data);
    free(parse_copy);
    free(data);
    return rc;
}
/* Purpose: compare two identity-compatible measurements under one explicit optional policy.
 * Inputs: sealed records and optional threshold. Effects: fills deltas and identity-bound performance status.
 * Failure: incompatibility/policy errors refuse. Boundary: cannot change correctness or qualification. */
int yvex_runtime_benchmark_compare(
    const yvex_runtime_benchmark_baseline *current,
    const yvex_runtime_benchmark_baseline *baseline,
    const yvex_runtime_benchmark_regression_policy *policy,
    yvex_runtime_benchmark_comparison *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    const char *mismatch;
    size_t index;
    int rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!result || (rc = baseline_validate(current, failure, err)) != YVEX_OK ||
        (rc = baseline_validate(baseline, failure, err)) != YVEX_OK)
        return result ? rc : benchmark_fail(
            failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "result",
            YVEX_ERR_INVALID_ARG, "benchmark comparison result is required", err);
    mismatch = keys_mismatch(current, baseline);
    if (mismatch)
        return benchmark_fail(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INCOMPATIBLE, mismatch,
            YVEX_ERR_STATE, "benchmark baseline identity field is incompatible", err);
    result->compatible = 1;
    yvex_core_text_copy(result->current_commit, sizeof(result->current_commit), current->key.commit);
    yvex_core_text_copy(result->baseline_commit, sizeof(result->baseline_commit), baseline->key.commit);
    yvex_core_text_copy(result->current_source_state,
                        sizeof(result->current_source_state),
                        current->key.build_source_state);
    yvex_core_text_copy(result->baseline_source_state,
                        sizeof(result->baseline_source_state),
                        baseline->key.build_source_state);
    yvex_core_text_copy(result->current_identity, sizeof(result->current_identity), current->identity);
    yvex_core_text_copy(result->baseline_identity, sizeof(result->baseline_identity), baseline->identity);
    result->cold_total_delta_ns =
        metric_delta(current->metrics.cold_total_ns, baseline->metrics.cold_total_ns);
    result->device_timing_available = current->metrics.device_timing.available;
    for (index = 0u; index < YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT; ++index) {
        result->host_delta_ns[index] = metric_delta(
            current->metrics.host_timing.values[index],
            baseline->metrics.host_timing.values[index]);
        result->device_delta_ns[index] = metric_delta(
            current->metrics.device_timing.values[index],
            baseline->metrics.device_timing.values[index]);
    }
    rc = regression_policy_apply(current, baseline, policy, result, failure, err);
    if (rc != YVEX_OK) return rc;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: publish one deterministic dependency-free SVG benchmark chart.
 * Inputs: sealed records and safe SVG path. Effects: creates one identity-bound chart without replacement.
 * Failure: preserves prior files. Boundary: baseline publication remains separately owned. */
int yvex_runtime_benchmark_chart_write(
    const yvex_runtime_benchmark_chart_request *request,
    yvex_runtime_benchmark_chart_result *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err)
{
    yvex_runtime_benchmark_publication publication;
    yvex_runtime_benchmark_comparison comparison;
    benchmark_bytes svg = {0};
    char identity[YVEX_SHA256_HEX_BYTES];
    int rc = YVEX_OK;

    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !request->path ||
        (rc = baseline_validate(request->current, failure, err)) != YVEX_OK)
        return request && result ? rc : benchmark_fail(
            failure, YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT, "chart",
            YVEX_ERR_INVALID_ARG, "benchmark chart request is incomplete", err);
    if (request->baseline &&
        (rc = yvex_runtime_benchmark_compare(request->current, request->baseline,
                                             NULL, &comparison, failure, err)) != YVEX_OK)
        return rc;
    if (!chart_serialize(request->current, request->baseline, &svg) ||
        !chart_bytes_identity(&svg, identity)) {
        free(svg.data);
        return benchmark_reject(failure, YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS,
                                "chart-bytes", BENCHMARK_CHART_MAX, svg.count, YVEX_ERR_BOUNDS,
                                "benchmark SVG exceeded its bound", err);
    }
    rc = publish_bytes(request->path, svg.data, svg.count, request->file_faults,
                       &publication, failure, err);
    if (rc == YVEX_OK) {
        result->generated = 1;
        result->file_bytes = publication.file_bytes;
        yvex_core_text_copy(result->path, sizeof(result->path), publication.path);
        yvex_core_text_copy(result->identity, sizeof(result->identity), identity);
    }
    free(svg.data);
    return rc;
}
