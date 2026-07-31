/*
 * Expose one independently reopenable baseline and chart lifecycle to runtime and operator
 * consumers.
 *
 * Records are versioned, content-addressed, bounded, and atomically published without replacement.
 * External benchmark evidence is serialized only after production execution has produced typed
 * facts.
 */
#ifndef INCLUDE_YVEX_INTERNAL_BENCHMARK_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_BENCHMARK_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_RUNTIME_BENCHMARK_SCHEMA_V1 1u
#define YVEX_RUNTIME_BENCHMARK_SCHEMA_V2 2u
#define YVEX_RUNTIME_BENCHMARK_SCHEMA_V3 3u
#define YVEX_RUNTIME_BENCHMARK_SCHEMA_V4 4u
#define YVEX_RUNTIME_BENCHMARK_SCHEMA_V5 5u
#define YVEX_RUNTIME_BENCHMARK_COMMIT_CAP 41u
#define YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP 8u
#define YVEX_RUNTIME_BENCHMARK_TEXT_CAP 128u
#define YVEX_RUNTIME_BENCHMARK_MODE_CAP 16u
#define YVEX_RUNTIME_BENCHMARK_SCOPE_CAP 32u
#define YVEX_RUNTIME_BENCHMARK_BUCKET_CAP 64u

typedef enum {
    YVEX_RUNTIME_BENCHMARK_FAILURE_NONE = 0,
    YVEX_RUNTIME_BENCHMARK_FAILURE_INVALID_ARGUMENT,
    YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
    YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
    YVEX_RUNTIME_BENCHMARK_FAILURE_IDENTITY,
    YVEX_RUNTIME_BENCHMARK_FAILURE_PATH,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CREATE,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CONFLICT,
    YVEX_RUNTIME_BENCHMARK_FAILURE_WRITE,
    YVEX_RUNTIME_BENCHMARK_FAILURE_SYNC,
    YVEX_RUNTIME_BENCHMARK_FAILURE_PUBLISH,
    YVEX_RUNTIME_BENCHMARK_FAILURE_OPEN,
    YVEX_RUNTIME_BENCHMARK_FAILURE_READ,
    YVEX_RUNTIME_BENCHMARK_FAILURE_FORMAT,
    YVEX_RUNTIME_BENCHMARK_FAILURE_BOUNDS,
    YVEX_RUNTIME_BENCHMARK_FAILURE_INCOMPATIBLE,
    YVEX_RUNTIME_BENCHMARK_FAILURE_CLEANUP
} yvex_runtime_benchmark_failure_code;

typedef struct {
    yvex_runtime_benchmark_failure_code code;
    char field[64];
    unsigned long long expected, actual;
    const char *reason;
} yvex_runtime_benchmark_failure;

typedef struct {
    char commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP];
    char build_source_state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP];
    char source_delta_identity[YVEX_SHA256_HEX_BYTES], build_identity[YVEX_SHA256_HEX_BYTES];
    char benchmark_scope[YVEX_RUNTIME_BENCHMARK_SCOPE_CAP];
    char artifact_identity[YVEX_SHA256_HEX_BYTES], materialization_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_binding_identity[YVEX_SHA256_HEX_BYTES], logical_model_identity[YVEX_SHA256_HEX_BYTES];
    char runtime_numeric_identity[YVEX_SHA256_HEX_BYTES], runtime_descriptor_identity[YVEX_SHA256_HEX_BYTES];
    char semantic_graph_identity[YVEX_SHA256_HEX_BYTES], executable_graph_identity[YVEX_SHA256_HEX_BYTES];
    char execution_descriptor_identity[YVEX_SHA256_HEX_BYTES], residency_identity[YVEX_SHA256_HEX_BYTES];
    char workspace_identity[YVEX_SHA256_HEX_BYTES], state_layout_identity[YVEX_SHA256_HEX_BYTES];
    char kernel_bundle_identity[YVEX_SHA256_HEX_BYTES];
    char machine_identity[YVEX_RUNTIME_BENCHMARK_TEXT_CAP], cpu_model[YVEX_RUNTIME_BENCHMARK_TEXT_CAP];
    char gpu_model[YVEX_RUNTIME_BENCHMARK_TEXT_CAP], device[YVEX_RUNTIME_BENCHMARK_TEXT_CAP];
    char driver[YVEX_RUNTIME_BENCHMARK_TEXT_CAP], cuda_build[YVEX_RUNTIME_BENCHMARK_TEXT_CAP];
    char mode[YVEX_RUNTIME_BENCHMARK_MODE_CAP], phase[YVEX_RUNTIME_BENCHMARK_MODE_CAP];
    char scope[YVEX_RUNTIME_BENCHMARK_SCOPE_CAP];
    char coverage[YVEX_RUNTIME_BENCHMARK_MODE_CAP], attention_class[YVEX_RUNTIME_BENCHMARK_MODE_CAP];
    char trace_policy[YVEX_RUNTIME_BENCHMARK_MODE_CAP];
    char capture_bucket[YVEX_RUNTIME_BENCHMARK_BUCKET_CAP];
    unsigned long long memory_bytes, compute_capability_major, compute_capability_minor;
    unsigned long long layer_start, layer_count, token_count, history_tokens;
    unsigned long long warmup_count, iteration_count;
} yvex_runtime_benchmark_key;

typedef enum {
    YVEX_RUNTIME_BENCHMARK_MINIMUM = 0,
    YVEX_RUNTIME_BENCHMARK_P50,
    YVEX_RUNTIME_BENCHMARK_P90,
    YVEX_RUNTIME_BENCHMARK_P95,
    YVEX_RUNTIME_BENCHMARK_P99,
    YVEX_RUNTIME_BENCHMARK_MAXIMUM,
    YVEX_RUNTIME_BENCHMARK_MEAN,
    YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION,
    YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT
} yvex_runtime_benchmark_statistic;

typedef struct {
    int available;
    unsigned long long values[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
} yvex_runtime_benchmark_timing_distribution;

typedef struct {
    unsigned long long cold_total_ns, cold_artifact_ns, cold_binding_ns;
    unsigned long long cold_model_ns, cold_residency_ns, cold_workspace_ns;
    unsigned long long cold_graph_warmup_ns, cold_graph_capture_ns, cold_graph_instantiate_ns;
    unsigned long long first_execution_ns, publication_ns, cleanup_ns;
    yvex_runtime_benchmark_timing_distribution host_timing;
    yvex_runtime_benchmark_timing_distribution device_timing;
    unsigned long long artifact_bytes_hashed, artifact_bytes_read, weight_bytes_read;
    unsigned long long resident_encoded_bytes, resident_h2d_bytes;
    unsigned long long h2d_bytes, d2h_bytes;
    unsigned long long warm_weight_reads, warm_upload_bytes;
    unsigned long long warm_host_allocations, warm_device_allocations, warm_device_frees;
    unsigned long long last_dispatch_kernel_launches, cuda_graph_launches, cuda_graph_captures;
    unsigned long long cuda_graph_replays, cuda_graph_nodes;
    unsigned long long peak_host_bytes, last_dispatch_peak_device_bytes, resident_bytes;
    unsigned long long host_workspace_bytes, state_bytes;
} yvex_runtime_benchmark_metrics;

typedef struct {
    unsigned int schema_version;
    yvex_runtime_benchmark_key key;
    yvex_runtime_benchmark_metrics metrics;
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_benchmark_baseline;

typedef struct {
    int published;
    unsigned long long file_bytes;
    char path[YVEX_PATH_CAP];
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_benchmark_publication;

typedef struct {
    int enabled;
    unsigned long long basis_points;
} yvex_runtime_benchmark_regression_policy;

typedef struct {
    int compatible, performance_passed;
    char current_commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP], baseline_commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP];
    char current_source_state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP];
    char baseline_source_state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP];
    char current_identity[YVEX_SHA256_HEX_BYTES], baseline_identity[YVEX_SHA256_HEX_BYTES];
    char regression_policy_identity[YVEX_SHA256_HEX_BYTES], comparison_identity[YVEX_SHA256_HEX_BYTES];
    yvex_runtime_benchmark_regression_policy regression_policy;
    long long cold_total_delta_ns;
    long long host_delta_ns[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    long long device_delta_ns[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    int device_timing_available;
} yvex_runtime_benchmark_comparison;

typedef struct {
    const char *path;
    const yvex_runtime_benchmark_baseline *current;
    const yvex_runtime_benchmark_baseline *baseline;
    const yvex_core_file_faults *file_faults;
} yvex_runtime_benchmark_chart_request;

typedef struct {
    int generated;
    unsigned long long file_bytes;
    char path[YVEX_PATH_CAP];
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_runtime_benchmark_chart_result;

typedef struct {
    char identity[YVEX_SHA256_HEX_BYTES], baseline_identity[YVEX_SHA256_HEX_BYTES];
    char current_commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP], baseline_commit[YVEX_RUNTIME_BENCHMARK_COMMIT_CAP];
    char current_source_state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP];
    char baseline_source_state[YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP];
    char chart_identity[YVEX_SHA256_HEX_BYTES], regression_policy_identity[YVEX_SHA256_HEX_BYTES];
    char comparison_identity[YVEX_SHA256_HEX_BYTES];
    char path[YVEX_PATH_CAP], chart_path[YVEX_PATH_CAP];
    unsigned long long file_bytes, chart_file_bytes;
    unsigned long long regression_basis_points;
    double cold_delta_seconds;
    double host_delta_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    double device_delta_seconds[YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT];
    int baseline_written, baseline_compatible, chart_generated;
    int device_timing_available, regression_policy_enabled, performance_passed;
} yvex_runtime_benchmark_operator_summary;

struct yvex_graph_attention_operator_result;
int yvex_runtime_benchmark_samples_finish(
    double *host_seconds, double *device_seconds, unsigned long long count,
    int device_requested, struct yvex_graph_attention_operator_result *result,
    yvex_error *err);
int yvex_runtime_benchmark_baseline_from_attention(
    const struct yvex_graph_attention_operator_result *result,
    yvex_runtime_benchmark_baseline *record,
    yvex_runtime_benchmark_failure *failure, yvex_error *err);
int yvex_runtime_benchmark_baseline_seal(yvex_runtime_benchmark_baseline *record,
                                         yvex_runtime_benchmark_failure *failure,
                                         yvex_error *err);
int yvex_runtime_benchmark_baseline_write(
    const char *path, const yvex_runtime_benchmark_baseline *record,
    yvex_runtime_benchmark_publication *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err);
int yvex_runtime_benchmark_baseline_open(const char *path,
                                         yvex_runtime_benchmark_baseline *record,
                                         yvex_runtime_benchmark_failure *failure,
                                         yvex_error *err);
int yvex_runtime_benchmark_compare(
    const yvex_runtime_benchmark_baseline *current,
    const yvex_runtime_benchmark_baseline *baseline,
    const yvex_runtime_benchmark_regression_policy *policy,
    yvex_runtime_benchmark_comparison *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err);
int yvex_runtime_benchmark_chart_write(
    const yvex_runtime_benchmark_chart_request *request,
    yvex_runtime_benchmark_chart_result *result,
    yvex_runtime_benchmark_failure *failure, yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_BENCHMARK_H_INCLUDED */
