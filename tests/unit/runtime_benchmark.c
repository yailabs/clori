/*
 * Verify canonical benchmark evidence survives independent storage and rejects unsafe or stale
 * input. Every external fixture lives beneath one unique owned temporary directory and is
 * removed exactly. Bounded fixtures prove the runtime benchmark file lifecycle without promoting
 * measured capability.
 */
#define _GNU_SOURCE
#include <yvex/internal/benchmark.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/runtime_operator.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tests/test.h"

typedef struct {
    char root[YVEX_PATH_CAP];
    char baseline[YVEX_PATH_CAP];
    char dirty_baseline[YVEX_PATH_CAP];
    char chart[YVEX_PATH_CAP];
    char deterministic_chart[YVEX_PATH_CAP];
    char cpu_chart[YVEX_PATH_CAP];
    char compared_chart[YVEX_PATH_CAP];
    char cleanup_chart[YVEX_PATH_CAP];
    char transaction_baseline[YVEX_PATH_CAP];
    char transaction_chart[YVEX_PATH_CAP];
    char qualification[YVEX_PATH_CAP];
    char corrupt[YVEX_PATH_CAP];
    char symlink_parent[YVEX_PATH_CAP];
} benchmark_fixture;

static void fixture_text(char *output, size_t capacity, const char *text)
{
    (void)snprintf(output, capacity, "%s", text);
}

/* Fill one independently valid benchmark record with deterministic real-shape metadata. */
static void fixture_record(yvex_runtime_benchmark_baseline *record)
{
    memset(record, 0, sizeof(*record));
    record->schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V5;
    fixture_text(record->key.commit, sizeof(record->key.commit),
                 "0123456789abcdef0123456789abcdef01234567");
    fixture_text(record->key.build_source_state, sizeof(record->key.build_source_state),
                 "clean");
    fixture_text(record->key.source_delta_identity,
                 sizeof(record->key.source_delta_identity),
                 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    fixture_text(record->key.build_identity, sizeof(record->key.build_identity),
                 "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    fixture_text(record->key.benchmark_scope, sizeof(record->key.benchmark_scope),
                 "attention_component");
    fixture_text(record->key.artifact_identity, sizeof(record->key.artifact_identity),
                 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    fixture_text(record->key.materialization_identity,
                 sizeof(record->key.materialization_identity),
                 "1111111111111111111111111111111111111111111111111111111111111111");
    fixture_text(record->key.runtime_binding_identity,
                 sizeof(record->key.runtime_binding_identity),
                 "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    fixture_text(record->key.logical_model_identity,
                 sizeof(record->key.logical_model_identity),
                 "2222222222222222222222222222222222222222222222222222222222222222");
    fixture_text(record->key.runtime_numeric_identity,
                 sizeof(record->key.runtime_numeric_identity),
                 "3333333333333333333333333333333333333333333333333333333333333333");
    fixture_text(record->key.runtime_descriptor_identity,
                 sizeof(record->key.runtime_descriptor_identity),
                 "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    fixture_text(record->key.semantic_graph_identity,
                 sizeof(record->key.semantic_graph_identity),
                 "4444444444444444444444444444444444444444444444444444444444444444");
    fixture_text(record->key.executable_graph_identity,
                 sizeof(record->key.executable_graph_identity),
                 "5555555555555555555555555555555555555555555555555555555555555555");
    fixture_text(record->key.execution_descriptor_identity,
                 sizeof(record->key.execution_descriptor_identity),
                 "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
    fixture_text(record->key.residency_identity, sizeof(record->key.residency_identity),
                 "6666666666666666666666666666666666666666666666666666666666666666");
    fixture_text(record->key.workspace_identity, sizeof(record->key.workspace_identity),
                 "7777777777777777777777777777777777777777777777777777777777777777");
    fixture_text(record->key.state_layout_identity,
                 sizeof(record->key.state_layout_identity),
                 "8888888888888888888888888888888888888888888888888888888888888888");
    fixture_text(record->key.kernel_bundle_identity,
                 sizeof(record->key.kernel_bundle_identity),
                 "9999999999999999999999999999999999999999999999999999999999999999");
    fixture_text(record->key.machine_identity, sizeof(record->key.machine_identity),
                 "fixture-machine");
    fixture_text(record->key.cpu_model, sizeof(record->key.cpu_model), "fixture-cpu");
    fixture_text(record->key.gpu_model, sizeof(record->key.gpu_model), "NVIDIA GB10");
    fixture_text(record->key.device, sizeof(record->key.device), "NVIDIA GB10 <& verified>");
    fixture_text(record->key.driver, sizeof(record->key.driver), "CUDA Driver 13.0");
    fixture_text(record->key.cuda_build, sizeof(record->key.cuda_build), "nvcc 13.0 sm_121");
    fixture_text(record->key.mode, sizeof(record->key.mode), "full");
    fixture_text(record->key.phase, sizeof(record->key.phase), "decode");
    fixture_text(record->key.scope, sizeof(record->key.scope), "release-attention-set");
    fixture_text(record->key.coverage, sizeof(record->key.coverage), "full");
    fixture_text(record->key.attention_class, sizeof(record->key.attention_class), "mixed");
    fixture_text(record->key.trace_policy, sizeof(record->key.trace_policy), "none");
    fixture_text(record->key.capture_bucket, sizeof(record->key.capture_bucket), "decode-1");
    record->key.memory_bytes = 137438953472ull;
    record->key.compute_capability_major = 12ull;
    record->key.compute_capability_minor = 1ull;
    record->key.layer_count = 43ull;
    record->key.token_count = 1ull;
    record->key.warmup_count = 3ull;
    record->key.iteration_count = 20ull;
    record->metrics.cold_total_ns = 95000000000ull;
    record->metrics.cold_artifact_ns = 60000000000ull;
    record->metrics.cold_binding_ns = 1000000000ull;
    record->metrics.cold_model_ns = 2000000000ull;
    record->metrics.cold_residency_ns = 25000000000ull;
    record->metrics.cold_workspace_ns = 2000000000ull;
    record->metrics.cold_graph_warmup_ns = 1000000000ull;
    record->metrics.cold_graph_capture_ns = 3000000000ull;
    record->metrics.cold_graph_instantiate_ns = 1000000000ull;
    record->metrics.first_execution_ns = 8000000ull;
    record->metrics.publication_ns = 100000ull;
    record->metrics.cleanup_ns = 50000ull;
    record->metrics.host_timing.available = 1;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MINIMUM] = 4000000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P50] = 5000000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P90] = 6000000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P95] = 6500000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P99] = 7000000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] = 8000000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN] = 5500000ull;
    record->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION] = 750000ull;
    record->metrics.device_timing.available = 1;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MINIMUM] = 2000000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P50] = 3000000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P90] = 4000000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P95] = 4500000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P99] = 5000000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] = 6000000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN] = 3500000ull;
    record->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_STANDARD_DEVIATION] = 500000ull;
    record->metrics.artifact_bytes_hashed = 102408545440ull;
    record->metrics.artifact_bytes_read = 102408545440ull;
    record->metrics.weight_bytes_read = 5698713600ull;
    record->metrics.resident_encoded_bytes = 5698713600ull;
    record->metrics.resident_h2d_bytes = 5698713600ull;
    record->metrics.h2d_bytes = 5698713600ull;
    record->metrics.d2h_bytes = 32768ull;
    record->metrics.last_dispatch_kernel_launches = 1259ull;
    record->metrics.cuda_graph_launches = 20ull;
    record->metrics.cuda_graph_captures = 1ull;
    record->metrics.cuda_graph_replays = 20ull;
    record->metrics.cuda_graph_nodes = 1259ull;
    record->metrics.peak_host_bytes = 1048576ull;
    record->metrics.last_dispatch_peak_device_bytes = 6442450944ull;
    record->metrics.resident_bytes = 5698713600ull;
    record->metrics.host_workspace_bytes = 1048576ull;
    record->metrics.state_bytes = 524288ull;
}

static int fixture_open(benchmark_fixture *fixture)
{
    char root[] = "/tmp/yvex-runtime-benchmark.XXXXXX";
    memset(fixture, 0, sizeof(*fixture));
    if (!mkdtemp(root)) return 0;
    fixture_text(fixture->root, sizeof(fixture->root), root);
#define PATH(NAME, SUFFIX)                                                                   \
    do {                                                                                     \
        if (snprintf(fixture->NAME, sizeof(fixture->NAME), "%s/%s", root, SUFFIX) >=        \
            (int)sizeof(fixture->NAME))                                                      \
            return 0;                                                                        \
    } while (0)
    PATH(baseline, "baseline.yvex-benchmark");
    PATH(dirty_baseline, "dirty.yvex-benchmark");
    PATH(chart, "benchmark.svg");
    PATH(deterministic_chart, "benchmark-repeat.svg");
    PATH(cpu_chart, "benchmark-cpu.svg");
    PATH(compared_chart, "comparison.svg");
    PATH(cleanup_chart, "cleanup.svg");
    PATH(transaction_baseline, "transaction.yvex-benchmark");
    PATH(transaction_chart, "transaction.svg");
    PATH(qualification, "execution.yvex-qualification");
    PATH(corrupt, "corrupt.yvex-benchmark");
    PATH(symlink_parent, "linked");
#undef PATH
    return 1;
}

static int fixture_close(const benchmark_fixture *fixture)
{
    const char *const files[] = {
        fixture->baseline, fixture->dirty_baseline, fixture->chart, fixture->deterministic_chart,
        fixture->cpu_chart,
        fixture->compared_chart, fixture->cleanup_chart,
        fixture->transaction_baseline, fixture->transaction_chart,
        fixture->qualification, fixture->corrupt, fixture->symlink_parent,
    };
    size_t index;
    int ok = 1;
    for (index = 0u; index < sizeof(files) / sizeof(files[0]); ++index) {
        if (unlink(files[index]) != 0 && errno != ENOENT) ok = 0;
    }
    if (rmdir(fixture->root) != 0) ok = 0;
    return ok;
}

static char *fixture_read(const char *path, size_t *count)
{
    struct stat state;
    char *data;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    size_t offset = 0u;
    if (fd < 0 || fstat(fd, &state) != 0 || state.st_size <= 0 || state.st_size > 65536) {
        if (fd >= 0) (void)close(fd);
        return NULL;
    }
    data = (char *)malloc((size_t)state.st_size + 1u);
    if (!data) {
        (void)close(fd);
        return NULL;
    }
    while (offset < (size_t)state.st_size) {
        ssize_t got = read(fd, data + offset, (size_t)state.st_size - offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            free(data);
            (void)close(fd);
            return NULL;
        }
        offset += (size_t)got;
    }
    (void)close(fd);
    data[offset] = '\0';
    *count = offset;
    return data;
}

static int fixture_digest(const char *data, size_t count,
                          char output[YVEX_SHA256_HEX_BYTES])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, data, count) || !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int test_sample_distributions(void)
{
    double host[] = {0.004, 0.001, 0.003, 0.002};
    double device[] = {0.0004, 0.0001, 0.0003, 0.0002};
    double cpu[] = {0.002, 0.001};
    double invalid[] = {0.0001, 0.0, 0.0003, 0.0002};
    yvex_graph_attention_operator_result result;
    yvex_error err;

    memset(&result, 0, sizeof(result));
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_samples_finish(
                         host, device, 4ull, 1, &result, &err) == YVEX_OK &&
                         result.benchmark_sample_count == 4ull &&
                         result.benchmark_host_seconds[
                             YVEX_RUNTIME_BENCHMARK_MINIMUM] == 0.001 &&
                         result.benchmark_host_seconds[
                             YVEX_RUNTIME_BENCHMARK_P50] == 0.002 &&
                         result.benchmark_host_seconds[
                             YVEX_RUNTIME_BENCHMARK_P95] == 0.004 &&
                         result.benchmark_host_seconds[
                             YVEX_RUNTIME_BENCHMARK_P99] == 0.004 &&
                         result.benchmark_first_execution_seconds == 0.004 &&
                         result.benchmark_device_timing_available == 1 &&
                         result.benchmark_device_seconds[
                             YVEX_RUNTIME_BENCHMARK_P50] == 0.0002 &&
                         result.benchmark_scope[0] &&
                         result.benchmark_correctness_precondition_passed &&
                         result.benchmark_runtime_precondition_passed,
                     "host and device samples produce independent ordered distributions");
    memset(&result, 0, sizeof(result));
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_samples_finish(
                         cpu, NULL, 2ull, 0, &result, &err) == YVEX_OK &&
                         !result.benchmark_device_timing_available &&
                         result.benchmark_device_seconds[
                             YVEX_RUNTIME_BENCHMARK_MAXIMUM] == 0.0,
                     "CPU samples publish device timing as unavailable rather than measured zero");
    memset(&result, 0, sizeof(result));
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_samples_finish(
                         host, invalid, 4ull, 1, &result, &err) == YVEX_ERR_BOUNDS &&
                         result.benchmark_sample_count == 0ull,
                     "requested device timing refuses missing or zero event samples atomically");
    return 0;
}

/* Prove record identities are deterministic and cover every measured and compatibility fact. */
static int test_seal_identity(void)
{
    yvex_runtime_benchmark_baseline first, second;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    char identity[YVEX_SHA256_HEX_BYTES];

    fixture_record(&first);
    second = first;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&first, &failure, &err) == YVEX_OK,
                     "canonical benchmark record seals");
    fixture_text(identity, sizeof(identity), first.identity);
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) == 0,
                     "equivalent benchmark records have one identity");
    second.metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P99]++;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "measured metric mutation changes benchmark identity");
    second = first;
    second.metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P99]++;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "device timing mutation changes benchmark identity");
    second = first;
    second.metrics.device_timing.available = 2;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         strcmp(failure.field, "device-timing") == 0,
                     "device timing availability is a canonical boolean");
    second = first;
    second.metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P90] =
        second.metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P50] - 1ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         strcmp(failure.field, "device-timing") == 0,
                     "unordered device timing distribution refuses");
    second = first;
    fixture_text(second.key.build_source_state,
                 sizeof(second.key.build_source_state), "dirty");
    fixture_text(second.key.source_delta_identity,
                 sizeof(second.key.source_delta_identity),
                 "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "exact dirty source delta changes benchmark identity");
    second = first;
    fixture_text(second.key.build_identity, sizeof(second.key.build_identity),
                 "9999999999999999999999999999999999999999999999999999999999999999");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "compiler and linker build identity changes benchmark identity");
    second = first;
    fixture_text(second.key.build_source_state, sizeof(second.key.build_source_state), "dirty");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                     "source state cannot disagree with its exact delta identity");
    second = first;
    fixture_text(second.key.build_source_state, sizeof(second.key.build_source_state), "unknown");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                     "benchmark source state accepts only canonical clean or dirty values");
    second = first;
    fixture_text(second.key.mode, sizeof(second.key.mode), "auto");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
                         strcmp(failure.field, "mode") == 0,
                     "benchmark record requires the selected canonical execution mode");
    second = first;
    fixture_text(second.key.phase, sizeof(second.key.phase), "mixed");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
                         strcmp(failure.field, "phase") == 0,
                     "unsupported benchmark runtime phase refuses explicitly");
    second = first;
    fixture_text(second.key.scope, sizeof(second.key.scope), "transformer");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_FORMAT &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
                         strcmp(failure.field, "scope") == 0,
                     "unsupported benchmark operation scope refuses explicitly");
    second = first;
    second.metrics.cuda_graph_launches = 0ull;
    second.metrics.cuda_graph_captures = 0ull;
    second.metrics.cuda_graph_replays = 0ull;
    second.metrics.cuda_graph_nodes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
                         strcmp(failure.field, "execution-mode-evidence") == 0,
                     "full benchmark mode requires complete CUDA Graph evidence");
    second = first;
    fixture_text(second.key.mode, sizeof(second.key.mode), "eager");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
                         strcmp(failure.field, "execution-mode-evidence") == 0,
                     "eager benchmark mode refuses CUDA Graph evidence");
    second.metrics.cuda_graph_launches = 0ull;
    second.metrics.cuda_graph_captures = 0ull;
    second.metrics.cuda_graph_replays = 0ull;
    second.metrics.cuda_graph_nodes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "eager benchmark seals only without CUDA Graph evidence");
    second = first;
    fixture_text(second.key.mode, sizeof(second.key.mode), "eager");
    fixture_text(second.key.device, sizeof(second.key.device), "cpu:test:x86_64:1");
    fixture_text(second.key.driver, sizeof(second.key.driver), "kernel:test:1");
    fixture_text(second.key.cuda_build, sizeof(second.key.cuda_build), "not-applicable");
    memset(&second.metrics.device_timing, 0, sizeof(second.metrics.device_timing));
    second.metrics.cuda_graph_launches = 0ull;
    second.metrics.cuda_graph_captures = 0ull;
    second.metrics.cuda_graph_replays = 0ull;
    second.metrics.cuda_graph_nodes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK,
                     "CPU benchmark preserves typed unavailable device timing");
    second = first;
    second.metrics.last_dispatch_kernel_launches++;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "structural metric mutation changes benchmark identity");
    second = first;
    second.metrics.resident_encoded_bytes--;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "resident encoded-byte mutation changes benchmark identity");
    second = first;
    second.metrics.resident_encoded_bytes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                     "missing resident encoded-weight evidence refuses while warm reads may be zero");
    second = first;
    second.metrics.state_bytes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                     "missing attention-state memory evidence refuses benchmark sealing");
    second = first;
    second.metrics.resident_encoded_bytes = second.metrics.resident_bytes + 1ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_BOUNDS &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD,
                     "encoded weight extent cannot exceed total resident ownership");
    second = first;
    fixture_text(second.key.capture_bucket, sizeof(second.key.capture_bucket), "decode-2");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) == YVEX_OK &&
                         strcmp(identity, second.identity) != 0,
                     "compatibility mutation changes benchmark identity");
    second = first;
    second.schema_version++;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&second, &failure, &err) ==
                         YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
                     "unsupported baseline schema refuses explicitly");
    return 0;
}

static int test_publication(const benchmark_fixture *fixture,
                            yvex_runtime_benchmark_baseline *record)
{
    yvex_runtime_benchmark_publication publication, conflict;
    yvex_runtime_benchmark_baseline reopened, dirty;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    char *serialized;
    size_t serialized_count;

    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_write(
                         fixture->baseline, record, &publication, &failure, &err) == YVEX_OK &&
                         publication.published && publication.file_bytes > 0ull &&
                         strcmp(publication.identity, record->identity) == 0,
                     "sealed baseline publishes atomically");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_open(
                         fixture->baseline, &reopened, &failure, &err) == YVEX_OK &&
                         strcmp(reopened.identity, record->identity) == 0 &&
                         memcmp(&reopened.metrics, &record->metrics,
                                sizeof(record->metrics)) == 0,
                     "published baseline independently reopens every schema-five metric");
    serialized = fixture_read(fixture->baseline, &serialized_count);
    YVEX_TEST_ASSERT(serialized && serialized_count > 0u &&
                         strstr(serialized, "build_source_state\tclean\n") &&
                         strstr(serialized,
                                "source_delta_identity\te3b0c44298fc1c149afbf4c8996fb924"
                                "27ae41e4649b934ca495991b7852b855\n") &&
                         strstr(serialized,
                                "build_identity\teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                                "eeeeeeeeeeeeeeeeeeeeeeee\n") &&
                         strstr(serialized,
                                "resident_encoded_bytes\t0000000153ab7800\n") &&
                         strstr(serialized, "device_timing_available\t1\n") &&
                         strstr(serialized, "device_p50_ns\t00000000002dc6c0\n"),
                     "serialized schema binds provenance, residency, and device timing");
    free(serialized);
    dirty = *record;
    fixture_text(dirty.key.build_source_state, sizeof(dirty.key.build_source_state), "dirty");
    fixture_text(dirty.key.source_delta_identity,
                 sizeof(dirty.key.source_delta_identity),
                 "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&dirty, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_baseline_write(
                             fixture->dirty_baseline, &dirty, &publication, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_baseline_open(
                             fixture->dirty_baseline, &reopened, &failure, &err) == YVEX_OK &&
                         strcmp(reopened.key.build_source_state, "dirty") == 0,
                     "dirty build provenance serializes and reopens canonically");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_write(
                         fixture->baseline, record, &conflict, &failure, &err) ==
                         YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_CONFLICT,
                     "baseline publication never replaces an existing file");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_open(
                         fixture->baseline, &reopened, &failure, &err) == YVEX_OK &&
                         strcmp(reopened.identity, record->identity) == 0,
                     "conflict preserves the original baseline");
    return 0;
}

static int test_comparison(const yvex_runtime_benchmark_baseline *baseline,
                           yvex_runtime_benchmark_baseline *current)
{
    yvex_runtime_benchmark_comparison comparison;
    yvex_runtime_benchmark_failure failure;
    yvex_runtime_benchmark_regression_policy policy = {0};
    yvex_error err;

    *current = *baseline;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P50] += 250000ull;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P90] += 250000ull;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P95] += 250000ull;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_P99] += 250000ull;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] += 250000ull;
    current->metrics.host_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN] += 250000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P50] += 125000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P90] += 125000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P95] += 125000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_P99] += 125000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MAXIMUM] += 125000ull;
    current->metrics.device_timing.values[YVEX_RUNTIME_BENCHMARK_MEAN] += 125000ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK,
                     "compatible current measurement seals");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_compare(
                         current, baseline, NULL, &comparison, &failure, &err) == YVEX_OK &&
                         comparison.compatible &&
                         comparison.host_delta_ns[YVEX_RUNTIME_BENCHMARK_P50] == 250000ll &&
                         comparison.host_delta_ns[YVEX_RUNTIME_BENCHMARK_P95] == 250000ll &&
                         comparison.cold_total_delta_ns == 0ll &&
                         comparison.device_timing_available &&
                         comparison.device_delta_ns[YVEX_RUNTIME_BENCHMARK_P50] == 125000ll &&
                         comparison.device_delta_ns[YVEX_RUNTIME_BENCHMARK_P95] == 125000ll &&
                         comparison.performance_passed &&
                         strlen(comparison.regression_policy_identity) == 64u &&
                         strlen(comparison.comparison_identity) == 64u,
                     "comparison reports host and device deltas without a threshold");
    policy.enabled = 1;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_compare(
                         current, baseline, &policy, &comparison, &failure, &err) == YVEX_OK &&
                         !comparison.performance_passed,
                     "zero basis points detects measured latency and throughput regressions");
    policy.basis_points = 600ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_compare(
                         current, baseline, &policy, &comparison, &failure, &err) == YVEX_OK &&
                         comparison.performance_passed,
                     "explicit latency allowance passes without creating a universal default");
    fixture_text(current->key.commit, sizeof(current->key.commit),
                 "fedcba9876543210fedcba9876543210fedcba98");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_compare(
                             current, baseline, NULL, &comparison, &failure, &err) == YVEX_OK &&
                         strcmp(comparison.current_commit, comparison.baseline_commit) != 0 &&
                         strcmp(comparison.current_source_state, "clean") == 0 &&
                         strcmp(comparison.baseline_source_state, "clean") == 0,
                     "cross-commit comparison preserves provenance without changing workload compatibility");
    fixture_text(current->key.build_source_state,
                 sizeof(current->key.build_source_state), "dirty");
    fixture_text(current->key.source_delta_identity,
                 sizeof(current->key.source_delta_identity),
                 "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_compare(
                             current, baseline, NULL, &comparison, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_INCOMPATIBLE &&
                         strcmp(failure.field, "build_source_state") == 0,
                     "clean and dirty benchmark evidence cannot be compared");
    fixture_text(current->key.build_source_state,
                 sizeof(current->key.build_source_state), baseline->key.build_source_state);
    fixture_text(current->key.source_delta_identity,
                 sizeof(current->key.source_delta_identity),
                 baseline->key.source_delta_identity);
    fixture_text(current->key.build_identity, sizeof(current->key.build_identity),
                 "9999999999999999999999999999999999999999999999999999999999999999");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_compare(
                             current, baseline, NULL, &comparison, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_INCOMPATIBLE &&
                         strcmp(failure.field, "build_identity") == 0,
                     "different compiler or linker builds cannot compare as one build");
    fixture_text(current->key.build_identity, sizeof(current->key.build_identity),
                 baseline->key.build_identity);
    fixture_text(current->key.driver, sizeof(current->key.driver), "other-driver");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK &&
                         yvex_runtime_benchmark_compare(
                             current, baseline, NULL, &comparison, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_INCOMPATIBLE &&
                         strcmp(failure.field, "driver") == 0,
                     "driver mismatch refuses unrelated baseline comparison");
    fixture_text(current->key.driver, sizeof(current->key.driver), baseline->key.driver);
    return yvex_runtime_benchmark_baseline_seal(current, &failure, &err) == YVEX_OK ? 0 : 1;
}

static int test_regression_dimensions(const yvex_runtime_benchmark_baseline *baseline)
{
    yvex_runtime_benchmark_regression_policy policy = {.enabled = 1};
    yvex_runtime_benchmark_comparison comparison;
    yvex_runtime_benchmark_baseline current;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    unsigned int dimension;

    for (dimension = 0u; dimension < 3u; ++dimension) {
        current = *baseline;
        if (dimension == 0u) current.metrics.peak_host_bytes += 4096ull;
        if (dimension == 1u) current.metrics.h2d_bytes += 4096ull;
        if (dimension == 2u) current.metrics.last_dispatch_kernel_launches += 1ull;
        YVEX_TEST_ASSERT(
            yvex_runtime_benchmark_baseline_seal(&current, &failure, &err) == YVEX_OK &&
                yvex_runtime_benchmark_compare(
                    &current, baseline, &policy, &comparison, &failure, &err) == YVEX_OK &&
                !comparison.performance_passed,
            "zero-threshold policy detects memory, transfer, and launch regressions");
    }
    current = *baseline;
    current.metrics.warm_host_allocations = 1ull;
    YVEX_TEST_ASSERT(
        yvex_runtime_benchmark_baseline_seal(&current, &failure, &err) == YVEX_ERR_STATE &&
            failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_FIELD &&
            strcmp(failure.field, "steady-state") == 0,
        "warm allocation violates runtime qualification before performance policy");
    return 0;
}

static int test_independent_publication(const benchmark_fixture *fixture,
                                        const yvex_runtime_benchmark_baseline *current)
{
    yvex_runtime_benchmark_chart_request request = {
        .path = fixture->transaction_chart,
        .current = current,
    };
    yvex_runtime_benchmark_chart_result result;
    yvex_runtime_benchmark_baseline reopened;
    yvex_runtime_benchmark_publication baseline;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    const char marker[] = "pre-existing-chart";
    char *preserved;
    size_t preserved_count;
    int fd;

    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_write(
                         fixture->transaction_baseline, current, &baseline,
                         &failure, &err) == YVEX_OK && baseline.published,
                     "authoritative baseline publishes before optional visualization");
    request.path = fixture->transaction_baseline;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &result, &failure, &err) == YVEX_ERR_STATE &&
                         yvex_runtime_benchmark_baseline_open(
                             fixture->transaction_baseline, &reopened, &failure, &err) == YVEX_OK &&
                         strcmp(reopened.identity, current->identity) == 0,
                     "same-name chart conflict preserves the authenticated baseline");

    fd = open(fixture->transaction_chart,
              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    YVEX_TEST_ASSERT(fd >= 0 && write(fd, marker, sizeof(marker) - 1u) ==
                                      (ssize_t)(sizeof(marker) - 1u) &&
                         fsync(fd) == 0 && close(fd) == 0,
                     "pre-existing chart fixture is durably created");
    request.path = fixture->transaction_chart;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &result, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_CONFLICT &&
                         access(fixture->transaction_baseline, F_OK) == 0,
                     "chart conflict leaves the independently published baseline intact");
    preserved = fixture_read(fixture->transaction_chart, &preserved_count);
    YVEX_TEST_ASSERT(preserved && preserved_count == sizeof(marker) - 1u &&
                         memcmp(preserved, marker, sizeof(marker) - 1u) == 0,
                     "no-replace chart publication preserves the pre-existing bytes exactly");
    free(preserved);
    YVEX_TEST_ASSERT(unlink(fixture->transaction_chart) == 0,
                     "pre-existing chart fixture is removed exactly");

    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &result, &failure, &err) == YVEX_OK && result.generated &&
                         access(fixture->transaction_baseline, F_OK) == 0 &&
                         access(fixture->transaction_chart, F_OK) == 0,
                     "individually atomic baseline and chart publications both complete");
    return 0;
}

/* Prove malformed bytes, truncation, and symlink traversal fail closed. */
static int test_refusals(const benchmark_fixture *fixture,
                         const yvex_runtime_benchmark_baseline *record)
{
    yvex_runtime_benchmark_chart_request chart_request = {
        .path = "relative.svg",
        .current = record,
    };
    yvex_runtime_benchmark_chart_result chart;
    yvex_runtime_benchmark_baseline reopened;
    yvex_runtime_benchmark_baseline legacy = *record;
    yvex_runtime_benchmark_publication publication;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    char unsafe[YVEX_PATH_CAP];
    int input, output;
    char buffer[4096];
    ssize_t got;

    legacy.schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V1;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(
                         &legacy, &failure, &err) == YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
                     "legacy timing-only benchmark schema refuses explicit upgrade");
    legacy = *record;
    legacy.schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V2;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(
                         &legacy, &failure, &err) == YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
                     "schema two without exact source/build identities requires regeneration");
    legacy = *record;
    legacy.schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V3;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(
                         &legacy, &failure, &err) == YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
                     "schema three without typed device timing requires regeneration");
    legacy = *record;
    legacy.schema_version = YVEX_RUNTIME_BENCHMARK_SCHEMA_V4;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(
                         &legacy, &failure, &err) == YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA,
                     "schema four without complete reproducibility identity requires regeneration");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_write(
                         "relative.yvex-benchmark", record, &publication,
                         &failure, &err) == YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_PATH,
                     "runtime benchmark publication rejects relative destinations");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &chart_request, &chart, &failure, &err) == YVEX_ERR_INVALID_ARG &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_PATH,
                     "runtime benchmark chart publication rejects relative destinations");

    output = open(fixture->corrupt,
                  O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    YVEX_TEST_ASSERT(output >= 0 &&
                         write(output, "YVEX_RUNTIME_BENCHMARK_BASELINE\t1\n", 34u) == 34 &&
                         close(output) == 0,
                     "legacy file fixture writes one exact schema header");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_open(
                         fixture->corrupt, &reopened, &failure, &err) == YVEX_ERR_UNSUPPORTED &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_SCHEMA &&
                         failure.actual == YVEX_RUNTIME_BENCHMARK_SCHEMA_V1,
                     "legacy on-disk schema refuses with typed regeneration evidence");
    YVEX_TEST_ASSERT(unlink(fixture->corrupt) == 0,
                     "legacy file fixture is removed exactly before corruption proof");

    input = open(fixture->baseline, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    output = open(fixture->corrupt, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    YVEX_TEST_ASSERT(input >= 0 && output >= 0, "corruption copy opens safely");
    while ((got = read(input, buffer, sizeof(buffer))) != 0) {
        if (got < 0 && errno == EINTR) continue;
        YVEX_TEST_ASSERT(got > 0 && write(output, buffer, (size_t)got) == got,
                         "corruption copy writes exact fixture bytes");
    }
    YVEX_TEST_ASSERT(close(input) == 0 && close(output) == 0, "corruption copy closes");
    output = open(fixture->corrupt, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    YVEX_TEST_ASSERT(output >= 0 && pwrite(output, "z", 1u, 80) == 1 && close(output) == 0,
                     "corruption mutation writes one bounded byte");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_open(
                         fixture->corrupt, &reopened, &failure, &err) != YVEX_OK,
                     "corrupt baseline refuses independent reopen");
    YVEX_TEST_ASSERT(symlink(fixture->root, fixture->symlink_parent) == 0,
                     "symlink-parent fixture is created");
    YVEX_TEST_ASSERT(snprintf(unsafe, sizeof(unsafe), "%s/unsafe.yvex-benchmark",
                              fixture->symlink_parent) < (int)sizeof(unsafe),
                     "unsafe fixture path fits");
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_write(
                         unsafe, record, &publication, &failure, &err) != YVEX_OK &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_PATH,
                     "symlink parent is never traversed for publication");
    return 0;
}

/* Prove SVG bytes are escaped, identity-bound, deterministic, and never overwritten. */
static int test_chart(const benchmark_fixture *fixture,
                      const yvex_runtime_benchmark_baseline *current,
                      const yvex_runtime_benchmark_baseline *baseline)
{
    yvex_runtime_benchmark_chart_request request;
    yvex_runtime_benchmark_chart_result chart, repeated, conflict;
    yvex_runtime_benchmark_baseline cpu;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    char digest[YVEX_SHA256_HEX_BYTES];
    char *data, *repeated_data;
    size_t count, repeated_count;

    request = (yvex_runtime_benchmark_chart_request){
        .path = fixture->chart,
        .current = current,
        .baseline = NULL,
    };
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &chart, &failure, &err) == YVEX_OK && chart.generated,
                     "current-only chart publishes");
    data = fixture_read(fixture->chart, &count);
    YVEX_TEST_ASSERT(data && strstr(data, "NVIDIA GB10 &lt;&amp; verified&gt;") &&
                         strstr(data, "data-chart-schema=\"5\"") &&
                         strstr(data, "data-chart-design=\"2\"") &&
                         strstr(data, "<rect width=\"1440\" height=\"1024\"") &&
                         strstr(data, "Attention execution profile") &&
                         strstr(data, "Steady-state latency") &&
                         strstr(data, "Cold-start composition") &&
                         strstr(data, "Memory &amp; movement") &&
                         strstr(data, "Steady-state contract") &&
                         strstr(data, "Artifact authentication") &&
                         strstr(data, "Runtime binding") && strstr(data, "Runtime model seal") &&
                         strstr(data, "Weight residency") &&
                         strstr(data, "Encoded attention weights") &&
                         strstr(data, "Host workspace") && strstr(data, "Attention state") &&
                         strstr(data, "Cold resident H2D") && strstr(data, "SOURCE clean") &&
                         strstr(data, ">P95</text>") &&
                         strstr(data, "CUDA event timing") &&
                         strstr(data, "No warm weight I/O or allocation.") &&
                         strstr(data, ">QUALIFIED</text>") &&
                         strstr(data, current->identity) &&
                         fixture_digest(data, count, digest) && strcmp(digest, chart.identity) == 0,
                     "premium chart hierarchy escapes labels and hashes exact SVG bytes");
    request.path = fixture->deterministic_chart;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &repeated, &failure, &err) == YVEX_OK &&
                         strcmp(chart.identity, repeated.identity) == 0,
                     "equivalent chart inputs produce one exact identity on another path");
    repeated_data = fixture_read(fixture->deterministic_chart, &repeated_count);
    YVEX_TEST_ASSERT(repeated_data && count == repeated_count &&
                         memcmp(data, repeated_data, count) == 0,
                     "equivalent chart inputs serialize byte-identically across paths");
    free(repeated_data);
    free(data);
    cpu = *current;
    fixture_text(cpu.key.mode, sizeof(cpu.key.mode), "eager");
    fixture_text(cpu.key.device, sizeof(cpu.key.device), "cpu:test:x86_64:1");
    fixture_text(cpu.key.driver, sizeof(cpu.key.driver), "kernel:test:1");
    fixture_text(cpu.key.cuda_build, sizeof(cpu.key.cuda_build), "not-applicable");
    memset(&cpu.metrics.device_timing, 0, sizeof(cpu.metrics.device_timing));
    cpu.metrics.cuda_graph_launches = 0ull;
    cpu.metrics.cuda_graph_captures = 0ull;
    cpu.metrics.cuda_graph_replays = 0ull;
    cpu.metrics.cuda_graph_nodes = 0ull;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_baseline_seal(&cpu, &failure, &err) == YVEX_OK,
                     "CPU chart fixture seals with unavailable device timing");
    request.path = fixture->cpu_chart;
    request.current = &cpu;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &chart, &failure, &err) == YVEX_OK,
                     "CPU timing chart publishes");
    data = fixture_read(fixture->cpu_chart, &count);
    YVEX_TEST_ASSERT(data && strstr(data, "DEVICE TIMING UNAVAILABLE"),
                     "CPU chart never renders unavailable device timing as measured zero");
    free(data);
    request.current = current;
    request.path = fixture->chart;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &conflict, &failure, &err) == YVEX_ERR_STATE &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_CONFLICT,
                     "chart publication never overwrites an existing asset");
    request.path = fixture->compared_chart;
    request.baseline = baseline;
    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &chart, &failure, &err) == YVEX_OK,
                     "compatible comparison chart publishes");
    data = fixture_read(fixture->compared_chart, &count);
    YVEX_TEST_ASSERT(data && strstr(data, current->identity) && strstr(data, baseline->identity) &&
                         strstr(data, "COMPARISON / BASELINE") &&
                         fixture_digest(data, count, digest) && strcmp(digest, chart.identity) == 0,
                     "comparison chart binds both exact benchmark identities");
    free(data);
    return 0;
}

static int test_chart_cleanup_fault(const benchmark_fixture *fixture,
                                    const yvex_runtime_benchmark_baseline *current)
{
    yvex_core_file_faults faults = {
        .inject_file_close_failure = 1,
        .inject_cleanup_unlink_failure = 1,
    };
    yvex_runtime_benchmark_chart_request request = {
        .path = fixture->cleanup_chart,
        .current = current,
        .file_faults = &faults,
    };
    yvex_runtime_benchmark_chart_result result;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    char temporary[YVEX_PATH_CAP];

    YVEX_TEST_ASSERT(yvex_runtime_benchmark_chart_write(
                         &request, &result, &failure, &err) == YVEX_ERR_IO &&
                         failure.code == YVEX_RUNTIME_BENCHMARK_FAILURE_CLEANUP &&
                         strcmp(failure.field, "cleanup-temporary-unlink") == 0 &&
                         !result.generated && access(fixture->cleanup_chart, F_OK) != 0,
                     "benchmark cleanup failure is reachable and typed");
    YVEX_TEST_ASSERT(snprintf(temporary, sizeof(temporary), "%s/.cleanup.svg.%llu.0.tmp",
                              fixture->root, (unsigned long long)getpid()) <
                             (int)sizeof(temporary) &&
                         access(temporary, F_OK) != 0,
                     "cleanup fault removes only its exact temporary SVG name");
    return 0;
}

static void qualification_request(
    yvex_execution_qualification_request *request)
{
    memset(request, 0, sizeof(*request));
    request->schema_version = YVEX_EXECUTION_QUALIFICATION_SCHEMA_V1;
    request->source_relation = YVEX_EXECUTION_SOURCE_AUTHENTICATED;
    request->warm_state = YVEX_EXECUTION_WARM_STATE_COLD;
    request->contention_state = YVEX_EXECUTION_CONTENTION_ISOLATED;
    request->source_repository = "fixture/model";
    request->source_revision = "0123456789abcdef0123456789abcdef01234567";
    request->product_model = "fixture-model";
    request->specialization = "text";
    request->artifact_identity =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    request->runtime_binding_identity =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    request->deployment_profile = "fixture-profile";
    request->engine_kind = "text";
    request->execution_strategy = "target-only";
    request->runtime_model_identity =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    request->runtime_descriptor_identity =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    request->semantic_graph_identity =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    request->executable_graph_identity =
        "1111111111111111111111111111111111111111111111111111111111111111";
    request->backend = "cuda";
    request->device = "fixture-gpu";
    request->hardware_profile_identity =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    request->execution_profile_identity =
        "9999999999999999999999999999999999999999999999999999999999999999";
    request->context_capacity = 4096ull;
    request->prompt_identity =
        "2222222222222222222222222222222222222222222222222222222222222222";
    request->prompt_token_identity =
        "3333333333333333333333333333333333333333333333333333333333333333";
    request->generation_plan_identity =
        "4444444444444444444444444444444444444444444444444444444444444444";
    request->sampling_policy_identity =
        "5555555555555555555555555555555555555555555555555555555555555555";
    request->reasoning_policy = "disabled";
    request->maximum_new_tokens = 8ull;
    request->maximum_output_bytes = 1024ull;
    request->seed_present = 1;
    request->seed = 42ull;
    request->generation_execution_identity =
        "6666666666666666666666666666666666666666666666666666666666666666";
    request->generated_text_digest =
        "7777777777777777777777777777777777777777777777777777777777777777";
    request->measurement_identity =
        "8888888888888888888888888888888888888888888888888888888888888888";
    request->stop_reason = "max-new-tokens";
    request->prompt_tokens = 4ull;
    request->generated_tokens = 8ull;
    request->final_position = 12ull;
    request->time_to_first_token_ns = 1000000ull;
    request->total_generation_ns = 9000000ull;
}

static int test_execution_qualification(const benchmark_fixture *fixture)
{
    static const char other_artifact[] =
        "9999999999999999999999999999999999999999999999999999999999999999";
    yvex_execution_qualification_request request;
    yvex_execution_qualification_record base, changed, reopened, corrupt;
    yvex_execution_qualification_publication publication, conflict;
    yvex_error err;

    qualification_request(&request);
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_seal(&request, &base, &err) == YVEX_OK &&
            yvex_execution_qualification_validate(&base, &err) == YVEX_OK &&
            strlen(base.build_identity) == 64u &&
            strlen(base.execution_identity) == 64u &&
            strlen(base.environment_identity) == 64u &&
            strlen(base.run_identity) == 64u,
        "qualification seals independent build execution environment and run identities");
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_write(
            fixture->qualification, &base, &publication, &err) == YVEX_OK &&
            publication.published && publication.file_bytes > 0ull &&
            yvex_execution_qualification_open(
                fixture->qualification, &reopened, &err) == YVEX_OK &&
            !strcmp(reopened.run_identity, base.run_identity) &&
            !strcmp(reopened.build_identity, base.build_identity) &&
            !strcmp(reopened.execution_identity, base.execution_identity),
        "one bounded qualification independently reopens its exact identity chain");
    request.maximum_new_tokens = 16ull;
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_seal(&request, &changed, &err) == YVEX_OK &&
            !strcmp(changed.build_identity, base.build_identity) &&
            !strcmp(changed.execution_identity, base.execution_identity) &&
            !strcmp(changed.environment_identity, base.environment_identity) &&
            strcmp(changed.run_identity, base.run_identity),
        "runtime-only generation settings change run identity alone");
    qualification_request(&request);
    request.artifact_identity = other_artifact;
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_seal(&request, &changed, &err) == YVEX_OK &&
            !strcmp(changed.build_identity, base.build_identity) &&
            strcmp(changed.execution_identity, base.execution_identity) &&
            strcmp(changed.run_identity, base.run_identity),
        "immutable artifact mutation changes execution and run identity, not build");
    qualification_request(&request);
    request.warm_state = YVEX_EXECUTION_WARM_STATE_WARM;
    request.contention_state = YVEX_EXECUTION_CONTENTION_CONTENDED;
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_seal(&request, &changed, &err) == YVEX_OK &&
            !strcmp(changed.build_identity, base.build_identity) &&
            !strcmp(changed.execution_identity, base.execution_identity) &&
            strcmp(changed.environment_identity, base.environment_identity) &&
            strcmp(changed.run_identity, base.run_identity),
        "warm and contention conditions change environment and run identity alone");
    corrupt = base;
    corrupt.runtime_binding_identity[0] =
        corrupt.runtime_binding_identity[0] == '0' ? '1' : '0';
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_validate(&corrupt, &err) == YVEX_ERR_FORMAT,
        "execution identity mutation invalidates the qualification chain");
    YVEX_TEST_ASSERT(
        yvex_execution_qualification_write(
            fixture->qualification, &base, &conflict, &err) == YVEX_ERR_STATE &&
            !conflict.published,
        "qualification publication never replaces existing evidence");
    return 0;
}

int yvex_test_runtime_benchmark(void)
{
    benchmark_fixture fixture;
    yvex_runtime_benchmark_baseline baseline, current;
    yvex_runtime_benchmark_failure failure;
    yvex_error err;
    int result = 1;

    if (test_sample_distributions() != 0 || test_seal_identity() != 0 ||
        !fixture_open(&fixture))
        return 1;
    fixture_record(&baseline);
    if (yvex_runtime_benchmark_baseline_seal(&baseline, &failure, &err) != YVEX_OK) goto done;
    if (test_publication(&fixture, &baseline) != 0 ||
        test_comparison(&baseline, &current) != 0 ||
        test_regression_dimensions(&baseline) != 0 ||
        test_execution_qualification(&fixture) != 0 ||
        test_refusals(&fixture, &baseline) != 0 ||
        test_chart(&fixture, &current, &baseline) != 0 ||
        test_chart_cleanup_fault(&fixture, &current) != 0 ||
        test_independent_publication(&fixture, &current) != 0)
        goto done;
    result = 0;

done:
    if (!fixture_close(&fixture)) result = 1;
    return result;
}
