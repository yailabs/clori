/* Owner: src/cli/commands
 * Owns: graph command dispatch from typed input parser to report builder and renderer.
 * Does not own: graph construction, memory planning, backend probing, primitive execution, guard facts, report
 *   construction, rendering internals, generation, eval, benchmark, or release decisions.
 * Invariants: adapter stays thin: parse input, call one graph report API, render a typed report, and return an exit
 *   code.
 * Boundary: command dispatch is not graph runtime support.
 * Purpose: provide graph command dispatch from typed input parser to report builder and renderer.
 * Inputs: typed command arguments and borrowed domain APIs.
 * Effects: dispatches domain calls and routes operator bytes only through CLI I/O.
 * Failure: returns a stable CLI status while preserving domain ownership. */
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/model_artifacts/private.h"
#include "src/cli/render/private.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <yvex/internal/benchmark.h>
#include <yvex/internal/compilation.h>
#include <yvex/internal/core.h>
#include <yvex/internal/decode.h>
#include <yvex/internal/generation.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>

#include <build_commit.h>
#include <yvex/internal/source.h>

static volatile sig_atomic_t graph_attention_signal_seen;

typedef struct {
    int enabled;
    yvex_runtime_lifecycle_phase phase;
    double started, last_update;
} graph_attention_progress;
/* Purpose: return a stable operator label for one measured runtime lifecycle phase. */
static const char *graph_attention_progress_phase(yvex_runtime_lifecycle_phase phase)
{
    static const char *const names[YVEX_RUNTIME_LIFECYCLE_COUNT] = {
        "artifact-open", "artifact-hash", "artifact-admission", "binding-open",
        "materialization-open", "model-seal", "residency", "backend-open",
        "workspace-prepare", "graph-warmup", "graph-capture", "graph-instantiate",
        "execution", "publication", "cleanup"
    };

    return (unsigned int)phase < YVEX_RUNTIME_LIFECYCLE_COUNT ? names[phase] : "unknown";
}
/* Purpose: obtain monotonic seconds for progress throttling without affecting evidence identity. */
static double graph_attention_monotonic_seconds(void)
{
    struct timespec value;

    return clock_gettime(CLOCK_MONOTONIC, &value) == 0
               ? (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0 : 0.0;
}
/* Purpose: publish bounded cold-runtime progress to stderr and propagate cancellation.
 * Inputs: CLI-owned progress state plus exact runtime phase and byte counters.
 * Effects: emits at phase boundaries and at most once per second during long phases.
 * Failure: a pending signal cancels the underlying runtime operation.
 * Boundary: progress never enters stdout or semantic execution identities. */
static int graph_attention_progress_update(void *opaque,
                                           yvex_runtime_lifecycle_phase phase,
                                           unsigned long long completed,
                                           unsigned long long total)
{
    graph_attention_progress *state = (graph_attention_progress *)opaque;
    double now = graph_attention_monotonic_seconds();
    int phase_changed = !state || state->phase != phase;

    if (graph_attention_signal_seen)
        return 0;
    if (!state || !state->enabled)
        return 1;
    if (!phase_changed && completed != total && now - state->last_update < 1.0)
        return 1;
    state->phase = phase;
    state->last_update = now;
    if (total)
        (void)yvex_cli_out_writef(
            yvex_cli_out_stderr(),
            "runtime phase=%s completed=%llu total=%llu percent=%.1f elapsed=%.1fs\n",
            graph_attention_progress_phase(phase), completed, total,
            100.0 * (double)completed / (double)total, now - state->started);
    else
        (void)yvex_cli_out_writef(yvex_cli_out_stderr(),
                                  "runtime phase=%s elapsed=%.1fs\n",
                                  graph_attention_progress_phase(phase),
                                  now - state->started);
    return 1;
}
/* Purpose: enable progress only when explicitly requested or stderr is interactive.
 * Inputs: caller-owned progress state and validated mode text.
 * Effects: initializes local throttling timestamps and TTY policy.
 * Failure: unsupported modes are excluded by the CLI parser.
 * Boundary: configures presentation only and never changes runtime execution. */
static void graph_attention_progress_init(graph_attention_progress *state,
                                          const char *mode)
{
    memset(state, 0, sizeof(*state));
    state->enabled = strcmp(mode, "plain") == 0 ||
                     (strcmp(mode, "auto") == 0 && isatty(fileno(yvex_cli_out_stderr())));
    state->phase = YVEX_RUNTIME_LIFECYCLE_COUNT;
    state->started = graph_attention_monotonic_seconds();
}
/* Purpose: Record SIGINT/SIGTERM in one signal-safe scalar. */
static void graph_attention_signal_handler(int signal_number) {
    if (signal_number == SIGINT || signal_number == SIGTERM)
        graph_attention_signal_seen = signal_number;
}
/* Purpose: Report whether the signal-safe cancellation scalar is set. */
static int graph_attention_cancel_requested(void *context) {
    (void)context;
    return graph_attention_signal_seen != 0;
}
/* Purpose: Install cancellation handlers.
 * Inputs: prior-action storage. Effects: changes signal actions.
 * Failure: rolls back partial install. Boundary: cancellation request only; no runtime cleanup. */
static int graph_attention_signals_install(struct sigaction *old_interrupt,
                                           struct sigaction *old_terminate, yvex_error *err) {
    struct sigaction action;

    memset(&action, 0, sizeof(action));
    action.sa_handler = graph_attention_signal_handler;
    sigemptyset(&action.sa_mask);
    graph_attention_signal_seen = 0;
    if (sigaction(SIGINT, &action, old_interrupt) != 0) {
        yvex_error_setf(err, YVEX_ERR_IO, "graph_attention_cli",
                        "cannot install SIGINT handler: %s", strerror(errno));
        return YVEX_ERR_IO;
    }
    if (sigaction(SIGTERM, &action, old_terminate) != 0) {
        int saved_errno = errno;
        (void)sigaction(SIGINT, old_interrupt, NULL);
        yvex_error_setf(err, YVEX_ERR_IO, "graph_attention_cli",
                        "cannot install SIGTERM handler: %s", strerror(saved_errno));
        return YVEX_ERR_IO;
    }
    return YVEX_OK;
}
/* Purpose: Restore signal actions. Inputs: saved actions. Effects: changes handlers. Failure: I/O refusal.
 * Boundary: signal lifecycle only; graph resources remain runtime-owned. */
static int graph_attention_signals_restore(const struct sigaction *old_interrupt,
                                           const struct sigaction *old_terminate, yvex_error *err) {
    int interrupt_rc = sigaction(SIGINT, old_interrupt, NULL);
    int terminate_rc = sigaction(SIGTERM, old_terminate, NULL);

    if (interrupt_rc == 0 && terminate_rc == 0)
        return YVEX_OK;
    yvex_error_set(err, YVEX_ERR_IO, "graph_attention_cli",
                   "cannot restore attention cancellation handlers");
    return YVEX_ERR_IO;
}
/* Purpose: Print one parser refusal.
 * Inputs: typed error. Effects: writes CLI stderr.
 * Failure: stream state. Boundary: CLI diagnostics only. */
static int graph_cli_print_parse_error(const yvex_error *err) {
    yvex_cli_out_writef(yvex_cli_out_stderr(), "%s\n", yvex_error_message(err));
    return 2;
}
/* Purpose: Render graph print runtime error from typed facts (`graph_cli_print_runtime_error`). */
static int graph_cli_print_runtime_error(const yvex_error *err, int exit_code) {
    yvex_cli_out_writef(yvex_cli_out_stderr(), "yvex: %s: %s\n", yvex_error_where(err),
                        yvex_error_message(err));
    return exit_code;
}

typedef struct {
    yvex_graph_attention_operator_request request;
    yvex_operator_paths operator_paths;
    char source_path[YVEX_PATH_CAP];
    char source_manifest_path[YVEX_PATH_CAP];
    char artifact_path[YVEX_PATH_CAP];
    char runtime_binding_path[YVEX_PATH_CAP];
    char runtime_binding_dir[YVEX_PATH_CAP];
} graph_attention_request;

static const yvex_graph_attention_operator_result benchmark_comparison_result_default = {
    .completed = 1, .status = "complete",
    .command = "graph attention benchmark compare",
    .target = "not_applicable", .backend = "not_applicable",
    .scope = "attention_component", .operation_scope = "not_applicable",
    .phase = "not_applicable", .trace_policy = "none",
    .requested_mode = "not_applicable", .selected_mode = "not_applicable",
    .selection_reason = "identity_validation_required",
    .quality_status = {
        [YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK] = "compared",
        [YVEX_RUNTIME_QUALITY_CORRECTNESS] = "not_evaluated",
        [YVEX_RUNTIME_QUALITY_STRUCTURAL] = "not_evaluated",
        [YVEX_RUNTIME_QUALITY_PERFORMANCE] = "measured",
    },
    .benchmark_scope = "attention_component",
};
static const char benchmark_comparison_refusal_quality[YVEX_RUNTIME_QUALITY_STATUS_COUNT][24] = {
    [YVEX_RUNTIME_QUALITY_COMPONENT_BENCHMARK] = "unavailable",
    [YVEX_RUNTIME_QUALITY_CORRECTNESS] = "not_evaluated",
    [YVEX_RUNTIME_QUALITY_STRUCTURAL] = "not_evaluated",
    [YVEX_RUNTIME_QUALITY_PERFORMANCE] = "not_measured",
};

typedef struct {
    size_t destination, source, capacity;
} graph_benchmark_text_projection;

static const graph_benchmark_text_projection graph_benchmark_comparison_text[] = {
    {offsetof(yvex_runtime_benchmark_operator_summary, current_commit),
     offsetof(yvex_runtime_benchmark_comparison, current_commit), YVEX_RUNTIME_BENCHMARK_COMMIT_CAP},
    {offsetof(yvex_runtime_benchmark_operator_summary, baseline_commit),
     offsetof(yvex_runtime_benchmark_comparison, baseline_commit), YVEX_RUNTIME_BENCHMARK_COMMIT_CAP},
    {offsetof(yvex_runtime_benchmark_operator_summary, current_source_state),
     offsetof(yvex_runtime_benchmark_comparison, current_source_state), YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP},
    {offsetof(yvex_runtime_benchmark_operator_summary, baseline_source_state),
     offsetof(yvex_runtime_benchmark_comparison, baseline_source_state), YVEX_RUNTIME_BENCHMARK_SOURCE_STATE_CAP},
    {offsetof(yvex_runtime_benchmark_operator_summary, baseline_identity),
     offsetof(yvex_runtime_benchmark_comparison, baseline_identity), YVEX_SHA256_HEX_BYTES},
    {offsetof(yvex_runtime_benchmark_operator_summary, regression_policy_identity),
     offsetof(yvex_runtime_benchmark_comparison, regression_policy_identity), YVEX_SHA256_HEX_BYTES},
    {offsetof(yvex_runtime_benchmark_operator_summary, comparison_identity),
     offsetof(yvex_runtime_benchmark_comparison, comparison_identity), YVEX_SHA256_HEX_BYTES},
};

typedef struct {
    const char *text;
    unsigned int value;
} graph_attention_value;

typedef struct {
    const char *name;
    yvex_runtime_operator_action runtime_action;
} graph_attention_action;

static const graph_attention_value graph_attention_phases[] = {
    {"prefill", YVEX_RUNTIME_PHASE_ATTENTION_PREFILL}, {"mixed", YVEX_RUNTIME_PHASE_ATTENTION_MIXED},
{"verify", YVEX_RUNTIME_PHASE_ATTENTION_SPECULATIVE_VERIFY},
};
static const graph_attention_value graph_attention_modes[] = {
    {"piecewise", YVEX_RUNTIME_MODE_PIECEWISE}, {"full", YVEX_RUNTIME_MODE_FULL}, {"auto", YVEX_RUNTIME_MODE_AUTO},
};
static const graph_attention_value graph_attention_scopes[] = {
    {"envelope", YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE},
{"release-attention-set", YVEX_RUNTIME_SCOPE_RELEASE_ATTENTION_SET},
};
static const graph_attention_value graph_attention_traces[] = {
    {"summary", YVEX_RUNTIME_TRACE_SUMMARY}, {"stages", YVEX_RUNTIME_TRACE_STAGES}, {"full", YVEX_RUNTIME_TRACE_FULL},
};

static const graph_attention_action graph_attention_actions[] = {
    {"none", YVEX_RUNTIME_OPERATOR_EXECUTE}, {"prepare", YVEX_RUNTIME_OPERATOR_EXECUTE},
{"describe", YVEX_RUNTIME_OPERATOR_EXECUTE}, {"capabilities", YVEX_RUNTIME_OPERATOR_CAPABILITIES},
    {"plan", YVEX_RUNTIME_OPERATOR_PLAN}, {"execute", YVEX_RUNTIME_OPERATOR_EXECUTE},
{"compare", YVEX_RUNTIME_OPERATOR_EXECUTE}, {"state inspect", YVEX_RUNTIME_OPERATOR_STATE_INSPECT},
    {"state validate", YVEX_RUNTIME_OPERATOR_STATE_VALIDATE}, {"state exercise", YVEX_RUNTIME_OPERATOR_STATE_EXERCISE},
{"residency inspect", YVEX_RUNTIME_OPERATOR_RESIDENCY_INSPECT}, {"capture", YVEX_RUNTIME_OPERATOR_CAPTURE},
    {"replay", YVEX_RUNTIME_OPERATOR_REPLAY}, {"cuda-graph list", YVEX_RUNTIME_OPERATOR_GRAPH_LIST},
{"cuda-graph inspect", YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT},
{"cuda-graph warmup", YVEX_RUNTIME_OPERATOR_GRAPH_WARMUP},
    {"cuda-graph update", YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE},
{"cuda-graph invalidate", YVEX_RUNTIME_OPERATOR_GRAPH_INVALIDATE},
    {"cuda-graph release", YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE}, {"trace", YVEX_RUNTIME_OPERATOR_TRACE},
{"profile", YVEX_RUNTIME_OPERATOR_PROFILE}, {"benchmark", YVEX_RUNTIME_OPERATOR_BENCHMARK},
    {"benchmark compare", YVEX_RUNTIME_OPERATOR_EXECUTE}, {"qualify", YVEX_RUNTIME_OPERATOR_QUALIFY},
};
/* Purpose: Map one validated grammar value through its typed runtime catalog. */
static unsigned int graph_attention_value_find(
    const char *text, const graph_attention_value *values, size_t count,
    unsigned int fallback)
{
    size_t index;

    for (index = 0u; index < count; ++index)
        if (strcmp(text, values[index].text) == 0)
            return values[index].value;
    return fallback;
}
/* Purpose: Return the bounded action projection or the stable invalid sentinel. */
static const graph_attention_action *graph_attention_action_find(
    yvex_graph_attention_action action)
{
    static const graph_attention_action invalid = {"invalid", YVEX_RUNTIME_OPERATOR_EXECUTE};

    return (unsigned int)action < sizeof(graph_attention_actions) /
                                      sizeof(graph_attention_actions[0])
               ? &graph_attention_actions[action]
               : &invalid;
}
/* Purpose: Resolve exactly one immutable binding from an external registry directory.
 * Inputs: safely opened directory and caller-owned output.
 * Effects: reads directory entries only; never opens source/compiler assets.
 * Failure: missing or ambiguous content addresses require explicit operator selection.
 * Boundary: filename discovery is not runtime-binding admission. */
static int graph_attention_binding_discover(const char *directory, char *output,
                                            size_t capacity, yvex_error *err) {
    const size_t suffix_length = strlen(YVEX_RUNTIME_BINDING_SUFFIX);
    char selected[96] = {0};
    struct dirent *entry;
    DIR *stream = NULL;
    int directory_fd;
    unsigned int count = 0u;

    directory_fd = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (directory_fd < 0) {
        yvex_error_set(err, YVEX_ERR_IO, "graph_attention_cli",
                       "runtime binding is missing; run `yvex graph attention prepare`");
        return YVEX_ERR_IO;
    }
    stream = fdopendir(directory_fd);
    if (!stream) {
        (void)close(directory_fd);
        yvex_error_set(err, YVEX_ERR_IO, "graph_attention_cli",
                       "runtime binding registry could not be inspected");
        return YVEX_ERR_IO;
    }
    while ((entry = readdir(stream)) != NULL) {
        size_t length = strlen(entry->d_name);
        struct stat status;

        if (length <= suffix_length ||
            strcmp(entry->d_name + length - suffix_length, YVEX_RUNTIME_BINDING_SUFFIX) != 0)
            continue;
        if (length >= sizeof(selected))
            continue;
        if (fstatat(dirfd(stream), entry->d_name, &status, AT_SYMLINK_NOFOLLOW) != 0 ||
            !S_ISREG(status.st_mode))
            continue;
        ++count;
        if (count == 1u) {
            memcpy(selected, entry->d_name, length);
            selected[length] = '\0';
        }
    }
    if (closedir(stream) != 0 && count == 1u) {
        yvex_error_set(err, YVEX_ERR_IO, "graph_attention_cli",
                       "runtime binding registry close failed");
        return YVEX_ERR_IO;
    }
    if (count != 1u) {
        yvex_error_set(err, count == 0u ? YVEX_ERR_IO : YVEX_ERR_STATE,
                       "graph_attention_cli",
                       count == 0u
                           ? "runtime binding is missing; run `yvex graph attention prepare`"
                           : "runtime binding registry is ambiguous; use --runtime-binding FILE");
        return count == 0u ? YVEX_ERR_IO : YVEX_ERR_STATE;
    }
    return path_join2(output, capacity, directory, selected, err, "graph_attention_cli");
}
/* Purpose: resolve preparation-only family facts without extending the runtime adapter ABI. */
static const yvex_graph_family_preparation *graph_family_preparation_find(const char *target)
{
    unsigned long long index;

    for (index = 0ull;; ++index) {
        const yvex_graph_family_preparation *entry = yvex_graph_family_preparation_at(index);
        if (!entry) return NULL;
        if (entry->target_id && target && strcmp(entry->target_id, target) == 0) return entry;
    }
}
/* Purpose: Resolve attention paths.
 * Inputs: CLI args and output. Effects: builds a request.
 * Failure: typed path refusal. Boundary: no source, artifact, or runtime open. */
static int graph_cli_attention_request_build(const yvex_graph_args *args,
                                             graph_attention_request *out, yvex_error *err) {
    yvex_paths paths = {0};
    const yvex_runtime_family_adapter *adapter;
    const yvex_graph_family_api *graph;
    const yvex_graph_family_preparation *preparation = NULL;
    char gguf_dir[YVEX_PATH_CAP];
    char registry_runtime_dir[YVEX_PATH_CAP];
    int exists = 0;
    int rc;

    if (!args || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_cli",
                       "attention arguments and request output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->request.target = args->attention.target;
    out->request.artifact_path = out->artifact_path;
    out->request.activation_input_path = args->attention.input_file;
    out->request.probe = args->attention.input_file
                             ? YVEX_ATTENTION_PROBE_UNSPECIFIED
                             : YVEX_ATTENTION_PROBE_CANONICAL_V2;
    out->request.scope = strcmp(args->attention.coverage, "full") == 0
                             ? YVEX_ATTENTION_PROBE_SCOPE_FULL
                             : YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    out->request.compare_backends = args->attention.compare_backends;
    out->request.phase = (yvex_runtime_phase)graph_attention_value_find(
        args->attention.phase, graph_attention_phases,
        sizeof(graph_attention_phases) / sizeof(graph_attention_phases[0]),
        YVEX_RUNTIME_PHASE_ATTENTION_DECODE);
    out->request.mode = (yvex_runtime_execution_mode)graph_attention_value_find(
        args->attention.mode, graph_attention_modes,
        sizeof(graph_attention_modes) / sizeof(graph_attention_modes[0]),
        YVEX_RUNTIME_MODE_EAGER);
    out->request.operation_scope =
        (yvex_runtime_execution_scope)graph_attention_value_find(
            args->attention.operation_scope, graph_attention_scopes,
            sizeof(graph_attention_scopes) / sizeof(graph_attention_scopes[0]),
            YVEX_RUNTIME_SCOPE_ATTENTION_CORE);
    out->request.trace_policy = (yvex_runtime_trace_policy)graph_attention_value_find(
        args->attention.trace_level, graph_attention_traces,
        sizeof(graph_attention_traces) / sizeof(graph_attention_traces[0]),
        YVEX_RUNTIME_TRACE_NONE);
    out->request.operator_action =
        graph_attention_action_find(args->attention.action)->runtime_action;
    out->request.capture_bucket = args->attention.capture_bucket;
    out->request.attention_class = args->attention.attention_class;
    out->request.token_count = args->attention.token_count;
    out->request.chunk_tokens = args->attention.chunk_tokens;
    out->request.context_capacity = args->attention.context_capacity;
    out->request.warmup = args->attention.warmup;
    out->request.repeat = args->attention.repeat;
    if (args->attention.layer_seen || args->attention.layer_start_seen) {
        out->request.select_layer = 1;
        out->request.layer_start = args->attention.layer_seen
                                       ? args->attention.layer : args->attention.layer_start;
        out->request.layer_count = args->attention.layer_seen ? 1ull : args->attention.layer_count;
    }
    out->request.history_tokens = args->attention.history_tokens_seen
                                      ? args->attention.history_tokens : args->attention.position;
    memcpy(&out->request.maximum_host_bytes, &args->attention.maximum_host_bytes,
           2u * sizeof(unsigned long long));
    out->request.require_mode = args->attention.require_mode;
    out->request.backend = YVEX_BACKEND_KIND_CPU;
    adapter = yvex_runtime_family_adapter_find(args->attention.target);
    if (!adapter)
        return YVEX_OK;
    if (!adapter->operator_family_key || !adapter->operator_family_key[0] ||
        !adapter->operator_artifact_filename || !adapter->operator_artifact_filename[0]) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_cli",
                       "runtime family adapter lacks operator artifact facts");
        return YVEX_ERR_STATE;
    }
    graph = adapter->graph ? adapter->graph() : NULL;
    if (!graph || !graph->selection_key_resolve) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_cli",
                       "runtime family adapter lacks selection-key resolution");
        return YVEX_ERR_STATE;
    }
    if (args->attention.attention_class) {
        out->request.select_selection_key = 1;
        rc = graph->selection_key_resolve(args->attention.attention_class,
                                          &out->request.selection_key, err);
        if (rc != YVEX_OK) return rc;
    }
    if (!args->attention.compare_backends && args->attention.backend) {
        rc = yvex_backend_kind_parse(args->attention.backend, &out->request.backend, err);
        if (rc != YVEX_OK) return rc;
    }
    rc = yvex_graph_attention_operator_selection_validate(&out->request, err);
    if (rc != YVEX_OK) return rc;
    rc = yvex_operator_paths_resolve(&paths, args->attention.models_root,
                                     &out->operator_paths, err);
    if (rc != YVEX_OK)
        return rc;
    rc = path_join2(registry_runtime_dir, sizeof(registry_runtime_dir),
                    out->operator_paths.registry_root, "runtime", err,
                    "graph_attention_cli");
    if (rc == YVEX_OK)
        rc = path_join2(out->runtime_binding_dir, sizeof(out->runtime_binding_dir),
                        registry_runtime_dir, adapter->operator_family_key, err,
                        "graph_attention_cli");
    if (rc != YVEX_OK)
        return rc;
    if (args->attention.runtime_binding_dir) {
        rc = expand_operator_path(args->attention.runtime_binding_dir,
                                  out->runtime_binding_dir,
                                  sizeof(out->runtime_binding_dir), err,
                                  "graph_attention_cli");
        if (rc != YVEX_OK)
            return rc;
    }
    rc = yvex_operator_paths_resolve_target(
        &out->operator_paths, adapter->operator_family_key, "gguf", gguf_dir,
        sizeof(gguf_dir), &exists, err);
    if (rc != YVEX_OK)
        return rc;
    if (args->attention.artifact_path) {
        rc = expand_operator_path(args->attention.artifact_path, out->artifact_path,
                                  sizeof(out->artifact_path), err, "graph_attention_cli");
    } else {
        rc = path_join2(out->artifact_path, sizeof(out->artifact_path), gguf_dir,
                        adapter->operator_artifact_filename, err, "graph_attention_cli");
    }
    if (rc != YVEX_OK)
        return rc;

    if (args->attention.action == YVEX_GRAPH_ATTENTION_ACTION_PREPARE) {
        preparation = graph_family_preparation_find(args->attention.target);
        if (!preparation || !preparation->source_manifest_filename ||
            !preparation->source_manifest_filename[0] ||
            !preparation->prepare_runtime_binding) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_cli",
                           "runtime family adapter lacks operator preparation facts");
            return YVEX_ERR_STATE;
        }
        rc = yvex_operator_paths_resolve_target(
            &out->operator_paths, adapter->operator_family_key, "source", out->source_path,
            sizeof(out->source_path), &exists, err);
        if (rc == YVEX_OK)
            rc = path_join2(out->source_manifest_path, sizeof(out->source_manifest_path),
                            gguf_dir, preparation->source_manifest_filename, err,
                            "graph_attention_cli");
        if (rc != YVEX_OK)
            return rc;
    } else if (args->attention.runtime_binding_path) {
        rc = expand_operator_path(args->attention.runtime_binding_path,
                                  out->runtime_binding_path,
                                  sizeof(out->runtime_binding_path), err,
                                  "graph_attention_cli");
        if (rc != YVEX_OK)
            return rc;
        out->request.runtime_binding_path = out->runtime_binding_path;
    } else {
        rc = graph_attention_binding_discover(out->runtime_binding_dir,
                                              out->runtime_binding_path,
                                              sizeof(out->runtime_binding_path), err);
        if (rc != YVEX_OK)
            return rc;
        out->request.runtime_binding_path = out->runtime_binding_path;
    }
    if (args->attention.compare_backends) {
        out->request.backend = YVEX_BACKEND_KIND_CPU;
        return YVEX_OK;
    }
    return YVEX_OK;
}
/* Purpose: invoke one family-owned compiler preparation adapter.
 * Inputs: resolved preparation paths, typed family adapter facts, and output storage.
 * Effects: may transactionally publish one content-addressed binding outside the repository.
 * Failure: propagates the family preparation refusal without opening runtime state.
 * Boundary: the generic CLI never constructs source, transform, quant, writer, or family plans. */
static int graph_attention_binding_prepare(
    const graph_attention_request *request,
    yvex_compilation_runtime_binding_result *result, yvex_error *err)
{
    const yvex_runtime_family_adapter *adapter;
    const yvex_graph_family_preparation *preparation;
    yvex_compilation_runtime_binding_request prepare = {0};

    if (result) memset(result, 0, sizeof(*result));
    if (!request || !request->runtime_binding_dir[0] || !result ||
        !request->source_path[0] || !request->operator_paths.models_root[0] ||
        !request->source_manifest_path[0] || !request->artifact_path[0]) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_prepare",
                       "source, artifact, binding directory, and result are required");
        return YVEX_ERR_INVALID_ARG;
    }
    adapter = yvex_runtime_family_adapter_find(request->request.target);
    preparation = graph_family_preparation_find(request->request.target);
    if (!adapter || !preparation || !preparation->prepare_runtime_binding) {
        yvex_error_set(err, YVEX_ERR_UNSUPPORTED, "graph_attention_prepare",
                       "target has no complete runtime family preparation adapter");
        return YVEX_ERR_UNSUPPORTED;
    }
    prepare.source_path = request->source_path;
    prepare.models_root = request->operator_paths.models_root;
    prepare.source_manifest_path = request->source_manifest_path;
    prepare.artifact_path = request->artifact_path;
    prepare.directory = request->runtime_binding_dir;
    prepare.family_adapter_id = adapter->adapter_id;
    prepare.family_adapter_version = adapter->adapter_version;
    return preparation->prepare_runtime_binding(&prepare, result, err);
}

typedef yvex_graph_attention_operator_result graph_attention_result;
typedef yvex_runtime_binding_summary graph_attention_binding;
typedef struct {
    size_t result_offset, binding_offset;
} graph_binding_projection;

static const graph_binding_projection graph_binding_common_fields[] = {
    {offsetof(graph_attention_result, runtime_binding_identity), offsetof(graph_attention_binding, identity)},
{offsetof(graph_attention_result, artifact_identity), offsetof(graph_attention_binding, artifact_identity)},
    {offsetof(graph_attention_result, materialization_identity),
    offsetof(graph_attention_binding, materialization_identity)},
    {offsetof(graph_attention_result, logical_model_identity),
    offsetof(graph_attention_binding, logical_model_identity)},
    {offsetof(graph_attention_result, runtime_numeric_identity),
    offsetof(graph_attention_binding, runtime_numeric_identity)},
    {offsetof(graph_attention_result, runtime_descriptor_identity),
    offsetof(graph_attention_binding, runtime_descriptor_identity)},
    {offsetof(graph_attention_result, attention_plan_identity),
    offsetof(graph_attention_binding, attention_plan_identity)},
    {offsetof(graph_attention_result, semantic_graph_identity),
    offsetof(graph_attention_binding, semantic_graph_identity)},
    {offsetof(graph_attention_result, executable_graph_identity),
    offsetof(graph_attention_binding, executable_graph_identity)},
};

static const graph_binding_projection graph_binding_transform_fields[] = {
    {offsetof(graph_attention_result, artifact_transform_identity),
    offsetof(graph_attention_binding, artifact_transform_identity)},
    {offsetof(graph_attention_result, logical_transform_identity),
    offsetof(graph_attention_binding, logical_transform_identity)},
};
/* Purpose: project an ordered binding-field catalog into one operator result.
 * Inputs: immutable binding, caller-owned result, and typed field catalog.
 * Effects: copies only the catalogued bounded text fields in catalog order.
 * Failure: source contracts guarantee fit; bounded copies remain terminated.
 * Boundary: this projection neither admits the artifact nor derives identities. */
static void graph_attention_binding_project(
    const graph_attention_binding *binding, graph_attention_result *result,
    const graph_binding_projection *fields, size_t count)
{
    size_t index;

    for (index = 0u; index < count; ++index) {
        char *output = (char *)result + fields[index].result_offset;
        const char *value = (const char *)binding + fields[index].binding_offset;

        yvex_core_text_copy(output, YVEX_SHA256_HEX_BYTES, value);
    }
}
/* Purpose: initialize facts common to prepared and independently reopened runtime bindings.
 * Inputs: validated CLI paths, one immutable binding summary, and its external path.
 * Effects: initializes caller-owned presentation storage.
 * Failure: bounded source contracts keep every copied field terminated.
 * Boundary: common projection performs no runtime admission or execution. */
static void graph_attention_result_init(
    const yvex_graph_args *args, const graph_attention_request *request,
    const graph_attention_binding *binding, const char *binding_path,
    graph_attention_result *result)
{
    const yvex_runtime_family_adapter *adapter =
        yvex_runtime_family_adapter_find(args->attention.target);

    memset(result, 0, sizeof(*result));
    yvex_core_text_copy(result->status, sizeof(result->status), "complete");
    yvex_core_text_copy(result->target, sizeof(result->target), args->attention.target);
    yvex_core_text_copy(result->family, sizeof(result->family),
                              adapter ? adapter->family_name : "unavailable");
    yvex_core_text_copy(result->artifact_path, sizeof(result->artifact_path),
                              request->artifact_path);
    yvex_core_text_copy(result->runtime_binding_path,
                              sizeof(result->runtime_binding_path), binding_path);
    graph_attention_binding_project(
        binding, result, graph_binding_common_fields,
        sizeof(graph_binding_common_fields) / sizeof(graph_binding_common_fields[0]));
    result->main_layers_total = binding->layer_count;
    result->operator_command_available = 1;
    result->production_api_available = 1;
}
/* Purpose: project an independently reopened preparation result into stable operator facts.
 * Inputs: validated CLI paths, sealed binding summary, and content-addressed path.
 * Effects: initializes presentation data.
 * Failure: bounded fields remain terminated. Boundary: projection owns no compiler or runtime lifecycle. */
static void graph_attention_prepare_result(
    const yvex_graph_args *args, const graph_attention_request *request,
    const yvex_runtime_binding_summary *summary, const char *binding_path,
    yvex_graph_attention_operator_result *result)
{
    graph_attention_result_init(args, request, summary, binding_path, result);
    yvex_core_text_copy(result->command, sizeof(result->command),
                              "graph attention prepare");
    yvex_core_text_copy(result->backend, sizeof(result->backend), "not_applicable");
    yvex_core_text_copy(result->scope, sizeof(result->scope), "preparation");
}

static int graph_attention_result_render(
    const yvex_graph_args *args, const yvex_graph_attention_operator_result *result,
    yvex_error *err);
/* Purpose: publish one content-addressed binding through the preparation-plane API only.
 * Inputs: parsed CLI request and error output.
 * Effects: invokes the compiler-side producer once and renders its typed result.
 * Failure: maps preparation or rendering failures to a nonzero operator status.
 * Boundary: this is the sole CLI action allowed to rebuild preparation-plane truth. */
static int graph_cli_attention_prepare(const yvex_graph_args *args, yvex_error *err)
{
    graph_attention_request request;
    yvex_compilation_runtime_binding_result prepared;
    yvex_runtime_binding *binding = NULL;
    yvex_runtime_binding_failure binding_failure;
    yvex_runtime_binding_summary summary;
    yvex_graph_attention_operator_result result;
    char directory_probe[YVEX_PATH_CAP];
    int rc;

    memset(&request, 0, sizeof(request));
    memset(&prepared, 0, sizeof(prepared));
    memset(&binding_failure, 0, sizeof(binding_failure));
    if (!yvex_runtime_family_adapter_find(args->attention.target)) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "graph_attention_cli",
                        "unsupported attention target: %s", args->attention.target);
        return graph_cli_print_runtime_error(err, exit_for_status(YVEX_ERR_UNSUPPORTED));
    }
    rc = graph_cli_attention_request_build(args, &request, err);
    if (rc == YVEX_OK)
        rc = path_join2(directory_probe, sizeof(directory_probe), request.runtime_binding_dir,
                        ".binding-owner", err, "graph_attention_cli");
    if (rc == YVEX_OK)
        rc = yvex_core_mkdir_parent(directory_probe, "graph_attention_cli", err);
    if (rc == YVEX_OK)
        rc = graph_attention_binding_prepare(&request, &prepared, err);
    if (rc == YVEX_OK && (!prepared.published || !prepared.path[0])) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_cli",
                       "family preparation returned no published runtime binding");
        rc = YVEX_ERR_STATE;
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_open(
            &binding, prepared.path, &summary, NULL, &binding_failure, err);
    if (rc == YVEX_OK)
        graph_attention_prepare_result(args, &request, &summary, prepared.path, &result);
    if (rc == YVEX_OK) rc = graph_attention_result_render(args, &result, err);
    yvex_runtime_binding_close(binding);
    if (rc != YVEX_OK) {
        if (!yvex_error_is_set(err))
            yvex_error_set(err, rc, "graph_attention_cli", "runtime binding command failed");
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    }
    return 0;
}
/* Purpose: project one independently reopened binding into operator presentation facts.
 * Inputs: parsed CLI facts, resolved paths, sealed binding summary, and output storage.
 * Effects: initializes only caller-owned presentation fields.
 * Failure: all copied values are bounded by their typed source contracts.
 * Boundary: presentation projection neither admits the artifact nor executes attention. */
static void graph_attention_binding_result(
    const yvex_graph_args *args, const graph_attention_request *request,
    const yvex_runtime_binding_summary *binding,
    yvex_graph_attention_operator_result *result)
{
    graph_attention_result_init(args, request, binding, request->runtime_binding_path, result);
    (void)snprintf(result->command, sizeof(result->command), "graph attention %s",
                   graph_attention_action_find(args->attention.action)->name);
    yvex_core_text_copy(
        result->backend, sizeof(result->backend),
        args->attention.backend ? args->attention.backend : "not_applicable");
    yvex_core_text_copy(result->scope, sizeof(result->scope), args->attention.coverage);
    yvex_core_text_copy(result->operation_scope, sizeof(result->operation_scope),
                              args->attention.operation_scope);
    yvex_core_text_copy(result->phase, sizeof(result->phase), args->attention.phase);
    yvex_core_text_copy(result->requested_mode, sizeof(result->requested_mode),
                              args->attention.mode);
    yvex_core_text_copy(result->selection_reason,
                              sizeof(result->selection_reason), "not_applicable");
    graph_attention_binding_project(
        binding, result, graph_binding_transform_fields,
        sizeof(graph_binding_transform_fields) / sizeof(graph_binding_transform_fields[0]));
    result->internal_live_runner_available = 1;
}
/* Purpose: render and flush one typed attention result through the canonical CLI output owner.
 * Inputs: parsed render mode, immutable result, and caller-owned error.
 * Effects: writes one complete operator record to stdout.
 * Failure: preserves the renderer refusal with one stable CLI boundary.
 * Boundary: rendering never changes runtime status, evidence, or capability facts. */
static int graph_attention_result_render(
    const yvex_graph_args *args, const yvex_graph_attention_operator_result *result,
    yvex_error *err)
{
    int rc = yvex_graph_attention_render(
        yvex_cli_out_stdout(), args->render_mode, result);

    if (rc == YVEX_OK) rc = yvex_cli_out_flush(yvex_cli_out_stdout());
    if (rc != YVEX_OK)
        yvex_error_set(err, rc, "graph_attention_cli",
                       "attention result rendering failed");
    return rc;
}
/* Purpose: describe one independently reopened immutable runtime binding.
 * Inputs: typed paths and one content-addressed binding.
 * Effects: renders preparation facts without opening a runtime model or session.
 * Failure: typed binding refusal leaves no runtime or presentation state.
 * Boundary: every admitted capability and residency inspection uses the production operator. */
static int graph_cli_attention_describe(const yvex_graph_args *args, yvex_error *err) {
    graph_attention_request request;
    yvex_runtime_binding *binding = NULL;
    yvex_runtime_binding_failure binding_failure;
    yvex_graph_attention_operator_result result;
    yvex_runtime_binding_summary summary;
    int rc;

    memset(&request, 0, sizeof(request));
    memset(&binding_failure, 0, sizeof(binding_failure));
    rc = graph_cli_attention_request_build(args, &request, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_binding_open(
            &binding, request.runtime_binding_path, &summary, NULL, &binding_failure, err);
    if (rc == YVEX_OK)
        graph_attention_binding_result(args, &request, &summary, &result);
    if (rc == YVEX_OK) rc = graph_attention_result_render(args, &result, err);
    yvex_runtime_binding_close(binding);
    return rc == YVEX_OK ? 0 : graph_cli_print_runtime_error(err, exit_for_status(rc));
}
/* Purpose: project validated benchmark deltas and policy evidence into presentation units.
 * Inputs: caller-owned operator summary and a compatible runtime comparison.
 * Effects: copies bounded provenance, thresholds, status, and signed timing deltas.
 * Failure: none after runtime comparison validation. Boundary: CLI owns no benchmark policy. */
static void graph_attention_benchmark_comparison_apply(
    yvex_runtime_benchmark_operator_summary *summary,
    const yvex_runtime_benchmark_comparison *comparison)
{
    const double scale = 1.0 / 1000000000.0;
    unsigned char *destination = (unsigned char *)summary;
    const unsigned char *source = (const unsigned char *)comparison;
    size_t index;

    summary->baseline_compatible = comparison->compatible;
    for (index = 0u; index < sizeof(graph_benchmark_comparison_text) /
                                  sizeof(graph_benchmark_comparison_text[0]); ++index)
        yvex_core_text_copy(
            (char *)destination + graph_benchmark_comparison_text[index].destination,
            graph_benchmark_comparison_text[index].capacity,
            (const char *)source + graph_benchmark_comparison_text[index].source);
    summary->regression_basis_points =
        comparison->regression_policy.basis_points;
    summary->regression_policy_enabled = comparison->regression_policy.enabled;
    summary->performance_passed = comparison->performance_passed;
    summary->cold_delta_seconds = (double)comparison->cold_total_delta_ns * scale;
    summary->device_timing_available = comparison->device_timing_available;
    for (index = 0u; index < YVEX_RUNTIME_BENCHMARK_STATISTIC_COUNT; ++index) {
        summary->host_delta_seconds[index] =
            (double)comparison->host_delta_ns[index] * scale;
        summary->device_delta_seconds[index] =
            (double)comparison->device_delta_ns[index] * scale;
    }
}
/* Purpose: project explicitly supplied CLI thresholds into one domain policy.
 * Inputs: parsed sentinel-bearing basis points. Effects: fills one canonical enable/value pair.
 * Failure: none; the domain validates the resulting policy. Boundary: no threshold default is invented. */
static yvex_runtime_benchmark_regression_policy graph_attention_regression_policy(
    const yvex_graph_args *args)
{
    return (yvex_runtime_benchmark_regression_policy){
        .enabled = args->attention.regression_basis_points != ULLONG_MAX,
        .basis_points = args->attention.regression_basis_points == ULLONG_MAX
                            ? 0ull : args->attention.regression_basis_points,
    };
}
/* Purpose: publish one deterministic SVG through the runtime chart owner.
 * Inputs: validated path, sealed records, and typed summary.
 * Effects: publishes exact path, byte count, and chart identity.
 * Failure: preserves every pre-existing destination.
 * Boundary: the CLI adapts paths and owns no chart layout or benchmark semantics. */
static int graph_attention_benchmark_chart_publish(
    const char *path, const yvex_runtime_benchmark_baseline *current,
    const yvex_runtime_benchmark_baseline *baseline,
    yvex_runtime_benchmark_operator_summary *summary, yvex_error *err)
{
    yvex_runtime_benchmark_chart_request request = {
        .current = current, .baseline = baseline,
    };
    yvex_runtime_benchmark_chart_result chart;
    yvex_runtime_benchmark_failure failure;
    int rc = expand_operator_path(path, summary->chart_path, sizeof(summary->chart_path), err,
                                  "graph_attention_benchmark");
    if (rc != YVEX_OK) return rc;
    request.path = summary->chart_path;
    rc = yvex_runtime_benchmark_chart_write(&request, &chart, &failure, err);
    if (rc != YVEX_OK) return rc;
    summary->chart_generated = 1;
    summary->chart_file_bytes = chart.file_bytes;
    yvex_runtime_identity_copy(summary->chart_identity, chart.identity);
    return YVEX_OK;
}
/* Purpose: publish/compare baseline evidence and optionally create one exact-byte SVG asset.
 * Inputs: completed benchmark/profile result and validated external paths.
 * Effects: invokes the runtime benchmark file owner and fills typed operator evidence.
 * Failure: returns the failing operation while retaining any baseline published before the chart.
 * Boundary: this adapter does not measure, own thresholds, or serialize SVG itself. */
static int graph_attention_benchmark_output(
    const yvex_graph_args *args, yvex_graph_attention_operator_result *result,
    yvex_error *err)
{
    yvex_runtime_benchmark_baseline current, baseline;
    yvex_runtime_benchmark_publication publication;
    yvex_runtime_benchmark_comparison comparison;
    yvex_runtime_benchmark_failure failure;
    const yvex_runtime_benchmark_baseline *chart_baseline = NULL;
    int rc;

    if (args->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK &&
        args->attention.action != YVEX_GRAPH_ATTENTION_ACTION_PROFILE)
        return YVEX_OK;
    rc = yvex_runtime_benchmark_baseline_from_attention(
        result, &current, &failure, err);
    if (rc != YVEX_OK) return rc;
    yvex_runtime_identity_copy(result->benchmark.identity, current.identity);
    yvex_core_text_copy(result->benchmark.current_commit,
                        sizeof(result->benchmark.current_commit), current.key.commit);
    yvex_core_text_copy(result->benchmark.current_source_state,
                        sizeof(result->benchmark.current_source_state),
                        current.key.build_source_state);
    if (args->attention.baseline_path) {
        rc = expand_operator_path(args->attention.baseline_path, result->benchmark.path,
                                  sizeof(result->benchmark.path), err,
                                  "graph_attention_benchmark");
        if (rc != YVEX_OK) return rc;
        if (args->attention.write_baseline) {
            rc = yvex_runtime_benchmark_baseline_write(
                result->benchmark.path, &current, &publication, &failure, err);
            if (rc != YVEX_OK) return rc;
            result->benchmark.baseline_written = 1;
            result->benchmark.file_bytes = publication.file_bytes;
        } else if (!args->attention.write_baseline) {
            rc = yvex_runtime_benchmark_baseline_open(
                result->benchmark.path, &baseline, &failure, err);
            if (rc != YVEX_OK) return rc;
            rc = yvex_runtime_benchmark_compare(
                &current, &baseline, NULL, &comparison, &failure, err);
            if (rc != YVEX_OK) return rc;
            graph_attention_benchmark_comparison_apply(&result->benchmark, &comparison);
            chart_baseline = &baseline;
        }
    }
    if (!args->attention.chart_path) return YVEX_OK;
    return graph_attention_benchmark_chart_publish(
        args->attention.chart_path, &current, chart_baseline, &result->benchmark, err);
}
/* Purpose: preflight one benchmark asset path before runtime model or artifact admission.
 * Inputs: operator path, whether a new destination is required, and typed error output.
 * Effects: reads only parent/file metadata; creates, removes, and opens no model asset.
 * Failure: rejects relative, repository-owned, noncanonical, symlinked, conflicting, or absent paths.
 * Boundary: the runtime file owner repeats no-symlink and no-replace checks at publication. */
static int graph_attention_external_path_preflight(
    const char *input, int destination, yvex_error *err)
{
    char expanded[YVEX_PATH_CAP], parent[YVEX_PATH_CAP], canonical[YVEX_PATH_CAP];
    const char *root = YVEX_BUILD_SOURCE_ROOT;
    struct stat status;
    char *slash;
    size_t root_length;
    int rc;

    rc = expand_operator_path(input, expanded, sizeof(expanded), err,
                              "graph_attention_benchmark");
    if (rc != YVEX_OK) return rc;
    if (expanded[0] != '/') goto unsafe;
    if (snprintf(parent, sizeof(parent), "%s", expanded) >= (int)sizeof(parent))
        goto unsafe;
    slash = strrchr(parent, '/');
    if (!slash || !slash[1]) goto unsafe;
    if (slash == parent) slash[1] = '\0';
    else *slash = '\0';
    if (!realpath(parent, canonical) || strcmp(parent, canonical) != 0) goto unsafe;
    root_length = strlen(root);
    if (!root_length ||
        (strncmp(expanded, root, root_length) == 0 &&
         (expanded[root_length] == '\0' || expanded[root_length] == '/')))
        goto unsafe;
    if (lstat(expanded, &status) == 0) {
        if (S_ISLNK(status.st_mode)) goto unsafe;
        if (destination) {
            yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_benchmark",
                           "benchmark output path already exists");
            return YVEX_ERR_STATE;
        }
        if (!S_ISREG(status.st_mode)) goto unsafe;
    } else if (errno != ENOENT || !destination) {
        goto unsafe;
    }
    return YVEX_OK;

unsafe:
    yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph_attention_benchmark",
                   "benchmark assets require canonical absolute paths outside the source repository");
    return YVEX_ERR_INVALID_ARG;
}
/* Purpose: reject every unsafe benchmark output before expensive runtime preparation begins.
 * Inputs: fully parsed graph attention arguments. Effects: performs bounded path metadata checks only.
 * Failure: returns the first baseline or chart path refusal.
 * Boundary: no runtime binding, artifact, model, residency, or backend is opened. */
static int graph_attention_benchmark_paths_preflight(
    const yvex_graph_args *args, yvex_error *err)
{
    int rc;

    if (args->attention.action == YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE) {
        rc = graph_attention_external_path_preflight(args->attention.baseline_path, 0, err);
        if (rc != YVEX_OK) return rc;
        rc = graph_attention_external_path_preflight(args->attention.current_path, 0, err);
        if (rc != YVEX_OK || !args->attention.chart_path) return rc;
        return graph_attention_external_path_preflight(args->attention.chart_path, 1, err);
    }
    if (args->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK &&
        args->attention.action != YVEX_GRAPH_ATTENTION_ACTION_PROFILE)
        return YVEX_OK;
    if (args->attention.baseline_path) {
        rc = graph_attention_external_path_preflight(
            args->attention.baseline_path, args->attention.write_baseline, err);
        if (rc != YVEX_OK) return rc;
    }
    if (args->attention.chart_path)
        return graph_attention_external_path_preflight(
            args->attention.chart_path, 1, err);
    return YVEX_OK;
}
/* Purpose: project one typed CLI refusal without changing its domain error.
 * Inputs: caller-owned operator result and populated error. Effects: marks publication incomplete.
 * Failure: bounded copies retain stable fallback text. Boundary: this helper owns no refusal policy. */
static void graph_attention_result_refuse(graph_attention_result *result, const yvex_error *err)
{
    result->completed = 0;
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->failure_code, sizeof(result->failure_code), yvex_status_name(yvex_error_code(err)));
    yvex_core_text_copy(result->failure_where, sizeof(result->failure_where), yvex_error_where(err));
    yvex_core_text_copy(result->reason, sizeof(result->reason), yvex_error_message(err));
}
/* Purpose: compare two independently authenticated component-benchmark records.
 * Inputs: validated external baseline and current paths.
 * Effects: reads the two records and renders typed performance deltas.
 * Failure: incompatible identities or malformed records return a nonzero operator status.
 * Boundary: comparison executes no model work and cannot promote correctness or release claims. */
static int graph_cli_attention_benchmark_compare(const yvex_graph_args *args,
                                                  yvex_error *err)
{
    yvex_runtime_benchmark_baseline current, baseline;
    yvex_runtime_benchmark_comparison comparison;
    yvex_runtime_benchmark_regression_policy policy;
    yvex_runtime_benchmark_failure failure;
    yvex_graph_attention_operator_result result;
    int render_rc;
    int rc;

    result = benchmark_comparison_result_default;
    policy = graph_attention_regression_policy(args);
    rc = graph_attention_benchmark_paths_preflight(args, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_benchmark_baseline_open(args->attention.current_path, &current, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_benchmark_baseline_open(
            args->attention.baseline_path, &baseline, &failure, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_benchmark_compare(
            &current, &baseline, &policy, &comparison, &failure, err);
    if (rc == YVEX_OK && args->attention.chart_path)
        rc = graph_attention_benchmark_chart_publish(
            args->attention.chart_path, &current, &baseline, &result.benchmark, err);
    if (rc != YVEX_OK) {
        graph_attention_result_refuse(&result, err);
        memcpy(result.quality_status, benchmark_comparison_refusal_quality, sizeof(result.quality_status));
        render_rc = graph_attention_result_render(args, &result, err);
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc == YVEX_OK ? rc : render_rc));
    }
    yvex_runtime_identity_copy(result.benchmark.identity, current.identity);
    graph_attention_benchmark_comparison_apply(&result.benchmark, &comparison);
    yvex_core_text_copy(
        result.quality_status[YVEX_RUNTIME_QUALITY_PERFORMANCE], 24u,
        policy.enabled
            ? (comparison.performance_passed ? "pass" : "regressed")
            : "measured");
    yvex_core_text_copy(result.benchmark.path, sizeof(result.benchmark.path),
                        args->attention.baseline_path);
    rc = graph_attention_result_render(args, &result, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    return policy.enabled && !comparison.performance_passed ? 1 : 0;
}
/* Purpose: Execute attention.
 * Inputs: CLI args and runtime API. Effects: renders a typed result.
 * Failure: nonzero CLI status. Boundary: no attention math, oracle, or test indirection. */
static int graph_cli_attention_execute(const yvex_graph_args *args,
                                       yvex_runtime_cleanup_lease **retained_cleanup,
                                       yvex_error *err) {
    graph_attention_request request;
    yvex_graph_attention_operator_result result;
    struct sigaction old_interrupt;
    struct sigaction old_terminate;
    yvex_error restore_error;
    graph_attention_progress progress;
    int render_rc;
    int restore_rc;
    int cancellation_seen;
    int rc;

    memset(&request, 0, sizeof(request));
    memset(&result, 0, sizeof(result));
    rc = graph_attention_benchmark_paths_preflight(args, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    rc = graph_cli_attention_request_build(args, &request, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    rc = graph_attention_signals_install(&old_interrupt, &old_terminate, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.request.cancel_requested = graph_attention_cancel_requested;
    graph_attention_progress_init(&progress, args->attention.progress);
    request.request.progress = graph_attention_progress_update;
    request.request.progress_context = &progress;
    rc = yvex_graph_attention_operator_execute(
        &request.request, &result, retained_cleanup, err);
    yvex_error_clear(&restore_error);
    restore_rc = graph_attention_signals_restore(&old_interrupt, &old_terminate, &restore_error);
    cancellation_seen = graph_attention_signal_seen != 0;
    graph_attention_signal_seen = 0;
    if (restore_rc != YVEX_OK) {
        *err = restore_error;
        result.completed = 0;
        (void)snprintf(result.status, sizeof(result.status), "failed");
        yvex_core_text_copy(result.reason, sizeof(result.reason), yvex_error_message(&restore_error));
        rc = restore_rc;
    } else if (cancellation_seen && rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "graph_attention_cli",
                       "attention execution cancelled before command publication");
        result.completed = 0;
        (void)snprintf(result.status, sizeof(result.status), "cancelled");
        yvex_core_text_copy(result.reason, sizeof(result.reason), yvex_error_message(err));
        rc = YVEX_ERR_CANCELLED;
    }
    if (rc == YVEX_OK) {
        rc = graph_attention_benchmark_output(args, &result, err);
        if (rc != YVEX_OK) {
            graph_attention_result_refuse(&result, err);
        }
    }
    if (result.status[0] &&
        (render_rc = graph_attention_result_render(args, &result, err)) != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    if (!result.completed) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph_attention_cli",
                       "attention execution returned an incomplete result");
        return graph_cli_print_runtime_error(err, exit_for_status(YVEX_ERR_STATE));
    }
    return 0;
}

/* Purpose: execute the typed MoE command through the production runtime API.
 * Inputs: parsed CLI facts. Effects: renders once. Failure: typed exit. Boundary: adapter only. */
static int graph_cli_moe_execute(const yvex_graph_args *args,
                                 yvex_runtime_cleanup_lease **retained_cleanup,
                                 yvex_error *err)
{
    yvex_moe_operator_request request = {0};
    yvex_moe_operator_result result;
    struct sigaction old_interrupt, old_terminate;
    yvex_error restore_error;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP], input[YVEX_PATH_CAP];
    int rc, render_rc, restore_rc, cancellation_seen;

    memset(&result, 0, sizeof(result));
    rc = expand_operator_path(args->moe.artifact_path, artifact, sizeof(artifact), err,
                              "graph_moe_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->moe.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_moe_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->moe.input_file, input, sizeof(input), err,
                                  "graph_moe_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->moe.backend, &request.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.target = args->moe.target;
    request.artifact_path = artifact;
    request.runtime_binding_path = binding;
    request.input_path = input;
    request.maximum_host_bytes = args->moe.maximum_host_bytes;
    request.maximum_device_bytes = args->moe.maximum_device_bytes;
    rc = graph_attention_signals_install(&old_interrupt, &old_terminate, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.cancel_requested = graph_attention_cancel_requested;
    rc = yvex_runtime_moe_operator_execute(&request, &result, retained_cleanup, err);
    yvex_error_clear(&restore_error);
    restore_rc = graph_attention_signals_restore(&old_interrupt, &old_terminate,
                                                 &restore_error);
    cancellation_seen = graph_attention_signal_seen != 0;
    graph_attention_signal_seen = 0;
    if (restore_rc != YVEX_OK) {
        *err = restore_error;
        rc = restore_rc;
    } else if (cancellation_seen && rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "graph_moe_cli",
                       "MoE execution cancelled before command publication");
        rc = YVEX_ERR_CANCELLED;
    }
    render_rc = yvex_graph_moe_render(yvex_cli_out_stdout(), args->render_mode, &result);
    if (render_rc != YVEX_OK) {
        yvex_error_set(err, render_rc, "graph_moe_cli", "MoE result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    return result.completed ? 0 : graph_cli_print_runtime_error(
        err, exit_for_status(YVEX_ERR_STATE));
}

/* Purpose: execute the typed transformer command through the production runtime API.
 * Inputs: parsed CLI facts. Effects: renders once. Failure: typed exit. Boundary: adapter only. */
static int graph_cli_transformer_execute(
    const yvex_graph_args *args, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err)
{
    yvex_transformer_operator_request request = {0};
    yvex_transformer_operator_result result;
    struct sigaction old_interrupt, old_terminate;
    yvex_error restore_error;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP], input[YVEX_PATH_CAP];
    int rc, render_rc, restore_rc, cancellation_seen;

    memset(&result, 0, sizeof(result));
    rc = expand_operator_path(args->transformer.artifact_path, artifact,
                              sizeof(artifact), err, "graph_transformer_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_transformer_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.input_file, input,
                                  sizeof(input), err, "graph_transformer_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->transformer.backend, &request.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.target = args->transformer.target;
    request.artifact_path = artifact;
    request.runtime_binding_path = binding;
    request.input_path = input;
    request.chunk_tokens = args->transformer.chunk_tokens;
    request.context_capacity = args->transformer.context_capacity;
    request.maximum_host_bytes = args->transformer.maximum_host_bytes;
    request.maximum_device_bytes = args->transformer.maximum_device_bytes;
    rc = graph_attention_signals_install(&old_interrupt, &old_terminate, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.cancel_requested = graph_attention_cancel_requested;
    rc = yvex_transformer_operator_execute(&request, &result, retained_cleanup, err);
    yvex_error_clear(&restore_error);
    restore_rc = graph_attention_signals_restore(&old_interrupt, &old_terminate,
                                                 &restore_error);
    cancellation_seen = graph_attention_signal_seen != 0;
    graph_attention_signal_seen = 0;
    if (restore_rc != YVEX_OK) {
        *err = restore_error;
        rc = restore_rc;
    } else if (cancellation_seen && rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "graph_transformer_cli",
                       "transformer execution cancelled before command publication");
        rc = YVEX_ERR_CANCELLED;
    }
    render_rc = yvex_graph_transformer_render(
        yvex_cli_out_stdout(), args->render_mode, &result);
    if (render_rc != YVEX_OK) {
        yvex_error_set(err, render_rc, "graph_transformer_cli",
                       "transformer result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    return result.completed ? 0 : graph_cli_print_runtime_error(
        err, exit_for_status(YVEX_ERR_STATE));
}

/* Purpose: execute the typed prefill-to-decode command through one production runtime context.
 * Inputs: parsed CLI facts. Effects: renders complete or partial evidence once.
 * Failure: typed exit while preserving the domain-owned partial result. Boundary: adapter only. */
static int graph_cli_transformer_decode(
    const yvex_graph_args *args, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err)
{
    yvex_decode_operator_request request = {0};
    yvex_decode_operator_result result;
    struct sigaction old_interrupt, old_terminate;
    yvex_error restore_error;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP], input[YVEX_PATH_CAP];
    int rc, render_rc, restore_rc, cancellation_seen, exit_code;
    memset(&result, 0, sizeof(result));
    rc = expand_operator_path(args->transformer.artifact_path, artifact,
                              sizeof(artifact), err, "graph_decode_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_decode_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.input_file, input,
                                  sizeof(input), err, "graph_decode_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->transformer.backend, &request.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.target = args->transformer.target;
    request.artifact_path = artifact;
    request.runtime_binding_path = binding;
    request.input_path = input;
    request.prefill_tokens = args->transformer.prefill_tokens;
    request.prefill_chunk_tokens = args->transformer.prefill_chunk_tokens;
    request.context_capacity = args->transformer.context_capacity;
    request.maximum_host_bytes = args->transformer.maximum_host_bytes;
    request.maximum_device_bytes = args->transformer.maximum_device_bytes;
    rc = graph_attention_signals_install(&old_interrupt, &old_terminate, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.cancel_requested = graph_attention_cancel_requested;
    rc = yvex_runtime_decode_operator_execute(
        &request, &result, retained_cleanup, err);
    yvex_error_clear(&restore_error);
    restore_rc = graph_attention_signals_restore(&old_interrupt, &old_terminate,
                                                 &restore_error);
    cancellation_seen = graph_attention_signal_seen != 0;
    graph_attention_signal_seen = 0;
    if (restore_rc != YVEX_OK) {
        *err = restore_error;
        rc = restore_rc;
    } else if (cancellation_seen && rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "graph_decode_cli",
                       "decode execution cancelled before command publication");
        rc = YVEX_ERR_CANCELLED;
    }
    render_rc = yvex_graph_decode_render(
        yvex_cli_out_stdout(), args->render_mode, &result);
    if (render_rc != YVEX_OK) {
        yvex_runtime_decode_operator_result_release(&result);
        yvex_error_set(err, render_rc, "graph_decode_cli",
                       "decode result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    exit_code = rc == YVEX_OK ? (result.completed ? 0 : exit_for_status(YVEX_ERR_STATE))
                              : exit_for_status(rc);
    yvex_runtime_decode_operator_result_release(&result);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_code);
    if (exit_code)
        return graph_cli_print_runtime_error(err, exit_code);
    return 0;
}

/* Purpose: execute complete-vocabulary logits through the production runtime API.
 * Inputs: parsed transformer/logits paths, split, backend, and budgets.
 * Effects: renders typed complete-row evidence and releases its directory.
 * Failure: typed operator or rendering refusal preserves cleanup ownership.
 * Boundary: dispatches production logits without token selection or generation. */
static int graph_cli_transformer_logits(
    const yvex_graph_args *args, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err)
{
    yvex_logits_operator_request request = {0};
    yvex_logits_operator_result result;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP], input[YVEX_PATH_CAP];
    int rc, render_rc, exit_code;
    memset(&result, 0, sizeof(result));
    rc = expand_operator_path(args->transformer.artifact_path, artifact,
                              sizeof(artifact), err, "graph_logits_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_logits_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.input_file, input,
                                  sizeof(input), err, "graph_logits_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->transformer.backend,
                                     &request.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.target = args->transformer.target;
    request.artifact_path = artifact;
    request.runtime_binding_path = binding;
    request.input_path = input;
    request.prefill_tokens = args->transformer.prefill_tokens;
    request.prefill_chunk_tokens = args->transformer.prefill_chunk_tokens;
    request.context_capacity = args->transformer.context_capacity;
    request.maximum_host_bytes = args->transformer.maximum_host_bytes;
    request.maximum_device_bytes = args->transformer.maximum_device_bytes;
    rc = yvex_runtime_logits_operator_execute(
        &request, &result, retained_cleanup, err);
    render_rc = yvex_graph_logits_render(yvex_cli_out_stdout(),
                                         args->render_mode, &result);
    if (render_rc != YVEX_OK) {
        yvex_runtime_logits_operator_result_release(&result);
        yvex_error_set(err, render_rc, "graph_logits_cli",
                       "logits result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    exit_code = rc == YVEX_OK ? (result.completed ? 0 : exit_for_status(YVEX_ERR_STATE))
                              : exit_for_status(rc);
    yvex_runtime_logits_operator_result_release(&result);
    if (rc != YVEX_OK || exit_code)
        return graph_cli_print_runtime_error(err, exit_code);
    return 0;
}

/* Purpose: execute real-logits token selection through the common production sampler.
 * Inputs: parsed transformer workflow, explicit sampling policy, paths, backend, and budgets.
 * Effects: renders selected-token evidence without feeding it into decode or mutating KV.
 * Failure: typed operator or rendering refusal preserves cleanup ownership.
 * Boundary: sampling is operator-reachable but tokenizer and generation remain absent. */
static int graph_cli_transformer_sample(
    const yvex_graph_args *args, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err)
{
    yvex_sampling_operator_request request = {0};
    yvex_sampling_operator_result result;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP], input[YVEX_PATH_CAP];
    int rc, render_rc, exit_code;
    memset(&result, 0, sizeof(result));
    rc = expand_operator_path(args->transformer.artifact_path, artifact,
                              sizeof(artifact), err, "graph_sampling_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_sampling_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.input_file, input,
                                  sizeof(input), err, "graph_sampling_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->transformer.backend,
                                     &request.logits.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.logits.target = args->transformer.target;
    request.logits.artifact_path = artifact;
    request.logits.runtime_binding_path = binding;
    request.logits.input_path = input;
    request.logits.prefill_tokens = args->transformer.prefill_tokens;
    request.logits.prefill_chunk_tokens = args->transformer.prefill_chunk_tokens;
    request.logits.context_capacity = args->transformer.context_capacity;
    request.logits.maximum_host_bytes = args->transformer.maximum_host_bytes;
    request.logits.maximum_device_bytes = args->transformer.maximum_device_bytes;
    request.maximum_sampling_host_bytes = args->transformer.maximum_host_bytes;
    request.policy.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    request.policy.strategy = strcmp(args->transformer.strategy, "stochastic") == 0
                                  ? YVEX_SAMPLING_STRATEGY_STOCHASTIC
                                  : YVEX_SAMPLING_STRATEGY_GREEDY;
    request.policy.temperature = args->transformer.temperature;
    request.policy.top_k = args->transformer.top_k;
    request.policy.top_p = args->transformer.top_p;
    request.policy.min_p = args->transformer.min_p;
    request.policy.typical_p = args->transformer.typical_p;
    request.policy.seed_present = args->transformer.seed_seen;
    request.policy.seed = args->transformer.seed;
    rc = yvex_runtime_sampling_operator_execute(
        &request, &result, retained_cleanup, err);
    render_rc = yvex_graph_sampling_render(yvex_cli_out_stdout(),
                                           args->render_mode, &result);
    if (render_rc != YVEX_OK) {
        yvex_runtime_sampling_operator_result_release(&result);
        yvex_error_set(err, render_rc, "graph_sampling_cli",
                       "sampling result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    exit_code = rc == YVEX_OK ? (result.completed ? 0 : exit_for_status(YVEX_ERR_STATE))
                              : exit_for_status(rc);
    yvex_runtime_sampling_operator_result_release(&result);
    if (rc != YVEX_OK || exit_code)
        return graph_cli_print_runtime_error(err, exit_code);
    return 0;
}

/* Purpose: execute real autoregressive composition through one production generation context.
 * Inputs: validated text/messages, sampling policy, artifact paths, backend, and budgets.
 * Effects: renders bounded generated token/text evidence and preserves retry cleanup ownership.
 * Failure: typed runtime status retains exact partial progress. Boundary: no generation logic lives in CLI. */
static int graph_cli_transformer_generate(
    const yvex_graph_args *args, yvex_runtime_cleanup_lease **retained_cleanup,
    yvex_error *err)
{
    yvex_generation_operator_request request = {0};
    yvex_generation_operator_result result;
    yvex_prompt_message messages[2];
    struct sigaction old_interrupt, old_terminate;
    yvex_error restore_error;
    char artifact[YVEX_PATH_CAP], binding[YVEX_PATH_CAP];
    unsigned long long message_count = 0ull;
    int rc, render_rc, restore_rc, cancellation_seen, exit_code;
    memset(&result, 0, sizeof(result));
    memset(messages, 0, sizeof(messages));
    rc = expand_operator_path(args->transformer.artifact_path, artifact,
                              sizeof(artifact), err, "graph_generation_cli");
    if (rc == YVEX_OK)
        rc = expand_operator_path(args->transformer.runtime_binding_path, binding,
                                  sizeof(binding), err, "graph_generation_cli");
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_parse(args->transformer.backend,
                                     &request.backend, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.target = args->transformer.target;
    request.artifact_path = artifact;
    request.runtime_binding_path = binding;
    if (args->transformer.text) {
        request.input_kind = YVEX_GENERATION_INPUT_TEXT;
        request.text = (const unsigned char *)args->transformer.text;
        request.text_bytes = strlen(args->transformer.text);
    } else {
        request.input_kind = YVEX_GENERATION_INPUT_MESSAGES;
        if (args->transformer.system) {
            messages[message_count].role = YVEX_PROMPT_ROLE_SYSTEM;
            messages[message_count].content = args->transformer.system;
            messages[message_count++].content_len = strlen(args->transformer.system);
        }
        messages[message_count].role = YVEX_PROMPT_ROLE_USER;
        messages[message_count].content = args->transformer.user;
        messages[message_count++].content_len = strlen(args->transformer.user);
        request.messages = messages;
        request.message_count = message_count;
        request.prompt_options.mode = YVEX_PROMPT_MODE_CHAT;
        request.prompt_options.add_generation_prompt = 1;
        request.prompt_options.drop_thinking = 1;
    }
    request.encode_options.allow_special_tokens = 1;
    request.encode_options.maximum_tokens = args->transformer.context_capacity;
    request.context_capacity = args->transformer.context_capacity;
    request.prefill_chunk_tokens = args->transformer.prefill_chunk_tokens;
    request.maximum_new_tokens = args->transformer.maximum_new_tokens;
    request.maximum_output_bytes = args->transformer.maximum_output_bytes;
    request.maximum_host_bytes = args->transformer.maximum_host_bytes;
    request.maximum_device_bytes = args->transformer.maximum_device_bytes;
    request.sampling_policy.schema_version = YVEX_RUNTIME_SAMPLING_SCHEMA_V1;
    request.sampling_policy.strategy = strcmp(args->transformer.strategy, "stochastic") == 0
                                           ? YVEX_SAMPLING_STRATEGY_STOCHASTIC
                                           : YVEX_SAMPLING_STRATEGY_GREEDY;
    request.sampling_policy.temperature = args->transformer.temperature;
    request.sampling_policy.top_k = args->transformer.top_k;
    request.sampling_policy.top_p = args->transformer.top_p;
    request.sampling_policy.min_p = args->transformer.min_p;
    request.sampling_policy.typical_p = args->transformer.typical_p;
    request.sampling_policy.seed_present = args->transformer.seed_seen;
    request.sampling_policy.seed = args->transformer.seed;
    rc = graph_attention_signals_install(&old_interrupt, &old_terminate, err);
    if (rc != YVEX_OK)
        return graph_cli_print_runtime_error(err, exit_for_status(rc));
    request.cancel_requested = graph_attention_cancel_requested;
    rc = yvex_runtime_generation_operator_execute(
        &request, &result, retained_cleanup, err);
    yvex_error_clear(&restore_error);
    restore_rc = graph_attention_signals_restore(&old_interrupt, &old_terminate,
                                                 &restore_error);
    cancellation_seen = graph_attention_signal_seen != 0;
    graph_attention_signal_seen = 0;
    if (restore_rc != YVEX_OK) {
        *err = restore_error;
        rc = restore_rc;
    } else if (cancellation_seen && rc == YVEX_OK) {
        yvex_error_set(err, YVEX_ERR_CANCELLED, "graph_generation_cli",
                       "generation cancelled before command publication");
        rc = YVEX_ERR_CANCELLED;
    }
    render_rc = yvex_graph_generation_render(yvex_cli_out_stdout(),
                                              args->render_mode, &result);
    exit_code = rc == YVEX_OK ? (result.completed ? 0 : exit_for_status(YVEX_ERR_STATE))
                              : exit_for_status(rc);
    yvex_runtime_generation_operator_result_release(&result);
    if (render_rc != YVEX_OK) {
        yvex_error_set(err, render_rc, "graph_generation_cli",
                       "generation result rendering failed");
        return graph_cli_print_runtime_error(err, exit_for_status(render_rc));
    }
    if (rc != YVEX_OK || exit_code)
        return graph_cli_print_runtime_error(err, exit_code);
    return 0;
}
/* Purpose: Dispatch graph.
 * Inputs: argv. Effects: executes and renders a typed request.
 * Failure: nonzero CLI status. Boundary: domain owners retain capability truth. */
int yvex_graph_command(int argc, char **argv,
                       yvex_runtime_cleanup_lease **retained_cleanup) {
    yvex_graph_args args;
    yvex_error err;
    int rc;

    yvex_error_clear(&err);

    if (!retained_cleanup || *retained_cleanup) {
        yvex_error_set(&err, YVEX_ERR_INVALID_ARG, "graph_cli",
                       "empty retained-cleanup output is required");
        return graph_cli_print_runtime_error(&err, exit_for_status(YVEX_ERR_INVALID_ARG));
    }

    rc = yvex_graph_args_parse(argc, argv, &args, &err);
    if (rc != YVEX_OK) {
        return graph_cli_print_parse_error(&err);
    }

    if (args.help_requested) {
        (void)yvex_graph_render_help(yvex_cli_out_stdout());
        return args.help_exit_code;
    }

    if (args.moe.active)
        return graph_cli_moe_execute(&args, retained_cleanup, &err);
    if (args.transformer.active && args.transformer.generate)
        return graph_cli_transformer_generate(&args, retained_cleanup, &err);
    if (args.transformer.active && args.transformer.sample)
        return graph_cli_transformer_sample(&args, retained_cleanup, &err);
    if (args.transformer.active && args.transformer.logits)
        return graph_cli_transformer_logits(&args, retained_cleanup, &err);
    if (args.transformer.active && args.transformer.decode)
        return graph_cli_transformer_decode(&args, retained_cleanup, &err);
    if (args.transformer.active)
        return graph_cli_transformer_execute(&args, retained_cleanup, &err);
    if (args.attention.action == YVEX_GRAPH_ATTENTION_ACTION_PREPARE)
        return graph_cli_attention_prepare(&args, &err);
    if (args.attention.action == YVEX_GRAPH_ATTENTION_ACTION_DESCRIBE)
        return graph_cli_attention_describe(&args, &err);
    if (args.attention.action == YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE)
        return graph_cli_attention_benchmark_compare(&args, &err);
    if (args.attention.action >= YVEX_GRAPH_ATTENTION_ACTION_CAPABILITIES) {
        return graph_cli_attention_execute(&args, retained_cleanup, &err);
    }
    yvex_error_set(&err, YVEX_ERR_STATE, "graph_cli",
                   "validated graph request has no attention action");
    return graph_cli_print_runtime_error(&err, exit_for_status(YVEX_ERR_STATE));
}
/* Purpose: Render graph help.
 * Inputs: output stream. Effects: writes CLI text.
 * Failure: stream state. Boundary: CLI presentation. */
void yvex_graph_help(FILE *fp) {
    (void)yvex_graph_render_help(fp);
}
