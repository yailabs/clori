/* Owner: src/cli/input
 * Owns: CLI grammar, option defaults, and parser validation for `yvex graph`.
 * Does not own: model reference resolution, artifact inspection, graph construction, backend probing, primitive
 *   execution, report building, rendering, stdout/stderr output, generation, eval, benchmark, or
 *   release decisions.
 * Invariants: this parser only reads borrowed argc/argv and fills a typed request.
 * Boundary: parsing graph options is not graph runtime support.
 * Purpose: provide cLI grammar, option defaults, and parser validation for `yvex graph`.
 * Inputs: bounded command arguments and caller-owned typed request storage.
 * Effects: publishes request fields only after complete grammar validation.
 * Failure: invalid or ambiguous grammar leaves the request uncommitted. */
#include "src/cli/input/private.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* Purpose: Compute graph arg error for its CLI invariant (`graph_arg_error`). */
static int graph_arg_error(yvex_error *err, const char *message) {
    yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph", message);
    return YVEX_ERR_INVALID_ARG;
}
/* Purpose: Compute graph arg errorf for its CLI invariant (`graph_arg_errorf`). */
static int graph_arg_errorf(yvex_error *err, const char *fmt, const char *value) {
    yvex_error_setf(err, YVEX_ERR_INVALID_ARG, "graph", fmt, value ? value : "");
    return YVEX_ERR_INVALID_ARG;
}
/* Purpose: initialize canonical graph CLI defaults.
 * Inputs: caller-owned argument storage. Effects: replaces it with deterministic defaults.
 * Failure: none. Boundary: defaults neither resolve paths nor admit capabilities. */
static void graph_args_defaults(yvex_graph_args *out) {
    memset(out, 0, sizeof(*out));
    out->attention.probe = "canonical";
    out->attention.coverage = "quick";
    out->attention.phase = "decode";
    out->attention.mode = "eager";
    out->attention.operation_scope = "core";
    out->attention.trace_level = "none";
    out->attention.progress = "auto";
    out->attention.token_count = 1ull;
    out->attention.repeat = 1ull;
    out->attention.regression_basis_points = ULLONG_MAX;
    out->moe.coverage = "full";
    out->moe.progress = "off";
    out->transformer.phase = "prefill";
    out->transformer.input_class = "token-ids";
    out->transformer.progress = "off";
    out->transformer.strategy = "greedy";
    out->transformer.temperature = 1.0;
    out->transformer.top_p = 1.0;
    out->transformer.typical_p = 1.0;
}

/* Purpose: parse one complete finite decimal without accepting suffixes.
 * Inputs: borrowed text and caller result. Effects: publishes only on success.
 * Failure: returns false for syntax, range, or non-finite values. Boundary: CLI parsing only. */
static int graph_parse_finite_double(const char *text, double *out)
{
    char *end = NULL;
    double value;
    if (!text || !text[0] || !out) return 0;
    errno = 0;
    value = strtod(text, &end);
    if (errno || !end || *end || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

/* Purpose: parse one complete nonnegative U64, including zero.
 * Inputs: borrowed text and caller result. Effects: publishes only on success.
 * Failure: returns false for syntax or range. Boundary: CLI parsing only. */
static int graph_parse_nonnegative_u64(const char *text, unsigned long long *out)
{
    char *end = NULL;
    unsigned long long value;
    if (!text || !text[0] || text[0] == '-' || !out) return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *out = value;
    return 1;
}

typedef enum {
    GRAPH_OPTION_TEXT,
    GRAPH_OPTION_POSITIVE,
    GRAPH_OPTION_NONNEGATIVE,
    GRAPH_OPTION_FLAG,
    GRAPH_OPTION_OUTPUT,
    GRAPH_OPTION_AUDIT,
    GRAPH_OPTION_JSON
} graph_option_kind;

enum { GRAPH_PARSE_ATTENTION = 1u };

typedef struct {
    unsigned int seen;
} graph_parse_state;

typedef struct {
    const char *flag;
    graph_option_kind kind;
    size_t offset;
    size_t companion_offset;
    unsigned int seen;
    unsigned int modes;
    unsigned long long maximum;
    const char *error;
} graph_option_spec;

#define GRAPH_ARG(member) offsetof(yvex_graph_args, member)
#define GRAPH_ATTN(member) GRAPH_ARG(attention.member)
#define NO_FIELD ((size_t) - 1)

static const graph_option_spec graph_options[] = {
    {"--target", GRAPH_OPTION_TEXT, GRAPH_ATTN(target), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--artifact", GRAPH_OPTION_TEXT, GRAPH_ATTN(artifact_path), NO_FIELD, 0, GRAPH_PARSE_ATTENTION,
0, "yvex: graph attention option requires a value"},
    {"--runtime-binding", GRAPH_OPTION_TEXT, GRAPH_ATTN(runtime_binding_path), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--runtime-binding-dir", GRAPH_OPTION_TEXT, GRAPH_ATTN(runtime_binding_dir), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--models-root", GRAPH_OPTION_TEXT, GRAPH_ATTN(models_root), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--backend", GRAPH_OPTION_TEXT, GRAPH_ATTN(backend), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--probe", GRAPH_OPTION_TEXT, GRAPH_ATTN(probe), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--scope", GRAPH_OPTION_TEXT, GRAPH_ATTN(coverage), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--phase", GRAPH_OPTION_TEXT, GRAPH_ATTN(phase), GRAPH_ATTN(phase_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--mode", GRAPH_OPTION_TEXT, GRAPH_ATTN(mode), GRAPH_ATTN(mode_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: graph attention option requires a value"},
    {"--operation-scope", GRAPH_OPTION_TEXT, GRAPH_ATTN(operation_scope), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--trace-level", GRAPH_OPTION_TEXT, GRAPH_ATTN(trace_level), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--progress", GRAPH_OPTION_TEXT, GRAPH_ATTN(progress), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: graph attention option requires a value"},
    {"--input", GRAPH_OPTION_TEXT, GRAPH_ATTN(input_class), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --input requires tensor-file"},
    {"--input-file", GRAPH_OPTION_TEXT, GRAPH_ATTN(input_file), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --input-file requires a file path"},
    {"--layer", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(layer), GRAPH_ATTN(layer_seen), 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --layer requires a non-negative integer"},
    {"--layer-start", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(layer_start),
GRAPH_ATTN(layer_start_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --layer-start requires a non-negative integer"},
    {"--layer-count", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(layer_count),
GRAPH_ATTN(layer_count_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --layer-count requires a positive integer"},
    {"--class", GRAPH_OPTION_TEXT, GRAPH_ATTN(attention_class), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --class requires a family selection token"},
    {"--position", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(position),
GRAPH_ATTN(position_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --position requires a non-negative integer"},
    {"--history-tokens", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(history_tokens),
GRAPH_ATTN(history_tokens_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --history-tokens requires a non-negative integer"},
    {"--local-capacity", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(local_capacity),
GRAPH_ATTN(local_capacity_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --local-capacity requires a non-negative integer"},
    {"--compressed-capacity", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(compressed_capacity),
GRAPH_ATTN(compressed_capacity_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --compressed-capacity requires a non-negative integer"},
    {"--indexer-capacity", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(indexer_capacity),
GRAPH_ATTN(indexer_capacity_seen), 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --indexer-capacity requires a non-negative integer"},
    {"--capture-bucket", GRAPH_OPTION_TEXT, GRAPH_ATTN(capture_bucket), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --capture-bucket requires a bucket identifier"},
    {"--max-host-bytes", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(maximum_host_bytes), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --max-host-bytes requires a positive integer"},
    {"--max-device-bytes", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(maximum_device_bytes), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --max-device-bytes requires a positive integer"},
    {"--require-mode", GRAPH_OPTION_FLAG, GRAPH_ATTN(require_mode), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0, NULL},
{"--baseline", GRAPH_OPTION_TEXT, GRAPH_ATTN(baseline_path), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --baseline requires a file path"},
    {"--current", GRAPH_OPTION_TEXT, GRAPH_ATTN(current_path), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --current requires a file path"},
    {"--max-regression-bps", GRAPH_OPTION_NONNEGATIVE,
GRAPH_ATTN(regression_basis_points), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --max-regression-bps requires non-negative basis points"},
    {"--write-baseline", GRAPH_OPTION_FLAG, GRAPH_ATTN(write_baseline), NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0, NULL},
{"--chart", GRAPH_OPTION_TEXT, GRAPH_ATTN(chart_path), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --chart requires an external SVG path"},
    {"--tokens", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(token_count), GRAPH_ATTN(token_count_seen), 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --tokens requires a positive integer"},
    {"--chunk-tokens", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(chunk_tokens), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --chunk-tokens requires a positive integer"},
    {"--context-capacity", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(context_capacity), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --context-capacity requires a positive integer"},
    {"--warmup", GRAPH_OPTION_NONNEGATIVE, GRAPH_ATTN(warmup), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --warmup requires a non-negative integer"},
    {"--repeat", GRAPH_OPTION_POSITIVE, GRAPH_ATTN(repeat), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, "yvex: --repeat requires a positive integer"},
    {"--compare-backends", GRAPH_OPTION_FLAG, GRAPH_ATTN(compare_backends), NO_FIELD, 0,
GRAPH_PARSE_ATTENTION, 0, NULL},
    {"--output", GRAPH_OPTION_OUTPUT, NO_FIELD, NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0,
     "yvex: --output requires normal|table|audit|json|csv for attention"},
    {"--audit", GRAPH_OPTION_AUDIT, NO_FIELD, NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0, NULL},
{"--json", GRAPH_OPTION_JSON, NO_FIELD, NO_FIELD, 0, GRAPH_PARSE_ATTENTION, 0, NULL},
};

#undef GRAPH_ATTN
#undef GRAPH_ARG

static const char *const graph_output_names[] = {
    "normal", "table", "audit", "json", "csv"};
static const yvex_graph_report_mode graph_output_modes[] = {
    YVEX_GRAPH_REPORT_MODE_NORMAL, YVEX_GRAPH_REPORT_MODE_TABLE,
    YVEX_GRAPH_REPORT_MODE_AUDIT, YVEX_GRAPH_REPORT_MODE_JSON,
    YVEX_GRAPH_REPORT_MODE_CSV};
/* Purpose: resolve one option in the immutable grammar for the selected command mode. */
static const graph_option_spec *graph_option_find(const char *flag, unsigned int mode) {
    size_t index;

    for (index = 0; index < sizeof(graph_options) / sizeof(graph_options[0]); ++index) {
        if ((graph_options[index].modes & mode) && strcmp(flag, graph_options[index].flag) == 0)
            return &graph_options[index];
    }
    return NULL;
}
/* Purpose: Parse an output mode.
 * Inputs: text and output. Effects: sets the enum.
 * Failure: returns false. Boundary: CLI grammar. */
static int graph_parse_output_mode(const char *text, int allow_json, yvex_graph_report_mode *mode) {
    size_t count = allow_json ? 5u : 3u;
    size_t index;

    if (!text || !mode)
        return 0;
    for (index = 0; index < count; ++index) {
        if (strcmp(text, graph_output_names[index]) == 0) {
            *mode = graph_output_modes[index];
            return 1;
        }
    }
    return 0;
}
/* Purpose: Bind one option.
 * Inputs: schema, argv, and cursor. Effects: updates request and cursor.
 * Failure: typed refusal. Boundary: no artifact or runtime access. */
static int graph_option_bind(const graph_option_spec *spec, unsigned int mode, int argc,
                             char **argv, int *index, yvex_graph_args *out,
                             graph_parse_state *state, yvex_error *err) {
    unsigned char *field = spec->offset == NO_FIELD ? NULL : (unsigned char *)out + spec->offset;
    const char *value;

    if (spec->kind == GRAPH_OPTION_FLAG)
        *(int *)field = 1;
    else if (spec->kind == GRAPH_OPTION_AUDIT || spec->kind == GRAPH_OPTION_JSON) {
        out->render_mode = spec->kind == GRAPH_OPTION_JSON ? YVEX_GRAPH_REPORT_MODE_JSON
                                                           : YVEX_GRAPH_REPORT_MODE_AUDIT;
    } else {
        if (*index + 1 >= argc) {
            graph_arg_error(err, spec->error);
            return 0;
        }
        value = argv[++*index];
        if (spec->kind == GRAPH_OPTION_TEXT)
            *(const char **)field = value;
        else if (spec->kind == GRAPH_OPTION_OUTPUT) {
            if (!graph_parse_output_mode(value, mode == GRAPH_PARSE_ATTENTION, &out->render_mode)) {
                graph_arg_errorf(err, "yvex: unsupported graph output mode: %s", value);
                return 0;
            }
        } else if (!(spec->kind == GRAPH_OPTION_POSITIVE
                         ? parse_positive_ull(value, (unsigned long long *)field)
                         : parse_ull_allow_zero(value, (unsigned long long *)field))) {
            graph_arg_error(err, spec->error);
            return 0;
        }
        if (spec->maximum && *(unsigned long long *)field > spec->maximum) {
            graph_arg_error(err, "yvex: requires 1 <= --layers <= 16");
            return 0;
        }
    }
    if (spec->companion_offset != NO_FIELD)
        *(int *)((unsigned char *)out + spec->companion_offset) = 1;
    state->seen |= spec->seen;
    return 1;
}
/* Purpose: Parse attention execution.
 * Inputs: argv and output. Effects: publishes options.
 * Failure: typed refusal. Boundary: no artifact or graph execution. */
static int graph_attention_value_valid(const char *value, const char *const *values,
                                       size_t count) {
    size_t index;

    for (index = 0; value && index < count; ++index)
        if (strcmp(value, values[index]) == 0)
            return 1;
    return 0;
}

static int graph_attention_action_executes(yvex_graph_attention_action action);

typedef struct {
    const char *owner, *operation;
    yvex_graph_attention_action action;
} graph_attention_action_spec;

static const graph_attention_action_spec graph_attention_action_specs[] = {
    {"prepare", NULL, YVEX_GRAPH_ATTENTION_ACTION_PREPARE}, {"describe", NULL, YVEX_GRAPH_ATTENTION_ACTION_DESCRIBE},
{"capabilities", NULL, YVEX_GRAPH_ATTENTION_ACTION_CAPABILITIES}, {"plan", NULL, YVEX_GRAPH_ATTENTION_ACTION_PLAN},
    {"execute", NULL, YVEX_GRAPH_ATTENTION_ACTION_EXECUTE}, {"compare", NULL, YVEX_GRAPH_ATTENTION_ACTION_COMPARE},
{"state", "inspect", YVEX_GRAPH_ATTENTION_ACTION_STATE_INSPECT},
{"state", "validate", YVEX_GRAPH_ATTENTION_ACTION_STATE_VALIDATE},
    {"state", "exercise", YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE},
{"residency", "inspect", YVEX_GRAPH_ATTENTION_ACTION_RESIDENCY_INSPECT},
    {"capture", NULL, YVEX_GRAPH_ATTENTION_ACTION_CAPTURE}, {"replay", NULL, YVEX_GRAPH_ATTENTION_ACTION_REPLAY},
{"cuda-graph", "list", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_LIST},
{"cuda-graph", "inspect", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_INSPECT},
    {"cuda-graph", "warmup", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_WARMUP},
{"cuda-graph", "update", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_UPDATE},
    {"cuda-graph", "invalidate", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_INVALIDATE},
{"cuda-graph", "release", YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_RELEASE},
    {"trace", NULL, YVEX_GRAPH_ATTENTION_ACTION_TRACE}, {"profile", NULL, YVEX_GRAPH_ATTENTION_ACTION_PROFILE},
{"benchmark", NULL, YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK},
{"benchmark", "compare", YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE},
    {"qualify", NULL, YVEX_GRAPH_ATTENTION_ACTION_QUALIFY},
};
static const char *const graph_attention_phase_values[] = {
    "prefill", "decode", "mixed", "verify"};
static const char *const graph_attention_mode_values[] = {
    "eager", "piecewise", "full", "auto"};
static const char *const graph_attention_scope_values[] = {
    "core", "envelope", "release-attention-set"};
static const char *const graph_attention_trace_values[] = {
    "none", "summary", "stages", "full"};
static const char *const graph_attention_progress_values[] = {
    "auto", "plain", "off"};
/* Purpose: reject a parsed control whose production descriptor has no owner yet. */
static int graph_attention_control_refuse(yvex_error *err, const char *control,
                                          const char *boundary) {
    yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "graph",
                    "yvex: %s is unavailable until %s owns it", control, boundary);
    return YVEX_ERR_UNSUPPORTED;
}
/* Purpose: prevent accepted operator controls from becoming silently ignored state.
 * Inputs: fully parsed attention request. Effects: none.
 * Failure: names the exact missing production boundary.
 * Boundary: parser refusal never fabricates a runtime descriptor field. */
static int graph_attention_controls_validate(const yvex_graph_args *out, yvex_error *err) {
    const int executes = graph_attention_action_executes(out->attention.action);

    if (out->attention.regression_basis_points != ULLONG_MAX &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE)
        return graph_arg_error(
            err, "yvex: regression thresholds require graph attention benchmark compare");
    if (out->attention.input_class && strcmp(out->attention.input_class, "tensor-file") != 0)
        return graph_arg_errorf(err, "yvex: unsupported attention input class: %s",
                                out->attention.input_class);
    if (!!out->attention.input_class != !!out->attention.input_file)
        return graph_arg_error(
            err, "yvex: --input tensor-file and --input-file are required together");
    if (out->attention.input_class &&
        (out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_EXECUTE ||
         strcmp(out->attention.phase, "prefill") != 0 ||
         strcmp(out->attention.mode, "eager") != 0 ||
         strcmp(out->attention.coverage, "full") != 0))
        return graph_arg_error(
            err, "yvex: tensor-file input requires graph attention execute --phase prefill --mode eager --scope full");
    if ((out->attention.chunk_tokens || out->attention.context_capacity) &&
        !out->attention.input_class)
        return graph_arg_error(
            err, "yvex: prefill chunk controls require --input tensor-file");
    if (out->attention.layer_seen && (out->attention.layer_start_seen ||
                                     out->attention.layer_count_seen))
        return graph_arg_error(err, "yvex: --layer conflicts with --layer-start/--layer-count");
    if (out->attention.layer_start_seen != out->attention.layer_count_seen)
        return graph_arg_error(err, "yvex: --layer-start and --layer-count are required together");
    if (out->attention.layer_count_seen && out->attention.layer_count != 1ull)
        return graph_attention_control_refuse(err, "multi-layer ranges",
                                              "range-aware graph publication");
    if ((out->attention.layer_seen || out->attention.layer_start_seen) && !executes &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_PLAN)
        return graph_attention_control_refuse(err, "layer selection",
                                              "an executable attention request");
    if (out->attention.attention_class &&
        (out->attention.layer_seen || out->attention.layer_start_seen))
        return graph_arg_error(err, "yvex: --class conflicts with explicit layer selection");
    if (out->attention.attention_class &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK)
        return graph_attention_control_refuse(err, "--class",
                                              "graph attention state exercise or benchmark");
    if ((out->attention.position_seen || out->attention.history_tokens_seen) &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK)
        return graph_attention_control_refuse(err, "explicit history geometry",
                                              "graph attention state exercise or benchmark");
    if (out->attention.position_seen && out->attention.history_tokens_seen &&
        out->attention.position != out->attention.history_tokens)
        return graph_arg_error(err, "yvex: --position and --history-tokens must agree");
    if (out->attention.local_capacity_seen || out->attention.compressed_capacity_seen ||
        out->attention.indexer_capacity_seen)
        return graph_attention_control_refuse(err, "explicit state capacity",
                                              "the persistent attention-state provider");
    if ((out->attention.baseline_path || out->attention.write_baseline ||
         out->attention.chart_path) &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_PROFILE &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE)
        return graph_arg_error(err,
                               "yvex: --baseline, --write-baseline, and --chart require "
                               "graph attention benchmark or profile");
    if (out->attention.write_baseline && !out->attention.baseline_path)
        return graph_arg_error(err, "yvex: --write-baseline requires --baseline FILE");
    if (out->attention.current_path &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE)
        return graph_arg_error(err, "yvex: --current requires graph attention benchmark compare");
    if (out->attention.chart_path) {
        size_t length = strlen(out->attention.chart_path);
        if (length < 5u || strcmp(out->attention.chart_path + length - 4u, ".svg") != 0)
            return graph_arg_error(err, "yvex: --chart path must end in .svg");
    }
    if (strcmp(out->attention.trace_level, "none") != 0 && !executes)
        return graph_attention_control_refuse(err, "--trace-level",
                                              "an executable attention request");
    if (out->attention.maximum_device_bytes && out->attention.backend &&
        strcmp(out->attention.backend, "cpu") == 0)
        return graph_arg_error(err, "yvex: --max-device-bytes requires a CUDA execution path");
    if (out->attention.capture_bucket &&
        (!out->attention.backend || strcmp(out->attention.backend, "cuda") != 0 ||
         strcmp(out->attention.mode, "eager") == 0))
        return graph_arg_error(
            err, "yvex: --capture-bucket requires CUDA piecewise, full, or auto mode");
    if (out->attention.maximum_device_bytes && !executes)
        return graph_attention_control_refuse(err, "--max-device-bytes",
                                              "an executing backend session");
    if (out->attention.maximum_host_bytes &&
        (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_PREPARE ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_DESCRIBE))
        return graph_attention_control_refuse(err, "--max-host-bytes",
                                              "a runtime model or execution session");
    if (out->attention.require_mode && !executes)
        return graph_attention_control_refuse(err, "--require-mode",
                                              "an executable attention request");
    return YVEX_OK;
}
/* Purpose: validate the typed attention action without opening runtime assets.
 * Inputs: action text and caller-owned enum output. Effects: writes one recognized action value.
 * Failure: unknown or null text returns false without modifying runtime state.
 * Boundary: grammar recognition performs no path resolution or capability admission. */
static int graph_attention_action_parse(int argc, char **argv,
                                        yvex_graph_attention_action *action,
                                        int *option_start)
{
    size_t index;

    for (index = 0; argc > 3 && index < sizeof(graph_attention_action_specs) /
                                           sizeof(graph_attention_action_specs[0]); ++index) {
        const graph_attention_action_spec *spec = &graph_attention_action_specs[index];
        if (strcmp(argv[3], spec->owner) == 0 &&
            ((!spec->operation && (argc == 4 || argv[4][0] == '-')) ||
             (spec->operation && argc > 4 && strcmp(argv[4], spec->operation) == 0))) {
            *action = spec->action;
            *option_start = spec->operation ? 5 : 4;
            return 1;
        }
    }
    return 0;
}
/* Purpose: classify actions that invoke production execution.
 * Inputs: typed CLI action. Effects: none.
 * Failure: unknown actions return false.
 * Boundary: classification neither opens a session nor executes graph work. */
static int graph_attention_action_executes(yvex_graph_attention_action action)
{
    return action == YVEX_GRAPH_ATTENTION_ACTION_EXECUTE ||
           action == YVEX_GRAPH_ATTENTION_ACTION_COMPARE ||
           action == YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE ||
           (action >= YVEX_GRAPH_ATTENTION_ACTION_CAPTURE &&
            action <= YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK) ||
           action == YVEX_GRAPH_ATTENTION_ACTION_QUALIFY;
}
/* Purpose: classify actions whose production contract requires a CUDA session. */
static int graph_attention_action_cuda(yvex_graph_attention_action action)
{
    return action >= YVEX_GRAPH_ATTENTION_ACTION_CAPTURE &&
           action <= YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_RELEASE;
}
/* Purpose: Parse one runtime-attention operator action.
 * Inputs: argv, selected action, and caller-owned output. Effects: publishes typed options only.
 * Failure: unsupported or contradictory grammar returns a typed refusal.
 * Boundary: no source, artifact, binding, or backend access. */
static int graph_parse_attention(int argc, char **argv, yvex_graph_args *out,
                                 yvex_error *err) {
    graph_parse_state state = {0};
    int i, option_start = 4;

    out->attention.active = 1;
    if (!graph_attention_action_parse(argc, argv, &out->attention.action, &option_start))
        return graph_arg_errorf(err, "yvex: unknown graph attention action: %s", argv[3]);
    if (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_COMPARE)
        out->attention.compare_backends = 1;
    for (i = option_start; i < argc; ++i) {
        const graph_option_spec *spec;

        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            out->help_requested = 1;
            return YVEX_OK;
        }
        spec = graph_option_find(argv[i], GRAPH_PARSE_ATTENTION);
        if (!spec) {
            graph_arg_errorf(err, "yvex: unknown graph attention option: %s", argv[i]);
            return YVEX_ERR_INVALID_ARG;
        }
        if (!graph_option_bind(spec, GRAPH_PARSE_ATTENTION, argc, argv, &i, out, &state, err))
            return YVEX_ERR_INVALID_ARG;
    }
    if (graph_attention_action_cuda(out->attention.action)) {
        if (out->attention.backend && strcmp(out->attention.backend, "cuda") != 0)
            return graph_arg_error(err, "yvex: CUDA graph actions require --backend cuda");
        out->attention.backend = "cuda";
        if (!out->attention.mode_seen) out->attention.mode = "full";
        if (strcmp(out->attention.mode, "eager") == 0)
            return graph_arg_error(
                err, "yvex: CUDA graph actions require piecewise, full, or auto mode");
    }
    if ((out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_STATE_INSPECT ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_STATE_VALIDATE ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE) &&
        !out->attention.backend)
        out->attention.backend = "cpu";
    if (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE) {
        if (!out->attention.phase_seen)
            out->attention.phase = "prefill";
        if (!out->attention.token_count_seen)
            out->attention.token_count = 4ull;
    }
    if (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_TRACE &&
        strcmp(out->attention.trace_level, "none") == 0)
        out->attention.trace_level = "summary";
    if (!out->attention.target &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE) {
        graph_arg_error(err, "yvex: graph attention action requires --target TARGET");
        return YVEX_ERR_INVALID_ARG;
    }
    if (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE &&
        (!out->attention.baseline_path || !out->attention.current_path))
        return graph_arg_error(
            err, "yvex: benchmark compare requires --baseline FILE and --current FILE");
    if (graph_attention_action_executes(out->attention.action) &&
        out->attention.action != YVEX_GRAPH_ATTENTION_ACTION_COMPARE &&
        out->attention.compare_backends == (out->attention.backend != NULL)) {
        graph_arg_error(err, out->attention.compare_backends
                                 ? "yvex: --compare-backends cannot be combined with --backend"
                                 : "yvex: graph attention execute requires --backend cpu|cuda or "
                                   "--compare-backends");
        return YVEX_ERR_INVALID_ARG;
    }
    if (out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_COMPARE &&
        out->attention.backend) {
        return graph_arg_error(err, "yvex: graph attention compare does not accept --backend");
    }
    if ((out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_CAPABILITIES ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_PLAN ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_RESIDENCY_INSPECT ||
         out->attention.action == YVEX_GRAPH_ATTENTION_ACTION_QUALIFY) &&
        !out->attention.backend) {
        return graph_arg_error(err, "yvex: graph attention action requires --backend cpu|cuda");
    }
    if (out->attention.backend && !cli_backend_name_valid(out->attention.backend)) {
        graph_arg_errorf(err, "yvex: unknown backend kind: %s", out->attention.backend);
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(out->attention.probe, "canonical") != 0) {
        graph_arg_errorf(err, "yvex: unsupported attention probe: %s", out->attention.probe);
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(out->attention.coverage, "quick") != 0 &&
        strcmp(out->attention.coverage, "full") != 0) {
        graph_arg_errorf(err, "yvex: unsupported attention scope: %s", out->attention.coverage);
        return YVEX_ERR_INVALID_ARG;
    }
    if (!graph_attention_value_valid(
            out->attention.phase, graph_attention_phase_values,
            sizeof(graph_attention_phase_values) / sizeof(graph_attention_phase_values[0])))
        return graph_arg_errorf(err, "yvex: unsupported attention phase: %s",
                                out->attention.phase);
    if (!graph_attention_value_valid(
            out->attention.mode, graph_attention_mode_values,
            sizeof(graph_attention_mode_values) / sizeof(graph_attention_mode_values[0])))
        return graph_arg_errorf(err, "yvex: unsupported attention mode: %s",
                                out->attention.mode);
    if (!graph_attention_value_valid(
            out->attention.operation_scope, graph_attention_scope_values,
            sizeof(graph_attention_scope_values) / sizeof(graph_attention_scope_values[0])))
        return graph_arg_errorf(err, "yvex: unsupported attention operation scope: %s",
                                out->attention.operation_scope);
    if (!graph_attention_value_valid(
            out->attention.trace_level, graph_attention_trace_values,
            sizeof(graph_attention_trace_values) / sizeof(graph_attention_trace_values[0])))
        return graph_arg_errorf(err, "yvex: unsupported attention trace level: %s",
                                out->attention.trace_level);
    if (!graph_attention_value_valid(
            out->attention.progress, graph_attention_progress_values,
            sizeof(graph_attention_progress_values) / sizeof(graph_attention_progress_values[0])))
        return graph_arg_errorf(err, "yvex: unsupported attention progress mode: %s",
                                out->attention.progress);
    if (out->attention.runtime_binding_path && out->attention.runtime_binding_dir)
        return graph_arg_error(err,
                               "yvex: --runtime-binding conflicts with --runtime-binding-dir");
    if (graph_attention_controls_validate(out, err) != YVEX_OK)
        return yvex_error_code(err);
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: consume one required MoE option value without opening an operator asset. */
static const char *graph_moe_value(int argc, char **argv, int *index, yvex_error *err)
{
    if (*index + 1 >= argc) {
        graph_arg_errorf(err, "yvex: graph moe option requires a value: %s", argv[*index]);
        return NULL;
    }
    return argv[++*index];
}

/* Purpose: parse the bounded production MoE execution grammar.
 * Inputs: argv. Effects: fills args. Failure: typed refusal. Boundary: no runtime execution. */
static int graph_parse_moe(int argc, char **argv, yvex_graph_args *out, yvex_error *err)
{
    int index;
    if (argc < 4 || strcmp(argv[3], "execute") != 0)
        return graph_arg_error(err, "yvex: graph moe requires the execute action");
    out->moe.active = 1;
    for (index = 4; index < argc; ++index) {
        const char *flag = argv[index];
        const char *value;
        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            out->help_requested = 1;
            return YVEX_OK;
        }
        value = graph_moe_value(argc, argv, &index, err);
        if (!value) return YVEX_ERR_INVALID_ARG;
        if (strcmp(flag, "--target") == 0) out->moe.target = value;
        else if (strcmp(flag, "--artifact") == 0) out->moe.artifact_path = value;
        else if (strcmp(flag, "--runtime-binding") == 0) out->moe.runtime_binding_path = value;
        else if (strcmp(flag, "--backend") == 0) out->moe.backend = value;
        else if (strcmp(flag, "--input") == 0) out->moe.input_class = value;
        else if (strcmp(flag, "--input-file") == 0) out->moe.input_file = value;
        else if (strcmp(flag, "--scope") == 0) out->moe.coverage = value;
        else if (strcmp(flag, "--progress") == 0) out->moe.progress = value;
        else if (strcmp(flag, "--max-host-bytes") == 0) {
            if (!parse_positive_ull(value, &out->moe.maximum_host_bytes))
                return graph_arg_error(err, "yvex: --max-host-bytes requires a positive integer");
        } else if (strcmp(flag, "--max-device-bytes") == 0) {
            if (!parse_positive_ull(value, &out->moe.maximum_device_bytes))
                return graph_arg_error(err, "yvex: --max-device-bytes requires a positive integer");
        } else if (strcmp(flag, "--output") == 0) {
            if (!graph_parse_output_mode(value, 1, &out->render_mode))
                return graph_arg_errorf(err, "yvex: unsupported graph output mode: %s", value);
        } else {
            return graph_arg_errorf(err, "yvex: unknown graph moe option: %s", flag);
        }
    }
    if (!out->moe.target || !out->moe.artifact_path || !out->moe.runtime_binding_path ||
        !out->moe.backend || !out->moe.input_class || !out->moe.input_file)
        return graph_arg_error(
            err, "yvex: graph moe execute requires target, artifact, runtime binding, backend, "
                 "and tensor-file input");
    if (!cli_backend_name_valid(out->moe.backend))
        return graph_arg_errorf(err, "yvex: unknown backend kind: %s", out->moe.backend);
    if (strcmp(out->moe.input_class, "tensor-file") != 0)
        return graph_arg_error(err, "yvex: graph moe execute requires --input tensor-file");
    if (strcmp(out->moe.coverage, "full") != 0)
        return graph_arg_error(err, "yvex: graph moe execute supports only --scope full");
    if (strcmp(out->moe.progress, "off") != 0)
        return graph_arg_error(err, "yvex: graph moe execute requires --progress off");
    if (out->moe.maximum_device_bytes && strcmp(out->moe.backend, "cpu") == 0)
        return graph_arg_error(err, "yvex: --max-device-bytes requires backend cuda");
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: parse the bounded production transformer grammar.
 * Inputs: argv and caller output. Effects: fills typed arguments only after validation.
 * Failure: typed parser refusal. Boundary: never opens input or executes runtime work. */
static int graph_parse_transformer(int argc, char **argv, yvex_graph_args *out,
                                   yvex_error *err)
{
    int index;
    if (argc < 4 || (strcmp(argv[3], "execute") != 0 &&
                     strcmp(argv[3], "decode") != 0 &&
                     strcmp(argv[3], "logits") != 0 &&
                     strcmp(argv[3], "sample") != 0))
        return graph_arg_error(err,
                               "yvex: graph transformer requires execute, decode, logits, or sample");
    out->transformer.active = 1;
    out->transformer.decode = strcmp(argv[3], "decode") == 0;
    out->transformer.logits = strcmp(argv[3], "logits") == 0;
    out->transformer.sample = strcmp(argv[3], "sample") == 0;
    for (index = 4; index < argc; ++index) {
        const char *flag = argv[index];
        const char *value;
        if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            out->help_requested = 1;
            return YVEX_OK;
        }
        value = graph_moe_value(argc, argv, &index, err);
        if (!value) return YVEX_ERR_INVALID_ARG;
        if (strcmp(flag, "--target") == 0) out->transformer.target = value;
        else if (strcmp(flag, "--artifact") == 0) out->transformer.artifact_path = value;
        else if (strcmp(flag, "--runtime-binding") == 0)
            out->transformer.runtime_binding_path = value;
        else if (strcmp(flag, "--backend") == 0) out->transformer.backend = value;
        else if (strcmp(flag, "--phase") == 0) out->transformer.phase = value;
        else if (strcmp(flag, "--input") == 0) out->transformer.input_class = value;
        else if (strcmp(flag, "--input-file") == 0) out->transformer.input_file = value;
        else if (strcmp(flag, "--progress") == 0) out->transformer.progress = value;
        else if (strcmp(flag, "--strategy") == 0) out->transformer.strategy = value;
        else if (strcmp(flag, "--temperature") == 0) {
            if (!graph_parse_finite_double(value, &out->transformer.temperature))
                return graph_arg_error(err, "yvex: --temperature requires a finite decimal");
        } else if (strcmp(flag, "--top-k") == 0) {
            if (!graph_parse_nonnegative_u64(value, &out->transformer.top_k))
                return graph_arg_error(err, "yvex: --top-k requires a nonnegative integer");
        } else if (strcmp(flag, "--top-p") == 0) {
            if (!graph_parse_finite_double(value, &out->transformer.top_p))
                return graph_arg_error(err, "yvex: --top-p requires a finite decimal");
        } else if (strcmp(flag, "--min-p") == 0) {
            if (!graph_parse_finite_double(value, &out->transformer.min_p))
                return graph_arg_error(err, "yvex: --min-p requires a finite decimal");
        } else if (strcmp(flag, "--typical-p") == 0) {
            if (!graph_parse_finite_double(value, &out->transformer.typical_p))
                return graph_arg_error(err, "yvex: --typical-p requires a finite decimal");
        } else if (strcmp(flag, "--seed") == 0) {
            if (!graph_parse_nonnegative_u64(value, &out->transformer.seed))
                return graph_arg_error(err, "yvex: --seed requires a nonnegative integer");
            out->transformer.seed_seen = 1;
        }
        else if (strcmp(flag, "--chunk-tokens") == 0) {
            if (!parse_positive_ull(value, &out->transformer.chunk_tokens))
                return graph_arg_error(err, "yvex: --chunk-tokens requires a positive integer");
        } else if (strcmp(flag, "--prefill-tokens") == 0) {
            if (!parse_positive_ull(value, &out->transformer.prefill_tokens))
                return graph_arg_error(err,
                                       "yvex: --prefill-tokens requires a positive integer");
        } else if (strcmp(flag, "--prefill-chunk-tokens") == 0) {
            if (!parse_positive_ull(value, &out->transformer.prefill_chunk_tokens))
                return graph_arg_error(
                    err, "yvex: --prefill-chunk-tokens requires a positive integer");
        } else if (strcmp(flag, "--context-capacity") == 0) {
            if (!parse_positive_ull(value, &out->transformer.context_capacity))
                return graph_arg_error(err,
                                       "yvex: --context-capacity requires a positive integer");
        } else if (strcmp(flag, "--max-host-bytes") == 0) {
            if (!parse_positive_ull(value, &out->transformer.maximum_host_bytes))
                return graph_arg_error(err, "yvex: --max-host-bytes requires a positive integer");
        } else if (strcmp(flag, "--max-device-bytes") == 0) {
            if (!parse_positive_ull(value, &out->transformer.maximum_device_bytes))
                return graph_arg_error(err,
                                       "yvex: --max-device-bytes requires a positive integer");
        } else if (strcmp(flag, "--output") == 0) {
            if (!graph_parse_output_mode(value, 1, &out->render_mode))
                return graph_arg_errorf(err, "yvex: unsupported graph output mode: %s", value);
        } else {
            return graph_arg_errorf(err, "yvex: unknown graph transformer option: %s", flag);
        }
    }
    if (!out->transformer.target || !out->transformer.artifact_path ||
        !out->transformer.runtime_binding_path || !out->transformer.backend ||
        !out->transformer.input_file ||
        !out->transformer.context_capacity)
        return graph_arg_error(
            err, (out->transformer.decode || out->transformer.logits || out->transformer.sample)
                     ? "yvex: graph transformer decode/logits/sample requires target, artifact, runtime binding, "
                       "backend, token input, prefill split, and context capacity"
                     : "yvex: graph transformer execute requires target, artifact, runtime binding, "
                       "backend, token input, chunk tokens, and context capacity");
    if (!cli_backend_name_valid(out->transformer.backend))
        return graph_arg_errorf(err, "yvex: unknown backend kind: %s", out->transformer.backend);
    if (strcmp(out->transformer.phase, "prefill") != 0)
        return graph_arg_error(err, "yvex: graph transformer supports only --phase prefill");
    if (strcmp(out->transformer.input_class, "token-ids") != 0)
        return graph_arg_error(err, "yvex: graph transformer requires --input token-ids");
    if (strcmp(out->transformer.progress, "off") != 0)
        return graph_arg_error(err, "yvex: graph transformer requires --progress off");
    if ((!out->transformer.decode && !out->transformer.logits && !out->transformer.sample &&
         !out->transformer.chunk_tokens) ||
        ((out->transformer.decode || out->transformer.logits || out->transformer.sample) &&
         (!out->transformer.prefill_tokens ||
          !out->transformer.prefill_chunk_tokens)))
        return graph_arg_error(
            err, (out->transformer.decode || out->transformer.logits || out->transformer.sample)
                     ? "yvex: graph transformer decode/logits/sample requires prefill tokens and prefill chunk tokens"
                     : "yvex: graph transformer execute requires chunk tokens");
    if (out->transformer.sample) {
        int stochastic = strcmp(out->transformer.strategy, "stochastic") == 0;
        int greedy = strcmp(out->transformer.strategy, "greedy") == 0;
        if (!greedy && !stochastic)
            return graph_arg_error(err, "yvex: --strategy requires greedy or stochastic");
        if (greedy && (out->transformer.seed_seen || out->transformer.temperature != 1.0 ||
                       out->transformer.top_k || out->transformer.top_p != 1.0 ||
                       out->transformer.min_p != 0.0 || out->transformer.typical_p != 1.0))
            return graph_arg_error(err, "yvex: greedy strategy requires neutral filters and no seed");
        if (stochastic && (!out->transformer.seed_seen || out->transformer.temperature <= 0.0 ||
                           out->transformer.top_p <= 0.0 || out->transformer.top_p > 1.0 ||
                           out->transformer.min_p < 0.0 || out->transformer.min_p > 1.0 ||
                           out->transformer.typical_p <= 0.0 ||
                           out->transformer.typical_p > 1.0))
            return graph_arg_error(err, "yvex: stochastic strategy parameters are invalid");
    } else if (strcmp(out->transformer.strategy, "greedy") != 0 ||
               out->transformer.seed_seen || out->transformer.temperature != 1.0 ||
               out->transformer.top_k || out->transformer.top_p != 1.0 ||
               out->transformer.min_p != 0.0 || out->transformer.typical_p != 1.0)
        return graph_arg_error(err, "yvex: sampling options require graph transformer sample");
    if (out->transformer.maximum_device_bytes &&
        strcmp(out->transformer.backend, "cpu") == 0)
        return graph_arg_error(err, "yvex: --max-device-bytes requires backend cuda");
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: Parse graph argv.
 * Inputs: argv and output. Effects: publishes request.
 * Failure: typed refusal. Boundary: CLI grammar. */
int yvex_graph_args_parse(int argc, char **argv, yvex_graph_args *out, yvex_error *err) {
    if (!out) {
        graph_arg_error(err, "graph parser requires output");
        return YVEX_ERR_INVALID_ARG;
    }
    graph_args_defaults(out);
    if (argc < 3) {
        out->help_requested = 1;
        out->help_exit_code = 2;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (strcmp(argv[2], "--help") == 0 || strcmp(argv[2], "-h") == 0) {
        out->help_requested = 1;
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (strcmp(argv[2], "attention") == 0) {
        if (argc == 4 && (strcmp(argv[3], "--help") == 0 || strcmp(argv[3], "-h") == 0)) {
            out->help_requested = 1;
            yvex_error_clear(err);
            return YVEX_OK;
        }
        if (argc < 4) {
            graph_arg_error(err, "yvex: graph attention requires an action");
            return YVEX_ERR_INVALID_ARG;
        }
        return graph_parse_attention(argc, argv, out, err);
    }
    if (strcmp(argv[2], "moe") == 0)
        return graph_parse_moe(argc, argv, out, err);
    if (strcmp(argv[2], "transformer") == 0)
        return graph_parse_transformer(argc, argv, out, err);

    return graph_arg_errorf(err, "yvex: unknown graph namespace: %s", argv[2]);
}
