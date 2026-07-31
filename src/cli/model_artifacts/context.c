/*
 * Render bounded context diagnostics for the engineering command lane. The reported plans do not
 * inspect or mutate the hosted runtime.
 */
#include "src/cli/model_artifacts/private.h"

#include <string.h>

static const char *const literal_pair_0[] = { "context report:",
    "  report-only boundary for model/requested/active context, token counts, chunking policy, overflow "
        "behavior, prefill boundary, decode position policy, attention dependency, KV dependency, and runtime "
        "blockers."
};

static const char *const literal_pair_1[] = { "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_pair_2[] = { "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_pair_3[] = { "kv_capacity_required: true",
    "kv_capacity_status: planned"};

static const char *const literal_pair_4[] = { "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_pair_5[] = { "next: V010.CONTEXT.*",
    "boundary: report-only, no runtime execution"};

static const char *const literal_lines_0[] = { "context_class_status: report-only",
    "context_stage: report-only",
    "runtime_claim: unsupported",
    "generation: unsupported-full-model",
    "benchmark_status: not-measured"};

static const char *const literal_lines_1[] = { "model_max_context: unknown",
    "model_max_context_source: source-manifest-planned",
    "model_max_context_status: unsupported-source-only"};

static const char *const literal_lines_2[] = { "active_context: unsupported",
    "active_context_source: source-only-target",
    "active_context_status: unsupported",
    "token_input_status: not-performed-source-only-target",
    "token_count: 0",
    "prompt_token_count: 0",
    "prefill_token_count: 0",
    "generated_token_count: 0",
    "token_range_start: none",
    "token_range_end: none",
    "chunking_required: false",
    "chunk_size: not-requested",
    "chunk_count: 0",
    "last_chunk_size: 0",
    "chunking_policy: unsupported-source-only",
    "chunking_status: unsupported",
    "prefill_boundary_status: unsupported",
    "prefill_position_start: unknown",
    "prefill_position_end: unknown",
    "prefill_context_ready: false",
    "full_transformer_prefill_ready: false",
    "decode_position_policy: unsupported-source-only",
    "decode_start_position: unknown",
    "decode_position_status: unsupported",
    "decode_context_ready: false",
    "decode_ready: false",
    "overflow_policy: unsupported-source-only",
    "context_overflow: unsupported",
    "overflow_check_status: unsupported",
    "overflow_stop_reason: unsupported",
    "overflow_mutates_state: false",
    "attention_dependency_status: unsupported-source-only",
    "attention_context_ready: false",
    "attention_position_policy: planned",
    "rope_context_status: planned",
    "mask_context_status: planned",
    "kv_dependency_status: unsupported-source-only",
    "kv_capacity_required: true",
    "kv_capacity_status: planned",
    "kv_context_positions: unknown",
    "kv_context_ready: false",
    "real_attention_kv_ready: false",
    "generation_context_status: unsupported-full-model",
    "bounded_generation_context_policy: diagnostic-only-not-run",
    "full_generation_context_ready: false",
    "generation_ready: false",
    "context_blockers: source-only target has no YVEX-produced GGUF tensor inventory; GLM context mapping "
        "planned; real transformer prefill unsupported",
    "next_required_rows: MOE.CLASS.0,source-manifest-context-profile,YVEX-produced-GGUF,real-transformer-"
        "prefill,real-decode,GEN.DEEPSEEK.0",
    "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_lines_3[] = { "model_max_context: unknown",
    "model_max_context_source: unsupported-family",
    "model_max_context_status: unsupported",
    "requested_context: not-requested",
    "requested_context_source: none",
    "requested_context_status: not-requested",
    "active_context: unsupported",
    "active_context_source: unsupported-family",
    "active_context_status: unsupported",
    "token_input_status: not-performed",
    "token_count: 0",
    "prompt_token_count: 0",
    "prefill_token_count: 0",
    "generated_token_count: 0",
    "token_range_start: none",
    "token_range_end: none",
    "chunking_required: false",
    "chunk_size: not-requested",
    "chunk_count: 0",
    "last_chunk_size: 0",
    "chunking_policy: unsupported-family",
    "chunking_status: unsupported",
    "prefill_boundary_status: unsupported",
    "prefill_position_start: unknown",
    "prefill_position_end: unknown",
    "prefill_context_ready: false",
    "full_transformer_prefill_ready: false",
    "decode_position_policy: unsupported-family",
    "decode_start_position: unknown",
    "decode_position_status: unsupported",
    "decode_context_ready: false",
    "decode_ready: false",
    "overflow_policy: unsupported-family",
    "context_overflow: unsupported",
    "overflow_check_status: unsupported",
    "overflow_stop_reason: unsupported",
    "overflow_mutates_state: false",
    "attention_dependency_status: unsupported-family",
    "attention_context_ready: false",
    "attention_position_policy: unsupported",
    "rope_context_status: unsupported",
    "mask_context_status: unsupported",
    "kv_dependency_status: unsupported-family",
    "kv_capacity_required: unknown",
    "kv_capacity_status: unsupported",
    "kv_context_positions: unknown",
    "kv_context_ready: false",
    "real_attention_kv_ready: false",
    "generation_context_status: unsupported-full-model",
    "bounded_generation_context_policy: diagnostic-only-not-run",
    "full_generation_context_ready: false",
    "generation_ready: false"};

static const char *const literal_lines_4[] = {
    "next_required_rows: CONTEXT.CLASS.0,MOE.CLASS.0,real-transformer-prefill,real-decode,GEN.DEEPSEEK.0",
    "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_lines_5[] = { "prefill_context_ready: false",
    "full_transformer_prefill_ready: false",
    "decode_position_policy: prefill-end-token-count"};

static const char *const literal_lines_6[] = { "decode_position_status: report-only",
    "decode_context_ready: false",
    "decode_ready: false",
    "overflow_policy: bounded-diagnostic-refuse-before-mutation"};

static const char *const literal_lines_7[] = { "attention_context_ready: false",
    "attention_position_policy: rope-or-family-specific-planned",
    "rope_context_status: planned",
    "mask_context_status: planned"};

static const char *const literal_lines_8[] = { "",
    "kv_context_ready: false",
    "real_attention_kv_ready: false",
    "generation_context_status: bounded-diagnostic-only",
    "bounded_generation_context_policy: context-limit-pre-append",
    "full_generation_context_ready: false",
    "generation_ready: false"};

static const char *const literal_lines_9[] = {
    "next_required_rows: MOE.CLASS.0,RUNTIME.KV.1,real-transformer-prefill,real-decode,real-output-head-"
        "logits,GEN.DEEPSEEK.0",
    "cleanup_attempted: true",
    "cleanup_status: pass"};

static const char *const literal_lines_10[] = { "Examples:",
    "  yvex inspect context report --model deepseek4-v4-flash-selected-embed-rmsnorm --family deepseek --backend "
        "cpu --tokens 0,1,2,3",
    "  yvex inspect context report --model deepseek4-v4-flash-selected-embed-rmsnorm --family deepseek --backend "
        "cpu --tokens 0,1,2,3 --chunk-size 2",
    "  yvex inspect context report --model deepseek4-v4-flash-selected-embed-rmsnorm --family deepseek --backend "
        "cpu --tokens 0,1,2,3 --context-length 2\n"
};

static const char *const literal_lines_11[] = {
    "  It reports bounded diagnostic context behavior separately from full model context readiness.",
    "  It does not run full transformer prefill, does not execute real decode, does not write real "
        "attention-backed KV, does not generate, does not evaluate, does not benchmark, and does not report "
        "throughput.",
    "Boundary: no long-context runtime support, no context extension support, no full model execution, no "
        "DeepSeek generation, no provider generation, no streaming generation, no eval, no benchmark."
};

typedef struct {
    const char *model;
    const char *backend;
    const char *family;
    const char *registry_path;
    const char *tokens_text;
    unsigned long long context_length;
    unsigned long long chunk_size;
    int context_length_seen;
    int chunk_size_seen;
    int include_attention;
    int include_kv;
    int include_prefill;
    int include_decode;
    int include_blockers;
    yvex_models_output_mode output_mode;
} yvex_cli_context_options;

#define CONTEXT_OPTION_BOOL(key_name, member) \
    {key_name, YVEX_CLI_FIELD_BOOL, offsetof(yvex_cli_context_options, member), NULL}
static const yvex_cli_field_spec context_audit_option_fields[] = {
    CONTEXT_OPTION_BOOL("report_options.include_attention", include_attention),
    CONTEXT_OPTION_BOOL("report_options.include_kv", include_kv),
    CONTEXT_OPTION_BOOL("report_options.include_prefill", include_prefill),
    CONTEXT_OPTION_BOOL("report_options.include_decode", include_decode),
    CONTEXT_OPTION_BOOL("report_options.include_blockers", include_blockers),
};
#undef CONTEXT_OPTION_BOOL

static const yvex_models_option_spec context_option_specs[] = {
    {"--model", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_context_options, model)},
    {"--backend", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_context_options, backend)},
    {"--family", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_context_options, family)},
    {"--registry", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_context_options, registry_path)},
    {"--tokens", YVEX_MODELS_OPTION_TEXT, offsetof(yvex_cli_context_options, tokens_text)},
    {"--include-attention", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_context_options, include_attention)},
    {"--include-kv", YVEX_MODELS_OPTION_FLAG, offsetof(yvex_cli_context_options, include_kv)},
    {"--include-prefill", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_context_options, include_prefill)},
    {"--include-decode", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_context_options, include_decode)},
    {"--include-blockers", YVEX_MODELS_OPTION_FLAG,
     offsetof(yvex_cli_context_options, include_blockers)},
    {"--output", YVEX_MODELS_OPTION_OUTPUT, offsetof(yvex_cli_context_options, output_mode)},
};

static int parse_context_options(int arg_count,
                                 char **args,
                                 yvex_cli_context_options *options)
{
    int i;

    if (!options) return 2;
    memset(options, 0, sizeof(*options));
    options->backend = "cpu";
    options->family = "auto";
    options->output_mode = YVEX_MODELS_OUTPUT_NORMAL;

    if (arg_count >= 3 && (strcmp(args[2], "--help") == 0 || strcmp(args[2], "-h") == 0)) {
        yvex_model_artifacts_surface_context_help(stdout);
        return 1;
    }
    if (arg_count < 3 || strcmp(args[2], "report") != 0) {
        yvex_cli_out_writef(stderr, "yvex: context requires report\n");
        yvex_cli_out_writef(stderr,
            "usage: yvex inspect context report --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] [--backend "
                "cpu|cuda] [options]\n");
        return 2;
    }

    for (i = 3; i < arg_count; ++i) {
        const char *flag = args[i];
        const char *value = NULL;
        int handled = 0;
        int rc = parse_models_bound_option("context", arg_count, args, &i,
                                           options, context_option_specs,
                                           sizeof(context_option_specs) /
                                               sizeof(context_option_specs[0]),
                                           &handled);
        if (rc != 0) return rc;
        if (handled) {
            if (strcmp(flag, "--backend") == 0 &&
                strcmp(options->backend, "cpu") != 0 &&
                strcmp(options->backend, "cuda") != 0) {
                yvex_cli_out_writef(stderr,
                                    "yvex: context --backend must be cpu or cuda\n");
                return 2;
            }
            continue;
        }
        if (strcmp(flag, "--context-length") == 0 ||
            strcmp(flag, "--chunk-size") == 0) {
            unsigned long long *destination = strcmp(flag, "--context-length") == 0
                                                   ? &options->context_length
                                                   : &options->chunk_size;
            rc = parse_models_value_option("context", flag, arg_count,
                                           args, &i, &value);
            if (rc != 0) return rc;
            if (!parse_positive_ull(value, destination)) {
                yvex_cli_out_writef(stderr,
                                    "yvex: context %s requires a positive integer\n",
                                    flag);
                return 2;
            }
            if (strcmp(flag, "--context-length") == 0) options->context_length_seen = 1;
            else options->chunk_size_seen = 1;
        } else if (strcmp(flag, "--audit") == 0) {
            options->output_mode = YVEX_MODELS_OUTPUT_AUDIT;
        } else if (strcmp(flag, "--help") == 0 || strcmp(flag, "-h") == 0) {
            yvex_model_artifacts_surface_context_help(stdout);
            return 1;
        } else {
            yvex_cli_out_writef(stderr, "yvex: unknown context option: %s\n", flag);
            return 2;
        }
    }
    if (!options->model || !options->model[0]) {
        yvex_cli_out_writef(stderr, "yvex: context report requires --model FILE_OR_ALIAS\n");
        return 2;
    }
    return 0;
}

static void context_print_phases(const char *family_status,
                                 const char *attention_status,
                                 const char *kv_status,
                                 const char *context_status,
                                 const char *failure_phase)
{
    static const char *const phases[] = {
        "preflight",
        "resolve-model",
        "resolve-family",
        "load-family-runtime",
        "load-attention-class",
        "load-kv-class",
        "context-profile",
        "model-context",
        "requested-context",
        "active-context",
        "token-input",
        "chunking-policy",
        "prefill-boundary",
        "decode-position-policy",
        "overflow-policy",
        "runtime-blockers",
        "complete",
        "failed",
        "cleanup"
    };
    unsigned int i;
    int failed_seen = 0;

    for (i = 0; i < sizeof(phases) / sizeof(phases[0]); ++i) {
        const char *status = "pass";
        if (failure_phase && strcmp(failure_phase, phases[i]) == 0) {
            status = "failed";
            failed_seen = 1;
        } else if (failed_seen && strcmp(phases[i], "cleanup") != 0) {
            status = "blocked";
        } else if (strcmp(phases[i], "load-family-runtime") == 0) {
            status = family_status ? family_status : "unknown";
        } else if (strcmp(phases[i], "load-attention-class") == 0) {
            status = attention_status ? attention_status : "unknown";
        } else if (strcmp(phases[i], "load-kv-class") == 0) {
            status = kv_status ? kv_status : "unknown";
        } else if (strcmp(phases[i], "context-profile") == 0) {
            status = context_status ? context_status : "planned";
        } else if (strcmp(phases[i], "prefill-boundary") == 0 ||
                   strcmp(phases[i], "decode-position-policy") == 0 ||
                   strcmp(phases[i], "runtime-blockers") == 0) {
            status = "blocked";
        } else if (strcmp(phases[i], "failed") == 0 && !failure_phase) {
            status = "unsupported";
        }
        model_phase_print("context_phase", i, phases[i], status, "unknown");
    }
}

static unsigned long long context_metadata_max_context(yvex_model_context *ctx,
                                                       const char **source)
{
    static const char *const keys[] = {
        "llama.context_length",
        "deepseek.context_length",
        "qwen.context_length",
        "glm.context_length",
        "general.context_length"
    };
    unsigned int i;
    unsigned long long value = 0ull;

    if (source) *source = "unknown";
    if (ctx && ctx->model) {
        value = yvex_model_context_length(ctx->model);
        if (value > 0ull) {
            if (source) *source = "model-descriptor";
            return value;
        }
    }
    for (i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        const yvex_gguf_value *meta = ctx && ctx->gguf ? yvex_gguf_metadata_find(ctx->gguf, keys[i]) : NULL;
        if (meta && yvex_gguf_value_as_u64(meta, &value) == YVEX_OK && value > 0ull) {
            if (source) *source = keys[i];
            return value;
        }
    }
    return 0ull;
}

static const char *context_status_from_collections(const yvex_fullmodel_collections *collections,
                                                   int selected_target)
{
    if (selected_target) return "partial";
    if (collections &&
        collections->has_token_embedding &&
        collections->has_attention_q &&
        collections->has_attention_k &&
        collections->has_attention_v &&
        collections->has_attention_out) {
        return "complete";
    }
    return "partial";
}

static void context_print_common_header(const yvex_cli_context_options *options,
                                        const char *status,
                                        const char *resolved_path,
                                        const char *target_id,
                                        const char *target_class,
                                        const char *family,
                                        const char *detected,
                                        const char *family_status,
                                        const char *attention_status,
                                        const char *kv_status)
{
    yvex_cli_out_writef(stdout, "context: report\n");
    yvex_cli_out_writef(stdout, "status: %s\n", status ? status : "context-report");
    yvex_cli_out_writef(stdout, "model: %s\n", options && options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n", resolved_path ? resolved_path : "");
    yvex_cli_out_writef(stdout, "target_id: %s\n", target_id ? target_id : "unknown");
    yvex_cli_out_writef(stdout, "target_class: %s\n", target_class ? target_class : "unknown");
    yvex_cli_out_writef(stdout, "backend: %s\n", options && options->backend ? options->backend : "cpu");
    yvex_cli_out_writef(stdout, "family: %s\n", family ? family : "unknown");
    yvex_cli_out_writef(stdout, "family_detected: %s\n", detected ? detected : "unknown");
    yvex_cli_out_writef(stdout, "family_requested: %s\n", model_requested_family(options ? options->family : NULL));
    yvex_cli_out_writef(stdout, "family_runtime_status: %s\n", family_status ? family_status : "unknown");
    yvex_cli_out_writef(stdout, "attention_class_status: %s\n", attention_status ? attention_status : "unknown");
    yvex_cli_out_writef(stdout, "kv_class_status: %s\n", kv_status ? kv_status : "unknown");
    yvex_cli_out_lines(stdout, literal_lines_0, sizeof(literal_lines_0) / sizeof(literal_lines_0[0]));
}

static void context_print_unsupported_source_only(const yvex_cli_context_options *options)
{
    const char *family = strcmp(model_requested_family(options ? options->family : NULL), "auto") == 0
                             ? "glm"
                             : model_requested_family(options ? options->family : NULL);
    context_print_common_header(options,
                                "context-report-unsupported",
                                "source-only-target",
                                "glm-5.2-official-safetensors",
                                "official-source-huge-model",
                                family,
                                "glm",
                                "unsupported",
                                "unsupported",
                                "unsupported");
    yvex_cli_out_lines(stdout, literal_lines_1, sizeof(literal_lines_1) / sizeof(literal_lines_1[0]));
    yvex_cli_out_writef(stdout, "requested_context: %s",
        options && options->context_length_seen ? "" : "not-requested");
    if (options && options->context_length_seen) yvex_cli_out_writef(stdout, "%llu", options->context_length);
    yvex_cli_out_writef(stdout, "\n");
    yvex_cli_out_writef(stdout, "requested_context_source: %s\n",
        options && options->context_length_seen ? "operator-request" : "none");
    yvex_cli_out_writef(stdout, "requested_context_status: %s\n",
        options && options->context_length_seen ? "reported-only" : "not-requested");
    yvex_cli_out_lines(stdout, literal_lines_2, sizeof(literal_lines_2) / sizeof(literal_lines_2[0]));
    context_print_phases("unsupported", "unsupported", "unsupported", "unsupported", "context-profile");
}

static int context_print_unsupported_family(const yvex_cli_context_options *options,
                                            const char *resolved_path,
                                            const char *target_id,
                                            const char *target_class,
                                            const char *detected,
                                            const char *reason)
{
    context_print_common_header(options,
                                "context-report-unsupported",
                                resolved_path,
                                target_id,
                                target_class,
                                model_requested_family(options ? options->family : NULL),
                                detected ? detected : "unknown",
                                "unsupported",
                                "unsupported",
                                "unsupported");
    yvex_cli_out_lines(stdout, literal_lines_3, sizeof(literal_lines_3) / sizeof(literal_lines_3[0]));
    yvex_cli_out_writef(stdout, "context_blockers: %s\n", reason ? reason : "unsupported context family");
    yvex_cli_out_lines(stdout, literal_lines_4, sizeof(literal_lines_4) / sizeof(literal_lines_4[0]));
    context_print_phases("unsupported", "unsupported", "unsupported", "unsupported", "resolve-family");
    yvex_cli_out_writef(stdout, "reason: %s\n", reason ? reason : "unsupported context family");
    return 5;
}

static void context_print_normal(const yvex_cli_context_options *options,
                                 const char *coverage_status,
                                 int selected_target)
{
    yvex_cli_out_writef(stdout, "report: context\n");
    yvex_cli_out_writef(stdout, "model: %s\n", options->model ? options->model : "");
    yvex_cli_out_writef(stdout, "family: deepseek\n");
    yvex_cli_out_writef(stdout, "status: %s\n",
                        coverage_status ? coverage_status : "report-only");
    yvex_cli_out_writef(stdout, "top_blocker: %s\n",
                        selected_target
                            ? "selected-slice context metadata incomplete"
                            : "full transformer prefill unsupported");
    yvex_cli_out_lines(stdout, literal_pair_5, sizeof(literal_pair_5) / sizeof(literal_pair_5[0]));
}

static int context_print_report(const yvex_cli_context_options *options,
                                yvex_model_ref *ref,
                                yvex_model_context *ctx,
                                const char *target_id,
                                const char *target_class,
                                yvex_arch arch,
                                const yvex_fullmodel_collections *collections,
                                int selected_target)
{
    yvex_cli_fullmodel_options family_probe;
    yvex_token_input input;
    yvex_error err;
    const char *requested = model_requested_family(options ? options->family : NULL);
    const char *detected;
    const char *source = "unknown";
    const char *coverage_status;
    unsigned long long model_max_context;
    unsigned long long active_context = 0ull;
    unsigned long long token_count = 0ull;
    unsigned long long chunk_count = 0ull;
    unsigned long long last_chunk = 0ull;
    unsigned long long decode_start = 0ull;
    int overflow_known = 0;
    int overflow = 0;
    int has_qkv;
    int rc;

    memset(&family_probe, 0, sizeof(family_probe));
    memset(&input, 0, sizeof(input));
    yvex_error_clear(&err);

    family_probe.model = options ? options->model : NULL;
    family_probe.family = options ? options->family : "auto";
    detected = fullmodel_detect_family(&family_probe, arch, target_id);
    if (!fullmodel_family_request_matches(requested, detected) ||
        strcmp(detected, "deepseek") != 0) {
        return context_print_unsupported_family(options,
                                                ref && ref->path ? ref->path : "",
                                                target_id,
                                                target_class,
                                                detected,
                                                "context report supports DeepSeek-family GGUF artifacts first");
    }

    model_max_context = context_metadata_max_context(ctx, &source);
    coverage_status = context_status_from_collections(collections, selected_target);
    has_qkv = collections &&
              collections->has_attention_q &&
              collections->has_attention_k &&
              collections->has_attention_v;

    if (options && options->tokens_text) {
        rc = yvex_token_input_parse_explicit(options->tokens_text, &input, &err);
        if (rc != YVEX_OK) {
            context_print_common_header(options,
                                        "context-report-fail",
                                        ref && ref->path ? ref->path : "",
                                        target_id,
                                        target_class,
                                        "deepseek",
                                        detected,
                                        coverage_status,
                                        coverage_status,
                                        coverage_status);
            yvex_cli_out_writef(stdout, "token_input_status: failed\n");
            yvex_cli_out_writef(stdout, "reason: %s\n", yvex_error_message(&err));
            yvex_cli_out_lines(stdout, literal_pair_4, sizeof(literal_pair_4) / sizeof(literal_pair_4[0]));
            context_print_phases(coverage_status, coverage_status, coverage_status, "failed", "token-input");
            return exit_for_status(rc);
        }
        token_count = input.token_count;
    }

    if (options && options->context_length_seen) {
        active_context = options->context_length;
        overflow_known = 1;
    } else if (model_max_context > 0ull && !selected_target) {
        active_context = model_max_context;
        overflow_known = 1;
    } else {
        active_context = 0ull;
    }
    if (overflow_known && token_count > active_context) {
        overflow = 1;
    }
    if (options && options->chunk_size_seen && token_count > 0ull) {
        chunk_count = (token_count + options->chunk_size - 1ull) / options->chunk_size;
        last_chunk = token_count % options->chunk_size;
        if (last_chunk == 0ull) last_chunk = options->chunk_size;
    }
    decode_start = token_count;

    if (options && options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        context_print_normal(options, coverage_status, selected_target);
        return 0;
    }

    context_print_common_header(options,
                                "context-report",
                                ref && ref->path ? ref->path : "",
                                target_id,
                                target_class,
                                "deepseek",
                                detected,
                                coverage_status,
                                coverage_status,
                                coverage_status);
    if (options) {
        (void)yvex_cli_out_fields(stdout, options, context_audit_option_fields,
                                  sizeof(context_audit_option_fields) /
                                      sizeof(context_audit_option_fields[0]));
    }

    yvex_cli_out_writef(stdout, "model_max_context: ");
    if (model_max_context > 0ull) yvex_cli_out_writef(stdout, "%llu", model_max_context);
    else yvex_cli_out_writef(stdout, "unknown");
    yvex_cli_out_writef(stdout, "\n");
    yvex_cli_out_writef(stdout, "model_max_context_source: %s\n",
        model_max_context > 0ull ? source : "metadata-missing-or-selected-slice");
    yvex_cli_out_writef(stdout, "model_max_context_status: %s\n", model_max_context > 0ull ? "available" : "planned");
    yvex_cli_out_writef(stdout, "requested_context: ");
    if (options && options->context_length_seen) yvex_cli_out_writef(stdout, "%llu", options->context_length);
    else yvex_cli_out_writef(stdout, "not-requested");
    yvex_cli_out_writef(stdout, "\n");
    yvex_cli_out_writef(stdout, "requested_context_source: %s\n",
        options && options->context_length_seen ? "operator-request" : "none");
    yvex_cli_out_writef(stdout, "requested_context_status: %s\n",
        options && options->context_length_seen ? "reported-only" : "not-requested");
    yvex_cli_out_writef(stdout, "active_context: ");
    if (active_context > 0ull) yvex_cli_out_writef(stdout, "%llu", active_context);
    else yvex_cli_out_writef(stdout, "diagnostic");
    yvex_cli_out_writef(stdout, "\n");
    yvex_cli_out_writef(stdout, "active_context_source: %s\n",
           options && options->context_length_seen ? "operator-request" :
           (model_max_context > 0ull && !selected_target ? "model-metadata" : "diagnostic-runtime-boundary"));
    yvex_cli_out_writef(stdout, "active_context_status: %s\n", active_context > 0ull ? "reported-only" : "diagnostic");

    yvex_cli_out_writef(stdout, "token_input_status: %s\n",
        options && options->tokens_text ? "available" : "not-provided");
    yvex_cli_out_writef(stdout, "token_count: %llu\n", token_count);
    yvex_cli_out_writef(stdout, "prompt_token_count: %llu\n", token_count);
    yvex_cli_out_writef(stdout, "prefill_token_count: %llu\n", token_count);
    yvex_cli_out_writef(stdout, "generated_token_count: 0\n");
    yvex_cli_out_writef(stdout, "token_range_start: %s\n", token_count > 0ull ? "0" : "none");
    yvex_cli_out_writef(stdout, "token_range_end: ");
    if (token_count > 0ull) yvex_cli_out_writef(stdout, "%llu", token_count - 1ull);
    else yvex_cli_out_writef(stdout, "none");
    yvex_cli_out_writef(stdout, "\n");

    yvex_cli_out_writef(stdout, "chunking_required: %s\n", options && options->chunk_size_seen ? "true" : "false");
    yvex_cli_out_writef(stdout, "chunk_size: ");
    if (options && options->chunk_size_seen) yvex_cli_out_writef(stdout, "%llu", options->chunk_size);
    else yvex_cli_out_writef(stdout, "not-requested");
    yvex_cli_out_writef(stdout, "\n");
    yvex_cli_out_writef(stdout, "chunk_count: %llu\n", chunk_count);
    yvex_cli_out_writef(stdout, "last_chunk_size: %llu\n", last_chunk);
    yvex_cli_out_writef(stdout, "chunking_policy: %s\n",
        options && options->chunk_size_seen ? "explicit-token-chunks-report-only" : "not-requested");
    yvex_cli_out_writef(stdout, "chunking_status: %s\n",
        options && options->chunk_size_seen ? "report-only" : "planned");

    yvex_cli_out_writef(stdout, "prefill_boundary_status: report-only\n");
    yvex_cli_out_writef(stdout, "prefill_position_start: %s\n", token_count > 0ull ? "0" : "none");
    yvex_cli_out_writef(stdout, "prefill_position_end: %llu\n", token_count);
    yvex_cli_out_lines(stdout, literal_lines_5, sizeof(literal_lines_5) / sizeof(literal_lines_5[0]));
    yvex_cli_out_writef(stdout, "decode_start_position: %llu\n", decode_start);
    yvex_cli_out_lines(stdout, literal_lines_6, sizeof(literal_lines_6) / sizeof(literal_lines_6[0]));
    yvex_cli_out_writef(stdout, "context_overflow: %s\n",
        overflow_known ? (overflow ? "overflow" : "none") : "unknown");
    yvex_cli_out_writef(stdout, "overflow_check_status: %s\n", overflow_known ? "pass" : "unknown");
    yvex_cli_out_writef(stdout, "overflow_stop_reason: %s\n", overflow ? "context-limit" : "none");
    yvex_cli_out_writef(stdout, "overflow_mutates_state: false\n");

    yvex_cli_out_writef(stdout, "attention_dependency_status: %s\n",
        has_qkv ? "blocked-runtime-integration" : "blocked-missing-qkv");
    yvex_cli_out_lines(stdout, literal_lines_7, sizeof(literal_lines_7) / sizeof(literal_lines_7[0]));
    yvex_cli_out_writef(stdout, "kv_dependency_status: %s\n",
        has_qkv ? "blocked-runtime-integration" : "blocked-missing-qkv");
    yvex_cli_out_lines(stdout, literal_pair_3, sizeof(literal_pair_3) / sizeof(literal_pair_3[0]));
    yvex_cli_out_writef(stdout, "kv_context_positions: %s", active_context > 0ull ? "" : "unknown");
    if (active_context > 0ull) yvex_cli_out_writef(stdout, "%llu", active_context);
    yvex_cli_out_lines(stdout, literal_lines_8, sizeof(literal_lines_8) / sizeof(literal_lines_8[0]));
    yvex_cli_out_writef(stdout, "context_blockers: %s\n",
           selected_target
               ? "selected runtime slice has no full context metadata, no real attention KV, no full "
                   "transformer prefill, no real decode, no full generation"
               : "full transformer prefill unsupported; real attention-backed KV unsupported; real decode "
                   "unsupported; full generation unsupported");
    yvex_cli_out_lines(stdout, literal_lines_9, sizeof(literal_lines_9) / sizeof(literal_lines_9[0]));
    context_print_phases(coverage_status, coverage_status, coverage_status, "report-only", NULL);
    return 0;
}

int yvex_model_artifacts_surface_context_command(int arg_count, char **args)
{
    yvex_cli_context_options options;
    yvex_model_ref_options ref_options;
    yvex_model_ref ref;
    yvex_model_context ctx;
    yvex_fullmodel_collections collections;
    yvex_error err;
    const char *target_id;
    const char *target_class;
    yvex_arch arch;
    unsigned long long tensor_count;
    unsigned long long i;
    int selected_target;
    int rc;

    yvex_error_clear(&err);
    memset(&options, 0, sizeof(options));
    memset(&ref_options, 0, sizeof(ref_options));
    memset(&ref, 0, sizeof(ref));
    memset(&ctx, 0, sizeof(ctx));
    memset(&collections, 0, sizeof(collections));

    rc = parse_context_options(arg_count, args, &options);
    if (rc == 1) return 0;
    if (rc != 0) return rc;

    if (strcmp(options.model, "glm-5.2-official-safetensors") == 0) {
        context_print_unsupported_source_only(&options);
        return 5;
    }

    ref_options.allow_registry = 1;
    ref_options.registry_path = options.registry_path;
    rc = yvex_model_ref_resolve(&ref, options.model, &ref_options, &err);
    if (rc != YVEX_OK) {
        context_print_common_header(&options,
                                    "context-report-fail",
                                    "",
                                    "unknown",
                                    "unknown",
                                    "unknown",
                                    "unknown",
                                    "failed",
                                    "failed",
                                    "failed");
        yvex_cli_out_writef(stdout, "reason: %s\n", yvex_error_message(&err));
        yvex_cli_out_lines(stdout, literal_pair_2, sizeof(literal_pair_2) / sizeof(literal_pair_2[0]));
        context_print_phases("failed", "failed", "failed", "failed", "resolve-model");
        yvex_error_clear(&err);
        return exit_for_status(rc);
    }

    rc = yvex_model_context_open(ref.path, &ctx, &err);
    if (rc != YVEX_OK) {
        context_print_common_header(&options,
                                    "context-report-fail",
                                    ref.path,
                                    ref.alias && ref.alias[0] ? ref.alias : "path",
                                    "GGUF-artifact",
                                    "unknown",
                                    "unknown",
                                    "failed",
                                    "failed",
                                    "failed");
        yvex_cli_out_writef(stdout, "reason: %s\n", yvex_error_message(&err));
        yvex_cli_out_lines(stdout, literal_pair_1, sizeof(literal_pair_1) / sizeof(literal_pair_1[0]));
        context_print_phases("failed", "failed", "failed", "failed", "context-profile");
        yvex_error_clear(&err);
        yvex_model_ref_clear(&ref);
        return exit_for_status(rc);
    }

    tensor_count = yvex_tensor_table_count(ctx.table);
    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(ctx.table, i);
        if (tensor) fullmodel_classify_tensor(tensor, &collections);
    }
    if (yvex_gguf_metadata_find(ctx.gguf, "tokenizer.ggml.tokens") ||
        yvex_gguf_metadata_find(ctx.gguf, "tokenizer.ggml.model")) {
        collections.tokenizer = 1ull;
        collections.has_tokenizer_metadata = 1;
    }

    arch = yvex_model_arch(ctx.model);
    target_id = ref.alias && ref.alias[0] ? ref.alias : "path";
    selected_target = fullmodel_is_selected_target(target_id) ||
                      fullmodel_is_selected_target(options.model);
    target_class = selected_target ? "selected-runtime-slice" : "candidate-GGUF-path";
    rc = context_print_report(&options,
                              &ref,
                              &ctx,
                              target_id,
                              target_class,
                              arch,
                              &collections,
                              selected_target);
    yvex_model_context_close(&ctx);
    yvex_model_ref_clear(&ref);
    return rc;
}

void yvex_model_artifacts_surface_context_help(FILE *fp)
{
    yvex_cli_out_writef(fp,
        "usage: yvex inspect context report --model FILE_OR_ALIAS [--family auto|deepseek|glm|qwen] [--backend cpu|"
            "cuda] [--audit | --output normal|table|audit] [options]\n\n");
    yvex_cli_out_lines(fp, literal_lines_10, sizeof(literal_lines_10) / sizeof(literal_lines_10[0]));
    yvex_cli_out_writef(fp,
        "Options: --registry FILE --context-length N --tokens IDS --chunk-size N --include-attention --"
            "include-kv --include-prefill --include-decode --include-blockers\n\n");
    yvex_cli_out_lines(fp, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
    yvex_cli_out_writef(fp, "  Default output is compact. Use --audit for full diagnostic fields.\n");
    yvex_cli_out_lines(fp, literal_lines_11, sizeof(literal_lines_11) / sizeof(literal_lines_11[0]));
}
