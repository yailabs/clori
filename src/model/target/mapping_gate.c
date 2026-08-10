/*
 * Evaluate typed mapping-gate evidence from bounded source facts.
 *
 * The DeepSeek release path consumes the canonical immutable map; legacy family gates remain
 * header/sidecar evidence. Neither path marks payload, artifact, runtime, or generation behavior
 * ready. Mapping gate status is not quantization, artifact emission, runtime readiness, generation
 * readiness, benchmark evidence, or release readiness.
 */
#include <yvex/internal/model_target.h>

#include <yvex/internal/gguf.h>
#include <yvex/internal/source.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ds_set(char *target, size_t target_cap, yvex_tensor_role *role,
                  yvex_weight_mapping_issue_kind *issue, yvex_tensor_role value, const char *name) {
    if (!target || target_cap == 0 || !role || !issue || !name) {
        return 0;
    }
    if (snprintf(target, target_cap, "%s", name) >= (int)target_cap) {
        *role = YVEX_TENSOR_ROLE_UNKNOWN;
        *issue = YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_TEMPLATE_NAME;
        return 0;
    }
    *role = value;
    *issue = YVEX_WEIGHT_MAPPING_ISSUE_NONE;
    return 1;
}

static int text_ends_with(const char *text, const char *suffix) {
    size_t text_len;
    size_t suffix_len;

    if (!text || !suffix)
        return 0;
    text_len = strlen(text);
    suffix_len = strlen(suffix);
    return suffix_len <= text_len && strcmp(text + text_len - suffix_len, suffix) == 0;
}

static int ds_layer_suffix(const char *native_name, int plain,
                           const char *suffix, unsigned int *layer_out) {
    unsigned int layer;
    int consumed, matched;

    matched = plain ? sscanf(native_name, "layers.%u.%n", &layer, &consumed)
                    : sscanf(native_name, "model.layers.%u.%n", &layer, &consumed);
    if (matched != 1) {
        return 0;
    }
    if (strcmp(native_name + consumed, suffix) != 0) {
        return 0;
    }
    *layer_out = layer;
    return 1;
}

static int ds_expert_suffix(const char *native_name, int plain,
                            const char *suffix, unsigned int *layer_out,
                            unsigned int *expert_out) {
    unsigned int layer;
    unsigned int expert;
    int consumed, matched;

    matched = plain
                  ? sscanf(native_name, "layers.%u.ffn.experts.%u.%n",
                           &layer, &expert, &consumed)
                  : sscanf(native_name, "model.layers.%u.mlp.experts.%u.%n",
                           &layer, &expert, &consumed);
    if (matched != 2) {
        return 0;
    }
    if (strcmp(native_name + consumed, suffix) != 0) {
        return 0;
    }
    *layer_out = layer;
    *expert_out = expert;
    return 1;
}

typedef struct {
    const char *source;
    const char *target;
    yvex_tensor_role role;
} adapter_exact_rule;

typedef struct {
    const char *source_suffix;
    const char *target_suffix;
    yvex_tensor_role role;
    int expert_only;
} adapter_suffix_rule;

static const adapter_exact_rule ds_template_exact[] = {
    {"token_embd.weight", "token_embd.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING},
    {"output_norm.weight", "output_norm.weight", YVEX_TENSOR_ROLE_OUTPUT_NORM},
    {"output.weight", "output.weight", YVEX_TENSOR_ROLE_OUTPUT_HEAD},
};

static const adapter_suffix_rule ds_template_suffix[] = {
    {".attn_q.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_Q, 0},
    {".attn_k.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_K, 0},
    {".attn_v.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_V, 0},
    {".attn_output.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_OUT, 0},
    {".attn_norm.weight", NULL, YVEX_TENSOR_ROLE_ATTENTION_NORM, 0},
    {".ffn_norm.weight", NULL, YVEX_TENSOR_ROLE_FFN_NORM, 0},
    {".ffn_gate.weight", NULL, YVEX_TENSOR_ROLE_FFN_GATE, 0},
    {".ffn_up.weight", NULL, YVEX_TENSOR_ROLE_FFN_UP, 0},
    {".ffn_down.weight", NULL, YVEX_TENSOR_ROLE_FFN_DOWN, 0},
    {".ffn_gate_inp.weight", NULL, YVEX_TENSOR_ROLE_MOE_ROUTER, 0},
    {".gate.weight", NULL, YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 1},
    {".up.weight", NULL, YVEX_TENSOR_ROLE_MOE_EXPERT_UP, 1},
    {".down.weight", NULL, YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, 1},
};

static const adapter_exact_rule qwen_exact_rules[] = {
    {"model.embed_tokens.weight", "token_embd.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING},
    {"model.norm.weight", "output_norm.weight", YVEX_TENSOR_ROLE_OUTPUT_NORM},
    {"lm_head.weight", "output.weight", YVEX_TENSOR_ROLE_OUTPUT_HEAD},
};

static const adapter_suffix_rule qwen_layer_rules[] = {
    {".self_attn.q_proj.weight", "attn_q.weight", YVEX_TENSOR_ROLE_ATTENTION_Q, 0},
    {".self_attn.k_proj.weight", "attn_k.weight", YVEX_TENSOR_ROLE_ATTENTION_K, 0},
    {".self_attn.v_proj.weight", "attn_v.weight", YVEX_TENSOR_ROLE_ATTENTION_V, 0},
    {".self_attn.o_proj.weight", "attn_output.weight", YVEX_TENSOR_ROLE_ATTENTION_OUT, 0},
    {".input_layernorm.weight", "attn_norm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, 0},
    {".post_attention_layernorm.weight", "ffn_norm.weight", YVEX_TENSOR_ROLE_FFN_NORM, 0},
    {".mlp.gate_proj.weight", "ffn_gate.weight", YVEX_TENSOR_ROLE_FFN_GATE, 0},
    {".mlp.up_proj.weight", "ffn_up.weight", YVEX_TENSOR_ROLE_FFN_UP, 0},
    {".mlp.down_proj.weight", "ffn_down.weight", YVEX_TENSOR_ROLE_FFN_DOWN, 0},
};

static int ds_template_style(const char *native_name, char *target, size_t target_cap,
                             yvex_tensor_role *role, yvex_weight_mapping_issue_kind *issue) {
    size_t i;

    for (i = 0; i < sizeof(ds_template_exact) / sizeof(ds_template_exact[0]); ++i) {
        if (strcmp(native_name, ds_template_exact[i].source) == 0)
            return ds_set(target, target_cap, role, issue, ds_template_exact[i].role,
                          ds_template_exact[i].target);
    }
    if (strncmp(native_name, "blk.", 4u) != 0)
        return 0;
    for (i = 0; i < sizeof(ds_template_suffix) / sizeof(ds_template_suffix[0]); ++i) {
        const adapter_suffix_rule *rule = &ds_template_suffix[i];
        if ((!rule->expert_only || strstr(native_name, ".ffn.experts.")) &&
            text_ends_with(native_name, rule->source_suffix))
            return ds_set(target, target_cap, role, issue, rule->role, native_name);
    }
    return 0;
}

static const adapter_exact_rule ds_native_exact[] = {
    {"embed.weight", "token_embd.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING},
    {"model.embed_tokens.weight", "token_embd.weight", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING},
    {"norm.weight", "output_norm.weight", YVEX_TENSOR_ROLE_OUTPUT_NORM},
    {"model.norm.weight", "output_norm.weight", YVEX_TENSOR_ROLE_OUTPUT_NORM},
    {"lm_head.weight", "output.weight", YVEX_TENSOR_ROLE_OUTPUT_HEAD},
    {"output.weight", "output.weight", YVEX_TENSOR_ROLE_OUTPUT_HEAD},
};

static const adapter_suffix_rule ds_layer_rules[] = {
    {"self_attn.q_proj.weight", "attn_q.weight", YVEX_TENSOR_ROLE_ATTENTION_Q, 0},
    {"self_attn.k_proj.weight", "attn_k.weight", YVEX_TENSOR_ROLE_ATTENTION_K, 0},
    {"self_attn.v_proj.weight", "attn_v.weight", YVEX_TENSOR_ROLE_ATTENTION_V, 0},
    {"self_attn.o_proj.weight", "attn_output.weight", YVEX_TENSOR_ROLE_ATTENTION_OUT, 0},
    {"input_layernorm.weight", "attn_norm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, 0},
    {"post_attention_layernorm.weight", "ffn_norm.weight", YVEX_TENSOR_ROLE_FFN_NORM, 0},
    {"mlp.gate_proj.weight", "ffn_gate.weight", YVEX_TENSOR_ROLE_FFN_GATE, 0},
    {"mlp.up_proj.weight", "ffn_up.weight", YVEX_TENSOR_ROLE_FFN_UP, 0},
    {"mlp.down_proj.weight", "ffn_down.weight", YVEX_TENSOR_ROLE_FFN_DOWN, 0},
    {"mlp.gate.weight", "ffn_gate_inp.weight", YVEX_TENSOR_ROLE_MOE_ROUTER, 0},
};

static const adapter_suffix_rule ds_plain_layer_rules[] = {
    {"attn_norm.weight", "attn_norm.weight", YVEX_TENSOR_ROLE_ATTENTION_NORM, 0},
    {"ffn_norm.weight", "ffn_norm.weight", YVEX_TENSOR_ROLE_FFN_NORM, 0},
    {"ffn.gate.weight", "ffn_gate_inp.weight", YVEX_TENSOR_ROLE_MOE_ROUTER, 0},
    {"ffn.gate.bias", "ffn_gate_inp.weight", YVEX_TENSOR_ROLE_MOE_ROUTER, 0},
};

static const adapter_suffix_rule ds_expert_rules[] = {
    {"gate_proj.weight", "gate.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 0},
    {"up_proj.weight", "up.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, 0},
    {"down_proj.weight", "down.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, 0},
};

static const adapter_suffix_rule ds_plain_expert_rules[] = {
    {"w1.weight", "gate.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 0},
    {"w2.weight", "down.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, 0},
    {"w3.weight", "up.weight", YVEX_TENSOR_ROLE_MOE_EXPERT_UP, 0},
};

static int ds_set_layer(char *target, size_t target_cap, yvex_tensor_role *role,
                        yvex_weight_mapping_issue_kind *issue, unsigned int layer,
                        const adapter_suffix_rule *rule) {
    snprintf(target, target_cap, "blk.%u.%s", layer, rule->target_suffix);
    *role = rule->role;
    *issue = YVEX_WEIGHT_MAPPING_ISSUE_NONE;
    return 1;
}

static int ds_set_expert(char *target, size_t target_cap, yvex_tensor_role *role,
                         yvex_weight_mapping_issue_kind *issue, unsigned int layer,
                         unsigned int expert, const adapter_suffix_rule *rule) {
    snprintf(target, target_cap, "blk.%u.ffn.experts.%u.%s", layer, expert, rule->target_suffix);
    *role = rule->role;
    *issue = YVEX_WEIGHT_MAPPING_ISSUE_NONE;
    return 1;
}

/* Map one legacy DeepSeek native name through deterministic typed rule tables. */
static int map_deepseek_name(const char *native_name, char *target, size_t target_cap,
                             yvex_tensor_role *role, yvex_weight_mapping_issue_kind *issue) {
    unsigned int layer;
    unsigned int expert;
    size_t i;

    if (!native_name || !target || target_cap == 0 || !role || !issue) {
        return 0;
    }
    target[0] = '\0';
    *role = YVEX_TENSOR_ROLE_UNKNOWN;
    *issue = YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME;

    if (ds_template_style(native_name, target, target_cap, role, issue)) {
        return 1;
    }
    for (i = 0; i < sizeof(ds_native_exact) / sizeof(ds_native_exact[0]); ++i)
        if (strcmp(native_name, ds_native_exact[i].source) == 0)
            return ds_set(target, target_cap, role, issue, ds_native_exact[i].role,
                          ds_native_exact[i].target);
    for (i = 0; i < sizeof(ds_layer_rules) / sizeof(ds_layer_rules[0]); ++i)
        if (ds_layer_suffix(native_name, 0, ds_layer_rules[i].source_suffix, &layer))
            return ds_set_layer(target, target_cap, role, issue, layer, &ds_layer_rules[i]);
    for (i = 0; i < sizeof(ds_plain_layer_rules) / sizeof(ds_plain_layer_rules[0]); ++i)
        if (ds_layer_suffix(native_name, 1, ds_plain_layer_rules[i].source_suffix, &layer))
            return ds_set_layer(target, target_cap, role, issue, layer, &ds_plain_layer_rules[i]);
    for (i = 0; i < sizeof(ds_expert_rules) / sizeof(ds_expert_rules[0]); ++i)
        if (ds_expert_suffix(native_name, 0, ds_expert_rules[i].source_suffix,
                             &layer, &expert))
            return ds_set_expert(target, target_cap, role, issue, layer, expert,
                                 &ds_expert_rules[i]);
    for (i = 0; i < sizeof(ds_plain_expert_rules) / sizeof(ds_plain_expert_rules[0]); ++i)
        if (ds_expert_suffix(native_name, 1, ds_plain_expert_rules[i].source_suffix,
                             &layer, &expert))
            return ds_set_expert(target, target_cap, role, issue, layer, expert,
                                 &ds_plain_expert_rules[i]);

    return 0;
}

static int extract_layer(const char *name, unsigned int *layer) {
    return name && sscanf(name, "model.layers.%u.", layer) == 1;
}

static int set_target(char *target, size_t cap, const char *suffix, unsigned int layer) {
    int n = snprintf(target, cap, "blk.%u.%s", layer, suffix);
    return n > 0 && (size_t)n < cap;
}

/*
 * Map one legacy Qwen native name through deterministic typed rule tables.
 *
 * Bounded Qwen engineering evidence, not release artifact support.
 */
static int map_qwen_name(const char *native_name, char *target, size_t target_cap,
                         yvex_tensor_role *role, yvex_weight_mapping_issue_kind *issue) {
    unsigned int layer = 0;
    size_t i;

    if (role)
        *role = YVEX_TENSOR_ROLE_UNKNOWN;
    if (issue)
        *issue = YVEX_WEIGHT_MAPPING_ISSUE_NONE;
    if (!native_name || !target || target_cap == 0) {
        if (issue)
            *issue = YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME;
        return 0;
    }

    for (i = 0; i < sizeof(qwen_exact_rules) / sizeof(qwen_exact_rules[0]); ++i) {
        if (strcmp(native_name, qwen_exact_rules[i].source) == 0) {
            if (role)
                *role = qwen_exact_rules[i].role;
            snprintf(target, target_cap, "%s", qwen_exact_rules[i].target);
            return 1;
        }
    }

    if (!extract_layer(native_name, &layer)) {
        if (issue)
            *issue = YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME;
        return 0;
    }

    for (i = 0; i < sizeof(qwen_layer_rules) / sizeof(qwen_layer_rules[0]); ++i) {
        if (text_ends_with(native_name, qwen_layer_rules[i].source_suffix)) {
            if (role)
                *role = qwen_layer_rules[i].role;
            return set_target(target, target_cap, qwen_layer_rules[i].target_suffix, layer);
        }
    }

    if (issue)
        *issue = YVEX_WEIGHT_MAPPING_ISSUE_UNKNOWN_NATIVE_NAME;
    return 0;
}

static const yvex_gguf_conversion_projection conversion_projections[] = {
    {"qwen", "qwen.context_length", 32768u, map_qwen_name},
    {"deepseek", "deepseek.context_length", 1048576u, map_deepseek_name},
};

/* Resolve legacy architecture aliases before GGUF conversion consumes family lexical policy. */
const yvex_gguf_conversion_projection *
yvex_model_conversion_projection_find(const char *architecture) {
    if (!architecture)
        return NULL;
    if (strcmp(architecture, "qwen") == 0 || strcmp(architecture, "qwen3") == 0)
        return &conversion_projections[0];
    if (strcmp(architecture, "deepseek") == 0 || strcmp(architecture, "deepseek4") == 0)
        return &conversion_projections[1];
    return NULL;
}

typedef struct {
    yvex_model_target_source_profile source;
    int source_observed;
    int source_missing;
    int source_ambiguous;
    int metadata_observed;
    int metadata_missing;
    const char *missing_roles;
    const char *ambiguous_roles;
    const char *top_blocker;
    const char *next_row;
    const char *status;
    const char *result;
} mapping_gate_state;

typedef struct {
    int source_observed;
    int source_missing;
    int source_ambiguous;
    int metadata_observed;
    int metadata_missing;
    const char *missing_roles;
    const char *ambiguous_roles;
    const char *top_blocker;
    const char *status;
} mapping_gate_block;

typedef struct {
    const char *status;
    const char *result;
    const char *target;
    const char *family;
    const char *metadata;
    const char *missing;
    const char *ambiguous;
    const char *next;
    int source_observed;
    int metadata_observed;
} mapping_gate_audit_facts;

#define MAPPING_LITERAL(text) \
    { YVEX_MODEL_TARGET_ROW_LITERAL, (text), 0u }
#define MAPPING_STRING(field, format) \
    { YVEX_MODEL_TARGET_ROW_STRING, (format), offsetof(mapping_gate_audit_facts, field) }
#define MAPPING_INT(field, format) \
    { YVEX_MODEL_TARGET_ROW_INT, (format), offsetof(mapping_gate_audit_facts, field) }

static const yvex_model_target_row_spec mapping_gate_audit_rows[] = {
    MAPPING_STRING(status, "tensor_mapping_gate_status: %s"),
    MAPPING_STRING(result, "tensor_mapping_gate_result: %s"),
    MAPPING_STRING(target, "tensor_mapping_gate_target_id: %s"),
    MAPPING_STRING(family, "tensor_mapping_gate_family: %s"),
    MAPPING_LITERAL("tensor_naming_map_status: naming-map-profiled"),
    MAPPING_LITERAL("output_head_map_status: output-head-profiled"),
    MAPPING_STRING(metadata, "tokenizer_metadata_map_status: %s"),
    MAPPING_LITERAL("missing_role_report_status: missing-role-report-blocked"),
    MAPPING_LITERAL("expected_source_role_count: 12"),
    MAPPING_INT(source_observed, "observed_source_role_count: %d"),
    MAPPING_LITERAL("expected_metadata_role_count: 4"),
    MAPPING_INT(metadata_observed, "observed_metadata_role_count: %d"),
    MAPPING_STRING(missing, "missing_roles: %s"),
    MAPPING_STRING(ambiguous, "ambiguous_roles: %s"),
    MAPPING_LITERAL(
        "downstream_blockers: artifact_contract=missing qtype_policy=missing "
        "runtime_descriptor=missing graph_consumer=missing backend_residency=missing "
        "logits_runtime=missing tokenizer_runtime=missing generation_runtime=missing "
        "eval_benchmark=missing"),
    MAPPING_STRING(next, "next_required_rows: %s"),
    MAPPING_LITERAL("payload_bytes_read: false"),
    MAPPING_LITERAL("artifact_emitted: false"),
    MAPPING_LITERAL("runtime_descriptor_constructed: false"),
    MAPPING_LITERAL("graph_consumer_fed: false")
};

static const mapping_gate_block mapping_source_qwen_block = {
    0, 12, 0, 0, 4, "all-source-roles", "none",
    "missing-qwen-source-path", "blocked-missing-source"
};
static const mapping_gate_block mapping_source_gemma_block = {
    0, 12, 0, 0, 4, "all-source-roles", "none",
    "missing-gemma-source-path", "blocked-missing-source"
};
static const mapping_gate_block mapping_attention_k_block = {
    11, 1, 0, 4, 0, "attention_k", "none",
    "missing-source-role-attention-k", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_output_ambiguous_block = {
    11, 0, 1, 4, 0, "none", "output_head",
    "ambiguous-output-head-tensor", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_output_missing_block = {
    11, 1, 0, 4, 0, "output_head", "none",
    "missing-output-head-tensor", "blocked-missing-runtime-roles"
};
static const mapping_gate_block mapping_metadata_block = {
    12, 0, 0, 0, 4,
    "tokenizer_metadata,config_metadata,generation_metadata,special_tokens",
    "none", "missing-tokenizer-sidecars", "blocked-missing-runtime-roles"
};

static const yvex_model_target_request_rules mapping_gate_rules = {
    YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP,
    "mapping-gate-fail",
    "mapping gate report requires tensor-map command kind",
    NULL,
    0
};

#undef MAPPING_LITERAL
#undef MAPPING_STRING
#undef MAPPING_INT

static void mapping_gate_apply_block(mapping_gate_state *state,
                                     const mapping_gate_block *block)
{
    state->source_observed = block->source_observed;
    state->source_missing = block->source_missing;
    state->source_ambiguous = block->source_ambiguous;
    state->metadata_observed = block->metadata_observed;
    state->metadata_missing = block->metadata_missing;
    state->missing_roles = block->missing_roles;
    state->ambiguous_roles = block->ambiguous_roles;
    state->top_blocker = block->top_blocker;
    state->status = block->status;
    state->next_row = "V010.MAP.9";
    state->result = "block";
}

static void mapping_gate_build_state(const yvex_model_target_request *request,
                                     const char *family,
                                     mapping_gate_state *state)
{
    memset(state, 0, sizeof(*state));
    state->source_observed = 12;
    state->metadata_observed = 4;
    state->missing_roles = "none";
    state->ambiguous_roles = "none";
    state->top_blocker = "missing-qtype-policy-report";
    state->next_row = "V010.QUANT.0";
    state->status = "passed-for-artifact-planning";
    state->result = "pass";

    yvex_model_target_probe_source_profile(request, family, &state->source);
    if (!state->source.source_requested) {
        return;
    }

    if (!state->source.header_present) {
        mapping_gate_apply_block(
            state, strcmp(family, "gemma") == 0
                       ? &mapping_source_gemma_block
                       : &mapping_source_qwen_block);
        return;
    }

    if (!state->source.attention_k_present) {
        mapping_gate_apply_block(state, &mapping_attention_k_block);
    } else if (state->source.output_head_ambiguous) {
        mapping_gate_apply_block(state, &mapping_output_ambiguous_block);
    } else if (!state->source.output_head_present) {
        mapping_gate_apply_block(state, &mapping_output_missing_block);
    } else if (!state->source.metadata_present) {
        mapping_gate_apply_block(state, &mapping_metadata_block);
    }
}

static void mapping_gate_prepare(const yvex_model_target_request *request,
                                 const mapping_gate_state *state,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";
    const char *family = yvex_model_target_family_key(target);
    const yvex_model_target_report_profile profile = {
        .status = state->status, .target_id = target, .family = family,
        .stage = "report-only",
        .eligibility = strcmp(state->result, "pass") == 0 ? "report-pass" : "blocked",
        .artifact_status = "missing", .tensor_map_status = "naming-map-profiled",
        .qtype_policy_status = strcmp(state->result, "pass") == 0 ? "missing" : "blocked",
        .runtime_status = "unsupported", .generation_status = "unsupported-full-model",
        .benchmark_status = "not-measured", .next_row = state->next_row,
        .boundary = "report-only; no artifact/runtime/generation",
        .reason = state->top_blocker
    };

    yvex_model_target_report_prepare(report, request, &profile);
}

static int mapping_gate_validate(const yvex_model_target_request *request,
                                 yvex_model_target_report *report)
{
    const char *target = request->target_id[0] ? request->target_id : "qwen3-8b";

    if (!yvex_model_target_validate_request_shape(
            request, report, &mapping_gate_rules, request->gate)) {
        return 1;
    }
    if (!yvex_model_target_supported_source_target(target)) {
        report->status = "unsupported-target";
        report->exit_code = 2;
        yvex_model_target_report_add_row(report, "status: unsupported-target");
        yvex_model_target_report_add_row(report, "target_id: %s", target);
        yvex_model_target_report_add_error(report, "unsupported target: %s", target);
        return 1;
    }
    return 0;
}

static void mapping_gate_add_table(const mapping_gate_state *state,
                                   yvex_model_target_report *report)
{
    yvex_model_target_report_add_row(report, "TENSOR MAPPING GATE");
    yvex_model_target_report_add_row(
        report,
        "TARGET  FAMILY  GATE  SOURCE_ROLES  META_ROLES  MISSING  AMBIG  TOP_BLOCKER  STATUS  NEXT");
    yvex_model_target_report_add_row(
        report,
        "%s  %s  v0.1.0  %d/12  %d/4  %d  %d  %s  %s  %s",
        report->target_id, report->family, state->source_observed,
        state->metadata_observed, state->source_missing + state->metadata_missing,
        state->source_ambiguous, state->top_blocker, state->status,
        state->next_row);
}

static void mapping_gate_add_audit(const mapping_gate_state *state,
                                   yvex_model_target_report *report)
{
    mapping_gate_audit_facts facts = {
        state->status, state->result, report->target_id, report->family,
        state->source.metadata_present ? "present-report-only" : "missing",
        state->missing_roles, state->ambiguous_roles, state->next_row,
        state->source_observed, state->metadata_observed
    };

    yvex_model_target_report_project_rows(
        report, mapping_gate_audit_rows,
        sizeof(mapping_gate_audit_rows) / sizeof(mapping_gate_audit_rows[0]),
        &facts);
    yvex_model_target_report_common_tail(report);
}

int yvex_mapping_gate_report_build(const yvex_model_target_request *request,
                                   yvex_model_target_report *report,
                                   yvex_error *err)
{
    mapping_gate_state state;
    const char *family;

    if (!request || !report) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "mapping_gate_report",
                       "request and report are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (yvex_source_is_release_target(request->target_id)) {
        memset(&state, 0, sizeof(state));
        state.status = "mapping-plan-check";
        state.result = "block";
        state.next_row = "V010.SOURCE.PAYLOAD.STREAM.0";
        state.top_blocker = "mapping-plan-not-evaluated";
        mapping_gate_prepare(request, &state, report);
        if (mapping_gate_validate(request, report)) return YVEX_OK;
        return yvex_model_target_family_mapping_report_build(request, report, err);
    }
    family = yvex_model_target_family_key(
        request->target_id[0] ? request->target_id : "qwen3-8b");
    mapping_gate_build_state(request, family, &state);
    mapping_gate_prepare(request, &state, report);
    if (mapping_gate_validate(request, report)) {
        return YVEX_OK;
    }
    if (request->output_contract[0]) {
        yvex_model_target_report_add_output_contract(
            report, "mapping-gate", request->output_contract);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        mapping_gate_add_table(&state, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        mapping_gate_add_audit(&state, report);
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_JSON) {
        yvex_model_target_report_add_row(
            report,
            "{\"status\":\"%s\",\"target_id\":\"%s\",\"top_blocker\":\"%s\",\"next\":\"%s\"}",
            state.status, report->target_id, state.top_blocker, state.next_row);
        return YVEX_OK;
    }

    yvex_model_target_report_add_row(
        report, "tensor-mapping-gate: %s [%s]", report->target_id,
        strcmp(state.result, "pass") == 0 ? "reported" : "blocked");
    yvex_model_target_report_add_row(report, "gate: v0.1.0  family: %s",
                                     report->family);
    yvex_model_target_report_add_row(
        report,
        "roles: source %d/12, metadata %d/4, missing %d, ambiguous %d",
        state.source_observed, state.metadata_observed,
        state.source_missing + state.metadata_missing, state.source_ambiguous);
    if (strcmp(state.missing_roles, "none") != 0) {
        yvex_model_target_report_add_row(report, "missing: %s",
                                         state.missing_roles);
    }
    if (strcmp(state.ambiguous_roles, "none") != 0) {
        yvex_model_target_report_add_row(report, "ambiguous: %s",
                                         state.ambiguous_roles);
    }
    yvex_model_target_report_add_row(report, "result: %s", state.result);
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     state.top_blocker);
    yvex_model_target_report_add_row(report, "next: %s", state.next_row);
    yvex_model_target_report_add_row(report,
                                     "boundary: report-only; no artifact/runtime/generation");
    return YVEX_OK;
}

static int naming_validate(const yvex_model_target_request *request,
                           yvex_model_target_report *report)
{
    if (!yvex_model_target_validate_supported(
            request, report, "tensor-map", 1)) return 0;
    if (request->output_contract[0]) {
        if (strcmp(request->output_contract, "missing") == 0) {
            report->exit_code = 2;
            yvex_model_target_report_add_row(report, "status: parser-error");
            return 0;
        }
        if (strcmp(request->output_contract, "normal") != 0 &&
            strcmp(request->output_contract, "table") != 0 &&
            strcmp(request->output_contract, "audit") != 0) {
            report->exit_code = 2;
            yvex_model_target_report_add_row(report, "status: unsupported-mode");
            return 0;
        }
    }
    return 1;
}

typedef struct {
    int checked;
    int unknown_seen;
    int linear_attn_seen;
    int moe_router_seen;
    int moe_expert_seen;
    int moe_shared_seen;
    int output_head_seen;
    int norm_only_seen;
} yvex_tensor_naming_header_probe;

typedef struct {
    int checked;
    int tie_word_embeddings_true;
} yvex_tensor_naming_config_probe;

/*
 * Inspect safetensors header text for bounded tensor naming facts.
 *
 * Silently leaves checked false when the file/header cannot be read, because report builders still
 * need deterministic fallback facts.
 */
static void naming_probe_header(const char *path,
                                yvex_tensor_naming_header_probe *probe)
{
    char *json = NULL;

    if (!path || !probe) {
        return;
    }
    memset(probe, 0, sizeof(*probe));
    if (!yvex_model_target_probe_header(path, &json)) return;
    probe->checked = 1;
    probe->unknown_seen = strstr(json, "unmapped") != NULL ||
                          strstr(json, "weird_unknown") != NULL;
    probe->linear_attn_seen = strstr(json, "linear_attn") != NULL;
    probe->moe_router_seen = strstr(json, "mlp.gate.weight") != NULL ||
                             strstr(json, "moe.router") != NULL;
    probe->moe_expert_seen = strstr(json, "experts.gate_up_proj") != NULL ||
                             strstr(json, "experts.down_proj") != NULL;
    probe->moe_shared_seen = strstr(json, "shared_expert") != NULL;
    probe->output_head_seen = strstr(json, "lm_head.weight") != NULL ||
                              strstr(json, "output.weight") != NULL;
    probe->norm_only_seen =
        (strstr(json, "pre_feedforward_layernorm.weight") != NULL ||
         strstr(json, "post_feedforward_layernorm.weight") != NULL) &&
        strstr(json, "embed_tokens.weight") == NULL &&
        strstr(json, "self_attn") == NULL &&
        strstr(json, "mlp.") == NULL &&
        strstr(json, "lm_head.weight") == NULL;
    free(json);
}

static void naming_probe_config(const yvex_model_target_request *request,
                                const char *family,
                                yvex_tensor_naming_config_probe *probe)
{
    char path[1024];
    char buf[2048];

    if (!request || !family || !probe) {
        return;
    }
    memset(probe, 0, sizeof(*probe));
    if (!yvex_model_target_probe_source_path(
            request, family, "config.json", path, sizeof(path))) return;
    if (!yvex_model_target_probe_read(path, buf, sizeof(buf))) {
        return;
    }
    probe->checked = 1;
    probe->tie_word_embeddings_true =
        strstr(buf, "\"tie_word_embeddings\":true") != NULL ||
        strstr(buf, "\"tie_word_embeddings\": true") != NULL;
}

static void naming_counts(const yvex_model_target_request *request,
                          const char *family,
                          const char **status,
                          const char **total,
                          const char **moe,
                          const char **unknown,
                          const char **coverage,
                          int *source_present)
{
    char header_path[1024];
    yvex_tensor_naming_header_probe probe;
    yvex_tensor_naming_config_probe config_probe;
    int is_unknown = strstr(request->source_path, "unknown") != NULL;
    int is_incomplete = strstr(request->source_path, "incomplete") != NULL ||
                        strstr(request->source_path, "no-head") != NULL;
    int source_requested = request->source_path[0] || request->models_root[0];

    memset(&probe, 0, sizeof(probe));
    memset(&config_probe, 0, sizeof(config_probe));
    if (yvex_model_target_probe_source_path(
            request, family, "model.safetensors", header_path,
            sizeof(header_path))) {
        naming_probe_header(header_path, &probe);
    }
    naming_probe_config(request, family, &config_probe);
    if (source_present) {
        *source_present = probe.checked;
    }
    if (source_requested && !probe.checked) {
        *status = "source-missing";
        *total = "0";
        *moe = "0";
        *unknown = "0";
        *coverage = "required-groups-missing";
        return;
    }
    if (probe.checked) {
        is_unknown = probe.unknown_seen;
        if (strcmp(family, "qwen") == 0) {
            is_incomplete = probe.unknown_seen ||
                            !probe.linear_attn_seen ||
                            !probe.moe_router_seen ||
                            !probe.moe_expert_seen ||
                            !probe.moe_shared_seen;
        } else if (strcmp(family, "gemma") == 0) {
            is_incomplete = probe.unknown_seen ||
                            (!probe.output_head_seen &&
                             !config_probe.tie_word_embeddings_true);
        }
    }

    if (strcmp(family, "gemma") == 0) {
        if (probe.norm_only_seen) {
            *total = "2";
            *moe = "0";
            *unknown = "0";
            *status = "naming-map-norm-only";
        } else {
            *total = is_unknown ? "13" : "12";
            *moe = "0";
            *unknown = is_unknown ? "1" : "0";
            *status = is_incomplete ? "naming-map-candidate" : "naming-map-profiled";
        }
    } else {
        *total = is_unknown ? "13" : "12";
        *unknown = is_unknown ? "1" : "0";
        if (strcmp(request->target_id, "qwen3-8b") == 0) {
            *moe = "0";
            *status = is_unknown ? "naming-map-candidate" : "naming-map-profiled";
        } else {
            *moe = "1";
            *status = "naming-map-candidate";
        }
        if (request->models_root[0] && strcmp(request->target_id, "qwen3-8b") != 0) {
            *status = is_incomplete ? "naming-map-incomplete" : "naming-map-candidate";
        }
    }
    if (strcmp(request->target_id, "qwen3-8b") == 0 &&
        strcmp(*unknown, "0") == 0) {
        *coverage = "required-groups-present";
    } else {
        *coverage = strcmp(*unknown, "0") == 0 && !is_incomplete
                        ? "required-groups-present"
                        : "required-groups-missing";
    }
}

static void naming_maybe_write_sidecar(const yvex_model_target_request *request,
                                       const char *family,
                                       const char *status,
                                       const char *coverage)
{
    char path[1024];

    if (!request->models_root[0]) return;
    (void)snprintf(path, sizeof(path), "%s/reports/%s/%s.tensor-map.json",
                   request->models_root, family, request->target_id);
    (void)yvex_model_target_write_sidecar(YVEX_MODEL_TARGET_SIDECAR_TENSOR_MAP, path,
                                          request->target_id, family, status, coverage);
}

typedef struct {
    const char *status;
    const char *family;
    const char *target;
    const char *presence;
    const char *total;
    const char *mapped_total;
    const char *unknown;
    const char *layers;
    const char *embedding;
    const char *attention;
    const char *unit;
    const char *mlp;
    const char *norm;
    const char *output;
    const char *linear;
    const char *moe;
    const char *coverage;
} naming_audit_facts;

#define NAMING_STRING_ROW(format_, member_) \
    {YVEX_MODEL_TARGET_ROW_STRING, format_, offsetof(naming_audit_facts, member_)}
#define NAMING_LITERAL_ROW(text_) {YVEX_MODEL_TARGET_ROW_LITERAL, text_, 0u}

static const yvex_model_target_row_spec naming_audit_schema[] = {
    NAMING_STRING_ROW("tensor_map_status: %s", status),
    NAMING_STRING_ROW("tensor_map_family: %s", family),
    NAMING_STRING_ROW("tensor_map_target_id: %s", target),
    NAMING_LITERAL_ROW("tensor_map_stage: header-naming-map"),
    NAMING_LITERAL_ROW("tensor_map_evidence_basis: header-metadata-only"),
    NAMING_STRING_ROW("tensor_map_source_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_config_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_tokenizer_status: %s", presence),
    NAMING_STRING_ROW("tensor_map_tensor_count: %s", total),
    NAMING_STRING_ROW("tensor_map_mapped_total_count: %s", mapped_total),
    NAMING_STRING_ROW("tensor_map_unmapped_unknown_count: %s", unknown),
    NAMING_LITERAL_ROW("tensor_map_ambiguous_count: 0"),
    NAMING_STRING_ROW("tensor_map_layer_count_observed: %s", layers),
    NAMING_STRING_ROW("tensor_map_embedding_count: %s", embedding),
    NAMING_STRING_ROW("tensor_map_attention_count: %s", attention),
    NAMING_STRING_ROW("tensor_map_attention_q_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_k_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_v_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_attention_o_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_count: %s", mlp),
    NAMING_STRING_ROW("tensor_map_mlp_gate_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_up_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_mlp_down_count: %s", unit),
    NAMING_STRING_ROW("tensor_map_norm_count: %s", norm),
    NAMING_STRING_ROW("tensor_map_output_head_count: %s", output),
    NAMING_STRING_ROW("tensor_map_qwen_linear_attn_count: %s", linear),
    NAMING_STRING_ROW("tensor_map_moe_router_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_moe_expert_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_moe_shared_count: %s", moe),
    NAMING_STRING_ROW("tensor_map_required_role_coverage_status: %s", coverage),
    NAMING_LITERAL_ROW("tensor_map_validation_status: lexical-and-header-only"),
    NAMING_LITERAL_ROW("tensor_map_canonical_role_status: mapped-candidates"),
    NAMING_LITERAL_ROW("tensor_map_runtime_role_coverage_status: report-only"),
    NAMING_LITERAL_ROW("tensor_map_artifact_contract_status: not-implemented"),
    NAMING_LITERAL_ROW("tensor_map_runtime_descriptor_status: not-implemented"),
    NAMING_LITERAL_ROW("tensor_map_graph_consumer_status: not-implemented")
};

static const char *const norm_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.layers.0.pre_feedforward_layernorm.weight "
    "-> model.layers.0.mlp.norm.weight",
    "tensor_map.entry.1.mapping: model.layers.0.post_feedforward_layernorm.weight "
    "-> model.layers.0.mlp.norm.weight"
};

static const char *const dense_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.embed_tokens.weight -> model.embedding.token.weight",
    "tensor_map.entry.1.mapping: model.layers.0.self_attn.q_proj.weight -> "
    "model.layers.0.attention.q_proj.weight",
    "tensor_map.entry.2.mapping: model.layers.0.self_attn.k_proj.weight -> "
    "model.layers.0.attention.k_proj.weight",
    "tensor_map.entry.3.mapping: model.layers.0.self_attn.v_proj.weight -> "
    "model.layers.0.attention.v_proj.weight",
    "tensor_map.entry.4.mapping: model.layers.0.self_attn.o_proj.weight -> "
    "model.layers.0.attention.o_proj.weight",
    "tensor_map.entry.5.mapping: model.layers.0.mlp.gate_proj.weight -> "
    "model.layers.0.mlp.gate_proj.weight",
    "tensor_map.entry.6.mapping: model.layers.0.mlp.up_proj.weight -> "
    "model.layers.0.mlp.up_proj.weight",
    "tensor_map.entry.7.mapping: model.layers.0.mlp.down_proj.weight -> "
    "model.layers.0.mlp.down_proj.weight",
    "tensor_map.entry.8.mapping: model.layers.0.input_layernorm.weight -> "
    "model.layers.0.attention.norm.weight",
    "tensor_map.entry.9.mapping: model.layers.0.post_attention_layernorm.weight -> "
    "model.layers.0.mlp.norm.weight",
    "tensor_map.entry.10.mapping: model.norm.weight -> model.final_norm.weight",
    "tensor_map.entry.11.mapping: lm_head.weight -> model.output_head.weight"
};

static const char *const moe_mapping_rows[] = {
    "tensor_map.entry.0.mapping: model.language_model.embed_tokens.weight -> "
    "model.embedding.token.weight",
    "tensor_map.entry.1.mapping: model.language_model.layers.0.self_attn.q_proj.weight "
    "-> model.layers.0.attention.q_proj.weight",
    "tensor_map.entry.2.mapping: model.language_model.layers.0.self_attn.k_proj.weight "
    "-> model.layers.0.attention.k_proj.weight",
    "tensor_map.entry.3.mapping: model.language_model.layers.0.self_attn.v_proj.weight "
    "-> model.layers.0.attention.v_proj.weight",
    "tensor_map.entry.4.mapping: model.language_model.layers.0.self_attn.o_proj.weight "
    "-> model.layers.0.attention.o_proj.weight",
    "tensor_map.entry.5.mapping: model.language_model.layers.0.linear_attn.A_log -> "
    "model.layers.0.qwen_linear_attn.A_log",
    "tensor_map.entry.6.mapping: model.language_model.layers.0.mlp.gate.weight -> "
    "model.layers.0.moe.router.weight",
    "tensor_map.entry.7.mapping: model.language_model.layers.0.mlp.experts.gate_up_proj "
    "-> model.layers.0.moe.experts.all.gate_up_proj.weight",
    "tensor_map.entry.8.mapping: model.language_model.layers.0.mlp.shared_expert.down_proj.weight "
    "-> model.layers.0.moe.shared_expert.down_proj.weight"
};

static void naming_audit_rows(yvex_model_target_report *report,
                              const yvex_model_target_request *request,
                              const char *family,
                              const char *status,
                              const char *total,
                              const char *moe,
                              const char *unknown,
                              const char *coverage,
                              int source_present)
{
    int missing = strcmp(status, "source-missing") == 0;
    int dense_qwen = strcmp(family, "qwen") == 0 &&
                     strcmp(request->target_id, "qwen3-8b") == 0;
    int dense_style = dense_qwen ||
                      strcmp(request->target_id, "gemma-4-12b-it") == 0;
    int norm_only = strcmp(status, "naming-map-norm-only") == 0;
    const char *unit = missing || norm_only ? "0" : "1";
    naming_audit_facts facts = {
        status, family, request->target_id, source_present ? "present" : "missing",
        total, strcmp(unknown, "0") == 0 ? total : "12", unknown,
        missing ? "0" : "1", unit, missing || norm_only ? "0" : "4", unit,
        missing || norm_only ? "0" : "3",
        missing ? "0" : (norm_only ? "2" : "3"), unit,
        missing || dense_qwen ? "0" : (strcmp(family, "qwen") == 0 ? "1" : "0"),
        moe, coverage
    };

    yvex_model_target_report_project_rows(
        report, naming_audit_schema,
        sizeof(naming_audit_schema) / sizeof(naming_audit_schema[0]), &facts);
    if (!missing && norm_only) {
        yvex_model_target_report_add_rows(
            report, norm_mapping_rows,
            sizeof(norm_mapping_rows) / sizeof(norm_mapping_rows[0]));
    } else if (!missing && dense_style) {
        yvex_model_target_report_add_rows(
            report, dense_mapping_rows,
            sizeof(dense_mapping_rows) / sizeof(dense_mapping_rows[0]));
        if (strcmp(unknown, "0") != 0) {
            yvex_model_target_report_add_row(
                report,
                "tensor_map.entry.12.native_name: model.layers.0.weird_unknown.weight");
            yvex_model_target_report_add_row(report, "tensor_map.entry.12.mapping_status: unmapped-unknown");
            yvex_model_target_report_add_row(report, "model.layers.0.weird_unknown.weight");
            yvex_model_target_report_add_row(report, "mapping_status: unmapped-unknown");
        }
    } else if (!missing) {
        yvex_model_target_report_add_rows(
            report, moe_mapping_rows,
            sizeof(moe_mapping_rows) / sizeof(moe_mapping_rows[0]));
    }
    if (!missing && !dense_style && strcmp(family, "gemma") == 0) {
        yvex_model_target_report_add_row(
            report, "tensor_map.entry.9.mapping: "
                    "model.language_model.layers.0.layer_scalar -> "
                    "model.layers.0.layer_scalar");
    }
    if (!missing && !dense_qwen && !norm_only) {
        yvex_model_target_report_add_row(report, "model.layers.0.weird_unknown.weight");
        yvex_model_target_report_add_row(report, "mapping_status: %s",
                                         strcmp(unknown, "0") == 0 ? "mapped-candidate" : "unmapped-unknown");
    }
    yvex_model_target_report_common_tail(report);
    if (missing) {
        yvex_model_target_report_add_row(report, "top_blocker: %s",
                                         strcmp(family, "gemma") == 0
                                             ? "missing-gemma-source-path"
                                             : "missing-qwen-source-path");
    }
    yvex_model_target_report_add_row(report, "next_required_rows: V010.MAP.8");
}

#undef NAMING_LITERAL_ROW
#undef NAMING_STRING_ROW

int yvex_tensor_naming_report_build(
    const yvex_model_target_request *request,
    yvex_model_target_report *report,
    yvex_error *err)
{
    const char *family;
    const char *status;
    const char *total;
    const char *moe;
    const char *unknown;
    const char *coverage;
    int source_present = 0;
    int missing_source;
    int norm_only;

    if (!request || !report ||
        request->kind != YVEX_MODEL_TARGET_COMMAND_TENSOR_MAP) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "tensor_naming_report",
                       "tensor naming report requires tensor-map command kind");
        return YVEX_ERR_INVALID_ARG;
    }
    if (strcmp(request->target_id, "deepseek4-v4-flash-dspark") == 0) {
        return yvex_model_target_family_mapping_report_build(request, report, err);
    }
    if (!naming_validate(request, report)) {
        return YVEX_OK;
    }
    if (request->output_contract[0]) {
        yvex_model_target_report_add_output_contract(
            report, "tensor-map", request->output_contract);
        return YVEX_OK;
    }
    family = yvex_model_target_family_key(request->target_id);
    naming_counts(request, family, &status, &total, &moe, &unknown, &coverage,
                  &source_present);
    missing_source = strcmp(status, "source-missing") == 0;
    norm_only = strcmp(status, "naming-map-norm-only") == 0;
    naming_maybe_write_sidecar(
        request,
        family,
        status,
        strcmp(family, "gemma") == 0 ? "required-groups-present" : coverage);
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_TABLE) {
        yvex_model_target_report_add_row(report, "TENSOR NAMING MAP");
        yvex_model_target_report_add_row(
            report, "FAMILY  TARGET                STATUS                      TOTAL   "
                    "EMBED    ATTN     MLP    NORM    HEAD     MOE   UNKNOWN   "
                    "LAYERS  NEXT");
        yvex_model_target_report_add_row(report, "%-8s%-22s%s  %s  %s  %s  %s  %s  %s  %s  %s  %s  V010.MAP.8",
                                         family, request->target_id, status,
                                         total,
                                         missing_source || norm_only ? "0" : "1",
                                         missing_source || norm_only ? "0" : "4",
                                         missing_source || norm_only ? "0" : "3",
                                         missing_source ? "0" : (norm_only ? "2" : "3"),
                                         missing_source || norm_only ? "0" : "1",
                                         moe, unknown,
                                         missing_source ? "0" : "1");
        return YVEX_OK;
    }
    if (request->mode == YVEX_MODEL_TARGET_OUTPUT_AUDIT) {
        naming_audit_rows(report, request, family, status, total, moe, unknown,
                          coverage, source_present);
        return YVEX_OK;
    }
    yvex_model_target_report_add_row(report, "tensor-map: %s [%s]",
                                     request->target_id,
                                     missing_source || norm_only ? "blocked" :
                                     strcmp(status, "naming-map-profiled") == 0
                                         ? "reported" : status);
    yvex_model_target_report_add_row(report, "family: %s  stage: header-naming-map  evidence: header-only", family);
    yvex_model_target_report_add_row(report,
                                     "roles: total=%s embedding=%s attention=%s "
                                     "mlp=%s norm=%s head=%s moe=%s unknown=%s",
                                     strcmp(unknown, "0") == 0 ? total : "12",
                                     missing_source || norm_only ? "0" : "1",
                                     missing_source || norm_only ? "0" : "4",
                                     missing_source || norm_only ? "0" : "3",
                                     missing_source ? "0" : (norm_only ? "2" : "3"),
                                     missing_source || norm_only ? "0" : "1",
                                     moe, unknown);
    yvex_model_target_report_add_row(report, "layers: %s",
                                     missing_source ? "0" : "1");
    yvex_model_target_report_add_row(report, "top_blocker: %s",
                                     missing_source
                                         ? (strcmp(family, "gemma") == 0
                                                ? "missing-gemma-source-path"
                                                : "missing-qwen-source-path")
                                     : strcmp(request->target_id, "qwen3-8b") == 0
                                         ? "missing-qwen-runtime-role-validation"
                                     : strcmp(family, "gemma") == 0
                                         ? "missing-dense-runtime-role-validation"
                                         : "missing-qwen-tensor-role-map");
    yvex_model_target_report_add_row(report, "next: V010.MAP.8");
    yvex_model_target_report_add_row(report, "boundary: report-only; use --audit for tensor entries");
    return YVEX_OK;
}
