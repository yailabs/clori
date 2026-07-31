/*
 * Compose inventory, descriptor, and materialization diagnostics for the finite CLI lane. Lower
 * owners retain every admission decision; this file only presents their typed facts.
 */
#include "src/cli/model_artifacts/private.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    char name[32];
    unsigned long long count;
    unsigned long long bytes;
} fullmodel_dtype_bucket;

int fullmodel_file_size(const char *path,
                               unsigned long long *bytes)
{
    struct stat st;

    if (bytes) *bytes = 0ull;
    if (!path || stat(path, &st) != 0) return 0;
    if (bytes) *bytes = (unsigned long long)st.st_size;
    return 1;
}

const char *fullmodel_family_from_arch(yvex_arch arch)
{
    switch (arch) {
    case YVEX_ARCH_DEEPSEEK: return "deepseek";
    case YVEX_ARCH_GLM: return "glm";
    case YVEX_ARCH_LLAMA: return "llama";
    case YVEX_ARCH_QWEN: return "qwen";
    case YVEX_ARCH_GEMMA: return "gemma";
    case YVEX_ARCH_PHI: return "phi";
    case YVEX_ARCH_KIMI: return "kimi";
    default: return "unknown";
    }
}

static const char *const literal_pair_0[] = { "fullmodel: report",
    "status: fullmodel-report"};

static void fullmodel_csv_append(char *buf,
                                 size_t cap,
                                 const char *item)
{
    size_t used;
    int n;

    if (!buf || cap == 0u || !item || !item[0]) return;
    used = strlen(buf);
    if (used >= cap - 1u) return;
    n = snprintf(buf + used, cap - used, "%s%s", used == 0u ? "" : ",", item);
    if (n < 0 || (size_t)n >= cap - used) buf[cap - 1u] = '\0';
}

static void fullmodel_collection_add(unsigned long long *count,
                                     unsigned long long *bytes,
                                     const yvex_tensor_info *tensor)
{
    if (count) (*count)++;
    if (bytes && tensor) *bytes += tensor->storage_bytes;
}

typedef struct tensor_collection_rule {
    yvex_tensor_role role;
    const char *name_a;
    const char *name_b;
    int names_must_both_match;
    int name_a_must_equal;
    size_t count_offset;
    size_t bytes_offset;
    size_t flag_offset;
} tensor_collection_rule;

#define COLLECTION_RULE(role_, a_, b_, both_, exact_, count_, flag_) \
    {role_, a_, b_, both_, exact_, offsetof(yvex_fullmodel_collections, count_), \
     offsetof(yvex_fullmodel_collections, count_##_bytes), \
     offsetof(yvex_fullmodel_collections, flag_)}

static const tensor_collection_rule typed_collection_rules[] = {
    COLLECTION_RULE(YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, NULL, NULL, 0, 0,
                    embedding, has_token_embedding),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_OUTPUT_NORM, NULL, NULL, 0, 0,
                    normalization, has_output_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_OUTPUT_HEAD, NULL, NULL, 0, 0,
                    output, has_output_head),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_ATTENTION_NORM, NULL, NULL, 0, 0,
                    normalization, has_attention_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_ATTENTION_Q, NULL, NULL, 0, 0,
                    attention, has_attention_q),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_ATTENTION_K, NULL, NULL, 0, 0,
                    attention, has_attention_k),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_ATTENTION_V, NULL, NULL, 0, 0,
                    attention, has_attention_v),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_ATTENTION_OUT, NULL, NULL, 0, 0,
                    attention, has_attention_out),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_FFN_NORM, NULL, NULL, 0, 0,
                    normalization, has_post_attention_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_FFN_GATE, NULL, NULL, 0, 0, mlp, has_ffn_gate),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_FFN_UP, NULL, NULL, 0, 0, mlp, has_ffn_up),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_FFN_DOWN, NULL, NULL, 0, 0, mlp, has_ffn_down),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_MOE_ROUTER, NULL, NULL, 0, 0, moe, has_moe_router),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, NULL, NULL, 0, 0,
                    moe, has_moe_expert),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_MOE_EXPERT_UP, NULL, NULL, 0, 0, moe, has_moe_expert),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN, NULL, NULL, 0, 0,
                    moe, has_moe_expert),
};

static const tensor_collection_rule named_collection_rules[] = {
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "token_embd", "embed", 0, 0,
                    embedding, has_token_embedding),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "attn_norm", "input_layernorm", 0, 0,
                    normalization, has_attention_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "ffn_norm", "post_attention_layernorm", 0, 0,
                    normalization, has_post_attention_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "attn_q", "q_proj", 0, 0,
                    attention, has_attention_q),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "attn_k", "k_proj", 0, 0,
                    attention, has_attention_k),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "attn_v", "v_proj", 0, 0,
                    attention, has_attention_v),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "attn_output", "o_proj", 0, 0,
                    attention, has_attention_out),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "ffn_gate", "gate_proj", 0, 0,
                    mlp, has_ffn_gate),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "ffn_up", "up_proj", 0, 0, mlp, has_ffn_up),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "ffn_down", "down_proj", 0, 0,
                    mlp, has_ffn_down),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "router", "gate.weight", 0, 0,
                    moe, has_moe_router),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "expert", NULL, 0, 0, moe, has_moe_expert),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "output_norm", "norm.weight", 0, 0,
                    normalization, has_output_norm),
    COLLECTION_RULE(YVEX_TENSOR_ROLE_UNKNOWN, "output.weight", "lm_head", 0, 1,
                    output, has_output_head),
};

typedef struct descriptor_tensor_rule {
    const char *logical_role;
    yvex_tensor_role typed_role;
    const char *name_a;
    const char *name_b;
    int names_must_both_match;
    int name_a_must_equal;
} descriptor_tensor_rule;

#define DESCRIPTOR_RULE(name_, role_, a_, b_, both_, exact_) \
    {name_, role_, a_, b_, both_, exact_}

static const descriptor_tensor_rule descriptor_tensor_rules[] = {
    DESCRIPTOR_RULE("token_embedding", YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
                    "token_embd", "embed", 0, 0),
    DESCRIPTOR_RULE("attention_norm", YVEX_TENSOR_ROLE_ATTENTION_NORM,
                    "attn_norm", "input_layernorm", 0, 0),
    DESCRIPTOR_RULE("post_attention_norm", YVEX_TENSOR_ROLE_FFN_NORM,
                    "ffn_norm", "post_attention_layernorm", 0, 0),
    DESCRIPTOR_RULE("final_norm", YVEX_TENSOR_ROLE_OUTPUT_NORM,
                    "output_norm", "final_norm", 0, 0),
    DESCRIPTOR_RULE("q_projection", YVEX_TENSOR_ROLE_ATTENTION_Q, "attn_q", "q_proj", 0, 0),
    DESCRIPTOR_RULE("k_projection", YVEX_TENSOR_ROLE_ATTENTION_K, "attn_k", "k_proj", 0, 0),
    DESCRIPTOR_RULE("v_projection", YVEX_TENSOR_ROLE_ATTENTION_V, "attn_v", "v_proj", 0, 0),
    DESCRIPTOR_RULE("o_projection", YVEX_TENSOR_ROLE_ATTENTION_OUT,
                    "attn_output", "o_proj", 0, 0),
    DESCRIPTOR_RULE("mlp_gate", YVEX_TENSOR_ROLE_FFN_GATE, "ffn_gate", "gate_proj", 0, 0),
    DESCRIPTOR_RULE("mlp_up", YVEX_TENSOR_ROLE_FFN_UP, "ffn_up", "up_proj", 0, 0),
    DESCRIPTOR_RULE("mlp_down", YVEX_TENSOR_ROLE_FFN_DOWN, "ffn_down", "down_proj", 0, 0),
    DESCRIPTOR_RULE("moe_router", YVEX_TENSOR_ROLE_MOE_ROUTER, "router", NULL, 0, 0),
    DESCRIPTOR_RULE("moe_expert_gate", YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
                    "expert", "gate", 1, 0),
    DESCRIPTOR_RULE("moe_expert_up", YVEX_TENSOR_ROLE_MOE_EXPERT_UP,
                    "expert", "up", 1, 0),
    DESCRIPTOR_RULE("moe_expert_down", YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
                    "expert", "down", 1, 0),
    DESCRIPTOR_RULE("output_head", YVEX_TENSOR_ROLE_OUTPUT_HEAD,
                    "output.weight", "lm_head", 0, 1),
    DESCRIPTOR_RULE("unknown", YVEX_TENSOR_ROLE_UNKNOWN, NULL, NULL, 0, 0),
};

static const yvex_tensor_role materialize_typed_roles[] = {
    YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
    YVEX_TENSOR_ROLE_OUTPUT_NORM,
    YVEX_TENSOR_ROLE_OUTPUT_HEAD,
    YVEX_TENSOR_ROLE_ATTENTION_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_Q,
    YVEX_TENSOR_ROLE_ATTENTION_K,
    YVEX_TENSOR_ROLE_ATTENTION_V,
    YVEX_TENSOR_ROLE_ATTENTION_OUT,
    YVEX_TENSOR_ROLE_FFN_NORM,
    YVEX_TENSOR_ROLE_FFN_GATE,
    YVEX_TENSOR_ROLE_FFN_UP,
    YVEX_TENSOR_ROLE_FFN_DOWN,
};

static const char *const materialize_name_fragments[] = {
    "token_embd", "attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v",
    "attn_output", "q_proj", "k_proj", "v_proj", "o_proj", "ffn_gate",
    "ffn_up", "ffn_down", "gate_proj", "up_proj", "down_proj", "output_norm",
    "lm_head",
};

static const char *const detectable_families[] = {"deepseek", "glm", "qwen"};

#undef DESCRIPTOR_RULE

#undef COLLECTION_RULE

static int tensor_collection_name_matches(const tensor_collection_rule *rule,
                                          const char *name)
{
    int a = rule->name_a_must_equal ? strcmp(name, rule->name_a) == 0
                                    : model_download_name_contains(name, rule->name_a);
    int b = rule->name_b && model_download_name_contains(name, rule->name_b);

    return rule->names_must_both_match ? a && b : a || b;
}

static int descriptor_tensor_rule_matches(const descriptor_tensor_rule *rule,
                                          const yvex_tensor_info *tensor,
                                          const char *name)
{
    int a;
    int b;

    if (tensor->role == rule->typed_role) return 1;
    if (!rule->name_a) return 0;
    a = rule->name_a_must_equal ? strcmp(name, rule->name_a) == 0
                                : model_download_name_contains(name, rule->name_a);
    b = rule->name_b && model_download_name_contains(name, rule->name_b);
    return rule->names_must_both_match ? a && b : a || b;
}

static void tensor_collection_apply(const tensor_collection_rule *rule,
                                    const yvex_tensor_info *tensor,
                                    yvex_fullmodel_collections *collections)
{
    unsigned char *base = (unsigned char *)collections;

    fullmodel_collection_add((unsigned long long *)(base + rule->count_offset),
                             (unsigned long long *)(base + rule->bytes_offset), tensor);
    *(int *)(base + rule->flag_offset) = 1;
}

static void fullmodel_record_dtype(fullmodel_dtype_bucket buckets[32],
                                   unsigned int *bucket_count,
                                   const yvex_tensor_info *tensor)
{
    const char *name;
    unsigned int i;

    if (!buckets || !bucket_count || !tensor) return;
    name = yvex_dtype_name(tensor->dtype);
    for (i = 0; i < *bucket_count; ++i) {
        if (strcmp(buckets[i].name, name) == 0) {
            buckets[i].count++;
            buckets[i].bytes += tensor->storage_bytes;
            return;
        }
    }
    if (*bucket_count < 32u) {
        snprintf(buckets[*bucket_count].name, sizeof(buckets[*bucket_count].name), "%s", name);
        buckets[*bucket_count].count = 1ull;
        buckets[*bucket_count].bytes = tensor->storage_bytes;
        (*bucket_count)++;
    }
}

static void fullmodel_dtype_summary(char *out,
                                    size_t out_cap,
                                    const fullmodel_dtype_bucket buckets[32],
                                    unsigned int bucket_count)
{
    unsigned int i;
    size_t used = 0u;

    if (!out || out_cap == 0u) return;
    out[0] = '\0';
    for (i = 0; i < bucket_count; ++i) {
        int n = snprintf(out + used,
                         used < out_cap ? out_cap - used : 0u,
                         "%s%s:%llu:%llu",
                         i == 0 ? "" : ",",
                         buckets[i].name,
                         buckets[i].count,
                         buckets[i].bytes);
        if (n < 0 || (size_t)n >= (used < out_cap ? out_cap - used : 0u)) {
            out[out_cap - 1u] = '\0';
            return;
        }
        used += (size_t)n;
    }
    if (bucket_count == 0u) snprintf(out, out_cap, "none");
}

static void fullmodel_record_largest(fullmodel_largest_tensor top[16],
                                     unsigned int *top_count,
                                     unsigned int limit,
                                     const yvex_tensor_info *tensor)
{
    unsigned int i;
    unsigned int pos;

    if (!top || !top_count || !tensor || limit == 0u) return;
    if (limit > 16u) limit = 16u;
    pos = *top_count;
    for (i = 0; i < *top_count; ++i) {
        if (tensor->storage_bytes > top[i].bytes) {
            pos = i;
            break;
        }
    }
    if (*top_count < limit) {
        (*top_count)++;
    } else if (pos >= limit) {
        return;
    }
    for (i = *top_count - 1u; i > pos; --i) {
        top[i] = top[i - 1u];
    }
    top[pos].tensor = tensor;
    top[pos].bytes = tensor->storage_bytes;
}

void fullmodel_classify_tensor(const yvex_tensor_info *tensor,
                                      yvex_fullmodel_collections *collections)
{
    const char *name;
    size_t i;

    if (!tensor || !collections) return;
    name = tensor->name ? tensor->name : "";
    for (i = 0; i < sizeof(typed_collection_rules) / sizeof(typed_collection_rules[0]); ++i) {
        if (typed_collection_rules[i].role == tensor->role) {
            tensor_collection_apply(&typed_collection_rules[i], tensor, collections);
            return;
        }
    }
    for (i = 0; i < sizeof(named_collection_rules) / sizeof(named_collection_rules[0]); ++i) {
        if (tensor_collection_name_matches(&named_collection_rules[i], name)) {
            tensor_collection_apply(&named_collection_rules[i], tensor, collections);
            return;
        }
    }
    fullmodel_collection_add(&collections->unknown, &collections->unknown_bytes, tensor);
}

int fullmodel_is_selected_target(const char *text)
{
    return text &&
           (strcmp(text, "deepseek4-v4-flash-selected-embed") == 0 ||
            strcmp(text, "deepseek4-v4-flash-selected-embed-rmsnorm") == 0);
}

static int fullmodel_descriptor_tensor_matches(const yvex_tensor_info *tensor,
                                               const char *role)
{
    const char *name;
    size_t i;

    if (!tensor || !role) return 0;
    name = tensor->name ? tensor->name : "";
    for (i = 0; i < sizeof(descriptor_tensor_rules) / sizeof(descriptor_tensor_rules[0]); ++i) {
        if (strcmp(role, descriptor_tensor_rules[i].logical_role) == 0) {
            return descriptor_tensor_rule_matches(&descriptor_tensor_rules[i], tensor, name);
        }
    }
    return 0;
}

const yvex_tensor_info *fullmodel_descriptor_find_tensor(yvex_model_context *ctx,
                                                                const char *role)
{
    unsigned long long count;
    unsigned long long i;

    if (!ctx || !ctx->table || !role) return NULL;
    count = yvex_tensor_table_count(ctx->table);
    for (i = 0; i < count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(ctx->table, i);
        if (fullmodel_descriptor_tensor_matches(tensor, role)) return tensor;
    }
    return NULL;
}

const char *fullmodel_detect_family(const yvex_cli_fullmodel_options *options,
                                           yvex_arch arch,
                                           const char *target_id)
{
    const char *from_arch = fullmodel_family_from_arch(arch);
    size_t index;

    if (from_arch && strcmp(from_arch, "unknown") != 0) return from_arch;
    for (index = 0; index < sizeof(detectable_families) / sizeof(detectable_families[0]); ++index) {
        const char *family = detectable_families[index];
        if ((target_id && model_download_name_contains(target_id, family)) ||
            (options && options->model && model_download_name_contains(options->model, family))) {
            return family;
        }
    }
    return "unknown";
}

int fullmodel_family_request_matches(const char *requested,
                                            const char *detected)
{
    if (!requested || !requested[0] || strcmp(requested, "auto") == 0) {
        return detected && strcmp(detected, "unknown") != 0;
    }
    return detected && strcmp(requested, detected) == 0;
}

const char *fullmodel_role_status_from_tensor(yvex_model_context *ctx,
                                                     const yvex_fullmodel_collections *collections,
                                                     const char *role)
{
    if (role && strcmp(role, "tokenizer_metadata") == 0) {
        return collections && collections->has_tokenizer_metadata ? "present" : "missing";
    }
    return fullmodel_descriptor_find_tensor(ctx, role) ? "present" : "missing";
}

const char *fullmodel_identity_status(const yvex_model_ref *ref,
                                             unsigned long long artifact_bytes)
{
    if (!ref || ref->kind != YVEX_MODEL_REF_ALIAS) return "not-checked";
    if (!ref->sha256 || !ref->sha256[0]) return "registered-without-digest";
    if (ref->registered_file_size != 0ull && ref->registered_file_size != artifact_bytes) {
        return "registered-size-drift";
    }
    return "registered-size-match";
}

void fullmodel_probe_backend_fit(const char *backend,
                                        unsigned long long required_bytes,
                                        yvex_fullmodel_backend_fit *fit)
{
    yvex_backend *opened = NULL;
    yvex_backend_options options;
    yvex_backend_memory_stats stats;
    yvex_backend_device_info device_info;
    yvex_error err;
    int rc;

    if (!fit) return;
    memset(fit, 0, sizeof(*fit));
    fit->required_bytes = required_bytes;
    fit->fit_status = "unknown";
    snprintf(fit->fit_reason, sizeof(fit->fit_reason),
             "system memory availability is not queried");

    if (!backend || strcmp(backend, "cpu") == 0) {
        fit->available = 1;
        return;
    }

    memset(&options, 0, sizeof(options));
    memset(&stats, 0, sizeof(stats));
    memset(&device_info, 0, sizeof(device_info));
    yvex_error_clear(&err);
    options.kind = YVEX_BACKEND_KIND_CUDA;
    rc = yvex_backend_open(&opened, &options, &err);
    if (rc != YVEX_OK) {
        fit->available = 0;
        fit->fit_status = "unsupported";
        snprintf(fit->fit_reason, sizeof(fit->fit_reason),
                 "CUDA backend unavailable: %s", yvex_error_message(&err));
        yvex_error_clear(&err);
        return;
    }

    fit->available = 1;
    rc = yvex_backend_get_memory_stats(opened, &stats, &err);
    if (rc == YVEX_OK &&
        yvex_backend_get_device_info(opened, &device_info, &err) == YVEX_OK &&
        device_info.total_memory_bytes > 0ull) {
        fit->memory_known = 1;
        fit->total_bytes = device_info.total_memory_bytes;
        fit->available_bytes = device_info.free_memory_bytes;
        if (required_bytes <= device_info.free_memory_bytes) {
            fit->fit_status = "fits";
            snprintf(fit->fit_reason, sizeof(fit->fit_reason),
                     "required bytes fit current CUDA free memory; no allocation attempted");
        } else {
            fit->fit_status = "does-not-fit";
            snprintf(fit->fit_reason, sizeof(fit->fit_reason),
                     "required bytes exceed current CUDA free memory; no allocation attempted");
        }
    } else {
        fit->fit_status = "unknown";
        snprintf(fit->fit_reason, sizeof(fit->fit_reason),
                 "CUDA memory info unavailable; no allocation attempted");
        yvex_error_clear(&err);
    }
    yvex_backend_close(opened);
}

int fullmodel_has_attention_collection(const yvex_fullmodel_collections *collections)
{
    return collections &&
           collections->has_attention_q &&
           collections->has_attention_k &&
           collections->has_attention_v &&
           collections->has_attention_out;
}

int fullmodel_has_mlp_collection(const yvex_fullmodel_collections *collections)
{
    return collections &&
           collections->has_ffn_gate &&
           collections->has_ffn_up &&
           collections->has_ffn_down;
}

int fullmodel_has_normalization_collection(const yvex_fullmodel_collections *collections)
{
    return collections &&
           (collections->has_attention_norm ||
            collections->has_post_attention_norm ||
            collections->has_output_norm);
}

static int fullmodel_tensor_is_materialize_required(const yvex_tensor_info *tensor)
{
    const char *name;
    size_t index;

    if (!tensor) return 0;
    name = tensor->name ? tensor->name : "";
    for (index = 0; index < sizeof(materialize_typed_roles) / sizeof(materialize_typed_roles[0]); ++index) {
        if (tensor->role == materialize_typed_roles[index]) return 1;
    }
    if (strcmp(name, "output.weight") == 0) return 1;
    for (index = 0; index < sizeof(materialize_name_fragments) /
                                  sizeof(materialize_name_fragments[0]); ++index) {
        if (model_download_name_contains(name, materialize_name_fragments[index])) return 1;
    }
    return 0;
}

typedef struct {
    const char *name;
    size_t offset;
} fullmodel_role_presence;

#define ROLE_PRESENCE(name_, member_) {name_, offsetof(yvex_fullmodel_collections, member_)}

static const fullmodel_role_presence fullmodel_role_presence_table[] = {
    ROLE_PRESENCE("token-embedding", has_token_embedding),
    ROLE_PRESENCE("attention-norm", has_attention_norm),
    ROLE_PRESENCE("post-attention-norm", has_post_attention_norm),
    ROLE_PRESENCE("attention-q-projection", has_attention_q),
    ROLE_PRESENCE("attention-k-projection", has_attention_k),
    ROLE_PRESENCE("attention-v-projection", has_attention_v),
    ROLE_PRESENCE("attention-output-projection", has_attention_out),
    ROLE_PRESENCE("mlp-gate", has_ffn_gate),
    ROLE_PRESENCE("mlp-up", has_ffn_up),
    ROLE_PRESENCE("mlp-down", has_ffn_down),
    ROLE_PRESENCE("moe-router", has_moe_router),
    ROLE_PRESENCE("moe-experts", has_moe_expert),
    ROLE_PRESENCE("final-norm", has_output_norm),
    ROLE_PRESENCE("output-head", has_output_head),
    ROLE_PRESENCE("tokenizer-metadata", has_tokenizer_metadata),
};

static const char *const fullmodel_required_roles[] = {
    "token-embedding", "attention-norm", "post-attention-norm",
    "attention-q-projection", "attention-k-projection", "attention-v-projection",
    "attention-output-projection", "mlp-gate", "mlp-up", "mlp-down", "final-norm",
    "output-head", "tokenizer-metadata",
};

typedef enum {
    FULLMODEL_COLLECTION_COUNT,
    FULLMODEL_COLLECTION_ATTENTION,
    FULLMODEL_COLLECTION_MLP,
    FULLMODEL_COLLECTION_TOKENIZER,
} fullmodel_collection_rule;

typedef struct {
    const char *name;
    size_t offset;
    fullmodel_collection_rule rule;
} fullmodel_collection_presence;

static const fullmodel_collection_presence fullmodel_collection_presence_table[] = {
    {"embedding", offsetof(yvex_fullmodel_collections, embedding), FULLMODEL_COLLECTION_COUNT},
    {"normalization", offsetof(yvex_fullmodel_collections, normalization), FULLMODEL_COLLECTION_COUNT},
    {"attention", 0u, FULLMODEL_COLLECTION_ATTENTION},
    {"mlp", 0u, FULLMODEL_COLLECTION_MLP},
    {"moe", offsetof(yvex_fullmodel_collections, moe), FULLMODEL_COLLECTION_COUNT},
    {"output", offsetof(yvex_fullmodel_collections, output), FULLMODEL_COLLECTION_COUNT},
    {"tokenizer", 0u, FULLMODEL_COLLECTION_TOKENIZER},
    {"tokenizer-runtime-input", 0u, FULLMODEL_COLLECTION_TOKENIZER},
};

#undef ROLE_PRESENCE

static int fullmodel_role_present(const yvex_fullmodel_collections *collections,
                                  const char *role)
{
    size_t index;

    if (!collections || !role) return 0;
    for (index = 0; index < sizeof(fullmodel_role_presence_table) /
                                  sizeof(fullmodel_role_presence_table[0]); ++index) {
        if (strcmp(role, fullmodel_role_presence_table[index].name) == 0)
            return *(const int *)((const unsigned char *)collections +
                                  fullmodel_role_presence_table[index].offset);
    }
    return 0;
}

static int fullmodel_collection_present_by_name(const yvex_fullmodel_collections *collections,
                                                const char *collection)
{
    size_t index;

    if (!collections || !collection) return 0;
    for (index = 0; index < sizeof(fullmodel_collection_presence_table) /
                                  sizeof(fullmodel_collection_presence_table[0]); ++index) {
        const fullmodel_collection_presence *spec = &fullmodel_collection_presence_table[index];
        if (strcmp(collection, spec->name) != 0) continue;
        if (spec->rule == FULLMODEL_COLLECTION_ATTENTION)
            return fullmodel_has_attention_collection(collections);
        if (spec->rule == FULLMODEL_COLLECTION_MLP)
            return fullmodel_has_mlp_collection(collections);
        if (spec->rule == FULLMODEL_COLLECTION_TOKENIZER)
            return collections->has_tokenizer_metadata;
        return *(const unsigned long long *)((const unsigned char *)collections + spec->offset) > 0ull;
    }
    return 0;
}

static void fullmodel_materialize_missing_roles(const yvex_cli_fullmodel_options *options,
                                                const yvex_fullmodel_collections *collections,
                                                char *out,
                                                size_t out_cap)
{
    size_t index;

    if (!out || out_cap == 0u) return;
    out[0] = '\0';
    for (index = 0; index < sizeof(fullmodel_required_roles) /
                                  sizeof(fullmodel_required_roles[0]); ++index) {
        if (!fullmodel_role_present(collections, fullmodel_required_roles[index]))
            fullmodel_csv_append(out, out_cap, fullmodel_required_roles[index]);
    }
    if (options && options->require_role &&
        !fullmodel_role_present(collections, options->require_role)) {
        fullmodel_csv_append(out, out_cap, options->require_role);
    }
    if (options && options->require_collection &&
        !fullmodel_collection_present_by_name(collections, options->require_collection)) {
        char tmp[160];
        snprintf(tmp, sizeof(tmp), "collection:%s", options->require_collection);
        fullmodel_csv_append(out, out_cap, tmp);
    }
    if (!out[0]) snprintf(out, out_cap, "none");
}

static int fullmodel_fail_after(const yvex_cli_fullmodel_options *options,
                                const char *phase)
{
    return options && options->fail_after_phase && phase &&
           strcmp(options->fail_after_phase, phase) == 0;
}

static int fullmodel_open_requested_backend(const char *backend_name,
                                            yvex_backend **out,
                                            yvex_error *err)
{
    yvex_backend_options options;

    if (!out) return YVEX_ERR_INVALID_ARG;
    *out = NULL;
    if (!backend_name || strcmp(backend_name, "cpu") == 0) {
        return yvex_backend_open_cpu(out, err);
    }
    memset(&options, 0, sizeof(options));
    options.kind = YVEX_BACKEND_KIND_CUDA;
    return yvex_backend_open(out, &options, err);
}

static int fullmodel_allocate_required_tensors(const yvex_cli_fullmodel_options *options,
                                               yvex_model_context *ctx,
                                               unsigned long long *materialized_count,
                                               unsigned long long *materialized_bytes,
                                               const char **failed_phase,
                                               const char **failed_reason)
{
    yvex_backend *backend = NULL;
    yvex_device_tensor **allocated = NULL;
    yvex_error err;
    unsigned long long tensor_count;
    unsigned long long i;
    unsigned long long allocated_count = 0ull;
    int rc;

    if (materialized_count) *materialized_count = 0ull;
    if (materialized_bytes) *materialized_bytes = 0ull;
    if (failed_phase) *failed_phase = "none";
    if (failed_reason) *failed_reason = "none";
    if (!options || !ctx || !ctx->table) return YVEX_ERR_INVALID_ARG;

    tensor_count = yvex_tensor_table_count(ctx->table);
    allocated = (yvex_device_tensor **)calloc((size_t)(tensor_count ? tensor_count : 1ull),
                                              sizeof(*allocated));
    if (!allocated) {
        if (failed_phase) *failed_phase = "backend-preflight";
        if (failed_reason) *failed_reason = "allocation-list";
        return YVEX_ERR_NOMEM;
    }

    yvex_error_clear(&err);
    rc = fullmodel_open_requested_backend(options->backend, &backend, &err);
    if (rc != YVEX_OK) {
        if (failed_phase) *failed_phase = "backend-preflight";
        if (failed_reason) *failed_reason = "backend-open-failed";
        yvex_error_clear(&err);
        free(allocated);
        return rc;
    }

    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(ctx->table, i);
        yvex_backend_tensor_desc desc;
        yvex_device_tensor *device_tensor = NULL;
        unsigned int d;

        if (!fullmodel_tensor_is_materialize_required(tensor)) continue;
        memset(&desc, 0, sizeof(desc));
        desc.name = tensor->name;
        desc.dtype = tensor->dtype;
        desc.rank = tensor->rank;
        desc.bytes = tensor->storage_bytes;
        for (d = 0; d < tensor->rank && d < YVEX_TENSOR_MAX_DIMS; ++d) {
            desc.dims[d] = tensor->dims[d];
        }
        rc = yvex_backend_tensor_alloc(backend, &desc, &device_tensor, &err);
        if (rc != YVEX_OK) {
            if (failed_phase) *failed_phase = "backend-preflight";
            if (failed_reason) *failed_reason = "tensor-allocation-failed";
            yvex_error_clear(&err);
            break;
        }
        allocated[allocated_count++] = device_tensor;
        if (materialized_count) (*materialized_count)++;
        if (materialized_bytes) *materialized_bytes += tensor->storage_bytes;
    }

    while (allocated_count > 0ull) {
        allocated_count--;
        yvex_backend_tensor_free(backend, allocated[allocated_count]);
    }
    yvex_backend_close(backend);
    free(allocated);
    return rc;
}

static const fullmodel_materialize_report materialize_report_defaults = {
    .status = "fullmodel-materialize-fail",
    .tensor_inventory_status = "pass",
    .unsupported_required_roles =
        "runtime-family-adapter,real-transformer-prefill,real-attention-backed-KV,real-DeepSeek-decode,"
        "real-output-head-logits,real-vocabulary-sampling",
    .placement_plan_status = "pass",
    .memory_budget_status = "pass",
    .backend_preflight_status = "pass",
    .materialization_mode = "controlled-fullmodel-proof",
    .full_model_materialization = "controlled-tiny-proof",
    .full_model_materialization_proof = "fail",
    .phase = "failed",
    .failed_phase = "none",
    .failed_reason = "none",
    .cleanup_attempted = "true",
    .cleanup_status = "pass",
    .cleanup_idempotent = "true",
    .owned_state_released = "true",
    .partial_materialization = "false",
    .runtime_blockers =
        "runtime family adapter not implemented; real transformer prefill unsupported; decode/logits/"
        "sampling/generation remain unsupported-full-model",
};

static void fullmodel_materialize_report_init(fullmodel_materialize_report *report,
                                              const yvex_cli_fullmodel_options *options,
                                              const yvex_model_ref *ref,
                                              const char *target_id,
                                              const char *target_class,
                                              unsigned long long artifact_bytes,
                                              unsigned long long required_count,
                                              unsigned long long required_bytes,
                                              const char *missing_roles,
                                              int selected_target,
                                              int role_complete)
{
    *report = materialize_report_defaults;
    report->options = options;
    report->model_resolved_path = ref && ref->path ? ref->path : "";
    report->target_id = target_id;
    report->target_class = target_class;
    report->artifact_identity_status = fullmodel_identity_status(ref, artifact_bytes);
    report->required_role_coverage = selected_target ? "partial" :
                                     role_complete ? "complete" : "partial";
    report->missing_required_roles = missing_roles;
    report->required_tensor_count = required_count;
    report->required_tensor_bytes = required_bytes;
    report->peak_planned_bytes = required_bytes;
    report->cpu_resident_bytes = strcmp(options->backend, "cuda") == 0 ? 0ull : required_bytes;
    report->cuda_resident_bytes = strcmp(options->backend, "cuda") == 0 ? required_bytes : 0ull;
    report->residency_plan = strcmp(options->backend, "cuda") == 0
                                 ? "cuda-resident-controlled-proof"
                                 : "cpu-resident-controlled-proof";
}

static int fullmodel_materialize_initial_fault(const yvex_cli_fullmodel_options *options,
                                               fullmodel_materialize_report *report)
{
    static const char *phases[] = {
        "preflight", "resolve-model", "artifact-identity", "tensor-inventory"
    };
    size_t i;

    for (i = 0; i < sizeof(phases) / sizeof(phases[0]); ++i) {
        if (!fullmodel_fail_after(options, phases[i])) continue;
        report->failed_phase = phases[i];
        report->failed_reason = "injected-failure";
        fullmodel_print_materialize_report(report);
        return exit_for_status(YVEX_ERR_STATE);
    }
    return 0;
}

static int fullmodel_materialize_command_run(const yvex_cli_fullmodel_options *options,
                                             yvex_model_ref *ref,
                                             yvex_model_context *ctx,
                                             const char *target_id,
                                             const char *target_class,
                                             unsigned long long artifact_bytes,
                                             unsigned long long tensor_count,
                                             unsigned long long total_tensor_bytes,
                                             const yvex_fullmodel_collections *collections,
                                             int selected_target)
{
    static const unsigned long long proof_byte_limit = 64ull * 1024ull * 1024ull;
    fullmodel_materialize_report report;
    char materialize_missing_roles[768];
    unsigned long long required_tensor_count = 0ull;
    unsigned long long required_tensor_bytes = 0ull;
    unsigned long long materialized_count = 0ull;
    unsigned long long materialized_bytes = 0ull;
    unsigned long long i;
    const char *alloc_failed_phase = "none";
    const char *alloc_failed_reason = "none";
    int role_complete;
    int rc;

    (void)total_tensor_bytes;
    memset(materialize_missing_roles, 0, sizeof(materialize_missing_roles));
    fullmodel_materialize_missing_roles(options,
                                        collections,
                                        materialize_missing_roles,
                                        sizeof(materialize_missing_roles));
    role_complete = strcmp(materialize_missing_roles, "none") == 0;

    for (i = 0; i < tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(ctx->table, i);
        if (!fullmodel_tensor_is_materialize_required(tensor)) continue;
        required_tensor_count++;
        required_tensor_bytes += tensor->storage_bytes;
    }

    fullmodel_materialize_report_init(&report, options, ref, target_id, target_class,
                                      artifact_bytes, required_tensor_count,
                                      required_tensor_bytes, materialize_missing_roles,
                                      selected_target, role_complete);
    rc = fullmodel_materialize_initial_fault(options, &report);
    if (rc != 0) return rc;

    if (selected_target) {
        report.status = "fullmodel-materialize-refused";
        report.placement_plan_status = "refused";
        report.memory_budget_status = "skipped";
        report.backend_preflight_status = "skipped";
        report.materialization_mode = "selected-runtime-slice-refusal";
        report.full_model_materialization = "refused-selected-runtime-slice";
        report.full_model_materialization_proof = "refused";
        report.failed_phase = "role-coverage";
        report.failed_reason = "selected-runtime-slice";
        report.refused_tensor_count = tensor_count;
        report.skipped_tensor_count = tensor_count;
        report.residency_plan = "selected-slice-not-full-model";
        report.runtime_blockers = "selected runtime slice cannot satisfy full required tensor materialization";
        fullmodel_print_materialize_report(&report);
        return 0;
    }

    if (!role_complete) {
        report.status = "fullmodel-materialize-refused";
        report.placement_plan_status = "partial";
        report.memory_budget_status = "skipped";
        report.backend_preflight_status = "skipped";
        report.materialization_mode = "role-coverage-refusal";
        report.full_model_materialization = "refused-incomplete-role-coverage";
        report.full_model_materialization_proof = "refused";
        report.failed_phase = "role-coverage";
        report.failed_reason = "required-role-missing";
        report.refused_tensor_count = tensor_count;
        report.skipped_tensor_count = tensor_count;
        report.residency_plan = "not-planned";
        report.runtime_blockers = "required fullmodel proof roles are missing";
        fullmodel_print_materialize_report(&report);
        return 0;
    }

    if (fullmodel_fail_after(options, "role-coverage")) {
        report.failed_phase = "role-coverage";
        report.failed_reason = "injected-failure";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_STATE);
    }
    if (fullmodel_fail_after(options, "placement-plan")) {
        report.failed_phase = "placement-plan";
        report.failed_reason = "injected-failure";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_STATE);
    }

    if ((options->has_limit_bytes && required_tensor_bytes > options->limit_bytes) ||
        (!options->has_limit_bytes && required_tensor_bytes > proof_byte_limit)) {
        report.status = "fullmodel-materialize-fail";
        report.memory_budget_status = "fail";
        report.backend_preflight_status = "skipped";
        report.materialization_mode = "memory-budget-refusal";
        report.full_model_materialization = "failed";
        report.full_model_materialization_proof = "fail";
        report.failed_phase = "memory-budget";
        report.failed_reason = options->has_limit_bytes ? "byte-limit" : "controlled-proof-limit";
        report.refused_tensor_count = required_tensor_count;
        report.skipped_tensor_count = required_tensor_count;
        report.residency_plan = "not-planned";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_NOMEM);
    }

    if (fullmodel_fail_after(options, "memory-budget")) {
        report.failed_phase = "memory-budget";
        report.failed_reason = "injected-failure";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_STATE);
    }

    if (options->dry_run || options->plan_only) {
        report.status = options->plan_only ? "fullmodel-materialize-plan-only" : "fullmodel-materialize-dry-run";
        report.backend_preflight_status = "skipped";
        report.materialization_mode = options->plan_only ? "plan-only" : "dry-run";
        report.full_model_materialization = "planned";
        report.full_model_materialization_proof = "planned";
        report.phase = options->plan_only ? "placement-plan" : "memory-budget";
        report.cleanup_attempted = "false";
        report.cleanup_status = "not-needed";
        report.skipped_tensor_count = required_tensor_count;
        fullmodel_print_materialize_report(&report);
        return 0;
    }

    if (fullmodel_fail_after(options, "backend-preflight")) {
        report.failed_phase = "backend-preflight";
        report.failed_reason = "injected-failure";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_STATE);
    }

    rc = fullmodel_allocate_required_tensors(options,
                                             ctx,
                                             &materialized_count,
                                             &materialized_bytes,
                                             &alloc_failed_phase,
                                             &alloc_failed_reason);
    if (rc != YVEX_OK) {
        report.status = "fullmodel-materialize-fail";
        report.backend_preflight_status = strcmp(alloc_failed_phase, "backend-preflight") == 0 ? "fail" : "partial";
        report.full_model_materialization = "failed";
        report.full_model_materialization_proof = "fail";
        report.failed_phase = alloc_failed_phase ? alloc_failed_phase : "backend-preflight";
        report.failed_reason = alloc_failed_reason ? alloc_failed_reason : "allocation failed";
        report.materialized_tensor_count = materialized_count;
        report.materialized_tensor_bytes = materialized_bytes;
        report.partial_materialization = materialized_count > 0ull ? "true" : "false";
        report.refused_tensor_count = required_tensor_count > materialized_count
                                          ? required_tensor_count - materialized_count
                                          : 0ull;
        fullmodel_print_materialize_report(&report);
        return exit_for_status(rc);
    }

    if (fullmodel_fail_after(options, "materialize-embedding") ||
        fullmodel_fail_after(options, "materialize-normalization") ||
        fullmodel_fail_after(options, "materialize-attention") ||
        fullmodel_fail_after(options, "materialize-mlp") ||
        fullmodel_fail_after(options, "materialize-moe") ||
        fullmodel_fail_after(options, "materialize-output") ||
        fullmodel_fail_after(options, "materialize-tokenizer") ||
        fullmodel_fail_after(options, "cleanup")) {
        report.failed_phase = options->fail_after_phase;
        report.failed_reason = "injected-failure";
        report.materialized_tensor_count = materialized_count;
        report.materialized_tensor_bytes = materialized_bytes;
        report.partial_materialization = "false";
        fullmodel_print_materialize_report(&report);
        return exit_for_status(YVEX_ERR_STATE);
    }

    report.status = "fullmodel-materialize-pass";
    report.full_model_materialization_proof = "pass";
    report.phase = "complete";
    report.materialized_tensor_count = materialized_count;
    report.materialized_tensor_bytes = materialized_bytes;
    report.refused_tensor_count = 0ull;
    report.skipped_tensor_count = tensor_count > materialized_count
                                      ? tensor_count - materialized_count
                                      : 0ull;
    fullmodel_print_materialize_report(&report);
    return 0;
}

typedef struct {
    yvex_cli_fullmodel_options *options;
    yvex_model_ref *ref;
    yvex_model_context *ctx;
    yvex_fullmodel_collections *collections;
    fullmodel_largest_tensor *largest;
    const char *target_id;
    const char *target_class;
    const char *source_artifact_class;
    const char *target_artifact_class;
    const char *model_class;
    const char *inventory_status;
    const char *role_coverage;
    const char *backend_placement_status;
    const char *cpu_placement;
    const char *cuda_placement;
    const char *dtype_summary;
    const char *missing_roles;
    const char *descriptor_missing_roles;
    const char *unsupported_roles;
    yvex_arch arch;
    unsigned long long artifact_bytes;
    unsigned long long tensor_count;
    unsigned long long total_tensor_bytes;
    unsigned int largest_count;
    int selected_target;
} fullmodel_surface_view;

#define SURFACE_TEXT(key_, type_, member_, fallback_) \
    {key_, YVEX_CLI_FIELD_TEXT, offsetof(type_, member_), fallback_}
#define SURFACE_U64(key_, type_, member_) \
    {key_, YVEX_CLI_FIELD_U64, offsetof(type_, member_), NULL}

static const yvex_cli_field_spec surface_identity_fields[] = {
    SURFACE_TEXT("target_id", fullmodel_surface_view, target_id, "unknown"),
    SURFACE_TEXT("target_class", fullmodel_surface_view, target_class, "unknown"),
    SURFACE_TEXT("source_artifact_class", fullmodel_surface_view, source_artifact_class, "unknown"),
    SURFACE_TEXT("target_artifact_class", fullmodel_surface_view, target_artifact_class, "unknown"),
};

static const yvex_cli_field_spec surface_model_fields[] = {
    SURFACE_TEXT("model_class", fullmodel_surface_view, model_class, "unknown"),
    SURFACE_TEXT("fullmodel_inventory", fullmodel_surface_view, inventory_status, "unknown"),
};

static const yvex_cli_field_spec surface_dtype_fields[] = {
    SURFACE_TEXT("qtype_summary", fullmodel_surface_view, dtype_summary, "none"),
    SURFACE_TEXT("dtype_summary", fullmodel_surface_view, dtype_summary, "none"),
};

static const yvex_cli_field_spec surface_memory_fields[] = {
    SURFACE_U64("total_tensor_bytes", fullmodel_surface_view, total_tensor_bytes),
    SURFACE_U64("estimated_cpu_resident_bytes", fullmodel_surface_view, total_tensor_bytes),
    SURFACE_U64("estimated_cuda_resident_bytes", fullmodel_surface_view, total_tensor_bytes),
};

static const yvex_cli_field_spec surface_placement_fields[] = {
    SURFACE_TEXT("backend_placement_status", fullmodel_surface_view, backend_placement_status, "unknown"),
    SURFACE_TEXT("cpu_placement", fullmodel_surface_view, cpu_placement, "unknown"),
    SURFACE_TEXT("cuda_placement", fullmodel_surface_view, cuda_placement, "unknown"),
};

static const yvex_cli_field_spec surface_attention_collection_fields[] = {
    SURFACE_U64("embedding_tensors", yvex_fullmodel_collections, embedding),
    SURFACE_U64("normalization_tensors", yvex_fullmodel_collections, normalization),
    SURFACE_U64("attention_tensors", yvex_fullmodel_collections, attention),
};

static const yvex_cli_field_spec surface_remaining_collection_fields[] = {
    SURFACE_U64("mlp_tensors", yvex_fullmodel_collections, mlp),
    SURFACE_U64("moe_tensors", yvex_fullmodel_collections, moe),
    SURFACE_U64("output_tensors", yvex_fullmodel_collections, output),
    SURFACE_U64("tokenizer_tensors", yvex_fullmodel_collections, tokenizer),
    SURFACE_U64("unknown_tensors", yvex_fullmodel_collections, unknown),
};

static const yvex_cli_field_spec surface_role_fields[] = {
    SURFACE_TEXT("required_role_coverage", fullmodel_surface_view, role_coverage, "unknown"),
    SURFACE_TEXT("missing_required_roles", fullmodel_surface_view, missing_roles, "unknown"),
    SURFACE_TEXT("unsupported_required_roles", fullmodel_surface_view, unsupported_roles, "unknown"),
};

static const char *const surface_inventory_lines[] = {
    "tensor_inventory_status: pass", "metadata_status: pass",
};

static const char *const surface_memory_boundary_lines[] = {
    "estimated_kv_bytes: planned", "estimated_scratch_bytes: planned",
    "estimated_total_runtime_bytes: unknown",
};

static const char *const surface_collection_boundary_lines[] = {
    "collection_supported: partial", "runtime_consumer: unsupported",
};

static const char *const surface_runtime_blocker_lines[] = {
    "runtime_blockers: full tensor set missing; attention projection tensors may be missing; MLP/MoE tensors "
        "may be missing; output head may be missing; real transformer prefill unsupported; real DeepSeek decode "
        "unsupported; real output-head logits unsupported; real vocabulary sampling unsupported; full model "
        "materialization not implemented",
};

#undef SURFACE_U64
#undef SURFACE_TEXT

typedef struct {
    yvex_model_ref ref;
    yvex_model_context ctx;
    yvex_fullmodel_collections collections;
    fullmodel_dtype_bucket dtype_buckets[32];
    fullmodel_largest_tensor largest[16];
    char dtype_summary[512];
    char missing_roles[768];
    char descriptor_missing_roles[768];
    char unsupported_roles[512];
    fullmodel_surface_view view;
} fullmodel_surface_state;

typedef struct missing_role_spec {
    size_t present_offset;
    const char *name;
} missing_role_spec;

#define MISSING_ROLE(member_, name_) \
    {offsetof(yvex_fullmodel_collections, member_), name_}

static const missing_role_spec missing_role_specs[] = {
    MISSING_ROLE(has_token_embedding, "token-embedding"),
    MISSING_ROLE(has_attention_norm, "attention-norm"),
    MISSING_ROLE(has_post_attention_norm, "post-attention-norm"),
    MISSING_ROLE(has_attention_q, "attention-q-projection"),
    MISSING_ROLE(has_attention_k, "attention-k-projection"),
    MISSING_ROLE(has_attention_v, "attention-v-projection"),
    MISSING_ROLE(has_attention_out, "attention-output-projection"),
    MISSING_ROLE(has_ffn_gate, "mlp-gate"),
    MISSING_ROLE(has_ffn_up, "mlp-up"),
    MISSING_ROLE(has_ffn_down, "mlp-down"),
    MISSING_ROLE(has_moe_router, "moe-router"),
    MISSING_ROLE(has_moe_expert, "moe-experts"),
    MISSING_ROLE(has_output_norm, "final-norm"),
    MISSING_ROLE(has_output_head, "output-head"),
    MISSING_ROLE(has_tokenizer_metadata, "tokenizer-metadata"),
};

static const char *const unsupported_runtime_roles[] = {
    "runtime-family-adapter", "real-transformer-prefill", "real-DeepSeek-decode",
    "real-output-head-logits", "real-vocabulary-sampling"};

#undef MISSING_ROLE

static int fullmodel_source_only_dispatch(const yvex_cli_fullmodel_options *options,
                                          int *handled)
{
    const char *target = "glm-5.2-official-safetensors";

    *handled = strcmp(options->model, target) == 0 ||
               (options->target && strcmp(options->target, target) == 0);
    if (!*handled) return 0;
    if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN) {
        return print_fullmodel_source_only_plan(options, target);
    }
    if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZE) {
        return print_fullmodel_source_only_materialize(options, target);
    }
    if (options->command == YVEX_FULLMODEL_COMMAND_DESCRIPTOR) {
        return print_fullmodel_source_only_descriptor(options, target);
    }
    if (options->command == YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME) {
        return print_fullmodel_source_only_family_runtime(options, target);
    }
    return print_fullmodel_source_only_report(target, options->backend);
}

static void fullmodel_surface_inventory(fullmodel_surface_state *state,
                                        const yvex_cli_fullmodel_options *options)
{
    unsigned long long i;
    unsigned int bucket_count = 0u;
    unsigned int largest_count = 0u;

    state->view.tensor_count = yvex_tensor_table_count(state->ctx.table);
    for (i = 0; i < state->view.tensor_count; ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(state->ctx.table, i);
        if (!tensor) continue;
        state->view.total_tensor_bytes += tensor->storage_bytes;
        fullmodel_record_dtype(state->dtype_buckets, &bucket_count, tensor);
        fullmodel_record_largest(state->largest, &largest_count,
                                 (unsigned int)options->limit_tensors, tensor);
        fullmodel_classify_tensor(tensor, &state->collections);
    }
    if (yvex_gguf_metadata_find(state->ctx.gguf, "tokenizer.ggml.tokens") ||
        yvex_gguf_metadata_find(state->ctx.gguf, "tokenizer.ggml.model")) {
        state->collections.tokenizer = 1ull;
        state->collections.tokenizer_bytes = 0ull;
        state->collections.has_tokenizer_metadata = 1;
    }
    state->view.largest_count = largest_count;
    fullmodel_dtype_summary(state->dtype_summary, sizeof(state->dtype_summary),
                            state->dtype_buckets, bucket_count);
}

static void fullmodel_surface_roles(fullmodel_surface_state *state,
                                    const yvex_cli_fullmodel_options *options)
{
    yvex_fullmodel_collections *c = &state->collections;
    size_t i;

    for (i = 0; i < sizeof(missing_role_specs) / sizeof(missing_role_specs[0]); ++i) {
        const int *present = (const int *)((const unsigned char *)c +
                                          missing_role_specs[i].present_offset);
        if (!*present) fullmodel_csv_append(state->missing_roles,
                                            sizeof(state->missing_roles),
                                            missing_role_specs[i].name);
    }
    if (!state->missing_roles[0]) {
        snprintf(state->missing_roles, sizeof(state->missing_roles), "none");
    }
    for (i = 0; i < sizeof(unsupported_runtime_roles) / sizeof(unsupported_runtime_roles[0]); ++i) {
        fullmodel_csv_append(state->unsupported_roles, sizeof(state->unsupported_roles),
                             unsupported_runtime_roles[i]);
    }
    fullmodel_materialize_missing_roles(options, c, state->descriptor_missing_roles,
                                        sizeof(state->descriptor_missing_roles));
}

static int fullmodel_surface_prepare(yvex_cli_fullmodel_options *options,
                                     fullmodel_surface_state *state)
{
    yvex_model_ref_options ref_options;
    yvex_error err;
    fullmodel_surface_view *view = &state->view;
    int rc;

    memset(state, 0, sizeof(*state));
    memset(&ref_options, 0, sizeof(ref_options));
    yvex_error_clear(&err);
    ref_options.allow_registry = 1;
    ref_options.registry_path = options->registry_path;
    rc = yvex_model_ref_resolve(&state->ref, options->model, &ref_options, &err);
    if (rc != YVEX_OK) return print_yvex_error(&err, exit_for_status(rc));
    if (!fullmodel_file_size(state->ref.path, &view->artifact_bytes)) {
        rc = print_fullmodel_missing_report(options, state->ref.path);
        yvex_model_ref_clear(&state->ref);
        return rc;
    }
    rc = yvex_model_context_open(state->ref.path, &state->ctx, &err);
    if (rc != YVEX_OK) {
        int out = print_fullmodel_parse_failure_report(options, &state->ref,
                                                       yvex_error_message(&err), rc);
        yvex_error_clear(&err);
        yvex_model_ref_clear(&state->ref);
        return out;
    }
    view->options = options;
    view->ref = &state->ref;
    view->ctx = &state->ctx;
    view->collections = &state->collections;
    view->largest = state->largest;
    fullmodel_surface_inventory(state, options);
    view->arch = yvex_model_arch(state->ctx.model);
    view->target_id = options->target ? options->target :
        (state->ref.alias && state->ref.alias[0] ? state->ref.alias : "path");
    view->selected_target = fullmodel_is_selected_target(view->target_id) ||
                            fullmodel_is_selected_target(options->model);
    view->target_class = view->selected_target ? "selected-runtime-slice" :
        (options->target && strcmp(options->target, "deepseek4-v4-flash") == 0
             ? "full-runtime-model-planned" : "candidate-GGUF-path");
    view->source_artifact_class = view->selected_target
        ? "YVEX-produced selected GGUF" : "GGUF artifact";
    view->target_artifact_class = view->selected_target
        ? "YVEX-produced selected GGUF" : "candidate GGUF artifact";
    view->model_class = view->selected_target
        ? "selected-runtime-slice" : "descriptor-only-candidate";
    view->inventory_status = view->selected_target ? "incomplete" : "partial";
    fullmodel_surface_roles(state, options);
    view->role_coverage = strcmp(state->missing_roles, "none") == 0 ? "observed" : "partial";
    if (view->selected_target) view->role_coverage = "partial";
    view->backend_placement_status = view->selected_target
        ? "selected-tensor-plan-only" : "report-only";
    view->cpu_placement = view->selected_target
        ? "selected-tensors-only" : "planned-full-model";
    view->cuda_placement = strcmp(options->backend, "cuda") == 0
        ? (yvex_backend_cuda_context_available()
               ? "selected-or-candidate-tensors-only" : "unavailable")
        : "not-requested";
    view->dtype_summary = state->dtype_summary;
    view->missing_roles = state->missing_roles;
    view->descriptor_missing_roles = state->descriptor_missing_roles;
    view->unsupported_roles = state->unsupported_roles;
    return YVEX_OK;
}

static int fullmodel_surface_dispatch(fullmodel_surface_view *view, int *handled)
{
    yvex_cli_fullmodel_options *options = view->options;
    const char *descriptor_coverage = view->selected_target ? "partial" :
        strcmp(view->descriptor_missing_roles, "none") == 0 ? "complete" : "partial";
    int rc = 0;

    *handled = 1;
    if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZE) {
        rc = fullmodel_materialize_command_run(options, view->ref, view->ctx,
            view->target_id, view->target_class, view->artifact_bytes,
            view->tensor_count, view->total_tensor_bytes, view->collections,
            view->selected_target);
    } else if (options->command == YVEX_FULLMODEL_COMMAND_DESCRIPTOR) {
        if (options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
            fullmodel_print_descriptor_normal(options, view->target_id,
                view->target_class, descriptor_coverage, view->descriptor_missing_roles);
        } else {
            fullmodel_print_descriptor_report(options, view->ref, view->ctx,
                view->target_id, view->target_class, view->artifact_bytes, view->arch,
                view->tensor_count, view->total_tensor_bytes, view->collections,
                descriptor_coverage, view->descriptor_missing_roles,
                view->unsupported_roles, view->selected_target);
        }
    } else if (options->command == YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME) {
        if (options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
            fullmodel_print_family_runtime_normal(options, view->target_id,
                view->target_class, descriptor_coverage, view->descriptor_missing_roles);
        } else {
            rc = fullmodel_print_family_runtime_report(options, view->ref, view->ctx,
                view->target_id, view->target_class, view->artifact_bytes, view->arch,
                view->tensor_count, view->total_tensor_bytes, view->collections,
                descriptor_coverage, view->descriptor_missing_roles,
                view->unsupported_roles, view->selected_target);
        }
    } else if (options->command == YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN) {
        if (options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
            fullmodel_print_plan_normal(options,
                view->selected_target ? "blocked" : "partial", view->target_class,
                view->selected_target ? "blocked" : "unknown",
                view->selected_target ? "selected-slice-not-full-runtime" :
                                        "full-runtime candidate artifact required");
        } else {
            fullmodel_print_materialization_plan(options, view->ref, view->target_id,
                view->target_class, view->artifact_bytes, view->arch, view->tensor_count,
                view->total_tensor_bytes, view->collections, view->dtype_summary,
                view->role_coverage, view->missing_roles, view->selected_target);
        }
    } else if (options->output_mode != YVEX_MODELS_OUTPUT_AUDIT) {
        fullmodel_print_report_normal(options, "fullmodel-report", view->target_id,
            view->target_class, view->role_coverage,
            view->selected_target ? "selected-slice-not-full-runtime" :
                                    "missing-full-runtime-tensor-coverage",
            "tensor/source/artifact row required");
    } else {
        *handled = 0;
        return 0;
    }
    yvex_model_context_close(view->ctx);
    yvex_model_ref_clear(view->ref);
    return rc;
}

static int fullmodel_surface_render_audit(fullmodel_surface_view *view)
{
    const yvex_cli_fullmodel_options *options = view->options;
    const yvex_fullmodel_collections *collections = view->collections;
    int cuda_available = yvex_backend_cuda_context_available();

    yvex_cli_out_lines(stdout, literal_pair_0, sizeof(literal_pair_0) / sizeof(literal_pair_0[0]));
    yvex_cli_out_writef(stdout, "model: %s\n", options->model);
    yvex_cli_out_writef(stdout, "model_resolved_path: %s\n", view->ref->path ? view->ref->path : "");
    yvex_cli_out_fields(stdout, view, surface_identity_fields,
                        sizeof(surface_identity_fields) / sizeof(surface_identity_fields[0]));
    yvex_cli_out_writef(stdout, "artifact_exists: true\n");
    yvex_cli_out_writef(stdout, "artifact_bytes: %llu\n", view->artifact_bytes);
    yvex_cli_out_writef(stdout, "artifact_identity_status: %s\n",
                        fullmodel_identity_status(view->ref, view->artifact_bytes));
    yvex_cli_out_writef(stdout, "tensor_count: %llu\n", view->tensor_count);
    yvex_cli_out_lines(stdout, surface_inventory_lines,
                       sizeof(surface_inventory_lines) / sizeof(surface_inventory_lines[0]));
    yvex_cli_out_writef(stdout, "architecture: %s\n", yvex_arch_name(view->arch));
    yvex_cli_out_writef(stdout, "family: %s\n", fullmodel_family_from_arch(view->arch));
    yvex_cli_out_fields(stdout, view, surface_model_fields,
                        sizeof(surface_model_fields) / sizeof(surface_model_fields[0]));
    yvex_cli_out_writef(stdout, "full_runtime_model: false\n");
    yvex_cli_out_fields(stdout, view, surface_dtype_fields,
                        sizeof(surface_dtype_fields) / sizeof(surface_dtype_fields[0]));
    yvex_cli_out_fields(stdout, view, surface_memory_fields,
                        sizeof(surface_memory_fields) / sizeof(surface_memory_fields[0]));
    yvex_cli_out_lines(stdout, surface_memory_boundary_lines,
                       sizeof(surface_memory_boundary_lines) / sizeof(surface_memory_boundary_lines[0]));
    yvex_cli_out_writef(stdout, "backend: %s\n", options->backend);
    yvex_cli_out_fields(stdout, view, surface_placement_fields,
                        sizeof(surface_placement_fields) / sizeof(surface_placement_fields[0]));
    yvex_cli_out_writef(stdout, "cuda_context_available: %s\n", cuda_available ? "true" : "false");
    yvex_cli_out_writef(stdout, "cuda_memory_status: %s\n",
                        cuda_available ? "context-available-no-allocation" : "unavailable");
    yvex_cli_out_writef(stdout, "residency_plan: report-only-no-allocation\n");
    yvex_cli_out_writef(stdout, "tensor_collections_status: %s\n", view->role_coverage);
    yvex_cli_out_writef(stdout, "collection_detected: %s\n",
                        view->tensor_count ? "yes" : "no");
    yvex_cli_out_lines(stdout, surface_collection_boundary_lines,
                       sizeof(surface_collection_boundary_lines) /
                           sizeof(surface_collection_boundary_lines[0]));
    yvex_cli_out_fields(stdout, collections, surface_attention_collection_fields,
                        sizeof(surface_attention_collection_fields) /
                            sizeof(surface_attention_collection_fields[0]));
    yvex_cli_out_writef(stdout, "kv_cache_requirements: planned\n");
    yvex_cli_out_fields(stdout, collections, surface_remaining_collection_fields,
                        sizeof(surface_remaining_collection_fields) /
                            sizeof(surface_remaining_collection_fields[0]));
    yvex_cli_out_fields(stdout, view, surface_role_fields,
                        sizeof(surface_role_fields) / sizeof(surface_role_fields[0]));
    yvex_cli_out_lines(stdout, surface_runtime_blocker_lines,
                       sizeof(surface_runtime_blocker_lines) /
                           sizeof(surface_runtime_blocker_lines[0]));
    print_fullmodel_common_boundaries();
    yvex_cli_out_writef(stdout, "largest_tensor_report_limit: %llu\n", options->limit_tensors);
    fullmodel_print_largest(view->largest, view->largest_count);
    yvex_model_context_close(view->ctx);
    yvex_model_ref_clear(view->ref);
    return 0;
}

int yvex_model_artifacts_surface_fullmodel_command(int arg_count, char **args)
{
    yvex_cli_fullmodel_options options;
    fullmodel_surface_state state;
    int handled;
    int rc;

    memset(&options, 0, sizeof(options));
    rc = model_artifacts_fullmodel_options_parse(arg_count, args, &options);
    if (rc == 1) return 0;
    if (rc != 0) return rc;
    rc = fullmodel_source_only_dispatch(&options, &handled);
    if (handled) return rc;
    rc = fullmodel_surface_prepare(&options, &state);
    if (rc != YVEX_OK) return rc;
    rc = fullmodel_surface_dispatch(&state.view, &handled);
    if (handled) return rc;
    return fullmodel_surface_render_audit(&state.view);
}

/* Models command dispatch and help. */
