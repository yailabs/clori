/*
 * Coordinate exact admission across canonical source owners.
 *
 * Success requires complete reopened manifest and one header scan. Verified metadata is not
 * transformed bytes or artifact support.
 */
#define _XOPEN_SOURCE 700
#include <ctype.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#define SOURCE_CONFIG_CAP (1024u * 1024u)
#define SOURCE_TOKENIZER_CAP (32u * 1024u * 1024u)

typedef enum {
    SOURCE_SIDECAR_CONFIG = 0,
    SOURCE_SIDECAR_TOKENIZER,
    SOURCE_SIDECAR_TOKENIZER_CONFIG,
    SOURCE_SIDECAR_GENERATION_CONFIG,
    SOURCE_SIDECAR_INFERENCE_CONFIG
} source_sidecar_kind;

typedef struct {
    const char *name;
    const char *missing;
    const char *malformed;
} source_sidecar_rule;

static const source_sidecar_rule source_sidecar_rules[] = {
    {"config.json", "missing-source-config", "malformed-source-config"},
    {"tokenizer.json", "missing-tokenizer-json", "malformed-tokenizer-json"},
    {"tokenizer_config.json", "missing-tokenizer-config", "malformed-tokenizer-config"},
    {"generation_config.json", "missing-generation-config", "malformed-generation-config"},
    {"inference/config.json", "missing-dspark-inference-config",
     "malformed-dspark-inference-config"},
};

static const source_sidecar_rule source_sidecar_unknown = {
    "", "missing-source-sidecar", "malformed-source-sidecar"};

static int source_parse_sidecar(source_sidecar_kind kind,
                                const char *data,
                                size_t length,
                                const yvex_source_target_identity *identity,
                                yvex_source_verification *out);
static int source_parse_config_json(const char *data,
                                    size_t length,
                                    const yvex_source_target_identity *identity,
                                    yvex_source_verification *out);
static int source_parse_tokenizer_config(const char *data,
                                         size_t length,
                                         yvex_source_verification *out);
static int source_parse_generation_config(const char *data,
                                          size_t length,
                                          yvex_source_verification *out);
static int source_parse_inference_config(const char *data,
                                         size_t length,
                                         yvex_source_verification *out);
static int source_parse_tokenizer_json(const char *data,
                                       size_t length,
                                       yvex_source_verification *out);

typedef int (*source_sidecar_parser)(const char *, size_t, yvex_source_verification *);

static const source_sidecar_parser source_sidecar_parsers[] = {
    source_parse_tokenizer_json,
    source_parse_tokenizer_config,
    source_parse_generation_config,
    source_parse_inference_config,
};

static const size_t source_sidecar_valid_offsets[] = {
    offsetof(yvex_source_verification, tokenizer_json_valid),
    offsetof(yvex_source_verification, tokenizer_config_valid),
    offsetof(yvex_source_verification, generation_config_valid),
    offsetof(yvex_source_verification, inference_config_valid),
};

static unsigned int source_sidecar_bit(source_sidecar_kind kind)
{
    return kind >= SOURCE_SIDECAR_CONFIG && kind <= SOURCE_SIDECAR_INFERENCE_CONFIG
               ? 1u << (unsigned int)kind
               : 0u;
}

int yvex_source_verification_required_sidecars_valid(
    const yvex_source_target_identity *identity,
    const yvex_source_verification *verification)
{
    unsigned int required;

    if (!identity || !verification) return 0;
    required = identity->required_sidecars;
    return (!(required & YVEX_SOURCE_SIDECAR_CONFIG) || verification->config_valid) &&
           (!(required & YVEX_SOURCE_SIDECAR_TOKENIZER) ||
            verification->tokenizer_json_valid) &&
           (!(required & YVEX_SOURCE_SIDECAR_TOKENIZER_CONFIG) ||
            verification->tokenizer_config_valid) &&
           (!(required & YVEX_SOURCE_SIDECAR_GENERATION_CONFIG) ||
            verification->generation_config_valid) &&
           (!(required & YVEX_SOURCE_SIDECAR_INFERENCE_CONFIG) ||
            verification->inference_config_valid);
}

int yvex_source_path_join(char *out, size_t cap, const char *left, const char *right) {
    int n;

    if (!out || cap == 0u || !left || !right)
        return 0;
    n = snprintf(
        out, cap, "%s%s%s", left, left[0] && left[strlen(left) - 1u] == '/' ? "" : "/", right);
    return n >= 0 && (size_t)n < cap;
}

/*
 * Allocate one exact source-relative path with checked size arithmetic.
 *
 * Null input, overflow, or allocation failure returns null.
 */
char *yvex_source_path_alloc(const char *left, const char *right) {
    size_t left_length, right_length;
    const char *separator;
    char *path;

    if (!left || !right)
        return NULL;
    left_length = strlen(left);
    right_length = strlen(right);
    if (left_length > SIZE_MAX - right_length - 2u)
        return NULL;
    separator = left_length && left[left_length - 1u] == '/' ? "" : "/";
    path = (char *)malloc(left_length + right_length + 2u);
    if (path)
        snprintf(path, left_length + right_length + 2u, "%s%s%s", left, separator, right);
    return path;
}

int yvex_source_ends_with(const char *text, const char *suffix) {
    size_t text_length, suffix_length;

    if (!text || !suffix)
        return 0;
    text_length = strlen(text);
    suffix_length = strlen(suffix);
    return suffix_length <= text_length &&
           strcmp(text + text_length - suffix_length, suffix) == 0;
}

int yvex_source_regular_file(const char *path, unsigned long long *size) {
    struct stat st;

    if (!path || lstat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
        return 0;
    if (size)
        *size = (unsigned long long)st.st_size;
    return 1;
}

int yvex_source_revision_is_commit(const char *text) {
    size_t i;

    if (!text || strlen(text) != 40u)
        return 0;
    for (i = 0u; i < 40u; ++i) {
        if (!isxdigit((unsigned char)text[i]))
            return 0;
    }
    return 1;
}

void yvex_source_verification_add_blocker(yvex_source_verification *out, const char *reason) {
    unsigned int i;

    if (!out || !reason)
        return;
    for (i = 0u; i < out->blocker_count; ++i) {
        if (strcmp(out->blockers[i], reason) == 0)
            return;
    }
    if (out->blocker_count < YVEX_SOURCE_VERIFY_BLOCKER_CAP) {
        out->blockers[out->blocker_count++] = reason;
    }
}

int yvex_source_verification_has_blocker(const yvex_source_verification *out, const char *reason) {
    unsigned int i;

    if (!out || !reason)
        return 0;
    for (i = 0u; i < out->blocker_count; ++i) {
        if (strcmp(out->blockers[i], reason) == 0)
            return 1;
    }
    return 0;
}

static const source_sidecar_rule *source_sidecar_rule_at(source_sidecar_kind kind) {
    return kind >= SOURCE_SIDECAR_CONFIG && kind <= SOURCE_SIDECAR_INFERENCE_CONFIG
               ? &source_sidecar_rules[(size_t)kind]
               : &source_sidecar_unknown;
}

static int source_verify_sidecar(const yvex_source_verify_options *options,
                                 source_sidecar_kind kind,
                                 yvex_source_verification *out,
                                 yvex_error *err) {
    const source_sidecar_rule *rule = source_sidecar_rule_at(kind);
    const char *name = rule->name;
    char path[YVEX_PATH_CAP];
    char *data;
    size_t length;
    size_t cap = kind == SOURCE_SIDECAR_TOKENIZER ? SOURCE_TOKENIZER_CAP : SOURCE_CONFIG_CAP;
    int rc;

    if (!(options->identity->required_sidecars & source_sidecar_bit(kind)))
        return YVEX_OK;

    if (!yvex_source_path_join(path, sizeof(path), options->source_path, name) ||
        !yvex_source_regular_file(path, NULL)) {
        yvex_source_verification_add_blocker(out, rule->missing);
        return YVEX_OK;
    }
    data = yvex_read_bounded_file(path, cap, &length, err);
    if (!data) {
        if (yvex_error_code(err) == YVEX_ERR_NOMEM)
            return YVEX_ERR_NOMEM;
        yvex_source_verification_add_blocker(out, rule->malformed);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if (!source_parse_sidecar(kind, data, length, options->identity, out)) {
        yvex_source_verification_add_blocker(out, rule->malformed);
    }
    free(data);
    rc = yvex_source_provenance_verify_file(options, name, 0, out, err);
    return rc;
}

static int source_manifest_is_current(const yvex_source_verify_options *options,
                                      const yvex_source_verification *out) {
    return strcmp(out->manifest_status, "complete") == 0 &&
           yvex_source_provenance_manifest_matches(options, out);
}

static int source_promote_manifest(const yvex_source_verify_options *options,
                                   yvex_source_verification *out,
                                   yvex_error *err) {
    int rc;

    snprintf(out->verification_stage,
             sizeof(out->verification_stage),
             "%s",
             "exact-source-metadata-header-verified");
    rc = yvex_source_publish(&(yvex_source_publication_request){
                                 .kind = YVEX_SOURCE_PUBLICATION_VERIFIED_MANIFEST,
                                 .out_path = out->manifest_path,
                                 .options = options,
                                 .verification = out,
                             },
                             err);
    if (rc != YVEX_OK) {
        yvex_source_verification_add_blocker(out, "source-manifest-publish-failed");
        yvex_error_clear(err);
        return YVEX_OK;
    }
    out->manifest_published = 1;
    rc = yvex_source_provenance_manifest_read(options, out, err);
    if (rc != YVEX_OK)
        return rc;
    out->manifest_reopened = 1;
    if (!source_manifest_is_current(options, out)) {
        yvex_source_verification_add_blocker(out, "source-manifest-reopen-mismatch");
    } else {
        out->manifest_verified = 1;
    }
    return YVEX_OK;
}

const char *yvex_source_verification_status(const yvex_source_verification *verification) {
    if (!verification)
        return "invalid";
    return verification->verified ? "verified" : "blocked";
}

int yvex_source_verify_with_snapshot(const yvex_source_verify_options *options,
                                     yvex_source_verification *out,
                                     yvex_source_tensor_snapshot **snapshot,
                                     yvex_error *err) {
    struct stat st;
    yvex_source_derived_inventory derived;
    yvex_source_tensor_snapshot *candidate_snapshot = NULL;
    int rc;
    unsigned int kind;

    if (!options || !out || !options->identity || !options->source_path) {
        yvex_error_set(err,
                       YVEX_ERR_INVALID_ARG,
                       "source_verify",
                       "identity, source path, and output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    if (snapshot)
        *snapshot = NULL;
    memset(&derived, 0, sizeof(derived));
    yvex_error_clear(err);

    if (lstat(options->source_path, &st) != 0) {
        yvex_source_verification_add_blocker(out, "missing-source-path");
        return YVEX_OK;
    }
    if (!S_ISDIR(st.st_mode) || !realpath(options->source_path, out->resolved_source_path)) {
        yvex_source_verification_add_blocker(out, "wrong-source-path-type");
        return YVEX_OK;
    }
    out->path_verified = 1;
    rc = yvex_source_provenance_manifest_read(options, out, err);
    if (rc != YVEX_OK)
        goto cleanup;
    for (kind = SOURCE_SIDECAR_CONFIG; kind <= SOURCE_SIDECAR_INFERENCE_CONFIG; ++kind) {
        rc = source_verify_sidecar(options, (source_sidecar_kind)kind, out, err);
        if (rc != YVEX_OK)
            goto cleanup;
    }
    if (options->identity->config_validation ==
            YVEX_SOURCE_CONFIG_VALIDATION_DEEPSEEK_V4 &&
        out->config_valid && out->generation_config_valid &&
        (out->bos_token_id != out->generation_bos_token_id ||
         out->eos_token_id != out->generation_eos_token_id)) {
        yvex_source_verification_add_blocker(out, "generation-config-token-mismatch");
    }
    rc = yvex_source_inventory_verify(options, out, &derived, &candidate_snapshot, err);
    if (rc != YVEX_OK)
        goto cleanup;
    if (candidate_snapshot) {
        yvex_source_tensor_snapshot_facts snapshot_facts;
        rc = yvex_source_tensor_snapshot_facts_get(candidate_snapshot, &snapshot_facts, err);
        if (rc != YVEX_OK)
            goto cleanup;
        out->source_snapshot_identity = snapshot_facts.identity;
    }
    yvex_source_provenance_finalize(options, out);

    if (strcmp(out->inventory_authority, "header-derived") == 0 && out->blocker_count == 0u) {
        if (!options->derived_inventory_path) {
            yvex_source_verification_add_blocker(out, "missing-derived-inventory-output");
        } else {
            rc = yvex_source_publish(&(yvex_source_publication_request){
                                         .kind = YVEX_SOURCE_PUBLICATION_DERIVED_INVENTORY,
                                         .out_path = options->derived_inventory_path,
                                         .options = options,
                                         .inventory = &derived,
                                     },
                                     err);
            if (rc != YVEX_OK) {
                yvex_source_verification_add_blocker(out, "derived-inventory-publish-failed");
                yvex_error_clear(err);
            }
        }
    }
    if (out->blocker_count == 0u && source_manifest_is_current(options, out)) {
        out->manifest_verified = 1;
        out->manifest_reopened = 1;
        out->manifest_payload_trusted =
            strcmp(out->manifest_schema, "yvex.source_manifest.v3") == 0;
    } else if (out->blocker_count == 0u &&
               strcmp(out->manifest_schema, "yvex.source_manifest.v3") == 0 &&
               out->manifest_payload_identity[0]) {
        yvex_source_verification_add_blocker(out, "payload-snapshot-drift");
    } else if (out->blocker_count == 0u && options->promote_manifest) {
        rc = source_promote_manifest(options, out, err);
        if (rc != YVEX_OK)
            goto cleanup;
    } else if (out->blocker_count == 0u) {
        yvex_source_verification_add_blocker(out,
                                             strcmp(out->manifest_status, "complete") == 0
                                                 ? "source-manifest-stale"
                                                 : "source-manifest-incomplete");
    }
    out->verified = out->blocker_count == 0u && out->path_verified && out->repository_verified &&
                    out->revision_verified &&
                    yvex_source_verification_required_sidecars_valid(options->identity, out) &&
                    out->shard_index_headers_match && out->header_scan_count == 1u &&
                    out->manifest_verified &&
                    (strcmp(out->inventory_authority, "header-derived") == 0 ||
                     out->upstream_index_identity_verified);
    if (out->verified && snapshot) {
        *snapshot = candidate_snapshot;
        candidate_snapshot = NULL;
    }
    rc = YVEX_OK;
cleanup:
    yvex_source_tensor_snapshot_release(candidate_snapshot);
    yvex_source_derived_inventory_free(&derived);
    if (rc == YVEX_OK)
        yvex_error_clear(err);
    return rc;
}

int yvex_source_verify(const yvex_source_verify_options *options,
                       yvex_source_verification *out,
                       yvex_error *err) {
    yvex_source_tensor_snapshot *snapshot = NULL;
    int rc = yvex_source_verify_with_snapshot(options, out, &snapshot, err);

    yvex_source_tensor_snapshot_release(snapshot);
    return rc;
}
/* DeepSeek sidecar parsing publishes exact facts into source admission. */

typedef struct {
    unsigned long long seen;
    int architecture_matches;
} source_config_parse_state;

#define CONFIG_MODEL_TYPE (1ull << 0)
#define CONFIG_ARCHITECTURES (1ull << 1)
#define CONFIG_HIDDEN_SIZE (1ull << 2)
#define CONFIG_LAYER_COUNT (1ull << 3)
#define CONFIG_ATTENTION_HEADS (1ull << 4)
#define CONFIG_KV_HEADS (1ull << 5)
#define CONFIG_HEAD_DIM (1ull << 6)
#define CONFIG_QK_ROPE_DIM (1ull << 7)
#define CONFIG_MAX_POSITION (1ull << 8)
#define CONFIG_MOE_INTERMEDIATE (1ull << 9)
#define CONFIG_ROUTED_EXPERTS (1ull << 10)
#define CONFIG_SHARED_EXPERTS (1ull << 11)
#define CONFIG_EXPERTS_PER_TOKEN (1ull << 12)
#define CONFIG_HASH_LAYERS (1ull << 13)
#define CONFIG_Q_LORA (1ull << 14)
#define CONFIG_O_LORA (1ull << 15)
#define CONFIG_VOCAB (1ull << 16)
#define CONFIG_SLIDING_WINDOW (1ull << 17)
#define CONFIG_TIED (1ull << 18)
#define CONFIG_TORCH_DTYPE (1ull << 19)
#define CONFIG_EXPERT_DTYPE (1ull << 20)
#define CONFIG_HIDDEN_ACT (1ull << 21)
#define CONFIG_ROPE_SCALING (1ull << 22)
#define CONFIG_QUANTIZATION (1ull << 23)
#define CONFIG_ATTENTION_BIAS (1ull << 24)
#define CONFIG_ATTENTION_DROPOUT (1ull << 25)
#define CONFIG_BOS_TOKEN (1ull << 26)
#define CONFIG_EOS_TOKEN (1ull << 27)
#define CONFIG_COMPRESS_RATIOS (1ull << 28)
#define CONFIG_COMPRESS_ROPE_THETA (1ull << 29)
#define CONFIG_HC_EPS (1ull << 30)
#define CONFIG_HC_MULT (1ull << 31)
#define CONFIG_HC_SINKHORN_ITERS (1ull << 32)
#define CONFIG_INDEX_HEAD_DIM (1ull << 33)
#define CONFIG_INDEX_HEADS (1ull << 34)
#define CONFIG_INDEX_TOPK (1ull << 35)
#define CONFIG_NEXTN_LAYERS (1ull << 36)
#define CONFIG_O_GROUPS (1ull << 37)
#define CONFIG_RMS_NORM_EPS (1ull << 38)
#define CONFIG_ROPE_THETA (1ull << 39)
#define CONFIG_ROUTED_SCALING (1ull << 40)
#define CONFIG_SCORING_FUNC (1ull << 41)
#define CONFIG_TOPK_METHOD (1ull << 42)
#define CONFIG_NORM_TOPK (1ull << 43)
#define CONFIG_SWIGLU_LIMIT (1ull << 44)
#define CONFIG_USE_CACHE (1ull << 45)
#define CONFIG_DSPARK_BLOCK_SIZE (1ull << 46)
#define CONFIG_DSPARK_NOISE_TOKEN (1ull << 47)
#define CONFIG_DSPARK_TARGET_LAYERS (1ull << 48)
#define CONFIG_DSPARK_MARKOV_RANK (1ull << 49)
#define CONFIG_REQUIRED_MASK ((1ull << 50) - 1ull)

static int source_config_mark(source_config_parse_state *state, unsigned long long field) {
    if (state->seen & field)
        return 0;
    state->seen |= field;
    return 1;
}

typedef enum {
    SOURCE_JSON_TEXT,
    SOURCE_JSON_U64,
    SOURCE_JSON_BOOL,
    SOURCE_JSON_NUMBER,
    SOURCE_JSON_SKIP,
    SOURCE_JSON_U64_ARRAY,
    SOURCE_JSON_ARCHITECTURES,
    SOURCE_JSON_ROPE,
    SOURCE_JSON_QUANTIZATION,
    SOURCE_JSON_BLOCK_PAIR,
    SOURCE_JSON_LITERAL,
    SOURCE_JSON_TOKEN_ID,
    SOURCE_JSON_TOKEN_VOCAB,
    SOURCE_JSON_ADDED_TOKENS,
    SOURCE_JSON_TOKEN_MODEL
} source_json_kind;

typedef struct {
    const char *key;
    unsigned long long bit;
    source_json_kind kind;
    size_t offset;
    size_t size;
    size_t auxiliary_offset;
    size_t capacity;
} source_json_field;

static int source_parse_rope_scaling(yvex_json *json, yvex_source_verification *out);
static int source_parse_quantization(yvex_json *json, yvex_source_verification *out);
static int source_tokenizer_record_id(yvex_source_verification *out,
                                      unsigned long long token_id);
static int source_parse_tokenizer_vocab(yvex_json *json,
                                        yvex_source_verification *out);
static int source_parse_added_tokens(yvex_json *json,
                                     yvex_source_verification *out);
static int source_parse_tokenizer_model(yvex_json *json,
                                        yvex_source_verification *out);

static const source_json_field source_config_fields[] = {
    {"model_type", CONFIG_MODEL_TYPE, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, model_type), sizeof(((yvex_source_verification *)0)->model_type), 0u, 0u},
    {"architectures", CONFIG_ARCHITECTURES, SOURCE_JSON_ARCHITECTURES, 0u, 0u, 0u, 0u},
    {"hidden_size", CONFIG_HIDDEN_SIZE, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, hidden_size), 0u, 0u, 0u},
    {"num_hidden_layers", CONFIG_LAYER_COUNT, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_hidden_layers), 0u, 0u, 0u},
    {"num_attention_heads", CONFIG_ATTENTION_HEADS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_attention_heads), 0u, 0u, 0u},
    {"num_key_value_heads", CONFIG_KV_HEADS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_key_value_heads), 0u, 0u, 0u},
    {"head_dim", CONFIG_HEAD_DIM, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, head_dim), 0u, 0u, 0u},
    {"qk_rope_head_dim", CONFIG_QK_ROPE_DIM, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, qk_rope_head_dim), 0u, 0u, 0u},
    {"max_position_embeddings", CONFIG_MAX_POSITION, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, max_position_embeddings), 0u, 0u, 0u},
    {"moe_intermediate_size", CONFIG_MOE_INTERMEDIATE, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, moe_intermediate_size), 0u, 0u, 0u},
    {"n_routed_experts", CONFIG_ROUTED_EXPERTS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, n_routed_experts), 0u, 0u, 0u},
    {"n_shared_experts", CONFIG_SHARED_EXPERTS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, n_shared_experts), 0u, 0u, 0u},
    {"num_experts_per_tok", CONFIG_EXPERTS_PER_TOKEN, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_experts_per_tok), 0u, 0u, 0u},
    {"num_hash_layers", CONFIG_HASH_LAYERS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_hash_layers), 0u, 0u, 0u},
    {"q_lora_rank", CONFIG_Q_LORA, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, q_lora_rank), 0u, 0u, 0u},
    {"o_lora_rank", CONFIG_O_LORA, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, o_lora_rank), 0u, 0u, 0u},
    {"vocab_size", CONFIG_VOCAB, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, vocab_size), 0u, 0u, 0u},
    {"sliding_window", CONFIG_SLIDING_WINDOW, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, sliding_window), 0u, 0u, 0u},
    {"tie_word_embeddings", CONFIG_TIED, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, tie_word_embeddings), 0u, 0u, 0u},
    {"torch_dtype", CONFIG_TORCH_DTYPE, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, torch_dtype), sizeof(((yvex_source_verification *)0)->torch_dtype), 0u, 0u},
    {"expert_dtype", CONFIG_EXPERT_DTYPE, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, expert_dtype), sizeof(((yvex_source_verification *)0)->expert_dtype), 0u, 0u},
    {"hidden_act", CONFIG_HIDDEN_ACT, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, hidden_act), sizeof(((yvex_source_verification *)0)->hidden_act), 0u, 0u},
    {"rope_scaling", CONFIG_ROPE_SCALING, SOURCE_JSON_ROPE, 0u, 0u, 0u, 0u},
    {"quantization_config", CONFIG_QUANTIZATION, SOURCE_JSON_QUANTIZATION, 0u, 0u, 0u, 0u},
    {"attention_bias", CONFIG_ATTENTION_BIAS, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, attention_bias), 0u, 0u, 0u},
    {"attention_dropout", CONFIG_ATTENTION_DROPOUT, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, attention_dropout),
            sizeof(((yvex_source_verification *)0)->attention_dropout), 0u, 0u},
    {"bos_token_id", CONFIG_BOS_TOKEN, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, bos_token_id), 0u, 0u, 0u},
    {"eos_token_id", CONFIG_EOS_TOKEN, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, eos_token_id), 0u, 0u, 0u},
    {"compress_ratios", CONFIG_COMPRESS_RATIOS, SOURCE_JSON_U64_ARRAY,
     offsetof(yvex_source_verification, compress_ratios), 0u,
     offsetof(yvex_source_verification, compress_ratio_count), YVEX_SOURCE_VERIFY_COMPRESS_RATIO_CAP},
    {"compress_rope_theta", CONFIG_COMPRESS_ROPE_THETA, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, compress_rope_theta), 0u, 0u, 0u},
    {"hc_eps", CONFIG_HC_EPS, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, hc_eps), sizeof(((yvex_source_verification *)0)->hc_eps), 0u, 0u},
    {"hc_mult", CONFIG_HC_MULT, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, hc_mult), 0u, 0u, 0u},
    {"hc_sinkhorn_iters", CONFIG_HC_SINKHORN_ITERS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, hc_sinkhorn_iters), 0u, 0u, 0u},
    {"index_head_dim", CONFIG_INDEX_HEAD_DIM, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, index_head_dim), 0u, 0u, 0u},
    {"index_n_heads", CONFIG_INDEX_HEADS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, index_n_heads), 0u, 0u, 0u},
    {"index_topk", CONFIG_INDEX_TOPK, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, index_topk), 0u, 0u, 0u},
    {"num_nextn_predict_layers", CONFIG_NEXTN_LAYERS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, num_nextn_predict_layers), 0u, 0u, 0u},
    {"dspark_block_size", CONFIG_DSPARK_BLOCK_SIZE, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_block_size), 0u, 0u, 0u},
    {"dspark_noise_token_id", CONFIG_DSPARK_NOISE_TOKEN, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_noise_token_id), 0u, 0u, 0u},
    {"dspark_target_layer_ids", CONFIG_DSPARK_TARGET_LAYERS, SOURCE_JSON_U64_ARRAY,
     offsetof(yvex_source_verification, dspark_target_layer_ids), 0u,
     offsetof(yvex_source_verification, dspark_target_layer_count),
     YVEX_SOURCE_VERIFY_DSPARK_TARGET_LAYER_CAP},
    {"dspark_markov_rank", CONFIG_DSPARK_MARKOV_RANK, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_markov_rank), 0u, 0u, 0u},
    {"o_groups", CONFIG_O_GROUPS, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, o_groups), 0u, 0u, 0u},
    {"rms_norm_eps", CONFIG_RMS_NORM_EPS, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, rms_norm_eps), sizeof(((yvex_source_verification *)0)->rms_norm_eps), 0u, 0u},
    {"rope_theta", CONFIG_ROPE_THETA, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, rope_theta), 0u, 0u, 0u},
    {"routed_scaling_factor", CONFIG_ROUTED_SCALING, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, routed_scaling_factor),
            sizeof(((yvex_source_verification *)0)->routed_scaling_factor), 0u, 0u},
    {"scoring_func", CONFIG_SCORING_FUNC, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, scoring_func), sizeof(((yvex_source_verification *)0)->scoring_func), 0u, 0u},
    {"topk_method", CONFIG_TOPK_METHOD, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, topk_method), sizeof(((yvex_source_verification *)0)->topk_method), 0u, 0u},
    {"norm_topk_prob", CONFIG_NORM_TOPK, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, norm_topk_prob), 0u, 0u, 0u},
    {"swiglu_limit", CONFIG_SWIGLU_LIMIT, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, swiglu_limit), sizeof(((yvex_source_verification *)0)->swiglu_limit), 0u, 0u},
    {"use_cache", CONFIG_USE_CACHE, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, use_cache), 0u, 0u, 0u},
};

static const source_json_field source_identity_config_fields[] = {
    {"model_type", CONFIG_MODEL_TYPE, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, model_type),
     sizeof(((yvex_source_verification *)0)->model_type), 0u, 0u},
    {"architectures", CONFIG_ARCHITECTURES, SOURCE_JSON_ARCHITECTURES,
     0u, 0u, 0u, 0u},
};

static const source_json_field source_rope_fields[] = {
    {"type", 1u, SOURCE_JSON_TEXT, offsetof(yvex_source_verification, rope_scaling_type),
     sizeof(((yvex_source_verification *)0)->rope_scaling_type), 0u, 0u},
    {"factor", 2u, SOURCE_JSON_U64, offsetof(yvex_source_verification, rope_scaling_factor), 0u, 0u, 0u},
    {"original_max_position_embeddings", 4u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, rope_original_context), 0u, 0u, 0u},
    {"beta_fast", 8u, SOURCE_JSON_U64, offsetof(yvex_source_verification, rope_beta_fast), 0u, 0u, 0u},
    {"beta_slow", 16u, SOURCE_JSON_U64, offsetof(yvex_source_verification, rope_beta_slow), 0u, 0u, 0u},
};

static const source_json_field source_quant_fields[] = {
    {"quant_method", 1u, SOURCE_JSON_TEXT, offsetof(yvex_source_verification, quant_method),
     sizeof(((yvex_source_verification *)0)->quant_method), 0u, 0u},
    {"fmt", 2u, SOURCE_JSON_TEXT, offsetof(yvex_source_verification, quant_format),
     sizeof(((yvex_source_verification *)0)->quant_format), 0u, 0u},
    {"weight_block_size", 4u, SOURCE_JSON_BLOCK_PAIR,
     offsetof(yvex_source_verification, quant_block_rows), 0u,
     offsetof(yvex_source_verification, quant_block_columns), 0u},
    {"activation_scheme", 8u, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, quant_activation_scheme),
     sizeof(((yvex_source_verification *)0)->quant_activation_scheme), 0u, 0u},
    {"scale_fmt", 16u, SOURCE_JSON_TEXT, offsetof(yvex_source_verification, quant_scale_format),
     sizeof(((yvex_source_verification *)0)->quant_scale_format), 0u, 0u},
};

static const source_json_field source_tokenizer_config_fields[] = {
    {"tokenizer_class", 1u, SOURCE_JSON_TEXT, offsetof(yvex_source_verification, tokenizer_class),
     sizeof(((yvex_source_verification *)0)->tokenizer_class), 0u, 0u},
    {"model_max_length", 2u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, tokenizer_model_max_length), 0u, 0u, 0u},
    {"bos_token", 4u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
    {"eos_token", 8u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
};

static const source_json_field source_generation_fields[] = {
    {"_from_model_config", 1u, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, generation_from_model_config), 0u, 0u, 0u},
    {"bos_token_id", 2u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, generation_bos_token_id), 0u, 0u, 0u},
    {"eos_token_id", 4u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, generation_eos_token_id), 0u, 0u, 0u},
    {"do_sample", 8u, SOURCE_JSON_BOOL,
     offsetof(yvex_source_verification, generation_do_sample), 0u, 0u, 0u},
    {"temperature", 16u, SOURCE_JSON_NUMBER,
     offsetof(yvex_source_verification, generation_temperature),
     sizeof(((yvex_source_verification *)0)->generation_temperature), 0u, 0u},
    {"top_p", 32u, SOURCE_JSON_NUMBER, offsetof(yvex_source_verification, generation_top_p),
     sizeof(((yvex_source_verification *)0)->generation_top_p), 0u, 0u},
    {"transformers_version", 64u, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, generation_transformers_version),
     sizeof(((yvex_source_verification *)0)->generation_transformers_version), 0u, 0u},
};

static const source_json_field source_inference_fields[] = {
    {"n_mtp_layers", 1u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_inference_layer_count), 0u, 0u, 0u},
    {"dspark_block_size", 2u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_block_size), 0u, 0u, 0u},
    {"dspark_noise_token_id", 4u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_noise_token_id), 0u, 0u, 0u},
    {"dspark_target_layer_ids", 8u, SOURCE_JSON_U64_ARRAY,
     offsetof(yvex_source_verification, dspark_target_layer_ids), 0u,
     offsetof(yvex_source_verification, dspark_target_layer_count),
     YVEX_SOURCE_VERIFY_DSPARK_TARGET_LAYER_CAP},
    {"dspark_markov_rank", 16u, SOURCE_JSON_U64,
     offsetof(yvex_source_verification, dspark_markov_rank), 0u, 0u, 0u},
};

static const source_json_field source_added_token_fields[] = {
    {"id", 1u, SOURCE_JSON_TOKEN_ID, 0u, 0u, 0u, 0u},
};

static const source_json_field source_tokenizer_model_fields[] = {
    {"type", 1u, SOURCE_JSON_TEXT,
     offsetof(yvex_source_verification, tokenizer_model_type),
     sizeof(((yvex_source_verification *)0)->tokenizer_model_type), 0u, 0u},
    {"vocab", 2u, SOURCE_JSON_TOKEN_VOCAB, 0u, 0u, 0u, 0u},
};

static const source_json_field source_tokenizer_fields[] = {
    {"version", 1u, SOURCE_JSON_LITERAL, 0u, 16u, 0u, 0u},
    {"added_tokens", 2u, SOURCE_JSON_ADDED_TOKENS, 0u, 0u, 0u, 0u},
    {"normalizer", 4u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
    {"pre_tokenizer", 8u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
    {"post_processor", 16u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
    {"decoder", 32u, SOURCE_JSON_SKIP, 0u, 0u, 0u, 0u},
    {"model", 64u, SOURCE_JSON_TOKEN_MODEL, 0u, 0u, 0u, 0u},
};

static int source_parse_architectures(yvex_json *json,
                                      const yvex_source_target_identity *identity,
                                      yvex_source_verification *out,
                                      source_config_parse_state *state) {
    char architecture[128];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (!yvex_json_string(json, architecture, sizeof(architecture)))
            return 0;
        if (!out->architecture[0]) {
            snprintf(out->architecture, sizeof(out->architecture), "%s", architecture);
        }
        if (strcmp(architecture, identity->config_architecture) == 0) {
            state->architecture_matches = 1;
        }
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator && iter.count != 0u;
}

/*
 * Decode one schema-selected value into its typed verification field.
 *
 * Active JSON cursor, immutable field descriptor, source identity, and output state.
 */
static int source_json_field_apply(yvex_json *json,
                                   const source_json_field *field,
                                   const yvex_source_target_identity *identity,
                                   yvex_source_verification *out,
                                   source_config_parse_state *state) {
    unsigned char *base = (unsigned char *)out;

    switch (field->kind) {
    case SOURCE_JSON_TEXT:
        return yvex_json_string(json, (char *)(base + field->offset), field->size);
    case SOURCE_JSON_U64:
        return yvex_json_u64(json, (unsigned long long *)(void *)(base + field->offset));
    case SOURCE_JSON_BOOL:
        return yvex_json_bool(json, (int *)(void *)(base + field->offset));
    case SOURCE_JSON_NUMBER:
        return yvex_json_number_text(json, (char *)(base + field->offset), field->size);
    case SOURCE_JSON_SKIP:
        return yvex_json_skip_value(json);
    case SOURCE_JSON_U64_ARRAY:
        return yvex_json_u64_array(
            json,
            (unsigned long long *)(void *)(base + field->offset),
            field->capacity,
            (unsigned long long *)(void *)(base + field->auxiliary_offset));
    case SOURCE_JSON_ARCHITECTURES:
        return identity && source_parse_architectures(json, identity, out, state);
    case SOURCE_JSON_ROPE:
        return source_parse_rope_scaling(json, out);
    case SOURCE_JSON_QUANTIZATION:
        return source_parse_quantization(json, out);
    case SOURCE_JSON_LITERAL: {
        char literal[32];

        return field->size <= sizeof(literal) &&
               yvex_json_string(json, literal, field->size) && strcmp(literal, "1.0") == 0;
    }
    case SOURCE_JSON_TOKEN_ID: {
        unsigned long long token_id;

        return yvex_json_u64(json, &token_id) && source_tokenizer_record_id(out, token_id);
    }
    case SOURCE_JSON_TOKEN_VOCAB:
        return source_parse_tokenizer_vocab(json, out);
    case SOURCE_JSON_ADDED_TOKENS:
        return source_parse_added_tokens(json, out);
    case SOURCE_JSON_TOKEN_MODEL:
        return source_parse_tokenizer_model(json, out);
    case SOURCE_JSON_BLOCK_PAIR: {
        unsigned long long *rows = (unsigned long long *)(void *)(base + field->offset);
        unsigned long long *columns =
            (unsigned long long *)(void *)(base + field->auxiliary_offset);

        yvex_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != '[' ||
            !yvex_json_u64(json, rows))
            return 0;
        yvex_json_space(json);
        if (json->cursor >= json->end || *json->cursor++ != ',' ||
            !yvex_json_u64(json, columns))
            return 0;
        yvex_json_space(json);
        return json->cursor < json->end && *json->cursor++ == ']';
    }
    }
    return 0;
}

/*
 * Parse one JSON object through an immutable typed field schema.
 *
 * Active cursor, schema, required mask, optional identity, and caller-owned output.
 */
static int source_json_object_parse(yvex_json *json,
                                    const source_json_field *fields,
                                    size_t field_count,
                                    unsigned long long required,
                                    const yvex_source_target_identity *identity,
                                    yvex_source_verification *out,
                                    source_config_parse_state *state,
                                    int require_document_end) {
    char key[YVEX_JSON_KEY_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY) {
        const source_json_field *field;

        field = yvex_core_keyed_row_find(
            fields, field_count, sizeof(fields[0]), offsetof(source_json_field, key), key);
        if (!field) {
            if (!yvex_json_skip_value(json))
                return 0;
        } else if (!source_config_mark(state, field->bit) ||
                   !source_json_field_apply(json, field, identity, out, state)) {
            return 0;
        }
    }
    return item == YVEX_JSON_ITEM_END && (state->seen & required) == required &&
           (!require_document_end || yvex_json_complete(json));
}

static int source_parse_rope_scaling(yvex_json *json, yvex_source_verification *out) {
    source_config_parse_state state = {0};

    return source_json_object_parse(json,
                                    source_rope_fields,
                                    sizeof(source_rope_fields) / sizeof(source_rope_fields[0]),
                                    31u,
                                    NULL,
                                    out,
                                    &state,
                                    0);
}

static int source_parse_quantization(yvex_json *json, yvex_source_verification *out) {
    source_config_parse_state state = {0};

    return source_json_object_parse(json,
                                    source_quant_fields,
                                    sizeof(source_quant_fields) / sizeof(source_quant_fields[0]),
                                    31u,
                                    NULL,
                                    out,
                                    &state,
                                    0);
}

/*
 * Parse every required DeepSeek configuration field through immutable schema facts.
 *
 * Bounded JSON, pinned source identity, and caller-owned verification output. Fills only declared
 * configuration fields and records exact identity blockers. Malformed, duplicate, incomplete, or
 * identity-mismatched facts refuse admission.
 */
static int source_parse_config_json(const char *data,
                                    size_t length,
                                    const yvex_source_target_identity *identity,
                                    yvex_source_verification *out) {
    yvex_json json;
    source_config_parse_state state = {0};

    if (!identity || !identity->config_model_type ||
        !identity->config_architecture)
        return 0;

    yvex_json_init(&json, data, length);
    if (identity->config_validation ==
        YVEX_SOURCE_CONFIG_VALIDATION_FAMILY_SEMANTIC) {
        if (!source_json_object_parse(
                &json, source_identity_config_fields,
                sizeof(source_identity_config_fields) /
                    sizeof(source_identity_config_fields[0]),
                0u, identity, out, &state, 1))
            return 0;
        if ((state.seen & (CONFIG_MODEL_TYPE | CONFIG_ARCHITECTURES)) !=
            (CONFIG_MODEL_TYPE | CONFIG_ARCHITECTURES))
            yvex_source_verification_add_blocker(
                out, "missing-source-config-identity");
        if (strcmp(out->model_type, identity->config_model_type) != 0)
            yvex_source_verification_add_blocker(
                out, "wrong-source-model-type");
        if (!state.architecture_matches)
            yvex_source_verification_add_blocker(
                out, "wrong-source-architecture");
        out->config_valid =
            (state.seen & (CONFIG_MODEL_TYPE | CONFIG_ARCHITECTURES)) ==
                (CONFIG_MODEL_TYPE | CONFIG_ARCHITECTURES) &&
            strcmp(out->model_type, identity->config_model_type) == 0 &&
            state.architecture_matches;
        return 1;
    }
    if (!source_json_object_parse(&json,
                                  source_config_fields,
                                  sizeof(source_config_fields) / sizeof(source_config_fields[0]),
                                  0u,
                                  identity,
                                  out,
                                  &state,
                                  1)) {
        return 0;
    }
    if (state.seen != CONFIG_REQUIRED_MASK)
        yvex_source_verification_add_blocker(out, "missing-source-config-fact");
    if (strcmp(out->model_type, identity->config_model_type) != 0)
        yvex_source_verification_add_blocker(out, "wrong-source-model-type");
    if (!state.architecture_matches)
        yvex_source_verification_add_blocker(out, "wrong-source-architecture");
    out->config_valid = strcmp(out->model_type, identity->config_model_type) == 0 &&
                        state.architecture_matches && out->compress_ratio_count > 0u;
    return 1;
}

static int source_parse_json_object(const char *data, size_t length)
{
    char key[YVEX_JSON_KEY_CAP];
    yvex_json json;
    yvex_json_iter iter;
    yvex_json_item item;

    yvex_json_init(&json, data, length);
    if (!yvex_json_iter_begin(&json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, key, sizeof(key))) ==
           YVEX_JSON_ITEM_READY)
        if (!yvex_json_skip_value(&json)) return 0;
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator &&
           yvex_json_complete(&json);
}

static int
source_parse_tokenizer_config(const char *data, size_t length, yvex_source_verification *out) {
    yvex_json json;
    source_config_parse_state state = {0};

    yvex_json_init(&json, data, length);
    return source_json_object_parse(&json,
                                    source_tokenizer_config_fields,
                                    sizeof(source_tokenizer_config_fields) /
                                        sizeof(source_tokenizer_config_fields[0]),
                                    15u,
                                    NULL,
                                    out,
                                    &state,
                                    1);
}

static int
source_parse_generation_config(const char *data, size_t length, yvex_source_verification *out) {
    yvex_json json;
    source_config_parse_state state = {0};

    yvex_json_init(&json, data, length);
    return source_json_object_parse(&json,
                                    source_generation_fields,
                                    sizeof(source_generation_fields) /
                                        sizeof(source_generation_fields[0]),
                                    127u,
                                    NULL,
                                    out,
                                    &state,
                                    1);
}

/* The bundled inference configuration resolves the legacy next-token field into
 * the three-block DSpark execution topology. Duplicate DSpark facts must agree
 * with the public model configuration before either source is admitted. */
static int source_parse_inference_config(const char *data, size_t length,
                                         yvex_source_verification *out)
{
    yvex_source_verification candidate;
    source_config_parse_state state = {0};
    yvex_json json;
    unsigned long long index;

    if (!data || !out) return 0;
    candidate = *out;
    yvex_json_init(&json, data, length);
    if (!source_json_object_parse(
            &json, source_inference_fields,
            sizeof(source_inference_fields) / sizeof(source_inference_fields[0]),
            31u, NULL, &candidate, &state, 1) ||
        candidate.dspark_block_size != out->dspark_block_size ||
        candidate.dspark_noise_token_id != out->dspark_noise_token_id ||
        candidate.dspark_markov_rank != out->dspark_markov_rank ||
        candidate.dspark_target_layer_count != out->dspark_target_layer_count)
        return 0;
    for (index = 0u; index < out->dspark_target_layer_count; ++index)
        if (candidate.dspark_target_layer_ids[index] !=
            out->dspark_target_layer_ids[index])
            return 0;
    out->dspark_inference_layer_count = candidate.dspark_inference_layer_count;
    return out->dspark_inference_layer_count != 0u;
}

static int source_tokenizer_record_id(yvex_source_verification *out, unsigned long long token_id) {
    if (!out || token_id == ULLONG_MAX)
        return 0;
    if ((out->tokenizer_base_vocab_count == 0u && out->tokenizer_added_token_count == 0u) ||
        token_id > out->tokenizer_max_token_id)
        out->tokenizer_max_token_id = token_id;
    return 1;
}

static int source_parse_tokenizer_vocab(yvex_json *json, yvex_source_verification *out) {
    char token[YVEX_JSON_KEY_CAP];
    yvex_json_iter iter;
    yvex_json_item item;

    if (!json || !out || !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_OBJECT))
        return 0;
    while ((item = yvex_json_object_member(&iter, token, sizeof(token))) ==
           YVEX_JSON_ITEM_READY) {
        unsigned long long token_id;

        if (!yvex_json_u64(json, &token_id) || out->tokenizer_base_vocab_count == ULLONG_MAX ||
            !source_tokenizer_record_id(out, token_id))
            return 0;
        out->tokenizer_base_vocab_count++;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int source_parse_added_token(yvex_json *json, yvex_source_verification *out) {
    source_config_parse_state state = {0};

    return json && out &&
           source_json_object_parse(json,
                                    source_added_token_fields,
                                    sizeof(source_added_token_fields) /
                                        sizeof(source_added_token_fields[0]),
                                    1u,
                                    NULL,
                                    out,
                                    &state,
                                    0);
}

static int source_parse_added_tokens(yvex_json *json, yvex_source_verification *out) {
    yvex_json_iter iter;
    yvex_json_item item;

    if (!json || !out || !yvex_json_iter_begin(json, &iter, YVEX_JSON_COLLECTION_ARRAY))
        return 0;
    while ((item = yvex_json_array_value(&iter)) == YVEX_JSON_ITEM_READY) {
        if (!source_parse_added_token(json, out) || out->tokenizer_added_token_count == ULLONG_MAX)
            return 0;
        out->tokenizer_added_token_count++;
    }
    return item == YVEX_JSON_ITEM_END && !iter.trailing_separator;
}

static int source_parse_tokenizer_model(yvex_json *json, yvex_source_verification *out) {
    source_config_parse_state state = {0};

    return source_json_object_parse(json,
                                    source_tokenizer_model_fields,
                                    sizeof(source_tokenizer_model_fields) /
                                        sizeof(source_tokenizer_model_fields[0]),
                                    3u,
                                    NULL,
                                    out,
                                    &state,
                                    0);
}

static int
source_parse_tokenizer_json(const char *data, size_t length, yvex_source_verification *out) {
    yvex_json json;
    source_config_parse_state state = {0};

    yvex_json_init(&json, data, length);
    if (!source_json_object_parse(&json,
                                  source_tokenizer_fields,
                                  sizeof(source_tokenizer_fields) /
                                      sizeof(source_tokenizer_fields[0]),
                                  127u,
                                  NULL,
                                  out,
                                  &state,
                                  1) ||
        out->tokenizer_model_type[0] == '\0' ||
        (out->tokenizer_base_vocab_count == 0u && out->tokenizer_added_token_count == 0u) ||
        out->tokenizer_max_token_id == ULLONG_MAX)
        return 0;
    out->tokenizer_effective_vocab_size = out->tokenizer_max_token_id + 1u;
    return 1;
}

static int source_parse_sidecar(source_sidecar_kind kind,
                                const char *data,
                                size_t length,
                                const yvex_source_target_identity *identity,
                                yvex_source_verification *out) {
    size_t parser_index;
    int valid;

    if (!data || !identity || !out)
        return 0;
    if (kind == SOURCE_SIDECAR_CONFIG)
        return source_parse_config_json(data, length, identity, out);
    if (kind < SOURCE_SIDECAR_TOKENIZER || kind > SOURCE_SIDECAR_INFERENCE_CONFIG)
        return 0;
    parser_index = (size_t)kind - (size_t)SOURCE_SIDECAR_TOKENIZER;
    valid = identity->config_validation ==
                    YVEX_SOURCE_CONFIG_VALIDATION_FAMILY_SEMANTIC
                ? source_parse_json_object(data, length)
                : source_sidecar_parsers[parser_index](data, length, out);
    *(int *)(void *)((unsigned char *)out + source_sidecar_valid_offsets[parser_index]) = valid;
    return valid;
}
