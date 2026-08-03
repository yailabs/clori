/* Exact DeepSeek tensor coverage, mutation, scale, and lifetime tests. */
#include "tests/test.h"

#include <yvex/internal/compilation.h>
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/source.h>
#include <yvex/internal/source_payload.h>

#include <stdlib.h>
#include <string.h>

typedef enum {
    FIXTURE_VALID = 0,
    FIXTURE_MISSING,
    FIXTURE_MISSING_SCALE,
    FIXTURE_RANK,
    FIXTURE_SHAPE,
    FIXTURE_DTYPE,
    FIXTURE_SCALE,
    FIXTURE_UNEXPECTED,
    FIXTURE_INVALID_LAYER,
    FIXTURE_INVALID_EXPERT,
    FIXTURE_INDEX_OVERFLOW,
    FIXTURE_MISSING_FINAL,
    FIXTURE_MISSING_DRAFT_SHARED,
    FIXTURE_MISSING_DRAFT_SCALE,
    FIXTURE_MISSING_MARKOV,
    FIXTURE_MISSING_CONFIDENCE
} fixture_mutation;

static void fixture_copy(char *out, size_t cap, const char *value)
{
    (void)snprintf(out, cap, "%s", value);
}

static void fixture_verification(yvex_source_verification *source)
{
    unsigned long long i;

    memset(source, 0, sizeof(*source));
    source->verified = 1;
    source->path_verified = 1;
    source->repository_verified = 1;
    source->revision_verified = 1;
    source->manifest_verified = 1;
    source->manifest_reopened = 1;
    source->upstream_index_identity_verified = 1;
    source->config_valid = 1;
    source->tokenizer_json_valid = 1;
    source->tokenizer_config_valid = 1;
    source->generation_config_valid = 1;
    source->inference_config_valid = 1;
    source->shard_index_present = 1;
    source->shard_index_valid = 1;
    source->shard_index_headers_match = 1;
    source->header_scan_count = 1u;
    source->header_shard_count = 48u;
    source->header_tensor_count = 72317u;
    source->shard_count = 48u;
    source->indexed_tensor_count = 72317u;
    fixture_copy(source->manifest_target_id,
                 sizeof(source->manifest_target_id),
                 yvex_source_release_identity()->target_id);
    fixture_copy(source->verification_stage,
                 sizeof(source->verification_stage),
                 "exact-source-metadata-header-verified");
    fixture_copy(source->inventory_authority,
                 sizeof(source->inventory_authority), "upstream-index");
    fixture_copy(source->upstream_index_oid,
                 sizeof(source->upstream_index_oid),
                 yvex_source_release_identity()->upstream_index_oid);
    fixture_copy(source->local_index_oid, sizeof(source->local_index_oid),
                 yvex_source_release_identity()->upstream_index_oid);
    fixture_copy(source->source_kind, sizeof(source->source_kind),
                 "huggingface");
    fixture_copy(source->repository_id, sizeof(source->repository_id),
                 yvex_source_release_identity()->upstream_repo_id);
    fixture_copy(source->revision, sizeof(source->revision),
                 yvex_source_release_identity()->upstream_revision);
    fixture_copy(source->model_type, sizeof(source->model_type),
                 yvex_source_release_identity()->config_model_type);
    fixture_copy(source->architecture, sizeof(source->architecture),
                 yvex_source_release_identity()->config_architecture);
    fixture_copy(source->torch_dtype, sizeof(source->torch_dtype), "bfloat16");
    fixture_copy(source->expert_dtype, sizeof(source->expert_dtype), "fp4");
    fixture_copy(source->hidden_act, sizeof(source->hidden_act), "silu");
    fixture_copy(source->scoring_func, sizeof(source->scoring_func),
                 "sqrtsoftplus");
    fixture_copy(source->topk_method, sizeof(source->topk_method), "noaux_tc");
    fixture_copy(source->tokenizer_model_type,
                 sizeof(source->tokenizer_model_type), "BPE");
    fixture_copy(source->tokenizer_class, sizeof(source->tokenizer_class),
                 "PreTrainedTokenizerFast");
    fixture_copy(source->rope_scaling_type,
                 sizeof(source->rope_scaling_type), "yarn");
    fixture_copy(source->quant_method, sizeof(source->quant_method), "fp8");
    fixture_copy(source->quant_format, sizeof(source->quant_format), "e4m3");
    fixture_copy(source->quant_activation_scheme,
                 sizeof(source->quant_activation_scheme), "dynamic");
    fixture_copy(source->quant_scale_format,
                 sizeof(source->quant_scale_format), "ue8m0");
    fixture_copy(source->attention_dropout,
                 sizeof(source->attention_dropout), "0.0");
    fixture_copy(source->hc_eps, sizeof(source->hc_eps), "0.000001");
    fixture_copy(source->rms_norm_eps, sizeof(source->rms_norm_eps),
                 "0.000001");
    fixture_copy(source->routed_scaling_factor,
                 sizeof(source->routed_scaling_factor), "1.5");
    fixture_copy(source->swiglu_limit, sizeof(source->swiglu_limit), "10.0");
    source->hidden_size = 4096u;
    source->num_hidden_layers = 43u;
    source->num_attention_heads = 64u;
    source->num_key_value_heads = 1u;
    source->head_dim = 512u;
    source->qk_rope_head_dim = 64u;
    source->max_position_embeddings = 1048576u;
    source->moe_intermediate_size = 2048u;
    source->n_routed_experts = 256u;
    source->n_shared_experts = 1u;
    source->num_experts_per_tok = 6u;
    source->num_hash_layers = 3u;
    source->q_lora_rank = 1024u;
    source->o_lora_rank = 1024u;
    source->vocab_size = 129280u;
    source->sliding_window = 128u;
    source->bos_token_id = 0u;
    source->eos_token_id = 1u;
    source->compress_ratio_count = 46u;
    for (i = 0u; i < 46u; ++i)
        source->compress_ratios[i] =
            i < 2u || i >= 43u ? 0u : (i % 2u == 0u ? 4u : 128u);
    source->compress_rope_theta = 160000u;
    source->hc_mult = 4u;
    source->hc_sinkhorn_iters = 20u;
    source->index_head_dim = 128u;
    source->index_n_heads = 64u;
    source->index_topk = 512u;
    source->num_nextn_predict_layers = 1u;
    source->dspark_block_size = 5u;
    source->dspark_noise_token_id = 128799u;
    source->dspark_target_layer_ids[0] = 40u;
    source->dspark_target_layer_ids[1] = 41u;
    source->dspark_target_layer_ids[2] = 42u;
    source->dspark_target_layer_count = 3u;
    source->dspark_markov_rank = 256u;
    source->dspark_inference_layer_count = 3u;
    source->o_groups = 8u;
    source->rope_theta = 10000u;
    source->norm_topk_prob = 1;
    source->use_cache = 1;
    source->rope_scaling_factor = 16u;
    source->rope_original_context = 65536u;
    source->rope_beta_fast = 32u;
    source->rope_beta_slow = 1u;
    source->quant_block_rows = 128u;
    source->quant_block_columns = 128u;
    source->tokenizer_base_vocab_count = 128000u;
    source->tokenizer_added_token_count = 1283u;
    source->tokenizer_max_token_id = 129279u;
    source->tokenizer_effective_vocab_size = 129280u;
    source->tokenizer_model_max_length = 1048576u;
    source->generation_bos_token_id = 0u;
    source->generation_eos_token_id = 1u;
    source->generation_from_model_config = 1;
    source->generation_do_sample = 1;
}

typedef struct {
    yvex_native_weight_table *table;
    fixture_mutation mutation;
    yvex_error err;
} fixture_inventory;

/*
 * Assigns a stable synthetic shard from tensor identity so fixture insertion
 * order can change without changing the represented source snapshot.
 */
static unsigned int fixture_shard_for_name(const char *name)
{
    unsigned long long hash = 1469598103934665603ull;
    const unsigned char *cursor = (const unsigned char *)name;
    while (*cursor) {
        hash ^= (unsigned long long)*cursor++;
        hash *= 1099511628211ull;
    }
    return (unsigned int)(hash % 48u) + 1u;
}

static int fixture_add(fixture_inventory *fixture,
                       const char *name,
                       yvex_native_dtype dtype,
                       unsigned int rank,
                       const unsigned long long *input_dims)
{
    char shard[64];
    unsigned long long dims[2] = {0u, 0u};
    unsigned int i;

    if (fixture->mutation == FIXTURE_MISSING &&
        strcmp(name, "embed.weight") == 0) return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_SCALE &&
        strcmp(name, "layers.0.attn.wq_a.scale") == 0) return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_FINAL &&
        strcmp(name, "mtp.2.hc_head_scale") == 0) return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_DRAFT_SHARED &&
        strcmp(name, "mtp.1.ffn.shared_experts.w1.weight") == 0)
        return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_DRAFT_SCALE &&
        strcmp(name, "mtp.0.attn.wq_a.scale") == 0)
        return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_MARKOV &&
        strcmp(name, "mtp.2.markov_head.markov_w1.weight") == 0)
        return YVEX_OK;
    if (fixture->mutation == FIXTURE_MISSING_CONFIDENCE &&
        strcmp(name, "mtp.2.confidence_head.proj.weight") == 0)
        return YVEX_OK;
    for (i = 0u; i < rank; ++i) dims[i] = input_dims[i];
    if (fixture->mutation == FIXTURE_RANK &&
        strcmp(name, "head.weight") == 0) rank = 1u;
    if (fixture->mutation == FIXTURE_SHAPE &&
        strcmp(name, "layers.0.attn.wq_a.weight") == 0) dims[0]--;
    if (fixture->mutation == FIXTURE_SCALE &&
        strcmp(name, "layers.0.attn.wq_a.scale") == 0) dims[1]--;
    if (fixture->mutation == FIXTURE_DTYPE &&
        strcmp(name, "head.weight") == 0) dtype = YVEX_NATIVE_DTYPE_F32;
    (void)snprintf(shard, sizeof(shard),
                   "model-%05u-of-00048.safetensors",
                   fixture_shard_for_name(name));
    return yvex_native_weight_table_add(
        fixture->table, name, shard, yvex_native_dtype_name(dtype), rank,
        dims, 0u, 1u, &fixture->err);
}

static int fixture_v(fixture_inventory *f, const char *name,
                     yvex_native_dtype dtype, unsigned long long width)
{
    unsigned long long dims[1] = {width};
    return fixture_add(f, name, dtype, 1u, dims);
}

static int fixture_m(fixture_inventory *f, const char *name,
                     yvex_native_dtype dtype, unsigned long long rows,
                     unsigned long long columns)
{
    unsigned long long dims[2] = {rows, columns};
    return fixture_add(f, name, dtype, 2u, dims);
}

static int fixture_fp8(fixture_inventory *f, const char *base,
                       unsigned long long rows, unsigned long long columns)
{
    char name[512];
    int rc;
    (void)snprintf(name, sizeof(name), "%s.weight", base);
    rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_F8_E4M3, rows, columns);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.scale", base);
    return fixture_m(f, name, YVEX_NATIVE_DTYPE_F8_E8M0,
                     rows / 128u, columns / 128u);
}

static int fixture_fp4(fixture_inventory *f, const char *base,
                       unsigned long long rows, unsigned long long columns)
{
    char name[512];
    int rc;
    (void)snprintf(name, sizeof(name), "%s.weight", base);
    rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_I8, rows, columns / 2u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.scale", base);
    return fixture_m(f, name, YVEX_NATIVE_DTYPE_F8_E8M0,
                     rows, columns / 32u);
}

static int fixture_mhc(fixture_inventory *f, const char *prefix,
                       const char *kind)
{
    char name[256];
    int rc;
    (void)snprintf(name, sizeof(name), "%s.hc_%s_fn", prefix, kind);
    rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_F32, 24u, 16384u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.hc_%s_base", prefix, kind);
    rc = fixture_v(f, name, YVEX_NATIVE_DTYPE_F32, 24u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.hc_%s_scale", prefix, kind);
    return fixture_v(f, name, YVEX_NATIVE_DTYPE_F32, 3u);
}

static int fixture_attention(fixture_inventory *f, const char *prefix,
                             unsigned long long ratio)
{
    char name[256];
    char base[256];
    int rc;
#define FV(suffix, dtype, width)                                                \
    do {                                                                        \
        (void)snprintf(name, sizeof(name), "%s.%s", prefix, suffix);          \
        rc = fixture_v(f, name, dtype, width);                                  \
        if (rc != YVEX_OK) return rc;                                           \
    } while (0)
#define FP8(suffix, rows, columns)                                              \
    do {                                                                        \
        (void)snprintf(base, sizeof(base), "%s.%s", prefix, suffix);          \
        rc = fixture_fp8(f, base, rows, columns);                               \
        if (rc != YVEX_OK) return rc;                                           \
    } while (0)
    FV("attn.attn_sink", YVEX_NATIVE_DTYPE_F32, 64u);
    FV("attn.q_norm.weight", YVEX_NATIVE_DTYPE_BF16, 1024u);
    FV("attn.kv_norm.weight", YVEX_NATIVE_DTYPE_BF16, 512u);
    FP8("attn.wkv", 512u, 4096u);
    FP8("attn.wq_a", 1024u, 4096u);
    FP8("attn.wq_b", 32768u, 1024u);
    FP8("attn.wo_a", 8192u, 4096u);
    FP8("attn.wo_b", 4096u, 8192u);
    if (ratio) {
        unsigned long long width = ratio == 4u ? 1024u : 512u;
        (void)snprintf(name, sizeof(name), "%s.attn.compressor.ape", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_F32, ratio, width);
        if (rc != YVEX_OK) return rc;
        FV("attn.compressor.norm.weight", YVEX_NATIVE_DTYPE_BF16, 512u);
        (void)snprintf(name, sizeof(name),
                       "%s.attn.compressor.wgate.weight", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, width, 4096u);
        if (rc != YVEX_OK) return rc;
        (void)snprintf(name, sizeof(name),
                       "%s.attn.compressor.wkv.weight", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, width, 4096u);
        if (rc != YVEX_OK) return rc;
    }
    if (ratio == 4u) {
        (void)snprintf(name, sizeof(name),
                       "%s.attn.indexer.compressor.ape", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_F32, 4u, 256u);
        if (rc != YVEX_OK) return rc;
        FV("attn.indexer.compressor.norm.weight",
           YVEX_NATIVE_DTYPE_BF16, 128u);
        (void)snprintf(name, sizeof(name),
                       "%s.attn.indexer.compressor.wgate.weight", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, 256u, 4096u);
        if (rc != YVEX_OK) return rc;
        (void)snprintf(name, sizeof(name),
                       "%s.attn.indexer.compressor.wkv.weight", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, 256u, 4096u);
        if (rc != YVEX_OK) return rc;
        FP8("attn.indexer.wq_b", 8192u, 1024u);
        (void)snprintf(name, sizeof(name),
                       "%s.attn.indexer.weights_proj.weight", prefix);
        rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, 64u, 4096u);
        if (rc != YVEX_OK) return rc;
    }
#undef FP8
#undef FV
    return YVEX_OK;
}

static int fixture_layer(fixture_inventory *f, const char *prefix,
                         unsigned long long layer, unsigned long long ratio,
                         int hash_router)
{
    char name[256];
    char base[256];
    unsigned long long expert;
    int rc = fixture_attention(f, prefix, ratio);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.attn_norm.weight", prefix);
    rc = fixture_v(f, name, YVEX_NATIVE_DTYPE_BF16, 4096u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.ffn_norm.weight", prefix);
    rc = fixture_v(f, name, YVEX_NATIVE_DTYPE_BF16, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_mhc(f, prefix, "attn");
    if (rc != YVEX_OK) return rc;
    rc = fixture_mhc(f, prefix, "ffn");
    if (rc != YVEX_OK) return rc;
    for (expert = 0u; expert < 256u; ++expert) {
        (void)snprintf(base, sizeof(base), "%s.ffn.experts.%llu.w1",
                       prefix, expert);
        rc = fixture_fp4(f, base, 2048u, 4096u);
        if (rc != YVEX_OK) return rc;
        (void)snprintf(base, sizeof(base), "%s.ffn.experts.%llu.w2",
                       prefix, expert);
        rc = fixture_fp4(f, base, 4096u, 2048u);
        if (rc != YVEX_OK) return rc;
        (void)snprintf(base, sizeof(base), "%s.ffn.experts.%llu.w3",
                       prefix, expert);
        rc = fixture_fp4(f, base, 2048u, 4096u);
        if (rc != YVEX_OK) return rc;
    }
    (void)snprintf(base, sizeof(base), "%s.ffn.shared_experts.w1", prefix);
    rc = fixture_fp8(f, base, 2048u, 4096u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(base, sizeof(base), "%s.ffn.shared_experts.w2", prefix);
    rc = fixture_fp8(f, base, 4096u, 2048u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(base, sizeof(base), "%s.ffn.shared_experts.w3", prefix);
    rc = fixture_fp8(f, base, 2048u, 4096u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s.ffn.gate.weight", prefix);
    rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_BF16, 256u, 4096u);
    if (rc != YVEX_OK) return rc;
    if (hash_router) {
        (void)snprintf(name, sizeof(name), "%s.ffn.gate.tid2eid", prefix);
        return fixture_m(f, name, YVEX_NATIVE_DTYPE_I64, 129280u, 6u);
    }
    (void)snprintf(name, sizeof(name), "%s.ffn.gate.bias", prefix);
    (void)layer;
    return fixture_v(f, name, YVEX_NATIVE_DTYPE_F32, 256u);
}

static int fixture_globals(fixture_inventory *f)
{
    int rc = fixture_m(f, "embed.weight", YVEX_NATIVE_DTYPE_BF16,
                       129280u, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_v(f, "norm.weight", YVEX_NATIVE_DTYPE_BF16, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_m(f, "head.weight", YVEX_NATIVE_DTYPE_BF16,
                   129280u, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_m(f, "hc_head_fn", YVEX_NATIVE_DTYPE_F32, 4u, 16384u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_v(f, "hc_head_base", YVEX_NATIVE_DTYPE_F32, 4u);
    if (rc != YVEX_OK) return rc;
    return fixture_v(f, "hc_head_scale", YVEX_NATIVE_DTYPE_F32, 1u);
}

static int fixture_dspark(fixture_inventory *f)
{
    char name[256];
    unsigned long long layer;
    int rc = YVEX_OK;

    for (layer = 0u; layer < 3u && rc == YVEX_OK; ++layer) {
        char prefix[32];
        (void)snprintf(prefix, sizeof(prefix), "mtp.%llu", layer);
        rc = fixture_layer(f, prefix, 43u + layer, 0u, 0);
    }
    if (rc != YVEX_OK) return rc;
    rc = fixture_fp8(f, "mtp.0.main_proj", 4096u, 12288u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_v(f, "mtp.0.main_norm.weight", YVEX_NATIVE_DTYPE_BF16, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_v(f, "mtp.2.norm.weight", YVEX_NATIVE_DTYPE_BF16, 4096u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_m(f, "mtp.2.markov_head.markov_w1.weight",
                   YVEX_NATIVE_DTYPE_BF16, 129280u, 256u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_m(f, "mtp.2.markov_head.markov_w2.weight",
                   YVEX_NATIVE_DTYPE_BF16, 129280u, 256u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_m(f, "mtp.2.confidence_head.proj.weight",
                   YVEX_NATIVE_DTYPE_BF16, 1u, 4352u);
    if (rc != YVEX_OK) return rc;
    (void)snprintf(name, sizeof(name), "%s", "mtp.2.hc_head_fn");
    rc = fixture_m(f, name, YVEX_NATIVE_DTYPE_F32, 4u, 16384u);
    if (rc != YVEX_OK) return rc;
    rc = fixture_v(f, "mtp.2.hc_head_base", YVEX_NATIVE_DTYPE_F32, 4u);
    if (rc != YVEX_OK) return rc;
    return fixture_v(f, "mtp.2.hc_head_scale", YVEX_NATIVE_DTYPE_F32, 1u);
}

static int fixture_snapshot(yvex_source_tensor_snapshot **out,
                            yvex_source_verification *verification,
                            fixture_mutation mutation,
                            int globals_last)
{
    fixture_inventory fixture;
    yvex_source_tensor_snapshot_facts facts;
    unsigned long long layer;
    char prefix[64];
    int rc;

    memset(&fixture, 0, sizeof(fixture));
    fixture.mutation = mutation;
    fixture.table = (yvex_native_weight_table *)calloc(1u,
                                                       sizeof(*fixture.table));
    if (!fixture.table) return YVEX_ERR_NOMEM;
    if (!globals_last && fixture_globals(&fixture) != YVEX_OK) goto fail;
    for (layer = 0u; layer < 43u; ++layer) {
        unsigned long long ratio = layer < 2u ? 0u
                                  : (layer % 2u == 0u ? 4u : 128u);
        (void)snprintf(prefix, sizeof(prefix), "layers.%llu", layer);
        if (fixture_layer(&fixture, prefix, layer, ratio, layer < 3u) !=
            YVEX_OK) goto fail;
    }
    if (fixture_dspark(&fixture) != YVEX_OK) goto fail;
    if (globals_last && fixture_globals(&fixture) != YVEX_OK) goto fail;
    if (mutation == FIXTURE_UNEXPECTED &&
        fixture_v(&fixture, "unexpected.weight", YVEX_NATIVE_DTYPE_F32, 1u) !=
            YVEX_OK) goto fail;
    if (mutation == FIXTURE_INVALID_LAYER &&
        fixture_v(&fixture, "layers.43.attn_norm.weight",
                  YVEX_NATIVE_DTYPE_BF16, 4096u) != YVEX_OK) goto fail;
    if (mutation == FIXTURE_INVALID_EXPERT &&
        fixture_m(&fixture, "layers.0.ffn.experts.256.w1.weight",
                  YVEX_NATIVE_DTYPE_I8, 2048u, 2048u) != YVEX_OK) goto fail;
    if (mutation == FIXTURE_INDEX_OVERFLOW &&
        fixture_v(&fixture,
                  "layers.184467440737095516160.attn_norm.weight",
                  YVEX_NATIVE_DTYPE_BF16, 4096u) != YVEX_OK) goto fail;
    rc = yvex_source_tensor_snapshot_take_table(
        out, &fixture.table, 48u, 1u, &fixture.err);
    if (rc != YVEX_OK) goto fail;
    YVEX_TEST_ASSERT(yvex_source_tensor_snapshot_facts_get(
                         *out, &facts, &fixture.err) == YVEX_OK,
                     "snapshot facts available");
    verification->header_tensor_count = facts.tensor_count;
    verification->indexed_tensor_count = facts.tensor_count;
    verification->source_snapshot_identity = facts.identity;
    verification->manifest_payload_source_snapshot_identity = facts.identity;
    verification->manifest_payload_trusted = 1;
    fixture_copy(verification->manifest_payload_identity,
                 sizeof(verification->manifest_payload_identity),
                 "e22b3678d131d334f154a93214bdddfafc172c9869f4c52db28fea198eaa9165");
    fixture_copy(verification->manifest_payload_trust_class,
                 sizeof(verification->manifest_payload_trust_class),
                 "upstream_payload_verified");
    return YVEX_OK;
fail:
    yvex_native_weight_table_close(fixture.table);
    return yvex_error_code(&fixture.err) == YVEX_OK
               ? YVEX_ERR_FORMAT
               : yvex_error_code(&fixture.err);
}

static int fixture_build_case(
    fixture_mutation mutation,
    int globals_last,
    yvex_deepseek_tensor_coverage **coverage,
    yvex_deepseek_tensor_coverage_failure *failure)
{
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_error err;
    int rc;

    fixture_verification(&verification);
    yvex_error_clear(&err);
    rc = yvex_model_register_deepseek_v4()->ir.build(&ir, &verification, &ir_failure, &err);
    if (rc == YVEX_OK &&
        fixture_snapshot(&snapshot, &verification, mutation, globals_last) !=
            YVEX_OK) {
        rc = YVEX_ERR_FORMAT;
    }
    if (rc == YVEX_OK) {
        rc = yvex_model_register_deepseek_v4()->coverage.build(
            coverage, &verification, ir, snapshot, NULL, failure, &err);
    }
    yvex_model_register_deepseek_v4()->ir.close(ir);
    yvex_source_tensor_snapshot_release(snapshot);
    return rc;
}

typedef struct {
    unsigned int calls;
    unsigned int fail_at;
    unsigned int live;
    size_t fail_size;
    unsigned int fail_size_hits;
} fixture_allocator;

static void *fixture_allocate(size_t size, void *context)
{
    fixture_allocator *allocator = (fixture_allocator *)context;
    void *result;
    if (allocator->calls++ == allocator->fail_at) return NULL;
    if (allocator->fail_size == size && allocator->fail_size_hits++ == 0u)
        return NULL;
    result = malloc(size);
    if (result) allocator->live++;
    return result;
}

static void fixture_release(void *allocation, void *context)
{
    fixture_allocator *allocator = (fixture_allocator *)context;
    if (!allocation) return;
    free(allocation);
    allocator->live--;
}

static int test_valid_target_scale(void)
{
    yvex_deepseek_tensor_coverage *first = NULL;
    yvex_deepseek_tensor_coverage *shuffled = NULL;
    yvex_deepseek_tensor_coverage_failure failure;
    const yvex_deepseek_tensor_coverage_summary *summary;
    unsigned long long identity;

    {
        int fixture_rc = fixture_build_case(FIXTURE_VALID, 0, &first, &failure);
        YVEX_TEST_ASSERT(fixture_rc == YVEX_OK && first,
                     "72,317-entry coverage fixture closes");
    }
    summary = yvex_model_register_deepseek_v4()->coverage.summary(first);
    YVEX_TEST_ASSERT(summary->complete && summary->source_tensor_count == 72317u &&
                     summary->main_layer_count == 43u &&
                     summary->auxiliary_layer_count == 3u &&
                     summary->required_tensor_count == 72317u &&
                     summary->matched_tensor_count == 72317u &&
                     summary->missing_count == 0u &&
                     summary->unexpected_count == 0u,
                     "target-scale one-to-one summary is exact");
    YVEX_TEST_ASSERT(summary->collection_counts[
                         YVEX_TENSOR_COLLECTION_ROUTED_EXPERT] ==
                         70656u &&
                     summary->collection_counts[
                         YVEX_TENSOR_COLLECTION_ATTENTION] == 598u &&
                     summary->collection_counts[
                         YVEX_TENSOR_COLLECTION_AUXILIARY] == 10u,
                     "major collection counts include target and DSpark tensors");
    YVEX_TEST_ASSERT(summary->header_scan_count == 1u &&
                     summary->payload_bytes_read == 0u &&
                     summary->source_lookup_count < 3u * 72317u &&
                     summary->source_collision_count > 0u &&
                     summary->source_maximum_probe > 0u &&
                     summary->source_maximum_probe < 64u,
                     "indexed reconciliation is linear and payload-free");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.find(
                         first, "layers.2.attn.indexer.wq_b.scale") != NULL &&
                     yvex_model_register_deepseek_v4()->coverage.find(
                         first, "mtp.2.hc_head_fn") != NULL,
                     "CSA scale and DSpark head are typed requirements");
    identity = summary->coverage_identity;
    YVEX_TEST_ASSERT(fixture_build_case(FIXTURE_VALID, 1, &shuffled, &failure) ==
                         YVEX_OK && shuffled,
                     "shuffled insertion order closes");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.summary(shuffled)->
                             coverage_identity == identity,
                     "coverage identity is insertion-order invariant");
    yvex_model_register_deepseek_v4()->coverage.close(shuffled);
    yvex_model_register_deepseek_v4()->coverage.close(first);
    return 0;
}

static int test_mutations(void)
{
    const struct {
        fixture_mutation mutation;
        yvex_deepseek_tensor_coverage_failure_code expected;
    } cases[] = {
        {FIXTURE_MISSING, YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT},
        {FIXTURE_MISSING_SCALE,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION},
        {FIXTURE_RANK, YVEX_DEEPSEEK_COVERAGE_FAILURE_RANK_MISMATCH},
        {FIXTURE_SHAPE, YVEX_DEEPSEEK_COVERAGE_FAILURE_SHAPE_MISMATCH},
        {FIXTURE_DTYPE, YVEX_DEEPSEEK_COVERAGE_FAILURE_DTYPE_MISMATCH},
        {FIXTURE_SCALE, YVEX_DEEPSEEK_COVERAGE_FAILURE_SHAPE_MISMATCH},
        {FIXTURE_UNEXPECTED,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_UNEXPECTED_SOURCE},
        {FIXTURE_INVALID_LAYER,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX},
        {FIXTURE_INVALID_EXPERT,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_INVALID_INDEX},
        {FIXTURE_INDEX_OVERFLOW,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_ARITHMETIC_OVERFLOW},
        {FIXTURE_MISSING_FINAL,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT},
        {FIXTURE_MISSING_DRAFT_SHARED,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT},
        {FIXTURE_MISSING_DRAFT_SCALE,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_SCALE_COMPANION},
        {FIXTURE_MISSING_MARKOV,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT},
        {FIXTURE_MISSING_CONFIDENCE,
         YVEX_DEEPSEEK_COVERAGE_FAILURE_MISSING_REQUIREMENT}
    };
    size_t i;

    for (i = 0u; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        yvex_deepseek_tensor_coverage *coverage = NULL;
        yvex_deepseek_tensor_coverage_failure failure;
        YVEX_TEST_ASSERT(fixture_build_case(cases[i].mutation, 0, &coverage,
                                            &failure) != YVEX_OK && !coverage,
                         "mutated inventory refuses coverage");
        YVEX_TEST_ASSERT(failure.code == cases[i].expected,
                         "mutation returns stable typed reason");
    }
    return 0;
}

static int test_limits_and_allocation(void)
{
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_deepseek_tensor_coverage_failure failure;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_deepseek_tensor_coverage_options options;
    yvex_error err;
    unsigned int fail_at;

    fixture_verification(&verification);
    YVEX_TEST_ASSERT(fixture_snapshot(&snapshot, &verification,
                                      FIXTURE_VALID, 0) == YVEX_OK,
                     "allocation test snapshot builds");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.build(
                         &ir, &verification, &ir_failure, &err) == YVEX_OK,
                     "allocation test IR builds");
    fixture_copy(verification.inventory_authority,
                 sizeof(verification.inventory_authority), "header-derived");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.build(
                         &coverage, &verification, ir, snapshot, NULL,
                         &failure, &err) != YVEX_OK && !coverage &&
                     failure.code ==
                         YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_AUTHORITY,
                     "non-pinned inventory authority refuses coverage");
    fixture_copy(verification.inventory_authority,
                 sizeof(verification.inventory_authority), "upstream-index");
    verification.header_tensor_count--;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.build(
                         &coverage, &verification, ir, snapshot, NULL,
                         &failure, &err) != YVEX_OK && !coverage &&
                     failure.code ==
                         YVEX_DEEPSEEK_COVERAGE_FAILURE_INVENTORY_DRIFT,
                     "snapshot and verification count drift refuses coverage");
    verification.header_tensor_count++;
    memset(&options, 0, sizeof(options));
    options.maximum_tensors = 72316u;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.build(
                         &coverage, &verification, ir, snapshot, &options,
                         &failure, &err) != YVEX_OK && !coverage &&
                     failure.code ==
                         YVEX_DEEPSEEK_COVERAGE_FAILURE_RESOURCE_LIMIT,
                     "resource limit fails before construction");
    for (fail_at = 0u; fail_at < 4u; ++fail_at) {
        fixture_allocator allocator;
        memset(&allocator, 0, sizeof(allocator));
        allocator.fail_at = fail_at;
        memset(&options, 0, sizeof(options));
        options.allocate = fixture_allocate;
        options.release = fixture_release;
        options.context = &allocator;
        YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.build(
                             &coverage, &verification, ir, snapshot, &options,
                             &failure, &err) != YVEX_OK && !coverage,
                         "injected allocation failure refuses");
        YVEX_TEST_ASSERT(allocator.live == 0u,
                         "partial coverage allocation rolls back");
    }
    yvex_model_register_deepseek_v4()->ir.close(ir);
    yvex_source_tensor_snapshot_release(snapshot);
    return 0;
}

static int fixture_build_map(int globals_last,
                             yvex_deepseek_gguf_map **map,
                             yvex_deepseek_gguf_map_failure *failure,
                             const yvex_deepseek_gguf_map_allocator *allocator)
{
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_transform_ir *transform_ir = NULL;
    yvex_transform_failure transform_failure;
    yvex_error err;
    int rc;

    fixture_verification(&verification);
    if (fixture_snapshot(&snapshot, &verification, FIXTURE_VALID,
                         globals_last) != YVEX_OK) return YVEX_ERR_FORMAT;
    yvex_error_clear(&err);
    rc = yvex_model_register_deepseek_v4()->ir.build(&ir, &verification, &ir_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_model_register_deepseek_v4()->coverage.build(
            &coverage, &verification, ir, snapshot, NULL, &coverage_failure,
            &err);
    if (rc == YVEX_OK)
        rc = yvex_model_register_deepseek_v4()->transform.build(
            &transform_ir, &verification, ir, coverage, NULL,
            &transform_failure, &err);
    if (rc == YVEX_OK) {
        rc = allocator
                 ? yvex_model_register_deepseek_v4()->lowering.build_with_allocator(
                       map, ir, transform_ir, allocator, failure, &err)
                 : yvex_model_register_deepseek_v4()->lowering.build(
                       map, ir, transform_ir, failure, &err);
    }
    yvex_transform_ir_release(&transform_ir);
    yvex_model_register_deepseek_v4()->coverage.close(coverage);
    yvex_model_register_deepseek_v4()->ir.close(ir);
    yvex_source_tensor_snapshot_release(snapshot);
    return rc;
}

int yvex_test_deepseek_map_fixture_build(yvex_deepseek_gguf_map **out)
{
    yvex_deepseek_gguf_map_failure failure;

    if (out) *out = NULL;
    return out ? fixture_build_map(0, out, &failure, NULL) : YVEX_ERR_INVALID_ARG;
}

static int fixture_build_transform(
    int globals_last,
    yvex_transform_ir **transform_ir,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure)
{
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *architecture = NULL;
    yvex_deepseek_v4_ir_failure architecture_failure;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_error err;
    int rc;

    if (transform_ir) *transform_ir = NULL;
    fixture_verification(&verification);
    if (!transform_ir ||
        fixture_snapshot(&snapshot, &verification, FIXTURE_VALID,
                         globals_last) != YVEX_OK)
        return YVEX_ERR_FORMAT;
    yvex_error_clear(&err);
    rc = yvex_model_register_deepseek_v4()->ir.build(
        &architecture, &verification, &architecture_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_model_register_deepseek_v4()->coverage.build(
            &coverage, &verification, architecture, snapshot, NULL,
            &coverage_failure, &err);
    if (rc == YVEX_OK)
        rc = yvex_model_register_deepseek_v4()->transform.build(
            transform_ir, &verification, architecture, coverage, options,
            failure, &err);
    yvex_model_register_deepseek_v4()->coverage.close(coverage);
    yvex_model_register_deepseek_v4()->ir.close(architecture);
    yvex_source_tensor_snapshot_release(snapshot);
    return rc;
}

/* Proves complete DeepSeek graph accounting, identity stability, and IR cleanup. */
static int test_transform_target_scale(void)
{
    yvex_transform_ir *first = NULL;
    yvex_transform_ir *shuffled = NULL;
    yvex_transform_failure failure;
    const yvex_transform_ir_summary *summary;
    fixture_allocator allocator_state;
    yvex_transform_builder_options options;

    YVEX_TEST_ASSERT(fixture_build_transform(
                         0, &first, NULL, &failure) == YVEX_OK && first,
                     "complete artifact-neutral DeepSeek plan seals");
    summary = yvex_transform_ir_summary_get(first);
    YVEX_TEST_ASSERT(summary && summary->complete &&
                         summary->schema_version ==
                             YVEX_TRANSFORM_IR_SCHEMA_VERSION &&
                         summary->source_value_count == 72317u &&
                         summary->intermediate_value_count == 0u &&
                         summary->value_count == 73726u &&
                         summary->node_count == 1409u &&
                         summary->edge_count == 72317u &&
                         summary->terminal_count == 1409u &&
                         summary->maximum_fan_in == 512u &&
                         summary->maximum_depth == 1u &&
                         summary->operation_counts[
                             YVEX_TRANSFORM_OP_IDENTITY] == 878u &&
                         summary->operation_counts[
                             YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR] == 390u &&
                         summary->operation_counts[
                             YVEX_TRANSFORM_OP_CHECKED_CAST] == 3u &&
                         summary->operation_counts[
                             YVEX_TRANSFORM_OP_EXPERT_AGGREGATE] == 138u,
                     "DeepSeek operation graph accounting is exact");
    YVEX_TEST_ASSERT(strlen(summary->logical_model_identity) == 64u &&
                         strlen(summary->required_payload_identity) == 64u &&
                         strlen(summary->transform_identity) == 64u &&
                         summary->header_scan_count == 1u &&
                         summary->payload_bytes_read == 0u &&
                         summary->builder_peak_bytes > 0u &&
                         summary->sealed_ir_bytes > 0u &&
                         summary->temporary_validation_bytes > 0u &&
                         summary->total_owned_bytes ==
                             summary->sealed_ir_bytes &&
                         summary->total_owned_bytes < 128u * 1024u * 1024u &&
                         summary->validation_steps < 3u * 72317u,
                     "DeepSeek planning is identity-bound, bounded, and payload-free");
    YVEX_TEST_ASSERT(fixture_build_transform(
                         1, &shuffled, NULL, &failure) == YVEX_OK && shuffled &&
                         strcmp(summary->transform_identity,
                                yvex_transform_ir_summary_get(shuffled)->
                                    transform_identity) == 0,
                     "DeepSeek identity is source-registration-order independent");
    yvex_transform_ir_release(&shuffled);
    yvex_transform_ir_release(&first);

    memset(&options, 0, sizeof(options));
    memset(&allocator_state, 0, sizeof(allocator_state));
    allocator_state.fail_at = ~0u;
    allocator_state.fail_size = 512u * sizeof(unsigned long long);
    options.allocator.allocate = fixture_allocate;
    options.allocator.release = fixture_release;
    options.allocator.context = &allocator_state;
    yvex_transform_budget_default(&options.budget);
    YVEX_TEST_ASSERT(fixture_build_transform(
                         0, &first, &options, &failure) != YVEX_OK && !first &&
                         failure.code == YVEX_TRANSFORM_FAILURE_ALLOCATION &&
                         allocator_state.fail_size_hits != 0u &&
                         allocator_state.live == 0u,
                     "DeepSeek large-fan-in allocation failure unwinds completely");
    return 0;
}

typedef struct {
    yvex_tensor_role role;
    const char *pattern;
} fixture_pinned_name;

/* Compact independent oracle extracted from llama.cpp e920c523. */
static const fixture_pinned_name fixture_pinned_names[] = {
    {YVEX_TENSOR_ROLE_TOKEN_EMBEDDING, "token_embd.weight"},
    {YVEX_TENSOR_ROLE_OUTPUT_NORM, "output_norm.weight"},
    {YVEX_TENSOR_ROLE_OUTPUT_HEAD, "output.weight"},
    {YVEX_TENSOR_ROLE_HC_HEAD_FUNCTION, "output_hc_fn.weight"},
    {YVEX_TENSOR_ROLE_HC_HEAD_BASE, "output_hc_base.weight"},
    {YVEX_TENSOR_ROLE_HC_HEAD_SCALE, "output_hc_scale.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_NORM, "blk.%llu.attn_norm.weight"},
    {YVEX_TENSOR_ROLE_FFN_NORM, "blk.%llu.ffn_norm.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_SINKS, "blk.%llu.attn_sinks.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_Q_A, "blk.%llu.attn_q_a.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_Q_B, "blk.%llu.attn_q_b.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM,
     "blk.%llu.attn_q_a_norm.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_KV, "blk.%llu.attn_kv.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_KV_NORM,
     "blk.%llu.attn_kv_a_norm.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
     "blk.%llu.attn_output_a.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_OUT_B,
     "blk.%llu.attn_output_b.weight"},
    {YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION,
     "blk.%llu.hc_attn_fn.weight"},
    {YVEX_TENSOR_ROLE_HC_ATTENTION_BASE,
     "blk.%llu.hc_attn_base.weight"},
    {YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE,
     "blk.%llu.hc_attn_scale.weight"},
    {YVEX_TENSOR_ROLE_HC_FFN_FUNCTION, "blk.%llu.hc_ffn_fn.weight"},
    {YVEX_TENSOR_ROLE_HC_FFN_BASE, "blk.%llu.hc_ffn_base.weight"},
    {YVEX_TENSOR_ROLE_HC_FFN_SCALE, "blk.%llu.hc_ffn_scale.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV,
     "blk.%llu.attn_compressor_kv.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE,
     "blk.%llu.attn_compressor_gate.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE,
     "blk.%llu.attn_compressor_ape.weight"},
    {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM,
     "blk.%llu.attn_compressor_norm.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_PROJECTION, "blk.%llu.indexer.proj.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
     "blk.%llu.indexer.attn_q_b.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV,
     "blk.%llu.indexer_compressor_kv.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE,
     "blk.%llu.indexer_compressor_gate.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE,
     "blk.%llu.indexer_compressor_ape.weight"},
    {YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM,
     "blk.%llu.indexer_compressor_norm.weight"},
    {YVEX_TENSOR_ROLE_MOE_ROUTER, "blk.%llu.ffn_gate_inp.weight"},
    {YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS, "blk.%llu.exp_probs_b.bias"},
    {YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
     "blk.%llu.ffn_gate_tid2eid.weight"},
    {YVEX_TENSOR_ROLE_MOE_EXPERT_GATE,
     "blk.%llu.ffn_gate_exps.weight"},
    {YVEX_TENSOR_ROLE_MOE_EXPERT_DOWN,
     "blk.%llu.ffn_down_exps.weight"},
    {YVEX_TENSOR_ROLE_MOE_EXPERT_UP, "blk.%llu.ffn_up_exps.weight"},
    {YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_GATE,
     "blk.%llu.ffn_gate_shexp.weight"},
    {YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN,
     "blk.%llu.ffn_down_shexp.weight"},
    {YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_UP,
     "blk.%llu.ffn_up_shexp.weight"}
};

static const char *fixture_pinned_pattern(yvex_tensor_role role)
{
    size_t i;
    for (i = 0u;
         i < sizeof(fixture_pinned_names) / sizeof(fixture_pinned_names[0]);
         ++i) {
        if (fixture_pinned_names[i].role == role)
            return fixture_pinned_names[i].pattern;
    }
    return NULL;
}

static int test_mapping_target_scale(void)
{
    static const struct {
        yvex_tensor_role role;
        yvex_tensor_scope scope;
        unsigned long long layer;
        unsigned long long predictor;
        const char *name;
    } pinned_names[] = {
        {YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
         YVEX_TENSOR_SCOPE_GLOBAL,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
         "token_embd.weight"},
        {YVEX_TENSOR_ROLE_HC_HEAD_SCALE,
         YVEX_TENSOR_SCOPE_GLOBAL,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, YVEX_DEEPSEEK_GGUF_NO_INDEX,
         "output_hc_scale.weight"},
        {YVEX_TENSOR_ROLE_ATTENTION_KV_NORM,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 0u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.0.attn_kv_a_norm.weight"},
        {YVEX_TENSOR_ROLE_MOE_ROUTER_TABLE,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 0u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.0.ffn_gate_tid2eid.weight"},
        {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 2u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.2.attn_compressor_kv.weight"},
        {YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 2u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.2.indexer.proj.weight"},
        {YVEX_TENSOR_ROLE_MOE_ROUTER_BIAS,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 6u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.6.exp_probs_b.bias"},
        {YVEX_TENSOR_ROLE_MOE_SHARED_EXPERT_DOWN,
         YVEX_TENSOR_SCOPE_MAIN_LAYER, 42u,
         YVEX_DEEPSEEK_GGUF_NO_INDEX, "blk.42.ffn_down_shexp.weight"}
    };
    yvex_deepseek_gguf_map *map = NULL;
    yvex_deepseek_gguf_map *shuffled = NULL;
    yvex_deepseek_gguf_map_failure failure;
    const yvex_deepseek_gguf_map_summary *summary;
    const yvex_deepseek_gguf_descriptor *descriptor;
    const yvex_deepseek_gguf_metadata *metadata;
    unsigned long long trunk_counts[YVEX_TENSOR_COLLECTION_COUNT] = {0};
    unsigned long long descriptor_index;
    unsigned long long contribution_index;
    size_t reference_index;
    unsigned long long identity;

    YVEX_TEST_ASSERT(fixture_build_map(0, &map, &failure, NULL) == YVEX_OK &&
                         map,
                     "complete source-to-GGUF mapping builds");
    summary = yvex_model_register_deepseek_v4()->lowering.summary(map);
    YVEX_TEST_ASSERT(summary && summary->complete &&
                         summary->source_contribution_count == 72317u &&
                         summary->descriptor_count == 1409u &&
                         summary->trunk_descriptor_count == 1328u &&
                         summary->draft_descriptor_count == 81u &&
                         summary->metadata_count == 53u,
                     "mapping reconciles exact source and descriptor counts");
    YVEX_TEST_ASSERT(summary->pinned_standard_count == 1328u &&
                         summary->semantic_standard_count == 0u &&
                         summary->extension_count == 81u &&
                         summary->header_scan_count == 1u &&
                         summary->payload_bytes_read == 0u,
                     "official target and DSpark extension boundary is exact");
    for (descriptor_index = 0u;
         descriptor_index < summary->descriptor_count; ++descriptor_index) {
        const yvex_deepseek_gguf_descriptor *current =
            yvex_model_register_deepseek_v4()->lowering.at(map, descriptor_index);
        unsigned long long local;
        YVEX_TEST_ASSERT(current && current->logical_rank > 0u &&
                             current->logical_rank <= YVEX_TENSOR_MAX_DIMS &&
                             current->contribution_count > 0u,
                         "every descriptor has logical geometry and sources");
        YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->lowering.find_emitted(
                             map, current->emitted_name) == current &&
                             yvex_model_register_deepseek_v4()->lowering.find_role(
                                 map, current->role, current->scope,
                                 current->layer_index,
                                 current->predictor_index) == current,
                         "emitted-name and typed-role indexes are exact");
        if (current->scope == YVEX_TENSOR_SCOPE_DRAFT) {
            YVEX_TEST_ASSERT(
                current->name_provenance == YVEX_GGUF_NAME_YVEX_EXTENSION &&
                    strncmp(current->emitted_name, "yvex.draft.v1.",
                            strlen("yvex.draft.v1.")) == 0,
                "every DSpark descriptor uses the versioned extension");
        } else {
            const char *pattern = fixture_pinned_pattern(current->role);
            const char *layer_slot;
            char expected_name[192];
            int written;

            YVEX_TEST_ASSERT(
                current->name_provenance == YVEX_GGUF_NAME_PINNED_STANDARD,
                "every trunk descriptor uses a pinned standard name");
            YVEX_TEST_ASSERT(pattern != NULL,
                             "every trunk role exists in the pinned oracle");
            layer_slot = strstr(pattern, "%llu");
            written = layer_slot
                          ? snprintf(expected_name, sizeof(expected_name), "%.*s%llu%s",
                                     (int)(layer_slot - pattern), pattern,
                                     current->layer_index, layer_slot + 4)
                          : snprintf(expected_name, sizeof(expected_name), "%s", pattern);
            YVEX_TEST_ASSERT(written > 0 &&
                                 (size_t)written < sizeof(expected_name) &&
                                 strcmp(current->emitted_name,
                                        expected_name) == 0,
                             "every trunk name matches the pinned oracle");
            trunk_counts[current->collection]++;
        }
        for (local = 0u; local < current->contribution_count; ++local) {
            const yvex_deepseek_gguf_contribution *contribution =
                yvex_model_register_deepseek_v4()->lowering.contribution_at(
                    map, current->contribution_offset + local);
            YVEX_TEST_ASSERT(contribution &&
                                 contribution->descriptor_index ==
                                     descriptor_index &&
                                 yvex_model_register_deepseek_v4()->lowering.find_source(
                                     map, contribution->source_name) == current,
                             "every source contribution resolves to one descriptor");
            if (current->transform ==
                YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4) {
                YVEX_TEST_ASSERT(
                    contribution->expert_index == local / 2u &&
                        contribution->kind ==
                            (local % 2u == 0u
                                 ? YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_WEIGHT
                                 : YVEX_DEEPSEEK_GGUF_CONTRIBUTION_EXPERT_SCALE),
                    "expert members are ordered weight/scale by numeric expert");
            }
        }
        if (current->transform ==
            YVEX_DEEPSEEK_GGUF_TRANSFORM_FP8_E4M3_E8M0) {
            const yvex_deepseek_gguf_contribution *primary =
                yvex_model_register_deepseek_v4()->lowering.contribution_at(
                    map, current->contribution_offset);
            const yvex_deepseek_gguf_contribution *scale =
                yvex_model_register_deepseek_v4()->lowering.contribution_at(
                    map, current->contribution_offset + 1u);
            YVEX_TEST_ASSERT(current->contribution_count == 2u && primary &&
                                 scale &&
                                 primary->kind ==
                                     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_PRIMARY &&
                                 primary->source_dtype ==
                                     YVEX_NATIVE_DTYPE_F8_E4M3 &&
                                 scale->kind ==
                                     YVEX_DEEPSEEK_GGUF_CONTRIBUTION_SCALE &&
                                 scale->source_dtype ==
                                     YVEX_NATIVE_DTYPE_F8_E8M0,
                             "every FP8 descriptor has its exact E8M0 companion");
        }
        if (current->transform !=
            YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4) {
            const yvex_deepseek_gguf_contribution *primary =
                yvex_model_register_deepseek_v4()->lowering.contribution_at(
                    map, current->contribution_offset);
            unsigned int dimension;

            YVEX_TEST_ASSERT(primary &&
                                 primary->source_rank == current->logical_rank,
                             "direct logical rank matches the pinned converter");
            for (dimension = 0u; dimension < current->logical_rank;
                 ++dimension) {
                YVEX_TEST_ASSERT(
                    current->logical_dims[dimension] ==
                        primary->source_dims[current->logical_rank -
                                             dimension - 1u] &&
                        current->source_axis_for_logical[dimension] ==
                            current->logical_rank - dimension - 1u,
                    "direct logical axes match GGML reversed source order");
            }
        }
    }
    YVEX_TEST_ASSERT(
        trunk_counts[YVEX_TENSOR_COLLECTION_GLOBAL] == 6u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_ATTENTION] == 344u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_MHC] == 258u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_NORM] == 86u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_ROUTED_EXPERT] == 129u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_SHARED_EXPERT] == 129u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_ROUTER] == 86u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_COMPRESSOR] == 164u &&
            trunk_counts[YVEX_TENSOR_COLLECTION_INDEXER] == 126u,
        "official trunk collection reconciliation is exact");
    for (reference_index = 0u;
         reference_index < sizeof(pinned_names) / sizeof(pinned_names[0]);
         ++reference_index) {
        const yvex_deepseek_gguf_descriptor *reference =
            yvex_model_register_deepseek_v4()->lowering.find_role(
                map, pinned_names[reference_index].role,
                pinned_names[reference_index].scope,
                pinned_names[reference_index].layer,
                pinned_names[reference_index].predictor);
        YVEX_TEST_ASSERT(
            reference &&
                strcmp(reference->emitted_name,
                       pinned_names[reference_index].name) == 0,
            "compact pinned llama.cpp name oracle matches each owner family");
    }
    for (contribution_index = 0u;
         contribution_index < summary->source_contribution_count;
         ++contribution_index) {
        const yvex_deepseek_gguf_contribution *contribution =
            yvex_model_register_deepseek_v4()->lowering.contribution_at(map, contribution_index);
        YVEX_TEST_ASSERT(contribution &&
                             contribution->source_row_index < 72317u,
                         "descriptor index iteration covers all source rows");
    }
    descriptor = yvex_model_register_deepseek_v4()->lowering.find_source(
        map, "layers.2.ffn.experts.255.w1.scale");
    YVEX_TEST_ASSERT(descriptor &&
                         descriptor->role == YVEX_TENSOR_ROLE_MOE_EXPERT_GATE &&
                         descriptor->transform ==
                             YVEX_DEEPSEEK_GGUF_TRANSFORM_EXPERT_MXFP4 &&
                         descriptor->forced_qtype == 39u &&
                         descriptor->logical_rank == 3u &&
                         descriptor->logical_dims[0] == 4096u &&
                         descriptor->logical_dims[1] == 2048u &&
                         descriptor->logical_dims[2] == 256u &&
                         descriptor->source_axis_for_logical[2] ==
                             YVEX_DEEPSEEK_GGUF_AGGREGATED_AXIS &&
                         descriptor->contribution_count == 512u &&
                         strcmp(descriptor->emitted_name,
                                "blk.2.ffn_gate_exps.weight") == 0,
                     "routed experts aggregate and repack to official MXFP4 shape");
    descriptor = yvex_model_register_deepseek_v4()->lowering.find_emitted(
        map, "blk.0.ffn_gate_tid2eid.weight");
    YVEX_TEST_ASSERT(descriptor &&
                         descriptor->transform ==
                             YVEX_DEEPSEEK_GGUF_TRANSFORM_I64_TO_I32 &&
                         descriptor->forced_qtype == 26u &&
                         descriptor->logical_dims[0] == 6u &&
                         descriptor->logical_dims[1] == 129280u,
                     "hash routing table has checked official I32 projection");
    descriptor = yvex_model_register_deepseek_v4()->lowering.find_role(
        map, YVEX_TENSOR_ROLE_ATTENTION_Q_A,
        YVEX_TENSOR_SCOPE_MAIN_LAYER, 42u,
        YVEX_DEEPSEEK_GGUF_NO_INDEX);
    YVEX_TEST_ASSERT(descriptor &&
                         strcmp(descriptor->emitted_name,
                                "blk.42.attn_q_a.weight") == 0 &&
                         descriptor->logical_dims[0] == 4096u &&
                         descriptor->logical_dims[1] == 1024u &&
                         descriptor->name_provenance ==
                             YVEX_GGUF_NAME_PINNED_STANDARD,
                     "official attention name and logical axes match loader");
    descriptor = yvex_model_register_deepseek_v4()->lowering.find_source(map, "mtp.0.main_proj.scale");
    YVEX_TEST_ASSERT(descriptor &&
                         descriptor->scope == YVEX_TENSOR_SCOPE_DRAFT &&
                         descriptor->name_provenance ==
                             YVEX_GGUF_NAME_YVEX_EXTENSION &&
                         strncmp(descriptor->emitted_name, "yvex.draft.v1.0.",
                                 strlen("yvex.draft.v1.0.")) == 0,
                     "DSpark survives under an explicit versioned extension");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "deepseek4.attention.compress_ratios");
    YVEX_TEST_ASSERT(metadata &&
                         metadata->type == YVEX_DEEPSEEK_GGUF_METADATA_U64_ARRAY &&
                         metadata->array_count == 43u &&
                         metadata->array_values[2] == 4u &&
                         metadata->array_values[3] == 128u,
                     "metadata plan preserves the complete trunk schedule");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "yvex.deepseek4.dspark.runtime_supported");
    YVEX_TEST_ASSERT(metadata &&
                         metadata->type == YVEX_DEEPSEEK_GGUF_METADATA_BOOL &&
                         metadata->bool_value,
                     "DSpark metadata records the admitted target and draft runtime contract");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "deepseek4.attention.layer_norm_rms_epsilon");
    YVEX_TEST_ASSERT(metadata &&
                         metadata->type == YVEX_DEEPSEEK_GGUF_METADATA_F64 &&
                         metadata->f64_value > 0.0,
                     "official loader RMS epsilon is planned");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "deepseek4.expert_gating_func");
    YVEX_TEST_ASSERT(metadata &&
                         metadata->type == YVEX_DEEPSEEK_GGUF_METADATA_U64 &&
                         metadata->u64_value == 4u,
                     "official sqrt-softplus expert gating enum is planned");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "deepseek4.swiglu_clamp_exp");
    YVEX_TEST_ASSERT(
        metadata &&
            metadata->type == YVEX_DEEPSEEK_GGUF_METADATA_F64_ARRAY &&
            metadata->array_count == 43u &&
            metadata->f64_array_values[0] > 0.0,
        "official per-layer expert clamp metadata is complete");
    metadata = yvex_model_register_deepseek_v4()->lowering.metadata_find(
        map, "deepseek4.hyper_connection.count");
    YVEX_TEST_ASSERT(metadata && metadata->u64_value == 4u &&
                         !yvex_model_register_deepseek_v4()->lowering.metadata_find(
                             map, "deepseek4.hyper_connection_count"),
                     "hyper-connection metadata uses the pinned canonical key");
    for (descriptor_index = 0u;
         descriptor_index < summary->metadata_count; ++descriptor_index) {
        unsigned long long other;
        const yvex_deepseek_gguf_metadata *entry =
            yvex_model_register_deepseek_v4()->lowering.metadata_at(map, descriptor_index);
        YVEX_TEST_ASSERT(entry && entry->key[0],
                         "every planned metadata entry has a key");
        for (other = descriptor_index + 1u;
             other < summary->metadata_count; ++other) {
            const yvex_deepseek_gguf_metadata *candidate =
                yvex_model_register_deepseek_v4()->lowering.metadata_at(map, other);
            YVEX_TEST_ASSERT(candidate && strcmp(entry->key, candidate->key) != 0,
                             "metadata keys are unique");
        }
    }
    identity = summary->mapping_identity;
    YVEX_TEST_ASSERT(fixture_build_map(1, &shuffled, &failure, NULL) == YVEX_OK &&
                         shuffled &&
                         yvex_model_register_deepseek_v4()->lowering.summary(shuffled)->
                                 mapping_identity == identity,
                     "mapping identity is source-discovery-order invariant");
    yvex_model_register_deepseek_v4()->lowering.close(shuffled);
    yvex_model_register_deepseek_v4()->lowering.close(map);
    return 0;
}

static int test_mapping_allocation_rollback(void)
{
    unsigned int fail_at;
    for (fail_at = 0u; fail_at < 6u; ++fail_at) {
        fixture_allocator state;
        yvex_deepseek_gguf_map_allocator allocator;
        yvex_deepseek_gguf_map *map = NULL;
        yvex_deepseek_gguf_map_failure failure;
        memset(&state, 0, sizeof(state));
        state.fail_at = fail_at;
        allocator.allocate = fixture_allocate;
        allocator.release = fixture_release;
        allocator.context = &state;
        YVEX_TEST_ASSERT(fixture_build_map(0, &map, &failure, &allocator) !=
                             YVEX_OK && !map,
                         "mapping allocation failure refuses publication");
        YVEX_TEST_ASSERT(state.live == 0u,
                         "partial mapping allocation rolls back");
    }
    return 0;
}

static int test_mapping_refusal_boundary(void)
{
    yvex_deepseek_gguf_map *map = NULL;
    yvex_deepseek_gguf_map_failure failure;
    yvex_error err;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->lowering.build(
                         &map, NULL, NULL, &failure, &err) != YVEX_OK &&
                         !map &&
                         failure.code ==
                             YVEX_DEEPSEEK_GGUF_MAP_FAILURE_INVALID_ARGUMENT,
                     "mapping refuses missing typed prerequisites");
    return 0;
}

static int test_mapping_typed_owner_refusals(void)
{
    char name[16];
    unsigned long long valid_mxfp4[3] = {4096u, 2048u, 256u};
    unsigned long long invalid_mxfp4[3] = {4095u, 2048u, 256u};
    yvex_gguf_name_provenance provenance;
    const char *reason = NULL;

    YVEX_TEST_ASSERT(
        !yvex_gguf_name_map_resolve(YVEX_TENSOR_ROLE_UNKNOWN, 0, 0u, 0u,
                                    name, sizeof(name), &provenance, &reason),
        "unknown typed role refuses emitted naming");
    YVEX_TEST_ASSERT(
        !yvex_gguf_name_map_resolve(YVEX_TENSOR_ROLE_ATTENTION_Q_A, 0, 42u,
                                    0u, name, sizeof(name), &provenance,
                                    &reason),
        "bounded emitted-name buffer refuses truncation");
    YVEX_TEST_ASSERT(
        yvex_gguf_layout_map_shape_supported(
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 39u, 3u, valid_mxfp4,
            &reason),
        "canonical MXFP4 expert geometry is admitted");
    YVEX_TEST_ASSERT(
        !yvex_gguf_layout_map_shape_supported(
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 39u, 3u, invalid_mxfp4,
            &reason),
        "row-incompatible MXFP4 expert geometry refuses");
    YVEX_TEST_ASSERT(
        !yvex_gguf_layout_map_shape_supported(
            YVEX_TENSOR_ROLE_MOE_EXPERT_GATE, 39u, 0u, valid_mxfp4,
            &reason),
        "zero-rank logical layout refuses");
    return 0;
}

static int test_transform_stale_coverage_refusals(void)
{
    yvex_source_verification verification;
    yvex_source_tensor_snapshot *snapshot = NULL;
    yvex_deepseek_v4_ir *ir = NULL;
    yvex_deepseek_v4_ir_failure ir_failure;
    yvex_deepseek_tensor_coverage *coverage = NULL;
    yvex_deepseek_tensor_coverage_failure coverage_failure;
    yvex_transform_ir *transform_ir = NULL;
    yvex_transform_failure failure;
    yvex_deepseek_gguf_map *map = NULL;
    yvex_deepseek_gguf_map_failure map_failure;
    yvex_transform_node *node;
    yvex_transform_operation_kind saved_kind;
    yvex_deepseek_tensor_coverage_row *row;
    yvex_native_weight_info *source;
    yvex_native_dtype saved_dtype;
    yvex_tensor_scope saved_scope;
    unsigned long long saved_expert;
    yvex_error err;

    fixture_verification(&verification);
    YVEX_TEST_ASSERT(fixture_snapshot(&snapshot, &verification,
                                      FIXTURE_VALID, 0) == YVEX_OK,
                     "stale-coverage fixture snapshot builds");
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->ir.build(
                         &ir, &verification, &ir_failure, &err) == YVEX_OK,
                     "stale-coverage fixture IR builds");
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->coverage.build(
                         &coverage, &verification, ir, snapshot, NULL,
                         &coverage_failure, &err) == YVEX_OK,
                     "stale-coverage fixture closes before mutation");

    row = (yvex_deepseek_tensor_coverage_row *)
        yvex_model_register_deepseek_v4()->coverage.find(coverage, "head.weight");
    source = row ? (yvex_native_weight_info *)row->source : NULL;
    YVEX_TEST_ASSERT(source != NULL, "dtype mutation source is addressable");
    saved_dtype = source->dtype;
    source->dtype = YVEX_NATIVE_DTYPE_F32;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->transform.build(
                         &transform_ir, &verification, ir, coverage, NULL,
                         &failure, &err) != YVEX_OK && !transform_ir &&
                         failure.code ==
                             YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE,
                     "post-coverage source dtype drift refuses IR construction");
    source->dtype = saved_dtype;

    row = (yvex_deepseek_tensor_coverage_row *)
        yvex_model_register_deepseek_v4()->coverage.find(
            coverage, "layers.0.ffn.experts.1.w1.weight");
    YVEX_TEST_ASSERT(row != NULL, "expert mutation row is addressable");
    saved_expert = row->expert_index;
    row->expert_index = 2u;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->transform.build(
                         &transform_ir, &verification, ir, coverage, NULL,
                         &failure, &err) != YVEX_OK && !transform_ir &&
                         failure.code == YVEX_TRANSFORM_FAILURE_UNEXPECTED_SOURCE,
                     "post-coverage expert sequence drift refuses IR construction");
    row->expert_index = saved_expert;

    row = (yvex_deepseek_tensor_coverage_row *)
        yvex_model_register_deepseek_v4()->coverage.find(coverage,
                                           "layers.0.attn_norm.weight");
    YVEX_TEST_ASSERT(row != NULL, "scope mutation row is addressable");
    saved_scope = row->scope;
    row->scope = YVEX_TENSOR_SCOPE_GLOBAL;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->transform.build(
                         &transform_ir, &verification, ir, coverage, NULL,
                         &failure, &err) != YVEX_OK && !transform_ir &&
                         failure.code == YVEX_TRANSFORM_FAILURE_UNEXPECTED_SOURCE,
                     "post-coverage typed scope drift refuses IR construction");
    row->scope = saved_scope;

    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->transform.build(
                         &transform_ir, &verification, ir, coverage, NULL,
                         &failure, &err) == YVEX_OK && transform_ir,
                     "restored coverage seals one valid Transformation IR");
    node = (yvex_transform_node *)yvex_transform_ir_node_at(transform_ir, 0u);
    YVEX_TEST_ASSERT(node != NULL,
                     "sealed lowering fixture exposes an immutable node view");
    saved_kind = node->kind;
    node->kind = YVEX_TRANSFORM_OP_RESHAPE;
    YVEX_TEST_ASSERT(yvex_model_register_deepseek_v4()->lowering.build(
                         &map, ir, transform_ir, &map_failure, &err) != YVEX_OK &&
                         !map &&
                         map_failure.code ==
                             YVEX_DEEPSEEK_GGUF_MAP_FAILURE_LOWERING_DIVERGENCE,
                     "GGUF lowering refuses an unsupported semantic operation");
    node->kind = saved_kind;

    yvex_transform_ir_release(&transform_ir);
    yvex_model_register_deepseek_v4()->coverage.close(coverage);
    yvex_model_register_deepseek_v4()->ir.close(ir);
    yvex_source_tensor_snapshot_release(snapshot);
    return 0;
}

int yvex_test_deepseek_tensor_coverage(void)
{
    if (test_valid_target_scale() != 0) return 1;
    if (test_mutations() != 0) return 1;
    if (test_limits_and_allocation() != 0) return 1;
    if (test_transform_target_scale() != 0) return 1;
    if (test_mapping_target_scale() != 0) return 1;
    if (test_mapping_allocation_rollback() != 0) return 1;
    if (test_mapping_refusal_boundary() != 0) return 1;
    if (test_mapping_typed_owner_refusals() != 0) return 1;
    if (test_transform_stale_coverage_refusals() != 0) return 1;
    return 0;
}
