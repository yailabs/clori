/*
 * Source verification owners share retained metadata, provenance, inventory, and publication
 * lifecycles here. Metadata admission remains distinct from payload trust.
 */
#ifndef INCLUDE_YVEX_INTERNAL_SOURCE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SOURCE_H_INCLUDED

#include <stddef.h>
#include <string.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source_catalog.h>
#include <yvex/source.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return the final non-empty component of one borrowed source path. */
static inline const char *yvex_source_path_basename(const char *path)
{
    const char *slash;

    if (!path || !path[0]) return NULL;
    slash = strrchr(path, '/');
    return slash && slash[1] ? slash + 1 : path;
}

/* Match one source target label against an admitted family prefix. */
static inline int yvex_source_target_matches_family_name(const char *family,
                                                         const char *target)
{
    if (!family || !target) return 0;
    if (strcmp(family, "qwen") == 0) return strncmp(target, "qwen", 4u) == 0;
    if (strcmp(family, "gemma") == 0) return strncmp(target, "gemma", 5u) == 0;
    return 0;
}

/* Admit one normalized root-relative shard path for source payload sessions. */
static inline int yvex_source_payload_name_is_canonical(const char *name)
{
    const char *cursor;
    size_t segment = 0u;
    if (!name || !name[0] || name[0] == '/' || strcmp(name, ".") == 0 ||
        strcmp(name, "..") == 0)
        return 0;
    for (cursor = name; *cursor; ++cursor) {
        if (*cursor == '\\' || *cursor == '\n' || *cursor == '\r') return 0;
        if (*cursor == '/') {
            if (!segment || (segment == 1u && cursor[-1] == '.') ||
                (segment == 2u && cursor[-1] == '.' && cursor[-2] == '.')) return 0;
            segment = 0u;
        } else segment++;
    }
    return segment && !(segment == 1u && cursor[-1] == '.') &&
           !(segment == 2u && cursor[-1] == '.' && cursor[-2] == '.');
}

/* Source verification. */
struct yvex_source_tensor_snapshot;
#define YVEX_SOURCE_VERIFY_BLOCKER_CAP 24u
#define YVEX_SOURCE_VERIFY_COMPRESS_RATIO_CAP 128u
#define YVEX_SOURCE_VERIFY_DSPARK_TARGET_LAYER_CAP 8u
typedef struct {
    const yvex_source_target_identity *identity;
    const char *source_path;
    const char *models_root;
    const char *manifest_path;
    const char *upstream_inventory_path;
    const char *derived_inventory_path;
    int promote_manifest;
} yvex_source_verify_options;
typedef struct yvex_source_verification {
    int verified;
    char resolved_source_path[YVEX_PATH_CAP];
    char manifest_path[YVEX_PATH_CAP];
    char manifest_schema[64];
    char manifest_target_id[128];
    char verification_stage[64];
    char inventory_authority[32];
    char upstream_index_oid[64];
    char local_index_oid[64];
    char source_kind[32];
    char repository_id[256];
    char manifest_revision[128];
    char revision[128];
    char manifest_status[32];
    char manifest_config_status[32];
    char manifest_tokenizer_status[32];
    char manifest_payload_digest_status[32];
    int path_verified;
    int repository_verified;
    int revision_verified;
    int manifest_verified;
    int manifest_published;
    int manifest_reopened;
    int upstream_index_identity_verified;
    int config_valid;
    int inference_config_valid;
    int tokenizer_json_valid;
    int tokenizer_config_valid;
    int generation_config_valid;
    int shard_index_present;
    int shard_index_valid;
    int shard_index_headers_match;
    int footprint_overflow;
    char model_type[64];
    char architecture[128];
    char torch_dtype[32];
    char expert_dtype[32];
    char hidden_act[32];
    char scoring_func[32];
    char topk_method[32];
    char tokenizer_model_type[32];
    char rope_scaling_type[32];
    char quant_method[32];
    char quant_format[32];
    char tokenizer_class[64];
    char generation_transformers_version[32];
    char generation_temperature[32];
    char generation_top_p[32];
    unsigned long long hidden_size;
    unsigned long long num_hidden_layers;
    unsigned long long num_attention_heads;
    unsigned long long num_key_value_heads;
    unsigned long long head_dim;
    unsigned long long qk_rope_head_dim;
    unsigned long long max_position_embeddings;
    unsigned long long moe_intermediate_size;
    unsigned long long n_routed_experts;
    unsigned long long n_shared_experts;
    unsigned long long num_experts_per_tok;
    unsigned long long num_hash_layers;
    unsigned long long q_lora_rank;
    unsigned long long o_lora_rank;
    unsigned long long vocab_size;
    unsigned long long sliding_window;
    unsigned long long bos_token_id;
    unsigned long long eos_token_id;
    unsigned long long compress_ratios[YVEX_SOURCE_VERIFY_COMPRESS_RATIO_CAP];
    unsigned long long compress_ratio_count;
    unsigned long long compress_rope_theta;
    char attention_dropout[32];
    char hc_eps[32];
    unsigned long long hc_mult;
    unsigned long long hc_sinkhorn_iters;
    unsigned long long index_head_dim;
    unsigned long long index_n_heads;
    unsigned long long index_topk;
    unsigned long long num_nextn_predict_layers;
    unsigned long long dspark_block_size;
    unsigned long long dspark_noise_token_id;
    unsigned long long dspark_target_layer_ids[
        YVEX_SOURCE_VERIFY_DSPARK_TARGET_LAYER_CAP];
    unsigned long long dspark_target_layer_count;
    unsigned long long dspark_markov_rank;
    unsigned long long dspark_inference_layer_count;
    unsigned long long o_groups;
    char rms_norm_eps[32];
    unsigned long long rope_theta;
    char routed_scaling_factor[32];
    char swiglu_limit[32];
    int attention_bias;
    int norm_topk_prob;
    int use_cache;
    unsigned long long rope_scaling_factor;
    unsigned long long rope_original_context;
    unsigned long long rope_beta_fast;
    unsigned long long rope_beta_slow;
    char quant_activation_scheme[32];
    char quant_scale_format[32];
    unsigned long long quant_block_rows;
    unsigned long long quant_block_columns;
    unsigned long long tokenizer_base_vocab_count;
    unsigned long long tokenizer_added_token_count;
    unsigned long long tokenizer_max_token_id;
    unsigned long long tokenizer_effective_vocab_size;
    unsigned long long tokenizer_model_max_length;
    unsigned long long generation_bos_token_id;
    unsigned long long generation_eos_token_id;
    int tie_word_embeddings;
    int generation_from_model_config;
    int generation_do_sample;
    unsigned long long source_file_count;
    unsigned long long source_total_bytes;
    unsigned long long manifest_source_file_count;
    unsigned long long manifest_source_total_bytes;
    unsigned long long manifest_shard_count;
    unsigned long long manifest_shard_bytes;
    unsigned long long manifest_header_tensor_count;
    unsigned long long shard_count;
    unsigned long long shard_bytes;
    unsigned long long indexed_tensor_count;
    unsigned long long referenced_shard_count;
    unsigned long long header_shard_count;
    unsigned long long header_scan_count;
    unsigned long long header_tensor_count;
    unsigned long long header_bytes;
    unsigned long long declared_tensor_bytes;
    unsigned long long source_snapshot_identity;
    unsigned long long max_tensor_rank;
    unsigned long long dtype_f16_count;
    unsigned long long dtype_bf16_count;
    unsigned long long dtype_f32_count;
    unsigned long long dtype_i64_count;
    unsigned long long dtype_i8_count;
    unsigned long long dtype_fp4_count;
    unsigned long long dtype_f8_count;
    unsigned long long dtype_f8_e8m0_count;
    unsigned long long dtype_other_count;
    char manifest_payload_identity[65];
    char manifest_payload_trust_class[40];
    char manifest_payload_digest_algorithm[24];
    unsigned long long manifest_payload_shard_count;
    unsigned long long manifest_payload_bytes;
    unsigned long long manifest_payload_source_snapshot_identity;
    unsigned long long manifest_payload_tensor_count;
    unsigned long long manifest_payload_logical_tensor_bytes;
    int manifest_payload_trusted;
    const char *blockers[YVEX_SOURCE_VERIFY_BLOCKER_CAP];
    unsigned int blocker_count;
} yvex_source_verification;
int yvex_source_verify(const yvex_source_verify_options *options,
                       yvex_source_verification *out,
                       yvex_error *err);
int yvex_source_verify_with_snapshot(
    const yvex_source_verify_options *options,
    yvex_source_verification *out,
    struct yvex_source_tensor_snapshot **snapshot,
    yvex_error *err);
const char *yvex_source_verification_status(
    const yvex_source_verification *verification);

/* Retained source inventory. */
typedef struct {
    char *tensor;
    char *shard;
} yvex_source_inventory_row;
typedef struct {
    yvex_source_inventory_row *rows;
    size_t count;
} yvex_source_derived_inventory;
typedef struct yvex_source_tensor_snapshot yvex_source_tensor_snapshot;
typedef struct {
    unsigned long long canonical_id;
    const char *canonical_name;
    unsigned long long file_bytes;
    unsigned long long data_region_offset;
    unsigned long long payload_bytes;
} yvex_source_shard_snapshot;
typedef struct {
    unsigned long long tensor_count;
    unsigned long long shard_count;
    unsigned long long header_scan_count;
    unsigned long long payload_bytes_read;
    unsigned long long lookup_count;
    unsigned long long collision_count;
    unsigned long long maximum_probe;
    unsigned long long identity;
} yvex_source_tensor_snapshot_facts;
void yvex_source_derived_inventory_free(
    yvex_source_derived_inventory *inventory);
int yvex_source_inventory_verify(
    const yvex_source_verify_options *options,
    yvex_source_verification *out,
    yvex_source_derived_inventory *derived,
    yvex_source_tensor_snapshot **snapshot,
    yvex_error *err);
void yvex_source_tensor_snapshot_retain(yvex_source_tensor_snapshot *snapshot);
void yvex_source_tensor_snapshot_release(yvex_source_tensor_snapshot *snapshot);
const yvex_native_weight_info *yvex_source_tensor_snapshot_at(
    const yvex_source_tensor_snapshot *snapshot,
    unsigned long long index);
const yvex_native_weight_info *yvex_source_tensor_snapshot_find(
    const yvex_source_tensor_snapshot *snapshot,
    const char *name);
int yvex_source_tensor_snapshot_find_index(
    const yvex_source_tensor_snapshot *snapshot,
    const char *name,
    unsigned long long *index);
int yvex_source_tensor_snapshot_facts_get(
    const yvex_source_tensor_snapshot *snapshot,
    yvex_source_tensor_snapshot_facts *out,
    yvex_error *err);
int yvex_source_tensor_snapshot_take_table(
    yvex_source_tensor_snapshot **out,
    yvex_native_weight_table **table,
    unsigned long long shard_count,
    unsigned long long header_scan_count,
    yvex_error *err);
int yvex_source_tensor_snapshot_take_table_with_shards(
    yvex_source_tensor_snapshot **out,
    yvex_native_weight_table **table,
    const yvex_source_shard_snapshot *shards,
    unsigned long long shard_count,
    unsigned long long header_scan_count,
    yvex_error *err);
const yvex_source_shard_snapshot *yvex_source_tensor_snapshot_shard_at(
    const yvex_source_tensor_snapshot *snapshot,
    unsigned long long index);
const yvex_source_shard_snapshot *yvex_source_tensor_snapshot_shard_find(
    const yvex_source_tensor_snapshot *snapshot,
    const char *canonical_name);
int yvex_source_tensor_snapshot_has_shard_catalog(
    const yvex_source_tensor_snapshot *snapshot);

/* Provider provenance. */
int yvex_source_provenance_manifest_read(
    const yvex_source_verify_options *options,
    yvex_source_verification *out,
    yvex_error *err);
int yvex_source_provenance_verify_file(
    const yvex_source_verify_options *options,
    const char *name,
    int verify_upstream_index,
    yvex_source_verification *out,
    yvex_error *err);
void yvex_source_provenance_finalize(
    const yvex_source_verify_options *options,
    yvex_source_verification *out);
int yvex_source_provenance_manifest_matches(
    const yvex_source_verify_options *options,
    const yvex_source_verification *out);
int yvex_source_git_blob_oid_file(const char *path,
                                  char out_hex[41],
                                  yvex_error *err);
typedef struct {
    int available;
    int revision_matches;
    char algorithm[24];
    char authority[40];
    char expected_digest[65];
} yvex_source_payload_digest_fact;
typedef struct {
    char canonical_name[YVEX_PATH_CAP];
    char revision[65];
    char expected_git_blob_oid[41];
    char observed_git_blob_oid[41];
    unsigned long long file_bytes;
    int revision_matches;
    int identity_verified;
} yvex_source_metadata_identity_fact;
typedef struct {
    yvex_source_metadata_identity_fact identity;
    unsigned char *bytes;
    size_t byte_count;
} yvex_source_metadata_blob;
int yvex_source_provenance_payload_digest(
    const yvex_source_verification *verification,
    const char *canonical_name,
    yvex_source_payload_digest_fact *out,
    yvex_error *err);
int yvex_source_provenance_metadata_read(
    const yvex_source_verification *verification,
    const char *canonical_name,
    size_t maximum_bytes,
    yvex_source_metadata_blob *out,
    yvex_error *err);
void yvex_source_metadata_blob_release(yvex_source_metadata_blob *blob);

/* Immutable multi-component source acquisition admission. */
#define YVEX_SOURCE_ACQUISITION_SCHEMA "yvex.source-acquisition.v1"
#define YVEX_SOURCE_ACQUISITION_MANIFEST "yvex-source-acquisition.json"
#define YVEX_SOURCE_ACQUISITION_FILE_CAP 256u
typedef enum {
    YVEX_SOURCE_ACQUISITION_FILE_METADATA = 0,
    YVEX_SOURCE_ACQUISITION_FILE_SHARD
} yvex_source_acquisition_file_class;
typedef enum {
    YVEX_SOURCE_ACQUISITION_FAILURE_NONE = 0,
    YVEX_SOURCE_ACQUISITION_FAILURE_INVALID_ARGUMENT,
    YVEX_SOURCE_ACQUISITION_FAILURE_MANIFEST_MISSING,
    YVEX_SOURCE_ACQUISITION_FAILURE_MANIFEST_FORMAT,
    YVEX_SOURCE_ACQUISITION_FAILURE_SOURCE_IDENTITY,
    YVEX_SOURCE_ACQUISITION_FAILURE_RESOURCE_BUDGET,
    YVEX_SOURCE_ACQUISITION_FAILURE_PATH,
    YVEX_SOURCE_ACQUISITION_FAILURE_SYMLINK,
    YVEX_SOURCE_ACQUISITION_FAILURE_NON_REGULAR,
    YVEX_SOURCE_ACQUISITION_FAILURE_SIZE,
    YVEX_SOURCE_ACQUISITION_FAILURE_DIGEST,
    YVEX_SOURCE_ACQUISITION_FAILURE_DUPLICATE,
    YVEX_SOURCE_ACQUISITION_FAILURE_IO,
    YVEX_SOURCE_ACQUISITION_FAILURE_ALLOCATION
} yvex_source_acquisition_failure_code;
typedef struct {
    char path[YVEX_PATH_CAP];
    char component[48];
    char expected_sha256[65], actual_sha256[65], git_oid[65], lfs_oid[65], xet_hash[65];
    unsigned long long expected_size;
    unsigned long long actual_size;
    unsigned long long verified_device, verified_inode;
    long long verified_mtime_seconds, verified_ctime_seconds;
    long verified_mtime_nanoseconds, verified_ctime_nanoseconds;
    yvex_source_acquisition_file_class classification;
    int local_identity_verified;
} yvex_source_acquisition_file;
typedef struct {
    const char *source_root;
    const char *expected_repository;
    const char *expected_revision;
    const char *expected_subtree;
    unsigned long long maximum_files;
    unsigned long long maximum_source_bytes;
    int verify_digests;
} yvex_source_acquisition_options;
typedef struct {
    yvex_source_acquisition_failure_code code;
    unsigned long long file_index;
    char path[YVEX_PATH_CAP];
} yvex_source_acquisition_failure;
typedef struct yvex_source_acquisition yvex_source_acquisition;
typedef struct {
    char repository[256];
    char revision[65];
    char subtree[64];
    char acquisition_identity[65];
    unsigned long long file_count;
    unsigned long long shard_count;
    unsigned long long metadata_bytes;
    unsigned long long shard_bytes;
    unsigned long long source_bytes;
    unsigned long long payload_bytes_read;
    int complete;
} yvex_source_acquisition_facts;
void yvex_source_acquisition_options_default(yvex_source_acquisition_options *options);
int yvex_source_acquisition_open(
    yvex_source_acquisition **out,
    const yvex_source_acquisition_options *options,
    yvex_source_acquisition_failure *failure,
    yvex_error *err);
void yvex_source_acquisition_release(yvex_source_acquisition **acquisition);
const yvex_source_acquisition_facts *yvex_source_acquisition_facts_get(
    const yvex_source_acquisition *acquisition);
const yvex_source_acquisition_file *yvex_source_acquisition_file_at(
    const yvex_source_acquisition *acquisition, unsigned long long index);
int yvex_source_acquisition_metadata_read(const yvex_source_acquisition *acquisition,
                                          const char *source_root, const char *canonical_path,
                                          size_t maximum_bytes, yvex_source_metadata_blob *out,
                                          yvex_error *err);
typedef struct {
    const char *component;
    const char *index_path;
    const char *shard_prefix;
    unsigned long long expected_shards;
    unsigned long long expected_tensors;
} yvex_source_component_inventory_spec;
typedef struct {
    unsigned long long component_count;
    unsigned long long indexed_component_count;
    unsigned long long shard_count;
    unsigned long long tensor_count;
    int complete;
} yvex_source_component_inventory_facts;
int yvex_source_component_inventory_verify(
    const char *source_root, const yvex_native_weight_table *table,
    const yvex_source_component_inventory_spec *specs, size_t spec_count,
    yvex_source_component_inventory_facts *facts, yvex_error *err);

/* Manifest publication. */
struct yvex_source_payload_session;
typedef enum {
    YVEX_SOURCE_PUBLICATION_VERIFIED_MANIFEST = 0,
    YVEX_SOURCE_PUBLICATION_PAYLOAD_MANIFEST = 1,
    YVEX_SOURCE_PUBLICATION_DERIVED_INVENTORY = 2
} yvex_source_publication_kind;
typedef struct {
    yvex_source_publication_kind kind;
    const char *out_path;
    const yvex_source_verify_options *options;
    const yvex_source_verification *verification;
    const yvex_source_derived_inventory *inventory;
    const struct yvex_source_payload_session *payload_session;
} yvex_source_publication_request;
int yvex_source_publish(const yvex_source_publication_request *request,
                        yvex_error *err);

typedef struct {
    char *path;
    unsigned long long size_bytes;
    const char *kind;
} yvex_source_manifest_file;
typedef struct {
    yvex_source_manifest_file *items;
    size_t count;
    size_t cap;
    yvex_source_manifest_summary summary;
} yvex_source_manifest_file_list;
struct yvex_native_weight_table {
    yvex_native_weight_info *items;
    unsigned long long count;
    unsigned long long cap;
    yvex_native_weight_summary summary;
    unsigned long long header_read_count;
    unsigned long long header_error_count;
    unsigned long long header_bytes;
    unsigned long long *name_slots;
    size_t name_slot_count;
    unsigned long long lookup_count;
    unsigned long long collision_count;
    unsigned long long maximum_probe;
    int finalized;
};
void yvex_source_manifest_file_list_init(yvex_source_manifest_file_list *list);
void yvex_source_manifest_file_list_free(yvex_source_manifest_file_list *list);
int yvex_source_manifest_scan_files(const char *local_path,
                                    int include_files,
                                    yvex_source_manifest_file_list *out,
                                    yvex_error *err);
int yvex_native_weight_table_add(yvex_native_weight_table *table,
                                 const char *name,
                                 const char *shard_path,
                                 const char *dtype_name,
                                 unsigned int rank,
                                 const unsigned long long *dims,
                                 unsigned long long data_start,
                                 unsigned long long data_end,
                                 yvex_error *err);
int yvex_native_weight_table_finalize(yvex_native_weight_table *table,
                                      yvex_error *err);
int yvex_safetensors_read_header_file(const char *abs_path,
                                      const char *shard_path,
                                      yvex_native_weight_table *table,
                                      yvex_error *err);
typedef struct {
    unsigned long long file_bytes;
    unsigned long long header_json_bytes;
    unsigned long long data_region_offset;
    unsigned long long payload_bytes;
} yvex_safetensors_file_facts;
int yvex_safetensors_read_header_file_with_facts(
    const char *abs_path,
    const char *shard_path,
    yvex_native_weight_table *table,
    yvex_safetensors_file_facts *facts,
    yvex_error *err);
void yvex_source_verification_add_blocker(yvex_source_verification *out,
                                          const char *reason);
int yvex_source_verification_has_blocker(
    const yvex_source_verification *out,
    const char *reason);
int yvex_source_path_join(char *out, size_t cap, const char *left, const char *right);
char *yvex_source_path_alloc(const char *left, const char *right);
int yvex_source_ends_with(const char *text, const char *suffix);
int yvex_source_regular_file(const char *path, unsigned long long *size);
int yvex_source_revision_is_commit(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_SOURCE_H_INCLUDED */
