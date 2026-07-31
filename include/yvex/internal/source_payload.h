/*
 * Source owners exchange admitted shard handles, bounded payload plans, and delivery receipts
 * through this ABI. Reading a range never implies transformation or artifact support.
 */
#ifndef INCLUDE_YVEX_INTERNAL_SOURCE_PAYLOAD_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_SOURCE_PAYLOAD_H_INCLUDED

#include <pthread.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <yvex/core.h>
#include <yvex/source.h>
#include <yvex/internal/core.h>
#include <yvex/internal/source.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Payload sessions and bounded reads. */
#define YVEX_SOURCE_PAYLOAD_IDENTITY_CAP 65u
#define YVEX_SOURCE_PAYLOAD_DIGEST_CAP 65u
#define YVEX_SOURCE_PAYLOAD_DEFAULT_CHUNK_BYTES (8u * 1024u * 1024u)
#define YVEX_SOURCE_PAYLOAD_MIN_CHUNK_BYTES 4096u
#define YVEX_SOURCE_PAYLOAD_MAX_CHUNK_BYTES (64u * 1024u * 1024u)
typedef enum {
    YVEX_SOURCE_PAYLOAD_TRUST_NONE = 0,
    YVEX_SOURCE_PAYLOAD_TRUST_LOCAL_SNAPSHOT_SEALED,
    YVEX_SOURCE_PAYLOAD_TRUST_UPSTREAM_VERIFIED
} yvex_source_payload_trust_class;
typedef enum {
    YVEX_SOURCE_PAYLOAD_STATE_CONSTRUCTING = 0,
    YVEX_SOURCE_PAYLOAD_STATE_UNTRUSTED,
    YVEX_SOURCE_PAYLOAD_STATE_VERIFYING,
    YVEX_SOURCE_PAYLOAD_STATE_READY,
    YVEX_SOURCE_PAYLOAD_STATE_POISONED,
    YVEX_SOURCE_PAYLOAD_STATE_CLOSED
} yvex_source_payload_state;
typedef struct {
    unsigned long long canonical_id;
    const char *canonical_name;
    unsigned long long file_bytes, data_region_offset, payload_bytes;
    const char *digest_algorithm;
    const char *digest_authority;
    const char *expected_digest;
    const char *observed_digest;
    yvex_source_payload_trust_class trust_class;
} yvex_source_payload_shard;
typedef struct {
    unsigned long long source_tensor_index;
    const char *source_tensor_name;
    unsigned long long shard_index, data_region_offset, relative_begin, relative_end, absolute_begin;
    unsigned long long absolute_end, byte_length;
    yvex_native_dtype dtype;
    unsigned int rank;
    unsigned long long dims[YVEX_NATIVE_WEIGHT_MAX_DIMS], source_snapshot_identity;
    const char *payload_identity;
    yvex_source_payload_trust_class trust_class;
} yvex_source_payload_range;
typedef struct {
    unsigned long long ordinal, range_ordinal, source_tensor_index;
    const char *source_tensor_name;
    unsigned long long shard_index, logical_offset, absolute_offset;
    size_t byte_length;
    unsigned long long first_page, last_page;
} yvex_source_payload_chunk;
typedef struct yvex_source_payload_session yvex_source_payload_session;
typedef struct yvex_source_payload_plan yvex_source_payload_plan;
typedef struct {
    const yvex_source_verify_options *verification_options;
    const yvex_source_verification *verification;
    yvex_source_tensor_snapshot *snapshot;
    yvex_source_payload_budget budget;
    const char *manifest_path;
} yvex_source_payload_open_options;
typedef struct {
    unsigned long long range_count, chunk_count, logical_bytes, page_count;
    size_t chunk_bytes, page_bytes;
} yvex_source_payload_plan_summary;
typedef struct {
    unsigned long long requested_logical_bytes, delivered_logical_bytes, physical_bytes_read;
    unsigned long long trust_bytes_read, chunks_attempted, chunks_completed;
    int complete, committed, aborted;
} yvex_source_payload_stream_result;
typedef struct {
    int (*begin)(void *context, const yvex_source_payload_plan_summary *summary);
    int (*chunk)(void *context,
                 const yvex_source_payload_chunk *chunk,
                 const unsigned char *bytes);
    int (*commit)(void *context,
                  const yvex_source_payload_stream_result *result);
    void (*abort)(void *context,
                  const yvex_source_payload_failure *failure,
                  const yvex_source_payload_stream_result *result);
    void *context;
} yvex_source_payload_sink;
typedef struct {
    yvex_source_payload_state state;
    yvex_source_payload_trust_class trust_class;
    unsigned long long source_snapshot_identity;
    char admitted_payload_identity[YVEX_SOURCE_PAYLOAD_IDENTITY_CAP];
    char payload_identity[YVEX_SOURCE_PAYLOAD_IDENTITY_CAP];
    unsigned long long shard_count, trusted_shard_count, tensor_count, logical_tensor_bytes;
    unsigned long long handle_cache_hits, handle_cache_misses, handle_opens, handle_reopens, handle_evictions;
    unsigned long long short_reads, digest_mismatches, identity_drifts, cancellations, failed_streams;
    unsigned long long active_streams, active_plans, peak_active_streams, peak_inflight_host_bytes;
    unsigned long long buffer_allocations, buffer_reuses, buffer_releases;
    unsigned int open_handles, peak_open_handles;
    unsigned long long range_lookup_count, shard_lookup_count, header_scan_count;
} yvex_source_payload_session_facts;
typedef enum {
    YVEX_SOURCE_PAYLOAD_PROBE_COLD_ADVISORY = 0,
    YVEX_SOURCE_PAYLOAD_PROBE_WARM,
    YVEX_SOURCE_PAYLOAD_PROBE_REPEATED,
    YVEX_SOURCE_PAYLOAD_PROBE_STAGED
} yvex_source_payload_probe_mode;
typedef struct {
    yvex_source_payload_probe_mode mode;
    unsigned int repetitions;
    int page_cache_advice_requested, page_cache_advice_accepted;
    unsigned long long elapsed_nanoseconds, logical_bytes, physical_bytes, chunks, handle_hits, handle_misses;
    unsigned long long buffer_reuses;
} yvex_source_payload_probe_result;
void yvex_source_payload_budget_default(yvex_source_payload_budget *budget);
const char *yvex_source_payload_trust_class_name(
    yvex_source_payload_trust_class trust_class);
int yvex_source_payload_session_open(
    yvex_source_payload_session **out,
    const yvex_source_payload_open_options *options,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_verify(
    yvex_source_payload_session *session,
    const yvex_source_payload_plan *delivery_plan,
    const yvex_source_payload_sink *sink,
    yvex_source_payload_stream_result *result,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_stream(
    yvex_source_payload_session *session,
    const yvex_source_payload_plan *plan,
    const yvex_source_payload_sink *sink,
    yvex_source_payload_stream_result *result,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_cancel(
    yvex_source_payload_session *session,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_reset_cancel(
    yvex_source_payload_session *session,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_close(
    yvex_source_payload_session *session,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_release(
    yvex_source_payload_session **session,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_session_facts_get(
    const yvex_source_payload_session *session,
    yvex_source_payload_session_facts *out,
    yvex_error *err);
const yvex_source_payload_shard *yvex_source_payload_shard_at(
    yvex_source_payload_session *session,
    unsigned long long index);
const yvex_source_payload_range *yvex_source_payload_range_at(
    yvex_source_payload_session *session,
    unsigned long long tensor_index);
const yvex_source_payload_range *yvex_source_payload_range_find(
    yvex_source_payload_session *session,
    const char *tensor_name);
int yvex_source_payload_plan_build(
    yvex_source_payload_plan **out,
    yvex_source_payload_session *session,
    const unsigned long long *tensor_indices,
    unsigned long long tensor_count,
    size_t chunk_bytes,
    size_t page_bytes,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_plan_build_all(
    yvex_source_payload_plan **out,
    yvex_source_payload_session *session,
    size_t chunk_bytes,
    size_t page_bytes,
    yvex_source_payload_failure *failure,
    yvex_error *err);
const yvex_source_payload_plan_summary *yvex_source_payload_plan_summary_get(
    const yvex_source_payload_plan *plan);
const yvex_source_payload_range *yvex_source_payload_plan_range_at(
    const yvex_source_payload_plan *plan,
    unsigned long long ordinal);
const yvex_source_payload_chunk *yvex_source_payload_plan_chunk_at(
    const yvex_source_payload_plan *plan,
    unsigned long long ordinal);
void yvex_source_payload_plan_close(yvex_source_payload_plan *plan);
int yvex_source_payload_probe(
    yvex_source_payload_session *session,
    const yvex_source_payload_plan *plan,
    yvex_source_payload_probe_mode mode,
    unsigned int repetitions,
    yvex_source_payload_probe_result *out,
    yvex_source_payload_failure *failure,
    yvex_error *err);

typedef struct {
    int (*openat_fn)(int, const char *, int);
    int (*fstat_fn)(int, struct stat *);
    int (*fstatat_fn)(int, const char *, struct stat *, int);
    ssize_t (*pread_fn)(int, void *, size_t, off_t);
    int (*close_fn)(int);
    void *(*malloc_fn)(size_t);
    void *(*calloc_fn)(size_t, size_t);
    void (*free_fn)(void *);
} yvex_source_payload_ops;
typedef struct {
    int fd;
    unsigned long long shard_index;
    unsigned int pins;
    unsigned long long use_tick;
} yvex_source_payload_handle;
typedef struct {
    unsigned char *bytes;
    size_t capacity;
    int in_use;
} yvex_source_payload_buffer;
typedef struct {
    dev_t device;
    ino_t inode;
    off_t size;
    struct timespec mtime;
    struct timespec ctime;
} yvex_source_payload_file_identity;
typedef struct {
    yvex_source_payload_shard public_fact;
    char *name;
    char expected_digest[YVEX_SOURCE_PAYLOAD_DIGEST_CAP];
    char observed_digest[YVEX_SOURCE_PAYLOAD_DIGEST_CAP];
    char digest_algorithm[24];
    char digest_authority[40];
    yvex_source_payload_file_identity admitted_identity;
} yvex_source_payload_owned_shard;
struct yvex_source_payload_session {
    pthread_mutex_t mutex;
    int mutex_initialized, root_fd;
    yvex_source_payload_state state;
    int cancelled;
    yvex_source_payload_budget budget;
    yvex_source_payload_ops ops;
    yvex_source_tensor_snapshot *snapshot;
    yvex_source_verification verification;
    char target_id[128];
    char family_key[64];
    char repository_id[256];
    char *manifest_path;
    yvex_source_payload_owned_shard *shards;
    yvex_shard_index_entry *shard_index_entries;
    yvex_shard_index shard_index;
    yvex_source_payload_range *ranges;
    yvex_source_payload_handle *handles;
    yvex_source_payload_buffer *buffers;
    unsigned long long shard_count, tensor_count, logical_tensor_bytes, use_tick, active_plans;
    size_t inflight_host_bytes;
    yvex_source_payload_session_facts facts;
};
struct yvex_source_payload_plan {
    yvex_source_payload_session *session;
    yvex_source_payload_range *ranges;
    yvex_source_payload_chunk *chunks;
    yvex_source_payload_plan_summary summary;
    int registered;
};
void yvex_source_payload_default_ops(yvex_source_payload_ops *ops);
int yvex_source_payload_session_open_with_ops(
    yvex_source_payload_session **out,
    const yvex_source_payload_open_options *options,
    const yvex_source_payload_ops *ops,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_handle_acquire(
    yvex_source_payload_session *session,
    unsigned long long shard_index,
    int *fd,
    yvex_source_payload_failure *failure,
    yvex_error *err);
void yvex_source_payload_handle_release(
    yvex_source_payload_session *session,
    unsigned long long shard_index);
int yvex_source_payload_buffer_acquire(
    yvex_source_payload_session *session,
    size_t bytes,
    unsigned char **buffer,
    unsigned int *slot,
    yvex_source_payload_failure *failure,
    yvex_error *err);
void yvex_source_payload_buffer_release(
    yvex_source_payload_session *session,
    unsigned int slot);
int yvex_source_payload_exact_read(
    yvex_source_payload_session *session,
    unsigned long long shard_index,
    int fd,
    unsigned long long offset,
    unsigned char *buffer,
    size_t length,
    yvex_source_payload_stream_result *result,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_identity_compute(
    yvex_source_payload_session *session,
    yvex_source_payload_failure *failure,
    yvex_error *err);
int yvex_source_payload_manifest_publish(
    const yvex_source_payload_session *session,
    yvex_error *err);
void yvex_source_payload_fail(
    yvex_source_payload_failure *failure,
    yvex_source_payload_failure_code code,
    unsigned long long shard_index,
    unsigned long long tensor_index,
    unsigned long long requested,
    unsigned long long delivered,
    int system_error,
    yvex_error *err,
    int status,
    const char *where,
    const char *message);
int yvex_source_payload_refuse(yvex_source_payload_failure *failure,
                               yvex_source_payload_failure_code code,
                               yvex_error *err,
                               int status,
                               const char *where,
                               const char *message);
int yvex_source_payload_refuse_at(yvex_source_payload_failure *failure,
                                  yvex_source_payload_failure_code code,
                                  unsigned long long shard,
                                  unsigned long long tensor,
                                  unsigned long long requested,
                                  unsigned long long delivered,
                                  int system_error,
                                  yvex_error *err,
                                  int status,
                                  const char *where,
                                  const char *message);

/* Payload reports. */
typedef struct {
    const char *family_key;
    const char *display_family;
    const char *report_name;
    const char *target_id;
    const char *target_class;
    const char *model;
    const char *source_target_status;
    const char *source_family_profile_status;
    const char *source_artifact_class;
    const char *source_artifact_format;
    const char *source_artifact_origin;
    const char *source_artifact_authority;
    const char *source_tensor_container;
    const char *target_artifact_class;
    const char *target_artifact_origin;
    const char *target_artifact_required;
    const char *external_reference_status;
    const char *yvex_produced_artifact_status;
    const char *pressure_purpose;
    const char *runtime_shape;
    const char *hardware_lane;
    const char *backend_lane;
    const char *source_class;
    const char *source_path_blocker;
    const char *source_manifest_blocker;
    const char *native_inventory_blocker;
    const char *source_config_blocker;
    const char *tokenizer_blocker;
    const char *model_class_blocker;
    const char *model_class_next;
    const char *const *tail_blockers;
    unsigned long tail_blocker_count;
} yvex_source_family_profile;
typedef struct {
    const char *family;
    const char *release;
    const char *models_root;
    const char *source;
    const char *target;
    const yvex_source_family_profile *profile;
    char resolved_target[128];
    char resolved_model[128];
    int include_files, include_config, include_blockers, include_next, include_tensors, strict;
    unsigned long long tensor_limit;
} yvex_source_report_request;
#define YVEX_SOURCE_TENSOR_SAMPLE_CAP 20u
#define YVEX_SOURCE_TENSOR_SHAPE_CAP 128u
typedef struct {
    char name[192];
    char file[192];
    char dtype[24];
    char shape[YVEX_SOURCE_TENSOR_SHAPE_CAP];
    unsigned long long rank, elements, declared_bytes;
} yvex_source_tensor_sample;
typedef struct {
    const char *verification_status;
    const char *repository_status;
    const char *revision_status;
    const char *config_identity_status;
    const char *tokenizer_verification_status;
    const char *generation_config_status;
    const char *shard_index_status;
    const char *inventory_authority;
    const char *upstream_index_identity_status;
    int tokenizer_verified;
    const char *model_name;
    const char *config_presence;
    const char *generation_config_presence;
    const char *tokenizer_json_presence;
    const char *tokenizer_config_presence;
    const char *tokenizer_status;
    const char *safetensors_status;
    const char *manifest_status;
    const char *native_inventory_report_status;
    const char *tensor_map_report_status;
    const char *tensor_role_map_report_status;
    const char *output_head_map_report_status;
    const char *tokenizer_map_report_status;
    const char *native_inventory_status;
    const char *native_inventory_source;
    const char *tensor_metadata_status;
    const char *tensor_metadata_source;
    const char *native_tensor_metadata_status;
    const char *native_tensor_payload_status;
    const char *sidecar_status;
    const char *tensor_payload_status;
    const char *target_artifact_status;
    const char *footprint_class;
    const char *footprint_status;
    const char *provenance_origin_normal;
    const char *provenance_origin_audit;
    const char *provenance_status;
    const char *identity_status;
    const char *authority;
    const char *authority_status;
    const char *manifest_provenance_status;
    const char *manifest_authority;
    const char *manifest_schema_status;
    const char *manifest_family_status;
    const char *manifest_target_status;
    const char *manifest_artifact_class_status;
    const char *manifest_footprint_status;
    const char *manifest_native_inventory_status;
    const char *manifest_tensor_metadata_status;
    const char *manifest_consistency_status;
    const char *manifest_hardening_status;
    int manifest_creation_performed;
} yvex_source_report_semantics;
typedef struct {
    const yvex_source_family_profile *profile;
    yvex_source_report_request request;
    const char *status;
    const char *source_state;
    const char *top_blocker;
    const char *next_row;
    char identity_target_id[128];
    char identity_model[128];
    char identity_family[32];
    char identity_repo_id[256];
    char identity_revision[128];
    char identity_local_source_dir[YVEX_PATH_CAP];
    char download_registry_path[YVEX_PATH_CAP];
    char download_report_path[YVEX_PATH_CAP];
    char tensor_map_path[YVEX_PATH_CAP];
    char output_head_map_path[YVEX_PATH_CAP];
    char tokenizer_map_path[YVEX_PATH_CAP];
    int download_registry_exists, download_report_exists, tensor_map_exists, output_head_map_exists;
    int tokenizer_map_exists, tensor_map_incomplete, output_head_map_missing, source_identity_from_path;
    int source_identity_from_download_sidecar;
    char source_path[YVEX_PATH_CAP];
    char source_path_source[64];
    char manifest_path[YVEX_PATH_CAP];
    char native_inventory_path[YVEX_PATH_CAP];
    int source_exists, config_exists, generation_config_exists, tokenizer_json_exists;
    int tokenizer_config_exists, readme_exists, license_exists, manifest_exists, manifest_probe_checked;
    int manifest_probe_error, manifest_has_schema, manifest_schema_matches, manifest_has_family;
    int manifest_family_matches, manifest_has_target, manifest_target_matches, manifest_has_artifact_class;
    int manifest_has_footprint, manifest_has_provenance, manifest_has_native_inventory;
    int manifest_has_tensor_metadata;
    char manifest_schema_version[64];
    int native_inventory_exists;
    unsigned long long source_file_count, source_regular_file_count, safetensors_count, bin_count, dat_count;
    unsigned long long json_count, tokenizer_file_count, config_file_count, total_size_bytes;
    unsigned long long safetensors_size_bytes, sidecar_size_bytes, other_size_bytes;
    unsigned long long largest_source_file_bytes;
    char largest_source_file_name[YVEX_PATH_CAP];
    unsigned long long native_safetensors_count, native_safetensors_opened;
    unsigned long long native_safetensors_header_read_count, native_safetensors_header_error_count;
    unsigned long long native_safetensors_header_bytes, native_tensor_count, native_declared_data_bytes;
    unsigned long long native_declared_tensor_bytes, native_max_rank, native_max_tensor_elements;
    char native_largest_tensor_name[YVEX_PATH_CAP];
    unsigned long long native_largest_tensor_bytes, native_dtype_f16_count, native_dtype_bf16_count;
    unsigned long long native_dtype_f32_count, native_dtype_i8_count, native_dtype_i16_count;
    unsigned long long native_dtype_i32_count, native_dtype_i64_count, native_dtype_u8_count;
    unsigned long long native_dtype_other_count, native_invalid_file_count, native_inventory_error_count;
    unsigned long long source_tensor_count, source_tensor_name_count, source_tensor_file_count;
    unsigned long long source_tensor_dtype_count, source_tensor_rank_count, source_tensor_shape_count;
    unsigned long long source_tensor_declared_data_bytes, source_tensor_declared_tensor_bytes;
    unsigned long long source_tensor_total_elements, source_tensor_max_rank, source_tensor_max_elements;
    char source_tensor_largest_name[YVEX_PATH_CAP];
    char source_tensor_largest_file[YVEX_PATH_CAP];
    char source_tensor_largest_dtype[24];
    unsigned long long source_tensor_largest_rank;
    char source_tensor_largest_shape[YVEX_SOURCE_TENSOR_SHAPE_CAP];
    unsigned long long source_tensor_largest_elements, source_tensor_largest_declared_bytes;
    unsigned long long source_tensor_dtype_f16_count, source_tensor_dtype_bf16_count;
    unsigned long long source_tensor_dtype_f32_count, source_tensor_dtype_i8_count;
    unsigned long long source_tensor_dtype_i16_count, source_tensor_dtype_i32_count;
    unsigned long long source_tensor_dtype_i64_count, source_tensor_dtype_u8_count;
    unsigned long long source_tensor_dtype_other_count, source_tensor_rank_0_count;
    unsigned long long source_tensor_rank_1_count, source_tensor_rank_2_count, source_tensor_rank_3_count;
    unsigned long long source_tensor_rank_4_count, source_tensor_rank_other_count;
    unsigned long long source_tensor_metadata_error_count, source_tensor_name_embed_count;
    unsigned long long source_tensor_name_attn_count, source_tensor_name_mlp_count;
    unsigned long long source_tensor_name_norm_count, source_tensor_name_lm_head_count;
    unsigned long long source_tensor_name_other_count;
    yvex_source_tensor_sample source_tensor_samples[YVEX_SOURCE_TENSOR_SAMPLE_CAP];
    unsigned long long source_tensor_sample_count;
    const char *blockers[32];
    unsigned long blocker_count;
    yvex_source_verification verification;
    yvex_source_report_semantics semantics;
    int exit_code;
} yvex_source_report;
const yvex_source_family_profile *yvex_source_report_find_profile(const char *family);
int yvex_source_report_build(const yvex_source_report_request *request,
                             yvex_source_report *report,
                             yvex_error *err);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_SOURCE_PAYLOAD_H_INCLUDED */
