/* Source metadata and inventories remain bound to their admitted snapshot. Metadata verification
 * does not authenticate payload bytes or imply transformation, artifact, or runtime support. */
#ifndef YVEX_SOURCE_H
#define YVEX_SOURCE_H

#include <stddef.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provider accounts. */
#define YVEX_ACCOUNT_PROVIDER_CAP 32
#define YVEX_ACCOUNT_AUTH_CAP 32
#define YVEX_ACCOUNT_HINT_CAP 128
#define YVEX_ACCOUNT_SOURCE_CAP 64
#define YVEX_ACCOUNT_STATUS_CAP 64
#define YVEX_ACCOUNT_BLOCKER_CAP 128
#define YVEX_ACCOUNT_NEXT_CAP 160
#define YVEX_ACCOUNT_TOKEN_ENV_CAP 64
#define YVEX_ACCOUNT_LOGIN_METHOD_CAP 64
#define YVEX_ACCOUNT_ARG_CAP 192

typedef enum {
    YVEX_ACCOUNT_PROVIDER_UNKNOWN = 0,
    YVEX_ACCOUNT_PROVIDER_HUGGINGFACE,
    YVEX_ACCOUNT_PROVIDER_GITHUB
} yvex_account_provider;

typedef enum {
    YVEX_ACCOUNT_INTERACTIVE_AUTO = 0,
    YVEX_ACCOUNT_INTERACTIVE_ALWAYS,
    YVEX_ACCOUNT_INTERACTIVE_NEVER
} yvex_account_interactive_mode;

typedef struct {
    yvex_account_provider provider;
    const char *cli_override;
    const char *token_env_name;
} yvex_account_observe_options;

typedef struct {
    yvex_account_provider provider;
    const char *cli_override;
    const char *token_env_name;
    yvex_account_interactive_mode interactive;
    int required;
} yvex_account_ensure_options;

typedef struct {
    yvex_account_provider provider;
    char provider_name[YVEX_ACCOUNT_PROVIDER_CAP];
    char cli_path[YVEX_PATH_CAP];
    char cli_source[YVEX_ACCOUNT_SOURCE_CAP];
    char cli_status[YVEX_ACCOUNT_STATUS_CAP];
    char auth_state[YVEX_ACCOUNT_AUTH_CAP];
    char account_hint[YVEX_ACCOUNT_HINT_CAP];
    char credential_source[YVEX_ACCOUNT_SOURCE_CAP];
    char token_env_name[YVEX_ACCOUNT_TOKEN_ENV_CAP];
    char login_method[YVEX_ACCOUNT_LOGIN_METHOD_CAP];
    char status[YVEX_ACCOUNT_STATUS_CAP];
    char top_blocker[YVEX_ACCOUNT_BLOCKER_CAP];
    char next[YVEX_ACCOUNT_NEXT_CAP];
    char state_path[YVEX_PATH_CAP];
    char last_checked_at[32];
    int cli_present;
    int token_env_present;
    int token_value_redacted;
    int raw_token_stored_by_yvex;
    int command_exit_code;
} yvex_account_observation;

typedef struct {
    const char *args[YVEX_ACCOUNT_ARG_CAP];
    const char *stdout_path;
    const char *stderr_path;
} yvex_account_command_options;

/* Bounded provider output remains caller-owned and is never persisted by this API. */
typedef struct {
    const char *args[YVEX_ACCOUNT_ARG_CAP];
    char *stdout_bytes;
    size_t stdout_capacity;
    char *stderr_bytes;
    size_t stderr_capacity;
    size_t stdout_count;
    size_t stderr_count;
    int stdout_truncated;
    int stderr_truncated;
    int exit_code;
} yvex_account_capture_options;

int yvex_account_provider_from_name(const char *name, yvex_account_provider *out);
const char *yvex_account_provider_name(yvex_account_provider provider);
const char *yvex_account_default_token_env(yvex_account_provider provider);
int yvex_account_observe(const yvex_account_observe_options *options,
                         yvex_account_observation *out,
                         yvex_error *err);
int yvex_account_ensure(const yvex_account_ensure_options *options,
                        yvex_account_observation *out,
                        yvex_error *err);
int yvex_account_write_state(const yvex_account_observation *observations,
                             unsigned long count,
                             yvex_error *err);
int yvex_accounts_run_provider_command(const yvex_account_command_options *options,
                                       yvex_error *err);
int yvex_accounts_capture_provider_command(yvex_account_capture_options *options,
                                           yvex_error *err);

/* Source manifests. */
typedef enum {
    YVEX_SOURCE_STATUS_UNKNOWN = 0,
    YVEX_SOURCE_STATUS_IN_PROGRESS,
    YVEX_SOURCE_STATUS_INCOMPLETE,
    YVEX_SOURCE_STATUS_COMPLETE,
    YVEX_SOURCE_STATUS_FAILED
} yvex_source_status;

typedef struct {
    const char *repo;
    const char *revision;
    const char *license;
    const char *model_card;
    const char *local_path;
    const char *node_name;
    const char *dry_run_log;
    const char *download_log;
    const char *pid_file;
    const char *download_command;
    yvex_source_status status;
    int include_files;
} yvex_source_manifest_options;

typedef struct {
    unsigned long long file_count;
    unsigned long long safetensors_count;
    unsigned long long total_size_bytes;
    int has_config;
    int has_tokenizer;
    int has_safetensors;
} yvex_source_manifest_summary;

const char *yvex_source_status_name(yvex_source_status status);

int yvex_source_manifest_write_json(const char *out_path,
                                    const yvex_source_manifest_options *options,
                                    yvex_source_manifest_summary *summary_out,
                                    yvex_error *err);

int yvex_source_manifest_scan_local(const char *local_path,
                                    yvex_source_manifest_summary *out,
                                    yvex_error *err);

/* Native tensor inventory. */
#define YVEX_NATIVE_WEIGHT_MAX_DIMS 8

typedef struct yvex_native_weight_table yvex_native_weight_table;

typedef enum {
    YVEX_NATIVE_DTYPE_UNKNOWN = 0,
    YVEX_NATIVE_DTYPE_F64,
    YVEX_NATIVE_DTYPE_F32,
    YVEX_NATIVE_DTYPE_F16,
    YVEX_NATIVE_DTYPE_BF16,
    YVEX_NATIVE_DTYPE_I64,
    YVEX_NATIVE_DTYPE_I32,
    YVEX_NATIVE_DTYPE_I16,
    YVEX_NATIVE_DTYPE_I8,
    YVEX_NATIVE_DTYPE_U8,
    YVEX_NATIVE_DTYPE_BOOL,
    YVEX_NATIVE_DTYPE_F8_E4M3,
    YVEX_NATIVE_DTYPE_F8_E5M2,
    YVEX_NATIVE_DTYPE_F8_E8M0,
    YVEX_NATIVE_DTYPE_FP4,
    YVEX_NATIVE_DTYPE_OTHER
} yvex_native_dtype;

typedef struct {
    const char *name;
    const char *shard_path;
    yvex_native_dtype dtype;
    const char *dtype_name;
    unsigned int rank;
    unsigned long long dims[YVEX_NATIVE_WEIGHT_MAX_DIMS];
    unsigned long long data_start;
    unsigned long long data_end;
    unsigned long long data_bytes;
} yvex_native_weight_info;

typedef struct {
    unsigned long long shard_count;
    unsigned long long tensor_count;
    unsigned long long total_tensor_bytes;
    unsigned long long unknown_dtype_count;
    unsigned long long malformed_shard_count;
} yvex_native_weight_summary;

typedef struct {
    const char *source_dir;
    int recursive;
    int include_metadata;
} yvex_native_weight_options;

int yvex_native_weight_table_open(yvex_native_weight_table **out,
                                  const yvex_native_weight_options *options,
                                  yvex_error *err);

void yvex_native_weight_table_close(yvex_native_weight_table *table);

unsigned long long yvex_native_weight_table_count(const yvex_native_weight_table *table);

const yvex_native_weight_info *yvex_native_weight_table_at(const yvex_native_weight_table *table,
                                                           unsigned long long index);

const yvex_native_weight_info *yvex_native_weight_table_find(const yvex_native_weight_table *table,
                                                             const char *name);

int yvex_native_weight_table_summary(const yvex_native_weight_table *table,
                                     yvex_native_weight_summary *out,
                                     yvex_error *err);

const char *yvex_native_dtype_name(yvex_native_dtype dtype);

/* Bounded payload sessions expose their budgets and typed failure facts to
 * cross-subsystem planners without exposing session internals. */
typedef enum {
    YVEX_SOURCE_PAYLOAD_FAILURE_NONE = 0,
    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_ARGUMENT,
    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_STATE,
    YVEX_SOURCE_PAYLOAD_FAILURE_METADATA_NOT_VERIFIED,
    YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_NOT_TRUSTED,
    YVEX_SOURCE_PAYLOAD_FAILURE_MANIFEST_VERSION_UNSUPPORTED,
    YVEX_SOURCE_PAYLOAD_FAILURE_SOURCE_IDENTITY_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_MAPPING_IDENTITY_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_SHARD,
    YVEX_SOURCE_PAYLOAD_FAILURE_DUPLICATE_TENSOR,
    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_NOT_INDEXED,
    YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_NOT_INDEXED,
    YVEX_SOURCE_PAYLOAD_FAILURE_PATH_ESCAPE,
    YVEX_SOURCE_PAYLOAD_FAILURE_SYMLINK_REFUSED,
    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_OPEN,
    YVEX_SOURCE_PAYLOAD_FAILURE_NON_REGULAR_SHARD,
    YVEX_SOURCE_PAYLOAD_FAILURE_STAT,
    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_SIZE_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT,
    YVEX_SOURCE_PAYLOAD_FAILURE_EXPECTED_DIGEST_UNAVAILABLE,
    YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_ALGORITHM_UNSUPPORTED,
    YVEX_SOURCE_PAYLOAD_FAILURE_DIGEST_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OVERFLOW,
    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_DATA_REGION,
    YVEX_SOURCE_PAYLOAD_FAILURE_RANGE_OUTSIDE_FILE,
    YVEX_SOURCE_PAYLOAD_FAILURE_TENSOR_LENGTH_MISMATCH,
    YVEX_SOURCE_PAYLOAD_FAILURE_INVALID_CHUNK,
    YVEX_SOURCE_PAYLOAD_FAILURE_RESOURCE_BUDGET,
    YVEX_SOURCE_PAYLOAD_FAILURE_ALLOCATION,
    YVEX_SOURCE_PAYLOAD_FAILURE_HANDLE_CACHE_EXHAUSTED,
    YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ,
    YVEX_SOURCE_PAYLOAD_FAILURE_IO,
    YVEX_SOURCE_PAYLOAD_FAILURE_CONSUMER,
    YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED,
    YVEX_SOURCE_PAYLOAD_FAILURE_CLOSE_BUSY,
    YVEX_SOURCE_PAYLOAD_FAILURE_CLEANUP
} yvex_source_payload_failure_code;
typedef struct {
    yvex_source_payload_failure_code code;
    unsigned long long shard_index, tensor_index, requested_bytes, delivered_bytes;
    int system_error;
} yvex_source_payload_failure;
typedef struct {
    unsigned long long maximum_shards, maximum_tensors, maximum_plan_chunks;
    unsigned int maximum_open_handles, maximum_streams;
    size_t chunk_bytes, page_bytes, maximum_inflight_host_bytes;
    int allow_local_snapshot_seal;
} yvex_source_payload_budget;

#ifdef __cplusplus
}
#endif

#endif /* YVEX_SOURCE_H */
