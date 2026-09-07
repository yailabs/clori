/* This is the only CLI-local boundary allowed to write operator streams directly. */
#ifndef SRC_CLI_IO_PRIVATE_H_INCLUDED
#define SRC_CLI_IO_PRIVATE_H_INCLUDED

#include <stdarg.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/catalog.h>
#include <yvex/content.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/internal/cli_table.h>
#include <yvex/model.h>
#include <yvex/registry.h>
#include <yvex/server.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>

struct yvex_operator_descriptor;
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YVEX_SOURCE_RENDER_NORMAL = 0,
    YVEX_SOURCE_RENDER_TABLE,
    YVEX_SOURCE_RENDER_AUDIT,
    YVEX_SOURCE_RENDER_JSON
} yvex_source_render_mode;
typedef enum {
    YVEX_MODELS_OUTPUT_NORMAL = 0,
    YVEX_MODELS_OUTPUT_TABLE,
    YVEX_MODELS_OUTPUT_AUDIT,
    YVEX_MODELS_OUTPUT_JSON
} yvex_models_output_mode;
typedef struct {
    char alias[YVEX_SERVER_MODEL_ALIAS_CAP];
    unsigned long long generation;
} yvex_cli_engine_binding;
typedef struct {
    char model_selector[YVEX_MODEL_LIBRARY_ID_CAP];
    char model_name[YVEX_MODEL_LIBRARY_NAME_CAP];
    char family[YVEX_REMOTE_FAMILY_CAP];
    char profile_alias[YVEX_MODEL_LIBRARY_NAME_CAP];
    char variant[YVEX_REMOTE_PRECISION_CAP + 24u];
    char artifact_identity[YVEX_MODEL_ARTIFACT_ID_CAP];
    char format[YVEX_REMOTE_FORMAT_CAP];
    char quant_precision[YVEX_REMOTE_PRECISION_CAP];
    char backend[32];
    char engine_kind[32];
    char execution_strategy[32];
    char runtime_target[YVEX_MODEL_LIBRARY_NAME_CAP];
    unsigned long long representation_bytes;
    unsigned long long context_capacity;
} yvex_cli_model_profile_selection;
typedef struct {
    yvex_cli_model_profile_selection model;
    yvex_cli_engine_binding engine;
    unsigned long long session_count;
} yvex_cli_loaded_model_fact;
typedef enum {
    YVEX_MODEL_CATALOG_OUTPUT_TABLE = 0,
    YVEX_MODEL_CATALOG_OUTPUT_AUDIT,
    YVEX_MODEL_CATALOG_OUTPUT_JSON
} yvex_model_catalog_output_mode;
typedef struct {
    const char *query;
    const char *author;
    const char *filter;
    const char *provider;
    const char *models_root;
    unsigned int page;
    unsigned int page_size;
    yvex_model_catalog_output_mode output_mode;
} yvex_cli_model_search_options;
typedef struct {
    const char *repository;
    const char *revision;
    const char *provider;
    const char *models_root;
    yvex_model_catalog_output_mode output_mode;
} yvex_cli_model_inspect_options;
typedef struct {
    const char *models_root;
    const char *registry_path;
    yvex_model_catalog_output_mode output_mode;
    int all_representations;
    int wide;
} yvex_cli_model_list_options;
int yvex_cli_model_profile_select(const char *model, const char *variant,
                                  int text_only,
                                  yvex_cli_model_profile_selection *selection);
int yvex_cli_model_profile_resolve_alias(
    const char *alias, yvex_cli_model_profile_selection *selection);
int yvex_cli_model_load_command(int argc, char **argv, size_t consumed);
int yvex_cli_model_unload_command(int argc, char **argv, size_t consumed);
int yvex_cli_model_loaded_select(const char *model, const char *variant,
                                 int text_only,
                                 yvex_cli_engine_binding *binding,
                                 yvex_cli_model_profile_selection *selection);
int yvex_cli_loaded_models_snapshot(yvex_cli_loaded_model_fact *models,
                                    unsigned long long capacity,
                                    unsigned long long *count,
                                    yvex_error *err);
typedef enum {
    YVEX_CLI_FIELD_TEXT = 0,
    YVEX_CLI_FIELD_TEXT_ARRAY,
    YVEX_CLI_FIELD_U64,
    YVEX_CLI_FIELD_U32,
    YVEX_CLI_FIELD_I32,
    YVEX_CLI_FIELD_BOOL,
    YVEX_CLI_FIELD_DOUBLE,
    YVEX_CLI_FIELD_FLOAT9,
    YVEX_CLI_FIELD_HEX64
} yvex_cli_field_kind;
typedef struct {
    const char *key;
    yvex_cli_field_kind kind;
    size_t offset;
    const char *fallback;
} yvex_cli_field_spec;
typedef struct {
    const char *reset;
    const char *strong;
    const char *accent;
    const char *dim;
    const char *success;
    const char *warning;
    const char *error;
} yvex_cli_terminal_style;
typedef enum {
    YVEX_CLI_STREAM_STYLE_NORMAL = 0,
    YVEX_CLI_STREAM_STYLE_DIM,
    YVEX_CLI_STREAM_STYLE_ACCENT,
    YVEX_CLI_STREAM_STYLE_STRONG
} yvex_cli_stream_style;
#define YVEX_CLI_STREAM_LINE_CAP 16384u
typedef struct {
    FILE *output;
    yvex_cli_terminal_style style;
    yvex_client_stream_channel channel;
    yvex_cli_stream_style active_style, line_style;
    unsigned char line[YVEX_CLI_STREAM_LINE_CAP], inline_previous;
    size_t line_count;
    unsigned int column, prose_width, line_indent, inline_flags;
    int enhanced, in_fence, pending_cr, channel_announced, line_started;
    int wrote_bytes, last_newline, pending_space;
} yvex_cli_stream_renderer;
typedef struct {
    yvex_cli_terminal_style style;
    char session_id[YVEX_SERVER_ID_CAP];
    char request_id[YVEX_SERVER_ID_CAP];
    unsigned long long cycles, proposed, accepted, rejected;
    unsigned long long discarded, progress_tokens;
    double progress_seconds;
    int request_open, detailed, progress_reasoning;
} yvex_cli_watch_renderer;
typedef enum {
    YVEX_MODELS_OPTION_TEXT = 0,
    YVEX_MODELS_OPTION_FLAG,
    YVEX_MODELS_OPTION_OUTPUT
} yvex_models_option_kind;

typedef struct {
    const char *flag;
    yvex_models_option_kind kind;
    size_t offset;
} yvex_models_option_spec;
#define YVEX_MODEL_DOWNLOAD_PATTERN_CAP 32u
#define YVEX_MODELS_ARTIFACT_ROWS_CAP 256u
enum { YVEX_MODEL_DOWNLOAD_INTERRUPT_TIMEOUT_SECONDS = 5 };
typedef enum {
    YVEX_MODEL_DOWNLOAD_AUTH_AUTO = 0,
    YVEX_MODEL_DOWNLOAD_AUTH_REQUIRED,
    YVEX_MODEL_DOWNLOAD_AUTH_NEVER
} yvex_model_download_auth_mode;
typedef enum {
    YVEX_MODEL_DOWNLOAD_PROGRESS_AUTO = 0,
    YVEX_MODEL_DOWNLOAD_PROGRESS_LIVE,
    YVEX_MODEL_DOWNLOAD_PROGRESS_PLAIN,
    YVEX_MODEL_DOWNLOAD_PROGRESS_LOG,
    YVEX_MODEL_DOWNLOAD_PROGRESS_OFF
} yvex_model_download_progress_mode;
typedef struct yvex_cli_models_download_options {
    const char *target;
    const char *repo;
    const char *family;
    const char *name;
    const char *revision;
    const char *source;
    const char *provider;
    const char *asset;
    const char *asset_name;
    const char *release;
    const char *github_source;
    const char *models_root;
    const char *token_env;
    const char *cli;
    const char *include_patterns[YVEX_MODEL_DOWNLOAD_PATTERN_CAP];
    const char *exclude_patterns[YVEX_MODEL_DOWNLOAD_PATTERN_CAP];
    unsigned int include_count, exclude_count;
    int selection_restored;
    unsigned long long max_workers;
    yvex_model_download_auth_mode auth_mode;
    int dry_run;
    int no_manifest;
    int no_native_inventory;
    int force_sidecars;
    int yes;
    int resume;
    int clear_stale_locks;
    int force;
    int match_provider_process;
    int cleanup_stale_locks;
    int cleanup_logs;
    int cleanup_receipts;
    int cleanup_failed_partials;
    int cleanup_all_provider_cache;
    yvex_models_output_mode output_mode;
    yvex_model_download_progress_mode progress_mode;
    unsigned long long tick_seconds;
    unsigned long long timeout_seconds;
} yvex_cli_models_download_options;
typedef struct yvex_model_download_source_scan {
    unsigned long long file_count;
    unsigned long long safetensors_count;
    unsigned long long gguf_count;
    unsigned long long total_regular_file_bytes;
    unsigned long long largest_file_bytes;
    unsigned long long partial_file_count;
    unsigned long long cache_file_count;
    unsigned long long lock_count;
    unsigned long long lock_age_seconds[YVEX_MODEL_DOWNLOAD_PATTERN_CAP];
    int config_present;
    int tokenizer_present;
    char largest_file_name[YVEX_PATH_CAP];
    char lock_paths[YVEX_MODEL_DOWNLOAD_PATTERN_CAP][YVEX_PATH_CAP];
} yvex_model_download_source_scan;
typedef struct yvex_model_download_report {
    char status[64];
    char target_id[128];
    char family[32];
    char provider[32];
    char repo_id[256];
    char revision[128];
    char local_name[128];
    char local_source_dir[YVEX_PATH_CAP];
    char models_root[YVEX_PATH_CAP];
    char models_root_source[32];
    char reports_dir[YVEX_PATH_CAP];
    char registry_dir[YVEX_PATH_CAP];
    char manifest_path[YVEX_PATH_CAP];
    char native_inventory_path[YVEX_PATH_CAP];
    char download_report_path[YVEX_PATH_CAP];
    char registry_path[YVEX_PATH_CAP];
    char receipt_path[YVEX_PATH_CAP];
    char active_receipt_path[YVEX_PATH_CAP];
    char last_receipt_path[YVEX_PATH_CAP];
    char stdout_log_path[YVEX_PATH_CAP];
    char stderr_log_path[YVEX_PATH_CAP];
    char hf_cli_path[YVEX_PATH_CAP];
    char hf_cli_source[32];
    char provider_cli_path[YVEX_PATH_CAP];
    char provider_cli_source[64];
    char provider_cli_status[32];
    char auth_state[32];
    char credential_source[64];
    char account_hint[128];
    char accounts_state_path[YVEX_PATH_CAP];
    char token_env_name[64];
    char source_payload_digest[65], representation_format[YVEX_REMOTE_FORMAT_CAP];
    char representation_precision[YVEX_REMOTE_PRECISION_CAP];
    char created_at[32];
    char top_blocker[128];
    char error[256];
    char stage_resolve_target[16];
    char stage_resolve_paths[16];
    char stage_prepare_dirs[16];
    char stage_account_provider[16];
    char stage_provider_cli[16];
    char stage_hf_cli[16];
    char stage_download[16];
    char stage_progress_stream[16];
    char stage_progress_ticks[16];
    char stage_source_scan[16];
    char stage_source_manifest[16];
    char stage_native_inventory[16];
    char stage_sidecar[16];
    yvex_model_download_source_scan source_scan;
    yvex_native_weight_summary native_summary;
    int hf_exit_code;
    int provider_exit_code;
    int stdout_streamed;
    int stderr_streamed;
    int interrupted;
    int interrupt_signal;
    int signal_forwarded;
    int child_terminated;
    int child_killed_after_timeout;
    int orphan_check_performed;
    int partial_source_preserved;
    int lock_files_deleted;
    int upstream_identity_verified, remote_lookup_performed, payload_hash_verified;
    pid_t provider_pid;
    pid_t provider_process_group;
    char child_exit_status[32];
    char orphan_check_status[16];
    unsigned long long stdout_bytes;
    unsigned long long stderr_bytes;
    unsigned long long tick_count;
    unsigned long long tick_last_elapsed_seconds;
    unsigned long long tick_last_file_count;
    unsigned long long tick_last_safetensors_count;
    unsigned long long tick_last_gguf_count;
    unsigned long long tick_last_partial_file_count;
    unsigned long long tick_last_cache_file_count;
    unsigned long long tick_last_total_regular_file_bytes;
    unsigned long long tick_last_largest_file_bytes;
    char tick_last_largest_file_name[YVEX_PATH_CAP];
    int source_manifest_written;
    int native_inventory_written;
    int report_written;
    int registry_written;
} yvex_model_download_report;
typedef struct yvex_model_download_safetensors_check {
    int checked;
    int ok_count;
    int truncated_count;
    int invalid_count;
    char status[32];
} yvex_model_download_safetensors_check;
typedef struct yvex_model_download_resolved_target {
    int found;
    char includes[YVEX_MODEL_DOWNLOAD_PATTERN_CAP][1024], excludes[YVEX_MODEL_DOWNLOAD_PATTERN_CAP][1024];
    char source_payload_digest[65];
    unsigned int include_count, exclude_count;
    char target_id[128];
    char family[32];
    char provider[32];
    char repo_id[256];
    char revision[128];
    char local_name[128];
    char local_source_dir[YVEX_PATH_CAP];
    char registry_path[YVEX_PATH_CAP];
    char download_report_path[YVEX_PATH_CAP];
    char manifest_path[YVEX_PATH_CAP];
    char native_inventory_path[YVEX_PATH_CAP];
} yvex_model_download_resolved_target;
typedef struct yvex_model_download_process_match {
    unsigned int count;
    pid_t first_pid;
    pid_t first_pgid;
} yvex_model_download_process_match;
typedef enum {
    YVEX_FULLMODEL_COMMAND_REPORT = 0,
    YVEX_FULLMODEL_COMMAND_MATERIALIZATION_PLAN,
    YVEX_FULLMODEL_COMMAND_MATERIALIZE,
    YVEX_FULLMODEL_COMMAND_DESCRIPTOR,
    YVEX_FULLMODEL_COMMAND_FAMILY_RUNTIME
} yvex_fullmodel_command_kind;
typedef struct yvex_cli_fullmodel_options {
    const char *model;
    const char *backend;
    const char *target;
    const char *registry_path;
    const char *residency;
    const char *require_role;
    const char *require_collection;
    const char *fail_after_phase;
    const char *report_dir;
    const char *format;
    const char *family;
    unsigned long long limit_tensors;
    unsigned long long limit_bytes;
    int has_limit_bytes;
    int dry_run;
    int plan_only;
    int include_blockers;
    int include_roles;
    int include_placement;
    int include_graph;
    int include_kv;
    int include_logits;
    int include_moe;
    int include_output;
    yvex_models_output_mode output_mode;
    yvex_fullmodel_command_kind command;
} yvex_cli_fullmodel_options;
typedef struct fullmodel_materialize_report {
    const yvex_cli_fullmodel_options *options;
    const char *status;
    const char *model_resolved_path;
    const char *target_id;
    const char *target_class;
    const char *artifact_identity_status;
    const char *tensor_inventory_status;
    const char *required_role_coverage;
    const char *missing_required_roles;
    const char *unsupported_required_roles;
    const char *placement_plan_status;
    const char *memory_budget_status;
    const char *backend_preflight_status;
    const char *materialization_mode;
    const char *full_model_materialization;
    const char *full_model_materialization_proof;
    const char *phase;
    const char *failed_phase;
    const char *failed_reason;
    const char *cleanup_attempted;
    const char *cleanup_status;
    const char *cleanup_idempotent;
    const char *owned_state_released;
    const char *partial_materialization;
    const char *residency_plan;
    const char *runtime_blockers;
    unsigned long long materialized_tensor_count;
    unsigned long long materialized_tensor_bytes;
    unsigned long long refused_tensor_count;
    unsigned long long skipped_tensor_count;
    unsigned long long required_tensor_count;
    unsigned long long required_tensor_bytes;
    unsigned long long peak_planned_bytes;
    unsigned long long cpu_resident_bytes;
    unsigned long long cuda_resident_bytes;
} fullmodel_materialize_report;
typedef struct yvex_fullmodel_collections {
    unsigned long long embedding;
    unsigned long long embedding_bytes;
    unsigned long long normalization;
    unsigned long long normalization_bytes;
    unsigned long long attention;
    unsigned long long attention_bytes;
    unsigned long long mlp;
    unsigned long long mlp_bytes;
    unsigned long long moe;
    unsigned long long moe_bytes;
    unsigned long long output;
    unsigned long long output_bytes;
    unsigned long long tokenizer;
    unsigned long long tokenizer_bytes;
    unsigned long long unknown;
    unsigned long long unknown_bytes;
    int has_token_embedding;
    int has_attention_norm;
    int has_post_attention_norm;
    int has_attention_q;
    int has_attention_k;
    int has_attention_v;
    int has_attention_out;
    int has_ffn_gate;
    int has_ffn_up;
    int has_ffn_down;
    int has_moe_router;
    int has_moe_expert;
    int has_output_norm;
    int has_output_head;
    int has_tokenizer_metadata;
} yvex_fullmodel_collections;
typedef struct yvex_fullmodel_backend_fit {
    int available;
    int memory_known;
    unsigned long long total_bytes;
    unsigned long long available_bytes;
    unsigned long long required_bytes;
    const char *fit_status;
    char fit_reason[160];
} yvex_fullmodel_backend_fit;

typedef struct fullmodel_largest_tensor {
    const yvex_tensor_info *tensor;
    unsigned long long bytes;
} fullmodel_largest_tensor;

typedef struct yvex_models_prepare_source_report {
    char target_id[128];
    char family[32];
    char provider[32];
    char repo_id[256];
    char revision[128];
    char models_root[YVEX_PATH_CAP];
    char source_path[YVEX_PATH_CAP];
    char source_manifest_path[YVEX_PATH_CAP];
    char native_inventory_path[YVEX_PATH_CAP];
    char tensor_map_path[YVEX_PATH_CAP];
    char output_head_map_path[YVEX_PATH_CAP];
    char tokenizer_map_path[YVEX_PATH_CAP];
    char expected_artifact_path[YVEX_PATH_CAP];
    char download_registry_path[YVEX_PATH_CAP];
    char download_report_path[YVEX_PATH_CAP];
    const char *source_status;
    const char *model_class_status;
    const char *tensor_map_status;
    const char *output_head_map_status;
    const char *tokenizer_map_status;
    const char *artifact_status;
    const char *artifact_plan_status;
    const char *artifact_emission_status;
    const char *artifact_identity_status;
    unsigned int blocker_count;
    const char *top_blocker;
    const char *reason;
    const char *next;
    const char *final_status;
    int downloaded_target_resolved;
} yvex_models_prepare_source_report;

/* Shared CLI adaptation helpers used across command-domain subtrees. */
int print_yvex_error(const yvex_error *err, int exit_code);
int exit_for_status(int status);
int parse_positive_ull(const char *text, unsigned long long *out);
int parse_ull_allow_zero(const char *text, unsigned long long *out);
int parse_uint_allow_zero(const char *text, unsigned int *out);
void print_quoted_bytes(const char *data, unsigned long long len);
int open_artifact_for_gguf(const char *path, yvex_artifact **artifact, yvex_error *err);
void print_tensor_dims(const unsigned long long *dims, unsigned int rank);
void print_native_dims(const unsigned long long *dims, unsigned int rank);
void print_token_ids(const yvex_tokens *tokens);
int parse_id_list(const char *text, unsigned int **out_ids, unsigned long long *out_len);
int parse_dims_csv(const char *text, unsigned int rank, unsigned long long dims[4]);
void print_metadata_drift_cli(const yvex_model_metadata_drift_report *report);
int models_registry_open(yvex_model_registry **registry, const char *registry_path,
                         int create_if_missing, yvex_error *err);
int yvex_operator_paths_resolve_target(const yvex_operator_paths *operator_paths,
                                       const char *family, const char *kind, char *out, size_t cap,
                                       int *out_exists, yvex_error *err);
int yvex_quant_command_execute(int arg_count, char **args, int render_result);
typedef struct yvex_cli_content_stage yvex_cli_content_stage;
int yvex_cli_content_stage_open(yvex_cli_content_stage **, yvex_error *);
void yvex_cli_content_stage_close(yvex_cli_content_stage **);
void yvex_cli_content_stage_clear(yvex_cli_content_stage *);
unsigned long long yvex_cli_content_stage_count(const yvex_cli_content_stage *);
const yvex_content_part *yvex_cli_content_stage_parts(const yvex_cli_content_stage *);
int yvex_cli_content_stage_attach(
    yvex_cli_content_stage *, const char *, yvex_content_part *, yvex_error *);
int yvex_cli_content_stage_turn(
    const yvex_cli_content_stage *, const unsigned char *, unsigned long long,
    yvex_content_part[YVEX_CONTENT_MAX_PARTS], unsigned long long *, yvex_error *);

/* JSON output. */
void yvex_cli_out_json_string(FILE *fp, const char *text);
void yvex_cli_json_begin(FILE *fp);
void yvex_cli_json_end(FILE *fp);
void yvex_cli_json_field_str(FILE *fp, const char *key, const char *value, int comma);
void yvex_cli_json_field_u64(FILE *fp, const char *key, unsigned long long value, int comma);
void yvex_cli_json_field_bool(FILE *fp, const char *key, int value, int comma);
int yvex_cli_json_fields(FILE *fp, const void *object, const yvex_cli_field_spec *fields,
                         size_t field_count, int comma);

/* Human output. */
int yvex_cli_out_writef(FILE *fp, const char *fmt, ...);
int yvex_cli_out_vwritef(FILE *fp, const char *fmt, va_list ap);
int yvex_cli_completion_command(int argc, char **argv, size_t consumed);
void yvex_cli_client_request_init(yvex_client_request *request, yvex_client_operation operation);
int yvex_cli_client_request_open(yvex_client **client, const yvex_client_request *request, yvex_error *err);
int yvex_client_render_help_path(size_t path_count, const char *const *path,
                                 int advanced, int json);
void yvex_client_render_usage_error(
    const struct yvex_operator_descriptor *operation);
int yvex_cli_out_puts(FILE *fp, const char *text);
int yvex_cli_out_fputs(const char *text, FILE *fp);
int yvex_cli_out_char(FILE *fp, int ch);
int yvex_cli_out_flush(FILE *fp);
FILE *yvex_cli_out_stdout(void);
FILE *yvex_cli_out_stderr(void);
void yvex_cli_terminal_style_get(FILE *fp, yvex_cli_terminal_style *style);
void yvex_cli_stream_renderer_open(yvex_cli_stream_renderer *renderer,
                                   FILE *output, int enhanced);
int yvex_cli_stream_renderer_write(yvex_cli_stream_renderer *renderer,
                                   yvex_client_stream_channel channel,
                                   const unsigned char *bytes,
                                   unsigned long long count);
int yvex_cli_stream_renderer_finish(yvex_cli_stream_renderer *renderer,
                                    int separate_terminal_status);
const char *yvex_cli_out_stop_reason(unsigned long long reason);
void yvex_cli_out_turn_metrics(
    FILE *, const yvex_client_message *, unsigned long long,
    const yvex_cli_terminal_style *);
void yvex_cli_out_turn_complete(FILE *, const yvex_client_message *, unsigned long long,
    const yvex_cli_terminal_style *);
int yvex_cli_out_server_event(const yvex_server_event *event, int detailed);
void yvex_cli_watch_renderer_open(yvex_cli_watch_renderer *renderer, int detailed);
int yvex_cli_watch_renderer_event(yvex_cli_watch_renderer *renderer,
    const yvex_server_event *event, const yvex_server_summary *live);
void yvex_cli_watch_renderer_finish(yvex_cli_watch_renderer *renderer);
void yvex_cli_out_repl_catalog(void);
void yvex_cli_out_line(FILE *fp, const char *text);
void yvex_cli_out_lines(FILE *fp, const char *const *lines, size_t line_count);
void yvex_cli_out_kv_str(FILE *fp, const char *key, const char *value);
void yvex_cli_out_kv_bool(FILE *fp, const char *key, int value);
int yvex_cli_out_fields(FILE *fp, const void *object, const yvex_cli_field_spec *fields,
                        size_t field_count);

#ifdef __cplusplus
}
#endif

#endif /* SRC_CLI_IO_PRIVATE_H_INCLUDED */
