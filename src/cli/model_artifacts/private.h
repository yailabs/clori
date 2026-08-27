/*
 * Share the minimal types and calls that genuinely cross the model-artifact CLI translation-unit
 * boundary.
 *
 * One private contract serves this subtree; no declaration enters libyvex.a or the installed
 * public ABI. Typed CLI reports render domain facts without promoting support.
 */
#ifndef YVEX_CLI_MODEL_ARTIFACTS_PRIVATE_H
#define YVEX_CLI_MODEL_ARTIFACTS_PRIVATE_H

#include "src/cli/io/private.h"

#include <stddef.h>
#include <sys/types.h>
#include <time.h>

#include <yvex/artifact.h>
#include <yvex/backend.h>
#include <yvex/catalog.h>
#include <yvex/gguf.h>
#include <yvex/core.h>
#include <yvex/internal/core.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/source.h>
#include <yvex/registry.h>
#include <yvex/source.h>
#include <yvex/tokenizer.h>

typedef const char *(*yvex_cli_engine_state_resolver)(
    const yvex_local_package_record *package, const void *context);

int yvex_context_command(int arg_count, char **args);
void yvex_context_help(FILE *fp);
int yvex_fullmodel_command(int arg_count, char **args);
void yvex_fullmodel_help(FILE *fp);
int yvex_models_command(int arg_count, char **args);
void yvex_models_help(FILE *fp);
int model_search_options_parse(int arg_count,
                               char **args,
                               int start,
                               yvex_cli_model_search_options *options);
int model_remote_inspect_options_parse(int arg_count,
                                       char **args,
                                       int start,
                                       yvex_cli_model_inspect_options *options);
int model_local_list_options_parse(int arg_count,
                                   char **args,
                                   int start,
                                   yvex_cli_model_list_options *options);
int yvex_remote_catalog_render(FILE *fp,
                               const yvex_remote_catalog *catalog,
                               const yvex_local_catalog *local_catalog,
                               yvex_model_catalog_output_mode mode,
                               int representations);
int yvex_local_catalog_render(FILE *fp,
                              const yvex_local_catalog *catalog,
                              yvex_cli_engine_state_resolver engine_state,
                              const void *engine_context,
                              int engine_host_observed,
                              yvex_model_catalog_output_mode mode);
int yvex_moe_command(int arg_count, char **args);
void yvex_moe_help(FILE *fp);
int yvex_tensor_collection_command(int arg_count, char **args);
void yvex_tensor_collection_help(FILE *fp);

typedef struct {
    const char *status, *target, *component, *backend, *output_path;
    yvex_component_execution_result execution;
    unsigned long long published_bytes;
    int published;
} yvex_cli_component_result;
int yvex_component_render(FILE *fp, yvex_graph_report_mode mode,
                          const yvex_cli_component_result *result);

int parse_models_download_options_from(int arg_count,
                                       char **args,
                                       int start_index,
                                       yvex_cli_models_download_options *options);
int model_artifacts_fullmodel_options_parse(int arg_count,
                                            char **args,
                                            yvex_cli_fullmodel_options *options);

/* Cross-unit model-artifact presentation and transactional output contract. */
void model_prepare_source_report_render(
    const yvex_models_prepare_source_report *report,
    yvex_models_output_mode mode);
int print_fullmodel_source_only_report(const char *target, const char *backend);
int model_download_write_json_sidecar(const char *path,
                                      const char *schema,
                                      const yvex_cli_models_download_options *options,
                                      const yvex_model_download_report *report,
                                      yvex_error *err);
void fullmodel_print_materialize_report(const fullmodel_materialize_report *report);
int fullmodel_print_family_runtime_report(const yvex_cli_fullmodel_options *options,
                                          yvex_model_ref *ref,
                                          yvex_model_context *ctx,
                                          const char *target_id,
                                          const char *target_class,
                                          unsigned long long artifact_bytes,
                                          yvex_arch arch,
                                          unsigned long long tensor_count,
                                          unsigned long long total_tensor_bytes,
                                          const yvex_fullmodel_collections *collections,
                                          const char *role_coverage,
                                          const char *missing_roles,
                                          const char *unsupported_roles,
                                          int selected_target);
int print_fullmodel_source_only_materialize(const yvex_cli_fullmodel_options *options,
                                            const char *target);
void print_fullmodel_common_boundaries(void);
void model_download_print_status_report(
    const yvex_cli_models_download_options *options,
    const yvex_model_download_report *report,
    const yvex_model_download_process_match *match,
    const yvex_model_download_safetensors_check *safe_check,
    int active_receipt_present,
    int active_receipt_stale,
    int last_receipt_present,
    const char *last_receipt_status,
    pid_t provider_pid,
    pid_t provider_pgid);
int print_fullmodel_missing_report(const yvex_cli_fullmodel_options *options,
                                   const char *resolved_path);
int print_fullmodel_parse_failure_report(const yvex_cli_fullmodel_options *options,
                                         const yvex_model_ref *ref,
                                         const char *reason,
                                         int rc);
int model_download_write_control_receipt(
    const char *path,
    const yvex_cli_models_download_options *options,
    const yvex_model_download_report *report,
    const char *status,
    yvex_error *err);
int print_fullmodel_source_only_plan(const yvex_cli_fullmodel_options *options,
                                     const char *target);
void fullmodel_print_descriptor_normal(const yvex_cli_fullmodel_options *options,
                                       const char *target_id,
                                       const char *target_class,
                                       const char *role_coverage,
                                       const char *missing_roles);
void fullmodel_print_materialization_plan(
    const yvex_cli_fullmodel_options *options,
    const yvex_model_ref *ref,
    const char *target_id,
    const char *target_class,
    unsigned long long artifact_bytes,
    yvex_arch arch,
    unsigned long long tensor_count,
    unsigned long long total_tensor_bytes,
    const yvex_fullmodel_collections *collections,
    const char *dtype_summary,
    const char *role_coverage,
    const char *missing_roles,
    int selected_target);
void fullmodel_print_family_runtime_normal(const yvex_cli_fullmodel_options *options,
                                           const char *target_id,
                                           const char *target_class,
                                           const char *role_coverage,
                                           const char *missing_roles);
unsigned int fullmodel_print_blocker(unsigned int index,
                                     const char *category,
                                     const char *severity,
                                     const char *message,
                                     int blocks_full_materialization,
                                     int blocks_generation);
void fullmodel_print_descriptor_report(const yvex_cli_fullmodel_options *options,
                                       yvex_model_ref *ref,
                                       yvex_model_context *ctx,
                                       const char *target_id,
                                       const char *target_class,
                                       unsigned long long artifact_bytes,
                                       yvex_arch arch,
                                       unsigned long long tensor_count,
                                       unsigned long long total_tensor_bytes,
                                       const yvex_fullmodel_collections *collections,
                                       const char *role_coverage,
                                       const char *missing_roles,
                                       const char *unsupported_roles,
                                       int selected_target);
void fullmodel_print_descriptor_phases(const char *role_status,
                                       const char *collection_status,
                                       const char *failure_phase);
int model_download_finalize_control_receipt(
    const yvex_cli_models_download_options *options,
    const yvex_model_download_report *report,
    const char *status);
int model_download_write_native_inventory_json(const char *path,
                                               const char *source_dir,
                                               const yvex_native_weight_table *table,
                                               yvex_error *err);
void fullmodel_print_plan_normal(const yvex_cli_fullmodel_options *options,
                                 const char *status,
                                 const char *target_class,
                                 const char *fit,
                                 const char *top_blocker);
void fullmodel_print_phase(unsigned int index,
                           const char *name,
                           const char *status,
                           unsigned long long tensor_count,
                           unsigned long long tensor_bytes,
                           const char *residency,
                           int required,
                           int blocked,
                           const char *blocker);
int print_fullmodel_source_only_family_runtime(const yvex_cli_fullmodel_options *options,
                                               const char *target);
int model_download_write_receipt(const char *path,
                                 const yvex_cli_models_download_options *options,
                                 const yvex_model_download_report *report,
                                 int token_present,
                                 yvex_error *err);
void fullmodel_print_report_normal(const yvex_cli_fullmodel_options *options,
                                   const char *status,
                                   const char *target_id,
                                   const char *target_class,
                                   const char *role_coverage,
                                   const char *top_blocker,
                                   const char *next);
void model_download_format_bytes(char *out, size_t cap, unsigned long long bytes);
void model_download_short_file_name(char *out, size_t cap, const char *path);
int print_fullmodel_source_only_descriptor(const yvex_cli_fullmodel_options *options,
                                           const char *target);
void fullmodel_print_largest(const fullmodel_largest_tensor *top,
                             unsigned int top_count);
int model_download_finish(const yvex_cli_models_download_options *options,
                          yvex_model_download_report *report);

int path_exists(const char *path);
int is_path_like_reference(const char *input);
void model_artifact_append_role(char *out, size_t out_cap, const char *role);

int parse_models_output_mode(const char *value, yvex_models_output_mode *mode);
int parse_models_bound_option(const char *command,
                              int arg_count,
                              char **args,
                              int *index,
                              void *options,
                              const yvex_models_option_spec *specs,
                              size_t spec_count,
                              int *handled);
void print_model_registry_entry_cli(const yvex_model_registry_entry *entry);
void print_model_registry_entry_audit(const yvex_model_registry_entry *entry);
void print_model_registry_scan_entry_cli(const yvex_model_registry_entry *entry);
void dims_to_text(const unsigned long long *dims,
                  unsigned int rank,
                  char *out,
                  size_t out_cap);
int populate_registry_identity(yvex_model_registry_entry *entry,
                               char *sha256,
                               char *format,
                               char *architecture,
                               char *primary_name,
                               char *primary_role,
                               char *primary_dtype,
                               char *primary_dims,
                               yvex_error *err);
void model_stage_print(const char *stage, const char *status);
const char *model_requested_family(const char *family);
void model_phase_print(const char *prefix,
                       unsigned int index,
                       const char *name,
                       const char *status,
                       const char *fallback);
void model_print_runtime_generation(const char *runtime_execution);
int cli_arg_value_valid(const char *value);
int parse_models_value_option(const char *command,
                              const char *flag,
                              int arg_count,
                              char **args,
                              int *index,
                              const char **value);
int model_backend_kind_from_name(const char *backend_name, yvex_backend_kind *kind);
int expand_operator_path(const char *input,
                         char *out,
                         size_t out_cap,
                         yvex_error *err,
                         const char *where);
int path_join2(char *out,
               size_t out_cap,
               const char *dir,
               const char *file,
               yvex_error *err,
               const char *where);
int path_parent_dir(const char *path, char *out, size_t out_cap);

int yvex_models_artifacts_surface_command(int arg_count, char **args);
int yvex_models_download_surface_command(int arg_count, char **args);
int yvex_models_prepare_surface_command(int arg_count, char **args);
int yvex_models_check_surface_command(int arg_count, char **args);
long long model_download_json_i64_field(const char *text, const char *key);
int model_download_identity_paths(const char *target,
                                  const char *family,
                                  const yvex_operator_paths *operator_paths,
                                  yvex_model_download_resolved_target *out,
                                  yvex_error *err);
int model_download_read_identity_file(const char *path,
                                      const char *fallback_target,
                                      const char *fallback_family,
                                      yvex_model_download_resolved_target *out);
int model_download_resolve_downloaded_target(const char *target,
                                             const yvex_operator_paths *operator_paths,
                                             yvex_model_download_resolved_target *out,
                                             yvex_error *err);
int model_download_family_valid(const char *family);
int model_download_local_name_valid(const char *name);
int model_download_repo_valid(const char *repo);
unsigned int model_download_effective_include_count(
    const yvex_cli_models_download_options *options);
unsigned int model_download_effective_exclude_count(
    const yvex_cli_models_download_options *options);
const char *model_download_effective_include_at(
    const yvex_cli_models_download_options *options, unsigned int index);
const char *model_download_effective_exclude_at(
    const yvex_cli_models_download_options *options, unsigned int index);
void model_download_report_init(yvex_model_download_report *report);
int model_download_source_path_allowed(const yvex_operator_paths *paths,
                                       const char *source_dir,
                                       yvex_model_download_report *report);
int model_download_name_starts_with(const char *name, const char *prefix);
int model_download_name_contains(const char *name, const char *needle);
int model_download_scan_source(const char *root,
                               yvex_model_download_source_scan *scan,
                               yvex_error *err);
int model_download_check_safetensors_source(
    const char *root,
    yvex_model_download_safetensors_check *check,
    yvex_error *err);
int model_download_run_hf(const yvex_cli_models_download_options *options,
                          yvex_model_download_report *report,
                          const char *token_value,
                          yvex_error *err);
int model_download_run_github(const yvex_cli_models_download_options *options,
                              const yvex_model_download_report *report,
                              yvex_error *err);
int model_download_pid_alive(pid_t pid);
int model_download_pgid_alive(pid_t pgid);
const char *model_download_progress_mode_name(yvex_model_download_progress_mode mode);
const char *model_download_signal_name(int signo);
yvex_model_download_progress_mode model_download_effective_progress_mode(
    yvex_model_download_progress_mode mode);
const char *model_download_auth_mode_name(yvex_model_download_auth_mode mode);
int fullmodel_string_is_empty(const char *text);
int fullmodel_file_size(const char *path, unsigned long long *out);
const char *fullmodel_family_from_arch(yvex_arch arch);
int fullmodel_has_attention_collection(const yvex_fullmodel_collections *collections);
int fullmodel_has_mlp_collection(const yvex_fullmodel_collections *collections);
int fullmodel_has_normalization_collection(const yvex_fullmodel_collections *collections);
void fullmodel_classify_tensor(const yvex_tensor_info *tensor,
                               yvex_fullmodel_collections *collections);
int fullmodel_is_selected_target(const char *text);
const yvex_tensor_info *fullmodel_descriptor_find_tensor(yvex_model_context *ctx,
                                                         const char *role);
const char *fullmodel_detect_family(const yvex_cli_fullmodel_options *options,
                                    yvex_arch arch,
                                    const char *target_id);
int fullmodel_family_request_matches(const char *requested, const char *detected);
const char *fullmodel_role_status_from_tensor(yvex_model_context *ctx,
                                              const yvex_fullmodel_collections *collections,
                                              const char *role);
const char *fullmodel_identity_status(const yvex_model_ref *ref,
                                      unsigned long long artifact_bytes);
void fullmodel_probe_backend_fit(const char *backend,
                                 unsigned long long required_bytes,
                                 yvex_fullmodel_backend_fit *fit);

void prepare_probe_map_sidecar_status(const char *tensor_map_path,
                                      const char *output_head_map_path,
                                      int *tensor_map_incomplete,
                                      int *output_head_map_missing);

#endif /* YVEX_CLI_MODEL_ARTIFACTS_PRIVATE_H */
