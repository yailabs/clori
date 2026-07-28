/* Owner: cli.input.private (cli.input).
 * Owns: typed CLI argument records and parsers.
 * Does not own: domain policy, rendering, or command execution.
 * Invariants: declarations have one owner, stable ordering, and no hidden capability promotion.
 * Boundary: CLI input adaptation.
 * Purpose: provide the canonical CLI input adaptation contract.
 * Inputs: typed immutable facts and explicitly owned mutable lifecycle objects.
 * Effects: only declared lifecycle, allocation, I/O, and publication operations mutate state.
 * Failure: typed refusals leave outputs defined and preserve caller-owned state. */
#ifndef SRC_CLI_INPUT_PRIVATE_H_INCLUDED
#define SRC_CLI_INPUT_PRIVATE_H_INCLUDED

#include "src/cli/io/private.h"

#include <stdio.h>
#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/model_artifact.h>
#include <yvex/internal/model_target.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/sampling.h>
#include <yvex/internal/transformer.h>
#include <yvex/internal/source_payload.h>
#include <yvex/model.h>
#include <yvex/registry.h>
#include <yvex/tokenizer.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared CLI command, parser, and diagnostic rendering contract. */
int cli_backend_name_valid(const char *name);
int enforce_registered_identity_cli(const yvex_model_ref *ref, const char *surface);
void print_token_input_summary(const yvex_token_input *input, const char *status,
                               const char *bounds_status, unsigned long long selected_index,
                               unsigned int selected_token, int has_selected);
int yvex_accounts_command(int arg_count, char **args);
void yvex_accounts_help(FILE *fp);
int yvex_backend_command(int arg_count, char **args);
void yvex_backend_help(FILE *fp);
int yvex_context_command(int arg_count, char **args);
void yvex_context_help(FILE *fp);
int yvex_convert_command(int arg_count, char **args);
void yvex_convert_help(FILE *fp);
int yvex_cuda_info_command(int arg_count, char **args);
void yvex_cuda_info_help(FILE *fp);
int yvex_detokenize_command(int arg_count, char **args);
void yvex_detokenize_help(FILE *fp);
int yvex_graph_command(int arg_count, char **args,
                       yvex_runtime_cleanup_lease **retained_cleanup);
void yvex_graph_help(FILE *fp);
int yvex_fullmodel_command(int arg_count, char **args);
void yvex_fullmodel_help(FILE *fp);
int yvex_gguf_template_command(int arg_count, char **args);
void yvex_gguf_template_help(FILE *fp);
int yvex_gguf_emit_command(int arg_count, char **args);
void yvex_gguf_emit_help(FILE *fp);
int yvex_imatrix_command(int arg_count, char **args);
void yvex_imatrix_help(FILE *fp);
int yvex_inspect_command(int arg_count, char **args);
void yvex_inspect_help(FILE *fp);
int yvex_input_command(int arg_count, char **args);
void yvex_input_help(FILE *fp);
int yvex_integrity_command(int arg_count, char **args);
void yvex_integrity_help(FILE *fp);
int yvex_materialize_command(int arg_count, char **args);
void yvex_materialize_help(FILE *fp);
int yvex_materialize_gate_command(int arg_count, char **args);
void yvex_materialize_gate_help(FILE *fp);
int yvex_metadata_command(int arg_count, char **args);
void yvex_metadata_help(FILE *fp);
int yvex_model_gate_command(int arg_count, char **args);
void yvex_model_gate_help(FILE *fp);
int yvex_model_target_command(int arg_count, char **args);
void yvex_model_target_help(FILE *fp);
int yvex_moe_command(int arg_count, char **args);
void yvex_moe_help(FILE *fp);
int yvex_models_command(int arg_count, char **args);
void yvex_models_help(FILE *fp);
int yvex_native_weights_command(int arg_count, char **args);
void yvex_native_weights_help(FILE *fp);
int yvex_paths_command(int arg_count, char **args);
void yvex_paths_help(FILE *fp);
int yvex_prompt_command(int arg_count, char **args);
void yvex_prompt_help(FILE *fp);
int yvex_quant_job_command(int arg_count, char **args);
void yvex_quant_job_help(FILE *fp);
int yvex_quant_policy_command(int arg_count, char **args);
void yvex_quant_policy_help(FILE *fp);
int yvex_qtype_support_command(int arg_count, char **args);
void yvex_qtype_support_help(FILE *fp);
int yvex_source_manifest_command(int arg_count, char **args);
void yvex_source_manifest_help(FILE *fp);
int yvex_source_manifest_report_command(int arg_count, char **args);
int yvex_tensor_collection_command(int arg_count, char **args);
void yvex_tensor_collection_help(FILE *fp);
int yvex_tensor_map_command(int arg_count, char **args);
void yvex_tensor_map_help(FILE *fp);
int yvex_tokenize_command(int arg_count, char **args);
void yvex_tokenize_help(FILE *fp);
int yvex_tokenizer_command(int arg_count, char **args);
void yvex_tokenizer_help(FILE *fp);
int yvex_tensors_command(int arg_count, char **args);
void yvex_tensors_help(FILE *fp);

/* Backend contract. */
typedef struct {
    yvex_backend_report_request request;
    int help;
} yvex_backend_args;
int yvex_backend_args_parse(int argc, char **argv, yvex_backend_args *out, yvex_error *err);
int yvex_cuda_info_args_parse(int argc, char **argv, yvex_backend_args *out, yvex_error *err);

/* Graph contract. */
typedef enum {
    YVEX_GRAPH_ATTENTION_ACTION_NONE = 0,
    YVEX_GRAPH_ATTENTION_ACTION_PREPARE,
    YVEX_GRAPH_ATTENTION_ACTION_DESCRIBE,
    YVEX_GRAPH_ATTENTION_ACTION_CAPABILITIES,
    YVEX_GRAPH_ATTENTION_ACTION_PLAN,
    YVEX_GRAPH_ATTENTION_ACTION_EXECUTE,
    YVEX_GRAPH_ATTENTION_ACTION_COMPARE,
    YVEX_GRAPH_ATTENTION_ACTION_STATE_INSPECT,
    YVEX_GRAPH_ATTENTION_ACTION_STATE_VALIDATE,
    YVEX_GRAPH_ATTENTION_ACTION_STATE_EXERCISE,
    YVEX_GRAPH_ATTENTION_ACTION_RESIDENCY_INSPECT,
    YVEX_GRAPH_ATTENTION_ACTION_CAPTURE,
    YVEX_GRAPH_ATTENTION_ACTION_REPLAY,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_LIST,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_INSPECT,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_WARMUP,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_UPDATE,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_INVALIDATE,
    YVEX_GRAPH_ATTENTION_ACTION_CUDA_GRAPH_RELEASE,
    YVEX_GRAPH_ATTENTION_ACTION_TRACE,
    YVEX_GRAPH_ATTENTION_ACTION_PROFILE,
    YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK,
    YVEX_GRAPH_ATTENTION_ACTION_BENCHMARK_COMPARE,
    YVEX_GRAPH_ATTENTION_ACTION_QUALIFY
} yvex_graph_attention_action;

typedef struct {
    yvex_graph_report_mode render_mode;
    struct {
        yvex_graph_attention_action action;
        const char *target, *artifact_path, *runtime_binding_path, *runtime_binding_dir;
        const char *models_root, *backend, *probe, *coverage, *phase, *mode;
        const char *operation_scope, *trace_level, *progress, *input_class, *input_file;
        const char *attention_class;
        const char *capture_bucket, *baseline_path, *current_path, *chart_path;
        unsigned long long token_count, chunk_tokens, context_capacity, warmup, repeat;
        unsigned long long layer, layer_start, layer_count, position, history_tokens;
        unsigned long long local_capacity, compressed_capacity, indexer_capacity;
        unsigned long long maximum_host_bytes, maximum_device_bytes;
        unsigned long long regression_basis_points;
        int layer_seen, layer_start_seen, layer_count_seen, position_seen, history_tokens_seen;
        int phase_seen, mode_seen, token_count_seen;
        int local_capacity_seen, compressed_capacity_seen, indexer_capacity_seen;
        int active;
        int compare_backends;
        int require_mode;
        int write_baseline;
    } attention;
    struct {
        const char *target, *artifact_path, *runtime_binding_path, *backend;
        const char *input_class, *input_file, *coverage, *progress;
        unsigned long long maximum_host_bytes, maximum_device_bytes;
        int active;
    } moe;
    struct {
        const char *target, *artifact_path, *runtime_binding_path, *backend;
        const char *phase, *input_class, *input_file, *progress;
        const char *strategy, *system, *user, *text;
        unsigned long long chunk_tokens, context_capacity;
        unsigned long long prefill_tokens, prefill_chunk_tokens;
        unsigned long long maximum_new_tokens, maximum_output_bytes;
        unsigned long long maximum_host_bytes, maximum_device_bytes;
        unsigned long long top_k, seed;
        double temperature, top_p, min_p, typical_p;
        int active, decode, logits, sample, generate, seed_seen;
    } transformer;
    int help_requested;
    int help_exit_code;
} yvex_graph_args;
int yvex_graph_args_parse(int argc, char **argv, yvex_graph_args *out, yvex_error *err);

/* Model Target contract. */
typedef struct {
    yvex_model_target_request request;
    int help_requested;
    int parse_failed;
    char error_message[256];
} yvex_model_target_args;
int yvex_model_target_args_parse(int argc, char **argv, yvex_model_target_args *out,
                                 yvex_error *err);

/* Source contract. */
typedef struct {
    const char *family;
    const char *release;
    const char *models_root;
    const char *source;
    const char *target;
    int include_files;
    int include_config;
    int include_blockers;
    int include_next;
    int include_tensors;
    int strict;
    unsigned long long tensor_limit;
    yvex_source_render_mode render_mode;
    int help;
} yvex_source_args;
int yvex_source_args_parse(int argc, char **argv, yvex_source_args *out, yvex_error *err);
void yvex_source_report_request_from_parsed(yvex_source_report_request *request,
                                            const yvex_source_args *args);

#ifdef __cplusplus
}
#endif

#endif /* SRC_CLI_INPUT_PRIVATE_H_INCLUDED */
