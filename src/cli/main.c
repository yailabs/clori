/*
 * Make the canonical operator registry the sole executable command-path authority.
 *
 * One descriptor selects one lane; runtime commands never fall through to engine adapters.
 * Generated immutable operator metadata over typed runtime-client and offline adapters.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <operator/registry.h>
#include "src/cli/input/private.h"
#include "src/cli/io/private.h"
#include "src/cli/private.h"

#include <yvex/core.h>

typedef int (*offline_handler)(int argc, char **argv);

static const yvex_operator_descriptor *descriptor_find(int argc, char **argv,
                                                        size_t *consumed)
{
    const yvex_operator_descriptor *best = NULL;
    size_t best_count = 0u, index, word;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *candidate = &yvex_operator_descriptors[index];
        if (!candidate->cli_projection || !candidate->command_word_count ||
            candidate->command_word_count > (size_t)(argc - 1))
            continue;
        for (word = 0u; word < candidate->command_word_count; ++word)
            if (strcmp(candidate->command_words[word], argv[word + 1u])) break;
        if (word == candidate->command_word_count && word > best_count) {
            best = candidate;
            best_count = word;
        }
    }
    for (index = 0u; index < yvex_operator_alias_count; ++index) {
        const yvex_operator_alias_descriptor *alias = &yvex_operator_aliases[index];
        if (!alias->word_count && argc != 1)
            continue;
        if (alias->word_count > (size_t)(argc - 1) || alias->word_count < best_count)
            continue;
        for (word = 0u; word < alias->word_count; ++word) {
            if (strcmp(alias->words[word], argv[word + 1u]))
                break;
        }
        if (word == alias->word_count &&
            (word > best_count || (word == 0u && !best))) {
            best = &yvex_operator_descriptors[alias->operation_index];
            best_count = word;
        }
    }
    *consumed = best_count;
    return best;
}

static int command_text(int argc, char **argv, char output[256])
{
    size_t count = 0u;
    int index;
    output[0] = '\0';
    for (index = 1; index < argc && argv[index][0] != '-'; ++index) {
        size_t extent = strlen(argv[index]);
        if (count && count + 1u >= 256u) return 0;
        if (count) output[count++] = ' ';
        if (extent >= 256u - count) return 0;
        memcpy(output + count, argv[index], extent);
        count += extent;
        output[count] = '\0';
    }
    return count != 0u;
}

static size_t command_distance(const char *left, const char *right)
{
    size_t prior[256], next[256], left_count = strlen(left), right_count = strlen(right);
    size_t row, column;
    if (left_count >= 256u || right_count >= 256u) return SIZE_MAX;
    for (column = 0u; column <= right_count; ++column) prior[column] = column;
    for (row = 1u; row <= left_count; ++row) {
        next[0] = row;
        for (column = 1u; column <= right_count; ++column) {
            size_t insertion = next[column - 1u] + 1u;
            size_t deletion = prior[column] + 1u;
            size_t replacement = prior[column - 1u] +
                                 (left[row - 1u] != right[column - 1u]);
            next[column] = insertion < deletion ? insertion : deletion;
            if (replacement < next[column]) next[column] = replacement;
        }
        memcpy(prior, next, (right_count + 1u) * sizeof(prior[0]));
    }
    return prior[right_count];
}

static const char *nearest_command(int argc, char **argv, char input[256])
{
    const char *best = NULL;
    size_t best_distance = SIZE_MAX, index;
    int tied = 0;
    if (!command_text(argc, argv, input)) return NULL;
    for (index = 0u; index < yvex_operator_descriptor_count; ++index) {
        const yvex_operator_descriptor *candidate = &yvex_operator_descriptors[index];
        size_t distance;
        if (!candidate->cli_projection || candidate->visibility == YVEX_OPERATOR_VISIBILITY_REMOVED)
            continue;
        distance = command_distance(input, candidate->command_path);
        if (distance < best_distance) {
            best_distance = distance;
            best = candidate->command_path;
            tied = 0;
        } else if (distance == best_distance) {
            tied = 1;
        }
    }
    if (tied || best_distance > 3u) return NULL;
    return best;
}

static int removed_path_matches(const char *path, int argc, char **argv)
{
    const char *cursor = path;
    int index = 1;
    while (*cursor) {
        const char *end = strchr(cursor, ' ');
        size_t count = end ? (size_t)(end - cursor) : strlen(cursor);
        if (index >= argc || strlen(argv[index]) != count ||
            memcmp(argv[index], cursor, count))
            return 0;
        index++;
        if (!end) return 1;
        cursor = end + 1;
    }
    return 0;
}

static int removed_path_refusal(int argc, char **argv)
{
    size_t index;
    for (index = 0u; index < yvex_operator_removed_path_count; ++index) {
        if (!removed_path_matches(yvex_operator_removed_paths[index].path, argc, argv))
            continue;
        yvex_cli_out_writef(stderr, "yvex: removed command: %s\nhint: %s\n",
                            yvex_operator_removed_paths[index].path,
                            yvex_operator_removed_paths[index].hint);
        return 1;
    }
    return 0;
}

static int offline_invoke(yvex_operator_offline_adapter adapter, int argc, char **argv,
                          yvex_runtime_cleanup_lease **cleanup)
{
    switch (adapter) {
    case YVEX_OPERATOR_OFFLINE_ACCOUNTS: return yvex_accounts_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_BACKEND: return yvex_backend_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_CONTEXT: return yvex_context_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_CONVERT: return yvex_convert_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_CUDA: return yvex_cuda_info_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_DETOKENIZE: return yvex_detokenize_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_FULLMODEL: return yvex_fullmodel_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_GGUF_EMIT: return yvex_gguf_emit_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_GGUF_TEMPLATE: return yvex_gguf_template_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_GRAPH: return yvex_graph_command(argc, argv, cleanup);
    case YVEX_OPERATOR_OFFLINE_IMATRIX: return yvex_imatrix_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_INPUT: return yvex_input_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_INSPECT: return yvex_inspect_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_INTEGRITY: return yvex_integrity_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MATERIALIZE: return yvex_materialize_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MATERIALIZE_GATE:
        return yvex_materialize_gate_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_METADATA: return yvex_metadata_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MODEL_GATE: return yvex_model_gate_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MODEL_TARGET: return yvex_model_target_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MODELS: return yvex_models_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_MOE: return yvex_moe_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_NATIVE_WEIGHTS: return yvex_native_weights_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_PATHS: return yvex_paths_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_PROMPT: return yvex_prompt_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_QTYPE_SUPPORT: return yvex_qtype_support_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_QUANT: return yvex_quant_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_QUANT_JOB: return yvex_quant_job_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_QUANT_POLICY: return yvex_quant_policy_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_SOURCE_MANIFEST: return yvex_source_manifest_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_TENSOR_COLLECTION: return yvex_tensor_collection_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_TENSOR_MAP: return yvex_tensor_map_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_TENSORS: return yvex_tensors_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_TOKENIZE: return yvex_tokenize_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_TOKENIZER: return yvex_tokenizer_command(argc, argv);
    case YVEX_OPERATOR_OFFLINE_COUNT: break;
    }
    yvex_cli_out_writef(stderr, "yvex: unbound offline adapter\n");
    return 2;
}

static char **offline_argv(const yvex_operator_descriptor *operation, int argc,
                           char **argv, size_t consumed, int *adapted_count)
{
    size_t remaining = (size_t)argc - consumed - 1u;
    size_t capacity = 2u + operation->adapter_argc + remaining;
    char **adapted = calloc(capacity, sizeof(*adapted));
    size_t index, output = 0u;
    if (!adapted) return NULL;
    adapted[output++] = argv[0];
    for (index = 0u; index < operation->adapter_argc; ++index)
        adapted[output++] = (char *)operation->adapter_argv[index];
    for (index = consumed + 1u; index < (size_t)argc; ++index)
        adapted[output++] = argv[index];
    adapted[output] = NULL;
    *adapted_count = (int)output;
    return adapted;
}

static int offline_dispatch(const yvex_operator_descriptor *operation, int argc,
                            char **argv, size_t consumed)
{
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_error cleanup_error;
    char **adapted;
    int adapted_count = 0, status, close_status;
    adapted = offline_argv(operation, argc, argv, consumed, &adapted_count);
    if (!adapted) {
        yvex_cli_out_writef(stderr, "yvex: argument allocation failed\n");
        return 1;
    }
    status = offline_invoke(operation->offline_adapter, adapted_count, adapted, &cleanup);
    free(adapted);
    yvex_error_clear(&cleanup_error);
    close_status = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_status != YVEX_OK) {
        yvex_cli_out_writef(stderr, "yvex: cleanup failed: %s\n",
                            yvex_error_message(&cleanup_error));
        return status ? status : 1;
    }
    return status;
}

int main(int argc, char **argv)
{
    const yvex_operator_descriptor *operation;
    const char *nearest;
    yvex_cli_operator_invocation invocation;
    char input[256];
    size_t consumed = 0u;
    int status;
    if (removed_path_refusal(argc, argv))
        return 2;
    operation = descriptor_find(argc, argv, &consumed);
    if (!operation) {
        if (argc > 2 && (!strcmp(argv[argc - 1], "--help") ||
                         !strcmp(argv[argc - 1], "-h")))
            return yvex_client_render_help_path((size_t)argc - 2u,
                                                (const char *const *)&argv[1], 0, 0);
        nearest = nearest_command(argc, argv, input);
        yvex_cli_out_writef(stderr, "yvex: unknown command: %s\n", argc > 1 ? input : "");
        if (nearest)
            yvex_cli_out_writef(stderr, "hint: did you mean `yvex %s`?\n", nearest);
        else
            yvex_cli_out_writef(stderr, "hint: use `yvex help`\n");
        return 2;
    }
    status = yvex_cli_operator_argv_parse(operation, argc, argv, consumed,
                                          &invocation);
    if (status) {
        yvex_cli_out_writef(stderr, "yvex: %s: %s\n", operation->command_path,
                            invocation.message);
        yvex_client_render_usage_error(operation);
        return status;
    }
    if (invocation.help_requested)
        return yvex_client_render_help_path(operation->command_word_count,
                                            operation->command_words, 0, 0);
    if (operation->lane == YVEX_OPERATOR_LANE_RUNTIME_CLIENT)
        return yvex_client_dispatch(operation, argc, argv, consumed);
    if (operation->lane == YVEX_OPERATOR_LANE_OFFLINE_ENGINE)
        return offline_dispatch(operation, argc, argv, consumed);
    yvex_cli_out_writef(stderr, "yvex: operation is not available on the CLI: %s\n",
                        operation->operation_id);
    return 2;
}
