/* Owner: client.yvex_dev.
 * Owns: nested developer/plumbing grammar and dispatch into retained typed CLI adapters.
 * Does not own: product-client grammar, domain capability, runtime hosting, or compatibility aliases.
 * Invariants: every admitted command has a namespace and no retired flat public command is accepted.
 * Boundary: optional developer entrypoint over existing domain/report adapters.
 * Purpose: retain engineering reachability after the incompatible product-client cutover.
 * Inputs: one namespace, one action, and owner-specific remaining arguments.
 * Effects: dispatches exactly one existing typed adapter and closes any retained runtime lease.
 * Failure: unknown namespace/action returns parser status without fallback to the retired registry. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/cli/input/private.h"
#include "src/cli/io/private.h"

#include <yvex/core.h>

typedef int (*developer_handler)(int argc, char **argv);
typedef int (*developer_owned_handler)(
    int argc, char **argv, yvex_runtime_cleanup_lease **retained_cleanup);

typedef struct {
    const char *name_space;
    const char *action;
    const char *adapter_name;
    developer_handler handler;
    developer_owned_handler owned_handler;
} developer_route;

static const developer_route routes[] = {
    {"artifact", "show", "inspect", yvex_inspect_command, NULL},
    {"artifact", "verify", "integrity", yvex_integrity_command, NULL},
    {"artifact", "metadata", "metadata", yvex_metadata_command, NULL},
    {"artifact", "tensors", "tensors", yvex_tensors_command, NULL},
    {"artifact", "materialize", "materialize", yvex_materialize_command, NULL},
    {"artifact", "materialize-gate", "materialize-gate", yvex_materialize_gate_command, NULL},
    {"artifact", "model-gate", "model-gate", yvex_model_gate_command, NULL},
    {"artifact", "template", "gguf-template", yvex_gguf_template_command, NULL},
    {"artifact", "emit", "gguf-emit", yvex_gguf_emit_command, NULL},
    {"graph", NULL, "graph", NULL, yvex_graph_command},
    {"quant", "preset", "quant", yvex_quant_command, NULL},
    {"quant", "plan", "quant", yvex_quant_command, NULL},
    {"quant", "emit", "quant", yvex_quant_command, NULL},
    {"quant", "summarize", "quant", yvex_quant_command, NULL},
    {"quant", "explain", "quant", yvex_quant_command, NULL},
    {"quant", "policy", "quant-policy", yvex_quant_policy_command, NULL},
    {"quant", "imatrix", "imatrix", yvex_imatrix_command, NULL},
    {"quant", "job", "quant-job", yvex_quant_job_command, NULL},
    {"quant", "qtype", "qtype-support", yvex_qtype_support_command, NULL},
    {"quant", "convert", "convert", yvex_convert_command, NULL},
    {"runtime", "input", "input", yvex_input_command, NULL},
    {"runtime", "context", "context", yvex_context_command, NULL},
    {"tokenizer", "show", "tokenizer", yvex_tokenizer_command, NULL},
    {"tokenizer", "encode", "tokenize", yvex_tokenize_command, NULL},
    {"tokenizer", "decode", "detokenize", yvex_detokenize_command, NULL},
    {"tokenizer", "prompt", "prompt", yvex_prompt_command, NULL},
    {"source", "manifest", "source-manifest", yvex_source_manifest_command, NULL},
    {"source", "native", "native-weights", yvex_native_weights_command, NULL},
    {"tensor", "map", "tensor-map", yvex_tensor_map_command, NULL},
    {"tensor", "collection", "tensor-collection", yvex_tensor_collection_command, NULL},
    {"evidence", "target", "model-target", yvex_model_target_command, NULL},
    {"evidence", "model", "fullmodel", yvex_fullmodel_command, NULL},
    {"evidence", "moe", "moe", yvex_moe_command, NULL},
    {"evidence", "backend", "backend", yvex_backend_command, NULL},
    {"evidence", "cuda", "cuda-info", yvex_cuda_info_command, NULL},
    {"evidence", "accounts", "accounts", yvex_accounts_command, NULL},
    {"evidence", "paths", "paths", yvex_paths_command, NULL},
    {"evidence", "models", "models", yvex_models_command, NULL},
};

/* Purpose: render the complete bounded developer namespace without old flat catalog metadata. */
static void print_help(FILE *output)
{
    yvex_cli_out_writef(
        output,
        "YVEX developer tools\n\n"
        "  yvex-dev graph ...\n"
        "  yvex-dev artifact show|verify|metadata|tensors|materialize|emit ...\n"
        "  yvex-dev quant preset|plan|emit|summarize|explain|policy|imatrix ...\n"
        "  yvex-dev tokenizer show|encode|decode|prompt ...\n"
        "  yvex-dev source manifest|native ...\n"
        "  yvex-dev tensor map|collection ...\n"
        "  yvex-dev runtime input|context ...\n"
        "  yvex-dev evidence target|model|moe|backend|cuda ...\n"
        "  yvex-dev help | version\n");
}

/* Purpose: choose one exact nested route; graph retains its already nested graph grammar. */
static const developer_route *route_find(int argc, char **argv, int *skip)
{
    size_t index;
    if (argc < 2) return NULL;
    for (index = 0u; index < sizeof(routes) / sizeof(routes[0]); ++index) {
        if (strcmp(routes[index].name_space, argv[1]) != 0) continue;
        if (!routes[index].action) {
            *skip = 2;
            return &routes[index];
        }
        if (argc >= 3 && strcmp(routes[index].action, argv[2]) == 0) {
            *skip = 3;
            return &routes[index];
        }
    }
    return NULL;
}

/* Purpose: reconstruct one nested developer argv vector for a retained typed adapter.
 * Inputs: admitted route, source argv, and first remaining argument. Effects: allocates argv.
 * Failure: returns NULL on allocation failure. Boundary: never dispatches or restores flat aliases. */
static char **adapter_argv(const developer_route *route, int argc,
                           char **argv, int skip, int *adapter_argc)
{
    int source, output = 0;
    char **adapted = calloc((size_t)(argc - skip + 3), sizeof(*adapted));
    if (!adapted) return NULL;
    adapted[output++] = argv[0];
    adapted[output++] = (char *)route->adapter_name;
    if (!strcmp(route->name_space, "quant") &&
        !strcmp(route->adapter_name, "quant"))
        adapted[output++] = argv[2];
    for (source = skip; source < argc; ++source)
        adapted[output++] = argv[source];
    adapted[output] = NULL;
    *adapter_argc = output;
    return adapted;
}

/* Purpose: dispatch exactly one nested developer command and preserve cleanup as secondary evidence.
 * Inputs: process argv in the redesigned developer grammar. Effects: invokes one retained adapter.
 * Failure: returns stable parse, adapter, or cleanup status. Boundary: no product-client fallback. */
int main(int argc, char **argv)
{
    const developer_route *route;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_error cleanup_error;
    char **adapted;
    int skip = 0, adapted_count = 0, status, close_status;
    if (argc == 1 || !strcmp(argv[1], "help") || !strcmp(argv[1], "--help") ||
        !strcmp(argv[1], "-h")) {
        print_help(stdout);
        return 0;
    }
    if (!strcmp(argv[1], "version") || !strcmp(argv[1], "--version")) {
        yvex_cli_out_writef(stdout, "yvex-dev %s\n", yvex_version_string());
        return 0;
    }
    route = route_find(argc, argv, &skip);
    if (!route) {
        yvex_cli_out_writef(stderr, "yvex-dev: unknown developer command\n");
        yvex_cli_out_writef(stderr, "hint: use `yvex-dev help`\n");
        return 2;
    }
    adapted = adapter_argv(route, argc, argv, skip, &adapted_count);
    if (!adapted) {
        yvex_cli_out_writef(stderr, "yvex-dev: argument allocation failed\n");
        return 1;
    }
    status = route->owned_handler
                 ? route->owned_handler(adapted_count, adapted, &cleanup)
                 : route->handler(adapted_count, adapted);
    free(adapted);
    yvex_error_clear(&cleanup_error);
    close_status = yvex_runtime_cleanup_lease_close(&cleanup, &cleanup_error);
    if (close_status != YVEX_OK) {
        yvex_cli_out_writef(stderr, "yvex-dev: cleanup failed: %s\n",
                            yvex_error_message(&cleanup_error));
        return status ? status : 1;
    }
    return status;
}
