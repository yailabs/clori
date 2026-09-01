/* Project porcelain model residency over the exact native engine protocol. */
#define _POSIX_C_SOURCE 200809L
#include "src/cli/io/private.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    yvex_server_engine_summary engine;
    yvex_cli_model_profile_selection model;
    char ordinal[16], generation[24], sessions[24];
} loaded_model_candidate;

typedef struct {
    const char *model, *variant;
    int json;
} model_runtime_options;

static int model_runtime_error(const yvex_error *err)
{
    yvex_cli_out_writef(stderr, "yvex: %s\n", yvex_error_message(err));
    if (yvex_error_code(err) == YVEX_ERR_IO)
        yvex_cli_out_fputs("\nstart one with:\n  yvex serve\n", stderr);
    return 1;
}

static int model_runtime_options_parse(int argc, char **argv, size_t consumed,
                                       model_runtime_options *out)
{
    size_t index;
    memset(out, 0, sizeof(*out));
    for (index = consumed + 1u; index < (size_t)argc; ++index) {
        if (!strcmp(argv[index], "--variant")) {
            if (out->variant || index + 1u >= (size_t)argc || !argv[index + 1u][0])
                return 2;
            out->variant = argv[++index];
        } else if (!strcmp(argv[index], "--json")) out->json = 1;
        else if (argv[index][0] == '-') return 2;
        else if (!out->model) out->model = argv[index];
        else return 2;
    }
    return 0;
}

static int model_request(yvex_client_operation operation, const char *alias,
                         unsigned long long generation,
                         yvex_server_engine_summary *engine, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    yvex_cli_client_request_init(&request, operation);
    snprintf(request.model_alias, sizeof(request.model_alias), "%s", alias);
    request.engine_generation = generation;
    rc = yvex_cli_client_request_open(&client, &request, err);
    if (rc == YVEX_OK) rc = yvex_client_receive(client, &message, err);
    if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
        rc = message.status;
        yvex_error_set(err, (yvex_status)rc, "client.model-runtime", message.reason);
    } else if (rc == YVEX_OK && message.kind == YVEX_CLIENT_MESSAGE_ENGINE) {
        *engine = message.engine;
    } else if (rc == YVEX_OK) {
        rc = YVEX_ERR_FORMAT;
        yvex_error_set(err, YVEX_ERR_FORMAT, "client.model-runtime",
                       "host returned an unexpected engine response");
    }
    yvex_client_close(&client);
    return rc;
}

static const char *model_engine_state(yvex_server_engine_state state)
{
    switch (state) {
    case YVEX_SERVER_ENGINE_UNLOADED: return "unloaded";
    case YVEX_SERVER_ENGINE_LOADING: return "loading";
    case YVEX_SERVER_ENGINE_LOADED: return "loaded";
    case YVEX_SERVER_ENGINE_DRAINING: return "draining";
    case YVEX_SERVER_ENGINE_UNLOADING: return "unloading";
    case YVEX_SERVER_ENGINE_FAILED: return "failed";
    }
    return "unknown";
}

static const char *model_engine_kind_name(yvex_server_engine_kind kind)
{
    return kind == YVEX_SERVER_ENGINE_MEDIA ? "media" : "text";
}

static void model_selection_from_engine(
    const yvex_server_engine_summary *engine,
    yvex_cli_model_profile_selection *model)
{
    const char *selector = engine->target_id[0] ? engine->target_id : engine->alias;
    memset(model, 0, sizeof(*model));
    snprintf(model->model_selector, sizeof(model->model_selector), "%s", selector);
    snprintf(model->model_name, sizeof(model->model_name), "%s", selector);
    snprintf(model->profile_alias, sizeof(model->profile_alias), "%s", engine->alias);
    snprintf(model->runtime_target, sizeof(model->runtime_target), "%s", selector);
    snprintf(model->artifact_identity, sizeof(model->artifact_identity), "%s",
             engine->artifact_identity);
    snprintf(model->variant, sizeof(model->variant), "%.16s",
             engine->artifact_identity[0] ? engine->artifact_identity : engine->alias);
    snprintf(model->format, sizeof(model->format), "hosted");
    snprintf(model->quant_precision, sizeof(model->quant_precision), "host-reported");
    snprintf(model->backend, sizeof(model->backend), "%s",
             yvex_backend_kind_name(engine->backend));
    snprintf(model->engine_kind, sizeof(model->engine_kind), "%s",
             model_engine_kind_name(engine->engine_kind));
    model->context_capacity = engine->context_capacity;
}

static void model_runtime_json(const char *operation,
                               const yvex_cli_model_profile_selection *model,
                               const yvex_server_engine_summary *engine)
{
    yvex_cli_out_fputs("{\"schema\":\"yvex.model.runtime.v1\",\"operation\":",
                       stdout);
    yvex_cli_out_json_string(stdout, operation);
    yvex_cli_out_fputs(",\"model\":", stdout);
    yvex_cli_out_json_string(stdout, model->model_selector);
    yvex_cli_out_fputs(",\"name\":", stdout);
    yvex_cli_out_json_string(stdout, model->model_name);
    yvex_cli_out_fputs(",\"variant\":", stdout);
    yvex_cli_out_json_string(stdout, model->variant);
    yvex_cli_out_fputs(",\"format\":", stdout);
    yvex_cli_out_json_string(stdout, model->format);
    yvex_cli_out_fputs(",\"quant_precision\":", stdout);
    yvex_cli_out_json_string(stdout, model->quant_precision);
    yvex_cli_out_fputs(",\"backend\":", stdout);
    yvex_cli_out_json_string(stdout, model->backend);
    yvex_cli_out_fputs(",\"profile_identity\":", stdout);
    yvex_cli_out_json_string(stdout, model->profile_alias);
    yvex_cli_out_fputs(",\"artifact_identity\":", stdout);
    yvex_cli_out_json_string(stdout, model->artifact_identity);
    yvex_cli_out_writef(stdout, ",\"generation\":%llu,\"state\":",
                        engine->generation);
    yvex_cli_out_json_string(stdout, model_engine_state(engine->state));
    yvex_cli_out_fputs("}\n", stdout);
}

static void model_runtime_human(const char *operation,
                                const yvex_cli_model_profile_selection *model,
                                const yvex_server_engine_summary *engine)
{
    static const yvex_cli_table_column columns[] = {
        {"FIELD", 10u, 18u, YVEX_CLI_TABLE_LEFT, 0},
        {"VALUE", 16u, 90u, YVEX_CLI_TABLE_LEFT, 1}
    };
    const char *keys[] = {"Model", "Format", "Quant / precision", "Variant",
                          "Backend", "Generation", "State"};
    const char *values[7];
    char generation[32];
    yvex_cli_table_cell cells[7][2];
    yvex_cli_table_row rows[7];
    size_t index;
    snprintf(generation, sizeof(generation), "%llu", engine->generation);
    values[0] = model->model_selector;
    values[1] = model->format;
    values[2] = model->quant_precision[0] ? model->quant_precision : "--";
    values[3] = model->variant;
    values[4] = model->backend;
    values[5] = generation;
    values[6] = model_engine_state(engine->state);
    yvex_cli_out_writef(stdout, "MODEL %s\n\n", operation);
    for (index = 0u; index < 7u; ++index) {
        cells[index][0] = (yvex_cli_table_cell){keys[index], YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){values[index],
            index == 6u && engine->state == YVEX_SERVER_ENGINE_LOADED
                ? YVEX_CLI_TABLE_SUCCESS : YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){cells[index], NULL, YVEX_CLI_TABLE_PLAIN};
    }
    (void)yvex_cli_table_render(stdout, columns, 2u, rows, 7u);
}

static int engine_catalog_fetch(yvex_server_engine_summary *engines,
                                unsigned long long capacity,
                                unsigned long long *count, yvex_error *err)
{
    yvex_client_request request;
    yvex_client_message message;
    yvex_client *client = NULL;
    int rc;
    *count = 0u;
    yvex_cli_client_request_init(&request, YVEX_CLIENT_OP_ENGINE_LIST);
    rc = yvex_cli_client_request_open(&client, &request, err);
    while (rc == YVEX_OK) {
        rc = yvex_client_receive(client, &message, err);
        if (rc != YVEX_OK || message.kind == YVEX_CLIENT_MESSAGE_ACK) break;
        if (message.kind == YVEX_CLIENT_MESSAGE_ERROR) {
            rc = message.status;
            yvex_error_set(err, (yvex_status)rc, "client.model-catalog",
                           message.reason);
            break;
        }
        if (message.kind != YVEX_CLIENT_MESSAGE_ENGINE || *count == capacity) {
            rc = YVEX_ERR_FORMAT;
            yvex_error_set(err, YVEX_ERR_FORMAT, "client.model-catalog",
                           "host returned an invalid engine catalog");
            break;
        }
        engines[(*count)++] = message.engine;
    }
    yvex_client_close(&client);
    return rc;
}

static int loaded_candidate_matches(const loaded_model_candidate *candidate,
                                    const char *model, const char *variant,
                                    int text_only)
{
    if (candidate->engine.state != YVEX_SERVER_ENGINE_LOADED ||
        !candidate->engine.execution_ready ||
        (text_only && candidate->engine.engine_kind != YVEX_SERVER_ENGINE_TEXT)) return 0;
    if (model && strcmp(candidate->model.model_selector, model) &&
        strcmp(candidate->model.model_name, model) &&
        strcmp(candidate->model.runtime_target, model) &&
        strcmp(candidate->engine.alias, model) &&
        strcmp(candidate->engine.target_id, model)) return 0;
    if (variant && strcmp(candidate->model.variant, variant) &&
        strcmp(candidate->model.profile_alias, variant) &&
        strcmp(candidate->model.artifact_identity, variant) &&
        strcmp(candidate->engine.alias, variant) &&
        strcmp(candidate->engine.artifact_identity, variant)) return 0;
    return 1;
}

static unsigned long long loaded_candidates_build(
    const yvex_server_engine_summary *engines, unsigned long long engine_count,
    const char *model, const char *variant, int text_only,
    loaded_model_candidate out[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES])
{
    unsigned long long index, count = 0u;
    for (index = 0u; index < engine_count; ++index) {
        loaded_model_candidate candidate;
        memset(&candidate, 0, sizeof(candidate));
        candidate.engine = engines[index];
        if (yvex_cli_model_profile_resolve_alias(candidate.engine.alias,
                                                 &candidate.model))
            model_selection_from_engine(&candidate.engine, &candidate.model);
        if (!loaded_candidate_matches(&candidate, model, variant, text_only)) continue;
        snprintf(candidate.ordinal, sizeof(candidate.ordinal), "%llu", count + 1u);
        snprintf(candidate.generation, sizeof(candidate.generation), "%llu",
                 candidate.engine.generation);
        snprintf(candidate.sessions, sizeof(candidate.sessions), "%llu",
                 candidate.engine.session_count);
        out[count++] = candidate;
    }
    return count;
}

int yvex_cli_loaded_models_snapshot(yvex_cli_loaded_model_fact *models,
                                    unsigned long long capacity,
                                    unsigned long long *count,
                                    yvex_error *err)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long engine_count, index;
    int rc;
    if (!models || !capacity || !count || !err) return YVEX_ERR_INVALID_ARG;
    *count = 0u;
    rc = engine_catalog_fetch(engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
                              &engine_count, err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < engine_count && *count < capacity; ++index) {
        yvex_cli_loaded_model_fact *fact;
        if (engines[index].state != YVEX_SERVER_ENGINE_LOADED ||
            !engines[index].execution_ready) continue;
        fact = &models[*count];
        memset(fact, 0, sizeof(*fact));
        if (yvex_cli_model_profile_resolve_alias(engines[index].alias,
                                                 &fact->model))
            model_selection_from_engine(&engines[index], &fact->model);
        snprintf(fact->engine.alias, sizeof(fact->engine.alias), "%s",
                 engines[index].alias);
        fact->engine.generation = engines[index].generation;
        fact->session_count = engines[index].session_count;
        (*count)++;
    }
    return YVEX_OK;
}

static void loaded_candidates_render(const loaded_model_candidate *candidates,
                                     unsigned long long count)
{
    static const yvex_cli_table_column columns[] = {
        {"#", 1u, 3u, YVEX_CLI_TABLE_RIGHT, 0},
        {"MODEL", 12u, 34u, YVEX_CLI_TABLE_LEFT, 0},
        {"VARIANT", 18u, 42u, YVEX_CLI_TABLE_LEFT, 0},
        {"BACKEND", 5u, 10u, YVEX_CLI_TABLE_LEFT, 0},
        {"GEN", 3u, 8u, YVEX_CLI_TABLE_RIGHT, 0},
        {"SESS", 4u, 6u, YVEX_CLI_TABLE_RIGHT, 0}
    };
    yvex_cli_table_cell cells[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES][6];
    yvex_cli_table_row rows[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    unsigned long long index;
    for (index = 0u; index < count; ++index) {
        const loaded_model_candidate *candidate = &candidates[index];
        cells[index][0] = (yvex_cli_table_cell){candidate->ordinal, YVEX_CLI_TABLE_DIM};
        cells[index][1] = (yvex_cli_table_cell){candidate->model.model_selector,
                                                YVEX_CLI_TABLE_ACCENT};
        cells[index][2] = (yvex_cli_table_cell){candidate->model.variant,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][3] = (yvex_cli_table_cell){candidate->model.backend,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][4] = (yvex_cli_table_cell){candidate->generation,
                                                YVEX_CLI_TABLE_PLAIN};
        cells[index][5] = (yvex_cli_table_cell){candidate->sessions,
                                                YVEX_CLI_TABLE_PLAIN};
        rows[index] = (yvex_cli_table_row){cells[index], NULL, YVEX_CLI_TABLE_PLAIN};
    }
    (void)yvex_cli_table_render(stdout, columns, 6u, rows, (size_t)count);
}

static int loaded_choice_read(unsigned long long count, unsigned long long *choice)
{
    char line[64], *end = NULL, *newline;
    unsigned long long value;
    for (;;) {
        yvex_cli_out_writef(stdout, "Model [1-%llu, q to cancel] > ", count);
        yvex_cli_out_flush(stdout);
        if (!fgets(line, sizeof(line), stdin)) return 0;
        newline = strpbrk(line, "\r\n");
        if (newline) *newline = '\0';
        if (!line[0] || !strcmp(line, "q") || !strcmp(line, "Q")) return 0;
        errno = 0;
        value = strtoull(line, &end, 10);
        if (!errno && end != line && !*end && value && value <= count) {
            *choice = value - 1u;
            return 1;
        }
        yvex_cli_out_fputs("Choose a listed number, or q.\n", stdout);
    }
}

int yvex_cli_model_loaded_select(const char *model, const char *variant,
                                 int text_only,
                                 yvex_cli_engine_binding *binding,
                                 yvex_cli_model_profile_selection *selection)
{
    yvex_server_engine_summary engines[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    loaded_model_candidate candidates[YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES];
    yvex_error err;
    unsigned long long engine_count, count, choice = 0u;
    int rc = engine_catalog_fetch(engines, YVEX_SERVER_IMPLEMENTATION_MAXIMUM_ENGINES,
                                  &engine_count, &err);
    if (rc != YVEX_OK) return model_runtime_error(&err);
    count = loaded_candidates_build(engines, engine_count, model, variant, text_only,
                                    candidates);
    if (!count) {
        yvex_cli_out_writef(stderr, "yvex: no loaded%s model%s%s\n",
                            text_only ? " text" : "",
                            model ? " matches " : " is available",
                            model ? model : "");
        yvex_cli_out_fputs("\nload one with:\n  yvex model load MODEL\n", stderr);
        return 1;
    }
    if (count > 1u) {
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            yvex_cli_out_fputs(
                "yvex: multiple loaded models match; pass exact MODEL and --variant\n",
                stderr);
            return 2;
        }
        yvex_cli_out_fputs("Select loaded model\n\n", stdout);
        loaded_candidates_render(candidates, count);
        if (!loaded_choice_read(count, &choice)) return 2;
    }
    snprintf(binding->alias, sizeof(binding->alias), "%s",
             candidates[choice].engine.alias);
    binding->generation = candidates[choice].engine.generation;
    if (selection) *selection = candidates[choice].model;
    return 0;
}

int yvex_cli_model_load_command(int argc, char **argv, size_t consumed)
{
    model_runtime_options options;
    yvex_cli_model_profile_selection model;
    yvex_server_engine_summary engine;
    yvex_error err;
    int rc = model_runtime_options_parse(argc, argv, consumed, &options);
    if (rc) return rc;
    rc = yvex_cli_model_profile_select(options.model, options.variant, 0, &model);
    if (rc) return rc;
    rc = model_request(YVEX_CLIENT_OP_ENGINE_LOAD, model.profile_alias, 0u, &engine,
                       &err);
    if (rc != YVEX_OK) return model_runtime_error(&err);
    if (options.json) model_runtime_json("load", &model, &engine);
    else model_runtime_human("LOADED", &model, &engine);
    return 0;
}

int yvex_cli_model_unload_command(int argc, char **argv, size_t consumed)
{
    model_runtime_options options;
    yvex_cli_model_profile_selection model;
    yvex_cli_engine_binding binding;
    yvex_server_engine_summary engine;
    yvex_error err;
    int rc = model_runtime_options_parse(argc, argv, consumed, &options);
    if (rc) return rc;
    rc = yvex_cli_model_loaded_select(options.model, options.variant, 0, &binding,
                                      &model);
    if (rc) return rc;
    rc = model_request(YVEX_CLIENT_OP_ENGINE_UNLOAD, binding.alias,
                       binding.generation, &engine, &err);
    if (rc != YVEX_OK) return model_runtime_error(&err);
    if (options.json) model_runtime_json("unload", &model, &engine);
    else model_runtime_human("UNLOADED", &model, &engine);
    return 0;
}
