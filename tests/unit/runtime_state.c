/*
 * Exercises the session-local provider is allocation-stable and transactionally exact. Expected
 * state is independently compared across chunk and decode transactions. Focused internal-ABI
 * coverage; no fixture enters production objects.
 */
#include "tests/test.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/graph_state.h>
#include <yvex/internal/candidate.h>
#include <yvex/internal/runtime.h>

#include "src/graph/private.h"

typedef struct {
    yvex_attention_publication publication;
    unsigned int token_id;
    float raw[2], compressed[2], indexer[2];
    unsigned long long compressed_position, indexer_position;
} state_token;

typedef struct {
    yvex_attention_plan plan;
    yvex_attention_layer_plan layers[3];
} state_plan_fixture;

static int state_cancel_requested(void *context)
{
    return context && *(const int *)context;
}

static yvex_attention_layer_plan state_layer(unsigned long long index,
                                             yvex_attention_class attention_class)
{
    yvex_attention_layer_plan layer;

    memset(&layer, 0, sizeof(layer));
    layer.layer_index = index;
    layer.attention_class = attention_class;
    layer.head_dimension = 2ull;
    layer.hidden_dimension = 4ull;
    layer.query_heads = 2ull;
    layer.kv_heads = 1ull;
    layer.sliding_window = attention_class == YVEX_ATTENTION_CLASS_HCA ? 128ull : 4ull;
    if (attention_class != YVEX_ATTENTION_CLASS_SWA) {
        layer.compressor_required = 1;
        layer.compression_ratio = attention_class == YVEX_ATTENTION_CLASS_CSA ? 4ull : 128ull;
    }
    if (attention_class == YVEX_ATTENTION_CLASS_CSA) {
        layer.indexer_required = 1;
        layer.indexer_heads = 1ull;
        layer.indexer_head_dimension = 2ull;
        layer.indexer_topk = 512ull;
        layer.sparse_topk.required = 1;
        layer.sparse_topk.version = 1u;
        layer.sparse_topk.k = 512ull;
        layer.sparse_topk.score_descending = 1;
        layer.sparse_topk.equal_score_ordinal_ascending = 1;
        layer.sparse_topk.duplicate_ordinal_refused = 1;
        layer.sparse_topk.output_ranked_order = 1;
    }
    return layer;
}

static void state_plan_open(state_plan_fixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    fixture->layers[0] = state_layer(0ull, YVEX_ATTENTION_CLASS_SWA);
    fixture->layers[1] = state_layer(1ull, YVEX_ATTENTION_CLASS_CSA);
    fixture->layers[2] = state_layer(2ull, YVEX_ATTENTION_CLASS_HCA);
    fixture->plan.layers = fixture->layers;
    fixture->plan.layer_count = 3ull;
    fixture->plan.summary.layer_count = 3ull;
    fixture->plan.summary.swa_layer_count = 1ull;
    fixture->plan.summary.csa_layer_count = 1ull;
    fixture->plan.summary.hca_layer_count = 1ull;
    fixture->plan.summary.status = YVEX_ATTENTION_STATUS_EXECUTION_READY;
    (void)snprintf(fixture->plan.summary.attention_plan_identity,
                   sizeof(fixture->plan.summary.attention_plan_identity),
                   "%064x", 0x5a7eu);
}

static void state_recipe_history(yvex_attention_state_recipe *recipe,
                                 yvex_attention_state_binding binding,
                                 unsigned long long capacity,
                                 unsigned long long width)
{
    yvex_attention_state_component_recipe *component =
        &recipe->components[recipe->component_count];

    component->schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1;
    component->ordinal = recipe->component_count++;
    component->kind = YVEX_ATTENTION_STATE_COMPONENT_HISTORY;
    component->binding = binding;
    component->capacity = capacity;
    component->value_width = width;
}

static int state_recipe_rolling(yvex_attention_state_recipe *recipe,
                                const yvex_attention_layer_plan *layer,
                                yvex_attention_rolling_kind kind,
                                yvex_attention_state_binding binding)
{
    yvex_attention_state_component_recipe *component =
        &recipe->components[recipe->component_count];
    yvex_attention_rolling_state_view *rolling = &component->rolling;
    unsigned long long extent;
    int overlap, rotated;

    component->schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1;
    component->ordinal = recipe->component_count++;
    component->kind = YVEX_ATTENTION_STATE_COMPONENT_ROLLING;
    component->binding = binding;
    if (!yvex_attention_rolling_geometry(
            layer, kind, &rolling->ratio, &rolling->head_dimension,
            &rolling->state_width, &rolling->state_slots, &overlap, &rotated) ||
        !yvex_core_u64_mul(rolling->state_slots, rolling->state_width, &extent))
        return 0;
    component->capacity = rolling->state_slots;
    component->value_width = rolling->state_width;
    rolling->present = 1;
    rolling->schema_version = YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1;
    rolling->kind = kind;
    rolling->attention_class = layer->attention_class;
    rolling->layer_index = layer->layer_index;
    rolling->next_token_position = recipe->initial_position;
    rolling->cursor = recipe->initial_position % rolling->ratio;
    rolling->kv_state_stride = rolling->state_width;
    rolling->score_state_stride = rolling->state_width;
    rolling->kv_state_extent = extent;
    rolling->score_state_extent = extent;
    rolling->overlap = overlap;
    rolling->rotated = rotated;
    (void)snprintf(rolling->attention_plan_identity,
                   sizeof(rolling->attention_plan_identity), "%s",
                   recipe->attention_plan_identity);
    return 1;
}

static int state_recipe_project(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe_request *request,
    yvex_attention_state_recipe *recipe, yvex_attention_failure *failure,
    yvex_error *err)
{
    unsigned long long local_limit, compressed_capacity = 0ull;

    if (!layer || !request || !recipe ||
        request->layer_ordinal != layer->layer_index ||
        !request->attention_plan_identity) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "test.graph.state.recipe",
                       "complete family state recipe facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(recipe, 0, sizeof(*recipe));
    recipe->schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1;
    recipe->layer_index = layer->layer_index;
    recipe->selection_key = (unsigned long long)layer->attention_class + 1ull;
    recipe->initial_position = request->initial_position;
    recipe->final_position = request->final_position;
    (void)snprintf(recipe->attention_plan_identity,
                   sizeof(recipe->attention_plan_identity), "%s",
                   request->attention_plan_identity);
    local_limit = layer->sliding_window - 1ull;
    state_recipe_history(recipe, YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY,
                         request->final_position < local_limit
                             ? request->final_position : local_limit,
                         layer->head_dimension);
    if (layer->compressor_required) {
        compressed_capacity = request->final_position / layer->compression_ratio;
        state_recipe_history(
            recipe, YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY,
            compressed_capacity, layer->head_dimension);
        if (!state_recipe_rolling(recipe, layer, YVEX_ATTENTION_ROLLING_MAIN,
                                  YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING))
            goto malformed;
    }
    if (layer->indexer_required) {
        state_recipe_history(recipe, YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY,
                             compressed_capacity,
                             layer->indexer_head_dimension);
        if (!state_recipe_rolling(recipe, layer, YVEX_ATTENTION_ROLLING_INDEXER,
                                  YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING))
            goto malformed;
    }
    return yvex_attention_state_recipe_seal(recipe, err);

malformed:
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = YVEX_ATTENTION_FAILURE_HISTORY;
        failure->layer_index = layer->layer_index;
        failure->reason = "family rolling recipe is malformed";
    }
    yvex_error_set(err, YVEX_ERR_FORMAT, "test.graph.state.recipe",
                   "family rolling recipe is malformed");
    return YVEX_ERR_FORMAT;
}

static const yvex_graph_family_api *state_family(void)
{
    static const yvex_graph_family_api family = {
        .state_recipe = state_recipe_project,
        .workspace_recipe = yvex_attention_workspace_recipe_build,
        .history_validate = yvex_attention_history_validate,
        .rolling_state_step_cpu = yvex_attention_rolling_state_step_cpu};
    return &family;
}

static unsigned long long state_recipe_capacity(
    const yvex_attention_state_recipe *recipe,
    yvex_attention_state_binding binding)
{
    unsigned int index;
    for (index = 0u; index < recipe->component_count; ++index)
        if (recipe->components[index].binding == binding)
            return recipe->components[index].capacity;
    return 0ull;
}

static int state_recipe_unseal(yvex_attention_state_recipe *recipe)
{
    recipe->identity[0] = '\0';
    return 1;
}

static int test_state_recipe_identity(const state_plan_fixture *fixture)
{
    yvex_attention_state_recipe_request request;
    yvex_attention_state_recipe recipe, changed;
    yvex_attention_failure failure;
    char baseline[YVEX_ATTENTION_IDENTITY_CAP];
    yvex_error err;

    memset(&request, 0, sizeof(request));
    request.layer_ordinal = 1ull;
    request.initial_position = 12ull;
    request.final_position = 2052ull;
    request.attention_plan_identity = fixture->plan.summary.attention_plan_identity;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_recipe_project(&fixture->layers[1], &request, &recipe,
                             &failure, &err) == YVEX_OK &&
            yvex_attention_state_recipe_seal(&recipe, &err) == YVEX_OK,
        "family state recipe seals and independently validates");
    (void)snprintf(baseline, sizeof(baseline), "%s", recipe.identity);
    changed = recipe;
    changed.components[0].capacity++;
    YVEX_TEST_ASSERT(
        yvex_attention_state_recipe_seal(&changed, &err) == YVEX_ERR_STATE &&
            state_recipe_unseal(&changed) &&
            yvex_attention_state_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "history-capacity mutation is stale until resealed and changes identity");
    changed = recipe;
    changed.components[1].binding = changed.components[0].binding;
    YVEX_TEST_ASSERT(
        yvex_attention_state_recipe_seal(&changed, &err) == YVEX_ERR_FORMAT,
        "duplicate state binding refuses before state allocation");
    changed = recipe;
    changed.components[2].rolling.cursor++;
    YVEX_TEST_ASSERT(
        yvex_attention_state_recipe_seal(&changed, &err) == YVEX_ERR_STATE &&
            state_recipe_unseal(&changed) &&
            yvex_attention_state_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "rolling-policy mutation changes the sealed state recipe identity");
    changed = recipe;
    changed.selection_key++;
    YVEX_TEST_ASSERT(
        yvex_attention_state_recipe_seal(&changed, &err) == YVEX_ERR_STATE &&
            state_recipe_unseal(&changed) &&
            yvex_attention_state_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "family selection-key mutation changes the state recipe identity");
    return 0;
}

/* Prove workspace recipes bind ordered extents without backend or pointer identity. */
static int test_workspace_recipe_identity(void)
{
    yvex_attention_workspace_recipe recipe, changed;
    char baseline[YVEX_ATTENTION_IDENTITY_CAP];
    yvex_error err;
    unsigned int index;

    memset(&recipe, 0, sizeof(recipe));
    recipe.schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1;
    recipe.layer_index = 1ull;
    recipe.token_capacity = 4ull;
    recipe.mode = YVEX_ATTENTION_EXECUTION_FULL;
    recipe.scope = YVEX_ATTENTION_OPERATION_ENVELOPE;
    recipe.component_count = 2u;
    (void)snprintf(recipe.state_recipe_identity,
                   sizeof(recipe.state_recipe_identity), "%064x", 0x713u);
    for (index = 0u; index < recipe.component_count; ++index) {
        recipe.components[index].schema_version =
            YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1;
        recipe.components[index].ordinal = index;
        recipe.components[index].kind = index == 0u
                                            ? YVEX_ATTENTION_WORKSPACE_INGRESS
                                            : YVEX_ATTENTION_WORKSPACE_QUERY;
        recipe.components[index].lifetime = YVEX_ATTENTION_WORKSPACE_EXECUTION;
        recipe.components[index].element_count = index == 0u ? 4ull : 8ull;
        recipe.components[index].element_width = sizeof(float);
        recipe.components[index].alignment = 8ull;
        recipe.components[index].scales_with_tokens = 1;
    }
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_recipe_seal(&recipe, &err) == YVEX_OK,
        "pointer-free workspace recipe seals deterministically");
    (void)snprintf(baseline, sizeof(baseline), "%s", recipe.identity);
    changed = recipe;
    changed.token_capacity++;
    changed.identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "token bucket mutation changes workspace recipe identity");
    changed = recipe;
    changed.evidence_level = YVEX_ATTENTION_EVIDENCE_FULL;
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "evidence policy changes workspace recipe identity");
    changed = recipe;
    changed.components[1].element_count++;
    changed.identity[0] = '\0';
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_recipe_seal(&changed, &err) == YVEX_OK &&
            strcmp(baseline, changed.identity) != 0,
        "component-extent mutation changes workspace recipe identity");
    changed = recipe;
    changed.components[1].kind = changed.components[0].kind;
    YVEX_TEST_ASSERT(
        yvex_attention_workspace_recipe_seal(&changed, &err) == YVEX_ERR_FORMAT,
        "duplicate workspace component kind refuses before backend lowering");
    return 0;
}

static int test_workspace_capture_geometry(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    yvex_attention_state_recipe_request request;
    yvex_attention_state_recipe state, index_state;
    yvex_attention_workspace_recipe workspace, index_workspace;
    const yvex_attention_workspace_component *local = NULL;
    const yvex_attention_workspace_component *current = NULL;
    const yvex_attention_workspace_component *candidate = NULL;
    const yvex_attention_workspace_component *positions = NULL;
    const yvex_attention_workspace_component *scores = NULL, *valid = NULL;
    yvex_attention_failure failure;
    yvex_error err;
    unsigned int index;

    memset(&request, 0, sizeof(request));
    request.layer_ordinal = 2ull;
    request.final_position = 384ull;
    request.attention_plan_identity =
        fixture->plan.summary.attention_plan_identity;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_recipe_project(&fixture->layers[2], &request, &state,
                             &failure, &err) == YVEX_OK &&
            yvex_attention_state_recipe_seal(&state, &err) == YVEX_OK &&
            family->workspace_recipe(
                &fixture->layers[2], &state, YVEX_ATTENTION_EXECUTION_FULL,
                YVEX_ATTENTION_OPERATION_CORE, YVEX_ATTENTION_EVIDENCE_FULL,
                4ull, &workspace, &failure, &err) == YVEX_OK,
        "HCA capture workspace recipe seals from the admitted state geometry");
    for (index = 0u; index < workspace.component_count; ++index) {
        const yvex_attention_workspace_component *component =
            &workspace.components[index];
        if (component->kind == YVEX_ATTENTION_WORKSPACE_LOCAL_VALUES)
            local = component;
        else if (component->kind == YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_VALUES)
            current = component;
        else if (component->kind ==
                 YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_CANDIDATE_VALUES)
            candidate = component;
    }
    YVEX_TEST_ASSERT(
        local &&
            local->element_count ==
                state.components[0].capacity + workspace.token_capacity &&
            !local->scales_with_tokens,
        "captured local staging reserves the sealed history plus token bucket");
    YVEX_TEST_ASSERT(
        current && candidate &&
            current->element_count == state.components[2].rolling.kv_state_extent &&
            candidate->element_count == current->element_count &&
            !current->scales_with_tokens &&
            !candidate->scales_with_tokens,
        "rolling current and final delta own exact token-independent semantic extents");
    request.layer_ordinal = 1ull;
    request.final_position = 524288ull;
    YVEX_TEST_ASSERT(
        state_recipe_project(&fixture->layers[1], &request, &index_state,
                             &failure, &err) == YVEX_OK &&
            family->workspace_recipe(
                &fixture->layers[1], &index_state, YVEX_ATTENTION_EXECUTION_FULL,
                YVEX_ATTENTION_OPERATION_CORE, YVEX_ATTENTION_EVIDENCE_NONE,
                4ull, &index_workspace, &failure, &err) == YVEX_OK,
        "deep CSA workspace seals every index-selection scratch class");
    for (index = 0u; index < index_workspace.component_count; ++index) {
        const yvex_attention_workspace_component *component =
            &index_workspace.components[index];
        if (component->kind == YVEX_ATTENTION_WORKSPACE_TOPK_POSITIONS)
            positions = component;
        else if (component->kind == YVEX_ATTENTION_WORKSPACE_TOPK_SCORES)
            scores = component;
        else if (component->kind == YVEX_ATTENTION_WORKSPACE_TOPK_VALID_INDICES)
            valid = component;
    }
    YVEX_TEST_ASSERT(
        positions && positions->element_count == 512ull && positions->scales_with_tokens &&
            scores && scores->element_count == 131073ull && !scores->scales_with_tokens &&
            valid && valid->element_count == scores->element_count && !valid->scales_with_tokens,
        "deep CSA recipe reserves selected positions and token-local candidate scratch");
    return 0;
}

/* Prove one immutable capacity plan owns selection, exact per-layer geometry, and identity. */
static int test_capacity_plan(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    yvex_graph_attention_capacity_plan *first = NULL, *second = NULL;
    yvex_graph_attention_capacity_request request;
    const yvex_graph_attention_capacity_summary *summary;
    const yvex_graph_attention_capacity_layer *swa, *csa, *hca;
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_error err;

    memset(&request, 0, sizeof(request));
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    request.token_count = 2ull;
    request.execution_count = 2ull;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &first, family, &fixture->plan, &request, &err) == YVEX_OK,
                     "quick capacity plan seals without state allocation");
    summary = yvex_graph_attention_capacity_plan_summary(first);
    swa = yvex_graph_attention_capacity_plan_layer(first, 0ull);
    csa = yvex_graph_attention_capacity_plan_layer(first, 1ull);
    hca = yvex_graph_attention_capacity_plan_layer(first, 2ull);
    YVEX_TEST_ASSERT(summary && summary->schema_version ==
                                    YVEX_GRAPH_ATTENTION_CAPACITY_SCHEMA_V1 &&
                         summary->selected_layer_count == 3ull &&
                         summary->first_layer == 0ull &&
                         summary->maximum_token_count == 2ull &&
                         yvex_sha256_hex_valid(summary->identity),
                     "capacity summary publishes sealed selection and identity facts");
    YVEX_TEST_ASSERT(
        summary->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].capacity == 133ull,
                     "capacity summary publishes exact aggregate local capacity");
    YVEX_TEST_ASSERT(
        summary->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].capacity == 515ull,
                     "capacity summary publishes exact aggregate compressed capacity");
    YVEX_TEST_ASSERT(
        summary->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].capacity == 514ull,
                     "capacity summary publishes exact aggregate indexer capacity");
    YVEX_TEST_ASSERT(
        summary->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].maximum_capacity == 127ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].maximum_capacity == 514ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].maximum_capacity == 514ull &&
                         summary->maximum_compression_ratio == 128ull &&
                         summary->maximum_topk_capacity == 512ull,
                     "capacity summary publishes exact maxima and policy facts");
    YVEX_TEST_ASSERT(
        swa && swa->selected && swa->recipe.initial_position == 4ull &&
            swa->recipe.final_position == 8ull &&
            state_recipe_capacity(&swa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) == 3ull &&
            csa && csa->selected && csa->recipe.initial_position == 2052ull &&
            csa->recipe.final_position == 2056ull &&
            state_recipe_capacity(&csa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) == 514ull &&
            state_recipe_capacity(&csa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) == 514ull &&
            hca && hca->selected && hca->recipe.initial_position == 127ull &&
            hca->recipe.final_position == 131ull &&
            state_recipe_capacity(&hca->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) == 1ull,
                     "quick capacity plan derives exact canonical class positions");
    (void)snprintf(identity, sizeof(identity), "%s", summary->identity);
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &second, family, &fixture->plan, &request, &err) == YVEX_OK &&
                         strcmp(identity,
                                yvex_graph_attention_capacity_plan_summary(second)->identity) == 0,
                     "equivalent capacity facts produce one deterministic identity");
    yvex_graph_attention_capacity_plan_close(&second);
    request.execution_count++;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &second, family, &fixture->plan, &request, &err) == YVEX_OK &&
                         strcmp(identity,
                                yvex_graph_attention_capacity_plan_summary(second)->identity) != 0,
                     "execution extent mutation changes capacity identity");
    yvex_graph_attention_capacity_plan_close(&second);
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    request.execution_count = 1ull;
    request.history_tokens = request.start_position = 2052ull;
    request.select_selection_key = 1;
    request.selection_key = (unsigned long long)YVEX_ATTENTION_CLASS_CSA + 1ull;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &second, family, &fixture->plan, &request, &err) == YVEX_OK &&
                         yvex_graph_attention_capacity_plan_summary(second)->selected_layer_count == 1ull &&
                         state_recipe_capacity(
                             &yvex_graph_attention_capacity_plan_layer(second, 1ull)->recipe,
                             YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) == 513ull &&
                         state_recipe_capacity(
                             &yvex_graph_attention_capacity_plan_layer(second, 1ull)->recipe,
                             YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) == 513ull,
                     "selected full-scope history derives exact CSA boundary capacity");
    yvex_graph_attention_capacity_plan_close(&second);
    request.start_position++;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &second, family, &fixture->plan, &request, &err) == YVEX_ERR_INVALID_ARG && !second,
                     "history and start-position mismatch refuses before allocation");
    memset(&request, 0, sizeof(request));
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    request.token_count = 1ull;
    request.execution_count = 23ull;
    YVEX_TEST_ASSERT(yvex_graph_attention_capacity_plan_build(
                         &second, family, &fixture->plan, &request, &err) == YVEX_OK,
                     "mixed-class 23-dispatch capacity plan seals");
    summary = yvex_graph_attention_capacity_plan_summary(second);
    swa = yvex_graph_attention_capacity_plan_layer(second, 0ull);
    csa = yvex_graph_attention_capacity_plan_layer(second, 1ull);
    hca = yvex_graph_attention_capacity_plan_layer(second, 2ull);
    YVEX_TEST_ASSERT(
        summary && summary->selected_layer_count == 3ull &&
            summary->maximum_token_count == 1ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].capacity == 29ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].capacity == 5ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].capacity == 5ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].maximum_capacity == 23ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].maximum_capacity == 5ull &&
            summary->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].maximum_capacity == 5ull &&
            summary->maximum_compression_ratio == 128ull &&
            summary->maximum_topk_capacity == 512ull && swa &&
            swa->recipe.final_position == 23ull &&
            state_recipe_capacity(&swa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) == 3ull &&
            !state_recipe_capacity(&swa->recipe,
                                   YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) &&
            csa && csa->recipe.final_position == 23ull &&
            state_recipe_capacity(&csa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) == 3ull &&
            state_recipe_capacity(&csa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) == 5ull &&
            state_recipe_capacity(&csa->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) == 5ull &&
            hca && hca->recipe.final_position == 23ull &&
            state_recipe_capacity(&hca->recipe,
                                  YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) == 23ull &&
            !state_recipe_capacity(&hca->recipe,
                                   YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY),
        "mixed-class capacity keeps per-layer facts and publishes shared graph-bucket maxima");
    yvex_graph_attention_capacity_plan_close(&second);
    yvex_graph_attention_capacity_plan_close(&first);
    yvex_graph_attention_capacity_plan_close(&first);
    return 0;
}

static void state_token_release(state_token *token)
{
    free(token->publication.next_main_rolling_state.kv_state);
    free(token->publication.next_main_rolling_state.score_state);
    free(token->publication.next_indexer_rolling_state.kv_state);
    free(token->publication.next_indexer_rolling_state.score_state);
    memset(token, 0, sizeof(*token));
}

static int state_token_rolling(const yvex_attention_layer_plan *layer,
                               const yvex_attention_rolling_state_view *before,
                               yvex_attention_rolling_state_output *after,
                               float compressed[2], int *emitted)
{
    const yvex_graph_family_api *family = state_family();
    float *values = NULL, *scores = NULL, *ape = NULL;
    unsigned long long index, extent;
    yvex_attention_failure failure;
    yvex_error err;
    int rc;

    if (!before || !before->present) return 0;
    extent = before->kv_state_extent;
    values = (float *)calloc((size_t)before->state_width, sizeof(float));
    scores = (float *)calloc((size_t)before->state_width, sizeof(float));
    ape = (float *)calloc((size_t)before->state_width, sizeof(float));
    after->kv_state = (float *)calloc((size_t)extent, sizeof(float));
    after->score_state = (float *)calloc((size_t)before->score_state_extent, sizeof(float));
    if (!values || !scores || !ape || !after->kv_state || !after->score_state) {
        free(values);
        free(scores);
        free(ape);
        return 0;
    }
    after->kv_state_stride = before->kv_state_stride;
    after->score_state_stride = before->score_state_stride;
    after->kv_state_extent = before->kv_state_extent;
    after->score_state_extent = before->score_state_extent;
    for (index = 0ull; index < before->state_width; ++index) {
        values[index] = (float)((before->next_token_position + index) % 17ull) / 16.0f;
        scores[index] = values[index] * 0.5f;
        ape[index] = (float)(index % 3ull) / 32.0f;
    }
    yvex_error_clear(&err);
    rc = family->rolling_state_step_cpu(layer, before, values, scores, ape, after,
                                        compressed, layer->head_dimension, emitted,
                                        &failure, &err);
    free(values);
    free(scores);
    free(ape);
    return rc == YVEX_OK;
}

/* Construct one deterministic complete token publication from an immutable prior view. */
static int state_token_open(state_token *token,
                            const yvex_attention_layer_plan *layer,
                            const yvex_attention_history_view *history,
                            unsigned long long position)
{
    int main_emitted = 0, indexer_emitted = 0;

    memset(token, 0, sizeof(*token));
    token->raw[0] = (float)(position % 29ull) / 29.0f;
    token->raw[1] = -token->raw[0] - 0.03125f;
    token->publication.owned = 1;
    token->publication.complete = 1;
    (void)snprintf(token->publication.execution_identity,
                   sizeof(token->publication.execution_identity), "%064llx",
                   1ull + layer->layer_index * 4096ull + position);
    token->publication.layer_index = layer->layer_index;
    token->publication.attention_class = layer->attention_class;
    token->publication.token_position = position;
    token->publication.token_count = 1ull;
    token->token_id = (unsigned int)(position + 1ull);
    token->publication.token_ids = &token->token_id;
    token->publication.kv_width = layer->head_dimension;
    token->publication.raw_kv = token->raw;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA) return 1;
    if (!state_token_rolling(layer, &history->main_rolling_state,
                             &token->publication.next_main_rolling_state,
                             token->compressed, &main_emitted))
        goto fail;
    if (main_emitted) {
        token->compressed_position = position + 1ull - layer->compression_ratio;
        token->publication.compressed_count = 1ull;
        token->publication.compressed_stride = layer->head_dimension;
        token->publication.compressed_kv = token->compressed;
        token->publication.compressed_positions = &token->compressed_position;
    }
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        if (!state_token_rolling(layer, &history->indexer_rolling_state,
                                 &token->publication.next_indexer_rolling_state,
                                 token->indexer, &indexer_emitted) ||
            indexer_emitted != main_emitted)
            goto fail;
        if (indexer_emitted) {
            token->indexer_position = token->compressed_position;
            token->publication.indexer_count = 1ull;
            token->publication.indexer_stride = layer->indexer_head_dimension;
            token->publication.indexer_kv = token->indexer;
            token->publication.indexer_positions = &token->indexer_position;
        }
    }
    return 1;
fail:
    state_token_release(token);
    return 0;
}

typedef yvex_attention_state_provider test_state;

static int state_open(test_state *state, const yvex_graph_family_api *family,
                      const yvex_attention_plan *plan,
                      unsigned long long maximum_host_bytes,
                      yvex_attention_failure *failure, yvex_error *err)
{
    return yvex_attention_state_provider_open_persistent(
        family, plan, maximum_host_bytes, state, failure, err);
}

static int state_close(test_state *state)
{
    yvex_error err;
    int rc;

    if (!state || !state->context) return 1;
    yvex_error_clear(&err);
    rc = state->release ? state->release(&state->context, &err) : YVEX_ERR_STATE;
    if (rc == YVEX_OK) memset(state, 0, sizeof(*state));
    return rc == YVEX_OK;
}

static const yvex_attention_history_view *state_view(
    const test_state *state, unsigned long long layer_index,
    yvex_attention_state_view_kind kind)
{
    return state && state->view
               ? state->view(state->context, layer_index, kind) : NULL;
}

static int state_summary(const test_state *state,
                         yvex_graph_attention_state_summary *summary,
                         yvex_error *err)
{
    return state && state->summary
               ? state->summary(state->context, summary, err) : YVEX_ERR_STATE;
}

static int state_identity(const test_state *state, unsigned long long layer_index,
                          char output[YVEX_SHA256_HEX_CAP], yvex_error *err)
{
    return state && state->identity
               ? state->identity(state->context, layer_index, output, err)
               : YVEX_ERR_STATE;
}

static int state_begin(test_state *state, const yvex_attention_layer_plan *layer,
                       unsigned long long token_position,
                       unsigned long long token_count,
                       const yvex_attention_cancellation *cancellation,
                       yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_attention_history_view *history = NULL;

    return state && state->begin && layer
               ? state->begin(state->context, layer->layer_index, layer, NULL,
                              token_position, token_count, cancellation,
                              &history, failure, err)
               : YVEX_ERR_STATE;
}

static int state_prepare(test_state *state, const yvex_attention_layer_plan *layer,
                         const char *plan_identity)
{
    yvex_attention_state_recipe_request request;
    yvex_attention_state_recipe recipe;
    yvex_attention_failure failure;
    yvex_error err;

    memset(&request, 0, sizeof(request));
    request.layer_ordinal = layer->layer_index;
    request.final_position = layer->attention_class == YVEX_ATTENTION_CLASS_CSA
                                 ? 2052ull
                             : layer->attention_class == YVEX_ATTENTION_CLASS_HCA
                                 ? 384ull : 6ull;
    request.attention_plan_identity = plan_identity;
    yvex_error_clear(&err);
    return state_recipe_project(layer, &request, &recipe, &failure, &err) == YVEX_OK &&
           state->prepare(state->context, layer->layer_index, &recipe, NULL,
                          &failure, &err) == YVEX_OK;
}

/* Apply one generated token using either an outer chunk or one decode transaction. */
static int state_apply_token(test_state *state,
                             const yvex_attention_layer_plan *layer,
                             unsigned long long position, int outer_transaction,
                             char delta_identity[YVEX_SHA256_HEX_CAP])
{
    const yvex_attention_history_view *history;
    yvex_attention_failure failure;
    yvex_error err;
    state_token token;
    int rc;

    yvex_error_clear(&err);
    if (!outer_transaction &&
        state_begin(state, layer, position, 1ull, NULL, &failure, &err) != YVEX_OK)
        return 0;
    history = state_view(state, layer->layer_index,
                         YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
    if (!history || !state_token_open(&token, layer, history, position)) return 0;
    rc = state->stage(state->context, &token.publication, NULL, delta_identity,
                      &failure, &err);
    state_token_release(&token);
    if (rc != YVEX_OK) return 0;
    if (!outer_transaction)
        return state->commit(state->context, &failure, &err) == YVEX_OK;
    return 1;
}

static int state_stage_prefix(test_state *state,
                              const yvex_attention_layer_plan *layer,
                              unsigned long long token_count,
                              float raw[12])
{
    yvex_attention_publication publication;
    yvex_attention_failure failure;
    yvex_error err;
    char delta_identity[YVEX_SHA256_HEX_CAP];
    unsigned int tokens[5] = {1u, 2u, 3u, 4u, 5u};
    unsigned long long row;

    memset(&publication, 0, sizeof(publication));
    for (row = 0ull; row < token_count; ++row) {
        raw[row * 2ull] = (float)(row % 29ull) / 29.0f;
        raw[row * 2ull + 1ull] = -raw[row * 2ull] - 0.03125f;
    }
    publication.owned = 1;
    publication.complete = 1;
    publication.prefix_addressable = 1;
    (void)snprintf(publication.execution_identity,
                   sizeof(publication.execution_identity), "%064llx",
                   1ull + layer->layer_index * 4096ull + token_count);
    publication.layer_index = layer->layer_index;
    publication.attention_class = layer->attention_class;
    publication.token_count = token_count;
    publication.token_ids = tokens;
    publication.kv_width = layer->head_dimension;
    publication.raw_kv = raw;
    yvex_error_clear(&err);
    return state_begin(state, layer, 0ull, token_count, NULL, &failure, &err) ==
               YVEX_OK &&
           state->stage(state->context, &publication, NULL, delta_identity,
                        &failure, &err) == YVEX_OK &&
           yvex_sha256_hex_valid(delta_identity);
}

static int test_state_prefix_promotion(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    const yvex_attention_layer_plan *layer = &fixture->layers[0];
    unsigned long long accepted;

    for (accepted = 1ull; accepted <= 5ull; ++accepted) {
        test_state state = {0};
        yvex_graph_attention_state_summary summary;
        yvex_attention_failure failure;
        yvex_error err;
        const yvex_attention_history_view *view;
        float raw[12] = {0};
        unsigned long long row, tail, first;

        yvex_error_clear(&err);
        YVEX_TEST_ASSERT(
            state_open(&state, family, &fixture->plan, 1024ull * 1024ull,
                       &failure, &err) == YVEX_OK &&
                state_prepare(&state, layer,
                              fixture->plan.summary.attention_plan_identity),
            "prefix promotion state opens and prepares");
        YVEX_TEST_ASSERT(state_stage_prefix(&state, layer, 5ull, raw),
                         "one verification publication retains five checkpoints");
        YVEX_TEST_ASSERT(
            state.select_prefix(state.context, accepted, 0ull, &failure,
                                &err) == YVEX_OK,
            "arbitrary verified prefix selects without replay");
        YVEX_TEST_ASSERT(
            state_summary(&state, &summary, &err) == YVEX_OK &&
                summary.prefix_selected && summary.staged_batch_complete &&
                summary.selected_prefix_count == accepted &&
                summary.staged_next_position == accepted,
            "selected prefix becomes one complete private transaction");
        YVEX_TEST_ASSERT(state.commit(state.context, &failure, &err) == YVEX_OK,
                         "selected prefix publishes atomically");
        view = state_view(&state, 0ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        tail = accepted < layer->sliding_window - 1ull
                   ? accepted : layer->sliding_window - 1ull;
        first = accepted - tail;
        YVEX_TEST_ASSERT(view && view->token_count == accepted &&
                             view->local_tail_count == tail,
                         "prefix promotion publishes only accepted rows");
        for (row = 0ull; row < tail; ++row)
            YVEX_TEST_ASSERT(
                view->local_positions[row] == first + row &&
                    view->local_kv[row * view->local_kv_stride] ==
                        raw[(first + row) * 2ull] &&
                    view->local_kv[row * view->local_kv_stride + 1ull] ==
                        raw[(first + row) * 2ull + 1ull],
                "promoted prefix preserves verified state bytes and positions");
        YVEX_TEST_ASSERT(state_close(&state),
                         "prefix promotion state closes cleanly");
    }
    return 0;
}

static int test_state_prefix_extension(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    const yvex_attention_layer_plan *layer = &fixture->layers[0];
    test_state state = {0};
    yvex_graph_attention_state_summary summary;
    yvex_attention_failure failure;
    yvex_error err;
    const yvex_attention_history_view *view;
    char delta_identity[YVEX_SHA256_HEX_CAP];
    float raw[12] = {0};

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&state, layer,
                          fixture->plan.summary.attention_plan_identity) &&
            state_stage_prefix(&state, layer, 5ull, raw) &&
            state.select_prefix(state.context, 3ull, 1ull, &failure, &err) ==
                YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            summary.prefix_selected && summary.extension_ready &&
            !summary.staged_batch_complete && summary.staged_next_position == 3ull,
        "verified prefix remains private while one target extension is required");
    YVEX_TEST_ASSERT(
        state_begin(&state, layer, 3ull, 1ull, NULL, &failure, &err) == YVEX_OK &&
            state_apply_token(&state, layer, 3ull, 1, delta_identity) &&
            state.commit(state.context, &failure, &err) == YVEX_OK,
        "one target-authored extension joins the selected prefix transaction");
    view = state_view(&state, 0ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
    YVEX_TEST_ASSERT(
        view && view->token_count == 4ull && view->local_tail_count == 3ull &&
            view->local_positions[0] == 1ull &&
            view->local_positions[1] == 2ull &&
            view->local_positions[2] == 3ull &&
            view->local_kv[0] == raw[2] && view->local_kv[2] == raw[4] &&
            view->local_kv[4] == 3.0f / 29.0f,
        "extension publishes the selected verification checkpoint plus one new row");
    YVEX_TEST_ASSERT(state_close(&state),
                     "prefix extension state closes cleanly");
    return 0;
}

static int test_state_identity_is_execution_shape_neutral(
    const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    const yvex_attention_layer_plan *layer = &fixture->layers[0];
    test_state sequential = {0}, verified = {0};
    yvex_graph_attention_state_summary left, right;
    yvex_attention_failure failure;
    yvex_error err;
    char delta[YVEX_SHA256_HEX_CAP];
    float raw[12] = {0};
    unsigned long long position;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&sequential, family, &fixture->plan, 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&sequential, layer,
                          fixture->plan.summary.attention_plan_identity) &&
            state_open(&verified, family, &fixture->plan, 1024ull * 1024ull,
                       &failure, &err) == YVEX_OK &&
            state_prepare(&verified, layer,
                          fixture->plan.summary.attention_plan_identity),
        "shape-neutral state identity fixtures open");
    for (position = 0ull; position < 2ull; ++position)
        YVEX_TEST_ASSERT(state_apply_token(
                             &sequential, layer, position, 0, delta),
                         "sequential token state commits");
    YVEX_TEST_ASSERT(
        state_stage_prefix(&verified, layer, 2ull, raw) &&
            verified.select_prefix(verified.context, 2ull, 0ull, &failure,
                                   &err) == YVEX_OK &&
            verified.commit(verified.context, &failure, &err) == YVEX_OK &&
            state_summary(&sequential, &left, &err) == YVEX_OK &&
            state_summary(&verified, &right, &err) == YVEX_OK &&
            strcmp(left.state_content_identity,
                   right.state_content_identity) == 0,
        "sequential and promoted token prefixes have one state identity");
    YVEX_TEST_ASSERT(state_close(&sequential) && state_close(&verified),
                     "shape-neutral state identity fixtures close");
    return 0;
}

/* A compressed entry is named by its group start even though a later row emits it. */
static int test_candidate_prefix_compression_boundary(void)
{
    yvex_attention_candidate_delta *delta = NULL;
    yvex_attention_history_view committed;
    yvex_attention_publication source, projected;
    yvex_error err;
    float raw[6] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
    float compressed[2] = {4.0f, -4.0f};
    float main_kv[3] = {10.0f, 11.0f, 12.0f};
    float main_score[3] = {20.0f, 21.0f, 22.0f};
    float index_kv[3] = {30.0f, 31.0f, 32.0f};
    float index_score[3] = {40.0f, 41.0f, 42.0f};
    float committed_main_kv = 9.0f, committed_main_score = 19.0f;
    float committed_index_kv = 29.0f, committed_index_score = 39.0f;
    unsigned long long compressed_position = 36ull;
    unsigned int tokens[3] = {1u, 2u, 3u};

    memset(&committed, 0, sizeof(committed));
    committed.token_count = 37ull;
    committed.main_rolling_state.present = 1;
    committed.main_rolling_state.ratio = 4ull;
    committed.main_rolling_state.next_token_position = 37ull;
    committed.main_rolling_state.kv_state_extent = 1ull;
    committed.main_rolling_state.score_state_extent = 1ull;
    committed.main_rolling_state.kv_state = &committed_main_kv;
    committed.main_rolling_state.score_state = &committed_main_score;
    committed.indexer_rolling_state = committed.main_rolling_state;
    committed.indexer_rolling_state.kv_state = &committed_index_kv;
    committed.indexer_rolling_state.score_state = &committed_index_score;

    memset(&source, 0, sizeof(source));
    source.owned = 1;
    source.complete = 1;
    source.prefix_addressable = 1;
    source.layer_index = 1ull;
    source.attention_class = YVEX_ATTENTION_CLASS_CSA;
    source.token_position = 37ull;
    source.token_count = 3ull;
    source.token_ids = tokens;
    source.kv_width = 2ull;
    source.raw_kv = raw;
    source.compressed_count = 1ull;
    source.compressed_stride = 2ull;
    source.compressed_kv = compressed;
    source.compressed_positions = &compressed_position;
    source.indexer_count = 1ull;
    source.indexer_stride = 2ull;
    source.indexer_kv = compressed;
    source.indexer_positions = &compressed_position;
    source.rolling_checkpoint_count = 3ull;
    source.next_main_rolling_state.present = 1;
    source.next_main_rolling_state.kv_state_extent = 1ull;
    source.next_main_rolling_state.score_state_extent = 1ull;
    source.next_indexer_rolling_state.present = 1;
    source.next_indexer_rolling_state.kv_state_extent = 1ull;
    source.next_indexer_rolling_state.score_state_extent = 1ull;
    source.main_rolling_kv_checkpoints = main_kv;
    source.main_rolling_score_checkpoints = main_score;
    source.indexer_rolling_kv_checkpoints = index_kv;
    source.indexer_rolling_score_checkpoints = index_score;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        yvex_attention_candidate_delta_open(&delta, &source, &err) == YVEX_OK,
        "cross-boundary candidate checkpoints are retained");
    YVEX_TEST_ASSERT(
        yvex_attention_candidate_delta_project(
            delta, &committed, 1ull, &projected, &err) == YVEX_OK &&
            projected.compressed_count == 0ull &&
            projected.indexer_count == 0ull &&
            projected.next_main_rolling_state.next_token_position == 38ull,
        "prefix before the completing row excludes the future compressed entry");
    YVEX_TEST_ASSERT(
        yvex_attention_candidate_delta_project(
            delta, &committed, 2ull, &projected, &err) == YVEX_OK &&
            projected.compressed_count == 0ull &&
            projected.indexer_count == 0ull &&
            projected.next_main_rolling_state.next_token_position == 39ull,
        "prefix ending immediately before the boundary still excludes the entry");
    YVEX_TEST_ASSERT(
        yvex_attention_candidate_delta_project(
            delta, &committed, 3ull, &projected, &err) == YVEX_OK &&
            projected.compressed_count == 1ull &&
            projected.indexer_count == 1ull &&
            projected.next_main_rolling_state.next_token_position == 40ull,
        "prefix containing the completing row exposes exactly one compressed entry");
    yvex_attention_candidate_delta_close(&delta);
    return 0;
}

static void state_plan_copy(state_plan_fixture *output,
                            const state_plan_fixture *input)
{
    *output = *input;
    memcpy(output->layers, input->layers, sizeof(output->layers));
    output->plan.layers = output->layers;
}

static int state_identity_sample(
    const state_plan_fixture *fixture, unsigned long long layer_index,
    char layout[YVEX_SHA256_HEX_CAP], char committed[YVEX_SHA256_HEX_CAP],
    char delta_identity[YVEX_SHA256_HEX_CAP])
{
    const yvex_graph_family_api *family = state_family();
    const yvex_attention_layer_plan *layer = &fixture->layers[layer_index];
    test_state state = {0};
    yvex_graph_attention_state_summary summary;
    yvex_attention_failure failure;
    yvex_error err;
    int ok = 0;

    yvex_error_clear(&err);
    if (state_open(&state, family, &fixture->plan, 16ull * 1024ull * 1024ull,
                   &failure, &err) != YVEX_OK ||
        !state_prepare(&state, layer, fixture->plan.summary.attention_plan_identity) ||
        state_summary(&state, &summary, &err) != YVEX_OK ||
        state_identity(&state, layer_index, committed, &err) != YVEX_OK ||
        state_begin(&state, layer, 0ull, 1ull, NULL, &failure, &err) != YVEX_OK ||
        !state_apply_token(&state, layer, 0ull, 1, delta_identity))
        goto done;
    (void)snprintf(layout, YVEX_SHA256_HEX_CAP, "%s", summary.state_layout_identity);
    ok = state.abort(state.context, &failure, &err) == YVEX_OK;
done:
    (void)state_close(&state);
    return ok;
}

/* Prove identical values cannot alias across plan, layout, or rolling-geometry changes. */
static int test_state_identity_geometry(const state_plan_fixture *fixture)
{
    state_plan_fixture plan_changed, geometry_changed;
    char layout[3][YVEX_SHA256_HEX_CAP], committed[3][YVEX_SHA256_HEX_CAP];
    char delta[3][YVEX_SHA256_HEX_CAP];

    state_plan_copy(&plan_changed, fixture);
    (void)snprintf(plan_changed.plan.summary.attention_plan_identity,
                   sizeof(plan_changed.plan.summary.attention_plan_identity),
                   "%064x", 0x6b8fu);
    state_plan_copy(&geometry_changed, fixture);
    geometry_changed.layers[1].compression_ratio = 8ull;
    YVEX_TEST_ASSERT(
        state_identity_sample(fixture, 1ull, layout[0], committed[0], delta[0]) &&
            state_identity_sample(&plan_changed, 1ull, layout[1], committed[1],
                                  delta[1]) &&
            state_identity_sample(&geometry_changed, 1ull, layout[2], committed[2],
                                  delta[2]),
        "state identity fixtures execute one equal-content transition");
    YVEX_TEST_ASSERT(
        strcmp(layout[0], layout[1]) != 0 && strcmp(committed[0], committed[1]) != 0 &&
            strcmp(delta[0], delta[1]) != 0,
        "plan identity changes layout, committed, and delta identities");
    YVEX_TEST_ASSERT(
        strcmp(layout[0], layout[2]) != 0 && strcmp(committed[0], committed[2]) != 0 &&
            strcmp(delta[0], delta[2]) != 0,
        "rolling ratio changes layout, committed, and delta identities");
    return 0;
}

/* Compare one N-token candidate transaction against N committed decode steps. */
static int state_phase_equivalence(const state_plan_fixture *fixture,
                                   unsigned long long layer_index,
                                   unsigned long long token_count)
{
    const yvex_graph_family_api *family = state_family();
    const yvex_attention_layer_plan *layer = &fixture->layers[layer_index];
    test_state chunk = {0}, decode = {0};
    char chunk_delta[YVEX_SHA256_HEX_CAP], decode_delta[YVEX_SHA256_HEX_CAP];
    char chunk_identity[YVEX_SHA256_HEX_CAP];
    char decode_identity[YVEX_SHA256_HEX_CAP];
    yvex_attention_failure failure;
    yvex_error err;
    unsigned long long position;
    int result = 0;

    yvex_error_clear(&err);
    if (state_open(&chunk, family, &fixture->plan, 16ull * 1024ull * 1024ull,
                   &failure, &err) != YVEX_OK ||
        state_open(&decode, family, &fixture->plan, 16ull * 1024ull * 1024ull,
                   &failure, &err) != YVEX_OK ||
        !state_prepare(&chunk, layer, fixture->plan.summary.attention_plan_identity) ||
        !state_prepare(&decode, layer, fixture->plan.summary.attention_plan_identity) ||
        state_begin(&chunk, layer, 0ull, token_count, NULL, &failure, &err) != YVEX_OK)
        goto done;
    for (position = 0ull; position < token_count; ++position) {
        if (!state_apply_token(&chunk, layer, position, 1, chunk_delta) ||
            !state_apply_token(&decode, layer, position, 0, decode_delta))
            goto done;
        if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
            const yvex_attention_history_view *view =
                state_view(&decode, layer_index, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
            if ((position == 27ull && view->compressed_entry_count != 7ull) ||
                (position == 2047ull && view->compressed_entry_count != 512ull))
                goto done;
        }
        if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA) {
            const yvex_attention_history_view *view =
                state_view(&decode, layer_index, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
            if ((position == 126ull && view->compressed_entry_count != 0ull) ||
                (position == 127ull && view->compressed_entry_count != 1ull) ||
                (position == 128ull && view->compressed_entry_count != 1ull))
                goto done;
        }
    }
    if (chunk.commit(chunk.context, &failure, &err) != YVEX_OK ||
        !yvex_sha256_hex_valid(chunk_delta) ||
        !yvex_sha256_hex_valid(decode_delta) ||
        state_identity(&chunk, layer_index, chunk_identity, &err) != YVEX_OK ||
        state_identity(&decode, layer_index, decode_identity, &err) != YVEX_OK ||
        strcmp(chunk_identity, decode_identity) != 0)
        goto done;
    {
        const yvex_attention_history_view *left =
            state_view(&chunk, layer_index, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        const yvex_attention_history_view *right =
            state_view(&decode, layer_index, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
        if (!left || !right || left->token_count != right->token_count ||
            left->local_tail_count != right->local_tail_count ||
            left->compressed_entry_count != right->compressed_entry_count ||
            left->indexer_entry_count != right->indexer_entry_count)
            goto done;
        if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA &&
            (left->local_tail_count != 3ull || left->local_positions[0] != 3ull ||
             left->local_positions[2] != 5ull))
            goto done;
        if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
            (left->compressed_entry_count != 513ull ||
             left->indexer_entry_count != 513ull))
            goto done;
        if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA &&
            left->compressed_entry_count != 3ull)
            goto done;
    }
    result = 1;
done:
    (void)state_close(&chunk);
    (void)state_close(&decode);
    return result;
}

static int test_state_lifecycle(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    test_state state = {0};
    char delta[YVEX_SHA256_HEX_CAP];
    yvex_graph_attention_state_summary transaction_summary;
    yvex_attention_failure failure;
    yvex_attention_cancellation cancellation;
    yvex_error err;
    const yvex_attention_history_view *view;
    char invalid_identity[YVEX_SHA256_HEX_CAP];
    int cancel_requested = 0;

    yvex_error_clear(&err);
    cancellation.requested = state_cancel_requested;
    cancellation.context = &cancel_requested;
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&state, &fixture->layers[0],
                          fixture->plan.summary.attention_plan_identity),
        "persistent attention state opens and prepares one layer");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[0], 0ull, 1ull, &cancellation,
                    &failure, &err) == YVEX_OK &&
            state_apply_token(&state, &fixture->layers[0], 0ull, 1, delta),
        "candidate state accepts one complete publication");
    view = state_view(&state, 0ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
    YVEX_TEST_ASSERT(view && view->token_count == 0ull &&
                         yvex_sha256_hex_valid(delta),
                     "candidate apply leaves committed prior immutable");
    cancel_requested = 1;
    YVEX_TEST_ASSERT(
        state.commit(state.context, &failure, &err) ==
                YVEX_ERR_CANCELLED &&
            state.abort(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &transaction_summary, &err) == YVEX_OK &&
            transaction_summary.cancellation_count == 1ull &&
            state_view(&state, 0ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull,
        "cancellation after staging refuses publication and preserves committed state");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[0], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            setenv("YVEX_TEST_RUNTIME_STATE_ABORT_FAILURE", "1", 1) == 0 &&
            state.abort(state.context, &failure, &err) == YVEX_ERR_STATE &&
            state_summary(&state, &transaction_summary, &err) == YVEX_OK &&
            transaction_summary.transaction_active &&
            state_view(&state, 0ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull,
        "abort synchronization failure retains the candidate and committed prior");
    YVEX_TEST_ASSERT(
        unsetenv("YVEX_TEST_RUNTIME_STATE_ABORT_FAILURE") == 0 &&
            state.abort(state.context, &failure, &err) == YVEX_OK,
        "abort retry discharges the retained candidate transaction");
    {
        yvex_graph_attention_state_summary before, after;

        YVEX_TEST_ASSERT(
            state_summary(&state, &before, &err) == YVEX_OK &&
                state.invalidate(state.context, &err) == YVEX_OK &&
                state_summary(&state, &after, &err) == YVEX_OK &&
                after.invalidated && after.generation == before.generation + 1ull,
            "state invalidation advances one synchronized fail-closed generation");
        YVEX_TEST_ASSERT(
            state_begin(&state, &fixture->layers[0], 0ull, 1ull, NULL,
                        &failure, &err) == YVEX_ERR_STATE,
            "invalidated state refuses new transactions without revival");
        YVEX_TEST_ASSERT(
            state_view(&state, 0ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED) == NULL &&
                state_identity(&state, 0ull, invalid_identity, &err) == YVEX_ERR_STATE &&
                !state_prepare(&state, &fixture->layers[1],
                               fixture->plan.summary.attention_plan_identity) &&
                state_summary(&state, &after, &err) == YVEX_OK && after.invalidated,
            "invalidated state exposes diagnostics but no data-bearing access or preparation");
    }
    YVEX_TEST_ASSERT(state_close(&state) && state_close(&state) && state.context == NULL,
                     "state close is idempotent through provider ownership");
    return 0;
}

static int test_state_reset(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    test_state state = {0};
    char delta[YVEX_SHA256_HEX_CAP];
    yvex_graph_attention_state_summary before, after;
    yvex_attention_failure failure;
    yvex_error err;
    char empty_identity[YVEX_SHA256_HEX_CAP], populated_identity[YVEX_SHA256_HEX_CAP];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 16ull * 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&state, &fixture->layers[1],
                          fixture->plan.summary.attention_plan_identity) &&
            state_identity(&state, 1ull, empty_identity, &err) == YVEX_OK &&
            state_summary(&state, &before, &err) == YVEX_OK &&
            before.persistent && before.position_consistent &&
            before.capacity == 2052ull && !before.committed_sequence_length &&
            !before.next_position && yvex_sha256_hex_valid(before.state_content_identity),
        "reset fixture owns one allocation-stable empty CSA bank pair");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[1], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            state.reset(state.context, &failure, &err) == YVEX_ERR_STATE &&
            state.abort(state.context, &failure, &err) == YVEX_OK,
        "reset refuses an active candidate and preserves explicit rollback ownership");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[1], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            state_apply_token(&state, &fixture->layers[1], 0ull, 1, delta) &&
            state.commit(state.context, &failure, &err) == YVEX_OK &&
            state_identity(&state, 1ull, populated_identity, &err) == YVEX_OK &&
            state_summary(&state, &before, &err) == YVEX_OK &&
            before.committed_sequence_length == 1ull && before.next_position == 1ull &&
            strcmp(empty_identity, populated_identity) != 0,
        "reset fixture first commits non-empty history");
    YVEX_TEST_ASSERT(
        state.reset(state.context, &failure, &err) == YVEX_OK,
        "idle prepared state resets without allocation or refusal");
    YVEX_TEST_ASSERT(
        state_summary(&state, &after, &err) == YVEX_OK &&
            after.allocated_bytes == before.allocated_bytes &&
            after.prepared_layer_count == before.prepared_layer_count &&
            after.reset_count == before.reset_count + 1ull &&
            after.generation == before.generation + 1ull &&
            !after.committed_sequence_length && !after.next_position &&
            after.position_consistent && yvex_sha256_hex_valid(after.state_content_identity),
        "reset advances lifecycle evidence while preserving prepared allocation");
    YVEX_TEST_ASSERT(
        state_view(&state, 1ull,
                   YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull &&
            state_identity(&state, 1ull, populated_identity, &err) == YVEX_OK &&
            strcmp(empty_identity, populated_identity) == 0,
        "reset restores the canonical empty state identity");
    (void)state_close(&state);
    return 0;
}

/* Prove a failed later-layer preparation cannot publish or corrupt prior state ownership. */
static int test_prepare_failure_is_atomic(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    test_state state = {0};
    yvex_attention_state_recipe_request request;
    yvex_attention_state_recipe recipe;
    yvex_graph_attention_state_summary before, after;
    yvex_attention_failure failure;
    yvex_error err;
    char delta[YVEX_SHA256_HEX_CAP];
    char identity_before[YVEX_SHA256_HEX_CAP], identity_after[YVEX_SHA256_HEX_CAP];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 4096ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&state, &fixture->layers[0],
                          fixture->plan.summary.attention_plan_identity) &&
            state_summary(&state, &before, &err) == YVEX_OK &&
            state_identity(&state, 0ull, identity_before, &err) == YVEX_OK,
        "first prepared layer establishes committed state before injected budget failure");
    memset(&request, 0, sizeof(request));
    request.layer_ordinal = 1ull;
    request.final_position = 2052ull;
    request.attention_plan_identity = fixture->plan.summary.attention_plan_identity;
    YVEX_TEST_ASSERT(
        state_recipe_project(&fixture->layers[1], &request, &recipe,
                             &failure, &err) == YVEX_OK &&
            state.prepare(state.context, 1ull, &recipe, NULL,
                          &failure, &err) == YVEX_ERR_BOUNDS &&
            state_summary(&state, &after, &err) == YVEX_OK &&
            before.prepared_layer_count == after.prepared_layer_count &&
            before.allocated_bytes == after.allocated_bytes &&
            memcmp(before.components, after.components,
                   sizeof(before.components)) == 0 &&
            before.generation == after.generation &&
            strcmp(before.state_layout_identity, after.state_layout_identity) == 0 &&
            state_view(&state, 1ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED) == NULL,
        "failed preparation publishes no layer, counters, or layout identity");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[0], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            state_apply_token(&state, &fixture->layers[0], 0ull, 1, delta) &&
            state.commit(state.context, &failure, &err) == YVEX_OK &&
            state_identity(&state, 0ull, identity_after, &err) == YVEX_OK &&
            strcmp(identity_before, identity_after) != 0 &&
            yvex_sha256_hex_valid(delta),
        "prior prepared bank remains owned and executable after later preparation failure");
    (void)state_close(&state);
    return 0;
}

/* Prove a multi-layer batch flips no selector until one atomic publish. */
static int test_batch_publication_is_atomic(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    test_state state = {0};
    char delta[YVEX_SHA256_HEX_CAP];
    yvex_graph_attention_state_summary summary;
    yvex_attention_failure failure;
    yvex_error err;
    char committed_identity[YVEX_SHA256_HEX_CAP] = {0};
    char staged_identity[YVEX_SHA256_HEX_CAP] = {0};
    unsigned long long committed_generation = 0ull, committed_next_position = 0ull;
    unsigned long long staged_generation = 0ull, staged_next_position = 0ull;
    unsigned long long layer;

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 16ull * 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_prepare(&state, &fixture->layers[0],
                          fixture->plan.summary.attention_plan_identity) &&
            state_prepare(&state, &fixture->layers[1],
                          fixture->plan.summary.attention_plan_identity),
        "two-layer atomic state fixture opens and prepares");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[0], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            state_apply_token(&state, &fixture->layers[0], 0ull, 1, delta),
        "first layer stages without publishing its alternate bank");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[1], 1ull, 1ull, NULL,
                    &failure, &err) == YVEX_ERR_STATE &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            summary.transaction_active && summary.abort_required &&
            !summary.candidate_active && summary.staged_layer_count == 1ull &&
            summary.commit_count == 0ull &&
            state.commit(state.context, &failure, &err) == YVEX_ERR_STATE &&
            state_view(&state, 0ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull &&
            state_view(&state, 1ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull,
        "failure after staging poisons the batch and cannot publish a subset");
    YVEX_TEST_ASSERT(
        state.abort(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            !summary.transaction_active && !summary.abort_required &&
            !summary.staged_batch_complete && !summary.staged_generation &&
            !summary.staged_next_position &&
            !summary.staged_state_content_identity[0] &&
            summary.abort_count == 1ull && summary.commit_count == 0ull,
        "abort discharges the poisoned batch without changing committed state");
    for (layer = 0ull; layer < 2ull; ++layer) {
        YVEX_TEST_ASSERT(
            state_begin(&state, &fixture->layers[layer], 0ull, 1ull, NULL,
                        &failure, &err) == YVEX_OK &&
                state_apply_token(&state, &fixture->layers[layer], 0ull, 1, delta),
            "retry batch stages each complete layer");
    }
    YVEX_TEST_ASSERT(
        setenv("YVEX_TEST_RUNTIME_STATE_PUBLISH_FAILURE", "1", 1) == 0 &&
            state.commit(state.context, &failure, &err) == YVEX_ERR_STATE &&
            unsetenv("YVEX_TEST_RUNTIME_STATE_PUBLISH_FAILURE") == 0 &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            summary.abort_required && summary.staged_layer_count == 2ull &&
            summary.commit_count == 0ull &&
            state.commit(state.context, &failure, &err) == YVEX_ERR_STATE &&
            state_view(&state, 0ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull &&
            state_view(&state, 1ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 0ull,
        "publication failure also requires abort and preserves every prior selector");
    YVEX_TEST_ASSERT(
        state.abort(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            !summary.transaction_active && !summary.abort_required &&
            !summary.staged_batch_complete && !summary.staged_generation &&
            !summary.staged_next_position &&
            !summary.staged_state_content_identity[0] &&
            summary.abort_count == 2ull && summary.commit_count == 0ull,
        "second poisoned batch aborts before a clean retry");
    committed_generation = summary.generation;
    committed_next_position = summary.next_position;
    memcpy(committed_identity, summary.state_content_identity,
           sizeof(committed_identity));
    for (layer = 0ull; layer < 2ull; ++layer) {
        YVEX_TEST_ASSERT(
            state_begin(&state, &fixture->layers[layer], 0ull, 1ull, NULL,
                        &failure, &err) == YVEX_OK &&
                state_apply_token(&state, &fixture->layers[layer], 0ull, 1, delta),
            "clean batch restages every layer after explicit abort");
    }
    YVEX_TEST_ASSERT(
        state_summary(&state, &summary, &err) == YVEX_OK &&
            summary.transaction_active && !summary.candidate_active &&
            !summary.abort_required && summary.staged_layer_count == 2ull &&
            summary.staged_batch_complete &&
            summary.next_position == committed_next_position &&
            summary.generation == committed_generation &&
            strcmp(summary.state_content_identity, committed_identity) == 0 &&
            summary.staged_next_position == committed_next_position + 1ull &&
            summary.staged_generation == committed_generation + 1ull &&
            yvex_sha256_hex_valid(summary.staged_state_content_identity) &&
            strcmp(summary.staged_state_content_identity,
                   summary.state_content_identity) != 0,
        "complete private batch exposes exact candidate facts without publication");
    staged_generation = summary.staged_generation;
    staged_next_position = summary.staged_next_position;
    memcpy(staged_identity, summary.staged_state_content_identity,
           sizeof(staged_identity));
    YVEX_TEST_ASSERT(
        state.commit(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            !summary.transaction_active && summary.staged_layer_count == 0ull &&
            !summary.staged_batch_complete && !summary.staged_generation &&
            !summary.staged_next_position &&
            !summary.staged_state_content_identity[0] &&
            summary.commit_count == 2ull &&
            summary.persistent && summary.position_consistent &&
            summary.committed_sequence_length == 1ull &&
            summary.next_position == staged_next_position &&
            summary.generation == staged_generation &&
            strcmp(summary.state_content_identity, staged_identity) == 0 &&
            yvex_sha256_hex_valid(summary.state_content_identity) &&
            state_view(&state, 0ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 1ull &&
            state_view(&state, 1ull,
                       YVEX_ATTENTION_STATE_VIEW_COMMITTED)->token_count == 1ull,
        "successful publication flips the complete staged set together");
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[0], 2ull, 1ull, NULL,
                    &failure, &err) == YVEX_ERR_STATE &&
            state_begin(&state, &fixture->layers[0], 1ull, 6ull, NULL,
                        &failure, &err) == YVEX_ERR_BOUNDS &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            !summary.transaction_active && summary.next_position == 1ull,
        "non-contiguous and over-capacity appends refuse before persistent-state mutation");
    (void)state_close(&state);
    return 0;
}

/* Prove aggregate capacity accounting remains distinct from one-layer capture maxima. */
static int test_summary_capacity_accounting(const state_plan_fixture *fixture)
{
    const yvex_graph_family_api *family = state_family();
    test_state state = {0};
    yvex_graph_attention_state_summary summary;
    yvex_graph_attention_state_summary initial;
    yvex_attention_failure failure;
    yvex_error err;
    char initial_identity[YVEX_SHA256_HEX_CAP];

    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&state, family, &fixture->plan, 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK,
        "capacity-accounting fixture opens");
    YVEX_TEST_ASSERT(
        state_summary(&state, &initial, &err) == YVEX_OK &&
            snprintf(initial_identity, sizeof(initial_identity), "%s",
                     initial.state_layout_identity) > 0 &&
            state_prepare(&state, &fixture->layers[0],
                          fixture->plan.summary.attention_plan_identity) &&
            state_prepare(&state, &fixture->layers[1],
                          fixture->plan.summary.attention_plan_identity) &&
            state_prepare(&state, &fixture->layers[2],
                          fixture->plan.summary.attention_plan_identity) &&
            state_summary(&state, &summary, &err) == YVEX_OK,
        "all class capacities prepare and summarize atomically");
    YVEX_TEST_ASSERT(
        summary.components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].capacity == 133ull &&
            summary.components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].capacity == 516ull &&
            summary.components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].capacity == 513ull &&
            summary.components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].maximum_capacity == 127ull &&
            summary.components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].maximum_capacity == 513ull &&
            summary.components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].maximum_capacity == 513ull,
        "summary separates aggregate accounting from maximum per-layer capacities");
    YVEX_TEST_ASSERT(
        strcmp(initial_identity, summary.state_layout_identity) != 0,
        "synchronized summary exposes the bound aggregate layout");
    (void)state_close(&state);
    return 0;
}

static void execution_descriptor_fixture(
    yvex_runtime_execution_descriptor_facts *facts)
{
    memset(facts, 0, sizeof(*facts));
    facts->schema_version = YVEX_RUNTIME_EXECUTION_DESCRIPTOR_SCHEMA_V2;
    facts->runtime_model_identity =
        "1111111111111111111111111111111111111111111111111111111111111111";
    facts->runtime_binding_identity =
        "2222222222222222222222222222222222222222222222222222222222222222";
    facts->artifact_identity =
        "3333333333333333333333333333333333333333333333333333333333333333";
    facts->runtime_numeric_identity =
        "4444444444444444444444444444444444444444444444444444444444444444";
    facts->runtime_descriptor_identity =
        "5555555555555555555555555555555555555555555555555555555555555555";
    facts->semantic_graph_identity =
        "6666666666666666666666666666666666666666666666666666666666666666";
    facts->executable_graph_identity =
        "7777777777777777777777777777777777777777777777777777777777777777";
    facts->residency_identity =
        "8888888888888888888888888888888888888888888888888888888888888888";
    facts->workspace_identity =
        "9999999999999999999999999999999999999999999999999999999999999999";
    facts->capacity_plan_identity =
        "abababababababababababababababababababababababababababababababab";
    facts->state_layout_identity =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    facts->selected_mode = "eager";
    facts->capture_bucket = "not-applicable";
    facts->family_adapter_id = 0x44535634ull;
    facts->family_adapter_version = 1ull;
    facts->probe = YVEX_ATTENTION_PROBE_CANONICAL_V2;
    facts->probe_scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
    facts->operation_scope = YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE;
    facts->phase = YVEX_EXECUTION_PHASE_PREFILL;
    facts->backend = YVEX_BACKEND_KIND_CPU;
    facts->requested_mode = YVEX_RUNTIME_MODE_EAGER;
    facts->token_count = 4ull;
    facts->request_count = 1ull;
    facts->layer_start = 0ull;
    facts->layer_count = 43ull;
    facts->selection_key = ~0ull;
    facts->binding_count = 634ull;
    facts->state_component_entries[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY] = 64ull;
    facts->state_component_entries[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY] = 8ull;
    facts->state_component_entries[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY] = 4ull;
    facts->state_component_capacities[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY] = 129ull;
    facts->state_component_capacities[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY] = 16ull;
    facts->state_component_capacities[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY] = 8ull;
    facts->maximum_compression_ratio = 128ull;
    facts->maximum_topk_capacity = 512ull;
    facts->maximum_host_bytes = 8ull * 1024ull * 1024ull;
    facts->residency_generation = 1ull;
    facts->resident_binding_count = 634ull;
    facts->resident_encoded_bytes = 5693189120ull;
    facts->workspace_bytes = 1024ull * 1024ull;
    facts->workspace_generation = 1ull;
    facts->prepared_state_layers = 3ull;
    facts->state_allocated_bytes = 65536ull;
    facts->state_generation = 4ull;
    facts->qtype_binding_counts[0] = 105ull;
    facts->qtype_binding_counts[1] = 236ull;
    facts->qtype_bytes[0] = 4096ull;
    facts->qtype_bytes[1] = 8192ull;
    facts->device_kind = YVEX_BACKEND_KIND_CPU;
    facts->device_index = 0;
    facts->total_device_bytes = 16ull * 1024ull * 1024ull * 1024ull;
}

static int execution_descriptor_changed(
    const char *baseline, const yvex_runtime_execution_descriptor_facts *facts)
{
    char identity[YVEX_SHA256_HEX_CAP];
    yvex_error err;

    yvex_error_clear(&err);
    return yvex_runtime_execution_descriptor_identity_compute(
               facts, identity, &err) == YVEX_OK &&
           strcmp(baseline, identity) != 0;
}

/* Prove descriptor identity covers compatibility and excludes orchestration evidence. */
static int test_execution_descriptor_identity(void)
{
    yvex_runtime_execution_descriptor_facts facts, changed;
    yvex_graph_attention_operator_request orchestration, unrelated;
    char first[YVEX_SHA256_HEX_CAP], second[YVEX_SHA256_HEX_CAP];
    double timing = 1.0, changed_timing = 999.0;
    yvex_error err;

    execution_descriptor_fixture(&facts);
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(yvex_runtime_execution_descriptor_identity_compute(
                         &facts, first, &err) == YVEX_OK,
                     "execution descriptor fixture seals");
    YVEX_TEST_ASSERT(strcmp(first,
                            "1b1461b1dbcf06f7ae22786d8eba6c052356b115cbb0b8be532e503982922997") == 0,
                     "execution descriptor canonical field order is stable");
    changed = facts;
    changed.runtime_binding_identity =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "binding mutation changes execution descriptor");
    changed = facts;
    changed.runtime_model_identity =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "model mutation changes execution descriptor");
    changed = facts;
    changed.probe = YVEX_ATTENTION_PROBE_UNSPECIFIED;
    YVEX_TEST_ASSERT(
        yvex_runtime_execution_descriptor_identity_compute(
            &changed, second, &err) == YVEX_ERR_INVALID_ARG,
        "legacy numeric zero probe refuses descriptor admission");
    changed.probe = (yvex_attention_probe_kind)(YVEX_ATTENTION_PROBE_CANONICAL_V2 + 1u);
    YVEX_TEST_ASSERT(
        yvex_runtime_execution_descriptor_identity_compute(
            &changed, second, &err) == YVEX_ERR_INVALID_ARG &&
            strcmp(facts.runtime_model_identity, changed.runtime_model_identity) == 0 &&
            strcmp(facts.runtime_binding_identity, changed.runtime_binding_identity) == 0 &&
            strcmp(facts.runtime_descriptor_identity,
                   changed.runtime_descriptor_identity) == 0 &&
            strcmp(facts.semantic_graph_identity, changed.semantic_graph_identity) == 0 &&
            strcmp(facts.executable_graph_identity,
                   changed.executable_graph_identity) == 0,
        "unknown probe refuses without changing upstream identity facts");
    changed = facts;
    changed.phase = YVEX_EXECUTION_PHASE_DECODE;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "phase mutation changes execution descriptor");
    changed = facts;
    changed.requested_mode = YVEX_RUNTIME_MODE_AUTO;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "requested-mode mutation changes execution descriptor");
    changed = facts;
    changed.selected_mode = "full";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "selected-mode mutation changes execution descriptor");
    changed = facts;
    changed.capture_bucket = "prefill-4";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "capture-bucket mutation changes execution descriptor");
    changed = facts;
    changed.token_count++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "request geometry mutation changes execution descriptor");
    changed = facts;
    changed.layer_start++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "layer-range mutation changes execution descriptor");
    changed = facts;
    changed.selection_key--;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "family selection-key mutation changes execution descriptor");
    changed = facts;
    changed.state_component_entries[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "local-history mutation changes execution descriptor");
    changed = facts;
    changed.state_component_entries[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "compressed-history mutation changes execution descriptor");
    changed = facts;
    changed.state_component_entries[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "indexer-history mutation changes execution descriptor");
    changed = facts;
    changed.state_component_capacities[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "local-capacity mutation changes execution descriptor");
    changed = facts;
    changed.state_component_capacities[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "compressed-capacity mutation changes execution descriptor");
    changed = facts;
    changed.state_component_capacities[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "indexer-capacity mutation changes execution descriptor");
    changed = facts;
    changed.maximum_compression_ratio++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "compression-ratio mutation changes execution descriptor");
    changed = facts;
    changed.maximum_topk_capacity++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "top-k capacity mutation changes execution descriptor");
    changed = facts;
    changed.residency_identity =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "residency mutation changes execution descriptor");
    changed = facts;
    changed.residency_generation++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "residency-generation mutation changes execution descriptor");
    changed = facts;
    changed.workspace_identity =
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "workspace mutation changes execution descriptor");
    changed = facts;
    changed.workspace_generation++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "workspace-generation mutation changes execution descriptor");
    changed = facts;
    changed.capacity_plan_identity =
        "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "capacity-plan mutation changes execution descriptor");
    changed = facts;
    changed.state_layout_identity =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "state-layout mutation changes execution descriptor");
    changed = facts;
    changed.state_generation++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "state-generation mutation changes execution descriptor");
    changed = facts;
    changed.device_index++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "device mutation changes execution descriptor");
    changed = facts;
    changed.qtype_binding_counts[1]++;
    YVEX_TEST_ASSERT(execution_descriptor_changed(first, &changed),
                     "qtype requirement mutation changes execution descriptor");
    memset(&orchestration, 0, sizeof(orchestration));
    orchestration.operator_action = YVEX_RUNTIME_OPERATOR_EXECUTE;
    orchestration.repeat = 1ull;
    unrelated = orchestration;
    unrelated.operator_action = YVEX_RUNTIME_OPERATOR_BENCHMARK;
    unrelated.repeat = 99ull;
    unrelated.warmup = 7ull;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        yvex_runtime_execution_descriptor_identity_compute(
            &facts, second, &err) == YVEX_OK && strcmp(first, second) == 0 &&
            (orchestration.operator_action != unrelated.operator_action) &&
            orchestration.repeat != unrelated.repeat &&
            orchestration.warmup != unrelated.warmup && timing != changed_timing,
        "action, repeat, warmup, and timing remain outside descriptor identity");
    changed = facts;
    changed.schema_version++;
    YVEX_TEST_ASSERT(yvex_runtime_execution_descriptor_identity_compute(
                         &changed, second, &err) == YVEX_ERR_INVALID_ARG,
                     "unsupported execution descriptor schema refuses");
    return 0;
}

static int test_operator_missing_binding_refusal(void)
{
    yvex_graph_attention_operator_request request;
    yvex_graph_attention_operator_result result;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_error err;
    int rc;

    memset(&request, 0, sizeof(request));
    request.target = "deepseek4-v4-flash-dspark";
    request.artifact_path = "/tmp/yvex-definitely-missing.gguf";
    request.runtime_binding_path = "/tmp/yvex-definitely-missing.runtime-binding";
    request.backend = YVEX_BACKEND_KIND_CPU;
    request.probe = YVEX_ATTENTION_PROBE_UNSPECIFIED;
    request.scope = YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    request.phase = YVEX_EXECUTION_PHASE_PREFILL;
    request.mode = YVEX_RUNTIME_MODE_EAGER;
    request.operation_scope = YVEX_RUNTIME_SCOPE_ATTENTION_CORE;
    request.operator_action = YVEX_RUNTIME_OPERATOR_STATE_EXERCISE;
    request.token_count = 2ull;
    request.repeat = 1ull;
    request.history_tokens = 4ull;
    request.selection_key = 2ull;
    request.select_selection_key = 1;
    yvex_error_clear(&err);
    rc = yvex_graph_attention_operator_execute(&request, &result, &cleanup, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_ERR_INVALID_ARG && cleanup == NULL && !result.completed &&
            strcmp(result.status, "refused") == 0,
        "legacy numeric zero probe refuses before runtime binding access");
    request.probe = YVEX_ATTENTION_PROBE_CANONICAL_V2;
    yvex_error_clear(&err);
    rc = yvex_graph_attention_operator_execute(&request, &result, &cleanup, &err);
    YVEX_TEST_ASSERT(rc != YVEX_OK && cleanup == NULL && !result.completed &&
                         strcmp(result.status, "refused") == 0 && yvex_error_is_set(&err),
                     "missing runtime binding refuses selected state exercise without mutation");
    return 0;
}

static void state_page_capacity_open(yvex_execution_capacity_plan *capacity)
{
    unsigned long long index;

    memset(capacity, 0, sizeof(*capacity));
    capacity->schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    capacity->per_session_maximum = 524288ull;
    capacity->state_pool_bytes = 64ull * 1024ull;
    capacity->state_class_count = 6ull;
    (void)snprintf(capacity->model_execution_identity, YVEX_SHA256_HEX_CAP,
                   "%064x", 0x101u);
    (void)snprintf(capacity->hardware_profile_identity, YVEX_SHA256_HEX_CAP,
                   "%064x", 0x102u);
    (void)snprintf(capacity->workload_profile_identity, YVEX_SHA256_HEX_CAP,
                   "%064x", 0x103u);
    (void)snprintf(capacity->identity, YVEX_SHA256_HEX_CAP, "%064x", 0x104u);
    for (index = 0ull; index < capacity->state_class_count; ++index) {
        yvex_execution_state_class_plan *state = &capacity->state_classes[index];
        state->state_class = (yvex_model_state_class)index;
        state->logical_block_tokens =
            index == YVEX_MODEL_STATE_COMPRESSED_HISTORY ||
                    index == YVEX_MODEL_STATE_HCA_HISTORY ||
                    index == YVEX_MODEL_STATE_INDEXER_HISTORY
                ? 128ull
                : 1ull;
        state->bytes_per_block = 4096ull;
        state->page_tokens = index == YVEX_MODEL_STATE_COMPRESSED_HISTORY
                                 ? 128ull
                             : index == YVEX_MODEL_STATE_HCA_HISTORY
                                 ? 128ull
                             : index == YVEX_MODEL_STATE_INDEXER_HISTORY
                                 ? 256ull : 16ull;
        state->page_bytes = state->page_tokens /
                            state->logical_block_tokens *
                            state->bytes_per_block;
    }
}

static int test_state_checkpoint_restore(const state_plan_fixture *fixture)
{
    state_plan_fixture single = *fixture;
    yvex_attention_history_view layers[1];
    char identities[1][YVEX_SHA256_HEX_CAP];
    yvex_attention_state_checkpoint checkpoint = {0};
    yvex_graph_attention_state_summary before, after, empty;
    yvex_attention_failure failure;
    yvex_error err;
    test_state source = {0}, restored = {0};
    const yvex_attention_history_view *view;
    char delta[YVEX_SHA256_HEX_CAP];
    unsigned long long position;
    int restore_rc;

    single.plan.layer_count = single.plan.summary.layer_count = 1ull;
    single.plan.summary.csa_layer_count = single.plan.summary.hca_layer_count = 0ull;
    yvex_error_clear(&err);
    YVEX_TEST_ASSERT(
        state_open(&source, state_family(), &single.plan, 1024ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state_open(&restored, state_family(), &single.plan,
                       1024ull * 1024ull, &failure, &err) == YVEX_OK &&
            state_prepare(&source, &single.layers[0],
                          single.plan.summary.attention_plan_identity) &&
            state_prepare(&restored, &single.layers[0],
                          single.plan.summary.attention_plan_identity),
        "checkpoint providers open with identical sealed geometry");
    for (position = 0ull; position < 3ull; ++position)
        YVEX_TEST_ASSERT(
            state_apply_token(&source, &single.layers[0], position, 0, delta),
            "checkpoint source commits an exact prefix");
    view = state_view(&source, 0ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
    YVEX_TEST_ASSERT(
        view && state_summary(&source, &before, &err) == YVEX_OK &&
            state_identity(&source, 0ull, identities[0], &err) == YVEX_OK &&
            state_summary(&restored, &empty, &err) == YVEX_OK,
        "checkpoint captures committed state and identities");
    layers[0] = *view;
    checkpoint.schema_version = YVEX_ATTENTION_STATE_CHECKPOINT_SCHEMA_V1;
    checkpoint.layer_count = 1ull;
    checkpoint.committed_sequence_length = before.committed_sequence_length;
    checkpoint.layers = layers;
    checkpoint.layer_identities =
        (const char (*)[YVEX_SHA256_HEX_CAP])identities;
    strcpy(checkpoint.state_layout_identity, before.state_layout_identity);
    strcpy(checkpoint.state_content_identity, before.state_content_identity);
    strcpy(checkpoint.capacity_plan_identity, before.capacity_plan_identity);
    restore_rc = restored.restore
                     ? restored.restore(restored.context, &checkpoint, &failure,
                                        &err)
                     : YVEX_ERR_STATE;
    YVEX_TEST_ASSERT(
        restored.schema_version == YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V6 &&
            restored.restore && restore_rc == YVEX_OK &&
            state_summary(&restored, &after, &err) == YVEX_OK &&
            after.committed_sequence_length == 3ull &&
            after.next_position == 3ull &&
            strcmp(after.state_content_identity,
                   before.state_content_identity) == 0 &&
            state_identity(&restored, 0ull, delta, &err) == YVEX_OK &&
            strcmp(delta, identities[0]) == 0,
        "checkpoint restore publishes the exact committed prefix atomically");
    identities[0][0] = identities[0][0] == '0' ? '1' : '0';
    YVEX_TEST_ASSERT(
        restored.reset(restored.context, &failure, &err) == YVEX_OK &&
            restored.restore(restored.context, &checkpoint, &failure, &err) ==
                YVEX_ERR_FORMAT &&
            state_summary(&restored, &after, &err) == YVEX_OK &&
            after.committed_sequence_length == 0ull &&
            strcmp(after.state_content_identity, empty.state_content_identity) == 0,
        "corrupt checkpoint identity refuses without publishing partial state");
    YVEX_TEST_ASSERT(state_close(&source) && state_close(&restored),
                     "checkpoint providers close cleanly");
    return 0;
}

static int test_state_pages(const state_plan_fixture *fixture)
{
    yvex_execution_capacity_plan capacity, changed;
    yvex_attention_state_recipe_request request = {0};
    yvex_attention_state_recipe recipe, initial_recipe;
    yvex_graph_attention_state_summary summary, touched;
    yvex_attention_failure failure;
    test_state state = {0}, initial = {0}, reference = {0}, limited = {0};
    yvex_error err;
    const float *stable;
    char delta[YVEX_SHA256_HEX_CAP];

    yvex_error_clear(&err);
    state_page_capacity_open(&capacity);
    request.layer_ordinal = 1ull;
    request.final_position = 1ull;
    request.attention_plan_identity = fixture->plan.summary.attention_plan_identity;
    YVEX_TEST_ASSERT(
        state_recipe_project(&fixture->layers[1], &request, &initial_recipe,
                             &failure, &err) == YVEX_OK &&
            initial_recipe.components[1].capacity == 0ull &&
            state_open(&reference, state_family(), &fixture->plan,
                       512ull * 1024ull, &failure, &err) == YVEX_OK &&
            reference.prepare(reference.context, 1ull, &initial_recipe, NULL,
                              &failure, &err) == YVEX_OK &&
            reference.reset(reference.context, &failure, &err) == YVEX_OK &&
            state_close(&reference) &&
            state_open(&initial, state_family(), &fixture->plan, 512ull * 1024ull,
                       &failure, &err) == YVEX_OK &&
            initial.configure_pages(initial.context, &capacity, &failure, &err) == YVEX_OK &&
            initial.prepare(initial.context, 1ull, &initial_recipe, NULL,
                            &failure, &err) == YVEX_OK &&
            initial.reset(initial.context, &failure, &err) == YVEX_OK &&
            state_close(&initial),
        "paged and reference periodic state remain valid before the first history row");
    memset(&request, 0, sizeof(request));
    request.layer_ordinal = 1ull;
    request.final_position = capacity.per_session_maximum;
    request.attention_plan_identity = fixture->plan.summary.attention_plan_identity;
    YVEX_TEST_ASSERT(
        state_open(&state, state_family(), &fixture->plan, 512ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            state.configure_pages(state.context, &capacity, &failure, &err) == YVEX_OK &&
            state_recipe_project(&fixture->layers[1], &request, &recipe,
                                 &failure, &err) == YVEX_OK &&
            state.prepare(state.context, 1ull, &recipe, NULL, &failure, &err) == YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK && summary.paged &&
            summary.paging_configured && summary.virtual_bytes > summary.resident_bytes &&
            strcmp(summary.capacity_plan_identity, capacity.identity) == 0,
        "512K logical state prepares through class pages without full physical allocation");
    stable = state_view(&state, 1ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED)->local_kv;
    YVEX_TEST_ASSERT(
        state_begin(&state, &fixture->layers[1], 0ull, 1ull, NULL,
                    &failure, &err) == YVEX_OK &&
            state_apply_token(&state, &fixture->layers[1], 0ull, 1, delta) &&
            state.abort(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &touched, &err) == YVEX_OK &&
            touched.next_position == 0ull,
        "first candidate abort does not read an uncommitted history page");
    YVEX_TEST_ASSERT(
        state_apply_token(&state, &fixture->layers[1], 0ull, 0, delta) &&
            state_summary(&state, &touched, &err) == YVEX_OK &&
            touched.resident_bytes > summary.resident_bytes,
        "first state publication commits only its reached semantic pages");
    changed = capacity;
    (void)snprintf(changed.identity, sizeof(changed.identity), "%064x", 0x105u);
    YVEX_TEST_ASSERT(
        state.configure_pages(state.context, &changed, &failure, &err) == YVEX_ERR_STATE &&
            state.reset(state.context, &failure, &err) == YVEX_OK &&
            state_summary(&state, &summary, &err) == YVEX_OK &&
            summary.page_release_count > 0ull &&
            state_view(&state, 1ull, YVEX_ATTENTION_STATE_VIEW_COMMITTED)->local_kv == stable &&
            state_close(&state),
        "reset releases physical pages without relocating the admitted state span");
    capacity.state_pool_bytes = 4096ull;
    YVEX_TEST_ASSERT(
        state_open(&limited, state_family(), &fixture->plan, 512ull * 1024ull,
                   &failure, &err) == YVEX_OK &&
            limited.configure_pages(limited.context, &capacity, &failure, &err) == YVEX_OK &&
            limited.prepare(limited.context, 1ull, &recipe, NULL,
                            &failure, &err) == YVEX_ERR_BOUNDS &&
            state_summary(&limited, &summary, &err) == YVEX_OK &&
            summary.prepared_layer_count == 0ull && state_close(&limited),
        "state-pool budget refuses deep storage before a layer becomes visible");
    return 0;
}

int yvex_test_runtime_state(void)
{
    state_plan_fixture fixture;

    state_plan_open(&fixture);
    if (test_state_recipe_identity(&fixture) != 0) return 1;
    if (test_workspace_recipe_identity() != 0) return 1;
    if (test_workspace_capture_geometry(&fixture) != 0) return 1;
    if (test_capacity_plan(&fixture) != 0) return 1;
    if (test_execution_descriptor_identity() != 0) return 1;
    if (test_operator_missing_binding_refusal() != 0) return 1;
    if (test_state_pages(&fixture) != 0) return 1;
    if (test_state_identity_geometry(&fixture) != 0) return 1;
    if (test_state_lifecycle(&fixture) != 0) return 1;
    if (test_state_prefix_promotion(&fixture) != 0) return 1;
    if (test_state_prefix_extension(&fixture) != 0) return 1;
    if (test_state_identity_is_execution_shape_neutral(&fixture) != 0) return 1;
    if (test_candidate_prefix_compression_boundary() != 0) return 1;
    if (test_state_reset(&fixture) != 0) return 1;
    if (test_state_checkpoint_restore(&fixture) != 0) return 1;
    if (test_summary_capacity_accounting(&fixture) != 0) return 1;
    if (test_prepare_failure_is_atomic(&fixture) != 0) return 1;
    if (test_batch_publication_is_atomic(&fixture) != 0) return 1;
    YVEX_TEST_ASSERT(state_phase_equivalence(&fixture, 0ull, 6ull),
                     "SWA chunk and ordered decode preserve rollover state exactly");
    YVEX_TEST_ASSERT(state_phase_equivalence(&fixture, 1ull, 2052ull),
                     "CSA chunk and decode preserve fewer/exact/513 candidate state exactly");
    YVEX_TEST_ASSERT(state_phase_equivalence(&fixture, 2ull, 384ull),
                     "HCA chunk and decode preserve 127/128/129 and three-group state exactly");
    return 0;
}
