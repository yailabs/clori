/* Seal the canonical operator topology before physical compilation or runtime consumption. */
#include <yvex/internal/operator_graph.h>

#include <yvex/internal/graph.h>

#include <stdlib.h>
#include <string.h>

#define OPERATOR_GRAPH_MAX_NODES 1048576ull
#define OPERATOR_GRAPH_MAX_EDGES 4194304ull

struct yvex_operator_graph_ir {
    yvex_operator_graph_summary summary;
    yvex_operator_node *nodes;
    yvex_operator_edge *edges;
};

static int operator_refuse(yvex_error *err, yvex_status status,
                           const char *reason)
{
    yvex_error_set(err, status, "graph.operator-ir", reason);
    return status;
}

static int operator_hash_finish(
    yvex_sha256 *hash, char identity[YVEX_SHA256_HEX_BYTES])
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!yvex_sha256_final(hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}

static int operator_node_seal(
    yvex_operator_node *node, const char *semantic_identity, yvex_error *err)
{
    yvex_sha256 hash;
    if (!node || node->schema_version != YVEX_OPERATOR_GRAPH_SCHEMA_V1 ||
        node->kind >= YVEX_OPERATOR_KIND_COUNT ||
        node->scope > YVEX_TENSOR_SCOPE_DRAFT || !node->input_width ||
        !node->output_width ||
        node->numeric_contract > YVEX_OPERATOR_NUMERIC_DISTRIBUTION_PRESERVING ||
        (node->state_read_mask |
         node->state_write_mask) >= (1ull << YVEX_MODEL_STATE_CLASS_COUNT) ||
        !yvex_sha256_hex_valid(node->attribute_identity) ||
        !yvex_sha256_hex_valid(semantic_identity))
        return operator_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "operator nodes require typed geometry, state, and semantic identity");
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.operator-node.v1") ||
        !yvex_sha256_update_text(&hash, semantic_identity) ||
        !yvex_sha256_update_u64(&hash, node->ordinal) ||
        !yvex_sha256_update_u64(&hash, node->kind) ||
        !yvex_sha256_update_u64(&hash, node->scope) ||
        !yvex_sha256_update_u64(&hash, node->layer_index) ||
        !yvex_sha256_update_u64(&hash, node->predictor_index) ||
        !yvex_sha256_update_u64(&hash, node->input_width) ||
        !yvex_sha256_update_u64(&hash, node->output_width) ||
        !yvex_sha256_update_u64(&hash, node->state_read_mask) ||
        !yvex_sha256_update_u64(&hash, node->state_write_mask) ||
        !yvex_sha256_update_u64(&hash, node->numeric_contract) ||
        !yvex_sha256_update_text(&hash, node->attribute_identity) ||
        !operator_hash_finish(&hash, node->identity))
        return operator_refuse(err, YVEX_ERR_STATE,
                               "operator node identity derivation failed");
    return YVEX_OK;
}

static int operator_edge_valid(const yvex_operator_edge *edge,
                               unsigned long long node_count)
{
    int state_read = edge && edge->kind == YVEX_OPERATOR_EDGE_STATE_READ;
    int state_write = edge && edge->kind == YVEX_OPERATOR_EDGE_STATE_WRITE;
    if (!edge || edge->schema_version != YVEX_OPERATOR_GRAPH_SCHEMA_V1 ||
        edge->kind > YVEX_OPERATOR_EDGE_STATE_WRITE)
        return 0;
    if (state_read || state_write)
        return edge->state_class < YVEX_MODEL_STATE_CLASS_COUNT &&
               (state_read
                    ? edge->source_node == YVEX_OPERATOR_GRAPH_NO_NODE &&
                          edge->target_node < node_count
                    : edge->source_node < node_count &&
                          edge->target_node == YVEX_OPERATOR_GRAPH_NO_NODE);
    return edge->source_node < node_count && edge->target_node < node_count &&
           edge->source_node < edge->target_node;
}

static int operator_edge_seal(yvex_operator_edge *edge,
                              const yvex_operator_node *nodes,
                              unsigned long long node_count, yvex_error *err)
{
    yvex_sha256 hash;
    const char *source, *target;
    if (!operator_edge_valid(edge, node_count))
        return operator_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "operator edges require ordered data or typed external state endpoints");
    source = edge->source_node == YVEX_OPERATOR_GRAPH_NO_NODE
                 ? "external-state" : nodes[edge->source_node].identity;
    target = edge->target_node == YVEX_OPERATOR_GRAPH_NO_NODE
                 ? "external-state" : nodes[edge->target_node].identity;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.operator-edge.v1") ||
        !yvex_sha256_update_u64(&hash, edge->ordinal) ||
        !yvex_sha256_update_u64(&hash, edge->kind) ||
        !yvex_sha256_update_u64(&hash, edge->state_class) ||
        !yvex_sha256_update_text(&hash, source) ||
        !yvex_sha256_update_text(&hash, target) ||
        !operator_hash_finish(&hash, edge->identity))
        return operator_refuse(err, YVEX_ERR_STATE,
                               "operator edge identity derivation failed");
    return YVEX_OK;
}

static int operator_graph_identity(
    yvex_operator_graph_ir *graph, yvex_error *err)
{
    yvex_sha256 hash;
    unsigned long long index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.operator-graph.v1") ||
        !yvex_sha256_update_u64(&hash, graph->summary.family_adapter_id) ||
        !yvex_sha256_update_u64(&hash, graph->summary.family_adapter_version) ||
        !yvex_sha256_update_text(&hash, graph->summary.semantic_model_identity) ||
        !yvex_sha256_update_u64(&hash, graph->summary.maximum_context) ||
        !yvex_sha256_update_u64(&hash, graph->summary.target_layer_count) ||
        !yvex_sha256_update_u64(&hash, graph->summary.draft_layer_count) ||
        !yvex_sha256_update_u64(&hash, graph->summary.node_count) ||
        !yvex_sha256_update_u64(&hash, graph->summary.edge_count))
        return operator_refuse(err, YVEX_ERR_STATE,
                               "operator graph identity preamble failed");
    for (index = 0ull; index < graph->summary.node_count; ++index)
        if (!yvex_sha256_update_text(&hash, graph->nodes[index].identity))
            return operator_refuse(err, YVEX_ERR_STATE,
                                   "operator graph node identity failed");
    for (index = 0ull; index < graph->summary.edge_count; ++index)
        if (!yvex_sha256_update_text(&hash, graph->edges[index].identity))
            return operator_refuse(err, YVEX_ERR_STATE,
                                   "operator graph edge identity failed");
    if (!operator_hash_finish(&hash, graph->summary.identity))
        return operator_refuse(err, YVEX_ERR_STATE,
                               "operator graph identity finalization failed");
    return YVEX_OK;
}

int yvex_operator_graph_ir_seal(
    yvex_operator_graph_ir **out,
    const yvex_operator_graph_request *request, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic;
    yvex_operator_graph_ir *graph = NULL;
    unsigned long long index;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    semantic = request
                   ? yvex_semantic_model_ir_summary_get(request->semantic_model)
                   : NULL;
    if (!out || !request || !semantic || !request->nodes ||
        !request->node_count || request->node_count > OPERATOR_GRAPH_MAX_NODES ||
        request->edge_count > OPERATOR_GRAPH_MAX_EDGES ||
        (request->edge_count && !request->edges))
        return operator_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "operator graph requires sealed semantics, nodes, and layer geometry");
    graph = (yvex_operator_graph_ir *)calloc(1u, sizeof(*graph));
    if (!graph) return operator_refuse(err, YVEX_ERR_NOMEM,
                                       "operator graph allocation failed");
    graph->nodes = (yvex_operator_node *)calloc(
        (size_t)request->node_count, sizeof(*graph->nodes));
    graph->edges = request->edge_count
                       ? (yvex_operator_edge *)calloc(
                             (size_t)request->edge_count, sizeof(*graph->edges))
                       : NULL;
    if (!graph->nodes || (request->edge_count && !graph->edges)) {
        yvex_operator_graph_ir_close(&graph);
        return operator_refuse(err, YVEX_ERR_NOMEM,
                               "operator graph directory allocation failed");
    }
    graph->summary.schema_version = YVEX_OPERATOR_GRAPH_SCHEMA_V1;
    graph->summary.family_adapter_id = semantic->family_adapter_id;
    graph->summary.family_adapter_version = semantic->family_adapter_version;
    graph->summary.maximum_context =
        semantic->execution_descriptor.maximum_context;
    graph->summary.node_count = request->node_count;
    graph->summary.edge_count = request->edge_count;
    graph->summary.target_layer_count = request->target_layer_count;
    graph->summary.draft_layer_count = request->draft_layer_count;
    yvex_core_text_copy(graph->summary.semantic_model_identity,
                        sizeof(graph->summary.semantic_model_identity),
                        semantic->identity);
    memcpy(graph->nodes, request->nodes,
           (size_t)request->node_count * sizeof(*graph->nodes));
    if (request->edge_count)
        memcpy(graph->edges, request->edges,
               (size_t)request->edge_count * sizeof(*graph->edges));
    for (index = 0ull; rc == YVEX_OK && index < request->node_count; ++index) {
        if (graph->nodes[index].ordinal != index) rc = YVEX_ERR_FORMAT;
        else rc = operator_node_seal(
            &graph->nodes[index], semantic->identity, err);
        graph->summary.state_class_mask |= graph->nodes[index].state_read_mask |
                                           graph->nodes[index].state_write_mask;
    }
    for (index = 0ull; rc == YVEX_OK && index < request->edge_count; ++index) {
        if (graph->edges[index].ordinal != index) rc = YVEX_ERR_FORMAT;
        else rc = operator_edge_seal(
            &graph->edges[index], graph->nodes, request->node_count, err);
    }
    if (rc == YVEX_ERR_FORMAT)
        rc = operator_refuse(err, YVEX_ERR_FORMAT,
                             "operator graph ordinals are not canonical");
    if (rc == YVEX_OK) rc = operator_graph_identity(graph, err);
    if (rc == YVEX_OK) {
        *out = graph;
        yvex_error_clear(err);
    } else yvex_operator_graph_ir_close(&graph);
    return rc;
}

const yvex_operator_graph_summary *yvex_operator_graph_ir_summary(
    const yvex_operator_graph_ir *graph)
{
    return graph ? &graph->summary : NULL;
}

const yvex_operator_node *yvex_operator_graph_ir_node_at(
    const yvex_operator_graph_ir *graph, unsigned long long index)
{
    return graph && index < graph->summary.node_count
               ? &graph->nodes[index] : NULL;
}

void yvex_operator_graph_ir_close(yvex_operator_graph_ir **graph)
{
    yvex_operator_graph_ir *owner;
    if (!graph || !*graph) return;
    owner = *graph;
    free(owner->edges);
    free(owner->nodes);
    memset(owner, 0, sizeof(*owner));
    free(owner);
    *graph = NULL;
}

typedef struct {
    yvex_operator_node *nodes;
    yvex_operator_edge *edges;
    unsigned long long node_count, node_capacity, edge_count, edge_capacity;
    unsigned long long target_previous, draft_previous;
    const char *attribute_identity;
} transformer_graph_builder;

static int transformer_edge_add(
    transformer_graph_builder *builder, yvex_operator_edge_kind kind,
    unsigned long long source, unsigned long long target,
    yvex_model_state_class state_class)
{
    yvex_operator_edge *edge;
    if (builder->edge_count >= builder->edge_capacity) return 0;
    edge = &builder->edges[builder->edge_count];
    edge->schema_version = YVEX_OPERATOR_GRAPH_SCHEMA_V1;
    edge->ordinal = builder->edge_count++;
    edge->source_node = source;
    edge->target_node = target;
    edge->kind = kind;
    edge->state_class = state_class;
    return 1;
}

static int transformer_state_edges(
    transformer_graph_builder *builder, unsigned long long node,
    unsigned long long read_mask, unsigned long long write_mask)
{
    unsigned int state_class;
    for (state_class = 0u; state_class < YVEX_MODEL_STATE_CLASS_COUNT;
         ++state_class) {
        unsigned long long bit = 1ull << state_class;
        if ((read_mask & bit) && !transformer_edge_add(
                builder, YVEX_OPERATOR_EDGE_STATE_READ,
                YVEX_OPERATOR_GRAPH_NO_NODE, node,
                (yvex_model_state_class)state_class))
            return 0;
        if ((write_mask & bit) && !transformer_edge_add(
                builder, YVEX_OPERATOR_EDGE_STATE_WRITE, node,
                YVEX_OPERATOR_GRAPH_NO_NODE,
                (yvex_model_state_class)state_class))
            return 0;
    }
    return 1;
}

static int transformer_node_add(
    transformer_graph_builder *builder, yvex_operator_kind kind,
    yvex_tensor_scope scope, unsigned long long layer_index,
    unsigned long long predictor_index, unsigned long long input_width,
    unsigned long long output_width, unsigned long long state_mask)
{
    unsigned long long *previous = scope == YVEX_TENSOR_SCOPE_DRAFT
                                       ? &builder->draft_previous
                                       : &builder->target_previous;
    unsigned long long ordinal;
    yvex_operator_node *node;
    if (builder->node_count >= builder->node_capacity) return 0;
    ordinal = builder->node_count++;
    node = &builder->nodes[ordinal];
    node->schema_version = YVEX_OPERATOR_GRAPH_SCHEMA_V1;
    node->ordinal = ordinal;
    node->kind = kind;
    node->scope = scope;
    node->layer_index = layer_index;
    node->predictor_index = predictor_index;
    node->input_width = input_width;
    node->output_width = output_width;
    node->state_read_mask = state_mask;
    node->state_write_mask = state_mask;
    node->numeric_contract = YVEX_OPERATOR_NUMERIC_REFERENCE_TOLERANCE;
    yvex_core_text_copy(node->attribute_identity,
                        sizeof(node->attribute_identity),
                        builder->attribute_identity);
    if (*previous != YVEX_OPERATOR_GRAPH_NO_NODE &&
        !transformer_edge_add(builder, YVEX_OPERATOR_EDGE_DATA,
                              *previous, ordinal,
                              YVEX_MODEL_STATE_CLASS_COUNT))
        return 0;
    *previous = ordinal;
    return transformer_state_edges(builder, ordinal, state_mask, state_mask);
}

static unsigned long long transformer_attention_state_mask(
    const yvex_attention_layer_plan *layer, int draft)
{
    unsigned long long mask = draft
        ? YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_DRAFT_PERSISTENT) : 0ull;
    if (!layer) return mask;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING);
    if (layer->compressor_required)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_COMPRESSED_HISTORY) |
                YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_MAIN_ROLLING);
    if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_HCA_HISTORY);
    if (layer->indexer_required)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_INDEXER_HISTORY) |
                YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_INDEXER_ROLLING);
    if (layer->mhc_mixing_rows)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_RESIDUAL_MIXING);
    return mask;
}

static unsigned long long semantic_attention_state_mask(
    const yvex_semantic_attention_layer *layer)
{
    unsigned long long mask = 0ull;

    if (!layer) return 0ull;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_SWA_RING);
    if (layer->compressor_required)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_COMPRESSED_HISTORY) |
                YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_MAIN_ROLLING);
    if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_HCA_HISTORY);
    if (layer->indexer_required)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_INDEXER_HISTORY) |
                YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_INDEXER_ROLLING);
    if (layer->mhc_mixing_rows)
        mask |= YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_RESIDUAL_MIXING);
    return mask;
}

static int transformer_scope_build(
    transformer_graph_builder *builder,
    const yvex_attention_plan *attention, unsigned long long hidden_width,
    yvex_tensor_scope scope)
{
    unsigned long long index, count = yvex_attention_plan_layer_count(attention);
    int draft = scope == YVEX_TENSOR_SCOPE_DRAFT;
    for (index = 0ull; index < count; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(attention, index);
        unsigned long long state_mask =
            transformer_attention_state_mask(layer, draft);
        if (!layer || !transformer_node_add(
                builder, YVEX_OPERATOR_ATTENTION, scope, layer->layer_index,
                layer->predictor_index, hidden_width, hidden_width, state_mask) ||
            !transformer_node_add(
                builder, YVEX_OPERATOR_MOE, scope, layer->layer_index,
                layer->predictor_index, hidden_width, hidden_width, 0ull))
            return 0;
    }
    return 1;
}

static int transformer_graph_nodes(
    transformer_graph_builder *builder,
    const yvex_model_execution_descriptor *model,
    const yvex_attention_plan *attention,
    const yvex_attention_plan *draft_attention)
{
    unsigned long long feature_width;
    if (!transformer_node_add(
            builder, YVEX_OPERATOR_EMBEDDING, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_ATTENTION_NO_LAYER, 0ull, 1ull, model->hidden_width, 0ull) ||
        !transformer_scope_build(builder, attention, model->hidden_width,
                                 YVEX_TENSOR_SCOPE_MAIN_LAYER) ||
        !transformer_node_add(
            builder, YVEX_OPERATOR_NORMALIZATION, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_ATTENTION_NO_LAYER, 0ull, model->hidden_width,
            model->output_input_width, 0ull) ||
        !transformer_node_add(
            builder, YVEX_OPERATOR_OUTPUT_PROJECTION,
            YVEX_TENSOR_SCOPE_GLOBAL, YVEX_ATTENTION_NO_LAYER, 0ull,
            model->output_input_width, model->output_vocabulary_size, 0ull))
        return 0;
    if (!draft_attention) return 1;
    if (!yvex_core_u64_mul(model->target_feature_count,
                           model->target_feature_width, &feature_width) ||
        !feature_width || !transformer_node_add(
            builder, YVEX_OPERATOR_FEATURE_PROJECTION,
            YVEX_TENSOR_SCOPE_DRAFT, YVEX_ATTENTION_NO_LAYER, 0ull,
            feature_width, model->hidden_width, 0ull) ||
        !transformer_scope_build(builder, draft_attention,
                                 model->hidden_width, YVEX_TENSOR_SCOPE_DRAFT) ||
        !transformer_node_add(
            builder, YVEX_OPERATOR_MARKOV_PROJECTION,
            YVEX_TENSOR_SCOPE_DRAFT, YVEX_ATTENTION_NO_LAYER, 0ull,
            model->hidden_width, model->markov_rank, 0ull) ||
        (model->confidence_width && !transformer_node_add(
            builder, YVEX_OPERATOR_CONFIDENCE, YVEX_TENSOR_SCOPE_DRAFT,
            YVEX_ATTENTION_NO_LAYER, 0ull, model->hidden_width,
            model->confidence_width, 0ull)) ||
        !transformer_node_add(
            builder, YVEX_OPERATOR_OUTPUT_PROJECTION,
            YVEX_TENSOR_SCOPE_DRAFT, YVEX_ATTENTION_NO_LAYER, 0ull,
            model->output_input_width, model->output_vocabulary_size, 0ull))
        return 0;
    return 1;
}

static int decoder_graph_nodes(
    transformer_graph_builder *builder,
    const yvex_model_execution_descriptor *model,
    const yvex_semantic_decoder_layer *layers, unsigned long long layer_count,
    const yvex_semantic_attention_layer *semantic_attention,
    unsigned long long semantic_attention_count,
    const yvex_attention_plan *attention)
{
    unsigned long long index, attention_index = 0ull;

    if (!transformer_node_add(
            builder, YVEX_OPERATOR_EMBEDDING, YVEX_TENSOR_SCOPE_GLOBAL,
            YVEX_ATTENTION_NO_LAYER, 0ull, 1ull, model->hidden_width, 0ull))
        return 0;
    for (index = 0ull; index < layer_count; ++index) {
        const yvex_semantic_decoder_layer *layer = &layers[index];
        unsigned long long state_mask = 0ull;
        yvex_operator_kind kind;
        char mixer_identity[YVEX_SEMANTIC_DECODER_IDENTITY_CAP];

        if (layer->mixer ==
            YVEX_SEMANTIC_DECODER_MIXER_FULL_CAUSAL_ATTENTION) {
            const yvex_attention_layer_plan *physical =
                attention ? yvex_attention_plan_layer_at(
                                attention, attention_index) : NULL;
            if (attention_index >= semantic_attention_count ||
                semantic_attention[attention_index].layer_index != index ||
                (attention &&
                 (!physical || physical->layer_index != index)))
                return 0;
            kind = YVEX_OPERATOR_ATTENTION;
            state_mask = physical
                             ? transformer_attention_state_mask(physical, 0)
                             : semantic_attention_state_mask(
                                   &semantic_attention[attention_index]);
            builder->attribute_identity = model->identity;
            attention_index++;
        } else if (layer->mixer ==
                   YVEX_SEMANTIC_DECODER_MIXER_GATED_DELTA) {
            if (!yvex_semantic_gated_delta_requirement_identity(
                    &layer->gated_delta, mixer_identity))
                return 0;
            kind = YVEX_OPERATOR_STATEFUL_SEQUENCE_MIXER;
            state_mask = YVEX_MODEL_STATE_CLASS_BIT(
                YVEX_MODEL_STATE_RECURRENT_SEQUENCE);
            builder->attribute_identity = mixer_identity;
        } else {
            return 0;
        }
        if (!transformer_node_add(
                builder, kind, YVEX_TENSOR_SCOPE_MAIN_LAYER, index, 0ull,
                layer->hidden_width, layer->hidden_width, state_mask))
            return 0;
        builder->attribute_identity = model->identity;
        if (!transformer_node_add(
                builder, YVEX_OPERATOR_DENSE_FEED_FORWARD,
                YVEX_TENSOR_SCOPE_MAIN_LAYER, index, 0ull,
                layer->hidden_width, layer->hidden_width, 0ull))
            return 0;
    }
    return attention_index == semantic_attention_count &&
           transformer_node_add(
               builder, YVEX_OPERATOR_NORMALIZATION,
               YVEX_TENSOR_SCOPE_GLOBAL, YVEX_ATTENTION_NO_LAYER, 0ull,
               model->hidden_width, model->output_input_width, 0ull) &&
           transformer_node_add(
               builder, YVEX_OPERATOR_OUTPUT_PROJECTION,
               YVEX_TENSOR_SCOPE_GLOBAL, YVEX_ATTENTION_NO_LAYER, 0ull,
               model->output_input_width, model->output_vocabulary_size, 0ull);
}

int yvex_operator_graph_ir_build_transformer(
    yvex_operator_graph_ir **out,
    const yvex_semantic_model_ir *semantic_model,
    const yvex_attention_plan *attention,
    const yvex_attention_plan *draft_attention, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_model_execution_descriptor *model =
        semantic ? &semantic->execution_descriptor : NULL;
    transformer_graph_builder builder = {0};
    yvex_operator_graph_request request = {0};
    unsigned long long layers, node_capacity, edge_capacity;
    int rc;
    if (out) *out = NULL;
    if (!out || !semantic || !model || !attention ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V1 ||
        strcmp(model->identity, semantic->semantic_payload_identity) != 0 ||
        yvex_attention_plan_layer_count(attention) != model->layer_count ||
        ((draft_attention != NULL) != (model->draft_layer_count != 0ull)) ||
        (draft_attention && yvex_attention_plan_layer_count(draft_attention) !=
                                model->draft_layer_count))
        return operator_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "transformer graph requires matching semantic, attention, and layer facts");
    if (!yvex_core_u64_add(model->layer_count, model->draft_layer_count,
                           &layers) ||
        !yvex_core_u64_mul(layers, 2ull, &node_capacity) ||
        !yvex_core_u64_add(node_capacity,
                           draft_attention ? 7ull : 3ull, &node_capacity) ||
        !yvex_core_u64_mul(
            node_capacity, 1ull + 2ull * YVEX_MODEL_STATE_CLASS_COUNT,
            &edge_capacity) ||
        node_capacity > SIZE_MAX / sizeof(*builder.nodes) ||
        edge_capacity > SIZE_MAX / sizeof(*builder.edges))
        return operator_refuse(err, YVEX_ERR_BOUNDS,
                               "transformer graph extent overflowed");
    builder.nodes = (yvex_operator_node *)calloc(
        (size_t)node_capacity, sizeof(*builder.nodes));
    builder.edges = (yvex_operator_edge *)calloc(
        (size_t)edge_capacity, sizeof(*builder.edges));
    if (!builder.nodes || !builder.edges) {
        free(builder.edges);
        free(builder.nodes);
        return operator_refuse(err, YVEX_ERR_NOMEM,
                               "transformer graph workspace allocation failed");
    }
    builder.node_capacity = node_capacity;
    builder.edge_capacity = edge_capacity;
    builder.target_previous = YVEX_OPERATOR_GRAPH_NO_NODE;
    builder.draft_previous = YVEX_OPERATOR_GRAPH_NO_NODE;
    builder.attribute_identity = semantic->semantic_payload_identity;
    rc = transformer_graph_nodes(
        &builder, model, attention, draft_attention)
             ? YVEX_OK : YVEX_ERR_STATE;
    request.semantic_model = semantic_model;
    request.nodes = builder.nodes;
    request.node_count = builder.node_count;
    request.edges = builder.edges;
    request.edge_count = builder.edge_count;
    request.target_layer_count = model->layer_count;
    request.draft_layer_count = model->draft_layer_count;
    if (rc == YVEX_OK)
        rc = yvex_operator_graph_ir_seal(out, &request, err);
    else
        rc = operator_refuse(err, YVEX_ERR_STATE,
                             "transformer operator composition failed");
    free(builder.edges);
    free(builder.nodes);
    return rc;
}

int yvex_operator_graph_ir_build_decoder(
    yvex_operator_graph_ir **out,
    const yvex_semantic_model_ir *semantic_model,
    const yvex_attention_plan *attention,
    const yvex_attention_plan *draft_attention, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_model_execution_descriptor *model =
        semantic ? &semantic->execution_descriptor : NULL;
    const yvex_semantic_decoder_layer *layers = NULL;
    const yvex_semantic_attention_layer *semantic_attention = NULL;
    unsigned long long layer_count = 0ull, attention_count = 0ull;
    unsigned long long node_capacity, edge_capacity;
    transformer_graph_builder builder = {0};
    yvex_operator_graph_request request = {0};
    int rc;

    if (out) *out = NULL;
    if (!out || !semantic || !model || draft_attention ||
        semantic->schema_version != YVEX_SEMANTIC_MODEL_IR_SCHEMA_V2 ||
        model->schema_version != YVEX_MODEL_EXECUTION_DESCRIPTOR_SCHEMA_V2 ||
        strcmp(model->identity, semantic->semantic_payload_identity) != 0 ||
        !yvex_semantic_model_ir_decoder_view(
            semantic_model, &layers, &layer_count) ||
        !yvex_semantic_model_ir_attention_view(
            semantic_model, YVEX_TENSOR_SCOPE_MAIN_LAYER,
            &semantic_attention, &attention_count) ||
        layer_count != model->layer_count ||
        (attention && yvex_attention_plan_layer_count(attention) != attention_count))
        return operator_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "decoder graph requires matching semantic and attention topology");
    if (!yvex_core_u64_mul(layer_count, 2ull, &node_capacity) ||
        !yvex_core_u64_add(node_capacity, 3ull, &node_capacity) ||
        !yvex_core_u64_mul(
            node_capacity, 1ull + 2ull * YVEX_MODEL_STATE_CLASS_COUNT,
            &edge_capacity) ||
        node_capacity > SIZE_MAX / sizeof(*builder.nodes) ||
        edge_capacity > SIZE_MAX / sizeof(*builder.edges))
        return operator_refuse(
            err, YVEX_ERR_BOUNDS, "decoder graph extent overflowed");
    builder.nodes = calloc((size_t)node_capacity, sizeof(*builder.nodes));
    builder.edges = calloc((size_t)edge_capacity, sizeof(*builder.edges));
    if (!builder.nodes || !builder.edges) {
        free(builder.edges);
        free(builder.nodes);
        return operator_refuse(
            err, YVEX_ERR_NOMEM, "decoder graph workspace allocation failed");
    }
    builder.node_capacity = node_capacity;
    builder.edge_capacity = edge_capacity;
    builder.target_previous = YVEX_OPERATOR_GRAPH_NO_NODE;
    builder.draft_previous = YVEX_OPERATOR_GRAPH_NO_NODE;
    builder.attribute_identity = model->identity;
    rc = decoder_graph_nodes(
             &builder, model, layers, layer_count, semantic_attention,
             attention_count, attention)
             ? YVEX_OK : YVEX_ERR_STATE;
    request = (yvex_operator_graph_request){
        .semantic_model = semantic_model,
        .nodes = builder.nodes,
        .node_count = builder.node_count,
        .edges = builder.edges,
        .edge_count = builder.edge_count,
        .target_layer_count = layer_count};
    if (rc == YVEX_OK)
        rc = yvex_operator_graph_ir_seal(out, &request, err);
    else
        rc = operator_refuse(
            err, YVEX_ERR_STATE, "decoder operator composition failed");
    if (rc == YVEX_OK &&
        yvex_operator_graph_ir_summary(*out)->state_class_mask !=
            model->persistent_state_class_mask) {
        yvex_operator_graph_ir_close(out);
        rc = operator_refuse(
            err, YVEX_ERR_STATE,
            "decoder graph state classes disagree with model semantics");
    }
    free(builder.edges);
    free(builder.nodes);
    return rc;
}
