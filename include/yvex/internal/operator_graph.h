/*
 * Canonical operators terminate family composition before physical execution planning.
 * Nodes and edges are pointer-free semantic facts; process-local compiler payloads never cross
 * this boundary.
 */
#ifndef INCLUDE_YVEX_INTERNAL_OPERATOR_GRAPH_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_OPERATOR_GRAPH_H_INCLUDED

#include <yvex/internal/compiler.h>
#include <yvex/internal/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_OPERATOR_GRAPH_SCHEMA_V1 1u
#define YVEX_OPERATOR_GRAPH_NO_NODE (~0ull)

struct yvex_attention_plan;

typedef enum {
    YVEX_OPERATOR_EMBEDDING = 0,
    YVEX_OPERATOR_ATTENTION,
    YVEX_OPERATOR_MOE,
    YVEX_OPERATOR_NORMALIZATION,
    YVEX_OPERATOR_OUTPUT_PROJECTION,
    YVEX_OPERATOR_FEATURE_PROJECTION,
    YVEX_OPERATOR_MARKOV_PROJECTION,
    YVEX_OPERATOR_CONFIDENCE,
    YVEX_OPERATOR_TEXT_CONDITIONING,
    YVEX_OPERATOR_AUDIO_CODEC,
    YVEX_OPERATOR_MULTIMODAL_TRANSFORMER,
    YVEX_OPERATOR_RECTIFIED_FLOW,
    YVEX_OPERATOR_KIND_COUNT
} yvex_operator_kind;

typedef enum {
    YVEX_OPERATOR_NUMERIC_EXACT = 0,
    YVEX_OPERATOR_NUMERIC_REFERENCE_TOLERANCE,
    YVEX_OPERATOR_NUMERIC_DISTRIBUTION_PRESERVING
} yvex_operator_numeric_contract;

typedef enum {
    YVEX_OPERATOR_EDGE_DATA = 0,
    YVEX_OPERATOR_EDGE_ORDER,
    YVEX_OPERATOR_EDGE_STATE_READ,
    YVEX_OPERATOR_EDGE_STATE_WRITE
} yvex_operator_edge_kind;

typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal;
    yvex_operator_kind kind;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index;
    unsigned long long input_width, output_width;
    unsigned long long state_read_mask, state_write_mask;
    yvex_operator_numeric_contract numeric_contract;
    char attribute_identity[YVEX_SHA256_HEX_BYTES];
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_operator_node;

typedef struct {
    unsigned int schema_version;
    unsigned long long ordinal, source_node, target_node;
    yvex_operator_edge_kind kind;
    yvex_model_state_class state_class;
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_operator_edge;

typedef struct {
    unsigned int schema_version;
    unsigned long long family_adapter_id, family_adapter_version;
    unsigned long long maximum_context, node_count, edge_count;
    unsigned long long target_layer_count, draft_layer_count;
    unsigned long long state_class_mask;
    char semantic_model_identity[YVEX_SHA256_HEX_BYTES];
    char identity[YVEX_SHA256_HEX_BYTES];
} yvex_operator_graph_summary;

typedef struct {
    const yvex_semantic_model_ir *semantic_model;
    const yvex_operator_node *nodes;
    unsigned long long node_count;
    const yvex_operator_edge *edges;
    unsigned long long edge_count;
    unsigned long long target_layer_count, draft_layer_count;
} yvex_operator_graph_request;

int yvex_operator_graph_ir_seal(
    yvex_operator_graph_ir **out,
    const yvex_operator_graph_request *request, yvex_error *err);
int yvex_operator_graph_ir_build_transformer(
    yvex_operator_graph_ir **out,
    const yvex_semantic_model_ir *semantic_model,
    const struct yvex_attention_plan *attention,
    const struct yvex_attention_plan *draft_attention, yvex_error *err);
const yvex_operator_graph_summary *yvex_operator_graph_ir_summary(
    const yvex_operator_graph_ir *graph);
const yvex_operator_node *yvex_operator_graph_ir_node_at(
    const yvex_operator_graph_ir *graph, unsigned long long index);
void yvex_operator_graph_ir_close(yvex_operator_graph_ir **graph);

#ifdef __cplusplus
}
#endif
#endif
