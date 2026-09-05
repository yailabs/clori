/* Immutable physical package facts consumed without reconstructing model topology. */
#ifndef INCLUDE_YVEX_INTERNAL_EXECUTION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_EXECUTION_H_INCLUDED

#include <yvex/core.h>
#include <yvex/internal/artifact.h>
#include <yvex/internal/model.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_PHYSICAL_EXECUTION_SCHEMA_V5 5u

typedef enum {
    YVEX_EXECUTION_CONSUMER_EMBEDDING = 0,
    YVEX_EXECUTION_CONSUMER_ATTENTION_PROJECTION,
    YVEX_EXECUTION_CONSUMER_ATTENTION_STATE,
    YVEX_EXECUTION_CONSUMER_MOE_ROUTER,
    YVEX_EXECUTION_CONSUMER_ROUTED_GATE_UP,
    YVEX_EXECUTION_CONSUMER_ROUTED_DOWN,
    YVEX_EXECUTION_CONSUMER_SHARED_EXPERT,
    YVEX_EXECUTION_CONSUMER_FINAL_NORMALIZATION,
    YVEX_EXECUTION_CONSUMER_OUTPUT_HEAD,
    YVEX_EXECUTION_CONSUMER_DRAFT_FEATURE_PROJECTION,
    YVEX_EXECUTION_CONSUMER_DRAFT_BACKBONE,
    YVEX_EXECUTION_CONSUMER_MARKOV,
    YVEX_EXECUTION_CONSUMER_CONFIDENCE,
    YVEX_EXECUTION_CONSUMER_COUNT
} yvex_execution_consumer_class;

typedef enum {
    YVEX_EXECUTION_LAYOUT_CANONICAL_ROW = 0,
    YVEX_EXECUTION_LAYOUT_CONTIGUOUS_DENSE,
    YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR
} yvex_execution_layout_class;

typedef enum {
    YVEX_EXECUTION_SHARING_EXCLUSIVE = 0,
    YVEX_EXECUTION_SHARING_MODEL_READ_ONLY,
    YVEX_EXECUTION_SHARING_ALIAS
} yvex_execution_sharing_class;

typedef struct {
    unsigned int schema_version;
    unsigned long long terminal_tensor_id;
    yvex_tensor_role role;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_count;
    unsigned int canonical_qtype;
    unsigned long long canonical_row_width, canonical_row_count;
    unsigned long long encoded_offset, encoded_bytes, alignment;
    yvex_execution_consumer_class consumer;
    yvex_execution_layout_class layout;
    yvex_execution_sharing_class sharing;
    char terminal_identity[YVEX_SHA256_HEX_CAP];
    char decision_identity[YVEX_SHA256_HEX_CAP];
} yvex_physical_execution_decision;

typedef struct {
    unsigned int schema_version;
    unsigned long long decision_count, encoded_bytes;
    unsigned long long consumer_counts[YVEX_EXECUTION_CONSUMER_COUNT];
    unsigned long long layout_counts[4];
    char physical_variant_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
} yvex_physical_execution_summary;

typedef struct yvex_physical_execution_ir yvex_physical_execution_ir;

int yvex_physical_execution_ir_build(
    yvex_physical_execution_ir **out,
    const yvex_materialization_session *materialization,
    const yvex_runtime_descriptor *descriptor,
    const char *physical_variant_identity, yvex_error *err);
int yvex_physical_execution_ir_import(
    yvex_physical_execution_ir **out, const yvex_physical_execution_summary *summary,
    const yvex_physical_execution_decision *decisions, unsigned long long count,
    yvex_error *err);
const yvex_physical_execution_summary *yvex_physical_execution_ir_summary(
    const yvex_physical_execution_ir *ir);
const yvex_physical_execution_decision *yvex_physical_execution_ir_decision_at(
    const yvex_physical_execution_ir *ir, unsigned long long index);
void yvex_physical_execution_ir_close(yvex_physical_execution_ir **ir);

#ifdef __cplusplus
}
#endif
#endif /* INCLUDE_YVEX_INTERNAL_EXECUTION_H_INCLUDED */
