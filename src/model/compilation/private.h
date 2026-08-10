/* Mutable builder representation shared only by the compilation implementation. */
#ifndef SRC_MODEL_COMPILATION_PRIVATE_H_INCLUDED
#define SRC_MODEL_COMPILATION_PRIVATE_H_INCLUDED

#include <yvex/internal/compilation.h>

typedef struct {
    unsigned long long hash;
    unsigned long long value_plus_one;
} yvex_transform_index_slot;
typedef struct {
    yvex_transform_node node;
    unsigned long long provisional_id;
} yvex_transform_builder_node;
struct yvex_transform_builder {
    yvex_transform_ir_state state;
    yvex_transform_allocator allocator;
    yvex_transform_budget budget;
    yvex_transform_header header;
    char logical_model_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char required_payload_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char payload_trust_class[40];
    char component_manifest_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char architecture_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char role_map_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char unresolved_requirements_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    yvex_transform_source_value *sources;
    yvex_transform_value *values;
    yvex_transform_builder_node *nodes;
    unsigned long long *edges;
    unsigned long long source_count, source_capacity;
    unsigned long long value_count, value_capacity;
    unsigned long long node_count, node_capacity;
    unsigned long long edge_count, edge_capacity;
    unsigned long long terminal_count;
    size_t owned_bytes, peak_bytes;
};
struct yvex_transform_ir {
    yvex_transform_allocator allocator;
    yvex_transform_source_value *sources;
    yvex_transform_value *values;
    yvex_transform_node *nodes;
    unsigned long long *edges;
    unsigned long long *topological_order;
    unsigned long long *terminal_values;
    yvex_transform_index_slot *source_index;
    yvex_transform_index_slot *terminal_index;
    unsigned long long source_index_capacity, terminal_index_capacity;
    yvex_transform_ir_summary summary;
};

void *yvex_transform_allocate_zero(yvex_transform_allocator *allocator, size_t size);
unsigned long long yvex_transform_hash_logical_key(const yvex_transform_logical_key *key);
unsigned long long yvex_transform_index_capacity(unsigned long long count);
int yvex_transform_index_insert(yvex_transform_index_slot *slots,
                                unsigned long long capacity,
                                unsigned long long hash,
                                unsigned long long value);
int yvex_transform_ir_validate_and_seal(yvex_transform_builder *builder,
                                        yvex_transform_ir **out,
                                        yvex_transform_failure *failure,
                                        yvex_error *err);
int yvex_transform_ir_compute_identity(yvex_transform_ir *ir,
                                       yvex_transform_failure *failure,
                                       yvex_error *err);

#endif
