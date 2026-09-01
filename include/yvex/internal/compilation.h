/*
 * Compilation owners exchange immutable transformation IR and runtime-binding facts through this
 * ABI. Neither representation reads payload bytes or executes a backend.
 */
#ifndef INCLUDE_YVEX_INTERNAL_COMPILATION_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_COMPILATION_H_INCLUDED

#include <stddef.h>
#include <yvex/core.h>
#include <yvex/model.h>
#include <yvex/source.h>
#include <yvex/internal/source_payload.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ir. */
#define YVEX_TRANSFORM_IR_SCHEMA_VERSION 1u
#define YVEX_TRANSFORM_IR_COMPONENT_SCHEMA_VERSION 2u
#define YVEX_TRANSFORM_IR_IDENTITY_CAP 65u
#define YVEX_TRANSFORM_IR_SOURCE_NAME_CAP 256u
#define YVEX_TRANSFORM_IR_SHARD_NAME_CAP 128u
#define YVEX_TRANSFORM_IR_MAX_RANK YVEX_NATIVE_WEIGHT_MAX_DIMS
#define YVEX_TRANSFORM_IR_NO_ID (~0ull)
typedef enum {
    YVEX_TRANSFORM_IR_STATE_BUILDING = 0,
    YVEX_TRANSFORM_IR_STATE_SEALED,
    YVEX_TRANSFORM_IR_STATE_FAILED,
    YVEX_TRANSFORM_IR_STATE_RELEASED
} yvex_transform_ir_state;
typedef enum {
    YVEX_TRANSFORM_FAILURE_NONE = 0,
    YVEX_TRANSFORM_FAILURE_INVALID_ARGUMENT,
    YVEX_TRANSFORM_FAILURE_INVALID_STATE,
    YVEX_TRANSFORM_FAILURE_ARCHITECTURE_NOT_ADMITTED,
    YVEX_TRANSFORM_FAILURE_COVERAGE_INCOMPLETE,
    YVEX_TRANSFORM_FAILURE_SOURCE_IDENTITY_MISMATCH,
    YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH,
    YVEX_TRANSFORM_FAILURE_SCHEMA_UNSUPPORTED,
    YVEX_TRANSFORM_FAILURE_INVALID_LOGICAL_KEY,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_LOGICAL_KEY,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_SOURCE,
    YVEX_TRANSFORM_FAILURE_MISSING_SOURCE,
    YVEX_TRANSFORM_FAILURE_UNEXPECTED_SOURCE,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_VALUE,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_NODE,
    YVEX_TRANSFORM_FAILURE_MISSING_PRODUCER,
    YVEX_TRANSFORM_FAILURE_MULTIPLE_PRODUCERS,
    YVEX_TRANSFORM_FAILURE_UNRESOLVED_EDGE,
    YVEX_TRANSFORM_FAILURE_CYCLE,
    YVEX_TRANSFORM_FAILURE_UNSUPPORTED_OPERATION,
    YVEX_TRANSFORM_FAILURE_INVALID_ARITY,
    YVEX_TRANSFORM_FAILURE_INVALID_DTYPE,
    YVEX_TRANSFORM_FAILURE_UNSUPPORTED_SOURCE_DTYPE,
    YVEX_TRANSFORM_FAILURE_INVALID_RANK,
    YVEX_TRANSFORM_FAILURE_INVALID_SHAPE,
    YVEX_TRANSFORM_FAILURE_DIMENSION_OVERFLOW,
    YVEX_TRANSFORM_FAILURE_INVALID_AXIS,
    YVEX_TRANSFORM_FAILURE_INVALID_PERMUTATION,
    YVEX_TRANSFORM_FAILURE_ELEMENT_COUNT_MISMATCH,
    YVEX_TRANSFORM_FAILURE_INVALID_AGGREGATION,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_EXPERT,
    YVEX_TRANSFORM_FAILURE_MISSING_EXPERT,
    YVEX_TRANSFORM_FAILURE_UNCONSUMED_SOURCE,
    YVEX_TRANSFORM_FAILURE_DUPLICATE_TERMINAL,
    YVEX_TRANSFORM_FAILURE_MISSING_TERMINAL,
    YVEX_TRANSFORM_FAILURE_RESOURCE_BUDGET,
    YVEX_TRANSFORM_FAILURE_ALLOCATION,
    YVEX_TRANSFORM_FAILURE_IDENTITY_ENCODING,
    YVEX_TRANSFORM_FAILURE_SEAL,
    YVEX_TRANSFORM_FAILURE_LOWERING_DIVERGENCE,
    YVEX_TRANSFORM_FAILURE_MAPPING_IDENTITY,
    YVEX_TRANSFORM_FAILURE_CLEANUP
} yvex_transform_failure_code;
typedef enum {
    YVEX_TRANSFORM_SCOPE_GLOBAL = 0,
    YVEX_TRANSFORM_SCOPE_MAIN_LAYER,
    YVEX_TRANSFORM_SCOPE_AUXILIARY
} yvex_transform_scope;
typedef enum {
    YVEX_TRANSFORM_SUBSYSTEM_GLOBAL = 0,
    YVEX_TRANSFORM_SUBSYSTEM_ATTENTION,
    YVEX_TRANSFORM_SUBSYSTEM_COMPRESSOR,
    YVEX_TRANSFORM_SUBSYSTEM_INDEXER,
    YVEX_TRANSFORM_SUBSYSTEM_NORMALIZATION,
    YVEX_TRANSFORM_SUBSYSTEM_RESIDUAL,
    YVEX_TRANSFORM_SUBSYSTEM_ROUTER,
    YVEX_TRANSFORM_SUBSYSTEM_ROUTED_EXPERT,
    YVEX_TRANSFORM_SUBSYSTEM_SHARED_EXPERT,
    YVEX_TRANSFORM_SUBSYSTEM_OUTPUT,
    YVEX_TRANSFORM_SUBSYSTEM_AUXILIARY,
    YVEX_TRANSFORM_SUBSYSTEM_SEQUENCE_MIXER,
    YVEX_TRANSFORM_SUBSYSTEM_COUNT
} yvex_transform_subsystem;
typedef enum {
    YVEX_TRANSFORM_VALUE_SOURCE = 0,
    YVEX_TRANSFORM_VALUE_INTERMEDIATE,
    YVEX_TRANSFORM_VALUE_TERMINAL
} yvex_transform_value_kind;
typedef enum {
    YVEX_TRANSFORM_DTYPE_UNKNOWN = 0,
    YVEX_TRANSFORM_DTYPE_F32,
    YVEX_TRANSFORM_DTYPE_F16,
    YVEX_TRANSFORM_DTYPE_BF16,
    YVEX_TRANSFORM_DTYPE_I32,
    YVEX_TRANSFORM_DTYPE_I64,
    YVEX_TRANSFORM_DTYPE_FP8_E4M3,
    YVEX_TRANSFORM_DTYPE_E8M0_SCALE,
    YVEX_TRANSFORM_DTYPE_PACKED_FP4,
    YVEX_TRANSFORM_DTYPE_REAL
} yvex_transform_dtype;
typedef enum {
    YVEX_TRANSFORM_OP_IDENTITY = 0,
    YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR,
    YVEX_TRANSFORM_OP_CHECKED_CAST,
    YVEX_TRANSFORM_OP_RESHAPE,
    YVEX_TRANSFORM_OP_TRANSPOSE,
    YVEX_TRANSFORM_OP_CONCATENATE,
    YVEX_TRANSFORM_OP_STACK,
    YVEX_TRANSFORM_OP_AGGREGATE,
    YVEX_TRANSFORM_OP_EXPERT_AGGREGATE,
    YVEX_TRANSFORM_OP_COUNT
} yvex_transform_operation_kind;
typedef enum {
    YVEX_TRANSFORM_NUMERIC_EXACT = 0,
    YVEX_TRANSFORM_NUMERIC_LOSSLESS,
    YVEX_TRANSFORM_NUMERIC_RANGE_PROOF,
    YVEX_TRANSFORM_NUMERIC_APPROXIMATION_ALLOWED
} yvex_transform_numeric_semantics;
typedef enum {
    YVEX_TRANSFORM_ORDER_INPUT = 0,
    YVEX_TRANSFORM_ORDER_AXIS,
    YVEX_TRANSFORM_ORDER_EXPERT_INDEX
} yvex_transform_ordering_semantics;
enum {
    YVEX_TRANSFORM_PRECISION_EXACT = 1u << 0,
    YVEX_TRANSFORM_PRECISION_LOSSLESS = 1u << 1,
    YVEX_TRANSFORM_PRECISION_RANGE_PROOF = 1u << 2,
    YVEX_TRANSFORM_PRECISION_SCALE_PAIRED = 1u << 3,
    YVEX_TRANSFORM_PRECISION_QUANTIZABLE_WEIGHT = 1u << 4,
    YVEX_TRANSFORM_PRECISION_INTEGER_ONLY = 1u << 5,
    YVEX_TRANSFORM_PRECISION_REFERENCE_COMPUTE = 1u << 6
};
enum {
    YVEX_TRANSFORM_PHYSICAL_F32 = 1u << 0,
    YVEX_TRANSFORM_PHYSICAL_F16 = 1u << 1,
    YVEX_TRANSFORM_PHYSICAL_BF16 = 1u << 2,
    YVEX_TRANSFORM_PHYSICAL_I32 = 1u << 3,
    YVEX_TRANSFORM_PHYSICAL_QUANTIZED = 1u << 4
};
typedef struct {
    unsigned int rank;
    unsigned long long dims[YVEX_TRANSFORM_IR_MAX_RANK];
} yvex_transform_shape;
typedef struct {
    yvex_transform_scope scope;
    yvex_transform_subsystem subsystem;
    yvex_tensor_role role;
    unsigned long long component_identity;
    unsigned long long semantic_role;
    unsigned long long phase_identity;
    unsigned long long lifetime_identity;
    unsigned long long layer_index;
    unsigned long long auxiliary_index;
    unsigned long long group_index;
} yvex_transform_logical_key;
typedef struct {
    unsigned int flags;
    unsigned int allowed_physical_classes;
    int approximation_allowed;
    int range_proof_required;
    int reference_compute_required;
} yvex_transform_precision_constraint;
typedef struct {
    const char *source_name;
    const char *shard_name;
    unsigned long long source_tensor_index;
    unsigned long long requirement_index;
    unsigned long long source_snapshot_identity;
    yvex_native_dtype source_dtype;
    yvex_transform_dtype value_dtype;
    yvex_transform_shape shape;
    unsigned long long relative_begin;
    unsigned long long relative_end;
    unsigned long long requirement_identity;
    yvex_transform_scope scope;
    yvex_transform_subsystem subsystem;
    yvex_tensor_role role_hint;
    unsigned long long component_identity;
    unsigned long long semantic_role;
    unsigned long long phase_identity;
    unsigned long long lifetime_identity;
    unsigned long long unresolved_requirement_identity;
    unsigned long long layer_index;
    unsigned long long auxiliary_index;
    unsigned long long expert_index;
    unsigned long long required_uses;
} yvex_transform_source_spec;
typedef struct {
    const char *source_name;
    yvex_native_dtype source_dtype;
    yvex_transform_dtype value_dtype;
    yvex_transform_shape shape;
    unsigned long long requirement_index;
    yvex_transform_scope scope;
    yvex_transform_subsystem subsystem;
    yvex_tensor_role role_hint;
    unsigned long long component_identity;
    unsigned long long semantic_role;
    unsigned long long phase_identity;
    unsigned long long lifetime_identity;
    unsigned long long unresolved_requirement_identity;
    unsigned long long layer_index;
    unsigned long long auxiliary_index;
    unsigned long long expert_index;
    unsigned long long required_uses;
} yvex_transform_source_requirement;
typedef struct {
    yvex_transform_value_kind kind;
    unsigned long long semantic_id;
    unsigned long long canonical_ordinal;
    yvex_transform_shape shape;
    yvex_transform_dtype dtype;
    yvex_transform_precision_constraint precision;
    yvex_transform_logical_key logical_key;
} yvex_transform_value_spec;
typedef struct {
    yvex_transform_operation_kind kind;
    unsigned long long output_value_id;
    const unsigned long long *input_value_ids;
    unsigned long long input_count;
    unsigned int axis;
    unsigned int permutation_rank;
    unsigned int permutation[YVEX_TRANSFORM_IR_MAX_RANK];
    unsigned long long expert_count;
    unsigned long long packing_factor;
    unsigned long long scale_group_width;
    unsigned long long scale_block_rows;
    unsigned long long scale_block_columns;
    yvex_transform_numeric_semantics numeric;
    yvex_transform_ordering_semantics ordering;
    int payload_execution_required;
} yvex_transform_node_spec;
typedef struct {
    unsigned long long value_id;
    char source_name[YVEX_TRANSFORM_IR_SOURCE_NAME_CAP];
    char shard_name[YVEX_TRANSFORM_IR_SHARD_NAME_CAP];
    unsigned long long source_tensor_index;
    unsigned long long requirement_index;
    unsigned long long source_snapshot_identity;
    yvex_native_dtype source_dtype;
    yvex_transform_dtype value_dtype;
    yvex_transform_shape shape;
    unsigned long long relative_begin;
    unsigned long long relative_end;
    unsigned long long requirement_identity;
    yvex_transform_scope scope;
    yvex_transform_subsystem subsystem;
    yvex_tensor_role role_hint;
    unsigned long long component_identity;
    unsigned long long semantic_role;
    unsigned long long phase_identity;
    unsigned long long lifetime_identity;
    unsigned long long unresolved_requirement_identity;
    unsigned long long layer_index;
    unsigned long long auxiliary_index;
    unsigned long long expert_index;
    unsigned long long required_uses;
} yvex_transform_source_value;
typedef struct {
    unsigned long long id;
    yvex_transform_value_kind kind;
    unsigned long long semantic_id;
    unsigned long long canonical_ordinal;
    unsigned long long source_index;
    unsigned long long producer_node_id;
    unsigned long long consumer_count;
    unsigned long long depth;
    yvex_transform_shape shape;
    yvex_transform_dtype dtype;
    yvex_transform_precision_constraint precision;
    yvex_transform_logical_key logical_key;
} yvex_transform_value;
typedef struct {
    unsigned long long id;
    yvex_transform_operation_kind kind;
    unsigned long long output_value_id;
    unsigned long long input_offset;
    unsigned long long input_count;
    unsigned int axis;
    unsigned int permutation_rank;
    unsigned int permutation[YVEX_TRANSFORM_IR_MAX_RANK];
    unsigned long long expert_count;
    unsigned long long packing_factor;
    unsigned long long scale_group_width;
    unsigned long long scale_block_rows;
    unsigned long long scale_block_columns;
    yvex_transform_numeric_semantics numeric;
    yvex_transform_ordering_semantics ordering;
    int payload_execution_required;
} yvex_transform_node;
typedef struct yvex_transform_failure {
    yvex_transform_failure_code code;
    unsigned long long value_id;
    unsigned long long node_id;
    unsigned long long source_index;
    unsigned long long terminal_ordinal;
    unsigned long long input_index;
    unsigned long long expected;
    unsigned long long actual;
    unsigned int axis;
} yvex_transform_failure;
typedef void *(*yvex_transform_allocate_fn)(size_t size, void *context);
typedef void (*yvex_transform_release_fn)(void *allocation, void *context);
typedef struct {
    yvex_transform_allocate_fn allocate;
    yvex_transform_release_fn release;
    void *context;
} yvex_transform_allocator;
typedef struct {
    unsigned long long maximum_sources;
    unsigned long long maximum_values;
    unsigned long long maximum_nodes;
    unsigned long long maximum_edges;
    unsigned long long maximum_terminals;
    size_t maximum_owned_bytes;
} yvex_transform_budget;
typedef struct {
    unsigned int schema_version;
    const char *logical_model_identity;
    unsigned long long source_snapshot_identity;
    unsigned long long coverage_identity;
    const char *required_payload_identity;
    const char *payload_trust_class;
    const char *component_manifest_identity;
    const char *architecture_identity;
    const char *role_map_identity;
    const char *unresolved_requirements_identity;
    unsigned long long expected_source_count;
    unsigned long long expected_terminal_count;
    unsigned long long header_scan_count;
} yvex_transform_header;
typedef struct yvex_transform_builder_options {
    yvex_transform_allocator allocator;
    yvex_transform_budget budget;
    const yvex_source_tensor_snapshot *source_snapshot;
} yvex_transform_builder_options;
typedef struct {
    unsigned int schema_version;
    yvex_transform_ir_state state;
    char logical_model_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    unsigned long long source_snapshot_identity;
    unsigned long long coverage_identity;
    char required_payload_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char payload_trust_class[40];
    char component_manifest_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char architecture_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char role_map_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char unresolved_requirements_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char transform_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    unsigned long long source_value_count;
    unsigned long long intermediate_value_count;
    unsigned long long value_count;
    unsigned long long node_count;
    unsigned long long edge_count;
    unsigned long long terminal_count;
    unsigned long long operation_counts[YVEX_TRANSFORM_OP_COUNT];
    unsigned long long maximum_fan_in;
    unsigned long long maximum_depth;
    unsigned long long index_capacity;
    unsigned long long source_lookup_count;
    unsigned long long terminal_lookup_count;
    unsigned long long validation_steps;
    size_t builder_peak_bytes;
    size_t sealed_ir_bytes;
    size_t temporary_validation_bytes;
    size_t total_owned_bytes;
    unsigned long long header_scan_count;
    unsigned long long payload_bytes_read;
    int complete;
} yvex_transform_ir_summary;
typedef struct yvex_transform_builder yvex_transform_builder;
typedef struct yvex_transform_ir yvex_transform_ir;
void yvex_transform_budget_default(yvex_transform_budget *budget);
int yvex_transform_builder_create(
    yvex_transform_builder **out,
    const yvex_transform_header *header,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure,
    yvex_error *err);
int yvex_transform_builder_add_source(
    yvex_transform_builder *builder,
    const yvex_transform_source_spec *spec,
    unsigned long long *value_id,
    yvex_transform_failure *failure,
    yvex_error *err);
int yvex_transform_builder_declare_value(
    yvex_transform_builder *builder,
    const yvex_transform_value_spec *spec,
    unsigned long long *value_id,
    yvex_transform_failure *failure,
    yvex_error *err);
int yvex_transform_builder_add_node(
    yvex_transform_builder *builder,
    const yvex_transform_node_spec *spec,
    unsigned long long *node_id,
    yvex_transform_failure *failure,
    yvex_error *err);
int yvex_transform_builder_seal(
    yvex_transform_builder *builder,
    yvex_transform_ir **out,
    yvex_transform_failure *failure,
    yvex_error *err);
void yvex_transform_builder_release(yvex_transform_builder **builder);
void yvex_transform_ir_release(yvex_transform_ir **ir);
const yvex_transform_ir_summary *yvex_transform_ir_summary_get(
    const yvex_transform_ir *ir);
const yvex_transform_source_value *yvex_transform_ir_source_at(
    const yvex_transform_ir *ir, unsigned long long index);
const yvex_transform_source_value *yvex_transform_ir_source_find(
    const yvex_transform_ir *ir, const char *source_name);
const yvex_transform_node *yvex_transform_ir_node_at(
    const yvex_transform_ir *ir, unsigned long long node_id);
const yvex_transform_node *yvex_transform_ir_node_topological_at(
    const yvex_transform_ir *ir, unsigned long long ordinal);
const yvex_transform_value *yvex_transform_ir_terminal_at(
    const yvex_transform_ir *ir, unsigned long long ordinal);
const yvex_transform_value *yvex_transform_ir_node_input_at(
    const yvex_transform_ir *ir, const yvex_transform_node *node,
    unsigned long long ordinal);

int yvex_transform_logical_key_equal(
    const yvex_transform_logical_key *left,
    const yvex_transform_logical_key *right);
int yvex_transform_shape_element_count(
    const yvex_transform_shape *shape,
    unsigned long long *out,
    yvex_transform_failure *failure,
    yvex_error *err);

typedef struct yvex_transform_recipe_sink yvex_transform_recipe_sink;
typedef struct {
    const yvex_transform_source_spec *sources;
    unsigned long long source_count;
    yvex_transform_value_spec terminal;
    yvex_transform_node_spec operation;
} yvex_transform_recipe;
typedef int (*yvex_transform_recipe_project_fn)(
    void *context, yvex_transform_recipe_sink *sink,
    yvex_transform_failure *failure, yvex_error *err);
int yvex_transform_recipe_compile(
    yvex_transform_ir **out, const yvex_transform_header *header,
    yvex_transform_recipe_project_fn project, void *context,
    const yvex_transform_builder_options *options,
    yvex_transform_failure *failure, yvex_error *err);
int yvex_transform_recipe_sink_add(
    yvex_transform_recipe_sink *sink, const yvex_transform_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err);
int yvex_transform_recipe_sink_resolve_source(
    yvex_transform_recipe_sink *sink,
    const yvex_transform_source_requirement *requirement,
    yvex_transform_source_spec *resolved,
    yvex_transform_failure *failure, yvex_error *err);
typedef struct {
    const char *source_name;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer, auxiliary, expert, requirement_index;
    yvex_native_dtype source_dtype;
    yvex_transform_shape shape;
    int packed_fp4, checked_cast;
} yvex_transform_direct_recipe;
typedef struct {
    const char *base_name;
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer, auxiliary, rows, columns;
    unsigned long long block_rows, block_columns, requirement_index;
} yvex_transform_scale_pair_recipe;
int yvex_transform_recipe_add_direct(
    yvex_transform_recipe_sink *sink, const yvex_transform_direct_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err);
int yvex_transform_recipe_add_scale_pair(
    yvex_transform_recipe_sink *sink, const yvex_transform_scale_pair_recipe *recipe,
    yvex_transform_failure *failure, yvex_error *err);
int yvex_transform_recipe_add_terminal(
    yvex_transform_recipe_sink *sink, yvex_tensor_role role,
    yvex_tensor_collection collection, yvex_tensor_scope scope, unsigned long long layer,
    unsigned long long auxiliary, const yvex_transform_source_spec *sources,
    unsigned long long source_count, const yvex_transform_shape *shape,
    yvex_transform_dtype dtype, const yvex_transform_precision_constraint *precision,
    const yvex_transform_node_spec *operation, yvex_transform_failure *failure,
    yvex_error *err);

/* Binding. */
typedef struct yvex_transform_binding yvex_transform_binding;
typedef struct {
    unsigned int physical_class;
    unsigned int encoding_id;
    int approximation_selected;
} yvex_transform_physical_decision;
typedef struct {
    unsigned long long source_count;
    unsigned long long resolved_range_count;
    unsigned long long terminal_count;
    unsigned long long node_count;
    unsigned long long range_lookup_count;
    unsigned long long source_snapshot_identity;
    char required_payload_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    char transform_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    size_t owned_bytes;
    unsigned long long payload_bytes_read;
    int payload_readable_at_bind;
    int complete;
} yvex_transform_binding_summary;
int yvex_transform_binding_create(
    yvex_transform_binding **out,
    const yvex_transform_ir *ir,
    yvex_source_payload_session *session,
    const yvex_transform_allocator *allocator,
    yvex_transform_failure *failure,
    yvex_error *err);
void yvex_transform_binding_release(yvex_transform_binding **binding);
const yvex_transform_binding_summary *yvex_transform_binding_summary_get(
    const yvex_transform_binding *binding);
const yvex_transform_ir *yvex_transform_binding_ir(
    const yvex_transform_binding *binding);
yvex_source_payload_session *yvex_transform_binding_payload_session(
    const yvex_transform_binding *binding);
const yvex_transform_value *yvex_transform_binding_terminal_at(
    const yvex_transform_binding *binding,
    unsigned long long ordinal);
const yvex_transform_node *yvex_transform_binding_terminal_operation(
    const yvex_transform_binding *binding,
    unsigned long long ordinal);
const yvex_source_payload_range *yvex_transform_binding_range_at(
    const yvex_transform_binding *binding,
    unsigned long long source_index);
int yvex_transform_binding_decision_validate(
    const yvex_transform_binding *binding,
    unsigned long long terminal_ordinal,
    const yvex_transform_physical_decision *decision,
    yvex_transform_failure *failure,
    yvex_error *err);
int yvex_transform_binding_readable_validate(
    const yvex_transform_binding *binding,
    yvex_transform_failure *failure,
    yvex_error *err);
/* The caller owns the returned plan; its ranges are exactly the sealed binding source set. */
int yvex_transform_binding_payload_plan_build(
    yvex_source_payload_plan **out,
    const yvex_transform_binding *binding,
    size_t chunk_bytes,
    size_t page_bytes,
    yvex_source_payload_failure *failure,
    yvex_error *err);

/* Runtime-binding publication is a preparation-plane operation.  The CLI
 * supplies resolved paths and adapter identity, then receives only the
 * content-addressed path; runtime admission independently reopens the file. */
typedef struct yvex_compilation_runtime_binding_request {
    const char *source_path;
    const char *models_root;
    const char *source_manifest_path;
    const char *artifact_path;
    const char *directory;
    const char *quant_policy_path;
    const char *quant_preset_name;
    const char *imatrix_path;
    const char *physical_variant_plan_path;
    unsigned long long family_adapter_id;
    unsigned long long family_adapter_version;
} yvex_compilation_runtime_binding_request;
typedef struct yvex_compilation_runtime_binding_result {
    char path[YVEX_PATH_CAP];
    int published;
} yvex_compilation_runtime_binding_result;

unsigned long long yvex_transform_hash_string(const char *text);
int yvex_transform_fail(yvex_transform_failure *failure,
                        yvex_transform_failure_code code,
                        unsigned long long value_id,
                        unsigned long long node_id,
                        unsigned long long source_index,
                        unsigned long long terminal_ordinal,
                        unsigned long long input_index,
                        unsigned long long expected,
                        unsigned long long actual,
                        unsigned int axis,
                        yvex_error *err,
                        const char *where);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_COMPILATION_H_INCLUDED */
